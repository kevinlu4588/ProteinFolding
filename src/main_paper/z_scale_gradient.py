"""
Z Scaling Gradient Analysis
===========================

This script computes the gradient of mean pairwise CA distance with respect to 
scaling the pairwise representation (z) before the structure module.

Key insight: We don't want gradients flowing through the entire model (memory explosion).
We only care about the structure module's sensitivity to z scaling.

Approach:
1. Run model up to structure module with torch.no_grad()
2. Detach s and z tensors
3. Create learnable z_scale parameter
4. Run structure module with gradients enabled
5. Compute mean pairwise distance and backprop to get gradient

This tells us: "How much does the output geometry change per unit change in z scale?"
"""

import os
import sys
import types
import argparse
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils import Rigid, Rotation


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_PARQUET_PATH = 'data/block_patching_successes.csv'
DEFAULT_OUTPUT_DIR = './z_gradient_analysis'
DEFAULT_N_CASES = 400


# ============================================================================
# GEOMETRY UTILITIES (must be differentiable!)
# ============================================================================
def compute_mean_ca_distance_differentiable(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute mean CA-CA distance (differentiable).
    
    Args:
        positions: [batch, seq_len, 14, 3] or [seq_len, 14, 3]
    
    Returns:
        Scalar tensor (mean pairwise CA distance)
    """
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    # CA is atom index 1
    ca_pos = positions[:, :, 1, :]  # [batch, seq_len, 3]
    
    # Compute pairwise distances
    diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)  # [batch, seq_len, seq_len, 3]
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # [batch, seq_len, seq_len]
    
    # Mean of upper triangle (excluding diagonal)
    seq_len = distances.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    mean_dist = (distances * mask).sum() / n_pairs
    
    return mean_dist


def compute_radius_of_gyration_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute radius of gyration for a region (differentiable)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    ca_pos = positions[0, start:end, 1, :]  # [region_len, 3]
    com = ca_pos.mean(dim=0)  # [3]
    diff = ca_pos - com
    rg = torch.sqrt((diff ** 2).sum(-1).mean() + 1e-8)
    
    return rg


def compute_local_ca_distance_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute mean CA distance within a region (differentiable)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    ca_pos = positions[0, start:end, 1, :]  # [region_len, 3]
    
    diff = ca_pos.unsqueeze(1) - ca_pos.unsqueeze(0)  # [region_len, region_len, 3]
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
    
    region_len = distances.shape[0]
    mask = torch.triu(torch.ones(region_len, region_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    mean_dist = (distances * mask).sum() / (n_pairs + 1e-8)
    
    return mean_dist


# ============================================================================
# MODIFIED TRUNK FORWARD TO INTERCEPT BEFORE STRUCTURE MODULE
# ============================================================================
class TrunkOutputs:
    """Container for trunk outputs before structure module."""
    def __init__(self, s_s, s_z, s_s_proj, s_z_proj, aa, position_ids, mask):
        self.s_s = s_s              # Trunk sequence state
        self.s_z = s_z              # Trunk pairwise state
        self.s_s_proj = s_s_proj    # Projected for structure module (single)
        self.s_z_proj = s_z_proj    # Projected for structure module (pair)
        self.aa = aa                # Amino acid types
        self.position_ids = position_ids
        self.mask = mask


def get_trunk_outputs(model, tokenizer, device, sequence: str, num_recycles: int = 0) -> TrunkOutputs:
    """
    Run ESMFold up to (but not including) the structure module.
    Returns the intermediate representations that would be fed to the structure module.
    """
    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids = inputs['input_ids']
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        
        cfg = model.config.esmfold_config
        aa = input_ids
        B, L = aa.shape
        
        # ESM language model
        esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)
        esm_s = model.compute_language_model_representations(esmaa)
        esm_s = esm_s.to(model.esm_s_combine.dtype)
        
        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0
        
        esm_s = esm_s.detach()
        
        # Preprocessing
        esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = model.esm_s_mlp(esm_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)
        
        if model.config.esmfold_config.embed_aa:
            s_s_0 = s_s_0 + model.embedding(aa)
        
        # Run trunk (evoformer-like blocks)
        trunk = model.trunk
        
        no_recycles = num_recycles if num_recycles is not None else trunk.config.max_recycles
        no_recycles += 1
        
        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)
        
        def trunk_iter(s, z, residx, mask):
            z = z + trunk.pairwise_positional_embedding(residx, mask=mask)
            for block in trunk.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=trunk.chunk_size)
            return s, z
        
        for recycle_idx in range(no_recycles):
            recycle_s = trunk.recycle_s_norm(recycle_s.detach())
            recycle_z = trunk.recycle_z_norm(recycle_z.detach())
            recycle_z = recycle_z + trunk.recycle_disto(recycle_bins.detach())
            
            s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, position_ids, attention_mask)
            
            if recycle_idx < no_recycles - 1:
                # Run structure module to get recycle bins (but don't save)
                structure = trunk.structure_module(
                    {"single": trunk.trunk2sm_s(s_s), "pair": trunk.trunk2sm_z(s_z)},
                    aa,
                    attention_mask.float(),
                )
                recycle_s = s_s
                recycle_z = s_z
                recycle_bins = trunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    trunk.recycle_bins,
                )
        
        # Project to structure module dimensions
        s_s_proj = trunk.trunk2sm_s(s_s)
        s_z_proj = trunk.trunk2sm_z(s_z)
        
        return TrunkOutputs(
            s_s=s_s.detach(),
            s_z=s_z.detach(),
            s_s_proj=s_s_proj.detach(),
            s_z_proj=s_z_proj.detach(),
            aa=aa.detach(),
            position_ids=position_ids.detach(),
            mask=attention_mask.float().detach(),
        )


# ============================================================================
# GRADIENT COMPUTATION
# ============================================================================
def compute_z_scale_gradient(
    model,
    trunk_outputs: TrunkOutputs,
    z_scale_value: float = 1.0,
    metric: str = 'mean_ca_dist',
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
    debug: bool = False,
    verify_numerical: bool = False,
) -> Dict[str, float]:
    """
    Compute the gradient of a structural metric with respect to z_scale.
    
    NOTE: We need to scale z AFTER the layer norm inside the structure module,
    otherwise the layer norm will normalize away the scaling effect!
    
    Args:
        model: ESMFold model
        trunk_outputs: Pre-computed trunk outputs
        z_scale_value: The point at which to compute the gradient
        metric: Which metric to compute gradient for
        hp_start, hp_end: Hairpin region (for local metrics)
        debug: If True, print debug info
        verify_numerical: If True, also compute numerical gradient for comparison
    
    Returns:
        Dictionary with gradient and metric value
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap
    
    device = trunk_outputs.s_s_proj.device
    dtype = trunk_outputs.s_z_proj.dtype
    
    # Get inputs
    s = trunk_outputs.s_s_proj.clone()
    z = trunk_outputs.s_z_proj.clone()
    aa = trunk_outputs.aa
    mask = trunk_outputs.mask
    
    # Create z_scale as a learnable parameter
    z_scale = torch.tensor(z_scale_value, dtype=dtype, device=device, requires_grad=True)
    
    structure_module = model.trunk.structure_module
    
    # We need to manually run the structure module with scaling AFTER layer norm
    # This mimics what the working z_vs_s_scaling_experiment.py does
    
    if mask is None:
        mask = s.new_ones(s.shape[:-1])
    
    # Apply layer norms first
    s_normed = structure_module.layer_norm_s(s)
    z_normed = structure_module.layer_norm_z(z)
    
    # SCALE Z HERE - after layer norm!
    z_scaled = z_normed * z_scale
    
    if debug:
        print(f"  z dtype: {z.dtype}, z_scale dtype: {z_scale.dtype}")
        print(f"  z_normed mean: {z_normed.mean().item():.4f}, std: {z_normed.std().item():.4f}")
        print(f"  z_scaled mean: {z_scaled.mean().item():.4f}, std: {z_scaled.std().item():.4f}")
        print(f"  z_scaled requires_grad: {z_scaled.requires_grad}")
    
    # Continue with structure module forward pass
    s_initial = s_normed
    s_current = structure_module.linear_in(s_normed)
    
    rigids = Rigid.identity(s_current.shape[:-1], s_current.dtype, s_current.device, 
                           structure_module.training, fmt="quat")
    
    outputs = []
    for i in range(structure_module.config.num_blocks):
        s_current = s_current + structure_module.ipa(s_current, z_scaled, rigids, mask)
        s_current = structure_module.ipa_dropout(s_current)
        s_current = structure_module.layer_norm_ipa(s_current)
        s_current = structure_module.transition(s_current)
        
        rigids = rigids.compose_q_update_vec(structure_module.bb_update(s_current))
        
        backb_to_global = Rigid(
            Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )
        backb_to_global = backb_to_global.scale_translation(structure_module.config.trans_scale_factor)
        
        unnormalized_angles, angles = structure_module.angle_resnet(s_current, s_initial)
        all_frames_to_global = structure_module.torsion_angles_to_frames(backb_to_global, angles, aa)
        pred_xyz = structure_module.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aa)
        
        scaled_rigids = rigids.scale_translation(structure_module.config.trans_scale_factor)
        
        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s_current,
        }
        outputs.append(preds)
        rigids = rigids.stop_rot_gradient()
    
    outputs = dict_multimap(torch.stack, outputs)
    
    # Get final positions
    positions = outputs["positions"][-1]  # [B, N, 14, 3]
    
    if debug:
        print(f"  positions requires_grad: {positions.requires_grad}")
        print(f"  positions dtype: {positions.dtype}")
    
    # Compute metric
    if metric == 'mean_ca_dist':
        metric_value = compute_mean_ca_distance_differentiable(positions)
    elif metric == 'full_rg':
        metric_value = compute_radius_of_gyration_differentiable(positions, 0, positions.shape[1])
    elif metric == 'hairpin_ca_dist' and hp_start is not None and hp_end is not None:
        metric_value = compute_local_ca_distance_differentiable(positions, hp_start, hp_end)
    elif metric == 'hairpin_rg' and hp_start is not None and hp_end is not None:
        metric_value = compute_radius_of_gyration_differentiable(positions, hp_start, hp_end)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if debug:
        print(f"  metric_value: {metric_value.item():.4f}, requires_grad: {metric_value.requires_grad}")
    
    # Backprop
    metric_value.backward()
    
    gradient = z_scale.grad
    
    if debug:
        print(f"  z_scale.grad (autodiff): {gradient}")
    
    if gradient is None:
        print(f"  WARNING: gradient is None! The computation graph may be broken.")
        gradient_value = 0.0
    else:
        gradient_value = gradient.item()
    
    # Optionally verify with numerical gradient
    numerical_grad = None
    if verify_numerical:
        eps = 0.1
        
        def run_with_scale(scale_val):
            with torch.no_grad():
                z_s = z_normed * scale_val
                s_i = s_normed
                s_c = structure_module.linear_in(s_normed)
                rigs = Rigid.identity(s_c.shape[:-1], s_c.dtype, s_c.device, False, fmt="quat")
                
                for _ in range(structure_module.config.num_blocks):
                    s_c = s_c + structure_module.ipa(s_c, z_s, rigs, mask)
                    s_c = structure_module.ipa_dropout(s_c)
                    s_c = structure_module.layer_norm_ipa(s_c)
                    s_c = structure_module.transition(s_c)
                    rigs = rigs.compose_q_update_vec(structure_module.bb_update(s_c))
                    rigs = rigs.stop_rot_gradient()
                
                backb = Rigid(Rotation(rot_mats=rigs.get_rots().get_rot_mats(), quats=None), rigs.get_trans())
                backb = backb.scale_translation(structure_module.config.trans_scale_factor)
                _, angles = structure_module.angle_resnet(s_c, s_i)
                frames = structure_module.torsion_angles_to_frames(backb, angles, aa)
                pos = structure_module.frames_and_literature_positions_to_atom14_pos(frames, aa)
                return pos
        
        pos_plus = run_with_scale(z_scale_value + eps)
        pos_minus = run_with_scale(z_scale_value - eps)
        
        if metric == 'mean_ca_dist':
            metric_plus = compute_mean_ca_distance_differentiable(pos_plus).item()
            metric_minus = compute_mean_ca_distance_differentiable(pos_minus).item()
        elif metric == 'full_rg':
            metric_plus = compute_radius_of_gyration_differentiable(pos_plus, 0, pos_plus.shape[1]).item()
            metric_minus = compute_radius_of_gyration_differentiable(pos_minus, 0, pos_minus.shape[1]).item()
        elif metric == 'hairpin_ca_dist':
            metric_plus = compute_local_ca_distance_differentiable(pos_plus, hp_start, hp_end).item()
            metric_minus = compute_local_ca_distance_differentiable(pos_minus, hp_start, hp_end).item()
        elif metric == 'hairpin_rg':
            metric_plus = compute_radius_of_gyration_differentiable(pos_plus, hp_start, hp_end).item()
            metric_minus = compute_radius_of_gyration_differentiable(pos_minus, hp_start, hp_end).item()
        
        numerical_grad = (metric_plus - metric_minus) / (2 * eps)
        
        if debug:
            print(f"  metric at scale {z_scale_value + eps:.2f}: {metric_plus:.4f}")
            print(f"  metric at scale {z_scale_value - eps:.2f}: {metric_minus:.4f}")
            print(f"  numerical gradient: {numerical_grad:.6f}")
            print(f"  autodiff gradient:  {gradient_value:.6e}")
            if abs(gradient_value) > 1e-10:
                print(f"  ratio (numerical/autodiff): {numerical_grad / gradient_value:.2f}")
    
    result = {
        'metric': metric,
        'z_scale': z_scale_value,
        'metric_value': metric_value.item(),
        'gradient': gradient_value,
    }
    
    if numerical_grad is not None:
        result['numerical_gradient'] = numerical_grad
    
    return result


def compute_gradient_across_scales(
    model,
    trunk_outputs: TrunkOutputs,
    scales: List[float],
    metrics: List[str],
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute gradients at multiple z_scale values for multiple metrics.
    """
    results = []
    
    for scale in tqdm(scales, desc="Computing gradients", leave=False):
        for metric in metrics:
            try:
                result = compute_z_scale_gradient(
                    model, trunk_outputs, z_scale_value=scale,
                    metric=metric, hp_start=hp_start, hp_end=hp_end
                )
                results.append(result)
            except Exception as e:
                print(f"Error computing gradient for scale={scale}, metric={metric}: {e}")
                results.append({
                    'metric': metric,
                    'z_scale': scale,
                    'metric_value': float('nan'),
                    'gradient': float('nan'),
                })
    
    return pd.DataFrame(results)


# ============================================================================
# ALSO COMPUTE S GRADIENTS FOR COMPARISON
# ============================================================================
def compute_s_scale_gradient(
    model,
    trunk_outputs: TrunkOutputs,
    s_scale_value: float = 1.0,
    metric: str = 'mean_ca_dist',
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the gradient of a structural metric with respect to s_scale.
    Scale is applied AFTER layer norm to avoid normalization undoing the effect.
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap
    
    device = trunk_outputs.s_s_proj.device
    dtype = trunk_outputs.s_s_proj.dtype
    
    s = trunk_outputs.s_s_proj.clone()
    z = trunk_outputs.s_z_proj.clone()
    aa = trunk_outputs.aa
    mask = trunk_outputs.mask
    
    s_scale = torch.tensor(s_scale_value, dtype=dtype, device=device, requires_grad=True)
    
    structure_module = model.trunk.structure_module
    
    if mask is None:
        mask = s.new_ones(s.shape[:-1])
    
    # Apply layer norms first
    s_normed = structure_module.layer_norm_s(s)
    z_normed = structure_module.layer_norm_z(z)
    
    # SCALE S HERE - after layer norm!
    s_scaled = s_normed * s_scale
    
    # Continue with structure module forward pass
    s_initial = s_scaled  # Note: s_initial should be the scaled version for angle_resnet
    s_current = structure_module.linear_in(s_scaled)
    
    rigids = Rigid.identity(s_current.shape[:-1], s_current.dtype, s_current.device, 
                           structure_module.training, fmt="quat")
    
    outputs = []
    for i in range(structure_module.config.num_blocks):
        s_current = s_current + structure_module.ipa(s_current, z_normed, rigids, mask)
        s_current = structure_module.ipa_dropout(s_current)
        s_current = structure_module.layer_norm_ipa(s_current)
        s_current = structure_module.transition(s_current)
        
        rigids = rigids.compose_q_update_vec(structure_module.bb_update(s_current))
        
        backb_to_global = Rigid(
            Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )
        backb_to_global = backb_to_global.scale_translation(structure_module.config.trans_scale_factor)
        
        unnormalized_angles, angles = structure_module.angle_resnet(s_current, s_initial)
        all_frames_to_global = structure_module.torsion_angles_to_frames(backb_to_global, angles, aa)
        pred_xyz = structure_module.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aa)
        
        scaled_rigids = rigids.scale_translation(structure_module.config.trans_scale_factor)
        
        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s_current,
        }
        outputs.append(preds)
        rigids = rigids.stop_rot_gradient()
    
    outputs = dict_multimap(torch.stack, outputs)
    positions = outputs["positions"][-1]
    
    if metric == 'mean_ca_dist':
        metric_value = compute_mean_ca_distance_differentiable(positions)
    elif metric == 'full_rg':
        metric_value = compute_radius_of_gyration_differentiable(positions, 0, positions.shape[1])
    elif metric == 'hairpin_ca_dist' and hp_start is not None and hp_end is not None:
        metric_value = compute_local_ca_distance_differentiable(positions, hp_start, hp_end)
    elif metric == 'hairpin_rg' and hp_start is not None and hp_end is not None:
        metric_value = compute_radius_of_gyration_differentiable(positions, hp_start, hp_end)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    metric_value.backward()
    gradient = s_scale.grad
    
    if gradient is None:
        gradient_value = 0.0
    else:
        gradient_value = gradient.item()
    
    return {
        'metric': metric,
        's_scale': s_scale_value,
        'metric_value': metric_value.item(),
        'gradient': gradient_value,
    }


def compute_both_gradients(
    model,
    trunk_outputs: TrunkOutputs,
    scales: List[float],
    metrics: List[str],
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
    debug_first: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute gradients for both z and s scaling."""
    
    z_results = []
    s_results = []
    
    first_call = True
    
    for scale in tqdm(scales, desc="Computing gradients"):
        for metric in metrics:
            # Z gradient
            try:
                z_result = compute_z_scale_gradient(
                    model, trunk_outputs, z_scale_value=scale,
                    metric=metric, hp_start=hp_start, hp_end=hp_end,
                    debug=(debug_first and first_call),
                    verify_numerical=(debug_first and first_call)
                )
                z_results.append(z_result)
                first_call = False
            except Exception as e:
                print(f"Z gradient error: scale={scale}, metric={metric}: {e}")
                import traceback
                traceback.print_exc()
                z_results.append({
                    'metric': metric, 'z_scale': scale,
                    'metric_value': float('nan'), 'gradient': float('nan'),
                })
            
            # S gradient
            try:
                s_result = compute_s_scale_gradient(
                    model, trunk_outputs, s_scale_value=scale,
                    metric=metric, hp_start=hp_start, hp_end=hp_end
                )
                s_results.append(s_result)
            except Exception as e:
                print(f"S gradient error: scale={scale}, metric={metric}: {e}")
                s_results.append({
                    'metric': metric, 's_scale': scale,
                    'metric_value': float('nan'), 'gradient': float('nan'),
                })
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(z_results), pd.DataFrame(s_results)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_gradients(
    z_results: pd.DataFrame,
    s_results: pd.DataFrame,
    output_dir: str,
    case_name: str,
):
    """Plot gradient comparison for z vs s scaling."""
    
    metrics = z_results['metric'].unique()
    n_metrics = len(metrics)
    n_scales = len(z_results['z_scale'].unique())
    
    if n_scales == 1:
        # Single scale - just show bar chart of gradients
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        z_grads = [z_results[z_results['metric'] == m]['gradient'].values[0] for m in metrics]
        s_grads = [s_results[s_results['metric'] == m]['gradient'].values[0] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, z_grads, width, label='∂/∂(z_scale)', color='steelblue')
        ax.bar(x + width/2, s_grads, width, label='∂/∂(s_scale)', color='indianred')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Gradient (Å per unit scale)')
        ax.set_title(f'Gradients at scale=1.0: {case_name}')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add ratio annotations
        for i, (zg, sg) in enumerate(zip(z_grads, s_grads)):
            ratio = abs(zg) / (abs(sg) + 1e-10)
            ax.annotate(f'{ratio:.1f}x', xy=(i, max(abs(zg), abs(sg)) * 1.1),
                       ha='center', fontsize=8, color='green' if ratio > 1 else 'purple')
        
        plt.tight_layout()
    else:
        # Multiple scales - show metric value and gradient vs scale
        fig, axes = plt.subplots(2, n_metrics, figsize=(5*n_metrics, 10))
        if n_metrics == 1:
            axes = axes.reshape(2, 1)
        
        for idx, metric in enumerate(metrics):
            z_metric = z_results[z_results['metric'] == metric]
            s_metric = s_results[s_results['metric'] == metric]
            
            # Top row: Metric value vs scale
            ax = axes[0, idx]
            ax.plot(z_metric['z_scale'], z_metric['metric_value'], 'b-o', linewidth=2, markersize=6, label='z scaling')
            ax.plot(s_metric['s_scale'], s_metric['metric_value'], 'r-s', linewidth=2, markersize=6, label='s scaling')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel(f'{metric} (Å)')
            ax.set_title(f'{metric}: Value vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Bottom row: Gradient vs scale
            ax = axes[1, idx]
            ax.plot(z_metric['z_scale'], z_metric['gradient'], 'b-o', linewidth=2, markersize=6, label='∂/∂(z_scale)')
            ax.plot(s_metric['s_scale'], s_metric['gradient'], 'r-s', linewidth=2, markersize=6, label='∂/∂(s_scale)')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Gradient (Å per unit scale)')
            ax.set_title(f'{metric}: Gradient vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Z vs S Scaling Gradients: {case_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{case_name}_gradients.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_gradient_summary(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
):
    """Plot summary of gradients across all cases."""
    
    metrics = all_z_results['metric'].unique()
    n_scales = len(all_z_results['z_scale'].unique())
    
    if n_scales == 1:
        # Single scale - simplified summary
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Gradient at scale=1.0 for each metric (bar chart)
        ax = axes[0]
        
        z_at_1 = all_z_results[all_z_results['z_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        s_at_1 = all_s_results[all_s_results['s_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, z_at_1.loc[metrics, 'mean'].values, width, 
               yerr=z_at_1.loc[metrics, 'std'].values, 
               label='z gradient', color='steelblue', capsize=3)
        ax.bar(x + width/2, s_at_1.loc[metrics, 'mean'].values, width, 
               yerr=s_at_1.loc[metrics, 'std'].values,
               label='s gradient', color='indianred', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Gradient at scale=1.0')
        ax.set_title('Gradient Magnitude Comparison')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Plot 2: Gradient ratio (|z|/|s|) at scale=1.0
        ax = axes[1]
        
        ratios = np.abs(z_at_1.loc[metrics, 'mean'].values) / (np.abs(s_at_1.loc[metrics, 'mean'].values) + 1e-10)
        colors = ['steelblue' if r > 1 else 'indianred' for r in ratios]
        bars = ax.bar(range(len(metrics)), ratios, color=colors)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('|z gradient| / |s gradient|')
        ax.set_title('Gradient Ratio (Z vs S)')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            ax.annotate(f'{ratio:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Z vs S Scaling: Gradient Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
    else:
        # Multiple scales - full summary
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Gradient at scale=1.0 for each metric (bar chart)
        ax = axes[0, 0]
        
        z_at_1 = all_z_results[all_z_results['z_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        s_at_1 = all_s_results[all_s_results['s_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, z_at_1.loc[metrics, 'mean'].values, width, 
               yerr=z_at_1.loc[metrics, 'std'].values, 
               label='z gradient', color='steelblue', capsize=3)
        ax.bar(x + width/2, s_at_1.loc[metrics, 'mean'].values, width, 
               yerr=s_at_1.loc[metrics, 'std'].values,
               label='s gradient', color='indianred', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Gradient at scale=1.0')
        ax.set_title('Gradient Magnitude at Normal Scale')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Plot 2: Gradient ratio (z/s) at scale=1.0
        ax = axes[0, 1]
        
        ratios = np.abs(z_at_1.loc[metrics, 'mean'].values) / (np.abs(s_at_1.loc[metrics, 'mean'].values) + 1e-10)
        colors = ['steelblue' if r > 1 else 'indianred' for r in ratios]
        bars = ax.bar(range(len(metrics)), ratios, color=colors)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('|z gradient| / |s gradient|')
        ax.set_title('Gradient Ratio at scale=1.0')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            ax.annotate(f'{ratio:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Gradient vs scale for mean_ca_dist (averaged across cases)
        ax = axes[1, 0]
        
        if 'mean_ca_dist' in metrics:
            z_mcd = all_z_results[all_z_results['metric'] == 'mean_ca_dist'].groupby('z_scale')['gradient'].agg(['mean', 'std'])
            s_mcd = all_s_results[all_s_results['metric'] == 'mean_ca_dist'].groupby('s_scale')['gradient'].agg(['mean', 'std'])
            
            ax.errorbar(z_mcd.index, z_mcd['mean'], yerr=z_mcd['std'], fmt='b-o', capsize=3, linewidth=2, label='z gradient')
            ax.errorbar(s_mcd.index, s_mcd['mean'], yerr=s_mcd['std'], fmt='r-s', capsize=3, linewidth=2, label='s gradient')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Gradient')
            ax.set_title('Mean CA Distance Gradient vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 4: Metric value vs scale for mean_ca_dist
        ax = axes[1, 1]
        
        if 'mean_ca_dist' in metrics:
            z_mcd_val = all_z_results[all_z_results['metric'] == 'mean_ca_dist'].groupby('z_scale')['metric_value'].agg(['mean', 'std'])
            s_mcd_val = all_s_results[all_s_results['metric'] == 'mean_ca_dist'].groupby('s_scale')['metric_value'].agg(['mean', 'std'])
            
            ax.errorbar(z_mcd_val.index, z_mcd_val['mean'], yerr=z_mcd_val['std'], fmt='b-o', capsize=3, linewidth=2, label='z scaling')
            ax.errorbar(s_mcd_val.index, s_mcd_val['mean'], yerr=s_mcd_val['std'], fmt='r-s', capsize=3, linewidth=2, label='s scaling')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Mean CA Distance (Å)')
            ax.set_title('Mean CA Distance vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Z vs S Scaling: Gradient Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'gradient_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_gradient_analysis(
    parquet_path: str,
    n_cases: int,
    output_dir: str,
    device: Optional[str] = None,
    scales: Optional[List[float]] = None,
):
    """Run the gradient analysis experiment."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if scales is None:
        scales = [1.0]  # Only need gradient at the normal operating point
    
    # Metrics to analyze
    metrics = ['mean_ca_dist', 'full_rg', 'hairpin_ca_dist', 'hairpin_rg']
    
    # Load model
    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
    model.eval()
    # Don't use model.requires_grad_(False) - we need gradients through structure module
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print("Model loaded.")
    
    # Load data
    print(f"\nLoading data from {parquet_path}...")
    if parquet_path.endswith('.parquet'):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path)
    print(f"Loaded {len(df)} rows")
    
    cases = df.head(n_cases)
    
    all_z_results = []
    all_s_results = []
    
    for idx, row in tqdm(cases.iterrows(), total=len(cases), desc="Analyzing cases"):
        case_name = f"case_{idx}"
        
        target_seq = row['target_sequence']
        hp_start = int(row['target_patch_start'])
        hp_end = int(row['target_patch_end'])
        
        print(f"\n{'='*60}")
        print(f"Case {idx}: {row.get('target_name', 'Unknown')}")
        print(f"Sequence length: {len(target_seq)}")
        print(f"Hairpin region: {hp_start}-{hp_end}")
        print(f"{'='*60}")
        
        # Get trunk outputs (run model up to structure module)
        print("  Computing trunk outputs...")
        trunk_outputs = get_trunk_outputs(model, tokenizer, device, target_seq, num_recycles=0)
        
        # Compute gradients
        print("  Computing gradients...")
        z_results, s_results = compute_both_gradients(
            model, trunk_outputs, scales, metrics,
            hp_start=hp_start, hp_end=hp_end
        )
        
        # Add case info
        z_results['case_idx'] = idx
        z_results['case_name'] = row.get('target_name', f'case_{idx}')
        s_results['case_idx'] = idx
        s_results['case_name'] = row.get('target_name', f'case_{idx}')
        
        all_z_results.append(z_results)
        all_s_results.append(s_results)
        
        # Plot individual case
        plot_gradients(z_results, s_results, output_dir, case_name)
        
        torch.cuda.empty_cache()
    
    # Combine results
    combined_z = pd.concat(all_z_results, ignore_index=True)
    combined_s = pd.concat(all_s_results, ignore_index=True)
    
    # Save results
    combined_z.to_csv(os.path.join(output_dir, 'z_gradients.csv'), index=False)
    combined_s.to_csv(os.path.join(output_dir, 's_gradients.csv'), index=False)
    print(f"\nSaved results to {output_dir}")
    
    # Summary plot
    plot_gradient_summary(combined_z, combined_s, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Gradient Analysis at scale=1.0")
    print("="*60)
    
    z_at_1 = combined_z[combined_z['z_scale'] == 1.0]
    s_at_1 = combined_s[combined_s['s_scale'] == 1.0]
    
    for metric in metrics:
        z_grad = z_at_1[z_at_1['metric'] == metric]['gradient'].mean()
        s_grad = s_at_1[s_at_1['metric'] == metric]['gradient'].mean()
        z_grad_std = z_at_1[z_at_1['metric'] == metric]['gradient'].std()
        s_grad_std = s_at_1[s_at_1['metric'] == metric]['gradient'].std()
        ratio = abs(z_grad) / (abs(s_grad) + 1e-10)
        
        print(f"\n{metric}:")
        print(f"  Z gradient: {z_grad:.6e} ± {z_grad_std:.6e}")
        print(f"  S gradient: {s_grad:.6e} ± {s_grad_std:.6e}")
        print(f"  |Z|/|S| ratio: {ratio:.2f}x")
    
    print(f"\nAll outputs saved to: {output_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute gradients of structural metrics w.r.t. z and s scaling"
    )
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET_PATH,
                        help=f"Path to data file (default: {DEFAULT_PARQUET_PATH})")
    parser.add_argument("--n_cases", type=int, default=DEFAULT_N_CASES,
                        help=f"Number of cases to analyze (default: {DEFAULT_N_CASES})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--scales", type=float, nargs='+', default=None,
                        help="Scale factors to compute gradients at (default: 1.0 only)")
    
    args = parser.parse_args()
    
    run_gradient_analysis(
        parquet_path=args.parquet,
        n_cases=args.n_cases,
        output_dir=args.output_dir,
        device=args.device,
        scales=args.scales,
    )


if __name__ == "__main__":
    main()