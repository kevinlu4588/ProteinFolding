"""
Z vs S Scaling & Gradient Experiment
=====================================

This script analyzes the relative importance of pair representations (z) vs
single representations (s) by:

1. Scaling them independently before they enter ESMFold's structure module
   and measuring structural metric changes (discrete scaling experiment).
2. Computing autograd gradients of structural metrics w.r.t. z_scale and
   s_scale at the structure module input (gradient analysis).

Hypothesis: If z encodes crucial pairwise distance/geometry information,
scaling z should have a larger effect on output geometry than scaling s.

The structure module receives:
    - s: [batch, N_res, C_s] - per-residue features
    - z: [batch, N_res, N_res, C_z] - pairwise features
"""

import os
import sys
import types
import argparse
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    EsmFoldingTrunk, 
    EsmForProteinFoldingOutput,
    EsmFoldStructureModule,
)
from transformers.models.esm.openfold_utils import Rigid, Rotation


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_PARQUET_PATH = 'data/block_patching_successes.csv'
DEFAULT_OUTPUT_DIR = './z_vs_s_scaling'
DEFAULT_N_CASES = 400


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================
def compute_ca_distances(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute CA-CA distance matrix from positions.
    
    Args:
        positions: [batch, seq_len, 14, 3] or [seq_len, 14, 3]
    
    Returns:
        Distance matrix
    """
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    # CA is atom index 1
    ca_pos = positions[:, :, 1, :]  # [batch, seq_len, 3]
    
    # Compute pairwise distances
    diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
    
    return distances.squeeze(0) if distances.shape[0] == 1 else distances


def compute_radius_of_gyration(positions: torch.Tensor, start: int, end: int) -> float:
    """Compute radius of gyration for a region."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    ca_pos = positions[0, start:end, 1, :]
    com = ca_pos.mean(dim=0)
    diff = ca_pos - com
    rg = torch.sqrt((diff ** 2).sum(-1).mean())
    
    return rg.item()


def compute_strand_separation(positions: torch.Tensor, hp_start: int, hp_end: int) -> float:
    """Compute average distance between paired residues in a hairpin."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    hp_len = hp_end - hp_start
    half_len = hp_len // 2
    
    ca_pos = positions[0, hp_start:hp_end, 1, :]
    
    separations = []
    for i in range(half_len):
        j = hp_len - 1 - i
        if i < j:
            dist = torch.sqrt(((ca_pos[i] - ca_pos[j]) ** 2).sum())
            separations.append(dist.item())
    
    return np.mean(separations) if separations else 0.0


def compute_contact_map(positions: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """Compute binary contact map (CA-CA < threshold)."""
    distances = compute_ca_distances(positions)
    contacts = (distances < threshold).float()
    return contacts


# ============================================================================
# DIFFERENTIABLE GEOMETRY UTILITIES (for gradient computation)
# ============================================================================
def compute_mean_ca_distance_differentiable(positions: torch.Tensor) -> torch.Tensor:
    """Compute mean CA-CA distance (differentiable, returns scalar tensor)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)

    ca_pos = positions[:, :, 1, :]
    diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)

    seq_len = distances.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    return (distances * mask).sum() / n_pairs


def compute_radius_of_gyration_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute radius of gyration for a region (differentiable, returns tensor)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)

    ca_pos = positions[0, start:end, 1, :]
    com = ca_pos.mean(dim=0)
    diff = ca_pos - com
    return torch.sqrt((diff ** 2).sum(-1).mean() + 1e-8)


def compute_local_ca_distance_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute mean CA distance within a region (differentiable, returns tensor)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)

    ca_pos = positions[0, start:end, 1, :]
    diff = ca_pos.unsqueeze(1) - ca_pos.unsqueeze(0)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)

    region_len = distances.shape[0]
    mask = torch.triu(torch.ones(region_len, region_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    return (distances * mask).sum() / (n_pairs + 1e-8)


# ============================================================================
# TRUNK OUTPUT INTERCEPTION (for gradient computation)
# ============================================================================
class TrunkOutputs:
    """Container for trunk outputs before structure module."""
    def __init__(self, s_s, s_z, s_s_proj, s_z_proj, aa, position_ids, mask):
        self.s_s = s_s
        self.s_z = s_z
        self.s_s_proj = s_s_proj
        self.s_z_proj = s_z_proj
        self.aa = aa
        self.position_ids = position_ids
        self.mask = mask


def get_trunk_outputs(model, tokenizer, device, sequence: str, num_recycles: int = 0) -> TrunkOutputs:
    """
    Run ESMFold up to (but not including) the structure module.
    Returns the intermediate representations that would be fed to the structure module.
    """
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids = inputs['input_ids']
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        cfg = model.config.esmfold_config
        aa = input_ids
        B, L = aa.shape

        esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)
        esm_s = model.compute_language_model_representations(esmaa)
        esm_s = esm_s.to(model.esm_s_combine.dtype)

        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0

        esm_s = esm_s.detach()

        esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = model.esm_s_mlp(esm_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

        if model.config.esmfold_config.embed_aa:
            s_s_0 = s_s_0 + model.embedding(aa)

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
                structure = trunk.structure_module(
                    {"single": trunk.trunk2sm_s(s_s), "pair": trunk.trunk2sm_z(s_z)},
                    aa, attention_mask.float(),
                )
                recycle_s = s_s
                recycle_z = s_z
                recycle_bins = trunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375, 21.375, trunk.recycle_bins,
                )

        s_s_proj = trunk.trunk2sm_s(s_s)
        s_z_proj = trunk.trunk2sm_z(s_z)

        return TrunkOutputs(
            s_s=s_s.detach(), s_z=s_z.detach(),
            s_s_proj=s_s_proj.detach(), s_z_proj=s_z_proj.detach(),
            aa=aa.detach(), position_ids=position_ids.detach(),
            mask=attention_mask.float().detach(),
        )


# ============================================================================
# GRADIENT COMPUTATION
# ============================================================================
def _run_structure_module_with_scale(
    structure_module, s_normed, z_normed, scale_param, aa, mask,
    scale_target='z',
):
    """
    Run structure module forward with a differentiable scale on z or s.
    Scale is applied AFTER layer norm to avoid normalization undoing the effect.
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap

    if scale_target == 'z':
        z_input = z_normed * scale_param
        s_input = s_normed
    else:
        z_input = z_normed
        s_input = s_normed * scale_param

    s_initial = s_input
    s_current = structure_module.linear_in(s_input)

    rigids = Rigid.identity(s_current.shape[:-1], s_current.dtype, s_current.device,
                           structure_module.training, fmt="quat")
    outputs = []
    for i in range(structure_module.config.num_blocks):
        s_current = s_current + structure_module.ipa(s_current, z_input, rigids, mask)
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
    return outputs["positions"][-1]


def _compute_metric(positions, metric, hp_start=None, hp_end=None):
    """Compute a differentiable structural metric from positions."""
    if metric == 'mean_ca_dist':
        return compute_mean_ca_distance_differentiable(positions)
    elif metric == 'full_rg':
        return compute_radius_of_gyration_differentiable(positions, 0, positions.shape[1])
    elif metric == 'hairpin_ca_dist' and hp_start is not None and hp_end is not None:
        return compute_local_ca_distance_differentiable(positions, hp_start, hp_end)
    elif metric == 'hairpin_rg' and hp_start is not None and hp_end is not None:
        return compute_radius_of_gyration_differentiable(positions, hp_start, hp_end)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_scale_gradient(
    model,
    trunk_outputs: TrunkOutputs,
    scale_value: float,
    metric: str,
    scale_target: str = 'z',
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the gradient of a structural metric w.r.t. a scale parameter.

    Args:
        scale_target: 'z' or 's'
    """
    device = trunk_outputs.s_s_proj.device
    dtype = trunk_outputs.s_z_proj.dtype

    s = trunk_outputs.s_s_proj.clone()
    z = trunk_outputs.s_z_proj.clone()
    aa = trunk_outputs.aa
    mask = trunk_outputs.mask

    scale_param = torch.tensor(scale_value, dtype=dtype, device=device, requires_grad=True)

    sm = model.trunk.structure_module

    if mask is None:
        mask = s.new_ones(s.shape[:-1])

    s_normed = sm.layer_norm_s(s)
    z_normed = sm.layer_norm_z(z)

    positions = _run_structure_module_with_scale(
        sm, s_normed, z_normed, scale_param, aa, mask, scale_target=scale_target,
    )

    metric_value = _compute_metric(positions, metric, hp_start, hp_end)
    metric_value.backward()

    gradient = scale_param.grad
    gradient_value = gradient.item() if gradient is not None else 0.0

    return {
        'metric': metric,
        f'{scale_target}_scale': scale_value,
        'metric_value': metric_value.item(),
        'gradient': gradient_value,
    }


def compute_both_gradients(
    model,
    trunk_outputs: TrunkOutputs,
    metrics: List[str],
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute gradients for both z and s at the normal operating point (scale=1.0)."""
    z_results = []
    s_results = []

    for metric in metrics:
        try:
            z_results.append(compute_scale_gradient(
                model, trunk_outputs, 1.0, metric, 'z', hp_start, hp_end))
        except Exception as e:
            print(f"Z gradient error: metric={metric}: {e}")
            z_results.append({'metric': metric, 'z_scale': 1.0,
                              'metric_value': float('nan'), 'gradient': float('nan')})

        try:
            s_results.append(compute_scale_gradient(
                model, trunk_outputs, 1.0, metric, 's', hp_start, hp_end))
        except Exception as e:
            print(f"S gradient error: metric={metric}: {e}")
            s_results.append({'metric': metric, 's_scale': 1.0,
                              'metric_value': float('nan'), 'gradient': float('nan')})

    torch.cuda.empty_cache()

    return pd.DataFrame(z_results), pd.DataFrame(s_results)


# ============================================================================
# MODIFIED STRUCTURE MODULE WITH Z/S SCALING
# ============================================================================
def create_scaled_sm_forward(
    s_scale: float = 1.0,
    z_scale: float = 1.0,
):
    """
    Create a structure module forward that scales s and/or z before processing.
    
    Args:
        s_scale: Scale factor for single representations
        z_scale: Scale factor for pair representations
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap
    
    def modified_forward(self_sm, evoformer_output_dict, aatype, mask=None, _offload_inference=False):
        # Get s and z from evoformer output
        s = evoformer_output_dict["single"]
        z = evoformer_output_dict["pair"]
        
        if mask is None:
            mask = s.new_ones(s.shape[:-1])
        
        # Apply layer norms first (as in original)
        s = self_sm.layer_norm_s(s)
        z = self_sm.layer_norm_z(z)
        
        # SCALE S AND Z HERE - this is the key intervention
        s = s * s_scale
        z = z * z_scale
        
        s_initial = s
        s = self_sm.linear_in(s)
        
        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, self_sm.training, fmt="quat")
        outputs = []
        
        for i in range(self_sm.config.num_blocks):
            s = s + self_sm.ipa(s, z, rigids, mask)
            s = self_sm.ipa_dropout(s)
            s = self_sm.layer_norm_ipa(s)
            s = self_sm.transition(s)
            
            rigids = rigids.compose_q_update_vec(self_sm.bb_update(s))
            
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )
            backb_to_global = backb_to_global.scale_translation(self_sm.config.trans_scale_factor)
            
            unnormalized_angles, angles = self_sm.angle_resnet(s, s_initial)
            all_frames_to_global = self_sm.torsion_angles_to_frames(backb_to_global, angles, aatype)
            pred_xyz = self_sm.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aatype)
            
            scaled_rigids = rigids.scale_translation(self_sm.config.trans_scale_factor)
            
            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }
            outputs.append(preds)
            rigids = rigids.stop_rot_gradient()
        
        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        return outputs
    
    return modified_forward


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================
def run_with_scaling(
    model, tokenizer, device, sequence: str,
    s_scale: float = 1.0,
    z_scale: float = 1.0,
) -> EsmForProteinFoldingOutput:
    """
    Run ESMFold with scaled s and/or z inputs to structure module.
    """
    # Store original forward
    original_sm_forward = model.trunk.structure_module.forward
    
    # Patch with scaled version
    model.trunk.structure_module.forward = types.MethodType(
        create_scaled_sm_forward(s_scale=s_scale, z_scale=z_scale),
        model.trunk.structure_module
    )
    
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=0)
    finally:
        # Restore original
        model.trunk.structure_module.forward = original_sm_forward
    
    return outputs


def run_scaling_comparison(
    model, tokenizer, device, sequence: str,
    hp_start: int, hp_end: int,
    scales: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run scaling experiments for both z and s independently.
    
    Returns:
        z_results: DataFrame with z-scaling results
        s_results: DataFrame with s-scaling results
    """
    z_results = []
    s_results = []
    
    # First, get baseline (scale=1.0 for both)
    print("  Getting baseline...")
    baseline_outputs = run_with_scaling(model, tokenizer, device, sequence, s_scale=1.0, z_scale=1.0)
    baseline_pos = baseline_outputs.positions[-1, 0]
    baseline_contacts = compute_contact_map(baseline_pos)
    
    # Scale z only
    print("  Scaling z (pair representations)...")
    for scale in tqdm(scales, desc="Z scaling", leave=False):
        outputs = run_with_scaling(model, tokenizer, device, sequence, s_scale=1.0, z_scale=scale)
        final_pos = outputs.positions[-1, 0]
        
        metrics = compute_structure_metrics(
            final_pos, baseline_pos, baseline_contacts, hp_start, hp_end, outputs
        )
        metrics['scale'] = scale
        metrics['scaled_repr'] = 'z (pair)'
        z_results.append(metrics)
        
        torch.cuda.empty_cache()
    
    # Scale s only
    print("  Scaling s (single representations)...")
    for scale in tqdm(scales, desc="S scaling", leave=False):
        outputs = run_with_scaling(model, tokenizer, device, sequence, s_scale=scale, z_scale=1.0)
        final_pos = outputs.positions[-1, 0]
        
        metrics = compute_structure_metrics(
            final_pos, baseline_pos, baseline_contacts, hp_start, hp_end, outputs
        )
        metrics['scale'] = scale
        metrics['scaled_repr'] = 's (single)'
        s_results.append(metrics)
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(z_results), pd.DataFrame(s_results)


def compute_structure_metrics(
    positions: torch.Tensor,
    baseline_positions: torch.Tensor,
    baseline_contacts: torch.Tensor,
    hp_start: int,
    hp_end: int,
    outputs: EsmForProteinFoldingOutput,
) -> Dict[str, float]:
    """Compute various structural metrics."""
    
    # Basic geometry
    hairpin_rg = compute_radius_of_gyration(positions, hp_start, hp_end)
    full_rg = compute_radius_of_gyration(positions, 0, positions.shape[0])
    strand_sep = compute_strand_separation(positions, hp_start, hp_end)
    
    # CA distances
    ca_dists = compute_ca_distances(positions)
    hp_mean_dist = ca_dists[hp_start:hp_end, hp_start:hp_end].mean().item()
    full_mean_dist = ca_dists.mean().item()
    
    # RMSD from baseline (CA only)
    ca_pos = positions[:, 1, :]  # [N, 3]
    baseline_ca = baseline_positions[:, 1, :]
    
    # Simple RMSD (no alignment)
    rmsd_all = torch.sqrt(((ca_pos - baseline_ca) ** 2).sum(-1).mean()).item()
    rmsd_hairpin = torch.sqrt(((ca_pos[hp_start:hp_end] - baseline_ca[hp_start:hp_end]) ** 2).sum(-1).mean()).item()
    
    # Contact map comparison
    contacts = compute_contact_map(positions)
    contact_precision = ((contacts == 1) & (baseline_contacts == 1)).sum() / (contacts.sum() + 1e-8)
    contact_recall = ((contacts == 1) & (baseline_contacts == 1)).sum() / (baseline_contacts.sum() + 1e-8)
    
    # pLDDT
    mean_plddt = outputs.plddt[0].mean().item()
    hairpin_plddt = outputs.plddt[0, hp_start:hp_end].mean().item()
    
    return {
        'hairpin_rg': hairpin_rg,
        'full_rg': full_rg,
        'strand_sep': strand_sep,
        'hairpin_mean_ca_dist': hp_mean_dist,
        'full_mean_ca_dist': full_mean_dist,
        'rmsd_all': rmsd_all,
        'rmsd_hairpin': rmsd_hairpin,
        'contact_precision': contact_precision.item(),
        'contact_recall': contact_recall.item(),
        'mean_plddt': mean_plddt,
        'hairpin_plddt': hairpin_plddt,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_z_vs_s_comparison(
    z_results: pd.DataFrame,
    s_results: pd.DataFrame,
    output_dir: str,
    case_name: str,
):
    """Create comparison plots for z vs s scaling effects."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    
    z_scales = z_results['scale'].values
    s_scales = s_results['scale'].values
    
    # Row 1: Geometry metrics
    # Plot 1: Hairpin RG
    ax = axes[0, 0]
    ax.plot(z_scales, z_results['hairpin_rg'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['hairpin_rg'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Radius of Gyration (Å)')
    ax.set_title('Hairpin RG')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Full protein RG
    ax = axes[0, 1]
    ax.plot(z_scales, z_results['full_rg'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['full_rg'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Radius of Gyration (Å)')
    ax.set_title('Full Protein RG')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Strand separation
    ax = axes[0, 2]
    ax.plot(z_scales, z_results['strand_sep'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['strand_sep'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Strand Separation (Å)')
    ax.set_title('Hairpin Strand Separation')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Row 2: Distance/RMSD metrics
    # Plot 4: Mean CA distance (hairpin and full)
    ax = axes[1, 0]
    ax.plot(z_scales, z_results['hairpin_mean_ca_dist'], 'b-o', linewidth=2, markersize=6, label='z hairpin')
    ax.plot(z_scales, z_results['full_mean_ca_dist'], 'b--^', linewidth=2, markersize=6, label='z full')
    ax.plot(s_scales, s_results['hairpin_mean_ca_dist'], 'r-s', linewidth=2, markersize=6, label='s hairpin')
    ax.plot(s_scales, s_results['full_mean_ca_dist'], 'r--v', linewidth=2, markersize=6, label='s full')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Mean CA Distance (Å)')
    ax.set_title('Mean CA Distance')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 5: RMSD from baseline (full)
    ax = axes[1, 1]
    ax.plot(z_scales, z_results['rmsd_all'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['rmsd_all'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('RMSD from Baseline (Å)')
    ax.set_title('Full Protein RMSD')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 6: RMSD from baseline (hairpin only)
    ax = axes[1, 2]
    ax.plot(z_scales, z_results['rmsd_hairpin'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['rmsd_hairpin'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('RMSD from Baseline (Å)')
    ax.set_title('Hairpin RMSD')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Row 3: Quality metrics
    # Plot 7: pLDDT
    ax = axes[2, 0]
    ax.plot(z_scales, z_results['mean_plddt'], 'b-o', linewidth=2, markersize=6, label='z full')
    ax.plot(z_scales, z_results['hairpin_plddt'], 'b--^', linewidth=2, markersize=6, label='z hairpin')
    ax.plot(s_scales, s_results['mean_plddt'], 'r-s', linewidth=2, markersize=6, label='s full')
    ax.plot(s_scales, s_results['hairpin_plddt'], 'r--v', linewidth=2, markersize=6, label='s hairpin')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('pLDDT')
    ax.set_title('Confidence Scores')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 8: Contact map metrics
    ax = axes[2, 1]
    ax.plot(z_scales, z_results['contact_precision'], 'b-o', linewidth=2, markersize=6, label='z precision')
    ax.plot(z_scales, z_results['contact_recall'], 'b--^', linewidth=2, markersize=6, label='z recall')
    ax.plot(s_scales, s_results['contact_precision'], 'r-s', linewidth=2, markersize=6, label='s precision')
    ax.plot(s_scales, s_results['contact_recall'], 'r--v', linewidth=2, markersize=6, label='s recall')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Metric')
    ax.set_title('Contact Map Quality')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 9: Normalized sensitivity comparison
    ax = axes[2, 2]
    
    # Compute sensitivity as (max - min) / baseline for key metrics
    def compute_sensitivity(results, metric):
        baseline_val = results.loc[results['scale'] == 1.0, metric].values[0]
        if baseline_val == 0:
            return 0
        range_val = results[metric].max() - results[metric].min()
        return range_val / baseline_val
    
    metrics_to_compare = ['hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist']
    metric_labels = ['HP RG', 'Full RG', 'Strand Sep', 'HP CA Dist']
    
    z_sensitivities = [compute_sensitivity(z_results, m) for m in metrics_to_compare]
    s_sensitivities = [compute_sensitivity(s_results, m) for m in metrics_to_compare]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    ax.bar(x - width/2, z_sensitivities, width, label='z (pair)', color='steelblue')
    ax.bar(x + width/2, s_sensitivities, width, label='s (single)', color='indianred')
    ax.set_ylabel('Sensitivity (range/baseline)')
    ax.set_title('Metric Sensitivity to Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'Z vs S Scaling Comparison: {case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{case_name}_z_vs_s_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_across_cases(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
):
    """Plot summary statistics across all cases."""
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # Select only numeric columns for aggregation
    numeric_cols = ['scale', 'hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist',
                    'full_mean_ca_dist', 'rmsd_all', 'rmsd_hairpin', 'contact_precision', 
                    'contact_recall', 'mean_plddt', 'hairpin_plddt']
    
    z_numeric = all_z_results[numeric_cols]
    s_numeric = all_s_results[numeric_cols]
    
    # Group by scale
    z_grouped = z_numeric.groupby('scale').agg(['mean', 'std'])
    s_grouped = s_numeric.groupby('scale').agg(['mean', 'std'])
    scales = z_grouped.index.values
    
    metrics = [
        ('hairpin_rg', 'Hairpin RG (Å)', axes[0, 0]),
        ('full_rg', 'Full Protein RG (Å)', axes[0, 1]),
        ('strand_sep', 'Strand Separation (Å)', axes[0, 2]),
        ('hairpin_mean_ca_dist', 'Hairpin Mean CA Dist (Å)', axes[0, 3]),
        ('full_mean_ca_dist', 'Full Mean CA Dist (Å)', axes[1, 0]),
        ('rmsd_all', 'RMSD from Baseline (Å)', axes[1, 1]),
        ('mean_plddt', 'pLDDT', axes[1, 2]),
        ('rmsd_hairpin', 'Hairpin RMSD (Å)', axes[1, 3]),
    ]
    
    for metric, ylabel, ax in metrics:
        z_mean = z_grouped[(metric, 'mean')].values
        z_std = z_grouped[(metric, 'std')].values
        s_mean = s_grouped[(metric, 'mean')].values
        s_std = s_grouped[(metric, 'std')].values
        
        ax.errorbar(scales, z_mean, yerr=z_std, fmt='b-o', capsize=3, linewidth=2, 
                   markersize=6, label='z (pair)')
        ax.errorbar(scales, s_mean, yerr=s_std, fmt='r-s', capsize=3, linewidth=2,
                   markersize=6, label='s (single)')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Z vs S Scaling: Summary Across All Cases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'z_vs_s_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_effect_size_comparison(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
):
    """
    Create a summary figure showing the relative effect sizes of z vs s scaling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Compute effect sizes (deviation from baseline at each scale)
    metrics = ['hairpin_rg', 'full_rg', 'strand_sep', 'rmsd_all', 'hairpin_mean_ca_dist', 'full_mean_ca_dist']
    
    # Get baseline values (scale = 1.0)
    z_baseline = all_z_results[all_z_results['scale'] == 1.0][metrics].mean()
    s_baseline = all_s_results[all_s_results['scale'] == 1.0][metrics].mean()
    
    # Left plot: Effect size at extreme scales (0.0 and 2.0)
    ax = axes[0]
    
    extreme_scales = [0.0, 2.0]
    bar_data = {'z': [], 's': []}
    
    for scale in extreme_scales:
        z_at_scale = all_z_results[all_z_results['scale'] == scale]
        s_at_scale = all_s_results[all_s_results['scale'] == scale]
        
        for metric in metrics:
            z_effect = abs(z_at_scale[metric].mean() - z_baseline[metric]) / (z_baseline[metric] + 1e-8)
            s_effect = abs(s_at_scale[metric].mean() - s_baseline[metric]) / (s_baseline[metric] + 1e-8)
            bar_data['z'].append(z_effect)
            bar_data['s'].append(s_effect)
    
    x = np.arange(len(metrics) * len(extreme_scales))
    width = 0.35
    
    ax.bar(x - width/2, bar_data['z'], width, label='z (pair)', color='steelblue')
    ax.bar(x + width/2, bar_data['s'], width, label='s (single)', color='indianred')
    
    labels = [f'{m}\n(s={s})' for s in extreme_scales for m in ['HP_RG', 'Full_RG', 'Strand', 'RMSD', 'HP_CA', 'Full_CA']]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Relative Effect Size (|change|/baseline)')
    ax.set_title('Effect Magnitude at Extreme Scales')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Right plot: Summary bar chart of total sensitivity
    ax = axes[1]
    
    metric_labels = ['Hairpin RG', 'Full RG', 'Strand Sep', 'RMSD', 'HP CA Dist', 'Full CA Dist']
    
    z_total_sensitivity = []
    s_total_sensitivity = []
    
    for metric in metrics:
        # Total range normalized by baseline
        z_range = (all_z_results.groupby('scale')[metric].mean().max() - 
                   all_z_results.groupby('scale')[metric].mean().min())
        z_base = z_baseline[metric]
        z_total_sensitivity.append(z_range / (z_base + 1e-8))
        
        s_range = (all_s_results.groupby('scale')[metric].mean().max() - 
                   all_s_results.groupby('scale')[metric].mean().min())
        s_base = s_baseline[metric]
        s_total_sensitivity.append(s_range / (s_base + 1e-8))
    
    x = np.arange(len(metrics))
    
    ax.bar(x - width/2, z_total_sensitivity, width, label='z (pair)', color='steelblue')
    ax.bar(x + width/2, s_total_sensitivity, width, label='s (single)', color='indianred')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_ylabel('Total Sensitivity (range/baseline)')
    ax.set_title('Overall Metric Sensitivity')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, (z_sens, s_sens) in enumerate(zip(z_total_sensitivity, s_total_sensitivity)):
        ratio = z_sens / (s_sens + 1e-8)
        ax.annotate(f'{ratio:.1f}x', xy=(i, max(z_sens, s_sens) + 0.05), 
                   ha='center', fontsize=8, color='green' if ratio > 1 else 'purple')
    
    plt.suptitle('Z vs S: Which Representation Matters More?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'z_vs_s_effect_sizes.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_running_averages(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
    plot_every_n: int = 10,
):
    """
    Plot running averages of metrics, saving a plot every N cases.
    Shows how the average results stabilize as more cases are added.
    """
    # Get unique case indices in order
    case_indices = sorted(all_z_results['case_idx'].unique())
    n_total_cases = len(case_indices)
    
    if n_total_cases < plot_every_n:
        print(f"Not enough cases ({n_total_cases}) to plot running averages every {plot_every_n} cases")
        return
    
    metrics = ['hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist', 'full_mean_ca_dist', 'rmsd_all']
    metric_labels = ['Hairpin RG (Å)', 'Full RG (Å)', 'Strand Sep (Å)', 'HP CA Dist (Å)', 'Full CA Dist (Å)', 'RMSD (Å)']
    
    # Checkpoints to plot
    checkpoints = list(range(plot_every_n, n_total_cases + 1, plot_every_n))
    if n_total_cases not in checkpoints:
        checkpoints.append(n_total_cases)
    
    # For each checkpoint, compute and plot average results
    for n_cases in checkpoints:
        cases_to_include = case_indices[:n_cases]
        
        z_subset = all_z_results[all_z_results['case_idx'].isin(cases_to_include)]
        s_subset = all_s_results[all_s_results['case_idx'].isin(cases_to_include)]
        
        # Select only numeric columns for aggregation
        numeric_cols = ['scale', 'hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist',
                        'full_mean_ca_dist', 'rmsd_all', 'rmsd_hairpin', 'contact_precision', 
                        'contact_recall', 'mean_plddt', 'hairpin_plddt']
        
        z_numeric = z_subset[numeric_cols]
        s_numeric = s_subset[numeric_cols]
        
        z_grouped = z_numeric.groupby('scale').agg(['mean', 'std'])
        s_grouped = s_numeric.groupby('scale').agg(['mean', 'std'])
        scales = z_grouped.index.values
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            z_mean = z_grouped[(metric, 'mean')].values
            z_std = z_grouped[(metric, 'std')].values
            s_mean = s_grouped[(metric, 'mean')].values
            s_std = s_grouped[(metric, 'std')].values
            
            ax.errorbar(scales, z_mean, yerr=z_std, fmt='b-o', capsize=3, linewidth=2, 
                       markersize=6, label='z (pair)')
            ax.errorbar(scales, s_mean, yerr=s_std, fmt='r-s', capsize=3, linewidth=2,
                       markersize=6, label='s (single)')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Z vs S Scaling: Average over {n_cases} Cases', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'z_vs_s_avg_{n_cases:03d}_cases.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # Also create a summary plot showing how metrics evolve with number of cases
    plot_metric_convergence(all_z_results, all_s_results, output_dir, checkpoints)


def plot_metric_convergence(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
    checkpoints: List[int],
):
    """
    Plot how the Z/S effect ratio converges as more cases are added.
    """
    case_indices = sorted(all_z_results['case_idx'].unique())
    
    metrics = ['hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist', 'full_mean_ca_dist']
    metric_labels = ['Hairpin RG', 'Full RG', 'Strand Sep', 'HP CA Dist', 'Full CA Dist']
    
    # Compute Z/S ratio at each checkpoint
    ratios_per_checkpoint = {m: [] for m in metrics}
    
    for n_cases in checkpoints:
        cases_to_include = case_indices[:n_cases]
        
        z_subset = all_z_results[all_z_results['case_idx'].isin(cases_to_include)]
        s_subset = all_s_results[all_s_results['case_idx'].isin(cases_to_include)]
        
        for metric in metrics:
            z_range = z_subset.groupby('scale')[metric].mean().max() - z_subset.groupby('scale')[metric].mean().min()
            s_range = s_subset.groupby('scale')[metric].mean().max() - s_subset.groupby('scale')[metric].mean().min()
            ratio = z_range / (s_range + 1e-8)
            ratios_per_checkpoint[metric].append(ratio)
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Z/S ratio convergence
    ax = axes[0]
    for metric, label in zip(metrics, metric_labels):
        ax.plot(checkpoints, ratios_per_checkpoint[metric], '-o', linewidth=2, markersize=6, label=label)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal effect')
    ax.set_xlabel('Number of Cases')
    ax.set_ylabel('Z/S Effect Ratio')
    ax.set_title('Effect Ratio Convergence')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Right plot: Final bar chart comparison
    ax = axes[1]
    final_ratios = [ratios_per_checkpoint[m][-1] for m in metrics]
    colors = ['steelblue' if r > 1 else 'indianred' for r in final_ratios]
    bars = ax.bar(metric_labels, final_ratios, color=colors)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Z/S Effect Ratio')
    ax.set_title(f'Final Z/S Ratios ({checkpoints[-1]} Cases)')
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, final_ratios):
        ax.annotate(f'{ratio:.2f}x', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Z vs S: Effect Ratio Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'z_vs_s_convergence.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_z_vs_s_experiment(
    parquet_path: str,
    n_cases: int,
    output_dir: str,
    device: Optional[str] = None,
    scales: Optional[List[float]] = None,
    plot_every_n: int = 10,
):
    """Run the full z vs s scaling experiment with gradient analysis."""

    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if scales is None:
        scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    gradient_metrics = ['mean_ca_dist', 'full_rg', 'hairpin_ca_dist', 'hairpin_rg']

    # Load model
    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print("Model loaded.")

    # Load data
    print(f"\nLoading data from {parquet_path}...")
    if parquet_path.endswith('.parquet'):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path)
    print(f"Loaded {len(df)} rows")

    # Select cases
    cases = df.head(n_cases)

    all_z_results = []
    all_s_results = []
    all_z_grad_results = []
    all_s_grad_results = []

    for idx, row in tqdm(cases.iterrows(), total=len(cases), desc="Analyzing cases"):
        case_name = f"case_{idx}"

        target_seq = row['target_sequence']
        hp_start = int(row['target_patch_start'])
        hp_end = int(row['target_patch_end'])

        print(f"\n{'='*60}")
        print(f"Case {idx}: {row.get('target_name', 'Unknown')}")
        print(f"Sequence length: {len(target_seq)}")
        print(f"Hairpin region: {hp_start}-{hp_end} ({hp_end - hp_start} residues)")
        print(f"{'='*60}")

        # Run scaling comparison
        z_results, s_results = run_scaling_comparison(
            model, tokenizer, device, target_seq,
            hp_start, hp_end,
            scales=scales,
        )

        # Add case info
        z_results['case_idx'] = idx
        z_results['case_name'] = row.get('target_name', f'case_{idx}')
        s_results['case_idx'] = idx
        s_results['case_name'] = row.get('target_name', f'case_{idx}')

        all_z_results.append(z_results)
        all_s_results.append(s_results)

        # Plot individual case
        plot_z_vs_s_comparison(z_results, s_results, output_dir, case_name)

        # Compute gradients
        print("  Computing gradients...")
        trunk_outputs = get_trunk_outputs(model, tokenizer, device, target_seq, num_recycles=0)

        z_grad_results, s_grad_results = compute_both_gradients(
            model, trunk_outputs, gradient_metrics,
            hp_start=hp_start, hp_end=hp_end,
        )

        z_grad_results['case_idx'] = idx
        z_grad_results['case_name'] = row.get('target_name', f'case_{idx}')
        s_grad_results['case_idx'] = idx
        s_grad_results['case_name'] = row.get('target_name', f'case_{idx}')

        all_z_grad_results.append(z_grad_results)
        all_s_grad_results.append(s_grad_results)

        del trunk_outputs
        torch.cuda.empty_cache()

    # Combine results
    combined_z = pd.concat(all_z_results, ignore_index=True)
    combined_s = pd.concat(all_s_results, ignore_index=True)

    # Save scaling results
    combined_z.to_csv(os.path.join(output_dir, 'z_scaling_results.csv'), index=False)
    combined_s.to_csv(os.path.join(output_dir, 's_scaling_results.csv'), index=False)

    # Save gradient results
    combined_z_grad = pd.concat(all_z_grad_results, ignore_index=True)
    combined_s_grad = pd.concat(all_s_grad_results, ignore_index=True)
    combined_z_grad.to_csv(os.path.join(output_dir, 'z_gradients.csv'), index=False)
    combined_s_grad.to_csv(os.path.join(output_dir, 's_gradients.csv'), index=False)

    print(f"\nSaved results to {output_dir}")

    # Summary plots (scaling only)
    if len(all_z_results) > 1:
        plot_summary_across_cases(combined_z, combined_s, output_dir)

    plot_effect_size_comparison(combined_z, combined_s, output_dir)

    # Plot running averages every N cases
    plot_running_averages(combined_z, combined_s, output_dir, plot_every_n=plot_every_n)

    # Print scaling summary
    print("\n" + "="*60)
    print("SUMMARY: Z vs S Scaling Effects")
    print("="*60)

    for metric in ['hairpin_rg', 'full_rg', 'strand_sep', 'rmsd_all', 'hairpin_mean_ca_dist', 'full_mean_ca_dist']:
        z_range = combined_z.groupby('scale')[metric].mean().max() - combined_z.groupby('scale')[metric].mean().min()
        s_range = combined_s.groupby('scale')[metric].mean().max() - combined_s.groupby('scale')[metric].mean().min()

        print(f"\n{metric}:")
        print(f"  Z scaling range: {z_range:.2f}")
        print(f"  S scaling range: {s_range:.2f}")
        print(f"  Z/S ratio: {z_range/(s_range+1e-8):.2f}x")

    # Print gradient summary
    print("\n" + "="*60)
    print("SUMMARY: Gradient Analysis at scale=1.0")
    print("="*60)

    for metric in gradient_metrics:
        z_grad = combined_z_grad[combined_z_grad['metric'] == metric]['gradient'].mean()
        s_grad = combined_s_grad[combined_s_grad['metric'] == metric]['gradient'].mean()
        z_grad_std = combined_z_grad[combined_z_grad['metric'] == metric]['gradient'].std()
        s_grad_std = combined_s_grad[combined_s_grad['metric'] == metric]['gradient'].std()
        ratio = abs(z_grad) / (abs(s_grad) + 1e-10)

        print(f"\n{metric}:")
        print(f"  Z gradient: {z_grad:.6e} +/- {z_grad_std:.6e}")
        print(f"  S gradient: {s_grad:.6e} +/- {s_grad_std:.6e}")
        print(f"  |Z|/|S| ratio: {ratio:.2f}x")

    print(f"\nAll outputs saved to: {output_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compare effects of scaling z (pair) vs s (single) representations"
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
                        help="Scale factors to test (default: 0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0)")
    parser.add_argument("--plot_every_n", type=int, default=10,
                        help="Plot running averages every N cases (default: 10)")

    args = parser.parse_args()

    run_z_vs_s_experiment(
        parquet_path=args.parquet,
        n_cases=args.n_cases,
        output_dir=args.output_dir,
        device=args.device,
        scales=args.scales,
        plot_every_n=args.plot_every_n,
    )


if __name__ == "__main__":
    main()