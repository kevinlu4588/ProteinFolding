#!/usr/bin/env python
"""
Representation Flow Analysis with Baseline Freezing and Zero Ablation (Refactored)
==================================================================================

Analyzes how representations flow through ESMFold's trunk blocks, with support for:
1. Collecting baseline representations, pair2seq biases, and seq2pair updates
2. Patching donor representations into target sequences
3. Intervening on information flow (freeze to baseline or zero ablation)

Uses hooks and context managers for clean representation collection and intervention.

Usage:
    python representation_flow_analysis.py \
        --dataset next_experiment.parquet \
        --output results/ \
        --intervention_windows early_10 late_10
"""

import argparse
import os
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformers import AutoTokenizer, EsmForProteinFolding

warnings.filterwarnings("ignore", message=".*mmCIF.*")


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_BLOCKS = 48

INTERVENTION_CONDITIONS = [
    (None, None),  # No intervention (baseline)
    ('freeze', 'seq2pair'),
    ('freeze', 'pair2seq'),
    ('zero', 'seq2pair'),
    ('zero', 'pair2seq'),
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CollectedRepresentations:
    """Container for collected representations from ESMFold trunk blocks."""
    s_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    z_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    pair2seq_bias_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    seq2pair_update_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        self.s_blocks.clear()
        self.z_blocks.clear()
        self.pair2seq_bias_blocks.clear()
        self.seq2pair_update_blocks.clear()


# ============================================================================
# HOOK MANAGERS
# ============================================================================

class TrunkCollectionHooks:
    """
    Collect trunk block outputs and intermediate values via hooks.
    
    Collects:
    - s (sequence representation) after each block
    - z (pairwise representation) after each block
    - pair2seq bias (output of pair_to_sequence projection)
    - seq2pair update (output of sequence_to_pair projection)
    
    Usage:
        collector = CollectedRepresentations()
        with TrunkCollectionHooks(model.trunk, collector) as hooks:
            outputs = model(**inputs)
    """
    
    def __init__(
        self,
        trunk: nn.Module,
        collector: CollectedRepresentations,
        blocks: Any = 'all',
        collect_s: bool = True,
        collect_z: bool = True,
        collect_pair2seq: bool = True,
        collect_seq2pair: bool = True,
    ):
        self.trunk = trunk
        self.collector = collector
        self.blocks = blocks
        self.collect_s = collect_s
        self.collect_z = collect_z
        self.collect_pair2seq = collect_pair2seq
        self.collect_seq2pair = collect_seq2pair
        self.handles: List = []
    
    def register(self):
        """Register hooks on trunk blocks and their submodules."""
        blocks = self.blocks
        if blocks == 'all':
            blocks = range(len(self.trunk.blocks))
        
        for idx in blocks:
            block = self.trunk.blocks[idx]
            
            # Hook for collecting s and z after the full block
            if self.collect_s or self.collect_z:
                def make_block_hook(block_idx, do_s, do_z):
                    def hook(module, inputs, outputs):
                        s, z = outputs
                        if do_s:
                            self.collector.s_blocks[block_idx] = s.detach().cpu()
                        if do_z:
                            self.collector.z_blocks[block_idx] = z.detach().cpu()
                    return hook
                
                handle = block.register_forward_hook(
                    make_block_hook(idx, self.collect_s, self.collect_z)
                )
                self.handles.append(handle)
            
            # Hook for collecting pair2seq bias (output of pair_to_sequence)
            if self.collect_pair2seq:
                def make_pair2seq_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.pair2seq_bias_blocks[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.pair_to_sequence.register_forward_hook(
                    make_pair2seq_hook(idx)
                )
                self.handles.append(handle)
            
            # Hook for collecting seq2pair update (output of sequence_to_pair)
            if self.collect_seq2pair:
                def make_seq2pair_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.seq2pair_update_blocks[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.sequence_to_pair.register_forward_hook(
                    make_seq2pair_hook(idx)
                )
                self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


class TrunkPatchingHooks:
    """
    Apply representation patches after specific blocks via hooks.
    
    Patches donor representations into target positions after the specified block.
    
    Usage:
        with TrunkPatchingHooks(model.trunk, patch_block=0, donor_s=..., donor_z=...) as hooks:
            outputs = model(**inputs)
    """
    
    def __init__(
        self,
        trunk: nn.Module,
        patch_block: int,
        patch_mode: str,  # 'sequence', 'pairwise', or 'both'
        donor_s: Optional[torch.Tensor] = None,  # [1, region_len, dim]
        donor_z: Optional[torch.Tensor] = None,  # [1, L_donor, L_donor, dim]
        target_start: int = 0,
        target_end: int = 0,
        donor_hairpin_start: int = 0,
        pairwise_mask: Optional[torch.Tensor] = None,
    ):
        self.trunk = trunk
        self.patch_block = patch_block
        self.patch_mode = patch_mode
        self.donor_s = donor_s
        self.donor_z = donor_z
        self.target_start = target_start
        self.target_end = target_end
        self.donor_hairpin_start = donor_hairpin_start
        self.pairwise_mask = pairwise_mask
        self.handles: List = []
    
    def register(self):
        """Register post-hook on the patch block."""
        block = self.trunk.blocks[self.patch_block]
        
        def patch_hook(module, inputs, outputs):
            s, z = outputs
            
            # Patch sequence representation
            if self.patch_mode in ('both', 'sequence') and self.donor_s is not None:
                donor_repr = self.donor_s.to(s.device, dtype=s.dtype)
                s = s.clone()
                s[:, self.target_start:self.target_end, :] = donor_repr
            
            # Patch pairwise representation
            if self.patch_mode in ('both', 'pairwise') and self.donor_z is not None:
                donor_z = self.donor_z.to(z.device, dtype=z.dtype)
                z = z.clone()
                
                if self.pairwise_mask is not None:
                    mask_tensor = self.pairwise_mask.to(z.device)
                    indices = torch.where(mask_tensor)
                    
                    for idx in range(len(indices[0])):
                        t_i, t_j = indices[0][idx].item(), indices[1][idx].item()
                        d_i = t_i - self.target_start + self.donor_hairpin_start
                        d_j = t_j - self.target_start + self.donor_hairpin_start
                        if 0 <= d_i < donor_z.shape[1] and 0 <= d_j < donor_z.shape[2]:
                            z[:, t_i, t_j, :] = donor_z[:, d_i, d_j, :]
            
            return (s, z)
        
        handle = block.register_forward_hook(patch_hook)
        self.handles.append(handle)
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


class TrunkInterventionHooks:
    """
    Apply interventions (freeze or zero) on seq2pair or pair2seq pathways.
    
    For 'freeze': Replace the output with pre-computed baseline values
    For 'zero': Replace the output with zeros
    
    Usage:
        with TrunkInterventionHooks(
            model.trunk,
            intervention_type='freeze',
            intervention_pathway='seq2pair',
            start_block=0, end_block=9,
            baseline_values=baseline_seq2pair_list
        ):
            outputs = model(**inputs)
    """
    
    def __init__(
        self,
        trunk: nn.Module,
        intervention_type: str,  # 'freeze' or 'zero'
        intervention_pathway: str,  # 'seq2pair' or 'pair2seq'
        start_block: int,
        end_block: int,
        baseline_values: Optional[List[torch.Tensor]] = None,  # For freeze
    ):
        self.trunk = trunk
        self.intervention_type = intervention_type
        self.intervention_pathway = intervention_pathway
        self.start_block = start_block
        self.end_block = end_block
        self.baseline_values = baseline_values
        self.handles: List = []
    
    def register(self):
        """Register hooks to intervene on the specified pathway."""
        for block_idx in range(self.start_block, self.end_block + 1):
            if block_idx >= len(self.trunk.blocks):
                continue
            
            block = self.trunk.blocks[block_idx]
            
            if self.intervention_pathway == 'pair2seq':
                # Intervene on pair_to_sequence output
                def make_pair2seq_intervention(idx, int_type, baseline_vals):
                    def hook(module, inputs, outputs):
                        if int_type == 'freeze':
                            return baseline_vals[idx].to(outputs.device, dtype=outputs.dtype)
                        elif int_type == 'zero':
                            return torch.zeros_like(outputs)
                    return hook
                
                handle = block.pair_to_sequence.register_forward_hook(
                    make_pair2seq_intervention(block_idx, self.intervention_type, self.baseline_values)
                )
                self.handles.append(handle)
            
            elif self.intervention_pathway == 'seq2pair':
                # Intervene on sequence_to_pair output
                def make_seq2pair_intervention(idx, int_type, baseline_vals):
                    def hook(module, inputs, outputs):
                        if int_type == 'freeze':
                            return baseline_vals[idx].to(outputs.device, dtype=outputs.dtype)
                        elif int_type == 'zero':
                            return torch.zeros_like(outputs)
                    return hook
                
                handle = block.sequence_to_pair.register_forward_hook(
                    make_seq2pair_intervention(block_idx, self.intervention_type, self.baseline_values)
                )
                self.handles.append(handle)
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

@contextmanager
def collect_all_representations(
    model: EsmForProteinFolding,
    collect_s: bool = True,
    collect_z: bool = True,
    collect_pair2seq: bool = True,
    collect_seq2pair: bool = True,
):
    """
    Context manager for collecting all representations during forward pass.
    
    Yields:
        CollectedRepresentations with s_blocks, z_blocks, pair2seq_bias_blocks, seq2pair_update_blocks
    """
    collector = CollectedRepresentations()
    hooks = TrunkCollectionHooks(
        model.trunk, collector,
        collect_s=collect_s,
        collect_z=collect_z,
        collect_pair2seq=collect_pair2seq,
        collect_seq2pair=collect_seq2pair,
    )
    hooks.register()
    try:
        yield collector
    finally:
        hooks.remove()


@contextmanager
def apply_patch(
    model: EsmForProteinFolding,
    patch_block: int,
    patch_mode: str,
    donor_s: Optional[torch.Tensor] = None,
    donor_z: Optional[torch.Tensor] = None,
    target_start: int = 0,
    target_end: int = 0,
    donor_hairpin_start: int = 0,
    pairwise_mask: Optional[torch.Tensor] = None,
):
    """Context manager for applying representation patches."""
    hooks = TrunkPatchingHooks(
        model.trunk,
        patch_block=patch_block,
        patch_mode=patch_mode,
        donor_s=donor_s,
        donor_z=donor_z,
        target_start=target_start,
        target_end=target_end,
        donor_hairpin_start=donor_hairpin_start,
        pairwise_mask=pairwise_mask,
    )
    hooks.register()
    try:
        yield
    finally:
        hooks.remove()


@contextmanager
def apply_intervention(
    model: EsmForProteinFolding,
    intervention_type: str,
    intervention_pathway: str,
    start_block: int,
    end_block: int,
    baseline_values: Optional[List[torch.Tensor]] = None,
):
    """Context manager for applying freeze or zero interventions."""
    hooks = TrunkInterventionHooks(
        model.trunk,
        intervention_type=intervention_type,
        intervention_pathway=intervention_pathway,
        start_block=start_block,
        end_block=end_block,
        baseline_values=baseline_values,
    )
    hooks.register()
    try:
        yield
    finally:
        hooks.remove()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_pairwise_mask(
    donor_hairpin_start: int, donor_hairpin_end: int, donor_len: int,
    target_start: int, target_end: int, target_len: int, mode: str = 'touch',
) -> torch.Tensor:
    """Create pairwise mask for target sequence."""
    mask = torch.zeros(target_len, target_len, dtype=torch.bool)
    if mode == 'intra':
        mask[target_start:target_end, target_start:target_end] = True
    elif mode == 'touch':
        mask[target_start:target_end, :] = True
        mask[:, target_start:target_end] = True
    return mask


def compute_representation_metrics(
    baseline_s, baseline_z, donor_s, donor_z, patched_s, patched_z,
    target_start, target_end, donor_hairpin_start, touch_mask, intra_mask,
) -> Dict[str, float]:
    """Compute interpolation and similarity metrics."""
    results = {}
    
    def cosine_sim(a, b):
        a_flat, b_flat = a.flatten().float(), b.flatten().float()
        if a_flat.numel() == 0:
            return float('nan')
        return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    
    def interpolation_coefficient(baseline, donor, patched):
        b, d, p = baseline.flatten().float(), donor.flatten().float(), patched.flatten().float()
        direction = d - b
        direction_norm_sq = torch.sum(direction ** 2)
        if direction_norm_sq == 0:
            return float('nan')
        movement = p - b
        return (torch.sum(movement * direction) / direction_norm_sq).item()
    
    def extract_masked_pairwise(baseline_z, donor_z, patched_z, mask, target_start, donor_hairpin_start):
        indices = torch.where(mask)
        t_i_indices, t_j_indices = indices[0], indices[1]
        if len(t_i_indices) == 0:
            return None, None, None
        d_i_indices = t_i_indices - target_start + donor_hairpin_start
        d_j_indices = t_j_indices - target_start + donor_hairpin_start
        valid_mask = (
            (d_i_indices >= 0) & (d_i_indices < donor_z.shape[1]) &
            (d_j_indices >= 0) & (d_j_indices < donor_z.shape[2])
        )
        if not valid_mask.any():
            return None, None, None
        t_i_valid, t_j_valid = t_i_indices[valid_mask], t_j_indices[valid_mask]
        d_i_valid, d_j_valid = d_i_indices[valid_mask], d_j_indices[valid_mask]
        return (baseline_z[0, t_i_valid, t_j_valid, :],
                donor_z[0, d_i_valid, d_j_valid, :],
                patched_z[0, t_i_valid, t_j_valid, :])
    
    def compute_pairwise_metrics(baseline_masked, donor_masked, patched_masked, prefix):
        metrics = {}
        if baseline_masked is None:
            metrics[f'{prefix}_cos_baseline_donor'] = None
            metrics[f'{prefix}_cos_donor_patched'] = None
            metrics[f'{prefix}_interp_alpha'] = None
            return metrics
        metrics[f'{prefix}_cos_baseline_donor'] = cosine_sim(baseline_masked, donor_masked)
        metrics[f'{prefix}_cos_donor_patched'] = cosine_sim(donor_masked, patched_masked)
        metrics[f'{prefix}_interp_alpha'] = interpolation_coefficient(baseline_masked, donor_masked, patched_masked)
        return metrics
    
    # Sequence metrics
    results['seq_cos_baseline_donor'] = cosine_sim(baseline_s, donor_s)
    results['seq_cos_donor_patched'] = cosine_sim(donor_s, patched_s)
    results['seq_interp_alpha'] = interpolation_coefficient(baseline_s, donor_s, patched_s)
    
    # Pairwise metrics with different masks
    for mask, prefix in [(touch_mask, 'pw_touch'), (intra_mask, 'pw_intra'), (touch_mask & ~intra_mask, 'pw_cross')]:
        b, d, p = extract_masked_pairwise(baseline_z, donor_z, patched_z, mask, target_start, donor_hairpin_start)
        results.update(compute_pairwise_metrics(b, d, p, prefix))
    
    return results


def get_intervention_windows(patch_block: int, num_blocks: int = NUM_BLOCKS) -> List[Tuple[str, int, int]]:
    """Get intervention windows: early_10, late_10, full_remaining."""
    windows = [
        ('early_10', patch_block, min(patch_block + 9, num_blocks - 1)),
        ('late_10', max(patch_block, 38), num_blocks - 1),
        ('full_remaining', patch_block, num_blocks - 1),
    ]
    return [(name, start, end) for name, start, end in windows if start <= end]


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(
    df: pd.DataFrame,
    model: EsmForProteinFolding,
    tokenizer,
    device: str,
    output_dir: str,
    save_every: int = 10,
) -> pd.DataFrame:
    """Run the representation flow analysis experiment."""
    all_results = []
    
    unique_configs = df[['patch_mode', 'patch_mask_mode', 'block_idx']].drop_duplicates()
    print(f"\nUnique patch configurations: {len(unique_configs)}")
    
    cases_processed = 0
    
    for _, config_row in unique_configs.iterrows():
        patch_mode = config_row['patch_mode']
        patch_mask_mode = config_row['patch_mask_mode']
        patch_block = int(config_row['block_idx'])
        
        config_df = df[
            (df['patch_mode'] == patch_mode) &
            (df['patch_mask_mode'] == patch_mask_mode) &
            (df['block_idx'] == patch_block)
        ]
        
        print(f"\n{'='*60}")
        print(f"Config: {patch_mode} @ block {patch_block}, {patch_mask_mode} mask")
        print(f"Cases: {len(config_df)}")
        print(f"{'='*60}")
        
        for idx, row in tqdm(config_df.iterrows(), total=len(config_df), desc="Cases"):
            target_seq = row['target_sequence']
            donor_seq = row['donor_sequence']
            target_start = int(row['target_patch_start'])
            target_end = int(row['target_patch_end'])
            donor_hairpin_start = int(row['donor_hairpin_start'])
            donor_hairpin_end = int(row['donor_hairpin_end'])
            
            touch_mask = create_pairwise_mask(
                donor_hairpin_start, donor_hairpin_end, len(donor_seq),
                target_start, target_end, len(target_seq), mode='touch',
            )
            intra_mask = create_pairwise_mask(
                donor_hairpin_start, donor_hairpin_end, len(donor_seq),
                target_start, target_end, len(target_seq), mode='intra',
            )
            patch_mask = touch_mask if patch_mask_mode == 'touch' else intra_mask
            
            # ================================================================
            # Step 1: Collect baseline representations
            # ================================================================
            with torch.no_grad():
                target_inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                
                with collect_all_representations(model) as baseline_collector:
                    _ = model(**target_inputs, num_recycles=0)
            
            baseline_s_blocks = {k: v[:, target_start:target_end, :] for k, v in baseline_collector.s_blocks.items()}
            baseline_z_blocks = dict(baseline_collector.z_blocks)
            baseline_pair2seq_list = [baseline_collector.pair2seq_bias_blocks.get(i) for i in range(NUM_BLOCKS)]
            baseline_seq2pair_list = [baseline_collector.seq2pair_update_blocks.get(i) for i in range(NUM_BLOCKS)]
            
            # ================================================================
            # Step 2: Collect donor representations
            # ================================================================
            with torch.no_grad():
                donor_inputs = tokenizer(donor_seq, return_tensors='pt', add_special_tokens=False).to(device)
                
                with collect_all_representations(model, collect_pair2seq=False, collect_seq2pair=False) as donor_collector:
                    _ = model(**donor_inputs, num_recycles=0)
            
            donor_s_blocks = {k: v[:, donor_hairpin_start:donor_hairpin_end, :] for k, v in donor_collector.s_blocks.items()}
            donor_z_blocks = dict(donor_collector.z_blocks)
            
            # ================================================================
            # Step 3: Run patching with various intervention conditions
            # ================================================================
            for window_name, intervention_start, intervention_end in get_intervention_windows(patch_block):
                for intervention_type, intervention_pathway in INTERVENTION_CONDITIONS:
                    # Create condition name
                    if intervention_type is None:
                        condition_name = 'none'
                    else:
                        condition_name = f'{intervention_type}_{intervention_pathway}'
                    
                    with torch.no_grad():
                        # Get donor representations for this patch block
                        donor_s_for_patch = donor_s_blocks[patch_block].to(device)
                        donor_z_for_patch = donor_z_blocks[patch_block].to(device)
                        
                        # Set up context managers
                        patch_ctx = apply_patch(
                            model,
                            patch_block=patch_block,
                            patch_mode=patch_mode,
                            donor_s=donor_s_for_patch,
                            donor_z=donor_z_for_patch,
                            target_start=target_start,
                            target_end=target_end,
                            donor_hairpin_start=donor_hairpin_start,
                            pairwise_mask=patch_mask,
                        )
                        
                        # Build nested context managers
                        if intervention_type is not None:
                            baseline_vals = (baseline_seq2pair_list if intervention_pathway == 'seq2pair' 
                                           else baseline_pair2seq_list)
                            
                            intervention_ctx = apply_intervention(
                                model,
                                intervention_type=intervention_type,
                                intervention_pathway=intervention_pathway,
                                start_block=intervention_start,
                                end_block=intervention_end,
                                baseline_values=baseline_vals,
                            )
                            
                            with patch_ctx, intervention_ctx, collect_all_representations(
                                model, collect_pair2seq=False, collect_seq2pair=False
                            ) as patched_collector:
                                _ = model(**target_inputs, num_recycles=0)
                        else:
                            with patch_ctx, collect_all_representations(
                                model, collect_pair2seq=False, collect_seq2pair=False
                            ) as patched_collector:
                                _ = model(**target_inputs, num_recycles=0)
                    
                    patched_s_blocks = {k: v[:, target_start:target_end, :] for k, v in patched_collector.s_blocks.items()}
                    patched_z_blocks = dict(patched_collector.z_blocks)
                    
                    # Compute metrics for each observation block
                    for obs_block in range(NUM_BLOCKS):
                        metrics = compute_representation_metrics(
                            baseline_s=baseline_s_blocks[obs_block],
                            baseline_z=baseline_z_blocks[obs_block],
                            donor_s=donor_s_blocks[obs_block],
                            donor_z=donor_z_blocks[obs_block],
                            patched_s=patched_s_blocks[obs_block],
                            patched_z=patched_z_blocks[obs_block],
                            target_start=target_start,
                            target_end=target_end,
                            donor_hairpin_start=donor_hairpin_start,
                            touch_mask=touch_mask,
                            intra_mask=intra_mask,
                        )
                        
                        result = {
                            'patch_mode': patch_mode,
                            'patch_mask_mode': patch_mask_mode,
                            'patch_block': patch_block,
                            'condition': condition_name,
                            'intervention_type': intervention_type if intervention_type else 'none',
                            'intervention_pathway': intervention_pathway if intervention_pathway else 'none',
                            'window_name': window_name,
                            'intervention_start': intervention_start,
                            'intervention_end': intervention_end,
                            'observation_block': obs_block,
                            **metrics,
                        }
                        all_results.append(result)
            
            cases_processed += 1
            if cases_processed % save_every == 0:
                pd.DataFrame(all_results).to_parquet(
                    os.path.join(output_dir, 'intervention_experiment_results_checkpoint.parquet'),
                    index=False
                )
                print(f"\n[Checkpoint] Saved {len(all_results)} results")
            
            torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(os.path.join(output_dir, 'intervention_experiment_results.parquet'), index=False)
    print(f"\nSaved {len(results_df)} rows")
    
    return results_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_results(results_df: pd.DataFrame, output_dir: str):
    """Generate all analysis plots."""
    print("\nGenerating plots...")
    
    conditions_grid = [
        [(0, 'sequence', 'touch'), (0, 'pairwise', 'touch'), (0, 'pairwise', 'intra')],
        [(27, 'sequence', 'touch'), (27, 'pairwise', 'touch'), (27, 'pairwise', 'intra')],
    ]
    
    colors = {
        'none': 'black',
        'freeze_seq2pair': 'blue',
        'freeze_pair2seq': 'cyan',
        'zero_seq2pair': 'red',
        'zero_pair2seq': 'orange',
    }
    linestyles = {
        'none': '-',
        'freeze_seq2pair': '--',
        'freeze_pair2seq': '--',
        'zero_seq2pair': ':',
        'zero_pair2seq': ':',
    }
    
    window_names = [w for w in results_df['window_name'].unique() if w != 'none']
    
    for window_name in window_names:
        window_df = results_df[
            (results_df['window_name'] == window_name) | 
            (results_df['condition'] == 'none')
        ].copy()
        
        sample_rows = results_df[results_df['window_name'] == window_name]
        if len(sample_rows) == 0:
            continue
        
        # Plot sequence alpha
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for row_idx, row_conditions in enumerate(conditions_grid):
            for col_idx, (patch_block, patch_mode, patch_mask_mode) in enumerate(row_conditions):
                ax = axes[row_idx, col_idx]
                
                subset_sample = sample_rows[sample_rows['patch_block'] == patch_block]
                if len(subset_sample) > 0:
                    intervention_start = int(subset_sample.iloc[0]['intervention_start'])
                    intervention_end = int(subset_sample.iloc[0]['intervention_end'])
                else:
                    intervention_start, intervention_end = patch_block, 47
                
                for condition in colors.keys():
                    subset = window_df[
                        (window_df['patch_block'] == patch_block) &
                        (window_df['patch_mode'] == patch_mode) &
                        (window_df['patch_mask_mode'] == patch_mask_mode) &
                        (window_df['condition'] == condition)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    avg = subset.groupby('observation_block').mean(numeric_only=True).reset_index()
                    ax.plot(avg['observation_block'], avg['seq_interp_alpha'],
                            label=condition, color=colors[condition],
                            linestyle=linestyles[condition], linewidth=2)
                
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=patch_block, color='black', linestyle='--', linewidth=2)
                ax.axvspan(intervention_start, intervention_end, alpha=0.1, color='yellow')
                
                ax.set_xlabel('Observation Block', fontsize=11)
                ax.set_ylabel('Sequence α', fontsize=11)
                ax.set_title(f'Block {patch_block}, {patch_mode.upper()}, {patch_mask_mode}', fontsize=12)
                ax.legend(loc='best', fontsize=7)
                ax.set_ylim(-0.1, 1.2)
                ax.grid(alpha=0.3)
        
        plt.suptitle(f'SEQUENCE Interpolation α: Freezing vs Zero Ablation\n{window_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sequence_alpha_{window_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot pairwise intra alpha
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for row_idx, row_conditions in enumerate(conditions_grid):
            for col_idx, (patch_block, patch_mode, patch_mask_mode) in enumerate(row_conditions):
                ax = axes[row_idx, col_idx]
                
                subset_sample = sample_rows[sample_rows['patch_block'] == patch_block]
                if len(subset_sample) > 0:
                    intervention_start = int(subset_sample.iloc[0]['intervention_start'])
                    intervention_end = int(subset_sample.iloc[0]['intervention_end'])
                else:
                    intervention_start, intervention_end = patch_block, 47
                
                for condition in colors.keys():
                    subset = window_df[
                        (window_df['patch_block'] == patch_block) &
                        (window_df['patch_mode'] == patch_mode) &
                        (window_df['patch_mask_mode'] == patch_mask_mode) &
                        (window_df['condition'] == condition)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    avg = subset.groupby('observation_block').mean(numeric_only=True).reset_index()
                    ax.plot(avg['observation_block'], avg['pw_intra_interp_alpha'],
                            label=condition, color=colors[condition],
                            linestyle=linestyles[condition], linewidth=2)
                
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=patch_block, color='black', linestyle='--', linewidth=2)
                ax.axvspan(intervention_start, intervention_end, alpha=0.1, color='yellow')
                
                ax.set_xlabel('Observation Block', fontsize=11)
                ax.set_ylabel('PW Intra α', fontsize=11)
                ax.set_title(f'Block {patch_block}, {patch_mode.upper()}, {patch_mask_mode}', fontsize=12)
                ax.legend(loc='best', fontsize=7)
                ax.set_ylim(-0.1, 1.2)
                ax.grid(alpha=0.3)
        
        plt.suptitle(f'PAIRWISE INTRA Interpolation α: Freezing vs Zero Ablation\n{window_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pairwise_alpha_{window_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved plots to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Representation flow analysis with freeze/zero interventions'
    )
    parser.add_argument('--dataset', type=str, default='data/single_block_patching_successes.csv',
                        help='Path to input parquet dataset')
    parser.add_argument('--output', type=str, default='representation_analysis_results',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N cases')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only generate plots from existing results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.plot_only:
        results_path = os.path.join(args.output, 'intervention_experiment_results.parquet')
        if os.path.exists(results_path):
            results_df = pd.read_parquet(results_path)
            print(f"Loaded {len(results_df)} rows from {results_path}")
            plot_results(results_df, args.output)
        else:
            print(f"No results found at {results_path}")
        return
    
    print("\nLoading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print("Model loaded")
    
    print(f"\nLoading dataset from {args.dataset}...")
    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} rows")
    print("\nBreakdown:")
    print(df.groupby(['patch_mode', 'patch_mask_mode', 'block_idx']).size())
    
    results_df = run_experiment(
        df=df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output,
        save_every=args.save_every,
    )
    
    if len(results_df) > 0:
        plot_results(results_df, args.output)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()