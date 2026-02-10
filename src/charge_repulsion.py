#!/usr/bin/env python
"""
Charge-Based Hairpin Disruption
===============================

Disrupts existing hairpins by adding "same charge" (repulsive) signals to
cross-strand positions, providing a complementary test of charge steering.

Hypothesis: If ESMFold uses charge complementarity for strand-strand contacts,
then adding repulsive signals (same charge on both strands) should disrupt
hairpins by pushing the strands apart.

Key differences from induction experiment:
- Applied to DONOR sequences that already have hairpins
- Uses SAME charge on both strands (repulsive) instead of opposite (attractive)
- Measures INCREASED distances and DECREASED H-bonds

Supports multiple charge modes:
- both_positive: Add positive charge to both strands
- both_negative: Add negative charge to both strands
- opposite: Control condition (should not disrupt)

Usage:
    python charge_repulsion.py \
        --training_parquet training_sequences.parquet \
        --patch_parquet patch_cases.parquet \
        --output results/ \
        --window_sizes 3 5 10 \
        --magnitudes 1 2 3 \
        --charge_mode both_positive
"""

import argparse
import os
import types
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput
)
from transformers.models.esm.openfold_utils import make_atom14_masks
from transformers.utils import ContextManagers
import sys

# Add project root (parent of src/) to path so `src.*` imports work without PYTHONPATH
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.charge_dom_training import (
    ChargeDirections, load_directions, save_directions,
    train_dom_vectors, AA_TO_IDX,
)
import warnings

warnings.filterwarnings("ignore", message="parse error at line.*mmCIF")

# ============================================================================
# CONSTANTS
# ============================================================================

NUM_BLOCKS = 48

# Intervention type configurations (s-only for this experiment)
INTERVENTION_CONFIGS = {
    's': ['s'],
}

# Charge modes for disruption
CHARGE_MODES = {
    'both_positive': (+1, +1),   # Both strands get positive charge signal
    'both_negative': (-1, -1),   # Both strands get negative charge signal
    'opposite': (+1, -1),        # Original induction mode (for comparison)
}

# Block sweep modes
BLOCK_SWEEP_MODES = {
    'single': 'Intervene at single block only',
    'cumulative_forward': 'Intervene at block X and all blocks after',
    'cumulative_backward': 'Intervene at block X and all blocks before',
    'sliding_window': 'Sliding window of specified size(s)',
    'all': 'Intervene at all blocks (no sweep)',
}

# H-bond distance threshold (Angstroms)
HBOND_DISTANCE_THRESHOLD = 3.5


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HairpinTopology:
    """Defines the topology of a hairpin."""
    strand1_start: int
    strand1_end: int
    turn_start: int
    turn_end: int
    strand2_start: int
    strand2_end: int
    cross_strand_pairs: List[Tuple[int, int]]


# ============================================================================
# MODEL FORWARD PASSES
# ============================================================================

def baseline_forward_trunk(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """Standard forward pass collecting representations including seq2pair."""
    device = seq_feats.device
    s_s_0, s_z_0 = seq_feats, pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        no_recycles = no_recycles + 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        
        s_list, z_list, seq2pair_list = [], [], []
        for block in self.blocks:
            seq2pair_out = block.sequence_to_pair(block.layernorm_1(s))
            seq2pair_list.append(seq2pair_out.detach().clone())
            
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            s_list.append(s.detach().clone())
            z_list.append(z.detach().clone())
        return s, z, s_list, z_list, seq2pair_list

    s_s, s_z = s_s_0, s_z_0
    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)
            
            s_s, s_z, s_list, z_list, seq2pair_list = trunk_iter(
                s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask
            )
            
            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa, mask.float(),
            )
            
            recycle_s, recycle_z = s_s, s_z
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3], 3.375, 21.375, self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z
    structure["s_list"] = s_list
    structure["z_list"] = z_list
    structure["seq2pair_list"] = seq2pair_list
    structure["aatype"] = true_aa
    return structure


def baseline_forward(self, input_ids, attention_mask=None, position_ids=None,
                     masking_pattern=None, num_recycles=None, **kwargs):
    """Baseline forward pass."""
    cfg = self.config.esmfold_config
    aa = input_ids
    B, L = aa.shape
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)
    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype).detach()
    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    
    s_s_0 = self.esm_s_mlp(esm_s)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(aa)

    return self.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)


def intervention_forward_trunk(
    self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles,
    intervention_blocks: set,
    s_interventions: Optional[Dict[int, torch.Tensor]] = None,
):
    """Forward pass with s-only interventions at specified blocks."""
    device = seq_feats.device
    s_s_0, s_z_0 = seq_feats, pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        no_recycles = no_recycles + 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        
        for block_idx, block in enumerate(self.blocks):
            # Apply s intervention before block processes it
            if block_idx in intervention_blocks:
                if s_interventions is not None and block_idx in s_interventions:
                    s = s + s_interventions[block_idx].unsqueeze(0)
            
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
        
        return s, z

    s_s, s_z = s_s_0, s_z_0
    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)
            
            s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)
            
            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa, mask.float(),
            )
            
            recycle_s, recycle_z = s_s, s_z
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3], 3.375, 21.375, self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z
    structure["aatype"] = true_aa
    return structure


def intervention_forward(
    self, input_ids, attention_mask=None, position_ids=None,
    masking_pattern=None, num_recycles=None,
    intervention_blocks: set = None,
    s_interventions: Optional[Dict[int, torch.Tensor]] = None,
    **kwargs
):
    """Forward pass with s-only interventions."""
    cfg = self.config.esmfold_config
    aa = input_ids
    B, L = aa.shape
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)
    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype).detach()
    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    
    s_s_0 = self.esm_s_mlp(esm_s)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(aa)

    return self.trunk(
        s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles,
        intervention_blocks=intervention_blocks or set(),
        s_interventions=s_interventions,
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_cb_distances(positions: torch.Tensor) -> torch.Tensor:
    """Compute Cβ-Cβ distances from atom positions."""
    L = positions.shape[0]
    CB_IDX, CA_IDX = 3, 1
    
    cb_coords = []
    for i in range(L):
        cb = positions[i, CB_IDX]
        if torch.norm(cb) < 0.1:
            cb = positions[i, CA_IDX]
        cb_coords.append(cb)
    cb_coords = torch.stack(cb_coords)
    
    diff = cb_coords.unsqueeze(0) - cb_coords.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)
    return distances


def compute_backbone_no_distances(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute N-O distances for potential H-bond detection.
    Returns matrix where [i,j] is distance from N of residue i to O of residue j.
    
    Atom indices in atom14: N=0, CA=1, C=2, O=3
    """
    N_IDX, O_IDX = 0, 3
    
    n_coords = positions[:, N_IDX, :]  # [L, 3]
    o_coords = positions[:, O_IDX, :]  # [L, 3]
    
    # N[i] to O[j] distances
    diff = n_coords.unsqueeze(1) - o_coords.unsqueeze(0)  # [L, L, 3]
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


def compute_hbond_metrics(
    positions: torch.Tensor,
    cross_strand_pairs: List[Tuple[int, int]],
    threshold: float = HBOND_DISTANCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute hydrogen bonding metrics for cross-strand pairs.
    
    In antiparallel beta sheets, H-bonds occur between:
    - N of residue i and O of residue j
    - O of residue i and N of residue j
    
    Args:
        positions: [L, 14, 3] atom positions
        cross_strand_pairs: List of (i, j) residue pairs
        threshold: Distance threshold for H-bond (default 3.5 Å)
    
    Returns:
        Dict with H-bond metrics
    """
    no_distances = compute_backbone_no_distances(positions)
    
    hbond_count = 0
    hbond_distances = []
    pair_hbond_details = []
    
    for i, j in cross_strand_pairs:
        # Check both directions: N_i-O_j and N_j-O_i
        dist_ni_oj = no_distances[i, j].item()
        dist_nj_oi = no_distances[j, i].item()
        
        pair_has_hbond = False
        pair_min_dist = min(dist_ni_oj, dist_nj_oi)
        
        if dist_ni_oj < threshold:
            hbond_count += 1
            hbond_distances.append(dist_ni_oj)
            pair_has_hbond = True
        
        if dist_nj_oi < threshold:
            hbond_count += 1
            hbond_distances.append(dist_nj_oi)
            pair_has_hbond = True
        
        pair_hbond_details.append({
            'i': i,
            'j': j,
            'dist_ni_oj': dist_ni_oj,
            'dist_nj_oi': dist_nj_oi,
            'has_hbond': pair_has_hbond,
            'min_dist': pair_min_dist,
        })
    
    # Summary statistics
    n_pairs = len(cross_strand_pairs)
    n_pairs_with_hbond = sum(1 for p in pair_hbond_details if p['has_hbond'])
    
    return {
        'n_hbonds': hbond_count,
        'n_pairs_with_hbond': n_pairs_with_hbond,
        'n_cross_strand_pairs': n_pairs,
        'hbond_fraction': n_pairs_with_hbond / n_pairs if n_pairs > 0 else 0,
        'mean_hbond_dist': np.mean(hbond_distances) if hbond_distances else None,
        'min_hbond_dist': np.min(hbond_distances) if hbond_distances else None,
        'mean_min_no_dist': np.mean([p['min_dist'] for p in pair_hbond_details]) if pair_hbond_details else None,
        'pair_details': pair_hbond_details,
    }



def define_hairpin_topology(
    hairpin_start: int,
    strand1_length: int,
    loop_length: int,
    strand2_length: int,
) -> HairpinTopology:
    """
    Define a hairpin topology from explicit strand and loop lengths.
    
    All boundaries are half-open: strand1 = [strand1_start, strand1_end), etc.
    
    Args:
        hairpin_start: Start of hairpin region (inclusive, 0-indexed)
        strand1_length: Length of first strand
        loop_length: Length of the turn/loop
        strand2_length: Length of second strand
    """
    strand1_start = hairpin_start
    strand1_end = strand1_start + strand1_length
    turn_start = strand1_end
    turn_end = turn_start + loop_length
    strand2_start = turn_end
    strand2_end = strand2_start + strand2_length
    
    # Build antiparallel cross-strand pairs
    cross_strand_pairs = []
    for i in range(min(strand1_length, strand2_length)):
        res_i = strand1_start + i
        res_j = strand2_end - 1 - i
        if res_j >= strand2_start:
            cross_strand_pairs.append((res_i, res_j))
    
    return HairpinTopology(
        strand1_start=strand1_start,
        strand1_end=strand1_end,
        turn_start=turn_start,
        turn_end=turn_end,
        strand2_start=strand2_start,
        strand2_end=strand2_end,
        cross_strand_pairs=cross_strand_pairs,
    )



def get_block_sets_for_sweep(
    sweep_mode: str,
    sweep_blocks: List[int],
    total_blocks: int = NUM_BLOCKS,
    window_sizes: List[int] = None,
) -> List[Tuple[str, set]]:
    """
    Generate block sets for sweeping experiments.
    
    Returns:
        List of (name, block_set) tuples
    """
    if sweep_mode == 'all':
        return [('all_blocks', set(range(total_blocks)))]
    
    block_sets = []
    
    if sweep_mode == 'sliding_window':
        for window_size in (window_sizes or [5]):
            for start_block in range(total_blocks - window_size + 1):
                end_block = start_block + window_size - 1
                name = f'window_{window_size}_blocks_{start_block}_to_{end_block}'
                blocks = set(range(start_block, start_block + window_size))
                block_sets.append((name, blocks))
        return block_sets
    
    for block in sweep_blocks:
        if sweep_mode == 'single':
            name = f'block_{block}'
            blocks = {block}
        
        elif sweep_mode == 'cumulative_forward':
            name = f'blocks_{block}_to_47'
            blocks = set(range(block, total_blocks))
        
        elif sweep_mode == 'cumulative_backward':
            name = f'blocks_0_to_{block}'
            blocks = set(range(0, block + 1))
        
        else:
            raise ValueError(f"Unknown sweep mode: {sweep_mode}")
        
        block_sets.append((name, blocks))
    
    return block_sets


def get_baseline_structure(model, seq: str, tokenizer, device: str) -> EsmForProteinFoldingOutput:
    """Run baseline forward pass and return structure outputs."""
    model.forward = types.MethodType(baseline_forward, model)
    model.trunk.forward = types.MethodType(baseline_forward_trunk, model.trunk)
    
    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(**inputs, num_recycles=0)
    
    L = len(seq)
    B = 1
    
    structure = {
        "aatype": outputs["aatype"],
        "positions": outputs["positions"],
        "states": outputs["states"],
        "s_s": outputs["s_s"],
        "s_z": outputs["s_z"],
    }
    
    make_atom14_masks(structure)
    structure["residue_index"] = torch.arange(L, device=device).unsqueeze(0)
    
    lddt_head = model.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, model.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
    structure["plddt"] = plddt
    
    s_list = outputs.get("s_list")
    z_list = outputs.get("z_list")
    seq2pair_list = outputs.get("seq2pair_list")
    
    output = EsmForProteinFoldingOutput(
        positions=structure["positions"],
        states=structure["states"],
        s_s=structure["s_s"],
        s_z=structure["s_z"],
        aatype=structure["aatype"],
        atom14_atom_exists=structure["atom14_atom_exists"],
        residx_atom14_to_atom37=structure["residx_atom14_to_atom37"],
        residx_atom37_to_atom14=structure["residx_atom37_to_atom14"],
        atom37_atom_exists=structure["atom37_atom_exists"],
        residue_index=structure["residue_index"],
        lddt_head=structure["lddt_head"],
        plddt=structure["plddt"],
    )
    
    output.s_list = s_list
    output.z_list = z_list
    output.seq2pair_list = seq2pair_list
    
    return output


def run_intervention(
    model, seq: str, tokenizer, device: str,
    intervention_blocks: set,
    s_interventions: Optional[Dict[int, torch.Tensor]] = None,
) -> EsmForProteinFoldingOutput:
    """Run forward pass with s-only intervention."""
    model.forward = types.MethodType(intervention_forward, model)
    model.trunk.forward = types.MethodType(intervention_forward_trunk, model.trunk)
    
    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(
            **inputs, num_recycles=0,
            intervention_blocks=intervention_blocks,
            s_interventions=s_interventions,
        )
    
    L = len(seq)
    B = 1
    
    structure = {
        "aatype": outputs["aatype"],
        "positions": outputs["positions"],
        "states": outputs["states"],
        "s_s": outputs["s_s"],
        "s_z": outputs["s_z"],
    }
    
    make_atom14_masks(structure)
    structure["residue_index"] = torch.arange(L, device=device).unsqueeze(0)
    
    lddt_head = model.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, model.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
    structure["plddt"] = plddt
    
    output = EsmForProteinFoldingOutput(
        positions=structure["positions"],
        states=structure["states"],
        s_s=structure["s_s"],
        s_z=structure["s_z"],
        aatype=structure["aatype"],
        atom14_atom_exists=structure["atom14_atom_exists"],
        residx_atom14_to_atom37=structure["residx_atom14_to_atom37"],
        residx_atom37_to_atom14=structure["residx_atom37_to_atom14"],
        atom37_atom_exists=structure["atom37_atom_exists"],
        residue_index=structure["residue_index"],
        lddt_head=structure["lddt_head"],
        plddt=structure["plddt"],
    )
    
    return output


# ============================================================================
# INTERVENTION CONSTRUCTION - MODIFIED FOR DISRUPTION
# ============================================================================

def create_hairpin_disruption_interventions(
    topology: HairpinTopology,
    seq_len: int,
    directions: ChargeDirections,
    blocks: List[int],
    magnitude: float,
    intervention_config: str,
    charge_mode: str,  # 'both_positive', 'both_negative', or 'opposite'
    device: str,
) -> Optional[Dict[int, torch.Tensor]]:
    """
    Create s-only interventions to DISRUPT hairpin formation by applying same charge to both strands.
    
    Args:
        topology: Hairpin topology with strand positions
        seq_len: Length of sequence
        directions: Computed charge directions (s-only)
        blocks: Which blocks to intervene at
        magnitude: How many std devs to move
        intervention_config: Which representations to intervene on (only 's' supported)
        charge_mode: 'both_positive', 'both_negative', or 'opposite' (for comparison)
        device: torch device
    
    Returns:
        s_interventions dict, or None
    """
    active_reps = INTERVENTION_CONFIGS.get(intervention_config, [intervention_config])
    
    # Get charge signs for each strand
    strand1_sign, strand2_sign = CHARGE_MODES[charge_mode]
    
    s_interventions = None
    
    # s interventions - apply charge signal to each residue in strands
    if 's' in active_reps:
        s_interventions = {}
        for block in blocks:
            if block not in directions.s_directions:
                continue
            
            s_dir = directions.s_directions[block]
            s_std = directions.s_stds[block]
            
            s_int = torch.zeros(seq_len, len(s_dir), dtype=torch.float32, device=device)
            
            # Strand 1: apply strand1_sign charge
            delta1 = strand1_sign * magnitude * s_std * s_dir
            for i in range(topology.strand1_start, topology.strand1_end):
                s_int[i] = torch.tensor(delta1, device=device)
            
            # Strand 2: apply strand2_sign charge (SAME sign for disruption)
            delta2 = strand2_sign * magnitude * s_std * s_dir
            for j in range(topology.strand2_start, topology.strand2_end):
                s_int[j] = torch.tensor(delta2, device=device)
            
            s_interventions[block] = s_int
    
    return s_interventions


# ============================================================================
# EVALUATION
# ============================================================================

def compute_full_structure_metrics(
    outputs: EsmForProteinFoldingOutput,
    topology: HairpinTopology,
) -> Dict[str, Any]:
    """Compute comprehensive structure metrics including H-bonds."""
    positions = outputs.positions[-1, 0].cpu()
    
    # CB distances
    cb_distances = compute_cb_distances(positions)
    cross_strand_cb_dists = []
    for i, j in topology.cross_strand_pairs:
        cross_strand_cb_dists.append(cb_distances[i, j].item())
    
    # H-bond metrics
    hbond_metrics = compute_hbond_metrics(positions, topology.cross_strand_pairs)
    
    return {
        # CB distance metrics
        'mean_cross_strand_dist': np.mean(cross_strand_cb_dists) if cross_strand_cb_dists else None,
        'min_cross_strand_dist': np.min(cross_strand_cb_dists) if cross_strand_cb_dists else None,
        'max_cross_strand_dist': np.max(cross_strand_cb_dists) if cross_strand_cb_dists else None,
        'n_cb_contacts': sum(1 for d in cross_strand_cb_dists if d < 8.0),
        'n_pairs': len(cross_strand_cb_dists),
        # H-bond metrics
        'n_hbonds': hbond_metrics['n_hbonds'],
        'n_pairs_with_hbond': hbond_metrics['n_pairs_with_hbond'],
        'hbond_fraction': hbond_metrics['hbond_fraction'],
        'mean_hbond_dist': hbond_metrics['mean_hbond_dist'],
        'min_hbond_dist': hbond_metrics['min_hbond_dist'],
        'mean_min_no_dist': hbond_metrics['mean_min_no_dist'],
    }


# ============================================================================
# RESULTS MANAGER
# ============================================================================

class ResultsManager:
    """Manages incremental saving of results."""
    
    def __init__(self, output_path: str, output_dir: str, save_every: int = 10):
        self.output_path = output_path
        self.output_dir = output_dir
        self.save_every = save_every
        self.results = []
        self.cases_processed = 0
    
    def add_results(self, new_results: List[Dict]):
        self.results.extend(new_results)
    
    def mark_case_done(self):
        self.cases_processed += 1
        if self.cases_processed % self.save_every == 0:
            self.save()
    
    def save(self):
        if len(self.results) == 0:
            return
        df = pd.DataFrame(self.results)
        df.to_parquet(self.output_path, index=False)
        print(f"\n[Checkpoint] Saved {len(self.results)} results after {self.cases_processed} cases")
        
        try:
            analyze_disruption_results(df, self.output_dir)
        except Exception as e:
            print(f"[Warning] Could not update plots: {e}")
    
    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_hairpin_disruption_experiment(
    cases_df: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    directions: ChargeDirections,
    magnitudes: List[float],
    intervention_configs: List[str],
    charge_modes: List[str],
    block_sets: List[Tuple[str, set]],
    output_dir: str,
    save_every: int = 10,
) -> pd.DataFrame:
    """
    Run hairpin DISRUPTION experiments on sequences with hairpins.
    
    Supports two column naming conventions:
    1. From bb_motifs_local.csv: FullChainSequence, strand1_start_res, strand2_end_res, PDB
    2. From patching results: donor_sequence, donor_hairpin_start, donor_hairpin_end, donor_pdb
    
    Key differences from induction:
    - Uses sequences that already have hairpins
    - Applies same-charge interventions to disrupt
    - Expects increased distances and decreased H-bonds
    """
    results_manager = ResultsManager(
        output_path=os.path.join(output_dir, 'hairpin_disruption_results.parquet'),
        output_dir=output_dir,
        save_every=save_every
    )
    
    available_blocks = set(directions.s_directions.keys())
    
    for case_idx, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Cases"):
        # =====================================================================
        # Read sequence — supports both CSV formats
        # =====================================================================
        if 'FullChainSequence' in cases_df.columns:
            seq = row['FullChainSequence']
        elif 'donor_sequence' in cases_df.columns:
            seq = row['donor_sequence']
        else:
            print(f"  Skipping case {case_idx}: no sequence column found")
            continue
        
        L = len(seq)
        
        # =====================================================================
        # Read hairpin boundaries — supports both CSV formats
        # =====================================================================
        has_4_cols = 'strand1_start_res' in cases_df.columns
        has_2_cols = 'donor_hairpin_start' in cases_df.columns
        
        if has_4_cols:
            # donor_hairpins.csv: has all 4 strand boundaries
            s1_start = int(row['strand1_start_res'])
            s1_end = int(row['strand1_end_res'])
            s2_start = int(row['strand2_start_res'])
            s2_end = int(row['strand2_end_res'])
            
            # Bounds check
            if s2_end >= L:
                print(f"  Skipping case {case_idx}: strand end ({s2_end}) "
                      f"exceeds sequence length ({L})")
                continue
            
            # Convert inclusive boundaries to lengths
            topology = define_hairpin_topology(
                hairpin_start=s1_start,
                strand1_length=s1_end - s1_start + 1,  # inclusive -> length
                loop_length=s2_start - s1_end - 1,      # gap between strands
                strand2_length=s2_end - s2_start + 1,   # inclusive -> length
            )
        
        elif has_2_cols:
            # patching_successes.csv: has strand/loop lengths
            hairpin_start = int(row['donor_hairpin_start'])
            hairpin_end = int(row['donor_hairpin_end'])
            s1_len = int(row['donor_strand1_length'])
            loop_len = int(row['donor_loop_length'])
            s2_len = int(row['donor_strand2_length'])
            
            # Bounds check
            if hairpin_start + s1_len + loop_len + s2_len > L:
                print(f"  Skipping case {case_idx}: hairpin region exceeds "
                      f"sequence length ({L})")
                continue
            
            topology = define_hairpin_topology(
                hairpin_start=hairpin_start,
                strand1_length=s1_len,
                loop_length=loop_len,
                strand2_length=s2_len,
            )
                    
        else:
            print(f"  Skipping case {case_idx}: no hairpin boundary columns found")
            continue
        
        # Name/PDB
        if 'PDB' in cases_df.columns:
            name = row['PDB']
        elif 'donor_pdb' in cases_df.columns:
            name = row['donor_pdb']
        elif 'donor_name' in cases_df.columns:
            name = row['donor_name']
        else:
            name = f'case_{case_idx}'
        
        case_results = []
        
        print(f"\nCase {case_idx}: {name}")
        print(f"  Sequence length: {L}")
        print(f"  Topology: strand1=[{topology.strand1_start},{topology.strand1_end}), "
              f"turn=[{topology.turn_start},{topology.turn_end}), "
              f"strand2=[{topology.strand2_start},{topology.strand2_end})")
        print(f"  Cross-strand pairs: {topology.cross_strand_pairs}")
        
        # Baseline
        try:
            baseline_outputs = get_baseline_structure(model, seq, tokenizer, device)
        except Exception as e:
            print(f"  Baseline error: {e}")
            continue
        
        baseline_metrics = compute_full_structure_metrics(baseline_outputs, topology)

        print(f"  Baseline: "
              f"mean_dist={baseline_metrics['mean_cross_strand_dist']:.1f}Å, "
              f"n_hbonds={baseline_metrics['n_hbonds']}, "
              f"hbond_frac={baseline_metrics['hbond_fraction']:.2f}")
        
        # Skip cases where cross-strand distances are already too large
        # (strands not in contact → not a well-formed hairpin to disrupt)
        # if (baseline_metrics['mean_cross_strand_dist'] is not None
        #         and baseline_metrics['mean_cross_strand_dist'] > 8.5):
        #     print(f"  Skipping case {case_idx}: mean cross-strand distance "
        #           f"({baseline_metrics['mean_cross_strand_dist']:.1f}Å) > 8.5Å threshold")
        #     continue
        
        # Record baseline
        case_results.append({
            'case_idx': case_idx,
            'name': name,
            'sequence': seq,
            'strand1_start': topology.strand1_start,
            'strand1_end': topology.strand1_end,
            'strand2_start': topology.strand2_start,
            'strand2_end': topology.strand2_end,
            'intervention_config': 'baseline',
            'charge_mode': 'baseline',
            'block_set': 'baseline',
            'window_size': 0,
            'window_start': -1,
            'n_blocks': 0,
            'magnitude': 0.0,
            'mean_cross_strand_dist': baseline_metrics['mean_cross_strand_dist'],
            'min_cross_strand_dist': baseline_metrics['min_cross_strand_dist'],
            'n_cb_contacts': baseline_metrics['n_cb_contacts'],
            'n_cross_strand_pairs': baseline_metrics['n_pairs'],
            'n_hbonds': baseline_metrics['n_hbonds'],
            'n_pairs_with_hbond': baseline_metrics['n_pairs_with_hbond'],
            'hbond_fraction': baseline_metrics['hbond_fraction'],
            'mean_hbond_dist': baseline_metrics['mean_hbond_dist'],
            'min_hbond_dist': baseline_metrics['min_hbond_dist'],
            'mean_min_no_dist': baseline_metrics['mean_min_no_dist'],
        })
        
        # Interventions
        for block_set_name, intervention_blocks in block_sets:
            intervention_blocks = intervention_blocks & available_blocks
            
            if len(intervention_blocks) == 0:
                continue
            
            # Parse window info from block_set_name
            window_size = 0
            window_start = -1
            if 'window_' in block_set_name:
                parts = block_set_name.split('_')
                window_size = int(parts[1])
                window_start = int(parts[3])
            
            for charge_mode in charge_modes:
                for int_config in intervention_configs:
                    for magnitude in magnitudes:
                        s_ints= create_hairpin_disruption_interventions(
                            topology=topology,
                            seq_len=L,
                            directions=directions,
                            blocks=list(intervention_blocks),
                            magnitude=magnitude,
                            intervention_config=int_config,
                            charge_mode=charge_mode,
                            device=device,
                        )
                        
                        try:
                            int_outputs = run_intervention(
                                model, seq, tokenizer, device,
                                intervention_blocks=intervention_blocks,
                                s_interventions=s_ints,
                            )
                        except Exception as e:
                            print(f"  Intervention error ({int_config}, {charge_mode}, {block_set_name}, mag={magnitude}): {e}")
                            continue
                        
                        int_metrics = compute_full_structure_metrics(int_outputs, topology)
                        
                        # Compute changes
                        dist_change = None
                        hbond_change = None
                        if baseline_metrics['mean_cross_strand_dist'] and int_metrics['mean_cross_strand_dist']:
                            dist_change = int_metrics['mean_cross_strand_dist'] - baseline_metrics['mean_cross_strand_dist']
                        if baseline_metrics['n_hbonds'] is not None and int_metrics['n_hbonds'] is not None:
                            hbond_change = int_metrics['n_hbonds'] - baseline_metrics['n_hbonds']
                        
                        case_results.append({
                            'case_idx': case_idx,
                            'name': name,
                            'sequence': seq,
                            'strand1_start': topology.strand1_start,
                            'strand1_end': topology.strand1_end,
                            'strand2_start': topology.strand2_start,
                            'strand2_end': topology.strand2_end,
                            'intervention_config': int_config,
                            'charge_mode': charge_mode,
                            'block_set': block_set_name,
                            'window_size': window_size,
                            'window_start': window_start,
                            'n_blocks': len(intervention_blocks),
                            'magnitude': magnitude,
                            'mean_cross_strand_dist': int_metrics['mean_cross_strand_dist'],
                            'min_cross_strand_dist': int_metrics['min_cross_strand_dist'],
                            'n_cb_contacts': int_metrics['n_cb_contacts'],
                            'n_cross_strand_pairs': int_metrics['n_pairs'],
                            'n_hbonds': int_metrics['n_hbonds'],
                            'n_pairs_with_hbond': int_metrics['n_pairs_with_hbond'],
                            'hbond_fraction': int_metrics['hbond_fraction'],
                            'mean_hbond_dist': int_metrics['mean_hbond_dist'],
                            'min_hbond_dist': int_metrics['min_hbond_dist'],
                            'mean_min_no_dist': int_metrics['mean_min_no_dist'],
                            'dist_change': dist_change,
                            'hbond_change': hbond_change,
                        })
                        
                        # Log successful disruption (H-bonds lost)
                        if hbond_change is not None and hbond_change < 0:
                            print(f"  *** H-BONDS DISRUPTED! charge={charge_mode}, config={int_config}, "
                                  f"blocks={block_set_name}, mag={magnitude}, "
                                  f"hbond_change={hbond_change} ***")
                        
                        # Log if distance increased (expected for disruption)
                        if dist_change is not None and dist_change > 2.0:
                            print(f"  + Distance increased: {baseline_metrics['mean_cross_strand_dist']:.1f} -> "
                                  f"{int_metrics['mean_cross_strand_dist']:.1f} (+{dist_change:.1f}Å) | "
                                  f"{charge_mode}, {block_set_name}, mag={magnitude}")
                        
                        # Log if H-bonds decreased (expected for disruption)
                        if hbond_change is not None and hbond_change < 0:
                            print(f"  - H-bonds decreased: {baseline_metrics['n_hbonds']} -> "
                                  f"{int_metrics['n_hbonds']} ({hbond_change}) | "
                                  f"{charge_mode}, {block_set_name}, mag={magnitude}")
                            print(f"  Distance change: {baseline_metrics['mean_cross_strand_dist']:.1f} -> "
                                  f"{int_metrics['mean_cross_strand_dist']:.1f} (+{dist_change:.1f}Å) | "
                                  f"{charge_mode}, {block_set_name}, mag={magnitude}")
                        
                        del s_ints, int_outputs
                        torch.cuda.empty_cache()
        
        results_manager.add_results(case_results)
        results_manager.mark_case_done()
        
        del baseline_outputs
        torch.cuda.empty_cache()
    
    results_manager.save()
    return results_manager.get_dataframe()


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_disruption_results(results_df: pd.DataFrame, output_dir: str):
    """Analyze and visualize disruption results."""
    print("\n" + "="*60)
    print("HAIRPIN DISRUPTION RESULTS")
    print("="*60)
    
    baseline = results_df[results_df['intervention_config'] == 'baseline']
    print(f"\nBaseline: {len(baseline)} cases")
    print(f"  Mean cross-strand distance: {baseline['mean_cross_strand_dist'].mean():.1f}Å")
    print(f"  Mean H-bonds: {baseline['n_hbonds'].mean():.1f}")
    print(f"  Mean H-bond fraction: {baseline['hbond_fraction'].mean():.2f}")

    interventions = results_df[results_df['intervention_config'] != 'baseline']

    print(f"\nInterventions: {len(interventions)} total")

    # Check for sliding window data
    has_window = 'window_size' in interventions.columns and interventions['window_size'].max() > 0

    # Distance change by charge mode
    print("\nMean distance change by charge mode (positive = disruption):")
    for charge_mode in sorted(interventions['charge_mode'].unique()):
        subset = interventions[interventions['charge_mode'] == charge_mode]
        mean_change = subset['dist_change'].mean()
        std_change = subset['dist_change'].std()
        print(f"  {charge_mode}: {mean_change:+.2f} ± {std_change:.2f} Å")
    
    # H-bond change by charge mode
    print("\nMean H-bond change by charge mode (negative = disruption):")
    for charge_mode in sorted(interventions['charge_mode'].unique()):
        subset = interventions[interventions['charge_mode'] == charge_mode]
        mean_change = subset['hbond_change'].mean()
        std_change = subset['hbond_change'].std()
        print(f"  {charge_mode}: {mean_change:+.2f} ± {std_change:.2f}")
    
    # By intervention config
    print("\nMean distance change by intervention config:")
    for int_config in sorted(interventions['intervention_config'].unique()):
        subset = interventions[interventions['intervention_config'] == int_config]
        mean_change = subset['dist_change'].mean()
        std_change = subset['dist_change'].std()
        print(f"  {int_config}: {mean_change:+.2f} ± {std_change:.2f} Å")
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    
    charge_modes_list = sorted(interventions['charge_mode'].unique())
    charge_colors = {
        'both_positive': 'red',
        'both_negative': 'blue', 
        'opposite': 'green',
    }
    
    configs = sorted(interventions['intervention_config'].unique())
    config_markers = dict(zip(configs, ['o', 's', '^', 'D', 'v', '<', '>', 'p']))
    
    if has_window:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])

        # Top-left: Distance change by window position (grouped by charge mode)
        ax = axes[0, 0]
        for charge_mode in charge_modes_list:
            cm_data = interventions[interventions['charge_mode'] == charge_mode]
            for ws in window_sizes:
                ws_data = cm_data[cm_data['window_size'] == ws]
                window_starts = sorted(ws_data['window_start'].unique())

                means = []
                valid_starts = []
                for wstart in window_starts:
                    subset = ws_data[ws_data['window_start'] == wstart]
                    if len(subset) > 0:
                        dist_changes = subset['dist_change'].dropna()
                        if len(dist_changes) > 0:
                            means.append(dist_changes.mean())
                            valid_starts.append(wstart)

                if len(means) > 0:
                    linestyle = '-' if ws == window_sizes[0] else '--' if ws == window_sizes[-1] else ':'
                    ax.plot(valid_starts, means, linestyle, label=f'{charge_mode}, w={ws}',
                           color=charge_colors.get(charge_mode, 'gray'), linewidth=2,
                           markersize=4, alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Window Start Block', fontsize=12)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=12)
        ax.set_title('Distance Change by Window Position\n(positive = disruption)', fontsize=13)
        ax.legend(loc='best', fontsize=7)
        ax.grid(alpha=0.3)

        # Top-right: H-bond change by window position
        ax = axes[0, 1]
        for charge_mode in charge_modes_list:
            cm_data = interventions[interventions['charge_mode'] == charge_mode]
            for ws in window_sizes:
                ws_data = cm_data[cm_data['window_size'] == ws]
                window_starts = sorted(ws_data['window_start'].unique())

                means = []
                valid_starts = []
                for wstart in window_starts:
                    subset = ws_data[ws_data['window_start'] == wstart]
                    if len(subset) > 0:
                        hbond_changes = subset['hbond_change'].dropna()
                        if len(hbond_changes) > 0:
                            means.append(hbond_changes.mean())
                            valid_starts.append(wstart)

                if len(means) > 0:
                    linestyle = '-' if ws == window_sizes[0] else '--' if ws == window_sizes[-1] else ':'
                    ax.plot(valid_starts, means, linestyle, label=f'{charge_mode}, w={ws}',
                           color=charge_colors.get(charge_mode, 'gray'), linewidth=2,
                           markersize=4, alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Window Start Block', fontsize=12)
        ax.set_ylabel('Mean H-bond Change', fontsize=12)
        ax.set_title('H-bond Change by Window Position\n(negative = disruption)', fontsize=13)
        ax.legend(loc='best', fontsize=7)
        ax.grid(alpha=0.3)

        # Bottom-left: Distance change by magnitude, colored by charge mode
        ax = axes[1, 0]
        for charge_mode in charge_modes_list:
            subset = interventions[interventions['charge_mode'] == charge_mode]
            if len(subset) == 0:
                continue

            means = []
            stds = []
            mags = sorted(subset['magnitude'].unique())
            for mag in mags:
                mag_subset = subset[subset['magnitude'] == mag]
                means.append(mag_subset['dist_change'].mean())
                stds.append(mag_subset['dist_change'].std() / np.sqrt(len(mag_subset)))

            ax.errorbar(mags, means, yerr=stds, fmt='o-',
                       color=charge_colors.get(charge_mode, 'gray'),
                       label=charge_mode, linewidth=2, markersize=6, capsize=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Magnitude (std devs)', fontsize=12)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=12)
        ax.set_title('Distance Change by Magnitude', fontsize=13)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

        # Bottom-right: H-bond change by magnitude
        ax = axes[1, 1]
        for charge_mode in charge_modes_list:
            subset = interventions[interventions['charge_mode'] == charge_mode]
            if len(subset) == 0:
                continue

            means = []
            stds = []
            mags = sorted(subset['magnitude'].unique())
            for mag in mags:
                mag_subset = subset[subset['magnitude'] == mag]
                means.append(mag_subset['hbond_change'].mean())
                stds.append(mag_subset['hbond_change'].std() / np.sqrt(len(mag_subset)))

            ax.errorbar(mags, means, yerr=stds, fmt='o-',
                       color=charge_colors.get(charge_mode, 'gray'),
                       label=charge_mode, linewidth=2, markersize=6, capsize=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Magnitude (std devs)', fontsize=12)
        ax.set_ylabel('Mean H-bond Change', fontsize=12)
        ax.set_title('H-bond Change by Magnitude', fontsize=13)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    else:
        # Non-window version
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Distance change by magnitude
        ax = axes[0]
        for charge_mode in charge_modes_list:
            subset = interventions[interventions['charge_mode'] == charge_mode]
            if len(subset) == 0:
                continue

            means = []
            stds = []
            mags = sorted(subset['magnitude'].unique())
            for mag in mags:
                mag_subset = subset[subset['magnitude'] == mag]
                means.append(mag_subset['dist_change'].mean())
                stds.append(mag_subset['dist_change'].std() / np.sqrt(len(mag_subset)))

            ax.errorbar(mags, means, yerr=stds, fmt='o-',
                       color=charge_colors.get(charge_mode, 'gray'),
                       label=charge_mode, linewidth=2, markersize=6, capsize=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Magnitude (std devs)', fontsize=12)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=12)
        ax.set_title('Distance Change (positive = disruption)', fontsize=13)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

        # Right: H-bond change
        ax = axes[1]
        for charge_mode in charge_modes_list:
            subset = interventions[interventions['charge_mode'] == charge_mode]
            if len(subset) == 0:
                continue

            means = []
            stds = []
            mags = sorted(subset['magnitude'].unique())
            for mag in mags:
                mag_subset = subset[subset['magnitude'] == mag]
                means.append(mag_subset['hbond_change'].mean())
                stds.append(mag_subset['hbond_change'].std() / np.sqrt(len(mag_subset)))

            ax.errorbar(mags, means, yerr=stds, fmt='o-',
                       color=charge_colors.get(charge_mode, 'gray'),
                       label=charge_mode, linewidth=2, markersize=6, capsize=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Magnitude (std devs)', fontsize=12)
        ax.set_ylabel('Mean H-bond Change', fontsize=12)
        ax.set_title('H-bond Change (negative = disruption)', fontsize=13)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disruption_summary.png'), dpi=150)
    plt.close()
    
    # Additional comparison plot: both_positive vs both_negative
    if 'both_positive' in charge_modes_list and 'both_negative' in charge_modes_list:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Compare distance changes
        ax = axes[0]
        for charge_mode in ['both_positive', 'both_negative']:
            subset = interventions[interventions['charge_mode'] == charge_mode]
            ax.hist(subset['dist_change'].dropna(), bins=30, alpha=0.5,
                   label=charge_mode, color=charge_colors[charge_mode])
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Distance Change (Å)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Distance Changes', fontsize=13)
        ax.legend()
        
        # Compare H-bond changes
        ax = axes[1]
        for charge_mode in ['both_positive', 'both_negative']:
            subset = interventions[interventions['charge_mode'] == charge_mode]
            ax.hist(subset['hbond_change'].dropna(), bins=30, alpha=0.5,
                   label=charge_mode, color=charge_colors[charge_mode])
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('H-bond Change', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of H-bond Changes', fontsize=13)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'charge_mode_comparison.png'), dpi=150)
        plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hairpin DISRUPTION experiment with same-charge interventions'
    )
    parser.add_argument('--hairpin_dataset', type=str, default='data/single_block_patching_successes.csv',
                        help='CSV with hairpin sequences for disruption experiments')
    parser.add_argument('--output', type=str, default='final_final_seperator_one_two_unfiltered_05',
                        help='Output directory')
    parser.add_argument('--directions', type=str, default=None,
                        help='Path to pre-trained DoM directions. If not provided, trains from --probing_dataset')
    parser.add_argument('--probing_dataset', type=str, default=None,
                        help='Path to probing dataset CSV (required if --directions not provided)')
    parser.add_argument('--n_cases', type=int, default=400,
                        help='Limit number of cases')
    parser.add_argument('--magnitudes', type=float, nargs='+',
                        default=[0.5, 1.0],
                        help='Intervention magnitudes (in std devs)')
    parser.add_argument('--intervention_configs', type=str, nargs='+',
                        default=['s'],
                        help='Intervention configs (s-only for this experiment)')
    parser.add_argument('--charge_modes', type=str, nargs='+',
                        default=['both_positive', 'both_negative'],
                        help='Charge modes: both_positive, both_negative, opposite')
    parser.add_argument('--save_every', type=int, default=3,
                        help='Save checkpoint every N cases')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--block_sweep', type=str, default='sliding_window',
                        choices=['single', 'cumulative_forward', 'cumulative_backward', 'sliding_window', 'all'],
                        help='Block sweep mode')
    parser.add_argument('--sweep_blocks', type=int, nargs='+', default=None,
                        help='Which blocks to sweep over (for single/cumulative modes)')
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[15],
                        help='Window sizes for sliding_window mode')
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    
    # =========================================================================
    # Load hairpin cases for disruption experiments
    # =========================================================================
    print(f"\nLoading hairpin cases from {args.hairpin_dataset}...")
    hairpin_df = pd.read_csv(args.hairpin_dataset)
    print(f"Loaded {len(hairpin_df)} hairpin sequences")
    
    # Show columns
    print(f"Columns: {list(hairpin_df.columns)}")
    
    if args.n_cases is not None:
        hairpin_df = hairpin_df.head(args.n_cases)
        print(f"Limited to {len(hairpin_df)} cases")
    
    hairpin_df = hairpin_df.reset_index(drop=True)
    
    # =========================================================================
    # Load or Train DoM directions (s-only)
    # =========================================================================
    blocks_to_use = list(range(0, NUM_BLOCKS))

    if args.directions and os.path.exists(args.directions):
        print(f"\nLoading directions from {args.directions}...")
        directions = load_directions(args.directions)
        print(f"Loaded s directions for {len(directions.s_directions)} blocks")
    else:
        if args.probing_dataset is None:
            raise ValueError("Must provide either --directions (to a valid file) or --probing_dataset (to train new directions)")

        directions_save_path = args.directions or os.path.join(args.output, 'charge_directions.pt')

        print("\n" + "="*60)
        print("TRAINING DoM VECTORS")
        print("="*60)

        print(f"\nLoading probing dataset from {args.probing_dataset}...")
        probing_df = pd.read_csv(args.probing_dataset)

        if 'split' in probing_df.columns and 'structure_type' in probing_df.columns:
            print(f"\nDataset composition:")
            print(probing_df.groupby(['split', 'structure_type']).size().unstack(fill_value=0))

        train_df = probing_df[
            (probing_df['split'] == 'train') &
            (probing_df['structure_type'] == 'alpha_helical')
        ]

        seq_col = 'FullChainSequence' if 'FullChainSequence' in train_df.columns else 'sequence'
        train_seqs = train_df[seq_col].tolist()
        train_seqs = [s for s in train_seqs if all(aa in AA_TO_IDX for aa in s)]

        print(f"\nTrain sequences (alpha_helical only): {len(train_seqs)}")

        import random
        random.seed(args.seed)
        random.shuffle(train_seqs)

        directions = train_dom_vectors(
            sequences=train_seqs,
            model=model,
            tokenizer=tokenizer,
            device=device,
            blocks_to_train=blocks_to_use,
        )

        save_directions(directions, directions_save_path)
        print(f"Trained and saved directions for {len(directions.s_directions)} blocks")
    
    # =========================================================================
    # Set up block sets for sweep
    # =========================================================================
    block_sets = get_block_sets_for_sweep(
        sweep_mode=args.block_sweep,
        sweep_blocks=args.sweep_blocks or list(range(0, NUM_BLOCKS, 4)),
        total_blocks=NUM_BLOCKS,
        window_sizes=args.window_sizes,
    )
    
    print(f"\nBlock sweep mode: {args.block_sweep}")
    if args.block_sweep == 'sliding_window':
        print(f"Window sizes: {args.window_sizes}")
    print(f"Total block sets: {len(block_sets)}")
    
    # =========================================================================
    # Run disruption experiments
    # =========================================================================
    print(f"\nRunning hairpin DISRUPTION experiments...")
    print(f"Charge modes: {args.charge_modes}")
    print(f"Magnitudes: {args.magnitudes}")
    print(f"Intervention configs: {args.intervention_configs}")
    
    results_df = run_hairpin_disruption_experiment(
        cases_df=hairpin_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        directions=directions,
        magnitudes=args.magnitudes,
        intervention_configs=args.intervention_configs,
        charge_modes=args.charge_modes,
        block_sets=block_sets,
        output_dir=args.output,
        save_every=args.save_every,
    )
    
    if len(results_df) > 0:
        analyze_disruption_results(results_df, args.output)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()