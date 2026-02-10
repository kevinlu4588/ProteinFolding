#!/usr/bin/env python
"""
Charge-Based Hairpin Induction
==============================

Induces hairpin formation by adding "opposite charge" signals to the sequence
representation (s) at positions that would form cross-strand pairs in a hairpin.

Hypothesis: ESMFold uses residue charge information when determining strand-strand
contacts. By adding complementary charge signals (positive on one strand, negative
on the other) to helical sequences, we can induce hairpin formation.

Requires pre-trained DoM (Difference of Means) charge directions, which can be
generated using charge_dom_training.py.

Key finding: Interventions in early blocks (0-10) are most effective, consistent
with the patching experiments showing early structure determination.

Usage:
    python charge_steering.py \
        --directions charge_directions/charge_directions.pt \
        --target_loops_dataset data/target_loops_dataset.csv \
        --output results/ \
        --window_sizes 3 5 10 \
        --magnitudes 1 2 3
"""

import argparse
import os
import sys
import types

# Add project root (parent of src/) to path so `src.*` imports work without PYTHONPATH
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
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

from src.charge_dom_training import (
    ChargeDirections, load_directions, save_directions,
    train_dom_vectors, evaluate_dom_vectors,
    plot_dom_evaluation, plot_projection_histograms,
    AA_TO_IDX,
)

import warnings
warnings.filterwarnings("ignore", message=".*mmCIF.*")



# ============================================================================
# CONSTANTS
# ============================================================================

NUM_BLOCKS = 48


@dataclass
class HairpinTopology:
    """Defines the topology of a hypothetical hairpin."""
    strand1_start: int
    strand1_end: int
    turn_start: int
    turn_end: int
    strand2_start: int
    strand2_end: int
    cross_strand_pairs: List[Tuple[int, int]]

def save_structure_as_pdb(outputs: EsmForProteinFoldingOutput, model, path: str):
    """Save structure as PDB."""
    clean = {
        "positions": outputs.positions.detach(),
        "aatype": outputs.aatype.detach(),
        "atom14_atom_exists": outputs.atom14_atom_exists.detach(),
        "residue_index": outputs.residue_index.detach(),
        "plddt": outputs.plddt.detach(),
        "atom37_atom_exists": outputs.atom37_atom_exists.detach(),
        "residx_atom14_to_atom37": outputs.residx_atom14_to_atom37.detach(),
        "residx_atom37_to_atom14": outputs.residx_atom37_to_atom14.detach(),
    }
    pdb_string = model.output_to_pdb(clean)[0]
    with open(path, 'w') as f:
        f.write(pdb_string)
# ============================================================================
# MODEL FORWARD PASSES
# ============================================================================

def baseline_forward_trunk(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """Standard forward pass collecting sequence representations."""
    device = seq_feats.device
    s_s_0, s_z_0 = seq_feats, pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        no_recycles = no_recycles + 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        
        s_list = []
        for block in self.blocks:
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            s_list.append(s.detach().clone())
        return s, z, s_list

    s_s, s_z = s_s_0, s_z_0
    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)
            
            s_s, s_z, s_list = trunk_iter(
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
    """Forward pass with s interventions at specified blocks."""
    device = seq_feats.device
    s_s_0, s_z_0 = seq_feats, pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        no_recycles = no_recycles + 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        
        for block_idx, block in enumerate(self.blocks):
            # Apply s intervention before the block processes
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
    """Forward pass with s interventions."""
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


def define_hairpin_topology(
    region_start: int,
    region_end: int,
    turn_length: int = 4,
) -> HairpinTopology:
    """Define a hypothetical hairpin topology for a region."""
    region_length = region_end - region_start
    strand_length = (region_length - turn_length) // 2
    
    if strand_length < 3:
        strand_length = 3
        turn_length = max(2, region_length - 2 * strand_length)
    
    strand1_start = region_start
    strand1_end = region_start + strand_length
    turn_start = strand1_end
    turn_end = turn_start + turn_length
    strand2_start = turn_end
    strand2_end = min(strand2_start + strand_length, region_end)
    
    cross_strand_pairs = []
    actual_strand2_len = strand2_end - strand2_start
    actual_strand1_len = strand1_end - strand1_start
    
    for i in range(min(actual_strand1_len, actual_strand2_len)):
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


def get_sliding_window_block_sets(
    window_size: int,
    total_blocks: int = NUM_BLOCKS,
) -> List[Tuple[str, set]]:
    """
    Generate sliding window block sets.
    
    Each window covers `window_size` consecutive blocks, sliding from
    block 0 to block (total_blocks - window_size).
    
    Args:
        window_size: Number of consecutive blocks in each window
        total_blocks: Total number of blocks (default 48)
    
    Returns:
        List of (name, block_set) tuples
    """
    block_sets = []
    
    for start_block in range(total_blocks - window_size + 1):
        end_block = start_block + window_size - 1
        name = f'w{window_size}_blocks_{start_block}_to_{end_block}'
        blocks = set(range(start_block, start_block + window_size))
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
    
    return output


def run_intervention(
    model, seq: str, tokenizer, device: str,
    intervention_blocks: set,
    s_interventions: Optional[Dict[int, torch.Tensor]] = None,
) -> EsmForProteinFoldingOutput:
    """Run forward pass with s intervention."""
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
# INTERVENTION CONSTRUCTION
# ============================================================================

def create_hairpin_induction_interventions(
    topology: HairpinTopology,
    seq_len: int,
    directions: ChargeDirections,
    blocks: List[int],
    magnitude: float,
    device: str,
    polarity: str = 'pos_neg',
) -> Dict[int, torch.Tensor]:
    """
    Create s interventions to induce hairpin formation.
    
    Args:
        polarity: 'pos_neg' = positive charge on strand1, negative on strand2 (original)
                  'neg_pos' = negative charge on strand1, positive on strand2 (reversed)
    """
    s_interventions = {}
    
    if polarity == 'pos_neg':
        strand1_sign = +1.0
        strand2_sign = -1.0
    elif polarity == 'neg_pos':
        strand1_sign = -1.0
        strand2_sign = +1.0
    else:
        raise ValueError(f"Unknown polarity: {polarity}. Use 'pos_neg' or 'neg_pos'.")
    
    for block in blocks:
        if block not in directions.s_directions:
            continue
        
        s_dir = directions.s_directions[block]
        s_std = directions.s_stds[block]
        
        s_int = torch.zeros(seq_len, len(s_dir), dtype=torch.float32, device=device)
        delta = magnitude * s_std * s_dir
        
        # Apply charge to strand1
        for i in range(topology.strand1_start, topology.strand1_end):
            s_int[i] = torch.tensor(strand1_sign * delta, device=device)
        
        # Apply charge to strand2
        for j in range(topology.strand2_start, topology.strand2_end):
            s_int[j] = torch.tensor(strand2_sign * delta, device=device)
        
        s_interventions[block] = s_int
    
    return s_interventions


# ============================================================================
# EVALUATION
# ============================================================================

def compute_rg_from_positions(outputs: EsmForProteinFoldingOutput) -> float:
    """
    Compute radius of gyration from Cα atoms in ESMFold output.

    Args:
        outputs: ESMFold output with positions tensor

    Returns:
        Radius of gyration in Angstroms, or NaN if insufficient atoms
    """
    CA_IDX = 1
    positions = outputs.positions[-1, 0].cpu()  # [L, 14, 3]
    ca_coords = positions[:, CA_IDX, :]  # [L, 3]

    # Filter out zero/missing coordinates
    valid_mask = ca_coords.norm(dim=-1) > 0.1
    ca_coords = ca_coords[valid_mask]

    if len(ca_coords) < 3:
        return float('nan')

    center = ca_coords.mean(dim=0)
    rg = torch.sqrt(torch.mean(torch.sum((ca_coords - center) ** 2, dim=-1))).item()
    return rg


def compute_cross_strand_distances(
    outputs: EsmForProteinFoldingOutput,
    topology: HairpinTopology,
) -> Dict[str, float]:
    """Compute distances for cross-strand pairs."""
    positions = outputs.positions[-1, 0].cpu()
    distances = compute_cb_distances(positions)
    
    cross_strand_dists = []
    for i, j in topology.cross_strand_pairs:
        cross_strand_dists.append(distances[i, j].item())
    
    return {
        'mean_cross_strand_dist': np.mean(cross_strand_dists) if cross_strand_dists else None,
        'min_cross_strand_dist': np.min(cross_strand_dists) if cross_strand_dists else None,
        'max_cross_strand_dist': np.max(cross_strand_dists) if cross_strand_dists else None,
        'n_contacts': sum(1 for d in cross_strand_dists if d < 8.0),
        'n_pairs': len(cross_strand_dists),
    }


def compute_backbone_hbonds(
    outputs: EsmForProteinFoldingOutput,
    topology: HairpinTopology,
    hbond_dist_cutoff: float = 3.5,
) -> Dict[str, Any]:
    """
    Compute backbone hydrogen bonds between cross-strand residue pairs.
    
    In antiparallel beta sheets, H-bonds form between:
    - N-H of residue i on strand1 and C=O of residue j on strand2
    - C=O of residue i on strand1 and N-H of residue j on strand2
    
    Args:
        outputs: ESMFold output with positions
        topology: HairpinTopology defining cross-strand pairs
        hbond_dist_cutoff: Maximum N-O distance for H-bond (default 3.5Å)
    
    Returns:
        Dict with H-bond metrics
    """
    N_IDX = 0   # Backbone nitrogen
    CA_IDX = 1  # Alpha carbon
    C_IDX = 2   # Backbone carbonyl carbon
    O_IDX = 3   # Backbone carbonyl oxygen
    
    positions = outputs.positions[-1, 0].cpu()
    
    hbond_count = 0
    hbond_pairs = []
    no_distances = []
    
    for i, j in topology.cross_strand_pairs:
        n_i = positions[i, N_IDX]
        o_i = positions[i, O_IDX] if positions[i, O_IDX].norm() > 0.1 else positions[i, C_IDX]
        n_j = positions[j, N_IDX]
        o_j = positions[j, O_IDX] if positions[j, O_IDX].norm() > 0.1 else positions[j, C_IDX]
        
        dist_ni_oj = torch.norm(n_i - o_j).item()
        dist_nj_oi = torch.norm(n_j - o_i).item()
        
        no_distances.append(dist_ni_oj)
        no_distances.append(dist_nj_oi)
        
        if dist_ni_oj < hbond_dist_cutoff:
            hbond_count += 1
            hbond_pairs.append((i, j, 'N_i-O_j', dist_ni_oj))
        if dist_nj_oi < hbond_dist_cutoff:
            hbond_count += 1
            hbond_pairs.append((j, i, 'N_j-O_i', dist_nj_oi))
    
    max_possible_hbonds = 2 * len(topology.cross_strand_pairs)
    hbond_fraction = hbond_count / max_possible_hbonds if max_possible_hbonds > 0 else 0
    
    return {
        'n_hbonds': hbond_count,
        'max_possible_hbonds': max_possible_hbonds,
        'hbond_fraction': hbond_fraction,
        'hbond_percentage': hbond_fraction * 100,
        'mean_no_distance': np.mean(no_distances) if no_distances else None,
        'min_no_distance': np.min(no_distances) if no_distances else None,
        'hbond_pairs': hbond_pairs,
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
            analyze_results(df, self.output_dir)
        except Exception as e:
            print(f"[Warning] Could not update plots: {e}")
    
    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_hairpin_induction_experiment(
    cases_df: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    directions: ChargeDirections,
    magnitudes: List[float],
    block_sets: List[Tuple[str, set]],
    output_dir: str,
    save_every: int = 10,
    save_pdbs: bool = True,
    rg_ratio_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Run hairpin induction experiments with sliding window interventions.
    
    Updated to work with target_loops_dataset.csv which has columns:
    - target_name, target_sequence, target_length
    - loop_idx, loop_start, loop_end, loop_length, loop_sequence
    - target_patch_start, target_patch_end, patch_length
    
    No donor information is used - we're just testing if charge steering
    can induce hairpin formation in the target regions.
    """
    results_manager = ResultsManager(
        output_path=os.path.join(output_dir, 'steering_results.parquet'),
        output_dir=output_dir,
        save_every=save_every
    )
    
    available_blocks = set(directions.s_directions.keys())
    
    if save_pdbs:
        pdb_dir = os.path.join(output_dir, 'induced_hairpins')
        os.makedirs(pdb_dir, exist_ok=True)
    
    for case_idx, row in enumerate(cases_df.itertuples()):
        # Get target sequence
        target_seq = getattr(row, 'target_sequence', None)
        if target_seq is None:
            tqdm.write(f"Skipping case {case_idx}: no target_sequence")
            continue
        
        # Get target region (patch region around the loop)
        target_start = int(getattr(row, 'target_patch_start', 0))
        target_end = int(getattr(row, 'target_patch_end', len(target_seq)))
        
        # Get metadata
        target_name = getattr(row, 'target_name', f'target_{case_idx}')
        loop_idx = getattr(row, 'loop_idx', 0)
        loop_start = getattr(row, 'loop_start', None)
        loop_end = getattr(row, 'loop_end', None)
        loop_sequence = getattr(row, 'loop_sequence', '')
        
        L = len(target_seq)
        case_results = []
        
        # Define hairpin topology based on patch region
        topology = define_hairpin_topology(target_start, target_end, turn_length=4)
        
        tqdm.write(f"\nCase {case_idx}/{len(cases_df)}: {target_name} (loop {loop_idx})")
        tqdm.write(f"  Target length: {L}")
        tqdm.write(f"  Patch region: {target_start}-{target_end}")
        if loop_start is not None and loop_end is not None:
            tqdm.write(f"  Loop region: {loop_start}-{loop_end}")
        tqdm.write(f"  Topology: strand1={topology.strand1_start}-{topology.strand1_end}, "
                   f"turn={topology.turn_start}-{topology.turn_end}, "
                   f"strand2={topology.strand2_start}-{topology.strand2_end}")
        tqdm.write(f"  Cross-strand pairs: {len(topology.cross_strand_pairs)}")
        
        # Get baseline structure
        try:
            baseline_outputs = get_baseline_structure(model, target_seq, tokenizer, device)
        except Exception as e:
            tqdm.write(f"  Baseline error: {e}")
            continue
        
        # Evaluate baseline
        baseline_dists = compute_cross_strand_distances(baseline_outputs, topology)
        baseline_hbonds = compute_backbone_hbonds(baseline_outputs, topology)
        baseline_rg = compute_rg_from_positions(baseline_outputs)

        tqdm.write(f"  Baseline: "
                   f"mean_dist={baseline_dists['mean_cross_strand_dist']:.1f}Å, "
                   f"hbonds={baseline_hbonds['n_hbonds']}/{baseline_hbonds['max_possible_hbonds']}, "
                   f"Rg={baseline_rg:.1f}Å")
        
        # Record baseline result
        case_results.append({
            'case_idx': case_idx,
            'target_name': target_name,
            'target_sequence': target_seq,
            'target_length': L,
            'loop_idx': loop_idx,
            'loop_start': loop_start,
            'loop_end': loop_end,
            'loop_sequence': loop_sequence,
            'region_start': target_start,
            'region_end': target_end,
            'block_set': 'baseline',
            'window_size': 0,
            'window_start': -1,
            'window_end': -1,
            'n_blocks': 0,
            'magnitude': 0.0,
            'polarity': 'baseline',
            'mean_cross_strand_dist': baseline_dists['mean_cross_strand_dist'],
            'min_cross_strand_dist': baseline_dists['min_cross_strand_dist'],
            'n_contacts': baseline_dists['n_contacts'],
            'n_cross_strand_pairs': baseline_dists['n_pairs'],
            'n_hbonds': baseline_hbonds['n_hbonds'],
            'hbond_percentage': baseline_hbonds['hbond_percentage'],
            'mean_no_distance': baseline_hbonds['mean_no_distance'],
            'rg': baseline_rg,
            'baseline_rg': baseline_rg,
            'rg_ratio': 1.0,
            'rg_filtered': False,
        })
        
        # Precompute valid block sets
        valid_block_sets = []
        for block_set_name, intervention_blocks in block_sets:
            ib = intervention_blocks & available_blocks
            if len(ib) > 0:
                valid_block_sets.append((block_set_name, ib))
        
        n_total = len(valid_block_sets) * len(magnitudes) * 2  # x2 for both polarities
        pbar = tqdm(total=n_total, desc=f"Case {case_idx}", leave=False)
        
        polarities = ['pos_neg', 'neg_pos']
        
        for block_set_name, intervention_blocks in valid_block_sets:
            # Extract window info from name (format: w{size}_blocks_{start}_to_{end})
            parts = block_set_name.split('_')
            window_size = int(parts[0][1:])  # Remove 'w' prefix
            window_start = int(parts[2])
            window_end = int(parts[4])
            
            for magnitude in magnitudes:
                for polarity in polarities:
                    pbar.set_postfix_str(f"w={window_size} start={window_start} mag={magnitude} pol={polarity}")
                    
                    # Create interventions
                    s_ints = create_hairpin_induction_interventions(
                        topology=topology,
                        seq_len=L,
                        directions=directions,
                        blocks=list(intervention_blocks),
                        magnitude=magnitude,
                        device=device,
                        polarity=polarity,
                    )
                    
                    try:
                        int_outputs = run_intervention(
                            model, target_seq, tokenizer, device,
                            intervention_blocks=intervention_blocks,
                            s_interventions=s_ints,
                        )
                    except Exception as e:
                        tqdm.write(f"  Intervention error ({block_set_name}, mag={magnitude}, pol={polarity}): {e}")
                        pbar.update(1)
                        continue
                    
                    # Evaluate intervention
                    int_dists = compute_cross_strand_distances(int_outputs, topology)
                    int_hbonds = compute_backbone_hbonds(int_outputs, topology)
                    int_rg = compute_rg_from_positions(int_outputs)

                    # Compute Rg ratio and apply filter
                    rg_ratio = int_rg / baseline_rg if (baseline_rg and not np.isnan(baseline_rg) and baseline_rg > 0) else float('nan')
                    rg_filtered = (np.isnan(rg_ratio)) or (rg_ratio < rg_ratio_threshold)

                    # Compute changes from baseline
                    dist_change = None
                    if baseline_dists['mean_cross_strand_dist'] and int_dists['mean_cross_strand_dist']:
                        dist_change = int_dists['mean_cross_strand_dist'] - baseline_dists['mean_cross_strand_dist']

                    hbond_change = int_hbonds['n_hbonds'] - baseline_hbonds['n_hbonds']

                    # Zero out hbond results if Rg filter fails (structure collapsed)
                    reported_hbond_percentage = 0.0 if rg_filtered else int_hbonds['hbond_percentage']
                    reported_n_hbonds = 0 if rg_filtered else int_hbonds['n_hbonds']
                    reported_hbond_change = reported_n_hbonds - baseline_hbonds['n_hbonds']

                    case_results.append({
                        'case_idx': case_idx,
                        'target_name': target_name,
                        'target_sequence': target_seq,
                        'target_length': L,
                        'loop_idx': loop_idx,
                        'loop_start': loop_start,
                        'loop_end': loop_end,
                        'loop_sequence': loop_sequence,
                        'region_start': target_start,
                        'region_end': target_end,
                        'block_set': block_set_name,
                        'window_size': window_size,
                        'window_start': window_start,
                        'window_end': window_end,
                        'n_blocks': len(intervention_blocks),
                        'magnitude': magnitude,
                        'polarity': polarity,
                        'mean_cross_strand_dist': int_dists['mean_cross_strand_dist'],
                        'min_cross_strand_dist': int_dists['min_cross_strand_dist'],
                        'n_contacts': int_dists['n_contacts'],
                        'n_cross_strand_pairs': int_dists['n_pairs'],
                        'dist_change': dist_change,
                        'n_hbonds': reported_n_hbonds,
                        'hbond_percentage': reported_hbond_percentage,
                        'hbond_percentage_raw': int_hbonds['hbond_percentage'],
                        'mean_no_distance': int_hbonds['mean_no_distance'],
                        'hbond_change': reported_hbond_change,
                        'rg': int_rg,
                        'baseline_rg': baseline_rg,
                        'rg_ratio': rg_ratio,
                        'rg_filtered': rg_filtered,
                    })
                    # Save PDB if H-bonds were gained (use raw, unfiltered count)
                    if hbond_change > 0:
                        if save_pdbs:
                            pdb_path = os.path.join(
                                pdb_dir,
                                f'case{case_idx}_{target_name}_loop{loop_idx}_{block_set_name}_mag{magnitude:.0f}_{polarity}.pdb'
                            )
                            save_structure_as_pdb(int_outputs, model, pdb_path)

                    if hbond_change > 0:
                        filtered_tag = " [Rg-FILTERED]" if rg_filtered else ""
                        tqdm.write(
                            f"      + H-BONDS INDUCED @ window_start={window_start}, "
                            f"mag={magnitude}, pol={polarity}: +{hbond_change} "
                            f"(baseline={baseline_hbonds['n_hbonds']} → {int_hbonds['n_hbonds']}) | "
                            f"mean N–O dist={int_hbonds['mean_no_distance']:.2f}Å | "
                            f"Rg ratio={rg_ratio:.2f}{filtered_tag}"
                        )

                        # Optional: show which residue pairs formed H-bonds
                        for i, j, bond_type, dist in int_hbonds['hbond_pairs']:
                            tqdm.write(
                                f"        {bond_type} between residues {i}-{j}: {dist:.2f}Å"
                            )
                
                    del s_ints, int_outputs
                    torch.cuda.empty_cache()
                    
                    pbar.update(1)
        
        pbar.close()
        
        results_manager.add_results(case_results)
        results_manager.mark_case_done()
        
        del baseline_outputs
        torch.cuda.empty_cache()
    
    results_manager.save()
    return results_manager.get_dataframe()



# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_results(results_df: pd.DataFrame, output_dir: str):
    """Analyze and visualize Rg-filtered results."""
    print("\n" + "="*60)
    print("CHARGE STEERING RESULTS (Rg-filtered)")
    print("="*60)

    baseline = results_df[results_df['block_set'] == 'baseline']
    interventions = results_df[results_df['block_set'] != 'baseline']

    print(f"\nBaseline: {len(baseline)} cases")
    print(f"Interventions: {len(interventions)} total")

    if 'rg_filtered' in interventions.columns:
        n_filtered = interventions['rg_filtered'].sum()
        print(f"  Rg-filtered (zeroed hbond): {n_filtered}/{len(interventions)} "
              f"({100*n_filtered/len(interventions):.1f}%)")

    # All plots use the already-filtered hbond_percentage column
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())

    if len(window_sizes) == 0:
        print("No intervention data to plot yet.")
        return

    polarities = sorted([p for p in interventions['polarity'].unique()])

    # Color maps
    n_window_sizes = len(window_sizes)
    n_magnitudes = len(magnitudes)
    ws_colors = {ws: plt.cm.viridis(i / max(1, n_window_sizes - 1) * 0.8)
                 for i, ws in enumerate(window_sizes)}
    mag_colors = {mag: plt.cm.plasma(i / max(1, n_magnitudes - 1) * 0.8)
                  for i, mag in enumerate(magnitudes)}

    polarity_labels = {'pos_neg': '+strand1 / −strand2', 'neg_pos': '−strand1 / +strand2'}
    polarity_linestyles = {'pos_neg': '-', 'neg_pos': '--'}

    # =========================================================================
    # Single summary plot with Rg-filtered data
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top-left: H-bond formation (Rg-filtered) by window start ---
    ax = axes[0, 0]
    if 'hbond_percentage' in interventions.columns:
        for pol in polarities:
            for ws in window_sizes:
                ws_pol_data = interventions[
                    (interventions['window_size'] == ws) & (interventions['polarity'] == pol)
                ]
                window_starts = sorted(ws_pol_data['window_start'].unique())

                means, sems, valid_starts = [], [], []
                for wstart in window_starts:
                    subset = ws_pol_data[ws_pol_data['window_start'] == wstart]
                    hbonds = subset['hbond_percentage'].dropna()
                    if len(hbonds) > 0:
                        means.append(hbonds.mean())
                        sems.append(hbonds.sem() if len(hbonds) > 1 else 0)
                        valid_starts.append(wstart)

                if len(means) > 0:
                    ax.fill_between(valid_starts, means, color=ws_colors[ws], alpha=0.15)
                    ax.errorbar(valid_starts, means, yerr=sems,
                               fmt='o', linestyle=polarity_linestyles[pol],
                               label=f'w={ws} {polarity_labels[pol]}', color=ws_colors[ws],
                               linewidth=2, markersize=4, capsize=2,
                               alpha=0.7 if pol == 'neg_pos' else 1.0)

        if 'hbond_percentage' in baseline.columns:
            baseline_mean = baseline['hbond_percentage'].mean()
            ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5,
                      label=f'baseline ({baseline_mean:.1f}%)')

    ax.set_xlabel('Window Start Block', fontsize=12)
    ax.set_ylabel('H-bond Formation (%, Rg-filtered)', fontsize=12)
    ax.set_title('H-bond Formation by Window Position', fontsize=13)
    ax.legend(loc='best', fontsize=7)
    ax.grid(alpha=0.3)

    # --- Top-right: Mean N-O distance by window start ---
    ax = axes[0, 1]
    if 'mean_no_distance' in interventions.columns:
        for pol in polarities:
            for ws in window_sizes:
                ws_pol_data = interventions[
                    (interventions['window_size'] == ws) & (interventions['polarity'] == pol)
                ]
                window_starts = sorted(ws_pol_data['window_start'].unique())

                means, valid_starts = [], []
                for wstart in window_starts:
                    subset = ws_pol_data[ws_pol_data['window_start'] == wstart]
                    vals = subset['mean_no_distance'].dropna()
                    if len(vals) > 0:
                        means.append(vals.mean())
                        valid_starts.append(wstart)

                if len(means) > 0:
                    ax.plot(valid_starts, means, linestyle=polarity_linestyles[pol],
                            marker='o', label=f'w={ws} {polarity_labels[pol]}', color=ws_colors[ws],
                            linewidth=2, markersize=4, alpha=0.7 if pol == 'neg_pos' else 1.0)

        if 'mean_no_distance' in baseline.columns:
            baseline_mean = baseline['mean_no_distance'].mean()
            ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5,
                      label=f'baseline ({baseline_mean:.1f}Å)')

    ax.set_xlabel('Window Start Block', fontsize=12)
    ax.set_ylabel('Mean N-O Distance (Å)', fontsize=12)
    ax.set_title('Mean N-O Distance by Window Position', fontsize=13)
    ax.legend(loc='best', fontsize=7)
    ax.grid(alpha=0.3)

    # --- Bottom-left: Distance change by window start ---
    ax = axes[1, 0]
    for pol in polarities:
        for ws in window_sizes:
            ws_pol_data = interventions[
                (interventions['window_size'] == ws) & (interventions['polarity'] == pol)
            ]
            window_starts = sorted(ws_pol_data['window_start'].unique())

            means, stds, valid_starts = [], [], []
            for wstart in window_starts:
                subset = ws_pol_data[ws_pol_data['window_start'] == wstart]
                if 'dist_change' in subset.columns:
                    dist_changes = subset['dist_change'].dropna()
                    if len(dist_changes) > 0:
                        means.append(dist_changes.mean())
                        stds.append(dist_changes.std() / np.sqrt(len(dist_changes)) if len(dist_changes) > 1 else 0)
                        valid_starts.append(wstart)

            if len(means) > 0:
                ax.errorbar(valid_starts, means, yerr=stds,
                           fmt='o', linestyle=polarity_linestyles[pol],
                           label=f'w={ws} {polarity_labels[pol]}',
                           color=ws_colors[ws], linewidth=2, markersize=4, capsize=2,
                           alpha=0.7 if pol == 'neg_pos' else 1.0)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Window Start Block', fontsize=12)
    ax.set_ylabel('Mean Distance Change (Å)', fontsize=12)
    ax.set_title('Distance Change by Window Position', fontsize=13)
    ax.legend(loc='best', fontsize=7)
    ax.grid(alpha=0.3)

    # --- Bottom-right: H-bond % (Rg-filtered) by magnitude ---
    ax = axes[1, 1]
    if 'hbond_percentage' in interventions.columns:
        for pol in polarities:
            for ws in window_sizes:
                means = []
                for mag in magnitudes:
                    subset = interventions[
                        (interventions['window_size'] == ws) &
                        (interventions['magnitude'] == mag) &
                        (interventions['polarity'] == pol)
                    ]
                    mean_hbond = subset['hbond_percentage'].mean() if len(subset) > 0 else 0
                    means.append(mean_hbond)

                ax.plot(magnitudes, means, linestyle=polarity_linestyles[pol],
                        marker='o', label=f'w={ws} {polarity_labels[pol]}', color=ws_colors[ws],
                        linewidth=2, markersize=5, alpha=0.7 if pol == 'neg_pos' else 1.0)

        if 'hbond_percentage' in baseline.columns:
            baseline_mean = baseline['hbond_percentage'].mean()
            ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5,
                      label=f'baseline ({baseline_mean:.1f}%)')

    ax.set_xlabel('Magnitude (std devs)', fontsize=12)
    ax.set_ylabel('H-bond Formation (%, Rg-filtered)', fontsize=12)
    ax.set_title('H-bond % by Magnitude', fontsize=13)
    ax.legend(loc='best', fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steering_summary.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probing_dataset', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'probing_train_test.csv'),
                        help='Path to probing dataset CSV (required if --directions_path not provided)')
    parser.add_argument('--target_loops_dataset', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'target_loops_dataset.csv'),
                        help='Path to target loops dataset CSV for interventions')
    parser.add_argument('--output', type=str, default=os.path.join(_PROJECT_ROOT, 'results', 'charge_steering'))
    parser.add_argument('--directions_path', type=str, default=None,
                        help='Path to pre-trained DoM directions. If not provided, trains from --probing_dataset')
    parser.add_argument('--n_cases', type=int, default=None,
                        help='Number of target/loop cases to run')
    parser.add_argument('--magnitudes', type=float, nargs='+',
                        default=[3.0])
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[15],
                        help='List of window sizes to test (consecutive blocks per window)')
    parser.add_argument('--rg_ratio_threshold', type=float, default=0.90,
                        help='Rg ratio threshold: if intervention Rg / baseline Rg < this, '
                             'zero out hbond results as fake (default 0.90)')
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

    # Blocks to train/evaluate
    blocks_to_use = list(range(0, 48))

    # =========================================================================
    # PHASE 1: Load or Train DoM Vectors
    # =========================================================================
    if args.directions_path and os.path.exists(args.directions_path):
        print(f"\nLoading DoM directions from {args.directions_path}...")
        directions = load_directions(args.directions_path)
        print(f"Loaded directions for {len(directions.s_directions)} blocks")
    else:
        if args.probing_dataset is None:
            raise ValueError("Must provide either --directions_path (to a valid file) or --probing_dataset (to train new directions)")

        directions_save_path = args.directions_path or os.path.join(args.output, 'charge_directions.pt')

        print("\n" + "="*60)
        print("PHASE 1: TRAIN DoM VECTORS")
        print("="*60)

        # Load probing dataset
        print(f"\nLoading probing dataset from {args.probing_dataset}...")
        probing_df = pd.read_csv(args.probing_dataset)

        # Show dataset composition
        print(f"\nDataset composition:")
        if 'split' in probing_df.columns and 'structure_type' in probing_df.columns:
            print(probing_df.groupby(['split', 'structure_type']).size().unstack(fill_value=0))

        # =====================================================================
        # TRAINING: Use only alpha_helical sequences from train split
        # =====================================================================
        train_df = probing_df[
            (probing_df['split'] == 'train') &
            (probing_df['structure_type'] == 'alpha_helical')
        ]

        # Get sequences - handle both possible column names
        seq_col = 'FullChainSequence' if 'FullChainSequence' in train_df.columns else 'sequence'
        train_seqs = train_df[seq_col].tolist()
        train_seqs = [s for s in train_seqs if all(aa in AA_TO_IDX for aa in s)]

        print(f"\nTrain sequences (alpha_helical only): {len(train_seqs)}")

        import random
        random.seed(args.seed)
        random.shuffle(train_seqs)

        # Train DoM vectors
        directions = train_dom_vectors(
            sequences=train_seqs,
            model=model,
            tokenizer=tokenizer,
            device=device,
            blocks_to_train=blocks_to_use,
        )

        # Save directions
        save_directions(directions, directions_save_path)

        # =====================================================================
        # EVALUATION: Use entire test split
        # =====================================================================
        print("\n" + "="*60)
        print("PHASE 1b: EVALUATE DoM VECTORS")
        print("="*60)

        test_df = probing_df[probing_df['split'] == 'test']
        seq_col = 'FullChainSequence' if 'FullChainSequence' in test_df.columns else 'sequence'
        test_seqs = test_df[seq_col].tolist()
        test_seqs = [s for s in test_seqs if all(aa in AA_TO_IDX for aa in s)]

        print(f"\nTest sequences: {len(test_seqs)}")

        # Evaluate
        eval_df, projections_data = evaluate_dom_vectors(
            directions=directions,
            sequences=test_seqs,
            model=model,
            tokenizer=tokenizer,
            device=device,
            blocks_to_eval=blocks_to_use,
        )

        # Save evaluation results
        eval_df.to_csv(os.path.join(args.output, 'dom_evaluation.csv'), index=False)

        # Plot separation scores
        plot_dom_evaluation(eval_df, args.output)

        # Plot centered histograms for selected blocks
        blocks_to_show = [0, 8, 16, 24, 32, 40, 47]
        blocks_to_show = [b for b in blocks_to_show if b in blocks_to_use]
        plot_projection_histograms(projections_data, blocks_to_show, args.output)

        print("\nDoM Evaluation Results (Separation Scores):")
        print(eval_df.to_string(index=False))
    
    # =========================================================================
    # PHASE 2: Run Hairpin Induction Experiments
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: HAIRPIN INDUCTION EXPERIMENTS")
    print("="*60)
    
    # Load target loops dataset
    print(f"\nLoading target loops dataset from {args.target_loops_dataset}...")
    loops_df = pd.read_csv(args.target_loops_dataset)
    print(f"Total target/loop combinations: {len(loops_df)}")
    
    # Show dataset stats
    if 'target_name' in loops_df.columns:
        n_targets = loops_df['target_name'].nunique()
        print(f"Unique targets: {n_targets}")
    
    if 'loop_length' in loops_df.columns:
        print(f"Loop length range: {loops_df['loop_length'].min()}-{loops_df['loop_length'].max()}")
    
    # Limit cases if specified
    if args.n_cases is not None:
        loops_df = loops_df.head(args.n_cases)
        print(f"Limited to {len(loops_df)} cases")
    
    loops_df = loops_df.reset_index(drop=True)
    
    if len(loops_df) == 0:
        print("No cases to run!")
        return
    
    # Set up sliding window block sets for all window sizes
    all_block_sets = []
    for window_size in args.window_sizes:
        block_sets = get_sliding_window_block_sets(
            window_size=window_size,
            total_blocks=NUM_BLOCKS,
        )
        all_block_sets.extend(block_sets)
    
    print(f"\nWindow sizes: {args.window_sizes}")
    print(f"Total number of windows across all sizes: {len(all_block_sets)}")
    print(f"Magnitudes: {args.magnitudes}")
    
    # Run experiments
    print(f"Rg ratio threshold: {args.rg_ratio_threshold}")
    results_df = run_hairpin_induction_experiment(
        cases_df=loops_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        directions=directions,
        magnitudes=args.magnitudes,
        block_sets=all_block_sets,
        output_dir=args.output,
        save_every=args.save_every,
        rg_ratio_threshold=args.rg_ratio_threshold,
    )
    
    if len(results_df) > 0:
        analyze_results(results_df, args.output)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()