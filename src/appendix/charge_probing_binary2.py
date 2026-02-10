#!/usr/bin/env python
"""
Pairwise Charge Probing Experiment (Early Blocks Sweep Version)
================================================================

Trains linear probes on pairwise representations (z, seq2pair) to predict:
1. Charge of residue i (row residue)
2. Charge of residue j (column residue)  
3. Same vs opposite charge relationship

Analyzes intervention effects across multiple early block windows (0-3, 1-4, ..., 9-12)
regardless of whether hairpin induction succeeded.

Key changes from original:
- Sweeps window_start from 0 to 9 (with window_size=4)
- Does NOT require successful hairpin induction cases
- Runs a fixed number of random cases per window configuration

Usage:
    python pairwise_charge_probing_early_blocks.py \
        --output probing_results_early_blocks/ \
        --n_cases 20 \
        --window_size 4 \
        --magnitude 2.0
"""

import argparse
import os
import types
import pickle
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput
)
from transformers.models.esm.openfold_utils import make_atom14_masks
from transformers.utils import ContextManagers


# ============================================================================
# CONSTANTS
# ============================================================================

AA_CHARGE = {
    'K': +1, 'R': +1, 'H': +1,
    'D': -1, 'E': -1,
    'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0,
    'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}

NUM_BLOCKS = 48


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChargeDirections:
    """Cached charge directions for all blocks."""
    s_directions: Dict[int, np.ndarray]
    seq2pair_directions: Dict[int, np.ndarray]
    z_directions: Dict[int, np.ndarray]
    s_stds: Dict[int, float]
    seq2pair_stds: Dict[int, float]
    z_stds: Dict[int, float]


@dataclass 
class PairwiseProbes:
    """Trained probes for pairwise representations."""
    # Binary probes: positive vs not-positive, negative vs not-negative
    positive_probes: Dict[int, LogisticRegression]  # P(positive) - K,R,H=1, else=0
    negative_probes: Dict[int, LogisticRegression]  # P(negative) - D,E=1, else=0
    positive_accuracies: Dict[int, float]
    negative_accuracies: Dict[int, float]


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


# ============================================================================
# MODEL FORWARD PASSES
# ============================================================================

def baseline_forward_trunk(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """Standard forward pass collecting representations."""
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


def intervention_forward_trunk_s_only(
    self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles,
    intervention_blocks: set,
    s_interventions: Optional[Dict[int, torch.Tensor]] = None,
):
    """Forward pass with s-only interventions, returning intermediate representations."""
    device = seq_feats.device
    s_s_0, s_z_0 = seq_feats, pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        no_recycles = no_recycles + 1

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        
        s_list, z_list, seq2pair_list = [], [], []
        for block_idx, block in enumerate(self.blocks):
            # Apply s intervention before block processes it
            if block_idx in intervention_blocks and s_interventions is not None:
                if block_idx in s_interventions:
                    s = s + s_interventions[block_idx].unsqueeze(0)
            
            # Capture seq2pair AFTER intervention is applied to s
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


def intervention_forward_s_only(
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


def get_baseline_structure(model, seq: str, tokenizer, device: str):
    """Run baseline forward pass and return outputs with representation lists."""
    model.forward = types.MethodType(baseline_forward, model)
    model.trunk.forward = types.MethodType(baseline_forward_trunk, model.trunk)
    
    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(**inputs, num_recycles=0)
    
    return outputs


def get_intervened_structure(
    model, seq: str, tokenizer, device: str,
    intervention_blocks: set,
    s_interventions: Dict[int, torch.Tensor],
):
    """Run forward pass with s intervention and return outputs with representation lists."""
    model.forward = types.MethodType(intervention_forward_s_only, model)
    model.trunk.forward = types.MethodType(intervention_forward_trunk_s_only, model.trunk)
    
    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(
            **inputs, num_recycles=0,
            intervention_blocks=intervention_blocks,
            s_interventions=s_interventions,
        )
    
    return outputs


# ============================================================================
# DIRECTION LOADING
# ============================================================================

def load_directions(path: str) -> ChargeDirections:
    """Load DoM directions from file. Handles both old and new formats."""
    data = torch.load(path, weights_only=False)
    
    # Handle new format (s_directions only) vs old format (with z_directions)
    s_directions = data.get('s_directions', {})
    s_stds = data.get('s_stds', {})
    
    # For new format, z_directions might not exist
    z_directions = data.get('z_directions', {})
    z_stds = data.get('z_stds', {})
    seq2pair_directions = data.get('seq2pair_directions', {})
    seq2pair_stds = data.get('seq2pair_stds', {})
    
    return ChargeDirections(
        s_directions=s_directions,
        seq2pair_directions=seq2pair_directions,
        z_directions=z_directions,
        s_stds=s_stds,
        seq2pair_stds=seq2pair_stds,
        z_stds=z_stds,
    )


# ============================================================================
# PROBE SAVING/LOADING
# ============================================================================

def save_probes(probes: PairwiseProbes, path: str, rep_type: str):
    """Save trained probes to disk."""
    save_path = os.path.join(path, f'probes_{rep_type}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'positive_probes': probes.positive_probes,
            'negative_probes': probes.negative_probes,
            'positive_accuracies': probes.positive_accuracies,
            'negative_accuracies': probes.negative_accuracies,
        }, f)
    print(f"Saved probes to {save_path}")


def load_probes(path: str, rep_type: str) -> Optional[PairwiseProbes]:
    """Load trained probes from disk."""
    load_path = os.path.join(path, f'probes_{rep_type}.pkl')
    if not os.path.exists(load_path):
        return None
    
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded probes from {load_path}")
    return PairwiseProbes(
        positive_probes=data['positive_probes'],
        negative_probes=data['negative_probes'],
        positive_accuracies=data['positive_accuracies'],
        negative_accuracies=data['negative_accuracies'],
    )


# ============================================================================
# PROBE TRAINING (kept for reference, but we'll use pre-trained)
# ============================================================================

class ProbeWrapper:
    def __init__(self, linear_layer):
        self.linear = linear_layer
        self.linear.eval()
    
    def predict_proba(self, X):
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(next(self.linear.parameters()).device)
            logits = self.linear(X)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=-1)


# ============================================================================
# INTERVENTION ANALYSIS
# ============================================================================

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


def create_s_interventions(
    topology: HairpinTopology,
    seq_len: int,
    directions: ChargeDirections,
    blocks: List[int],
    magnitude: float,
    device: str,
) -> Dict[int, torch.Tensor]:
    """Create s interventions for hairpin induction."""
    s_interventions = {}
    
    for block in blocks:
        if block not in directions.s_directions:
            continue
        
        s_dir = directions.s_directions[block]
        s_std = directions.s_stds[block]
        
        s_int = torch.zeros(seq_len, len(s_dir), dtype=torch.float32, device=device)
        delta = magnitude * s_std * s_dir
        
        for i in range(topology.strand1_start, topology.strand1_end):
            s_int[i] = torch.tensor(+delta, device=device)
        
        for j in range(topology.strand2_start, topology.strand2_end):
            s_int[j] = torch.tensor(-delta, device=device)
        
        s_interventions[block] = s_int
    
    return s_interventions


def get_random_cases_from_parquet(
    results_parquet: str,
    n_cases: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Get random cases from the parquet file (regardless of success status).
    
    Args:
        results_parquet: Path to hairpin_induction_results.parquet
        n_cases: Number of cases to return
        seed: Random seed
    
    Returns:
        DataFrame with random cases (baseline rows only, for sequence info)
    """
    df = pd.read_parquet(results_parquet)
    
    print(f"Total rows: {len(df)}")
    
    # Get baseline cases (these have all the sequence info)
    baseline = df[df['block_set'] == 'baseline'].copy()
    print(f"Baseline cases: {len(baseline)}")
    
    # Sample random cases
    np.random.seed(seed)
    if len(baseline) > n_cases:
        sampled = baseline.sample(n=n_cases, random_state=seed)
    else:
        sampled = baseline
    
    print(f"Selected {len(sampled)} cases for analysis")
    
    return sampled


def analyze_probe_readouts_single_window(
    cases: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    directions: ChargeDirections,
    probes_z: PairwiseProbes,
    blocks_to_analyze: List[int],
    window_size: int,
    window_start: int,
    magnitude: float,
) -> Dict[str, Any]:
    """
    Analyze probe readouts for a SINGLE window configuration.
    
    Returns dict with per-block probabilities for plotting.
    """
    print(f"\n  Analyzing window {window_start}-{window_start + window_size - 1}...")
    
    # Track per-block probabilities
    results = {
        'baseline': {
            'strand1_pos_prob': {block: [] for block in blocks_to_analyze},
            'strand2_neg_prob': {block: [] for block in blocks_to_analyze},
        },
        'intervened': {
            'strand1_pos_prob': {block: [] for block in blocks_to_analyze},
            'strand2_neg_prob': {block: [] for block in blocks_to_analyze},
        },
        'case_info': [],
    }
    
    intervention_blocks = set(range(window_start, window_start + window_size))
    
    for idx, row in tqdm(cases.iterrows(), total=len(cases), desc=f"  Window {window_start}-{window_start+window_size-1}", leave=False):
        seq = row['target_sequence']
        region_start = int(row['region_start'])
        region_end = int(row['region_end'])
        
        L = len(seq)
        topology = define_hairpin_topology(region_start, region_end, turn_length=4)
        
        # Skip if topology doesn't make sense
        if topology.strand1_end > L or topology.strand2_end > L:
            continue
        
        # Get baseline representations
        try:
            baseline_outputs = get_baseline_structure(model, seq, tokenizer, device)
        except Exception as e:
            print(f"Error on case {idx}: {e}")
            continue
        
        z_list_baseline = baseline_outputs.get("z_list")
        if z_list_baseline is None:
            continue
        
        # Create and apply intervention
        s_interventions = create_s_interventions(
            topology=topology,
            seq_len=L,
            directions=directions,
            blocks=list(intervention_blocks),
            magnitude=magnitude,
            device=device,
        )
        
        try:
            intervened_outputs = get_intervened_structure(
                model, seq, tokenizer, device,
                intervention_blocks=intervention_blocks,
                s_interventions=s_interventions,
            )
        except Exception as e:
            print(f"Intervention error on case {idx}: {e}")
            continue
        
        z_list_intervened = intervened_outputs.get("z_list")
        if z_list_intervened is None:
            continue
        
        # Record case info
        results['case_info'].append({
            'case_idx': row.get('case_idx', idx),
            'sequence': seq,
            'region_start': region_start,
            'region_end': region_end,
            'strand1': f"{topology.strand1_start}-{topology.strand1_end}",
            'strand2': f"{topology.strand2_start}-{topology.strand2_end}",
        })
        
        # Sample j positions for pairwise features
        j_positions = [0, L//4, L//2, 3*L//4, L-1]
        j_positions = [j for j in j_positions if j < L]
        
        # Analyze each block
        for block in blocks_to_analyze:
            if block not in probes_z.positive_probes or block not in probes_z.negative_probes:
                continue
            
            z_baseline = z_list_baseline[block][0].cpu().numpy()
            z_intervened = z_list_intervened[block][0].cpu().numpy()
            
            probe_pos = probes_z.positive_probes[block]
            probe_neg = probes_z.negative_probes[block]
            
            # Strand 1: Track P(positive)
            for i in range(topology.strand1_start, topology.strand1_end):
                for j in j_positions:
                    if j == i:
                        continue
                    
                    feat_base = z_baseline[i, j].reshape(1, -1)
                    prob_pos_base = probe_pos.predict_proba(feat_base)[0][1]
                    results['baseline']['strand1_pos_prob'][block].append(prob_pos_base)
                    
                    feat_int = z_intervened[i, j].reshape(1, -1)
                    prob_pos_int = probe_pos.predict_proba(feat_int)[0][1]
                    results['intervened']['strand1_pos_prob'][block].append(prob_pos_int)
            
            # Strand 2: Track P(negative)
            for i in range(topology.strand2_start, topology.strand2_end):
                for j in j_positions:
                    if j == i:
                        continue
                    
                    feat_base = z_baseline[i, j].reshape(1, -1)
                    prob_neg_base = probe_neg.predict_proba(feat_base)[0][1]
                    results['baseline']['strand2_neg_prob'][block].append(prob_neg_base)
                    
                    feat_int = z_intervened[i, j].reshape(1, -1)
                    prob_neg_int = probe_neg.predict_proba(feat_int)[0][1]
                    results['intervened']['strand2_neg_prob'][block].append(prob_neg_int)
        
        del baseline_outputs, intervened_outputs, z_list_baseline, z_list_intervened, s_interventions
        torch.cuda.empty_cache()
    
    return results


def analyze_all_window_configurations(
    cases: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    directions: ChargeDirections,
    probes_z: PairwiseProbes,
    blocks_to_analyze: List[int],
    window_size: int,
    window_starts: List[int],
    magnitude: float,
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze probe readouts for ALL window configurations.
    
    Returns dict mapping window_start -> results
    """
    all_results = {}
    
    print(f"\nAnalyzing {len(window_starts)} window configurations...")
    print(f"Window size: {window_size}, Magnitude: {magnitude}")
    
    for window_start in window_starts:
        results = analyze_probe_readouts_single_window(
            cases=cases,
            model=model,
            tokenizer=tokenizer,
            device=device,
            directions=directions,
            probes_z=probes_z,
            blocks_to_analyze=blocks_to_analyze,
            window_size=window_size,
            window_start=window_start,
            magnitude=magnitude,
        )
        all_results[window_start] = results
    
    return all_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_probe_accuracies(
    probes_z: PairwiseProbes,
    probes_seq2pair: Optional[PairwiseProbes],
    output_dir: str,
):
    """Plot probe accuracies across blocks for binary probes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    blocks = sorted(probes_z.positive_accuracies.keys())
    
    # Positive probe accuracy
    ax = axes[0]
    acc_pos_z = [probes_z.positive_accuracies[b] for b in blocks]
    ax.plot(blocks, acc_pos_z, 'o-', label='z', linewidth=2, markersize=4, color='tab:blue')
    if probes_seq2pair is not None:
        acc_pos_s2p = [probes_seq2pair.positive_accuracies.get(b, 0) for b in blocks]
        ax.plot(blocks, acc_pos_s2p, 's-', label='seq2pair', linewidth=2, markersize=4)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Probe: Is Residue Positive? (K, R, H)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Negative probe accuracy
    ax = axes[1]
    acc_neg_z = [probes_z.negative_accuracies[b] for b in blocks]
    ax.plot(blocks, acc_neg_z, 'o-', label='z', linewidth=2, markersize=4, color='tab:red')
    if probes_seq2pair is not None:
        acc_neg_s2p = [probes_seq2pair.negative_accuracies.get(b, 0) for b in blocks]
        ax.plot(blocks, acc_neg_s2p, 's-', label='seq2pair', linewidth=2, markersize=4)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Probe: Is Residue Negative? (D, E)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probe_accuracies.png'), dpi=150)
    plt.close()
    print(f"Saved probe_accuracies.png")


def plot_charge_shift_single_window(
    results: Dict[str, Any],
    window_start: int,
    window_size: int,
    output_dir: str,
):
    """
    Plot P(+) for strand1 and P(-) for strand2 across blocks for a single window.
    """
    window_end = window_start + window_size - 1
    
    data = results
    blocks = sorted(data['baseline']['strand1_pos_prob'].keys())
    
    # Compute means for each block
    baseline_s1_pos = []
    baseline_s1_pos_std = []
    intervened_s1_pos = []
    intervened_s1_pos_std = []
    
    baseline_s2_neg = []
    baseline_s2_neg_std = []
    intervened_s2_neg = []
    intervened_s2_neg_std = []
    
    valid_blocks = []
    
    for block in blocks:
        s1_base = data['baseline']['strand1_pos_prob'][block]
        s1_int = data['intervened']['strand1_pos_prob'][block]
        s2_base = data['baseline']['strand2_neg_prob'][block]
        s2_int = data['intervened']['strand2_neg_prob'][block]
        
        if len(s1_base) > 0 and len(s2_base) > 0:
            valid_blocks.append(block)
            
            baseline_s1_pos.append(np.mean(s1_base))
            baseline_s1_pos_std.append(np.std(s1_base) / np.sqrt(len(s1_base)))
            intervened_s1_pos.append(np.mean(s1_int))
            intervened_s1_pos_std.append(np.std(s1_int) / np.sqrt(len(s1_int)))
            
            baseline_s2_neg.append(np.mean(s2_base))
            baseline_s2_neg_std.append(np.std(s2_base) / np.sqrt(len(s2_base)))
            intervened_s2_neg.append(np.mean(s2_int))
            intervened_s2_neg_std.append(np.std(s2_int) / np.sqrt(len(s2_int)))
    
    if len(valid_blocks) == 0:
        print(f"No valid blocks for window {window_start}-{window_end}!")
        return
    
    valid_blocks = np.array(valid_blocks)
    baseline_s1_pos = np.array(baseline_s1_pos)
    intervened_s1_pos = np.array(intervened_s1_pos)
    baseline_s2_neg = np.array(baseline_s2_neg)
    intervened_s2_neg = np.array(intervened_s2_neg)
    
    # Create subdir for this window
    window_dir = os.path.join(output_dir, f'window_{window_start}_{window_end}')
    os.makedirs(window_dir, exist_ok=True)
    
    # Plot 1: Main figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top: Strand 1 P(+)
    ax = axes[0]
    ax.fill_between(valid_blocks, 
                    baseline_s1_pos - np.array(baseline_s1_pos_std),
                    baseline_s1_pos + np.array(baseline_s1_pos_std),
                    alpha=0.3, color='tab:blue')
    ax.plot(valid_blocks, baseline_s1_pos, 'o-', color='tab:blue', 
            linewidth=2, markersize=4, label='Baseline')
    
    ax.fill_between(valid_blocks,
                    intervened_s1_pos - np.array(intervened_s1_pos_std),
                    intervened_s1_pos + np.array(intervened_s1_pos_std),
                    alpha=0.3, color='tab:red')
    ax.plot(valid_blocks, intervened_s1_pos, 's-', color='tab:red',
            linewidth=2, markersize=4, label='After Intervention')
    
    ax.axvspan(window_start, window_end, alpha=0.2, color='yellow', label='Intervention Window')
    ax.axvline(x=window_start, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('P(positive charge)', fontsize=12)
    ax.set_title(f'Strand 1: P(+) Across Blocks\n(Intervention at blocks {window_start}-{window_end})', fontsize=13)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Bottom: Strand 2 P(-)
    ax = axes[1]
    ax.fill_between(valid_blocks,
                    baseline_s2_neg - np.array(baseline_s2_neg_std),
                    baseline_s2_neg + np.array(baseline_s2_neg_std),
                    alpha=0.3, color='tab:blue')
    ax.plot(valid_blocks, baseline_s2_neg, 'o-', color='tab:blue',
            linewidth=2, markersize=4, label='Baseline')
    
    ax.fill_between(valid_blocks,
                    intervened_s2_neg - np.array(intervened_s2_neg_std),
                    intervened_s2_neg + np.array(intervened_s2_neg_std),
                    alpha=0.3, color='tab:red')
    ax.plot(valid_blocks, intervened_s2_neg, 's-', color='tab:red',
            linewidth=2, markersize=4, label='After Intervention')
    
    ax.axvspan(window_start, window_end, alpha=0.2, color='yellow', label='Intervention Window')
    ax.axvline(x=window_start, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('P(negative charge)', fontsize=12)
    ax.set_title(f'Strand 2: P(-) Across Blocks\n(Intervention at blocks {window_start}-{window_end})', fontsize=13)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(window_dir, 'charge_shift_across_blocks.png'), dpi=150)
    plt.close()
    
    # Plot 2: Delta plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    delta_s1_pos = intervened_s1_pos - baseline_s1_pos
    delta_s2_neg = intervened_s2_neg - baseline_s2_neg
    
    ax.plot(valid_blocks, delta_s1_pos, 'o-', color='tab:blue',
            linewidth=2, markersize=5, label='Δ P(+) Strand 1')
    ax.plot(valid_blocks, delta_s2_neg, 's-', color='tab:red',
            linewidth=2, markersize=5, label='Δ P(-) Strand 2')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.axvspan(window_start, window_end, alpha=0.2, color='yellow', label='Intervention Window')
    ax.axvline(x=window_start, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Change in Probability (Intervened - Baseline)', fontsize=12)
    ax.set_title(f'Effect of DoM Intervention on Charge Readouts\n(Intervention at blocks {window_start}-{window_end})', fontsize=13)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(window_dir, 'charge_shift_delta.png'), dpi=150)
    plt.close()
    
    print(f"  Saved plots for window {window_start}-{window_end}")


def plot_all_windows_comparison(
    all_results: Dict[int, Dict[str, Any]],
    window_size: int,
    blocks_to_analyze: List[int],
    output_dir: str,
):
    """
    Create a comparison plot showing delta charge shift for ALL windows.
    """
    window_starts = sorted(all_results.keys())
    n_windows = len(window_starts)
    
    # Compute deltas for each window
    all_deltas_s1 = {}  # window_start -> array of deltas per block
    all_deltas_s2 = {}
    
    for window_start in window_starts:
        data = all_results[window_start]
        blocks = sorted(data['baseline']['strand1_pos_prob'].keys())
        
        delta_s1 = []
        delta_s2 = []
        valid_blocks = []
        
        for block in blocks:
            s1_base = data['baseline']['strand1_pos_prob'][block]
            s1_int = data['intervened']['strand1_pos_prob'][block]
            s2_base = data['baseline']['strand2_neg_prob'][block]
            s2_int = data['intervened']['strand2_neg_prob'][block]
            
            if len(s1_base) > 0 and len(s2_base) > 0:
                valid_blocks.append(block)
                delta_s1.append(np.mean(s1_int) - np.mean(s1_base))
                delta_s2.append(np.mean(s2_int) - np.mean(s2_base))
        
        all_deltas_s1[window_start] = (valid_blocks, delta_s1)
        all_deltas_s2[window_start] = (valid_blocks, delta_s2)
    
    # Create comparison plots
    
    # Plot 1: All windows on same plot (Strand 1)
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_windows))
    
    for i, window_start in enumerate(window_starts):
        valid_blocks, delta_s1 = all_deltas_s1[window_start]
        window_end = window_start + window_size - 1
        ax.plot(valid_blocks, delta_s1, 'o-', color=colors[i],
                linewidth=1.5, markersize=3, alpha=0.8,
                label=f'Window {window_start}-{window_end}')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Δ P(+) (Intervened - Baseline)', fontsize=12)
    ax.set_title(f'Strand 1 Charge Shift: All Window Configurations\n(window_size={window_size})', fontsize=13)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_windows_strand1_comparison.png'), dpi=150)
    plt.close()
    
    # Plot 2: All windows on same plot (Strand 2)
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, window_start in enumerate(window_starts):
        valid_blocks, delta_s2 = all_deltas_s2[window_start]
        window_end = window_start + window_size - 1
        ax.plot(valid_blocks, delta_s2, 's-', color=colors[i],
                linewidth=1.5, markersize=3, alpha=0.8,
                label=f'Window {window_start}-{window_end}')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Δ P(-) (Intervened - Baseline)', fontsize=12)
    ax.set_title(f'Strand 2 Charge Shift: All Window Configurations\n(window_size={window_size})', fontsize=13)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_windows_strand2_comparison.png'), dpi=150)
    plt.close()
    
    # Plot 3: Heatmap of delta at each block for each window
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Get common blocks
    common_blocks = None
    for ws in window_starts:
        vb, _ = all_deltas_s1[ws]
        if common_blocks is None:
            common_blocks = set(vb)
        else:
            common_blocks = common_blocks.intersection(set(vb))
    common_blocks = sorted(common_blocks)
    
    # Build matrices
    delta_matrix_s1 = np.zeros((n_windows, len(common_blocks)))
    delta_matrix_s2 = np.zeros((n_windows, len(common_blocks)))
    
    for i, window_start in enumerate(window_starts):
        vb_s1, d_s1 = all_deltas_s1[window_start]
        vb_s2, d_s2 = all_deltas_s2[window_start]
        
        block_to_delta_s1 = dict(zip(vb_s1, d_s1))
        block_to_delta_s2 = dict(zip(vb_s2, d_s2))
        
        for j, block in enumerate(common_blocks):
            delta_matrix_s1[i, j] = block_to_delta_s1.get(block, 0)
            delta_matrix_s2[i, j] = block_to_delta_s2.get(block, 0)
    
    # Strand 1 heatmap
    ax = axes[0]
    im = ax.imshow(delta_matrix_s1, aspect='auto', cmap='RdBu_r', 
                   vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Intervention Window Start', fontsize=12)
    ax.set_title('Strand 1: Δ P(+)', fontsize=13)
    ax.set_xticks(range(0, len(common_blocks), 5))
    ax.set_xticklabels([common_blocks[i] for i in range(0, len(common_blocks), 5)])
    ax.set_yticks(range(n_windows))
    ax.set_yticklabels([f'{ws}-{ws+window_size-1}' for ws in window_starts])
    plt.colorbar(im, ax=ax, label='Δ P(+)')
    
    # Strand 2 heatmap
    ax = axes[1]
    im = ax.imshow(delta_matrix_s2, aspect='auto', cmap='RdBu_r',
                   vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Intervention Window Start', fontsize=12)
    ax.set_title('Strand 2: Δ P(-)', fontsize=13)
    ax.set_xticks(range(0, len(common_blocks), 5))
    ax.set_xticklabels([common_blocks[i] for i in range(0, len(common_blocks), 5)])
    ax.set_yticks(range(n_windows))
    ax.set_yticklabels([f'{ws}-{ws+window_size-1}' for ws in window_starts])
    plt.colorbar(im, ax=ax, label='Δ P(-)')
    
    plt.suptitle(f'Charge Shift Heatmap Across All Windows (window_size={window_size})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_windows_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Grid of subplots showing each window
    n_cols = 3
    n_rows = (n_windows + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), sharex=True, sharey=True)
    axes = axes.flatten() if n_windows > 1 else [axes]
    
    for i, window_start in enumerate(window_starts):
        ax = axes[i]
        valid_blocks, delta_s1 = all_deltas_s1[window_start]
        _, delta_s2 = all_deltas_s2[window_start]
        window_end = window_start + window_size - 1
        
        ax.plot(valid_blocks, delta_s1, 'o-', color='tab:blue',
                linewidth=1.5, markersize=3, label='Δ P(+) S1')
        ax.plot(valid_blocks, delta_s2, 's-', color='tab:red',
                linewidth=1.5, markersize=3, label='Δ P(-) S2')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvspan(window_start, window_end, alpha=0.2, color='yellow')
        ax.axvline(x=window_start, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f'Window {window_start}-{window_end}', fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=7)
    
    # Hide empty subplots
    for i in range(n_windows, len(axes)):
        axes[i].set_visible(False)
    
    fig.supxlabel('Block', fontsize=12)
    fig.supylabel('Δ Probability', fontsize=12)
    plt.suptitle(f'Charge Shift for Each Window Configuration (window_size={window_size})', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_windows_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plots to {output_dir}")


def plot_summary_all_windows(
    all_results: Dict[int, Dict[str, Any]],
    probes_z: PairwiseProbes,
    window_size: int,
    output_dir: str,
):
    """Create a summary figure for all windows."""
    
    window_starts = sorted(all_results.keys())
    
    # Compute average delta for each window (at blocks >= window_end)
    summary_data = []
    
    for window_start in window_starts:
        data = all_results[window_start]
        window_end = window_start + window_size - 1
        blocks = sorted(data['baseline']['strand1_pos_prob'].keys())
        
        # Average delta at blocks AFTER the intervention window
        post_delta_s1 = []
        post_delta_s2 = []
        
        for block in blocks:
            if block > window_end:  # Only look at blocks after intervention
                s1_base = data['baseline']['strand1_pos_prob'][block]
                s1_int = data['intervened']['strand1_pos_prob'][block]
                s2_base = data['baseline']['strand2_neg_prob'][block]
                s2_int = data['intervened']['strand2_neg_prob'][block]
                
                if len(s1_base) > 0:
                    post_delta_s1.append(np.mean(s1_int) - np.mean(s1_base))
                    post_delta_s2.append(np.mean(s2_int) - np.mean(s2_base))
        
        summary_data.append({
            'window_start': window_start,
            'window_end': window_end,
            'avg_delta_s1': np.mean(post_delta_s1) if post_delta_s1 else 0,
            'avg_delta_s2': np.mean(post_delta_s2) if post_delta_s2 else 0,
            'std_delta_s1': np.std(post_delta_s1) if post_delta_s1 else 0,
            'std_delta_s2': np.std(post_delta_s2) if post_delta_s2 else 0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Bar chart of average post-intervention delta
    ax = axes[0, 0]
    x = np.arange(len(window_starts))
    width = 0.35
    ax.bar(x - width/2, summary_df['avg_delta_s1'], width, label='Δ P(+) Strand 1', color='tab:blue', alpha=0.7)
    ax.bar(x + width/2, summary_df['avg_delta_s2'], width, label='Δ P(-) Strand 2', color='tab:red', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Intervention Window', fontsize=11)
    ax.set_ylabel('Average Δ Probability (post-intervention blocks)', fontsize=11)
    ax.set_title('A. Average Charge Shift by Window', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ws}-{ws+window_size-1}' for ws in window_starts], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Top-right: Probe accuracies
    ax = axes[0, 1]
    probe_blocks = sorted(probes_z.positive_accuracies.keys())
    acc_pos = [probes_z.positive_accuracies[b] for b in probe_blocks]
    acc_neg = [probes_z.negative_accuracies[b] for b in probe_blocks]
    ax.plot(probe_blocks, acc_pos, 'o-', color='tab:blue', linewidth=2, markersize=3, label='P(+) probe')
    ax.plot(probe_blocks, acc_neg, 's-', color='tab:red', linewidth=2, markersize=3, label='P(-) probe')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
    ax.set_xlabel('Block', fontsize=11)
    ax.set_ylabel('Balanced Accuracy', fontsize=11)
    ax.set_title('B. Probe Accuracy', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Bottom: Line plot showing delta across blocks for first and last windows
    ax = axes[1, 0]
    
    first_ws = window_starts[0]
    last_ws = window_starts[-1]
    
    data_first = all_results[first_ws]
    data_last = all_results[last_ws]
    
    blocks = sorted(data_first['baseline']['strand1_pos_prob'].keys())
    
    delta_s1_first = []
    delta_s1_last = []
    valid_blocks = []
    
    for block in blocks:
        s1_base_f = data_first['baseline']['strand1_pos_prob'][block]
        s1_int_f = data_first['intervened']['strand1_pos_prob'][block]
        s1_base_l = data_last['baseline']['strand1_pos_prob'][block]
        s1_int_l = data_last['intervened']['strand1_pos_prob'][block]
        
        if len(s1_base_f) > 0 and len(s1_base_l) > 0:
            valid_blocks.append(block)
            delta_s1_first.append(np.mean(s1_int_f) - np.mean(s1_base_f))
            delta_s1_last.append(np.mean(s1_int_l) - np.mean(s1_base_l))
    
    ax.plot(valid_blocks, delta_s1_first, 'o-', color='tab:blue',
            linewidth=2, markersize=4, label=f'Window {first_ws}-{first_ws+window_size-1}')
    ax.plot(valid_blocks, delta_s1_last, 's-', color='tab:orange',
            linewidth=2, markersize=4, label=f'Window {last_ws}-{last_ws+window_size-1}')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.axvspan(first_ws, first_ws+window_size-1, alpha=0.15, color='tab:blue')
    ax.axvspan(last_ws, last_ws+window_size-1, alpha=0.15, color='tab:orange')
    ax.set_xlabel('Block', fontsize=11)
    ax.set_ylabel('Δ P(+) Strand 1', fontsize=11)
    ax.set_title('C. Strand 1 Charge Shift: First vs Last Window', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Bottom-right: Text summary
    ax = axes[1, 1]
    ax.axis('off')
    
    text_lines = [
        "Summary Statistics",
        "=" * 40,
        f"Window size: {window_size}",
        f"Windows tested: {len(window_starts)}",
        f"  Range: {window_starts[0]}-{window_starts[0]+window_size-1} to {window_starts[-1]}-{window_starts[-1]+window_size-1}",
        "",
        "Best window for Strand 1 (highest Δ P(+)):",
        f"  {summary_df.loc[summary_df['avg_delta_s1'].idxmax(), 'window_start']:.0f}-"
        f"{summary_df.loc[summary_df['avg_delta_s1'].idxmax(), 'window_end']:.0f}: "
        f"{summary_df['avg_delta_s1'].max():.3f}",
        "",
        "Best window for Strand 2 (highest Δ P(-)):",
        f"  {summary_df.loc[summary_df['avg_delta_s2'].idxmax(), 'window_start']:.0f}-"
        f"{summary_df.loc[summary_df['avg_delta_s2'].idxmax(), 'window_end']:.0f}: "
        f"{summary_df['avg_delta_s2'].max():.3f}",
    ]
    
    ax.text(0.1, 0.9, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Early Block Windows Analysis Summary', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'early_blocks_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary data
    summary_df.to_csv(os.path.join(output_dir, 'window_summary.csv'), index=False)
    
    print(f"Saved summary to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='probing_results_early_blocks')
    parser.add_argument('--directions', type=str, 
                        default="/share/u/kevin/ProteinFolding/fixed_charge/charge_directions.pt")
    parser.add_argument('--results_parquet', type=str,
                        default="/share/u/kevin/ProteinFolding/fixed_charge/hairpin_induction_results.parquet",
                        help='Parquet with hairpin induction results (used for sequence info)')
    parser.add_argument('--probes_path', type=str, 
                        default="/share/u/kevin/ProteinFolding/fixed_pairwise_probing_results_sliding_window",
                        help='Path to pre-trained probes')
    parser.add_argument('--n_cases', type=int, default=20,
                        help='Number of random cases to analyze per window')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Size of intervention window (e.g., 4 means 4 consecutive blocks)')
    parser.add_argument('--window_start_min', type=int, default=10,
                        help='Minimum window start block')
    parser.add_argument('--window_start_max', type=int, default=44,
                        help='Maximum window start block')
    parser.add_argument('--magnitude', type=float, default=2.0,
                        help='Intervention magnitude')
    parser.add_argument('--blocks', type=int, nargs='+', 
                        default=list(range(0, 48)))  # All blocks for tracking
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define window configurations
    window_starts = list(range(args.window_start_min, args.window_start_max + 1))
    
    print(f"\n" + "="*60)
    print(f"EARLY BLOCKS SWEEP PARAMETERS")
    print(f"="*60)
    print(f"Window size: {args.window_size}")
    print(f"Window starts: {window_starts}")
    print(f"Window ranges: {[f'{ws}-{ws+args.window_size-1}' for ws in window_starts]}")
    print(f"Magnitude: {args.magnitude}")
    print(f"Cases per window: {args.n_cases}")
    
    # Load model
    print("\nLoading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    
    # Load directions
    print(f"\nLoading directions from {args.directions}...")
    directions = load_directions(args.directions)
    
    # Load pre-trained probes
    print(f"\nLoading pre-trained probes from {args.probes_path}...")
    probes_z = load_probes(args.probes_path, 'z')
    
    if probes_z is None:
        print("ERROR: Could not load pre-trained probes!")
        print(f"Please ensure probes exist at {args.probes_path}/probes_z.pkl")
        return
    
    # Print probe accuracies
    print("\nProbe accuracies (z representation):")
    print("  Block | P(+) Acc | P(-) Acc")
    print("  " + "-"*30)
    for block in sorted(probes_z.positive_accuracies.keys())[:10]:  # Just first 10
        print(f"  {block:5d} | {probes_z.positive_accuracies[block]:.3f}    | {probes_z.negative_accuracies[block]:.3f}")
    print("  ...")
    
    # Plot probe accuracies
    plot_probe_accuracies(probes_z, None, args.output)
    
    # Load random cases
    print(f"\n" + "="*60)
    print(f"LOADING RANDOM CASES")
    print(f"="*60)
    
    cases = get_random_cases_from_parquet(
        results_parquet=args.results_parquet,
        n_cases=args.n_cases,
        seed=args.seed,
    )
    
    if len(cases) == 0:
        print("ERROR: No cases found!")
        return
    
    # Analyze all window configurations
    print(f"\n" + "="*60)
    print(f"ANALYZING ALL WINDOW CONFIGURATIONS")
    print(f"="*60)
    
    all_results = analyze_all_window_configurations(
        cases=cases,
        model=model,
        tokenizer=tokenizer,
        device=device,
        directions=directions,
        probes_z=probes_z,
        blocks_to_analyze=args.blocks,
        window_size=args.window_size,
        window_starts=window_starts,
        magnitude=args.magnitude,
    )
    
    # Generate plots for each window
    print(f"\n" + "="*60)
    print(f"GENERATING PLOTS")
    print(f"="*60)
    
    for window_start in window_starts:
        plot_charge_shift_single_window(
            results=all_results[window_start],
            window_start=window_start,
            window_size=args.window_size,
            output_dir=args.output,
        )
    
    # Generate comparison plots
    plot_all_windows_comparison(
        all_results=all_results,
        window_size=args.window_size,
        blocks_to_analyze=args.blocks,
        output_dir=args.output,
    )
    
    # Generate summary
    plot_summary_all_windows(
        all_results=all_results,
        probes_z=probes_z,
        window_size=args.window_size,
        output_dir=args.output,
    )
    
    # Save all results
    torch.save({
        'all_results': all_results,
        'window_starts': window_starts,
        'window_size': args.window_size,
        'magnitude': args.magnitude,
        'n_cases': args.n_cases,
        'probe_accuracies_z': {
            'positive': probes_z.positive_accuracies,
            'negative': probes_z.negative_accuracies,
        },
    }, os.path.join(args.output, 'all_results.pt'))
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    
    for window_start in window_starts:
        data = all_results[window_start]
        window_end = window_start + args.window_size - 1
        
        # Compute average delta across all blocks after intervention
        all_delta_s1 = []
        all_delta_s2 = []
        
        for block in sorted(data['baseline']['strand1_pos_prob'].keys()):
            if block > window_end:
                s1_base = data['baseline']['strand1_pos_prob'][block]
                s1_int = data['intervened']['strand1_pos_prob'][block]
                s2_base = data['baseline']['strand2_neg_prob'][block]
                s2_int = data['intervened']['strand2_neg_prob'][block]
                
                if len(s1_base) > 0:
                    all_delta_s1.append(np.mean(s1_int) - np.mean(s1_base))
                    all_delta_s2.append(np.mean(s2_int) - np.mean(s2_base))
        
        avg_s1 = np.mean(all_delta_s1) if all_delta_s1 else 0
        avg_s2 = np.mean(all_delta_s2) if all_delta_s2 else 0
        
        print(f"Window {window_start:2d}-{window_end:2d}: "
              f"Avg Δ P(+) S1 = {avg_s1:+.4f}, "
              f"Avg Δ P(-) S2 = {avg_s2:+.4f}")
    
    print(f"\nResults saved to {args.output}/")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()