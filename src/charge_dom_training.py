#!/usr/bin/env python
"""
Charge Direction Training (Difference of Means)
================================================

Trains Difference-of-Means (DoM) vectors to identify charge-associated directions
in ESMFold's sequence representation (s). These directions are used by the charge
steering experiments to induce or disrupt hairpin formation.

The DoM approach:
1. Collect s representations for residues across many sequences
2. Group by residue charge: positive (K, R, H), negative (D, E), neutral
3. Compute direction = mean(positive) - mean(negative) for each block
4. Normalize and save for use in steering experiments

Output:
    charge_directions.pt: Contains s_directions and s_stds for all blocks

Usage:
    python charge_dom_training.py \
        --probing_dataset data/probing_train_test.csv \
        --output charge_directions/ \
        --blocks 0 1 2 3 ... 47
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

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput
)
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

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

NUM_BLOCKS = 48


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChargeDirections:
    """
    Cached charge directions for all blocks (s representation only).

    Attributes:
        s_directions: Dict mapping block_idx -> normalized direction vector
        s_stds: Dict mapping block_idx -> standard deviation for magnitude scaling
    """
    s_directions: Dict[int, np.ndarray]
    s_stds: Dict[int, float]


# ============================================================================
# MODEL FORWARD PASS FOR COLLECTION
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


# ============================================================================
# REPRESENTATION COLLECTION
# ============================================================================

def collect_representations_for_sequence(
    model, seq: str, tokenizer, device: str
) -> Dict[str, List[torch.Tensor]]:
    """Collect s representations for a sequence."""
    model.forward = types.MethodType(baseline_forward, model)
    model.trunk.forward = types.MethodType(baseline_forward_trunk, model.trunk)

    with torch.no_grad():
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(**inputs, num_recycles=0)

    return {
        's_list': [s.cpu() for s in outputs['s_list']],
    }


def compute_charge_labels(seq: str) -> List[int]:
    """Get charge labels for each residue: +1, -1, or 0."""
    return [AA_CHARGE.get(aa, 0) for aa in seq]


# ============================================================================
# DOM TRAINING
# ============================================================================

def train_dom_vectors(
    sequences: List[str],
    model,
    tokenizer,
    device: str,
    blocks_to_train: List[int],
) -> ChargeDirections:
    """
    Train Difference of Means vectors for charge direction (s representation only).

    For each block, compute:
    - s direction: mean(positive_charged_residues) - mean(negative_charged_residues)

    Args:
        sequences: List of protein sequences
        model: ESMFold model
        tokenizer: ESMFold tokenizer
        device: Device to run on
        blocks_to_train: Which blocks to train directions for

    Returns:
        ChargeDirections with s_directions and s_stds
    """
    print(f"\nTraining DoM vectors on {len(sequences)} sequences...")

    # Collect representations grouped by charge
    s_positive = {block: [] for block in blocks_to_train}
    s_negative = {block: [] for block in blocks_to_train}

    for seq in tqdm(sequences, desc="Collecting representations"):
        if not all(aa in AA_TO_IDX for aa in seq):
            continue

        try:
            data = collect_representations_for_sequence(model, seq, tokenizer, device)
        except Exception as e:
            print(f"Error processing sequence: {e}")
            continue

        charges = compute_charge_labels(seq)

        for block in blocks_to_train:
            s = data['s_list'][block][0]  # [L, dim]

            # Collect s representations by charge
            for i, charge in enumerate(charges):
                if charge > 0:
                    s_positive[block].append(s[i].numpy())
                elif charge < 0:
                    s_negative[block].append(s[i].numpy())

        del data
        torch.cuda.empty_cache()

    # Compute DoM vectors
    print("\nComputing DoM vectors...")

    s_directions = {}
    s_stds = {}

    for block in tqdm(blocks_to_train, desc="Computing directions"):
        # s direction: positive - negative
        if len(s_positive[block]) > 0 and len(s_negative[block]) > 0:
            s_pos_mean = np.mean(s_positive[block], axis=0)
            s_neg_mean = np.mean(s_negative[block], axis=0)
            s_dir = s_pos_mean - s_neg_mean
            s_dir = s_dir / (np.linalg.norm(s_dir) + 1e-8)
            s_directions[block] = s_dir

            # Compute std for scaling
            all_s = np.concatenate([s_positive[block], s_negative[block]], axis=0)
            s_stds[block] = np.std(all_s @ s_dir)

    print(f"Trained directions for {len(s_directions)} blocks")

    return ChargeDirections(
        s_directions=s_directions,
        s_stds=s_stds,
    )


# ============================================================================
# DOM EVALUATION
# ============================================================================

def evaluate_dom_vectors(
    directions: ChargeDirections,
    sequences: List[str],
    model,
    tokenizer,
    device: str,
    blocks_to_eval: List[int],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate DoM vectors on test set using separation scores.

    Returns:
        eval_df: DataFrame with separation scores per block
        projections_data: Dict with projections for plotting
    """
    print(f"\nEvaluating DoM vectors on {len(sequences)} sequences...")

    # Collect projections
    s_projections = {block: [] for block in blocks_to_eval}
    s_charges = {block: [] for block in blocks_to_eval}

    for seq in tqdm(sequences, desc="Evaluating"):
        if not all(aa in AA_TO_IDX for aa in seq):
            continue

        try:
            data = collect_representations_for_sequence(model, seq, tokenizer, device)
        except Exception as e:
            continue

        charges = compute_charge_labels(seq)

        for block in blocks_to_eval:
            if block not in directions.s_directions:
                continue

            s = data['s_list'][block][0].numpy()
            s_dir = directions.s_directions[block]

            # s projections: project each residue onto direction
            for i, charge in enumerate(charges):
                if charge != 0:
                    proj = np.dot(s[i], s_dir)
                    s_projections[block].append(proj)
                    s_charges[block].append(charge)

        del data
        torch.cuda.empty_cache()

    # Compute separation scores and Cohen's d
    print("\nComputing separation scores...")

    results = []

    for block in tqdm(blocks_to_eval, desc="Computing metrics"):
        if len(s_projections[block]) > 0:
            s_projs = np.array(s_projections[block])
            s_chrgs = np.array(s_charges[block])

            pos_mask = s_chrgs > 0
            neg_mask = s_chrgs < 0

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_mean = s_projs[pos_mask].mean()
                neg_mean = s_projs[neg_mask].mean()
                separation = pos_mean - neg_mean

                # Cohen's d
                pos_std = s_projs[pos_mask].std()
                neg_std = s_projs[neg_mask].std()
                pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
                cohens_d = separation / pooled_std if pooled_std > 0 else 0

                results.append({
                    'block': block,
                    'separation': separation,
                    'cohens_d': cohens_d,
                    'pos_mean': pos_mean,
                    'neg_mean': neg_mean,
                    'n_positive': pos_mask.sum(),
                    'n_negative': neg_mask.sum(),
                })

    eval_df = pd.DataFrame(results)

    # Package projections for plotting
    projections_data = {
        'projections': s_projections,
        'labels': s_charges,
    }

    return eval_df, projections_data


# ============================================================================
# PLOTTING
# ============================================================================

def plot_dom_evaluation(eval_df: pd.DataFrame, output_dir: str):
    """Plot DoM evaluation results - separation scores across blocks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Separation score by block
    ax = axes[0]
    ax.plot(eval_df['block'], eval_df['separation'], 'o-',
            color='#2ecc71', linewidth=2, markersize=5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Separation Score', fontsize=12)
    ax.set_title('Charge Separation Score Across Blocks (Test Set)', fontsize=13)
    ax.grid(alpha=0.3)

    # Right: Cohen's d by block
    ax = axes[1]
    ax.plot(eval_df['block'], eval_df['cohens_d'], 'o-',
            color='#3498db', linewidth=2, markersize=5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect (0.8)')
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title('Effect Size Across Blocks (Test Set)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dom_evaluation_separation.png'), dpi=150)
    plt.close()
    print(f"Saved DoM evaluation plot to {output_dir}/dom_evaluation_separation.png")


def plot_projection_histograms(
    projections_data: Dict,
    blocks_to_show: List[int],
    output_dir: str,
):
    """Plot centered projection histograms for selected blocks."""
    n_blocks = len(blocks_to_show)
    n_cols = min(4, n_blocks)
    n_rows = (n_blocks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    s_projs = projections_data['projections']
    s_labels = projections_data['labels']

    for ax_idx, block in enumerate(blocks_to_show):
        ax = axes[ax_idx]

        if block not in s_projs or len(s_projs[block]) == 0:
            ax.set_visible(False)
            continue

        projs = np.array(s_projs[block])
        labels = np.array(s_labels[block])

        pos_mask = labels > 0
        neg_mask = labels < 0

        # Center the projections (subtract overall mean)
        overall_mean = projs.mean()
        projs_centered = projs - overall_mean

        bins = np.linspace(projs_centered.min(), projs_centered.max(), 40)

        ax.hist(projs_centered[pos_mask], bins=bins, alpha=0.7,
                color='#3498db', density=True, label=f'Positive (n={pos_mask.sum()})')
        ax.hist(projs_centered[neg_mask], bins=bins, alpha=0.7,
                color='#e74c3c', density=True, label=f'Negative (n={neg_mask.sum()})')

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Add separation score
        sep = projs_centered[pos_mask].mean() - projs_centered[neg_mask].mean()
        ax.set_title(f'Block {block} (sep={sep:.2f})', fontsize=11)
        ax.set_xlabel('Centered Projection', fontsize=10)

        if ax_idx % n_cols == 0:
            ax.set_ylabel('Density', fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(len(blocks_to_show), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('s Component: Centered Charge Projections (Test Set)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dom_s_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved projection histograms to {output_dir}/")


# ============================================================================
# SAVE/LOAD
# ============================================================================

def save_directions(directions: ChargeDirections, path: str):
    """Save DoM directions to file."""
    torch.save({
        's_directions': directions.s_directions,
        's_stds': directions.s_stds,
    }, path)
    print(f"Saved directions to {path}")


def load_directions(path: str) -> ChargeDirections:
    """Load DoM directions from file."""
    data = torch.load(path, weights_only=False)
    return ChargeDirections(
        s_directions=data['s_directions'],
        s_stds=data['s_stds'],
    )


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train charge direction (DoM) vectors for ESMFold steering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--probing_dataset', type=str, required=True,
                        help='Path to probing dataset CSV with train/test splits')
    parser.add_argument('--output', type=str, default='charge_directions',
                        help='Output directory for directions and plots')
    parser.add_argument('--blocks', type=int, nargs='+', default=list(range(48)),
                        help='Which blocks to train directions for')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available)')
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

    # Load model
    print("\nLoading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    blocks_to_use = args.blocks

    # =========================================================================
    # Load and prepare data
    # =========================================================================
    print(f"\nLoading probing dataset from {args.probing_dataset}...")
    probing_df = pd.read_csv(args.probing_dataset)

    # Show dataset composition
    print(f"\nDataset composition:")
    if 'split' in probing_df.columns and 'structure_type' in probing_df.columns:
        print(probing_df.groupby(['split', 'structure_type']).size().unstack(fill_value=0))

    # =========================================================================
    # TRAINING: Use only alpha_helical sequences from train split
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: TRAIN DoM VECTORS")
    print("="*60)

    train_df = probing_df[
        (probing_df['split'] == 'train') &
        (probing_df['structure_type'] == 'alpha_helical')
    ]

    # Get sequences - handle both possible column names
    seq_col = 'FullChainSequence' if 'FullChainSequence' in train_df.columns else 'sequence'
    train_seqs = train_df[seq_col].tolist()

    # Filter for valid AA sequences
    train_seqs = [s for s in train_seqs if all(aa in AA_TO_IDX for aa in s)]

    print(f"\nTrain sequences (alpha_helical only): {len(train_seqs)}")

    # Shuffle training data
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
    directions_path = os.path.join(args.output, 'charge_directions.pt')
    save_directions(directions, directions_path)

    # =========================================================================
    # EVALUATION: Use entire test split
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: EVALUATE DoM VECTORS")
    print("="*60)

    test_df = probing_df[probing_df['split'] == 'test']

    seq_col = 'FullChainSequence' if 'FullChainSequence' in test_df.columns else 'sequence'
    test_seqs = test_df[seq_col].tolist()

    # Filter for valid AA sequences
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

    print("\n" + "="*60)
    print(f"DONE - Directions saved to {directions_path}")
    print("="*60)


if __name__ == '__main__':
    main()
