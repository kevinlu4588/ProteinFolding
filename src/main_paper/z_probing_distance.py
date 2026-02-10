#!/usr/bin/env python
"""
Distance Probing for ESMFold z Representations
==============================================

Train linear probes to predict CA-CA distances from z[i,j] representations.
Uses standard Ridge regression (closed-form solution).

Usage:
    python distance_probing.py \
        --probing_dataset data/probing_dataset.csv \
        --output results/
"""

import argparse
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge  # Only used for comparison if needed

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput

from src.utils.representation_utils import CollectedRepresentations, TrunkHooks


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_BLOCKS = 48
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DistanceProbe:
    """A trained linear probe for predicting distance from z."""
    weights: np.ndarray
    bias: float
    block_idx: int
    r2_train: float = None

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Predict distance(s) from z representation(s)."""
        return np.dot(z, self.weights) + self.bias

    def get_gradient_direction(self) -> np.ndarray:
        """Get normalized direction that DECREASES predicted distance."""
        direction = -self.weights
        return direction / (np.linalg.norm(direction) + 1e-8)

    def get_steering_vector(self, current_z: np.ndarray, target_distance: float) -> np.ndarray:
        """
        Compute the steering vector to move predicted distance toward target.

        For a linear probe: pred = z . w + b
        To change pred by delta, we need to change z by delta * w / ||w||^2
        """
        current_pred = self.predict(current_z)
        delta_distance = target_distance - current_pred

        w_norm_sq = np.dot(self.weights, self.weights)
        if w_norm_sq < 1e-8:
            return np.zeros_like(self.weights)

        return delta_distance * self.weights / w_norm_sq


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def run_and_collect_z(
    model,
    tokenizer,
    device: str,
    sequence: str,
) -> Tuple[EsmForProteinFoldingOutput, CollectedRepresentations]:
    """
    Run model and collect z representations from all trunk blocks.
    """
    collector = CollectedRepresentations()
    trunk_hooks = TrunkHooks(model.trunk, collector)
    trunk_hooks.register(collect_s=False, collect_z=True)
    
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=0)
    finally:
        trunk_hooks.remove()
    
    return outputs, collector


def compute_ca_distances(positions: torch.Tensor) -> torch.Tensor:
    """Compute CA-CA distance matrix."""
    CA_IDX = 1
    ca_coords = positions[:, CA_IDX, :]
    diff = ca_coords.unsqueeze(0) - ca_coords.unsqueeze(1)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


# ============================================================================
# ONLINE RIDGE REGRESSION (SUFFICIENT STATISTICS)
# ============================================================================

@dataclass
class OnlineRidgeAccumulator:
    """
    Accumulates sufficient statistics for Ridge regression.
    
    Only stores X'X [dim, dim] and X'y [dim], not the raw data.
    This is mathematically equivalent to batch Ridge regression.
    """
    dim: int
    block_idx: int
    
    # Sufficient statistics
    XtX: np.ndarray = None  # [dim, dim]
    Xty: np.ndarray = None  # [dim]
    yty: float = 0.0        # scalar (for R² computation)
    y_sum: float = 0.0      # sum of y (for mean)
    n_samples: int = 0
    
    def __post_init__(self):
        self.XtX = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.Xty = np.zeros(self.dim, dtype=np.float64)
    
    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Update sufficient statistics with a batch of samples.
        
        Args:
            X: Feature matrix [batch_size, dim]
            y: Target values [batch_size]
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.yty += np.dot(y, y)
        self.y_sum += np.sum(y)
        self.n_samples += len(y)
    
    def solve(self, alpha: float = 1.0) -> DistanceProbe:
        """
        Solve Ridge regression: w = (X'X + αI)^(-1) X'y
        
        Args:
            alpha: Regularization strength
        
        Returns:
            Trained DistanceProbe
        """
        if self.n_samples == 0:
            raise ValueError(f"No samples for block {self.block_idx}")
        
        # Solve (X'X + αI) w = X'y
        regularized = self.XtX + alpha * np.eye(self.dim)
        weights = np.linalg.solve(regularized, self.Xty)
        
        # Bias: for Ridge without centering, bias = 0
        # For proper centering: bias = y_mean - X_mean @ weights
        # We'll use bias = 0 since we're not centering
        bias = 0.0
        
        # Compute training R² from sufficient statistics
        # R² = 1 - SS_res / SS_tot
        # SS_res = y'y - 2*w'X'y + w'X'X*w
        # SS_tot = y'y - n * y_mean²
        y_mean = self.y_sum / self.n_samples
        ss_tot = self.yty - self.n_samples * y_mean ** 2
        ss_res = self.yty - 2 * np.dot(weights, self.Xty) + weights @ self.XtX @ weights
        r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return DistanceProbe(
            weights=weights,
            bias=bias,
            block_idx=self.block_idx,
            r2_train=r2_train,
        )


# ============================================================================
# PROBING
# ============================================================================

def train_probes_online(
    sequences: List[str],
    model,
    tokenizer,
    device: str,
    n_pairs_per_protein: int = 500,
    alpha: float = 1.0,
    max_proteins: int = None,
) -> Dict[int, DistanceProbe]:
    """
    Train Ridge regression probes using online sufficient statistics accumulation.
    
    Memory efficient: only stores [dim, dim] matrix per block, not raw samples.
    Mathematically identical to batch Ridge regression.
    
    Args:
        sequences: List of protein sequences
        model: ESMFold model
        tokenizer: Tokenizer
        device: Torch device
        n_pairs_per_protein: Pairs to sample per protein
        alpha: Ridge regularization strength
        max_proteins: Max proteins to process (None = all)
    
    Returns:
        Dict mapping block_idx -> trained DistanceProbe
    """
    # We need to know the dimension - get it from first valid protein
    dim = None
    accumulators = None
    
    proteins_processed = 0
    total_samples = 0
    
    for seq in tqdm(sequences, desc="Training probes (online)"):
        # Skip invalid sequences
        if not all(aa in AA_TO_IDX for aa in seq):
            continue
        if len(seq) < 10:
            continue
        
        L = len(seq)
        
        try:
            outputs, collector = run_and_collect_z(model, tokenizer, device, seq)
        except Exception as e:
            print(f"Error processing sequence: {e}")
            continue
        
        # Initialize accumulators on first protein
        if dim is None:
            # Get dimension from first block's z
            first_z = next(iter(collector.z_blocks.values()))
            dim = first_z.shape[-1]
            print(f"Z dimension: {dim}")
            accumulators = {b: OnlineRidgeAccumulator(dim=dim, block_idx=b) for b in range(NUM_BLOCKS)}
        
        # Compute CA distances
        final_positions = outputs.positions[-1, 0]
        distances = compute_ca_distances(final_positions)
        
        # Sample pairs (all pairs where i != j)
        all_pairs = [(i, j) for i in range(L) for j in range(L) if i != j]
        
        if len(all_pairs) > n_pairs_per_protein:
            sampled_indices = np.random.choice(len(all_pairs), n_pairs_per_protein, replace=False)
            sampled_pairs = [all_pairs[k] for k in sampled_indices]
        else:
            sampled_pairs = all_pairs
        
        # Update accumulators for each block
        for block_idx, z in collector.z_blocks.items():
            z_np = z[0].cpu().numpy()  # [L, L, dim]
            
            # Gather batch for this protein
            X_batch = np.array([z_np[i, j] for i, j in sampled_pairs], dtype=np.float64)
            y_batch = np.array([distances[i, j].item() for i, j in sampled_pairs], dtype=np.float64)
            
            accumulators[block_idx].update(X_batch, y_batch)
        
        proteins_processed += 1
        total_samples += len(sampled_pairs)
        
        # Clean up
        del outputs, collector
        torch.cuda.empty_cache()
        
        if max_proteins and proteins_processed >= max_proteins:
            break
    
    print(f"\nProcessed {proteins_processed} proteins, {total_samples} total samples")
    
    # Solve all probes
    print("\nSolving Ridge regression for each block...")
    probes = {}
    for block in tqdm(range(NUM_BLOCKS), desc="Solving probes"):
        if accumulators[block].n_samples < 100:
            print(f"Block {block}: insufficient samples ({accumulators[block].n_samples})")
            continue
        
        probe = accumulators[block].solve(alpha=alpha)
        probes[block] = probe
        
        if block % 8 == 0:
            tqdm.write(f"  Block {block}: R² = {probe.r2_train:.4f}, n = {accumulators[block].n_samples}")
    
    return probes


def evaluate_probes_online(
    probes: Dict[int, DistanceProbe],
    sequences: List[str],
    model,
    tokenizer,
    device: str,
    n_pairs_per_protein: int = 500,
    max_proteins: int = None,
) -> pd.DataFrame:
    """
    Evaluate probes on test data (processes one protein at a time).
    
    Returns:
        DataFrame with evaluation metrics per block
    """
    # Accumulators for evaluation metrics
    # We need: sum of squared residuals, sum of squared total, n
    ss_res = {b: 0.0 for b in probes.keys()}
    ss_tot_acc = {b: 0.0 for b in probes.keys()}  # accumulate (y - y_mean)² later
    y_sum = {b: 0.0 for b in probes.keys()}
    y_sq_sum = {b: 0.0 for b in probes.keys()}
    n_samples = {b: 0 for b in probes.keys()}
    
    # For correlation: need sum(y), sum(y_pred), sum(y²), sum(y_pred²), sum(y*y_pred)
    sum_y = {b: 0.0 for b in probes.keys()}
    sum_yhat = {b: 0.0 for b in probes.keys()}
    sum_y2 = {b: 0.0 for b in probes.keys()}
    sum_yhat2 = {b: 0.0 for b in probes.keys()}
    sum_yyhat = {b: 0.0 for b in probes.keys()}
    sum_abs_err = {b: 0.0 for b in probes.keys()}
    
    proteins_processed = 0
    
    for seq in tqdm(sequences, desc="Evaluating probes"):
        if not all(aa in AA_TO_IDX for aa in seq):
            continue
        if len(seq) < 10:
            continue
        
        L = len(seq)
        
        try:
            outputs, collector = run_and_collect_z(model, tokenizer, device, seq)
        except Exception as e:
            continue
        
        final_positions = outputs.positions[-1, 0]
        distances = compute_ca_distances(final_positions)
        
        all_pairs = [(i, j) for i in range(L) for j in range(L) if i != j]
        
        if len(all_pairs) > n_pairs_per_protein:
            sampled_indices = np.random.choice(len(all_pairs), n_pairs_per_protein, replace=False)
            sampled_pairs = [all_pairs[k] for k in sampled_indices]
        else:
            sampled_pairs = all_pairs
        
        for block_idx in probes.keys():
            z_np = collector.z_blocks[block_idx][0].cpu().numpy()
            
            X_batch = np.array([z_np[i, j] for i, j in sampled_pairs], dtype=np.float64)
            y_batch = np.array([distances[i, j].item() for i, j in sampled_pairs], dtype=np.float64)
            
            y_pred = probes[block_idx].predict(X_batch)
            
            # Accumulate statistics
            sum_y[block_idx] += np.sum(y_batch)
            sum_yhat[block_idx] += np.sum(y_pred)
            sum_y2[block_idx] += np.sum(y_batch ** 2)
            sum_yhat2[block_idx] += np.sum(y_pred ** 2)
            sum_yyhat[block_idx] += np.sum(y_batch * y_pred)
            sum_abs_err[block_idx] += np.sum(np.abs(y_batch - y_pred))
            n_samples[block_idx] += len(y_batch)
        
        proteins_processed += 1
        
        del outputs, collector
        torch.cuda.empty_cache()
        
        if max_proteins and proteins_processed >= max_proteins:
            break
    
    print(f"\nEvaluated on {proteins_processed} proteins")
    
    # Compute final metrics
    results = []
    for block in sorted(probes.keys()):
        n = n_samples[block]
        if n == 0:
            continue
        
        # MAE
        mae = sum_abs_err[block] / n
        
        # R² = 1 - SS_res / SS_tot
        # SS_res = sum((y - yhat)²) = sum(y²) - 2*sum(y*yhat) + sum(yhat²)
        # SS_tot = sum((y - y_mean)²) = sum(y²) - n * y_mean²
        y_mean = sum_y[block] / n
        ss_res_val = sum_y2[block] - 2 * sum_yyhat[block] + sum_yhat2[block]
        ss_tot_val = sum_y2[block] - n * y_mean ** 2
        r2 = 1 - ss_res_val / ss_tot_val if ss_tot_val > 0 else 0.0
        
        # Correlation using the formula:
        # r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x²) - sum(x)²) * (n*sum(y²) - sum(y)²))
        numer = n * sum_yyhat[block] - sum_y[block] * sum_yhat[block]
        denom1 = n * sum_y2[block] - sum_y[block] ** 2
        denom2 = n * sum_yhat2[block] - sum_yhat[block] ** 2
        corr = numer / np.sqrt(denom1 * denom2) if denom1 > 0 and denom2 > 0 else 0.0
        
        results.append({
            'block': block,
            'r2': r2,
            'mae': mae,
            'correlation': corr,
            'n_samples': n,
            'r2_train': probes[block].r2_train,
        })
    
    return pd.DataFrame(results)


def save_probes(probes: Dict[int, DistanceProbe], path: str):
    """Save probes to disk."""
    data = {
        block: {
            'weights': p.weights,
            'bias': p.bias,
            'block_idx': p.block_idx,
            'r2_train': p.r2_train,
        }
        for block, p in probes.items()
    }
    torch.save(data, path)
    print(f"Saved {len(probes)} probes to {path}")


def load_probes(path: str) -> Dict[int, DistanceProbe]:
    """Load probes from disk."""
    data = torch.load(path, weights_only=False)
    probes = {
        block: DistanceProbe(**pdata)
        for block, pdata in data.items()
    }
    print(f"Loaded {len(probes)} probes from {path}")
    return probes


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_probe_results(eval_df: pd.DataFrame, output_dir: str, prefix: str = ""):
    """Plot probe evaluation results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # R² by block
    ax = axes[0]
    ax.bar(eval_df['block'], eval_df['r2'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Distance Probe R² by Block', fontsize=13)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3, axis='y')
    
    # MAE by block
    ax = axes[1]
    ax.bar(eval_df['block'], eval_df['mae'], color='coral', alpha=0.7)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('MAE (Å)', fontsize=12)
    ax.set_title('Distance Probe MAE by Block', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    
    # Correlation by block
    ax = axes[2]
    ax.bar(eval_df['block'], eval_df['correlation'], color='forestgreen', alpha=0.7)
    ax.set_xlabel('Block', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Distance Probe Correlation by Block', fontsize=13)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = f'{prefix}probe_evaluation.png' if prefix else 'probe_evaluation.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"Saved plot to {output_dir}/{filename}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train distance probes on ESMFold z representations")
    parser.add_argument('--probing_dataset', type=str, default='data/fixed.csv',
                        help='Path to probing dataset CSV with train/test splits')
    parser.add_argument('--output', type=str, default='probing_results',
                        help='Output directory')
    parser.add_argument('--n_pairs_per_protein', type=int, default=500,
                        help='Number of residue pairs to sample per protein')
    parser.add_argument('--max_train_proteins', type=int, default=None,
                        help='Max proteins to use for training (None = all)')
    parser.add_argument('--max_test_proteins', type=int, default=50,
                        help='Max proteins to use for testing')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge regularization strength')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    
    # Load dataset
    print(f"\nLoading dataset from {args.probing_dataset}...")
    df = pd.read_csv(args.probing_dataset)
    
    # Show composition
    if 'split' in df.columns and 'label' in df.columns:
        print("\nDataset composition:")
        print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
    
    # Get train/test sequences
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
    else:
        # Random split if no split column
        train_df = df.sample(frac=0.8, random_state=args.seed)
        test_df = df.drop(train_df.index)
    
    train_seqs = train_df['sequence'].tolist()
    test_seqs = test_df['sequence'].tolist()
    
    # Filter valid sequences
    train_seqs = [s for s in train_seqs if all(aa in AA_TO_IDX for aa in s)]
    test_seqs = [s for s in test_seqs if all(aa in AA_TO_IDX for aa in s)]
    
    print(f"\nTrain sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")
    
    # =========================================================================
    # TRAIN PROBES (ONLINE - MEMORY EFFICIENT)
    # =========================================================================
    print("\n" + "="*60)
    print("TRAINING PROBES (ONLINE)")
    print("="*60)
    
    probes = train_probes_online(
        sequences=train_seqs,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_pairs_per_protein=args.n_pairs_per_protein,
        alpha=args.alpha,
        max_proteins=args.max_train_proteins,
    )
    
    # Save probes
    save_probes(probes, os.path.join(args.output, 'distance_probes.pt'))
    
    # Training results (from the solve step)
    train_results = []
    for block, probe in probes.items():
        train_results.append({
            'block': block,
            'r2_train': probe.r2_train,
        })
    train_eval_df = pd.DataFrame(train_results)
    train_eval_df.to_csv(os.path.join(args.output, 'train_evaluation.csv'), index=False)
    
    print("\nTraining R² by block:")
    print(train_eval_df.to_string(index=False))
    
    # =========================================================================
    # EVALUATE ON TEST SET (ONLINE - MEMORY EFFICIENT)
    # =========================================================================
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET (ONLINE)")
    print("="*60)
    
    test_eval_df = evaluate_probes_online(
        probes=probes,
        sequences=test_seqs,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_pairs_per_protein=args.n_pairs_per_protein,
        max_proteins=args.max_test_proteins,
    )
    
    test_eval_df.to_csv(os.path.join(args.output, 'test_evaluation.csv'), index=False)
    
    print("\nTest Results:")
    print(test_eval_df.to_string(index=False))
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_probe_results(test_eval_df, args.output, prefix="test_")
    
    # Train vs test plot
    if 'r2_train' in test_eval_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        blocks = test_eval_df['block'].values
        width = 0.35
        ax.bar(blocks - width/2, test_eval_df['r2_train'], width, label='Train', color='steelblue', alpha=0.7)
        ax.bar(blocks + width/2, test_eval_df['r2'], width, label='Test', color='coral', alpha=0.7)
        ax.set_xlabel('Block', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('Train vs Test R² by Block', fontsize=13)
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'train_vs_test_r2.png'), dpi=150)
        plt.close()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    best_block = test_eval_df.loc[test_eval_df['r2'].idxmax()]
    print(f"\nBest block: {int(best_block['block'])}")
    print(f"  Test R²: {best_block['r2']:.4f}")
    print(f"  Test MAE: {best_block['mae']:.2f}Å")
    print(f"  Test Correlation: {best_block['correlation']:.4f}")
    
    print(f"\nResults saved to {args.output}/")
    print("  - distance_probes.pt")
    print("  - train_evaluation.csv")
    print("  - test_evaluation.csv")
    print("  - test_probe_evaluation.png")
    print("  - train_vs_test_r2.png")


if __name__ == '__main__':
    main()