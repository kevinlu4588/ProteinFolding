"""
Charge DoM Projection Histograms
=================================

Loads pre-trained charge directions, projects test-set residue representations
onto them, and saves individual + grid histogram plots showing the separation
of positive, neutral, and negative residues.

Usage:
    python -m src.appendix.charge_projection_histograms \
        --directions models/charge_directions.pt \
        --probing_dataset data/probing_train_test.csv \
        --output results/charge_projections
"""

import argparse
import os
import sys

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.charge_dom_training import (
    load_directions,
    collect_representations_for_sequence,
    compute_charge_labels,
    AA_TO_IDX,
)

import warnings
warnings.filterwarnings("ignore", message=".*mmCIF.*")


BLOCKS_TO_PLOT = list(range(0, 48, 4))  # [0, 4, 8, ..., 44]

POS_COLOR = '#4a90d9'   # Blue
NEG_COLOR = '#d9534f'   # Red
NEU_COLOR = '#999999'   # Gray


def collect_projections(directions, sequences, model, tokenizer, device, blocks):
    """Collect projections for positive, negative, and neutral residues."""
    projections = {b: {'positive': [], 'negative': [], 'neutral': []} for b in blocks}

    for seq in tqdm(sequences, desc="Collecting projections"):
        if not all(aa in AA_TO_IDX for aa in seq):
            continue
        try:
            data = collect_representations_for_sequence(model, seq, tokenizer, device)
        except Exception:
            continue

        charges = compute_charge_labels(seq)
        for block in blocks:
            if block not in directions.s_directions:
                continue
            s = data['s_list'][block][0].numpy()
            s_dir = directions.s_directions[block]
            for i, charge in enumerate(charges):
                proj = float(np.dot(s[i], s_dir))
                if charge > 0:
                    projections[block]['positive'].append(proj)
                elif charge < 0:
                    projections[block]['negative'].append(proj)
                else:
                    projections[block]['neutral'].append(proj)

        del data
        torch.cuda.empty_cache()

    return projections


def _plot_block(ax, projections, block, show_legend=False):
    """Plot a single block histogram onto the given axes."""
    pos = np.array(projections[block]['positive'])
    neg = np.array(projections[block]['negative'])
    neu = np.array(projections[block]['neutral'])

    all_projs = np.concatenate([pos, neg, neu])
    center = all_projs.mean()
    pos_c, neg_c, neu_c = pos - center, neg - center, neu - center

    lo = min(pos_c.min(), neg_c.min(), neu_c.min())
    hi = max(pos_c.max(), neg_c.max(), neu_c.max())
    bins = np.linspace(lo, hi, 50)

    ax.hist(neg_c, bins=bins, alpha=0.55, color=NEG_COLOR, density=True, label='Negative')
    ax.hist(neu_c, bins=bins, alpha=0.30, color=NEU_COLOR, density=True, label='Neutral')
    ax.hist(pos_c, bins=bins, alpha=0.55, color=POS_COLOR, density=True, label='Positive')

    ax.axvline(x=neg_c.mean(), color=NEG_COLOR, linewidth=1.5)
    ax.axvline(x=pos_c.mean(), color=POS_COLOR, linewidth=1.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4)

    ax.set_title(f'Block {block}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Centered Projection', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.tick_params(labelsize=12)

    if show_legend:
        ax.legend(fontsize=11)


def plot_single_block(projections, block, output_dir):
    """Save an individual histogram for one block."""
    fig, ax = plt.subplots(figsize=(5, 4))
    _plot_block(ax, projections, block, show_legend=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'block_{block}.png'), dpi=150)
    plt.close()


def plot_grid(projections, blocks, output_dir):
    """Save a 3x4 grid of all block histograms."""
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 13))

    for idx, block in enumerate(blocks):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        _plot_block(ax, projections, block, show_legend=(idx == 0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'charge_projection_grid.png'), dpi=150)
    plt.close()
    print(f"Saved grid to {output_dir}/charge_projection_grid.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot charge DoM projection histograms for every 4th block',
    )
    parser.add_argument('--directions', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'models', 'charge_directions.pt'))
    parser.add_argument('--probing_dataset', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'probing_train_test.csv'))
    parser.add_argument('--output', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'results', 'charge_projections'))
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n_seqs', type=int, default=None,
                        help='Limit number of test sequences')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load directions
    print(f"Loading directions from {args.directions}...")
    directions = load_directions(args.directions)
    print(f"  Directions for {len(directions.s_directions)} blocks")

    # Load model
    print("Loading ESMFold model...")
    from transformers import EsmForProteinFolding, AutoTokenizer
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # Load test sequences
    print(f"Loading probing dataset from {args.probing_dataset}...")
    probing_df = pd.read_csv(args.probing_dataset)
    test_df = probing_df[probing_df['split'] == 'test']
    seq_col = 'FullChainSequence' if 'FullChainSequence' in test_df.columns else 'sequence'
    test_seqs = test_df[seq_col].tolist()
    test_seqs = [s for s in test_seqs if all(aa in AA_TO_IDX for aa in s)]

    if args.n_seqs is not None:
        test_seqs = test_seqs[:args.n_seqs]
    print(f"  Test sequences: {len(test_seqs)}")

    # Collect projections
    blocks = BLOCKS_TO_PLOT
    projections = collect_projections(
        directions, test_seqs, model, tokenizer, device, blocks,
    )

    # Save raw projections
    proj_path = os.path.join(args.output, 'projections.pt')
    torch.save(projections, proj_path)
    print(f"Saved projections to {proj_path}")

    # Individual plots
    print("Saving individual block plots...")
    individual_dir = os.path.join(args.output, 'individual')
    os.makedirs(individual_dir, exist_ok=True)
    for block in blocks:
        plot_single_block(projections, block, individual_dir)

    # Grid plot
    print("Saving grid plot...")
    plot_grid(projections, blocks, args.output)

    # Print summary
    print(f"\nProjection counts per block:")
    print(f"{'Block':>6}  {'Positive':>10}  {'Neutral':>10}  {'Negative':>10}")
    print("-" * 42)
    for block in blocks:
        p = projections[block]
        print(f"{block:>6}  {len(p['positive']):>10}  {len(p['neutral']):>10}  {len(p['negative']):>10}")

    print(f"\nDone. Results in {args.output}/")


if __name__ == '__main__':
    main()
