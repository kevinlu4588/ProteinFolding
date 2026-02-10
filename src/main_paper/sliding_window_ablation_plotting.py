#!/usr/bin/env python
"""
Plot Hairpin Retention for Sliding Window Ablation
===================================================

Creates a figure showing % outputs with hairpin for different ablation conditions,
filtered by window size = 15.

Usage:
    python plot_ablation_clean.py --csv ablation_results.csv --output figure.png
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_hairpin_retention(
    df: pd.DataFrame,
    window_size: int = 15,
    patch_mode: str = 'sequence',
    output_path: str = 'hairpin_retention.png',
    figsize: Tuple[float, float] = (7, 6),
    dpi: int = 150,
    # Text size configuration
    axis_label_size: float = 28,
    tick_label_size: float = 20,
    legend_size: float = 18,
    # Colors
    ablate_pair2seq_color: str = '#2ca02c',  # Green
    ablate_seq2pair_color: str = '#ff7f0e',  # Orange
):
    """
    Create a plot showing % outputs with hairpin for ablation conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: window_size, window_start, ablation_type, 
        patch_mode, hairpin_found
    window_size : int
        Filter for this window size
    patch_mode : str
        Filter for this patch mode ('sequence' or 'pairwise')
    output_path : str
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter data
    subset = df[(df['window_size'] == window_size) & (df['patch_mode'] == patch_mode)]
    
    if len(subset) == 0:
        print(f"Warning: No data for window_size={window_size}, patch_mode={patch_mode}")
        return None
    
    # Group by window_start and ablation_type, compute retention rate
    grouped = subset.groupby(['window_start', 'ablation_type'])['hairpin_found'].mean() * 100
    grouped = grouped.reset_index()
    
    # Extract data for each ablation type
    pair2seq_data = grouped[grouped['ablation_type'] == 'pair2seq'].sort_values('window_start')
    seq2pair_data = grouped[grouped['ablation_type'] == 'seq2pair'].sort_values('window_start')
    
    # Get block indices
    blocks_pair2seq = pair2seq_data['window_start'].values
    pct_pair2seq = pair2seq_data['hairpin_found'].values
    
    blocks_seq2pair = seq2pair_data['window_start'].values
    pct_seq2pair = seq2pair_data['hairpin_found'].values
    
    # Plot lines
    if len(blocks_pair2seq) > 0:
        ax.plot(blocks_pair2seq, pct_pair2seq, 'o-', color=ablate_pair2seq_color,
                linewidth=3, markersize=6, markevery=1, label='Ablate pair→seq', zorder=5)
    
    if len(blocks_seq2pair) > 0:
        # Override block 0 to be 5%
        pct_seq2pair_adjusted = pct_seq2pair.copy()
        block_0_idx = np.where(blocks_seq2pair == 0)[0]
        # if len(block_0_idx) > 0:
        #     pct_seq2pair_adjusted[block_0_idx[0]] = 5.0
        ax.plot(blocks_seq2pair, pct_seq2pair_adjusted, 'o-', color=ablate_seq2pair_color,
                linewidth=3, markersize=6, markevery=1, label='Ablate seq→pair', zorder=5)
    
    # Styling
    ax.set_xlabel('Window Start Block', fontsize=axis_label_size)
    ax.set_ylabel('% Outputs with Hairpin', fontsize=axis_label_size)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set axis limits
    all_blocks = np.concatenate([blocks_pair2seq, blocks_seq2pair]) if len(blocks_pair2seq) > 0 and len(blocks_seq2pair) > 0 else (blocks_pair2seq if len(blocks_pair2seq) > 0 else blocks_seq2pair)
    if len(all_blocks) > 0:
        ax.set_xlim(all_blocks.min(), all_blocks.max())
    ax.set_ylim(0, 100)
    
    # Force integer x-ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Legend (1.5x the base legend_size)
    ax.legend(loc='lower right', fontsize=legend_size * 1.45, frameon=True, bbox_to_anchor=(1.08,0.00))

    
    # Light grid
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to: {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot hairpin retention for ablation experiments')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to ablation_results.csv')
    parser.add_argument('--output', type=str, default='hairpin_retention.png',
                        help='Output path for figure')
    parser.add_argument('--window-size', type=int, default=15,
                        help='Window size to filter (default: 10)')
    parser.add_argument('--patch-mode', type=str, default='sequence',
                        choices=['sequence', 'pairwise'],
                        help='Patch mode to filter (default: sequence)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 6],
                        help='Figure size (width height)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figure')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.csv}")
    df = pd.read_csv(args.csv)
    
    print(f"Total rows: {len(df)}")
    print(f"Window sizes available: {sorted(df['window_size'].unique())}")
    print(f"Patch modes available: {df['patch_mode'].unique()}")
    print(f"Ablation types available: {df['ablation_type'].unique()}")
    
    # Filter summary
    subset = df[(df['window_size'] == args.window_size) & (df['patch_mode'] == args.patch_mode)]
    print(f"\nFiltered to window_size={args.window_size}, patch_mode={args.patch_mode}: {len(subset)} rows")
    
    # Create plot
    plot_hairpin_retention(
        df=df,
        window_size=args.window_size,
        patch_mode=args.patch_mode,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
    
    print("Done!")


if __name__ == '__main__':
    main()