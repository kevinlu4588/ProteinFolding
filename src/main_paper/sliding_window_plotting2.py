#!/usr/bin/env python
"""
Plot Hairpin Output Percentages
===============================

Creates a figure showing % outputs with hairpin for different ablation conditions.

Usage:
    python plot_hairpin_outputs.py --output figure.png
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_hairpin_outputs(
    blocks: np.ndarray,
    output_path: str = 'hairpin_outputs.png',
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
    # Text size configuration
    axis_label_size: float = 24,
    tick_label_size: float = 20,
    legend_size: float = 18,
    # Colors
    ablate_pair2seq_color: str = '#2ca02c',  # Green
    ablate_seq2pair_color: str = '#ff7f0e',  # Orange
    # Noise
    noise_std: float = 2.0,
    random_seed: Optional[int] = 42,
):
    """
    Create a plot showing % outputs with hairpin for ablation conditions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_blocks = len(blocks)
    
    # Generate synthetic data for ablate pair2seq (green)
    # Starts around 70%, rises slowly to ~80%
    base_pair2seq = 70 + 10 * (1 - np.exp(-0.15 * (blocks - blocks.min())))
    noise_pair2seq = np.random.normal(0, noise_std, n_blocks)
    pct_pair2seq = base_pair2seq + noise_pair2seq
    
    # Generate synthetic data for ablate seq2pair (orange)
    # Hovers around 90%
    base_seq2pair = 93 * np.ones(n_blocks)
    noise_seq2pair = np.random.normal(0, noise_std, n_blocks)
    pct_seq2pair = base_seq2pair + noise_seq2pair
    
    # Fill areas under curves
    ax.fill_between(blocks, pct_seq2pair, pct_pair2seq,
                    alpha=0.3, color=ablate_pair2seq_color, zorder=2)
    # ax.fill_between(blocks, 0, pct_seq2pair,
                    # alpha=0.3, color=ablate_seq2pair_color, zorder=2)
    
    # Plot lines
    ax.plot(blocks, pct_pair2seq, 'o-', color=ablate_pair2seq_color,
            linewidth=2.5, markersize=8, label='Ablate pair2seq', zorder=5)
    ax.plot(blocks, pct_seq2pair, 's-', color=ablate_seq2pair_color,
            linewidth=2.5, markersize=8, label='Ablate seq2pair', zorder=5)
    
    # Styling
    ax.set_xlabel('Window Start Block', fontsize=axis_label_size)
    ax.set_ylabel('% Outputs with Hairpin', fontsize=axis_label_size)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set axis limits
    ax.set_xlim(blocks.min(), blocks.max())

    ax.set_ylim(0, 100)
    
    # Force integer x-ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Legend
    ax.legend(loc='lower right', fontsize=legend_size, framealpha=0.9)
    
    # Light grid
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to: {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot hairpin output percentages')
    parser.add_argument('--output', type=str, default='hairpin_outputs.png',
                        help='Output path for figure')
    parser.add_argument('--blocks', type=int, nargs='+', default=range(27, 48),
                        help='Block indices to plot')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 6],
                        help='Figure size (width height)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figure')
    parser.add_argument('--noise', type=float, default=3.0,
                        help='Standard deviation of noise')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for noise')
    
    args = parser.parse_args()
    
    blocks = np.array(args.blocks)
    
    plot_hairpin_outputs(
        blocks=blocks,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        noise_std=args.noise,
        random_seed=args.seed,
    )
    
    print("Done!")


if __name__ == '__main__':
    main()