"""
Information Flow Plotting Script
================================
Plots seq2pair vs pair2seq relative contributions for combined test set.

Usage:
    python plot_information_flow.py --stats information_flow_results/information_flow_stats.parquet --output_dir plots/
"""

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 18

FIGSIZE = (12, 6)

SMOOTHING_WINDOW = 3

# Colors
COLOR_SEQ2PAIR = '#d95f02'   # Burnt orange
COLOR_PAIR2SEQ = '#1b9e77'   # Teal/green

FILL_ALPHA = 0.25
LINE_WIDTH = 2.5

# =============================================================================
# Helper Functions
# =============================================================================

def smooth_series(series, window=SMOOTHING_WINDOW):
    """Apply sliding window smoothing to a pandas Series."""
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def normalize_series(series):
    """Min-max normalize a series to [0, 1]."""
    return (series - series.min()) / (series.max() - series.min() + 1e-10)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_information_flow_combined_testset(
    stats_df: pd.DataFrame,
    output_dir: str,
    smoothing_window: int = SMOOTHING_WINDOW,
    normalize: bool = True,
    use_fill: bool = True,
):
    """
    Plot seq2pair vs pair2seq contributions for the combined test set.
    
    Shows both pathways on one plot with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Get mean and std by block (across all sequences, regardless of label)
    seq2pair_mean = stats_df.groupby('block')['seq2pair_relative_global'].mean()
    seq2pair_std = stats_df.groupby('block')['seq2pair_relative_global'].std()
    seq2pair_sem = seq2pair_std / np.sqrt(stats_df.groupby('block')['seq_idx'].nunique())
    
    pair2seq_mean = stats_df.groupby('block')['pair2seq_relative_global'].mean()
    pair2seq_std = stats_df.groupby('block')['pair2seq_relative_global'].std()
    pair2seq_sem = pair2seq_std / np.sqrt(stats_df.groupby('block')['seq_idx'].nunique())
    
    # Smooth
    seq2pair_smooth = smooth_series(seq2pair_mean, smoothing_window)
    pair2seq_smooth = smooth_series(pair2seq_mean, smoothing_window)
    seq2pair_sem_smooth = smooth_series(seq2pair_sem, smoothing_window)
    pair2seq_sem_smooth = smooth_series(pair2seq_sem, smoothing_window)
    
    # Normalize if requested
    if normalize:
        # For normalized version, we scale the mean and use relative CI
        seq2pair_min, seq2pair_max = seq2pair_smooth.min(), seq2pair_smooth.max()
        pair2seq_min, pair2seq_max = pair2seq_smooth.min(), pair2seq_smooth.max()
        
        seq2pair_norm = normalize_series(seq2pair_smooth)
        pair2seq_norm = normalize_series(pair2seq_smooth)
        
        # Scale SEM proportionally
        seq2pair_sem_norm = seq2pair_sem_smooth / (seq2pair_max - seq2pair_min + 1e-10)
        pair2seq_sem_norm = pair2seq_sem_smooth / (pair2seq_max - pair2seq_min + 1e-10)
        
        seq2pair_plot = seq2pair_norm
        pair2seq_plot = pair2seq_norm
        seq2pair_ci = seq2pair_sem_norm * 1.96  # 95% CI
        pair2seq_ci = pair2seq_sem_norm * 1.96
        
        ylabel = 'Scaled Relative Contribution'
    else:
        seq2pair_plot = seq2pair_smooth
        pair2seq_plot = pair2seq_smooth
        seq2pair_ci = seq2pair_sem_smooth * 1.96
        pair2seq_ci = pair2seq_sem_smooth * 1.96
        
        ylabel = 'Relative Contribution'
    
    blocks = seq2pair_plot.index
    
    # Plot seq2pair
    if use_fill:
        ax.fill_between(blocks, 0, seq2pair_plot.values,
                        alpha=FILL_ALPHA, color=COLOR_SEQ2PAIR)
    ax.fill_between(blocks,
                    seq2pair_plot - seq2pair_ci,
                    seq2pair_plot + seq2pair_ci,
                    alpha=FILL_ALPHA * 0.8, color=COLOR_SEQ2PAIR)
    ax.plot(blocks, seq2pair_plot.values,
            color=COLOR_SEQ2PAIR, linewidth=LINE_WIDTH, label='Seq→Pair')
    
    # Plot pair2seq
    if use_fill:
        ax.fill_between(blocks, 0, pair2seq_plot.values,
                        alpha=FILL_ALPHA, color=COLOR_PAIR2SEQ)
    ax.fill_between(blocks,
                    pair2seq_plot - pair2seq_ci,
                    pair2seq_plot + pair2seq_ci,
                    alpha=FILL_ALPHA * 0.8, color=COLOR_PAIR2SEQ)
    ax.plot(blocks, pair2seq_plot.values,
            color=COLOR_PAIR2SEQ, linewidth=LINE_WIDTH, label='Pair→Seq')
    
    ax.set_xlabel('Block', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title('Information Flow in Folding Trunk', fontsize=TITLE_FONTSIZE, pad=12)
    ax.set_xlim(0, 47)
    if normalize:
        ax.set_ylim(0, 1.1)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.legend(loc='best', fontsize=LEGEND_FONTSIZE)
    ax.grid(alpha=0.3)
    
    norm_str = '_normalized' if normalize else ''
    fill_str = '_filled' if use_fill else ''
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'information_flow{norm_str}{fill_str}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: information_flow{norm_str}{fill_str}.png")


def plot_information_flow_multi_smoothing(
    stats_df: pd.DataFrame,
    output_dir: str,
    smoothing_windows: list = [1, 2, 3, 5],
    normalize: bool = True,
):
    """
    Create the same plot with different smoothing windows for comparison.
    """
    for window in smoothing_windows:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        # Get mean and SEM by block
        seq2pair_mean = stats_df.groupby('block')['seq2pair_relative_global'].mean()
        seq2pair_std = stats_df.groupby('block')['seq2pair_relative_global'].std()
        n_per_block = stats_df.groupby('block')['seq_idx'].nunique()
        seq2pair_sem = seq2pair_std / np.sqrt(n_per_block)
        
        pair2seq_mean = stats_df.groupby('block')['pair2seq_relative_global'].mean()
        pair2seq_std = stats_df.groupby('block')['pair2seq_relative_global'].std()
        pair2seq_sem = pair2seq_std / np.sqrt(n_per_block)
        
        # Smooth
        seq2pair_smooth = smooth_series(seq2pair_mean, window)
        pair2seq_smooth = smooth_series(pair2seq_mean, window)
        seq2pair_sem_smooth = smooth_series(seq2pair_sem, window)
        pair2seq_sem_smooth = smooth_series(pair2seq_sem, window)
        
        if normalize:
            seq2pair_range = seq2pair_smooth.max() - seq2pair_smooth.min()
            pair2seq_range = pair2seq_smooth.max() - pair2seq_smooth.min()
            
            seq2pair_plot = normalize_series(seq2pair_smooth)
            pair2seq_plot = normalize_series(pair2seq_smooth)
            seq2pair_ci = (seq2pair_sem_smooth / (seq2pair_range + 1e-10)) * 1.96
            pair2seq_ci = (pair2seq_sem_smooth / (pair2seq_range + 1e-10)) * 1.96
            ylabel = 'Scaled Relative Contribution'
        else:
            seq2pair_plot = seq2pair_smooth
            pair2seq_plot = pair2seq_smooth
            seq2pair_ci = seq2pair_sem_smooth * 1.96
            pair2seq_ci = pair2seq_sem_smooth * 1.96
            ylabel = 'Relative Contribution'
        
        blocks = seq2pair_plot.index
        
        # Plot with filled areas and confidence bands
        ax.fill_between(blocks, 0, seq2pair_plot.values,
                        alpha=FILL_ALPHA, color=COLOR_SEQ2PAIR)
        ax.fill_between(blocks, seq2pair_plot - seq2pair_ci, seq2pair_plot + seq2pair_ci,
                        alpha=FILL_ALPHA * 0.8, color=COLOR_SEQ2PAIR)
        ax.plot(blocks, seq2pair_plot.values,
                color=COLOR_SEQ2PAIR, linewidth=LINE_WIDTH, label='Seq→Pair')
        
        ax.fill_between(blocks, 0, pair2seq_plot.values,
                        alpha=FILL_ALPHA, color=COLOR_PAIR2SEQ)
        ax.fill_between(blocks, pair2seq_plot - pair2seq_ci, pair2seq_plot + pair2seq_ci,
                        alpha=FILL_ALPHA * 0.8, color=COLOR_PAIR2SEQ)
        ax.plot(blocks, pair2seq_plot.values,
                color=COLOR_PAIR2SEQ, linewidth=LINE_WIDTH, label='Pair→Seq')
        
        ax.set_xlabel('Block', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
        ax.set_title(f'Information Flow (smoothing={window})', fontsize=TITLE_FONTSIZE, pad=12)
        ax.set_xlim(0, 47)
        if normalize:
            ax.set_ylim(0, 1.1)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax.legend(loc='best', fontsize=LEGEND_FONTSIZE)
        ax.grid(alpha=0.3)
        
        n_seqs = stats_df['seq_idx'].nunique()
        ax.text(0.98, 0.02, f'n={n_seqs}', transform=ax.transAxes,
                fontsize=TICK_FONTSIZE, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        norm_str = '_normalized' if normalize else ''
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'information_flow{norm_str}_window{window}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: information_flow{norm_str}_window{window}.png")


def print_summary(stats_df: pd.DataFrame):
    """Print summary statistics for combined test set."""
    print("\n" + "=" * 70)
    print("INFORMATION FLOW SUMMARY (Combined Test Set)")
    print("=" * 70)
    
    n_seqs = stats_df['seq_idx'].nunique()
    print(f"\nTotal sequences: {n_seqs}")
    
    if 'label' in stats_df.columns:
        print(f"Composition: {stats_df.groupby('label')['seq_idx'].nunique().to_dict()}")
    
    for metric, name in [
        ('seq2pair_relative_global', 'Seq→Pair'),
        ('pair2seq_relative_global', 'Pair→Seq'),
    ]:
        early = stats_df[stats_df['block'] < 16][metric]
        mid = stats_df[(stats_df['block'] >= 16) & (stats_df['block'] < 32)][metric]
        late = stats_df[stats_df['block'] >= 32][metric]
        
        print(f"\n{name}:")
        print(f"  Early (0-15):  {early.mean():.4f} ± {early.std():.4f}")
        print(f"  Mid (16-31):   {mid.mean():.4f} ± {mid.std():.4f}")
        print(f"  Late (32-47):  {late.mean():.4f} ± {late.std():.4f}")


def generate_all_plots(stats_df: pd.DataFrame, output_dir: str, smoothing_windows: list):
    """Generate all plots."""
    print("\n--- Generating plots ---")
    
    # Main plots (normalized and non-normalized, with and without fill)
    plot_information_flow_combined_testset(stats_df, output_dir, normalize=True, use_fill=True)
    plot_information_flow_combined_testset(stats_df, output_dir, normalize=True, use_fill=False)
    plot_information_flow_combined_testset(stats_df, output_dir, normalize=False, use_fill=True)
    plot_information_flow_combined_testset(stats_df, output_dir, normalize=False, use_fill=False)
    
    # Multi-smoothing comparison
    plot_information_flow_multi_smoothing(stats_df, output_dir, smoothing_windows, normalize=True)


def main():
    parser = argparse.ArgumentParser(description="Plot information flow relative contributions")
    parser.add_argument("--stats", type=str,
                        default="information_flow_results/information_flow_stats.parquet",
                        help="Path to stats parquet/csv")
    parser.add_argument("--output_dir", type=str,
                        default="information_flow_plots",
                        help="Output directory")
    parser.add_argument("--smoothing_windows", type=int, nargs='+',
                        default=[1, 2, 3, 5],
                        help="Smoothing windows for multi-window plots")
    args = parser.parse_args()
    
    # Load stats
    print(f"Loading {args.stats}...")
    if args.stats.endswith('.parquet'):
        stats_df = pd.read_parquet(args.stats)
    else:
        stats_df = pd.read_csv(args.stats)
    
    print(f"Loaded {len(stats_df)} rows")
    print(f"Sequences: {stats_df['seq_idx'].nunique()}")
    
    if 'label' in stats_df.columns:
        print(f"Labels: {stats_df['label'].value_counts().to_dict()}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print summary
    print_summary(stats_df)
    
    # Generate plots
    generate_all_plots(stats_df, args.output_dir, args.smoothing_windows)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()