"""
ESMFold Block Patching Analysis - Standalone Plotting Script
=============================================================
Generates summary plots from experiment results.

Usage:
    python plot_block_patching_results.py --results results/block_patching_results.csv --output_dir results/
"""

import argparse
import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION - Edit these to iterate quickly
# =============================================================================

# Font sizes for formation rate lines plot
FORMATION_TITLE_FONTSIZE = 28
FORMATION_LABEL_FONTSIZE = 18
FORMATION_TICK_FONTSIZE = 18
FORMATION_LEGEND_FONTSIZE = 18

# Y-axis limits for formation rate plot
FORMATION_YLIM = (0.0, 60.0)
plt.rcParams['font.family'] = 'Trebuchet MS'
# Figure size
FORMATION_FIGSIZE = (14, 4)

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_formation_rate_lines(results_df: pd.DataFrame, output_dir: str):
    """
    Line plot showing hairpin formation rate by block for sequence and pairwise (touch) modes,
    with filled area under the curves.
    """
    fig, ax = plt.subplots(figsize=FORMATION_FIGSIZE)
    
    # Slightly muted, professional colors
    # colors = {"sequence": "#E07B53", "pairwise": "#5BA08C"}  # Muted orange, Muted teal
    colors = {"sequence": "#d95f02", "pairwise": "#1b9e77"}  # Burnt orange, Teal

    # Track block range for xlim
    all_blocks = []
    
    # Plot sequence mode
    seq_df = results_df[results_df["patch_mode"] == "sequence"]
    if len(seq_df) > 0:
        fraction_per_block = seq_df.groupby("block_idx")["hairpin_found"].mean()
        pct_per_block = 100.0 * fraction_per_block

        all_blocks.extend(pct_per_block.index.tolist())
        ax.plot(pct_per_block.index, pct_per_block.values,
                label="Sequence Patch",
                color=colors["sequence"], linewidth=2.5)
        ax.fill_between(pct_per_block.index, pct_per_block.values,
                        alpha=0.25, color=colors["sequence"])
    
    # Plot pairwise (touch) mode only
    pair_df = results_df[(results_df["patch_mode"] == "pairwise") & 
                         (results_df["patch_mask_mode"] == "intra")]
    if len(pair_df) > 0:
        fraction_per_block = pair_df.groupby("block_idx")["hairpin_found"].mean()
        pct_per_block = 100.0 * fraction_per_block

        all_blocks.extend(pct_per_block.index.tolist())
        ax.plot(pct_per_block.index, pct_per_block.values,
                label="Pairwise Patch",
                color=colors["pairwise"], linewidth=2.5)
        ax.fill_between(pct_per_block.index, pct_per_block.values,
                        alpha=0.25, color=colors["pairwise"])

    
    # Set x limits to remove whitespace
    if all_blocks:
        ax.set_xlim(min(all_blocks), max(all_blocks))
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # # Subtle horizontal grid only
    # ax.xaxis.grid(True, linestyle='-', alpha=0.4, color='gray')
    # ax.yaxis.grid(False)
    # ax.set_axisbelow(True)
    
    ax.set_xlabel("Block Index", fontsize=FORMATION_LABEL_FONTSIZE)
    ax.set_ylabel("% of Outputs with Hairpin", fontsize=FORMATION_LABEL_FONTSIZE)
    ax.set_ylim(FORMATION_YLIM)
    ax.set_yticks(np.arange(FORMATION_YLIM[0], FORMATION_YLIM[1] + 0.01, 20))

    ax.tick_params(axis='both', labelsize=FORMATION_TICK_FONTSIZE)
    ax.legend(loc='center right', fontsize=FORMATION_LEGEND_FONTSIZE, frameon=False)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "formation_rate_lines.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_all_plots(results_df: pd.DataFrame, output_dir: str, include_helix: bool = False):
    """Generate all summary plots."""
    print("\nGenerating summary plots...")
    
    # Hairpin formation plots
    plot_formation_rate_lines(results_df, output_dir)

    print("\nAll plots generated!")


def print_summary_statistics(results_df: pd.DataFrame, include_helix: bool = False):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nTotal experiments: {len(results_df)}")
    print(f"Unique cases: {results_df['case_idx'].nunique()}")
    print(f"Blocks tested: {results_df['block_idx'].nunique()}")
    
    mask_modes = results_df['patch_mask_mode'].unique()
    
    for mask_mode in mask_modes:
        print(f"\n{'='*60}")
        print(f"MASK MODE: {mask_mode.upper()}")
        print("="*60)
        
        mask_df = results_df[results_df['patch_mask_mode'] == mask_mode]
        
        print("\nFormation rates by patch mode:")
        rates = mask_df.groupby('patch_mode')['hairpin_found'].mean()
        for mode, rate in rates.items():
            n = len(mask_df[mask_df['patch_mode'] == mode])
            print(f"  {mode:12s}: {rate:.1%} ({int(rate*n)}/{n})")
        
        print("\nMean magnitude when hairpin found:")
        for mode in ["sequence", "pairwise", "both"]:
            mode_df = mask_df[(mask_df['patch_mode'] == mode) & 
                              (mask_df['hairpin_found'] == True)]
            if len(mode_df) > 0:
                mean_mag = mode_df['magnitude'].mean()
                print(f"  {mode:12s}: {mean_mag:.3f}")
            else:
                print(f"  {mode:12s}: N/A (no hairpins found)")
        
        print("\nFormation rate by block (averaged across patch modes):")
        block_rates = mask_df.groupby('block_idx')['hairpin_found'].mean()
        if len(block_rates) > 0:
            print(f"  Min: Block {block_rates.idxmin()} ({block_rates.min():.1%})")
            print(f"  Max: Block {block_rates.idxmax()} ({block_rates.max():.1%})")
        
        # Alpha helix statistics
        if include_helix and 'patched_helix_pct' in mask_df.columns:
            print("\n" + "-"*60)
            print(f"ALPHA HELIX CONTENT ANALYSIS ({mask_mode.upper()})")
            print("-"*60)
            
            # Original helix content (should be same across all rows for a case)
            orig_helix = mask_df.groupby('case_idx')['original_helix_pct'].first()
            print(f"\nOriginal structure helix content:")
            print(f"  Mean: {orig_helix.mean():.1f}%")
            print(f"  Range: {orig_helix.min():.1f}% - {orig_helix.max():.1f}%")
            
            print("\nMean helix change by patch mode:")
            for mode in ["sequence", "pairwise", "both"]:
                mode_df = mask_df[mask_df['patch_mode'] == mode]
                if len(mode_df) > 0 and mode_df['helix_absolute_change'].notna().any():
                    mean_change = mode_df['helix_absolute_change'].mean()
                    std_change = mode_df['helix_absolute_change'].std()
                    print(f"  {mode:12s}: {mean_change:+.2f} ± {std_change:.2f} pp")
                else:
                    print(f"  {mode:12s}: N/A")
            
            print("\nHelix change by block (averaged across patch modes):")
            block_helix = mask_df.groupby('block_idx')['helix_absolute_change'].mean()
            if len(block_helix) > 0 and block_helix.notna().any():
                print(f"  Most helix loss:    Block {block_helix.idxmin()} ({block_helix.min():+.2f} pp)")
                print(f"  Most helix gain:    Block {block_helix.idxmax()} ({block_helix.max():+.2f} pp)")
            
            # Correlation between hairpin and helix
            valid_df = mask_df.dropna(subset=['helix_absolute_change'])
            if len(valid_df) > 0:
                corr = valid_df['hairpin_found'].astype(float).corr(valid_df['helix_absolute_change'])
                print(f"\nCorrelation (hairpin found vs helix change): {corr:.3f}")
    
    # Compare mask modes
    if len(mask_modes) > 1:
        print("\n" + "="*60)
        print("COMPARISON ACROSS MASK MODES")
        print("="*60)
        
        comparison = results_df.groupby(['patch_mask_mode', 'patch_mode'])['hairpin_found'].mean().unstack()
        print("\nFormation rates (rows=mask mode, cols=patch mode):")
        print(comparison.to_string())


def main():
    parser = argparse.ArgumentParser(description="Plot ESMFold patching experiment results")
    parser.add_argument(
        "--results", type=str, default="/share/u/kevin/ProteinFolding/base_patching_results/block_patching_results.parquet",
        help="Path to results parquet file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots/base_patching",
        help="Output directory for plots (default: same as results)"
    )
    parser.add_argument(
        "--include_helix", action="store_true",
        help="Include alpha helix analysis plots"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results_df = pd.read_parquet(args.results)
    print(f"Loaded {len(results_df)} rows")
    
    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.results)
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect helix data
    include_helix = args.include_helix or 'patched_helix_pct' in results_df.columns
    
    # Generate outputs
    print_summary_statistics(results_df, include_helix=include_helix)
    generate_all_plots(results_df, output_dir, include_helix=include_helix)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()