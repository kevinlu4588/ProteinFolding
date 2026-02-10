#!/usr/bin/env python
"""
Plot probe R² scores by block for train and test sets.

Creates two separate plots:
1. Train R² vs block index
2. Test R² vs block index

Usage:
    python plot_probe_scores.py --train_path path/to/train_evaluation.csv \
                                --test_path path/to/test_evaluation.csv \
                                --output plots/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# FONT SIZE CONFIGURATION - Edit these values to customize
# =============================================================================
FONT_SIZES = {
    'title': 24,
    'axis_label': 18,
    'tick_label': 16,
    'legend': 15,
    'legend_title': 18,
}

# =============================================================================
# PLOT CONFIGURATION
# =============================================================================
PLOT_CONFIG = {
    'figsize': (8, 4),
    'dpi': 300,
    'train_color': '#1a7a3a',  # Dark green
    'test_color': '#1a7a3a',   # Dark green
    'linewidth': 3.5,
    'markersize': 8,
    'fill_alpha': 0.3,
    'grid_alpha': 0.3,
}


def plot_r2_by_block(eval_df: pd.DataFrame, output_path: str, color: str, label: str):
    """
    Plot R² vs block index.
    
    X-axis: Block index
    Y-axis: R² - coefficient of determination for distance prediction
    """
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    
    # Sort by block index
    eval_df = eval_df.sort_values('block')
    blocks = eval_df['block'].values
    
    # Get R² values - handle different column names
    if 'r2' in eval_df.columns:
        r2_values = eval_df['r2'].values
    elif 'r2_train' in eval_df.columns:
        r2_values = eval_df['r2_train'].values
    else:
        raise ValueError("No R² column found. Expected 'r2' or 'r2_train'.")
    
    # Compute confidence interval (using standard error if available, otherwise estimate)
    if 'r2_std' in eval_df.columns:
        r2_std = eval_df['r2_std'].values
    elif 'n_samples' in eval_df.columns:
        # Rough approximation of SE for R²
        n = eval_df['n_samples'].values
        r2_std = np.sqrt((1 - r2_values**2) / (n - 2 + 1e-8)) * 1.96
    else:
        # Default: use 5% of R² value as approximate CI
        r2_std = np.abs(r2_values) * 0.05
    
    # Confidence interval bounds
    ci_lower = r2_values - r2_std
    ci_upper = r2_values + r2_std
    
    # Plot confidence interval as shaded region
    ax.fill_between(blocks, ci_lower, ci_upper,
                    alpha=PLOT_CONFIG['fill_alpha'],
                    color=color)
    
    # Plot line
    ax.plot(blocks, r2_values, 'o-',
            color=color,
            linewidth=PLOT_CONFIG['linewidth'],
            markersize=PLOT_CONFIG['markersize'],
            label=label)
    
    # Labels and formatting
    ax.set_xlabel('Block Index', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('R²', fontsize=FONT_SIZES['axis_label'])
    
    # Tick labels
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick_label'])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(alpha=PLOT_CONFIG['grid_alpha'])
    
    # Legend
    ax.legend(loc='best',
              fontsize=FONT_SIZES['legend'],
              title_fontsize=FONT_SIZES['legend_title'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG['dpi'])
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot probe R² scores by block')
    parser.add_argument('--train_path', type=str,
                        default="probing_results/train_evaluation.csv",
                        help='Path to train_evaluation.csv')
    parser.add_argument('--test_path', type=str,
                        default="probing_results/test_evaluation.csv",
                        help='Path to test_evaluation.csv')
    parser.add_argument('--output', type=str, default='probe_plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Plot 1: Train R²
    if os.path.exists(args.train_path):
        print(f"Loading train evaluation from {args.train_path}...")
        train_df = pd.read_csv(args.train_path)
        print(f"Loaded {len(train_df)} rows")
        
        print("Generating train R² plot...")
        plot_r2_by_block(
            train_df, 
            os.path.join(args.output, 'train_r2_by_block.png'),
            color=PLOT_CONFIG['train_color'],
            label='Train R²'
        )
    else:
        print(f"Warning: Train evaluation file not found at {args.train_path}")
    
    # Plot 2: Test R²
    if os.path.exists(args.test_path):
        print(f"\nLoading test evaluation from {args.test_path}...")
        test_df = pd.read_csv(args.test_path)
        print(f"Loaded {len(test_df)} rows")
        
        print("Generating test R² plot...")
        plot_r2_by_block(
            test_df,
            os.path.join(args.output, 'test_r2_by_block.png'),
            color=PLOT_CONFIG['test_color'],
            label='Test R²'
        )
    else:
        print(f"Warning: Test evaluation file not found at {args.test_path}")
    
    print(f"\nDone! Plots saved to {args.output}/")


if __name__ == '__main__':
    main()