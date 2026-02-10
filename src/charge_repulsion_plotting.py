#!/usr/bin/env python
"""
Hairpin Disruption Plotting Script
==================================

Generates plots from hairpin disruption experiment results.
Main focus: mean distance change vs block position.

Usage:
    python charge_disruption_plots.py \
        --results results_disp_ws/hairpin_disruption_results.parquet \
        --output charge_disruption_plots/
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NUM_BLOCKS = 48


def load_results(path: str) -> pd.DataFrame:
    """Load results from parquet or CSV."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def plot_charge_mode_comparison_ws10(
    results_df: pd.DataFrame,
    induction_df: pd.DataFrame,
    output_dir: str,
):
    from matplotlib.gridspec import GridSpec

    # --- Disruption data: both_positive, both_negative ---
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    ws_data = interventions[
        (interventions['window_size'] == 15) & (interventions['magnitude'] == 0.5)
    ]

    # --- Induction data: pos_neg, neg_pos ---
    ind_interventions = induction_df[induction_df['block_set'] != 'baseline']
    ind_data = ind_interventions[
        (ind_interventions['window_size'] == 15) & (ind_interventions['magnitude'] == 3.0)
    ]

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    axes = [ax_top, ax_bot]

    # Plot pos-pos and neg-neg from disruption data
    for charge_mode, color, label in [
        ('both_positive', 'blue', 'Pos-Pos'),
        ('both_negative', 'red', 'Neg-Neg'),
    ]:
        cm_data = ws_data[ws_data['charge_mode'] == charge_mode]
        if len(cm_data) == 0:
            continue
        mean_change = cm_data.groupby('window_start')['dist_change'].mean()
        se_change = cm_data.groupby('window_start')['dist_change'].sem().fillna(0)
        ci = 1.96 * se_change.reindex(mean_change.index, fill_value=0)

        for ax in axes:
            ax.plot(mean_change.index, mean_change.values, 'o-',
                    color=color, linewidth=2.5, markersize=5,
                    label=label, alpha=0.9, markevery=2)
            ax.fill_between(mean_change.index,
                            mean_change.values - ci.values,
                            mean_change.values + ci.values,
                            color=color, alpha=0.15)

    # Plot pos-neg and neg-pos from induction data
    for polarity, color, label in [
        ('pos_neg', 'green', 'Pos-Neg'),
        ('neg_pos', 'orange', 'Neg-Pos'),
    ]:
        pol_data = ind_data[ind_data['polarity'] == polarity]
        if len(pol_data) == 0:
            continue
        mean_change = pol_data.groupby('window_start')['dist_change'].mean()
        se_change = pol_data.groupby('window_start')['dist_change'].sem().fillna(0)
        ci = 1.96 * se_change.reindex(mean_change.index, fill_value=0)

        for ax in axes:
            ax.plot(mean_change.index, mean_change.values, 'o-',
                    color=color, linewidth=2.5, markersize=5,
                    label=label, alpha=0.9, markevery=2)
            ax.fill_between(mean_change.index,
                            mean_change.values - ci.values,
                            mean_change.values + ci.values,
                            color=color, alpha=0.15)

    # Set asymmetric y-limits
    all_ymin = min(ax_bot.get_ylim()[0], ax_top.get_ylim()[0])
    all_ymax = max(ax_bot.get_ylim()[1], ax_top.get_ylim()[1])
    ax_top.set_ylim(0, all_ymax)
    ax_bot.set_ylim(all_ymin, 0)

    for ax in axes:
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=31)

    # Hide seam between axes
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    # Labels
    ax_bot.set_xlabel('Window Start Block', fontsize=38)
    fig.text(0.02, 0.5, 'Strand Distance Δ (Å)', fontsize=38,
             va='center', rotation='vertical')

    # Legend only on top
    ax_top.legend(loc='upper right', fontsize=30, ncols=2)

    ax_bot.set_xlim(-1, NUM_BLOCKS - 15 + 1)

    fig.subplots_adjust(left=0.14, top=0.75, right=0.95, bottom=0.20)

    save_path = os.path.join(output_dir, 'dist_change_all_modes_ws15.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_dist_change_by_window_size(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create one plot per window size showing mean distance change vs start block,
    with different lines for different magnitudes, separating pos/pos and neg/neg.
    """
    # Separate baseline and interventions
    baseline = results_df[results_df['intervention_config'] == 'baseline']
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())
    charge_modes = sorted(interventions['charge_mode'].unique())
    
    # Color map for magnitudes
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(magnitudes)))
    
    # Line styles for charge modes
    charge_linestyles = {
        'both_positive': '-',
        'both_negative': '--',
    }
    charge_markers = {
        'both_positive': 'o',
        'both_negative': 's',
    }
    charge_labels = {
        'both_positive': '++',
        'both_negative': '--',
    }
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        fig, ax = plt.subplots(figsize=(147, 6))
        
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            
            for charge_mode in charge_modes:
                cm_data = mag_data[mag_data['charge_mode'] == charge_mode]
                
                if len(cm_data) == 0:
                    continue
                
                # Group by window_start and compute mean distance change
                mean_change = cm_data.groupby('window_start')['dist_change'].mean()
                
                linestyle = charge_linestyles.get(charge_mode, '-')
                marker = charge_markers.get(charge_mode, 'o')
                label_suffix = charge_labels.get(charge_mode, charge_mode)
                
                ax.plot(mean_change.index, mean_change.values, 
                        linestyle=linestyle, marker=marker,
                        color=color, linewidth=2, markersize=4, 
                        label=f'mag={mag} ({label_suffix})', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add vertical lines for block region boundaries (48 blocks total)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=13)
        ax.set_title(f'Cross-Strand Distance Change vs Block Position (Window Size = {window_size})\n(solid = ++, dashed = --, positive = disruption)', fontsize=14)
        ax.legend(loc='best', fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Set x-axis limits
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'dist_change_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_hbond_change_by_window_size(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create one plot per window size showing mean H-bond change vs start block,
    with different lines for different magnitudes.
    """
    baseline = results_df[results_df['intervention_config'] == 'baseline']
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())
    
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(magnitudes)))
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            
            # Group by window_start and compute mean H-bond change
            mean_change = mag_data.groupby('window_start')['hbond_change'].mean()
            std_change = mag_data.groupby('window_start')['hbond_change'].std()
            
            ax.plot(mean_change.index, mean_change.values, 'o-', 
                    color=color, linewidth=2, markersize=4, 
                    label=f'mag={mag}', alpha=0.8)
            
            ax.fill_between(mean_change.index, 
                            mean_change.values - std_change.values,
                            mean_change.values + std_change.values,
                            color=color, alpha=0.1)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Mean H-bond Change', fontsize=13)
        ax.set_title(f'H-bond Change vs Block Position (Window Size = {window_size})\n(negative = disruption)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'hbond_change_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_disruption_rate_by_window_size(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create one plot per window size showing hairpin disruption rate vs start block,
    with different lines for different magnitudes.
    
    Disruption = baseline had hairpin but intervention doesn't.
    """
    baseline = results_df[results_df['intervention_config'] == 'baseline']
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    # Get cases where baseline had hairpin
    baseline_with_hairpin = baseline[baseline['hairpin_found'] == True]['case_idx'].values
    
    # Filter interventions to only those cases
    interventions = interventions[interventions['case_idx'].isin(baseline_with_hairpin)]
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())
    
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(magnitudes)))
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            
            # Disruption rate = fraction where hairpin_found is False (was True at baseline)
            disruption_rate = mag_data.groupby('window_start')['hairpin_found'].apply(
                lambda x: (~x).mean() * 100
            )
            
            ax.plot(disruption_rate.index, disruption_rate.values, 'o-', 
                    color=color, linewidth=2, markersize=4, 
                    label=f'mag={mag}', alpha=0.8)
        
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Disruption Rate (%)', fontsize=13)
        ax.set_title(f'Hairpin Disruption Rate vs Block Position (Window Size = {window_size})', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'disruption_rate_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_combined_summary(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create a combined summary plot with all window sizes stacked.
    Shows distance change with different magnitude lines.
    """
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    
    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(14, 5 * len(window_sizes)))
    if len(window_sizes) == 1:
        axes = [axes]
    
    magnitudes = sorted(interventions['magnitude'].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(magnitudes)))
    
    for ws_idx, window_size in enumerate(window_sizes):
        ws_data = interventions[interventions['window_size'] == window_size]
        
        ax = axes[ws_idx]
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            mean_change = mag_data.groupby('window_start')['dist_change'].mean()
            ax.plot(mean_change.index, mean_change.values, 'o-', 
                    color=color, linewidth=2, markersize=3, 
                    label=f'mag={mag}', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel('Start Block', fontsize=11)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=11)
        ax.set_title(f'Distance Change (window_size={window_size})', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-1, NUM_BLOCKS - int(window_size) + 1)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'combined_dist_change.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create heatmaps of distance change (window_start x magnitude) for each window size.
    """
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        # Pivot to create heatmap data
        pivot = ws_data.pivot_table(
            values='dist_change', 
            index='magnitude', 
            columns='window_start', 
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        vmax = np.abs(pivot.values).max()
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r', 
                       vmin=-vmax, vmax=vmax)
        
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{m}' for m in pivot.index])
        
        # Show every 5th block on x-axis
        xtick_positions = range(0, len(pivot.columns), 5)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([int(pivot.columns[i]) for i in xtick_positions])
        
        ax.set_xlabel('Start Block', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_title(f'Distance Change Heatmap (Window Size = {window_size})\n(red = increased distance = disruption)', fontsize=13)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Distance Change (Å)', fontsize=11)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'heatmap_dist_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_by_charge_mode(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create plots comparing different charge modes if multiple are present.
    """
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    charge_modes = sorted(interventions['charge_mode'].unique())
    if len(charge_modes) < 2:
        print("Only one charge mode, skipping charge mode comparison plots")
        return
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    
    charge_colors = {
        'both_positive': 'red',
        'both_negative': 'blue',
        'opposite': 'green',
    }
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for charge_mode in charge_modes:
            cm_data = ws_data[ws_data['charge_mode'] == charge_mode]
            
            # Average across magnitudes
            mean_change = cm_data.groupby('window_start')['dist_change'].mean()
            std_change = cm_data.groupby('window_start')['dist_change'].std()
            
            color = charge_colors.get(charge_mode, 'gray')
            ax.plot(mean_change.index, mean_change.values, 'o-', 
                    color=color, linewidth=2, markersize=4, 
                    label=charge_mode, alpha=0.8)
            ax.fill_between(mean_change.index, 
                            mean_change.values - std_change.values,
                            mean_change.values + std_change.values,
                            color=color, alpha=0.1)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=13)
        ax.set_title(f'Distance Change by Charge Mode (Window Size = {window_size})', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'charge_mode_comparison_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot hairpin disruption results')
    parser.add_argument('--results', type=str, default='/share/NFS/u/kevin/ProteinFolding-1/final_final_seperator_one_two_unfiltered_05/hairpin_disruption_results.parquet',
                        help='Path to results file (parquet or csv)')
    parser.add_argument('--output', type=str, default='charge_disruption_plots',
                        help='Output directory for plots')
    parser.add_argument('--induction_results', type=str,
                    default='/share/NFS/u/kevin/ProteinFolding-1/final_charge_three_ws_fifteen/hairpin_induction_results.parquet')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading results from {args.results}...")
    results_df = load_results(args.results)

# Load induction data for pos_neg / neg_pos
    induction_df = load_results(args.induction_results)

    plot_charge_mode_comparison_ws10(results_df, induction_df, args.output)
    print(f"Loaded {len(results_df)} rows")
    print(f"Columns: {results_df.columns.tolist()}")
    
    # Print summary
    baseline = results_df[results_df['intervention_config'] == 'baseline']
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    print(f"\nBaseline cases: {len(baseline)}")
    print(f"Intervention rows: {len(interventions)}")
    
    if len(interventions) > 0:
        print(f"Window sizes: {sorted([ws for ws in interventions['window_size'].unique() if ws > 0])}")
        print(f"Magnitudes: {sorted(interventions['magnitude'].unique())}")
        print(f"Charge modes: {sorted(interventions['charge_mode'].unique())}")
        print(f"Window starts: {sorted(interventions['window_start'].unique())[:5]}...{sorted(interventions['window_start'].unique())[-5:]}")
    
    print("\nGenerating plots...")
    
    # # Generate all plots
    # plot_charge_mode_comparison_ws10(results_df, args.output)
    # plot_dist_change_by_window_size(results_df, args.output)
    # plot_hbond_change_by_window_size(results_df, args.output)
    # plot_disruption_rate_by_window_size(results_df, args.output)
    # plot_combined_summary(results_df, args.output)
    # plot_heatmap(results_df, args.output)
    # plot_by_charge_mode(results_df, args.output)
    
    print(f"\nAll plots saved to {args.output}/")


if __name__ == '__main__':
    main()