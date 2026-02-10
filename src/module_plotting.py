"""
Multi-Module Patching Results Visualization
============================================

Generates comparison plots for the module patching experiment, showing hairpin
formation rates when patching ESM encoder, folding trunk, or structure module.

Usage:
    python module_plotting.py --results results/module_patching.parquet --output plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(results_path: str) -> pd.DataFrame:
    """Load results from parquet file."""
    df = pd.read_parquet(results_path)
    print(f"Loaded {len(df)} results")
    print(f"Columns: {list(df.columns)}")
    return df


def create_method_label(row: pd.Series) -> str:
    """Create a readable label for each patching method."""
    if row.get("patch_mode") == "input_intervention":
        return "Input\nIntervention"
    
    module = row.get("patch_module", "unknown")
    
    if module == "encoder":
        return "ESM\nEncoder"
    elif module == "trunk":
        mode = row.get("patch_mode", "")
        mask = row.get("patch_mask_mode", "")
        if mode == "sequence":
            return "Trunk\nSequence"
        elif mode == "pairwise":
            return f"Trunk\nPairwise\n({mask})"
        elif mode == "both":
            return f"Trunk\nBoth\n({mask})"
        else:
            return f"Trunk\n{mode}"
    elif module == "structure_module":
        return "SM\nPost-IPA"
    else:
        return f"{module}"


def plot_success_rates(df: pd.DataFrame, output_dir: Path):
    """Plot hairpin detection success rates for all methods."""
    df = df.copy()
    df["method"] = df.apply(create_method_label, axis=1)
    
    success_rates = df.groupby("method").agg({
        "hairpin_found": ["mean", "sum", "count"]
    }).reset_index()
    success_rates.columns = ["method", "success_rate", "n_success", "n_total"]
    success_rates = success_rates.sort_values("success_rate", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(success_rates)))
    bars = ax.barh(success_rates["method"], success_rates["success_rate"], color=colors)
    
    for bar, (_, row) in zip(bars, success_rates.iterrows()):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.1%} ({int(row["n_success"])}/{int(row["n_total"])})',
                va='center', fontsize=10)
    
    ax.set_xlabel("Hairpin Detection Rate", fontsize=12)
    ax.set_title("Hairpin Transfer Success Rate by Patching Method", fontsize=14)
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "success_rates.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved success rates plot to {output_dir / 'success_rates.png'}")
    return success_rates


def plot_success_rates_grouped(df: pd.DataFrame, output_dir: Path):
    """Plot success rates grouped by module type."""
    df = df.copy()
    df["method"] = df.apply(create_method_label, axis=1)
    
    def get_module_group(row):
        if row.get("patch_mode") == "input_intervention":
            return "Baseline"
        return row.get("patch_module", "unknown").replace("_", " ").title()
    
    df["module_group"] = df.apply(get_module_group, axis=1)
    
    success_by_method = df.groupby(["module_group", "method"])["hairpin_found"].agg(["mean", "count"]).reset_index()
    success_by_method.columns = ["module_group", "method", "success_rate", "n"]
    
    group_order = ["Baseline", "Encoder", "Trunk", "Structure Module"]
    success_by_method["module_group"] = pd.Categorical(
        success_by_method["module_group"], 
        categories=group_order, 
        ordered=True
    )
    success_by_method = success_by_method.sort_values(["module_group", "success_rate"])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    group_colors = {
        "Baseline": "#2ecc71",
        "Encoder": "#3498db", 
        "Trunk": "#e74c3c",
        "Structure Module": "#9b59b6"
    }
    
    colors = [group_colors.get(g, "#95a5a6") for g in success_by_method["module_group"]]
    
    bars = ax.barh(range(len(success_by_method)), success_by_method["success_rate"], color=colors)
    ax.set_yticks(range(len(success_by_method)))
    ax.set_yticklabels(success_by_method["method"])
    
    for i, (bar, (_, row)) in enumerate(zip(bars, success_by_method.iterrows())):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.1%} (n={int(row["n"])})',
                va='center', fontsize=9)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colors.items() if g in df["module_group"].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_xlabel("Hairpin Detection Rate", fontsize=12)
    ax.set_title("Hairpin Transfer Success Rate by Patching Method", fontsize=14)
    ax.set_xlim(0, 1.2)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "success_rates_grouped.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grouped success rates plot to {output_dir / 'success_rates_grouped.png'}")


def plot_helix_degradation(df: pd.DataFrame, output_dir: Path):
    """Plot alpha helix content changes for successful vs unsuccessful patches."""
    if "helix_change" not in df.columns or df["helix_change"].isna().all():
        print("No helix change data available, skipping helix degradation plot")
        return None
    
    df = df.copy()
    df["method"] = df.apply(create_method_label, axis=1)
    df["outcome"] = df["hairpin_found"].map({True: "Success", False: "Failure"})
    
    df_helix = df[df["helix_change"].notna()].copy()
    
    if len(df_helix) == 0:
        print("No valid helix change data, skipping plot")
        return None
    
    helix_stats = df_helix.groupby(["method", "outcome"])["helix_change"].agg(["mean", "std", "count"]).reset_index()
    helix_stats.columns = ["method", "outcome", "mean_change", "std_change", "n"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    methods_order = df_helix.groupby("method")["hairpin_found"].mean().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df_helix, 
        x="method", 
        y="helix_change", 
        hue="outcome",
        order=methods_order,
        ax=ax1,
        palette={"Success": "#2ecc71", "Failure": "#e74c3c"}
    )
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("")
    ax1.set_ylabel("Helix Content Change (%)", fontsize=11)
    ax1.set_title("Alpha Helix Change by Method and Outcome", fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title="Hairpin Found")
    
    ax2 = axes[1]
    
    overall_by_outcome = df_helix.groupby("outcome")["helix_change"].agg(["mean", "std", "count"])
    
    x = [0, 1]
    means = [overall_by_outcome.loc["Failure", "mean"], overall_by_outcome.loc["Success", "mean"]]
    stds = [overall_by_outcome.loc["Failure", "std"], overall_by_outcome.loc["Success", "std"]]
    ns = [overall_by_outcome.loc["Failure", "count"], overall_by_outcome.loc["Success", "count"]]
    colors = ["#e74c3c", "#2ecc71"]
    
    bars = ax2.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Failed\nPatches", "Successful\nPatches"])
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Mean Helix Content Change (%)", fontsize=11)
    ax2.set_title("Overall Helix Degradation:\nSuccessful vs Failed Patches", fontsize=12)
    
    for i, (bar, n) in enumerate(zip(bars, ns)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + stds[i] + 0.5,
                f'n={int(n)}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "helix_degradation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved helix degradation plot to {output_dir / 'helix_degradation.png'}")
    return helix_stats


def plot_helix_by_method(df: pd.DataFrame, output_dir: Path):
    """Plot helix degradation separately for each method."""
    if "helix_change" not in df.columns or df["helix_change"].isna().all():
        print("No helix change data available, skipping method-specific helix plot")
        return
    
    df = df.copy()
    df["method"] = df.apply(create_method_label, axis=1)
    df_helix = df[df["helix_change"].notna()].copy()
    
    if len(df_helix) == 0:
        return
    
    stats = df_helix.groupby(["method", "hairpin_found"]).agg({
        "helix_change": ["mean", "std", "count"]
    }).reset_index()
    stats.columns = ["method", "hairpin_found", "mean", "std", "n"]
    
    success_stats = stats[stats["hairpin_found"] == True].set_index("method")
    failure_stats = stats[stats["hairpin_found"] == False].set_index("method")
    
    methods = stats["method"].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    success_means = [success_stats.loc[m, "mean"] if m in success_stats.index else 0 for m in methods]
    success_stds = [success_stats.loc[m, "std"] if m in success_stats.index else 0 for m in methods]
    failure_means = [failure_stats.loc[m, "mean"] if m in failure_stats.index else 0 for m in methods]
    failure_stds = [failure_stats.loc[m, "std"] if m in failure_stats.index else 0 for m in methods]
    
    bars1 = ax.bar(x - width/2, failure_means, width, yerr=failure_stds, 
                   label='Failed', color='#e74c3c', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, success_means, width, yerr=success_stds,
                   label='Success', color='#2ecc71', alpha=0.8, capsize=3)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Mean Helix Content Change (%)', fontsize=11)
    ax.set_title('Alpha Helix Degradation by Method and Outcome', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "helix_by_method.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved helix by method plot to {output_dir / 'helix_by_method.png'}")


def plot_plddt_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot pLDDT scores for successful vs unsuccessful patches."""
    df = df.copy()
    df["method"] = df.apply(create_method_label, axis=1)
    df["outcome"] = df["hairpin_found"].map({True: "Success", False: "Failure"})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    methods_order = df.groupby("method")["hairpin_found"].mean().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df,
        x="method",
        y="mean_plddt",
        hue="outcome",
        order=methods_order,
        ax=ax1,
        palette={"Success": "#2ecc71", "Failure": "#e74c3c"}
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Mean pLDDT", fontsize=11)
    ax1.set_title("Overall Structure Confidence", fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title="Hairpin Found")
    
    ax2 = axes[1]
    sns.boxplot(
        data=df,
        x="method",
        y="patch_region_plddt",
        hue="outcome",
        order=methods_order,
        ax=ax2,
        palette={"Success": "#2ecc71", "Failure": "#e74c3c"}
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("Patch Region pLDDT", fontsize=11)
    ax2.set_title("Patched Region Confidence", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title="Hairpin Found")
    
    plt.tight_layout()
    plt.savefig(output_dir / "plddt_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved pLDDT comparison plot to {output_dir / 'plddt_comparison.png'}")


def plot_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create and save a summary statistics table."""
    df = df.copy()
    df["method"] = df.apply(create_method_label, axis=1)
    
    summary = df.groupby("method").agg({
        "hairpin_found": ["mean", "sum", "count"],
        "mean_plddt": ["mean", "std"],
        "patch_region_plddt": ["mean", "std"],
    })
    
    summary.columns = [
        "success_rate", "n_success", "n_total",
        "mean_plddt_avg", "mean_plddt_std",
        "patch_plddt_avg", "patch_plddt_std"
    ]
    
    if "helix_change" in df.columns and not df["helix_change"].isna().all():
        helix_stats = df.groupby("method")["helix_change"].agg(["mean", "std"])
        helix_stats.columns = ["helix_change_avg", "helix_change_std"]
        summary = summary.join(helix_stats)
    
    summary = summary.reset_index()
    summary = summary.sort_values("success_rate", ascending=False)
    
    summary.to_csv(output_dir / "summary_stats.csv", index=False)
    print(f"Saved summary statistics to {output_dir / 'summary_stats.csv'}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary.to_string(index=False))
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot ESMFold patching experiment results")
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to results parquet file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots (default: same as results)"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_results(results_path)
    
    print("\nGenerating plots...")
    
    plot_success_rates(df, output_dir)
    plot_success_rates_grouped(df, output_dir)
    plot_helix_degradation(df, output_dir)
    plot_helix_by_method(df, output_dir)
    plot_plddt_comparison(df, output_dir)
    plot_summary_table(df, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()