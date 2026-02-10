#!/usr/bin/env python
"""
Plot H-bond formation (%) vs Window Start Block
================================================

Reads gradient_steering_results.parquet and produces a single clean figure
styled to match the publication-ready format (large fonts, thick spines,
fill-under-curve, no top/right spines).

For each window_start block, computes the percentage of interventions that
gained ≥1 potential H-bond (i.e. hbond_change > 0) relative to baseline.

Usage:
    python plot_hbond_formation.py --results path/to/gradient_steering_results.parquet
    python plot_hbond_formation.py --results_dir final_contact_steering_std/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration — easy to adjust
# =============================================================================
FONT_SIZE = 42
FIGSIZE = (14, 10)
COLOR = "#16a34a"

# Intervention window shading (set to None to disable)
INTERVENTION_WINDOW = None  # e.g., (0, 3) to shade blocks 0-3
SHOW_INTERVENTION_IN_LEGEND = False


# =============================================================================
# Helpers
# =============================================================================

def detect_scaling(magnitudes: np.ndarray) -> str:
    if len(magnitudes) == 0:
        return "raw"
    return "raw" if magnitudes.max() > 20 else "std"


def load_data(args):
    """Resolve paths and load the parquet file."""
    if args.results_dir:
        results_path = args.results or os.path.join(
            args.results_dir, "gradient_steering_results.parquet"
        )
        out = args.output or os.path.join(args.results_dir, "plots")
    elif args.results:
        results_path = args.results
        out = args.output or os.path.join(os.path.dirname(results_path), "plots")
    else:
        print("Provide --results_dir or --results. Run with -h for help.")
        sys.exit(1)

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    os.makedirs(out, exist_ok=True)

    df = pd.read_parquet(results_path)
    print(f"Loaded {len(df)} rows, {df['case_idx'].nunique()} cases")

    iv_mags = df.loc[df["block_set"] != "baseline", "magnitude"].unique()
    scaling = args.scaling or detect_scaling(iv_mags)
    print(f"Magnitude scaling: {scaling}")

    return df, out, scaling


# =============================================================================
# Main plot
# =============================================================================

def plot_hbond_formation(df: pd.DataFrame, out: str, scaling: str):
    """
    For each window_start block, compute the fraction of interventions where
    hbond_change > 0 (i.e. at least one H-bond gained), expressed as %.

    Produces a single-panel figure with the publication-ready style.
    """
    iv = df[(df["block_set"] != "baseline") & df["hbond_change"].notna()].copy()
    if iv.empty:
        print("No intervention data with hbond_change — nothing to plot.")
        return

    # Aggregate: % of interventions with positive H-bond change per block
    blocks = sorted(iv["window_start"].unique())
    hbond_pct = []
    for b in blocks:
        sub = iv[iv["window_start"] == b]
        pct = (sub["hbond_change"] > 0).mean() * 100
        hbond_pct.append(pct)

    blocks = np.array(blocks)
    hbond_pct = np.array(hbond_pct)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, top=0.70, right=0.95, bottom=0.35)

    # Intervention window shading
    if INTERVENTION_WINDOW is not None:
        ax.axvspan(
            INTERVENTION_WINDOW[0],
            INTERVENTION_WINDOW[1],
            alpha=0.2,
            color="gray",
            zorder=0,
            label="Intervention Window" if SHOW_INTERVENTION_IN_LEGEND else None,
        )

    # Fill under the curve
    ax.fill_between(blocks, hbond_pct, color=COLOR, alpha=0.3)

    # Line + markers
    ax.plot(
        blocks,
        hbond_pct,
        color=COLOR,
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="H-bond\n(N–O < 3.5 Å)",
    )

    ax.set_xlabel("Window Start Block", fontsize=FONT_SIZE)
    ax.set_ylabel("H-bond Formation (%)", fontsize=FONT_SIZE)

    ax.set_xlim(blocks.min(), blocks.max())
    ax.set_ylim(0, None)  # auto upper limit from real data

    ax.tick_params(axis="both", labelsize=FONT_SIZE, width=1.5, length=6)

    ax.legend(loc="upper left", fontsize=FONT_SIZE-5, frameon=True)

    # Spine styling
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    ax.grid(alpha=0.3, axis="both")

    # Save
    png_path = os.path.join(out, "hbond_formation_plot.png")
    pdf_path = os.path.join(out, "hbond_formation_plot.pdf")
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot H-bond formation (%) vs window start block"
    )
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument(
        "--results",
        type=str,
        default="/share/NFS/u/kevin/ProteinFolding-1/final_final_final_contact_steering_std/gradient_steering_results.parquet",
        help="Path to gradient_steering_results.parquet",
    )
    p.add_argument("--output", type=str, default=None, help="Output directory")
    p.add_argument(
        "--scaling",
        type=str,
        default=None,
        choices=["std", "raw"],
        help="Magnitude scaling convention (auto-detected if omitted)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df, out, scaling = load_data(args)
    plot_hbond_formation(df, out, scaling)
    print("Done.")


if __name__ == "__main__":
    main()