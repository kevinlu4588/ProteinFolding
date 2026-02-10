#!/usr/bin/env python
"""
Plot H-bond formation (%) by window start block, with Rg quality filter applied.

Loads the parquet + pdb_index, zeros out hbond% for interventions that fail
the Rg ratio threshold or have no Rg data, then plots the mean curve.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Rg computation (same as analyze_charge4.py)
# ============================================================================

def compute_rg_from_pdb(pdb_path):
    """Compute radius of gyration from Cα atoms in a PDB file."""
    ca_coords = []
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
    except Exception:
        return np.nan
    if len(ca_coords) < 3:
        return np.nan
    coords = np.array(ca_coords)
    center = coords.mean(axis=0)
    return np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))


def add_rg_to_pdb_index(pdb_index, pdb_root):
    """Compute Rg for every PDB in the index."""
    print(f"  Computing Rg for {len(pdb_index)} PDBs ...")
    rg_values = []
    n_missing = 0
    for i, row in pdb_index.iterrows():
        pdb_path = os.path.join(pdb_root, row["pdb_file"])
        rg = compute_rg_from_pdb(pdb_path)
        if np.isnan(rg):
            n_missing += 1
        rg_values.append(rg)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(pdb_index)} done")
    pdb_index = pdb_index.copy()
    pdb_index["rg"] = rg_values
    print(f"  Done. {n_missing} PDBs could not be read.")
    return pdb_index


# ============================================================================
# Data loading + Rg filter
# ============================================================================

def load_and_filter(args):
    """Load parquet + pdb_index, merge Rg, zero out bad rows. Returns (df_raw, df_filtered)."""
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  {df['case_idx'].nunique()} cases, {len(df)} rows")

    print(f"Loading pdb_index: {args.pdb_index}")
    pdb_idx = pd.read_csv(args.pdb_index)
    print(f"  {len(pdb_idx)} rows, {pdb_idx['case_idx'].nunique()} cases")

    if "rg" in pdb_idx.columns and args.skip_rg_compute:
        print("  Using cached Rg values")
    else:
        pdb_idx = add_rg_to_pdb_index(pdb_idx, args.pdb_root)

    bl_pdb = pdb_idx[pdb_idx["block_set"] == "baseline"]
    iv_pdb = pdb_idx[pdb_idx["block_set"] != "baseline"]
    bl_rg = bl_pdb.groupby("case_idx")["rg"].mean().rename("baseline_rg")

    # Aggregate quality per (case_idx, window_start)
    quality = iv_pdb.groupby(["case_idx", "window_start"]).agg(
        rg=("rg", "mean"),
    ).reset_index()
    quality = quality.merge(bl_rg, on="case_idx", how="left")
    quality["rg_ratio"] = quality["rg"] / quality["baseline_rg"]

    df_merged = df.merge(
        quality[["case_idx", "window_start", "rg", "rg_ratio", "baseline_rg"]],
        on=["case_idx", "window_start"], how="left"
    )

    # Zero out hbond% for interventions that fail Rg or have no Rg data
    df_filtered = df_merged.copy()
    is_iv = df_filtered["block_set"] != "baseline"
    has_rg = df_filtered["rg_ratio"].notna()
    fails_rg = df_filtered["rg_ratio"] < args.rg_ratio_threshold
    zero_mask = is_iv & (~has_rg | fails_rg)
    df_filtered.loc[zero_mask, "hbond_percentage"] = 0.0

    n_no_rg = (is_iv & ~has_rg).sum()
    n_fail = (is_iv & has_rg & fails_rg).sum()
    print(f"  Rg filter: zeroed {zero_mask.sum()} rows "
          f"({n_fail} failed threshold, {n_no_rg} missing Rg)")

    return df_merged, df_filtered


# ============================================================================
# Plotting
# ============================================================================

def plot_hbond_curve(df, label, color, ax, polarity=None):
    """Plot mean hbond% ± SEM by window_start for interventions."""
    iv = df[df["block_set"] != "baseline"]
    if polarity is not None:
        iv = iv[iv["polarity"] == polarity]
    grp = iv.groupby("window_start")["hbond_percentage"]
    means = grp.mean().sort_index()
    sems = grp.sem().fillna(0).reindex(means.index)

    ax.fill_between(means.index, means.values, color=color, alpha=0.3)
    ax.plot(means.index, means.values, color=color, linewidth=2.5,
            marker="o", markersize=5, label=label)
    # SEM as thin error bars
    ax.errorbar(means.index, means.values, yerr=sems.values,
                fmt="none", ecolor=color, alpha=0.5, capsize=2, lw=1)


def main():
    parser = argparse.ArgumentParser(
        description="Plot H-bond formation with Rg quality filter")
    parser.add_argument("--parquet",
                        default="/share/NFS/u/kevin/ProteinFolding-1/final_charge_three_ws_fifteen/hairpin_induction_results.parquet")
    parser.add_argument("--pdb_index",
                        default="/share/NFS/u/kevin/ProteinFolding-1/animation_searching_three/pdb_index.csv")
    parser.add_argument("--pdb_root",
                        default="/share/NFS/u/kevin/ProteinFolding-1/animation_searching_three")
    parser.add_argument("--output", default="hbond_rg_filtered")
    parser.add_argument("--rg_ratio_threshold", type=float, default=0.70)
    parser.add_argument("--skip_rg_compute", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df_raw, df_filtered = load_and_filter(args)

    # ==========================================================================
    # Configuration
    # ==========================================================================
    FONT_SIZE = 30
    FIGSIZE = (10, 6)
    COLOR_RAW = "#aaaaaa"
    COLOR_FILTERED = "#d95f02"

    # Intervention window shading (set to None to disable)
    INTERVENTION_WINDOW = None  # e.g., (0, 3)
    SHOW_INTERVENTION_IN_LEGEND = False

    # ==========================================================================
    # Plot: raw vs Rg-filtered overlay
    # ==========================================================================
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, top=0.70, right=0.95, bottom=0.35)

    if INTERVENTION_WINDOW is not None:
        ax.axvspan(INTERVENTION_WINDOW[0], INTERVENTION_WINDOW[1],
                   alpha=0.2, color="gray", zorder=0,
                   label="Intervention Window" if SHOW_INTERVENTION_IN_LEGEND else None)

    plot_hbond_curve(df_raw, "Unfiltered", COLOR_RAW, ax)
    plot_hbond_curve(df_filtered, "Rg-filtered", COLOR_FILTERED, ax)

    # Baseline reference line
    bl = df_raw[df_raw["block_set"] == "baseline"]
    bl_mean = bl["hbond_percentage"].mean() if len(bl) > 0 else 0
    ax.axhline(bl_mean, color="gray", ls="--", lw=1.5, alpha=0.5)

    ax.set_xlabel("Window Start Block", fontsize=FONT_SIZE)
    ax.set_ylabel("H-bond Formation (%)", fontsize=FONT_SIZE)
    ax.set_xlim(0, 34)
    ax.tick_params(axis="both", labelsize=FONT_SIZE, width=1.5, length=6)
    ax.legend(loc="upper right", fontsize=FONT_SIZE * 0.6, frameon=True)

    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, axis="both")

    for ext in ["png", "pdf"]:
        path = os.path.join(args.output, f"hbond_rg_filtered.{ext}")
        plt.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.close()

    # ==========================================================================
    # Plot: filtered only (clean version for presentations)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, top=0.70, right=0.95, bottom=0.35)

    if INTERVENTION_WINDOW is not None:
        ax.axvspan(INTERVENTION_WINDOW[0], INTERVENTION_WINDOW[1],
                   alpha=0.2, color="gray", zorder=0,
                   label="Intervention Window" if SHOW_INTERVENTION_IN_LEGEND else None)

    plot_hbond_curve(df_filtered, "H-bond\n(N-O < 3.5Å)", COLOR_FILTERED, ax)

    ax.axhline(bl_mean, color="gray", ls="--", lw=1.5, alpha=0.5)

    ax.set_xlabel("Window Start Block", fontsize=FONT_SIZE)
    ax.set_ylabel("H-bond Formation (%)", fontsize=FONT_SIZE)
    ax.set_xlim(0, 34)
    ax.tick_params(axis="both", labelsize=FONT_SIZE, width=1.5, length=6)
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.legend(loc="upper right", fontsize=FONT_SIZE, frameon=True)

    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, axis="both")

    for ext in ["png", "pdf"]:
        path = os.path.join(args.output, f"hbond_clean.{ext}")
        plt.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()