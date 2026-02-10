#!/usr/bin/env python
"""
Bias Map Extraction for 1l0s

Generates pair-to-sequence bias maps with contact map overlays for a single
protein (1l0s) at selected blocks, plus per-head breakdowns at block 30.

Outputs saved to bias_maps/ as individual PNGs.

Usage:
    python bias_map_1l0s.py --parquet data_old/all_block_patching_results.parquet
"""

import argparse
import os
import types

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import EsmFoldingTrunk
from transformers.utils import ContextManagers
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================
BLOCKS = [0, 4, 8, 12, 16, 20, 24, 28, 30, 32, 36, 40, 44, 47]
HEAD_DETAIL_BLOCK = 32
NUM_HEADS = 8
CONTACT_THRESHOLD = 8.0  # Angstroms
CONTACT_LINEWIDTH = 4  # doubled from default ~2
OUTPUT_DIR = "bias_maps"

# Plot styling
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
CBAR_LABEL_FONTSIZE = 14
FIGSIZE = (8, 7)


# =============================================================================
# Model hooking — collect biases (mean over heads + per-head at detail block)
# =============================================================================
def make_collecting_trunk_forward(blocks_to_collect, detail_block, collected):
    """Return a replacement trunk.forward that stores bias tensors."""

    def trunk_forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        device = seq_feats.device
        s_s_0, s_z_0 = seq_feats, pair_feats

        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            no_recycles += 1

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            for block_idx, block in enumerate(self.blocks):
                if block_idx in blocks_to_collect:
                    bias = block.pair_to_sequence(z)  # (1, L, L, num_heads)
                    collected["biases"][block_idx] = bias.detach().cpu()
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        s_s, s_z = s_s_0, s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        for recycle_idx in range(no_recycles):
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
                recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
                recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)
                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)
                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa, mask.float(),
                )
                recycle_s, recycle_z = s_s, s_z
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3], 3.375, 21.375, self.recycle_bins,
                )

        structure["s_s"], structure["s_z"] = s_s, s_z
        return structure

    return trunk_forward


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_bias_with_contacts(bias_2d, contacts, title, save_path, font_scale=1.0):
    """Plot a single bias heatmap with contact contour overlay."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    vmax = np.percentile(np.abs(bias_2d), 98)
    if vmax < 1e-6:
        vmax = 1.0  # avoid degenerate color scale for near-zero biases

    im = ax.imshow(bias_2d, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.contour(contacts, levels=[0.5], colors="lime", linewidths=CONTACT_LINEWIDTH)

    ax.set_xlabel("Residue", fontsize=LABEL_FONTSIZE * font_scale)
    ax.set_ylabel("Residue", fontsize=LABEL_FONTSIZE * font_scale)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE * font_scale, pad=12)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE * font_scale)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Bias", fontsize=CBAR_LABEL_FONTSIZE * font_scale)
    cbar.ax.tick_params(labelsize=(TICK_FONTSIZE - 2) * font_scale)

    legend_elements = [
        Line2D([0], [0], color="lime", linewidth=CONTACT_LINEWIDTH,
               label=f"Contacts\n(<{CONTACT_THRESHOLD}Å)")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=(LABEL_FONTSIZE - 2) * font_scale)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract bias maps for 1l0s",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet", type=str, default="data_old/all_block_patching_results.parquet",
                        help="Input parquet with sequences")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (auto if None)")
    parser.add_argument("--contact-threshold", type=float, default=CONTACT_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load sequence for 1l0s
    # ------------------------------------------------------------------
    print(f"Loading data from {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    donors = df[["donor_pdb", "donor_sequence", "donor_hairpin_start", "donor_hairpin_end"]].drop_duplicates()
    row = donors[donors["donor_pdb"].str.contains("1l0s", case=False)]
    if len(row) == 0:
        raise ValueError("1l0s not found in parquet. Available PDBs: "
                         + ", ".join(donors["donor_pdb"].unique()[:20]))
    row = row.iloc[0]
    sequence = row["donor_sequence"]
    print(f"1l0s: length {len(sequence)}, "
          f"hairpin [{int(row['donor_hairpin_start'])}:{int(row['donor_hairpin_end'])}]")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"Loading ESMFold on {device} ...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # ------------------------------------------------------------------
    # 3. Single forward pass — collect biases at all requested blocks
    # ------------------------------------------------------------------
    blocks_needed = set(BLOCKS)  # HEAD_DETAIL_BLOCK (30) is already in BLOCKS
    collected = {"biases": {}}
    original_forward = model.trunk.forward

    new_forward = make_collecting_trunk_forward(blocks_needed, HEAD_DETAIL_BLOCK, collected)
    model.trunk.forward = types.MethodType(new_forward, model.trunk)

    print("Running forward pass ...")
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model(**inputs, num_recycles=0)

    model.trunk.forward = original_forward  # restore

    # ------------------------------------------------------------------
    # 4. Compute contact map from predicted structure
    # ------------------------------------------------------------------
    ca_positions = outputs.positions[-1, 0, :, 1, :]  # (L, 3) — CA atoms
    diff = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1)).cpu().numpy()
    contacts = (distances < args.contact_threshold).astype(float)
    print(f"Contact map: {int(contacts.sum())} pairs < {args.contact_threshold}Å")

    # ------------------------------------------------------------------
    # 5. Plot head-averaged bias for each block
    # ------------------------------------------------------------------
    print(f"\nSaving head-averaged bias maps for {len(BLOCKS)} blocks ...")
    for block_idx in tqdm(BLOCKS, desc="Block maps"):
        bias = collected["biases"][block_idx]          # (1, L, L, H)
        bias_avg = bias[0].mean(dim=-1).cpu().numpy()  # (L, L)

        save_path = os.path.join(args.output, f"1l0s_block{block_idx:02d}_avg.png")

        if block_idx == HEAD_DETAIL_BLOCK:
            # Clean version: no title, 2.5x fonts
            clean_path = os.path.join(args.output, f"1l0s_block{block_idx:02d}_avg_clean.png")
            plot_bias_with_contacts(
                bias_avg, contacts,
                title=None,
                save_path=clean_path,
                font_scale=2.0,
            )
            # Normal version: with title, 2x fonts
            plot_bias_with_contacts(
                bias_avg, contacts,
                title=f"Block {block_idx}",
                save_path=save_path,
                font_scale=2.0,
            )
        else:
            plot_bias_with_contacts(
                bias_avg, contacts,
                title=f"Block {block_idx}",
                save_path=save_path,
                font_scale=2.0,
            )

    # ------------------------------------------------------------------
    # 6. Plot per-head bias at the detail block (32)
    # ------------------------------------------------------------------
    print(f"\nSaving per-head bias maps for block {HEAD_DETAIL_BLOCK} ...")
    detail_bias = collected["biases"][HEAD_DETAIL_BLOCK]  # (1, L, L, H)
    for head_idx in tqdm(range(NUM_HEADS), desc="Head maps"):
        bias_head = detail_bias[0, :, :, head_idx].cpu().numpy()  # (L, L)

        save_path = os.path.join(args.output, f"1l0s_block{HEAD_DETAIL_BLOCK:02d}_head{head_idx}.png")
        plot_bias_with_contacts(
            bias_head, contacts,
            title=f"Block {HEAD_DETAIL_BLOCK}, Head {head_idx}",
            save_path=save_path,
            font_scale=2.0,
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    n_files = len(BLOCKS) + NUM_HEADS + 1  # +1 for clean version of detail block
    print(f"\n✓ Saved {n_files} plots to {args.output}/")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()