#!/usr/bin/env python
"""
Render a dual-panel animation of steering magnitude sweep:
  Left:  3D protein structure (PyMOL ray-traced)
  Right: Info panel with magnitude progress bar + pairwise distance matrix

Usage:
    python render_steering_sweep.py <pdb_folder> --output sweep.mp4 \
        --strand1 "10,11,12,13" --strand2 "20,21,22,23" --mode closer

The PDB folder should contain files named like:
    frame_0000_mag_0.0000.pdb
    frame_0001_mag_0.1899.pdb
    ...

Requires: pymol, ffmpeg, matplotlib, numpy, pillow
"""

import argparse
import os
import re
import glob
import subprocess
import tempfile
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image


# ── Parse magnitude from filename ───────────────────────────────────

def parse_magnitude(filename):
    """Extract magnitude from filename like frame_0003_mag_1.2345.pdb"""
    m = re.search(r"mag_([\d]+\.[\d]+)\.pdb", os.path.basename(filename))
    if m:
        return float(m.group(1))
    return None


# ── Extract CB (or CA) coordinates by residue index ─────────────────

def get_cb_coords_by_residue(pdb_path):
    """Extract per-residue CB coordinates (fallback to CA for GLY) from PDB.
    Returns dict: residue_index (0-based) -> np.array([x,y,z])
    """
    ca_coords = {}
    cb_coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            # Use 0-based residue index
            res_seq = int(line[22:26].strip()) - 1
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if atom_name == "CA":
                ca_coords[res_seq] = np.array([x, y, z])
            elif atom_name == "CB":
                cb_coords[res_seq] = np.array([x, y, z])

    # Merge: prefer CB, fall back to CA
    merged = {}
    all_res = set(ca_coords.keys()) | set(cb_coords.keys())
    for r in all_res:
        merged[r] = cb_coords.get(r, ca_coords.get(r))
    return merged


def compute_cross_strand_distances(pdb_path, strand1_residues, strand2_residues):
    """Compute pairwise CB distances between two sets of residues.
    Returns matrix of shape (len(strand1), len(strand2)).
    """
    coords = get_cb_coords_by_residue(pdb_path)
    n1 = len(strand1_residues)
    n2 = len(strand2_residues)
    dist_matrix = np.full((n1, n2), np.nan)

    for i, r1 in enumerate(strand1_residues):
        for j, r2 in enumerate(strand2_residues):
            if r1 in coords and r2 in coords:
                dist_matrix[i, j] = np.linalg.norm(coords[r1] - coords[r2])

    return dist_matrix


# ── Color scheme ────────────────────────────────────────────────────

def mag_color_at_t(t):
    """Light green (low mag) -> Neon green (high mag)."""
    r = 0.75 - t * 0.65
    g = 0.90 - t * 0.10
    b = 0.65 - t * 0.60
    return (r, g, b)


# Distance colormap for "closer" mode:
# far (light/pale green) -> close (dark green)
CMAP_CLOSER = LinearSegmentedColormap.from_list(
    'closer',
    [
        '#1a5c2e',  # Dark green (close, low distance)
        '#2d8a4e',
        '#5cb578',
        '#8ed4a5',
        '#c5ebd2',  # Light green (far, high distance)
    ],
    N=256
)

# Distance colormap for "farther" mode:
# close (dark green) -> far (light/pale green)
CMAP_FARTHER = LinearSegmentedColormap.from_list(
    'farther',
    [
        '#c5ebd2',  # Light green (close)
        '#8ed4a5',
        '#5cb578',
        '#2d8a4e',
        '#1a5c2e',  # Dark green (far)
    ],
    N=256
)

CONTACT_THRESHOLD = 8.0  # Angstroms for neon highlight


# ── PyMOL rendering ────────────────────────────────────────────────

def render_pymol_frames(pdb_folder, frame_dir, magnitudes,
                        strand1_residues, strand2_residues,
                        mean_dists, pw, ph):
    """Ray-trace each PDB frame in PyMOL independently.
    Cross-strand residues colored by mean distance (matching matrix colormap),
    rest gray."""

    mag_list = json.dumps(magnitudes.tolist())
    mag_max = float(magnitudes.max())
    mag_min = float(magnitudes.min())
    dist_list = json.dumps([float(d) for d in mean_dists])
    dist_min = float(min(mean_dists))
    dist_max = float(max(mean_dists))

    # Find baseline frame (mag closest to 0)
    baseline_idx = int(np.argmin(np.abs(magnitudes)))

    # PyMOL uses 1-based residue numbering
    s1_sele = " or ".join(f"resi {r+1}" for r in strand1_residues)
    s2_sele = " or ".join(f"resi {r+1}" for r in strand2_residues)

    pymol_script = f'''
import os, glob, json
from pymol import cmd

pdb_files = sorted(glob.glob(os.path.join("{pdb_folder}", "frame_*.pdb")))
magnitudes = json.loads('{mag_list}')
mean_dists = json.loads('{dist_list}')
mag_min = {mag_min}
mag_max = {mag_max}
dist_min = {dist_min}
dist_max = {dist_max}
baseline_idx = {baseline_idx}
n = len(pdb_files)

def dist_rgb(t):
    """t=0 -> close (dark green), t=1 -> far (light green).
    Matches the CMAP_CLOSER colormap."""
    r = 0.10 + t * 0.67   # 0.10 -> 0.77
    g = 0.36 + t * 0.56   # 0.36 -> 0.92
    b = 0.18 + t * 0.64   # 0.18 -> 0.82
    return [r, g, b]

# Global rendering settings
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("cartoon_sampling", 14)
cmd.set("cartoon_loop_radius", 0.2)
cmd.set("antialias", 2)
cmd.set("ray_shadows", 1)
cmd.set("ray_trace_mode", 0)
cmd.set("ray_opaque_background", 1)
cmd.bg_color("white")

# First pass: load baseline frame to set camera
cmd.load(pdb_files[baseline_idx], "ref")
cmd.hide("everything", "ref")
cmd.show("cartoon", "ref")
cmd.dss("ref")
cmd.orient("ref")
saved_view = cmd.get_view()
cmd.delete("ref")

# Second pass: render each frame independently
for i, pdb_path in enumerate(pdb_files):
    d = mean_dists[i] if i < len(mean_dists) else mean_dists[-1]
    # t=0 means close (dark), t=1 means far (light)
    t = max(0.0, min(1.0, (d - dist_min) / (dist_max - dist_min + 1e-12)))

    cmd.load(pdb_path, "frame")
    cmd.hide("everything", "frame")
    cmd.show("cartoon", "frame")
    cmd.dss("frame")

    # Color everything gray first
    cmd.color("gray80", "frame")

    # Color cross-strand residues by distance
    rgb = dist_rgb(t)
    cmd.set_color("dist_c", rgb)
    cmd.color("dist_c", "frame and ({s1_sele})")
    cmd.color("dist_c", "frame and ({s2_sele})")

    cmd.set_view(saved_view)

    frame_path = os.path.join("{frame_dir}", f"pymol_{{i:04d}}.png")
    cmd.ray({pw}, {ph})
    cmd.png(frame_path, dpi=150)
    print(f"  PyMOL {{i+1}}/{{n}} (dist={{d:.2f}})")

    cmd.delete("frame")

cmd.quit()
'''

    script_path = os.path.join(frame_dir, "_pymol_render.py")
    with open(script_path, "w") as f:
        f.write(pymol_script)

    print("Rendering PyMOL frames...")
    result = subprocess.run(["pymol", "-cq", script_path])
    os.remove(script_path)
    if result.returncode != 0:
        raise RuntimeError("PyMOL rendering failed")


# ── Matplotlib info panel ──────────────────────────────────────────

def render_info_frames(frame_dir, magnitudes, all_dist_matrices,
                       strand1_residues, strand2_residues,
                       pw, ph, dpi=150, mode="closer"):
    """
    Right panel with:
      - Title + current magnitude
      - Progress bar
      - Pairwise distance matrix (strand1 x strand2)
        Closer mode: cells darken as distances shrink, neon border on contacts
        Farther mode: cells darken as distances grow
    """
    n_frames = len(magnitudes)
    mag_min, mag_max = magnitudes.min(), magnitudes.max()

    # Compute global distance range across all frames for consistent coloring
    all_dists = np.concatenate([m.flatten() for m in all_dist_matrices])
    all_dists = all_dists[~np.isnan(all_dists)]
    d_min_global = max(all_dists.min() - 1, 0)
    d_max_global = all_dists.max() + 1

    cmap = CMAP_CLOSER
    norm = Normalize(vmin=d_min_global, vmax=d_max_global)

    fig_w, fig_h = pw / dpi, ph / dpi
    n1 = len(strand1_residues)
    n2 = len(strand2_residues)

    print(f"Rendering {n_frames} info panels (mode={mode})...")

    for idx in range(n_frames):
        mag = magnitudes[idx]
        t = np.clip((mag - mag_min) / (mag_max - mag_min + 1e-12), 0, 1)
        color = mag_color_at_t(t)
        dist_matrix = all_dist_matrices[idx]

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.5],
                              hspace=0.3, left=0.14, right=0.90, top=0.96, bottom=0.06)

        # ── Title + Progress bar (combined) ─────────────
        ax = fig.add_subplot(gs[0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        title_text = "Distance Steering" if mode == "closer" else "Distance Repulsion"
        ax.text(0.5, 0.92, title_text, fontsize=20,
                fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
        ax.text(0.5, 0.65, f"magnitude = {mag:.2f}σ",
                fontsize=18, ha="center", va="center", color="#444",
                fontfamily="monospace", transform=ax.transAxes)

        # Progress bar in lower portion of this same axes
        bar_y = 0.35
        bar_h = 0.12
        ax.barh(bar_y, 1.0, height=bar_h, color="#e8e8e8", edgecolor="#ccc", linewidth=0.5)
        ax.barh(bar_y, t, height=bar_h, color=color, edgecolor="none")

        # Tick labels
        n_ticks = 5
        tick_vals = np.linspace(mag_min, mag_max, n_ticks)
        for tv in tick_vals:
            tick_t = (tv - mag_min) / (mag_max - mag_min + 1e-12)
            if 0 <= tick_t <= 1:
                ax.text(tick_t, bar_y - bar_h, f"{tv:.0f}", ha="center", va="top",
                        fontsize=10, color="#888")

        ax.plot(t, bar_y + bar_h / 2, 'v', color="black", markersize=8,
                zorder=5, clip_on=False)

        # ── Pairwise distance matrix ────────────────────
        ax = fig.add_subplot(gs[1])

        im = ax.imshow(dist_matrix, cmap=cmap, norm=norm, aspect='equal',
                        interpolation='nearest')

        # Add neon green highlights on contact cells
        for i in range(n1):
            for j in range(n2):
                d = dist_matrix[i, j]
                if not np.isnan(d) and d < CONTACT_THRESHOLD:
                    rect = mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=3,
                        edgecolor='#00ff00',
                        facecolor='none',
                        zorder=10,
                    )
                    ax.add_patch(rect)

        # Axis labels: residue indices
        ax.set_xticks(range(n2))
        ax.set_xticklabels([str(r) for r in strand2_residues], fontsize=13)
        ax.set_yticks(range(n1))
        ax.set_yticklabels([str(r) for r in strand1_residues], fontsize=13)

        ax.set_xlabel("Strand 2", fontsize=16)
        ax.set_ylabel("Strand 1", fontsize=16)

        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Cβ-Cβ Distance (Å)", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("#666")

        fig.savefig(os.path.join(frame_dir, f"info_{idx:04d}.png"), dpi=dpi)
        plt.close(fig)

        if (idx + 1) % 20 == 0 or idx == n_frames - 1:
            print(f"  Info panel {idx+1}/{n_frames}")


# ── Composite ──────────────────────────────────────────────────────

def composite_frames(frame_dir, n_frames):
    print("Compositing dual panels...")
    for idx in range(n_frames):
        pymol_path = os.path.join(frame_dir, f"pymol_{idx:04d}.png")
        info_path  = os.path.join(frame_dir, f"info_{idx:04d}.png")
        out_path   = os.path.join(frame_dir, f"final_{idx:04d}.png")

        if not os.path.exists(pymol_path) or not os.path.exists(info_path):
            print(f"  Skipping frame {idx}: missing files")
            continue

        pymol_img = Image.open(pymol_path).convert("RGB")
        info_img  = Image.open(info_path).convert("RGB")

        canvas_h = max(pymol_img.height, info_img.height)
        canvas_w = pymol_img.width + info_img.width

        composite = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        pymol_y = (canvas_h - pymol_img.height) // 2
        info_y  = (canvas_h - info_img.height) // 2

        composite.paste(pymol_img, (0, pymol_y))
        composite.paste(info_img, (pymol_img.width, info_y))
        composite.save(out_path)

        if (idx + 1) % 20 == 0 or idx == n_frames - 1:
            print(f"  Composited {idx+1}/{n_frames}")


# ── ffmpeg ──────────────────────────────────────────────────────────

def stitch_video(frame_dir, output_mp4, fps):
    print(f"\nStitching video ({fps} fps)...")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "final_%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_mp4,
    ], check=True)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render dual-panel steering sweep animation with distance matrix"
    )
    parser.add_argument("pdb_folder",
                        help="Folder with frame_NNNN_mag_X.XXXX.pdb files")
    parser.add_argument("--strand1", required=True,
                        help="Comma-separated 0-based residue indices for strand 1")
    parser.add_argument("--strand2", required=True,
                        help="Comma-separated 0-based residue indices for strand 2")
    parser.add_argument("--n_pairs", type=int, default=4,
                        help="Number of evenly-spaced cross-strand pairs to show (default 4)")
    parser.add_argument("--mode", choices=["closer", "farther"], default="closer",
                        help="'closer' = darkens as strands approach; "
                             "'farther' = darkens as strands separate")
    parser.add_argument("--output", default="steering_sweep.mp4")
    parser.add_argument("--width", type=int, default=800,
                        help="Width of EACH panel in pixels")
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--contact_threshold", type=float, default=8.0,
                        help="Distance threshold for neon contact highlight (Å)")
    args = parser.parse_args()

    global CONTACT_THRESHOLD
    CONTACT_THRESHOLD = args.contact_threshold

    strand1_full = [int(x.strip()) for x in args.strand1.split(",")]
    strand2_full = [int(x.strip()) for x in args.strand2.split(",")]

    # Subsample to n_pairs evenly spaced residues from each strand
    n_pairs = min(args.n_pairs, len(strand1_full), len(strand2_full))
    idx1 = np.linspace(0, len(strand1_full) - 1, n_pairs, dtype=int)
    idx2 = np.linspace(0, len(strand2_full) - 1, n_pairs, dtype=int)
    strand1_residues = [strand1_full[i] for i in idx1]
    strand2_residues = [strand2_full[i] for i in idx2]

    print(f"Strand 1 (full): {strand1_full}")
    print(f"Strand 2 (full): {strand2_full}")
    print(f"Showing {n_pairs} pairs:")
    print(f"  Strand 1 selected: {strand1_residues}")
    print(f"  Strand 2 selected: {strand2_residues}")
    print(f"Mode: {args.mode}")

    # Find and sort PDBs
    pdb_files = sorted(glob.glob(os.path.join(args.pdb_folder, "frame_*.pdb")))
    if not pdb_files:
        print(f"No frame_*.pdb files found in {args.pdb_folder}")
        return

    n_frames = len(pdb_files)
    print(f"Found {n_frames} PDB files")
    print(f"Output: {args.output} ({args.width*2}x{args.height} @ {args.fps} fps)")

    # Extract magnitudes from filenames
    magnitudes = np.array([parse_magnitude(f) for f in pdb_files])
    if None in magnitudes:
        print("Warning: couldn't parse magnitude from some filenames, using linear ramp")
        magnitudes = np.linspace(0, 15, n_frames)
    print(f"Magnitude range: {magnitudes.min():.4f} → {magnitudes.max():.4f}")

    # Compute pairwise distance matrices for each frame
    print("Computing cross-strand distance matrices...")
    all_dist_matrices = []
    for f in pdb_files:
        dm = compute_cross_strand_distances(f, strand1_residues, strand2_residues)
        all_dist_matrices.append(dm)

    # Print summary
    mean_dists = [np.nanmean(m) for m in all_dist_matrices]
    print(f"Mean cross-strand distance range: {min(mean_dists):.2f} → {max(mean_dists):.2f} Å")

    # Save for later analysis
    np.save(os.path.join(args.pdb_folder, "magnitudes.npy"), magnitudes)
    np.save(os.path.join(args.pdb_folder, "mean_distances.npy"), np.array(mean_dists))

    frame_dir = tempfile.mkdtemp(prefix="steering_sweep_")

    try:
        render_pymol_frames(
            os.path.abspath(args.pdb_folder), frame_dir,
            magnitudes, strand1_full, strand2_full,
            mean_dists, args.width, args.height,
        )

        render_info_frames(
            frame_dir, magnitudes, all_dist_matrices,
            strand1_residues, strand2_residues,
            args.width, args.height, args.dpi,
            mode=args.mode,
        )

        composite_frames(frame_dir, n_frames)
        stitch_video(frame_dir, args.output, args.fps)
        print(f"\nDone! → {args.output}")

    finally:
        for f in glob.glob(os.path.join(frame_dir, "*.png")):
            os.remove(f)
        for f in glob.glob(os.path.join(frame_dir, "*.py")):
            os.remove(f)
        if os.path.isdir(frame_dir):
            os.rmdir(frame_dir)


if __name__ == "__main__":
    main()