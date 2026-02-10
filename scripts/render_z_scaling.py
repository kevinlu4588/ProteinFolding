#!/usr/bin/env python
"""
Render a dual-panel animation of z-scale sweep:
  Left:  3D protein structure (PyMOL ray-traced)
  Right: Info panel with z-scale indicator + running mean pairwise CA distance plot

Usage:
    python render_z_sweep.py <pdb_folder> --output sweep.mp4
    python render_z_sweep.py <pdb_folder> --output sweep.mp4 --fps 10 --width 800 --height 700

The PDB folder should contain files named like:
    frame_000_zscale_0.0000.pdb
    frame_001_zscale_0.0253.pdb
    ...

Requires: pymol, ffmpeg, matplotlib, numpy, pillow, biopython (optional, for CA parsing)
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
import matplotlib.patheffects as pe
from PIL import Image


# ── Parse z-scale from filename ─────────────────────────────────────

def parse_z_scale(filename):
    """Extract z-scale value from filename like frame_003_zscale_0.1234.pdb"""
    m = re.search(r"zscale_([\d]+\.[\d]+)\.pdb", os.path.basename(filename))
    if m:
        return float(m.group(1))
    return None


# ── Compute mean pairwise CA distance from PDB ─────────────────────

def get_ca_coords(pdb_path):
    """Extract CA atom coordinates from a PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords)


def mean_pairwise_ca_distance(pdb_path):
    """Compute mean pairwise CA-CA distance from a PDB file."""
    ca = get_ca_coords(pdb_path)
    if len(ca) < 2:
        return 0.0
    # Pairwise distances (upper triangle only)
    diff = ca[:, None, :] - ca[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    n = len(ca)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return dists[mask].mean()


# ── Color scheme ────────────────────────────────────────────────────

def z_color_at_t(t):
    """Light green (low z-scale) -> Neon green (high z-scale)."""
    # 0.0 -> light/pale green, 1.0 -> neon/bright green
    r = 0.75 - t * 0.65   # 0.75 -> 0.10
    g = 0.90 - t * 0.10   # 0.90 -> 0.80
    b = 0.65 - t * 0.60   # 0.65 -> 0.05
    return (r, g, b)


# ── PyMOL rendering ────────────────────────────────────────────────

def render_pymol_frames(pdb_folder, frame_dir, z_scales, pw, ph):
    """Ray-trace each PDB frame in PyMOL, coloring by z-scale.

    Loads each PDB independently (not as multi-state) so that PyMOL
    computes secondary structure correctly for every frame.
    """

    zs_list = json.dumps(z_scales.tolist())
    zs_max = float(z_scales.max())
    zs_min = float(z_scales.min())

    # Find the frame closest to z_scale=1.0 for camera setup
    baseline_idx = int(np.argmin(np.abs(z_scales - 1.0)))

    pymol_script = f'''
import os, glob, json
from pymol import cmd

pdb_files = sorted(glob.glob(os.path.join("{pdb_folder}", "frame_*.pdb")))
z_scales = json.loads('{zs_list}')
z_min = {zs_min}
z_max = {zs_max}
baseline_idx = {baseline_idx}
n = len(pdb_files)

def z_rgb(t):
    r = 0.75 - t * 0.65
    g = 0.90 - t * 0.10
    b = 0.65 - t * 0.60
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

# First pass: load the baseline (z~1.0) frame to set the camera
cmd.load(pdb_files[baseline_idx], "ref")
cmd.hide("everything", "ref")
cmd.show("cartoon", "ref")
cmd.dss("ref")
cmd.orient("ref")
saved_view = cmd.get_view()
cmd.delete("ref")

# Second pass: render each frame independently
for i, pdb_path in enumerate(pdb_files):
    zs = z_scales[i] if i < len(z_scales) else z_scales[-1]
    t = max(0.0, min(1.0, (zs - z_min) / (z_max - z_min + 1e-12)))

    cmd.load(pdb_path, "frame")
    cmd.hide("everything", "frame")
    cmd.show("cartoon", "frame")
    cmd.dss("frame")  # assign secondary structure for this frame

    rgb = z_rgb(t)
    cmd.set_color("zscale_c", rgb)
    cmd.color("zscale_c", "frame")

    cmd.set_view(saved_view)

    frame_path = os.path.join("{frame_dir}", f"pymol_{{i:04d}}.png")
    cmd.ray({pw}, {ph})
    cmd.png(frame_path, dpi=150)
    print(f"  PyMOL {{i+1}}/{{n}} (z_scale={{zs:.4f}})")

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

def render_info_frames(frame_dir, z_scales, ca_distances, pw, ph, dpi=150):
    """
    Right panel with:
      - Title + current z-scale value
      - Color-coded progress bar
      - Running plot of mean pairwise CA distance vs z-scale
    """
    n_frames = len(z_scales)
    z_min, z_max = z_scales.min(), z_scales.max()
    d_min, d_max = ca_distances.min(), ca_distances.max()
    d_pad = (d_max - d_min) * 0.1 + 0.5  # padding for y-axis

    fig_w, fig_h = pw / dpi, ph / dpi

    print(f"Rendering {n_frames} info panels...")

    for idx in range(n_frames):
        zs = z_scales[idx]
        t = np.clip((zs - z_min) / (z_max - z_min + 1e-12), 0, 1)
        color = z_color_at_t(t)

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(4, 1, height_ratios=[0.6, 0.3, 0.15, 2.5],
                              hspace=0.5, left=0.15, right=0.90, top=0.93, bottom=0.10)

        # ── Title ───────────────────────────────────────
        ax = fig.add_subplot(gs[0])
        ax.axis("off")
        ax.text(0.5, 0.75, "Z-Scale Sweep", fontsize=22,
                fontweight="bold", ha="center", va="center")
        ax.text(0.5, 0.15, f"z-scale = {zs:.4f}",
                fontsize=20, ha="center", va="center", color="#444",
                fontfamily="monospace")

        # ── Progress bar ────────────────────────────────
        ax = fig.add_subplot(gs[1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Background bar
        ax.barh(0.5, 1.0, height=0.6, color="#e8e8e8", edgecolor="#ccc", linewidth=0.5)
        # Filled portion
        ax.barh(0.5, t, height=0.6, color=color, edgecolor="none")

        # Tick labels
        for tick_val in [0.0, 0.5, 1.0, 1.5, 2.0]:
            tick_t = (tick_val - z_min) / (z_max - z_min + 1e-12)
            if 0 <= tick_t <= 1:
                ax.text(tick_t, -0.15, f"{tick_val:.1f}", ha="center", va="top",
                        fontsize=9, color="#888")

        # Marker for current position
        ax.plot(t, 0.5, 'v', color="black", markersize=8, zorder=5, clip_on=False)

        # ── Label ───────────────────────────────────────
        ax = fig.add_subplot(gs[2])
        ax.axis("off")
        ax.text(0.5, 0.5, "pair representation scale factor",
                fontsize=11, ha="center", va="center", color="#999", style="italic")

        # ── Running CA distance plot ────────────────────
        ax = fig.add_subplot(gs[3])

        # Plot all data as faint line
        ax.plot(z_scales, ca_distances, color="#ddd", linewidth=1.5, zorder=1)

        # Plot up to current frame with color gradient
        if idx > 0:
            for i in range(idx):
                seg_t = np.clip((z_scales[i] - z_min) / (z_max - z_min + 1e-12), 0, 1)
                seg_color = z_color_at_t(seg_t)
                ax.plot(z_scales[i:i+2], ca_distances[i:i+2],
                        color=seg_color, linewidth=2.5, zorder=2)

        # Current point
        ax.plot(zs, ca_distances[idx], 'o', color=color, markersize=10,
                zorder=4, markeredgecolor="white", markeredgewidth=2)

        # Baseline marker at z=1.0
        baseline_idx = np.argmin(np.abs(z_scales - 1.0))
        ax.axvline(x=1.0, color="#ccc", linestyle="--", linewidth=1, zorder=0)
        ax.text(1.0, d_max + d_pad * 0.6, "baseline\n(z=1.0)",
                ha="center", va="bottom", fontsize=9, color="#aaa")

        ax.set_xlim(z_min - 0.05, z_max + 0.05)
        ax.set_ylim(d_min - d_pad, d_max + d_pad)
        ax.set_xlabel("z-scale", fontsize=13)
        ax.set_ylabel("Mean Pairwise CA Distance (Å)", fontsize=13)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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
        description="Render dual-panel z-scale sweep animation"
    )
    parser.add_argument("pdb_folder", help="Folder with frame_NNN_zscale_X.XXXX.pdb files")
    parser.add_argument("--output", default="z_sweep.mp4", help="Output MP4 path")
    parser.add_argument("--width", type=int, default=800,
                        help="Width of EACH panel in pixels")
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # Find and sort PDBs
    pdb_files = sorted(glob.glob(os.path.join(args.pdb_folder, "frame_*.pdb")))
    if not pdb_files:
        print(f"No frame_*.pdb files found in {args.pdb_folder}")
        return

    n_frames = len(pdb_files)
    print(f"Found {n_frames} PDB files")
    print(f"Output: {args.output} ({args.width*2}x{args.height} @ {args.fps} fps)")

    # Extract z-scales from filenames
    z_scales = np.array([parse_z_scale(f) for f in pdb_files])
    if None in z_scales:
        print("Warning: couldn't parse z-scale from some filenames, using linear ramp")
        z_scales = np.linspace(0, 2, n_frames)
    print(f"Z-scale range: {z_scales.min():.4f} → {z_scales.max():.4f}")

    # Compute mean pairwise CA distance for each frame
    print("Computing mean pairwise CA distances...")
    ca_distances = np.array([mean_pairwise_ca_distance(f) for f in pdb_files])
    print(f"CA distance range: {ca_distances.min():.2f} → {ca_distances.max():.2f} Å")

    # Save for later use
    np.save(os.path.join(args.pdb_folder, "z_scales.npy"), z_scales)
    np.save(os.path.join(args.pdb_folder, "ca_distances.npy"), ca_distances)

    frame_dir = tempfile.mkdtemp(prefix="z_sweep_panel_")

    try:
        render_pymol_frames(
            os.path.abspath(args.pdb_folder), frame_dir,
            z_scales, args.width, args.height,
        )

        render_info_frames(
            frame_dir, z_scales, ca_distances,
            args.width, args.height, args.dpi,
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