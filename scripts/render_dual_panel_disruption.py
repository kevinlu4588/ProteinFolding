#!/usr/bin/env python
"""
Render a dual-panel animation of hairpin disruption:
  Left:  3D protein structure (PyMOL ray-traced)
  Right: Intervention visualization (matplotlib)

Usage:
    python render_dual_panel.py <pdb_folder> <output.mp4>
    python render_dual_panel.py <pdb_folder> <output.mp4> --fps 10 --width 800 --height 700

Requires: pymol, ffmpeg, matplotlib, numpy, pillow
"""

import argparse
import os
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


# ── Color scheme ────────────────────────────────────────────────────

def strand_color_at_t(t):
    """Light sky blue -> deep royal blue."""
    r = 0.68 - t * 0.63
    g = 0.85 - t * 0.70
    b = 1.00 - t * 0.30
    return (r, g, b)


# ── PyMOL rendering ────────────────────────────────────────────────

def render_pymol_frames(pdb_folder, frame_dir, magnitudes, pw, ph,
                        strand1_resi, turn_resi, strand2_resi, display_max=None):
    """
    Load all PDBs as states of one object (preserves the nice multi-state
    alignment). For each frame, extract that state into a temporary single-state
    object, color it, ray-trace, then delete the temp object.
    """

    mag_list = json.dumps(magnitudes.tolist())
    dm = display_max if display_max is not None else float(magnitudes.max())

    pymol_script = f'''
import os, glob, json
from pymol import cmd

pdb_files = sorted(glob.glob(os.path.join("{pdb_folder}", "frame_*.pdb")))
magnitudes = json.loads('{mag_list}')
mag_min = min(magnitudes)
mag_max = {dm}
n = len(pdb_files)

def strand_rgb(t):
    r = 0.68 - t * 0.63
    g = 0.85 - t * 0.70
    b = 1.00 - t * 0.30
    return [r, g, b]

# Load all as states of one object
for i, pdb_path in enumerate(pdb_files):
    if i == 0:
        cmd.load(pdb_path, "sweep")
    else:
        cmd.load(pdb_path, "sweep", state=i + 1)

# Set up view from all-states object
cmd.hide("everything")
cmd.show("cartoon", "sweep")
cmd.zoom("sweep", state=1)
cmd.orient("sweep")
saved_view = cmd.get_view()

cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("antialias", 2)
cmd.set("ray_shadows", 1)
cmd.set("ray_trace_mode", 1)
cmd.set("ray_opaque_background", 1)
cmd.bg_color("white")

n_states = cmd.count_states("sweep")

for state in range(1, n_states + 1):
    mag = magnitudes[state - 1] if state - 1 < len(magnitudes) else magnitudes[-1]
    t = max(0.0, min(1.0, (mag - mag_min) / (mag_max - mag_min + 1e-12)))

    # Extract this state into a temporary single-state object
    cmd.create("frame", "sweep", source_state=state, target_state=1)
    cmd.hide("everything", "frame")
    cmd.show("cartoon", "frame")

    # Color the single-state object — this works reliably
    cmd.color("gray80", "frame")
    cmd.color("splitpea", f"frame and resi {turn_resi}")

    rgb = strand_rgb(t)
    cmd.set_color("strand_c", rgb)
    cmd.color("strand_c", f"frame and resi {strand1_resi}")
    cmd.color("strand_c", f"frame and resi {strand2_resi}")

    # Hide the multi-state object, show only the colored frame
    cmd.disable("sweep")
    cmd.enable("frame")
    cmd.set_view(saved_view)

    frame_path = os.path.join("{frame_dir}", f"pymol_{{(state-1):04d}}.png")
    cmd.ray({pw}, {ph})
    cmd.png(frame_path, dpi=150)
    print(f"  PyMOL {{state}}/{{n_states}} (mag={{mag:.5f}}, t={{t:.3f}})")

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

def render_info_frames(frame_dir, magnitudes, seq,
                       strand1_0, strand1_1, turn_0, turn_1, strand2_0, strand2_1,
                       pw, ph, dpi=150, display_max=None):
    """
    Sequence strip with charge arrows and repulsion annotation.
    Matched to attraction script layout (block_size, spacing, fonts).
    """

    mag_min = 0.0
    actual_max = 0.19 
    norm_max = display_max if display_max is not None else actual_max
    
    n_frames = len(magnitudes)
    hp_start, hp_end = strand1_0, strand2_1
    hp_len = hp_end - hp_start
    residues = list(range(hp_start, hp_end))

    fig_w, fig_h = pw / dpi, ph / dpi

    # Layout constants — matched to attraction script
    spacing_factor = 3.4
    region_gap = 4.0
    block_size = 3.6
    y_offset = 4.0
    plus_h_offset = -1.4
    
    # Precompute x positions with gaps between regions
    def _build_x_positions(residues, strand1_0, strand1_1, turn_0, turn_1, strand2_0):
        positions = []
        x = 0.0
        prev_region = None
        for res_idx in residues:
            if strand1_0 <= res_idx < strand1_1:
                cur_region = "strand1"
            elif turn_0 <= res_idx < turn_1:
                cur_region = "turn"
            else:
                cur_region = "strand2"
            if prev_region is not None:
                if cur_region != prev_region:
                    x += region_gap
                else:
                    x += spacing_factor
            prev_region = cur_region
            positions.append(x)
        return positions

    print(f"Rendering {n_frames} info panels (Max Label: {actual_max})...")

    for idx, raw_mag in enumerate(magnitudes):
        display_mag = raw_mag if magnitudes.max() <= 0.2 else raw_mag * actual_max
        t = np.clip((display_mag - mag_min) / (norm_max - mag_min + 1e-12), 0, 1)
        sc = strand_color_at_t(t)

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(4, 1, height_ratios=[0.8, 0.4, 3.2, 1.2],
                              hspace=0.45, left=0.08, right=0.92, top=0.95, bottom=0.05)

        # ── Title + magnitude ───────────────────────────────
        ax = fig.add_subplot(gs[0])
        ax.axis("off")
        ax.text(0.5, 0.8, "Charge Repulsion Intervention", fontsize=22, fontweight="bold", ha="center")
        ax.text(0.5, 0.3, f"magnitude = {display_mag:.5f} σ", fontsize=18, ha="center", color="#555")

        # ── Progress bar ────────────────────────────────────
        ax = fig.add_subplot(gs[1])
        ax.axis("off")
        ax.barh(0.5, 1.0, height=0.4, color="#e8e8e8")
        ax.barh(0.5, t, height=0.4, color=sc)

        # ── Sequence strip + arrows ─────────────────────────
        ax = fig.add_subplot(gs[2])
        x_positions = _build_x_positions(residues, strand1_0, strand1_1, turn_0, turn_1, strand2_0)
        x_max = max(x_positions) if x_positions else 10
        ax.set_xlim(-4.0, x_max + 4.0)
        ax.set_ylim(-4.5, 8.5) 
        ax.axis("off")
        ax.set_aspect("equal")

        s1_x, s2_x, turn_x = [], [], []

        for pos, res_idx in enumerate(residues):
            scaled_x = x_positions[pos]
            aa = seq[res_idx] if res_idx < len(seq) else "?"

            if strand1_0 <= res_idx < strand1_1:
                color, region, target_list = sc, "strand", s1_x
            elif turn_0 <= res_idx < turn_1:
                color, region, target_list = "#8fbc8f", "turn", turn_x
            elif strand2_0 <= res_idx < strand2_1:
                color, region, target_list = sc, "strand", s2_x
            else:
                color, region, target_list = "#d4d4d4", "other", []
            
            if target_list is not None: target_list.append(scaled_x)

            # Boxes
            rect = plt.Rectangle((scaled_x - block_size/2, y_offset - block_size/2), 
                                 block_size, block_size,
                                 facecolor=color, edgecolor="white", linewidth=1.5, zorder=2)
            ax.add_patch(rect)

            text_color = "white" if region == "strand" and t > 0.25 else "#333"
            ax.text(scaled_x, y_offset, aa, ha="center", va="center", fontsize=18,
                    fontweight="bold", color=text_color, zorder=3)

            arrow_scale = t * 3.5
            arrow_base_y = y_offset + (block_size / 2) + 1.0 

            if region == "strand" and t > 0:
                ax.arrow(scaled_x, arrow_base_y, 0, arrow_scale,
                        width=0.2,
                        head_width=0.8,
                        head_length=0.5, 
                        length_includes_head=False, 
                        color=sc, 
                        edgecolor="white", 
                        zorder=6,
                        linewidth=0.5,
                        clip_on=False)

        s1_mid, s2_mid, turn_mid = [np.mean(lst) if lst else 0 for lst in [s1_x, s2_x, turn_x]]

        # "+" signs
        if t > 0.1:
            for xpos in [s1_mid, s2_mid]:
                ax.text(xpos + plus_h_offset, arrow_base_y + arrow_scale + 0.8, "+",
                        fontsize=32, fontweight="bold", color=sc, zorder=6,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        # Region labels
        label_y = y_offset - 4.0
        ax.text(s1_mid, label_y, "Strand 1", ha="center", fontsize=18, color=sc, fontweight="bold")
        ax.text(turn_mid, label_y, "Turn", ha="center", fontsize=18, color="#6b8e6b")
        ax.text(s2_mid, label_y, "Strand 2", ha="center", fontsize=18, color=sc, fontweight="bold")

        # Repulsion
        if t > 0.0:
            rep_y = label_y - 1.5
            ax.annotate("", xy=(max(s1_x) + 0.8, rep_y), xytext=(min(s2_x) - 0.8, rep_y),
                        arrowprops=dict(arrowstyle="<->", color=sc, lw=2.5 + t*2))
            ax.text((s1_mid + s2_mid)/2, rep_y - 2.0, "repulsion", 
                    ha="center", fontsize=18, color=sc, style="italic")

        # ── Equation Subplot ────────────────────────
        ax = fig.add_subplot(gs[3])
        ax.axis("off")
        ax.text(0.5, 1.2, "s  ←  s  +  α · σ · d_charge", fontsize=20, ha="center", 
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#ccc"))
        ax.text(0.5, 0.15, "same charge (+) on both strands\n→ electrostatic repulsion",
                fontsize=15, ha="center", color="black", style="italic")

        fig.savefig(os.path.join(frame_dir, f"info_{idx:04d}.png"), dpi=dpi)
        plt.close(fig)
# ── Composite ──────────────────────────────────────────────────────

def composite_frames(frame_dir, n_frames):
    """Side-by-side, preserving native aspect ratios."""

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
        description="Render dual-panel hairpin disruption animation"
    )
    parser.add_argument("pdb_folder", help="Folder with frame_NNNN.pdb files")
    parser.add_argument("--output", default="disruption.mp4", help="Output MP4 path")
    parser.add_argument("--width", type=int, default=800,
                        help="Width of EACH panel in pixels")
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max_std", type=float, default=0.19,
                        help="Max magnitude in σ for display (default: use actual max from magnitudes.npy)")
    parser.add_argument("--strand1", default="2-5",
                        help="Strand 1 resi range for info panel, 1-indexed (default: 2-5)")
    parser.add_argument("--turn", default="6-9",
                        help="Turn resi range, 1-indexed (default: 6-9)")
    parser.add_argument("--strand2", default="10-13",
                        help="Strand 2 resi range for info panel, 1-indexed (default: 10-13)")
    parser.add_argument("--strand1_full", default="2-6",
                        help="Full strand 1 resi range for PyMOL coloring (default: 2-6)")
    parser.add_argument("--strand2_full", default="11-15",
                        help="Full strand 2 resi range for PyMOL coloring (default: 11-15)")
    parser.add_argument("--seq", default="GEWTYDDATKTFTVTE",
                        help="Protein sequence (for amino acid labels)")
    args = parser.parse_args()

    pdb_files = sorted(glob.glob(os.path.join(args.pdb_folder, "frame_*.pdb")))
    if not pdb_files:
        print(f"No frame_*.pdb files found in {args.pdb_folder}")
        return

    n_frames = len(pdb_files)
    print(f"Found {n_frames} PDB files")
    print(f"Output: {args.output} ({args.width*2}x{args.height} @ {args.fps} fps)")

    mag_path = os.path.join(args.pdb_folder, "magnitudes.npy")
    if os.path.exists(mag_path):
        magnitudes = np.load(mag_path)
        print(f"Loaded magnitudes ({len(magnitudes)} values, "
              f"range {magnitudes.min():.5f} - {magnitudes.max():.5f})")
    else:
        magnitudes = np.linspace(0, 1, n_frames)
        print("No magnitudes.npy — using linear ramp")

    if len(magnitudes) != n_frames:
        n_frames = min(len(magnitudes), n_frames)
        magnitudes = magnitudes[:n_frames]
        print(f"Truncated to {n_frames} frames")

    s1a, s1b = [int(x) for x in args.strand1.split("-")]
    ta, tb   = [int(x) for x in args.turn.split("-")]
    s2a, s2b = [int(x) for x in args.strand2.split("-")]

    strand1_0, strand1_1 = s1a - 1, s1b
    turn_0, turn_1       = ta - 1, tb
    strand2_0, strand2_1 = s2a - 1, s2b

    display_max = args.max_std if args.max_std is not None else magnitudes.max()

    frame_dir = tempfile.mkdtemp(prefix="dual_panel_")

    try:
        render_pymol_frames(
            os.path.abspath(args.pdb_folder), frame_dir,
            magnitudes, args.width, args.height,
            args.strand1_full, f"{ta}-{tb}", args.strand2_full,
            display_max,
        )

        render_info_frames(
            frame_dir, magnitudes, args.seq,
            strand1_0, strand1_1, turn_0, turn_1, strand2_0, strand2_1,
            args.width, args.height, args.dpi,
            display_max,
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