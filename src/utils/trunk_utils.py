"""
Protein Structure Analysis Utilities
====================================

Utilities for analyzing protein structures predicted by ESMFold, with a focus
on beta-hairpin detection and characterization.

Key functions:
    run_dssp_on_pdb: Run DSSP secondary structure assignment on a PDB file
    detect_hairpins: Detect beta-hairpins from DSSP output
    compute_handedness_from_structure: Determine hairpin handedness (Type I/II)
    get_CB_or_virtual: Get CB coordinates (or virtual CB for glycine)
    visualize_hairpin_handedness_from_cif_or_pdb: 3D visualization with py3Dmol
"""

import pandas as pd
import os
import requests
import torch
# from nnsight import NNsight
from transformers import AutoTokenizer, EsmForProteinFolding

from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


from difflib import SequenceMatcher
import torch
import py3Dmol
from Bio import PDB
from Bio.PDB import DSSP
import tempfile
import warnings
import pandas as pd
def clean_pdb_string(pdb_string):
    """Remove non-standard records that confuse DSSP/mkdssp."""
    skip_prefixes = ("PARENT", "REMARK 220")
    return "\n".join(
        line for line in pdb_string.split("\n")
        if not line.startswith(skip_prefixes)
    )
# ---------- DSSP from outputs ----------
def run_dssp_from_outputs(outputs, model, batch_idx=0):
    """Save ESMFold output to a temp PDB and run DSSP once."""
    detached_outputs = {
        k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
        for k, v in outputs.items()
    }

    pdb_str = model.output_to_pdb(detached_outputs)[batch_idx]
    pdb_str = clean_pdb_string(pdb_str)

    pdb_path = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    with open(pdb_path, "w") as f:
        f.write(pdb_str)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("ESMFold", pdb_path)
    model0 = structure[0]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
            dssp = DSSP(model0, pdb_path, dssp="mkdssp")
    except Exception as e:
        warnings.warn(f"DSSP failed: {e}", RuntimeWarning)
        return None, None, None

    rows = []
    for key in dssp.keys():
        chain_id, (hetatm_flag, resseq, icode) = key
        aa, ss = dssp[key][1], dssp[key][2]
        simp = "E" if ss in ["E", "B"] else ("H" if ss in ["H", "I", "G"] else "C")
        rows.append((chain_id, resseq, aa, ss, simp))
    df = pd.DataFrame(rows, columns=["Chain", "ResNum", "AA", "SecStruct", "SimpleSS"])
    return df

def run_dssp_on_pdb(pdb_path):
    """
    Parse a PDB file, run DSSP, and return:
      - structure: Bio.PDB structure object
      - dssp_df: DataFrame with columns:
          [Chain, ResNum, AA, SecStruct, SimpleSS]
    """
    from Bio.PDB import PDBParser, DSSP
    import warnings
    import pandas as pd

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    model0 = structure[0]

    try:
        # DSSP requires both structure object + path to the PDB file
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
            dssp = DSSP(model0, pdb_path, dssp="mkdssp")
    except Exception as e:
        warnings.warn(f"DSSP failed: {e}", RuntimeWarning)
        return None, None

    rows = []
    for key in dssp.keys():
        chain_id, (hetflag, resseq, icode) = key
        aa = dssp[key][1]  # 1-letter amino acid
        ss = dssp[key][2]  # DSSP annotation (H, E, C, etc.)

        # Simple 3-state mapping: E (beta), H (helix), C (coil)
        simp = (
            "E" if ss in ["E", "B"]
            else "H" if ss in ["H", "I", "G"]
            else "C"
        )

        rows.append((chain_id, resseq, aa, ss, simp))

    df = pd.DataFrame(rows, columns=["Chain", "ResNum", "AA", "SecStruct", "SimpleSS"])
    return structure, df


# ------------------------------------------------------------
# 2. Detect β-hairpins from DSSP (two adjacent beta strands)
# ------------------------------------------------------------
def detect_hairpins(outputs, model, min_len=2, max_loop=5):

    """
    Detect β-hairpins defined as:
      - Two β-strand segments ('E' in SimpleSS) of length >= min_len
      - Separated by a loop of length in [0, max_loop]
    Returns:
      hairpin_detected: bool
      hairpins: list of tuples
        (chain_id, s1_start_idx, s1_end_idx, s2_start_idx, s2_end_idx)
    """

    dssp_df = run_dssp_from_outputs(outputs, model)
    if dssp_df is None:
        print("❌ DSSP failed.")
        return None, None

    hairpins = []

    for chain_id in dssp_df["Chain"].unique():
        cdf = dssp_df[dssp_df["Chain"] == chain_id].reset_index(drop=True)

        strands = []
        start = None

        # Find continuous β-strand segments in SimpleSS
        for i, row in cdf.iterrows():
            if row["SimpleSS"] == "E":
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= min_len:
                    strands.append((start, i - 1))
                start = None

        # Handle strand that reaches to the end
        if start is not None and len(cdf) - start >= min_len:
            strands.append((start, len(cdf) - 1))

        # Pair neighboring strands into hairpins
        for i in range(len(strands) - 1):
            s1_start, s1_end = strands[i]
            s2_start, s2_end = strands[i + 1]
            loop_len = s2_start - s1_end - 1

            if 0 <= loop_len <= max_loop:
                hairpins.append((chain_id, s1_start, s1_end, s2_start, s2_end))

    hairpin_detected = (len(hairpins) > 0)
    return hairpin_detected, hairpins


# ------------------------------------------------------------
# 3. Virtual CB helper (for GLY / missing CB)
# ------------------------------------------------------------
def get_CB_or_virtual(residue):
    """
    Return CB coordinate if present.
    If CB is missing (e.g., Gly), construct a virtual CB from N, CA, C.
    Uses the standard OpenFold/ESMFold virtual CB construction.
    """
    if "CB" in residue:
        return residue["CB"].coord

    # Need N, CA, C to build virtual CB
    N = residue["N"].coord
    CA = residue["CA"].coord
    C = residue["C"].coord

    b = CA - N
    c = C - CA

    # Normalize
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    # Virtual CB direction (OpenFold coefficients)
    v = -0.58273431 * b + 0.56802827 * c + 0.54067466 * np.cross(b, c)
    v = v / np.linalg.norm(v)

    CB = CA + 1.522 * v  # CA–CB bond length ~1.522 Å
    return CB


# ------------------------------------------------------------
# 4. Compute handedness from structure (PDB/CIF only)
# ------------------------------------------------------------
def compute_handedness_from_structure(structure, chain_df, s1_end_idx, s2_start_idx, eps=1e-8):
    """
    Compute hairpin handedness using:
      - residue at end of strand 1 (s1_end_idx)
      - residue at start of strand 2 (s2_start_idx)
    from the given structure and chain_df (subset of dssp_df for one chain).

    triple = (u x v) · n
      u = C1 - N1       (strand direction)
      v = CA2 - CA1     (bridge vector)
      n = CB1 - CA1     (side-chain normal / virtual CB if needed)
    """
    model = structure[0]
    chain_id = chain_df.loc[0, "Chain"]
    chain = model[chain_id]

    res1 = int(chain_df.loc[s1_end_idx, "ResNum"])
    res2 = int(chain_df.loc[s2_start_idx, "ResNum"])

    # Handle possible missing residues gracefully
    try:
        r1 = chain[(" ", res1, " ")]
        r2 = chain[(" ", res2, " ")]
    except KeyError as e:
        raise KeyError(f"Residue not found in structure for ResNum {e}") from e

    N1 = r1["N"].coord
    CA1 = r1["CA"].coord
    C1 = r1["C"].coord
    CB1 = get_CB_or_virtual(r1)
    CA2 = r2["CA"].coord

    u = C1 - N1
    v = CA2 - CA1
    n = CB1 - CA1

    triple = np.dot(np.cross(u, v), n)
    denom = (np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(n)) + eps
    mag = triple / denom

    return np.sign(mag), mag


# ------------------------------------------------------------
# 5. Get vectors for visualization (u, v, n, anchor atoms)
# ------------------------------------------------------------
def get_handedness_vectors_from_structure(structure, chain_df, s1_end_idx, s2_start_idx):
    """
    Build vectors u, v, n and anchor points from the structure.
    Returns a dict with:
      - "u", "v", "n": {start, end}
      - "points": {N1, CA1, C1, CB1, CA2}
    """
    model = structure[0]
    chain_id = chain_df.loc[0, "Chain"]
    chain = model[chain_id]

    res1 = int(chain_df.loc[s1_end_idx, "ResNum"])
    res2 = int(chain_df.loc[s2_start_idx, "ResNum"])

    r1 = chain[(" ", res1, " ")]
    r2 = chain[(" ", res2, " ")]

    N1 = r1["N"].coord
    CA1 = r1["CA"].coord
    C1 = r1["C"].coord
    CB1 = get_CB_or_virtual(r1)
    CA2 = r2["CA"].coord

    u = C1 - N1
    v = CA2 - CA1
    n = CB1 - CA1

    scale = 2.0

    arrows = {
        "u": {"start": N1, "end": N1 + scale * u},
        "v": {"start": CA1, "end": CA1 + scale * v},
        "n": {"start": CA1, "end": CA1 + scale * n},
        "points": {
            "N1": N1,
            "CA1": CA1,
            "C1": C1,
            "CB1": CB1,
            "CA2": CA2,
        },
    }
    return arrows


# ------------------------------------------------------------
# 6. Main visualization function
# ------------------------------------------------------------
def visualize_hairpin_handedness_from_cif_or_pdb(
    cif_or_pdb: str,
    is_cif: bool,
    true_hairpin_seq: str,
    width: int = 600,
    height: int = 600,
):
    """
    From a CIF file:
      - run DSSP
      - detect β-hairpins
      - pick the best-matching hairpin to `true_hairpin_seq`
      - compute handedness from structure
      - build a py3Dmol view with:
          * strands & loop colored
          * u, v, n vectors
          * anchor atoms

    Returns:
      metric_dict, py3Dmol_view

    metric_dict keys:
      - handedness (sign)
      - magnitude (float)
      - similarity (seq similarity to true_hairpin_seq)
      - matched_sequence (full hairpin sequence)
      - hairpin_indices: (chain_id, s1s, s1e, s2s, s2e)
    """
    # --- run DSSP ---
    if is_cif:
        structure, dssp_df = run_dssp_on_cif(cif_or_pdb)
        if dssp_df is None:
            print("❌ DSSP failed.")
            return None, None
    else:
        structure, dssp_df = run_dssp_on_pdb(cif_or_pdb)

    hairpins = detect_hairpins(dssp_df)
    if len(hairpins) == 0:
        print("❌ No β-hairpins detected")
        return None, None

    # Pick best matching hairpin by sequence similarity
    best = {
        "similarity": -1.0,
        "hairpin": None,
        "chain_df": None,
        "handed": None,
        "magnitude": None,
        "matched_sequence": None,
    }

    for chain_id, s1s, s1e, s2s, s2e in hairpins:
        cdf = dssp_df[dssp_df["Chain"] == chain_id].reset_index(drop=True)

        # hairpin sequence from strand1 start to strand2 end
        hairpin_seq = "".join(cdf.loc[s1s:s2e, "AA"].tolist())
        sim = SequenceMatcher(None, true_hairpin_seq, hairpin_seq).ratio()

        hand, mag = compute_handedness_from_structure(structure, cdf, s1e, s2s)

        if sim > best["similarity"]:
            best.update({
                "hairpin": (chain_id, s1s, s1e, s2s, s2e),
                "chain_df": cdf,
                "handed": hand,
                "magnitude": mag,
                "matched_sequence": hairpin_seq,
                "similarity": sim,
            })

    if best["hairpin"] is None:
        print("❌ Failed to pick a hairpin")
        return None, None

    chain_id, s1s, s1e, s2s, s2e = best["hairpin"]
    cdf = best["chain_df"]

    # --- vectors for visualization ---
    arrows = get_handedness_vectors_from_structure(structure, cdf, s1e, s2s)

    # --- convert CIF → PDB string for py3Dmol visualization ---
    io = PDBIO()
    io.set_structure(structure)
    tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    io.save(tmp_pdb)
    with open(tmp_pdb, "r") as f:
        pdb_str = f.read()

    # --- py3Dmol viewer ---
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "lightgrey"}})

    # Color strands + loop
    resnums = cdf["ResNum"].tolist()
    s1_res = resnums[s1s:s1e + 1]
    s2_res = resnums[s2s:s2e + 1]
    loop_res = resnums[s1e + 1:s2s]

    for r in s1_res:
        view.setStyle({"resi": str(r)}, {"cartoon": {"color": "blue"}})
    for r in s2_res:
        view.setStyle({"resi": str(r)}, {"cartoon": {"color": "red"}})
    for r in loop_res:
        view.setStyle({"resi": str(r)}, {"cartoon": {"color": "orange"}})

    # Add arrows
    def add_arrow(start, end, color):
        view.addArrow({
            "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
            "end":   {"x": float(end[0]),   "y": float(end[1]),   "z": float(end[2])},
            "color": color,
            "radius": 0.25,
        })

    add_arrow(arrows["u"]["start"], arrows["u"]["end"], "blue")
    add_arrow(arrows["v"]["start"], arrows["v"]["end"], "green")
    add_arrow(arrows["n"]["start"], arrows["n"]["end"], "yellow")

    # Anchor atoms
    for name, p in arrows["points"].items():
        view.addSphere({
            "center": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "radius": 0.6,
            "color": "white",
        })

    view.zoomTo()

    metric = {
        "handedness": best["handed"],
        "magnitude": best["magnitude"],
        "similarity": best["similarity"],
        "matched_sequence": best["matched_sequence"],
        "hairpin_indices": (chain_id, s1s, s1e, s2s, s2e),
    }

    return metric, view
