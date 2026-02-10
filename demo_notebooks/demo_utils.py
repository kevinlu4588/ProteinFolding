import numpy as np
import py3Dmol


# ---------------------------------------------------------------------------
# PDB preprocessing
# ---------------------------------------------------------------------------

def fix_pdb_for_visualization(pdb_string, max_ca_dist=4.0, target_ca_dist=3.8):
    """Bridge chain breaks in a PDB string so py3Dmol renders continuous cartoon.

    3Dmol.js breaks cartoon rendering when consecutive CA-CA distance > 4.0 Å.
    This function detects such gaps and rigidly shifts all downstream atoms to
    close each gap to *target_ca_dist*. Local residue geometry is preserved;
    only inter-residue spacing is adjusted.  For visualization only.
    """
    lines = pdb_string.split("\n")

    # Collect CA positions and map residue numbers to line indices
    ca_list = []          # [(resnum, np.array([x,y,z])), ...]
    residue_lines = {}    # resnum -> [line_index, ...]

    for i, line in enumerate(lines):
        if not line.startswith("ATOM"):
            continue
        resnum = int(line[22:26])
        residue_lines.setdefault(resnum, []).append(i)
        if line[12:16].strip() == "CA":
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            ca_list.append((resnum, np.array([x, y, z])))

    if len(ca_list) < 2:
        return pdb_string

    # Walk N→C, accumulating a shift whenever a gap is found
    cumulative_shift = np.zeros(3)
    res_shift = {ca_list[0][0]: np.zeros(3)}

    for idx in range(len(ca_list) - 1):
        rn, ca = ca_list[idx]
        rn_next, ca_next = ca_list[idx + 1]

        shifted_ca = ca + res_shift[rn]
        shifted_next = ca_next + cumulative_shift
        dist = np.linalg.norm(shifted_next - shifted_ca)

        if dist > max_ca_dist:
            direction = (shifted_next - shifted_ca) / dist
            desired = shifted_ca + direction * target_ca_dist
            cumulative_shift += desired - shifted_next

        res_shift[rn_next] = cumulative_shift.copy()

    # Apply shifts to ATOM lines
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            resnum = int(line[22:26])
            s = res_shift.get(resnum)
            if s is not None and np.any(s != 0):
                x = float(line[30:38]) + s[0]
                y = float(line[38:46]) + s[1]
                z = float(line[46:54]) + s[2]
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        new_lines.append(line)

    return "\n".join(new_lines)


def fix_pdb_chain_breaks(pdb_string, max_ca_gap=5.0):
    """Fix chain breaks in PDB by shifting residues across gaps (cosmetic only).

    Simpler per-gap fix used by the contact steering notebook.
    """
    lines = pdb_string.split('\n')
    residues = {}
    for i, line in enumerate(lines):
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        atom_name = line[12:16].strip()
        resi = int(line[22:26].strip())
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        if resi not in residues:
            residues[resi] = []
        residues[resi].append((i, atom_name, x, y, z))

    sorted_resi = sorted(residues.keys())
    if len(sorted_resi) < 2:
        return pdb_string

    def get_ca(resi):
        for _, atom_name, x, y, z in residues[resi]:
            if atom_name == 'CA':
                return np.array([x, y, z])
        return None

    lines = list(lines)
    for idx in range(len(sorted_resi) - 1):
        r1, r2 = sorted_resi[idx], sorted_resi[idx + 1]
        ca1, ca2 = get_ca(r1), get_ca(r2)
        if ca1 is None or ca2 is None:
            continue
        dist = np.linalg.norm(ca2 - ca1)
        if dist <= max_ca_gap:
            continue
        direction = (ca2 - ca1) / dist
        target_ca2 = ca1 + direction * max_ca_gap
        shift = target_ca2 - ca2
        for line_idx, atom_name, x, y, z in residues[r2]:
            nx, ny, nz = x + shift[0], y + shift[1], z + shift[2]
            line = lines[line_idx]
            lines[line_idx] = line[:30] + f"{nx:8.3f}{ny:8.3f}{nz:8.3f}" + line[54:]
        residues[r2] = [
            (li, an, x + shift[0], y + shift[1], z + shift[2])
            for li, an, x, y, z in residues[r2]
        ]
    return '\n'.join(lines)


def _parse_backbone_coords(pdb_string):
    """Return {resnum: {'N': xyz, 'CA': xyz, 'C': xyz, 'O': xyz}} from a PDB."""
    coords = {}
    for line in pdb_string.split("\n"):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in ("N", "CA", "C", "O"):
            continue
        resnum = int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        coords.setdefault(resnum, {})[name] = xyz
    return coords


def _get_atom_xyz(pdb_string, resnum, atom_name):
    """Get {x, y, z} dict for a specific atom from a PDB string."""
    for line in pdb_string.split("\n"):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if int(line[22:26].strip()) == resnum and line[12:16].strip() == atom_name:
            return {
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
            }
    return None


# ---------------------------------------------------------------------------
# H-bond visualisation
# ---------------------------------------------------------------------------

def _add_hbond_lines(view, pdb_string, hbond_pairs, viewer=None):
    """Draw solid purple cylinders with capped ends and small spheres on N/O atoms.

    *hbond_pairs* is a list of (res_i, res_j, bond_type, dist) where res_i
    donates N-H and res_j accepts C=O.  Coordinates are read from *pdb_string*
    (which should already be the fixed version if chain breaks were patched).
    """
    if not hbond_pairs:
        return

    backbone = _parse_backbone_coords(pdb_string)

    for res_n, res_o, _bond_type, _dist in hbond_pairs:
        pdb_rn = res_n + 1   # 0-indexed → 1-indexed
        pdb_ro = res_o + 1
        n_xyz = backbone.get(pdb_rn, {}).get("N")
        o_xyz = backbone.get(pdb_ro, {}).get("O")
        if n_xyz is None or o_xyz is None:
            continue

        start = {"x": float(n_xyz[0]), "y": float(n_xyz[1]), "z": float(n_xyz[2])}
        end   = {"x": float(o_xyz[0]), "y": float(o_xyz[1]), "z": float(o_xyz[2])}

        cyl_spec = {
            "start": start, "end": end,
            "radius": 0.12, "color": "purple",
            "fromCap": 1, "toCap": 1,
        }
        n_sphere = {"resi": pdb_rn, "atom": "N"}
        o_sphere = {"resi": pdb_ro, "atom": "O"}
        sphere_style = {"sphere": {"color": "purple", "radius": 0.2}}

        if viewer is not None:
            view.addCylinder(cyl_spec, viewer=viewer)
            view.addStyle(n_sphere, sphere_style, viewer=viewer)
            view.addStyle(o_sphere, sphere_style, viewer=viewer)
        else:
            view.addCylinder(cyl_spec)
            view.addStyle(n_sphere, sphere_style)
            view.addStyle(o_sphere, sphere_style)


# ---------------------------------------------------------------------------
# Protein viewers (notebook 1 – simple highlight)
# ---------------------------------------------------------------------------

def show_protein(pdb_string, highlight_start, highlight_end, title="", width=500, height=400):
    """Show a protein with a highlighted region in purple."""
    pdb_string = fix_pdb_for_visualization(pdb_string)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_string, "pdb")

    view.setStyle({"cartoon": {"color": "lightgray", "opacity": 0.8, "arrows": True}})

    view.addStyle(
        {"resi": list(range(highlight_start + 1, highlight_end + 1))},
        {"cartoon": {"color": "purple", "opacity": 1.0}},
    )

    view.zoomTo()

    if title:
        view.addLabel(
            title,
            {"backgroundColor": "white", "fontColor": "black", "fontSize": 14,
             "position": {"x": 0, "y": 0, "z": 0}, "showBackground": True},
        )

    return view


# ---------------------------------------------------------------------------
# Charge steering viewers (notebook 2)
# H-bond tuples: (res_n, res_o, bond_type, dist) — purple cylinders
# ---------------------------------------------------------------------------

def _charge_strand_style(view, topology, viewer=None):
    """Apply strand1=blue, strand2=red coloring with N/O sticks."""
    s1_resi = list(range(topology.strand1_start + 1, topology.strand1_end + 1))
    s2_resi = list(range(topology.strand2_start + 1, topology.strand2_end + 1))
    kw = {"viewer": viewer} if viewer is not None else {}

    view.addStyle({"resi": s1_resi},
                  {"cartoon": {"color": "blue", "opacity": 1.0}}, **kw)
    view.addStyle({"resi": s2_resi},
                  {"cartoon": {"color": "red", "opacity": 1.0}}, **kw)
    view.addStyle(
        {"resi": s1_resi + s2_resi, "atom": ["N", "O"]},
        {"stick": {"color": "gray", "radius": 0.06, "opacity": 0.3}}, **kw,
    )
    return s1_resi, s2_resi


def _charge_draw_hbonds(view, pdb_string, hbond_pairs, viewer=None):
    """Draw purple H-bond cylinders for charge-style 4-element tuples."""
    if not hbond_pairs:
        return
    kw = {"viewer": viewer} if viewer is not None else {}
    for res_n, res_o, _bond_type, _dist in hbond_pairs:
        start = _get_atom_xyz(pdb_string, res_n + 1, "N")
        end = _get_atom_xyz(pdb_string, res_o + 1, "O")
        if start and end:
            view.addCylinder({
                "start": start, "end": end,
                "radius": 0.12, "color": "purple",
                "fromCap": 1, "toCap": 1,
            }, **kw)
        view.addStyle({"resi": res_n + 1, "atom": "N"},
                      {"sphere": {"color": "purple", "radius": 0.2}}, **kw)
        view.addStyle({"resi": res_o + 1, "atom": "O"},
                      {"sphere": {"color": "purple", "radius": 0.2}}, **kw)


def charge_show_protein(pdb_string, topology=None, hbond_pairs=None,
                        width=500, height=400):
    """Show protein with strand1=blue, strand2=red, H-bonds as purple cylinders."""
    pdb_string = fix_pdb_for_visualization(pdb_string)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_string, "pdb")
    view.setStyle({"cartoon": {"color": "lightgray", "opacity": 0.8, "arrows": True}})

    if topology is not None:
        s1_resi, s2_resi = _charge_strand_style(view, topology)
        _charge_draw_hbonds(view, pdb_string, hbond_pairs)
        view.zoomTo({"resi": s1_resi + s2_resi})
    else:
        view.zoomTo()

    return view


def charge_show_side_by_side(pdb_left, pdb_right, topology,
                             hbond_pairs_left=None, hbond_pairs_right=None,
                             label_left="Baseline", label_right="Steered",
                             width=900, height=450):
    """Show two structures side by side with strand coloring and purple H-bond cylinders."""
    pdb_left = fix_pdb_for_visualization(pdb_left)
    pdb_right = fix_pdb_for_visualization(pdb_right)
    view = py3Dmol.view(width=width, height=height, viewergrid=(1, 2))

    s1_resi = list(range(topology.strand1_start + 1, topology.strand1_end + 1))
    s2_resi = list(range(topology.strand2_start + 1, topology.strand2_end + 1))
    zoom_resi = s1_resi + s2_resi

    for col, (pdb_str, hb) in enumerate([
        (pdb_left, hbond_pairs_left or []),
        (pdb_right, hbond_pairs_right or []),
    ]):
        v = (0, col)
        view.addModel(pdb_str, "pdb", viewer=v)
        view.setStyle(
            {"cartoon": {"color": "lightgray", "opacity": 0.8, "arrows": True}},
            viewer=v,
        )
        _charge_strand_style(view, topology, viewer=v)
        _charge_draw_hbonds(view, pdb_str, hb, viewer=v)
        view.zoomTo({"resi": zoom_resi}, viewer=v)

    view.show()
    print(f"Left: {label_left}  |  Right: {label_right}")


# ---------------------------------------------------------------------------
# Contact / distance steering viewers (notebook 3)
# H-bond tuples: (res_i, res_j, dist) — green cylinders
# ---------------------------------------------------------------------------

def _contact_strand_style(view, topology, viewer=None):
    """Apply strand1=blue, strand2=red coloring (no N/O sticks)."""
    s1_resi = list(range(topology.strand1_start + 1, topology.strand1_end + 1))
    s2_resi = list(range(topology.strand2_start + 1, topology.strand2_end + 1))
    kw = {"viewer": viewer} if viewer is not None else {}

    view.addStyle({"resi": s1_resi},
                  {"cartoon": {"color": "blue", "opacity": 1.0}}, **kw)
    view.addStyle({"resi": s2_resi},
                  {"cartoon": {"color": "red", "opacity": 1.0}}, **kw)
    return s1_resi, s2_resi


def _contact_draw_hbonds(view, pdb_string, hbonds, viewer=None):
    """Draw green H-bond cylinders for contact-style 3-element tuples."""
    if not hbonds:
        return
    kw = {"viewer": viewer} if viewer is not None else {}
    for res_i, res_j, dist in hbonds:
        start_xyz = _get_atom_xyz(pdb_string, res_i + 1, "N")
        end_xyz = _get_atom_xyz(pdb_string, res_j + 1, "O")
        if start_xyz and end_xyz:
            view.addCylinder({
                'start': start_xyz, 'end': end_xyz,
                'radius': 0.12, 'color': '#2ecc71',
                'fromCap': 1, 'toCap': 1,
            }, **kw)
        view.addStyle({"resi": res_i + 1, "atom": "N"},
                      {"sphere": {"color": "#2ecc71", "radius": 0.2}}, **kw)
        view.addStyle({"resi": res_j + 1, "atom": "O"},
                      {"sphere": {"color": "#2ecc71", "radius": 0.2}}, **kw)


def contact_show_protein(pdb_string, topology=None, hbonds=None,
                         width=500, height=400):
    """Show protein with strand1=blue, strand2=red, H-bonds as green cylinders."""
    pdb_fixed = fix_pdb_chain_breaks(pdb_string)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_fixed, "pdb")
    view.setStyle({"cartoon": {"color": "lightgray", "opacity": 0.8, "arrows": True}})

    if topology is not None:
        s1_resi, s2_resi = _contact_strand_style(view, topology)
        view.zoomTo({"resi": s1_resi + s2_resi})
    else:
        view.zoomTo()

    _contact_draw_hbonds(view, pdb_fixed, hbonds)
    return view


def contact_show_side_by_side(pdb_left, pdb_right, topology,
                              hb_left=None, hb_right=None,
                              label_left="Baseline", label_right="Steered",
                              width=900, height=450):
    """Show two structures side by side with strand coloring and green H-bond cylinders."""
    view = py3Dmol.view(width=width, height=height, viewergrid=(1, 2))
    s1_resi = list(range(topology.strand1_start + 1, topology.strand1_end + 1))
    s2_resi = list(range(topology.strand2_start + 1, topology.strand2_end + 1))
    zoom_resi = s1_resi + s2_resi

    hb_lists = [hb_left or [], hb_right or []]
    for col, (pdb_str, hb_pairs) in enumerate(zip([pdb_left, pdb_right], hb_lists)):
        v = (0, col)
        pdb_fixed = fix_pdb_chain_breaks(pdb_str)
        view.addModel(pdb_fixed, "pdb", viewer=v)
        view.setStyle({"cartoon": {"color": "lightgray", "opacity": 0.8, "arrows": True}}, viewer=v)
        _contact_strand_style(view, topology, viewer=v)
        _contact_draw_hbonds(view, pdb_fixed, hb_pairs, viewer=v)
        view.zoomTo({"resi": zoom_resi}, viewer=v)

    view.show()
    print(f"Left: {label_left}  |  Right: {label_right}")
