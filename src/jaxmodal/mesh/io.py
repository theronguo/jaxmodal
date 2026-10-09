"""Mesh IO: read Abaqus flat .inp (own reader), Gmsh 4.1 ASCII and VTU (meshio); write VTU.

Node orderings: VTK internally. meshio permutes Gmsh hex20/hex27; Abaqus C3D20 equals VTK hex20,
so no ordering map is needed. meshio.read is avoided because it calls sys.exit on errors.
"""

import re
import warnings
from pathlib import Path

import meshio
import numpy as np

from .model import TYPES, Mesh

MESHIO = {"hexahedron": "hex8", "hexahedron20": "hex20", "hexahedron27": "hex27"}
LABEL_MAX = 2**31


def read(path):
    ext = Path(path).suffix.lower()
    if ext == ".inp":
        return _read_inp(path)
    if ext == ".msh":
        return _read_gmsh(path)
    if ext == ".vtu":
        return _read_vtu(path)
    raise ValueError(f"unsupported mesh file extension {ext!r} (use .inp, .msh or .vtu)")


def write(path, mesh):
    """Write a VTU file with `material`/`orientation` cell data. Sets are not written."""
    if Path(path).suffix.lower() != ".vtu":
        raise ValueError("only .vtu can be written")
    names = {v: k for k, v in MESHIO.items()}
    off = mesh.offsets
    rng = {t: slice(off[t], off[t] + len(c)) for t, c in mesh.blocks.items()}
    m = meshio.Mesh(
        mesh.points,
        [(names[t], c) for t, c in mesh.blocks.items()],
        cell_data={
            k: [getattr(mesh, k)[rng[t]] for t in mesh.blocks] for k in ("material", "orientation")
        },
    )
    meshio.vtu.write(path, m)


def _labels(a, what):
    a = np.asarray(a)
    if a.dtype.kind == "f" and not (np.isfinite(a).all() and (a == np.round(a)).all()):
        raise ValueError(f"{what} labels must be integral")
    if a.dtype.kind not in "iuf" or (np.abs(a) >= LABEL_MAX).any():
        raise ValueError(f"{what} labels must be integers below 2**31 in magnitude")
    return a.astype(np.int64)


def _build(points, cells, material, orientation, node_sets, element_sets, hint=""):
    """Assemble a Mesh from file-order data.

    cells: [(type, conn)] in file order; material/orientation and element_sets index elements in
    that order. Blocks are concatenated per type in TYPES order; orphan nodes are removed; sets are
    restricted (warning) and dropped when empty (warning).
    """
    if not cells:
        raise ValueError("no 3D elements in file" + hint)
    types = np.concatenate([np.full(len(c), TYPES.index(t)) for t, c in cells])
    perm = np.argsort(types, kind="stable")  # global index -> file-order index
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    used = np.zeros(len(points), bool)
    for _, c in cells:
        used[c] = True
    new = np.cumsum(used) - 1
    blocks = {
        t: new[np.concatenate([c for s, c in cells if s == t])]
        for t in TYPES
        if any(s == t for s, _ in cells)
    }

    def restrict(kind, sets, keep, remap):
        out = {}
        for name, s in sets.items():
            s = np.unique(np.asarray(s, np.int64))
            k = s[keep[s]] if keep is not None else s
            if len(k) == 0:
                warnings.warn(f"{kind} set {name!r} is empty after reading; dropped", stacklevel=3)
                continue
            if len(k) < len(s):
                msg = f"{kind} set {name!r}: {len(s) - len(k)} members not in the mesh dropped"
                warnings.warn(msg, stacklevel=3)
            out[name] = np.sort(remap[k])
        return out

    return Mesh(
        points=np.asarray(points, np.float64)[used],
        blocks=blocks,
        node_sets=restrict("node", node_sets, used, new),
        element_sets=restrict("element", element_sets, None, inv),
        material=np.asarray(material, np.int64)[perm],
        orientation=np.asarray(orientation, np.int64)[perm],
    )


# --- Abaqus ---------------------------------------------------------------------------------

REJECT = ("PART", "INSTANCE", "ASSEMBLY", "INCLUDE")  # Abaqus/CAE structure
REJECT += ("NGEN", "NFILL", "NCOPY", "NMAP", "ELGEN", "ELCOPY", "SYSTEM")  # generated geometry
LABEL_SET = re.compile(r"(MATERIAL|ORIENTATION)_(\d+)")


def _read_inp(path):
    nlab, xyz, elab, runs = [], [], [], []  # runs: [type, node-label rows] of same-type elements
    sets = {"node": {}, "element": {}}  # upper name -> labels
    spell = {"node": {}, "element": {}}  # upper name -> first spelling
    skipped = set()
    mode, buf = None, []

    def add(kind, name, labels):
        spell[kind].setdefault(name.upper(), name)
        sets[kind].setdefault(name.upper(), []).extend(labels)

    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            if buf:
                raise ValueError("element cut off by a keyword line")
            parts = [p.strip() for p in s[1:].split(",") if p.strip()]
            kw, opt = parts[0].upper(), {}
            for p in parts[1:]:
                k, _, v = p.partition("=")
                opt[k.strip().upper()] = v.strip().strip('"')
            if kw in REJECT:
                raise ValueError(f"*{kw} is not supported (flat .inp files only)")
            mode = None
            if kw == "NODE":
                mode, setname = "node", opt.get("NSET")
            elif kw == "ELEMENT":
                if "TYPE" not in opt:
                    raise ValueError("*ELEMENT without TYPE=")
                t = opt["TYPE"].upper()
                m = re.fullmatch(r"C3D(8|20)[A-Z]*", t)
                if m:
                    mode, setname = "element", opt.get("ELSET")
                    etype, nper = ("hex8", 8) if m[1] == "8" else ("hex20", 20)
                elif t.startswith("C3D"):
                    raise ValueError(f"unsupported solid element type {t}")
                elif t not in skipped:
                    skipped.add(t)
                    warnings.warn(f"skipping element type {t}", stacklevel=3)
            elif kw in ("NSET", "ELSET"):
                if kw == "NSET" and "ELSET" in opt:
                    raise ValueError("*NSET, ELSET= is not supported")
                mode = "node set" if kw == "NSET" else "element set"
                setname, generate = opt[kw], "GENERATE" in opt
            continue
        tok = [t.strip() for t in s.split(",") if t.strip()]
        if mode == "node":
            if len(tok) != 4:
                raise ValueError(f"node line needs a label and 3 coordinates: {s!r}")
            nlab.append(int(tok[0]))
            xyz.append([float(t) for t in tok[1:]])
            if setname:
                add("node", setname, [nlab[-1]])
        elif mode == "element":
            buf += [int(t) for t in tok]
            if len(buf) > nper + 1:
                raise ValueError(f"too many entries for a {etype} element: {buf}")
            if len(buf) == nper + 1:
                elab.append(buf[0])
                if not runs or runs[-1][0] != etype:
                    runs.append((etype, []))
                runs[-1][1].append(buf[1:])
                buf = []
                if setname:
                    add("element", setname, [elab[-1]])
        elif mode in ("node set", "element set"):
            try:
                ids = [int(t) for t in tok]
            except ValueError:
                raise ValueError(f"{mode} {setname!r}: only integer labels are supported") from None
            if generate:
                ids = list(range(ids[0], ids[1] + 1, ids[2] if len(ids) > 2 else 1))
            add(mode.split()[0], setname, ids)
    if buf:
        raise ValueError("element cut off by end of file")

    def lookup(labels, query, what, strict):
        order = np.argsort(labels)
        q = np.asarray(query, np.int64)
        i = np.searchsorted(labels, q, sorter=order).clip(0, max(len(labels) - 1, 0))
        found = labels[order[i]] == q if len(labels) else np.zeros(len(q), bool)
        if strict and not found.all():
            raise ValueError(f"undefined {what} label {q[~found][0]}")
        return order[i], found

    nlab, elab = np.asarray(nlab, np.int64), np.asarray(elab, np.int64)
    for labels, what in ((nlab, "node"), (elab, "element")):
        if len(np.unique(labels)) < len(labels):
            raise ValueError(f"duplicate {what} label")
    cells = [(t, lookup(nlab, np.ravel(r), "node", True)[0].reshape(len(r), -1)) for t, r in runs]

    node_sets = {
        spell["node"][k]: lookup(nlab, v, "node", True)[0] for k, v in sets["node"].items()
    }
    element_sets, n_dropped = {}, 0
    labels = {"MATERIAL": {}, "ORIENTATION": {}}
    for k, v in sets["element"].items():
        idx, found = lookup(elab, v, "element", False)
        n_dropped += int((~found).sum())
        m = LABEL_SET.fullmatch(k)
        if m:
            if int(m[2]) >= LABEL_MAX:
                raise ValueError(f"label in set {spell['element'][k]!r} must be below 2**31")
            labels[m[1]][k] = (int(m[2]), np.unique(idx[found]))
        else:
            element_sets[spell["element"][k]] = idx[found]
    if n_dropped:
        warnings.warn(
            f"{n_dropped} element set labels are not solid elements; dropped", stacklevel=3
        )

    n_el = len(elab)
    out = {}
    for kind, d in labels.items():
        lab, count = np.zeros(n_el, np.int64), np.zeros(n_el, np.int64)
        for value, idx in d.values():
            lab[idx] = value
            count[idx] += 1
        if (count > 1).any():
            raise ValueError(f"an element is in more than one {kind.lower()}_<k> set")
        out[kind] = lab
    return _build(
        np.asarray(xyz, np.float64).reshape(-1, 3),
        cells,
        out["MATERIAL"],
        out["ORIENTATION"],
        node_sets,
        element_sets,
    )


# --- Gmsh -----------------------------------------------------------------------------------

GMSH_MSG = "put every volume in exactly one physical group and don't use Mesh.SaveAll"


def _gmsh_header(path):
    """Version check, then $PhysicalNames -> {(dim, tag): name}, $Entities -> {(dim, ent): tags}."""
    names, phys = {}, {}
    with open(path, "rb") as f:
        if f.readline().strip() != b"$MeshFormat":
            raise ValueError("not a Gmsh .msh file")
        if f.readline().split()[:2] != [b"4.1", b"0"]:
            raise ValueError("save as Gmsh 4.1 ASCII (Mesh.MshFileVersion = 4.1, Mesh.Binary = 0)")
        for line in f:
            line = line.strip()
            if line == b"$PhysicalNames":
                for _ in range(int(f.readline())):
                    d, t, name = f.readline().decode().split(maxsplit=2)
                    names[int(d), int(t)] = name.strip().strip('"')
            elif line == b"$Entities":
                counts = [int(x) for x in f.readline().split()[:4]]
                for dim, n in enumerate(counts):
                    for _ in range(n):
                        tok = f.readline().split()
                        k = 4 if dim == 0 else 7  # tag, then a point or a bounding box
                        phys[dim, int(tok[0])] = [int(x) for x in tok[k + 1 : k + 1 + int(tok[k])]]
            elif line == b"$PartitionedEntities":
                raise ValueError("partitioned Gmsh files are not supported; save unpartitioned")
            elif line == b"$Nodes":
                break
    return names, phys


def _read_gmsh(path):
    names, phys = _gmsh_header(path)
    vols = [p for (d, _), p in phys.items() if d == 3]
    if any(len(p) > 1 for p in vols) or (any(vols) and not all(vols)):
        raise ValueError(GMSH_MSG)
    try:
        m = meshio.gmsh.read(path)
    except ValueError as e:
        if "Incompatible cell data" in str(e):
            raise ValueError(GMSH_MSG) from e
        raise
    groups, cells, material = {}, [], []
    for block, geo in zip(m.cells, m.cell_data["gmsh:geometrical"], strict=True):
        tags = [phys.get((block.dim, int(g)), []) for g in geo]
        if block.dim < 3:
            for row, ts in zip(block.data, tags, strict=True):
                for t in ts:
                    groups.setdefault((block.dim, t), []).append(row)
        elif block.type in MESHIO:
            cells.append((MESHIO[block.type], block.data))
            material += [ts[0] if ts else 0 for ts in tags]
        else:
            raise ValueError(f"unsupported 3D cell type {block.type}")
    node_sets = {}
    for (d, t), rows in groups.items():
        name = names.get((d, t), f"physical_{d}_{t}")
        if name in node_sets:
            raise ValueError(f"duplicate physical group name {name!r}")
        node_sets[name] = np.unique(np.concatenate([np.ravel(r) for r in rows]))
    hint = " (are your volumes in a physical group?)"
    return _build(m.points, cells, material, np.zeros(len(material), np.int64), node_sets, {}, hint)


# --- VTU ------------------------------------------------------------------------------------


def _read_vtu(path):
    m = meshio.vtu.read(path)
    cells, labels = [], {"material": [], "orientation": []}
    for i, block in enumerate(m.cells):
        if block.dim < 3:
            continue
        if block.type not in MESHIO:
            raise ValueError(f"unsupported 3D cell type {block.type}")
        cells.append((MESHIO[block.type], block.data))
        for k in labels:
            data = m.cell_data[k][i] if k in m.cell_data else np.zeros(len(block.data))
            labels[k].append(_labels(np.ravel(data), k))
    cat = {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in labels.items()}
    return _build(m.points, cells, cat["material"], cat["orientation"], {}, {})
