"""Solid-shell accuracy suite (PR 16c): `ssh8` against Srinivas–Rao plates, a clamped cylinder
and a clamped skew plate, the last two against graded hex27 references. Rules and setups:
docs/verification.md#solid-shell-accuracy-pr-16c; plan docs/plans/pr-16c-accuracy.md.

python -m benchmarks.shell_accuracy --parts srao cyl skew
Each part reruns its keys and replaces them in results/shell_accuracy.json (a progress line per
run).
"""

from jaxmodal import config

config.enable_x64()  # before any JAX constant: ISO below, PAG/ORTHO in tests.helpers

import argparse
import dataclasses
import functools
import hashlib
import json
import resource
import subprocess
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla
from scipy.optimize import brentq

from benchmarks import harness
from jaxmodal.eigen import reference_modes
from jaxmodal.materials import Laminate, Material, Ply, isotropic
from jaxmodal.mesh import box_mesh, solid_shell
from jaxmodal.operators import assemble, dof_mask, restrict
from tests.exact import plate_modes
from tests.helpers import ORTHO, PAG, PLATES, H

EPS = np.finfo(float).eps
ISO = {0: Material(isotropic(1.0, 0.3), 1.0)}
R, T, L = 1.0, 0.02, 2.0  # cylinder radius, wall, full length
ANGLE, TS = np.radians(30), 1 / 20  # skew angle, thickness
PZ = 1.42  # thickness rate: 2 lambda_s of the clamped-free corner (plane strain, nu = 0.3)

SRAO = {
    ("pag", 10): (32, 12),
    ("ortho", 10): (32, 24),
    ("pag", 100): (48, 3),
    ("ortho", 100): (48, 4),
}
LABELS = {  # lowest 10 of the full plate (m, n, family)
    ("pag", 10): [
        (1, 1, 0),
        (1, 2, 0),
        (1, 0, 0),
        (0, 1, 0),
        (2, 1, 0),
        (1, 3, 0),
        (2, 2, 0),
        (2, 3, 0),
        (0, 2, 0),
        (2, 0, 0),
    ],
    ("ortho", 10): [
        (1, 1, 0),
        (1, 2, 0),
        (2, 1, 0),
        (1, 0, 0),
        (0, 1, 0),
        (2, 2, 0),
        (1, 3, 0),
        (2, 3, 0),
        (3, 1, 0),
        (1, 1, 1),
    ],
    ("pag", 100): [
        (1, 1, 0),
        (1, 2, 0),
        (1, 3, 0),
        (2, 1, 0),
        (2, 2, 0),
        (1, 4, 0),
        (2, 3, 0),
        (2, 4, 0),
        (1, 5, 0),
        (2, 5, 0),
    ],
    ("ortho", 100): [
        (1, 1, 0),
        (1, 2, 0),
        (2, 1, 0),
        (2, 2, 0),
        (1, 3, 0),
        (2, 3, 0),
        (3, 1, 0),
        (1, 4, 0),
        (3, 2, 0),
        (2, 4, 0),
    ],
}
CLASSES = ("SS", "SA", "AS", "AA")  # circumferential (x = 0 plane), axial (mid-length)
CYL = {
    "levels": [(12, 6, 3), (24, 12, 3), (48, 24, 3), (96, 48, 3)],
    "coarse": [(24, 12, 3), (24, 12, 4), (24, 12, 5)],
    "mid": [(48, 24, 3), (48, 24, 4)],
    "shell": (512, 96, 1),
}
SKEW = {
    "levels": [(6, 3), (12, 3), (24, 3), (48, 3)],
    "coarse": [(12, 3), (12, 4), (12, 5)],
    "mid": [(24, 3), (24, 4)],
    "shell": (192, 1),
}


def _plies(name, Lt):
    """PLATES[name] scaled to thickness 1 / Lt: [(C, rho, t)]."""
    return [(C, rho, t / (H * Lt)) for C, rho, t in PLATES[name]]


def _shell_mats(name):
    """One Material per ply; pag's middle ply as a 90 deg ply angle (PLATES rotates C instead)."""
    if name == "ortho":
        return {0: Material(ORTHO, 0.7)}
    pag = Material(PAG, 1.0)
    return {0: pag, 1: Laminate((Ply(pag, np.pi / 2, 1.0),)), 2: pag}


def _laminate(name):
    """The plies of _shell_mats as one Laminate (equal fractions)."""
    sm = _shell_mats(name).values()
    return Laminate(
        tuple(
            m.plies[0]._replace(fraction=1 / len(sm))
            if isinstance(m, Laminate)
            else Ply(m, 0.0, 1 / len(sm))
            for m in sm
        )
    )


def srao_case(name, Lt, label, q, k, width=None):
    """Quarter of the half-wave sub-plate of label on [0, 1]^2 x [0, 1/Lt]: q x q `ssh8` (one
    element across, width `width` or one element, along a zero index), k elements per ply, or for
    k = 0 one element through the thickness with the whole Laminate. SS on xmin/ymin, a mirror on
    the far faces (SS on a zero-index cut)."""
    m_, n_, _ = label
    pl = _plies(name, Lt)
    a, b = 1 / max(m_, 1), 1 / max(n_, 1)
    hx = a / 2 / q if m_ else (width or b / 2 / q)
    hy = b / 2 / q if n_ else (width or a / 2 / q)
    nz = k * len(pl) or 1
    mesh = box_mesh(((q if m_ else 1), (q if n_ else 1), nz), spacing=(hx, hy, 1 / Lt / nz))
    if k:
        zc = mesh.points[mesh.blocks["hex8"]].mean(1)[:, 2]
        mats = _shell_mats(name)
        material = np.searchsorted(np.cumsum([p[2] for p in pl])[:-1], zc)
    else:
        mats, material = {0: _laminate(name)}, np.zeros(mesh.n_el, int)
    mesh = dataclasses.replace(mesh, voxel=None, material=material)
    mesh, _ = solid_shell(mesh, direction=(0.0, 0.0, 1.0))
    bcs = {"xmin": "yz", "ymin": "xz", "xmax": "x" if m_ else "yz", "ymax": "y" if n_ else "xz"}
    return mesh, mats, bcs


def cylinder(typ, cls, nt, na, nz, half=False):
    """Clamped cylinder R/t = 50, L/R = 2. 1/8 model: theta in [0, pi/2] (box x), the clamped end
    to mid-length (box y), class cls = (x = 0 plane, mid-length) each S or A; y = 0 is symmetric.
    half: theta in [0, pi], full length, both ends clamped, cls "S" or "A" about y = 0. hex27 is
    cosine-graded toward the clamped ends; `ssh8` is uniform (typ "ssh8" from a hex8 box)."""
    j = np.arange(na + 1) / na
    if typ != "hex27":
        ya = j
    else:
        ya = (1 - np.cos(np.pi * j)) / 2 if half else 1 - np.cos(np.pi * j / 2)
    m = box_mesh(
        (nt, na, nz),
        type="hex8" if typ == "ssh8" else typ,
        coords=(np.linspace(0, 1, nt + 1), ya, np.linspace(0, 1, nz + 1)),
    )
    p = m.points
    th, s = p[:, 0] * np.pi / (1 if half else 2), p[:, 1] * L / (1 if half else 2)
    r = R - T / 2 + p[:, 2] * T
    m = dataclasses.replace(m, points=np.c_[r * np.cos(th), r * np.sin(th), s], voxel=None)
    if typ == "ssh8":
        m, _ = solid_shell(m)
    if half:
        face = "y" if cls == "S" else "xz"
        return m, ISO, {"ymin": "xyz", "ymax": "xyz", "xmin": face, "xmax": face}
    bcs = {
        "ymin": "xyz",
        "xmin": "y",
        "xmax": "x" if cls[0] == "S" else "yz",
        "ymax": "z" if cls[1] == "S" else "xy",
    }
    return m, ISO, bcs


def skew(typ, n, nz):
    """Clamped 30 deg rhombus, side 1, L/t = 20: n x n x nz, hex27 cosine-graded toward the edges,
    `ssh8` uniform."""
    c = (
        (1 - np.cos(np.pi * np.arange(n + 1) / n)) / 2
        if typ == "hex27"
        else np.linspace(0, 1, n + 1)
    )
    m = box_mesh(
        (n, n, nz), type="hex8" if typ == "ssh8" else typ, coords=(c, c, np.linspace(0, TS, nz + 1))
    )
    p = m.points
    pts = np.c_[p[:, 0] + p[:, 1] * np.sin(ANGLE), p[:, 1] * np.cos(ANGLE), p[:, 2]]
    m = dataclasses.replace(m, points=pts, voxel=None)
    if typ == "ssh8":
        m, _ = solid_shell(m)
    return m, ISO, {f: "xyz" for f in ("xmin", "xmax", "ymin", "ymax")}


BUILDERS = {"srao": srao_case, "cylinder": cylinder, "skew": skew}
PARTS = {"srao": "srao", "cyl": "cylinder", "skew": "skew"}


def keys(part):
    """(builder, args) of a part, each unique key once."""
    if part == "srao":
        out = [
            (name, Lt, lab, q, k) for (name, Lt), (q, k) in SRAO.items() for lab in LABELS[name, Lt]
        ]
        out += [
            (name, Lt, lab, q, 0) for (name, Lt), (q, _) in SRAO.items() for lab in LABELS[name, Lt]
        ]
    elif part == "cyl":
        s = dict.fromkeys(CYL["levels"] + CYL["coarse"] + CYL["mid"])
        out = [("hex27", c, *x) for c in CLASSES for x in s] + [
            ("ssh8", c, *CYL["shell"]) for c in CLASSES
        ]
    else:
        s = dict.fromkeys(SKEW["levels"] + SKEW["coarse"] + SKEW["mid"])
        out = [("hex27", *x) for x in s] + [("ssh8", *SKEW["shell"])]
    return [(PARTS[part], a) for a in out]


def _key(x):
    return tuple(_key(y) for y in x) if isinstance(x, list | tuple) else x


@functools.cache
def target(name, Lt, label):
    """(exact lam on the full plate, rank in the quarter's class from 12 sub-plate modes)."""
    pl = _plies(name, Lt)
    lam, lab, _ = plate_modes(pl, 1.0, 1.0, 10)
    ex = lam[[tuple(x) for x in lab].index(label)]
    m_, n_, f_ = label
    sl, sll, _ = plate_modes(pl, 1 / max(m_, 1), 1 / max(n_, 1), 12, mmax=12)
    cls = [
        i
        for i, (m, n, _) in enumerate(sll)
        if (m % 2 if m_ else m == 0) and (n % 2 if n_ else n == 0)
    ]
    rank = [tuple(sll[i]) for i in cls].index((1 if m_ else 0, 1 if n_ else 0, f_))
    assert np.isclose(sl[cls[rank]], ex, rtol=1e-9), (label, sl[cls[rank]], ex)
    return float(ex), rank


def filter_strip(mesh, label, V):
    """Keep mask of a zero-index strip's modes: the free in-plane component (u_y on (m, 0), u_x on
    (0, n)) equal on the two cut faces (s <= 1e-6) is kept, opposite (s >= 1e6) dropped."""
    ax = 1 if label[0] else 0
    P = mesh.points
    lo, hi = (np.flatnonzero(np.isclose(P[:, ax], f(P[:, ax]))) for f in (np.min, np.max))
    lo, hi = (i[np.lexsort(np.delete(P[i], ax, 1).T)] for i in (lo, hi))
    U = V.reshape(-1, 3, V.shape[1])[:, ax]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.linalg.norm(U[lo] - U[hi], axis=0) / np.linalg.norm(U[lo] + U[hi], axis=0)
    if not ((s <= 1e-6) | (s >= 1e6)).all():
        raise ValueError(f"strip mode neither symmetric nor antisymmetric: s = {s}")
    return s <= 1e-6


def skew_parity(mesh, V):
    """+-1 per mode under the rhombus mirror across its long diagonal (box x <-> y)."""
    P = mesh.points
    y = P[:, 1] / np.cos(ANGLE)
    x = P[:, 0] - y * np.sin(ANGLE)
    A, B = (np.rint(np.c_[u, v, P[:, 2]] * 1e9).astype(np.int64) for u, v in ((x, y), (y, x)))
    ia, ib = np.lexsort(A.T[::-1]), np.lexsort(B.T[::-1])
    assert (A[ia] == B[ib]).all(), "mesh not mirror symmetric"
    perm = np.empty(len(P), int)
    perm[ia] = ib
    d = np.array([1 + np.sin(ANGLE), np.cos(ANGLE)]) / np.hypot(1 + np.sin(ANGLE), np.cos(ANGLE))
    Q = np.eye(3)
    Q[:2, :2] = 2 * np.outer(d, d) - np.eye(2)
    U = V.reshape(-1, 3, V.shape[1])
    s = np.einsum("nik,nik->k", U, np.einsum("ij,njk->nik", Q, U[perm])) / (U**2).sum((0, 1))
    if not np.allclose(np.abs(s), 1, atol=1e-6):
        raise ValueError(f"skew mode without mirror parity: {s}")
    return np.sign(s).astype(int)


def digest(mesh, mats, free):
    """Builder fingerprint: SHA-256 of the integer data (block types, connectivity, material and
    orientation ids, free mask) and float moments of the points plus the material constants,
    compared to 1e-12 (`same`), so ulp differences across platforms do not trip it."""
    h = hashlib.sha256()
    for t, conn in mesh.blocks.items():
        h.update(t.encode())
        h.update(np.ascontiguousarray(conn, np.int64).tobytes())
    for a in (mesh.material, mesh.orientation, free):
        h.update(np.ascontiguousarray(a, np.int64).tobytes())
    P = mesh.points
    w = np.arange(len(P)) % 97 + 1.0
    floats = [*P.sum(0), *(w @ P), *(P**2).sum(0)]
    for i in sorted(mats):
        mat = mats[i]
        for ply in mat.plies if isinstance(mat, Laminate) else (Ply(mat, 0.0, 1.0),):
            floats += [
                *np.ravel(ply.material.C),
                float(ply.material.rho),
                float(ply.angle),
                float(ply.fraction),
            ]
    return {"sha": h.hexdigest(), "floats": [float(x) for x in floats]}


def same(a, b):
    return a["sha"] == b["sha"] and np.allclose(a["floats"], b["floats"], rtol=1e-12, atol=1e-14)


def n_modes(builder, args):
    """Modes per run: 6, 7 on hex27 cylinders (an upper neighbour for the crossing guard), 10 / 8 on
    the skew (both mirror classes), 6 + rank + 1 on a zero-index strip (filtered)."""
    if builder == "srao":
        label = args[2]
        return 6 if label[0] and label[1] else 6 + target(*args[:3])[1] + 1
    if builder == "skew":
        return 10 if args[0] == "hex27" else 8
    return 7 if args[0] == "hex27" else 6


def run(builder, args, lmax=True, n=None, **kw):
    """Solve one key: a record entry (lam ascending, filtered on strips; lmax by eigsh LA). n
    overrides the mode count, kw go to the builder (tests only)."""
    args = _key(args)
    t0 = time.perf_counter()
    mesh, mats, bcs = BUILDERS[builder](*args, **kw)
    K, M = assemble(mesh, mats)
    free = dof_mask(mesh, bcs)
    lam, V = reference_modes(K, M, n or n_modes(builder, args), free, points=mesh.points)
    rec = {
        "builder": builder,
        "args": args,
        "dof": int(free.sum()),
        "digest": digest(mesh, mats, free),
    }
    if builder == "srao" and not (args[2][0] and args[2][1]):
        keep = filter_strip(mesh, args[2], V)
        rank = target(*args[:3])[1]
        assert keep.sum() > rank, (args, keep)
        lam, rec["removed"] = lam[keep], int((~keep).sum())
    if builder == "skew":
        rec["parity"] = skew_parity(mesh, V).tolist()
    if lmax:
        Kf, Mf, _ = restrict(K, M, free)
        rec["lmax"] = float(sla.eigsh(Kf, 1, Mf, which="LA", tol=1e-3)[0][0])
    del K, M, V
    rss = harness.maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform)
    rec |= {"lam": lam.tolist(), "time": time.perf_counter() - t0, "rss_gb": rss / 1e9}
    print(
        f"{builder} {args} dof {rec['dof']} {rec['time']:.1f} s rss {rec['rss_gb']:.2f} GB",
        flush=True,
    )
    return rec


# ---- analysis ----


def richardson(F):
    """F (3, g) f on three nested levels (ratio 2), coarse first: relative estimate of the finest
    error 1.25 |D2| / (2^min(p, 4) - 1), p, and whether the data are monotone with p >= 1."""
    d1, d2 = F[0] - F[1], F[1] - F[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.log2(d1 / d2)
    ok = (d1 * d2 > 0) & (p >= 1)
    return 1.25 * np.abs(d2) / (2 ** np.minimum(np.where(ok, p, 1), 4) - 1) / F[2], p, ok


def r_z(p, nz=3):
    """D1/D2 of runs at nz, nz + 1, nz + 2 for an error ~ nz^-p (increasing in p)."""
    return (nz**-p - (nz + 1) ** -p) / ((nz + 1) ** -p - (nz + 2) ** -p)


def p_z(ratio, nz=3):
    """Observed thickness rate (reported only); nan outside [0.05, 20]."""
    g = lambda p: r_z(p, nz) - ratio  # noqa: E731
    return brentq(g, 0.05, 20) if g(0.05) * g(20) < 0 else np.nan


def nz_term(c, m, nz=3):
    """Thickness term from the coarse (c: (3, g) at nz, nz + 1, nz + 2) and mid (m: (2, g))
    levels at the fixed rate PZ, extrapolated in h at the same rate; plus the coarse ratio D1/D2
    and whether the coarse runs are monotone."""
    q = 1 - (nz / (nz + 1)) ** PZ
    tc, tm = np.abs(c[0] - c[1]) / c[0] / q, np.abs(m[0] - m[1]) / m[0] / q
    d1, d2 = c[0] - c[1], c[1] - c[2]
    return 1.25 * (tm + np.abs(tm - tc) / (2**PZ - 1)), d1 / d2, d1 * d2 > 0


def _select(r, n=6):
    """(class, rank in class) of the lowest n modes of a run (class = skew parity, else one)."""
    c = r.get("parity", [1] * len(r["lam"]))
    return [(c[i], c[:i].count(c[i])) for i in range(n)]


def _f(r, sel):
    """f of the modes sel = [(class, rank)] of run r."""
    lam, c = np.array(r["lam"]), np.array(r.get("parity", [1] * len(r["lam"])))
    return np.sqrt([lam[np.flatnonzero(c == k)[j]] for k, j in sel])


def crossing(F_all, sel, ext):
    """Monotone crossing guard. F_all (levels, len(ext)) f of the modes ext (the gated modes sel and
    their upper neighbours, by class). Returns (decreasing, ratios (levels - 1, pairs), pair
    members): ratio = upper mode's move / its class gap below at the coarser level."""
    pairs = [(ext.index((k, j)), ext.index((k, j + 1))) for k, j in ext if (k, j + 1) in ext]
    dec = bool((F_all[:-1] >= F_all[1:]).all())
    rat = np.array(
        [
            [abs(F_all[i, b] - F_all[i + 1, b]) / (F_all[i, b] - F_all[i, a]) for a, b in pairs]
            for i in range(len(F_all) - 1)
        ]
    )
    return dec, rat, [(ext[a], ext[b]) for a, b in pairs]


def reference_case(runs, builder, pre, S):
    """Analysis of one hex27 reference and its `ssh8` run; keys (builder, (type, *pre, *spec)) for
    the specs in S."""
    key = lambda t, x: (builder, (t, *pre, *x))  # noqa: E731
    levels, coarse, mid = ([key("hex27", x) for x in S[n]] for n in ("levels", "coarse", "mid"))
    shell = key("ssh8", S["shell"])
    sel = _select(runs[levels[-1]])
    ext = sorted(set(sel) | {(k, j + 1) for k, j in sel})
    F_all = np.array([_f(runs[k], ext) for k in levels])
    dec, rat, members = crossing(F_all, sel, ext)
    F = np.array([_f(runs[k], sel) for k in levels])
    est, p, ok = richardson(F[1:])
    estb, pb, okb = richardson(F[:3])
    crossed = {m for (lo, up), x in zip(members, rat[0], strict=True) if x >= 1 for m in (lo, up)}
    checked = np.array([s not in crossed for s in sel])
    with np.errstate(invalid="ignore", divide="ignore"):
        back = estb / (np.abs(F[2] / F[3] - 1) + est)
    t, ratio, zmono = nz_term(
        np.array([_f(runs[k], sel) for k in coarse]), np.array([_f(runs[k], sel) for k in mid])
    )
    fin = runs[levels[-1]]
    flam = 10 * EPS * fin["lmax"] / F[3] ** 2
    ref = est + t + flam
    rs = runs[shell]
    fs = _f(rs, sel)
    return {
        "f": F[3],
        "p": p,
        "monotone": ok,
        "p_back": pb,
        "back_ok": okb,
        "back": back,
        "checked": checked,
        "decreasing": dec,
        "guard": rat,
        "richardson": est,
        "nz": t,
        "ratio_z": ratio,
        "nz_monotone": zmono,
        "p_z": np.array([p_z(x) for x in ratio]),
        "flam": flam,
        "ref": ref,
        "err": np.abs(fs / F[3] - 1),
        "flam_shell": 10 * EPS * rs["lmax"] / fs**2,
    }


def analyze(rec):
    """Numbers per case from a record (list of run entries); the tests hold the gates."""
    runs = {(r["builder"], _key(r["args"])): r for r in rec}
    out = {}
    for (name, Lt), (q, k) in SRAO.items():
        if ("srao", (name, Lt, LABELS[name, Lt][0], q, k)) not in runs:
            continue
        rows = {"err": [], "flam": [], "removed": [], "layered": []}
        for lab in LABELS[name, Lt]:
            ex, rank = target(name, Lt, lab)
            for kk, col in ((k, "err"), (0, "layered")):
                r = runs.get(("srao", (name, Lt, lab, q, kk)))
                if r is None:
                    continue
                lam = r["lam"][rank]
                rows[col].append(np.sqrt(lam / ex) - 1)
                if kk:
                    rows["flam"].append(10 * EPS * r["lmax"] / lam)
                    rows["removed"].append(r.get("removed", 0))
        out[f"srao {name} {Lt}"] = {c: np.array(v) for c, v in rows.items()}
    cases = [(f"cyl {c}", "cylinder", (c,), CYL) for c in CLASSES] + [("skew", "skew", (), SKEW)]
    for name, builder, pre, S in cases:
        if all((builder, ("hex27", *pre, *x)) in runs for x in S["levels"]):
            out[name] = reference_case(runs, builder, pre, S)
    return out


def load():
    path = harness.RESULTS / "shell_accuracy.json"
    return json.loads(path.read_text()) if path.exists() else {"parts": {}, "runs": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    a = ap.parse_args()
    data = load()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    for part in a.parts:
        t0 = time.perf_counter()
        new = [run(b, args) for b, args in keys(part)]
        data["runs"] = [r for r in data["runs"] if r["builder"] != PARTS[part]] + new
        data["parts"][part] = {
            "commit": commit,
            "date": time.strftime("%Y-%m-%d"),
            "minutes": (time.perf_counter() - t0) / 60,
        }
        harness.write_result("shell_accuracy", data)
    np.set_printoptions(precision=3, linewidth=150)
    for name, res in analyze(data["runs"]).items():
        print(name)
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
