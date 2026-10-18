"""PR 18a: AMG on solid-shell meshes (ssh8, ssh8 + hex8): fibers, candidates, the Schwarz
smoothers and the setup against NumPy references, PCG counts against L/t, refinement, junctions,
stacks, mixed meshes and constraints, reuse, and the unchanged hex path."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from exact import SIMPLY_SUPPORTED
from helpers import STEEL, cheb, dense, lam1, rhs, vcycle
from pyamg.aggregation import standard_aggregation
from test_post import disjoint
from test_shell import C_ORT, laminate, three_plies

from jaxmodal.linalg import pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import Mesh, box_mesh, solid_shell
from jaxmodal.operators import ap_operator, assemble, dof_mask, restrict, unstructured
from jaxmodal.precond import GMG, amg, jacobi
from jaxmodal.precond import schwarz as SW
from jaxmodal.precond import shell as SH
from jaxmodal.precond.pmg import _hex8_galerkin

AM = importlib.import_module("jaxmodal.precond.amg")
E, NU, RHO = 210e9, 0.3, 7850.0
CLAMPED = {k: "xyz" for k in ("xmin", "xmax", "ymin", "ymax")}

# Measured PCG counts (max over 4 columns, tol 1e-6); each gate is 1.5x (docs/preconditioners.md).
COUNTS = {
    "flat128-10": 12, "flat128-1000": 21, "curved128-10": 11, "curved128-1000": 30,
    "flat64-100": 14, "flat128-100": 13, "flat256-1000": 19,
    "flat32-50": 13, "plies32-50": 12, "curved32-50": 14, "cylinder": 46, "T1": 13, "L1": 11,
    "T3": 18, "flat32-300": 17, "web32": 19, "block16": 14, "clamped32": 12, "clamped32-0": 12,
    "bottom32": 12, "parts": 14, "free24": 14, "stack32-10": 11, "stack32-100": 16,
}  # fmt: skip


def gate(name, its):
    assert its <= 1.5 * COUNTS[name], (name, its)


# ---------------------------------------------------------------- meshes


def kirchhoff(t, c=19.74):
    """lambda_1 of a unit square Kirchhoff plate: SS (c = 2 pi^2 = 19.74) or clamped
    (35.99), as D c^2 / (rho t)."""
    D = E * t**3 / (12 * (1 - NU**2))
    return D * c**2 / (RHO * t)


def plate(n, Lt, curved=False, layers=1):
    """n x n x layers ssh8 plate with L = 1 and t = 1 / Lt; curved maps it onto a 90 degree
    panel of radius 1 (straight side L, arc 1.57). Node sets of box_mesh."""
    t = 1 / Lt
    box = box_mesh((n, n, layers), spacing=(1 / n, 1 / n, t / layers))
    if not curved:
        return solid_shell(box, direction=(0, 0, 1))[0], t
    x, y, z = box.points.T
    th, r = y * np.pi / 2, 1 + z
    box.points[:] = np.c_[x, r * np.sin(th), r * np.cos(th)]
    d = box.points[box.blocks["hex8"]].mean(1) * [0, 1, 1]
    return solid_shell(box, direction=d / np.linalg.norm(d, axis=1, keepdims=True))[0], t


def sets(x, t):
    """Node sets from coordinates: the unit plate's edges (x/y min/max), its bottom (zmin) and
    the bottom edges (bottom)."""
    tol = 1e-6 * t
    s = {
        "xmin": np.abs(x[:, 0]) < tol, "xmax": np.abs(x[:, 0] - 1) < tol,
        "ymin": np.abs(x[:, 1]) < tol, "ymax": np.abs(x[:, 1] - 1) < tol,
        "zmin": np.abs(x[:, 2]) < tol,
    }  # fmt: skip
    s["bottom"] = s["zmin"] & (s["xmin"] | s["xmax"] | s["ymin"] | s["ymax"])
    return {k: np.flatnonzero(v) for k, v in s.items()}


def glue(parts, t):
    """One mesh from (hex8 box, direction or None) parts, coincident nodes merged by the key
    round(x / (1e-3 t)); parts with a direction become ssh8."""
    pts = np.vstack([p.points for p, _ in parts])
    _, first, inv = np.unique(
        np.round(pts / (1e-3 * t)), axis=0, return_index=True, return_inverse=True
    )
    inv, off, conn, dirs = inv.ravel(), 0, [], []
    for p, d in parts:
        conn.append(inv[off + p.blocks["hex8"]])
        dirs.append(np.tile(d if d is not None else [np.nan] * 3, (p.n_el, 1)))
        off += len(p.points)
    conn, dirs = np.vstack(conn), np.vstack(dirs)
    x = pts[first]
    n = len(conn)
    mesh = Mesh(x, {"hex8": conn}, sets(x, t), {}, np.zeros(n, np.int64), np.zeros(n, np.int64))
    sel = np.flatnonzero(~np.isnan(dirs[:, 0]))
    return solid_shell(mesh, sel, dirs[sel])[0]


def grid(n, lo, hi, k):
    """linspace(0, 1, n + 1) with the points inside (lo, hi) replaced by k intervals."""
    g = np.linspace(0, 1, n + 1)
    return np.sort(np.r_[g[(g < lo - 1e-9) | (g > hi + 1e-9)], np.linspace(lo, hi, k + 1)])


def junction(kind, n, Lt, ks=1, kw=1):
    """An ssh8 skin (n x n, ks layers) with an ssh8 web (kw layers, height 1/2) standing on it:
    "T" at y = 1/2, "L" at the edge y = 1."""
    t = 1 / Lt
    lo = 0.5 - t / 2 if kind == "T" else 1 - t
    cy = grid(n, lo, lo + t, kw)
    cx = np.linspace(0, 1, n + 1)
    skin = box_mesh((n, len(cy) - 1, ks), coords=(cx, cy, np.linspace(0, t, ks + 1)))
    wy = np.linspace(lo, lo + t, kw + 1)
    web = box_mesh((n, kw, n // 2), coords=(cx, wy, np.linspace(t, t + 0.5, n // 2 + 1)))
    return glue([(skin, (0, 0, 1)), (web, (0, 1, 0))], t), t


def web(n, Lt):
    """An n x n ssh8 plate with a hex8 rib one cell wide and 3 cells deep on top (x across the
    plate, at the cell row y in [1/2, 1/2 + 1/n]): pure hex aggregates."""
    t, h = 1 / Lt, 1 / n
    c = np.linspace(0, 1, n + 1)
    skin = box_mesh((n, n, 1), coords=(c, c, np.array([0, t])))
    rib = box_mesh((n, 1, 3), coords=(c, np.array([0.5, 0.5 + h]), t + h * np.arange(4)))
    return glue([(skin, (0, 0, 1)), (rib, None)], t), t


def block(n, Lt=100):
    """n^3 hex8 unit cube with a one-element ssh8 skin of thickness 1/Lt on top."""
    t = 1 / Lt
    c = np.linspace(0, 1, n + 1)
    box = box_mesh((n, n, n + 1), coords=(c, c, np.r_[c, 1 + t]))
    top = np.arange(n**3, n**3 + n**2)
    return solid_shell(box, top, direction=(0, 0, 1))[0], t


def cylinder(consistent):
    """A 270 degree open cylinder (radius 1, length 1), 24 x 8 x 1 ssh8, thickness 0.02: by
    the topological sweep (orientations flip) or with radial directions."""
    box = box_mesh((24, 8, 1), spacing=(1 / 24, 1 / 8, 0.02))
    x, y, z = box.points.T
    th, r = x * 1.5 * np.pi, 1 + z
    box.points[:] = np.c_[r * np.cos(th), r * np.sin(th), y]
    d = None
    if consistent:
        d = box.points[box.blocks["hex8"]].mean(1) * [1, 1, 0]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    return solid_shell(box, direction=d)[0]


def setup(mesh, sigma, bcs, mats=STEEL, max_coarse=100, **kw):
    """(op, free, A_P, P) with P = amg on the mesh."""
    op, free = unstructured(mesh, mats), dof_mask(mesh, bcs)
    P = amg(op, free, sigma, mesh.points, max_coarse=max_coarse, **kw)
    return op, free, ap_operator(op, free, sigma), P


def count(mesh, sigma, bcs, mats=STEEL, max_coarse=100, levels=2, maxit=500, **kw):
    """Max PCG iterations to 1e-6 over 4 fixed-seed columns with amg; asserts convergence and
    at least `levels` levels."""
    op, free, A, P = setup(mesh, sigma, bcs, mats, max_coarse, **kw)
    assert len(P.levels) >= levels
    _, info = pcg(A, jnp.asarray(rhs(free, 4)), P, 1e-6, maxit)
    assert np.asarray(info["converged"]).all()
    return int(np.asarray(info["iterations"]).max())


def levels_of(mesh, sigma, bcs, max_coarse=100, mats=STEEL):
    """(op, free, A, Ab, levels, coarsest, shell, info) of the host shell setup."""
    op, free = unstructured(mesh, mats), dof_mask(mesh, bcs)
    A = ap_operator(op, free, sigma)
    Ab = _hex8_galerkin(A)[0]
    lv, Ac, shell, info = SH._shell_levels(op, Ab, A.free, mesh.points, max_coarse)
    return op, free, A, Ab, lv, Ac, shell, info


def ap_dense(mesh, free, sigma, A):
    """Dense A_P = m (K + sigma M) m + s (I - m) from assemble."""
    K, M = assemble(mesh, STEEL)
    m = free.astype(float)
    return (K + sigma * M).toarray() * np.outer(m, m) + np.diag(float(A.s) * (1 - m))


# ---------------------------------------------------------------- independent references


def union_find(mesh):
    """Fiber label per node from a Python union-find over the (a, a + 4) pairs of ssh8."""
    parent = list(range(len(mesh.points)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in mesh.blocks.get("ssh8", []):
        for a in range(4):
            parent[find(int(e[a]))] = find(int(e[a + 4]))
    return np.array([find(a) for a in range(len(parent))])


def same_partition(a, b):
    pairs = np.unique(np.c_[a, b], axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def column_patches(mesh):
    """Level-0 patches as a set of frozensets of nodes, built from the union-find."""
    lab = union_find(mesh)
    members = {}
    for a, g in enumerate(lab):
        members.setdefault(g, []).append(a)
    out = set()
    shell = mesh.blocks.get("ssh8", np.zeros((0, 8), int))
    for e in shell:
        out.add(frozenset(a for g in {lab[a] for a in e} for a in members[g]))
    for a in sorted(set(range(len(lab))) - set(shell.ravel())):
        out.add(frozenset([a]))
    return out


def patch_sets(blocks, sel, b):
    """Patches (blocks, sel) as a list of sorted DOF arrays."""
    out = []
    for bl, s in zip(blocks, sel, strict=True):
        d = (bl[:, None] * b + np.arange(b)).ravel()[s[: len(bl) * b]]
        out.append(np.sort(d[d >= 0]))
    return out


def dense_schwarz(A, patches):
    """sum_i R_i^T A_i^{-1} R_i for the dense A over DOF patches."""
    D = np.zeros_like(A)
    for p in patches:
        D[np.ix_(p, p)] += np.linalg.inv(A[np.ix_(p, p)])
    return D


def node_patch_dofs(patches):
    return [np.sort((np.array(sorted(p))[:, None] * 3 + np.arange(3)).ravel()) for p in patches]


def rings(S, zero, shell):
    """Coarse patches per group, by Python loops over the graph S."""
    S = sp.csr_array(S)
    out = []
    for g in range(S.shape[0]):
        d = list(range(9 * g, 9 * g + 9))
        if shell[g]:
            for j in S.indices[S.indptr[g] : S.indptr[g + 1]]:
                if shell[j] and j != g:
                    d += [9 * j + o for o in range(9) if not zero[9 * j + o]]
        out.append(np.sort(d))
    return out


# ---------------------------------------------------------------- 1: fibers and the guard


def test_fibers():
    """Test 1: fibers are pairs on one layer and columns through a stack; hex-only nodes are
    singletons; junction fibers merge and patches repeat no DOF; the stack has one patch per
    column of size 12 (k + 1); the size guard."""
    for mesh in (
        plate(4, 50)[0],
        plate(4, 50, layers=3)[0],
        web(8, 50)[0],
        junction("T", 8, 50)[0],
    ):
        op = unstructured(mesh, STEEL)
        g = SH._fibers(op)
        assert same_partition(g, union_find(mesh))
        cnt = np.bincount(g)
        hexonly = np.setdiff1d(np.arange(len(g)), mesh.blocks["ssh8"])
        assert (cnt[g[hexonly]] == 1).all()
        cols = SH._columns(op, g)
        assert all(len(set(r[r >= 0])) == (r >= 0).sum() for r in cols)  # no repeated node
        got = {frozenset(r[r >= 0].tolist()) for r in cols}
        assert len(got) == len(cols) and got == column_patches(mesh)
    assert set(np.bincount(SH._fibers(unstructured(plate(4, 50)[0], STEEL)))) == {2}
    op = unstructured(plate(4, 50, layers=3)[0], STEEL)
    cols = SH._columns(op, SH._fibers(op))
    assert len(cols) == 16 and ((cols >= 0).sum(1) == 16).all()  # 48 = 12 (k + 1) DOFs
    T = junction("T", 8, 50)[0]
    assert np.bincount(SH._fibers(unstructured(T, STEEL))).max() > 2  # the joint fibers merge


def test_guard():
    """Test 1: the size guard on a wrong axis and at 31 / 32 layers (also when nothing
    coarsens), the 5/5-layer T builds, and ssh8 with hex20 or hex27 raises."""
    box = box_mesh((40, 40, 1), spacing=(1 / 40, 1 / 40, 0.01))
    bad = solid_shell(box, direction=(1, 0, 0))[0]
    with pytest.raises(ValueError, match="ssh8 patch of 492 DOFs"):
        amg(unstructured(bad, STEEL), dof_mask(bad, {}), 1.0, bad.points)
    for k, ok in ((31, True), (32, False)):
        m = solid_shell(box_mesh((2, 2, k), spacing=(1, 1, 0.01)), direction=(0, 0, 1))[0]
        op, free = unstructured(m, STEEL), dof_mask(m, {"xmin": "xyz"})
        if ok:
            assert len(amg(op, free, 1.0, m.points).levels) == 0
        else:
            with pytest.raises(ValueError, match="ssh8 patch of 396 DOFs"):
                amg(op, free, 1.0, m.points)
    T = junction("T", 8, 20, ks=5, kw=5)[0]
    op = unstructured(T, STEEL)
    cols = SH._columns(op, SH._fibers(op))
    assert 3 * (cols >= 0).sum(1).max() == 3 * max(len(p) for p in column_patches(T)) == 252
    amg(op, dof_mask(T, SIMPLY_SUPPORTED), 1.0, T.points)
    mesh = disjoint()
    for t in ("hex20", "hex27"):
        m = Mesh(*_compact(mesh.points, {k: v for k, v in mesh.blocks.items() if k in (t, "ssh8")}))
        with pytest.raises(ValueError, match="supports ssh8 and hex8 blocks"):
            amg(unstructured(m, STEEL), dof_mask(m, {}), 1.0, m.points)


def _compact(points, blocks):
    """Mesh fields of these blocks with orphan nodes dropped."""
    used = np.unique(np.concatenate([c.ravel() for c in blocks.values()]))
    new = np.full(len(points), -1)
    new[used] = np.arange(len(used))
    z = np.zeros(sum(len(c) for c in blocks.values()), np.int64)
    return points[used], {k: new[c] for k, c in blocks.items()}, {}, {}, z, z


# ---------------------------------------------------------------- 2: candidates


def tensors(mesh):
    """(n, 3, 3) sums of v v^T over ssh8 thickness edges, by a loop over elements."""
    x, Tn = mesh.points, np.zeros((len(mesh.points), 3, 3))
    for e in mesh.blocks.get("ssh8", []):
        for a in range(4):
            v = x[e[a + 4]] - x[e[a]]
            Tn[e[a]] += np.outer(v, v)
            Tn[e[a + 4]] += np.outer(v, v)
    return Tn


def flat_fields(x, free, z0):
    """[rigid, k11, k22, k12] global fields on a flat plate (normal z, midplane z0), masked."""
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2] - z0
    F = np.zeros((len(x), 3, 3))
    F[:, 0, 0], F[:, 2, 0] = -Z * X, X**2 / 2
    F[:, 1, 1], F[:, 2, 1] = -Z * Y, Y**2 / 2
    F[:, 0, 2], F[:, 1, 2], F[:, 2, 2] = -Z * Y, -Z * X, X * Y
    return np.c_[AM._nullspace(x, free), F.reshape(-1, 3) * free[:, None]]


def span_angle(A, B):
    """Largest principal angle between the column spans of A and B (zero columns dropped);
    pi if the dimensions differ."""
    A, B = (sla.orth(M[:, np.abs(M).sum(0) > 0]) if M.size else M for M in (A, B))
    if A.shape[1] != B.shape[1]:
        return np.pi
    return 0.0 if A.shape[1] == 0 else float(sla.subspace_angles(A, B).max())


def fine(mesh, free, info, shell, lvl):
    """B^(lvl)_fine: rigid modes and the curvature fields of the level-(lvl + 1) aggregates."""
    na = info["nodes"][lvl]
    K, _ = SH._curvatures(mesh.points, tensors(mesh), na, shell[1][lvl][0].max() + 1, free)
    return np.c_[AM._nullspace(mesh.points, free), K]


def test_candidates_flat():
    """Test 2: on flat plates and the stack T_0 reproduces B^(1)_fine, its per-aggregate span is
    the global [rigid, kappa] span; on 16^2 the level-2 fields pass through T_0 T_1."""
    for mesh, t in (plate(16, 50), plate(8, 50, layers=3)):
        free = dof_mask(mesh, SIMPLY_SUPPORTED)
        *_, lv, _, shell, info = levels_of(mesh, kirchhoff(t), SIMPLY_SUPPORTED, max_coarse=20)
        assert len(lv) >= (3 if mesh.blocks["ssh8"].shape[0] == 256 else 2)
        T0 = shell[1][0][1].toarray()
        B1 = fine(mesh, free, info, shell, 1)
        assert np.linalg.norm(B1 - T0 @ (T0.T @ B1)) <= 1e-10 * np.linalg.norm(B1)
        G = flat_fields(mesh.points, free, t / 2)
        dofs = np.arange(3 * len(mesh.points)).reshape(-1, 3)
        na = info["nodes"][0]
        for a in range(na.max() + 1):
            r = dofs[na == a].ravel()
            assert span_angle(T0[r][:, 9 * a : 9 * a + 9], G[r]) <= 1e-10
        if len(lv) < 3:
            continue
        T1 = shell[1][1][1].toarray()
        B2 = fine(mesh, free, info, shell, 2)
        Q = T0 @ T1
        assert np.linalg.norm(B2 - Q @ (Q.T @ B2)) <= 1e-10 * np.linalg.norm(B2)


def test_candidates_cylinder():
    """Test 2: the sweep-built 270 degree cylinder (orientations flip): normals equal a
    per-aggregate eigh up to sign, no NaN, lie within each level-1 aggregate's angular span,
    and the curvature span equals the consistently oriented cylinder's."""
    meshes = [cylinder(False), cylinder(True)]
    assert (meshes[0].blocks["ssh8"] != meshes[1].blocks["ssh8"]).any()
    Ks = []
    for mesh in meshes:
        free = dof_mask(mesh, {})
        *_, lv, _, shell, info = levels_of(mesh, 1e4, {}, max_coarse=100)
        na = info["nodes"][0]
        n_agg = na.max() + 1
        Tn = tensors(mesh)
        nrm = SH._frames(mesh.points, Tn, na, n_agg)[0]
        x = mesh.points
        for a in range(n_agg):
            ref = np.linalg.eigh(Tn[na == a].sum(0))[1][:, -1]
            assert abs(abs(ref @ nrm[a]) - 1) <= 1e-12
            th = np.arctan2(x[na == a, 1], x[na == a, 0])
            span = np.ptp(np.unwrap(np.sort(th)))
            assert span < np.pi / 2
            c = x[na == a].mean(0) * [1, 1, 0]
            assert abs(nrm[a] @ c) / np.linalg.norm(c) >= np.cos(span) - 1e-12
        K = SH._curvatures(x, Tn, na, n_agg, free)[0]
        assert np.isfinite(K).all()
        Ks.append((na, K))
    (na, K0), (na1, K1) = Ks
    assert (na == na1).all()
    dofs = np.arange(3 * len(meshes[0].points)).reshape(-1, 3)
    for a in range(na.max() + 1):
        r = dofs[na == a].ravel()
        assert span_angle(K0[r], K1[r]) <= 1e-12


def test_candidates_mixed():
    """Test 2: on the hex8-web mesh B^(1) equals an independent per-aggregate construction
    (frame from sum v v^T, fields on every node, hex nodes included); pure-hex aggregates have
    zero curvature columns."""
    mesh, t = web(8, 50)
    free = dof_mask(mesh, SIMPLY_SUPPORTED)
    *_, lv, _, shell, info = levels_of(mesh, 1e4, SIMPLY_SUPPORTED, max_coarse=20)
    na, x, Tn = info["nodes"][0], mesh.points, tensors(mesh)
    K = SH._curvatures(x, Tn, na, na.max() + 1, free)[0]
    dofs = np.arange(3 * len(x)).reshape(-1, 3)
    hexonly = 0
    for a in range(na.max() + 1):
        nodes = np.flatnonzero(na == a)
        r = dofs[nodes].ravel()
        T = Tn[nodes].sum(0)
        if not T.any():
            assert not K[r].any()
            hexonly += 1
            continue
        n = np.linalg.eigh(T)[1][:, -1]
        e = np.eye(3)[np.argmin(np.abs(n))]
        t1 = np.cross(n, e) / np.linalg.norm(np.cross(n, e))
        t2 = np.cross(n, t1)
        d = x[nodes] - x[nodes].mean(0)
        X, Y, Z = d @ t1, d @ t2, d @ n
        F = [np.outer(-Z * X, t1) + np.outer(X**2 / 2, n),
             np.outer(-Z * Y, t2) + np.outer(Y**2 / 2, n),
             np.outer(-Z * Y, t1) + np.outer(-Z * X, t2) + np.outer(X * Y, n)]  # fmt: skip
        ref = np.stack([f.ravel() for f in F], 1) * free[r][:, None]
        assert span_angle(K[r], ref) <= 1e-12
    assert hexonly > 0


# ---------------------------------------------------------------- 3: the Schwarz smoother


def strip():
    """A 12 x 2 strip clamped on its long edges: the free fibers are collinear (y = 1/2), so
    their aggregates' k11 and k12 columns vanish exactly (identity DOFs in shell groups)."""
    box = box_mesh((12, 2, 1), spacing=(1 / 12, 1 / 2, 0.02))
    return solid_shell(box, direction=(0, 0, 1))[0]


SCHWARZ_MESHES = {
    "web": lambda: (web(8, 50)[0], SIMPLY_SUPPORTED),
    "stack": lambda: (plate(4, 50, layers=3)[0], SIMPLY_SUPPORTED),
    "T": lambda: (junction("T", 8, 50)[0], SIMPLY_SUPPORTED),
    "clamped": lambda: (plate(8, 50)[0], CLAMPED),
    "T55": lambda: (junction("T", 8, 20, ks=5, kw=5)[0], SIMPLY_SUPPORTED),
    "strip": lambda: (strip(), {"ymin": "xyz", "ymax": "xyz"}),
}


def check_schwarz(Asp, blocks, sel, ref, Ad, free, cb):
    """Schwarz.build(Asp) applied to masked r equals the dense sum over the reference DOF
    patches of the dense Ad; symmetric; exact 0 on constrained DOFs; nbytes and pad slots."""
    S = SW.build(Asp, blocks, sel, chunk_bytes=cb)
    n = Asp.shape[0]
    r = rhs(free, 3)
    D = dense_schwarz(Ad, ref)
    y = np.asarray(S(jnp.asarray(r)))
    assert np.abs(y - D @ r).max() <= 1e-10 * np.abs(D @ r).max()
    assert (y[~free] == 0).all()
    Sd = dense(S, n)
    assert np.abs(Sd - Sd.T).max() <= 1e-12 * np.abs(Sd).max()
    size = sel.sum(1)
    assert SW.nbytes(size, "float64", cb) == sum(a.nbytes for a in S.idx + S.inv)
    for (s, nc, ch), idx in zip(SW.layout(size, cb), S.idx, strict=True):
        assert idx.shape == (nc, ch, s) and nc * ch - (np.asarray(idx)[..., 0] < n).sum() < nc
    return S


@pytest.mark.parametrize("name", SCHWARZ_MESHES)
def test_schwarz(name):
    """Test 3: the level-0 Schwarz (small chunk_bytes: multi-chunk, padded classes) against
    the dense sum over the union-find patches of the assembled A_P; on the strip also level 1
    against its SciPy A_1 and the NumPy rings, with an excluded neighbour identity DOF
    present. Tolerance 1e-10: the stack's patches are ill-conditioned."""
    mesh, bcs = SCHWARZ_MESHES[name]()
    sigma = 1e4
    op, free, A, Ab, lv, Ac, shell, info = levels_of(mesh, sigma, bcs, max_coarse=20)
    Ad = ap_dense(mesh, free, sigma, A)
    if lv:
        blocks, sel = lv[0][1]
    else:
        blocks = SH._columns(op, SH._fibers(op))
        sel = np.repeat(blocks >= 0, 3, axis=1)
    ref = node_patch_dofs(column_patches(mesh))
    cb = max(2**14, 8 * int(sel.sum(1).max()) ** 2)
    check_schwarz(Ab, blocks, sel, ref, Ad, free, cb)
    if name == "T55":
        assert sel.sum(1).max() == 252
    if name != "strip":
        return
    zero, flags = shell[1][0][2], shell[1][0][3]
    S1 = info["graphs"][1]
    excluded = any(
        flags[g] and flags[j] and zero[9 * j : 9 * j + 9].any()
        for g in range(S1.shape[0])
        for j in S1.indices[S1.indptr[g] : S1.indptr[g + 1]]
    )
    assert excluded
    A1 = lv[1][0]
    b1, s1 = lv[1][1]
    cb = max(2**14, 8 * int(s1.sum(1).max()) ** 2)
    check_schwarz(A1, b1, s1, rings(S1, zero, flags), A1.toarray(), np.ones(A1.shape[0], bool), cb)


def test_schwarz_large_keys():
    """Test 3: block keys past int32 (nb > 46340 blocks, int32 patch indices): patches at the
    end of a block-tridiagonal 9 x 9 BSR matrix against dense inverses."""
    nb, b = 50_000, 9
    rng = np.random.default_rng(0)
    off = sp.kron(sp.eye(nb, k=1) + sp.eye(nb, k=-1), -np.eye(b))
    A = sp.bsr_array(sp.kron(sp.eye(nb), 4 * np.eye(b)) + off, blocksize=(b, b))
    A.data[:] *= 1 + 0.1 * rng.random((len(A.data), 1, 1))
    A = sp.bsr_array((A + A.T) / 2, blocksize=(b, b))
    A.sort_indices()
    blocks = np.array([[nb - 2, nb - 1], [0, nb - 1], [nb - 1, -1]], np.int32)
    sel = np.repeat(blocks >= 0, b, axis=1)
    r = rng.standard_normal((nb * b, 1))
    y = np.asarray(SW.build(A, blocks, sel)(jnp.asarray(r)))
    ref, Ac = np.zeros_like(r), A.tocsr()
    for row in blocks:
        d = (b * row[row >= 0][:, None] + np.arange(b)).ravel()
        ref[d] += np.linalg.solve(Ac[d][:, d].toarray(), r[d])
    assert np.abs(y - ref).max() <= 1e-12 * np.abs(ref).max()


def test_schwarz_layout():
    """Test 3: size classes on hand-computed cases (one class for uniform 24 and for a
    plate's coarse 27/45/63/81 mix, two for 24 and 3, the cap at 8), the bound and a
    non-finite inverse raise."""
    assert [c[0] for c in SW.layout([24] * 1000)] == [24]
    assert [c[0] for c in SW.layout([81] * 950 + [63] * 30 + [45] * 15 + [27] * 5)] == [81]
    assert [c[0] for c in SW.layout([24] * 1000 + [3] * 300)] == [24, 3]
    assert len(SW.layout(np.repeat(np.arange(1, 13) * 30, 10))) <= 8
    s = 2 ** np.arange(3, 13)  # equal bytes per size: every merge adds > 10 %
    assert len(SW.layout(np.repeat(s, (4096 // s) ** 2))) == 8
    c = SW.layout([24] * 100, chunk_bytes=8 * 24 * 24 * 7)
    assert c == [(24, 15, 7)]
    mesh = plate(4, 50)[0]
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, SIMPLY_SUPPORTED)
    Ab = _hex8_galerkin(ap_operator(op, free, 1e4))[0]
    blocks = SH._columns(op, SH._fibers(op))
    sel = np.repeat(blocks >= 0, 3, axis=1)
    with pytest.raises(ValueError, match="exceeds 23"):
        SW.build(Ab, blocks, sel, chunk_bytes=8 * 24 * 24 - 1)
    Ab.data[:] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        SW.build(Ab, blocks, sel)


# ---------------------------------------------------------------- 4: groups and patches


def group_graph(mesh, group, free):
    """The free-group graph by loops over elements, vertices numbered as the free groups in
    group-label order."""
    fg = np.zeros(group.max() + 1, bool)
    fg[group[free.reshape(-1, 3).any(1)]] = True
    num = np.cumsum(fg) - 1
    edges = set()
    for c in mesh.blocks.values():
        for e in c:
            gs = {num[group[a]] for a in e if fg[group[a]]}
            edges |= {(a, b) for a in gs for b in gs if a != b}
    n = int(fg.sum())
    r, c = zip(*sorted(edges), strict=True) if edges else ((), ())
    return sp.csr_array((np.ones(len(r)), (r, c)), shape=(n, n))


def coarse_graph(S, agg):
    """Aggregates a != b adjacent when some of their groups are."""
    S = sp.coo_array(S)
    n = agg.max() + 1
    edges = {(agg[i], agg[j]) for i, j in zip(S.row, S.col, strict=True) if agg[i] != agg[j]}
    r, c = zip(*sorted(edges), strict=True) if edges else ((), ())
    return sp.csr_array((np.ones(len(r)), (r, c)), shape=(n, n))


def same_graph(A, B):
    return A.shape == B.shape and ((sp.csr_array(A) != 0) != (sp.csr_array(B) != 0)).nnz == 0


@pytest.mark.parametrize("name", ["web", "clamped"])
def test_groups(name):
    """Test 4: the level-0 graph, the aggregation (pyamg plus singletons), the one-ring
    level-1 graph and the level-1 patches against NumPy loops; identity DOFs only in their own
    group's patch."""
    edge = sp.csr_array((np.ones(2), ([0, 1], [1, 0])), shape=(3, 3))
    assert (SH._aggregate(SH._graph(sp.csr_array((5, 5)))) == np.arange(5)).all()  # no edges
    assert (SH._aggregate(SH._graph(edge)) == [0, 0, 1]).all()  # one isolated vertex
    mesh, bcs = SCHWARZ_MESHES[name]()
    op, free, A, Ab, lv, Ac, shell, info = levels_of(mesh, 1e4, bcs, max_coarse=20)
    group = shell[0]
    S0 = group_graph(mesh, group, free)
    assert same_graph(S0, info["graphs"][0])
    agg0 = shell[1][0][0]
    lab = sp.csr_array(standard_aggregation(SH._graph(S0))[0])
    ref = np.full(S0.shape[0], -1)
    ref[np.flatnonzero(np.diff(lab.indptr))] = lab.indices
    ref[ref < 0] = lab.shape[1] + np.arange((ref < 0).sum())
    assert (agg0 == ref).all()
    S1 = coarse_graph(S0, agg0)
    assert same_graph(S1, info["graphs"][1])
    zero, flags = shell[1][0][2], shell[1][0][3]
    got = patch_sets(*lv[1][1], 9)
    want = rings(S1, zero, flags)
    assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))
    for g, p in enumerate(got):
        assert ((p // 9 == g) | ~zero[p]).all()


# ---------------------------------------------------------------- 5: setup against references


def dense_levels(mesh, free, sigma, A, P, lv, Ac, shell, info):
    """NumPy V-cycle data: [(A_l, S_l)] with Chebyshev over the dense Schwarz of the reference
    patches and the build's lmax, the dense P_l, the coarsest inverse."""
    out, Ps = [], []
    for i, (Al, _, Pl) in enumerate(lv):
        if i == 0:
            Ad = ap_dense(mesh, free, sigma, A)
            ref = node_patch_dofs(column_patches(mesh))
        else:
            Ad = Al.toarray()
            ref = rings(info["graphs"][i], shell[1][i - 1][2], shell[1][i - 1][3])
        S = P.levels[i][1]
        out.append((Ad, cheb(Ad, dense_schwarz(Ad, ref), float(S.lmax), S.degree, float(S.ratio))))
        Ps.append(Pl.toarray())
    return out, Ps, np.linalg.inv(Ac.toarray())


def test_setup_dense():
    """Test 5: on the 8^2 hex8-web mesh P_l, rho and A_l+1 against dense recomputation (the
    identity rows from T's zero columns), and the device V-cycle at degree 3, ratio 20 against
    the NumPy V-cycle; V symmetric, positive on free DOFs, 0 on constrained ones; P's leaves."""
    mesh, _ = web(8, 50)
    sigma = lam1(mesh, dof_mask(mesh, SIMPLY_SUPPORTED))
    op, free, A, Ab, lv, Ac, shell, info = levels_of(mesh, sigma, SIMPLY_SUPPORTED, max_coarse=20)
    assert len(lv) >= 2
    Ad = ap_dense(mesh, free, sigma, A)  # A_b = A_P on ssh8 + hex8 (all nodes are corners)
    assert np.abs(Ab.toarray() - Ad).max() <= 1e-12 * np.abs(Ad).max()
    nxt = [lv[i + 1][0] for i in range(len(lv) - 1)] + [Ac]
    n_id = 0
    for i, (Al, _, Pl) in enumerate(lv):
        Ad = Al.toarray()
        T = shell[1][i][1].toarray()
        Dinv = 1 / np.diag(Ad)
        rho = np.abs(np.linalg.eigvals(Dinv[:, None] * Ad)).max()
        assert 0.9 * rho <= info["rho"][i] <= 1.1 * rho
        Pd = T
        for _ in range(SH.STEPS[min(i, 1)]):
            Pd = Pd - AM.OMEGA / info["rho"][i] * Dinv[:, None] * (Ad @ Pd)
        assert np.abs(Pl.toarray() - Pd).max() <= 1e-12 * np.abs(Pd).max()
        zero = np.abs(T).sum(0) == 0
        n_id += zero.sum()
        G = Pd.T @ Ad @ Pd
        G = (G + G.T) / 2
        G[zero, zero] = np.diag(G)[~zero].mean()
        assert np.abs(nxt[i].toarray() - G).max() <= 1e-12 * np.abs(G).max()
    assert n_id > 0
    P = amg(op, free, sigma, mesh.points, max_coarse=20, degree=3, ratio=20.0)
    levels, Ps, Ci = dense_levels(mesh, free, sigma, A, P, lv, Ac, shell, info)
    U = rhs(free, 3)
    ref = vcycle(levels, Ps, Ci, free.astype(float), U)
    got = np.asarray(P(jnp.asarray(U)))
    assert np.abs(got - ref).max() <= 1e-10 * np.abs(ref).max()
    V = dense(P, op.n_dof)
    assert (V[~free] == 0).all() and (V[:, ~free] == 0).all()
    Vf = V[np.ix_(free, free)]
    assert np.abs(Vf - Vf.T).max() <= 1e-12 * np.abs(Vf).max()
    assert np.linalg.eigvalsh((Vf + Vf.T) / 2).min() > 0
    leaves = jax.tree_util.tree_leaves_with_path(P)
    assert all(
        isinstance(x, jax.Array) or jax.tree_util.keystr(k).endswith(".padding") for k, x in leaves
    )
    assert type(P) is GMG


@pytest.mark.parametrize("name", ["web", "clamped", "strip"])
def test_lmax_bound(name):
    """Test 5: on every level the dense lambda_max(D^-1 A) over all DOFs is within the
    Chebyshev upper bound (1.1 x the build's lmax); identity DOFs have (D^-1 A) e_i = e_i."""
    mesh, bcs = SCHWARZ_MESHES[name]()
    sigma = 1e4
    op, free, A, Ab, lv, Ac, shell, info = levels_of(mesh, sigma, bcs, max_coarse=20)
    P = amg(op, free, sigma, mesh.points, max_coarse=20)
    n_id = 0
    for i, (Al, (blocks, sel), _) in enumerate(lv):
        Ad = ap_dense(mesh, free, sigma, A) if i == 0 else Al.toarray()
        S = SW.build(Al, blocks, sel)
        DA = dense(S, len(Ad)) @ Ad
        lmax = np.linalg.eigvals(DA).real.max()
        assert lmax <= 1.1 * float(P.levels[i][1].lmax) * (1 + 1e-12)
        if i > 0 and shell[1][i - 1][2].any():
            zero = shell[1][i - 1][2]
            e = np.eye(len(Ad))[:, zero]
            assert np.abs(DA @ e - e).max() <= 1e-12
            n_id += 1
    assert n_id > 0 or name == "clamped"


def test_one_level():
    """Test 5: the M4 plate (486 DOF) does not coarsen: the V-cycle is the dense A_P^-1."""
    box = box_mesh((8, 8, 1), spacing=(1 / 8, 1 / 8, 0.05))
    mesh = solid_shell(box, direction=(0, 0, 1))[0]
    sigma = kirchhoff(0.05)
    op, free, A, P = setup(mesh, sigma, SIMPLY_SUPPORTED, max_coarse=1000)
    assert len(P.levels) == 0
    V = dense(P, op.n_dof)[np.ix_(free, free)]
    ref = np.linalg.inv(ap_dense(mesh, free, sigma, A)[np.ix_(free, free)])
    assert np.abs(V - ref).max() <= 1e-10 * np.abs(ref).max()


# ---------------------------------------------------------------- 6-11: PCG counts


def eig(mesh, free, mats=STEEL, k=8):
    """The k lowest eigenvalues (sorted, rigid modes included) by eigsh (fixed v0)."""
    K, M = assemble(mesh, mats)
    Kf, Mf, _ = restrict(K, M, free)
    return np.sort(spla.eigsh(Kf, k, M=Mf, sigma=-1.0, v0=np.ones(Kf.shape[0]))[0])


def spd(mesh, sigma, bcs, **kw):
    """The dense V-cycle is symmetric, SPD on the free DOFs and exactly 0 on constrained ones."""
    op, free, _, P = setup(mesh, sigma, bcs, max_coarse=20, **kw)
    assert len(P.levels) >= 2
    V = dense(P, op.n_dof)
    assert (V[~free] == 0).all() and (V[:, ~free] == 0).all()
    Vf = V[np.ix_(free, free)]
    assert np.abs(Vf - Vf.T).max() <= 1e-10 * np.abs(Vf).max()
    assert np.linalg.eigvalsh((Vf + Vf.T) / 2).min() > 0


@pytest.mark.slow
def test_lt_growth():
    """Test 6 (roadmap): on the 128^2 flat SS plate count(L/t 1000) <= 2 count(L/t 10); the
    128^2 curved panel's counts (sigma from eigsh on its 64^2 version) are gated."""
    its = {}
    for Lt in (10, 1000):
        mesh, t = plate(128, Lt)
        its[Lt] = count(mesh, kirchhoff(t), SIMPLY_SUPPORTED, max_coarse=1000)
        gate(f"flat128-{Lt}", its[Lt])
        m64 = plate(64, Lt, curved=True)[0]
        sigma = lam1(m64, dof_mask(m64, SIMPLY_SUPPORTED))
        mesh = plate(128, Lt, curved=True)[0]
        its[f"curved{Lt}"] = count(mesh, sigma, SIMPLY_SUPPORTED, max_coarse=1000)
        gate(f"curved128-{Lt}", its[f"curved{Lt}"])
    print("counts", its)
    assert its[1000] <= 2 * its[10], its


@pytest.mark.slow
@pytest.mark.parametrize("Lt, n", [(100, 64), (1000, 128)])
def test_refinement(Lt, n):
    """Test 7: at fixed L/t the count on 2n^2 is <= 1.3x the n^2 count (max_coarse 1000)."""
    its = []
    for m in (n, 2 * n):
        mesh, t = plate(m, Lt)
        its.append(count(mesh, kirchhoff(t), SIMPLY_SUPPORTED, max_coarse=1000))
        gate(f"flat{m}-{Lt}", its[-1])
    assert its[1] <= 1.3 * its[0], its


def test_shells():
    """Test 8: counts at L/t = 50 on the flat plate (with Jacobi's reported), 3 plies, the
    curved panel, the 270 degree sweep-built cylinder, the T and L junctions and the 3/3 T, and
    the flat plate at L/t = 300."""
    mesh, t = plate(32, 50)
    gate("flat32-50", count(mesh, kirchhoff(t), SIMPLY_SUPPORTED))
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, SIMPLY_SUPPORTED)
    A = ap_operator(op, free, kirchhoff(t))
    _, info = pcg(A, jnp.asarray(rhs(free, 4)), jacobi(op, free, kirchhoff(t)), 1e-6, 2000)
    print("Jacobi", np.asarray(info["iterations"]).max())
    mats = {0: laminate(three_plies(C_ORT))}
    sigma = eig(mesh, free, mats, 1)[0]
    gate("plies32-50", count(mesh, sigma, SIMPLY_SUPPORTED, mats))
    mesh = plate(32, 50, curved=True)[0]
    gate("curved32-50", count(mesh, lam1(mesh, dof_mask(mesh, SIMPLY_SUPPORTED)), SIMPLY_SUPPORTED))
    mesh = cylinder(False)
    gate("cylinder", count(mesh, lam1(mesh, dof_mask(mesh, {})), {}))
    for kind, k in (("T", 1), ("L", 1), ("T", 3)):
        mesh = junction(kind, 24, 50, k, k)[0]
        gate(f"{kind}{k}", count(mesh, lam1(mesh, dof_mask(mesh, CLAMPED)), CLAMPED))
    mesh, t = plate(32, 300)
    gate("flat32-300", count(mesh, kirchhoff(t), SIMPLY_SUPPORTED))


def test_mixed():
    """Test 9: the 32^2 hex8-web plate and the 16^3 hex8 block with an ssh8 skin (the PR 13
    count of the block without the skin is reported)."""
    mesh = web(32, 50)[0]
    gate("web32", count(mesh, lam1(mesh, dof_mask(mesh, SIMPLY_SUPPORTED)), SIMPLY_SUPPORTED))
    bc = {"zmin": "xyz"}
    mesh = block(16)[0]
    gate("block16", count(mesh, lam1(mesh, dof_mask(mesh, bc)), bc))
    cube = box_mesh((16, 16, 16), spacing=(1 / 16,) * 3)
    print("PR 13 hex8", count(cube, lam1(cube, dof_mask(cube, bc)), bc))


def bottom(n, Lt):
    """The SS plate's mesh with the node set "bottom" (its bottom edges: partly constrained
    fibers)."""
    mesh, t = plate(n, Lt)
    mesh.node_sets["bottom"] = sets(mesh.points, t)["bottom"]
    return mesh, t


def parts(t):
    """A 32^2 plate and a 4^2 one (same h) 1/2 apart, each clamped on its edge x = min."""
    big = box_mesh((32, 32, 1), spacing=(1 / 32, 1 / 32, t))
    small = box_mesh((4, 4, 1), spacing=(1 / 32, 1 / 32, t))
    small.points[:, 0] += 1.5
    mesh = glue([(big, (0, 0, 1)), (small, (0, 0, 1))], t)
    x = mesh.points[:, 0]
    mesh.node_sets["ends"] = np.flatnonzero((np.abs(x) < 1e-9) | (np.abs(x - 1.5) < 1e-9))
    return mesh


def test_constraints():
    """Test 10: clamped and bottom-supported plates (V-cycle SPD, 0 on constrained DOFs,
    counts gated), two separate plates of very different sizes (a singleton aggregate at
    max_coarse 40), a free plate at sigma = lambda_7 and a clamped one at sigma = 0 with
    nonsingular=True."""
    B = {"bottom": "xyz"}
    spd(plate(8, 50)[0], kirchhoff(0.02, 35.99), CLAMPED)
    spd(bottom(8, 50)[0], 1e4, B)
    mesh, t = plate(32, 50)
    gate("clamped32", count(mesh, kirchhoff(t, 35.99), CLAMPED))
    gate("clamped32-0", count(mesh, 0.0, CLAMPED, nonsingular=True))
    spd(plate(8, 50)[0], 0.0, CLAMPED, nonsingular=True)
    mesh, _ = bottom(32, 50)
    gate("bottom32", count(mesh, lam1(mesh, dof_mask(mesh, B)), B))
    mesh = parts(0.02)
    bc = {"ends": "xyz"}
    sigma = lam1(mesh, dof_mask(mesh, bc))
    *_, shell, _ = levels_of(mesh, sigma, bc, max_coarse=40)
    sizes = [np.bincount(agg) for agg, *_ in shell[1]]
    assert any((s == 1).any() for s in sizes[1:]), sizes
    gate("parts", count(mesh, sigma, bc, max_coarse=40))
    mesh = plate(24, 50)[0]
    lam = eig(mesh, dof_mask(mesh, {}))
    assert lam[5] < 1e-6 * lam[6]
    m8 = plate(8, 50)[0]
    spd(m8, eig(m8, dof_mask(m8, {}))[6], {})
    gate("free24", count(mesh, lam[6], {}))


def test_stacks():
    """Test 11: the 3-layer 32^2 plate at L/t 10 / 100 gated; 300 / 1000 reported."""
    for Lt in (10, 100, 300, 1000):
        mesh, t = plate(32, Lt, layers=3)
        its = count(mesh, kirchhoff(t), SIMPLY_SUPPORTED, maxit=2000)
        if Lt <= 100:
            gate(f"stack32-{Lt}", its)
        print("stack", Lt, its)


# ---------------------------------------------------------------- 12: reuse


def test_reuse(monkeypatch):
    """Test 12: aggregates= with new materials, s_K and sigma equals a fresh setup; fibers,
    aggregation and fits are skipped, patches rebuilt; connectivity or mask changes raise; two
    fresh builds are equal exactly."""
    mesh, _ = web(8, 50)
    free = dof_mask(mesh, SIMPLY_SUPPORTED)
    op = unstructured(mesh, STEEL)
    P1, agg = amg(op, free, 1e4, mesh.points, max_coarse=20, return_aggregates=True)
    U = jnp.asarray(rhs(free, 2))
    assert np.array_equal(
        np.asarray(P1(U)), np.asarray(amg(op, free, 1e4, mesh.points, max_coarse=20)(U))
    )
    op2 = unstructured(mesh, {0: Material(isotropic(70e9, 0.33), 2700.0)})
    op2 = op2.scaled(s_K=jnp.linspace(0.5, 1.5, op2.n_el))
    fresh = amg(op2, free, 3e4, mesh.points, max_coarse=20)
    calls = {"rings": 0, "columns": 0}

    def no(*a, **k):
        raise AssertionError("skipped on reuse")

    for f in ("_fibers", "_aggregate", "fit_candidates", "_curvatures"):
        monkeypatch.setattr(SH, f, no)
    for f, key in (("_rings", "rings"), ("_columns", "columns")):
        orig = getattr(SH, f)

        def spy(*a, _o=orig, _k=key, **k):
            calls[_k] += 1
            return _o(*a, **k)

        monkeypatch.setattr(SH, f, spy)
    P2 = amg(op2, free, 3e4, mesh.points, aggregates=agg)
    monkeypatch.undo()
    assert calls["columns"] == 1 and calls["rings"] >= 1
    a, b = np.asarray(P2(U)), np.asarray(fresh(U))
    assert np.abs(a - b).max() <= 1e-12 * np.abs(b).max()
    with pytest.raises(ValueError, match="another free mask"):
        amg(op, dof_mask(mesh, CLAMPED), 1e4, mesh.points, aggregates=agg)
    other = plate(8, 50)[0]
    with pytest.raises(ValueError, match="another connectivity"):
        amg(unstructured(other, STEEL), dof_mask(other, SIMPLY_SUPPORTED), 1e4, other.points,
            aggregates=agg)  # fmt: skip
    box = box_mesh((8, 8, 1), spacing=(1 / 8, 1 / 8, 0.02))  # the ssh8 plate's connectivity
    m = plate(8, 50)[0]
    _, ag = amg(unstructured(m, STEEL), dof_mask(m, {}), 1e4, m.points, return_aggregates=True)
    with pytest.raises(ValueError, match="another connectivity"):
        amg(unstructured(box, STEEL), dof_mask(box, {}), 1e4, box.points, aggregates=ag)


# ---------------------------------------------------------------- 13: the hex path


@pytest.mark.parametrize("t", ["hex20", "hex8"])
def test_hex_path(monkeypatch, t):
    """Test 13: hex meshes never reach the shell code (build, PCG, and hex8 reuse)."""

    def no(*a, **k):
        raise AssertionError("shell path on a hex mesh")

    monkeypatch.setattr(SH, "_shell_levels", no)
    monkeypatch.setattr(SW, "build", no)
    mesh = box_mesh((4, 3, 3), type=t)
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"})
    sigma = 1e6
    P, agg = amg(op, free, sigma, mesh.points, max_coarse=20, return_aggregates=True)
    _, info = pcg(ap_operator(op, free, sigma), jnp.asarray(rhs(free, 2)), P, 1e-8, 300)
    assert np.asarray(info["converged"]).all()
    if t == "hex8":
        amg(op, free, sigma, mesh.points, aggregates=agg)
