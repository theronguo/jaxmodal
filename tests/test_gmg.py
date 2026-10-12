"""Voxel GMG against SciPy Galerkin products, dense V-cycle references and PCG iteration bounds."""

import dataclasses
import functools
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmodal.eigen import lobpcg, reference_modes, sigma_estimate
from jaxmodal.linalg import pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, rcm, voxel_mesh
from jaxmodal.operators import (
    a_p,
    ap_operator,
    assemble,
    dof_mask,
    m_orthonormalize,
    rigid_modes,
    stencil,
    unstructured,
)
from jaxmodal.precond import gmg, jacobi

G = importlib.import_module("jaxmodal.precond.gmg")
krylov = importlib.import_module("jaxmodal.linalg.krylov")
lob = importlib.import_module("jaxmodal.eigen.lobpcg")

UNIT = {0: Material(isotropic(3.0, 0.3), 2.0), 1: Material(isotropic(30.0, 0.25), 0.5)}
STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
SIMP = Path(__file__).parent / "data" / "simp32.npy"


def rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max()


# ---------------------------------------------------------------- exactness meshes (unit materials)


@functools.cache
def exact_case(name, mass="consistent"):
    """(mesh, op, free, A_P SciPy, sigma) for the exactness tests: two phases, random s_K, s_M,
    nodal masses on half the nodes. A: 7x5x3 masked with an inactive coarse cell next to active
    ones (odd x with ceil(7/2) even: level-2 weights 1/3, 2/3); B: 7x3x2 (stopped axes from the
    start); C: 3x4x9 (z longest: the other stored axis order), masked, sliding BCs."""
    rng = np.random.default_rng({"A": 0, "B": 1, "C": 2}[name])
    shape, bcs = {
        "A": ((7, 5, 3), {"xmin": "xyz"}),
        "B": ((7, 3, 2), {"xmin": "xyz"}),
        "C": ((3, 4, 9), {"zmin": "z", "xmin": "x"}),
    }[name]
    mask = rng.random(shape) < 0.85
    mask[0] = True
    mask[:, :, 0] = True
    if name == "A":
        mask[4:6, 2:4, 2] = False  # coarse cell (2, 1, 1) inactive, its neighbours active
        mask[6, 4, 2] = False  # corner coarse cell (3, 2, 1) inactive: coarse node dropped
    m = voxel_mesh(shape, (0.5, 1.0, 0.7), mask=mask)
    m = dataclasses.replace(m, material=rng.integers(0, 2, m.n_el))
    nodal = rng.uniform(0, 1, len(m.points)) * (rng.uniform(size=len(m.points)) < 0.5)
    s_K, s_M = rng.uniform(0.1, 2, m.n_el), rng.uniform(0.1, 2, m.n_el)
    free = dof_mask(m, bcs)
    op = stencil(m, UNIT, mass=mass, nodal_mass=nodal).scaled(s_K=s_K, s_M=s_M)
    K, M = assemble(m, UNIT, mass=mass, nodal_mass=nodal, s_K=s_K, s_M=s_M)
    sigma = 0.37
    return m, op, free, a_p(K + sigma * M, free).toarray(), sigma, M


def dense(f, n):
    return np.asarray(f(jnp.eye(n)))


def hierarchy(P, A0):
    """Dense transfers and Galerkin levels P_l^T A_l P_l from the dense A0."""
    Ps, As = [], [A0]
    for T in P.transfers:
        Ps.append(dense(T, 3 * T.n_coarse))
        As.append(Ps[-1].T @ As[-1] @ Ps[-1])
    return Ps, As


def grids(mesh):
    """Host level grids from the module's own coarsening (positions, numbering)."""
    vx = mesh.voxel
    F = G._Grid(vx.shape, (vx.element_map >= 0).reshape(vx.shape[::-1]).transpose(2, 1, 0),
                tuple(np.arange(n + 1.0) for n in vx.shape))  # fmt: skip
    out = [F]
    while max(out[-1].shape) > 2:
        out.append(G._coarse(out[-1])[0])
    return out


# ---------------------------------------------------------------- 1-5: exactness


@pytest.mark.parametrize("chunk", [None, 7])
@pytest.mark.parametrize("mass", ["consistent", "lumped"])
@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_galerkin(name, mass, chunk, monkeypatch):
    """Test 1: every coarse level equals P^T A_P P from SciPy (several chunks with chunk 7)."""
    if chunk:
        monkeypatch.setattr(G, "SETUP_CHUNK", chunk)
    m, op, free, A0, sigma, M = exact_case(name, mass)
    P = gmg(op, free, sigma)
    Ps, As = hierarchy(P, A0)
    assert len(P.levels) >= 2 and len(P.transfers) == len(P.levels)
    for lvl in range(1, len(P.levels)):
        assert rel(dense(P.levels[lvl][0], As[lvl].shape[0]), As[lvl]) <= 1e-12
    assert rel(np.asarray(P.L @ P.L.T), As[-1]) <= 1e-12
    mass_part = sigma * Ps[0].T @ (M.toarray() * np.outer(free, free)) @ Ps[0]
    assert np.abs(mass_part).max() / np.abs(As[1]).max() > 1e-4


@pytest.mark.parametrize("name", ["A", "beam"])
def test_transfers(name):
    """Test 2: P reproduces linear fields in physical coordinates on every level, restriction is
    P^T, kept columns are nonzero, dropped grid nodes' columns are exactly zero."""
    if name == "A":
        m, op, free = exact_case("A")[:3]
    else:
        m = voxel_mesh((15, 2, 2), (0.3, 1.0, 1.0))
        op, free = stencil(m, UNIT), dof_mask(m, {"xmin": "xyz"})
    P = gmg(op, free, 0.37)
    h = np.asarray(m.voxel.spacing)
    gs = grids(m)
    rng = np.random.default_rng(0)
    for lvl, T in enumerate(P.transfers):
        F, C = gs[lvl], gs[lvl + 1]
        xf = np.stack([F.pos[d][F.node_ijk()[:, d]] for d in range(3)], 1) * h
        xc = np.stack([C.pos[d][C.node_ijk()[:, d]] for d in range(3)], 1) * h
        g = rng.standard_normal((3, 4))
        u = np.asarray(T(jnp.asarray((xc @ g[:, :3].T + g[:, 3]).reshape(-1, 1))))
        assert rel(u.ravel(), (xf @ g[:, :3].T + g[:, 3]).ravel()) <= 1e-12
        Pm = dense(T, 3 * T.n_coarse)
        r = rng.standard_normal((Pm.shape[0], 2))
        assert rel(G._restrict(T, jnp.asarray(r)), Pm.T @ r) <= 1e-14
        assert (np.abs(Pm).sum(0) > 0).all()
        if name == "A" and lvl == 0:  # the full coarse grid: dropped nodes give zero columns
            nx, ny, nz = C.shape
            full = dataclasses.replace(
                T, gc=jnp.arange((nx + 1) * (ny + 1) * (nz + 1)), n_coarse=len(C.node_map)
            )
            Pf = dense(full, 3 * len(C.node_map)).reshape(Pm.shape[0], -1, 3)
            dropped = C.node_map < 0
            assert dropped.any() and (Pf[:, dropped] == 0).all()
            assert (np.abs(Pf[:, ~dropped]).sum((0, 2)) > 0).all()


def group_blocks_dense(A, Gr):
    """(n_groups, 3k, 3k) diagonal group blocks of the dense level matrix (absent slots: I)."""
    idx = G._groups(Gr)[0]
    n = Gr.n_nodes
    B = np.zeros((len(idx), 3 * idx.shape[1], 3 * idx.shape[1]))
    for g, row in enumerate(idx):
        d = np.concatenate([3 * v + np.arange(3) if v < n else [-1] * 3 for v in row])
        ok = d >= 0
        B[g][np.ix_(ok, ok)] = A[np.ix_(d[ok], d[ok])]
        B[g][~ok, ~ok] = 1.0
    return B, idx


@pytest.mark.parametrize("chunk", [None, 7])
@pytest.mark.parametrize("name", ["A", "B"])
def test_group_blocks(name, chunk, monkeypatch):
    """Test 3: each level's smoother blocks are the diagonal group blocks of the dense level
    operator (level 0: SciPy A_P), and the group apply is their block inverse, masked."""
    if chunk:
        monkeypatch.setattr(G, "SETUP_CHUNK", chunk)
    m, op, free, A0, sigma, _ = exact_case(name)
    P = gmg(op, free, sigma)
    As = hierarchy(P, A0)[1]
    rng = np.random.default_rng(1)
    ks = []
    for lvl, Gr in enumerate(grids(m)[: len(P.levels)]):
        B, idx = group_blocks_dense(As[lvl], Gr)
        D = P.levels[lvl][1].precond
        v = rng.standard_normal((As[lvl].shape[0], 2))
        n = Gr.n_nodes
        vg = np.concatenate([v.reshape(n, 3, 2), np.zeros((1, 3, 2))])[idx].reshape(len(idx), -1, 2)
        yg = np.linalg.solve(B, vg).reshape(-1, 3, 2)
        y = np.zeros((n + 1, 3, 2))
        y[idx.ravel()] = yg
        assert rel(D(jnp.asarray(v)), y[:n].reshape(-1, 2)) <= 1e-12
        ks.append(idx.shape[1])
    assert max(ks) > 1  # group blocks (B: level 0; A: a coarse level with z stopped)


def vcycle_dense(Ps, As, Ss, b, lvl=0):
    if lvl == len(Ss):
        return np.linalg.solve(As[lvl], b)
    A, S, P = As[lvl], Ss[lvl], Ps[lvl]
    x = S @ b
    x = x + P @ vcycle_dense(Ps, As, Ss, P.T @ (b - A @ x), lvl + 1)
    return x + S @ (b - A @ x)


@pytest.mark.parametrize("name", ["A", "B"])
def test_vcycle_dense(name):
    """Test 4: the V-cycle equals the NumPy composition of the densified pieces."""
    m, op, free, A0, sigma, _ = exact_case(name)
    P = gmg(op, free, sigma)
    Ps, As = hierarchy(P, A0)
    Ss = [dense(S, A.shape[0]) for (_, S), A in zip(P.levels, As, strict=False)]
    U = np.random.default_rng(2).standard_normal((A0.shape[0], 3))
    f = free[:, None]
    ref = f * vcycle_dense(Ps, As, Ss, f * U)
    assert rel(P(jnp.asarray(U)), ref) <= 1e-10
    assert rel(P(jnp.asarray(f * U)), ref) <= 1e-10


@pytest.mark.parametrize("bcs", [{"xmin": "xyz"}, {}])
def test_symmetric(bcs):
    """Test 5: symmetric on unmasked inputs, positive definite on masked ones, masked output."""
    m = voxel_mesh((6, 5, 4))
    op, free = stencil(m, UNIT), dof_mask(m, bcs)
    P = gmg(op, free, 0.37)
    rng = np.random.default_rng(3)
    u, v = (rng.standard_normal((op.n_dof, 4)) for _ in range(2))
    Pu, Pv = np.asarray(P(jnp.asarray(u))), np.asarray(P(jnp.asarray(v)))
    assert np.abs(u.T @ Pv - (v.T @ Pu).T).max() <= 1e-12 * np.abs(u.T @ Pv).max()
    w = u * free[:, None]
    assert (np.einsum("ij,ij->j", w, np.asarray(P(jnp.asarray(w)))) > 0).all()
    assert (Pu[~free] == 0).all()


# ---------------------------------------------------------------- 5b: island deflation


def island_case(soft=1e-6):
    """12x6x4 unit grid, clamped xmin, in soft cells s_K = s_M = soft: a stiff body (x < 4,
    y < 3), a stiff island (x 7-8, y 2-3, z 1-2), an island of two cells touching only at a
    vertex ((10, 0, 3), (11, 1, 2)) and a clamped single-cell island (0, 5, 3)."""
    m = voxel_mesh((12, 6, 4), (0.5, 1.0, 0.7))
    s = np.full((4, 6, 12), soft)  # (z, y, x) = element order
    s[:, :3, :4] = 1.0
    s[1:3, 2:4, 7:9] = 1.0
    s[3, 0, 10] = s[2, 1, 11] = 1.0
    s[3, 5, 0] = 1.0
    op = stencil(m, UNIT).scaled(s_K=s.ravel(), s_M=s.ravel())
    free = dof_mask(m, {"xmin": "xyz"})
    K, M = assemble(m, UNIT, s_K=s.ravel(), s_M=s.ravel())
    return m, op, free, a_p(K + 0.37 * M, free).toarray(), s


def test_islands():
    """Islands: node-connected stiff components but the largest, largest first; none below
    the contrast threshold or on a uniform field."""
    m, op, _, _, s = island_case()
    ids = np.arange(m.n_el).reshape(4, 6, 12)
    cells = G.grid_ids(op)[0]
    got = G._islands(cells, np.asarray(op.s_K))
    want = [sorted(ids[1:3, 2:4, 7:9].ravel()), sorted([ids[3, 0, 10], ids[2, 1, 11]])]
    assert [sorted(i) for i in got] == want + [[ids[3, 5, 0]]]
    assert G._islands(cells, np.where(s.ravel() < 1, 2e-3, 1.0)) == []  # contrast 500
    assert G._islands(cells, np.ones(m.n_el)) == []
    got = G._islands(cells, np.where(s.ravel() < 1, 1e-3, 1.0))  # contrast exactly 1e3
    assert [sorted(i) for i in got] == want + [[ids[3, 5, 0]]]
    t = s.copy()
    t[1, 1, 6] = 1.0  # touches the block island along an edge only
    t[0, 5, 6] = 5e-6  # 5 min(s_K): soft (stiff means > 10 min)
    t[0, 5, 9] = 2e-5  # 20 min(s_K): stiff, its own island
    got = G._islands(cells, t.ravel())
    assert sorted(got[0]) == sorted([*want[0], ids[1, 1, 6]]) and len(got) == 4
    assert sorted(map(list, got[2:])) == [[ids[0, 5, 9]], [ids[3, 5, 0]]]


def test_deflation_dense():
    """The deflated cycle equals Q + (I - Q A) V (I - A Q) with dense A_P, the undeflated V and
    Z built here from the island nodes' positions (translations, rotations about the
    centroid, constrained rows 0)."""
    m, op, free, A0, s = island_case()
    P = gmg(op, free, 0.37)
    assert P.defl is not None
    V = dense(dataclasses.replace(P, defl=None), A0.shape[0])
    conn = m.blocks["hex8"]
    cols = []
    for e in G._islands(G.grid_ids(op)[0], np.asarray(op.s_K)):
        nodes = np.unique(conn[e])
        x = m.points[nodes] - m.points[nodes].mean(0)
        for j in range(6):
            u = np.zeros((len(m.points), 3))
            if j < 3:
                u[nodes, j] = 1
            else:
                u[nodes] = np.cross(np.eye(3)[j - 3], x)
            cols.append(u.ravel() * free)
    Zr = np.array(cols).T
    Q = Zr @ np.linalg.pinv(Zr.T @ A0 @ Zr) @ Zr.T
    eye = np.eye(A0.shape[0])
    ref = Q + (eye - Q @ A0) @ V @ (eye - A0 @ Q)
    got = dense(P, A0.shape[0])
    assert rel(got, ref) <= 1e-10
    assert (got[~free] == 0).all() and (got[:, ~free] == 0).all()
    assert rel(got, got.T) <= 1e-12
    assert np.linalg.eigvalsh(got[np.ix_(free, free)]).min() > 0


def test_deflation_shapes():
    """Per-row deflation storage padded to a power of 2 rows, 6 MAX_ISLANDS columns in E^+
    (one compiled cycle per row bucket); rows on two islands' rings appear once per island;
    no island gives defl None (the plain V-cycle)."""
    m, op, free, _, s = island_case()
    rows, isl, Z, AZ, Einv = gmg(op, free, 0.37).defl
    n = len(rows)
    assert Z.shape == AZ.shape == (n, 6) and n >= 256 and n & (n - 1) == 0
    assert Einv.shape == (6 * G.MAX_ISLANDS,) * 2 and not np.asarray(Einv)[18:].any()
    isl, rows = np.asarray(isl), np.asarray(rows)
    live = np.abs(np.asarray(Z)).sum(1) + np.abs(np.asarray(AZ)).sum(1) > 0
    assert set(isl[live]) == {0, 1, 2}
    for c in (0, 1, 2):  # each island's rows are distinct
        assert len(np.unique(rows[live & (isl == c)])) == (live & (isl == c)).sum()
    t = s.copy()
    t[1:3, 2:4, 7:9] = t[3, 0, 10] = t[2, 1, 11] = 1e-6  # only the single-cell island left
    rows = gmg(op.scaled(s_K=t.ravel(), s_M=t.ravel()), free, 0.37).defl[0]
    assert len(rows) == 256  # 81 live rows: the 256-row floor
    _, op, free, _, _ = island_case(soft=1.0)
    assert gmg(op, free, 0.37).defl is None


def test_defaults():
    """Chebyshev degree 2 on the fine level, 4 on coarse levels, ratio 30."""
    op, free, _ = steel((8,) * 3, {"xmin": "xyz"})
    P = gmg(op, free, 1e7)
    degs = [S.degree for _, S in P.levels]
    assert degs[0] == 2 and set(degs[1:]) == {4} and len(degs) > 1
    assert all(float(S.ratio) == 30.0 for _, S in P.levels)


def test_grid_ids_padded():
    """grid_ids crops slab padding and empty trailing scan layers; gmg does not depend on the
    slab."""
    mask = np.ones((7, 5, 3), bool)
    mask[6] = False
    m = voxel_mesh((7, 5, 3), mask=mask)
    ops = [stencil(m, UNIT, slab=s) for s in (None, 2, 4)]
    assert ops[1].cell.shape[0] > 6
    ref = G.grid_ids(ops[0])
    free = dof_mask(m, {"xmin": "xyz"})
    U = jnp.asarray(np.random.default_rng(7).standard_normal((ops[0].n_dof, 2)))
    for o in ops[1:]:
        c, n = G.grid_ids(o)
        assert c.shape == (6, 5, 3) and n.shape == (7, 6, 4)
        assert (c == ref[0]).all() and (n == ref[1]).all()
        assert rel(gmg(o, free, 0.37)(U), gmg(ops[0], free, 0.37)(U)) <= 1e-13


# ---------------------------------------------------------------- 6-10: PCG iterations (steel)


def steel(shape, bcs, rho=None):
    """(op, free, mesh) on the unit-length grid (spacing 1 / max(shape))."""
    m = voxel_mesh(shape, (1 / max(shape),) * 3)
    op = stencil(m, STEEL)
    if rho is not None:
        op = op.scaled(s_K=rho**3, s_M=rho)
    return op, dof_mask(m, bcs), m


def sigma_of(op, free, m):
    """sigma_estimate (seed 0), with the M-orthonormal rigid modes for free bodies."""
    R = jnp.asarray(rigid_modes(m, free))
    R = m_orthonormalize(R, op.apply_M(R)) if R.shape[1] else None
    return sigma_estimate(op, free, R=R)[0]


def iterations(op, free, sigma):
    P = gmg(op, free, sigma)
    b = np.random.default_rng(1).standard_normal((op.n_dof, 4)) * free[:, None]
    info = pcg(ap_operator(op, free, sigma), jnp.asarray(b), P, 1e-8, 1000)[1]
    assert np.asarray(info["converged"]).all()
    return int(np.asarray(info["iterations"]).max()), P


@pytest.mark.parametrize("n", [8, 16, 32, pytest.param(64, marks=pytest.mark.slow)])
def test_h_independence(n):
    """Test 6: at fixed sigma (the 8^3 estimate), iterations grow <= 20% from 8^3."""
    op8, free8, m8 = steel((8,) * 3, {"xmin": "xyz"})
    sigma = sigma_of(op8, free8, m8)
    base = iterations(op8, free8, sigma)[0]
    op, free, _ = steel((n,) * 3, {"xmin": "xyz"})
    assert iterations(op, free, sigma)[0] <= 1.2 * base  # measured 13 at 8^3 ... 64^3


@pytest.mark.slow
def test_simp():
    """Test 7: the checked-in filtered binary SIMP field (rho in [1e-3, 1], s_K = rho^3, s_M =
    rho) at its own lam_1 (reference_modes, an island mode) and at the solid box's: <= 100
    iterations (measured 87 and 33)."""
    from benchmarks.gmg_study import lam1

    rho = np.load(SIMP)
    assert rho.shape == (32, 32, 32) and rho.dtype == np.float32
    assert rho.min() == np.float32(1e-3) and rho.max() == 1.0
    assert abs(float(rho.mean()) - 0.40059) < 1e-4
    op, free, m = steel((32,) * 3, {"xmin": "xyz"})
    solid = sigma_of(op, free, m)
    r = rho.transpose(2, 1, 0).ravel().astype(float)
    op = op.scaled(s_K=r**3, s_M=r)
    own = lam1(op, free, m)
    assert own < 1e-5 * solid
    for sigma in (own, solid):
        assert iterations(op, free, sigma)[0] <= 100


def test_single_layer():
    """Test 8: plate 32^2 x 1, clamped xmin, sigma = lam_1: <= 100 iterations (measured 37;
    thin plates grow with h, docs/preconditioners.md)."""
    from benchmarks.gmg_study import lam1

    op, free, m = steel((32, 32, 1), {"xmin": "xyz"})
    assert iterations(op, free, lam1(op, free, m))[0] <= 100


def test_free():
    """Test 9 (iterations): free 32^3 at sigma > 0: <= 100 (measured 13); finite coarsest
    factor."""
    op, free, m = steel((32,) * 3, {})
    its, P = iterations(op, free, sigma_of(op, free, m))
    assert its <= 100 and np.isfinite(np.asarray(P.L)).all()


@pytest.mark.parametrize(
    "shape, bound", [((128, 1, 1), 100), ((64, 2, 2), 60), ((33, 8, 8), 30)]
)  # measured 58, 23, 14
def test_beam_and_odd(shape, bound):
    """Test 10: beams 128 x 1 x 1 and 64 x 2 x 2 and the odd 33 x 8 x 8 box at sigma = lam_1."""
    from benchmarks.gmg_study import lam1

    op, free, m = steel(shape, {"xmin": "xyz"})
    assert iterations(op, free, lam1(op, free, m))[0] <= bound


# ---------------------------------------------------------------- 9: singular guard


def unit(shape, mask=None, bcs=None):
    m = voxel_mesh(shape, mask=mask)
    return stencil(m, UNIT), dof_mask(m, bcs or {}), m


def test_guard_raises():
    """Test 9 (guard): sigma = 0 with rigid modes, zero-stiffness cells or meshes rigid_modes
    rejects raises the guard's message; the backstop has its own."""
    op, free, _ = unit((16, 16, 16))
    with pytest.raises(ValueError, match="sigma > 0"):
        gmg(op, free, 0.0)
    op, free, _ = unit((64, 64, 1), bcs={"xmin": "x"})
    with pytest.raises(ValueError, match="sigma > 0"):
        gmg(op, free, 0.0)
    op, free, _ = unit((8, 8, 8), bcs={"xmin": "xyz"})
    s_K = np.ones(op.n_el)
    s_K[np.asarray(voxel_mesh((8, 8, 8)).voxel.element_map).reshape(8, 8, 8)[:, :, 4].ravel()] = 0
    with pytest.raises(ValueError, match="zero-stiffness"):
        gmg(op.scaled(s_K=s_K), free, 0.0)
    edge = np.zeros((2, 2, 1), bool)
    edge[0, 0, 0] = edge[1, 1, 0] = True
    islands = np.zeros((4, 24, 1), bool)
    islands[:, ::2] = True
    for mask, bcs in ((edge, {"xmin": "xyz", "xmax": "xyz"}), (islands, {"xmin": "xyz"})):
        op, free, _ = unit(mask.shape, mask, bcs)
        with pytest.raises(ValueError, match="sigma > 0") as err:
            gmg(op, free, 0.0)
        assert isinstance(err.value.__cause__, ValueError)
    op, free, _ = unit(edge.shape, edge, {"xmin": "xyz", "xmax": "xyz"})
    gmg(op, free, sigma_estimate(op, free)[0])  # both cells clamped: no rigid mode


def test_guard_builds(monkeypatch):
    """Test 9 (guard): SPD cases build at sigma = 0 or the estimate; zero cells that leave free
    nodes without stiffness or mass raise at the smoother; a NaN coarsest level raises the
    backstop message."""
    op, free, _ = unit((128, 128, 1), bcs={"xmin": "xyz"})
    gmg(op, free, 0.0)
    rho = np.where(np.arange(32**3) % 32**2 < 32**2 // 2, 1e-3, 1.0)  # y < 16 void
    op, free, m = steel((32,) * 3, {"xmin": "xyz"}, rho=rho)
    gmg(op, free, sigma_of(op, free, m))
    op, free, _ = unit((16, 4, 4), bcs={"xmin": "xyz"})
    s = np.where(np.isin(np.arange(op.n_el) % 16, (6, 7, 8)), 0.0, 1.0)
    with pytest.raises(ValueError, match="smoother blocks are not finite"):
        gmg(op.scaled(s_K=s, s_M=s), free, 0.37)
    real = G._dense
    monkeypatch.setattr(G, "_dense", lambda *a: real(*a) * np.nan)
    op, free, _ = unit((6, 5, 4), bcs={"xmin": "xyz"})
    with pytest.raises(ValueError, match="not finite") as err:
        gmg(op, free, 0.37)
    assert "sigma > 0" not in str(err.value)


# ---------------------------------------------------------------- 11-13


def test_lobpcg():
    """Test 11: LOBPCG with GMG on a 9x6x4 clamped masked box (odd x) matches reference_modes,
    in fewer iterations than with Jacobi."""
    rng = np.random.default_rng(4)
    mask = rng.random((9, 6, 4)) < 0.9
    mask[0] = True
    m = voxel_mesh((9, 6, 4), (1 / 9, 1 / 9, 1 / 9), mask=mask)
    op, free = stencil(m, STEEL), dof_mask(m, {"xmin": "xyz"})
    K, M = assemble(m, STEEL)
    lr = reference_modes(K, M, 12, free, points=m.points)[0]
    sigma = sigma_of(op, free, m)
    lam, _, info = lobpcg(op, free, 6, tol=1e-8, precond=gmg(op, free, sigma))
    lam_j, _, info_j = lobpcg(op, free, 6, tol=1e-8, precond=jacobi(op, free))
    assert info["converged"] and info_j["converged"]
    assert np.abs(np.asarray(lam) / lr[:6] - 1).max() <= 1e-8  # isolated modes (9a floor)
    assert info["iterations"] < info_j["iterations"]


def test_no_recompile():
    """Test 12: a new design on the same grid reuses the compiled pcg, V-cycle and LOBPCG step."""
    m = voxel_mesh((9, 6, 4), (1 / 9,) * 3)
    op, free = stencil(m, STEEL), dof_mask(m, {"xmin": "xyz"})
    b = jnp.asarray(np.random.default_rng(5).standard_normal((op.n_dof, 2)) * free[:, None])
    sizes = []
    for s in (np.ones(op.n_el), np.random.default_rng(6).uniform(0.5, 1.5, op.n_el)):
        o = op.scaled(s_K=s)
        P = gmg(o, free, 1e7)
        pcg(ap_operator(o, free, 1e7), b, P, 1e-8, 200)
        lobpcg(o, free, 2, tol=1e-4, precond=P, maxit=3)
        sizes.append(
            (
                krylov._pcg._cache_size(),
                G._vcycle_apply._cache_size(),
                lob.lobpcg_step._cache_size(),
            )
        )
    assert sizes[0] == sizes[1]


def test_errors():
    """Test 13: unstructured operator, sigma < 0, renumbered voxel mesh, x64 off."""
    b = box_mesh((2, 2, 2))
    with pytest.raises(TypeError):
        gmg(unstructured(b, UNIT), dof_mask(b, {"xmin": "xyz"}), 1.0)
    op, free, m = unit((4, 3, 3), bcs={"xmin": "xyz"})
    with pytest.raises(ValueError, match="sigma"):
        gmg(op, free, -1.0)
    r = rcm(m)[0]
    with pytest.raises(ValueError, match="renumber"):
        gmg(stencil(r, UNIT), dof_mask(r, {"xmin": "xyz"}), 1.0)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="float64"):
            gmg(op, free, 1.0)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_study_smoke():
    """The GMG study on two small cases, without the large run."""
    from benchmarks import gmg_study

    cases = [
        ("box4", (4,) * 3, "clamped", None, None, "own"),
        ("plate8", (8, 8, 1), "clamped", None, None, "box4"),
    ]
    r = gmg_study.main(cases=cases, n_large=0, write=False)
    assert len(r["runs"]) == 2 and all(all(x["converged"]) for x in r["runs"])
