"""AMG below the hex8 level against SciPy Galerkin products, pyamg's own prolongator, a dense
NumPy V-cycle and PCG/LOBPCG iteration bounds."""

import dataclasses
import functools
import gc
import importlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from helpers import (
    MATERIALS,
    STEEL,
    R,
    cheb,
    checkerboard,
    cube,
    dense,
    distort,
    lam1,
    mixed_mesh,
    owned_bytes,
    rel,
    rhs,
    sigma6,
    step_bytes,
    vcycle,
)
from pyamg import smoothed_aggregation_solver
from pyamg.aggregation import fit_candidates, standard_aggregation
from pyamg.strength import symmetric_strength_of_connection

from jaxmodal.eigen import lobpcg, reference_modes
from jaxmodal.linalg import pcg
from jaxmodal.memory import (
    AMG_BUILD,
    AMG_STEP,
    amg_memory,
    lobpcg_memory,
    solve_modes_memory,
    unstructured_memory,
)
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import a_p, ap_operator, assemble, dof_mask, stencil, unstructured
from jaxmodal.operators.constraints import APOperator, rigid_modes
from jaxmodal.precond import GMG, SingularShiftError, amg, jacobi, pmg

AM = importlib.import_module("jaxmodal.precond.amg")
PM = importlib.import_module("jaxmodal.precond.pmg")

SIGMA = 0.37  # unit-material shift of the exactness tests
SLIDE = {"xmin": "xyz", "ymin": "y"}


def setup(mesh, bcs, max_coarse, sigma=SIGMA, mats=MATERIALS):
    """(op, free, A_P, hex8 Galerkin Ac, corner, B, SA levels, coarsest) as amg builds them."""
    free = bcs if isinstance(bcs, np.ndarray) else dof_mask(mesh, bcs)
    op = unstructured(mesh, mats)
    A = ap_operator(op, free, sigma)
    Ac, pattern = PM._hex8_galerkin(A)
    corner = pattern.corner
    fc = free.reshape(-1, 3)[corner].ravel()
    B = AM._nullspace(mesh.points[corner], fc)
    return op, free, A, Ac, corner, B, *AM._sa_levels(Ac, B, max_coarse)[:2]


def nonzero(A):
    A = sp.csr_array(A, copy=True)
    A.eliminate_zeros()
    return A


# ---------------------------------------------------------------- 1-3: setup


def test_setup_scipy():
    """Test 1: every level's coarse matrix is P^T A_z P (SciPy), symmetric, P has no zero
    column; the device levels and transfers equal the SciPy matrices."""
    mesh = box_mesh((6,) * 3, type="hex20")
    op, free, _, Ac, _, _, levels, coarsest = setup(mesh, SLIDE, 100)
    assert len(levels) == 2
    nxt = [lv[0] for lv in levels[1:]] + [coarsest]
    P_dev = amg(op, free, SIGMA, mesh.points, max_coarse=100)
    assert len(P_dev.levels) == 3
    rng = np.random.default_rng(0)
    for i, ((Al, P), Anext) in enumerate(zip(levels, nxt, strict=True)):
        Az = nonzero(Al)
        assert rel(Anext.toarray(), (P.T @ Az @ P).toarray()) <= 1e-12
        assert (Anext != Anext.T).nnz == 0
        assert (abs(P).sum(axis=0) > 0).all()
        X = rng.standard_normal((Al.shape[0], 3))
        assert rel(P_dev.levels[i + 1][0](jnp.asarray(X)), Al @ X) <= 1e-12
        T = P_dev.transfers[i + 1]
        Y = rng.standard_normal((P.shape[1], 2))
        assert rel(T(jnp.asarray(Y)), P @ Y) <= 1e-12
        assert rel(T.restrict(jnp.asarray(X)), P.T @ X) <= 1e-12
    L = np.asarray(P_dev.coarse.L)
    assert rel(L @ L.T, coarsest.toarray()) <= 1e-12


def test_setup_deterministic():
    """Test 1: two setups are bitwise identical and the global np.random state is unchanged."""
    mesh = box_mesh((6,) * 3, type="hex20")
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = setup(mesh, SLIDE, 100)
    assert (np.random.get_state()[1] == before).all()
    b = setup(mesh, SLIDE, 100)
    for (A1, P1), (A2, P2) in zip(a[6], b[6], strict=True):
        assert (P1 != P2).nnz == 0 and (A1 != A2).nnz == 0
    assert (a[7] != b[7]).nnz == 0


def test_nullspace():
    """Test 1: B spans the corner restriction of rigid_modes on a straight hex20 box; rows of
    constrained DOFs are exactly 0."""
    mesh = box_mesh((3, 2, 2), (0.5, 0.7, 1.1), type="hex20")
    free = dof_mask(mesh, SLIDE)
    corner = PM._hex8_galerkin(ap_operator(unstructured(mesh, MATERIALS), free, SIGMA))[1].corner
    fc = free.reshape(-1, 3)[corner].ravel()
    B = AM._nullspace(mesh.points[corner], fc)
    dof = (3 * corner[:, None] + np.arange(3)).ravel()
    Rm = rigid_modes(mesh)[dof] * fc[:, None]
    assert (B[~fc] == 0).all()
    assert np.linalg.matrix_rank(B) == np.linalg.matrix_rank(Rm) == 6
    assert sla.subspace_angles(B, Rm).max() <= 1e-10


@pytest.mark.parametrize("t", ["hex20", "hex8"])
def test_pyamg_prolongator(t):
    """Test 2: the first-level P equals pyamg's smoothed_aggregation_solver with node blocks,
    symmetric strength theta = 0, block-weighted Jacobi omega = 4/3, no candidate improvement."""
    mesh = box_mesh((12,) * 3, type=t)
    *_, Ac, _, B, levels, _ = setup(mesh, {"xmin": "xyz"}, 1000)
    Az = nonzero(Ac)
    np.random.seed(0)
    ml = smoothed_aggregation_solver(
        sp.bsr_matrix(Az, blocksize=(3, 3)),
        B=B,
        strength=("symmetric", {"theta": 0.0}),
        improve_candidates=None,
        smooth=("jacobi", {"omega": 4 / 3, "weighting": "block"}),
        max_levels=2,
        keep=True,
    )
    P, Pr = levels[0][1], sp.csr_array(ml.levels[0].P)
    assert P.shape == Pr.shape
    assert spla.norm(P - Pr) <= 1e-12 * spla.norm(Pr)


@pytest.mark.parametrize("t", ["hex20", "hex8"])
def test_zero_columns(t):
    """Test 3: a random 30%-free mask gives zero tentative columns on the first level; they are
    dropped, every coarse matrix is SPD, and PCG converges."""
    mesh = box_mesh((6,) * 3, type=t)
    free = np.random.default_rng(0).random(3 * len(mesh.points)) < 0.3
    op, _, A, Ac, _, B, levels, coarsest = setup(mesh, free, 100)
    Az = nonzero(Ac)
    Ab = sp.bsr_array(Az, blocksize=(3, 3))
    agg = sp.csr_array(standard_aggregation(symmetric_strength_of_connection(Ab, 0.0))[0])
    Tt = sp.csc_array(fit_candidates(agg, B)[0])
    zero = int((np.asarray(Tt.multiply(Tt).sum(axis=0)).ravel() == 0).sum())
    assert zero > 0 and levels[0][1].shape[1] == Tt.shape[1] - zero
    for M in [lv[0] for lv in levels[1:]] + [coarsest]:
        assert np.linalg.eigvalsh(M.toarray()).min() > 0
    P = amg(op, free, SIGMA, mesh.points, max_coarse=100)
    _, info = pcg(A, jnp.asarray(rhs(free, 4)), P, 1e-8, 200)
    assert np.asarray(info["converged"]).all()


def test_straddling():
    """Test 3 (roadmap): first-level aggregates mix constrained and free corner nodes; setup
    succeeds and PCG converges."""
    mesh = box_mesh((6,) * 3, type="hex20")
    op, free, A, Ac, corner, *_ = setup(mesh, SLIDE, 100)
    C = symmetric_strength_of_connection(sp.bsr_array(nonzero(Ac), blocksize=(3, 3)), 0.0)
    agg = sp.csr_array(standard_aggregation(C)[0])
    node = np.repeat(np.arange(agg.shape[0]), np.diff(agg.indptr))  # aggregated nodes
    fc = free.reshape(-1, 3)[corner]
    fixed, loose = (
        np.bincount(agg.indices, f[node], agg.shape[1]) > 0 for f in (~fc.any(1), fc.all(1))
    )
    assert (fixed & loose).any()
    P = amg(op, free, SIGMA, mesh.points, max_coarse=100)
    _, info = pcg(A, jnp.asarray(rhs(free, 4)), P, 1e-8, 200)
    assert np.asarray(info["converged"]).all()


# ---------------------------------------------------------------- 4: the cycle


def block_inv(A, b, m):
    """Dense block Jacobi: inverse b x b diagonal blocks, constrained rows/columns masked."""
    n = len(A) // b
    D = np.zeros_like(A)
    for i in range(n):
        s = slice(b * i, b * i + b)
        mi = m[s]
        Bk = A[s, s] * np.outer(mi, mi) + np.diag(1 - mi)
        D[s, s] = np.linalg.inv(Bk) * mi[:, None]
    return D


@pytest.mark.parametrize("t", ["hex20", "hex8"])
def test_vcycle_dense(t):
    """Test 4: amg(degree=3, ratio=20) = the NumPy V-cycle from SciPy a_p and the SciPy SA
    levels, test-built smoothers (the device's seeded lmax) and the dense coarsest inverse; V is
    symmetric, SPD on free DOFs, and constrained outputs are exactly 0."""
    if t == "hex20":
        mesh, bcs, mc = box_mesh((3, 2, 2), type="hex20"), {"xmin": "xyz"}, 10
    else:
        mesh, bcs, mc = box_mesh((4,) * 3), {"xmin": "x", "ymin": "y", "zmin": "z"}, 20
    op, free, Ap, Ac, _, _, levels, coarsest = setup(mesh, bcs, mc)
    P = amg(op, free, SIGMA, mesh.points, degree=3, ratio=20.0, max_coarse=mc)
    assert isinstance(P.levels[0][0], APOperator)  # matrix-free fine level (also all-hex8)
    K, M = assemble(mesh, MATERIALS)
    A = a_p(K + SIGMA * M, free, float(Ap.s)).toarray()
    m = free.astype(float)
    mats = [A] + [lv[0].toarray() for lv in levels[1 if t == "hex8" else 0 :]]
    masks = [m] + [np.ones(len(x)) for x in mats[1:]]
    blocks = [3, 3] + [1] * (len(mats) - 2) if t == "hex20" else [3] + [1] * (len(mats) - 1)
    lmax = [float(S.lmax) for _, S in P.levels]
    assert len(lmax) == len(mats)
    lev = [
        (Am, cheb(Am, block_inv(Am, b, mk), lm, 3, 20.0))
        for Am, b, mk, lm in zip(mats, blocks, masks, lmax, strict=True)
    ]
    Ps = [lv[1].toarray() for lv in levels]
    if t == "hex20":
        Ps = [dense(P.transfers[0], Ac.shape[0])] + Ps
    else:
        assert (Ps[0][~free] == 0).all()  # constrained rows of P_0
    assert len(Ps) == len(lev)
    U = np.random.default_rng(6).standard_normal((op.n_dof, 3))
    ref = vcycle(lev, Ps, np.linalg.inv(coarsest.toarray()), m, U)
    assert rel(P(jnp.asarray(U)), ref) <= 1e-10
    V = dense(P, op.n_dof)
    u, v = U[:, :1], U[:, 1:2]
    assert abs((u.T @ V @ v - v.T @ V @ u).item()) <= 1e-12 * abs((u.T @ V @ v).item())
    Vf = V[np.ix_(free, free)]
    assert np.linalg.eigvalsh((Vf + Vf.T) / 2).min() > 0
    assert (V[~free] == 0).all()


def test_no_amg_level():
    """Test 4c: below max_coarse, hex20 is two-level p-MG with an exact coarse Cholesky (equal
    to pmg) and all-hex8 is a direct solve (levels == (), PCG in 1 iteration in exact
    arithmetic; 2 allowed)."""
    mesh = box_mesh((3,) * 3, type="hex20")
    free = dof_mask(mesh, {"xmin": "xyz"})
    op = unstructured(mesh, MATERIALS)
    P = amg(op, free, SIGMA, mesh.points)
    assert len(P.levels) == 1 and P.coarse.L.shape == (192, 192)
    U = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 2)))
    assert rel(P(U), pmg(op, free, SIGMA)(U)) <= 1e-10
    mesh = box_mesh((3,) * 3)
    free = dof_mask(mesh, {"xmin": "xyz"})
    op = unstructured(mesh, MATERIALS)
    P = amg(op, free, SIGMA, mesh.points)
    assert P.levels == () and P.transfers == ()
    _, info = pcg(ap_operator(op, free, SIGMA), jnp.asarray(rhs(free, 2)), P, 1e-8, 20)
    assert np.asarray(info["converged"]).all() and np.asarray(info["iterations"]).max() <= 2


# ---------------------------------------------------------------- 5-7: iterations (steel)


def iterations(mesh, bcs, sigma=None, op=None, max_coarse=100):
    free = dof_mask(mesh, bcs)
    op = unstructured(mesh, STEEL) if op is None else op
    sigma = lam1(mesh, free) if sigma is None else sigma
    A, b = ap_operator(op, free, sigma), jnp.asarray(rhs(free, 4))
    P = amg(op, free, sigma, mesh.points, max_coarse=max_coarse)
    assert len(P.levels) > (0 if mesh.blocks.keys() == {"hex8"} else 1)  # an AMG level
    _, info = pcg(A, b, P, 1e-8, 1000)
    assert np.asarray(info["converged"]).all()
    return int(np.asarray(info["iterations"]).max())


FIRST = {"hex20": 3, "hex27": 3, "hex8": 6}
MEASURED = {"hex20": 15, "hex27": 13, "hex8": 13}  # device cycle, every size


@functools.cache
def first(t):
    return iterations(cube(t, FIRST[t]), {"xmin": "xyz"}, sigma6(t))


@pytest.mark.parametrize(
    "t, n",
    [("hex20", 6), ("hex20", 12), pytest.param("hex20", 24, marks=pytest.mark.slow)]
    + [("hex27", 6), ("hex27", 12), ("hex8", 12), ("hex8", 24)],
)
def test_h_independence(t, n):
    """Test 5 (roadmap): iterations grow <= 20% over the h-sequence at a fixed sigma, each
    <= 1.5x the measured count (hex20 15 / 14 / 14 / 14, hex27 and hex8 13)."""
    its = iterations(cube(t, n), {"xmin": "xyz"}, sigma6(t))
    assert first(t) <= 1.5 * MEASURED[t]
    assert its <= 1.2 * first(t) and its <= 1.5 * MEASURED[t]


ROBUST = {"beam": 16, "free": 14, "curved": 14, "nodal": 14, "contrast": 20, "plate": 36}


@pytest.mark.parametrize("case", list(ROBUST))
def test_robust(case):
    """Test 6: <= 1.5x the measured count at each case's own lambda_1, max_coarse = 100; the
    single-layer plate (a documented flat-element limitation) as a regression bound."""
    rng = np.random.default_rng(8)
    t, clamp = "hex20", {"xmin": "xyz"}
    if case == "beam":
        its = iterations(box_mesh((20, 2, 2), (0.05,) * 3, type=t), clamp)
    elif case == "free":
        its = iterations(cube(t, 6), {})
    elif case == "plate":
        its = iterations(box_mesh((16, 16, 1), (1 / 16,) * 3, type=t), clamp)
    elif case == "curved":
        its = iterations(distort(box_mesh((6,) * 3, type=t), True, rng), clamp)
    elif case == "nodal":
        m = cube(t, 6)
        nodal = 7850.0 / 6**3 * (rng.random(len(m.points)) < 0.5)
        op = unstructured(m, STEEL, nodal_mass=nodal)
        its = iterations(m, clamp, lam1(m, dof_mask(m, clamp), nodal_mass=nodal), op)
    else:
        m = cube(t, 4)
        op = checkerboard(m, unstructured(m, STEEL))
        its = iterations(m, clamp, lam1(m, dof_mask(m, clamp), s_K=np.asarray(op.s_K)), op)
    assert its <= 1.5 * ROBUST[case]


@pytest.mark.parametrize("t", ["hex20", "hex8"])
def test_lobpcg(t):
    """Test 7: LOBPCG with AMG on a clamped 6x4x3 box matches reference_modes in <= 40
    iterations."""
    m = box_mesh((6, 4, 3), (1 / 6,) * 3, type=t)
    op, free = unstructured(m, STEEL), dof_mask(m, {"xmin": "xyz"})
    K, M = assemble(m, STEEL)
    lr = reference_modes(K, M, 12, free, points=m.points)[0]
    P = amg(op, free, lr[0], m.points, max_coarse=50)
    assert len(P.levels) >= (1 if t == "hex8" else 2)
    lam, V, info = lobpcg(op, free, 6, tol=1e-8, precond=P)
    assert info["converged"] and info["iterations"] <= 40
    assert np.abs(np.asarray(lam) / lr[:6] - 1).max() <= 1e-8
    V, lam = np.asarray(V), np.asarray(lam)
    assert (V[~free] == 0).all()
    KV, MV = K @ V, M @ V
    assert (
        np.linalg.norm((KV - MV * lam)[free], axis=0) / np.linalg.norm(KV[free], axis=0)
    ).max() <= 1e-6


# ---------------------------------------------------------------- 13b: reuse, sigma = 0


def density_case(t, n, seed=0):
    """(mesh, free, op_rho, op_rho2): clamped plus a sliding face, s_K = rho^3, s_M = rho with
    rho in [0.5, 1] and a 20% perturbation rho2 (seeded)."""
    mesh = box_mesh((n,) * 3, type=t)
    free = dof_mask(mesh, {"xmin": "xyz", "ymin": "y"})
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.5, 1.0, mesh.n_el)
    rho2 = rho * rng.uniform(0.8, 1.2, mesh.n_el)
    op = unstructured(mesh, MATERIALS)
    ops = [op.scaled(s_K=jnp.asarray(r**3), s_M=jnp.asarray(r)) for r in (rho, rho2)]
    return mesh, free, *ops


def pcg_its(op, free, sigma, P):
    _, info = pcg(ap_operator(op, free, sigma), jnp.asarray(rhs(free, 4)), P, 1e-8, 500)
    assert np.asarray(info["converged"]).all()
    return int(np.asarray(info["iterations"]).max())


@pytest.mark.parametrize("t, n", [("hex20", 6), ("hex8", 8)])
def test_reuse(t, n):
    """13b test 2 (roadmap): aggregates from rho reused on rho2 give the fresh rho2 V-cycle to
    1e-12 and <= 1.2x its PCG iterations; sigma = 0 aggregates reused at sigma > 0 equal a
    fresh sigma > 0 setup."""
    mesh, free, op, op2 = density_case(t, n)
    _, agg = amg(op, free, SIGMA, mesh.points, max_coarse=100, return_aggregates=True)
    assert len(agg.tents) >= 1
    fresh = amg(op2, free, SIGMA, mesh.points, max_coarse=100)
    reused = amg(op2, free, SIGMA, mesh.points, aggregates=agg)
    U = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 3)))
    assert rel(reused(U), fresh(U)) <= 1e-12
    assert pcg_its(op2, free, SIGMA, reused) <= 1.2 * pcg_its(op2, free, SIGMA, fresh)
    _, agg0 = amg(op, free, 0.0, mesh.points, max_coarse=100, nonsingular=True,
                  return_aggregates=True)  # fmt: skip
    reused = amg(op, free, SIGMA, mesh.points, aggregates=agg0)
    assert rel(reused(U), amg(op, free, SIGMA, mesh.points, max_coarse=100)(U)) <= 1e-12


def test_reuse_moved_points():
    """13b test 2: reuse after a 2% node perturbation (same connectivity) keeps the old
    near-null space and converges within 1.2x a fresh setup."""
    mesh, free, op, _ = density_case("hex20", 6)
    _, agg = amg(op, free, SIGMA, mesh.points, max_coarse=100, return_aggregates=True)
    pts = mesh.points + np.random.default_rng(2).uniform(-0.02, 0.02, mesh.points.shape)
    moved = dataclasses.replace(mesh, points=pts)
    op2 = unstructured(moved, MATERIALS).scaled(s_K=op.s_K, s_M=op.s_M)
    fresh = amg(op2, free, SIGMA, moved.points, max_coarse=100)
    reused = amg(op2, free, SIGMA, moved.points, aggregates=agg)
    assert pcg_its(op2, free, SIGMA, reused) <= 1.2 * pcg_its(op2, free, SIGMA, fresh)


def test_reuse_skips_topology(monkeypatch):
    """13b test 2: reuse calls none of the pattern, tables, strength, aggregation or fit."""
    mesh, free, op, op2 = density_case("hex20", 6)
    _, agg = amg(op, free, SIGMA, mesh.points, max_coarse=100, return_aggregates=True)

    def boom(*a, **kw):
        raise AssertionError("topology recomputed")

    for mod, name in [(PM, "_pattern"), (PM, "_tables"), (AM, "symmetric_strength_of_connection"),
                      (AM, "standard_aggregation"), (AM, "fit_candidates")]:  # fmt: skip
        monkeypatch.setattr(mod, name, boom)
    amg(op2, free, SIGMA, mesh.points, aggregates=agg)


def test_reuse_errors():
    """13b test 2: another free mask or another connectivity (renumbered elements, same nodes)
    raises."""
    mesh, free, op, _ = density_case("hex20", 4)
    _, agg = amg(op, free, SIGMA, mesh.points, max_coarse=100, return_aggregates=True)
    with pytest.raises(ValueError, match="free mask"):
        amg(op, dof_mask(mesh, {"xmin": "xyz"}), SIGMA, mesh.points, aggregates=agg)
    perm = box_mesh((4,) * 3, type="hex20")
    perm.blocks["hex20"] = perm.blocks["hex20"][::-1].copy()
    with pytest.raises(ValueError, match="connectivity"):
        amg(unstructured(perm, MATERIALS), free, SIGMA, mesh.points, aggregates=agg)


@pytest.mark.parametrize("t", ["hex20", "hex8"])
def test_sigma_zero(t):
    """13b test 3: sigma = 0 needs nonsingular=True; with it, PCG on A_P(0) converges on a
    clamped box and the V-cycle is finite; nonsingular with a zero s_K raises."""
    mesh = box_mesh((6,) * 3, type=t)
    free = dof_mask(mesh, {"xmin": "xyz"})
    op = unstructured(mesh, MATERIALS)
    with pytest.raises(SingularShiftError):
        amg(op, free, 0.0, mesh.points)
    P = amg(op, free, 0.0, mesh.points, max_coarse=100, nonsingular=True)
    assert len(P.levels) > (0 if t == "hex8" else 1)
    U = jnp.asarray(rhs(free, 2))
    assert np.isfinite(np.asarray(P(U))).all()
    pcg_its(op, free, 0.0, P)
    s_K = np.ones(mesh.n_el)
    s_K[3] = 0.0
    with pytest.raises(ValueError, match="s_K > 0"):
        amg(op.scaled(s_K=jnp.asarray(s_K)), free, 0.0, mesh.points, nonsingular=True)


def sparse_count():
    return sum(isinstance(o, (sp.sparray, sp.spmatrix)) for o in gc.get_objects())


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_no_survivors(dtype):
    """13b test 11: after a warm-up build, a second amg leaves no SciPy matrix and no device
    array beyond the returned pytree's leaves (PR 18c: also no fp64 copy of an fp32 build)."""
    mesh = box_mesh((6,) * 3, type="hex20")
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, SLIDE)
    amg(op, free, SIGMA, mesh.points, max_coarse=100, dtype=dtype)
    gc.collect()
    before, n_sp = {id(a) for a in jax.live_arrays()}, sparse_count()
    P = amg(op, free, SIGMA, mesh.points, max_coarse=100, dtype=dtype)
    gc.collect()
    leaves = {id(x) for x in jax.tree_util.tree_leaves(P)}
    assert {id(a) for a in jax.live_arrays()} - before - leaves == set()
    assert sparse_count() == n_sp


# ---------------------------------------------------------------- 18c: fp32 V-cycle


def fp32_box(t, n):
    mesh = box_mesh((n,) * 3, type=t)
    return unstructured(mesh, MATERIALS), dof_mask(mesh, SLIDE), SIGMA, mesh.points, 1e-5


def fp32_mixed():
    mesh = mixed_mesh(np.random.default_rng(0))
    free = np.ones(3 * len(mesh.points), bool)
    free[:12] = False  # no node sets: 4 nodes pinned
    op = unstructured(mesh, MATERIALS, orientations={1: R})
    return op, free, SIGMA, mesh.points, 1e-5


def fp32_shell(name):
    import test_shell_amg as TS

    mesh = (TS.block(8) if name == "block" else TS.plate(32, 10))[0]
    tol = 1e-5 if name == "block" else 2e-5
    return unstructured(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"}), 1.0, mesh.points, tol


# (op, free, sigma, points, tol) of PR 18c test 2
FP32_CASES = {
    "hex8": lambda: fp32_box("hex8", 8),
    "hex20": lambda: fp32_box("hex20", 6),
    "hex27": lambda: fp32_box("hex27", 4),
    "mixed": fp32_mixed,
    "block": lambda: fp32_shell("block"),
    "plate": lambda: fp32_shell("plate"),
}


@pytest.mark.parametrize("name", list(FP32_CASES))
def test_fp32_vcycle(name):
    """PR 18c test 2 (max_coarse 100: 2-3 levels): the fp32 cycle returns fp64 within tol of
    the fp64 cycle (relative Frobenius) and is symmetric to tol / 10 (2e-5 on the thin plate,
    energy-normalized); its float leaves are fp32 except the fine A_P, the p-level transfer,
    free and the coarsest factor; the int leaves equal fp64's; the Schwarz packed inverses
    are the fp64 ones cast."""
    op, free, sigma, points, tol = FP32_CASES[name]()
    P64 = amg(op, free, sigma, points, max_coarse=100)
    P32 = amg(op, free, sigma, points, max_coarse=100, dtype="float32")
    assert P32.dtype == "float32" and len(P32.levels) >= 2
    U = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, 4)) * free[:, None])
    Y64, Y32 = P64(U), P32(U)
    assert Y32.dtype == jnp.float64
    assert np.linalg.norm(Y32 - Y64) <= tol * np.linalg.norm(Y64)
    G = np.asarray(U.T @ Y32)
    d = np.sqrt(np.diag(G))
    assert (np.abs(G - G.T) / np.outer(d, d)).max() <= (2e-5 if name == "plate" else tol / 10)
    shell = name in ("block", "plate")
    keep = [P32.levels[0][0], P32.levels[0][1].A, P32.free, P32.coarse.L]
    keep += [] if shell or name == "hex8" else [P32.transfers[0]]
    keep = jax.tree_util.tree_leaves(keep)
    assert all(
        a.dtype == jnp.float64 for a in keep if isinstance(a, jax.Array) and a.dtype.kind == "f"
    )
    keep = {id(a) for a in keep}
    f64 = [a for a in jax.tree_util.tree_leaves(P32) if isinstance(a, jax.Array)
           and a.dtype == jnp.float64]  # fmt: skip
    assert all(id(a) in keep for a in f64)
    ints = [[a for a in jax.tree_util.tree_leaves(P) if isinstance(a, jax.Array)
             and a.dtype.kind in "iub"] for P in (P64, P32)]  # fmt: skip
    assert len(ints[0]) == len(ints[1]) and all(map(np.array_equal, *ints))
    if shell:
        for (_, S64), (_, S32) in zip(P64.levels, P32.levels, strict=True):
            for a, b in zip(S64.precond.inv, S32.precond.inv, strict=True):
                assert b.dtype == jnp.float32 and np.array_equal(a.astype(jnp.float32), b)


def fp32_pcg(op, free, sigma, points, tol):
    """PR 18c test 3: PCG on A_P with the fp64 and fp32 V-cycles (4 masked random columns): the
    max iterations per dtype; both converge and the fp32 true residual is <= tol."""
    A = ap_operator(op, free, sigma)
    b = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 4)) * free[:, None])
    its = []
    for dtype in ("float64", "float32"):
        x, info = pcg(A, b, amg(op, free, sigma, points, dtype=dtype), tol, 1000)
        assert np.asarray(info["converged"]).all(), dtype
        its.append(int(np.asarray(info["iterations"]).max()))
    res = np.linalg.norm(b - A(x), axis=0) / np.linalg.norm(b, axis=0)
    assert res.max() <= tol, res
    return its


def test_fp32_pcg():
    """PR 18c test 3 (roadmap): hex20 12^3, xmin clamped, sigma 3e5, tol 1e-8: the fp32
    iterations within +-1 of fp64."""
    mesh = box_mesh((12,) * 3, (1 / 12,) * 3, type="hex20")
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"})
    i64, i32 = fp32_pcg(op, free, 3e5, mesh.points, 1e-8)
    assert abs(i32 - i64) <= 1, (i64, i32)


def test_fp32_not_finite():
    """PR 18c test 2: an operator scaled by 1e-40 / max|diag| has finite fp64 inverses above
    the fp32 maximum, so the fp32 build raises: on the block from _invert's check, on hex8 8^3
    at the default max_coarse (no assembled level) from the check after the cast."""
    import test_shell_amg as TS

    cases = [(TS.block(8)[0], STEEL, 1.0, "Schwarz patch inverses"),
             (box_mesh((8,) * 3), MATERIALS, SIGMA, "AMG level smoother")]  # fmt: skip
    for mesh, mats, sigma, match in cases:
        op = unstructured(mesh, mats)
        free = dof_mask(mesh, {"xmin": "xyz"})
        c = 1e-40 / float(jnp.abs(ap_operator(op, free, sigma).op.diag_K()).max())
        op = op.scaled(s_K=c * op.s_K, s_M=c * op.s_M)
        P64 = amg(op, free, sigma, mesh.points)
        assert match != "AMG level smoother" or len(P64.levels) == 1  # no assembled level
        inv = jax.tree_util.tree_leaves(P64.levels[0][1].precond)
        big = max(float(jnp.abs(x).max()) for x in inv if x.dtype == jnp.float64)
        assert np.isfinite(big) and big > 3.4e38
        with pytest.raises(ValueError, match=match):
            amg(op, free, sigma, mesh.points, dtype="float32")


# ---------------------------------------------------------------- 13b: amg_memory, benchmarks

LOB = importlib.import_module("jaxmodal.eigen.lobpcg")
MEM = importlib.import_module("jaxmodal.memory")


STORAGE_CASES = {
    "hex20 12^3": (lambda: box_mesh((12,) * 3, type="hex20"), {"xmin": "xyz"}, True),
    "hex27 6^3": (lambda: box_mesh((6,) * 3, type="hex27"), {"xmin": "xyz"}, False),
    "hex8 16^3": (lambda: box_mesh((16,) * 3), {"xmin": "xyz"}, True),
    "sliding hex20 8^3": (lambda: box_mesh((8,) * 3, type="hex20"), SLIDE, False),
    "distorted hex20 8^3": (
        lambda: distort(box_mesh((8,) * 3, type="hex20"), True, np.random.default_rng(4)),
        {"xmin": "xyz"},
        False,
    ),
}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("name", list(STORAGE_CASES))
def test_amg_memory_storage(name, dtype):
    """13b test 7: storage >= the preconditioner-owned device bytes, the coarse term >= the
    coarsest factor; at hex20 12^3 and hex8 16^3 (AMG part near the per-corner envelope) the
    part that scales with n within 1.3x; elsewhere only the upper bound (box calibration).
    PR 18c test 5: fp32 too; the exact part (free mask, fine node-block inverses and mask) is
    120 / 72 B per node."""
    make, bcs, tight = STORAGE_CASES[name]
    mesh = make()
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, bcs)
    P = amg(op, free, SIGMA, mesh.points, dtype=dtype)
    got, est = owned_bytes(P, op), amg_memory(mesh, 15, dtype=dtype)["storage"]
    exact = sum(x.nbytes for x in jax.tree_util.tree_leaves((P.free, P.levels[0][1].precond)))
    assert exact == (72 if dtype == "float32" else 120) * len(mesh.points)
    n_c = 3 * len(np.unique(np.concatenate([c[:, :8].ravel() for c in mesh.blocks.values()])))
    coarse = 8 * min(n_c, 1000) ** 2
    assert got <= est and P.coarse.L.nbytes <= coarse, (got, est)
    if tight:  # the n-scaling part alone covers the non-coarse bytes, and within 1.3x
        assert est - coarse >= got - P.coarse.L.nbytes
        assert est - coarse <= 1.3 * (got - P.coarse.L.nbytes), (est - coarse) / (
            got - P.coarse.L.nbytes
        )


def test_amg_memory_build_mixed():
    """13b test 7: a mixed mesh's build term uses the largest per-type constant."""
    mesh = mixed_mesh(np.random.default_rng(0))
    n_c = 3 * len(np.unique(np.concatenate([c[:, :8].ravel() for c in mesh.blocks.values()])))
    beta, alpha = AMG_BUILD
    assert amg_memory(mesh, 15)["build"] == int(beta + alpha["hex20"] * n_c)


def step_case(t, n, dtype="float64"):
    mesh = box_mesh((n,) * 3, (1 / n,) * 3, type=t)
    free = dof_mask(mesh, {"xmin": "xyz"})
    op = unstructured(mesh, STEEL)
    return mesh, op, free, amg(op, free, 1e6, mesh.points, dtype=dtype)


def step_model(mesh, op, low, n_modes=10, dtype="float64", polish=True):
    """(lobpcg_memory's step with amg_memory, its block size); polish False drops the polish
    term (the step's max would hide the 9-block and low terms)."""
    k = LOB._block_size(n_modes, None, None)[2]
    pc = lambda w: {key: v for key, v in amg_memory(mesh, w, dtype=dtype).items()  # noqa: E731
                    if polish or key != "polish"}  # fmt: skip
    est = lobpcg_memory(
        lambda w: unstructured_memory(mesh, w), op.n_dof, n_modes, low_memory=low,
        precond_memory=pc,
    )["step"]  # fmt: skip
    return est, k


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("t, n", [("hex20", 8), ("hex20", 12), ("hex27", 6), ("hex8", 12)])
@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
def test_amg_step(t, n, low, dtype):
    """13b test 8: the LOBPCG step with AMG <= lobpcg_memory's step (amg_memory), measured at
    the block size lobpcg_memory derives from n_modes (PR 18c test 6: fp32 too, without the
    polish term, which is checked by test_hex_polish_fp32)."""
    mesh, op, _, P = step_case(t, n, dtype)
    est, k = step_model(mesh, op, low, dtype=dtype, polish=False)
    assert k == 15
    got = step_bytes(op, P, low, k)
    assert got <= est, (got, est)


@pytest.mark.slow
@pytest.mark.parametrize("t, n", [("hex20", 24), ("hex8", 48)])
def test_amg_step_tight(t, n):
    """13b test 8: at hex20 24^3 and hex8 48^3 (m = 15) the AMG step's extra over the Jacobi step
    is within 1 block below AMG_STEP for both steps (pins the constants)."""
    mesh, op, free, P = step_case(t, n)
    fam = "hex8" if t == "hex8" else "p"
    for low in (False, True):
        b = 8 * op.n_dof * 15
        extra = (step_bytes(op, P, low) - step_bytes(op, jacobi(op, free, 0.0), low)) / b
        a, c, K = AMG_STEP[(fam, "low" if low else "9-block")]
        model = a + c / 15 + K / op.n_dof
        assert model - 1 <= extra <= model, (low, extra, model)


def fp32_study():
    """benchmarks/results/amg_fp32_study.json and the study module."""
    import json

    from benchmarks import amg_fp32_study as F
    from benchmarks import harness

    return json.load(open(harness.RESULTS / "amg_fp32_study.json")), F


@pytest.mark.slow
def test_amg_step_tight32():
    """PR 18c test 6: at hex20 24^3, m = 25 (the fit's TIGHT point) the fp32 step terms at the
    lobpcg_memory level (9-block, low, polish) are within the recorded gates of the measured."""
    from benchmarks import shell_solve_study as SSS

    d, F = fp32_study()
    mesh, op, free, P = step_case("hex20", 24, "float32")
    r = dict(n_dof=op.n_dof, m=25, ws=unstructured_memory(mesh, 25)["workspace"])
    got = {"9-block": step_bytes(op, P, False, 25), "low": step_bytes(op, P, True, 25),
           "polish": SSS.polish(op, jnp.asarray(free), P, 25)}  # fmt: skip
    for key, g in got.items():
        est = F.total("p", MEM.AMG_STEP32[("p", key)], r, key)
        assert g <= est <= d["fit"]["gates"].get(f"p/{key}", F.GATE) * g, (key, est / g)


def test_fp32_constants_pinned():
    """PR 18c test 7: STORAGE32, AMG_STEP32 and SHELL_STEP32 are the recorded fits as stored
    (ratio32 and rounded), the recorded gate flags hold, and every recorded upper-gate ratio
    is <= its gate (1.3, or the logged widened gate <= 1.4)."""
    d, F = fp32_study()
    f = d["fit"]
    assert F.ratio32(d["storage"]) == pytest.approx(MEM.STORAGE32, rel=1e-12)
    assert f["storage32"] == pytest.approx(MEM.STORAGE32, rel=1e-12)
    for name, c in f["consts"].items():
        fam, key = name.split("/")
        stored = MEM.SHELL_STEP32[key] if fam == "shell" else MEM.AMG_STEP32[(fam, key)]
        assert F.rounded(fam, c["lp"]) == pytest.approx(stored, rel=1e-12), name
        assert c["stored"] == pytest.approx(stored, rel=1e-12), name
    assert len(f["consts"]) == len(MEM.AMG_STEP32) + len(MEM.SHELL_STEP32)
    assert all(f["gate_ok"].values()), f["gate_ok"]
    assert all(x >= 0 for x in f["margins"].values())  # blocks above every target
    for name, rows in f["rows"].items():
        assert all(r["ratio"] <= f["gates"].get(name, F.GATE) for r in rows), name
    assert all(F.GATE <= g <= 1.4 for g in f["gates"].values())


@pytest.mark.slow
def test_fp32_study_smoke():
    """PR 18c test 11: amg_fp32_study's accuracy, storage, steps (hex, and the shell steps the fit
    needs) and fit on tiny cases record their keys."""
    from benchmarks import amg_fp32_study as F

    (r,) = F.accuracy([("hex8 6^3", lambda: F.hex_case("hex8", 6), 3e5, (1e-8,))])
    assert {"1e-08", "vcycle_s"} <= set(r["float32"]) and r["float64"]["1e-08"]["iterations"]
    data = dict(storage=F.storage({"hex8": (6,)}, [("flat", 8)]),
                steps_hex=F.steps_hex({"hex8": (4, 8), "hex20": (2, 4)}, (6,),
                                      [("hex8", 8, (10,))]),
                steps_shell=F.steps_shell([("flat", 8), ("flat", 16)], (6,), (), ()))  # fmt: skip
    assert {"owned", "exact", "coarse", "fitted64"} <= set(data["storage"][0])
    assert {"9-block", "low", "polish", "jacobi_low", "ws"} <= set(data["steps_hex"][0])
    assert "polish64" in data["steps_hex"][-1] and data["steps_hex"][-1]["storage"]
    f = F.fit(data, tight=[("hex8", 8, 6)], report=[])
    assert set(f["storage32"]) == {"hex8", "shell"} and len(f["consts"]) == 9
    assert set(f["gate_ok"]) == {"storage", "steps", "upper"} and f["polish64_slack"]


@pytest.mark.peak
@pytest.mark.slow
def test_amg_build_peak():
    """13b test 12: amg_memory's build >= the whole amg's host peak and <= 1.3x it (fresh
    subprocesses, peaks above the built operator)."""
    from benchmarks import amg_study

    for r in amg_study.setup([("hex20", 12), ("hex20", 24), ("hex27", 12), ("hex8", 48)]):
        mesh = box_mesh((r["n"],) * 3, type=r["type"])
        peak, est = r["amg"]["peak_bytes"], amg_memory(mesh, 15)["build"]
        assert peak <= est <= 1.3 * peak, (r["type"], r["n"], peak, est)


@pytest.mark.slow
def test_benchmark_smokes():
    """13b test 9: amg_study.setup and simp at small sizes record times, peaks and counts."""
    from benchmarks import amg_study

    (r,) = amg_study.setup([("hex20", 4)])
    assert r["assemble"]["time_s"] > 0 and r["amg"]["peak_bytes"] > 0 and r["reuse"]["reuse_s"] > 0
    runs = amg_study.simp(8, max_coarse=(100,))
    assert len(runs) == 3
    for x in runs:
        assert all(x["gmg"]["converged"]) and all(x["amg max_coarse=100"]["converged"])


@pytest.mark.peak  # its time budget has a 10x margin
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="Mac acceptance")
def test_mac_acceptance():
    """13b test 10 (roadmap): hex20 44^3 (1.08M DOF), 20 modes, tol 1e-6, defaults (AMG): the
    cold run within 15 min, all converged, the larger of two warm peaks within 20% of
    solve_modes_memory."""
    from benchmarks import amg_study

    r = amg_study.acceptance(44)
    assert r["converged"] and r["time_s"] <= 900, r["time_s"]
    peak, est = max(r["peaks_bytes"]), r["estimate_bytes"]
    assert est >= 2e9 and abs(est - peak) <= 0.2 * peak, (peak, est)


@pytest.mark.peak  # its time budget has a 10x margin
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="Mac acceptance")
def test_mac_acceptance_fp32():
    """PR 18c test 8 (roadmap): test_mac_acceptance with precond_dtype "float32": converged, the
    cold run within 15 min, the larger warm peak within 20% of solve_modes_memory, which is
    below the fp64 estimate."""
    from benchmarks import amg_study

    r = amg_study.acceptance(44, dtype="float32")
    assert r["converged"] and r["time_s"] <= 900, r["time_s"]
    peak, est = max(r["peaks_bytes"]), r["estimate_bytes"]
    assert est >= 2e9 and abs(est - peak) <= 0.2 * peak, (peak, est)
    mesh = box_mesh((44,) * 3, (1 / 44,) * 3, type="hex20")
    assert est < solve_modes_memory(mesh, {"xmin": "xyz"}, 20, n_rigid=0)["total"]


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_acceptance_smoke(dtype):
    """13b test 10: amg_study.acceptance at 8^3 records the keys; the estimate is
    solve_modes_memory's (PR 18c: per precond_dtype)."""
    from benchmarks import amg_study

    r = amg_study.acceptance(8, n_modes=4, dtype=dtype)
    assert r["converged"] and r["iterations"] > 0 and len(r["peaks_bytes"]) == 2
    assert r["dtype"] == dtype
    mesh = box_mesh((8,) * 3, (1 / 8,) * 3, type="hex20")
    est = solve_modes_memory(mesh, {"xmin": "xyz"}, 4, precond_dtype=dtype)["total"]
    assert r["estimate_bytes"] == est


# ---------------------------------------------------------------- 8: errors, cap, leaves


def test_errors():
    """Test 8: a Stencil, a wrong points shape, sigma 0 / < 0 / NaN, x64 off."""
    v = voxel_mesh((2, 2, 2))
    with pytest.raises(TypeError, match="gmg"):
        amg(stencil(v, MATERIALS), dof_mask(v, {"xmin": "xyz"}), 1.0, v.points)
    mesh = box_mesh((2, 2, 2), type="hex20")
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, {"xmin": "xyz"})
    with pytest.raises(ValueError, match="points"):
        amg(op, free, 1.0, mesh.points[:-1])
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            amg(op, free, bad, mesh.points)
    with pytest.raises(SingularShiftError):
        amg(op, free, 0.0, mesh.points)
    with pytest.raises(ValueError, match="dtype must be"):  # PR 18c
        amg(op, free, 1.0, mesh.points, dtype="float16")
    with pytest.raises(ValueError, match="dtype must be"):
        amg_memory(mesh, 15, dtype="float16")
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="float64"):
            amg(op, free, 1.0, mesh.points)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_coarsest_cap(monkeypatch):
    """Test 8: singleton aggregates stall the coarsening (the first level only drops the 48
    clamped corner DOFs, the second keeps every DOF) and raise instead of a dense factor."""
    mesh = box_mesh((3,) * 3, type="hex20")
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, {"xmin": "xyz"})

    def single(C):
        return sp.csr_array(sp.eye(C.shape[0])), np.arange(C.shape[0])

    monkeypatch.setattr(AM, "standard_aggregation", single)
    with pytest.raises(ValueError, match="stalled at 144 DOF"):
        amg(op, free, SIGMA, mesh.points, max_coarse=10)


def test_not_finite(monkeypatch):
    """Test 8: a non-SPD level or coarsest matrix raises instead of returning NaNs."""
    mesh = box_mesh((3,) * 3, type="hex20")
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, {"xmin": "xyz"})
    bad = sp.csr_array(-sp.eye(6))
    monkeypatch.setattr(AM, "_sa_levels", lambda *a: ([], bad, ()))
    with pytest.raises(ValueError, match="coarsest factor"):
        amg(op, free, SIGMA, mesh.points)
    level = (sp.csr_array(-sp.eye(192)), sp.csr_array(np.ones((192, 6))))
    monkeypatch.setattr(AM, "_sa_levels", lambda *a: ([level], sp.csr_array(sp.eye(6)), ()))
    with pytest.raises(ValueError, match="smoother blocks"):
        amg(op, free, SIGMA, mesh.points)


def test_degenerate_and_leaves():
    """Test 8: max_coarse = 3 stops at the 6-DOF level; every leaf is a jax.Array (but the
    BlockELL padding statistic); amg and pmg are GMGs (one shared _vcycle_apply)."""
    mesh = box_mesh((3, 2, 2), type="hex20")
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, {"xmin": "xyz"})
    P = amg(op, free, SIGMA, mesh.points, max_coarse=3)
    assert [lv[0].n_dof for lv in P.levels[1:]] == [108, 12] and P.coarse.L.shape == (6, 6)
    leaves = jax.tree_util.tree_leaves_with_path(P)  # no host matrices; BlockELL.padding is a float
    assert all(
        isinstance(x, jax.Array) or jax.tree_util.keystr(k).endswith(".padding") for k, x in leaves
    )
    assert type(P) is GMG and type(pmg(op, free, SIGMA)) is GMG


def test_study_smoke():
    """The AMG study on three small cases, without the 1M-DOF cost run."""
    from benchmarks import amg_study

    cases = [c for c in amg_study.CASES if c[0] in ("hex20 6^3", "hex20 3^3", "hex8 6^3")]
    r = amg_study.main(cases, large=False, write=False)
    assert all(all(x["amg"]["converged"]) for x in r["runs"])
    assert all(all(x["pmg"]["converged"]) for x in r["runs"] if "pmg" in x)
