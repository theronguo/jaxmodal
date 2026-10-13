"""AMG below the hex8 level against SciPy Galerkin products, pyamg's own prolongator, a dense
NumPy V-cycle and PCG/LOBPCG iteration bounds."""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from helpers import MATERIALS, STEEL, checkerboard, cube, dense, distort, lam1, rel, rhs, sigma6
from pyamg import smoothed_aggregation_solver
from pyamg.aggregation import fit_candidates, standard_aggregation
from pyamg.strength import symmetric_strength_of_connection

from jaxmodal.eigen import lobpcg, reference_modes
from jaxmodal.linalg import pcg
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import a_p, ap_operator, assemble, dof_mask, stencil, unstructured
from jaxmodal.operators.constraints import APOperator, rigid_modes
from jaxmodal.precond import GMG, SingularShiftError, amg, pmg

AM = importlib.import_module("jaxmodal.precond.amg")
PM = importlib.import_module("jaxmodal.precond.pmg")

SIGMA = 0.37  # unit-material shift of the exactness tests
SLIDE = {"xmin": "xyz", "ymin": "y"}


def setup(mesh, bcs, max_coarse, sigma=SIGMA, mats=MATERIALS):
    """(op, free, A_P, hex8 Galerkin Ac, corner, B, SA levels, coarsest) as amg builds them."""
    free = bcs if isinstance(bcs, np.ndarray) else dof_mask(mesh, bcs)
    op = unstructured(mesh, mats)
    A = ap_operator(op, free, sigma)
    Ac, _, corner = PM._hex8_galerkin(A)
    fc = free.reshape(-1, 3)[corner].ravel()
    B = AM._nullspace(mesh.points[corner], fc)
    return op, free, A, Ac, corner, B, *AM._sa_levels(Ac, B, max_coarse)


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
    _, T, corner = PM._hex8_galerkin(ap_operator(unstructured(mesh, MATERIALS), free, SIGMA))
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


def cheb(A, Dinv, lmax, degree, ratio):
    """Dense Chebyshev smoother matrix, the recurrence of precond.chebyshev."""
    hi = 1.1 * lmax
    lo = hi / ratio
    theta, delta = (hi + lo) / 2, (hi - lo) / 2
    sig = theta / delta
    rho = 1 / sig
    R = np.eye(len(A))
    d = Dinv @ R / theta
    x = d
    for _ in range(degree - 1):
        R = R - A @ d
        rho_new = 1 / (2 * sig - rho)
        d = rho_new * rho * d + 2 * rho_new / delta * (Dinv @ R)
        x, rho = x + d, rho_new
    return x


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


def vcycle(levels, Ps, Ci, m, U):
    def go(b, lvl):
        if lvl == len(levels):
            return Ci @ b
        A, S = levels[lvl]
        x = S @ b
        x = x + Ps[lvl] @ go(Ps[lvl].T @ (b - A @ x), lvl + 1)
        return x + S @ (b - A @ x)

    return m[:, None] * go(m[:, None] * U, 0)


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
    monkeypatch.setattr(AM, "_sa_levels", lambda *a: ([], bad))
    with pytest.raises(ValueError, match="coarsest factor"):
        amg(op, free, SIGMA, mesh.points)
    level = (sp.csr_array(-sp.eye(192)), sp.csr_array(np.ones((192, 6))))
    monkeypatch.setattr(AM, "_sa_levels", lambda *a: ([level], sp.csr_array(sp.eye(6))))
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
