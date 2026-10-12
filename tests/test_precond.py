"""A_P operator, node blocks, Jacobi / block Jacobi, Lanczos lambda_max and Chebyshev against
SciPy and dense references."""

import dataclasses
import functools

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
from helpers import MATERIALS, R, box, mixed_mesh
from jax.tree_util import Partial

import jaxmodal.eigen
import jaxmodal.precond
from jaxmodal.linalg import block_ell, pcg
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import a_p, ap_operator, assemble, stencil, unstructured
from jaxmodal.precond import block_jacobi, chebyshev, jacobi, lmax_estimate
from jaxmodal.precond.chebyshev import SAFETY

ORIENT = {1: R}


def bc_mask(mesh):
    """xmin: xyz plus ymin: y (partially constrained nodes), from the coordinates."""
    p = mesh.points
    free = np.ones((len(p), 3), bool)
    free[np.isclose(p[:, 0], p[:, 0].min())] = False
    free[np.isclose(p[:, 1], p[:, 1].min()), 1] = False
    return free.ravel()


@functools.cache
def case(name):
    """(op, K, M, free): the operator and its assembled SciPy counterparts."""
    rng = np.random.default_rng(5)
    kw, s_K = {}, None
    if name in ("stored", "otf"):
        m = box_mesh((3, 2, 2), spacing=(0.5, 0.7, 0.6), type="hex20")
        m = dataclasses.replace(m, points=m.points + rng.uniform(-0.03, 0.03, m.points.shape))
        kw = dict(mass="consistent" if name == "stored" else "lumped")
        kw["nodal_mass"] = rng.uniform(0, 1, len(m.points)) * (
            rng.uniform(size=len(m.points)) < 0.5
        )
        s_K = rng.uniform(0.1, 2, m.n_el)
    elif name == "mixed":
        m = mixed_mesh(rng)
    elif name == "hex27":
        m = box_mesh((2, 2, 2), spacing=(0.5, 0.5, 0.5), type="hex27")
    else:  # masked voxel grid, 2 phases, scaled s_K
        mask = np.ones((4, 3, 3), bool)
        mask[2:, 1:, 2] = mask[0, 2, :2] = False
        m = voxel_mesh((4, 3, 3), spacing=(0.5, 1.0, 0.75), mask=mask)
        m = dataclasses.replace(m, material=rng.integers(0, 2, m.n_el))
        s_K = rng.uniform(0.1, 2, m.n_el)
    K, M = assemble(m, MATERIALS, ORIENT, s_K=s_K, **kw)
    if name == "voxel":
        op = stencil(m, MATERIALS, ORIENT)
    else:
        op = unstructured(m, MATERIALS, ORIENT, mode="otf" if name == "otf" else "stored", **kw)
    op = op if s_K is None else op.scaled(s_K=s_K)
    return op, K, M, bc_mask(m)


NAMES = ("stored", "otf", "mixed", "hex27", "voxel")


def blocks(A):
    """(n_nodes, 3, 3) diagonal node blocks of a SciPy matrix."""
    return np.stack(
        [A[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].toarray() for i in range(A.shape[0] // 3)]
    )


@pytest.mark.parametrize("name", ["stored", "voxel"])
@pytest.mark.parametrize("sigma", [0.0, 1.3, -2.0])
def test_ap_operator(name, sigma):
    """Test 12: ap_operator equals a_p(K + sigma M, free, s) on unmasked random vectors; the
    default s is the mean free diag K + |sigma| diag M."""
    op, K, M, free = case(name)
    A = ap_operator(op, free, sigma)
    s = (K.diagonal() + abs(sigma) * M.diagonal())[free].mean()
    assert np.isclose(float(A.s), s, rtol=1e-14) and A.n_dof == op.n_dof
    x = np.random.default_rng(0).standard_normal((op.n_dof, 3))
    ref = a_p(K + sigma * M, free, s) @ x
    y = np.asarray(A(jnp.asarray(x)))
    assert np.abs(y - ref).max() <= 1e-12 * np.abs(ref).max()


@pytest.mark.parametrize("name", NAMES)
def test_node_blocks(name):
    """Test 13: block_diag_K and the block-Jacobi blocks against SciPy, identity on
    constrained components; test 14: the apply is inv(block) @ v with masked output."""
    op, K, M, free = case(name)
    Kb = blocks(K)
    assert np.abs(np.asarray(op.block_diag_K()) - Kb).max() <= 1e-12 * np.abs(Kb).max()
    sigma = 0.8
    m = free.reshape(-1, 3)
    B = blocks(K + sigma * M) * m[:, :, None] * m[:, None, :] + (~m)[:, :, None] * np.eye(3)
    P = block_jacobi(op, free, sigma)
    inv = np.linalg.inv(B)
    assert np.abs(np.asarray(P.args[0]) - inv).max() <= 1e-12 * np.abs(inv).max()
    v = np.random.default_rng(1).standard_normal((op.n_dof, 2))  # constrained entries nonzero
    y = np.asarray(P(jnp.asarray(v)))
    ref = np.einsum("nij,njk->nik", inv, v.reshape(-1, 3, 2)).reshape(-1, 2) * free[:, None]
    assert np.abs(y - ref).max() <= 1e-12 * np.abs(ref).max()
    assert np.all(y[~free] == 0)
    d = np.asarray(jacobi(op, free, sigma)(jnp.asarray(v)))
    ref = v / (K + sigma * M).diagonal()[:, None] * free[:, None]
    assert np.abs(d - ref).max() <= 1e-12 * np.abs(ref).max() and np.all(d[~free] == 0)
    for f in (jacobi, block_jacobi):
        with pytest.raises(ValueError, match="sigma"):
            f(op, free, -0.1)


def d_dense(op, free, kind, sigma):
    """D (n, n) of Jacobi or block Jacobi with identity on constrained components."""
    if kind == "jacobi":
        return np.diag(np.where(free, np.asarray(op.diag_K() + sigma * op.diag_M()), 1.0))
    P = block_jacobi(op, free, sigma)
    return sla.block_diag(*np.linalg.inv(np.asarray(P.args[0])))


PRE = {"jacobi": jacobi, "block": block_jacobi}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["jacobi", "block"])
@pytest.mark.parametrize("name", ["stored", "voxel"])
def test_lmax(name, kind, seed):
    """Test 15: 1.1 lmax_estimate in [1, 1.5] x lambda_max(A_P,ff, D_ff)."""
    op, K, M, free = case(name)
    sigma = 0.8
    A, P = ap_operator(op, free, sigma), PRE[kind](op, free, sigma)
    Aff = a_p(K + sigma * M, free).toarray()[np.ix_(free, free)]
    Dff = d_dense(op, free, kind, sigma)[np.ix_(free, free)]
    lmax = sla.eigh(Aff, Dff, eigvals_only=True)[-1]
    est = float(lmax_estimate(A, P, seed=seed))
    assert 1.0 <= SAFETY * est / lmax <= 1.5, SAFETY * est / lmax


def test_lmax_invariant_start():
    """An invariant start (A = 2.5 I, identity precond) hits beta = 0 at once; the guard
    returns 2.5 instead of NaN."""
    E = block_ell(2.5 * sp.eye(1, format="csr"), b=1)  # v / |v| = +-1: w = 0 exactly
    assert float(lmax_estimate(E, Partial(lambda r: r))) == pytest.approx(2.5, rel=1e-14)


def cheb_ref(Aff, Dff, lmax, k, ratio=30.0):
    """Dense spectral reference S = L^-T Q diag((1 - r_k(mu)) / mu) Q^T L^-1."""
    L = np.linalg.cholesky(Dff)
    Li = sla.solve_triangular(L, np.eye(len(L)), lower=True)
    mu, Q = np.linalg.eigh(Li @ Aff @ Li.T)
    hi = SAFETY * lmax
    lo = hi / ratio
    theta, delta = (hi + lo) / 2, (hi - lo) / 2
    T = np.polynomial.chebyshev.Chebyshev.basis(k)
    r = T((theta - mu) / delta) / T(theta / delta)
    return Li.T @ Q @ np.diag((1 - r) / mu) @ Q.T @ Li, theta


@pytest.mark.parametrize("kind", ["jacobi", "block"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_chebyshev(degree, kind):
    """Test 16: the dense spectral reference to 1e-12; symmetry, A_P-self-adjoint error
    propagation, positivity; degree 1 is D^-1 / theta."""
    op, K, M, free = case("stored")
    sigma = 0.8
    A, P = ap_operator(op, free, sigma), PRE[kind](op, free, sigma)
    lmax = lmax_estimate(A, P)
    S = chebyshev(A, P, lmax, degree)
    rng = np.random.default_rng(degree)
    U = rng.standard_normal((op.n_dof, 3)) * free[:, None]
    SU = np.asarray(S(jnp.asarray(U)))
    Aff = a_p(K + sigma * M, free).toarray()[np.ix_(free, free)]
    Sref, theta = cheb_ref(
        Aff, d_dense(op, free, kind, sigma)[np.ix_(free, free)], float(lmax), degree
    )
    ref = Sref @ U[free]
    assert np.abs(SU[free] - ref).max() <= 1e-12 * np.abs(ref).max()
    assert np.all(SU[~free] == 0)
    u, v, su, sv = U[:, 0], U[:, 1], SU[:, 0], SU[:, 1]
    assert abs(u @ sv - v @ su) <= 1e-12 * abs(u @ sv)
    Ap = a_p(K + sigma * M, free).toarray()
    E = lambda x: x - np.asarray(S(jnp.asarray(Ap @ x)))  # noqa: E731
    Eu, Ev = E(U[:, :2])[:, 0], E(U[:, :2])[:, 1]
    assert abs(Ev @ Ap @ u - v @ Ap @ Eu) <= 1e-12 * np.abs(Ap @ u).max() * np.abs(v).sum()
    assert (np.einsum("ij,ij->j", U, SU) > 0).all()
    if degree == 1:
        D1 = np.asarray(P(jnp.asarray(U))) / theta
        assert np.abs(SU - D1).max() <= 1e-15 * np.abs(D1).max()


def test_chebyshev_beats_block_jacobi():
    """Test 16: degree-2 Chebyshev over block Jacobi takes fewer PCG iterations (recorded)."""
    op, free, K, M, lam = box()
    A, P = ap_operator(op, free, lam[0]), block_jacobi(op, free, lam[0])
    b = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 3)) * free[:, None])
    _, ib = pcg(A, b, P, 1e-12, 3000)
    _, ic = pcg(A, b, chebyshev(A, P, lmax_estimate(A, P)), 1e-12, 3000)
    assert ib["converged"].all() and ic["converged"].all()
    assert (ic["iterations"] < ib["iterations"]).all()


def test_errors():
    op, K, M, free = case("stored")
    A, P = ap_operator(op, free), jacobi(op, free)
    with pytest.raises(ValueError, match="degree"):
        chebyshev(A, P, 1.0, 0)
    with pytest.raises(ValueError, match="free"):
        ap_operator(op, np.zeros(op.n_dof, bool))


def test_jacobi_moved():
    """Test 17: the PR 9 jacobi is the precond one."""
    assert jaxmodal.eigen.jacobi is jaxmodal.precond.jacobi


def test_study_smoke():
    """Test 18: the h-study on its smallest meshes."""
    from benchmarks import pcg_study

    r = pcg_study.main(sizes={"hex20": (2,), "voxel": (4,)}, write=False)
    assert {"runs", "sigma", "system"} <= r.keys() and len(r["runs"]) == 6
    assert all(min(x["iterations"]) > 0 and all(x["converged"]) for x in r["runs"])
