"""PCG and MINRES against spsolve on the PR 4 A_P."""

import json
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spla
from helpers import box, rhs
from jax.tree_util import Partial

from jaxmodal.linalg import minres, pcg
from jaxmodal.linalg.krylov import CHECK, STALL_C
from jaxmodal.operators import a_p, ap_operator, restrict
from jaxmodal.precond import block_jacobi, chebyshev, jacobi, lmax_estimate


def rel(x, xr):
    return (np.abs(np.asarray(x) - xr).max(0) / np.abs(xr).max(0)).max()


def run(solver, A, b, P, tol, maxit=3000):
    x, info = solver(A, jnp.asarray(b), P, tol, maxit)
    return np.asarray(x), {k: np.asarray(v) for k, v in info.items()}


@pytest.mark.parametrize("pre", ["none", "jacobi", "block", "chebyshev"])
@pytest.mark.parametrize("case", [("clamped", 1), ("clamped", 100), ("free", 1)])
def test_pcg_accuracy(case, pre):
    """Test 1: rel. error <= 1e-10 against spsolve, a zero column, constrained rows 0."""
    bcs, f = case
    op, free, K, M, lam = box(bcs)
    sigma = f * lam[0]
    A = ap_operator(op, free, sigma)
    P = {"none": None, "jacobi": jacobi(op, free, sigma), "block": block_jacobi(op, free, sigma)}
    P["chebyshev"] = chebyshev(A, P["block"], lmax_estimate(A, P["block"]))
    b = rhs(free, 4)
    b[:, 2] = 0
    xr = spla.spsolve(a_p(K + sigma * M, free).tocsc(), b)
    x, info = run(pcg, A, b, P[pre], 1e-12)
    keep = [0, 1, 3]
    assert rel(x[:, keep], xr[:, keep]) <= 1e-10
    assert info["converged"].all() and (info["residuals"] <= 1e-12).all()
    assert np.all(x[:, 2] == 0) and info["iterations"][2] == 0
    assert np.all(x[~free] == 0)


def easy(op, free):
    """An easy right-hand side: M times a combination of the 3 lowest modes."""
    K, M = box()[2:4]
    Kf, Mf, idx = restrict(K, M, free)
    V = np.zeros((len(free), 3))
    V[idx] = spla.eigsh(Kf, 3, M=Mf, sigma=-1.0, v0=np.ones(len(idx)))[1]
    return (M @ V) @ np.array([1.0, -0.5, 0.25])


@pytest.mark.parametrize("solver", [pcg, minres], ids=["pcg", "minres"])
def test_columns_independent(solver):
    """Test 2: a block of an easy and a random column equals the single-column solves, and the
    easy column stays frozen once converged."""
    op, free, K, M, lam = box()
    A, P = ap_operator(op, free, lam[0]), jacobi(op, free, lam[0])
    b = np.stack([easy(op, free) * free, rhs(free, 1)[:, 0]], 1)
    x, info = run(solver, A, b, P, 1e-8)
    for j in range(2):
        xj, ij = run(solver, A, b[:, j : j + 1], P, 1e-8)
        assert abs(info["iterations"][j] - ij["iterations"][0]) <= 1  # m=1 vs 2 rounding
        assert rel(x[:, j : j + 1], xj) <= 1e-9
    # frozen: the easy column's x does not change after it converged
    x0, _ = run(solver, A, b, P, 1e-8, maxit=int(info["iterations"][0]))
    assert np.array_equal(x0[:, 0], x[:, 0])
    assert info["iterations"][0] < info["iterations"][1]
    b2 = b * np.array([1.0, 1e6])
    x2, i2 = run(solver, A, b2, P, 1e-8)
    assert np.array_equal(i2["iterations"], info["iterations"])
    assert rel(x2[:, :1], x[:, :1]) <= 1e-12


def test_pcg_exits():
    """Test 3: the floor exit after STALL_R restarts, convergence after a restart, no restart
    above the floor, the maxit exit."""
    op, free, K, M, lam = box()
    A, P = ap_operator(op, free, lam[0]), jacobi(op, free, lam[0])
    b = rhs(free, 3)
    _, info = run(pcg, A, b, P, 1e-16)
    assert info["stagnated"].all() and not info["converged"].any()
    assert (info["replacements"] == 5).all() and (info["iterations"] < 3000).all()
    assert (info["residuals"] <= 1e-13).all()
    _, info = run(pcg, A, b, P, 1e-14)
    assert (info["converged"] & (info["replacements"] >= 1)).any()
    _, info = run(pcg, A, b, P, 1e-12)
    assert info["converged"].all() and (info["replacements"] == 0).all()
    _, info = run(pcg, A, b, P, 1e-12, maxit=5)
    assert not info["converged"].any() and not info["stagnated"].any()
    assert (info["iterations"] == 5).all()


def test_minres_accuracy():
    """Test 4: SPD (with a zero column) and indefinite K - sigma_b M, sigma_b between lambda_6
    and lambda_7, preconditioned by Jacobi of K + sigma_b M."""
    op, free, K, M, lam = box()
    b = rhs(free, 4)
    b[:, 2] = 0
    keep = [0, 1, 3]
    xr = spla.spsolve(a_p(K + lam[0] * M, free).tocsc(), b)
    x, info = run(minres, ap_operator(op, free, lam[0]), b, jacobi(op, free, lam[0]), 1e-12)
    assert rel(x[:, keep], xr[:, keep]) <= 1e-10 and info["converged"].all()
    assert (info["residuals"] <= 1e-12).all()
    assert np.all(x[:, 2] == 0) and info["iterations"][2] == 0
    sb = (lam[5] + lam[6]) / 2
    A = ap_operator(op, free, -sb)
    s = K.diagonal()[free].mean() + sb * M.diagonal()[free].mean()
    assert np.isclose(float(A.s), s, rtol=1e-14)
    xr = spla.spsolve(a_p(K - sb * M, free, s).tocsc(), b)
    x, info = run(minres, A, b, jacobi(op, free, sb), 1e-12)
    assert rel(x[:, keep], xr[:, keep]) <= 1e-10 and info["converged"].all()
    assert (info["residuals"] <= 1e-12).all()
    assert np.all(x[~free] == 0)


def test_minres_stall_count():
    """CHECK and STALL_C: 3 distinct eigenvalues end MINRES in 3 steps, x then stays frozen
    by the beta = 0 guards (no NaN) and the column stagnates at exactly (1 + STALL_C) CHECK."""
    A = Partial(lambda d, x: d[:, None] * x, jnp.array([1.0, 3.0, 3.0, 7.0]))
    b = jnp.array([[1.0, 0.5], [2.0, 0.0], [0.0, 1.0], [-1.0, 2.0]])
    x, info = minres(A, b, None, 1e-300, 1000)
    assert info["stagnated"].all() and (np.asarray(info["iterations"]) == 60).all()
    assert np.abs(np.asarray(x) - b / A.args[0][:, None]).max() <= 1e-15


def test_minres_best():
    """With sigma on an eigenvalue MINRES stagnates and its explicit 2-norm residual drifts up
    after the minimum: the best iterate is returned, no worse than a run stopped at that
    minimum (one column: the same iterates for any maxit)."""
    op, free, K, M, lam = box()
    A, P = ap_operator(op, free, -lam[2]), jacobi(op, free, lam[2])
    b = rhs(free, 1)
    x, info = run(minres, A, b, P, 1e-8, maxit=20000)
    its = int(info["iterations"][0])
    assert info["stagnated"].all() and info["residuals"][0] > 1e-3
    r = np.linalg.norm(b - np.asarray(A(jnp.asarray(x)))) / np.linalg.norm(b)
    assert r == pytest.approx(info["residuals"][0], rel=1e-12)
    _, early = run(minres, A, b, P, 1e-8, maxit=its - (1 + STALL_C) * CHECK + CHECK)
    assert info["residuals"][0] <= early["residuals"][0]
    # maxit between checks: the unchecked last iterate is better than the check at 10
    A, P, b = ap_operator(op, free, -(lam[2] + lam[3]) / 2), jacobi(op, free, lam[2]), rhs(free, 2)
    x, info = run(minres, A, b, P, 1e-14, maxit=15)
    _, check = run(minres, A, b, P, 1e-14, maxit=10)
    r = np.linalg.norm(b - np.asarray(A(jnp.asarray(x))), axis=0) / np.linalg.norm(b, axis=0)
    assert np.allclose(r, info["residuals"], rtol=1e-12) and (r < check["residuals"]).all()


def test_minres_exits():
    """Test 5: stagnation well before maxit, the maxit exit, counts in multiples of 10."""
    op, free, K, M, lam = box()
    sb = (lam[5] + lam[6]) / 2
    A, P = ap_operator(op, free, -sb), jacobi(op, free, sb)
    b = rhs(free, 3)
    _, info = run(minres, A, b, P, 1e-15, maxit=10000)
    assert info["stagnated"].all() and not info["converged"].any()
    assert (info["iterations"] < 2000).all() and (info["iterations"] % 10 == 0).all()
    _, info = run(minres, A, b, P, 1e-12, maxit=30)
    assert not info["converged"].any() and not info["stagnated"].any()
    assert (info["iterations"] == 30).all()


@pytest.mark.parametrize("solver", [pcg, minres], ids=["pcg", "minres"])
def test_nan_exit(solver):
    """A residual that turns NaN mid-run stops its column at once (first check for MINRES);
    the other column is bit for bit unchanged."""
    op, free, K, M, lam = box()
    A, P = ap_operator(op, free, lam[0]), jacobi(op, free, lam[0])
    b = rhs(free, 2)
    W = np.ones_like(b)
    W[np.flatnonzero(free)[0], 1] = np.nan  # column 1's preconditioner returns NaN
    Pn = Partial(lambda w, r: w * P(r), jnp.asarray(W))
    x, info = run(solver, A, b, Pn, 1e-8)
    xr, ir = run(solver, A, b, Partial(Pn.func, jnp.ones_like(b)), 1e-8)  # without the NaN
    assert info["iterations"][1] <= 10 and np.isnan(info["residuals"][1])
    assert not info["converged"][1] and not info["stagnated"][1]
    assert info["converged"][0] and info["iterations"][0] == ir["iterations"][0]
    assert np.array_equal(x[:, 0], xr[:, 0])


def test_errors():
    op, free, K, M, lam = box()
    A = ap_operator(op, free, lam[0])
    with pytest.raises(ValueError, match="shape"):
        pcg(A, jnp.ones(op.n_dof))
    with pytest.raises(ValueError, match="maxit"):
        minres(A, jnp.ones((op.n_dof, 1)), maxit=0)


COMPILES = """
import json, jax
from jaxmodal import config
config.enable_x64()
jax.config.update("jax_enable_compilation_cache", False)
n = {"jit(_pcg)": 0, "jit(_minres)": 0}
def count(event, duration, **kw):
    if "backend_compile" in event and kw.get("fun_name") in n:
        n[kw["fun_name"]] += 1
jax.monitoring.register_event_duration_secs_listener(count)
import jax.numpy as jnp, numpy as np
from jaxmodal.linalg import minres, pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import ap_operator, dof_mask, unstructured
from jaxmodal.precond import jacobi
m = box_mesh((3, 3, 3), type="hex20")
op = unstructured(m, {0: Material(isotropic(3.0, 0.3), 2.0)})
free = dof_mask(m, {"xmin": "xyz"})
b = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, 2)) * free[:, None])
out = []
for sigma, tol in [(0.5, 1e-8), (2.0, 1e-6)]:
    A = ap_operator(op, free, sigma)
    pcg(A, b * sigma, jacobi(op, free, sigma), tol)
    minres(A, b * sigma, jacobi(op, free, sigma), tol)
    out.append(dict(n))
print(json.dumps(out))
"""


def test_compiles():
    """Test 6: a new operator (sigma), preconditioner data, tol and b recompile nothing."""
    res = subprocess.run(
        [sys.executable, "-c", COMPILES], capture_output=True, text=True, check=True
    )
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out[0] == {"jit(_pcg)": 1, "jit(_minres)": 1} and out[1] == out[0]
