"""lobpcg, jacobi and sigma_estimate against reference_modes (SciPy) and dense linear algebra."""

import dataclasses
import functools
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl
from jax.tree_util import Partial
from scipy.sparse.linalg import eigsh

from jaxmodal.eigen import jacobi, lobpcg, reference_modes, sigma_estimate
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import (
    assemble,
    dof_mask,
    m_orthonormalize,
    restrict,
    rigid_modes,
    unstructured,
)
from jaxmodal.operators.common import register

# the module itself: the package exports the function under the same name
lob = importlib.import_module("jaxmodal.eigen.lobpcg")

EPS = np.finfo(float).eps
MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
VOID = Material(isotropic(3e-9, 0.3), 2e-12)
BOX = (1.0, 0.83, 0.67)
BCS = {"clamped": {"xmin": "xyz"}, "free": {}}


@functools.cache
def problem(t, shape, bc, L=BOX, void=0.0):
    """Operator, mask, assembled free blocks and (free bodies) M-orthonormal rigid modes."""
    m = box_mesh(shape, spacing=tuple(a / s for a, s in zip(L, shape, strict=True)), type=t)
    mats = MAT
    if void:
        m.material[:] = np.random.default_rng(0).random(m.n_el) < void
        mats = {**MAT, 1: VOID}
    op = unstructured(m, mats)
    free = dof_mask(m, BCS[bc])
    K, M = assemble(m, mats)
    Kf, Mf, idx = restrict(K, M, free)
    R = None
    if bc == "free":
        Rn = jnp.asarray(rigid_modes(m))
        R = m_orthonormalize(Rn, op.apply_M(Rn))
    lmax = eigsh(Kf, 1, M=Mf, which="LA", tol=1e-3, return_eigenvectors=False)[0]
    return SimpleNamespace(op=op, free=free, K=K, M=M, Kf=Kf, Mf=Mf, idx=idx, R=R, lmax=lmax)


@functools.cache
def reference(p_args, n):
    """The lowest n elastic modes plus 6 more (to finish clusters): lam, V on the free DOFs."""
    p = problem(*p_args)
    nr = 0 if p.R is None else p.R.shape[1]
    lam, V = reference_modes(p.K, p.M, nr + n + 6, p.free)
    return lam[nr:], np.asarray(V)[p.idx][:, nr:]


@functools.cache
def chol(p_args):
    return sl.cholesky(problem(*p_args).Mf.toarray(), lower=True)


def f_lam(p, lam):
    return np.maximum(1e-8, 10 * EPS * p.lmax / lam)


def clusters(lam, rel):
    """Index groups of consecutive eigenvalues with relative gaps < rel."""
    out = [[0]]
    for i in range(1, len(lam)):
        if lam[i] - lam[i - 1] < rel * lam[i]:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def r_M(p_args, lam, X):
    """lam-scaled residuals in the M^-1 norm of M-normalized columns X (free DOFs), dense."""
    p, Lf = problem(*p_args), chol(p_args)
    X = X / np.sqrt(np.einsum("ij,ij->j", X, p.Mf @ X))
    res = p.Kf @ X - (p.Mf @ X) * lam
    return np.linalg.norm(sl.solve_triangular(Lf, res, lower=True), axis=0) / np.abs(lam)


def max_sine(p_args, Q, Qr):
    """Largest principal sine between M-orthonormal Q and Qr, stably: sigma_max(L^T (Q - Qr Qr^T
    M Q)), never sqrt(1 - cos^2), which bottoms out near 1e-8."""
    p, Lf = problem(*p_args), chol(p_args)
    E = Q - Qr @ (Qr.T @ (p.Mf @ Q))
    return np.linalg.norm(Lf.T @ E, 2)


def residuals(p, lam, V):
    """lam-scaled residuals with the solver's own masked operator."""
    m = jnp.asarray(p.free, float)[:, None]
    KV, MV = m * p.op.apply_K(m * V), m * p.op.apply_M(m * V)
    return np.linalg.norm(KV - MV * lam, axis=0) / (np.abs(lam) * np.linalg.norm(MV, axis=0))


def check_modes(p_args, lam, V, info, tol):
    """Tests 1 and 3: lam errors, principal angles, orthonormality, masking, residuals."""
    p = problem(*p_args)
    n = len(lam)
    lr, Vr = reference(p_args, n)
    V = np.asarray(V)
    assert info["converged"] and not info["stagnated"]
    assert np.all(np.diff(lam) >= 0)
    assert np.all(V[~p.free] == 0)
    Vf = V[p.idx]
    assert np.abs(Vf.T @ (p.Mf @ Vf) - np.eye(n)).max() <= 1e-10
    # rounding differs between the jitted and eager products (about 1e-7 relative at r ~ 1e-11);
    # residuals from implicitly updated products would differ by about 1e-3
    assert np.allclose(info["residuals"], residuals(p, lam, jnp.asarray(V)), rtol=1e-5, atol=0)
    if p.R is not None:
        assert np.abs(np.asarray(p.R).T @ (p.M @ V)).max() <= 1e-10
    rm, rm_ref = r_M(p_args, lam, Vf), r_M(p_args, lr[:n], Vr[:, :n])
    groups = clusters(lr, 1e-4)
    for g in groups:
        if g[0] >= n:
            break
        assert g[-1] < n, "a cluster straddles the wanted modes"
        assert rm_ref[g].max() <= 0.1 * rm[g].max()
        others = np.delete(lr, g)
        gap = np.abs(others[:, None] - lr[g][None]).min()
        bound = 10 * (rm[g] * lam[g]).max() / gap
        assert max_sine(p_args, Vf[:, g], Vr[:, g]) <= bound
        rel_gap = gap / lr[g].max()
        if rel_gap >= 1e-2:
            assert np.all(np.abs(lam[g] - lr[g]) / lr[g] <= f_lam(p, lr[g]))
        else:
            assert np.all(np.abs(lam[g] - lr[g]) / lr[g] <= np.maximum(10 * tol, f_lam(p, lr[g])))


CASES = [
    ("hex20", (6, 6, 6), "clamped"),
    ("hex20", (6, 6, 6), "free"),
    ("hex27", (4, 4, 4), "clamped"),
    ("hex8", (10, 10, 10), "clamped"),
]


@pytest.mark.parametrize("p_args", CASES, ids=lambda c: f"{c[0]}-{c[2]}")
def test_modes(p_args):
    p = problem(*p_args)
    lam, V, info = lobpcg(p.op, p.free, 12, tol=1e-10, R=p.R)
    check_modes(p_args, lam, V, info, 1e-10)


@pytest.mark.parametrize(
    "p_args",
    [("hex20", (6, 3, 3), "free", (2.0, 1.0, 1.0)), ("hex20", (4, 4, 4), "free", (1.0, 1.0, 1.0))],
    ids=["beam", "cube"],
)
def test_multiplicities(p_args):
    p = problem(*p_args)
    lr, _ = reference(p_args, 12)
    n = max(i for i in range(6, 13) if lr[i] - lr[i - 1] > 1e-6 * lr[i])  # no cluster at the cut
    groups = [g for g in clusters(lr, 1e-8) if g[-1] < n]
    assert any(len(g) > 1 for g in groups)
    lam, _, info = lobpcg(p.op, p.free, n, tol=1e-10, R=p.R)
    assert info["converged"]
    for g in groups:
        c = lr[g[0]]
        assert np.sum(np.abs(lam - c) <= f_lam(p, c) * c) == len(g)


def test_gap_band():
    p_args = ("hex20", (6, 3, 3), "clamped", (2.0, 1.0, 1.002))
    p = problem(*p_args)
    lr, _ = reference(p_args, 6)
    gaps = np.diff(lr[:6]) / lr[1:6]
    assert np.any((gaps >= 1e-4) & (gaps < 1e-2))  # bending pairs in the band
    lam, V, info = lobpcg(p.op, p.free, 6, tol=1e-10)
    check_modes(p_args, lam, V, info, 1e-10)


def test_stagnation(monkeypatch):
    p = problem("hex20", (4, 4, 4), "clamped")
    lam, _, info = lobpcg(p.op, p.free, 10, tol=1e-16, maxit=2000)
    assert info["converged"] and info["stagnated"] and info["iterations"] < 2000
    assert np.all(info["residuals"] <= info["thresholds"])
    monkeypatch.setattr(lob, "C_FLOOR", 0.0)  # floor unreachable: the no-progress exit
    _, _, info = lobpcg(p.op, p.free, 10, tol=1e-16, maxit=2000)
    assert not info["converged"] and info["stagnated"] and info["iterations"] < 2000


def test_one_mode_no_false_stagnation():
    """The residual alone has no new minimum for 85 iterations here; the Ritz-value clause."""
    p = problem("hex20", (6, 6, 6), "clamped")
    _, _, info = lobpcg(p.op, p.free, 1, tol=1e-10)
    assert info["converged"] and not info["stagnated"]


def test_rank_deficient_start():
    p_args = ("hex20", (4, 4, 4), "clamped")
    p = problem(*p_args)
    X0 = np.random.default_rng(0).standard_normal((p.op.n_dof, 15))
    X0[:, 5:10] = X0[:, :5]
    lam, _, info = lobpcg(p.op, p.free, 10, tol=1e-8, X0=X0)
    lr, _ = reference(p_args, 10)
    assert info["refills"] >= 1 and info["converged"]
    assert np.all(np.abs(lam - lr[:10]) / lr[:10] <= f_lam(p, lr[:10]))


def start(p, k, seed=0):
    """A converging solver state as lobpcg builds it, plus the step's other arguments."""
    m = jnp.asarray(p.free, float)[:, None]
    R = jnp.zeros((p.op.n_dof, 0))
    X = jax.random.normal(jax.random.key(seed), (p.op.n_dof, k))
    X, KX, MX, _ = lob._start(X, p.op, m, R, R)
    X, KX, MX, lam = lob._rr_start(X, KX, MX)
    Z = [jnp.zeros_like(X) for _ in range(3)]
    nK, nM = map(
        float, lob._norms(p.op, m, m * jax.random.normal(jax.random.key(9), (len(m), 1)), 30)
    )
    consts = jnp.array([nK, nM, 1e-10, lob.C_FLOOR])
    return [X, KX, MX, *Z, jnp.zeros(k, bool), lam], (p.op, m, jacobi(p.op, p.free), R, R, consts)


def test_forced_p_drop():
    p = problem("hex20", (4, 4, 4), "clamped")
    state, args = start(p, 15)
    X, m = state[0], args[1]
    rng = jax.random.split(jax.random.key(1), 3)
    for noise in (1e-12, None):
        if noise:  # P inside span(X) up to noise: breaks the invariant, the pivot test fires
            P = X @ jax.random.normal(rng[0], (15, 15)) + noise * m * jax.random.normal(
                rng[1], X.shape
            )
        else:  # a random P, M-orthonormal and M-orthogonal to X
            P = m * jax.random.normal(rng[2], X.shape)
            P = P - X @ (state[2].T @ P)
            P = m_orthonormalize(P, m * p.op.apply_M(m * P))
        KP, MP = m * p.op.apply_K(m * P), m * p.op.apply_M(m * P)
        st = [a + 0 for a in state[:3]] + [P, KP, MP, jnp.ones(15, bool), state[7] + 0]
        _, out = lob.lobpcg_step(tuple(st), *args, False)
        assert np.isfinite(np.asarray(out)).all()
        assert bool(out[-1]) == (noise is not None)


def test_p_invariant():
    p = problem("hex20", (4, 4, 4), "clamped")
    state, args = start(p, 15)
    state = tuple(state)
    for it in range(30):
        state, _ = lob.lobpcg_step(state, *args, (it + 1) % lob.RECOMPUTE == 0)
    m = args[1]
    X, P, live = state[0], state[3], np.asarray(state[6])
    assert live.any()
    MX, MP = m * p.op.apply_M(m * X), m * p.op.apply_M(m * P)
    assert np.abs(np.asarray(P.T @ MX)[live]).max() <= 1e-6
    G = np.asarray(P.T @ MP)[np.ix_(live, live)]
    assert np.abs(G - np.eye(live.sum())).max() <= 1e-6


def test_p_coeffs():
    rng = np.random.default_rng(3)
    k = 15
    A = rng.standard_normal((3 * k, 3 * k))
    G = np.eye(3 * k) + 0.01 * (A + A.T) / 2
    assert np.linalg.cond(G) <= 10
    Lg = np.linalg.cholesky(G)
    Cx = sl.solve_triangular(Lg.T, np.linalg.qr(rng.standard_normal((3 * k, k)))[0], lower=False)
    Cp = rng.standard_normal((3 * k, k))
    Cp[:, 4] = Cx @ rng.standard_normal(k) + 1e-12 * rng.standard_normal(3 * k)
    Cp, live, keep = map(
        np.asarray, lob._p_coeffs(jnp.asarray(Cx), jnp.asarray(Cp), jnp.asarray(G))
    )
    assert np.array_equal(live, np.arange(k) != 4) and keep.sum() == k - 1
    Ck = Cp[:, keep]
    assert np.abs(Ck.T @ G @ Ck - np.eye(k - 1)).max() <= 1e-12
    assert np.abs(Cx.T @ G @ Ck).max() <= 1e-12


def test_near_singular_mass():
    p_args = ("hex20", (6, 6, 6), "clamped", BOX, 0.3)
    p = problem(*p_args)
    lam, _, info = lobpcg(p.op, p.free, 10, tol=1e-8, maxit=500)
    lr, _ = reference(p_args, 10)
    assert info["converged"]
    assert np.all(np.abs(lam - lr[:10]) / lr[:10] <= f_lam(p, lr[:10]))


@pytest.mark.slow
@pytest.mark.parametrize("bc", ["clamped", "free"])
def test_floor(bc):
    p_args = ("hex20", (4, 4, 4), bc)
    p = problem(*p_args)
    lr, _ = reference(p_args, 10)
    _, _, info = lobpcg(p.op, p.free, 10, tol=1e-16, R=p.R, maxit=2000)
    ratio = info["history"].min() / (EPS * p.lmax / lr[0])
    print(f"floor {bc}: min residual {info['history'].min():.2e}, ratio {ratio:.2f}")
    assert 0.1 <= ratio <= 10


COMPILES = """
import json, jax
from jaxmodal import config
config.enable_x64()
n = [0]
def count(event, duration, **kw):
    if "backend_compile" in event and "lobpcg_step" in str(kw.get("fun_name")):
        n[0] += 1
jax.monitoring.register_event_duration_secs_listener(count)
from jaxmodal.eigen import jacobi, lobpcg, sigma_estimate
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import dof_mask, unstructured
m = box_mesh((3, 3, 3), type="hex20")
op = unstructured(m, {0: Material(isotropic(3.0, 0.3), 2.0)})
free = dof_mask(m, {"xmin": "xyz"})
out = []
lobpcg(op, free, 6, tol=1e-8); out.append(n[0])
op2 = op.scaled(s_K=jax.numpy.full(m.n_el, 2.0))
lobpcg(op2, free, 6, tol=1e-6, precond=jacobi(op2, free, 0.5)); out.append(n[0])
sigma_estimate(op, free); out.append(n[0])
print(json.dumps(out))
"""


def test_compiles():
    """One step compile per solve; new data (operator, preconditioner, tol) does not recompile."""
    res = subprocess.run(
        [sys.executable, "-c", COMPILES], capture_output=True, text=True, check=True
    )
    first, second, sigma = json.loads(res.stdout.strip().splitlines()[-1])
    assert (first, second) == (1, 1) and sigma - second <= 1


@pytest.mark.parametrize(
    "p_args",
    [("hex20", (8, 8, 8), "clamped"), ("hex20", (6, 6, 6), "free")],
    ids=["clamped", "free"],
)
def test_sigma_estimate(p_args):
    p = problem(*p_args)
    lr, _ = reference(p_args, 1)
    sigma, its = sigma_estimate(p.op, p.free, p.R)
    assert lr[0] * (1 - 1e-10) <= sigma <= 2 * lr[0] and its < 300


def test_jacobi():
    p = problem("hex20", (4, 4, 4), "clamped")
    dK, dM = p.K.diagonal(), p.M.diagonal()
    s = 0.7 * dK[p.free].mean() / dM[p.free].mean()
    v = np.random.default_rng(0).standard_normal((p.op.n_dof, 3))
    got = np.asarray(jacobi(p.op, p.free, s)(jnp.asarray(v)))
    want = np.where(p.free[:, None], v / (dK + s * dM)[:, None], 0.0)
    assert np.abs(got - want).max() <= 1e-12 * np.abs(want).max()
    assert np.all(got[~p.free] == 0)


def test_errors():
    m = box_mesh((1, 1, 1), type="hex8")
    op = unstructured(m, MAT)
    free = dof_mask(m, {"xmin": "xyz"})  # 12 free DOFs
    with pytest.raises(ValueError, match="block size 13"):
        lobpcg(op, free, 8)
    with pytest.raises(ValueError, match="free must be"):
        lobpcg(op, free.astype(int), 2)
    R = np.zeros((op.n_dof, 1))
    R[free, 0] = 1.0  # not M-normalized
    with pytest.raises(ValueError, match="M-orthonormal"):
        lobpcg(op, free, 2, R=R)


@dataclasses.dataclass(frozen=True)
class Diagonal:
    """K = diag(dK), M = diag(dM): the operator interface without operator workspace."""

    dK: jax.Array
    dM: jax.Array

    @property
    def n_dof(self):
        return self.dK.shape[0]

    def apply_K(self, v):
        return self.dK[:, None] * v

    def apply_M(self, v):
        return self.dM[:, None] * v

    def diag_K(self):
        return self.dK

    def diag_M(self):
        return self.dM


register(Diagonal, [])


def test_step_memory():
    """temp + aliased state of the compiled step <= 16.5 blocks of (n, m) fp64 (sequenced
    update; 18 without the barriers). A diagonal operator keeps operator workspace out."""
    n, k = 100_000, 17
    S = jax.ShapeDtypeStruct
    blk, vec = S((n, k), jnp.float64), S((n,), jnp.float64)
    state = (blk,) * 6 + (S((k,), bool), S((k,), jnp.float64))
    R = S((n, 6), jnp.float64)
    op = Diagonal(vec, vec)
    args = (state, op, S((n, 1), jnp.float64), Partial(lob._jacobi, vec), R, R,
            S((4,), jnp.float64), S((), bool))  # fmt: skip
    mem = lob.lobpcg_step.lower(*args).compile().memory_analysis()
    blocks = (mem.temp_size_in_bytes + mem.alias_size_in_bytes) / (n * k * 8)
    print(f"step memory: {blocks:.2f} blocks")
    assert blocks <= 16.5, blocks


def test_non_finite_step():
    p = problem("hex20", (4, 4, 4), "clamped")
    with pytest.raises(FloatingPointError, match="iteration 1"):
        lobpcg(p.op, p.free, 6, precond=Partial(lambda r: r * jnp.nan))
