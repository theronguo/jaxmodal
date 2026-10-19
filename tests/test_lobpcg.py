"""lobpcg, jacobi and sigma_estimate against reference_modes (SciPy) and dense linear algebra."""

import dataclasses
import functools
import gc
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
from helpers import max_sine, r_M
from jax.tree_util import Partial
from scipy.sparse.linalg import eigsh

from jaxmodal.eigen import jacobi, lobpcg, reference_modes, sigma_estimate
from jaxmodal.linalg import count_below
from jaxmodal.materials import Material, isotropic
from jaxmodal.memory import (
    amg_memory,
    gmg_memory,
    lobpcg_memory,
    solve_modes_memory,
    stencil_memory,
    unstructured_memory,
)
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import (
    assemble,
    dof_mask,
    m_orthonormalize,
    restrict,
    rigid_modes,
    unstructured,
)
from jaxmodal.operators.common import register
from jaxmodal.precond.jacobi import _jacobi

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
    # jitted and eager products agree far below this (about 5e-15 relative measured in 9b);
    # residuals from implicitly updated products would differ by 1e-4 or more
    assert np.allclose(info["residuals"], residuals(p, lam, jnp.asarray(V)), rtol=1e-5, atol=0)
    if p.R is not None:
        assert np.abs(np.asarray(p.R).T @ (p.M @ V)).max() <= 1e-10
    Lf = chol(p_args)
    rm, rm_ref = r_M(p.Kf, p.Mf, Lf, lam, Vf), r_M(p.Kf, p.Mf, Lf, lr[:n], Vr[:, :n])
    groups = clusters(lr, 1e-4)
    for g in groups:
        if g[0] >= n:
            break
        assert g[-1] < n, "a cluster straddles the wanted modes"
        assert rm_ref[g].max() <= 0.1 * rm[g].max()
        others = np.delete(lr, g)
        gap = np.abs(others[:, None] - lr[g][None]).min()
        bound = 10 * (rm[g] * lam[g]).max() / gap
        assert max_sine(p.Mf, Lf, Vf[:, g], Vr[:, g]) <= bound
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


@functools.cache
def solve12(p_args, low=False):
    p = problem(*p_args)
    return lobpcg(p.op, p.free, 12, tol=1e-10, R=p.R, low_memory=low)


@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
@pytest.mark.parametrize("p_args", CASES, ids=lambda c: f"{c[0]}-{c[2]}")
def test_modes(p_args, low):
    p = problem(*p_args)
    lam, V, info = solve12(p_args, low)
    check_modes(p_args, lam, V, info, 1e-10)
    assert info["p_drops"] == 0
    if low:  # test 2: the low-memory step finds the same lam
        assert np.all(np.abs(lam - solve12(p_args)[0]) / lam <= f_lam(p, lam))
    for A, key in ((p.Kf, "norm_K"), (p.Mf, "norm_M")):  # power iteration: from below
        exact = eigsh(A, 1, which="LA", return_eigenvectors=False)[0]
        assert 0.8 <= info[key] / exact <= 1 + 1e-10  # >= 0.90 over 40 seeds; 3 its: <= 0.77


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
    confirm, calls = lob._confirm, []

    def counted(*args):  # every exit reports residuals of explicit products
        calls.append(1)
        return confirm(*args)

    monkeypatch.setattr(lob, "_confirm", counted)
    lam, _, info = lobpcg(p.op, p.free, 10, tol=1e-16, maxit=2000)
    assert info["converged"] and info["stagnated"] and info["iterations"] < 2000
    assert np.all(info["residuals"] <= info["thresholds"])
    monkeypatch.setattr(lob, "C_FLOOR", 0.0)  # floor unreachable: the no-progress exit
    calls.clear()
    _, _, info = lobpcg(p.op, p.free, 10, tol=1e-16, maxit=2000)
    assert not info["converged"] and info["stagnated"] and info["iterations"] < 2000
    assert calls
    calls.clear()
    lam, V, info = lobpcg(p.op, p.free, 10, tol=1e-10, maxit=30)  # the maxit exit
    assert len(calls) == 1
    assert not info["converged"] and not info["stagnated"] and info["iterations"] == 30
    assert np.allclose(info["residuals"], residuals(p, lam, V), rtol=1e-5)


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


def start(p, k, seed=0, low=False):
    """A single-chunk solver state as lobpcg builds it, plus the step's other arguments."""
    n = p.op.n_dof
    m = jnp.asarray(p.free, float)[:, None]
    R = B = jnp.zeros((n, 0))
    key = jax.random.key(seed)
    nK, nM = map(float, lob._norms(p.op, m, m * jax.random.normal(key, (n, 1))))
    X = jax.random.normal(jax.random.fold_in(key, 1), (n, k))
    state, _, _ = lob._begin(X, key, p.op, m, R, R, B, jnp.int32(0), 1, low)
    consts = jnp.array([nK, nM, 1e-10, lob.C_FLOOR])
    return list(state), (p.op, m, jacobi(p.op, p.free), R, R, B, jnp.int32(0), consts)


def test_forced_p_drop():
    p = problem("hex20", (4, 4, 4), "clamped")
    state, args = start(p, 15)
    X, m = state[0], args[1]
    rng = jax.random.split(jax.random.key(1), 3)
    for noise in (1e-12, 1e-6, None):
        # P inside span(X) up to noise breaks the invariant. At 1e-12 the Cholesky of G_M fails;
        # at 1e-6 it is finite, but min diag(L)^2 is below PIVOT
        if noise:
            C, xi = jax.random.normal(rng[0], (15, 15)), m * jax.random.normal(rng[1], X.shape)
            P = X @ C + noise * xi
        else:  # a random P, M-orthonormal and M-orthogonal to X
            P = m * jax.random.normal(rng[2], X.shape)
            P = P - X @ (state[2].T @ P)
            P = m_orthonormalize(P, m * p.op.apply_M(m * P))
        KP, MP = m * p.op.apply_K(m * P), m * p.op.apply_M(m * P)
        st = [a + 0 for a in state[:3]] + [P, KP, MP, jnp.ones(15, bool), state[7] + 0]
        new, out = lob.lobpcg_step(tuple(st), *args, False, c=1)
        assert np.isfinite(np.asarray(out)).all()
        assert bool(out[-1]) == (noise is not None)
        Xn = new[0]
        G = np.asarray(Xn.T @ (m * p.op.apply_M(m * Xn)))
        assert np.abs(G - np.eye(15)).max() <= 1e-10


@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
def test_p_invariant(low):
    p = problem("hex20", (4, 4, 4), "clamped")
    state, args = start(p, 15, low=low)
    state = tuple(state)
    m = args[1]
    for it in range(30):
        if low:
            state, _ = lob.lobpcg_step_low(state, *args, c=1)
            continue
        state, _ = lob.lobpcg_step(state, *args, (it + 1) % lob.RECOMPUTE == 0, c=1)
        if it + 1 == lob.RECOMPUTE:  # the products were just recomputed explicitly
            X = state[0]
            KX = m * p.op.apply_K(m * X)
            assert np.linalg.norm(state[1] - KX) <= 1e-15 * np.linalg.norm(KX)
    X, P, live = state[0], state[-3 if low else 3], np.asarray(state[-2])
    assert live.any()
    MX, MP = m * p.op.apply_M(m * X), m * p.op.apply_M(m * P)
    assert np.abs(np.asarray(P.T @ MX)[live]).max() <= 1e-6
    G = np.asarray(P.T @ MP)[np.ix_(live, live)]
    assert np.abs(G - np.eye(live.sum())).max() <= 1e-6


def test_recompute_schedule(monkeypatch):
    p = problem("hex8", (2, 2, 2), "clamped")
    step, flags = lob.lobpcg_step, []

    def record(*args, **kw):
        flags.append(bool(args[-1]))
        return step(*args, **kw)

    monkeypatch.setattr(lob, "lobpcg_step", record)
    lobpcg(p.op, p.free, 2, tol=1e-16, maxit=40)
    assert len(flags) >= 20
    assert [i + 1 for i, f in enumerate(flags) if f] == [i for i in (20, 40) if i <= len(flags)]


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
jax.config.update("jax_enable_compilation_cache", False)
n = {"jit(lobpcg_step)": 0, "jit(lobpcg_step_low)": 0, "jit(_to_low)": 0}
def count(event, duration, **kw):
    if "backend_compile" in event and kw.get("fun_name") in n:
        n[kw["fun_name"]] += 1
jax.monitoring.register_event_duration_secs_listener(count)
from jax.tree_util import Partial
from jaxmodal.eigen import jacobi, lobpcg, sigma_estimate
from jaxmodal.materials import Material, isotropic
from jaxmodal.memory import lobpcg_memory, unstructured_memory
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import dof_mask, unstructured
m = box_mesh((3, 3, 3), type="hex20")
op = unstructured(m, {0: Material(isotropic(3.0, 0.3), 2.0)})
free = dof_mask(m, {"xmin": "xyz"})
op2 = op.scaled(s_K=jax.numpy.full(m.n_el, 2.0))
out = []
def solve(*args, **kw):
    lobpcg(*args, **kw)
    out.append(dict(n))
solve(op, free, 6, tol=1e-8)
solve(op2, free, 6, tol=1e-6, precond=jacobi(op2, free, 0.5))
sigma_estimate(op, free); out.append(dict(n))
solve(op, free, 9, tol=1e-8, chunk=3, guards=3)
solve(op2, free, 9, tol=1e-6, chunk=3, guards=3, precond=jacobi(op2, free, 0.5))
solve(op, free, 9, tol=1e-8, chunk=3, guards=3, low_memory=True)
solve(op2, free, 9, tol=1e-8, chunk=3, guards=3, low_memory=True)
zero = Partial(jax.numpy.zeros_like)  # W = 0: stuck, both switch
for o in (op, op2):
    _, _, info = lobpcg(o, free, 6, tol=1e-10, maxit=20, precond=zero)
    assert info["chunks"][0]["switched"]
    out.append(dict(n))
print(json.dumps([[d["jit(lobpcg_step)"], d["jit(lobpcg_step_low)"], d["jit(_to_low)"]]
                  for d in out]))
"""


def test_compiles():
    """One step compile per solve shape; new data (operator, preconditioner, tol) and the
    chunk count (traced) never recompile; the low-memory step is one more variant, and a switch
    adds it (and _to_low) once per shape. Measured: [[1, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
    [3, 0, 0], [3, 1, 0], [3, 1, 0], [4, 2, 1], [4, 2, 1]] (the zero preconditioner is a new
    pytree: one 9-block compile)."""
    res = subprocess.run(
        [sys.executable, "-c", COMPILES], capture_output=True, text=True, check=True
    )
    out = json.loads(res.stdout.strip().splitlines()[-1])
    nine = [o[0] for o in out]
    assert nine[:2] == [1, 1] and nine[2] - nine[1] <= 1  # sigma_estimate: one mode
    assert nine[3] - nine[2] == 1 and nine[4] == nine[3]  # chunked: 3 chunks, one compile
    assert [o[1] for o in out] == [0, 0, 0, 0, 0, 1, 1, 2, 2] and nine[6] == nine[4]
    assert [o[2] for o in out] == [0] * 7 + [1, 1] and nine[8] == nine[7]


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
    with pytest.raises(ValueError, match="n_modes and maxit"):
        lobpcg(op, free, 2, maxit=0)
    with pytest.raises(ValueError, match="n_modes and maxit"):
        lobpcg(op, free, 0)
    with pytest.raises(ValueError, match="free must be"):
        lobpcg(op, free.astype(int), 2)
    for X0 in (np.ones((op.n_dof, 8)), np.ones(op.n_dof)):  # block size 7
        with pytest.raises(ValueError, match="X0 must"):
            lobpcg(op, free, 2, X0=X0)
    R = np.zeros((op.n_dof, 1))
    R[free, 0] = 1.0  # not M-normalized
    with pytest.raises(ValueError, match="M-orthonormal"):
        lobpcg(op, free, 2, R=R)


def test_refill_limit(monkeypatch):
    p = problem("hex8", (2, 2, 2), "clamped")
    start = lob._start

    def never_full(*args, **kw):  # SVQB always drops every column
        X, KX, MX, keep = start(*args, **kw)
        return X, KX, MX, jnp.zeros_like(keep)

    monkeypatch.setattr(lob, "_start", never_full)
    with pytest.raises(RuntimeError, match="rank-deficient"):
        lobpcg(p.op, p.free, 2)


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


S = jax.ShapeDtypeStruct


def step_args(op, n, k, low, cap, basis=jnp.float64):
    """Abstract arguments of a step on op with block size k and basis capacity cap."""
    blk, vec = S((n, k), jnp.float64), S((n,), jnp.float64)
    tail = (S((k,), bool), S((k,), jnp.float64))
    state = (blk,) * 2 + tail if low else (blk,) * 6 + tail
    R = S((n, 6), jnp.float64)
    args = (state, op, S((n, 1), jnp.float64), Partial(_jacobi, vec), R, R,
            S((n, cap), basis), S((), jnp.int32), S((4,), jnp.float64))  # fmt: skip
    return args if low else args + (S((), bool),)


def step_blocks(op, n, k, low, cap, basis=jnp.float64):
    """temp + aliased state of the compiled step, in blocks of (n, k) fp64."""
    c, _ = lob._capacity(100, 1, k, low)
    f = lob.lobpcg_step_low if low else lob.lobpcg_step
    mem = f.lower(*step_args(op, n, k, low, cap, basis), c=c).compile().memory_analysis()
    return (mem.temp_size_in_bytes + mem.alias_size_in_bytes) / (n * k * 8)


@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
def test_step_memory(low):
    """Test 9: temp + aliased state with a diagonal operator (no operator workspace) is the
    same (to 0.01) for basis capacities c ceil(2m / c) and 4x that, fp64 or fp32: <= 16.5
    blocks for the 9-block step (sequenced update; 18 without barriers), <= 7.5 low-memory."""
    n, k = 100_000, 17
    vec = S((n,), jnp.float64)
    op = Diagonal(vec, vec)
    c, _ = lob._capacity(100, 1, k, low)
    cap = c * -(-2 * k // c)
    got = {
        step_blocks(op, n, k, low, x, d) for x in (cap, 4 * cap) for d in (jnp.float64, jnp.float32)
    }
    print(f"step memory ({'low' if low else '9-block'}): {got}")
    assert max(got) - min(got) <= 0.01 and max(got) <= (7.5 if low else 16.5)
    assert step_blocks(op, n, k, low, 0) <= (7.5 if low else 16.5)  # single chunk: no basis


def test_non_finite_step():
    p = problem("hex20", (4, 4, 4), "clamped")
    with pytest.raises(FloatingPointError, match="iteration 1"):
        lobpcg(p.op, p.free, 6, precond=Partial(lambda r: r * jnp.nan))


# ---------------------------------------------------------------- PR 9b: chunks and locking


@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
@pytest.mark.parametrize("bc", ["clamped", "free"])
def test_chunks(bc, low):
    """Test 1: 30 modes in 3 chunks of 10 (default maxit, per chunk: about 900 in total)."""
    p_args = ("hex20", (6, 6, 6), bc)
    p = problem(*p_args)
    lam, V, info = lobpcg(p.op, p.free, 30, tol=1e-10, R=p.R, chunk=10, low_memory=low)
    check_modes(p_args, lam, V, info, 1e-10)
    assert [(c["want"], c["converged"], c["stagnated"]) for c in info["chunks"]] == [
        (10, True, False)
    ] * 3


CUBE = ("hex20", (4, 4, 4), "free", (1.0, 1.0, 1.0))


def test_cluster_at_chunk_boundary():
    """Test 3: elastic multiplicities 2, 3, 3, 2, 1, 3, 3: chunks of 4 extend over their
    triples, and the last triple (modes 12-14) straddles n_modes, so the return is cut."""
    p = problem(*CUBE)
    lr, _ = reference(CUBE, 12)
    lam, V, info = lobpcg(p.op, p.free, 12, tol=1e-10, R=p.R, chunk=4)
    assert info["converged"] and [c["locked"] for c in info["chunks"]] == [5, 5, 4]
    assert len(lam) == len(info["residuals"]) == 12 and V.shape == (p.op.n_dof, 12)
    for g in clusters(lr, 1e-8):
        if g[-1] < 12:
            c = lr[g[0]]
            assert np.sum(np.abs(lam - c) <= f_lam(p, c) * c) == len(g)


def test_guard_error():
    """Test 4: one guard cannot show the end of the triple at the chunk boundary. A chunk that
    is never locked (a single chunk) returns instead: its tail past n_modes is cut, as in 9a."""
    p = problem(*CUBE)
    with pytest.raises(ValueError, match="guards >= 2"):
        lobpcg(p.op, p.free, 12, tol=1e-10, R=p.R, chunk=4, guards=1)
    lr, _ = reference(CUBE, 4)
    lam, _, info = lobpcg(p.op, p.free, 4, tol=1e-8, R=p.R, guards=1)
    assert len(lam) == 4 and np.allclose(lam[:2], lr[:2], rtol=1e-8)


@pytest.mark.parametrize("tol", [1e-6, 1e-10])
def test_fp32_basis(tol):
    """Tests 5 and 5b: chunked solve with an fp32 basis; at tol 1e-10 the chunks after the
    first floor-accept at the 2 u_32 term (converged and stagnated)."""
    p_args = ("hex20", (6, 6, 6), "clamped")
    p = problem(*p_args)
    lr, _ = reference(p_args, 31)
    lam, V, info = lobpcg(p.op, p.free, 30, tol=tol, chunk=10, basis_dtype="float32")
    assert info["converged"] and len(lam) == 30
    assert [c["stagnated"] for c in info["chunks"]] == [False] + [tol < 1e-7] * 2
    Vf = V[p.idx]
    assert np.abs(Vf.T @ (p.Mf @ Vf) - np.eye(30)).max() <= 1e-5
    for g in clusters(lr, 1e-4):
        if g[0] >= 30:
            break
        gap = np.abs(np.delete(lr, g)[:, None] - lr[g][None]).min() / lr[g].max()
        bound = f_lam(p, lr[g]) if gap >= 1e-2 else np.maximum(10 * tol, f_lam(p, lr[g]))
        assert np.all(np.abs(lam[g] - lr[g]) / lr[g] <= np.maximum(1e-8, bound))
    assert (lr[30] - lr[29]) / lr[29] >= 1e-3
    sigma = (lr[29] + lr[30]) / 2
    assert np.sum(lam < sigma) == count_below(p.K, p.M, sigma, p.free) == 30


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_projector(dtype):
    """Test 6: the chunked projector equals the one-pass fp64 one (to rounding: the chunks
    subtract in a different order), with a count that is not a multiple of c; its temp is
    (c + m) n 8 bytes, independent of the capacity."""
    n, k, c = 2000, 17, 17
    rng = np.random.default_rng(0)
    count = 3 * c + 5
    B = np.zeros((n, 6 * c))
    B[:, :count] = np.linalg.qr(rng.standard_normal((n, count)))[0]
    B = jnp.asarray(B, dtype)
    Y, MY = jnp.asarray(rng.standard_normal((n, k))), jnp.asarray(rng.standard_normal((n, k)))
    V = jnp.asarray(B[:, :count], jnp.float64)
    want = np.asarray(Y - V @ (V.T @ MY))
    f = jax.jit(lob._project, static_argnums=4)
    got = np.asarray(f(B, jnp.int32(count), Y, MY, c))
    assert np.abs(got - want).max() <= 1e-14 * np.abs(want).max()
    n, c = 100_000, 64
    temps = set()
    for cap in (2 * c, 8 * c):  # cap = c: the slice is the whole basis (no copy)
        args = (S((n, cap), dtype), S((), jnp.int32), *(S((n, k), jnp.float64),) * 2)
        temps.add(f.lower(*args, c).compile().memory_analysis().temp_size_in_bytes)
    assert len(temps) == 1 and temps.pop() <= 1.01 * (c + k) * n * 8


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_insert_in_place(dtype):
    """Test 7: the lock writes into the donated basis buffer."""
    B = jnp.zeros((1000, 40), dtype)
    ptr = B.unsafe_buffer_pointer()
    B2 = lob._insert(B, jnp.ones((1000, 10)), jnp.int32(10))
    assert B.is_deleted() and B2.unsafe_buffer_pointer() == ptr
    assert np.all(np.asarray(B2)[:, 10:20] == 1) and np.all(np.asarray(B2)[:, 20:] == 0)


def test_capacity():
    """Every lock writes m columns starting at count <= n_modes - 1 - (the final chunk's
    want), so the write ends stay within the capacity; the +m matters here."""
    assert lob._capacity(30, 10, 15, True) == (15, 45)
    for n_modes, chunk, k, low in ((30, 10, 15, True), (100, 7, 12, False), (12, 4, 9, False)):
        c, cap = lob._capacity(n_modes, chunk, k, low)
        assert cap % c == 0
        # worst case: every lock extends to j = m - 1 (one short of the guard error)
        count = 0
        while count + chunk < n_modes:
            assert count + k <= cap
            count += k - 1
    assert lob._capacity(12, 12, 17, False)[1] == 0  # one chunk: no basis


def test_reortho():
    """_reortho: columns past j zero, the rest M-orthonormal and M-orthogonal to B; a column
    inside span(B) shows as a small projected norm (the solve raises on it)."""
    n, k, c = 50, 4, 3
    rng = np.random.default_rng(0)
    op = Diagonal(jnp.ones(n), jnp.asarray(rng.uniform(1, 2, n)))
    m = jnp.ones((n, 1))
    Bq, _ = np.linalg.qr(rng.standard_normal((n, 5)))
    Bm = Bq / np.sqrt(np.asarray(op.dM))[:, None]  # M-orthonormal
    B = jnp.zeros((n, 6)).at[:, :5].set(Bm)
    X = jnp.asarray(rng.standard_normal((n, k)))
    Y, g = lob._reortho(X, 3, op, m, B, jnp.int32(5), c=c)
    MY = np.asarray(op.dM)[:, None] * np.asarray(Y)
    assert np.all(np.asarray(Y)[:, 3:] == 0) and g > 0.5
    assert np.abs(MY[:, :3].T @ np.asarray(Y)[:, :3] - np.eye(3)).max() < 1e-13
    assert np.abs(Bm.T @ MY).max() < 1e-13
    x0 = np.asarray(X[:, 0]) - Bm @ (Bm.T @ (np.asarray(op.dM) * np.asarray(X[:, 0])))
    y0 = np.asarray(Y[:, 0])
    cos = abs(x0 @ y0) / np.linalg.norm(x0) / np.linalg.norm(y0)
    assert abs(cos - 1) < 1e-12  # column order kept
    _, g = lob._reortho(X.at[:, 1].set(B[:, 2]), 3, op, m, B, jnp.int32(5), c=c)
    assert g < 1e-20
    # columns 1e-6 off span(B): one projection pass leaves 7e-11, two leave 1e-16
    rng2 = np.random.default_rng(1)
    Xc = jnp.asarray(Bm @ rng2.standard_normal((5, k)) + 1e-6 * rng2.standard_normal((n, k)))
    Y, _ = lob._reortho(Xc, 3, op, m, B, jnp.int32(5), c=c)
    assert np.abs(Bm.T @ (np.asarray(op.dM)[:, None] * np.asarray(Y))).max() < 1e-13


def test_lock_in_span(monkeypatch):
    """A locked column inside span(B) (projected norm below 0.5) raises."""
    p = problem("hex20", (4, 4, 4), "clamped")
    real = lob._reortho

    def zero_norm(*a, **kw):
        return real(*a, **kw)[0], 0.0

    monkeypatch.setattr(lob, "_reortho", zero_norm)
    with pytest.raises(RuntimeError, match="span of the earlier"):
        lobpcg(p.op, p.free, 4, chunk=2)


def test_lock_full():
    """Test 8: a lock past the capacity raises before writing (dynamic_update_slice would
    clamp the start onto earlier columns) and leaves the basis intact."""
    B = jnp.asarray(np.random.default_rng(0).standard_normal((100, 40)))
    before = np.asarray(B).copy()
    with pytest.raises(RuntimeError, match="locked basis full"):
        lob._lock(B, 35, jnp.ones((100, 10)), 10)
    assert not B.is_deleted() and np.array_equal(np.asarray(B), before)
    B = lob._lock(B, 30, jnp.ones((100, 10)), 4)  # exactly full; unlocked columns zeroed
    assert np.all(np.asarray(B)[:, 30:34] == 1) and np.all(np.asarray(B)[:, 34:] == 0)


@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
@pytest.mark.parametrize(
    "t,s,k", [("hex20", 24, 15), ("hex20", 24, 48), ("hex20", 8, 15), ("hex8", 20, 24),
              ("hex27", 12, 24)],
)  # fmt: skip
def test_step_within_estimate(t, s, k, low):
    """Test 10: compiled step temp + aliased state <= lobpcg_memory's step term (hex20: fit
    points, 8^3 where operator workspace dominates; hex8 20^3 and hex27 12^3 at m = 24: held
    out of the fit)."""
    mesh = box_mesh((s, s, s), type=t)
    op = unstructured(mesh, MAT)
    n = op.n_dof
    got = step_blocks(op, n, k, low, 4 * k)
    est = lobpcg_memory(lambda m: unstructured_memory(mesh, m), n, k, chunk=k, guards=0,
                        low_memory=low)["step"]  # fmt: skip
    print(f"{t} {s}^3 m={k} low={low}: {got:.2f} blocks, estimate {est / (8 * n * k):.2f}")
    assert got * 8 * n * k <= est


PEAK_SETUP = """
from jaxmodal import config
config.enable_x64()
from jaxmodal.eigen import lobpcg
from jaxmodal.memory import lobpcg_memory, unstructured_memory
from jaxmodal.mesh import box_mesh
from jaxmodal.materials import Material, isotropic
from jaxmodal.operators import dof_mask, unstructured
mats = {0: Material(isotropic(3.0, 0.3), 2.0)}
mesh = box_mesh((44, 44, 44), type="hex20")
free = dof_mask(mesh, {"xmin": "xyz"})
op = unstructured(mesh, mats)  # warm-up: full run, freed below
lobpcg(op, free, 40, guards=8, maxit=3, low_memory=LOW)
del op
"""
PEAK = """
op = unstructured(mesh, mats)
lobpcg(op, free, 40, guards=8, maxit=3, low_memory=LOW)
"""


@pytest.mark.peak
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
def test_estimate_1m(low):
    """Test 11: hex20 44^3 (1.08M DOF), 40 modes in one chunk (block 48), 3 iterations:
    measured peak within 20% of lobpcg_memory's total. Test 11b covers locking."""
    from benchmarks import harness

    mesh = box_mesh((44, 44, 44), type="hex20")
    n = 3 * len(mesh.points)
    est = lobpcg_memory(lambda m: unstructured_memory(mesh, m), n, 40, guards=8,
                        low_memory=low)["total"]  # fmt: skip
    peak = harness.measure(PEAK.replace("LOW", str(low)), PEAK_SETUP.replace("LOW", str(low)))
    peak = peak["peak_bytes"]
    print(f"1M low={low}: estimate {est / 1e9:.2f} GB, peak {peak / 1e9:.2f} GB")
    assert est >= 2e9 and abs(est - peak) <= 0.2 * peak, (est, peak)


CHUNKS_SETUP = """
import dataclasses
from jaxmodal import config
config.enable_x64()
from jaxmodal.eigen import lobpcg
from jaxmodal.operators.common import register
@dataclasses.dataclass(frozen=True)
class Diagonal:
    dK: jax.Array
    dM: jax.Array
    n_dof = property(lambda self: self.dK.shape[0])
    apply_K = lambda self, v: self.dK[:, None] * v
    apply_M = lambda self, v: self.dM[:, None] * v
    diag_K = lambda self: self.dK
    diag_M = lambda self: self.dM
register(Diagonal, [])
n = N_DOF
d = np.full(n, 1e6)
d[:400] = 1 + 0.37 * np.arange(400)
op = Diagonal(jnp.asarray(np.random.default_rng(0).permutation(d)), jnp.ones(n))
free = np.ones(n, bool)
lobpcg(op, free, 30, tol=1e-8, chunk=10, basis_dtype=DT)  # warm-up compiles
"""
CHUNKS = """
lam, V, info = lobpcg(op, free, 100, tol=1e-8, chunk=10, basis_dtype=DT)
assert info["converged"] and len(lam) == 100
"""


@pytest.mark.peak
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_estimate_chunks(dtype):
    """Test 11b: a diagonal pencil (300k DOF) solved in 10 locked chunks: the measured peak,
    including the end of the solve, is 0.8-1.1 x lobpcg_memory's total (the peak varies
    between runs, 0.95-1.07 x the estimate; roadmap band +-20%, docs/open_items.md)."""
    from benchmarks import harness

    n = 300_000
    op_mem = lambda m: dict(storage=16 * n, workspace=0, build=0, overhead=10e6)  # noqa: E731
    est = lobpcg_memory(op_mem, n, 100, chunk=10, basis_dtype=dtype)["total"]
    sub = {"N_DOF": str(n), "DT": repr(dtype)}
    code, setup = CHUNKS, CHUNKS_SETUP
    for key, val in sub.items():
        code, setup = code.replace(key, val), setup.replace(key, val)
    peak = harness.measure(code, setup)["peak_bytes"]
    print(f"chunks {dtype}: estimate {est / 1e9:.2f} GB, peak {peak / 1e9:.2f} GB")
    assert 0.8 * est <= peak <= 1.1 * est, (est, peak)


@pytest.mark.slow
def test_many_chunks():
    """30 locks on a diagonal pencil whose Jacobi residuals are dominated by the locked
    coordinates (up to 1e6 x): without re-orthonormalizing at lock the basis loses M-
    orthonormality (1e-9) and chunk 9 drifts into span(B) and fails."""
    n = 20_000
    d = np.full(n, 1e6)
    d[:1200] = 1 + 0.37 * np.arange(1200)
    op = Diagonal(jnp.asarray(np.random.default_rng(0).permutation(d)), jnp.ones(n))
    lam, V, info = lobpcg(op, np.ones(n, bool), 300, tol=1e-8, chunk=10)
    assert info["converged"] and len(info["chunks"]) == 30
    assert np.abs(V.T @ V - np.eye(300)).max() <= 1e-13
    np.testing.assert_allclose(lam, np.sort(d)[:300], rtol=0, atol=1e-9)


def test_failed_chunk():
    """Test 13: chunk 1 converges at once from the reference modes, chunk 2 hits maxit: the
    solve stops and returns the 10 locked modes plus chunk 2's 10 Ritz vectors."""
    p_args = ("hex20", (6, 6, 6), "clamped")
    p = problem(*p_args)
    lr, Vr = reference(p_args, 15)
    X0 = np.zeros((p.op.n_dof, 15))
    X0[p.idx] = Vr[:, :15]  # the reference returns the free rows
    lam, V, info = lobpcg(p.op, p.free, 30, tol=1e-10, chunk=10, X0=X0, maxit=5)
    assert len(lam) == 20 and not info["converged"]
    assert [c["converged"] for c in info["chunks"]] == [True, False]
    assert np.all(np.abs(lam[:10] - lr[:10]) / lr[:10] <= f_lam(p, lr[:10]))
    Vf = V[p.idx]
    assert np.abs(Vf.T @ (p.Mf @ Vf) - np.eye(20)).max() <= 1e-8
    assert np.allclose(info["residuals"], residuals(p, lam, jnp.asarray(V)), rtol=1e-5)


def test_chunk_errors():
    """Test 14."""
    p = problem("hex8", (2, 2, 2), "clamped")
    for kw in (dict(chunk=0), dict(chunk=5), dict(guards=0)):
        with pytest.raises(ValueError, match="chunk <= n_modes"):
            lobpcg(p.op, p.free, 4, **kw)
    with pytest.raises(ValueError, match="basis_dtype"):
        lobpcg(p.op, p.free, 4, chunk=2, basis_dtype="float16")
    m = box_mesh((1, 1, 1), type="hex8")
    op = unstructured(m, MAT)
    free = dof_mask(m, {"xmin": "xyz"})  # 12 free DOFs
    with pytest.raises(ValueError, match="block size 13"):
        lobpcg(op, free, 10, chunk=3, guards=10)


def test_precond_composition():
    """PR 11b test 9: precond_memory adds the preconditioner's storage and step bytes and enters
    its build into the max; solve_modes_memory is the composition of the three estimators."""
    mesh = voxel_mesh((12, 10, 8), (0.1,) * 3)
    n = 3 * len(mesh.points)

    def op_mem(m):
        return stencil_memory(mesh, m)

    for dtype in ("float64", "float32"):
        pc = functools.partial(gmg_memory, mesh, dtype=dtype)
        g, o = pc(15), op_mem(15)  # block 15 for 10 modes
        for low in (False, True):
            a = lobpcg_memory(op_mem, n, 10, low_memory=low)
            b = lobpcg_memory(op_mem, n, 10, low_memory=low, precond_memory=pc)
            assert b["step"] - a["step"] == g["step_low" if low else "step"]
            assert b["precond"] == g["storage"] and a["precond"] == 0
            extra = (
                g["storage"] + max(o["build"], g["build"], b["step"]) - max(o["build"], a["step"])
            )
            assert abs(b["total"] - a["total"] - extra) <= 1
        big = lobpcg_memory(op_mem, n, 10, precond_memory=lambda m, g=g: g | dict(build=10**12))
        assert 10**12 < big["total"] < 10**12 + b["total"]  # a dominant precond build counts
    for bcs, nr in (({"xmin": "xyz"}, 0), ({}, 6)):
        for precond, dtype in (("auto", "float64"), ("gmg", "float32"), ("jacobi", "float64")):
            got = solve_modes_memory(mesh, bcs, 10, precond=precond, precond_dtype=dtype)
            pc = None if precond == "jacobi" else functools.partial(gmg_memory, mesh, dtype=dtype)
            assert got == lobpcg_memory(op_mem, n, 10, n_rigid=nr, precond_memory=pc)
    box = box_mesh((3, 3, 3), type="hex20")
    um = functools.partial(unstructured_memory, box)
    n_box, amg_pc = 3 * len(box.points), functools.partial(amg_memory, box)
    assert solve_modes_memory(box, {}, 4) == lobpcg_memory(
        um, n_box, 4, n_rigid=6, precond_memory=amg_pc
    )
    got = solve_modes_memory(box, {}, 4, precond="jacobi")
    assert got == lobpcg_memory(um, n_box, 4, n_rigid=6)


# ---------------------------------------------------------------- PR 18b: switch, polish, RR

PENCIL = ("hex20", (4, 4, 4), "clamped")
ZERO = Partial(jnp.zeros_like)  # W = 0: only X and P move, the chunk stalls


@functools.cache
def dense_modes(p_args):
    p = problem(*p_args)
    return sl.eigh(p.Kf.toarray(), p.Mf.toarray())


def near_start(p_args, k, delta, cols=None, seed=0):
    """The k lowest exact eigenvectors (full rows) plus delta V_hi G / sqrt(n_hi) on cols."""
    p = problem(*p_args)
    _, U = dense_modes(p_args)
    nr = 0 if p.R is None else p.R.shape[1]
    hi = U[:, U.shape[1] // 2 :]
    G = np.random.default_rng(seed).standard_normal((hi.shape[1], k))
    X = U[:, nr : nr + k].copy()
    cols = range(k) if cols is None else cols
    X[:, cols] += delta * (hi @ G[:, cols]) / np.sqrt(hi.shape[1])
    X0 = np.zeros((p.op.n_dof, k))
    X0[p.idx] = X
    return X0


def exact_polish(p_args, calls=None):
    """Y = (K_f + lam_1 M_f)^-1 M_f X on the free rows: an exact inverse-iteration step."""
    p = problem(*p_args)
    lu = sl.lu_factor((p.Kf + dense_modes(p_args)[0][0] * p.Mf).toarray())

    def polish(X):
        if calls is not None:
            calls.append(X.shape)
        Y = np.zeros(X.shape)
        Y[p.idx] = sl.lu_solve(lu, p.Mf @ np.asarray(X)[p.idx])
        return jnp.asarray(Y), 7

    return polish


def stalled(monkeypatch, polish=None, n=6, tol=1e-10, **kw):
    """A STALL exit on the PR 9 pencil from a near-exact start (zero preconditioner)."""
    monkeypatch.setattr(lob, "STALL", 3)
    p = problem(*PENCIL)
    k = lob._block_size(kw.get("chunk") or n, kw.get("chunk"), None)[2]
    X0 = near_start(PENCIL, k, 1e-2 * tol)
    return lobpcg(p.op, p.free, n, tol, precond=ZERO, X0=X0, polish=polish, **kw)


@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
def test_polish_accepts(monkeypatch, low):
    *_, info0 = stalled(monkeypatch, low_memory=low)
    assert not info0["converged"] and info0["stagnated"]
    assert info0["chunks"][0]["polish"] is None
    calls = []
    lam, V, info = stalled(monkeypatch, exact_polish(PENCIL, calls), low_memory=low)
    assert info["converged"] and info["stagnated"] and info["chunks"][0]["polish"] == 7
    assert len(calls) == 1 and info["iterations"] == info0["iterations"]
    assert np.all(info["residuals"] <= info["thresholds"])
    p, lr = problem(*PENCIL), dense_modes(PENCIL)[0][:6]
    assert np.all(np.abs(lam - lr) / lr <= f_lam(p, lr))
    Vf = V[p.idx]
    assert np.abs(Vf.T @ (p.Mf @ Vf) - np.eye(6)).max() <= 1e-12


@pytest.mark.parametrize("bad", ["noise", "rank", "nan"])
def test_polish_reverts(monkeypatch, bad):
    """A polish that does not pass leaves the result bitwise equal to no polish."""
    lam0, V0, info0 = stalled(monkeypatch)
    exact = exact_polish(PENCIL)

    def polish(X):
        Y, its = exact(X)
        if bad == "noise":
            return X + jax.random.normal(jax.random.key(3), X.shape) * (X != 0), its
        if bad == "nan":
            return Y * jnp.nan, its
        return Y.at[:, 1].set(Y[:, 0]), its

    rr, real = [], lob._rr_block
    monkeypatch.setattr(lob, "_rr_block", lambda *a: rr.append(1) or real(*a))
    lam, V, info = stalled(monkeypatch, polish)
    assert not info["converged"] and info["chunks"][0]["polish"] == 7
    assert len(rr) == (1 if bad == "nan" else 2)  # a NaN Y never reaches _rr_block
    assert np.array_equal(lam, lam0) and np.array_equal(V, V0)
    assert np.array_equal(info["residuals"], info0["residuals"])


def test_polish_not_on_maxit(monkeypatch):
    calls = []
    _, _, info = stalled(monkeypatch, exact_polish(PENCIL, calls), maxit=2)
    assert not info["converged"] and not info["stagnated"] and not calls
    assert info["chunks"][0]["polish"] is None


@pytest.fixture
def extent_calls(monkeypatch):
    """The failed= flag of every _extent call."""
    extent, calls = lob._extent, []

    def spy(*a, **kw):
        calls.append(kw.get("failed", False))
        return extent(*a, **kw)

    monkeypatch.setattr(lob, "_extent", spy)
    return calls


def test_polish_guard_check(monkeypatch, extent_calls):
    """Locked chunks (12 modes, chunks of 6): the failed-exit guard check (_extent with
    failed=True) runs after a polish that did not accept, never after one that did."""
    failed = extent_calls
    _, _, info = stalled(monkeypatch, lambda X: (X, 1), n=12, chunk=6)
    assert not info["converged"] and True in failed
    failed.clear()
    _, _, info = stalled(monkeypatch, exact_polish(PENCIL), n=12, chunk=6, maxit=60)
    assert info["chunks"][0]["converged"] and info["chunks"][0]["polish"] == 7
    assert True not in failed


@pytest.mark.parametrize("want", [1, 4])
def test_polish_extends_cluster(monkeypatch, extent_calls, want):
    """The cube's clusters [0, 1] and [2, 3, 4]: noise on the whole multiplet, so only the
    polish (here an oracle returning the exact block) shows the cluster and extends j."""
    monkeypatch.setattr(lob, "STALL", 3)
    p = problem(*CUBE)
    k = lob._block_size(want, None, None)[2]
    cols = [0, 1] if want == 1 else [2, 3, 4]
    X0 = near_start(CUBE, k, 1e-2, cols)
    calls = extent_calls
    kw = dict(R=p.R, precond=ZERO, X0=X0)
    _, _, info = lobpcg(p.op, p.free, want, 1e-6, **kw)
    assert not info["converged"] and False not in calls
    exact = jnp.asarray(near_start(CUBE, k, 0.0))
    _, _, info = lobpcg(p.op, p.free, want, 1e-6, polish=lambda X: (exact, 1), **kw)
    assert info["converged"] and info["chunks"][0]["locked"] > want
    assert info["chunks"][0]["polish"] is not None


@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
def test_t_iteration_switch(monkeypatch, low):
    monkeypatch.setattr(lob, "SWITCH", 2)
    _, _, info = stalled(monkeypatch, low_memory=low)
    assert (info["chunks"][0]["switched"] is None) == low
    assert low or info["chunks"][0]["switched"] >= 2
    assert np.isfinite(info["t_iteration"]) and info["t_iteration"] > 0


def test_final_rr(monkeypatch):
    """A STALL exit returns the Rayleigh-Ritz block of the last X (Ritz values ascending, V^T K
    V diagonal); an SVQB drop in _final_rr's own _start leaves X and lam as they were and
    is never accepted."""
    p = problem(*PENCIL)
    lam, V, info = stalled(monkeypatch)
    assert not info["converged"] and np.all(np.diff(lam) >= 0)
    Vf = V[p.idx]
    G = Vf.T @ (p.Kf @ Vf)
    assert np.abs(G - np.diag(np.diag(G))).max() <= 1e-10 * np.abs(np.diag(G)).max()
    assert np.allclose(np.diag(G) / np.einsum("ij,ij->j", Vf, p.Mf @ Vf), lam, rtol=1e-12)
    final, start, accept, n_start, seen = lob._final_rr, lob._start, lob._accept, [], []

    def drop(*a, **kw):  # every call after _begin's (_final_rr's): one column dropped
        X, KX, MX, keep = start(*a, **kw)
        n_start.append(1)
        return X, KX, MX, keep.at[0].set(len(n_start) == 1)

    def spy(X, lam, *a):
        seen.append((np.asarray(X), lam))
        return final(X, lam, *a)

    def accepts(*a):  # never reached: the dropped block is not accepted
        seen.append("_accept")
        return accept(*a)

    monkeypatch.setattr(lob, "_start", drop)
    monkeypatch.setattr(lob, "_final_rr", spy)
    monkeypatch.setattr(lob, "_accept", accepts)
    lam, V, info = stalled(monkeypatch)
    ((X_in, lam_in),) = seen
    assert len(n_start) == 2 and not info["converged"]
    assert np.array_equal(lam, lam_in[: len(lam)]) and np.array_equal(V, X_in[:, : len(lam)])


@pytest.mark.parametrize("maxit", [300, 160], ids=["stall", "maxit"])
def test_final_rr_accepts(monkeypatch, maxit):
    """In-loop explicit checks always fail (as under drift): the failed exit's Rayleigh-Ritz
    accepts, after a STALL exit (235 iterations, switched at 165) or a maxit exit at 160."""
    real = lob._explicit
    monkeypatch.setattr(
        lob, "_explicit", lambda s, *a: (lambda o: (o[0], o[1] + np.inf, o[2]))(real(s, *a))
    )
    p = problem(*PENCIL)
    lam, V, info = lobpcg(p.op, p.free, 6, 1e-8, maxit=maxit)
    assert info["converged"] and info["chunks"][0]["polish"] is None
    assert info["stagnated"] == (maxit == 300) and info["iterations"] == (
        235 if maxit == 300 else 160
    )
    assert np.allclose(lam, dense_modes(PENCIL)[0][:6], rtol=1e-10)


def test_no_rr_or_switch_with_stop(monkeypatch):
    """sigma_estimate (a stop callback) neither switches nor runs _final_rr, also on maxit;
    SWITCH = 1 so that any iteration without progress would switch."""
    monkeypatch.setattr(lob, "SWITCH", 1)
    calls = []
    for name in ("_final_rr", "_to_low"):
        real = getattr(lob, name)
        monkeypatch.setattr(
            lob, name, lambda *a, _r=real, _n=name, **kw: calls.append(_n) or _r(*a, **kw)
        )
    p = problem(*PENCIL)
    sigma_estimate(p.op, p.free)
    assert sigma_estimate(p.op, p.free, maxit=10)[1] == 10  # a maxit exit
    # a stop that never fires: without the guard this run switches (at 147) and its maxit
    # exit gets _final_rr
    info = lob._solve(p.op, p.free, 1, 1e-6, None, None, None, 170, 0, stop=lambda s: False)[2]
    assert info["iterations"] == 170 and not calls
    lob.lobpcg(p.op, p.free, 1, 1e-6, maxit=170)
    assert calls == ["_to_low", "_final_rr"]  # the same run without stop


def test_project_clamp():
    """Capacity 10, c = 4: the last chunk starts at cap - c and masks the columns before i c."""
    n, k, cap, c = 200, 3, 10, 4
    rng = np.random.default_rng(0)
    f = jax.jit(lob._project, static_argnums=4)
    for count in (8, 9, 10):
        B = np.zeros((n, cap))
        B[:, :count] = np.linalg.qr(rng.standard_normal((n, count)))[0]
        Y = jnp.asarray(rng.standard_normal((n, k)))
        Z = f(jnp.asarray(B), jnp.int32(count), Y, Y, c)
        assert np.abs(B.T @ np.asarray(Z)).max() <= 1e-13, count
    B = jnp.asarray(np.linalg.qr(rng.standard_normal((n, 12)))[0])  # c divides 12: unmasked

    def plain(B, count, Y, MY, c):
        def body(i, Y):
            Bc = jax.lax.dynamic_slice_in_dim(B, i * c, c, 1)
            return Y - Bc @ (Bc.T @ MY)

        return jax.lax.fori_loop(0, (count + c - 1) // c, body, Y)

    Y = jnp.asarray(rng.standard_normal((n, k)))
    assert np.array_equal(
        f(B, jnp.int32(10), Y, Y, 4), jax.jit(plain, static_argnums=4)(B, jnp.int32(10), Y, Y, 4)
    )


def test_to_low():
    """_to_low removes a locked-basis component from X and P and keeps the rest."""
    p = problem(*PENCIL)
    n = p.op.n_dof
    m = jnp.asarray(p.free, float)[:, None]
    _, U = dense_modes(PENCIL)
    B = np.zeros((n, 8))
    B[p.idx, :4] = U[:, :4]
    B = jnp.asarray(B)
    Q = np.zeros((n, 6))
    Q[p.idx] = U[:, 10:16]
    X = jnp.asarray(Q) + 1e-6 * B[:, :1]
    P = jnp.asarray(Q[:, ::-1]) + 1e-6 * B[:, 1:2]
    R = jnp.zeros((n, 0))
    X, P = lob._to_low(X, P, p.op, m, R, R, B, jnp.int32(4), c=4)
    for Y, Z in ((X, Q), (P, Q[:, ::-1])):
        assert np.abs(np.asarray(B).T @ np.asarray(p.op.apply_M(Y))).max() <= 1e-13
        assert np.abs(np.asarray(Y) - Z).max() <= 1e-12


def test_polish_live_bytes(monkeypatch):
    """At the polish only the chunk's X (and the basis) is live: a leaked KX, MX or P fails."""
    monkeypatch.setattr(lob, "STALL", 3)
    n = 200_000
    op = Diagonal(jnp.asarray(1 + np.arange(n) / n), jnp.ones(n))
    seen = []

    def live():
        gc.collect()
        return sum(a.nbytes for a in jax.live_arrays())

    def polish(X):
        seen.append(live())
        return X, 1

    base = live()
    _, _, info = lobpcg(op, np.ones(n, bool), 1, 1e-10, precond=ZERO, polish=polish)
    k = 6  # block of one mode
    assert info["chunks"][0]["polish"] == 1 and info["stagnated"]
    assert seen[0] - base <= 1.5 * 8 * n * k, (seen[0] - base) / (8 * n * k)


def _converging(name):
    """A representative converging solve: (lam, info)."""
    from jaxmodal.eigen import solve_modes
    from jaxmodal.precond import pmg

    if name == "pmg":
        p = problem(*PENCIL)
        lam, _, info = lobpcg(p.op, p.free, 6, 1e-8, precond=pmg(p.op, p.free, 1.0))
        return lam, info
    if name == "gmg":
        mesh, bcs = voxel_mesh((10, 8, 6), (0.1,) * 3), {"xmin": "xyz"}
    elif name == "amg-gap":  # test_gap_band's pencil
        mesh, bcs = box_mesh((6, 3, 3), (2 / 6, 1 / 3, 1.002 / 3), type="hex20"), {"xmin": "xyz"}
    else:
        mesh, bcs = box_mesh((4, 4, 4), (0.25,) * 3, type="hex20"), {"xmin": "xyz"}
    lam, _, info = solve_modes(mesh, MAT, bcs, 6, tol=1e-10 if name == "amg-gap" else 1e-8)
    return lam, info


@pytest.mark.parametrize("name", ["amg-gap", "amg-box", "gmg", "pmg"])
def test_no_switch_when_converging(monkeypatch, name):
    """Converging AMG / GMG / p-MG chunks never switch: bitwise equal to SWITCH = infinity."""
    lam, info = _converging(name)
    assert info["converged"] and all(c["switched"] is None for c in info["chunks"])
    monkeypatch.setattr(lob, "SWITCH", 10**9)
    assert np.array_equal(_converging(name)[0], lam)


def test_band_clamp(monkeypatch):
    """Band mode, chunks of 40 (block 48, c = 64, capacity 128): the 4th chunk starts at count
    120 with ceil(120 / 48) 48 = 144 > 128 and switches (the clamped _project); an oracle polish
    (the exact block) locks every chunk: no mode twice, the band complete. A wiring check: the
    clamp's arithmetic is test_project_clamp's (the oracle polish hides an unclamped one)."""
    monkeypatch.setattr(lob, "STALL", 3)
    monkeypatch.setattr(lob, "SWITCH", 1)
    n = 3000
    d = 1 + np.arange(n, dtype=float)
    op = Diagonal(jnp.asarray(d), jnp.ones(n))
    calls, seen, real = [], [], lob._iterate

    def oracle(X):  # chunk i (40 locked per chunk): the unit vectors 40 i ... 40 i + 47
        i = len(calls)
        calls.append(i)
        return jnp.eye(n, 48, -40 * i), 1

    def spy(state, args, *a):
        out = real(state, args, *a)
        B, cnt, k = args[5], int(args[6]), state[0].shape[1]
        seen.append((out[-1]["switched"], -(-cnt // k) * k > B.shape[1]))
        return out

    monkeypatch.setattr(lob, "_iterate", spy)
    lam, V, info = lob._solve(op, np.ones(n, bool), 80, 1e-8, None, ZERO, None, 100, 0,
                              chunk=40, guards=8, lam_stop=160.5, max_capacity=1024,
                              polish=oracle)  # fmt: skip
    assert seen[3][0] is not None and seen[3][1]
    assert info["n_band"] == 160 and np.allclose(lam[:160], d[:160], rtol=1e-12)
    W = V[:, :160]
    assert np.abs(W.T @ W - np.eye(160)).max() <= 1e-12
