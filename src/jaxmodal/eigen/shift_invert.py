"""Block shift-invert Lanczos for the eigenvalues of one slice [lo, hi] of the masked pencil
(K_m, M_m): inner MINRES on K - sigma M with an SPD preconditioner of K + sigma M, full
M-reorthogonalization, Rayleigh-Ritz on (K, M). See docs/eigensolvers.md#shift-invert-slices."""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..linalg.krylov import minres
from ..operators import ap_operator
from ..operators.common import check_x64
from .counting import _check_ortho
from .lobpcg import _K, _M, C_FLOOR, _deflate, _insert, _norms, _resid, _svqb_t

SIGMA_MOVE = 0.01  # sigma moves by this fraction of the slice width when it hits an eigenvalue
STAGNATION = 1e3  # first-solve residual (relative to tau) above which sigma moves
INNER_GROWTH = 10  # later inner solves are capped at this multiple of the first one
TAU_FAIL = 1e-4  # tau_eff above this ends the slice ("inner solves failed")
REFILLS = 3  # random refills of dropped columns before a breakdown error


def _chunk(b):
    """Projector and Ritz chunk width c for block b (as LOBPCG's)."""
    return max(b, min(4 * b, 64))


@functools.partial(jax.jit, static_argnames="c")
def _extend(Y, op, m, R, MR, Q, count, c):
    """Y masked, M-orthogonalized against R and Q[:, :count] and M-orthonormalized by SVQB, two
    rounds: (Y, keep), dropped columns zero."""
    keep = jnp.ones(Y.shape[1], bool)
    for _ in range(2):
        Y = _deflate(op, m, R, MR, Q, count, Y, c)
        T, keep = _svqb_t(Y.T @ _M(op, m, Y), keep)
        Y = Y @ T
    return Y, keep


@functools.partial(jax.jit, donate_argnums=0, static_argnames="c")
def _append(Q, Y, op, m, L0, k, c):
    """Q with Y written at Lanczos column k (Q is donated) and the new projected stiffness
    columns G[:, k : k + b] = Q_L^T K Y over the Lanczos columns Q_L = Q[:, L0:], in chunks
    (rows past k + b are zero)."""
    Q = jax.lax.dynamic_update_slice_in_dim(Q, Y, L0 + k, 1)
    KY = _K(op, m, Y)

    def body(i, G):
        Qc = jax.lax.dynamic_slice_in_dim(Q, L0 + i * c, c, 1)
        return jax.lax.dynamic_update_slice_in_dim(G, Qc.T @ KY, i * c, 0)

    G = jax.lax.fori_loop(
        0, (k + Y.shape[1] + c - 1) // c, body, jnp.zeros((Q.shape[1], Y.shape[1]))
    )
    return Q, G


@functools.partial(jax.jit, static_argnames="c")
def _combine(Q, L0, k, Yc, c):
    """Q_L Yc over the first k Lanczos columns, in chunks of c (Yc rows past k are zero)."""

    def body(i, X):
        Qc = jax.lax.dynamic_slice_in_dim(Q, L0 + i * c, c, 1)
        return X + Qc @ jax.lax.dynamic_slice_in_dim(Yc, i * c, c, 0)

    return jax.lax.fori_loop(0, (k + c - 1) // c, body, jnp.zeros((Q.shape[0], Yc.shape[1])))


@functools.partial(jax.jit, static_argnames="c")
def _ritz(Q, L0, k, Yc, lam, op, m, consts, c):
    """Residuals and round-off thresholds of the Ritz vectors Q_L Yc (lobpcg._resid)."""
    X = _combine(Q, L0, k, Yc, c)
    return _resid(X, _K(op, m, X), _M(op, m, X), lam, consts)


@jax.jit
def _explicit(X, lam, op, m, consts):
    return _resid(X, _K(op, m, X), _M(op, m, X), lam, consts)


def _pad(Y, rows, w):
    """Columns of Y (k, <= w) in a zero (rows, w) block."""
    out = np.zeros((rows, w))
    out[: Y.shape[0], : Y.shape[1]] = Y
    return jnp.asarray(out)


def _form_v(Q, L0, Y, c):
    """Host V = Q_L Y (fp64), formed in chunks of c columns: each device chunk is copied to the
    host and dropped, so only V and one chunk coexist with Q."""
    V = np.empty((Q.shape[0], Y.shape[1]))
    for j in range(0, Y.shape[1], c):
        w = min(c, Y.shape[1] - j)
        X = _combine(Q, L0, Y.shape[0], _pad(Y[:, j : j + w], Q.shape[1], c), c=c)
        V[:, j : j + w] = np.asarray(X)[:, :w]
    return V


def _chunked(f, cols, lam, b, sigma, tau_eff):
    """(r, t) of the columns cols with Ritz values lam, in chunks of b: f(cols_chunk, lam_b)
    gives the device (r, t) at width b (padded columns lam = 1, ignored); t raised to the
    inexact-solve term 10 tau_eff delta."""
    rr, tt = np.zeros(len(cols)), np.zeros(len(cols))
    for j in range(0, len(cols), b):
        w = min(b, len(cols) - j)
        lc = np.ones(b)
        lc[:w] = lam[j : j + w]
        a, t = f(cols[j : j + w], jnp.asarray(lc))
        rr[j : j + w], tt[j : j + w] = np.asarray(a)[:w], np.asarray(t)[:w]
    return rr, np.maximum(tt, 10 * tau_eff * np.abs(lam - sigma) / np.abs(lam))


def _ritz_chunk(Q, L0, k, Yr, op, m, consts, c, cols, lam):
    Yc = _pad(Yr[:, cols], Q.shape[1], len(lam))
    return _ritz(Q, L0, k, Yc, lam, op, m, consts, c=c)


def _v_chunk(V, op, m, consts, cols, lam):
    X = np.zeros((V.shape[0], len(lam)))
    X[:, : len(cols)] = V[:, cols]
    return _explicit(jnp.asarray(X), lam, op, m, consts)


def _cands(f, mu, b, sigma, tau_eff, cache, idx):
    """_slice_check's resid: (r, t) of the Ritz pairs idx, also stored in cache."""
    rr, tt = _chunked(f, idx, mu[idx], b, sigma, tau_eff)
    cache.update(zip(idx.tolist(), zip(rr, tt, strict=True), strict=True))
    return rr, tt


def _slice_sets(mu, lo, hi, r):
    """(inside, below, above) of the ascending Ritz values mu for the slice [lo, hi]: inside =
    the values in [lo (1 - 100 r), hi (1 + 100 r)] extended over chains with relative gaps
    < 100 r on both sides; below / above = the nearest value beyond the extended set (None if
    there is none, or for below when lo = 0)."""
    i0 = int(np.searchsorted(mu, lo * (1 - 100 * r), side="left"))
    i1 = int(np.searchsorted(mu, hi * (1 + 100 * r), side="right"))
    if i1 > i0:
        while i0 > 0 and mu[i0] - mu[i0 - 1] < 100 * r * abs(mu[i0 - 1]):
            i0 -= 1
        while i1 < len(mu) and mu[i1] - mu[i1 - 1] < 100 * r * abs(mu[i1 - 1]):
            i1 += 1
    below = i0 - 1 if i0 > 0 and lo > 0 else None
    above = i1 if i1 < len(mu) else None
    return np.arange(i0, i1), below, above


def _slice_check(mu, lo, hi, r, resid):
    """(done, inside, below, above, r_acc): the sets of mu at r, re-formed with r = r_acc while
    r_acc (the largest threshold over the candidates, inside plus the neighbours) exceeds r.
    resid(idx) returns (residuals, thresholds) of the new candidate indices idx. done: every
    candidate converged, every needed neighbour present and r_acc <= r."""
    res = {}
    while True:
        inside, below, above = _slice_sets(mu, lo, hi, r)
        cand = [*inside, *(i for i in (below, above) if i is not None)]
        new = np.array([i for i in cand if i not in res], int)
        if len(new):
            rr, tt = resid(new)
            res.update(zip(new.tolist(), zip(rr, tt, strict=True), strict=True))
        r_acc = max((res[i][1] for i in cand), default=r)
        if r_acc <= r:
            ok = all(res[i][0] <= res[i][1] for i in cand)
            need = above is not None and (lo == 0 or below is not None)
            return ok and need, inside, below, above, r_acc
        r = r_acc


def _random_block(key, n, b):
    return jax.random.normal(key, (n, b))


def si_slice(
    op, free, lo, hi, precond, *, capacity, sigma=None, R=None, W=None, tol=1e-6, tau=1e-8,
    block=6, maxit_inner=20000, seed=0,
):  # fmt: skip
    """Eigenpairs of the masked pencil (K_m, M_m) in the slice [lo, hi] (eigenvalues, 0 <= lo <
    hi) by block shift-invert Lanczos at sigma (default the midpoint). precond: SPD
    preconditioner of K + sigma M (fp64; a GMG with fp32 levels raises); R: M-orthonormal rigid
    modes; W: known M-orthonormal modes to deflate (host array, never returned). capacity:
    Lanczos columns (rounded up to a multiple of the chunk width), the only resource limit.
    Pair i is accepted at r_i <= t_i = max(tol, 10 tau_eff delta_i, round-off floor).

    Returns lam (ascending, the slice's modes), V (host fp64, M-orthonormal) and info:
    converged, reason, sigma, sigma_moved, steps, basis, minres_iterations, inner_cap,
    inner_maxit, tau_eff, tau_raised, r_acc, delta_max, residuals, thresholds, neighbours
    ({"lo": (lam, v) or None, "hi": ...}), capacity_full, edge_unreached, time."""
    check_x64("si_slice")
    t0 = time.perf_counter()
    n = op.n_dof
    free = np.asarray(free)
    if free.shape != (n,) or free.dtype != bool:
        raise ValueError(f"free must be a ({n},) bool mask")
    if not 0 <= lo < hi:
        raise ValueError(f"need 0 <= lo < hi, got lo = {lo}, hi = {hi}")
    if not isinstance(block, int) or block < 1:
        raise ValueError(f"block must be an int >= 1, got {block!r}")
    if not isinstance(capacity, int) or capacity < 2 * block:
        raise ValueError(f"capacity must be an int >= 2 block = {2 * block}, got {capacity!r}")
    if getattr(precond, "dtype", "float64") != "float64":
        raise ValueError("precond must run in fp64 inside MINRES (a GMG with fp32 levels)")
    sigma = float((lo + hi) / 2 if sigma is None else sigma)
    b, c = block, _chunk(block)
    capacity = -(-capacity // c) * c
    m = jnp.asarray(free, jnp.float64)[:, None]
    R = jnp.zeros((n, 0)) if R is None else jnp.asarray(R, jnp.float64)
    if R.ndim != 2 or R.shape[0] != n:
        raise ValueError(f"R must have shape ({n}, k), got {R.shape}")
    MR = _M(op, m, R) if R.shape[1] else R
    if R.shape[1] and np.abs(np.asarray(R.T @ MR) - np.eye(R.shape[1])).max() > 1e-8:
        raise ValueError("R must be M-orthonormal on the free DOFs")
    nw = 0
    if W is not None:
        W = np.asarray(W, np.float64)
        if W.ndim != 2 or W.shape[0] != n:
            raise ValueError(f"W must have shape ({n}, k), got {W.shape}")
        nw = W.shape[1]
        if nw:
            _check_ortho(op, m, W, seed)
    L0 = -(-nw // c) * c  # the Lanczos columns start at a chunk boundary
    Q = jnp.zeros((n, L0 + capacity))
    if nw:
        Q = _insert(Q, jnp.asarray(W), jnp.asarray(0, jnp.int32))
    key, sub = jax.random.split(jax.random.key(seed))
    nK, nM = map(float, _norms(op, m, m * jax.random.normal(sub, (n, 1))))
    consts = jnp.array([nK, nM, tol, C_FLOOR])

    def extend(Y, key):  # orthonormal new block, dropped columns refilled with random ones
        filled = jnp.asarray(L0 + k, jnp.int32)
        for i in range(REFILLS + 1):
            Y, keep = _extend(m * Y, op, m, R, MR, Q, filled, c=c)
            if keep.all():
                return Y, key
            if i == REFILLS:
                raise RuntimeError(f"Lanczos block still rank-deficient after {REFILLS} refills")
            key, sub = jax.random.split(key)
            Y = jnp.where(keep, Y, _random_block(sub, n, b))

    k = 0  # Lanczos columns
    G = np.zeros((capacity, capacity))
    key, sub = jax.random.split(key)
    Y, key = extend(_random_block(sub, n, b), key)
    Q, g = _append(Q, Y, op, m, jnp.asarray(L0, jnp.int32), jnp.asarray(k, jnp.int32), c=c)
    G[: k + b, k : k + b] = np.asarray(g[: k + b])
    k += b
    A = ap_operator(op, free, -sigma)
    tau_eff, r = tau, tol
    iters, cap, moved, inner_maxit, reason = [], maxit_inner, False, False, None
    seen = {"lo": lo == 0, "hi": False}
    times = dict(setup=time.perf_counter() - t0, minres=0.0, ortho=0.0, ritz=0.0)
    done, sets, last = False, None, None
    while True:
        if k + b > capacity:
            reason = "capacity full"
            break
        t = time.perf_counter()
        rhs = _M(op, m, jax.lax.dynamic_slice_in_dim(Q, L0 + k - b, b, 1))
        X, inf = minres(A, rhs, precond, tol=tau, maxit=cap)
        res = np.asarray(inf["residuals"])
        its = int(np.asarray(inf["iterations"]).max())
        capped = ~np.asarray(inf["converged"]) & ~np.asarray(inf["stagnated"])
        bad = np.asarray(inf["stagnated"]) | capped
        times["minres"] += time.perf_counter() - t
        if not iters and not moved and bad.any() and res[bad].max() > STAGNATION * tau:
            sigma += SIGMA_MOVE * (hi - lo)  # sigma on an eigenvalue: move once, repeat
            A, moved = ap_operator(op, free, -sigma), True
            continue
        iters.append(its)
        if len(iters) == 1:
            cap = min(maxit_inner, INNER_GROWTH * its)
        inner_maxit |= bool(capped.any())
        if bad.any():
            tau_eff = max(tau_eff, float(res[bad].max()))
        if tau_eff > TAU_FAIL:
            reason = "inner solves failed"
            break
        t = time.perf_counter()
        Y, key = extend(X, key)
        Q, g = _append(Q, Y, op, m, jnp.asarray(L0, jnp.int32), jnp.asarray(k, jnp.int32), c=c)
        G[: k + b, k : k + b] = np.asarray(g[: k + b])
        G[k : k + b, :k] = G[:k, k : k + b].T
        k += b
        times["ortho"] += time.perf_counter() - t
        t = time.perf_counter()
        mu, Yr = np.linalg.eigh((G[:k, :k] + G[:k, :k].T) / 2)
        cache = {}
        f = functools.partial(_ritz_chunk, Q, L0, k, Yr, op, m, consts, c)
        resid = functools.partial(_cands, f, mu, b, sigma, tau_eff, cache)
        done, inside, below, above, r = _slice_check(mu, lo, hi, r, resid)
        for name, i in (("lo", below), ("hi", above)):
            if i is not None and cache[i][0] <= cache[i][1]:
                seen[name] = True
        sets, last = (inside, below, above), (mu, Yr, cache)
        times["ritz"] += time.perf_counter() - t
        if done:
            break
    # final stage: the accepted pairs and neighbours, V formed on the host, Q released
    lam_out, V, res_out, thr_out = np.zeros(0), np.zeros((n, 0)), np.zeros(0), np.zeros(0)
    nb = {"lo": None, "hi": None}
    converged = done
    if last is not None:
        mu, Yr, cache = last
        inside, below, above = sets
        acc = [i for i in inside if cache[i][0] <= cache[i][1]]
        nbs = [
            (name, i)
            for name, i in (("lo", below), ("hi", above))
            if i is not None and cache[i][0] <= cache[i][1]
        ]
        sel = acc + [i for _, i in nbs]
        Vs = _form_v(Q, L0, Yr[:, sel], c) if sel else np.zeros((n, 0))
        Q = None
        f = functools.partial(_v_chunk, Vs, op, m, consts)
        rr, tt = _chunked(f, np.arange(len(sel)), mu[sel], b, sigma, tau_eff)
        na = len(acc)
        converged = done and bool((rr[:na] <= tt[:na]).all())
        lam_out, V, res_out, thr_out = mu[acc], Vs[:, :na], rr[:na], tt[:na]
        for j, (name, i) in enumerate(nbs):
            nb[name] = (float(mu[i]), Vs[:, na + j])
    Q = None
    unreached = [e for e in ("lo", "hi") if not seen[e]]
    delta = np.abs(lam_out - sigma) / np.abs(lam_out) if len(lam_out) else np.zeros(0)
    info = dict(
        converged=converged,
        reason=reason,
        sigma=sigma,
        sigma_moved=moved,
        steps=len(iters),
        basis=k,
        minres_iterations=iters,
        inner_cap=cap,
        inner_maxit=inner_maxit,
        tau_eff=tau_eff,
        tau_raised=tau_eff > tau,
        r_acc=float(r),
        delta_max=float(delta.max()) if len(delta) else 0.0,
        residuals=res_out,
        thresholds=thr_out,
        neighbours=nb,
        capacity_full=reason == "capacity full",
        edge_unreached=(unreached or [None])[0] if len(unreached) < 2 else "both",
        time=times,
    )
    return lam_out, V, info
