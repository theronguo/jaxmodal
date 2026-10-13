"""Generalized LOBPCG for K x = lam M x on masked matrix-free operators (Hetmaniuk-Lehoucq,
Duersch et al. 2018), with fixed block shapes, the Jacobi preconditioner and the sigma
pre-estimate. See docs/eigensolvers.md."""

import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from ..operators.common import check_x64
from ..precond.jacobi import jacobi

EPS = np.finfo(float).eps
RECOMPUTE = 20  # explicit K and M products every RECOMPUTE iterations
C_FLOOR = 10.0  # c of the floor term in the acceptance threshold
TAU = 1e-14  # SVQB drop tolerance, relative to the largest Gram eigenvalue
TAU_P = 1e-8  # P noise rule: drop a column whose norm falls below TAU_P of its norm before
PIVOT = 1e-10  # Rayleigh-Ritz pivot test: min diag(L)^2 of the Gram Cholesky
STALL = 50  # iterations without progress before the stagnation exit
POWER = 30  # power iterations for ||K|| and ||M||


def _K(op, m, v):
    return m * op.apply_K(m * v)


def _M(op, m, v):
    return m * op.apply_M(m * v)


def _svqb_t(G, act):
    """(T, keep) for a Gram matrix G: inactive rows/columns are zeroed (the exact theta = 0
    eigenspace), the rest scaled to unit diagonal; directions with theta <= TAU theta_max are
    dropped. T maps onto G-orthonormal columns; dropped and inactive outputs are exact zeros."""
    a = act.astype(G.dtype)
    G = (G + G.T) / 2 * a[:, None] * a[None, :]
    g = jnp.diag(G)
    d = jnp.where(act & (g > 0), 1 / jnp.sqrt(jnp.where(g > 0, g, 1.0)), 1.0)
    th, Q = jnp.linalg.eigh(G * d[:, None] * d[None, :])
    keep = th > TAU * th.max()
    T = d[:, None] * Q * jnp.where(keep, 1 / jnp.sqrt(jnp.where(keep, th, 1.0)), 0.0)
    return T * a[:, None], keep


def _svqb(V, KV, MV, act):
    T, keep = _svqb_t(V.T @ MV, act)
    return V @ T, KV @ T, MV @ T, keep


def _ortho(B, V):
    """V - B (B^T M V) for each of (V, KV, MV), with B = (X, KX, MX) M-orthonormal."""
    C = B[2].T @ V[0]
    return tuple(v - b @ C for v, b in zip(V, B, strict=True))


def _resid(X, KX, MX, lam, consts):
    """lam-scaled residuals r and thresholds max(tol, c eps (||K|| + |lam| ||M||) ||x|| /
    (|lam| ||Mx||))."""
    nK, nM, tol, c = consts
    al, nmx = jnp.abs(lam), jnp.linalg.norm(MX, axis=0)
    r = jnp.linalg.norm(KX - MX * lam, axis=0) / (al * nmx)
    floor = c * EPS * (nK + al * nM) * jnp.linalg.norm(X, axis=0) / (al * nmx)
    return r, jnp.maximum(tol, floor)


def _rr(GK, GM, act, beta):
    """Lowest Ritz pairs of (GK, GM) on the active rows/columns; masked slots get GM = 1 and
    GK = beta on the diagonal. ok = the Cholesky passes the pivot test."""
    a = act.astype(GK.dtype)
    GK = (GK + GK.T) / 2 * a[:, None] * a[None, :] + jnp.diag(beta * (1 - a))
    GM = (GM + GM.T) / 2 * a[:, None] * a[None, :] + jnp.diag(1 - a)
    L = jnp.linalg.cholesky(GM)
    Li = solve_triangular(L, jnp.eye(len(L)), lower=True)
    th, Y = jnp.linalg.eigh(Li @ GK @ Li.T)
    C = Li.T @ Y * a[:, None]
    ok = jnp.isfinite(C).all() & (jnp.min(jnp.diag(L)) ** 2 > PIVOT)
    return th, C, GM, ok


def _p_coeffs(Cx, Cp, G):
    """New P coefficients: Cp G-orthogonalized against Cx (twice), noise columns masked (norm
    after <= TAU_P norm before), then G-orthonormalized. Returns (Cp, live, keep): live over the
    input columns, keep over the output columns."""

    def gnorm(C):
        return jnp.sqrt(jnp.maximum(jnp.einsum("ij,ij->j", C, G @ C), 0.0))

    n0 = gnorm(Cp)
    for _ in range(2):
        Cp = Cp - Cx @ (Cx.T @ (G @ Cp))
    live = (n0 > 0) & (gnorm(Cp) > TAU_P * n0)
    T, keep = _svqb_t(Cp.T @ G @ Cp, live)
    return Cp @ T, live, keep


def _update(V, Cx, Cp):
    """(V_X Cx_X + V_W Cx_W + V_P Cx_P, same for Cp), one block at a time: XLA does not fuse a
    sum of GEMMs, and the barriers (which also hold back the blocks still to come) keep only
    two partial sums alive."""
    k = Cx.shape[1]
    x, p = V[0] @ Cx[:k], V[0] @ Cp[:k]
    for i in (1, 2):
        x, p, rest = jax.lax.optimization_barrier((x, p, V[i:]))
        x, p = x + rest[0] @ Cx[i * k : (i + 1) * k], p + rest[0] @ Cp[i * k : (i + 1) * k]
    return x, p


def _project(B, count, Y, MY, c):
    """Y - V (V^T MY) with V = B[:, :count], over column chunks of width c (upcast to fp64).
    Columns past count are zero, so the last chunk may read past it."""

    def body(i, Y):
        Bc = jax.lax.dynamic_slice_in_dim(B, i * c, c, 1).astype(Y.dtype)
        return Y - Bc @ (Bc.T @ MY)

    return jax.lax.fori_loop(0, (count + c - 1) // c, body, Y)


def _deflate(op, m, R, MR, B, count, Y, c):
    """Y M-orthogonalized against the locked basis B[:, :count] (one extra M apply, skipped
    while count = 0) and the rigid modes R, masked. B with no columns: a single-chunk solve."""
    if B.shape[1]:
        Y = jax.lax.cond(
            count > 0, lambda Y: _project(B, count, Y, _M(op, m, Y), c), lambda Y: Y, Y
        )
    return m * (Y - R @ (MR.T @ Y))


def _rr_coeffs(GK, GM, k, actW, actP, lam, consts):
    """Rayleigh-Ritz with the pivot-test fallback, and the new P coefficients:
    (theta, Cx, Cp, keep, dropped)."""
    beta = 10 * jnp.max(jnp.abs(lam)) + consts[0] / consts[1]
    ones, none = jnp.ones(k, bool), jnp.zeros(k, bool)
    th, C, G, ok = _rr(GK, GM, jnp.concatenate([ones, actW, actP]), beta)
    th2, C2, _, _ = _rr(GK, GM, jnp.concatenate([ones, actW, none]), beta)
    th, C = jnp.where(ok, th, th2), jnp.where(ok, C, C2)  # G fits both: C2 has no P rows
    Cx = C[:, :k]
    Cp, live, keep = _p_coeffs(Cx, Cx.at[:k].set(0.0), G)
    # a P column was lost (noise rule or SVQB drop) iff fewer kept than nonzero inputs
    dropped = ~ok | (keep.sum() < (jnp.abs(Cx[k:]).sum(0) > 0).sum())
    return th[:k], Cx, Cp, keep, dropped


@functools.partial(jax.jit, donate_argnums=0, static_argnames="c")
def lobpcg_step(state, op, m, precond, R, MR, B, count, consts, recompute, *, c):
    """One LOBPCG iteration. state = (X, KX, MX, P, KP, MP, actP, lam), donated; everything is
    an argument, so new data never recompiles. B, count: the locked basis (projector chunk
    width c). Returns the new state and one packed array: residuals, thresholds and lam of the
    new X, and the P drop flag."""
    X, KX, MX, P, KP, MP, actP, lam = state
    k = X.shape[1]
    r, t = _resid(X, KX, MX, lam, consts)
    act = r > t  # soft locking: converged columns get no new direction
    W = _deflate(op, m, R, MR, B, count, precond(KX - MX * lam) * act, c)
    W = (W, _K(op, m, W), _M(op, m, W))
    Xb, Pb = (X, KX, MX), (P, KP, MP)
    W = _ortho(Pb, _ortho(Xb, W))
    *W, actW = _svqb(*W, act)
    W = _ortho(Pb, _ortho(Xb, W))
    *W, actW = _svqb(*W, actW)
    blocks = (Xb, tuple(W), Pb)
    GK = jnp.block([[a[0].T @ b[1] for b in blocks] for a in blocks])
    GM = jnp.block([[a[0].T @ b[2] for b in blocks] for a in blocks])
    lam, Cx, Cp, keep, dropped = _rr_coeffs(GK, GM, k, actW, actP, lam, consts)
    (Xn, Pn), (KXn, KPn), (MXn, MPn) = (
        _update(tuple(b[j] for b in blocks), Cx, Cp) for j in range(3)
    )
    Xn, KXn, MXn, Pn, KPn, MPn = jax.lax.cond(
        recompute,
        lambda a: (a[0], _K(op, m, a[0]), _M(op, m, a[0]), a[3], _K(op, m, a[3]), _M(op, m, a[3])),
        lambda a: a,
        (Xn, KXn, MXn, Pn, KPn, MPn),
    )
    r, t = _resid(Xn, KXn, MXn, lam, consts)
    out = jnp.concatenate([r, t, lam, dropped[None].astype(r.dtype)])
    return (Xn, KXn, MXn, Pn, KPn, MPn, keep, lam), out


def _bar(*a):
    return jax.lax.optimization_barrier(a)


@functools.partial(jax.jit, donate_argnums=0, static_argnames="c")
def lobpcg_step_low(state, op, m, precond, R, MR, B, count, consts, *, c):
    """Low-memory iteration: state = (X, P, actP, lam), donated. Every K and M product is
    formed when needed, used for its Gram blocks and freed (the barriers bound each lifetime).
    Returns the new state and the packed residuals, thresholds and lam of the INPUT X."""
    X, P, actP, lam = state
    k = X.shape[1]
    KX = _K(op, m, X)
    X, P, KX = _bar(X, P, KX)
    MX = _M(op, m, X)
    r, t = _resid(X, KX, MX, lam, consts)
    act = r > t
    W = precond(KX - MX * lam) * act
    GK_XX, GK_PX = X.T @ KX, P.T @ KX
    X, P, MX, W = _bar(X, P, MX, W)  # KX dies
    W = _deflate(op, m, R, MR, B, count, W, c)
    X, P, MX, W = _bar(X, P, MX, W)
    MP = _M(op, m, P)
    for _ in range(2):
        W = W - X @ (MX.T @ W)
        W = W - P @ (MP.T @ W)
    GM_XX, GM_PX, GM_PP = X.T @ MX, P.T @ MX, P.T @ MP
    X, W, P = _bar(X, W, P)  # MX, MP die
    MW = _M(op, m, W)
    T, actW = _svqb_t(W.T @ MW, act)
    W, MW = W @ T, MW @ T
    GM_XW, GM_WW, GM_PW = X.T @ MW, W.T @ MW, P.T @ MW
    X, W, P = _bar(X, W, P)
    KW = _K(op, m, W)
    GK_XW, GK_WW, GK_PW = X.T @ KW, W.T @ KW, P.T @ KW
    X, W, P = _bar(X, W, P)
    GK_PP = P.T @ _K(op, m, P)
    X, W, P = _bar(X, W, P)
    GK = jnp.block([[GK_XX, GK_XW, GK_PX.T], [GK_XW.T, GK_WW, GK_PW.T], [GK_PX, GK_PW, GK_PP]])
    GM = jnp.block([[GM_XX, GM_XW, GM_PX.T], [GM_XW.T, GM_WW, GM_PW.T], [GM_PX, GM_PW, GM_PP]])
    th, Cx, Cp, keep, dropped = _rr_coeffs(GK, GM, k, actW, actP, lam, consts)
    Xn, Pn = _update((X, W, P), Cx, Cp)
    return (Xn, Pn, keep, th), jnp.concatenate([r, t, lam, dropped[None].astype(r.dtype)])


@functools.partial(jax.jit, donate_argnums=0)
def _insert(B, X, count):
    return jax.lax.dynamic_update_slice_in_dim(B, X.astype(B.dtype), count, 1)


def _lock(B, count, X, j):
    """B with columns count .. count + m holding X[:, :j] and zeros (m = X.shape[1]); B is
    donated. Checks the capacity before writing, so B stays intact when full."""
    if count + X.shape[1] > B.shape[1]:
        raise RuntimeError(f"locked basis full: {count} + {X.shape[1]} > {B.shape[1]} columns")
    return _insert(B, X * (jnp.arange(X.shape[1]) < j), jnp.asarray(count, jnp.int32))


@functools.partial(jax.jit, static_argnames="c")
def _reortho(X, j, op, m, B, count, c):
    """X[:, :j] M-orthogonalized twice against B[:, :count], then M-orthonormalized by
    Cholesky (keeps the column order); the other columns zero, as `_lock` expects. Also
    returns the smallest projected M-norm^2 (about 1; near 0 means a column lies in span(B)).
    The locked basis then stays M-orthonormal to rounding: a single deflation pass leaves the
    Jacobi-amplified locked directions in W otherwise, and the block drifts into span(B)."""
    act = jnp.arange(X.shape[1]) < j
    X = X * act
    for _ in range(2 if B.shape[1] else 0):
        X = _project(B, count, X, _M(op, m, X), c)
    G = jnp.where(act[:, None] & act[None, :], X.T @ _M(op, m, X), jnp.eye(X.shape[1]))
    L = jnp.linalg.cholesky((G + G.T) / 2)
    return jax.scipy.linalg.solve_triangular(L, X.T, lower=True).T, jnp.diag(G).min()


@jax.jit
def _norms(op, m, v):
    """||K_m||_2 and ||M_m||_2 by power iteration from v."""

    def power(A):
        def body(_, xn):
            y = A(op, m, xn[0])
            return y / jnp.linalg.norm(y), jnp.linalg.norm(y)

        return jax.lax.fori_loop(0, POWER, body, (v / jnp.linalg.norm(v), 0.0))[1]

    return power(_K), power(_M)


@functools.partial(jax.jit, static_argnames="c")
def _start(X, op, m, R, MR, B, count, c):
    """Mask, deflate and M-orthonormalize a start block: (X, KX, MX, keep)."""
    X = _deflate(op, m, R, MR, B, count, m * X, c)
    return _svqb(X, _K(op, m, X), _M(op, m, X), jnp.ones(X.shape[1], bool))


@jax.jit
def _rr_start(X, KX, MX):
    lam, Y = jnp.linalg.eigh((X.T @ KX + KX.T @ X) / 2)
    return X @ Y, KX @ Y, MX @ Y, lam


@jax.jit
def _confirm(X, lam, op, m, consts):
    """Explicit K X and M X with their residuals and thresholds: (KX, MX, r, t)."""
    KX, MX = _K(op, m, X), _M(op, m, X)
    return KX, MX, *_resid(X, KX, MX, lam, consts)


def _begin(X, key, op, m, R, MR, B, count, c, low):
    """Initial state from start columns X: deflated and SVQB-orthonormalized (dependent columns
    refilled with random ones, at most 3 times), Rayleigh-Ritz, P a masked zero block.
    Returns (state, key, refills)."""
    refills = 0
    while True:
        X, KX, MX, keep = _start(X, op, m, R, MR, B, count, c=c)
        if keep.all():
            break
        if refills == 3:
            raise RuntimeError("start block still rank-deficient after 3 refills")
        refills += 1
        key, sub = jax.random.split(key)
        X = jnp.where(keep, X, jax.random.normal(sub, X.shape))
    X, KX, MX, lam = _rr_start(X, KX, MX)
    actP = jnp.zeros(X.shape[1], bool)
    if low:
        return (X, jnp.zeros_like(X), actP, lam), key, refills
    P = [jnp.zeros_like(X) for _ in range(3)]  # separate buffers: the state is donated
    return (X, KX, MX, *P, actP, lam), key, refills


def _block_size(n_modes, chunk, guards):
    """(chunk, guards, block size m) with the defaults: one chunk, max(5, chunk // 5) guards."""
    chunk = n_modes if chunk is None else chunk
    guards = max(5, chunk // 5) if guards is None else guards
    return chunk, guards, chunk + guards


def band_defaults(n_modes, chunk, guards, n_avail):
    """(n_modes, chunk) of a band solve: chunk min(40, ceil(1.25 n_modes) + 2) (40 without
    n_modes), at most n_avail minus the guards (8 by default, the guards of chunk 40); n_modes
    (the expected count) defaults to 4 chunk."""
    if chunk is None:
        chunk = 40 if n_modes is None else min(40, math.ceil(1.25 * n_modes) + 2)
        chunk = max(1, min(chunk, n_avail - (8 if guards is None else guards)))
    return (4 * chunk if n_modes is None else n_modes), chunk


def _band_count(lam, t, lam_stop):
    """Columns of the ascending Ritz values lam (thresholds t) in the band [0, lam_stop]: those
    <= lam_stop (1 + 100 t), extended over a cluster (relative gaps < 100 t, as _extent)."""
    i = int((lam <= lam_stop * (1 + 100 * t)).sum())
    while 0 < i < len(lam) and lam[i] - lam[i - 1] < 100 * t[i - 1] * abs(lam[i - 1]):
        i += 1
    return i


def _capacity(n_modes, chunk, k, low):
    """(projector chunk width c, basis capacity): c = m in the low-memory step, else
    max(m, min(4m, 64)); no basis for a single chunk."""
    c = k if low else max(k, min(4 * k, 64))
    return c, (0 if chunk >= n_modes else -(-(n_modes + k) // c) * c)


def lobpcg(
    op, free, n_modes, tol=1e-6, R=None, precond=None, X0=None, maxit=500, seed=0, *,
    chunk=None, guards=None, basis_dtype="float64", low_memory=False,
):  # fmt: skip
    """Lowest n_modes eigenpairs of the masked pencil (K_m, M_m). Returns lam (n_modes,)
    ascending (NumPy), V (n_dof, n_modes) M-orthonormal with constrained rows 0 (NumPy fp64),
    and info. op: operator with apply_K, apply_M, diag_K, diag_M, n_dof; free: (n_dof,) bool;
    R: (n_dof, k) M-orthonormal deflation basis (rigid modes) or None; precond: pytree callable
    (n, j) -> (n, j), fixed, linear, SPD on the free DOFs, masked output (default
    jacobi(op, free)); X0: (n_dof, j <= block size) start columns of the first chunk.
    chunk: wanted modes per chunk (default n_modes), block size chunk + guards (default
    max(5, chunk // 5)); converged chunks are locked into a basis of dtype basis_dtype
    ("float64" or "float32"; with float32, chunks after the first accept at
    max(tol, 1.2e-7)). maxit counts per chunk. low_memory: the low-memory step (11 instead of
    16 blocks, about 1.4-1.9x the time). Never raises on non-convergence: check
    info["converged"] and info["stagnated"]; a failed chunk ends the solve, so fewer than
    n_modes can come back. Breakdowns raise: RuntimeError (start block rank-deficient after 3
    refills, a locked column in the span of the earlier ones), FloatingPointError (non-finite
    step)."""
    return _solve(
        op, free, n_modes, tol, R, precond, X0, maxit, seed,
        chunk=chunk, guards=guards, basis_dtype=basis_dtype, low=low_memory,
    )  # fmt: skip


def _extent(lam, r, t, want, guards, lock, failed=False):
    """want extended over a cluster at the block boundary: relative gaps < 100 t, or, at a
    failed exit, within 10 r^2 (capped at 1e-3; the Ritz value uncertainty of a slowly
    converging guard). A cluster that reaches the block end raises the guard error in a chunk
    that will be locked (lock); the final chunk keeps want (its tail is cut at n_modes)."""
    win = 100 * t[:-1]
    if failed:
        win = np.maximum(win, np.minimum(10 * r[1:] ** 2, 1e-3))
    j, k = want, len(lam)
    while j < k and lam[j] - lam[j - 1] < win[j - 1] * abs(lam[j - 1]):
        j += 1
    if j < k:
        return j
    if not lock:
        return want
    raise ValueError(
        f"a cluster at the chunk boundary (from mode {want}) fills the block: it needs more "
        f"than {guards} guard vectors; pass guards >= {guards + 1} or a chunk size that ends "
        "in a gap"
    )


def _iterate(state, args, consts, want, lock, guards, tol, maxit, low, c, stop):
    """The iteration loop of one chunk: (state, r, t, j, record), with j = want extended over
    a cluster at the block boundary and record the info["chunks"] entry plus history and
    drops."""
    op, m = args[0], args[1]
    k = state[0].shape[1]
    hist, lsum, drops = [], [], 0
    best, best_l, it_best = np.inf, np.inf, 0
    converged = stagnated = False
    j = want
    t_warm = 0.0  # step seconds after the compiling first iteration
    for it in range(maxit):
        ts = time.perf_counter()
        if low:
            state, out = lobpcg_step_low(state, *args, consts, c=c)
        else:
            recompute = (it + 1) % RECOMPUTE == 0
            state, out = lobpcg_step(state, *args, consts, recompute, c=c)
        out = np.asarray(out)
        if it:
            t_warm += time.perf_counter() - ts
        if not np.isfinite(out).all():
            raise FloatingPointError(f"non-finite LOBPCG step at iteration {it + 1}")
        r, t, lam = out[:k], out[k : 2 * k], out[2 * k : 3 * k]
        drops += int(out[-1])
        # cluster rule once the wanted residuals are <= sqrt(t): Ritz values are then accurate
        # to about t, well inside the 100 t window. j only grows; progress restarts with it
        near = bool((r[:want] ** 2 <= t[:want]).all())
        if near and (jn := _extent(lam, r, t, want, guards, lock)) > j:
            j, best, best_l, it_best = jn, np.inf, np.inf, it
        hist.append(r[:j].max())
        lsum.append(lam[:j].sum())
        if stop is not None and stop(lsum):
            break
        if hist[-1] < 0.9 * best or lsum[-1] < (1 - 1e-10) * best_l:
            best, best_l, it_best = min(best, hist[-1]), min(best_l, lsum[-1]), it
        if it - it_best >= STALL:
            stagnated = True
            break
        if (r[:j] <= t[:j]).all():
            state, r2, t2 = _explicit(state, op, m, consts, low)
            if (r2[:j] <= t2[:j]).all():
                r, t, converged = r2, t2, True
                stagnated = bool((r[:j] > tol).any())
                break
    if not converged:  # report residuals of explicit products on every exit
        # wanted columns near convergence but a failed chunk: a split cluster (guard error)
        if lock and near:
            _extent(lam, r, t, want, guards, lock, failed=True)
        state, r, t = _explicit(state, op, m, consts, low)
    rec = dict(want=want, locked=j if converged else 0, iterations=it + 1, converged=converged,
               stagnated=stagnated, history=hist, drops=drops,
               t_warm=t_warm / it if it else 0.0)  # fmt: skip
    return state, r, t, j, rec


def _explicit(state, op, m, consts, low):
    """_confirm on the state's X; the 9-block state takes the explicit products."""
    KX, MX, r, t = _confirm(state[0], state[-1], op, m, consts)
    if not low:
        state = (state[0], KX, MX) + state[3:]
    return state, np.asarray(r), np.asarray(t)


def _solve(
    op, free, n_modes, tol, R, precond, X0, maxit, seed, chunk=None, guards=None,
    basis_dtype="float64", low=False, stop=None, lam_stop=None,
):  # fmt: skip
    """lobpcg's solve; stop: a callback on the Ritz-sum history ending a chunk (sigma_estimate).
    lam_stop (band mode, solve_modes(band=...)): every chunk is lockable and the solve stops at
    the first chunk with a converged mode beyond the band [0, lam_stop] (_band_count); n_modes is
    the expected count (the basis and the host V grow past it) and chunk may exceed it. Returns
    every converged mode, info["n_band"] of them in the band."""
    check_x64("lobpcg")
    band = lam_stop is not None
    t0 = time.perf_counter()
    n = op.n_dof
    free = np.asarray(free)
    if free.shape != (n,) or free.dtype != bool:
        raise ValueError(f"free must be a ({n},) bool mask")
    R = jnp.zeros((n, 0)) if R is None else jnp.asarray(R, jnp.float64)
    if n_modes < 1 or maxit < 1:
        raise ValueError(f"n_modes and maxit must be >= 1, got {n_modes}, {maxit}")
    chunk, guards, k = _block_size(n_modes, chunk, guards)
    if not 1 <= chunk <= (np.inf if band else n_modes) or guards < 1:
        raise ValueError(f"need 1 <= chunk <= n_modes and guards >= 1, got {chunk}, {guards}")
    if basis_dtype not in ("float64", "float32"):
        raise ValueError(f"basis_dtype must be 'float64' or 'float32', got {basis_dtype!r}")
    if k > free.sum() - R.shape[1]:
        raise ValueError(
            f"block size {k} for n_modes = {n_modes} exceeds the {free.sum()} free DOFs minus "
            f"the deflation rank {R.shape[1]}"
        )
    m = jnp.asarray(free, jnp.float64)[:, None]
    MR = _M(op, m, R) if R.shape[1] else R
    if R.shape[1] and np.abs(np.asarray(R.T @ MR) - np.eye(R.shape[1])).max() > 1e-8:
        raise ValueError("R must be M-orthonormal on the free DOFs (m_orthonormalize with this op)")
    if X0 is not None:
        X0 = jnp.asarray(X0, jnp.float64)
        if X0.ndim != 2 or X0.shape[0] != n or X0.shape[1] > k:
            raise ValueError(f"X0 must have shape ({n}, j) with j <= {k}, got {X0.shape}")
    precond = jacobi(op, free) if precond is None else precond
    c, cap = _capacity(n_modes, chunk, k, low)
    fp32 = basis_dtype == "float32"
    B = jnp.zeros((n, cap), basis_dtype)
    key, sub = jax.random.split(jax.random.key(seed))
    nK, nM = map(float, _norms(op, m, m * jax.random.normal(sub, (n, 1))))
    t1 = time.perf_counter()
    # filled at each lock: the only host copy of the modes (band: room for the final block)
    V = np.empty((n, n_modes + (k if band else 0)))
    n_avail = int(free.sum()) - R.shape[1]
    count, lams, res, thr, chunks, hist = 0, [], [], [], [], []
    drops = refills = iterations = 0
    while True:
        if band and count + k > n_avail:
            f = np.sqrt(lam_stop) / (2 * np.pi)
            raise ValueError(
                f"the band needs more modes than the mesh provides at block size {k} (f_max = "
                f"{f:.6g} Hz, {count} modes locked, {n_avail} free DOFs minus the rigid modes)"
            )
        if band and count + k > V.shape[1]:
            V = np.concatenate([V[:, :count], np.empty((n, max(V.shape[1] // 2, k)))], axis=1)
        want = chunk if band else min(chunk, n_modes - count)
        key, sub = jax.random.split(key)
        X = jax.random.normal(sub, (n, k))
        if X0 is not None:
            X = X.at[:, : X0.shape[1]].set(X0)
        cnt = jnp.asarray(count, jnp.int32)
        state, key, nref = _begin(X, key, op, m, R, MR, B, cnt, c, low)
        # fp32 basis: residuals floor near 1e-8 against the rounded locked modes (2 u_32 term)
        t_eff = max(tol, 2.0**-23) if fp32 and count else tol
        consts = jnp.array([nK, nM, t_eff, C_FLOOR])
        args = (op, m, precond, R, MR, B, cnt)
        lock = band or count + want < n_modes  # the final chunk is never locked
        state, r, t, j, rec = _iterate(
            state, args, consts, want, lock, guards, tol, maxit, low, c, stop
        )
        ok = rec["converged"]
        refills, iterations = refills + nref, iterations + rec["iterations"]
        drops += rec.pop("drops")
        hist += rec.pop("history")
        chunks.append(rec)
        jj = j if ok else want
        lams.append(np.asarray(state[-1][:jj]))
        res.append(r[:jj])
        thr.append(t[:jj])
        i = _band_count(lams[-1], thr[-1], lam_stop) if band else 0
        if not ok or (not band and count + j >= n_modes):
            break
        X, g = _reortho(state[0], j, op, m, B, cnt, c=c)
        if not g > 0.5:
            raise RuntimeError(f"a converged mode lies in the span of the earlier ones ({g:.1e})")
        rl, tl = _confirm(X, state[-1], op, m, consts)[2:]  # of the columns actually kept
        res[-1], thr[-1] = np.asarray(rl[:j]), np.asarray(tl[:j])
        if band and i < j:  # the final chunk, re-orthonormalized for the count: not locked
            state = (X,) + state[1:]
            break
        V[:, count : count + j] = np.asarray(X[:, :j])
        if count + k > B.shape[1]:  # band: grow the basis (new zeros, copy, old freed)
            B = _insert(jnp.zeros((n, -(-max(3 * B.shape[1] // 2, count + k) // c) * c),
                                  B.dtype), B, jnp.asarray(0, jnp.int32))  # fmt: skip
            n_modes = B.shape[1] - k  # the expected count this capacity serves
        B = _lock(B, count, X, j)
        count += j
        X0, state = state[0][:, j:], None  # free the old state before the next start
    end = count + jj if band else min(count + jj, n_modes)
    V[:, count:end] = np.asarray(state[0][:, : end - count])
    V = np.ascontiguousarray(V[:, :end]) if band else V[:, :end]  # band: free the buffer
    info = dict(
        iterations=iterations,
        converged=all(ch["converged"] for ch in chunks),
        stagnated=any(ch["stagnated"] for ch in chunks),
        residuals=np.concatenate(res)[:end],
        thresholds=np.concatenate(thr)[:end],
        p_drops=drops,
        refills=refills,
        norm_K=nK,
        norm_M=nM,
        history=np.array(hist),
        chunks=chunks,
        time=dict(setup=t1 - t0, iterations=time.perf_counter() - t1),
        t_iteration=chunks[0]["t_warm"],
    )
    if band:
        info.update(n_band=count + i, chunk=chunk, block=k, capacity=B.shape[1], n_modes=n_modes)
    return np.concatenate(lams)[:end], V, info


def sigma_estimate(op, free, R=None, maxit=300, seed=0):
    """(sigma, iterations): the Rayleigh quotient lam_1 estimate (an upper bound) from 1-mode
    Jacobi LOBPCG, stopped when it decreased by less than 5% over the last 10 iterations. Pass
    the rigid modes R for a free body, else sigma is a rigid mode (about 0)."""

    def stop(lsum):
        return len(lsum) > 10 and lsum[-11] - lsum[-1] < 0.05 * lsum[-1]

    lam, _, info = _solve(op, free, 1, 0.0, R, None, None, maxit, seed, stop=stop)
    return float(lam[0]), info["iterations"]
