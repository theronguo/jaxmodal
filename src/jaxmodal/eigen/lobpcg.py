"""Generalized LOBPCG for K x = lam M x on masked matrix-free operators (Hetmaniuk-Lehoucq,
Duersch et al. 2018), with fixed block shapes, the Jacobi preconditioner and the sigma
pre-estimate. See docs/eigensolvers.md."""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.tree_util import Partial

from ..operators.common import check_x64

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


def _jacobi(inv, r):
    return inv[:, None] * r


def jacobi(op, free, sigma=0.0):
    """Jacobi on A_P = K + sigma M as a pytree callable: v -> m * v / (diag K + sigma diag M),
    0 on constrained DOFs (masked output)."""
    free = jnp.asarray(free)
    d = op.diag_K() + sigma * op.diag_M()
    return Partial(_jacobi, jnp.where(free, 1 / jnp.where(free, d, 1.0), 0.0))


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


@functools.partial(jax.jit, donate_argnums=0)
def lobpcg_step(state, op, m, precond, R, MR, consts, recompute):
    """One LOBPCG iteration. state = (X, KX, MX, P, KP, MP, actP, lam), donated; everything is
    an argument, so new data never recompiles. Returns the new state and one packed array:
    residuals, thresholds and lam of the new X, and the P drop flag."""
    X, KX, MX, P, KP, MP, actP, lam = state
    k = X.shape[1]
    r, t = _resid(X, KX, MX, lam, consts)
    act = r > t  # soft locking: converged columns get no new direction
    W = precond(KX - MX * lam) * act
    W = m * (W - R @ (MR.T @ W))
    W = (W, _K(op, m, W), _M(op, m, W))
    Xb, Pb = (X, KX, MX), (P, KP, MP)
    W = _ortho(Pb, _ortho(Xb, W))
    *W, actW = _svqb(*W, act)
    W = _ortho(Pb, _ortho(Xb, W))
    *W, actW = _svqb(*W, actW)
    blocks = (Xb, tuple(W), Pb)
    GK = jnp.block([[a[0].T @ b[1] for b in blocks] for a in blocks])
    GM = jnp.block([[a[0].T @ b[2] for b in blocks] for a in blocks])
    beta = 10 * jnp.max(jnp.abs(lam)) + consts[0] / consts[1]
    ones, none = jnp.ones(k, bool), jnp.zeros(k, bool)
    th, C, G, ok = _rr(GK, GM, jnp.concatenate([ones, actW, actP]), beta)
    th2, C2, G2, _ = _rr(GK, GM, jnp.concatenate([ones, actW, none]), beta)
    th, C, G = jnp.where(ok, th, th2), jnp.where(ok, C, C2), jnp.where(ok, G, G2)
    Cx = C[:, :k]
    Cp, live, keep = _p_coeffs(Cx, Cx.at[:k].set(0.0), G)
    noisy = (jnp.abs(Cx[k:]).sum(0) > 0) & ~live
    dropped = ~ok | noisy.any() | (keep.sum() < live.sum())
    new = [_update(tuple(b[j] for b in blocks), Cx, Cp) for j in range(3)]
    Xn, KXn, MXn, Pn, KPn, MPn = jax.lax.cond(
        recompute,
        lambda a: (a[0], _K(op, m, a[0]), _M(op, m, a[0]), a[3], _K(op, m, a[3]), _M(op, m, a[3])),
        lambda a: a,
        (new[0][0], new[1][0], new[2][0], new[0][1], new[1][1], new[2][1]),
    )
    lam = th[:k]
    r, t = _resid(Xn, KXn, MXn, lam, consts)
    out = jnp.concatenate([r, t, lam, dropped[None].astype(r.dtype)])
    return (Xn, KXn, MXn, Pn, KPn, MPn, keep, lam), out


@functools.partial(jax.jit, static_argnums=3)
def _norms(op, m, v, its):
    """||K_m||_2 and ||M_m||_2 by power iteration from v."""

    def power(A):
        def body(_, xn):
            y = A(op, m, xn[0])
            return y / jnp.linalg.norm(y), jnp.linalg.norm(y)

        return jax.lax.fori_loop(0, its, body, (v / jnp.linalg.norm(v), 0.0))[1]

    return power(_K), power(_M)


@jax.jit
def _start(X, op, m, R, MR):
    """Mask, deflate and M-orthonormalize a start block: (X, KX, MX, keep)."""
    X = m * (X - R @ (MR.T @ X))
    return _svqb(X, _K(op, m, X), _M(op, m, X), jnp.ones(X.shape[1], bool))


@jax.jit
def _rr_start(X, KX, MX):
    lam, Y = jnp.linalg.eigh((X.T @ KX + KX.T @ X) / 2)
    return X @ Y, KX @ Y, MX @ Y, lam


@jax.jit
def _confirm(state, op, m, consts):
    """Explicit K X, M X written back into the state, with their residuals and thresholds."""
    X, lam = state[0], state[7]
    KX, MX = _K(op, m, X), _M(op, m, X)
    return (X, KX, MX) + state[3:], *_resid(X, KX, MX, lam, consts)


def lobpcg(op, free, n_modes, tol=1e-6, R=None, precond=None, X0=None, maxit=500, seed=0):
    """Lowest n_modes eigenpairs of the masked pencil (K_m, M_m). Returns lam (n_modes,)
    ascending (NumPy), V (n_dof, n_modes) M-orthonormal with constrained rows 0 (device), and
    info. op: operator with apply_K, apply_M, diag_K, diag_M, n_dof; free: (n_dof,) bool;
    R: (n_dof, k) M-orthonormal deflation basis (rigid modes) or None; precond: pytree callable
    (n, j) -> (n, j), fixed, linear, SPD on the free DOFs, masked output (default
    jacobi(op, free)); X0: (n_dof, j <= block size) start columns. Never raises on
    non-convergence: check info["converged"] and info["stagnated"]."""
    return _solve(op, free, n_modes, tol, R, precond, X0, maxit, seed)


def _solve(op, free, n_modes, tol, R, precond, X0, maxit, seed, stop=None):
    check_x64("lobpcg")
    t0 = time.perf_counter()
    n = op.n_dof
    free = np.asarray(free)
    if free.shape != (n,) or free.dtype != bool:
        raise ValueError(f"free must be a ({n},) bool mask")
    R = jnp.zeros((n, 0)) if R is None else jnp.asarray(R, jnp.float64)
    k = n_modes + max(5, n_modes // 5)
    if n_modes < 1 or k > free.sum() - R.shape[1]:
        raise ValueError(
            f"block size {k} for n_modes = {n_modes} exceeds the {free.sum()} free DOFs minus "
            f"the deflation rank {R.shape[1]}"
        )
    m = jnp.asarray(free, jnp.float64)[:, None]
    MR = _M(op, m, R) if R.shape[1] else R
    if R.shape[1] and np.abs(np.asarray(R.T @ MR) - np.eye(R.shape[1])).max() > 1e-8:
        raise ValueError("R must be M-orthonormal on the free DOFs (m_orthonormalize with this op)")
    precond = jacobi(op, free) if precond is None else precond
    key, sub = jax.random.split(jax.random.key(seed))
    nK, nM = map(float, _norms(op, m, m * jax.random.normal(sub, (n, 1)), POWER))
    key, sub = jax.random.split(key)
    X = jax.random.normal(sub, (n, k))
    if X0 is not None:
        X0 = jnp.asarray(X0, jnp.float64)
        if X0.ndim != 2 or X0.shape[0] != n or X0.shape[1] > k:
            raise ValueError(f"X0 must have shape ({n}, j) with j <= {k}, got {X0.shape}")
        X = X.at[:, : X0.shape[1]].set(X0)
    refills = 0
    while True:  # SVQB drops dependent columns; refill them with random vectors
        X, KX, MX, keep = _start(X, op, m, R, MR)
        if keep.all():
            break
        if refills == 3:
            raise RuntimeError("start block still rank-deficient after 3 refills")
        refills += 1
        key, sub = jax.random.split(key)
        X = jnp.where(keep, X, jax.random.normal(sub, X.shape))
    X, KX, MX, lam = _rr_start(X, KX, MX)
    state = (X, KX, MX, jnp.zeros_like(X), jnp.zeros_like(X), jnp.zeros_like(X),
             jnp.zeros(k, bool), lam)  # fmt: skip
    consts = jnp.array([nK, nM, tol, C_FLOOR])
    t1 = time.perf_counter()
    hist, lsum, drops = [], [], 0
    best, best_l, it_best = np.inf, np.inf, 0
    converged = stagnated = False
    for it in range(maxit):
        state, out = lobpcg_step(state, op, m, precond, R, MR, consts, (it + 1) % RECOMPUTE == 0)
        out = np.asarray(out)
        if not np.isfinite(out).all():
            raise FloatingPointError(f"non-finite LOBPCG step at iteration {it + 1}")
        r, t, lam = out[:k], out[k : 2 * k], out[2 * k : 3 * k]
        drops += int(out[-1])
        hist.append(r[:n_modes].max())
        lsum.append(lam[:n_modes].sum())
        if stop is not None and stop(lsum):
            break
        if hist[-1] < 0.9 * best or lsum[-1] < (1 - 1e-10) * best_l:
            best, best_l, it_best = min(best, hist[-1]), min(best_l, lsum[-1]), it
        if it - it_best >= STALL:
            stagnated = True
            break
        if (r[:n_modes] <= t[:n_modes]).all():
            state, r2, t2 = _confirm(state, op, m, consts)
            r2, t2 = np.asarray(r2), np.asarray(t2)
            if (r2[:n_modes] <= t2[:n_modes]).all():
                r, t, converged = r2, t2, True
                stagnated = bool((r[:n_modes] > tol).any())
                break
    info = dict(
        iterations=it + 1,
        converged=converged,
        stagnated=stagnated,
        residuals=r[:n_modes],
        thresholds=t[:n_modes],
        p_drops=drops,
        refills=refills,
        norm_K=nK,
        norm_M=nM,
        history=np.array(hist),
        time=dict(setup=t1 - t0, iterations=time.perf_counter() - t1),
    )
    return np.asarray(state[7][:n_modes]), state[0][:, :n_modes], info


def sigma_estimate(op, free, R=None, maxit=300, seed=0):
    """(sigma, iterations): the Rayleigh quotient lam_1 estimate (an upper bound) from 1-mode
    Jacobi LOBPCG, stopped when it decreased by less than 5% over the last 10 iterations. Pass
    the rigid modes R for a free body, else sigma is a rigid mode (about 0)."""

    def stop(lsum):
        return len(lsum) > 10 and lsum[-11] - lsum[-1] < 0.05 * lsum[-1]

    lam, _, info = _solve(op, free, 1, 0.0, R, None, None, maxit, seed, stop)
    return float(lam[0]), info["iterations"]
