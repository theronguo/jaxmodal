"""Stochastic completeness count: the deflated estimator with a Chebyshev-Jackson step
polynomial of T = (K + lam_c M)^-1 M and preconditioned inner PCG solves. See
docs/eigensolvers.md#completeness-count."""

import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.tree_util import Partial
from scipy.fft import dct

from ..linalg.krylov import _pcg
from ..operators import ap_operator
from ..operators.common import check_x64, is_count
from ..operators.sampling import mass_sample_keys
from ..precond.jacobi import jacobi

# Acceptance criteria and model constants (internal, docs/eigensolvers.md#completeness-count).
Q_MIN = 0.95  # q^2 at the in-band edge: one missed mode gives E s >= Q_MIN
BIAS = 0.05  # Jackson-tail bias bound (Weyl model)
WEYL_A = 1.5  # N(lam) ~ lam^a: the 3D exponent, an upper bound on fits to computed modes
MAX_DEGREE = 100_000  # cost cap: above it the count is not run (unverifiable)
D_RULE = 23.0  # d ~ D_RULE / g (measured)
CHUNK = 64  # projector chunk width (columns of W per host-to-device transfer)
SKETCH = 3  # columns of the M-orthonormality sketch
ORTHO_TOL = {"float64": 1e-6, "float32": 1e-4}


def jackson_step(d):
    """(d + 1,) Chebyshev coefficients of the Jackson-damped indicator of x >= 0 on [-1, 1]."""
    k = np.arange(1, d + 1)
    c = np.r_[0.5, 2 * np.sin(k * np.pi / 2) / (k * np.pi)]
    a = np.pi / (d + 1)
    j = np.arange(d + 1)
    g = ((d - j + 1) * np.cos(j * a) + np.sin(j * a) / np.tan(a)) / (d + 1)
    return c * g


def step_value(gamma, theta):
    """q(cos theta) = sum_k gamma_k cos(k theta), by the direct O(d) sum."""
    return np.cos(np.outer(np.atleast_1d(theta), np.arange(len(gamma)))) @ gamma


def _theta(lam):
    """theta with cos theta = x = (1 - lam) / (1 + lam), lam in units of lam_c."""
    return 2 * np.arctan(np.sqrt(lam))


def _grid(gamma, factor=8):
    """(theta, q) on the uniform grid theta_j = pi (j + 1/2) / N, N = factor d (a DCT-III)."""
    n = factor * (len(gamma) - 1)
    x = np.zeros(n)
    x[0], x[1 : len(gamma)] = gamma[0], gamma[1:] / 2
    return np.pi * (np.arange(n) + 0.5) / n, dct(x, type=3)


def tail_bias(gamma, gap, lam_top, n_top, n_free):
    """Jackson-tail bias int q^2 dN over (lam_top, lam_max] with N(lam) = n_top (lam /
    lam_top)^a, N(lam_max) = n_free; lam_top in units of lam_c. Trapezoid on a 64 d grid
    (second order: 0.04% against quad where dN/dtheta grows fastest; 8 d gives 2.5%)."""
    if n_top >= n_free:
        return 0.0
    lam_max = lam_top * (n_free / n_top) ** (1 / WEYL_A)
    t0, t1 = _theta(lam_top), _theta(lam_max)
    th, q = _grid(gamma, 64)
    sel = (th > t0) & (th < t1)
    th = np.r_[t0, th[sel], t1]  # both ends exact: dN/dtheta peaks at t1
    q = np.r_[step_value(gamma, t0), q[sel], step_value(gamma, t1)]
    lam = np.tan(th / 2) ** 2
    dlam = np.tan(th / 2) / np.cos(th / 2) ** 2  # d lam / d theta
    dN = n_top * WEYL_A * lam ** (WEYL_A - 1) / lam_top**WEYL_A * dlam
    return float(np.trapezoid(q**2 * dN, th))


def _passes(d, gap, lam_top, n_top, n_free):
    gamma = jackson_step(d)
    tl = _theta(1 - gap / 2)
    if step_value(gamma, tl)[0] ** 2 < Q_MIN:
        return False, np.nan
    th, q = _grid(gamma)
    if (q[th < tl] ** 2).min(initial=1.0) < Q_MIN:  # ripples inside the band
        return False, np.nan
    bias = tail_bias(gamma, gap, lam_top, n_top, n_free)
    return bias <= BIAS, bias


def count_degree(gap, lam_top, n_top, n_free):
    """(d, bias): the smallest degree with q^2 >= Q_MIN on the band [0, lam_c (1 - gap / 2)]
    and tail bias <= BIAS (lam_top in units of lam_c), by doubling then bisection; (None, nan)
    when no d <= MAX_DEGREE passes (at once when D_RULE / gap > MAX_DEGREE)."""
    if D_RULE / gap > MAX_DEGREE:
        return None, np.nan
    hi = 2
    while not (ok := _passes(hi, gap, lam_top, n_top, n_free))[0]:
        hi *= 2
        if hi > 2 * MAX_DEGREE:
            return None, np.nan
    lo, best = hi // 2, (hi, ok[1])
    while best[0] - lo > 1:
        mid = (lo + best[0]) // 2
        ok = _passes(mid, gap, lam_top, n_top, n_free)
        lo, best = (lo, (mid, ok[1])) if ok[0] else (mid, best)
    return best if best[0] <= MAX_DEGREE else (None, np.nan)


# ---------------------------------------------------------------- device kernels


def _mm(op, m, v):
    return m * op.apply_M(m * v)


def _mp(op, m, v):  # M_P = m M m + (1 - m)
    return _mm(op, m, v) + (1 - m) * v


def _jac(inv, r):
    return inv * r


def _solve_y(op, m, g, tau, maxit):
    """(y, ok): y = M_P^-1 g (masked g), by Jacobi PCG, exactly for lumped mass."""
    d = op.diag_M()[:, None]
    inv = m / jnp.where(m > 0, d, 1.0) + (1 - m)
    if op.mass == "lumped":
        return inv * g, True
    y, info = _pcg(Partial(_mp, op, m), Partial(_jac, inv), g, tau, maxit)
    return y, bool((info["converged"] & ~info["stagnated"]).all())


@functools.partial(jax.jit, donate_argnums=0)
def _chunk_step(Y, Wc, MY):
    Wc = Wc.astype(Y.dtype)
    return Y - Wc @ (Wc.T @ MY)


@jax.jit
def _rigid_step(Y, R, MR, m):
    return m * (Y - R @ (MR.T @ Y))


def _chunks(W):
    """Column chunks of W (width CHUNK, the last zero-padded on the device): one compiled
    width; a device W is sliced in place."""
    for i in range(0, W.shape[1], CHUNK):
        Wc = jnp.asarray(W[:, i : i + CHUNK])
        yield jnp.pad(Wc, ((0, 0), (0, CHUNK - Wc.shape[1]))) if Wc.shape[1] < CHUNK else Wc


def _project(op, m, W, R, MR, Y):
    """Pi Y = Y - W W^T M Y - R R^T M Y, masked; W streamed in column chunks (never all on the
    device). Y is donated."""
    MY = _mm(op, m, Y)
    for Wc in _chunks(W):
        Y = _chunk_step(Y, Wc, MY)
    return _rigid_step(Y, R, MR, m)


@jax.jit
def _recurrence(A, P, m, w0, gamma, d, lam_c, tau, maxit):
    """u = sum_k gamma_k T_k(X) w0 for k <= d, X = 2 lam_c T - I, T w by PCG on A (= A_P of
    K + lam_c M). Stops early when a solve stagnates or does not converge (bad). Returns
    (u, steps, iteration sum, iteration max, bad)."""

    def body(s):
        k, w_prev, w, u, it_sum, it_max, _ = s
        t, info = _pcg(A, P, _mm(A.op, m, w), tau, maxit)
        first = k == 1
        w_new = jnp.where(first, 1.0, 2.0) * (2 * lam_c * t - w)
        w_new = w_new - jnp.where(first, 0.0, 1.0) * w_prev
        bad = ~(info["converged"] & ~info["stagnated"]).all()
        its = info["iterations"]
        return (k + 1, w, w_new, u + gamma[k] * w_new, it_sum + its.sum(),
                jnp.maximum(it_max, its.max()), bad)  # fmt: skip

    zero = jnp.zeros((), int)
    s = (zero + 1, jnp.zeros_like(w0), w0, gamma[0] * w0, zero, zero, jnp.zeros((), bool))
    k, _, _, u, it_sum, it_max, bad = lax.while_loop(lambda s: (s[0] <= d) & ~s[6], body, s)
    return u, k - 1, it_sum, it_max, bad


def _pow2(n):
    return 1 << (int(n) - 1).bit_length()


def _check_ortho(op, m, W, seed):
    """Streamed sketch of W^T M W - I: two passes over W, e = W^T M (W z) - z."""
    k = W.shape[1]
    key = jax.random.fold_in(jax.random.key(seed), 2**32 - 1)
    z = np.asarray(jax.random.normal(key, (k, SKETCH)))
    z_pad = np.pad(z, ((0, -k % CHUNK), (0, 0)))
    v = jnp.zeros((W.shape[0], SKETCH))
    for i, Wc in enumerate(_chunks(W)):
        v = v + Wc.astype(v.dtype) @ jnp.asarray(z_pad[i * CHUNK : (i + 1) * CHUNK])
    Mv = _mm(op, m, v)
    e = np.concatenate([np.asarray(Wc.astype(Mv.dtype).T @ Mv) for Wc in _chunks(W)])[:k] - z
    tol = ORTHO_TOL.get(str(W.dtype), ORTHO_TOL["float32"])
    if not np.linalg.norm(e) <= tol * np.linalg.norm(z):
        raise ValueError(
            f"W must be M-orthonormal on the free DOFs: sketch error "
            f"{np.linalg.norm(e) / np.linalg.norm(z):.2e} > {tol:g}"
        )


def count(
    op, free, W, lam, lam_last, lam_next, *, R=None, precond=None, n_probes=50, seed=0,
    tau=None, degree=None, block=None, maxit=1000,
):  # fmt: skip
    """Stochastic count of the eigenvalues below lam_c = (lam_last + lam_next) / 2 that are
    missing from W. Returns (estimate, info); passed = verifiable and estimate < 0.5.

    W (n, k): the computed modes, M-orthonormal (NumPy or device, fp64 or fp32), with
    eigenvalues lam (k,); lam_last: the last mode below the cut (or the band edge when none is
    returned), lam_next: the first mode above it, both in lam's units, lam_next <= max(lam)
    (the tail model needs every mode up to lam_next in W). R: the rigid modes (M-orthonormal).
    precond: SPD preconditioner for K + lam_c M (default jacobi(op, free, lam_c); PR 14b passes
    the fp64 multigrid). degree overrides the degree rule, tau the inner tolerance
    0.01 / (d sqrt(n_free)); block: probes per block (default n_probes); maxit: per PCG solve.
    A stagnated or unconverged solve makes the count unverifiable: the remaining work is
    skipped and the estimate is NaN.
    """
    check_x64("count")
    t0 = time.perf_counter()
    n = op.n_dof
    free = np.asarray(free)
    if free.shape != (n,) or free.dtype != bool:
        raise ValueError(f"free must be a ({n},) bool mask")
    lam = np.asarray(lam, float)
    if lam.ndim != 1 or len(lam) == 0:
        raise ValueError("lam must be a non-empty 1-D array")
    if W.ndim != 2 or W.shape != (n, len(lam)):
        raise ValueError(f"W must have shape ({n}, {len(lam)}), got {W.shape}")
    if not 0 <= lam_last < lam_next <= lam.max():
        raise ValueError(
            f"need 0 <= lam_last < lam_next <= max(lam), got {lam_last}, {lam_next}, {lam.max()}"
        )
    for name, v in (("n_probes", n_probes), ("maxit", maxit)):
        if not is_count(v):
            raise ValueError(f"{name} must be an int >= 1, got {v!r}")
    for name, v in (("block", block), ("degree", degree)):
        if v is not None and not is_count(v):
            raise ValueError(f"{name} must be an int >= 1 or None, got {v!r}")
    if tau is not None and not tau > 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    m = jnp.asarray(free, jnp.float64)[:, None]
    R = jnp.zeros((n, 0)) if R is None else jnp.asarray(R, jnp.float64)
    MR = _mm(op, m, R) if R.shape[1] else R
    if R.shape[1] and np.abs(np.asarray(R.T @ MR) - np.eye(R.shape[1])).max() > 1e-8:
        raise ValueError("R must be M-orthonormal on the free DOFs")
    _check_ortho(op, m, W, seed)

    lam_c = (lam_last + lam_next) / 2
    gap = (lam_next - lam_last) / lam_c
    n_free, n_top = int(free.sum()), len(lam) + R.shape[1]
    block = n_probes if block is None else min(block, n_probes)
    n_blocks = -(-n_probes // block)
    info = dict(gap=gap, lam_c=lam_c, block=block, steps=0, pcg_iterations=(np.nan, 0),
                verifiable=False, passed=False, probes=np.full(n_probes, np.nan),
                std=np.nan)  # fmt: skip
    lam_top = lam.max() / lam_c
    if degree is None:
        d, bias = count_degree(gap, lam_top, n_top, n_free)
        if d is None:  # cost cap: not run
            info.update(degree=math.ceil(D_RULE / gap), tau=tau, bias=np.nan,
                        time=dict(total=time.perf_counter() - t0))  # fmt: skip
            return np.nan, info
    else:
        d = degree
        bias = tail_bias(jackson_step(d), gap, lam_top, n_top, n_free)
    tau = 0.01 / (d * math.sqrt(n_free)) if tau is None else float(tau)
    info.update(degree=d, tau=tau, bias=bias)

    precond = jacobi(op, free, lam_c) if precond is None else precond
    A = ap_operator(op, free, lam_c)
    gamma = np.zeros(_pow2(d + 1))
    gamma[: d + 1] = jackson_step(d)
    gamma = jnp.asarray(gamma)
    # probe j from fold_in(key, j): independent of the block size and of n_probes
    fold = jax.vmap(jax.random.fold_in, (None, 0))
    keys = fold(jax.random.key(seed), jnp.arange(n_blocks * block))
    s_all = np.full(n_blocks * block, np.nan)
    it_sum = it_max = steps = 0
    t_sample = t_rec = 0.0
    ok = True
    for i in range(n_blocks):
        ta = time.perf_counter()
        g = m * mass_sample_keys(op, keys[i * block : (i + 1) * block])
        y, ok = _solve_y(op, m, g, tau, maxit)
        del g
        tb = time.perf_counter()
        t_sample += tb - ta
        if not ok:
            break
        w0 = _project(op, m, W, R, MR, y)
        del y  # only w0 and the recurrence are alive in the step phase
        u, k, isum, imax, bad = _recurrence(A, precond, m, w0, gamma, d, float(lam_c), tau, maxit)
        del w0
        k, bad = int(k), bool(bad)
        steps, it_sum, it_max = steps + k, it_sum + int(isum), max(it_max, int(imax))
        if not bad:
            u = _project(op, m, W, R, MR, u)
            s_all[i * block : (i + 1) * block] = np.asarray(jnp.sum(u * _mm(op, m, u), 0))
        del u
        t_rec += time.perf_counter() - tb
        if bad:
            ok = False
            break
    probes = s_all[:n_probes]  # NaN for the columns of a stopped or skipped block
    estimate = float(probes.mean()) if ok else np.nan
    info.update(
        probes=probes,
        std=float(probes.std(ddof=1)) if ok and n_probes > 1 else np.nan,
        steps=steps,
        pcg_iterations=(it_sum / max(steps * block, 1), it_max),
        verifiable=ok,
        passed=bool(ok and estimate < 0.5),
        time=dict(
            sampling=t_sample,
            recurrence=t_rec,
            step=t_rec / max(steps, 1),
            total=time.perf_counter() - t0,
        ),  # fmt: skip
    )
    return estimate, info
