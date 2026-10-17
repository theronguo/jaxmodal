"""Stochastic completeness count: the deflated estimator with a Chebyshev-Jackson step (or
band-pass) polynomial of T = (K + lam_c M)^-1 M and preconditioned inner PCG solves. See
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
PRINT_S = 10.0  # verbose: at most one progress line per PRINT_S seconds


def jackson_step(d, x0=0.0):
    """(d + 1,) Chebyshev coefficients of the Jackson-damped indicator of x >= x0 on [-1, 1]."""
    t = np.arccos(x0)
    k = np.arange(1, d + 1)
    c = np.r_[t / np.pi, 2 * np.sin(k * t) / (k * np.pi)]
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


def _bias(gamma, l0, l1, n_ref, lam_ref):
    """int q^2 dN over [l0, l1] (units of lam_c) with N(lam) = n_ref (lam / lam_ref)^a.
    Trapezoid on a 64 d grid (second order: 0.04% against quad where dN/dtheta grows fastest;
    8 d gives 2.5%)."""
    t0, t1 = _theta(l0), _theta(l1)
    th, q = _grid(gamma, 64)
    sel = (th > t0) & (th < t1)
    th = np.r_[t0, th[sel], t1]  # both ends exact: dN/dtheta peaks at the top
    q = np.r_[step_value(gamma, t0), q[sel], step_value(gamma, t1)]
    lam = np.tan(th / 2) ** 2
    dlam = np.tan(th / 2) / np.cos(th / 2) ** 2  # d lam / d theta
    dN = n_ref * WEYL_A * lam ** (WEYL_A - 1) / lam_ref**WEYL_A * dlam
    return float(np.trapezoid(q**2 * dN, th))


def tail_bias(gamma, gap, lam_top, n_top, n_free):
    """Jackson-tail bias int q^2 dN over (lam_top, lam_max] with N(lam) = n_top (lam /
    lam_top)^a, N(lam_max) = n_free; lam_top in units of lam_c."""
    if n_top >= n_free:
        return 0.0
    lam_max = lam_top * (n_free / n_top) ** (1 / WEYL_A)
    return _bias(gamma, lam_top, lam_max, n_top, lam_top)


def _gamma(d, lower):
    """The step coefficients, or the band-pass ones for lower = (lam_in, lam_bot)."""
    gamma = jackson_step(d)
    if lower is None:
        return gamma
    a = sum(lower) / 2  # c_lo / lam_c
    return gamma - jackson_step(d, (1 - a) / (1 + a))


def _g_eff(gap, lower):
    """The governing gap: the smaller edge gap in theta-resolution terms (lower edge: the
    relative gap g_lo times 2 sqrt(a) / (1 + a), a = c_lo / lam_c)."""
    if lower is None:
        return gap
    a = sum(lower) / 2
    return min(gap, (lower[0] - lower[1]) / a * 2 * np.sqrt(a) / (1 + a))


def _passes(d, gap, lam_top, n_top, n_free, lower):
    gamma = _gamma(d, lower)
    t0, t1 = (0.0 if lower is None else _theta(lower[0])), _theta(1 - gap / 2)
    ends = step_value(gamma, [t1] if lower is None else [t0, t1])
    if (ends**2).min() < Q_MIN:
        return False, np.nan
    th, q = _grid(gamma)
    if (q[(th > t0) & (th < t1)] ** 2).min(initial=1.0) < Q_MIN:  # ripples inside the band
        return False, np.nan
    bias = _total_bias(gamma, gap, lam_top, n_top, n_free, lower)
    return bias <= BIAS, bias


def _total_bias(gamma, gap, lam_top, n_top, n_free, lower):
    bias = tail_bias(gamma, gap, lam_top, n_top, n_free)
    if lower is not None:  # the lower tail, from 0 (rigid modes deflated: N(0) = 0)
        bias += _bias(gamma, 0.0, lower[1], n_top, lam_top)
    return bias


def count_degree(gap, lam_top, n_top, n_free, lower=None):
    """(d, bias): the smallest degree with q^2 >= Q_MIN on the band [0, lam_c (1 - gap / 2)]
    and tail bias <= BIAS (lam_top in units of lam_c), by doubling then bisection; (None, nan)
    when no d <= MAX_DEGREE passes (at once when D_RULE / g_eff > MAX_DEGREE). lower =
    (lam_in, lam_bot) in units of lam_c (the lowest in-band eigenvalue and the nearest one
    below the band) switches to the band-pass polynomial on [lam_in, lam_c (1 - gap / 2)],
    with the lower tail over [0, lam_bot] added to the bias."""
    if D_RULE / _g_eff(gap, lower) > MAX_DEGREE:
        return None, np.nan
    hi = 2  # d = 1 always fails: its Jackson factor g_1 = 0, so q = 1/2
    while not (ok := _passes(hi, gap, lam_top, n_top, n_free, lower))[0]:
        hi *= 2
        if hi > 2 * MAX_DEGREE:
            return None, np.nan
    lo, best = hi // 2, (hi, ok[1])
    while best[0] - lo > 1:
        mid = (lo + best[0]) // 2
        ok = _passes(mid, gap, lam_top, n_top, n_free, lower)
        lo, best = (lo, (mid, ok[1])) if ok[0] else (mid, best)
    return best if best[0] <= MAX_DEGREE else (None, np.nan)


def _bound(steps, b, m, t_iteration, tau, rule):
    """A-priori lower bound on the count's seconds: a step runs >= 7 PCG iterations at width
    b, a LOBPCG iteration costs at most about 3 of their applies at width m (factor 2; 0.3, one
    PCG iteration, when tau is looser than the rule). None without a warm iteration time."""
    if not t_iteration:
        return None
    return (0.3 if tau is not None and tau > rule else 2.0) * steps * b / m * t_iteration


def _calibrate(run, d, n_blocks):
    """(predicted seconds of the count, calibration seconds, t_step, t_block): run(s) counts one
    block with at most s steps; s = 1 compiles, s = 3 is timed."""
    t = time.perf_counter()
    run(1)
    tm = run(3)[1]["time"]
    t_step, t_block = tm["step"], tm["block"]
    return n_blocks * (d * t_step + t_block), time.perf_counter() - t, t_step, t_block


def _decide(verify, d, n_probes, block, m, t_ref, t_it, tau, n_free, run, prepare=None):
    """(count?, record) of the completeness-count rule
    (docs/eigensolvers.md#frequency-band-0-f_max). d: the degree (None above MAX_DEGREE); block:
    probes per block; m, t_it: the solver's block width and warm iteration seconds (the a-priori
    bound); t_ref: the solve's seconds. verify True or False: count = verify, with the bound
    recorded. None: skip above MAX_DEGREE or when the bound exceeds t_ref, else prepare() (timed
    into build_s; e.g. the count's preconditioner), then calibrate with run(s) (the count with
    _steps = s) and count iff the prediction <= t_ref."""
    rec = dict(reason=None, predicted_steps=None, predicted_s=None, predicted_kind="unknown",
               build_s=0.0, calibration_s=0.0)  # fmt: skip
    n_blocks = -(-n_probes // block)
    if d is not None:
        bound = _bound(n_blocks * d, block, m, t_it, tau, 0.01 / (d * math.sqrt(n_free)))
        rec.update(predicted_steps=n_blocks * d, predicted_s=bound,
                   predicted_kind="unknown" if bound is None else "bound")  # fmt: skip
    if verify is not None:
        return verify, rec
    if d is None:
        return False, {**rec, "reason": "degree above MAX_DEGREE: the gap is too small to count"}
    if (rec["predicted_s"] or 0) > t_ref:
        return False, {**rec, "reason": "predicted cost (a lower bound) above the solve"}
    if prepare is not None:
        t = time.perf_counter()
        prepare()
        rec["build_s"] = time.perf_counter() - t
    pred, cal, t_step, t_block = _calibrate(run, d, n_blocks)
    rec.update(predicted_s=pred + rec["build_s"] + cal, predicted_kind="calibrated",
               calibration_s=cal, calibrated_step=t_step, calibrated_block=t_block)  # fmt: skip
    if rec["predicted_s"] > t_ref:
        return False, {**rec, "reason": "predicted cost above the solve"}
    return True, rec


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


def _chunks(Ws):
    """(chunk, width): column chunks of the arrays Ws (width CHUNK, each array's last chunk
    zero-padded on the device): one compiled width; a device array is sliced in place."""
    for W in Ws:
        for i in range(0, W.shape[1], CHUNK):
            Wc = jnp.asarray(W[:, i : i + CHUNK])
            w = Wc.shape[1]
            yield (jnp.pad(Wc, ((0, 0), (0, CHUNK - w))) if w < CHUNK else Wc), w


def _project(op, m, Ws, R, MR, Y):
    """Pi Y = Y - W W^T M Y - R R^T M Y, masked, W the columns of the arrays Ws streamed in
    chunks (never all on the device). Y is donated."""
    MY = _mm(op, m, Y)
    for Wc, _ in _chunks(Ws):
        Y = _chunk_step(Y, Wc, MY)
    return _rigid_step(Y, R, MR, m)


_PROGRESS = {}  # verbose state of the running count: d, block width, t0, last line time


def _report(k, it_sum, it_max):
    """Host side of the verbose callback: one line per PRINT_S seconds at most."""
    s, t = _PROGRESS, time.perf_counter()
    if t - s["last"] >= PRINT_S:
        s["last"] = t
        print(f"count: step {int(k)} of {s['d']}, pcg {int(it_sum) / (int(k) * s['b']):.1f} mean"
              f" / {int(it_max)} max, {t - s['t0']:.0f} s", flush=True)  # fmt: skip


@functools.partial(jax.jit, static_argnames="verbose")
def _recurrence(A, P, m, w0, gamma, d, lam_c, tau, maxit, verbose=False):
    """u = sum_k gamma_k T_k(X) w0 for k <= d, X = 2 lam_c T - I, T w by PCG on A (= A_P of
    K + lam_c M). Stops early when a solve stagnates or does not converge (bad). Returns
    (u, steps, iteration sum, iteration max, bad). verbose: a host callback per step
    (scalars only)."""

    def body(s):
        k, w_prev, w, u, it_sum, it_max, _ = s
        t, info = _pcg(A, P, _mm(A.op, m, w), tau, maxit)
        first = k == 1
        w_new = jnp.where(first, 1.0, 2.0) * (2 * lam_c * t - w)
        w_new = w_new - jnp.where(first, 0.0, 1.0) * w_prev
        bad = ~(info["converged"] & ~info["stagnated"]).all()
        its = info["iterations"]
        it_sum, it_max = it_sum + its.sum(), jnp.maximum(it_max, its.max())
        if verbose:
            jax.debug.callback(_report, k, it_sum, it_max)
        return k + 1, w, w_new, u + gamma[k] * w_new, it_sum, it_max, bad

    zero = jnp.zeros((), int)
    s = (zero + 1, jnp.zeros_like(w0), w0, gamma[0] * w0, zero, zero, jnp.zeros((), bool))
    k, _, _, u, it_sum, it_max, bad = lax.while_loop(lambda s: (s[0] <= d) & ~s[6], body, s)
    return u, k - 1, it_sum, it_max, bad


def _pow2(n):
    return 1 << (int(n) - 1).bit_length()


def _check_ortho(op, m, W, seed):
    """Streamed sketch of W^T M W - I (W an array or a tuple of arrays, their columns side by
    side): two passes over W, e = W^T M (W z) - z."""
    Ws = W if isinstance(W, tuple) else (W,)
    k = sum(A.shape[1] for A in Ws)
    key = jax.random.fold_in(jax.random.key(seed), 2**32 - 1)
    z = np.asarray(jax.random.normal(key, (k, SKETCH)))
    v, j = jnp.zeros((Ws[0].shape[0], SKETCH)), 0
    for Wc, w in _chunks(Ws):
        zc = np.zeros((CHUNK, SKETCH))
        zc[:w] = z[j : j + w]
        v, j = v + Wc.astype(v.dtype) @ jnp.asarray(zc), j + w
    Mv = _mm(op, m, v)
    e = np.concatenate([np.asarray(Wc.astype(Mv.dtype).T @ Mv)[:w] for Wc, w in _chunks(Ws)])
    e = e - z
    tol = max(ORTHO_TOL.get(str(A.dtype), ORTHO_TOL["float32"]) for A in Ws)
    if not np.linalg.norm(e) <= tol * np.linalg.norm(z):
        raise ValueError(
            f"W must be M-orthonormal on the free DOFs: sketch error "
            f"{np.linalg.norm(e) / np.linalg.norm(z):.2e} > {tol:g}"
        )


def _inputs(op, free, R, n_probes, maxit, block, degree, tau):
    """Shared checks: (free, m, R, MR)."""
    n = op.n_dof
    free = np.asarray(free)
    if free.shape != (n,) or free.dtype != bool:
        raise ValueError(f"free must be a ({n},) bool mask")
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
    return free, m, R, MR


def _columns(op, W, name):
    """W (an array or a tuple of arrays, each (n_dof, k_i)) as a tuple, and its total columns."""
    Ws = W if isinstance(W, tuple) else (W,)
    for A in Ws:
        if A.ndim != 2 or A.shape[0] != op.n_dof:
            raise ValueError(f"{name} must have shape ({op.n_dof}, k), got {A.shape}")
    return Ws, sum(A.shape[1] for A in Ws)


def _estimate(op, free, m, Ws, R, MR, precond, gamma, d, lam_c, tau, *, n_probes, block, seed,
              maxit, verbose, _steps, t0):  # fmt: skip
    """The probe loop shared by count and band_count: (estimate, info updates). gamma: the
    (d + 1,) coefficients; W the columns of the arrays Ws."""
    A = ap_operator(op, free, lam_c)
    g = np.zeros(_pow2(d + 1))
    g[: d + 1] = gamma
    gamma = jnp.asarray(g)
    n_blocks = -(-n_probes // block)
    # probe j from fold_in(key, j): independent of the block size and of n_probes
    fold = jax.vmap(jax.random.fold_in, (None, 0))
    keys = fold(jax.random.key(seed), jnp.arange(n_blocks * block))
    s_all = np.full(n_blocks * block, np.nan)
    it_sum = it_max = steps = 0
    t_sample = t_rec = t_proj = 0.0
    d_run = d if _steps is None else min(d, _steps)
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
        w0 = _project(op, m, Ws, R, MR, y)
        del y  # only w0 and the recurrence are alive in the step phase
        w0.block_until_ready()
        tc = time.perf_counter()
        if verbose:
            _PROGRESS.update(d=d_run, b=block, t0=t0, last=time.perf_counter())
        u, k, isum, imax, bad = _recurrence(
            A, precond, m, w0, gamma, d_run, float(lam_c), tau, maxit, verbose=verbose
        )
        del w0
        k, bad = int(k), bool(bad)
        td = time.perf_counter()
        steps, it_sum, it_max = steps + k, it_sum + int(isum), max(it_max, int(imax))
        if not bad:
            u = _project(op, m, Ws, R, MR, u)
            s_all[i * block : (i + 1) * block] = np.asarray(jnp.sum(u * _mm(op, m, u), 0))
        del u
        t_rec += td - tc
        t_proj += time.perf_counter() - td + tc - tb
        if verbose:
            print(f"count: block {i + 1} of {n_blocks}, {k} steps, "
                  f"{'failed' if bad else 'done'}, {time.perf_counter() - t0:.0f} s",
                  flush=True)  # fmt: skip
        if bad:
            ok = False
            break
    probes = s_all[:n_probes]  # NaN for the columns of a stopped or skipped block
    estimate = float(probes.mean()) if ok else np.nan
    return estimate, dict(
        probes=probes,
        std=float(probes.std(ddof=1)) if ok and n_probes > 1 else np.nan,
        steps=steps,
        pcg_iterations=(it_sum / max(steps * block, 1), it_max),
        verifiable=ok,
        passed=bool(ok and estimate < 0.5),
        time=dict(
            sampling=t_sample,
            recurrence=t_rec,
            projection=t_proj,
            step=t_rec / max(steps, 1),
            block=(t_sample + t_proj) / (i + 1),
            total=time.perf_counter() - t0,
        ),  # fmt: skip
    )


def _run(op, free, m, Ws, R, MR, info, lower, gap, lam_top, n_top, lam_c, *, precond, n_probes,
         seed, tau, degree, block, maxit, verbose, _steps, _predict, t0):  # fmt: skip
    """Degree, tau and the probe loop for count and band_count; info is filled in place.
    _predict: stop after the degree (the count's cost prediction, nothing run)."""
    n_free = int(free.sum())
    block = n_probes if block is None else min(block, n_probes)
    info.update(lam_c=lam_c, gap=_g_eff(gap, lower), block=block, steps=0, n_top=n_top,
                capped=False, pcg_iterations=(np.nan, 0), verifiable=False, passed=False,
                probes=np.full(n_probes, np.nan), std=np.nan)  # fmt: skip
    if degree is None:
        d, bias = count_degree(gap, lam_top, n_top, n_free, lower)
        if d is None:  # cost cap: not run
            info.update(degree=math.ceil(D_RULE / info["gap"]), tau=tau, bias=np.nan, capped=True,
                        time=dict(total=time.perf_counter() - t0))  # fmt: skip
            return np.nan, info
    else:
        d = degree
        bias = _total_bias(_gamma(d, lower), gap, lam_top, n_top, n_free, lower)
    tau = 0.01 / (d * math.sqrt(n_free)) if tau is None else float(tau)
    info.update(degree=d, tau=tau, bias=bias)
    if _predict:
        info.update(time=dict(total=time.perf_counter() - t0))
        return np.nan, info
    precond = jacobi(op, free, lam_c) if precond is None else precond
    est, out = _estimate(op, free, m, Ws, R, MR, precond, _gamma(d, lower), d, lam_c, tau,
                         n_probes=n_probes, block=block, seed=seed, maxit=maxit,
                         verbose=verbose, _steps=_steps, t0=t0)  # fmt: skip
    info.update(out)
    return est, info


def count(
    op, free, W, lam, lam_last, lam_next, *, R=None, precond=None, n_probes=50, seed=0,
    tau=None, degree=None, block=None, maxit=1000, verbose=False, _steps=None, _predict=False,
):  # fmt: skip
    """Stochastic count of the eigenvalues below lam_c = (lam_last + lam_next) / 2 that are
    missing from W. Returns (estimate, info); passed = verifiable and estimate < 0.5.

    W (n, k): the computed modes, M-orthonormal (NumPy or device, fp64 or fp32; or a tuple of
    such arrays, their columns side by side, streamed without a copy), with eigenvalues lam
    (k,); lam_last: the last mode below the cut (or the band edge when none is returned),
    lam_next: the first mode above it, both in lam's units, lam_next <= max(lam) (the tail model
    needs every mode up to lam_next in W). R: the rigid modes (M-orthonormal).
    precond: SPD preconditioner for K + lam_c M (default jacobi(op, free, lam_c); PR 14b passes
    the fp64 multigrid). degree overrides the degree rule, tau the inner tolerance
    0.01 / (d sqrt(n_free)); block: probes per block (default n_probes); maxit: per PCG solve.
    A stagnated or unconverged solve makes the count unverifiable: the remaining work is
    skipped and the estimate is NaN. verbose: progress lines (at most one per PRINT_S seconds
    in the recurrence, one per block; compiles its own loop variant). info["time"]: sampling
    (the y solves), recurrence (the recurrences alone), projection (the projections and the
    reduction), step (recurrence per step), block ((sampling + projection) per block run) and
    total. _steps (internal, the calibration in solve_modes): run at most this many steps per
    block, with the degree-d polynomial's compiled shape (the estimate is then not the count's).
    _predict (internal, si_checked without a count): return after the degree, nothing run.
    """
    check_x64("count")
    t0 = time.perf_counter()
    free, m, R, MR = _inputs(op, free, R, n_probes, maxit, block, degree, tau)
    lam = np.asarray(lam, float)
    if lam.ndim != 1 or len(lam) == 0:
        raise ValueError("lam must be a non-empty 1-D array")
    Ws, k = _columns(op, W, "W")
    if k != len(lam):
        raise ValueError(f"W must have shape ({op.n_dof}, {len(lam)}), got {k} columns")
    if not 0 <= lam_last < lam_next <= lam.max():
        raise ValueError(
            f"need 0 <= lam_last < lam_next <= max(lam), got {lam_last}, {lam_next}, {lam.max()}"
        )
    if not (_predict or _steps):  # predictions and calibrations: the full count checks W
        _check_ortho(op, m, Ws, seed)
    lam_c = (lam_last + lam_next) / 2
    gap = (lam_next - lam_last) / lam_c
    return _run(op, free, m, Ws, R, MR, {}, None, gap, lam.max() / lam_c,
                len(lam) + R.shape[1], lam_c, precond=precond, n_probes=n_probes, seed=seed,
                tau=tau, degree=degree, block=block, maxit=maxit, verbose=verbose,
                _steps=_steps, _predict=_predict, t0=t0)  # fmt: skip


def _edges(lam, lo, hi, lam_out):
    """(a, b, nb_lo, nb_hi): the outermost in-band eigenvalues (the band edges when lam is
    empty) and the nearest lam_out beyond them."""
    lam, lam_out = np.asarray(lam, float), np.asarray(lam_out, float)
    a = lam.min() if len(lam) else lo
    b = lam.max() if len(lam) else hi
    if ((lam_out >= a) & (lam_out <= b)).any():
        raise ValueError(f"lam_out has values inside the in-band interval [{a}, {b}]")
    below, above = lam_out[lam_out < a], lam_out[lam_out > b]
    if not len(below) or not len(above):
        raise ValueError("lam_out needs a value below and above the band (the edge neighbours)")
    return a, b, below.max(), above.min()


def band_edges(lam, lo, hi, lam_out):
    """(c_lo, c_hi, g_lo, g_hi): the counting edges of the band [lo, hi] with in-band modes lam
    (possibly none) and known out-of-band eigenvalues lam_out, each midway between the outermost
    in-band eigenvalue (the band edge when lam is empty) and the nearest lam_out beyond it, and
    the edge gaps relative to them. Raises when lam_out has no value on a side or one inside the
    in-band interval."""
    a, b, nb_lo, nb_hi = _edges(lam, lo, hi, lam_out)
    c_lo, c_hi = (nb_lo + a) / 2, (nb_hi + b) / 2
    return c_lo, c_hi, (a - nb_lo) / c_lo, (nb_hi - b) / c_hi


def band_count(
    op, free, V, lam, lo, hi, V_out, lam_out, *, R=None, precond=None, n_probes=50, seed=0,
    tau=None, degree=None, block=None, maxit=1000, verbose=False, _steps=None,
    _predict=False,
):  # fmt: skip
    """Stochastic count of the eigenvalues in the interior band [lo, hi] (0 < lo < hi) missing
    from V, by a Jackson band-pass polynomial of T = (K + c_hi M)^-1 M between the counting
    edges of band_edges. Returns (estimate, info); passed = verifiable and estimate < 0.5.

    V (n, k >= 0; or a tuple of arrays, as count's W), lam (k,): the in-band modes found (the
    out-of-band pairs are added here, not passed in V). V_out (n, j), lam_out (j,): known
    converged out-of-band pairs, at least the nearest one on each side (si_slice's neighbours);
    all are deflated with V. All M-orthonormal together. precond: SPD preconditioner for
    K + c_hi M (default jacobi(op, free, c_hi)). The Weyl tail model (no mode index in a band)
    is fitted to the in-band modes and the nearest out-of-band pair on each side. Other options
    and info as count's, plus edges (c_lo, c_hi) and gaps (g_lo, g_hi); lam_c = c_hi, gap = the
    governing gap, n_top the fitted model count at the upper neighbour. _steps, _predict: as
    count's.
    """
    check_x64("band_count")
    t0 = time.perf_counter()
    free, m, R, MR = _inputs(op, free, R, n_probes, maxit, block, degree, tau)
    if not 0 < lo < hi:
        raise ValueError(f"need 0 < lo < hi (a band from 0: count), got {lo}, {hi}")
    lam, lam_out = np.atleast_1d(np.asarray(lam, float)), np.asarray(lam_out, float)
    Vs, k = _columns(op, V, "V")
    if k != len(lam):
        raise ValueError(f"V must have shape ({op.n_dof}, {len(lam)}), got {k} columns")
    if V_out.ndim != 2 or V_out.shape != (op.n_dof, len(lam_out)):
        raise ValueError(f"V_out must have shape ({op.n_dof}, {len(lam_out)}), got {V_out.shape}")
    c_lo, c_hi, g_lo, g_hi = band_edges(lam, lo, hi, lam_out)
    a, _, nb_lo, nb_hi = _edges(lam, lo, hi, lam_out)
    Ws = (*Vs, V_out)
    if not (_predict or _steps):
        _check_ortho(op, m, Ws, seed)
    top, bot = nb_hi / c_hi, nb_lo / c_hi
    n_top = (len(lam) + 1) / (top**WEYL_A - bot**WEYL_A) * top**WEYL_A  # n - 1 intervals
    info = dict(edges=(c_lo, c_hi), gaps=(g_lo, g_hi))
    return _run(op, free, m, Ws, R, MR, info, (a / c_hi, bot), g_hi, top, n_top, c_hi,
                precond=precond, n_probes=n_probes, seed=seed, tau=tau, degree=degree,
                block=block, maxit=maxit, verbose=verbose, _steps=_steps, _predict=_predict,
                t0=t0)  # fmt: skip
