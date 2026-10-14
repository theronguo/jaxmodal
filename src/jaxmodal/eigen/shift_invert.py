"""Block shift-invert Lanczos for the eigenvalues of one slice [lo, hi] of the masked pencil
(K_m, M_m): inner MINRES on K - sigma M with an SPD preconditioner of K + sigma M, full
M-reorthogonalization, Rayleigh-Ritz on (K, M); si_checked adds the completeness count and the
restarts. See docs/eigensolvers.md#shift-invert-slices."""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from ..linalg.krylov import minres
from ..operators import ap_operator
from ..operators.common import check_x64
from .counting import _check_ortho, band_count, count
from .lobpcg import _K, _M, C_FLOOR, _deflate, _insert, _norms, _resid, _svqb_t

SIGMA_MOVE = 0.01  # sigma moves by this fraction of the slice width when it hits an eigenvalue
STAGNATION = 1e3  # first-solve residual (relative to max(tau, TAU_MIN)) above which sigma moves
TAU_MIN = 1e-10  # above the fp64 MINRES floor (2e-14 - 1e-12): a tiny tau does not move sigma
INNER_GROWTH = 10  # later inner solves are capped at this multiple of the first one
TAU_FAIL = 1e-4  # tau_eff above this ends the slice ("inner solves failed")
REFILLS = 3  # random refills of dropped columns before a breakdown error
RESTARTS = 2  # a slice whose count stays >= 0.5 is returned flagged after this many restarts
WMODE = -1  # _slice_check: a side closed by a deflated known mode (W), not a Ritz value


def _chunk(b):
    """Projector and Ritz chunk width c for block b (as LOBPCG's)."""
    return max(b, min(4 * b, 64))


def _up(x, c):
    """x rounded up to a multiple of c."""
    return -(-x // c) * c


def _consts(op, m, key, tol):
    """lobpcg._resid's constants: the norm estimates of K and M from one random vector."""
    nK, nM = map(float, _norms(op, m, m * jax.random.normal(key, (op.n_dof, 1))))
    return jnp.array([nK, nM, tol, C_FLOOR])


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
    return _explicit(_combine(Q, L0, k, Yc, c), lam, op, m, consts)


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


def _cands(f, mu, b, sigma, tau_eff, idx):
    """_slice_check's resid: (r, t) of the Ritz pairs idx."""
    return _chunked(f, idx, mu[idx], b, sigma, tau_eff)


def _extend_chain(v, i0, i1, chain):
    """[i0, i1) of the ascending v extended over relative gaps < chain."""
    if i1 > i0:
        while i0 > 0 and v[i0] - v[i0 - 1] < chain * abs(v[i0 - 1]):
            i0 -= 1
        while i1 < len(v) and v[i1] - v[i1 - 1] < chain * abs(v[i1 - 1]):
            i1 += 1
    return i0, i1


def _slice_sets(mu, lo, hi, r):
    """(inside, below, above) of the ascending Ritz values mu for the slice [lo, hi]: inside =
    the values in [lo (1 - 100 r), hi (1 + 100 r)] extended over chains with relative gaps
    < 100 r on both sides; below / above = the nearest value beyond the extended set (None if
    there is none, or for below when lo = 0)."""
    i0 = int(np.searchsorted(mu, lo * (1 - 100 * r), side="left"))
    i1 = int(np.searchsorted(mu, hi * (1 + 100 * r), side="right"))
    i0, i1 = _extend_chain(mu, i0, i1, 100 * r)
    below = i0 - 1 if i0 > 0 and lo > 0 else None
    above = i1 if i1 < len(mu) else None
    return np.arange(i0, i1), below, above


def _slice_check(mu, lo, hi, r, resid, lam_W=()):
    """(done, inside, below, above, r_acc, res, ritz): the sets of mu at r, re-formed with r =
    r_acc while r_acc (the largest threshold over the candidates, inside plus the neighbours)
    exceeds r. The deflated known values lam_W join mu as converged candidates (no threshold,
    never returned): inside holds Ritz indices only, below / above a Ritz index, WMODE (a W value
    closes that side) or None. resid(idx) returns (residuals, thresholds) of the new candidate
    indices idx. done: every candidate converged, every needed neighbour present and r_acc <= r.
    res: {index: (r, t)} of every candidate evaluated; ritz: (values, converged) of mu and lam_W
    merged, ascending (W True, Ritz values never evaluated False)."""
    nm = len(mu)
    v = np.r_[mu, np.asarray(lam_W, float)]
    order = np.argsort(v, kind="stable")
    v = v[order]
    res = {}
    while True:
        ins, bel, abv = _slice_sets(v, lo, hi, r)
        cand = [order[i] for i in (*ins, bel, abv) if i is not None]
        rit = [i for i in cand if i < nm]
        new = np.array([i for i in rit if i not in res], int)
        if len(new):
            rr, tt = resid(new)
            res.update(zip(new.tolist(), zip(rr, tt, strict=True), strict=True))
        r_acc = max((res[i][1] for i in rit), default=r)
        if r_acc <= r:
            ok = all(res[i][0] <= res[i][1] for i in rit)
            need = abv is not None and (lo == 0 or bel is not None)
            side = lambda i: None if i is None else WMODE if order[i] >= nm else order[i]  # noqa: E731
            conv = np.array([j >= nm or (j in res and res[j][0] <= res[j][1]) for j in order])
            inside = [order[i] for i in ins if order[i] < nm]
            return ok and need, inside, side(bel), side(abv), r_acc, res, (v, conv)
        r = r_acc


def _random_block(key, n, b):
    return jax.random.normal(key, (n, b))


def si_slice(
    op, free, lo, hi, precond, *, capacity, sigma=None, R=None, W=None, lam_W=None, tol=1e-6,
    tau=1e-8, block=6, maxit_inner=20000, seed=0, verbose=False,
):  # fmt: skip
    """Eigenpairs of the masked pencil (K_m, M_m) in the slice [lo, hi] (eigenvalues, 0 <= lo <
    hi) by block shift-invert Lanczos at sigma (default the midpoint). precond: SPD
    preconditioner of K + sigma M (fp64; a GMG with fp32 levels raises); R: M-orthonormal rigid
    modes; W: known M-orthonormal modes to deflate (host array, never returned), lam_W their
    eigenvalues (optional: they then count as converged candidates of the stop rule, so a known
    mode beyond an edge closes that side; an unknown mode between still appears as a nearer
    Ritz value that must converge). capacity:
    Lanczos columns (rounded up to a multiple of the chunk width; the basis also stops within
    block columns of the free space left by R and W).
    Pair i is accepted at r_i <= t_i = max(tol, 10 tau_eff delta_i, round-off floor).
    verbose: print one progress line per Lanczos step (and on a sigma move).

    Returns lam (ascending: the converged pairs in [lo, hi] widened by 100 r_acc and over
    clusters straddling an edge), V (host fp64, M-orthonormal) and info:
    converged, reason, sigma, sigma_moved, move_iterations (of the discarded solve, or None),
    steps, basis, minres_iterations, inner_cap, inner_maxit, tau_eff, tau_raised, r_acc,
    delta_max, residuals, thresholds, neighbours ({"lo": (lam, v) or None, "hi": ...}; Ritz
    neighbours only), capacity_full, edge_unreached, ritz ((values, converged) of the last
    step's Ritz values and lam_W merged, ascending; None before the first step), time."""
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
    capacity = _up(capacity, c)
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
    lam_W = np.zeros(0) if lam_W is None else np.asarray(lam_W, float)
    if len(lam_W) not in (0, nw) or lam_W.ndim != 1:
        raise ValueError(f"lam_W must have shape ({nw},), got {lam_W.shape}")
    L0 = _up(nw, c)  # the Lanczos columns start at a chunk boundary
    Q = jnp.zeros((n, L0 + capacity))
    for i in range(0, nw, c):  # c columns at a time (zero-padded): no n x nW device copy
        Wc = np.zeros((n, c))
        Wc[:, : min(c, nw - i)] = W[:, i : i + c]
        Q = _insert(Q, jnp.asarray(Wc), jnp.asarray(i, jnp.int32))
    key, sub = jax.random.split(jax.random.key(seed))
    consts = _consts(op, m, sub, tol)

    n_avail = int(free.sum()) - R.shape[1] - nw  # the Krylov space cannot grow beyond it
    if n_avail < 2 * b:
        raise ValueError(f"2 block = {2 * b} exceeds the free space left by R and W ({n_avail})")
    limit = min(capacity, n_avail)

    def add(Y, key):  # append the new block, dropped columns refilled with random ones
        nonlocal Q, k
        filled = jnp.asarray(L0 + k, jnp.int32)
        for i in range(REFILLS + 1):
            Y, keep = _extend(m * Y, op, m, R, MR, Q, filled, c=c)
            if keep.all():
                break
            if i == REFILLS:
                raise RuntimeError(f"Lanczos block still rank-deficient after {REFILLS} refills")
            key, sub = jax.random.split(key)
            Y = jnp.where(keep, Y, _random_block(sub, n, b))
        Q, g = _append(Q, Y, op, m, jnp.asarray(L0, jnp.int32), jnp.asarray(k, jnp.int32), c=c)
        G[: k + b, k : k + b] = np.asarray(g[: k + b])
        G[k : k + b, :k] = G[:k, k : k + b].T
        k += b
        return key

    k = 0  # Lanczos columns
    G = np.zeros((capacity, capacity))
    key, sub = jax.random.split(key)
    key = add(_random_block(sub, n, b), key)
    A = ap_operator(op, free, -sigma)
    tau_eff, r = tau, tol
    iters, cap, moved, inner_maxit, reason = [], maxit_inner, None, False, None
    seen = {"lo": lo == 0, "hi": False}
    times = dict(setup=time.perf_counter() - t0, minres=0.0, ortho=0.0, ritz=0.0)
    done, last, ritz = False, None, None
    while True:
        if k + b > limit:
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
        if not np.isfinite(res).all():
            reason = "inner solves failed"
            break
        stag = STAGNATION * max(tau, TAU_MIN)
        if not iters and moved is None and bad.any() and res[bad].max() > stag:
            sigma += SIGMA_MOVE * (hi - lo)  # sigma on an eigenvalue: move once, repeat
            A, moved = ap_operator(op, free, -sigma), its  # the discarded solve's iterations
            if verbose:
                print(f"si_slice: sigma moved to {sigma:.6g} ({its} its discarded)", flush=True)
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
        key = add(X, key)
        times["ortho"] += time.perf_counter() - t
        t = time.perf_counter()
        mu, Yr = np.linalg.eigh((G[:k, :k] + G[:k, :k].T) / 2)
        f = functools.partial(_ritz_chunk, Q, L0, k, Yr, op, m, consts, c)
        resid = functools.partial(_cands, f, mu, b, sigma, tau_eff)
        done, inside, below, above, r, cres, ritz = _slice_check(mu, lo, hi, r, resid, lam_W)
        ok = [i for i in (*inside, below, above)
              if i == WMODE or (i is not None and cres[i][0] <= cres[i][1])]  # fmt: skip
        seen["lo"] |= below in ok
        seen["hi"] |= above in ok
        last = (
            mu,
            Yr,
            [i for i in inside if i in ok],
            [(e, i) for e, i in (("lo", below), ("hi", above)) if i in ok and i != WMODE],
        )
        times["ritz"] += time.perf_counter() - t
        if verbose:
            print(f"si_slice: step {len(iters)}, basis {k}, minres {its} its (res {res.max():.1e}),"
                  f" tau_eff {tau_eff:.1e}, accepted {len(last[2])}/{len(inside)}, r {r:.1e},"
                  f" {time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
        if done:
            break
    # final stage: the accepted pairs and neighbours, V formed on the host, Q released
    lam_out, V, res_out, thr_out = np.zeros(0), np.zeros((n, 0)), np.zeros(0), np.zeros(0)
    nb = {"lo": None, "hi": None}
    converged = done
    if last is not None:
        mu, Yr, acc, nbs = last
        sel = acc + [i for _, i in nbs]
        Vs = _form_v(Q, L0, Yr[:, sel], c) if sel else np.zeros((n, 0))
        del f, resid, Q  # release the basis before the residual pass
        g = functools.partial(_v_chunk, Vs, op, m, consts)
        rr, tt = _chunked(g, np.arange(len(sel)), mu[sel], b, sigma, tau_eff)
        na = len(acc)
        converged = done and bool((rr[:na] <= tt[:na]).all())
        lam_out, V, res_out, thr_out = mu[acc], Vs[:, :na].copy(), rr[:na], tt[:na]
        for j, (name, i) in enumerate(nbs):
            nb[name] = (float(mu[i]), Vs[:, na + j].copy())
    unreached = [e for e in ("lo", "hi") if not seen[e]]
    delta = np.abs(lam_out - sigma) / np.abs(lam_out) if len(lam_out) else np.zeros(0)
    info = dict(
        converged=converged,
        reason=reason,
        sigma=sigma,
        sigma_moved=moved is not None,
        move_iterations=moved,
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
        edge_unreached="both" if len(unreached) == 2 else (unreached + [None])[0],
        ritz=ritz,
        time=times,
    )
    return lam_out, V, info


def _beyond(a, b, nb, W, lam_W):
    """[(lam, v)] of the nearest known out-of-band pair below a and above b: si_slice's
    neighbours nb or the caller's pairs (W, lam_W), whichever is nearer; None for a side with
    neither."""
    out = []
    for side, sel, pick in (("lo", lam_W < a, np.argmax), ("hi", lam_W > b, np.argmin)):
        p = nb[side]  # a guard: si_slice's neighbours lie beyond its returned modes
        cand = [p] if p is not None and (p[0] < a if side == "lo" else p[0] > b) else []
        if sel.any():
            i = np.flatnonzero(sel)[pick(lam_W[sel])]
            cand.append((float(lam_W[i]), W[:, i]))
        out.append(min(cand, key=lambda p: abs(p[0] - (a if side == "lo" else b)), default=None))
    return out


def _count_slice(op, free, V_c, lam_c, lo, hi, nb_lo, nb_hi, **kw):
    """The completeness count of the slice [lo, hi] with the found modes V_c (a tuple of arrays)
    and lam_c, and the nearest known out-of-band pairs nb_lo, nb_hi ((lam, v); nb_lo unused at
    lo = 0): band_count, or at lo = 0 count (the step polynomial below the upper pair)."""
    if lo > 0:
        out = [nb_lo, nb_hi]
        return band_count(op, free, V_c, lam_c, lo, hi, np.stack([p[1] for p in out], 1),
                          np.array([p[0] for p in out]), **kw)  # fmt: skip
    e = lam_c.max() if len(lam_c) else hi
    return count(op, free, (*V_c, nb_hi[1][:, None]), np.r_[lam_c, nb_hi[0]], e, nb_hi[0], **kw)


def _unverified(ci):
    """The reason of an unverifiable count."""
    if ci["capped"]:
        return "degree above MAX_DEGREE: the gap is too small to count"
    return "count unverifiable: an inner solve failed"


def _count_out(ci, ran, reason, est, estimates, flagged, restarts):
    """si_checked's info["count"]: the count's info plus the outcome."""
    return {"passed": False, "verifiable": False, **ci, "ran": ran, "reason": reason,
            "estimate": est, "estimates": estimates, "flagged": flagged,
            "restarts": restarts}  # fmt: skip


def si_checked(
    op, free, lo, hi, precond, *, capacity, W=None, lam_W=None, count_precond=None,
    count_options=None, verify=True, seed=0, verbose=False, **slice_options,
):  # fmt: skip
    """si_slice on [lo, hi], then the completeness count of the slice, with restarts: a
    verifiable count >= 0.5 flags the slice and reruns si_slice with a fresh random block (seed
    + j), deflated by the in-band modes found so far (within the first run's device memory: the
    capacity drops by the extra deflation columns), and counts again with fresh probes; after
    RESTARTS restarts it is returned flagged. lo > 0: band_count with the nearest out-of-band
    pair on each side (si_slice's neighbours or the caller's W, whichever is nearer); lo = 0:
    count, the step polynomial, below the nearest pair above. W (host), lam_W: the caller's
    known M-orthonormal modes (e.g. adjacent slices'), deflated by si_slice (and converged
    candidates of its stop rule). verify False: the slice alone, no count and no restarts
    (info["count"] then holds the predicted degree and gap, reason "not requested").
    count_precond: the count's preconditioner (default precond, the slice's). count_options:
    count's options (verbose defaults to verbose). slice_options: si_slice's.

    The caller's W modes inside [lo, hi] or the returned span count as found (deflated in the
    count, not returned).

    Returns lam, V (all in-band modes found, ascending) and info: the last run's si_slice info
    (r_acc, tau_eff the max over runs; residuals, thresholds of the returned pairs) plus runs
    (each run's info) and count: the last count's info (passed and verifiable False when none
    ran) plus ran, reason, estimate, estimates, flagged (some verifiable count >= 0.5) and
    restarts. reason: None (passed), "slice not converged", "restart capacity", "degree above
    MAX_DEGREE: the gap is too small to count", "count unverifiable: an inner solve failed" or
    "count >= 0.5 after RESTARTS restarts". A failed check is not passed; only a verifiable
    count >= 0.5 flags it."""
    n = op.n_dof
    if (W is None) != (lam_W is None):
        raise ValueError("W and lam_W must be given together")
    W = np.zeros((n, 0)) if W is None else np.asarray(W, np.float64)
    lam_W = np.zeros(0) if lam_W is None else np.asarray(lam_W, float)
    if W.ndim != 2 or W.shape != (n, len(lam_W)):
        raise ValueError(f"W must have shape ({n}, {len(lam_W)}) (lam_W), got {W.shape}")
    opts = {"verbose": verbose, **(count_options or {})}
    c0 = opts.pop("seed", 0)
    b = slice_options.get("block", 6)
    c, nw = _chunk(b), len(lam_W)
    lam_f, V_f, res, thr = np.zeros(0), np.zeros((n, 0)), np.zeros(0), np.zeros(0)
    runs, estimates, ci, est, reason, flagged = [], [], {}, np.nan, None, False
    for j in range(RESTARTS + 1):
        # the restart's Q (n, L0 + capacity) within the first run's (capacity rounded to c)
        # (the first run takes capacity as given, so si_slice raises on a bad one)
        cap = _up(capacity, c) - (_up(nw + len(lam_f), c) - _up(nw, c)) if j else capacity
        if j and cap < 2 * b:
            reason = "restart capacity"
            break
        Wj, lj = (np.hstack([W, V_f]), np.r_[lam_W, lam_f]) if len(lam_f) else (W, lam_W)
        lam, V, info = si_slice(op, free, lo, hi, precond, capacity=cap,
                                W=Wj if Wj.shape[1] else None, lam_W=lj if len(lj) else None,
                                seed=seed + j, verbose=verbose, **slice_options)  # fmt: skip
        runs.append(info)
        if j == 0:  # si_slice returns them sorted: no copy
            lam_f, V_f, res, thr = lam, V, info["residuals"], info["thresholds"]
        else:  # one host copy, the columns written into place
            k1 = len(lam_f)
            lam_f, res = np.r_[lam_f, lam], np.r_[res, info["residuals"]]
            thr = np.r_[thr, info["thresholds"]]
            order = np.argsort(lam_f, kind="stable")
            pos = np.argsort(order)  # the sorted position of each column
            Vn = np.empty((n, len(lam_f)))
            Vn[:, pos[:k1]], Vn[:, pos[k1:]] = V_f, V
            lam_f, V_f, res, thr = lam_f[order], Vn, res[order], thr[order]
        if not info["converged"]:
            reason = "slice not converged"
            break
        # the caller's modes inside [lo, hi] or the returned span (si_slice's edge margins)
        x0, x1 = lam_f.min(initial=lo), lam_f.max(initial=hi)
        inb = (lam_W >= x0) & (lam_W <= x1)
        W_in, lam_in = W[:, inb], lam_W[inb]
        lam_c = np.r_[lam_f, lam_in]  # the count's in-band set: found, not returned
        V_c = (V_f, W_in)
        a = lam_c.min() if len(lam_c) else lo
        e = lam_c.max() if len(lam_c) else hi
        nb_lo, nb_hi = _beyond(a, e, info["neighbours"], W, lam_W)
        if nb_hi is None or (lo > 0 and nb_lo is None):
            raise RuntimeError("si_checked: no converged neighbour to place a counting edge")
        P = precond if count_precond is None else count_precond
        kw = dict(R=slice_options.get("R"), precond=P, seed=c0 + j, **opts)
        est, ci = _count_slice(op, free, V_c, lam_c, lo, hi, nb_lo, nb_hi, _predict=not verify,
                               **kw)  # fmt: skip
        if not verify:
            reason = "not requested"
            break
        estimates.append(est)
        miss = ci["verifiable"] and est >= 0.5
        flagged |= miss
        action = ("pass" if ci["passed"] else "unverifiable" if not ci["verifiable"]
                  else f"restart {j + 1}" if j < RESTARTS else "flagged")  # fmt: skip
        if verbose:
            print(f"si_checked: count {est:.3g} (degree {ci['degree']}, {ci['steps']} steps,"
                  f" {ci['time']['total']:.0f} s): {action}", flush=True)  # fmt: skip
        if not miss:
            if not ci["verifiable"]:
                reason = _unverified(ci)
            break
        if j == RESTARTS:
            reason = f"count >= 0.5 after {RESTARTS} restarts"
    out = dict(runs[-1]) if runs else {}
    out.update(
        r_acc=max((r["r_acc"] for r in runs), default=np.nan),
        tau_eff=max((r["tau_eff"] for r in runs), default=np.nan),
        residuals=res,
        thresholds=thr,
        runs=runs,
        count=_count_out(ci, bool(estimates), reason, est, estimates, flagged, len(runs) - 1),
    )
    return lam_f, V_f, out


# ---------------------------------------------------------------- slicing a band (si_band)


def _cap_error(a, b, capacity, why):
    return ValueError(
        f"slice [{a:.6g}, {b:.6g}] fills capacity {capacity} without converging a gap >= 100"
        f" r_acc on either side ({why}): raise capacity (measured: a slice needs about 50-85"
        " columns before it converges anything); lo may be below lambda_1 or hi above the"
        " spectrum"
    )


def _starts(v, chain):
    """Chain starts of the ascending v: True where the relative gap below is >= chain."""
    return np.r_[True, np.diff(v) >= chain * np.abs(v[:-1])]


def _window(v, a, b, margin, chain):
    """Indices of the ascending v in [a (1 - margin), b (1 + margin)] extended over chains of
    relative gaps < chain, plus the nearest value (with its chain) beyond each end."""
    i0 = int(np.searchsorted(v, a * (1 - margin), "left"))
    i1 = int(np.searchsorted(v, b * (1 + margin), "right"))
    i0, i1 = _extend_chain(v, i0, i1, chain)
    return np.arange(*_extend_chain(v, max(i0 - 1, 0), min(i1 + 1, len(v)), chain))


def _split(v, conv, sigma, a, b, r):
    """The span of a full slice [a, b] from its last step's values v (Ritz values and W merged,
    ascending) with converged flags conv, at its r_acc r. The run: the maximal sequence of whole
    converged chains (relative gaps < 100 r) holding the converged value nearest sigma. A side
    is done when the run reaches the value beyond the edge (si_slice's rule; at a = 0 every value
    below); otherwise it is cut at the largest gap >= 100 r between run values inside (a, b), in
    its half of them (gaps g <= (k - 2) // 2 below, the rest above). Returns None (nothing
    converged) or (s_lo, s_hi, done_lo, done_hi), a cut or None per side."""
    if not len(v):
        return None
    start = _starts(v, 100 * r)
    cid = np.cumsum(start) - 1
    ok = np.logical_and.reduceat(conv, np.flatnonzero(start))  # whole chains converged
    cand = np.flatnonzero(ok[cid])
    if not len(cand):
        return None
    q0 = q1 = cid[cand[np.argmin(np.abs(v[cand] - sigma))]]
    while q0 > 0 and ok[q0 - 1]:
        q0 -= 1
    while q1 < len(ok) - 1 and ok[q1 + 1]:
        q1 += 1
    run = np.flatnonzero((cid >= q0) & (cid <= q1))
    _, below, above = _slice_sets(v, a, b, r)
    done_lo = run[0] == 0 if a == 0 else below is not None and run[0] <= below
    done_hi = above is not None and run[-1] >= above
    idx = run[(v[run] > a) & (v[run] < b)]
    gaps = (v[idx[1:]] - v[idx[:-1]]) / np.abs(v[idx[1:]])
    lower = np.arange(len(gaps)) <= (len(idx) - 2) // 2
    cuts = []
    for done, sel in ((done_lo, lower), (done_hi, ~lower)):
        g = np.flatnonzero(sel & (gaps >= 100 * r))
        j = g[np.argmax(gaps[g])] if len(g) and not done else None
        cuts.append(None if j is None else float((v[idx[j]] + v[idx[j + 1]]) / 2))
    return cuts[0], cuts[1], bool(done_lo), bool(done_hi)


@jax.jit
def _km(op, m, X):
    return _K(op, m, X), _M(op, m, X)


def _apply_km(op, m, X, c):
    """Host (K_m X, M_m X) of the host columns X, in device chunks of c columns."""
    KX, MX = np.empty_like(X), np.empty_like(X)
    for j in range(0, X.shape[1], c):
        w = min(c, X.shape[1] - j)
        k, mx = _km(op, m, _pad(X[:, j : j + w], X.shape[0], c))
        KX[:, j : j + w], MX[:, j : j + w] = np.asarray(k)[:, :w], np.asarray(mx)[:, :w]
    return KX, MX


def _merge(op, m, groups, consts, c):
    """One window held by several runs, groups = [V (n, k)] in run order (each M-orthonormal;
    the list and its arrays are consumed): each later run's columns are projected against the
    kept ones, P = U - Q Q^T M U, and the directions with ||p||_M >= 0.5 (eigenvalues >= 0.25 of
    P^T M P) kept, the rest are duplicates; then Rayleigh-Ritz of (K, M) on the kept basis. K
    and M are applied in device chunks of c columns. Returns (theta, X, r, t, dropped): the Ritz
    pairs with their explicit lambda-scaled residuals and round-off thresholds (also in device
    chunks)."""
    n, w = groups[0].shape[0], sum(U.shape[1] for U in groups)
    Q, KQ, MQ = (np.empty((n, w), order="F") for _ in range(3))
    k, drop = 0, 0
    while groups:  # consumed (and overwritten), so each run's copy is freed once appended
        U = groups.pop(0)
        KU, MU = _apply_km(op, m, U, c)
        S = None
        if k:  # project in place, one temporary at a time
            C = Q[:, :k].T @ MU
            for X, B in ((U, Q), (KU, KQ), (MU, MQ)):
                X -= (C.T @ B[:, :k].T).T  # F-order result: a contiguous in-place update
            ev, Y = np.linalg.eigh((U.T @ MU + MU.T @ U) / 2)
            keep = ev >= 0.25
            drop += int((~keep).sum())
            S = Y[:, keep] / np.sqrt(ev[keep])
        j = U.shape[1] if S is None else S.shape[1]
        src = [U, KU, MU]
        del U, KU, MU
        for B in (Q, KQ, MQ):  # each source freed right after its write
            X = src.pop(0)
            B[:, k : k + j] = X if S is None else (S.T @ X.T).T
            del X
        k += j
    Q, KQ, MQ = Q[:, :k], KQ[:, :k], MQ[:, :k]
    GK, GM = Q.T @ KQ, Q.T @ MQ
    theta, Z = sla.eigh((GK + GK.T) / 2, (GM + GM.T) / 2)
    X = (Z.T @ Q.T).T
    del Q
    KX = (Z.T @ KQ.T).T
    del KQ
    MX = (Z.T @ MQ.T).T
    del MQ
    r, t = np.empty(len(theta)), np.empty(len(theta))
    for j in range(0, len(theta), c):  # explicit residuals, one device chunk at a time
        sl = slice(j, j + c)
        r[sl], t[sl] = _resid(X[:, sl], KX[:, sl], MX[:, sl], theta[sl], consts)
    return theta, X, r, t, drop


def _span_count(op, free, P, lam_s, cols, x0, x1, nbs, R, verify, opts):
    """info["count"] of a span leaf [x0, x1]: its pairs are the stored ones inside, the nearest
    stored pair (or the parent's Ritz neighbour nbs) beyond each end its out-of-band pairs; P the
    parent's preconditioner. A verifiable count >= 0.5 is flagged (the span is rerun)."""
    inb = np.flatnonzero((lam_s >= x0) & (lam_s <= x1))
    lam_c = lam_s[inb]
    a = lam_c.min() if len(lam_c) else x0
    e = lam_c.max() if len(lam_c) else x1
    near = [np.flatnonzero(s)[pick(lam_s[s])]
            for s, pick in ((lam_s < a, np.argmax), (lam_s > e, np.argmin)) if s.any()]  # fmt: skip
    nb_lo, nb_hi = _beyond(a, e, nbs, cols(near), lam_s[near])
    if nb_hi is None or (x0 > 0 and nb_lo is None):
        raise RuntimeError("si_band: no converged neighbour to place a span's counting edge")
    est, ci = _count_slice(op, free, (cols(inb),), lam_c, x0, x1, nb_lo, nb_hi, R=R, precond=P,
                           _predict=not verify, **opts)  # fmt: skip
    if not verify:
        return _count_out(ci, False, "not requested", est, [], False, 0)
    miss = bool(ci["verifiable"] and est >= 0.5)
    reason = (None if ci["passed"] else "count >= 0.5: the span is rerun" if miss
              else _unverified(ci))  # fmt: skip
    return _count_out(ci, True, reason, est, [est], miss, 0)


def _band_count(leaves):
    """info["count"] of a band: si_checked's shape over the leaves' counts."""
    cs = [x for _, x in leaves]
    ran = [x for x in cs if x["ran"]]
    passed = bool(cs) and all(x["passed"] for x in cs)
    est = [x["estimate"] for x in cs]
    fail = next(((iv, x) for iv, x in leaves if not x["passed"]), None)
    return dict(
        ran=bool(ran), passed=passed, verifiable=bool(ran) and all(x["verifiable"] for x in ran),
        flagged=any(x["flagged"] for x in cs),
        estimate=float(np.nanmax(est)) if np.isfinite(est).any() else np.nan, estimates=est,
        reason=None if passed or fail is None else
        f"[{fail[0][0]:.6g}, {fail[0][1]:.6g}]: {fail[1]['reason']}",
    )  # fmt: skip


def si_band(
    op, free, lo, hi, make, *, capacity, R=None, verify=True, edges=None, count_options=None,
    tol=1e-6, block=6, seed=0, verbose=False, max_slices=64, **slice_options,
):  # fmt: skip
    """Eigenpairs of the masked pencil (K_m, M_m) in the band [lo, hi] (0 <= lo < hi) by si_checked
    slices, each within a device basis of capacity columns, deflation included. make(sigma): the SPD
    preconditioner at sigma (one alive at a time; the slices' and the counts'). Slices are processed
    lowest first, each deflating the stored modes of its window and the nearest known pair beyond
    each end (which can close its sides); a slice whose r_acc widens its window over more stored
    modes is rerun with them deflated. A slice that fills its capacity keeps the span verified by
    si_slice's stop rule on its last Ritz values (a leaf without a run, counted with the parent's
    preconditioner, rerun if its count flags) and requeues its remainders: each side cut at the
    largest gap >= 100 r_acc in its half of the converged run, a side without a cut as one remainder
    (a binary split); nothing converged, no progress, or no side done or cut raises. Clusters
    held by two or more runs are merged: duplicates (projected M-norm < 0.5) dropped,
    Rayleigh-Ritz, the window widened once if a Ritz residual fails (edge_rr_failed if it still
    fails). edges: provisional interior edges, used as given (default one slice). verify: count
    every leaf (False: none, the predicted degree and gap recorded). Slice i uses seed +
    (RESTARTS + 1) i, the count's seed likewise. slice_options: si_checked's and si_slice's (tau,
    maxit_inner, ...; not sigma or count_precond).

    Returns lam (ascending), V (host fp64; M-orthonormal within a window, to residual level across
    runs that deflate neither the other: |x_i^T M x_j| <= (||x_j|| ||r_i|| + ||x_i|| ||r_j||) /
    |lam_i - lam_j|, r_i = K x_i - lam_i M x_i) and info: converged, reason, r_acc, tau_eff,
    residuals, thresholds, slices (per processed slice: a run's si_checked info without
    neighbours and ritz plus interval, kind "run", leaf, n_new, reruns; a span {interval, kind
    "span", leaf, parent (the parent's index among the runs), count}),
    merged_edges, collapsed, edge_rr_failed, duplicates, count (si_checked's shape over the leaves)
    and time (total, slices, merge). See docs/eigensolvers.md#slicing-a-band."""
    check_x64("si_band")
    t0 = time.perf_counter()
    n = op.n_dof
    if not 0 <= lo < hi:
        raise ValueError(f"need 0 <= lo < hi, got lo = {lo}, hi = {hi}")
    cuts = [] if edges is None else [float(x) for x in edges]
    if cuts != sorted(set(cuts)) or any(not lo < x < hi for x in cuts):
        raise ValueError(f"edges must be sorted, distinct and inside (lo, hi), got {edges}")
    if not isinstance(block, int) or block < 1:
        raise ValueError(f"block must be an int >= 1, got {block!r}")
    if not isinstance(capacity, int) or capacity < 2 * block:
        raise ValueError(f"capacity must be an int >= 2 block = {2 * block}, got {capacity!r}")
    if not isinstance(verify, bool):
        raise ValueError(f"verify must be True or False, got {verify!r}")
    if bad := sorted({"sigma", "count_precond"} & set(slice_options)):
        raise ValueError(f"si_band sets {bad} per slice (sigma the midpoint, make(sigma))")
    b, c = block, _chunk(block)
    opts = {"verbose": verbose, **(count_options or {})}
    c0 = opts.pop("seed", 0)
    m = jnp.asarray(np.asarray(free), jnp.float64)[:, None]
    lam_s, run_s, res_s, thr_s, vecs = np.zeros(0), np.zeros(0, int), np.zeros(0), np.zeros(0), []
    work = [(x0, x1, {}) for x0, x1 in zip([lo, *cuts], [*cuts, hi], strict=True)]
    slices, leaves = [], []  # leaves: (interval, count info)
    r_s, tau_eff, converged, reason, i = 0.0, 0.0, True, None, 0

    def cols(idx):  # F order: a column touches only its own pages, no transposing copies
        U = np.empty((n, len(idx)), order="F")
        for j, q in enumerate(idx):
            U[:, j] = vecs[q]
        return U

    while work:
        if i >= max_slices:
            raise RuntimeError(f"si_band: more than max_slices = {max_slices} slices")
        work.sort(key=lambda x: x[0])
        a, e, prev = work.pop(0)
        order = np.argsort(lam_s, kind="stable")
        rw, win, reruns, sd = max(r_s, tol), None, 0, (RESTARTS + 1) * i
        P = None  # the previous slice's released first: one alive at a time
        P = make((a + e) / 2)
        while True:  # rerun while the run's r_acc widens its window over more stored modes
            wide = order[_window(lam_s[order], a, e, min(1000 * rw, 1e-2), 100 * rw)]
            if win is not None:
                if np.isin(wide, win).all():
                    break
                reruns, V, inf = reruns + 1, None, None  # the rerun's block discarded
                if verbose:
                    print(f"si_band: slice {i} rerun, r_acc {rw:.3g} widens its window", flush=True)
            win, cap = wide, _up(capacity, c) - _up(len(wide), c)
            if cap < 2 * b:
                raise ValueError(
                    f"slice [{a:.6g}, {e:.6g}]: {len(win)} deflated modes leave {cap}"
                    f" columns of capacity {capacity} (below 2 block): raise capacity"
                )
            lam, V, inf = si_checked(
                op, free, a, e, P, capacity=cap, W=cols(win) if len(win) else None,
                lam_W=lam_s[win] if len(win) else None, count_options={**opts, "seed": c0 + sd},
                verify=verify, seed=seed + sd, verbose=verbose, R=R, tol=tol, block=block,
                **slice_options,
            )  # fmt: skip
            rw = max(rw, inf["r_acc"])
        lam_s, run_s = np.r_[lam_s, lam], np.r_[run_s, np.full(len(lam), i)]
        res_s, thr_s = np.r_[res_s, inf["residuals"]], np.r_[thr_s, inf["thresholds"]]
        vecs.extend(V[:, q] for q in range(V.shape[1]))
        r_s, tau_eff = max(r_s, inf["r_acc"]), max(tau_eff, inf["tau_eff"])
        n_new = int(((lam >= a) & (lam <= e)).sum())
        strip = ("neighbours", "ritz", "runs")
        entry = {k: x for k, x in inf.items() if k not in strip}
        entry.update(runs=[{k: x for k, x in r.items() if k not in strip} for r in inf["runs"]],
                     interval=(a, e), kind="run", leaf=True, n_new=n_new,
                     reruns=reruns)  # fmt: skip
        slices.append(entry)
        why = inf["reason"]
        if verbose:
            print(f"si_band: slice {i} [{a:.6g}, {e:.6g}], sigma {inf['sigma']:.6g},"
                  f" capacity {cap}, W {len(win)}:"
                  f" {why or 'converged'}, {n_new} new, count"
                  f" {inf['count']['reason'] or 'passed'}", flush=True)  # fmt: skip
        i += 1
        bad = not (inf["residuals"] <= inf["thresholds"]).all()  # si_slice's final check
        if why != "capacity full" or bad:
            leaves.append(((a, e), inf["count"]))
            if not inf["converged"]:  # the band stops with the pairs found
                msg = why if why not in (None, "capacity full") else "slice not converged"
                converged, reason = False, f"[{a:.6g}, {e:.6g}]: {msg}"
                break
            continue
        # a full slice: its verified span (a leaf without a run) and its remainders
        entry["leaf"] = False
        sp = _split(*inf["ritz"], inf["sigma"], a, e, inf["r_acc"])
        if sp is None:
            raise _cap_error(a, e, capacity, "nothing converged")
        s_lo, s_hi, done_lo, done_hi = sp
        if not n_new and not ((done_lo and not prev.get(a)) or (done_hi and not prev.get(e))):
            raise _cap_error(a, e, capacity, "no progress")
        if not (done_lo or s_lo) and not (done_hi or s_hi):
            raise _cap_error(a, e, capacity, "neither side done or cut")
        here = {a: done_lo, e: done_hi}
        if not done_lo:
            work.append((a, s_lo or s_hi or e, here))
        if not done_hi:
            work.append((s_hi or s_lo or a, e, here))
        if (done_lo or s_lo) and (done_hi or s_hi):
            x0, x1 = a if done_lo else s_lo, e if done_hi else s_hi
            cnt = _span_count(op, free, P, lam_s, cols, x0, x1, inf["neighbours"], R, verify,
                              {**opts, "seed": c0 + sd})  # fmt: skip
            slices.append(dict(interval=(x0, x1), kind="span", leaf=not cnt["flagged"],
                               parent=i - 1, count=cnt))  # fmt: skip
            if cnt["flagged"]:  # a missed mode: the span is rerun, its pairs deflated
                work.append((x0, x1, {}))
            else:
                leaves.append(((x0, x1), cnt))
            if verbose:
                print(f"si_band: span [{x0:.6g}, {x1:.6g}], count {cnt['reason'] or 'passed'}",
                      flush=True)  # fmt: skip
    P = None
    t_slices = time.perf_counter() - t0

    # clusters held by several runs: duplicates, Rayleigh-Ritz, the re-check at the band's r_acc
    order = np.argsort(lam_s, kind="stable")
    ls = lam_s[order]
    windows = np.split(order, np.flatnonzero(_starts(ls, 100 * r_s))[1:]) if len(ls) else []
    V = None  # the last run's block (its columns are in the store)
    consts = None
    parts, ranges, dup, rr_failed, w = [], [], 0, False, 0  # parts: (lam, V or store idx, r, t)
    while w < len(windows):
        idx = windows[w]
        if len(set(run_s[idx])) < 2:
            parts.append((lam_s[idx], idx, res_s[idx], thr_s[idx]))
            w += 1
            continue
        if consts is None:
            consts = _consts(op, m, jax.random.key(seed), tol)
        g = [w]
        for attempt in range(2):
            idx = np.concatenate([windows[q] for q in g])
            runs = sorted(set(run_s[idx]))
            th, X, rr, tt, d = _merge(op, m, [cols(idx[run_s[idx] == q]) for q in runs],
                                      consts, c)  # fmt: skip
            tt = np.maximum(tt, thr_s[idx].max())
            if (rr <= tt).all() or attempt:
                break
            n_g = len(g)
            if parts and parts[-1][1].ndim == 1:  # the window below joins if kept as is
                g = [w - 1] + g
                parts.pop()
            g = g + ([g[-1] + 1] if g[-1] + 1 < len(windows) else [])
            if len(g) == n_g:  # nothing to widen by: the same result again
                break
            del X  # freed before the wider merge
        rr_failed |= not (rr <= tt).all()
        dup += d
        parts.append((th, X, rr, tt))
        for q in idx:
            vecs[q] = None
        ranges.append((lam_s[idx].min(), lam_s[idx].max()))
        w = g[-1] + 1
    # windows are ascending and separated by gaps >= 100 r_acc: the parts are in order; a run's
    # block is freed once all its columns are copied, so the store and V_f stay about 1x + a block
    lam_f, res_f, thr_f = (np.concatenate([p[q] for p in parts] + [np.zeros(0)]) for q in (0, 2, 3))
    V_f, k = np.empty((n, len(lam_f)), order="F"), 0  # a column touches only its own pages
    while parts:  # popped, so a merged window's X is freed once copied
        X = parts.pop(0)[1]
        if X.ndim == 1:
            for s in X:
                V_f[:, k], vecs[s] = vecs[s], None
                k += 1
        else:
            V_f[:, k : k + X.shape[1]] = X
            k += X.shape[1]
    bounds = sorted({x for (x0, x1), _ in leaves for x in (x0, x1)} - {lo, hi})
    near = lambda s: any(v0 * (1 - 100 * r_s) <= s <= v1 * (1 + 100 * r_s) for v0, v1 in ranges)  # noqa: E731
    merged = [s for s in bounds if near(s)]
    info = dict(
        converged=converged and not rr_failed,
        reason=reason or ("a merged window failed its re-check" if rr_failed else None),
        r_acc=r_s,
        tau_eff=tau_eff,
        residuals=res_f,
        thresholds=thr_f,
        slices=slices,
        merged_edges=merged,
        collapsed=bool(bounds) and len(merged) == len(bounds),
        edge_rr_failed=rr_failed,
        duplicates=dup,
        count=_band_count(leaves),
        time=dict(
            total=time.perf_counter() - t0,
            slices=t_slices,
            merge=time.perf_counter() - t0 - t_slices,
        ),  # fmt: skip
    )
    return lam_f, V_f, info
