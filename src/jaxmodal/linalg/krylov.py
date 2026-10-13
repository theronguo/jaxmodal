"""PCG and MINRES on (n, k) blocks of independent right-hand sides, with explicit-residual exits.
See docs/preconditioners.md."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial

from ..operators.common import check_x64

STALL_R = 5  # PCG: residual replacements without convergence before the stagnation exit
CHECK = 10  # MINRES: iterations between explicit residual checks
STALL_C = 5  # MINRES: checks without a new minimum before the stagnation exit


def _norm(a):
    return jnp.sqrt(jnp.sum(a * a, 0))


def _div(a, b):
    return a / jnp.where(b != 0, b, 1.0)


def _identity(r):
    return r


def _check(name, b, maxit):
    check_x64(name)
    if b.ndim != 2:
        raise ValueError(f"b must have shape (n, k), got {b.shape}")
    if maxit < 1:
        raise ValueError(f"maxit must be >= 1, got {maxit}")


@jax.jit
def _pcg(A, P, b, tol, maxit):
    k = b.shape[1]
    nb = _norm(b)
    lim = tol * nb
    zeros_k = jnp.zeros(k, bool)
    x, r = jnp.zeros_like(b), b
    z = P(r)
    st = dict(x=x, r=r, p=z, rz=jnp.sum(r * z, 0), it=0, act=nb > 0, conv=nb == 0,
              stag=zeros_k, rep=jnp.zeros(k, int), its=jnp.zeros(k, int))  # fmt: skip

    def body(s):
        act = s["act"]
        q = A(s["p"])
        al = jnp.where(act, _div(s["rz"], jnp.sum(s["p"] * q, 0)), 0.0)
        x = jnp.where(act, s["x"] + al * s["p"], s["x"])
        r = jnp.where(act, s["r"] - al * q, s["r"])
        rn = _norm(r)
        hit = act & (rn <= lim)

        def explicit(xr):
            x, r = xr
            e = b - A(x)
            good = hit & (_norm(e) <= lim)
            bad = hit & ~good
            return good, bad, jnp.where(bad, e, r)  # restart from the explicit residual

        good, bad, r = lax.cond(hit.any(), explicit, lambda xr: (zeros_k, zeros_k, xr[1]), (x, r))
        rep = s["rep"] + bad
        stag = s["stag"] | (bad & (rep >= STALL_R))
        act = act & ~good & ~stag & jnp.isfinite(rn)
        z = P(r)
        rz = jnp.sum(r * z, 0)
        beta = jnp.where(act & ~bad, _div(rz, s["rz"]), 0.0)  # restart: p = z
        return dict(x=x, r=r, p=z + beta * s["p"], rz=rz, it=s["it"] + 1, act=act,
                    conv=s["conv"] | good, stag=stag, rep=rep, its=s["its"] + s["act"])  # fmt: skip

    s = lax.while_loop(lambda s: (s["it"] < maxit) & s["act"].any(), body, st)
    x = s["x"]
    info = dict(iterations=s["its"], residuals=_div(_norm(b - A(x)), nb), converged=s["conv"],
                stagnated=s["stag"], replacements=s["rep"])  # fmt: skip
    return x, info


def pcg(A, b, precond=None, tol=1e-8, maxit=1000):
    """Preconditioned CG for SPD A on independent columns of b (n, k): x and info (iterations,
    residuals (explicit, relative), converged, stagnated, replacements, all (k,)). A column
    converges when its explicit residual meets tol ||b||; a recursive residual that meets tol
    while the explicit one does not restarts the column, and STALL_R restarts stagnate it.
    A and precond are pytree callables (n, k) -> (n, k); b must vanish on constrained DOFs."""
    _check("pcg", b, maxit)
    return _pcg(A, Partial(_identity) if precond is None else precond, b, tol, maxit)


@jax.jit
def _minres(A, P, b, tol, maxit):
    k = b.shape[1]
    nb = _norm(b)
    lim = tol * nb
    zk = jnp.zeros(k)
    y = P(b)
    beta = jnp.sqrt(jnp.maximum(jnp.sum(b * y, 0), 0.0))
    zero = jnp.zeros_like(b)
    st = dict(x=zero, xb=zero, r1=b, r2=b, y=y, w=zero, w2=zero, oldb=zk, beta=beta, dbar=zk,
              eps=zk, phibar=beta, cs=-jnp.ones(k), sn=zk, it=0, act=nb > 0, conv=nb == 0,
              stag=jnp.zeros(k, bool), best=jnp.full(k, jnp.inf), since=jnp.zeros(k, int),
              its=jnp.zeros(k, int))  # fmt: skip

    def body(s):  # the recurrences of Paige-Saunders (scipy.sparse.linalg.minres)
        act = s["act"]
        v = _div(s["y"], s["beta"])
        y = A(v) - jnp.where(s["it"] > 0, _div(s["beta"], s["oldb"]), 0.0) * s["r1"]
        alfa = jnp.sum(v * y, 0)
        y = y - _div(alfa, s["beta"]) * s["r2"]
        r1, r2 = s["r2"], y
        y = P(r2)
        oldb, beta = s["beta"], jnp.sqrt(jnp.maximum(jnp.sum(r2 * y, 0), 0.0))
        delta = s["cs"] * s["dbar"] + s["sn"] * alfa
        gbar = s["sn"] * s["dbar"] - s["cs"] * alfa
        eps, dbar = s["sn"] * beta, -s["cs"] * beta
        gamma = jnp.sqrt(gbar**2 + beta**2)
        cs, sn = _div(gbar, gamma), _div(beta, gamma)
        phi, phibar = cs * s["phibar"], sn * s["phibar"]
        w = _div(v - s["eps"] * s["w2"] - delta * s["w"], gamma)
        x = jnp.where(act, s["x"] + phi * w, s["x"])
        it = s["it"] + 1

        def explicit(_):
            e = _norm(b - A(x))
            new = e < s["best"]
            xb = jnp.where(new, x, s["xb"])  # the 2-norm residual is not monotone in MINRES
            since = jnp.where(new, 0, s["since"] + 1)
            return act & (e <= lim), jnp.minimum(e, s["best"]), since, e, xb

        def skip(_):
            return jnp.zeros(k, bool), s["best"], s["since"], jnp.zeros(k), s["xb"]

        good, best, since, e, xb = lax.cond(it % CHECK == 0, explicit, skip, None)
        stag = s["stag"] | (act & ~good & (since >= STALL_C))
        act = act & ~good & ~stag & jnp.isfinite(e)
        return dict(x=x, xb=xb, r1=r1, r2=r2, y=y, w=w, w2=s["w"], oldb=oldb, beta=beta, dbar=dbar,
                    eps=eps, phibar=phibar, cs=cs, sn=sn, it=it, act=act, conv=s["conv"] | good,
                    stag=stag, best=best, since=since, its=s["its"] + s["act"])  # fmt: skip

    s = lax.while_loop(lambda s: (s["it"] < maxit) & s["act"].any(), body, st)
    e = _norm(b - A(s["x"]))
    last = e <= s["best"]  # the last iterate unless an earlier check was better
    x = jnp.where(last, s["x"], s["xb"])
    info = dict(iterations=s["its"], residuals=_div(jnp.minimum(e, s["best"]), nb),
                converged=s["conv"], stagnated=s["stag"])  # fmt: skip
    return x, info


def minres(A, b, precond=None, tol=1e-8, maxit=10000):
    """Preconditioned MINRES for symmetric (possibly indefinite) A with an SPD precond, on
    independent columns of b (n, k): x and info (iterations, residuals, converged, stagnated).
    The explicit 2-norm residual is checked every CHECK iterations; a column converges when it
    meets tol ||b|| and stagnates after STALL_C checks without a new minimum. Returns the iterate
    with the smallest explicit residual (the last one or the best check); a non-finite residual is
    reported as NaN, as in pcg."""
    _check("minres", b, maxit)
    return _minres(A, Partial(_identity) if precond is None else precond, b, tol, maxit)
