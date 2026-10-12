"""Chebyshev smoother over a Jacobi-type preconditioner, with the Lanczos lambda_max estimate.
See docs/preconditioners.md."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax

from ..operators.common import check_x64, register

SAFETY = 1.1  # lambda_max safety factor, applied inside chebyshev


def _dot(a, b):
    return jnp.sum(a * b, 0)


@functools.partial(jax.jit, static_argnums=2)
def _lanczos(A, precond, steps, v):
    z = precond(v)
    b = jnp.sqrt(_dot(v, z))[0]
    v, z = v / b, z / b

    def step(j, c):
        v, z, v_old, b_old, T = c
        w = A(z) - b_old * v_old
        a = _dot(w, z)[0]
        w = w - a * v
        zt = precond(w)
        b = jnp.sqrt(jnp.maximum(_dot(w, zt)[0], 0.0))
        inv = jnp.where(b > 0, 1 / jnp.where(b > 0, b, 1.0), 0.0)  # an invariant start stops at 0
        T = T.at[j, j].set(a).at[j, j + 1].set(b).at[j + 1, j].set(b)
        return w * inv, zt * inv, v, b, T

    T = jnp.zeros((steps + 1, steps + 1))
    T = lax.fori_loop(0, steps, step, (v, z, jnp.zeros_like(v), 0.0, T))[-1]
    return jnp.linalg.eigvalsh(T[:steps, :steps])[-1]


def lmax_estimate(A, precond, steps=20, seed=0):
    """Lower estimate of lambda_max(D^{-1} A), D^{-1} = precond (masked output), from `steps`
    Lanczos steps in the D^{-1}-weighted form from one seeded random start. A needs n_dof."""
    check_x64("lmax_estimate")
    v = jax.random.normal(jax.random.key(seed), (A.n_dof, 1))
    return _lanczos(A, precond, steps, v)


@dataclasses.dataclass(frozen=True)
class Chebyshev:
    """S: r -> p(D^{-1} A) D^{-1} r, the degree-step Chebyshev iteration from x = 0 on
    [1.1 lmax / ratio, 1.1 lmax]. Fixed, linear and SPD."""

    A: object
    precond: object
    lmax: jax.Array
    ratio: jax.Array
    degree: int

    def __call__(self, r):
        return _chebyshev(self, r)


register(Chebyshev, ["degree"])


@jax.jit
def _chebyshev(S, r):
    hi = SAFETY * S.lmax
    lo = hi / S.ratio
    theta, delta = (hi + lo) / 2, (hi - lo) / 2
    sigma = theta / delta
    rho = 1 / sigma
    d = S.precond(r) / theta
    x = d
    for _ in range(S.degree - 1):
        r = r - S.A(d)
        rho_new = 1 / (2 * sigma - rho)
        d = rho_new * rho * d + 2 * rho_new / delta * S.precond(r)
        x, rho = x + d, rho_new
    return x


def chebyshev(A, precond, lmax, degree=2, ratio=30.0):
    """Chebyshev preconditioner/smoother S over precond (D^{-1}) for A. lmax is the raw
    estimate (lmax_estimate); the 1.1 safety factor is applied here. degree >= 1 is static:
    degree precond applies and degree - 1 A applies."""
    check_x64("chebyshev")
    if not (isinstance(degree, int) and degree >= 1):
        raise ValueError(f"degree must be an int >= 1, got {degree!r}")
    f = functools.partial(jnp.asarray, dtype=jnp.float64)
    return Chebyshev(A, precond, f(lmax), f(ratio), degree)
