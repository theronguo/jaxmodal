"""Jacobi and node-block Jacobi on A_P = K + sigma M, masked output. See docs/preconditioners.md."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.tree_util import Partial

from ..operators.common import check_x64


def _check_sigma(sigma):
    """Preconditioners are SPD: a concrete sigma < 0 or NaN raises (a traced one is not checked)."""
    if not isinstance(sigma, jax.core.Tracer) and not np.all(np.asarray(sigma) >= 0):
        raise ValueError(
            f"sigma must be >= 0 (preconditioners approximate K + sigma M), got {sigma}"
        )


def _jacobi(inv, r):
    return inv[:, None] * r


def jacobi(op, free, sigma=0.0):
    """Jacobi on A_P = K + sigma M as a pytree callable: v -> m * v / (diag K + sigma diag M),
    0 on constrained DOFs (masked output)."""
    check_x64("jacobi")
    _check_sigma(sigma)
    free = jnp.asarray(free)
    d = op.diag_K() + sigma * op.diag_M()
    return Partial(_jacobi, jnp.where(free, 1 / jnp.where(free, d, 1.0), 0.0))


def _inverse_blocks(B, free):
    """(inv, m): inverses (n, b, b) of the SPD blocks B with constrained rows and columns
    replaced by the identity, and the (n, b) float mask. A non-SPD block gives NaN. The
    package-internal entry for multigrid level blocks."""
    n, b = B.shape[:2]
    m = jnp.asarray(free).reshape(n, b).astype(B.dtype)
    B = B * m[:, :, None] * m[:, None, :] + (1 - m)[:, :, None] * jnp.eye(b, dtype=B.dtype)
    Linv = solve_triangular(
        jnp.linalg.cholesky(B), jnp.broadcast_to(jnp.eye(b, dtype=B.dtype), B.shape), lower=True
    )
    return jnp.swapaxes(Linv, 1, 2) @ Linv, m


def _block_jacobi(inv, m, r):
    n, b = m.shape
    y = jnp.einsum("nij,njk->nik", inv, r.reshape(n, b, -1)) * m[:, :, None]
    return y.reshape(r.shape)


def block_jacobi(op, free, sigma=0.0):
    """Node-block Jacobi on A_P = K + sigma M as a pytree callable: the inverse 3x3 diagonal
    node blocks K_nn + sigma diag(M)_n I_3 (constrained components masked), masked output."""
    check_x64("block_jacobi")
    _check_sigma(sigma)
    B = op.block_diag_K() + sigma * op.diag_M().reshape(-1, 3)[:, :, None] * jnp.eye(3)
    return Partial(_block_jacobi, *_inverse_blocks(B, free))
