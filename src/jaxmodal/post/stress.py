"""Strain and stress recovery at the quadrature points (RECOVER), element averages and von
Mises. See docs/postprocessing.md."""

import jax
import jax.numpy as jnp
import numpy as np

from ..elements import RECOVER
from ..operators.assembled import _blocks, _chunked, _tables
from ..operators.common import check_x64
from ..operators.dofmap import element_dofs


def _full(rec):
    def one(X, p, u):
        table, dofs = p
        return rec(X, table, u[dofs])

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def _mean(rec):
    def one(X, p, U):
        table, dofs = p

        def col(u):  # geometry and the ssh8 Kaa factorization do not depend on u: not batched
            _, w, eps, sigma = rec(X, table, u)
            return w @ eps / w.sum(), w @ sigma / w.sum()

        return jax.vmap(col, in_axes=1)(U[dofs])

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


_FULL = {t: _full(rec) for t, rec in RECOVER.items()}
_MEAN = {t: _mean(rec) for t, rec in RECOVER.items()}


def _setup(name, mesh, materials, orientations, chunk):
    """Checks, then (type, conn, element slice, (kernel table, element DOFs)) per block."""
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    check_x64(name)
    index, C, _, plies = _tables(mesh, materials, orientations, "consistent")
    dofs = element_dofs(mesh)
    return [(t, conn, e, (p, dofs[t])) for t, conn, e, p in _blocks(mesh, index, C, plies)]


def _field(U, n_dof):
    U = np.asarray(U, np.float64)
    if U.shape[:1] != (n_dof,) or U.ndim > 2:
        raise ValueError(f"expected an ({n_dof},) or ({n_dof}, k) array, got shape {U.shape}")
    return U


def recover(mesh, materials, u, orientations=None, chunk=1024):
    """Quadrature-point values of one field u (n_dof,), per block t: {"points" (n_e, nq, 3),
    "weights" w det J (n_e, nq), "eps", "sigma" (n_e, nq, 6)}, global Voigt, engineering shear.
    ssh8 padded plies have weight 0."""
    blocks = _setup("recover", mesh, materials, orientations, chunk)
    u = _field(u, 3 * len(mesh.points))
    if u.ndim != 1:
        raise ValueError(f"u must be one field (n_dof,), got shape {u.shape}")
    u = jnp.asarray(u)
    out = {}
    for t, conn, _, p in blocks:
        x, w, eps, sigma = _chunked(_FULL[t], mesh.points, conn, p, chunk, u)
        out[t] = {"points": x, "weights": w, "eps": eps, "sigma": sigma}
    return out


def element_stress(mesh, materials, U, orientations=None, chunk=1024, batch=64):
    """Volume averages sum_q w eps / sum_q w and the same for sigma, (n_el, k, 6) in element order
    (1-D U: (n_el, 6)), of the fields U (n_dof, k). Columns go to the kernel in batches of
    min(batch, k) (the last one zero-padded), elements in chunks."""
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    blocks = _setup("element_stress", mesh, materials, orientations, chunk)
    U = _field(U, 3 * len(mesh.points))
    one = U.ndim == 1
    U = U[:, None] if one else U
    k = U.shape[1]
    kb = max(min(batch, k), 1)
    eps, sigma = np.empty((mesh.n_el, k, 6)), np.empty((mesh.n_el, k, 6))
    for j in range(0, k, kb):
        m = min(kb, k - j)
        b = jnp.asarray(np.pad(U[:, j : j + m], ((0, 0), (0, kb - m))))
        for t, conn, e, p in blocks:
            ee, ss = _chunked(_MEAN[t], mesh.points, conn, p, chunk, b)
            eps[e, j : j + m], sigma[e, j : j + m] = ee[:, :m], ss[:, :m]
    return (eps[:, 0], sigma[:, 0]) if one else (eps, sigma)


def von_mises(sigma):
    """von Mises stress (...,) of Voigt (..., 6) stresses (11, 22, 33, 23, 13, 12)."""
    s = np.asarray(sigma, float)
    a, b, c = s[..., 0], s[..., 1], s[..., 2]
    shear = s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2
    return np.sqrt(0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) + 3 * shear)
