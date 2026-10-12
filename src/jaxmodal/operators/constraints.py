"""Dirichlet BC masks, the masked eigen-operator and A_P (SciPy and matrix-free); rigid body
modes and the deflation projector. See docs/boundary_conditions.md."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from jax.scipy.linalg import solve_triangular

from ..mesh import parts
from .common import check_x64, register

MAX_PARTS = 10  # default cap on parts: R is dense, 8 n_dof 6 n_parts bytes
RANK_TOL = 1e-8  # sigma_i <= RANK_TOL sigma_max counts as zero (unit-normalized columns)


def dof_mask(mesh, bcs):
    """(n_dof,) bool, True = free. bcs maps a node-set name to constrained components, e.g.
    {"xmin": "xyz", "ymin": "y"}: zero displacement in global axes; {} is free-free."""
    free = np.ones((len(mesh.points), 3), bool)
    for name, comps in bcs.items():
        if name not in mesh.node_sets:
            raise ValueError(f"unknown node set {name!r}")
        ok = isinstance(comps, str) and comps and set(comps) <= set("xyz")
        if not ok or len(set(comps)) != len(comps):
            raise ValueError(f"bcs[{name!r}] must be distinct letters from 'xyz', got {comps!r}")
        free[np.ix_(mesh.node_sets[name], ["xyz".index(c) for c in comps])] = False
    return free.ravel()


def _check_mask(n, mask):
    if not (isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (n,)):
        raise ValueError(f"mask must be a bool ({n},) np.ndarray")


def masked(A, mask):
    """D A D with D = diag(mask): constrained rows and columns zero, free block A_ff."""
    _check_mask(A.shape[0], mask)
    D = sp.diags_array(mask.astype(float))
    return (D @ A @ D).tocsr()


def restrict(K, M, free=None):
    """(K_ff, M_ff, idx): free rows and columns in CSC, idx = free DOF indices (all for None).
    Restriction, not masking: no spurious zero eigenvalues or pivots on constrained DOFs."""
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape != M.shape:
        raise ValueError(f"K and M must be square and of equal shape, got {K.shape}, {M.shape}")
    if free is None:
        idx = np.arange(K.shape[0])
    else:
        _check_mask(K.shape[0], free)
        idx = np.flatnonzero(free)
    return sp.csc_array(K)[idx][:, idx], sp.csc_array(M)[idx][:, idx], idx


def a_p(A, mask, s=None):
    """A_P = D A D + s (I - D); default s = mean of diag(A) over free DOFs (for SPD A)."""
    _check_mask(A.shape[0], mask)
    if not mask.any():
        raise ValueError("mask has no free DOF")
    if s is None:
        s = A.diagonal()[mask].mean()
        if s <= 0:
            raise ValueError(f"mean free diagonal s = {s:.3g} <= 0; pass s for indefinite A")
    return (masked(A, mask) + sp.diags_array(s * ~mask)).tocsr()


@dataclasses.dataclass(frozen=True)
class APOperator:
    """Matrix-free A_P = m (K + sigma M) m + s (I - diag m) as a pytree callable on (n, k)."""

    op: object
    free: jax.Array  # (n_dof,) float mask
    sigma: jax.Array
    s: jax.Array

    @property
    def n_dof(self):
        return self.op.n_dof

    def __call__(self, x):
        return _ap_apply(self, x)


register(APOperator, [])


@jax.jit
def _ap_apply(A, x):
    m, sigma, s = (a.astype(x.dtype) for a in (A.free[:, None], A.sigma, A.s))
    mx = m * x
    return m * (A.op.apply_K(mx) + sigma * A.op.apply_M(mx)) + s * (1 - m) * x


def ap_operator(op, free, sigma=0.0, s=None):
    """Matrix-free A_P of K + sigma M (any operator with apply_K, apply_M, diag_K, diag_M);
    the counterpart of a_p(K + sigma * M, free, s). Default s: the mean over free DOFs of
    diag K + |sigma| diag M (PR 4's rule for sigma >= 0, the shift-invert rule for sigma < 0)."""
    check_x64("ap_operator")
    _check_mask(op.n_dof, np.asarray(free))
    if not np.asarray(free).any():
        raise ValueError("mask has no free DOF")
    f = jnp.asarray(free, jnp.float64)
    sigma = jnp.asarray(sigma, jnp.float64)
    if s is None:
        s = jnp.sum(f * (op.diag_K() + jnp.abs(sigma) * op.diag_M())) / jnp.sum(f)
    return APOperator(op, f, sigma, jnp.asarray(s, jnp.float64))


def rigid_modes(mesh, free=None, max_parts=MAX_PARTS):
    """(n_dof, k) float64 rigid body modes, unit-norm columns, not M-orthonormal.

    free=None: T_x, T_y, T_z, R_x, R_y, R_z (about the node centroid) of each face-connected
    part, k = 6 n_parts. free (bool (n_dof,), True = free): the remaining modes R null((I - m) R),
    per part, with constrained rows exactly 0.
    """
    x = mesh.points
    n = len(x)
    if free is not None:
        _check_mask(3 * n, free)
    part = parts(mesh)
    n_parts = int(part.max()) + 1
    if n_parts > max_parts:
        raise ValueError(
            f"mesh has {n_parts} parts > max_parts = {max_parts}: remove floating islands, or "
            "raise max_parts (R is dense, 48 n_dof n_parts bytes)"
        )
    count = np.bincount(part, minlength=n_parts)
    c = np.stack([np.bincount(part, x[:, i], n_parts) for i in range(3)], 1) / count[:, None]
    d = x - c[part]
    R = np.zeros((n, 3, 6 * n_parts))
    nodes, col = np.arange(n), 6 * part
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        R[nodes, i, col + i] = 1.0
        R[nodes, k, col + 3 + i] = d[:, j]  # e_i x d
        R[nodes, j, col + 3 + i] = -d[:, k]
    R = R.reshape(3 * n, -1)
    R /= np.sqrt(np.einsum("ij,ij->j", R, R))  # column norms without an |R|^2 copy
    if free is None:
        return R
    fixed = np.flatnonzero(~free)
    fixed_part = part[fixed // 3]
    null = []
    for p in range(n_parts):
        A = R[fixed[fixed_part == p], 6 * p : 6 * p + 6]
        if len(A) > 6:
            A = np.linalg.qr(A, mode="r")  # same singular values, 6 x 6
        if len(A):
            _, sv, vt = np.linalg.svd(A)  # full_matrices: all 6 right singular vectors
            null.append(vt[int((sv > RANK_TOL * sv[0]).sum()) :].T)
        else:
            null.append(np.eye(6))
    RN = R @ scipy.linalg.block_diag(*null)
    del R
    RN *= free[:, None]
    return RN


def m_orthonormalize(V, MV):
    """V L^-T with L L^T = V^T M V, twice (CholQR2), so V'^T M V' = I. MV = M V from the caller.
    V is donated (consumed); V and MV must be distinct arrays. V must have full M-rank, else
    the result is NaN."""
    if V.shape[1] == 0:
        return jnp.asarray(V)
    return _cholqr2(V, MV)


def _inv_chol_t(V, MV):
    G = V.T @ MV
    L = jnp.linalg.cholesky((G + G.T) / 2)
    return solve_triangular(L, jnp.eye(len(G)), lower=True).T


@functools.partial(jax.jit, donate_argnums=(0,))
def _cholqr2(V, MV):
    T = _inv_chol_t(V, MV)
    V, MV = V @ T, MV @ T
    return V @ _inv_chol_t(V, MV)


@jax.jit
def project(R, X, MX):
    """X - R (R^T MX): removes the R-components of X, with R^T M R = I and MX = M X."""
    return X - R @ (R.T @ MX)
