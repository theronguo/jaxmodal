"""Smoothed-aggregation AMG below the hex8 level of unstructured meshes: a host setup loop over
pyamg's primitives (node aggregation, zero-column drop) exported to the device as one flat
V-cycle with PR 12's p-level on top (none for all-hex8 meshes). See docs/preconditioners.md."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.tree_util import Partial
from pyamg.aggregation import fit_candidates, standard_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother
from pyamg.strength import symmetric_strength_of_connection

from ..linalg.ell import BlockELL, block_ell
from ..operators.common import register
from .chebyshev import chebyshev, lmax_estimate
from .gmg import GMG, Cholesky
from .jacobi import _block_jacobi, _inverse_blocks
from .pmg import _check, _p_level

OMEGA = 4 / 3  # Jacobi prolongation smoothing weight (over the spectral radius estimate)
STALL = 0.9  # stop coarsening when a level would keep more than this fraction of its DOFs
CAP = 4  # the coarsest matrix may exceed max_coarse by this factor before amg raises


@dataclasses.dataclass(frozen=True)
class ELLTransfer:
    """Prolongation P and restriction P^T, both stored as rectangular scalar block-ELL."""

    P: BlockELL
    PT: BlockELL

    def __call__(self, x):
        return self.P(x)

    def restrict(self, r):
        return self.PT(r)


register(ELLTransfer, [])


def _nullspace(x, free):
    """(3 n, 6) rigid modes at the points x (n, 3): translations and rotations e_j x (x - mean),
    rows of constrained DOFs (free False) zeroed."""
    x = x - x.mean(0)
    B = np.zeros((len(x), 3, 6))
    B[:, :, :3] = np.eye(3)
    B[:, :, 3:] = np.cross(np.eye(3)[:, None], x).transpose(1, 2, 0)
    return B.reshape(-1, 6) * free[:, None]


def _sa_levels(A, B, max_coarse):
    """([(A_l, P_l)], A_coarsest) by smoothed aggregation of the SciPy matrix A with near-null
    space B: node-block (3 x 3) strength and block Jacobi weighting on the first level, scalar
    below; zero tentative columns dropped. Runs under a fixed local NumPy seed (pyamg's spectral
    radius estimate draws from np.random) and leaves the global state unchanged."""
    state = np.random.get_state()
    np.random.seed(0)
    try:
        levels, b = [], 3
        while A.shape[0] > max_coarse:
            A = sp.csr_array(A, copy=True)
            A.eliminate_zeros()
            Ab = sp.bsr_array(A, blocksize=(b, b)) if b > 1 else A
            C = symmetric_strength_of_connection(Ab, 0.0)
            T, R = fit_candidates(sp.csr_array(standard_aggregation(C)[0]), B)
            T = sp.csc_array(T)
            keep = np.flatnonzero(np.asarray(T.multiply(T).sum(axis=0)).ravel() > 0)
            if len(keep) > STALL * A.shape[0]:
                break
            T, R = sp.csr_array(T[:, keep]), R[keep]
            if b > 1:
                T = sp.bsr_array(T, blocksize=(b, 1))
            w = "block" if b > 1 else "diagonal"
            P = sp.csr_array(jacobi_prolongation_smoother(Ab, T, C, R, OMEGA, weighting=w))
            P.eliminate_zeros()
            Ac = sp.csr_array(P.T @ (A @ P))
            levels.append((A, P))
            A, B, b = ((Ac + Ac.T) / 2).tocsr(), R, 1
    finally:
        np.random.set_state(state)
    if A.shape[0] > CAP * max_coarse:
        raise ValueError(
            f"AMG coarsening stalled at {A.shape[0]} DOF (max_coarse = {max_coarse}); "
            "the dense coarsest factor would be too large"
        )
    return levels, A


def _diag_blocks(A, b):
    """(n / b, b, b) diagonal blocks of the SciPy matrix A."""
    Ab = sp.bsr_array(A, blocksize=(b, b))
    Ab.sort_indices()
    rows = np.repeat(np.arange(A.shape[0] // b), np.diff(Ab.indptr))
    sel = Ab.indices == rows
    D = np.zeros((A.shape[0] // b, b, b))
    D[rows[sel]] = Ab.data[sel]
    return D


def _level(A, b, degree, ratio):
    """(BlockELL A, Chebyshev over its b x b diagonal block Jacobi) for an assembled level."""
    E = block_ell(A, b)
    inv, m = _inverse_blocks(jnp.asarray(_diag_blocks(A, b)), jnp.ones(A.shape[0], bool))
    if not bool(jnp.isfinite(inv).all()):
        raise ValueError("AMG level smoother blocks are not finite")
    D = Partial(_block_jacobi, inv, m)
    return E, chebyshev(E, D, lmax_estimate(E, D), degree, ratio)


def _cholesky(A):
    L = jnp.linalg.cholesky(jnp.asarray(A.toarray()))
    if not bool(jnp.isfinite(L).all()):
        raise ValueError("coarsest factor is not finite")
    return Cholesky(L)


def amg(op, free, sigma, points, *, degree=2, ratio=30.0, max_coarse=1000):
    """Multilevel V-cycle for A_P = m (K + sigma M) m + s (I - m) of an Unstructured operator, as
    a GMG pytree callable on (n_dof, k) blocks (masked output): the p-level (Chebyshev over
    node-block Jacobi on the matrix-free A_P, hex8 transfer) unless the mesh is all hex8, then
    smoothed-aggregation levels on the hex8 Galerkin matrix down to max_coarse DOF, then a dense
    Cholesky. points (n_nodes, 3) are the mesh coordinates in the operator's node order (the
    near-null space is their masked rigid modes). Chebyshev degree `degree` and `ratio` on every
    level. sigma > 0 is concrete and required."""
    sigma = _check("amg", op, sigma)
    points = np.asarray(points, np.float64)
    if points.shape != (op.n_nodes, 3):
        raise ValueError(f"points must have shape ({op.n_nodes}, 3), got {points.shape}")
    fine, T, Ac, corner = _p_level(op, free, sigma, degree, ratio)
    A = fine[0]
    fc = np.asarray(A.free, bool).reshape(-1, 3)[corner].ravel()
    levels, coarsest = _sa_levels(Ac, _nullspace(points[corner], fc), max_coarse)
    hex8 = all(b.dofs.shape[1] == 24 for b in op.blocks)
    out, transfers = ([], []) if hex8 else ([fine], [T])
    for i, (Al, P) in enumerate(levels):
        out.append(fine if hex8 and i == 0 else _level(Al, 3 if i == 0 else 1, degree, ratio))
        transfers.append(ELLTransfer(block_ell(P, 1), block_ell(P.T.tocsr(), 1)))
    return GMG(tuple(out), tuple(transfers), _cholesky(coarsest), A.free, None)
