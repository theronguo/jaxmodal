"""Smoothed-aggregation AMG below the hex8 level of unstructured meshes: a host setup loop over
pyamg's primitives (node aggregation, zero-column drop) exported to the device as one flat
V-cycle with PR 12's p-level on top (none for all-hex8 meshes). Meshes with ssh8 blocks take
the solid-shell path (precond/shell.py, precond/schwarz.py). See docs/preconditioners.md."""

import dataclasses
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.tree_util import Partial
from pyamg.aggregation import fit_candidates, standard_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother
from pyamg.strength import symmetric_strength_of_connection

from ..linalg.ell import BlockELL, block_ell
from ..operators.common import is_count, register
from ..operators.constraints import ap_operator
from . import schwarz
from .chebyshev import chebyshev, lmax_estimate
from .gmg import GMG, Cholesky, _to32
from .jacobi import _block_jacobi, _inverse_blocks
from .pmg import Hex8Pattern, _check, _hex8_galerkin, _p_level

OMEGA = 4 / 3  # Jacobi prolongation smoothing weight (over the spectral radius estimate)
STALL = 0.9  # stop coarsening when a level would keep more than this fraction of its DOFs
CAP = 4  # the coarsest matrix may exceed max_coarse by this factor before amg raises
SHELL_BLOCKS = "solid-shell AMG supports ssh8 and hex8 blocks"


def unsupported_shell(types):
    """True for ssh8 meshes with blocks other than ssh8 and hex8 (nonconforming)."""
    return "ssh8" in types and bool(set(types) - {"ssh8", "hex8"})


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


def _sa_levels(A, B, max_coarse, tents=None):
    """([(A_l, P_l)], A_coarsest, tents) by smoothed aggregation of the SciPy matrix A (BSR with
    3 x 3 blocks, or anything SciPy converts to it) with near-null space B: node-block strength
    and block Jacobi weighting on the first level, scalar below; zero tentative columns
    dropped. tents [(T_l, R_l)] are the tentative prolongators and coarse candidates per level;
    given, they are reused (no strength, aggregation or fit; B and max_coarse unused). Runs
    under a fixed local NumPy seed (pyamg's spectral radius estimate draws from np.random; the
    estimate is the only consumer, so reuse draws the same stream) and leaves the global state
    unchanged. Removes explicit zero blocks from A in place."""
    state = np.random.get_state()
    np.random.seed(0)
    reuse, out = tents is not None, []
    try:
        levels, b = [], 3
        while (len(levels) < len(tents)) if reuse else (A.shape[0] > max_coarse):
            A.eliminate_zeros()  # BSR: drops all-zero blocks
            Ab = sp.bsr_array(A, blocksize=(b, b)) if b > 1 else A
            if reuse:
                C, (T, R) = None, tents[len(levels)]  # C is read only when filtering
            else:
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
            out.append((T, R))
            A, B, b = ((Ac + Ac.T) / 2).tocsr(), R, 1
    finally:
        np.random.set_state(state)
    if A.shape[0] > CAP * max_coarse and not reuse:
        raise ValueError(
            f"AMG coarsening stalled at {A.shape[0]} DOF (max_coarse = {max_coarse}); "
            "the dense coarsest factor would be too large"
        )
    return levels, A, tuple(out)


def _diag_blocks(A, b):
    """(n / b, b, b) diagonal blocks of the SciPy matrix A."""
    Ab = sp.bsr_array(A, blocksize=(b, b))
    Ab.sort_indices()
    rows = np.repeat(np.arange(A.shape[0] // b), np.diff(Ab.indptr))
    sel = Ab.indices == rows
    D = np.zeros((A.shape[0] // b, b, b))
    D[rows[sel]] = Ab.data[sel]
    return D


def _level(A, b, degree, ratio, dtype="float64"):
    """(BlockELL A, Chebyshev over its b x b diagonal block Jacobi) for an assembled level, the
    values stored in dtype (the Jacobi blocks stay fp64 until _cast)."""
    E = block_ell(A, b, dtype)
    inv, m = _inverse_blocks(jnp.asarray(_diag_blocks(A, b)), jnp.ones(A.shape[0], bool))
    if not bool(jnp.isfinite(inv).all()):
        raise ValueError("AMG level smoother blocks are not finite")
    D = Partial(_block_jacobi, inv, m)
    return E, chebyshev(E, D, lmax_estimate(E, D), degree, ratio)


def _cast(out, dtype):
    """The levels with fp32 smoothers and Chebyshev scalars (gmg._to32; level 0 keeps its
    A_P) when dtype is "float32", checked finite after the cast (the fine level's smoother
    never passes _level's check)."""
    if dtype == "float64":
        return out
    out = _to32(out)
    leaves = jax.tree.leaves([(S.precond, S.lmax) for _, S in out])
    if not all(bool(jnp.isfinite(x).all()) for x in leaves if x.dtype.kind == "f"):
        raise ValueError("AMG level smoother blocks are not finite")
    return out


def _cholesky(A):
    L = jnp.linalg.cholesky(jnp.asarray(A.toarray()))
    if not bool(jnp.isfinite(L).all()):
        raise ValueError("coarsest factor is not finite")
    return Cholesky(L)


@dataclasses.dataclass(frozen=True, eq=False)
class AMGAggregates:
    """The topology part of an amg setup, for amg(..., aggregates=): the Hex8Pattern, the
    tentative prolongators and coarse candidates per SA level, the mask and a connectivity
    fingerprint (host data). On solid-shell meshes tents is empty and shell holds the fiber
    groups and, per level, the vertex -> aggregate map, the tentative prolongator, its
    zero-column (identity) flags and the shell-group flags."""

    pattern: Hex8Pattern
    tents: tuple
    free: np.ndarray
    key: tuple
    shell: tuple | None = None


def _fingerprint(op):
    """Per block: type, element count and a hash of the DOF table (connectivity and element
    order)."""
    return tuple((b.type, b.n_el, hashlib.sha1(np.asarray(b.dofs)).hexdigest()) for b in op.blocks)


def amg(
    op, free, sigma, points, *, degree=2, ratio=30.0, max_coarse=1000, nonsingular=False,
    aggregates=None, return_aggregates=False, dtype="float64",
):  # fmt: skip
    """Multilevel V-cycle for A_P = m (K + sigma M) m + s (I - m) of an Unstructured operator, as
    a GMG pytree callable on (n_dof, k) blocks (masked output): the p-level (Chebyshev over
    node-block Jacobi on the matrix-free A_P, hex8 transfer) unless the mesh is all hex8, then
    smoothed-aggregation levels on the hex8 Galerkin matrix down to max_coarse DOF, then a dense
    Cholesky. points (n_nodes, 3) are the mesh coordinates in the operator's node order (the
    near-null space is their masked rigid modes). Chebyshev degree `degree` and `ratio` on every
    level. sigma >= 0 is concrete; sigma = 0 needs nonsingular=True, the caller's statement
    that A_P(0) is nonsingular (no rigid modes left by the BCs; s_K > 0 is checked here).

    Meshes with ssh8 blocks (ssh8 alone or with hex8; hex20/hex27 raise) take the solid-shell
    path: fiber groups, 9 candidates per aggregate (the masked rigid modes plus 3 curvature
    fields per level), additive Schwarz smoothers under Chebyshev on every level.

    return_aggregates=True returns (P, AMGAggregates); aggregates= reuses one from the same mesh
    and mask (connectivity and free are checked, points are not): it skips the pattern,
    strength, aggregation and the tentative fit (on solid shells also fibers and frames) and
    recomputes everything that depends on the values. For new materials, s_K, s_M or sigma
    the result equals a fresh setup; for moved points it keeps the old near-null space (an
    approximation). max_coarse is then ignored.

    dtype "float32" runs the V-cycle in fp32: the assembled levels, transfers and smoothers
    are stored in fp32; the fine A_P (shared with the caller), the p-level transfer, the
    coarsest factor and the mask stay fp64. Aggregates are dtype-independent."""
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
    if not is_count(max_coarse):
        raise ValueError(f"max_coarse must be an int >= 1, got {max_coarse!r}")
    sigma = _check("amg", op, sigma, nonsingular)
    if nonsingular and not bool((op.s_K > 0).all()):
        raise ValueError("nonsingular=True needs s_K > 0 on every element")
    points = np.asarray(points, np.float64)
    if points.shape != (op.n_nodes, 3):
        raise ValueError(f"points must have shape ({op.n_nodes}, 3), got {points.shape}")
    key = _fingerprint(op) if return_aggregates or aggregates is not None else None
    mask = np.asarray(free, bool).ravel()
    if aggregates is not None:
        if key != aggregates.key:
            raise ValueError("aggregates come from a mesh with another connectivity or chunking")
        if not np.array_equal(mask, aggregates.free):
            raise ValueError("aggregates come from another free mask")
    pattern = aggregates.pattern if aggregates is not None else None
    types = {b.type for b in op.blocks}
    if "ssh8" in types:
        if unsupported_shell(types):
            raise ValueError(SHELL_BLOCKS)
        shell = aggregates.shell if aggregates is not None else None
        P, pattern, shell = _shell_amg(
            op, free, sigma, points, degree, ratio, max_coarse, pattern, shell, dtype
        )
        if return_aggregates:
            return P, AMGAggregates(pattern, (), mask.copy(), key, shell)
        return P
    fine, Ab, pattern = _p_level(op, free, sigma, degree, ratio, pattern)
    A = fine[0]
    if aggregates is None:
        fc = np.asarray(A.free, bool).reshape(-1, 3)[pattern.corner].ravel()
        B, tents = _nullspace(points[pattern.corner], fc), None
    else:
        B, tents = None, aggregates.tents
    levels, coarsest, tents = _sa_levels(Ab, B, max_coarse, tents)
    hex8 = all(b.dofs.shape[1] == 24 for b in op.blocks)
    out, transfers = ([], []) if hex8 else ([fine], [pattern.T])
    for i, (Al, P) in enumerate(levels):
        # all-hex8: the matrix-free A_P stands in for the assembled first SA level
        b = 3 if i == 0 else 1
        out.append(fine if hex8 and i == 0 else _level(Al, b, degree, ratio, dtype))
        transfers.append(ELLTransfer(block_ell(P, 1, dtype), block_ell(P.T.tocsr(), 1, dtype)))
    out = _cast(out, dtype)
    P = GMG(tuple(out), tuple(transfers), _cholesky(coarsest), A.free, None, dtype)
    if return_aggregates:
        return P, AMGAggregates(pattern, tents, mask.copy(), key)
    return P


def _shell_amg(op, free, sigma, points, degree, ratio, max_coarse, pattern, shell, dtype):
    """(GMG, pattern, shell): the solid-shell V-cycle, Chebyshev over additive Schwarz on
    every level (the matrix-free A_P on level 0, A_b = A_P there), down to a dense Cholesky."""
    from .shell import B, _shell_levels  # shell imports amg's constants

    A = ap_operator(op, free, sigma)
    Ab, pattern = _hex8_galerkin(A, pattern)
    levels, coarsest, shell = _shell_levels(op, Ab, A.free, points, max_coarse, shell)[:3]
    out, transfers = [], []
    for i, (Al, (blocks, sel), P) in enumerate(levels):
        S = schwarz.build(Al, blocks, sel, dtype=dtype)
        E = A if i == 0 else block_ell(Al, B, dtype)
        out.append((E, chebyshev(E, S, lmax_estimate(E, S), degree, ratio)))
        b = 3 if i == 0 else B
        transfers.append(ELLTransfer(block_ell(P, b, dtype), block_ell(P.T.tocsr(), b, dtype)))
    out = _cast(out, dtype)
    P = GMG(tuple(out), tuple(transfers), _cholesky(coarsest), A.free, None, dtype)
    return P, pattern, shell
