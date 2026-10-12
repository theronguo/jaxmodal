"""Two-level p-multigrid for unstructured hex20/hex27 meshes: Chebyshev on the matrix-free A_P,
the hex8 Galerkin product P^T A_P P on the corner nodes, solved on the host by SciPy splu
(development and tests only; PR 13 swaps in AMG). See docs/preconditioners.md."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax import lax

from ..mesh.model import CORNERS, REFERENCE_NODES
from ..operators import matfree as mf
from ..operators.common import check_x64, register
from ..operators.constraints import ap_operator
from ..operators.dofmap import check_int32
from .chebyshev import chebyshev, lmax_estimate
from .gmg import GMG, SingularShiftError
from .jacobi import _check_sigma, block_jacobi


def _w8(n):
    """(n, 8) trilinear corner weights N8_c(xi_a) at the first n reference nodes."""
    xi = REFERENCE_NODES[:n]
    return np.prod(1 + xi[:, None, :] * CORNERS[None], axis=2) / 8


def _gather(idx, w, x):
    """y[i] = sum_j w[i, j] x[idx[i, j]] per node (3 components), unrolled over j."""
    m = x.shape[1]
    x3 = x.reshape(-1, 3, m)
    y = sum(w[:, j, None, None] * x3[idx[:, j]] for j in range(idx.shape[1]))
    return y.reshape(-1, m)


@dataclasses.dataclass(frozen=True)
class NodalTransfer:
    """Prolongation from the corner nodes by per-node (parent, weight) tables; restriction is
    its transpose as a gather over per-coarse-node (child, weight) tables (weight 0 pads)."""

    parent: jax.Array  # (n_nodes, w) int32
    weight: jax.Array  # (n_nodes, w)
    child: jax.Array  # (n_coarse, w') int32
    cweight: jax.Array  # (n_coarse, w')

    def __call__(self, x):
        return _gather(self.parent, self.weight, x)

    def restrict(self, r):
        return _gather(self.child, self.cweight, r)


register(NodalTransfer, [])


@dataclasses.dataclass(frozen=True, eq=False)
class HostLU:
    """Coarse solve b -> A_c^{-1} b by a host SciPy SuperLU factor through pure_callback. A
    pytree with no leaves whose treedef is the instance itself (hashed by identity): each build
    recompiles its jitted callers, and the jit caches keep it alive until jax.clear_caches()."""

    lu: object

    def __call__(self, b):
        def solve(r):
            return np.asarray(self.lu.solve(np.asarray(r, np.float64)), r.dtype)

        return jax.pure_callback(
            solve, jax.ShapeDtypeStruct(b.shape, b.dtype), b, vmap_method="sequential"
        )


jax.tree_util.register_pytree_node(HostLU, lambda h: ((), h), lambda h, _: h)


def _nodes(b):
    return np.asarray(b.dofs)[: b.n_el, ::3] // 3


def _tables(op):
    """Coarse ids of the corner nodes (-1 elsewhere), the parent and child tables, and per
    block the (n_el, n) owner flags (each node's first element in block order owns it)."""
    blocks = [_nodes(b) for b in op.blocks]
    corner = np.unique(np.concatenate([n[:, :8].ravel() for n in blocks]))
    cid = np.full(op.n_nodes, -1)
    cid[corner] = np.arange(len(corner))
    locs = [_w8(n.shape[1]) for n in blocks]
    w = max(int((W > 0).sum(1).max()) for W in locs)
    parent, weight = np.zeros((op.n_nodes, w), np.int64), np.zeros((op.n_nodes, w))
    seen, owns = np.zeros(op.n_nodes, bool), []
    for nodes, W in zip(blocks, locs, strict=True):
        lp = np.argsort(W <= 0, axis=1, kind="stable")[:, :w]  # nonzero corners first
        lw = np.take_along_axis(W, lp, 1)
        flat = nodes.ravel()
        first = np.zeros(flat.size, bool)
        first[np.unique(flat, return_index=True)[1]] = True
        first &= ~seen[flat]
        seen[flat] = True
        owns.append(first.reshape(nodes.shape))
        e, a = np.divmod(np.flatnonzero(first), nodes.shape[1])
        parent[flat[first]] = cid[nodes[e[:, None], lp[a]]]
        weight[flat[first]] = lw[a]
    f, j = np.nonzero(weight > 0)
    c = parent[f, j]
    order = np.argsort(c, kind="stable")
    f, j, c = f[order], j[order], c[order]
    slot = np.arange(len(c)) - np.searchsorted(c, c)
    child = np.zeros((len(corner), slot.max() + 1), np.int64)
    cweight = np.zeros(child.shape)
    child[c, slot], cweight[c, slot] = f, weight[f, j]
    T = NodalTransfer(
        jnp.asarray(parent, jnp.int32),
        jnp.asarray(weight),
        jnp.asarray(child, jnp.int32),
        jnp.asarray(cweight),
    )
    return cid, T, owns


@jax.jit
def _cells(op, b, sigma, m, d, own):
    """(n_pad, 24, 24) coarse cells P_loc^T (diag(m_e) A_e diag(m_e) + D_e) P_loc of a block."""
    k = b.dofs.shape[1]
    n = k // 3
    Ploc = jnp.asarray(_w8(n))
    data = mf._block_data(op, b) | {"own": own}
    xs = jax.tree.map(lambda a: a.reshape(-1, b.chunk, *a.shape[1:]), data)

    def f(c):
        K, M = mf.element_matrices(op, b, c)
        me = m[c["dofs"]]
        A = (K + sigma * M) * me[:, :, None] * me[:, None, :]
        A = A + jnp.eye(k) * (d[c["dofs"]] * jnp.repeat(c["own"], 3, axis=1))[:, :, None]
        A = A.reshape(-1, n, 3, n, 3)
        return jnp.einsum("ac,eaibj,bd->ecidj", Ploc, A, Ploc).reshape(-1, 24, 24)

    return lax.map(f, xs).reshape(-1, 24, 24)


def _hex8_galerkin(A):
    """(Ac, T) for an APOperator A of an Unstructured operator: the SciPy CSR hex8 Galerkin
    matrix P^T A P on the corner nodes and the NodalTransfer P. The fine operator is never
    assembled. The pattern is structural (every cell pair, no pruned zeros) and the values are
    symmetrized on it exactly."""
    op = A.op
    # owned diagonal (added once per node, by its owner): nodal masses, constrained diagonal
    d = A.sigma * op.nodal * A.free + A.s * (1 - A.free)
    cid, T, owns = _tables(op)
    n_c = 3 * int(T.child.shape[0])
    check_int32(coarse_dofs=n_c)
    rows, cols, vals = [], [], []
    for b, own in zip(op.blocks, owns, strict=True):
        ownp = np.zeros((b.dofs.shape[0], own.shape[1]))
        ownp[: b.n_el] = own
        C = np.asarray(_cells(op, b, A.sigma, A.free, d, jnp.asarray(ownp)))[: b.n_el]
        cd = (3 * cid[_nodes(b)[:, :8]][:, :, None] + np.arange(3)).reshape(-1, 24).astype(np.int32)
        rows.append(np.repeat(cd, 24, axis=1).ravel())
        cols.append(np.tile(cd, (1, 24)).ravel())
        vals.append(C.ravel())
    Ac = sp.coo_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_c, n_c)
    ).tocsr()  # sums duplicates, keeps explicit zeros
    At = Ac.T.tocsr()
    At.sort_indices()
    assert (At.indptr == Ac.indptr).all() and (At.indices == Ac.indices).all()  # symmetric pattern
    Ac.data = (Ac.data + At.data) / 2
    return Ac, T


def pmg(op, free, sigma, *, degree=2, ratio=30.0):
    """Two-level p-multigrid V-cycle for A_P = m (K + sigma M) m + s (I - m) of an Unstructured
    operator, as a GMG pytree callable on (n_dof, k) blocks (masked output): Chebyshev degree
    `degree` over node-block Jacobi on the fine level, the exact hex8 Galerkin coarse solve by
    host splu (test-scale only). sigma > 0 is concrete and required."""
    check_x64("pmg")
    if not isinstance(op, mf.Unstructured):
        raise TypeError(
            f"pmg needs an Unstructured operator, got {type(op).__name__} (voxel stencils: gmg)"
        )
    _check_sigma(sigma)
    sigma = float(sigma)
    if sigma == 0:
        raise SingularShiftError(
            "pmg needs sigma > 0: A_P at sigma = 0 is singular when the BCs leave rigid modes"
        )
    A = ap_operator(op, free, sigma)
    Ac, T = _hex8_galerkin(A)
    D = block_jacobi(op, free, sigma)
    S = chebyshev(A, D, lmax_estimate(A, D), degree, ratio)
    return GMG(((A, S),), (T,), HostLU(spla.splu(Ac.tocsc())), A.free, None)
