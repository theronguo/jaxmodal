"""Two-level p-multigrid for unstructured hex20/hex27 meshes: Chebyshev on the matrix-free A_P,
the hex8 Galerkin product P^T A_P P on the corner nodes, solved on the host by SciPy splu
(development and tests only; amg is the production variant). See docs/preconditioners.md."""

import dataclasses
import functools

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


@dataclasses.dataclass(frozen=True, eq=False)
class Hex8Pattern:
    """Host topology of the hex8 Galerkin matrix of an Unstructured operator: the NodalTransfer
    T, the corner node ids in coarse order, per block the padded (n_pad, n) owner flags and
    (n_pad, 64) positions of the 8 x 8 corner pairs in the node-block pattern (padded elements
    point past the end), the int32 BSR indptr / indices (sorted) and the transpose permutation
    of the pattern's blocks."""

    T: NodalTransfer
    corner: np.ndarray
    owns: tuple
    pos: tuple
    indptr: np.ndarray
    indices: np.ndarray
    tp: np.ndarray


def _pattern(op):
    """The Hex8Pattern of op (topology only)."""
    cid, T, owns = _tables(op)
    nc = int(T.child.shape[0])
    check_int32(coarse_dofs=3 * nc)
    cns = [cid[_nodes(b)[:, :8]] for b in op.blocks]
    cn = np.concatenate(cns)
    E = sp.csr_array(
        (np.ones(cn.size), cn.ravel(), np.arange(0, cn.size + 1, 8)), shape=(len(cn), nc)
    )
    G = (E.T @ E).tocsr()
    G.sort_indices()
    nnzb = len(G.indices)
    check_int32(nnzb=nnzb, nnz=9 * nnzb)  # pyamg's CSR intermediates
    rows = np.repeat(np.arange(nc, dtype=np.int64), np.diff(G.indptr))
    keys = rows * nc + G.indices
    pos, ownp = [], []
    for b, c, own in zip(op.blocks, cns, owns, strict=True):
        p = np.full((b.dofs.shape[0], 64), nnzb, np.int32)  # nnzb: dropped (-1 would wrap)
        p[: b.n_el] = np.searchsorted(keys, (c[:, :, None] * nc + c[:, None, :]).reshape(-1, 64))
        o = np.zeros((b.dofs.shape[0], own.shape[1]), bool)
        o[: b.n_el] = own
        pos.append(p)
        ownp.append(o)
    tp = np.searchsorted(keys, G.indices.astype(np.int64) * nc + rows).astype(np.int32)
    return Hex8Pattern(
        T, np.flatnonzero(cid >= 0), tuple(ownp), tuple(pos),
        G.indptr.astype(np.int32), G.indices.astype(np.int32), tp,
    )  # fmt: skip


@functools.partial(jax.jit, donate_argnums=0)
def _scatter_cells(data, op, b, sigma, m, d, own, pos):
    """data (nnzb, 3, 3) plus the block's coarse cells P_loc^T (diag(m_e) A_e diag(m_e) + D_e)
    P_loc scattered into the node-block pattern, one chunk per scan step."""
    k = b.dofs.shape[1]
    n = k // 3
    Ploc = jnp.asarray(_w8(n))
    xs = mf._block_data(op, b) | {"own": own, "pos": pos}
    xs = jax.tree.map(lambda a: a.reshape(-1, b.chunk, *a.shape[1:]), xs)

    def f(data, c):
        K, M = mf.element_matrices(op, b, c)
        me = m[c["dofs"]]
        A = (K + sigma * M) * me[:, :, None] * me[:, None, :]
        A = A + jnp.eye(k) * (d[c["dofs"]] * jnp.repeat(c["own"], 3, axis=1))[:, :, None]
        C = jnp.einsum("ac,eaibj,bd->ecdij", Ploc, A.reshape(-1, n, 3, n, 3), Ploc)
        return data.at[c["pos"].reshape(-1)].add(C.reshape(-1, 3, 3), mode="drop"), None

    return lax.scan(f, data, xs)[0]


def _hex8_galerkin(A, pattern=None):
    """(Ab, pattern) for an APOperator A of an Unstructured operator: the SciPy BSR (3 x 3
    blocks) hex8 Galerkin matrix P^T A P on the corner nodes and its Hex8Pattern (built unless
    given). The fine operator is never assembled. The pattern is structural (every corner pair
    of an element, no pruned zeros) and the values are symmetrized on it exactly."""
    op = A.op
    pattern = _pattern(op) if pattern is None else pattern
    # owned diagonal (added once per node, by its owner): nodal masses, constrained diagonal
    d = A.sigma * op.nodal * A.free + A.s * (1 - A.free)
    data = jnp.zeros((len(pattern.indices), 3, 3))
    for b, own, pos in zip(op.blocks, pattern.owns, pattern.pos, strict=True):
        data = _scatter_cells(data, op, b, A.sigma, A.free, d, own, pos)
    v = np.array(data)  # one host copy; symmetrize in place (d[tp] is a copy, IEEE + commutes)
    del data
    v += v[pattern.tp].swapaxes(1, 2)
    v *= 0.5
    n_c = 3 * len(pattern.corner)
    # own index arrays: callers may compact Ab in place (amg's eliminate_zeros)
    Ab = sp.bsr_array((v, pattern.indices.copy(), pattern.indptr.copy()), shape=(n_c, n_c))
    return Ab, pattern


def no_shell(name):
    """The refusal of solve_modes' AMG and amg_memory on solid-shell blocks (until PR 18b)."""
    return ValueError(f"{name} does not support ssh8 blocks; solid-shell AMG arrives in PR 18b")


def _check(name, op, sigma, nonsingular=False):
    """The shared argument checks of pmg and amg; returns sigma as a float. sigma = 0 needs
    nonsingular (the caller's proof that A_P(0) is nonsingular)."""
    check_x64(name)
    if not isinstance(op, mf.Unstructured):
        raise TypeError(
            f"{name} needs an Unstructured operator, got {type(op).__name__} (voxel stencils: gmg)"
        )
    _check_sigma(sigma)
    sigma = float(sigma)
    if sigma == 0 and not nonsingular:
        raise SingularShiftError(
            f"{name} needs sigma > 0: A_P at sigma = 0 is singular when the BCs leave rigid modes"
        )
    return sigma


def _p_level(op, free, sigma, degree, ratio, pattern=None):
    """((A, S), Ab, pattern): the matrix-free A_P with Chebyshev over node-block Jacobi, the BSR
    hex8 Galerkin matrix and its Hex8Pattern (pmg and amg)."""
    A = ap_operator(op, free, sigma)
    Ab, pattern = _hex8_galerkin(A, pattern)
    D = block_jacobi(op, free, sigma)
    return (A, chebyshev(A, D, lmax_estimate(A, D), degree, ratio)), Ab, pattern


def pmg(op, free, sigma, *, degree=2, ratio=30.0):
    """Two-level p-multigrid V-cycle for A_P = m (K + sigma M) m + s (I - m) of an Unstructured
    operator, as a GMG pytree callable on (n_dof, k) blocks (masked output): Chebyshev degree
    `degree` over node-block Jacobi on the fine level, the exact hex8 Galerkin coarse solve by
    host splu (test-scale only; amg is the production variant). sigma > 0 is concrete and
    required."""
    sigma = _check("pmg", op, sigma)
    if any(b.type == "ssh8" for b in op.blocks):
        raise ValueError("pmg does not support ssh8 blocks; use amg")
    (A, S), Ab, pattern = _p_level(op, free, sigma, degree, ratio)
    return GMG(((A, S),), (pattern.T,), HostLU(spla.splu(Ab.tocsc())), A.free, None)
