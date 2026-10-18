"""Host setup of the solid-shell AMG hierarchy (amg on meshes with ssh8 blocks): fiber groups,
per-level curvature candidates in sign-free aggregate frames, prolongators smoothed d times, the
one-ring group graph and the Schwarz patch sets. See docs/preconditioners.md."""

import numpy as np
import scipy.sparse as sp
from pyamg.aggregation import fit_candidates, standard_aggregation
from pyamg.util.linalg import approximate_spectral_radius
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator

from .amg import CAP, OMEGA, STALL, _nullspace

GUARD = 384  # level-0 patch DOFs above which amg raises (a wrong thickness axis)
B = 9  # candidates (coarse DOFs) per aggregate: 6 rigid modes and 3 curvatures
STEPS = (1, 2)  # prolongator smoothing steps on level 0 and below


def _shell_nodes(op):
    """(n_el, 8) node ids of the ssh8 elements (bottom 0-3, top 4-7)."""
    out = [np.asarray(b.dofs)[: b.n_el, ::3] // 3 for b in op.blocks if b.type == "ssh8"]
    return np.concatenate(out)


def _fibers(op):
    """Group id per node: connected components of the through-thickness node pairs (a, a + 4)
    of the ssh8 elements; nodes in no ssh8 element are singleton groups."""
    E = _shell_nodes(op)
    i, j = E[:, :4].ravel(), E[:, 4:].ravel()
    G = sp.coo_array((np.ones(len(i)), (i, j)), shape=(op.n_nodes, op.n_nodes))
    return connected_components(G, directed=False)[1]


def _columns(op, group):
    """Level-0 patches as (n_p, m) node ids (valid first, pad -1): per distinct set of groups
    of an ssh8 element the nodes of those groups, per node in no ssh8 element the node. A patch
    above GUARD DOFs raises ValueError."""
    E = _shell_nodes(op)
    g = np.sort(group[E], axis=1)
    g[:, 1:][g[:, 1:] == g[:, :-1]] = -1  # repeated groups (junctions)
    g = np.unique(-np.sort(-g, axis=1), axis=0)  # valid first; dedup over a column's elements
    order = np.argsort(group, kind="stable")
    cnt = np.bincount(group)
    start = np.cumsum(cnt) - cnt
    P, S = np.nonzero(g >= 0)
    c = cnt[g[P, S]]
    size = np.bincount(P, c, len(g)).astype(np.int64)
    if 3 * size.max() > GUARD:
        raise ValueError(
            f"ssh8 patch of {3 * size.max()} DOFs: check solid_shell's direction, or the stack "
            "is too deep"
        )
    first = np.repeat(np.cumsum(c) - c, c)
    node = order[np.repeat(start[g[P, S]], c) + np.arange(c.sum()) - first]
    pr = np.repeat(P, c)
    blocks = np.full((len(g), size.max()), -1, np.int32)
    blocks[pr, np.arange(len(pr)) - (np.cumsum(size) - size)[pr]] = node
    lone = np.setdiff1d(np.arange(op.n_nodes), E)
    pad = np.full((len(lone), blocks.shape[1]), -1, np.int32)
    pad[:, 0] = lone
    return np.concatenate([blocks, pad])


def _rings(S, zero, is_shell):
    """Coarse patches as (blocks, sel): per shell group its 9 DOFs and the non-identity DOFs of
    its shell neighbours in the group graph S, per other group its 9 DOFs. zero (9 n_g,) flags
    the identity DOFs, is_shell (n_g,) the groups with an ssh8 node."""
    S = sp.csr_array(S)
    r = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
    k = is_shell[r] & is_shell[S.indices]
    r, j = r[k], S.indices[k]
    cnt = np.bincount(r, minlength=S.shape[0])
    blocks = np.full((S.shape[0], 1 + cnt.max()), -1, np.int32)
    blocks[:, 0] = np.arange(S.shape[0])
    blocks[r, 1 + np.arange(len(r)) - (np.cumsum(cnt) - cnt)[r]] = j
    sel = np.repeat(blocks >= 0, B, axis=1)
    nb = blocks[:, 1:]
    sel[:, B:] &= ~zero.reshape(-1, B)[np.maximum(nb, 0)].reshape(len(nb), -1)
    return blocks, sel


def _indicator(lab, n_cols):
    """CSR (len(lab), n_cols) with a one at (i, lab[i]) for lab[i] >= 0."""
    i = np.flatnonzero(lab >= 0)
    M = sp.csr_array((np.ones(len(i), np.int32), (i, lab[i])), shape=(len(lab), n_cols))
    M.indptr, M.indices = M.indptr.astype(np.int32), M.indices.astype(np.int32)
    return M


def _graph(S):
    """S with the diagonal and zeros removed, int32 indices (pyamg's aggregation needs them)."""
    S = sp.csr_array(S, dtype=np.int32)
    S.setdiag(0)
    S.eliminate_zeros()
    S.indptr, S.indices = S.indptr.astype(np.int32), S.indices.astype(np.int32)
    return S


def _aggregate(S):
    """Aggregate per vertex of the graph S: standard aggregation, unaggregated vertices (no
    neighbours) as singleton aggregates."""
    agg = sp.csr_array(standard_aggregation(S)[0])
    lab = np.full(S.shape[0], -1)
    rows = np.flatnonzero(np.diff(agg.indptr))
    lab[rows] = agg.indices
    lone = lab < 0
    lab[lone] = (agg.shape[1] if agg.nnz else 0) + np.arange(lone.sum())  # none: (n, 1) empty
    return lab


def _frames(x, Tn, na, n_agg):
    """Per aggregate (normal, t1, t2, mean, is_shell): the dominant eigenvector of the summed
    node tensors Tn (sign-free), the tangents and the node mean; is_shell where the sum is > 0."""
    ok = na >= 0
    T = np.zeros((n_agg, 3, 3))
    np.add.at(T, na[ok], Tn[ok])
    xbar = np.zeros((n_agg, 3))
    np.add.at(xbar, na[ok], x[ok])
    xbar /= np.maximum(np.bincount(na[ok], minlength=n_agg), 1)[:, None]
    nrm = np.linalg.eigh(T)[1][:, :, -1]
    e = np.eye(3)[np.argmin(np.abs(nrm), axis=1)]
    t1 = np.cross(nrm, e)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return nrm, t1, np.cross(nrm, t1), xbar, np.trace(T, axis1=1, axis2=2) > 0


def _curvatures(x, Tn, na, n_agg, free):
    """((3 n, 3) Kirchhoff curvature fields k11, k22, k12 of each node's aggregate na in its
    frame, masked; zero on nodes without an aggregate or in non-shell aggregates), is_shell."""
    nrm, t1, t2, xbar, is_shell = _frames(x, Tn, na, n_agg)
    a = np.maximum(na, 0)
    n, t1, t2 = nrm[a], t1[a], t2[a]
    d = x - xbar[a]
    X, Y, Z = (np.sum(d * v, 1)[:, None] for v in (t1, t2, n))
    k11 = -Z * X * t1 + X**2 / 2 * n
    k22 = -Z * Y * t2 + Y**2 / 2 * n
    k12 = -Z * Y * t1 - Z * X * t2 + X * Y * n
    F = np.stack([k11, k22, k12], axis=2)  # (n, 3, 3): node, component, field
    F *= ((na >= 0) & is_shell[a])[:, None, None]
    return F.reshape(-1, 3) * free[:, None], is_shell


def _tensors(op, x):
    """(n_nodes, 3, 3) sums of v v^T over the thickness edges v of the ssh8 elements."""
    E = _shell_nodes(op)
    v = (x[E[:, 4:]] - x[E[:, :4]]).reshape(-1, 3)
    Tn = np.zeros((op.n_nodes, 3, 3))
    vv = v[:, :, None] * v[:, None, :]
    np.add.at(Tn, E[:, :4].ravel(), vv)
    np.add.at(Tn, E[:, 4:].ravel(), vv)
    return Tn


def _smooth(A, T, steps):
    """(P, rho): P = (I - omega / rho D^{-1} A)^steps T, rho = rho(D^{-1} A) by pyamg's
    estimate (np.random, the caller seeds it)."""
    dinv = 1 / A.diagonal()
    L = LinearOperator(A.shape, matvec=lambda v: dinv * (A @ v), dtype=np.float64)
    rho = approximate_spectral_radius(L)
    P, w = sp.csr_array(T), sp.diags_array(OMEGA / rho * dinv)
    for _ in range(steps):
        P = P - w @ (A @ P)
    P.eliminate_zeros()
    return P, rho


def _shell_levels(op, Ab, free, points, max_coarse, shell=None):
    """(levels, A_coarsest, shell, info) for the solid-shell AMG of the BSR hex8 Galerkin
    matrix Ab (3 x 3 node blocks, the Hex8Pattern's sorted indices): levels [(A_l BSR, (blocks,
    sel) patches, P_l)], shell = (group, ((agg, T, zero, shell flags) per level)) for reuse
    (given, fibers, aggregation, frames and fits are skipped) and info: per level rho, the
    group graph and the node -> next aggregate map ("rho", "graphs", "nodes"). Runs under a
    fixed local NumPy seed and leaves the global state unchanged."""
    reuse = shell is not None
    group = _fibers(op) if not reuse else shell[0]
    cols = _columns(op, group)
    patches = (cols, np.repeat(cols >= 0, 3, axis=1))
    fn = np.asarray(free, bool).reshape(-1, 3)
    ng = group.max() + 1
    fg = np.zeros(ng, bool)
    np.logical_or.at(fg, group, fn.any(1))
    gf = np.full(ng, -1)
    gf[fg] = np.arange(fg.sum())  # free group -> graph vertex
    S = sp.csr_array((np.ones(len(Ab.indices), np.int32), Ab.indices, Ab.indptr))
    G = _indicator(gf[group], fg.sum())
    S = _graph(G.T @ S @ G)
    Tn = _tensors(op, points) if not reuse else None
    R = _nullspace(points, fn.ravel()) if not reuse else None
    state = np.random.get_state()
    np.random.seed(0)
    levels, stored, info = [], [], {"rho": [], "graphs": [S], "nodes": []}
    A, na, Q = Ab, gf[group], None
    try:
        while (len(levels) < len(shell[1])) if reuse else (A.shape[0] > max_coarse):
            if reuse:
                agg, T, zero, flags = shell[1][len(levels)]
            else:
                agg = _aggregate(S)
            n_agg = agg.max() + 1
            if not reuse and B * n_agg > STALL * A.shape[0]:
                break
            na = np.where(na >= 0, agg[np.maximum(na, 0)], -1)  # node -> next aggregate
            if not reuse:
                K, flags = _curvatures(points, Tn, na, n_agg, fn.ravel())
                Bf = np.concatenate([R, K], axis=1)
                Bl = Bf if Q is None else Q.T @ Bf
                lab = na if Q is None else agg  # vertex (free group / aggregate) -> aggregate
                T = sp.bsr_array(fit_candidates(_indicator(lab, n_agg).astype(np.float64), Bl)[0])
                zero = np.asarray(T.multiply(T).sum(axis=0)).ravel() == 0
            P, rho = _smooth(A, T, STEPS[min(len(levels), 1)])
            Ac = sp.csr_array(P.T @ (A @ P))
            Ac = (Ac + Ac.T) / 2
            Ac = Ac + sp.diags_array(zero * Ac.diagonal()[~zero].mean())
            levels.append((A, patches, P))
            stored.append((agg, T, zero, flags))
            info["rho"].append(rho)
            Ind = _indicator(agg, n_agg)
            S = _graph(Ind.T @ S @ Ind)
            info["graphs"].append(S)
            patches = _rings(S, zero, flags)
            if not reuse:
                Q = sp.csr_array(T) if Q is None else Q @ T
            info["nodes"].append(na)
            A = sp.bsr_array(Ac, blocksize=(B, B))
            A.sort_indices()
    finally:
        np.random.set_state(state)
    if A.shape[0] > CAP * max_coarse and not reuse:
        raise ValueError(
            f"AMG coarsening stalled at {A.shape[0]} DOF (max_coarse = {max_coarse}); "
            "the dense coarsest factor would be too large"
        )
    return levels, A, (group, tuple(stored)), info
