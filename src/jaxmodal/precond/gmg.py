"""Geometric multigrid V-cycle for voxel stencils: semi-coarsened nested grids, Galerkin cell
matrices in block-ELL, stopped-axis group block smoothers under Chebyshev, dense coarsest
Cholesky, and deflation of the rigid modes of stiff islands. See docs/preconditioners.md."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import cho_solve
from jax.tree_util import Partial
from scipy import ndimage

from ..linalg.ell import BlockELL
from ..mesh import voxel_mesh
from ..operators.common import check_x64, register
from ..operators.constraints import ap_operator, rigid_modes
from ..operators.dofmap import check_int32
from ..operators.voxel import OFFSETS, Stencil, grid_ids, reference_element
from .chebyshev import chebyshev, lmax_estimate
from .jacobi import _block_jacobi, _check_sigma, _inverse_blocks

SETUP_CHUNK = 2**14  # child cells per setup loop step (bounds the setup temporaries)
ELL_CHUNK = 2048  # block rows per ELL scan step (block_ell's default chunk)
ISLAND_CONTRAST = 1e3  # max / min s_K from which stiff islands are deflated
MAX_ISLANDS = 16  # largest islands deflated (6 rigid modes each)
SLOT = (OFFSETS[None, :] - OFFSETS[:, None] + 1) @ np.array([1, 3, 9])  # (8, 8) ELL slot of (a, b)


# ---------------------------------------------------------------- host hierarchy


def _coarsen(n, p):
    """1D rule for n elements with node positions p (n+1,): coarse count c, the coarse node
    indices X into the fine nodes, the parent of each fine cell, and per fine cell the (2, 2)
    weights of its corners (rows) on its parent's corners (columns), from the positions."""
    if n <= 2:
        X, parent = np.arange(n + 1), np.arange(n)
    else:
        c = -(-n // 2)
        X, parent = np.append(2 * np.arange(c), n), np.arange(n) // 2
    lo, hi = p[X[parent]], p[X[parent + 1]]
    w = (p[np.stack([np.arange(n), np.arange(1, n + 1)], 1)] - lo[:, None]) / (hi - lo)[:, None]
    return len(X) - 1, X, parent, np.stack([1 - w, w], -1)  # (n, 2 fine, 2 coarse)


def _interp(X, p):
    """Fine node i -> (j0, j1, w0, w1) on the coarse nodes X, linear in the positions p."""
    i = np.arange(len(p))
    k = np.clip(np.searchsorted(X, i, side="right") - 1, 0, max(len(X) - 2, 0))
    j1 = np.minimum(k + 1, len(X) - 1)
    w1 = np.where(j1 > k, (p - p[X[k]]) / (p[X[j1]] - p[X[k]]), 0.0)
    return k, j1, 1 - w1, w1


@dataclasses.dataclass
class _Grid:
    """One level on the host: shape, active cells, positions, numbering and connectivity."""

    shape: tuple
    mask: np.ndarray
    pos: tuple

    def __post_init__(self):
        mesh = voxel_mesh(self.shape, mask=self.mask)
        vx = mesh.voxel
        self.node_map, self.element_map, self.conn = (
            vx.node_map,
            vx.element_map,
            mesh.blocks["hex8"],
        )
        nx, ny, nz = self.shape
        c = np.flatnonzero(vx.element_map >= 0)
        self.ijk = np.stack([c % nx, c // nx % ny, c // (nx * ny)], 1)
        self.n_nodes = int((self.node_map >= 0).sum())

    def node_ijk(self):
        nx, ny, _ = self.shape
        g = np.flatnonzero(self.node_map >= 0)
        return np.stack([g % (nx + 1), g // (nx + 1) % (ny + 1), g // ((nx + 1) * (ny + 1))], 1)


def _coarse(F):
    """Coarse grid of F, the coarsening tables and each fine cell's parent coarse cell id."""
    rules = [_coarsen(n, p) for n, p in zip(F.shape, F.pos, strict=True)]
    shape = tuple(r[0] for r in rules)
    parent = np.stack([rules[d][2][F.ijk[:, d]] for d in range(3)], 1)
    mask = np.zeros(shape, bool)
    mask[tuple(parent.T)] = True
    C = _Grid(shape, mask, tuple(p[r[1]] for p, r in zip(F.pos, rules, strict=True)))
    cell = C.element_map[parent[:, 0] + shape[0] * (parent[:, 1] + shape[1] * parent[:, 2])]
    interp = [_interp(r[1], p) for p, r in zip(F.pos, rules, strict=True)]
    return C, cell, [r[3] for r in rules], interp


def _groups(G):
    """Stopped-axis groups of G: (idx (n_groups, k) node ids, n_nodes for absent slots; pos
    (n_nodes,) each node's flat slot; grp, loc per node)."""
    stopped = [n <= 2 for n in G.shape]
    ijk = G.node_ijk()
    if any(stopped):
        _, grp = np.unique(np.where(stopped, 0, ijk)[:, ::-1], axis=0, return_inverse=True)
        grp = grp.ravel()
    else:
        grp = np.arange(G.n_nodes)  # k = 1: groups are the nodes
    loc, k = np.zeros(len(ijk), np.int64), 1
    for d in range(3):
        if stopped[d]:
            loc, k = loc + k * ijk[:, d], k * (G.shape[d] + 1)
    idx = np.full((grp.max() + 1, k), G.n_nodes)
    idx[grp, loc] = np.arange(G.n_nodes)
    pairs = [
        (a, b)
        for a in range(8)
        for b in range(8)
        if all(stopped[d] or OFFSETS[a, d] == OFFSETS[b, d] for d in range(3))
    ]
    return idx, grp * k + loc, grp, loc, np.array(pairs).T


def _ell_cols(G):
    """(n_nodes, 27) neighbour node per slot (self where absent)."""
    nx, ny, nz = G.shape
    ijk = G.node_ijk()
    cols = np.empty((G.n_nodes, 27), np.int64)
    for s in range(27):
        q = ijk + np.array([s % 3, s // 3 % 3, s // 9]) - 1
        ok = ((q >= 0) & (q <= np.array(G.shape))).all(1)
        nb = np.full(G.n_nodes, -1)
        nb[ok] = G.node_map[q[ok, 0] + (nx + 1) * (q[ok, 1] + (ny + 1) * q[ok, 2])]
        cols[:, s] = np.where(nb >= 0, nb, np.arange(G.n_nodes))
    return cols


# ---------------------------------------------------------------- device setup sweep


def _ploc(P1, ijk):
    """(c, 8, 8) local trilinear prolongation per child cell from the per-axis (n, 2, 2)."""
    o = OFFSETS
    out = 1.0
    for d in range(3):
        t = P1[d][ijk[:, d]]
        out = out * t[:, o[:, None, d], o[None, :, d]]
    return out


def _fine_mats(src, e):
    """(c, 24, 24) masked fine element matrices of A_P plus the owned nodal diagonal."""
    Ke, Me, phase, s_K, w_M, conn, free, own, d = src
    dof = (3 * conn[e][:, :, None] + jnp.arange(3)).reshape(-1, 24)
    m = free[dof]
    A = s_K[e][:, None, None] * Ke[phase[e]] + w_M[e][:, None, None] * Me
    A = A * m[:, :, None] * m[:, None, :]
    return A + jax.vmap(jnp.diag)(d[dof] * jnp.repeat(own[e], 3, axis=1))


@functools.partial(jax.jit, static_argnames=("n_parent", "chunk", "n_groups", "k", "n_rows"))
def _sweep(src, P1, ijk, parent, conn, gi, loc, pairs, n_parent, chunk, n_groups, k, n_rows):
    """One pass over the child matrices of a level: the parent cells sum P^T A P, the level's
    group blocks (n_groups, 3k, 3k) and, if n_rows, its 27-slot ELL values."""
    n = ijk.shape[0]
    cells = jnp.zeros((n_parent, 8, 3, 8, 3))
    B = jnp.zeros((n_groups, k, 3, k, 3))
    vals = jnp.zeros((max(n_rows, 1), 27, 3, 3))
    pa, pb = pairs
    slot = jnp.asarray(SLOT)

    def step(i, acc):
        cells, B, vals = acc
        start = jnp.minimum(i * chunk, n - chunk)
        e = start + jnp.arange(chunk)
        A = src[e] if isinstance(src, jax.Array) else _fine_mats(src, e)
        A = A * (e >= i * chunk)[:, None, None]  # rows a clamped last chunk already did
        A5 = A.reshape(-1, 8, 3, 8, 3)
        Pl = _ploc(P1, ijk[e])
        cells = cells.at[parent[e]].add(jnp.einsum("eac,eaibj,ebd->ecidj", Pl, A5, Pl))
        Ab = A5.transpose(0, 1, 3, 2, 4)  # (c, a, b, 3, 3)
        na, nb = conn[e][:, pa], conn[e][:, pb]
        B = B.at[gi[na], loc[na], :, loc[nb], :].add(Ab[:, pa, pb])
        if n_rows:
            vals = vals.at[conn[e][:, :, None], slot].add(Ab)
        return cells, B, vals

    cells, B, vals = lax.fori_loop(0, -(-n // chunk), step, (cells, B, vals))
    return cells.reshape(-1, 24, 24), B.reshape(n_groups, 3 * k, 3 * k), vals


# ---------------------------------------------------------------- pytrees and apply


@dataclasses.dataclass(frozen=True)
class Transfer:
    """Prolongation coarse -> fine through the full coarse grid (per-axis 2-point gathers)."""

    gc: jax.Array  # coarse active nodes' flat grid index
    gf: jax.Array  # fine active nodes' flat grid index
    tabs: tuple  # per axis (j0, j1, w0, w1) on the fine grid nodes
    cshape: tuple
    n_coarse: int

    def __call__(self, x):
        return _prolong(self, x)


register(Transfer, ["cshape", "n_coarse"])


def _prolong(T, x):
    nx, ny, nz = T.cshape
    m = x.shape[1]
    G = jnp.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3, m), x.dtype)
    G = G.at[T.gc].set(x.reshape(-1, 3, m), unique_indices=True)
    G = G.reshape(nz + 1, ny + 1, nx + 1, 3, m)
    for ax, (j0, j1, w0, w1) in zip((2, 1, 0), T.tabs, strict=True):  # x, y, z
        sh = [1] * 5
        sh[ax] = -1
        w0, w1 = (w.reshape(sh).astype(x.dtype) for w in (w0, w1))
        G = jnp.take(G, j0, ax) * w0 + jnp.take(G, j1, ax) * w1
    return G.reshape(-1, 3, m)[T.gf].reshape(-1, m)


def _restrict(T, r):
    shape = jax.ShapeDtypeStruct((3 * T.n_coarse, r.shape[1]), r.dtype)
    return jax.linear_transpose(functools.partial(_prolong, T), shape)(r)[0]


def _group_apply(inv, idx, pos, r):
    """Gather-only group block Jacobi: absent slots read a zero row; groups partition nodes."""
    m = r.shape[1]
    rp = jnp.concatenate([r.reshape(-1, 3, m), jnp.zeros((1, 3, m), r.dtype)])
    rg = rp[idx].reshape(idx.shape[0], -1, m)
    y = jnp.einsum("gij,gjm->gim", inv, rg).reshape(-1, 3, m)
    return y[pos].reshape(r.shape)


@dataclasses.dataclass(frozen=True)
class GMG:
    """V-cycle approximating A_P^{-1}, deflated: r -> Q r + (I - Q A) m V(m (I - A Q) r),
    Q = Z E^+ Z^T. Levels are (A, S) for all but the coarsest, transfers[l] maps level l+1 to
    l, L is the coarsest Cholesky factor."""

    levels: tuple
    transfers: tuple
    L: jax.Array
    free: jax.Array
    defl: tuple  # (rows, Z, AZ, Einv): island rigid modes and A_P Z on their support rows

    @property
    def n_dof(self):
        return self.free.shape[0]

    def __call__(self, r):
        return _vcycle_apply(self, r)


register(GMG, [])


def _vcycle(P, b, lvl):
    if lvl == len(P.levels):
        return cho_solve((P.L, True), b)
    A, S = P.levels[lvl]
    T = P.transfers[lvl]
    x = S(b)
    x = x + T(_vcycle(P, _restrict(T, b - A(x)), lvl + 1))
    return x + S(b - A(x))


@jax.jit
def _vcycle_apply(P, r):
    m = P.free[:, None]
    rows, Z, AZ, Einv = P.defl
    r = m * r
    y = Einv @ (Z.T @ r[rows])
    v = m * _vcycle(P, r - jnp.zeros_like(r).at[rows].add(AZ @ y), 0)
    w = Einv @ (AZ.T @ v[rows])
    return v + jnp.zeros_like(r).at[rows].add(Z @ (y - w))


# ---------------------------------------------------------------- builder


def _check_sigma0(op, free, shape, cells, s_K):
    """sigma == 0: refuse unless A_P is provably nonsingular (no zero-stiffness cell, no rigid
    mode left by the BCs on a mesh rigid_modes accepts)."""
    msg = "cannot verify that A_P is nonsingular at sigma = 0"
    if (s_K == 0).any():
        raise ValueError(f"zero-stiffness cells at sigma = 0: {msg}; use sigma > 0")
    mesh = voxel_mesh(shape, tuple(np.asarray(op.spacing)), mask=cells >= 0)
    try:
        k = rigid_modes(mesh, free).shape[1]
    except ValueError as err:
        raise ValueError(f"{msg} ({err}); use sigma > 0") from err
    if k:
        raise ValueError(f"A_P is singular at sigma = 0 ({k} rigid modes left); use sigma > 0")


def _islands(cell_ids, s_K):
    """Cell ids of the stiff islands: node-connected components of the cells with s_K above
    10 min(s_K), all but the largest, if max / min s_K >= ISLAND_CONTRAST (largest first)."""
    lo = s_K.min()
    if not s_K.max() >= ISLAND_CONTRAST * lo:
        return []
    act = cell_ids >= 0
    stiff = np.zeros(cell_ids.shape, bool)
    stiff[act] = s_K[cell_ids[act]] > 10 * lo
    lab, k = ndimage.label(stiff, np.ones((3, 3, 3)))
    parts = sorted((cell_ids[lab == c] for c in range(1, k + 1)), key=len, reverse=True)
    return parts[1 : 1 + MAX_ISLANDS]


def _deflation(A0, F, spacing, free, islands):
    """(rows, Z, AZ, Einv): masked rigid modes Z of each island (translations, rotations about
    its centroid) and A_P Z on their support rows, padded to power-of-2 sizes (padding rows
    point at DOF 0 with zero values); Einv = (Z^T A_P Z)^+."""
    xyz = F.node_ijk() * spacing
    n = 1 << max(len(islands) - 1, 0).bit_length()
    cols, sup = [], [np.zeros(0, np.int64)]
    for e in islands:
        nodes = np.unique(F.conn[e])
        x = xyz[nodes] - xyz[nodes].mean(0)
        z = np.zeros((len(nodes), 3, 6))
        z[:, :, :3] = np.eye(3)
        z[:, :, 3:] = np.cross(np.eye(3)[:, None], x).transpose(1, 2, 0)  # e_j x x
        dof = (3 * nodes[:, None] + np.arange(3)).ravel()
        cols.append((dof, z.reshape(-1, 6) * free[dof, None]))
        ring = np.unique(F.conn[np.isin(F.conn, nodes).any(1)])
        sup.append((3 * ring[:, None] + np.arange(3)).ravel())
    rows = np.unique(np.concatenate(sup))
    nr = max(256, 1 << (len(rows) - 1).bit_length())
    Z = np.zeros((nr, 6 * n))
    AZ = jnp.zeros((nr, 6 * n))
    for c, (dof, z) in enumerate(cols):
        Z[np.searchsorted(rows, dof), 6 * c : 6 * c + 6] = z
        full = jnp.zeros((len(free), 6)).at[dof].set(z)
        AZ = AZ.at[: len(rows), 6 * c : 6 * c + 6].set(A0(full)[rows])
    E = Z.T @ np.asarray(AZ)
    dg = np.diag(E)
    sc = np.where(dg > 0, 1 / np.sqrt(np.where(dg > 0, dg, 1)), 0.0)
    lam, U = np.linalg.eigh(sc[:, None] * E * sc[None])
    keep = lam > 1e-12 * max(lam.max(), 0)
    Einv = sc[:, None] * (U[:, keep] / lam[keep]) @ U[:, keep].T * sc[None]
    rows = np.concatenate([rows, np.zeros(nr - len(rows), np.int64)])
    return jnp.asarray(rows, jnp.int32), jnp.asarray(Z), AZ, jnp.asarray(Einv)


def _smoother(A, B, idx, pos, degree, ratio):
    pres = np.repeat(idx < len(pos), 3, axis=1)
    inv, m = _inverse_blocks(B, pres)
    if not bool(jnp.isfinite(inv).all()):
        raise ValueError("smoother blocks are not finite (A_P is singular)")
    if idx.shape[1] == 1:
        D = Partial(_block_jacobi, inv, m)
    else:
        D = Partial(_group_apply, inv, jnp.asarray(idx, jnp.int32), jnp.asarray(pos, jnp.int32))
    return chebyshev(A, D, lmax_estimate(A, D), degree, ratio)


def _ell_rows(n):
    """ELL chunk and padded row count for n block rows."""
    c = min(ELL_CHUNK, -(-n // 8) * 8)
    return c, -(-n // c) * c


def _ell(vals, cols):
    """BlockELL from (rp, 27, 3, 3) values already padded to _ell_rows and (n, 27) columns."""
    n = len(cols)
    c, rp = _ell_rows(n)
    colsp = np.zeros((rp, 27), np.int32)
    colsp[:n] = cols
    nnz = (cols != np.arange(n)[:, None]).sum() + n
    return BlockELL((jnp.asarray(colsp),), (vals,), None, 3 * n, (n,), (c,), 27 * n / nnz)


def gmg(op, free, sigma, *, degree=2, coarse_degree=4, ratio=30.0):
    """Geometric multigrid V-cycle for A_P = m (K + sigma M) m + s (I - m) of a voxel Stencil,
    as a pytree callable on (n_dof, k) blocks (masked output). sigma >= 0 is concrete and
    required: choose it physically (the sigma pre-estimate); sigma = 0 is refused unless A_P
    is provably nonsingular. Chebyshev degree `degree` on the fine level, `coarse_degree` on
    the others. Needs the voxel_mesh node order (no rcm/renumber)."""
    check_x64("gmg")
    if not isinstance(op, Stencil):
        raise TypeError(f"gmg needs a voxel Stencil operator, got {type(op).__name__}")
    _check_sigma(sigma)
    sigma = float(sigma)
    free = np.asarray(free)
    cell_ids, node_ids = grid_ids(op)
    shape = cell_ids.shape
    flat_c, flat_n = cell_ids.transpose(2, 1, 0).ravel(), node_ids.transpose(2, 1, 0).ravel()
    if (flat_c[flat_c >= 0] != np.arange(op.n_el)).any() or (
        flat_n[flat_n >= 0] != np.arange(op.n_nodes)
    ).any():
        raise ValueError("gmg needs the voxel_mesh node order; do not renumber voxel meshes")
    check_int32(grid_nodes=int(np.prod(np.array(shape) + 1)))
    s_K = np.asarray(op.s_K)
    if sigma == 0:
        _check_sigma0(op, free, shape, cell_ids, s_K)

    A0 = ap_operator(op, free, sigma)
    fl = A0.free
    d = sigma * op.nodal * fl + A0.s * (1 - fl)
    Ke, me = reference_element(op)
    Me = jnp.kron(me if op.mass == "consistent" else jnp.diag(me.sum(1)), jnp.eye(3))
    F = _Grid(shape, cell_ids >= 0, tuple(np.arange(n + 1.0) for n in shape))
    own = np.zeros((op.n_el, 8), bool)  # each node's first (lowest element, corner) owns it
    own.flat[np.unique(F.conn.ravel(), return_index=True)[1]] = True
    w_M = sigma * op.rho[op.phase] * op.s_M
    src = (Ke, Me, op.phase, op.s_K, w_M, jnp.asarray(F.conn, jnp.int32), fl, jnp.asarray(own), d)

    levels, transfers, G, A = [], [], F, A0
    while True:
        C, parent, P1, interp = _coarse(G)
        idx, pos, grp, loc, pairs = _groups(G)
        n_rows = _ell_rows(G.n_nodes)[1] if levels else 0
        cells, B, vals = _sweep(
            src,
            tuple(jnp.asarray(t) for t in P1),
            jnp.asarray(G.ijk, jnp.int32),
            jnp.asarray(parent, jnp.int32),
            jnp.asarray(G.conn, jnp.int32),
            jnp.asarray(grp, jnp.int32),
            jnp.asarray(loc, jnp.int32),
            tuple(jnp.asarray(p, jnp.int32) for p in pairs),
            n_parent=len(C.conn),
            chunk=min(SETUP_CHUNK, len(G.conn)),
            n_groups=len(idx),
            k=idx.shape[1],
            n_rows=n_rows,
        )
        if levels:
            A = _ell(vals, _ell_cols(G))
        levels.append((A, _smoother(A, B, idx, pos, coarse_degree if levels else degree, ratio)))
        gc = np.flatnonzero(C.node_map >= 0)
        transfers.append(
            Transfer(
                jnp.asarray(gc, jnp.int32),
                jnp.asarray(np.flatnonzero(G.node_map >= 0), jnp.int32),
                tuple(tuple(jnp.asarray(a) for a in t) for t in interp),
                C.shape,
                C.n_nodes,
            )
        )
        G, src = C, cells
        if max(G.shape) <= 2:
            break

    L = jnp.linalg.cholesky(_dense(cells, G.conn, G.n_nodes))
    if not bool(jnp.isfinite(L).all()):
        raise ValueError("coarsest factor is not finite")
    defl = _deflation(A0, F, np.asarray(op.spacing), free, _islands(cell_ids, s_K))
    return GMG(tuple(levels), tuple(transfers), L, fl, defl)


def _dense(cells, conn, n):
    """Dense (3n, 3n) matrix summed from (c, 24, 24) cell matrices."""
    dof = (3 * conn[:, :, None] + np.arange(3)).reshape(-1, 24)
    D = np.zeros((3 * n, 3 * n))
    np.add.at(D, (dof[:, :, None], dof[:, None, :]), np.asarray(cells))
    return jnp.asarray(D)
