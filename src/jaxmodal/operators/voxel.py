"""Matrix-free voxel stencil operator: one reference hex8 Ke per phase over grid slabs.
See docs/operators.md."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..elements import ELEMENTS, kernels
from ..materials import check_materials, phase_table, phases
from ..mesh.model import CORNERS
from .common import check_nodal_mass, check_v, check_x64, register, scaled
from .dofmap import check_int32

SLAB_CELLS = 2**16  # default slab: about this many cells per scan step (benchmarks/stencil_slab.py)
OFFSETS = ((CORNERS + 1) // 2).astype(int)  # (8, 3) VTK corner offsets in {0, 1}, (x, y, z)


def grid_layout(shape, slab=None):
    """Scan axis, permutation axes (x, y, z) -> stored order, layer count, slab and padded L."""
    if slab is not None and slab < 1:
        raise ValueError(f"slab must be >= 1, got {slab}")
    scan = max(range(3), key=lambda d: (shape[d], d))  # longest axis, ties: the last
    axes = (scan,) + tuple(d for d in (2, 1, 0) if d != scan)
    layers = shape[scan]
    area = shape[axes[1]] * shape[axes[2]]
    slab = max(1, SLAB_CELLS // area) if slab is None else slab
    slab = min(slab, layers)
    return axes, layers, slab, -(-layers // slab) * slab


def _check_grid(mesh):
    """Raise unless mesh.points and the connectivity are exactly the voxel grid."""
    vx = mesh.voxel
    if vx is None:
        raise ValueError("stencil needs a voxel mesh: mesh.voxel is None (no voxel grid)")
    nx, ny, nz = vx.shape
    sx, sxy = nx + 1, (nx + 1) * (ny + 1)
    g = np.flatnonzero(vx.node_map >= 0)
    ijk = np.stack([g % sx, g // sx % (ny + 1), g // sxy], 1)
    expect = np.asarray(vx.origin) + np.asarray(vx.spacing) * ijk
    tol = 1e-12 * max(
        abs(o) + h * n for o, h, n in zip(vx.origin, vx.spacing, vx.shape, strict=True)
    )
    if np.abs(mesh.points[vx.node_map[g]] - expect).max() > tol:
        raise ValueError("mesh does not match its voxel grid: points moved")
    c = np.flatnonzero(vx.element_map >= 0)
    base = c % nx + sx * (c // nx % ny) + sxy * (c // (nx * ny))
    corners = vx.node_map[base[:, None] + OFFSETS @ np.array([1, sx, sxy])]
    if (mesh.blocks["hex8"][vx.element_map[c]] != corners).any():
        raise ValueError("mesh does not match its voxel grid: connectivity differs")


def _grids(vx, axes, L, n_nodes, n_el):
    """node (L+1, B+1, A+1) and cell (L, B, A) id grids in stored order, sentinels n_nodes, n_el."""
    out = []
    for ids, n, sentinel, pad in ((vx.node_map, 1, n_nodes, L + 1), (vx.element_map, 0, n_el, L)):
        g = np.where(ids >= 0, ids, sentinel).reshape(*(s + n for s in vx.shape[::-1]))
        g = g.transpose([2 - d for d in axes])  # (z, y, x) -> stored order
        out.append(np.pad(g, [(0, pad - len(g)), (0, 0), (0, 0)], constant_values=sentinel))
    return out


@dataclasses.dataclass(frozen=True)
class Stencil:
    node: jax.Array  # (L+1, B+1, A+1) int32 mesh node per grid node, n_nodes if dropped/padding
    cell: jax.Array  # (L, B, A) int32 mesh element per grid cell, n_el if inactive/padding
    phase: jax.Array  # (n_el,) int32
    C: jax.Array  # (P, 6, 6)
    rho: jax.Array  # (P,)
    spacing: jax.Array  # (3,)
    s_K: jax.Array  # (n_el,)
    s_M: jax.Array  # (n_el,)
    nodal: jax.Array  # (n_dof,)
    slab: int
    axes: tuple  # grid axes (x, y, z) in stored order, scan axis first
    mass: str
    n_nodes: int
    n_el: int

    @property
    def n_dof(self):
        return 3 * self.n_nodes

    def apply_K(self, v):
        return _apply_K(self, v)

    def apply_M(self, v):
        return _apply_M(self, v)

    def diag_K(self):
        return _diag_K(self)

    def diag_M(self):
        return _diag_M(self)

    def block_diag_K(self):
        return _block_diag_K(self)

    def scaled(self, s_K=None, s_M=None):
        """Copy with new per-element scalings (n_el,); None keeps the current one. Traceable."""
        return scaled(self, s_K, s_M)


register(Stencil, ["slab", "axes", "mass", "n_nodes", "n_el"])


def stencil(mesh, materials, orientations=None, mass="consistent", nodal_mass=None, slab=None):
    """Voxel stencil operator with apply_K, apply_M, diag_K, diag_M, block_diag_K and scaled
    methods."""
    if mass not in ("consistent", "lumped"):
        raise ValueError(f"mass must be 'consistent' or 'lumped', got {mass!r}")
    check_x64("stencil")
    _check_grid(mesh)
    index, pairs = phases(mesh)
    check_materials(pairs, materials, orientations)
    n_nodes, n_el = len(mesh.points), mesh.n_el
    nodal_mass = check_nodal_mass(nodal_mass, n_nodes)
    axes, _, slab, L = grid_layout(mesh.voxel.shape, slab)
    node, cell = _grids(mesh.voxel, axes, L, n_nodes, n_el)
    check_int32(n_dof=3 * n_nodes, node=node.size, cell=cell.size, phase=len(pairs))
    C, rho = phase_table(pairs, materials, orientations)
    return Stencil(
        node=jnp.asarray(node, jnp.int32),
        cell=jnp.asarray(cell, jnp.int32),
        phase=jnp.asarray(index, jnp.int32),
        C=C,
        rho=rho,
        spacing=jnp.asarray(mesh.voxel.spacing, jnp.float64),
        s_K=jnp.ones(n_el),
        s_M=jnp.ones(n_el),
        nodal=jnp.asarray(np.repeat(nodal_mass, 3)),
        slab=slab,
        axes=axes,
        mass=mass,
        n_nodes=n_nodes,
        n_el=n_el,
    )


def reference_element(op):
    """Reference cell matrices: Ke (P, 24, 24) per phase and the scalar mass m_e (8, 8) per unit
    density, on X = spacing * (CORNERS + 1) / 2 (VTK order). Traceable."""
    X = op.spacing * OFFSETS
    Ke = kernels.stiffness("hex8", X[None], op.C)
    _, wdet = kernels.jacobian("hex8", X)
    N = jnp.asarray(ELEMENTS["hex8"].N)
    return Ke, jnp.einsum("q,qa,qb->ab", wdet, N, N)


# ---------------------------------------------------------------- slab loop


def _slabs(op, kern, m, init=None):
    """init (default 0) + the sum over cells of the shifted corner contributions, (n_dof, m);
    kern(nd, el) gives (8, 3, slab, B, A, m) corner-DOF-major values for the slab's cells."""
    L, B, A = op.cell.shape
    s = op.slab

    def step(out, j):
        nd = lax.dynamic_slice_in_dim(op.node, j * s, s + 1)
        el = lax.dynamic_slice_in_dim(op.cell, j * s, s)
        Y = kern(nd, el)
        y = jnp.zeros((3, s + 1, B + 1, A + 1, m))
        for a, (d0, d1, d2) in enumerate(_shifts(op)):
            y = y.at[:, d0 : d0 + s, d1 : d1 + B, d2 : d2 + A].add(Y[a])
        return out.at[nd].add(jnp.moveaxis(y, 0, -2), mode="drop"), None

    out = jnp.zeros((op.n_dof, m)) if init is None else init  # carried, updated in place
    out, _ = lax.scan(step, out.reshape(op.n_nodes, 3, m), jnp.arange(L // s, dtype=jnp.int32))
    return out.reshape(-1, m)


def _shifts(op):
    """(8, 3) corner offsets in stored axis order: the gather and scatter shifts."""
    return OFFSETS[:, list(op.axes)]


def _gather(x, el):
    """Per-cell values of a per-element array; inactive and padding cells get 0."""
    return x.at[el].get(mode="fill", fill_value=0)


def _corners(op, v, nd):
    """U (24, slab B A m): the 8 shifted corner slices of v on the slab, corner-DOF-major."""
    s, (B, A) = nd.shape[0] - 1, (nd.shape[1] - 1, nd.shape[2] - 1)
    u = v.reshape(op.n_nodes, 3, -1).at[nd].get(mode="fill", fill_value=0.0)
    u = jnp.moveaxis(u, -2, 0)  # (3, s+1, B+1, A+1, m)
    return jnp.stack([u[:, d0 : d0 + s, d1 : d1 + B, d2 : d2 + A] for d0, d1, d2 in _shifts(op)])


def _k_apply(op, Ke, v, nd, el):
    U = _corners(op, v, nd)
    shape = U.shape
    U = U.reshape(24, -1)
    s = _gather(op.s_K, el)[..., None]
    if op.rho.shape[0] == 1:
        return (Ke[0] @ U).reshape(shape) * s
    ph = _gather(op.phase, el)[..., None]

    def phase_step(Y, p):  # one accumulator: workspace independent of the phase count
        return Y + (Ke[p] @ U).reshape(shape) * jnp.where(ph == p, s, 0.0), None

    return lax.scan(phase_step, jnp.zeros(shape), jnp.arange(op.rho.shape[0], dtype=jnp.int32))[0]


def _m_weight(op, el):
    return (op.rho[_gather(op.phase, el)] * _gather(op.s_M, el))[..., None]


def _m_apply(op, me, v, nd, el):
    U = _corners(op, v, nd)
    Me = jnp.kron(me, jnp.eye(3))  # (24, 24): the same GEMM as K (fewer temporaries than 8 x 8)
    return (Me @ U.reshape(24, -1)).reshape(U.shape) * _m_weight(op, el)


def _k_diag(op, Ke, el):
    P = op.rho.shape[0]
    d = jnp.diagonal(Ke, axis1=1, axis2=2)  # (P, 24)
    ph, s = _gather(op.phase, el), _gather(op.s_K, el)
    w = jnp.where(ph == jnp.arange(P, dtype=jnp.int32)[:, None, None, None], s, 0.0)  # (P, s, B, A)
    return jnp.einsum("pk,p...->k...", d, w).reshape(8, 3, *el.shape, 1)


def _k_blocks(op, Ke, el):
    P = op.rho.shape[0]
    a = np.arange(24)
    blocks = Ke[:, a[:, None], 3 * (a // 3)[:, None] + np.arange(3)].reshape(P, 8, 3, 3)
    ph, s = _gather(op.phase, el), _gather(op.s_K, el)
    w = jnp.where(ph == jnp.arange(P, dtype=jnp.int32)[:, None, None, None], s, 0.0)
    return jnp.einsum("paij,psyx->aisyxj", blocks, w)


def _m_diag(op, me, el):
    d = me.sum(1) if op.mass == "lumped" else jnp.diagonal(me)  # (8,): row-sum = V/8, or m_aa
    w = _m_weight(op, el)[..., 0]
    return jnp.broadcast_to((d[:, None, None, None] * w)[:, None, ..., None], (8, 3, *el.shape, 1))


@jax.jit
def _apply_K(op, v):
    check_v(v, op.n_dof)
    Ke, _ = reference_element(op)
    return _slabs(op, lambda nd, el: _k_apply(op, Ke, v, nd, el), v.shape[1])


@jax.jit
def _apply_M(op, v):
    check_v(v, op.n_dof)
    if op.mass == "lumped":
        return _diag_M(op)[:, None] * v
    _, me = reference_element(op)
    kern = lambda nd, el: _m_apply(op, me, v, nd, el)  # noqa: E731
    return _slabs(op, kern, v.shape[1], op.nodal[:, None] * v)  # adding it after costs a vector


@jax.jit
def _diag_K(op):
    Ke, _ = reference_element(op)
    return _slabs(op, lambda nd, el: _k_diag(op, Ke, el), 1)[:, 0]


@jax.jit
def _block_diag_K(op):
    Ke, _ = reference_element(op)
    return _slabs(op, lambda nd, el: _k_blocks(op, Ke, el), 3).reshape(op.n_nodes, 3, 3)


@jax.jit
def _diag_M(op):
    _, me = reference_element(op)
    return _slabs(op, lambda nd, el: _m_diag(op, me, el), 1, op.nodal[:, None])[:, 0]
