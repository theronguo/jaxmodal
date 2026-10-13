"""Gaussian probes with covariance M from element-wise quadrature factors of M_e. See
docs/operators.md.

The consistent element mass is m_e = sum_q (w det J)_q N_q N_q^T = B_e B_e^T with
B_e = N^T diag(sqrt(w det J)), so sqrt(rho s_M) B_e (x) I_3 is an exact factor of M_e: no
Cholesky, padded elements (w det J = 0) give B_e = 0. Lumped masses are diagonal:
g = sqrt(diag M) xi. The element inputs are drawn per chunk (unstructured) or per slab (voxel)
inside the scatter loop, so they never exist for the whole mesh.
"""

import jax
import jax.numpy as jnp
from jax import lax

from ..elements import ELEMENTS, kernels
from .matfree import Unstructured, _block_data, _m_scale, _wdet
from .voxel import OFFSETS, _gather, _shifts


def element_factor(op, block, chunk):
    """(c, n, nq) nodal factor F = sqrt(rho s_M) B_e for one chunk of an Unstructured block (a
    slice of _block_data): (F F^T) (x) I_3 = M_e (consistent mass)."""
    N = jnp.asarray(ELEMENTS[block.type].N)  # (nq, n)
    w = jnp.sqrt(_m_scale(op, chunk))[:, None] * jnp.sqrt(_wdet(block, chunk))  # (c, nq)
    return N.T * w[:, None]


def reference_factor(op):
    """(8, 8) B with B B^T = m_e of reference_element(op) (voxel cells, per unit density, VTK
    corner order)."""
    _, wdet = kernels.jacobian("hex8", op.spacing * OFFSETS)
    return jnp.asarray(ELEMENTS["hex8"].N).T * jnp.sqrt(wdet)


def xi_shapes(op):
    """Shapes (without the probe axis) of the standard-normal inputs of mass_sample: "nodal"
    (n_dof,) and, for consistent mass, "elem": per block (unstructured) or for the grid (voxel)
    (n_steps, cells per step, nq, 3), one entry per scatter-loop step."""
    if op.mass == "lumped":
        return {"nodal": (op.n_dof,)}
    if isinstance(op, Unstructured):
        elem = tuple(
            (b.dofs.shape[0] // b.chunk, b.chunk, len(ELEMENTS[b.type].weights), 3)
            for b in op.blocks
        )
    else:
        L, B, A = op.cell.shape
        elem = ((L // op.slab, op.slab * B * A, 8, 3),)
    return {"elem": elem, "nodal": (op.n_dof,)}


def _sample(op, nodal, draw):
    """g = S xi with xi["nodal"] = nodal (n_dof, b) and the element inputs draw(j, i) (cells,
    nq, 3, b) for block j, loop step i."""
    if op.mass == "lumped":
        return jnp.sqrt(op.diag_M())[:, None] * nodal
    g = jnp.sqrt(op.nodal)[:, None] * nodal
    if isinstance(op, Unstructured):
        for j, b in enumerate(op.blocks):
            g = _scatter_block(op, b, lambda i, j=j: draw(j, i), g)
        return g
    return _scatter_cells(op, lambda i: draw(0, i), g)


def _scatter_block(op, b, draw, g):
    d = _block_data(op, b)
    n_chunks = b.dofs.shape[0] // b.chunk
    xs = jax.tree.map(lambda a: a.reshape(n_chunks, b.chunk, *a.shape[1:]), d)

    def step(g, args):
        i, d = args
        u = jnp.einsum("eaq,eqim->eaim", element_factor(op, b, d), draw(i))  # (c, n, 3, m)
        return g.at[d["dofs"]].add(u.reshape(u.shape[0], -1, u.shape[-1])), None

    return lax.scan(step, g, (jnp.arange(n_chunks, dtype=jnp.int32), xs))[0]


def _scatter_cells(op, draw, g):
    """Voxel: the reference factor on every grid cell of a slab (inactive cells weigh 0),
    scattered to the cell corners (dropped nodes are the sentinel n_nodes)."""
    L, B, A = op.cell.shape
    s = op.slab
    Bf = reference_factor(op)

    def step(out, i):
        nd = lax.dynamic_slice_in_dim(op.node, i * s, s + 1)
        el = lax.dynamic_slice_in_dim(op.cell, i * s, s)
        w = jnp.sqrt(op.rho[_gather(op.phase, el)] * _gather(op.s_M, el)).reshape(-1)
        u = jnp.einsum("aq,c,cqim->caim", Bf, w, draw(i))  # (cells, 8, 3, m)
        corner = jnp.stack([nd[a : a + s, b : b + B, c : c + A] for a, b, c in _shifts(op)])
        return out.at[corner.reshape(8, -1).T].add(u, mode="drop"), None

    out = g.reshape(op.n_nodes, 3, -1)
    out, _ = lax.scan(step, out, jnp.arange(L // s, dtype=jnp.int32))
    return out.reshape(g.shape)


@jax.jit
def mass_sample(op, xi):
    """g (n_dof, b) = S xi, linear in xi (a dict shaped as xi_shapes plus a trailing probe axis
    b), with S S^T = M: Cov(g) = M for standard-normal xi. Not masked."""
    elem = xi.get("elem", ())
    return _sample(op, xi["nodal"], lambda j, i: elem[j][i])


@jax.jit
def mass_sample_keys(op, keys):
    """g (n_dof, b) with Cov(g) = M: column j drawn from keys[j] alone (the element inputs per
    loop step from fold_in), so a probe does not depend on the block it is computed in."""
    shapes = xi_shapes(op)

    def nodal(key):
        return jax.random.normal(jax.random.fold_in(key, 0), shapes["nodal"])

    def draw(j, i):
        shape = shapes["elem"][j][1:]

        def one(key):
            return jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, j + 1), i), shape)

        return jnp.moveaxis(jax.vmap(one)(keys), 0, -1)

    return _sample(op, jnp.moveaxis(jax.vmap(nodal)(keys), 0, -1), draw)
