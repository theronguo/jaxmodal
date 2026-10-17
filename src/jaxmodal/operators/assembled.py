"""Assembled SciPy K and M: the reference oracle for the matrix-free operators.
See docs/operators.md."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from ..elements import check_elements, kernels
from ..elements.shell import shell_mass, shell_stiffness
from ..materials import check_materials, phase_table, phases, ply_table
from .common import check_laminates, check_nodal_mass, check_x64
from .dofmap import element_dofs

_KERNELS = {
    k: jax.jit(getattr(kernels, k), static_argnums=0) for k in ("stiffness", "mass", "lumped_mass")
}


_SHELL = {
    "stiffness": jax.jit(jax.vmap(shell_stiffness)),
    "mass": jax.jit(
        jax.vmap(lambda X, t: jnp.kron(shell_mass(X, t[0]), jnp.eye(3)))
    ),  # m_e (x) I_3
}


def _chunked(kernel, X, conn, p, chunk):
    """kernel(X[conn], p) in chunks of min(chunk, n_el); the last chunk is padded. p is an array
    or a pytree of per-element arrays."""
    n, out = len(conn), []
    c = min(chunk, n)
    for s in range(0, n, c):
        e = np.arange(s, s + c)
        e = np.where(e < n, e, s)  # pad by repeating the chunk's first element
        out.append(np.asarray(kernel(X[conn[e]], jax.tree.map(lambda a, e=e: a[e], p)))[: n - s])
    return np.concatenate(out)


def assemble(
    mesh,
    materials,
    orientations=None,
    mass="consistent",
    nodal_mass=None,
    chunk=1024,
    s_K=None,
    s_M=None,
):
    """Global K and M as symmetrized scipy.sparse.csr_array (n_dof, n_dof), float64, int32 indices.

    nodal_mass: None or (n_nodes,) non-negative point masses, added to the 3 DOFs of each node.
    s_K, s_M: None or (n_el,) non-negative per-element scalings of K_e and M_e (not nodal_mass).
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    check_x64("assemble")
    index, pairs = phases(mesh)
    check_materials(pairs, materials, orientations)
    check_laminates(mesh, index, pairs, materials)
    check_elements(mesh, mass)
    n_nodes = len(mesh.points)
    n_dof = 3 * n_nodes
    if nodal_mass is not None:
        nodal_mass = check_nodal_mass(nodal_mass, n_nodes)
    for name, x in (("s_K", s_K), ("s_M", s_M)):
        x = None if x is None else np.asarray(x, float)
        if x is not None and (x.shape != (mesh.n_el,) or not (np.isfinite(x) & (x >= 0)).all()):
            raise ValueError(f"{name} must be None or a finite, non-negative ({mesh.n_el},) array")
    C, rho = (np.asarray(a) for a in phase_table(pairs, materials, orientations))
    if "ssh8" in mesh.blocks:
        plies = jax.tree.map(np.asarray, ply_table(pairs, materials, orientations))
    dofs = element_dofs(mesh)

    def values(kernel, table, scale):
        out = []
        for t, conn in mesh.blocks.items():
            e = slice(mesh.offsets[t], mesh.offsets[t] + len(conn))
            if t == "ssh8":
                p = jax.tree.map(lambda a, e=e: a[index[e]], plies)
                v = _chunked(_SHELL[kernel], mesh.points, conn, p, chunk)
            else:
                k = partial(_KERNELS[kernel], t)
                v = _chunked(k, mesh.points, conn, table[index[e]], chunk)
            if scale is not None:
                v = v * np.asarray(scale, float)[e].reshape(-1, *[1] * (v.ndim - 1))
            out.append(v.ravel())
        return np.concatenate(out)

    rows = np.concatenate([np.repeat(d, d.shape[1], axis=1).ravel() for d in dofs.values()])
    cols = np.concatenate([np.tile(d, d.shape[1]).ravel() for d in dofs.values()])

    def matrix(vals):  # duplicates summed; symmetrized because the summation order differs
        A = sp.coo_array((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
        return 0.5 * (A + A.T)

    K = matrix(values("stiffness", C, s_K))  # K before M: one matrix's triplets alive at a time
    if mass == "lumped":
        all_dofs = np.concatenate([d.ravel() for d in dofs.values()])
        diag = np.bincount(all_dofs, weights=values("lumped_mass", rho, s_M), minlength=n_dof)
        M = sp.diags_array(diag, format="csr")
    else:
        M = matrix(values("mass", rho, s_M))
    if nodal_mass is not None:
        M = M + sp.diags_array(np.repeat(nodal_mass, 3))
    return K, M
