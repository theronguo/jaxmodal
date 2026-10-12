"""solve_modes: the public modal-analysis entry point. See docs/eigensolvers.md."""

import jax.numpy as jnp
import numpy as np

from ..operators import dof_mask, m_orthonormalize, rigid_modes, stencil, unstructured
from .lobpcg import lobpcg


def solve_modes(
    mesh, materials, bcs, n_modes, orientations=None, *, mass="consistent", nodal_mass=None,
    tol=1e-6, maxit=500, seed=0, chunk=None, guards=None, basis_dtype="float64",
    low_memory=False,
):  # fmt: skip
    """Lowest n_modes elastic modes of mesh under the Dirichlet sets bcs (see dof_mask).

    Voxel meshes get the stencil operator, all others the unstructured one (both run the
    material and element checks first). The rigid modes left by bcs are deflated and returned
    in info["rigid"] (M-orthonormal). Returns lam (ascending), V (n_dof, n_modes) and lobpcg's
    info plus operator, n_rigid, rigid and frequencies (sqrt(lam) / 2 pi)."""
    free = dof_mask(mesh, bcs)  # before the build: bad bcs fail fast
    if mesh.voxel is not None:
        name, build = "stencil", stencil
    else:
        name, build = "unstructured", unstructured
    op = build(mesh, materials, orientations, mass=mass, nodal_mass=nodal_mass)
    R = jnp.asarray(rigid_modes(mesh, free))
    k = R.shape[1]
    if k:
        R = m_orthonormalize(R, op.apply_M(R))  # consumes its first argument
    lam, V, info = lobpcg(
        op, free, n_modes, tol, R if k else None, maxit=maxit, seed=seed, chunk=chunk,
        guards=guards, basis_dtype=basis_dtype, low_memory=low_memory,
    )  # fmt: skip
    info.update(
        operator=name,
        n_rigid=k,
        rigid=np.asarray(R),
        frequencies=np.sqrt(np.maximum(lam, 0)) / (2 * np.pi),
    )
    return lam, V, info
