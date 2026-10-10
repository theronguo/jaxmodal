"""Elastic stiffness constructors, checks and the per-phase table. See docs/materials.md.

Constructors and phase_table are pure jax.numpy (traceable, differentiable); check_stiffness,
check_materials and phases are host NumPy.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rotation import rotate

_BLOCK = np.zeros((6, 6))
_BLOCK[:3, :3] = 1.0
_LAME_MU = np.diag([2.0, 2, 2, 1, 1, 1])


class Material(NamedTuple):
    C: jax.Array  # (6, 6) local stiffness, Voigt [11, 22, 33, 23, 13, 12]
    rho: jax.Array  # () density


def isotropic(E, nu):
    E, nu = jnp.asarray(E, float), jnp.asarray(nu, float)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return lam * _BLOCK + E / (2 * (1 + nu)) * _LAME_MU


def orthotropic(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23):
    """C = S^-1; nu_ij = -eps_j / eps_i under sigma_i, nu_ij / E_i = nu_ji / E_j (i < j given)."""
    E1, E2, E3, nu12, nu13, nu23, G12, G13, G23 = (
        jnp.asarray(x, float) for x in (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)
    )
    z = jnp.zeros_like(E1)
    S = jnp.stack(
        [
            jnp.stack([1 / E1, -nu12 / E1, -nu13 / E1, z, z, z]),
            jnp.stack([-nu12 / E1, 1 / E2, -nu23 / E2, z, z, z]),
            jnp.stack([-nu13 / E1, -nu23 / E2, 1 / E3, z, z, z]),
            jnp.stack([z, z, z, 1 / G23, z, z]),
            jnp.stack([z, z, z, z, 1 / G13, z]),
            jnp.stack([z, z, z, z, z, 1 / G12]),
        ]
    )
    return jnp.linalg.inv(S)


def transversely_isotropic(E1, E2, nu12, nu23, G12):
    """Symmetry axis = local 1 (isotropic in the local 2-3 plane)."""
    return orthotropic(E1, E2, E2, nu12, nu12, nu23, G12, G12, E2 / (2 * (1 + nu23)))


def anisotropic(C):
    return jnp.asarray(C, float)


def check_stiffness(C):
    """Raise ValueError unless C is a finite, symmetric, positive definite (6, 6) array."""
    C = np.asarray(C, float)
    if C.shape != (6, 6) or not np.isfinite(C).all():
        raise ValueError("stiffness must be a finite (6, 6) array")
    if np.abs(C - C.T).max() > 1e-12 * np.abs(C).max():
        raise ValueError("stiffness is not symmetric")
    lam = np.linalg.eigvalsh(C)
    if lam[0] <= 1e-12 * lam[-1]:
        raise ValueError(
            f"stiffness is not positive definite (eigenvalues {lam[0]:.3g} .. {lam[-1]:.3g})"
        )


def phases(mesh):
    """Element -> phase index (int64) and the distinct (material, orientation) label pairs,
    in order of first appearance."""
    keys = np.stack([mesh.material, mesh.orientation], 1)
    uniq, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    pairs = tuple((int(m), int(o)) for m, o in uniq[order])
    return np.argsort(order)[inv.ravel()], pairs


def phase_table(pairs, materials, orientations=None):
    """Rotated stiffness (P, 6, 6) and density (P,) per phase; pure jax.numpy."""
    orientations = {0: jnp.eye(3), **(orientations or {})}  # only label 0 defaults
    C = jnp.stack([jnp.asarray(materials[m].C, float) for m, _ in pairs])
    R = jnp.stack([jnp.asarray(orientations[o], float) for _, o in pairs])
    rho = jnp.stack([jnp.asarray(materials[m].rho, float) for m, _ in pairs])
    return rotate(C, R), rho


def check_materials(pairs, materials, orientations=None):
    """Host validation of the inputs of phase_table: labels, stiffness, density, rotations."""
    orientations = {0: np.eye(3), **(orientations or {})}
    for m in sorted({m for m, _ in pairs}):
        if m not in materials:
            raise ValueError(f"material label {m} is in the mesh but not in materials")
        check_stiffness(materials[m].C)
        rho = np.asarray(materials[m].rho, float)
        if rho.shape != () or not np.isfinite(rho) or rho <= 0:
            raise ValueError(f"material {m}: rho must be a finite positive scalar")
    for o in sorted({o for _, o in pairs}):
        if o not in orientations:
            raise ValueError(f"orientation label {o} is in the mesh but not in orientations")
        R = np.asarray(orientations[o], float)
        if R.shape != (3, 3) or not np.isfinite(R).all():
            raise ValueError(f"orientation {o} must be a finite (3, 3) array")
        if np.abs(R.T @ R - np.eye(3)).max() > 1e-12 or np.linalg.det(R) <= 0:
            raise ValueError(f"orientation {o} is not a rotation (orthogonal, det = +1)")
