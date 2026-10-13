"""solve_modes: the public modal-analysis entry point. See docs/eigensolvers.md."""

import math
import time

import jax.numpy as jnp
import numpy as np

from ..operators import dof_mask, m_orthonormalize, rigid_modes, stencil, unstructured
from ..precond import SingularShiftError, amg, gmg, jacobi
from .lobpcg import lobpcg, sigma_estimate


def resolve_precond(mesh, precond, precond_dtype):
    """The preconditioner name solve_modes uses for these options; raises on bad options."""
    voxel = mesh.voxel is not None
    if precond == "auto":
        precond = "gmg" if voxel else "amg"
    if precond not in ("gmg", "amg", "jacobi"):
        raise ValueError(f"precond must be 'auto', 'gmg', 'amg' or 'jacobi', got {precond!r}")
    if precond == "gmg" and not voxel:
        raise ValueError("precond 'gmg' needs a voxel mesh (voxel_mesh)")
    if precond == "amg" and voxel:
        raise ValueError("precond 'amg' needs an unstructured mesh (voxel meshes: 'gmg')")
    if precond_dtype not in ("float64", "float32"):
        raise ValueError(f"precond_dtype must be 'float64' or 'float32', got {precond_dtype!r}")
    if precond != "gmg" and precond_dtype != "float64":
        raise ValueError("precond_dtype 'float32' needs precond 'gmg'")
    return precond


def _precond_sigma(op, free, R, build):
    """(sigma, iterations): the lam_1 estimate (an upper bound) from 1-mode LOBPCG with the
    multigrid build(sigma), started at sigma = 0 unless build refuses it (SingularShiftError),
    else at sigma_estimate's upper bound, and refined while it drops by more than 3x (at most
    4 rounds)."""
    try:
        sigma, P = 0.0, build(0.0)
    except SingularShiftError:
        sigma = sigma_estimate(op, free, R, maxit=10)[0]
        P = build(sigma)
    its = 0
    for rnd in range(4):
        # tol 0.1 and maxit 20 are tuning (docs/eigensolvers.md); never raises on non-convergence
        lam, _, info = lobpcg(op, free, 1, 0.1, R, P, maxit=20)
        its += info["iterations"]
        new, prev = float(lam[0]), sigma
        sigma = new if prev == 0 else min(prev, new)
        if prev == 0 or new > prev / 3 or rnd == 3:
            return sigma, its
        P = None  # one hierarchy alive at a time
        P = build(sigma)


def solve_modes(
    mesh, materials, bcs, n_modes, orientations=None, *, mass="consistent", nodal_mass=None,
    tol=1e-6, maxit=500, seed=0, chunk=None, guards=None, basis_dtype="float64",
    low_memory=False, precond="auto", sigma=None, precond_dtype="float64",
):  # fmt: skip
    """Lowest n_modes elastic modes of mesh under the Dirichlet sets bcs (see dof_mask).

    Voxel meshes get the stencil operator, all others the unstructured one (both run the
    material and element checks first). The rigid modes left by bcs are deflated and returned
    in info["rigid"] (M-orthonormal). precond: "auto" (GMG on voxel meshes, else AMG), "gmg",
    "amg" or "jacobi". The multigrid is built at sigma (a hint >= 0) or at the refined lam_1
    pre-estimate; precond_dtype "float32" runs the GMG V-cycle in fp32. Returns lam (ascending),
    V (n_dof, n_modes) and lobpcg's info plus operator, n_rigid, rigid, frequencies
    (sqrt(lam) / 2 pi), precond, sigma, sigma_iterations and time["precond"]."""
    precond = resolve_precond(mesh, precond, precond_dtype)
    if sigma is not None:
        if not (math.isfinite(sigma) and sigma >= 0):
            raise ValueError(f"sigma must be a finite float >= 0, got {sigma}")
        sigma = float(sigma)
    free = dof_mask(mesh, bcs)  # before the build: bad bcs fail fast
    voxel = mesh.voxel is not None
    name, build = ("stencil", stencil) if voxel else ("unstructured", unstructured)
    op = build(mesh, materials, orientations, mass=mass, nodal_mass=nodal_mass)
    t0 = time.perf_counter()
    R = jnp.asarray(rigid_modes(mesh, free))
    k = R.shape[1]
    if k:
        R = m_orthonormalize(R, op.apply_M(R))  # consumes its first argument
    Rd = R if k else None
    its = 0
    if precond == "jacobi":
        sigma = sigma or 0.0
        P = jacobi(op, free, sigma)
    elif precond == "gmg":
        if sigma is None:
            sigma, its = _precond_sigma(op, free, Rd, lambda s: gmg(op, free, s))
        P = gmg(op, free, sigma, dtype=precond_dtype)
    else:
        # k == 0: rigid_modes found none (per face-connected part), and solve_modes passes no
        # s_K, so A_P(0) is nonsingular
        def build(s):
            return amg(op, free, s, mesh.points, nonsingular=k == 0)

        if sigma is None:
            sigma, its = _precond_sigma(op, free, Rd, build)
        P = build(sigma)
    t_p = time.perf_counter() - t0
    lam, V, info = lobpcg(
        op, free, n_modes, tol, Rd, P, maxit=maxit, seed=seed, chunk=chunk,
        guards=guards, basis_dtype=basis_dtype, low_memory=low_memory,
    )  # fmt: skip
    info["time"]["precond"] = t_p
    info.update(
        operator=name,
        n_rigid=k,
        rigid=np.asarray(R),
        frequencies=np.sqrt(np.maximum(lam, 0)) / (2 * np.pi),
        precond=precond,
        sigma=sigma,
        sigma_iterations=its,
    )
    return lam, V, info
