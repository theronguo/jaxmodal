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


# Options solve_modes forwards (docs/eigensolvers.md#options); tests pin them to the signatures.
OPERATOR_OPTIONS = {
    "unstructured": ("mode", "path", "chunk", "range_nodes"),
    "stencil": ("slab",),
}
PRECOND_OPTIONS = {
    "gmg": ("degree", "coarse_degree", "ratio"),
    "amg": ("degree", "ratio", "max_coarse"),
    "jacobi": (),
}
SIGMA_OPTIONS = ("tol", "maxit", "rounds", "drop", "estimate_maxit")
RESERVED = (
    "mesh, materials, orientations, mass, nodal_mass, sigma, dtype (precond_dtype), nonsingular, "
    "aggregates"
)


def _check_options(name, opts, allowed):
    """opts as a dict, or ValueError naming the unknown keys and the allowed ones."""
    opts = dict(opts or {})
    bad = sorted(set(opts) - set(allowed))
    if bad:
        raise ValueError(
            f"unknown {name} option(s) {bad}; allowed: {sorted(allowed)}. solve_modes sets "
            f"{RESERVED} itself or through its own keywords"
        )
    return opts


def _check_sigma_options(opts):
    for key in ("maxit", "rounds", "estimate_maxit"):
        if key in opts and not (isinstance(opts[key], int) and opts[key] >= 1):
            raise ValueError(f"sigma_options {key} must be an int >= 1, got {opts[key]!r}")
    if "tol" in opts and not opts["tol"] > 0:
        raise ValueError(f"sigma_options tol must be > 0, got {opts['tol']!r}")
    if "drop" in opts and not opts["drop"] > 1:
        raise ValueError(f"sigma_options drop must be > 1, got {opts['drop']!r}")


def _precond_sigma(
    op, free, R, build, *, seed=0, tol=0.1, maxit=20, rounds=4, drop=3.0, estimate_maxit=10
):
    """(sigma, iterations): the lam_1 estimate (an upper bound) from 1-mode LOBPCG (tol, maxit
    per round) with the multigrid build(sigma), started at sigma = 0 unless build refuses it
    (SingularShiftError), else at sigma_estimate's upper bound (estimate_maxit iterations), and
    refined while it drops by more than drop x (at most rounds rounds)."""
    try:
        sigma, P = 0.0, build(0.0)
    except SingularShiftError:
        sigma = sigma_estimate(op, free, R, maxit=estimate_maxit, seed=seed)[0]
        P = build(sigma)
    its = 0
    for rnd in range(rounds):
        # the defaults are tuning (docs/eigensolvers.md); never raises on non-convergence
        lam, _, info = lobpcg(op, free, 1, tol, R, P, maxit=maxit, seed=seed)
        its += info["iterations"]
        new, prev = float(lam[0]), sigma
        sigma = new if prev == 0 else min(prev, new)
        if prev == 0 or new > prev / drop or rnd == rounds - 1:
            return sigma, its
        P = None  # one hierarchy alive at a time
        P = build(sigma)


def solve_modes(
    mesh, materials, bcs, n_modes, orientations=None, *, mass="consistent", nodal_mass=None,
    tol=1e-6, maxit=500, seed=0, chunk=None, guards=None, basis_dtype="float64",
    low_memory=False, precond="auto", sigma=None, precond_dtype="float64", X0=None,
    operator_options=None, precond_options=None, sigma_options=None,
):  # fmt: skip
    """Lowest n_modes elastic modes of mesh under the Dirichlet sets bcs (see dof_mask).

    Voxel meshes get the stencil operator, all others the unstructured one (both run the
    material and element checks first). The rigid modes left by bcs are deflated and returned
    in info["rigid"] (M-orthonormal). precond: "auto" (GMG on voxel meshes, else AMG), "gmg",
    "amg" or "jacobi". The multigrid is built at sigma (a hint >= 0) or at the refined lam_1
    pre-estimate; precond_dtype "float32" runs the GMG V-cycle in fp32. X0 (n_dof, j <= block
    size) seeds lobpcg's first chunk. operator_options, precond_options and sigma_options are
    dicts forwarded to the operator builder, the preconditioner builder and the pre-estimate
    (allowed keys: OPERATOR_OPTIONS, PRECOND_OPTIONS, SIGMA_OPTIONS; docs/eigensolvers.md#options).
    Returns lam (ascending),
    V (n_dof, n_modes) and lobpcg's info plus operator, n_rigid, rigid, frequencies
    (sqrt(lam) / 2 pi), precond, sigma, sigma_iterations and time["precond"]."""
    precond = resolve_precond(mesh, precond, precond_dtype)
    voxel = mesh.voxel is not None
    name, build = ("stencil", stencil) if voxel else ("unstructured", unstructured)
    op_opts = _check_options("operator", operator_options, OPERATOR_OPTIONS[name])
    pc_opts = _check_options(f"{precond} precond", precond_options, PRECOND_OPTIONS[precond])
    sg_opts = _check_options("sigma", sigma_options, SIGMA_OPTIONS)
    if sg_opts and (sigma is not None or precond == "jacobi"):
        raise ValueError("sigma_options need the pre-estimate: no sigma= and no precond 'jacobi'")
    _check_sigma_options(sg_opts)
    if sigma is not None:
        if not (math.isfinite(sigma) and sigma >= 0):
            raise ValueError(f"sigma must be a finite float >= 0, got {sigma}")
        sigma = float(sigma)
    free = dof_mask(mesh, bcs)  # before the build: bad bcs fail fast
    op = build(mesh, materials, orientations, mass=mass, nodal_mass=nodal_mass, **op_opts)
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
            sigma, its = _precond_sigma(
                op, free, Rd, lambda s: gmg(op, free, s, **pc_opts), seed=seed, **sg_opts
            )
        P = gmg(op, free, sigma, dtype=precond_dtype, **pc_opts)
    else:
        # k == 0: rigid_modes found none (per face-connected part), and solve_modes passes no
        # s_K, so A_P(0) is nonsingular
        def build(s):
            return amg(op, free, s, mesh.points, nonsingular=k == 0, **pc_opts)

        if sigma is None:
            sigma, its = _precond_sigma(op, free, Rd, build, seed=seed, **sg_opts)
        P = build(sigma)
    t_p = time.perf_counter() - t0
    lam, V, info = lobpcg(
        op, free, n_modes, tol, Rd, P, X0, maxit=maxit, seed=seed, chunk=chunk,
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
