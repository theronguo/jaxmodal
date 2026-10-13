"""solve_modes: the public modal-analysis entry point. See docs/eigensolvers.md."""

import dataclasses
import math
import time
import warnings

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from ..operators import dof_mask, m_orthonormalize, rigid_modes, stencil, unstructured
from ..operators.common import is_count
from ..precond import SingularShiftError, amg, gmg, jacobi
from .counting import count, count_degree
from .lobpcg import _solve, band_defaults, lobpcg, sigma_estimate


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
COUNT_OPTIONS = ("n_probes", "seed", "tau", "degree", "block", "maxit")
# Builder parameters solve_modes sets itself or through its own keywords (dtype: precond_dtype).
RESERVED = {
    "unstructured": {"mesh", "materials", "orientations", "mass", "nodal_mass"},
    "stencil": {"mesh", "materials", "orientations", "mass", "nodal_mass"},
    "gmg": {"op", "free", "sigma", "dtype"},
    "amg": {"op", "free", "sigma", "points", "nonsingular", "aggregates", "return_aggregates"},
    "jacobi": {"op", "free", "sigma"},
    "count": {"op", "free", "W", "lam", "lam_last", "lam_next", "R", "precond"},
}


def _check_options(name, opts, allowed):
    """opts as a dict, or ValueError naming the unknown keys and the allowed ones."""
    opts = dict(opts or {})
    bad = sorted(set(opts) - set(allowed))
    if bad:
        raise ValueError(
            f"unknown {name} option(s) {bad}; allowed: {sorted(allowed)}. solve_modes sets "
            f"{sorted(set().union(*RESERVED.values()))} itself or through its own keywords "
            "(dtype: precond_dtype)"
        )
    return opts


def _check_sigma_options(opts):
    for key in ("maxit", "rounds", "estimate_maxit"):
        if key in opts and not is_count(opts[key]):
            raise ValueError(f"sigma_options {key} must be an int >= 1, got {opts[key]!r}")
    if "tol" in opts and not opts["tol"] > 0:
        raise ValueError(f"sigma_options tol must be > 0, got {opts['tol']!r}")
    if "drop" in opts and not opts["drop"] > 1:
        raise ValueError(f"sigma_options drop must be > 1, got {opts['drop']!r}")


def _check_count_options(opts):
    for key in ("n_probes", "maxit", "degree", "block"):
        if key in opts and not is_count(opts[key]):
            raise ValueError(f"count_options {key} must be an int >= 1, got {opts[key]!r}")
    if "tau" in opts and not opts["tau"] > 0:
        raise ValueError(f"count_options tau must be > 0, got {opts['tau']!r}")


def check_band(band):
    """(0, f_max) as floats, or the error: interior bands (f1 > 0) are PR 15."""
    try:
        f1, f2 = map(float, band)
    except (TypeError, ValueError):
        raise ValueError(f"band must be a pair (0, f_max), got {band!r}") from None
    if not (math.isfinite(f1) and math.isfinite(f2)):
        raise ValueError(f"band must be finite, got {band!r}")
    if f1 > 0:
        raise NotImplementedError("interior bands (f1 > 0) come with PR 15; pass band=(0, f_max)")
    if f1 != 0 or not f2 > 0:
        raise ValueError(f"band must be (0, f_max) with f_max > 0, got {band!r}")
    return f1, f2


def _nbytes(x):
    """Array bytes held by x (arrays, sparse matrices, dataclasses, tuples), for the budget."""
    if hasattr(x, "nbytes") and hasattr(x, "shape"):
        return int(x.nbytes)
    if sp.issparse(x):
        return sum(_nbytes(getattr(x, a, None)) for a in ("data", "indices", "indptr"))
    if dataclasses.is_dataclass(x):
        return sum(_nbytes(getattr(x, f.name)) for f in dataclasses.fields(x))
    if isinstance(x, (tuple, list)):
        return sum(_nbytes(v) for v in x)
    return 0


def _bound(steps, b, m, t_iteration, tau, rule):
    """A-priori lower bound on the count's seconds: a step runs >= 7 PCG iterations at width
    b, a LOBPCG iteration costs at most about 3 of their applies at width m (factor 2; 0.3, one
    PCG iteration, when tau is looser than the rule). None without a warm iteration time."""
    if not t_iteration:
        return None
    return (0.3 if tau is not None and tau > rule else 2.0) * steps * b / m * t_iteration


def _calibrate(run, d, n_blocks):
    """(predicted seconds of the count, calibration seconds, t_step, t_block): run(s) counts one
    block with at most s steps; s = 1 compiles, s = 3 is timed."""
    t = time.perf_counter()
    run(1)
    tm = run(3)[1]["time"]
    t_step, t_block = tm["step"], tm["block"]
    return n_blocks * (d * t_step + t_block), time.perf_counter() - t, t_step, t_block


def _verify(verify, opts, op, free, R, lam, V, n_band, lam_stop, m, t_it, solve_s, build, budget):
    """info["count"] of a band solve: the predicted counting cost, and the count itself unless
    verify is False or (verify None) predicted to cost more than the solve. budget() -> the
    probe block (or a ValueError: the count does not fit the solve's memory); build(lam_c) -> the
    preconditioner at lam_c."""
    out = dict(ran=False, reason=None, predicted_steps=None, predicted_s=None,
               predicted_kind="unknown", solve_s=solve_s, build_s=0.0,
               calibration_s=0.0)  # fmt: skip
    lam_last = lam[n_band - 1] if n_band else lam_stop
    lam_next = lam[n_band]
    lam_c, n_free = (lam_last + lam_next) / 2, int(free.sum())
    gap = (lam_next - lam_last) / lam_c
    n_rigid = 0 if R is None else R.shape[1]
    d = opts.get("degree") or count_degree(gap, lam.max() / lam_c, len(lam) + n_rigid, n_free)[0]
    n_probes = opts.get("n_probes", 50)
    out.update(lam_c=lam_c, gap=gap, degree=d)
    try:
        b = min(opts["block"], n_probes) if "block" in opts else budget(n_probes)
    except ValueError as e:
        out["reason"] = (f"count does not fit the solve's memory ({e}); pass "
                         "count_options={'block': 1} to run it with a larger peak")  # fmt: skip
        if verify:
            warnings.warn(out["reason"], stacklevel=3)
        if verify is not False:
            return out
        b = None
    out["block"] = b
    if d is not None and b is not None:
        n_blocks = -(-n_probes // b)
        steps = n_blocks * d
        bound = _bound(steps, b, m, t_it, opts.get("tau"), 0.01 / (d * math.sqrt(n_free)))
        out.update(predicted_steps=steps, predicted_s=bound,
                   predicted_kind="unknown" if bound is None else "bound")  # fmt: skip
    if verify is False:
        out["reason"] = out["reason"] or "verify=False"
        return out
    args = (op, free, V, lam, lam_last, lam_next)
    if d is None and verify is None:
        out["reason"] = "degree above MAX_DEGREE: the gap is too small to count"
        return out
    if verify is None and (out["predicted_s"] or 0) > solve_s:
        out["reason"] = "predicted cost (a lower bound) above the solve"
        return out
    P = None  # d None: count returns NaN, unverifiable, before using it
    if d is not None:
        t = time.perf_counter()
        P = build(lam_c)
        out["build_s"] = time.perf_counter() - t
    kw = {**opts, "R": R, "precond": P, "block": b}
    if verify is None:
        pred, cal, t_step, t_block = _calibrate(
            lambda s: count(*args, **{**kw, "n_probes": b, "_steps": s}), d, n_blocks
        )
        pred += out["build_s"] + cal
        out.update(predicted_s=pred, predicted_kind="calibrated", calibration_s=cal,
                   calibrated_step=t_step, calibrated_block=t_block)  # fmt: skip
        if pred > solve_s:
            out["reason"] = "predicted cost above the solve"
            return out
    est, ci = count(*args, **kw)
    return {**out, **ci, "estimate": est, "ran": True}


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
    mesh, materials, bcs, n_modes=None, orientations=None, *, mass="consistent",
    nodal_mass=None, tol=1e-6, maxit=500, seed=0, chunk=None, guards=None,
    basis_dtype="float64", low_memory=False, precond="auto", sigma=None,
    precond_dtype="float64", X0=None, operator_options=None, precond_options=None,
    sigma_options=None, band=None, verify=None, count_options=None,
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
    (sqrt(lam) / 2 pi), precond, sigma, sigma_iterations and time["precond"].

    band=(0, f_max) (Hz) instead returns every elastic mode with f <= f_max (and whole
    clusters reaching into the band), possibly none: solve_modes(mesh, materials, bcs,
    band=(0, f_max)). n_modes is then the expected count (the basis capacity; default 4 chunk,
    chunk default min(40, ceil(1.25 n_modes) + 2)). verify: True runs the completeness count at
    the cut after the band, False only predicts its cost, None (default) runs it when predicted
    to cost no more than the solve; count_options (COUNT_OPTIONS) go to count. info adds band,
    n_beyond, chunk, capacity and count (docs/eigensolvers.md#frequency-band-0-f_max)."""
    if band is None:
        if n_modes is None:
            raise ValueError("pass n_modes or band")
        if verify is not None or count_options:
            raise ValueError("verify and count_options need band")
    else:
        f_max = check_band(band)[1]
    if verify not in (None, True, False):
        raise ValueError(f"verify must be None, True or False, got {verify!r}")
    cn_opts = _check_options("count", count_options, COUNT_OPTIONS)
    _check_count_options(cn_opts)
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
    aggs, R = None, jnp.asarray(rigid_modes(mesh, free))
    k = R.shape[1]
    if k:
        R = m_orthonormalize(R, op.apply_M(R))  # consumes its first argument
    Rd = R if k else None
    its = 0

    def make(s, **kw):  # the preconditioner at shift s (fp64 unless kw says otherwise)
        if precond == "jacobi":
            return jacobi(op, free, s)
        if precond == "gmg":
            return gmg(op, free, s, **pc_opts, **kw)
        # k == 0: rigid_modes found none (per face-connected part), and solve_modes passes no
        # s_K, so A_P(0) is nonsingular
        return amg(op, free, s, mesh.points, nonsingular=k == 0, **pc_opts, **kw)

    if sigma is None and precond != "jacobi":
        sigma, its = _precond_sigma(op, free, Rd, make, seed=seed, **sg_opts)
    sigma = sigma or 0.0
    if precond == "gmg":
        P = make(sigma, dtype=precond_dtype)
    elif precond == "amg" and band is not None:  # the aggregates are reused at lam_c (exact)
        P, aggs = make(sigma, return_aggregates=True)
    else:
        P = make(sigma)
    t_p = time.perf_counter() - t0
    if band is None:
        lam, V, info = lobpcg(
            op, free, n_modes, tol, Rd, P, X0, maxit=maxit, seed=seed, chunk=chunk,
            guards=guards, basis_dtype=basis_dtype, low_memory=low_memory,
        )  # fmt: skip
    else:
        n_modes, chunk = band_defaults(n_modes, chunk, guards, int(free.sum()) - k)
        lam_stop = (2 * np.pi * f_max) ** 2
        lam, V, info = _solve(
            op, free, n_modes, tol, Rd, P, X0, maxit, seed, chunk=chunk, guards=guards,
            basis_dtype=basis_dtype, low=low_memory, lam_stop=lam_stop,
        )  # fmt: skip
    solve_s = time.perf_counter() - t0
    P = None  # one hierarchy alive at a time
    info["time"]["precond"] = t_p
    info.update(operator=name, n_rigid=k, rigid=np.asarray(R), precond=precond, sigma=sigma,
                sigma_iterations=its)  # fmt: skip
    if band is not None:
        nb, n_eff, m = info.pop("n_band"), info.pop("n_modes"), info.pop("block")
        if not info["converged"]:
            info["count"] = dict(ran=False, reason="solve not converged")
            if verify:
                warnings.warn("band solve not converged: count not run", stacklevel=2)
        else:
            from ..memory import band_probe_block  # memory imports this module

            def budget(n_probes):  # n_eff: the expected count, or the one the grown basis serves
                return band_probe_block(
                    mesh, bcs, n_eff, n_probes, 8 * op.n_dof * len(lam) + _nbytes(aggs),
                    n_rigid=k, mass=mass, chunk=info["chunk"], guards=guards,
                    basis_dtype=basis_dtype, low_memory=low_memory, precond=precond,
                    precond_dtype=precond_dtype, operator_options=op_opts,
                    precond_options=pc_opts, band=band,
                )  # fmt: skip

            info["count"] = _verify(
                verify, cn_opts, op, free, Rd, lam, V, nb, lam_stop, m, info["t_iteration"],
                solve_s, lambda s: make(s) if aggs is None else make(s, aggregates=aggs), budget,
            )  # fmt: skip
        info.update(band=(0.0, f_max), n_beyond=len(lam) - nb, residuals=info["residuals"][:nb],
                    thresholds=info["thresholds"][:nb])  # fmt: skip
        lam, V = lam[:nb], V[:, :nb].copy()  # not a view of all converged
    info["frequencies"] = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
    return lam, V, info
