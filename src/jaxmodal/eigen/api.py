"""solve_modes: the public modal-analysis entry point. See docs/eigensolvers.md."""

import dataclasses
import functools
import gc
import math
import numbers
import time
import warnings

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from ..linalg import pcg
from ..operators import (
    ap_operator,
    dof_mask,
    m_orthonormalize,
    rigid_modes,
    stencil,
    unstructured,
)
from ..operators.common import is_count
from ..precond import SingularShiftError, amg, gmg, jacobi
from . import counting
from .counting import count, count_degree
from .lobpcg import (
    _band_count,
    _block_size,
    _capacity,
    _solve,
    band_defaults,
    lobpcg,
    sigma_estimate,
)
from .shift_invert import si_band


def resolve_precond(mesh, precond, precond_dtype):
    """The preconditioner name solve_modes uses for these options; raises on bad options."""
    voxel = mesh.voxel is not None
    # ssh8 with hex20 / hex27 (nonconforming): no solid-shell AMG
    mixed = "ssh8" in mesh.blocks and bool(set(mesh.blocks) - {"ssh8", "hex8"})
    if precond == "auto":
        precond = "gmg" if voxel else "jacobi" if mixed else "amg"
    if precond not in ("gmg", "amg", "jacobi"):
        raise ValueError(f"precond must be 'auto', 'gmg', 'amg' or 'jacobi', got {precond!r}")
    if precond == "amg" and mixed:
        raise ValueError("solid-shell AMG supports ssh8 and hex8 blocks")
    if precond == "gmg" and not voxel:
        raise ValueError("precond 'gmg' needs a voxel mesh (voxel_mesh)")
    if precond == "amg" and voxel:
        raise ValueError("precond 'amg' needs an unstructured mesh (voxel meshes: 'gmg')")
    if precond_dtype not in ("float64", "float32"):
        raise ValueError(f"precond_dtype must be 'float64' or 'float32', got {precond_dtype!r}")
    if precond != "gmg" and precond_dtype != "float64":
        raise ValueError("precond_dtype 'float32' needs precond 'gmg'")
    return precond


def precond_label(requested, name):
    """name for error messages, "auto (resolved to name)" when the caller asked for auto."""
    return f"auto (resolved to {name})" if requested == "auto" else name


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
POLISH_OPTIONS = ("on", "tol", "maxit")
COUNT_OPTIONS = ("n_probes", "seed", "tau", "degree", "block", "maxit", "verbose")
SLICE_OPTIONS = ("tau", "maxit_inner", "block", "max_slices", "capacity", "edges", "verbose")
STRATEGIES = ("auto", "lobpcg", "shift_invert")
MU1_MARGIN = 1e-2  # the lam_lo < mu_1 rule's margin: max(MU1_MARGIN, 300 tol)
STAGNATION_DEPTH = 1000  # locked modes at a capacity stop above which MINRES may stagnate
SLICE_MIN = 96  # PR 15c's measured slice minimum at block 6: a smaller capacity warns
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


def check_polish_options(opts, precond, label):
    """polish_options as a dict, or the error (AMG only: the polish is PCG with the V-cycle)."""
    opts = _check_options("polish", opts, POLISH_OPTIONS)
    if opts and precond != "amg":
        raise ValueError(f"polish_options need precond='amg' (precond: {label})")
    if "on" in opts and not isinstance(opts["on"], bool):
        raise ValueError(f"polish_options on must be a bool, got {opts['on']!r}")
    if "tol" in opts and not opts["tol"] > 0:
        raise ValueError(f"polish_options tol must be > 0, got {opts['tol']!r}")
    if "maxit" in opts and not is_count(opts["maxit"]):
        raise ValueError(f"polish_options maxit must be an int >= 1, got {opts['maxit']!r}")
    return opts


def _check_count_options(opts):
    for key in ("n_probes", "maxit", "degree", "block"):
        if key in opts and not is_count(opts[key]):
            raise ValueError(f"count_options {key} must be an int >= 1, got {opts[key]!r}")
    if "tau" in opts and not opts["tau"] > 0:
        raise ValueError(f"count_options tau must be > 0, got {opts['tau']!r}")
    if "verbose" in opts and not isinstance(opts["verbose"], bool):
        raise ValueError(f"count_options verbose must be a bool, got {opts['verbose']!r}")


def check_band(band):
    """(f1, f2) as floats with 0 <= f1 < f2, finite, or the error."""
    try:
        f1, f2 = map(float, band)
    except (TypeError, ValueError):
        raise ValueError(f"band must be a pair (f1, f2), got {band!r}") from None
    if not (math.isfinite(f1) and math.isfinite(f2)):
        raise ValueError(f"band must be finite, got {band!r}")
    if not f1 >= 0:
        raise ValueError(f"band needs f1 >= 0, got {band!r}")
    if not f1 < f2:
        raise ValueError(f"band needs f1 < f2, got {band!r}")
    return f1, f2


def _check_slice_options(opts):
    for key in ("maxit_inner", "block", "max_slices", "capacity"):
        if key in opts and not is_count(opts[key]):
            raise ValueError(f"slice_options {key} must be an int >= 1, got {opts[key]!r}")
        if key in opts:
            opts[key] = int(opts[key])  # si_band takes Python ints only
    if "tau" in opts and not opts["tau"] > 0:
        raise ValueError(f"slice_options tau must be > 0, got {opts['tau']!r}")
    if "verbose" in opts and not isinstance(opts["verbose"], bool):
        raise ValueError(f"slice_options verbose must be a bool, got {opts['verbose']!r}")
    if opts.get("capacity", math.inf) < 2 * opts.get("block", 6):
        raise ValueError(f"slice_options capacity must be >= 2 block, got {opts['capacity']}")
    try:
        edges = [float(x) for x in opts.get("edges") or []]
    except (TypeError, ValueError):
        edges = None
    if edges is None or edges != sorted(set(edges)) or any(not x > 0 for x in edges):
        raise ValueError(
            f"slice_options edges must be sorted, distinct and > 0, got {opts['edges']!r}"
        )


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


def _verify(
    verify, opts, op, free, R, lam, V, n_band, lam_stop, m, t_it, solve_s, build, probe_block
):
    """info["count"] of a band solve: the predicted counting cost, and the count itself unless
    verify is False or (verify None) predicted to cost more than the solve. probe_block() -> the
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
        b = min(opts["block"], n_probes) if "block" in opts else probe_block(n_probes)
    except ValueError as e:
        out["reason"] = (f"count does not fit the solve's memory ({e}); pass "
                         "count_options={'block': 1} to run it with a larger peak")  # fmt: skip
        if verify:
            warnings.warn(out["reason"], stacklevel=3)
        if verify is not False:
            return out
        b = None
    out["block"] = b
    if b is None:  # verify False: nothing predicted without a block
        return out
    P = None  # built by prepare (verify None, the bound passed) or below; skips never build

    def prepare():
        nonlocal P
        P = build(lam_c)

    args = (op, free, V, lam, lam_last, lam_next)
    kw = {**opts, "R": R, "block": b}
    run = lambda s: count(*args, **{**kw, "precond": P, "n_probes": b, "_steps": s})  # noqa: E731
    go, rec = counting._decide(verify, d, n_probes, b, m, solve_s, t_it, opts.get("tau"), n_free,
                               run, prepare)  # fmt: skip
    out.update(rec)
    if not go:
        out["reason"] = rec["reason"] or "verify=False"
        return out
    if P is None and d is not None:  # d None: count returns NaN, unverifiable, before using it
        t = time.perf_counter()
        P = build(lam_c)
        out["build_s"] = time.perf_counter() - t
    est, ci = count(*args, **kw, precond=P)
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


def _halvings(top):
    """The ladder's chunks: top, then halved while >= 5."""
    out, c = [top], top // 2
    while c >= 5:
        out.append(c)
        c //= 2
    return out


def _ladder(reach, need):
    """The ladder's chunk from reach = {chunk: C_max or None (does not fit)} and need = {chunk:
    the columns of the known depth} or None: the largest fitting chunk reaching its need, else the
    largest whose C_max is >= 0.9 x the best; None when no chunk fits."""
    fit = {c: r for c, r in reach.items() if r is not None}
    if not fit:
        return None
    ok = [c for c in fit if need and fit[c] >= need[c]]
    if ok:
        return max(ok)
    best = max(fit.values())
    return max(c for c, r in fit.items() if r >= 0.9 * best)


def _keep(lam, t, lam_lo, n_band):
    """i0: the band filter's range [i0, n_band) of the ascending lam (thresholds t, per mode or one
    value): from the first mode >= lam_lo (1 - 100 t) (len(lam) if none), extended down over
    chains (relative gaps < 100 t) below n_band; i0 = 0 for lam_lo = 0."""
    if lam_lo == 0:
        return 0
    t = np.broadcast_to(np.asarray(t, float), np.shape(lam))
    ok = np.flatnonzero(lam >= lam_lo * (1 - 100 * t))
    i0 = int(ok[0]) if len(ok) else len(lam)
    while 0 < i0 < n_band and lam[i0] - lam[i0 - 1] < 100 * t[i0 - 1] * lam[i0 - 1]:
        i0 -= 1
    return min(i0, n_band)


def _permute(V, order):
    """V[:, :len(order)] = V[:, order] in place, along the permutation's cycles with one column
    buffer (fixed columns untouched; V[:, order] would copy, np.take loses the F order)."""
    done = order == np.arange(len(order))
    for s in range(len(order)):
        if done[s]:
            continue
        tmp, j = V[:, s].copy(), s
        while order[j] != s:
            V[:, j] = V[:, order[j]]
            done[j], j = True, order[j]
        V[:, j], done[j] = tmp, True


def solve_modes(
    mesh, materials, bcs, n_modes=None, orientations=None, *, mass="consistent",
    nodal_mass=None, tol=1e-6, maxit=500, seed=0, chunk=None, guards=None,
    basis_dtype="float64", low_memory=False, precond="auto", sigma=None,
    precond_dtype="float64", X0=None, operator_options=None, precond_options=None,
    sigma_options=None, band=None, verify=None, count_options=None, strategy="auto",
    memory_budget=None, slice_options=None, polish_options=None,
):  # fmt: skip
    """Lowest n_modes elastic modes of mesh under the Dirichlet sets bcs (see dof_mask).

    Voxel meshes get the stencil operator, all others the unstructured one (both run the
    material and element checks first). The rigid modes left by bcs are deflated and returned
    in info["rigid"] (M-orthonormal). precond: "auto" (GMG on voxel meshes, Jacobi with ssh8
    and hex20 / hex27 blocks, else AMG), "gmg", "amg" or "jacobi". The multigrid is built at
    sigma (a hint >= 0) or at the refined lam_1 pre-estimate; precond_dtype "float32" runs the
    GMG V-cycle in fp32. X0 (n_dof, j <= block size) seeds lobpcg's first chunk.
    operator_options, precond_options and sigma_options are dicts forwarded to the operator
    builder, the preconditioner builder and the pre-estimate (allowed keys: OPERATOR_OPTIONS,
    PRECOND_OPTIONS, SIGMA_OPTIONS; docs/eigensolvers.md#options). polish_options
    (POLISH_OPTIONS, AMG only): the polish of a stalled chunk, one PCG solve with the V-cycle
    (on, default True; tol, default min(1e-7, 0.1 tol); maxit, default 500).
    Returns lam (ascending), V (n_dof, n_modes; with band one column per band mode) and
    lobpcg's info plus operator, n_rigid, rigid, frequencies (sqrt(lam) / 2 pi), precond,
    sigma, sigma_iterations and time["precond"].

    band=(f1, f2) (Hz, 0 <= f1 < f2) instead returns every elastic mode with f1 <= f <= f2 (and
    whole clusters reaching into the band), possibly none: solve_modes(mesh, materials, bcs,
    band=(f1, f2)). strategy: "auto" (LOBPCG from 0 with its basis capped by memory_budget,
    falling back to shift-invert slices at the cap or when no chunk fits), "lobpcg" (raises
    instead of falling back) or "shift_invert" (si_band). memory_budget: bytes (default the free
    GPU pool, or 0.8 x RAM on the CPU). slice_options (SLICE_OPTIONS) go to si_band. n_modes is
    the expected count (f1 = 0: the basis and the ladder's top chunk, chunk default min(40,
    ceil(1.25 n_modes) + 2); f1 > 0: the slice store, chunk default 40). verify: True runs the
    completeness count(s), False only predicts the cost, None (default) counts when predicted to
    cost no more than the solve; count_options (COUNT_OPTIONS) go to count. V is the solver's
    own array compacted in place. info adds band, strategy, reason, chunk, c_max, budget, mu1_ub,
    n_store, from_zero, stagnation_risk, count, and LOBPCG's n_beyond, capacity, t_iteration
    or si_band's slices (docs/eigensolvers.md#interior-bands-and-strategy)."""
    if band is None:
        if n_modes is None:
            raise ValueError("pass n_modes or band")
        if verify is not None or count_options:
            raise ValueError("verify and count_options need band")
        if strategy != "auto" or memory_budget is not None or slice_options:
            raise ValueError("strategy, memory_budget and slice_options need band")
    else:
        f1, f2 = check_band(band)
        for name, x in (("n_modes", n_modes), ("chunk", chunk)):
            if x is not None and not is_count(x):
                raise ValueError(f"{name} must be an int >= 1, got {x!r}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if memory_budget is not None and not (
        isinstance(memory_budget, numbers.Real) and not isinstance(memory_budget, bool)
        and math.isfinite(memory_budget) and memory_budget > 0
    ):  # fmt: skip
        raise ValueError(
            f"memory_budget must be a finite number of bytes > 0, got {memory_budget!r}"
        )
    if verify not in (None, True, False):
        raise ValueError(f"verify must be None, True or False, got {verify!r}")
    cn_opts = _check_options("count", count_options, COUNT_OPTIONS)
    _check_count_options(cn_opts)
    sl_opts = _check_options("slice", slice_options, SLICE_OPTIONS)
    _check_slice_options(sl_opts)
    requested, precond = precond, resolve_precond(mesh, precond, precond_dtype)
    label = precond_label(requested, precond)
    voxel = mesh.voxel is not None
    name, build = ("stencil", stencil) if voxel else ("unstructured", unstructured)
    op_opts = _check_options("operator", operator_options, OPERATOR_OPTIONS[name])
    pc_opts = _check_options(f"{label} precond", precond_options, PRECOND_OPTIONS[precond])
    sg_opts = _check_options("sigma", sigma_options, SIGMA_OPTIONS)
    if sg_opts and (sigma is not None or precond == "jacobi"):
        raise ValueError(
            f"sigma_options need the pre-estimate: no sigma= and no precond 'jacobi' (precond: "
            f"{label})"
        )
    _check_sigma_options(sg_opts)
    po_opts = check_polish_options(polish_options, precond, label)
    if sigma is not None:
        if not (math.isfinite(sigma) and sigma >= 0):
            raise ValueError(f"sigma must be a finite float >= 0, got {sigma}")
        sigma = float(sigma)
    free = dof_mask(mesh, bcs)  # before the build: bad bcs fail fast
    from .. import memory as mem  # memory imports this module

    # before the build: a GPU pool's bytes_in_use would count the operator a second time
    budget, host_budget = mem._default_budget() if band is not None else (None, None)
    op = build(mesh, materials, orientations, mass=mass, nodal_mass=nodal_mass, **op_opts)
    t0 = time.perf_counter()
    aggs, R = None, jnp.asarray(rigid_modes(mesh, free))
    k = R.shape[1]
    if k:
        R = m_orthonormalize(R, op.apply_M(R))  # consumes its first argument
    Rd = R if k else None
    its = 0
    base = dict(operator=name, n_rigid=k, rigid=np.asarray(R), precond=precond)

    def make(s, **kw):  # the preconditioner at shift s (fp64 unless kw says otherwise)
        if precond == "jacobi":
            return jacobi(op, free, s)
        if precond == "gmg":
            return gmg(op, free, s, **pc_opts, **kw)
        # k == 0: rigid_modes found none (per face-connected part), and solve_modes passes no
        # s_K, so A_P(0) is nonsingular
        return amg(op, free, s, mesh.points, nonsingular=k == 0, **pc_opts, **kw)

    def polish(P, s):  # LOBPCG's polish closure: PCG on A_P(s) with its preconditioner P
        if precond != "amg" or not po_opts.get("on", True):
            return None
        A, mf = ap_operator(op, free, s), jnp.asarray(free, jnp.float64)[:, None]
        ptol, pmax = po_opts.get("tol", min(1e-7, 0.1 * tol)), po_opts.get("maxit", 500)

        def run(X):
            Y, inf = pcg(A, mf * op.apply_M(mf * X), P, ptol, pmax)
            return Y, int(inf["iterations"].max())

        return run

    def lobpcg_precond(s):  # LOBPCG's preconditioner (and AMG's aggregates in band mode)
        if precond == "gmg":
            return make(s, dtype=precond_dtype), None
        if precond == "amg" and band is not None:  # the aggregates are reused at lam_c (exact)
            return make(s, return_aggregates=True)
        return make(s), None

    if band is None:
        if sigma is None and precond != "jacobi":
            sigma, its = _precond_sigma(op, free, Rd, make, seed=seed, **sg_opts)
        sigma = sigma or 0.0
        P = lobpcg_precond(sigma)[0]
        t_p = time.perf_counter() - t0
        lam, V, info = lobpcg(
            op, free, n_modes, tol, Rd, P, X0, maxit=maxit, seed=seed, chunk=chunk,
            guards=guards, basis_dtype=basis_dtype, low_memory=low_memory,
            polish=polish(P, sigma),
        )  # fmt: skip
        P = None
        info["time"]["precond"] = t_p
        info.update(base, sigma=sigma, sigma_iterations=its)
        info["frequencies"] = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
        return lam, V, info

    # ---------------------------------------------------------------- a band (f1, f2)
    lam_lo, lam_hi = (2 * np.pi * f1) ** 2, (2 * np.pi * f2) ** 2
    n, n_avail = op.n_dof, int(free.sum()) - k
    if memory_budget is not None:
        budget = float(memory_budget)

    def models(dtype, ops=op_opts):  # the estimators, cached per call (distinct widths cost)
        return [f and functools.cache(f) for f in
                mem._models(mesh, mass, precond, dtype, ops, pc_opts, po_opts)]  # fmt: skip

    op_memory, pc = models(precond_dtype)
    pc64 = models("float64")[1] if precond == "gmg" and precond_dtype == "float32" else pc

    def c_max(c, low=low_memory, dtype=basis_dtype, op_m=None, pc_m=None):
        return mem._c_max(op_m or op_memory, pc_m or pc, n, chunk=c, guards=guards, low=low,
                          basis_dtype=dtype, n_rigid=k, n_avail=n_avail, budget=budget,
                          host_budget=host_budget, need=need and need.get(c))  # fmt: skip

    def levers(c):  # the memory levers not set, with their C_max at chunk c
        out = [f"low_memory=True (C_max {c_max(c, low=True)})"] if not low_memory else []
        if basis_dtype != "float32":
            out.append(f"basis_dtype='float32' (C_max {c_max(c, dtype='float32')})")
        if precond == "gmg" and precond_dtype != "float32":
            out.append(f"precond_dtype='float32' (C_max {c_max(c, pc_m=models('float32')[1])})")
        if not voxel and "ssh8" not in mesh.blocks and op_opts.get("mode") != "otf":
            otf = models(precond_dtype, op_opts | {"mode": "otf"})[0]
            out.append(f"operator_options={{'mode': 'otf'}} (C_max {c_max(c, op_m=otf)})")
        return ", ".join(out)

    reach, ch, need = {}, None, None
    if strategy != "shift_invert":
        top = band_defaults(n_modes if f1 == 0 else None, chunk, guards, n_avail)[1]
        chunks = [chunk] if chunk is not None else _halvings(top)
        if f1 == 0 and n_modes is not None:  # the known depth: n_modes, at most the free space
            need = {}
            for c in chunks:
                kc = _block_size(c, c, guards)[2]
                cl = _capacity(0, c, kc, low_memory)[0]
                need[c] = min(math.ceil(1.25 * n_modes) + kc, -(-(n_avail + kc) // cl) * cl)
        reach = {c: c_max(c) for c in chunks}
        ch = _ladder(reach, need)
        if ch is None and strategy == "lobpcg":
            raise ValueError(
                f"no LOBPCG chunk fits memory_budget = {budget:.4g} bytes (C_max {reach}): raise"
                f" memory_budget, or pass {levers(min(reach))}, or strategy='shift_invert'"
            )
    out = dict(band=(f1, f2), strategy="lobpcg", reason="forced" if strategy != "auto" else "fits",
               chunk=ch, c_max=reach, stagnation_risk=False, budget=(budget, host_budget),
               mu1_ub=None, n_store=None, from_zero=False)  # fmt: skip
    # the sigma pre-estimate, the in-band locked count (sizes n_store), LOBPCG's time, its stop
    pre, L, t_lob, stopped = None, 0, None, None
    if ch is not None:
        if sigma is None and precond != "jacobi":
            sigma, its = _precond_sigma(op, free, Rd, make, seed=seed, **sg_opts)
            pre = sigma
        sigma = sigma or 0.0
        P, aggs = lobpcg_precond(sigma)
        t_p = time.perf_counter() - t0
        n_m = band_defaults(n_modes if f1 == 0 else None, ch, guards, n_avail)[0]
        lam, V, info = _solve(
            op, free, n_m, tol, Rd, P, X0, maxit, seed, chunk=ch, guards=guards,
            basis_dtype=basis_dtype, low=low_memory, lam_stop=lam_hi, max_capacity=reach[ch],
            polish=polish(P, sigma),
        )  # fmt: skip
        solve_s = time.perf_counter() - t0
        P = None  # one hierarchy alive at a time
        stopped = info.get("stopped")
        if (np.diff(lam) < 0).any():  # a mode an earlier chunk missed, back in a later one
            order = np.argsort(lam, kind="stable")
            lam = lam[order]
            info.update(residuals=info["residuals"][order], thresholds=info["thresholds"][order])
            if stopped is None:  # after a capacity stop V is dropped unsorted
                _permute(V, order)
                info["n_band"] = _band_count(lam, info["thresholds"], lam_hi)
        if stopped == "capacity":
            depth = len(lam)
            if strategy == "lobpcg":
                raise ValueError(
                    f"LOBPCG reached its basis cap C_max = {reach[ch]} at {depth} locked modes"
                    f" below f2: raise memory_budget (now {budget:.4g} bytes), or pass"
                    f" {levers(ch)}, or strategy='shift_invert' or 'auto'"
                )  # fmt: skip
            L = depth - _keep(lam, info["thresholds"], lam_lo, depth)
            out["mu1_ub"] = min(float(lam[0]), math.inf if pre is None else pre)
            t_lob = info["time"]
            V = lam = info = aggs = None  # LOBPCG's modes and aggregates freed before the slices
            risk = depth >= STAGNATION_DEPTH
            out.update(reason="capacity", stagnation_risk=risk)
            warnings.warn(
                f"LOBPCG stopped at its basis cap C_max = {reach[ch]} with {depth} modes locked"
                f" below f2: the band continues with shift-invert slices"
                + (" (MINRES may stagnate at this depth)" if risk else "")
                + f"; levers: {levers(ch)}", stacklevel=2,
            )  # fmt: skip
    if ch is not None and stopped != "capacity":  # LOBPCG reached the band's end
        nb, n_eff, m = info.pop("n_band"), info.pop("n_modes"), info.pop("block")
        if not info["converged"]:
            info["count"] = dict(ran=False, reason="solve not converged")
            if verify:
                warnings.warn("band solve not converged: count not run", stacklevel=2)
        elif nb == len(lam):  # a sort moved the cut to the end: no mode above it bounds it
            info.update(converged=False, stopped="sorted cut at the end")
            info["count"] = dict(ran=False, reason="sorted cut at the end")
        else:

            def probe_budget(n_probes):  # n_eff: the count the grown basis serves
                return mem.band_probe_block(
                    mesh, bcs, n_eff, n_probes, 8 * n * len(lam) + _nbytes(aggs),
                    n_rigid=k, mass=mass, chunk=info["chunk"], guards=guards,
                    basis_dtype=basis_dtype, low_memory=low_memory, precond=precond,
                    precond_dtype=precond_dtype, operator_options=op_opts,
                    precond_options=pc_opts, band=(0.0, f2),
                )  # fmt: skip

            info["count"] = _verify(
                verify, cn_opts, op, free, Rd, lam, V[:, : len(lam)], nb, lam_hi, m,
                info["t_iteration"], solve_s,
                lambda s: make(s) if aggs is None else make(s, aggregates=aggs), probe_budget,
            )  # fmt: skip
        aggs = None
        i0, i1 = _keep(lam, info["thresholds"], lam_lo, nb), nb
        info["n_beyond"] = len(lam) - nb
        info["time"]["precond"] = t_p
        info.update({**base, **out}, sigma=sigma, sigma_iterations=its)
    else:  # shift-invert slices: forced, no chunk fits ("memory") or after a capacity stop
        if ch is None:
            t_p = 0.0
            if strategy == "auto":
                out["reason"] = "memory"
                warnings.warn(
                    f"no LOBPCG chunk fits memory_budget = {budget:.4g} bytes: the band is solved"
                    f" by shift-invert slices; levers: {levers(min(reach))}", stacklevel=2,
                )  # fmt: skip
        out["strategy"] = "shift_invert"
        if f1 > 0 and out["mu1_ub"] is None:  # no LOBPCG bound: the pre-estimate
            t = time.perf_counter()
            if precond == "jacobi":
                pre, its = sigma_estimate(op, free, Rd, seed=seed)
            else:
                pre, its = _precond_sigma(op, free, Rd, make, seed=seed, **sg_opts)
            sigma, out["mu1_ub"], t_p = pre, pre, time.perf_counter() - t
        out["from_zero"] = f1 == 0 or lam_lo < out["mu1_ub"] * (1 + max(MU1_MARGIN, 300 * tol))
        n_store = max(n_modes or 200, math.ceil(1.25 * L))
        b = sl_opts.get("block", 6)
        C = sl_opts.get("capacity")
        if C is None:
            C = mem._slice_capacity(op_memory, pc64, n, n_store=n_store, block=b, n_rigid=k,
                                    budget=budget, host_budget=host_budget)  # fmt: skip
            if C is None:
                raise ValueError(
                    f"the smallest shift-invert slice does not fit memory_budget = {budget:.4g}"
                    " bytes: raise memory_budget"
                )
            if C < SLICE_MIN:
                warnings.warn(
                    f"slice capacity {C} is below {SLICE_MIN} columns (memory_budget"
                    f" {budget:.4g} bytes): slices may fill without converging", stacklevel=2,
                )  # fmt: skip
        v_si, cn, no_count = verify, cn_opts, None
        if "block" not in cn_opts:
            bp = mem._count_block(
                op_memory, pc64, n, voxel=voxel, n_rigid=k, n_store=n_store,
                window=2 * (C - 2 * b), n_probes=cn_opts.get("n_probes", 50), budget=budget,
                host_budget=host_budget,
            )  # fmt: skip
            if bp is None:  # one probe does not fit beside the store and window
                no_count = (
                    "the slice counts do not fit memory_budget (one probe); pass"
                    " count_options={'block': 1} to run them with a larger peak"
                )
                if verify:
                    warnings.warn(f"{no_count}: no counts run", stacklevel=2)
                v_si = False
            else:
                cn = {**cn_opts, "block": bp}
        lam, V, info = si_band(
            op, free, 0.0 if out["from_zero"] else lam_lo, lam_hi, make, capacity=C, R=Rd,
            verify=v_si, count_options=cn, tol=tol, seed=seed, n_store=n_store,
            **{key: x for key, x in sl_opts.items() if key != "capacity"},
        )  # fmt: skip
        if no_count and verify is not False:
            info["count"]["reason"] = no_count
        info["slice_reason"] = info.pop("reason")  # si_band's; reason is the strategy's
        i0, i1 = _keep(lam, info["r_acc"], lam_lo, len(lam)), len(lam)
        info["time"]["precond"] = t_p
        if t_lob is not None:
            info["time"]["lobpcg"] = t_lob
        info.update({**base, **out}, capacity=C, n_store=n_store, sigma=sigma, sigma_iterations=its)
        if stopped:
            info["stopped"] = stopped
    info.update(residuals=info["residuals"][i0:i1], thresholds=info["thresholds"][i0:i1])
    # compacted in place (the destination never runs ahead of the source), then shrunk: V must
    # have no other reference. jaxlib holds the last host view it transferred (the count's W
    # chunks) until a collection, so no jax op runs between _verify and the resize
    if i0:
        for j in range(i1 - i0):
            V[:, j] = V[:, i0 + j]
    gc.collect()
    try:
        V.resize((V.shape[0], i1 - i0))
    except ValueError:  # another holder now sees shifted columns
        warnings.warn("returned modes copied: another reference to V exists", RuntimeWarning,
                      stacklevel=2)  # fmt: skip
        V = np.array(V[:, : i1 - i0], order="F")
    lam = lam[i0:i1]
    info["frequencies"] = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
    return lam, V, info
