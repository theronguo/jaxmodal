"""AMG study: PCG iterations of amg against pmg's exact coarse solve (h-sequences, robustness,
element aspect), the cost at hex20 44^3 (about 1M DOF): t_A, the AMG cycle, the transfers and
the whole V-cycle in t_A, V-cycle temporaries, setup times, device storage and level sizes; the
setup times and host peaks per size (fresh subprocesses), the SIMP study (AMG against GMG on
voxel density fields), the solve_modes acceptance run (hex20 44^3, 20 modes) and the optional
10M run (--large, by hand).

Cases, shifts and PCG settings as in pmg_study (unit-length steel boxes, xmin clamped unless
"free", 4 random masked right-hand sides, relative residual 1e-8, Chebyshev degree 2, ratio 30 on
every level). max_coarse = 100 in the iteration runs, so that every size has an AMG level; the
cost run uses the default 1000. Times include compilation unless "best" (min of 5 runs).
Writes results/amg_study.json.
"""

import argparse
import importlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import gmg_study, harness
from benchmarks.pmg_study import best, case, lam1
from jaxmodal import config
from jaxmodal.linalg import pcg
from jaxmodal.memory import solve_modes_memory
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import ap_operator, unstructured
from jaxmodal.precond import amg, pmg
from jaxmodal.precond.gmg import GMG, _vcycle_apply

AM = importlib.import_module("jaxmodal.precond.amg")
PM = importlib.import_module("jaxmodal.precond.pmg")

TOL = 1e-8
MAX_COARSE = 100
# name, type, shape, spacing factors, bcs, sigma source, extra (pmg_study's case)
CASES = [
    ("hex20 6^3", "hex20", (6,) * 3, None, "clamped", "own", None),
    *[(f"hex20 {n}^3", "hex20", (n,) * 3, None, "clamped", "hex20 6^3", None) for n in (3, 12, 24)],
    ("hex27 6^3", "hex27", (6,) * 3, None, "clamped", "own", None),
    *[(f"hex27 {n}^3", "hex27", (n,) * 3, None, "clamped", "hex27 6^3", None) for n in (3, 12)],
    ("hex8 6^3", "hex8", (6,) * 3, None, "clamped", "own", None),
    *[(f"hex8 {n}^3", "hex8", (n,) * 3, None, "clamped", "hex8 6^3", None) for n in (12, 24, 48)],
    ("beam 20x2x2", "hex20", (20, 2, 2), None, "clamped", "own", None),
    ("free 6^3", "hex20", (6,) * 3, None, "free", "own", None),
    ("curved 6^3", "hex20", (6,) * 3, None, "clamped", "own", "curved"),
    ("checkerboard 4^3 s_K 1e-3", "hex20", (4,) * 3, None, "clamped", "own", "checker"),
    ("nodal 6^3", "hex20", (6,) * 3, None, "clamped", "own", "nodal"),
    *[(f"plate 16^2x1 a={a}", "hex20", (16, 16, 1), (1, 1, 1 / a), "clamped", "own", None)
      for a in (1, 2, 4, 10)],
    *[(f"box 8^3 hz=h/{a}", "hex20", (8,) * 3, (1, 1, 1 / a), "clamped", "own", None)
      for a in (1, 2, 4, 10)],
]  # fmt: skip
VARIANTS = {"plate 16^2x1 a=1": (1000, 2000)}  # extra max_coarse runs (fewer AMG levels)


def its(op, free, sigma, P):
    b = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 4)) * free[:, None])
    info = pcg(ap_operator(op, free, sigma), b, P, TOL, 1000)[1]
    return dict(
        iterations=np.asarray(info["iterations"]).tolist(),
        converged=np.asarray(info["converged"]).tolist(),
    )


def sizes(P):
    return [int(A.n_dof) for A, _ in P.levels] + [int(P.coarse.L.shape[0])]


def nbytes(tree):
    return sum(x.nbytes for x in jax.tree.leaves(tree) if isinstance(x, jax.Array))


def cost(n=44, m=25):
    """Cost and storage at hex20 n^3 with m columns (amg defaults)."""
    m6, _, f6, *_ = case("hex20", (6,) * 3, None, "clamped", None)
    sigma = lam1(m6, f6)
    mesh, op, free, *_ = case("hex20", (n,) * 3, None, "clamped", None)
    A = ap_operator(op, free, sigma)
    jax.block_until_ready(A)
    t = time.perf_counter()
    Ac, pattern = PM._hex8_galerkin(A)
    galerkin = time.perf_counter() - t
    fc = free.reshape(-1, 3)[pattern.corner].ravel()
    t = time.perf_counter()
    AM._sa_levels(Ac, AM._nullspace(mesh.points[pattern.corner], fc), 1000)
    sa = time.perf_counter() - t
    del Ac, pattern
    t = time.perf_counter()
    P = amg(op, free, sigma, mesh.points)
    jax.block_until_ready(P)
    total = time.perf_counter() - t
    X = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, m)))
    nc = P.levels[1][0].n_dof
    Xc = jnp.asarray(np.random.default_rng(0).standard_normal((nc, m)))
    sub = GMG(P.levels[1:], P.transfers[1:], P.coarse, jnp.ones(nc), None)
    T = P.transfers[1]
    Xn = jnp.asarray(np.random.default_rng(0).standard_normal((P.levels[2][0].n_dof, m)))
    t_A = best(jax.jit(A.__call__), X)
    temp = _vcycle_apply.lower(P, X).compile().memory_analysis().temp_size_in_bytes
    nnz = [sum(int((np.asarray(v) != 0).sum()) for v in L.vals) for L, _ in P.levels[1:]]
    return dict(
        n=n,
        m=m,
        n_dof=op.n_dof,
        sizes=sizes(P),
        t_A_s=t_A,
        amg_cycle_t_A=best(sub, Xc) / t_A,
        P_t_A=best(jax.jit(T.__call__), Xn) / t_A,
        PT_t_A=best(jax.jit(T.restrict), Xc) / t_A,
        vcycle_t_A=best(P, X) / t_A,
        vcycle_temp_blocks=temp / X.nbytes,
        setup_s=dict(galerkin=galerkin, sa_loop=sa, amg_total=total),
        storage_GB=dict(
            levels=nbytes([L for L, _ in P.levels[1:]]) / 1e9,
            transfers=nbytes(P.transfers[1:]) / 1e9,
            coarse=nbytes(P.coarse) / 1e9,
        ),
        padding=dict(
            levels=[float(L.padding) for L, _ in P.levels[1:]],
            P=[float(T.P.padding) for T in P.transfers[1:]],
            PT=[float(T.PT.padding) for T in P.transfers[1:]],
        ),
        operator_complexity=sum(nnz) / nnz[0],  # stored nonzeros of the AMG levels
    )


SETUP = """
import sys, time, json
sys.path.insert(0, {root!r})
import jax
from jaxmodal import config
config.enable_x64()
from benchmarks.pmg_study import case
from jaxmodal.precond import amg
mesh, op, free, *_ = case({t!r}, ({n},) * 3, None, "clamped", None)
jax.block_until_ready(op)
"""
ASSEMBLE = """
import importlib
from jaxmodal.operators import ap_operator
PM = importlib.import_module("jaxmodal.precond.pmg")
A = ap_operator(op, free, 1e6)
t = time.perf_counter()
Ab = PM._hex8_galerkin(A)[0]
json.dump(dict(time_s=time.perf_counter() - t, nnz=int(Ab.nnz)), open("{out}", "w"))
"""
BUILD = """
t = time.perf_counter()
jax.block_until_ready(amg(op, free, 1e6, mesh.points))
json.dump(dict(time_s=time.perf_counter() - t), open("{out}", "w"))
"""
REUSE = """
P, agg = amg(op, free, 1e6, mesh.points, return_aggregates=True)
P = None
t = time.perf_counter()
jax.block_until_ready(amg(op, free, 1e6, mesh.points))
fresh = time.perf_counter() - t
t = time.perf_counter()
jax.block_until_ready(amg(op, free, 1e6, mesh.points, aggregates=agg))
json.dump(dict(fresh_s=fresh, reuse_s=time.perf_counter() - t), open("{out}", "w"))
"""
SETUP_SIZES = [("hex20", 12), ("hex20", 24), ("hex20", 44), ("hex27", 12), ("hex27", 24),
               ("hex8", 24), ("hex8", 48), ("hex8", 68)]  # fmt: skip


def setup(sizes=SETUP_SIZES):
    """Per size (xmin clamped, sigma 1e6, defaults): time and host peak of the hex8 Galerkin
    assembler alone and of the whole amg (with compilation, fresh subprocesses, peaks above the
    built operator), and warm times of a fresh build and of one reusing the aggregates."""
    root = str(harness.RESULTS.parents[1])
    out = []
    for t, n in sizes:
        s = SETUP.format(root=root, t=t, n=n)
        r = dict(type=t, n=n, assemble=harness.measure_json(ASSEMBLE, s))
        r["amg"] = harness.measure_json(BUILD, s)
        r["reuse"] = harness.measure_json(REUSE, s)
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


def simp(n=32, max_coarse=(1000, 100)):
    """PCG iterations of AMG (theta = 0, per max_coarse) and GMG on gmg_study's density fields
    as unstructured hex8 meshes (s_K = rho^3, s_M = rho, xmin clamped), at each field's own
    lam_1 and (simp32) at the solid box's; n != 32 skips the checked-in simp32 field."""
    G = gmg_study
    fields = [("s4", ("gen", 4, True)), ("s5", ("gen", 5, True)), ("binary", ("gen", 3, False))]
    if n == 32:
        fields = [("simp32", "simp")] + fields
    solid = G.lam1(*G.case((8,) * 3, "clamped", None, None))
    out = []
    for name, dens in fields:
        op_s, free, mesh = G.case((n,) * 3, "clamped", None, dens)
        op = unstructured(mesh, G.STEEL).scaled(s_K=op_s.s_K, s_M=op_s.s_M)
        at = [("own", G.lam1(op_s, free, mesh))] + ([("solid", solid)] if dens == "simp" else [])
        for where, sigma in at:
            r = dict(case=f"{name}@{where}", n=n, sigma=sigma, gmg=G.iterations(op_s, free, sigma))
            for mc in max_coarse:
                t = time.perf_counter()
                P = amg(op, free, sigma, mesh.points, max_coarse=mc)
                r[f"amg max_coarse={mc}"] = its(op, free, sigma, P) | dict(
                    build_s=time.perf_counter() - t, sizes=sizes(P)
                )
            print(json.dumps(r), flush=True)
            out.append(r)
            jax.clear_caches()
    return out


SOLVE = """
from jaxmodal import config
config.enable_x64()
import json, time
from jaxmodal.eigen import solve_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
mesh = box_mesh(({n},) * 3, (1 / {n},) * 3, type="hex20")
mats, bcs = {{0: Material(isotropic(210e9, 0.3), 7850.0)}}, {{"xmin": "xyz"}}
def run():
    return solve_modes(mesh, mats, bcs, {k}, tol=1e-6)
"""


def acceptance(n=44, n_modes=20):
    """solve_modes (AMG) on the hex20 n^3 steel box (xmin clamped, tol 1e-6, defaults): the cold
    run (time_s with compilation, its info) and the peaks of two warm runs (after a full run),
    next to solve_modes_memory's estimate."""
    mesh = box_mesh((n,) * 3, (1 / n,) * 3, type="hex20")
    setup = SOLVE.format(n=n, k=n_modes)
    r = harness.measure_json(gmg_study.RUN, setup)
    r["cold_peak_bytes"] = r.pop("peak_bytes")
    r["peaks_bytes"] = [
        harness.measure_json(gmg_study.RUN, setup + "run()\n")["peak_bytes"] for _ in range(2)
    ]
    est = solve_modes_memory(mesh, {"xmin": "xyz"}, n_modes, n_rigid=0)["total"]
    out = r | dict(n=n, n_modes=n_modes, n_dof=3 * len(mesh.points), estimate_bytes=est)
    print(json.dumps(out), flush=True)
    return out


def large(n=95, n_modes=10, limit=52e9):
    """The optional 10M run (hex20 n^3, cold, one subprocess); skipped when solve_modes_memory
    exceeds limit (the 64 GB Mac)."""
    mesh = box_mesh((n,) * 3, (1 / n,) * 3, type="hex20")
    est = solve_modes_memory(mesh, {"xmin": "xyz"}, n_modes, n_rigid=0)["total"]
    out = dict(n=n, n_modes=n_modes, n_dof=3 * len(mesh.points), estimate_bytes=est)
    print(json.dumps(out), flush=True)
    if est > limit:
        return out | dict(skipped=f"estimate above {limit / 1e9:.0f} GB")
    r = harness.measure_json(gmg_study.RUN, SOLVE.format(n=n, k=n_modes))
    return out | r


def main(cases=CASES, large=True, out=harness.RESULTS, write=True):
    config.enable_x64()
    runs, sigmas = [], {}
    for name, t, shape, factors, bcs, src, extra in cases:
        mesh, op, free, nodal, s_K = case(t, shape, factors, bcs, extra)
        sigma = sigmas[name] = lam1(mesh, free, nodal, s_K) if src == "own" else sigmas[src]
        r = dict(case=name, n_dof=op.n_dof, sigma=sigma)
        tm = time.perf_counter()
        P = amg(op, free, sigma, mesh.points, max_coarse=MAX_COARSE)
        r["amg"] = its(op, free, sigma, P) | dict(build_s=time.perf_counter() - tm, sizes=sizes(P))
        for mc in VARIANTS.get(name, ()):
            P = amg(op, free, sigma, mesh.points, max_coarse=mc)
            r[f"amg max_coarse={mc}"] = its(op, free, sigma, P) | dict(sizes=sizes(P))
        if t != "hex8" and op.n_dof < 300_000:
            r["pmg"] = its(op, free, sigma, pmg(op, free, sigma))
        print(json.dumps(r), flush=True)
        runs.append(r)
        jax.clear_caches()
    result = dict(
        runs=runs, tol=TOL, max_coarse=MAX_COARSE, system="benchmarks/results/system.json"
    )
    if large:
        result["cost"] = cost()
        print(json.dumps(result["cost"]), flush=True)
        jax.clear_caches()
        result["setup"] = setup()
        result["simp"] = simp()
        result["acceptance"] = acceptance()
    if write:
        harness.write_result("amg_study", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    p.add_argument("--large", action="store_true", help="only the optional 10M run")
    a = p.parse_args()
    if a.large:
        config.enable_x64()
        r = large()
        print(json.dumps(r))
        harness.write_result("amg_study_large", r, a.out)
    else:
        main(out=a.out)
