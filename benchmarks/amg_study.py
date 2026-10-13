"""AMG study: PCG iterations of amg against pmg's exact coarse solve (h-sequences, robustness,
element aspect), and the cost at hex20 44^3 (about 1M DOF): t_A, the AMG cycle, the transfers
and the whole V-cycle in t_A, V-cycle temporaries, setup times, device storage and level sizes.

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

from benchmarks import harness
from benchmarks.pmg_study import best, case, lam1
from jaxmodal import config
from jaxmodal.linalg import pcg
from jaxmodal.operators import ap_operator
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
    Ac, _, corner = PM._hex8_galerkin(A)
    galerkin = time.perf_counter() - t
    fc = free.reshape(-1, 3)[corner].ravel()
    t = time.perf_counter()
    AM._sa_levels(Ac, AM._nullspace(mesh.points[corner], fc), 1000)
    sa = time.perf_counter() - t
    del Ac
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
    if write:
        harness.write_result("amg_study", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    main(out=p.parse_args().out)
