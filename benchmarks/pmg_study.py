"""Unstructured p-MG study: PCG iterations per case (h-sequences, robustness, Chebyshev ratio and
degree, element aspect), LOBPCG against Jacobi, the V-cycle cost in t_A and the build time at
hex20 24^3, and _hex8_galerkin alone at hex20 44^3 (about 1M DOF).

Unit-length steel boxes (spacing 1 / max(shape)), xmin clamped unless "free". sigma = lam_1 of the
case from reference_modes (SciPy), or of the named case (h-sequences use their 6^3 box). 4 random
masked right-hand sides, relative residual 1e-8, Chebyshev ratio 30 over node-block Jacobi.
Times include compilation. jax.clear_caches() between cases frees the host LU factors.
Writes results/pmg_study.json.
"""

import argparse
import dataclasses
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import harness
from jaxmodal import config
from jaxmodal.eigen import lobpcg, reference_modes
from jaxmodal.linalg import pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import ap_operator, assemble, dof_mask, rigid_modes, unstructured
from jaxmodal.precond import jacobi, pmg
from tests.helpers import distort

STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
TOL = 1e-8
# name, type, shape, spacing factors, bcs, sigma source, degree, ratio, extra
CASES = [
    ("hex20 6^3", "hex20", (6,) * 3, None, "clamped", "own", 2, 30.0, None),
    *[(f"hex20 {n}^3", "hex20", (n,) * 3, None, "clamped", "hex20 6^3", 2, 30.0, None)
      for n in (3, 12, 24)],
    *[(f"hex20 {n}^3 d4", "hex20", (n,) * 3, None, "clamped", "hex20 6^3", 4, 30.0, None)
      for n in (3, 6, 12)],
    *[(f"hex20 {n}^3 d3", "hex20", (n,) * 3, None, "clamped", "hex20 6^3", 3, 30.0, None)
      for n in (3, 6)],
    ("hex27 6^3", "hex27", (6,) * 3, None, "clamped", "own", 2, 30.0, None),
    *[(f"hex27 {n}^3", "hex27", (n,) * 3, None, "clamped", "hex27 6^3", 2, 30.0, None)
      for n in (3, 12)],
    ("beam 20x2x2", "hex20", (20, 2, 2), None, "clamped", "own", 2, 30.0, None),
    ("free 6^3", "hex20", (6,) * 3, None, "free", "own", 2, 30.0, None),
    ("curved 6^3", "hex20", (6,) * 3, None, "clamped", "own", 2, 30.0, "curved"),
    ("checkerboard 4^3 s_K 1e-3", "hex20", (4,) * 3, None, "clamped", "own", 2, 30.0, "checker"),
    ("nodal 6^3", "hex20", (6,) * 3, None, "clamped", "own", 2, 30.0, "nodal"),
    *[(f"cube 8^3 ratio {r:g}", "hex20", (8,) * 3, None, "clamped", "own", 2, r, None)
      for r in (10.0, 20.0, 30.0, 60.0)],
    *[(f"plate 16^2x1 a={a}", "hex20", (16, 16, 1), (1, 1, 1 / a), "clamped", "own", 2, 30.0,
       None) for a in (1, 2, 4, 10)],
    *[(f"box 8^3 hz=h/{a}", "hex20", (8,) * 3, (1, 1, 1 / a), "clamped", "own", 2, 30.0, None)
      for a in (1, 2, 4, 10)],
]  # fmt: skip
LOBPCG = [("hex20", (12,) * 3), ("hex20", (20, 2, 2)), ("hex27", (16,) * 3)]


def case(t, shape, factors, bcs, extra):
    h = 1 / max(shape)
    mesh = box_mesh(shape, tuple(h * f for f in (factors or (1, 1, 1))), type=t)
    nodal, s_K = None, None
    if extra == "curved":  # distort works on unit spacing
        mesh = distort(box_mesh(shape, type=t), True, np.random.default_rng(8))
        mesh = dataclasses.replace(mesh, points=mesh.points * h)
    elif extra == "nodal":  # one element mass on a seeded half of the nodes
        nodal = 7850.0 * h**3 * (np.random.default_rng(8).random(len(mesh.points)) < 0.5)
    elif extra == "checker":
        c = np.floor(mesh.points[mesh.blocks[t][:, :8]].mean(1) / h).astype(int).sum(1) % 2
        s_K = np.where(c == 1, 1e-3, 1.0)
    op = unstructured(mesh, STEEL, nodal_mass=nodal)
    if s_K is not None:
        op = op.scaled(s_K=jnp.asarray(s_K))
    free = dof_mask(mesh, {} if bcs == "free" else {"xmin": "xyz"})
    return mesh, op, free, nodal, s_K


def lam1(mesh, free, nodal=None, s_K=None, k=1):
    """The lowest k non-rigid eigenvalues from the SciPy reference (reference_modes)."""
    K, M = assemble(mesh, STEEL, nodal_mass=nodal, s_K=s_K)
    nr = rigid_modes(mesh, free).shape[1]
    lam = reference_modes(K, M, nr + max(k, 3), free, points=mesh.points)[0][nr:]
    return float(lam[0]) if k == 1 else lam[:k]


def iterations(op, free, sigma, degree, ratio):
    t = time.perf_counter()
    P = pmg(op, free, sigma, degree=degree, ratio=ratio)
    build = time.perf_counter() - t
    b = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 4)) * free[:, None])
    info = pcg(ap_operator(op, free, sigma), b, P, TOL, 1000)[1]
    return dict(
        iterations=np.asarray(info["iterations"]).tolist(),
        converged=np.asarray(info["converged"]).tolist(),
        build_s=build,
        n_coarse=int(P.transfers[0].child.shape[0]) * 3,
    )


def best(f, X, reps=5):
    jax.block_until_ready(f(X))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(X))
        ts.append(time.perf_counter() - t)
    return min(ts)


def cost(n=24, m=25):
    """t_A, the V-cycle and its coarse solve alone at hex20 n^3 (m columns), and the build."""
    m6, _, f6, *_ = case("hex20", (6,) * 3, None, "clamped", None)
    sigma = lam1(m6, f6)
    mesh, op, free, *_ = case("hex20", (n,) * 3, None, "clamped", None)
    t = time.perf_counter()
    P = pmg(op, free, sigma)
    jax.block_until_ready(P)
    build = time.perf_counter() - t
    X = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, m)))
    Xc = jnp.asarray(np.random.default_rng(0).standard_normal((3 * len(P.transfers[0].child), m)))
    t_A = best(jax.jit(P.levels[0][0].__call__), X)
    out = dict(n=n, m=m, n_dof=op.n_dof, build_s=build, t_A_s=t_A, vcycle_s=best(P, X))
    out["coarse_s"] = best(jax.jit(P.coarse.__call__), Xc)
    return out


GAL_SETUP = """
from jaxmodal import config
config.enable_x64()
import jax
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import ap_operator, dof_mask, unstructured
from jaxmodal.precond.pmg import _hex8_galerkin
mesh = box_mesh(({n},) * 3, (1 / {n},) * 3, type="hex20")
op = unstructured(mesh, {{0: Material(isotropic(210e9, 0.3), 7850.0)}}, mode="{mode}")
A = ap_operator(op, dof_mask(mesh, {{"xmin": "xyz"}}), 1e7)
jax.block_until_ready(A)
"""

GAL_RUN = """
import json, time
t = time.perf_counter()
Ac = _hex8_galerkin(A)[0]
json.dump(dict(n_dof=op.n_dof, time_s=time.perf_counter() - t, nnz=int(Ac.nnz)), open("{out}", "w"))
"""


def galerkin_1m(n=44):
    """_hex8_galerkin alone (no splu) at hex20 n^3, stored and otf, with compilation, each in a
    fresh subprocess: peak_bytes is the process peak above the built operator (harness)."""
    return {
        mode: harness.measure_json(GAL_RUN, GAL_SETUP.format(n=n, mode=mode))
        for mode in ("stored", "otf")
    }


def lobpcg_runs(cases=LOBPCG, k=10, tol=1e-6):
    runs = []
    for t, shape in cases:
        mesh, op, free, *_ = case(t, shape, None, "clamped", None)
        sigma = lam1(mesh, free)
        r = dict(case=f"{t} {'x'.join(map(str, shape))}", n_dof=op.n_dof, sigma=sigma)
        for name in ("jacobi", "pmg"):
            P = jacobi(op, free) if name == "jacobi" else pmg(op, free, sigma)
            tm = time.perf_counter()
            lam, _, info = lobpcg(op, free, k, tol=tol, precond=P, maxit=2000)
            r[name] = dict(
                iterations=int(info["iterations"]),
                converged=bool(info["converged"]),
                time_s=time.perf_counter() - tm,
                lam=np.asarray(lam).tolist(),
            )
        print(json.dumps(r), flush=True)
        runs.append(r)
        jax.clear_caches()
    return runs


def main(cases=CASES, lobpcg_cases=LOBPCG, large=True, out=harness.RESULTS, write=True):
    config.enable_x64()
    runs, sigmas = [], {}
    for name, t, shape, factors, bcs, src, degree, ratio, extra in cases:
        mesh, op, free, nodal, s_K = case(t, shape, factors, bcs, extra)
        sigma = sigmas[name] = lam1(mesh, free, nodal, s_K) if src == "own" else sigmas[src]
        r = dict(case=name, n_dof=op.n_dof, sigma=sigma, degree=degree, ratio=ratio)
        r |= iterations(op, free, sigma, degree, ratio)
        print(json.dumps(r), flush=True)
        runs.append(r)
        jax.clear_caches()
    result = dict(runs=runs, tol=TOL, system="benchmarks/results/system.json")
    result["lobpcg"] = lobpcg_runs(lobpcg_cases)
    if large:
        result["cost"] = cost()
        print(json.dumps(result["cost"]), flush=True)
        jax.clear_caches()
        result["galerkin_1m"] = galerkin_1m()
        print(json.dumps(result["galerkin_1m"]), flush=True)
    if write:
        harness.write_result("pmg_study", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    main(out=p.parse_args().out)
