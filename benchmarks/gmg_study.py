"""Voxel GMG study: PCG iterations per case, the setup peak and the V-cycle cost at 87^3.

Unit-length steel voxel grids (spacing 1 / max(shape)), xmin clamped unless "free"; sigma from
sigma_estimate on the case (rigid modes deflated for free bodies), or as noted; 4 random masked
right-hand sides; relative residual 1e-8; Chebyshev degree 2 over the group block smoother.
The setup peak is the build's process peak RSS above the operator + block Jacobi + lambda_max
baseline, in a fresh subprocess (harness.measure). Writes results/gmg_study.json.
"""

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from benchmarks import harness
from jaxmodal import config
from jaxmodal.eigen import sigma_estimate
from jaxmodal.linalg import pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import voxel_mesh
from jaxmodal.operators import ap_operator, dof_mask, m_orthonormalize, rigid_modes, stencil
from jaxmodal.precond import gmg

STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
SIMP = Path(__file__).parents[1] / "tests" / "data" / "simp32.npy"
TOL = 1e-8
# name, shape, bcs, spacing factors, density, sigma source ("own", or a case name)
CASES = [
    ("box8", (8,) * 3, "clamped", None, None, "own"),
    *[(f"box{n}", (n,) * 3, "clamped", None, None, "box8") for n in (16, 32, 64)],
    *[(f"odd{n}", (n, 8, 8), "clamped", None, None, "own") for n in (9, 17, 33)],
    ("free32", (32,) * 3, "free", None, None, "own"),
    *[(f"plate{n}", (n, n, 1), "clamped", None, None, "own") for n in (16, 32, 64, 128)],
    ("plate64x2", (64, 64, 2), "clamped", None, None, "own"),
    ("beam64x2", (64, 2, 2), "clamped", None, None, "own"),
    ("beam128", (128, 1, 1), "clamped", None, None, "own"),
    ("simp32@solid", (32,) * 3, "clamped", None, "simp", "box32"),
    ("simp32@own", (32,) * 3, "clamped", None, "simp", "own"),
    *[(f"aniso{f}", (32,) * 3, "clamped", (1, 1, f), None, "own") for f in (0.5, 0.25, 0.1)],
]


def case(shape, bcs, factors, density):
    h = 1 / max(shape)
    spacing = tuple(h * f for f in (factors or (1, 1, 1)))
    mesh = voxel_mesh(shape, spacing)
    op = stencil(mesh, STEEL)
    if density == "simp":
        rho = np.load(SIMP).transpose(2, 1, 0).ravel().astype(float)
        op = op.scaled(s_K=rho**3, s_M=rho)
    free = dof_mask(mesh, {} if bcs == "free" else {"xmin": "xyz"})
    return op, free, mesh


def sigma_of(op, free, mesh):
    R = jnp.asarray(rigid_modes(mesh, free))
    R = m_orthonormalize(R, op.apply_M(R)) if R.shape[1] else None
    return sigma_estimate(op, free, R=R)[0]


def iterations(op, free, sigma):
    t = time.perf_counter()
    P = gmg(op, free, sigma)
    build = time.perf_counter() - t
    b = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 4)) * free[:, None])
    info = pcg(ap_operator(op, free, sigma), b, P, TOL, 2000)[1]
    return dict(
        iterations=np.asarray(info["iterations"]).tolist(),
        converged=np.asarray(info["converged"]).tolist(),
        build_s=build,
    )


SETUP = """
from jaxmodal import config
config.enable_x64()
import jax, jax.numpy as jnp, numpy as np
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import voxel_mesh
from jaxmodal.operators import ap_operator, dof_mask, stencil
from jaxmodal.precond import block_jacobi, gmg, lmax_estimate
n = {n}
m = voxel_mesh((n,) * 3, (1 / n,) * 3)
op = stencil(m, {{0: Material(isotropic(210e9, 0.3), 7850.0)}})
free = dof_mask(m, {{"xmin": "xyz"}})
A = ap_operator(op, free, {sigma})
D = block_jacobi(op, free, {sigma})
float(lmax_estimate(A, D))
"""

BUILD = "P = gmg(op, free, {sigma}); jax.block_until_ready(P)"

TIMING = """
import time
P = gmg(op, free, {sigma})
X = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, {m})))
def best(f):
    jax.block_until_ready(f(X))
    ts = []
    for _ in range(3):
        t = time.perf_counter(); jax.block_until_ready(f(X)); ts.append(time.perf_counter() - t)
    return min(ts)
import json
json.dump(dict(vcycle_s=best(P), apply_s=best(A)), open("{out}", "w"))
"""


def large(n, sigma, m=15):
    """Setup peak above the baseline and warm V-cycle / A_P apply times at n^3, m columns."""
    setup = SETUP.format(n=n, sigma=sigma)
    peak = harness.measure(BUILD.format(sigma=sigma), setup)
    t = harness.measure_json(TIMING.format(sigma=sigma, m=m, out="{out}"), setup)
    t["timing_peak_bytes"] = t.pop("peak_bytes")
    return dict(n=n, n_dof=3 * (n + 1) ** 3, m=m, setup_peak_bytes=peak["peak_bytes"], **t)


def main(cases=CASES, n_large=87, out=harness.RESULTS, write=True):
    config.enable_x64()
    runs, sigmas = [], {}
    for name, shape, bcs, factors, density, src in cases:
        op, free, mesh = case(shape, bcs, factors, density)
        sigma = sigmas[name] = sigma_of(op, free, mesh) if src == "own" else sigmas[src]
        r = dict(case=name, shape=shape, n_dof=op.n_dof, sigma=sigma) | iterations(op, free, sigma)
        print(json.dumps(r), flush=True)
        runs.append(r)
    result = dict(runs=runs, tol=TOL, system="benchmarks/results/system.json")
    if n_large:
        result["large"] = large(n_large, sigma_of(*case((8,) * 3, "clamped", None, None)))
        print(json.dumps(result["large"]), flush=True)
    if write:
        harness.write_result("gmg_study", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    main(out=p.parse_args().out)
