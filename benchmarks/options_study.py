"""Options study (PR 13c): solve_modes at Chebyshev degree 2 / 3 / 4 (AMG on hex20 24^3, GMG on
voxel 32^3; 20 modes, warm second run), V-cycle times and temporaries per degree (AMG hex20
16^3, GMG voxel 64^3 fp64 and fp32; m = 15), the X0 warm start and Jacobi against AMG
(hex20 8^3, 10 modes), and amg_study.cost at 1M hex20. Steel boxes, xmin clamped, tol 1e-6.
Writes results/options_study.json.
"""

import argparse
import functools
import importlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import amg_study, harness
from benchmarks.pmg_study import best
from jaxmodal import config
from jaxmodal.eigen import solve_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import dof_mask, stencil, unstructured
from jaxmodal.precond import amg, gmg

G = importlib.import_module("jaxmodal.precond.gmg")
STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
BCS = {"xmin": "xyz"}


def hex20(n):
    return box_mesh((n,) * 3, (1 / n,) * 3, type="hex20")


def voxel(n):
    return voxel_mesh((n,) * 3, (1 / n,) * 3)


def degree(mesh, n_modes=20, degrees=(2, 3, 4)):
    """Per degree: iterations and warm times (LOBPCG, precond, total) of solve_modes."""
    out = []
    for d in degrees:
        for _ in range(2):  # the second run is warm
            t = time.perf_counter()
            _, _, info = solve_modes(mesh, STEEL, BCS, n_modes, precond_options={"degree": d})
            total = time.perf_counter() - t
        out.append(
            dict(degree=d, iterations=info["iterations"], converged=bool(info["converged"]),
                 lobpcg_s=info["time"]["iterations"], precond_s=info["time"]["precond"],
                 total_s=total)
        )  # fmt: skip
        print(json.dumps(out[-1]), flush=True)
    return out


def vcycle(build, mesh, op, degrees=(2, 4), m=15):
    """Per degree: V-cycle time (best of 5) and temporaries in fp64 blocks of (n_dof, m)."""
    free = dof_mask(mesh, BCS)
    X = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, m)))
    out = []
    for d in degrees:
        P = build(op, free, d)
        temp = G._vcycle_apply.lower(P, X).compile().memory_analysis().temp_size_in_bytes
        out.append(dict(degree=d, time_s=best(P, X), temp_blocks=temp / X.nbytes))
    return out


def run(small=False):
    config.enable_x64()
    n = (8, 8, 6, 8, 4) if small else (24, 32, 16, 64, 8)
    mesh = hex20(n[4])
    _, V, info = solve_modes(mesh, STEEL, BCS, 10)
    _, _, warm = solve_modes(mesh, STEEL, BCS, 10, X0=V)
    _, _, jac = solve_modes(mesh, STEEL, BCS, 10, precond="jacobi")
    r = dict(
        x0=dict(mesh=f"hex20 {n[4]}^3", cold=info["iterations"], warm=warm["iterations"]),
        jacobi=dict(mesh=f"hex20 {n[4]}^3", amg=info["iterations"], jacobi=jac["iterations"]),
        degree_amg=degree(hex20(n[0])),
        degree_gmg=degree(voxel(n[1])),
    )
    m16 = hex20(n[2])
    op = unstructured(m16, STEEL)
    r["vcycle_amg"] = vcycle(lambda o, f, d: amg(o, f, 1e6, m16.points, degree=d), m16, op)
    v = voxel(n[3])
    op = stencil(v, STEEL)
    for dt in ("float64", "float32"):
        build = functools.partial(lambda o, f, d, dt: gmg(o, f, 1e5, degree=d, dtype=dt), dt=dt)
        r[f"vcycle_gmg_{dt}"] = vcycle(build, v, op)
    jax.clear_caches()
    if not small:
        r["cost"] = amg_study.cost(44)
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    a = p.parse_args()
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "cost"}, indent=1))
    harness.write_result("options_study", r, a.out)
