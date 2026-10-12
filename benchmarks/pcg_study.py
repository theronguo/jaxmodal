"""h-study of PCG on A_P = K + sigma M with the single-level preconditioners.

Clamped (xmin) unit steel boxes, hex20 and voxel, refined in h; sigma fixed per family at
lambda_1 of its coarsest mesh (sigma_estimate); 4 random masked right-hand sides; relative
residual 1e-8. Preconditioners: Jacobi, block Jacobi, Chebyshev degree 2 over block Jacobi.
Writes results/pcg_study.json.
"""

import argparse
import json
import time

import jax.numpy as jnp
import numpy as np

from benchmarks import harness
from jaxmodal import config
from jaxmodal.eigen import sigma_estimate
from jaxmodal.linalg import pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import ap_operator, dof_mask, stencil, unstructured
from jaxmodal.precond import block_jacobi, chebyshev, jacobi, lmax_estimate

SIZES = {"hex20": (4, 8, 16), "voxel": (8, 16, 32)}
STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
TOL = 1e-8


def operator(family, n):
    if family == "hex20":
        mesh = box_mesh((n,) * 3, spacing=(1 / n,) * 3, type="hex20")
        return unstructured(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"})
    mesh = voxel_mesh((n,) * 3, spacing=(1 / n,) * 3)
    return stencil(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"})


def run(op, free, sigma, name):
    A = ap_operator(op, free, sigma)
    if name == "jacobi":
        P = jacobi(op, free, sigma)
    else:
        P = block_jacobi(op, free, sigma)
        if name == "chebyshev2":
            P = chebyshev(A, P, lmax_estimate(A, P))
    b = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, 4)) * free[:, None])
    pcg(A, b, P, TOL, 1)[0].block_until_ready()  # compile
    t = time.perf_counter()
    x, info = pcg(A, b, P, TOL, 20000)
    x.block_until_ready()
    return dict(
        time_s=time.perf_counter() - t,
        iterations=np.asarray(info["iterations"]).tolist(),
        converged=np.asarray(info["converged"]).tolist(),
    )


def main(sizes=SIZES, out=harness.RESULTS, write=True):
    config.enable_x64()
    runs, sigmas = [], {}
    for family, ns in sizes.items():
        for n in ns:
            op, free = operator(family, n)
            if family not in sigmas:
                sigmas[family] = sigma_estimate(op, free)[0]
            for name in ("jacobi", "block_jacobi", "chebyshev2"):
                r = dict(family=family, n=n, n_dof=op.n_dof, precond=name)
                r |= run(op, free, sigmas[family], name)
                print(json.dumps(r), flush=True)
                runs.append(r)
    result = dict(runs=runs, sigma=sigmas, tol=TOL, system="benchmarks/results/system.json")
    if write:
        harness.write_result("pcg_study", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    main(out=p.parse_args().out)
