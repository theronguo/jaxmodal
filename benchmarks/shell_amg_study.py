"""Solid-shell AMG study (PR 18a): the PCG counts of tests 6-11 in tests/test_shell_amg.py
(levels, sizes and build times), the cost on the flat SS plate 256^2 at L/t = 1000 (396k DOF,
m = 15 columns): t_A, the V-cycle in t_A, device storage per level and component, host setup
time; and on the hex-dominant 32^3 block with an ssh8 skin the same next to PR 13's hex8 build
of the block without the skin. PCG to 1e-6 on 4 fixed-seed masked columns, max over columns.
Times include compilation unless "best" (min of 5 runs). Writes results/shell_amg_study.json.
"""

from jaxmodal import config

config.enable_x64()  # before any JAX constant (the test helpers)

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import harness
from benchmarks.pmg_study import best

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
import test_shell_amg as T
from exact import SIMPLY_SUPPORTED as SS
from helpers import STEEL, lam1, rhs
from test_shell import C_ORT, laminate, three_plies

from jaxmodal.linalg import pcg
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import ap_operator, dof_mask, unstructured
from jaxmodal.precond import amg

B = {"zmin": "xyz"}


def cases():
    """(name, mesh, sigma, bcs, mats, max_coarse) of tests 6-11, built lazily."""
    for Lt in (10, 1000):
        m, t = T.plate(128, Lt)
        yield f"flat128-{Lt}", m, T.kirchhoff(t), SS, STEEL, 1000
        m64 = T.plate(64, Lt, curved=True)[0]
        s = lam1(m64, dof_mask(m64, SS))
        yield f"curved128-{Lt}", T.plate(128, Lt, curved=True)[0], s, SS, STEEL, 1000
    for n, Lt in ((64, 100), (128, 100), (256, 1000)):
        m, t = T.plate(n, Lt)
        yield f"flat{n}-{Lt}", m, T.kirchhoff(t), SS, STEEL, 1000
    m, t = T.plate(32, 50)
    yield "flat32-50", m, T.kirchhoff(t), SS, STEEL, 100
    mats = {0: laminate(three_plies(C_ORT))}
    yield "plies32-50", m, T.eig(m, dof_mask(m, SS), mats, 1)[0], SS, mats, 100
    m = T.plate(32, 50, curved=True)[0]
    yield "curved32-50", m, lam1(m, dof_mask(m, SS)), SS, STEEL, 100
    m = T.cylinder(False)
    yield "cylinder", m, lam1(m, dof_mask(m, {})), {}, STEEL, 100
    for kind, k in (("T", 1), ("L", 1), ("T", 3)):
        m = T.junction(kind, 24, 50, k, k)[0]
        yield f"{kind}{k}", m, lam1(m, dof_mask(m, T.CLAMPED)), T.CLAMPED, STEEL, 100
    m, t = T.plate(32, 300)
    yield "flat32-300", m, T.kirchhoff(t), SS, STEEL, 100
    m = T.web(32, 50)[0]
    yield "web32", m, lam1(m, dof_mask(m, SS)), SS, STEEL, 100
    m = T.block(16)[0]
    yield "block16", m, lam1(m, dof_mask(m, B)), B, STEEL, 100
    m, t = T.plate(32, 50)
    yield "clamped32", m, T.kirchhoff(t, 35.99), T.CLAMPED, STEEL, 100
    m = T.bottom(32, 50)[0]
    yield "bottom32", m, lam1(m, dof_mask(m, {"bottom": "xyz"})), {"bottom": "xyz"}, STEEL, 100
    m = T.parts(0.02)
    yield "parts", m, lam1(m, dof_mask(m, {"ends": "xyz"})), {"ends": "xyz"}, STEEL, 40
    m = T.plate(24, 50)[0]
    yield "free24", m, T.eig(m, dof_mask(m, {}))[6], {}, STEEL, 100
    for Lt in (10, 100, 300, 1000):
        m, t = T.plate(32, Lt, layers=3)
        yield f"stack32-{Lt}", m, T.kirchhoff(t), SS, STEEL, 100


def its(op, free, sigma, P, maxit=2000):
    info = pcg(ap_operator(op, free, sigma), jnp.asarray(rhs(free, 4)), P, 1e-6, maxit)[1]
    return int(np.asarray(info["iterations"]).max()), bool(np.asarray(info["converged"]).all())


def sizes(P):
    return [int(A.n_dof) for A, _ in P.levels] + [int(P.coarse.L.shape[0])]


def nbytes(tree, exclude=()):
    """Device bytes of the distinct arrays of tree that are not in exclude."""
    skip = {id(x) for x in jax.tree.leaves(exclude)}
    own = {id(x): x.nbytes for x in jax.tree.leaves(tree) if isinstance(x, jax.Array)}
    return sum(v for k, v in own.items() if k not in skip)


def cost(mesh, sigma, bcs, m=15, max_coarse=1000):
    """t_A, V-cycle / t_A, the host setup time and device bytes per fine DOF by component
    (distinct arrays; smoothers without their level matrix, everything without the operator)."""
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, bcs)
    A = ap_operator(op, free, sigma)
    t = time.perf_counter()
    P = amg(op, free, sigma, mesh.points, max_coarse=max_coarse)
    jax.block_until_ready(P)
    setup = time.perf_counter() - t
    X = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, m)))
    t_A = best(jax.jit(A.__call__), X)
    n = op.n_dof
    levels = [
        dict(n_dof=int(L.n_dof), smoother=nbytes(S, L) / n, A=nbytes(L, A) / n,
             transfer=nbytes(P.transfers[i]) / n)
        for i, (L, S) in enumerate(P.levels)
    ]  # fmt: skip
    return dict(
        n_dof=n,
        m=m,
        its=its(op, free, sigma, P),
        t_A_s=t_A,
        vcycle_t_A=best(P, X) / t_A,
        setup_s=setup,
        operator_B_per_dof=nbytes(A) / n,
        levels_B_per_dof=levels,
        coarse_B_per_dof=nbytes(P.coarse) / n,
        total_B_per_dof=nbytes(P, A) / n,
    )


def main(out=harness.RESULTS, write=True):
    runs = []
    for name, mesh, sigma, bcs, mats, mc in cases():
        op, free = unstructured(mesh, mats), dof_mask(mesh, bcs)
        t = time.perf_counter()
        P = amg(op, free, float(sigma), mesh.points, max_coarse=mc)
        jax.block_until_ready(P)
        build = time.perf_counter() - t
        n, ok = its(op, free, float(sigma), P)
        r = dict(case=name, n_dof=op.n_dof, sigma=float(sigma), max_coarse=mc, iterations=n,
                 converged=ok, sizes=sizes(P), build_s=build)  # fmt: skip
        print(json.dumps(r), flush=True)
        runs.append(r)
        jax.clear_caches()
    m, t = T.plate(256, 1000)
    plate = cost(m, T.kirchhoff(t), SS)
    print(json.dumps(plate), flush=True)
    jax.clear_caches()
    m = T.block(32)[0]
    s = lam1(T.block(16)[0], dof_mask(T.block(16)[0], B))
    skin = cost(m, s, B, max_coarse=1000)
    print(json.dumps(skin), flush=True)
    jax.clear_caches()
    cube = box_mesh((32, 32, 32), spacing=(1 / 32,) * 3)
    hex8 = cost(cube, s, B, max_coarse=1000)
    print(json.dumps(hex8), flush=True)
    result = dict(
        runs=runs,
        cost_plate256_Lt1000=plate,
        cost_block32_skin=skin,
        cost_block32_hex8_pr13=hex8,
        tol=1e-6,
        system="benchmarks/results/system.json",
    )
    if write:
        harness.write_result("shell_amg_study", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    a = p.parse_args()
    main(out=a.out)
