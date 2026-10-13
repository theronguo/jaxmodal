"""Shift-invert study (PR 15a): MINRES iterations against depth and delta (hex20 16^3 with AMG,
voxel 32^3 with fp64 GMG, sigma in the gap after modes 5 ... 1000 at three gap fractions), the
tau = 1e-14 floor per depth, a full slice (modes 51-70, hex20 12^3) against
solve_modes(band=...), and one real short slice at the largest Mac sizes: 1M hex20 DOF (AMG,
near mode 100: about 1.7 h) and 2M voxel DOF (GMG, near mode 6). References: reference_modes
(SciPy, certified). --parts runs a subset and merges it into results/shift_invert_study.json.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import harness
from jaxmodal import config
from jaxmodal.eigen import solve_modes
from jaxmodal.eigen.reference import reference_modes
from jaxmodal.eigen.shift_invert import _slice_sets, si_slice
from jaxmodal.linalg import minres
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import ap_operator, assemble, dof_mask, stencil, unstructured
from jaxmodal.precond import amg, gmg

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
BCS = {"xmin": "xyz"}
FRACS = (0.5, 1e-2, 1e-4)


def hex20(n):
    return box_mesh((n,) * 3, (1 / n,) * 3, type="hex20")


def voxel(n):
    return voxel_mesh((n,) * 3, (1 / n,) * 3)


def setup(mesh):
    """op, free and the preconditioner builder of mesh (AMG on hex20, fp64 GMG on voxel)."""
    free = dof_mask(mesh, BCS)
    if mesh.voxel is not None:
        return stencil(mesh, MAT), free, lambda op, s: gmg(op, free, s)
    return unstructured(mesh, MAT), free, lambda op, s: amg(op, free, s, mesh.points)


def reference(mesh, k):
    K, M = assemble(mesh, MAT)
    return reference_modes(K, M, k, dof_mask(mesh, BCS), points=mesh.points)[0]


def widest(lam, i, j):
    """The depth d in [i, j) whose gap lam[d - 1] -> lam[d] is relatively widest (the cube
    has many degenerate pairs)."""
    return max(range(i, j), key=lambda d: (lam[d] - lam[d - 1]) / lam[d])


def rhs(op, free, b=6, seed=0):
    m = jnp.asarray(free, float)[:, None]
    q = m * jax.random.normal(jax.random.key(seed), (op.n_dof, b))
    return m * op.apply_M(q)


def inner(op, free, P, sigma, tau, maxit=20000):
    """One 6-column MINRES solve of (K - sigma M) x = M q: iterations (max), seconds, the
    explicit relative residual (max over columns)."""
    A, b = ap_operator(op, free, -sigma), rhs(op, free)
    minres(A, b, P, tol=tau, maxit=10)  # compile
    t = time.perf_counter()
    x, info = minres(A, b, P, tol=tau, maxit=maxit)
    x.block_until_ready()
    s = time.perf_counter() - t
    r = jnp.linalg.norm(b - A(x), axis=0) / jnp.linalg.norm(b, axis=0)
    its = int(np.asarray(info["iterations"]).max())
    stagnated = bool(np.asarray(info["stagnated"]).any())
    return dict(
        iterations=its,
        seconds=s,
        s_per_it=s / max(1, its),
        residual=float(r.max()),
        stagnated=stagnated,
    )


def t_iteration(op, free, P, sigma):
    """Seconds per MINRES iteration from 10 timed iterations (for the part estimates)."""
    A, b = ap_operator(op, free, -sigma), rhs(op, free)
    minres(A, b, P, tol=0.0, maxit=10)[0].block_until_ready()
    t = time.perf_counter()
    minres(A, b, P, tol=0.0, maxit=10)[0].block_until_ready()
    return (time.perf_counter() - t) / 10


def grid(mesh, depths, name):
    """MINRES iterations and seconds at sigma in the gap after each depth (fractions FRACS of
    the gap above lam_d), and the tau = 1e-14 floor at fraction 0.5."""
    lam = reference(mesh, max(depths) + 6)
    op, free, build = setup(mesh)
    rows = []
    for d in (widest(lam, d, d + 6) for d in depths):
        lo, hi = lam[d - 1], lam[d]
        P = build(op, (lo + hi) / 2)
        print(f"{name} depth {d}: {t_iteration(op, free, P, (lo + hi) / 2):.2e} s/it", flush=True)
        for f in FRACS:
            sigma = lo + f * (hi - lo)
            delta = min(sigma - lo, hi - sigma) / sigma
            rows.append(dict(depth=d, frac=f, sigma=sigma, delta=delta,
                             **inner(op, free, P, sigma, 1e-8)))  # fmt: skip
            print(json.dumps(rows[-1]), flush=True)
        rows.append(dict(depth=d, frac=0.5, floor=True, **inner(op, free, P, (lo + hi) / 2, 1e-14)))
        print(json.dumps(rows[-1]), flush=True)
    return dict(n_dof=op.n_dof, rows=rows)


def slice_info(info):
    keep = ("converged", "reason", "capacity_full", "edge_unreached", "r_acc", "tau_eff",
            "delta_max", "steps", "basis", "sigma", "sigma_moved", "minres_iterations")  # fmt: skip
    return {k: info[k] for k in keep} | {"time": info["time"]}


def large(mesh, coarse, name, depth, block=6):
    """A short slice of about 3 modes after mode depth (edges at the coarse mesh's widest gaps;
    capacity 10 blocks, once 20 if it ends full)."""
    lam = reference(coarse, depth + 10)
    i0 = widest(lam, max(1, depth - 5), depth + 1)  # edges at the widest coarse gaps
    i1 = widest(lam, i0 + 3, i0 + 8)
    lo, hi = (lam[i0 - 1] + lam[i0]) / 2, (lam[i1 - 1] + lam[i1]) / 2
    op, free, build = setup(mesh)
    t = time.perf_counter()
    P = build(op, (lo + hi) / 2)
    out = dict(n_dof=op.n_dof, lo=lo, hi=hi, precond_s=time.perf_counter() - t)
    t_it = t_iteration(op, free, P, (lo + hi) / 2)
    print(f"{name}: {t_it:.2e} s/it, {op.n_dof} DOF, depth {depth}", flush=True)
    for cap in (10 * block, 20 * block):
        print(f"{name} slice, capacity {cap}", flush=True)
        t = time.perf_counter()
        lam_s, _, info = si_slice(
            op, free, lo, hi, P, capacity=cap, tol=1e-6, block=block, verbose=True
        )
        out[f"slice_{cap}"] = slice_info(info) | dict(lam=lam_s.tolist(),
                                                     wall=time.perf_counter() - t)  # fmt: skip
        print(json.dumps(out[f"slice_{cap}"], default=str), flush=True)
        if not info["capacity_full"]:
            break
    return out


def floor(mesh, coarse, name, depth=6):
    """The tau = 1e-14 floor at the largest Mac size: one 6-column solve with sigma mid-gap
    after mode depth of the coarse mesh."""
    lam = reference(coarse, depth + 10)
    d = widest(lam, depth, depth + 6)
    sigma = (lam[d - 1] + lam[d]) / 2
    op, free, build = setup(mesh)
    P = build(op, sigma)
    print(f"{name}: {t_iteration(op, free, P, sigma):.2e} s/it, depth {d}", flush=True)
    out = dict(n_dof=op.n_dof, depth=d, sigma=sigma, **inner(op, free, P, sigma, 1e-14))
    print(json.dumps(out), flush=True)
    return out


def recheck(path, depth=1005):
    """The mid-gap tau = 1e-8 solve at depth of the hex20 16^3 grid again, sigma read from the
    stored grid (a second run: at depth 1000 the outcome depends on round-off)."""
    rows = json.loads(Path(path).read_text())["grid_hex20"]["rows"]
    sigma = next(r["sigma"] for r in rows if r["depth"] == depth and r["frac"] == 0.5)
    op, free, build = setup(hex20(16))
    P = build(op, sigma)
    t = t_iteration(op, free, P, sigma)
    print(f"recheck {depth}: {t:.2e} s/it, at most {t * 20000 / 60:.0f} min", flush=True)
    out = dict(depth=depth, frac=0.5, sigma=sigma, **inner(op, free, P, sigma, 1e-8))
    print(json.dumps(out), flush=True)
    return out


def full_slice(n, first=51, last=70):
    """Modes first..last (1-based) of hex20 n^3 at tol 1e-8 by one slice and by
    solve_modes(band=(0, f(hi))) (LOBPCG from 0)."""
    mesh = hex20(n)
    lam = reference(mesh, last + 6)
    lo, hi = (lam[first - 2] + lam[first - 1]) / 2, (lam[last - 1] + lam[last]) / 2
    op, free, build = setup(mesh)
    t = time.perf_counter()
    P = build(op, (lo + hi) / 2)
    lam_s, _, info = si_slice(op, free, lo, hi, P, capacity=60 * 6, tol=1e-8, verbose=True)
    wall = time.perf_counter() - t
    ref = lam[_slice_sets(lam, lo, hi, info["r_acc"])[0]]  # the slice's rule: edge clusters whole
    out = dict(n_dof=op.n_dof, slice=slice_info(info) | dict(
        wall=wall, n_modes=len(lam_s), n_ref=len(ref), columns_per_mode=info["basis"] / len(lam_s),
        error=float(np.abs(lam_s - ref).max() / hi) if len(lam_s) == len(ref) else None,
    ))  # fmt: skip
    t = time.perf_counter()
    lb, _, ib = solve_modes(mesh, MAT, BCS, tol=1e-8, band=(0, np.sqrt(hi) / (2 * np.pi)))
    out["band"] = dict(wall=time.perf_counter() - t, n_modes=len(lb), iterations=ib["iterations"])
    print(json.dumps(out, default=str), flush=True)
    return out


PARTS = ("grid_hex20", "grid_voxel", "full", "hex20_1M", "voxel_2M", "floor_1M", "floor_2M")
EXTRA = ("recheck_1005",)  # run only when named


def run(small=False, parts=PARTS, out=harness.RESULTS):
    config.enable_x64()
    if small:
        r = dict(grid=dict(hex20=grid(hex20(4), (5, 10), "hex20")))
        r["slice"] = full_slice(3, 11, 16)["slice"]
        return r
    r, walls = {}, {}
    jobs = dict(
        grid_hex20=lambda: grid(hex20(16), (5, 20, 50, 100, 200, 500, 1000), "hex20"),
        grid_voxel=lambda: grid(voxel(32), (5, 20, 50, 100, 200), "voxel"),
        full=lambda: full_slice(12),
        hex20_1M=lambda: large(hex20(44), hex20(12), "hex20_1M", 100),
        voxel_2M=lambda: large(voxel(86), voxel(32), "voxel_2M", 6),
        floor_1M=lambda: floor(hex20(44), hex20(12), "floor_1M"),
        floor_2M=lambda: floor(voxel(86), voxel(32), "floor_2M"),
        recheck_1005=lambda: recheck(Path(out) / "shift_invert_study.json"),
    )
    for name in parts:
        print(f"== {name}", flush=True)
        t = time.perf_counter()
        r[name] = jobs[name]()
        walls[name] = time.perf_counter() - t
        print(f"== {name} done in {walls[name] / 60:.1f} min", flush=True)
        jax.clear_caches()
    r["walls"] = walls
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    p.add_argument("--parts", nargs="+", choices=PARTS + EXTRA, default=PARTS)
    a = p.parse_args()
    r = run(parts=a.parts, out=a.out)
    path = Path(a.out) / "shift_invert_study.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.pop("versions", None)
    walls = old.pop("walls", {}) | r.pop("walls")
    harness.write_result("shift_invert_study", old | r | dict(walls=walls), a.out)
