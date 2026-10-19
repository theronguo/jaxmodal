"""Strategy study (PR 15d): (1) the MINRES cost model alpha D (1 + D / D0) refitted per family on
results/shift_invert_study.json (frac 0.5, non-floor, the converged recheck_1005 at depth 1005);
(2) on hex20 16^3 with AMG (shift_invert_study's mesh), LOBPCG from 0 per chunk to about 1005
modes (chunk 40) and 200 modes (chunks 5, 10, 20), the crossover table against the measured slice
cost and the held-out full slice; (3) the memory record (estimators only): the ladder's C_max and
slice capacities per lever at 10M hex20 (76 GB pool), 1M hex20 and 2M voxel (the Mac budget), the
reach lost to the fit test's factors and si_memory / LOBPCG chunk 5; at 10M also with a
separate host (an 80 GB GPU node). About 12 min; progress on stdout. Writes
results/strategy_study.json (with the git commit); --parts memory re-runs only (3) (seconds) and
keeps the rest, recording memory_commit and memory_versions.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks import harness
from jaxmodal import config
from jaxmodal import memory as mem
from jaxmodal.eigen.lobpcg import _block_size, _capacity, _solve
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import dof_mask, unstructured
from jaxmodal.precond import amg

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
BCS = {"xmin": "xyz"}
COLUMNS_PER_MODE = "full"  # the slice columns per mode: shift_invert_study's full slice
CHUNKS = (40, 20, 10, 5)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- (1) the MINRES cost model


def rows(si, family):
    """(depths, iterations) of the frac-0.5 non-floor rows, depth 1005 from recheck_1005."""
    out = []
    for r in si[f"grid_{family}"]["rows"]:
        if r["frac"] == 0.5 and not r.get("floor"):
            its = si["recheck_1005"]["iterations"] if r["depth"] == 1005 else r["iterations"]
            out.append((r["depth"], its))
    return np.array(out, float).T


def fit(d, its):
    """(alpha, D0) of its ~ alpha d (1 + d / D0) by log least squares (D0 on a fine log grid,
    alpha closed form)."""
    best = None
    for d0 in np.geomspace(50, 1e5, 4000):
        r = np.log(its) - np.log(d * (1 + d / d0))
        a = r.mean()
        err = ((r - a) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, math.exp(a), d0)
    return best[1], best[2]


def model(alpha, d0, d):
    return alpha * d * (1 + d / d0)


def constants(si):
    out = {}
    for family in ("hex20", "voxel"):
        d, its = rows(si, family)
        a, d0 = fit(d, its)
        out[family] = dict(alpha=a, D0=d0, depths=d.tolist(),
                           ratios=(its / model(a, d0, d)).tolist())  # fmt: skip
    return out


# ---------------------------------------------------------------- (2) LOBPCG and the crossover


def lobpcg_run(op, free, P, chunk, n_target):
    """LOBPCG from 0 (band mode, never reaching lam_stop) until the capacity stop at about
    n_target modes: per chunk its locked count, iterations and t_warm, and the wall time."""
    k = _block_size(chunk, chunk, None)[2]
    c = _capacity(0, chunk, k, False)[0]
    cap = -(-(n_target + k) // c) * c
    t = time.perf_counter()
    lam, _, info = _solve(op, free, 4 * chunk, 1e-8, None, P, None, 1000, 0, chunk=chunk,
                          lam_stop=1e30, max_capacity=cap)  # fmt: skip
    wall = time.perf_counter() - t
    ch = [dict(locked=x["locked"], iterations=x["iterations"], t_warm=x["t_warm"])
          for x in info["chunks"] if x["locked"]]  # fmt: skip
    return dict(chunk=chunk, modes=len(lam), wall=wall, per_mode=wall / len(lam), chunks=ch)


def cumulative(run):
    """(modes, seconds): LOBPCG's cumulative time per locked count (iterations x t_warm)."""
    n = np.cumsum([x["locked"] for x in run["chunks"]])
    s = np.cumsum([x["iterations"] * x["t_warm"] for x in run["chunks"]])
    return np.r_[0, n], np.r_[0, s]


def crossover(si, run, const):
    """Per depth D of the hex20 grid: the slice seconds per mode (columns per mode / 6 steps of
    the measured iterations x s_per_it), LOBPCG's marginal seconds per mode at D, their ratio
    and the smallest N (D + N within the run) with N slice modes cheaper than LOBPCG to D + N
    (None when there is none)."""
    cpm = si[COLUMNS_PER_MODE]["slice"]["columns_per_mode"]
    n, s = cumulative(run)
    out = []
    for r in si["grid_hex20"]["rows"]:
        if r["frac"] != 0.5 or r.get("floor") or r["depth"] + 1 > n[-1]:
            continue
        d = r["depth"]
        its = si["recheck_1005"]["iterations"] if d == 1005 else r["iterations"]
        slice_s = cpm / 6 * its * r["s_per_it"]
        i = min(int(np.searchsorted(n, d)), len(n) - 1)
        marginal = (s[i] - s[i - 1]) / (n[i] - n[i - 1])
        win = next((m for m in range(1, int(n[-1]) - d + 1)
                    if m * slice_s < np.interp(d + m, n, s)), None)  # fmt: skip
        out.append(dict(depth=d, slice_s_per_mode=slice_s, lobpcg_s_per_mode=marginal,
                        ratio=slice_s / marginal, crossover_n=win,
                        model_iterations=model(const["alpha"], const["D0"], d),
                        iterations=its))  # fmt: skip
    return out


def held_out(si, const):
    """The full slice (hex20 12^3, modes 51-70, sigma near mode 60): the model's iterations at
    depth 60 against the measured mean."""
    its = np.mean(si["full"]["slice"]["minres_iterations"])
    pred = model(const["alpha"], const["D0"], 60)
    return dict(depth=60, measured=float(its), predicted=pred, ratio=pred / its)


def per_chunk(runs, deep):
    """Seconds per mode per chunk to about 200 modes (chunk 40: the first 200 modes of the deep
    run), extrapolated to depth 1000 with the deep run's t_warm rise."""
    tw = [(np.cumsum([x["locked"] for x in deep["chunks"]])[i], x["t_warm"])
          for i, x in enumerate(deep["chunks"])]  # fmt: skip
    n_tw, t_tw = zip(*tw, strict=True)
    rise = np.interp(1000, n_tw, t_tw) / np.interp(200, n_tw, t_tw)
    n, s = cumulative(deep)
    out = {40: float(np.interp(200, n, s) / 200)}
    out |= {r["chunk"]: r["per_mode"] for r in runs}
    return {str(c): dict(s_per_mode_200=t, s_per_mode_1000=t * rise, vs_chunk40=t / out[40])
            for c, t in sorted(out.items())}  # fmt: skip


def lobpcg_part(si, const):
    mesh = box_mesh((16,) * 3, (1 / 16,) * 3, type="hex20")
    free = dof_mask(mesh, BCS)
    op = unstructured(mesh, MAT)
    P = amg(op, free, 0.0, mesh.points, nonsingular=True)
    log("chunk 40 to about 1005 modes (about 6 min)")
    deep = lobpcg_run(op, free, P, 40, 1005)
    log(f"chunk 40: {deep['modes']} modes in {deep['wall']:.0f} s")
    runs = []
    for c in (20, 10, 5):
        log(f"chunk {c} to about 200 modes")
        runs.append(lobpcg_run(op, free, P, c, 200))
        log(f"chunk {c}: {runs[-1]['modes']} modes in {runs[-1]['wall']:.0f} s")
    return dict(deep=deep, runs=runs, crossover=crossover(si, deep, const),
                held_out=held_out(si, const), per_chunk=per_chunk(runs, deep))  # fmt: skip


# ---------------------------------------------------------------- (3) the memory record


def ladder(mesh, budget, host_budget, n_rigid=0):
    """Per lever set: C_max per chunk and the slice capacity at n_store 200."""
    voxel = mesh.voxel is not None
    n = 3 * len(mesh.points)
    n_avail = int(dof_mask(mesh, BCS).sum()) - n_rigid
    pcd = ("float64", "float32")  # GMG or AMG V-cycles (fp32 AMG since PR 18c)
    modes = ({},) if voxel else ({}, {"mode": "otf"})
    out = []
    for oo in modes:
        for pd in pcd:
            op, pc = mem._models(mesh, "consistent", "auto", pd, oo or None, None)
            pc64 = mem._models(mesh, "consistent", "auto", "float64", oo or None, None)[1]
            for low in (False, True):
                for dt in ("float64", "float32"):
                    c_max = {c: mem._c_max(op, pc, n, chunk=c, guards=None, low=low,
                                           basis_dtype=dt, n_rigid=n_rigid, n_avail=n_avail,
                                           budget=budget, host_budget=host_budget)
                             for c in CHUNKS}  # fmt: skip
                    out.append(dict(operator=oo.get("mode", "stored"), precond_dtype=pd, low=low,
                                    basis_dtype=dt, c_max=c_max))  # fmt: skip
            slc = mem._slice_capacity(op, pc64, n, n_store=200, block=6, n_rigid=n_rigid,
                                      budget=budget, host_budget=host_budget)  # fmt: skip
            out.append(dict(operator=oo.get("mode", "stored"), precond_dtype=pd,
                            slice_capacity=slc))  # fmt: skip
    return out


def reach_lost(mesh, budget, host_budget):
    """C_max at chunk 5 (9-block, fp64, stored) under the fit test against unit factors."""
    n = 3 * len(mesh.points)
    op, pc = mem._models(mesh, "consistent", "auto", "float64", None, None)
    n_avail = int(dof_mask(mesh, BCS).sum())
    kw = dict(chunk=5, guards=None, low=False, basis_dtype="float64", n_rigid=0,
              n_avail=n_avail, budget=budget, host_budget=host_budget)  # fmt: skip
    fitted = mem._c_max(op, pc, n, **kw)
    saved = mem.FIT_EXACT, mem.FIT_MODEL
    mem.FIT_EXACT = mem.FIT_MODEL = 1.0
    try:
        unit = mem._c_max(op, pc, n, **kw)
    finally:
        mem.FIT_EXACT, mem.FIT_MODEL = saved
    return dict(fit=fitted, unit=unit, lost=None if not unit else 1 - (fitted or 0) / unit)


def si_ratio(mesh):
    """si_memory (block 6, store 200) / lobpcg_memory (chunk 5) totals at C = 48 ... 144."""
    n = 3 * len(mesh.points)
    op, pc = mem._models(mesh, "consistent", "auto", "float64", None, None)
    out = {}
    for C in range(48, 145, 24):
        s = mem.si_memory(op, n, C, 200, 6, pc)["total"]
        lob = mem.lobpcg_memory(op, n, C, 5, None, False, "float64", 0, pc, True, capacity=C)
        out[str(C)] = s / lob["total"]
    return out


TARGETS = {  # the roadmap's max locked modes (fp32 basis, low-memory step)
    "hex20_10M": "220 stored / 300 otf with fp32 AMG (PR 18c): unified 76 GB pool (Mac accounting)",
    "hex20_10M_gpu": "690 stored / 940 otf with fp32 AMG (PR 18c): 76 GB device, separate host",
    "hex20_1M": "> 2000",
    "voxel_2M": "> 2000 (fp32 V-cycle)",
}
# (name, mesh, budget, host_budget); budget None: _default_budget() at run time
MEMORY_CASES = [
    ("hex20_10M", lambda: box_mesh((95,) * 3, type="hex20"), 76e9, None),
    ("hex20_10M_gpu", lambda: box_mesh((95,) * 3, type="hex20"), 76e9, 0.8 * 512e9),
    ("hex20_1M", lambda: box_mesh((44,) * 3, type="hex20"), None, None),
    ("voxel_2M", lambda: voxel_mesh((86,) * 3), None, None),
]


def memory_part():
    out = {}
    for name, make, b, h in MEMORY_CASES:
        if b is None:
            b, h = mem._default_budget()
        log(f"memory record {name}")
        mesh = make()
        out[name] = dict(n_dof=3 * len(mesh.points), budget=b, host_budget=h, target=TARGETS[name],
                         ladder=ladder(mesh, b, h), reach_lost=reach_lost(mesh, b, h),
                         si_over_lobpcg=si_ratio(mesh))  # fmt: skip
        mesh = None
    return out


def commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def run():
    config.enable_x64()
    si = json.loads((harness.RESULTS / "shift_invert_study.json").read_text())
    t = time.perf_counter()
    r = dict(commit=commit())
    r["constants"] = constants(si)
    log(f"constants: {json.dumps({k: (v['alpha'], v['D0']) for k, v in r['constants'].items()})}")
    r["lobpcg"] = lobpcg_part(si, r["constants"]["hex20"])
    r["memory"] = memory_part()
    r["wall_s"] = time.perf_counter() - t
    log(f"done in {r['wall_s'] / 60:.1f} min")
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    p.add_argument("--parts", choices=("all", "memory"), default="all")
    a = p.parse_args(argv)
    if a.parts == "all":
        return harness.write_result("strategy_study", run(), a.out)
    config.enable_x64()
    path = Path(a.out) / "strategy_study.json"
    r = json.loads(path.read_text())  # commit and versions stay those of the full run
    r.update(memory=memory_part(), memory_commit=commit(), memory_versions=harness.versions())
    path.write_text(json.dumps(r, indent=2) + "\n")
    return path


if __name__ == "__main__":
    main()
