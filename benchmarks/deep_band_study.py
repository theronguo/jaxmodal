"""Deep-band study (PR 18f) on hex20 16^3 with AMG (shift_invert_study's mesh and setup).
solves: the first MINRES solve at the stored sigma of shift_invert_study's depths 503 and 1005
(6 columns, M q, tau 1e-8), deflated by every LOBPCG mode below sigma and by the nearest half,
and the deflation's per-iteration overhead; Z from a chunk-40 LOBPCG run to 1045 modes cached
in --cache (about 8 min without it). band: solve_modes(band=...) from mode ~951 to ~1008 at
chunk 40 with levers=False, verify=False and a budget whose C_max = 960 capacity-stops LOBPCG at
D = 964 (the cut path of the junction), per slice the MINRES iterations, deflation and sigma
moves, the band against eigsh at its centre. About 20 + 45 min; progress on stdout. --parts
merges into results/deep_band_study.json. Undeflated references (cited, not rerun): 4970
iterations at depth 503 and 14,370 at 1005 (shift_invert_study grid_hex20, recheck_1005).
"""

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

from benchmarks import harness
from benchmarks.shift_invert_study import BCS, MAT, hex20, rhs, setup, widest
from jaxmodal import config
from jaxmodal import memory as mem
from jaxmodal.eigen import solve_modes
from jaxmodal.eigen.lobpcg import _block_size, _capacity, _solve
from jaxmodal.linalg import minres
from jaxmodal.operators import ap_operator, assemble
from jaxmodal.precond import amg

api = importlib.import_module("jaxmodal.eigen.api")
si = importlib.import_module("jaxmodal.eigen.shift_invert")
CACHE = Path.home() / ".cache" / "jaxmodal" / "deep_band_lobpcg.npz"
TAU, CHUNK, C_MAX = 1e-8, 40, 960
PARTS = ("solves", "band")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lobpcg_cache(path):
    """(lam, V, thresholds, locked per chunk) of LOBPCG from 0 at chunk 40 to 1045 modes
    (strategy_study's deep run), from path or run and saved there."""
    if path.exists():
        z = np.load(path)
        return z["lam"], z["V"], z["t"], z["locked"]
    mesh = hex20(16)
    op, free, _ = setup(mesh)
    P = amg(op, free, 0.0, mesh.points, nonsingular=True)
    k = _block_size(CHUNK, CHUNK, None)[2]
    c = _capacity(0, CHUNK, k, False)[0]
    log("LOBPCG chunk 40 to about 1045 modes (about 8 min)")
    lam, V, info = _solve(op, free, 4 * CHUNK, 1e-8, None, P, None, 1000, 0, chunk=CHUNK,
                          lam_stop=1e30, max_capacity=-(-(1005 + k) // c) * c)  # fmt: skip
    o = np.argsort(lam)
    locked = np.array([x["locked"] for x in info["chunks"] if x["locked"]])
    out = lam[o], np.asfortranarray(V[:, o]), info["thresholds"][o], locked
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, lam=out[0], V=out[1], t=out[2], locked=locked)
    log(f"cached {len(lam)} modes in {path}")
    return out


def deflated(op, free, P, sigma, cols, tau=TAU, maxit=20000):
    """inner()'s solve deflated by the host columns cols: iterations, seconds, the explicit
    residual, stagnation, the setup and per-iteration seconds."""
    A, b, c = ap_operator(op, free, -sigma), rhs(op, free), si._chunk(6)
    t = time.perf_counter()
    Zd = si._zblock(cols, op.n_dof, c)
    H = si._deflate_setup(A, Zd, len(cols), c)
    H.block_until_ready()
    t_setup = time.perf_counter() - t
    si._dminres(A, b, P, Zd, H, 0.0, 10)[0].block_until_ready()  # compile
    t = time.perf_counter()
    si._dminres(A, b, P, Zd, H, 0.0, 10)[0].block_until_ready()
    t_it = (time.perf_counter() - t) / 10
    t = time.perf_counter()
    x, inf = si._dminres(A, b, P, Zd, H, tau, maxit)
    x.block_until_ready()
    s = time.perf_counter() - t
    r = jnp.linalg.norm(b - A(x), axis=0) / jnp.linalg.norm(b, axis=0)
    its = int(np.asarray(inf["iterations"]).max())
    return dict(n_z=len(cols), iterations=its, seconds=s, residual=float(r.max()),
                stagnated=bool(np.asarray(inf["stagnated"]).any()), setup_s=t_setup,
                s_per_it=t_it)  # fmt: skip


def solves(cache):
    """Per stored depth: deflated by all LOBPCG modes below sigma (u = 0) and by the nearest
    half, the undeflated seconds per iteration (10 iterations) for the overhead."""
    lam, V, _, _ = lobpcg_cache(cache)
    si_rows = json.loads((Path(harness.RESULTS) / "shift_invert_study.json").read_text())
    rows = si_rows["grid_hex20"]["rows"]
    ref = {r["depth"]: r for r in rows if r["frac"] == 0.5 and not r.get("floor")}
    ref[1005] = si_rows["recheck_1005"]
    op, free, build = setup(hex20(16))
    out = {}
    for depth in (503, 1005):
        sigma = ref[depth]["sigma"]
        below = np.flatnonzero(lam < sigma)
        log(f"depth {depth}: sigma {sigma:.6g}, {len(below)} modes below")
        P = build(op, sigma)
        A, b = ap_operator(op, free, -sigma), rhs(op, free)
        minres(A, b, P, tol=0.0, maxit=10)[0].block_until_ready()
        t = time.perf_counter()
        minres(A, b, P, tol=0.0, maxit=10)[0].block_until_ready()
        t_plain = (time.perf_counter() - t) / 10
        res = dict(sigma=sigma, depth_below=len(below), plain_iterations=ref[depth]["iterations"],
                   plain_s_per_it=t_plain)  # fmt: skip
        for name, idx in (("full", below), ("half", below[len(below) // 2 :])):
            log(f"depth {depth}: deflated by {len(idx)} ({name})")
            res[name] = deflated(op, free, P, sigma, [V[:, i] * free for i in idx])
            res[name]["ratio"] = res[name]["iterations"] / ref[depth]["iterations"]
            res[name]["overhead"] = res[name]["s_per_it"] / t_plain
            log(json.dumps(res[name]))
        out[f"depth_{depth}"] = res
        jax.clear_caches()
    f = out["depth_1005"]["full"]  # the acceptance: <= 0.1x, no stagnation, residual <= 1.01 tau
    out["accepted"] = f["ratio"] <= 0.1 and not f["stagnated"] and f["residual"] <= 1.01 * TAU
    return out


def band(cache):
    """The junction's cut path: C_max = 960 at chunk 40 (D = 964 predicted from the cached
    chunks), lo ~ mode 952, hi ~ 1005; the predicted cut and full deflation checked before
    LOBPCG runs, per slice the MINRES record, the band against eigsh."""
    lam_c, _, t_c, locked = lobpcg_cache(cache)
    mesh = hex20(16)
    n, free = 3 * len(mesh.points), setup(mesh)[1]
    k = _block_size(CHUNK, CHUNK, None)[2]
    cum = np.cumsum(locked)
    D = int(cum[np.flatnonzero(cum[:-1] + k > C_MAX)[0] + 1])  # the stopping chunk kept
    lo_i, hi_i = widest(lam_c, D - 14, D - 10), widest(lam_c, 1000, 1010)
    lo, hi = (lam_c[lo_i - 1] + lam_c[lo_i]) / 2, (lam_c[hi_i - 1] + lam_c[hi_i]) / 2
    cut = api._cut(lam_c[:D], t_c[:D], min(k, si._chunk(6)), lo, hi, 1e-8)
    assert cut is not None and cut > lo, (D, cut, lo)
    j = int(np.searchsorted(lam_c[:D], cut))
    op_m, pc = mem._models(mesh, "consistent", "auto", "float64", {}, None)

    kw = dict(chunk=CHUNK, guards=None, low=False, basis_dtype="float64", n_rigid=0)

    def cmax(B):
        return mem._c_max(op_m, pc, n, **kw, n_avail=int(free.sum()), budget=B, host_budget=None)

    a, B = 1e8, 1e11  # the smallest budget with C_max >= 960
    while B - a > 1e-6 * B:
        mid = (a + B) / 2
        a, B = (mid, B) if (cmax(mid) or 0) < C_MAX else (a, mid)
    assert cmax(B) == C_MAX
    n_new = api._n_new(None, 0, D, j)
    C, d = mem._slice_plan(op_m, pc, n, n_store=j + n_new, n_new=n_new, block=6, n_rigid=0,
                           budget=B, host_budget=None, max_deflate=j + n_new,
                           host_extra=8 * n * (D - j))  # fmt: skip
    assert C is not None and d >= j, (C, d, j)
    log(f"budget {B / 1e9:.3f} GB: D {D}, cut {cut:.6g} (j {j}), lo {lo:.6g} (mode {lo_i}),"
        f" hi {hi:.6g} (mode {hi_i}), slice C {C}, d {d}")  # fmt: skip
    seen, real_solve, real_si = {}, api._solve, api.si_band

    def lob(*a_, **kw):
        out = real_solve(*a_, **kw)
        seen.update(D=len(out[0]), t_lobpcg=sum(out[2]["time"].values()))
        return out

    def slices(*a_, **kw):
        seen.update(max_deflate=kw["max_deflate"], j=len(kw["known"][0]))
        log(f"slices from {a_[2]:.6g}: {seen}")
        return real_si(*a_, **kw, verbose=True)

    api._solve, api.si_band = lob, slices
    f1, f2 = (math.sqrt(x) / (2 * math.pi) for x in (lo, hi))
    t = time.perf_counter()
    try:
        lam, _, info = solve_modes(mesh, MAT, BCS, band=(f1, f2), chunk=CHUNK, levers=False,
                                   memory_budget=B, tol=1e-8, verify=False)  # fmt: skip
    finally:
        api._solve, api.si_band = real_solve, real_si
    wall = time.perf_counter() - t
    assert info["reason"] == "capacity" and seen["D"] == D, (info["reason"], seen)
    jn = info["junction"]
    assert jn["known"] == j and abs(jn["cut"] - cut) <= 1e-6 * cut, (jn, cut, j)
    keep = ("interval", "kind", "n_new", "deflated", "minres_iterations", "sigma", "sigma_moved",
            "move_iterations", "inner_maxit", "steps", "basis", "reruns", "converged",
            "reason")  # fmt: skip
    rec = [{x: s.get(x) for x in keep} | {"time": s.get("time", {}).get("total")}
           for s in info["slices"]]  # fmt: skip
    log(f"band: {len(lam)} modes in {wall / 60:.1f} min; checking against eigsh")
    K, M = assemble(mesh, MAT)
    f = np.flatnonzero(free)
    width = int(((lam_c >= lo) & (lam_c <= hi)).sum())
    ref = spla.eigsh(K[f][:, f].tocsc(), width + 20, M[f][:, f].tocsc(), sigma=(lo + hi) / 2,
                     which="LM", maxiter=10000)[0]  # fmt: skip
    ref = np.sort(ref[(ref >= lo) & (ref <= hi)])
    err = float(np.abs(lam - ref).max() / hi) if len(lam) == len(ref) else None
    return dict(budget=B, D=D, cut=cut, j=j, lo=lo, hi=hi, lo_mode=lo_i, hi_mode=hi_i,
                plan=dict(C=C, d=d), max_deflate=seen["max_deflate"], n_modes=len(lam),
                n_ref=len(ref), error=err, wall=wall, t_lobpcg=seen["t_lobpcg"],
                time=info["time"], converged=info["converged"], slices=rec)  # fmt: skip


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    a = p.parse_args()
    config.enable_x64()
    path = Path(a.out) / "deep_band_study.json"
    r = json.loads(path.read_text()) if path.exists() else {}
    r.pop("versions", None)
    walls = r.pop("walls", {})
    for name in a.parts:
        log(f"== {name}")
        t = time.perf_counter()
        r[name] = dict(solves=solves, band=band)[name](a.cache)
        walls[name] = time.perf_counter() - t
        log(f"== {name} done in {walls[name] / 60:.1f} min")
        harness.write_result("deep_band_study", r | dict(walls=walls), a.out)
        jax.clear_caches()
