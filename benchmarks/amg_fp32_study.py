"""fp32 AMG study (PR 18c): PCG iterations and true residuals with fp64 and fp32 V-cycles, V-cycle
times; the calibration of amg_memory's fp32 terms: owned device bytes (storage), compile-only
memory_analysis of the LOBPCG steps and the polish (hex and solid-shell grids plus held-out
points), the fits (storage ratios, linear-programming envelopes with asymptote rows) and their
gates; fp32 build peaks in fresh subprocesses; the 1M hex20 and 791k plate acceptance runs with
precond_dtype "float32". Per-step progress is printed; --parts selects sections. Writes
results/amg_fp32_study.json (a run of some parts keeps the others' entries).
"""

from jaxmodal import config

config.enable_x64()  # before any JAX constant (the test helpers)

import argparse
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import amg_study, harness
from benchmarks import shell_solve_study as SSS

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
import test_shell_amg as T
from helpers import STEEL, owned_bytes, step_bytes

from jaxmodal import memory as mem
from jaxmodal.linalg import pcg
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import ap_operator, dof_mask, unstructured
from jaxmodal.precond import amg, jacobi, schwarz

PARTS = ("accuracy", "storage", "steps_hex", "steps_shell", "fit", "peaks", "acceptance")
SIGMA = 1e6  # the 13b hex shift (xmin clamped steel boxes)
MS = (6, 15, 25, 60, 120)
# the hex grid (13b's sizes plus hex27 32^3) and the held-out points (size, block widths)
HEX = {"hex20": (8, 12, 16, 24, 32, 44), "hex27": (6, 12, 16, 24, 32),
       "hex8": (12, 24, 32, 48, 68)}  # fmt: skip
HEX_HELD = [("hex20", 52, (10, 25, 40, 120)), ("hex8", 80, (10, 25, 40, 120)),
            ("hex27", 40, (10, 25)), ("hex20", 44, (10,))]  # fmt: skip
HEX_STORAGE = {"hex20": (8, 12, 16, 24, 32, 44), "hex27": (6, 12, 16, 24),
               "hex8": (12, 24, 32, 48, 68)}  # fmt: skip
# shell held-out points (case, size, m); flat 512^2 also gives the m = 10 report
SHELL_HELD = [("stack3", 256, 120), ("T", 96, 120), ("flat", 512, 10), ("flat", 512, 25)]
LONG = {"hex8": "hex8", "hex20": "p", "hex27": "p", "flat": "shell", "block": "shell"}
GATE = 1.3  # model <= GATE x measured on the tight cases (wider gates are logged)
TIGHT = [("hex20", 24, 25), ("flat", 128, 25), ("flat", 128, 120), ("stack3", 128, 25),
         ("stack3", 128, 120)]  # fmt: skip
REPORT = [("hex20", 44, m) for m in (6, 10, 120)] + [("flat", 512, m) for m in (6, 10, 120)]


def hex_case(t, n):
    mesh = box_mesh((n,) * 3, (1 / n,) * 3, type=t)
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"})
    return mesh, op, free


def hex_exact(mesh):
    """fp32 exact storage: free mask (fp64) and node-block Jacobi (fp32) per node, the fp64
    transfer tables; and the fp64 fitted part."""
    fam, n_c, tables = mem._amg_scan(mesh)
    return fam, 72 * len(mesh.points) + tables, mem.AMG_STORAGE[fam] * n_c


def shell_exact(mesh):
    sizes, n_s, n_h, n_f = mem._shell_scan(mesh)
    st = mem.SHELL_STORAGE
    return schwarz.nbytes(sizes, "float32"), st["shell"] * n_s + st["fiber"] * n_f + st["hex"] * n_h


def split_bytes(P, op):
    """(float, int) owned device bytes."""
    shared = {id(x) for x in jax.tree.leaves(op)}
    leaves = {id(x): x for x in jax.tree.leaves(P) if isinstance(x, jax.Array)}
    f = sum(x.nbytes for i, x in leaves.items() if i not in shared and x.dtype.kind == "f")
    return f, owned_bytes(P, op) - f


# ---------------------------------------------------------------- accuracy


def _pcg(A, b, P, tol):
    x, info = pcg(A, b, P, tol, 3000)
    res = np.linalg.norm(np.asarray(b - A(x)), axis=0) / np.linalg.norm(np.asarray(b), axis=0)
    return dict(iterations=np.asarray(info["iterations"]).tolist(), true_res=res.max().item())


def _best(f, x, k=5):
    jax.block_until_ready(f(x))
    t = []
    for _ in range(k):
        t0 = time.perf_counter()
        jax.block_until_ready(f(x))
        t.append(time.perf_counter() - t0)
    return min(t)


def accuracy(cases=None):
    """PCG on A_P (fp64) with the fp64 and fp32 V-cycles, 4 masked random columns: iterations
    and the true relative residual per tolerance; warm V-cycle times (best of 5) at m = 25.
    cases: (name, make() -> (mesh, op, free), sigma, tols)."""
    ss = {"xmin": "xyz", "xmax": "xyz"}  # two edges clamped
    cases = cases or [
        ("hex20 16^3", lambda: hex_case("hex20", 16), 3e5, (1e-8, 1e-10)),
        ("hex27 12^3", lambda: hex_case("hex27", 12), 3e5, (1e-8, 1e-10)),
        ("hex8 32^3", lambda: hex_case("hex8", 32), 3e5, (1e-8, 1e-10)),
        ("plate 128^2 L/t 10", lambda: _shell(T.plate(128, 10)[0], ss), 1.0, (1e-8, 1e-10)),
        ("plate 128^2 L/t 1000", lambda: _shell(T.plate(128, 1000)[0], ss), 1.0, (1e-6, 1e-8)),
        ("stack3 128^2", lambda: _shell(SSS.MESHES["stack3"](128), ss), 1.0, (1e-8,)),
        ("T 48^2", lambda: _shell(SSS.MESHES["T"](48), T.CLAMPED), 1.0, (1e-8,)),
    ]
    out = []
    for name, make, sigma, tols in cases:
        mesh, op, free = make()
        A = ap_operator(op, free, sigma)
        rng = np.random.default_rng(1)
        b = jnp.asarray(rng.standard_normal((op.n_dof, 4)) * np.asarray(free)[:, None])
        X = jnp.asarray(rng.standard_normal((op.n_dof, 25)))
        r = dict(case=name, n_dof=op.n_dof, sigma=sigma)
        for dtype in ("float64", "float32"):
            P = amg(op, free, sigma, mesh.points, dtype=dtype)
            r[dtype] = {f"{tol:g}": _pcg(A, b, P, tol) for tol in tols}
            r[dtype]["vcycle_s"] = _best(P, X)
            P = None
        print(json.dumps(r), flush=True)
        out.append(r)
        jax.clear_caches()
    return out


def _shell(mesh, bcs):
    return mesh, unstructured(mesh, STEEL), dof_mask(mesh, bcs)


# ---------------------------------------------------------------- storage


def storage(hex_sizes=HEX_STORAGE, shell=SSS.STORAGE):
    """fp32 owned device bytes (float / int), the exact parts, the coarsest factor and the fp64
    fitted part, per hex and shell storage case."""
    out = []
    for t, sizes in hex_sizes.items():
        for n in sizes:
            mesh, op, free = hex_case(t, n)
            out.append(_storage_hex(t, n, mesh, op, amg(op, free, SIGMA, mesh.points,
                                                        dtype="float32")))  # fmt: skip
            jax.clear_caches()
    for case, n in shell:
        mesh, op, _, P = SSS.build(case, n, dtype="float32")
        f, i = split_bytes(P, op)
        exact, fitted = shell_exact(mesh)
        r = dict(case=case, size=n, fam="shell", n_dof=op.n_dof, owned=f + i, float=f, int=i,
                 exact=exact, coarse=P.coarse.L.nbytes, fitted64=fitted)  # fmt: skip
        print(json.dumps(r), flush=True)
        out.append(r)
        jax.clear_caches()
    return out


def _storage_hex(t, n, mesh, op, P):
    f, i = split_bytes(P, op)
    fam, exact, fitted = hex_exact(mesh)
    r = dict(case=t, size=n, fam=fam, n_dof=op.n_dof, owned=f + i, float=f, int=i, exact=exact,
             coarse=P.coarse.L.nbytes, fitted64=fitted)  # fmt: skip
    print(json.dumps(r), flush=True)
    return r


# ---------------------------------------------------------------- steps


def steps_hex(grid=HEX, ms=MS, held=HEX_HELD):
    """Compile only, per hex case and block width: the fp32-AMG and Jacobi steps (9-block, low;
    step_bytes, no basis), the fp32 polish (and at the held-out points the fp64 polish and the
    fp32 owned bytes) and the operator workspace, in bytes."""
    todo = [(t, n, ms, False) for t, sizes in grid.items() for n in sizes]
    todo += [(t, n, widths, True) for t, n, widths in held]
    out = []
    for t, n, widths, is_held in todo:
        mesh, op, free = hex_case(t, n)
        P = amg(op, free, SIGMA, mesh.points, dtype="float32")
        J = jacobi(op, free, 0.0)
        st = _storage_hex(t, n, mesh, op, P) if is_held else None
        rows = []
        for m in widths:
            r = dict(case=t, size=n, n_dof=op.n_dof, m=m, held=is_held,
                     ws=mem.unstructured_memory(mesh, m)["workspace"],
                     polish=SSS.polish(op, free, P, m))  # fmt: skip
            for low, key in ((False, "9-block"), (True, "low")):
                r[key] = step_bytes(op, P, low, m)
                r["jacobi_" + key] = step_bytes(op, J, low, m)
            rows.append(r)
        if is_held:  # the fp64 polish against the fp64 low model
            P = None
            P = amg(op, free, SIGMA, mesh.points)
            for r in rows:
                r["polish64"] = SSS.polish(op, free, P, r["m"])
                r["storage"] = st
        for r in rows:
            print(json.dumps(r), flush=True)
        out += rows
        P = J = None
        jax.clear_caches()
    return out


def steps_shell(cases=SSS.FIT, ms=SSS.MS, extra=SSS.EXTRA, held=SHELL_HELD):
    """Compile only, per shell case and block width: shell_solve_study.measure (9-block, low,
    polish) with the fp32 AMG and the operator workspace, in bytes."""
    todo = {}
    for c, n in cases:
        todo.setdefault((c, n, 2), []).extend((m, False) for m in ms)
    for c, n, m, d in extra:
        todo.setdefault((c, n, d), []).append((m, False))
    for c, n, m in held:
        todo.setdefault((c, n, 2), []).append((m, True))
    out = []
    for (case, n, degree), widths in todo.items():
        mesh, op, free, P = SSS.build(case, n, degree, dtype="float32")
        for m, is_held in widths:
            r = dict(case=case, size=n, degree=degree, n_dof=op.n_dof, m=m, held=is_held,
                     ws=mem.unstructured_memory(mesh, m)["workspace"])  # fmt: skip
            r |= SSS.measure(op, free, P, m)
            print(json.dumps(r), flush=True)
            out.append(r)
        P = None
        jax.clear_caches()
    return out


# ---------------------------------------------------------------- fit


def ceil_to(x, q):
    return math.ceil(round(x / q, 9)) * q


def rounded(fam, x):
    """Stored constants from an LP fit (a, c, K): c + 0.5 against rounding, then rounded up
    (hex: a to 0.01, c to 0.1, K to 100 DOF; shell: a and c to 0.1, K0 to 1 MB)."""
    a, c, K = x
    if fam == "shell":
        return (ceil_to(a, 0.1), ceil_to(c + 0.5, 0.1), ceil_to(K, 1e6))
    return (ceil_to(a, 0.01), ceil_to(c + 0.5, 0.1), ceil_to(K, 100))


def ratio32(rows):
    """STORAGE32 per family: the maximum fp32 fitted-part bytes over the fp64 fitted part,
    rounded up to 3 decimals."""
    out = {}
    for r in rows:
        x = (r["owned"] - r["exact"] - r["coarse"]) / r["fitted64"]
        out[r["fam"]] = max(out.get(r["fam"], 0.0), x)
    return {k: ceil_to(v, 1e-3) for k, v in out.items()}


def targets(r, fam, key):
    """(row, y) in blocks of 8 n m: the pc term's target net of the Jacobi step (hex) or of
    STEP_FIT's part (shell; the polish with the low f), and its LP row."""
    b, m, n = 8 * r["n_dof"] * r["m"], r["m"], r["n_dof"]
    fl = mem.STEP_FIT["low"][1]
    if key == "polish":
        y = (r["polish"] - fl * r["ws"]) / b
    elif fam == "shell":
        A, f = mem.STEP_FIT[key]
        y = (r[key] - f * r["ws"]) / b - A
    else:
        y = (r[key] - r["jacobi_" + key]) / b
    return [1, 1 / m, 1 / (8 * n * m) if fam == "shell" else 1 / n], y


def asymptotes(rows, fam, key):
    """One row a + c / m >= y_inf per (long series, m): y_inf = max(y(n2), the 1/n
    extrapolation through n2 and the largest n1 <= n2 / 2)."""
    out = []
    for s in [s for s, f in LONG.items() if f == fam]:
        for m in sorted({r["m"] for r in rows if r["case"] == s}):
            pts = sorted((r["n_dof"], targets(r, fam, key)[1]) for r in rows
                         if r["case"] == s and r["m"] == m)  # fmt: skip
            n2, y2 = pts[-1]
            low = [p for p in pts if p[0] <= n2 / 2]
            if not low:
                continue
            n1, y1 = low[-1]
            y = max(y2, (n2 * y2 - n1 * y1) / (n2 - n1))
            out.append(dict(case=s, m=m, n1=n1, n2=n2, y=y))
    return out


def _model(fam, x, r):
    a, c, K = x
    m, n = r["m"], r["n_dof"]
    return a + c / m + (K / (8 * n * m) if fam == "shell" else K / n)


def fit(data, tight=TIGHT, report=REPORT):
    """The fp32 constants from the storage and steps entries, and their gates: every term >= the
    target on every fit and held-out point, the ratio of lobpcg_memory's step to the measured
    step on TIGHT (<= GATE, or widened to the ratio rounded up to 0.05, at most 1.4, flagged),
    the REPORT ratios, the hex fp64 low model minus the fp64 polish (blocks)."""
    st = data["storage"] + [r["storage"] for r in data["steps_hex"] if r.get("storage")]
    out = dict(storage32=ratio32(data["storage"]), consts={}, rows={}, margins={},
               gates={}, gate_ok={}, report=[], polish64_slack=[])  # fmt: skip
    hexr = data["steps_hex"]
    shellr = [r for r in data["steps_shell"] if r["degree"] == 2]
    series = [(fam, key) for fam in ("hex8", "p") for key in ("9-block", "low", "polish")]
    series += [("shell", key) for key in ("9-block", "low", "polish")]
    for fam, key in series:
        rows = shellr if fam == "shell" else [r for r in hexr if LONG[r["case"]] == fam]
        fitr = [r for r in rows if not r["held"]]
        F, y = map(list, zip(*(targets(r, fam, key) for r in fitr), strict=True))
        asy = asymptotes(fitr, fam, key)
        F += [[1, 1 / a["m"], 0] for a in asy]
        y += [a["y"] for a in asy]
        lp = SSS._envelope(F, y).tolist()
        x = rounded(fam, lp)
        name = f"{fam}/{key}"
        out["consts"][name] = dict(lp=lp, stored=x, asymptotes=asy)
        out["margins"][name] = min(_model(fam, x, r) - targets(r, fam, key)[1] for r in rows)
    # gates: storage model >= owned everywhere
    sm = []
    for r in st:
        rho = out["storage32"][r["fam"]]
        model = r["exact"] + rho * r["fitted64"] + r["coarse"]
        sm.append(model / r["owned"])
    out["storage_min_ratio"] = min(sm)
    out["gate_ok"]["storage"] = min(sm) >= 1
    out["gate_ok"]["steps"] = all(v >= 0 for v in out["margins"].values())
    allr = hexr + shellr
    for case, n, m in tight:
        (r,) = [r for r in allr if (r["case"], r["size"], r["m"]) == (case, n, m)]
        fam = "shell" if case in SSS.MESHES else LONG[case]
        for key in ("9-block", "low", "polish"):
            name = f"{fam}/{key}"
            q = total(fam, out["consts"][name]["stored"], r, key) / r[key]
            g = out["gates"].setdefault(name, GATE)
            if q > g:
                out["gates"][name] = max(g, ceil_to(q, 0.05))
            out["rows"].setdefault(name, []).append(dict(case=case, size=n, m=m, ratio=q))
    out["gate_ok"]["upper"] = all(g <= 1.4 for g in out["gates"].values())
    for case, n, m in report:
        hit = [r for r in allr if (r["case"], r["size"], r["m"]) == (case, n, m)]
        if hit:
            r, fam = hit[0], "shell" if case in SSS.MESHES else LONG[case]
            out["report"].append(dict(case=case, size=n, m=m, **{
                key: total(fam, out["consts"][f"{fam}/{key}"]["stored"], r, key) / r[key]
                for key in ("9-block", "low", "polish")}))  # fmt: skip
    for r in (r for r in hexr if "polish64" in r):
        fam, b = LONG[r["case"]], 8 * r["n_dof"] * r["m"]
        al, cl, Kl = mem.AMG_STEP[(fam, "low")]
        Al, fl = mem.STEP_FIT["low"]
        low = Al * b + fl * r["ws"] + 8 * r["n_dof"] * (al * r["m"] + cl) + 8 * r["m"] * Kl
        out["polish64_slack"].append(dict(case=r["case"], size=r["size"], m=r["m"],
                                          blocks=(low - r["polish64"]) / b))  # fmt: skip
    print(json.dumps(out), flush=True)
    return out


def total(fam, x, r, key):
    """lobpcg_memory's step term (A b + f ws + the pc term; the polish fl ws + its term)."""
    b = 8 * r["n_dof"] * r["m"]
    pc = _model(fam, x, r) * b
    if key == "polish":
        return mem.STEP_FIT["low"][1] * r["ws"] + pc
    A, f = mem.STEP_FIT[key]
    return A * b + f * r["ws"] + pc


# ---------------------------------------------------------------- peaks

HEX_BUILD = """
t = time.perf_counter()
jax.block_until_ready(amg(op, free, 1e6, mesh.points, dtype="float32"))
json.dump(dict(time_s=time.perf_counter() - t), open("{out}", "w"))
"""
SHELL_BUILD = SSS.BUILD.replace("mesh.points)", 'mesh.points, dtype="float32")')
REBUILD_SETUP = """
import gc
from jaxmodal.eigen.api import _nbytes
P, agg = amg(op, free, 1.0, mesh.points, return_aggregates=True)
P = None
gc.collect()
"""
REBUILD = """
import json, time
t = time.perf_counter()
jax.block_until_ready(amg(op, free, 2.0, mesh.points, aggregates=agg))
json.dump(dict(time_s=time.perf_counter() - t, aggs=_nbytes(agg)), open("{out}", "w"))
"""
HEX_PEAKS = [("hex20", 12), ("hex20", 24), ("hex27", 12), ("hex8", 48)]


def peaks(shell=SSS.PEAKS, hexes=HEX_PEAKS, rebuild=(("flat", 256),)):
    """fp32 build host peaks above the built operator (fresh subprocesses), and the fp64 rebuild
    with the aggregates held (the sigma rounds), next to amg_memory's build."""
    root, out = str(harness.RESULTS.parents[1]), []
    for case, n in shell:
        s = SSS.SETUP.format(root=root, case=case, n=n)
        r = dict(case=case, size=n) | harness.measure_json(SHELL_BUILD, s)
        r["build"] = mem.amg_memory(SSS.MESHES[case](n), 6, dtype="float32")["build"]
        print(json.dumps(r), flush=True)
        out.append(r)
    for t, n in hexes:
        r = dict(case=t, size=n) | harness.measure_json(
            HEX_BUILD, amg_study.SETUP.format(root=root, t=t, n=n)
        )
        r["build"] = mem.amg_memory(box_mesh((n,) * 3, type=t), 15, dtype="float32")["build"]
        print(json.dumps(r), flush=True)
        out.append(r)
    for case, n in rebuild:
        s = SSS.SETUP.format(root=root, case=case, n=n) + REBUILD_SETUP
        r = dict(case=case, size=n, rebuild=True) | harness.measure_json(REBUILD, s)
        r["build"] = mem.amg_memory(SSS.MESHES[case](n), 6)["build"]
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


# ---------------------------------------------------------------- acceptance


def band_cmax(mesh, bcs, chunks=(40, 20, 10, 5)):
    """The band C_max per chunk with fp64 and fp32 AMG at the default budget (estimators)."""
    budget, host = mem._default_budget()
    free = dof_mask(mesh, bcs)
    n = 3 * len(mesh.points)
    out = {}
    for dtype in ("float64", "float32"):
        op_m, pc_m = mem._models(mesh, "consistent", "auto", dtype, None, None)
        out[dtype] = {
            c: mem._c_max(
                op_m,
                pc_m,
                n,
                chunk=c,
                guards=None,
                low=False,
                basis_dtype="float64",
                n_rigid=0,
                n_avail=int(free.sum()),
                budget=budget,
                host_budget=host,
            )  # fmt: skip
            for c in chunks
        }
    return dict(budget=budget, host_budget=host, c_max=out)


def acceptance():
    """The 1M hex20 (amg_study.acceptance) and the 791k plate (shell_solve_study.solve_peak) with
    precond_dtype "float32", the fp64 records beside them, and the plate's band C_max."""
    res = harness.RESULTS
    hex64 = json.loads((res / "amg_study.json").read_text())["acceptance"]
    shell64 = json.loads((res / "shell_solve_study.json").read_text())["solve_peak"]
    out = dict(hex20=amg_study.acceptance(44, dtype="float32"), hex20_fp64=hex64)
    jax.clear_caches()
    out |= dict(plate=SSS.solve_peak("flat", 362, 20, dtype="float32"), plate_fp64=shell64)
    out["plate_band"] = band_cmax(SSS.MESHES["flat"](362), SSS.SS)
    print(json.dumps(out["plate_band"]), flush=True)
    return out


def main(parts=PARTS, out=harness.RESULTS, write=True):
    path = Path(out) / "amg_fp32_study.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.pop("versions", None)
    for part in parts:
        print(f"--- {part}", flush=True)
        data[part] = fit(data) if part == "fit" else globals()[part]()
    if write:
        harness.write_result("amg_fp32_study", data, out)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    p.add_argument("--out", default=harness.RESULTS)
    a = p.parse_args()
    main(a.parts, a.out)
