"""Solid-shell solve study (PR 18b): solve_modes with shell AMG (20 modes; iterations, time,
the low-memory switch and the polish per chunk, lambda errors against eigsh where small), the
SIMP GMG stagnation case of docs/open_items.md, and the calibration of amg_memory's shell
family: compile-only memory_analysis of the LOBPCG steps, the polish PCG, the count recurrence
and MINRES at every fit case and block width; device storage per component; host setup peaks
in fresh subprocesses (above the built operator); the fit of the constants (linear-programming
upper envelopes). Per-step progress is printed; --parts selects sections. Writes
results/shell_solve_study.json (a run of some parts keeps the others' entries).
"""

from jaxmodal import config

config.enable_x64()  # before any JAX constant (the test helpers)

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct as S
from scipy.optimize import linprog

from benchmarks import harness

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
import test_shell_amg as T
from exact import SIMPLY_SUPPORTED as SS
from helpers import STEEL, owned_bytes
from test_lobpcg import step_args

from jaxmodal.eigen import counting, reference_modes, solve_modes
from jaxmodal.linalg import krylov
from jaxmodal.memory import (
    COUNT_STEP,
    SI_MINRES,
    STEP_FIT,
    _shell_scan,
    solve_modes_memory,
    unstructured_memory,
)
from jaxmodal.operators import ap_operator, assemble, dof_mask, unstructured
from jaxmodal.precond import amg, gmg, schwarz

LOB = importlib.import_module("jaxmodal.eigen.lobpcg")  # the module, as the tests name it
PARTS = ("accuracy", "steps", "count", "storage", "peaks", "solve_peak", "fit")
MESHES = {
    "flat": lambda n: T.plate(n, 100)[0],
    "web": lambda n: T.web(n, 100)[0],
    "block": lambda n: T.block(n)[0],
    "stack3": lambda n: T.plate(n, 100, layers=3)[0],
    "stack10": lambda n: T.plate(n, 100, layers=10)[0],
    "T": lambda n: T.junction("T", n, 100, 3, 3)[0],
    "L": lambda n: T.junction("L", n, 100)[0],
    "cylinder": lambda n: T.cylinder(False),
}
# the fit cases (25k-396k DOF) and block widths of the step, low step and polish
FIT = [("flat", 64), ("flat", 128), ("flat", 256), ("web", 64), ("block", 32), ("stack10", 64),
       ("stack3", 128)]  # fmt: skip
MS = (6, 8, 15, 25, 60, 120)
# compile-only extras (case, size, m, degree): T, degree 3 and the 791k plate
EXTRA = (("T", 48, 15, 2), ("flat", 128, 15, 3), ("flat", 362, 6, 2), ("flat", 362, 120, 2))
COUNT = [("flat", 64), ("flat", 128), ("stack3", 128)]
STORAGE = FIT + [("T", 48), ("L", 48), ("cylinder", 0), ("flat", 48), ("block", 64),
                 ("flat", 362)]  # fmt: skip
PEAKS = [("flat", 128), ("flat", 256), ("flat", 362), ("flat", 512), ("stack3", 128),
         ("stack10", 64), ("T", 48), ("block", 32), ("block", 48), ("block", 64)]  # fmt: skip


def build(case, n, degree=2):
    """(mesh, op, free, P): AMG at sigma 1 with 4 nodes pinned (A_P(1) SPD, no rigid modes)."""
    mesh = MESHES[case](n)
    op = unstructured(mesh, STEEL)
    free = np.ones(op.n_dof, bool)
    free[:12] = False
    return mesh, op, jnp.asarray(free), amg(op, free, 1.0, mesh.points, degree=degree)


def step(op, P, low, k, cap=0):
    """Temporaries + aliased state of the compiled LOBPCG step (block k, basis capacity cap)."""
    c, _ = LOB._capacity(100, 1, k, low)
    f = LOB.lobpcg_step_low if low else LOB.lobpcg_step
    args = list(step_args(op, op.n_dof, k, low, cap))
    args[1], args[3] = op, P
    mem = f.lower(*args, c=c).compile().memory_analysis()
    return mem.temp_size_in_bytes + mem.alias_size_in_bytes


def polish(op, free, P, k):
    """The polish: _pcg's temporaries + output, and X and b."""
    A, b = ap_operator(op, free, 1.0), S((op.n_dof, k), jnp.float64)
    mem = krylov._pcg.lower(A, P, b, 1e-10, 500).compile().memory_analysis()
    return mem.temp_size_in_bytes + mem.output_size_in_bytes + 2 * 8 * op.n_dof * k


def measure(op, free, P, m):
    """The 9-block and low steps (max over no basis and a basis of capacity c ceil(2m / c)) and
    the polish at block width m, in bytes."""
    out = {"polish": polish(op, free, P, m)}
    for low in (False, True):
        c = m if low else max(m, min(4 * m, 64))
        out["low" if low else "9-block"] = max(
            step(op, P, low, m), step(op, P, low, m, c * -(-2 * m // c))
        )
    return out


def steps(cases=FIT, ms=MS, extra=EXTRA):
    """Per case and block width: measure's entries and the operator workspace, in bytes."""
    out, todo = [], [(c, n, ms, 2) for c, n in cases] + [(c, n, (m,), d) for c, n, m, d in extra]
    for case, n, widths, degree in todo:
        mesh, op, free, P = build(case, n, degree)
        for m in widths:
            ws = unstructured_memory(mesh, m)["workspace"]
            r = dict(case=case, size=n, degree=degree, n_dof=op.n_dof, m=m, ws=ws)
            r |= measure(op, free, P, m)
            print(json.dumps(r), flush=True)
            out.append(r)
        jax.clear_caches()
    return out


def count(cases=COUNT, bs=(4, 8, 16, 32)):
    """The count recurrence's temporaries with the V-cycle at probe blocks bs, and the shell
    MINRES temporaries at b = 6 (A_P at a negative shift)."""
    out = []
    for case, n in cases:
        mesh, op, free, P = build(case, n)
        A, m = ap_operator(op, free, 1.0), jnp.asarray(free, float)[:, None]
        for b in bs:
            w0 = S((op.n_dof, b), jnp.float64)
            low = counting._recurrence.lower(A, P, m, w0, jnp.zeros(64), 3, 1.0, 1e-8, 100)
            r = dict(case=case, size=n, n_dof=op.n_dof, b=b,
                     ws=unstructured_memory(mesh, b)["workspace"],
                     recurrence=low.compile().memory_analysis().temp_size_in_bytes)  # fmt: skip
            if b == 4 and case == "flat":
                Ai = ap_operator(op, free, -1.0)
                mem = krylov._minres.lower(Ai, P, S((op.n_dof, 6), jnp.float64), 1e-8, 100)
                r["minres_b6"] = mem.compile().memory_analysis().temp_size_in_bytes
                r["ws_b6"] = unstructured_memory(mesh, 6)["workspace"]
            print(json.dumps(r), flush=True)
            out.append(r)
        jax.clear_caches()
    return out


def storage(cases=STORAGE):
    """Device bytes per component: level-0 Schwarz (and schwarz.nbytes of the scan's sizes),
    T_0 (P_0 and P_0^T), the coarse levels and factor, the rest, the total (owned leaves)."""
    out = []
    for case, n in cases:
        mesh, op, free, P = build(case, n)
        A = P.levels[0][0]
        sizes, n_shell, n_hex, n_s = _shell_scan(mesh)
        r = dict(case=case, size=n, n_dof=op.n_dof, n_shell=n_shell, n_hex=n_hex, n_s=n_s,
                 schwarz0=owned_bytes(P.levels[0][1], (op, A)),
                 scan=schwarz.nbytes(sizes), T0=owned_bytes(P.transfers[0], op),
                 coarse_factor=owned_bytes(P.coarse, op), n_levels=len(P.levels),
                 total=owned_bytes(P, op))  # fmt: skip
        r["coarse"] = sum(owned_bytes(L, op) for L in P.levels[1:]) + sum(
            owned_bytes(t, op) for t in P.transfers[1:]
        )
        r["rest"] = r["total"] - r["schwarz0"] - r["T0"] - r["coarse"] - r["coarse_factor"]
        print(json.dumps(r), flush=True)
        out.append(r)
        jax.clear_caches()
    return out


SETUP = """
import sys
sys.path[:0] = [{root!r}, {root!r} + "/tests"]
import jax
from jaxmodal import config
config.enable_x64()
from benchmarks.shell_solve_study import MESHES
from helpers import STEEL, owned_bytes
from jaxmodal.operators import unstructured
from jaxmodal.precond import amg
import numpy as np
mesh = MESHES[{case!r}]({n})
op = unstructured(mesh, STEEL)
free = np.ones(op.n_dof, bool)
free[:12] = False
jax.block_until_ready(op)
"""
BUILD = """
import json, time
t = time.perf_counter()
P = amg(op, free, 1.0, mesh.points)
jax.block_until_ready(P)
json.dump(dict(time_s=time.perf_counter() - t, owned=owned_bytes(P, op)), open("{out}", "w"))
"""
SOLVE = """
import json, time
from exact import SIMPLY_SUPPORTED
from jaxmodal.eigen import solve_modes
t = time.perf_counter()
lam, V, info = solve_modes(mesh, STEEL, SIMPLY_SUPPORTED, N_MODES, tol=1e-6)
keys = ("iterations", "converged", "precond")
json.dump({{k: info[k] for k in keys}} | dict(time_s=time.perf_counter() - t), open("{out}", "w"))
"""


def peaks(cases=PEAKS):
    """Host peak of the AMG setup above the built operator (fresh subprocesses, macOS)."""
    root, out = str(harness.RESULTS.parents[1]), []
    for case, n in cases:
        r = dict(case=case, size=n) | harness.measure_json(
            BUILD, SETUP.format(root=root, case=case, n=n)
        )
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


def solve_peak(case="flat", n=362, n_modes=20):
    """One cold solve_modes (tol 1e-6) in a fresh subprocess: its info, time and process peak
    above the built operator, next to solve_modes_memory's total."""
    root = str(harness.RESULTS.parents[1])
    # replace, not format: measure_json formats the code once more (its {out})
    code, setup = SOLVE.replace("N_MODES", str(n_modes)), SETUP.format(root=root, case=case, n=n)
    r = harness.measure_json(code, setup)
    est = solve_modes_memory(MESHES[case](n), SS, n_modes)["total"]
    r |= dict(case=case, size=n, n_modes=n_modes, estimate_bytes=est)
    print(json.dumps(r), flush=True)
    return r


def _solve(name, mesh, bcs, n_modes=20, tol=1e-6, ref=False, **kw):
    t = time.perf_counter()
    lam, _, info = solve_modes(mesh, STEEL, bcs, n_modes, tol=tol, maxit=3000, **kw)
    r = dict(case=name, n_dof=3 * len(mesh.points), tol=tol, precond=info["precond"],
             time_s=time.perf_counter() - t, iterations=info["iterations"],
             converged=info["converged"], stagnated=info["stagnated"],
             chunks=[dict(iterations=c["iterations"], switched=c["switched"],
                          polish=c["polish"], converged=c["converged"]) for c in info["chunks"]],
             worst_r_over_t=float((info["residuals"] / info["thresholds"]).max()))  # fmt: skip
    if ref:
        K, M = assemble(mesh, STEEL)
        free = dof_mask(mesh, bcs)
        lr = reference_modes(K, M, n_modes + info["n_rigid"], free)[0][info["n_rigid"] :]
        r["lam_rel_err"] = float(np.max(np.abs(lam - lr[: len(lam)]) / lr[: len(lam)]))
    print(json.dumps(r), flush=True)
    return r


def accuracy():
    """The accuracy / time cases (AMG; Jacobi where it converges), the switch and polish cases,
    the PR 9 tolerance cases and the SIMP GMG stagnation case."""
    out = []
    cases = [
        ("flat32-50", T.plate(32, 50)[0], SS), ("free24-50", T.plate(24, 50)[0], {}),
        ("T24-50", T.junction("T", 24, 50)[0], T.CLAMPED), ("flat64-100", T.plate(64, 100)[0], SS),
        ("curved32-50", T.plate(32, 50, curved=True)[0], SS),
        ("stack32x3-100", T.plate(32, 100, layers=3)[0], SS), ("web32-50", T.web(32, 50)[0], SS),
    ]  # fmt: skip
    for name, mesh, bcs in cases:
        out.append(_solve(name, mesh, bcs, ref=True))
        if name in ("flat32-50", "curved32-50", "web32-50"):
            out.append(_solve(name + "-jacobi", mesh, bcs, precond="jacobi"))
        out.append(_solve(name + "-tol1e-10", mesh, bcs, tol=1e-10, ref=True))
        jax.clear_caches()
    for n, Lt in ((32, 300), (32, 1000), (128, 300), (128, 1000)):
        mesh = T.plate(n, Lt)[0]
        out.append(_solve(f"flat{n}-{Lt}", mesh, SS, ref=n == 32))
        if Lt == 1000:
            out.append(_solve(f"flat{n}-{Lt}-nopolish", mesh, SS, polish_options={"on": False}))
        jax.clear_caches()
    out.append(_solve("flat32-1000-chunks10", T.plate(32, 1000)[0], SS, 30, chunk=10))
    out.append(_solve("free24-1000", T.plate(24, 1000)[0], {}, ref=True))
    out.append(simp())
    return out


def simp():
    """PR 11b's SIMP stagnation case: 10 modes at tol 1e-6, fp64 GMG at lambda_1."""
    from benchmarks import gmg_study as G

    op, free, mesh = G.case((32,) * 3, "clamped", None, "simp")
    s = G.lam1(op, free, mesh)
    t = time.perf_counter()
    _, _, info = LOB.lobpcg(op, free, 10, 1e-6, precond=gmg(op, free, s))
    r = dict(case="simp32-gmg", time_s=time.perf_counter() - t, iterations=info["iterations"],
             converged=info["converged"], stagnated=info["stagnated"],
             switched=info["chunks"][0]["switched"])  # fmt: skip
    print(json.dumps(r), flush=True)
    return r


def _envelope(F, y):
    """Least-sum upper envelope x of y over the rows F (F x >= y), a free first coefficient,
    the others >= 0."""
    F, y = np.asarray(F, float), np.asarray(y, float)
    bounds = [(None, None)] + [(0, None)] * (F.shape[1] - 1)
    return linprog(F.sum(0), A_ub=-F, b_ub=-y, bounds=bounds).x


def fit(data):
    """The shell constants from the steps and count entries: pc terms net of STEP_FIT /
    COUNT_STEP, in blocks of 8 n m as a + c / m + K0 / (8 n m) (K0 bytes), and the slack of
    the composed model at the large-n gate points."""
    rows, out = [r for r in data["steps"] if r["degree"] == 2], {}
    for key, fk in (("9-block", "9-block"), ("low", "low"), ("polish", "low")):
        A, f = STEP_FIT[fk]
        F = [[1, 1 / r["m"], 1 / (8 * r["n_dof"] * r["m"])] for r in rows]
        y = [(r[key] - f * r["ws"]) / (8 * r["n_dof"] * r["m"]) - (0 if key == "polish" else A)
             for r in rows]  # fmt: skip
        out[key] = _envelope(F, y).tolist()
    a, c, _ = COUNT_STEP["unstructured"]
    cr = data["count"]
    F = [[1, 1 / r["b"], 1 / (8 * r["n_dof"] * r["b"])] for r in cr]
    y = [(r["recurrence"] - r["ws"]) / (8 * r["n_dof"] * r["b"]) - a - c / r["b"] for r in cr]
    for r in (r for r in cr if "minres_b6" in r):  # si_memory's minres: ws + SI_MINRES b + count
        F.append([1, 1 / 6, 1 / (8 * r["n_dof"] * 6)])
        y.append((r["minres_b6"] - r["ws_b6"]) / (8 * r["n_dof"] * 6) - SI_MINRES)
    out["count"] = _envelope(F, y).tolist()
    print(json.dumps(out), flush=True)
    return out


def main(parts=PARTS, out=harness.RESULTS, write=True):
    path = Path(out) / "shell_solve_study.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.pop("versions", None)
    for part in parts:
        print(f"--- {part}", flush=True)
        data[part] = fit(data) if part == "fit" else globals()[part]()
    if write:
        harness.write_result("shell_solve_study", data, out)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    p.add_argument("--out", default=harness.RESULTS)
    a = p.parse_args()
    main(a.parts, a.out)
