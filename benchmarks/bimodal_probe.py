"""Bimodal LOBPCG peak probe (PR 18d; docs/open_items.md, "Bimodal LOBPCG peak"). Run, not tested.

The cold fp32 solve_modes on the 791k plate (shell_solve_study.solve_peak) in fresh processes:
5 baseline runs interleaved with 5 runs per XLA setting (B, S1, S2, B, ...), each classified
by its interval peak (high above 5.8 GB), with XLA_FLAGS, wall time and load average per row;
then 3 wrapper runs: the parent samples the child's phys_footprint every 100 ms and the child
logs the live jax.Array bytes and its footprint at the start and end of every LOBPCG step.
Writes results/bimodal_probe.json. Run on an otherwise idle machine (about 20 min). Follow-up
parts, each adding a key to that file: `variants` (the wrapper's full-output sync and its
instruments apart, 3 runs each, about 6 min), `instruments` (jax.live_arrays and the process's
own proc_pid_rusage apart, 2 runs each, about 4 min), `checks` (footprint against ru_maxrss on
unwrapped runs, and parent-only sampling, 2 runs each, about 4 min).
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmarks import harness
from benchmarks import shell_solve_study as SSS

HIGH = 5.8e9  # midway between fp32's 4.75 and 6.9-7.1 GB
SETTINGS = dict(S1="--xla_cpu_multi_thread_eigen=false",
                S2="--xla_cpu_enable_concurrency_optimized_scheduler")  # fmt: skip
WRAP = """
import atexit, ctypes, json, os, sys, time
import jaxmodal.eigen.lobpcg
L = sys.modules["jaxmodal.eigen.lobpcg"]  # jaxmodal.eigen.lobpcg is the function
_lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("f", ctypes.c_uint64 * 35)]
def _fp():
    r = _RU()
    _lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r))
    return r.f[7], r.f[33]
def _live():
    return sum(x.nbytes for x in jax.live_arrays())
STEPS = []
def _wrap(f):
    def g(*a, **k):
        if {mode!r} == "block":  # the sync alone: the loop itself waits only for `out`
            return jax.block_until_ready(f(*a, **k))
        if {mode!r} in ("live", "rusage"):  # one instrument alone, at the step's start and end
            probe = _live if {mode!r} == "live" else _fp
            v0 = probe()
            out = f(*a, **k)
            STEPS.append(dict(v0=v0, v1=probe()))
            return out
        t0, live0, fp0 = time.monotonic(), _live(), _fp()
        out = f(*a, **k)
        if {mode!r} == "full":
            jax.block_until_ready(out)  # state and out
        STEPS.append(dict(t0=t0, t1=time.monotonic(), live0=live0, fp0=fp0, fp1=_fp(),
                          live1=_live()))
        return out
    return g
L.lobpcg_step, L.lobpcg_step_low = _wrap(L.lobpcg_step), _wrap(L.lobpcg_step_low)
atexit.register(lambda: open({log!r}, "w").write(json.dumps(STEPS)))
"""


def solve_child(wrap_log=None, n=362, mode="full"):
    """shell_solve_study.solve_peak's setup and code (fp32, 791k plate), plus the wrapper."""
    root = str(harness.RESULTS.parents[1])
    code = SSS.SOLVE.replace("N_MODES", "20").replace("DTYPE", repr("float32"))
    setup = SSS.SETUP.format(root=root, case="flat", n=n)
    setup = 'import jax\njax.config.update("jax_cpu_enable_async_dispatch", False)\n' + setup
    if wrap_log:
        setup += WRAP.format(log=wrap_log, mode=mode)
    return setup, code


def run_one(label, flags, mode=None):
    """One unwrapped run (or, with mode, the wrapper variant "block", "instr", "live" or
    "rusage")."""
    if flags:
        os.environ["XLA_FLAGS"] = flags
    else:
        os.environ.pop("XLA_FLAGS", None)
    t = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        setup, code = solve_child(os.path.join(d, "steps.json") if mode else None, mode=mode)
        r = harness.measure_json(code, setup)
    row = dict(label=label, xla_flags=flags, wall_s=time.perf_counter() - t,
               load=os.getloadavg(), peak_bytes=r["peak_bytes"], solve_s=r["time_s"],
               iterations=r["iterations"], high=r["peak_bytes"] > HIGH)  # fmt: skip
    print(json.dumps(row), flush=True)
    return row


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("f", ctypes.c_uint64 * 35)]


LIB = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def _add(key, rows):
    """Add rows to results/bimodal_probe.json under key."""
    path = harness.RESULTS / "bimodal_probe.json"
    data = json.loads(path.read_text())
    data[key] = rows
    path.write_text(json.dumps(data, indent=2) + "\n")


def _sampled(setup, code, out):
    """The child (harness CHILD, its env); the parent samples its phys_footprint every 100 ms."""
    p = subprocess.Popen([sys.executable, "-c", harness.CHILD, out, setup, code],
                         env=harness.child_env())  # fmt: skip
    samples, ru = [], _RU()
    while p.poll() is None:
        if LIB.proc_pid_rusage(p.pid, 4, ctypes.byref(ru)) == 0:
            samples.append((time.monotonic(), ru.f[7]))
        time.sleep(0.1)
    if p.returncode:
        raise RuntimeError(f"child failed: {p.returncode}")
    return samples


def wrapper_run(n=362):
    """The child (harness CHILD, its env) with the step wrapper; the parent samples the child's
    phys_footprint every 100 ms. One clock: time.monotonic."""
    os.environ.pop("XLA_FLAGS", None)
    with tempfile.TemporaryDirectory() as d:
        out, info, log = (os.path.join(d, f) for f in ("r.json", "info.json", "steps.json"))
        setup, code = solve_child(log, n)
        samples = _sampled(setup, code.format(out=info), out)
        r, steps = json.loads(Path(out).read_text()), json.loads(Path(log).read_text())
        r |= json.loads(Path(info).read_text())
    peak = max(r["fp"], harness.maxrss_bytes(r["rss"], sys.platform))  # as measure()
    row = dict(label="wrapper", peak_bytes=peak, high=peak > HIGH, load=os.getloadavg(),
               solve_s=r["time_s"], iterations=r["iterations"], samples=samples,
               steps=steps)  # fmt: skip
    print(json.dumps({k: v for k, v in row.items() if k not in ("samples", "steps")}), flush=True)
    return row


def variants(runs=3, modes=("block", "instr"), key="variants"):
    """The wrapper's parts apart (no parent sampler): the full-output sync, the instruments, and
    each instrument alone (modes "live", "rusage")."""
    rows = [run_one(mode, None, mode) for _ in range(runs) for mode in modes]
    _add(key, rows)
    return rows


def checks(runs=2):
    """Unwrapped runs recording max(footprint, ru_maxrss) and ru_maxrss apart (is the extra
    resident?), and runs sampled by the parent only (the child's proc_pid_rusage every 100 ms,
    no wrapper)."""
    os.environ.pop("XLA_FLAGS", None)
    rows = []
    setup, code = solve_child()
    with tempfile.TemporaryDirectory() as d:
        out, info = os.path.join(d, "r.json"), os.path.join(d, "i.json")
        for _ in range(runs):
            r = harness.measure(code.format(out=info), setup)
            rows.append(dict(label="baseline", fp_or_rss=r["peak_bytes"], rss=r["rss_bytes"]))
            print(json.dumps(rows[-1]), flush=True)
        for _ in range(runs):
            n = len(_sampled(setup, code.format(out=info), out))
            r = json.loads(Path(out).read_text())
            rss = harness.maxrss_bytes(r["rss"], sys.platform)
            rows.append(dict(label="parent_sampler", fp_or_rss=max(r["fp"], rss), rss=rss,
                             samples=n))  # fmt: skip
            print(json.dumps(rows[-1]), flush=True)
    _add("checks", rows)
    return rows


def main(n=5, wrappers=3):
    rows = []
    for _ in range(n):
        rows.append(run_one("B", None))
        for s, flags in SETTINGS.items():
            rows.append(run_one(s, flags))
    os.environ.pop("XLA_FLAGS", None)
    wrap = [wrapper_run() for _ in range(wrappers)]
    base = [r for r in rows if r["label"] == "B"]
    summary = dict(baseline_high=sum(r["high"] for r in base), n=len(base))
    for s in SETTINGS:
        rs = [r for r in rows if r["label"] == s]
        summary[s] = dict(high=sum(r["high"] for r in rs), n=len(rs),
                          wall_ratio=sum(r["solve_s"] for r in rs) /
                          sum(r["solve_s"] for r in base))  # fmt: skip
    summary["wrapper_high"] = sum(r["high"] for r in wrap)
    print(json.dumps(summary), flush=True)
    return harness.write_result("bimodal_probe", dict(high_bytes=HIGH, rows=rows, wrapper=wrap,
                                                      summary=summary))  # fmt: skip


if __name__ == "__main__":
    parts = dict(variants=variants, checks=checks,
                 instruments=lambda: variants(2, ("live", "rusage"), "instruments"))  # fmt: skip
    print(parts[sys.argv[1]]() if sys.argv[1:] else main())
