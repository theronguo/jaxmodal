"""Benchmark harness: per-measurement subprocess memory/time, and machine peaks.

Rules (details in docs/conventions.md): `code` and `setup` block on their results; warm up
in `setup` with compiles only (full runs only for buffers > 2 MB).
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from jaxmodal import config

RESULTS = Path(__file__).parent / "results"

CHILD = r"""
import gc, json, os, resource, sys, time
import numpy as np, jax, jax.numpy as jnp

out, setup, code = sys.argv[1:4]
ns = {"np": np, "jax": jax, "jnp": jnp}
exec(setup, ns)
gc.collect()  # jax leaves cyclic garbage that would otherwise be freed during `code`
darwin = sys.platform == "darwin"
if darwin:
    import ctypes

    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

    class RUsage(ctypes.Structure):  # rusage_info_v4: ri_phys_footprint = 7, interval max = 33
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("f", ctypes.c_uint64 * 35)]

    def rusage():
        r = RUsage()
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)):
            raise OSError("proc_pid_rusage failed")
        return r.f

    # XLA frees executable temporaries asynchronously; wait until the footprint is flat.
    t0, prev = time.perf_counter(), rusage()[7]
    while time.perf_counter() - t0 < 2:
        time.sleep(0.01)
        cur = rusage()[7]
        if abs(cur - prev) < 2**20 and time.perf_counter() - t0 >= 0.05:
            break
        prev = cur
    if lib.proc_reset_footprint_interval(os.getpid()):  # not asserts: they vanish under -O
        raise OSError("proc_reset_footprint_interval failed")
    base = rusage()[7]
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.perf_counter()
exec(code, ns)
dt = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0
fp = (rusage()[33] - base) if darwin else None
with open(out, "w") as f:
    json.dump({"fp": fp, "rss": rss, "time_s": dt}, f)
"""


def maxrss_bytes(value, sys_platform):
    return value if sys_platform == "darwin" else value * 1024


def measure(code, setup=""):
    """Run `setup`, then `code`, in a fresh subprocess; return its peak/RSS deltas and wall time.

    On macOS, peak_bytes = max(interval footprint delta, ru_maxrss delta); None elsewhere.
    """
    drop = ("JAX_ENABLE_X64", "JAX_COMPILATION_CACHE_DIR")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["MallocLargeCache"] = "0"  # freed large blocks are unmapped, not reused uncharged
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, out, setup, code],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        r = json.loads(Path(out).read_text())
    rss = maxrss_bytes(r["rss"], sys.platform)
    peak = None if r["fp"] is None else max(r["fp"], rss)
    return {"peak_bytes": peak, "rss_bytes": rss, "time_s": r["time_s"]}


def measure_json(code, setup=""):
    """measure() for `code` that writes a JSON dict to "{out}"; that dict plus peak_bytes."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        r = measure(code.format(out=out), setup)
        return json.loads(Path(out).read_text()) | {"peak_bytes": r["peak_bytes"]}


def _sysctl(name):
    try:
        p = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    except OSError:
        return None
    return int(p.stdout) if p.returncode == 0 else None


def versions():
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "p_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "e_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "jax_num_cpu_devices": "jax_num_cpu_devices" in jax.config.values,
    }


def triad_threads(v):
    p, e = v["p_cores"], v["e_cores"]
    return [p, p + e] if p and e else [v["cpu_count"]]


def _best(f, args, reps):
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - t)
    return min(times)


def c_triad():
    """Best pthread STREAM triad GB/s over triad_threads(versions()), or a skip string."""
    cc = shutil.which(os.environ.get("CC", "cc"))
    if cc is None:
        return "skipped: no C compiler"
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "triad")
        src = str(Path(__file__).with_name("triad.c"))
        if subprocess.run([cc, "-O3", "-pthread", src, "-o", exe], capture_output=True).returncode:
            return "skipped: compile failed"
        runs = []
        for t in triad_threads(versions()):
            out = subprocess.run([exe, str(t)], capture_output=True, text=True, check=True)
            runs.append({"gbps": float(out.stdout.split()[0]), "threads": t})
    return max(runs, key=lambda r: r["gbps"])


def system_info():
    config.enable_x64()
    n = 4096
    a = jnp.ones((n, n))
    assert a.dtype == jnp.float64
    matmul = jax.jit(lambda x, y: x @ y)
    gflops = 2 * n**3 / _best(matmul, (a, a), 5) / 1e9

    m = 20_000_000
    b, c = jnp.ones(m), jnp.full(m, 2.0)
    triad = jax.jit(lambda b, c: b + 3.0 * c)
    gbps = 24 * m / _best(triad, (b, c), 50) / 1e9  # STREAM convention: 24 B per element

    ct = c_triad()
    return {
        "fp64_matmul_gflops": gflops,
        "triad_gbps": gbps,
        "c_triad": ct,
        "c_triad_ratio": ct["gbps"] / gbps if isinstance(ct, dict) else None,
    }


def write_result(name, data, out_dir=RESULTS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({**data, "versions": versions()}, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure machine peaks into system.json.")
    parser.add_argument("--out", default=RESULTS)
    args = parser.parse_args()
    print(write_result("system", system_info(), args.out).read_text())
