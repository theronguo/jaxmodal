"""CPU scatter vs gather benchmark of the matrix-free operator; sets DEFAULT_PATH["cpu"].

Each configuration runs in a fresh subprocess (harness.measure): apply_K time (min of 5),
peak memory of one apply_K, compile time of apply_K. Writes results/scatter_gather.json.
Machine bandwidth (STREAM triad): results/system.json.
"""

import argparse
import json

from benchmarks import harness

SIZES = {"hex8": 69, "hex20": 44, "hex27": 35}  # ~1M DOF each

SETUP = """
import json, time
from jaxmodal import config
config.enable_x64()
from jaxmodal.mesh import box_mesh, rcm
from jaxmodal.materials import Material, isotropic
from jaxmodal.operators import unstructured
mesh = box_mesh(({n},) * 3, type="{t}")
if {rcm}:
    mesh = rcm(mesh)[0]
op = unstructured(mesh, {{0: Material(isotropic(1.0, 0.3), 1.0)}}, mode="{mode}", path="{path}",
                  chunk={chunk}, range_nodes={R})
v = jnp.ones((op.n_dof, {m}))
t = time.perf_counter()
f = jax.jit(lambda o, v: o.apply_K(v)).lower(op, v).compile()
compile_s = time.perf_counter() - t
f(op, v).block_until_ready()
"""
CODE = """
ts = []
for _ in range(5):
    t = time.perf_counter()
    f(op, v).block_until_ready()
    ts.append(time.perf_counter() - t)
b = op.blocks[0]
r = dict(apply_s=min(ts), compile_s=compile_s, chunk_used=b.chunk, R_used=op.R, W=b.W)
open("{out}", "w").write(json.dumps(r))
"""


def run(t, n, rcm=False, mode="stored", path="scatter", m=1, chunk=None, R=None):
    cfg = dict(type=t, n=n, rcm=rcm, mode=mode, path=path, m=m, chunk=chunk, range_nodes=R)
    setup = SETUP.format(t=t, n=n, rcm=rcm, mode=mode, path=path, m=m, chunk=chunk, R=R)
    cfg |= harness.measure_json(CODE, setup)
    print(json.dumps(cfg), flush=True)
    return cfg


def main(sizes=SIZES, out=harness.RESULTS, write=True):
    runs = []
    for t, n in sizes.items():
        for rcm in (False, True):
            for mode in ("stored", "otf"):
                for path in ("scatter", "gather"):
                    for m in (1, 16):
                        runs.append(run(t, n, rcm, mode, path, m))
        c = next(r for r in runs if r["type"] == t)["chunk_used"]
        for m in (1, 16):  # chunk rule: the default within 10% of the best of c/4, c, 4c
            for c2 in (c // 4, c * 4):
                runs.append(run(t, n, m=m, chunk=c2))
        R = next(r for r in runs if r["type"] == t and r["path"] == "gather")["R_used"]
        for R2 in (R // 2, 2 * R):  # information only: R is re-tuned in the GPU check
            runs.append(run(t, n, path="gather", m=16, R=R2))

    default = [r for r in runs if r["chunk"] is None and r["range_nodes"] is None]

    def match(r, **kw):
        keys = ("type", "rcm", "mode", "path", "m")
        return next(s for s in default if all(s[k] == kw.get(k, r[k]) for k in keys))

    gathers = [r for r in default if r["path"] == "gather"]
    wins = sum(match(r, path="scatter")["apply_s"] <= r["apply_s"] for r in gathers)
    chunk_ok = {}
    for t in sizes:
        for m in (1, 16):
            rule = match(dict(type=t, rcm=False, mode="stored", path="scatter", m=m))
            sweep = [r for r in runs if r["type"] == t and r["m"] == m and r["chunk"] is not None]
            chunk_ok[f"{t}_m{m}"] = rule["apply_s"] <= 1.1 * min(
                r["apply_s"] for r in sweep + [rule]
            )
    result = dict(
        runs=runs,
        default_path_cpu="scatter" if 2 * wins >= len(gathers) else "gather",
        scatter_wins=f"{wins}/{len(gathers)}",
        chunk_rule_ok=chunk_ok,
        system="benchmarks/results/system.json",
    )
    if write:
        harness.write_result("scatter_gather", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    print(
        json.dumps({k: v for k, v in main(out=p.parse_args().out).items() if k != "runs"}, indent=2)
    )
