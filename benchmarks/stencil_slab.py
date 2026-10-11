"""Slab-size benchmark of the voxel stencil; checks the default slab rule (SLAB_CELLS).

Each configuration runs in a fresh subprocess (harness.measure): apply_K time (min of 5) and
peak memory of one apply_K. Writes results/stencil_slab.json. Machine bandwidth (STREAM
triad): results/system.json.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from benchmarks import harness

N = 87  # full N^3 voxel box, 2.04M DOF

SETUP = """
import json, time
from jaxmodal import config
config.enable_x64()
from jaxmodal.mesh import voxel_mesh
from jaxmodal.materials import Material, isotropic
from jaxmodal.operators import stencil
mesh = voxel_mesh(({n},) * 3)
op = stencil(mesh, {{0: Material(isotropic(1.0, 0.3), 1.0)}}, slab={slab})
v = jnp.ones((op.n_dof, {m}))
op.apply_K(v).block_until_ready()
"""
CODE = """
ts = []
for _ in range(5):
    t = time.perf_counter()
    op.apply_K(v).block_until_ready()
    ts.append(time.perf_counter() - t)
open("{out}", "w").write(json.dumps(dict(apply_s=min(ts), slab_used=op.slab)))
"""


def run(n, m, slab=None):
    cfg = dict(n=n, m=m, slab=slab)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        r = harness.measure(CODE.format(out=out), SETUP.format(n=n, m=m, slab=slab))
        cfg |= json.loads(Path(out).read_text())
    cfg["peak_bytes"] = r["peak_bytes"]
    print(json.dumps(cfg), flush=True)
    return cfg


def main(n=N, out=harness.RESULTS, write=True):
    runs, ok = [], {}
    for m in (1, 16):
        rule = run(n, m)
        s = rule["slab_used"]
        sweep = [run(n, m, s2) for s2 in sorted({max(1, s // 4), min(4 * s, n)} - {s})]
        runs += [rule] + sweep
        ok[f"m{m}"] = rule["apply_s"] <= 1.1 * min(r["apply_s"] for r in sweep + [rule])
    result = dict(runs=runs, slab_rule_ok=ok, system="benchmarks/results/system.json")
    if write:
        harness.write_result("stencil_slab", result, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    print(json.dumps(main(out=p.parse_args().out)["slab_rule_ok"], indent=2))
