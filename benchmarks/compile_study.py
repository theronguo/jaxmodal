"""Compile-time study (PR 18d) -> results/compile_study.json.

A full solve_modes (20 modes) per case in a fresh subprocess with the persistent compilation
cache disabled: GMG (voxel 20^3, 40^3, 80^3) and AMG (box hex20 8^3, 24^3; the ssh8 plate 64^2,
256^2). The compile time is the sum of the /jax/core/compile/backend_compile_duration events,
beside the summed lowering (jaxpr_to_mlir_module) and trace durations, the compile count and
the wall time; a run with no compile event or any cache hit raises. Next to each, the structure
counts that drive it (levels, ELL width classes per AMG level, stencil slabs).
"""

import argparse
import json

from benchmarks import harness

CASES = [("voxel", 20), ("voxel", 40), ("voxel", 80), ("box", 8), ("box", 24)]
CASES += [("plate", 64), ("plate", 256)]

SETUP = """
import json, time
from collections import defaultdict
jax.config.update("jax_enable_compilation_cache", False)
sums, counts, hits = defaultdict(float), defaultdict(int), []
def on_duration(event, duration, **kw):
    for key in ("backend_compile", "jaxpr_to_mlir_module", "jaxpr_trace"):
        if key in event:
            sums[key] += duration
            counts[key] += 1
jax.monitoring.register_event_duration_secs_listener(on_duration)
jax.monitoring.register_event_listener(
    lambda name, **kw: hits.append(name) if "cache_hits" in name else None)
from jaxmodal import config
config.enable_x64()
from jaxmodal.eigen import solve_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import dof_mask, stencil, unstructured
from jaxmodal.precond import amg, gmg
from benchmarks.shell_solve_study import MESHES, SS
MAT = {{0: Material(isotropic(210e9, 0.3), 7850.0)}}
case, n = "{case}", {n}
if case == "voxel":
    mesh, bcs = voxel_mesh((n,) * 3), {{"xmin": "xyz"}}
elif case == "box":
    mesh, bcs = box_mesh((n,) * 3, (1 / n,) * 3, type="hex20"), {{"xmin": "xyz"}}
else:
    mesh, bcs = MESHES["flat"](n), SS
"""
CODE = """
t = time.perf_counter()
_, _, out = solve_modes(mesh, MAT, bcs, 20)
wall = time.perf_counter() - t
if not counts["backend_compile"] or hits:
    raise RuntimeError(f"no compile events or cache hits: {{dict(counts)}} {{hits}}")
info = dict(n_dof=3 * len(mesh.points), wall_s=wall, compile_s=sums["backend_compile"],
            lowering_s=sums["jaxpr_to_mlir_module"], trace_s=sums["jaxpr_trace"],
            compiles=counts["backend_compile"], iterations=out["iterations"])
free = dof_mask(mesh, bcs)
if case == "voxel":
    op = stencil(mesh, MAT)
    P = gmg(op, free, 1.0)
    info.update(slab=op.slab, slabs=-(-n // op.slab))
else:
    op = unstructured(mesh, MAT)
    P = amg(op, free, 1.0, mesh.points)
    info["ell_classes"] = [len(A.vals) for A, _ in P.levels if hasattr(A, "vals")]
info["levels"] = len(P.levels) + 1
json.dump(info, open("{out}", "w"))
"""


def run(cases=CASES):
    rows = []
    for case, n in cases:
        r = harness.measure_json(CODE, SETUP.format(case=case, n=n))
        r.pop("peak_bytes")
        rows.append(dict(case=case, size=n, **r))
        print(json.dumps(rows[-1]), flush=True)
    return dict(rows=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    return harness.write_result("compile_study", run(), p.parse_args(argv).out)


if __name__ == "__main__":
    print(main())
