"""Band study (PR 14b): the a-priori bound of verify=None against the warm count step (the
calibration's recurrence-only t_step at b = 10), on hex20 16^3 with AMG and voxel 32^3 with
GMG, steel, xmin clamped, band below the first gap after mode 3, the default band chunk (40,
block 48). Writes results/band_study.json.
"""

import argparse
import json

from benchmarks import harness
from jaxmodal import config
from jaxmodal.eigen import counting, solve_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh

STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
BCS = {"xmin": "xyz"}


def ratio(mesh):
    """Real warm step / the bound's step: the calibration timed (forced), the bound recorded."""
    rec = {}
    real_bound, real_cal = counting._bound, counting._calibrate

    def bound(*a):
        rec["bound"], rec["steps"] = real_bound(*a), a[0]
        return 0.0  # always calibrate

    def calibrate(*a):
        rec["cal"] = real_cal(*a)
        return (1e9,) + rec["cal"][1:]  # skip the count itself

    counting._bound, counting._calibrate = bound, calibrate
    try:
        f = solve_modes(mesh, STEEL, BCS, 4)[2]["frequencies"]
        opts = {"block": 10}
        _, _, info = solve_modes(mesh, STEEL, BCS, band=(0, (f[2] + f[3]) / 2), count_options=opts)
    finally:
        counting._bound, counting._calibrate = real_bound, real_cal
    t_step = rec["cal"][2]
    per_step = rec["bound"] / rec["steps"]
    return dict(n_dof=3 * len(mesh.points), chunk=info["chunk"], t_iteration=info["t_iteration"],
                t_step=t_step, bound_step=per_step, ratio=t_step / per_step,
                solve_s=info["count"]["solve_s"], degree=info["count"]["degree"])  # fmt: skip


def run():
    config.enable_x64()
    return dict(
        hex20=ratio(box_mesh((16,) * 3, (1 / 16,) * 3, type="hex20")),
        voxel=ratio(voxel_mesh((32,) * 3, (1 / 32,) * 3)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    a = p.parse_args()
    r = run()
    print(json.dumps(r, indent=1))
    harness.write_result("band_study", r, a.out)
