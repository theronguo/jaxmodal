"""M0 foundations: read and generate meshes, write VTU, materials and phases.

Usage: python examples/m0_foundations.py [output_dir]
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from jaxmodal import config

config.enable_x64()

from jaxmodal.materials import (  # noqa: E402
    Material,
    check_materials,
    from_euler,
    isotropic,
    phase_table,
    phases,
    transversely_isotropic,
)
from jaxmodal.mesh import box_mesh, read, voxel_mesh, write  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
out.mkdir(parents=True, exist_ok=True)
data = Path(__file__).parents[1] / "tests" / "data"

pair = read(data / "gmsh_pair.msh")  # Gmsh 4.1 ASCII, two hex20 volumes
blocks = {t: len(c) for t, c in pair.blocks.items()}
print(f"gmsh_pair.msh: {len(pair.points)} nodes, elements per type {blocks}")

x, y, z = np.meshgrid(*[np.arange(20) - 9.5] * 3, indexing="ij")
sphere = voxel_mesh((20, 20, 20), spacing=(0.05, 0.05, 0.05), mask=x**2 + y**2 + z**2 < 100)
beam = box_mesh((10, 2, 2), spacing=(0.1, 0.05, 0.05), type="hex20")
print(f"voxel sphere: {sphere.n_el} cells; beam: {beam.n_el} hex20, sets {sorted(beam.node_sets)}")

steel = Material(isotropic(210e9, 0.3), 7850.0)
cfrp = Material(transversely_isotropic(140e9, 10e9, 0.3, 0.45, 5e9), 1600.0)
beam.material[beam.n_el // 2 :] = 1  # second half: fibers rotated 30 degrees about z
beam.orientation[beam.n_el // 2 :] = 1
materials, orientations = {0: steel, 1: cfrp}, {1: from_euler((np.pi / 6, 0.0, 0.0))}
index, pairs = phases(beam)
check_materials(pairs, materials, orientations)
C, rho = phase_table(pairs, materials, orientations)
print(f"phases {pairs}: C {C.shape}, rho {np.asarray(rho)}")

write(out / "sphere.vtu", sphere)
back = read(out / "sphere.vtu")
assert np.array_equal(back.points, sphere.points)
assert np.array_equal(back.blocks["hex8"], sphere.blocks["hex8"])
print(f"check passed: {out / 'sphere.vtu'} reads back identically")
