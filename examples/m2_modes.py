"""M2 first modes end-to-end: solve_modes on a clamped steel cantilever and a free-free
aluminium voxel cube, checked against Euler-Bernoulli and the SciPy reference.

Usage: python examples/m2_modes.py
"""

import numpy as np

from jaxmodal import config

config.enable_x64()

from jaxmodal.eigen import reference_modes, solve_modes  # noqa: E402
from jaxmodal.materials import Material, isotropic  # noqa: E402
from jaxmodal.mesh import box_mesh, voxel_mesh  # noqa: E402
from jaxmodal.operators import assemble, dof_mask  # noqa: E402

E, rho, L, h = 210e9, 7850.0, 1.0, 0.1
steel = {0: Material(isotropic(E, 0.3), rho)}
beam = box_mesh((20, 2, 2), spacing=(L / 20, h / 2, h / 2), type="hex20")
bcs = {"xmin": "xyz"}
lam, V, info = solve_modes(beam, steel, bcs, 6, tol=1e-6, maxit=1000)
f = info["frequencies"]
f_eb = 1.8751**2 / (2 * np.pi * L**2) * np.sqrt(E * h**4 / 12 / (rho * h * h))
print(f"cantilever ({info['operator']}, {info['iterations']} iterations): f = {np.round(f, 2)} Hz")
print(f"  first bending {f[0]:.2f} Hz vs Euler-Bernoulli {f_eb:.2f} Hz (ratio {f[0] / f_eb:.4f})")
assert info["converged"] and abs(f[0] / f_eb - 1) < 0.01

K, M = assemble(beam, steel)
lr = reference_modes(K, M, 6, dof_mask(beam, bcs), points=beam.points)[0]
err = np.abs(lam / lr - 1).max()
print(f"  vs reference_modes: max rel. error {err:.1e}")
assert err <= 1e-5

alu = {0: Material(isotropic(70e9, 0.33), 2700.0)}
cube = voxel_mesh((8, 8, 8), spacing=(0.0125, 0.0125, 0.0125))  # 10 cm cube
lam, V, info = solve_modes(cube, alu, {}, 6, tol=1e-6)
print(
    f"free cube ({info['operator']}, {info['n_rigid']} rigid modes deflated): "
    f"f = {np.round(info['frequencies'], 1)} Hz"
)
K, M = assemble(cube, alu)
lr = reference_modes(K, M, 12, dof_mask(cube, {}), points=cube.points)[0][6:]
err = np.abs(lam / lr - 1).max()
print(f"  vs reference_modes: max rel. error {err:.1e}")
assert info["converged"] and info["n_rigid"] == 6 and err <= 1e-5
print("check passed: solve_modes matches Euler-Bernoulli and the reference")
