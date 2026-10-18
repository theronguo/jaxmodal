"""M4 solid shell: the lowest modes of a simply supported square steel plate (L/t = 20) of
ssh8 elements, one element through the thickness, with LOBPCG and the solid-shell AMG (the
default since PR 18b); checked against Kirchhoff plate theory and the SciPy reference.

Usage: python examples/m4_shell.py
"""

import numpy as np

from jaxmodal import config

config.enable_x64()

from jaxmodal.eigen import reference_modes, solve_modes  # noqa: E402
from jaxmodal.materials import Material, isotropic  # noqa: E402
from jaxmodal.mesh import box_mesh, solid_shell  # noqa: E402
from jaxmodal.operators import assemble, dof_mask  # noqa: E402

E, nu, rho, L, t = 210e9, 0.3, 7850.0, 1.0, 0.05
steel = {0: Material(isotropic(E, nu), rho)}
plate = solid_shell(box_mesh((8, 8, 1), spacing=(L / 8, L / 8, t)), direction=(0, 0, 1))[0]
ss = {"xmin": "yz", "xmax": "yz", "ymin": "xz", "ymax": "xz"}  # simply supported edges
lam, V, info = solve_modes(plate, steel, ss, 6, tol=1e-6)
f = info["frequencies"]
# Kirchhoff: f_11 = pi / 2 (2 / L^2) sqrt(D / (rho t)), D = E t^3 / (12 (1 - nu^2))
f11 = np.pi * np.sqrt(E * t**2 / (12 * (1 - nu**2) * rho)) / L**2
print(f"plate ({info['precond']}, {info['iterations']} iterations): f = {np.round(f, 1)} Hz")
print(f"  f_11 {f[0]:.1f} Hz vs Kirchhoff {f11:.1f} Hz (ratio {f[0] / f11:.4f})")
assert info["converged"] and abs(f[0] / f11 - 1) < 0.03

K, M = assemble(plate, steel)
lr = reference_modes(K, M, 6, dof_mask(plate, ss), points=plate.points)[0]
err = np.abs(lam / lr - 1).max()
print(f"  vs reference_modes: max rel. error {err:.1e}")
assert err <= 1e-5
print("check passed: the ssh8 plate matches Kirchhoff and the reference")
