"""M3b frequency bands: solve_modes(band=(f1, f2)) on a clamped hex20 block under the default
strategy (LOBPCG from 0 when its basis fits the memory budget) from 0 and for an interior band,
and the interior band forced onto shift-invert slices; each against the SciPy reference.

Usage: python examples/m3b_bands.py
"""

import numpy as np

from jaxmodal import config

config.enable_x64()

from jaxmodal.eigen import reference_modes, solve_modes  # noqa: E402
from jaxmodal.materials import Material, isotropic  # noqa: E402
from jaxmodal.mesh import box_mesh  # noqa: E402
from jaxmodal.operators import assemble, dof_mask  # noqa: E402

mat = {0: Material(isotropic(3.0, 0.3), 2.0)}
block = box_mesh((6, 5, 4), spacing=(1 / 6, 1 / 5, 1 / 4), type="hex20")
bcs = {"xmin": "xyz"}
K, M = assemble(block, mat)
lr = reference_modes(K, M, 40, dof_mask(block, bcs), points=block.points)[0]
fr = np.sqrt(lr) / (2 * np.pi)


def check(name, f1, f2, lam, info):
    ref = lr[(fr >= f1) & (fr <= f2)]
    err = np.abs(lam / ref - 1).max()
    print(f"{name}: {len(lam)} modes in [{f1:.4f}, {f2:.4f}] Hz by {info['strategy']}"
          f" ({info['reason']}), max rel. error {err:.1e}")  # fmt: skip
    assert len(lam) == len(ref) and err <= 1e-6


cut = lambda i: (fr[i - 1] + fr[i]) / 2  # noqa: E731 (midway between modes i - 1 and i)
lam, _, info = solve_modes(block, mat, bcs, band=(0, cut(8)))
check("from 0", 0.0, cut(8), lam, info)
f1, f2 = cut(16), cut(26)
lam, _, info = solve_modes(block, mat, bcs, band=(f1, f2), n_modes=10)
check("interior", f1, f2, lam, info)
lam, _, info = solve_modes(block, mat, bcs, band=(f1, f2), n_modes=10, strategy="shift_invert")
check("interior, slices", f1, f2, lam, info)
print("check passed: every band matches the reference")
