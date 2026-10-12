"""M1 elements and operators: element matrices, assembled reference matrices, the matrix-free
and voxel stencil operators, Dirichlet masks, rigid modes and deflation.

Usage: python examples/m1_operators.py
"""

import numpy as np

from jaxmodal import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from jaxmodal.elements import check_elements, mass, stiffness  # noqa: E402
from jaxmodal.materials import Material, isotropic  # noqa: E402
from jaxmodal.mesh import box_mesh, voxel_mesh  # noqa: E402
from jaxmodal.operators import (  # noqa: E402
    assemble,
    dof_mask,
    m_orthonormalize,
    project,
    rigid_modes,
    stencil,
    unstructured,
)

steel = {0: Material(isotropic(210e9, 0.3), 7850.0)}
beam = box_mesh((10, 2, 2), spacing=(0.1, 0.05, 0.05), type="hex20")
check_elements(beam)
X = beam.points[beam.blocks["hex20"]]
Ke, Me = stiffness("hex20", X, steel[0].C), mass("hex20", X, steel[0].rho)
print(f"element matrices: Ke {Ke.shape}, Me {Me.shape}; total mass {float(Me.sum()) / 3:.3f} kg")

v = jnp.asarray(np.random.default_rng(0).standard_normal((3 * len(beam.points), 4)))
K = assemble(beam, steel)[0]
op = unstructured(beam, steel)
err = np.abs(op.apply_K(v) - K @ v).max() / np.abs(K @ v).max()
print(f"unstructured: n_dof {op.n_dof}, apply_K vs assembled: {err:.1e}")
assert err <= 1e-12

cube = voxel_mesh((8, 8, 8), spacing=(0.025, 0.025, 0.025))
vop = stencil(cube, steel)
Kv = assemble(cube, steel)[0]
w = jnp.asarray(np.random.default_rng(1).standard_normal((vop.n_dof, 4)))
errv = np.abs(vop.apply_K(w) - Kv @ w).max() / np.abs(Kv @ w).max()
print(f"stencil: n_dof {vop.n_dof}, apply_K vs assembled: {errv:.1e}")
assert errv <= 1e-12

free = dof_mask(beam, {"xmin": "x"})  # sliding on xmin: T_y, T_z and R_x remain
R = jnp.asarray(rigid_modes(beam, free))
R = m_orthonormalize(R, op.apply_M(R))
Y = project(R, v, op.apply_M(v))
orth = np.abs(R.T @ op.apply_M(Y)).max()
print(f"rigid modes left by sliding: {R.shape[1]}; |R^T M Y| = {orth:.1e}")
assert R.shape[1] == 3 and orth <= 1e-12
print("check passed: both operators match the assembled K; the projector deflates R")
