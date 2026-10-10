"""Rotations and Bond matrices (pure jax.numpy, batched over leading axes). See docs/materials.md.

R has the local material axes as columns: x_global = R x_local, sigma_global = R sigma_local R^T.
"""

import jax.numpy as jnp
import numpy as np

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_I, _J = (np.array(a)[:, None] for a in zip(*VOIGT, strict=True))  # row pair (i, j)
_K, _L = (np.array(a)[None, :] for a in zip(*VOIGT, strict=True))  # column pair (k, l)
_SHEAR = (_K != _L).astype(float)
_D = np.array([1.0, 1, 1, 2, 2, 2])  # engineering shear factor


def _rz(t):
    c, s, z, o = jnp.cos(t), jnp.sin(t), jnp.zeros_like(t), jnp.ones_like(t)
    return jnp.stack(
        [jnp.stack([c, -s, z], -1), jnp.stack([s, c, z], -1), jnp.stack([z, z, o], -1)], -2
    )


def _rx(t):
    c, s, z, o = jnp.cos(t), jnp.sin(t), jnp.zeros_like(t), jnp.ones_like(t)
    return jnp.stack(
        [jnp.stack([o, z, z], -1), jnp.stack([z, c, -s], -1), jnp.stack([z, s, c], -1)], -2
    )


def from_euler(angles):
    """Bunge angles (phi1, Phi, phi2) in radians, (..., 3) -> R = Rz(phi1) Rx(Phi) Rz(phi2)."""
    a = jnp.asarray(angles, float)
    return _rz(a[..., 0]) @ _rx(a[..., 1]) @ _rz(a[..., 2])


def from_vectors(a, b):
    """Local axis 1 along a, b in the local 1-2 plane (Abaqus RECTANGULAR); NaN if a || b."""
    a, b = jnp.asarray(a, float), jnp.asarray(b, float)
    e1 = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    e3 = jnp.cross(a, b)
    e3 = e3 / jnp.linalg.norm(e3, axis=-1, keepdims=True)
    return jnp.stack([e1, jnp.cross(e3, e1), e3], -1)


def bond_stress(R):
    """M with sigma_g = M sigma_l in Voigt order, (..., 6, 6)."""
    R = jnp.asarray(R, float)
    r = lambda p, q: R[..., p, q]  # noqa: E731
    return r(_I, _K) * r(_J, _L) + _SHEAR * r(_I, _L) * r(_J, _K)


def bond_strain(R):
    """N with eps_g = N eps_l (engineering shear); N = D M D^-1 = M^-T."""
    return bond_stress(R) * _D[:, None] / _D[None, :]


def rotate(C, R):
    """Local -> global stiffness M C M^T, batched over leading axes."""
    M = bond_stress(R)
    return M @ jnp.asarray(C, float) @ jnp.swapaxes(M, -1, -2)
