"""Element kernels: pure jax.numpy functions of (X, C, rho), batched over leading axes.
See docs/elements.md.

The element type is a static name; its tables are constants. X is (..., n, 3); C broadcasts as
(6, 6) or (..., 6, 6), rho as () or (...); DOF 3a + i is component i of node a. Solid shells
(ssh8) have their own kernels (shell.py): stiffness, mass, lumped_nodal and the forms raise
for them; the geometry helpers (jacobian, gradients, bmatrix) do not.
"""

import jax.numpy as jnp
import numpy as np

from ..materials.rotation import VOIGT
from .registry import ELEMENTS, _det3

# eps_v = S[v, i, j] du_i/dx_j with engineering shear.
S = np.zeros((6, 3, 3))
for _v, (_i, _j) in enumerate(VOIGT):
    S[_v, _i, _j] = S[_v, _j, _i] = 1.0


def jacobian(name, X):
    """J^-1 (..., nq, 3, 3) and w det J (..., nq) at the quadrature points (closed form)."""
    el = ELEMENTS[name]
    dt = jnp.result_type(X, float)
    J = jnp.einsum("...ai,qaj->...qij", jnp.asarray(X, dt), jnp.asarray(el.dN, dt))
    a, b, c = J[..., 0, :], J[..., 1, :], J[..., 2, :]
    det = _det3(J)
    inv = jnp.stack([jnp.cross(b, c), jnp.cross(c, a), jnp.cross(a, b)], -1) / det[..., None, None]
    return inv, det * jnp.asarray(el.weights, dt)


def gradients(name, invJ):
    """Physical shape-function gradients G (..., nq, n, 3) = J^-T grad_xi N."""
    dN = jnp.asarray(ELEMENTS[name].dN, invJ.dtype)
    return jnp.einsum("qaj,...qji->...qai", dN, invJ)


def bmatrix(G):
    """Strain-displacement matrix (..., nq, 6, 3n), Voigt rows, node-major columns."""
    B = jnp.einsum("vij,...qaj->...qvai", jnp.asarray(S, G.dtype), G)
    return B.reshape(*B.shape[:-2], -1)


def _hex_only(name):
    if name == "ssh8":
        raise ValueError("ssh8 uses the solid-shell kernels (shell_stiffness, shell_mass)")


def _sym(A):
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def stiffness(name, X, C):
    """Ke (..., 3n, 3n) = sum_q w det J B^T C B, exactly symmetric."""
    _hex_only(name)
    invJ, wdet = jacobian(name, X)
    B = bmatrix(gradients(name, invJ))
    return _sym(jnp.einsum("...q,...qva,...vw,...qwb->...ab", wdet, B, C, B))


def _wdet_N(name, X):
    _, wdet = jacobian(name, X)
    return wdet, jnp.asarray(ELEMENTS[name].N, wdet.dtype)


def mass(name, X, rho):
    """Consistent Me (..., 3n, 3n) = rho m (x) I_3, exactly symmetric."""
    _hex_only(name)
    wdet, N = _wdet_N(name, X)
    m = _sym(jnp.asarray(rho)[..., None, None] * jnp.einsum("...q,qa,qb->...ab", wdet, N, N))
    n = m.shape[-1]
    return (m[..., :, None, :, None] * jnp.eye(3, dtype=m.dtype)[:, None, :]).reshape(
        *m.shape[:-2], 3 * n, 3 * n
    )


def lumped_nodal(name, wdet):
    """Lumped nodal masses per unit density (..., n) from w det J (..., nq)."""
    if ELEMENTS[name].lumping is None:
        raise ValueError("solid-shell elements are consistent-mass only")
    N = jnp.asarray(ELEMENTS[name].N, wdet.dtype)
    if ELEMENTS[name].lumping == "rowsum":
        return jnp.einsum("...q,qa->...a", wdet, N)
    d = jnp.einsum("...q,qa->...a", wdet, N * N)  # HRZ: diagonal of m scaled to the element mass
    return d * (wdet.sum(-1) / d.sum(-1))[..., None]


def lumped_mass(name, X, rho):
    """Diagonal of the lumped Me (..., 3n): row-sum (hex8, hex27) or HRZ (hex20)."""
    _, wdet = jacobian(name, X)
    return jnp.repeat(jnp.asarray(rho)[..., None] * lumped_nodal(name, wdet), 3, axis=-1)


def _strain(G, u):
    U = u.reshape(*u.shape[:-1], -1, 3)
    H = jnp.einsum("...ai,...qaj->...qij", U, G)  # displacement gradient
    return jnp.einsum("vij,...qij->...qv", jnp.asarray(S, H.dtype), H)


def stiffness_form(name, X, C, u, w):
    """u^T Ke w (...) without forming Ke."""
    _hex_only(name)
    invJ, wdet = jacobian(name, X)
    G = gradients(name, invJ)
    return jnp.einsum("...q,...qv,...vw,...qw->...", wdet, _strain(G, u), C, _strain(G, w))


def mass_form(name, X, rho, u, w):
    """u^T Me w (...) without forming Me."""
    _hex_only(name)
    wdet, N = _wdet_N(name, X)
    U, W = (jnp.einsum("qa,...ai->...qi", N, v.reshape(*v.shape[:-1], -1, 3)) for v in (u, w))
    return rho * jnp.einsum("...q,...qi,...qi->...", wdet, U, W)
