"""Solid-shell element ssh8 (Klinkel, Gruttmann, Wagner 1999): ANS + 5-parameter EAS, layered.
See docs/elements.md.

Pure jax.numpy, one element per call (callers vmap). X is (8, 3) in VTK order with reference zeta
the thickness (corners 0-3 bottom, 4-7 top). table_e = (plies, R_o): plies is a Ply whose leaves
have a leading ply axis (L,) (one phase of materials.ply_table), R_o (3, 3) the orientation.

Covariant strains E_ij = (g_i . u_,j + g_j . u_,i) / 2 in VOIGT order with engineering shear;
Cartesian strain eps = T E with T = bond_strain(J^-T). ANS: 2E13 tied at (0, -+1, zeta), 2E23 at
(-+1, 0, zeta) (Bathe-Dvorkin), E33 at the 4 corner lines at zeta = 0 (Betsch-Stein). EAS:
eps~ = (det J0 / det J) T0 M alpha, condensed with a 5x5 Cholesky. Quadrature: 2x2 in-plane
times 2 Gauss points per ply.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import lax

from ..materials.rotation import VOIGT, _rz, bond_strain, rotate
from ..mesh.model import CORNERS

_G = 1 / np.sqrt(3)
_INPLANE = np.array([[x, y] for y in (-_G, _G) for x in (-_G, _G)])  # (4, 2), xi fastest
_LINES = CORNERS[:4, :2]  # (xi, eta) of the 4 corner lines


def _sym(A):
    return 0.5 * (A + A.T)


def _shape(p):
    """Trilinear N (8,) and dN/dxi (8, 3) at the reference point p (3,)."""
    c = jnp.asarray(CORNERS, p.dtype)
    f = 1 + c * p
    dN = jnp.stack(
        [c[:, 0] * f[:, 1] * f[:, 2], f[:, 0] * c[:, 1] * f[:, 2], f[:, 0] * f[:, 1] * c[:, 2]], 1
    )
    return f.prod(1) / 8, dN / 8


def _covariant(X, p):
    """J (3, 3) with J_ij = dx_i/dxi_j, and the covariant strain matrix (6, 24) at p."""
    _, dN = _shape(p)
    J = X.T @ dN
    g = J.T  # g[j] = dx/dxi_j
    rows = [
        dN[:, i, None] * g[i] if i == j else dN[:, j, None] * g[i] + dN[:, i, None] * g[j]
        for i, j in VOIGT
    ]
    return J, jnp.stack([r.reshape(24) for r in rows])


def _unit(v):
    """v / |v|, 0 for v = 0 (NaN-free values and gradients)."""
    n2 = v @ v
    ok = n2 > 0
    return jnp.where(ok, v, 0) / jnp.sqrt(jnp.where(ok, n2, 1))


def _frame(X, R_o):
    """Shell frame R_shell = [t1 t2 t3] at the element centre: t3 = g1 x g2 normalized, t1 = R_o
    axis 1 projected onto the shell plane; if that projection is shorter than 0.1, t2 = R_o axis 2
    projected and t1 = t2 x t3."""
    J0 = X.T @ _shape(jnp.zeros(3, X.dtype))[1]
    t3 = _unit(jnp.cross(J0[:, 0], J0[:, 1]))
    p1, p2 = (a - (a @ t3) * t3 for a in (R_o[:, 0], R_o[:, 1]))
    primary = p1 @ p1 >= 0.01
    t1p = _unit(p1)
    t2f = _unit(p2)
    t1 = jnp.where(primary, t1p, jnp.cross(t2f, t3))
    t2 = jnp.where(primary, jnp.cross(t3, t1p), t2f)
    return jnp.stack([t1, t2, t3], 1)


def _ply_zeta(fraction):
    """Per ply the 2 Gauss zetas (L, 2); their weight is the fraction (0 for a padded ply)."""
    z0 = -1 + 2 * (jnp.cumsum(fraction) - fraction)
    return (z0 + fraction)[:, None] + fraction[:, None] * jnp.asarray([-_G, _G], fraction.dtype)


def _element(X):
    """Element constants: det J0, T0 and the E33 rows at the corner lines (4, 24)."""
    z = jnp.zeros(3, X.dtype)
    J0, _ = _covariant(X, z)
    T0 = bond_strain(jnp.linalg.inv(J0).T)
    lines = jnp.asarray(_LINES, X.dtype)
    b33 = jnp.stack([_covariant(X, jnp.stack([a, b, z[0]]))[1][2] for a, b in lines])
    return jnp.linalg.det(J0), T0, b33


def _point(X, p, const):
    """w-free point terms: det J, the Cartesian ANS strain matrix TB (6, 24), the EAS matrix
    G (6, 5) and N (8,)."""
    detJ0, T0, b33 = const
    xi, eta, zeta = p
    J, B = _covariant(X, p)
    one, zero = jnp.ones_like(xi), jnp.zeros_like(xi)
    tie = [_covariant(X, jnp.stack(q))[1] for q in ((zero, -one, zeta), (zero, one, zeta))]
    B = B.at[4].set(0.5 * (1 - eta) * tie[0][4] + 0.5 * (1 + eta) * tie[1][4])
    tie = [_covariant(X, jnp.stack(q))[1] for q in ((-one, zero, zeta), (one, zero, zeta))]
    B = B.at[3].set(0.5 * (1 - xi) * tie[0][3] + 0.5 * (1 + xi) * tie[1][3])
    lines = jnp.asarray(_LINES, X.dtype)
    B = B.at[2].set(((1 + lines[:, 0] * xi) * (1 + lines[:, 1] * eta) / 4) @ b33)
    det = jnp.linalg.det(J)
    M = jnp.zeros((6, 5), X.dtype)
    M = M.at[0, 0].set(xi).at[1, 1].set(eta).at[5, 2].set(xi).at[5, 3].set(eta).at[2, 4].set(zeta)
    G = (detJ0 / det) * T0 @ M
    return det, bond_strain(jnp.linalg.inv(J).T) @ B, G, _shape(p)[0]


def _ply_coords(zeta):
    """The 8 reference points (8, 3) of a ply with Gauss zetas (2,): in-plane fastest."""
    ip = jnp.asarray(_INPLANE, zeta.dtype)
    return jnp.concatenate([jnp.column_stack([ip, jnp.full(4, z)]) for z in zeta])


def _ply(X, const, R_shell, C, angle, fraction, zeta):
    """One ply's 8 points: w det J (8,), TB (8, 6, 24), G (8, 6, 5), N (8, 8), C' (6, 6)."""
    det, TB, G, N = jax.vmap(lambda q: _point(X, q, const))(_ply_coords(zeta))
    return fraction * det, TB, G, N, rotate(C, R_shell @ _rz(angle))


def _terms(X, table_e):
    """Element constants, the shell frame and the per-ply scan inputs (C, angle, fraction, zeta)."""
    plies, R_o = table_e
    xs = (plies.material.C, plies.angle, plies.fraction, _ply_zeta(plies.fraction))
    return _element(X), _frame(X, R_o), xs


def shell_stiffness(X, table_e):
    """Condensed Ke (24, 24), exactly symmetric; plies accumulated by lax.scan."""
    const, R_shell, xs = _terms(X, table_e)

    def step(K, x):
        wd, TB, G, _, Cp = _ply(X, const, R_shell, *x)
        CTB, CG = Cp @ TB, Cp @ G
        Kuu = jnp.einsum("q,qva,qvb->ab", wd, TB, CTB)
        Kua = jnp.einsum("q,qva,qvb->ab", wd, TB, CG)
        Kaa = jnp.einsum("q,qva,qvb->ab", wd, G, CG)
        return tuple(k + d for k, d in zip(K, (Kuu, Kua, Kaa), strict=True)), None

    dt = jnp.result_type(X, float)  # rotate works in the default float
    K0 = (jnp.zeros((24, 24), dt), jnp.zeros((24, 5), dt), jnp.zeros((5, 5), dt))
    (Kuu, Kua, Kaa), _ = lax.scan(step, K0, xs)
    return _sym(Kuu - Kua @ jsl.cho_solve(jsl.cho_factor(Kaa), Kua.T))


def shell_mass(X, plies):
    """Scalar consistent mass block m_e (8, 8), rho included: M_e = m_e (x) I_3; plies
    accumulated by lax.scan (all plies at once cost about 3.5 KB per ply per element in a build)."""

    def at(q):
        N, dN = _shape(q)
        return N, jnp.linalg.det(X.T @ dN)

    def step(m, x):
        rho, fraction, zeta = x
        N, det = jax.vmap(at)(_ply_coords(zeta))  # (8, 8), (8,)
        return m + jnp.einsum("q,qa,qb->ab", rho * fraction * det, N, N), None

    xs = (plies.material.rho, plies.fraction, _ply_zeta(plies.fraction))
    m, _ = lax.scan(step, jnp.zeros((8, 8), jnp.result_type(X, float)), xs)
    return _sym(m)


def shell_recover(X, table_e, u):
    """Recovery at the 8L quadrature points (padded plies have weight 0): physical points (8L, 3),
    weights w det J (8L,), global Cartesian Voigt strain and stress (8L, 6), engineering shear,
    from the ANS strains plus the EAS strains with the condensed alpha."""
    const, R_shell, xs = _terms(X, table_e)
    wd, TB, G, N, Cp = jax.vmap(lambda *x: _ply(X, const, R_shell, *x))(*xs)
    CG = jnp.einsum("lvw,lqwb->lqvb", Cp, G)
    Kua = jnp.einsum("lq,lqva,lqvb->ab", wd, TB, CG)
    Kaa = jnp.einsum("lq,lqva,lqvb->ab", wd, G, CG)
    alpha = -jsl.cho_solve(jsl.cho_factor(Kaa), Kua.T @ u)
    eps = TB @ u + G @ alpha
    sigma = jnp.einsum("lvw,lqw->lqv", Cp, eps)
    return (
        (N @ X).reshape(-1, 3),
        wd.reshape(-1),
        eps.reshape(-1, 6),
        sigma.reshape(-1, 6),
    )
