"""Element registry: hex shape functions, quadrature and faces (NumPy only). See docs/elements.md.

Shape functions are N(xi) = m(xi) V^-1 for a monomial space m and the node coordinates
(V_bk = m_k(xi_b)), so N_a(xi_b) = delta_ab by construction.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from ..mesh.model import EDGES, FACE_CORNERS, REFERENCE_NODES


def gauss(order):
    """Tensor Gauss-Legendre rule on [-1, 1]^3, xi fastest: points (order**3, 3), weights."""
    g, w = np.polynomial.legendre.leggauss(order)
    idx = np.array(list(itertools.product(range(order), repeat=3)))[:, ::-1]
    return g[idx], w[idx].prod(1)


def _det3(J):
    """Determinant of (..., 3, 3); indexing and arithmetic only, so NumPy and jax arrays work."""
    a, b, c = J[..., 0, :], J[..., 1, :], J[..., 2, :]
    return (
        a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
        - a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0])
        + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
    )


def _monomials(xi, exponents):
    return (np.asarray(xi, float)[..., None, :] ** exponents).prod(-1)


def _monomial_grad(xi, exponents):
    xi = np.asarray(xi, float)[..., None, :]
    out = []
    for d in range(3):
        e = exponents.copy()
        e[:, d] = np.maximum(e[:, d] - 1, 0)  # p xi^max(p-1, 0): no 0 * inf at xi = 0
        out.append(exponents[:, d] * (xi**e).prod(-1))
    return np.stack(out, -1)


@dataclass(frozen=True, eq=False)
class Element:
    name: str
    nodes: np.ndarray  # (n, 3) reference coordinates, VTK order
    exponents: np.ndarray  # (n, 3) monomial space
    coef: np.ndarray  # (n, n) inverse Vandermonde: N(xi) = monomials(xi) @ coef
    order: int  # Gauss points per direction
    lumping: str  # "rowsum" or "hrz"
    faces: tuple  # per face: corners, then edge nodes, then the face centre (hex27)
    points: np.ndarray  # (nq, 3) quadrature points
    weights: np.ndarray  # (nq,)
    N: np.ndarray  # (nq, n) shape functions at the points
    dN: np.ndarray  # (nq, n, 3) reference derivatives at the points

    def shape(self, xi):
        """(..., 3) -> (..., n)."""
        return _monomials(xi, self.exponents) @ self.coef

    def grad(self, xi):
        """(..., 3) -> (..., n, 3), derivatives with respect to xi."""
        return np.einsum("...kd,ka->...ad", _monomial_grad(xi, self.exponents), self.coef)


def _element(name, n, order, lumping):
    nodes = REFERENCE_NODES[:n]
    exponents = np.array(list(itertools.product(range(2 if n == 8 else 3), repeat=3)))
    if n == 20:
        exponents = exponents[(exponents == 2).sum(1) <= 1]  # serendipity: at most one square
    coef = np.linalg.inv(_monomials(nodes, exponents))
    edge = {frozenset(e): 8 + i for i, e in enumerate(EDGES)}
    ring = [
        tuple(edge[frozenset(p)] for p in zip(c, c[1:] + c[:1], strict=True)) for c in FACE_CORNERS
    ]
    size = {8: 4, 20: 8, 27: 9}[n]
    faces = tuple(
        (c + r + (20 + f,))[:size] for f, (c, r) in enumerate(zip(FACE_CORNERS, ring, strict=True))
    )
    points, weights = gauss(order)
    N = _monomials(points, exponents) @ coef
    dN = np.einsum("qkd,ka->qad", _monomial_grad(points, exponents), coef)
    return Element(name, nodes, exponents, coef, order, lumping, faces, points, weights, N, dN)


ELEMENTS = {
    "hex8": _element("hex8", 8, 2, "rowsum"),
    "hex20": _element("hex20", 20, 3, "hrz"),
    "hex27": _element("hex27", 27, 3, "rowsum"),
}
