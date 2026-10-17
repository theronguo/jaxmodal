"""Independent NumPy reference of the ssh8 solid shell: point loops, explicit tensor transforms,
dense condensation. Plies are a list of (C, rho, angle, fraction) bottom to top."""

import numpy as np
from helpers import PAIRS, tensor
from scipy.spatial.transform import Rotation

VTK = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    float,
)  # fmt: skip
G = 1 / np.sqrt(3)
I3 = np.eye(3)


def shape(p):
    N, dN = np.zeros(8), np.zeros((8, 3))
    for a, c in enumerate(VTK):
        f = 1 + c * p
        N[a] = f.prod() / 8
        for j in range(3):
            dN[a, j] = c[j] * np.prod([f[k] for k in range(3) if k != j]) / 8
    return N, dN


def jac(X, p):
    return X.T @ shape(p)[1]  # J_ij = dx_i / dxi_j


def covariant(X, p):
    """(6, 24): rows E11, E22, E33, 2E23, 2E13, 2E12 of E_ij = (g_i.u_,j + g_j.u_,i) / 2."""
    dN = shape(p)[1]
    g = jac(X, p).T
    B = np.zeros((6, 24))
    for r, (i, j) in enumerate(PAIRS):
        s = 1 if i == j else 2
        for a in range(8):
            B[r, 3 * a : 3 * a + 3] = s * (g[i] * dN[a, j] + g[j] * dN[a, i]) / 2
    return B


def to_cartesian(J):
    """T with eps = T E: eps_kl = (J^-1)_ik E_ij (J^-1)_jl, engineering shear both sides."""
    Ji, T = np.linalg.inv(J), np.zeros((6, 6))
    for c in range(6):
        E = np.zeros((3, 3))
        i, j = PAIRS[c]
        E[i, j] = E[j, i] = 1.0 if i == j else 0.5
        eps = Ji.T @ E @ Ji
        T[:, c] = [eps[k, n] * (1 if k == n else 2) for k, n in PAIRS]
    return T


def rotate_tensor(C, R):
    """C'_ijkl = R_ia R_jb R_kc R_ld C_abcd, back to Voigt."""
    T = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, tensor(C))
    return np.array([[T[i, j, k, n] for k, n in PAIRS] for i, j in PAIRS])


def rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def frame(X, R_o):
    J0 = jac(X, np.zeros(3))
    n = np.cross(J0[:, 0], J0[:, 1])
    t3 = n / np.linalg.norm(n)
    p1 = R_o[:, 0] - (R_o[:, 0] @ t3) * t3
    if np.linalg.norm(p1) >= 0.1:
        t1 = p1 / np.linalg.norm(p1)
        t2 = np.cross(t3, t1)
    else:
        p2 = R_o[:, 1] - (R_o[:, 1] @ t3) * t3
        t2 = p2 / np.linalg.norm(p2)
        t1 = np.cross(t2, t3)
    return np.column_stack([t1, t2, t3])


def ans(X, p):
    """Covariant ANS strain matrix (6, 24) at p."""
    xi, eta, zeta = p
    B = covariant(X, p)
    B[4] = (1 - eta) / 2 * covariant(X, [0, -1, zeta])[4] + (1 + eta) / 2 * covariant(
        X, [0, 1, zeta]
    )[4]
    B[3] = (1 - xi) / 2 * covariant(X, [-1, 0, zeta])[3] + (1 + xi) / 2 * covariant(
        X, [1, 0, zeta]
    )[3]
    B[2] = sum(
        (1 + a * xi) * (1 + b * eta) / 4 * covariant(X, [a, b, 0])[2] for a, b in VTK[:4, :2]
    )
    return B


def eas(X, p):
    xi, eta, zeta = p
    J0 = jac(X, np.zeros(3))
    M = np.zeros((6, 5))
    M[0, 0], M[1, 1], M[5, 2], M[5, 3], M[2, 4] = xi, eta, xi, eta, zeta
    return np.linalg.det(J0) / np.linalg.det(jac(X, p)) * to_cartesian(J0) @ M


def points(plies):
    """(p, w, ply index) over plies, in-plane xi fastest, then the 2 ply Gauss points."""
    out, z0 = [], -1.0
    for k, (_, _, _, f) in enumerate(plies):
        for g in (-G, G):
            zeta = z0 + f + f * g
            for eta in (-G, G):
                for xi in (-G, G):
                    out.append((np.array([xi, eta, zeta]), f, k))
        z0 += 2 * f
    return out


def matrices(X, plies, R_o):
    R = frame(X, R_o)
    Kuu, Kua, Kaa = np.zeros((24, 24)), np.zeros((24, 5)), np.zeros((5, 5))
    data = []
    for p, w, k in points(plies):
        C, _, angle, _ = plies[k]
        Cp = rotate_tensor(C, R @ rz(angle))
        J = jac(X, p)
        TB, Gm, wd = to_cartesian(J) @ ans(X, p), eas(X, p), w * np.linalg.det(J)
        Kuu += wd * TB.T @ Cp @ TB
        Kua += wd * TB.T @ Cp @ Gm
        Kaa += wd * Gm.T @ Cp @ Gm
        data.append((shape(p)[0] @ X, wd, TB, Gm, Cp))
    return Kuu, Kua, Kaa, data


def stiffness(X, plies, R_o=I3):
    Kuu, Kua, Kaa, _ = matrices(X, plies, R_o)
    return Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)


def mass(X, plies):
    m = np.zeros((8, 8))
    for p, w, k in points(plies):
        N = shape(p)[0]
        m += plies[k][1] * w * np.linalg.det(jac(X, p)) * np.outer(N, N)
    return m


def recover(X, plies, u, R_o=I3):
    """(points, weights, eps, sigma, alpha)."""
    _, Kua, Kaa, data = matrices(X, plies, R_o)
    alpha = -np.linalg.solve(Kaa, Kua.T @ u)
    eps = [TB @ u + Gm @ alpha for _, _, TB, Gm, _ in data]
    sig = [d[4] @ e for d, e in zip(data, eps, strict=True)]
    x, w = np.array([d[0] for d in data]), np.array([d[1] for d in data])
    return x, w, np.array(eps), np.array(sig), alpha


def det_ok(X, plies=((None, None, None, 1.0),)):
    """det J > 0 at the nodes and at the ply points."""
    pts = list(VTK) + [p for p, _, _ in points(list(plies))]
    return min(np.linalg.det(jac(X, p)) for p in pts) > 0


def distorted_hex8(rng, ht, plies=((None, None, None, 1.0),)):
    """Random element as test_elements.distorted (edges in [1, 2], nodes moved <= 0.2 x the
    shortest edge), scaled along the zeta edges to h/t = ht, then rigidly rotated and moved."""
    while True:
        L = rng.uniform(1, 2, 3)
        X = VTK / 2 * L + rng.uniform(-1, 1, (8, 3)) * 0.2 * L.min()
        mid = X[:, 2].reshape(2, 4).mean(0)  # per corner line: bottom/top mean height
        X[:, 2] = np.tile(mid, 2) + (X[:, 2] - np.tile(mid, 2)) * L[:2].mean() / (ht * L[2])
        X = X @ Rotation.random(random_state=rng).as_matrix().T + rng.uniform(-5, 5, 3)
        if det_ok(X, plies):
            return X
