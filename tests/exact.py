"""Exact reference spectra (in-test oracles). See docs/verification.md."""

import functools
import itertools

import numpy as np
import scipy.linalg as sl
from helpers import tensor
from numpy.polynomial import legendre as lg
from scipy.optimize import brentq
from scipy.special import spherical_jn

SLIDING = {"xmin": "x", "xmax": "x", "ymin": "y", "ymax": "y", "zmin": "z", "zmax": "z"}
SIMPLY_SUPPORTED = {"xmin": "yz", "xmax": "yz", "ymin": "xz", "ymax": "xz"}


def sliding_block(C, rho, L, n, kmax=8):
    """Lowest n eigenvalues (sorted) of a box L with sliding BCs on all faces, for an
    orthotropic C (Voigt, engineering shear) with axes aligned to the box.

    Mode (p, q, r) has u_x ~ sin(k_x x) cos(k_y y) cos(k_z z), etc., k = pi (p, q, r) / L;
    component i exists only for index i >= 1. Indices run to kmax; every mode with an index
    above kmax has lam >= lam_min(C) pi^2 (kmax + 1)^2 / (2 rho max(L)^2), checked below.
    """
    C, L = np.asarray(C, float), np.asarray(L, float)
    T, lam = tensor(C), []
    for pqr in itertools.product(range(kmax + 1), repeat=3):
        k = np.pi * np.array(pqr) / L
        G = np.einsum("ijkl,j,l->ik", T, k, k)  # Christoffel: Navier equations for the ansatz
        s = np.flatnonzero(np.array(pqr) > 0)
        lam.extend(np.linalg.eigvalsh(G[np.ix_(s, s)]) / rho)
    lam = np.sort(lam)[:n]
    bound = np.linalg.eigvalsh(C)[0] * np.pi**2 * (kmax + 1) ** 2 / (2 * rho * L.max() ** 2)
    if lam[-1] >= bound:
        raise ValueError(f"kmax = {kmax} too small for {n} modes")
    return lam


def _legendre(x, N):
    """Legendre P_0..P_N and their derivatives at x, (len(x), N + 1) each."""
    return lg.legvander(x, N), np.array([lg.legval(x, lg.legder(c)) for c in np.eye(N + 1)]).T


def _basis(xn, x):
    """Lagrange basis on nodes xn and its derivative at points x, (len(x), len(xn)) each."""
    Vi = np.linalg.inv(lg.legvander(xn, len(xn) - 1))
    P, dP = _legendre(x, len(xn) - 1)
    return P @ Vi, dP @ Vi


@functools.cache
def _gll(P):
    """GLL nodes of degree P, Gauss rule of P + 3 points, basis tables at the Gauss points."""
    xn = np.r_[-1, np.sort(lg.legroots(lg.legder([0] * P + [1])).real), 1]
    xg, wg = lg.leggauss(P + 3)
    return xn, wg, *_basis(xn, xg)


def _plate_block(plies, p, q, P):
    """1D Ritz in z for wavenumbers (p, q): eigenvalues and profiles (3, n_nodes, n_modes)."""
    xn, wg, N, dN = _gll(P)
    nn = len(plies) * P + 1
    K, M = np.zeros((3, nn, 3, nn)), np.zeros((3, nn, 3, nn))
    NN = np.einsum("g,ga,gb->ab", wg, N, N)
    for k, (C, rho, t) in enumerate(plies):
        J, s, D = t / 2, slice(k * P, k * P + P + 1), dN / (t / 2)
        B = np.zeros((len(wg), 6, 3, P + 1))  # strains (-pU, -qV, W', V'+qW, U'+pW, qU+pV)
        B[:, 0, 0], B[:, 1, 1], B[:, 2, 2] = -p * N, -q * N, D
        B[:, 3, 1], B[:, 3, 2], B[:, 4, 0], B[:, 4, 2] = D, q * N, D, p * N
        B[:, 5, 0], B[:, 5, 1] = q * N, p * N
        BCB = np.einsum("g,geca,ef,gfdb->cadb", wg, B, np.asarray(C), B, optimize=True)
        K[:, s, :, s] += J * BCB
        for c in range(3):
            M[c, s, c, s] += J * rho * NN
    keep = np.repeat([q > 0, p > 0, p > 0 and q > 0], nn)  # component rule
    K, M = K.reshape(3 * nn, -1)[np.ix_(keep, keep)], M.reshape(3 * nn, -1)[np.ix_(keep, keep)]
    lam, X = sl.eigh(K, M)
    full = np.zeros((3 * nn, len(lam)))
    full[keep] = X
    return lam, full.reshape(3, nn, -1)


def plate_modes(plies, a, b, k, points=None, P=12, mmax=8):
    """Lowest k modes of a simply supported plate [0, a] x [0, b] (Srinivas et al. 1970).

    plies: list of (C, rho, t) stacked from z = 0, C orthotropic in the plate axes. Mode
    (m, n) has u = U(z) cos(px) sin(qy), v = V(z) sin(px) cos(qy), w = W(z) sin(px) sin(qy),
    p = m pi / a, q = n pi / b; U exists iff n >= 1, V iff m >= 1, W iff both. U, V, W from a
    1D Ritz in z (GLL degree P per ply, C0 across plies). Returns lam (k,), labels (k, 3)
    (m, n, family) and Phi (3 n_points, k) at points (None without points).
    """
    blocks, rows = {}, []
    for m, n in itertools.product(range(mmax + 1), repeat=2):
        if m or n:
            blocks[m, n] = _plate_block(plies, m * np.pi / a, n * np.pi / b, P)
            rows += [(lam, m, n, f) for f, lam in enumerate(blocks[m, n][0])]
    rows.sort()
    lam, labels = np.array([r[0] for r in rows[:k]]), np.array([r[1:] for r in rows[:k]])
    edge = min(r[0] for r in rows if max(r[1], r[2]) == mmax)
    if edge <= 2 * lam[-1]:  # a check, not a bound: see docs/verification.md
        raise ValueError(f"mmax = {mmax} too small for {k} modes")
    if points is None:
        return lam, labels, None
    x, y, z = np.asarray(points, float).T
    zu, inv = np.unique(z, return_inverse=True)
    t = np.array([ply[2] for ply in plies])
    zi = np.r_[0, np.cumsum(t)]
    ply = np.clip(np.searchsorted(zi, zu, side="right") - 1, 0, len(plies) - 1)
    Nz = np.zeros((len(zu), len(plies) * P + 1))  # z profiles at the distinct node z
    cols = ply[:, None] * P + np.arange(P + 1)
    np.put_along_axis(Nz, cols, _basis(_gll(P)[0], 2 * (zu - zi[ply]) / t[ply] - 1)[0], 1)
    Phi = []
    for m, n, f in labels:
        p, q = m * np.pi / a, n * np.pi / b
        U = (Nz @ blocks[m, n][1][:, :, f].T)[inv]
        sx, cx, sy, cy = np.sin(p * x), np.cos(p * x), np.sin(q * y), np.cos(q * y)
        Phi.append(np.stack([U[:, 0] * cx * sy, U[:, 1] * sx * cy, U[:, 2] * sx * sy], 1).ravel())
    return lam, labels, np.stack(Phi, 1)


def visscher(C, rho, L, N, k=16):
    """Lowest k eigenvalues of the free box [0, L] for any C: Rayleigh-Ritz with Legendre
    products P_a P_b P_c, a + b + c <= N, per component (Visscher et al. 1991).

    Point inversion u(x) -> -u(-x) splits the basis by the parity of a + b + c into two
    uncoupled blocks; M is diagonal (Legendre orthogonality), so each is a standard problem.
    """
    T, L = tensor(np.asarray(C, float)), np.asarray(L, float)
    x, w = lg.leggauss(N + 2)
    Pd = [A.T for A in _legendre(x, N)]  # P_a and P_a' at the Gauss points
    G = {(i, j): (Pd[i] * w) @ Pd[j].T for i in (0, 1) for j in (0, 1)}
    abc = np.array([i for i in itertools.product(range(N + 1), repeat=3) if sum(i) <= N])
    vol, out = np.prod(L) / 8, []
    for par in (0, 1):
        idx = abc[abc.sum(1) % 2 == par]
        # g[d][da, db]: 1D integrals of d^da P_I d^db P_J along axis d, scaled to the box
        g = [{(da, db): G[da, db][np.ix_(i, i)] * (2 / Ld) ** (da + db) for da, db in G}
             for i, Ld in zip(idx.T, L, strict=True)]  # fmt: skip
        D = {(j, n): vol * g[0][j == 0, n == 0] * g[1][j == 1, n == 1] * g[2][j == 2, n == 2]
             for j in range(3) for n in range(3)}  # fmt: skip
        K = np.block([[sum(T[i, j, m, n] * D[j, n] for j in range(3) for n in range(3))
                       for m in range(3)] for i in range(3)])  # fmt: skip
        mass = rho * vol * np.prod([G[0, 0].diagonal()[idx[:, d]] for d in range(3)], 0)
        s = np.tile(mass, 3) ** -0.5
        K *= s[:, None] * s
        out.append(sl.eigh(K, eigvals_only=True, subset_by_index=[0, k - 1], overwrite_a=True))
    return np.sort(np.concatenate(out))[:k]


def mac(A, B, M):
    """M-weighted MAC matrix (a_i^T M b_j)^2 / ((a_i^T M a_i)(b_j^T M b_j))."""
    MA, MB = M @ A, M @ B
    return (A.T @ MB) ** 2 / (np.einsum("ij,ij->j", A, MA)[:, None] * np.einsum("ij,ij->j", B, MB))


def lamb_sphere(nu, k, nmax=8, top=12.0):
    """Lowest k distinct elastic roots eta = omega a / c_s of a free isotropic sphere (Lamb 1882),
    sorted, as (eta, kind, n): torsional "T" and spheroidal "S" of order n = 1..nmax, with
    multiplicity 2n + 1. Both frequency functions are entire (the spheroidal one is the usual
    form times j_n(eta) j_n(xi)), so every sign change on (0.3, top] is a root. Raises unless
    the k-th root lies below the lowest root of order nmax.
    """
    r = np.sqrt((1 - 2 * nu) / (2 * (1 - nu)))  # c_s / c_p

    def tors(e, n):
        return (n - 1) * spherical_jn(n, e) - e * spherical_jn(n + 1, e)

    def spher(e, n):
        x = r * e
        je, ge = spherical_jn(n, e), e * spherical_jn(n + 1, e)
        jx, gx = spherical_jn(n, x), x * spherical_jn(n + 1, x)
        a = (n - 1) * (n + 2)
        return (2 * (e**2 * je + a * (ge - (n + 1) * je)) * gx
                + (-(e**4) / 2 + (n - 1) * (2 * n + 1) * e**2) * je * jx
                + (e**2 - 2 * n * a) * ge * jx)  # fmt: skip

    grid, out = np.linspace(0.3, top, 4000), []
    for n, (kind, f) in itertools.product(range(1, nmax + 1), (("T", tors), ("S", spher))):
        y = f(grid, n)
        for i in np.flatnonzero(y[:-1] * y[1:] < 0):
            out.append((brentq(f, grid[i], grid[i + 1], (n,), xtol=1e-14), kind, n))
    out, edge = sorted(out), min((e for e, _, n in out if n == nmax), default=top)
    if len(out) < k or out[k - 1][0] >= edge:  # lowest roots rise with n: a check, not a bound
        raise ValueError(f"nmax = {nmax}, top = {top} too small for {k} roots")
    return out[:k]
