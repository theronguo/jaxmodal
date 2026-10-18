"""Shared test references."""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as spla

from jaxmodal.elements import check_elements, kernels
from jaxmodal.materials import Material, from_euler, isotropic, orthotropic, rotate
from jaxmodal.mesh import Mesh, box_mesh
from jaxmodal.operators import assemble, dof_mask, restrict, unstructured

ORTHO = orthotropic(10.0, 4.0, 2.0, 0.3, 0.2, 0.25, 1.5, 1.2, 0.8)
R = np.asarray(from_euler(np.array([0.3, 1.1, -0.7])))
MATERIALS = {0: Material(isotropic(3.0, 0.3), 2.0), 1: Material(ORTHO, 0.7)}
PAG = np.asarray(orthotropic(25.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.2))
RZ = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z
H = 0.1  # plates [0, 1]^2 x [0, H]
PLATES = {
    "ortho": [(np.asarray(ORTHO), 0.7, H)],
    "pag": [(PAG, 1.0, H / 3), (np.asarray(rotate(PAG, RZ)), 1.0, H / 3), (PAG, 1.0, H / 3)],
}
PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]  # Voigt index -> tensor indices


def rel(a, b):
    """Principle 1: ||a - b||_inf / ||b||_inf."""
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max()


def fd_rel(f, x, g):
    """rel(g, central differences of f at x), best over steps 1e-4..1e-6 x max(|x|, 1)."""
    best = np.inf
    for h in (1e-4, 1e-5, 1e-6):
        step = h * np.maximum(np.abs(x), 1)
        fd = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            e = np.zeros_like(x)
            e[idx] = step[idx]
            fd[idx] = (f(x + e) - f(x - e)) / (2 * step[idx])
        best = min(best, rel(g, fd))
    return best


def tensor(C):
    """Voigt (6, 6) stiffness -> 4th-order tensor C_ijkl."""
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(PAIRS):
        for b, (k, n) in enumerate(PAIRS):
            for p, q in ((i, j), (j, i)):
                for r, s in ((k, n), (n, k)):
                    T[p, q, r, s] = C[a, b]
    return T


def mixed_mesh(rng):
    """hex8 and hex20 2x2x2 boxes glued at x = 2 (shared nodes; the interface is deliberately
    nonconforming and only exercises DOFs shared across blocks), nodes perturbed <= 5%."""
    a, b = (
        box_mesh((2, 2, 2), type="hex8", origin=(2.0, 0.0, 0.0)),
        box_mesh((2, 2, 2), type="hex20"),
    )
    pts = np.vstack([a.points, b.points])
    key, first, inv = np.unique(np.rint(2 * pts), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    points = pts[first] + rng.uniform(-0.05, 0.05, (len(first), 3))
    m = Mesh(
        points=points,
        blocks={"hex8": inv[a.blocks["hex8"]], "hex20": inv[len(a.points) + b.blocks["hex20"]]},
        node_sets={},
        element_sets={},
        material=np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]),
        orientation=np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0]),
    )
    check_elements(m, mass="lumped")
    return m


def dense_reference(m, lumped, s_K=None, s_M=None):
    """Dense K, M by an explicit element loop (per-element rotate, optional scalings)."""
    s_K = np.ones(m.n_el) if s_K is None else s_K
    s_M = np.ones(m.n_el) if s_M is None else s_M
    n = 3 * len(m.points)
    K, M = np.zeros((n, n)), np.zeros((n, n))
    e = 0
    for t, conn in m.blocks.items():
        for row in conn:
            mat = MATERIALS[m.material[e]]
            C = rotate(mat.C, R if m.orientation[e] == 1 else np.eye(3))
            X = m.points[row]
            d = [3 * a + i for a in row for i in range(3)]
            K[np.ix_(d, d)] += s_K[e] * kernels.stiffness(t, X, C)
            if lumped:
                M[d, d] += s_M[e] * kernels.lumped_mass(t, X, mat.rho)
            else:
                M[np.ix_(d, d)] += s_M[e] * kernels.mass(t, X, mat.rho)
            e += 1
    return K, M


def distort(m, curved, rng):
    """Interior corner nodes moved per component <= 15% of the edge; other nodes on the
    trilinear map of the corners (straight), plus <= 5% for non-corner interior nodes (curved)."""
    ijk = np.rint(2 * m.points).astype(int)  # unit spacing, origin 0
    top = ijk.max(0)
    P = (np.indices(top // 2 + 1).transpose(1, 2, 3, 0)).astype(float)
    inner = np.zeros(P.shape[:3], bool)
    inner[1:-1, 1:-1, 1:-1] = True
    P[inner] += rng.uniform(-0.15, 0.15, (inner.sum(), 3))
    lo, hi = ijk // 2, (ijk + 1) // 2
    pts = np.mean([P[tuple(np.where(np.array(s), hi, lo).T)] for s in np.ndindex(2, 2, 2)], 0)
    interior = ((ijk > 0) & (ijk < top)).all(1) & (ijk % 2).any(1)
    if curved:
        pts[interior] += rng.uniform(-0.05, 0.05, (interior.sum(), 3))
    return Mesh(pts, m.blocks, m.node_sets, m.element_sets, m.material, m.orientation)


@functools.cache
def box(bcs="clamped"):
    """PR 10 solver box, hex20 6x5x4, material 0, xmin clamped or free: (op, free, K, M, lam),
    lam the lowest 8 nonzero eigenvalues."""
    mesh = box_mesh((6, 5, 4), spacing=(1 / 6, 0.83 / 5, 0.67 / 4), type="hex20")
    free = dof_mask(mesh, {"xmin": "xyz"} if bcs == "clamped" else {})
    K, M = assemble(mesh, MATERIALS)
    Kf, Mf, _ = restrict(K, M, free)
    v0 = np.ones(Kf.shape[0])  # a fixed ARPACK start: sigma = lam[0] identical in every run
    lam = np.sort(spla.eigsh(Kf, 14, M=Mf, sigma=-1.0, v0=v0)[0])
    return unstructured(mesh, MATERIALS), free, K, M, lam[lam > 1e-6][:8]


def rhs(free, k, seed=1):
    """(n_dof, k) random right-hand sides, zero on constrained DOFs."""
    return np.random.default_rng(seed).standard_normal((len(free), k)) * free[:, None]


STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}


def dense(f, n):
    """Dense matrix of the linear map f on (n, k) blocks."""
    return np.asarray(f(jnp.eye(n)))


def lam1(mesh, free, **kw):
    """Lowest nonzero steel eigenvalue by eigsh (fixed v0); kw go to assemble (nodal_mass, s_K)."""
    K, M = assemble(mesh, STEEL, **kw)
    Kf, Mf, _ = restrict(K, M, free)
    lam = spla.eigsh(Kf, 8, M=Mf, sigma=-1.0, v0=np.ones(Kf.shape[0]))[0]
    return float(np.sort(lam[lam > 1e-6 * lam.max()])[0])


def cube(t, n):
    return box_mesh((n,) * 3, (1 / n,) * 3, type=t)


@functools.cache
def sigma6(t):
    """lambda_1 of the clamped steel 6^3 box: one shift for a whole h-sequence."""
    m = cube(t, 6)
    return lam1(m, dof_mask(m, {"xmin": "xyz"}))


def checkerboard(mesh, op):
    """op with s_K = 1e-3 on a checkerboard of the hex20 unit-box elements (4 per unit)."""
    c = np.floor(mesh.points[mesh.blocks["hex20"][:, :8]].mean(1) * 4).astype(int).sum(1) % 2
    return op.scaled(s_K=jnp.asarray(np.where(c == 1, 1e-3, 1.0)))


def step_bytes(op, P, low, k=15):
    """Temporaries + aliased bytes of the compiled LOBPCG step (block k) with precond P."""
    from test_lobpcg import step_args

    lob = importlib.import_module("jaxmodal.eigen.lobpcg")
    c, _ = lob._capacity(100, 1, k, low)
    f = lob.lobpcg_step_low if low else lob.lobpcg_step
    args = list(step_args(op, op.n_dof, k, low, 0))
    args[1], args[3] = op, P
    mem = f.lower(*args, c=c).compile().memory_analysis()
    return mem.temp_size_in_bytes + mem.alias_size_in_bytes


def owned_bytes(P, op):
    """Device bytes of P's leaves, de-duplicated, without the operator's."""
    shared = {id(x) for x in jax.tree_util.tree_leaves(op)}
    leaves = {id(x): x for x in jax.tree_util.tree_leaves(P) if isinstance(x, jax.Array)}
    return sum(x.nbytes for i, x in leaves.items() if i not in shared)


def r_M(Kf, Mf, L, lam, X):
    """lam-scaled residuals in the M^-1 norm of M-normalized columns X (free DOFs), dense; L the
    dense lower Cholesky factor of Mf."""
    X = X / np.sqrt(np.einsum("ij,ij->j", X, Mf @ X))
    res = Kf @ X - (Mf @ X) * lam
    return np.linalg.norm(sl.solve_triangular(L, res, lower=True), axis=0) / np.abs(lam)


def max_sine(Mf, L, Q, Qr):
    """Largest principal sine between M-orthonormal Q and Qr, stably: sigma_max(L^T (Q - Qr Qr^T
    M Q)), never sqrt(1 - cos^2), which bottoms out near 1e-8."""
    E = Q - Qr @ (Qr.T @ (Mf @ Q))
    return np.linalg.norm(L.T @ E, 2)


def cheb(A, Dinv, lmax, degree, ratio):
    """Dense Chebyshev smoother matrix, the recurrence of precond.chebyshev."""
    hi = 1.1 * lmax
    lo = hi / ratio
    theta, delta = (hi + lo) / 2, (hi - lo) / 2
    sig = theta / delta
    rho = 1 / sig
    R = np.eye(len(A))
    d = Dinv @ R / theta
    x = d
    for _ in range(degree - 1):
        R = R - A @ d
        rho_new = 1 / (2 * sig - rho)
        d = rho_new * rho * d + 2 * rho_new / delta * (Dinv @ R)
        x, rho = x + d, rho_new
    return x


def vcycle(levels, Ps, Ci, m, U):
    def go(b, lvl):
        if lvl == len(levels):
            return Ci @ b
        A, S = levels[lvl]
        x = S @ b
        x = x + Ps[lvl] @ go(Ps[lvl].T @ (b - A @ x), lvl + 1)
        return x + S @ (b - A @ x)

    return m[:, None] * go(m[:, None] * U, 0)
