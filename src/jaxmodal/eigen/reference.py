"""SciPy reference eigensolver: the lowest modes of (K_ff, M_ff) by eigsh shift-invert at a
negative shift, certified by residuals, M-orthonormality and an inertia count. See
docs/eigensolvers.md."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from ..operators.constraints import restrict

EPS = np.finfo(float).eps
ND_LEAF = 64  # nested-dissection leaf size in DOFs
ORTH_TOL = 1e-10  # ||V^T M V - I||_max
RESID_TOL = 100  # residual bound in units of eps (||K|| + |lam| ||M||) ||v||
GAP_REL, GAP_ABS = 1e-6, 100  # gap > max(GAP_REL |lam_{n+1}|, GAP_ABS eps s_bar)


def _order(A, x):
    """Geometric nested dissection of the DOFs of A at coordinates x (n, 3): split at the median
    of the widest axis; left DOFs coupled to the right form the separator, ordered last."""
    B = abs(sp.csr_array(A))
    out, right = [], np.zeros(A.shape[0])

    def split(s):
        if len(s) <= ND_LEAF:
            return out.append(s)
        c = x[s, np.ptp(x[s], 0).argmax()]
        med = np.median(c)
        left = c <= med  # keeps whole node planes on one side
        if left.all():
            left = c < med
        if not left.any():
            return out.append(s)
        L = s[left]
        right[s[~left]] = 1.0
        sep = B[L] @ right > 0
        right[s[~left]] = 0.0
        split(L[~sep])
        split(s[~left])
        out.append(L[sep])

    split(np.arange(A.shape[0]))
    return np.concatenate(out)


def _lu(A, p, spec):
    """SuperLU of A[p][:, p] with diagonal pivots only, so perm_r == perm_c and diag(U) = D."""
    return splu(
        sp.csc_array(A)[p][:, p],
        permc_spec=spec,
        diag_pivot_thresh=0.0,
        options=dict(SymmetricMode=True),
    )


def _solve(Kf, Mf, k, sigma, p, spec):
    """k eigenpairs of (Kf, Mf) nearest sigma by shift-invert, sorted ascending."""
    lu = _lu(Kf - sigma * Mf, p, spec)

    def inv(b):
        y = np.empty(len(p))
        y[p] = lu.solve(np.asarray(b, float).ravel()[p])
        return y

    op = LinearOperator(Kf.shape, inv, dtype=float)
    v0 = np.random.default_rng(0).standard_normal(Kf.shape[0])
    lam, v = eigsh(Kf, k, M=Mf, sigma=sigma, OPinv=op, tol=0, v0=v0)
    o = np.argsort(lam)
    return lam[o], v[:, o]


def _certify(Kf, Mf, lam, v, n_modes, s_bar, p, spec):
    """Raise unless (lam, v) are accurate M-orthonormal eigenpairs and none is missing below
    the last gap at or below n_modes (Sylvester count of Kf - mu Mf)."""
    MV = Mf @ v
    orth = np.abs(v.T @ MV - np.eye(len(lam))).max()
    if orth > ORTH_TOL:
        raise RuntimeError(f"eigsh vectors are not M-orthonormal: {orth:.2e}")
    nK, nM = (abs(A).sum(1).max() for A in (Kf, Mf))
    res = np.abs(Kf @ v - MV * lam).max(0)
    bound = RESID_TOL * EPS * (nK + np.abs(lam) * nM) * np.abs(v).max(0)
    if (res > bound).any():
        raise RuntimeError(f"eigsh residuals too large: {(res / bound).max() * RESID_TOL:.1f} eps")
    tau = max(GAP_REL * abs(lam[-1]), GAP_ABS * EPS * s_bar)
    gaps = np.flatnonzero(np.diff(lam) > tau)
    if not len(gaps):
        raise ValueError(f"no gap among the lowest {n_modes + 1} eigenvalues; ask for more modes")
    j = gaps[-1] + 1  # modes below the last gap
    mu = (lam[j - 1] + lam[j]) / 2
    lu = _lu(Kf - mu * Mf, p, spec)  # sparse, unpivoted count (docs/eigensolvers.md)
    if not (lu.perm_r == lu.perm_c).all():
        raise RuntimeError("SuperLU pivoted off the diagonal: inertia count unavailable")
    c = int((lu.U.diagonal() < 0).sum())
    if c != j:
        raise RuntimeError(f"eigsh missed modes: {c} eigenvalues below {mu:.6g}, {j} found")


def reference_modes(K, M, n_modes, free=None, sigma=None, points=None):
    """Lowest n_modes eigenpairs of K_ff v = lam M_ff v: lam (n_modes,) ascending, V (n_dof,
    n_modes) with constrained rows 0 and V^T M V = I. K, M: assembled SciPy matrices; free: bool
    mask (None = all free); sigma: None (automatic) or a negative shift; points: the mesh.points
    K, M were assembled from (DOF 3n + i), enabling nested dissection (much faster for bulky
    meshes). Wrong points only slow the factorization; they never change the result.
    """
    Kf, Mf, idx = restrict(K, M, free)
    n, k = len(idx), n_modes + 1
    if not 1 <= n_modes or k >= n:
        raise ValueError(f"need 1 <= n_modes and n_modes + 1 < {n} free DOFs, got {n_modes}")
    if sigma is not None and not (np.isfinite(sigma) and sigma < 0):
        raise ValueError(f"sigma must be None or a finite negative shift, got {sigma}")
    if points is None:
        p, spec = np.arange(n), "MMD_AT_PLUS_A"
    else:
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or 3 * len(points) != K.shape[0]:
            raise ValueError(f"points must be (n_dof / 3, 3), got {points.shape}")
        p, spec = _order(Kf, points[idx // 3]), "NATURAL"
    s_bar = Kf.diagonal().mean() / Mf.diagonal().mean()
    if sigma is None:
        s0 = -np.sqrt(EPS) * s_bar
        lam, v = _solve(Kf, Mf, k, s0, p, spec)
        # near-zero modes below a lam_k that is not itself zero: re-solve near lam_k
        if lam[0] < 100 * abs(s0) and lam[-1] > GAP_ABS * EPS * s_bar:
            lam, v = _solve(Kf, Mf, k, -0.3 * lam[-1], p, spec)
    else:
        lam, v = _solve(Kf, Mf, k, sigma, p, spec)
    _certify(Kf, Mf, lam, v, n_modes, s_bar, p, spec)
    V = np.zeros((K.shape[0], n_modes))
    V[idx] = v[:, :n_modes]
    return lam[:n_modes], V
