"""Mode normalization, M-weighted MAC, influence vectors, participation and effective mass,
clusters and the frequency table. See docs/postprocessing.md."""

import jax
import numpy as np

from ..operators.common import check_x64
from ..operators.constraints import _check_mask

DIRECTIONS = ("x", "y", "z", "rx", "ry", "rz")


def _cols(V, n_dof=None):
    """V as a host (n_dof, n) array (1-D: one column); n_dof a multiple of 3."""
    V = np.asarray(V)
    V = V[:, None] if V.ndim == 1 else V
    if V.ndim != 2 or V.shape[0] % 3 or (n_dof is not None and V.shape[0] != n_dof):
        raise ValueError(f"expected a ({n_dof or '3 n_nodes'}, n) array, got shape {V.shape}")
    return V


def _mbatches(M, B, batch):
    """(column slice, M B[:, slice] on the host) over batches of width min(batch, n); the last
    batch is zero-padded (one compile per width), the next batch is dispatched before the
    current one is pulled. M: an operator with apply_M, a matrix, or None (identity)."""
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    n = B.shape[1]
    w = max(min(batch, n), 1)

    def pull(cols, y):
        return cols, np.asarray(y)[:, : cols.stop - cols.start]

    pending = None
    for j in range(0, n, w):
        cols = slice(j, min(j + w, n))
        b = np.asarray(B[:, cols], np.float64)
        if b.shape[1] < w:
            b = np.pad(b, ((0, 0), (0, w - b.shape[1])))
        if M is None:
            y = b
        elif hasattr(M, "apply_M"):
            y = M.apply_M(jax.device_put(b))  # async: overlaps the previous batch's host work
        else:
            y = np.asarray(M @ b)
        if pending is not None:
            yield pull(*pending)
        pending = cols, y
    if pending is not None:
        yield pull(*pending)


def _gram(A, M, B, batch=64):
    """Host A^T M B and diag(B^T M B) (fp64), M applied once to batches of B."""
    A = np.asarray(A)
    G, d = np.zeros((A.shape[1], B.shape[1])), np.zeros(B.shape[1])
    for cols, MB in _mbatches(M, B, batch):
        G[:, cols] = A.T @ MB
        d[cols] = np.einsum("ij,ij->j", np.asarray(B[:, cols], np.float64), MB)
    return G, d


def _mdiag(M, B, batch=64):
    """diag(B^T M B)."""
    return _gram(B[:, :0], M, B, batch)[1]


def _nonzero(d, what):
    bad = np.flatnonzero(~(d > 0))
    if len(bad):
        raise ValueError(f"{what} has zero (non-positive or NaN M-norm) columns {bad}")


def _blocks(n, size=64):
    return (slice(c, min(c + size, n)) for c in range(0, n, size))


def normalize(V, M=None, kind="mass", batch=64):
    """Scaled copy of the modes V (n_dof, n) (1-D: one mode). kind "mass": phi^T M phi = 1;
    "max": the largest nodal displacement norm is 1. Sign: the lowest-index DOF with |v| within
    a relative 1e-6 of max |v| is positive. Typical calls: normalize(V, op), normalize(V,
    kind="max")."""
    if kind not in ("mass", "max"):
        raise ValueError(f"kind must be 'mass' or 'max', got {kind!r}")
    if kind == "mass" and M is None:
        raise ValueError("normalize(kind='mass') needs M: pass M or kind='max'")
    check_x64("normalize")
    one = np.ndim(V) == 1
    V = np.asarray(_cols(V), np.float64)  # no copy for host fp64
    n = V.shape[1]
    j, s = np.zeros(n, int), np.zeros(n)
    for c in _blocks(n):  # column blocks: temporaries of 64 columns, not of V
        a = np.abs(V[:, c])
        j[c] = np.argmax(a >= (1 - 1e-6) * a.max(0, initial=0), axis=0)
        if kind == "max":
            s[c] = np.linalg.norm(V[:, c].reshape(len(V) // 3, 3, -1), axis=1).max(0, initial=0)
    if kind == "mass":
        s = np.sqrt(np.maximum(_mdiag(M, V, batch), 0))
    _nonzero(s, "V")
    out = V * (np.sign(V[j, np.arange(n)]) / s)
    return out[:, 0] if one else out


def mac(A, B, M=None, batch=64):
    """MAC_ij = (a_i^T M b_j)^2 / ((a_i^T M a_i)(b_j^T M b_j)) (nA, nB); M=None: unweighted.
    M is applied once to each column of B (and of A unless B is A)."""
    check_x64("mac")
    same = B is A
    A = _cols(A)
    B = A if same else _cols(B, A.shape[0])
    G, db = _gram(A, M, B, batch)
    da = db if same else _mdiag(M, A, batch)
    _nonzero(da, "A")
    _nonzero(db, "B")
    return G**2 / (da[:, None] * db[None, :])


def influence(mesh, free, origin=None):
    """Rigid-body fields r (n_dof, 6): translations e_x, e_y, e_z and rotations e_k x (x - x0)
    about origin (default: the node centroid, as rigid_modes), zero on constrained DOFs. free:
    the bool (n_dof,) mask (True = free), or None for no constraints."""
    x = np.asarray(mesh.points, float)
    n = len(x)
    if free is not None:
        _check_mask(3 * n, free)
    x0 = x.mean(0) if origin is None else np.asarray(origin, float)
    if x0.shape != (3,):
        raise ValueError(f"origin must be a point (3,), got shape {x0.shape}")
    d = x - x0
    r = np.zeros((n, 3, 6))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        r[:, i, i] = 1.0
        r[:, k, 3 + i] = d[:, j]  # e_i x d
        r[:, j, 3 + i] = -d[:, k]
    r = r.reshape(3 * n, 6)
    return r if free is None else r * np.asarray(free)[:, None]


def participation(V, M, r, batch=64):
    """Participation factors gamma (n, 6), effective masses m_eff (n, 6) and totals r^T M r (6,)
    of the modes V (n_dof, n), M-orthogonal, any scaling (n = 0 allowed). Over all free modes
    (rigid included) sum_i m_eff = total."""
    check_x64("participation")
    r = _cols(r)
    V = _cols(V, r.shape[0])
    Mr = np.zeros(r.shape)
    for cols, MB in _mbatches(M, r, batch):
        Mr[:, cols] = MB
    b = V.T @ Mr
    mu = _mdiag(M, V, batch)
    _nonzero(mu, "V")
    return b / mu[:, None], b**2 / mu[:, None], np.einsum("ij,ij->j", r, Mr)


def clusters(lam, rtol=1e-4, atol=0.0):
    """Index arrays of clusters of the ascending lam: neighbours join when
    lam[i+1] - lam[i] <= rtol |lam[i+1]| + atol."""
    lam = np.asarray(lam, float)
    if lam.ndim != 1 or not np.isfinite(lam).all() or (np.diff(lam) < 0).any():
        raise ValueError("lam must be a finite, ascending 1-D array")
    if len(lam) == 0:
        return []
    d = np.diff(lam)
    split = np.flatnonzero(d > rtol * np.abs(lam[1:]) + atol) + 1
    return np.split(np.arange(len(lam)), split)


def cluster_rtol(info):
    """rtol matching the solver's cluster merge for a converged solve_modes result: 100 times the
    largest acceptance threshold (per-mode thresholds, the band slices' r_acc, at least 1e-6).
    The recipe passes max(1e-4, cluster_rtol(info))."""
    t = np.max(np.asarray(info["thresholds"], float), initial=0.0)
    return 100 * max(1e-6, t, info.get("r_acc", 0.0))


def frequency_table(lam, m_eff, total, rtol=1e-4, atol=0.0):
    """Per cluster: indices, mean lam, f (Hz), summed m_eff (basis invariant), fraction of the
    total (0 where the total is 0) and its cumulative sum; total echoed. After an n_modes solve
    the last cluster (and after a band from f1 > 0 the first) may be incomplete."""
    m_eff, total = np.asarray(m_eff, float), np.asarray(total, float)
    if m_eff.shape != (len(lam), 6) or total.shape != (6,):
        raise ValueError(
            f"m_eff must be (len(lam), 6) = ({len(lam)}, 6) and total (6,), got {m_eff.shape} "
            f"and {total.shape} (prepend zeros to lam for the rigid modes' rows)"
        )
    cl = clusters(lam, rtol, atol)
    lam = np.asarray(lam, float)
    lam_c = np.array([lam[c].mean() for c in cl])
    m = np.array([m_eff[c].sum(0) for c in cl]).reshape(-1, 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, m / total, 0.0)
    return {
        "clusters": cl,
        "lam": lam_c,
        "f": np.sqrt(np.maximum(lam_c, 0)) / (2 * np.pi),
        "m_eff": m,
        "fraction": frac,
        "cumulative": np.cumsum(frac, 0),
        "total": total,
    }


def format_table(t):
    """Fixed-width text: modes, f, the 6 fractions, the 3 translational cumulative fractions."""
    head = f"{'modes':>11} {'f [Hz]':>12} " + " ".join(f"{d:>7}" for d in DIRECTIONS)
    head += " " + " ".join(f"{'sum ' + d:>7}" for d in DIRECTIONS[:3])
    lines = [head]
    for c, f, fr, cu in zip(t["clusters"], t["f"], t["fraction"], t["cumulative"], strict=True):
        modes = f"{c[0] + 1}" if len(c) == 1 else f"{c[0] + 1}-{c[-1] + 1}"
        row = f"{modes:>11} {f:12.6g} " + " ".join(f"{x:7.4f}" for x in fr)
        lines.append(row + " " + " ".join(f"{x:7.4f}" for x in cu[:3]))
    return "\n".join(lines)
