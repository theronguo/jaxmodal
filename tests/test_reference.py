"""reference_modes, count_below and restrict against dense scipy.linalg.eigh."""

import dataclasses
import functools
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as sl
from helpers import MATERIALS, ORTHO, R, distort, mixed_mesh
from scipy.linalg import lapack

from jaxmodal.eigen import reference, reference_modes
from jaxmodal.linalg import count_below, inertia
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import assemble, dof_mask, restrict

EPS = np.finfo(float).eps
ORTHO_MAT = {0: Material(ORTHO, 0.7)}
STEEL = {0: Material(isotropic(200e9, 0.3), 7850.0)}


def _box(mass):
    m = box_mesh((4, 3, 3), spacing=(0.25, 0.83 / 3, 0.67 / 3), type="hex20")
    nodal = np.random.default_rng(1).uniform(0, 1e-3, len(m.points))
    nodal[::3] = 0.0
    return m, assemble(m, ORTHO_MAT, mass=mass, nodal_mass=nodal), {}, 6


@functools.cache
def case(name):
    """(mesh, (K, M), bcs, number of rigid modes)."""
    rng = np.random.default_rng(5)
    if name in ("box", "box_lumped"):
        return _box("lumped" if name == "box_lumped" else "consistent")
    if name == "clamped":
        m = distort(box_mesh((3, 3, 3), type="hex27"), True, rng)
        return m, assemble(m, ORTHO_MAT), {"xmin": "xyz"}, 0
    if name == "mixed":
        m = mixed_mesh(rng)
        m = dataclasses.replace(m, node_sets={"xmin": np.flatnonzero(m.points[:, 0] < 0.1)})
        return m, assemble(m, MATERIALS, {1: R}), {"xmin": "x"}, 3
    if name == "plate":
        m = box_mesh((8, 8, 1), spacing=(1.25, 1.25, 0.05), type="hex27")
        return m, assemble(m, STEEL), {}, 6
    if name == "cube":
        m = box_mesh((4, 4, 4), type="hex20")
        return m, assemble(m, {0: Material(isotropic(3.0, 0.3), 2.0)}), {}, 6
    if name == "thin":
        m = box_mesh((8, 8, 1), spacing=(1, 1, 0.005), type="hex27")
        return m, assemble(m, {0: Material(isotropic(3.0, 0.3), 2.0)}), {}, 6
    raise KeyError(name)


@functools.cache
def dense(name):
    """Free mask and all eigenvalues of (K_ff, M_ff)."""
    m, (K, M), bcs, _ = case(name)
    free = dof_mask(m, bcs)
    Kf, Mf, _ = restrict(K, M, free)
    return free, sl.eigh(Kf.toarray(), Mf.toarray(), eigvals_only=True)


def check(name, lam, V, n_rigid):
    """Step 6 criteria against dense eigh."""
    _, (K, M), _, _ = case(name)
    free, w = dense(name)
    lmax = w[-1]
    n = len(lam)
    assert np.abs(lam[:n_rigid]).max(initial=0) <= 10 * EPS * lmax
    el = slice(n_rigid, n)
    assert np.all(np.abs(lam[el] - w[el]) <= 2 * 10 * EPS * lmax)  # 2 f_lam lam
    assert np.abs(V.T @ (M @ V) - np.eye(n)).max() <= 1e-12
    assert np.all(V[~free] == 0)
    Kf, Mf, idx = restrict(K, M, free)
    v = V[idx]
    res = np.abs(Kf @ v - (Mf @ v) * lam).max(0)
    nK, nM = abs(Kf).sum(1).max(), abs(Mf).sum(1).max()
    assert np.all(res <= 100 * EPS * (nK + np.abs(lam) * nM) * np.abs(v).max(0))


@pytest.mark.parametrize("name", ["box", "box_lumped", "clamped", "mixed", "plate"])
def test_dense(name):
    m, (K, M), _, n_rigid = case(name)
    free, w = dense(name)
    lam, V = reference_modes(K, M, 20, free, points=m.points)
    check(name, lam, V, n_rigid)
    lam2, _ = reference_modes(K, M, 20, free, sigma=-0.01 * w[20], points=m.points)
    assert np.all(np.abs(lam2 - lam) <= 2 * 10 * EPS * w[-1])
    if name == "box":  # without points (MMD): the same as with nested dissection
        lam3, V3 = reference_modes(K, M, 20, free)
        check(name, lam3, V3, n_rigid)
        assert np.all(np.abs(lam3 - lam) <= 2 * 10 * EPS * w[-1])


def test_order():
    m, (K, _), _, _ = case("box")
    p = reference._order(K, np.repeat(m.points, 3, 0))
    assert np.array_equal(np.sort(p), np.arange(K.shape[0]))
    # 200 DOFs, widest axis with two values, 70% at the larger: the median split takes all
    rng = np.random.default_rng(0)
    x = np.c_[np.where(rng.random(200) < 0.7, 10.0, 0.0), rng.random((200, 2))]
    A = np.eye(200) + np.diag(np.ones(199), 1)
    p = reference._order(A + A.T, x)
    assert np.array_equal(np.sort(p), np.arange(200))


# --- certification ---


def _patched(monkeypatch, edit):
    """Replace reference._solve by edit(Kf, Mf, lam, v) applied to the real result."""
    solve = reference._solve
    monkeypatch.setattr(reference, "_solve", lambda Kf, Mf, *a: edit(Kf, Mf, *solve(Kf, Mf, *a)))


def test_missed_mode(monkeypatch):
    m, (K, M), _, _ = case("box")
    solve = reference._solve

    def drop(Kf, Mf, k, *a):  # one mode below the gap missing, one more from the top
        lam, v = solve(Kf, Mf, k + 1, *a)
        return np.delete(lam, 9), np.delete(v, 9, 1)

    monkeypatch.setattr(reference, "_solve", drop)
    with pytest.raises(RuntimeError, match="eigsh missed modes"):
        reference_modes(K, M, 20, points=m.points)


@pytest.mark.slow
def test_arpack_failure(monkeypatch):
    """ARPACK silently returns wrong pairs for |sigma| >> lam_k; the certification catches it."""
    m, (K, M), _, _ = case("plate")
    _, w = dense("plate")
    raw, solve = [], reference._solve
    monkeypatch.setattr(reference, "_solve", lambda *a: raw.append(solve(*a)) or raw[-1])
    sigma = -1e-2 * K.diagonal().mean() / M.diagonal().mean()  # 8e6 lam_7; ~40 s
    try:
        reference_modes(K, M, 20, sigma=sigma, points=m.points)
        error = None
    except RuntimeError as e:
        error = e
    if np.all(np.abs(raw[0][0][6:] - w[6:21]) <= 2 * 10 * EPS * w[-1]):
        pytest.skip("this ARPACK finds the right pairs at sigma = -1e-2 s_bar")
    assert "missed modes" in str(error)


def test_ghost(monkeypatch):
    m, (K, M), _, _ = case("box")

    def ghost(Kf, Mf, lam, v):
        lam, v = lam.copy(), v.copy()
        lam[9], v[:, 9] = lam[8], v[:, 8]
        return lam, v

    _patched(monkeypatch, ghost)
    with pytest.raises(RuntimeError, match="not M-orthonormal"):
        reference_modes(K, M, 20, points=m.points)


def test_perturbed(monkeypatch):
    m, (K, M), _, _ = case("box")

    def perturb(Kf, Mf, lam, v):
        r = np.random.default_rng(2).standard_normal(len(v))
        r -= v @ (v.T @ (Mf @ r))
        v = v.copy()
        v[:, 10] += 1e-6 * r / np.sqrt(r @ (Mf @ r))
        return lam, v

    _patched(monkeypatch, perturb)
    with pytest.raises(RuntimeError, match="residuals too large"):
        reference_modes(K, M, 20, points=m.points)


def test_off_diagonal_pivot(monkeypatch):
    m, (K, M), _, _ = case("clamped")
    free = dof_mask(m, {"xmin": "xyz"})
    splu = reference.splu

    def fake(*a, **k):
        lu = splu(*a, **k)
        return SimpleNamespace(
            solve=lu.solve, U=lu.U, perm_c=lu.perm_c, perm_r=np.roll(lu.perm_r, 1)
        )

    monkeypatch.setattr(reference, "splu", fake)
    with pytest.raises(RuntimeError, match="pivoted off the diagonal"):
        reference_modes(K, M, 10, free)


# --- clusters ---


def test_cluster_end():
    """n_modes ending inside an exact FE multiplicity: certified up to the last gap."""
    m, (K, M), _, _ = case("cube")
    _, w = dense("cube")
    rel_gap = np.diff(w) / w[1:]
    ends = [n for n in range(7, 40) if rel_gap[n - 1] < 1e-10][:4]
    assert len(ends) == 4
    for n in ends:
        lam, _ = reference_modes(K, M, n, points=m.points)
        j = n - 1
        while rel_gap[j - 1] < 1e-10:
            j -= 1
        assert np.all(np.abs(lam[6:j] - w[6:j]) <= 2 * 10 * EPS * w[-1])


def test_inside_rigid_cluster():
    m, (K, M), _, _ = case("box")
    with pytest.raises(ValueError, match="no gap"):
        reference_modes(K, M, 3, points=m.points)


@pytest.mark.parametrize("n", [7, 10, 12, 17, 21, 25])
def test_thin_plate_gaps(n, thin_rough):
    """L/t = 1600: eigsh splits degenerate pairs; the absolute gap term keeps mu out of them.
    kappa = 3e13 lies beyond the sigma rule's validated range: explicit sigma near lam_n."""
    m, (K, M), _, _ = case("thin")
    lam, _ = reference_modes(K, M, n, sigma=-0.01 * thin_rough[n], points=m.points)
    assert len(lam) == n


@pytest.fixture(scope="module")
def thin_rough():
    """Rough lowest 27 eigenvalues of the thin plate (one shift-invert run near lam_7)."""
    _, (K, M), _, _ = case("thin")
    s = -1e-14 * K.diagonal().mean() / M.diagonal().mean()
    return reference._solve(K.tocsc(), M.tocsc(), 27, s, np.arange(K.shape[0]), "MMD_AT_PLUS_A")[0]


# --- inputs and restrict ---


def test_inputs():
    m, (K, M), _, _ = case("box")
    n = K.shape[0]
    for kw in (
        dict(n_modes=0),
        dict(n_modes=n - 1),
        dict(n_modes=5, free=np.ones(n)),
        dict(n_modes=5, free=np.ones(n - 3, bool)),
        dict(n_modes=10, sigma=0.0),
        dict(n_modes=10, sigma=1.0),
        dict(n_modes=10, sigma=np.nan),
        dict(n_modes=10, points=m.points[:-1]),
        dict(n_modes=10, points=m.points[:, :2]),
    ):
        with pytest.raises(ValueError):
            reference_modes(K, M, **kw)
    with pytest.raises(ValueError):
        reference_modes(K, M[:-1, :-1], 10)


def test_restrict():
    m, (K, M), bcs, _ = case("clamped")
    free = dof_mask(m, bcs)
    Kf, Mf, idx = restrict(K, M, free)
    assert np.array_equal(idx, np.flatnonzero(free))
    assert np.array_equal(Kf.toarray(), K.toarray()[np.ix_(free, free)])
    assert np.array_equal(Mf.toarray(), M.toarray()[np.ix_(free, free)])
    Kf, _, idx = restrict(K, M)
    assert np.array_equal(idx, np.arange(K.shape[0]))
    assert np.array_equal(Kf.toarray(), K.toarray())
    with pytest.raises(ValueError):
        restrict(K, M[:-1, :-1])


# --- count_below ---


@pytest.mark.parametrize("name", ["clamped", "box", "box_lumped"])
def test_count_below(name):
    _, (K, M), _, n_rigid = case(name)
    free, w = dense(name)
    gaps = [q for q in range(max(n_rigid, 1) - 1, 40) if w[q + 1] - w[q] >= 1e-3 * abs(w[q + 1])]
    if n_rigid:
        assert gaps[0] == n_rigid - 1  # first gap above the rigid cluster
    assert len(gaps) >= 4
    two = 0
    Kf, Mf, _ = restrict(K, M, free)
    for q in gaps[:6]:
        sigma = (w[q] + w[q + 1]) / 2
        assert count_below(K, M, sigma, free) == q + 1
        two += (lapack.dsytrf((Kf - sigma * Mf).toarray(), lower=1)[1] < 0).sum()
    assert two  # 2x2 pivots occur, so their counting is exercised


def test_count_below_max(monkeypatch):
    _, (K, M), _, _ = case("box")
    monkeypatch.setattr(inertia, "MAX_DENSE", K.shape[0] - 1)
    with pytest.raises(ValueError, match="MAX_DENSE"):
        count_below(K, M, 1.0)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="finite"):
        count_below(K, M, np.inf)
