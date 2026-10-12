"""Exact spectra against hex8/hex20/hex27 meshes (accuracy, rates, bounds, mode matching):
sliding block, Srinivas-Rao plates, Visscher Ritz box."""

import dataclasses
import functools
import itertools

import numpy as np
import pytest
from exact import SIMPLY_SUPPORTED, SLIDING, mac, plate_modes, sliding_block, visscher
from helpers import ORTHO, R
from scipy.sparse.linalg import eigsh

from jaxmodal.eigen import reference_modes
from jaxmodal.materials import Material, isotropic, orthotropic, rotate
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import assemble, dof_mask, restrict

EPS = np.finfo(float).eps
L = np.array([1.0, 0.83, 0.67])
MATS = {"iso": Material(isotropic(3.0, 0.3), 2.0), "ortho": Material(ORTHO, 0.7)}


def lmax(Kf, Mf):
    """Largest eigenvalue; only its order of magnitude matters (round-off allowances)."""
    return eigsh(Kf, 1, M=Mf, which="LA", tol=1e-3, return_eigenvectors=False)[0]


@functools.cache
def exact(mat, n):
    m = MATS[mat]
    return sliding_block(np.asarray(m.C), m.rho, L, n)


@functools.cache
def modes(t, mass, mat, n, k):
    """Lowest k eigenvalues and lam_max of the sliding block on an n^3 mesh."""
    m = box_mesh((n, n, n), spacing=tuple(L / n), type=t)
    K, M = assemble(m, {0: MATS[mat]}, mass=mass)
    free = dof_mask(m, SLIDING)
    lam, _ = reference_modes(K, M, k, free, points=m.points)
    return lam, lmax(*restrict(K, M, free)[:2])


def test_oracle_isotropic():
    E, nu, rho = 3.0, 0.3, 2.0
    lame, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    lam = []
    for pqr in itertools.product(range(9), repeat=3):
        k2 = np.sum((np.pi * np.array(pqr) / L) ** 2)
        zeros = pqr.count(0)
        p, s = (lame + 2 * mu) * k2 / rho, mu * k2 / rho
        lam += {0: [p, s, s], 1: [s, p], 2: [p], 3: []}[zeros]
    ref = np.sort(lam)[:30]
    assert np.allclose(sliding_block(np.asarray(isotropic(E, nu)), rho, L, 30), ref, rtol=1e-13)


def test_oracle_kmax():
    with pytest.raises(ValueError, match="kmax"):
        sliding_block(ORTHO, 0.7, L, 30, kmax=1)
    assert np.array_equal(
        sliding_block(ORTHO, 0.7, L, 30), sliding_block(ORTHO, 0.7, L, 30, kmax=16)
    )


@pytest.mark.parametrize("t", ["hex20", "hex27"])
@pytest.mark.parametrize("mat", ["iso", "ortho"])
def test_target(t, mat):
    """Roadmap: lowest 10 frequencies within 0.1% (n = 8), consistent mass above the exact."""
    lam, lmax = modes(t, "consistent", mat, 8, 10)
    ex = exact(mat, 10)
    assert np.abs(np.sqrt(lam / ex) - 1).max() <= 1e-3
    assert np.all(lam >= ex - 10 * EPS * lmax)


LEVELS = {
    ("hex8", "consistent"): (6, 8, 12, 16),
    ("hex8", "lumped"): (6, 8, 12, 16),
    ("hex20", "lumped"): (4, 6, 8, 12),  # HRZ
    ("hex27", "lumped"): (3, 4, 6, 8),
    ("hex20", "consistent"): (3, 4, 6, 8),
    ("hex27", "consistent"): (3, 4, 6, 8),
}
NESTED = {"hex8": ((6, 12), (8, 16)), "hex20": ((3, 6), (4, 8)), "hex27": ((3, 6), (4, 8))}


@pytest.mark.slow
@pytest.mark.parametrize("mat", ["iso", "ortho"])
@pytest.mark.parametrize("t,mass", list(LEVELS))
def test_rates(t, mass, mat):
    """Slope of log |lam_h / lam - 1| over log h per mode (lowest 6); consistent mass: upper
    bound at every level and monotone on nested meshes."""
    levels = LEVELS[t, mass]
    ex = exact(mat, 6)
    runs = {n: modes(t, mass, mat, n, 6) for n in levels}
    err = np.array([np.abs(runs[n][0] / ex - 1) for n in levels])
    slope = np.polyfit(np.log(1 / np.array(levels)), np.log(err), 1)[0]
    lam_f, lmax_f = runs[levels[-1]]
    assert slope.min() >= (3.6 if (t != "hex8" and mass == "consistent") else 1.8)
    assert np.all(err[-1] >= 100 * 10 * EPS * lmax_f / lam_f)  # above the round-off floor
    if mass == "consistent":
        for lam, lmax in runs.values():
            assert np.all(lam >= ex - 10 * EPS * lmax)
        for a, b in NESTED[t]:
            assert np.all(runs[b][0] <= runs[a][0] + 10 * EPS * runs[b][1])


PAG = np.asarray(orthotropic(25.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.2))
RZ = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z
H = 0.1  # plates [0, 1]^2 x [0, H]
PLATES = {
    "ortho": [(np.asarray(ORTHO), 0.7, H)],
    "pag": [(PAG, 1.0, H / 3), (np.asarray(rotate(PAG, RZ)), 1.0, H / 3), (PAG, 1.0, H / 3)],
}
TARGET_NZ = {"ortho": 4, "pag": 3}  # elements per ply on the 16 x 16 target mesh
ORDER = {  # labels (m, n, family) of the lowest 10, grouped into degenerate sets
    "ortho": [{(1, 1, 0)}, {(1, 2, 0)}, {(2, 1, 0)}, {(1, 0, 0), (0, 1, 0)}, {(2, 2, 0)},
              {(1, 3, 0)}, {(2, 3, 0)}, {(3, 1, 0)}, {(1, 1, 1)}],
    "pag": [{(1, 1, 0)}, {(1, 2, 0)}, {(0, 1, 0), (1, 0, 0)}, {(2, 1, 0)}, {(1, 3, 0)},
            {(2, 2, 0)}, {(2, 3, 0)}, {(2, 0, 0), (0, 2, 0)}],
}  # fmt: skip


@functools.cache
def plate_exact(name):
    return plate_modes(PLATES[name], 1.0, 1.0, 10)[0]


@functools.cache
def plate_fe(name, t, n, nz):
    """Lowest 10 modes of a simply supported plate on an n x n x (nz per ply) mesh:
    lam, V, lam_max, mesh, M, free."""
    plies = PLATES[name]
    m = box_mesh((n, n, nz * len(plies)), spacing=(1 / n, 1 / n, H / (nz * len(plies))), type=t)
    zc = m.points[m.blocks[t]].mean(1)[:, 2]  # ply index from the element centroid
    m = dataclasses.replace(m, material=np.searchsorted(np.cumsum([p[2] for p in plies])[:-1], zc))
    K, M = assemble(m, {i: Material(C, rho) for i, (C, rho, _) in enumerate(plies)})
    free = dof_mask(m, SIMPLY_SUPPORTED)
    lam, V = reference_modes(K, M, 10, free, points=m.points)
    return lam, V, lmax(*restrict(K, M, free)[:2]), m, M, free


@functools.cache
def visscher_exact():
    """Visscher Ritz at N = 18 and 20 (lowest 16, 6 rigid)."""
    C = np.asarray(rotate(ORTHO, R))
    return visscher(C, 0.7, L, 18), visscher(C, 0.7, L, 20)


@functools.cache
def box_fe(t, n):
    """Lowest 16 modes and lam_max of the free anisotropic box on an n^3 mesh."""
    m = box_mesh((n, n, n), spacing=tuple(L / n), type=t)
    K, M = assemble(m, {0: Material(ORTHO, 0.7)}, {0: R})
    return reference_modes(K, M, 16, points=m.points)[0], lmax(K, M)


def test_plate_oracle():
    for plies in PLATES.values():
        lam, labels, _ = plate_modes(plies, 1.0, 1.0, 10)
        assert np.allclose(plate_modes(plies, 1.0, 1.0, 10, P=24)[0], lam, rtol=1e-9, atol=0)
        assert np.array_equal(plate_modes(plies, 1.0, 1.0, 10, mmax=10)[0], lam)
        C, rho, _ = plies[0]  # (0, 1, 0): U constant, tau_xz = 0
        lam010 = lam[(labels == (0, 1, 0)).all(1)][0]
        assert np.isclose(lam010, C[5, 5] * np.pi**2 / rho, rtol=1e-12)
    with pytest.raises(ValueError, match="mmax"):
        plate_modes(PLATES["ortho"], 1.0, 1.0, 10, mmax=1)


def test_visscher_oracle():
    """Rigid modes; the roadmap's convergence rule N = 18 -> 20 on modes 7-16."""
    a, b = visscher_exact()
    assert np.abs(b[:6]).max() <= 1e-8 * b[6]
    assert np.abs(a[6:] / b[6:] - 1).max() < 1e-6


@pytest.mark.parametrize(
    "name,t",
    [("ortho", "hex20"), ("ortho", "hex27"), ("pag", "hex20"),
     pytest.param("pag", "hex27", marks=pytest.mark.slow)],
)  # fmt: skip
def test_plate_target(name, t):
    """Roadmap: lowest 10 frequencies within 0.1%, consistent mass above the exact."""
    lam, _, lmax_, *_ = plate_fe(name, t, 16, TARGET_NZ[name])
    ex = plate_exact(name)
    assert np.abs(np.sqrt(lam / ex) - 1).max() <= 1e-3
    assert np.all(lam >= ex - 10 * EPS * lmax_)


@pytest.mark.parametrize("name", ["ortho", "pag"])
def test_plate_mac(name):
    """Sets by relative gap, labels, and M-weighted MAC >= 0.9999 summed over each set."""
    _, V, _, m, M, free = plate_fe(name, "hex20", 16, TARGET_NZ[name])
    ex, labels, Phi = plate_modes(PLATES[name], 1.0, 1.0, 11, points=m.points)
    gap = np.diff(ex) / ex[:-1]
    assert gap[9] >= 1e-2  # the cutoff does not split a set
    sets = np.split(np.arange(10), np.flatnonzero(gap[:9] > 1e-2) + 1)
    assert [set(map(tuple, labels[S])) for S in sets] == ORDER[name]
    Phi = Phi[:, :10]
    # M-orthogonal on these meshes and symmetric layups; kills an unweighted MAC
    assert np.abs(mac(Phi, Phi, M) - np.eye(10)).max() <= 1e-12
    assert np.abs(Phi[~free]).max() <= 1e-12 * np.abs(Phi).max()
    V, c = V.copy(), np.sqrt(0.5)
    for S in sets:
        if len(S) == 2:  # any basis of the eigenspace is valid: force mixing
            V[:, S] = V[:, S] @ np.array([[c, -c], [c, c]])
    A = mac(V, Phi, M)
    for S in sets:
        assert np.all(A[np.ix_(S, S)].sum(1) >= 0.9999)


@pytest.mark.parametrize("t", ["hex20", "hex27"])
def test_visscher_target(t):
    """Roadmap: modes 7-16 within 0.1% (n = 8); above the Ritz value up to its error."""
    ex = visscher_exact()[1][6:]
    assert (np.diff(ex) / ex[:-1]).min() >= 1e-2  # generic: sorted pairing is valid
    lam = box_fe(t, 8)[0][6:]
    assert np.abs(np.sqrt(lam / ex) - 1).max() <= 1e-3
    assert np.all(lam >= ex * (1 - 1e-6))


def test_visscher_nested():
    (a, _), (b, lmax_) = box_fe("hex20", 4), box_fe("hex20", 8)
    assert np.all(b[6:] <= a[6:] + 10 * EPS * lmax_)


PLATE_LEVELS = {
    ("ortho", "hex8"): (8, 12, 16, 24),
    ("ortho", "hex20"): (4, 8, 12, 16),
    ("ortho", "hex27"): (4, 8, 12, 16),
    ("pag", "hex20"): (4, 8, 12),
    ("pag", "hex27"): (4, 8, 12),
}
PLATE_NESTED = {"hex8": ((8, 16), (12, 24)), "hex20": ((4, 8), (8, 16)), "hex27": ((4, 8), (8, 16))}


@pytest.mark.slow
@pytest.mark.parametrize("name,t", list(PLATE_LEVELS))
def test_plate_rates(name, t):
    """Rates of the lowest 6 on levels n x n x (n/4 per ply); upper bound at every level and
    monotone on nested meshes. hex8 on the cross-ply is pre-asymptotic (not tested)."""
    levels = PLATE_LEVELS[name, t]
    ex = plate_exact(name)[:6]
    runs = {n: plate_fe(name, t, n, n // 4)[:3] for n in levels}
    err = np.array([np.abs(runs[n][0][:6] / ex - 1) for n in levels])
    slope = np.polyfit(np.log(1 / np.array(levels)), np.log(err), 1)[0]
    lam_f, _, lmax_f = runs[levels[-1]]
    assert slope.min() >= (1.8 if t == "hex8" else 3.6)
    assert np.all(err[-1] >= 100 * 10 * EPS * lmax_f / lam_f[:6])
    for lam, _, lmax_ in runs.values():
        assert np.all(lam[:6] >= ex - 10 * EPS * lmax_)
    for a, b in PLATE_NESTED[t]:
        if b in runs:  # PAG levels stop at n = 12
            assert np.all(runs[b][0] <= runs[a][0] + 10 * EPS * runs[b][2])
