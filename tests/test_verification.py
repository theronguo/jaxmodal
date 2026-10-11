"""Sliding block: exact spectrum against hex8/hex20/hex27 meshes (accuracy, rates, bounds)."""

import functools
import itertools

import numpy as np
import pytest
from exact import SLIDING, sliding_block
from helpers import ORTHO
from scipy.sparse.linalg import eigsh

from jaxmodal.eigen import reference_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import assemble, dof_mask, restrict

EPS = np.finfo(float).eps
L = np.array([1.0, 0.83, 0.67])
MATS = {"iso": Material(isotropic(3.0, 0.3), 2.0), "ortho": Material(ORTHO, 0.7)}


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
    Kf, Mf, _ = restrict(K, M, free)
    return lam, eigsh(Kf, 1, M=Mf, which="LA", return_eigenvectors=False)[0]


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
