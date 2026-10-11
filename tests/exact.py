"""Exact reference spectra (in-test oracles). See docs/verification.md."""

import itertools

import numpy as np
from helpers import tensor

SLIDING = {"xmin": "x", "xmax": "x", "ymin": "y", "ymax": "y", "zmin": "z", "zmax": "z"}


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
