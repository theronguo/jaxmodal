"""Shared test references."""

import numpy as np

PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]  # Voigt index -> tensor indices


def rel(a, b):
    """Principle 1: ||a - b||_inf / ||b||_inf."""
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max()


def tensor(C):
    """Voigt (6, 6) stiffness -> 4th-order tensor C_ijkl."""
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(PAIRS):
        for b, (k, n) in enumerate(PAIRS):
            for p, q in ((i, j), (j, i)):
                for r, s in ((k, n), (n, k)):
                    T[p, q, r, s] = C[a, b]
    return T
