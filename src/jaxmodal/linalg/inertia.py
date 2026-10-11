"""Dense inertia count: Sylvester's law with LAPACK Bunch-Kaufman (sytrf). See
docs/eigensolvers.md."""

import numpy as np
from scipy.linalg import lapack

from ..operators.constraints import restrict

MAX_DENSE = 20_000  # free DOFs: one dense (n, n) fp64 copy, 3.2 GB and ~16 s at 20k


def count_below(K, M, sigma, free=None):
    """Number of eigenvalues of (K_ff, M_ff) below sigma, exact for sigma away from eigenvalues.

    Dense and pivoted (backward stable). reference_modes has a cheaper sparse, unpivoted count
    for its own check; see docs/eigensolvers.md for why both exist.
    """
    Kf, Mf, _ = restrict(K, M, free)
    n = Kf.shape[0]
    if n > MAX_DENSE:
        raise ValueError(f"{n} free DOFs > MAX_DENSE = {MAX_DENSE} (dense count)")
    if not np.isfinite(sigma):
        raise ValueError(f"sigma must be finite, got {sigma}")
    A = (Kf - sigma * Mf).toarray(order="F")  # Fortran order: sytrf works in place
    lwork = int(lapack.dsytrf_lwork(n, lower=1)[0])
    lu, ipiv, info = lapack.dsytrf(A, lower=1, lwork=lwork, overwrite_a=1)
    if info > 0:
        raise ValueError(f"sigma = {sigma:.6g} is an eigenvalue (zero pivot)")
    two = ipiv < 0  # rows of 2x2 pivots; a Bunch-Kaufman 2x2 block has det < 0: one negative
    return int((np.diag(lu)[~two] < 0).sum() + two.sum() // 2)
