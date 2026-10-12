"""Preconditioners and smoothers for A_P: Jacobi, node-block Jacobi, Chebyshev, voxel GMG,
unstructured p-MG. See docs/preconditioners.md."""

from .chebyshev import Chebyshev, chebyshev, lmax_estimate
from .gmg import GMG, SingularShiftError, gmg
from .jacobi import block_jacobi, jacobi
from .pmg import pmg

__all__ = [
    "GMG",
    "Chebyshev",
    "SingularShiftError",
    "block_jacobi",
    "chebyshev",
    "gmg",
    "jacobi",
    "lmax_estimate",
    "pmg",
]
