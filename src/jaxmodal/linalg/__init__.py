"""Linear algebra: PCG, MINRES and block-ELL SpMV on the device, the dense inertia count on the
host. See docs/preconditioners.md and docs/eigensolvers.md."""

from .ell import BlockELL, block_ell
from .inertia import count_below
from .krylov import minres, pcg

__all__ = ["BlockELL", "block_ell", "count_below", "minres", "pcg"]
