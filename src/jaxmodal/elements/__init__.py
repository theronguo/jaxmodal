"""Element registry, hex8/hex20/hex27 and solid-shell (ssh8) kernels, the host element check and
the recovery registry. See docs/elements.md."""

from functools import partial

from . import kernels
from .check import check_elements
from .kernels import (
    bmatrix,
    gradients,
    jacobian,
    lumped_mass,
    mass,
    mass_form,
    stiffness,
    stiffness_form,
)
from .registry import ELEMENTS, Element, gauss
from .shell import shell_mass, shell_recover, shell_stiffness

# Recovery per type: RECOVER[t](X, table_e, u) -> points (nq, 3), weights w det J (nq,), and global
# Cartesian Voigt strain and stress (nq, 6); table_e is the type's kernel input for one element:
# the rotated C (6, 6) of its phase (hex), the (plies, R_o) slice of materials.ply_table (ssh8).
RECOVER = {t: partial(kernels.hex_recover, t) for t in ("hex8", "hex20", "hex27")}
RECOVER["ssh8"] = shell_recover

__all__ = [
    "ELEMENTS",
    "RECOVER",
    "Element",
    "bmatrix",
    "check_elements",
    "gauss",
    "gradients",
    "jacobian",
    "lumped_mass",
    "mass",
    "mass_form",
    "shell_mass",
    "shell_recover",
    "shell_stiffness",
    "stiffness",
    "stiffness_form",
]
