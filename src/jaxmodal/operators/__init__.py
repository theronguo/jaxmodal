"""DOF map, assembled reference operators, matrix-free operators, Dirichlet masking, rigid
modes and deflation. See docs/operators.md and docs/boundary_conditions.md."""

from .assembled import assemble
from .constraints import (
    a_p,
    dof_mask,
    m_orthonormalize,
    masked,
    project,
    restrict,
    rigid_modes,
)
from .dofmap import check_int32, element_dofs
from .matfree import Unstructured, unstructured
from .voxel import Stencil, reference_element, stencil

__all__ = [
    "Stencil",
    "Unstructured",
    "a_p",
    "assemble",
    "check_int32",
    "dof_mask",
    "element_dofs",
    "m_orthonormalize",
    "masked",
    "project",
    "reference_element",
    "restrict",
    "rigid_modes",
    "stencil",
    "unstructured",
]
