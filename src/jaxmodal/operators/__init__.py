"""DOF map, assembled reference operators, matrix-free operators, Dirichlet masking, rigid
modes and deflation. See docs/operators.md and docs/boundary_conditions.md."""

from .assembled import assemble
from .constraints import (
    APOperator,
    a_p,
    ap_operator,
    dof_mask,
    m_orthonormalize,
    masked,
    project,
    restrict,
    rigid_modes,
)
from .dofmap import check_int32, element_dofs
from .matfree import Unstructured, unstructured
from .sampling import mass_sample, mass_sample_keys
from .voxel import Stencil, reference_element, stencil

__all__ = [
    "APOperator",
    "Stencil",
    "Unstructured",
    "a_p",
    "ap_operator",
    "assemble",
    "check_int32",
    "dof_mask",
    "element_dofs",
    "m_orthonormalize",
    "mass_sample",
    "mass_sample_keys",
    "masked",
    "project",
    "reference_element",
    "restrict",
    "rigid_modes",
    "stencil",
    "unstructured",
]
