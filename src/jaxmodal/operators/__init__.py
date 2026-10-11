"""DOF map, assembled reference operators and Dirichlet masking. See docs/operators.md."""

from .assembled import assemble
from .constraints import a_p, dof_mask, masked
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
    "masked",
    "reference_element",
    "stencil",
    "unstructured",
]
