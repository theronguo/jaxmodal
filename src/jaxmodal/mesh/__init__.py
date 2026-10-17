"""Mesh data model, IO, voxel and box generators, solid-shell conversion. See docs/mesh.md."""

from .box import box_mesh
from .io import read, write
from .model import TYPES, Mesh, Voxel, corner_jacobians
from .parts import parts
from .reorder import rcm, renumber
from .shell import solid_shell
from .voxel import voxel_mesh

__all__ = [
    "TYPES",
    "Mesh",
    "Voxel",
    "box_mesh",
    "corner_jacobians",
    "parts",
    "rcm",
    "read",
    "renumber",
    "solid_shell",
    "voxel_mesh",
    "write",
]
