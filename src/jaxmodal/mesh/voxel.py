"""Structured voxel mesh generator. See docs/mesh.md."""

import numpy as np

from .model import Mesh, Voxel, check_grid, face_sets


def voxel_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), mask=None):
    """hex8 mesh of a regular grid; `mask[i, j, k]` (bool, shape `shape`) selects active cells.

    Cell (i, j, k) has flat index i + nx (j + ny k); grid node (i, j, k) has
    i + (nx+1)(j + (ny+1) k). Elements and used nodes are kept in grid order.
    """
    check_grid(shape, spacing, origin)
    shape = tuple(int(s) for s in shape)
    spacing, origin = tuple(float(h) for h in spacing), tuple(float(o) for o in origin)
    nx, ny, nz = shape
    mask = np.ones(shape, bool) if mask is None else np.asarray(mask)
    if mask.shape != shape or mask.dtype != bool:
        raise ValueError(f"mask must be a bool array of shape {shape}")
    active = np.flatnonzero(mask.transpose(2, 1, 0))  # flat cell index, x fastest
    n_el = len(active)
    if n_el == 0:
        raise ValueError("mask has no active cells")

    sy, sz = nx + 1, (nx + 1) * (ny + 1)
    base = active % nx + sy * (active // nx % ny) + sz * (active // (nx * ny))
    offsets = (0, 1, 1 + sy, sy, sz, sz + 1, sz + 1 + sy, sz + sy)  # VTK corner order
    used = np.zeros(sz * (nz + 1), bool)
    for o in offsets:  # per corner: bounds temporaries at 100M-DOF scale
        used[base + o] = True
    node_map = np.full(len(used), -1, np.int64)
    node_map[used] = np.arange(used.sum())
    conn = np.empty((n_el, 8), np.int64)
    for a, o in enumerate(offsets):
        conn[:, a] = node_map[base + o]
    element_map = np.full(nx * ny * nz, -1, np.int64)
    element_map[active] = np.arange(n_el)

    g = np.flatnonzero(used)
    ijk = np.stack([g % sy, g // sy % (ny + 1), g // sz], 1)
    points = np.asarray(origin) + np.asarray(spacing) * ijk
    return Mesh(
        points=points,
        blocks={"hex8": conn},
        node_sets=face_sets(ijk, shape),
        element_sets={},
        material=np.zeros(n_el, np.int64),
        orientation=np.zeros(n_el, np.int64),
        voxel=Voxel(shape, spacing, origin, node_map, element_map),
    )
