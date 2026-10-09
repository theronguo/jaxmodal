"""Structured voxel mesh generator. See docs/mesh.md."""

import numbers

import numpy as np

from .model import Mesh, Voxel


def voxel_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), mask=None):
    """hex8 mesh of a regular grid; `mask[i, j, k]` (bool, shape `shape`) selects active cells.

    Cell (i, j, k) has flat index i + nx (j + ny k); grid node (i, j, k) has
    i + (nx+1)(j + (ny+1) k).
    Elements and used nodes are kept in grid order.
    """
    if len(shape) != 3 or not all(
        isinstance(s, numbers.Integral) and not isinstance(s, bool) and s > 0 for s in shape
    ):
        raise ValueError(f"shape must be 3 positive ints, got {shape}")
    if len(spacing) != 3 or len(origin) != 3:
        raise ValueError("spacing and origin must have length 3")
    shape = tuple(int(s) for s in shape)
    spacing, origin = tuple(float(h) for h in spacing), tuple(float(o) for o in origin)
    if not all(np.isfinite(spacing + origin)) or min(spacing) <= 0:
        raise ValueError("spacing must be positive and spacing/origin finite")
    nx, ny, nz = shape
    if mask is None:
        mask = np.ones(shape, bool)
    mask = np.asarray(mask)
    if mask.shape != shape or mask.dtype != bool:
        raise ValueError(f"mask must be a bool array of shape {shape}")
    active = np.flatnonzero(mask.transpose(2, 1, 0))  # flat cell index, x fastest
    if len(active) == 0:
        raise ValueError("mask has no active cells")

    sy, sz = nx + 1, (nx + 1) * (ny + 1)
    i, j, k = active % nx, active // nx % ny, active // (nx * ny)
    base = i + sy * j + sz * k
    offsets = np.array([0, 1, 1 + sy, sy, sz, sz + 1, sz + 1 + sy, sz + sy])
    grid_conn = base[:, None] + offsets

    n_grid = sz * (nz + 1)
    used = np.zeros(n_grid, bool)
    used[grid_conn] = True
    node_map = np.full(n_grid, -1, np.int64)
    node_map[used] = np.arange(used.sum())
    element_map = np.full(nx * ny * nz, -1, np.int64)
    element_map[active] = np.arange(len(active))

    g = np.flatnonzero(used)
    ijk = np.stack([g % sy, g // sy % (ny + 1), g // sz], axis=1)
    points = np.asarray(origin) + np.asarray(spacing) * ijk
    n_el = len(active)
    return Mesh(
        points=points,
        blocks={"hex8": node_map[grid_conn]},
        node_sets={},
        element_sets={},
        material=np.zeros(n_el, np.int64),
        orientation=np.zeros(n_el, np.int64),
        voxel=Voxel(shape, spacing, origin, node_map, element_map),
    )
