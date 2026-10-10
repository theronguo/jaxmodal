"""Structured hex8/hex20/hex27 box mesh generator. See docs/mesh.md."""

import numpy as np

from .model import NODES, REFERENCE_NODES, Mesh, check_grid, face_sets


def box_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), type="hex8"):
    """Block of nx x ny x nz elements of `type` with face node sets "xmin" ... "zmax".

    Nodes are the used points of the half-spacing grid (2nx+1) x (2ny+1) x (2nz+1); nodes and
    elements are in grid order (x fastest), like voxel_mesh.
    """
    check_grid(shape, spacing, origin)
    if type not in NODES:
        raise ValueError(f"type must be one of {tuple(NODES)}, got {type!r}")
    nx, ny, nz = (int(s) for s in shape)
    gx, gy = 2 * nx + 1, 2 * ny + 1
    e = np.arange(nx * ny * nz)
    cell = np.stack([e % nx, e // nx % ny, e // (nx * ny)], 1)
    ijk = 2 * cell[:, None, :] + (REFERENCE_NODES[: NODES[type]] + 1).astype(np.int64)
    used, conn = np.unique(ijk[..., 0] + gx * (ijk[..., 1] + gy * ijk[..., 2]), return_inverse=True)
    ijk = np.stack([used % gx, used // gx % gy, used // (gx * gy)], 1)
    return Mesh(
        points=np.asarray(origin, float) + np.asarray(spacing, float) * (ijk / 2),
        blocks={type: conn.reshape(len(e), -1)},
        node_sets=face_sets(ijk, (2 * nx, 2 * ny, 2 * nz)),
        element_sets={},
        material=np.zeros(len(e), np.int64),
        orientation=np.zeros(len(e), np.int64),
    )
