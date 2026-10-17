"""Structured hex8/hex20/hex27 box mesh generator (solid shells: mesh.solid_shell).
See docs/mesh.md."""

import numpy as np

from .model import NODES, REFERENCE_NODES, Mesh, check_grid, face_sets

BOX_TYPES = ("hex8", "hex20", "hex27")


def box_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), type="hex8", coords=None):
    """Block of nx x ny x nz elements of `type` with face node sets "xmin" ... "zmax".

    Nodes are the used points of the half-spacing grid (2nx+1) x (2ny+1) x (2nz+1); nodes and
    elements are in grid order (x fastest), like voxel_mesh. coords: None or three increasing
    arrays of element boundaries per axis (lengths nx+1, ny+1, nz+1) for graded meshes; it
    overrides spacing and origin. Midside and centre nodes sit at element midpoints.
    """
    check_grid(shape, spacing, origin)
    if type not in BOX_TYPES:
        raise ValueError(f"type must be one of {BOX_TYPES}, got {type!r}")
    shape = tuple(int(s) for s in shape)
    if coords is not None:
        coords = [np.asarray(c, float) for c in coords]
        if len(coords) != 3 or any(
            c.shape != (n + 1,) or not np.isfinite(c).all() or (np.diff(c) <= 0).any()
            for c, n in zip(coords, shape, strict=True)
        ):
            raise ValueError(
                f"coords must be 3 finite increasing arrays of lengths {np.add(shape, 1)}"
            )
    nx, ny, nz = shape
    gx, gy = 2 * nx + 1, 2 * ny + 1
    e = np.arange(nx * ny * nz)
    cell = np.stack([e % nx, e // nx % ny, e // (nx * ny)], 1)
    ijk = 2 * cell[:, None, :] + (REFERENCE_NODES[: NODES[type]] + 1).astype(np.int64)
    used, conn = np.unique(ijk[..., 0] + gx * (ijk[..., 1] + gy * ijk[..., 2]), return_inverse=True)
    ijk = np.stack([used % gx, used // gx % gy, used // (gx * gy)], 1)
    if coords is None:
        points = np.asarray(origin, float) + np.asarray(spacing, float) * (ijk / 2)
    else:
        points = np.stack(
            [np.interp(ijk[:, d] / 2, np.arange(shape[d] + 1), coords[d]) for d in range(3)], 1
        )
    sets = {}
    for d in range(3):
        sets |= face_sets(d, ijk[:, d], 2 * shape[d])
    return Mesh(
        points=points,
        blocks={type: conn.reshape(len(e), -1)},
        node_sets=sets,
        element_sets={},
        material=np.zeros(len(e), np.int64),
        orientation=np.zeros(len(e), np.int64),
    )
