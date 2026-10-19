"""Face-connected parts of a mesh (NumPy/SciPy only). See docs/boundary_conditions.md."""

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import connected_components

from .model import FACE_CORNERS


def parts(mesh):
    """(n_nodes,) int64 part label per node, 0..n_parts-1, for parts connected through whole
    faces. Raises if parts share nodes without a shared face (edge, vertex, nonconforming).
    Voxel meshes label the cell mask directly (faster and leaner); the labels are the same."""
    conn = list(mesh.blocks.values())
    label = _voxel_labels(mesh) if mesh.voxel is not None else _face_labels(mesh, conn)
    part = np.empty(len(mesh.points), np.int64)
    bad = 0
    for c, s in zip(conn, mesh.offsets.values(), strict=True):
        part[c] = label[s : s + len(c), None]
    for c, s in zip(conn, mesh.offsets.values(), strict=True):
        bad += (part[c] != label[s : s + len(c), None]).any(1).sum()
    if bad:
        raise ValueError(
            "mesh parts touch without sharing a whole face (edge, vertex or nonconforming "
            f"contact): {bad} elements"
        )
    return part


def _voxel_labels(mesh):
    """Element labels of a voxel mesh: ndimage.label on the cell mask (face connectivity),
    renumbered by each part's lowest element index (connected_components' order)."""
    nx, ny, nz = mesh.voxel.shape
    em = mesh.voxel.element_map.reshape(nz, ny, nx)
    lab, k = ndimage.label(em >= 0)
    sel = em >= 0
    label = np.empty(mesh.n_el, np.int64)
    label[em[sel]] = lab[sel] - 1
    first = np.full(k, mesh.n_el)
    np.minimum.at(first, label, np.arange(mesh.n_el))
    rank = np.empty(k, np.int64)
    rank[np.argsort(first)] = np.arange(k)
    return rank[label]


def _face_labels(mesh, conn):
    """Element labels from shared faces (sorted face keys, connected components)."""
    for c in conn:
        corners = np.sort(c[:, :8], axis=1)
        if (corners[:, 1:] == corners[:, :-1]).any():
            raise ValueError("mesh has collapsed elements (a repeated corner node)")
    faces = np.array(FACE_CORNERS)
    dtype = np.int32 if len(mesh.points) < 2**31 else np.int64  # exact keys, half the memory
    keys = np.concatenate([np.sort(c[:, faces].astype(dtype), -1).reshape(-1, 4) for c in conn])
    order = np.lexsort(keys.T[::-1])  # 6 faces per element, in global element order
    keys = keys[order]
    same = (keys[1:] == keys[:-1]).all(1)
    del keys
    a, b = order[:-1][same] // 6, order[1:][same] // 6
    n_el = mesh.n_el
    graph = sp.coo_array((np.ones(len(a), np.int8), (a, b)), shape=(n_el, n_el))
    return connected_components(graph, directed=False)[1].astype(np.int64)
