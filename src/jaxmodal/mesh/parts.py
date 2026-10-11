"""Face-connected parts of a mesh (NumPy/SciPy only). See docs/boundary_conditions.md."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .model import FACE_CORNERS


def parts(mesh):
    """(n_nodes,) int64 part label per node, 0..n_parts-1, for parts connected through whole
    faces. Raises if parts share nodes without a shared face (edge, vertex, nonconforming)."""
    conn = list(mesh.blocks.values())
    for c in conn:
        corners = np.sort(c[:, :8], axis=1)
        if (corners[:, 1:] == corners[:, :-1]).any():
            raise ValueError("mesh has collapsed elements (a repeated corner node)")
    faces = np.array(FACE_CORNERS)
    keys = np.concatenate([np.sort(c[:, faces], axis=-1).reshape(-1, 4) for c in conn])
    order = np.lexsort(keys.T[::-1])  # 6 faces per element, in global element order
    keys = keys[order]
    same = (keys[1:] == keys[:-1]).all(1)
    del keys
    a, b = order[:-1][same] // 6, order[1:][same] // 6
    n_el = mesh.n_el
    graph = sp.coo_array((np.ones(len(a), np.int8), (a, b)), shape=(n_el, n_el))
    label = connected_components(graph, directed=False)[1].astype(np.int64)
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
