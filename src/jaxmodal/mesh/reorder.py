"""Node and element renumbering (NumPy/SciPy): renumber, rcm. See docs/mesh.md."""

import dataclasses

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .model import Mesh


def _inverse(p):
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    return inv


def _is_permutation(p, n):
    return isinstance(p, np.ndarray) and p.shape == (n,) and (np.sort(p) == np.arange(n)).all()


def renumber(mesh, p, q):
    """Mesh with nodes permuted by p and elements by q (both new -> old, int arrays).

    q must map each block's global index range onto itself (blocks keep TYPES order).
    """
    n, n_el = len(mesh.points), mesh.n_el
    if not _is_permutation(p, n) or not _is_permutation(q, n_el):
        raise ValueError(f"p and q must be permutations of {n} nodes and {n_el} elements")
    p, q = p.astype(np.int64), q.astype(np.int64)
    pi, qi = _inverse(p), _inverse(q)
    blocks = {}
    for t, s in mesh.offsets.items():
        e = q[s : s + len(mesh.blocks[t])]
        if len(e) and (e.min() < s or e.max() >= s + len(e)):
            raise ValueError(f"q moves elements out of block {t!r}")
        blocks[t] = pi[mesh.blocks[t][e - s]]

    def sets(d, inv):
        return {k: np.sort(inv[v]) for k, v in d.items()}

    voxel = mesh.voxel
    if voxel is not None:
        nm, em = voxel.node_map, voxel.element_map
        voxel = dataclasses.replace(
            voxel,
            node_map=np.where(nm >= 0, pi[np.maximum(nm, 0)], -1),
            element_map=np.where(em >= 0, qi[np.maximum(em, 0)], -1),
        )
    return Mesh(
        points=mesh.points[p],
        blocks=blocks,
        node_sets=sets(mesh.node_sets, pi),
        element_sets=sets(mesh.element_sets, qi),
        material=mesh.material[q],
        orientation=mesh.orientation[q],
        voxel=voxel,
    )


def rcm(mesh):
    """Reverse Cuthill-McKee node order, then elements sorted by min node within each block.

    Returns (new mesh, p, q) with p, q new -> old. Helps arbitrarily ordered meshes (Gmsh,
    Abaqus); the grid order of box_mesh and voxel_mesh is already narrower.
    """
    conn = [c for c in mesh.blocks.values()]
    rows = np.concatenate(
        [
            np.repeat(np.arange(len(c)) + s, c.shape[1])
            for c, s in zip(conn, mesh.offsets.values(), strict=True)
        ]
    )
    cols = np.concatenate([c.ravel() for c in conn])
    E = sp.csr_array(
        (np.ones(len(rows), np.int32), (rows, cols)), shape=(mesh.n_el, len(mesh.points))
    )
    A = (E.T @ E).tocsr()
    p = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    pi = _inverse(p)
    q = np.concatenate(
        [
            s + np.argsort(pi[c].min(1), kind="stable")
            for c, s in zip(conn, mesh.offsets.values(), strict=True)
        ]
    )
    return renumber(mesh, p, q), p, q
