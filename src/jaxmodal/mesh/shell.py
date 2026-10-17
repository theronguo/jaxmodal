"""hex8 -> solid-shell (ssh8) conversion: thickness axis per element, node permutation (NumPy).
See docs/mesh.md. Imports only mesh.model, not elements: elements.registry imports mesh.model,
so importing elements here would cycle."""

import numpy as np

from .model import CORNERS, FACE_CORNERS, PAIRS, TYPES, Mesh

# Proper rotations Q (xi_new = Q xi_old) taking reference axis d to +zeta; FLIP turns 180 degrees
# about xi (zeta and eta flip). New node i (at CORNERS[i]) is the old node at Q^T CORNERS[i].
_TO_ZETA = (
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    np.eye(3, dtype=int),
)
_FLIP = np.diag([1, -1, -1])


def _perm(Q):
    return [int(np.flatnonzero((CORNERS == c).all(1))[0]) for c in CORNERS @ Q]


_PERMS = np.array([[_perm(Q), _perm(_FLIP @ Q)] for Q in _TO_ZETA])  # (axis, sign +/-, 8)


def _neighbours(conn):
    """(n, 6) flat index 6 e + f of the element face sharing face f of element e, or -1."""
    keys = np.sort(conn[:, np.array(FACE_CORNERS)], -1).reshape(-1, 4)
    order = np.lexsort(keys.T[::-1])
    same = (keys[order[1:]] == keys[order[:-1]]).all(1)
    a, b = order[:-1][same], order[1:][same]
    nb = np.full(len(keys), -1)
    nb[a], nb[b] = b, a
    return nb.reshape(-1, 6)


def _sweep(conn):
    """Thickness axis per element: the unique shortest of its 3 face-to-face chains."""
    nb = _neighbours(conn)
    n = len(conn)
    count = np.zeros((n, 3), np.int64)
    for e, d in np.ndindex(n, 3):
        if count[e, d]:
            continue
        chain, seen = [(e, d)], {(e, d)}
        for face in (2 * d + 1, 2 * d):  # forward, then backward (a ring stops at once)
            cur, f = e, face
            while (k := nb[cur, f]) >= 0:
                key = (k // 6, k % 6 // 2)  # entered through face k % 6; its axis
                if key in seen:  # visited mark per (element, axis): a closed ring
                    break
                chain.append(key)
                seen.add(key)
                cur, f = k // 6, (k % 6) ^ 1  # leave through the opposite face
        for c in chain:
            count[c] = len(chain)
    s = np.sort(count, 1)
    if (s[:, 0] == s[:, 1]).any():
        raise ValueError("ambiguous thickness direction: pass direction=")
    return count.argmin(1)


def solid_shell(mesh, elements=None, direction=None):
    """Convert hex8 elements to solid shells (ssh8): returns (mesh, order).

    elements: an element set name, global element indices, or None for all hex8. direction:
    None (topological sweep, the axis with the unique shortest chain), a (3,) vector or (n_sel, 3)
    vectors in sorted selected order (the reference axis best aligned, zeta along +direction).
    The selected elements are appended to the ssh8 block with connectivity permuted (a proper
    rotation) so reference zeta is the thickness; an emptied hex8 block is dropped. order is new
    -> old over all elements (like renumber's q, but it moves elements across blocks).
    """
    if "hex8" not in mesh.blocks:
        raise ValueError("solid_shell: the mesh has no hex8 elements")
    off, hexes = mesh.offsets, mesh.blocks["hex8"]
    s0 = off["hex8"]
    if elements is None:
        sel = np.arange(s0, s0 + len(hexes))
    elif isinstance(elements, str):
        if elements not in mesh.element_sets:
            raise ValueError(f"no element set {elements!r}")
        sel = mesh.element_sets[elements]
    else:
        given = np.asarray(elements)
        if given.ndim != 1 or (len(given) and given.dtype.kind not in "iu"):
            raise ValueError("elements must be a set name, a 1-D integer array or None")
        sel = np.unique(given).astype(np.int64)
        if np.ndim(direction) == 2 and not np.array_equal(given, sel):
            raise ValueError("a per-element direction needs sorted, unique elements")
    if len(sel) == 0:
        raise ValueError("solid_shell: empty element selection")
    if sel[0] < s0 or sel[-1] >= s0 + len(hexes):
        raise ValueError(
            f"solid_shell converts hex8 elements only (indices {s0}..{s0 + len(hexes) - 1})"
        )
    conn = hexes[sel - s0]
    n, rows = len(sel), np.arange(len(sel))
    X = mesh.points[conn]
    edge = np.stack([sum(X[:, hi] - X[:, lo] for lo, hi in PAIRS[d]) / 4 for d in range(3)], 1)
    if direction is None:
        axis = _sweep(conn)
        e = edge[rows, axis]
        a = np.abs(e)
        top = a >= a.max(1, keepdims=True) - 1e-8 * np.linalg.norm(e, axis=1, keepdims=True)
        dot = e[rows, top.argmax(1)]  # tolerant argmax: the first near-largest component
    else:
        v = np.asarray(direction, float)
        if v.shape not in ((3,), (n, 3)) or not np.isfinite(v).all():
            raise ValueError(f"direction must be a finite (3,) or ({n}, 3) array")
        v = np.broadcast_to(v, (n, 3))
        if (np.linalg.norm(v, axis=1) == 0).any():
            raise ValueError("direction must be nonzero")
        proj = np.einsum("nai,ni->na", edge, v)
        cos = np.abs(proj) / (np.linalg.norm(edge, axis=2) * np.linalg.norm(v, axis=1)[:, None])
        c = np.sort(cos, 1)
        if (bad := c[:, 2] - c[:, 1] < 0.1).any():
            k = int(np.argmax(bad))
            raise ValueError(
                f"element {sel[k]}: thickness axis ambiguous for direction (|cos| {cos[k]})"
            )
        axis = cos.argmax(1)
        dot = proj[rows, axis]
    new = conn[rows[:, None], _PERMS[axis, (dot < 0).astype(int)]]

    idx = {t: np.arange(off[t], off[t] + len(c)) for t, c in mesh.blocks.items()}
    keep = np.setdiff1d(idx["hex8"], sel)
    blocks = dict(mesh.blocks, hex8=hexes[keep - s0])
    blocks["ssh8"] = np.concatenate([mesh.blocks.get("ssh8", np.zeros((0, 8), np.int64)), new])
    idx = dict(idx, hex8=keep, ssh8=np.concatenate([idx.get("ssh8", sel[:0]), sel]))
    blocks = {t: blocks[t] for t in TYPES if t in blocks and len(blocks[t])}
    order = np.concatenate([idx[t] for t in blocks]).astype(np.int64)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = Mesh(
        points=mesh.points,
        blocks=blocks,
        node_sets=dict(mesh.node_sets),
        element_sets={k: np.sort(inv[v]) for k, v in mesh.element_sets.items()},
        material=mesh.material[order],
        orientation=mesh.orientation[order],
    )
    return out, order
