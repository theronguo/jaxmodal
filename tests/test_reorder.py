"""renumber, rcm, morton: renumbering rules checked geometrically, RCM against SciPy, Morton
against a per-bit reference, and the operators after morton against the permuted original."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from helpers import MATERIALS, R, mixed_mesh, rel
from scipy.sparse.csgraph import reverse_cuthill_mckee

from jaxmodal.elements import check_elements
from jaxmodal.mesh import Mesh, box_mesh, morton, rcm, renumber, solid_shell, voxel_mesh
from jaxmodal.mesh.reorder import _morton_keys
from jaxmodal.operators import stencil, unstructured


def with_sets(m):
    """mixed_mesh plus node sets and element sets spanning both blocks (one empty)."""
    x = m.points[:, 0]
    return Mesh(
        m.points,
        m.blocks,
        {"left": np.flatnonzero(x < 1.0), "far": np.flatnonzero(x > 3.0)},
        {
            "odd": np.arange(1, m.n_el, 2),
            "span": np.array([2, 7, 8, 13]),
            "none": np.array([], int),
        },
        m.material,
        m.orientation,
    )


def shuffled(m, rng):
    p = rng.permutation(len(m.points))
    q = np.concatenate(
        [
            s + rng.permutation(len(c))
            for s, c in zip(m.offsets.values(), m.blocks.values(), strict=True)
        ]
    )
    return renumber(m, p, q)


def islands():
    mask = np.zeros((5, 3, 2), bool)
    mask[:2, :, :] = True
    mask[3:, :2, 1] = True  # disconnected from the first island
    return voxel_mesh((5, 3, 2), spacing=(0.5, 1.0, 2.0), origin=(1.0, 0.0, -1.0), mask=mask)


def element_coords(m):
    """Global element order -> (n_el, n_max, 3) node coordinates, NaN-padded."""
    out = np.full((m.n_el, 27, 3), np.nan)
    for t, c in m.blocks.items():
        out[m.offsets[t] : m.offsets[t] + len(c), : c.shape[1]] = m.points[c]
    return out


def centroids(m, e):
    return np.sort(np.nanmean(element_coords(m)[e], 1), axis=0)


def node_graph(m):
    rows = np.concatenate(
        [
            np.repeat(np.arange(len(c)) + s, c.shape[1])
            for s, c in zip(m.offsets.values(), m.blocks.values(), strict=True)
        ]
    )
    cols = np.concatenate([c.ravel() for c in m.blocks.values()])
    E = sp.csr_array((np.ones(len(rows)), (rows, cols)), shape=(m.n_el, len(m.points)))
    return (E.T @ E).tocsr()


def check_renumbered(old, new, p, q):
    n, n_el = len(old.points), old.n_el
    assert sorted(p) == list(range(n)) and sorted(q) == list(range(n_el))
    for t, s in old.offsets.items():  # q keeps every block's range
        e = q[s : s + len(old.blocks[t])]
        assert e.min() >= s and e.max() < s + len(old.blocks[t])
    assert list(new.blocks) == list(old.blocks)
    np.testing.assert_array_equal(element_coords(new), element_coords(old)[q])  # connectivity
    np.testing.assert_array_equal(new.material, old.material[q])
    np.testing.assert_array_equal(new.orientation, old.orientation[q])
    for k, s in old.element_sets.items():
        np.testing.assert_array_equal(centroids(new, new.element_sets[k]), centroids(old, s))
    for k, s in old.node_sets.items():
        a, b = (np.sort(m.points[v], axis=0) for m, v in ((new, new.node_sets[k]), (old, s)))
        np.testing.assert_array_equal(a, b)
    if old.voxel is not None:  # maps still point at the same coordinates and cells
        v = new.voxel
        nx, ny, nz = v.shape
        g = np.flatnonzero(v.node_map >= 0)
        ijk = np.stack([g % (nx + 1), g // (nx + 1) % (ny + 1), g // ((nx + 1) * (ny + 1))], 1)
        np.testing.assert_allclose(
            new.points[v.node_map[g]], np.add(v.origin, np.multiply(v.spacing, ijk))
        )
        c = np.flatnonzero(v.element_map >= 0)
        ijk = np.stack([c % nx, c // nx % ny, c // (nx * ny)], 1) + 0.5
        cen = np.nanmean(element_coords(new)[v.element_map[c]], 1)
        np.testing.assert_allclose(cen, np.add(v.origin, np.multiply(v.spacing, ijk)))


@pytest.mark.parametrize("case", ["mixed", "islands"])
def test_rcm(case):
    rng = np.random.default_rng(5)
    base = with_sets(mixed_mesh(rng)) if case == "mixed" else islands()
    old = shuffled(base, rng)
    new, p, q = rcm(old)
    check_renumbered(old, new, p, q)
    np.testing.assert_array_equal(p, reverse_cuthill_mckee(node_graph(old), symmetric_mode=True))
    for c in new.blocks.values():
        assert (np.diff(c.min(1)) >= 0).all()
    check_elements(new, mass="consistent")
    if case == "mixed":
        assert len(new.element_sets["none"]) == 0


def test_renumber():
    rng = np.random.default_rng(6)
    old = with_sets(mixed_mesh(rng))
    p = rng.permutation(len(old.points))
    q = np.concatenate(
        [
            s + rng.permutation(len(c))
            for s, c in zip(old.offsets.values(), old.blocks.values(), strict=True)
        ]
    )
    check_renumbered(old, renumber(old, p, q), p, q)
    with pytest.raises(ValueError, match="out of block"):
        renumber(old, p, np.roll(np.arange(old.n_el), 1))
    with pytest.raises(ValueError, match="permutations"):
        renumber(old, p[:-1], q)
    with pytest.raises(ValueError, match="permutations"):
        renumber(old, p, np.zeros(old.n_el, int))


def ref_keys(x, lo, hi):
    """Morton keys by the formula, one bit at a time in Python integers."""
    q = np.floor((x - lo) / np.maximum(hi - lo, 1e-300) * (2**21 - 1)).astype(np.int64)
    out = []
    for row in q:
        k = 0
        for b in range(21):
            for d in range(3):
                k |= ((int(row[d]) >> b) & 1) << (3 * b + d)
        out.append(k)
    return np.array(out, np.uint64)


def test_morton_keys():
    rng = np.random.default_rng(8)
    x = rng.uniform(-1, 3, (50, 3))
    lo, hi = x.min(0), x.max(0)
    np.testing.assert_array_equal(_morton_keys(x, lo, hi), ref_keys(x, lo, hi))
    x[:, 2] = 0.5  # coplanar: zero extent on z
    lo, hi = x.min(0), x.max(0)
    np.testing.assert_array_equal(_morton_keys(x, lo, hi), ref_keys(x, lo, hi))


@pytest.mark.parametrize("case", ["box", "mixed"])
def test_morton_order(case):
    """Node and per-block element order equal the reference, on the nodes' bounding box."""
    rng = np.random.default_rng(9)
    if case == "box":
        old = box_mesh((4, 3, 2), spacing=(1.0, 2.0, 0.5))
    else:
        old = shuffled(with_sets(mixed_mesh(rng)), rng)
    new, p, q = morton(old)
    x = old.points
    lo, hi = x.min(0), x.max(0)
    np.testing.assert_array_equal(p, np.argsort(ref_keys(x, lo, hi), kind="stable"))
    for t, s in old.offsets.items():
        c = old.blocks[t]
        e = s + np.argsort(ref_keys(x[c].mean(1), lo, hi), kind="stable")
        np.testing.assert_array_equal(q[s : s + len(c)], e)
    check_renumbered(old, new, p, q)


def test_morton_voxel():
    old = islands()
    new, p, q = morton(old)
    check_renumbered(old, new, p, q)


def plate():
    """ssh8 + hex8 plate."""
    box = box_mesh((4, 3, 1), spacing=(1.0, 1.0, 0.1))
    sel = np.flatnonzero(box.points[box.blocks["hex8"]].mean(1)[:, 0] > 2)
    return solid_shell(box, sel, direction=(0, 0, 1))[0]


def voxel_case():
    mask = np.ones((4, 3, 3), bool)
    mask[2:, 1:, 2] = mask[0, 2, :2] = mask[3, 0, 0] = False
    return voxel_mesh((4, 3, 3), spacing=(0.5, 1.0, 0.75), mask=mask)


EQUIV = [("hex8", "scatter"), ("hex8", "gather"), ("hex20", "scatter"), ("mixed", "scatter"),
         ("plate", "scatter"), ("voxel", "stencil")]  # fmt: skip


@pytest.mark.parametrize("case, path", EQUIV)
def test_morton_equivalence(case, path):
    """apply_K and apply_M after morton equal the permuted original (roadmap)."""
    rng = np.random.default_rng(10)
    old = {
        "hex8": lambda: box_mesh((3, 2, 2)),
        "hex20": lambda: box_mesh((2, 2, 2), type="hex20"),
        "mixed": lambda: mixed_mesh(rng),
        "plate": plate,
        "voxel": voxel_case,
    }[case]()
    new, p, q = morton(old)
    v = rng.standard_normal((3 * len(old.points), 3))
    vn = v.reshape(-1, 3, 3)[p].reshape(v.shape)
    if path == "stencil":
        a, b = stencil(old, MATERIALS, {1: R}), stencil(new, MATERIALS, {1: R})
    else:
        a, b = (unstructured(m, MATERIALS, {1: R}, path=path) for m in (old, new))
    for f in ("apply_K", "apply_M"):
        ya = np.asarray(getattr(a, f)(jnp.asarray(v))).reshape(-1, 3, 3)[p].reshape(v.shape)
        assert rel(np.asarray(getattr(b, f)(jnp.asarray(vn))), ya) < 1e-12
