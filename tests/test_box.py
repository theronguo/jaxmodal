"""box_mesh and the face node sets of box and voxel meshes."""

import subprocess
import sys

import numpy as np
import pytest

from jaxmodal.elements import ELEMENTS, check_elements
from jaxmodal.mesh import box_mesh, voxel_mesh

SHAPE = (3, 2, 4)
SPACING = (0.5, 0.25, 0.75)  # dyadic: coordinates are exact
ORIGIN = (1.0, -0.5, 0.25)


def assert_faces(m, shape):
    """Face sets = nodes whose coordinate equals the bounding grid plane."""
    lo, hi = np.array(ORIGIN), np.array(ORIGIN) + np.array(shape) * np.array(SPACING)
    assert sorted(m.node_sets) == sorted(f"{a}{s}" for a in "xyz" for s in ("min", "max"))
    for d, a in enumerate("xyz"):
        assert np.array_equal(m.node_sets[a + "min"], np.flatnonzero(m.points[:, d] == lo[d]))
        assert np.array_equal(m.node_sets[a + "max"], np.flatnonzero(m.points[:, d] == hi[d]))


@pytest.mark.parametrize("t", ["hex8", "hex20", "hex27"])
def test_box(t):
    nx, ny, nz = SHAPE
    m = box_mesh(SHAPE, SPACING, ORIGIN, t)
    corners = (nx + 1) * (ny + 1) * (nz + 1)
    edges = nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1) + (nx + 1) * (ny + 1) * nz
    n = {
        "hex8": corners,
        "hex20": corners + edges,
        "hex27": (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1),
    }
    assert len(m.points) == n[t] and list(m.blocks) == [t] and m.n_el == nx * ny * nz
    cells = np.array([(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx)])
    expect = np.array(ORIGIN) + np.array(SPACING) * (
        cells[:, None, :] + (ELEMENTS[t].nodes + 1) / 2
    )
    assert np.array_equal(m.points[m.blocks[t]], expect)  # affine reference map, grid cell order
    x, y, z = m.points.T
    assert np.array_equal(np.lexsort((x, y, z)), np.arange(len(m.points)))  # nodes x fastest
    assert len(np.unique(m.points, axis=0)) == len(m.points)
    assert_faces(m, SHAPE)
    assert m.element_sets == {} and m.voxel is None
    assert not m.material.any() and not m.orientation.any()
    check_elements(m, mass="lumped")


def test_box_hex8_is_voxel():
    b, v = box_mesh(SHAPE, SPACING, ORIGIN), voxel_mesh(SHAPE, SPACING, ORIGIN)
    assert np.array_equal(b.points, v.points)
    assert np.array_equal(b.blocks["hex8"], v.blocks["hex8"])
    assert b.node_sets.keys() == v.node_sets.keys()
    assert all(np.array_equal(b.node_sets[k], v.node_sets[k]) for k in b.node_sets)


def test_voxel_face_sets_masked():
    mask = np.random.default_rng(0).random(SHAPE) < 0.6
    mask[0, :, :] = False  # the x = ORIGIN plane loses all cells
    mask[1, 0, 0] = True
    m = voxel_mesh(SHAPE, SPACING, ORIGIN, mask)
    assert_faces(m, SHAPE)
    assert len(m.node_sets["xmin"]) == 0 and len(m.node_sets["xmax"]) > 0


def test_box_raises():
    with pytest.raises(ValueError, match="type must be"):
        box_mesh(SHAPE, type="hex9")
    for args in [((0, 1, 1),), ((1, 1, 1), (1.0, -1.0, 1.0)), ((1, 1), (1.0, 1.0, 1.0))]:
        with pytest.raises(ValueError, match="^(shape|spacing)"):
            box_mesh(*args)


def test_mesh_imports_no_jax():
    code = "import sys, jaxmodal.mesh; assert 'jax' not in sys.modules, 'jax imported'"
    subprocess.run([sys.executable, "-c", code], check=True)
