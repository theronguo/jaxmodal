"""Rigid body modes, M-orthonormalization and the deflation projector against dense references."""

import dataclasses
import functools

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, R, distort, mixed_mesh, rel

from jaxmodal.mesh import box_mesh, parts, rcm, renumber, voxel_mesh
from jaxmodal.operators import (
    assemble,
    constraints,
    dof_mask,
    m_orthonormalize,
    masked,
    project,
    rigid_modes,
    stencil,
    unstructured,
)

ORIENT = {1: R}
OFFSET = (1e3, -2e3, 3e3)  # offset origin (centroid rotations: test_inputs); fine for SciPy's K


def inf(A):
    """||A||_inf of a sparse or dense matrix."""
    return abs(A).sum(1).max()


def sphere():
    shape, h = (10, 12, 11), np.array([0.5, 0.4, 0.3])
    ext = np.array(shape) * h
    c = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) + 0.5) * h
    return voxel_mesh(shape, tuple(h), mask=np.linalg.norm(c - ext / 2, axis=-1) < 0.45 * ext.min())


def beam():
    """hex8 300 x 1 x 1 rotated 45 degrees about z: nearly parallel rotation columns."""
    m = box_mesh((300, 1, 1))
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    return dataclasses.replace(m, points=m.points @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]))


@functools.cache
def mesh(name):
    rng = np.random.default_rng(5)
    return {
        "box": lambda: box_mesh((3, 2, 2), spacing=(1 / 3, 0.5, 0.5), origin=OFFSET),
        "B": lambda: rcm(distort(box_mesh((3, 3, 3), type="hex27"), True, rng))[0],
        "mixed": lambda: mixed_mesh(rng),
        "sphere": sphere,
        "beam": beam,
    }[name]()


def matrices(m, mass="consistent"):
    nodal = np.random.default_rng(1).uniform(0, 1, len(m.points))
    nodal[::3] = 0.0
    return assemble(m, MATERIALS, ORIENT, mass=mass, nodal_mass=nodal)


def null_space(K):
    w, v = np.linalg.eigh(K)
    return v[:, w < 1e-10 * w.max()]


def sin_angle(A, B):
    """Largest principal angle between span(A) and span(B) (equal dimensions)."""
    assert A.shape == B.shape
    Q = np.linalg.qr(A)[0]
    N = np.linalg.qr(B)[0]
    return np.linalg.norm(Q - N @ (N.T @ Q), 2) if A.shape[1] else 0.0


def check_modes(K, M, Rm):
    """||K R|| and ||R^T M R - I|| after m_orthonormalize; returns the M-orthonormal R."""
    if Rm.shape[1]:
        assert inf(K @ Rm) <= 1e-12 * inf(K) * np.abs(Rm).max()
    W = np.asarray(m_orthonormalize(Rm, M @ Rm))
    assert np.abs(W.T @ (M @ W) - np.eye(W.shape[1])).max(initial=0) <= 1e-12
    return W


@pytest.mark.parametrize("name", ["box", "B", "mixed", "sphere", "beam"])
def test_free(name):
    m = mesh(name)
    Rm = rigid_modes(m)
    assert Rm.shape == (3 * len(m.points), 6)
    for mass in ("consistent", "lumped"):
        K, M = matrices(m, mass)
        check_modes(K, M, Rm)
    if name != "beam":
        assert sin_angle(Rm, null_space(K.toarray())) <= 1e-10


def with_sets(m):
    x = m.points - m.points.min(0)
    sets = {"p": np.array([0]), "edge": np.flatnonzero((x[:, 1] == 0) & (x[:, 2] == 0))}
    return dataclasses.replace(m, node_sets=m.node_sets | sets)


BCS = [
    ({"xmin": "x"}, 3),
    ({"xmin": "x", "ymin": "y"}, 1),
    ({"xmin": "x", "ymin": "y", "zmin": "z"}, 0),
    ({"p": "xyz"}, 3),
    ({"edge": "xyz"}, 1),
    ({"p": "x"}, 5),
    ({"xmin": "xyz"}, 0),
]


@pytest.mark.parametrize("bcs,k", BCS)
def test_bcs(bcs, k):
    m = with_sets(box_mesh((3, 2, 2), spacing=(1.0, 0.7, 0.5)))
    K, M = matrices(m)
    free = dof_mask(m, bcs)
    Km, Mm = masked(K, free), masked(M, free)
    Rm = rigid_modes(m, free)
    assert Rm.shape[1] == k
    sv = np.linalg.svd(rigid_modes(m)[~free], compute_uv=False)
    assert ((sv <= 1e-14 * sv[0]) | (sv >= 1e-3 * sv[0])).all()  # the gap RANK_TOL relies on
    assert (Rm[~free] == 0).all()
    W = check_modes(Km, Mm, Rm)
    N = null_space(K.toarray()[np.ix_(free, free)])
    dense = np.zeros((len(free), N.shape[1]))
    dense[free] = N
    assert sin_angle(Rm, dense) <= 1e-10
    if bcs == {"xmin": "x", "ymin": "y"}:  # T_z
        W = W * np.sign(W[2, 0])
        assert max(np.abs(W[0::3]).max(), np.abs(W[1::3]).max()) <= 1e-12 * W[2, 0]
        assert rel(W[2::3], np.full_like(W[2::3], W[2, 0])) <= 1e-12
    for scale in (1e6, 1e-6):  # unit independence
        s = box_mesh((3, 2, 2), spacing=(scale, 0.7 * scale, 0.5 * scale))
        assert rigid_modes(with_sets(s), free).shape[1] == k


def blocks(mask_cells, shape):
    mask = np.zeros(shape, bool)
    for c in mask_cells:
        mask[c] = True
    return voxel_mesh(shape, mask=mask)


def test_parts():
    mask = np.zeros((5, 2, 2), bool)
    mask[:2] = mask[3:] = True
    two = voxel_mesh((5, 2, 2), mask=mask)
    part = parts(two)
    assert (part == (two.points[:, 0] > 2.5)).all()
    R2 = rigid_modes(two)
    assert R2.shape[1] == 12
    for q in (0, 6):  # rotations about each part's centroid
        assert np.abs(R2[:, q : q + 3].T @ R2[:, q + 3 : q + 6]).max() <= 1e-10
    second = np.repeat(two.points[:, 0] > 2.5, 3)
    p2 = np.flatnonzero(two.points[:, 0] > 2.5)[:1]
    two = dataclasses.replace(two, node_sets=two.node_sets | {"p2": p2})
    clamp1 = rigid_modes(two, dof_mask(two, {"xmin": "xyz"}))
    assert clamp1.shape[1] == 6 and (clamp1[~second] == 0).all()
    both = rigid_modes(two, dof_mask(two, {"xmin": "xyz", "p2": "xyz"}))
    only = rigid_modes(two, dof_mask(two, {"p2": "xyz"}))
    assert both.shape[1] == 3 and only.shape[1] == 9
    assert sin_angle(both, only[:, np.abs(only[~second]).max(0) == 0]) <= 1e-12  # per part
    # contacts
    for cells in ([(0, 0, 0), (1, 1, 0)], [(0, 0, 0), (1, 1, 1)]):  # edge, vertex
        m = blocks(cells, (2, 2, 2))
        for f in (parts, generic, rigid_modes):
            with pytest.raises(ValueError, match="without sharing a whole face"):
                f(m)
    assert parts(blocks([(0, 0, 0), (1, 0, 0)], (2, 2, 2))).max() == 0
    assert parts(mesh("mixed")).max() == 0  # faces match across hex8/hex20 blocks
    b = box_mesh((2, 1, 1))
    for i, j in ((1, 0), (5, 4)):  # bottom and top face
        conn = b.blocks["hex8"].copy()
        conn[0, i] = conn[0, j]
        with pytest.raises(ValueError, match="collapsed"):
            parts(dataclasses.replace(b, blocks={"hex8": conn}))
    # labels follow renumbering
    rng = np.random.default_rng(0)
    q = rng.permutation(two.n_el)
    for m2, p in (rcm(two)[:2], (renumber(two, pr := rng.permutation(len(two.points)), q), pr)):
        assert len(set(zip(part[p], parts(m2), strict=True))) == 2
        dofs = (3 * p[:, None] + np.arange(3)).ravel()
        assert sin_angle(rigid_modes(two)[dofs], rigid_modes(m2)) <= 1e-12
    # many parts
    many = blocks([(2 * i, 0, 0) for i in range(constraints.MAX_PARTS + 1)], (21, 1, 1))
    with pytest.raises(ValueError, match="max_parts"):
        rigid_modes(many)
    assert rigid_modes(many, max_parts=constraints.MAX_PARTS + 1).shape[1] == 66


def generic(m):
    return parts(dataclasses.replace(m, voxel=None))


def test_parts_voxel(monkeypatch):
    """Voxel meshes label the cell mask (ndimage.label): the same labels as the generic face
    search, also after rcm and a random renumber (contacts on both paths: test_parts)."""
    import scipy.ndimage

    calls = []
    real = scipy.ndimage.label
    monkeypatch.setattr(scipy.ndimage, "label", lambda *a, **k: calls.append(1) or real(*a, **k))
    mask = np.zeros((6, 3, 2), bool)
    mask[:2] = True  # a block
    mask[3:, 0, :] = mask[5, 1:, :] = True  # an L-shaped part across the gap at x = 2
    mask[3, 2, 1] = True  # a lone cell
    lshape = voxel_mesh((6, 3, 2), spacing=(1.0, 0.5, 2.0), mask=mask)
    two = blocks([np.s_[:2], np.s_[3:]], (5, 2, 2))
    big = blocks([np.s_[:30], np.s_[31:]], (61, 40, 25))
    assert len(big.points) > 2**16 and (parts(big) == (big.points[:, 0] > 30.5)).all()  # key range
    rng = np.random.default_rng(1)
    cases = [lshape, two, big, rcm(lshape)[0]]
    cases.append(
        renumber(lshape, rng.permutation(len(lshape.points)), rng.permutation(lshape.n_el))
    )
    for m in cases:
        calls.clear()
        got = parts(m)
        assert calls and np.array_equal(got, generic(m))
    assert parts(lshape).max() == 2
    calls.clear()
    generic(lshape)
    parts(mixed_mesh(np.random.default_rng(2)))
    assert not calls


def test_projector():
    m0 = mesh("mixed")
    m0 = dataclasses.replace(m0, node_sets={"xmin": np.flatnonzero(m0.points[:, 0] < 0.5)})
    for m, bcs, k in ((m0, {"xmin": "x"}, 3), (mesh("sphere"), {}, 6)):
        K, M = matrices(m)
        free = dof_mask(m, bcs)
        Mm = masked(M, free)
        W = jnp.asarray(check_modes(masked(K, free), Mm, rigid_modes(m, free)))
        assert W.shape[1] == k
        X = np.random.default_rng(2).standard_normal((len(free), 5))
        P = np.asarray(project(W, X, Mm @ X))
        assert np.abs(np.asarray(W).T @ (Mm @ P)).max() <= 1e-12 * np.abs(X).max()
        assert rel(project(W, P, Mm @ P), P) <= 1e-12
        assert np.abs(project(W, W, jnp.asarray(Mm @ np.asarray(W)))).max() <= 1e-12
        assert (project(jnp.zeros((len(free), 0)), X, Mm @ X) == X).all()


@pytest.mark.filterwarnings("error")  # donation warnings are emitted at lowering
def test_device_operators():
    m = sphere()
    _, M = assemble(m, MATERIALS)
    Rm = rigid_modes(m)
    ref = np.asarray(m_orthonormalize(Rm, M @ Rm))
    for op in (stencil(m, MATERIALS), unstructured(m, MATERIALS)):
        Rd = jnp.asarray(Rm)
        assert rel(m_orthonormalize(Rd, op.apply_M(Rd)), ref) <= 1e-12
    n, k = 300_000, 6
    V = jnp.asarray(np.random.default_rng(3).standard_normal((n, k)))
    MV = 2.0 * V
    ma = constraints._cholqr2.lower(V, MV).compile().memory_analysis()
    assert ma.alias_size_in_bytes == ma.output_size_in_bytes
    assert ma.temp_size_in_bytes <= 1.05 * 8 * n * k
    m_orthonormalize(V, MV).block_until_ready()
    assert V.is_deleted() and not MV.is_deleted()


@pytest.mark.filterwarnings("error")
def test_inputs():
    m = mesh("box")
    n_dof = 3 * len(m.points)
    for free in (np.ones(n_dof, int), np.ones(n_dof - 1, bool)):
        with pytest.raises(ValueError, match="mask"):
            rigid_modes(m, free)
    none = rigid_modes(m, np.zeros(n_dof, bool))
    assert none.shape == (n_dof, 0)
    assert m_orthonormalize(jnp.asarray(none), jnp.zeros((n_dof, 0))).shape == (n_dof, 0)
    R0 = rigid_modes(m)
    assert rel(np.linalg.norm(R0, axis=0), np.ones(6)) <= 1e-15
    assert np.abs(R0[:, :3].T @ R0[:, 3:]).max() <= 1e-10  # rotations about the centroid
