"""PR 2: elastic constructors, rotations, Bond matrices, phases.

References are independent of the implementation: einsum rotation of the 4th-order tensor,
hand-typed compliance entries, SciPy's Euler rotations, hand-derived rotation cases.
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jaxmodal.materials import (
    Material,
    anisotropic,
    bond_strain,
    bond_stress,
    check_materials,
    check_stiffness,
    from_euler,
    from_vectors,
    isotropic,
    orthotropic,
    phase_table,
    phases,
    rotate,
    transversely_isotropic,
)
from jaxmodal.mesh import read, voxel_mesh

DATA = Path(__file__).parent / "data"
PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]  # Voigt index -> tensor indices
CFRP = (140.0, 10.0, 0.3, 0.45, 5.0)  # transversely isotropic, GPa
ORTHO = [
    (140.0, 10.0, 8.0, 0.3, 0.25, 0.45, 5.0, 4.0, 3.4),  # E1..E3, nu12, nu13, nu23, G12, G13, G23
    (200.0, 150.0, 100.0, 0.25, 0.2, 0.33, 70.0, 60.0, 50.0),
]
RNG = np.random.default_rng(0)
ROTS = Rotation.random(5, random_state=1).as_matrix()


def rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max()


def tensor(C):
    """Voigt (6, 6) stiffness -> 4th-order tensor C_ijkl."""
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(PAIRS):
        for b, (k, n) in enumerate(PAIRS):
            for p, q in ((i, j), (j, i)):
                for r, s in ((k, n), (n, k)):
                    T[p, q, r, s] = C[a, b]
    return T


def voigt_stiffness(T):
    return np.array([[T[i, j, k, n] for k, n in PAIRS] for i, j in PAIRS])


def strain_voigt(e):
    """Symmetric strain tensor -> engineering Voigt vector."""
    return np.array([e[i, j] * (1 if i == j else 2) for i, j in PAIRS])


def stress_voigt(s):
    return np.array([s[i, j] for i, j in PAIRS])


def random_spd():
    A = RNG.normal(size=(6, 6))
    return A @ A.T + 6 * np.eye(6)


# --- 1. Bond = einsum -----------------------------------------------------------------------


def test_bond_against_einsum():
    C = random_spd()
    for R in ROTS:
        ref = voigt_stiffness(np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, tensor(C)))
        assert rel(rotate(C, R), ref) < 1e-14
        M, N = np.asarray(bond_stress(R)), np.asarray(bond_strain(R))
        assert rel(N, np.linalg.inv(M).T) < 1e-14
        e = RNG.normal(size=(3, 3))
        e = e + e.T
        assert rel(N @ strain_voigt(e), strain_voigt(R @ e @ R.T)) < 1e-14
        s = RNG.normal(size=(3, 3))
        s = s + s.T
        assert rel(M @ stress_voigt(s), stress_voigt(R @ s @ R.T)) < 1e-14
        el = strain_voigt(e)
        sl = C @ el
        assert rel((M @ sl) @ (N @ el), sl @ el) < 1e-14  # work is frame independent


# --- 2. Invariance --------------------------------------------------------------------------


def test_invariance():
    C = isotropic(210.0, 0.3)
    for R in ROTS:
        assert rel(rotate(C, R), C) < 1e-14
    T = transversely_isotropic(*CFRP)
    for t in RNG.uniform(0, 2 * np.pi, 5):
        assert rel(rotate(T, from_euler((0.0, t, 0.0))), T) < 1e-14  # Rx: about local axis 1
    assert rel(rotate(T, from_euler((np.pi / 2, 0.0, 0.0))), T) > 0.1


# --- 3. Positive definiteness ---------------------------------------------------------------


def spectrum(t, Q):
    C = Q @ np.diag([t, 0.3, 0.5, 0.7, 0.9, 1.0]) @ Q.T * 200e9
    return (C + C.T) / 2


def test_check_stiffness_passes():
    cubic = np.zeros((6, 6))
    cubic[:3, :3] = 124.0
    cubic[range(3), range(3)] = 168.0
    cubic[range(3, 6), range(3, 6)] = 75.0  # copper, GPa
    for C in (isotropic(210e9, 0.3), transversely_isotropic(*CFRP), anisotropic(cubic)):
        check_stiffness(C)
        for R in ROTS:
            check_stiffness(rotate(C, R))
    for c in ORTHO:
        check_stiffness(orthotropic(*c))
    for _ in range(20):
        Q = np.linalg.qr(RNG.normal(size=(6, 6)))[0]
        check_stiffness(spectrum(1e-11, Q))
        with pytest.raises(ValueError, match="positive definite"):
            check_stiffness(spectrum(1e-13, Q))


@pytest.mark.parametrize(
    "C, match",
    [
        (isotropic(1.0, 0.5), "finite"),
        (isotropic(1.0, 0.5 - 1e-14), "positive definite"),
        (orthotropic(140.0, 10.0, 10.0, 4.0, 0.3, 0.45, 5.0, 5.0, 3.4), "positive definite"),
        (np.eye(6) + np.triu(np.ones((6, 6)), 1) * 0.1, "symmetric"),
        (np.eye(6) + np.triu(np.ones((6, 6)), 1) * 1e-11, "symmetric"),  # pins the 1e-12 tolerance
        (np.where(np.eye(6) > 0, np.nan, 0.0), "finite"),
        (np.eye(6)[:, :5], "finite"),
    ],
)
def test_check_stiffness_raises(C, match):
    with pytest.raises(ValueError, match=match):
        check_stiffness(C)


# --- 4. Compliance --------------------------------------------------------------------------


@pytest.mark.parametrize("c", ORTHO)
def test_orthotropic_compliance(c):
    E1, E2, E3, n12, n13, n23, G12, G13, G23 = c
    n21, n31, n32 = n12 * E2 / E1, n13 * E3 / E1, n23 * E3 / E2
    S = np.array(
        [
            [1 / E1, -n21 / E2, -n31 / E3, 0, 0, 0],
            [-n12 / E1, 1 / E2, -n32 / E3, 0, 0, 0],
            [-n13 / E1, -n23 / E2, 1 / E3, 0, 0, 0],
            [0, 0, 0, 1 / G23, 0, 0],
            [0, 0, 0, 0, 1 / G13, 0],
            [0, 0, 0, 0, 0, 1 / G12],
        ]
    )
    assert np.isclose(S[1, 0], -n21 / E2) and np.isclose(S[0, 1], S[1, 0])
    assert rel(np.linalg.inv(orthotropic(*c)), S) < 1e-14


def test_constructors_agree():
    E, nu = 210.0, 0.3
    G = E / (2 * (1 + nu))
    assert rel(isotropic(E, nu), orthotropic(E, E, E, nu, nu, nu, G, G, G)) < 1e-14
    E1, E2, n12, n23, G12 = CFRP
    full = orthotropic(E1, E2, E2, n12, n12, n23, G12, G12, E2 / (2 * (1 + n23)))
    assert rel(transversely_isotropic(*CFRP), full) < 1e-14
    C = random_spd()
    assert anisotropic(C).dtype == jnp.float64 and rel(anisotropic(C), C) == 0
    assert isotropic(200, 0).dtype == jnp.float64


# --- 5. Orientations ------------------------------------------------------------------------


def test_from_euler():
    angles = RNG.uniform(-np.pi, np.pi, (5, 3))
    for a in angles:
        assert rel(from_euler(a), Rotation.from_euler("ZXZ", a).as_matrix()) < 1e-14
    ex, ey, ez = np.eye(3)
    assert np.allclose(from_euler((0.0, 0.0, 0.0)), np.eye(3), atol=0)
    assert np.allclose(np.asarray(from_euler((np.pi / 2, 0.0, 0.0)))[:, 0], ey, atol=1e-15)
    R = from_euler((np.pi / 2, np.pi / 2, 0.0))
    assert np.allclose(R, np.stack([ey, ez, ex], 1), atol=1e-15)
    assert rel(from_euler(angles), np.stack([from_euler(a) for a in angles])) < 1e-15


def test_from_vectors():
    angles = RNG.uniform(-np.pi, np.pi, (5, 3))
    R = np.asarray(from_euler(angles))
    a, b = 3 * R[:, :, 0], 2 * R[:, :, 0] + 5 * R[:, :, 1]
    assert rel(from_vectors(a, b), R) < 1e-14
    assert (
        rel(from_vectors(a, b), np.stack([from_vectors(x, y) for x, y in zip(a, b, strict=True)]))
        < 1e-15
    )
    assert np.isnan(np.asarray(from_vectors([1.0, 0, 0], [2.0, 0, 0]))[:, 1:]).all()  # a || b


def test_rotation_physical_and_batched():
    C = transversely_isotropic(*CFRP)
    Cg = rotate(C, from_euler((np.pi / 2, 0.0, 0.0)))  # fiber along global y
    assert np.isclose(Cg[1, 1], C[0, 0], rtol=1e-14) and np.isclose(Cg[0, 0], C[1, 1], rtol=1e-14)
    Cs = np.stack([random_spd() for _ in ROTS])
    looped = np.stack([rotate(c, R) for c, R in zip(Cs, ROTS, strict=True)])
    assert rel(rotate(Cs, ROTS), looped) < 1e-15


# --- 6. Phases ------------------------------------------------------------------------------


def labeled(material, orientation):
    m = voxel_mesh((len(material), 1, 1))
    return replace(m, material=np.array(material), orientation=np.array(orientation))


def materials():
    return {
        5: Material(transversely_isotropic(*CFRP), 1.6),
        -1: Material(isotropic(210.0, 0.3), 7.8),
        0: Material(orthotropic(*ORTHO[1]), 4.5),
        9: Material(isotropic(70.0, 0.33), 2.7),  # unused entries are allowed
    }


def test_phases():
    index, pairs = phases(labeled([5, -1, 5, 0], [0, 2, 2, 0]))
    assert pairs == ((5, 0), (-1, 2), (5, 2), (0, 0)) and index.tolist() == [0, 1, 2, 3]
    assert index.dtype == np.int64
    index, pairs = phases(labeled([5, 5, -1, 5], [0, 0, 2, 0]))
    assert pairs == ((5, 0), (-1, 2)) and index.tolist() == [0, 0, 1, 0]
    mats, R2 = materials(), from_euler((0.3, 1.1, -0.4))
    m = labeled([5, -1, 5, 0], [0, 2, 2, 0])
    index, pairs = phases(m)
    check_materials(pairs, mats, {2: R2})
    C, rho = phase_table(pairs, mats, {2: R2})
    for e, (lm, lo) in enumerate(zip(m.material, m.orientation, strict=True)):
        R = R2 if lo == 2 else np.eye(3)
        assert rel(C[index[e]], rotate(mats[lm].C, R)) < 1e-15
        assert rho[index[e]] == mats[lm].rho


def test_orientation_default():
    index, pairs = phases(labeled([5, 5], [0, 2]))
    mats, R2 = materials(), from_euler((0.3, 1.1, -0.4))
    check_materials(pairs, mats, {2: R2})
    C, _ = phase_table(pairs, mats, {2: R2})
    assert rel(C[0], mats[5].C) == 0
    R0 = from_euler((0.2, 0.0, 0.0))  # an explicit label-0 entry overrides the identity
    check_materials(pairs, mats, {0: R0, 2: R2})
    C, _ = phase_table(pairs, mats, {0: R0, 2: R2})
    assert rel(C[0], rotate(mats[5].C, R0)) < 1e-15


@pytest.mark.parametrize(
    "mats, ors, match",
    [
        ({}, {}, "material label 5"),
        ({5: Material(isotropic(1.0, 0.3), 1.0)}, {}, "orientation label 2"),
        ({5: Material(isotropic(1.0, 0.5 - 1e-14), 1.0)}, None, "positive definite"),
        ({5: Material(isotropic(1.0, 0.3), 0.0)}, None, "rho"),
        ({5: Material(isotropic(1.0, 0.3), np.nan)}, None, "rho"),
        ({5: Material(isotropic(1.0, 0.3), np.ones(2))}, None, "rho"),
        ({5: Material(isotropic(1.0, 0.3), 1.0)}, {2: np.diag([1.0, 1, -1])}, "not a rotation"),
        ({5: Material(isotropic(1.0, 0.3), 1.0)}, {2: 2 * np.eye(3)}, "not a rotation"),
        ({5: Material(isotropic(1.0, 0.3), 1.0)}, {2: np.eye(3) + 1e-10}, "not a rotation"),
        (
            {5: Material(isotropic(1.0, 0.3), 1.0)},
            {2: np.eye(3), 0: np.diag([1.0, 1, -1])},
            "orientation 0",
        ),
        (
            {-1: Material(isotropic(1.0, 0.3), 1.0), 5: Material(isotropic(1.0, 0.3), -1.0)},
            None,
            "material 5",
        ),
        (
            {5: Material(isotropic(1.0, 0.3), 1.0)},
            {2: from_vectors([1.0, 0, 0], [1.0, 0, 0])},
            "finite",
        ),
    ],
)
def test_check_materials_raises(mats, ors, match):
    pairs = ((-1, 0), (5, 0), (5, 2)) if -1 in mats else ((5, 0), (5, 2))
    if ors is None:  # orientation 2 valid; the material is the bad part
        ors = {2: np.eye(3)}
    with pytest.raises(ValueError, match=match):
        check_materials(pairs, mats, ors)


def test_phase_table_missing_label():
    with pytest.raises(KeyError):
        phase_table(((3, 0),), materials())
    with pytest.raises(KeyError):
        phase_table(((5, 2),), materials())  # only orientation 0 defaults


def first_appearance(m):
    pairs = []
    for p in zip(m.material.tolist(), m.orientation.tolist(), strict=True):
        if p not in pairs:
            pairs.append(p)
    return tuple(pairs), [
        pairs.index(p) for p in zip(m.material.tolist(), m.orientation.tolist(), strict=True)
    ]


INP = """*NODE
1, 0., 0., 0.
2, 1., 0., 0.
3, 1., 1., 0.
4, 0., 1., 0.
5, 0., 0., 1.
6, 1., 0., 1.
7, 1., 1., 1.
8, 0., 1., 1.
9, 2., 0., 0.
10, 2., 1., 0.
11, 2., 0., 1.
12, 2., 1., 1.
*ELEMENT, TYPE=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
2, 2, 9, 10, 3, 6, 11, 12, 7
3, 1, 2, 3, 4, 5, 6, 7, 8
*ELSET, ELSET=material_3
2, 3
*ELSET, ELSET=orientation_4
1, 3
"""


def test_phases_real_meshes(tmp_path):
    (tmp_path / "m.inp").write_text(INP)
    for m in (read(DATA / "gmsh_messy.msh"), read(tmp_path / "m.inp")):
        index, pairs = phases(m)
        ref_pairs, ref_index = first_appearance(m)
        assert pairs == ref_pairs and index.tolist() == ref_index
    assert phases(read(DATA / "gmsh_messy.msh"))[1] == ((7, 0), (8, 0))
    assert set(phases(read(tmp_path / "m.inp"))[1]) == {(0, 4), (3, 0), (3, 4)}


def test_jit_vmap_pytree():
    _, pairs = phases(labeled([5, -1, 5, 0], [0, 2, 2, 0]))
    table = jax.jit(phase_table, static_argnames="pairs")
    for t in (0.3, 0.7):
        args = materials(), {2: from_euler((t, 1.1, -0.4))}
        C, rho = table(pairs, *args)
        C0, rho0 = phase_table(pairs, *args)
        assert rel(C, C0) < 1e-15 and rel(rho, rho0) == 0
    assert table._cache_size() == 1
    mat = jax.jit(lambda m: m)(Material(isotropic(1.0, 0.3), 2.0))
    assert isinstance(mat, Material) and mat.rho == 2.0
    E, nu = jnp.array([1.0, 2.0, 3.0]), jnp.array([0.1, 0.2, 0.3])
    looped = np.stack([isotropic(a, b) for a, b in zip(E, nu, strict=True)])
    assert rel(jax.vmap(isotropic)(E, nu), looped) < 1e-15
    for f, x in ((isotropic, (1.0, 0.3)), (orthotropic, ORTHO[0]), (transversely_isotropic, CFRP)):
        assert rel(jax.jit(f)(*x), f(*x)) < 1e-15
    C = random_spd()
    assert rel(jax.jit(rotate)(C, ROTS[0]), rotate(C, ROTS[0])) < 1e-15


# --- 7. Gradients ---------------------------------------------------------------------------

W = RNG.normal(size=(6, 6))


def fd_error(f, x):
    """Relative inf-norm error of jax.grad against central differences (best of three steps)."""
    x = np.asarray(x, float)
    g = np.asarray(jax.grad(f)(jnp.asarray(x)))
    best = np.inf
    for h in (1e-4, 1e-5, 1e-6):
        fd = np.empty_like(x)
        for i in range(len(x)):
            d = np.zeros_like(x)
            d[i] = h * max(abs(x[i]), 1.0)
            fd[i] = (f(jnp.asarray(x + d)) - f(jnp.asarray(x - d))) / (2 * d[i])
        best = min(best, np.abs(fd - g).max() / np.abs(g).max())
    return best


def test_gradients():
    R = from_euler((0.3, 1.1, -0.4))
    cases = [
        (lambda p: jnp.sum(W * rotate(isotropic(p[0], p[1]), R)), [210.0, 0.3]),
        (lambda p: jnp.sum(W * rotate(orthotropic(*p), R)), list(ORTHO[0])),
        (
            lambda p: jnp.sum(W * rotate(transversely_isotropic(*CFRP), from_euler(p))),
            [0.3, 1.1, -0.4],
        ),
        (
            lambda p: jnp.sum(
                W * rotate(transversely_isotropic(*CFRP), from_vectors(p[:3], p[3:]))
            ),
            [1.0, 0.2, -0.3, 0.1, 1.0, 0.4],
        ),
    ]

    def through_table(p):
        _, pairs = phases(labeled([5, 0], [0, 2]))
        c = list(ORTHO[1])
        c[0] = p[1]
        mats = {5: Material(transversely_isotropic(*CFRP), p[2]), 0: Material(orthotropic(*c), 4.5)}
        C, rho = phase_table(pairs, mats, {2: from_euler(jnp.stack([p[0], 1.1, -0.4]))})
        return jnp.sum(W * C) + rho.sum()

    cases.append((through_table, [0.3, 200.0, 1.6]))
    for f, x in cases:
        assert fd_error(f, x) < 1e-6
