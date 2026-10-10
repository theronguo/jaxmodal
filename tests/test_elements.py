"""PR 3: element registry, hex8/hex20/hex27 kernels, host element check.

References are independent of the implementation: hand-typed VTK coordinates and shape functions,
complex-step derivatives, leggauss, np.linalg, explicit loops for J, B and the quadrature sums, and
the 4th-order tensor C_ijkl for the Voigt-free patch energy.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jaxmodal.elements import (
    ELEMENTS,
    bmatrix,
    check_elements,
    gauss,
    gradients,
    jacobian,
    lumped_mass,
    mass,
    mass_form,
    stiffness,
    stiffness_form,
)
from jaxmodal.elements.registry import _det3
from jaxmodal.materials import from_euler, isotropic, orthotropic, rotate
from jaxmodal.mesh import Mesh, corner_jacobians, read, voxel_mesh
from tests.helpers import rel, tensor

DATA = Path(__file__).parent / "data"
TYPES = ("hex8", "hex20", "hex27")
NODES = {"hex8": 8, "hex20": 20, "hex27": 27}
ORDER = {"hex8": 2, "hex20": 3, "hex27": 3}
PERT = {"hex8": 0.2, "hex20": 0.1, "hex27": 0.1}  # roadmap distortion bounds
# VTK reference coordinates, typed by hand: corners, edge midpoints, face centres, centre.
VTK = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
        [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
        [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
        [0, 0, 0],
    ],
    float,
)  # fmt: skip
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
EDGES += [(0, 4), (1, 5), (2, 6), (3, 7)]
C_ISO = np.asarray(isotropic(1.0, 0.49))
ORTHO = (140.0, 10.0, 8.0, 0.3, 0.25, 0.45, 5.0, 4.0, 3.4)
C_ORT = np.asarray(rotate(orthotropic(*ORTHO), from_euler((0.5, 0.3, 1.0))))


# ---------------------------------------------------------------- hand-typed references


def hand_shape(name, xi):
    """Textbook shape functions at one point xi (3,), complex-safe."""
    x = VTK[: NODES[name]]
    if name == "hex8":
        return np.prod(1 + xi * x, 1) / 8
    if name == "hex27":
        lag = {-1.0: lambda t: t * (t - 1) / 2, 0.0: lambda t: 1 - t * t}
        lag[1.0] = lambda t: t * (t + 1) / 2
        return np.array([np.prod([lag[c](xi[d]) for d, c in enumerate(a)]) for a in x])
    out = []
    for a in x:
        if (a != 0).all():  # serendipity corner
            out.append(np.prod(1 + xi * a) * (xi @ a - 2) / 8)
        else:
            d = np.flatnonzero(a == 0)[0]
            o = [k for k in range(3) if k != d]
            out.append((1 - xi[d] ** 2) * np.prod([1 + xi[k] * a[k] for k in o]) / 4)
    return np.array(out)


def hand_grad(name, xi, h=1e-30):
    """Complex-step derivatives (exact for polynomials): (n, 3)."""
    return np.stack([hand_shape(name, xi + 1j * h * np.eye(3)[d]).imag / h for d in range(3)], -1)


def hand_points(name):
    g, w = np.polynomial.legendre.leggauss(ORDER[name])
    ijk = [idx[::-1] for idx in np.ndindex((len(g),) * 3)]  # xi fastest
    return [(g[list(t)], w[list(t)].prod()) for t in ijk]


def reference(name, X, C, rho):
    """Loop reference: J, G, wdet per point, Ke, Me, scalar m, V."""
    n = len(X)
    K, m, out = np.zeros((3 * n, 3 * n)), np.zeros((n, n)), []
    for xi, w in hand_points(name):
        N, dN = hand_shape(name, xi), hand_grad(name, xi)
        J = np.zeros((3, 3))
        for a, i, j in np.ndindex(n, 3, 3):
            J[i, j] += X[a, i] * dN[a, j]
        det, G = np.linalg.det(J), dN @ np.linalg.inv(J)
        B = np.zeros((6, 3 * n))
        for a, (gx, gy, gz) in enumerate(G):
            rows = [[gx, 0, 0], [0, gy, 0], [0, 0, gz], [0, gz, gy], [gz, 0, gx], [gy, gx, 0]]
            B[:, 3 * a : 3 * a + 3] = rows
        K += w * det * B.T @ C @ B
        m += w * det * np.outer(N, N)
        out.append((np.linalg.inv(J), w * det, G))
    invJ, wdet, G = (np.array(a) for a in zip(*out, strict=True))
    return invJ, wdet, G, K, np.kron(rho * m, np.eye(3)), m, wdet.sum()


def det_nodes_and_points(name, X):
    xi = np.vstack([VTK[: NODES[name]], [p for p, _ in hand_points(name)]])
    return np.array([np.linalg.det(X.T @ hand_grad(name, x)) for x in xi])


def distorted(name, rng, count, pert=None):
    """Random elements: edges in [1, 2], rigidly rotated, all nodes perturbed by <= pert * shortest
    edge; kept if det J > 0 at the nodes and the quadrature points (>= 90% must be kept)."""
    pert, out, tries = PERT[name] if pert is None else pert, [], 0
    while len(out) < count:
        tries += 1
        L = rng.uniform(1, 2, 3)
        X = VTK[: NODES[name]] / 2 * L + rng.uniform(-1, 1, (NODES[name], 3)) * pert * L.min()
        X = X @ Rotation.random(random_state=rng).as_matrix().T + rng.uniform(-5, 5, 3)
        if det_nodes_and_points(name, X).min() > 0:
            out.append(X)
    assert count / tries >= 0.9
    return out


def rigid_modes(X):
    n, c = len(X), X - X.mean(0)
    R = np.zeros((3 * n, 6))
    for i in range(3):
        R[i::3, i] = 1
    for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):  # rotation about axis k
        R[i::3, 3 + k], R[j::3, 3 + k] = -c[:, j], c[:, i]
    return R


def one_mesh(blocks):
    """Mesh from {type: [X_e, ...]} with unshared nodes."""
    pts, cells, start = [], {}, 0
    for t, Xs in blocks.items():
        cells[t] = np.arange(start, start + len(Xs) * NODES[t]).reshape(len(Xs), NODES[t])
        pts += list(Xs)
        start += len(Xs) * NODES[t]
    n_el = sum(len(c) for c in cells.values())
    z = np.zeros(n_el, np.int64)
    return Mesh(np.vstack(pts).astype(float), cells, {}, {}, z, z.copy())


def unit(name):
    return VTK[: NODES[name]] / 2 + 0.5


# ---------------------------------------------------------------- 1-3: quadrature, shape functions


@pytest.mark.parametrize("n", [2, 3])
def test_gauss_exactness(n):
    p, w = gauss(n)
    assert np.isclose(w.sum(), 8, rtol=1e-14, atol=0)
    for e in np.ndindex(2 * n, 2 * n, 2 * n):
        exact = np.prod([(1 - (-1) ** (k + 1)) / (k + 1) for k in e])
        assert abs(w @ np.prod(p ** np.array(e), 1) - exact) <= 1e-14


@pytest.mark.parametrize("name", TYPES)
def test_shape_functions_match_hand_formulas(name):
    el, rng = ELEMENTS[name], np.random.default_rng(1)
    assert np.array_equal(el.nodes, VTK[: NODES[name]])
    for xi in rng.uniform(-1, 1, (50, 3)):
        assert rel(el.shape(xi), hand_shape(name, xi)) <= 1e-14
        assert rel(el.grad(xi), hand_grad(name, xi)) <= 1e-14
    assert np.abs(el.shape(el.nodes) - np.eye(NODES[name])).max() <= 1e-14
    assert np.array_equal(el.N, el.shape(el.points)) and np.array_equal(el.dN, el.grad(el.points))
    assert el.N.shape == (ORDER[name] ** 3, NODES[name])


@pytest.mark.parametrize("name", TYPES)
def test_partition_of_unity_and_completeness(name):
    el = ELEMENTS[name]
    xi = np.random.default_rng(2).uniform(-1, 1, (50, 3))
    N, dN, x = el.shape(xi), el.grad(xi), el.nodes
    assert np.abs(N.sum(1) - 1).max() <= 1e-14
    assert np.abs(dN.sum(1)).max() <= 1e-14
    assert np.abs(N @ x - xi).max() <= 1e-14
    assert np.abs(np.einsum("ai,paj->pij", x, dN) - np.eye(3)).max() <= 1e-14
    if name != "hex8":
        assert np.abs(N @ x**2 - xi**2).max() <= 1e-14
        assert np.abs(N @ (x[:, 0] * x[:, 1]) - xi[:, 0] * xi[:, 1]).max() <= 1e-14


# ---------------------------------------------------------------- 4-6: kernels vs references


@pytest.mark.parametrize("name", TYPES)
def test_kernels_match_numpy_reference(name):
    for X in distorted(name, np.random.default_rng(3), 5):
        invJ, wdet, G, K, M, m, V = reference(name, X, C_ORT, 2.5)
        a_invJ, a_wdet = jacobian(name, X)
        assert rel(a_invJ, invJ) <= 1e-12 and rel(a_wdet, wdet) <= 1e-12
        assert rel(gradients(name, a_invJ), G) <= 1e-12
        assert rel(stiffness(name, X, C_ORT), K) <= 1e-12
        assert rel(mass(name, X, 2.5), M) <= 1e-12
        d = np.diag(m)
        nodal = m.sum(1) if name != "hex20" else V * d / d.sum()
        assert rel(lumped_mass(name, X, 2.5), np.repeat(2.5 * nodal, 3)) <= 1e-12
        # host check's det J (NumPy path) equals the traced one
        host = _det3(np.einsum("ai,qaj->qij", X, ELEMENTS[name].dN))
        assert rel(host, np.asarray(a_wdet) / ELEMENTS[name].weights) <= 1e-14


def test_closed_form_inverse_matches_linalg():
    J = np.eye(3) + 0.3 * np.random.default_rng(4).standard_normal((2000, 3, 3))
    J = J[np.linalg.cond(J) < 10]  # well-conditioned: the comparison measures round-off only
    assert len(J) > 1900
    X = np.einsum("pji,ai->paj", J, ELEMENTS["hex8"].nodes)  # x = J xi: affine, Jacobian J
    invJ, wdet = jacobian("hex8", X)
    assert rel(invJ[:, 0], np.linalg.inv(J)) <= 1e-12
    assert rel(wdet[:, 0], np.linalg.det(J)) <= 1e-14
    assert rel(_det3(J), np.linalg.det(J)) <= 1e-14


@pytest.mark.parametrize("name", TYPES)
def test_patch_energy_is_voigt_free(name):
    rng = np.random.default_rng(5)
    T = tensor(C_ORT)
    for X in distorted(name, rng, 20):
        H = rng.standard_normal((3, 3))
        u = (X @ H.T).ravel()
        eps = (H + H.T) / 2
        V = np.asarray(jacobian(name, X)[1]).sum()
        energy = V * np.einsum("ij,ijkl,kl", eps, T, eps)
        assert rel(u @ stiffness(name, X, C_ORT) @ u, energy) <= 1e-12


@pytest.mark.parametrize("name", TYPES)
def test_six_zero_modes_on_distorted_elements(name):
    for X in distorted(name, np.random.default_rng(6), 20):
        R = rigid_modes(X)
        for C in (C_ISO, C_ORT):
            K = np.asarray(stiffness(name, X, C))
            lam = np.sort(np.abs(np.linalg.eigvalsh(K)))
            assert lam[5] < 1e-10 * lam[-1] and lam[6] > 1e-6 * lam[-1]
            assert np.abs(K @ R).max() <= 1e-12 * np.abs(K).max() * np.abs(R).max()


# ---------------------------------------------------------------- 7: lumping


def test_lumped_closed_form_on_undistorted_box():
    L = np.array([2.0, 1.0, 0.5])
    V = L.prod()
    for name, expect in [
        ("hex8", [27] * 8),
        ("hex20", [7 * 216 / 248] * 8 + [16 * 216 / 248] * 12),
        ("hex27", [1] * 8 + [4] * 12 + [16] * 6 + [64]),
    ]:
        X = VTK[: NODES[name]] / 2 * L
        assert rel(lumped_mass(name, X, 3.0)[::3], 3.0 * V * np.array(expect) / 216) <= 1e-14
    # hex20 row-sum would give negative corners: -V/8 (midsides V/6); hence HRZ
    X = VTK[:20] / 2 * L
    rowsum = np.asarray(jacobian("hex20", X)[1]) @ ELEMENTS["hex20"].N
    assert rel(rowsum, V * np.array([-1 / 8] * 8 + [1 / 6] * 12)) <= 1e-14


@pytest.mark.parametrize("name", TYPES)
def test_total_mass_per_direction(name):
    for X in distorted(name, np.random.default_rng(7), 5):
        rho, V = 2.5, np.asarray(jacobian(name, X)[1]).sum()
        ml, M = np.asarray(lumped_mass(name, X, rho)), np.asarray(mass(name, X, rho))
        for i in range(3):
            t = np.zeros(3 * NODES[name])
            t[i::3] = 1
            assert rel(ml[i::3].sum(), rho * V) <= 1e-12
            assert rel(t @ M @ t, rho * V) <= 1e-12


def mild_hex27(rng):
    """Corners perturbed <= 10%, other nodes offset <= 5% of the shortest edge from straight."""
    L = rng.uniform(1, 2, 3)
    X = np.zeros((27, 3))
    X[:8] = VTK[:8] / 2 * L + rng.uniform(-1, 1, (8, 3)) * 0.1 * L.min()
    X[8:20] = [(X[a] + X[b]) / 2 for a, b in EDGES]
    faces = [(0, 3, 7, 4), (1, 2, 6, 5), (0, 1, 5, 4), (3, 2, 6, 7), (0, 1, 2, 3), (4, 5, 6, 7)]
    X[20:26] = [X[list(f)].mean(0) for f in faces]
    X[26] = X[:8].mean(0)
    X[8:] += rng.uniform(-1, 1, (19, 3)) * 0.05 * L.min()
    return X


@pytest.mark.parametrize("name", TYPES)
def test_lumped_masses_positive_on_mild_distortion(name):
    rng = np.random.default_rng(8)
    Xs = [mild_hex27(rng) for _ in range(50)] if name == "hex27" else distorted(name, rng, 50)
    for X in Xs:
        assert np.asarray(lumped_mass(name, X, 1.0)).min() > 0


def test_lumped_check_raises_on_negative_hex27_mass():
    X = unit("hex27")
    X[8, 0] -= 0.45  # midside node of edge 0-1 slid 45% toward corner 0
    assert np.asarray(jacobian("hex27", X)[1]).min() > 0
    assert np.asarray(lumped_mass("hex27", X, 1.0)).min() < 0
    mesh = one_mesh({"hex27": [X]})
    check_elements(mesh)
    with pytest.raises(
        ValueError, match=r"element 0 \(hex27\): row-sum lumped nodal mass -0.000704 "
    ):
        check_elements(mesh, mass="lumped")  # value equals the traced row-sum minimum above
    assert f"{np.asarray(lumped_mass('hex27', X, 1.0)).min():.3g}" == "-0.000704"
    # global index across blocks and chunks
    mesh = one_mesh({"hex8": [unit("hex8")] * 2, "hex27": [unit("hex27")] * 3 + [X]})
    with pytest.raises(ValueError, match=r"element 5 \(hex27\): row-sum"):
        check_elements(mesh, mass="lumped", chunk=2)


# ---------------------------------------------------------------- 8: gradients


def fd_rel(f, x, g):
    best = np.inf
    for h in (1e-4, 1e-5, 1e-6):
        step = h * np.maximum(np.abs(x), 1)
        fd = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            e = np.zeros_like(x)
            e[idx] = step[idx]
            fd[idx] = (f(x + e) - f(x - e)) / (2 * step[idx])
        best = min(best, rel(g, fd))
    return best


@pytest.mark.parametrize("name", TYPES)
def test_grad_wrt_coordinates_matches_fd(name):
    rng = np.random.default_rng(9)
    X = distorted(name, rng, 1)[0]
    u = rng.standard_normal(3 * NODES[name])
    for f in (
        jax.jit(lambda X: u @ stiffness(name, X, C_ORT) @ u),
        jax.jit(lambda X: u @ mass(name, X, 2.5) @ u),
    ):
        assert fd_rel(lambda X, f=f: float(f(X)), X, np.asarray(jax.grad(f)(X))) < 1e-6


def test_grad_wrt_euler_angles_matches_fd():
    rng = np.random.default_rng(10)
    X, u = distorted("hex8", rng, 1)[0], rng.standard_normal(24)
    C = orthotropic(*ORTHO)
    f = jax.jit(lambda a: u @ stiffness("hex8", X, rotate(C, from_euler(a))) @ u)
    a = np.array([0.5, 0.3, 1.0])
    assert fd_rel(lambda a: float(f(a)), a, np.asarray(jax.grad(f)(a))) < 1e-6


# ---------------------------------------------------------------- 9: Jacobian check


def test_check_passes_on_valid_meshes():
    meshes = [voxel_mesh((3, 2, 2)), read(DATA / "gmsh_pair.msh"), read(DATA / "gmsh_messy.msh")]
    for mesh in meshes:
        check_elements(mesh)
        check_elements(mesh, mass="lumped")


@pytest.mark.parametrize("name", TYPES)
def test_check_raises_on_mirrored_element(name):
    X = unit(name) * [1, 1, -1]
    assert np.asarray(jacobian(name, X)[1]).max() < 0
    with pytest.raises(ValueError, match=rf"element 0 \({name}\): det J"):
        check_elements(one_mesh({name: [X]}))


def test_check_raises_on_flat_element():
    X = unit("hex8") * [1, 1, 0]  # zero thickness: det J = 0 exactly
    with pytest.raises(ValueError, match=r"det J = 0 <= 0"):
        check_elements(one_mesh({"hex8": [X]}))


def test_check_raises_on_pushed_through_corner():
    X = unit("hex8")
    X[6] = (-0.2, -0.2, -0.2)
    assert (np.asarray(jacobian("hex8", X)[1]) < 0).sum() == 4
    with pytest.raises(ValueError, match="det J"):
        check_elements(one_mesh({"hex8": [X]}))


def test_check_catches_quadrature_only_failure():
    X = unit("hex20")
    X[8, 0] = 0.95  # midside of edge 0-1 pulled next to corner 1
    mesh = one_mesh({"hex20": [X]})
    assert corner_jacobians(mesh)["hex20"].min() > 0  # PR 1's corner check passes
    with pytest.raises(ValueError, match="det J"):
        check_elements(mesh)


def test_check_reports_global_index_across_blocks_and_chunks():
    bad = unit("hex20")
    bad[8, 0] = 0.95
    mesh = one_mesh({"hex8": [unit("hex8")] * 2, "hex20": [unit("hex20")] * 3 + [bad]})
    with pytest.raises(ValueError, match=r"element 5 \(hex20\)"):
        check_elements(mesh, chunk=2)


def test_check_rejects_unknown_mass():
    with pytest.raises(ValueError, match="mass must be"):
        check_elements(voxel_mesh((1, 1, 1)), mass="lumpy")


# ---------------------------------------------------------------- 10-11: forms, batching, dtype


@pytest.mark.parametrize("name", TYPES)
def test_forms_equal_matrix_products(name):
    rng = np.random.default_rng(11)
    X = np.stack(distorted(name, rng, 3))
    u, w = rng.standard_normal((2, 3, 3 * NODES[name]))
    K, M = np.asarray(stiffness(name, X, C_ORT)), np.asarray(mass(name, X, 2.5))
    assert rel(stiffness_form(name, X, C_ORT, u, w), np.einsum("ea,eab,eb->e", u, K, w)) <= 1e-12
    assert rel(mass_form(name, X, 2.5, u, w), np.einsum("ea,eab,eb->e", u, M, w)) <= 1e-12


@pytest.mark.parametrize("name", TYPES)
def test_batching_jit_symmetry_dtype(name):
    rng = np.random.default_rng(12)
    X = np.stack(distorted(name, rng, 6)).reshape(2, 3, NODES[name], 3)
    C = np.stack([C_ISO, C_ORT, 2 * C_ISO, 2 * C_ORT, 3 * C_ISO, 3 * C_ORT]).reshape(2, 3, 6, 6)
    rho = rng.uniform(1, 3, (2, 3))
    K, M, ml = stiffness(name, X, C), mass(name, X, rho), lumped_mass(name, X, rho)
    for i, j in np.ndindex(2, 3):
        assert rel(K[i, j], stiffness(name, X[i, j], C[i, j])) <= 1e-14
        assert rel(M[i, j], mass(name, X[i, j], rho[i, j])) <= 1e-14
        assert rel(ml[i, j], lumped_mass(name, X[i, j], rho[i, j])) <= 1e-14
    assert rel(jax.jit(stiffness, static_argnums=0)(name, X, C), K) <= 1e-14
    assert (K == jnp.swapaxes(K, -1, -2)).all() and (M == jnp.swapaxes(M, -1, -2)).all()
    B = bmatrix(gradients(name, jacobian(name, X)[0]))
    assert B.shape == (2, 3, ORDER[name] ** 3, 6, 3 * NODES[name])
    f32 = [a.astype(np.float32) for a in (X[0, 0], C[0, 0], rho[0, 0])]
    u = np.ones(3 * NODES[name], np.float32)
    for out in (
        stiffness(name, f32[0], f32[1]),
        mass(name, f32[0], f32[2]),
        lumped_mass(name, f32[0], f32[2]),
        stiffness_form(name, f32[0], f32[1], u, u),
        mass_form(name, f32[0], f32[2], u, u),
        *jacobian(name, f32[0]),
    ):
        assert out.dtype == jnp.float32


# ---------------------------------------------------------------- 12: faces


@pytest.mark.parametrize("name", TYPES)
def test_faces(name):
    el = ELEMENTS[name]
    edges = {frozenset(e) for e in EDGES}
    assert [f[:4] for f in el.faces] == [
        (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7)
    ]  # fmt: skip
    for f, nodes in enumerate(el.faces):
        assert len(nodes) == {8: 4, 20: 8, 27: 9}[NODES[name]]
        d, s = f // 2, 2 * (f % 2) - 1
        x = VTK[list(nodes)]
        assert (x[:, d] == s).all()
        c = nodes[:4]
        assert all(frozenset(p) in edges for p in zip(c, c[1:] + c[:1], strict=True))
        p = VTK[list(c)]
        assert np.cross(p[2] - p[0], p[3] - p[1])[d] * s > 0  # outward
        for k in range(4 * (len(nodes) > 4)):
            assert np.array_equal(x[4 + k], (p[k] + p[(k + 1) % 4]) / 2)
        if len(nodes) == 9:
            assert nodes[8] == 20 + f
    small = {"hex20": "hex8", "hex27": "hex20"}.get(name)
    if small:
        k = len(ELEMENTS[small].faces[0])
        assert all(a[:k] == b for a, b in zip(el.faces, ELEMENTS[small].faces, strict=True))
