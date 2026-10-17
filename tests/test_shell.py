"""PR 16a: the ssh8 solid shell (KGW 1999), laminates, solid_shell conversion, assemble.

References: tests/shell_ref.py (independent NumPy element), hex8 kernels, SciPy assembly and
solves, analytic patch fields, MacNeal-Harder (1985) beam references.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spla
import shell_ref as ref
from helpers import ORTHO, R, fd_rel, rel
from scipy.spatial.transform import Rotation

from jaxmodal.elements import check_elements, kernels, shell_mass, shell_recover, shell_stiffness
from jaxmodal.elements.check import _thickness_min
from jaxmodal.materials import (
    Laminate,
    Material,
    Ply,
    bond_strain,
    bond_stress,
    check_materials,
    from_vectors,
    isotropic,
    ply_table,
    rotate,
)
from jaxmodal.memory import amg_memory, unstructured_memory
from jaxmodal.mesh import (
    Mesh,
    box_mesh,
    corner_jacobians,
    parts,
    read,
    solid_shell,
    voxel_mesh,
    write,
)
from jaxmodal.mesh.model import CORNERS
from jaxmodal.operators import assemble, dof_mask, rigid_modes, stencil, unstructured

C_ANI = np.asarray(rotate(ORTHO, R))  # fully anisotropic
C_ORT = np.asarray(ORTHO)
C_ISO = np.asarray(isotropic(1e7, 0.3))
I3 = np.eye(3)
_K, _M, _REC = (jax.jit(f) for f in (shell_stiffness, shell_mass, shell_recover))


def laminate(plies):
    return Laminate(tuple(Ply(Material(C, rho), a, f) for C, rho, a, f in plies))


def table(mat, R_o=I3):
    P, Ro = ply_table([(0, 0)], {0: mat}, {0: R_o})
    return jax.tree.map(lambda a: a[0], P), Ro[0]


def K(X, plies, R_o=I3):
    return np.asarray(_K(jnp.asarray(X), table(laminate(plies), R_o)))


def m(X, plies):
    return np.asarray(_M(jnp.asarray(X), table(laminate(plies))[0]))


def node_perm(Q):
    """New node i is the old node at Q^T CORNERS[i] (xi_new = Q xi_old)."""
    return np.array([np.flatnonzero((CORNERS == c).all(1))[0] for c in CORNERS @ Q])


def dofs(p):
    return (3 * np.asarray(p)[:, None] + np.arange(3)).ravel()


def three_plies(C, rho=(2.0, 1.0, 3.0)):
    return [(C, rho[0], 0.3, 0.2), (C, rho[1], -0.7, 0.5), (C, rho[2], 1.1, 0.3)]


def prismatic(rng, t=0.2):
    bottom = ref.VTK[:4] / 2 + np.c_[rng.uniform(-0.15, 0.15, (4, 2)), np.zeros(4)]
    d = np.array([0.1, -0.05, t])
    return np.vstack([bottom, bottom + d]) @ Rotation.random(random_state=rng).as_matrix().T


def normal(X):
    J0 = ref.jac(X, np.zeros(3))
    n = np.cross(J0[:, 0], J0[:, 1])
    return n / np.linalg.norm(n), J0


# ---------------------------------------------------------------- 1: independent reference


STACKS = {  # a different C per ply
    1: [(C_ANI, 2.0, 0.3, 1.0)],
    2: [(C_ANI, 2.0, 0.3, 0.4), (C_ORT, 1.0, -0.7, 0.6)],
    3: [
        (C_ANI, 2.0, 0.3, 0.2),
        (C_ORT, 1.0, -0.7, 0.5),
        (np.asarray(rotate(C_ANI, ref.rz(1.0))), 3.0, 1.1, 0.3),
    ],
}


@pytest.mark.parametrize("ht", [1, 2, 20])
@pytest.mark.parametrize("n_ply", [1, 2, 3])
def test_kernels_match_reference(ht, n_ply):
    rng = np.random.default_rng(10 * ht + n_ply)
    plies = STACKS[n_ply]
    X = ref.distorted_hex8(rng, ht, plies)
    Ke, Kr = K(X, plies, R), ref.stiffness(X, plies, R)
    assert rel(Ke, Kr) <= 1e-12
    assert np.array_equal(Ke, Ke.T)
    assert rel(m(X, plies), ref.mass(X, plies)) <= 1e-12
    u = rng.standard_normal(24)
    out = _REC(jnp.asarray(X), table(laminate(plies), R), jnp.asarray(u))
    *expect, alpha = ref.recover(X, plies, u, R)
    assert np.linalg.norm(alpha) >= 1e-3 * np.linalg.norm(u)  # linear fields give alpha = 0
    for a, b in zip(out, expect, strict=True):
        assert rel(a, b) <= 1e-12
    _, w, eps, sig = (np.asarray(a) for a in out)
    assert abs(u @ Ke @ u - (w * (eps * sig).sum(1)).sum()) <= 1e-12 * abs(u @ Ke @ u)


# ---------------------------------------------------------------- 2: laminate path


@pytest.mark.parametrize("branch", ["primary", "fallback"])
def test_connectivity_rotations_and_flip(branch):
    rng = np.random.default_rng(3)
    plies = three_plies(C_ORT)
    X = ref.distorted_hex8(rng, 5, plies)
    n, J0 = normal(X)
    if branch == "primary":
        R_o = np.asarray(from_vectors(J0[:, 0], J0[:, 1]))
    else:  # axis 1 near the normal (projection 0.05 < 0.1)
        R_o = np.asarray(from_vectors(n + 0.05 * J0[:, 0] / np.linalg.norm(J0[:, 0]), J0[:, 1]))
    assert (np.linalg.norm(R_o[:, 0] - (R_o[:, 0] @ n) * n) >= 0.1) == (branch == "primary")
    Ke = K(X, plies, R_o)
    assert rel(Ke, ref.stiffness(X, plies, R_o)) <= 1e-12
    for k in (1, 2, 3):  # in-plane connectivity rotations
        p = node_perm(np.rint(ref.rz(k * np.pi / 2)).astype(int))
        assert rel(K(X[p], plies, R_o), Ke[np.ix_(dofs(p), dofs(p))]) <= 1e-12
    p = node_perm(np.diag([1, -1, -1]))  # zeta flip: plies reversed, angles negated
    flipped = [(C, rho, -a, f) for C, rho, a, f in plies[::-1]]
    assert rel(K(X[p], flipped, R_o), Ke[np.ix_(dofs(p), dofs(p))]) <= 1e-12


@pytest.mark.parametrize("proj", [0.095, 0.105])
def test_frame_threshold(proj):
    """R_o axis 1 projected onto the shell plane just below / above 0.1: fallback / primary."""
    rng = np.random.default_rng(13)
    X = ref.distorted_hex8(rng, 4, STACKS[3])
    n, J0 = normal(X)
    t = J0[:, 0] - (J0[:, 0] @ n) * n
    a = np.sqrt(1 - proj**2) * n + proj * t / np.linalg.norm(t)
    R_o = np.asarray(from_vectors(a, J0[:, 1]))
    assert rel(K(X, STACKS[3], R_o), ref.stiffness(X, STACKS[3], R_o)) <= 1e-12


def test_rigid_rotation():
    rng = np.random.default_rng(4)
    plies = three_plies(C_ANI)
    X = ref.distorted_hex8(rng, 3, plies)
    Q = Rotation.random(random_state=rng).as_matrix()
    B = np.kron(np.eye(8), Q)
    assert rel(K(X @ Q.T, plies, Q @ R), B @ K(X, plies, R) @ B.T) <= 1e-12


def test_ply_angle_and_identical_plies():
    rng = np.random.default_rng(5)
    X = ref.distorted_hex8(rng, 4)
    C_th = np.asarray(rotate(C_ANI, ref.rz(0.6)))
    assert rel(K(X, [(C_ANI, 1.0, 0.6, 1.0)]), K(X, [(C_th, 1.0, 0.0, 1.0)])) <= 1e-12
    Xp = prismatic(rng)
    same = [(C_ANI, 1.0, 0.4, f) for f in (0.3, 0.5, 0.2)]
    assert rel(K(Xp, same), K(Xp, [(C_ANI, 1.0, 0.4, 1.0)])) <= 1e-12


def test_one_ply_material_and_flat_orientation():
    rng = np.random.default_rng(6)
    X = ref.distorted_hex8(rng, 2)
    mat = Material(C_ANI, 1.5)
    lam = Laminate((Ply(mat, 0.0, 1.0),))
    a, b = (np.asarray(_K(jnp.asarray(X), table(x, R))) for x in (lam, mat))
    assert np.array_equal(a, b)
    bottom = np.c_[ref.VTK[:4, :2] / 2 + rng.uniform(-0.15, 0.15, (4, 2)), np.zeros(4)]
    Xf = np.vstack([bottom, bottom + [0, 0, 0.1]])  # flat in xy, zeta along +z
    Rz = ref.rz(0.8)
    flat = np.asarray(_K(jnp.asarray(Xf), table(mat, Rz)))
    rotated = np.asarray(_K(jnp.asarray(Xf), table(Material(rotate(C_ANI, Rz), 1.5))))
    assert rel(flat, rotated) <= 1e-12


# ---------------------------------------------------------------- 3: mass


def test_mass():
    rng = np.random.default_rng(7)
    plies = three_plies(C_ANI)
    X = ref.distorted_hex8(rng, 3, plies)
    hex_m = np.asarray(kernels.mass("hex8", jnp.asarray(X), 2.5))[::3, ::3]
    assert rel(m(X, [(C_ANI, 2.5, 0.3, 1.0)]), hex_m) <= 1e-14
    Xp = prismatic(rng)
    hex_p = np.asarray(kernels.mass("hex8", jnp.asarray(Xp), 2.5))[::3, ::3]
    assert rel(m(Xp, three_plies(C_ANI, (2.5,) * 3)), hex_p) <= 1e-14
    # per ply: hex8 mass of the ply sub-element, mapped back by the trilinear interpolation P
    total, z0, Vs = np.zeros((8, 8)), -1.0, []
    for _, rho, _, f in plies:
        sub = [np.r_[c[:2], z0 + f * (1 + c[2])] for c in ref.VTK]
        P = np.array([ref.shape(s)[0] for s in sub])
        m_sub = np.asarray(kernels.mass("hex8", jnp.asarray(P @ X), rho))[::3, ::3]
        total += P.T @ m_sub @ P
        g, w = np.polynomial.legendre.leggauss(3)  # exact volume: det J is quadratic per axis
        Vs.append(
            sum(
                wa * wb * wc * f * np.linalg.det(ref.jac(X, [a, b, z0 + f * (1 + c)]))
                for a, wa in zip(g, w, strict=True)
                for b, wb in zip(g, w, strict=True)
                for c, wc in zip(g, w, strict=True)
            )
        )
        z0 += 2 * f
    me = m(X, plies)
    assert rel(me, total) <= 1e-14
    assert abs(me.sum() - sum(p[1] * V for p, V in zip(plies, Vs, strict=True))) <= 1e-14 * me.sum()


# ---------------------------------------------------------------- 4: zero modes


@pytest.mark.parametrize("ht", [1, 2, 100])
@pytest.mark.parametrize("shape", ["regular", "distorted"])
def test_zero_modes(ht, shape):
    rng = np.random.default_rng(8)
    plies = three_plies(C_ANI)
    if shape == "regular":
        X = (ref.VTK / 2) * [1, 1, 1 / ht]
    else:
        X = ref.distorted_hex8(rng, ht, plies)
    lam = np.linalg.eigvalsh(K(X, plies))
    print(f"h/t {ht} {shape}: lambda_7 / lambda_max = {lam[6] / lam[-1]:.3g}")
    if ht <= 2:
        assert (np.abs(lam[:6]) <= 1e-10 * lam[-1]).all() and lam[6] >= 1e-3 * lam[-1]


# ---------------------------------------------------------------- 5: patch tests

MH = np.array([[0, 0], [0.24, 0], [0.24, 0.12], [0, 0.12],
               [0.04, 0.02], [0.18, 0.03], [0.16, 0.08], [0.08, 0.08]])  # fmt: skip
MH_QUADS = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]
T = 0.01
TILT = np.asarray(Rotation.from_euler("zxz", [0.4, 0.9, -0.3]).as_matrix())


def plate(xy, quads, inner, t=T, lift=0.0, Q=I3):
    """hex8 plate of the quads (two node layers z = 0, t), inner nodes lifted, rotated by Q."""
    n = len(xy)
    z = np.where(np.isin(np.arange(n), inner), lift, 0.0)
    pts = np.vstack([np.c_[xy, z], np.c_[xy, z + t]]) @ Q.T
    q = np.asarray(quads)
    conn = np.c_[q, q + n].astype(np.int64)
    e = len(conn)
    mesh = Mesh(pts, {"hex8": conn}, {}, {}, np.zeros(e, np.int64), np.zeros(e, np.int64))
    return mesh, np.r_[inner, np.asarray(inner) + n]


def grid_patch():
    i, j = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
    xy = np.c_[(i + 0.3 * j).ravel(), j.ravel()] * 0.08  # the MacNeal-Harder patch size
    quads = [[a + 4 * b, a + 1 + 4 * b, a + 5 + 4 * b, a + 4 + 4 * b]
             for b in range(3) for a in range(3)]  # fmt: skip
    return xy, quads, [5, 6, 9, 10]


PATCHES = {
    "i": (grid_patch(), np.eye(3), 0.0),
    "ii": ((MH, MH_QUADS, [4, 5, 6, 7]), np.eye(3), 0.0),
    "iii": ((MH, MH_QUADS, [4, 5, 6, 7]), TILT, 0.0),
    "iv": ((MH, MH_QUADS, [4, 5, 6, 7]), np.eye(3), 0.1 * T),
}


def linear_field(C, Q, s, points):
    """u = A x for the unit stress state s (in the frame Q): (u, global sigma)."""
    sig = np.eye(6)[s]
    eps = np.asarray(bond_strain(Q)) @ np.linalg.solve(C, sig)
    A = np.zeros((3, 3))
    for r, (i, j) in enumerate(ref.PAIRS):
        A[i, j] = A[j, i] = eps[r] if i == j else eps[r] / 2
    return (points @ A.T).ravel(), np.asarray(bond_stress(Q)) @ sig


def patch(name):
    (xy, quads, inner), Q, lift = PATCHES[name]
    hexes, interior = plate(xy, quads, inner, lift=lift, Q=Q)
    shells, _ = solid_shell(hexes, direction=Q[:, 2])
    C = C_ISO if name == "iv" else C_ANI
    return hexes, shells, interior, Q, C


def lift(Kh, Ks, inner, uA):
    """u = A x on the boundary; interior surface DOFs solved under the consistent traction."""
    d_I = dofs(inner)
    d_B = np.setdiff1d(np.arange(Ks.shape[0]), d_I)
    u = uA.copy()
    u[d_I] = spla.spsolve(Ks[d_I][:, d_I].tocsc(), (Kh @ uA)[d_I] - Ks[d_I][:, d_B] @ uA[d_B])
    return u


def recover(mesh, mat, R_o, u):
    conn = mesh.blocks["ssh8"]
    f = jax.jit(jax.vmap(shell_recover, in_axes=(0, None, 0)))
    U = u.reshape(-1, 3)[conn].reshape(len(conn), 24)
    return [np.asarray(a) for a in f(mesh.points[conn], table(mat, R_o), U)]


@pytest.mark.parametrize("name", ["i", "ii", "iii"])
def test_patch_lift(name):
    hexes, shells, inner, Q, C = patch(name)
    mat = Material(C, 1.0)
    Kh = assemble(hexes, {0: Material(rotate(C, Q), 1.0)})[0]
    Ks = assemble(shells, {0: mat}, {0: Q})[0]
    for s in range(6):
        uA, sig = linear_field(C, Q, s, hexes.points)
        u = lift(Kh, Ks, inner, uA)
        err_u = np.abs(u - uA).max() / np.abs(uA).max()
        err_s = np.abs(recover(shells, mat, Q, u)[3] - sig).max() / np.abs(sig).max()
        if name == "i" or s in (0, 1, 2, 5):
            assert err_u <= 1e-10 and err_s <= 1e-10, (s, err_u, err_s)
        else:
            print(f"patch {name} transverse shear state {s}: errors {err_u:.2g}, {err_s:.2g}")


@pytest.mark.parametrize("name", ["i", "ii", "iii", "iv"])
def test_patch_recovery(name):
    hexes, shells, _, Q, C = patch(name)
    Q = np.eye(3) if name == "iv" else Q  # iv: isotropic, global states
    mat = Material(C, 1.0)
    conn = shells.blocks["ssh8"]
    for s in range(6):
        uA, sig = linear_field(C, Q, s, hexes.points)
        _, w, eps, sigma = recover(shells, mat, Q, uA)
        assert np.abs(sigma - sig).max() <= 1e-10 * np.abs(sig).max()
        assert np.abs(sigma - eps @ np.asarray(rotate(C, Q)).T).max() <= 1e-12 * np.abs(sig).max()
    gauss = list(itertools.product((-ref.G, ref.G), repeat=3))
    vol = [sum(np.linalg.det(ref.jac(shells.points[c], np.array(p))) for p in gauss) for c in conn]
    assert rel(w.reshape(len(conn), -1).sum(1), vol) <= 1e-12


def test_patch_reported():
    """Reported only: the lift on the lifted (iv) and warped patches."""
    for name, warp in (("iv", None), ("ii", 0.1 * T)):
        hexes, shells, inner, _, C = patch(name)
        if warp is not None:
            pts = hexes.points.copy()
            pts[[4, 4 + len(MH)], 2] += warp
            hexes = Mesh(pts, hexes.blocks, {}, {}, hexes.material, hexes.orientation)
            shells, _ = solid_shell(hexes, direction=(0, 0, 1))
        Kh = assemble(hexes, {0: Material(C, 1.0)})[0]
        Ks = assemble(shells, {0: Material(C, 1.0)})[0]
        for s in range(6):
            uA, sig = linear_field(C, I3, s, hexes.points)
            u = lift(Kh, Ks, inner, uA)
            err = np.abs(recover(shells, Material(C, 1.0), np.eye(3), u)[3] - sig).max()
            print(f"lift {name} warp {warp}: state {s} stress error {err / np.abs(sig).max():.2g}")


# ---------------------------------------------------------------- 6: MacNeal-Harder beam


def mh_beam(kind):
    """6 x 0.2 x 0.1 beam, 6 x 1 x 1 hex8; interior stations skewed by +-0.1 at y = 0 / 0.2."""
    s = {"regular": [0] * 7, "parallelogram": [0] + [1] * 5 + [0],
         "trapezoid": [0] + [(-1) ** i for i in range(1, 6)] + [0]}[kind]  # fmt: skip
    pts = np.array([[i - 0.1 * s[i] * (1 - 2 * y), 0.2 * y, 0.1 * z]
                    for z in (0, 1) for y in (0, 1) for i in range(7)])  # fmt: skip
    nid = lambda i, y, z: i + 7 * (y + 2 * z)  # noqa: E731
    conn = np.array([[nid(i + a, b, c) for a, b, c in
                      [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]] for i in range(6)])  # fmt: skip
    z = np.zeros(6, np.int64)
    mesh = Mesh(pts, {"hex8": conn}, {}, {}, z, z)
    return solid_shell(mesh, direction=(0, 0, 1))[0]


def mh_solve(mesh, f):
    Kg = assemble(mesh, {0: Material(C_ISO, 1.0)})[0]
    free = np.repeat(mesh.points[:, 0] > 1e-9, 3)
    u = np.zeros(len(f))
    u[free] = spla.spsolve(Kg[free][:, free].tocsc(), f[free])
    return u


@pytest.mark.parametrize("kind", ["regular", "parallelogram", "trapezoid"])
def test_macneal_harder_beam(kind):
    mesh = mh_beam(kind)
    tip = np.flatnonzero(np.abs(mesh.points[:, 0] - 6) < 1e-9)
    refs = {"extension": (0, 3.0e-5), "in-plane": (1, 0.1081), "out-of-plane": (2, 0.4321)}
    gates = {"regular": {"extension": 0.95, "in-plane": 0.95, "out-of-plane": 0.95},
             "parallelogram": {"extension": 0.9, "out-of-plane": 0.9}}.get(kind, {})  # fmt: skip
    for load, (d, value) in refs.items():
        f = np.zeros(3 * len(mesh.points))
        f[3 * tip + d] = 1 / len(tip)
        ratio = mh_solve(mesh, f)[3 * tip + d].mean() / value
        print(f"MacNeal-Harder {kind} {load}: {ratio:.4f}")
        if load in gates:
            assert ratio >= gates[load], (load, ratio)
    r = mesh.points[tip][:, 1:] - [0.1, 0.05]  # twist: unit couple about x (reported)
    f = np.zeros(3 * len(mesh.points))
    f[3 * tip + 1], f[3 * tip + 2] = -r[:, 1] / (r**2).sum(), r[:, 0] / (r**2).sum()
    u = mh_solve(mesh, f)
    twist = ((r[:, 0] * u[3 * tip + 2] - r[:, 1] * u[3 * tip + 1]) / (r**2).sum(1)).mean()
    print(f"MacNeal-Harder {kind} twist: {twist / 0.03208:.4f}")


# ---------------------------------------------------------------- 7: assemble


def mixed():
    box = box_mesh((4, 2, 2), spacing=(1.0, 1.0, 0.1))
    c = box.points[box.blocks["hex8"]].mean(1)
    sel = np.flatnonzero(c[:, 0] > 2)
    material = np.where(c[:, 0] > 2, np.where(c[:, 1] > 1, 2, 1), 0)
    box = Mesh(box.points, box.blocks, box.node_sets, box.element_sets, material, box.orientation)
    mats = {
        0: Material(C_ANI, 1.3),
        1: laminate([(C_ORT, 2.0, 0.5, 0.4), (C_ANI, 1.0, -0.5, 0.6)]),
        2: laminate(three_plies(C_ORT)),
    }
    return solid_shell(box, sel, direction=(0, 0, 1))[0], mats


def test_assemble_mixed():
    mesh, mats = mixed()
    Kg, Mg = (A.toarray() for A in assemble(mesh, mats))
    n = Kg.shape[0]
    Kr, Mr, e = np.zeros((n, n)), np.zeros((n, n)), 0
    for t, conn in mesh.blocks.items():
        for row in conn:
            mat, X, d = mats[mesh.material[e]], mesh.points[row], dofs(row)
            if t == "hex8":
                Kr[np.ix_(d, d)] += np.asarray(kernels.stiffness(t, jnp.asarray(X), mat.C))
                Mr[np.ix_(d, d)] += np.asarray(kernels.mass(t, jnp.asarray(X), mat.rho))
            else:  # unpadded per-element table
                Kr[np.ix_(d, d)] += np.asarray(_K(jnp.asarray(X), table(mat)))
                Mr[np.ix_(d, d)] += np.kron(np.asarray(_M(jnp.asarray(X), table(mat)[0])),
                                            np.eye(3))  # fmt: skip
            e += 1
    assert rel(Kg, Kr) <= 1e-12 and rel(Mg, Mr) <= 1e-12
    assert np.array_equal(Kg, Kg.T) and np.linalg.eigvalsh(Kg)[0] >= -1e-12 * np.abs(Kg).max()
    Rm = rigid_modes(mesh)
    assert np.linalg.norm(Kg @ Rm) <= 1e-12 * np.linalg.norm(Kg) * np.linalg.norm(Rm)
    vol = 0.1  # every element is 1 x 1 x 0.1
    mass = sum(mats[k].rho if k == 0 else sum(p.material.rho * p.fraction for p in mats[k].plies)
               for k in mesh.material) * vol  # fmt: skip
    ones = np.zeros(n)
    ones[::3] = 1
    assert abs(ones @ Mg @ ones - mass) <= 1e-12 * mass
    assert (parts(mesh) == 0).all()
    free = dof_mask(mesh, {"xmin": "xyz"})
    assert np.array_equal(~free, np.repeat(mesh.points[:, 0] == 0, 3))


def test_one_ply_shell_mass_equals_hex8():
    box = box_mesh((3, 2, 1), spacing=(1.0, 0.7, 0.1))
    mats = {0: Material(C_ANI, 2.0)}
    Mh = assemble(box, mats)[1]
    Ms = assemble(solid_shell(box, direction=(0, 0, 1))[0], mats)[1]
    assert rel(Ms.toarray(), Mh.toarray()) <= 1e-14


# ---------------------------------------------------------------- 8: gradients


def test_gradients():
    rng = np.random.default_rng(11)
    plies = three_plies(C_ANI)
    X = ref.distorted_hex8(rng, 3, plies)
    P, R_o = table(laminate(plies), R)
    u, v = rng.standard_normal((2, 24))
    a, b = rng.standard_normal((2, 8))

    def k(X=X, C=P.material.C, angle=P.angle, fraction=P.fraction):
        p = P._replace(material=P.material._replace(C=C), angle=angle, fraction=fraction)
        return u @ shell_stiffness(X, (p, R_o)) @ v

    def mm(X=X, rho=P.material.rho):
        return a @ shell_mass(X, P._replace(material=P.material._replace(rho=rho))) @ b

    cases = [(k, "X", X), (k, "C", P.material.C), (k, "angle", P.angle),
             (k, "fraction", P.fraction), (mm, "X", X), (mm, "rho", P.material.rho)]  # fmt: skip
    for f, name, x in cases:
        g = jax.jit(jax.grad(lambda y, f=f, name=name: f(**{name: y})))
        fx = jax.jit(lambda y, f=f, name=name: f(**{name: y}))
        x = np.asarray(x, float)
        assert fd_rel(lambda y, fx=fx: float(fx(y)), x, np.asarray(g(x))) < 1e-6, name


def test_gradients_finite_at_edge_cases():
    """No NaN in any gradient (X, plies, R_o): padded plies, R_o axis 1 or 2 exactly normal."""
    rng = np.random.default_rng(12)
    X = ref.distorted_hex8(rng, 3)
    n, J0 = normal(X)
    P, _ = ply_table([(0, 0), (1, 0)], {0: Material(C_ANI, 1.0), 1: laminate(three_plies(C_ORT))})
    padded = jax.tree.map(lambda a: a[0], P)  # 1 ply + 2 padded
    t = J0[:, 0] / np.linalg.norm(J0[:, 0])
    axis2 = np.column_stack([t, n, np.cross(t, n)])  # axis 2 exactly normal
    u = rng.standard_normal(24)
    fs = (lambda X, p, R_o: jnp.sum(shell_stiffness(X, (p, R_o))),
          lambda X, p, R_o: jnp.sum(shell_mass(X, p)),
          lambda X, p, R_o: sum(jnp.sum(a) for a in shell_recover(X, (p, R_o), u)))  # fmt: skip
    for R_o in (np.eye(3), np.asarray(from_vectors(n, J0[:, 0])), axis2):  # axis 1 normal: 2nd
        for f in fs:
            g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(X), padded, jnp.asarray(R_o))
            assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(g))


# ---------------------------------------------------------------- 9: solid_shell


def zeta_edges(mesh):
    X = mesh.points[mesh.blocks["ssh8"]]
    return (X[:, 4:] - X[:, :4]).mean(1)


def proper_rotations():
    out = []
    for p in [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]:
        for s in np.ndindex(2, 2, 2):
            Q = np.eye(3, dtype=int)[list(p)] * (1 - 2 * np.array(s))[:, None]
            if round(np.linalg.det(Q)) == 1:
                out.append(Q)
    return out


def test_solid_shell_sweep():
    box = box_mesh((8, 6, 2))
    rots = proper_rotations()
    assert len(rots) == 24
    for Q in rots:
        p = node_perm(Q)
        m_ = Mesh(box.points, {"hex8": box.blocks["hex8"][:, p]}, {}, {}, box.material,
                  box.orientation)  # fmt: skip
        out = solid_shell(m_)[0]
        assert np.allclose(zeta_edges(out), [0, 0, 1]) and (corner_jacobians(out)["ssh8"] > 0).all()
    for shape in [(3, 3, 3), (6, 1, 1), (8, 8, 12)]:
        with pytest.raises(ValueError, match="ambiguous thickness direction: pass direction="):
            solid_shell(box_mesh(shape))
    for shape, d in (((8, 10, 12), 0), ((10, 8, 12), 1)):  # x and y thickness axes
        out = solid_shell(box_mesh(shape))[0]
        assert (
            np.allclose(zeta_edges(out), np.eye(3)[d]) and (corner_jacobians(out)["ssh8"] > 0).all()
        )


def test_solid_shell_direction_and_sign():
    Q = TILT
    box = box_mesh((6, 4, 1), spacing=(1.0, 1.0, 0.1))
    tilted = Mesh(box.points @ Q.T, box.blocks, {}, {}, box.material, box.orientation)
    e = zeta_edges(solid_shell(tilted, direction=Q[:, 2])[0])
    assert np.allclose(e, 0.1 * Q[:, 2])
    e = zeta_edges(solid_shell(tilted, direction=-Q[:, 2])[0])
    assert np.allclose(e, -0.1 * Q[:, 2])
    Rx = Rotation.from_rotvec([-3 * np.pi / 4, 0, 0]).as_matrix()  # normal (0, 1, -1) / sqrt 2
    box = box_mesh((8, 6, 2))
    jitter = np.random.default_rng(14).uniform(-1e-12, 1e-12, box.points.shape)  # splits the tie
    plate_ = Mesh(box.points @ Rx.T + jitter, box.blocks, {}, {}, box.material, box.orientation)
    e = zeta_edges(solid_shell(plate_)[0])
    assert np.allclose(np.abs(e), [0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert (e @ e[0] > 0).all()  # one sign throughout


def ring(n_t, n_r, n_z, r0=2.0):
    """hex8 cylinder ring: n_t around (closed), n_r radial, n_z axial."""
    t, r, z = np.meshgrid(np.arange(n_t) * 2 * np.pi / n_t, r0 + np.arange(n_r + 1) * 0.1,
                          np.arange(n_z + 1) * 0.3, indexing="ij")  # fmt: skip
    pts = np.c_[(r * np.cos(t)).ravel(), (r * np.sin(t)).ravel(), z.ravel()]
    nid = lambda i, j, k: ((i % n_t) * (n_r + 1) + j) * (n_z + 1) + k  # noqa: E731
    corners = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0),
               (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]  # fmt: skip
    conn = np.array([[nid(i + a, j + b, k + c) for a, b, c in corners]
                     for i in range(n_t) for j in range(n_r) for k in range(n_z)])  # fmt: skip
    z0 = np.zeros(len(conn), np.int64)
    mesh = Mesh(pts, {"hex8": conn}, {}, {}, z0, z0)
    assert (corner_jacobians(mesh)["hex8"] > 0).all()
    return mesh


def test_solid_shell_ring():
    mesh = ring(4, 5, 6)  # the closed chain (count 4) is the shortest: zeta circumferential
    out = solid_shell(mesh)[0]
    c = out.points[out.blocks["ssh8"]].mean(1)
    e = zeta_edges(out)
    assert np.allclose(e[:, 2], 0) and np.allclose((e[:, :2] * c[:, :2]).sum(1), 0)
    assert (corner_jacobians(out)["ssh8"] > 0).all()
    mesh = ring(16, 1, 2)  # radial is thinnest; per-element radial directions
    c = mesh.points[mesh.blocks["hex8"]].mean(1)
    radial = c * [1, 1, 0]
    out = solid_shell(mesh, direction=radial)[0]
    e = zeta_edges(out)
    assert ((e * radial).sum(1) > 0.09 * np.linalg.norm(radial, axis=1)).all()  # chord 0.098
    assert (corner_jacobians(out)["ssh8"] > 0).all()
    swept = solid_shell(mesh)[0]  # the sweep finds radial too (the sign may flip on the ring)
    e = zeta_edges(swept)
    assert swept.blocks["ssh8"].shape == (32, 8)
    assert (np.abs((e * radial).sum(1)) > 0.09 * np.linalg.norm(radial, axis=1)).all()
    assert (corner_jacobians(swept)["ssh8"] > 0).all()


def test_solid_shell_blocks_sets_and_errors():
    box = box_mesh((4, 3, 1), spacing=(1.0, 1.0, 0.1))
    n = box.n_el
    sets = {"left": np.arange(0, n, 4), "right": np.arange(3, n, 4)}
    mat = np.arange(n) % 3
    box = Mesh(box.points, box.blocks, box.node_sets, sets, mat, np.arange(n) % 2)
    once, o1 = solid_shell(box, "left", direction=(0, 0, 1))
    assert list(once.blocks) == ["hex8", "ssh8"] and len(once.blocks["ssh8"]) == 3
    assert np.array_equal(once.material, mat[o1]) and np.array_equal(once.orientation, o1 % 2)
    for k, v in sets.items():
        assert np.array_equal(np.sort(o1[once.element_sets[k]]), v)
    assert once.voxel is None and (corner_jacobians(once)["ssh8"] > 0).all()
    twice, o2 = solid_shell(once, np.arange(once.n_el)[once.offsets["hex8"]:][:9],
                            direction=(0, 0, -1))  # fmt: skip
    assert list(twice.blocks) == ["ssh8"] and len(twice.blocks["ssh8"]) == n
    assert np.array_equal(twice.material, mat[o1[o2]])
    assert (corner_jacobians(twice)["ssh8"] > 0).all()
    e = zeta_edges(twice)
    assert np.allclose(e[:3], [0, 0, 0.1]) and np.allclose(e[3:], [0, 0, -0.1])
    with pytest.raises(ValueError, match="empty element selection"):
        solid_shell(box, np.array([], np.int64))
    for bad in ([-1], [n]):
        with pytest.raises(ValueError, match="hex8 elements only"):
            solid_shell(box, bad)
    dup, _ = solid_shell(box, [2, 0, 2], direction=(0, 0, 1))  # duplicates and order ignored
    assert len(dup.blocks["ssh8"]) == 2
    with pytest.raises(ValueError, match="per-element direction needs sorted, unique"):
        solid_shell(box, [2, 0], direction=[(0, 0, 1), (0, 0, -1)])
    for no_hex8 in (twice, box_mesh((2, 2, 1), type="hex20")):
        with pytest.raises(ValueError, match="no hex8 elements"):
            solid_shell(no_hex8)
    with pytest.raises(ValueError, match="elements must be"):
        solid_shell(box, np.array([0.0]))
    assert voxel().voxel is not None and solid_shell(voxel())[0].voxel is None


def test_solid_shell_cosine_margin():
    """Per-element directions in the xz plane at angle a: |cos| x, z = cos a, sin a."""
    box = box_mesh((3, 1, 1))  # unit cubes
    for gap, raises in ((0.05, True), (0.15, False)):
        a = np.pi / 4 + np.arcsin(gap / np.sqrt(2))  # sin a - cos a = gap
        d = np.tile([0.0, 0.0, 1.0], (3, 1))
        d[1] = [np.cos(a), 0, np.sin(a)]
        assert abs(d[1, 2] - d[1, 0] - gap) < 1e-12
        if raises:
            with pytest.raises(ValueError, match=r"element 1: thickness axis ambiguous"):
                solid_shell(box, direction=d)
        else:
            solid_shell(box, direction=d)


def test_solid_shell_mixed_blocks():
    from helpers import mixed_mesh

    mesh = mixed_mesh(np.random.default_rng(15))  # hex8 + hex20
    mesh = Mesh(mesh.points, mesh.blocks, mesh.node_sets, {"all": np.arange(mesh.n_el)},
                mesh.material, mesh.orientation)  # fmt: skip
    out, order = solid_shell(mesh, direction=(0, 0, 1))
    assert list(out.blocks) == ["hex20", "ssh8"]
    assert np.array_equal(order[:8], np.arange(8, 16)) and np.array_equal(order[8:], np.arange(8))
    assert np.array_equal(out.material, mesh.material[order])
    assert np.array_equal(out.element_sets["all"], np.arange(16))
    with pytest.raises(ValueError, match="hex8 elements only"):
        solid_shell(mesh, [8])


def voxel():
    return voxel_mesh((4, 3, 1), spacing=(1.0, 1.0, 0.1))


def test_solid_shell_vtu_round_trip(tmp_path):
    box = box_mesh((3, 2, 1), spacing=(1.0, 1.0, 0.1))
    shells = solid_shell(box, direction=(0, 0, 1))[0]
    write(tmp_path / "s.vtu", shells)
    back = read(tmp_path / "s.vtu")
    assert list(back.blocks) == ["hex8"]
    again, order = solid_shell(back, direction=(0, 0, 1))
    assert np.array_equal(again.points, shells.points)
    assert np.array_equal(again.blocks["ssh8"], shells.blocks["ssh8"][order])


# ---------------------------------------------------------------- 10: refusals


def test_refusals():
    box = box_mesh((2, 2, 1), spacing=(1.0, 1.0, 0.1))
    shells = solid_shell(box, direction=(0, 0, 1))[0]
    mat = {0: Material(C_ANI, 1.0)}
    lam = {0: laminate(three_plies(C_ORT))}
    with pytest.raises(ValueError, match="consistent-mass only"):
        assemble(shells, mat, mass="lumped")
    with pytest.raises(ValueError, match="consistent-mass only"):
        kernels.lumped_mass("ssh8", jnp.zeros((8, 3)), 1.0)
    for f in (kernels.stiffness, kernels.mass):
        with pytest.raises(ValueError, match="solid-shell kernels"):
            f("ssh8", jnp.zeros((8, 3)), 1.0)
    for f in (kernels.stiffness_form, kernels.mass_form):
        with pytest.raises(ValueError, match="solid-shell kernels"):
            f("ssh8", jnp.zeros((8, 3)), 1.0, jnp.zeros(24), jnp.zeros(24))
    msg = "material 0 is a Laminate on hex8 elements; laminates need ssh8"
    with pytest.raises(ValueError, match=msg):
        assemble(box, lam)
    for mode in ("stored", "otf"):
        with pytest.raises(ValueError, match=msg):
            unstructured(box, lam, mode=mode)
    with pytest.raises(ValueError, match=msg):
        stencil(voxel(), lam)
    with pytest.raises(ValueError, match="type") as err:
        box_mesh((2, 2, 1), type="ssh8")
    assert "ssh8" not in str(err.value).split("got")[0]  # no error text lists ssh8
    for f in (unstructured, lambda m, mat, mode: unstructured_memory(m, 4, mode=mode)):
        with pytest.raises(ValueError, match="ssh8 operators arrive in PR 16b"):
            f(shells, mat, mode="stored")
        with pytest.raises(ValueError, match=r"stored mode only \(mode='otf'\)"):
            f(shells, mat, mode="otf")
    with pytest.raises(ValueError, match="ssh8 operators arrive in PR 16b"):
        amg_memory(shells, 4)


def test_check_elements_thickness():
    X = (ref.VTK + 1) / 2 * [1, 1, 0.3]
    for a, s in ((6, 1), (2, -1), (4, -1), (0, 1)):
        X[a, :2] += s * 0.7  # twisted: det J > 0 at the Gauss points, < 0 near the faces
    gauss = itertools.product((-ref.G, ref.G), repeat=3)
    assert min(np.linalg.det(ref.jac(X, np.array(p))) for p in gauss) > 0
    z = np.zeros(1, np.int64)
    mesh = Mesh(X, {"hex8": np.arange(8)[None]}, {}, {}, z, z)
    check_elements(mesh)  # hex8: Gauss points only
    shell = Mesh(X, {"ssh8": np.arange(8)[None]}, {}, {}, z, z)
    with pytest.raises(ValueError, match=r"element 0 \(ssh8\): det J"):
        check_elements(shell)


def test_thickness_min():
    """The minimum over zeta in [-1, 1] of the quadratic through det J at zeta = -1, 0, 1."""
    det = np.array([[1, 0.01, 4], [1, 2, 3], [3, 1, 3], [-1, 5, 2]], float)  # (e, 3)
    exact = []
    for dm, d0, dp in det:
        z = np.linspace(-1, 1, 200001)
        exact.append(((dp + dm) / 2 - d0) * z**2 + (dp - dm) / 2 * z + d0)
    got = _thickness_min(np.repeat(det[:, :, None], 4, 2))
    assert np.allclose(got, np.min(exact, 1)[:, None], atol=1e-9)
    assert got[0, 0] < 0 and np.isclose(got[0, 0], 0.01 - 1.5**2 / (4 * 2.49))


def test_material_messages():
    bad_C = np.diag([1.0, 1, 1, 1, 1, -1])
    pairs = [(0, 0)]
    cases = [
        (laminate([(C_ORT, 1.0, 0.0, 0.5), (bad_C, 1.0, 0.0, 0.5)]), "material 0 ply 1: stiffness"),
        (laminate([(C_ORT, 1.0, 0.0, 0.5), (C_ORT, -1.0, 0.0, 0.5)]), "material 0 ply 1: rho"),
        (laminate([(C_ORT, 1.0, np.nan, 1.0)]), "material 0 ply 0: angle"),
        (laminate([(C_ORT, 1.0, 0.0, 0.5), (C_ORT, 1.0, 0.0, 0.4)]),
         "material 0: ply fractions must be > 0 and sum to 1"),
        (laminate([(C_ORT, 1.0, 0.0, 1.0), (C_ORT, 1.0, 0.0, 0.0)]),
         "material 0: ply fractions must be > 0 and sum to 1"),
        (Laminate(()), "non-empty tuple of Ply"),
    ]  # fmt: skip
    for lam_, msg in cases:
        with pytest.raises(ValueError, match=msg):
            check_materials(pairs, {0: lam_})
