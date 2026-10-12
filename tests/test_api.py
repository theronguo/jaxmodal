"""solve_modes against reference_modes (SciPy) on the assembled matrices."""

import importlib

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from jaxmodal.eigen import reference_modes, solve_modes
from jaxmodal.materials import Material, from_euler, isotropic, transversely_isotropic
from jaxmodal.mesh import Mesh, box_mesh, voxel_mesh
from jaxmodal.operators import assemble, dof_mask, restrict

api = importlib.import_module("jaxmodal.eigen.api")

EPS = np.finfo(float).eps
MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
BOX = (1.0, 0.83, 0.67)


def box(shape=(6, 5, 4), t="hex20"):
    return box_mesh(shape, spacing=tuple(a / s for a, s in zip(BOX, shape, strict=True)), type=t)


def reference(mesh, mats, bcs, n, orientations=None, **kw):
    """(elastic lam, rigid count, lam_max, full lam): n + 6 elastic modes (to finish clusters);
    the rigid count is pinned by the zero eigenvalues, independently of solve_modes."""
    K, M = assemble(mesh, mats, orientations, **kw)
    free = dof_mask(mesh, bcs)
    lam = reference_modes(K, M, n + 12 + 6, free, points=mesh.points)[0]
    nr = int(np.sum(lam <= 1e-8 * lam[lam > 1e-8][0]))
    Kf, Mf, _ = restrict(K, M, free)
    lmax = eigsh(Kf, 1, M=Mf, which="LA", tol=1e-3, return_eigenvectors=False)[0]
    return lam[nr : nr + n + 6], nr, lmax, lam


def check_lam(lam, lr, lmax, tol):
    """9a rule: max(1e-8, 10 eps lam_max / lam) for relative gap >= 1e-2, 10 tol cluster-wise
    (sorted) otherwise."""
    n = len(lam)
    gap = np.diff(lr) / lr[1:]
    isolated = np.r_[True, gap[: n - 1] >= 1e-2] & (gap[:n] >= 1e-2)
    err = np.abs(lam / lr[:n] - 1)
    f = np.maximum(1e-8, 10 * EPS * lmax / lr[:n])
    assert np.all(err[isolated] <= f[isolated]), err
    assert np.all(err <= np.maximum(f, 10 * tol)), err


@pytest.fixture
def spies(monkeypatch):
    """Record the calls of the builders and lobpcg, forwarding to the real functions."""
    calls = {"unstructured": [], "stencil": [], "lobpcg": []}
    for name in calls:
        real = getattr(api, name)

        def spy(*a, _real=real, _name=name, **kw):
            calls[_name].append((a, kw))
            return _real(*a, **kw)

        monkeypatch.setattr(api, name, spy)
    return calls


def m_dot(mesh, mats, A, B, **kw):
    M = assemble(mesh, mats, **kw)[1]
    return A.T @ (M @ B)


def test_clamped(spies):
    """Test 1: unstructured dispatch, lam, V M-orthonormal with constrained rows 0, no rigid
    modes, frequencies."""
    mesh, bcs = box(), {"xmin": "xyz"}
    lam, V, info = solve_modes(mesh, MAT, bcs, 10, tol=1e-10)
    lr, nr, lmax, _ = reference(mesh, MAT, bcs, 10)
    assert nr == 0 and info["converged"]
    check_lam(lam, lr, lmax, 1e-10)
    free = dof_mask(mesh, bcs)
    assert np.all(V[~free] == 0)
    assert np.abs(m_dot(mesh, MAT, V, V) - np.eye(10)).max() <= 1e-10
    assert info["operator"] == "unstructured"
    assert len(spies["unstructured"]) == 1 and not spies["stencil"]
    assert info["n_rigid"] == 0 and info["rigid"].shape == (mesh.points.size, 0)
    np.testing.assert_allclose(info["frequencies"], np.sqrt(lam) / (2 * np.pi), rtol=1e-15)


def test_free_nodal_mass():
    """Test 2: free-free with nodal masses: 6 rigid modes deflated and returned M-orthonormal
    (full mass), lam above them; the nodal masses change lam observably."""
    mesh = box()
    nodal = np.random.default_rng(0).uniform(0.5, 1.5, len(mesh.points)) * 2.0 * 0.67 / 480
    lam, V, info = solve_modes(mesh, MAT, {}, 10, nodal_mass=nodal, tol=1e-10)
    lr, nr, lmax, _ = reference(mesh, MAT, {}, 10, nodal_mass=nodal)
    assert nr == 6 and info["n_rigid"] == 6 and info["converged"]
    check_lam(lam, lr, lmax, 1e-10)
    R = info["rigid"]
    assert np.abs(m_dot(mesh, MAT, R, R, nodal_mass=nodal) - np.eye(6)).max() <= 1e-10
    assert np.abs(m_dot(mesh, MAT, R, V, nodal_mass=nodal)).max() <= 1e-10
    plain = reference(mesh, MAT, {}, 10)[0]
    assert np.abs(plain[:10] / lr[:10] - 1).min() > 100 * 1e-8


def test_sliding():
    """Test 3: sliding on xmin leaves T_y, T_z and R_x."""
    mesh, bcs = box(), {"xmin": "x"}
    lam, _, info = solve_modes(mesh, MAT, bcs, 10, tol=1e-10)
    lr, nr, lmax, _ = reference(mesh, MAT, bcs, 10)
    assert nr == 3 and info["n_rigid"] == 3 and info["converged"]
    check_lam(lam, lr, lmax, 1e-10)


@pytest.mark.parametrize("bcs,n_rigid", [({"zmin": "xyz"}, 0), ({}, 6)], ids=["clamped", "free"])
def test_voxel(spies, bcs, n_rigid):
    """Test 4: voxel meshes get the stencil."""
    mesh = voxel_mesh((10, 8, 6), spacing=(0.1, 0.1, 0.1))
    lam, _, info = solve_modes(mesh, MAT, bcs, 10, tol=1e-10)
    lr, nr, lmax, _ = reference(mesh, MAT, bcs, 10)
    assert nr == n_rigid == info["n_rigid"] and info["converged"]
    check_lam(lam, lr, lmax, 1e-10)
    assert info["operator"] == "stencil"
    assert len(spies["stencil"]) == 1 and not spies["unstructured"]


def test_rotated_lumped():
    """Test 5: hex27, a rotated transversely isotropic phase on half the elements, HRZ lumped
    mass; the rotation changes lam observably."""
    mesh = box((4, 3, 3), "hex27")
    half = np.arange(mesh.n_el) % 2 == 1
    mesh.material[half], mesh.orientation[half] = 1, 1
    mats = {**MAT, 1: Material(transversely_isotropic(9.0, 2.0, 0.3, 0.4, 1.0), 1.5)}
    rot = {1: from_euler(np.array([0.4, 0.9, -0.3]))}
    lam, _, info = solve_modes(mesh, mats, {"xmin": "xyz"}, 10, rot, mass="lumped", tol=1e-10)
    lr, nr, lmax, _ = reference(mesh, mats, {"xmin": "xyz"}, 10, rot, mass="lumped")
    assert nr == 0 and info["converged"]
    check_lam(lam, lr, lmax, 1e-10)
    plain = reference(mesh, mats, {"xmin": "xyz"}, 10, {1: np.eye(3)}, mass="lumped")[0]
    assert np.abs(plain[:10] / lr[:10] - 1).min() > 100 * 1e-8


def test_forwarding(spies):
    """Test 6: every lobpcg option reaches lobpcg."""
    mesh, bcs = box(), {"xmin": "xyz"}
    lr, _, lmax, _ = reference(mesh, MAT, bcs, 12)
    opts = [
        dict(tol=1e-9, maxit=900, seed=3, chunk=4, guards=2),
        dict(tol=1e-6, chunk=4, basis_dtype="float32", low_memory=True),
    ]
    for kw in opts:
        lam, _, info = solve_modes(mesh, MAT, bcs, 12, **kw)
        a, got = spies["lobpcg"][-1]
        got = {**got, "tol": a[3]}
        assert all(got[k] == v for k, v in kw.items()), got
        assert info["converged"] and len(info["chunks"]) == 3
        check_lam(lam, lr, lmax, kw["tol"])


def inverted():
    m = box((2, 2, 2))
    blocks = {"hex20": m.blocks["hex20"].copy()}
    flip = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 16, 17, 18, 19]  # top <-> bottom
    blocks["hex20"][0] = blocks["hex20"][0, flip]
    return Mesh(m.points, blocks, m.node_sets, m.element_sets, m.material, m.orientation)


@pytest.mark.parametrize(
    "case",
    ["unknown set", "bad components", "missing material", "density", "inverted"],
)
def test_host_checks(spies, monkeypatch, case):
    """Test 7: bad bcs raise before any operator is built; material and element errors raise
    in the builders' host checks, before lobpcg is reached."""
    mesh, mats, bcs = box((2, 2, 2)), MAT, {"xmin": "xyz"}
    if case in ("unknown set", "bad components"):
        bcs = {"nowhere": "xyz"} if case == "unknown set" else {"xmin": "xw"}
        match = "unknown node set" if case == "unknown set" else "distinct letters"

        def fail(*a, **kw):
            raise AssertionError("operator built before the BC check")

        monkeypatch.setattr(api, "unstructured", fail)
        monkeypatch.setattr(api, "stencil", fail)
    else:
        if case == "missing material":
            mats, match = {1: MAT[0]}, "not in materials"
        elif case == "density":
            mats, match = {0: Material(MAT[0].C, 0.0)}, "rho must be"
        else:
            mesh, match = inverted(), "det J"

        def fail(*a, **kw):
            raise AssertionError("lobpcg reached before the host checks")

        monkeypatch.setattr(api, "lobpcg", fail)
    with pytest.raises(ValueError, match=match):
        solve_modes(mesh, mats, bcs, 2)
