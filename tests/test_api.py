"""solve_modes against reference_modes (SciPy) on the assembled matrices."""

import importlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from exact import mac
from scipy.sparse.linalg import eigsh

from benchmarks import gmg_study
from jaxmodal.eigen import lobpcg, reference_modes, solve_modes
from jaxmodal.materials import Material, from_euler, isotropic, transversely_isotropic
from jaxmodal.memory import solve_modes_memory
from jaxmodal.mesh import Mesh, box_mesh, voxel_mesh
from jaxmodal.operators import assemble, dof_mask, m_orthonormalize, restrict, rigid_modes
from jaxmodal.precond import GMG, SingularShiftError, amg, gmg, jacobi

api = importlib.import_module("jaxmodal.eigen.api")
G = importlib.import_module("jaxmodal.precond.gmg")

EPS = np.finfo(float).eps
MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
BOX = (1.0, 0.83, 0.67)


def box(shape=(6, 5, 4), t="hex20"):
    return box_mesh(shape, spacing=tuple(a / s for a, s in zip(BOX, shape, strict=True)), type=t)


def reference(mesh, mats, bcs, n, orientations=None, **kw):
    """(elastic lam, rigid count, lam_max): n + 6 elastic modes (to finish clusters); the rigid
    count (at most 6 here) is pinned by the zero eigenvalues, independently of solve_modes."""
    K, M = assemble(mesh, mats, orientations, **kw)
    free = dof_mask(mesh, bcs)
    lam = reference_modes(K, M, n + 6 + 6, free, points=mesh.points)[0]
    nr = int(np.sum(lam <= 1e-8 * lam[lam > 1e-8][0]))
    Kf, Mf, _ = restrict(K, M, free)
    lmax = eigsh(Kf, 1, M=Mf, which="LA", tol=1e-3, return_eigenvectors=False)[0]
    return lam[nr : nr + n + 6], nr, lmax


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
    lr, nr, lmax = reference(mesh, MAT, bcs, 10)
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
    nodal = (
        np.random.default_rng(0).uniform(0.5, 1.5, len(mesh.points)) * 2.0 * 0.67 / 480
    )  # ~2x body
    lam, V, info = solve_modes(mesh, MAT, {}, 10, nodal_mass=nodal, tol=1e-10)
    lr, nr, lmax = reference(mesh, MAT, {}, 10, nodal_mass=nodal)
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
    lr, nr, lmax = reference(mesh, MAT, bcs, 10)
    assert nr == 3 and info["n_rigid"] == 3 and info["converged"]
    check_lam(lam, lr, lmax, 1e-10)


@pytest.mark.parametrize("bcs,n_rigid", [({"zmin": "xyz"}, 0), ({}, 6)], ids=["clamped", "free"])
def test_voxel(spies, bcs, n_rigid):
    """Test 4: voxel meshes get the stencil."""
    mesh = voxel_mesh((10, 8, 6), spacing=(0.1, 0.1, 0.1))
    lam, _, info = solve_modes(mesh, MAT, bcs, 10, tol=1e-10)
    lr, nr, lmax = reference(mesh, MAT, bcs, 10)
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
    lr, nr, lmax = reference(mesh, mats, {"xmin": "xyz"}, 10, rot, mass="lumped")
    assert nr == 0 and info["converged"]
    check_lam(lam, lr, lmax, 1e-10)
    plain = reference(mesh, mats, {"xmin": "xyz"}, 10, {1: np.eye(3)}, mass="lumped")[0]
    assert np.abs(plain[:10] / lr[:10] - 1).min() > 100 * 1e-8


def test_forwarding(spies):
    """Test 6: every lobpcg option reaches lobpcg."""
    mesh, bcs = box(), {"xmin": "xyz"}
    lr, _, lmax = reference(mesh, MAT, bcs, 12)
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
def test_host_checks(monkeypatch, case):
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


# ---------------------------------------------------------------- PR 11b: GMG in solve_modes

STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}


@pytest.fixture
def built(monkeypatch):
    """The sigma of every successful gmg build inside eigen/api.py (forwarded to the real
    gmg; a refused sigma = 0 is not recorded)."""
    sigmas = []

    def spy(op, free, sigma, **kw):
        P = gmg(op, free, sigma, **kw)
        sigmas.append(sigma)
        return P

    monkeypatch.setattr(api, "gmg", spy)
    return sigmas


@pytest.fixture
def preconds(monkeypatch):
    """The precond of every lobpcg call inside eigen/api.py (the last is the final solve's)."""
    calls = []

    def spy(op, free, n_modes, tol, R, precond, *a, **kw):
        calls.append(precond)
        return lobpcg(op, free, n_modes, tol, R, precond, *a, **kw)

    monkeypatch.setattr(api, "lobpcg", spy)
    return calls


def rigid(op, mesh, free):
    R = jnp.asarray(rigid_modes(mesh, free))
    return m_orthonormalize(R, op.apply_M(R)) if R.shape[1] else None


SIGMA_CASES = [
    ("box8", (8,) * 3, "clamped", None, None),
    ("odd9", (9, 8, 8), "clamped", None, None),
    ("plate32", (32, 32, 1), "clamped", None, None),
    ("beam64x2", (64, 2, 2), "clamped", None, None),
    ("free_box8", (8,) * 3, "free", None, None),
    ("free_plate16", (16, 16, 1), "free", None, None),
    ("free_beam128", (128, 1, 1), "free", None, None),  # 2.7 lam_1 after one round
    pytest.param("aniso16", (16,) * 3, "clamped", (1, 1, 0.1), None, marks=pytest.mark.slow),
    pytest.param("simp32", (32,) * 3, "clamped", None, "simp", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("name,shape,bcs,factors,density", SIGMA_CASES)
def test_sigma_estimate(built, name, shape, bcs, factors, density):
    """Test 1: the GMG-refined pre-estimate is within [lam_1, 2 lam_1] (SciPy lam_1); clamped
    bodies build once at sigma = 0, free ones start above 0 and never raise sigma."""
    op, free, mesh = gmg_study.case(shape, bcs, factors, density)
    lam1 = gmg_study.lam1(op, free, mesh)
    sigma, its = api._precond_sigma(op, free, rigid(op, mesh, free), lambda s: api.gmg(op, free, s))
    # lower bound: the Rayleigh quotient and the reference round at eps ||K|| (4e-8 on SIMP)
    assert lam1 * (1 - 1e-6) <= sigma <= 2 * lam1, sigma / lam1
    if bcs == "clamped":
        assert built == [0.0]
    else:  # rebuilt only after a drop by more than 3x; stopped on a smaller drop or at 4 rounds
        assert built[0] > 0 and all(b < a / 3 for a, b in zip(built, built[1:], strict=False))
        assert sigma > built[-1] / 3 or len(built) == 4
    assert 0 < its <= 20 * len(built)


def test_sigma_rounds(monkeypatch):
    """The pre-estimate's round logic on scripted probes: rebuild while sigma drops by more
    than 3x, keep the smaller of the last two, at most 4 rounds."""

    def run(probes):
        builds, probes = [], iter(probes)

        def build(sigma):
            if sigma == 0:
                raise SingularShiftError("free")
            builds.append(sigma)
            return object()

        monkeypatch.setattr(api, "sigma_estimate", lambda *a, **kw: (100.0, None))
        monkeypatch.setattr(
            api, "lobpcg", lambda *a, **kw: ([next(probes)], None, {"iterations": 2})
        )
        return api._precond_sigma(None, None, None, build), builds

    assert run([10.0, 20.0]) == ((10.0, 4), [100.0, 10.0])  # a rising probe keeps the minimum
    assert run([50.0]) == ((50.0, 2), [100.0])
    assert run([30.0, 9.0, 2.0, 0.5]) == ((0.5, 8), [100.0, 30.0, 9.0, 2.0])


def test_sigma_hint(built, monkeypatch):
    """Test 2: a sigma hint skips the pre-estimate (one build, at the hint) and gives the same
    modes; sigma = 0 on a free body raises instead of being replaced."""
    mesh, bcs = voxel_mesh((9, 6, 5), (1 / 9,) * 3), {"xmin": "xyz"}
    lam0, _, info0 = solve_modes(mesh, STEEL, bcs, 4, tol=1e-8)
    hint = 1.5 * info0["sigma"]
    del built[:]

    def fail(*a, **kw):
        raise AssertionError("pre-estimate ran despite the hint")

    monkeypatch.setattr(api, "_precond_sigma", fail)
    lam, _, info = solve_modes(mesh, STEEL, bcs, 4, tol=1e-8, sigma=hint)
    assert info["sigma"] == hint and info["sigma_iterations"] == 0 and built == [hint]
    np.testing.assert_allclose(lam, lam0, rtol=1e-7)
    with pytest.raises(SingularShiftError):
        solve_modes(mesh, STEEL, {}, 4, sigma=0.0)


def odd_case(name):
    """(mesh, bcs, nodal_mass) on the odd 9x6x5 unit voxel grid."""
    if name == "masked":
        mask = np.ones((9, 6, 5), bool)
        mask[6:, 3:, 2:] = False  # an L-shape with a notch, face-connected
        mask[3:5, 2, 1:3] = False
        mesh = voxel_mesh((9, 6, 5), (1 / 9,) * 3, mask=mask)
        nodal = np.random.default_rng(3).uniform(0, 2e-3, len(mesh.points)) * 7850.0 / 729
        return mesh, {"xmin": "xyz"}, nodal
    return voxel_mesh((9, 6, 5), (1 / 9,) * 3), {"xmin": "xyz"} if name == "clamped" else {}, None


@pytest.mark.parametrize("name", ["clamped", "free", "masked"])
def test_gmg_modes(built, preconds, name):
    """Test 3: solve_modes with GMG on odd voxel meshes matches reference_modes (lam 1e-8, MAC
    1 - 1e-8); the final build is at info["sigma"]."""
    mesh, bcs, nodal = odd_case(name)
    lam, V, info = solve_modes(mesh, STEEL, bcs, 6, nodal_mass=nodal, tol=1e-10)
    K, M = assemble(mesh, STEEL, nodal_mass=nodal)
    free = dof_mask(mesh, bcs)
    nr = info["n_rigid"]
    lr, Vr = reference_modes(K, M, nr + 6, free, points=mesh.points)
    assert info["converged"] and info["precond"] == "gmg" and nr == (6 if name == "free" else 0)
    assert np.abs(lam / lr[nr:] - 1).max() <= 1e-8
    assert np.diag(mac(V, Vr[:, nr:], M)).min() >= 1 - 1e-8
    assert built[-1] == info["sigma"] > 0
    assert isinstance(preconds[-1], GMG) and preconds[-1].dtype == "float64"
    if name != "free":
        assert built == [0.0, info["sigma"]]


def test_dispatch(preconds):
    """Test 4 (13b test 6): "auto" picks GMG on voxel meshes and AMG otherwise; Jacobi on
    request; option errors."""
    lam, _, info = solve_modes(box((3, 3, 2)), MAT, {"xmin": "xyz"}, 2)
    assert info["precond"] == "amg" and info["sigma"] > 0 and info["sigma_iterations"] > 0
    assert isinstance(preconds[-1], GMG) and info["time"]["precond"] >= 0
    _, _, info = solve_modes(box((3, 3, 2)), MAT, {"xmin": "xyz"}, 2, precond="jacobi")
    assert info["precond"] == "jacobi" and info["sigma"] == 0.0 and info["sigma_iterations"] == 0
    vox = voxel_mesh((5, 4, 3), (0.2,) * 3)
    _, _, info = solve_modes(vox, MAT, {"xmin": "xyz"}, 2, precond="jacobi")
    assert info["precond"] == "jacobi" and info["sigma"] == 0.0 and info["sigma_iterations"] == 0
    _, _, info = solve_modes(vox, MAT, {"xmin": "xyz"}, 2, precond="jacobi", sigma=2e5)
    ref = jacobi(api.stencil(vox, MAT), dof_mask(vox, {"xmin": "xyz"}), 2e5)
    np.testing.assert_array_equal(preconds[-1].args[0], ref.args[0])  # built at the hint
    _, _, info = solve_modes(vox, MAT, {"xmin": "xyz"}, 2)
    assert info["precond"] == "gmg" and info["sigma"] > 0 and info["sigma_iterations"] > 0
    assert isinstance(preconds[-1], GMG) and preconds[-1].dtype == "float64"
    _, _, info = solve_modes(vox, MAT, {"xmin": "xyz"}, 2, precond_dtype="float32")
    assert isinstance(preconds[-1], GMG) and preconds[-1].dtype == "float32"
    assert {"setup", "iterations", "precond"} <= info["time"].keys()
    bad = [
        (box((2, 2, 2)), dict(precond="gmg"), "voxel mesh"),
        (vox, dict(precond="amg"), "unstructured mesh"),
        (vox, dict(precond="ilu"), "precond must be"),
        (vox, dict(precond="jacobi", precond_dtype="float32"), "needs precond 'gmg'"),
        (vox, dict(precond_dtype="float16"), "precond_dtype must be"),
        (vox, dict(sigma=-1.0), "sigma must be"),
        (vox, dict(sigma=float("nan")), "sigma must be"),
    ]
    for mesh, kw, match in bad:
        with pytest.raises(ValueError, match=match):
            solve_modes(mesh, MAT, {"xmin": "xyz"}, 2, **kw)


@pytest.fixture
def amg_builds(monkeypatch):
    """(sigma, nonsingular) of every successful amg build inside eigen/api.py (forwarded to the
    real amg; a refused sigma = 0 is not recorded)."""
    calls = []

    def spy(op, free, sigma, points, **kw):
        P = amg(op, free, sigma, points, **kw)
        calls.append((sigma, kw.get("nonsingular", False)))
        return P

    monkeypatch.setattr(api, "amg", spy)
    return calls


def check_amg_builds(calls, info):
    """13b tests 4-5: no rigid modes -> the first build at sigma = 0 with nonsingular; rigid
    modes -> every build at sigma > 0 without; the final build at info["sigma"]."""
    if info["n_rigid"] == 0:
        assert calls[0] == (0.0, True) and all(ns for _, ns in calls)
    else:
        assert all(s > 0 and not ns for s, ns in calls)
    assert calls[-1][0] == info["sigma"] > 0


AMG_BCS = {"clamped": {"xmin": "xyz"}, "free": {}, "sliding": {"xmin": "x"}}


@pytest.mark.parametrize("case", list(AMG_BCS))
@pytest.mark.parametrize("t, shape", [("hex20", (5, 4, 3)), ("hex8", (6, 5, 4))])
def test_amg_modes(amg_builds, t, shape, case):
    """13b test 4 (roadmap): solve_modes with AMG against reference_modes on non-degenerate
    boxes (free with nodal masses): lam to 1e-8, MAC >= 1 - 1e-8, the amg builds."""
    mesh, bcs = box(shape, t), AMG_BCS[case]
    nodal = None
    if case == "free":
        nodal = np.random.default_rng(0).uniform(0.5, 1.5, len(mesh.points)) * 1e-3
    lam, V, info = solve_modes(mesh, MAT, bcs, 6, nodal_mass=nodal, tol=1e-10)
    assert info["precond"] == "amg" and info["converged"]
    assert info["n_rigid"] == {"clamped": 0, "free": 6, "sliding": 3}[case]
    K, M = assemble(mesh, MAT, nodal_mass=nodal)
    k = info["n_rigid"]
    lr, Vr = reference_modes(K, M, 6 + k, dof_mask(mesh, bcs), points=mesh.points)
    assert np.abs(lam / lr[k:] - 1).max() <= 1e-8
    assert mac(np.asarray(V), Vr[:, k:], M).diagonal().min() >= 1 - 1e-8
    check_amg_builds(amg_builds, info)


AMG_SIGMA_CASES = [
    ("hex20", (8, 8, 8), "clamped"),
    ("hex8", (12, 12, 12), "clamped"),
    ("hex20", (16, 2, 2), "clamped"),  # beam
    ("hex20", (12, 12, 1), "clamped"),  # plate
    ("hex20", (6, 6, 6), "free"),
    ("hex20", (32, 2, 2), "free"),  # free beam
    pytest.param("hex20", (16, 16, 1), "free", marks=pytest.mark.slow),  # free plate
]


@pytest.mark.parametrize("t, shape, case", AMG_SIGMA_CASES)
def test_amg_sigma(amg_builds, t, shape, case):
    """13b test 5: the AMG-refined pre-estimate lands in [lam_1 (1 - 1e-6), 2 lam_1] (SciPy
    lam_1); the builds as in test 4; free bodies' build shifts never increase."""
    mesh = box_mesh(shape, (1 / max(shape),) * 3, type=t)
    free = dof_mask(mesh, AMG_BCS[case])
    op = api.unstructured(mesh, MAT)
    R = rigid(op, mesh, free)
    k = 0 if R is None else R.shape[1]

    def build(s):
        return api.amg(op, free, s, mesh.points, nonsingular=k == 0)

    sigma, its = api._precond_sigma(op, free, R, build)
    K, M = assemble(mesh, MAT)
    lam1 = reference_modes(K, M, 6 + k, free, points=mesh.points)[0][k]  # past clusters
    assert lam1 * (1 - 1e-6) <= sigma <= 2 * lam1, sigma / lam1
    shifts = [s for s, _ in amg_builds]
    if k == 0:
        assert amg_builds[0] == (0.0, True) and all(ns for _, ns in amg_builds)
    else:
        assert all(s > 0 and not ns for s, ns in amg_builds)
        assert all(b <= a for a, b in zip(shifts, shifts[1:], strict=False))
    assert its > 0


def test_amg_hint(amg_builds):
    """13b test 6: a sigma hint skips the pre-estimate (one build, at the hint); sigma = 0 on a
    free unstructured body raises SingularShiftError."""
    mesh = box((4, 3, 3))
    _, _, info = solve_modes(mesh, MAT, {"xmin": "xyz"}, 2, sigma=3.0)
    assert amg_builds == [(3.0, True)] and info["sigma_iterations"] == 0
    with pytest.raises(SingularShiftError):
        solve_modes(mesh, MAT, {}, 2, sigma=0.0)


def test_singular_narrow(monkeypatch):
    """Test 4: only the sigma = 0 refusals are SingularShiftError; the finite backstops and
    any other ValueError at sigma = 0 propagate out of solve_modes."""
    vox = voxel_mesh((5, 4, 3), (0.2,) * 3)
    op, free = api.stencil(vox, MAT), dof_mask(vox, {})
    with pytest.raises(SingularShiftError):
        gmg(op, free, 0.0)
    with monkeypatch.context() as mp:
        mp.setattr(G, "_inverse_blocks", lambda B, f: (jnp.full_like(B, jnp.nan), None))
        with pytest.raises(ValueError, match="not finite") as e:
            gmg(op, dof_mask(vox, {"xmin": "xyz"}), 1.0)
        assert not isinstance(e.value, SingularShiftError)

    def plain(op, free, sigma, **kw):
        if sigma == 0:
            raise ValueError("plain")
        return gmg(op, free, sigma, **kw)

    def no_estimate(*a, **kw):
        raise AssertionError("fell back to sigma_estimate")

    monkeypatch.setattr(api, "gmg", plain)
    monkeypatch.setattr(api, "sigma_estimate", no_estimate)
    with pytest.raises(ValueError, match="plain"):
        solve_modes(vox, MAT, {"xmin": "xyz"}, 2)


@pytest.mark.peak  # its time budget has a 10x margin
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="Mac acceptance")
def test_mac_acceptance():
    """Test 10: 87^3 steel box (2.04M DOF), 10 modes, tol 1e-6, defaults: a cold run within
    10 min, all converged, and the larger of two warm peaks within 20% of solve_modes_memory,
    for the fp64 and fp32 V-cycles."""
    for dtype, r in gmg_study.acceptance(87).items():
        assert r["converged"] and r["time_s"] <= 600, (dtype, r["time_s"])
        peak, est = max(r["peaks_bytes"]), r["estimate_bytes"]
        assert est >= 2e9 and abs(est - peak) <= 0.2 * peak, (dtype, peak, est)


@pytest.mark.slow
def test_acceptance_smoke():
    """Test 11: gmg_study.acceptance runs and records times, iterations, peaks and estimates;
    the estimate is solve_modes_memory's."""
    r = gmg_study.acceptance(8)
    mesh = voxel_mesh((8,) * 3, (1 / 8,) * 3)
    for dtype in ("float64", "float32"):
        assert r[dtype]["converged"] and r[dtype]["iterations"] > 0 and r[dtype]["time_s"] > 0
        assert len(r[dtype]["peaks_bytes"]) == 2 and r[dtype]["sigma_iterations"] > 0
        est = solve_modes_memory(mesh, {"xmin": "xyz"}, 10, precond_dtype=dtype)["total"]
        assert r[dtype]["estimate_bytes"] == est
