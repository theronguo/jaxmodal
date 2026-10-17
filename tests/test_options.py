"""PR 13c: every solver option reachable from solve_modes (docs/eigensolvers.md#options)."""

import functools
import importlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exact import mac
from helpers import STEEL, owned_bytes, step_bytes

from jaxmodal.eigen import count, lobpcg, reference_modes, solve_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.memory import (
    AMG_LOOP_K,
    GMG_LOOP,
    amg_memory,
    gmg_memory,
    lobpcg_memory,
    solve_modes_memory,
    stencil_memory,
    unstructured_memory,
)
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import ap_operator, assemble, dof_mask, stencil, unstructured
from jaxmodal.precond import (
    SingularShiftError,
    amg,
    block_jacobi,
    chebyshev,
    gmg,
    jacobi,
    lmax_estimate,
)  # fmt: skip

api = importlib.import_module("jaxmodal.eigen.api")
LOB = importlib.import_module("jaxmodal.eigen.lobpcg")
SI = importlib.import_module("jaxmodal.eigen.shift_invert")
G = importlib.import_module("jaxmodal.precond.gmg")
CHEB = importlib.import_module("jaxmodal.precond.chebyshev")

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
CLAMP = {"xmin": "xyz"}
SPIED = ("unstructured", "stencil", "gmg", "amg", "lobpcg", "sigma_estimate")


def box(shape=(6, 5, 4), t="hex20"):
    return box_mesh(shape, spacing=(1 / shape[0],) * 3, type=t)


def vox(n=6):
    return voxel_mesh((n, n - 1, n - 2), (1 / n,) * 3)


def scalars(a):
    """The scalar and None positional args (arrays and objects dropped)."""
    return [x for x in a if x is None or isinstance(x, (bool, int, float, str))]


@pytest.fixture
def calls(monkeypatch):
    """(name, scalar positional args, kwargs, positional args) of every builder / lobpcg /
    sigma_estimate call inside eigen/api.py, forwarded to the real functions."""
    rec = []
    for name in SPIED:
        real = getattr(api, name)

        def spy(*a, _real=real, _name=name, **kw):
            rec.append((_name, scalars(a), kw, a))
            return _real(*a, **kw)

        monkeypatch.setattr(api, name, spy)
    return rec


# ------------------------------------------------------------- 1: defaults and forwarding

LOBPCG_FINAL = dict(
    maxit=500, seed=0, chunk=None, guards=None, basis_dtype="float64", low_memory=False
)


def test_defaults_snapshot(calls):
    """Test 1: with every option None the calls equal today's (a clamped body: sigma = 0 is
    accepted, one pre-estimate round)."""
    lam, _, info = solve_modes(box(), MAT, CLAMP, 4)
    s = info["sigma"]
    assert [c[:3] for c in calls] == [
        ("unstructured", [None], dict(mass="consistent", nodal_mass=None)),
        ("amg", [0.0], dict(nonsingular=True)),
        ("lobpcg", [1, 0.1, None], dict(maxit=20, seed=0)),
        ("amg", [s], dict(nonsingular=True)),
        ("lobpcg", [4, 1e-6, None, None], LOBPCG_FINAL),
    ]
    del calls[:]
    lam, _, info = solve_modes(vox(), STEEL, CLAMP, 4)
    s = info["sigma"]
    assert [c[:3] for c in calls] == [
        ("stencil", [None], dict(mass="consistent", nodal_mass=None)),
        ("gmg", [0.0], {}),
        ("lobpcg", [1, 0.1, None], dict(maxit=20, seed=0)),
        ("gmg", [s], dict(dtype="float64")),
        ("lobpcg", [4, 1e-6, None, None], LOBPCG_FINAL),
    ]


OPERATOR_VALUES = {"mode": "otf", "path": "gather", "chunk": 7, "range_nodes": 64, "slab": 2}
PRECOND_VALUES = {"degree": 3, "coarse_degree": 2, "ratio": 20.0, "max_coarse": 500}
PRECOND_CASES = [(p, k) for p, keys in api.PRECOND_OPTIONS.items() for k in keys]
OPERATOR_CASES = [(o, k) for o, keys in api.OPERATOR_OPTIONS.items() for k in keys]


@pytest.mark.parametrize("op_name, key", OPERATOR_CASES)
def test_operator_forwarding(calls, op_name, key):
    """Test 1: each operator option reaches the operator build with its value."""
    mesh = vox() if op_name == "stencil" else box()
    v = OPERATOR_VALUES[key]
    _, _, info = solve_modes(mesh, STEEL, CLAMP, 2, operator_options={key: v})
    ((name, _, kw, _),) = [c for c in calls if c[0] in ("stencil", "unstructured")]
    assert name == op_name and kw[key] == v and info["converged"]


@pytest.mark.parametrize("pc, key", PRECOND_CASES)
def test_precond_forwarding(calls, pc, key):
    """Test 1: each preconditioner option reaches every pre-estimate build and the final one."""
    mesh = vox() if pc == "gmg" else box()
    v = PRECOND_VALUES[key]
    _, _, info = solve_modes(mesh, STEEL, CLAMP, 2, precond=pc, precond_options={key: v})
    builds = [c for c in calls if c[0] == pc]
    assert len(builds) >= 2 and all(c[2][key] == v for c in builds) and info["converged"]


def test_x0_and_seed_forwarding(calls):
    """Test 1: X0 reaches the final lobpcg; seed reaches the pre-estimate LOBPCG and
    sigma_estimate (a free body: sigma = 0 is refused)."""
    mesh = box()
    X0 = np.random.default_rng(0).standard_normal((3 * len(mesh.points), 2))
    solve_modes(mesh, MAT, {}, 2, seed=3, X0=X0)
    lob = [c for c in calls if c[0] == "lobpcg"]
    assert lob[-1][3][6] is X0 and all(len(c[3]) == 6 for c in lob[:-1])
    assert all(c[2]["seed"] == 3 for c in lob)
    ((_, _, kw, _),) = [c for c in calls if c[0] == "sigma_estimate"]
    assert kw == dict(maxit=10, seed=3)


# ------------------------------------------------------------- 2: effects against references


def check_reference(mesh, mats, bcs, lam, V, n_rigid=0):
    K, M = assemble(mesh, mats)
    lr, Vr = reference_modes(K, M, len(lam) + n_rigid, dof_mask(mesh, bcs), points=mesh.points)
    assert np.abs(lam / lr[n_rigid:] - 1).max() <= 1e-8
    assert mac(np.asarray(V), Vr[:, n_rigid:], M).diagonal().min() >= 1 - 1e-8


def test_otf_effect():
    """Test 2: operator_options mode "otf" gives the reference modes."""
    mesh = box()
    lam, V, info = solve_modes(mesh, MAT, CLAMP, 4, tol=1e-10, operator_options={"mode": "otf"})
    assert info["converged"]
    check_reference(mesh, MAT, CLAMP, lam, V)


def test_max_coarse_effect(calls):
    """Test 2: max_coarse 100 on hex20 8^3 (2187 corner DOF) adds an AMG level (2 -> 3, measured)
    and gives the same (reference) modes."""
    mesh = box_mesh((8, 8, 8), (1 / 8, 0.9 / 8, 0.8 / 8), type="hex20")  # no repeated modes
    for mc, n_levels in ((None, 2), (100, 3)):
        del calls[:]
        opts = None if mc is None else {"max_coarse": mc}
        lam, V, info = solve_modes(mesh, MAT, CLAMP, 4, tol=1e-10, precond_options=opts)
        assert info["converged"]
        check_reference(mesh, MAT, CLAMP, lam, V)
        P = [c for c in calls if c[0] == "lobpcg"][-1][3][5]
        assert len(P.levels) == n_levels


def test_gmg_degree_effect():
    """Test 2: GMG degree 3, coarse_degree 2 gives the reference modes."""
    mesh = vox(7)
    opts = {"degree": 3, "coarse_degree": 2}
    lam, V, info = solve_modes(mesh, STEEL, CLAMP, 4, tol=1e-10, precond_options=opts)
    assert info["converged"]
    check_reference(mesh, STEEL, CLAMP, lam, V)


@pytest.mark.parametrize("bcs, k", [(CLAMP, 0), ({}, 6)], ids=["clamped", "free"])
def test_x0_warm_start(bcs, k):
    """Test 2: X0 = the converged V converges in <= 2 iterations (default start: more); an X0
    wider than the block size raises."""
    mesh = box()
    lam, V, info = solve_modes(mesh, MAT, bcs, 4, tol=1e-8)
    assert info["iterations"] > 5
    lam2, V2, info2 = solve_modes(mesh, MAT, bcs, 4, tol=1e-8, X0=V)
    assert info2["converged"] and info2["iterations"] <= 2 and info2["n_rigid"] == k
    np.testing.assert_allclose(lam2, lam, rtol=1e-8)
    with pytest.raises(ValueError, match="X0"):
        solve_modes(mesh, MAT, bcs, 4, X0=np.zeros((3 * len(mesh.points), 50)))


# ------------------------------------------------------------- 3: sigma_options


def scripted(monkeypatch, probes, **opts):
    """(_precond_sigma's result, builds, lobpcg kwargs, sigma_estimate kwargs) on scripted
    probes; the build refuses sigma = 0 and sigma_estimate returns 100."""
    builds, lob, est, probes = [], [], [], iter(probes)

    def build(sigma):
        if sigma == 0:
            raise SingularShiftError("free")
        builds.append(sigma)
        return object()

    def fake_est(*a, **kw):
        est.append(kw)
        return 100.0, None

    def fake_lob(*a, **kw):
        lob.append((a[3], kw))
        return [next(probes)], None, {"iterations": 2}

    monkeypatch.setattr(api, "sigma_estimate", fake_est)
    monkeypatch.setattr(api, "lobpcg", fake_lob)
    return api._precond_sigma(None, None, None, build, **opts), builds, lob, est


def test_sigma_options_scripted(monkeypatch):
    """Test 3: rounds, drop, tol, maxit and estimate_maxit change the pre-estimate as stated."""
    probes = [30.0, 9.0, 2.0, 0.5]
    assert scripted(monkeypatch, probes)[:2] == ((0.5, 8), [100.0, 30.0, 9.0, 2.0])
    assert scripted(monkeypatch, probes, rounds=2)[:2] == ((9.0, 4), [100.0, 30.0])
    assert scripted(monkeypatch, probes, drop=10)[:2] == ((30.0, 2), [100.0])
    assert scripted(monkeypatch, [40.0, 25.0])[:2] == ((40.0, 2), [100.0])
    assert scripted(monkeypatch, [40.0, 25.0], drop=2)[:2] == ((25.0, 4), [100.0, 40.0])
    _, _, lob, est = scripted(monkeypatch, probes, tol=0.3, maxit=7, estimate_maxit=4, seed=5)
    assert lob == [(0.3, dict(maxit=7, seed=5))] * 4 and est == [dict(maxit=4, seed=5)]


def test_sigma_options_real(calls):
    """Test 3: maxit 1 gives one pre-estimate iteration on a clamped box (default: more);
    estimate_maxit reaches sigma_estimate on a free one."""
    mesh = box()
    _, _, info = solve_modes(mesh, MAT, CLAMP, 2)
    assert info["sigma_iterations"] > 1
    _, _, info = solve_modes(mesh, MAT, CLAMP, 2, sigma_options={"maxit": 1})
    assert info["sigma_iterations"] == 1 and info["converged"]
    del calls[:]
    solve_modes(mesh, MAT, {}, 2, sigma_options={"estimate_maxit": 3})
    ((_, _, kw, _),) = [c for c in calls if c[0] == "sigma_estimate"]
    assert kw["maxit"] == 3


# ------------------------------------------------------------- 4: validation

BAD_KEYS = [
    (dict(operator_options={"modee": "otf"}), "modee"),
    (dict(precond_options={"degre": 3}), "degre"),
    (dict(sigma_options={"tols": 0.1}), "tols"),
    (dict(operator_options={"slab": 2}), "slab"),  # unstructured mesh
    (dict(precond="jacobi", precond_options={"degree": 2}), "degree"),
    (dict(operator_options={"mass": "lumped"}), "mass"),
    (dict(sigma=1.0, sigma_options={"tol": 0.1}), "sigma_options"),
    (dict(precond="jacobi", sigma_options={"tol": 0.1}), "sigma_options"),
    (dict(precond_options={"max_coarse": 0}), "max_coarse"),
    (dict(precond_options={"max_coarse": 2.5}), "max_coarse"),
    (dict(precond_options={"ratio": 1.0}), "ratio"),
    (dict(precond_options={"ratio": -1.0}), "ratio"),
    (dict(sigma_options={"tol": 0.0}), "tol"),
    (dict(sigma_options={"maxit": 0}), "maxit"),
    (dict(sigma_options={"rounds": 0}), "rounds"),
    (dict(sigma_options={"estimate_maxit": 0}), "estimate_maxit"),
    (dict(sigma_options={"drop": 1.0}), "drop"),
    (dict(sigma_options={"rounds": True}), "rounds"),
    (dict(precond_options={"max_coarse": True}), "max_coarse"),
]


@pytest.mark.parametrize("kw, word", BAD_KEYS)
def test_validation(kw, word):
    """Test 4: unknown or misplaced keys and bad values raise ValueError naming them; unknown
    keys list the allowed ones."""
    with pytest.raises(ValueError, match=word) as e:
        solve_modes(box((2, 2, 2)), MAT, CLAMP, 1, **kw)
    if "unknown" in str(e.value):
        assert "allowed" in str(e.value) and "solve_modes sets" in str(e.value)


def test_numpy_ints():
    """Test 4: NumPy integers pass the count checks (bool does not, above)."""
    lam, _, _ = solve_modes(
        box((4, 4, 4)), MAT, CLAMP, 2, precond_options={"max_coarse": np.int64(500),
        "degree": np.int64(3)}, sigma_options={"rounds": np.int64(2)},
    )  # fmt: skip
    assert lam.shape == (2,)
    assert type(chebyshev(None, None, 1.0, degree=np.int64(3)).degree) is int  # one jit entry


def test_validation_voxel():
    """Test 4: max_coarse with gmg and a gmg ratio of 1 raise; the key check runs before the
    operator build."""
    mesh = vox()
    with pytest.raises(ValueError, match="max_coarse"):
        solve_modes(mesh, STEEL, CLAMP, 1, precond_options={"max_coarse": 10})
    with pytest.raises(ValueError, match="ratio"):
        solve_modes(mesh, STEEL, CLAMP, 1, precond_options={"ratio": 1.0})


def test_key_check_before_build(monkeypatch):
    """Test 4: a typo fails before the operator build."""

    def fail(*a, **kw):
        raise AssertionError("built")

    monkeypatch.setattr(api, "unstructured", fail)
    with pytest.raises(ValueError, match="unknown"):
        solve_modes(box(), MAT, CLAMP, 1, precond_options={"degre": 3})


# ------------------------------------------------------------- 5: solve_modes_memory


def test_memory_validation():
    """Test 5: solve_modes_memory checks the count values like solve_modes."""
    for opts in ({"degree": "x"}, {"degree": 4.0}, {"max_coarse": 0}):
        with pytest.raises(ValueError, match="int >= 1"):
            solve_modes_memory(box((4, 4, 4)), CLAMP, 2, n_rigid=0, precond_options=opts)


def test_memory_operator_options():
    """Test 5: operator_options reach the operator estimators (composition equality)."""
    mesh = box((8, 8, 8))
    got = solve_modes_memory(mesh, CLAMP, 10, operator_options={"mode": "otf", "chunk": 50})
    ref = lobpcg_memory(
        functools.partial(unstructured_memory, mesh, mass="consistent", mode="otf", chunk=50),
        3 * len(mesh.points), 10, n_rigid=0,
        precond_memory=functools.partial(amg_memory, mesh),
    )  # fmt: skip
    assert got == ref
    v = vox(8)
    got = solve_modes_memory(v, CLAMP, 10, operator_options={"slab": 2})
    ref = lobpcg_memory(
        functools.partial(stencil_memory, v, mass="consistent", slab=2), 3 * len(v.points), 10,
        n_rigid=0, precond_memory=functools.partial(gmg_memory, v, dtype="float64"),
    )  # fmt: skip
    assert got == ref
    with pytest.raises(ValueError, match="unknown"):
        solve_modes_memory(mesh, CLAMP, 10, operator_options={"slab": 2})
    with pytest.raises(ValueError, match="unknown"):
        solve_modes_memory(mesh, CLAMP, 10, precond_options={"coarse_degree": 2})


def test_memory_precond_options():
    """Test 5: max_coarse and degree reach amg_memory, degree reaches gmg_memory; ratio and
    coarse_degree change nothing; the loop terms start at degree 3."""
    mesh = box((8, 8, 8))
    n = 3 * len(mesh.points)
    for opts in ({"max_coarse": 3000}, {"degree": 4}, {"degree": 4, "ratio": 10.0}):
        keep = {k: v for k, v in opts.items() if k != "ratio"}
        ref = lobpcg_memory(
            functools.partial(unstructured_memory, mesh, mass="consistent"), n, 10, n_rigid=0,
            precond_memory=functools.partial(amg_memory, mesh, **keep),
        )  # fmt: skip
        assert solve_modes_memory(mesh, CLAMP, 10, precond_options=opts) == ref
    v = vox(8)
    ref = lobpcg_memory(
        functools.partial(stencil_memory, v, mass="consistent"), 3 * len(v.points), 10,
        n_rigid=0, precond_memory=functools.partial(gmg_memory, v, dtype="float64", degree=4),
    )  # fmt: skip
    got = solve_modes_memory(v, CLAMP, 10, precond_options={"degree": 4, "coarse_degree": 8})
    assert got == ref
    a = {d: amg_memory(mesh, 15, degree=d) for d in (1, 2, 3)}
    assert a[1] == a[2] and a[3]["step"] - a[2]["step"] == 8 * 15 * AMG_LOOP_K
    assert a[3]["step_low"] - a[2]["step_low"] == 8 * 15 * AMG_LOOP_K
    g = {d: gmg_memory(v, 15, degree=d) for d in (2, 3)}
    b = 8 * 3 * len(v.points) * 15
    for key in ("workspace", "step", "step_low"):
        assert abs(g[3][key] - g[2][key] - GMG_LOOP["float64"] * b) <= 1
    g32 = [gmg_memory(v, 15, "float32", degree=d)["step"] for d in (2, 3)]
    assert abs(g32[1] - g32[0] - GMG_LOOP["float32"] * b) <= 1


MAX_COARSE_MESHES = {
    "hex20 12^3": lambda: box_mesh((12,) * 3, type="hex20"),
    "hex20 8^3": lambda: box_mesh((8,) * 3, type="hex20"),
    "hex8 16^3": lambda: box_mesh((16,) * 3),
}


@pytest.mark.parametrize("name", list(MAX_COARSE_MESHES))
def test_amg_memory_max_coarse(name):
    """Test 5: amg_memory's storage >= the owned device bytes at max_coarse 100-3000 (100 and 300
    fail without the floor at 1000 on hex20 12^3); the coarse term follows max(max_coarse,
    1000)."""
    mesh = MAX_COARSE_MESHES[name]()
    op, free = unstructured(mesh, MAT), dof_mask(mesh, CLAMP)
    n_c = 3 * len(np.unique(mesh.blocks[next(iter(mesh.blocks))][:, :8]))
    base = amg_memory(mesh, 15)["storage"] - 8 * min(n_c, 1000) ** 2
    for mc in (100, 300, 1000, 3000):
        est = amg_memory(mesh, 15, max_coarse=mc)["storage"]
        assert est == base + 8 * min(n_c, max(mc, 1000)) ** 2
        got = owned_bytes(amg(op, free, 1e6, mesh.points, max_coarse=mc), op)
        assert got <= est, (mc, got, est)


def amg_step_estimate(mesh, low, degree):
    n_dof = 3 * len(mesh.points)
    return lobpcg_memory(
        lambda w: unstructured_memory(mesh, w), n_dof, 10, low_memory=low,
        precond_memory=lambda w: amg_memory(mesh, w, degree=degree),
    )["step"]  # fmt: skip


@pytest.mark.parametrize("t, n", [("hex20", 8), ("hex20", 12), ("hex8", 12)])
@pytest.mark.parametrize("low", [False, True], ids=["9-block", "low"])
def test_amg_step_degree(t, n, low):
    """Test 5: the LOBPCG step with AMG at degree 4 <= its estimate (hex20 8^3 low: 1.05x
    without the loop term)."""
    mesh = box_mesh((n,) * 3, (1 / n,) * 3, type=t)
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, CLAMP)
    got = step_bytes(op, amg(op, free, 1e6, mesh.points, degree=4), low)
    assert got <= amg_step_estimate(mesh, low, 4)


def gmg_step_estimate(mesh, low, dtype, degree):
    return lobpcg_memory(
        lambda w: stencil_memory(mesh, w), 3 * len(mesh.points), 10, low_memory=low,
        precond_memory=functools.partial(gmg_memory, mesh, dtype=dtype, degree=degree),
    )["step"]  # fmt: skip


GMG_DEGREE_CASES = [
    (8, "float64"), (16, "float64"), (16, "float32"),
    pytest.param(48, "float64", marks=pytest.mark.slow),
]  # fmt: skip


@pytest.mark.parametrize("n, dtype", GMG_DEGREE_CASES)
def test_gmg_step_degree(n, dtype):
    """Test 5: the LOBPCG step with GMG at degree 4 and at coarse_degree 8 <= its estimate (both
    steps); the V-cycle temporaries at degree 4 <= gmg_memory's workspace."""
    mesh = voxel_mesh((n,) * 3, (1 / n,) * 3)
    op, free = stencil(mesh, STEEL), dof_mask(mesh, CLAMP)
    r = jax.ShapeDtypeStruct((op.n_dof, 15), jnp.float64)
    for kw, deg in (({"degree": 4}, 4), ({"coarse_degree": 8}, 2)):
        P = gmg(op, free, 1e5, dtype=dtype, **kw)
        for low in (False, True):
            assert step_bytes(op, P, low) <= gmg_step_estimate(mesh, low, dtype, deg), (kw, low)
        work = G._vcycle_apply.lower(P, r).compile().memory_analysis().temp_size_in_bytes
        assert work <= gmg_memory(mesh, 15, dtype, degree=deg)["workspace"], kw


@pytest.mark.parametrize(
    "n", [16, pytest.param(48, marks=pytest.mark.slow)]
)  # fmt: skip
def test_gmg_loop_increment(n):
    """Test 5: GMG_LOOP pinned to the measured degree-4 minus degree-2 increase of the V-cycle
    temporaries (fp64 and fp32) and, at 48^3, of both LOBPCG steps (fp64): within [0.8, 1] of
    the model for fp64 and [0.7, 1] for fp32 (measured 0.84-0.86: margin for XLA versions; below
    48^3 the low step rises up to 2.06 blocks, inside the GMG_STEP slack)."""
    mesh = voxel_mesh((n,) * 3, (1 / n,) * 3)
    op, free = stencil(mesh, STEEL), dof_mask(mesh, CLAMP)
    r = jax.ShapeDtypeStruct((op.n_dof, 15), jnp.float64)
    b = 8 * op.n_dof * 15
    for dtype in ("float64", "float32"):
        P = {d: gmg(op, free, 1e5, dtype=dtype, degree=d) for d in (2, 4)}
        work = {
            d: G._vcycle_apply.lower(p, r).compile().memory_analysis().temp_size_in_bytes
            for d, p in P.items()
        }  # fmt: skip
        diffs = [work[4] - work[2]]
        if n >= 48 and dtype == "float64":
            diffs += [step_bytes(op, P[4], low) - step_bytes(op, P[2], low) for low in (0, 1)]
        for diff in diffs:
            lo = 0.8 if dtype == "float64" else 0.7
            assert lo <= diff / (GMG_LOOP[dtype] * b) <= 1, (dtype, diff / b)


def test_amg_loop_increment():
    """Test 5: AMG_LOOP_K pinned to the measured degree-4 minus degree-2 step increase at hex20
    8^3 (both steps): within [0.6, 1] of the model."""
    mesh = box_mesh((8,) * 3, (1 / 8,) * 3, type="hex20")
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, CLAMP)
    P = {d: amg(op, free, 1e6, mesh.points, degree=d) for d in (2, 4)}
    for low in (False, True):
        diff = step_bytes(op, P[4], low) - step_bytes(op, P[2], low)
        assert 0.6 <= diff / (8 * 15 * AMG_LOOP_K) <= 1, (low, diff / 120)


# ------------------------------------------------------------- 6: option sets pinned

LOBPCG_FORWARDED = {
    "op", "free", "n_modes", "tol", "R", "precond", "X0", "maxit", "seed", "chunk", "guards",
    "basis_dtype", "low_memory",
}  # fmt: skip


def params(f):
    return set(inspect.signature(f).parameters)


def test_option_sets_pinned():
    """Test 6: each builder's parameters minus the reserved ones equal the allowed keys, so a
    new builder option fails here until it is exposed or reserved."""
    builders = dict(
        unstructured=unstructured, stencil=stencil, gmg=gmg, amg=amg, jacobi=jacobi
    )  # fmt: skip
    allowed = {**api.OPERATOR_OPTIONS, **api.PRECOND_OPTIONS}
    for name, f in builders.items():
        assert params(f) - api.RESERVED[name] == set(allowed[name]), name
    assert params(lobpcg) == LOBPCG_FORWARDED
    assert params(LOB.lobpcg) <= params(solve_modes) | {"op", "free", "R", "precond"}
    assert params(api._precond_sigma) - {"op", "free", "R", "build", "seed"} == set(
        api.SIGMA_OPTIONS
    )
    assert params(count) - api.RESERVED["count"] - {"_steps", "_predict"} == set(api.COUNT_OPTIONS)
    # slice_options: si_band's own minus what solve_modes sets, plus si_slice's inner options
    band_set = {"op", "free", "lo", "hi", "make", "R", "verify", "count_options", "tol", "seed",
                "n_store", "slice_options"}  # fmt: skip
    slice_set = {"op", "free", "lo", "hi", "precond", "capacity", "sigma", "R", "W", "lam_W",
                 "tol", "block", "seed", "verbose"}  # fmt: skip
    inner = params(SI.si_slice) - slice_set
    assert inner == {"tau", "maxit_inner"}
    assert (params(SI.si_band) - band_set) | inner == set(api.SLICE_OPTIONS)


# ------------------------------------------------------------- 7: Chebyshev loop


def unrolled(S, r):
    """The pre-13c Python-unrolled Chebyshev iteration (the reference)."""
    hi = CHEB.SAFETY * S.lmax
    lo = hi / S.ratio
    theta, delta = (hi + lo) / 2, (hi - lo) / 2
    sigma = theta / delta
    rho = 1 / sigma
    d = S.precond(r) / theta
    x = d
    for _ in range(S.degree - 1):
        r = r - S.A(d)
        rho_new = 1 / (2 * sigma - rho)
        d = rho_new * rho * d + 2 * rho_new / delta * S.precond(r)
        x, rho = x + d, rho_new
    return x


def test_chebyshev_loop():
    """Test 7: the loop equals the unrolled iteration to 1e-14 (degrees 1-5); ratio <= 1 raises."""
    mesh = box_mesh((4, 4, 4), type="hex20")
    op, free = unstructured(mesh, MAT), dof_mask(mesh, CLAMP)
    A, D = ap_operator(op, free, 0.8), block_jacobi(op, free, 0.8)
    r = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, 3)) * free[:, None])
    lmax = lmax_estimate(A, D)
    for degree in range(1, 6):
        S = chebyshev(A, D, lmax, degree)
        ref = unrolled(S, r)
        assert jnp.linalg.norm(S(r) - ref) <= 1e-14 * jnp.linalg.norm(ref), degree
    for ratio in (1.0, 0.5):
        with pytest.raises(ValueError, match="ratio"):
            chebyshev(A, D, 1.0, ratio=ratio)


def test_chebyshev_loop_memory():
    """Test 7: AMG V-cycle temporaries at degree 4 <= those at degree 2 (hex20 16^3: 18.2 against
    19.1 blocks; at 8^3 the loop's fixed buffers dominate block units)."""
    mesh = box_mesh((16,) * 3, (1 / 16,) * 3, type="hex20")
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, CLAMP)
    r = jax.ShapeDtypeStruct((op.n_dof, 15), jnp.float64)
    t = [
        G._vcycle_apply.lower(amg(op, free, 1e6, mesh.points, degree=d), r)
        .compile().memory_analysis().temp_size_in_bytes
        for d in (2, 4)
    ]  # fmt: skip
    assert t[1] <= t[0], t


def test_chebyshev_loop_fp32():
    """Test 7: the fp32 GMG V-cycle (coarse_degree 4 runs the loop) stays within 1e-5 of the
    fp64 one at degree 4, and the loop keeps the fp32 dtype."""
    mesh = vox(8)
    op, free = stencil(mesh, STEEL), dof_mask(mesh, CLAMP)
    P64 = gmg(op, free, 0.0, degree=4)
    P32 = gmg(op, free, 0.0, degree=4, dtype="float32")
    U = jnp.asarray(np.random.default_rng(1).standard_normal((op.n_dof, 3)) * free[:, None])
    Y64, Y32 = P64(U), P32(U)
    assert jnp.linalg.norm(Y32 - Y64) <= 1e-5 * jnp.linalg.norm(Y64)
    S = P32.levels[0][1]
    u32 = U.astype(jnp.float32)
    assert S(u32).dtype == jnp.float32
    ref = unrolled(S, u32)
    assert jnp.linalg.norm(S(u32) - ref) <= 1e-5 * jnp.linalg.norm(ref)


@pytest.mark.slow
def test_options_study_smoke():
    """benchmarks/options_study.run(small=True) records every entry with converged solves."""
    from benchmarks import options_study

    r = options_study.run(small=True)
    assert r["x0"]["warm"] < r["x0"]["cold"] and r["jacobi"]["amg"] < r["jacobi"]["jacobi"]
    for key in ("degree_amg", "degree_gmg"):
        assert [d["degree"] for d in r[key]] == [2, 3, 4] and all(d["converged"] for d in r[key])
    for key in ("vcycle_amg", "vcycle_gmg_float64", "vcycle_gmg_float32"):
        assert [d["degree"] for d in r[key]] == [2, 4] and all(d["temp_blocks"] > 0 for d in r[key])
