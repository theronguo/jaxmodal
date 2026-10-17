"""PR 14b: the [0, f_max] band mode with verify (docs/plans/pr-14b-band.md, tests 1-14)."""

import functools
import importlib
import math
import sys

import numpy as np
import pytest
import scipy.linalg as sla

from jaxmodal.eigen import counting, solve_modes
from jaxmodal.eigen.lobpcg import _band_count, _capacity
from jaxmodal.materials import Material, isotropic
from jaxmodal.memory import (
    count_memory,
    lobpcg_memory,
    solve_modes_memory,
    unstructured_memory,
)
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import assemble, dof_mask, unstructured
from jaxmodal.precond import amg

api = importlib.import_module("jaxmodal.eigen.api")
memory = importlib.import_module("jaxmodal.memory")

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
CLAMP = {"xmin": "xyz"}
TOL = 1e-6


def freq(lam):
    return math.sqrt(lam) / (2 * math.pi)


MESHES = {
    "clamped": (lambda: box_mesh((6, 5, 4), (1 / 6, 1 / 5, 1 / 4), type="hex20"), CLAMP),
    "free": (lambda: box_mesh((4, 3, 3), (1 / 4, 1 / 3, 1 / 3), type="hex8"), {}),
    "beam": (lambda: box_mesh((8, 2, 2), (1 / 8, 1 / 16, 1 / 16), type="hex20"), CLAMP),
    "tiny": (lambda: box_mesh((2, 2, 2), (1 / 2,) * 3, type="hex8"), CLAMP),
    "voxel": (lambda: voxel_mesh((6, 5, 4), (1 / 6,) * 3), {"zmin": "xyz"}),
}


@functools.cache
def ref(name):
    """mesh, bcs, M (assembled) and the dense elastic eigenvalues of the free block."""
    make, bcs = MESHES[name]
    mesh = make()
    free = dof_mask(mesh, bcs)
    K, M = assemble(mesh, MAT)
    lam = sla.eigh(K.toarray()[np.ix_(free, free)], M.toarray()[np.ix_(free, free)],
                   eigvals_only=True)  # fmt: skip
    n_rigid = 0 if bcs else 6
    return mesh, bcs, M, lam[n_rigid:]


def mid(name, k):
    """f_max halfway (geometric) between elastic modes k and k + 1 (1-based)."""
    lam = ref(name)[3]
    return freq(math.sqrt(lam[k - 1] * lam[k]))


@pytest.fixture
def spy(monkeypatch):
    """Records every count call in solve_modes (args, kwargs) and every lam_c build; drop:
    elastic mode indices to remove from W before the real count runs."""
    rec = dict(count=[], build=[], drop=())
    real_count, real_amg, real_gmg = api.count, api.amg, api.gmg

    def count(op, free, W, lam, *a, **kw):
        keep = np.setdiff1d(np.arange(W.shape[1]), rec["drop"])
        rec["count"].append((np.array(W), lam, a, kw))  # a copy: no reference to the result
        return real_count(op, free, W[:, keep], lam[keep], *a, **kw)

    def amg(op, free, s, *a, **kw):
        if "aggregates" in kw:
            rec["build"].append(s)
        return real_amg(op, free, s, *a, **kw)

    def gmg(op, free, s, *a, **kw):
        rec["build"].append(s)
        return real_gmg(op, free, s, *a, **kw)

    monkeypatch.setattr(api, "count", count)
    monkeypatch.setattr(api, "amg", amg)
    monkeypatch.setattr(api, "gmg", gmg)
    return rec


@pytest.fixture
def budgets(monkeypatch):
    """Records the n_modes of every solve_modes_memory call and every probe_block budget."""
    rec = dict(n_modes=[], budget=[])
    real_smm, real_pb = memory.solve_modes_memory, memory.probe_block

    def smm(mesh, bcs, n_modes=None, **kw):
        rec["n_modes"].append(n_modes)
        return real_smm(mesh, bcs, n_modes, **kw)

    def pb(budget, *a, **kw):
        rec["budget"].append(budget)
        return real_pb(budget, *a, **kw)

    monkeypatch.setattr(memory, "solve_modes_memory", smm)
    monkeypatch.setattr(memory, "probe_block", pb)
    return rec


def no_build(monkeypatch):
    """Any lam_c build fails the test."""

    def fail(*a, **kw):
        raise AssertionError("lam_c preconditioner built")

    real = api.amg
    monkeypatch.setattr(
        api, "amg", lambda *a, **kw: fail() if "aggregates" in kw else real(*a, **kw)
    )
    monkeypatch.setattr(api, "jacobi", fail)


# ---------------------------------------------------------------- 1: the band rule


def test_band_count():
    """Test 1: edge, the 100 t neighbour window, a cluster chain, all and none in the band."""
    lam, t = np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 1e-6)
    assert _band_count(lam, t, 2.5) == 2
    assert _band_count(lam, t, 2.0) == 2  # exact edge
    assert _band_count(lam, t, 3.0 / (1 + 50e-6)) == 3  # inside the window
    assert _band_count(lam, t, 3.0 / (1 + 200e-6)) == 2  # outside
    chain = np.array([1.0, 2.0, 2 * (1 + 60e-6), 2 * (1 + 120e-6), 3.0])
    assert _band_count(chain, np.full(5, 1e-6), 2.0) == 4  # each gap < 100 t
    assert _band_count(lam, t, 0.5) == 0
    assert _band_count(lam, t, 10.0) == 4


# ---------------------------------------------------------------- 2, 3: no missed modes, wiring


def check_band(name, f_max, spy, expect, **kw):
    """Band solve against the dense inertia: the returned count, lam, the count's edges, W
    (every converged column, M-orthonormal), R and the lam_c build."""
    mesh, bcs, M, lr = ref(name)
    lam, V, info = solve_modes(mesh, MAT, bcs, band=(0, f_max), tol=TOL, **kw)
    lam_stop = (2 * math.pi * f_max) ** 2
    assert len(lam) == expect == len(info["residuals"]) == len(info["frequencies"])
    assert info["converged"] and info["n_beyond"] >= 1
    assert V.shape == (3 * len(mesh.points), expect) and V.base is None  # owning, compacted
    assert V.flags.f_contiguous and V.nbytes == 8 * V.shape[0] * expect
    if expect:
        assert np.abs(lam / lr[:expect] - 1).max() <= 10 * TOL
    c = info["count"]
    if not c["ran"]:
        return lam, info
    W, lam_w, (lam_last, lam_next), ckw = spy["count"][-1]
    assert W.shape[1] == len(lam) + info["n_beyond"] == len(lam_w)
    Wf = np.asarray(W)
    assert np.abs(Wf.T @ (M @ Wf) - np.eye(W.shape[1])).max() <= 1e-10
    assert lam_last == (lam[-1] if expect else lam_stop) and lam_next == lam_w[expect]
    lc = ((lr[expect - 1] if expect else lam_stop) + lr[expect]) / 2
    assert abs(c["lam_c"] / lc - 1) <= 10 * TOL
    g = (lr[expect] - (lr[expect - 1] if expect else lam_stop)) / lc
    assert abs(c["gap"] / g - 1) <= 10 * TOL / g
    assert (ckw["R"] is None) == bool(MESHES[name][1])
    if ckw["R"] is not None:
        assert ckw["R"].shape[1] == 6
    if info["precond"] != "jacobi":  # built at lam_c (AMG: from the solve's aggregates)
        assert spy["build"]
    if spy["build"]:
        assert spy["build"][-1] == pytest.approx(c["lam_c"], rel=1e-14)
    return lam, info


@pytest.mark.parametrize("name,k", [("clamped", 3), ("free", 2)])
def test_clear_gap(spy, name, k):
    """Tests 2, 3: f_max in a clear gap, verify=True: k modes, the count wired and passing."""
    lam, info = check_band(name, mid(name, k), spy, k, verify=True)
    assert info["count"]["ran"] and info["count"]["passed"]


def test_window():
    """Test 2: f_max 10 x inside / outside the 100 t window below mode 4 (thresholds from a
    first solve; mode 3 far below)."""
    mesh, bcs, _, lr = ref("clamped")
    t4 = solve_modes(mesh, MAT, bcs, 4, tol=TOL)[2]["thresholds"][3]
    for f, expect in ((10, 4), (1000, 3)):
        lam_stop = lr[3] / (1 + f * t4)
        assert lam_stop > lr[2] * (1 + 1000 * TOL)
        lam = solve_modes(mesh, MAT, bcs, band=(0, freq(lam_stop)), tol=TOL, verify=False)[0]
        assert len(lam) == expect


def test_chunks(spy):
    """Test 2: 19 modes in chunks of 8 (two locked, the final one stops), verify=False."""
    opts = {"degree": 4, "n_probes": 2}  # cheap: the W check of every converged column runs
    lam, info = check_band("clamped", mid("clamped", 19), spy, 19, chunk=8, verify=True,
                           count_options=opts)  # fmt: skip
    assert len(info["chunks"]) == 3 and info["chunk"] == 8 and info["count"]["ran"]


def test_voxel(spy):
    """Test 13: the clear-gap case on a voxel box with GMG."""
    lam, info = check_band("voxel", mid("voxel", 3), spy, 3, verify=True)
    assert info["precond"] == "gmg" and info["count"]["passed"]


# ---------------------------------------------------------------- 4-6: roadmap cases


@pytest.mark.parametrize("name", ["clamped", "free"])
def test_below_first(spy, name):
    """Test 4: f_max at half the first elastic frequency: no modes, the count verifiable < 0.5."""
    lam, info = check_band(name, freq(ref(name)[3][0]) / 2, spy, 0, verify=True)
    c = info["count"]
    assert lam.shape == (0,) and c["verifiable"] and c["estimate"] < 0.5


def test_detection(spy):
    """Test 5: W missing elastic mode 1 (not next to the cut): verify=True counts it."""
    spy["drop"] = (0,)
    _, info = check_band("clamped", mid("clamped", 3), spy, 3, verify=True)
    assert info["count"]["verifiable"] and info["count"]["estimate"] >= 0.5


@pytest.mark.parametrize("chunk", [None, 1])
def test_double(spy, chunk):
    """Test 6: f_max at a double eigenvalue (square beam): both copies, also with a chunk
    ending between them; the count < 0.5."""
    lr = ref("beam")[3]
    assert (lr[1] - lr[0]) / lr[0] < 1e-8 and lr[2] > 10 * lr[1]
    lam, info = check_band("beam", freq(lr[0]), spy, 2, verify=True, chunk=chunk)
    assert info["count"]["verifiable"] and info["count"]["estimate"] < 0.5


# ---------------------------------------------------------------- 7: verify


def band_solve(f_max=None, **kw):
    mesh, bcs, _, _ = ref("clamped")
    return solve_modes(mesh, MAT, bcs, band=(0, f_max or mid("clamped", 3)), tol=TOL, **kw)


def test_count_verbose(monkeypatch, capsys):
    """PR 15b test 11: count_options verbose reaches count: one progress line per step
    (PRINT_S = 0) and one per block."""
    monkeypatch.setattr(counting, "PRINT_S", 0.0)
    opts = {"degree": 4, "n_probes": 2, "verbose": True}
    info = band_solve(verify=True, count_options=opts)[2]
    out = capsys.readouterr().out
    assert info["count"]["ran"] and out.count("count: step") == 4
    assert out.count("count: block 1 of 1, 4 steps, done") == 1


def test_verify_false(monkeypatch, spy):
    """Test 7: False neither builds nor counts and reports the steps and the bound; a looser
    tau gives factor 0.3."""
    no_build(monkeypatch)
    for opts, c in ((None, 2.0), ({"tau": 1e-4}, 0.3)):
        _, _, info = band_solve(verify=False, count_options=opts)
        ci = info["count"]
        assert not ci["ran"] and ci["reason"] == "verify=False" and not spy["count"]
        b, d = ci["block"], ci["degree"]
        assert ci["predicted_steps"] == math.ceil(50 / b) * d
        m = info["chunk"] + max(5, info["chunk"] // 5)
        assert ci["predicted_s"] == pytest.approx(
            c * ci["predicted_steps"] * b / m * info["t_iteration"], rel=1e-12
        )
        assert ci["predicted_kind"] == "bound"


def test_verify_true(spy):
    """Test 7: True runs without calibration."""
    _, _, info = band_solve(verify=True)
    ci = info["count"]
    assert ci["ran"] and ci["calibration_s"] == 0 and ci["predicted_kind"] == "bound"


def test_verify_none(monkeypatch, spy):
    """Test 7: None runs or skips on the calibrated prediction; skips without building when the
    bound exceeds the solve; predicted_s = build + calibration + n_blocks (d t_step + t_block)."""
    monkeypatch.setattr(counting, "_bound", lambda *a: 0.0)
    monkeypatch.setattr(counting, "_calibrate", lambda *a: (0.0, 0.0, 0.0, 0.0))
    ci = band_solve()[2]["count"]
    assert ci["ran"] and ci["predicted_kind"] == "calibrated"
    monkeypatch.setattr(counting, "_calibrate", lambda *a: (1e9, 0.0, 0.0, 0.0))
    ci = band_solve()[2]["count"]
    assert not ci["ran"] and ci["reason"] == "predicted cost above the solve"
    monkeypatch.undo()
    monkeypatch.setattr(counting, "_bound", lambda *a: 0.0)
    ci = band_solve()[2]["count"]  # the real calibration
    nb = math.ceil(50 / ci["block"])
    pred = ci["build_s"] + ci["calibration_s"]
    pred += nb * (ci["degree"] * ci["calibrated_step"] + ci["calibrated_block"])
    assert ci["predicted_s"] == pytest.approx(pred, rel=1e-12) and ci["calibration_s"] > 0
    monkeypatch.setattr(counting, "_bound", lambda *a: 1e9)
    no_build(monkeypatch)
    ci = band_solve()[2]["count"]
    assert not ci["ran"] and ci["reason"] == "predicted cost (a lower bound) above the solve"


def test_no_warm_time(monkeypatch, spy):
    """Test 7: without a warm iteration time the bound is None: False reports 'unknown', None
    goes to the calibration."""
    assert counting._bound(10, 5, 10, 0.0, None, 1e-6) is None
    real = api._solve

    def solve(*a, **kw):
        lam, V, info = real(*a, **kw)
        info["t_iteration"] = 0.0
        return lam, V, info

    monkeypatch.setattr(api, "_solve", solve)
    ci = band_solve(verify=False)[2]["count"]
    assert ci["predicted_s"] is None and ci["predicted_kind"] == "unknown"
    monkeypatch.setattr(counting, "_calibrate", lambda *a: (1e9, 0.0, 0.0, 0.0))
    ci = band_solve()[2]["count"]
    assert ci["reason"] == "predicted cost above the solve"


def test_capped(monkeypatch, spy):
    """Test 7: a degree above MAX_DEGREE: None skips without building, True returns NaN."""
    monkeypatch.setattr(counting, "MAX_DEGREE", 4)
    no_build(monkeypatch)
    ci = band_solve()[2]["count"]
    assert not ci["ran"] and ci["degree"] is None and "MAX_DEGREE" in ci["reason"]
    ci = band_solve(verify=True)[2]["count"]
    assert ci["ran"] and np.isnan(ci["estimate"]) and not ci["verifiable"]


def test_does_not_fit(monkeypatch, spy):
    """Test 7: a budget below one probe: None skips with the reason, True keeps the modes,
    reports ran False and warns."""

    def raise_(*a, **kw):
        raise ValueError("budget below one probe")

    monkeypatch.setattr(memory, "probe_block", raise_)
    lam, _, info = band_solve()
    assert len(lam) == 3 and not info["count"]["ran"] and "does not fit" in info["count"]["reason"]
    with pytest.warns(UserWarning, match="does not fit"):
        lam, _, info = band_solve(verify=True)
    assert len(lam) == 3 and not info["count"]["ran"] and not spy["count"]


@pytest.mark.parametrize("chunk", [None, 8])
def test_not_converged(spy, chunk):
    """A failed band solve (maxit 3): converged False, no count, a warning under verify=True."""
    mesh, bcs, _, _ = ref("clamped")
    with pytest.warns(UserWarning, match="not converged"):
        lam, _, info = solve_modes(mesh, MAT, bcs, band=(0, mid("clamped", 19)), tol=TOL,
                                   maxit=3, chunk=chunk, verify=True)  # fmt: skip
    assert not info["converged"] and len(lam) <= 19 and not spy["count"]
    assert info["count"] == {"ran": False, "reason": "solve not converged"}


@pytest.mark.slow
def test_calibration(monkeypatch):
    """Test 7b: 50 steps on hex20 12^3: the calibrated count time within [0.5, 2] x the
    actual one, and the bound below the warm recurrence time."""
    rec = {}
    real = counting._calibrate

    def calibrate(*a):
        rec["out"] = real(*a)
        return (0.0, 0.0) + rec["out"][2:]  # force the run

    monkeypatch.setattr(counting, "_calibrate", calibrate)
    monkeypatch.setattr(counting, "_bound", lambda *a: rec.update(bound=api_bound(*a)) or 0.0)
    mesh = box_mesh((12,) * 3, (1 / 12,) * 3, type="hex20")
    f1 = solve_modes(mesh, MAT, CLAMP, 4)[2]["frequencies"]
    opts = {"degree": 50, "block": 10, "n_probes": 10}
    _, _, info = solve_modes(mesh, MAT, CLAMP, band=(0, (f1[2] + f1[3]) / 2), count_options=opts)
    ci = info["count"]
    assert ci["ran"] and ci["steps"] == 50
    pred, _, t_step, _ = rec["out"]
    actual = ci["time"]["total"]
    print(f"calibrated {pred:.2f} s, actual {actual:.2f} s, bound {rec['bound']:.2f} s")
    assert 0.5 * actual <= pred <= 2 * actual
    assert rec["bound"] <= 50 * t_step


api_bound = counting._bound


# ---------------------------------------------------------------- 8, 9: growth, the edge error


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_growth(budgets, dtype):
    """Test 8: chunk 8 from an expected count of 8: the basis grows, and lam, V equal a run at
    full capacity; the count runs on the grown solve, its budget at the grown capacity."""
    mesh, bcs, _, _ = ref("clamped")
    kw = dict(band=(0, mid("clamped", 30)), tol=TOL, chunk=8, basis_dtype=dtype)
    lam, V, info = solve_modes(mesh, MAT, bcs, n_modes=8, verify=True,
                               count_options={"degree": 4, "n_probes": 2}, **kw)  # fmt: skip
    lam2, V2, info2 = solve_modes(mesh, MAT, bcs, n_modes=40, verify=False, **kw)
    assert len(lam) == 30 and len(info["chunks"]) == 4
    assert 0 < info["capacity"] < info2["capacity"]
    assert np.abs(lam - lam2).max() <= 1e-12 * lam.max()
    assert np.abs(V - V2).max() <= 1e-12 * np.abs(V2).max()
    assert info["count"]["ran"] and info["count"]["verifiable"]
    m = 8 + 5  # chunk + guards
    assert budgets["n_modes"][0] == info["capacity"] - m > 8
    assert _capacity(info["capacity"] - m, 8, m, False)[1] == info["capacity"]


@pytest.mark.parametrize("low", [False, True])
def test_growth_from_zero(budgets, low):
    """Test 8: n_modes 4 below chunk 8 (no basis at first): the first lock grows the basis for
    the next lock too, and the budget's expected count maps back to the grown capacity."""
    mesh, bcs, _, lr = ref("clamped")
    lam, _, info = solve_modes(mesh, MAT, bcs, n_modes=4, chunk=8, band=(0, mid("clamped", 12)),
                               tol=TOL, low_memory=low, verify=True,
                               count_options={"degree": 4, "n_probes": 2})  # fmt: skip
    assert len(lam) == 12 and np.abs(lam / lr[:12] - 1).max() <= 10 * TOL
    n_eff = budgets["n_modes"][0]
    assert 8 < n_eff and _capacity(n_eff, 8, 13, low)[1] == info["capacity"] > 0


def test_edge_above():
    """Test 9: f_max above the spectrum raises; the tiny-mesh chunk default and the chunk from
    n_modes."""
    mesh, bcs, _, lr = ref("tiny")
    with pytest.raises(ValueError, match="more modes than the mesh provides"):
        solve_modes(mesh, MAT, bcs, band=(0, 2 * freq(lr[-1])), verify=False)
    lam, _, info = solve_modes(mesh, MAT, bcs, band=(0, freq(lr[0]) / 2), verify=False)
    assert len(lam) == 0 and info["chunk"] == 40
    info = solve_modes(mesh, MAT, bcs, n_modes=5, band=(0, freq(lr[0]) / 2), verify=False)[2]
    assert info["chunk"] == 9


# ---------------------------------------------------------------- 10-12: block, errors, memory


def test_probe_block(monkeypatch, budgets):
    """Test 10: the probe block is the largest that fits the solve's estimate minus the host W
    (and with AMG minus the kept aggregates); count_options block overrides; unknown keys raise
    before any build."""
    mesh, bcs, _, _ = ref("clamped")
    band = (0, mid("clamped", 3))
    opts = {"n_probes": 100_000}
    lam, _, info = solve_modes(mesh, MAT, bcs, band=band, precond="jacobi", verify=False,
                               count_options=opts)  # fmt: skip
    n, k_all, b = 3 * len(mesh.points), len(lam) + info["n_beyond"], info["count"]["block"]
    budget = solve_modes_memory(mesh, bcs, max(160, k_all), chunk=info["chunk"],
                                precond="jacobi", band=band)["total"] - 8 * n * k_all  # fmt: skip

    def total(b):
        return count_memory(lambda m: unstructured_memory(mesh, m), n, b)["total"]

    assert b < 100_000 and total(b) <= budget < total(b + 1)
    lam, _, info = solve_modes(mesh, MAT, bcs, band=band, verify=False)  # AMG
    k_all = len(lam) + info["n_beyond"]
    op, free = unstructured(mesh, MAT), dof_mask(mesh, bcs)
    aggs = amg(op, free, info["sigma"], mesh.points, nonsingular=True, return_aggregates=True)[1]
    est = solve_modes_memory(mesh, bcs, 160, chunk=info["chunk"], band=band)["total"]
    assert api._nbytes(aggs) > 0
    assert budgets["budget"][-1] == est - 8 * n * k_all - api._nbytes(aggs)
    info = solve_modes(mesh, MAT, bcs, band=band, verify=False, count_options={"block": 3})[2]
    assert info["count"]["block"] == 3
    no_build(monkeypatch)
    monkeypatch.setattr(api, "unstructured", lambda *a, **kw: pytest.fail("built"))
    with pytest.raises(ValueError, match="unknown count option"):
        solve_modes(mesh, MAT, bcs, band=band, count_options={"probes": 3})


@pytest.mark.parametrize(
    "kw,err,match",
    [
        ({}, ValueError, "pass n_modes or band"),
        ({"band": (2.0, 1.0)}, ValueError, "f1 < f2"),
        ({"band": (0.0, 0.0)}, ValueError, "f1 < f2"),
        ({"band": (0.0, -1.0)}, ValueError, "f1 < f2"),
        ({"band": (-1.0, 2.0)}, ValueError, "f1 >= 0"),
        ({"band": (0.0, np.inf)}, ValueError, "finite"),
        ({"band": 3.0}, ValueError, "pair"),
        ({"band": (0.0, 1.0), "verify": "yes"}, ValueError, "verify must be"),
        ({"band": (0.0, 1.0), "count_options": {"tau": 0}}, ValueError, "tau must be > 0"),
        ({"band": (0.0, 1.0), "count_options": {"block": 0}}, ValueError, "block must be"),
        ({"band": (0.0, 1.0), "count_options": {"verbose": 1}}, ValueError, "verbose must"),
        ({"n_modes": 3, "verify": True}, ValueError, "need band"),
    ],
)
def test_errors(kw, err, match):
    """Test 11: argument errors, before any build."""
    mesh, bcs, _, _ = ref("tiny")
    with pytest.raises(err, match=match):
        solve_modes(mesh, MAT, bcs, **kw)


def test_fp64_count_model():
    """Test 10: after an fp32 V-cycle solve the probe block is sized with the fp64 multigrid
    the count builds at lam_c: a budget that holds one probe under the fp32 model raises."""
    mesh, bcs = voxel_mesh((16,) * 3, (1 / 16,) * 3), {"zmin": "xyz"}
    n, band, kw = 3 * len(mesh.points), (0, 1.0), dict(precond="gmg", precond_dtype="float32")
    total = solve_modes_memory(mesh, bcs, 10, band=band, n_rigid=0, **kw)["total"]

    def one(dtype):
        pc = lambda m: memory.gmg_memory(mesh, m, dtype=dtype)  # noqa: E731
        return count_memory(lambda m: memory.stencil_memory(mesh, m), n, 1, voxel=True,
                            precond_memory=pc)["total"]  # fmt: skip

    assert one("float32") < one("float64")
    h = total - one("float32")  # host bytes leaving exactly one fp32-model probe
    with pytest.raises(ValueError, match="below one probe"):
        memory.band_probe_block(mesh, bcs, 10, 50, h, n_rigid=0, band=band, **kw)
    h = total - one("float64")
    assert memory.band_probe_block(mesh, bcs, 10, 50, h, n_rigid=0, band=band, **kw) == 1


def test_memory():
    """Test 12: the band estimate is lobpcg_memory at the band capacity (one more host block
    than the plain solve)."""
    mesh, bcs, _, _ = ref("clamped")
    n = 3 * len(mesh.points)
    band = solve_modes_memory(mesh, bcs, 10, band=(0, 1.0))
    plain = solve_modes_memory(mesh, bcs, 10, chunk=15)
    est = lobpcg_memory(lambda m: unstructured_memory(mesh, m), n, 10, 15, band=True,
                        precond_memory=lambda m: memory.amg_memory(mesh, m))  # fmt: skip
    assert band == est and band["total"] - plain["total"] == 8 * n * 20
    with pytest.raises(ValueError, match="pass n_modes or band"):
        solve_modes_memory(mesh, bcs)


PEAK_SETUP = """
from jaxmodal import config
config.enable_x64()
from jaxmodal.eigen import solve_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import voxel_mesh
mats = {0: Material(isotropic(3.0, 0.3), 2.0)}
mesh = voxel_mesh((S, S, S), (1 / S,) * 3)
kw = dict(n_modes=10, band=(0, F), verify=True, count_options={"degree": 3}, mass="lumped")
solve_modes(voxel_mesh((8, 8, 8), (1 / 8,) * 3), mats, {"xmin": "xyz"}, **kw)  # warm-up
"""
PEAK = """
solve_modes(mesh, mats, {"xmin": "xyz"}, **kw)
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_peak():
    """Test 12: the measured process peak of a band solve with verify=True within 20% of the
    estimate (>= 2 GB)."""
    from benchmarks import harness

    s = 48  # chunk 15 from n_modes = 10: a block inside the memory model's fit (15-48)
    mesh = voxel_mesh((s, s, s), (1 / s,) * 3)
    f = solve_modes(voxel_mesh((8, 8, 8), (1 / 8,) * 3), MAT, CLAMP, 4, mass="lumped")[2]
    f_max = 0.6 * f["frequencies"][0]  # below the first mode (same unit cube)
    e = solve_modes_memory(mesh, CLAMP, 10, band=(0, f_max), mass="lumped")["total"]
    setup = PEAK_SETUP.replace("S", str(s)).replace("F", repr(f_max))
    peak = harness.measure(PEAK, setup)["peak_bytes"]
    print(f"band {s}^3: estimate {e / 1e9:.2f} GB, peak {peak / 1e9:.2f}")
    assert e >= 2e9 and abs(e - peak) <= 0.2 * peak
