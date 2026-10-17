"""PR 15d: the band strategy and band=(f1, f2) (docs/plans/pr-15d-strategy.md, tests 1-17).
Real counts use the dense exact (K + s M)^-1 for every multigrid build (fake_builds); budgets
come from the searches under the fit test."""

import gc
import importlib
import math
import sys
import types
import warnings
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_shift_invert as tsi
from test_shift_invert import (
    EPS,
    MAT,
    MESHES,
    TAU,
    cb_edges,
    edges,
    exact,
    fake_band,
    lam_hat,
    prob,
    ref_slice,
    spy_slice,
)

from jaxmodal.eigen import counting, solve_modes
from jaxmodal.eigen.lobpcg import _band_count
from jaxmodal.precond import SingularShiftError

api = importlib.import_module("jaxmodal.eigen.api")
lob = importlib.import_module("jaxmodal.eigen.lobpcg")
si = importlib.import_module("jaxmodal.eigen.shift_invert")
mem = importlib.import_module("jaxmodal.memory")

pytestmark = [pytest.mark.filterwarnings("error:returned modes copied"),
              pytest.mark.filterwarnings("error:si_band. [0-9]+ stored modes exceed")]
TOL = 1e-8


def freq(lam):
    return math.sqrt(lam) / (2 * math.pi)


def fband(name, i0, i1):
    """(f1, f2) of edges(lr, i0, i1): the gap midpoints around modes i0 .. i1 - 1."""
    lo, hi = edges(prob(name)[6], i0, i1)
    return freq(lo), freq(hi)


def run(name, band, **kw):
    mesh = prob(name)[0]
    return solve_modes(mesh, MAT, MESHES[name][1], band=band, **({"tol": TOL} | kw))


def models(name, precond="auto", dtype="float64"):
    return mem._models(prob(name)[0], "consistent", precond, dtype, None, None)


def need(est, s=mem.FIT_MODEL, extra=0):
    """The budget at which est passes the fit test with factor s on the modelled part (CPU)."""
    e = est["device_exact"] + est["host_exact"]
    return mem.FIT_EXACT * (e + extra) + s * (est["total"] - e)


def lob_est(name, chunk, C, n_rigid=0, **kw):
    op, pc = models(name, **kw)
    n = prob(name)[2].n_dof
    return mem.lobpcg_memory(op, n, C, chunk, None, False, "float64", n_rigid, pc, True,
                             capacity=C)  # fmt: skip


def n_avail(name):
    free, R = prob(name)[1], prob(name)[5]
    return int(free.sum()) - (0 if R is None else R.shape[1])


def c_max(name, chunk, budget, **kw):
    op, pc = models(name)
    return mem._c_max(op, pc, prob(name)[2].n_dof, chunk=chunk, guards=None, low=False,
                      basis_dtype="float64", n_rigid=0, n_avail=n_avail(name), budget=budget,
                      host_budget=None, **kw)  # fmt: skip


def check(name, f1, f2, lam, V, info):
    """The roadmap's checks against eigsh: the reference modes in [lam_lo, lam_hi] (all of them,
    plus at most cluster copies within 100 t of an edge), each to max(1e-8, 10 eps lam_hat /
    lam) when alone (relative gaps >= 1e-2), else 10 t (its own threshold); V M-orthonormal,
    owning and F-contiguous with exactly the band's columns."""
    _, free, _, K, M, _, lr, _ = prob(name)
    lo, hi = (2 * np.pi * f1) ** 2, (2 * np.pi * f2) ** 2
    ref = lr[(lr >= lo) & (lr <= hi)]
    assert ((lam >= lo) & (lam <= hi)).sum() == len(ref) and len(lam) == len(info["residuals"])
    gap = np.diff(lr) / lr[1:]
    t = np.broadcast_to(info["thresholds"], lam.shape)
    for x, ti in zip(lam, t, strict=True):
        i = int(np.argmin(np.abs(lr - x)))
        alone = (i == 0 or gap[i - 1] >= 1e-2) and (i == len(gap) or gap[i] >= 1e-2)
        bound = max(1e-8, 10 * EPS * lam_hat(name) / x) if alone else 10 * ti
        assert abs(x - lr[i]) <= bound * lr[i], (x, lr[i], bound)
    assert V.base is None and V.flags.f_contiguous and V.nbytes == 8 * V.shape[0] * len(lam)
    assert np.abs(V.T @ (M @ V) - np.eye(len(lam))).max() <= 1e-6
    assert np.abs(np.einsum("ij,ij->j", V, K @ V) - lam).max() <= 1e-6 * max(lam.max(initial=1), 1)


class Aggs:  # AMG aggregates stand-in (weakly referenceable)
    pass


@pytest.fixture
def fake_builds(monkeypatch):
    """Every api.amg / api.gmg build is the exact (K + s M)^-1 of the named problem (set
    rec["name"]), honouring the builders' contract: SingularShiftError at s = 0 on a free body,
    return_aggregates / aggregates. rec["builds"]: (s, return_aggregates, aggregates given);
    rec["solves"]: per band-mode api._solve call its V's id, shape, F order, lowest Ritz value and
    stopped (no reference to V)."""
    rec = dict(name="short", builds=[], solves=[])

    def build(op, free, s, *a, return_aggregates=False, aggregates=None, **kw):
        rec["builds"].append((s, return_aggregates, aggregates is not None))
        if s == 0 and not MESHES[rec["name"]][1]:
            raise SingularShiftError("free body at s = 0")
        P = exact(rec["name"], s)
        return (P, Aggs()) if return_aggregates else P

    real = api._solve

    def solve(*a, **kw):
        lam, V, info = real(*a, **kw)
        if kw.get("lam_stop") is not None:
            rec["solves"].append(dict(id=id(V), shape=V.shape, f=V.flags.f_contiguous,
                                      lam0=float(lam.min()), stopped=info.get("stopped"),
                                      locked=len(lam)))  # fmt: skip
        return lam, V, info

    monkeypatch.setattr(api, "amg", build)
    monkeypatch.setattr(api, "gmg", build)
    monkeypatch.setattr(api, "_solve", solve)
    return rec


# ---------------------------------------------------------------- 1, 2: fits, capacity


def test_fits(fake_builds):
    """Test 1: an interior band of about 10 modes inside chunk 0 ("short", n_modes 10): LOBPCG
    reaches the band end ("fits"), chunk 40 (the band count never reaches band_defaults), the
    count passes, V is the solver's array compacted in place (i0 > 0), matches eigsh."""
    f1, f2 = fband("short", 15, 25)
    lam, V, info = run("short", (f1, f2), n_modes=10, verify=True)
    assert info["strategy"] == "lobpcg" and info["reason"] == "fits" and not info["from_zero"]
    assert info["chunk"] == 40 and info["count"]["ran"] and info["count"]["estimate"] < 0.5
    assert info["count"]["degree"] <= 1000
    (s,) = fake_builds["solves"]
    assert id(V) == s["id"] and len(lam) >= 10
    check("short", f1, f2, lam, V, info)


def test_capacity(fake_builds, monkeypatch):
    """Test 2: chunk 5 (k = 10, c_L = 40) at the top of the C_max = 40 window: LOBPCG stops at
    the cap (35 locked: test 12's rule) inside the band (modes 25-45), L = 10; si_band runs on
    the band itself with LOBPCG's state, V and aggregates freed; mu1_ub = min(pre-estimate,
    lowest Ritz value); n_store = n_modes = 22. The same budget under "lobpcg" raises."""
    lr = prob("short")[6]
    budget = need(lob_est("short", 5, 80)) - 1
    assert c_max("short", 5, budget) == 40
    f1, f2 = fband("short", 25, 46)
    lo, hi = (2 * np.pi * f1) ** 2, (2 * np.pi * f2) ** 2
    seen, calls = {}, dict(pre=0, est=0)
    real_band, real_pre, real_est = api.si_band, api._precond_sigma, api.sigma_estimate

    def band(op, free, a, b, make, **kw):
        gc.collect()
        seen.update(interval=(a, b), n_store=kw["n_store"], v_dead=seen["V"]() is None,
                    basis=[x.shape for x in jax.live_arrays() if x.shape == (op.n_dof, 40)])
        return real_band(op, free, a, b, make, **kw)

    real_solve = api._solve

    def solve(*a, **kw):
        out = real_solve(*a, **kw)
        seen["V"] = weakref.ref(out[1])  # the one allowed weakref: nothing resizes this V
        return out

    def pre(*a, **kw):
        calls["pre"] += 1
        return real_pre(*a, **kw)

    monkeypatch.setattr(api, "_solve", solve)
    monkeypatch.setattr(api, "si_band", band)
    monkeypatch.setattr(api, "_precond_sigma", pre)
    monkeypatch.setattr(api, "sigma_estimate", lambda *a, **kw: calls.update(est=1) or real_est(*a, **kw))
    kw = dict(n_modes=22, chunk=5, memory_budget=budget, slice_options={"capacity": 96},
              verify=True)  # fmt: skip
    with pytest.warns(UserWarning, match="35 modes locked"):
        lam, V, info = run("short", (f1, f2), **kw)
    (s,) = fake_builds["solves"]
    assert s["stopped"] == "capacity" and s["locked"] == 35 and lr[34] < hi
    assert info["stopped"] == "capacity" and info["reason"] == "capacity"
    assert info["strategy"] == "shift_invert" and not info["from_zero"]
    assert seen["interval"] == (lo, hi) and seen["v_dead"] and not seen["basis"]
    assert info["mu1_ub"] <= s["lam0"] and info["mu1_ub"] == min(s["lam0"], info["sigma"])
    assert info["n_store"] == seen["n_store"] == 22 and calls == dict(pre=1, est=0)
    assert all(e < 0.5 for e in info["count"]["estimates"]) and info["count"]["passed"]
    assert all(s.get("count", {}).get("degree") is None or s["count"]["degree"] <= 1000
               for s in info["slices"])  # fmt: skip
    check("short", f1, f2, lam, V, info)
    with pytest.raises(ValueError, match="memory_budget"):
        run("short", (f1, f2), **(kw | {"strategy": "lobpcg"}))


def capacity_stop(name, locked):
    """An api._solve stand-in returning a capacity stop with the reference modes locked."""
    lr, Vr = prob(name)[6:]

    def solve(op, free, *a, **kw):
        V = np.zeros((op.n_dof, 64), order="F")
        V[:, : len(locked)] = Vr[:, locked]
        z = np.zeros(len(locked))
        info = dict(stopped="capacity", converged=True, residuals=z, thresholds=z + TOL,
                    time=dict(setup=0.0, iterations=0.0), n_band=len(locked), n_modes=54,
                    block=10, chunk=5, t_iteration=0.0)  # fmt: skip
        return lr[locked], V, info

    return solve


def stub_band(seen):
    """An api.si_band stand-in recording its interval and kwargs, returning no modes."""

    def band(op, free, a, b, make, **kw):
        seen.update(kw, interval=(a, b))
        info = dict(converged=True, reason=None, r_acc=TOL, residuals=np.zeros(0),
                    thresholds=np.zeros(0), count=dict(ran=False, reason=None), time={})
        return np.zeros(0), np.zeros((op.n_dof, 0), order="F"), info

    return band


@pytest.mark.parametrize("case", ["interior", "zero"])
def test_capacity_store(fake_builds, monkeypatch, case):
    """Test 2 (b): 10 locked modes below lam_lo and 20 in band, n_modes 5: n_store = ceil(1.25
    L) = 25 (all locked would give 38), received by the need cap and si_band; (c) f1 = 0 with 30
    locked: si_band on (0, lam_hi), from_zero, n_store 38."""
    seen, caps = {}, []
    real_cap = mem._slice_capacity
    monkeypatch.setattr(api, "_solve", capacity_stop("short", list(range(30))))
    monkeypatch.setattr(api, "si_band", stub_band(seen))
    monkeypatch.setattr(mem, "_slice_capacity",
                        lambda *a, **kw: caps.append(kw["n_store"]) or real_cap(*a, **kw))
    lr = prob("short")[6]
    f1 = 0.0 if case == "zero" else freq((lr[9] + lr[10]) / 2)
    f2 = freq((lr[30] + lr[31]) / 2)
    with pytest.warns(UserWarning, match="30 modes locked"):
        _, _, info = run("short", (f1, f2), n_modes=5)
    want = 38 if case == "zero" else 25
    assert info["n_store"] == seen["n_store"] == caps[0] == want
    assert info["from_zero"] == (case == "zero")
    assert seen["interval"][0] == (0.0 if case == "zero" else (2 * np.pi * f1) ** 2)


# ---------------------------------------------------------------- 3: memory and the ladder


def voxel_budget(s):
    """Test 3's budgets: the chunk-5 fit at C = 40 on voxel 10^3 with factor s on the model."""
    return need(lob_est("voxel", 5, 40), s=s)


def spy_band(monkeypatch, seen):
    """api.si_band wrapped: its interval and kwargs recorded, the real slices run."""
    real = api.si_band

    def band(op, free, a, b, make, **kw):
        seen.update(kw, interval=(a, b))
        return real(op, free, a, b, make, **kw)

    monkeypatch.setattr(api, "si_band", band)


def test_memory_bracket(fake_builds):
    """Test 3: 1.15x and 1.25x the chunk-5 fit at C = 40 bracket the 1.2 model factor: LOBPCG
    chunk 5 (C_max 40, a band in the first 30 modes: "fits") above, "memory" below with the
    slice floor of 72 searched."""
    fake_builds["name"] = "voxel"
    f1, f2 = fband("voxel", 3, 12)
    _, _, info = run("voxel", (f1, f2), n_modes=10, memory_budget=voxel_budget(1.25))
    assert info["reason"] == "fits" and info["chunk"] == 5 and info["c_max"][5] == 40
    with pytest.warns(UserWarning, match=r"no LOBPCG chunk fits.*levers: low_memory=True"):
        _, _, info = run("voxel", (f1, f2), n_modes=10, memory_budget=voxel_budget(1.15))
    assert info["reason"] == "memory" and info["strategy"] == "shift_invert"
    assert info["capacity"] >= 72 and set(info["c_max"].values()) == {None}


@pytest.mark.parametrize("case", ["below", "above", "zero"])
def test_memory(fake_builds, monkeypatch, case):
    """Test 3, "memory" at 1.15x: lam_lo < mu_1 counts from 0 ("below"; band=(0, f) under
    "auto": "zero"), lam_lo well above mu_1 slices (lam_lo, lam_hi) and its counts pass; the
    capacity is the search's, the count block si_band receives the searched one (< n_probes)."""
    fake_builds["name"] = "voxel"
    lr, B = prob("voxel")[6], voxel_budget(1.15)
    f2 = fband("voxel", 0, 10)[1]
    f1 = dict(below=freq(0.5 * lr[0]), above=fband("voxel", 3, 12)[0], zero=0.0)[case]
    if case == "above":
        f2 = fband("voxel", 3, 12)[1]
    seen, blocks, caps = {}, [], []
    spy_band(monkeypatch, seen)
    real_block, real_cap = mem._count_block, mem._slice_capacity
    monkeypatch.setattr(mem, "_count_block", lambda *a, **kw: blocks.append(real_block(*a, **kw))
                        or blocks[-1])  # fmt: skip
    monkeypatch.setattr(mem, "_slice_capacity", lambda *a, **kw: caps.append(real_cap(*a, **kw))
                        or caps[-1])  # fmt: skip
    with pytest.warns(UserWarning, match="no LOBPCG chunk fits"):
        lam, V, info = run("voxel", (f1, f2), n_modes=10, memory_budget=B, verify=case == "above")
    assert info["reason"] == "memory" and info["from_zero"] == (case != "above")
    lo, hi = (2 * np.pi * f1) ** 2, (2 * np.pi * f2) ** 2
    assert seen["interval"] == ((lo, hi) if case == "above" else (0.0, hi))
    assert len(caps) == 1 and caps[0] >= 72 and info["capacity"] == seen["capacity"] == caps[0]
    assert blocks[0] is not None and blocks[0] < 50 and seen["count_options"]["block"] == blocks[0]
    if case == "above":  # from 0 the count's degree is 1912 here (147 s)
        assert info["count"]["passed"] and all(e < 0.5 for e in info["count"]["estimates"])
    check("voxel", f1, f2, lam, V, info)


def test_memory_block_sentinel(fake_builds, monkeypatch):
    """Test 3: a sentinel block from the search reaches si_band's counts."""
    fake_builds["name"] = "voxel"
    seen = {}
    spy_band(monkeypatch, seen)
    monkeypatch.setattr(mem, "_count_block", lambda *a, **kw: 3)
    f1, f2 = fband("voxel", 3, 12)
    with pytest.warns(UserWarning, match="no LOBPCG chunk fits"):
        run("voxel", (f1, f2), n_modes=10, memory_budget=voxel_budget(1.15))
    assert seen["count_options"]["block"] == 3


@pytest.mark.parametrize("f1", [0.0, "interior"])
def test_memory_no_lobpcg_build(fake_builds, monkeypatch, f1):
    """Test 3: on "memory" no LOBPCG preconditioner is built ("short", a budget below chunk 5's
    need, the slice capacity given): f1 = 0 builds nothing before si_band and never asks for
    aggregates; f1 > 0 builds only inside _precond_sigma."""
    seen, inside = {}, []
    real_band, real_pre = api.si_band, api._precond_sigma

    def band(*a, **kw):
        seen["before"] = len(fake_builds["builds"])
        return real_band(*a, **kw)

    def pre(op, free, R, build, **kw):
        n0 = len(fake_builds["builds"])
        out = real_pre(op, free, R, build, **kw)
        inside.append(len(fake_builds["builds"]) - n0)
        return out

    monkeypatch.setattr(api, "si_band", band)
    monkeypatch.setattr(api, "_precond_sigma", pre)
    f1, f2 = (0.0, fband("short", 0, 6)[1]) if f1 == 0 else fband("short", 3, 9)
    B = need(lob_est("short", 5, 40)) - 1e6
    with pytest.warns(UserWarning, match="no LOBPCG chunk fits"):
        run("short", (f1, f2), n_modes=6, memory_budget=B, verify=False,
            slice_options={"capacity": 96})  # fmt: skip
    assert seen["before"] == sum(inside) and (f1 == 0) == (not inside)
    assert not any(ra for _, ra, _ in fake_builds["builds"])


def chunk_need(name, c):
    """The budget at which chunk c's initial basis just fits."""
    k = lob._block_size(c, c, None)[2]
    cl = lob._capacity(0, c, k, False)[0]
    return need(lob_est(name, c, min(mem._up(4 * c + k, cl), mem._up(n_avail(name) + k, cl))))


def test_ladder_real(fake_builds):
    """Test 3: an interior band on "short" at a budget between chunk 20's and chunk 40's needs:
    the chunk is the reach rule's on the searched C_max table, below the top chunk (measured:
    C_max 960 / 1320 / 1360 at chunks 20 / 10 / 5, so chunk 10)."""
    B = (chunk_need("short", 20) + chunk_need("short", 40)) / 2
    reach = {c: c_max("short", c, B) for c in (40, 20, 10, 5)}
    assert reach[40] is None and reach[20] is not None
    f1, f2 = fband("short", 15, 25)
    _, _, info = run("short", (f1, f2), n_modes=10, memory_budget=B, verify=False)
    assert info["c_max"] == reach and info["chunk"] == api._ladder(reach, None) < 20


@pytest.mark.parametrize(
    "reach, need, want",
    [
        ({40: None, 20: 800, 10: 1000, 5: 1100}, None, 10),  # 1000 >= 0.9 x 1100
        ({40: None, 20: 800, 10: 980, 5: 1100}, None, 5),  # 980 < 990
        ({40: 300, 20: 800, 10: 1000, 5: 1100}, None, 10),  # the largest fitting misses 0.9 best
        ({40: 300, 20: 800, 10: 1000}, {40: 400, 20: 600, 10: 500}, 20),  # known depth reached
        ({40: 300, 20: 400}, {40: 500, 20: 450}, 20),  # no chunk reaches it: the 0.9 rule
        ({4: 120}, None, 4),  # a top chunk below 5
        ({40: None, 20: None}, None, None),
    ],
)
def test_ladder(reach, need, want):
    assert api._ladder(reach, need) == want


def test_ladder_tiny():
    """Test 3: a mesh whose free space is below up(4 chunk + k, c_L): C_max is capped at the free
    space (the window starts there), and _halvings keeps a top chunk below 5 alone."""
    op, pc = models("tiny")
    n, na = prob("tiny")[2].n_dof, n_avail("tiny")
    k = lob._block_size(40, 40, None)[2]
    cl = lob._capacity(0, 40, k, False)[0]
    assert mem._up(na + k, cl) < mem._up(4 * 40 + k, cl)
    got = mem._c_max(op, pc, n, chunk=40, guards=None, low=False, basis_dtype="float64",
                     n_rigid=0, n_avail=na, budget=1e12, host_budget=None)  # fmt: skip
    assert got == mem._up(na + k, cl)
    assert api._halvings(4) == [4] and api._halvings(40) == [40, 20, 10, 5]


# ---------------------------------------------------------------- 4-7: shift-invert bands


def slice_budget(name, C, n_store, block=6):
    """The smallest budget (to 1e-9 relative) whose searched slice capacity is >= C."""
    op, pc = models(name)
    n = prob(name)[2].n_dof

    def cap(B):
        return mem._slice_capacity(op, pc, n, n_store=n_store, block=block, n_rigid=0, budget=B,
                                   host_budget=None) or 0  # fmt: skip

    lo, hi = 1e6, 1e11
    while hi - lo > 1e-9 * hi:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cap(mid) < C else (lo, mid)
    assert cap(hi) == C
    return hi


def spy_ids(monkeypatch, ids):
    """api.si_band wrapped: the id of every V it returns."""
    real = api.si_band

    def band(*a, **kw):
        lam, V, info = real(*a, **kw)
        ids.append(id(V))
        return lam, V, info

    monkeypatch.setattr(api, "si_band", band)


def shift_invert_run(monkeypatch):
    """Test 4's run: 15c test 2(a)'s band (modes 5-28, block 6, seed 0) forced to slices at the
    budget whose searched capacity is 96 (>= 2 runs), n_modes 26 (the store; edges widens the
    band to 25 modes)."""
    f1, f2 = fband("short", 5, 29)
    blocks = []
    real = mem._count_block
    monkeypatch.setattr(mem, "_count_block", lambda *a, **kw: blocks.append(real(*a, **kw))
                        or blocks[-1])  # fmt: skip
    lam, V, info = run("short", (f1, f2), n_modes=26, strategy="shift_invert", verify=True,
                       memory_budget=slice_budget("short", 96, 26))  # fmt: skip
    assert blocks[0] is not None and info["capacity"] == 96 and info["reason"] == "forced"
    assert sum(s["kind"] == "run" for s in info["slices"]) >= 2
    c = info["count"]
    assert c["passed"] and len(c["estimates"]) == sum(s["leaf"] for s in info["slices"])
    assert all(e < 0.5 for e in c["estimates"])
    check("short", f1, f2, lam, V, info)
    ref, _, _ = run("short", (f1, f2), n_modes=26, strategy="lobpcg", verify=False)
    assert len(ref) == len(lam) and np.allclose(ref, lam, rtol=1e-6)


def test_shift_invert(fake_builds, monkeypatch):
    """Test 4 with the exact preconditioner."""
    shift_invert_run(monkeypatch)


@pytest.mark.slow
def test_shift_invert_amg(monkeypatch):
    """Test 4 with real AMG."""
    shift_invert_run(monkeypatch)


@pytest.mark.parametrize("strategy", ["lobpcg", "shift_invert"])
def test_empty_band(fake_builds, strategy):
    """Test 5: a band strictly inside the cube's 4.85 -> 8.86 gap (relative 0.45): no modes, the
    count < 0.5, not from 0."""
    fake_builds["name"] = "cube"
    lr = prob("cube")[6]
    assert lr[1] < 5.5 and lr[2] > 8.0 and lr[2] - lr[1] > 0.4 * lr[2]
    lam, V, info = run("cube", (freq(5.5), freq(8.0)), strategy=strategy, verify=True)
    assert len(lam) == 0 and V.shape == (prob("cube")[2].n_dof, 0) and not info["from_zero"]
    assert info["count"]["ran"] and info["count"]["estimate"] < 0.5


def test_from_zero_band(fake_builds, monkeypatch):
    """Test 6: band=(0, f) on the clamped box under "shift_invert": from 0, eigsh's modes, V is
    si_band's V (i0 = 0, all its columns kept)."""
    ids = []
    spy_ids(monkeypatch, ids)
    f2 = fband("short", 0, 10)[1]
    lam, V, info = run("short", (0.0, f2), strategy="shift_invert")
    assert info["from_zero"] and ids == [id(V)] and info["sigma"] is None
    check("short", 0.0, f2, lam, V, info)


@pytest.mark.parametrize("name", ["short", "free"])
def test_below_mu1(fake_builds, monkeypatch, name):
    """Test 7: lam_lo = lam_1 (1 + 0.5e-2) < mu1_ub (1 + 1e-2) under "shift_invert": from 0,
    the modes below lam_lo filtered (i0 > 0: the resize shrinks si_band's own V in place)."""
    fake_builds["name"] = name
    lr, ids = prob(name)[6], []
    spy_ids(monkeypatch, ids)
    f1, f2 = freq(lr[0] * (1 + 0.5e-2)), fband(name, 2, 10)[1]
    lam, V, info = run(name, (f1, f2), strategy="shift_invert")
    assert info["from_zero"] and ids == [id(V)] and info["mu1_ub"] >= lr[0]
    assert lam[0] > lr[0] * (1 + 0.5e-2)  # the filter ran (i0 > 0)
    check(name, f1, f2, lam, V, info)


@pytest.mark.parametrize(
    "rel, tol, want", [(0.9e-2, TOL, True), (1.1e-2, TOL, False), (2.9e-2, 1e-4, True),
                       (3.1e-2, 1e-4, False)]  # fmt: skip
)
def test_mu1_margin(monkeypatch, rel, tol, want):
    """Test 7: from 0 when lam_lo < mu1_ub (1 + max(1e-2, 300 tol)), mu1_ub >= mu_1 patched."""
    mu, seen = prob("short")[6][0], {}
    monkeypatch.setattr(api, "_precond_sigma", lambda *a, **kw: (mu, 1))
    monkeypatch.setattr(api, "si_band", stub_band(seen))
    _, _, info = run("short", (freq(mu * (1 + rel)), fband("short", 5, 10)[1]), tol=tol,
                     strategy="shift_invert")  # fmt: skip
    assert info["from_zero"] == want and info["mu1_ub"] == mu
    assert (seen["interval"][0] == 0) == want


def test_mu1_inflated(fake_builds, monkeypatch):
    """Test 7: an inflated mu1_ub (1.5 lam_1) above lam_lo = 1.2 lam_1: from 0, still eigsh's
    modes (the filter drops the two below lam_lo)."""
    lr = prob("short")[6]
    monkeypatch.setattr(api, "_precond_sigma", lambda *a, **kw: (1.5 * lr[0], 1))
    f1, f2 = freq(1.2 * lr[0]), fband("short", 2, 10)[1]
    lam, V, info = run("short", (f1, f2), strategy="shift_invert")
    assert info["from_zero"]
    check("short", f1, f2, lam, V, info)


def spy_pre(monkeypatch, calls):
    """api._precond_sigma and api.sigma_estimate wrapped: their kwargs per call."""
    real_pre, real_est = api._precond_sigma, api.sigma_estimate

    def pre(*a, **kw):
        calls.append(("pre", kw))
        return real_pre(*a, **kw)

    def est(*a, **kw):
        calls.append(("est", kw))
        return real_est(*a, **kw)

    monkeypatch.setattr(api, "_precond_sigma", pre)
    monkeypatch.setattr(api, "sigma_estimate", est)


def test_pre_estimate_hint(monkeypatch):
    """Test 7: a slice path with f1 > 0 and a sigma hint of 0.5 mu_1 on voxel 16^3 (real GMG),
    lam_lo ~ 3 mu_1: the pre-estimate runs once from 0 (the hint never enters it), mu_1 <=
    mu1_ub <= mu_1 (1 + 1e-3), sigma = mu1_ub, not from 0."""
    from scipy.sparse.linalg import eigsh

    from jaxmodal.mesh import voxel_mesh
    from jaxmodal.operators import assemble, dof_mask

    mesh = voxel_mesh((16,) * 3, (1 / 16,) * 3)
    K, M = assemble(mesh, MAT)
    f = np.flatnonzero(dof_mask(mesh, {"zmin": "xyz"}))
    mu = eigsh(K[f][:, f].tocsc(), 1, M[f][:, f].tocsc(), sigma=0, which="LM")[0][0]
    calls, seen = [], {}
    spy_pre(monkeypatch, calls)
    monkeypatch.setattr(api, "si_band", stub_band(seen))
    _, _, info = solve_modes(mesh, MAT, {"zmin": "xyz"}, band=(freq(3 * mu), freq(4 * mu)),
                             strategy="shift_invert", sigma=0.5 * mu)  # fmt: skip
    assert [c for c, _ in calls] == ["pre"]
    assert mu * (1 - 1e-12) <= info["mu1_ub"] <= mu * (1 + 1e-3)
    assert info["sigma"] == info["mu1_ub"] and info["sigma_iterations"] > 0
    assert not info["from_zero"] and seen["interval"][0] == (2 * np.pi * freq(3 * mu)) ** 2


def test_pre_estimate_zero(monkeypatch):
    """Test 7: a slice path with f1 = 0 runs no pre-estimate."""
    calls, seen = [], {}
    spy_pre(monkeypatch, calls)
    monkeypatch.setattr(api, "si_band", stub_band(seen))
    _, _, info = run("short", (0.0, fband("short", 0, 5)[1]), strategy="shift_invert")
    assert not calls and info["sigma"] is None and info["sigma_iterations"] == 0


@pytest.mark.parametrize("precond", ["auto", "jacobi"])
def test_pre_estimate_free(fake_builds, monkeypatch, precond):
    """Test 7 on the free hex8 at lam_lo ~ 3 mu_1: "auto" (sigma=0 given) falls back from the
    singular s = 0 build to sigma_estimate inside _precond_sigma; Jacobi calls sigma_estimate at
    its default maxit (300); mu_1 <= mu1_ub <= 1.1 mu_1, not from 0."""
    fake_builds["name"] = "free"
    mu, calls, seen = prob("free")[6][0], [], {}
    spy_pre(monkeypatch, calls)
    monkeypatch.setattr(api, "si_band", stub_band(seen))
    kw = dict(sigma=0.0) if precond == "auto" else {}
    _, _, info = run("free", (freq(3 * mu), freq(4 * mu)), strategy="shift_invert",
                     precond=precond, **kw)  # fmt: skip
    if precond == "auto":
        assert [c for c, _ in calls] == ["pre", "est"] and calls[1][1]["maxit"] == 10
    else:
        assert [c for c, _ in calls] == ["est"] and "maxit" not in calls[0][1]
    assert mu <= info["mu1_ub"] <= 1.1 * mu and not info["from_zero"]


# ---------------------------------------------------------------- 8: the band filter

LO = 3.0


@pytest.mark.parametrize(
    "lam, t, lam_lo, n_band, want",
    [
        ([1, 2, 3, 4, 5], 1e-8, LO, 5, (2, 5)),  # the exact edge
        ([1, 3 - 1e-7, 3.5, 4], 1e-8, LO, 4, (1, 4)),  # inside 100 t
        ([1, 2.99, 3.5, 4], 1e-8, LO, 4, (2, 4)),  # outside 100 t
        ([1, 3 - 4e-6, 3 - 2e-6, 4], 1e-8, LO, 4, (1, 4)),  # a chain crossing lam_lo
        ([1, 2.99, 2.9999, 3.1], [1e-8, 1e-2, 1e-9, 1e-8], LO, 4, (1, 4)),  # non-monotone t
        ([1, 3.5, 4, 5, 6], [1e-8, 1e-8, 1e-8, 1e-2, 1e-8], LO, 3, (1, 3)),  # the upper end
        ([1, 2], 1e-8, LO, 2, (2, 2)),  # all below lam_lo
        ([], 1e-8, LO, 0, (0, 0)),  # nothing
        ([1, 2, 3], 1e-8, 0.0, 2, (0, 2)),  # from 0
    ],
)
def test_keep(lam, t, lam_lo, n_band, want):
    assert api._keep(np.array(lam, float), t, lam_lo, n_band) == want


@pytest.mark.parametrize("seed", range(5))
def test_permute(seed):
    """The in-place cycle permutation equals V[:, order] and keeps V's F order and buffer."""
    rng = np.random.default_rng(seed)
    V = np.asfortranarray(rng.standard_normal((7, 12)))
    order = rng.permutation(9)
    want, ptr = V[:, order], V.ctypes.data
    rest = V[:, 9:].copy()
    api._permute(V, order)
    assert np.array_equal(V[:, :9], want) and np.array_equal(V[:, 9:], rest)
    assert V.flags.f_contiguous and V.ctypes.data == ptr


def harness(monkeypatch, lam, n_band, *, t=TOL, stopped=None, converged=True, hold=None):
    """solve_modes' band path on "short" with api._solve returning the crafted Ritz values lam
    (residuals 1e-12 lam, thresholds t: per mode or one value), F-order V of 3 spare zero columns
    with column j filled with j + 1, built inside (only its id kept, or the array appended to
    hold). rec: the id, the lam api._verify received (its stand-in returns a no-count record),
    the si_band calls."""
    rec = dict(verify=[], band=[])
    lam = np.array(lam, float)
    n = prob("short")[2].n_dof

    def solve(*a, **kw):
        V = np.zeros((n, len(lam) + 3), order="F")
        V[:, : len(lam)] = np.arange(1, len(lam) + 1)
        rec["id"] = id(V)
        if hold is not None:
            hold.append(V)
        info = dict(converged=converged, residuals=1e-12 * lam,
                    thresholds=np.broadcast_to(np.asarray(t, float), lam.shape).copy(),
                    n_band=n_band, n_modes=len(lam), block=10, chunk=5, t_iteration=0.0,
                    time=dict(setup=0.0, iterations=0.0))  # fmt: skip
        if stopped:
            info["stopped"] = stopped
        return lam.copy(), V, info

    def verify(verify, opts, op, free, R, lam, *a):
        rec["verify"].append(lam.copy())
        return dict(ran=False, reason="faked")

    seen = {}

    def band(*a, **kw):
        rec["band"].append(a[2:4])
        return stub_band(seen)(*a, **kw)

    monkeypatch.setattr(api, "_solve", solve)
    monkeypatch.setattr(api, "_verify", verify)
    monkeypatch.setattr(api, "si_band", band)
    return rec


def fake_run(lo, hi, **kw):
    """solve_modes on "short" for the band [lo, hi] (eigenvalues) through the harness."""
    return run("short", (freq(lo), freq(hi)), sigma=0.5, **kw)


def assert_compacted(rec, lam, V, cols):
    assert id(V) == rec["id"] and V.base is None and V.flags.f_contiguous
    assert V.nbytes == 8 * V.shape[0] * len(lam) and V.shape[1] == len(cols)
    assert np.array_equal(V[0], np.array(cols, float))


@pytest.mark.parametrize("lo, cols", [(0.0, [1, 2, 3, 4, 5]), (1.5, [2, 3, 4, 5])])
def test_filter_compacted(fake_builds, monkeypatch, lo, cols):
    """Test 8: the kept columns compacted in place in the solver's own array, then shrunk (i0 =
    0, and i0 = 1 with the source overlapping the destination)."""
    rec = harness(monkeypatch, [1, 2, 3, 4, 5, 7], 5)
    lam, V, info = fake_run(lo, 6.0)
    assert_compacted(rec, lam, V, cols)
    assert np.array_equal(lam, np.array(cols, float)) and len(info["residuals"]) == len(cols)
    assert np.array_equal(rec["verify"][0], [1, 2, 3, 4, 5, 7])


def test_filter_held(fake_builds, monkeypatch):
    """Test 8: another reference to V: the warning, and a copy with the kept columns."""
    hold = []
    harness(monkeypatch, [1, 2, 3, 4, 5, 7], 5, hold=hold)
    with pytest.warns(RuntimeWarning, match="returned modes copied"):
        lam, V, _ = fake_run(1.5, 6.0)
    assert V.flags.f_contiguous and V.base is None and np.array_equal(V[0], [2, 3, 4, 5])
    assert id(V) != id(hold[0])


@pytest.mark.parametrize("collect", [True, False])
def test_filter_jax_view(fake_builds, monkeypatch, collect):
    """Test 8: a _verify whose last jax call transfers a view of V: jaxlib holds that view until
    a collection, which the resize's gc.collect runs (no warning, the same id); with the
    collect a no-op and automatic collection off, the copy warning (the scenario needs it)."""
    rec = harness(monkeypatch, list(range(1, 11)), 9)

    def verify(verify, opts, op, free, R, lam, V, *a):
        rec["verify"].append(lam.copy())
        jnp.asarray(V[:, :8]).block_until_ready()
        return dict(ran=False, reason="faked")

    monkeypatch.setattr(api, "_verify", verify)
    if not collect:
        monkeypatch.setattr(api, "gc", types.SimpleNamespace(collect=lambda: 0))
    gc.disable()
    try:
        if collect:
            lam, V, _ = fake_run(1.5, 9.5)
            assert_compacted(rec, lam, V, list(range(2, 10)))
        else:
            with pytest.warns(RuntimeWarning, match="returned modes copied"):
                fake_run(1.5, 9.5)
    finally:
        gc.enable()


def test_filter_sorted(fake_builds, monkeypatch):
    """Test 8: out-of-order LOBPCG values sorted with their columns, residuals and thresholds
    before the cut; _verify sees the sorted values; the result is the solver's array."""
    rec = harness(monkeypatch, [1, 2, 5, 6, 4.8], 3, t=[1e-8, 2e-8, 3e-8, 4e-8, 5e-8])
    lam, V, info = fake_run(4.9, 5.5)
    assert np.array_equal(rec["verify"][0], [1, 2, 4.8, 5, 6])
    assert_compacted(rec, lam, V, [3])  # 5.0 was column 3
    assert np.array_equal(lam, [5.0]) and np.array_equal(info["residuals"], [5e-12])
    assert np.array_equal(info["thresholds"], [3e-8])


def test_filter_recut(fake_builds, monkeypatch):
    """Test 8: the cut recomputed after the sort: 5 moved below lam_hi = 5.5 is kept."""
    rec = harness(monkeypatch, [1, 5, 2, 3, 6], 1)
    lam, V, _ = fake_run(1.5, 5.5)
    assert np.array_equal(lam, [2, 3, 5]) and np.array_equal(rec["verify"][0], [1, 2, 3, 5, 6])
    assert_compacted(rec, lam, V, [3, 4, 2])


def test_filter_cut_at_end(fake_builds, monkeypatch):
    """Test 8: a sort that moves the cut to the end: not converged, "sorted cut at the end", no
    _verify call."""
    rec = harness(monkeypatch, [1, 5, 2], 1)
    lam, _, info = fake_run(1.5, 5.5)
    assert not info["converged"] and info["stopped"] == "sorted cut at the end"
    assert info["count"]["reason"] == "sorted cut at the end" and not rec["verify"]
    assert np.array_equal(lam, [2, 5])


def test_filter_capacity_sorted(fake_builds, monkeypatch):
    """Test 8: a capacity stop with values out of order still falls back to si_band."""
    rec = harness(monkeypatch, [1, 5, 2, 3], 4, stopped="capacity")
    with pytest.warns(UserWarning, match="4 modes locked"):
        _, _, info = fake_run(1.5, 9.0)
    assert len(rec["band"]) == 1 and not rec["verify"] and info["reason"] == "capacity"
    assert info["mu1_ub"] == 1.0


# ---------------------------------------------------------------- 9, 10: clusters, the top


@pytest.mark.parametrize("strategy", ["lobpcg", "shift_invert"])
@pytest.mark.parametrize("i_lo, i0", [(8, 8), (3, 2)])  # on a double; mid-triple (2-4)
def test_lower_edge_clusters(fake_builds, strategy, i_lo, i0):
    """Test 9: lam_lo exactly on a cube eigenvalue of a double (11.68) and in the middle of a
    triple (8.86): every copy returned, not from 0."""
    fake_builds["name"] = "cube"
    _, _, _, K, M, _, lr, _ = prob("cube")
    f1, f2 = freq(lr[i_lo]), fband("cube", i0, 12)[1]
    i1 = int((lr <= (2 * np.pi * f2) ** 2).sum())  # edges widens over the 12.74 triple
    lam, V, info = run("cube", (f1, f2), strategy=strategy)
    assert not info["from_zero"] and len(lam) == i1 - i0
    assert np.allclose(lam, lr[i0:i1], rtol=1e-8)
    assert np.abs(V.T @ (M @ V) - np.eye(len(lam))).max() <= 1e-6


def test_edge_above_spectrum(fake_builds):
    """Test 10: f2 above the largest eigenvalue of "tiny": LOBPCG raises 14b's free-space error,
    the slices their "no progress" error naming the upper side."""
    fake_builds["name"] = "tiny"
    lr = prob("tiny")[6]
    band = (fband("tiny", 3, 6)[0], freq(2 * lr[-1]))
    with pytest.raises(ValueError, match="more modes than the mesh provides"):
        run("tiny", band, strategy="lobpcg")
    with pytest.raises(ValueError, match="no progress.*hi above the spectrum"):
        run("tiny", band, strategy="shift_invert")


# ---------------------------------------------------------------- 12: _solve(max_capacity)


def band_solve(cap, lam_stop, n_modes=30, chunk=5, dtype="float64", maxit=200):
    """lobpcg._solve in band mode on "short" with the exact preconditioner at 0."""
    _, free, op, *_ = prob("short")
    return lob._solve(op, free, n_modes, TOL, None, exact("short", 0.0), None, maxit, 0,
                      chunk=chunk, basis_dtype=dtype, lam_stop=lam_stop,
                      max_capacity=cap)  # fmt: skip


def test_solve_capacity_sequence(monkeypatch):
    """Test 12: every basis width a multiple of c_L = 40 (chunk 5, k = 10), growing straight to
    the cap 120 (40 -> 80 -> 120), then the capacity stop below lam_stop: 110 + k = 120 locks,
    115 stops."""
    news, real = [], lob._grow
    monkeypatch.setattr(lob, "_grow", lambda V, count, new, c, dt: news.append(new)
                        or real(V, count, new, c, dt))  # fmt: skip
    lam, V, info = band_solve(120, 10 * prob("short")[6][-1])
    assert news == [80, 120] and info["capacity"] == 120 and info["stopped"] == "capacity"
    assert len(lam) == 115 and V.shape == (V.shape[0], 130)


def test_solve_boundary(monkeypatch):
    """Test 12: C_max = 40 at chunk 5: count 30 (count + k = C_max) locks to 35 (a >= mutant
    stops at 30), the next lock stops and is dropped; V is the preallocated F-order zeros
    array of C_max + k columns, zero from the locked count on; the initial basis is capped
    (n_modes 60 -> C_max - k = 30)."""
    n = prob("short")[2].n_dof
    made, real = [], np.zeros

    def zeros(shape, *a, **kw):
        out = real(shape, *a, **kw)
        if shape == (n, 50) and kw.get("order") == "F":
            made.append(id(out))
        return out

    monkeypatch.setattr(np, "zeros", zeros)
    lam, V, info = band_solve(40, prob("short")[6][-1], n_modes=60)
    monkeypatch.undo()
    assert info["stopped"] == "capacity" and len(lam) == 35 and info["capacity"] == 40
    assert len(info["residuals"]) == len(info["thresholds"]) == 35
    assert info["chunks"][-1]["locked"] == 0 and made == [id(V)]
    assert V.shape == (n, 50) and V.flags.f_contiguous and not V[:, 35:].any()
    assert info["n_modes"] == 30


def test_solve_boundary_one(monkeypatch):
    """Test 12: chunks locking one mode each reach count + k = C_max + 1 at count 31: stop."""
    real = lob._iterate

    def one(*a, **kw):
        state, r, t, j, rec = real(*a, **kw)
        return state, r, t, min(j, 1), rec

    monkeypatch.setattr(lob, "_iterate", one)
    lam, _, info = band_solve(40, prob("short")[6][-1])
    assert info["stopped"] == "capacity" and len(lam) == 31


def test_solve_capped_start():
    """Test 12: a band ending in the first chunk: no growth, the capped basis and n_modes."""
    lam, V, info = band_solve(40, prob("short")[6][3], n_modes=60)
    assert info["capacity"] == 40 and info["n_modes"] == 30 and "stopped" not in info
    assert V.shape[1] == 50 and len(lam) >= 4 and not V[:, len(lam):].any()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_solve_refill(monkeypatch, dtype):
    """Test 12: after a growth and its lock, B's first count + j columns are V's (cast to the
    basis dtype) and the rest 0; at _grow's entry no live array has the old basis's shape (the
    state and B freed first), while one had it at the lock before."""
    n = prob("short")[2].n_dof
    rec = dict(old=None, checks=0)
    real_grow, real_lock = lob._grow, lob._lock

    def grow(V, count, new, c, dt):
        shapes = [x.shape for x in jax.live_arrays()]
        assert rec["old"] > 0 and (n, rec["old"]) not in shapes
        rec.update(V=V, grown=True)
        return real_grow(V, count, new, c, dt)

    def lock(B, count, X, j):
        rec["old"] = B.shape[1]
        assert sum(x.shape == B.shape for x in jax.live_arrays()) == 1  # B alone has the width
        out = real_lock(B, count, X, j)
        if rec.pop("grown", False):
            Bh, Vh = np.asarray(out), rec.pop("V")
            assert np.array_equal(Bh[:, : count + j], Vh[:, : count + j].astype(dtype))
            assert not Bh[:, count + j :].any()
            rec["checks"] += 1
        return out

    monkeypatch.setattr(lob, "_grow", grow)
    monkeypatch.setattr(lob, "_lock", lock)
    _, _, info = band_solve(80, prob("short")[6][-1], dtype=dtype)
    assert rec["checks"] == 1 and info["capacity"] == 80


def test_solve_needs_cap():
    with pytest.raises(ValueError, match="needs max_capacity"):
        band_solve(None, 1.0)


# ---------------------------------------------------------------- 11: capacity and C_max


def test_lobpcg_terms():
    """Test 11: lobpcg_memory(capacity=C) on "short" (AMG): the basis exactly C columns, V the
    expected modes plus a block, the exact parts, and host_total the max of the operator build,
    V + AMG's host build and V + the refill's two chunks."""
    op, pc = models("short")
    n, C = prob("short")[2].n_dof, 120
    e = mem.lobpcg_memory(op, n, 30, 5, None, False, "float64", 2, pc, True, capacity=C)
    k, c = 10, 40
    V = 8 * n * 30 + 8 * n * k
    assert e["basis"] == 8 * n * C and e["host_exact"] == V and e["rigid"] == 2 * 8 * n * 2
    assert e["device_exact"] == op(k)["storage"] + e["basis"] + e["rigid"]
    assert pc(k)["host_build"] == pc(k)["build"]
    assert e["host_total"] == max(op(k)["build"], V + pc(k)["host_build"], V + 16 * n * c)
    assert e["host_total"] == V + pc(k)["host_build"]  # the AMG build binds here
    assert e["device_total"] == (op(k)["overhead"] + op(k)["storage"] + e["precond"] + e["basis"]
                                 + e["rigid"] + e["host"] - V + max(e["step"], 0))  # fmt: skip
    op, pc = models("voxel")  # GMG builds on the device: the refill binds
    n = prob("voxel")[2].n_dof
    g = mem.lobpcg_memory(op, n, 30, 5, None, False, "float64", 0, pc, True, capacity=C)
    assert g["host_total"] == 8 * n * (30 + k) + 16 * n * c > op(k)["build"]


def test_count_si_terms():
    """Test 11: count_memory all on the device; si_memory's device basis up(C, c) columns, the
    window and the store host-only, the Gram matrix 8 capacity^2, the per-space exact parts."""
    op, pc = models("voxel")
    n = prob("voxel")[2].n_dof
    v = 8 * n
    e = mem.count_memory(op, n, 4, voxel=True, precond_memory=pc, n_rigid=3)
    assert e["device_total"] == e["total"] and e["host_total"] == e["host_exact"] == 0
    assert e["device_exact"] == op(4)["storage"] + 2 * v * 3

    def si(C, **kw):
        return mem.si_memory(op, n, C, kw.pop("n_modes", 20), 6, pc, 0, n_deflate=48, **kw)

    a, b = si(48), si(47)
    assert a == b and si(72)["storage"] - a["storage"] == v * 24
    w = si(48, host_window=100)
    assert w["storage"] == a["storage"] and w["host"] - a["host"] == v * 52
    s = si(48, n_modes=30)
    assert s["storage"] == a["storage"] and s["host"] - a["host"] == v * 10
    assert a["host"] == v * 48 + v * 20 + 8 * 48**2 and a["end"] == v * (2 + 24)
    assert a["device_exact"] == a["storage"] - pc(6)["storage"]
    assert a["host_exact"] == a["host"] - 8 * 48**2
    assert a["host_total"] == a["host"] + max(op(6)["build"], 0)  # GMG builds on the device


def test_amg_scan_cached(monkeypatch):
    """Test 11: amg_memory scans the mesh once per mesh."""
    mem._amg_scan.cache_clear()
    mesh = prob("short")[0]
    for m in (6, 10, 15):
        mem.amg_memory(mesh, m)
    info = mem._amg_scan.cache_info()
    mem._amg_scan.cache_clear()
    assert info.misses == 1 and info.hits == 2


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="Mac peaks")
@pytest.mark.parametrize("t, n", [("hex20", 16), ("hex20", 32), ("hex8", 32)])
def test_amg_reuse_peak(t, n):
    """Test 11: amg(..., aggregates=aggs) with aggs held (a fresh subprocess, the peak above
    the built operator and aggregates) plus the aggregates' bytes <= amg_memory's build (ratios
    0.63-0.83 measured in plan round 13)."""
    from benchmarks import amg_study, harness

    from jaxmodal.mesh import box_mesh

    s = amg_study.SETUP.format(root=str(harness.RESULTS.parents[1]), t=t, n=n) + (
        "P, agg = amg(op, free, 1e6, mesh.points, return_aggregates=True)\nP = None\n")
    code = ("from jaxmodal.eigen.api import _nbytes\njax.block_until_ready(amg(op, free, 1e6,"
            " mesh.points, aggregates=agg))\njson.dump(dict(agg=_nbytes(agg)), open('{out}', 'w'))")
    r = harness.measure_json(code, s)
    est = mem.amg_memory(box_mesh((n,) * 3, type=t), 15)["build"]
    assert r["peak_bytes"] + r["agg"] <= est, (r, est)


def test_models_cached(fake_builds, monkeypatch):
    """Test 11: within one band call no estimator sees the same arguments twice (a "fits" run
    with the count block given, and a slice path); a second call computes again."""
    seen, real = [], mem._models

    def models_(*a, **kw):
        out = real(*a, **kw)

        def wrap(f, tag):
            def g(*x, **y):
                seen.append((tag, x, tuple(sorted(y.items()))))
                return f(*x, **y)

            return g

        return [f and wrap(f, (len(seen), i)) for i, f in enumerate(out)]

    monkeypatch.setattr(mem, "_models", models_)
    f1, f2 = fband("short", 15, 25)
    for kw in (dict(count_options={"block": 4}), dict(strategy="shift_invert", verify=False)):
        for _ in range(2):
            seen.clear()
            run("short", (f1, f2), n_modes=12, **kw)  # the band holds 12
            calls = [(i, x, y) for (_, i), x, y in seen]
            assert seen and len(calls) == len(set(calls)), kw


def test_levers(fake_builds):
    """Test 11: the "memory" warning names each lever's C_max at chunk 5 and the otf lever."""
    fake_builds["name"] = "voxel"
    B = voxel_budget(1.15)
    op, pc = models("voxel")
    kw = dict(chunk=5, guards=None, basis_dtype="float64", n_rigid=0, n_avail=n_avail("voxel"),
              budget=B, host_budget=None)  # fmt: skip
    n = prob("voxel")[2].n_dof
    low = mem._c_max(op, pc, n, low=True, **kw)
    f32 = mem._c_max(op, pc, n, low=False, **(kw | {"basis_dtype": "float32"}))
    with pytest.warns(UserWarning) as w:
        run("voxel", fband("voxel", 3, 12), n_modes=10, memory_budget=B, verify=False)
    (msg,) = [str(x.message) for x in w if "levers" in str(x.message)]
    assert f"low_memory=True (C_max {low})" in msg and f"'float32' (C_max {f32})" in msg
    assert "operator_options={'mode': 'otf'}" in msg and (low or f32)


def est(e, m, de=0, dm=0, he=0, hm=0):
    """A crafted estimate: E = e exact, M = m modelled (CPU); per space de/dm, he/hm."""
    return dict(device_exact=de or e, host_exact=he, total=e + m + he,
                device_total=(de or e) + dm, host_total=he + hm)  # fmt: skip


def test_fits():
    """Test 11: 1.05 E + 1.2 M against the budget: one check on total (CPU), per space with a
    host budget (GPU)."""
    x = est(100, 50)
    assert mem._fits(x, 165) and not mem._fits(x, 164.99)
    assert mem._fits(x, 165 + 1.05 * 10, host_extra=10)
    assert not mem._fits(x, 165 + 1.05 * 10 - 0.01, host_extra=10)
    g = est(0, 0, de=100, dm=50, he=20, hm=10)
    assert mem._fits(g, 165, 1.05 * 20 + 1.2 * 10)
    assert not mem._fits(g, 164.99, 1e9) and not mem._fits(g, 1e9, 1.05 * 20 + 11.99)
    assert mem._fits(g, 165, 1.05 * 25 + 12, host_extra=5)


@pytest.mark.parametrize("seed", range(4))
def test_largest(seed):
    """Test 11: the bisection equals brute force over the multiples of step (a monotone fit with
    a jump), None when nothing fits."""
    rng = np.random.default_rng(seed)
    step, lo, hi = int(rng.integers(1, 30)), int(rng.integers(0, 100)), int(rng.integers(100, 900))
    top = int(rng.integers(0, 1000))
    want = [x for x in range(0, hi + 1, step) if lo <= x <= top]
    assert mem._largest(lambda x: x <= top, step, lo, hi) == (max(want) if want else None)
    assert mem._largest(lambda x: False, step, lo, hi) is None


def slice_brute(name, B, n_store, block=6):
    op, pc = models(name)
    n, c = prob(name)[2].n_dof, si._chunk(block)
    ok = [C for C in range(c, 2000, c) if C >= mem._up(2 * c + 2 * block, c)
          and C <= mem._up(math.ceil(1.25 * (1.8 * n_store + 84)) + 2 * c, c)
          and mem._fits(mem.si_memory(op, n, C - 2 * c, n_store, block, pc, 0, n_deflate=2 * c,
                                      host_window=2 * (C - 2 * block)), B)]  # fmt: skip
    return max(ok) if ok else None


@pytest.mark.parametrize("name", ["tiny", "voxel"])
def test_slice_capacity(name):
    """Test 11: _slice_capacity against brute force (on "tiny" the Gram term and the end phase
    matter), the floor 72 at c = 24 when only it fits, the need cap binding, None below."""
    op, pc = models(name)
    n = prob(name)[2].n_dof

    def cap(B, n_store=20):
        return mem._slice_capacity(op, pc, n, n_store=n_store, block=6, n_rigid=0, budget=B,
                                   host_budget=None)  # fmt: skip

    floor = slice_budget(name, 72, 20)
    for B in np.linspace(floor, 3 * floor, 7):
        assert cap(B) == slice_brute(name, B, 20)
    assert cap(floor) == 72 and cap(floor * (1 - 1e-6)) is None
    need_cap = mem._up(math.ceil(1.25 * (1.8 * 20 + 84)) + 48, 24)
    assert cap(1e12) == need_cap and cap(1e12, n_store=200) > need_cap
    with pytest.raises(ValueError, match="below one probe"):
        mem.probe_block(1.0, 50, op, n, precond_memory=pc)


def test_c_max_caps():
    """Test 11: C_max is the free-space cap (a multiple of c_L) at a large budget, the budget's
    largest fit below it, None below the initial basis."""
    na = n_avail("short")
    cap = mem._up(na + 10, 40)
    assert c_max("short", 5, 1e12) == cap and cap % 40 == 0
    B = need(lob_est("short", 5, 400))
    got = c_max("short", 5, B)
    assert got == mem._largest(lambda C: mem._fits(lob_est("short", 5, C), B), 40, 40, cap)
    assert 40 <= got < cap and c_max("short", 5, need(lob_est("short", 5, 40)) - 1) is None


def test_c_max_host():
    """Test 11 (GPU accounting): a tight host budget (V binds) lowers C_max at the same device
    budget; AMG's host build counts in the host max."""
    op, pc = models("short")
    n = prob("short")[2].n_dof
    kw = dict(chunk=5, guards=None, low=False, basis_dtype="float64", n_rigid=0,
              n_avail=n_avail("short"), budget=1e12)  # fmt: skip
    free_ = mem._c_max(op, pc, n, host_budget=1e12, **kw)
    host = mem.lobpcg_memory(op, n, 400, 5, None, False, "float64", 0, pc, True, capacity=400)
    tight = 1.05 * host["host_exact"] + 1.2 * (host["host_total"] - host["host_exact"])
    got = mem._c_max(op, pc, n, host_budget=tight, **kw)
    assert got == 400 < free_
    small = 1.05 * 8 * n * 40 + 1.2 * (pc(10)["host_build"]) - 1
    assert mem._c_max(op, pc, n, host_budget=small, **kw) is None  # V + the AMG build binds


def test_slice_limits(fake_builds, monkeypatch):
    """Test 11, the slice limits in order: an override skips the search and its checks; below
    the floor raises naming the budget; below 96 warns; an override above the need is kept."""
    fake_builds["name"] = "voxel"
    seen = {}
    monkeypatch.setattr(api, "si_band", stub_band(seen))
    band = fband("voxel", 3, 12)
    kw = dict(strategy="shift_invert", n_modes=10)
    _, _, info = run("voxel", band, memory_budget=1.0, slice_options={"capacity": 48}, **kw)
    assert info["capacity"] == seen["capacity"] == 48
    _, _, info = run("voxel", band, slice_options={"capacity": 5000}, **kw)
    assert info["capacity"] == 5000
    floor = slice_budget("voxel", 72, 10)
    with pytest.raises(ValueError, match="smallest shift-invert slice does not fit memory_budget"):
        run("voxel", band, memory_budget=floor * 0.999, **kw)
    with pytest.warns(UserWarning, match="slice capacity 72 is below 96"):
        run("voxel", band, memory_budget=floor, **kw)


def test_count_block_fit():
    """Test 11: the slice count block is the largest passing the fit with the store and window
    as exact host bytes, per space."""
    op, pc = models("voxel")
    n = prob("voxel")[2].n_dof
    for B, H in ((60e6, None), (40e6, 30e6)):
        got = mem._count_block(op, pc, n, voxel=True, n_rigid=0, n_store=50, window=100,
                               n_probes=50, budget=B, host_budget=H)  # fmt: skip
        ok = [b for b in range(1, 51) if mem._fits(mem.count_memory(
            op, n, b, voxel=True, precond_memory=pc), B, H, 8 * n * 150)]  # fmt: skip
        assert got == (max(ok) if ok else None)


@pytest.mark.parametrize("verify", [True, None, False])
def test_count_block_none(fake_builds, monkeypatch, verify):
    """Test 11: no probe fits beside the store and window: the counts are off (a warning under
    True only), the reason recorded."""
    fake_builds["name"] = "voxel"
    seen = {}
    monkeypatch.setattr(api, "si_band", stub_band(seen))
    monkeypatch.setattr(mem, "_count_block", lambda *a, **kw: None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, info = run("voxel", fband("voxel", 3, 12), strategy="shift_invert", n_modes=10,
                         verify=verify)  # fmt: skip
    msgs = [str(x.message) for x in w if "no counts run" in str(x.message)]
    assert len(msgs) == (verify is True) and seen["verify"] is False
    assert "one probe" in info["count"]["reason"]


def test_solve_modes_memory_band():
    with pytest.raises(ValueError, match="needs band=\\(0, f_max\\)"):
        mem.solve_modes_memory(prob("short")[0], MESHES["short"][1], band=(1.0, 2.0))


# ---------------------------------------------------------------- 14: si_band(verify=None)

SKIP_BOUND = "predicted cost (a lower bound) above the solve"
SKIP_CAL = "predicted cost above the solve"
SKIP_DEG = "degree above MAX_DEGREE: the gap is too small to count"


class Clock:
    """A fake perf_counter advanced by hand."""

    def __init__(self):
        self.t = 0.0

    def perf_counter(self):
        return self.t


def decide(verify, d=7, t_ref=1.0, t_it=None, tau=None, step=0.0, block=0.0, monkeypatch=None):
    """counting._decide with 50 probes in blocks of 10 (5 blocks), m = 6, 100 free DOFs; run(s)
    returns the times step and block; rec: the run's steps, prepare's calls."""
    rec = dict(runs=[], prepared=0)
    clock = Clock()
    monkeypatch.setattr(counting, "time", clock)

    def run(s):
        rec["runs"].append(s)
        return np.nan, dict(time=dict(step=step, block=block))

    def prepare():
        rec["prepared"] += 1
        clock.t += 0.25

    go, out = counting._decide(verify, d, 50, 10, 6, t_ref, t_it, tau, 100, run, prepare)
    return go, out, rec


def test_decide_table(monkeypatch):
    """Test 14: the count rule as a pure table."""
    bound2 = 2.0 * 35 * 10 / 6 * 1e-3  # 5 blocks x d = 7 steps, b = 10, m = 6, t_it 1e-3
    for verify in (True, False):
        go, r, rec = decide(verify, t_it=1e-3, monkeypatch=monkeypatch)
        assert go is verify and r["predicted_kind"] == "bound" and not rec["runs"]
        assert r["predicted_s"] == pytest.approx(bound2) and r["predicted_steps"] == 35
        assert r["reason"] is None and not rec["prepared"]
    go, r, rec = decide(True, monkeypatch=monkeypatch)  # no warm time: no bound
    assert go and r["predicted_kind"] == "unknown" and r["predicted_s"] is None
    go, r, rec = decide(None, d=None, monkeypatch=monkeypatch)
    assert not go and r["reason"] == SKIP_DEG and not rec["runs"]
    go, r, rec = decide(None, t_it=1e-3, t_ref=0.5 * bound2, monkeypatch=monkeypatch)
    assert not go and r["reason"] == SKIP_BOUND and not rec["prepared"] and not rec["runs"]
    bound3 = 0.3 / 2.0 * bound2  # tau looser than the rule: one PCG iteration a step
    go, r, rec = decide(None, t_it=1e-3, tau=1.0, t_ref=0.5 * bound3, monkeypatch=monkeypatch)
    assert not go and r["reason"] == SKIP_BOUND and r["predicted_s"] == pytest.approx(bound3)
    go, r, rec = decide(None, t_it=1e-3, tau=1.0, t_ref=1.5 * bound3, step=1.0, block=1.0,
                        monkeypatch=monkeypatch)  # fmt: skip
    assert not go and r["reason"] == SKIP_CAL and rec == dict(runs=[1, 3], prepared=1)
    assert r["predicted_kind"] == "calibrated" and r["build_s"] == 0.25
    assert r["predicted_s"] == pytest.approx(5 * (7 + 1) + 0.25)
    go, r, rec = decide(None, t_it=1e-3, tau=1.0, t_ref=1.0, step=0.01, block=0.02,
                        monkeypatch=monkeypatch)  # fmt: skip
    assert go and r["reason"] is None and r["calibrated_step"] == 0.01
    assert r["predicted_s"] == pytest.approx(5 * (7 * 0.01 + 0.02) + 0.25)


def clocked_slice(monkeypatch, lo, hi, *, move=False, fail=False, capacity=96):
    """si_slice on "short" (exact preconditioner) under a fake clock: MINRES call 1 takes 10 s,
    later calls 1e-3 s per iteration; move: call 1 reports stagnation (sigma moves); fail: call
    1 returns non-finite residuals. Returns info and the per-call (seconds, iterations)."""
    clock, calls = Clock(), []
    real = si.minres

    def minres(A, b, P, **kw):
        x, info = real(A, b, P, **kw)
        its = int(np.asarray(info["iterations"]).max())
        dt = 10.0 if not calls else 1e-3 * its
        clock.t += dt
        calls.append((dt, its))
        if len(calls) == 1 and (move or fail):
            info = dict(info, stagnated=jnp.ones_like(info["stagnated"]),
                        residuals=jnp.full_like(info["residuals"], np.nan if fail else 1.0))
        return x, info

    monkeypatch.setattr(si, "minres", minres)
    monkeypatch.setattr(si, "time", clock)
    _, free, op = prob("short")[:3]
    _, _, info = si.si_slice(op, free, lo, hi, exact("short", (lo + hi) / 2), capacity=capacity,
                             tol=TOL, tau=TAU)  # fmt: skip
    return info, calls


def test_t_minres_warm(monkeypatch):
    """Test 14: the warm MINRES seconds per iteration drop the discarded move solve and the
    first kept step (its compile): (sum - move - first) / (iterations - first's)."""
    lo, hi = edges(prob("short")[6], 17, 24)
    info, calls = clocked_slice(monkeypatch, lo, hi, move=True)
    its = info["minres_iterations"]
    assert info["sigma_moved"] and len(its) >= 2 and [i for _, i in calls[1:]] == its
    want = (sum(d for d, _ in calls) - calls[0][0] - calls[1][0]) / (sum(its) - its[0])
    assert info["t_minres"] == pytest.approx(want)
    assert info["time"]["total"] == pytest.approx(sum(d for d, _ in calls))


def test_t_minres_cold(monkeypatch):
    """Test 14: a one-step slice (mode 3 of "short", _slice_check reporting done at once) gives
    the cold mean; a failed first step gives None."""
    lo, hi = edges(prob("short")[6], 3, 4)
    real = si._slice_check
    monkeypatch.setattr(si, "_slice_check", lambda *a: (True, *real(*a)[1:]))
    info, calls = clocked_slice(monkeypatch, lo, hi)
    assert info["steps"] == 1 and len(calls) == 1
    assert info["t_minres"] == pytest.approx(10.0 / calls[0][1])
    monkeypatch.undo()
    info, _ = clocked_slice(monkeypatch, lo, hi, fail=True)
    assert info["reason"] == "inner solves failed" and info["t_minres"] is None


def stub_count(monkeypatch, *, degree=7, capped=False, step=0.0, block=0.0, est=0.1):
    """si._count_slice replaced: a _predict call returns the degree, a calibration (_steps) the
    times step and block, a real count est (or est(call)); calls: (_predict, _steps) each."""
    calls, real = [], []

    def count_slice(*a, _predict=False, _steps=None, **kw):
        calls.append((_predict, _steps))
        t = dict(step=step, block=block, total=0.0)
        ci = dict(capped=capped, degree=degree, block=10, time=t, steps=_steps or 1)
        if _predict or _steps:
            return np.nan, ci
        real.append(kw)
        e = est(len(real)) if callable(est) else est
        return e, dict(ci, verifiable=True, passed=e < 0.5)

    monkeypatch.setattr(si, "_count_slice", count_slice)
    return calls


def leaf(monkeypatch, verify, t_leaf=1.0, t_minres=1e-3, **kw):
    """si_checked on CB with ref_slice (time t_leaf, t_minres) and the _count_slice stub."""
    spy_slice(monkeypatch, ref_slice, lambda j, out: (out[0], out[1], dict(
        out[2], time=dict(total=t_leaf), t_minres=t_minres)))  # fmt: skip
    calls = stub_count(monkeypatch, **{k: kw.pop(k) for k in list(kw) if k != "count_options"})
    _, free, op = prob("short")[:3]
    lo, hi = cb_edges()
    _, _, info = si.si_checked(op, free, lo, hi, None, capacity=300, verify=verify, **kw)
    return info["count"], calls


def test_leaf_branches(monkeypatch):
    """Test 14, run leaves under None: each branch makes one _predict call; calibrations
    (_steps 1, 3) only on calibrated branches; the real count only on the count branch."""
    P, C1, C3, REAL = (True, None), (False, 1), (False, 3), (False, None)
    c, calls = leaf(monkeypatch, None, capped=True)
    assert c["reason"] == SKIP_DEG and calls == [P] and not c["ran"]
    monkeypatch.undo()
    c, calls = leaf(monkeypatch, None, t_leaf=1e-9)
    assert c["reason"] == SKIP_BOUND and calls == [P] and c["predicted_kind"] == "bound"
    monkeypatch.undo()
    c, calls = leaf(monkeypatch, None, t_leaf=1e-9, count_options={"tau": 1.0})
    assert c["reason"] == SKIP_BOUND and calls == [P]
    assert c["predicted_s"] == pytest.approx(0.3 * 5 * 7 * 10 / 6 * 1e-3)
    monkeypatch.undo()
    c, calls = leaf(monkeypatch, None, t_minres=None, step=1.0, block=1.0)
    assert c["reason"] == SKIP_CAL and calls == [P, C1, C3] and c["predicted_kind"] == "calibrated"
    monkeypatch.undo()
    cal = []
    real = counting._calibrate
    monkeypatch.setattr(counting, "_calibrate", lambda *a: cal.append(a[1:]) or real(*a))
    c, calls = leaf(monkeypatch, None, t_leaf=100.0, t_minres=None)
    assert c["ran"] and c["reason"] is None and calls == [P, C1, C3, REAL] and cal == [(7, 5)]


@pytest.mark.parametrize("verify", [True, False])
def test_leaf_recorded(monkeypatch, verify):
    """Test 14: under True and False the prediction is recorded, no calibration."""
    c, calls = leaf(monkeypatch, verify)
    assert c["predicted_kind"] == "bound" and c["predicted_s"] > 0
    assert calls == [(True, None)] + ([(False, None)] if verify else [])
    assert c["reason"] == (None if verify else "not requested")


def test_leaf_restart(monkeypatch):
    """Test 14: restarts follow the first decision: one _predict call and one _decide call,
    two real counts (0.8 verifiable, then 0.1), restarts 1."""
    decided, real = [], counting._decide
    monkeypatch.setattr(counting, "_decide", lambda *a: decided.append(1) or real(*a))
    c, calls = leaf(monkeypatch, True, est=lambda j: 0.8 if j == 1 else 0.1)
    assert decided == [1] and calls.count((True, None)) == 1
    assert calls.count((False, None)) == 2 and c["restarts"] == 1 and c["passed"]


def span_spy(monkeypatch):
    """si._span_count wrapped: its t_leaf / t_minres and the counting._decide calls inside."""
    rec, decided = [], []
    real_decide, real_span = counting._decide, si._span_count
    monkeypatch.setattr(counting, "_decide", lambda *a: decided.append(a[0]) or real_decide(*a))

    def span(*a, **kw):
        n0 = len(decided)
        out = real_span(*a, **kw)
        rec.append(dict(t_leaf=kw["t_leaf"], t_minres=kw["t_minres"], decides=len(decided) - n0))
        return out

    monkeypatch.setattr(si, "_span_count", span)
    return rec


@pytest.mark.parametrize("verify", [None, True, False])
def test_span_inherits(monkeypatch, verify):
    """Test 14: a span leaf is decided once, on its parent run's t_leaf and t_minres (fake_band
    post): under None 1e-9 s against 1 s per MINRES iteration skips on the bound; under True
    and False the bound is recorded."""

    def post(j, info):
        return dict(info, count=dict(info["count"], t_leaf=1e-9, t_minres=1.0))

    fake_band(monkeypatch, tsi.split_once, post=post)
    rec = span_spy(monkeypatch)
    lo, hi = edges(prob("short")[6], 3, 16)
    _, _, info = tsi.fband("short", lo, hi, verify=verify)
    (sp,) = [x for x in info["slices"] if x["kind"] == "span"]
    assert rec == [dict(t_leaf=1e-9, t_minres=1.0, decides=1)]
    assert sp["count"]["predicted_kind"] == "bound"
    want = {None: SKIP_BOUND, True: None, False: "not requested"}[verify]
    assert sp["count"]["reason"] == want and sp["count"]["ran"] == bool(verify)


def slice_fake(monkeypatch, r_of):
    """si_slice replaced: each call converges the reference modes in [a, e] new to it (r_acc
    r_of.get(call, TOL), 1 s, 1e-3 s per MINRES iteration); returns the intervals."""
    lr, Vr = prob("short")[6:]
    calls = []

    def fake(op, free, a, e, P, **kw):
        calls.append((a, e))
        i = tsi.new_modes(a, e, kw)
        below, above = np.flatnonzero(lr < a), np.flatnonzero(lr > e)
        nb = {"lo": (lr[below[-1]], Vr[:, below[-1]]) if len(below) else None,
              "hi": (lr[above[0]], Vr[:, above[0]]) if len(above) else None}  # fmt: skip
        z = np.zeros(len(i))
        info = dict(converged=True, reason=None, sigma=(a + e) / 2, neighbours=nb, residuals=z,
                    thresholds=z + TOL, r_acc=r_of.get(len(calls), TOL), tau_eff=TAU, basis=0,
                    ritz=None, time=dict(total=1.0), t_minres=1e-3)  # fmt: skip
        return lr[i], Vr[:, i], info

    monkeypatch.setattr(si, "si_slice", fake)
    return calls


def test_rerun_redecides(monkeypatch):
    """Test 14: a slice whose r_acc widens its window over a stored mode is rerun, and the rerun
    decides again: counting._bound once per si_checked call, twice on the rerun slice."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 3, 16)
    s = (lr[11] + lr[12]) / 2
    slice_fake(monkeypatch, {2: 1e-4})
    stub_count(monkeypatch)
    bounds, checked = [], []
    real_bound, real_checked = counting._bound, si.si_checked
    monkeypatch.setattr(counting, "_bound", lambda *a: bounds.append(len(checked))
                        or real_bound(*a))  # fmt: skip

    def chk(op, free, a, e, P, **kw):
        checked.append((a, e))
        return real_checked(op, free, a, e, P, **kw)

    monkeypatch.setattr(si, "si_checked", chk)
    tsi.fband("short", lo, hi, edges=[s], block=1)
    assert checked == [(lo, s), (s, hi), (s, hi)] and bounds == [1, 2, 3]


@pytest.mark.parametrize("verify", [None, True])
def test_real_band_verify(verify, monkeypatch):
    """Test 14: a real si_band (exact preconditioner, test 4's band at capacity 96: >= 2 runs, a
    span): every leaf records its reason and predicted kind, each span leaf decided once on its
    parent's t_leaf / t_minres, never "not requested"; under True every leaf counted < 0.5."""
    rec = span_spy(monkeypatch)
    lo, hi = edges(prob("short")[6], 5, 29)
    lam, V, info = tsi.rband("short", lo, hi, capacity=96, verify=verify)
    tsi.check_band("short", lo, hi, lam, V, info, counted=verify is True)
    spans = [x for x in info["slices"] if x["kind"] == "span"]
    assert spans and sum(x["kind"] == "run" for x in info["slices"]) >= 2
    assert len(rec) == len(spans) and all(r["decides"] == 1 for r in rec)
    runs = [x for x in info["slices"] if x["kind"] == "run"]
    for r, x in zip(rec, spans, strict=True):
        parent = runs[x["parent"]]["count"]  # parent: the index among the runs
        assert (r["t_leaf"], r["t_minres"]) == (parent["t_leaf"], parent["t_minres"])
    for x in [x for x in info["slices"] if x["leaf"]]:  # a full slice is never decided
        c = x["count"]
        assert c["predicted_kind"] in ("unknown", "bound", "calibrated")
        assert c["reason"] != "not requested"
        if verify:
            assert c["ran"] and c["estimate"] < 0.5 and c["reason"] is None


def test_solve_modes_records(fake_builds):
    """Test 14: solve_modes(strategy="shift_invert") at the default verify records the rule."""
    _, _, info = run("short", fband("short", 15, 25), n_modes=12, strategy="shift_invert")
    for x in [x for x in info["slices"] if x["leaf"]]:
        assert {"reason", "predicted_kind", "predicted_s"} <= set(x["count"])


@pytest.mark.slow
def test_leaf_calibration(monkeypatch):
    """Test 14 (as 14b's test_calibration): one leaf on hex20 12^3 with real AMG under None,
    the count forced: the bound <= the measured count time, the calibrated prediction within
    [0.5, 2] x it."""
    from jaxmodal.mesh import box_mesh
    from jaxmodal.operators import dof_mask, unstructured
    from jaxmodal.precond import amg

    rec = {}
    real_cal, real_bound = counting._calibrate, counting._bound

    def calibrate(*a):
        rec["cal"] = real_cal(*a)
        return (0.0, 0.0) + rec["cal"][2:]  # force the count

    monkeypatch.setattr(counting, "_calibrate", calibrate)
    monkeypatch.setattr(counting, "_bound", lambda *a: rec.update(bound=real_bound(*a)) or 0.0)
    mesh = box_mesh((12,) * 3, (1 / 12,) * 3, type="hex20")
    f = solve_modes(mesh, MAT, {"xmin": "xyz"}, 8)[2]["frequencies"]
    lam = (2 * np.pi * f) ** 2
    lo, hi = (lam[2] + lam[3]) / 2, (lam[6] + lam[7]) / 2
    free = dof_mask(mesh, {"xmin": "xyz"})
    op = unstructured(mesh, MAT)
    P = amg(op, free, (lo + hi) / 2, mesh.points, nonsingular=True)
    _, _, info = si.si_checked(op, free, lo, hi, P, capacity=300, tol=1e-8, verify=None,
                               count_options={"n_probes": 12})  # fmt: skip
    c = info["count"]
    actual = c["time"]["total"]
    print(f"calibrated {rec['cal'][0]:.2f} s, actual {actual:.2f} s, bound {rec['bound']:.2f} s")
    assert c["ran"] and c["predicted_kind"] == "calibrated"
    assert rec["bound"] <= actual and 0.5 * actual <= rec["cal"][0] <= 2 * actual


def test_store_warning(monkeypatch):
    """Test 14: the store warning once, when the store passes n_store."""
    fake_band(monkeypatch, tsi.found)
    lo, hi = edges(prob("short")[6], 3, 16)
    with pytest.warns(UserWarning, match="stored modes exceed n_store = 5") as w:
        tsi.fband("short", lo, hi, edges=[(lo + hi) / 2], n_store=5)
    assert sum("exceed n_store" in str(x.message) for x in w) == 1
