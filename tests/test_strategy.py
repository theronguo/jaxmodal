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
from test_shift_invert import EPS, MAT, MESHES, edges, exact, lam_hat, prob

from jaxmodal.eigen import counting, solve_modes
from jaxmodal.eigen.lobpcg import _band_count
from jaxmodal.precond import SingularShiftError

api = importlib.import_module("jaxmodal.eigen.api")
lob = importlib.import_module("jaxmodal.eigen.lobpcg")
si = importlib.import_module("jaxmodal.eigen.shift_invert")
mem = importlib.import_module("jaxmodal.memory")

pytestmark = pytest.mark.filterwarnings("error:returned modes copied")
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
