"""PR 18f tests 3 and 4: si_band's MINRES deflation columns Z and the known pairs of the
LOBPCG/slice junction, assembled in place (fakes on "short", docs/eigensolvers.md#the-junction)."""

import importlib
import warnings

import numpy as np
import pytest
from test_shift_invert import TOL, edges, fake_band, fband, found, prob

si = importlib.import_module("jaxmodal.eigen.shift_invert")


# ---------------------------------------------------------------- 3: Z per work item


def test_z_cols():
    """Test 3: the nearest n stored values strictly below sigma, unsorted input."""
    lam = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    assert si._z_cols(lam, 3.0, 5).tolist() == [1, 2]  # 3 = sigma excluded
    assert si._z_cols(lam, np.nextafter(3.0, 4), 1).tolist() == [0]  # just above: 3 included
    assert si._z_cols(lam, np.nextafter(3.0, 0), 5).tolist() == [1, 2]  # just below
    assert si._z_cols(lam, 4.5, 2).tolist() == [0, 4]  # the 2 nearest: 3 and 4
    assert si._z_cols(lam, 4.5, 0).tolist() == []
    assert si._z_cols(np.zeros(0), 1.0, 3).tolist() == []


def test_band_z(monkeypatch):
    """Test 3: each slice gets Z = views of the stored modes nearest below its midpoint, capped
    by max_deflate (0, the default: none); a bool, None or negative max_deflate and Z in the
    slice options raise."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 5, 15)
    cuts = [(lr[8] + lr[9]) / 2, (lr[11] + lr[12]) / 2]
    for d, want in ((3, [0, 3, 3]), (0, [0, 0, 0])):
        calls, _ = fake_band(monkeypatch, found)
        lam, V, info = fband("short", lo, hi, edges=cuts, max_deflate=d)
        assert info["converged"] and [c["Z"] for c in calls] == want
        for c in calls:
            a, e = c["interval"]
            stored = lr[(lr >= lo) & (lr < (a + e) / 2) & (lr < a)]  # earlier slices' modes
            assert np.allclose(c["Z_lam"], stored[-d:] if d else [], rtol=1e-10)
            assert all(c["Z_view"])
    calls, _ = fake_band(monkeypatch, found)
    fband("short", lo, hi, edges=cuts)
    assert [c["Z"] for c in calls] == [0, 0, 0]
    for bad in (None, True, -1, 1.0):
        with pytest.raises(ValueError, match="max_deflate must be an int >= 0"):
            fband("short", lo, hi, max_deflate=bad)
    with pytest.raises(ValueError, match="si_band sets"):
        fband("short", lo, hi, Z=[lr])


# ---------------------------------------------------------------- 4: known pairs


def known_block(j, width=None, base=None):
    """[lam, V, r, t, None]: the reference modes 0..j-1 in an F-order block of width columns
    (default j + 3; the columns past j zero), or a view of base's leading columns."""
    lr, Vr = prob("short")[6:]
    n = Vr.shape[0]
    V = np.zeros((n, width or j + 3), order="F") if base is None else base[:, : width or j + 3]
    V[:, :j] = Vr[:, :j]
    return [lr[:j].copy(), V, np.arange(j) * 1e-12, np.full(j, 1e-9), None]


def band(j, top=15):
    """(lo, hi) at the midpoints below mode j and above mode top - 1."""
    lr = prob("short")[6]
    return (lr[j - 1] + lr[j]) / 2, (lr[top - 1] + lr[top]) / 2


def run_known(known, lo, hi, *, held=False, warn=False, **kw):
    """fband with known, asserting the in-place path (the block's id returned, no warning) or,
    with warn, the copy and its RuntimeWarning."""
    vid = id(known[1])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lam, V, info = fband("short", lo, hi, known=known, **kw)
    copied = [x for x in w if "returned modes copied" in str(x.message)]
    assert known == [] and bool(copied) == warn and (id(V) == vid) != warn
    if not warn:
        assert V.base is None and V.flags.f_contiguous and V.shape == (V.shape[0], len(lam))
    return lam, V, info


def assert_modes(lam, V, ids, exact=()):
    """lam the reference values of ids, V their vectors (up to sign in a merged window, exact
    for the columns in exact)."""
    lr, Vr = prob("short")[6:]
    M = prob("short")[4]
    assert np.allclose(lam, lr[ids], rtol=1e-10, atol=0)
    assert np.allclose(np.abs(Vr[:, ids].T @ (M @ V)), np.eye(len(ids)), atol=1e-8)
    for q in exact:
        assert np.array_equal(V[:, q], Vr[:, ids[q]])


def test_known(monkeypatch):
    """Test 4: the known pairs come first (with their residuals and thresholds), the first
    slice's window holds the top known value, known is emptied, the result is assembled in
    place in the given block (contents exact); a leaf is slices[0] and counted."""
    lr = prob("short")[6]
    lo, hi = band(5)
    calls, _ = fake_band(monkeypatch, found)
    lam, V, info = run_known(known_block(5), lo, hi, edges=[(lr[9] + lr[10]) / 2])
    ids = list(range(15))
    assert_modes(lam, V, ids, exact=ids)
    assert np.array_equal(info["residuals"][:5], np.arange(5) * 1e-12)
    assert (info["thresholds"][:5] == 1e-9).all() and (info["thresholds"][5:] == TOL).all()
    assert lr[4] in calls[0]["lam_W"] and lr[0] not in calls[0]["lam_W"]
    assert [s["kind"] for s in info["slices"]] == ["run", "run"]
    # a flagged leaf: slices[0], and the band count covers it
    leaf = ((0.0, lo), dict(passed=False, verifiable=True, ran=True, flagged=True, estimate=0.6,
                            estimates=[0.6], reason="flagged", restarts=0))  # fmt: skip
    known = known_block(5)
    known[4] = leaf
    lam, V, info = run_known(known, lo, hi, edges=[(lr[9] + lr[10]) / 2])
    assert info["slices"][0] == dict(interval=(0.0, lo), kind="known", leaf=True, count=leaf[1])
    c = info["count"]
    assert c["flagged"] and not c["passed"] and c["estimate"] == 0.6
    assert c["reason"] == f"[0, {lo:.6g}]: flagged"
    assert info["merged_edges"] == [] and not info["collapsed"]  # the leaf's lo_b no bound


def test_known_error(monkeypatch):
    """Test 4: err.result after the first slice raises (max_slices = 1) holds the known pairs,
    assembled in place, and the leaf in the count; a non-converged slice (a break) keeps the
    block too."""
    lr = prob("short")[6]
    lo, hi = band(5)
    fake_band(monkeypatch, found)
    known = known_block(5)
    leaf = ((0.0, lo), dict(passed=True, verifiable=True, ran=True, flagged=False,
                            estimate=0.1, estimates=[0.1], reason=None, restarts=0))  # fmt: skip
    known[4], vid = leaf, id(known[1])
    with pytest.raises(RuntimeError, match="max_slices") as err:
        fband("short", lo, hi, known=known, edges=[(lr[9] + lr[10]) / 2], max_slices=1)
    lam, V, info = err.value.result
    assert id(V) == vid and info["slices"][0]["kind"] == "known" and info["count"]["passed"]
    assert_modes(lam, V, list(range(10)), exact=range(10))

    def stop(k, a, e, kw):
        return found(k, a, e, kw)[0], "inner solve failed", None, None

    fake_band(monkeypatch, stop)
    lam, V, info = run_known(known_block(5), lo, hi)
    assert not info["converged"] and "inner solve failed" in info["reason"]
    assert_modes(lam, V, list(range(15)), exact=range(15))


def test_known_grow(monkeypatch):
    """Test 4: slice parts wider than the spare columns plus c grow the block in steps (block 2:
    c = 8; V of j + 1 columns, 20 slice modes), in place, contents exact."""
    lo, hi = band(5, 25)
    fake_band(monkeypatch, found)
    lam, V, info = run_known(known_block(5, width=6), lo, hi, block=2)
    assert len(lam) == 25
    assert_modes(lam, V, list(range(25)), exact=range(25))


def test_known_fallbacks(monkeypatch):
    """Test 4: another reference (the probe grow raises before any write: the holder's block
    unchanged), a borrowed block and a known index part after a merged window copy with the
    warning, same contents."""
    lr, Vr = prob("short")[6:]
    lo, hi = band(5)
    fake_band(monkeypatch, found)
    known = known_block(5)
    held, before = known[1], known[1].copy()
    lam, V, info = run_known(known, lo, hi, warn=True)
    assert np.array_equal(held, before) and id(V) != id(held)
    assert_modes(lam, V, list(range(15)), exact=range(15))
    del held
    big = np.zeros((Vr.shape[0], 20), order="F")
    lam, V, info = run_known(known_block(5, base=big), lo, hi, warn=True)
    assert_modes(lam, V, list(range(15)), exact=range(15))

    def dup0(k, a, e, kw):  # the first slice re-finds known mode 0: a merged window below known
        p = found(k, a, e, kw)[0]
        return ((lr[[0, *p]], Vr[:, [0, *p]]) if k == 1 else p), None, None, None

    fake_band(monkeypatch, dup0)
    lam, V, info = run_known(known_block(5), lo, hi, warn=True)
    assert info["duplicates"] == 1
    assert_modes(lam, V, list(range(15)), exact=range(1, 15))


@pytest.mark.parametrize("j", [5, 2, 1])
def test_known_prefix(monkeypatch, j):
    """Test 4: a slice that re-finds the top known mode merges with it (a duplicate), which
    only shortens the in-place prefix (p = j - 1; j = 2: p = 1, j = 1: p = 0)."""
    lr, Vr = prob("short")[6:]
    lo, hi = band(j)

    def dup(k, a, e, kw):
        p = found(k, a, e, kw)[0]
        return ((lr[[j - 1, *p]], Vr[:, [j - 1, *p]]) if k == 1 else p), None, None, None

    fake_band(monkeypatch, dup)
    lam, V, info = run_known(known_block(j), lo, hi)
    assert info["duplicates"] == 1
    ids = list(range(15))
    assert_modes(lam, V, ids, exact=[q for q in ids if q != j - 1])


def test_known_widened(monkeypatch):
    """Test 4: a merged window (known mode 4 and a slice's junk vector at its value) fails its
    re-check and widens over the known-only window below it (mode 3) and the window above: the
    prefix shortens to 3, still in place, edge_rr_failed."""
    lr, Vr = prob("short")[6:]
    free, M = prob("short")[1], prob("short")[4]
    lo, hi = band(5)
    w = np.where(free, np.random.default_rng(0).standard_normal(Vr.shape[0]), 0)
    w /= np.sqrt(w @ (M @ w))

    def junk(k, a, e, kw):
        p = found(k, a, e, kw)[0]
        return ((np.r_[lr[4], lr[p]], np.c_[w, Vr[:, p]]) if k == 1 else p), None, None, None

    fake_band(monkeypatch, junk)
    lam, V, info = run_known(known_block(5), lo, hi)
    assert info["edge_rr_failed"] and len(lam) == 16 and info["duplicates"] == 0
    for q in range(3):
        assert np.array_equal(V[:, q], Vr[:, q])
    # the widened merge also takes mode 5 (above): modes 3-5 and the junk, then 6-14 in place
    assert np.allclose(np.abs(Vr[:, 3:6].T @ (M @ V[:, 3:6])), np.eye(3), atol=1e-8)
    for q in range(6, 15):
        assert np.array_equal(V[:, q + 1], Vr[:, q])
