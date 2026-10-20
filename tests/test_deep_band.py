"""PR 18f test 10: a deep band through the junction, hex20 10^3 clamped: LOBPCG capacity-stops
near depth 280, every locked mode lies below the band (case (b)), and the slices from lo
deflate the known modes in MINRES (docs/eigensolvers.md#the-junction)."""

import functools
import importlib
import math
import warnings

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from jaxmodal.eigen import solve_modes
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import assemble, dof_mask

api = importlib.import_module("jaxmodal.eigen.api")
mem = importlib.import_module("jaxmodal.memory")

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
BCS = {"xmin": "xyz"}
TOL = 1e-8


@functools.cache
def problem():
    mesh = box_mesh((10, 10, 10), (0.1,) * 3, type="hex20")
    free = dof_mask(mesh, BCS)
    K, M = assemble(mesh, MAT)
    f = np.flatnonzero(free)
    lam = spla.eigsh(K[f][:, f].tocsc(), 310, M[f][:, f].tocsc(), sigma=0, which="LM")[0]
    return mesh, free, K, M, np.sort(lam)


def gap_edge(lam, i):
    """The midpoint below mode i, moved down to a relative gap >= 1e-2."""
    while lam[i] - lam[i - 1] < 1e-2 * lam[i]:
        i -= 1
    return (lam[i - 1] + lam[i]) / 2


@pytest.mark.slow
def test_deep_band(monkeypatch):
    """Test 10: chunk 40, levers off, the smallest budget with C_max >= 240 (256: LOBPCG stops at
    D = 283, 243 locked plus the kept stopping chunk); the band of modes 285-305 (gaps 1.3e-2,
    8.5e-3) lies above every locked mode (case (b): junction {cut: None, known:
    j}, j the locked modes below lo (1 - 100 max(t, tol))); every band mode found against eigsh,
    each slice's MINRES deflated. verify=False: the band count needs about 2700 steps here (about
    34 min); test_capacity counts after a junction."""
    mesh, free, K, M, lr = problem()
    n, k = len(free), 0
    lo, hi = gap_edge(lr, 285), (lr[305] + lr[306]) / 2
    op_m, pc = mem._models(mesh, "consistent", "auto", "float64", {}, None)
    n_avail = int(free.sum())

    def cmax(B):
        return mem._c_max(op_m, pc, n, chunk=40, guards=None, low=False, basis_dtype="float64",
                          n_rigid=k, n_avail=n_avail, budget=B, host_budget=None) or 0  # fmt: skip

    a, b = 1e8, 1e11  # the smallest budget with C_max >= 240
    while b - a > 1e-6 * b:
        mid = (a + b) / 2
        a, b = (mid, b) if cmax(mid) < 240 else (a, mid)
    seen = {}
    real_solve, real_si = api._solve, api.si_band

    def solve(*a_, **kw):
        out = real_solve(*a_, **kw)
        seen.update(lam=out[0].copy(), t=out[2]["thresholds"].copy())
        return out

    def band(*a_, **kw):
        seen.update(max_deflate=kw["max_deflate"], j=len(kw["known"][0]))
        return real_si(*a_, **kw)

    monkeypatch.setattr(api, "_solve", solve)
    monkeypatch.setattr(api, "si_band", band)
    f1, f2 = (math.sqrt(x) / (2 * math.pi) for x in (lo, hi))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        kw = dict(n_modes=25, chunk=40, levers=False, memory_budget=b, tol=TOL, verify=False)
        lam, V, info = solve_modes(mesh, MAT, BCS, band=(f1, f2), **kw)
    D = len(seen["lam"])
    assert info["reason"] == "capacity" and 250 <= D and seen["lam"].max() < lo
    j = int(np.sum(seen["lam"] < lo * (1 - 100 * np.maximum(seen["t"], TOL))))
    assert info["junction"] == {"cut": None, "known": j} and j == seen["j"] == D
    assert seen["max_deflate"] >= j  # the memory fits full deflation here
    ref = lr[(lr >= lo) & (lr <= hi)]
    assert len(lam) == len(ref) and np.allclose(lam, ref, rtol=1e-7)
    f = np.flatnonzero(free)
    Mf = M[f][:, f]
    assert np.abs(V[f].T @ (Mf @ V[f]) - np.eye(len(lam))).max() <= 1e-6
    runs = [s for s in info["slices"] if s["kind"] == "run"]
    assert runs and all(s["deflated"] > 0 for s in runs)
    print(f"D {D}, j {j}, d {seen['max_deflate']}, first slice MINRES",
          runs[0]["minres_iterations"][:3], "time", info["time"])  # fmt: skip
