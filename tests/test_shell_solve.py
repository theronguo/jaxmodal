"""PR 18b: solve_modes with solid-shell AMG: dispatch, accuracy against reference_modes, the
low-memory switch, the polish, the failed-exit Rayleigh-Ritz, the shell memory family and band
mode."""

import functools
import importlib
import sys

import helpers
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import test_shell_amg as T
from exact import SIMPLY_SUPPORTED as SS
from helpers import STEEL, max_sine, max_sine_sparse, owned_bytes, r_M, r_M_sparse
from test_api import check_lam
from test_shell import C_ORT, laminate, three_plies
from test_shell_ops import hex20_skin

from jaxmodal.eigen import reference_modes, solve_modes
from jaxmodal.memory import (
    STEP_FIT,
    amg_memory,
    count_memory,
    gmg_memory,
    lobpcg_memory,
    si_memory,
    solve_modes_memory,
    unstructured_memory,
)
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import assemble, dof_mask, restrict, unstructured

LOB = importlib.import_module("jaxmodal.eigen.lobpcg")
API = importlib.import_module("jaxmodal.eigen.api")
MEM = importlib.import_module("jaxmodal.memory")
EPS = np.finfo(float).eps

# ---------------------------------------------------------------- 2: accuracy

PLIES = {0: laminate(three_plies(C_ORT))}
CASES = {  # (mesh, bcs, materials); L/t = 50 unless stated
    "flat": lambda: (T.plate(32, 50)[0], SS, STEEL),
    "free": lambda: (T.plate(24, 50)[0], {}, STEEL),
    "stack": lambda: (T.plate(32, 100, layers=3)[0], SS, STEEL),  # L/t = 100
    "curved": lambda: (T.plate(32, 50, curved=True)[0], SS, STEEL),
    "cylinder": lambda: (T.cylinder(False), {}, STEEL),
    "T": lambda: (T.junction("T", 24, 50)[0], T.CLAMPED, STEEL),
    "web": lambda: (T.web(32, 50)[0], SS, STEEL),
    "plies": lambda: (T.plate(32, 50)[0], SS, PLIES),
}


def groups(lr, rel=1e-2):
    """Index groups of lr (ascending) split where the relative gap is >= rel (check_lam's)."""
    cut = np.flatnonzero(np.diff(lr) / lr[1:] >= rel) + 1
    return np.split(np.arange(len(lr)), cut)


@functools.cache
def _ref(mesh):
    """36 reference eigenvalues and lam_max of the SS steel mesh."""
    K, M = assemble(mesh, STEEL)
    Kf, Mf, _ = restrict(K, M, dof_mask(mesh, SS))
    lmax = helpers.lmax(Kf, Mf)
    return reference_modes(K, M, 36, dof_mask(mesh, SS), points=mesh.points)[0], lmax


def band_edge(lr, n):
    """(count, f): the first group boundary at or after n modes and a frequency in its gap."""
    i = next(g[-1] + 1 for g in groups(lr) if g[-1] + 1 >= n)
    return i, np.sqrt((lr[i - 1] + lr[i]) / 2) / (2 * np.pi)


def check_reference(mesh, mats, bcs, lam, V, info, tol, n_extra=4):
    """check_lam and per eigenvalue group sin theta <= 10 (r_M + r_M,ref) lam / gap against
    reference_modes (n + n_extra modes), the reference at its rounding floor; returns r_M."""
    K, M = assemble(mesh, mats)
    free = dof_mask(mesh, bcs)
    n, nr = len(lam), info["n_rigid"]
    lr_all, Vr = reference_modes(K, M, nr + n + n_extra, free, points=mesh.points)
    Kf, Mf, idx = restrict(K, M, free)
    lmax = helpers.lmax(Kf, Mf)
    lr, Vrf = lr_all[nr:], Vr[idx, nr:]
    check_lam(lam, lr, lmax, tol)
    lu = spla.splu(Mf.tocsc())
    Vf = np.asarray(V)[idx]
    rm, rr = r_M_sparse(Kf, Mf, lu, lam, Vf), r_M_sparse(Kf, Mf, lu, lr, Vrf)
    assert (rr <= 10 * EPS * lmax / lr).all()
    for g in groups(lr):
        if g[0] >= n:
            break
        gs = g[g < n]  # a group straddling mode n: the returned columns against the cluster
        gap = np.abs(np.delete(lr_all, nr + g)[:, None] - lr[g][None]).min()
        bound = 10 * max((rm[gs] * lam[gs]).max(), (rr[g] * lr[g]).max()) / gap
        assert max_sine_sparse(Mf, Vf[:, gs], Vrf[:, g]) <= bound, (g, bound)
    MV = M @ np.asarray(V)
    assert np.abs(np.asarray(V).T @ MV - np.eye(n)).max() <= 1e-10
    if nr:
        assert np.abs(info["rigid"].T @ MV).max() <= 1e-10
    return rm


@pytest.mark.slow
@pytest.mark.parametrize("name", CASES)
def test_accuracy(name):
    """The roadmap test: 20 modes at tol 1e-10 under "auto" (AMG) against reference_modes."""
    mesh, bcs, mats = CASES[name]()
    lam, V, info = solve_modes(mesh, mats, bcs, 20, tol=1e-10, maxit=3000)
    assert info["converged"] and info["precond"] == "amg" and len(lam) == 20
    check_reference(mesh, mats, bcs, lam, V, info, 1e-10)
    (ch,) = info["chunks"]
    # measured: the free bodies switch (iterations 26, 18), the stack switches at 27 and stalls
    # in the low step until the polish (58 PCG iterations) accepts
    assert (ch["switched"] is not None) == (name in ("free", "cylinder", "stack"))
    assert (ch["polish"] is not None) == (name == "stack")
    if name == "curved":
        assert info["iterations"] <= 35  # measured 28


# ---------------------------------------------------------------- 1: dispatch


def test_resolve():
    flat, web = T.plate(4, 50)[0], T.web(4, 50)[0]
    assert API.resolve_precond(flat, "auto", "float64") == "amg"
    assert API.resolve_precond(web, "auto", "float64") == "amg"
    assert API.resolve_precond(hex20_skin(), "auto", "float64") == "jacobi"


def test_amg_raises_up_front(monkeypatch):
    """precond="amg" with ssh8 + hex20 raises before any build or solve."""
    for name in ("amg", "lobpcg", "unstructured"):
        monkeypatch.setattr(API, name, lambda *a, _n=name, **kw: pytest.fail(_n))
    mesh = hex20_skin()
    msg = "solid-shell AMG supports ssh8 and hex8 blocks"
    with pytest.raises(ValueError, match=msg):
        solve_modes(mesh, STEEL, {}, 2, precond="amg")
    with pytest.raises(ValueError, match=msg):
        solve_modes_memory(mesh, {}, 2, precond="amg")


@pytest.fixture
def lobpcg_spy(monkeypatch):
    calls = []
    real = API.lobpcg

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(API, "lobpcg", spy)
    return calls


@pytest.mark.parametrize("precond", ["auto", "amg", "jacobi", "off"])
def test_polish_wiring(lobpcg_spy, precond):
    """The polish reaches lobpcg with AMG (not with Jacobi, {"on": False} or _precond_sigma's
    1-mode estimate, the first lobpcg call)."""
    mesh = T.plate(8, 50)[0]
    kw = dict(polish_options={"on": False}) if precond == "off" else {}
    _, _, info = solve_modes(mesh, STEEL, SS, 4, precond="amg" if precond == "off" else precond,
                             **kw)  # fmt: skip
    assert info["precond"] == ("jacobi" if precond == "jacobi" else "amg")
    polish = [c.get("polish") for c in lobpcg_spy]
    on = precond in ("auto", "amg")
    assert callable(polish[-1]) == on and all(p is None for p in polish[:-1])
    assert len(polish) == (2 if info["precond"] == "amg" else 1)


@pytest.mark.parametrize(
    ("tol", "opts", "want"),
    [(1e-10, {}, (0.1 * 1e-10, 500)), (1e-8, {"tol": 1e-9, "maxit": 50}, (1e-9, 50))],
)
def test_polish_pcg_options(lobpcg_spy, monkeypatch, tol, opts, want):
    """The polish closure passes polish_options' tol and maxit (default min(1e-7, 0.1 tol), 500)
    to pcg."""
    mesh = T.plate(8, 50)[0]
    solve_modes(mesh, STEEL, SS, 4, tol=tol, polish_options=opts)
    calls, real = [], API.pcg
    monkeypatch.setattr(API, "pcg", lambda *a: calls.append(a[3:]) or real(*a))
    X = jnp.ones((3 * len(mesh.points), 2))
    lobpcg_spy[-1]["polish"](X)
    assert calls == [want]


def test_polish_wiring_other(lobpcg_spy):
    """GMG (voxel) and p-MG never polish (pmg is not a solve_modes option: lobpcg only)."""
    mesh = voxel_mesh((6, 5, 4), (0.1,) * 3)
    solve_modes(mesh, STEEL, {"xmin": "xyz"}, 4)
    assert lobpcg_spy[-1].get("polish") is None


def test_polish_options_errors():
    flat = T.plate(4, 50)[0]
    bad = (({"on": 1}, "on must be a bool"), ({"tol": 0}, "tol must be > 0"),
           ({"maxit": 2.0}, "maxit must be an int"), ({"x": 1}, "unknown polish"))  # fmt: skip
    for opts, msg in bad:
        with pytest.raises(ValueError, match=msg):
            solve_modes(flat, STEEL, SS, 2, polish_options=opts)
        with pytest.raises(ValueError, match=msg):
            solve_modes_memory(flat, SS, 2, polish_options=opts)
    with pytest.raises(ValueError, match=r"precond='amg' \(precond: jacobi\)"):
        solve_modes(flat, STEEL, SS, 2, precond="jacobi", polish_options={"on": True})
    with pytest.raises(ValueError, match=r"\(precond: auto \(resolved to jacobi\)\)"):
        solve_modes(hex20_skin(), STEEL, {}, 2, polish_options={"maxit": 10})
    with pytest.raises(ValueError, match=r"\(precond: auto \(resolved to jacobi\)\)"):
        solve_modes_memory(hex20_skin(), {}, 2, polish_options={"on": False})


# ---------------------------------------------------------------- 3: the switch


@pytest.mark.slow
def test_switch_thin():
    """L/t = 100, tol 1e-10: the chunk switches and converges without a polish."""
    mesh = T.plate(32, 100)[0]
    lam, V, info = solve_modes(mesh, STEEL, SS, 20, tol=1e-10, maxit=3000)
    (ch,) = info["chunks"]
    assert info["converged"] and ch["switched"] is not None and ch["polish"] is None


# ---------------------------------------------------------------- 4, 5: polish, final RR


@pytest.fixture(scope="module")
def thin():
    """The 32^2 plate at L/t = 1000, tol 1e-6: unpolished and polished (pcg calls recorded)."""
    mesh = T.plate(32, 1000)[0]
    off = solve_modes(mesh, STEEL, SS, 20, tol=1e-6, maxit=3000, polish_options={"on": False})
    calls, real = [], API.pcg

    def spy(*a, **kw):
        calls.append(a[3:])
        return real(*a, **kw)

    API.pcg = spy
    try:
        on = solve_modes(mesh, STEEL, SS, 20, tol=1e-6, maxit=3000)
    finally:
        API.pcg = real
    return mesh, off, on, calls


@pytest.mark.slow
def test_polish_thin(thin):
    mesh, (_, _, off), (lam, V, info), calls = thin
    assert not off["converged"] and off["stagnated"]
    (ch,) = info["chunks"]
    assert info["converged"] and info["stagnated"] and 50 <= ch["polish"] <= 400
    assert info["iterations"] == off["iterations"]
    assert calls == [(1e-7, 500)]
    K, M = assemble(mesh, STEEL)
    free = dof_mask(mesh, SS)
    Kf, Mf, idx = restrict(K, M, free)
    Vf = V[idx]
    MV = Mf @ Vf
    r = np.linalg.norm(Kf @ Vf - MV * lam, axis=0) / (lam * np.linalg.norm(MV, axis=0))
    assert np.all(r <= info["thresholds"]), r / info["thresholds"]
    lr = reference_modes(K, M, 24, free, points=mesh.points)[0]
    lmax = helpers.lmax(Kf, Mf)
    check_lam(lam, lr, lmax, 1e-6)


def mf_residuals(op, free, lam, V):
    m = jnp.asarray(free, float)[:, None]
    KV, MV = np.asarray(m * op.apply_K(m * V)), np.asarray(m * op.apply_M(m * V))
    return KV, MV, np.linalg.norm(KV - MV * lam, axis=0) / (lam * np.linalg.norm(MV, axis=0))


@pytest.mark.slow
def test_final_rr_thin(thin):
    """The unpolished STALL exit returns Ritz pairs of the deflated block."""
    mesh, (lam, V, info), *_ = thin
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, SS)
    assert np.all(np.diff(lam) >= 0)
    KV, MV, r = mf_residuals(op, free, lam, jnp.asarray(V))
    rq = np.einsum("ij,ij->j", V, KV) / np.einsum("ij,ij->j", V, MV)
    assert np.abs(lam - rq).max() / lam.min() <= 1e-6  # measured 3.3e-8 (main 6.4e-5)
    G = V.T @ KV
    assert np.abs(G - np.diag(np.diag(G))).max() / np.diag(G).max() <= 1e-7  # measured 1.8e-9
    np.testing.assert_allclose(info["residuals"], r, rtol=1e-5)


@pytest.mark.slow
def test_polish_free_thin():
    mesh = T.plate(24, 1000)[0]
    lam, V, info = solve_modes(mesh, STEEL, {}, 20, tol=1e-6, maxit=3000)
    assert info["converged"] and info["n_rigid"] == 6
    check_reference(mesh, STEEL, {}, lam, V, info, 1e-6)


@pytest.mark.slow
def test_precond_sigma_rr(monkeypatch):
    """_precond_sigma's maxit rounds return _final_rr's Ritz value (free 24^2, L/t = 1000)."""
    mesh = T.plate(24, 1000)[0]
    finals, rounds, final, real = [], [], LOB._final_rr, API.lobpcg

    def spy_final(*a):
        out = final(*a)
        finals.append(out[1])
        return out

    def spy_lobpcg(*a, **kw):
        n = len(finals)
        lam, V, info = real(*a, **kw)
        if a[2] == 1 and not info["converged"] and not info["stagnated"]:  # a maxit round
            rounds.append((lam, finals[n:]))
        return lam, V, info

    monkeypatch.setattr(LOB, "_final_rr", spy_final)
    monkeypatch.setattr(API, "lobpcg", spy_lobpcg)
    solve_modes(mesh, STEEL, {}, 4, tol=1e-6, maxit=3000)
    assert rounds
    for lam, f in rounds:
        assert len(f) == 1 and np.array_equal(lam, f[0][:1])


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["fp64", "fp32", "band"])
def test_polish_chunks(mode):
    """Chunks of 10 on the thin plate (30 modes; band mode: chunk 10): every chunk converges,
    no mode twice, V M-orthonormal (the band clamp: test_lobpcg.py::test_band_clamp)."""
    mesh = T.plate(32, 1000)[0]
    kw = dict(tol=1e-6, maxit=3000, chunk=10)
    lr, lmax = _ref(mesh)
    n = 30
    if mode == "band":
        n, f = band_edge(lr, 30)
        lam, V, info = solve_modes(mesh, STEEL, SS, band=(0, f), strategy="lobpcg", **kw)
    else:
        basis = "float32" if mode == "fp32" else "float64"
        lam, V, info = solve_modes(mesh, STEEL, SS, 30, basis_dtype=basis, **kw)
    assert info["converged"] and len(lam) == n
    assert all(c["polish"] is not None and c["polish"] <= 500 for c in info["chunks"])
    K, M = assemble(mesh, STEEL)
    G = V.T @ (M @ V)
    assert np.abs(G - np.eye(n)).max() <= (1e-6 if mode == "fp32" else 1e-10)
    check_lam(lam, lr, lmax, 1e-6)


# ---------------------------------------------------------------- 7: band mode


@pytest.mark.slow
def test_band(monkeypatch):
    """band=(0, f) on the flat 32^2 plate: every band mode, the count's AMG reuses LOBPCG's
    aggregates."""
    mesh = T.plate(32, 50)[0]
    lr = _ref(mesh)[0]
    n, f = band_edge(lr, 12)
    calls, real = [], API.amg

    def spy(*a, **kw):
        calls.append(("aggregates" in kw, kw.get("return_aggregates", False)))
        return real(*a, **kw)

    monkeypatch.setattr(API, "amg", spy)
    lam, _, info = solve_modes(mesh, STEEL, SS, band=(0, f), tol=1e-8, strategy="lobpcg",
                               verify=True)  # fmt: skip
    assert info["converged"] and info["precond"] == "amg" and len(lam) == n
    assert info["count"]["passed"]
    assert np.abs(lam / lr[:n] - 1).max() <= 1e-7
    fresh = [c for c in calls if not c[0]]
    assert fresh[-1][1] and calls[-1][0] and sum(c[1] for c in calls) == 1


def test_sparse_helpers():
    """r_M_sparse and max_sine_sparse match the dense r_M and max_sine on a small SPD pencil."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 40))
    Kf, Mf = A @ A.T + 40 * np.eye(40), np.diag(rng.uniform(1, 2, 40))
    lam, Q = sl.eigh(Kf, Mf)
    X = Q[:, :4] + 1e-3 * rng.standard_normal((40, 4))
    L = np.linalg.cholesky(Mf)
    lu = spla.splu(sp.csc_matrix(Mf))
    np.testing.assert_allclose(r_M_sparse(Kf, Mf, lu, lam[:4], X), r_M(Kf, Mf, L, lam[:4], X),
                               rtol=1e-10)  # fmt: skip
    Qx = sl.qr(np.linalg.cholesky(Mf).T @ X, mode="economic")[0]
    Qx = sl.solve_triangular(L.T, Qx)  # M-orthonormal
    assert np.isclose(max_sine_sparse(Mf, Qx, Q[:, :4]), max_sine(Mf, L, Qx, Q[:, :4]), rtol=1e-10)


# ---------------------------------------------------------------- 6: memory


def study():
    from benchmarks import shell_solve_study

    return shell_solve_study


@functools.cache
def built(case, n, degree=2):
    """(mesh, op, free, P) of the benchmark's AMG build at sigma 1."""
    return study().build(case, n, degree)


def modelled(mesh, m, pc):
    """lobpcg_memory's three step terms (A b + f ws + the pc term) at block width m."""
    b, ws = 8 * 3 * len(mesh.points) * m, unstructured_memory(mesh, m)["workspace"]
    (A, f), (Al, fl) = STEP_FIT["9-block"], STEP_FIT["low"]
    out = {"9-block": A * b + f * ws + pc["step"], "low": Al * b + fl * ws + pc["step_low"]}
    if "polish" in pc:
        out["polish"] = fl * ws + pc["polish"]
    return out


def measured(case, n, m, degree=2):
    """The benchmark's measure on the cached build."""
    return study().measure(*built(case, n, degree)[1:], m)


GATE = 1.3  # model <= GATE x measured at large n (see test_shell_steps)
STEP_CASES = [(c, n, m) for c, n in [("flat", 64), ("flat", 128), ("web", 64), ("block", 32),
              ("stack10", 64), ("stack3", 128)] for m in (6, 25, 120)]  # fmt: skip
STEP_CASES += [("T", 48, 15), ("flat", 64, 15), ("L", 48, 25)]


def slow_if(cond, *args):
    return pytest.param(*args, marks=pytest.mark.slow) if cond else args


@pytest.mark.parametrize(
    "case, n, m",
    [
        slow_if(c in ("stack3", "block") or n >= 128 or m == 120, c, n, m) for c, n, m in STEP_CASES
    ],  # fmt: skip
)
def test_shell_steps(case, n, m):
    """The shell family's step, low step and polish at the lobpcg_memory level >= the
    measured temporaries; at large n (flat 128^2, stack x3 128^2, m 25 and 120) <= GATE x
    measured (up to 1.28x, the polish on flat 128^2 at m = 25; K0, about 70-130 MB, is most of
    the slack there; a fit of the totals without STEP_FIT's part would be about 16 + 2.5 ws
    blocks above it). Every point but L 48^2 is a fit point: there the lower bound is a
    regression check."""
    mesh = built(case, n)[0]
    model, got = modelled(mesh, m, amg_memory(mesh, m)), measured(case, n, m)
    blk = 8 * 3 * len(mesh.points) * m
    for key in got:
        assert model[key] >= got[key], (key, (model[key] - got[key]) / blk)
        if n == 128 and m >= 25 and key != "low":
            assert model[key] <= GATE * got[key], (key, model[key] / got[key])


@pytest.mark.slow
def test_shell_steps_degree3():
    """Degree 3 on flat 128^2 (on 64^2 its step is 12 MB, 4 blocks, above the model)."""
    mesh = built("flat", 128, 3)[0]
    model = modelled(mesh, 15, amg_memory(mesh, 15, degree=3))
    got = measured("flat", 128, 15, 3)
    assert all(model[k] >= got[k] for k in got), (model, got)


@pytest.mark.slow
@pytest.mark.parametrize("t, n", [("hex20", 14), ("hex8", 24), ("hex8", 40)])
def test_hex_polish_below_step(t, n):
    """Hex AMG has no polish term: its modelled step covers the measured polish."""
    S = study()
    mesh = box_mesh((n,) * 3, (1 / n,) * 3, type=t)
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"})
    P = API.amg(op, free, 1.0, mesh.points)
    for m in (8, 25):
        pc = amg_memory(mesh, m)
        assert "polish" not in pc
        assert modelled(mesh, m, pc)["9-block"] >= S.polish(op, jnp.asarray(free), P, m)


def test_band_probe_polish(monkeypatch):
    """band_probe_block passes polish_options to both of its models."""
    seen, real = [], MEM._models
    monkeypatch.setattr(MEM, "_models", lambda *a: seen.append(a[6]) or real(*a))
    MEM.band_probe_block(
        T.plate(64, 100)[0], SS, 10, 50, 0, n_rigid=0, polish_options={"on": False}
    )
    assert len(seen) == 2 and all(s == {"on": False} for s in seen)


def test_lobpcg_memory_max():
    """step = max(9-block, low, polish) (low_memory: max(low, polish)); no polish term with
    {"on": False}, Jacobi, GMG, hex AMG; on flat 64^2 at m = 15 the polish term wins; on hex8
    AMG at 100k and m = 60 the low term wins."""
    mesh = T.plate(64, 100)[0]
    n = 3 * len(mesh.points)
    om = functools.partial(unstructured_memory, mesh)
    pc = functools.partial(amg_memory, mesh)
    terms = modelled(mesh, 15, pc(15))
    assert lobpcg_memory(om, n, 10, precond_memory=pc)["step"] == max(terms.values())
    assert max(terms.values()) == terms["polish"]
    low = lobpcg_memory(om, n, 10, precond_memory=pc, low_memory=True)["step"]
    assert low == max(terms["low"], terms["polish"])
    off = solve_modes_memory(mesh, SS, 10, polish_options={"on": False})
    on = solve_modes_memory(mesh, SS, 10)
    no = functools.partial(MEM._no_polish, pc)
    assert off == lobpcg_memory(om, n, 10, precond_memory=no) and off["total"] < on["total"]
    assert solve_modes_memory(mesh, SS, 10, precond="jacobi") == lobpcg_memory(om, n, 10)
    box = box_mesh((32, 32, 32), (1 / 32,) * 3)  # hex8, 107k DOF
    nb = 3 * len(box.points)
    hp = amg_memory(box, 60)
    assert "polish" not in hp and "polish" not in gmg_memory(voxel_mesh((8, 8, 8)), 15)
    hterms = modelled(box, 60, hp)
    assert hterms["low"] > hterms["9-block"]
    hstep = lobpcg_memory(functools.partial(unstructured_memory, box), nb, 50,
                          precond_memory=functools.partial(amg_memory, box))["step"]  # fmt: skip
    assert hstep == hterms["low"]


def test_pc_builds_shell():
    """solve_modes_memory on a shell mesh composes as on hex meshes."""
    mesh = T.plate(16, 100)[0]
    n = 3 * len(mesh.points)
    om = functools.partial(unstructured_memory, mesh)
    pc = functools.partial(amg_memory, mesh)
    assert solve_modes_memory(mesh, SS, 6) == lobpcg_memory(om, n, 6, precond_memory=pc)


STORAGE_CASES = [("flat", 64), ("stack3", 64), ("stack10", 32), ("T", 24), ("web", 32),
                 ("block", 16), ("cylinder", 0), ("L", 24), ("flat", 48)]  # fmt: skip


@pytest.mark.parametrize("case, n", STORAGE_CASES)
def test_shell_storage(case, n):
    """amg_memory's storage >= the owned device bytes; the mesh scan's patches equal amg's."""
    mesh, op, _, P = built(case, n)
    assert amg_memory(mesh, 6)["storage"] >= owned_bytes(P, op)
    sizes = MEM._shell_scan(mesh)[0]
    assert np.array_equal(np.sort(sizes), np.sort(3 * (T.columns(op) >= 0).sum(1)))


@pytest.mark.slow
@pytest.mark.parametrize("case, n", [("flat", 128), ("block", 32)])
def test_shell_storage_tight(case, n):
    mesh, op, _, P = built(case, n)
    got = owned_bytes(P, op)
    assert got <= amg_memory(mesh, 6)["storage"] <= 1.3 * got


def test_shell_scan_cached():
    mesh = T.plate(8, 50)[0]
    assert MEM._shell_scan(mesh) is MEM._shell_scan(mesh)


@pytest.mark.parametrize("n", [64, slow_if(True, 128)])
def test_shell_count(n):
    """count_memory's step (without its 2 b blocks) >= the measured recurrence at b = 4, 16, 32
    (within 1.3x at 16 on 128^2); si_memory's minres phase >= the shell MINRES temps at b = 6."""
    mesh = built("flat", n)[0]
    nd = 3 * len(mesh.points)
    om, pc = functools.partial(unstructured_memory, mesh), functools.partial(amg_memory, mesh)
    for r in study().count([("flat", n)], (4, 16, 32)):
        b = r["b"]
        model = count_memory(om, nd, b, precond_memory=pc)["step"] - 2 * 8 * nd * b
        assert model >= r["recurrence"], (b, model / r["recurrence"])
        if n == 128 and b == 16:
            assert model <= 1.3 * r["recurrence"]
        if "minres_b6" in r:
            assert si_memory(om, nd, 60, 0, precond_memory=pc)["minres"] >= r["minres_b6"]


@pytest.mark.slow
def test_benchmark_smoke():
    """shell_solve_study at small sizes: a solve record, the calibration entries and the fit."""
    S = study()
    r = S._solve("flat8", T.plate(8, 50)[0], SS, n_modes=5, ref=True)
    assert r["converged"] and r["lam_rel_err"] < 1e-5 and r["chunks"]
    data = dict(steps=S.steps([("flat", 16)], (6, 8), extra=()),
                count=S.count([("flat", 16)], (4,)), storage=S.storage([("flat", 16)]))  # fmt: skip
    assert {"9-block", "low", "polish", "ws"} <= set(data["steps"][0])
    assert "minres_b6" in data["count"][0] and data["storage"][0]["total"] > 0
    assert set(S.fit(data)) == {"9-block", "low", "polish", "count"}


@pytest.mark.peak
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_shell_build_peak():
    """amg_memory's build >= the AMG setup's host peak (fresh subprocesses, above the built
    operator) on flat 128^2, T 3/3 48^2 and block 32^3; on block 64^3 (837k) <= 1.35x it."""
    for r in study().peaks([("flat", 128), ("T", 48), ("block", 32), ("block", 64)]):
        mesh = study().MESHES[r["case"]](r["size"])
        pc = amg_memory(mesh, 15)
        assert pc["storage"] >= r["owned"] and pc["build"] >= r["peak_bytes"], r
        if r["case"] == "block" and r["size"] == 64:
            assert pc["build"] <= 1.35 * r["peak_bytes"], pc["build"] / r["peak_bytes"]


@pytest.mark.peak
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_shell_791k_peak():
    """Flat 362^2 (791k, L/t 100, out of the build fit): storage >= the owned bytes, the build
    model >= the setup peak and <= 1.3x it; a cold solve_modes (20 modes, tol 1e-6, no switch,
    no polish, synchronous CPU dispatch) peaks within 20% of solve_modes_memory (>= 2 GB,
    ROADMAP.md's rule)."""
    (b,) = study().peaks([("flat", 362)])
    pc = amg_memory(study().MESHES["flat"](362), 15)
    assert pc["storage"] >= b["owned"] and b["peak_bytes"] <= pc["build"] <= 1.3 * b["peak_bytes"]
    r = study().solve_peak()
    assert np.all(r["converged"]) and r["precond"] == "amg"
    est, peak = r["estimate_bytes"], r["peak_bytes"]
    assert est >= 2e9 and abs(est - peak) <= 0.2 * peak, (peak, est)
