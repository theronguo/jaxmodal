"""PR 15a: block shift-invert Lanczos on one slice (docs/plans/pr-15a-shift-invert.md, tests
1-16); PR 15b: si_checked, the slice with its count and restarts
(docs/plans/pr-15b-bandpass-count.md, tests 7, 9-11)."""

import functools
import importlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from jax.tree_util import Partial

from jaxmodal.eigen.counting import band_edges
from jaxmodal.eigen.lobpcg import _norms
from jaxmodal.linalg import count_below, minres
from jaxmodal.materials import Material, isotropic
from jaxmodal.memory import (
    OVERHEAD,
    gmg_memory,
    si_memory,
    stencil_memory,
)
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import (
    a_p,
    ap_operator,
    assemble,
    dof_mask,
    m_orthonormalize,
    rigid_modes,
    stencil,
    unstructured,
)
from jaxmodal.precond import amg, gmg, jacobi

si = importlib.import_module("jaxmodal.eigen.shift_invert")

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
TOL, TAU = 1e-10, 1e-8  # the roadmap's test tolerances
EPS = np.finfo(float).eps

# name: (mesh, bcs, operator, preconditioner kind, number of reference modes)
MESHES = {
    "hex20": (lambda: box_mesh((8, 6, 5), (1 / 8, 1 / 6, 1 / 5), type="hex20"), {"xmin": "xyz"},
              unstructured, "amg", 60),
    "voxel": (lambda: voxel_mesh((10,) * 3, (1 / 10,) * 3), {"zmin": "xyz"}, stencil, "gmg", 60),
    "short": (lambda: box_mesh((6, 5, 4), (1 / 6, 1 / 5, 1 / 4), type="hex20"), {"xmin": "xyz"},
              unstructured, "amg", 60),
    "free": (lambda: box_mesh((4, 3, 3), (1 / 4, 1 / 3, 1 / 3), type="hex8"), {}, unstructured,
             "jacobi", 0),
    "cube": (lambda: voxel_mesh((8,) * 3, (1 / 8,) * 3), {}, stencil, "jacobi", 0),
    "tiny": (lambda: box_mesh((2, 2, 2), (1 / 2,) * 3, type="hex8"), {"xmin": "xyz"},
             unstructured, "jacobi", 0),
}  # fmt: skip


@functools.cache
def prob(name):
    """mesh, free, op, K, M, R (None if clamped), the reference eigenvalues and eigenvectors
    (eigsh at sigma = 0, or dense eigh with the rigid modes removed: n_ref = 0)."""
    make, bcs, kind, _, n_ref = MESHES[name]
    mesh = make()
    free = dof_mask(mesh, bcs)
    op = kind(mesh, MAT)
    K, M = assemble(mesh, MAT)
    f = np.flatnonzero(free)
    Kf, Mf = K[f][:, f], M[f][:, f]
    if n_ref:
        lam, X = spla.eigsh(Kf.tocsc(), n_ref, Mf.tocsc(), sigma=0, which="LM")
    else:
        lam, X = sla.eigh(Kf.toarray(), Mf.toarray())
    i = np.argsort(lam)
    lam, X = lam[i], X[:, i]
    V = np.zeros((op.n_dof, X.shape[1]))
    V[f] = X
    R = None
    if not bcs:
        Rn = jnp.asarray(rigid_modes(mesh))
        R = m_orthonormalize(Rn, op.apply_M(Rn))
        lam, V = lam[6:], V[:, 6:]
    return mesh, free, op, K, M, R, lam, V


def precond(name, sigma):
    mesh, free, op = prob(name)[:3]
    kind = MESHES[name][3]
    if kind == "amg":
        return amg(op, free, sigma, mesh.points, max_coarse=100)
    return gmg(op, free, sigma) if kind == "gmg" else jacobi(op, free, sigma)


def edges(lam, i0, i1):
    """(lo, hi) at the gap midpoints below mode i0 and above mode i1 - 1 (0-based), each moved
    outward to the nearest relative gap >= 1e-2."""
    while 0 < i0 and lam[i0] - lam[i0 - 1] < 1e-2 * lam[i0]:
        i0 -= 1
    while lam[i1] - lam[i1 - 1] < 1e-2 * lam[i1]:
        i1 += 1
    return (lam[i0 - 1] + lam[i0]) / 2 if i0 else 0.0, (lam[i1 - 1] + lam[i1]) / 2


def solve(name, lo, hi, *, sigma=None, capacity=300, **kw):
    s = (lo + hi) / 2 if sigma is None else sigma
    _, free, op, *_, R, _, _ = prob(name)
    kw = dict(tol=TOL, tau=TAU) | kw
    P = precond(name, s)
    return si.si_slice(op, free, lo, hi, P, capacity=capacity, sigma=sigma, R=R, **kw)


def lam_hat(name):
    _, free, op = prob(name)[:3]
    m = jnp.asarray(free, float)[:, None]
    nK, nM = _norms(op, m, m * jnp.ones((op.n_dof, 1)))
    return float(nK) / float(nM)


def check(name, lo, hi, lam, V, info, *, steps=True):
    """Test 1's checks: converged in fewer than half the capacity's steps, the dense count,
    lam against the reference (clusters at relative gaps >= 1e-2 to max(1e-8, 10 eps
    lam_hat / lam), others to 10 r_acc), multiplicities, V M-orthonormal, r_i <= t_i <= r_acc,
    a converged neighbour beyond each edge, and r_acc = max t_i over inside and neighbours."""
    _, free, op, K, M, _, lr, _ = prob(name)
    assert info["converged"] and not info["capacity_full"] and info["reason"] is None
    assert info["edge_unreached"] is None
    if steps:
        assert info["steps"] < 300 // 6 // 2
    ref = lr[(lr >= lo) & (lr <= hi)]
    if MESHES[name][4]:
        assert len(lam) == count_below(K, M, hi, free) - (count_below(K, M, lo, free) if lo else 0)
    assert len(lam) == len(ref)
    gap = np.diff(lr) / lr[1:]
    for x, y in zip(lam, ref, strict=True):
        i = int(np.argmin(np.abs(lr - y)))
        alone = (i == 0 or gap[i - 1] >= 1e-2) and (i == len(gap) or gap[i] >= 1e-2)
        bound = max(1e-8, 10 * EPS * lam_hat(name) / y) if alone else 10 * info["r_acc"]
        assert abs(x - y) <= bound * y, (x, y, bound)
    MV = np.asarray(M @ V)
    assert np.abs(V.T @ MV - np.eye(V.shape[1])).max(initial=0) <= 1e-10
    assert (info["residuals"] <= info["thresholds"]).all()
    assert (info["thresholds"] <= info["r_acc"]).all()
    nb = [info["neighbours"][e] for e in ("lo", "hi") if info["neighbours"][e] is not None]
    assert len(nb) == (1 if lo == 0 else 2)
    f = np.flatnonzero(free)
    for x, v in nb:  # converged: the true residual within r_acc
        Mv = (M @ v)[f]
        assert np.linalg.norm((K @ v)[f] - x * Mv) <= 1.01 * info["r_acc"] * x * np.linalg.norm(Mv)
    t_nb = [10 * info["tau_eff"] * abs(x - info["sigma"]) / x for x, _ in nb]
    # Test 10 (real slice): the neighbours' larger delta sets r_acc above every inside t_i
    assert info["r_acc"] == pytest.approx(max(info["thresholds"].max(initial=0), *t_nb), rel=1e-6)
    return nb


# ---------------------------------------------------------------- 1-6: slices vs references


@pytest.mark.slow
@pytest.mark.parametrize("name", ["hex20", "voxel"])
def test_interior(name):
    """Test 1: an interior slice around modes 31-45 vs eigsh (edges at gaps >= 1e-2); test 10's
    real-slice r_acc check."""
    lr = prob(name)[6]
    gap = np.diff(lr) / lr[1:]
    i0 = max(i for i in range(25, 31) if gap[i - 1] >= 1e-2)
    i1 = min(i for i in range(45, 55) if gap[i - 1] >= 1e-2)
    lo, hi = edges(lr, i0, i1)
    lam, V, info = solve(name, lo, hi)
    nb = check(name, lo, hi, lam, V, info)
    assert info["r_acc"] > info["thresholds"].max()
    assert nb[0][0] < lo and nb[1][0] > hi


def test_zero_edge():
    """Test 2: lo = 0 on a clamped body: no lower neighbour, the 12 lowest modes."""
    lr = prob("short")[6]
    i1 = min(i for i in range(12, 20) if lr[i] - lr[i - 1] > 1e-2 * lr[i])
    lo, hi = edges(lr, 0, i1)
    lam, V, info = solve("short", lo, hi)
    check("short", lo, hi, lam, V, info)
    assert info["neighbours"]["lo"] is None and len(lam) == i1 >= 12


def test_free_body():
    """Test 3: a free body with R deflated: no rigid mode returned, the elastic modes."""
    lr = prob("free")[6]
    lo, hi = edges(lr, 3, 9)
    lam, V, info = solve("free", lo, hi)
    check("free", lo, hi, lam, V, info)
    M, R = prob("free")[4], np.asarray(prob("free")[5])
    assert np.abs(V.T @ (M @ R)).max() <= 1e-10
    lo, hi = edges(lr, 0, 3)  # from 0: undeflated rigid Ritz values would land inside
    lam, V, info = solve("free", lo, hi)
    check("free", lo, hi, lam, V, info)


def clusters(lam, rtol=1e-8):
    groups, g = [], [0]
    for i in range(1, len(lam)):
        if lam[i] - lam[i - 1] < rtol * lam[i]:
            g.append(i)
        else:
            groups.append(g)
            g = [i]
    return groups + [g]


@pytest.mark.parametrize("size", [2, 3])
def test_multiplicity(size):
    """Test 4: the free voxel cube with R deflated: a slice holding a double / triple cluster
    returns it with its exact multiplicity."""
    lr = prob("cube")[6]
    g = next(g for g in clusters(lr) if len(g) == size and g[0] > 0)
    lo, hi = edges(lr, g[0], g[-1] + 1)
    lam, V, info = solve("cube", lo, hi)
    check("cube", lo, hi, lam, V, info)
    assert len(lam) == size and np.ptp(lam) <= 1e-8 * lam[0]


def test_empty(capsys):
    """Test 5: a slice strictly inside a gap: no modes, converged, both neighbours;
    verbose prints one line per step."""
    lr = prob("short")[6]
    i = int(np.argmax(np.diff(lr[:20]) / lr[1:20])) + 1  # the widest gap
    lo, hi = lr[i - 1] + 0.4 * (lr[i] - lr[i - 1]), lr[i - 1] + 0.6 * (lr[i] - lr[i - 1])
    lam, V, info = solve("short", lo, hi, verbose=True)
    check("short", lo, hi, lam, V, info)
    assert len(lam) == 0 and V.shape == (prob("short")[2].n_dof, 0)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == info["steps"] and lines[-1].startswith(f"si_slice: step {info['steps']},")


def test_edge_cluster():
    """Test 6: an edge between the copies of a double eigenvalue returns both."""
    lr = prob("cube")[6]
    g = next(g for g in clusters(lr) if len(g) == 2 and g[0] > 0)
    lo = (lr[g[0] - 1] + lr[g[0]]) / 2
    hi = (lr[g[0]] + lr[g[1]]) / 2  # between the copies (relative gap about 1e-15)
    assert lr[g[0]] < hi < lr[g[1]]
    lam, V, info = solve("cube", lo, hi)
    assert info["converged"] and len(lam) == 2
    assert np.abs(lam - lr[g]).max() <= 1e-8 * lr[g[0]]


# ---------------------------------------------------------------- 7-9: sigma, tau_eff


class Spy:
    """minres wrapper recording (maxit, info) of each call; optional info override."""

    def __init__(self, change=None):
        self.calls, self.change = [], change

    def __call__(self, A, b, precond=None, tol=1e-8, maxit=10000):
        x, info = minres(A, b, precond, tol=tol, maxit=maxit)
        if self.change:
            x, info = self.change(A, b, precond, tol, maxit, x, info)
        self.calls.append((maxit, {k: np.asarray(v) for k, v in info.items()}))
        return x, info


@pytest.mark.slow
def test_sigma_on_eigenvalue(monkeypatch):
    """Test 7: sigma at a reference eigenvalue: MINRES stagnates there (pre-check), sigma moves
    once, the inner cap is based on the moved start, the slice matches; without the move the
    slice stops ("inner solves failed")."""
    name = "hex20"
    _, free, op, *_, lr, _ = prob(name)
    sigma = lr[20]  # mode 21, where the prototype stagnated
    P = precond(name, sigma)
    m = jnp.asarray(free, float)[:, None]
    b = m * op.apply_M(m * jnp.ones((op.n_dof, 6)) * jnp.arange(1.0, 7.0))
    _, inf = minres(ap_operator(op, free, -sigma), b, P, tol=TAU, maxit=20000)
    res = np.asarray(inf["residuals"])
    assert np.asarray(inf["stagnated"]).any() and res.max() > 1e3 * TAU, (
        f"MINRES at sigma = mode 21 did not stagnate ({res.max():.1e}); pick another mode"
    )
    lo, hi = edges(lr, 17, 24)
    spy = Spy()
    monkeypatch.setattr(si, "minres", spy)
    lam, V, info = solve(name, lo, hi, sigma=sigma)
    check(name, lo, hi, lam, V, info)
    assert info["sigma_moved"] and info["sigma"] != sigma
    first = int(spy.calls[1][1]["iterations"].max())  # the repeated start
    assert info["move_iterations"] == int(spy.calls[0][1]["iterations"].max())
    assert spy.calls[0][0] == spy.calls[1][0] == 20000
    assert info["inner_cap"] == spy.calls[2][0] == min(20000, 10 * first)
    monkeypatch.setattr(si, "SIGMA_MOVE", 0.0)
    lam, V, info = solve(name, lo, hi, sigma=sigma)
    assert not info["converged"] and info["reason"] == "inner solves failed"


def test_tau_eff(monkeypatch):
    """Test 8: tau below the floor: tau_eff is the running max of the stagnated columns' final
    residuals, carried into r_acc, and the slice converges. Inner cap: maxit_inner at the first
    step, 10x its iterations after; maxit_inner = 20 caps the columns, sets inner_maxit and
    stops the slice."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 8, 12)
    spy = Spy()
    monkeypatch.setattr(si, "minres", spy)
    lam, V, info = solve("short", lo, hi, tau=1e-16)
    check("short", lo, hi, lam, V, info, steps=False)
    assert not info["sigma_moved"] and info["move_iterations"] is None  # floor < 1e3 TAU_MIN
    bad = [c[1]["residuals"][c[1]["stagnated"] | ~c[1]["converged"]] for c in spy.calls]
    assert info["tau_raised"] and info["tau_eff"] == max(x.max(initial=0) for x in bad) > 1e-16
    its = int(spy.calls[0][1]["iterations"].max())
    assert spy.calls[0][0] == 20000 and {c[0] for c in spy.calls[1:]} == {min(20000, 10 * its)}
    assert info["inner_cap"] == min(20000, 10 * its) and not info["inner_maxit"]
    spy = Spy()
    monkeypatch.setattr(si, "minres", spy)
    lam, V, info = solve("short", lo, hi, maxit_inner=20)
    capped = spy.calls[-1][1]
    assert {c[0] for c in spy.calls} == {20} and not capped["converged"].all()
    assert info["inner_maxit"] and info["tau_raised"]
    assert info["tau_eff"] == capped["residuals"][~capped["converged"]].max() > 1e-4
    assert not info["converged"] and info["reason"] == "inner solves failed"


def test_tau_eff_seam(monkeypatch):
    """Test 9: inner solves stopped at 1e-6 and reported stagnated: tau_eff is their largest
    residual, every t_i = 10 tau_eff delta_i (above tol), some residuals above tol, all within
    their t_i, lam within 10 r_acc of the reference."""

    def loose(A, b, P, tol, maxit, x, info):
        x, info = minres(A, b, P, tol=1e-6, maxit=maxit)
        return x, info | dict(stagnated=jnp.ones_like(info["stagnated"]))

    lr = prob("short")[6]
    lo, hi = edges(lr, 8, 12)
    spy = Spy(loose)
    monkeypatch.setattr(si, "minres", spy)
    lam, V, info = solve("short", lo, hi)
    assert info["converged"] and info["tau_raised"]
    assert info["tau_eff"] == max(c[1]["residuals"].max() for c in spy.calls)
    assert 1e-7 < info["tau_eff"] <= 1e-6
    delta = np.abs(lam - info["sigma"]) / lam
    assert info["thresholds"] == pytest.approx(10 * info["tau_eff"] * delta, rel=1e-12)
    assert info["r_acc"] >= 10 * info["tau_eff"] * info["delta_max"] > TOL
    r = info["residuals"]
    assert (r > TOL).any() and (r <= info["thresholds"]).all()
    ref = lr[(lr >= lo) & (lr <= hi)]
    assert len(lam) == len(ref) and (np.abs(lam - ref) <= 10 * info["r_acc"] * ref).all()


# ---------------------------------------------------------------- 10: sets and stop (synthetic)


def test_slice_sets():
    """Test 10: chains across both edges (one past hi (1 + 100 r)), the neighbours beyond them
    with gaps >= 100 r, lo = 0, a missing neighbour, an empty slice."""
    r = 1e-6
    mu = np.array([1.0, 1.9, 1.9997, 1.99985, 3.0, 4.0003, 4.0006, 5.0])
    inside, below, above = si._slice_sets(mu, 2.0, 4.0, r)
    # 1.99985 and 4.0003 are inside only through the 100 r margins; 1.9997 and 4.0006 (past
    # hi (1 + 100 r)) only through chains
    assert list(inside) == [2, 3, 4, 5, 6] and below == 1 and above == 7
    mu = np.array([1.0, 1.9, 2.0, 2.0001, 3.0, 4.0, 4.0001, 4.0002, 5.0])
    inside, below, above = si._slice_sets(mu, 0.0, 2.5, r)
    assert list(inside) == [0, 1, 2, 3] and below is None and above == 4
    inside, below, above = si._slice_sets(mu, 4.5, 6.0, r)
    assert list(inside) == [8] and below == 7 and above is None
    inside, below, above = si._slice_sets(np.array([-1e-14, 1.0, 2.0]), 0.0, 1.5, r)
    assert list(inside) == [1] and below is None and above == 2  # no lower neighbour at 0
    inside, below, above = si._slice_sets(mu, 3.2, 3.8, r)
    assert len(inside) == 0 and below == 4 and above == 5


def fake(r_of, t_of, asked):
    def resid(idx):
        asked.append(list(idx))
        return np.array([r_of[i] for i in idx]), np.array([t_of[i] for i in idx])

    return resid


def test_slice_check():
    """Test 10: _slice_check's done rule on synthetic Ritz values and a fake resid."""
    mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = {i: 1e-8 for i in range(5)}
    ok = {i: 1e-9 for i in range(6)}
    asked = []
    out = si._slice_check(mu, 1.5, 3.5, 1e-8, fake(ok, t, asked))
    done, inside, below, above, r_acc, res, _ = out
    assert sorted(res) == [0, 1, 2, 3] and res[3] == (1e-9, 1e-8)
    assert done and list(inside) == [1, 2] and (below, above) == (0, 3) and r_acc == 1e-8
    assert sorted(asked[0]) == [0, 1, 2, 3]
    for i, args in ((1, (1.5, 3.5)), (3, (1.5, 3.5))):  # unconverged inside pair / neighbour
        r = ok | {i: 1e-7}
        assert not si._slice_check(mu, *args, 1e-8, fake(r, t, []))[0]
    assert not si._slice_check(mu, 4.5, 6.0, 1e-8, fake(ok, t, []))[0]  # no upper neighbour
    done, _, below, *_ = si._slice_check(mu, 0.0, 2.5, 1e-8, fake(ok, t, []))
    assert done and below is None  # lo = 0
    done, inside, *_ = si._slice_check(mu, 2.2, 2.8, 1e-8, fake(ok, t, []))
    assert done and len(inside) == 0  # empty slice
    # a neighbour's large t raises r above the gap 3.0 -> 3.1: it merges into inside in the
    # same call, resid asked only for the new index
    mu = np.array([1.0, 2.0, 3.0, 3.1, 5.0])
    t = {0: 1e-8, 1: 1e-8, 2: 1e-8, 3: 1e-3, 4: 1e-8}
    asked = []
    done, inside, below, above, r_acc, *_ = si._slice_check(mu, 1.5, 3.05, 1e-8, fake(ok, t, asked))
    assert done and list(inside) == [1, 2, 3] and above == 4 and r_acc == 1e-3
    assert asked == [[1, 2, 0, 3], [4]]
    # a chain of re-forms: each merge brings a larger threshold until the set stops growing;
    # at r = 1e-3 the window ends at 3.30055 and 3.32 joins only through the chain from 3.3
    mu = np.array([1.0, 2.0, 3.0, 3.001, 3.05, 3.3, 3.32, 10.0])
    t = {i: 1e-8 for i in range(8)} | {3: 1e-4, 4: 1e-3}
    ok = {i: 1e-9 for i in range(8)}
    done, inside, _, above, r_acc, *_ = si._slice_check(mu, 1.5, 3.0005, 1e-8, fake(ok, t, []))
    assert list(inside) == [1, 2, 3, 4, 5, 6] and above == 7 and r_acc == 1e-3 and done


# ---------------------------------------------------------------- 11-14


def test_capacity_full():
    """Test 11: a capacity too small for the slice: converged False, capacity_full, the
    converged pairs returned match; hi above the spectrum: edge_unreached "hi"; lo below
    lambda_1 / 4: edge_unreached "lo"."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 8, 20)
    for cap, n_conv in ((48, 2), (72, 11)):  # 12 modes; capacity rounds up to the chunk 24
        lam, V, info = solve("short", lo, hi, capacity=cap)
        assert not info["converged"] and info["capacity_full"]
        assert info["reason"] == "capacity full" and info["basis"] == cap
        assert len(lam) == n_conv and info["edge_unreached"] == "both"
        assert info["neighbours"] == {"lo": None, "hi": None}
        for x in lam:
            assert np.abs(lr - x).min() <= 10 * info["r_acc"] * x
    lt = prob("tiny")[6]
    lo, hi = (lt[-2] + lt[-1]) / 2, lt[-1] * 2
    lam, V, info = solve("tiny", lo, hi, sigma=lt[-1] * 1.02, capacity=48)
    assert info["capacity_full"] and info["edge_unreached"] == "hi"
    assert len(lam) == 1 and abs(lam[0] - lt[-1]) <= 1e-8 * lt[-1]
    lam, V, info = solve("short", lr[0] / 4, (lr[1] + lr[2]) / 2, capacity=60)
    assert info["capacity_full"] and info["edge_unreached"] == "lo"
    assert info["neighbours"]["lo"] is None and info["neighbours"]["hi"] is not None
    # a capacity above the free space (54 DOFs, 9 blocks): the basis stops at the full space,
    # where the Rayleigh-Ritz is exact
    lo, hi = (lt[-5] + lt[-4]) / 2, lt[-1] * 2
    lam, V, info = solve("tiny", lo, hi, capacity=72)
    assert info["capacity_full"] and info["basis"] == 54 and info["edge_unreached"] == "hi"
    assert np.abs(lam - lt[-4:]).max() <= 1e-8 * lt[-1]


def test_failed_solves(monkeypatch):
    """A non-finite MINRES residual stops the slice ("inner solves failed"); a final residual
    pass above the thresholds makes it unconverged."""

    def nan(A, b, P, tol, maxit, x, info):
        return x, info | dict(residuals=info["residuals"].at[0].set(jnp.nan))

    lr = prob("short")[6]
    lo, hi = edges(lr, 8, 12)
    monkeypatch.setattr(si, "minres", Spy(nan))
    lam, V, info = solve("short", lo, hi)
    assert not info["converged"] and info["reason"] == "inner solves failed" and len(lam) == 0
    monkeypatch.setattr(si, "minres", minres)
    real = si._v_chunk

    def big(*a):  # the final residual pass reports residuals 1e3x larger
        r, t = real(*a)
        return 1e3 * r, t

    monkeypatch.setattr(si, "_v_chunk", big)
    lam, V, info = solve("short", lo, hi)
    assert not info["converged"] and info["reason"] is None


def test_deflation():
    """Test 12: half of a short slice as W: the other half returned, W never returned."""
    lr, Vr = prob("short")[6:]
    lo, hi = edges(lr, 8, 18)
    idx = np.flatnonzero((lr >= lo) & (lr <= hi))
    W = Vr[:, idx[::2]]
    lam, V, info = solve("short", lo, hi, W=W)
    M = prob("short")[4]
    assert info["converged"] and len(lam) == len(idx[1::2])
    assert np.abs(W.T @ (M @ V)).max() <= 1e-10
    assert np.abs(lam - lr[idx[1::2]]).max() <= 1e-8 * lr[idx[-1]]
    # W over two chunks (block 2: c = 8; modes 0-6 + half the slice; 7 stays the neighbour)
    W = Vr[:, np.r_[np.arange(7), idx[::2]]]
    lam, V, info = solve("short", lo, hi, W=W, block=2)
    assert info["converged"] and len(lam) == len(idx[1::2])
    assert np.abs(W.T @ (M @ V)).max() <= 1e-10
    assert np.abs(lam - lr[idx[1::2]]).max() <= 1e-8 * lr[idx[-1]]


def test_refill(monkeypatch):
    """Test 13: a start block with two identical columns: the SVQB drop is refilled."""
    real, calls = si._random_block, []

    def dup(key, n, b):
        X = real(key, n, b)
        calls.append(b)
        return X.at[:, 1].set(X[:, 0]) if len(calls) == 1 else X

    monkeypatch.setattr(si, "_random_block", dup)
    lr = prob("short")[6]
    lo, hi = edges(lr, 8, 12)
    lam, V, info = solve("short", lo, hi)
    check("short", lo, hi, lam, V, info)
    assert len(calls) >= 2  # the refill drew new columns


def test_errors():
    """Test 14: invalid arguments raise."""
    _, free, op, *_, lr, Vr = prob("short")
    P = jacobi(op, free, 1.0)
    run = functools.partial(si.si_slice, op, free, precond=P, capacity=60)
    for lo, hi in ((-1.0, 1.0), (2.0, 2.0)):
        with pytest.raises(ValueError, match="0 <= lo < hi"):
            run(lo, hi)
    with pytest.raises(ValueError, match="fp64"):
        si.si_slice(op, free, 1.0, 2.0, gmg_like(), capacity=60)
    with pytest.raises(ValueError, match="block"):
        run(1.0, 2.0, block=0)
    with pytest.raises(ValueError, match="capacity"):
        run(1.0, 2.0, capacity=6)
    tf, top = prob("tiny")[1:3]
    with pytest.raises(ValueError, match="free space"):  # 54 free DOFs
        si.si_slice(top, tf, 1.0, 2.0, jacobi(top, tf, 1.0), capacity=72, block=30)
    with pytest.raises(ValueError, match="W must have shape"):
        run(1.0, 2.0, W=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="R must have shape"):
        run(1.0, 2.0, R=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="M-orthonormal"):
        run(1.0, 2.0, W=2 * Vr[:, :3])


def gmg_like():
    class P:
        dtype = "float32"

    return P()


# ---------------------------------------------------------------- 15: memory


def test_memory_formulas():
    """Test 15: storage = operator + preconditioner + basis (chunk-rounded) + R and M R; the
    host term; the end phase; total = overhead + storage + host + max(builds, phases)."""
    mesh = voxel_mesh((16,) * 3, (1 / 16,) * 3)
    n = 3 * len(mesh.points)
    est = functools.partial(stencil_memory, mesh)
    pcm = functools.partial(gmg_memory, mesh)
    e = si_memory(est, n, 50, 30, precond_memory=pcm, n_rigid=6, n_deflate=10)
    cap = 72  # 50 rounded up to the chunk 24
    v = 8 * n
    storage = est(6)["storage"] + pcm(6)["storage"] + v * (24 + cap) + 2 * v * 6
    assert e["storage"] == storage and e["host"] == v * 10 + 8 * cap**2
    assert e["end"] == v * (30 + 2 + 24)
    phases = max(e[k] for k in ("minres", "reortho", "ritz", "end"))
    assert e["ritz"] < e["minres"] and e["reortho"] < e["minres"]
    total = OVERHEAD + storage + e["host"] + max(est(6)["build"], pcm(6)["build"], phases)
    assert e["total"] == total


PEAK_SETUP = """
import numpy as np
from jaxmodal import config
config.enable_x64()
from jaxmodal.eigen.shift_invert import si_slice
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import voxel_mesh
from jaxmodal.operators import dof_mask, stencil
from jaxmodal.precond import gmg
mats = {0: Material(isotropic(3.0, 0.3), 2.0)}
def run(s, lo, hi, cap):
    mesh = voxel_mesh((s, s, s), (1 / s,) * 3)
    free = dof_mask(mesh, {"xmin": "xyz"})
    op = stencil(mesh, mats, mass="lumped")
    P = gmg(op, free, (lo + hi) / 2)
    return si_slice(op, free, lo, hi, P, capacity=cap, tol=1e-6)
run(8, LO8, HI8, CAP)  # warm-up
"""


def lam_voxel(s, k):
    """The lowest k eigenvalues of the clamped unit voxel cube (lumped mass) at s^3 from the
    8^3 mesh (the same continuum: low modes converge in h)."""
    mesh = voxel_mesh((8,) * 3, (1 / 8,) * 3)
    free = dof_mask(mesh, {"xmin": "xyz"})
    K, M = assemble(mesh, MAT, mass="lumped")
    f = np.flatnonzero(free)
    return np.sort(spla.eigsh(K[f][:, f].tocsc(), k, M[f][:, f].tocsc(), sigma=0)[0])


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
@pytest.mark.parametrize("case", ["minres", "basis"])
def test_peak(case):
    """Test 15 (a), (b): the measured process peak of a slice within 20% of si_memory (>= 2
    GB): (a) MINRES-dominated (capacity 4 blocks, near mode 5; the MINRES phase >= 1.25x the
    GMG build), (b) basis-dominated (a large capacity, a low slice of few steps)."""
    from benchmarks import harness

    lr = lam_voxel(8, 8)
    s, cap = (80, 24) if case == "minres" else (48, 744)
    lo, hi = edges(lr, 4, 6) if case == "minres" else edges(lr, 0, 3)
    mesh = voxel_mesh((s,) * 3, (1 / s,) * 3)
    n = 3 * len(mesh.points)
    e = si_memory(
        functools.partial(stencil_memory, mesh, mass="lumped"), n, cap, 4,
        precond_memory=functools.partial(gmg_memory, mesh),
    )  # fmt: skip
    print({k: f"{v / 1e9:.2f}" for k, v in e.items()})
    if case == "minres":
        assert e["minres"] >= max(e["ritz"], e["end"], 1.25 * gmg_memory(mesh, 6)["build"])
    else:
        assert 8 * n * cap >= 0.6 * e["total"]
    setup = PEAK_SETUP.replace("LO8", repr(lo)).replace("HI8", repr(hi)).replace("CAP", str(cap))
    peak = harness.measure(f"run({s}, {lo!r}, {hi!r}, {cap})", setup)["peak_bytes"]
    print(f"si {case} {s}^3 cap {cap}: estimate {e['total'] / 1e9:.2f} GB, peak {peak / 1e9:.2f}")
    assert e["total"] >= 2e9 and abs(e["total"] - peak) <= 0.2 * peak


FORM_V = """
import numpy as np
from jaxmodal import config
config.enable_x64()
import jax, jax.numpy as jnp
from jaxmodal.eigen.shift_invert import _form_v
n, cap, k = N, CAP, 150
Y = np.random.default_rng(0).standard_normal((cap, k))
_form_v(jnp.ones((1000, cap)), 0, Y, 24)  # warm-up
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_form_v_peak():
    """Test 15 (c): the end phase alone: _form_v on a random basis (>= 2 GB) with 150 selected
    columns, within 20% of basis + host V + one chunk."""
    from benchmarks import harness

    n, cap, k = 800_000, 240, 150
    zero = lambda m: dict(storage=0, workspace=0, build=0, overhead=OVERHEAD)  # noqa: E731
    e = si_memory(zero, n, cap, k - 2)  # end: host V of k columns and one chunk
    model = OVERHEAD + e["storage"] + e["end"]
    setup = FORM_V.replace("N", str(n)).replace("CAP", str(cap))
    # values do not matter for memory; jax.random.normal alone peaks at 2.5x the basis
    code = "Q = jnp.ones((n, cap)); V = _form_v(Q, 0, Y, 24)"
    peak = harness.measure(code, setup)["peak_bytes"]
    print(f"_form_v: model {model / 1e9:.2f} GB, peak {peak / 1e9:.2f}")
    assert model >= 2e9 and abs(model - peak) <= 0.2 * peak


@pytest.mark.slow
def test_benchmark_smoke():
    """Test 16: shift_invert_study.run(small=True)."""
    from benchmarks import shift_invert_study

    out = shift_invert_study.run(small=True)
    assert out["grid"]["hex20"]["rows"] and out["slice"]["converged"]


# ---------------------------------------------------------------- PR 15b: si_checked

CB = (13, 19)  # modes 13-18 of "short": edge gaps 0.09 below and 0.16 above
MID = 15  # an interior mode, not next to an edge


def _dense_apply(Ainv, r):
    return Ainv @ r


def exact(name, shift):
    """A_P^-1 at the shift as a pytree callable (the count's PCG in 1-2 iterations)."""
    _, free, op, K, M = prob(name)[:5]
    s = float(ap_operator(op, free, shift).s)
    return Partial(_dense_apply, jnp.asarray(np.linalg.inv(a_p(K + shift * M, free, s).toarray())))


def cb_edges():
    lr = prob("short")[6]
    return (lr[CB[0] - 1] + lr[CB[0]]) / 2, (lr[CB[1] - 1] + lr[CB[1]]) / 2


def checked(lo, hi, c_hi, **kw):
    """si_checked on "short" with AMG for the slice and the exact count preconditioner."""
    _, free, op = prob("short")[:3]
    kw = dict(tol=TOL, tau=TAU, capacity=300) | kw
    return si.si_checked(op, free, lo, hi, precond("short", (lo + hi) / 2),
                         count_precond=exact("short", c_hi), **kw)  # fmt: skip


def _drop(out, x):
    lam, V, info = out
    keep = np.abs(lam - x) > 1e-6 * x
    info = dict(info, residuals=info["residuals"][keep], thresholds=info["thresholds"][keep])
    return lam[keep], V[:, keep], info


def spy_slice(monkeypatch, real, rule):
    """si_slice replaced by real(...) post-processed by rule(call number, result)."""
    calls = []

    def fake(*a, **kw):
        calls.append(kw)
        return rule(len(calls), real(*a, **kw))

    monkeypatch.setattr(si, "si_slice", fake)
    return calls


def ref_slice(op, free, lo, hi, P, **kw):
    """A converged si_slice result from the reference pairs (no Lanczos)."""
    lr, Vr = prob("short")[6:]
    i = np.flatnonzero((lr >= lo) & (lr <= hi))
    nb = {"lo": (lr[i[0] - 1], Vr[:, i[0] - 1]), "hi": (lr[i[-1] + 1], Vr[:, i[-1] + 1])}
    z = np.zeros(len(i))
    info = dict(converged=True, reason=None, neighbours=nb, residuals=z, thresholds=z + TOL,
                r_acc=TOL, tau_eff=TAU)  # fmt: skip
    return lr[i], Vr[:, i], info


def check_cb(lam, info):
    """All six modes of CB against the reference, multiplicities, V M-orthonormal (15a)."""
    lr = prob("short")[6]
    assert len(lam) == CB[1] - CB[0]
    assert np.abs(lam - lr[CB[0] : CB[1]]).max() <= 1e-8 * lr[CB[1]]
    assert (info["residuals"] <= info["thresholds"]).all()


def test_checked_empty():
    """15b test 7 (roadmap, the slice part): a band in the middle third of the largest gap
    (modes 2 and 3) returns 0 modes and a count < 0.5."""
    lr = prob("short")[6]
    lo, hi = lr[2] + (lr[3] - lr[2]) / 3, lr[2] + 2 * (lr[3] - lr[2]) / 3
    lam, V, info = checked(lo, hi, (hi + lr[3]) / 2)
    c = info["count"]
    assert info["converged"] and len(lam) == 0 and V.shape[1] == 0
    assert c["ran"] and c["passed"] and c["estimate"] < 0.5 and c["restarts"] == 0
    assert c["edges"][1] == pytest.approx((hi + lr[3]) / 2, rel=1e-10)


def test_restart(monkeypatch, capsys):
    """15b test 9 (roadmap): (a) one in-band mode removed after the first convergence: count
    >= 0.5, flagged, recovered on the restart (W = the found modes, capacity minus the extra
    deflation columns, a new seed), final count < 0.5, all modes against the reference; (b) the
    mode withheld on every run: 3 runs, 2 restarts, every count >= 0.5, flagged, not passed."""
    lr, Vr = prob("short")[6:]
    M = prob("short")[4]
    lo, hi = cb_edges()
    c_hi = band_edges(lr[CB[0] : CB[1]], lo, hi, lr[[CB[0] - 1, CB[1]]])[1]
    real = si.si_slice
    calls = spy_slice(monkeypatch, real, lambda j, out: _drop(out, lr[MID]) if j == 1 else out)
    seeds, real_bc = [], si.band_count  # the recounts use fresh probes: count seed + j
    monkeypatch.setattr(
        si, "band_count", lambda *a, **kw: seeds.append(kw["seed"]) or real_bc(*a, **kw)
    )
    lam, V, info = checked(lo, hi, c_hi, verbose=True)
    c = info["count"]
    assert len(calls) == 2 and c["restarts"] == 1 and c["flagged"] and c["passed"]
    assert c["estimates"][0] >= 0.5 and c["estimate"] < 0.5 and c["reason"] is None
    # 300 rounds up to 13 chunks of 24, minus one for the 5 found modes
    assert calls[1]["W"].shape[1] == 5 and calls[1]["capacity"] == 312 - 24
    assert [k["seed"] for k in calls] == [0, 1] and seeds == [0, 1]
    check_cb(lam, info)
    assert np.abs(V.T @ (M @ V) - np.eye(6)).max() <= 1e-10
    K = prob("short")[3]  # the merged columns match their eigenvalues
    assert np.abs(np.einsum("ij,ij->j", V, K @ V) - lam).max() <= 1e-10 * lam.max()
    out = capsys.readouterr().out
    assert "si_checked: count" in out and out.count("si_checked: count") == 2
    assert "restart 1" in out and out.rstrip().endswith("pass")
    assert out.count("count: block") == 2  # verbose reaches the count (one block per count)

    monkeypatch.setattr(si, "si_slice", real)
    calls = spy_slice(monkeypatch, real, lambda j, out: _drop(out, lr[MID]))
    seeds.clear()
    lam, V, info = checked(lo, hi, c_hi)
    c = info["count"]
    assert len(calls) == 3 and c["restarts"] == 2 and c["flagged"] and not c["passed"]
    assert all(e >= 0.5 for e in c["estimates"]) and len(c["estimates"]) == 3
    assert c["reason"] == "count >= 0.5 after 2 restarts" and len(lam) == 5
    assert [k["seed"] for k in calls] == [0, 1, 2] and seeds == [0, 1, 2]
    assert [k["W"].shape[1] for k in calls[1:]] == [5, 5]


def test_checked_edges():
    """15b test 10: lo = 0 uses the step count (lam_c midway between the last mode and the
    neighbour); a W mode beyond hi closes that side (15c) and places the upper edge."""
    lr, Vr = prob("short")[6:]
    hi = (lr[3] + lr[4]) / 2
    lam, V, info = checked(0.0, hi, (lr[3] + lr[4]) / 2)
    c = info["count"]
    assert len(lam) == 4 and c["passed"] and "edges" not in c
    assert c["lam_c"] == pytest.approx((lr[3] + lr[4]) / 2, rel=1e-10)
    lo, hi = cb_edges()
    lam, V, info = checked(lo, hi, (lr[18] + lr[19]) / 2, W=Vr[:, [19]], lam_W=lr[[19]])
    c = info["count"]
    check_cb(lam, info)
    assert info["neighbours"]["hi"] is None  # 15c: W mode 19 closes the upper side
    assert c["passed"] and c["edges"][1] == pytest.approx((lr[18] + lr[19]) / 2, rel=1e-10)
    # a W mode inside a band from 0 (the step count) counts as found
    hi0 = (lr[3] + lr[4]) / 2
    lam, V, info = checked(0.0, hi0, hi0, W=Vr[:, [1]], lam_W=lr[[1]])
    c = info["count"]
    assert len(lam) == 3 and c["passed"] and c["restarts"] == 0
    # a W mode inside the band (an adjacent slice's) counts as found and is not returned
    lam, V, info = checked(lo, hi, (lr[18] + lr[19]) / 2, W=Vr[:, [13]], lam_W=lr[[13]])
    c = info["count"]
    assert len(lam) == 5 and np.abs(lam - lr[14:19]).max() <= 1e-8 * lr[19]
    assert c["passed"] and c["restarts"] == 0 and c["edges"][0] == pytest.approx(
        (lr[12] + lr[13]) / 2, rel=1e-10)  # fmt: skip


def test_checked_w_margin(monkeypatch):
    """15b: a caller's W mode below lo but above a returned mode (si_slice returns modes within
    its edge margin) counts as found: W = mode 14, lo just above it, the slice returns 13 and
    15-18 (fake); the count passes without a restart."""
    lr, Vr = prob("short")[6:]
    _, free, op = prob("short")[:3]
    lo, hi = lr[14] * (1 + 1e-9), cb_edges()[1]
    c_hi = (lr[18] + lr[19]) / 2

    def fake(*a, **kw):
        i = [13, 15, 16, 17, 18]
        z = np.zeros(5)
        nb = {"lo": (lr[12], Vr[:, 12]), "hi": (lr[19], Vr[:, 19])}
        return lr[i], Vr[:, i], dict(converged=True, neighbours=nb, residuals=z,
                                     thresholds=z + TOL, r_acc=TOL, tau_eff=TAU)  # fmt: skip

    monkeypatch.setattr(si, "si_slice", fake)
    lam, V, info = si.si_checked(op, free, lo, hi, None, capacity=300, W=Vr[:, [14]],
                                 lam_W=lr[[14]], count_precond=exact("short", c_hi))  # fmt: skip
    c = info["count"]
    assert len(lam) == 5 and c["passed"] and c["restarts"] == 0


def test_restart_capacity(monkeypatch):
    """15b: the restart capacity is L0(capacity) - (L0(nW + k) - L0(nW)), L0 rounding up to
    the chunk c = 24: with 10 caller W modes and 5 found, L0 stays 24 and the capacity 312 (a
    rounding of k alone would give 288)."""
    lr, Vr = prob("short")[6:]
    _, free, op = prob("short")[:3]
    lo, hi = cb_edges()
    c_hi = (lr[18] + lr[19]) / 2

    def rule(j, out):
        lam, V, info = out
        keep = np.abs(lam - lr[MID]) > 1e-6 * lr[MID]
        keep = keep if j == 1 else ~keep  # the first run misses MID, the restart finds it
        return lam[keep], V[:, keep], dict(info, residuals=info["residuals"][keep],
                                           thresholds=info["thresholds"][keep])  # fmt: skip

    calls = spy_slice(monkeypatch, ref_slice, rule)
    w = np.arange(20, 30)
    lam, V, info = si.si_checked(op, free, lo, hi, None, capacity=300, W=Vr[:, w],
                                 lam_W=lr[w], count_precond=exact("short", c_hi))  # fmt: skip
    c = info["count"]
    assert [k["capacity"] for k in calls] == [300, 312] and calls[1]["W"].shape[1] == 15
    assert c["restarts"] == 1 and c["flagged"] and c["passed"] and len(lam) == 6


def test_checked_flags(monkeypatch, capsys):
    """15b test 11: not restarted: an unverifiable count, an unconverged slice, a restart without
    capacity; an unconverged restart returns the merged modes; W needs lam_W."""
    lr, Vr = prob("short")[6:]
    _, free, op = prob("short")[:3]
    lo, hi = cb_edges()
    c_hi = (lr[18] + lr[19]) / 2
    run = functools.partial(
        si.si_checked, op, free, lo, hi, None, count_precond=exact("short", c_hi)
    )

    def ident(j, out):
        return out

    spy_slice(monkeypatch, ref_slice, ident)
    monkeypatch.setattr(si, "band_count", lambda *a, **kw: (np.nan, dict(
        verifiable=False, passed=False, degree=10**6, steps=0, bias=np.nan, capped=True,
        time=dict(total=0.0))))  # fmt: skip
    lam, V, info = run(capacity=300)
    c = info["count"]
    assert c["ran"] and not c["passed"] and not c["flagged"] and c["restarts"] == 0
    assert c["reason"].startswith("degree above MAX_DEGREE") and len(lam) == 6
    monkeypatch.undo()

    def unconverged(j, out):
        lam, V, info = out
        return lam[:2], V[:, :2], dict(info, converged=False, residuals=info["residuals"][:2],
                                       thresholds=info["thresholds"][:2])  # fmt: skip

    spy_slice(monkeypatch, ref_slice, unconverged)
    lam, V, info = run(capacity=300)
    c = info["count"]
    assert not info["converged"] and not c["ran"] and c["reason"] == "slice not converged"
    assert not c["passed"] and not c["verifiable"]
    assert c["restarts"] == 0 and len(lam) == 2 and np.isnan(c["estimate"])
    monkeypatch.undo()

    calls = spy_slice(monkeypatch, ref_slice, lambda j, out: _drop(out, lr[MID]))
    lam, V, info = run(capacity=24, verbose=True)  # the restart would get 24 - 24 < 12 columns
    c = info["count"]
    assert len(calls) == 1 and c["reason"] == "restart capacity" and c["flagged"]
    assert c["restarts"] == 0 and c["estimate"] >= 0.5 and len(lam) == 5
    assert capsys.readouterr().out.count("si_checked: count") == 1
    monkeypatch.undo()

    def restart_fails(j, out):
        return _drop(out, lr[MID]) if j == 1 else unconverged(j, _drop(out, lr[MID]))

    calls = spy_slice(monkeypatch, ref_slice, restart_fails)
    lam, V, info = run(capacity=300)
    c = info["count"]
    assert len(calls) == 2 and c["restarts"] == 1 and c["flagged"] and not info["converged"]
    assert c["reason"] == "slice not converged" and len(lam) == 7  # 5 found + 2 (fake) more
    monkeypatch.undo()

    with pytest.raises(ValueError, match="W and lam_W"):
        run(capacity=300, W=Vr[:, [19]])
    with pytest.raises(ValueError, match="W must have shape"):
        run(capacity=300, W=Vr[:, [19]], lam_W=lr[[19, 20]])
    with pytest.raises(ValueError, match="capacity must be"):  # si_slice's, not "restart capacity"
        run(capacity=6)


# ---------------------------------------------------------------- PR 15c: si_band, fakes

OK_COUNT = dict(ran=True, passed=True, verifiable=True, flagged=False, estimate=0.1,
                estimates=[0.1], reason=None, restarts=0)  # fmt: skip
NO_COUNT = dict(ran=False, passed=False, verifiable=False, flagged=False, estimate=np.nan,
                estimates=[], reason="slice not converged", restarts=0)  # fmt: skip


def rz(conv, unconv=(), name="short"):
    """si_slice's info["ritz"]: the reference values of the mode indices conv (converged) and
    unconv (not), merged and ascending."""
    lr = prob(name)[6]
    v = np.r_[lr[list(conv)], lr[list(unconv)]]
    f = np.r_[np.ones(len(conv), bool), np.zeros(len(unconv), bool)]
    i = np.argsort(v, kind="stable")
    return v[i], f[i]


def new_modes(a, e, kw, name="short"):
    """The reference modes in [a, e] not in the caller's lam_W (a converged slice's pairs)."""
    lr = prob(name)[6]
    known = np.zeros(0) if kw.get("lam_W") is None else kw["lam_W"]
    return [j for j in np.flatnonzero((lr >= a) & (lr <= e))
            if not np.isclose(known, lr[j], rtol=1e-12, atol=0).any()]  # fmt: skip


def fake_band(monkeypatch, script, name="short", count=(0.1, True, False)):
    """si_checked replaced by script(call, a, e, kw) -> (pairs, reason, ritz, count info);
    pairs: mode indices or (lam, V); the neighbours are the reference modes beyond a and e.
    _count_slice returns (estimate, verifiable, capped) = count (or count(call)). Returns the
    recorded si_checked kwargs (plus interval, P) and _count_slice calls (args, kwargs)."""
    lr, Vr = prob(name)[6:]
    calls, counts = [], []

    def fake(op, free, a, e, P, **kw):
        calls.append(dict(kw, interval=(a, e), P=P))
        pairs, reason, ritz, cnt = script(len(calls), a, e, kw)
        lam, V = (lr[pairs], Vr[:, pairs]) if isinstance(pairs, list) else pairs
        below, above = np.flatnonzero(lr < a), np.flatnonzero(lr > e)
        nb = {"lo": (lr[below[-1]], Vr[:, below[-1]]) if len(below) else None,
              "hi": (lr[above[0]], Vr[:, above[0]]) if len(above) else None}  # fmt: skip
        z = np.zeros(len(lam))
        info = dict(converged=reason is None, reason=reason, sigma=(a + e) / 2, residuals=z,
                    thresholds=z + TOL, r_acc=TOL, tau_eff=TAU, neighbours=nb, ritz=ritz,
                    runs=[{"basis": 0, "ritz": ritz}],
                    count=cnt or (OK_COUNT if reason is None else NO_COUNT))  # fmt: skip
        return lam, V, info

    def count_slice(*a, **kw):
        counts.append((a, kw))
        est, ver, capped = count(len(counts)) if callable(count) else count
        return est, dict(verifiable=ver, passed=ver and est < 0.5, capped=capped, degree=7)

    monkeypatch.setattr(si, "si_checked", fake)
    monkeypatch.setattr(si, "_count_slice", count_slice)
    return calls, counts


def FULL_RZ():  # noqa: N802
    """A full slice's Ritz values: modes 7-11 converged, 2-6 and 12-16 not."""
    return rz(range(7, 12), [2, 3, 4, 5, 6, 12, 13, 14, 15, 16])


def fband(name, lo, hi, capacity=96, **kw):
    _, free, op = prob(name)[:3]
    return si.si_band(op, free, lo, hi, lambda s: ("P", s), capacity=capacity, tol=TOL, **kw)


def test_split_rules():
    """15c test 3, the span rule (_split) on chosen values: an unconverged value between two
    converged ones below sigma stops the run (no mode claimed past it); an unconverged value
    nearest sigma grows the run from the converged value nearest it; static halves with the
    cut at the largest gap within each half, not the run's overall largest; k = 2: the gap is
    the lower side's; a side whose edge neighbour converged (or is a W mode) is complete; no
    cut on an edge; a = 0 complete when the run reaches the bottom; nothing converged."""
    T, F = True, False
    v = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 9.0])
    assert si._split(v, np.array([T, T, F, T, T, T, T, F]), 4.5, 1.5, 7.5, 1e-10) == (
        3.5, 5.5, False, False)  # fmt: skip
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 4.0 unconverged and nearest sigma
    assert si._split(v, np.array([T, T, T, F, T, T]), 4.0, 1.5, 5.5, 1e-10) == (
        None, None, True, False)  # fmt: skip
    v = np.array([1.0, 2.0, 3.0, 3.1, 3.2, 6.0, 6.1, 9.0])  # lower half: 2|3, overall: 3.2|6
    s_lo, s_hi, d_lo, d_hi = si._split(v, np.array([F, T, T, T, T, T, T, F]), 4.0, 1.5, 8.0,
                                       1e-10)  # fmt: skip
    assert (s_lo, s_hi, d_lo, d_hi) == (2.5, 4.6, False, False)
    v = np.array([1.0, 2.0, 3.0, 4.0])  # the W neighbour 4.0 converged: upper side complete
    assert si._split(v, np.array([F, T, T, T]), 2.5, 1.5, 3.5, 1e-10) == (2.5, None, False, True)
    v = np.array([1.0, 2.0, 3.0])  # gaps 1|2 and 2|3 touch the edges' neighbours: no cuts
    assert si._split(v, np.array([T, T, T]), 2.0, 1.5, 2.5, 1e-10) == (None, None, True, True)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert si._split(v, np.array([T, T, F, F]), 1.0, 0.0, 3.5, 1e-10) == (None, None, True, False)
    assert si._split(v, np.zeros(4, bool), 1.0, 0.5, 3.5, 1e-10) is None
    # a chain (relative gap < 100 r) is taken whole: one unconverged copy stops the run there
    v = np.array([1.0, 2.0, 3.0, 3.0 + 1e-9, 4.0, 5.0])
    assert si._split(v, np.array([F, F, T, F, T, T]), 4.5, 1.5, 4.8, 1e-10) == (
        None, None, False, True)  # fmt: skip


def test_band_split(monkeypatch):
    """15c test 3, a full slice through si_band (fakes): its span [s_lo, s_hi] is a leaf
    counted with the parent's preconditioner and the stored pairs inside, the remainders run
    lowest first with W = the stored modes of their window plus the known neighbours, within
    the capacity; the result holds every mode once; the aggregate count includes the span."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 3, 16)

    def script(j, a, e, kw):
        if j == 1:
            return [7, 8, 9, 10, 11], "capacity full", FULL_RZ(), None
        return new_modes(a, e, kw), None, None, None

    calls, counts = fake_band(monkeypatch, script)
    lam, V, info = fband("short", lo, hi, capacity=96)
    s_lo, s_hi, *_ = si._split(*FULL_RZ(), (lo + hi) / 2, lo, hi,
                               TOL)  # fmt: skip
    assert lr[7] < s_lo < lr[9] and lr[9] < s_hi < lr[11]
    assert [c["interval"] for c in calls] == [(lo, hi), (lo, s_lo), (s_hi, hi)]
    assert [(s["kind"], s["leaf"]) for s in info["slices"]] == [
        ("run", False), ("span", True), ("run", True), ("run", True)]  # fmt: skip
    assert info["slices"][1]["interval"] == (s_lo, s_hi) and info["slices"][1]["parent"] == 0
    assert np.array_equal(lam, lr[3:16]) and info["converged"] and info["duplicates"] == 0
    ((args, kw),) = counts
    assert kw["precond"] == ("P", (lo + hi) / 2)  # the parent's, before it is released
    inb = (lr[7:12] >= s_lo) & (lr[7:12] <= s_hi)
    assert np.array_equal(args[3], lr[7:12][inb]) and args[4:6] == (s_lo, s_hi)
    assert args[6][0] == lr[7:12][~inb & (lr[7:12] < s_lo)].max()  # the out-of-band pairs
    assert args[7][0] == lr[7:12][~inb & (lr[7:12] > s_hi)].min()
    for c in calls[1:]:  # W: every stored mode near the window and the known neighbours
        a, e = c["interval"]
        near = lr[7:12][(lr[7:12] >= a * (1 - 1e-2)) & (lr[7:12] <= e * (1 + 1e-2))]
        assert set(near) <= set(c["lam_W"])
        assert c["capacity"] + -(-len(c["lam_W"]) // 24) * 24 <= 96
    assert c["lam_W"].min() < s_hi  # the known neighbour below the upper remainder
    assert info["count"]["passed"] and len(info["count"]["estimates"]) == 3


def test_band_rules(monkeypatch):
    """15c test 3: one side complete and the other without a cut reruns [a, b] with W; a rerun
    of the same interval without a new pair or a newly complete side raises; nothing converged
    and neither side done or cut raise; max_slices; the capacity error names both causes."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 3, 16)
    first = ([3], "capacity full", rz([2, 3], [4, 5]), None)

    def rerun(j, a, e, kw):
        return first if j == 1 else (new_modes(a, e, kw), None, None, None)

    calls, _ = fake_band(monkeypatch, rerun)
    lam, V, info = fband("short", lo, hi)
    assert [c["interval"] for c in calls] == [(lo, hi), (lo, hi)]
    assert lr[3] in calls[1]["lam_W"] and np.array_equal(lam, lr[3:16])
    assert [s["kind"] for s in info["slices"]] == ["run", "run"]  # no span: no upper cut

    def stuck(j, a, e, kw):
        return first if j == 1 else ([], "capacity full", rz([2, 3], [4, 5]), None)

    fake_band(monkeypatch, stuck)
    with pytest.raises(ValueError, match=r"\(no progress\).*hi above the spectrum"):
        fband("short", lo, hi)
    fake_band(monkeypatch, lambda *_: ([], "capacity full", rz([], [2, 3, 4]), None))
    with pytest.raises(ValueError, match=r"nothing converged.*below lambda_1"):
        fband("short", lo, hi)
    fake_band(monkeypatch, lambda *_: ([9], "capacity full", rz([9], [2, 8, 10, 16]), None))
    with pytest.raises(ValueError, match="neither side done or cut"):
        fband("short", lo, hi)

    def full(j, a, e, kw):
        return [7, 8, 9, 10, 11], "capacity full", FULL_RZ(), None

    fake_band(monkeypatch, full)
    with pytest.raises(RuntimeError, match="max_slices = 1"):
        fband("short", lo, hi, max_slices=1)


def test_band_leaves(monkeypatch, capsys):
    """15c tests 3 and 8: a span leaf whose count flags is rerun as a run with W = its pairs;
    an inner failure stops the band with the pairs found; a flagged and an unverifiable run
    leaf are not split and fail the band's count with the interval in the reason; verbose
    prints one line per slice and span."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 3, 16)

    def script(j, a, e, kw):
        if j == 1:
            return [7, 8, 9, 10, 11], "capacity full", FULL_RZ(), None
        return new_modes(a, e, kw), None, None, None

    calls, counts = fake_band(monkeypatch, script, count=lambda k: (0.9 if k == 1 else 0.1,
                                                                    True, False))  # fmt: skip
    lam, V, info = fband("short", lo, hi, verbose=True)
    span = info["slices"][1]
    assert span["kind"] == "span" and not span["leaf"] and span["count"]["flagged"]
    x0, x1 = span["interval"]
    assert [c["interval"] for c in calls] == [(lo, hi), (lo, x0), (x0, x1), (x1, hi)]
    assert set(lr[7:12][(lr[7:12] >= x0) & (lr[7:12] <= x1)]) <= set(calls[2]["lam_W"])
    assert np.array_equal(lam, lr[3:16]) and info["count"]["passed"]  # the replaced span's flag
    out = capsys.readouterr().out
    assert out.count("si_band: slice") == 4 and out.count("si_band: span") == 1

    flagged = dict(OK_COUNT, passed=False, flagged=True, estimate=0.8, estimates=[0.8, 0.9, 0.7],
                   reason="count >= 0.5 after 2 restarts")  # fmt: skip
    capped = dict(OK_COUNT, passed=False, verifiable=False, estimate=np.nan, estimates=[np.nan],
                  reason="degree above MAX_DEGREE: the gap is too small to count")  # fmt: skip
    for cnt, flag in ((flagged, True), (capped, False)):
        calls, _ = fake_band(monkeypatch, lambda j, a, e, kw, c=cnt: (new_modes(a, e, kw), None,
                                                                      None, c))  # fmt: skip
        lam, V, info = fband("short", lo, hi)
        c = info["count"]
        assert len(calls) == 1 and np.array_equal(lam, lr[3:16]) and info["converged"]
        assert not c["passed"] and c["flagged"] == flag and c["verifiable"] == flag
        assert c["reason"] == f"[{lo:.6g}, {hi:.6g}]: {cnt['reason']}"

    def fails(j, a, e, kw):
        if j == 1:
            return [3, 4, 5, 6, 7], None, None, None
        return [8], "inner solves failed", None, NO_COUNT

    calls, _ = fake_band(monkeypatch, fails)
    s = (lr[7] + lr[8]) / 2
    lam, V, info = fband("short", lo, hi, edges=[s])
    assert len(calls) == 2 and np.array_equal(lam, lr[3:9]) and not info["converged"]
    assert info["reason"] == f"[{s:.6g}, {hi:.6g}]: inner solves failed"
    assert not info["count"]["passed"]


def test_band_options(monkeypatch):
    """15c test 10: seeds seed + (RESTARTS + 1) i for slice i (start block and count), verify
    and the slice options passed through, R passed; argument errors."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 3, 16)
    s = [(lr[7] + lr[8]) / 2, (lr[10] + lr[11]) / 2]
    calls, _ = fake_band(monkeypatch, lambda j, a, e, kw: (new_modes(a, e, kw), None, None, None))
    lam, V, info = fband("short", lo, hi, edges=s, seed=5, count_options={"seed": 7},
                         verify=False, tau=1e-7, maxit_inner=99)  # fmt: skip
    step = si.RESTARTS + 1
    assert [c["seed"] for c in calls] == [5, 5 + step, 5 + 2 * step]
    assert [c["count_options"]["seed"] for c in calls] == [7, 7 + step, 7 + 2 * step]
    assert all(not c["verify"] and c["tau"] == 1e-7 and c["maxit_inner"] == 99 for c in calls)
    assert [c["capacity"] for c in calls] == [96, 72, 72]  # W: 1 chunk (the known neighbours)
    assert np.array_equal(lam, lr[3:16]) and info["merged_edges"] == [] and not info["collapsed"]
    for bad, match in (
        (dict(lo=hi, hi=lo), "need 0 <= lo < hi"),
        (dict(edges=[s[1], s[0]]), "edges must be sorted"),
        (dict(edges=[hi + 1]), "edges must be sorted"),
        (dict(capacity=11), "capacity must be an int >= 2 block"),
        (dict(verify=None), "verify must be True or False"),
        (dict(block=0), "block must be"),
    ):
        kw = dict(lo=lo, hi=hi) | bad
        with pytest.raises(ValueError, match=match):
            fband("short", kw.pop("lo"), kw.pop("hi"), **kw)
    # capacity_j below 2 block after deflation: the known neighbour takes the only chunk
    calls, _ = fake_band(monkeypatch, lambda j, a, e, kw: (new_modes(a, e, kw), None, None, None))
    with pytest.raises(ValueError, match="deflated modes leave 0 columns"):
        fband("short", lo, hi, capacity=24, edges=s)
    assert len(calls) == 1


def test_merge(monkeypatch):
    """15c tests 5b and 9 (fakes on the voxel cube's double eigenvalue v1, v2 and its
    neighbours): the same mode from two runs is kept once (duplicates 1, the boundary merged,
    collapsed); a 45-degree rotated double from the second run with one copy from the first
    gives both copies, exact pairs; a near copy cos(t) v1 + sin(t) v2 (t = 1e-3) of a kept pair
    is dropped, nothing lost; a genuinely new vector is kept; V M-orthonormal after each; a
    perturbed vector fails the re-check after widening (edge_rr_failed); two boundaries of which
    one merges: not collapsed."""
    _, free, op, K, M, R, lr, Vr = prob("cube")
    g = next(g for g in clusters(lr) if len(g) == 2 and g[0] > 3)
    i, j = g
    lo, hi = edges(lr, i - 1, j + 2)
    s = (lr[i] + lr[j]) / 2  # the interior edge between the copies
    v1, v2 = Vr[:, i], Vr[:, j]

    def two(first, second):
        def script(k, a, e, kw):
            lam, V = first if k == 1 else second
            return (np.asarray(lam), np.stack(V, 1)), None, None, None

        return script

    def check(lam, V, want):
        assert np.abs(lam - want).max() <= 1e-10 * want.max()
        assert np.abs(V.T @ (M @ V) - np.eye(V.shape[1])).max() <= 1e-10

    run = functools.partial(fband, "cube", lo, hi, edges=[s], R=R)
    lo_pairs = ([lr[i - 1], lr[i]], [Vr[:, i - 1], v1])
    cases = [  # second run, kept count, duplicates
        (([lr[i], lr[j + 1]], [v1, Vr[:, j + 1]]), 1),
        (([lr[i], lr[j], lr[j + 1]], [(v1 + v2) / 2**0.5, (v1 - v2) / 2**0.5, Vr[:, j + 1]]), 1),
        (([lr[j], lr[j + 1]], [v2, Vr[:, j + 1]]), 0),
    ]
    for second, dup in cases:
        fake_band(monkeypatch, two(lo_pairs, second), name="cube")
        lam, V, info = run()
        want = lr[i - 1 : j + 2] if len(second[0]) > 2 or dup == 0 else lr[[i - 1, i, j + 1]]
        check(lam, V, want)
        assert info["duplicates"] == dup and info["merged_edges"] == [s] and info["collapsed"]
        assert not info["edge_rr_failed"]
    t = 1e-3
    near = ([lr[i]], [np.cos(t) * v1 + np.sin(t) * v2])
    fake_band(monkeypatch, two(([lr[i - 1], lr[i], lr[j]], [Vr[:, i - 1], v1, v2]), near),
              name="cube")  # fmt: skip
    lam, V, info = run()
    check(lam, V, lr[i - 1 : j + 1])
    assert info["duplicates"] == 1
    rng = np.random.default_rng(0)
    w = v2 + 1e-3 * np.where(free, rng.standard_normal(len(v2)), 0)
    w /= np.sqrt(w @ (M @ w))
    fake_band(monkeypatch, two(lo_pairs, ([lr[j]], [w])), name="cube")
    lam, V, info = run()
    assert info["edge_rr_failed"] and info["duplicates"] == 0 and len(lam) == 3
    s2 = (lr[j + 1] + lr[j + 2]) / 2 if j + 2 < len(lr) else None
    third = ([lr[j + 2]], [Vr[:, j + 2]])

    def three(k, a, e, kw):
        lam, V = (lo_pairs, ([lr[i], lr[j], lr[j + 1]], [v1, v2, Vr[:, j + 1]]), third)[k - 1]
        return (np.asarray(lam), np.stack(V, 1)), None, None, None

    fake_band(monkeypatch, three, name="cube")
    lam, V, info = fband("cube", lo, max(hi, lr[j + 2] * 1.001), edges=[s, s2], R=R)
    check(lam, V, lr[i - 1 : j + 3])
    assert info["merged_edges"] == [s] and not info["collapsed"]


def test_merge_chunks():
    """15c test 5b: _merge streams K and M in chunks of c columns; a window wider than c gives
    the dense Rayleigh-Ritz (and the same result for c = 1 and c = 64)."""
    _, free, op, K, M, R, lr, Vr = prob("cube")
    m = jnp.asarray(free, jnp.float64)[:, None]
    nK, nM = map(float, _norms(op, m, m * jnp.ones((op.n_dof, 1))))
    consts = jnp.array([nK, nM, 1e-10, si.C_FLOOR])
    rng = np.random.default_rng(1)
    Q1 = Vr[:, :5] @ np.linalg.qr(rng.standard_normal((5, 5)))[0]  # rotated, M-orthonormal
    Q2 = Vr[:, 3:9] @ np.linalg.qr(rng.standard_normal((6, 6)))[0]  # overlaps modes 3, 4
    for c in (1, 64):
        th, X, r, t, drop = si._merge(op, m, [Q1, Q2], consts, c)
        assert drop == 2 and len(th) == 9
        assert np.abs(th - lr[:9]).max() <= 1e-10 * lr[8]
        assert np.abs(X.T @ (M @ X) - np.eye(9)).max() <= 1e-10 and (r <= 1e-10).all()


# ---------------------------------------------------------------- PR 15c: si_band, real runs


def rband(name, lo, hi, *, capacity, **kw):
    """si_band with the exact (K + sigma M)^-1 as every slice's preconditioner (MINRES and
    the counts in a few iterations)."""
    _, free, op, *_, R, _, _ = prob(name)
    kw = dict(tol=TOL, tau=TAU) | kw
    return si.si_band(op, free, lo, hi, functools.partial(exact, name), capacity=capacity, R=R,
                      **kw)  # fmt: skip


def check_band(name, lo, hi, lam, V, info, *, counted=True):
    """The roadmap's checks: the modes in [lo, hi] as the reference (count, lam to max(1e-8,
    10 eps lam_hat / lam) for clusters at relative gaps >= 1e-2, else 10 r_acc, so
    multiplicities), every returned pair a reference eigenvalue (the leaves' margins), V
    M-orthonormal (no duplicates), residuals within thresholds, every counted leaf < 0.5."""
    M, lr = prob(name)[4], prob(name)[6]
    ref = lr[(lr >= lo) & (lr <= hi)]
    assert info["converged"] and info["reason"] is None
    assert ((lam >= lo) & (lam <= hi)).sum() == len(ref)
    gap = np.diff(lr) / lr[1:]
    for x in lam:
        i = int(np.argmin(np.abs(lr - x)))
        alone = (i == 0 or gap[i - 1] >= 1e-2) and (i == len(gap) or gap[i] >= 1e-2)
        bound = max(1e-8, 10 * EPS * lam_hat(name) / x) if alone else 10 * info["r_acc"]
        assert abs(x - lr[i]) <= bound * lr[i], (x, lr[i], bound)
    assert np.abs(V.T @ np.asarray(M @ V) - np.eye(V.shape[1])).max(initial=0) <= 1e-10
    assert (info["residuals"] <= info["thresholds"]).all()
    if counted:
        c = info["count"]
        assert c["passed"] and len(c["estimates"]) == sum(s["leaf"] for s in info["slices"])
        assert all(e < 0.5 for e in c["estimates"])


def test_band_one_slice():
    """15c test 1: a large capacity: one leaf, no split, the same pairs as si_checked (same
    seed and slice preconditioner), count passed."""
    lo, hi = cb_edges()
    lam, V, info = rband("short", lo, hi, capacity=300)
    check_band("short", lo, hi, lam, V, info)
    assert [(s["kind"], s["leaf"]) for s in info["slices"]] == [("run", True)]
    _, free, op = prob("short")[:3]
    lam2, V2, _ = si.si_checked(op, free, lo, hi, exact("short", (lo + hi) / 2), capacity=300,
                                tol=TOL, tau=TAU)  # fmt: skip
    assert np.array_equal(lam, lam2) and np.array_equal(V, V2)
    assert not info["merged_edges"] and not info["collapsed"] and info["duplicates"] == 0


SPLITS = {  # name: (first mode, last mode + 1, capacity, block, {seed: runs})
    "block6": (5, 29, 96, 6, {0: 3}),
    "block2": (8, 22, 48, 2, {0: 3}),
}


@pytest.mark.parametrize("case", ["block6", pytest.param("block2", marks=pytest.mark.slow)])
def test_band_capacity_split(case, monkeypatch):
    """15c test 2 (roadmap: a forced small capacity splits the slice and matches eigsh): at
    least one full slice replaced by a span and remainders, every leaf converged and counted
    < 0.5, the reference matched, the recorded run count per seed; a spy checks every slice's
    basis within the first's and the remainders' W (stored modes of the window and the known
    neighbours)."""
    i0, i1, cap, block, runs = SPLITS[case]
    lr = prob("short")[6]
    lo, hi = edges(lr, i0, i1)
    c = si._chunk(block)
    up = lambda x: -(-x // c) * c  # noqa: E731
    real, calls = si.si_checked, []

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(si, "si_checked", spy)
    for seed, n_runs in runs.items():
        calls.clear()
        lam, V, info = rband("short", lo, hi, capacity=cap, block=block, seed=seed)
        check_band("short", lo, hi, lam, V, info)
        kinds = [s["kind"] for s in info["slices"]]
        assert kinds.count("run") == n_runs and "span" in kinds
        assert any(s["kind"] == "run" and not s["leaf"] for s in info["slices"])
        for kw in calls:
            n_w = 0 if kw["lam_W"] is None else len(kw["lam_W"])
            assert up(kw["capacity"]) + up(n_w) <= up(cap)
        stored = np.sort(lam)
        for kw, s in zip(calls[1:], [s for s in info["slices"] if s["kind"] == "run"][1:],
                         strict=True):  # fmt: skip
            a, e = s["interval"]
            inside = stored[(stored >= a * (1 - 1e-2)) & (stored <= e * (1 + 1e-2))]
            found = s["n_new"]
            assert len(set(kw["lam_W"])) >= len(inside) - found


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", SPLITS)
def test_band_split_sweep(case, seed):
    """15c test 2, robustness: at the chosen capacity and one chunk below and above it, and with
    a second seed, the band matches the reference (splits allowed) or raises the capacity
    error; never a short or wrong result."""
    i0, i1, cap, block, _ = SPLITS[case]
    lr = prob("short")[6]
    lo, hi = edges(lr, i0, i1)
    c = si._chunk(block)
    for capacity in (cap - c, cap, cap + c):
        try:
            lam, V, info = rband("short", lo, hi, capacity=capacity, block=block, seed=seed)
        except ValueError as e:
            assert "fills capacity" in str(e)
            continue
        check_band("short", lo, hi, lam, V, info)


def test_band_capacity_raises():
    """15c test 2c: modes 8-21 at block 6, capacity 72: the capacity error."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 8, 22)
    with pytest.raises(ValueError, match=r"fills capacity 72 .*raise capacity"):
        rband("short", lo, hi, capacity=72)


def test_minres_plateau_slice():
    """15c test 2d: plain si_slice on [32.120, 39.101], capacity 72, seed 0 (AMG): at step 2 one
    column plateaued above ||b|| for 90 iterations and stopped the slice with "inner solves
    failed" before the MINRES fix; now no inner solve fails and tau_eff stays tau."""
    lam, V, info = solve("short", 32.120, 39.101, capacity=72, seed=0)
    assert info["reason"] in (None, "capacity full") and info["tau_eff"] == TAU
    assert not info["tau_raised"]


@pytest.mark.slow
def test_band_many_slices():
    """15c test 2e: modes 5-40 at block 6, capacity 96 (6 runs; before the MINRES fix it stopped
    with "inner solves failed") matches the reference."""
    lr = prob("short")[6]
    lo, hi = edges(lr, 5, 41)
    lam, V, info = rband("short", lo, hi, capacity=96)
    check_band("short", lo, hi, lam, V, info)
    assert sum(s["kind"] == "run" for s in info["slices"]) == 6


def test_slice_w_neighbours():
    """15c test 3, real: the 15b slice of modes 13-18 with W = modes 11-12 and 19-20 closes both
    sides on the W modes (no Ritz neighbours, edge_unreached None), the same pairs as without W
    to 1e-10, in no more steps; with W = mode 20 only, the unknown mode 19 between the edge and
    it is still converged as the Ritz neighbour."""
    lr, Vr = prob("short")[6:]
    lo, hi = cb_edges()
    lam0, _, info0 = solve("short", lo, hi)
    w = [11, 12, 19, 20]
    lam, V, info = solve("short", lo, hi, W=Vr[:, w], lam_W=lr[w])
    assert info["converged"] and info["edge_unreached"] is None
    assert info["neighbours"] == {"lo": None, "hi": None}
    assert np.abs(lam - lam0).max() <= 1e-10 * lam0.max() and info["steps"] <= info0["steps"]
    v, conv = info["ritz"]
    assert conv[np.isin(v, lr[w])].all()
    nb = si._beyond(lam.min(), lam.max(), info["neighbours"], Vr[:, w], lr[w])
    assert (nb[0][0], nb[1][0]) == (lr[12], lr[19])
    lam, V, info = solve("short", lo, hi, W=Vr[:, [20]], lam_W=lr[[20]])
    assert info["converged"] and info["neighbours"]["hi"][0] == pytest.approx(lr[19], rel=1e-8)


def test_band_voxel_cube():
    """15c test 4 (roadmap): the free voxel cube with R, a band over two triples, a double, a
    single and a triple, split by an interior edge exactly on the second triple, tol 1e-6: the
    triple is held whole by the lower slice and deflated by the upper (no merge, no
    duplicates), count and multiplicities as the reference, each leaf's count < 0.5."""
    lr = prob("cube")[6]
    lo, hi = (lr[1] + lr[2]) / 2, (lr[13] + lr[14]) / 2
    lam, V, info = rband("cube", lo, hi, capacity=300, edges=[lr[6]], tol=1e-6)
    check_band("cube", lo, hi, lam, V, info)
    runs = [s for s in info["slices"] if s["kind"] == "run"]
    assert len(runs) == 2 and all(s["leaf"] for s in runs)
    assert not info["collapsed"] and info["merged_edges"] == [] and info["duplicates"] == 0
    g = [g for g in clusters(lam, 1e-6) if len(g) > 1]
    assert [len(x) for x in g] == [
        len(x) for x in clusters(lr[(lr >= lo) & (lr <= hi)]) if len(x) > 1
    ]


def test_band_collapse(monkeypatch):
    """15c test 5a (roadmap): modes 23-24 of "short" (relative gap 4.6e-4) with the interior
    edge s at their midpoint; slice 2 [s, 50] with inner solves stopped at 1e-6 (tau_eff 1e-6,
    100 r_acc > the gap): the window across s holds mode 23 (run 1) and mode 24 (run 2) and is
    merged by Rayleigh-Ritz, the only boundary merged (collapsed), the result matches the
    reference within 10 r_acc and is M-orthonormal. Slice 2 deflates mode 23 as the known
    neighbour below its window, so it is not found twice (no duplicate; the plan expected one
    before revision 4 added the neighbours to the windows; duplicates: test_merge)."""
    lr = prob("short")[6]
    lo, s, hi = (lr[21] + lr[22]) / 2, (lr[23] + lr[24]) / 2, 50.0

    def loose(A, b, P, tol, maxit, x, info):
        x, info = minres(A, b, P, tol=1e-6, maxit=maxit)
        return x, info | dict(stagnated=jnp.ones_like(info["stagnated"]))

    real, n = si.si_checked, []

    def second_loose(*a, **kw):
        n.append(kw["lam_W"])
        if len(n) == 2:
            monkeypatch.setattr(si, "minres", Spy(loose))
        try:
            return real(*a, **kw)
        finally:
            monkeypatch.setattr(si, "minres", minres)

    monkeypatch.setattr(si, "si_checked", second_loose)
    lam, V, info = rband("short", lo, hi, capacity=300, edges=[s], verify=False)
    r2 = info["slices"][1]
    assert 1e-7 < r2["tau_eff"] <= 1e-6 and 100 * r2["r_acc"] > (lr[24] - lr[23]) / lr[24]
    assert np.isclose(n[1], lr[23], rtol=1e-10).any() and info["duplicates"] == 0
    assert info["merged_edges"] == [s] and info["collapsed"]
    check_band("short", lo, hi, lam, V, info, counted=False)
    assert not info["edge_rr_failed"]


def test_band_from_zero():
    """15c test 6 (roadmap): the clamped band [0, hi] over modes 0-5 matches the reference; the
    count is the step count (no band edges)."""
    lr = prob("short")[6]
    hi = (lr[5] + lr[6]) / 2
    lam, V, info = rband("short", 0.0, hi, capacity=300)
    check_band("short", 0.0, hi, lam, V, info)
    assert "edges" not in info["slices"][0]["count"]


def test_band_edge_on_double():
    """15c test 7 (roadmap): the voxel cube with hi exactly on a double eigenvalue and lo
    further away (the upper edge governs): both copies returned, the count's gap is the upper
    edge gap and 19 <= d gap <= 24 (count_degree's rule, roadmap d ~ 25/g)."""
    lr = prob("cube")[6]
    g = next(g for g in clusters(lr) if len(g) == 2 and g[0] > 3)
    lo, hi = (lr[g[0] - 1] + lr[g[0]]) / 2, lr[g[0]]
    lam, V, info = rband("cube", lo, hi, capacity=300)
    assert info["converged"] and np.abs(lam - lr[g]).max() <= 1e-8 * hi and len(lam) == 2
    c = info["slices"][0]["count"]
    assert c["passed"] and c["gap"] == c["gaps"][1] and 19 <= c["degree"] * c["gap"] <= 24


def test_band_empty():
    """15c test 8 (roadmap, end to end): a band in the middle third of the largest gap: 0 modes,
    the count < 0.5."""
    lr = prob("short")[6]
    lo, hi = lr[2] + (lr[3] - lr[2]) / 3, lr[2] + 2 * (lr[3] - lr[2]) / 3
    lam, V, info = rband("short", lo, hi, capacity=300)
    assert len(lam) == 0 and info["converged"] and info["count"]["estimate"] < 0.5
    assert info["count"]["passed"]


def test_band_verify_false():
    """15c test 10: verify=False runs no count and records the degree and gap the count would
    use, equal to a counted run's."""
    lo, hi = cb_edges()
    _, _, on = rband("short", lo, hi, capacity=300)
    _, _, off = rband("short", lo, hi, capacity=300, verify=False)
    a, b = on["slices"][0]["count"], off["slices"][0]["count"]
    assert not b["ran"] and b["reason"] == "not requested" and not off["count"]["ran"]
    assert (b["degree"], b["gap"]) == (a["degree"], a["gap"]) and a["ran"]


def test_band_above_spectrum():
    """15c test 11: a tiny clamped mesh (54 free DOF), hi above its largest eigenvalue: the
    capacity error naming "hi above the spectrum" (path taken: the first slice converges
    nothing at capacity 48)."""
    _, free, op, K, M = prob("tiny")[:5]
    f = np.flatnonzero(free)
    top = sla.eigh(K[f][:, f].toarray(), M[f][:, f].toarray(), eigvals_only=True)
    with pytest.raises(ValueError, match=r"nothing converged.*hi above the spectrum"):
        si.si_band(op, free, top[-3], 2 * top[-1], functools.partial(exact, "tiny"),
                   capacity=48, tol=TOL)  # fmt: skip
