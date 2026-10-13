"""PR 15a: block shift-invert Lanczos on one slice (docs/plans/pr-15a-shift-invert.md, tests
1-16)."""

import functools
import importlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse.linalg as spla

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
              unstructured, "amg", 30),
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
    done, inside, below, above, r_acc, res = si._slice_check(mu, 1.5, 3.5, 1e-8, fake(ok, t, asked))
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
    done, inside, below, above, r_acc, _ = si._slice_check(mu, 1.5, 3.05, 1e-8, fake(ok, t, asked))
    assert done and list(inside) == [1, 2, 3] and above == 4 and r_acc == 1e-3
    assert asked == [[1, 2, 0, 3], [4]]
    # a chain of re-forms: each merge brings a larger threshold until the set stops growing;
    # at r = 1e-3 the window ends at 3.30055 and 3.32 joins only through the chain from 3.3
    mu = np.array([1.0, 2.0, 3.0, 3.001, 3.05, 3.3, 3.32, 10.0])
    t = {i: 1e-8 for i in range(8)} | {3: 1e-4, 4: 1e-3}
    ok = {i: 1e-9 for i in range(8)}
    done, inside, _, above, r_acc, _ = si._slice_check(mu, 1.5, 3.0005, 1e-8, fake(ok, t, []))
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
    code = "Q = jax.random.normal(jax.random.key(1), (n, cap)); V = _form_v(Q, 0, Y, 24)"
    peak = harness.measure(code, setup)["peak_bytes"]
    print(f"_form_v: model {model / 1e9:.2f} GB, peak {peak / 1e9:.2f}")
    assert model >= 2e9 and abs(model - peak) <= 0.2 * peak


@pytest.mark.slow
def test_benchmark_smoke():
    """Test 16: shift_invert_study.run(small=True)."""
    from benchmarks import shift_invert_study

    out = shift_invert_study.run(small=True)
    assert out["grid"]["hex20"]["rows"] and out["slice"]["converged"]
