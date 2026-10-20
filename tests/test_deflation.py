"""PR 18f: MINRES deflated by known modes below sigma (docs/plans/pr-18f-deep-bands.md, tests
1-2): _dminres against dense references, si_slice with Z."""

import functools
import gc
import sys
import weakref

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
from test_shift_invert import (
    MAT,
    TAU,
    Spy,
    check,
    edges,
    lam_voxel,
    precond,
    prob,
    si,
    solve,
)

from jaxmodal.linalg import minres
from jaxmodal.memory import gmg_memory, si_memory, stencil_memory
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import ap_operator, assemble, dof_mask, unstructured


@functools.cache
def dense(name):
    """free, op, the dense A_P builder, the free-DOF eigenpairs (lam, V on all DOFs, masked)."""
    if name == "tiny":
        _, free, op, K, M = prob("tiny")[:5]
    else:  # hex20 4^3 clamped
        mesh = box_mesh((4, 4, 4), (1 / 4,) * 3, type="hex20")
        free, op = dof_mask(mesh, {"xmin": "xyz"}), unstructured(mesh, MAT)
        K, M = assemble(mesh, MAT)
    f = np.flatnonzero(free)
    lam, X = sla.eigh(K[f][:, f].toarray(), M[f][:, f].toarray())
    V = np.zeros((op.n_dof, len(lam)))
    V[f] = X
    return free, op, K.toarray(), M.toarray(), lam, V


def setup(name, nz=None, noise=0.0, seed=0):
    """A at sigma mid-gap (relative gap >= 1e-2) above the first ~nz modes, Z = the modes below
    sigma (plus noise of relative size `noise`, masked), Zd, H."""
    free, op, K, M, lam, V = dense(name)
    gap = np.diff(lam) / lam[1:]
    i = next(i for i in range(nz or len(lam) // 3, len(lam) - 1) if gap[i] >= 1e-2)
    sigma = (lam[i] + lam[i + 1]) / 2
    A = ap_operator(op, free, -sigma)
    Z = V[:, : i + 1].copy()
    rng = np.random.default_rng(seed)
    Z += noise * free[:, None] * rng.standard_normal(Z.shape) * np.abs(Z).max()
    c = 24
    Zd = si._zblock(list(Z.T), op.n_dof, c)
    return A, Z, Zd, si._deflate_setup(A, Zd, Z.shape[1], c), sigma


def rhs(name, k=6, seed=1):
    free, op = dense(name)[:2]
    return jnp.asarray(np.random.default_rng(seed).standard_normal((op.n_dof, k)) * free[:, None])


def true_res(A, b, x):
    b = np.asarray(b)
    return np.linalg.norm(b - np.asarray(A(x)), axis=0) / np.linalg.norm(b, axis=0)


@pytest.mark.parametrize("name", ["tiny", "hex20"])
def test_dminres_accuracy(name):
    """Test 1: Z = the modes below sigma: at tol 1e-12, x matches the dense solve within
    1e3 kappa 1e-12; at 1e-8 the true residual is <= 1.01 tau, also with 1e-3 noise in Z."""
    free, op, K, M, lam, V = dense(name)
    A, Z, Zd, H, sigma = setup(name)
    b = rhs(name)
    f = np.flatnonzero(free)
    Af = (K - sigma * M)[np.ix_(f, f)]
    kappa = np.linalg.cond(Af)
    x, inf = si._dminres(A, b, None, Zd, H, 1e-12, 20000)
    xr = np.linalg.solve(Af, np.asarray(b)[f])
    err = np.linalg.norm(np.asarray(x)[f] - xr, axis=0) / np.linalg.norm(xr, axis=0)
    assert err.max() <= 1e3 * kappa * 1e-12, (err.max(), kappa)
    for noise in (0.0, 1e-3):
        A, Z, Zd, H, _ = setup(name, noise=noise)
        x, inf = si._dminres(A, b, None, Zd, H, TAU, 20000)
        assert np.asarray(inf["converged"]).all()
        assert true_res(A, b, x).max() <= 1.01 * TAU


def test_dminres_any_h():
    """Test 1: any symmetric H, through the real recombination: b - A x = P_Z b - P_Z A x~."""
    A, Z, Zd, _, _ = setup("hex20")
    rng = np.random.default_rng(2)
    H = rng.standard_normal((Zd.shape[1],) * 2)
    H = jnp.asarray(H + H.T)
    b, xt = rhs("hex20"), rhs("hex20", seed=3)
    x0, pb, _, _ = si._dprep(A, Zd, H, b, TAU)
    x = si._drecomb(A, Zd, H, x0, xt)
    lhs = np.asarray(b - A(x))
    ref = np.asarray(pb - si._zop(A, Zd, H, xt))
    assert np.abs(lhs - ref).max() <= 1e-12 * np.abs(ref).max()


@pytest.mark.parametrize("delta", [0.0, 1e-6, 1e-3])
def test_dminres_duplicates(delta):
    """Test 1: an exact duplicate column (E singular), a dropped near-duplicate (1e-6, |H| as
    without it) and a kept one (1e-3, |H| >> that): the true residual <= 1.01 tau; on the exact and
    dropped cases H is finite with HEH = H and EHE = E (inv fails these)."""
    free, op, *_ = dense("hex20")
    A, Z, _, H0, _ = setup("hex20")
    h0 = np.abs(np.asarray(H0)).max()  # |H| without the duplicate
    rng = np.random.default_rng(4)
    dup = Z[:, -1] + delta * free * rng.standard_normal(len(free)) * np.abs(Z[:, -1]).max()
    Z2 = np.c_[Z, dup]
    nz, c = Z2.shape[1], 24
    Zd = si._zblock(list(Z2.T), op.n_dof, c)
    H = si._deflate_setup(A, Zd, nz, c)
    b = rhs("hex20")
    x, inf = si._dminres(A, b, None, Zd, H, TAU, 20000)
    assert true_res(A, b, x).max() <= 1.01 * TAU
    Hn = np.asarray(H)[:nz, :nz]
    assert np.isfinite(Hn).all() and np.allclose(Hn, Hn.T, rtol=0, atol=0)
    if delta < 1e-4:
        E = np.asarray(si._zaz(A, Zd, c=c))[:nz, :nz]
        E = (E + E.T) / 2
        assert np.abs(Hn @ E @ Hn - Hn).max() <= 1e-10 * np.abs(Hn).max()
        assert np.abs(E @ Hn @ E - E).max() <= 1e-10 * np.abs(E).max()
        assert np.abs(Hn).max() <= 2 * h0  # the near-duplicate dropped
    else:
        assert np.abs(Hn).max() >= 100 * h0  # kept: |H| >> 1


def test_deflate_setup_symmetric(monkeypatch):
    """Test 1: H is symmetric when E is slightly nonsymmetric."""
    A, Z, Zd, _, _ = setup("tiny")
    real = si._zaz

    def skew(*a, **kw):
        E = np.asarray(real(*a, **kw))
        return jnp.asarray(E + 1e-9 * np.triu(np.abs(E).max() * np.ones_like(E), 1))

    monkeypatch.setattr(si, "_zaz", skew)
    H = np.asarray(si._deflate_setup(A, Zd, Z.shape[1], 24))
    assert np.array_equal(H, H.T)


def test_dminres_tol_scaling(monkeypatch):
    """Test 1: b = M (Z c + 0.01 ||Z c||_M noise): ||P_Z b|| / ||b|| <= 0.1; minres sees the
    per-column tolerance tau ||b|| / ||P_Z b||, and the returned residuals are within 2x of the
    true ones (unscaled they are >= 10x off)."""
    free, op, K, M, lam, V = dense("hex20")
    A, Z, Zd, H, _ = setup("hex20")
    rng = np.random.default_rng(5)
    Zc = Z @ rng.standard_normal((Z.shape[1], 6))
    nm = np.sqrt(np.einsum("ij,ij->j", Zc, M @ Zc))
    noise = free[:, None] * rng.standard_normal(Zc.shape)
    noise /= np.sqrt(np.einsum("ij,ij->j", noise, M @ noise))
    b = jnp.asarray(free[:, None] * (M @ (Zc + 0.01 * nm * noise)))
    _, pb, _, _ = si._dprep(A, Zd, H, b, TAU)
    nb, npb = (np.linalg.norm(np.asarray(v), axis=0) for v in (b, pb))
    assert (npb / nb).max() <= 0.1
    tols = []
    real = minres

    def rec(A_, b_, precond=None, tol=1e-8, maxit=10000):
        tols.append(np.asarray(tol))
        return real(A_, b_, precond, tol=tol, maxit=maxit)

    monkeypatch.setattr(si, "minres", rec)
    x, inf = si._dminres(A, b, None, Zd, H, TAU, 20000)
    assert np.allclose(tols[0], TAU * nb / npb, rtol=1e-10)
    tr, rr = true_res(A, b, x), np.asarray(inf["residuals"])
    assert (rr <= 2 * tr).all() and (tr <= 2 * rr).all()
    assert (rr / (npb / nb) >= 10 * tr).all()  # the unscaled ones


def test_dminres_solved_columns(monkeypatch):
    """Test 1: b = A Z c (P_Z b = 0 up to round-off) and an exactly zero P_Z b: residual <= tau,
    no NaN."""
    A, Z, Zd, H, _ = setup("hex20")
    rng = np.random.default_rng(6)
    b = A(jnp.asarray(Z @ rng.standard_normal((Z.shape[1], 6))))
    x, inf = si._dminres(A, b, None, Zd, H, TAU, 20000)
    r = np.asarray(inf["residuals"])
    assert np.isfinite(r).all() and (r <= TAU).all() and true_res(A, b, x).max() <= TAU
    real = si._dprep

    def zero(*a):
        x0, pb, tc, scale = real(*a)
        return x0, 0 * pb, a[4] * jnp.linalg.norm(a[3], axis=0) / jnp.finfo(float).tiny, 0 * scale

    monkeypatch.setattr(si, "_dprep", zero)
    x, inf = si._dminres(A, b, None, Zd, H, TAU, 20000)
    r = np.asarray(inf["residuals"])
    assert np.isfinite(r).all() and (r <= TAU).all() and np.isfinite(np.asarray(x)).all()


# ---------------------------------------------------------------- 2: si_slice with Z


@pytest.mark.slow
def test_slice_deflated():
    """Test 2: a slice above 30 modes of "hex20": with Z = the 30 modes below it the pairs
    match the Z=None run and eigsh, info["deflated"] = 30, and the first step's MINRES needs
    <= 0.6x the iterations; window modes below sigma (not the lo neighbour) passed both as W
    and in Z."""
    lr, Vr = prob("hex20")[6:]
    lo, hi = edges(lr, 30, 36)
    Z = [Vr[:, i] for i in range(30)]
    plain = solve("hex20", lo, hi)
    lam, V, info = solve("hex20", lo, hi, Z=Z)
    check("hex20", lo, hi, lam, V, info)
    assert info["deflated"] == 30 and plain[2]["deflated"] == 0
    assert np.allclose(lam, plain[0], rtol=0, atol=info["r_acc"] * lam.max() * 10)
    its, its0 = info["minres_iterations"][0], plain[2]["minres_iterations"][0]
    assert its <= 0.6 * its0, (its, its0)
    i0 = int(np.searchsorted(lr, lo))  # i0 - 1 stays the lo neighbour (W closes no side)
    w = np.arange(i0 - 4, i0 - 1)
    lam, V, info = solve("hex20", lo, hi, Z=Z, W=Vr[:, w], lam_W=lr[w])
    check("hex20", lo, hi, lam, V, info)
    assert info["deflated"] == 30


@pytest.mark.slow
def test_slice_deflated_move(monkeypatch):
    """Test 2: a first solve forced to stagnate moves sigma, and H is recomputed at the new
    sigma (a _deflate_setup spy sees both)."""
    lr, Vr = prob("hex20")[6:]
    lo, hi = edges(lr, 30, 36)
    sig, real = [], si._deflate_setup
    monkeypatch.setattr(si, "_deflate_setup",
                        lambda A, *a: sig.append(float(A.sigma)) or real(A, *a))  # fmt: skip

    def stall(A, b, P, tol, maxit, x, info):
        if spy.calls:
            return x, info
        return x, info | dict(residuals=jnp.full_like(info["residuals"], 1.0),
                              stagnated=jnp.ones_like(info["stagnated"]))  # fmt: skip

    spy = Spy(stall)
    monkeypatch.setattr(si, "minres", spy)
    lam, V, info = solve("hex20", lo, hi, Z=[Vr[:, i] for i in range(30)])
    check("hex20", lo, hi, lam, V, info)
    assert info["sigma_moved"] and len(sig) == 2 and sig[0] == -(lo + hi) / 2
    assert sig[1] == -info["sigma"] != sig[0]


def test_checked_keeps_z(monkeypatch):
    """Test 2: an si_checked restart (the count flagged once) passes the same Z again."""
    _, free, op, *_ = prob("tiny")
    lr, Vr = prob("tiny")[6:]
    lo, hi = edges(lr, 6, 9)
    Z = [Vr[:, i] for i in range(5)]
    seen, real = [], si.si_slice
    monkeypatch.setattr(si, "si_slice",
                        lambda *a, **kw: seen.append(kw.get("Z")) or real(*a, **kw))  # fmt: skip
    real_bc, est = si.band_count, iter([0.9])

    def bc(*a, _predict=False, **kw):
        out, ci = real_bc(*a, _predict=_predict, **kw)
        return (out if _predict else next(est, out)), ci

    monkeypatch.setattr(si, "band_count", bc)
    lam, V, info = si.si_checked(op, free, lo, hi, precond("tiny", (lo + hi) / 2),
                                 capacity=300, tol=1e-10, tau=TAU, Z=Z, verify=True)  # fmt: skip
    assert info["count"]["restarts"] == 1 and len(seen) == 2
    assert all(z is Z for z in seen) and info["runs"][1]["deflated"] == 5


def test_zd_freed(monkeypatch):
    """Test 7: si_slice's device Z block is dead after it returns (items and restarts upload
    it again)."""
    lr, Vr = prob("tiny")[6:]
    lo, hi = edges(lr, 6, 9)
    refs, real = [], si._upload

    def upload(B, cols, c):
        out = real(B, cols, c)
        refs.append((len(cols), weakref.ref(out)))
        return out

    monkeypatch.setattr(si, "_upload", upload)
    solve("tiny", lo, hi, Z=[Vr[:, i] for i in range(5)])
    gc.collect()
    z = [r for k, r in refs if k == 5]
    assert len(z) == 1 and z[0]() is None


Z_PEAK = """
import numpy as np
from jaxmodal import config
config.enable_x64()
from jaxmodal.eigen.shift_invert import si_slice
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import voxel_mesh
from jaxmodal.operators import dof_mask, stencil
from jaxmodal.precond import gmg
mats = {0: Material(isotropic(3.0, 0.3), 2.0)}
mesh = voxel_mesh((48,) * 3, (1 / 48,) * 3)
free = dof_mask(mesh, {"xmin": "xyz"})
op = stencil(mesh, mats, mass="lumped")
P = gmg(op, free, (LO + HI) / 2)
def run(nz):
    n, rng = op.n_dof, np.random.default_rng(0)
    B = np.zeros((n, NZ), order="F")  # the host store's block, counted in the peak
    for i in range(NZ):
        B[:, i] = np.where(free, rng.standard_normal(n), 0.0)
    Z = [B[:, i] for i in range(nz)] or None
    return si_slice(op, free, LO, HI, P, capacity=72, tol=1e-6, tau=1e-2, maxit_inner=200,
                    Z=Z)
"""


@pytest.mark.peak
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_z_peak():
    """Test 7: one si_slice on voxel 48^3 / GMG (353k DOF) below lambda_1 (A and E SPD) with Z =
    800 random masked columns of an 800-column host block, the capacity at its floor 72: the
    process peak within 20% of si_memory(n_modes=800, n_z=800); against Z=None (same block)
    the peak difference within 20% of the model's."""
    from benchmarks import harness

    lo, hi = (x * lam_voxel(8, 1)[0] for x in (0.2, 0.5))
    mesh = voxel_mesh((48,) * 3, (1 / 48,) * 3)
    n = 3 * len(mesh.points)

    op_m = functools.partial(stencil_memory, mesh, mass="lumped")
    pc_m = functools.partial(gmg_memory, mesh)

    def est(nz):
        return si_memory(op_m, n, 72, 800, precond_memory=pc_m, n_z=nz)["total"]

    setup = Z_PEAK.replace("LO", repr(lo)).replace("HI", repr(hi)).replace("NZ", "800")
    peak = {nz: harness.measure(f"run({nz})", setup)["peak_bytes"] for nz in (800, 0)}
    print(f"Z peak: estimates {est(800) / 1e9:.2f} / {est(0) / 1e9:.2f} GB, peaks"
          f" {peak[800] / 1e9:.2f} / {peak[0] / 1e9:.2f}")  # fmt: skip
    assert abs(est(800) - peak[800]) <= 0.2 * peak[800]
    delta = est(800) - est(0)
    assert 0.8 * delta <= peak[800] - peak[0] <= 1.2 * delta
