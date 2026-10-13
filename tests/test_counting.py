"""PR 14a: the stochastic completeness count (docs/plans/pr-14a-counting.md, tests 1-13)."""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
from helpers import MATERIALS, STEEL, R, mixed_mesh
from jax.tree_util import Partial

from jaxmodal.eigen import counting
from jaxmodal.eigen.counting import (
    MAX_DEGREE,
    Q_MIN,
    _grid,
    _theta,
    count,
    count_degree,
    jackson_step,
    step_value,
    tail_bias,
)
from jaxmodal.materials import Material, isotropic
from jaxmodal.memory import (
    amg_memory,
    count_memory,
    gmg_memory,
    probe_block,
    stencil_memory,
    unstructured_memory,
)
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import (
    a_p,
    ap_operator,
    assemble,
    dof_mask,
    rigid_modes,
    stencil,
    unstructured,
)
from jaxmodal.operators.matfree import _block_data, element_matrices
from jaxmodal.operators.sampling import (
    element_factor,
    mass_sample,
    mass_sample_keys,
    reference_factor,
    xi_shapes,
)
from jaxmodal.operators.voxel import reference_element
from jaxmodal.precond import amg, gmg, jacobi

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}
KEEP = 14  # modes in W; the cut is after mode 6 (g = 0.41)


def _dense_free(K, M, free):
    return K.toarray()[np.ix_(free, free)], M.toarray()[np.ix_(free, free)]


@functools.cache
def ref(mass="consistent", nodal=False, bcs="clamped"):
    """Reference mesh (the dense prototype's): op, free, K, M and all eigenpairs (lam, W with
    constrained rows 0)."""
    mesh = box_mesh((6, 5, 4), (1 / 6, 1 / 5, 1 / 4), type="hex20")
    free = dof_mask(mesh, {"xmin": "xyz"} if bcs == "clamped" else {})
    nm = np.linspace(0.0, 2e-3, len(mesh.points)) if nodal else None
    K, M = assemble(mesh, MAT, mass=mass, nodal_mass=nm)
    lam, V = sla.eigh(*_dense_free(K, M, free))
    W = np.zeros((len(free), len(lam)))
    W[free] = V
    return mesh, unstructured(mesh, MAT, mass=mass, nodal_mass=nm), free, K, M, lam, W


def _dense_apply(Ainv, r):
    return Ainv @ r


def exact_precond(op, free, K, M, lam_c):
    """A_P^-1 of ap_operator(op, free, lam_c) as a pytree callable (PCG in 1-2 iterations)."""
    s = float(ap_operator(op, free, lam_c).s)
    return Partial(_dense_apply, jnp.asarray(np.linalg.inv(a_p(K + lam_c * M, free, s).toarray())))


def expected(lam, kept, lam_c, d):
    """sum over modes not in W of q(lam_i)^2 (direct cosine sum)."""
    out = np.setdiff1d(np.arange(len(lam)), kept)
    q = step_value(jackson_step(d), _theta(np.maximum(lam[out], 0) / lam_c))
    return float((q**2).sum())


def run(miss=(), mass="consistent", nodal=False, exact=True, **kw):
    """count on the reference mesh, cut after mode 6, W = the first KEEP modes minus miss."""
    _, op, free, K, M, lam, W = ref(mass, nodal)
    kept = np.setdiff1d(np.arange(KEEP), miss)
    lam_c = (lam[5] + lam[6]) / 2
    P = exact_precond(op, free, K, M, lam_c) if exact else None
    est, info = count(op, free, W[:, kept], lam[kept], lam[5], lam[6], precond=P, **kw)
    return est, info, expected(lam, kept, lam_c, info["degree"])


# ---------------------------------------------------------------- 1, 2: probes


@pytest.mark.parametrize("t", ["hex8", "hex20", "hex27"])
def test_element_factor(t):
    """Test 1: F F^T = M_e (element_matrices) to 1e-12, padded elements give 0."""
    mesh = box_mesh((3, 2, 2), (0.3, 0.5, 0.2), type=t)
    mesh.material[:] = np.arange(mesh.n_el) % 2
    op = unstructured(mesh, MATERIALS, chunk=5).scaled(s_M=jnp.linspace(0.5, 2, mesh.n_el))
    b = op.blocks[0]
    d = _block_data(op, b)
    for i in range(b.dofs.shape[0] // b.chunk):  # the last chunk is padded (12 = 2 * 5 + 2)
        sl = {k: v[i * b.chunk : (i + 1) * b.chunk] for k, v in d.items()}
        F = element_factor(op, b, sl)
        Me = element_matrices(op, b, sl)[1]
        assert np.abs(F @ jnp.swapaxes(F, 1, 2) - Me).max() <= 1e-12 * np.abs(Me).max()
    assert np.abs(F[2:]).max() == 0


def test_reference_factor():
    """Test 1, voxel: B B^T = reference_element's m_e."""
    op = stencil(voxel_mesh((2, 2, 2), (0.3, 0.5, 0.2)), MAT)
    B, me = reference_factor(op), reference_element(op)[1]
    assert np.abs(B @ B.T - me).max() <= 1e-12 * np.abs(me).max()


def _S(op):
    """The dense sampling matrix S (mass_sample on unit xi columns)."""
    sh = xi_shapes(op)
    flat = [sh["nodal"], *sh.get("elem", ())]
    sizes = [int(np.prod(s)) for s in flat]
    parts = np.split(np.eye(sum(sizes)), np.cumsum(sizes)[:-1])
    xi = {"nodal": jnp.asarray(parts[0])}
    if "elem" in sh:
        xi["elem"] = tuple(
            jnp.asarray(p.reshape(*s, -1)) for p, s in zip(parts[1:], sh["elem"], strict=True)
        )
    return np.asarray(mass_sample(op, xi))


def _cov_cases():
    rng = np.random.default_rng(0)
    m = box_mesh((2, 2, 2), type="hex20")
    m.material[:] = rng.integers(0, 2, m.n_el)
    nm, sM = rng.uniform(0, 1, len(m.points)), rng.uniform(0.5, 2, m.n_el)
    mk = np.ones((3, 2, 2), bool)
    mk[0, 0, 0] = False
    v = voxel_mesh((3, 2, 2), (0.5, 0.3, 0.2), mask=mk)
    vn = np.linspace(0, 0.1, len(v.points))
    return {
        "hex20": (m, MATERIALS, dict(nodal_mass=nm), sM),
        "mixed": (mixed_mesh(rng), MATERIALS, dict(orientations={1: R}), None),
        "lumped": (m, MATERIALS, dict(mass="lumped", nodal_mass=nm), sM),
        "voxel": (v, MAT, dict(nodal_mass=vn), None),
        "voxel-lumped": (v, MAT, dict(mass="lumped", nodal_mass=vn), None),
    }


@pytest.mark.parametrize("case", ["hex20", "mixed", "lumped", "voxel", "voxel-lumped"])
def test_probe_covariance(case):
    """Test 2: S S^T masked equals the masked SciPy M to 1e-12."""
    mesh, mats, kw, sM = _cov_cases()[case]
    build = stencil if case.startswith("voxel") else unstructured
    op = build(mesh, mats, **kw)
    if sM is not None:
        op = op.scaled(s_M=jnp.asarray(sM))
    M = assemble(mesh, mats, s_M=sM, **kw)[1].toarray()
    f = np.arange(op.n_dof) % 7 != 3  # an arbitrary mask
    S = _S(op) * f[:, None]
    assert np.abs(S @ S.T - M * np.outer(f, f)).max() <= 1e-12 * np.abs(M).max()


def test_keys_per_probe():
    """mass_sample_keys: column j depends only on keys[j] (any block width)."""
    _, op, *_ = ref()
    keys = jax.vmap(jax.random.fold_in, (None, 0))(jax.random.key(3), jnp.arange(5))
    g = mass_sample_keys(op, keys)
    assert np.abs(g[:, 2:4] - mass_sample_keys(op, keys[2:4])).max() == 0


# ---------------------------------------------------------------- 3: polynomial


def test_polynomial():
    """Test 3: the DCT grid against the direct sum, Jackson bounds, the degree rule."""
    gamma = jackson_step(57)
    th, q = _grid(gamma)
    assert np.abs(q - step_value(gamma, th)).max() <= 1e-13
    assert q.min() >= -1e-15 and q.max() <= 1 + 1e-15
    for g in (0.41, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        d, bias = count_degree(g, 1 + g / 2, 50, 10**6)
        tl = _theta(1 - g / 2)
        assert step_value(jackson_step(d), tl)[0] ** 2 >= Q_MIN
        assert step_value(jackson_step(d - 1), tl)[0] ** 2 < Q_MIN
        assert 19 <= d * g <= 24 and bias <= 0.05
    d_in = count_degree(0.5, 1.25, 50, 10**6)[0]
    d, bias = count_degree(0.5, 1.25, 5000, 10**6)  # the tail bias sets d
    assert d > d_in and bias <= 0.05
    assert tail_bias(jackson_step(d - 1), 0.5, 1.25, 5000, 10**6) > 0.05


def test_degree_cap():
    """Test 3: 23/g > MAX_DEGREE: not run, NaN, unverifiable."""
    _, op, free, _, _, lam, W = ref()
    lam_next = lam[5] * (1 + 1e-4)
    est, info = count(op, free, W[:, :KEEP], lam[:KEEP], lam[5], lam_next)
    g = (lam_next - lam[5]) / ((lam_next + lam[5]) / 2)
    assert np.isnan(est) and not info["verifiable"] and not info["passed"]
    assert info["degree"] == int(np.ceil(23 / g)) > MAX_DEGREE and info["steps"] == 0
    assert np.isnan(info["probes"]).all()


# ---------------------------------------------------------------- 4, 5: estimator


@pytest.mark.parametrize(
    "miss,mass,nodal",
    [((), "consistent", False), ((2,), "consistent", False), ((2,), "lumped", False),
     ((2,), "consistent", True)],
    ids=["complete", "missing", "lumped", "nodal"],
)  # fmt: skip
def test_expected_value(miss, mass, nodal):
    """Test 4: the mean of 2000 probes against the exact eigen-sum, within 5 standard errors."""
    est, info, exp = run(miss, mass, nodal, n_probes=2000, block=500, tau=1e-10)
    se = info["std"] / np.sqrt(2000)
    print(f"{miss} {mass} nodal={nodal}: {est:.4g} vs {exp:.4g} (se {se:.2g})")
    assert info["verifiable"] and abs(est - exp) <= 5 * se


def test_expected_rigid():
    """Test 4: a free body with R (k = 6), and with R omitted (about 6 more)."""
    mesh, op, free, K, M, lam, W = ref(bcs="free")
    Rn = rigid_modes(mesh)
    L = np.linalg.cholesky(Rn.T @ (M @ Rn))
    R = np.linalg.solve(L, Rn.T).T
    el = np.arange(6, 6 + KEEP)  # elastic modes
    g = (lam[el[1:]] - lam[el[:-1]]) / (lam[el[1:]] + lam[el[:-1]])
    j = int(np.argmax(g[:8]))
    lo, hi = lam[el[j]], lam[el[j + 1]]
    lam_c = (lo + hi) / 2
    P = exact_precond(op, free, K, M, lam_c)
    for Rc, add in ((R, 0), (None, 6)):
        est, info = count(op, free, W[:, el], lam[el], lo, hi, R=Rc, precond=P, n_probes=2000,
                          block=500, tau=1e-10)  # fmt: skip
        exp = expected(lam, np.r_[np.arange(6), el], lam_c, info["degree"]) + add * float(
            step_value(jackson_step(info["degree"]), 0.0)[0] ** 2
        )
        se = info["std"] / np.sqrt(2000)
        print(f"R={Rc is not None}: {est:.4g} vs {exp:.4g} (se {se:.2g})")
        assert abs(est - exp) <= 5 * se


def dense_probes(op, free, K, M, W, lam_c, d, n, seed=0):
    """The dense estimator with exact solves on the same probes as count."""
    Kf, Mf = _dense_free(K, M, free)
    keys = jax.vmap(jax.random.fold_in, (None, 0))(jax.random.key(seed), jnp.arange(n))
    g = np.asarray(mass_sample_keys(op, keys))[free]
    Wf = W[free]
    proj = lambda Z: Z - Wf @ (Wf.T @ (Mf @ Z))  # noqa: E731
    cf = sla.cho_factor(Kf + lam_c * Mf)
    X = lambda Z: 2 * lam_c * sla.cho_solve(cf, Mf @ Z) - Z  # noqa: E731
    gamma = jackson_step(d)
    w_prev, w = 0, proj(np.linalg.solve(Mf, g))
    u = gamma[0] * w
    for k in range(1, d + 1):
        w_prev, w = w, (1 if k == 1 else 2) * X(w) - (0 if k == 1 else w_prev)
        u = u + gamma[k] * w
    u = proj(u)
    return np.einsum("ij,ij->j", u, Mf @ u), Mf


@pytest.mark.parametrize("case", ["hex20", "lumped", "voxel"])
def test_against_dense(case):
    """Test 5: count's s_j against the dense estimator on the same probes."""
    if case == "voxel":
        mesh = voxel_mesh((4, 4, 4), (0.25, 0.25, 0.25))
        free = dof_mask(mesh, {"xmin": "xyz"})
        K, M = assemble(mesh, MAT)
        lam, V = sla.eigh(*_dense_free(K, M, free))
        W = np.zeros((len(free), len(lam)))
        W[free] = V
        op = stencil(mesh, MAT)
    else:
        _, op, free, K, M, lam, W = ref("lumped" if case == "lumped" else "consistent")
    gaps = (lam[1:12] - lam[:11]) / (lam[1:12] + lam[:11])
    j = int(np.argmax(gaps[2:])) + 2  # cut after mode j + 1, mode 1 left out
    lam_c = (lam[j] + lam[j + 1]) / 2
    P = exact_precond(op, free, K, M, lam_c)
    for miss in ((), (1,)):
        kept = np.setdiff1d(np.arange(KEEP), miss)
        ref_s, Mf = None, None
        for tau in (1e-12, None):
            est, info = count(op, free, W[:, kept], lam[kept], lam[j], lam[j + 1], precond=P,
                              n_probes=20, tau=tau)  # fmt: skip
            if ref_s is None:
                ref_s, Mf = dense_probes(op, free, K, M, W[:, kept], lam_c, info["degree"], 20)
            err = np.abs(info["probes"] - ref_s)
            if tau is not None:
                assert (err <= (1e-8 * ref_s if miss else 1e-10)).all(), err.max()
            else:  # the linear model with sqrt(kappa(D^-1/2 M D^-1/2))
                dm = np.sqrt(np.diag(Mf))
                ev = np.linalg.eigvalsh(Mf / np.outer(dm, dm))
                bound = 2 * np.sqrt(ref_s) * 0.01 * np.sqrt(ev[-1] / ev[0]) + 1e-4
                assert (err <= bound).all()


# ---------------------------------------------------------------- 6, 7: detection, kappa


@pytest.mark.parametrize(
    "miss", [(2,), pytest.param((), marks=pytest.mark.slow)], ids=["missing", "complete"]
)
def test_detection(miss):
    """Test 6: 100 sets of 50 probes (one 5000-probe count): mode 3 left out is detected in
    >= 97 sets; with W complete every set is < 0.5."""
    _, info, _ = run(miss, n_probes=5000, block=500)
    sets = info["probes"].reshape(100, 50).mean(1)
    print(f"miss={miss}: {(sets >= 0.5).sum()} of 100 sets >= 0.5")
    if miss:
        assert (sets >= 0.5).sum() >= 97
    else:
        assert (sets < 0.5).all()


def test_kappa():
    """Test 7: slender hex20 cantilever with kappa >= 1e7, AMG at lam_c, default tau."""
    mesh = box_mesh((64, 1, 1), (1 / 64,) * 3, type="hex20")
    free = dof_mask(mesh, {"xmin": "xyz"})
    K, M = assemble(mesh, STEEL)
    lam, V = sla.eigh(*_dense_free(K, M, free))
    assert lam[-1] / lam[0] >= 1e7
    W = np.zeros((len(free), 10))
    W[free] = V[:, :10]
    op = unstructured(mesh, STEEL)
    lam_c = (lam[3] + lam[4]) / 2
    est, info = count(op, free, W, lam[:10], lam[3], lam[4],
                      precond=amg(op, free, lam_c, mesh.points))  # fmt: skip
    print(f"kappa {lam[-1] / lam[0]:.2e}: d {info['degree']}, PCG {info['pcg_iterations']}")
    assert info["verifiable"] and est < 0.5


# ---------------------------------------------------------------- 8: floor and flags


def test_floor():
    """Test 8: tau two decades below the measured floor, and 1e-17: unverifiable, NaN, early
    stop."""
    floor = None
    for tau in 10.0 ** -np.arange(10, 17):
        _, info, _ = run(exact=False, degree=1, tau=tau)
        if not info["verifiable"]:
            break
        floor = tau
    print(f"Jacobi floor on the reference mesh: {floor:g}")
    d = run(exact=False)[1]["degree"]
    for tau in (floor / 100, 1e-17):
        est, info, _ = run(exact=False, tau=tau)
        assert np.isnan(est) and not info["verifiable"] and not info["passed"]
        assert np.isnan(info["probes"]).all() and info["steps"] < d


def test_flags(monkeypatch):
    """Test 8: maxit = 2 and a failed y solve are unverifiable; a normal run is verifiable."""
    assert not run(exact=False, maxit=2)[1]["verifiable"]
    est, info, _ = run()
    assert info["verifiable"] and info["passed"] and est < 0.5
    monkeypatch.setattr(counting, "_solve_y", lambda op, m, g, tau, maxit: (g, False))
    est, info, _ = run()
    assert not info["verifiable"] and info["steps"] == 0 and np.isnan(est)


# ---------------------------------------------------------------- 9: memory


def _recurrence_temps(op, free, P, b):
    m = jnp.asarray(free, float)[:, None]
    A = ap_operator(op, free, 1.0)
    w0 = jnp.zeros((op.n_dof, b))
    lowered = counting._recurrence.lower(A, P, m, w0, jnp.zeros(64), 3, 1.0, 1e-8, 100)
    return lowered.compile().memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize("pc", ["jacobi", "mg"])
@pytest.mark.parametrize("kind", ["hex20", "voxel"])
def test_step_memory(kind, pc):
    """Test 9a: memory_analysis of the recurrence <= count_memory's step (w0 and u aside), and
    the fit is tight at one size."""
    if kind == "hex20":
        mesh = box_mesh((12,) * 3, (1 / 12,) * 3, type="hex20")
        op, est = unstructured(mesh, STEEL), lambda b: unstructured_memory(mesh, b)
        pcm = (lambda b: amg_memory(mesh, b)) if pc == "mg" else None
    else:
        mesh = voxel_mesh((16,) * 3, (1 / 16,) * 3)
        op, est = stencil(mesh, STEEL), lambda b: stencil_memory(mesh, b)
        pcm = (lambda b: gmg_memory(mesh, b)) if pc == "mg" else None
    free = dof_mask(mesh, {"xmin": "xyz"})
    if pc == "mg":
        P = amg(op, free, 1.0, mesh.points) if kind == "hex20" else gmg(op, free, 1.0)
    else:
        P = jacobi(op, free, 1.0)
    for b in (4, 16, 32):
        got = _recurrence_temps(op, free, P, b)
        e = count_memory(est, op.n_dof, b, voxel=kind == "voxel", precond_memory=pcm)
        model = e["step"] - 2 * 8 * op.n_dof * b
        print(f"{kind} {pc} b={b}: {got / 1e6:.1f} MB, model {model / 1e6:.1f} MB")
        assert got <= model and (pc == "mg" or got >= 0.6 * model)


def test_sampling_memory():
    """Test 9b: the sampling kernels <= the sampling phase <= the step phase."""
    from jaxmodal.linalg.krylov import _pcg

    for _mesh, op, est, voxel in (
        (m := box_mesh((12,) * 3, (1 / 12,) * 3, type="hex20"), unstructured(m, STEEL),
         lambda b, m=m: unstructured_memory(m, b), False),
        (v := voxel_mesh((16,) * 3, (1 / 16,) * 3), stencil(v, STEEL),
         lambda b, v=v: stencil_memory(v, b), True),
    ):  # fmt: skip
        n = op.n_dof
        m = jnp.ones((n, 1))
        for b in (4, 32):
            keys = jax.vmap(jax.random.fold_in, (None, 0))(jax.random.key(0), jnp.arange(b))
            g = jnp.zeros((n, b))
            temps = [
                mass_sample_keys.lower(op, keys).compile().memory_analysis().temp_size_in_bytes,
                _pcg.lower(Partial(counting._mp, op, m), Partial(counting._jac, m), g, 1e-8, 9)
                .compile().memory_analysis().temp_size_in_bytes,
            ]  # fmt: skip
            e = count_memory(est, n, b, voxel=voxel)
            assert max(temps) + 2 * 8 * n * b <= e["sampling"] <= e["step"]


def test_probe_block():
    """Test 9c: the largest b with total(b) <= budget < total(b + 1), capped, raises below 1."""
    mesh = box_mesh((12,) * 3, (1 / 12,) * 3, type="hex20")
    n = 3 * len(mesh.points)

    def est(b):
        return unstructured_memory(mesh, b)

    total = lambda b: count_memory(est, n, b)["total"]  # noqa: E731
    budget = (total(17) + total(18)) / 2
    assert probe_block(budget, 50, est, n) == 17
    assert probe_block(10 * total(50), 50, est, n) == 50
    with pytest.raises(ValueError, match="below one probe"):
        probe_block(total(1) - 1, 50, est, n)


PEAK_SETUP = """
import numpy as np
from jaxmodal import config
config.enable_x64()
from jaxmodal.eigen.counting import count
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import voxel_mesh
from jaxmodal.operators import dof_mask, stencil
mats = {0: Material(isotropic(3.0, 0.3), 2.0)}
mesh = voxel_mesh((S, S, S), (1 / S,) * 3)
free = dof_mask(mesh, {"xmin": "xyz"})
op = stencil(mesh, mats, mass="lumped")
d = np.asarray(op.diag_M())
k = K
idx = np.flatnonzero(free)[:: len(np.flatnonzero(free)) // k][:k]
W = np.zeros((op.n_dof, k))  # host, M-orthonormal (diagonal M, disjoint supports)
W[idx, np.arange(k)] = 1 / np.sqrt(d[idx])
lam = np.linspace(1.0, 2.0, k)
count(op, free, W, lam, 1.0, 1.5, degree=3, tau=1e-2, block=B, n_probes=B)  # warm-up
del op
"""
PEAK = """
op = stencil(mesh, mats, mass="lumped")
count(op, free, W, lam, 1.0, 1.5, degree=3, tau=1e-2, block=B, n_probes=B)
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_count_peak():
    """Test 9d: at count_memory >= 2 GB the measured process peak is within 20%, and W (an
    n x k device copy would add 60%) stays on the host."""
    from benchmarks import harness

    s, b = 56, 24
    mesh = voxel_mesh((s, s, s), (1 / s,) * 3)
    n = 3 * len(mesh.points)
    e = count_memory(lambda m: stencil_memory(mesh, m, mass="lumped"), n, b, voxel=True)
    k = int(0.6 * e["total"] / (8 * n))
    sub = {"S": str(s), "K": str(k), "B": str(b)}
    setup, code = PEAK_SETUP, PEAK
    for a, v in sub.items():
        setup, code = setup.replace(a, v), code.replace(a, v)
    peak = harness.measure(code, setup)["peak_bytes"]
    print(f"count {s}^3 b={b} k={k}: estimate {e['total'] / 1e9:.2f} GB, peak {peak / 1e9:.2f}")
    assert e["total"] >= 2e9 and abs(e["total"] - peak) <= 0.2 * peak


# ---------------------------------------------------------------- 10, 11: blocks, fp32 W


def test_blocks_and_compiles():
    """Test 10: b = 7 and b = 50 give the same s_j; degrees 100 and 120 share one compile."""
    a = run((2,), tau=1e-12, block=7)[1]["probes"]
    b = run((2,), tau=1e-12, block=50)[1]["probes"]
    assert np.abs(a - b).max() <= 1e-10 * np.abs(b).max()
    run(degree=100, n_probes=4)
    n0 = counting._recurrence._cache_size()
    run(degree=120, n_probes=4)
    assert counting._recurrence._cache_size() == n0


@pytest.mark.parametrize("miss", [(), (2,)], ids=["complete", "missing"])
def test_fp32_w(miss):
    """Test 11: an fp32 W changes the estimate by <= 1e-4 and not the verdict."""
    _, op, free, K, M, lam, W = ref()
    kept = np.setdiff1d(np.arange(KEEP), miss)
    lam_c = (lam[5] + lam[6]) / 2
    P = exact_precond(op, free, K, M, lam_c)
    out = [count(op, free, Wd[:, kept], lam[kept], lam[5], lam[6], precond=P)
           for Wd in (W, W.astype(np.float32))]  # fmt: skip
    assert abs(out[0][0] - out[1][0]) <= 1e-4 and out[0][1]["passed"] == out[1][1]["passed"]


# ---------------------------------------------------------------- 12, 13


@pytest.mark.slow
def test_benchmark_smoke():
    """Test 12: counting_study.run(small=True)."""
    from benchmarks import counting_study

    out = counting_study.run(small=True)
    assert out["cuts"] and out["errors"]


def test_errors():
    """Test 13: input checks."""
    _, op, free, _, _, lam, W = ref()
    W14, l14 = W[:, :KEEP], lam[:KEEP]
    bad = W14.copy()
    bad[:, 3] *= 1.01
    cases = [
        (dict(lam_last=lam[6], lam_next=lam[5]), "lam_last"),
        (dict(lam_last=-1.0), "lam_last"),
        (dict(lam_next=2 * lam[KEEP]), "lam_last"),
        (dict(W=W14[:, :5]), "W must have shape"),
        (dict(lam=l14[:5]), "W must have shape"),
        (dict(W=bad), "M-orthonormal"),
        (dict(n_probes=0), "n_probes"),
        (dict(n_probes=True), "n_probes"),
        (dict(n_probes=2.5), "n_probes"),
        (dict(block=0), "block"),
        (dict(maxit=0), "maxit"),
        (dict(tau=0.0), "tau"),
        (dict(tau=-1.0), "tau"),
    ]
    for kw, msg in cases:
        args = dict(W=W14, lam=l14, lam_last=lam[5], lam_next=lam[6]) | kw
        with pytest.raises(ValueError, match=msg):
            count(op, free, args.pop("W"), args.pop("lam"), args.pop("lam_last"),
                  args.pop("lam_next"), **args)  # fmt: skip
