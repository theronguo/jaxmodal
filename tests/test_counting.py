"""PR 14a: the stochastic completeness count (docs/plans/pr-14a-counting.md, tests 1-13); PR
15b: the band-pass count (docs/plans/pr-15b-bandpass-count.md, tests 1-8, 11, 12)."""

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
    band_count,
    band_edges,
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
    """Test 1: (F F^T) (x) I_3 = M_e (element_matrices) to 1e-12, padded elements give 0."""
    mesh = box_mesh((3, 2, 2), (0.3, 0.5, 0.2), type=t)
    mesh.material[:] = np.arange(mesh.n_el) % 2
    op = unstructured(mesh, MATERIALS, chunk=5).scaled(s_M=jnp.linspace(0.5, 2, mesh.n_el))
    b = op.blocks[0]
    d = _block_data(op, b)
    for i in range(b.dofs.shape[0] // b.chunk):  # the last chunk is padded (12 = 2 * 5 + 2)
        sl = {k: v[i * b.chunk : (i + 1) * b.chunk] for k, v in d.items()}
        F = element_factor(op, b, sl)
        Me = element_matrices(op, b, sl)[1]
        FF = jnp.einsum("eaq,ebq,ij->eaibj", F, F, jnp.eye(3)).reshape(Me.shape)
        assert np.abs(FF - Me).max() <= 1e-12 * np.abs(Me).max()
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


def test_tail_bias_quad():
    """Test 3: tail_bias against scipy quad of q(x(lam))^2 dN/dlam, the tail near the step."""
    from scipy.integrate import quad

    d, lam_top, n_top, n_free = 40, 1.25, 5000, 10**6
    gamma = jackson_step(d)
    a = counting.WEYL_A
    lam_max = lam_top * (n_free / n_top) ** (1 / a)
    f = lambda lam: (  # noqa: E731
        step_value(gamma, _theta(lam))[0] ** 2 * n_top * a * lam ** (a - 1) / lam_top**a
    )
    want = quad(f, lam_top, lam_max, limit=2000, epsabs=0, epsrel=1e-10)[0]
    got = tail_bias(gamma, 0.5, lam_top, n_top, n_free)
    assert abs(got - want) <= 1e-3 * want, (got, want)


def test_degree_cap():
    """Test 3: 23/g > MAX_DEGREE: not run, NaN, unverifiable."""
    _, op, free, _, _, lam, W = ref()
    lam_next = lam[5] * (1 + 1e-4)
    est, info = count(op, free, W[:, :KEEP], lam[:KEEP], lam[5], lam_next)
    g = (lam_next - lam[5]) / ((lam_next + lam[5]) / 2)
    assert np.isnan(est) and not info["verifiable"] and not info["passed"]
    assert info["degree"] == int(np.ceil(23 / g)) > MAX_DEGREE and info["steps"] == 0
    assert info["capped"]
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


def dense_probes(op, free, K, M, W, lam_c, d, n, seed=0, gamma=None):
    """The dense estimator with exact solves on the same probes as count (gamma: the
    coefficients, default the step's)."""
    Kf, Mf = _dense_free(K, M, free)
    keys = jax.vmap(jax.random.fold_in, (None, 0))(jax.random.key(seed), jnp.arange(n))
    g = np.asarray(mass_sample_keys(op, keys))[free]
    Wf = W[free]
    proj = lambda Z: Z - Wf @ (Wf.T @ (Mf @ Z))  # noqa: E731
    cf = sla.cho_factor(Kf + lam_c * Mf)
    X = lambda Z: 2 * lam_c * sla.cho_solve(cf, Mf @ Z) - Z  # noqa: E731
    gamma = jackson_step(d) if gamma is None else gamma
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
    """Test 8: tau one decade below the measured floor, and 1e-17: unverifiable, NaN, early
    stop."""
    floor = None
    for tau in 10.0 ** -np.arange(10, 17):
        _, info, _ = run(exact=False, degree=1, tau=tau)
        if not info["verifiable"]:
            break
        floor = tau
    assert floor is not None, "tau = 1e-10 is already unverifiable"
    print(f"Jacobi floor on the reference mesh: {floor:g}")
    d = run(exact=False)[1]["degree"]
    for tau in (floor / 10, 1e-17):
        est, info, _ = run(exact=False, tau=tau)
        assert np.isnan(est) and not info["verifiable"] and not info["passed"]
        assert np.isnan(info["probes"]).all() and info["steps"] < d


def test_flags(monkeypatch):
    """Test 8: maxit in the recurrence (lumped: y is exact) and in the y solve (exact
    preconditioner: the recurrence needs 1-2 iterations), and a failed y solve are unverifiable;
    a normal run is verifiable, with the default tau rule."""
    est, info, _ = run(mass="lumped", exact=False, maxit=2)
    assert not info["verifiable"] and info["steps"] >= 1 and np.isnan(est)
    est, info, _ = run(maxit=3)
    assert not info["verifiable"] and info["steps"] == 0 and np.isnan(est)
    est, info, _ = run()
    assert info["verifiable"] and info["passed"] and est < 0.5
    n_free = int(ref()[2].sum())
    assert info["tau"] == pytest.approx(0.01 / (info["degree"] * np.sqrt(n_free)), rel=1e-14)
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
    """Test 9a: memory_analysis of the recurrence <= count_memory's step (w0 and u aside), the
    fit is tight, and the multigrid increment over Jacobi is 0.7 (AMG) / 0.45 (GMG) to 1.0 of
    its `count` term."""
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
        assert got <= model and got >= (0.5 if pc == "mg" else 0.6) * model
        if pc == "mg":
            inc = got - _recurrence_temps(op, free, jacobi(op, free, 1.0), b)
            lo = 0.7 if kind == "hex20" else 0.45  # measured 0.92-0.98 / 0.54-0.93
            assert lo <= inc / pcm(b)["count"] <= 1.0, inc / pcm(b)["count"]


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
count(op, free, W, lam, 1.0, 1.5, degree=3, tau=1e-2, block=B, n_probes=3 * B)  # warm-up
del op
"""
PEAK = """
op = stencil(mesh, mats, mass="lumped")
count(op, free, W, lam, 1.0, 1.5, degree=3, tau=1e-2, block=B, n_probes=3 * B)
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_count_peak():
    """Test 9d: at count_memory >= 2 GB the measured process peak of a 3-block count is within
    20%, and W stays on the host (an n x k device copy would add 60%)."""
    from benchmarks import harness

    s, b = 60, 24
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


def test_steps_and_times():
    """PR 14b test 7c: _steps caps the steps per block (same compile), _steps >= d gives the
    count's estimate, and the time keys add up."""
    est, info, _ = run(degree=20, n_probes=6, block=3)
    n0 = counting._recurrence._cache_size()
    _, capped, _ = run(degree=20, n_probes=6, block=3, _steps=3)
    assert capped["steps"] == 2 * 3 and counting._recurrence._cache_size() == n0
    assert run(degree=20, n_probes=6, block=3, _steps=20)[0] == est
    t = info["time"]
    assert t["step"] == t["recurrence"] / info["steps"]
    assert t["block"] == pytest.approx((t["sampling"] + t["projection"]) / 2, rel=1e-12)
    assert t["sampling"] + t["recurrence"] + t["projection"] <= t["total"]


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
    assert out["hex20"]["cuts"] and out["errors"]


def test_errors():
    """Test 13: input checks."""
    _, op, free, _, _, lam, W = ref()
    W14, l14 = W[:, :KEEP], lam[:KEEP]
    bad = W14.copy()
    bad[:, 3] *= 1.01
    R = np.zeros((len(free), 1))
    R[np.flatnonzero(free)[0]] = 1.0  # not M-normalized
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
        (dict(degree=0), "degree"),
        (dict(R=R), "R must be M-orthonormal"),
    ]
    for kw, msg in cases:
        args = dict(W=W14, lam=l14, lam_last=lam[5], lam_next=lam[6]) | kw
        with pytest.raises(ValueError, match=msg):
            count(op, free, args.pop("W"), args.pop("lam"), args.pop("lam_last"),
                  args.pop("lam_next"), **args)  # fmt: skip
    for f in (free[:-1], free.astype(float)):
        with pytest.raises(ValueError, match="free must be"):
            count(op, f, W14, l14, lam[5], lam[6])


def test_chunks(monkeypatch):
    """Tests 5 and 13 with CHUNK = 4: the multi-chunk projector (last chunk padded) against the
    dense estimator, and the sketch catches a perturbed column in a later chunk."""
    monkeypatch.setattr(counting, "CHUNK", 4)
    _, op, free, K, M, lam, W = ref()
    lam_c = (lam[5] + lam[6]) / 2
    P = exact_precond(op, free, K, M, lam_c)
    for miss in ((), (2,)):
        kept = np.setdiff1d(np.arange(KEEP), miss)  # 14 or 13 columns: 4 chunks, last padded
        _, info = count(op, free, W[:, kept], lam[kept], lam[5], lam[6], precond=P, n_probes=10,
                        tau=1e-12)  # fmt: skip
        ref_s = dense_probes(op, free, K, M, W[:, kept], lam_c, info["degree"], 10)[0]
        err = np.abs(info["probes"] - ref_s)
        assert (err <= (1e-8 * ref_s if miss else 1e-10)).all(), err.max()
    bad = W[:, :KEEP].copy()
    bad[:, 9] *= 1.01
    with pytest.raises(ValueError, match="M-orthonormal"):
        count(op, free, bad, lam[:KEEP], lam[5], lam[6])


# ---------------------------------------------------------------- PR 15b: band-pass count

BAND = (13, 19)  # modes 13-18 (0-based) of ref(): edge gaps 0.09 below and 0.16 above
MID = 15  # an interior mode, not next to an edge


def _jackson_damping(d):
    a = np.pi / (d + 1)
    j = np.arange(d + 1)
    return ((d - j + 1) * np.cos(j * a) + np.sin(j * a) / np.tan(a)) / (d + 1)


def band_case(bcs="clamped", band=BAND):
    """(op, free, K, M, lam, W, lo, hi, out): the band between the gap midpoints around modes
    band[0] .. band[1] - 1, out = the two neighbours."""
    _, op, free, K, M, lam, W = ref(bcs=bcs)
    i0, i1 = band
    lo, hi = (lam[i0 - 1] + lam[i0]) / 2, (lam[i1 - 1] + lam[i1]) / 2
    return op, free, K, M, lam, W, lo, hi, np.array([i0 - 1, i1])


def band_expected(lam, deflated, info):
    """sum over modes not deflated of q(lam_i)^2 (direct cosine sum)."""
    c_hi = info["edges"][1]
    q = step_value(_band_gamma(info), _theta(np.maximum(np.delete(lam, deflated), 0) / c_hi))
    return float((q**2).sum())


def _band_gamma(info):
    c_lo, c_hi = info["edges"]
    a, d = c_lo / c_hi, info["degree"]
    return jackson_step(d) - jackson_step(d, (1 - a) / (1 + a))  # independent of _gamma


def band_run(miss=(), bcs="clamped", band=BAND, R=None, **kw):
    """band_count on band_case with the in-band modes minus miss, exact preconditioner at c_hi;
    (estimate, info, the exact expectation)."""
    op, free, K, M, lam, W, lo, hi, out = band_case(bcs, band)
    kept = np.setdiff1d(np.arange(*band), miss)
    c_hi = band_edges(lam[kept], lo, hi, lam[out])[1]
    P = exact_precond(op, free, K, M, c_hi)
    est, info = band_count(op, free, W[:, kept], lam[kept], lo, hi, W[:, out], lam[out], R=R,
                           precond=P, **kw)  # fmt: skip
    rig = np.arange(6) if R is not None else np.zeros(0, int)
    return est, info, band_expected(lam, np.r_[kept, out, rig], info)


def test_band_steps():
    """PR 15d test 14: band_count's _steps caps the steps per block (same compile), and _steps
    >= d gives the full count's estimate (the calibration of a leaf with lo > 0)."""
    est = band_run(degree=20, n_probes=6, block=3)[0]
    n0 = counting._recurrence._cache_size()
    capped = band_run(degree=20, n_probes=6, block=3, _steps=3)[1]
    assert capped["steps"] == 2 * 3 and counting._recurrence._cache_size() == n0
    assert band_run(degree=20, n_probes=6, block=3, _steps=20)[0] == est


def test_ortho_check_skipped(monkeypatch):
    """Predictions (_predict) and calibrations (_steps) skip the W orthonormality check; the
    full count and band_count run it."""
    calls = []
    real = counting._check_ortho
    monkeypatch.setattr(counting, "_check_ortho", lambda *a: calls.append(1) or real(*a))
    for f in (run, band_run):
        calls.clear()
        f(degree=20, n_probes=6, block=3, _steps=3)
        f(degree=20, n_probes=6, block=3, _predict=True)
        assert not calls
        f(degree=20, n_probes=6, block=3)
        assert calls == [1]


def test_band_polynomial():
    """15b test 1: jackson_step(d, 0) is 14a's; the coefficients of x >= x0 against quad of
    the indicator's Chebyshev projection; the band-pass grid stays in [0, 1]."""
    from scipy.integrate import quad

    d = 40
    k = np.arange(1, d + 1)
    old = np.r_[0.5, 2 * np.sin(k * np.pi / 2) / (k * np.pi)] * _jackson_damping(d)
    assert np.abs(jackson_step(d) - old).max() <= 1e-15
    for x0 in (-0.6, 0.2, 0.7):
        t = np.arccos(x0)
        cq = [quad(lambda th, j=j: np.cos(j * th), 0, t)[0] * (1 if j == 0 else 2) / np.pi
              for j in range(8)]  # fmt: skip
        got = jackson_step(d, x0)[:8] / _jackson_damping(d)[:8]
        assert np.abs(got - cq).max() <= 1e-12
    gamma = counting._gamma(57, (0.4, 0.3))
    th, q = _grid(gamma)
    assert q.min() >= -1e-14 and q.max() <= 1 + 1e-14


def _band_lower(a, g_lo):
    return a * (1 + g_lo / 2), a * (1 - g_lo / 2)


@pytest.mark.parametrize("a", [0.25, 0.5, 0.9])
def test_band_degree(a):
    """15b test 2: q^2 >= Q_MIN at both in-band ends for d, d - 1 fails; d g_hi in [19, 24]
    with the upper edge tighter, d g_lo 2 sqrt(a) / (1 + a) in [19, 24] with the lower one; a
    dense spectrum makes the bias set d."""
    for g_lo, g_hi in ((0.06, 0.02), (0.02, 0.06)):
        lower = _band_lower(a, g_lo)
        d, bias = count_degree(g_hi, 1 + g_hi / 2, 50, 10**6, lower)
        ends = _theta(np.array([lower[0], 1 - g_hi / 2]))
        assert (step_value(counting._gamma(d, lower), ends) ** 2 >= Q_MIN).all()
        assert not counting._passes(d - 1, g_hi, 1 + g_hi / 2, 50, 10**6, lower)[0]
        x = d * g_hi if g_hi < g_lo else d * g_lo * 2 * np.sqrt(a) / (1 + a)
        print(f"a {a}, g_lo {g_lo}, g_hi {g_hi}: d {d}, normalized {x:.1f}, bias {bias:.1e}")
        assert 19 <= x <= 24 and bias <= 0.05
    lower = _band_lower(a, 0.1)
    d_in = count_degree(0.1, 1.05, 50, 10**7, lower)[0]
    d, bias = count_degree(0.1, 1.05, 500_000, 10**7, lower)  # the tail bias sets d
    assert d > d_in and bias <= 0.05
    assert not counting._passes(d - 1, 0.1, 1.05, 500_000, 10**7, lower)[0]
    gamma = jackson_step(d) - jackson_step(d, (1 - a) / (1 + a))
    lo_tail = counting._bias(gamma, 0.0, lower[1], 500_000, 1.05)
    assert lo_tail > 0 and bias == pytest.approx(  # the lower tail is part of the bias
        tail_bias(gamma, 0.1, 1.05, 500_000, 10**7) + lo_tail, rel=1e-12)  # fmt: skip


def test_band_lower_tail():
    """15b test 3: the lower tail against scipy quad of q^2 dN/dlam over (0, lam_bot)."""
    from scipy.integrate import quad

    a = counting.WEYL_A
    lower, top, n_top = _band_lower(0.5, 0.2), 1.1, 400
    gamma = counting._gamma(60, lower)
    f = lambda lam: step_value(gamma, _theta(lam))[0] ** 2 * n_top * a * lam ** (a - 1) / top**a  # noqa: E731
    want = quad(f, 0, lower[1], limit=2000, epsabs=0, epsrel=1e-10)[0]
    got = counting._bias(gamma, 0.0, lower[1], n_top, top)
    assert abs(got - want) <= 1e-3 * want, (got, want)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["missing", "free"])
def test_band_expected(case):
    """15b test 4: the mean of 2000 probes against the exact sum over modes not deflated, within
    5 standard errors: an interior mode missing, and a free body with R (band above the rigid
    modes, elastic modes 8-16)."""
    if case == "missing":
        est, info, exp = band_run((MID,), n_probes=2000, block=500, tau=1e-10)
    else:
        mesh = ref(bcs="free")[0]
        M = ref(bcs="free")[4]
        Rn = rigid_modes(mesh)
        L = np.linalg.cholesky(Rn.T @ (M @ Rn))
        est, info, exp = band_run(bcs="free", band=(14, 23), R=np.linalg.solve(L, Rn.T).T,
                                  n_probes=2000, block=500, tau=1e-10)  # fmt: skip
    se = info["std"] / np.sqrt(2000)
    print(f"{case}: {est:.4g} vs {exp:.4g} (se {se:.2g}), d {info['degree']}")
    assert info["verifiable"] and abs(est - exp) <= 5 * se


def test_band_against_dense():
    """15b test 5: band_count's s_j against the dense band-pass estimator on the same probes."""
    op, free, K, M, lam, W, lo, hi, out = band_case()
    for miss in ((), (MID,)):
        _, info, _ = band_run(miss, n_probes=20, tau=1e-12)
        kept = np.r_[np.setdiff1d(np.arange(*BAND), miss), out]
        ref_s = dense_probes(op, free, K, M, W[:, kept], info["lam_c"], info["degree"], 20,
                             gamma=_band_gamma(info))[0]  # fmt: skip
        err = np.abs(info["probes"] - ref_s)
        assert (err <= (1e-8 * ref_s if miss else 1e-10)).all(), err.max()


@pytest.mark.slow
@pytest.mark.parametrize("miss", [(MID,), ()], ids=["missing", "complete"])
def test_band_detection(miss):
    """15b test 6 (roadmap): 100 sets of 50 probes (one 5000-probe count): the interior mode
    left out is detected in >= 97 sets; with V complete every set is < 0.5."""
    _, info, _ = band_run(miss, n_probes=5000, block=500)
    sets = info["probes"].reshape(100, 50).mean(1)
    print(f"miss={miss}: {(sets >= 0.5).sum()} of 100 sets >= 0.5")
    if miss:
        assert (sets >= 0.5).sum() >= 97
    else:
        assert (sets < 0.5).all()


def test_band_planted():
    """15b test 7 (roadmap, the count part; si_checked's empty band is in test_shift_invert):
    the band in the middle third of the largest gap (lam_2, lam_3); the density scaled by s =
    lam_3 / centre plants mode 3 at the band centre, its neighbours (scaled modes 2 and 4)
    stay outside; band_count with V empty gives >= 0.5, and < 0.5 on the unscaled model."""
    mesh, op, free, K, M, lam, W = ref()
    g = np.diff(lam[:30]) / lam[1:30]
    j = int(np.argmax(g))
    assert j == 2
    lo, hi = lam[j] + (lam[j + 1] - lam[j]) / 3, lam[j] + 2 * (lam[j + 1] - lam[j]) / 3
    none = np.zeros((op.n_dof, 0))
    out = np.array([j, j + 1])
    c_hi = band_edges([], lo, hi, lam[out])[1]
    est, info = band_count(op, free, none, [], lo, hi, W[:, out], lam[out],
                           precond=exact_precond(op, free, K, M, c_hi))  # fmt: skip
    assert info["verifiable"] and est < 0.5
    s = lam[j + 1] / ((lo + hi) / 2)
    mat = {0: Material(isotropic(3.0, 0.3), 2.0 * s)}
    ops = unstructured(mesh, mat)
    ls, Ws = lam / s, W / np.sqrt(s)
    assert ls[j] < lo and ls[j + 2] > hi and lo < ls[j + 1] < hi
    out = np.array([j, j + 2])
    c_hi = band_edges([], lo, hi, ls[out])[1]
    est, info = band_count(ops, free, none, [], lo, hi, Ws[:, out], ls[out],
                           precond=exact_precond(ops, free, K, s * M, c_hi))  # fmt: skip
    print(f"planted: {est:.3g}, d {info['degree']}")
    assert info["verifiable"] and est >= 0.5


def test_band_neighbours():
    """15b test 8 (roadmap): V complete, tau_c = 1e-12: the estimate is < 0.5; without the
    neighbours (same gamma, d, lam_c, probes) it rises by the probe mean of
    sum_i q(mu_i)^2 (v_i^T M y_j)^2 to 1e-8 relative."""
    op, free, K, M, lam, W, lo, hi, out = band_case()
    est, info, _ = band_run(tau=1e-12)
    assert info["verifiable"] and est < 0.5
    gamma, d, c_hi = _band_gamma(info), info["degree"], info["lam_c"]
    m = jnp.asarray(free, float)[:, None]
    R0 = jnp.zeros((op.n_dof, 0))
    P = exact_precond(op, free, K, M, c_hi)
    off, _ = counting._estimate(op, free, m, (W[:, np.arange(*BAND)],), R0, R0, P, gamma, d,
                                c_hi, 1e-12, n_probes=50, block=50, seed=0, maxit=1000,
                                verbose=False, _steps=None, t0=0.0)  # fmt: skip
    keys = jax.vmap(jax.random.fold_in, (None, 0))(jax.random.key(0), jnp.arange(50))
    g = np.asarray(mass_sample_keys(op, keys))[free]  # M y_j = g_j on the free DOFs
    q = step_value(gamma, _theta(lam[out] / c_hi))
    pred = ((q[:, None] * (W[free][:, out].T @ g)) ** 2).sum(0).mean()
    print(f"rise {off - est:.6e}, predicted {pred:.6e}")
    assert abs((off - est) - pred) <= 1e-8 * pred


def test_band_info_and_errors():
    """15b test 12: band_edges cases, info fields, n_top, count with a tuple W, input checks."""
    op, free, K, M, lam, W, lo, hi, out = band_case()
    kept = np.arange(*BAND)
    # band_edges: in-band modes, an empty band, a nearer out-of-band value
    c_lo, c_hi, g_lo, g_hi = band_edges(lam[kept], lo, hi, lam[out])
    assert c_lo == (lam[12] + lam[13]) / 2 and c_hi == (lam[18] + lam[19]) / 2
    assert g_lo == pytest.approx((lam[13] - lam[12]) / c_lo, rel=1e-14)
    assert band_edges([], lo, hi, lam[out])[:2] == ((lam[12] + lo) / 2, (lam[19] + hi) / 2)
    assert band_edges(lam[kept], lo, hi, lam[[12, 20, 19]])[1] == c_hi
    est, info, _ = band_run(n_probes=4, degree=30)
    assert info["edges"] == (c_lo, c_hi) and info["gaps"] == (g_lo, g_hi)
    a = c_lo / c_hi
    assert info["lam_c"] == c_hi and info["degree"] == 30 and not info["capped"]
    assert info["gap"] == min(g_hi, g_lo * 2 * np.sqrt(a) / (1 + a))
    top, bot, w = lam[19] / c_hi, lam[12] / c_hi, counting.WEYL_A
    assert info["n_top"] == pytest.approx(7 / (top**w - bot**w) * top**w, rel=1e-14)
    gamma, n_free = _band_gamma(info), int(band_case()[1].sum())  # the lower tail at given d
    assert info["bias"] == pytest.approx(tail_bias(gamma, g_hi, top, info["n_top"], n_free)
                                         + counting._bias(gamma, 0.0, bot, info["n_top"], top),
                                         rel=1e-12)  # fmt: skip
    # count with a tuple W equals its concatenation
    lc = (lam[5] + lam[6]) / 2
    P = exact_precond(op, free, K, M, lc)
    one = count(op, free, W[:, :KEEP], lam[:KEEP], lam[5], lam[6], precond=P, n_probes=4)
    two = count(op, free, (W[:, :5], W[:, 5:KEEP]), lam[:KEEP], lam[5], lam[6], precond=P,
                n_probes=4)  # fmt: skip
    assert np.abs(one[1]["probes"] - two[1]["probes"]).max() <= 1e-12 * one[0]
    with pytest.raises(ValueError, match="W must have shape"):
        count(op, free, (W[:, :5], W[:, 5:9]), lam[:KEEP], lam[5], lam[6])
    cases = [
        (dict(lam_out=lam[[12]], V_out=W[:, [12]]), "below and above"),
        (dict(lo=0.0), "0 < lo < hi"),
        (dict(lam_out=lam[[12, 15]], V_out=W[:, [12, 15]]), "inside the in-band"),
        (dict(V=W[:, kept[:-1]]), "V must have shape"),
        (dict(V_out=W[:, [12]]), "V_out must have shape"),
        (dict(n_probes=0), "n_probes"),
    ]
    for kw, msg in cases:
        args = dict(V=W[:, kept], lam=lam[kept], lo=lo, hi=hi, V_out=W[:, out],
                    lam_out=lam[out]) | kw  # fmt: skip
        with pytest.raises(ValueError, match=msg):
            band_count(op, free, args.pop("V"), args.pop("lam"), args.pop("lo"), args.pop("hi"),
                       args.pop("V_out"), args.pop("lam_out"), **args)  # fmt: skip


def test_verbose(monkeypatch, capsys):
    """15b test 11: verbose count lines (PRINT_S = 0: one per step and one per block), the same
    estimate and probes as without verbose; quiet at the default PRINT_S on this mesh."""
    est, info, _ = run(n_probes=4, degree=20, verbose=True)
    lines = capsys.readouterr().out.splitlines()
    assert sum(ln.startswith("count: block") for ln in lines) == 1
    assert not any(ln.startswith("count: step") for ln in lines)
    monkeypatch.setattr(counting, "PRINT_S", 0.0)
    est_v, info_v, _ = run(n_probes=4, degree=20, verbose=True)
    lines = capsys.readouterr().out.splitlines()
    assert sum(ln.startswith("count: step") for ln in lines) == 20
    assert lines[-1].startswith("count: block 1 of 1, 20 steps, done")
    q, _, _ = run(n_probes=4, degree=20)
    assert abs(est_v - q) <= 1e-12 * q and abs(est - q) <= 1e-12 * q
