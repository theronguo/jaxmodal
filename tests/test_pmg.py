"""Unstructured p-MG against an independent transfer, SciPy Galerkin products, a dense V-cycle
reference and PCG/LOBPCG iteration bounds."""

import functools
import gc
import importlib
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spla
from helpers import MATERIALS, R, distort, mixed_mesh, rel, rhs

from jaxmodal.eigen import lobpcg, reference_modes
from jaxmodal.linalg import pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.mesh.model import REFERENCE_NODES
from jaxmodal.operators import a_p, ap_operator, assemble, dof_mask, restrict, stencil, unstructured
from jaxmodal.precond import GMG, SingularShiftError, block_jacobi, chebyshev, lmax_estimate, pmg

PM = importlib.import_module("jaxmodal.precond.pmg")
mf = importlib.import_module("jaxmodal.operators.matfree")
krylov = importlib.import_module("jaxmodal.linalg.krylov")

STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
ORIENT = {1: R}  # mixed_mesh orientation labels


@pytest.fixture(autouse=True)
def _clear():
    """HostLU factors live in the jit caches until cleared (docs/preconditioners.md)."""
    yield
    jax.clear_caches()


def p_ref(mesh):
    """Dense P (n_dof, 3 n_c): an element loop over prod(1 + xi_a xi_c) / 8 at the reference
    nodes; coarse nodes are the corner nodes in increasing id."""
    corner = np.unique(np.concatenate([c[:, :8].ravel() for c in mesh.blocks.values()]))
    cid = {int(n): i for i, n in enumerate(corner)}
    P = np.zeros((len(mesh.points), len(corner)))
    for conn in mesh.blocks.values():
        xi = REFERENCE_NODES[: conn.shape[1]]
        for row in conn:
            for a, na in enumerate(row):
                for c in range(8):
                    P[na, cid[int(row[c])]] = np.prod(1 + xi[a] * xi[c]) / 8
    return np.kron(P, np.eye(3)), corner


def graded(t):
    rng = np.random.default_rng(0)
    coords = [np.cumsum(np.r_[0, rng.uniform(0.5, 1.5, n)]) for n in (3, 2, 2)]
    return box_mesh((3, 2, 2), coords=coords, type=t)


# ---------------------------------------------------------------- 1-2: transfer and Galerkin


@pytest.mark.parametrize("t", ["hex20", "hex27"])
def test_transfers(t):
    """Test 1: trilinear reproduction on a graded box, restriction = P_ref^T, identity corners."""
    mesh = graded(t)
    _, T, _ = PM._tables(unstructured(mesh, MATERIALS))
    Pr, corner = p_ref(mesh)
    a = np.random.default_rng(1).standard_normal((8, 3))
    x, y, z = mesh.points.T
    f = np.stack([np.ones_like(x), x, y, z, x * y, y * z, z * x, x * y * z], 1) @ a  # (n, 3)
    got = np.asarray(T(jnp.asarray(f[corner].reshape(-1, 1)))).reshape(-1, 3)
    assert rel(got, f) <= 1e-12
    r = np.random.default_rng(2).standard_normal((3 * len(mesh.points), 2))
    assert rel(T.restrict(jnp.asarray(r)), Pr.T @ r) <= 1e-14
    assert rel(T(jnp.eye(Pr.shape[1])), Pr) <= 1e-14
    assert (Pr[3 * corner][:, 0::3] == np.eye(len(corner))).all()
    W = PM._w8(mesh.blocks[t].shape[1])
    assert sorted(W[8][W[8] > 0]) == [0.5, 0.5]  # edge node 0-1
    if t == "hex27":
        assert sorted(W[20][W[20] > 0]) == [0.25] * 4 and (W[26] == 0.125).all()


def test_transfers_mixed():
    """Test 1: hex8 nodes of a hex8 + hex20 mesh get w = 1 identity rows; restriction = P^T."""
    mesh = mixed_mesh(np.random.default_rng(3))
    _, T, _ = PM._tables(unstructured(mesh, MATERIALS, ORIENT))
    Pr, corner = p_ref(mesh)
    h8 = np.unique(mesh.blocks["hex8"])
    assert (np.asarray(T.weight)[h8, 0] == 1).all() and (np.asarray(T.weight)[h8, 1:] == 0).all()
    r = np.random.default_rng(2).standard_normal((3 * len(mesh.points), 2))
    assert rel(T.restrict(jnp.asarray(r)), Pr.T @ r) <= 1e-14
    assert rel(T(jnp.eye(Pr.shape[1])), Pr) <= 1e-14


def galerkin_case(name, mode="stored", mass="consistent", chunk=None):
    """(mesh, op, free, sigma, A_P SciPy): unit materials, random s_K, s_M, nodal masses on half
    the nodes, a mixed clamped/sliding mask."""
    rng = np.random.default_rng(5)
    if name == "mixed":
        mesh = mixed_mesh(rng)
    elif name == "chunked":
        mesh = box_mesh((5, 4, 3), type="hex20")
    else:
        mesh = graded(name)
    nodal = rng.uniform(0, 1, len(mesh.points)) * (rng.random(len(mesh.points)) < 0.5)
    sK, sM = rng.uniform(0.5, 2, mesh.n_el), rng.uniform(0.5, 2, mesh.n_el)
    if name == "mixed":
        free = rng.random(3 * len(mesh.points)) < 0.8
    else:
        free = dof_mask(mesh, {"xmin": "xyz", "ymin": "y"})
    o = ORIENT if name == "mixed" else None
    op = unstructured(mesh, MATERIALS, o, mode=mode, mass=mass, nodal_mass=nodal, chunk=chunk)
    op = op.scaled(s_K=jnp.asarray(sK), s_M=jnp.asarray(sM))
    sigma = 0.37
    K, M = assemble(mesh, MATERIALS, o, mass=mass, nodal_mass=nodal, s_K=sK, s_M=sM)
    s = float(ap_operator(op, free, sigma).s)
    return mesh, op, free, sigma, a_p(K + sigma * M, free, s)


@pytest.mark.parametrize("mass", ["consistent", "lumped"])
@pytest.mark.parametrize("mode", ["stored", "otf"])
@pytest.mark.parametrize("name", ["hex20", "hex27", "mixed"])
def test_galerkin(name, mode, mass):
    """Test 2: _hex8_galerkin = P_ref^T a_p(K + sigma M, free, s) P_ref, exactly symmetric."""
    mesh, op, free, sigma, A = galerkin_case(name, mode, mass)
    Pr, _ = p_ref(mesh)
    Ac, _ = PM._hex8_galerkin(ap_operator(op, free, sigma))
    assert rel(Ac.toarray(), Pr.T @ A.toarray() @ Pr) <= 1e-12
    assert (Ac != Ac.T).nnz == 0


def test_galerkin_pattern():
    """Stored and otf give the same structural pattern (no pruned cancellations)."""
    _, op, free, sigma, _ = galerkin_case("hex20", "stored")
    _, op2, *_ = galerkin_case("hex20", "otf")
    a, b = (PM._hex8_galerkin(ap_operator(o, free, sigma))[0] for o in (op, op2))
    assert (a.indptr == b.indptr).all() and (a.indices == b.indices).all()


def test_galerkin_padded():
    """Test 2: several chunks with padded elements (60 elements, chunk 7); pads give zeros."""
    mesh, op, free, sigma, A = galerkin_case("chunked", chunk=7)
    b = op.blocks[0]
    assert b.dofs.shape[0] > b.n_el
    Pr, _ = p_ref(mesh)
    Ac, _ = PM._hex8_galerkin(ap_operator(op, free, sigma))
    assert rel(Ac.toarray(), Pr.T @ A.toarray() @ Pr) <= 1e-12
    d = {k: v[-b.chunk :] for k, v in mf._block_data(op, b).items()}
    K, M = mf.element_matrices(op, b, d)
    pads = b.dofs.shape[0] - b.n_el
    assert (np.asarray(K)[-pads:] == 0).all() and (np.asarray(M)[-pads:] == 0).all()
    assert np.abs(np.asarray(K)[:-pads]).max() > 0


def test_galerkin_hex8():
    """Test 2: an all-hex8 mesh gives the assembled hex8 a_p itself (P = I)."""
    mesh = box_mesh((3, 2, 2))
    free = dof_mask(mesh, {"xmin": "xyz"})
    op = unstructured(mesh, MATERIALS)
    K, M = assemble(mesh, MATERIALS)
    A = a_p(K + 0.37 * M, free, float(ap_operator(op, free, 0.37).s))
    assert rel(PM._hex8_galerkin(ap_operator(op, free, 0.37))[0].toarray(), A.toarray()) <= 1e-12


# ---------------------------------------------------------------- 3-4: the cycle


def dense(f, n):
    return np.asarray(f(jnp.eye(n)))


@pytest.mark.parametrize("bcs", [{"xmin": "xyz"}, {"xmin": "x", "ymin": "y", "zmin": "z"}])
def test_vcycle_dense(bcs):
    """Test 3: pmg(degree=3, ratio=20) = the NumPy V-cycle from SciPy a_p, P_ref, an
    independently built smoother and A_c^{-1} = (P_ref^T A P_ref)^{-1}; V is SPD on free DOFs."""
    mesh = box_mesh((3, 2, 2), type="hex20")
    free, sigma = dof_mask(mesh, bcs), 0.37
    op = unstructured(mesh, MATERIALS)
    K, M = assemble(mesh, MATERIALS)
    Ap = ap_operator(op, free, sigma)
    A = a_p(K + sigma * M, free, float(Ap.s)).toarray()
    Pr, _ = p_ref(mesh)
    D = block_jacobi(op, free, sigma)
    n = op.n_dof
    S = dense(chebyshev(Ap, D, lmax_estimate(Ap, D), 3, 20.0), n)
    Ci = np.linalg.inv(Pr.T @ A @ Pr)
    m = free[:, None].astype(float)

    def cycle(b):
        b = m * b
        x = S @ b
        x = x + Pr @ (Ci @ (Pr.T @ (b - A @ x)))
        return m * (x + S @ (b - A @ x))

    U = np.random.default_rng(6).standard_normal((n, 3))
    P = pmg(op, free, sigma, degree=3, ratio=20.0)
    assert rel(P(jnp.asarray(U)), cycle(U)) <= 1e-10
    V = dense(P, n)[np.ix_(free, free)]
    assert rel(V, V.T) <= 1e-12 and np.linalg.eigvalsh((V + V.T) / 2).min() > 0


@pytest.mark.parametrize("bcs", [{"xmin": "xyz"}, {}])
def test_symmetric(bcs):
    """Test 4: u^T V v = v^T V u on unmasked vectors, constrained outputs exactly 0, a GMG."""
    mesh = box_mesh((4, 3, 2), type="hex20")
    free = dof_mask(mesh, bcs)
    op = unstructured(mesh, MATERIALS)
    P = pmg(op, free, 0.37)
    assert type(P) is GMG
    u, v = np.random.default_rng(7).standard_normal((2, op.n_dof, 1))
    Vu, Vv = np.asarray(P(jnp.asarray(u))), np.asarray(P(jnp.asarray(v)))
    assert abs((u * Vv).sum() - (v * Vu).sum()) <= 1e-12 * abs((u * Vv).sum())
    if (~free).any():
        assert (Vu[~free] == 0).all()


def test_hex8_exact():
    """Test 4: on a pure hex8 box P = I and the coarse solve is exact: PCG in <= 2 iterations."""
    mesh = box_mesh((6, 4, 3), (1 / 6,) * 3)
    op, free = unstructured(mesh, STEEL), dof_mask(mesh, {"xmin": "xyz"})
    A, b = ap_operator(op, free, 1e7), jnp.asarray(rhs(free, 2))
    _, info = pcg(A, b, pmg(op, free, 1e7), 1e-8, 50)
    assert np.asarray(info["converged"]).all() and np.asarray(info["iterations"]).max() <= 2


# ---------------------------------------------------------------- 5-7: iterations (steel)


def lam1(mesh, free, **kw):
    """Lowest nonzero eigenvalue by eigsh (fixed v0); kw go to assemble (nodal_mass, s_K)."""
    K, M = assemble(mesh, STEEL, **kw)
    Kf, Mf, _ = restrict(K, M, free)
    lam = spla.eigsh(Kf, 8, M=Mf, sigma=-1.0, v0=np.ones(Kf.shape[0]))[0]
    return float(np.sort(lam[lam > 1e-6 * lam.max()])[0])


def cube(t, n):
    return box_mesh((n,) * 3, (1 / n,) * 3, type=t)


@functools.cache
def sigma6(t):
    """lambda_1 of the clamped 6^3 box: one shift for the whole h-sequence."""
    m = cube(t, 6)
    return lam1(m, dof_mask(m, {"xmin": "xyz"}))


def iterations(mesh, bcs, sigma=None, op=None):
    free = dof_mask(mesh, bcs)
    op = unstructured(mesh, STEEL) if op is None else op
    sigma = lam1(mesh, free) if sigma is None else sigma
    A, b = ap_operator(op, free, sigma), jnp.asarray(rhs(free, 4))
    _, info = pcg(A, b, pmg(op, free, sigma), 1e-8, 1000)
    assert np.asarray(info["converged"]).all()
    return int(np.asarray(info["iterations"]).max())


@functools.cache
def first(t):
    return iterations(cube(t, 3), {"xmin": "xyz"}, sigma6(t))


@pytest.mark.parametrize(
    "t, n",
    [("hex20", 6), ("hex20", 12), pytest.param("hex20", 24, marks=pytest.mark.slow)]
    + [("hex27", 6), ("hex27", 12)],
)
def test_h_independence(t, n):
    """Test 5: iterations grow <= 20% from 3^3 to n^3 at a fixed sigma (measured 14 / 13)."""
    its = iterations(cube(t, n), {"xmin": "xyz"}, sigma6(t))
    assert its <= 1.2 * first(t)


def checkerboard(mesh, op):
    c = np.floor(mesh.points[mesh.blocks["hex20"][:, :8]].mean(1) * 4).astype(int).sum(1) % 2
    return op.scaled(s_K=jnp.asarray(np.where(c == 1, 1e-3, 1.0)))


@pytest.mark.parametrize("case", ["beam", "free", "curved", "nodal", "contrast"])
def test_robust(case):
    """Test 6: <= 30 iterations at each case's own lambda_1 (measured 14-19)."""
    rng = np.random.default_rng(8)
    t, clamp = "hex20", {"xmin": "xyz"}
    if case == "beam":
        its = iterations(box_mesh((20, 2, 2), (0.05,) * 3, type=t), clamp)
    elif case == "free":
        its = iterations(cube(t, 6), {})
    elif case == "curved":
        its = iterations(distort(box_mesh((6,) * 3, type=t), True, rng), clamp)
    elif case == "nodal":
        m = cube(t, 6)
        me = 7850.0 / 6**3
        nodal = me * (rng.random(len(m.points)) < 0.5)
        op = unstructured(m, STEEL, nodal_mass=nodal)
        its = iterations(m, clamp, lam1(m, dof_mask(m, clamp), nodal_mass=nodal), op)
    else:
        m = cube(t, 4)
        op = checkerboard(m, unstructured(m, STEEL))
        its = iterations(m, clamp, lam1(m, dof_mask(m, clamp), s_K=np.asarray(op.s_K)), op)
    assert its <= 30


@pytest.mark.parametrize("t", ["hex20", "hex27"])
def test_lobpcg(t):
    """Test 7: LOBPCG with p-MG on a clamped 6x4x3 box matches reference_modes in <= 40
    iterations (measured about 15)."""
    m = box_mesh((6, 4, 3), (1 / 6,) * 3, type=t)
    op, free = unstructured(m, STEEL), dof_mask(m, {"xmin": "xyz"})
    K, M = assemble(m, STEEL)
    lr = reference_modes(K, M, 12, free, points=m.points)[0]
    lam, V, info = lobpcg(op, free, 6, tol=1e-8, precond=pmg(op, free, lr[0]))
    assert info["converged"] and info["iterations"] <= 40
    assert np.abs(np.asarray(lam) / lr[:6] - 1).max() <= 1e-8
    V, lam = np.asarray(V), np.asarray(lam)
    assert (V[~free] == 0).all()
    KV, MV = K @ V, M @ V
    assert (
        np.linalg.norm((KV - MV * lam)[free], axis=0) / np.linalg.norm(KV[free], axis=0)
    ).max() <= 1e-6


# ---------------------------------------------------------------- 8-9: errors, jit, lifetime


def test_errors():
    """Test 8: a Stencil, sigma < 0 / NaN / 0, x64 off."""
    v = voxel_mesh((2, 2, 2))
    with pytest.raises(TypeError, match="gmg"):
        pmg(stencil(v, MATERIALS), dof_mask(v, {"xmin": "xyz"}), 1.0)
    mesh = box_mesh((2, 2, 2), type="hex20")
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, {"xmin": "xyz"})
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            pmg(op, free, bad)
    with pytest.raises(SingularShiftError):
        pmg(op, free, 0.0)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="float64"):
            pmg(op, free, 1.0)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_jit_and_lifetime():
    """Test 9: a second solve with the same pmg reuses the compiled pcg; after the result is
    deleted and the jit caches are cleared, its host factor is freed (jax 0.11 behaviour:
    the static factor sits in the treedef the caches hold)."""
    mesh = box_mesh((3, 2, 2), type="hex20")
    op, free = unstructured(mesh, MATERIALS), dof_mask(mesh, {"xmin": "xyz"})
    A, b = ap_operator(op, free, 0.37), jnp.asarray(rhs(free, 2))
    P = pmg(op, free, 0.37)
    pcg(A, b, P, 1e-8, 50)
    size = krylov._pcg._cache_size()
    x = pcg(A, b, P, 1e-8, 50)[0]
    assert krylov._pcg._cache_size() == size
    ref = weakref.ref(P.coarse)
    del P, x
    gc.collect()
    jax.clear_caches()
    gc.collect()
    assert ref() is None


def test_study_smoke():
    """The p-MG study on two small cases and a small LOBPCG case, without the large runs."""
    from benchmarks import pmg_study

    cases = [
        ("hex20 3^3", "hex20", (3,) * 3, None, "clamped", "own", 2, 30.0, None),
        ("checker", "hex20", (4,) * 3, None, "clamped", "hex20 3^3", 2, 30.0, "checker"),
    ]
    r = pmg_study.main(cases, [("hex20", (4, 2, 2))], large=False, write=False)
    assert all(all(x["converged"]) for x in r["runs"])
    assert r["lobpcg"][0]["pmg"]["converged"] and r["lobpcg"][0]["jacobi"]["converged"]
