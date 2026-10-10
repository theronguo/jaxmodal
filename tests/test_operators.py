"""DOF map, assembled oracle, BC masks and A_P against dense references."""

import jax
import numpy as np
import pytest
import scipy.sparse as sp
from helpers import MATERIALS, ORTHO, R, dense_reference, distort, mixed_mesh, rel
from scipy.sparse.linalg import spsolve

from jaxmodal.elements import bmatrix, gradients, jacobian
from jaxmodal.materials import Material, isotropic, rotate
from jaxmodal.materials.rotation import VOIGT
from jaxmodal.mesh import Mesh, box_mesh
from jaxmodal.operators import a_p, assemble, check_int32, dof_mask, dofmap, element_dofs, masked

TYPES = ["hex8", "hex20", "hex27"]
FACES = [f"{a}{s}" for a in "xyz" for s in ("min", "max")]
DISTORTIONS = [(t, False) for t in TYPES] + [("hex20", True), ("hex27", True)]


def nnz_diff(A, B):
    return (sp.csr_array(A) != sp.csr_array(B)).nnz


# ---------------------------------------------------------------- 2: DOF map


def test_element_dofs(monkeypatch):
    m = mixed_mesh(np.random.default_rng(0))
    d = element_dofs(m)
    assert list(d) == ["hex8", "hex20"]
    for t, c in m.blocks.items():
        assert d[t].dtype == np.int32
        assert np.array_equal(d[t], [[3 * n + i for n in row for i in range(3)] for row in c])
    calls = []
    monkeypatch.setattr(dofmap, "check_int32", lambda **k: calls.append(k))
    element_dofs(m)
    assert calls == [{"n_dof": 3 * len(m.points), "hex8_dofs": 8 * 24, "hex20_dofs": 8 * 60}]


def test_check_int32():
    check_int32(n_dof=2**31 - 1)
    for sizes, bad in [
        ({"n_dof": 2**31}, ["n_dof"]),
        ({"n_dof": 2**31 + 1}, ["n_dof"]),
        ({"n_dof": 5, "hex20_dofs": 2**31, "hex8_dofs": 2**32}, ["hex20_dofs", "hex8_dofs"]),
    ]:
        with pytest.raises(ValueError) as err:
            check_int32(**sizes)
        msg = str(err.value)
        assert all(f"{k} = {sizes[k]}" in msg for k in bad)
        assert all(k not in msg for k in sizes.keys() - set(bad))


# ---------------------------------------------------------------- 3: assembly


@pytest.mark.parametrize("mass", ["consistent", "lumped"])
def test_assembly_matches_dense_loop(mass):
    m = mixed_mesh(np.random.default_rng(1))
    Kref, Mref = dense_reference(m, mass == "lumped")
    for chunk in (1, 7, 4096):  # one element per chunk; padded last chunk; one chunk > n_el
        K, M = assemble(m, MATERIALS, {1: R}, mass=mass, chunk=chunk)
        assert rel(K.toarray(), Kref) < 1e-14 and rel(M.toarray(), Mref) < 1e-14
        for A in (K, M):
            assert nnz_diff(A, A.T) == 0 and A.indices.dtype == A.indptr.dtype == np.int32


# ---------------------------------------------------------------- 4: patch test


def element_stresses(m, t, u, C):
    X = m.points[m.blocks[t]]
    invJ, _ = jacobian(t, X)
    B = bmatrix(gradients(t, invJ))
    eps = np.einsum("eqva,ea->eqv", B, u[element_dofs(m)[t]])
    return eps @ np.asarray(C).T


@pytest.mark.parametrize("t,curved", DISTORTIONS)
def test_patch(t, curved):
    rng = np.random.default_rng(2)
    m = distort(box_mesh((3, 3, 3), type=t), curved, rng)
    C = rotate(ORTHO, R)
    K, _ = assemble(m, {0: Material(ORTHO, 1.0)}, {0: R})
    A, b = rng.standard_normal((3, 3)), rng.standard_normal(3)
    free = dof_mask(m, {f: "xyz" for f in FACES})
    u = (m.points @ A.T + b).ravel()
    u[free] = spsolve(K[free][:, free].tocsc(), -K[free][:, ~free] @ u[~free])
    sigma = element_stresses(m, t, u, C)
    eps = np.array([A[i, j] + A[j, i] if i != j else A[i, i] for i, j in VOIGT])
    expect = np.asarray(C) @ eps
    assert rel(sigma, expect) < 1e-10


# ---------------------------------------------------------------- 5-6: masks and A_P


def clamped():
    m = distort(box_mesh((3, 2, 2), type="hex20"), True, np.random.default_rng(3))
    K, M = assemble(m, MATERIALS)
    free = ~np.repeat(m.points[:, 0] == 0, 3)
    return m, K, M, free


def test_masked_clamped():
    m, K, M, free = clamped()
    mask = dof_mask(m, {"xmin": "xyz"})
    assert np.array_equal(mask, free) and (~free).sum() > 0
    for A in (K, M):
        Am = masked(A, mask)
        assert np.abs(Am[~free]).max() == 0 and np.abs(Am[:, ~free]).max() == 0
        assert nnz_diff(Am[free][:, free], A[free][:, free]) == 0


def test_a_p():
    m, K, M, free = clamped()
    sigma = 0.1 * np.mean(K.diagonal() / M.diagonal())
    A = K + sigma * M
    Ap = a_p(A, free)
    dense = Ap.toarray()
    np.linalg.cholesky(dense)
    assert np.linalg.eigvalsh(dense).min() > 0 and (dense.diagonal() > 0).all()
    s = A.diagonal()[free].mean()
    assert (dense.diagonal()[~free] == s).all()
    assert nnz_diff(dense[free][:, free], A[free][:, free]) == 0
    assert nnz_diff(a_p(A * 2.0**20, free), Ap * 2.0**20) == 0  # unit independence, exact
    assert nnz_diff(a_p(A, np.ones_like(free)), A) == 0
    assert (a_p(A, free, s=3.0).diagonal()[~free] == 3.0).all()
    with pytest.raises(ValueError, match="no free DOF"):
        a_p(A, np.zeros_like(free))
    for neg in (-A, A - sp.diags_array(A.diagonal())):  # s < 0 and s = 0
        with pytest.raises(ValueError, match="<= 0"):
            a_p(neg, free)
    for bad in (free.astype(int), free[:-1], list(free)):
        with pytest.raises(ValueError, match="bool"):
            a_p(A, bad)
        with pytest.raises(ValueError, match="bool"):
            masked(A, bad)


# ---------------------------------------------------------------- 7-8: masses


@pytest.mark.parametrize("mass", ["consistent", "lumped"])
def test_nodal_mass(mass):
    m = mixed_mesh(np.random.default_rng(4))
    mn = np.random.default_rng(5).uniform(0, 2, len(m.points))
    mn[::2] = 0  # most nodes carry no point mass
    _, M0 = assemble(m, MATERIALS, {1: R}, mass=mass)
    _, M1 = assemble(m, MATERIALS, {1: R}, mass=mass, nodal_mass=mn)
    assert nnz_diff(M1, M0 + sp.diags_array(np.repeat(mn, 3))) == 0
    for bad in (-mn - 1, np.where(mn > 1, np.nan, mn), np.where(mn > 1, np.inf, mn), mn[:-1]):
        with pytest.raises(ValueError, match="nodal_mass"):
            assemble(m, MATERIALS, {1: R}, mass=mass, nodal_mass=bad)


@pytest.mark.parametrize("mass", ["consistent", "lumped"])
@pytest.mark.parametrize("t,curved", DISTORTIONS)
def test_rigid_translation_mass(t, curved, mass):
    rng = np.random.default_rng(6)
    m = distort(box_mesh((3, 3, 3), type=t), curved, rng)
    mn = rng.uniform(0, 1, len(m.points))
    _, M = assemble(m, {0: Material(isotropic(1.0, 0.3), 1.7)}, mass=mass, nodal_mass=mn)
    tr = np.eye(3)[:, None, :].repeat(len(m.points), 1).reshape(3, -1)  # t_x, t_y, t_z
    expect = 1.7 * 27 + mn.sum()  # boundary nodes fixed: V = 3^3 exactly
    for d in range(3):
        assert abs(tr[d] @ M @ tr[d] - expect) / expect < 1e-12
    assert tr[0] @ M @ tr[1] == tr[0] @ M @ tr[2] == tr[1] @ M @ tr[2] == 0


# ---------------------------------------------------------------- 9: BC masks


def test_dof_mask():
    m = box_mesh((2, 3, 1), (0.5, 0.25, 1.0), type="hex27")
    x, y, _ = m.points.T
    bcs = {"xmin": "xyz", "ymin": "y", "xmax": "zx"}
    expect = np.ones((len(x), 3), bool)
    expect[x == 0] = False
    expect[y == 0, 1] = False
    expect[x == 1, 0] = expect[x == 1, 2] = False
    assert np.array_equal(dof_mask(m, bcs), expect.ravel())
    assert dof_mask(m, {}).all() and dof_mask(m, {}).dtype == bool
    with pytest.raises(ValueError, match="unknown node set"):
        dof_mask(m, {"clamp": "x"})
    for comps in ("", "xw", "xx", ["x"]):
        with pytest.raises(ValueError, match="distinct letters"):
            dof_mask(m, {"xmin": comps})


# ---------------------------------------------------------------- 10: host checks


def test_assemble_host_checks():
    m = box_mesh((1, 1, 1), type="hex27", origin=(-0.5, -0.5, -0.5))  # PR 3's counterexample
    X = m.points.copy()
    edge01 = m.blocks["hex27"][0, 8]  # midside node of edge 0-1, slid 45% toward corner 0
    X[edge01, 0] -= 0.45
    bad = Mesh(X, m.blocks, m.node_sets, m.element_sets, m.material, m.orientation)
    assemble(bad, MATERIALS)  # consistent mass is fine on this mesh; only lumped must raise
    with pytest.raises(ValueError, match="row-sum lumped nodal mass"):
        assemble(bad, MATERIALS, mass="lumped")
    with pytest.raises(ValueError, match="chunk"):
        assemble(bad, MATERIALS, chunk=0)
    with pytest.raises(ValueError, match="material label 0"):
        assemble(bad, {1: MATERIALS[1]})
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="float64"):
            assemble(bad, MATERIALS)
    finally:
        jax.config.update("jax_enable_x64", True)
