"""PR 16b: the ssh8 stored operator, sampling, estimator, solve_modes and the .inp reader."""

import ast
import dataclasses
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl
from exact import SIMPLY_SUPPORTED
from helpers import R, fd_rel, max_sine, r_M, rel
from test_api import check_lam
from test_counting import _S
from test_shell import _K, _M, C_ANI, C_ORT, I3, laminate, mixed, three_plies

from jaxmodal.eigen import reference_modes, solve_modes
from jaxmodal.materials import Material, isotropic, phases, ply_table
from jaxmodal.memory import BUILD_BYTES, amg_memory, solve_modes_memory, unstructured_memory
from jaxmodal.mesh import box_mesh, read, solid_shell
from jaxmodal.operators import assemble, dof_mask, restrict, unstructured
from jaxmodal.operators import matfree as mf
from jaxmodal.operators.matfree import _block_data, element_matrices
from jaxmodal.operators.sampling import element_factor, xi_shapes
from jaxmodal.precond import amg, pmg

STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
PATHS = ("scatter", "gather")
AMG = "amg does not support ssh8 blocks; solid-shell AMG arrives in PR 18"


def shells():
    """3 x 2 laminated ssh8 elements, two phases (one tilted by R)."""
    box = box_mesh((3, 2, 1), spacing=(1.0, 0.7, 0.1))
    box.orientation[:] = np.arange(box.n_el) % 2
    mesh = solid_shell(box, direction=(0, 0, 1))[0]
    return mesh, {0: laminate(three_plies(C_ORT))}, {1: R}


def mixed_label0():
    """hex8 + ssh8 sharing label 0 (one anisotropic Material)."""
    box = box_mesh((4, 2, 2), spacing=(1.0, 1.0, 0.1))
    sel = np.flatnonzero(box.points[box.blocks["hex8"]].mean(1)[:, 0] > 2)
    return solid_shell(box, sel, direction=(0, 0, 1))[0], {0: Material(C_ANI, 1.3)}, None


def mixed_laminate():
    """hex8 (anisotropic Material) + ssh8 (two laminates): different rho and ply counts."""
    return (*mixed(), None)


CASES = {"shells": shells, "mixed": mixed_label0, "mixed-laminate": mixed_laminate}


def plate():
    """The M4 example: SS square steel plate, 8 x 8 x 1 ssh8, L/t = 20."""
    box = box_mesh((8, 8, 1), spacing=(1 / 8, 1 / 8, 0.05))
    return solid_shell(box, direction=(0, 0, 1))[0]


# ---------------------------------------------------------------- 1: against assemble


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("case", list(CASES))
def test_operator_matches_assemble(case, path):
    mesh, mats, ori = CASES[case]()
    rng = np.random.default_rng(1)
    sK, sM = rng.uniform(0.5, 2, (2, mesh.n_el))
    K, M = (A.toarray() for A in assemble(mesh, mats, ori, s_K=sK, s_M=sM))
    op = unstructured(mesh, mats, ori, path=path, chunk=4).scaled(jnp.asarray(sK), jnp.asarray(sM))
    for m in (1, 3):
        v = rng.standard_normal((op.n_dof, m))
        assert rel(np.asarray(op.apply_K(jnp.asarray(v))), K @ v) <= 1e-12
        assert rel(np.asarray(op.apply_M(jnp.asarray(v))), M @ v) <= 1e-12
    assert rel(np.asarray(op.diag_K()), np.diag(K)) <= 1e-12
    assert rel(np.asarray(op.diag_M()), np.diag(M)) <= 1e-12
    n = np.arange(len(mesh.points))
    blocks = K.reshape(len(n), 3, len(n), 3)[n, :, n, :]  # the 3x3 node blocks of K
    assert rel(np.asarray(op.block_diag_K()), blocks) <= 1e-12
    # element_matrices: per ssh8 element, s_K K_e and s_M m_e (x) I_3 from the kernels
    b = next(b for b in op.blocks if b.type == "ssh8")
    d = _block_data(op, b)
    Ke, Me = (np.asarray(a) for a in element_matrices(op, b, d))
    index, pairs = phases(mesh)
    P, Ro = ply_table(pairs, mats, ori)
    for e, row in enumerate(mesh.blocks["ssh8"]):
        g = b.offset + e
        tab = jax.tree.map(lambda a, p=index[g]: a[p], (P, Ro))
        X = jnp.asarray(mesh.points[row])
        assert rel(Ke[e], sK[g] * np.asarray(_K(X, tab))) <= 1e-12
        assert rel(Me[e], sM[g] * np.kron(np.asarray(_M(X, tab[0])), I3)) <= 1e-12
    assert np.abs(Ke[b.n_el :]).max(initial=0) == 0 and np.abs(Me[b.n_el :]).max(initial=0) == 0


# ---------------------------------------------------------------- 2: gradients through the build


def test_build_gradient():
    mesh, mats, ori = shells()
    index, pairs = phases(mesh)
    P, Ro = ply_table(pairs, mats, ori)
    conn = jnp.asarray(mesh.blocks["ssh8"].reshape(1, -1, 8), jnp.int32)
    phase = jnp.asarray(index.reshape(1, -1), jnp.int32)
    rng = np.random.default_rng(2)
    u, v = rng.standard_normal((2, 3 * len(mesh.points)))
    dofs = jnp.asarray(mf.conn_dofs(mesh.blocks["ssh8"]))

    def f(points, angle):
        Ke = mf._build("ssh8", "stored", points, conn, phase, (P._replace(angle=angle), Ro))[0]
        Kf = Ke[:, mf._packed_index(24)]
        return jnp.sum(u[dofs] * jnp.einsum("eab,eb->ea", Kf, v[dofs]))

    x0, a0 = np.asarray(mesh.points), np.asarray(P.angle)
    gx, ga = jax.grad(f, (0, 1))(jnp.asarray(x0), jnp.asarray(a0))
    fj = jax.jit(f)
    assert fd_rel(lambda x: float(fj(x, a0)), x0, np.asarray(gx)) < 1e-6
    assert fd_rel(lambda a: float(fj(x0, a)), a0, np.asarray(ga)) < 1e-6


# ---------------------------------------------------------------- 3: solve_modes (M4)


def test_example_bcs():
    tree = ast.parse((Path(__file__).parents[1] / "examples" / "m4_shell.py").read_text())
    ss = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", None) == "ss")  # fmt: skip
    assert ast.literal_eval(ss) == SIMPLY_SUPPORTED


@pytest.fixture(scope="module")
def plate_ref():
    """The M4 plate, its K, M, free mask and 12 reference modes (lam, V)."""
    mesh = plate()
    K, M = assemble(mesh, STEEL)
    free = dof_mask(mesh, SIMPLY_SUPPORTED)
    return mesh, K, M, free, *reference_modes(K, M, 12, free, points=mesh.points)


def freq(lam):
    return math.sqrt(lam) / (2 * math.pi)


def check_m4_band(lam, info, lr, lo, hi, tol):
    """The count, the multiplicities and lam to 10 tol (the band tests' rule), verify passed."""
    ref = lr[(lr >= lo) & (lr <= hi)]
    assert info["converged"] and len(lam) == len(ref) > 0
    assert np.abs(lam / ref - 1).max() <= 10 * tol
    assert info["count"]["ran"] and info["count"]["passed"]


def test_m4_bands(plate_ref):
    mesh, *_, lr, _ = plate_ref  # f = 248, 642 (x2), 1023, 1384 (x2), ...
    lo, hi = (lr[0] + lr[1]) / 2, (lr[3] + lr[4]) / 2
    tol = 1e-6
    lam, _, info = solve_modes(mesh, STEEL, SIMPLY_SUPPORTED, band=(0, freq(hi)), tol=tol,
                               maxit=2000, verify=True)  # fmt: skip
    assert info["precond"] == "jacobi" and info["strategy"] == "lobpcg"
    check_m4_band(lam, info, lr, 0, hi, tol)
    lam, _, info = solve_modes(mesh, STEEL, SIMPLY_SUPPORTED, band=(freq(lo), freq(hi)), tol=tol,
                               strategy="shift_invert", verify=True)  # fmt: skip
    assert info["precond"] == "jacobi" and info["strategy"] == "shift_invert"
    check_m4_band(lam, info, lr, lo, hi, tol)


def test_m4_reference(plate_ref):
    """The roadmap's comparison on the example's setup: the PR 9 tolerances (check_lam; principal
    angles by max_sine against r_M, the double modes compared as subspaces)."""
    mesh, K, M, free, lr_all, Vr = plate_ref
    tol = 1e-10
    lam, V, info = solve_modes(mesh, STEEL, SIMPLY_SUPPORTED, 6, tol=tol, maxit=3000)
    assert info["converged"] and info["precond"] == "jacobi"
    Kf, Mf, idx = restrict(K, M, free)
    lr = lr_all[:6]
    Ld = sl.cholesky(Mf.toarray(), lower=True)
    lmax = float(sl.eigh(Kf.toarray(), Mf.toarray(), eigvals_only=True)[-1])
    # here 10 eps lam_max / lam < 1e-8 for all six, so check_lam's floor applies; the subspace
    # check below is the one that depends on tol
    check_lam(lam, lr_all, lmax, tol)
    Vf, Vrf = np.asarray(V)[idx], Vr[idx, :6]
    rm, rm_ref = r_M(Kf, Mf, Ld, lam, Vf), r_M(Kf, Mf, Ld, lr, Vrf)
    for g in ([0], [1, 2], [3], [4, 5]):
        assert rm_ref[g].max() <= 0.1 * rm[g].max()  # the reference is the sharper one
        gap = np.abs(np.delete(lr_all, g)[:, None] - lr[g][None]).min()
        assert max_sine(Mf, Ld, Vf[:, g], Vrf[:, g]) <= 10 * (rm[g] * lam[g]).max() / gap


def test_rho_not_read():
    mesh, mats, ori = mixed_label0()
    op = unstructured(mesh, mats, ori)
    v = jnp.asarray(np.random.default_rng(3).standard_normal((op.n_dof, 2)))

    def touched(t):
        return np.isin(np.arange(op.n_dof), mf.conn_dofs(mesh.blocks[t]))

    shell, hexn = touched("ssh8"), touched("hex8")
    rows = shell & ~hexn  # rows touched by ssh8 elements only
    a, b = op.apply_M(v), dataclasses.replace(op, rho=2 * op.rho).apply_M(v)
    assert rows.any() and np.array_equal(np.asarray(a)[rows], np.asarray(b)[rows])
    assert not np.allclose(np.asarray(a)[hexn & ~shell], np.asarray(b)[hexn & ~shell])


def test_auto_jacobi_mixed():
    mesh, mats, ori = mixed_label0()
    _, _, info = solve_modes(mesh, mats, {"xmin": "xyz"}, 2, ori, tol=1e-4, maxit=2000)
    assert info["precond"] == "jacobi" and info["converged"]


# ---------------------------------------------------------------- 4: refusals


def test_refusals():
    mesh, mats, ori = shells()
    bcs = {"xmin": "xyz"}
    otf = r"stored mode only \(mode='otf'\)"
    with pytest.raises(ValueError, match=otf):
        unstructured(mesh, mats, ori, mode="otf")
    with pytest.raises(ValueError, match=otf):
        unstructured_memory(mesh, 4, mode="otf")
    with pytest.raises(ValueError, match=otf):
        solve_modes(mesh, mats, bcs, 2, ori, operator_options={"mode": "otf"})
    for f in (solve_modes, lambda m, mt, b, n, o, **kw: solve_modes_memory(m, b, n, **kw)):
        with pytest.raises(ValueError, match=AMG):
            f(mesh, mats, bcs, 2, ori, precond="amg")
    with pytest.raises(ValueError, match=AMG):
        amg_memory(mesh, 4)
    op, free = unstructured(mesh, mats, ori), dof_mask(mesh, bcs)
    with pytest.raises(ValueError, match=AMG):
        amg(op, free, 1.0, mesh.points)
    with pytest.raises(ValueError, match=AMG.replace("amg", "pmg", 1)):
        pmg(op, free, 1.0)
    with pytest.raises(ValueError, match="consistent-mass only"):
        solve_modes_memory(mesh, bcs, 2, mass="lumped")
    with pytest.raises(ValueError, match="consistent-mass only"):
        unstructured(mesh, mats, ori, mass="lumped")
    with pytest.raises(ValueError, match="needs a voxel mesh"):
        solve_modes(mesh, mats, bcs, 2, ori, precond="gmg")
    with pytest.raises(ValueError, match="no LOBPCG chunk fits") as err:
        solve_modes(mesh, mats, bcs, None, ori, band=(0, 1e3), strategy="lobpcg", memory_budget=1)
    assert "low_memory=True" in str(err.value) and "otf" not in str(err.value)


def test_auto_labels():
    mesh, mats, ori = shells()
    bcs = {"xmin": "xyz"}
    label = r"unknown auto \(resolved to jacobi\) precond option"
    with pytest.raises(ValueError, match=label):
        solve_modes(mesh, mats, bcs, 2, ori, precond_options={"degree": 3})
    with pytest.raises(ValueError, match=label):
        solve_modes_memory(mesh, bcs, 2, precond_options={"degree": 3})
    with pytest.raises(ValueError, match=r"\(precond: auto \(resolved to jacobi\)\)"):
        solve_modes(mesh, mats, bcs, 2, ori, sigma_options={"tol": 1e-3})
    with pytest.raises(ValueError, match="unknown jacobi precond option"):
        solve_modes(mesh, mats, bcs, 2, ori, precond="jacobi", precond_options={"degree": 3})
    box = box_mesh((2, 2, 2))  # every mesh type: hex8 resolves to amg
    with pytest.raises(ValueError, match=r"auto \(resolved to amg\) precond"):
        solve_modes(box, STEEL, bcs, 2, precond_options={"foo": 3})


# ---------------------------------------------------------------- 5: estimator


@pytest.mark.parametrize("path", PATHS)
def test_estimator_storage(path):
    mesh, mats, ori = mixed_label0()
    op = unstructured(mesh, mats, ori, path=path, chunk=4)
    est = unstructured_memory(mesh, 1, path=path, chunk=4)["storage"]
    assert op.blocks[-1].me.shape[1] == 36 and op.blocks[-1].wdet is None
    assert est == sum(x.nbytes for x in jax.tree.leaves(op)) - op.rho.nbytes


def test_solve_modes_memory():
    mesh = plate()
    est = solve_modes_memory(mesh, SIMPLY_SUPPORTED, 6)
    assert est["total"] > unstructured_memory(mesh, 1)["storage"] > 0


def test_build_envelope():
    """BUILD_BYTES["stored"]["ssh8"] >= the build kernel's temporaries (memory_analysis) at 1-32
    plies and one or two phases, and <= 1.30 x each measured value; fitted at 32 plies + 2%."""
    S = jax.ShapeDtypeStruct
    c = mf.default_chunk("ssh8", 10**6)
    plies = ply_table([(0, 0)], {0: Material(C_ORT, 1.0)})[0]
    for L in (1, 2, 8, 16, 24, 32):
        for P in (1, 2):
            spec = jax.tree.map(lambda a, P=P, L=L: S((P, L, *a.shape[2:]), a.dtype), plies)
            args = (S((5000, 3), jnp.float64), S((4, c, 8), jnp.int32), S((4, c), jnp.int32),
                    (spec, S((P, 3, 3), jnp.float64)))  # fmt: skip
            temp = mf._build.lower("ssh8", "stored", *args).compile().memory_analysis()
            per = temp.temp_size_in_bytes / c
            assert per <= BUILD_BYTES["stored"]["ssh8"] <= 1.30 * per, (L, P, per)


MEASURE_SETUP = """
import numpy as np
import jax.numpy as jnp
from jaxmodal import config
config.enable_x64()
from jaxmodal.materials import Laminate, Material, Ply, orthotropic
from jaxmodal.mesh import box_mesh, solid_shell
from jaxmodal.operators import unstructured
C = orthotropic(140e9, 10e9, 10e9, 0.3, 0.3, 0.45, 5e9, 5e9, 3.6e9)
stack = ((0.0, 0.25), (1.5708, 0.5), (0.0, 0.25))
lam = Laminate(tuple(Ply(Material(C, 1600.0), a, f) for a, f in stack))
mats = {{0: lam}}
mesh = solid_shell(box_mesh((820, 820, 1), spacing=(1e-3, 1e-3, 1e-3)), direction=(0, 0, 1))[0]
op = unstructured(mesh, mats, path="{path}")  # warm-up
v = jnp.ones((op.n_dof, {m}))
op.apply_K(v).block_until_ready(); op.apply_M(v).block_until_ready()
"""
MEASURE = """
op = unstructured(mesh, mats, path="{path}")
v = jnp.ones((op.n_dof, {m}))
op.apply_K(v).block_until_ready()
op.apply_M(v).block_until_ready()
"""


@pytest.mark.peak
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
@pytest.mark.parametrize("path", PATHS)
def test_estimator_large(path):
    """820 x 820 x 1 three-ply ssh8 (672k elements, 4.04M DOF): the measured peak of build +
    apply_K + apply_M within 20% of the estimate (>= 2 GB). It checks storage and vectors: the
    build term is below the tolerance (test_build_envelope checks it)."""
    from benchmarks import harness

    mesh = solid_shell(box_mesh((820, 820, 1), spacing=(1e-3, 1e-3, 1e-3)), direction=(0, 0, 1))[0]
    for m in (1, 16):
        est = unstructured_memory(mesh, m, path=path)
        kw = dict(path=path, m=m)
        peak = harness.measure(MEASURE.format(**kw), MEASURE_SETUP.format(**kw) + "del op, v")[
            "peak_bytes"
        ]
        assert est["total"] >= 2e9 and abs(est["total"] - peak) <= 0.2 * peak, (m, est, peak)


# ---------------------------------------------------------------- 6: probes


def test_element_factor():
    mesh, mats, ori = shells()
    op = unstructured(mesh, mats, ori, chunk=4).scaled(s_M=jnp.linspace(0.5, 2, mesh.n_el))
    b = op.blocks[0]
    d = _block_data(op, b)
    assert xi_shapes(op)["elem"][0][2] == 8
    for i in range(b.dofs.shape[0] // b.chunk):  # the last chunk is padded (8 = 6 + 2)
        sl_ = {k: v[i * b.chunk : (i + 1) * b.chunk] for k, v in d.items()}
        F, Me = element_factor(op, b, sl_), element_matrices(op, b, sl_)[1]
        FF = jnp.einsum("eaq,ebq,ij->eaibj", F, F, jnp.eye(3)).reshape(Me.shape)
        assert np.abs(FF - Me).max() <= 1e-12 * np.abs(Me).max()
    assert F.shape == (4, 8, 8) and np.abs(F[2:]).max() == 0


def test_probe_covariance():
    mesh, mats, ori = mixed_label0()
    sM = np.random.default_rng(4).uniform(0.5, 2, mesh.n_el)
    nm = np.linspace(0, 0.1, len(mesh.points))
    op = unstructured(mesh, mats, ori, nodal_mass=nm, chunk=4).scaled(s_M=jnp.asarray(sM))
    M = assemble(mesh, mats, ori, s_M=sM, nodal_mass=nm)[1].toarray()
    f = np.arange(op.n_dof) % 7 != 3  # an arbitrary mask
    S = _S(op) * f[:, None]
    assert np.abs(S @ S.T - M * np.outer(f, f)).max() <= 1e-12 * np.abs(M).max()


# ---------------------------------------------------------------- 7: .inp reader


INP = """*NODE
1, 0, 0, 0
2, 1, 0, 0
3, 1, 1, 0
4, 0, 1, 0
5, 0, 0, 1
6, 1, 0, 1
7, 1, 1, 1
8, 0, 1, 1
*ELEMENT, TYPE={t}, ELSET=PLATE
1, {conn}
"""


def test_inp_sc8r(tmp_path):
    p = tmp_path / "shell.inp"
    p.write_text(INP.format(t="SC8R", conn="1, 2, 3, 4, 5, 6, 7, 8"))
    mesh = read(p)
    assert list(mesh.blocks) == ["ssh8"] and mesh.blocks["ssh8"].tolist() == [list(range(8))]
    # the file's thickness axis: bottom face 1-4 on y = 0, top face 5-8 on y = 1
    p.write_text(INP.format(t="SC8R", conn="1, 5, 6, 2, 4, 8, 7, 3"))
    mesh = read(p)
    X = mesh.points[mesh.blocks["ssh8"][0]]
    assert (X[:4, 1] == 0).all() and (X[4:, 1] == 1).all()
    unstructured(mesh, {0: Material(C_ORT, 1.0)})  # det J > 0 (check_elements at build)
    for t in ("SC6R", "SC8"):  # SC8 is not an Abaqus element
        p.write_text(INP.format(t=t, conn="1, 2, 3, 4, 5, 6"))
        with pytest.raises(ValueError, match=f"unsupported solid element type {t}"):
            read(p)
