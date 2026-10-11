"""Voxel stencil operator against the scaled PR 4 SciPy oracle and PR 5's unstructured."""

import dataclasses
import functools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, R, rel

from jaxmodal.elements import ELEMENTS, kernels
from jaxmodal.materials import Material, phase_table, phases, rotate
from jaxmodal.memory import stencil_memory
from jaxmodal.mesh import box_mesh, rcm, voxel_mesh
from jaxmodal.operators import assemble, reference_element, stencil, unstructured, voxel
from jaxmodal.operators.voxel import grid_layout

ORIENT = {1: R}
MASSES = ("consistent", "lumped")
SLABS = (1, 4, None)  # 4 does not divide 5 (F) or 6 (T): padded layers
H = (0.5, 1.0, 0.75)


def t_mask():
    """4 x 5 x 6 mask (scan axis z) with the features the slab loop must handle."""
    mask = np.random.default_rng(7).random((4, 5, 6)) < 0.75
    mask[:2, :2, 4:] = False
    mask[0, 0, 4] = mask[1, 1, 5] = True  # corner-only contact at grid node (1, 1, 5)
    mask[3, 4, 3] = mask[3, 4, 4] = False  # grid node (4, 5, 4) dropped: a halo layer
    mask[2, 2, 3], mask[2, 2, 4] = True, False  # active / inactive across the z = 4 boundary
    return mask


@functools.cache
def case(name):
    """mesh, s_K, s_M, nodal_mass."""
    rng = np.random.default_rng(3)
    if name == "F":
        m = voxel_mesh((5, 4, 3), H, (1.0, -2.0, 0.5))
    elif name == "S":  # PR 1's voxelized sphere, scan axis y
        shape, h = (10, 12, 11), np.array([0.5, 0.4, 0.3])
        ext = np.array(shape) * h
        c = (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) + 0.5) * h
        m = voxel_mesh(
            shape, tuple(h), mask=np.linalg.norm(c - ext / 2, axis=-1) < 0.45 * ext.min()
        )
    else:  # T variants and R
        m = voxel_mesh((4, 5, 6), H, mask=t_mask())
        n = m.n_el
        mat, ori = {
            "Tmat": (rng.integers(0, 2, n), np.zeros(n, int)),
            "Tori": (np.ones(n, int), rng.integers(0, 2, n)),
            "Tmix": (rng.integers(0, 2, n), rng.integers(0, 2, n)),
            "R": (rng.integers(0, 2, n), np.zeros(n, int)),
        }[name]
        m = dataclasses.replace(m, material=mat, orientation=ori)
        if name == "R":
            m = rcm(m)[0]
    s_K, s_M = rng.uniform(0.1, 2, m.n_el), rng.uniform(0.1, 2, m.n_el)
    s_K[2] = s_M[4] = 0.0
    nodal = rng.uniform(0, 1, len(m.points)) * (rng.uniform(size=len(m.points)) < 0.5)
    return m, s_K, s_M, nodal


NAMES = ("F", "S", "Tmat", "Tori", "Tmix", "R")


@functools.cache
def oracle(name, mass):
    m, s_K, s_M, nodal = case(name)
    return assemble(m, MATERIALS, ORIENT, mass=mass, nodal_mass=nodal, s_K=s_K, s_M=s_M)


@functools.cache
def operator(name, mass, slab):
    m, s_K, s_M, nodal = case(name)
    op = stencil(m, MATERIALS, ORIENT, mass=mass, nodal_mass=nodal, slab=slab)
    return op.scaled(s_K=s_K, s_M=s_M)


def vec(n, m, seed=3):
    return np.random.default_rng(seed).standard_normal((n, m))


def test_t_mask_features():
    """T's mask has a dropped node in a halo layer (slab 1 and 4), an inactive cell next to an
    active one across the z = 4 slab boundary, and a corner-only contact."""
    mask = t_mask()
    used = np.zeros((5, 6, 7), bool)
    for i, j, k in zip(*np.nonzero(mask), strict=True):
        used[i : i + 2, j : j + 2, k : k + 2] = True
    assert not used[:, :, 4].all()  # z = 4 is a halo layer for slab 1 and slab 4
    assert (mask[:, :, 3] & ~mask[:, :, 4]).any()
    block = mask[0:2, 0:2, 4:6]
    assert block.sum() == 2 and block[0, 0, 0] and block[1, 1, 1]


# ---------------------------------------------------------------- 1, 2: applies and diagonals


@pytest.mark.parametrize("slab", SLABS)
@pytest.mark.parametrize("mass", MASSES)
@pytest.mark.parametrize("name", NAMES)
def test_apply_and_diagonals(name, mass, slab):
    K, M = oracle(name, mass)
    op = operator(name, mass, slab)
    for m in (1, 5):
        v = vec(K.shape[0], m)
        assert rel(op.apply_K(jnp.asarray(v)), K @ v) < 1e-12
        assert rel(op.apply_M(jnp.asarray(v)), M @ v) < 1e-12
    assert rel(op.diag_K(), K.diagonal()) < 1e-12
    assert rel(op.diag_M(), M.diagonal()) < 1e-12


@pytest.mark.parametrize("mass", MASSES)
@pytest.mark.parametrize("name", NAMES)
def test_matches_unstructured(name, mass):
    m, s_K, s_M, nodal = case(name)
    ref = unstructured(m, MATERIALS, ORIENT, mass=mass, nodal_mass=nodal).scaled(s_K=s_K, s_M=s_M)
    op = operator(name, mass, None)
    v = jnp.asarray(vec(op.n_dof, 3))
    for f in ("apply_K", "apply_M"):
        assert rel(getattr(op, f)(v), getattr(ref, f)(v)) < 1e-12
    for f in ("diag_K", "diag_M"):
        assert rel(getattr(op, f)(), getattr(ref, f)()) < 1e-12


# ---------------------------------------------------------------- 3: gradients


def rescaled(m, spacing):
    o, h = np.asarray(m.voxel.origin), np.asarray(m.voxel.spacing)
    return dataclasses.replace(m, points=o + (m.points - o) * (np.asarray(spacing) / h))


def central(f, x, d, h):
    return (f(x + h * d) - f(x - h * d)) / (2 * h)


def test_gradients():
    m, s_K, s_M, nodal = case("Tmix")
    rng = np.random.default_rng(9)
    v, w = vec(3 * len(m.points), 1, 4), vec(3 * len(m.points), 1, 5)
    for mass in MASSES:
        op = operator("Tmix", mass, 4)
        form = {
            "s_K": lambda x, op=op: jnp.vdot(w, op.scaled(s_K=x).apply_K(jnp.asarray(v))),
            "s_M": lambda x, op=op: jnp.vdot(w, op.scaled(s_M=x).apply_M(jnp.asarray(v))),
            "rho": lambda x, op=op: jnp.vdot(
                w, dataclasses.replace(op, rho=x).apply_M(jnp.asarray(v))
            ),
            "C": lambda x, op=op: jnp.vdot(w, dataclasses.replace(op, C=x).apply_K(jnp.asarray(v))),
        }
        for name, f in form.items():
            x = jnp.asarray(getattr(op, name))
            d = jnp.asarray(rng.standard_normal(x.shape))
            g = jnp.vdot(jax.grad(f)(x), d)
            assert abs(g - central(f, x, d, 1e-3)) <= 1e-8 * abs(g), (mass, name)
    # s_K gradient = per-element w_e^T K_e v_e (SciPy element matrices)
    op = operator("Tmix", "consistent", 4)
    g = jax.grad(lambda s: jnp.vdot(w, op.scaled(s_K=s).apply_K(jnp.asarray(v))))(jnp.asarray(s_K))
    ref = np.empty(m.n_el)
    for e, row in enumerate(m.blocks["hex8"]):
        C = rotate(MATERIALS[m.material[e]].C, R if m.orientation[e] == 1 else np.eye(3))
        dofs = (3 * row[:, None] + np.arange(3)).ravel()
        ref[e] = w[dofs, 0] @ np.asarray(kernels.stiffness("hex8", m.points[row], C)) @ v[dofs, 0]
    assert rel(g, ref) < 1e-12
    # spacing: against central differences of assemble on the rescaled mesh
    h0, d = np.array(H), rng.standard_normal(3)
    for i, f in enumerate(("apply_K", "apply_M")):
        g = jax.grad(
            lambda h, f=f: jnp.vdot(
                w, getattr(dataclasses.replace(op, spacing=h), f)(jnp.asarray(v))
            )
        )(jnp.asarray(h0))

        def ref_form(h, i=i):
            A = assemble(rescaled(m, h), MATERIALS, ORIENT, s_K=s_K, s_M=s_M, nodal_mass=nodal)[i]
            return float(w[:, 0] @ (A @ v[:, 0]))

        fd = central(ref_form, h0, d, 1e-5)
        assert abs(np.dot(g, d) - fd) <= 1e-7 * abs(fd), f


# ---------------------------------------------------------------- 4: no recompile


@pytest.mark.parametrize("mass", MASSES)
def test_no_recompile(mass):
    m, s_K, s_M, nodal = case("Tmix")
    op = operator("Tmix", mass, 4)
    v = jnp.asarray(vec(op.n_dof, 2))
    exe = {
        f: jax.jit(lambda o, v, f=f: getattr(o, f)(v)).lower(op, v).compile()
        for f in ("apply_K", "apply_M")
    }
    rng = np.random.default_rng(1)
    s_K2, s_M2, nodal2 = (
        rng.uniform(0.1, 2, m.n_el),
        rng.uniform(0.1, 2, m.n_el),
        rng.uniform(0, 1, len(m.points)),
    )
    mats2 = {k: Material(2.0 * mt.C, 3.0 * mt.rho) for k, mt in MATERIALS.items()}
    h2 = (0.6, 1.1, 0.8)
    C2, rho2 = phase_table(phases(m)[1], mats2, ORIENT)
    op2 = dataclasses.replace(
        op.scaled(s_K=s_K2, s_M=s_M2),
        C=C2,
        rho=rho2,
        spacing=jnp.asarray(h2),
        nodal=jnp.asarray(np.repeat(nodal2, 3)),
    )
    K, M = assemble(
        rescaled(m, h2), mats2, ORIENT, mass=mass, nodal_mass=nodal2, s_K=s_K2, s_M=s_M2
    )
    assert rel(exe["apply_K"](op2, v), K @ np.asarray(v)) < 1e-12
    assert rel(exe["apply_M"](op2, v), M @ np.asarray(v)) < 1e-12


# ---------------------------------------------------------------- 5, 6, 7, 8: checks and layout


def test_geometry_check():
    m = case("Tmat")[0]
    pts = m.points.copy()
    pts[7, 1] += 1e-6
    rot = m.blocks["hex8"].copy()
    rot[3] = rot[3][[1, 2, 3, 0, 5, 6, 7, 4]]  # same nodes, positive det J, wrong VTK order
    for bad in (dataclasses.replace(m, points=pts), dataclasses.replace(m, blocks={"hex8": rot})):
        with pytest.raises(ValueError, match="voxel grid"):
            stencil(bad, MATERIALS, ORIENT)
    for f in (lambda m: stencil(m, MATERIALS), lambda m: stencil_memory(m, 1)):
        with pytest.raises(ValueError, match="voxel grid"):
            f(box_mesh((2, 2, 2)))


def test_inputs():
    m = case("F")[0]
    for kw, match in (
        (dict(mass="diag"), "mass"),
        (dict(slab=0), "slab"),
        (dict(nodal_mass=-np.ones(len(m.points))), "nodal_mass"),
        (dict(nodal_mass=np.ones(3)), "nodal_mass"),
    ):
        with pytest.raises(ValueError, match=match):
            stencil(m, MATERIALS, **kw)
    op = stencil(m, MATERIALS, slab=100)
    assert op.slab == grid_layout(m.voxel.shape)[1] == 5
    for kw in (dict(s_K=np.ones(3)), dict(s_M=np.ones((m.n_el, 1)))):
        with pytest.raises(ValueError, match="shape"):
            op.scaled(**kw)
    for v in (jnp.ones(op.n_dof), jnp.ones((op.n_dof + 3, 1))):
        for f in (op.apply_K, op.apply_M):
            with pytest.raises(ValueError, match="v must have shape"):
                f(v)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="float64"):
            stencil(m, MATERIALS)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_int32(monkeypatch):
    seen = {}
    monkeypatch.setattr(voxel, "check_int32", lambda **k: seen.update(k))
    op = stencil(case("Tmix")[0], MATERIALS, ORIENT, slab=4)
    assert seen == dict(n_dof=op.n_dof, node=op.node.size, cell=op.cell.size, phase=4)
    assert op.node.dtype == op.cell.dtype == op.phase.dtype == jnp.int32


def test_reference_element_and_axes():
    for name, axes, layers in (("F", (0, 2, 1), 5), ("S", (1, 2, 0), 12), ("Tmix", (2, 1, 0), 6)):
        op = operator(name, "consistent", 4)
        m = case(name)[0]
        assert (op.axes, grid_layout(m.voxel.shape)[1]) == (axes, layers)
        Ke, me = reference_element(op)
        X = m.points[m.blocks["hex8"][0]]
        for p, (mat, ori) in enumerate(phases(m)[1]):
            C = rotate(MATERIALS[mat].C, R if ori == 1 else np.eye(3))
            assert rel(Ke[p], kernels.stiffness("hex8", X, C)) < 1e-14
        assert rel(np.kron(me, np.eye(3)), kernels.mass("hex8", X, 1.0)) < 1e-14
    n = ELEMENTS["hex8"].N.shape[1]
    assert me.shape == (n, n)


# ---------------------------------------------------------------- 9: estimator


@pytest.mark.parametrize("name", ("F", "S", "Tmix"))
def test_estimator_storage(name):
    op = operator(name, "consistent", 4)
    est = stencil_memory(case(name)[0], 1, slab=4)["storage"]
    small = op.C.nbytes + op.rho.nbytes + op.spacing.nbytes
    assert est == sum(x.nbytes for x in jax.tree.leaves(op)) - small


@pytest.mark.slow
@pytest.mark.parametrize("P", (1, 2))
def test_workspace_memory_analysis(P):
    mesh = voxel_mesh((87, 87, 87), H)
    if P == 2:
        mesh = dataclasses.replace(mesh, material=np.arange(mesh.n_el) % 2)
    op = stencil(mesh, MATERIALS)
    for m in (1, 4, 16, 24):
        ws = stencil_memory(mesh, m)["workspace"]
        v = jax.ShapeDtypeStruct((op.n_dof, m), jnp.float64)
        for f in (voxel._apply_K, voxel._apply_M):
            temp = f.lower(op, v).compile().memory_analysis().temp_size_in_bytes
            assert temp <= ws <= 1.1 * temp or f is voxel._apply_M and temp <= ws, (m, f, temp, ws)


ACCEPT_SETUP = """
from jaxmodal import config
config.enable_x64()
from jaxmodal.mesh import voxel_mesh
from jaxmodal.materials import Material, isotropic
from jaxmodal.operators import stencil
mats = {0: Material(isotropic(1.0, 0.3), 1.0)}
mesh = voxel_mesh((87, 87, 87))
op = stencil(mesh, mats)  # warm-up: full run, freed below
v = jnp.ones((op.n_dof, 24))
op.apply_K(v).block_until_ready(); op.apply_M(v).block_until_ready()
del op, v
"""
ACCEPT = """
op = stencil(mesh, mats)
key = jax.random.key(0)
arrays = [jax.random.normal(jax.random.fold_in(key, i), (op.n_dof, 24)) for i in range(7)]
v = arrays.pop()
jax.block_until_ready(arrays)
op.apply_K(v).block_until_ready()
op.apply_M(v).block_until_ready()
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
def test_mac_acceptance():
    """2.04M DOF voxel box: build + six (n_dof, 24) arrays + apply_K + apply_M, measured peak
    within 20% of the estimate (stencil_memory total plus the six arrays)."""
    from benchmarks import harness

    mesh = voxel_mesh((87, 87, 87))
    est = stencil_memory(mesh, 24)["total"] + 6 * 8 * 3 * len(mesh.points) * 24
    peak = harness.measure(ACCEPT, ACCEPT_SETUP)["peak_bytes"]
    assert est >= 2e9 and abs(est - peak) <= 0.2 * peak, (est, peak)


# ---------------------------------------------------------------- 11: benchmark


def test_benchmark_slab_rule():
    path = Path(__file__).parents[1] / "benchmarks" / "results" / "stencil_slab.json"
    r = json.loads(path.read_text())
    assert all(r["slab_rule_ok"].values()), r["slab_rule_ok"]
    assert all(
        x["slab_used"] == voxel.grid_layout((87, 87, 87))[2] for x in r["runs"] if x["slab"] is None
    )


@pytest.mark.slow
def test_benchmark_smoke():
    from benchmarks import stencil_slab

    r = stencil_slab.main(n=4, write=False)
    assert len(r["runs"]) >= 2 and set(r["slab_rule_ok"]) == {"m1", "m16"}
