"""Matrix-free unstructured operator against the scaled PR 4 SciPy oracle."""

import dataclasses
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, R, dense_reference, distort, mixed_mesh, rel

from jaxmodal.elements import kernels
from jaxmodal.materials import Material, phase_table, phases, rotate
from jaxmodal.memory import BUILD_BYTES, OTF_BYTES, unstructured_memory
from jaxmodal.mesh import box_mesh, rcm, renumber, voxel_mesh
from jaxmodal.operators import assemble, matfree, unstructured

ORIENT = {1: R}
MODES, PATHS, MASSES = ("stored", "otf"), ("scatter", "gather"), ("consistent", "lumped")
SMALL = dict(chunk=5, range_nodes=23)  # chunk 5 pads 8- and 27-element blocks; 23 ∤ n_nodes
# E: narrow windows, two ranges with a large overlap (exercises the last range's overlap pass)
SIZES = {"E": dict(chunk=4, range_nodes=203)}


def scalings(m, seed):
    rng = np.random.default_rng(seed)
    s_K, s_M = rng.uniform(0.1, 2, m.n_el), rng.uniform(0.1, 2, m.n_el)
    s_K[3] = s_M[5] = 0.0
    nodal = rng.uniform(0, 1, len(m.points)) * (rng.uniform(size=len(m.points)) < 0.5)
    return s_K, s_M, nodal


@functools.cache
def case(name):
    """mesh, s_K, s_M, nodal_mass; A also carries the original mesh and p, q."""
    rng = np.random.default_rng(11)
    if name in ("A", "C"):
        m0 = mixed_mesh(rng)
        s_K, s_M, nodal = scalings(m0, 1)
        if name == "C":  # shuffled, no rcm: elements not sorted by min node
            p = rng.permutation(len(m0.points))
            q = np.concatenate(
                [
                    s + rng.permutation(len(c))
                    for s, c in zip(m0.offsets.values(), m0.blocks.values(), strict=True)
                ]
            )
            m = renumber(m0, p, q)
            for c in m.blocks.values():
                assert (np.diff(c.min(1)) < 0).any()
            return m, s_K[q], s_M[q], nodal[p], None
        m, p, q = rcm(m0)
        return m, s_K[q], s_M[q], nodal[p], (m0, s_K, s_M, nodal, p, q)
    if name == "B":
        m = rcm(distort(box_mesh((3, 3, 3), type="hex27"), True, rng))[0]
    elif name == "E":
        m = rcm(box_mesh((12, 2, 2), type="hex20"))[0]
    else:  # D: asymmetric masked voxel mesh, mixed labels
        mask = np.ones((4, 3, 3), bool)
        mask[2:, 1:, 2] = mask[0, 2, :2] = mask[3, 0, 0] = False
        v = voxel_mesh((4, 3, 3), spacing=(0.5, 1.0, 0.75), mask=mask)
        v = dataclasses.replace(
            v, material=rng.integers(0, 2, v.n_el), orientation=rng.integers(0, 2, v.n_el)
        )
        m = rcm(v)[0]
    return (m, *scalings(m, 2), None)


@functools.cache
def oracle(name, mass):
    m, s_K, s_M, nodal, _ = case(name)
    return assemble(m, MATERIALS, ORIENT, mass=mass, nodal_mass=nodal, s_K=s_K, s_M=s_M)


@functools.cache
def operator(name, mode, path, mass, **kw):
    m, s_K, s_M, nodal, _ = case(name)
    kw = SIZES.get(name, SMALL) | kw
    op = unstructured(
        m,
        MATERIALS,
        ORIENT,
        mode=mode,
        mass=mass,
        nodal_mass=nodal,
        path=path,
        **kw,
    )
    return op.scaled(s_K=s_K, s_M=s_M)


def vec(n, m, seed=3):
    return np.random.default_rng(seed).standard_normal((n, m))


# ---------------------------------------------------------------- 2: assemble with scalings


@pytest.mark.parametrize("mass", MASSES)
def test_assemble_scalings(mass):
    m = mixed_mesh(np.random.default_rng(1))
    s_K, s_M, nodal = scalings(m, 4)
    K, M = assemble(m, MATERIALS, ORIENT, mass=mass, s_K=s_K, s_M=s_M)
    Kref, Mref = dense_reference(m, mass == "lumped", s_K, s_M)
    assert rel(K.toarray(), Kref) < 1e-14 and rel(M.toarray(), Mref) < 1e-14
    ones = np.ones(m.n_el)
    for a, b in zip(
        assemble(m, MATERIALS, ORIENT, mass=mass),
        assemble(m, MATERIALS, ORIENT, mass=mass, s_K=ones, s_M=ones),
        strict=True,
    ):
        assert (a != b).nnz == 0
    _, Mn = assemble(m, MATERIALS, ORIENT, mass=mass, s_K=s_K, s_M=s_M, nodal_mass=nodal)
    assert rel((Mn - M).diagonal(), np.repeat(nodal, 3)) < 1e-15
    for bad in (ones[:-1], -ones, ones * np.nan, ones * np.inf):
        for kw in ("s_K", "s_M"):
            with pytest.raises(ValueError, match=kw):
                assemble(m, MATERIALS, ORIENT, **{kw: bad})


# ---------------------------------------------------------------- 3, 4: applies and diagonals


CASES = [
    (n, mode, path, mass)
    for n in "ABCDE"
    for mode in MODES
    for path in PATHS
    for mass in MASSES
    if n not in "CE" or path == "gather"
]


@pytest.mark.parametrize("name, mode, path, mass", CASES)
def test_apply_and_diagonals(name, mode, path, mass):
    K, M = oracle(name, mass)
    op = operator(name, mode, path, mass)
    for m in (1, 5):
        v = vec(K.shape[0], m)
        assert rel(op.apply_K(jnp.asarray(v)), K @ v) < 1e-12
        assert rel(op.apply_M(jnp.asarray(v)), M @ v) < 1e-12
    assert rel(op.diag_K(), K.diagonal()) < 1e-12
    assert rel(op.diag_M(), M.diagonal()) < 1e-12


@pytest.mark.parametrize("path", PATHS)
def test_one_chunk_one_range(path):
    m = case("A")[0]
    K, M = oracle("A", "consistent")
    op = operator("A", "stored", path, "consistent", chunk=None, range_nodes=10**6)
    assert all(b.n_el <= b.chunk for b in op.blocks) and op.n_ranges in (0, 1)
    v = vec(3 * len(m.points), 2)
    assert (
        rel(op.apply_K(jnp.asarray(v)), K @ v) < 1e-12
        and rel(op.apply_M(jnp.asarray(v)), M @ v) < 1e-12
    )


@pytest.mark.parametrize("path", PATHS)
def test_permuted(path):
    """The reordered operator with s[q], nodal[p] and v permuted by p reproduces the original."""
    m0, s_K, s_M, nodal, p, _ = case("A")[4]
    K0, M0 = assemble(m0, MATERIALS, ORIENT, nodal_mass=nodal, s_K=s_K, s_M=s_M)
    op = operator("A", "otf", path, "consistent")
    v0 = vec(K0.shape[0], 3)
    for A0, f in ((K0, op.apply_K), (M0, op.apply_M)):
        y = np.asarray(f(jnp.asarray(v0.reshape(-1, 3, 3)[p].reshape(-1, 3)))).reshape(-1, 3, 3)
        back = np.empty_like(y)
        back[p] = y
        assert rel(back.reshape(-1, 3), A0 @ v0) < 1e-12


# ---------------------------------------------------------------- 5: no recompile


SWAPPED = {0: MATERIALS[1], 1: Material(MATERIALS[0].C, 1.3)}


@pytest.mark.parametrize("mode, path", [(mo, pa) for mo in MODES for pa in PATHS])
def test_no_recompile(mode, path):
    m, s_K, s_M, nodal, _ = case("A")
    v = jnp.asarray(vec(3 * len(m.points), 2))
    C2, rho2 = phase_table(phases(m)[1], SWAPPED, ORIENT)
    s_K2, s_M2, nodal2 = scalings(m, 9)
    for mass in MASSES:
        op = operator("A", mode, path, mass)
        fK = jax.jit(lambda op, v: op.apply_K(v)).lower(op, v).compile()
        fM = jax.jit(lambda op, v: op.apply_M(v)).lower(op, v).compile()
        new = op.scaled(s_K=s_K2, s_M=s_M2)
        new = dataclasses.replace(new, rho=rho2, nodal=jnp.repeat(jnp.asarray(nodal2), 3))
        mats = MATERIALS
        if mode == "otf":
            new, mats = (
                dataclasses.replace(new, C=C2),
                {
                    0: Material(SWAPPED[0].C, MATERIALS[0].rho),
                    1: Material(SWAPPED[1].C, MATERIALS[1].rho),
                },
            )
        K, _ = assemble(m, mats, ORIENT, s_K=s_K2)
        _, M = assemble(m, SWAPPED, ORIENT, mass=mass, nodal_mass=nodal2, s_M=s_M2)
        assert rel(fK(new, v), K @ np.asarray(v)) < 1e-12
        assert rel(fM(new, v), M @ np.asarray(v)) < 1e-12


# ---------------------------------------------------------------- 6, 7: int32, inputs


@pytest.mark.parametrize("path", PATHS)
def test_int32(monkeypatch, path):
    seen = {}
    monkeypatch.setattr(matfree, "check_int32", lambda **kw: seen.update(kw))
    op = operator.__wrapped__("A", "stored", path, "consistent")
    keys = {"n_dof", "phase", "hex8_dofs", "hex20_dofs"} | (
        {"inv", "window"} if path == "gather" else set()
    )
    assert set(seen) == keys
    assert seen["hex20_dofs"] == op.blocks[1].dofs.size
    if path == "gather":
        assert seen["inv"] == op.inv.size and op.inv.dtype == jnp.int32
        assert all(b.starts.dtype == jnp.int32 for b in op.blocks)
    assert all(b.dofs.dtype == b.phase.dtype == jnp.int32 for b in op.blocks)


def test_inputs():
    m, s_K, *_ = case("A")
    build = functools.partial(unstructured, m, MATERIALS, ORIENT)
    for kw, match in [
        (dict(mode="x"), "mode"),
        (dict(mass="x"), "mass"),
        (dict(path="x"), "path"),
        (dict(chunk=0), "chunk"),
        (dict(range_nodes=0), "range_nodes"),
        (dict(nodal_mass=-np.ones(len(m.points))), "nodal_mass"),
        (dict(nodal_mass=np.ones(3)), "nodal_mass"),
    ]:
        with pytest.raises(ValueError, match=match):
            build(**kw)
    with pytest.raises(ValueError, match="material label"):
        unstructured(m, {0: MATERIALS[0]}, ORIENT)
    bent = dataclasses.replace(m, points=m.points * np.array([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="det J"):
        unstructured(bent, MATERIALS, ORIENT)
    op = operator("A", "stored", "gather", "consistent")
    assert op.C is None and op.inv is not None
    assert all(b.geo is None and b.Ke is not None and b.starts is not None for b in op.blocks)
    op2 = operator("A", "otf", "scatter", "consistent")
    assert op2.inv is None and all(
        b.Ke is None and b.wdet is None and b.starts is None for b in op2.blocks
    )
    for bad in (s_K[:-1], s_K[None]):
        with pytest.raises(ValueError, match="s_K"):
            op.scaled(s_K=bad)
    with pytest.raises(ValueError, match="s_M"):
        op.scaled(s_M=s_K[:-1])
    assert op.scaled(s_M=2 * s_K).s_K is op.s_K  # None keeps the current value
    for v in (jnp.ones(op.n_dof), jnp.ones((op.n_dof + 3, 1))):
        with pytest.raises(ValueError, match="v must"):
            op.apply_K(v)
    with pytest.raises(ValueError, match="stored mode"):
        dataclasses.replace(op, C=op2.C).apply_K(jnp.ones((op.n_dof, 1)))
    big = operator("A", "stored", "gather", "consistent", range_nodes=10**6)
    assert big.R == len(m.points)  # range_nodes clamped to n_nodes


# ---------------------------------------------------------------- 8: gather layout


def covering(n, R, n_nodes, nr):
    """Ranges covering node n: n // R, plus the overlapping last range."""
    out = {min(n // R, nr - 1)}
    if n >= n_nodes - R:
        out.add(nr - 1)
    return out


@pytest.mark.parametrize("name", "ABCDE")
def test_gather_layout(name):
    m = case(name)[0]
    lay = matfree.layout(m, "gather", **SIZES.get(name, SMALL))
    R, nr, inv, n = lay["R"], lay["n_ranges"], lay["inv"], len(m.points)
    offs = np.cumsum([0] + [b["W"] * b["k"] for b in lay["blocks"]])
    assert lay["window"] == offs[-1]
    found = {}
    empty = False
    for r in range(nr):
        r0 = min(r * R, n - R)
        for b in lay["blocks"]:
            assert 0 <= b["starts"][r] and b["starts"][r] + b["W"] <= b["n_pad"]
            conn = m.blocks[b["type"]]
            touched = ((conn >= r0) & (conn < r0 + R)).any(1)
            empty |= not touched.any()
        for j, slots in enumerate(inv[r]):
            for val in slots[slots != offs[-1]]:
                bi = np.searchsorted(offs, val, side="right") - 1
                b = lay["blocks"][bi]
                e, ai = divmod(val - offs[bi], b["k"])
                e += b["starts"][r]
                assert e < b["n_el"]  # no padded element
                assert 3 * m.blocks[b["type"]][e, ai // 3] + ai % 3 == 3 * r0 + j
                key = (r, bi, e, ai)
                found[key] = found.get(key, 0) + 1
    expected = {
        (r, bi, e, ai)
        for bi, b in enumerate(lay["blocks"])
        for e, row in enumerate(m.blocks[b["type"]])
        for a, node in enumerate(row)
        for r in covering(node, R, n, nr)
        for ai in range(3 * a, 3 * a + 3)
    }
    assert set(found) == expected and set(found.values()) == {1}
    if name == "E":  # narrow windows, so a missing overlap pass is visible
        assert nr == 2 and lay["blocks"][0]["W"] < m.n_el
    if name in "ABE":
        assert n % R and nr * R > n  # the last range overlaps its predecessor
    if name == "A":
        assert empty
    assert matfree.layout(m, "gather")["R"] == n  # tiny mesh: the only candidate is n_nodes


@pytest.mark.parametrize("name", "ABD")
def test_estimator_storage(name):
    """storage counts exactly the per-element and per-node leaves (rho and C are tiny)."""
    for mode in MODES:
        for path in PATHS:
            op = operator(name, mode, path, "consistent")
            est = unstructured_memory(case(name)[0], 1, mode, path=path, **SMALL)["storage"]
            small = op.rho.nbytes + (op.C.nbytes if op.C is not None else 0)
            assert est == sum(x.nbytes for x in jax.tree.leaves(op)) - small


def brute_windows(conn, R, n, c):
    nr = -(-n // R)
    W = 0
    for r in range(nr):
        r0 = min(r * R, n - R)
        e = np.flatnonzero(((conn >= r0) & (conn < r0 + R)).any(1))
        if len(e):
            W = max(W, e[-1] - e[0] + 1)
    return -(-W // c) * c


def test_range_rule():
    """hex20 44^3 in natural order: the chosen R is the largest candidate within the cap."""
    m = box_mesh((44, 44, 44), type="hex20")
    conn, n = m.blocks["hex20"], len(m.points)
    lay = matfree.layout(m, "gather")
    c, val = lay["blocks"][0]["chunk"], lay["valence"]
    cands = [R for R in matfree.R_CANDIDATES if R < n] + [n]
    fits = [R for R in cands if brute_windows(conn, R, n, c) * 60 + 3 * R * val <= 6 * n]
    assert lay["R"] == max(fits) and lay["R"] not in (cands[0], cands[-1])


# ---------------------------------------------------------------- 9: gradient


@pytest.mark.parametrize("mode, path", [(mo, pa) for mo in MODES for pa in PATHS])
def test_grad_s_K(mode, path):
    m, s_K, *_ = case("A")
    op = operator("A", mode, path, "consistent")
    v, w = vec(op.n_dof, 1, 4), vec(op.n_dof, 1, 5)
    g = jax.grad(lambda s: jnp.vdot(w, op.scaled(s_K=s).apply_K(jnp.asarray(v))))(jnp.asarray(s_K))
    ref, e = np.empty(m.n_el), 0
    for t, conn in m.blocks.items():
        for row in conn:
            C = rotate(MATERIALS[m.material[e]].C, R if m.orientation[e] == 1 else np.eye(3))
            d = (3 * row[:, None] + np.arange(3)).ravel()
            ref[e] = w[d, 0] @ np.asarray(kernels.stiffness(t, m.points[row], C)) @ v[d, 0]
            e += 1
    assert rel(g, ref) < 1e-12


# ---------------------------------------------------------------- 10: compile time

COMPILE = """
import dataclasses, sys, time
import jax, jax.numpy as jnp, numpy as np
from jaxmodal import config
config.enable_x64()
from jaxmodal.mesh import box_mesh
from jaxmodal.materials import Material, isotropic
from jaxmodal.operators import unstructured

mode, path, big = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
op = unstructured(box_mesh((22, 22, 22), type="hex20"), {0: Material(isotropic(1.0, 0.3), 1.0)},
                  mode=mode, path=path, range_nodes=4096)
n_el = 10**6 if big else op.n_el
n_nodes = 4 * n_el if big else op.n_nodes
n_ranges = -(-n_nodes // op.R) if path == "gather" else 0
b = op.blocks[0]
n_pad = -(-n_el // b.chunk) * b.chunk
S = jax.ShapeDtypeStruct
size = {b.dofs.shape[0]: n_pad, op.n_el: n_el, op.n_dof: 3 * n_nodes, op.n_ranges: n_ranges}
assert len(size) == 4  # distinct leading sizes, so no leaf is remapped by mistake
sds = lambda a: S((size.get(a.shape[0], a.shape[0]),) + a.shape[1:], a.dtype)
block = dataclasses.replace(jax.tree.map(sds, b), n_el=n_el)
op = dataclasses.replace(jax.tree.map(sds, op), blocks=(block,), n_nodes=n_nodes, n_ranges=n_ranges)
v = S((3 * n_nodes, 1), jnp.float64)
best = 1e9
for _ in range(5):
    t = time.perf_counter()
    jax.jit(lambda o, v: o.apply_K(v)).lower(op, v).compile()
    jax.jit(lambda o, v: o.apply_M(v)).lower(op, v).compile()
    best = min(best, time.perf_counter() - t)
print(best)
"""


@pytest.mark.slow
@pytest.mark.parametrize("mode, path", [(mo, pa) for mo in MODES for pa in PATHS])
def test_compile_time(mode, path):
    """Compile time at 1M hex20 elements <= 1.5x that at 10k (fresh subprocess, no cache)."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_COMPILATION_CACHE_DIR"}

    def run(big):
        p = subprocess.run(
            [sys.executable, "-c", COMPILE, mode, path, big],
            env=env,
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
        return float(p.stdout.split()[-1])

    small, large = run("0"), run("1")
    assert large <= 1.5 * small, (small, large)


# ---------------------------------------------------------------- 11: memory estimator

SETUP = """
from jaxmodal import config
config.enable_x64()
from jaxmodal.mesh import box_mesh
from jaxmodal.materials import Material, isotropic
from jaxmodal.operators import unstructured
mats = {{0: Material(isotropic(1.0, 0.3), 1.0)}}
mesh = box_mesh((55, 55, 55), type="hex20")
op = unstructured(mesh, mats, path="{path}")  # warm-up: full run, its buffers are unmapped
v = jnp.ones((op.n_dof, {m}))
op.apply_K(v).block_until_ready(); op.apply_M(v).block_until_ready()
"""
BUILD = """
op = unstructured(mesh, mats, path="{path}")
v = jnp.ones((op.n_dof, {m}))
"""
APPLY = """
op.apply_K(v).block_until_ready()
op.apply_M(v).block_until_ready()
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes needs macOS")
@pytest.mark.parametrize("path", PATHS)
def test_estimator(path):
    """hex20 55^3 (2.08M DOF): measured peak of build + apply_K + apply_M within 20% of the
    estimate; apply_K alone within 20% of workspace + output."""
    from benchmarks import harness

    mesh = box_mesh((55, 55, 55), type="hex20")
    for m in (1, 16):
        est = unstructured_memory(mesh, m, path=path)
        peak = harness.measure(
            BUILD.format(path=path, m=m) + APPLY, SETUP.format(path=path, m=m) + "del op, v"
        )["peak_bytes"]
        assert est["total"] >= 2e9 and abs(est["total"] - peak) <= 0.2 * peak, (m, est, peak)
    # apply_K alone: its output is the only one alive (a second apply may overlap the async free
    # of the first output, which `vectors` covers in the total)
    peak = harness.measure("op.apply_K(v).block_until_ready()", SETUP.format(path=path, m=16))[
        "peak_bytes"
    ]
    want = est["workspace"] + 8 * 3 * len(mesh.points) * 16  # plus one output
    assert abs(want - peak) <= 0.2 * peak, (est, want, peak)


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_workspace_memory_analysis(mode):
    """XLA temporaries of apply_K and of the build kernel against the estimator's formulas."""
    mesh = box_mesh((30, 30, 30), type="hex20")
    mats = {0: Material(np.asarray(MATERIALS[0].C), 1.0)}
    for path in PATHS:
        op = unstructured(mesh, mats, mode=mode, path=path)
        for m in (1, 16):
            v = jax.ShapeDtypeStruct((op.n_dof, m), jnp.float64)
            temp = (
                jax.jit(lambda o, v: o.apply_K(v))
                .lower(op, v)
                .compile()
                .memory_analysis()
                .temp_size_in_bytes
            )
            ws = unstructured_memory(mesh, m, mode=mode, path=path)["workspace"]
            c = op.blocks[0].chunk
            if mode == "stored":
                assert abs(temp - ws) <= 0.1 * temp, (path, m, temp, ws)
            else:  # otf workspace is max(fitted apply_K envelope, diag_K's B matrices)
                assert temp <= ws, (path, m, temp, ws)
                if path == "scatter":
                    a, b = OTF_BYTES["hex20"]
                    assert temp <= c * (a + b * (m - 1)) <= 1.15 * temp, (m, temp)
        if mode == "otf":
            temp = jax.jit(lambda o: o.diag_K()).lower(op).compile().memory_analysis()
            assert (
                abs(temp.temp_size_in_bytes - c * BUILD_BYTES["stored"]["hex20"])
                <= 0.1 * temp.temp_size_in_bytes
            )
    S = jax.ShapeDtypeStruct
    args = (
        S(mesh.points.shape, jnp.float64),
        S((4, c, 20), jnp.int32),
        S((4, c), jnp.int32),
        S((1, 6, 6), jnp.float64),
    )
    temp = matfree._build.lower("hex20", mode, *args).compile().memory_analysis().temp_size_in_bytes
    assert abs(temp - c * BUILD_BYTES[mode]["hex20"]) <= 0.1 * temp


# ---------------------------------------------------------------- 12: benchmark defaults


def test_benchmark_defaults():
    path = Path(__file__).parents[1] / "benchmarks" / "results" / "scatter_gather.json"
    r = json.loads(path.read_text())
    assert matfree.DEFAULT_PATH["cpu"] == r["default_path_cpu"]
    assert all(r["chunk_rule_ok"].values()), r["chunk_rule_ok"]


@pytest.mark.slow
def test_benchmark_smoke():
    from benchmarks import scatter_gather

    r = scatter_gather.main(sizes={"hex8": 3}, write=False)
    assert r["default_path_cpu"] in PATHS and len(r["runs"]) == 22
