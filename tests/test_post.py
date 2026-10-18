"""PR 17: the postprocessor (normalization, MAC, participation, clusters, recovery, export).

References: dense scipy.linalg.eigh on the assembled K_ff, M_ff (all modes), analytic box masses
and inertias, test-written formulas (np.cross, Gauss sums, eigvalsh von Mises), tests/exact.mac,
tests/shell_ref.recover, imposed linear fields, meshio round trips.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np
import pytest
import scipy.linalg as sl
import shell_ref as ref
from exact import mac as exact_mac
from helpers import ORTHO, PAIRS, R, distort, rel
from test_shell import C_ANI, TILT, linear_field, mixed, patch

from jaxmodal import post
from jaxmodal.elements import ELEMENTS, gauss
from jaxmodal.materials import Material, isotropic, rotate
from jaxmodal.mesh import Mesh, box_mesh, read, solid_shell, voxel_mesh
from jaxmodal.operators import assemble, dof_mask, rigid_modes, stencil, unstructured
from jaxmodal.post import stress
from jaxmodal.post.modal import _dot, _gram

ROOT = Path(__file__).parents[1]
MATS = {0: Material(isotropic(3.0, 0.3), 2.0), 1: Material(rotate(ORTHO, R), 0.7)}
CLAMPED, SLIDING = {"xmin": "xyz"}, {"zmin": "z"}


def labelled(m, material, orientation):
    return Mesh(m.points, m.blocks, m.node_sets, m.element_sets, np.asarray(material),
                np.asarray(orientation))  # fmt: skip


def two_phase():
    """hex8 3x2x2 with two densities (node centroid != centre of mass) and two orientations."""
    m = box_mesh((3, 2, 2), spacing=(1 / 3, 0.83 / 2, 0.67 / 2), type="hex8")
    return labelled(m, np.arange(12) % 2, (np.arange(12) // 3) % 2)


def dense(mesh, free, materials=MATS, orientations=None, **kw):
    """All free modes by eigh: lam, V (n_dof, n_free) with zero constrained rows, K, M."""
    K, M = assemble(mesh, materials, orientations or {1: R}, **kw)
    f = np.flatnonzero(free)
    lam, X = sl.eigh(K.toarray()[np.ix_(f, f)], M.toarray()[np.ix_(f, f)])
    V = np.zeros((len(free), len(f)))
    V[f] = X
    return lam, V, K, M


def voigt(A):
    """Engineering Voigt strain of the displacement gradient A."""
    return np.array([A[i, j] if i == j else A[i, j] + A[j, i] for i, j in PAIRS])


# ---------------------------------------------------------------- 1: sum rule and formulas


@pytest.mark.parametrize("bcs", [CLAMPED, SLIDING], ids=["clamped", "sliding"])
@pytest.mark.parametrize("kind", ["consistent", "lumped", "nodal"])
def test_sum_rule_and_formulas(bcs, kind):
    mesh = two_phase()
    free = dof_mask(mesh, bcs)
    kw = {"mass": "lumped"} if kind == "lumped" else {}
    if kind == "nodal":
        kw["nodal_mass"] = np.random.default_rng(0).uniform(0, 0.1, len(mesh.points))
    lam, V, K, M = dense(mesh, free, **kw)
    r = post.influence(mesh, free)
    f = np.flatnonzero(free)
    expect = np.einsum("ij,ij->j", r[f], M.toarray()[np.ix_(f, f)] @ r[f])
    op = unstructured(mesh, MATS, {1: R}, **kw)
    rng = np.random.default_rng(1)
    Vs = V * rng.uniform(0.5, 2, V.shape[1]) * rng.choice([-1, 1], V.shape[1])
    MV = M @ Vs
    mu = np.einsum("ij,ij->j", Vs, MV)
    b = MV.T @ r
    for Mx, Vx in ((M, Vs), (op, jnp.asarray(Vs))):
        gamma, m_eff, total = post.participation(Vx, Mx, r)
        assert rel(total, expect) <= 1e-12
        assert (np.abs(m_eff.sum(0) - expect) <= 1e-10 * expect).all()  # per direction
        assert rel(gamma, b / mu[:, None]) <= 1e-12
        assert rel(m_eff, b**2 / mu[:, None]) <= 1e-12


def test_influence():
    mesh = two_phase()
    free = dof_mask(mesh, SLIDING)
    x = mesh.points
    for origin in (None, np.array([0.3, -1.0, 2.0])):
        r = post.influence(mesh, free, origin)
        x0 = x.mean(0) if origin is None else origin
        for k in range(3):
            e = np.eye(3)[k]
            assert np.array_equal(r[:, k], np.tile(e, len(x)) * free)
            rot = np.cross(e, x - x0).ravel() * free
            assert np.abs(r[:, 3 + k] - rot).max() <= 1e-15
    R6 = rigid_modes(mesh)
    r = post.influence(mesh, None)
    assert rel(r / np.linalg.norm(r, axis=0), R6) <= 1e-14  # same centroid and signs
    _, M = assemble(mesh, MATS, {1: R})
    com = np.array([r[:, k] @ M @ (x[:, k : k + 1] * np.eye(3)[k]).ravel() for k in range(3)])
    com /= r[:, 0] @ M @ r[:, 0]
    assert np.abs(com - x.mean(0)).max() >= 1e-2  # the default origin is not the centre of mass


def test_free_box_totals():
    L = np.array([1.0, 0.83, 0.67])
    mesh = box_mesh((3, 2, 2), spacing=L / (3, 2, 2), type="hex8")
    rho = 2.0
    lam, V, K, M = dense(mesh, np.ones(3 * len(mesh.points), bool), {0: MATS[0]}, {})
    gamma, m_eff, total = post.participation(V, M, post.influence(mesh, None))
    m = rho * L.prod()
    a, b, c = L
    expect = m * np.array([1, 1, 1, (b**2 + c**2) / 12, (a**2 + c**2) / 12, (a**2 + b**2) / 12])
    assert rel(total, expect) <= 1e-12
    assert (np.abs(m_eff[:6].sum(0) - total) <= 1e-12 * total).all()
    assert (m_eff[6:] <= 1e-20 * total).all()


def test_stencil_operator():
    mesh = voxel_mesh((3, 2, 2), spacing=(1 / 3, 0.83 / 2, 0.67 / 2))
    free = dof_mask(mesh, SLIDING)
    mats = {0: MATS[1]}
    lam, V, K, M = dense(mesh, free, mats, {})
    r = post.influence(mesh, free)
    op = stencil(mesh, mats)
    for a, b in zip(post.participation(V, op, r), post.participation(V, M, r), strict=True):
        assert rel(a, b) <= 1e-12
    W = V * 3.0 + 0.1 * np.roll(V, 1, axis=1)
    assert rel(post.normalize(W, op), post.normalize(W, M)) <= 1e-12


def test_empty_mode_set():
    mesh = two_phase()
    free = dof_mask(mesh, CLAMPED)
    _, M = assemble(mesh, MATS, {1: R})
    op = unstructured(mesh, MATS, {1: R})
    r = post.influence(mesh, free)
    for Mx in (M, op):
        gamma, m_eff, total = post.participation(np.zeros((len(free), 0)), Mx, r)
        assert gamma.shape == m_eff.shape == (0, 6) and total.shape == (6,) and (total > 0).all()
        assert post.normalize(np.zeros((len(free), 0)), Mx).shape == (len(free), 0)
    assert post.normalize(np.zeros((len(free), 0)), kind="max").shape == (len(free), 0)
    assert post.mac(np.zeros((len(free), 0)), np.zeros((len(free), 2)) + 1.0).shape == (0, 2)
    t = post.frequency_table([], np.zeros((0, 6)), total)
    assert t["clusters"] == [] and t["m_eff"].shape == (0, 6)
    assert len(post.format_table(t).splitlines()) == 1  # the header only
    shapes = [x.shape for x in post.element_stress(mesh, MATS, np.zeros((len(free), 0)), {1: R})]
    assert shapes == [(mesh.n_el, 0, 6)] * 2


# ---------------------------------------------------------------- 2: the docs recipe


def recipe():
    text = (ROOT / "docs/postprocessing.md").read_text()
    return re.search(r"<!-- recipe -->\n```python\n(.*?)```", text, re.S).group(1)


def test_recipe(tmp_path, monkeypatch):
    mesh = box_mesh((2, 2, 2), spacing=(0.5, 0.415, 0.335), type="hex20")
    monkeypatch.chdir(tmp_path)
    ns = {"mesh": mesh, "materials": MATS, "bcs": SLIDING}
    exec(recipe(), ns)
    info, table, total = ns["info"], ns["table"], ns["total"]
    assert info["converged"] and info["n_rigid"] == 3
    lam, V, _, M = dense(mesh, dof_mask(mesh, SLIDING), MATS, {})
    lam[:3] = 0
    r = post.influence(mesh, dof_mask(mesh, SLIDING))
    m_d = (V.T @ (M @ r)) ** 2
    cl = post.clusters(lam, 1e-4)
    n = info["n_rigid"] + len(ns["lam"])
    whole = [c for c in cl if c[-1] < n]  # dense clusters wholly inside the computed set
    assert [list(c) for c in table["clusters"][: len(whole)]] == [list(c) for c in whole]
    assert 1e-4 < (lam[9] - lam[8]) / lam[9] <= 3e-4  # elastic modes 6-7: apart in both partitions
    for i, c in enumerate(whole):
        assert (np.abs(table["m_eff"][i] - m_d[c].sum(0)) <= 1e-8 * total).all()
    elastic = sum(table["fraction"][i] for i in range(1, len(whole)))
    assert (elastic[[2, 3, 4]] >= 0.5).all(), elastic  # not vacuous: z, Rx, Ry
    assert rel(ns["m_rigid"].sum(0)[[0, 1, 5]], total[[0, 1, 5]]) <= 1e-12
    assert (tmp_path / "modes.vtu").exists() and (tmp_path / "modes.txt").exists()
    ns = {"mesh": mesh, "materials": MATS, "bcs": CLAMPED}
    exec(recipe(), ns)
    assert ns["info"]["n_rigid"] == 0 and len(ns["table"]["clusters"]) >= 1


# ---------------------------------------------------------------- 3: normalization and MAC


def test_normalize():
    mesh = two_phase()
    free = dof_mask(mesh, CLAMPED)
    _, V, _, M = dense(mesh, free)
    rng = np.random.default_rng(2)
    s = rng.uniform(0.1, 10, V.shape[1]) * rng.choice([-1, 1], V.shape[1])
    Vn = post.normalize(V * s, M)
    assert np.abs(Vn.T @ M @ Vn - np.eye(V.shape[1])).max() <= 1e-12
    W = rng.standard_normal(V.shape) * free[:, None]  # not M-orthogonal
    a = np.abs(W)
    j = np.argmax(a >= (1 - 1e-6) * a.max(0), axis=0)
    sign = np.sign(W[j, np.arange(W.shape[1])])
    expect = W * sign / np.sqrt(np.einsum("ij,ij->j", W, M @ W))
    assert np.abs(post.normalize(W, M) - expect).max() <= 1e-14 * np.abs(expect).max()
    assert np.array_equal(post.normalize(jnp.asarray(W), M), post.normalize(W, M))
    Vm = post.normalize(W, kind="max")
    assert np.allclose(np.linalg.norm(Vm.reshape(-1, 3, W.shape[1]), axis=1).max(0), 1, 0, 1e-15)
    w0 = post.normalize(W[:, 0], kind="max")
    assert w0.ndim == 1 and np.array_equal(w0, Vm[:, 0])
    W = rng.standard_normal((len(free), 150)) * free[:, None]  # several 64-column blocks
    W[:, 70] *= 1e-3
    a = np.abs(W)
    sign = np.sign(W[np.argmax(a >= (1 - 1e-6) * a.max(0), axis=0), np.arange(150)])
    nodal = np.linalg.norm(W.reshape(-1, 3, 150), axis=1).max(0)
    for got, scale in (
        (post.normalize(W, M), np.sqrt(np.einsum("ij,ij->j", W, M @ W))),
        (post.normalize(W, kind="max"), nodal),
    ):
        assert rel(got, W * sign / scale) <= 1e-14
    W32 = W.astype(np.float32)
    assert rel(post.normalize(W32, kind="max"), post.normalize(W32.astype(float), kind="max")) == 0


@pytest.mark.parametrize(
    "va, vb, positive",
    [(1, -(1 - 1e-12), "a"), (-(1 - 1e-12), 1, "a"), (-1, 1 - 1e-12, "a"), (-(1 - 1e-5), 1, "b")],
)
def test_sign_rule(va, vb, positive):
    a, b = 1, 4
    v = np.array([0.1, va, -0.3, 0.2, vb, 0.05])
    A = np.random.default_rng(3).standard_normal((6, 6))
    M = A @ A.T + 6 * np.eye(6)
    for kind, Mx in (("mass", M), ("max", None)):
        x, y = post.normalize(v, Mx, kind), post.normalize(-v, Mx, kind)
        assert np.array_equal(x, y)
        assert x[a if positive == "a" else b] > 0


def test_mac():
    mesh = two_phase()
    free = dof_mask(mesh, CLAMPED)
    _, V, _, M = dense(mesh, free)
    op = unstructured(mesh, MATS, {1: R})
    assert np.abs(post.mac(V, V, M) - np.eye(V.shape[1])).max() <= 1e-12
    rng = np.random.default_rng(4)
    A, B = (rng.standard_normal((len(free), k)) for k in (5, 7))
    assert np.abs(post.mac(A, B, M) - exact_mac(A, B, M)).max() <= 1e-13
    assert np.abs(post.mac(A, B, op) - exact_mac(A, B, M)).max() <= 1e-13
    A2 = A + 0.1 * A[:, ::-1]  # same shape as A, not A
    assert np.abs(post.mac(A, A2, op) - exact_mac(A, A2, M)).max() <= 1e-13
    assert np.abs(post.mac(jnp.asarray(A), jnp.asarray(B), op) - exact_mac(A, B, M)).max() <= 1e-13
    G = A.T @ B
    expect = G**2 / (np.sum(A * A, 0)[:, None] * np.sum(B * B, 0))
    assert np.abs(post.mac(A, B) - expect).max() <= 1e-14
    assert np.abs(post.mac(A, A, op, batch=2) - exact_mac(A, A, M)).max() <= 1e-13


class Spy:
    """An operator wrapper recording the widths apply_M receives."""

    def __init__(self, op):
        self.op, self.widths = op, []

    def apply_M(self, x):
        self.widths.append(x.shape[1])
        return self.op.apply_M(x)


def test_gram_batches():
    mesh = two_phase()
    _, M = assemble(mesh, MATS, {1: R})
    op = unstructured(mesh, MATS, {1: R})
    B = np.random.default_rng(5).standard_normal((M.shape[0], 130))
    A = B[:, :9]
    spy = Spy(op)
    G, d = _gram(A, spy, B)
    assert spy.widths == [64, 64, 64]  # 130 columns: two batches and a padded one
    assert G.dtype == np.float64 and rel(G, A.T @ (M @ B)) <= 1e-13
    assert rel(d, np.einsum("ij,ij->j", B, M @ B)) <= 1e-13
    spy = Spy(op)
    post.participation(B, spy, B[:, :6])
    assert spy.widths == [6, 64, 64, 64]  # M r: 6 columns, not 64
    spy = Spy(op)
    post.mac(A, B, spy)
    assert spy.widths == [64, 64, 64, 9]  # M once per column of B and of A
    spy = Spy(op)
    post.mac(B, B, spy)
    assert spy.widths == [64, 64, 64]  # B is A: one pass
    G32 = _gram(A.astype(np.float32), op, B.astype(np.float32), batch=7)[0]
    assert G32.dtype == np.float64 and rel(G32, A.T @ (M @ B)) <= 1e-6
    B32 = B.astype(np.float32)
    m32 = post.participation(B32, op, B[:, :6])[1]
    assert rel(m32, post.participation(B32.astype(float), op, B[:, :6])[1]) <= 1e-12
    A32 = A.astype(np.float32)
    assert rel(_dot(A32, B, rows=7), A32.astype(float).T @ B) <= 1e-14  # several row blocks
    N32 = post.normalize(B32, op)
    assert N32.dtype == np.float64 and rel(N32, post.normalize(B32.astype(float), op)) <= 1e-12


COMPILES = """
import json, jax
import numpy as np
from jaxmodal import config
config.enable_x64()
jax.config.update("jax_enable_compilation_cache", False)
from jaxmodal import post
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import unstructured
from jaxmodal.post.modal import _gram
m = box_mesh((3, 2, 2), type="hex20")
mats = {0: Material(isotropic(3.0, 0.3), 2.0)}
op = unstructured(m, mats)
B = np.random.default_rng(0).standard_normal((op.n_dof, 130))
U = B[:, :70]
n = [0]
def count(event, duration, **kw):
    if "backend_compile" in event:
        n[0] += 1
jax.monitoring.register_event_duration_secs_listener(count)
out = []
for f in (lambda: _gram(B[:, :3], op, B), lambda: post.element_stress(m, mats, U, chunk=5),
          lambda: post.recover(m, mats, U[:, 0], chunk=5)):
    for _ in range(2):
        c = n[0]
        f()
        out.append(n[0] - c)
print(json.dumps(out))
"""


def test_compile_counts():
    """One apply_M per batch width; repeat calls of _gram, element_stress, recover compile 0."""
    res = subprocess.run(
        [sys.executable, "-c", COMPILES], capture_output=True, text=True, check=True
    )
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out[0] == 1 and out[1::2] == [0, 0, 0], out


# ---------------------------------------------------------------- 4: cluster invariance


def prism(bcs=CLAMPED):
    mesh = box_mesh((3, 3, 4), spacing=(1 / 3,) * 3, type="hex8")
    free = dof_mask(mesh, {"zmin": "xyz"} if bcs else {})
    lam, V, _, M = dense(mesh, free, {0: MATS[0]}, {})
    return mesh, free, lam, V, M


def test_cluster_invariance():
    mesh, free, lam, V, M = prism()
    r = post.influence(mesh, free)
    _, m_eff, total = post.participation(V, M, r)
    cl = post.clusters(lam)
    assert max(len(c) for c in cl) >= 2
    rng = np.random.default_rng(6)
    W = V.copy()
    for c in cl:
        if len(c) > 1:
            Q, _ = np.linalg.qr(rng.standard_normal((len(c), len(c))))
            W[:, c] = V[:, c] @ Q
    _, m_rot, _ = post.participation(W, M, r)
    sums = [np.array([m[c].sum(0) for c in cl]) for m in (m_eff, m_rot)]
    assert (np.abs(sums[0] - sums[1]) <= 1e-12 * total).all()
    change = (np.abs(m_rot - m_eff) / total)[:, [0, 1, 3, 4]].max()
    assert change >= 1e-3, change  # per-mode values do change: not vacuous


def test_clusters():
    lam = np.array([0, 0, 0, 1.0, 1.0 + 0.5e-4, 2.0, 2.0 + 3e-4, 5.0])
    got = [list(c) for c in post.clusters(lam)]
    assert got == [[0, 1, 2], [3, 4], [5], [6], [7]]
    assert [list(c) for c in post.clusters(lam, atol=1e-3)][-3:] == [[3, 4], [5, 6], [7]]
    edge = [list(c) for c in post.clusters([1.0, 1.5, 2.0], rtol=0.25)]  # gaps 0.5: <= 0.25 * 2.0
    assert edge == [[0], [1, 2]]  # |lam[i+1]| and <= : 1.5 - 1.0 > 0.375, 2.0 - 1.5 <= 0.5
    for bad in ([1.0, 0.5], [[0.0, 1.0]]):
        with pytest.raises(ValueError, match="ascending"):
            post.clusters(bad)
    assert post.cluster_rtol({"thresholds": np.array([1e-6, 3e-6])}) == pytest.approx(3e-4)
    assert post.cluster_rtol({"thresholds": np.zeros(0), "r_acc": 2e-6}) == pytest.approx(2e-4)
    assert post.cluster_rtol({"thresholds": np.zeros(0)}) == pytest.approx(1e-4)


# ---------------------------------------------------------------- 5: recovery = C eps


def hex_mixed(t, curved, rng):
    m = distort(box_mesh((2, 2, 2), type=t), curved, rng)
    return labelled(m, [0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 1, 0])


@pytest.mark.parametrize("t", ["hex8", "hex20", "hex27"])
@pytest.mark.parametrize("curved", [False, True], ids=["straight", "curved"])
def test_hex_recovery_linear(t, curved):
    rng = np.random.default_rng(7)
    mesh = hex_mixed(t, curved, rng)
    A, b = rng.standard_normal((3, 3)), rng.standard_normal(3)
    u = (mesh.points @ A.T + b).ravel()
    out = post.recover(mesh, MATS, u, {1: R})[t]
    eps = voigt(A)
    assert np.abs(out["eps"] - eps).max() <= 1e-12 * np.abs(A).max()
    for e in range(mesh.n_el):
        C = np.asarray(rotate(MATS[mesh.material[e]].C, R if mesh.orientation[e] else np.eye(3)))
        assert rel(out["sigma"][e], np.tile(C @ eps, (len(out["sigma"][e]), 1))) <= 1e-10
    w = out["weights"]
    assert abs(w.sum() - 8.0) <= 1e-12 * 8
    rho = np.array([MATS[k].rho for k in mesh.material])
    _, M = assemble(mesh, MATS, {1: R})
    ones = np.tile([1.0, 0, 0], len(mesh.points))
    assert abs((rho[:, None] * w).sum() - ones @ M @ ones) <= 1e-12 * (ones @ M @ ones)
    h8, el = ELEMENTS["hex8"], ELEMENTS[t]
    X8 = mesh.points[mesh.blocks[t][:, :8]]
    if not curved:
        pts = np.einsum("qa,eai->eqi", h8.shape(el.points), X8)
        assert np.abs(out["points"] - pts).max() <= 1e-14
        J = np.einsum("eai,qaj->eqij", X8, h8.grad(el.points))
        assert rel(w, el.weights * np.linalg.det(J)) <= 1e-12
    if t == "hex8":
        p3, w3 = gauss(3)
        J = np.einsum("eai,qaj->eqij", X8, h8.grad(p3))
        assert rel(w.sum(1), (w3 * np.linalg.det(J)).sum(1)) <= 1e-12


GAUSS2 = [(2 * np.array(p) - 1) * ref.G for p in np.ndindex(2, 2, 2)]


@pytest.mark.parametrize("name", ["i", "ii", "iii", "iv"])
def test_ssh8_patch_recovery(name):
    """PR 16's patch-recovery test through post.recover (owned here)."""
    hexes, shells, _, Q, C = patch(name)
    Q = np.eye(3) if name == "iv" else Q
    conn = shells.blocks["ssh8"]
    for s in range(6):
        uA, sig = linear_field(C, Q, s, hexes.points)
        out = post.recover(shells, {0: Material(C, 1.0)}, uA, {0: Q})["ssh8"]
        assert np.abs(out["sigma"] - sig).max() <= 1e-10 * np.abs(sig).max()
        CQ = np.asarray(rotate(C, Q))
        assert np.abs(out["sigma"] - out["eps"] @ CQ.T).max() <= 1e-12 * np.abs(sig).max()
    vol = [sum(np.linalg.det(ref.jac(shells.points[c], np.array(p))) for p in GAUSS2) for c in conn]
    assert rel(out["weights"].sum(1), vol) <= 1e-12


def ref_plies(mat):
    """shell_ref ply tuples (C, rho, angle, fraction) of a Laminate."""
    return [(np.asarray(p.material.C), p.material.rho, p.angle, p.fraction) for p in mat.plies]


def test_laminate_recovery():
    mesh, mats = mixed()
    rng = np.random.default_rng(8)
    u = rng.standard_normal(3 * len(mesh.points))
    out = post.recover(mesh, mats, u, {0: TILT})
    off = mesh.offsets["ssh8"]
    s = out["ssh8"]
    for i, c in enumerate(mesh.blocks["ssh8"]):
        plies = ref_plies(mats[mesh.material[off + i]])
        d = (3 * c[:, None] + np.arange(3)).ravel()
        x, w, eps, sig, alpha = ref.recover(mesh.points[c], plies, u[d], TILT)
        n = len(w)
        for key, a in (("points", x), ("weights", w), ("eps", eps), ("sigma", sig)):
            assert rel(s[key][i, :n], a) <= 1e-12, key
        assert np.linalg.norm(alpha) >= 1e-4 * np.linalg.norm(eps)
        assert (s["weights"][i, n:] == 0).all() and np.isfinite(s["sigma"][i]).all()
    n_ply = {len(mats[k].plies) for k in mesh.material[off:]}
    assert n_ply == {2, 3}  # padded and full stacks
    h = out["hex8"]
    C = np.asarray(rotate(C_ANI, TILT))
    assert np.abs(h["sigma"] - h["eps"] @ C.T).max() <= 1e-12 * np.abs(h["sigma"]).max()


def test_chunks():
    mesh, mats = mixed()
    u = np.random.default_rng(9).standard_normal(3 * len(mesh.points))
    a, b = post.recover(mesh, mats, u, chunk=5), post.recover(mesh, mats, jnp.asarray(u))
    for t in a:
        for k in a[t]:
            assert np.abs(a[t][k] - b[t][k]).max() <= 1e-13 * np.abs(b[t][k]).max()
    U = np.random.default_rng(10).standard_normal((3 * len(mesh.points), 3))
    for x, y in zip(post.element_stress(mesh, mats, U, chunk=5), post.element_stress(mesh, mats, U),
                    strict=True):  # fmt: skip
        assert np.abs(x - y).max() <= 1e-13 * np.abs(y).max()


# ---------------------------------------------------------------- 6: energy


def energy_meshes():
    rng = np.random.default_rng(11)
    for t in ("hex8", "hex20", "hex27"):
        yield t, hex_mixed(t, True, rng), MATS, {1: R}
    mesh, mats = mixed()
    yield "mixed", mesh, mats, {0: TILT}


@pytest.mark.parametrize("case", ["hex8", "hex20", "hex27", "mixed"])
def test_energy(case):
    _, mesh, mats, ori = next(c for c in energy_meshes() if c[0] == case)
    K, M = assemble(mesh, mats, ori)
    _, X = sl.eigh(K.toarray(), M.toarray())
    for u in (np.random.default_rng(12).standard_normal(K.shape[0]), X[:, 6]):
        out = post.recover(mesh, mats, u, ori)
        e = sum((b["weights"][..., None] * b["eps"] * b["sigma"]).sum() for b in out.values())
        assert abs(e - u @ K @ u) <= 1e-10 * abs(u @ K @ u)


# ---------------------------------------------------------------- 7: element stress, von Mises


@pytest.mark.parametrize("t", ["hex8", "hex20", "hex27"])
def test_element_stress_linear(t):
    rng = np.random.default_rng(13)
    mesh = hex_mixed(t, True, rng)
    A = rng.standard_normal((3, 3))
    eps, sig = post.element_stress(mesh, MATS, jnp.asarray((mesh.points @ A.T).ravel()), {1: R})
    assert eps.shape == sig.shape == (mesh.n_el, 6)
    assert np.abs(eps - voigt(A)).max() <= 1e-12 * np.abs(A).max()
    for e in range(mesh.n_el):
        C = np.asarray(rotate(MATS[mesh.material[e]].C, R if mesh.orientation[e] else np.eye(3)))
        assert rel(sig[e], C @ voigt(A)) <= 1e-10


@pytest.mark.parametrize("case", ["hex8", "hex20", "hex27", "mixed"])
def test_element_stress(case, monkeypatch):
    rng = np.random.default_rng(13)
    if case != "mixed":
        mesh, mats, ori = hex_mixed(case, True, rng), MATS, {1: R}
    else:
        (mesh, mats), ori = mixed(), {0: TILT}
    t0, widths = next(iter(mesh.blocks)), []
    kernel = stress._MEAN[t0]

    def spy(X, p, U):
        widths.append(U.shape[1])
        return kernel(X, p, U)

    monkeypatch.setitem(stress._MEAN, t0, spy)
    U = rng.standard_normal((3 * len(mesh.points), 70))
    eps, sig = post.element_stress(mesh, mats, U, ori)
    assert eps.shape == (mesh.n_el, 70, 6) and set(widths) == {64}  # 64 + padded 6
    out = post.recover(mesh, mats, U[:, 3], ori)
    for t, b in out.items():
        e = slice(mesh.offsets[t], mesh.offsets[t] + len(mesh.blocks[t]))
        w = b["weights"][..., None]
        for key, got in (("eps", eps), ("sigma", sig)):
            mean = (w * b[key]).sum(1) / w.sum(1)
            assert np.abs(got[e, 3] - mean).max() <= 1e-13 * np.abs(mean).max()
    U32 = U.astype(np.float32)
    for x, y in zip(
        post.element_stress(mesh, mats, U32, ori),
        post.element_stress(mesh, mats, U32.astype(float), ori),
        strict=True,
    ):
        assert x.dtype == np.float64 and np.array_equal(x, y)  # fp32 input, cast per batch
    single = [post.element_stress(mesh, mats, U[:, j], ori) for j in range(70)]
    for i, got in enumerate((eps, sig)):
        one = np.stack([x[i] for x in single], 1)
        assert np.abs(one - got).max() <= 1e-13 * np.abs(got).max()


def test_von_mises():
    rng = np.random.default_rng(14)
    S = rng.standard_normal((20, 6))
    T = np.zeros((20, 3, 3))
    for v, (i, j) in enumerate(PAIRS):
        T[:, i, j] = T[:, j, i] = S[:, v]
    p = np.linalg.eigvalsh(T)
    expect = np.sqrt(((p[:, 0] - p[:, 1]) ** 2 + (p[:, 1] - p[:, 2]) ** 2
                      + (p[:, 2] - p[:, 0]) ** 2) / 2)  # fmt: skip
    assert rel(post.von_mises(S), expect) <= 1e-13
    for k in (3, 4, 5):
        assert post.von_mises(-2.0 * np.eye(6)[k]) == pytest.approx(2 * np.sqrt(3), rel=1e-15)
    assert post.von_mises(np.array([-3.0, 0, 0, 0, 0, 0])) == pytest.approx(3.0, rel=1e-15)
    assert post.von_mises(np.array([2.0, 2, 2, 0, 0, 0])) == 0


# ---------------------------------------------------------------- 8: export


def disjoint():
    """hex8, hex20, hex27 and ssh8 blocks with disjoint nodes, in TYPES order."""
    parts = [box_mesh((2, 1, 1), origin=(0, 3 * i, 0), type=t)
             for i, t in enumerate(("hex8", "hex20", "hex27"))]  # fmt: skip
    shell = solid_shell(box_mesh((2, 1, 1), spacing=(1, 1, 0.2), origin=(0, 9, 0)),
                        direction=(0, 0, 1))[0]  # fmt: skip
    parts.append(shell)
    points, blocks, off = [], {}, 0
    for p in parts:
        ((t, c),) = p.blocks.items()
        blocks[t] = c + off
        points.append(p.points)
        off += len(p.points)
    n = sum(len(c) for c in blocks.values())
    return Mesh(np.vstack(points), blocks, {}, {}, np.arange(n) % 3, np.arange(n) % 2)


def test_write_modes(tmp_path):
    mesh = disjoint()
    rng = np.random.default_rng(15)
    V = rng.standard_normal((3 * len(mesh.points), 4))
    cells = {"vm": rng.standard_normal(mesh.n_el), "s6": rng.standard_normal((mesh.n_el, 6)),
             "id": np.arange(mesh.n_el)}  # fmt: skip
    post.write_modes(tmp_path / "a.vtu", mesh, V, cells)
    m = meshio.vtu.read(tmp_path / "a.vtu")
    assert np.array_equal(m.points, mesh.points)
    assert len(m.cells) == 4
    for b, c in zip(m.cells, mesh.blocks.values(), strict=True):
        assert np.array_equal(b.data, c)
    data = {**cells, "material": mesh.material, "orientation": mesh.orientation}
    for k, v in data.items():
        assert np.array_equal(np.concatenate(m.cell_data[k]), v), k
    for j in range(4):
        for i in range(3):
            assert np.array_equal(m.point_data[f"mode_{j:03d}"][:, i], V[i::3, j])
    post.write_modes(tmp_path / "b.vtu", mesh, jnp.asarray(V[:, 1]))  # one device column
    one = meshio.vtu.read(tmp_path / "b.vtu").point_data
    assert list(one) == ["mode_000"] and np.array_equal(one["mode_000"], V[:, 1].reshape(-1, 3))


def test_write_modes_tensor_order(tmp_path):
    """The recipe's VTK reorder: component 3 is XY, strains carry tensor shear."""
    mesh = box_mesh((2, 1, 1), type="hex20")
    A = np.random.default_rng(16).standard_normal((3, 3))
    eps, sigma = post.element_stress(mesh, MATS, (mesh.points @ A.T).ravel())
    vtk, half = [0, 1, 2, 5, 3, 4], np.array([1, 1, 1, 0.5, 0.5, 0.5])  # as in the recipe
    post.write_modes(tmp_path / "t.vtu", mesh, np.zeros((3 * len(mesh.points), 0)),
                     {"sigma": sigma[:, vtk], "eps": (eps * half)[:, vtk]})  # fmt: skip
    m = meshio.vtu.read(tmp_path / "t.vtu")
    s, e = np.concatenate(m.cell_data["sigma"]), np.concatenate(m.cell_data["eps"])
    assert np.array_equal(s[:, 3], sigma[:, 5])  # sigma_12
    for c, (i, j) in zip((3, 4, 5), ((0, 1), (1, 2), (0, 2)), strict=True):
        assert np.allclose(e[:, c], (A[i, j] + A[j, i]) / 2, rtol=0, atol=1e-13)
    assert not any(k.startswith("mode_") for k in m.point_data)


def test_write_modes_names_and_merge(tmp_path):
    one = box_mesh((1, 1, 1))
    for n, w in ((1000, 3), (1001, 4)):
        post.write_modes(tmp_path / "n.vtu", one, np.ones((24, n)))
        names = list(meshio.vtu.read(tmp_path / "n.vtu").point_data)
        assert names == [f"mode_{i:0{w}d}" for i in range(n)] and sorted(names) == names
    mesh, mats = mixed()
    U = np.random.default_rng(17).standard_normal((3 * len(mesh.points), 2))
    vm = post.von_mises(post.element_stress(mesh, mats, U)[1])
    post.write_modes(tmp_path / "m.vtu", mesh, U, {f"vm_{i:03d}": vm[:, i] for i in range(2)})
    m = meshio.vtu.read(tmp_path / "m.vtu")
    assert len(m.cells) == 1 and m.cells[0].type == "hexahedron"  # hex8 + ssh8 merge on read
    assert np.array_equal(m.cells[0].data, np.vstack(list(mesh.blocks.values())))
    assert np.array_equal(np.concatenate(m.cell_data["vm_001"]), vm[:, 1])
    back = read(tmp_path / "m.vtu")
    assert list(back.blocks) == ["hex8"] and np.array_equal(back.points, mesh.points)
    assert np.array_equal(back.material, mesh.material)
    assert np.array_equal(back.orientation, mesh.orientation)


# ---------------------------------------------------------------- 9: frequency table


def test_frequency_table():
    mesh, free, lam, V, M = prism()
    _, m_eff, total = post.participation(V, M, post.influence(mesh, free))
    t = post.frequency_table(lam, m_eff, total)
    expect, cur = [], [0]
    for i in range(1, len(lam)):
        if lam[i] - lam[i - 1] <= 1e-4 * abs(lam[i]):
            cur.append(i)
        else:
            expect.append(cur)
            cur = [i]
    expect.append(cur)
    assert [list(c) for c in t["clusters"]] == expect
    assert rel(t["f"], [np.sqrt(lam[c].mean()) / (2 * np.pi) for c in expect]) <= 1e-14
    m = np.array([m_eff[c].sum(0) for c in expect])
    assert rel(t["m_eff"], m) <= 1e-12 and rel(t["fraction"], m / total) <= 1e-12
    assert rel(t["lam"], [lam[c].mean() for c in expect]) <= 1e-14
    assert np.array_equal(t["total"], total)
    assert np.abs(t["cumulative"][-1] - 1).max() <= 1e-10
    text = post.format_table(t).splitlines()
    assert len(text) == 1 + len(expect) and "f [Hz]" in text[0]
    i = next(i for i, c in enumerate(expect) if len(c) > 1)
    assert text[1 + i].split()[0] == f"{expect[i][0] + 1}-{expect[i][-1] + 1}"
    z = post.frequency_table(lam, m_eff, np.r_[total[:5], 0.0])
    assert (z["fraction"][:, 5] == 0).all() and (z["cumulative"][:, 5] == 0).all()
    mesh, free, lam, V, M = prism(bcs=None)
    lam = np.r_[np.zeros(6), lam[6:]]
    lam[1] = -1e-12  # tiny negative rigid value
    lam = np.sort(lam)
    _, m_eff, total = post.participation(V, M, post.influence(mesh, free))
    t = post.frequency_table(lam, m_eff, total, atol=1e-9)  # atol for dense rigid values
    assert t["f"][0] == 0 and list(t["clusters"][0]) == list(range(6))


# ---------------------------------------------------------------- 10: errors


def test_errors(tmp_path):
    mesh = two_phase()
    free = dof_mask(mesh, CLAMPED)
    n = len(free)
    _, M = assemble(mesh, MATS, {1: R})
    V = np.random.default_rng(18).standard_normal((n, 2))
    r = post.influence(mesh, free)
    cases = [
        (lambda: post.participation(V[:-3], M, r), "shape"),
        (lambda: post.participation(V, M, r[:-3]), "shape"),
        (lambda: post.influence(mesh, free[:-3]), "mask"),
        (lambda: post.influence(mesh, free.astype(int)), "mask"),
        (lambda: post.mac(V, V[:-3], M), "shape"),
        (lambda: post.normalize(np.c_[V, np.zeros(n)], M), "zero"),
        (lambda: post.normalize(np.zeros(n), kind="max"), "zero"),
        (lambda: post.normalize(np.r_[np.inf, np.ones(n - 1)], kind="max"), "non-finite"),
        (lambda: post.mac(V, np.zeros((n, 1))), "zero"),
        (lambda: post.participation(np.zeros((n, 1)), M, r), "zero"),
        (lambda: post.normalize(V, M, kind="unit"), "kind"),
        (lambda: post.normalize(V), "pass M"),
        (lambda: post.normalize(V, M, batch=0), "batch"),
        (lambda: post.write_modes(tmp_path / "a.xdmf", mesh, V), "vtu"),
        (lambda: post.write_modes(tmp_path / "a.vtu", mesh, V[:-3]), "shape"),
        (
            lambda: post.write_modes(tmp_path / "a.vtu", mesh, V, {"material": np.zeros(12)}),
            "reserved",
        ),
        (lambda: post.write_modes(tmp_path / "a.vtu", mesh, V, {"a<b": np.zeros(12)}), "name"),
        (lambda: post.write_modes(tmp_path / "a.vtu", mesh, V, {1: np.zeros(12)}), "name"),
        (
            lambda: post.write_modes(tmp_path / "a.vtu", mesh, V, {"x": np.zeros((12, 3, 3))}),
            "numeric",
        ),
        (lambda: post.write_modes(tmp_path / "a.vtu", mesh, V, {"x": np.zeros(11)}), "numeric"),
        (
            lambda: post.write_modes(tmp_path / "a.vtu", mesh, V, {"x": np.zeros(12, bool)}),
            "numeric",
        ),
        (lambda: post.recover(mesh, MATS, V, {1: R}), r"\(108,\) array"),
        (lambda: post.normalize(V[:-1], kind="max"), "shape"),
        (lambda: post.influence(mesh, free, np.zeros((len(mesh.points), 3))), "origin"),
        (lambda: post.frequency_table([1.0, 2.0], np.ones((5, 6)), np.ones(6)), "len"),
        (lambda: post.frequency_table([0.0, np.nan, 1.0], np.ones((3, 6)), np.ones(6)), "finite"),
        (lambda: post.recover(mesh, MATS, V[:-3, 0], {1: R}), "expected"),
        (lambda: post.recover(mesh, MATS, V[:, 0], chunk=0), "chunk"),
        (lambda: post.element_stress(mesh, MATS, V, batch=0), "batch"),
        (lambda: post.element_stress(mesh, MATS, V, chunk=0), "chunk"),
    ]
    for f, match in cases:
        with pytest.raises(ValueError, match=match):
            f()
    _, mats = mixed()
    with pytest.raises(ValueError, match="aminate"):
        post.recover(mesh, {0: mats[1], 1: MATS[1]}, V[:, 0], {1: R})
