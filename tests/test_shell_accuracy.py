"""PR 16c solid-shell accuracy suite: the gates on the recorded benchmark
(benchmarks/shell_accuracy.py, results/shell_accuracy.json), per-path recomputes, and fast checks
of its pieces. See docs/verification.md#solid-shell-accuracy-pr-16c."""

import dataclasses
import functools

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from benchmarks import shell_accuracy as sa
from jaxmodal.eigen import reference_modes
from jaxmodal.materials import Laminate
from jaxmodal.mesh import box_mesh, solid_shell
from jaxmodal.operators import assemble, dof_mask, restrict
from tests.exact import plate_modes

EPS = np.finfo(float).eps
GATE = {"srao": 1e-3, "cyl": 1e-3, "skew": 1e-2}  # ssh8 error in f
CRIT = {"cyl": 2e-4, "skew": 2e-3}  # hex27 reference error in f
CHECKED = {"cyl SS": 5, "cyl SA": 4, "cyl AS": 6, "cyl AA": 6, "skew": 5}  # back-checked modes
REMOVED = {("pag", 10): [0] * 10, ("ortho", 10): [0] * 10}  # strip modes dropped by the filter
SS = {"xmin": "yz", "xmax": "yz", "ymin": "xz", "ymax": "xz"}


@functools.cache
def record():
    return sa.load()["runs"]


@functools.cache
def results():
    return sa.analyze(record())


def lam_of(mesh, mats, bcs, n):
    K, M = assemble(mesh, mats)
    return reference_modes(K, M, n, dof_mask(mesh, bcs), points=mesh.points)


# ---- the record ----


def test_record_keys_and_digests():
    runs = record()
    want = {(b, sa._key(a)) for p in sa.PARTS for b, a in sa.keys(p)}
    assert {(r["builder"], sa._key(r["args"])) for r in runs} == want
    for r in runs:
        mesh, mats, bcs = sa.BUILDERS[r["builder"]](*sa._key(r["args"]))
        assert sa.same(sa.digest(mesh, mats, dof_mask(mesh, bcs)), r["digest"]), r["args"]


def test_labels():
    """The label constants are the live lowest 10 (as sets: degenerate pairs swap by round-off)."""
    for name, Lt in sa.SRAO:
        lab = plate_modes(sa._plies(name, Lt), 1.0, 1.0, 10)[1]
        assert {tuple(x) for x in lab} == set(sa.LABELS[name, Lt])


def test_srao_gate():
    for (name, Lt), labels in sa.LABELS.items():
        r = results()[f"srao {name} {Lt}"]
        assert len(r["err"]) == len(labels)
        assert np.abs(r["err"]).max() <= GATE["srao"], (name, Lt, r["err"])
        assert (2 * r["flam"] <= 0.1 * GATE["srao"]).all()
        if (name, Lt) in REMOVED:
            assert list(r["removed"]) == REMOVED[name, Lt]


@pytest.mark.parametrize("case", [*(f"cyl {c}" for c in sa.CLASSES), "skew"])
def test_reference_gate(case):
    r, kind = results()[case], case.split()[0]
    assert r["monotone"].all() and r["decreasing"]  # Richardson: D1 D2 > 0, p >= 1; f decreasing
    assert (r["guard"][1:] < 1).all()  # no crossing between the last three levels
    assert r["checked"].sum() == CHECKED[case]
    assert r["back_ok"][r["checked"]].all() and (r["back"][r["checked"]] >= 1).all()
    assert r["nz_monotone"].all() and (r["ratio_z"] >= sa.r_z(sa.PZ)).all()
    assert r["ref"].max() <= CRIT[kind]
    assert r["err"].max() <= GATE[kind], r["err"]
    assert (2 * r["flam_shell"] <= 0.1 * GATE[kind]).all()


def test_pinned():
    """analyze against the sizing study (its own script): a swapped level or coarse/mid level
    changes these."""
    r = results()
    assert np.isclose(r["cyl SA"]["ref"].max(), 1.17e-4, rtol=0.01)
    assert np.isclose(r["cyl SA"]["nz"][0], 3.95e-5, rtol=0.01)
    assert np.isclose(r["skew"]["ref"].max(), 1.09e-3, rtol=0.01)
    assert np.isclose(
        r["srao pag 10"]["err"][sa.LABELS["pag", 10].index((1, 3, 0))], 5.77e-4, rtol=0.01
    )


RECOMPUTE = [
    ("srao", ("pag", 10, (2, 0, 0), 32, 12)),  # laminate, 90 deg ply, zero-index strip and filter
    ("srao", ("ortho", 100, (2, 4, 0), 48, 4)),  # two mirror cuts
    ("cylinder", ("ssh8", "SA", 512, 96, 1)),
    ("cylinder", ("hex27", "SA", 24, 12, 3)),
    ("skew", ("ssh8", 192, 1)),
    ("skew", ("hex27", 24, 3)),  # 1728 elements: 2 assembly chunks
]


@pytest.mark.slow
@pytest.mark.parametrize("builder, args", RECOMPUTE)
def test_recompute(builder, args):
    old = next(r for r in record() if r["builder"] == builder and sa._key(r["args"]) == args)
    new = sa.run(builder, args, lmax=False)
    lam, ref = np.array(new["lam"]), np.array(old["lam"])
    tol = np.maximum(1e-9, 2 * 10 * EPS * old["lmax"] / ref)
    assert (np.abs(lam / ref - 1) <= tol).all()
    assert new.get("parity") == old.get("parity") and new.get("removed") == old.get("removed")


@pytest.mark.slow
def test_plate_oracle():
    """plate_modes at every (laminate, sub-plate) used: P = 24 equals P = 12 (round-off in the
    z-Ritz at L/t = 100), mmax = 10 gives the same full-plate targets."""
    for (name, Lt), labels in sa.LABELS.items():
        pl, tol = sa._plies(name, Lt), 1e-9 if Lt == 10 else 1e-5
        lam, lab, _ = plate_modes(pl, 1.0, 1.0, 10, mmax=10)
        assert {tuple(x) for x in lab} == set(labels)
        for a, b in {(1 / max(m, 1), 1 / max(n, 1)) for m, n, _ in labels}:
            lo, hi = (plate_modes(pl, a, b, 12, P=P, mmax=12)[0] for P in (12, 24))
            assert np.allclose(hi, lo, rtol=tol, atol=0)


# ---- estimators ----


def test_richardson():
    h = np.array([1, 0.5, 0.25])
    for p in (2.0, 3.5):
        F = 1 + 0.01 * h[:, None] ** p
        est, pp, ok = sa.richardson(F)
        assert ok.all() and np.isclose(pp, p).all()
        assert np.isclose(est, 1.25 * 0.01 * 0.25**p / F[2]).all()
    F = 1 + 0.01 * h[:, None] ** 5
    assert np.isclose(sa.richardson(F)[0], 1.25 * (F[1] - F[2]) / 15 / F[2]).all()  # p capped at 4
    assert not sa.richardson(1 + 0.01 * h[:, None] ** 0.5)[2].any()  # p < 1
    assert not sa.richardson(np.array([[1.0], [0.9], [0.95]]))[2].any()  # not monotone


def test_nz_check():
    nz = np.array([3, 4, 5])
    for p, passes in ((1.5, True), (1.3, False), (3.0, True)):
        c = 1 + 0.01 * nz**-p
        ratio = (c[0] - c[1]) / (c[1] - c[2])
        assert (ratio >= sa.r_z(sa.PZ)) == passes
        assert np.isclose(sa.p_z(ratio), p)
    assert np.isnan(sa.p_z(1.0))  # below r(0.05): no root in the bracket


@pytest.mark.parametrize("sign", [1, -1])
def test_nz_term(sign):
    """f = 1 + A h^4 + B nz^-1.42 (1 + sign h): the term bounds the finest-level thickness error."""
    f = lambda h, nz: 1 + 0.01 * h**4 + 0.01 * nz**-sa.PZ * (1 + sign * h)  # noqa: E731
    hc = 0.4
    c = np.array([[f(hc, z)] for z in (3, 4, 5)])
    m = np.array([[f(hc / 2, z)] for z in (3, 4)])
    t, ratio, mono = sa.nz_term(c, m)
    exact = 0.01 * 3**-sa.PZ * (1 + sign * hc / 4) / f(hc / 4, 3)
    assert mono.all() and np.isclose(ratio, sa.r_z(sa.PZ)).all()
    assert (t >= exact).all() and (t <= 1.5 * exact).all()


def test_crossing():
    ext = [(1, 0), (1, 1)]
    swap = np.array([[1.0, 1.02], [0.95, 0.96]])  # the upper mode moves 3x its gap
    benign = np.array([[1.0, 1.2], [0.99, 1.19]])
    assert sa.crossing(swap, ext[:1], ext)[1][0, 0] >= 1
    assert sa.crossing(benign, ext[:1], ext)[1][0, 0] < 1
    assert not sa.crossing(benign[::-1], ext[:1], ext)[0]  # increasing


# ---- builders, filter, digest ----


def full_plate(name, Lt, nx, ny, k, a=1.0, b=1.0):
    """The whole SS plate [0, a] x [0, b] in `ssh8`, k elements per ply."""
    pl = sa._plies(name, Lt)
    nz = k * len(pl)
    m = box_mesh((nx, ny, nz), spacing=(a / nx, b / ny, 1 / Lt / nz))
    zc = m.points[m.blocks["hex8"]].mean(1)[:, 2]
    material = np.searchsorted(np.cumsum([p[2] for p in pl])[:-1], zc)
    m, _ = solid_shell(dataclasses.replace(m, voxel=None, material=material), direction=(0, 0, 1.0))
    return m, sa._shell_mats(name), SS


def test_srao_equivalence():
    """The quarter of a label's sub-plate is a mode of the full plate: (1,1) on the sub-plate, and
    (2,1) of the full 2N x N plate (N = 4, pag, n_z = 3)."""
    q = sa.run("srao", ("pag", 10, (1, 1, 0), 2, 1), lmax=False)["lam"][0]
    assert np.isclose(lam_of(*full_plate("pag", 10, 4, 4, 1), 1)[0][0], q, rtol=1e-10)
    q = sa.run("srao", ("pag", 10, (2, 1, 0), 2, 1), lmax=False)["lam"][0]
    assert np.isclose(lam_of(*full_plate("pag", 10, 8, 4, 1), 6)[0], q, rtol=1e-10).any()


def test_filter_strip():
    mesh = sa.srao_case("pag", 10, (1, 0, 0), 4, 1)[0]
    y = mesh.points[:, 1]
    lo, hi = np.isclose(y, y.min()), np.isclose(y, y.max())
    U = np.zeros((len(y), 3, 3))
    U[:, 1, 0] = 1.0  # equal on both cut faces
    U[lo, 1, 1], U[hi, 1, 1] = 1.0, -1.0  # opposite
    assert list(sa.filter_strip(mesh, (1, 0, 0), U[:, :, :2].reshape(-1, 2))) == [True, False]
    U[lo, 1, 2], U[hi, 1, 2] = 1.0, 0.5  # neither
    with pytest.raises(ValueError, match="neither"):
        sa.filter_strip(mesh, (1, 0, 0), U.reshape(-1, 3))


@pytest.mark.parametrize("label", [(1, 0, 0), (0, 1, 0)])
def test_wide_strip(label):
    """A wide strip (width 2) has spurious cross-strip modes (for (1,0) two below the target); the
    filter removes them and the kept target equals the narrow strip's and a full-plate mode."""
    args = ("pag", 10, label, 4, 1)
    wide = sa.run("srao", args, lmax=False, width=2.0)
    narrow = sa.run("srao", args, lmax=False)
    assert wide["removed"] > 0
    assert np.isclose(wide["lam"][0], narrow["lam"][0], rtol=1e-10)
    lam = lam_of(*full_plate("pag", 10, 8 if label[0] else 2, 2 if label[0] else 8, 1), 10)[0]
    assert np.isclose(lam, narrow["lam"][0], rtol=1e-10).any()


@pytest.mark.parametrize("typ, nz", [("ssh8", 1), ("hex27", 2)])
def test_cylinder_classes(typ, nz):
    """The y-symmetric half model is the union of the 4 classes of the 1/8 model; the
    y-antisymmetric half adds only the n = 0 torsion mode below 1.1 lam_T, above every gated mode
    (lam_T = (pi / L)^2 G / rho, G = E / 2(1 + nu))."""
    lam_T = (np.pi / sa.L) ** 2 / (2 * 1.3)
    classes = [
        np.array(sa.run("cylinder", (typ, c, 8, 4, nz), lmax=False, n=24)["lam"])
        for c in sa.CLASSES
    ]
    assert min(c[-1] for c in classes) > 1.1 * lam_T  # the union is complete below 1.1 lam_T
    union = np.sort(np.concatenate(classes))
    sym = sa.run("cylinder", (typ, "S", 16, 8, nz), lmax=False, n=16, half=True)["lam"]
    assert np.allclose(sym, union[:16], rtol=1e-9)
    anti = np.array(sa.run("cylinder", (typ, "A", 16, 8, nz), lmax=False, n=48, half=True)["lam"])
    assert anti[-1] > 1.1 * lam_T
    inside = np.array([np.isclose(union, x, rtol=1e-8).any() for x in anti])
    assert inside[anti < 0.9 * lam_T].all()
    extra = anti[~inside & (anti < 1.1 * lam_T)]
    assert len(extra) == 1 and abs(extra[0] / lam_T - 1) <= (0.01 if typ == "hex27" else 0.06)
    gated = [
        (r["f"] * (1 + r["err"])).max() ** 2 for k, r in results().items() if k.startswith("cyl")
    ]
    assert max(gated) < 0.9 * lam_T


def test_builders():
    m, _, bcs = sa.skew("hex27", 4, 1)
    P = m.points
    assert np.isclose(P[:, 2].max() - P[:, 2].min(), sa.TS)
    for c in ([0, 0, 0], [1, 0, 0], [np.sin(sa.ANGLE), np.cos(sa.ANGLE), 0]):
        assert np.isclose(P, c).all(1).any()  # corners of the rhombus: 60 deg, skew angle 30 deg
    assert sorted(bcs) == ["xmax", "xmin", "ymax", "ymin"] and set(bcs.values()) == {"xyz"}
    m, _, bcs = sa.cylinder("hex27", "SA", 4, 2, 1)
    P, ns = m.points, m.node_sets
    r = np.hypot(P[:, 0], P[:, 1])
    assert np.isclose(r.min(), sa.R - sa.T / 2) and np.isclose(r.max(), sa.R + sa.T / 2)
    assert (P[:, :2] >= -1e-12).all() and np.isclose(P[:, 2].max(), sa.L / 2) and P[:, 2].min() == 0
    assert np.allclose(P[ns["xmin"], 1], 0) and np.allclose(P[ns["xmax"], 0], 0)
    assert np.allclose(P[ns["ymin"], 2], 0) and np.allclose(P[ns["ymax"], 2], sa.L / 2)
    assert bcs == {"ymin": "xyz", "xmin": "y", "xmax": "x", "ymax": "xy"}


def test_digest():
    mesh, mats, bcs = sa.srao_case("pag", 10, (1, 1, 0), 2, 1)
    free = dof_mask(mesh, bcs)
    d = sa.digest(mesh, mats, free)
    noisy = dataclasses.replace(mesh, points=mesh.points * (1 + 1e-15))
    assert sa.same(sa.digest(noisy, mats, free), d)
    moved = mesh.points.copy()
    moved[5, 0] += 1e-3
    lam = mats[1].plies[0]
    changed = [
        sa.digest(dataclasses.replace(mesh, points=moved), mats, free),
        sa.digest(mesh, {**mats, 1: Laminate((lam._replace(angle=0.0),))}, free),
        sa.digest(mesh, mats, free ^ (np.arange(len(free)) == np.flatnonzero(free)[0])),
        sa.digest(dataclasses.replace(mesh, material=np.roll(mesh.material, 1)), mats, free),
    ]
    assert not any(sa.same(c, d) for c in changed)


def test_smoke():
    r = sa.run("skew", ("hex27", 4, 1))
    assert len(r["lam"]) == len(r["parity"]) == 10 and r["lmax"] > r["lam"][-1]


# ---- no locking ----


def bending(typ, Lt, n, k=16):
    """Lowest 6 bending lam (z-parity) of the SS square n x n x 1, and its floor 10 eps lmax/lam."""
    m = box_mesh((n, n, 1), spacing=(1 / n, 1 / n, 1 / Lt), type="hex8")
    if typ == "ssh8":
        m, _ = solid_shell(dataclasses.replace(m, voxel=None))
    K, M = assemble(m, sa.ISO)
    free = dof_mask(m, SS)
    lam, V = reference_modes(K, M, k, free, points=m.points)
    z = m.points[:, 2]
    top, bot = np.flatnonzero(z > 0.5 / Lt), np.flatnonzero(z < 0.5 / Lt)
    top, bot = (i[np.lexsort(m.points[i, :2].T)] for i in (top, bot))
    U = V.reshape(-1, 3, V.shape[1])
    nrm = np.linalg.norm(V, axis=0)
    even = np.linalg.norm(U[top, 2] - U[bot, 2], axis=0) + np.linalg.norm(
        U[top, :2] + U[bot, :2], axis=(0, 1)
    )
    odd = np.linalg.norm(U[top, 2] + U[bot, 2], axis=0) + np.linalg.norm(
        U[top, :2] - U[bot, :2], axis=(0, 1)
    )
    assert (np.minimum(even, odd) <= 1e-6 * nrm).all()  # every mode even or odd
    b = even <= 1e-6 * nrm
    assert b.sum() >= 6
    Kf, Mf, _ = restrict(K, M, free)
    lmax = eigsh(Kf, 1, Mf, which="LA", tol=1e-3)[0][0]
    return lam[b][:6], 10 * EPS * lmax / lam[b][:6]


def lock_ratio(typ):
    e, d = {}, {}
    for Lt in (10, 1000):
        runs = {n: bending(typ, Lt, n, 16 if typ == "ssh8" else 60) for n in (6, 12, 24, 48)}
        lam = np.array([runs[n][0] for n in (12, 24, 48)])
        est, _, ok = sa.richardson(lam)
        assert ok.all() or typ == "hex8"
        e[Lt] = np.abs(runs[6][0] / runs[48][0] - 1).max()
        d[Lt] = est.max() + max(runs[6][1].max(), runs[48][1].max())
    return (e[1000] + (1 + e[1000]) * d[1000]) / (e[10] - (1 + e[10]) * d[10]), e, d


def test_no_locking():
    ratio, e, d = lock_ratio("ssh8")
    assert ratio <= 1.5
    assert all(d[Lt] <= 0.1 * e[Lt] for Lt in e)


@pytest.mark.slow
def test_hex8_locks():
    """Negative control: the same bounded ratio on hex8 is far above 1.5."""
    assert lock_ratio("hex8")[0] > 10
