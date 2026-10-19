"""PR 18d: the kernel and compile studies (benchmarks/kernels.py, compile_study.py): their flop and
byte model, HLO walker, compile listener, smoke runs and recorded results. No wall-time
assertion: the timed percentages are a benchmark (docs/performance.md)."""

import json
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
from helpers import STEEL

from benchmarks import compile_study as C
from benchmarks import harness
from benchmarks import kernels as K
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import stencil, unstructured


def compiled(op, m=1):
    return jax.jit(op.apply_K).lower(jnp.ones((op.n_dof, m))).compile()


def flops(c):
    a = c.cost_analysis()
    return (a[0] if isinstance(a, list) else a)["flops"]


def loops(c):
    return bool(re.search(r" while\(", c.as_text()))  # the opcode; "while" also is in metadata


def test_roofline_model():
    """The kernels.py flop and byte functions against hand numbers (voxel 87^3, one material:
    0.759 GFLOP, 43.34 MB on the active counts; hex8 1152 flops and 2504 B per element, hex20
    7200 and 14888), and analytic <= cost_analysis flops <= bound x analytic on one-trip programs
    (stencil 12^3 with slab 12: 1.057; hex8 / hex20 6^3 with chunk = n_el: 1.021 / 1.008 on
    jaxlib 0.11.2), one trip asserted by no while opcode, with multi-trip controls."""
    ne, nn = 87**3, 88**3
    assert K.stencil_flops(ne, 1) == pytest.approx(0.759e9, rel=1e-3)
    assert K.stencil_bytes(ne, nn, 1) == 43_338_580
    assert K.ke_flops(24, 1, 1) == 1152 and K.ke_bytes(24, 1, 0, 1) == 2504
    assert K.ke_flops(60, 1, 1) == 7200 and K.ke_bytes(60, 1, 0, 1) == 14888
    assert K.ke_impl_bytes(24, 1, 0, 1) - K.ke_bytes(24, 1, 0, 1) == 2 * 4608 + 4 * 192
    assert K.stencil_impl_bytes(ne, nn, 4) - K.stencil_bytes(ne, nn, 4) == 768 * 4 * ne
    assert K.roofline(1, 1, 185, 490) == 185 and K.roofline(10, 1, 185, 490) == 490
    v = voxel_mesh((12,) * 3)
    c = compiled(stencil(v, STEEL, slab=12))
    assert not loops(c) and 1 <= flops(c) / K.stencil_flops(v.n_el, 1) <= 1.5
    assert loops(compiled(stencil(v, STEEL, slab=4)))
    for t, k in (("hex8", 24), ("hex20", 60)):
        b = box_mesh((6,) * 3, (1 / 6,) * 3, type=t)
        c = compiled(unstructured(b, STEEL, path="scatter", chunk=b.n_el))
        assert not loops(c) and 1 <= flops(c) / K.ke_flops(k, b.n_el, 1) <= 1.25, t
        assert loops(compiled(unstructured(b, STEEL, path="scatter", chunk=32)))


def test_fusion_report():
    """The walker follows ENTRY into the chunk loop's body: on a two-chunk stored-Ke apply a
    computation other than ENTRY holds fusions, and ENTRY holds the while."""
    b = box_mesh((4,) * 3, (1 / 4,) * 3)
    w = K.walk(compiled(unstructured(b, STEEL, path="scatter", chunk=32), 2).as_text())
    comps = w["computations"]
    assert comps[w["entry"]]["unfused"].get("while") == 1
    assert any(c["fusions"] for name, c in comps.items() if name != w["entry"])
    assert all(len(c["largest"]) <= 5 for c in comps.values())
    hlo = """ENTRY %main (p: f64[4]) -> f64[4] {
  %p = f64[4]{0} parameter(0)
  ROOT %call = f64[4]{0} call(%p), to_apply=%inner
}

%inner (q: f64[4]) -> f64[4] {
  %q = f64[4]{0} parameter(0)
  ROOT %s = f64[4]{0} sine(%q)
}
"""  # XLA:CPU outlines small loops into call ops
    assert K.walk(hlo)["computations"]["inner"]["unfused"] == {"sine": 1}


LISTEN = """
import jax, jax.numpy as jnp
jax.config.update("jax_enable_compilation_cache", False)
ev = []
jax.monitoring.register_event_duration_secs_listener(
    lambda e, d, **kw: ev.append(d) if "backend_compile" in e else None)
jax.jit(lambda x: jnp.sin(x) * 2)(jnp.ones(8)).block_until_ready()
print(len(ev), sum(ev))
"""


def test_compile_listener():
    """A listener like compile_study's: a tiny jit yields >= 1 backend-compile event and > 0 s."""
    p = subprocess.run([sys.executable, "-c", LISTEN], env=harness.child_env(),
                       capture_output=True, text=True)  # fmt: skip
    assert p.returncode == 0, p.stderr
    n, s = p.stdout.split()
    assert int(n) >= 1 and float(s) > 0


@pytest.mark.slow
def test_benchmark_smoke():
    """kernels.py's roofline, fusion and orders and compile_study at tiny sizes record their
    keys (several JAX subprocesses)."""
    r = K.run_roofline(dict(stencil=(6,), hex8=(4,), hex20=(3,)), (1,))
    assert {"B_gbps", "F_gflops", "large_cache", "chunk512"} <= set(r) and len(r["rows"]) == 3
    assert {"pct_roofline", "pct_impl_roofline", "pct_gemm"} <= set(r["rows"][0])
    assert r["rows"][1]["trips"] == 1 and r["chunk512"]["result"] in ("faster", "not determined")
    f = K.run_fusion(6, 2)
    assert set(f) == {"stencil", "stored_ke", "vcycle"} and f["vcycle"]["temp_bytes"] > 0
    o = K.run_orders(dict(box=3, plate=8), ("grid", "shuffled"))["rows"]
    assert len(o) == 4 and {"span", "apply_m16_s", "vcycle_m16_s"} <= set(o[0])
    assert "gather_m1_s" in o[0] and "gather_m1_s" not in o[-1]
    c = C.run([("voxel", 4), ("box", 2)])["rows"]
    assert c[0]["compiles"] > 0 and c[0]["compile_s"] > 0 and c[0]["slabs"] >= 1
    assert "ell_classes" in c[1] and c[1]["levels"] >= 2


def results(name):
    return json.loads((harness.RESULTS / f"{name}.json").read_text())


def test_results_pinned():
    """kernels.json and compile_study.json hold the cases the docs tables read; the recorded flops
    and bytes equal the kernels.py functions, and the roofline and percentages recompute."""
    k = results("kernels")
    ro = k["roofline"]
    B, F = ro["B_gbps"], ro["F_gflops"]
    assert B == ro["system"]["triad_gbps"] and ro["large_cache"] is True
    want = {(t, n, m) for t, ns in K.ROOF.items() for n in ns for m in K.MS}
    assert {(r["kernel"], r["size"], r["m"]) for r in ro["rows"]} == want
    for r in ro["rows"] + [ro["chunk512"]["small"]]:
        fl, lb, ib = K.model(r["kernel"], r["n_el"], r["n_nodes"], r["m"])
        assert (r["flops"], r["bytes"], r["impl_bytes"]) == (fl, lb, ib)
        g = 100 * r["gflops"]
        assert r["pct_impl_roofline"] == pytest.approx(g / K.roofline(fl, ib, B, F))
        if r["kernel"] == "stencil":
            assert r["pct_gemm"] == pytest.approx(g / r["gemm_gflops"])
        assert r["roofline_gflops"] == pytest.approx(K.roofline(fl, lb, B, F))
        assert r["gflops"] == pytest.approx(fl / r["apply_s"] / 1e9)
        assert r["pct_roofline"] == pytest.approx(100 * r["gflops"] / r["roofline_gflops"])
    assert ro["chunk512"]["small"]["chunk"] == 512
    assert set(k["fusion"]) == {"stencil", "stored_ke", "vcycle"}
    for f in k["fusion"].values():
        assert f["temp_bytes"] > 0 and f["entry"] in f["computations"]
    rows = k["orders"]["rows"]
    assert {(r["case"], r["order"]) for r in rows} == {
        (c, o) for c in K.ORDER_MESHES for o in K.ORDERS
    }
    c = results("compile_study")["rows"]
    assert [(r["case"], r["size"]) for r in c] == C.CASES
    assert all(r["compiles"] > 0 and r["compile_s"] > 0 for r in c)
    for r in c:
        assert {"n_dof", "wall_s", "lowering_s", "trace_s", "levels"} <= set(r)
        assert ("slabs" if r["case"] == "voxel" else "ell_classes") in r
