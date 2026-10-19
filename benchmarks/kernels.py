"""Kernel study (PR 18d): roofline, fusion and node/element order -> results/kernels.json.

Parts (--parts, default all), each case in a fresh subprocess (harness.measure_json):
- roofline: voxel stencil and stored-Ke apply_K at m = 1, 4, 16 against min(B I, F) on
  lower-bound bytes (B the JAX triad, F the fp64 matmul peak, both rerun here), with the
  implementation bytes, the bare 24 x 24 GEMM ceiling (stencil) and a chunk-512 probe (hex8).
- fusion: the compiled HLO of the stencil, stored-Ke and hex20 AMG V-cycle programs, walked
  over every computation reachable from ENTRY; memory_analysis temp bytes. Report only.
- orders: apply_K (scatter, m = 1 and 16; gather at m = 1 when it fits) and the AMG V-cycle
  (m = 16) on box hex20 44^3 and the ssh8 256^2 plate in grid, RCM, Morton and shuffled order.
Timed children keep macOS's large-block cache (large_cache=True; docs/performance.md).
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from benchmarks import harness

LARGE = True  # timed children: freed blocks reused, first touch excluded
MS = (1, 4, 16)
ROOF = dict(stencil=(87,), hex8=(48,), hex20=(32,))
ORDERS = ("grid", "rcm", "morton", "shuffled")
ORDER_MESHES = dict(box=44, plate=256)


def stencil_flops(n_el, m):
    """24 x 24 element matvec per active cell and column."""
    return 2 * 24 * 24 * n_el * m


def ke_flops(k, n_el, m):
    return 2 * k * k * n_el * m


def stencil_bytes(n_el, n_nodes, m):
    """Lower bound: cell id (4) and s (8) per active cell; u and y (48 m) and node id (4) per
    active node (one material: phase not read)."""
    return 12 * n_el + (48 * m + 4) * n_nodes


def ke_bytes(k, n_el, n_nodes, m):
    """Lower bound: packed Ke (8 k (k + 1) / 2), dofs (4 k) and s_K (8) per element; u and y per
    node."""
    return (4 * k * (k + 1) + 4 * k + 8) * n_el + 48 * m * n_nodes


def stencil_impl_bytes(n_el, n_nodes, m):
    """Plus the corner stack and shifted adds through U and Y (4 x 24 x 8 m per cell)."""
    return stencil_bytes(n_el, n_nodes, m) + 768 * m * n_el


def ke_impl_bytes(k, n_el, n_nodes, m):
    """Plus the materialized unpack (8 k^2, written and read) and the gathered u and scattered
    output (8 k m, written and read) per element."""
    return ke_bytes(k, n_el, n_nodes, m) + (16 * k * k + 32 * k * m) * n_el


def model(t, n_el, n_nodes, m):
    """(flops, lower-bound bytes, implementation bytes) of kernel t (stencil, hex8, hex20)."""
    if t == "stencil":
        args = (n_el, n_nodes, m)
        return stencil_flops(n_el, m), stencil_bytes(*args), stencil_impl_bytes(*args)
    k = 24 if t == "hex8" else 60
    args = (k, n_el, n_nodes, m)
    return ke_flops(k, n_el, m), ke_bytes(*args), ke_impl_bytes(*args)


def roofline(flops, nbytes, B, F):
    """Attainable GFLOP/s min(B I, F), B in GB/s, F in GFLOP/s."""
    return min(B * flops / nbytes, F)


SIZE = dict(pred=1, s8=1, u8=1, s16=2, u16=2, f16=2, bf16=2, s32=4, u32=4, f32=4)
SIZE |= dict(s64=8, u64=8, f64=8, c64=8, c128=16)
FREE = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast", "fusion"}
CALLS = re.compile(r"(?:body|condition|true_computation|false_computation)=%?([\w.\-]+)")


def _bytes(shape):
    out = 0
    for t, dims in re.findall(r"\b([a-z]+\d*)\[([\d,]*)\]", shape):
        n = 1
        for d in filter(None, dims.split(",")):
            n *= int(d)
        out += SIZE.get(t, 0) * n
    return out


def walk(text, top=5):
    """Per computation reachable from ENTRY (while, conditional and call bodies; fused
    computations are counted as fusions): fusion count, non-fused ops by opcode, and the
    largest outputs [(bytes, opcode)] (a while's output is its carried state)."""
    comps, entry, cur = {}, None, None
    for line in text.splitlines():
        if not line.startswith((" ", "}", "HloModule")) and line.rstrip().endswith("{"):
            cur = line.split("(")[0].split()[-1].lstrip("%")
            comps[cur] = []
            entry = cur if line.startswith("ENTRY") else entry
        elif cur and " = " in line:
            rhs = line.split(" = ", 1)[1]
            op = re.search(r" ([a-z][\w\-]*)\(", " " + rhs)
            if op:
                comps[cur].append((op.group(1), _bytes(rhs[: op.start()]), rhs))
    seen, todo = {}, [entry]
    while todo:
        c = todo.pop()
        if c in seen or c not in comps:
            continue
        ops = comps[c]
        seen[c] = dict(
            fusions=sum(o == "fusion" for o, _, _ in ops),
            unfused=dict(Counter(o for o, _, _ in ops if o not in FREE)),
            largest=sorted(((b, o) for o, b, _ in ops if o not in FREE - {"fusion"}),
                           reverse=True)[:top],
        )  # fmt: skip
        for o, _, rhs in ops:
            todo += CALLS.findall(rhs)
            br = re.search(r"branch_computations=\{([^}]*)\}", rhs)
            todo += [s.strip().lstrip("%") for s in br.group(1).split(",")] if br else []
            todo += re.findall(r"to_apply=%?([\w.\-]+)", rhs) if o == "call" else []
    return dict(entry=entry, computations=seen)


SETUP = """
import json, time
from jaxmodal import config
config.enable_x64()
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import dof_mask, stencil, unstructured
MAT = {{0: Material(isotropic(210e9, 0.3), 7850.0)}}
def best(f, x):
    for _ in range(2):
        jax.block_until_ready(f(x))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        jax.block_until_ready(f(x))
        ts.append(time.perf_counter() - t)
    return min(ts)
{build}
"""
STENCIL = "mesh = voxel_mesh(({n},) * 3); op = stencil(mesh, MAT)"
KE = """mesh = box_mesh(({n},) * 3, (1 / {n},) * 3, type="{t}")
op = unstructured(mesh, MAT, path="scatter", chunk={chunk})
from jaxmodal.operators.matfree import layout
b = layout(mesh, "scatter", {chunk})["blocks"][0]"""
APPLY = """v = jnp.ones((op.n_dof, {m}))
info = dict(n_el=mesh.n_el, n_nodes=len(mesh.points), apply_s=best(op.apply_K, v))
if "{t}" != "stencil":
    info.update(chunk=b["chunk"], trips=b["n_pad"] // b["chunk"])
json.dump(info, open("{{out}}", "w"))
"""
GEMM = """K = jnp.ones((24, 24)); X = jnp.ones((24, {N}))
json.dump(dict(gemm_s=best(jax.jit(lambda X: K @ X), X)), open("{{out}}", "w"))
"""


def child(build, code, large_cache=LARGE):
    r = harness.measure_json(code, SETUP.format(build=build), large_cache)
    r.pop("peak_bytes")
    print(json.dumps(r), flush=True)
    return r


def apply_row(t, n, m, B, F, chunk=None):
    build = STENCIL.format(n=n) if t == "stencil" else KE.format(n=n, t=t, chunk=chunk)
    r = child(build, APPLY.format(m=m, t=t))
    ne, nn = r["n_el"], r["n_nodes"]
    fl, lb, ib = model(t, ne, nn, m)
    g = fl / r["apply_s"] / 1e9
    roof, iroof = roofline(fl, lb, B, F), roofline(fl, ib, B, F)
    return dict(kernel=t, size=n, m=m, **r, flops=fl,
                bytes=lb, intensity=fl / lb, impl_bytes=ib, impl_intensity=fl / ib,
                roofline_gflops=roof, gflops=g, pct_roofline=100 * g / roof,
                pct_impl_roofline=100 * g / iroof)  # fmt: skip


def run_roofline(sizes=ROOF, ms=MS, probe=True):
    sysinfo = harness.system_info()
    B, F = sysinfo["triad_gbps"], sysinfo["fp64_matmul_gflops"]
    rows = []
    for t, ns in sizes.items():
        for n in ns:
            for m in ms:
                r = apply_row(t, n, m, B, F)
                if t == "stencil":
                    g = child("", GEMM.format(N=r["n_el"] * m))["gemm_s"]
                    r["gemm_gflops"] = r["flops"] / g / 1e9
                    r["pct_gemm"] = 100 * r["gflops"] / r["gemm_gflops"]
                rows.append(r)
    out = dict(system=sysinfo, B_gbps=B, F_gflops=F, large_cache=LARGE, rows=rows)
    if probe:  # chunk 512 (2.4 MB of Ke, L2-sized) against the default chunk, hex8 m = 1
        n = sizes["hex8"][0]
        base = next(r for r in rows if r["kernel"] == "hex8" and r["m"] == 1)
        small = apply_row("hex8", n, 1, B, F, chunk=512)
        faster = small["apply_s"] < 0.9 * base["apply_s"]
        out["chunk512"] = dict(default=base, small=small,
                               result="faster" if faster else "not determined")  # fmt: skip
    return out


FUSION = """
from jaxmodal.precond import amg
from benchmarks.kernels import walk
def report(f, *args):
    c = jax.jit(f).lower(*args).compile()
    return dict(walk(c.as_text()), temp_bytes=c.memory_analysis().temp_size_in_bytes,
                vector_bytes=8 * args[-1].size)
s = stencil(voxel_mesh(({n},) * 3), MAT)
box = box_mesh(({n},) * 3, (1 / {n},) * 3)
k8 = unstructured(box, MAT, path="scatter")
b20 = box_mesh(({n20},) * 3, (1 / {n20},) * 3, type="hex20")
k20 = unstructured(b20, MAT)
P = amg(k20, dof_mask(b20, {{"xmin": "xyz"}}), 1.0, b20.points)
X = lambda op: jnp.ones((op.n_dof, 4))
info = dict(stencil=report(s.apply_K, X(s)), stored_ke=report(k8.apply_K, X(k8)),
            vcycle=report(lambda P, x: P(x), P, X(k20)))
"""
DUMP = 'json.dump(info, open("{out}", "w"))'


def run_fusion(n=24, n20=12):
    r = child(FUSION.format(n=n, n20=n20), DUMP, large_cache=False)
    return r


ORDER = """
import numpy as np
from jaxmodal.mesh import morton, rcm, renumber
from jaxmodal.memory import unstructured_memory
from jaxmodal.precond import amg
from benchmarks.shell_solve_study import MESHES
mesh = (box_mesh(({n},) * 3, (1 / {n},) * 3, type="hex20") if "{case}" == "box"
        else MESHES["flat"]({n}))
if "{order}" == "rcm":
    mesh = rcm(mesh)[0]
elif "{order}" == "morton":
    mesh = morton(mesh)[0]
elif "{order}" == "shuffled":
    rng = np.random.default_rng(0)
    q = np.concatenate([s + rng.permutation(len(mesh.blocks[t])) for t, s in mesh.offsets.items()])
    mesh = renumber(mesh, rng.permutation(len(mesh.points)), q)
conn = np.concatenate(list(mesh.blocks.values()))
info = dict(n_dof=3 * len(mesh.points), span=float(np.median(np.ptp(conn, axis=1))))
op = unstructured(mesh, MAT, path="scatter")
for m in (1, 16):
    info[f"apply_m{{m}}_s"] = best(op.apply_K, jnp.ones((op.n_dof, m)))
P = amg(op, dof_mask(mesh, {{"xmin": "xyz"}}), 1.0, mesh.points)
info["vcycle_m16_s"] = best(P, jnp.ones((op.n_dof, 16)))
del P, op
if "{case}" == "box":
    need = unstructured_memory(mesh, 1, path="gather")["total"]
    info["gather_need_bytes"] = need
    if need < 0.5 * {ram}:
        g = unstructured(mesh, MAT, path="gather")
        info["gather_m1_s"] = best(g.apply_K, jnp.ones((g.n_dof, 1)))
"""


def run_orders(meshes=ORDER_MESHES, orders=ORDERS):
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    rows = []
    for case, n in meshes.items():
        for order in orders:
            r = child(ORDER.format(case=case, n=n, order=order, ram=ram), DUMP)
            rows.append(dict(case=case, size=n, order=order, **r))
    return dict(large_cache=LARGE, rows=rows)


PARTS = dict(roofline=run_roofline, fusion=run_fusion, orders=run_orders)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    p.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    a = p.parse_args(argv)
    path = Path(a.out) / "kernels.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    data = {k: v for k, v in old.items() if k in PARTS}
    for part in a.parts:
        print(f"== {part}", flush=True)
        data[part] = PARTS[part]()
    return harness.write_result("kernels", data, a.out)


if __name__ == "__main__":
    print(main())
