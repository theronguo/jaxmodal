"""Device memory estimators (bytes). See docs/operators.md and docs/eigensolvers.md."""

import functools
import math
import os
import weakref

import jax
import numpy as np

from .eigen.api import (
    OPERATOR_OPTIONS,
    PRECOND_OPTIONS,
    _check_options,
    check_band,
    check_polish_options,
    precond_label,
    resolve_precond,
)
from .eigen.lobpcg import _block_size, _capacity, band_defaults  # shared with the solver
from .eigen.shift_invert import _chunk, _up
from .elements import ELEMENTS
from .materials import phases
from .operators import dof_mask, rigid_modes
from .operators.common import check_shell, is_count
from .operators.matfree import layout, resolve_path
from .operators.voxel import grid_layout
from .precond import schwarz
from .precond.amg import SHELL_BLOCKS, unsupported_shell
from .precond.gmg import _ell_rows, _islands
from .precond.pmg import _w8
from .precond.shell import _columns, _fibers

# XLA temporaries of the build kernel per element (memory_analysis, exactly linear in chunk).
# ssh8: an envelope, 1.02 x the 32-ply value (44.6 KB at 1 ply, 56.4 KB at 32; above about 35
# plies it underestimates by about 0.32 KB per ply).
BUILD_BYTES = {
    "stored": {"hex8": 29.95e3, "hex20": 241.06e3, "hex27": 322.70e3, "ssh8": 57.5e3},
    "otf": {"hex8": 1.22e3, "hex20": 4.10e3, "hex27": 4.10e3},
}
GATHER_FLOATS = 15  # stencil: per-cell slab buffers besides the 24 k m GEMM ones (memory_analysis)
OVERHEAD = 10e6  # fixed JAX runtime overhead beyond the formulas, measured (hex20 30^3, hex8 60^3)


# otf apply_K temporaries per element: a + b (m - 1) bytes, an upper envelope of memory_analysis
# at m = 1, 4, 16 (XLA's fusion of the strain/stress fields varies with m).
OTF_BYTES = {"hex8": (6.1e3, 1.28e3), "hex20": (41.0e3, 3.17e3), "hex27": (54.6e3, 2.55e3)}


_CACHE = weakref.WeakKeyDictionary()  # mesh -> {key: value}; entries die with the mesh


def _per_mesh(f):
    """Cache f(mesh, *args) per mesh (meshes are treated as immutable; Mesh hashes by
    identity). Cached values must not reference the mesh."""

    @functools.wraps(f)
    def g(mesh, *args):
        d = _CACHE.setdefault(mesh, {})
        key = (f.__name__, *args)
        if key not in d:
            d[key] = f(mesh, *args)
        return d[key]

    return g


@_per_mesh
def _layout_sizes(mesh, path, chunk, range_nodes):
    """The scalars of layout(...) that unstructured_memory reads."""
    lay = layout(mesh, path, chunk, range_nodes)
    keys = ("type", "k", "chunk", "n_pad", "W")
    out = dict(path=lay["path"], n_nodes=lay["n_nodes"],
               blocks=[{k: b[k] for k in keys if k in b} for b in lay["blocks"]])  # fmt: skip
    if lay["path"] == "gather":
        out.update(R=lay["R"], n_ranges=lay["n_ranges"], valence=lay["valence"],
                   inv=lay["inv"].size)  # fmt: skip
    return out


def _chunk_temps(mode, t, c, m):
    """XLA temporaries of one chunk: apply_K (stored: u_e, y_e, unpacked K_e; otf: fitted) and,
    in otf mode, diag_K, which forms B like the stored build."""
    k = 3 * ELEMENTS[t].N.shape[1]
    if mode == "stored":
        return 8 * c * (2 * k * m + k * k)
    a, b = OTF_BYTES[t]
    return c * max(a + b * (m - 1), BUILD_BYTES["stored"][t])


def unstructured_memory(
    mesh, m, mode="stored", mass="consistent", path=None, chunk=None, range_nodes=None
):
    """Estimated peak bytes of unstructured(...) plus apply_K and apply_M on (n_dof, m) blocks.

    Returns storage, vectors, workspace, build, overhead and
    total = overhead + storage + max(build, vectors + workspace). Later estimators add
    storage + workspace and count overhead once.
    """
    check_shell(mesh, mode)
    if "ssh8" in mesh.blocks and mass == "lumped":
        raise ValueError("solid-shell elements are consistent-mass only")
    lay = _layout_sizes(mesh, resolve_path(path), chunk, range_nodes)
    n_dof, n_el = 3 * lay["n_nodes"], mesh.n_el
    gather = lay["path"] == "gather"
    storage = 8 * (n_dof + 2 * n_el)  # nodal, s_K, s_M
    host = 8 * n_el  # phase index
    kernel, temps, window = 0, 0, 0
    for b in lay["blocks"]:
        n, k, c, n_pad = b["k"] // 3, b["k"], b["chunk"], b["n_pad"]
        nq = len(ELEMENTS[b["type"]].weights)
        per_el = k * (k + 1) // 2 + (36 if b["type"] == "ssh8" else nq)  # Ke + (me or wdet)
        per_el = per_el if mode == "stored" else 10 * nq
        storage += n_pad * (4 * k + 4 + 8 * per_el)  # dofs, phase, Ke + (wdet or me) or geo
        host += 8 * n_pad * (n + 1)  # padded connectivity and phase, alive during the build
        kernel = max(kernel, c * BUILD_BYTES[mode][b["type"]])
        temps = max(temps, _chunk_temps(mode, b["type"], c, m))
        if gather:
            storage += 4 * lay["n_ranges"]
            window += 8 * b["W"] * k * m
    if gather:
        inv = lay["inv"]
        storage += 4 * inv
        host += 8 * inv  # int64 inverse map, alive until the int32 copy
        workspace = window + max(8 * 3 * lay["R"] * lay["valence"] * m, temps)
    else:
        workspace = temps
    out = dict(
        storage=storage,
        vectors=3 * 8 * n_dof * m,  # v and two outputs (XLA frees the last one asynchronously)
        workspace=workspace,
        build=host + kernel,
        overhead=OVERHEAD,
    )
    out["total"] = out["overhead"] + storage + max(out["build"], out["vectors"] + workspace)
    return {k: int(v) for k, v in out.items()}


def stencil_memory(mesh, m, mass="consistent", slab=None):
    """Estimated peak bytes of stencil(...) plus apply_K and apply_M on (n_dof, m) blocks.

    Same keys and composition rule as unstructured_memory.
    """
    n_dof, n_el = 3 * len(mesh.points), mesh.n_el
    if mesh.voxel is None:
        raise ValueError("stencil_memory needs a voxel mesh: mesh.voxel is None (no voxel grid)")
    shape = mesh.voxel.shape
    axes, _, slab, L = grid_layout(shape, slab)
    B, A = shape[axes[1]], shape[axes[2]]
    grids = (L + 1) * (B + 1) * (A + 1) + L * B * A
    k = 2 if len(phases(mesh)[1]) == 1 else 3  # U and Y, plus the phase accumulator
    out = dict(
        storage=4 * (grids + n_el) + 8 * (n_dof + 2 * n_el),  # node, cell, phase; nodal, s_K, s_M
        vectors=3 * 8 * n_dof * m,
        workspace=8 * slab * B * A * (24 * k * m + GATHER_FLOATS),  # memory_analysis within 10%
        build=8 * grids,  # int64 grids on the host
        overhead=OVERHEAD,
    )
    out["total"] = (
        out["overhead"] + out["storage"] + max(out["build"], out["vectors"] + out["workspace"])
    )
    return {k: int(v) for k, v in out.items()}


# LOBPCG step temp + aliased state = A blocks + f * operator workspace: A is the large-n value,
# f the upper envelope of memory_analysis over hex8/20/27 at 6.6k-1.08M DOF, m = 7-48
# (docs/eigensolvers.md).
STEP_FIT = {"9-block": (16, 2.5), "low": (11, 1.75)}


def _pc_builds(pc):
    """(device, host) build bytes of a preconditioner estimate: AMG builds on the host."""
    return pc["build"] - pc.get("host_build", 0), pc.get("host_build", 0)


def lobpcg_memory(
    op_memory, n_dof, n_modes, chunk=None, guards=None, low_memory=False, basis_dtype="float64",
    n_rigid=0, precond_memory=None, band=False, capacity=None,
):  # fmt: skip
    """Estimated peak bytes of lobpcg(...) (band: a band solve expecting n_modes modes, whose
    host V holds one more block). op_memory: m -> the operator estimate at block width
    m, e.g. lambda m: unstructured_memory(mesh, m); precond_memory: m -> the preconditioner
    estimate (e.g. lambda m: gmg_memory(mesh, m)), None for the default Jacobi. capacity: the
    basis columns (default _capacity's for n_modes).

    step = max(9-block, low, polish) (low_memory: max(low, polish); polish for every AMG
    family). Returns step, basis, rigid, host, precond (its storage),
    total = overhead + storage + precond + basis + rigid + host + max(build, precond build, step)
    and per memory space (a GPU; the CPU uses total): device_total (the device terms and the
    device builds), host_total (the host peak: max(operator build, V + AMG build, V + 2 chunks
    of a band basis refill), the AMG aggregates inside the fit test's modelled factor) and their
    exact parts device_exact (operator storage, basis, R and M R) and host_exact (V).
    """
    chunk, guards, k = _block_size(n_modes, chunk, guards)
    c, cap = _capacity(n_modes, chunk, k, low_memory)
    cap = cap if capacity is None else capacity
    op = op_memory(k)
    pc = precond_memory(k) if precond_memory else dict(storage=0, build=0, step=0, step_low=0)
    b, ws = 8 * n_dof * k, op["workspace"]
    A, f = STEP_FIT["9-block"]
    Al, fl = STEP_FIT["low"]
    step = Al * b + fl * ws + pc["step_low"]  # the low step, also after a 9-block switch
    if not low_memory:
        step = max(step, A * b + f * ws + pc["step"])
    if "polish" in pc:  # the polish's PCG (AMG)
        step = max(step, fl * ws + pc["polish"])
    out = dict(
        step=step,
        basis=n_dof * cap * (4 if basis_dtype == "float32" else 8),
        rigid=2 * 8 * n_dof * n_rigid,  # R and M R, fp64
        # freed step output and the fp64 modes on the host; a chunked solve adds 4 blocks of
        # transients between chunks (start block, seed, allocator slack: up to about 6.5
        # blocks measured, varying between runs; docs/open_items.md)
        host=(5 if cap else 1) * b + 8 * n_dof * n_modes + (b if band else 0),
        precond=pc["storage"],
    )
    out["total"] = (
        op["overhead"] + op["storage"] + out["precond"] + out["basis"] + out["rigid"]
        + out["host"] + max(op["build"], pc["build"], out["step"])
    )  # fmt: skip
    V = 8 * n_dof * n_modes + (b if band else 0)  # the host modes
    pc_dev, pc_host = _pc_builds(pc)
    out.update(
        device_exact=op["storage"] + out["basis"] + out["rigid"],
        host_exact=V,
        device_total=op["overhead"] + op["storage"] + out["precond"] + out["basis"]
        + out["rigid"] + out["host"] - V + max(out["step"], pc_dev),
        # a band growth refills the basis from V: a host chunk and its F -> C copy
        host_total=max(op["build"], V + pc_host, V + (2 * 8 * n_dof * c if band else 0)),
    )  # fmt: skip
    return {k: int(v) for k, v in out.items()}


# GMG (docs/preconditioners.md). Extra LOBPCG step blocks with the V-cycle over Jacobi: upper
# envelope of memory_analysis of the steps at 16^3-87^3 (m = 15).
GMG_STEP = {
    ("9-block", "float64"): 4.3,
    ("9-block", "float32"): 0.0,
    ("low", "float64"): 9.0,
    ("low", "float32"): 4.5,
}
# Fine-level Chebyshev degree > 2, fp64 blocks: the smoother loop carries about one more fine
# vector, +1.07 (fp64) / +0.60 (fp32 V-cycle) blocks of V-cycle temporaries at 8^3-87^3; the
# steps rise 1.07 (fp64, from 48^3 up) and up to 2.06 (fp64 low step, 8^3-32^3, inside the
# GMG_STEP slack), fp32 steps by at most 0.6.
GMG_LOOP = {"float64": 1.2, "float32": 0.7}
# V-cycle temporaries beyond the stencil's slab workspace, fp64 blocks: upper envelope of
# memory_analysis at 16^3-87^3 before PR 13c's Chebyshev loop (m = 15; 10.6-14.9 fp64, 5.0-7.5
# fp32); kept: the workspace estimate is 7-39% above the temporaries with the loop (32^3-64^3).
GMG_WORK = {"float64": 15.0, "float32": 7.5}
GMG_BUILD = 1650  # setup process peak per fine node (1.12 GB at 87^3)
# Extra count-recurrence bytes with the V-cycle over Jacobi: 8 n (a b + c), the upper envelope
# of memory_analysis of _recurrence at 16^3-64^3, b = 4-32 (fp64 V-cycle).
GMG_COUNT = (7.8, 10.0)


def _gmg_grid(mesh):
    """(cell, node) activity of the grid gmg builds from (grid_ids: voxel grid cropped of its
    empty trailing layers along the scan axis), in (x, y, z) order."""
    vx = mesh.voxel
    nx, ny, nz = vx.shape
    cell = vx.element_map.reshape(nz, ny, nx).transpose(2, 1, 0)
    node = vx.node_map.reshape(nz + 1, ny + 1, nx + 1).transpose(2, 1, 0)
    scan = grid_layout(vx.shape)[0][0]
    used = np.flatnonzero((cell >= 0).any(axis=tuple(d for d in range(3) if d != scan)))
    n = used[-1] + 1
    return np.take(cell, np.arange(n), scan), np.take(node, np.arange(n + 1), scan) >= 0


def _smoother_bytes(shape, n, n_groups, dt):
    """Smoother storage of a level: inverse group blocks, and the mask (k = 1) or group index
    arrays."""
    k = int(np.prod([c + 1 for c in shape if c <= 2]))
    if k == 1:
        return 12 * n * dt
    return n_groups * (3 * k) ** 2 * dt + 4 * (n_groups * k + n)


def gmg_memory(mesh, m, dtype="float64", s_K=None, degree=2):
    """Estimated bytes of gmg(stencil(mesh, ...), free, sigma, dtype=dtype, degree=degree)
    applied to (n_dof, m) blocks: storage (an upper bound: the fine level exact, coarse levels as
    full boxes; with s_K (n_el,) the island deflation), workspace (V-cycle temporaries), build
    (the setup peak), overhead, step and step_low (extra LOBPCG step bytes over Jacobi), count
    (extra count-recurrence bytes over Jacobi). Same
    composition rule as the operator estimators."""
    if mesh.voxel is None:
        raise ValueError("gmg_memory needs a voxel mesh: mesh.voxel is None (no voxel grid)")
    dt = 4 if dtype == "float32" else 8
    cell, node = _gmg_grid(mesh)
    shape, n_nodes = cell.shape, len(mesh.points)
    n_dof = 3 * n_nodes
    stopped = tuple(d for d in range(3) if shape[d] <= 2)
    n_groups = int(node.any(axis=stopped).sum()) if stopped else n_nodes
    storage = 8 * n_dof + _smoother_bytes(shape, n_nodes, n_groups, dt)  # free, fine smoother
    n_fine = n_nodes
    while True:
        cs = tuple(c if c <= 2 else -(-c // 2) for c in shape)
        nc = int(np.prod([c + 1 for c in cs]))
        # transfer: gf, gc int32 node maps; per fine axis node j0, j1 int64 and w0, w1; scalars
        storage += 4 * (n_fine + nc) + sum(c + 1 for c in shape) * (16 + 2 * dt) + 64
        if max(cs) <= 2:
            break
        rp = _ell_rows(nc)[1]
        k = int(np.prod([c + 1 for c in cs if c <= 2]))
        storage += 27 * rp * (9 * dt + 4) + _smoother_bytes(cs, nc, nc // k, dt)
        shape, n_fine = cs, nc
    storage += (3 * nc) ** 2 * 8  # coarsest Cholesky factor
    b = 8 * n_dof * m
    loop = GMG_LOOP[dtype] * b if degree > 2 else 0
    work = GMG_WORK[dtype] * b + dt / 8 * stencil_memory(mesh, m)["workspace"] + loop
    if s_K is not None:
        islands = _islands(cell, np.asarray(s_K))
        if islands:
            conn = mesh.blocks["hex8"]
            rows = 0
            for e in islands:
                ring = np.unique(conn[np.isin(conn, np.unique(conn[e])).any(1)])
                rows += 3 * len(ring)
            rows = max(256, 1 << (rows - 1).bit_length())
            storage += 104 * rows + 96**2 * 8
            work += 3 * 8 * rows * m
    out = dict(
        storage=storage,
        workspace=work,
        build=GMG_BUILD * n_nodes,  # fp32: the cast copy (< 300 B/node) fits under it
        overhead=OVERHEAD,
        step=GMG_STEP[("9-block", dtype)] * b + loop,
        step_low=GMG_STEP[("low", dtype)] * b + loop,
        count=8 * n_dof * (GMG_COUNT[0] * m + GMG_COUNT[1]) + loop,
    )
    return {k: int(v) for k, v in out.items()}


# AMG (docs/preconditioners.md), calibrated on box meshes at the default max_coarse = 1000.
# Family "hex8": all-hex8 meshes (no p-level); "p": the rest. AMG levels, smoothers and
# transfers below the p-level per corner DOF: upper envelope (+2-4%) of hex20 8^3-44^3, hex27
# 6^3-24^3, hex8 12^3-68^3 without the coarsest factor (874-2344 B measured).
AMG_STORAGE = {"hex8": 1200, "p": 2450}
# Extra LOBPCG step bytes over Jacobi: 8 n (a m + c) + 8 m K (n DOF, block m), the minimal
# envelope (linear programming) of memory_analysis at the sizes above, m = 15 and 25, plus
# 1.5 in c (0.06-0.1 blocks) against rounding.
AMG_STEP = {
    ("hex8", "9-block"): (4.94, 115.4, 71_400),
    ("hex8", "low"): (9.17, 107.2, 384_682),
    ("p", "9-block"): (4.80, 42.0, 163_889),
    ("p", "low"): (9.57, 28.5, 227_703),
}
# Setup host peak: a fixed part plus bytes per corner DOF by element type (the largest present),
# measured on hex20 12^3-44^3, hex27 12^3-24^3, hex8 24^3-68^3: 3.1-4.8 kB beyond the fixed
# part, growing about 0.65 kB per decade of n_c; the constants extrapolate that trend to 10M DOF
# (hex27: two points only, steeper; see docs/open_items.md).
AMG_BUILD = (45e6, {"hex8": 4.6e3, "hex20": 5.5e3, "hex27": 6.0e3})
# Chebyshev degree > 2: the smoother loop's body keeps its own n-independent buffers (block-ELL
# scan gathers), +77k-103k DOF-equivalents in K measured at degree 3-6 (hex8/20/27, 6^3-16^3).
AMG_LOOP_K = 120_000
# Extra count-recurrence bytes with the V-cycle over Jacobi: 8 n (a b + c), the upper envelope
# of memory_analysis of _recurrence on hex20 12^3-28^3, b = 4-32 (below 20k DOF the fixed
# AMG buffers dominate and the fit does not apply).
AMG_COUNT = (11.5, 45.0)
# Solid-shell family (meshes with ssh8 blocks, docs/preconditioners.md). Storage beyond the
# level-0 Schwarz (exact from the mesh scan): bytes per shell DOF (T_0 and the rest), per shell
# fiber group (the coarse levels) and per hex-only DOF (its level-0 and coarse share); for the
# fit's geometries (a T junction 3/3 at 96^2, out of the fit, is 0.977x its owned bytes).
SHELL_STORAGE = {"shell": 500, "fiber": 6300, "hex": 1250}
# Setup host peak: beta + storage + gamma per shell / hex-only DOF (upper envelope, 45k-1.58M,
# up to 10 layers: a 20-layer stack is 1.3% above it).
SHELL_BUILD = (150e6, {"shell": 1400, "hex": 3300})
# pc terms 8 n (a m + c) + K0 (K0 bytes, independent of m): the upper envelope (linear
# programming, c, K0 >= 0) of memory_analysis minus lobpcg_memory's STEP_FIT part (A b + f ws;
# the polish with the low f) on flat 64^2-512^2, web 64^2, block 32^3 and 64^3, stack x10 64^2
# and x3 128^2 (25k-1.58M DOF), m = 6-120, and T 48^2 at m = 15, rounded up, c + 0.5 against
# rounding. count: _recurrence minus COUNT_STEP's part, b = 4-50 on flat 64^2-512^2, stack x3
# 128^2 and block 64^3, and si_memory's minres (SI_MINRES b + count) at b = 6 on the flat plates.
SHELL_STEP = {
    "9-block": (9.6, 17.1, 70e6),
    "low": (13.9, 13.9, 106e6),
    "polish": (26.7, 4.0, 128e6),
    "count": (15.2, 5.8, 116e6),
}
# fp32 V-cycles (amg dtype "float32", PR 18c; benchmarks/amg_fp32_study.py): the fitted storage
# times one ratio per family (the maximum over the storage grid, rounded up), and the step
# terms refitted as above (hex: 8 n (a m + c) + 8 m K; shell: 8 n (a m + c) + K0) with
# asymptote rows on the long series. The polish's PCG keeps fp64 vectors around the fp32
# cycle, so hex gains a polish term. Counts and builds stay fp64.
STORAGE32 = {"hex8": 0.636, "p": 0.548, "shell": 0.574}
AMG_STEP32 = {
    ("hex8", "9-block"): (3.26, 55.4, 351_800),
    ("hex8", "low"): (8.74, 53.1, 433_600),
    ("hex8", "polish"): (20.44, 40.1, 505_500),
    ("p", "9-block"): (3.64, 21.9, 525_400),
    ("p", "low"): (8.70, 14.5, 576_100),
    ("p", "polish"): (20.70, 4.0, 830_500),
}
# fp64 hex AMG polish: the fp32 polish term (with its f_l ws) scaled by POLISH64, chosen from
# the 11 measured fp64 polish rows of amg_fp32_study.json (max(low, polish) 1.04-1.15 of
# measured); the degree > 2 loop term is added unscaled.
POLISH64 = 1.05
SHELL_STEP32 = {
    "9-block": (5.2, 16.0, 37e6),
    "low": (11.1, 18.9, 65e6),
    "polish": (23.5, 10.0, 89e6),
}


@_per_mesh
def _shell_scan(mesh):
    """(level-0 patch sizes in DOFs, shell DOFs, hex-only DOFs, shell fiber groups) of the
    solid-shell AMG: the fibers and patches of amg's own scan, on the mesh's connectivity."""
    n_nodes = len(mesh.points)
    E = mesh.blocks["ssh8"]
    group = _fibers(E, n_nodes)
    sizes = 3 * (_columns(E, group, n_nodes) >= 0).sum(1)
    shell = np.zeros(n_nodes, bool)
    shell[E.ravel()] = True
    return sizes, 3 * int(shell.sum()), 3 * int((~shell).sum()), len(np.unique(group[shell]))


def _shell_amg_memory(mesh, m, max_coarse, degree, dtype):
    if unsupported_shell(mesh.blocks):
        raise ValueError(SHELL_BLOCKS)
    sizes, n_s, n_h, n_f = _shell_scan(mesh)
    n = 3 * len(mesh.points)
    st = SHELL_STORAGE
    fitted = st["shell"] * n_s + st["fiber"] * n_f + st["hex"] * n_h
    coarse = 8 * min(n, max(max_coarse, 1000)) ** 2
    storage = schwarz.nbytes(sizes) + fitted + coarse
    beta, g = SHELL_BUILD
    hb = beta + storage + g["shell"] * n_s + g["hex"] * n_h  # the setup runs in fp64
    fp32 = dtype == "float32"
    if fp32:
        storage = schwarz.nbytes(sizes, dtype) + STORAGE32["shell"] * fitted + coarse
    loop = 8 * m * AMG_LOOP_K if degree > 2 else 0

    def term(key):
        a, c, K0 = SHELL_STEP32[key] if fp32 and key != "count" else SHELL_STEP[key]
        return 8 * n * (a * m + c) + K0 + loop

    out = dict(storage=storage, build=hb, host_build=hb, step=term("9-block"),
               step_low=term("low"), polish=term("polish"), count=term("count"))  # fmt: skip
    return {k: int(v) for k, v in out.items()}


@_per_mesh
def _amg_scan(mesh):
    """(family, corner DOFs n_c, p-level table bytes) of amg_memory: the mesh scan alone."""
    n_nodes = len(mesh.points)
    blocks = [c for c in mesh.blocks.values() if len(c)]
    corner = np.unique(np.concatenate([c[:, :8].ravel() for c in blocks]))
    fam, tables = ("hex8" if mesh.blocks.keys() == {"hex8"} else "p"), 0
    if fam == "p":  # NodalTransfer: (int32, fp64) parent / child tables
        # w: parents per fine node; child.max(): fine nodes interpolating from one corner
        w = max(int((_w8(c.shape[1]) > 0).sum(1).max()) for c in blocks)
        pairs = []
        for c in blocks:
            a, j = np.nonzero(_w8(c.shape[1]) > 0)
            pairs.append(c[:, a].astype(np.int64).ravel() * n_nodes + c[:, j].ravel())
        child = np.unique(np.unique(np.concatenate(pairs)) % n_nodes, return_counts=True)[1]
        tables = 12 * (w * n_nodes + int(child.max()) * len(corner))
    return fam, 3 * len(corner), tables


def amg_memory(mesh, m, max_coarse=1000, degree=2, dtype="float64"):
    """Estimated bytes of amg(unstructured(mesh, ...), free, sigma, mesh.points, degree=degree,
    max_coarse=max_coarse, dtype=dtype) applied to (n_dof, m) blocks: storage (the p-level
    exact from the mesh, the AMG levels per corner DOF, the coarsest factor), build and
    host_build (the setup host peak, fp64 for both dtypes), step and step_low (extra LOBPCG
    step bytes over Jacobi), count (extra count-recurrence bytes, an fp64 build) and polish
    (the polish's PCG; fp64 hex: POLISH64 x the fp32 term). Calibrated on box meshes at
    max_coarse 1000 (the shell family on flat, stacked, web and block meshes); the coarse term
    uses max(max_coarse, 1000) (an upper bound, loose above 1000)."""
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
    if "ssh8" in mesh.blocks:
        return _shell_amg_memory(mesh, m, max_coarse, degree, dtype)
    n_nodes = len(mesh.points)
    fam, n_c, tables = _amg_scan(mesh)
    n_dof = 3 * n_nodes
    fp32 = dtype == "float32"
    # free mask (fp64), node-block Jacobi inverses and mask (dtype); the tables stay fp64
    storage = (72 if fp32 else 120) * n_nodes + tables
    storage += (STORAGE32[fam] if fp32 else 1) * AMG_STORAGE[fam] * n_c
    storage += 8 * min(n_c, max(max_coarse, 1000)) ** 2
    beta, alpha = AMG_BUILD
    loop = AMG_LOOP_K if degree > 2 else 0
    hb = beta + max(alpha[t] for t in mesh.blocks) * n_c

    def term(key, table=AMG_STEP32 if fp32 else AMG_STEP, scale=1):
        a, c, K = table[(fam, key)]
        return scale * (8 * n_dof * (a * m + c) + 8 * m * K) + 8 * m * loop

    out = dict(
        storage=storage,
        build=hb,
        host_build=hb,  # pyamg runs on the host
        step=term("9-block"),
        step_low=term("low"),
        count=8 * n_dof * (AMG_COUNT[0] * m + AMG_COUNT[1]) + 8 * m * loop,
        polish=term("polish", AMG_STEP32, 1 if fp32 else POLISH64),
    )
    return {k: int(v) for k, v in out.items()}


def _no_polish(pc, m):
    return {k: v for k, v in pc(m).items() if k != "polish"}


def _models(mesh, mass, precond, precond_dtype, operator_options, precond_options,
            polish_options=None):  # fmt: skip
    """(op_memory, precond_memory) of solve_modes' operator and preconditioner: m -> estimate
    (no polish term with polish_options {"on": False})."""
    name = resolve_precond(mesh, precond, precond_dtype)
    voxel = mesh.voxel is not None
    op_opts = _check_options(
        "operator", operator_options, OPERATOR_OPTIONS["stencil" if voxel else "unstructured"]
    )
    label = precond_label(precond, name)
    pc_opts = _check_options(f"{label} precond", precond_options, PRECOND_OPTIONS[name])
    for key in ("degree", "max_coarse"):
        if key in pc_opts and not is_count(pc_opts[key]):
            raise ValueError(f"{key} must be an int >= 1, got {pc_opts[key]!r}")
    pc = None
    if name == "gmg":
        pc = functools.partial(
            gmg_memory, mesh, dtype=precond_dtype, degree=pc_opts.get("degree", 2)
        )
    elif name == "amg":
        pc = functools.partial(
            amg_memory, mesh, max_coarse=pc_opts.get("max_coarse", 1000),
            degree=pc_opts.get("degree", 2), dtype=precond_dtype,
        )  # fmt: skip
    if not check_polish_options(polish_options, name, label).get("on", True):
        pc = functools.partial(_no_polish, pc)
    est = stencil_memory if voxel else unstructured_memory
    return functools.partial(est, mesh, mass=mass, **op_opts), pc


def solve_modes_memory(
    mesh, bcs, n_modes=None, *, mass="consistent", chunk=None, guards=None,
    basis_dtype="float64", low_memory=False, precond="auto", precond_dtype="float64",
    n_rigid=None, operator_options=None, precond_options=None, band=None, polish_options=None,
):  # fmt: skip
    """Estimated peak bytes of solve_modes(mesh, ..., bcs, n_modes, ...) with the same options:
    the operator, preconditioner and lobpcg estimates composed (lobpcg_memory's dict).
    operator_options and precond_options as in solve_modes (coarse_degree and ratio change no
    memory). n_rigid=None counts the rigid modes left by bcs on the host (builds the dense rigid
    modes: about 20 GB at 100M DOF; pass n_rigid for large meshes). band=(0, f_max): the band
    solve expecting n_modes modes (n_modes and chunk default as band_defaults gives them, not
    the ladder's possibly halved chunk); its count fits by construction (solve_modes sizes the
    probe block from this estimate). polish_options as in solve_modes ({"on": False} drops the
    polish term)."""
    if band is not None and check_band(band)[0] > 0:
        raise ValueError("solve_modes_memory needs band=(0, f_max): no estimate of f1 > 0 yet")
    if band is None and n_modes is None:
        raise ValueError("pass n_modes or band")
    op_memory, pc = _models(
        mesh, mass, precond, precond_dtype, operator_options, precond_options, polish_options
    )
    free = dof_mask(mesh, bcs)
    if n_rigid is None:
        n_rigid = rigid_modes(mesh, free).shape[1]
    if band is not None:
        n_modes, chunk = band_defaults(n_modes, chunk, guards, int(free.sum()) - n_rigid)
    return lobpcg_memory(
        op_memory, 3 * len(mesh.points), n_modes, chunk, guards, low_memory, basis_dtype,
        n_rigid, pc, band is not None,
    )  # fmt: skip


# Count (docs/eigensolvers.md#completeness-count). Recurrence step with Jacobi (memory_analysis
# temporaries of _recurrence): workspace + 8 n (a b + c), and on the stencil at least
# 2 workspace + 8 n a_dup b (the slab workspace appears twice while a slab covers most of the
# grid); upper envelope of hex20 12^3-28^3, ssh8 120^2-200^2 and voxel 16^3-64^3, b = 4-32
# (meshes below about 20k DOF, where one chunk holds the whole mesh, are outside it). Plus w0
# and u (2 blocks).
COUNT_STEP = {"unstructured": (10.3, 3.0, 0.0), "stencil": (10.2, 1.0, 2.2)}
# Sampling: the y PCG on M_P, workspace + 6 vectors per probe (memory_analysis, hex20 20^3,
# voxel 64^3), plus g and y.
COUNT_SAMPLE = 8.0


def count_memory(
    op_memory, n_dof, b, *, voxel=False, precond_memory=None, n_rigid=0, chunk=64,
    w_dtype="float64",
):  # fmt: skip
    """Estimated peak device bytes of count(...) at probe block b. op_memory: m -> the operator
    estimate (unstructured_memory or stencil_memory, voxel=True for the latter);
    precond_memory: m -> gmg_memory or amg_memory (its "count" term), None for Jacobi. W stays
    on the host: one chunk of `chunk` columns is on the device at a time.

    Returns storage, sampling, step, projector, total = overhead + storage + max(phases) and
    the per-space entries of lobpcg_memory (all device; exact: operator storage, R and M R)."""
    op = op_memory(b)
    pc = precond_memory(b) if precond_memory else dict(storage=0, count=0)
    v, ws = 8 * n_dof, op["workspace"]
    a, c, a_dup = COUNT_STEP["stencil" if voxel else "unstructured"]
    step = ws + v * (a * b + c)
    if a_dup:
        step = max(step, 2 * ws + v * (a_dup * b + c))
    out = dict(
        storage=op["storage"] + pc["storage"] + 2 * v * n_rigid,  # R and M R
        sampling=ws + v * COUNT_SAMPLE * b,
        step=step + pc["count"] + 2 * v * b,
        # two chunks in flight (plus their fp64 upcasts), Y (donated), M Y, the transfer's
        # source and the M apply
        projector=2 * n_dof * chunk * (12 if w_dtype == "float32" else 8) + 3 * v * b + ws,
    )
    peak = max(out["sampling"], out["step"], out["projector"])
    out["total"] = op["overhead"] + out["storage"] + peak
    out.update(device_total=out["total"], host_total=0, host_exact=0,
               device_exact=op["storage"] + 2 * v * n_rigid)  # fmt: skip
    return {k: int(x) for k, x in out.items()}


def probe_block(budget, n_probes, op_memory, n_dof, **kwargs):
    """The largest probe block b <= n_probes with count_memory(op_memory, n_dof, b,
    **kwargs)["total"] <= budget; raises if even b = 1 does not fit."""

    def total(b):
        return count_memory(op_memory, n_dof, b, **kwargs)["total"]

    b = _largest(lambda b: total(b) <= budget, 1, 1, n_probes)  # total is non-decreasing in b
    if b is None:
        one = total(1) / 1e9
        raise ValueError(f"budget {budget / 1e9:.2f} GB is below one probe ({one:.2f} GB)")
    return b


def band_probe_block(mesh, bcs, n_modes, n_probes, host_bytes, *, n_rigid, **options):
    """The probe block of a band solve's count: probe_block within solve_modes_memory(mesh,
    bcs, n_modes, n_rigid=n_rigid, **options) minus host_bytes (the converged modes and AMG
    aggregates on the host; host and device share memory on the Mac), the multigrid at lam_c in
    fp64. Raises ValueError when even one probe does not fit."""
    total = solve_modes_memory(mesh, bcs, n_modes, n_rigid=n_rigid, **options)["total"]
    op_memory, pc = _models(
        mesh, options.get("mass", "consistent"), options.get("precond", "auto"), "float64",
        options.get("operator_options"), options.get("precond_options"),
        options.get("polish_options"),
    )  # fmt: skip
    return probe_block(
        total - host_bytes, n_probes, op_memory, 3 * len(mesh.points),
        voxel=mesh.voxel is not None, precond_memory=pc, n_rigid=n_rigid,
    )  # fmt: skip


# Shift-invert slice (docs/eigensolvers.md#shift-invert-slices), memory_analysis fits (b = 6
# and 12, c = 24): MINRES with Jacobi: workspace + 15.5 blocks of width b (13.9-15.4 at hex20
# 20^3-32^3 and voxel 48^3-80^3; up to 18.3 on small voxel grids, 15k-108k DOF; including the
# best iterate), plus its right-hand side and solution; a multigrid adds at most its `count` term
# (0.5-0.96 of it).
# Reorthogonalization: workspace + 4 blocks. Ritz residuals (width b, the basis read in chunks of
# c): workspace + 6 blocks + 1 chunk (4.0-5.3 blocks at 107k-1.6M DOF; 11 on the voxel 32^3
# grid, where the slab workspace appears twice).
SI_MINRES = 17.5
SI_REORTHO = 4.0
SI_RITZ = 6.0


def si_memory(
    op_memory, n_dof, capacity, n_modes, block=6, precond_memory=None, n_rigid=0, n_deflate=0,
    host_window=None,
):  # fmt: skip
    """Estimated peak bytes of si_slice(...) with `capacity` Lanczos columns inside a band solve
    whose host store holds n_modes columns. op_memory, precond_memory: m -> the operator /
    preconditioner estimate (as lobpcg_memory); n_rigid: columns of R (R and M R are stored);
    n_deflate: columns of W on the device; host_window: W's host columns (default n_deflate;
    si_band: the window and si_checked's restart copy).

    Returns storage (operator, preconditioner, basis, R and M R), host (the window, the store and
    the projected matrix), the phases minres, reortho, ritz, end (one device chunk of the host
    V, while Q is alive), total = overhead + storage + host + max(operator build, preconditioner
    build, phases), and the per-space entries of lobpcg_memory (device: storage and phases;
    host: host + the host builds; exact: operator storage, basis, R, M R, window and store)."""
    c = _chunk(block)
    capacity = -(-capacity // c) * c
    op_b = op_memory(block)
    pc = precond_memory(block) if precond_memory else dict(storage=0, build=0, count=0)
    v, b = 8 * n_dof, 8 * n_dof * block
    out = dict(
        storage=op_b["storage"] + pc["storage"] + v * (-(-n_deflate // c) * c + capacity)
        + 2 * v * n_rigid,
        host=v * (n_deflate if host_window is None else host_window) + v * n_modes
        + 8 * capacity**2,
        minres=op_b["workspace"] + SI_MINRES * b + pc["count"],
        reortho=op_b["workspace"] + SI_REORTHO * b,
        ritz=op_b["workspace"] + SI_RITZ * b + v * c,
        end=v * (2 + c),
    )  # fmt: skip
    phases = max(out["minres"], out["reortho"], out["ritz"], out["end"])
    out["total"] = (
        op_b["overhead"] + out["storage"] + out["host"] + max(op_b["build"], pc["build"], phases)
    )
    pc_dev, pc_host = _pc_builds(pc)
    out.update(
        device_exact=out["storage"] - pc["storage"],
        host_exact=out["host"] - 8 * capacity**2,
        device_total=op_b["overhead"] + out["storage"] + max(pc_dev, phases),
        host_total=out["host"] + max(op_b["build"], pc_host),
    )
    return {k: int(x) for k, x in out.items()}


# ---------------------------------------------------------------- budgets (solve_modes band)

# The fit test (docs/eigensolvers.md#memory-budget): exact column arrays grow 5% with the
# allocator, the modelled terms run up to 9-16% low (the roadmap's 20% validation band).
FIT_EXACT, FIT_MODEL = 1.05, 1.2


def _fits(est, budget, host_budget=None, host_extra=0):
    """The fit test on one estimate, host_extra exact host bytes added: host_budget None is
    unified memory (one check on total), else device_total against budget and host_total against
    host_budget."""
    if host_budget is None:
        e = est["device_exact"] + est["host_exact"]
        return FIT_EXACT * (e + host_extra) + FIT_MODEL * (est["total"] - e) <= budget
    dev = FIT_EXACT * est["device_exact"] + FIT_MODEL * (est["device_total"] - est["device_exact"])
    host = FIT_EXACT * (est["host_exact"] + host_extra) + FIT_MODEL * (
        est["host_total"] - est["host_exact"])  # fmt: skip
    return dev <= budget and host <= host_budget


def _largest(fits, step, lo, hi):
    """The largest multiple of step in [lo, hi] with fits True (fits monotone: True below some
    value), by bisection; None when none fits."""
    lo, hi = -(-lo // step), hi // step
    if lo > hi or not fits(lo * step):
        return None
    while hi > lo:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(mid * step) else (lo, mid - 1)
    return int(lo * step)


def _c_max(op_memory, pc, n_dof, *, chunk, guards, low, basis_dtype, n_rigid, n_avail, budget,
           host_budget, need=None):  # fmt: skip
    """C_max: the largest LOBPCG band basis (a multiple of the projector chunk c_L, at most the
    free space up(n_avail + k, c_L)) whose band solve passes the fit test, from the initial basis
    up(4 chunk + k, c_L), or up(need, c_L) when a known depth needs less; None when that does
    not fit."""
    _, guards, k = _block_size(chunk, chunk, guards)
    c = _capacity(0, chunk, k, low)[0]
    cap = _up(n_avail + k, c)

    def fits(C):
        est = lobpcg_memory(op_memory, n_dof, C, chunk, guards, low, basis_dtype, n_rigid, pc,
                            True, capacity=C)  # fmt: skip
        return _fits(est, budget, host_budget)

    return _largest(fits, c, min(_up(min(4 * chunk + k, need or math.inf), c), cap), cap)


def _slice_capacity(op_memory, pc64, n_dof, *, n_store, block, n_rigid, budget, host_budget):
    """The slice capacity C (Lanczos and deflation columns, a multiple of c = _chunk(block)): the
    largest passing the fit test with the device window W = 2c inside C, the host window and
    si_checked's restart copy 2 (C - 2 block) and the store of n_store columns, between the
    floor up(2c + 2 block, c) and the need cap up(1.25 (1.8 n_store + 84) + W, c); None when the
    floor does not fit."""
    c = _chunk(block)
    w = 2 * c

    def fits(C):
        est = si_memory(op_memory, n_dof, C - w, n_store, block, pc64, n_rigid, n_deflate=w,
                        host_window=2 * (C - 2 * block))  # fmt: skip
        return _fits(est, budget, host_budget)

    need = _up(int(np.ceil(1.25 * (1.8 * n_store + 84))) + w, c)
    return _largest(fits, c, _up(2 * c + 2 * block, c), need)


def _count_block(op_memory, pc64, n_dof, *, voxel, n_rigid, n_store, window, n_probes, budget,
                 host_budget):  # fmt: skip
    """The largest probe block <= n_probes of a slice count passing the fit test with the store
    (n_store columns) and the host window (window columns) as exact host terms; None when one
    probe does not fit."""

    def fits(b):
        est = count_memory(op_memory, n_dof, b, voxel=voxel, precond_memory=pc64, n_rigid=n_rigid)
        return _fits(est, budget, host_budget, 8 * n_dof * (n_store + window))

    return _largest(fits, 1, 1, n_probes)


def _default_budget():
    """(budget, host_budget): on a GPU its free pool (bytes_limit - bytes_in_use) and 0.8 x
    physical RAM; on the CPU (unified), or another device without bytes_limit in memory_stats()
    (Metal, or memory_stats() raising or returning None), 0.8 x RAM and None. A "gpu" device
    without bytes_limit raises: RAM is no bound for its memory."""
    ram = 0.8 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    dev = jax.devices()[0]
    if dev.platform == "cpu":
        return ram, None
    try:
        st = dev.memory_stats()
    except Exception:  # some plugins do not implement it
        st = None
    if not st or "bytes_limit" not in st:  # Metal: no pool limit, a unified pool as on the CPU
        if dev.platform == "gpu":
            raise ValueError("the GPU reports no bytes_limit; pass memory_budget")
        return ram, None
    return st["bytes_limit"] - st.get("bytes_in_use", 0), ram
