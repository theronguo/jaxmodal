"""Device memory estimators (bytes). See docs/operators.md and docs/eigensolvers.md."""

import functools

import numpy as np

from .eigen.api import resolve_precond
from .eigen.lobpcg import _block_size, _capacity  # shared with the solver on purpose
from .elements import ELEMENTS
from .materials import phases
from .operators import dof_mask, rigid_modes
from .operators.matfree import layout
from .operators.voxel import grid_layout
from .precond.gmg import _ell_rows, _islands
from .precond.pmg import _w8

# XLA temporaries of the build kernel per element (memory_analysis, exactly linear in chunk).
BUILD_BYTES = {
    "stored": {"hex8": 29.95e3, "hex20": 241.06e3, "hex27": 322.70e3},
    "otf": {"hex8": 1.22e3, "hex20": 4.10e3, "hex27": 4.10e3},
}
GATHER_FLOATS = 15  # stencil: per-cell slab buffers besides the 24 k m GEMM ones (memory_analysis)
OVERHEAD = 10e6  # fixed JAX runtime overhead beyond the formulas, measured (hex20 30^3, hex8 60^3)


# otf apply_K temporaries per element: a + b (m - 1) bytes, an upper envelope of memory_analysis
# at m = 1, 4, 16 (XLA's fusion of the strain/stress fields varies with m).
OTF_BYTES = {"hex8": (6.1e3, 1.28e3), "hex20": (41.0e3, 3.17e3), "hex27": (54.6e3, 2.55e3)}


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
    lay = layout(mesh, path, chunk, range_nodes)
    n_dof, n_el = 3 * lay["n_nodes"], mesh.n_el
    gather = lay["path"] == "gather"
    storage = 8 * (n_dof + 2 * n_el)  # nodal, s_K, s_M
    host = 8 * n_el  # phase index
    kernel, temps, window = 0, 0, 0
    for b in lay["blocks"]:
        n, k, c, n_pad = b["k"] // 3, b["k"], b["chunk"], b["n_pad"]
        nq = len(ELEMENTS[b["type"]].weights)
        per_el = k * (k + 1) // 2 + nq if mode == "stored" else 10 * nq
        storage += n_pad * (4 * k + 4 + 8 * per_el)  # dofs, phase, Ke + wdet or geo
        host += 8 * n_pad * (n + 1)  # padded connectivity and phase, alive during the build
        kernel = max(kernel, c * BUILD_BYTES[mode][b["type"]])
        temps = max(temps, _chunk_temps(mode, b["type"], c, m))
        if gather:
            storage += 4 * lay["n_ranges"]
            window += 8 * b["W"] * k * m
    if gather:
        inv = lay["inv"].size
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


def lobpcg_memory(
    op_memory, n_dof, n_modes, chunk=None, guards=None, low_memory=False, basis_dtype="float64",
    n_rigid=0, precond_memory=None,
):  # fmt: skip
    """Estimated peak bytes of lobpcg(...). op_memory: m -> the operator estimate at block width
    m, e.g. lambda m: unstructured_memory(mesh, m); precond_memory: m -> the preconditioner
    estimate (e.g. lambda m: gmg_memory(mesh, m)), None for the default Jacobi.

    Returns step, basis, rigid, host, precond (its storage) and
    total = overhead + storage + precond + basis + rigid + host + max(build, precond build, step).
    """
    chunk, guards, k = _block_size(n_modes, chunk, guards)
    _, cap = _capacity(n_modes, chunk, k, low_memory)
    op = op_memory(k)
    pc = precond_memory(k) if precond_memory else dict(storage=0, build=0, step=0, step_low=0)
    b = 8 * n_dof * k
    A, f = STEP_FIT["low" if low_memory else "9-block"]
    out = dict(
        step=A * b + f * op["workspace"] + pc["step_low" if low_memory else "step"],
        basis=n_dof * cap * (4 if basis_dtype == "float32" else 8),
        rigid=2 * 8 * n_dof * n_rigid,  # R and M R, fp64
        # freed step output and the fp64 modes on the host; a chunked solve adds 4 blocks of
        # transients between chunks (start block, seed, allocator slack: up to about 6.5
        # blocks measured, varying between runs; docs/open_items.md)
        host=(5 if cap else 1) * b + 8 * n_dof * n_modes,
        precond=pc["storage"],
    )
    out["total"] = (
        op["overhead"] + op["storage"] + out["precond"] + out["basis"] + out["rigid"]
        + out["host"] + max(op["build"], pc["build"], out["step"])
    )  # fmt: skip
    return {k: int(v) for k, v in out.items()}


# GMG (docs/preconditioners.md). Extra LOBPCG step blocks with the V-cycle over Jacobi: upper
# envelope of memory_analysis of the steps at 16^3-87^3 (m = 15).
GMG_STEP = {
    ("9-block", "float64"): 5.1,
    ("9-block", "float32"): 0.0,
    ("low", "float64"): 9.9,
    ("low", "float32"): 4.5,
}
# V-cycle temporaries beyond the stencil's slab workspace, fp64 blocks: upper envelope of
# memory_analysis at 16^3-87^3 (m = 15; 10.6-14.9 fp64, 5.0-7.5 fp32).
GMG_WORK = {"float64": 15.0, "float32": 7.5}
GMG_BUILD = 1650  # setup process peak per fine node (1.12 GB at 87^3)


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


def gmg_memory(mesh, m, dtype="float64", s_K=None):
    """Estimated bytes of gmg(stencil(mesh, ...), free, sigma, dtype=dtype) applied to
    (n_dof, m) blocks: storage (an upper bound: the fine level exact, coarse levels as full
    boxes; with s_K (n_el,) the island deflation), workspace (V-cycle temporaries), build (the
    setup peak), overhead, step and step_low (extra LOBPCG step bytes over Jacobi). Same
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
    work = GMG_WORK[dtype] * b + dt / 8 * stencil_memory(mesh, m)["workspace"]
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
        step=GMG_STEP[("9-block", dtype)] * b,
        step_low=GMG_STEP[("low", dtype)] * b,
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
# an envelope (+5%) of hex20 12^3-44^3, hex27 12^3-24^3, hex8 24^3-68^3 (4.4-5.5 kB measured).
AMG_BUILD = (45e6, {"hex8": 4.1e3, "hex20": 4.9e3, "hex27": 5.8e3})


def amg_memory(mesh, m):
    """Estimated bytes of amg(unstructured(mesh, ...), free, sigma, mesh.points) at the default
    max_coarse applied to (n_dof, m) blocks: storage (the p-level exact from the mesh, the AMG
    levels per corner DOF, the coarsest factor), build (the setup host peak), step and
    step_low (extra LOBPCG step bytes over Jacobi). Calibrated on box meshes."""
    n_nodes = len(mesh.points)
    blocks = [c for c in mesh.blocks.values() if len(c)]
    corner = np.unique(np.concatenate([c[:, :8].ravel() for c in blocks]))
    n_c, n_dof = 3 * len(corner), 3 * n_nodes
    fam = "hex8" if mesh.blocks.keys() == {"hex8"} else "p"
    storage = 120 * n_nodes  # free mask, node-block Jacobi inverses and mask (fp64)
    if fam == "p":  # NodalTransfer: (int32, fp64) parent / child tables
        # w: parents per fine node; child.max(): fine nodes interpolating from one corner
        w = max(int((_w8(c.shape[1]) > 0).sum(1).max()) for c in blocks)
        pairs = []
        for c in blocks:
            a, j = np.nonzero(_w8(c.shape[1]) > 0)
            pairs.append(c[:, a].astype(np.int64).ravel() * n_nodes + c[:, j].ravel())
        child = np.unique(np.unique(np.concatenate(pairs)) % n_nodes, return_counts=True)[1]
        storage += 12 * (w * n_nodes + child.max() * len(corner))
    storage += AMG_STORAGE[fam] * n_c + 8 * min(n_c, 1000) ** 2
    beta, alpha = AMG_BUILD
    a9, c9, K9 = AMG_STEP[(fam, "9-block")]
    al, cl, Kl = AMG_STEP[(fam, "low")]
    out = dict(
        storage=storage,
        build=beta + max(alpha[t] for t in mesh.blocks) * n_c,
        step=8 * n_dof * (a9 * m + c9) + 8 * m * K9,
        step_low=8 * n_dof * (al * m + cl) + 8 * m * Kl,
    )
    return {k: int(v) for k, v in out.items()}


def solve_modes_memory(
    mesh, bcs, n_modes, *, mass="consistent", chunk=None, guards=None, basis_dtype="float64",
    low_memory=False, precond="auto", precond_dtype="float64", n_rigid=None,
):  # fmt: skip
    """Estimated peak bytes of solve_modes(mesh, ..., bcs, n_modes, ...) with the same options:
    the operator, preconditioner and lobpcg estimates composed (lobpcg_memory's dict).
    n_rigid=None counts the rigid modes left by bcs on the host (builds the dense rigid modes:
    about 20 GB at 100M DOF; pass n_rigid for large meshes)."""
    pc = None
    name = resolve_precond(mesh, precond, precond_dtype)
    if name == "gmg":
        pc = functools.partial(gmg_memory, mesh, dtype=precond_dtype)
    elif name == "amg":
        pc = functools.partial(amg_memory, mesh)
    if mesh.voxel is not None:
        op_memory = functools.partial(stencil_memory, mesh, mass=mass)
    else:
        op_memory = functools.partial(unstructured_memory, mesh, mass=mass)
    if n_rigid is None:
        n_rigid = rigid_modes(mesh, dof_mask(mesh, bcs)).shape[1]
    return lobpcg_memory(
        op_memory, 3 * len(mesh.points), n_modes, chunk, guards, low_memory, basis_dtype,
        n_rigid, pc,
    )  # fmt: skip
