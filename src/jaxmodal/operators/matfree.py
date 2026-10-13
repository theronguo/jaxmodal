"""Matrix-free K and M on unstructured hex8/hex20/hex27 meshes. See docs/operators.md.

The operator is a pytree: every array is a jit argument, only sizes and flags are static.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..elements import ELEMENTS, check_elements, kernels
from ..materials import check_materials, phase_table, phases
from .common import check_nodal_mass, check_v, check_x64, node_cols, register, scaled
from .dofmap import check_int32, conn_dofs

DEFAULT_PATH = {"cpu": "scatter", "gpu": "scatter"}  # CPU from benchmarks/scatter_gather.py
CHUNK_BYTES = 32 * 2**20  # default chunk: largest power of two with c k^2 8 B <= this
R_CANDIDATES = [2**j for j in range(14, 21)]


def _pow2_floor(x):
    return 1 << (int(x).bit_length() - 1)


def _pow2_ceil(x):
    return 1 << (int(x) - 1).bit_length()


def default_chunk(t, n_el):
    k = 3 * ELEMENTS[t].N.shape[1]
    # capped at pow2ceil(n_el) so the padded size stays a power of two (an explicit chunk is
    # capped at n_el instead)
    return min(_pow2_floor(CHUNK_BYTES // (8 * k * k)), _pow2_ceil(n_el))


def _windows(conn, R, n_nodes, n_ranges, c):
    """Per range r: the element interval [start, start + W) holding every element with a node in
    [r0, r0 + R), r0 = min(r R, n_nodes - R); W is the widest interval rounded up to c."""
    n_el = len(conn)
    start, end = np.full(n_ranges, n_el), np.full(n_ranges, -1)
    e = np.broadcast_to(np.arange(n_el)[:, None], conn.shape)
    np.minimum.at(start, conn // R, e)
    np.maximum.at(end, conn // R, e)
    last = np.flatnonzero((conn >= n_nodes - R).any(1))  # the last range overlaps its predecessor
    if len(last):
        start[-1], end[-1] = min(start[-1], last[0]), max(end[-1], last[-1])
    used = end >= 0
    W = -(-int((end - start + 1)[used].max()) // c) * c
    n_pad = -(-n_el // c) * c
    return np.where(used, np.minimum(start, n_pad - W), 0), W


def layout(mesh, path=None, chunk=None, range_nodes=None):
    """Host layout shared by unstructured and unstructured_memory: resolved path, per-block chunk,
    padded size and connectivity, and (gather) R, window starts and widths, inverse map."""
    path = DEFAULT_PATH.get(jax.default_backend(), "scatter") if path is None else path
    if path not in ("scatter", "gather"):
        raise ValueError(f"path must be 'scatter' or 'gather', got {path!r}")
    for name, x in (("chunk", chunk), ("range_nodes", range_nodes)):
        if x is not None and x < 1:
            raise ValueError(f"{name} must be >= 1, got {x}")
    n_nodes = len(mesh.points)
    blocks = []
    for t, conn in mesh.blocks.items():
        c = default_chunk(t, len(conn)) if chunk is None else min(chunk, len(conn))
        n_pad = -(-len(conn) // c) * c
        e = np.arange(n_pad)
        blocks.append(
            dict(
                type=t,
                offset=mesh.offsets[t],
                n_el=len(conn),
                chunk=c,
                n_pad=n_pad,
                k=3 * conn.shape[1],
                conn=conn[np.where(e < len(conn), e, 0)],
            )
        )
    out = dict(path=path, n_nodes=n_nodes, blocks=blocks)
    if path == "scatter":
        return out
    valence = int(np.bincount(np.concatenate([c.ravel() for c in mesh.blocks.values()])).max())

    def windows(R):
        nr = -(-n_nodes // R)
        return R, nr, [_windows(mesh.blocks[b["type"]], R, n_nodes, nr, b["chunk"]) for b in blocks]

    if range_nodes is None:  # largest candidate whose per-column workspace is <= two vectors
        cands = [windows(R) for R in [R for R in R_CANDIDATES if R < n_nodes] + [n_nodes]]
        fits = [
            w
            for w in cands
            if sum(W * b["k"] for (_, W), b in zip(w[2], blocks, strict=True)) + 3 * w[0] * valence
            <= 6 * n_nodes
        ]
        R, nr, wins = fits[-1] if fits else cands[0]
    else:
        R, nr, wins = windows(min(range_nodes, n_nodes))
    out.update(R=R, n_ranges=nr, valence=valence)
    # Inverse map: per range-local node, the window-local positions of its (element, a) pairs.
    rs, nl, val, boff = [], [], [], 0
    for b, (starts, W) in zip(blocks, wins, strict=True):
        b.update(starts=starts, W=W)
        conn = mesh.blocks[b["type"]]
        e, a = np.divmod(np.arange(conn.size), conn.shape[1])
        n = conn.ravel()
        for sel, r in (
            (np.ones(n.shape, bool), n // R),
            ((n >= n_nodes - R) & (n // R < nr - 1), nr - 1),
        ):
            rr = np.broadcast_to(r, n.shape)[sel]
            rs.append(rr)
            nl.append(n[sel] - np.minimum(rr * R, n_nodes - R))
            val.append(boff + (e[sel] - starts[rr]) * b["k"] + 3 * a[sel])
        boff += W * b["k"]
    rs, nl, val = np.concatenate(rs), np.concatenate(nl), np.concatenate(val)
    key = rs * R + nl
    order = np.argsort(key, kind="stable")
    key, val = key[order], val[order]
    slot = np.arange(len(key)) - np.searchsorted(key, key)
    inv = np.full((nr * R, valence, 3), boff, np.int64)  # boff: the zero sentinel row
    inv[key, slot] = val[:, None] + np.arange(3)
    out.update(
        inv=inv.reshape(nr, R, valence, 3).transpose(0, 1, 3, 2).reshape(nr, 3 * R, valence),
        window=boff,
    )
    return out


@dataclasses.dataclass(frozen=True)
class Block:
    dofs: jax.Array  # (n_pad, k) int32
    phase: jax.Array  # (n_pad,) int32
    Ke: jax.Array | None  # stored: (n_pad, k(k+1)/2) packed upper triangle
    wdet: jax.Array | None  # stored: (n_pad, nq) w det J
    geo: jax.Array | None  # otf: (n_pad, nq, 10) J^-1 (9) and w det J
    starts: jax.Array | None  # gather: (n_ranges,) int32 window starts
    type: str
    offset: int
    n_el: int
    chunk: int
    W: int  # gather window width (0 for scatter)


@dataclasses.dataclass(frozen=True)
class Unstructured:
    blocks: tuple
    C: jax.Array | None  # otf: (P, 6, 6)
    rho: jax.Array  # (P,)
    s_K: jax.Array  # (n_el,)
    s_M: jax.Array  # (n_el,)
    nodal: jax.Array  # (n_dof,)
    inv: jax.Array | None  # gather: (n_ranges, 3R, valence) int32
    n_nodes: int
    mode: str
    mass: str
    path: str
    R: int
    n_ranges: int

    @property
    def n_dof(self):
        return 3 * self.n_nodes

    @property
    def n_el(self):
        return sum(b.n_el for b in self.blocks)

    def apply_K(self, v):
        return _apply_K(self, v)

    def apply_M(self, v):
        return _apply_M(self, v)

    def diag_K(self):
        return _diag_K(self)

    def diag_M(self):
        return _diag_M(self)

    def block_diag_K(self):
        return _block_diag_K(self)

    def scaled(self, s_K=None, s_M=None):
        """Copy with new per-element scalings (n_el,); None keeps the current one. Traceable."""
        return scaled(self, s_K, s_M)


register(Block, ["type", "offset", "n_el", "chunk", "W"])
register(Unstructured, ["n_nodes", "mode", "mass", "path", "R", "n_ranges"])


def _packed_index(k):
    """(k, k) map from (a, b) to the packed upper-triangle position."""
    idx = np.zeros((k, k), np.int32)
    idx[np.triu_indices(k)] = np.arange(k * (k + 1) // 2)
    return np.maximum(idx, idx.T)


@partial(jax.jit, static_argnums=(0, 1))
def _build(t, mode, points, conn, phase, C):
    """Per-element leaves for one block; conn (n_chunks, c, n), so the output is the padded leaf."""

    def f(args):
        cn, ph = args
        X = points[cn]
        if mode == "stored":
            k = 3 * cn.shape[1]
            iu = np.triu_indices(k)
            _, wdet = kernels.jacobian(t, X)
            return kernels.stiffness(t, X, C[ph])[:, iu[0], iu[1]], wdet
        invJ, wdet = kernels.jacobian(t, X)
        return (jnp.concatenate([invJ.reshape(*wdet.shape, 9), wdet[..., None]], -1),)

    out = lax.map(f, (conn, phase))
    return tuple(a.reshape(-1, *a.shape[2:]) for a in out)


def unstructured(
    mesh,
    materials,
    orientations=None,
    mode="stored",
    mass="consistent",
    nodal_mass=None,
    path=None,
    chunk=None,
    range_nodes=None,
):
    """Matrix-free operator with apply_K, apply_M, diag_K, diag_M and scaled methods."""
    if mode not in ("stored", "otf"):
        raise ValueError(f"mode must be 'stored' or 'otf', got {mode!r}")
    check_x64("unstructured")
    index, pairs = phases(mesh)
    check_materials(pairs, materials, orientations)
    check_elements(mesh, mass)
    n_nodes = len(mesh.points)
    nodal_mass = check_nodal_mass(nodal_mass, n_nodes)
    lay = layout(mesh, path, chunk, range_nodes)
    sizes = {"n_dof": 3 * n_nodes, "phase": len(pairs)}
    sizes |= {f"{b['type']}_dofs": b["n_pad"] * b["k"] for b in lay["blocks"]}
    if lay["path"] == "gather":
        sizes |= {"inv": lay["inv"].size, "window": lay["window"] + 1}
    check_int32(**sizes)
    C, rho = phase_table(pairs, materials, orientations)
    points = jnp.asarray(mesh.points)
    blocks = []
    for b in lay["blocks"]:
        conn, c = b["conn"], b["chunk"]
        e = np.arange(b["n_pad"])
        ph = index[b["offset"] + np.where(e < b["n_el"], e, 0)].astype(np.int32)
        out = _build(
            b["type"],
            mode,
            points,
            jnp.asarray(conn.reshape(-1, c, conn.shape[1]), jnp.int32),
            jnp.asarray(ph.reshape(-1, c)),
            C,
        )
        jax.block_until_ready(out)  # build temporaries freed before the next block or apply
        stored = mode == "stored"
        blocks.append(
            Block(
                dofs=jnp.asarray(conn_dofs(conn)),
                phase=jnp.asarray(ph),
                Ke=out[0] if stored else None,
                wdet=out[1] if stored else None,
                geo=None if stored else out[0],
                starts=jnp.asarray(b["starts"], jnp.int32) if "starts" in b else None,
                type=b["type"],
                offset=b["offset"],
                n_el=b["n_el"],
                chunk=c,
                W=b.get("W", 0),
            )
        )
    gather = lay["path"] == "gather"
    return Unstructured(
        blocks=tuple(blocks),
        C=None if mode == "stored" else C,
        rho=rho,
        s_K=jnp.ones(mesh.n_el),
        s_M=jnp.ones(mesh.n_el),
        nodal=jnp.asarray(np.repeat(nodal_mass, 3)),
        inv=jnp.asarray(lay["inv"], jnp.int32) if gather else None,
        n_nodes=n_nodes,
        mode=mode,
        mass=mass,
        path=lay["path"],
        R=lay.get("R", 0),
        n_ranges=lay.get("n_ranges", 0),
    )


# ---------------------------------------------------------------- element kernels on a chunk d


def _wdet(b, d):
    return d["wdet"] if b.Ke is not None else d["geo"][..., 9]


def _k_apply(op, b, d, v):
    u = v[d["dofs"]]  # (c, k, m)
    if op.mode == "stored":
        k = u.shape[1]
        Ke = d["Ke"][:, _packed_index(k)]
        return d["s_K"][:, None, None] * jnp.einsum("eab,ebm->eam", Ke, u)
    c, k, m = u.shape
    G = kernels.gradients(b.type, d["geo"][..., :9].reshape(c, -1, 3, 3))  # (c, nq, n, 3)
    S = jnp.asarray(kernels.S)
    H = jnp.einsum("eaim,eqaj->eqijm", u.reshape(c, k // 3, 3, m), G)
    sig = jnp.einsum("evw,wij,eqijm->eqvm", op.C[d["phase"]], S, H)
    T = jnp.einsum("vij,eqvm,eq->eqijm", S, sig, d["geo"][..., 9])
    y = jnp.einsum("eqijm,eqaj->eaim", T, G).reshape(c, k, m)
    return d["s_K"][:, None, None] * y


def _bmat(b, d):
    """otf: the strain-displacement matrices B of a chunk."""
    geo = d["geo"]
    return kernels.bmatrix(kernels.gradients(b.type, geo[..., :9].reshape(len(geo), -1, 3, 3)))


def _k_diag(op, b, d):
    if op.mode == "stored":
        k = b.dofs.shape[1]
        i = np.arange(k)
        diag = d["Ke"][:, _packed_index(k)[i, i]]
    else:
        B = _bmat(b, d)
        diag = jnp.einsum("eq,eqvk,evw,eqwk->ek", d["geo"][..., 9], B, op.C[d["phase"]], B)
    return (d["s_K"][:, None] * diag)[..., None]


def _k_blocks(op, b, d):
    """(c, k, 3): row 3a + i, column j holds Ke[3a + i, 3a + j]."""
    k = b.dofs.shape[1]
    cols = node_cols(k)
    if op.mode == "stored":
        blocks = d["Ke"][:, _packed_index(k)[np.arange(k)[:, None], cols]]
    else:
        B = _bmat(b, d)
        blocks = jnp.einsum(
            "eq,eqvk,evw,eqwkj->ekj", d["geo"][..., 9], B, op.C[d["phase"]], B[..., cols]
        )
    return d["s_K"][:, None, None] * blocks


def _m_scale(op, d):
    return op.rho[d["phase"]] * d["s_M"]


def _m_apply(op, b, d, v):
    u = v[d["dofs"]]
    c, k, m = u.shape
    N = jnp.asarray(ELEMENTS[b.type].N)
    Nu = jnp.einsum("qb,ebim->eqim", N, u.reshape(c, k // 3, 3, m))
    y = jnp.einsum("eq,qa,eqim->eaim", _wdet(b, d), N, Nu).reshape(c, k, m)
    return _m_scale(op, d)[:, None, None] * y


def element_matrices(op, block, chunk):
    """(s_K K_e, rho s_M M_e), each (c, k, k), for one chunk of a block (a slice of
    _block_data); M_e consistent or lumped as op.mass. Padded elements give zeros."""
    k = block.dofs.shape[1]
    if op.mode == "stored":
        Ke = chunk["Ke"][:, _packed_index(k)]
    else:
        B = _bmat(block, chunk)
        Ke = jnp.einsum("eq,eqvk,evw,eqwl->ekl", chunk["geo"][..., 9], B, op.C[chunk["phase"]], B)
    wdet = _wdet(block, chunk)
    if op.mass == "lumped":
        mn = jax.vmap(jnp.diag)(kernels.lumped_nodal(block.type, wdet))
    else:
        N = jnp.asarray(ELEMENTS[block.type].N)
        mn = jnp.einsum("eq,qa,qb->eab", wdet, N, N)
    Me = jnp.einsum("eab,ij->eaibj", mn, jnp.eye(3)).reshape(Ke.shape)
    return chunk["s_K"][:, None, None] * Ke, _m_scale(op, chunk)[:, None, None] * Me


def _m_diag(op, b, d):
    wdet = _wdet(b, d)
    if op.mass == "lumped":
        nodal = kernels.lumped_nodal(b.type, wdet)
    else:
        nodal = jnp.einsum("eq,qa->ea", wdet, jnp.asarray(ELEMENTS[b.type].N) ** 2)
    return jnp.repeat(_m_scale(op, d)[:, None] * nodal, 3, axis=1)[..., None]


# ---------------------------------------------------------------- assembly paths


def _block_data(op, b):
    n_pad = b.dofs.shape[0]
    d = {"dofs": b.dofs, "phase": b.phase}
    for name in ("Ke", "wdet", "geo"):
        if getattr(b, name) is not None:
            d[name] = getattr(b, name)
    for name in ("s_K", "s_M"):  # padded elements get s = 0
        d[name] = jnp.pad(getattr(op, name)[b.offset : b.offset + b.n_el], (0, n_pad - b.n_el))
    return d


def _assemble(op, kern, m):
    """sum_e P_e kern(chunk) over all elements, (n_dof, m); kern(b, d) -> (c, k, m)."""
    out = jnp.zeros((op.n_dof, m))
    data = [_block_data(op, b) for b in op.blocks]
    if op.path == "scatter":
        for b, d in zip(op.blocks, data, strict=True):
            n_chunks = b.dofs.shape[0] // b.chunk
            xs = jax.tree.map(lambda a, n=n_chunks, c=b.chunk: a.reshape(n, c, *a.shape[1:]), d)

            def step(out, d, b=b):
                return out.at[d["dofs"]].add(kern(b, d)), None

            out, _ = lax.scan(step, out, xs)
        return out

    def range_step(out, r):
        Y = []
        for b, d in zip(op.blocks, data, strict=True):
            start = b.starts[r]

            def inner(_, i, b=b, d=d, start=start):
                sl = {
                    k: lax.dynamic_slice_in_dim(a, start + i * b.chunk, b.chunk)
                    for k, a in d.items()
                }
                return None, kern(b, sl)

            _, y = lax.scan(inner, None, jnp.arange(b.W // b.chunk, dtype=jnp.int32))
            Y.append(y.reshape(-1, m))
        Y = jnp.concatenate(Y + [jnp.zeros((1, m))])  # last row: the sentinel
        r0 = jnp.minimum(r * op.R, op.n_nodes - op.R)
        return lax.dynamic_update_slice(out, Y[op.inv[r]].sum(1), (3 * r0, jnp.int32(0))), None

    out, _ = lax.scan(range_step, out, jnp.arange(op.n_ranges, dtype=jnp.int32))
    return out


def _check(op, v=None):
    if op.mode == "stored" and op.C is not None:
        raise ValueError("stored mode bakes C into Ke: rebuild the operator to change materials")
    if v is not None:
        check_v(v, op.n_dof)


@jax.jit
def _apply_K(op, v):
    _check(op, v)
    return _assemble(op, lambda b, d: _k_apply(op, b, d, v), v.shape[1])


@jax.jit
def _apply_M(op, v):
    _check(op, v)
    if op.mass == "lumped":
        return _diag_M(op)[:, None] * v
    return _assemble(op, lambda b, d: _m_apply(op, b, d, v), v.shape[1]) + op.nodal[:, None] * v


@jax.jit
def _diag_K(op):
    _check(op)
    return _assemble(op, lambda b, d: _k_diag(op, b, d), 1)[:, 0]


@jax.jit
def _block_diag_K(op):
    _check(op)
    return _assemble(op, lambda b, d: _k_blocks(op, b, d), 3).reshape(op.n_nodes, 3, 3)


@jax.jit
def _diag_M(op):
    _check(op)
    return _assemble(op, lambda b, d: _m_diag(op, b, d), 1)[:, 0] + op.nodal
