"""Additive Schwarz D^{-1} r = sum_i R_i^T A_i^{-1} R_i r over overlapping DOF patches of an
assembled BSR matrix, the smoother of the solid-shell AMG levels: packed inverses in size
classes, built and applied by scans over byte-sized chunks. See docs/preconditioners.md."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..operators.common import register
from ..operators.matfree import CHUNK_BYTES, _packed_index
from .jacobi import _inverse_blocks

PAD = 1.1  # a level's padded bytes stay <= PAD times its natural bytes
MAX_CLASSES = 8  # size classes per level (one compile and one scan per apply each)


def _bytes(s, itemsize=8):
    """Stored bytes of one patch of size s: the packed inverse and the int32 idx row."""
    return s * (s + 1) // 2 * itemsize + 4 * s


def layout(sizes, chunk_bytes=CHUNK_BYTES):
    """[(s, n_chunks, chunk)] per size class, s descending. One class per distinct size, then
    the adjacent pair with the least added padding is merged while there are more than
    MAX_CLASSES or the padded bytes stay <= PAD x the natural bytes. Chunks hold at most
    chunk_bytes / (8 s max(s, 24)) patches and are balanced (fewer than n_chunks pad slots)."""
    s, n = np.unique(np.asarray(sizes, np.int64), return_counts=True)
    s, n = [int(v) for v in s[::-1]], [int(v) for v in n[::-1]]
    natural = total = sum(c * _bytes(v) for v, c in zip(s, n, strict=True))
    while len(s) > 1:
        add = [n[i + 1] * (_bytes(s[i]) - _bytes(s[i + 1])) for i in range(len(s) - 1)]
        i = int(np.argmin(add))
        if len(s) <= MAX_CLASSES and total + add[i] > PAD * natural:
            break
        total += add[i]
        n[i] += n.pop(i + 1)
        s.pop(i + 1)
    out = []
    for v, c in zip(s, n, strict=True):
        nc = -(-c // max(1, chunk_bytes // (8 * v * max(v, 24))))  # gathers <= chunk_bytes k / 24
        out.append((v, nc, -(-c // nc)))
    return out


def nbytes(sizes, dtype="float64", chunk_bytes=CHUNK_BYTES):
    """Device bytes of the Schwarz over patches of these sizes (packed inverses and idx)."""
    it = np.dtype(dtype).itemsize
    return sum(nc * c * _bytes(s, it) for s, nc, c in layout(sizes, chunk_bytes))


@dataclasses.dataclass(frozen=True)
class Schwarz:
    """r -> sum_i R_i^T A_i^{-1} R_i r as a pytree callable on (n, k) blocks. Per size class,
    idx (n_chunks, chunk, s) int32 (pad index n: gathered as 0, scatter dropped) and inv
    (n_chunks, chunk, s (s + 1) / 2) packed upper triangles (zero rows and columns on pads)."""

    idx: tuple
    inv: tuple

    def __call__(self, r):
        return _apply(self, r)


register(Schwarz, [])


@jax.jit
def _apply(S, r):
    y = jnp.zeros_like(r)
    for idx, inv in zip(S.idx, S.inv, strict=True):

        def step(y, c):
            i, v = c
            x = r.at[i].get(mode="fill", fill_value=0)
            V = v[:, _packed_index(i.shape[-1])].astype(r.dtype)
            z = jnp.einsum("cab,cbk->cak", V, x)
            return y.at[i].add(z, mode="drop"), None

        y = lax.scan(step, y, (idx, inv))[0]
    return y


@functools.partial(jax.jit, donate_argnums=0)
def _invert(buf, ok, B, valid, k):
    """buf (n_chunks, c, s (s + 1) / 2) with the packed inverses of the chunk B (c, s, s)
    written as chunk k (valid (c, s): real slots; pads get the identity in the factorization and
    zero rows and columns), and ok and-ed with their finiteness."""
    inv, m = _inverse_blocks(B, valid)
    inv = inv * m[:, :, None] * m[:, None, :]
    iu = np.triu_indices(B.shape[1])
    packed = inv[:, iu[0], iu[1]]
    return lax.dynamic_update_slice(buf, packed[None], (k, 0, 0)), ok & jnp.isfinite(packed).all()


def _extract(A, keys, blk, J):
    """(c, s, s) dense patch matrices of the BSR matrix A (sorted indices, block keys
    row * nb + col): the block pairs of blk (c, m) (pad -1) looked up by key, expanded to
    DOFs and restricted to the local DOF positions J (c, s) in the expanded m b patch."""
    b = A.blocksize[0]
    nb = A.shape[0] // b
    c, m = blk.shape
    blk = blk.astype(np.int64)  # keys reach nb^2
    k = blk[:, :, None] * nb + blk[:, None, :]
    pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
    hit = (blk[:, :, None] >= 0) & (blk[:, None, :] >= 0) & (keys[pos] == k)
    F = np.zeros((c, m, m, b, b))
    F[hit] = A.data[pos[hit]]
    F = F.transpose(0, 1, 3, 2, 4).reshape(c, m * b, m * b)
    return F[np.arange(c)[:, None, None], J[:, :, None], J[:, None, :]]


def build(A, blocks, sel, chunk_bytes=CHUNK_BYTES):
    """Schwarz of the SciPy BSR matrix A (b x b blocks, sorted indices) over patches given as
    blocks (n_p, m) block indices (valid first, pad -1) and sel (n_p, m b) the kept DOFs of
    each patch's expanded blocks (False on pads). A patch above sqrt(chunk_bytes / 8) DOFs, or
    a non-finite inverse, raises ValueError."""
    b = A.blocksize[0]
    n = A.shape[0]
    keys = np.repeat(np.arange(n // b, dtype=np.int64) * (n // b), np.diff(A.indptr)) + A.indices
    size = sel.sum(1)
    bound = int(np.sqrt(chunk_bytes / 8))
    if size.max() > bound:
        raise ValueError(f"Schwarz patch of {size.max()} DOFs exceeds {bound} (chunk_bytes)")
    cls = layout(size, chunk_bytes)
    edges = np.array([c[0] for c in cls])
    which = np.searchsorted(-edges, -size, side="right") - 1
    idxs, invs = [], []
    for k, (s, nc, ch) in enumerate(cls):
        p = np.flatnonzero(which == k)
        idx = np.full((nc * ch, s), n, np.int32)
        buf, ok = jnp.zeros((nc, ch, s * (s + 1) // 2)), jnp.asarray(True)
        for o in range(0, nc * ch, ch):
            q = p[o : o + ch]
            blk = blocks[q]
            blk = blk[:, : max(-(-s // b), int((blk >= 0).sum(1).max()))]  # >= s DOF slots for J
            J = np.argsort(~sel[q, : blk.shape[1] * b], axis=1, kind="stable")[:, :s]
            valid = np.zeros((ch, s), bool)
            valid[: len(q)] = np.arange(s) < size[q, None]
            B = np.zeros((ch, s, s))
            B[: len(q)] = _extract(A, keys, blk, J)
            dof = np.take_along_axis(blk, J // b, 1) * b + J % b
            idx[o : o + len(q)] = np.where(valid[: len(q)], dof, n)
            buf, ok = _invert(buf, ok, jnp.asarray(B), jnp.asarray(valid), o // ch)
        if not bool(ok):
            raise ValueError("Schwarz patch inverses are not finite")
        idxs.append(jnp.asarray(idx.reshape(nc, ch, s)))
        invs.append(buf)
    return Schwarz(tuple(idxs), tuple(invs))
