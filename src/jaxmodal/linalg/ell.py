"""Block-ELL SpMV on the device, with a sliced (width-bucketed) fallback. See
docs/preconditioners.md."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax

from ..operators.common import check_x64, register

SLICE_AT = 1.5  # sliced=None: plain ELL while its padding is <= SLICE_AT


@dataclasses.dataclass(frozen=True)
class BlockELL:
    """Block-ELL matrix as a pytree callable X (n_cols, m) -> A X (n_dof, m). Each bucket
    holds cols (rows_pad, w) int32 and vals (rows_pad, w, b, b) for its block rows; inv (None
    for one bucket) puts the concatenated bucket rows back in order."""

    cols: tuple
    vals: tuple
    inv: jax.Array | None
    n_dof: int  # output rows (the operator size of a square matrix)
    rows: tuple  # block rows per bucket
    chunks: tuple  # block rows per scan step, per bucket
    padding: float  # stored blocks / structural blocks (a leaf)

    @property
    def n_buckets(self):
        return len(self.cols)

    def __call__(self, X):
        return _apply(self, X)


register(BlockELL, ["n_dof", "rows", "chunks"])  # padding is a leaf: no recompile


@jax.jit
def _apply(A, X):
    m = X.shape[1]
    b = A.vals[0].shape[-1]
    Xb = X.reshape(-1, b, m)
    Y = []
    for cols, vals, r, c in zip(A.cols, A.vals, A.rows, A.chunks, strict=True):

        def step(_, cv):
            cl, vl = cv
            return None, jnp.einsum("rkij,rkjm->rim", vl.astype(X.dtype), Xb[cl])

        w = cols.shape[1]
        xs = (cols.reshape(-1, c, w), vals.reshape(-1, c, w, b, b))
        Y.append(lax.scan(step, None, xs)[1].reshape(-1, b, m)[:r])
    Y = Y[0] if A.inv is None else jnp.concatenate(Y)[A.inv]
    return Y.reshape(A.n_dof, m)


def block_ell(A, b=3, dtype="float64", sliced=None, chunk=2048):
    """Block-ELL copy of the SciPy sparse matrix A (both dimensions divisible by b; square or
    rectangular) with b x b blocks. sliced=None uses one bucket while its padding is <=
    SLICE_AT, else buckets of rows whose block counts round up to the same power of 2;
    True/False force it. A bucket of width w scans at most chunk rows, and at most
    chunk * 27 // w (>= 8) when wider than 27 blocks. dtype "float32" stores the values in
    fp32 (the apply casts them to the input's dtype)."""
    check_x64("block_ell")
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
    n = A.shape[0]
    if n % b or A.shape[1] % b:
        raise ValueError(f"A's dimensions must be divisible by b = {b}, got {A.shape}")
    B = sp.bsr_array(A, blocksize=(b, b))
    B.sort_indices()
    nb, nnz = n // b, len(B.indices)
    cnt = np.diff(B.indptr)
    width = np.full(nb, max(cnt.max(), 1))
    if sliced is None:
        sliced = nb * width[0] > SLICE_AT * max(nnz, 1)
    if sliced:
        width = 2 ** np.ceil(np.log2(np.maximum(cnt, 1))).astype(int)
    row_of = np.repeat(np.arange(nb), cnt)
    pos = np.arange(nnz) - B.indptr[row_of]
    cols, vals, rows, chunks, order, stored = [], [], [], [], [], 0
    for w in np.unique(width):
        idx = np.flatnonzero(width == w)
        r = len(idx)
        c = int(min(chunk, max(8, chunk * 27 // w), -(-r // 8) * 8))
        rp = -(-r // c) * c
        local = np.full(nb, -1)
        local[idx] = np.arange(r)
        sel = local[row_of] >= 0
        cl = np.zeros((rp, w), np.int32)  # padding entries: block column 0, zero values
        vl = np.zeros((rp, w, b, b), dtype)
        cl[local[row_of[sel]], pos[sel]] = B.indices[sel]
        vl[local[row_of[sel]], pos[sel]] = B.data[sel]
        cols.append(jnp.asarray(cl))
        vals.append(jnp.asarray(vl))
        rows.append(r)
        chunks.append(c)
        order.append(idx)
        stored += r * w
    order = np.concatenate(order)
    inv = None if len(cols) == 1 else jnp.asarray(np.argsort(order), jnp.int32)
    return BlockELL(
        tuple(cols), tuple(vals), inv, n, tuple(rows), tuple(chunks), stored / max(nnz, 1)
    )
