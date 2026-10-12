"""Block-ELL SpMV against SciPy CSR."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from jaxmodal.linalg import block_ell, pcg
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import a_p, ap_operator, assemble, dof_mask, unstructured
from jaxmodal.precond import jacobi

MAT = {0: Material(isotropic(3.0, 0.3), 2.0)}


@functools.cache
def ap(type, n=(2, 2, 4), bcs=(("xmin", "xyz"),)):
    """Assembled A_P = K + M (clamped xmin) on a box; 2x2x4 has an odd node count."""
    mesh = box_mesh(n, type=type)
    free = dof_mask(mesh, dict(bcs))
    K, M = assemble(mesh, MAT)
    return a_p(K + M, free), mesh, free


def random_blocks(nb, b, dense=(), seed=0):
    """Random symmetric block-sparse matrix: a banded pattern plus dense block rows/columns."""
    rng = np.random.default_rng(seed)
    P = sp.random(nb, nb, density=3 / nb, random_state=rng) + sp.eye(nb)
    P = P.tolil()
    for i in dense:
        P[i, :] = 1
    P = (P + P.T).tocsr()
    P.data[:] = 1
    A = sp.kron(P, np.ones((b, b))).tocsr()
    A.data = rng.standard_normal(A.nnz)
    return (A + A.T).tocsr()


def check(A, E, m, tol=1e-12, dtype=np.float64):
    X = np.random.default_rng(m).standard_normal((A.shape[0], m)).astype(dtype)
    Y = E(jnp.asarray(X))
    assert Y.dtype == dtype and Y.shape == X.shape
    ref = A @ X.astype(np.float64)
    assert np.abs(np.asarray(Y) - ref).max() <= tol * np.abs(ref).max()


@pytest.mark.parametrize("m", [1, 7])
@pytest.mark.parametrize("chunk", [2048, 4])
@pytest.mark.parametrize("case", ["hex8", "hex20", "b6"])
def test_plain(case, chunk, m):
    """Test 7: plain ELL to 1e-12; chunk 4 with a block-row count that is not a multiple of 4."""
    A, b = (random_blocks(41, 6), 6) if case == "b6" else (ap(case)[0], 3)
    assert (A.shape[0] // b) % 4
    E = block_ell(A, b, sliced=False, chunk=chunk)
    assert E.n_buckets == 1 and E.inv is None and E.n_dof == A.shape[0]
    assert E.chunks == (min(chunk, -(-A.shape[0] // b // 8) * 8),)
    check(A, E, m)


def structural(A, b):
    return len(sp.bsr_array(A, blocksize=(b, b)).indices)


@pytest.mark.parametrize("b", [3, 6])
def test_sliced(b):
    """Test 8: dense rows in the middle choose sliced ELL; sliced=False matches with more
    padding. Test 9: padding = stored / structural blocks."""
    A = random_blocks(60, b, dense=(20, 31))
    E = block_ell(A, b, chunk=8)
    assert E.n_buckets > 1 and E.inv is not None
    check(A, E, 3)
    F = block_ell(A, b, sliced=False)
    assert F.n_buckets == 1 and F.padding > 1.5 > E.padding
    check(A, F, 3)
    nnz, cnt = structural(A, b), np.diff(sp.bsr_array(A, blocksize=(b, b)).indptr)
    assert F.padding == len(cnt) * cnt.max() / nnz
    assert E.padding == (2 ** np.ceil(np.log2(cnt))).sum() / nnz
    assert sum(E.rows) == len(cnt)


def test_auto():
    """sliced=None: plain for a free hex8 6^3 A_P (padding 1.35), sliced for hex20 (clamped
    nodes are width-1 rows: the clamped hex8 6^3 has plain padding 1.59)."""
    A8, A20 = ap("hex8", (6, 6, 6), ())[0], ap("hex20")[0]
    E8, E20 = block_ell(A8), block_ell(A20)
    assert E8.n_buckets == 1 and E8.padding <= 1.5
    assert E20.n_buckets > 1 and block_ell(A20, sliced=False).padding > 1.5
    assert E8.padding == (A8.shape[0] // 3) * np.diff(
        sp.bsr_array(A8, blocksize=(3, 3)).indptr
    ).max() / structural(A8, 3)
    check(A20, E20, 2)


def test_fp32():
    """Test 10: fp32 values, fp32 out for fp32 in, 1e-6 relative; fp64 in stays fp64."""
    A = ap("hex20")[0]
    E = block_ell(A, dtype="float32")
    assert all(v.dtype == jnp.float32 for v in E.vals)
    check(A, E, 3, tol=1e-6, dtype=np.float32)
    check(A, E, 3, tol=1e-6)


def test_pcg():
    """Test 11: PCG on block_ell(A_P) equals PCG on ap_operator to 1e-10."""
    A, mesh, free = ap("hex20")
    op = unstructured(mesh, MAT)
    b = jnp.asarray(np.random.default_rng(0).standard_normal((len(free), 2)) * free[:, None])
    P = jacobi(op, free, 1.0)
    x1, i1 = pcg(block_ell(A), b, P, 1e-12)
    x2, i2 = pcg(ap_operator(op, free, 1.0), b, P, 1e-12)
    assert i1["converged"].all() and i2["converged"].all()
    assert np.abs(x1 - x2).max() <= 1e-10 * np.abs(x2).max()


def test_errors():
    with pytest.raises(ValueError, match="divisible"):
        block_ell(sp.eye(10).tocsr(), 3)
    with pytest.raises(ValueError, match="dtype"):
        block_ell(sp.eye(9).tocsr(), dtype="float16")
