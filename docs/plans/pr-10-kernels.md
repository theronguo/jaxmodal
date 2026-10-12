# PR 10 plan: linear algebra kernels and smoothers

Roadmap PR 10 (M3, depends on PR 5):

- Contents:
  - PCG and MINRES;
  - block-ELL SpMV, with the padding ratio reported and a sliced-ELL fallback;
  - 3×3 node-block Jacobi, with the node blocks computed from the element kernels;
  - Chebyshev smoother with eigenvalue estimation (safety factor 1.1);
  - all applied to $A_P$.
- Docs: `docs/preconditioners.md`.
- Tests: listed under [Tests](#tests).

The kernels serve these later PRs:

| PR | uses |
|---|---|
| 11 (GMG) | coarse levels in block-ELL; Chebyshev smoothing over node-block Jacobi; PCG iteration tests; fp32 V-cycles |
| 12–13 (p-MG, AMG) | the same, with 6×6 coarse blocks from smoothed aggregation with 6 near-null modes |
| 14 (counting) | blocks of PCG probe solves |
| 15 (interior bands) | MINRES with the explicit-residual stopping rule |

**Deviations from the roadmap:** none. Principle 2's operator interface gains a sixth method, `block_diag_K()`, which the roadmap's "node blocks computed from the element kernels" requires; the principle's wording is updated (Docs). Decisions the roadmap leaves open are made below.

**Sign convention.** `ap_operator(op, free, sigma)` builds $K + \sigma M$. The shift-invert operator of the roadmap, $K - \sigma_b M$ with $\sigma_b$ inside the spectrum, is `ap_operator(op, free, -sigma_b)`, and its SPD preconditioner is built at $+\sigma_b$ (e.g. `block_jacobi(op, free, sigma_b)`).

## Measured facts (prototypes `scratchpad/p10/`; jax 0.11.2, fp64, M3 Max CPU)

**Problem.**
- hex20 box $6\times5\times4$ (2,199 DOF), isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$, `xmin` clamped.
- $A_P = m(K + \sigma M)m + s(I - m)$, Jacobi preconditioner, 3 right-hand sides.
- Reference: `spsolve` on the PR 4 `a_p`.

**PCG.**

| σ | κ($A_P$) | tol | iterations | max rel. error in $x$ |
|---|---|---|---|---|
| $\lambda_1$ | $6.3\cdot10^3$ | $10^{-8}$ | 289 | $3.2\cdot10^{-9}$ |
| $\lambda_1$ | $6.3\cdot10^3$ | $10^{-12}$ | 350 | $2.7\cdot10^{-13}$ |
| $100\lambda_1$ | $3.9\cdot10^2$ | $10^{-12}$ | 136 | $6.7\cdot10^{-13}$ |

At tol $10^{-14}$ the explicit residual reaches $1.3$–$4.6\cdot10^{-14}$. The recursive residual meets tol, but the explicit one ends up to 4.6× above it: this is the floor, and it needs a defined exit.

**MINRES** on the indefinite $A_P(K - \sigma M)$, with σ between $\lambda_6$ and $\lambda_7$ and Jacobi of $K + \sigma M$:

| tol | iterations | max rel. error in $x$ |
|---|---|---|
| $10^{-8}$ | 340 | $9.2\cdot10^{-9}$ |
| $10^{-10}$ | 380 | $2.2\cdot10^{-10}$ |
| $10^{-12}$ | 420 | $2.2\cdot10^{-12}$ |

At tol $10^{-14}$ the explicit residual stalls at about $4\cdot10^{-14}$, and without a stagnation rule the run goes to the cap (20,000 iterations).

**Lanczos λ_max estimate** of $D^{-1}A_P$, $D$ = Jacobi, from one random masked start vector. The table gives $1.1\,\tilde\lambda_\max/\lambda_\max$ over seeds 0–4, against `eigsh(which="LA")` on the free blocks:

| steps | $\sigma = \lambda_1$ |
|---|---|
| 10 | 1.092–1.097 |
| 20 | 1.0975–1.0997 |
| 30 | 1.0985–1.1000 |

The estimate approaches $\lambda_\max$ from below and stays within 0.8% of it.

**Chebyshev** (interval $[\hat\lambda/30, \hat\lambda]$, $\hat\lambda = 1.1\tilde\lambda_\max$, over Jacobi):
- Symmetric to $\le 2.2\cdot10^{-15}$ (relative $|u^\top Sv - v^\top Su|$) for degrees 1–4.
- As a PCG preconditioner at $\sigma = \lambda_1$: 289 / 160 / 109 / 85 iterations for degrees 1 / 2 / 3 / 4. Degree 1 equals Jacobi.

**Block-ELL SpMV** on the assembled hex8 $A_P$, $44^3$ (273k DOF, 2.30M 3×3 blocks, padding 1.07):

| layout | $m = 1$ | $m = 15$ | $m = 25$ | temp at $m = 25$ |
|---|---|---|---|---|
| scan over the 27 slots | 9 ms | 71 ms | 86 ms | small |
| row-chunked, 512 block rows per chunk | 13 ms | 43 ms | 55 ms | 56 MB |
| row-chunked, 2048 | 10 ms | 30 ms | 45 ms | 61 MB |
| row-chunked, 8192 | 9 ms | 29 ms | 40 ms | 197 MB |
| row-chunked, 32768 | 9 ms | 24 ms | 36 ms | 654 MB |
| SciPy CSR (host) | 12 ms | 102 ms | 95 ms | — |

- The vector block at $m = 25$ is 55 MB, so at 2048 block rows per chunk the row-chunked temp is about one vector block.
- The error against SciPy is $\le 10^{-15}$. On a $12^3$ hex8 box the padding is 1.27.

## Design

### Modules

| module | contents |
|---|---|
| `linalg/krylov.py` | `pcg`, `minres` |
| `linalg/ell.py` | `BlockELL`, `block_ell` |
| `precond/jacobi.py` (new package `precond/`) | `jacobi` (moved from `eigen/lobpcg.py`; `jaxmodal.eigen.jacobi` stays as a re-export) and `block_jacobi` |
| `precond/chebyshev.py` | `lmax_estimate`, `chebyshev` |
| `operators/constraints.py` | `ap_operator`, the matrix-free $A_P$ |
| `operators/matfree.py`, `operators/voxel.py` | a sixth operator method, `block_diag_K()` |

Every public entry checks that x64 is enabled (`check_x64`, the flag, not the dtype of the data), and everything is traceable. The callables run in the dtype of their input where they can (PR 11's fp32 V-cycles cast the leaves with `tree_map`; Not covered). The callables (operator, preconditioner, ELL matrix) are pytrees passed as jit arguments, so new data never recompiles, as in `lobpcg`.

### $A_P$ (`ap_operator`)

`ap_operator(op, free, sigma=0.0, s=None)` returns a pytree `APOperator` with `__call__(x)` $= m \odot (K + \sigma M)(m \odot x) + s\,(1 - m) \odot x$ on $(n, k)$ blocks, plus `n_dof`, `op`, `free` (float mask), `sigma` and `s` leaves.
- **Default s:** the mean over free DOFs of $\mathrm{diag}K + |\sigma|\,\mathrm{diag}M$.
  - For $\sigma \ge 0$ this is PR 4's rule ($s$ = mean free diagonal of $A$).
  - For $\sigma < 0$ it is the roadmap's constrained diagonal of $K - |\sigma| M$ (mean diag K + |σ| mean diag M, positive), so MINRES's operator needs no explicit `s`.
- An explicit `s` overrides it.
- It is the matrix-free counterpart of PR 4's `a_p(K + sigma * M, free, s)`.

### Node blocks (`block_diag_K`)

`op.block_diag_K()` returns the `(n_nodes, 3, 3)` diagonal 3×3 blocks of $K$ (with `s_K` scaling) for both operators.

- **Unstructured:** it runs through `_assemble` with a kernel that returns each element's node blocks as `(c, k, 3)`. Row $3a + i$, column $j$ is $K^e_{3a+i,\,3a+j}$. The stored mode reads them from the packed `Ke`; the on-the-fly mode computes $\sum_q w\,B_a^\top C B_a$. `_assemble(op, kern, 3)` sums them into `(n_dof, 3)`, which is reshaped to `(n_nodes, 3, 3)`.
- **Stencil:** one 24×3 node-block slice per phase of the reference `Ke`, weighted like `_k_diag`, through `_slabs` with 3 columns.
- **Why no `block_diag_M`:**
  - consistent $M^e = m_{ab} \otimes I_3$;
  - the lumped mass is diagonal;
  - nodal masses are isotropic.
  
  So the node blocks of $M$ are exactly `diag_M()` times $I_3$.

### Node-block Jacobi (`block_jacobi`)

`block_jacobi(op, free, sigma=0.0)` builds the node blocks of $K + \sigma M$:
- $B_n = K_{nn} + \sigma\,\mathrm{diag}(M)_n I_3$;
- rows and columns of constrained components zeroed, with 1 on their diagonal. The output is masked, so the constrained diagonal never reaches it and no `s` is needed (as in `jacobi`). On the free entries $B_n$ equals the 3×3 diagonal block of $A_P$.

A concrete $\sigma < 0$ raises `ValueError` (preconditioners are SPD and approximate $K + \sigma M$ with $\sigma \ge 0$; MINRES's preconditioner is built at $+\sigma_b$), and `jacobi` gets the same check; a traced σ is not checked, so both stay traceable. The blocks are inverted through a batched Cholesky (`cho_solve` against $I_3$), so a non-SPD block gives NaN instead of silent garbage. A package-internal `_inverse_blocks(B, free)` does the masking and inversion from any `(n, b, b)` blocks with an `(n, b)` mask, so PRs 11 and 13 can feed it coarse-level blocks (3×3 and 6×6); the docs name it as the internal entry for level blocks. `block_jacobi` returns `Partial(_block_jacobi, inv, mask)`: $v \mapsto m \odot (B_n^{-1} v_n)$. Its storage is 9 values per node (3 vectors). Like `jacobi`, it fits any `precond` slot.

### PCG (`pcg`)

`pcg(A, b, precond=None, tol=1e-8, maxit=1000)`, for $(n, k)$ right-hand sides.

- **Iteration:** a standard preconditioned CG, run independently per column (column-wise scalars, as for PR 14's probe blocks). A column that is done is frozen: its $\alpha$ and $\beta$ are set to 0 through `jnp.where` (which also guards the $0/0$ of zero and converged columns), so its $x$ stops changing. The loop is one jitted `lax.while_loop`; `tol` and `maxit` are dynamic.
- **Stopping, per column:**
  1. When the recursive residual meets $\|r\|_2 \le \mathrm{tol}\,\|b\|_2$, the explicit residual $b - Ax$ is computed (one extra apply for the block, under `lax.cond` only in iterations where some column newly meets tol).
  2. If the explicit residual is also within tol, the column is converged.
  3. Otherwise the column restarts from its current $x$: $r = b - Ax$ (the explicit residual), $z = P r$, $\rho = r^\top z$, $p = z$. Keeping the old $p$ was measured not to work (correctness review): the recursive residual then creeps down from the floor without reaching tol, no further replacement happens, and the column runs to `maxit`. With the restart, tol $10^{-14}$ converges after a replacement, and $10^{-15}$, $10^{-16}$ stagnate after 5 replacements in 469–848 iterations; iteration counts above the floor are unchanged.
  4. After `STALL_R` = 5 replacements in a column without convergence, the column stops as `stagnated` (the attainable floor; PR 14's "unverifiable" input).
  5. `maxit` stops every remaining column.
  6. A column whose residual norm is not finite stops (not converged, not stagnated; its `residuals` entry is NaN), e.g. under a non-SPD preconditioner whose Cholesky gave NaN. The same applies to `minres`.
  
  A restart costs one operator apply and one preconditioner apply for the block, both inside the `lax.cond` branch, which runs only at the floor.
  
  Iteration counts are exact (no check granularity), which the ≤ 20%-growth tests of PRs 11–13 need.
- **Returns:** `x` and `info`, a dict of device arrays so that `pcg` stays traceable:
  - `iterations` `(k,)`;
  - `residuals` `(k,)`, explicit relative residuals at exit;
  - `converged` `(k,)`;
  - `stagnated` `(k,)`;
  - `replacements` `(k,)` (PCG only), so the restart path is observable.
- **Zero columns** of `b` return $x = 0$ and converged, with 0 iterations.
- **Masking:** `b` must vanish on constrained DOFs (it is not checked, because `pcg` knows no mask). With a masked-output preconditioner, $x$ then stays 0 there. That is exact for $A_P$, whose constrained part is decoupled.
- **Default `precond`:** the identity.
- **Memory:** state $x, r, z, p$ plus $q = Ap$, i.e. 5 blocks, plus $b$ and the operator and preconditioner workspace.

### MINRES (`minres`)

`minres(A, b, precond=None, tol=1e-8, maxit=10000)`.

- **Iteration:** preconditioned MINRES (Paige–Saunders, the recurrences of SciPy's `minres`), column-wise as for `pcg`. $A$ is symmetric (possibly indefinite) and the preconditioner SPD (Vecharynski–Knyazev absolute-value preconditioning by the SPD multigrid of $K + |\sigma|M$ in PR 15).
- **Stopping (roadmap):** the explicit 2-norm residual $\|b - Ax\|_2 \le \mathrm{tol}\,\|b\|_2$ is checked every `CHECK` = 10 iterations. A column stops as:
  - `converged` once its explicit residual meets tol;
  - `stagnated` after 5 consecutive checks without a new minimum of its explicit residual;
  - `maxit` (an `info` flag through `converged` False, `stagnated` False).
  
  The last iterate is returned. Iteration counts are multiples of 10 when `maxit` is (PR 15's cost model uses these). Frozen columns and zero columns ($\beta = 0$, MINRES's breakdown case) are guarded with `jnp.where` as in `pcg`.
- **Returns:** the same as `pcg` without `replacements`; `residuals` are the explicit residuals at exit (PR 15 takes $\tau_\mathrm{eff}$ from them).
- **Floor:** measured 3.5–8.8e-14 (by right-hand side and run) with σ between $\lambda_6$ and $\lambda_7$, 1.2e-12 between $\lambda_{30}$ and $\lambda_{31}$ (it grows with the shift's index); recorded in `docs/preconditioners.md` for PR 15.
- **Memory:** 7 blocks ($x$, $r_1$, $r_2$, $y$, $w$, $w_2$ and the apply output).

### Block-ELL (`block_ell`, `BlockELL`)

`block_ell(A, b=3, dtype="float64", sliced=None, chunk=2048)` converts a SciPy sparse matrix ($n$ divisible by $b$) through `tobsr((b, b))` into a pytree `BlockELL` with `__call__(X)`, `n_dof`, `padding` and `n_buckets`.

- **Layout:** each bucket holds:
  - `rows` `(r,)` int32, the block rows it covers;
  - `cols` `(r_pad, w)` int32;
  - `vals` `(r_pad, w, b, b)`.
  
  Here $w$ is the bucket's width (the maximum blocks per row) and $r_\mathrm{pad}$ is a multiple of `chunk` block rows. Padding entries point at the row's own block column with zero values.
- **Apply:**
  - per bucket, a `lax.scan` over row chunks: gather `X[cols]` as `(chunk, w, b, m)`, then `einsum` with `vals`;
  - one bucket with every row in order is written directly;
  - several buckets are concatenated and put back in row order by one gather with the inverse permutation.
  
  The workspace is about one vector block at `chunk` = 2048 (measured). `chunk` is a static field.
- **Chunk:** per bucket, `min(chunk, rows rounded up to 8)` block rows, so tiny coarse levels are not padded to 2048 rows. The gather temp is chunk·w·b·m·itemsize: about one vector block for 3×3 blocks, width 27 and 2048 rows; 79 MB for PR 13's level 1 (6×6, width 32, $m = 25$, measured shapes). The formula goes into the docs.
- **Padding ratio:** `padding` = stored blocks / structurally nonzero blocks (blocks present in the BSR structure), excluding the `chunk` round-up.
- **Sliced fallback:**
  - `sliced=None` chooses one bucket (plain ELL) when its padding is ≤ 1.5, else sliced ELL;
  - `sliced=True` / `False` force it.
  
  Sliced ELL groups the rows by width rounded up to a power of 2, one bucket per distinct width, so its padding is < 2.
- **Block size:** `b` is a parameter, because PR 13's smoothed-aggregation coarse levels have 6×6 blocks (6 near-null modes per aggregate), and PR 11's levels 3×3.
- **dtype:** `dtype="float32"` stores `vals` in fp32 for PR 11's fp32 V-cycles. The apply casts `vals` to the input's dtype per chunk (an fp64 input against fp32 values computes in fp64).
- **Built on the host** (vectorized NumPy, no Python row loop) from the matrices PRs 11–13 produce.

### Chebyshev (`lmax_estimate`, `chebyshev`)

- **`lmax_estimate(A, precond, steps=20, seed=0)`:**
  - What it estimates: $\lambda_\max(D^{-1}A)$, where $D^{-1}$ = `precond` (masked output, so the start vector needs no mask: $z = D^{-1}v$ vanishes on constrained DOFs). `A` must have `n_dof` (`APOperator` and `BlockELL` do).
  - Method: `steps` Lanczos steps in the $D^{-1}$-weighted form ($v_{j+1} \propto A z_j - \alpha_j v_j - \beta_j v_{j-1}$, $z = D^{-1} v$), from one seeded random start, without reorthogonalization, with a `jnp.where` guard on $\beta = 0$ (an invariant start on a tiny coarse level). It returns the largest Ritz value of the tridiagonal matrix (a lower bound).
  - Robustness (correctness review, seeds 0–4): $1.1\tilde\lambda_\max/\lambda_\max \in [1.0926, 1.1000]$ over voxel boxes up to 202k DOF, random two-phase contrast $10^4$, stiff and soft inclusions, hex20 and hex8 plates with $L/t$ 100 and 500, hex27, $\nu = 0.49$ and a free masked sphere, for Jacobi and block Jacobi.
  - Traceable: a jitted `fori_loop` with fixed `steps`; the eigenvalues of the small tridiagonal come from `jnp.linalg.eigvalsh`.
- **`chebyshev(A, precond, lmax, degree=2, ratio=30.0)`:**
  - Returns a pytree callable $S$, a fixed linear SPD preconditioner: $r \mapsto p(D^{-1}A)\,D^{-1}r$, the `degree`-step Chebyshev iteration from $x_0 = 0$.
  - Interval: $[\hat\lambda/\mathrm{ratio}, \hat\lambda]$, $\hat\lambda = 1.1\,$`lmax`. The safety factor is applied inside: callers pass the raw estimate.
  - Cost: `degree` preconditioner applies and `degree − 1` operator applies.
  - The smoothing step of PRs 11–13 is $x \leftarrow x + S(b - Ax)$, i.e. `degree` operator applies per smoothing ($\mathrm{degree} - 1$ for pre-smoothing from $x = 0$). A degree-2 V-cycle then takes 4 fine applies, below the budget model's 5.
  - $S$ is symmetric because $p(D^{-1}A)D^{-1} = D^{-1/2}p(D^{-1/2}AD^{-1/2})D^{-1/2}$. It is SPD because the residual polynomial is below 1 in magnitude on $(0, \hat\lambda]$, which contains the spectrum because $\hat\lambda \ge \lambda_\max$. For even degree $S$ stays SPD up to $\lambda_\max \le \hat\lambda(1 + 1/\mathrm{ratio})$; for odd degree it stays positive beyond $\hat\lambda$, but smoothing diverges there.
  - `degree` is static (a dataclass meta field). `lmax` is a leaf, so a new estimate does not recompile.
  - Ratio 30 is the common ML/Ifpack2 default; PR 11 may tune it.

### h-study (`benchmarks/pcg_study.py`)

The roadmap defines the iteration count as PCG iterations on $A_P$, at fixed σ, to a relative residual of $10^{-8}$.

- **Setup:**
  - clamped hex20 boxes $4^3$, $8^3$, $16^3$ and voxel boxes $8^3$, $16^3$, $32^3$, steel;
  - σ fixed at $\lambda_1$ of the coarsest mesh (`sigma_estimate`);
  - 4 random masked right-hand sides;
  - preconditioners: Jacobi, block Jacobi, and Chebyshev degree 2 over block Jacobi.
- **Output:** iterations and time to `benchmarks/results/pcg_study.json` through the harness's result writer, quoted in `docs/preconditioners.md`. Expected growth is about $h^{-1}$ for all three (no multigrid yet).
- **Test:** a fast test runs the study function on the smallest mesh only.

## Files

- `src/jaxmodal/linalg/krylov.py`, `linalg/ell.py`, `linalg/__init__.py` (exports).
- `src/jaxmodal/precond/__init__.py`, `precond/jacobi.py`, `precond/chebyshev.py`.
- `src/jaxmodal/operators/constraints.py` (`ap_operator`), `matfree.py` and `voxel.py` (`block_diag_K`), `operators/__init__.py`.
- `src/jaxmodal/eigen/lobpcg.py`: imports `jacobi` from `precond`. `eigen/__init__.py` keeps exporting it.
- `benchmarks/pcg_study.py`, `benchmarks/results/pcg_study.json`.
- Tests: `tests/test_krylov.py`, `tests/test_ell.py`, `tests/test_precond.py`.
- Docs:
  - `docs/preconditioners.md` (new);
  - `docs/operators.md` (`block_diag_K`, `ap_operator`);
  - `docs/eigensolvers.md` (the `jacobi` location);
  - `README.md`, `docs/ROADMAP.md`.

## Tests

### `tests/test_krylov.py`

All against `spsolve` on the PR 4 `a_p`, on the hex20 box above unless stated otherwise.

1. **PCG accuracy:**
   - Setup: σ = $\lambda_1$ and $100\lambda_1$; 3 masked right-hand sides plus one zero column; tol $10^{-12}$.
   - Assertions:
     - rel. error in $x$ $\le 10^{-10}$ per column (measured $\le 7\cdot10^{-13}$);
     - all converged, explicit residuals $\le 10^{-12}$;
     - the zero column has $x = 0$ and 0 iterations;
     - constrained rows of $x$ are exactly 0.
   - Repeated with the identity preconditioner, `block_jacobi` and `chebyshev` (degree 2).
   - Also run free-free, with σ > 0 and `bcs={}`.
2. **Columns are independent (PCG and MINRES):** at tol $10^{-8}$, a block of an easy column (a low-mode combination) and a random one, against the single-column solves:
   - iterations exactly equal;
   - $x$ within $10^{-9}$ relative (the batched rounding differs by up to $4\cdot10^{-11}$, measured);
   - the easy column's exit residual within 1% of its single-column value (an unfrozen column would keep stepping and push it far below tol: the tests review measured $x$ off by $1.6\cdot10^{-9}$ and survival of "$x$ to $10^{-12}$" at tol $10^{-12}$);
   - scaling one column by $10^{6}$ changes nothing for the others.
3. **PCG exits** (explicit `maxit=3000`):
   - tol $10^{-16}$: every column `stagnated` with exactly `STALL_R` = 5 replacements, not converged, fewer iterations than `maxit`; explicit residuals $\le 10^{-13}$ (measured $\le 4.6\cdot10^{-14}$ with the kept-$p$ prototype; re-measured with the restart at implementation before fixing the bound);
   - a tol in the $10^{-14}$ band (fixed at implementation from a measurement): a column converges after $\ge 1$ replacement;
   - tol $10^{-12}$: no replacements;
   - `maxit=5`: not converged, not stagnated, `iterations == 5`.
4. **MINRES accuracy:**
   - the SPD case as in test 1, including the zero column ($\beta = 0$);
   - the indefinite `ap_operator(op, free, -sigma_b)` ($K - \sigma_b M$) with $\sigma_b$ the midpoint of $\lambda_6$ and $\lambda_7$, default `s` (equal to the explicit roadmap $s$), preconditioner `jacobi(op, free, sigma_b)`, tol $10^{-12}$: rel. error $\le 10^{-10}$ (measured $2.2\cdot10^{-12}$; the explicit residual reached $3.3\cdot10^{-13}$, a 3× margin to tol, with the floor at $8.8\cdot10^{-14}$).
5. **MINRES exits:**
   - tol $10^{-15}$: `stagnated` well before `maxit`;
   - `maxit=30`: not converged, not stagnated;
   - iteration counts are multiples of 10.
6. **Compiles:** a second `pcg` (and `minres`) call with a new operator (new σ), preconditioner data, tol and right-hand side compiles nothing (subprocess, cache off, counted with `jax.monitoring` as in `test_lobpcg`).

### `tests/test_ell.py`

7. **Plain ELL against SciPy CSR to $10^{-12}$:**
   - matrices: the assembled hex8 and hex20 $A_P$ ($b = 3$) and a random symmetric block-sparse matrix with $b = 6$;
   - widths $m$ = 1 and 7;
   - `chunk` 2048 and a small `chunk` = 4, with a block-row count that is not a multiple of 4, so that several chunks and the round-up are exercised.
8. **Sliced ELL:**
   - Matrix: a matrix with a few dense rows in the middle, not at the ends (padding > 1.5, so `sliced=None` chooses sliced ELL); `n_buckets > 1`.
   - It matches CSR to $10^{-12}$ for $b$ = 3 and 6.
   - `sliced=False` on the same matrix also matches and reports the larger padding.
9. **Padding:** `padding` equals stored blocks / structural BSR blocks counted independently.
10. **fp32:** `dtype="float32"` stores fp32 `vals`, returns fp32 for fp32 input, and matches CSR to $10^{-6}$ relative.
11. **PCG through `BlockELL`:** PCG on `block_ell(A_P)` equals PCG on `ap_operator` to $10^{-10}$.

### `tests/test_precond.py`

12. **`ap_operator` against SciPy `a_p(K + sigma M, free, s)` @ x to $10^{-12}$:**
    - σ = 0, $\lambda_1$ and a negative σ (with its default $s$ equal to the roadmap's $s$);
    - unstructured and stencil;
    - on random vectors that are not masked (a missing inner $m \odot x$ fails).
13. **Node blocks against SciPy to $10^{-12}$:**
    - `block_diag_K` equals the 3×3 diagonal blocks of the assembled $K$;
    - `block_jacobi`'s blocks (before inversion) equal those of the PR 4 SciPy $A_P$ on the free entries, with identity on the constrained ones;
    - `block_jacobi` and `jacobi` raise for σ < 0;
    - cases:
      - unstructured stored and on-the-fly, consistent and lumped, with nodal masses, scaled `s_K`;
      - mixed hex8 + hex20 + a hex27 box;
      - the voxel stencil with a mask and 2 phases;
      - BCs `xmin: xyz` plus `ymin: y` (partially constrained nodes).
14. **Block-Jacobi apply:** on a vector with nonzero constrained entries, it equals `scipy.linalg.inv` of each block times $v$, with output exactly 0 on constrained DOFs.
15. **λ_max:** $1.1\,$`lmax_estimate` lies in $[1, 1.5]\times$ the `eigsh(which="LA")` $\lambda_\max$ of $(A_{P,ff}, D_{ff})$:
    - $D$ = Jacobi and block Jacobi;
    - unstructured hex20 and voxel;
    - seeds 0–2.
16. **Chebyshev:**
    - for degrees 1–4, with Jacobi and block Jacobi: `chebyshev(...)` applied to random masked vectors equals the dense spectral reference to $10^{-12}$: on the free block, with the Cholesky factor $D_{ff} = LL^\top$ (block diagonal for block Jacobi), $L^{-1}A_{P,ff}L^{-\top} = Q\,\mathrm{diag}(\mu)\,Q^\top$ and $r_k(\mu) = T_k((\theta - \mu)/\delta)/T_k(\theta/\delta)$ (same `lmax`), $S_\mathrm{ref} = L^{-\top}Q\,\mathrm{diag}((1 - r_k(\mu))/\mu)\,Q^\top L^{-1}$. The tests review measured the prototype within $6\cdot10^{-15}$ and four coefficient mutants (recurrence sign, missing factor 2, no 1.1, ratio 10) off by $7\cdot10^{-2}$–$5\cdot10^{-1}$, while all four passed symmetry, positivity and "fewer iterations than block Jacobi". This also pins that the 1.1 is applied exactly once;
    - for degrees 1–4:
      - $|u^\top Sv - v^\top Su| \le 10^{-12}\,|u^\top Sv|$;
      - error propagation $E = I - SA_P$ is $A_P$-self-adjoint to $10^{-12}$;
      - $u^\top Su > 0$ for random masked $u$;
    - degree 1 equals $D^{-1}/\theta$ exactly in form ($\theta$ the interval centre);
    - as a PCG preconditioner, degree 2 takes fewer iterations than block Jacobi on the hex20 box (recorded).
17. **`jacobi` move:** `jaxmodal.eigen.jacobi is jaxmodal.precond.jacobi`; the PR 9 tests pass unchanged.
18. **h-study:** the study function on its smallest mesh writes the JSON keys and positive iteration counts.

## Docs

- `docs/preconditioners.md` (new):
  - $A_P$ and `ap_operator` (the s rule, indefinite σ);
  - node blocks and `block_jacobi`;
  - the Chebyshev polynomial, the interval, the λ_max estimate and why $S$ is SPD;
  - PCG and MINRES: algorithms, stopping rules, `info`, block counts, masking;
  - block-ELL: layout, chunks, padding, the sliced fallback, fp32;
  - the measured tables above and the h-study.
- `docs/operators.md`: `block_diag_K` as the sixth interface method (the "same five" wording at the interface paragraph changes).
- `docs/boundary_conditions.md`: the module intro and the $A_P$ section gain `ap_operator` (matrix-free counterpart of `a_p`), its default $s$ for both signs of σ and the sign convention.
- `docs/eigensolvers.md`: `jacobi` now lives in `precond/` and raises for a concrete σ < 0 (its entry in the LOBPCG section).
- `README.md`: a short PCG with Chebyshev snippet; the status line.
- `docs/ROADMAP.md`: the status line; Principle 2's interface list gains `block_diag_K` (from PR 10).

## Not covered

- Multigrid hierarchies and transfers (PRs 11–13).
- A fused $K + \sigma M$ apply. This is the budget model's first lever, deferred until a PR measures the need.
- Assembling $A_P$ into block-ELL from elements (PR 11 builds its coarse levels element-wise).
- Block-Krylov variants: columns are independent solves.
- Node blocks of block-ELL coarse levels (PR 11 reads the diagonal slot and calls `_inverse_blocks`), and `BlockELL` storage bytes for the PR 11 and 13 estimators.
- fp32 smoother data (PR 11 casts the leaves of `block_jacobi`, `chebyshev` and the levels).
- PR 14's masked $M_P$: `ap_operator(op.scaled(s_K=zeros), free, 1.0)` is exactly $mMm + \overline{\mathrm{diag}M}\,(I - m)$, so no constructor is added; the docs note it as a correctness shortcut that still pays a (zero-scaled) K apply, about 2× the cost of M alone, which PR 14's cost model counts or replaces by a small $M_P$ pytree.

## Implementation notes

- Test 2: the batched and single-column applies round differently, so iteration counts may differ by 1 (measured 287 vs 286 for the random column), and the easy column's exit residual differed by 1.0% at equal iterations. The test allows ±1 iteration and replaces the 1% residual check by an exact one: the block's easy-column $x$ after `maxit` = its iteration count equals the full run's bit for bit (frozen).
- Test 12 uses σ = 1.3 (a positive value; the box of test 13 has no λ₁ computed).
- Tests 1–5 fix ARPACK's start vector, so σ = λ₁ (and the floor-band exits of test 3) are identical in every run.
- Test 7/auto: on small boxes the clamped nodes are width-1 rows of $A_P$, so plain padding exceeds 1.5 (hex8 6³: 1.59 clamped, 1.35 free); the automatic-choice test uses the free hex8 6³ box.
