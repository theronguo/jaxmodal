# Linear solvers, preconditioners and block-ELL

PR 10 adds the single-level kernels that the multigrid PRs (11–13), the eigenvalue counting (PR 14) and the interior bands (PR 15) build on. Everything works on the preconditioning operator $A_P$ ([`boundary_conditions.md`](boundary_conditions.md)).

| module | contents |
|---|---|
| `jaxmodal.linalg` (`linalg/krylov.py`) | `pcg`, `minres` |
| `jaxmodal.linalg` (`linalg/ell.py`) | `block_ell`, `BlockELL` |
| `jaxmodal.precond` (`precond/jacobi.py`) | `jacobi` (moved from `eigen/lobpcg.py`; `jaxmodal.eigen.jacobi` is the same function), `block_jacobi` |
| `jaxmodal.precond` (`precond/chebyshev.py`) | `lmax_estimate`, `chebyshev`, `Chebyshev` |
| `jaxmodal.operators` (`operators/constraints.py`) | `ap_operator`, the matrix-free $A_P$ |
| operators (`matfree.py`, `voxel.py`) | the sixth interface method `block_diag_K()` |

```python
from jaxmodal.linalg import block_ell, minres, pcg
from jaxmodal.operators import a_p, ap_operator
from jaxmodal.precond import block_jacobi, chebyshev, jacobi, lmax_estimate

A = ap_operator(op, free, sigma)  # m (K + sigma M) m + s (I - m), matrix-free
D = block_jacobi(op, free, sigma)  # 3x3 node blocks of K + sigma M, inverted
S = chebyshev(A, D, lmax_estimate(A, D))  # degree 2 over block Jacobi
x, info = pcg(A, b, S, tol=1e-8)  # b (n_dof, k), zero on constrained DOFs

Ab = ap_operator(op, free, -sigma_b)  # indefinite K - sigma_b M
x, info = minres(Ab, b, jacobi(op, free, sigma_b), tol=1e-10)  # SPD preconditioner at +sigma_b

E = block_ell(a_p(K + sigma * M, free))  # assembled SciPy matrix -> device block-ELL
y = E(x)
```

Every public entry checks that x64 is enabled (the flag, not the data's dtype). Operators, preconditioners and ELL matrices are pytrees passed to the jitted solvers as arguments, so new data (σ, preconditioner values, `tol`, `b`) never recompiles. Everything is traceable.

## Node blocks (`block_diag_K`)

`op.block_diag_K()` returns the `(n_nodes, 3, 3)` diagonal 3×3 blocks $K_{nn}$ of $K$, with the `s_K` scaling, for both operators:

- **Unstructured:** element node blocks $K^e_{3a+i,\,3a+j}$ (stored mode: read from the packed $K_e$; on-the-fly: $\sum_q w_q \det J_q\, B_{qa}^\top C_e B_{qa}$) are summed by the operator's own assembly path into `(n_dof, 3)` and reshaped.
- **Stencil:** a 24×3 node-block slice of each phase's reference $K_e$, weighted like `diag_K`, through the slab loop.
- **No `block_diag_M`:** the consistent $M^e = m_{ab} \otimes I_3$, the lumped mass is diagonal and nodal masses are isotropic, so the node blocks of $M$ are exactly `diag_M()` times $I_3$.

## Jacobi and node-block Jacobi

Both approximate $A_P$ of $K + \sigma M$ with $\sigma \ge 0$, are pytree callables (`jax.tree_util.Partial`) on `(n, k)` blocks, and have masked output: the constrained diagonal $s$ of $A_P$ never reaches the output, so they need no $s$.

- **`jacobi(op, free, sigma=0.0)`:** $v \mapsto m \odot v / (\mathrm{diag}K + \sigma\,\mathrm{diag}M)$.
- **`block_jacobi(op, free, sigma=0.0)`:** $v_n \mapsto m_n \odot (B_n^{-1} v_n)$ with
  $$B_n = K_{nn} + \sigma\,\mathrm{diag}(M)_n I_3,$$
  constrained rows and columns replaced by the identity. On the free entries $B_n$ is the 3×3 diagonal block of $A_P$. The blocks are inverted by a batched Cholesky (a non-SPD block gives NaN, which then shows up as NaN in the solver's `residuals`). Storage: 9 values per node.
- A concrete $\sigma < 0$ raises `ValueError` in both (a traced σ is not checked). For the indefinite $K - \sigma_b M$, build the preconditioner at $+\sigma_b$.
- **Internal entry for level blocks:** `precond.jacobi._inverse_blocks(B, free)` masks and inverts any `(n, b, b)` SPD blocks with an `(n·b,)` mask and returns `(inv, mask)`; PRs 11 and 13 feed it their coarse-level 3×3 and 6×6 blocks.

## Chebyshev

**$\lambda_\max$ estimate.** `lmax_estimate(A, precond, steps=20, seed=0)` estimates $\lambda_\max(D^{-1}A)$, $D^{-1}$ = `precond`, by `steps` Lanczos steps in the $D^{-1}$-weighted form ($v_{j+1} \propto A z_j - \alpha_j v_j - \beta_j v_{j-1}$, $z_j = D^{-1} v_j$) from one seeded random start, without reorthogonalization. It returns the largest Ritz value of the tridiagonal $T$, a lower bound. A breakdown ($\beta = 0$, an invariant start on a tiny level) is guarded. `A` must have `n_dof` (`APOperator` and `BlockELL` do). Over voxel boxes up to 202k DOF, two-phase contrast $10^4$, inclusions, thin plates ($L/t$ up to 500), hex20/hex8/hex27, $\nu = 0.49$ and a free masked sphere, seeds 0–4, Jacobi and block Jacobi: $1.1\,\tilde\lambda_\max/\lambda_\max \in [1.0926, 1.1000]$.

**Smoother.** `chebyshev(A, precond, lmax, degree=2, ratio=30.0)` returns the pytree callable
$$S = p(D^{-1}A)\,D^{-1},$$
the `degree`-step Chebyshev iteration for $Ax = r$ from $x_0 = 0$ on the interval $[\hat\lambda/\mathrm{ratio}, \hat\lambda]$, $\hat\lambda = 1.1\,$`lmax` (`SAFETY` = 1.1 is applied inside; pass the raw estimate). With $\theta = (\hat\lambda + \hat\lambda/\mathrm{ratio})/2$ and $\delta = (\hat\lambda - \hat\lambda/\mathrm{ratio})/2$, the residual polynomial is
$$1 - \mu\,p(\mu) = \frac{T_k\big((\theta - \mu)/\delta\big)}{T_k(\theta/\delta)}.$$

- Degree 1 is $D^{-1}/\theta$. Cost: `degree` preconditioner applies and `degree − 1` operator applies.
- The smoothing step of PRs 11–13 is $x \leftarrow x + S(b - Ax)$ (`degree` operator applies; `degree − 1` for pre-smoothing from $x = 0$).
- $S$ is symmetric, since $p(D^{-1}A)D^{-1} = D^{-1/2}p(D^{-1/2}AD^{-1/2})D^{-1/2}$, and positive definite because the residual polynomial is below 1 in magnitude on $(0, \hat\lambda]$, which holds the spectrum when $\hat\lambda \ge \lambda_\max$.
- `degree` is static (a dataclass meta field); `lmax` and `ratio` are leaves. Ratio 30 is the ML/Ifpack2 default.

## PCG (`pcg`)

`pcg(A, b, precond=None, tol=1e-8, maxit=1000)` solves $A x = b$ for SPD $A$ and an SPD preconditioner (default: the identity), with `b` of shape `(n, k)`.

- **Columns are independent:** column-wise scalars, as for PR 14's probe blocks. A finished column is frozen ($\alpha = \beta = 0$ by `jnp.where`, which also guards $0/0$), so its $x$ stops changing. One jitted `lax.while_loop`; `tol` and `maxit` are dynamic.
- **Stopping, per column:**
  1. When the recursive residual meets $\|r\|_2 \le \mathrm{tol}\,\|b\|_2$, the explicit residual $b - Ax$ is computed (one apply for the block, under `lax.cond`, only in iterations where some column meets tol).
  2. If it is also within tol, the column has converged.
  3. Otherwise the column restarts from its $x$: $r = b - Ax$, $z = Pr$, $p = z$.
  4. After `STALL_R` = 5 restarts without convergence it stops as `stagnated`: the attainable floor.
  5. `maxit` stops the remaining columns; a column whose residual is not finite stops too (not converged, not stagnated, NaN residual: e.g. `block_jacobi` with a non-SPD block, whose Cholesky gives NaN). MINRES sees a non-finite residual at its next check.
- Iteration counts are exact (no check granularity).
- **`info`** (device arrays, so `pcg` is traceable): `iterations`, `residuals` (explicit relative residuals at exit), `converged`, `stagnated`, `replacements`, all `(k,)`.
- Zero columns return $x = 0$, converged, with 0 iterations.
- **Masking:** `b` must vanish on constrained DOFs (not checked). With a masked-output preconditioner, $x$ is then exactly 0 there, which is exact for $A_P$.
- **Memory** (measured, plus `b`): 4 blocks with a diagonal operator, whose $Ap$ fuses away; with the hex20 operator (24³, $k = 4$) the temp is about 6 blocks above a bare apply's.

## MINRES (`minres`)

`minres(A, b, precond=None, tol=1e-8, maxit=10000)` solves symmetric, possibly indefinite systems with an SPD preconditioner: preconditioned MINRES (Paige–Saunders, the recurrences of SciPy's `minres`), column-wise as `pcg`.

- **Stopping:** the explicit residual $\|b - Ax\|_2 \le \mathrm{tol}\,\|b\|_2$ is checked every `CHECK` = 10 iterations. A column is `converged` when it meets tol and `stagnated` after `STALL_C` = 5 checks without a new minimum; `maxit` gives neither. The last iterate is returned. Iteration counts are multiples of 10 when `maxit` is.
- **`info`:** as `pcg` without `replacements`.
- **Floor:** on the hex20 box below with Jacobi of $K + \sigma_b M$, the explicit residual stalls at $3.5$–$8.8\cdot10^{-14}$ (by right-hand side and run) with $\sigma_b$ between $\lambda_6$ and $\lambda_7$ and at about $1.2\cdot10^{-12}$ between $\lambda_{30}$ and $\lambda_{31}$ (plan prototype): it grows with the shift's index (PR 15's $\tau_\mathrm{eff}$).
- **Memory** (measured, plus `b`): 9 blocks with a diagonal operator (the carried $x$, $r_1$, $r_2$, $y$, $w$, $w_2$, the body's $v$ and $Av$, the returned $x$); with the hex20 operator 6 blocks more than PCG.

## Block-ELL (`block_ell`, `BlockELL`)

`block_ell(A, b=3, dtype="float64", sliced=None, chunk=2048)` converts a SciPy sparse matrix ($n$ divisible by $b$) on the host, through BSR with $b \times b$ blocks, into a pytree `BlockELL`: `E(X)` for `(n, m)` blocks, `n_dof`, `padding`, `n_buckets`.

- **Layout:** per bucket, `cols` `(r_pad, w)` int32 and `vals` `(r_pad, w, b, b)`; $w$ is the bucket's width (most blocks per row). Padding entries point at the row's own block column with zero values.
- **Apply:** per bucket a `lax.scan` over row chunks gathers `X[cols]` as `(chunk, w, b, m)` and contracts with `vals`. Several buckets are concatenated and put back in row order by one gather.
- **Chunk:** `min(chunk, rows rounded up to 8)` block rows per bucket, so tiny coarse levels are not padded to 2048 rows. The gather temp is
  $$\mathrm{chunk}\cdot w\cdot b\cdot m\cdot\mathrm{itemsize}$$
  bytes: about one vector block for 3×3 blocks, width 27 and 2048 rows; 79 MB for PR 13's level 1 (6×6, width 32, $m = 25$).
- **Padding:** `padding` = stored blocks / structurally nonzero blocks (without the chunk round-up). Constrained nodes are width-1 rows of $A_P$, so clamped faces add padding: plain-ELL padding of the hex8 $A_P$ is 1.35 free and 1.59 clamped on $6^3$, 1.21 / 1.33 on $10^3$, 1.07 clamped on $44^3$.
- **Sliced fallback:** `sliced=None` uses one bucket (plain ELL) while its padding is $\le 1.5$, else one bucket per row width rounded up to a power of 2 (padding $< 2$); `True` / `False` force it. hex20 meshes exceed 1.5 and are sliced.
- **Block size** `b` is a parameter: 3 for PR 11's levels, 6 for PR 13's smoothed-aggregation levels.
- **dtype:** `"float32"` stores `vals` in fp32 for fp32 V-cycles; the apply casts them to the input's dtype (fp64 in computes in fp64).

## Measured (jax 0.11.2, fp64, M3 Max CPU)

hex20 box $6\times5\times4$ (2,199 DOF), $E = 3$, $\nu = 0.3$, $\rho = 2$, `xmin` clamped, 3 random right-hand sides (iterations: max over them), against `spsolve` on `a_p`:

| solver | σ | preconditioner | tol | iterations | rel. error in $x$ |
|---|---|---|---|---|---|
| PCG | $\lambda_1$ | Jacobi | $10^{-12}$ | 349 | $4.5\cdot10^{-13}$ |
| PCG | $\lambda_1$ | block Jacobi | $10^{-12}$ | 347 | $5.4\cdot10^{-13}$ |
| PCG | $\lambda_1$ | Chebyshev 2 / block Jacobi | $10^{-12}$ | 193 | $5.8\cdot10^{-13}$ |
| MINRES | $-\sigma_b$, $\sigma_b \in (\lambda_6, \lambda_7)$ | Jacobi at $\sigma_b$ | $10^{-12}$ | 420 | $1.8\cdot10^{-12}$ |

- PCG at tol $10^{-14}$: 2 of 3 columns converge after 1–2 restarts; at $10^{-16}$ all stagnate after exactly 5 restarts (746–852 iterations) with explicit residuals $\le 2.8\cdot10^{-14}$.
- MINRES at tol $10^{-15}$ stagnates after 550–610 iterations at explicit residuals $3.5$–$4.9\cdot10^{-14}$.
- Chebyshev over Jacobi, as a PCG preconditioner at $\sigma = \lambda_1$, tol $10^{-8}$: 289 / 160 / 109 / 85 iterations for degrees 1 / 2 / 3 / 4.
- Block-ELL against SciPy: $\le 10^{-15}$. On the assembled hex8 $A_P$, $44^3$ (273k DOF, 2.30M blocks), the row-chunked apply at chunk 2048 takes 10 / 30 / 45 ms at $m$ = 1 / 15 / 25 (SciPy CSR: 12 / 102 / 95 ms), with a 61 MB temp at $m = 25$.

**h-study** (`benchmarks/pcg_study.py` → `benchmarks/results/pcg_study.json`): PCG iterations on $A_P$ to relative residual $10^{-8}$, clamped (`xmin`) unit steel boxes, σ fixed per family at $\lambda_1$ of the coarsest mesh (`sigma_estimate`), 4 random right-hand sides (max over them):

| mesh | DOF | Jacobi | block Jacobi | Chebyshev 2 / block Jacobi |
|---|---|---|---|---|
| hex20 $4^3$ | 1,275 | 175 | 173 | 98 |
| hex20 $8^3$ | 8,019 | 336 | 335 | 182 |
| hex20 $16^3$ | 56,355 | 657 | 658 | 350 |
| voxel $8^3$ | 2,187 | 104 | 101 | 54 |
| voxel $16^3$ | 14,739 | 202 | 198 | 105 |
| voxel $32^3$ | 107,811 | 389 | 384 | 204 |

The counts grow as $h^{-1}$ (×1.86–1.96 per refinement) for all three, as expected without a coarse level; multigrid (PRs 11–13) must remove this growth. Block Jacobi barely beats Jacobi on these isotropic boxes; Chebyshev degree 2 halves the iterations for 2 operator applies per iteration, so without a coarse level it gains nothing in operator applies or wall time (hex20 $16^3$: 3.75 s against 3.52 s for Jacobi).

## Tests

- `tests/test_krylov.py`: PCG with the identity, Jacobi, block Jacobi and Chebyshev, σ = $\lambda_1$, $100\lambda_1$ and free-free, against `spsolve` to $10^{-10}$ with a zero column; columns independent and frozen; the PCG exits (floor after exactly 5 restarts, convergence after a restart, `maxit`, a residual turning NaN mid-run); MINRES on SPD and indefinite systems to $10^{-10}$ and its exits; no recompiles for new data.
- `tests/test_ell.py`: plain and sliced ELL against CSR to $10^{-12}$ ($b$ = 3, 6; $m$ = 1, 7; several chunks with a remainder), padding against an independent count, the automatic choice, fp32 to $10^{-6}$, PCG through `BlockELL`.
- `tests/test_precond.py`: `ap_operator` against `a_p` on unmasked vectors (σ = 0, positive, negative); node blocks and block-Jacobi inverses against SciPy for stored/on-the-fly, consistent/lumped, nodal masses, `s_K`, mixed hex8/hex20, hex27 and a masked two-phase voxel grid with `s_K`, all with partially constrained nodes; the σ < 0 raise; $1.1\,\tilde\lambda_\max/\lambda_\max \in [1, 1.5]$ and the invariant-start guard; Chebyshev against a dense spectral reference to $10^{-12}$ (degrees 1–4), symmetry, $A_P$-self-adjoint error propagation, positivity, and fewer PCG iterations than block Jacobi; the h-study smoke run.
