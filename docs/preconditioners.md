# Linear solvers, preconditioners and block-ELL

PR 10 adds the single-level kernels that the multigrid PRs (11–13), the eigenvalue counting (PR 14) and the interior bands (PR 15) build on; PR 11a adds the voxel geometric multigrid ([below](#voxel-geometric-multigrid-gmg)) PR 12 the unstructured p-multigrid ([below](#unstructured-p-multigrid-pmg)) and PR 13a the unstructured AMG ([below](#unstructured-amg-amg)), all on one shared V-cycle, and PR 18a its solid-shell path ([below](#amg-on-solid-shells-pr-18a)). Everything works on the preconditioning operator $A_P$ ([`boundary_conditions.md`](boundary_conditions.md)).

| module | contents |
|---|---|
| `jaxmodal.linalg` (`linalg/krylov.py`) | `pcg`, `minres` |
| `jaxmodal.linalg` (`linalg/ell.py`) | `block_ell`, `BlockELL` |
| `jaxmodal.precond` (`precond/jacobi.py`) | `jacobi` (moved from `eigen/lobpcg.py`; `jaxmodal.eigen.jacobi` is the same function), `block_jacobi` |
| `jaxmodal.precond` (`precond/chebyshev.py`) | `lmax_estimate`, `chebyshev`, `Chebyshev` |
| `jaxmodal.precond` (`precond/gmg.py`) | `gmg`, `GMG` (voxel geometric multigrid; the shared V-cycle) |
| `jaxmodal.precond` (`precond/pmg.py`) | `pmg` (unstructured p-multigrid, returns a `GMG`) |
| `jaxmodal.precond` (`precond/amg.py`) | `amg` (p-level plus smoothed-aggregation AMG, returns a `GMG`), `ELLTransfer` |
| `precond/shell.py` (PR 18a, internal) | the solid-shell AMG setup: fibers, column and ring patches, group graphs, curvature candidates, `_shell_levels` |
| `jaxmodal.precond.schwarz` (`precond/schwarz.py`, PR 18a) | `Schwarz` (additive Schwarz over packed patch inverses), `build`, `layout`, `nbytes` |
| `jaxmodal.operators` (`operators/constraints.py`) | `ap_operator`, the matrix-free $A_P$ |
| operators (`matfree.py`, `voxel.py`) | the sixth interface method `block_diag_K()`; `matfree.element_matrices` (PR 12) |

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
- A concrete $\sigma < 0$ or NaN raises `ValueError` in both (a traced σ is not checked). For the indefinite $K - \sigma_b M$, build the preconditioner at $+\sigma_b$.
- **Solid shells:** `jacobi` is the `solve_modes` default on `ssh8` meshes until PR 18b (LOBPCG iterations grow about like $(L/t)^2$: 57, 232, 1644 at $L/t$ = 10, 20, 50, `docs/eigensolvers.md`; 1289 PCG iterations on the $32^2$ plate at $L/t = 50$, `benchmarks/results/shell_amg_study.json`). `amg` works on them since PR 18a ([below](#amg-on-solid-shells-pr-18a)); `pmg` raises.
- **Internal entry for level blocks:** `precond.jacobi._inverse_blocks(B, free)` masks and inverts any `(n, b, b)` SPD blocks with an `(n·b,)` mask and returns `(inv, mask)`; PRs 11 and 13 feed it their coarse-level 3×3 and 6×6 blocks, and PR 18a its Schwarz patch matrices (padded slots the identity).

## Chebyshev

**$\lambda_\max$ estimate.** `lmax_estimate(A, precond, steps=20, seed=0)` estimates $\lambda_\max(D^{-1}A)$, $D^{-1}$ = `precond`, by `steps` Lanczos steps in the $D^{-1}$-weighted form ($v_{j+1} \propto A z_j - \alpha_j v_j - \beta_j v_{j-1}$, $z_j = D^{-1} v_j$) from one seeded random start, without reorthogonalization. It returns the largest Ritz value of the tridiagonal $T$, a lower bound. A breakdown ($\beta = 0$, an invariant start on a tiny level) is guarded. `A` must have `n_dof` (`APOperator` and `BlockELL` do). Over voxel boxes up to 202k DOF, two-phase contrast $10^4$, inclusions, thin plates ($L/t$ up to 500), hex20/hex8/hex27, $\nu = 0.49$ and a free masked sphere, seeds 0–4, Jacobi and block Jacobi: $1.1\,\tilde\lambda_\max/\lambda_\max \in [1.0926, 1.1000]$.

**Smoother.** `chebyshev(A, precond, lmax, degree=2, ratio=30.0)` returns the pytree callable
$$S = p(D^{-1}A)\,D^{-1},$$
the `degree`-step Chebyshev iteration for $Ax = r$ from $x_0 = 0$ on the interval $[\hat\lambda/\mathrm{ratio}, \hat\lambda]$, $\hat\lambda = 1.1\,$`lmax` (`SAFETY` = 1.1 is applied inside; pass the raw estimate). With $\theta = (\hat\lambda + \hat\lambda/\mathrm{ratio})/2$ and $\delta = (\hat\lambda - \hat\lambda/\mathrm{ratio})/2$, the residual polynomial is
$$1 - \mu\,p(\mu) = \frac{T_k\big((\theta - \mu)/\delta\big)}{T_k(\theta/\delta)}.$$

- Degree 1 is $D^{-1}/\theta$. Cost: `degree` preconditioner applies and `degree − 1` operator applies.
- The smoothing step of PRs 11–13 is $x \leftarrow x + S(b - Ax)$ (`degree` operator applies; `degree − 1` for pre-smoothing from $x = 0$).
- $S$ is symmetric, since $p(D^{-1}A)D^{-1} = D^{-1/2}p(D^{-1/2}AD^{-1/2})D^{-1/2}$, and positive definite because the residual polynomial is below 1 in magnitude on $(0, \hat\lambda]$, which holds the spectrum when $\hat\lambda \ge \lambda_\max$.
- `degree` is static (a dataclass meta field); `lmax` and `ratio` are leaves. Ratio 30 is the ML/Ifpack2 default; `chebyshev` raises `ValueError` unless `ratio > 1` (and `degree` is an int ≥ 1).
- The `degree − 1` steps run in a `lax.fori_loop` (PR 13c; before, a Python-unrolled loop whose temporaries XLA kept per step: AMG V-cycle 19.1 / 28.6 / 38.0 blocks at degree 2 / 4 / 6 on hex20 $16^3$, now 19.1 / 18.2 / 18.2; the loop figures in `benchmarks/results/options_study.json`, the unrolled ones a one-off measurement of the pre-13c code). Outputs equal the unrolled form to rounding. A loop body cannot share buffers with the code around it, so degrees ≥ 3 add a small term that `amg_memory` and `gmg_memory` model ([AMG](#memory-amg_memory), [GMG](#voxel-geometric-multigrid-gmg)); degree 2 is a one-trip loop that XLA removes. At the defaults the V-cycle times are unchanged and at degree 4 the loop is 2–6% slower than the unrolled form (one-off comparison with the pre-13c code; the loop times at degree 2 / 4 are in `benchmarks/results/options_study.json`: AMG hex20 $16^3$ 62 / 122 ms, GMG voxel $64^3$ 701 / 1252 ms fp64). Every smoother option is reachable from `solve_modes(..., precond_options=...)` ([`eigensolvers.md`](eigensolvers.md#options)).

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

- **Stopping:** the explicit residual $\|b - Ax\|_2 \le \mathrm{tol}\,\|b\|_2$ is checked every `CHECK` = 10 iterations. A column is `converged` when it meets tol and `stagnated` after `STALL_C` = 5 checks without a new minimum, once its best explicit residual is below $\|b\|_2$ (PR 15c): before any progress over $x = 0$ a plateau of the indefinite Krylov space is not a stall (a slice step plateaued at 1.29–1.33 $\|b\|$ for 100 iterations, then converged in 300); such a column runs on to `maxit` at worst (in a slice: that step's cap — `maxit_inner` on the first step, also after σ's one move, then at most 10× the first step's iterations — then "inner solves failed"; never seen on 23 AMG slices, where plateaus ended within 65% of a call's iterations). `maxit` gives neither. The iterate with the smallest explicit residual is returned (the last one, or the best earlier check: the 2-norm residual is not monotone in MINRES, which minimizes the $P^{-1}$-norm; with σ on an eigenvalue it drifted up to 40× above its minimum before the stall exit; fixed in PR 15a). The fix changes only the returned iterate and its residual, never the iterates or exits; a non-finite residual is still reported as NaN. Iteration counts are multiples of 10 when `maxit` is.
- **`info`:** as `pcg` without `replacements`.
- **Floor:** on the hex20 box below with Jacobi of $K + \sigma_b M$, the explicit residual stalls at $3.5$–$8.8\cdot10^{-14}$ (by right-hand side and run) with $\sigma_b$ between $\lambda_6$ and $\lambda_7$ and at about $1.2\cdot10^{-12}$ between $\lambda_{30}$ and $\lambda_{31}$ (plan prototype): it grows with the shift's index (PR 15's $\tau_\mathrm{eff}$).
- **Memory** (measured, plus `b`): 10 blocks with a diagonal operator (the carried $x$, best $x$, $r_1$, $r_2$, $y$, $w$, $w_2$, the body's $v$ and $Av$, the returned $x$); with the hex20 operator about 13 blocks above a bare apply (7 more than PCG).

## Block-ELL (`block_ell`, `BlockELL`)

`block_ell(A, b=3, dtype="float64", sliced=None, chunk=2048)` converts a SciPy sparse matrix (both dimensions divisible by $b$, square or rectangular) on the host, through BSR with $b \times b$ blocks, into a pytree `BlockELL`: `E(X)` maps `(n_cols, m)` blocks to `(n_dof, m)`, with `n_dof` the output row count (the operator size of a square matrix), `padding`, `n_buckets`. Rectangular matrices are AMG's transfers $P$ and $P^\top$; they never reach code that assumes a square operator.

- **Layout:** per bucket, `cols` `(r_pad, w)` int32 and `vals` `(r_pad, w, b, b)`; $w$ is the bucket's width (most blocks per row). Padding entries point at block column 0 with zero values (a row's own block column may not exist in a rectangular matrix).
- **Apply:** per bucket a `lax.scan` over row chunks gathers `X[cols]` as `(chunk, w, b, m)` and contracts with `vals`. Several buckets are concatenated and put back in row order by one gather.
- **Chunk:** `min(chunk, max(8, chunk * 27 // w), rows rounded up to 8)` block rows per bucket. Tiny coarse levels are not padded to 2048 rows, and a bucket wider than 27 blocks scans proportionally fewer rows, so square matrices up to width 27 keep 2048. The gather temp is
  $$\mathrm{chunk}\cdot w\cdot b\cdot m\cdot\mathrm{itemsize}$$
  bytes: about one vector block for 3×3 blocks, width 27 and 2048 rows. The width bound matters for AMG's restriction $P_0^\top$ (scalar, up to 429 entries per row at hex20 $44^3$): at chunk 512 its apply temp is 341 MB at $m = 25$, with the bound (108–128 rows) 135 MB and slightly faster.
- **Padding:** `padding` = stored blocks / structurally nonzero blocks (without the chunk round-up). Constrained nodes are width-1 rows of $A_P$, so clamped faces add padding: plain-ELL padding of the hex8 $A_P$ is 1.35 free and 1.59 clamped on $6^3$, 1.21 / 1.33 on $10^3$, 1.07 clamped on $44^3$.
- **Sliced fallback:** `sliced=None` uses one bucket (plain ELL) while its padding is $\le 1.5$, else one bucket per row width rounded up to a power of 2 (padding $< 2$); `True` / `False` force it. hex20 meshes exceed 1.5 and are sliced.
- **Block size** `b` is a parameter: 3 for PR 11's levels and AMG's first level, 1 for AMG's coarser levels and transfers.
- **dtype:** `"float32"` stores `vals` in fp32 for fp32 V-cycles; the apply casts them to the input's dtype (fp64 in computes in fp64).

## Voxel geometric multigrid (`gmg`)

`solve_modes` forwards `degree`, `coarse_degree` and `ratio` through `precond_options` ([`eigensolvers.md`](eigensolvers.md#options)).

```python
from jaxmodal.precond import gmg

P = gmg(
    op, free, sigma, degree=2, coarse_degree=4, ratio=30.0, dtype="float64"
)  # op: a voxel Stencil
x, info = pcg(ap_operator(op, free, sigma), b, P, tol=1e-8)  # or minres / lobpcg(precond=P)
```

`gmg` returns a pytree `GMG` (the multilevel cycle shared with `pmg`, below) whose call applies one V-cycle to an `(n_dof, k)` block, approximating $A_P^{-1}$ with $A_P = m(K + \sigma M)m + s(I - m)$ and $s$ = `ap_operator(op, free, sigma).s`. It applies $m \odot V(m \odot r)$: masking the input as well keeps it symmetric on the whole space. Transfers, level matrices, smoothers and the coarsest factor are leaves and the shapes depend only on the grid, so a new design on the same grid (new `s_K`, `s_M`, σ) never recompiles `pcg`, the V-cycle or the LOBPCG step, as long as the island deflation keeps its padded row bucket and islands neither appear nor vanish (see Island deflation). The fine operator is never assembled.

**Hierarchy** (host). For an axis with $n$ elements:
- $n \le 2$: the axis stops (identity along it);
- otherwise $c = \lceil n/2\rceil$ coarse elements with nodes at the fine nodes $X = (0, 2, \dots, 2(c-1), n)$; an odd $n$ gives one last coarse cell spanning a single fine element.

Each level carries its per-axis physical node positions $p$ ($p_0$ = the fine node index, $p_{l+1} = p_l[X]$), and the weights are linear in them: fine node $i$ between coarse nodes $X_k \le i \le X_{k+1}$ gets
$$w_1 = \frac{p_i - p_{X_k}}{p_{X_{k+1}} - p_{X_k}}, \qquad w_0 = 1 - w_1.$$
Index-space weights would be wrong from level 2 on, where a half-size tail cell can pair with a full cell (e.g. $n = 7$: level-1 nodes at 0, 2, 4, 6, 7, so the node at 6 gets $\tfrac13, \tfrac23$). So every transfer reproduces linear fields, rigid modes included. Coarsening repeats until every axis is stopped; that grid ($\le 3^3$ nodes) is the coarsest level.

A coarse cell is active when it holds an active fine cell; the coarse nodes are the corners of active coarse cells, numbered by `voxel_mesh(shape, mask=active)`. Every kept node has a nonzero prolongation weight at a node of an active fine cell, and every other coarse grid node has a zero $P$ column. **Deviation from the roadmap:** such coarse DOFs are dropped instead of kept "masked with diagonal $s$": a zero column makes them decoupled rows of $P^\top A_P P$ with no effect on the V-cycle.

**Galerkin coarse levels.** With $P_e = P_\text{loc}\otimes I_3$ the local prolongation of fine cell $e$ into its parent ($P_\text{loc}$ the $8\times8$ trilinear weights from the per-axis tables, VTK corner order), $m_e$ its 24 free flags and $D_e$ its owned nodal diagonal (below), the level-1 cell matrices are
$$C_c = \sum_{e\in c} P_e^\top\big(\mathrm{diag}(m_e)\,A_e\,\mathrm{diag}(m_e) + D_e\big)P_e, \qquad A_e = s_{K,e}K_{\phi(e)} + \sigma\rho_{\phi(e)}s_{M,e}M_e,$$
with $K_\phi$, $m_{ab}$ from `reference_element(op)` and $M_e = m\otimes I_3$ (consistent) or $\mathrm{diag}(\sum_b m_{ab})\otimes I_3$ (lumped). $D_e$ holds $d = \sigma\,n_\text{nodal}\,m + s(1 - m)$ (nodal masses and the constrained diagonal of $A_P$) on the DOFs of the nodes $e$ owns: each node is owned by the lowest-numbered element containing it, so its $d$ enters once. Levels $\ge 2$ sum $P_c^\top C_c P_c$ unmasked. Every level equals $P^\top A_P P$ of the PR 4 SciPy matrices to rounding (tests). The products contract the $(8,3,8,3)$ view of the cell matrices with the $8\times8$ $P_\text{loc}$, never forming $24\times24$ prolongations.

**Setup loop.** One jitted `lax.fori_loop` per level runs over chunks of `SETUP_CHUNK` = $2^{14}$ child cells (the last chunk clamped back and its repeated rows masked, no padded copies) and scatter-adds, from the same child matrices:
- the parent cell matrices;
- the level's smoother group blocks;
- on levels $\ge 1$, the level's 27-slot block-ELL values.

On the fine level the child matrices are formed per chunk from the phase-indexed $K_\phi$, so no $(n_\text{el}, 24, 24)$ array exists. All arrays enter the loop as arguments (closed-over arrays would become constants). Each level $\ge 1$ is a `BlockELL` (PR 10's fields, one bucket, so it shares the apply): 27 slots per node, slot = corner offset difference, missing neighbours pointing at the node itself with zero blocks; built on the device, not through the host `block_ell` (its host COO would be ~0.8 GB at 2M DOF, ~19 GB at 100M). The coarsest level is summed into a dense fp64 matrix and Cholesky-factored; `GMG.coarse` holds it as a `Cholesky` pytree (its call casts to fp64, runs `cho_solve`, and casts back).

**Transfers.** Prolongation (the `Transfer`'s call) scatters the coarse values into the full coarse grid (`unique_indices=True`), interpolates axis by axis with two-point gathers, and gathers the active fine nodes; restriction (`Transfer.restrict`) is its exact transpose (`jax.linear_transpose`).

**Smoothers: stopped-axis group blocks.** Point smoothing cannot damp the error that semi-coarsening leaves along stopped axes (plate $64^2\times1$ with node-block Jacobi: 240 iterations, prototype). A group is the set of nodes sharing coordinates on the coarsening axes; it has $k = \prod_{d\,\text{stopped}}(n_d + 1) \le 27$ nodes. The group blocks $D_l$ are the $3k\times3k$ diagonal blocks of the level operator (with $k = 1$: node blocks), inverted by PR 10's `_inverse_blocks` (absent slots of partial groups set to the identity). For $k > 1$ the apply is gather-only: gather, batched GEMM, then a precomputed inverse permutation (groups partition the nodes; 34 ms against 113 ms for a scatter-add at 2M DOF). Each level smooths with `chebyshev(A_l, D_l, lmax_estimate(A_l, D_l), degree_l, ratio)`, `degree` = 2 on the fine level and `coarse_degree` = 4 on the coarser ones: they cost about 1/8 of the fine level, and the extra steps cut iterations on the hard cases (SIMP field at its own $\lambda_1$ 116 → 87, plate $128^2\times1$ 186 → 136, spacing ratio 10 170 → 105) for about +0.5 $t_A$ per cycle at $87^3$ (4.7 → 5.2 $t_A$).

**V-cycle.**
$$x = S_l b,\qquad x \mathrel{+}= P_l\,V_{l+1}\big(P_l^\top(b - A_l x)\big),\qquad x \mathrel{+}= S_l(b - A_l x),$$
with the exact Cholesky solve on the coarsest level. Pre- and post-smoothing use the same symmetric $S_l$, so the V-cycle is symmetric. Chebyshev runs on $[1.1\tilde\lambda/\text{ratio}, 1.1\tilde\lambda]$ with $\tilde\lambda$ the 20-step Lanczos estimate; the V-cycle is positive definite provided $1.1\tilde\lambda \ge \lambda_\max(D_l^{-1}A_l)$ (then $\|I - S_lA_l\|_{A_l} < 1$), PR 10's safety assumption (measured $1.09$–$1.10\,\lambda_\max$ on every level of a binary SIMP box, a plate and a beam). When the fine grid already has every axis $\le 2$, it is smoothed once and then solved exactly. The recursion is unrolled at trace time.

**Island deflation.** A stiff part joined to the rest only through soft cells (a SIMP island) moves almost rigidly at tiny energy, and no geometric coarse space separates that motion from the body around it: the checked-in SIMP field has two islands, its $\lambda_1$ is one of their modes, and at that σ PCG needs 723 iterations without deflation. When $\max s_K \ge 10^3\min s_K$ (`ISLAND_CONTRAST`), `gmg` labels the node-connected components (cells sharing a face, edge or vertex) of the cells with $s_K > 10\min s_K$ and takes all but the largest, at most `MAX_ISLANDS` = 16, largest first. The threshold keeps each island's filtered gray halo, which matters: counting only cells above $10^{-3}\max s_K$ leaves 432 iterations. $Z$ holds each island's six rigid modes (translations, rotations about its centroid, constrained rows zero) and the call applies the balancing form
$$P = Q + (I - QA_P)\,mVm\,(I - A_PQ),\qquad Q = Z E^{+} Z^\top,\quad E = Z^\top A_P Z,$$
symmetric, and positive definite whenever $mVm$ is. $Z$ and $A_PZ$ are block-diagonal over the islands, so they are stored per support row (island nodes plus one ring of cells): the row's DOF, its island id and its 6 entries of $Z$ and of $A_PZ$, about 100 B per row (a row on two islands' rings appears twice). $Z^\top r$ is a segment sum over the island ids and $Zy$ a per-row gather of the island's coefficients, scattered into the output, so an apply adds no $A_P$ apply; $E^{+}$ is a Jacobi-scaled eigen-pseudo-inverse over a fixed $6\cdot$`MAX_ISLANDS` = 96 columns (absent islands, dependent or fully constrained columns drop). The row count is padded to a power of 2, at least 256 (padding rows point at DOF 0 with zero values), so a new design recompiles only when the row bucket changes or islands appear or vanish. Without islands `defl` is `None` and $P = mVm$. The setup applies $A_P$ once per island (6 columns), about 0.2 s per island at 2M DOF. The apply's per-row temporaries add about $3\,n_\text{rows}/n_\text{dof}$ blocks of V-cycle workspace (+3.3 blocks measured with 16 islands covering 55% of the cells at $87^3$, $n_\text{rows}$ padded to $2^{21}$); 11b's estimator counts it.

**Singular guard.** With σ = 0 and a singular $A_P$, the rounded coarsest Cholesky gives NaN or a finite factor that amplifies rigid components by ~$10^{13}$. At σ == 0, `gmg` raises `SingularShiftError` (a `ValueError` subclass, exported from `jaxmodal.precond`; `solve_modes`' σ pre-estimate catches it and starts at σ > 0) when it:
1. refuses any cell with `s_K` = 0 ("zero-stiffness cells at sigma = 0: ... use sigma > 0"): such cells add no stiffness and can cut the body apart;
2. calls `rigid_modes(voxel_mesh(shape, spacing, mask), free)` before any device setup and raises if it returns a column; rigid modes are linear, so they lie in the range of every $P$, and the coarsest level is singular exactly when $A_P$ is;
3. re-raises, chained, the `ValueError` of meshes `rigid_modes` rejects (edge or vertex contacts, more than `MAX_PARTS` parts): "cannot verify that A_P is nonsingular at sigma = 0 (...); use sigma > 0". This refuses a few SPD cases (a hinge clamped on both sides, many clamped islands); σ > 0 makes all of them SPD.

Two backstops act at any σ and raise a plain `ValueError` (not `SingularShiftError`): non-finite smoother blocks raise "smoother blocks are not finite" (e.g. cells with `s_K` = `s_M` = 0 leaving free nodes with no stiffness and no mass), and a non-finite coarsest factor raises "coarsest factor is not finite". A pivot-ratio threshold was rejected: legitimate SPD cases fall with size (clamped plate at σ = 0: $6\cdot10^{-7}$ at $128^2\times1$) and reach $1.6\cdot10^{-11}$ on SIMP designs with large void regions at σ > 0. The guard is exact only at σ == 0: a σ far below $\lambda_1$ passes it and gives a poorly conditioned V-cycle, so choose σ physically (the σ pre-estimate).

**Inputs.** `op` must be a `Stencil` (`TypeError` otherwise) in the `voxel_mesh` node order: a voxel mesh renumbered by `rcm`/`renumber` raises `ValueError` (its grid order is already narrow). A concrete σ < 0 or NaN raises; x64 must be on.

**Measured** (`benchmarks/gmg_study.py` → `benchmarks/results/gmg_study.json`; unit-length steel grids, spacing $1/\max(n)$, `xmin` clamped unless free, σ = $\lambda_1$ of the case from `reference_modes` unless noted, 4 right-hand sides, PCG to $10^{-8}$, max over the right-hand sides; the JSON also records `sigma_estimate`, which is within 10% of $\lambda_1$ on the bulky cases and up to 3800× too large as slenderness and contrast grow (plates $64^2$ and up, the $128\times1\times1$ beam and most SIMP fields 15–3800×; small plates and stubby beams within 2.5×), an input for 11b's σ pre-estimate):

| case | DOF | iterations |
|---|---|---|
| boxes $8^3$ / $16^3$ / $32^3$ / $64^3$ ($\lambda_1$ of $8^3$) | 2k – 824k | 13 / 13 / 13 / 13 |
| odd $9\times8\times8$ / $17\times8\times8$ / $33\times8\times8$ | 2.4k – 8.3k | 13 / 13 / 14 |
| free $32^3$ (first elastic $\lambda_1$) | 108k | 13 |
| plates $16^2$ / $32^2$ / $64^2$ / $128^2$ $\times1$ | 1.7k – 100k | 23 / 37 / 71 / 136 |
| plate $64^2\times2$ | 38k | 40 |
| beams $64\times2\times2$ / $128\times1\times1$ | 1.8k / 1.5k | 23 / 58 |
| filtered binary SIMP field $32^3$ (`tests/data/simp32.npy`, $s_K = \rho^3$, $s_M = \rho$), $\lambda_1$ of the solid $8^3$ box | 108k | 33 |
| the same field at its own $\lambda_1 = 15.1$ (an island mode; 2 islands deflated) | 108k | 87 |
| the same recipe, seeds 4 / 5, at their own $\lambda_1$ (4.07: 1 island; 2002: none) | 108k | 135 / 109 |
| unfiltered binary field (seed 3) at its own $\lambda_1 = 41.3$ | 108k | 452 |
| anisotropic spacing $h_z = h/2$ / $h/4$ / $h/10$, $32^3$ | 108k | 14 / 28 / 106 |

At $87^3$ (2.04M DOF, $\lambda_1$ of $8^3$): the build adds 1.12 GB of process peak over the operator + block Jacobi + $\lambda_\max$ baseline (the unchunked prototype added 22.7 GB), and one V-cycle at $m = 15$ takes 1.82 s against 0.348 s for one $A_P$ apply ($5.2\,t_A$, warm, best of 5 since PR 11b; the Mac budget model assumed $5.5\,t_A$). The setup transient is about 0.9 KB per fine node (level-1 cells, ELL values, level-2 cells) and the resident hierarchy about 450 B per fine node (150 B per DOF; 0.31 GB at $87^3$, ~15 GB at 100M DOF); at 100M DOF that is roughly 30 GB of transient, which fits a GPU only because `gmg` runs before LOBPCG allocates its blocks (as the rigid-mode setup does). The transfers work on the full box grid, so their temporaries scale with the box, not the active nodes (≈ 1/occupancy). Restriction is the transpose of the gathers, i.e. scatter-adds: 0.156 s against 0.026 s for prolongation on the fine level at $87^3$, $m = 15$ (a deferred GPU check). LOBPCG (10 modes, tol $10^{-6}$) takes 16 iterations at $87^3$ (the PR 11b acceptance run, GMG at the pre-estimated σ; the 11a prototype's 18 used σ = λ₁ of $8^3$).

**fp32 V-cycle** (PR 11b, `dtype="float32"`). The build runs in fp64 as before (the Galerkin sums, smoother inverses and $\lambda_\max$ need it); then the coarse `BlockELL` values, the smoother inverse blocks (and the $k = 1$ mask), the Chebyshev scalars and the transfer weights are cast to fp32, and the fp64 originals are dropped. Kept in fp64: the fine `APOperator` (the caller's Stencil, no copy; its apply follows the vector's dtype, [`operators.md`](operators.md)), the coarsest Cholesky factor (`Cholesky` casts the coarse right-hand side to fp64 and back), the deflation arrays and `free`. `levels[l][0] is levels[l][1].A` holds on every level. The cycle masks and pre-deflates in fp64, casts to fp32, runs the V-cycle, casts the correction back and post-deflates in fp64, so input and output stay fp64.
- Accuracy: on boxes the cycle differs from fp64 by about $4\cdot10^{-7}$ relative and stays symmetric to about $10^{-8}$ (energy-normalized $|u^\top Pv - v^\top Pu|/\sqrt{u^\top Pu\,v^\top Pv}$); on the thin plate $32^2\times1$ both are about $10^{-4}$ (its coarse levels amplify the rounding); it stays positive. LOBPCG (10 modes, tol $10^{-6}$) takes the same iterations as with the fp64 cycle (box $16^3$ 15/15, box $32^3$ 15/16, plate $32^2\times1$ 21/21, odd $33\times8\times8$ 14/14), with λ equal to $\le 3\cdot10^{-11}$. PCG to $10^{-8}$ with an fp32 cycle is not promised: the rounding asymmetry may stall it near $10^{-7}$.
- Cost at $87^3$, $m = 15$ (warm, best of 5, `gmg_study.json` "large"): 1.11 s against 1.82 s for the fp64 cycle (3.2 against 5.2 $t_A$, one fp64 $A_P$ apply 0.348 s); the fine-level applies run in fp32 too. The 2M-DOF solve takes 51 s instead of 61 s.
- Memory: V-cycle temporaries 6.5 instead of 13.6 fp64 blocks at $64^3$ (with PR 13c's Chebyshev loop, `benchmarks/results/options_study.json`; 0.48–0.50× at $16^3$–$87^3$), hierarchy storage 0.56×.

**Memory estimator.** `jaxmodal.memory.gmg_memory(mesh, m, dtype="float64", s_K=None, degree=2)` returns, in bytes, with the operator estimators' keys and composition rule:
- **`storage`**: an upper bound from shape arithmetic on the grid `gmg` builds from (the voxel grid cropped of empty trailing layers along the scan axis, as `grid_ids` does): the fine level exactly (smoother blocks, $9k$ entries per node in `dtype` for the group size $k$, group index arrays or the $k = 1$ mask, `free`); every coarse level as its full bounding box (ELL values $27\cdot9$ entries per padded row in `dtype`, int32 columns, smoother blocks); the transfers (int32 node maps, per-axis int64 indices and `dtype` weights); the coarsest factor $(3n_c)^2\cdot8$. With `s_K` it adds the island deflation (104 B per padded support row, plus $96^2\cdot8$). Measured on boxes and a plate: within 0.02% above the unique buffers of the built `GMG` (the operator's own leaves excluded).
- **`workspace`**: the V-cycle temporaries for PCG users, $c\,b$ plus the stencil's slab workspace in `dtype` ($b = 8\,n_\mathrm{dof}\,m$; $c$ = 15 fp64 and 7.5 fp32, the envelope of `memory_analysis` at $16^3$–$87^3$, which ranged 10.6–14.9 and 5.0–7.5 before PR 13c's Chebyshev loop lowered the default temporaries by about 1 block; $c$ is kept); with islands $+3\,n_\mathrm{rows}$ rows of $8m$ B. With the loop, the `workspace` estimate ($c\,b$ plus the slab workspace) is 39% above the measured temporaries at $64^3$ (13.55 blocks, `benchmarks/results/options_study.json`) and less at smaller sizes (about 7% at $32^3$). A fine-level `degree` > 2 adds `GMG_LOOP` = 1.2 fp64 blocks (0.7 with the fp32 V-cycle) to `workspace`, `step` and `step_low`: the loop carries about one more fine vector, +1.07 / +0.60 blocks of temporaries at $64^3$ (`benchmarks/results/options_study.json`) and pinned within 0.8–1× (fp32 0.7–1×) of the model at $16^3$ and $48^3$ by `test_gmg_loop_increment` (at $48^3$ also both fp64 steps; below $48^3$ the low-memory step rises up to 2.06 blocks, inside the `GMG_STEP` slack); `coarse_degree` is covered by the calibration, since the default 4 already runs the loop.
- **`build`**: the setup peak, 1650 B per fine node (1.12 GB at $87^3$, process peak above the baseline, resident hierarchy included; on the device in the per-space entries); for fp32 the same (the fp64 hierarchy and its fp32 copy are briefly alive together, but the copy, < 300 B per fine node, fits under the build's transient).
- **`step`**, **`step_low`**: the extra LOBPCG step bytes with GMG over Jacobi, `GMG_STEP` blocks (`lobpcg_memory`, [`eigensolvers.md`](eigensolvers.md#memory-model); re-fitted in PR 13c after the Chebyshev loop: 4.3 / 9.0 fp64, was 5.1 / 9.9).

**Acceptance** (PR 11b, `gmg_study.acceptance` → `gmg_study.json` "acceptance", `tests/test_api.py`): `solve_modes` on the $87^3$ steel box (2.04M DOF, `xmin` clamped), 10 modes, tol $10^{-6}$, defaults (GMG, pre-estimate, 9-block step, fp64 basis). Wall time of one cold subprocess (compilation included); peaks of two warm subprocesses against `solve_modes_memory`:

| V-cycle | wall time | σ pre-estimate + build | LOBPCG | iterations | warm peaks | estimate |
|---|---|---|---|---|---|---|
| fp64 | 60.9 s | 16.2 s (5 its) | 43.4 s | 16 | 6.23, 6.21 GB | 6.83 GB (+10%) |
| fp32 | 51.0 s | 16.7 s (5 its) | 32.9 s | 16 | 4.86, 4.79 GB | 5.44 GB (+12%) |

Both are far inside the 10-min budget and the ±20% band; the cold runs peak higher (6.65 / 5.25 GB, compilation) and are still below the estimates.

**Limitations** (tracked in [`docs/open_items.md`](open_items.md)).
- **Anisotropic spacing:** coarsening is by element count, not spacing; $h_z = h/10$ needs 106 iterations. Spacing-aware semi-coarsening needs line smoothers along long stopped axes.
- **Thin plates and slender beams:** at σ = $\lambda_1$ the iterations grow like $h^{-1}$ (single-layer plates 23 / 37 / 71 / 136 from $16^2$ to $128^2$). Coarse trilinear hex8 cells on a one- or two-element-thick layer get ever flatter and lock in bending, so the coarse correction misses the smooth bending error. Larger σ hides it (the same plates at the overestimated `sigma_estimate` took ≤ 60). AMG on the hex8 level (PR 13a) does not fix this (it makes the unstructured flat-element case worse, see `amg`); the remedy is the solid-shell element (M4) with its own AMG ([PR 18a](#amg-on-solid-shells-pr-18a)).
- **High contrast at low σ:** at its own $\lambda_1$ a SIMP field's lowest modes are spurious island and void-region modes of $s_K = \rho^3$, $s_M = \rho$ ($\lambda_1$ is $10^{-6}$–$10^{-3}$ of the solid's). Deflation removes the island modes, but soft-region error that trilinear coarse functions cannot separate from the stiff material remains: 87 on the checked-in field, 109–135 on other seeds, 452 on an unfiltered binary field (every case converges). A SciPy prototype with Jacobi-smoothed, phase-aware prolongation plus deflation reached 34 (checked-in field) and 55 (seed 4); such operator-dependent transfers widen the coarse stencils beyond 27 points; PR 13b's study runs `amg` on the checked-in field as an unstructured hex8 mesh. Topology optimization removes the spurious modes with a modified mass interpolation (PRs 21–23), and PR 23 re-runs the contrast test on a real design.

## Unstructured p-multigrid (`pmg`)

```python
from jaxmodal.precond import pmg

P = pmg(op, free, sigma, degree=2, ratio=30.0)  # op: an Unstructured operator, sigma > 0
x, info = pcg(ap_operator(op, free, sigma), b, P, tol=1e-8)  # or minres / lobpcg(precond=P)
```

`pmg` builds a two-level V-cycle for hex20/hex27 (and mixed) meshes:
- **Fine level:** the matrix-free $A_P$, smoothed by Chebyshev over node-block Jacobi.
- **Coarse level:** the hex8 Galerkin matrix $P^\top A_P P$ on the corner nodes, solved exactly by SciPy `splu` on the host. This solve is for development and tests only; `amg` (below) is the production variant, with AMG on the same matrix.

`pmg` has no type of its own. It returns a `GMG` (the shared cycle, below) with:
- one level `((A_P, S),)`;
- one `NodalTransfer`;
- `coarse = HostLU(...)`;
- no deflation.

Masking ($m\odot V(m\odot r)$), symmetry and the `precond` interface are therefore those of `gmg`. On an all-hex8 mesh, $P = I$ and the coarse matrix is the assembled hex8 $A_P$, so the coarse solve is exact and PCG converges in one iteration (in exact arithmetic; the test allows 2).

**Shared cycle.** `_vcycle` and `_vcycle_apply` are the same code for both preconditioners:
- `GMG.coarse` takes any pytree callable: `Cholesky` for `gmg` and `amg`, `HostLU` for `pmg`.
- The transfers can be any object with a prolongation call and a `restrict` method: the voxel `Transfer`, `NodalTransfer` or AMG's `ELLTransfer`.

**Transfer.** The coarse nodes are the nodes that are a corner (local index 0–7) of some element. They are numbered in increasing fine id, which keeps the RCM locality. Each fine node takes its parents and weights from the first element containing it:
$$w_{ac} = N^{(8)}_c(\xi_a) = \tfrac18\prod_{d=1}^3\big(1 + \xi_{a,d}\,\xi_{c,d}\big),$$
where $\xi_a$ is the type's reference node (`REFERENCE_NODES`, VTK order) and $\xi_c$ are the corners. This gives:
- a corner: itself, weight 1;
- an edge node: its two edge corners, ½ each;
- a hex27 face node: its four face corners, ¼ each;
- the hex27 centre: all eight corners, ⅛ each.

Tables and applies:
- **Parent table:** `(n_nodes, w)`, int32 parents and fp64 weights, padded with weight 0. $w$ is the largest parent count present: 1 for hex8, 2 with hex20, 8 with any hex27. On a mixed mesh with hex27, every node therefore pays 8 gathers.
- **Restriction:** gathers from the inverse table, `(n_coarse, w')` children per corner ($w'$ = 7 on hex20 boxes, 27 on hex27 boxes), built on the host.
- **Applies:** both are gathers unrolled over the table columns, so neither forms an $(n, w, 3, k)$ array. Restriction equals $P^\top$ by test, not by construction.

Accuracy:
- $P$ reproduces trilinear fields exactly on straight-sided elements whose midside nodes sit at the trilinear image of their reference position (`box_mesh`, graded `coords`).
- On curved elements it still spans a nested coarse space (trilinear in the reference coordinates on the fine geometry), so $P^\top A_P P$ remains a true Galerkin operator. Rigid rotations are then only approximately in that space, which matters for `amg`'s near-null space (evaluated at the corners, below).
- On a conforming mesh, every element containing a node gives it the same parents. This is assumed, not checked.

**Galerkin hex8 matrix.** The fine operator is never assembled. For element $e$:
- $P_\text{loc}$ is its local transfer ($k\times8$);
- $m_e$ holds its free flags;
- $A_e = s_{K,e}K_e + \sigma\rho_{\phi(e)}s_{M,e}M_e$, with $M_e$ consistent or lumped as the operator's `mass`.

Then
$$C_e = (P_\text{loc}\otimes I_3)^\top\big(\mathrm{diag}(m_e)\,A_e\,\mathrm{diag}(m_e) + D_e\big)(P_\text{loc}\otimes I_3), \qquad P^\top A_P P = \sum_e Z_e^\top C_e Z_e,$$
where $Z_e$ maps the element's 8 corners to the coarse numbering. $D_e$ is the owned diagonal $d = \sigma\,n_\text{nodal}\,m + s(1-m)$ on the nodes $e$ owns (each node's first element in block order), as in `gmg`.

Assembly (PR 13b's production assembler, `_hex8_galerkin(A, pattern=None) -> (A_c, pattern)`):
1. **Pattern** (`_pattern(op)`, host, topology only, a `Hex8Pattern`): PR 12's corner tables (`_tables`: corner ids, the `NodalTransfer`, owner flags), the node-block pattern $E^\top E$ of the element-to-corner incidence $E$ (int32 BSR `indptr`/`indices`, sorted), per block an `(n_pad, 64)` int32 table with the position of each of the element's 8 × 8 corner pairs in that pattern, and the transpose permutation of the pattern's blocks. Padded elements point one past the end (never −1, which JAX would wrap to the last block before dropping). The pattern is structural (every pair of corners sharing an element) and does not depend on rounding, storage mode or density.
2. **Values** (`_scatter_cells`, one jitted `lax.scan` per block over the operator's padded chunks): `element_matrices` ([`operators.md`](operators.md)) gives $s_K K_e$ and $\rho s_M M_e$; the scan contracts the $(c, n, 3, n, 3)$ view with $P_\text{loc}$ into $(c, 8, 8, 3, 3)$ cells and scatter-adds them (`mode="drop"`) into a carried device array `(nnzb, 3, 3)`. The cells never exist for a whole block at once.
3. **Symmetrization** on the host, in place: `v = np.array(data); v += v[tp].swapaxes(1, 2); v *= 0.5`. `v[tp]` is a copy taken before the add, and IEEE addition commutes, so blocks $ij$ and $ji$ come out exact transposes. Without it the scatter adds the $(i,j)$ and $(j,i)$ terms in different orders and leaves about $10^{-17}$ relative asymmetry. Two value-sized host arrays, one device copy.
4. The result is a SciPy BSR matrix with 3 × 3 blocks over the coarse DOFs $3c + i$. pyamg needs int32 indices; `check_int32` covers the coarse DOFs, the block count and the scalar count $9\,\text{nnzb}$.

The pattern is reusable for any values on the same mesh (`pattern=`, used by `amg`'s aggregate reuse). On the CPU the scatter is deterministic; on a GPU its summation order may vary at rounding level (deferred GPU check). The matrix is affine in σ: $G_K + \sigma G_M$, with the masked diagonal folded in. `_hex8_galerkin` is `amg`'s input and stays private; `_p_level` builds the fine level, $A_c$ and the pattern (whose `T` is the transfer) for both `pmg` and `amg`. PR 12's COO path (all cells materialized, CSR plus its transpose: 2.5 GB at 1M hex20, 9.0 GB at 1M hex8) is gone.

**Smoother and V-cycle.** $S$ = `chebyshev(A_P, block_jacobi(op, free, σ), lmax_estimate(...), degree, ratio)`, with degree 2 and ratio 30 by default (PR 10). The cycle is gmg's with one level:
$$x = S b,\qquad x \mathrel{+}= P\,A_c^{-1}P^\top(b - A_P x),\qquad x \mathrel{+}= S(b - A_P x).$$
- It is symmetric (the same $S$ twice, exact coarse solve).
- It is positive definite under PR 10's $\lambda_\max$ safety assumption.
- It runs in fp64; there is no fp32 option.
- One cycle costs $2d$ $A_P$ applies plus one coarse solve.

**Coarse solve (`HostLU`).** The host factors `splu(A_c.tocsc())` (CSC from the BSR) with the default COLAMD ordering. Inside the jitted cycle, `jax.pure_callback(..., vmap_method="sequential")` applies it; the result is declared with the input's shape and dtype (always fp64).

`HostLU` is a pytree with no leaves whose treedef is the instance itself, hashed by identity. This is the one closure exception the roadmap allows. Consequences:
- Every new `pmg` build recompiles its jitted callers (`pcg`, `lobpcg_step`) once; reusing a build does not.
- The jit caches hold the treedef, and with it the factor and the compiled executables, until `jax.clear_caches()`. `del` and `gc.collect()` do not free it. At hex20 $24^3$ the factor is about 1.5 GB. The p-MG tests and `benchmarks/pmg_study.py` clear the caches after each case.

`amg`'s device hierarchy has only array leaves and has neither issue.

Geometric nested dissection on the corner coordinates factors 4–8× faster: 3.5 s and 0.69 GB at $n = 24$, 17.8 s and 2.3 GB at $n = 32$. But `Unstructured` carries no coordinates, so COLAMD is kept for this test-only path ([`open_items.md`](open_items.md)).

**σ.** σ must be concrete and > 0. `pmg` has no mesh with which to prove $A_P$ nonsingular at σ = 0 (the BCs may leave rigid modes), so:
- σ == 0 raises `SingularShiftError`, the type `gmg` uses, so the σ pre-estimate (13b) can take the same fallback;
- σ < 0 or NaN raises `ValueError`.

Refusing σ = 0 has a cost. A pre-estimate started at σ = 0 needs fewer rounds and iterations on clamped bodies:

| case | from σ = 0 | from σ > 0 |
|---|---|---|
| cube | 1 round, 4 iterations | 2 rounds, 10 iterations |
| plate, $a = 4$ | 1 round, 20 iterations | 3 rounds, 60 iterations |

PR 13b can accept σ = 0 when `solve_modes` has found no rigid modes and every `s_K > 0` (open item).

**Measured** (`benchmarks/pmg_study.py` → `benchmarks/results/pmg_study.json`; jax 0.11.2, fp64, M3 Max CPU).

Setup:
- unit-length steel boxes, `xmin` clamped;
- PCG on $A_P(K + \sigma M)$ to $10^{-8}$ with 4 random right-hand sides (max over them);
- Chebyshev ratio 30;
- σ = $\lambda_1$ of the case (SciPy), except the h-sequences, which keep $\lambda_1$ of their $6^3$ box.

| case | DOF | PCG iterations |
|---|---|---|
| hex20 boxes $3^3$ / $6^3$ / $12^3$ / $24^3$, degree 2 | 624 – 182k | 14 / 14 / 14 / 14 |
| hex20 boxes $3^3$ / $6^3$, degree 3 | 624 – 3.7k | 11 / 10 |
| hex20 boxes $3^3$ / $6^3$ / $12^3$, degree 4 | 624 – 25k | 9 / 9 / 9 |
| hex27 boxes $3^3$ / $6^3$ / $12^3$ | 1k – 47k | 13 / 13 / 13 |
| hex20 beam $20\times2\times2$ ($L/h = 10$) | 1.9k | 16 |
| hex20 free $6^3$ (σ = first elastic λ) | 3.7k | 14 |
| hex20 curved $6^3$ (`tests/helpers.distort`) | 3.7k | 14 |
| hex20 $4^3$, `s_K` = $10^{-3}$ on a checkerboard | 1.3k | 19 |
| hex20 $6^3$, nodal masses of one element mass on a seeded half of the nodes | 3.7k | 14 |
| hex20 cube $8^3$, ratio 10 / 20 / 30 / 60 | 8.0k | 14 / 13 / 14 / 18 |
| hex20 single-layer plate $16^2\times1$, element aspect $a$ = 1 / 2 / 4 / 10 | 5.9k | 15 / 21 / 63 / not converged (stagnates after 620–720) |
| hex20 box $8^3$ with $h_z = h/a$, $a$ = 1 / 2 / 4 / 10 | 8.0k | 14 / 17 / 28 / 76 |

Degree 2 is the default. Per PCG solve (V-cycle $2d$ applies plus PCG's one):

| degree | $A_P$ applies | $t_A$ with a ~1 $t_A$ production coarse solve |
|---|---|---|
| 2 | 70 | ~84 |
| 3 | 70–77 | 80–88 |
| 4 | 81 | ~90 |

The cost is a tie here; in a full solve degree 2 is fastest (PR 13c, AMG, hex20 $24^3$, 20 modes: warm LOBPCG 7.0 / 8.8 / 10.3 s at degree 2 / 3 / 4 for 17 / 16 / 15 iterations, `benchmarks/results/options_study.json`), so it stays the default. Since PR 13c's Chebyshev loop the workspace no longer grows with the degree. At 1M hex20 the V-cycle costs 5.4–5.6 $t_A$ (varying between runs) with 11.74 blocks of temporaries, unchanged by the loop (`benchmarks/results/options_study.json`, `cost`). At hex20 $24^3$ ($m = 25$): $t_A$ = 53 ms; the V-cycle takes 987 ms, of which the test-only `splu` solve is 765 ms (14.4 $t_A$) and the rest 4.2 $t_A$ ($2d$ = 4 applies plus transfers). The build takes 13.3 s, mostly `splu`. PR 13b's `_hex8_galerkin` at hex20 $44^3$ takes 0.72 s with a 0.42 GB process peak (PR 12's COO path: 1.4 s, 2.5 GB); `amg_study.setup` measures it.

**LOBPCG** (10 modes, tol $10^{-6}$, σ = $\lambda_1$; times include compilation; eigenvalues agree with the Jacobi run to $\le 7\cdot10^{-11}$):

| mesh | DOF | Jacobi | p-MG degree 2 |
|---|---|---|---|
| hex20 $12^3$ | 24,843 | 435 iterations, 6.0 s | 15, 1.2 s |
| hex20 beam $20\times2\times2$ | 1,863 | 309 iterations, 1.6 s | 14, 0.3 s |
| hex27 $16^3$ | 107,811 | 509 iterations, 21.6 s | 15, 3.4 s |

**Limitations** ([`open_items.md`](open_items.md)).
- **Flat elements:** the hex8 coarse space locks in bending when elements are flat ($a \gg 1$), so the coarse correction misses the smooth bending error. See the table above; the hex27 plate $32^2\times1$ at $a = 10$ needs > 1000 iterations and does not converge. Slenderness alone with $a = 1$ is fine (beam: 16). AMG on the hex8 level makes it worse (below). Thin structures should use `ssh8` elements, whose AMG ([PR 18a](#amg-on-solid-shells-pr-18a)) handles them; `pmg` refuses `ssh8`.
- **`splu` scaling** (coarse hex8 matrix of an $n^3$ box, COLAMD):

  | $n$ | coarse DOF | factor time | $L + U$ | process peak |
  |---|---|---|---|---|
  | 16 | 15k | 1.0 s | 0.2 GB | 0.8 GB |
  | 24 | 47k | 14 s | 1.5 GB | 3.9 GB |
  | 32 | 108k | 140 s | 6.2 GB | 18 GB |

  The tests therefore stay at or below the 47k coarse DOF of hex20 $24^3$. `amg` replaces this solve.

## Unstructured AMG (`amg`)

`solve_modes` forwards `degree`, `ratio` and `max_coarse` through `precond_options` ([`eigensolvers.md`](eigensolvers.md#options)).

```python
from jaxmodal.precond import amg

P = amg(op, free, sigma, mesh.points, degree=2, ratio=30.0, max_coarse=1000)  # sigma > 0
x, info = pcg(ap_operator(op, free, sigma), b, P, tol=1e-8)  # or minres / lobpcg(precond=P)

P0 = amg(op, free, 0.0, mesh.points, nonsingular=True)  # clamped: no rigid modes left
P, agg = amg(op, free, sigma, mesh.points, return_aggregates=True)
P2 = amg(op.scaled(s_K=rho2**3, s_M=rho2), free, sigma, mesh.points, aggregates=agg)  # reuse
```

`amg` is the production preconditioner for `Unstructured` operators (PR 13a; `solve_modes`' default for unstructured meshes since 13b). It returns a `GMG`, the shared V-cycle, with one flat hierarchy:
- **hex20/hex27 or mixed meshes:** level 0 is `pmg`'s p-level (the matrix-free $A_P$ with Chebyshev over node-block Jacobi, and the `NodalTransfer`). Level 1 is the hex8 Galerkin matrix $A_c = P^\top A_P P$.
- **Solid-shell meshes** (`ssh8`, or `ssh8` + `hex8`; PR 18a): level 0 is the matrix-free $A_P$ with Chebyshev over additive Schwarz on column patches, then shell SA levels with 9 candidates per aggregate and Schwarz smoothers ([below](#amg-on-solid-shells-pr-18a)).
- **All-hex8 meshes:** $P = I$, so there is no p-level. Level 0 is the matrix-free $A_P$ with Chebyshev over `block_jacobi`, and its transfer is the first AMG prolongator $P_0$, set up on the assembled $A_P$.
- **Below that:** the smoothed-aggregation levels, down to `max_coarse` DOF, then a dense fp64 `Cholesky` of the coarsest matrix.

When the hex8 level already has ≤ `max_coarse` DOF there is no AMG level: hex20/hex27 meshes get two-level p-MG with an exact Cholesky coarse solve (equal to `pmg`), and all-hex8 meshes get `levels = ()`, a direct solve (PCG converges in one iteration in exact arithmetic). Solid-shell meshes likewise get `levels = ()`, a dense Cholesky of $A_b = A_P$.

Arguments and rules:
- **`points`** are the mesh coordinates `(n_nodes, 3)` **in the operator's node order** (for a renumbered mesh, the renumbered mesh's `points`). A wrong shape raises `ValueError`; a wrong order cannot be detected.
- **σ** is concrete and ≥ 0; σ < 0 or NaN raise `ValueError`. σ = 0 raises `SingularShiftError` unless `nonsingular=True` (13b), the caller's statement that $A_P(0)$ is nonsingular: no rigid modes left by the BCs and every `s_K > 0`. `amg` checks `s_K > 0` itself (`ValueError`) and trusts the rest; the coarsest factor's finiteness check stays as a backstop, but a singular coarsest matrix can factor with rounding-level pivots, hence the explicit flag. `solve_modes` passes `nonsingular=(k == 0)` with $k$ the rigid-mode count. `pmg` keeps refusing σ = 0.
- **`degree`, `ratio`:** Chebyshev degree and ratio on every level.
- **`max_coarse`:** coarsening stops at this size (ignored with `aggregates=`).
- **`return_aggregates`, `aggregates`:** see [Aggregate reuse](#aggregate-reuse) below.

**Near-null space.** $B$ is $(3n_c, 6)$: the translations and rotations $e_j\times(x - \bar x)$ at the corner coordinates $x$, with the rows of constrained corner DOFs zeroed (masked rigid modes). On straight elements it spans the corner restriction of `rigid_modes(mesh)`. On curved elements rotations lie in the hex8 coarse space only approximately; restricting the fine rigid modes was considered and not done, because the curved case converges in the same count as the straight one (14).

**Setup loop** (`_sa_levels`, host, pyamg primitives). Per level, while the matrix has more than `max_coarse` rows:
1. $A_z$: $A$ with explicit zeros removed (the hex8 Galerkin pattern is structural).
2. Strength: `symmetric_strength_of_connection` with θ = 0, on 3×3 node blocks on the first level and scalar below.
3. `standard_aggregation`, then `fit_candidates` (tentative prolongator $T$ and coarse $B$ by per-aggregate QR).
4. **Zero-column drop:** columns of $T$ with zero norm are removed with their rows of the coarse $B$. They come from aggregates whose rows of $B$ are zero or rank-deficient (fully constrained nodes, rank-deficient random masks). pyamg keeps them, and its own solver hides the resulting zero rows with a pseudo-inverse coarse solve; dropping them keeps every coarse matrix SPD, so the dense Cholesky works.
5. Stop if more than 0.9 of the rows would remain (no progress).
6. $P = (I - \tfrac{\omega}{\rho} D^{-1}A_z)\,T$: `jacobi_prolongation_smoother` with $\omega = 4/3$, $\rho$ = pyamg's spectral radius estimate of $D^{-1}A_z$, and $D$ the 3×3 block diagonal on the first level (scalar below).
7. Next matrix $P^\top A_z P$, symmetrized.

This is pyamg's smoothed aggregation with the zero-column drop added; the first level's $P$ equals `smoothed_aggregation_solver`'s (node blocks, θ = 0, block weighting, no candidate improvement) exactly (test). Coarse levels are scalar because after the drop an aggregate may keep fewer than 6 columns. Other rules:
- **Determinism:** the loop runs under a local `np.random.seed(0)` (pyamg's ρ estimate starts from `np.random`; without it two setups differ by about $10^{-4}$) and restores the caller's global state.
- **Coarsest cap:** if coarsening stalls above 4·`max_coarse` rows, `amg` raises `ValueError` instead of factoring a huge dense matrix. A non-finite coarsest factor or level smoother raises `ValueError`.

**The constrained corners are coarse unknowns.** On the hex20 → hex8 level, a free midside node interpolates from a constrained corner, so $P^\top A_P P$ couples constrained corner DOFs to free ones. The AMG cycle therefore solves all coarse DOFs; masking them inside the cycle made standalone PCG stagnate (> 1000 iterations). Their rows of $B$ are still masked: those DOFs carry energy of order $s$, not near-null energy.

**Constrained rows of $P_0$ (all-hex8).** These rows are exactly 0, so the fine mask is the only mask needed:
- a fully constrained node has no off-diagonal entries in $A_z$, so it stays unaggregated and its tentative row is empty;
- a partially constrained node's constrained rows of $B$ are masked, and the block-Jacobi row of a constrained DOF is diagonal, so smoothing keeps them zero.

With unmasked $B$ these rows would be nonzero (no measured change in the counts).

**Device format.**
- **First AMG level** (hex20/hex27): `block_ell(A_c, b=3)` with Chebyshev over its 3×3 diagonal-block Jacobi (all DOFs active).
- **Coarser levels:** `block_ell(A, b=1)` with scalar Jacobi under Chebyshev.
- **Transfers:** `ELLTransfer(P, PT)`, both stored as rectangular scalar block-ELL, so prolongation and restriction are gathers (no scatter-add). Restriction equals $P^\top$ by test.
- The pytree holds device arrays only (and `BlockELL`'s `padding`, a host float statistic by PR 10's design): the SciPy matrices and pyamg intermediates are host locals, freed when `amg` returns.

The V-cycle is symmetric and positive definite on the free DOFs: the same Chebyshev smoother before and after on every level and an exact coarsest solve.

### Aggregate reuse

`amg(..., return_aggregates=True)` returns `(P, agg)`; `agg` is an `AMGAggregates` (host data): the `Hex8Pattern`, the tentative prolongators $T_l$ (after the zero-column drop) and coarse candidates per SA level, a copy of `free` and a connectivity fingerprint (per block: type, element count and a hash of the DOF table, so the `hex8` box an `ssh8` mesh was converted from does not match). On solid-shell meshes `tents` is empty and `shell` holds the fiber groups and, per level, the group → aggregate map, $T_\ell$, its identity (zero-column) flags and the shell-group flags; nothing is dropped ([below](#amg-on-solid-shells-pr-18a)). `amg(..., aggregates=agg)` skips the pattern, the corner tables, strength, aggregation and the tentative fit; it recomputes the cells, the smoothed prolongators (`jacobi_prolongation_smoother` reads its strength argument only when filtering, so it gets `None`), the Galerkin products and the device export, under the same local seed (the spectral radius estimate is the only consumer of the random stream, so skipping aggregation does not shift it). Another `free` or another connectivity raises `ValueError`; `points` are not checked.
- **New values, same geometry** (σ, `s_K`, `s_M`, materials): with θ = 0 the first-level aggregates depend only on the pattern, the mask and $B$, and the coarser ones on the pattern of $P^\top A P$, which values change only through exact cancellations. A reused setup therefore equals a fresh one (test: the V-cycle to $10^{-12}$ after a 20% density perturbation that moves the smoothed $P$ by 12–13%, and σ = 0 aggregates reused at σ > 0).
- **Moved points** (shape optimization): reuse keeps the old near-null space $B$. The result is still an SPD V-cycle and only approximates a fresh setup (test: within 1.2× its iterations after a 2% node perturbation).
- It saves the topology part of a build (about 6–11% of a warm build at $24^3$ and up: 3.2 s against 3.6 s at 1M hex20, 10.2 s against 11.2 s at 1M hex8; table below) and pins the hierarchy sizes. `solve_modes` does not use it; it is for PRs 21–23.

### Measured

Source: `benchmarks/amg_study.py` → `benchmarks/results/amg_study.json` (jax 0.11.2, fp64, M3 Max CPU). Setup as in `pmg`'s table; `max_coarse` = 100, so that every size has an AMG level.

| case | DOF | AMG level sizes (hex8 level first) | `pmg` (exact coarse) | `amg` |
|---|---|---|---|---|
| hex20 boxes $3^3$ / $6^3$ / $12^3$ / $24^3$ | 624 – 182k | 192→48 / 1029→162→6 / 6591→750→48 / 46875→4374→162→6 | 14 / 14 / 14 / 14 | 15 / 14 / 14 / 14 |
| hex27 boxes $3^3$ / $6^3$ / $12^3$ | 1k – 47k | 192→48 / 1029→162→6 / 6591→750→48 | 13 / 13 / 13 | 13 / 13 / 13 |
| hex8 boxes $6^3$ / $12^3$ / $24^3$ / $48^3$ | 1k – 353k | 1029→108→6 / 6591→600→30 / 46875→3888→162→6 / 352947→27744→1176→48 | — | 13 / 13 / 13 / 13 |
| hex20 beam $20\times2\times2$ | 1.9k | 567→42 | 16 | 16 |
| hex20 free / curved / nodal masses $6^3$ | 3.7k | 1029→162→6 | 14 / 14 / 14 | 14 / 14 / 14 |
| hex20 $4^3$, `s_K` checkerboard $10^{-3}$ | 1.3k | 375→48 | 19 | 20 |
| hex20 plate $16^2\times1$, $a$ = 1 / 2 / 4 / 10 | 5.9k | 1734→216→24 | 15 / 21 / 63 / not converged (stagnates at 620–720) | 36 / 85 / 229 / not converged (> 1000) |
| hex20 box $8^3$, $h_z = h/a$, $a$ = 1 / 2 / 4 / 10 | 8.0k | 2187→162→6 | 14 / 17 / 28 / 76 | 14 / 20 / 52 / 244 |

**Cost at hex20 $44^3$** (1,075,275 DOF, default `max_coarse`, levels 273375 → 20250 → 750, $m = 25$):
- $t_A$ = 304 ms.
- The AMG cycle (everything below the p-level) costs 1.30 $t_A$; the first-level transfers cost 0.20 $t_A$ ($P$) and 0.18 $t_A$ ($P^\top$) per apply. Operator complexity 1.15.
- The whole V-cycle costs 5.4–5.6 $t_A$ between runs ($2d$ = 4 fine applies, the p-level transfers and the AMG cycle).
- V-cycle temporaries: 11.7 fine blocks (`memory_analysis`).
- Setup 4.1 s: the hex8 Galerkin matrix 0.66 s (PR 13b's assembler; 1.3 s with PR 12's), the SA loop 1.76 s, the rest the device export and the fine level.
- Device storage of the AMG levels and transfers: 0.48 GB (levels 0.24, transfers 0.24, coarsest factor 4.5 MB), int32 columns included. Padding: levels 1.05 / 1.24, $P$ 1.34 / 1.37, $P^\top$ 1.21 / 1.41.

**Setup** (`amg_study.setup`, one process per size, `xmin` clamped, σ = 1e6; times include compilation for the first two columns, peaks are the process peak over the built operator):

| mesh | DOF | assembler (time, peak) | `amg` (time, peak) | warm fresh / reused build |
|---|---|---|---|---|
| hex20 $12^3$ | 25k | 0.10 s, 0.06 GB | 1.0 s, 0.07 GB | 0.08 / 0.07 s |
| hex20 $24^3$ | 182k | 0.19 s, 0.09 GB | 1.9 s, 0.25 GB | 0.59 / 0.54 s |
| hex20 $44^3$ | 1.08M | 0.72 s, 0.42 GB | 4.9 s, 1.35 GB | 3.57 / 3.19 s |
| hex27 $12^3$ | 47k | 0.11 s, 0.05 GB | 1.1 s, 0.07 GB | 0.12 / 0.11 s |
| hex27 $24^3$ | 353k | 0.25 s, 0.10 GB | 2.0 s, 0.27 GB | 0.79 / 0.72 s |
| hex8 $24^3$ | 47k | 0.15 s, 0.08 GB | 1.5 s, 0.19 GB | 0.47 / 0.44 s |
| hex8 $48^3$ | 353k | 0.55 s, 0.50 GB | 5.1 s, 1.42 GB | 3.77 / 3.46 s |
| hex8 $68^3$ | 986k | 1.48 s, 1.42 GB | 12.3 s, 3.96 GB | 11.18 / 10.19 s |

**SIMP fields** (`amg_study.simp`: `gmg_study`'s $32^3$ density fields as unstructured hex8 meshes, $s_K = \rho^3$, $s_M = \rho$, `xmin` clamped; PCG iterations to $10^{-8}$, range over the right-hand sides; GMG on the voxel operator for comparison):

| field | σ | GMG | AMG, `max_coarse` 1000 | AMG, `max_coarse` 100 |
|---|---|---|---|---|
| filtered SIMP $32^3$ (`simp32.npy`) | own $\lambda_1$ | 87 | 789–803 | 900–911 |
| filtered SIMP $32^3$ | $\lambda_1$ of the solid $8^3$ box | 33 | 70–72 | 72–75 |
| the same recipe, seed 4 | own $\lambda_1$ | 133–135 | 646–761 | 720–737 |
| the same recipe, seed 5 | own $\lambda_1$ | 107–109 | 136–137 | 151–152 |
| unfiltered binary, seed 3 | own $\lambda_1$ (41.25) | 413–450 | 446–447 | 447–451 |

On the filtered fields at their own $\lambda_1$ AMG is far behind: GMG deflates the stiff islands, while AMG's near-null space is the global rigid modes and θ = 0 aggregates across void and solid. On the binary field the two are level. A strength threshold θ = 0.02 (plan prototype) brings the checked-in field at its own $\lambda_1$ to 49–51, but at the solid's $\lambda_1$ its coarse levels turn dense and coarsening stalls, and it raises AMG storage 34–56% on the 13a cases; 13b keeps θ = 0, which also keeps the reuse equality ([`open_items.md`](open_items.md)).

### AMG on solid shells (PR 18a)

`amg()` takes the solid-shell path when any block of the operator is `ssh8`: `ssh8` alone (also T-junctions and L-folds) or with `hex8` (stiffeners, hex8 solids glued to shells). A mesh with `ssh8` and `hex20`/`hex27` blocks raises `ValueError` ("solid-shell AMG supports ssh8 and hex8 blocks"; the meshes are nonconforming). Hex-only meshes run the code above unchanged. The signature, `return_aggregates` and `aggregates=` are the same; `solve_modes` and `amg_memory` still refuse AMG on `ssh8` until PR 18b (`jacobi` stays `solve_modes`' choice there). Code: `precond/shell.py` (host setup), `precond/schwarz.py` (the smoother).

**Why the hex8 path fails.** Measured in the 18a prototypes (plan measurements, not in the JSON; SS plate $n^2\times1$, PCG to $10^{-6}$ at σ = $\lambda_1$):
1. *Near-null space:* rigid modes do not represent plate bending on coarse aggregates.
2. *Smoothers:* once $h/t \gg 1$, point, node-block and fiber-block Jacobi lose the membrane–bending coupling, which lives inside the element. $n = 32$, $L/t$ = 10 / 100 / 300 / 1000: fiber blocks 10 / 17 / 51 / 183, element patches 11 / 13 / 16 / 38.
3. *Coarse levels:* each coarsening widens aggregates about 3× in-plane at the same thickness, so block Jacobi on the coarse levels fails the same way ($n = 64$, $L/t = 1000$: two-level 22, coarse 9×9 block Jacobi 79, coarse patches 24).
4. *Bending candidates must be one field per level:* frames fixed on the fine level give piecewise fields on level 1 (80 against 26).
5. *Bending is fourth order:* with one smoothing step on every level the counts grow with depth (26 / 40 at $n$ = 64 / 128); two steps below level 0 give 21 / 18 at $n$ = 128 / 256.
6. *Stacks need column patches:* an element patch covers $1/k$ of a $k$-layer stack (3 layers, $L/t$ = 10 / 100 / 300 / 1000: 10 / 43 / 322 / > 2000; column patches 11 / 15 / 40 / 117 in the prototype, shipped counts in the table below).

**Groups.** Level 0 works on the corner space ($A_b = A_P$, since every `ssh8` node is a corner; `_hex8_galerkin` is used directly). Its groups are the through-thickness **fibers**: connected components of the node pairs $(a, a+4)$, $a = 0..3$, over the `ssh8` elements (bottom face 0–3, top 4–7, [`elements.md`](elements.md)). One layer gives node pairs, $k$ layers give $(k+1)$-node fibers, and at a junction the members' fibers merge into one group. Nodes in no `ssh8` element (`hex8` nodes) are singleton groups. On level $\ell \ge 1$ the groups are the level-$\ell$ aggregates, each with 9 coarse DOFs.

**Size guard.** A level-0 patch (below) of more than 384 DOFs raises `ValueError` ("ssh8 patch of N DOFs: check solid_shell's direction, or the stack is too deep"). That allows 31-layer stacks and multi-layer junctions up to 6 layers per member (a T of two 5-layer members has $s = 252$), and catches a consistently wrong thickness axis on larger meshes (`direction=(1, 0, 0)` on a $40^2$ plate chains whole rows: $s = 492$). A wrong axis on smaller meshes is not detected; the counts are then poor.

**Aggregation.** pyamg's `standard_aggregation` on a group graph, without strength of connection:
- level 0: $G^\top S G$ with $S$ the `Hex8Pattern` structure and $G$ the node → group indicator over the **free** groups only (a group with a free DOF), diagonal removed, int32;
- level $\ell \ge 1$: the one-ring $\mathrm{Agg}^\top S_{\ell-1}\,\mathrm{Agg}$ of the previous group graph (the smoothed $P$ below widens the matrix pattern to a two-ring, which would cost 7× in patch storage).

Free groups left alone (separate parts, coarse leftovers) become singleton aggregates; fully constrained groups stay out (empty tentative rows). The graphs depend on the mask, not the values.

**Candidates.** Per level, 9 columns of $B^{(\ell)}_\text{fine}$ on the corner DOFs: the masked rigid modes and three constant-curvature Kirchhoff fields (also masked) of each node's level-$(\ell+1)$ aggregate, in the aggregate's frame $(t_1, t_2, n)$ with local coordinates $(X, Y, Z)$ of $x - \bar x$:
$$\kappa_{11}: w = \tfrac{X^2}{2},\ u_1 = -ZX; \qquad \kappa_{22}: w = \tfrac{Y^2}{2},\ u_2 = -ZY; \qquad \kappa_{12}: w = XY,\ u_1 = -ZY,\ u_2 = -ZX,$$
with $w$ along $n$ and $u_\alpha$ along $t_\alpha$.
- **Sign-free frame:** per node $T_a = \sum v v^\top$ over the thickness edges $v$ of its `ssh8` elements; $n$ is the dominant eigenvector of $\sum_{a \in \text{agg}} T_a$ (batched `eigh`), $t_1 = n\times e/\|n\times e\|$ with $e$ the axis least aligned with $n$, $t_2 = n\times t_1$. The span of the three fields does not depend on the sign of $n$, so flipped element orientations (the sweep on curved shells) do not matter. Aggregates without `ssh8` nodes get zero curvature columns.
- **Per level:** the fields of level $\ell$ are projected through the tentatives, $B_\ell = (T_0\cdots T_{\ell-1})^\top B^{(\ell)}_\text{fine}$ (exact for the rigid modes, and for the curvatures on flat plates and complete-fiber stacks), then `fit_candidates` gives $T_\ell$ with 9 columns per aggregate.
- **Identity rows:** columns of $T_\ell$ that `fit_candidates` writes as exact zeros (rank-deficient aggregates: hex groups, constrained or collinear fibers) give zero rows and columns of $A_{\ell+1}$. Their diagonal is set to the mean diagonal of the other rows, so the coarse matrix stays SPD with sizes a multiple of 9; these DOFs are inert in the V-cycle (zero restriction, decoupled, zero next prolongator rows). This replaces the zero-column drop of the hex8 path.

**Prolongators.** $P_\ell = (I - \tfrac{\omega}{\rho} D^{-1}A_\ell)^{d_\ell}\,T_\ell$ with $\omega = 4/3$, $D$ the point diagonal, $\rho$ pyamg's `approximate_spectral_radius` of $D^{-1}A_\ell$ (under the local seed) and $d_0 = 1$, $d_\ell = 2$ below. $A_{\ell+1} = P_\ell^\top A_\ell P_\ell$, symmetrized, as BSR(9). The stop rules (`STALL`, the 4·`max_coarse` cap) are the hex8 path's, counted with identity DOFs.

**Smoothers: additive Schwarz under Chebyshev.** On every level Chebyshev (`degree`, `ratio`) runs over
$$D^{-1} r = \sum_i R_i^\top A_i^{-1} R_i\, r, \qquad A_i = R_i A R_i^\top,$$
which is SPD (principal submatrices of an SPD matrix), so the V-cycle stays symmetric and positive definite.
- **Level 0:** a column patch per `ssh8` element: the DOFs of its 4 fibers ($12(k+1)$ for $k$ layers, 24 on one layer), deduplicated across the elements of a column by their set of groups; a node patch (3 DOFs) per hex-only group.
- **Level $\ell \ge 1$:** per shell group, its 9 DOFs plus the non-identity DOFs of its one-ring neighbours that are shell groups too ($s \le 81$ on a single layer); per hex group its own 9 DOFs. Identity DOFs are thus only in their own patch (Schwarz eigenvalue exactly 1), and hex regions do not pull 20+ neighbours into ring patches.
- **Bound:** a coarse patch above $\sqrt{\texttt{chunk\_bytes}/8}$ = 2048 DOFs raises (ring patches have no fiber guard; they can grow at hubs such as radially meshed disk centres).
- **Build:** the dense $A_i$ are read from the BSR matrix by sorted block keys (`np.searchsorted`, chunked, no CSR copy), inverted on the device by `_inverse_blocks` (Cholesky, padded slots the identity), and stored as packed upper triangles, $s(s+1)/2$ floats per patch. A non-finite inverse raises `ValueError`.
- **Size classes:** one class per distinct patch size, then the adjacent pair with the least added padding is merged while there are more than 8 classes or the padded bytes stay ≤ 1.1× the natural bytes. Each class is one scan per apply over balanced chunks of at most $\texttt{chunk\_bytes}/(8 s \max(s, 24))$ patches (`schwarz.layout`; `schwarz.nbytes` gives the device bytes, for PR 18b's `amg_memory`). Pad slots gather 0 and their scatter is dropped.
- **Apply:** a `lax.scan` per class: gather, unpack, `einsum`, scatter-add into the carry. Its temporaries are bounded by `chunk_bytes` (the unpacked chunk) and `chunk_bytes`·$k$/24 (gathers), independent of $n$.
- **Stacks** cost about $24(k+1)$ packed floats per DOF on level 0 (96 at $k = 3$, 264 at $k = 10$).

**Reuse.** `AMGAggregates.shell` holds the fiber groups and, per level, the group → aggregate map, $T_\ell$ and its identity and shell-group flags. `aggregates=` skips fibers, aggregation, frames and fits; it rebuilds the patch sets and recomputes $A_b$, the smoothed $P$, the Galerkin products, the patch inverses and the export. New materials, `s_K`, `s_M` or σ give a fresh setup's V-cycle (test: $10^{-12}$); another connectivity or `free` raises.

**Measured** (`benchmarks/shell_amg_study.py` → `benchmarks/results/shell_amg_study.json`, jax 0.11.2, fp64, M3 Max CPU). PCG to $10^{-6}$ on 4 fixed-seed masked columns (max over them), steel (except the plies case: PR 16's three-ply orthotropic laminate), σ = the Kirchhoff $\lambda_1$ on flat SS plates ($D\pi^4(2/L^2)^2/(\rho t)$; clamped: coefficient 35.99), else an `eigsh` $\lambda_1$ (128² panels: of the 64² version); `max_coarse` 1000 on the $\ge 64^2$ plates, else 100. The tests gate each count at 1.5×.

| case | DOF | coarse level sizes | iterations | build (s) |
|---|---|---|---|---|
| flat SS plate $128^2$, $L/t = 10$ | 99,846 | 16,641 / 2,025 / 225 | 12 | 2.2 |
| flat SS plate $128^2$, $L/t = 1000$ | 99,846 | 16,641 / 2,025 / 225 | 21 | 2.2 |
| 90° panel $128^2$, $L/t = 10$ | 99,846 | 16,641 / 2,025 / 225 | 11 | 2.2 |
| 90° panel $128^2$, $L/t = 1000$ | 99,846 | 16,641 / 2,025 / 225 | 30 | 2.2 |
| flat $64^2$, $L/t = 100$ | 25,350 | 4,356 / 576 | 14 | 1.1 |
| flat $128^2$, $L/t = 100$ | 99,846 | 16,641 / 2,025 / 225 | 13 | 2.2 |
| flat $256^2$, $L/t = 1000$ | 396,294 | 66,564 / 7,623 / 918 | 19 | 5.7 |
| flat $32^2$, $L/t = 50$ | 6,534 | 1,089 / 144 / 36 | 13 | 1.4 |
| 3 plies in one element, $32^2$, $L/t = 50$ | 6,534 | 1,089 / 144 / 36 | 12 | 1.3 |
| 90° panel $32^2$, $L/t = 50$ | 6,534 | 1,089 / 144 / 36 | 14 | 1.4 |
| free 270° cylinder 24×8 (sweep-built), $R/t = 50$ | 1,350 | 243 / 27 | 46 | 0.8 |
| T-junction $24^2$, $L/t = 50$ (clamped edges) | 5,700 | 864 / 108 / 9 | 13 | 1.5 |
| L-fold $24^2$, $L/t = 50$ (clamped edges) | 5,700 | 864 / 108 / 18 | 11 | 1.3 |
| T-junction $24^2$, 3-layer skin and web, $L/t = 50$ (clamped edges) | 12,000 | 864 / 108 / 9 | 18 | 1.7 |
| flat $32^2$, $L/t = 300$ | 6,534 | 1,089 / 144 / 36 | 17 | 1.4 |
| $32^2$ plate with a hex8 rib, $L/t = 50$ | 7,128 | 1,188 / 153 / 36 | 19 | 1.6 |
| hex8 $16^3$ block with an `ssh8` skin ($L/t$ = 100), base clamped | 15,606 | 1,944 / 72 | 14 | 1.3 |
| clamped $32^2$, $L/t = 50$ | 6,534 | 1,089 / 144 / 36 | 12 | 1.3 |
| clamped $32^2$, σ = 0 (`nonsingular=True`) | 6,534 | 1,089 / 144 / 36 | 12 | 1.3 |
| bottom edges supported $32^2$ | 6,534 | 1,089 / 144 / 36 | 12 | 1.3 |
| $32^2$ and $4^2$ plates, separate (`max_coarse` 40) | 6,684 | 1,125 / 153 / 45 / 18 | 14 | 1.6 |
| free $24^2$ at σ = $\lambda_7$ | 3,750 | 729 / 81 | 14 | 0.9 |
| 3-layer stack $32^2$, $L/t = 10$ | 13,068 | 1,089 / 144 / 36 | 11 | 1.5 |
| 3-layer stack, $L/t = 100$ | 13,068 | 1,089 / 144 / 36 | 16 | 1.5 |
| 3-layer stack, $L/t = 300$ (reported) | 13,068 | 1,089 / 144 / 36 | 40 | 1.5 |
| 3-layer stack, $L/t = 1000$ (reported) | 13,068 | 1,089 / 144 / 36 | 145 | 1.5 |

- **Roadmap gate:** on the flat $128^2$ plate the count grows 12 → 21 (1.75×) from $L/t$ = 10 to 1000. The counts follow the element flatness $h/t$ rather than $L/t$: at $L/t = 1000$ they fall with refinement, 21 / 19 at $128^2$ / $256^2$ ($h/t$ = 7.8 / 3.9), so the ratio depends on the mesh.
- **Limitations:** curved panels at $L/t = 1000$ (11 / 30 on $128^2$ from $L/t$ = 10 to 1000, 2.7×; `degree=3` lowers the counts to 8 / 23 but not the ratio), stacks at $L/t \ge 300$ (40 / 145 at 300 / 1000), and deep thin curved shells (the free 270° cylinder of radius 1: 16 / 46 / 135 at $R/t$ = 10 / 50 / 200, 46 / 44 / 32 for 24×8 / 48×16 / 96×32 elements at $R/t = 50$): the flat curvature fields of an aggregate spanning a large arc miss the inextensional modes. Jacobi on the $32^2$ plate at $L/t = 50$ needs 1289 iterations (JSON, `jacobi` of `flat32-50`). The degree-3 and cylinder series are review measurements, not in the JSON. Junctions at $L/t = 1000$ were not re-measured (prototype: about 80 against 39 on a flat plate). Folds: an aggregate across a fold gets an averaged normal; rigid modes stay exact, only the curvature columns degrade (L-fold $24^2$ at $L/t = 50$: 11).

**Cost** (same JSON, $m = 15$ columns, B per fine DOF, distinct device arrays; build includes compilation). Flat plate $256^2$ at $L/t = 1000$ (396,294 DOF, stored-Ke operator 480 B per DOF, $t_A$ = 49.8 ms):

| component | B per DOF |
|---|---|
| level-0 Schwarz ($s = 24$, packed, with `idx`) | 413 |
| $P_0$ and $P_0^\top$ ($b = 3$) | 529 |
| level-1 Schwarz / $A_1$ / $P_1$, $P_1^\top$ | 502 / 148 / 183 |
| level-0 $A_P$ leaves (`free`) | 8 |
| level 2 and the coarsest factor | 151 |
| **total** (without the operator) | **1934** |

The V-cycle costs 8.9 $t_A$ and the setup (`amg()` wall time, compilation included) 5.5 s; the preconditioner stores about 4× the operator (about 2 GB at 1M DOF, 20 GB at 10M in fp64). On the hex-dominant $32^3$ block with an `ssh8` skin (111,078 DOF, σ of the $16^3$ block) the shell path stores 1169 B per DOF, its V-cycle costs 5.1 $t_A$ and PCG takes 13 iterations; PR 13's hex8 AMG on the same block without the skin stores 1096 B per DOF, 5.9 $t_A$, 10 iterations.

### Memory: `amg_memory`

`amg_memory(mesh, m, max_coarse=1000, degree=2)` (PR 13b; `max_coarse` and `degree` PR 13c) returns, in bytes, for a hex8/hex20/hex27 mesh with $n_c$ corner DOF ($3\times$ corner nodes) and $n$ DOF:
- **`storage`** (device, resident): the p-level (non-hex8 only: $120\,n_\text{nodes}$ for the smoother blocks and owner data plus $12\,(w\,n_\text{nodes} + c_\max n_\text{corner})$ for the transfer, exact) plus `AMG_STORAGE[fam]` $\cdot\, n_c$ for the AMG levels and transfers ($1200$ B for all-hex8, $2450$ B otherwise) plus $8\min(n_c, \max(\texttt{max\_coarse}, 1000))^2$ for the dense coarsest factor. The floor at the calibration value keeps the bound: below 1000 the last large level becomes a stored ELL level with its own smoother and transfers, which the per-corner constant (calibrated at 1000) excludes, while the owned bytes stay at or below those at 1000 (hex20 $12^3$: 18.8 against 21.0 MB). Measured: ≥ 1.14× the owned bytes at `max_coarse` 100–3000 on hex20 $8^3$, $12^3$ and hex8 $16^3$ (without the floor 0.94× at 100); loose above 1000 (4.3× at 3000 on hex20 $12^3$).
- **`build`** and **`host_build`** (host, transient; the same value, `host_build` naming the host space for the per-space fit test): $\beta + \alpha_t n_c$ with $\beta$ = 45 MB and $\alpha$ = 4.6 / 5.5 / 6.0 kB (hex8 / hex20 / hex27). The measured peaks beyond $\beta$ grow with size, about 0.65 kB per corner DOF per decade of $n_c$ (hex20 3.7 / 4.3 / 4.8 kB at $12^3$ / $24^3$ / $44^3$; setup table), so the constants extrapolate that trend to 10M DOF (hex27, measured at two sizes only, sits above the hex20 slope): 1.14–1.24× the measured peaks up to 1M except hex8 $24^3$ (1.36×, where the per-corner peak is smallest, 3.1 kB), and above 1M an extrapolation ([`open_items.md`](open_items.md)). `_sa_levels` compacts the first-level matrix in place, so the assembled matrix exists once on the host. It also bounds the kept aggregates plus the count-time rebuild with the aggregates held (PR 15d: measured within it at hex20 $16^3$, $32^3$ and hex8 $32^3$, `test_amg_reuse_peak`).
- **`step`**, **`step_low`**: the V-cycle's extra temporaries in a LOBPCG step, $8n(a\,m + c) + 8mK$ (`AMG_STEP`, [`eigensolvers.md`](eigensolvers.md#memory-model)). `degree` > 2 adds `AMG_LOOP_K` = 120k to $K$: the Chebyshev loop body keeps its own n-independent buffers (block-ELL scan gathers, about 1 MB per column), +77k–103k measured at degrees 3–6 (the same for every degree ≥ 3) on hex8/hex20/hex27 $6^3$–$16^3$ and pinned within 0.6–1× of the model at hex20 $8^3$ by `test_amg_loop_increment`; without it the low-memory step at hex20 $8^3$, degree 4, was 1.05× the estimate. `AMG_STEP` is calibrated at `max_coarse` 1000; above it the step term is not a bound (+1.3 blocks of temporaries at 4000).

The constants come from box meshes only (hex20 $8^3$–$44^3$, hex27 $6^3$–$24^3$, hex8 $12^3$–$68^3$); the AMG part per corner DOF varies with the hierarchy (hex20: 1463–2317 B, hex8: 874–1150 B), so `AMG_STORAGE` is its maximum. Graded and curved meshes, and $n_c$ below $10^4$, are not calibrated ([`open_items.md`](open_items.md)). `solve_modes_memory(..., precond="amg")` composes `amg_memory` with `lobpcg_memory`; the build peak is not summed into `total` (it enters through $\max$(operator build, preconditioner build, step)), because the first build runs before LOBPCG allocates; it counts in `host_total` next to the host V (PR 15d), because the count-time rebuild with aggregates held runs while V is alive. The mesh scan behind the estimate (`_amg_scan`: family, corner DOF, tables) is cached per mesh (`lru_cache(maxsize=1)`; 2.3 s at 10.2M hex20), so the band searches' many calls cost it once; with the gather path the operator estimate still runs `layout` per call (3.2 s at 10.2M; [`operators.md`](operators.md), PR 18c).

**Limitations** ([`open_items.md`](open_items.md)).
- **Flat elements:** AMG on the hex8 coarse space makes `pmg`'s flat-element weakness worse (table). The loss comes from the deeper AMG levels, whose six rigid modes per aggregate represent plate bending poorly: on the plate with $a = 1$ (study runs `amg max_coarse=…`), `max_coarse` = 2000 (no AMG level) gives 15, 1000 (one AMG level) 19 and 100 (two) 36. Large meshes always have deep hierarchies, so the `max_coarse` = 100 numbers are the representative ones. Remedy: solid-shell elements (M4) and their AMG ([below](#amg-on-solid-shells-pr-18a)).
- **10M scale:** `amg_memory` puts the host build peak at 14.6 GB for 10M hex20 DOF and 46.6 GB for 10M hex8 DOF (extrapolated; all-hex8: the fine level is the assembled $A_P$ itself); the device storage at 7.1 / 12.6 GB. The 10M configurations are deferred GPU checks.
- **SIMP fields at their own $\lambda_1$:** 790–910 iterations against GMG's 87 (table above); θ = 0 and the reuse equality are kept, the remedy is logged in [`open_items.md`](open_items.md).

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

- `tests/test_krylov.py`: PCG with the identity, Jacobi, block Jacobi and Chebyshev, σ = $\lambda_1$, $100\lambda_1$ and free-free, against `spsolve` to $10^{-10}$ with a zero column; columns independent and frozen; the PCG exits (floor after exactly 5 restarts, convergence after a restart, `maxit`, a residual turning NaN mid-run); MINRES on SPD and indefinite systems to $10^{-10}$, its exits and the best iterate returned after a stagnation with σ on an eigenvalue; an initial 2-norm plateau above $\|b\|$ longer than `STALL_C` checks (diagonal indefinite A, a point in a spectral gap carrying most of $\|b\|$, its A entry scaled by $10^{12}$ and its preconditioner entry by $10^{-12}$, so it has almost no weight in MINRES's norm) converges without a stagnation exit, and a column on A's null space runs to `maxit`, neither converged nor stagnated (PR 15c); no recompiles for new data.
- `tests/test_ell.py`: plain and sliced ELL against CSR to $10^{-12}$ ($b$ = 3, 6; $m$ = 1, 7; several chunks with a remainder), padding against an independent count, the automatic choice, fp32 to $10^{-6}$, PCG through `BlockELL`; rectangular matrices (tall and wide, $b$ = 1, 3, plain and sliced) with padding entries in column 0; the width-bounded chunk (a width-512 bucket scans 108 rows, its gather temporaries under a quarter of the whole-bucket scan's) and a width-27 $b = 3$ square matrix keeping 2048.
- `tests/test_precond.py`: `ap_operator` against `a_p` on unmasked vectors (σ = 0, positive, negative); node blocks and block-Jacobi inverses against SciPy for stored/on-the-fly, consistent/lumped, nodal masses, `s_K`, mixed hex8/hex20, hex27 and a masked two-phase voxel grid with `s_K`, all with partially constrained nodes; the σ < 0 and NaN raises; $1.1\,\tilde\lambda_\max/\lambda_\max \in [1, 1.5]$ and the invariant-start guard; Chebyshev against a dense spectral reference to $10^{-12}$ (degrees 1–4), symmetry, $A_P$-self-adjoint error propagation, positivity, and fewer PCG iterations than block Jacobi; the h-study smoke run.
- `tests/test_gmg.py`: every coarse level equals $P^\top A_P P$ from SciPy to $10^{-12}$ on masked meshes with nodal masses, two phases, odd axes (level-2 weights ⅓, ⅔), stopped axes, both stored axis orders, consistent and lumped mass, at the default and a 7-cell setup chunk; transfers reproduce linear fields on every level, restriction is $P^\top$, kept columns nonzero and dropped ones zero; the smoother blocks are the dense diagonal group blocks and the group apply their inverse; the V-cycle equals a dense NumPy composition to $10^{-10}$; symmetry and definiteness; the iteration bounds (h-independence ≤ 20%, SIMP, plate, free, beam, odd); the singular guard's raises and builds; island detection and the deflated cycle against a dense $Q + (I - QA)V(I - AQ)$ with an independently built $Z$; padded deflation sizes; the default degrees; `grid_ids` on slab-padded stencils; LOBPCG with GMG against `reference_modes`; no recompiles; the input errors; the study smoke run. PR 11b: the fp32 hierarchy's leaves (fp32 except the fine operator, `free`, `coarse.L`, `defl`; `levels[l][0] is levels[l][1].A`), the fp32 cycle within $10^{-5}$ of fp64 ($3\cdot10^{-4}$ on the thin plate), symmetric and positive with ≤ 0.75× the temporaries, and LOBPCG with ≤ 1.2× the iterations and λ within $10^{-9}$ (box $16^3$; slow: box $32^3$, plate $32^2\times1$, odd $33\times8\times8$); `gmg_memory`'s storage ≥ the unique buffers of the built hierarchy (fp64, fp32; boxes, a masked odd grid, a cropped grid with $k = 3$ fine groups, the island case with `s_K`) and within 5% on full boxes; the V-cycle temporaries ≤ `workspace` ($16^3$; slow $32^3$, within 1.25×) and the LOBPCG steps with GMG ≤ `lobpcg_memory`'s step term; slow: at $48^3$ the GMG steps' extra temporaries over the Jacobi steps lie within 0.6 blocks below `GMG_STEP`; every σ = 0 refusal is a `SingularShiftError`.
- `tests/test_pmg.py` (PR 13b): the production assembler returns sorted int32 BSR whose padded positions point past the pattern, equals PR 12's reference values (hex20 chunked, hex27, mixed with shared nodes; hex8 against the assembled $a_p$) and, with a reused pattern at a new σ, a fresh build to $10^{-15}$.
- `tests/test_pmg.py`: the transfer reproduces trilinear fields on graded hex20 and hex27 boxes to $10^{-12}$, restriction and the dense prolongation equal an independent element-loop $P$ (also on the mixed hex8 + hex20 mesh, whose hex8 nodes get identity rows), with hand-checked ½ / ¼ / ⅛ rows; `_hex8_galerkin` equals $P^\top a_p(K + \sigma M, free, s) P$ from SciPy to $10^{-12}$ and is exactly symmetric (hex20, hex27, mixed; stored and otf; consistent and lumped; two phases, random `s_K`/`s_M`, nodal masses, a mixed clamped/sliding mask), also with padded chunks (whose `element_matrices` are zero) and on all-hex8 meshes (the assembled $a_p$); the V-cycle at degree 3, ratio 20 equals a NumPy composition with an independently built smoother and $(P^\top A P)^{-1}$ to $10^{-10}$ and is SPD on the free DOFs; symmetry on unmasked vectors, masked outputs, `type(pmg(...)) is GMG`; PCG in ≤ 2 iterations on a pure hex8 box; h-independence (≤ 20% growth from $3^3$ to $12^3$, hex20 and hex27; slow: hex20 $24^3$); ≤ 30 iterations on the beam, free, curved, nodal-mass and `s_K`-checkerboard cases; LOBPCG with p-MG against `reference_modes` in ≤ 40 iterations (hex20, hex27); the input errors; no recompile for a reused build; the factor freed after `jax.clear_caches()`; the study smoke run.
- `tests/test_amg.py`: every SA level's coarse matrix equals $P^\top A_z P$ from SciPy to $10^{-12}$ and is symmetric, $P$ has no zero column, and the device levels, transfers ($P$, $P^\top$) and coarsest factor equal the SciPy matrices (hex20 $6^3$, clamped plus sliding); two setups are bitwise identical and leave `np.random` unchanged; $B$ spans the corner restriction of `rigid_modes` with exactly zero constrained rows; the first-level $P$ equals pyamg's `smoothed_aggregation_solver` to $10^{-12}$ (hex20, hex8 $12^3$); a random 30%-free mask drops zero columns and keeps every coarse matrix SPD (hex20, hex8); aggregates straddle the Dirichlet boundary and the setup succeeds; the V-cycle at degree 3, ratio 20 equals a NumPy composition with test-built smoothers and the dense coarsest inverse to $10^{-10}$, symmetric, SPD on the free DOFs, masked (hex20 with two AMG levels; hex8 with a sliding mask, whose constrained rows of $P_0$ are exactly 0); the no-AMG-level paths (hex20 equal to `pmg`, hex8 a direct solve); h-independence (≤ 20% growth and ≤ 1.5× the measured count: hex20 $3^3$–$12^3$, slow $24^3$; hex27 $3^3$–$12^3$; hex8 $6^3$–$24^3$); ≤ 1.5× the measured count on the beam, free, curved, nodal-mass and checkerboard cases and, as a regression bound, the single-layer plate (36); LOBPCG with AMG against `reference_modes` in ≤ 40 iterations (hex20, hex8); the input errors, the coarsest cap (monkeypatched singleton aggregation), the non-finite level and coarsest raises, a degenerate `max_coarse`, array leaves, a matrix-free fine level; the study smoke run. PR 13b: aggregate reuse equals a fresh setup to $10^{-12}$ (and ≤ 1.2× its iterations) after a 20% density perturbation and at σ > 0 from σ = 0 aggregates (hex20, hex8), stays within 1.2× the iterations after a 2% node perturbation, calls none of the pattern, tables, strength, aggregation or fit (monkeypatched), and raises for another connectivity or `free`; σ = 0 raises `SingularShiftError` without `nonsingular=True` and with it gives a finite V-cycle under which PCG converges (hex20, hex8), and raises for a zero `s_K` element; a second build leaves no SciPy matrix and no device array beyond the pytree's leaves; `amg_memory`'s storage ≥ the owned device bytes of the built `amg` (hex20, hex27, hex8, sliding, distorted), with its n-scaling part alone ≥ the non-coarse bytes and within 1.3× at hex20 $12^3$ and hex8 $16^3$; a mixed mesh's build term uses the largest per-type constant; the LOBPCG step with AMG within `AMG_STEP` (slow: within 1 block below at hex20 $24^3$ and hex8 $48^3$); slow: the build peaks below `amg_memory`'s `build` and within 1.3×, the benchmark entries' smoke runs.
- `tests/test_shell_amg.py` (PR 18a): fibers against a union-find (single layer, 3-layer stack, junctions, hex-only nodes) and the size guard (a wrong axis on a $40^2$ plate, 31 layers pass and 32 raise, the 5/5-layer T builds) and the `ssh8` + `hex20`/`hex27` raise; the candidates on flat plates and stacks against the Kirchhoff fields in an independent frame (level 0 in span angle against the Kirchhoff fields, level 2 as the residual of $B^{(2)}_\text{fine}$ projected through $T_0T_1$, both to $10^{-10}$), sign-free on a sweep-built cylinder, zero curvature columns on hex aggregates; the Schwarz apply against the dense sum over independently built patches of the assembled $A_P$ (web, stack, T, clamped, a 5/5-layer T with $s = 252$, a strip whose collinear fibers give identity DOFs; level 1 against SciPy's $A_1$ and NumPy rings), at small `chunk_bytes` (several chunks, pad slots), the class rule (one class per distinct size, the 1.1× rule, the cap of 8), block keys past int32 ($n_b$ = 50 000 against dense inverses), `nbytes`, the coarse bound and the non-finite raise; group graphs, aggregation and coarse patch sets against NumPy; the setup against dense references (P, ρ, $A_{\ell+1}$, identity rows with $(D^{-1}A)e = e$, the V-cycle at degree 3, ratio 20 to $10^{-10}$, symmetric, SPD, masked, array leaves), $1.1\tilde\lambda_\max \ge \lambda_\max$ of the Schwarz-preconditioned levels, the one-level M4 plate as the dense inverse; PCG counts gated at 1.5× the measured ones (tests 6–11: the slow $L/t$ growth on $128^2$ with ratio ≤ 2 and the curved panel, refinement $64^2 \to 128^2$ at $L/t = 100$ and $128^2 \to 256^2$ at 1000 within 1.3×, plies, curved, cylinder, junctions, $L/t = 300$, the hex8-web plate, the hex-dominant block, clamped, bottom-supported, two separate parts with a singleton aggregate, a free plate at $\lambda_7$, σ = 0 with `nonsingular=True`, stacks), Jacobi and the PR 13 hex8 count reported; aggregate reuse equal to a fresh setup to $10^{-12}$ with fibers, aggregation and fits skipped (spies) and the raises; hex meshes never reach the shell code. `tests/test_shell_ops.py::test_refusals`: `pmg` refuses `ssh8`, `amg` refuses `ssh8` + `hex20`, and `solve_modes(precond="amg")` / `amg_memory` refuse `ssh8` until PR 18b.
