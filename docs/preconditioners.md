# Linear solvers, preconditioners and block-ELL

PR 10 adds the single-level kernels that the multigrid PRs (11–13), the eigenvalue counting (PR 14) and the interior bands (PR 15) build on; PR 11a adds the voxel geometric multigrid ([below](#voxel-geometric-multigrid-gmg)). Everything works on the preconditioning operator $A_P$ ([`boundary_conditions.md`](boundary_conditions.md)).

| module | contents |
|---|---|
| `jaxmodal.linalg` (`linalg/krylov.py`) | `pcg`, `minres` |
| `jaxmodal.linalg` (`linalg/ell.py`) | `block_ell`, `BlockELL` |
| `jaxmodal.precond` (`precond/jacobi.py`) | `jacobi` (moved from `eigen/lobpcg.py`; `jaxmodal.eigen.jacobi` is the same function), `block_jacobi` |
| `jaxmodal.precond` (`precond/chebyshev.py`) | `lmax_estimate`, `chebyshev`, `Chebyshev` |
| `jaxmodal.precond` (`precond/gmg.py`) | `gmg`, `GMG` (voxel geometric multigrid) |
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
- **Memory** (measured, plus `b`): 9 blocks with a diagonal operator (the carried $x$, $r_1$, $r_2$, $y$, $w$, $w_2$, the body's $v$ and $Av$, the returned $x$); with the hex20 operator about 12 blocks above a bare apply (6 more than PCG).

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

## Voxel geometric multigrid (`gmg`)

```python
from jaxmodal.precond import gmg

P = gmg(op, free, sigma, degree=2, coarse_degree=4, ratio=30.0)  # op: a voxel Stencil; sigma >= 0
x, info = pcg(ap_operator(op, free, sigma), b, P, tol=1e-8)  # or minres / lobpcg(precond=P)
```

`gmg` returns a pytree `GMG` whose call applies one V-cycle to an `(n_dof, k)` block, approximating $A_P^{-1}$ with $A_P = m(K + \sigma M)m + s(I - m)$ and $s$ = `ap_operator(op, free, sigma).s`. It applies $m \odot V(m \odot r)$: masking the input as well keeps it symmetric on the whole space. Transfers, level matrices, smoothers and the coarsest factor are leaves and the shapes depend only on the grid, so a new design on the same grid (new `s_K`, `s_M`, σ) never recompiles `pcg`, the V-cycle or the LOBPCG step, while the island deflation keeps its padded row bucket (see Island deflation). The fine operator is never assembled.

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

On the fine level the child matrices are formed per chunk from the phase-indexed $K_\phi$, so no $(n_\text{el}, 24, 24)$ array exists. All arrays enter the loop as arguments (closed-over arrays would become constants). Each level $\ge 1$ is a `BlockELL` (PR 10's fields, one bucket, so it shares the apply): 27 slots per node, slot = corner offset difference, missing neighbours pointing at the node itself with zero blocks; built on the device, not through the host `block_ell` (its host COO would be ~0.8 GB at 2M DOF, ~19 GB at 100M). The coarsest level is summed into a dense fp64 matrix and Cholesky-factored.

**Transfers.** Prolongation scatters the coarse values into the full coarse grid (`unique_indices=True`), interpolates axis by axis with two-point gathers, and gathers the active fine nodes; restriction is its exact transpose (`jax.linear_transpose`).

**Smoothers: stopped-axis group blocks.** Point smoothing cannot damp the error that semi-coarsening leaves along stopped axes (plate $64^2\times1$ with node-block Jacobi: 240 iterations, prototype). A group is the set of nodes sharing coordinates on the coarsening axes; it has $k = \prod_{d\,\text{stopped}}(n_d + 1) \le 27$ nodes. The group blocks $D_l$ are the $3k\times3k$ diagonal blocks of the level operator (with $k = 1$: node blocks), inverted by PR 10's `_inverse_blocks` (absent slots of partial groups set to the identity). For $k > 1$ the apply is gather-only: gather, batched GEMM, then a precomputed inverse permutation (groups partition the nodes; 34 ms against 113 ms for a scatter-add at 2M DOF). Each level smooths with `chebyshev(A_l, D_l, lmax_estimate(A_l, D_l), degree_l, ratio)`, `degree` = 2 on the fine level and `coarse_degree` = 4 on the coarser ones: they cost about 1/8 of the fine level, and the extra steps cut iterations on the hard cases (SIMP field at its own $\lambda_1$ 116 → 87, plate $128^2\times1$ 186 → 136, spacing ratio 10 170 → 105) for about +0.5 $t_A$ per cycle at $87^3$ (4.7 → 5.2 $t_A$).

**V-cycle.**
$$x = S_l b,\qquad x \mathrel{+}= P_l\,V_{l+1}\big(P_l^\top(b - A_l x)\big),\qquad x \mathrel{+}= S_l(b - A_l x),$$
with the exact Cholesky solve on the coarsest level. Pre- and post-smoothing use the same symmetric $S_l$, so the V-cycle is symmetric. Chebyshev runs on $[1.1\tilde\lambda/\text{ratio}, 1.1\tilde\lambda]$ with $\tilde\lambda$ the 20-step Lanczos estimate; the V-cycle is positive definite provided $1.1\tilde\lambda \ge \lambda_\max(D_l^{-1}A_l)$ (then $\|I - S_lA_l\|_{A_l} < 1$), PR 10's safety assumption (measured $1.09$–$1.10\,\lambda_\max$ on every level of a binary SIMP box, a plate and a beam). When the fine grid already has every axis $\le 2$, it is smoothed once and then solved exactly. The recursion is unrolled at trace time.

**Island deflation.** A stiff part joined to the rest only through soft cells (a SIMP island) moves almost rigidly at tiny energy, and no geometric coarse space separates that motion from the body around it: the checked-in SIMP field has two islands, its $\lambda_1$ is one of their modes, and at that σ PCG needs 723 iterations without deflation. When $\max s_K \ge 10^3\min s_K$ (`ISLAND_CONTRAST`), `gmg` labels the node-connected components (cells sharing a face, edge or vertex) of the cells with $s_K > 10\min s_K$ and takes all but the largest, at most `MAX_ISLANDS` = 16, largest first. The threshold keeps each island's filtered gray halo, which matters: counting only cells above $10^{-3}\max s_K$ leaves 432 iterations. $Z$ holds each island's six rigid modes (translations, rotations about its centroid, constrained rows zero) and the call applies the balancing form
$$P = Q + (I - QA_P)\,mVm\,(I - A_PQ),\qquad Q = Z E^{+} Z^\top,\quad E = Z^\top A_P Z,$$
symmetric, and positive definite whenever $mVm$ is. $Z$ and $A_PZ$ are block-diagonal over the islands, so they are stored per support row (island nodes plus one ring of cells): the row's DOF, its island id and its 6 entries of $Z$ and of $A_PZ$, about 100 B per row (a row on two islands' rings appears twice). $Z^\top r$ is six segment sums over the island ids and $Zy$ a per-row gather of the island's coefficients, so an apply adds no $A_P$ apply; $E^{+}$ is a Jacobi-scaled eigen-pseudo-inverse over a fixed $6\cdot$`MAX_ISLANDS` = 96 columns (absent islands, dependent or fully constrained columns drop). The row count is padded to a power of 2 (padding rows point at DOF 0 with zero values), so a new design recompiles only when the row bucket changes or islands appear or vanish. Without islands `defl` is `None` and $P = mVm$. The setup applies $A_P$ once per island (6 columns), about 0.2 s per island at 2M DOF.

**Singular guard.** With σ = 0 and a singular $A_P$, the rounded coarsest Cholesky gives NaN or a finite factor that amplifies rigid components by ~$10^{13}$. At σ == 0, `gmg`:
1. refuses any cell with `s_K` = 0 (`ValueError`: "zero-stiffness cells at sigma = 0: ... use sigma > 0"): such cells add no stiffness and can cut the body apart;
2. calls `rigid_modes(voxel_mesh(shape, spacing, mask), free)` before any device setup and raises if it returns a column; rigid modes are linear, so they lie in the range of every $P$, and the coarsest level is singular exactly when $A_P$ is;
3. re-raises, chained, the `ValueError` of meshes `rigid_modes` rejects (edge or vertex contacts, more than `MAX_PARTS` parts): "cannot verify that A_P is nonsingular at sigma = 0 (...); use sigma > 0". This refuses a few SPD cases (a hinge clamped on both sides, many clamped islands); σ > 0 makes all of them SPD.

Two backstops act at any σ: non-finite smoother blocks raise "smoother blocks are not finite" (e.g. cells with `s_K` = `s_M` = 0 leaving free nodes with no stiffness and no mass), and a non-finite coarsest factor raises "coarsest factor is not finite". A pivot-ratio threshold was rejected: legitimate SPD cases fall with size (clamped plate at σ = 0: $6\cdot10^{-7}$ at $128^2\times1$) and reach $1.6\cdot10^{-11}$ on SIMP designs with large void regions at σ > 0. The guard is exact only at σ == 0: a σ far below $\lambda_1$ passes it and gives a poorly conditioned V-cycle, so choose σ physically (the σ pre-estimate).

**Inputs.** `op` must be a `Stencil` (`TypeError` otherwise) in the `voxel_mesh` node order: a voxel mesh renumbered by `rcm`/`renumber` raises `ValueError` (its grid order is already narrow). A concrete σ < 0 raises; x64 must be on.

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

At $87^3$ (2.04M DOF, $\lambda_1$ of $8^3$): the build adds 1.12 GB of process peak over the operator + block Jacobi + $\lambda_\max$ baseline (the unchunked prototype added 22.7 GB), and one V-cycle at $m = 15$ takes 1.92 s against 0.370 s for one $A_P$ apply ($5.2\,t_A$, warm; the Mac budget model assumes $5.5\,t_A$). The setup transient is about 0.9 KB per fine node (level-1 cells, ELL values, level-2 cells) and the resident hierarchy about 450 B per fine node (150 B per DOF; 0.31 GB at $87^3$, ~15 GB at 100M DOF); at 100M DOF that is roughly 30 GB of transient, which fits a GPU only because `gmg` runs before LOBPCG allocates its blocks (as the rigid-mode setup does). The transfers work on the full box grid, so their temporaries scale with the box, not the active nodes (≈ 1/occupancy). Restriction is the transpose of the gathers, i.e. scatter-adds: 0.156 s against 0.026 s for prolongation on the fine level at $87^3$, $m = 15$ (a deferred GPU check). The prototype's LOBPCG (10 modes, tol $10^{-6}$) took 17 iterations at $32^3$ and 18 at $87^3$.

**Limitations** (tracked in [`docs/open_items.md`](open_items.md)).
- **Anisotropic spacing:** coarsening is by element count, not spacing; $h_z = h/10$ needs 106 iterations. Spacing-aware semi-coarsening needs line smoothers along long stopped axes.
- **Thin plates and slender beams:** at σ = $\lambda_1$ the iterations grow like $h^{-1}$ (single-layer plates 23 / 37 / 71 / 136 from $16^2$ to $128^2$). Coarse trilinear hex8 cells on a one- or two-element-thick layer get ever flatter and lock in bending, so the coarse correction misses the smooth bending error. Larger σ hides it (the same plates at the overestimated `sigma_estimate` took ≤ 60). Remedies belong to later PRs: algebraic coarsening (PR 13) and the solid-shell element (M4).
- **High contrast at low σ:** at its own $\lambda_1$ a SIMP field's lowest modes are spurious island and void-region modes of $s_K = \rho^3$, $s_M = \rho$ ($\lambda_1$ is $10^{-6}$–$10^{-3}$ of the solid's). Deflation removes the island modes, but soft-region error that trilinear coarse functions cannot separate from the stiff material remains: 87 on the checked-in field, 109–135 on other seeds, 452 on an unfiltered binary field (every case converges). A SciPy prototype with Jacobi-smoothed, phase-aware prolongation plus deflation reached 34 (checked-in field) and 55 (seed 4); such operator-dependent transfers widen the coarse stencils beyond 27 points and belong to PR 13 (algebraic multigrid). Topology optimization removes the spurious modes with a modified mass interpolation (PRs 21–23), and PR 23 re-runs the contrast test on a real design.

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
- `tests/test_gmg.py`: every coarse level equals $P^\top A_P P$ from SciPy to $10^{-12}$ on masked meshes with nodal masses, two phases, odd axes (level-2 weights ⅓, ⅔), stopped axes, both stored axis orders, consistent and lumped mass, at the default and a 7-cell setup chunk; transfers reproduce linear fields on every level, restriction is $P^\top$, kept columns nonzero and dropped ones zero; the smoother blocks are the dense diagonal group blocks and the group apply their inverse; the V-cycle equals a dense NumPy composition to $10^{-10}$; symmetry and definiteness; the iteration bounds (h-independence ≤ 20%, SIMP, plate, free, beam, odd); the singular guard's raises and builds; island detection and the deflated cycle against a dense $Q + (I - QA)V(I - AQ)$ with an independently built $Z$; padded deflation sizes; the default degrees; `grid_ids` on slab-padded stencils; LOBPCG with GMG against `reference_modes`; no recompiles; the input errors; the study smoke run.
