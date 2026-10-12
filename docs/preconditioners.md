# Linear solvers, preconditioners and block-ELL

PR 10 adds the single-level kernels that the multigrid PRs (11–13), the eigenvalue counting (PR 14) and the interior bands (PR 15) build on; PR 11a adds the voxel geometric multigrid ([below](#voxel-geometric-multigrid-gmg)) and PR 12 the unstructured p-multigrid ([below](#unstructured-p-multigrid-pmg)), both on one shared V-cycle. Everything works on the preconditioning operator $A_P$ ([`boundary_conditions.md`](boundary_conditions.md)).

| module | contents |
|---|---|
| `jaxmodal.linalg` (`linalg/krylov.py`) | `pcg`, `minres` |
| `jaxmodal.linalg` (`linalg/ell.py`) | `block_ell`, `BlockELL` |
| `jaxmodal.precond` (`precond/jacobi.py`) | `jacobi` (moved from `eigen/lobpcg.py`; `jaxmodal.eigen.jacobi` is the same function), `block_jacobi` |
| `jaxmodal.precond` (`precond/chebyshev.py`) | `lmax_estimate`, `chebyshev`, `Chebyshev` |
| `jaxmodal.precond` (`precond/gmg.py`) | `gmg`, `GMG` (voxel geometric multigrid; the shared V-cycle) |
| `jaxmodal.precond` (`precond/pmg.py`) | `pmg` (unstructured p-multigrid, returns a `GMG`) |
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
- Memory: V-cycle temporaries 6.1 instead of 12.1 fp64 blocks at $87^3$ (0.50× at $16^3$–$64^3$), hierarchy storage 0.56×.

**Memory estimator.** `jaxmodal.memory.gmg_memory(mesh, m, dtype="float64", s_K=None)` returns, in bytes, with the operator estimators' keys and composition rule:
- **`storage`**: an upper bound from shape arithmetic on the grid `gmg` builds from (the voxel grid cropped of empty trailing layers along the scan axis, as `grid_ids` does): the fine level exactly (smoother blocks, $9k$ entries per node in `dtype` for the group size $k$, group index arrays or the $k = 1$ mask, `free`); every coarse level as its full bounding box (ELL values $27\cdot9$ entries per padded row in `dtype`, int32 columns, smoother blocks); the transfers (int32 node maps, per-axis int64 indices and `dtype` weights); the coarsest factor $(3n_c)^2\cdot8$. With `s_K` it adds the island deflation (104 B per padded support row, plus $96^2\cdot8$). Measured on boxes and a plate: within 0.02% above the unique buffers of the built `GMG` (the operator's own leaves excluded).
- **`workspace`**: the V-cycle temporaries for PCG users, $c\,b$ plus the stencil's slab workspace in `dtype` ($b = 8\,n_\mathrm{dof}\,m$; $c$ = 15 fp64 and 7.5 fp32, the envelope of `memory_analysis` at $16^3$–$87^3$, which ranges 10.6–14.9 and 5.0–7.5); with islands $+3\,n_\mathrm{rows}$ rows of $8m$ B. 3% above the measured temporaries at $32^3$, 36% above at $87^3$.
- **`build`**: the setup peak, 1650 B per fine node (1.12 GB at $87^3$, process peak above the baseline, resident hierarchy included); for fp32 the same (the fp64 hierarchy and its fp32 copy are briefly alive together, but the copy, < 300 B per fine node, fits under the build's transient).
- **`step`**, **`step_low`**: the extra LOBPCG step bytes with GMG over Jacobi, `GMG_STEP` blocks (`lobpcg_memory`, [`eigensolvers.md`](eigensolvers.md#memory-model)).

**Acceptance** (PR 11b, `gmg_study.acceptance` → `gmg_study.json` "acceptance", `tests/test_api.py`): `solve_modes` on the $87^3$ steel box (2.04M DOF, `xmin` clamped), 10 modes, tol $10^{-6}$, defaults (GMG, pre-estimate, 9-block step, fp64 basis). Wall time of one cold subprocess (compilation included); peaks of two warm subprocesses against `solve_modes_memory`:

| V-cycle | wall time | σ pre-estimate + build | LOBPCG | iterations | warm peaks | estimate |
|---|---|---|---|---|---|---|
| fp64 | 60.9 s | 16.2 s (5 its) | 43.4 s | 16 | 6.23, 6.21 GB | 6.83 GB (+10%) |
| fp32 | 51.0 s | 16.7 s (5 its) | 32.9 s | 16 | 4.86, 4.79 GB | 5.44 GB (+12%) |

Both are far inside the 10-min budget and the ±20% band; the cold runs peak higher (6.65 / 5.25 GB, compilation) and are still below the estimates.

**Limitations** (tracked in [`docs/open_items.md`](open_items.md)).
- **Anisotropic spacing:** coarsening is by element count, not spacing; $h_z = h/10$ needs 106 iterations. Spacing-aware semi-coarsening needs line smoothers along long stopped axes.
- **Thin plates and slender beams:** at σ = $\lambda_1$ the iterations grow like $h^{-1}$ (single-layer plates 23 / 37 / 71 / 136 from $16^2$ to $128^2$). Coarse trilinear hex8 cells on a one- or two-element-thick layer get ever flatter and lock in bending, so the coarse correction misses the smooth bending error. Larger σ hides it (the same plates at the overestimated `sigma_estimate` took ≤ 60). Remedies belong to later PRs: algebraic coarsening (PR 13) and the solid-shell element (M4).
- **High contrast at low σ:** at its own $\lambda_1$ a SIMP field's lowest modes are spurious island and void-region modes of $s_K = \rho^3$, $s_M = \rho$ ($\lambda_1$ is $10^{-6}$–$10^{-3}$ of the solid's). Deflation removes the island modes, but soft-region error that trilinear coarse functions cannot separate from the stiff material remains: 87 on the checked-in field, 109–135 on other seeds, 452 on an unfiltered binary field (every case converges). A SciPy prototype with Jacobi-smoothed, phase-aware prolongation plus deflation reached 34 (checked-in field) and 55 (seed 4); such operator-dependent transfers widen the coarse stencils beyond 27 points and belong to PR 13 (algebraic multigrid). Topology optimization removes the spurious modes with a modified mass interpolation (PRs 21–23), and PR 23 re-runs the contrast test on a real design.

## Unstructured p-multigrid (`pmg`)

```python
from jaxmodal.precond import pmg

P = pmg(op, free, sigma, degree=2, ratio=30.0)  # op: an Unstructured operator, sigma > 0
x, info = pcg(ap_operator(op, free, sigma), b, P, tol=1e-8)  # or minres / lobpcg(precond=P)
```

`pmg` builds a two-level V-cycle for hex20/hex27 (and mixed) meshes:
- **Fine level:** the matrix-free $A_P$, smoothed by Chebyshev over node-block Jacobi.
- **Coarse level:** the hex8 Galerkin matrix $P^\top A_P P$ on the corner nodes, solved exactly by SciPy `splu` on the host. This solve is for development and tests only; PR 13 replaces it with AMG on the same matrix.

`pmg` has no type of its own. It returns a `GMG` (the shared cycle, below) with:
- one level `((A_P, S),)`;
- one `NodalTransfer`;
- `coarse = HostLU(...)`;
- no deflation.

Masking ($m\odot V(m\odot r)$), symmetry and the `precond` interface are therefore those of `gmg`. On an all-hex8 mesh, $P = I$ and the coarse matrix is the assembled hex8 $A_P$, so the coarse solve is exact and PCG converges in one iteration.

**Shared cycle.** `_vcycle` and `_vcycle_apply` are the same code for both preconditioners:
- `GMG.coarse` takes any pytree callable: `Cholesky` for `gmg`, `HostLU` for `pmg`, and PR 13's device AMG next.
- The transfers can be any object with a prolongation call and a `restrict` method: the voxel `Transfer` or `NodalTransfer`.

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
- On curved elements it still spans a nested coarse space (trilinear in the reference coordinates on the fine geometry), so $P^\top A_P P$ remains a true Galerkin operator. Rigid rotations are then only approximately in that space, which matters for PR 13's near-null space.
- On a conforming mesh, every element containing a node gives it the same parents. This is assumed, not checked.

**Galerkin hex8 matrix.** The fine operator is never assembled. For element $e$:
- $P_\text{loc}$ is its local transfer ($k\times8$);
- $m_e$ holds its free flags;
- $A_e = s_{K,e}K_e + \sigma\rho_{\phi(e)}s_{M,e}M_e$, with $M_e$ consistent or lumped as the operator's `mass`.

Then
$$C_e = (P_\text{loc}\otimes I_3)^\top\big(\mathrm{diag}(m_e)\,A_e\,\mathrm{diag}(m_e) + D_e\big)(P_\text{loc}\otimes I_3), \qquad P^\top A_P P = \sum_e Z_e^\top C_e Z_e,$$
where $Z_e$ maps the element's 8 corners to the coarse numbering. $D_e$ is the owned diagonal $d = \sigma\,n_\text{nodal}\,m + s(1-m)$ on the nodes $e$ owns (each node's first element in block order), as in `gmg`.

Assembly:
1. `element_matrices` ([`operators.md`](operators.md)) gives $s_K K_e$ and $\rho s_M M_e$ per chunk.
2. One jitted `lax.map` per block runs over the operator's padded chunks. It contracts the $(c, n, 3, n, 3)$ view with $P_\text{loc}$ into $(n_\text{pad}, 24, 24)$ cells. Pads are zero and own nothing.
3. The host drops the pads and sums the cells into a SciPy CSR over the coarse DOFs $3c + i$ (int32 indices).
4. The host symmetrizes once, $A_c = (A_c + A_c^\top)/2$. Without this, CSR duplicate summation adds the $(i,j)$ and $(j,i)$ terms in different orders and leaves about $10^{-17}$ relative asymmetry.

The matrix is affine in σ: $G_K + \sigma G_M$, with the masked diagonal folded in. PR 13's σ rounds may use this. `_hex8_galerkin(op, free, sigma)` returns $A_c$ and the transfer. It is PR 13's input and stays private until then.

**Smoother and V-cycle.** $S$ = `chebyshev(A_P, block_jacobi(op, free, σ), lmax_estimate(...), degree, ratio)`, with degree 2 and ratio 30 by default (PR 10). The cycle is gmg's with one level:
$$x = S b,\qquad x \mathrel{+}= P\,A_c^{-1}P^\top(b - A_P x),\qquad x \mathrel{+}= S(b - A_P x).$$
- It is symmetric (the same $S$ twice, exact coarse solve).
- It is positive definite under PR 10's $\lambda_\max$ safety assumption.
- It runs in fp64; there is no fp32 option.
- One cycle costs $2d$ $A_P$ applies plus one coarse solve.

**Coarse solve (`HostLU`).** The host factors `splu(A_c.tocsc())` with the default COLAMD ordering. Inside the jitted cycle, `jax.pure_callback(..., vmap_method="sequential")` applies it; the result is declared with the input's shape and dtype (always fp64).

`HostLU` is a pytree with no leaves whose treedef is the instance itself, hashed by identity. This is the one closure exception the roadmap allows. Consequences:
- Every new `pmg` build recompiles its jitted callers (`pcg`, `lobpcg_step`) once; reusing a build does not.
- The jit caches hold the treedef, and with it the factor and the compiled executables, until `jax.clear_caches()`. `del` and `gc.collect()` do not free it. At hex20 $24^3$ the factor is about 1.5 GB. The p-MG tests and `benchmarks/pmg_study.py` clear the caches after each case.

PR 13's device AMG has only array leaves and has neither issue.

Geometric nested dissection on the corner coordinates factors 4–8× faster: 3.5 s and 0.69 GB at $n = 24$, 17.8 s and 2.3 GB at $n = 32$. But `Unstructured` carries no coordinates, so COLAMD is kept for this test-only path ([`open_items.md`](open_items.md)).

**σ.** σ must be concrete and > 0. `pmg` has no mesh with which to prove $A_P$ nonsingular at σ = 0 (the BCs may leave rigid modes), so:
- σ == 0 raises `SingularShiftError`, the type `gmg` uses, so PR 13's σ pre-estimate can take the same fallback;
- σ < 0 or NaN raises `ValueError`.

Refusing σ = 0 has a cost. A pre-estimate started at σ = 0 needs fewer rounds and iterations on clamped bodies:

| case | from σ = 0 | from σ > 0 |
|---|---|---|
| cube | 1 round, 4 iterations | 2 rounds, 10 iterations |
| plate, $a = 4$ | 1 round, 20 iterations | 3 rounds, 60 iterations |

PR 13 can accept σ = 0 when `solve_modes` has found no rigid modes and every `s_K > 0` (open item).

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
| hex20 single-layer plate $16^2\times1$, element aspect $a$ = 1 / 2 / 4 / 10 | 5.9k | 15 / 21 / 63 / not converged (stagnates after 600–740) |
| hex20 box $8^3$ with $h_z = h/a$, $a$ = 1 / 2 / 4 / 10 | 8.0k | 14 / 17 / 28 / 76 |

Degree 2 is the default. Per PCG solve (V-cycle $2d$ applies plus PCG's one):

| degree | $A_P$ applies | $t_A$ with a ~1 $t_A$ production coarse solve |
|---|---|---|
| 2 | 70 | ~84 |
| 3 | 66–77 | ~80 |
| 4 | 81 | ~90 |

The cost is a tie, and degree 2 needs the least smoother workspace. At hex20 $24^3$ ($m = 25$): $t_A$ = 53 ms; the V-cycle takes 994 ms, of which the test-only `splu` solve is 772 ms (14.5 $t_A$) and the rest 4.2 $t_A$ ($2d$ = 4 applies plus transfers). The build takes 13.3 s, mostly `splu`. `_hex8_galerkin` alone at hex20 $44^3$ (1,075,275 DOF, 21.2M nonzeros) takes 1.4 s stored and 2.9 s otf, compilation included (about +1.5 GB host, measured in review).

**LOBPCG** (10 modes, tol $10^{-6}$, σ = $\lambda_1$; times include compilation; eigenvalues agree with the Jacobi run to $\le 9\cdot10^{-11}$):

| mesh | DOF | Jacobi | p-MG degree 2 |
|---|---|---|---|
| hex20 $12^3$ | 24,843 | 435 iterations, 6.1 s | 15, 1.3 s |
| hex20 beam $20\times2\times2$ | 1,863 | 309 iterations, 1.6 s | 14, 0.3 s |
| hex27 $16^3$ | 107,811 | 509 iterations, 21.7 s | 15, 3.4 s |

**Limitations** ([`open_items.md`](open_items.md)).
- **Flat elements:** the hex8 coarse space locks in bending when elements are flat ($a \gg 1$), so the coarse correction misses the smooth bending error. See the table above; the hex27 plate $32^2\times1$ at $a = 10$ needs > 1000 iterations and does not converge. Slenderness alone with $a = 1$ is fine (beam: 16). Revisit in PR 13 (AMG on the high-order level, a near-null space that keeps bending) and M4 (solid-shell elements).
- **`splu` scaling** (coarse hex8 matrix of an $n^3$ box, COLAMD):

  | $n$ | coarse DOF | factor time | $L + U$ | process peak |
  |---|---|---|---|---|
  | 16 | 15k | 1.0 s | 0.2 GB | 0.8 GB |
  | 24 | 47k | 14 s | 1.5 GB | 3.9 GB |
  | 32 | 108k | 140 s | 6.2 GB | 18 GB |

  The tests therefore stay at or below the 47k coarse DOF of hex20 $24^3$. PR 13's AMG replaces this solve.

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
- `tests/test_precond.py`: `ap_operator` against `a_p` on unmasked vectors (σ = 0, positive, negative); node blocks and block-Jacobi inverses against SciPy for stored/on-the-fly, consistent/lumped, nodal masses, `s_K`, mixed hex8/hex20, hex27 and a masked two-phase voxel grid with `s_K`, all with partially constrained nodes; the σ < 0 and NaN raises; $1.1\,\tilde\lambda_\max/\lambda_\max \in [1, 1.5]$ and the invariant-start guard; Chebyshev against a dense spectral reference to $10^{-12}$ (degrees 1–4), symmetry, $A_P$-self-adjoint error propagation, positivity, and fewer PCG iterations than block Jacobi; the h-study smoke run.
- `tests/test_gmg.py`: every coarse level equals $P^\top A_P P$ from SciPy to $10^{-12}$ on masked meshes with nodal masses, two phases, odd axes (level-2 weights ⅓, ⅔), stopped axes, both stored axis orders, consistent and lumped mass, at the default and a 7-cell setup chunk; transfers reproduce linear fields on every level, restriction is $P^\top$, kept columns nonzero and dropped ones zero; the smoother blocks are the dense diagonal group blocks and the group apply their inverse; the V-cycle equals a dense NumPy composition to $10^{-10}$; symmetry and definiteness; the iteration bounds (h-independence ≤ 20%, SIMP, plate, free, beam, odd); the singular guard's raises and builds; island detection and the deflated cycle against a dense $Q + (I - QA)V(I - AQ)$ with an independently built $Z$; padded deflation sizes; the default degrees; `grid_ids` on slab-padded stencils; LOBPCG with GMG against `reference_modes`; no recompiles; the input errors; the study smoke run. PR 11b: the fp32 hierarchy's leaves (fp32 except the fine operator, `free`, `coarse.L`, `defl`; `levels[l][0] is levels[l][1].A`), the fp32 cycle within $10^{-5}$ of fp64 ($3\cdot10^{-4}$ on the thin plate), symmetric and positive with ≤ 0.75× the temporaries, and LOBPCG with ≤ 1.2× the iterations and λ within $10^{-9}$ (box $16^3$; slow: box $32^3$, plate $32^2\times1$, odd $33\times8\times8$); `gmg_memory`'s storage ≥ the unique buffers of the built hierarchy (fp64, fp32; boxes, a masked odd grid, a cropped grid with $k = 3$ fine groups, the island case with `s_K`) and within 5% on full boxes; the V-cycle temporaries ≤ `workspace` ($16^3$; slow $32^3$, within 1.25×) and the LOBPCG steps with GMG ≤ `lobpcg_memory`'s step term; slow: at $48^3$ the GMG steps' extra temporaries over the Jacobi steps lie within 0.6 blocks below `GMG_STEP`; every σ = 0 refusal is a `SingularShiftError`.
- `tests/test_pmg.py`: the transfer reproduces trilinear fields on graded hex20 and hex27 boxes to $10^{-12}$, restriction and the dense prolongation equal an independent element-loop $P$ (also on the mixed hex8 + hex20 mesh, whose hex8 nodes get identity rows), with hand-checked ½ / ¼ / ⅛ rows; `_hex8_galerkin` equals $P^\top a_p(K + \sigma M, free, s) P$ from SciPy to $10^{-12}$ and is exactly symmetric (hex20, hex27, mixed; stored and otf; consistent and lumped; two phases, random `s_K`/`s_M`, nodal masses, a mixed clamped/sliding mask), also with padded chunks (whose `element_matrices` are zero) and on all-hex8 meshes (the assembled $a_p$); the V-cycle at degree 3, ratio 20 equals a NumPy composition with an independently built smoother and $(P^\top A P)^{-1}$ to $10^{-10}$ and is SPD on the free DOFs; symmetry on unmasked vectors, masked outputs, `type(pmg(...)) is GMG`; PCG in ≤ 2 iterations on a pure hex8 box; h-independence (≤ 20% growth from $3^3$ to $12^3$, hex20 and hex27; slow: hex20 $24^3$); ≤ 30 iterations on the beam, free, curved, nodal-mass and `s_K`-checkerboard cases; LOBPCG with p-MG against `reference_modes` in ≤ 40 iterations (hex20, hex27); the input errors; no recompile for a reused build; the factor freed after `jax.clear_caches()`; the study smoke run.
