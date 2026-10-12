# PR 11a plan: voxel geometric multigrid (core)

PR 11 (M3, depends on PRs 6, 9c, 9b's estimator and 10) is split:

- **11a (this plan):** the GMG hierarchy, the Galerkin coarse levels, the transfers, the smoothers, the V-cycle and the `gmg(...)` builder; PCG and LOBPCG tests.
- **11b:** `solve_modes` dispatch with the σ pre-estimate and the user hint, the fp32 V-cycle, the memory estimator and the Mac acceptance run.

11b needs everything here; 11a needs nothing from 11b. The roadmap tests split as follows:

| roadmap test | PR |
|---|---|
| each coarse level equals $P^\top A_P P$ (SciPy) to 1e-12 | 11a |
| iterations grow ≤ 20% over 3 refinements | 11a |
| SIMP-like field, single layer, free-free: ≤ 100 PCG iterations to 1e-8; coarsest Cholesky succeeds | 11a |
| σ pre-estimate within 2× of λ₁; a user hint skips it | 11b |
| `solve_modes` on an odd voxel mesh matches `eigsh` | 11b (11a runs the same check through `lobpcg(..., precond=gmg(...))`) |
| fp32 GMG: same tol, ≤ 20% more LOBPCG iterations | 11b |
| Mac acceptance (2M DOF, ≤ 10 min, peak within 20% of the estimators) | 11b |
| deferred GPU checks (100M low-memory step ≤ 7.5 blocks, 100M voxel solve) | 11b |

## Measured facts (prototype `scratchpad/p11/`; jax 0.11.2, fp64, M3 Max CPU)

Steel ($E = 210\cdot10^9$, $\nu = 0.3$, $\rho = 7850$), unit box, `xmin` clamped unless noted; σ from `sigma_estimate` on the $8^3$ mesh; PCG on $A_P(K + \sigma M)$ to $10^{-8}$, 4 right-hand sides; Chebyshev degree 2, ratio 30, over the group block smoother (below).

| case | PCG iterations |
|---|---|
| boxes $8^3$ / $16^3$ / $32^3$ / $64^3$ | 14 / 14 / 14 / 14 |
| odd $9\times8\times8$ / $17\times8\times8$ / $33\times8\times8$ | 14 / 14 / 17 |
| free $32^3$ (σ > 0) | 13 |
| plate $16^2\times1$ / $32^2\times1$ / $64^2\times1$ / $128^2\times1$ | 25 / 35 / 38 / 38 |
| plate $64^2\times1$ with node-block Jacobi smoothing instead | 240 |
| plate $64^2\times2$ | 28 |
| beam $64\times2\times2$ / $128\times1\times1$ | 22 / 35 |
| smooth SIMP field ($\rho \in [10^{-3}, 1]$, $p = 3$, Gaussian random field), $32^3$ | 19 |
| filtered binary SIMP field (below), $16^3$ / $32^3$ / $64^3$ | 24 / 55 / 76 |
| unfiltered binary SIMP field, $32^3$ (11 islands, contrast $10^9$) | 182 (degree 6: 100) |
| anisotropic spacing $h_z = h/2$ / $h/4$ / $h/10$, $32^3$ | 15 / 35 / 151 |

- **Galerkin exactness.** On a $5\times4\times3$ occupancy-masked clamped mesh with two phases, random `s_K`, `s_M` and nodal masses, every coarse level equals $P^\top A_P P$ from the PR 4 SciPy matrices to $\le 10^{-15}$ relative, consistent and lumped.
- **LOBPCG** (10 modes, tol $10^{-6}$, 9-block step, GMG at the σ estimate): 17 iterations at $32^3$ (108k DOF, 3.1 s) and 18 at $87^3$ (2.04M DOF, 46 s; LOBPCG times include compilation, the implementation re-measures every figure with the shipped code). The $87^3$ run took 134 s in total (σ pre-estimate 73 s with 252 Jacobi-LOBPCG iterations, GMG build 8.2 s) but peaked at 26.7 GB RSS, from the unchunked level-1 setup einsum; 11a chunks it (below).
- **V-cycle cost** at $87^3$, $m = 15$, warm (performance review): 1.64 s against $t_A$ = 0.33 s for one $A_P$ apply, $4.95\,t_A$, inside the budget model's $5.5\,t_A + 0.1$ s (`docs/eigensolvers.md`). Inside the LOBPCG step the GMG preconditioner adds about 5 blocks of workspace over block Jacobi (21.1 against 16.0 blocks); 11b's estimator models it.
- **Setup memory** at $87^3$ (process peak RSS, baseline operator + block Jacobi 0.98 GB): prototype 23.7 GB; with a chunked level-1 sum but the nodal term as a separate $(n_\text{nodes}, 24, 24)$ product, 13.8 GB; without that product, 7.1 GB at chunk 65536 and 1.35 GB less at chunk 16384 (same build time). The rest is chunk-independent transients (a full $(n_\text{el}, 8, 8)$ `child_P`, int64 broadcasts, a padded ELL copy) that the design below removes.

**Filtered binary SIMP field.** `rng = default_rng(3)`; $f$ = `gaussian_filter(rng.standard_normal((n,)*3), n/16)`; $b = [f > q_{0.6}(f)]$; $\rho = 10^{-3} + (1 - 10^{-3})\,\mathrm{uniform\_filter}(b, 3)$; $s_K = \rho^3$, $s_M = \rho$. It is binary with a 3-cell transition layer, like a filtered and projected SIMP design. Its iteration count grows with $n$ at fixed geometry (sharper interfaces relative to the coarse cells): GMG with geometric transfers is not robust to high-contrast interfaces that coarse cells cut. This is documented as a limitation (Docs); PR 23 re-runs the contrast test on a real design.

## Design

### Module and entry

New module `precond/gmg.py`, exported from `jaxmodal.precond`:

```python
P = gmg(op, free, sigma, *, degree=2, ratio=30.0)
```

- `op` is a `Stencil` (PR 6). A non-`Stencil` raises `TypeError`. A new helper `active_cells(op)` in `operators/voxel.py` returns the $(n_x, n_y, n_z)$ bool occupancy grid (cropping the slab padding of `op.cell` and undoing `op.axes`), so the knowledge of the stored layout stays in `voxel.py`; `gmg` is its only consumer, and the node numbering of every level comes from `voxel_mesh(shape, mask=...)` (level 0 must reproduce the stencil's own numbering: a voxel mesh renumbered by `rcm`/`renumber`, which `stencil` accepts, raises `ValueError` "gmg needs the voxel_mesh node order; do not renumber voxel meshes", checked before anything else; voxel grid order is already narrow, so refusing costs nothing).
- `free` is the `(n_dof,)` bool mask; `sigma` ≥ 0 is concrete and required (GMG is always built at a target σ; a negative σ raises, as in `block_jacobi`); `check_x64` runs first.
- The setup chunk is a module constant `SETUP_CHUNK = 16384` fine cells (like `SLAB_CELLS`), which 11b's estimator reads.
- It returns a pytree `GMG` whose `__call__(r)` applies one V-cycle to an `(n_dof, k)` block. It fits every `precond` slot (`pcg`, `minres`, `lobpcg`): the transfers, levels, smoothers and the coarsest factor are leaves, the grid shapes and level count are static metadata, so a new design (same grid, new `s_K`) never recompiles the solver step.
- The V-cycle approximates $A_P^{-1}$ with $A_P = m(K + \sigma M)m + s(I - m)$, $s$ = `ap_operator(op, free, sigma).s` (PR 10 default). It applies $m \odot V(m \odot r)$: masking the input too keeps it symmetric on the whole space (restriction would otherwise mix constrained residuals into the coarse correction), at no cost.
- **Singular guard.** With σ = 0 and rigid modes left by the BCs, $A_P$ is singular; the rounded coarsest Cholesky then gives NaN (free and sliding $8^3$–$32^3$ boxes, three reviewers' probes) or a finite factor that amplifies rigid components by ~$10^{13}$ (free $6^3$, correctness review). The guard is exact and host-side: when σ == 0, `gmg` calls `rigid_modes(voxel_mesh(shape, spacing, mask=active_cells(op) & (s_K > 0)), free)` (cells with `s_K` = 0, which `scaled` accepts, add no stiffness and can cut the body into islands or hinges, so they are left out of the check; one host copy of `s_K`), runs it before any device setup and keeps only the column count, so its transient (~0.4 GB host at 2M DOF for one part) never stacks on the build peak and 11b's estimator ignores it, and raises `ValueError` ("use sigma > 0") if it returns any column; rigid modes are trilinear, so they lie in the range of every $P$ and the coarsest level is singular exactly when $A_P$ is. As a backstop for breakdowns the check cannot foresee, a non-finite coarsest factor also raises. The level-0 numbering check runs before the guard, so `free` indexes the rebuilt mesh.
  - **Meshes `rigid_modes` rejects.** The stencil accepts masks that `parts()` rejects (edge or vertex contacts) and more than `MAX_PARTS` islands. At σ == 0 `gmg` catches that `ValueError` and raises its own, chained: "cannot verify that A_P is nonsingular at sigma = 0 (<original message>); use sigma > 0". This refuses a few SPD cases (a hinge clamped on both sides, more than 10 islands all clamped). Decision: accept this conservative refusal rather than add a cap-free counting path to `rigid_modes` (tests review suggested one): σ = 0 is an unusual request, 11b's dispatch always passes the σ estimate, and σ > 0 makes every such mesh SPD.
  - **Tiny σ.** The guard is exact only at σ == 0; a σ far below λ₁ (e.g. $10^{-30}\lambda_1$) passes it and gives a poorly conditioned V-cycle. Choosing σ physically (the σ pre-estimate) is the caller's responsibility, stated in the docstring and docs. A pivot-ratio threshold was rejected: legitimate SPD cases fall with size (clamped plate at σ = 0: $4.5\cdot10^{-6}$ at $64^2\times1$, $6.3\cdot10^{-7}$ at $128^2\times1$, about 7× per refinement; beam $128\times1\times1$: $1.7\cdot10^{-6}$), so any fixed threshold rejects large slender bodies, and SIMP designs with large ersatz-void regions ($\rho_\min = 10^{-3}$, $p = 3$) reach $1.6\cdot10^{-11}$ at σ > 0 while SPD (correctness review).

### Hierarchy (host)

**1D coarsening rule** for an axis with $n$ elements:
- $n \le 2$: the axis stops (identity transfer along it).
- otherwise $c = \lceil n/2 \rceil$ coarse elements with nodes at fine positions $X = (0, 2, \dots, 2(c-1), n)$. An odd $n$ gives one last coarse cell spanning a single fine element.

Each level carries its per-axis **physical** node positions ($p_0 = 0, 1, \dots, n$ in fine spacing units; $p_{l+1} = p_l[X]$), and the linear interpolation weights come from them: fine node $i$ between coarse nodes $X_k \le i \le X_{k+1}$ gets $w_1 = (p_i - p_{X_k})/(p_{X_{k+1}} - p_{X_k})$, $w_0 = 1 - w_1$. Index-space weights would be wrong from level 2 on: once a level holds a half-size tail cell, an even count pairs it with a full cell and the midpoint needs ⅓/⅔, not ½/½ (e.g. $n = 7$: level-1 nodes at physical 0, 2, 4, 6, 7; also 15, 31, 63 and 87 → 44). Index weights reproduced a linear field only to an error of 2.9 there, physical weights to $4\cdot10^{-15}$ at every level (correctness review); iteration counts were the same. Coarsening repeats until every axis is stopped; that grid is the coarsest level (at most $3^3$ nodes, 81 DOF), where no smoother is built.

**Coarse active cells.** A coarse cell is active when it contains an active fine cell. The coarse active nodes are the corners of active coarse cells, and the coarse numbering comes from `voxel_mesh(cshape, mask=active)` (the PR 1 order). Every such node has a nonzero prolongation weight at some node of an active fine cell (each fine cell in a coarse cell has a node strictly inside the corner's support or coincident with it), and every other coarse grid node has a zero $P$ column.

**Deviation from the roadmap:** the roadmap keeps coarse DOFs with no active fine support "masked with diagonal $s$". 11a drops them instead: their $P$ column is zero, so in $P^\top A_P P$ they would be decoupled rows with no effect on the V-cycle. Dropping them is equivalent, smaller, and keeps the coarsest Cholesky SPD the same way. The Galerkin test compares on the kept DOFs and checks that the dropped columns of $P$ are zero.

### Galerkin coarse levels

The fine operator is never assembled. With $P_e$ the $24\times24$ local prolongation of fine cell $e$ into its coarse parent ($P_\text{loc}\otimes I_3$, $P_\text{loc}$ the $8\times8$ trilinear weights from the per-axis tables, VTK corner order) and $m_e$ the 24 free flags of $e$:

- **Level 1 cell matrices** $(n_{c,1}, 24, 24)$:
$$
C_c = \sum_{e \in c} \tilde P_e^\top A_e \tilde P_e, \quad \tilde P_e = \mathrm{diag}(m_e)\,P_e, \quad A_e = s_{K,e} K_{\phi(e)} + \sigma \rho_{\phi(e)} s_{M,e} M_e,
$$
  $K_\phi$ and $m_{ab}$ from `reference_element(op)`; $M_e = m \otimes I_3$ (consistent) or $\mathrm{diag}(\textstyle\sum_b m_{ab}) \otimes I_3$ (lumped), matching `op.mass`.
- **Nodal terms.** $d = \sigma\,n_\text{nodal}\,m + s(1 - m)$ per fine DOF (nodal masses and the constrained diagonal of $A_P$). Each fine node's $d$ is added once, to the diagonal of the masked element matrix $\mathrm{diag}(m_e) A_e \mathrm{diag}(m_e)$ of the first fine element (lowest id) containing it. Since $d$ is diagonal and $\tilde P_e$ is masked only where $d$ is not, $P_e^\top(\mathrm{diag}(m_e)A_e\mathrm{diag}(m_e) + D_e)P_e$ gives the same sum as adding $p^\top \mathrm{diag}(d_n) p$ per node, without an $(n_\text{nodes}, 24, 24)$ array (that form cost 3.1 GB plus temporaries at 2M DOF). The owner table is one $(n_\text{el}, 24)$ array of $d$ values (zero where not owned).
- **Levels ≥ 2:** $C'_{c'} = \sum_{c \in c'} P_c^\top C_c P_c$, unmasked.
- **Contraction.** $P_e = P_\text{loc} \otimes I_3$, so the products view the 24×24 matrices as $(8, 3, 8, 3)$ and contract with the $8\times8$ $P_\text{loc}$ (`einsum("eac,eaibj,ebd->ecidj")`), never forming a $24\times24$ $P_e$; the per-cell $P_\text{loc}$ is built per chunk from the per-axis $2\times2$ tables.
- **Chunking.** Every level's sum (fine → level 1 and the coarse ones; level 2 alone is ~1.5 GB unchunked at 2M DOF) runs as one `lax.fori_loop` over chunks of `SETUP_CHUNK` child cells, padded to a whole number of chunks, scatter-adding into the parent cell matrices. On level 0 the same loop also scatter-adds the masked element matrices (plus $D_e$) into the level-0 group blocks when an axis is stopped from the start, so the fine cells are visited once. Host index arrays are int32.
- **Storage.** Each level's operator is a `BlockELL` (PR 10's fields, one bucket, so its `_apply` is shared) with 27 slots per node (slot = corner offset difference; missing neighbours point at the node itself with zero blocks, the slot fill is the reported padding), built on the device from the cell matrices by one scatter-add of their $8\times8$ node blocks into the padded value array; 3×3 blocks, fp64. The fine level stays the matrix-free `ap_operator`. PR 10's `block_ell` (host SciPy) is not used: its host COO would be ~0.8 GB at 2M DOF and ~19 GB at the 100M GPU target.
- **No hidden copies.** Every array enters the jitted setup loops as an argument, never through a closure (closed-over leaves became HLO constants and added 1.9 GB in the performance review's probe). The chunk loop clamps the last chunk's start (as `dynamic_slice` does) and masks the rows already done instead of padding the child arrays, and the ELL values are scatter-added per chunk of cells inside a loop, without a full transposed copy of the cell matrices.
- **Setup peak target.** At $87^3$ the build adds ≤ 3 GB over the operator and block-Jacobi baseline (measured with the harness, process peak RSS; the prototype added 22.7 GB; a lean prototype of this design without the two copy cuts above added 2.98 GB, and the cuts save an estimated 0.5–0.8 GB). The measured value goes into the docs and is what 11b's estimator reproduces.

### Transfers

Prolongation runs matrix-free per level: scatter the coarse nodal values into the full coarse grid (`.at[gc].set(..., unique_indices=True)`), apply the three 1D two-point interpolations (gather of the two coarse neighbours times the weights, axis by axis), gather the active fine nodes. Restriction is its exact transpose, `jax.linear_transpose` of the prolongation (`unique_indices=True` makes the scatter transposable to a gather). The full-grid temporaries are ≤ 8 coarse grids (one fine grid) of `(…, 3, k)` values per level.

### Smoothers: stopped-axis group blocks

Point (node-block) smoothing does not damp the error modes that semi-coarsening leaves along stopped axes: the $64^2\times1$ plate needs 240 iterations with node-block Jacobi. 11a smooths with **group blocks**:

- A group is the set of nodes sharing coordinates on the coarsening axes; its size is $k = \prod_{d\,\text{stopped}} (n_d + 1) \le 27$ nodes ($3k \le 81$ DOF). With no stopped axis, groups are single nodes ($k = 1$, node-block Jacobi).
- The group blocks $D_l$ are the $3k \times 3k$ diagonal blocks of the level operator, summed from the cell matrices (level ≥ 1) or, on level 0 with a stopped axis, from the masked fine element matrices plus $D_e$ in the level-1 loop. With $k = 1$ (no stopped axis) level 0 uses `block_jacobi(op, free, sigma)` and coarse levels the node blocks (ELL slot 13) through `_inverse_blocks` and `_block_jacobi`.
- They are inverted through PR 10's `_inverse_blocks` (batched Cholesky, absent slots of partial groups masked to identity). For $k > 1$ the apply is gather-only: gather the $(G, 3k)$ residuals (absent slots read a zero row), batched GEMM, then a precomputed inverse permutation takes each DOF's value back (groups partition the nodes, so the scatter-add is a permutation: 34 ms against 113 ms for `.at[].add` at 2M DOF, performance review).
- Each level's smoother is `chebyshev(A_l, D_l, lmax_estimate(A_l, D_l), degree, ratio)` (PR 10), symmetric.

With it the plates converge in 25–38 iterations (h-independent from $64^2$) and the beams in 22–35.

### V-cycle

On level $l$ with operator $A_l$, smoother $S_l$ and transfer $P_l$:
$$
x = S_l b,\quad x \mathrel{+}= P_l\,V_{l+1}(P_l^\top (b - A_l x)),\quad x \mathrel{+}= S_l(b - A_l x),
$$
and on the coarsest level $x = L^{-\top}L^{-1} b$ with $L$ the dense fp64 Cholesky of $A_\text{coarsest}$ (densified by applying the ELL to the identity). Pre- and post-smoothing use the same symmetric $S_l$ and the coarse solve is exact, so the V-cycle is symmetric; it is positive definite when $\|I - S_l A_l\|_{A_l} < 1$, which Chebyshev on $[\hat\lambda/30, \hat\lambda]$ with $\hat\lambda \ge \lambda_\max(D_l^{-1}A_l)$ gives (the Chebyshev residual polynomial has $|r| < 1$ on $(0, \hat\lambda]$). The recursion is unrolled at trace time (the level count is static).

### Limitations (documented)

- **Anisotropic spacing.** Coarsening is by element count, not spacing: $h_z = h/10$ needs 151 iterations. Spacing-aware semi-coarsening needs line smoothers along long stopped axes; not in the roadmap, noted in Not covered.
- **High contrast.** Binary densities with sharp interfaces grow with $n$ (above); smooth and filtered fields are fine at the tested sizes.

## Files

| file | change |
|---|---|
| `src/jaxmodal/precond/gmg.py` | new: hierarchy, cell matrices, ELL levels, transfers, group smoother, `GMG`, `gmg` |
| `src/jaxmodal/operators/voxel.py` | `active_cells(op)` |
| `src/jaxmodal/precond/__init__.py` | export `gmg`, `GMG` |
| `tests/test_gmg.py` | new |
| `tests/data/simp32.npy` | new: the filtered binary SIMP density, $32^3$, float32 (128 KB) |
| `benchmarks/gmg_study.py`, `benchmarks/results/gmg_study.json` | new: the iteration table above, re-measured with the shipped code |
| `docs/preconditioners.md`, `README.md`, `docs/ROADMAP.md` | GMG section, status |

## Tests (`tests/test_gmg.py`)

Iteration tests use steel, x64, PCG tol $10^{-8}$ with 4 right-hand sides (`tests/helpers.rhs`), and σ from `sigma_estimate` (fixed `seed`, with `R = rigid_modes(...)` for free bodies). Exactness tests (1–4) use unit materials ($E \sim 1$–$30$, $\rho \sim 1$), where σ = 0.37 makes the mass terms visible. Several tests read the hierarchy internals (levels, transfers, smoother blocks) from the `GMG` pytree and module helpers.

1. **Galerkin levels vs SciPy** (roadmap). Meshes:
   - A: $7\times5\times3$ occupancy-masked (`xmin` fully active, clamped), with one inactive coarse cell next to active ones; $x$ = 7 is odd with $\lceil 7/2 \rceil$ even, so level 2 has the ⅓/⅔ weights;
   - B: $7\times3\times2$, stopped axes from the start;
   - C: $3\times4\times9$ (z longest, so `op.axes` is the other permutation), masked, sliding `zmin: z` + `xmin: x`.

   Two phases, random `s_K`, `s_M`, nodal masses on half the nodes, consistent and lumped. Densify each transfer ($P_l$ on the identity) and level ($A_l$ on the identity); assert $A_{l+1} = P_l^\top A_l P_l$ with $A_0$ = PR 4's `a_p(K + σM, free)` to 1e-12 relative. Also assert $\|\sigma P_0^\top M P_0\| / \|A_1\| > 10^{-4}$ (the mass terms are observable). Run with `SETUP_CHUNK` patched to 7 (several chunks, a partial last chunk) and at the default.
2. **Transfers.** On every level of mesh A and of a $15\times2\times2$ beam: $P_l$ reproduces trilinear fields in physical coordinates exactly (1e-12); restriction equals $P_l^\top$ densified (1e-14); every kept coarse column of $P_l$ is nonzero; in the full-grid $P$ of mesh A the columns of dropped grid nodes are exactly zero.
3. **Group blocks vs dense.** On meshes A and B (and with `SETUP_CHUNK` = 7): each level's group blocks before inversion equal the $3k \times 3k$ diagonal blocks of the densified level operator (level 0: SciPy `a_p`, with nodal masses, constrained DOFs and partial groups) gathered by the group index, to 1e-12; the group apply equals the dense block-diagonal inverse times $v$ (1e-12), with masked output.
4. **Dense V-cycle reference.** On meshes A and B: densify $A_0$ (SciPy), every $P_l$, every smoother $S_l$ as built (on the identity) and the exact inverse of the densified coarsest level; compose $x = S b$, $x \mathrel{+}= P\,V(P^\top(b - Ax))$, $x \mathrel{+}= S(b - Ax)$ in NumPy with input and output masking; `gmg(...)(U)` equals it to 1e-10 relative on random $U$ (masked and unmasked).
5. **Symmetry and definiteness.** $u^\top V v = v^\top V u$ to 1e-12 relative on unmasked random $u, v$ and $u^\top V u > 0$ on masked ones, on a clamped and on a free box; constrained outputs are exactly 0.
6. **h-independence** (roadmap). Boxes $8^3$ … $64^3$ (the $64^3$ case `slow`): iterations grow ≤ 20% from $8^3$.
7. **SIMP field** (roadmap). `simp32.npy` ($s_K = \rho^3$, $s_M = \rho$): ≤ 100 iterations; the test also checks the stored field's min, max and mean, so a regenerated file is noticed.
8. **Single layer** (roadmap). Plate $64^2\times1$, clamped `xmin`: ≤ 100 iterations.
9. **Free-free** (roadmap). $32^3$ free box, σ > 0: ≤ 100 iterations; the coarsest Cholesky factor is finite. The singular guard: a free $16^3$ box and the plate of test 8 with sliding `xmin: x` at σ = 0 raise `ValueError`; the clamped plate $128^2\times1$ at σ = 0 builds (its coarsest pivot ratio is $6\cdot10^{-7}$). A patched-in NaN coarsest level raises through the backstop. A clamped $8^3$ box with one interior slab at `s_K` = 0 cutting off a free half raises at σ = 0 (the stiff-cell mask yields a floating part). An edge-contact mask (two cells sharing one edge, `xmin` clamped) and 12 clamped islands at σ = 0 raise the guard's message (containing "sigma > 0", chained from the `rigid_modes` error); the edge-contact mask at σ = estimate builds. A half-void SIMP design ($32^3$, half the cells at $\rho = 10^{-3}$, $p = 3$, clamped, σ = estimate; coarsest pivot ratio $1.6\cdot10^{-11}$, SPD) builds without raising.
10. **Beam and odd axes.** $128\times1\times1$ ≤ 100; $33\times8\times8$ ≤ 30.
11. **LOBPCG.** `lobpcg(op, free, 6, tol=1e-8, precond=gmg(...))` on a $9\times6\times4$ clamped occupancy-masked box (odd $x$) matches `reference_modes` by the 9a check helper at the call's tol (as `test_api` does), in fewer iterations than with Jacobi.
12. **No recompile.** Two `gmg` builds on the same grid with different `s_K`: one compiled `pcg`, one compiled jitted V-cycle call, one compiled `lobpcg_step` (cache sizes unchanged after the second).
13. **Errors.** Unstructured operator → `TypeError`; σ < 0 → `ValueError`; x64 off → error; an `rcm`-renumbered voxel mesh → `ValueError`.

Iteration bounds were measured with the prototype (the tests review got 50 for the beam with its right-hand sides, against 35 in the table); implementation re-measures them with the shipped code and the tests' right-hand sides, keeps ≥ 1.5× margin, and the docs table quotes the shipped values. The setup peak target (≤ 3 GB added at $87^3$) is checked by a harness benchmark in `benchmarks/gmg_study.py`, not a test.

## Docs

- `docs/preconditioners.md`: a GMG section (coarsening rule with the odd-axis cell and physical-position weights, Galerkin cell matrices with the LaTeX above, the nodal terms on owner elements, the dropped coarse nodes, the direct 27-slot `BlockELL` construction and how it relates to `block_ell`, transfers, group smoother and why, V-cycle with the $\lambda_\max(D_l^{-1}A_l)$ condition and input/output masking, the singular guard, limitations, the measured iteration and setup-memory table).
- `docs/operators.md`: `active_cells(op)`, and that GMG reads the stencil's stored grid layout through it.
- `README.md`: GMG in the feature list with a short `pcg(..., precond=gmg(...))` snippet (run by `test_readme`).
- `docs/ROADMAP.md`: the PR 11 split note (like PR 9's) with the 11a/11b test table, the status line, the changed Galerkin test text ("dropped coarse DOFs (zero $P$ columns)" instead of "masked coarse DOFs with diagonal s"), the group smoother, and the anisotropic-spacing limitation next to the semi-coarsening key decision.

## Not covered

- 11b: `solve_modes` dispatch, σ hint, fp32 V-cycle, memory estimator (including the ~5-block GMG workspace in the LOBPCG step), Mac acceptance, deferred GPU checks. 11b re-checks the budget model's V-cycle term ($5.5\,t_A$; measured $4.95\,t_A$ warm) and the σ pre-estimate cost (73 s of the 134 s 2M run, the main lever for margin) against warm measurements.
- Spacing-aware coarsening and line smoothers; robustness to unfiltered high-contrast densities (PR 23 re-runs the contrast test).
- W-cycles and smoother degree tuning (degree 2 fixed default, an argument).
