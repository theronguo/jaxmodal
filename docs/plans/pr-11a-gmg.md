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
- **LOBPCG** (10 modes, tol $10^{-6}$, 9-block step, GMG at the σ estimate): 17 iterations at $32^3$ (108k DOF, 3.1 s) and 18 at $87^3$ (2.04M DOF, 46 s). The $87^3$ run took 134 s in total (σ pre-estimate 73 s with 252 Jacobi-LOBPCG iterations, GMG build 8.2 s) but peaked at 26.7 GB RSS, from the unchunked level-1 setup einsum; 11a chunks it (below).
- **V-cycle cost** at $87^3$, $m = 15$: 1.6 s against 0.54 s for one $A_P$ apply (both first-call timings, compile included; 11a re-measures them warm).

**Filtered binary SIMP field.** `rng = default_rng(3)`; $f$ = `gaussian_filter(rng.standard_normal((n,)*3), n/16)`; $b = [f > q_{0.6}(f)]$; $\rho = 10^{-3} + (1 - 10^{-3})\,\mathrm{uniform\_filter}(b, 3)$; $s_K = \rho^3$, $s_M = \rho$. It is binary with a 3-cell transition layer, like a filtered and projected SIMP design. Its iteration count grows with $n$ at fixed geometry (sharper interfaces relative to the coarse cells): GMG with geometric transfers is not robust to high-contrast interfaces that coarse cells cut. This is documented as a limitation (Docs); PR 23 re-runs the contrast test on a real design.

## Design

### Module and entry

New module `precond/gmg.py`, exported from `jaxmodal.precond`:

```python
P = gmg(op, free, sigma, *, degree=2, ratio=30.0, chunk=65536)
```

- `op` is a `Stencil` (PR 6). The hierarchy is read on the host from `op.node` and `op.cell` (cropped to the real grid and permuted back to $(x, y, z)$ through `op.axes`), so no mesh argument is needed. A non-`Stencil` raises `TypeError`.
- `free` is the `(n_dof,)` bool mask; `sigma` ≥ 0 is concrete (a negative σ raises, as in `block_jacobi`), `check_x64` runs first.
- It returns a pytree `GMG` whose `__call__(r)` applies one V-cycle to an `(n_dof, k)` block. It fits every `precond` slot (`pcg`, `minres`, `lobpcg`): the transfers, levels, smoothers and the coarsest factor are leaves, the grid shapes and level count are static metadata, so a new design (same grid, new `s_K`) never recompiles the solver step.
- The V-cycle approximates $A_P^{-1}$ with $A_P = m(K + \sigma M)m + s(I - m)$, $s$ = `ap_operator(op, free, sigma).s` (PR 10 default). The output is masked (constrained rows 0), like `block_jacobi`.

### Hierarchy (host)

**1D coarsening rule** for an axis with $n$ elements:
- $n \le 2$: the axis stops (identity transfer along it).
- otherwise $c = \lceil n/2 \rceil$ coarse elements with nodes at fine positions $X = (0, 2, \dots, 2(c-1), n)$. An odd $n$ gives one last coarse cell spanning a single fine element.

Linear interpolation weights come from the positions (1, ½, or 1 at a single-element cell's coincident nodes). Coarsening repeats until every axis is stopped; that grid is the coarsest level (at most $3^3$ nodes, 81 DOF).

**Coarse active cells.** A coarse cell is active when it contains an active fine cell. The coarse active nodes are the corners of active coarse cells, and the coarse numbering comes from `voxel_mesh(cshape, mask=active)` (the PR 1 order). Every such node has a nonzero prolongation weight at some node of an active fine cell (each fine cell in a coarse cell has a node strictly inside the corner's support or coincident with it), and every other coarse grid node has a zero $P$ column.

**Deviation from the roadmap:** the roadmap keeps coarse DOFs with no active fine support "masked with diagonal $s$". 11a drops them instead: their $P$ column is zero, so in $P^\top A_P P$ they would be decoupled rows with no effect on the V-cycle. Dropping them is equivalent, smaller, and keeps the coarsest Cholesky SPD the same way. The Galerkin test compares on the kept DOFs and checks that the dropped columns of $P$ are zero.

### Galerkin coarse levels

The fine operator is never assembled. With $P_e$ the $24\times24$ local prolongation of fine cell $e$ into its coarse parent ($P_\text{loc}\otimes I_3$, $P_\text{loc}$ the $8\times8$ trilinear weights from the per-axis tables, VTK corner order) and $m_e$ the 24 free flags of $e$:

- **Level 1 cell matrices** $(n_{c,1}, 24, 24)$:
$$
C_c = \sum_{e \in c} \tilde P_e^\top A_e \tilde P_e, \quad \tilde P_e = \mathrm{diag}(m_e)\,P_e, \quad A_e = s_{K,e} K_{\phi(e)} + \sigma \rho_{\phi(e)} s_{M,e} M_e,
$$
  $K_\phi$ and $m_{ab}$ from `reference_element(op)`; $M_e = m \otimes I_3$ (consistent) or $\mathrm{diag}(\textstyle\sum_b m_{ab}) \otimes I_3$ (lumped), matching `op.mass`.
- **Nodal terms.** $d = \sigma\,n_\text{nodal}\,m + s(1 - m)$ per fine DOF (nodal masses and the constrained diagonal of $A_P$). Each fine node's $d$ is added once, as $p^\top \mathrm{diag}(d_n) p$ with $p$ its prolongation row, into the cell matrix of the coarse parent of the first fine element (lowest id) containing it.
- **Levels ≥ 2:** $C'_{c'} = \sum_{c \in c'} P_c^\top C_c P_c$, unmasked.
- **Chunking.** The level-1 sum runs as a `lax.fori_loop` over chunks of `chunk` fine cells, each gathering its $A_e$ (phase-indexed $K_\phi$ times scalars, never a stored $(n_\text{el}, 24, 24)$), forming $\tilde P_e^\top A_e \tilde P_e$ and scatter-adding into $C$. The peak is the level-1 cells ($n_\text{el}/8 \cdot 576$ values, 0.4 GB at 2M DOF) plus one chunk. The coarse-level sums are small and run unchunked.
- **Storage.** Each level's operator is a `BlockELL` (PR 10) with 27 slots per node (slot = corner offset difference, missing neighbours point at the node itself with zero blocks), built directly from the cell matrices by one scatter-add of their $8\times8$ node blocks; 3×3 blocks, fp64. The fine level stays the matrix-free `ap_operator`.

### Transfers

Prolongation runs matrix-free per level: scatter the coarse nodal values into the full coarse grid (`.at[gc].set(..., unique_indices=True)`), apply the three 1D two-point interpolations (gather of the two coarse neighbours times the weights, axis by axis), gather the active fine nodes. Restriction is its exact transpose, `jax.linear_transpose` of the prolongation (`unique_indices=True` makes the scatter transposable to a gather). The full-grid temporaries are ≤ 8 coarse grids (one fine grid) of `(…, 3, k)` values per level.

### Smoothers: stopped-axis group blocks

Point (node-block) smoothing does not damp the error modes that semi-coarsening leaves along stopped axes: the $64^2\times1$ plate needs 240 iterations with node-block Jacobi. 11a smooths with **group blocks**:

- A group is the set of nodes sharing coordinates on the coarsening axes; its size is $k = \prod_{d\,\text{stopped}} (n_d + 1) \le 27$ nodes ($3k \le 81$ DOF). With no stopped axis, groups are single nodes ($k = 1$, node-block Jacobi).
- The group blocks are the $3k \times 3k$ diagonal blocks of the level operator, summed from the cell matrices (level ≥ 1) or from the masked fine element matrices $\mathrm{diag}(m_e) A_e \mathrm{diag}(m_e)$ plus $d$ on the diagonal (level 0; chunked like the level-1 sum, and $k = 1$ uses `block_jacobi(op, free, sigma)` directly).
- They are inverted through PR 10's `_inverse_blocks` (batched Cholesky, absent slots of partial groups masked to identity), and applied as gather → batched $3k\times3k$ GEMM → scatter-add.
- Each level's smoother is `chebyshev(A_l, group_l, lmax_estimate(A_l, group_l), degree, ratio)` (PR 10), symmetric.

With it the plates converge in 25–38 iterations (h-independent from $64^2$) and the beams in 22–35.

### V-cycle

On level $l$ with operator $A_l$, smoother $S_l$ and transfer $P_l$:
$$
x = S_l b,\quad x \mathrel{+}= P_l\,V_{l+1}(P_l^\top (b - A_l x)),\quad x \mathrel{+}= S_l(b - A_l x),
$$
and on the coarsest level $x = L^{-\top}L^{-1} b$ with $L$ the dense fp64 Cholesky of $A_\text{coarsest}$ (densified by applying the ELL to the identity). Pre- and post-smoothing use the same symmetric $S_l$ and the coarse solve is exact, so the V-cycle is symmetric; it is positive definite when $\|I - S_l A_l\|_{A_l} < 1$, which Chebyshev on $[\hat\lambda/30, \hat\lambda]$ with $\hat\lambda \ge \lambda_\max(S^{-1}A)$ gives. The recursion is unrolled at trace time (the level count is static).

### Limitations (documented)

- **Anisotropic spacing.** Coarsening is by element count, not spacing: $h_z = h/10$ needs 151 iterations. Spacing-aware semi-coarsening needs line smoothers along long stopped axes; not in the roadmap, noted in Not covered.
- **High contrast.** Binary densities with sharp interfaces grow with $n$ (above); smooth and filtered fields are fine at the tested sizes.

## Files

| file | change |
|---|---|
| `src/jaxmodal/precond/gmg.py` | new: hierarchy, cell matrices, ELL levels, transfers, group smoother, `GMG`, `gmg` |
| `src/jaxmodal/precond/__init__.py` | export `gmg`, `GMG` |
| `tests/test_gmg.py` | new |
| `tests/data/simp32.npy` | new: the filtered binary SIMP density, $32^3$, float32 (128 KB) |
| `benchmarks/gmg_study.py`, `benchmarks/results/gmg_study.json` | new: the iteration table above, re-measured with the shipped code |
| `docs/preconditioners.md`, `README.md`, `docs/ROADMAP.md` | GMG section, status |

## Tests (`tests/test_gmg.py`)

All on steel, x64, PCG tol $10^{-8}$ with 4 seeded right-hand sides (`tests/helpers.rhs`), σ from `sigma_estimate` (fixed `v0`) unless noted.

1. **Galerkin levels vs SciPy** (roadmap). $5\times4\times3$ occupancy-masked mesh (`xmin` fully active, clamped), two phases, random `s_K`, `s_M`, nodal masses on half the nodes, consistent and lumped, σ = 0.37 (unit material scale): densify each transfer ($P_l$ on the identity) and level ($A_l$ on the identity); assert $A_{l+1} = P_l^\top A_l P_l$ with $A_0$ = PR 4's `a_p(K + σM, free)` to 1e-12 relative; the $x$ axis (5) is odd. A second mesh $7\times3\times2$ checks a stopped axis from the start.
2. **Transfers.** $P$ reproduces a trilinear field exactly (1e-12) on odd and even axes; restriction equals $P^\top$ densified (1e-14); dropped coarse nodes: on a mesh with an inactive coarse cell, the full-grid $P$ has zero columns exactly at the grid nodes not kept.
3. **Symmetry and definiteness.** $u^\top V v = v^\top V u$ to 1e-12 relative and $u^\top V u > 0$ for random $u, v$ on a clamped and on a free box; a constrained component of the output is exactly 0.
4. **h-independence** (roadmap). Boxes $8^3$ … $64^3$ (the $64^3$ case `slow`): iterations grow ≤ 20% from $8^3$ (measured 14 throughout).
5. **SIMP field** (roadmap). `simp32.npy` ($s_K = \rho^3$, $s_M = \rho$): ≤ 100 iterations (measured 55); the test also checks the stored field's statistics (min, max, mean) so a regenerated file is noticed.
6. **Single layer** (roadmap). Plate $64^2\times1$, clamped `xmin`: ≤ 100 iterations (measured 38).
7. **Free-free** (roadmap). $32^3$ free box, σ > 0: ≤ 100 iterations (measured 13); the coarsest Cholesky factor is finite.
8. **Beam and odd axes.** $128\times1\times1$ ≤ 100 (35); $33\times8\times8$ ≤ 30 (17).
9. **LOBPCG.** `lobpcg(op, free, 6, tol=1e-8, precond=gmg(...))` on a $9\times6\times4$ clamped occupancy-masked box matches `reference_modes` with the PR 9 tolerances, in fewer iterations than with Jacobi.
10. **No recompile.** Two `gmg` builds on the same grid with different `s_K` reuse one compiled `pcg` (cache size unchanged).
11. **Errors.** Unstructured operator → `TypeError`; σ < 0 → `ValueError`; x64 off → error.

Iteration bounds were measured with the prototype; implementation re-measures them with the shipped code and keeps ≥ 1.5× margin (or adjusts the bound to the roadmap's).

## Docs

- `docs/preconditioners.md`: a GMG section (coarsening rule with the odd-axis cell, Galerkin cell matrices with the LaTeX above, the dropped coarse nodes, transfers, group smoother and why, V-cycle, limitations, measured table).
- `README.md`: GMG in the feature list with a short `pcg(..., precond=gmg(...))` snippet (run by `test_readme`).
- `docs/ROADMAP.md`: the PR 11 split, 11a merged note, the deviation (dropped coarse DOFs), the group smoother.

## Not covered

- 11b: `solve_modes` dispatch, σ hint, fp32 V-cycle, memory estimator, Mac acceptance.
- Spacing-aware coarsening and line smoothers; robustness to unfiltered high-contrast densities (PR 23 re-runs the contrast test).
- W-cycles and smoother degree tuning (degree 2 fixed default, an argument).
