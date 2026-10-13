# PR 13a plan: AMG on the hex8 level (core)

PR 13 (M3) depends on PRs 9b, 9c, 10 and 12. It is split like PR 11:
- **13a (this plan): the AMG core.**
  - Smoothed-aggregation setup on the hex8 Galerkin $A_P$ from PR 12, with the masked rigid modes as the near-null space.
  - The hierarchy exported to the device as one flat V-cycle, combined with PR 12's p-level.
  - Pure hex8 meshes skip the p-level.
  - Zero tentative columns are dropped.
- **13b: everything around the core.**
  - The production chunked hex8 assembler into a precomputed CSR pattern.
  - Aggregate reuse.
  - The `solve_modes` dispatch and σ pre-estimate for unstructured meshes.
  - The memory estimator and the memory ordering (setup → free the assembled A → allocate LOBPCG).
  - The setup benchmarks, the 1M-DOF hex20 Mac acceptance, the optional 10M benchmark and the deferred GPU check.

Roadmap tests in 13a: aggregates straddling the Dirichlet boundary, and hex20 h-independence over 3 refinements. The `solve_modes` tests (hex20 and pure hex8 against `eigsh`), the reuse test and the acceptance runs are 13b.

## Measured facts (prototypes `scratchpad/p13/`; jax 0.11.2, pyamg 5.3.0, fp64, M3 Max CPU)

Setup unless noted:
- steel, unit box, `xmin` clamped;
- σ = $\lambda_1$ of the $6^3$ box of the same type (SciPy), fixed over each h-sequence;
- PCG on $A_P$ to $10^{-8}$ with 4 random right-hand sides;
- Chebyshev degree 2, ratio 30, on every level.

The setup loop is the one in Design:
- strength from 3×3 node blocks on the first AMG level;
- `standard_aggregation`, then `fit_candidates`;
- zero columns dropped;
- Jacobi-smoothed prolongator ($\omega = 4/3$);
- `max_coarse` = 1000.

**Iterations.**

| case | DOF | levels below the fine | PCG iterations |
|---|---|---|---|
| hex20 boxes $6^3$ / $12^3$ / $24^3$, p-MG + AMG | 3.7k – 182k | 2 / 2 / 3 | 14 / 14 / 14 (exact `splu` coarse: 14 / 14 / 14) |
| hex20 $44^3$, p-MG + AMG | 1.08M | 3 (coarse 273k, 20k, 750) | 14 |
| hex8 boxes $6^3$ / $12^3$ / $24^3$ / $48^3$, AMG on the matrix-free $A_P$ | 1k – 353k | 1 / 1 / 2 / 3 | 13 / 13 / 13 / 13 |
| AMG alone on the hex8 Galerkin matrix of hex20 $6^3$–$24^3$ | 1k – 47k | | 13 |

Robustness, p-MG with exact coarse against p-MG + AMG (pyamg's host cycle with block Gauss–Seidel; the device Chebyshev cycle gave the same counts on the boxes):

| case | `splu` | AMG |
|---|---|---|
| hex27 $6^3$ | 13 | 13 |
| beam $20\times2\times2$ | 16 | 17 |
| free, curved, nodal | 14 | 14 |
| checkerboard `s_K` $10^{-3}$ | 19 | 21 |
| plate $16^2\times1$, $a$ = 1 / 2 / 4 / 10 | 15 / 21 / 63 / 694 | 20 / 50 / 178 / > 1000 |
| box $8^3$, $h_z = h/a$, $a$ = 1 / 2 / 4 / 10 | 14 / 17 / 28 / 76 | 14 / 17 / 37 / 152 |

The implementation re-measures the robustness table with the device cycle.

**Cost at hex20 $44^3$** ($m = 25$):
- $t_A$ = 314 ms.
- The AMG cycle on the 273k-DOF hex8 level costs 0.97 $t_A$; the whole V-cycle 5.2 $t_A$ (PR 12 measured 4.0 $t_A$ without the coarse solve at $24^3$).
- Setup: Galerkin 1.3 s, SA 2.2 s, device export 1.2 s.
- Device storage of the AMG levels and transfers: about 0.25 GB. On the hex8 $48^3$ box it is 0.30 GB: levels, $P$ and $P^\top$, including padding.
- Operator complexity 1.15–1.20; smoothed $P$ has 23–26 nonzeros per row on the first AMG level.

**Zero tentative columns.**
- None occur on clamped or sliding boxes. Constrained nodes of the pure hex8 $A_P$ have no off-diagonal entries once the explicit zeros are removed, so they stay unaggregated.
- With a random DOF mask of 30% free DOFs on a $6^3$ box they occur: 10 columns (hex20) and 4 (hex8) on the first AMG level. Without the drop, the coarse matrices have zero rows; with it, the coarsest Cholesky succeeds.
- pyamg's `fit_candidates` zeroes columns below its tolerance but keeps them. Its own solver then hides the zero rows with a pseudo-inverse coarse solve, which is why PR 13 runs its own setup loop.

**The constrained corners of the p-MG coarse matrix are coarse unknowns.** On the hex20 → hex8 level, a free midside node interpolates from a constrained corner. So $P^\top A_P P$ couples constrained corner DOFs to free ones, and the coarse problem must be solved on all coarse DOFs.
- Masking the constrained corners inside the AMG cycle broke it: standalone PCG on the hex8 matrix stagnated (> 1000 iterations).
- Inside p-MG the outer smoother hid this, so the counts looked right.
- Masking the near-null space $B$ (as the roadmap says) or not makes no difference to the counts (13 / 14 either way).

## Design

### Entry: `amg(op, free, sigma, points, *, degree=2, ratio=30.0, max_coarse=1000)`

A new module `precond/amg.py`, exported from `jaxmodal.precond`. It returns a `GMG`, PR 11a's shared V-cycle, as one flat hierarchy:
- **hex20/hex27 or mixed meshes:** level 0 is PR 12's p-level, i.e. the matrix-free $A_P$ with Chebyshev over node-block Jacobi, and the `NodalTransfer`. Level 1 is the hex8 Galerkin matrix from `_hex8_galerkin`, the first AMG level.
- **All-hex8 meshes:** $P = I$, so there is no p-level. Level 0 is the matrix-free $A_P$ with Chebyshev over `block_jacobi(op)`, and its transfer is the first AMG prolongator, set up on the assembled $A_P$ (`_hex8_galerkin`, which is the assembled hex8 $A_P$ when $P = I$).
- **Below that:** the AMG levels, then a dense fp64 `Cholesky` of the coarsest matrix.

Because the hierarchy is flat, one `_vcycle` recursion covers all levels, with no nested `GMG` and no inner masking. The output is masked by `free` as for every `GMG`.

Arguments:
- **`op`:** an `Unstructured` operator (`TypeError` otherwise, naming `gmg` for stencils).
- **`points`:** the mesh's `(n_nodes, 3)` coordinates (`mesh.points`). `Unstructured` carries no coordinates, and the near-null space needs them. A wrong shape raises `ValueError`.
- **`sigma`:** concrete and > 0, as `pmg`. σ == 0 raises `SingularShiftError`; σ < 0 or NaN raises `ValueError`. Accepting σ = 0 when no rigid modes remain is the 13b σ pre-estimate's business (open item from PR 12).
- **`degree`, `ratio`:** Chebyshev degree and ratio on every level.
- **`max_coarse`:** the size at which coarsening stops.

`pmg` keeps its PR 12 signature and its test-only `splu` coarse solve, with no API change. `amg` is the production preconditioner for unstructured meshes; 13b's `solve_modes` dispatches to it.

### Near-null space

$B$ is $(3 n_c, 6)$ over the corner nodes in coarse numbering: the three translations and the three rotations $e_j \times (x - \bar x)$ about the corner centroid, with $x$ from `points`. Rows of constrained DOFs are zeroed ("masked rigid modes", roadmap; it does not change the counts above).
- The corner DOFs' free flags are `free` restricted to the corner nodes.
- On curved elements the rotations are only approximately in the coarse space (PR 12 open item); the curved case's count above (14) shows no effect.

### Host setup loop (`_sa_levels(A, B, max_coarse)`, about 30 lines, pyamg primitives)

```
levels = []; b = 3
while A.shape[0] > max_coarse:
    Az = A with explicit zeros removed (a copy; A keeps its structural pattern)
    C = symmetric_strength_of_connection(bsr(Az, b), theta=0)   # node blocks on the first level
    Agg = standard_aggregation(C)
    T, R = fit_candidates(Agg, B)
    keep = columns of T with nonzero norm; T, R = T[:, keep], R[keep]   # zero-column drop
    P = jacobi_prolongation_smoother(bsr(Az, b), T, C, R, omega=4/3, weighting="block" if b == 3 else "diagonal")
    Ac = P^T Az P (SciPy CSR)
    levels.append((A, P)); A, B, b = Ac, R, 1
return levels, A
```

- **Block size.** The first AMG level aggregates whole nodes (3×3 blocks, as pyamg's elasticity default). Coarse levels are scalar. After the drop, an aggregate may keep fewer than 6 columns, so coarse DOFs are not uniformly blocked.
- **Zero-column drop** (roadmap). A tentative column with zero norm comes from an aggregate whose $B$ rows are all zero or rank-deficient after the QR. It would give a zero row and column in every coarser matrix. Dropping it keeps each coarse matrix SPD, as the roadmap requires, and lets the coarsest dense Cholesky succeed.
- **Stopping.**
  - Coarsening stops at `max_coarse` (1000; the coarsest is 48–750 in the table above).
  - It also stops when a level fails to shrink the problem below 0.9× its size, so a pathological strength graph cannot loop.
  - The coarsest is factored densely in fp64; a non-finite factor raises `ValueError`, as in `gmg`.

### Device export

- **First AMG level.**
  - For hex20/hex27 meshes this is the hex8 Galerkin matrix: `block_ell(A, b=3)` with node-block Jacobi from its 3×3 diagonal blocks (`_inverse_blocks`, all DOFs active).
  - For all-hex8 meshes this level is the matrix-free fine operator itself; its smoother is `block_jacobi(op, free, σ)`.
- **Coarser levels:** `block_ell(A, b=1)` (sliced when padded) with scalar Jacobi.
- **Smoother:** `chebyshev(A, D, lmax_estimate(A, D), degree, ratio)` on every level.
- **Transfers.** $P_l$ and $P_l^\top$ are each stored as a rectangular scalar block-ELL. Prolongation and restriction are both gathers: no scatter-add, and restriction equals $P^\top$ by test.
  - This needs one extension to `linalg/ell.py`: `block_ell` accepts rectangular matrices (both dimensions divisible by `b`), and `BlockELL.n_dof` becomes the output row count. The reshape in `_apply` uses it instead of the input's row count. Square behaviour is unchanged.
  - A small `ELLTransfer(P, PT)` pytree supplies `__call__` and `restrict`.
- **What the pytree holds.** The returned pytree holds device arrays only. The SciPy matrices and the pyamg intermediates are host locals and are freed when `amg` returns. 13b's memory ordering relies on this.
- **Level masks.** No level needs one:
  - the first AMG level solves all coarse DOFs (see "constrained corners" above);
  - in the all-hex8 case the constrained DOFs are masked on the fine level, and their rows of $P_0$ are zero, because unaggregated isolated nodes have empty tentative rows.

### Limitations (documented, `docs/open_items.md`)

- **Flat and thin elements.** AMG on the hex8 coarse space makes PR 12's flat-element weakness worse (plate $a = 4$: 63 → 178; $h_z = h/10$: 76 → 152). This updates the existing "Thin plates" item; the fix needs bending-aware near-null spaces or solid-shell elements (M4).
- **Mesh-specific stagnation.** The setup is not robust to strength graphs from extreme contrasts. The checkerboard at $10^{-3}$ costs +2.

## Files

- `src/jaxmodal/precond/amg.py` (new): `amg`, `_sa_levels`, `_nullspace`, `ELLTransfer`.
- `src/jaxmodal/precond/__init__.py`: export `amg`.
- `src/jaxmodal/linalg/ell.py`: rectangular `block_ell`.
- `tests/test_amg.py` (new); `tests/test_ell.py`: rectangular cases.
- `benchmarks/amg_study.py` (new) → `benchmarks/results/amg_study.json`: h-sequences, robustness table, cost at hex20 $44^3$.
- Docs as below.

## Tests (`tests/test_amg.py`)

Common setup as in `tests/test_pmg.py`: steel, x64, PCG to $10^{-8}$ with 4 right-hand sides, σ = $\lambda_1$ of the $6^3$ box (SciPy, fixed `v0`), unit materials for exactness tests.

1. **Setup loop against SciPy.** On hex20 $6^3$ (clamped plus `ymin: y` sliding), for every level of `_sa_levels`:
   - the coarse matrix equals $P^\top A P$ recomputed in the test to $10^{-12}$ and is symmetric;
   - $P$ has no zero column;
   - the device level (`block_ell`) applied to random blocks equals the SciPy product to $10^{-12}$;
   - the transfer's `__call__` and `restrict` equal $P$ and $P^\top$.
2. **Zero-column drop** (roadmap: aggregates straddling the Dirichlet boundary).
   - A hex20 and a hex8 $6^3$ box with a seeded random mask, 30% free: the first level drops at least one column (the test asserts the count is > 0), the coarsest Cholesky is finite, and PCG converges.
   - Mutant check (reviewers): without the drop the coarsest factor is not finite.
   - A clamped hex20 box ($6^3$, `xmin` plus `ymin: y`) sets up, and the coarsest Cholesky succeeds.
3. **Dense V-cycle and symmetry.**
   - On a small hex20 box (with `max_coarse` small enough for two AMG levels), `amg(...)(U)` equals a NumPy V-cycle composed from the dense $A_l$, $P_l$, the test-built smoothers (seeded `lmax_estimate`) and the dense coarsest inverse, to $10^{-10}$.
   - $u^\top V v = v^\top V u$ to $10^{-12}$; the free block of $V$ is positive definite; constrained outputs are exactly 0.
4. **h-independence** (roadmap). Iterations grow ≤ 20% over 3 refinements:
   - hex20 $6^3$, $12^3$, $24^3$ ($24^3$ `slow`);
   - hex27 $3^3$, $6^3$, $12^3$;
   - hex8 $6^3$, $12^3$, $24^3$.
5. **Robustness bounds.** Each ≤ 1.5× the re-measured count: beam, free, curved, nodal masses, checkerboard `s_K`.
6. **LOBPCG.** `lobpcg(op, free, 6, tol=1e-8, precond=amg(...))` on a clamped hex20 $6\times4\times3$ box and on a pure hex8 box matches `reference_modes` (eigenvalues to $10^{-8}$, residuals ≤ $10^{-6}$) in ≤ 40 iterations.
7. **Errors and leaves.**
   - A `Stencil` raises `TypeError`; a wrong `points` shape raises `ValueError`.
   - σ = 0 raises `SingularShiftError`; σ < 0 and NaN raise `ValueError`.
   - x64 off raises.
   - Every leaf of the result is a `jax.Array` (no host objects held).
   - `type(amg(...)) is GMG`.

`tests/test_ell.py`: rectangular `block_ell` ($b = 1$ and $b = 3$, plain and sliced) equals SciPy to $10^{-12}$; square cases unchanged.

The implementation re-measures all iteration bounds and keeps at least a 1.5× margin.

## Docs

- **`docs/preconditioners.md`**: an AMG section covering:
  - `amg` and its hierarchy (p-level or matrix-free fine level, AMG levels, Cholesky);
  - the near-null space;
  - the setup loop with the zero-column drop;
  - why the constrained corners are coarse unknowns;
  - the device format (rectangular ELL transfers, gathers only);
  - the measured tables and the cost in $t_A$;
  - limitations;
  - the tests list.

  `pmg` stays documented as the test-only exact-coarse variant.
- **`docs/preconditioners.md`** (block-ELL section): rectangular matrices.
- **`README.md`**: status, plus an `amg` snippet (run by `test_readme`).
- **`docs/ROADMAP.md`**: the PR 13 split note and the 13a status line at merge.
- **`docs/open_items.md`**:
  - the flat-element item with the AMG numbers;
  - the decision to keep `pmg` (exact coarse) alongside `amg`, without user input;
  - masked vs unmasked $B$ (no measured difference; masked per roadmap).

## Not covered

13b covers:
- the production assembler;
- aggregate reuse;
- `solve_modes` with the σ pre-estimate (and σ = 0 when no rigid modes remain);
- `amg_memory` and `solve_modes_memory`;
- the memory ordering;
- the setup benchmarks;
- the Mac acceptance and the optional 10M benchmark;
- the deferred GPU check.

Not planned: an fp32 AMG cycle (no roadmap item).
