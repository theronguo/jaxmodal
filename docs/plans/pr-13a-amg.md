# PR 13a plan: AMG on the hex8 level (core)

PR 13 (M3) depends on PRs 9b, 9c, 10 and 12. It is split like PR 11.

**13a (this plan): the AMG core.**
- Smoothed-aggregation setup on the hex8 Galerkin $A_P$ from PR 12, with the masked rigid modes as the near-null space.
- The hierarchy exported to the device as one flat V-cycle, combined with PR 12's p-level.
- Pure hex8 meshes skip the p-level.
- Zero tentative columns are dropped.

**13b: everything around the core.**
- The production chunked hex8 assembler into a precomputed CSR pattern (it replaces the `_hex8_galerkin` call for the all-hex8 path at scale; see Performance).
- Aggregate reuse.
- The `solve_modes` dispatch and σ pre-estimate for unstructured meshes (including σ = 0 when no rigid modes remain).
- The memory estimator and the memory ordering (setup → free the assembled A → allocate LOBPCG).
- The setup benchmarks, the 1M-DOF hex20 Mac acceptance, the optional 10M benchmark and the deferred GPU check.

**Roadmap tests.**
- In 13a:
  - aggregates straddling the Dirichlet boundary;
  - hex20 h-independence over 3 refinements.
- In 13b:
  - the `solve_modes` tests (hex20 and pure hex8 against `eigsh`);
  - the reuse test;
  - the acceptance runs.

**Deviations from the roadmap wording** (recorded in the ROADMAP split note at the 13a merge):
- The roadmap says "pyamg smoothed-aggregation setup". 13a runs its own short loop over pyamg's primitives instead of `smoothed_aggregation_solver`. `fit_candidates` zeroes small tentative columns but keeps them, and pyamg's solver hides the resulting zero rows with a pseudo-inverse coarse solve, whereas the roadmap requires the columns dropped so that a dense Cholesky works.
- The first-level prolongator is still exactly pyamg's (test 2).

## Measured facts (prototypes `scratchpad/p13/`; reviewer probes `scratchpad/r13/`, `c13_corr1/`, `rev13a_perf/`, `r13a-p1-tests/`; jax 0.11.2, pyamg 5.3.0, fp64, M3 Max CPU)

Setup unless noted:
- steel, unit box, `xmin` clamped;
- σ = $\lambda_1$ of the $6^3$ box of the same type (SciPy), fixed over each h-sequence;
- PCG on $A_P$ to $10^{-8}$ with 4 random right-hand sides;
- Chebyshev degree 2, ratio 30, on every level.

The setup loop is the one in Design (node-block strength on the first AMG level, `standard_aggregation`, `fit_candidates`, zero-column drop, Jacobi-smoothed prolongator with $\omega = 4/3$). The device format is the one in Design (block-ELL levels, rectangular ELL transfers).

**Iterations.**

| case | DOF | AMG level sizes | PCG iterations |
|---|---|---|---|
| hex20 $3^3$ / $6^3$ / $12^3$ / $24^3$, p-MG + AMG, `max_coarse` = 100 | 624 – 182k | 192→48 / 1029→162→6 / 6591→750→48 / 46875→4374→162→6 | 15 / 14 / 14 / 14 (exact `splu` coarse: 14 / 14 / 14 / 14) |
| hex20 $44^3$, p-MG + AMG, `max_coarse` = 1000 | 1.08M | 273375→20250→750 | 14 |
| hex8 $6^3$ / $12^3$ / $24^3$ / $48^3$, AMG under the matrix-free $A_P$ | 1k – 353k | to 108 / 600 / 162 / 48 | 13 / 13 / 13 / 13 |
| AMG alone on the hex8 Galerkin matrix of hex20 $6^3$–$24^3$ | 1k – 47k | | 13 |

**Robustness, exact coarse against AMG.** The AMG counts were measured with pyamg's host cycle using block Gauss–Seidel; on the boxes the device Chebyshev cycle gave the same counts. The shipped device cycle re-measures this table into `amg_study.json`, and the docs quote only those numbers.

| case | `splu` | AMG |
|---|---|---|
| hex27 $6^3$ | 13 | 13 |
| beam $20\times2\times2$ | 16 | 17 |
| free, curved, nodal | 14 | 14 |
| checkerboard `s_K` $10^{-3}$ | 19 | 21 |
| plate $16^2\times1$, $a$ = 1 / 2 / 4 / 10 | 15 / 21 / 63 / 694 | 20 / 50 / 178 / > 1000 |
| box $8^3$, $h_z = h/a$, $a$ = 1 / 2 / 4 / 10 | 14 / 17 / 28 / 76 | 14 / 17 / 37 / 152 |

**Cost at hex20 $44^3$** ($m = 25$; performance reviewer's emulation of the planned design, which the shipped code re-measures):
- $t_A$ = 311 ms.
- The AMG cycle costs 1.20 $t_A$. Each level-0 transfer apply ($P$ or $P^\top$) costs about 0.18 $t_A$.
- The whole V-cycle costs 5.4 $t_A$ (PR 12: 4.2 $t_A$ without the coarse solve at $24^3$).
- V-cycle temporaries: 12.1 fine blocks ($A_P$ apply alone 3.1; PR 12 with exact coarse 9.3–10.3).
- Setup: Galerkin 1.2 s, SA loop 2.0 s, device export 1.4 s.
- **Device storage of the AMG levels and transfers: 0.50 GB**, int32 columns included.

| level | $A$ (ELL) | $P$ | $P^\top$ |
|---|---|---|---|
| hex8 Galerkin (273k) | 189 MB (padding 1.05) | 7.2M nonzeros, 26.3 per row (max 48), 118 MB, padding 1.34 | 355 per row (max 429), 129 MB, padding 1.36 |
| next (20k) | 47 MB | 11 MB | 8 MB |

- Operator complexity is 1.15–1.20.
- 13b consequence: the ROADMAP memory table's "AMG ≈ 0.4 GB" at 1M hex20 becomes about 0.5 GB, about 5 GB at 10M. 13b updates the table with its estimator.

**Zero tentative columns.**
- They occur only with rank-deficient constraints. With a seeded random DOF mask, 30% free, on a $6^3$ box: 10 columns (hex20) and 4 (hex8) on the first AMG level.
- Without the drop:
  - with two levels, the coarsest Cholesky is not finite;
  - with more levels, an intermediate level's Jacobi divides by zero and PCG fails at once.
- With the drop, everything is finite and PCG converges in 12 iterations.
- Clamped and sliding boxes give none: constrained nodes of the pure hex8 $A_P$ have no off-diagonal entries once explicit zeros are removed, so they stay unaggregated.

**The constrained corners of the p-MG coarse matrix are coarse unknowns.**
- On the hex20 → hex8 level a free midside node interpolates from a constrained corner, so $P^\top A_P P$ couples constrained corner DOFs to free ones. The coarse problem must be solved on all coarse DOFs.
- Masking the constrained corners inside the AMG cycle broke it: standalone PCG stagnated at > 1000 iterations, while inside p-MG the outer smoother hid it.
- Those coarse DOFs carry energy of order $s$, not near-null energy, so masking their rows of $B$ is right.

**The setup is nondeterministic unless seeded.** `jacobi_prolongation_smoother` scales $\omega$ by `approximate_spectral_radius`, which starts from the global `np.random`. Rebuilding $P$ on the same matrix gave a $8.6\cdot10^{-4}$ relative difference; with a fixed seed the difference is 0.

**First-level prolongator against pyamg.** On hex20 $12^3$ (clamped, no drops) the loop's first-level $P$ equals `smoothed_aggregation_solver(bsr(A_z, 3), B, strength=("symmetric", {"theta": 0}), smooth=("jacobi", {"omega": 4/3, "weighting": "block"}), improve_candidates=None)`'s up to the spectral-radius randomness ($3\cdot10^{-4}$); `weighting="diagonal"` differs by $9.4\cdot10^{-2}$.

**Constrained rows of $P_0$ (all-hex8).** These are exactly 0 on hex8 $6^3$–$24^3$, including with a sliding mask:
- a fully constrained node is isolated, so it stays unaggregated and its tentative row is empty;
- a partially constrained node's constrained rows of $B$ are masked, and the node-block Jacobi row of a constrained DOF is diagonal, so smoothing keeps them zero.

With unmasked $B$ those rows are nonzero (0.19 on hex8 $8^3$ with sliding faces), with no change in the counts.

## Design

### Entry: `amg(op, free, sigma, points, *, degree=2, ratio=30.0, max_coarse=1000)`

A new module `precond/amg.py`, exported from `jaxmodal.precond`. It returns a `GMG`, PR 11a's shared V-cycle, as one flat hierarchy:
- **hex20/hex27 or mixed meshes:** level 0 is PR 12's p-level (the matrix-free $A_P$ with Chebyshev over node-block Jacobi, and the `NodalTransfer`); level 1 is the hex8 Galerkin matrix.
- **All-hex8 meshes:** $P = I$, so there is no p-level. Level 0 is the matrix-free $A_P$ with Chebyshev over `block_jacobi(op)`, and its transfer is the first AMG prolongator, set up on the assembled $A_P$ (`_hex8_galerkin` with $P = I$; 13b swaps in the production assembler here).
- **Below that:** the AMG levels, then a dense fp64 `Cholesky` of the coarsest matrix.

One `_vcycle` recursion covers all levels, with no nested `GMG` and no inner masking. The output is masked by `free`.

**When no AMG level is needed** (hex8 level ≤ `max_coarse`):
- hex20/hex27 give one level (the p-level with its `NodalTransfer`), and `coarse` is the Cholesky of the hex8 Galerkin matrix (two-level p-MG with an exact coarse solve);
- all-hex8 gives `levels = ()`, `transfers = ()` and `coarse` = the Cholesky of the assembled $A_P$, i.e. a direct solve (`_vcycle` at level 0 calls `coarse` directly).

**Shared with PR 12.**
- A private `_p_level(op, free, sigma, degree, ratio) -> ((A, S), T, Ac, corner)` in `pmg.py` builds `ap_operator`, `block_jacobi`, the Chebyshev smoother and `_hex8_galerkin`. `pmg` adds `HostLU(splu(Ac))`; `amg` adds the AMG levels. The all-hex8 branch uses the same `(A, S)` and drops `T`.
- `_hex8_galerkin` also returns the corner node ids (`corner`, the coarse numbering of `_tables`); it is private, so the signature can change. All-hex8 gives `corner = arange(n_nodes)`.

Arguments:
- **`op`:** an `Unstructured` operator (`TypeError` otherwise, naming `gmg` for stencils).
- **`points`:** the mesh's `(n_nodes, 3)` coordinates, **in the operator's node order** (for a renumbered mesh, the renumbered mesh's `points`). The near-null space is the six rigid modes built from them; a later `B=` keyword would be additive. A wrong shape raises `ValueError`; a wrong order cannot be detected and is documented.
- **`sigma`:** concrete and > 0, as `pmg`. σ == 0 raises `SingularShiftError`; σ < 0 or NaN raises `ValueError`.
- **`degree`, `ratio`:** Chebyshev degree and ratio on every level.
- **`max_coarse`:** the size at which coarsening stops.

`pmg` keeps its PR 12 signature and test-only `splu` coarse solve. `amg` is the production preconditioner for unstructured meshes.

### Near-null space (`_nullspace(x, free_c)`)

$B$ is $(3 n_c, 6)$: the translations and the rotations $e_j \times (x - \bar x)$ about the centroid of `x = points[corner]`, with rows of constrained corner DOFs zeroed (masked rigid modes, roadmap).
- On straight elements it spans the corner restriction of `rigid_modes(mesh)` (test 1).
- On curved elements the rotations are only approximately in the hex8 coarse space. PR 12's open item asked PR 13 to restrict the fine rigid modes instead. Decision, taken without user input: keep corner-evaluated $B$. The curved case's count equals the straight one's (14 = 14), and restriction would need a fine rigid-mode build plus a least-squares fit per corner.

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
    if len(keep) > 0.9 * A.shape[0]: break                              # no progress
    Ac = P^T Az P (SciPy CSR)
    levels.append((A, P)); A, B, b = Ac, R, 1
if A.shape[0] > 4 * max_coarse: raise ValueError(...)                   # coarsest cap
return levels, A
```

- **Determinism.** The loop runs under a locally fixed NumPy seed (`get_state` / `seed(0)` / `set_state` around the pyamg calls), so the hierarchy is reproducible and the caller's global RNG is left untouched.
- **Block size.** The first AMG level aggregates whole nodes (3×3 blocks, pyamg's elasticity setting) with block Jacobi weighting. Coarse levels are scalar: after the drop an aggregate may keep fewer than 6 columns, so coarse DOFs are not uniformly blocked.
- **Zero-column drop** (roadmap). A tentative column with zero norm comes from an aggregate whose $B$ rows are zero or rank-deficient after the QR. Dropping it keeps every coarse matrix SPD and lets the coarsest dense Cholesky succeed.
- **Stopping and coarsest cap.**
  - Coarsening stops at `max_coarse`, or when a level would keep more than 0.9× its size.
  - If the coarsest is then larger than 4·`max_coarse`, `amg` raises `ValueError` ("AMG coarsening stalled at n DOF ...") instead of factoring a huge dense matrix (80 GB at 100k DOF).
  - A non-finite coarsest factor raises `ValueError`, as in `gmg`.

### Device export

- **First AMG level.**
  - hex20/hex27: `block_ell(Ac, b=3)` with node-block Jacobi from its 3×3 diagonal blocks (`_inverse_blocks`, all DOFs active).
  - All-hex8: this level is the matrix-free fine operator with `block_jacobi(op, free, σ)`.
- **Coarser levels:** `block_ell(A, b=1)` (sliced when padded) with scalar Jacobi.
- **Smoother:** `chebyshev(A, D, lmax_estimate(A, D), degree, ratio)` on every level.
- **Transfers.** $P_l$ and $P_l^\top$ are each stored as a rectangular scalar block-ELL, so prolongation and restriction are both gathers (no scatter-add; restriction equals $P^\top$ by test). `ELLTransfer(P, PT)` supplies `__call__` and `restrict`.

**`linalg/ell.py` extension** (square behaviour unchanged):
- `block_ell` accepts rectangular matrices (both dimensions divisible by `b`). `BlockELL.n_dof` is the output row count, i.e. the operator size for square matrices. Only transfers are rectangular, and they never reach code that assumes a square operator.
- Padding entries point to column 0, not to the row's own block column, which may be out of range for a rectangular matrix.
- Each bucket's scan step is bounded by entries, not rows: `c = clip(ENTRIES // (w b²), 8, chunk)` with `ENTRIES = 2048 · 27`, so the voxel GMG's 27-wide buckets keep today's chunk. Without this, a $P^\top$ bucket of width 512 gathers about one fine block per step at $m = 25$ (measured: 341 MB → 135 MB for the $44^3$ $P_0^\top$ apply at chunk 512, time 61 → 53 ms).

**What the pytree holds.** Device arrays only. The SciPy matrices and the pyamg intermediates are host locals and are freed when `amg` returns; 13b's memory ordering relies on this.

**Level masks.** No level needs one:
- the first AMG level solves all coarse DOFs (see "constrained corners");
- in the all-hex8 case the fine level masks the constrained DOFs, and their rows of $P_0$ are exactly 0 (see Measured facts).

The V-cycle is symmetric and positive definite on the free DOFs regardless: the same Chebyshev smoother before and after on each level, and an exact coarsest solve.

### Performance and 10M (13b)

No 13a choice blocks 13b's 10M hex8 case, but two pieces are 13b's:
- The all-hex8 path's `_hex8_galerkin` call would need about 15 GB of host cells plus about 30 GB of COO triples at 10M hex8 DOF. It is one internal call in `amg`, which 13b replaces with the production assembler.
- At 10M hex8 the scalar $P_0$ and $P_0^\top$ would be about 4 GB each. Blocking them is a 13b estimator and GPU item.

### Limitations (documented, `docs/open_items.md`)

- **Flat and thin elements.**
  - AMG on the hex8 coarse space makes PR 12's flat-element weakness worse: plate $a = 4$ goes 63 → 178, and $h_z = h/10$ goes 76 → 152.
  - The existing "Thin plates" item asked PR 13 for AMG on the high-order level or a bending-aware near-null space. Decision, taken without user input: 13a attempts neither. Both are a new setup design rather than a hex8-level AMG. The item is retargeted to M4 / PR 18 (solid-shell elements).
- **High-contrast SIMP (voxel GMG item, "Revisit: PR 13").** Decision: 13b's study runs `amg` on the checked-in SIMP field as an unstructured hex8 mesh and reports the count against GMG's, which answers the revisit. Any fix beyond that moves to PRs 21–23.

## Files

- `src/jaxmodal/precond/amg.py` (new): `amg`, `_sa_levels`, `_nullspace`, `ELLTransfer`.
- `src/jaxmodal/precond/pmg.py`: `_p_level`; `_hex8_galerkin` returns `corner`.
- `src/jaxmodal/precond/__init__.py`: export `amg`.
- `src/jaxmodal/linalg/ell.py`: rectangular `block_ell`, padding to column 0, entry-bounded chunks.
- `tests/test_amg.py` (new); `tests/test_ell.py` (rectangular cases); `tests/test_pmg.py` (the `_hex8_galerkin` return value).
- `benchmarks/amg_study.py` (new) → `benchmarks/results/amg_study.json`: h-sequences, robustness table, cost and storage at hex20 $44^3$.
- Docs as below.

## Tests (`tests/test_amg.py`)

Common setup as in `tests/test_pmg.py`:
- steel, x64;
- PCG to $10^{-8}$ with 4 right-hand sides;
- σ = $\lambda_1$ of the $6^3$ box (SciPy, fixed `v0`);
- unit materials for exactness tests.

1. **Setup loop against SciPy.** On hex20 $6^3$ (clamped plus `ymin: y` sliding) with `max_coarse` = 100, for every level of `_sa_levels`:
   - the coarse matrix equals $P^\top A_z P$ recomputed in the test to $10^{-12}$ and is symmetric;
   - $P$ has no zero column;
   - the device level applied to random blocks equals the SciPy product to $10^{-12}$;
   - the transfer's `__call__` and `restrict` equal $P$ and $P^\top$.

   Plus:
   - two calls give bitwise-identical hierarchies, and the global `np.random` state is unchanged;
   - `_nullspace` on a straight hex20 box spans the corner restriction of `rigid_modes(mesh)` (principal angles below $10^{-10}$), and rows of constrained DOFs are exactly 0.
2. **First-level prolongator against pyamg.** On hex20 $12^3$ and hex8 $12^3$ (clamped, no drops), the first-level $P$ equals the first-level $P$ of `smoothed_aggregation_solver` (settings as in Measured facts, same seed) to $10^{-12}$. This kills block-size, weighting, ω and strength mutants: a scalar-aggregation mutant gives the same sizes and counts, so only this test sees it.
3. **Zero-column drop and straddling aggregates** (roadmap).
   - A hex20 and a hex8 $6^3$ box with a seeded random mask, 30% free:
     - the first level drops at least one column (asserted > 0);
     - every coarse matrix is SPD (smallest eigenvalue > 0 on these small levels);
     - the coarsest Cholesky is finite;
     - PCG converges.
   - A clamped hex20 box ($6^3$, `xmin` plus `ymin: y`, `max_coarse` = 100):
     - at least one first-level aggregate contains both constrained and free corner nodes (the roadmap's premise);
     - setup succeeds, the coarsest Cholesky is finite, and PCG converges.
   - Reviewers' mutant: without the drop, the coarsest factor or the cycle is not finite (PCG fails).
4. **Dense V-cycle and symmetry.** Each case composes a NumPy V-cycle from the dense level matrices, transfers, test-built smoothers (seeded `lmax_estimate`) and the dense coarsest inverse, and compares it with `amg(...)(U)` to $10^{-10}$. Cases:
   - (a) hex20 $3\times2\times2$, `max_coarse` small enough for two AMG levels;
   - (b) hex8 $4^3$ with a sliding mask, two AMG levels; the constrained rows of $P_0$ are exactly 0;
   - (c) the no-AMG-level paths: hex20 $3^3$ at the default `max_coarse` (p-level plus Cholesky) and hex8 $3^3$ (`levels == ()`, a direct solve: PCG converges in 1 iteration).

   For (a) and (b): $u^\top V v = v^\top V u$ to $10^{-12}$; the free block of $V$ is positive definite; constrained outputs are exactly 0.
5. **h-independence** (roadmap). Iterations grow ≤ 20% from the first size to the last, at `max_coarse` = 100 so that every size has an AMG level:
   - hex20 $3^3$, $6^3$, $12^3$, $24^3$ ($24^3$ `slow`);
   - hex27 $3^3$, $6^3$, $12^3$;
   - hex8 $6^3$, $12^3$, $24^3$.
6. **Robustness bounds.** Each ≤ 1.5× the count the shipped device cycle re-measures: beam, free, curved, nodal masses, checkerboard `s_K`.
7. **LOBPCG.** `lobpcg(op, free, 6, tol=1e-8, precond=amg(...))` on a clamped hex20 $6\times4\times3$ box and on a pure hex8 box matches `reference_modes` (eigenvalues to $10^{-8}$, residuals ≤ $10^{-6}$) in ≤ 40 iterations.
8. **Errors, cap and leaves.**
   - A `Stencil` raises `TypeError`; a wrong `points` shape raises `ValueError`.
   - σ = 0 raises `SingularShiftError`; σ < 0 and NaN raise `ValueError`.
   - x64 off raises.
   - A monkeypatched `standard_aggregation` returning singletons triggers the coarsest-cap `ValueError`.
   - Every leaf of the result is a `jax.Array`.
   - `type(amg(...)) is GMG`, and `amg` and `pmg` apply through the same `gmg._vcycle_apply`.

`tests/test_ell.py`:
- rectangular `block_ell` ($b = 1$ and $b = 3$, plain and sliced, tall and wide) equals SciPy to $10^{-12}$, with the output shape asserted;
- padding columns are 0;
- a wide-bucket rectangular apply's `memory_analysis` temporaries stay below one entry-bounded chunk's gather;
- square cases unchanged.

The implementation re-measures all iteration bounds with the shipped code and keeps at least a 1.5× margin.

## Docs

- **`docs/preconditioners.md`**:
  - an AMG section covering:
    - `amg` and its hierarchy (p-level or matrix-free fine level, AMG levels, Cholesky; the no-AMG-level case);
    - the near-null space and `points` order;
    - the setup loop with the zero-column drop, determinism and the coarsest cap;
    - why the constrained corners are coarse unknowns and why the constrained rows of $P_0$ vanish;
    - the device format (rectangular ELL transfers, gathers only);
    - the measured tables from `amg_study.json` (device cycle only);
    - the cost breakdown in $t_A$ (transfers about 0.37 $t_A$ of the AMG cycle's 1.2);
    - storage;
    - limitations;
    - the tests list.
  - `pmg` stays documented as the test-only exact-coarse variant.
  - The block-ELL section: rectangular matrices, padding to column 0, entry-bounded chunks.
- **`README.md`**: status, plus an `amg` snippet (run by `test_readme`).
- **`docs/ROADMAP.md`**:
  - the PR 13 split and deviation note (own setup loop over pyamg primitives, and why);
  - the 13a status line at merge;
  - a 13b note that the memory table's AMG row becomes about 0.5 GB at 1M hex20.
- **`docs/open_items.md`** (decisions taken without user input marked as such):
  - "Thin plates": the AMG numbers; 13a attempts neither fix; retargeted to M4 / PR 18.
  - "p-MG on curved elements": resolved, with corner-evaluated $B$ (curved count unchanged).
  - "Voxel GMG on high-contrast SIMP fields": 13b's study runs `amg` on the field; fixes beyond that move to PRs 21–23.
  - "p-MG refuses σ = 0": retargeted to 13b.
  - `pmg` (exact coarse) kept alongside `amg`.
  - Masked versus unmasked $B$: no measured difference in counts; unmasked $B$ makes the constrained rows of $P_0$ nonzero; masked per roadmap.

## Not covered

13b covers:
- the production assembler;
- aggregate reuse;
- `solve_modes` with the σ pre-estimate, including σ = 0;
- `amg_memory` and `solve_modes_memory`;
- the memory ordering;
- the setup benchmarks;
- the SIMP study;
- the Mac acceptance and the optional 10M benchmark;
- the deferred GPU check.

Not planned: an fp32 AMG cycle (no roadmap item).
