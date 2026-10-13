# PR 13b plan: AMG in `solve_modes`, production hex8 assembler, reuse, estimator, Mac acceptance

Status: draft (plan review round 1).

PR 13b completes PR 13 (M3) on top of 13a's `amg(op, free, sigma, points)`. It makes AMG the default preconditioner of `solve_modes` on unstructured meshes and adds:
- the production hex8 Galerkin assembler (a device scatter into a precomputed node-block pattern; it replaces the COO path for every `amg` and `pmg` build);
- aggregate reuse;
- σ = 0 in `amg` when the caller has proven $A_P$ nonsingular, and the σ pre-estimate for unstructured meshes;
- `amg_memory` and its use in `solve_modes_memory`; the memory ordering;
- the setup benchmarks, the SIMP study, the 1M hex20 Mac acceptance, the optional 10M benchmark and the deferred GPU check.

Roadmap items in this PR:

| roadmap item | here |
|---|---|
| production hex8 $A_P$ assembler into a precomputed CSR pattern (≈ 1 GB per 1M hex8 DOF) | Design 1, test 1 |
| option to reuse aggregates (perturbed density: PCG within 20% of a fresh setup) | Design 2, test 2 |
| `solve_modes` dispatches unstructured meshes to AMG; pure hex8 runs AMG on the assembled $A_P$ | Design 4, tests 4–6 |
| memory ordering: setup → free the assembled A → allocate LOBPCG | Design 6, test 10 |
| memory estimator | Design 5, tests 7–8 |
| setup time and host-memory benchmarks (report-only) | Design 7, test 9 |
| `solve_modes` on hex20 and pure hex8 matches `eigsh` (PR 9 tolerances) | test 4 |
| Mac acceptance: 1M hex20, 20 modes, tol $10^{-6}$, ≤ 15 min, peak within 20% | test 10 |
| optional Mac 10M benchmark; deferred GPU check (10M on one GPU) | Design 8 (no test) |
| 13a carry-overs: σ = 0 (open item), SIMP study (open item), ROADMAP memory rows | Design 3, 7, 8 |

## Measured facts (prototypes in `scratchpad/p13b/`; jax 0.11.2, pyamg 5.3.0, fp64, M3 Max CPU)

Steel unit boxes, `xmin` clamped unless noted, as in `benchmarks/amg_study.py`.

**Production assembler** (`p13b/asm.py`, `peak.py`). The node-block pattern is $E^\top E$ of the element-to-corner incidence; each element's 64 node pairs get an int32 position in it; one jitted scan per block builds the 24×24 hex8 cells of a chunk (PR 12's `_cells` body) and scatter-adds their 3×3 blocks into a device `(nnzb, 3, 3)` array; a precomputed transpose permutation symmetrizes it exactly.
- Equals `_hex8_galerkin` to $3\cdot10^{-16}$ relative (hex20 $6^3$, hex8 $8^3$), same pattern (61 731 / 140 625 entries), exactly symmetric.
- Time and host peak (fresh subprocess, `harness.measure`):

  | | COO path (PR 12) | production | of which the pattern | device values |
  |---|---|---|---|---|
  | hex20 $44^3$ (1.08M DOF, 273k coarse) | 1.09 s, 2.47 GB | 0.53 s, 0.74 GB | 0.25 s | 0.17 GB |
  | hex8 $68^3$ (0.99M DOF) | 3.2 s, 8.98 GB | 1.1 s, 2.74 GB | 0.67 s | 0.62 GB |

- A first version that kept PR 12's separate `_cells` call materialized all cells of a block (1.4 GB at 1M hex8; peak 4.61 GB); the fused scan removes that.
- The roadmap's "≈ 1 GB per matrix per 1M hex8 DOF" is the 0.62 GB of values plus int32 indices (0.04 GB) and the host copy.
- pyamg needs int32 indices (`amg_core` refuses int64): the pattern is cast through `check_int32`.

**Where the SA setup time goes** (`p13b/saprof.py`, the 13a loop):

| | strength | aggregation | fit + drop | smoothing | Galerkin product | format conversions |
|---|---|---|---|---|---|---|
| hex20 $44^3$ | 0.01 s | 0.00 s | 0.02 s | 0.71 s | 1.14 s | 0.07 s |
| hex8 $68^3$ | 0.05 s | 0.00 s | 0.08 s | 2.59 s | 4.25 s | 0.22 s |

- Strength, aggregation and the tentative fit are under 2% of the loop. Reuse saves those plus the assembler pattern and PR 12's corner tables (`_tables`): about 0.6 s of a 4.5 s build at 1M hex20 (measured in the implementation).
- With θ = 0 the strength graph is the matrix pattern without explicit zeros, so aggregates depend only on the pattern, the mask and $B$: a reused setup reproduces a fresh one exactly on any values with the same zero pattern.
- Whole `amg` at 1M (13a code, COO path): hex20 5.6 s, 2.48 GB host peak (the Galerkin COO); hex8 15.1 s, 8.99 GB. The SA loop alone: 2.1 s / 1.06 GB and 7.8 s / 3.90 GB.

**σ pre-estimate with AMG** (`p13b/sig.py`; PR 11b's rule with `amg` in place of `gmg`, σ = 0 start when no rigid modes remain; λ₁ from SciPy):

| case | DOF | start σ/λ₁ | σ/λ₁ after each round | LOBPCG iterations |
|---|---|---|---|---|
| hex20 $8^3$ / hex8 $12^3$ / hex27 $6^3$ | 8k / 7k / 7k | 0 | 1.000 | 4 / 5 / 5 |
| beam $40\times2\times2$, plate $16^2\times1$ | 4k / 6k | 0 | 1.000 | 7 / 7 |
| plate $16^2\times1$, $a = 4$ | 6k | 0 | 1.001 | 20 |
| checkerboard `s_K` $10^{-3}$, nodal masses ($6^3$) | 4k | 0 | 1.000 | 7 / 5 |
| free hex20 $8^3$ | 8k | 5.5 | 1.000, 1.000 | 8 + 5 |
| free beam $40\times2\times2$ | 4k | 1578 | 1.000, 1.000 | 20 + 6 |
| free plate $16^2\times1$ | 6k | 712 | 1.001, 1.000 | 20 + 7 |
| free hex8 $12^3$ | 7k | 2.8 | 1.000 | 7 |

`amg` at σ = 0 on these clamped bodies works unchanged (finite factor, converged LOBPCG); only the refusal had to be lifted.

**Acceptance prototype** (`p13b/acc.py`, `sm.py`: the planned `solve_modes` path; hex20 $44^3$, 1.08M DOF, 20 modes, tol $10^{-6}$, 9-block step, fp64 basis, one cold process, compilation included):

| operator build | σ pre-estimate (σ = 0 build + 6 iterations) | final `amg` | LOBPCG | total | peak |
|---|---|---|---|---|---|
| 1.9 s | 15.2 s | 4.5 s | 51.8 s (19 iterations) | 73.4 s | 7.61 GB |

- The budget is 15 min (12× margin). The model in `docs/eigensolvers.md` assumed $N_\text{it}$ = 150; measured 19.
- σ = 1.000 λ₁ (11 923 132 against the converged λ₁ 11 923 130).
- Projected estimate: `lobpcg_memory` (Jacobi, stored) 5.28 GB + AMG storage 0.55 GB + 6.0 step blocks (1.29 GB) ≈ 7.1 GB, 6% below the cold peak; the warm peaks are measured in the implementation.

**LOBPCG step with AMG** (`p13b/stepmem.py`; temporaries plus aliased state of the compiled step, extra fp64 blocks over the Jacobi step; standalone V-cycle temporaries for comparison):

| mesh | $m$ | 9-block extra | low-memory extra | V-cycle alone |
|---|---|---|---|---|
| hex20 $24^3$ / $44^3$ | 15 | 8.40 / 6.70 | 12.05 / 11.58 | 14.4 / 12.5 |
| hex20 $24^3$ / $44^3$ | 25 | 7.32 / 5.99 | 11.63 / 10.86 | 13.2 / 11.7 |
| hex27 $24^3$ | 15 / 25 | 5.64 / 5.00 | 10.40 / 9.76 | 11.6 / 11.0 |
| hex8 $48^3$ / $68^3$ | 15 | 12.74 / 11.86 | 17.17 / 16.61 | 18.7 / 17.7 |
| hex8 $48^3$ / $68^3$ | 25 | 9.70 / 9.01 | 14.15 / 13.79 | 15.6 / 14.9 |
| hex20 $12^3$ (small: the Jacobi step itself is 25–29 blocks) | 15 | 10.19 | 13.46 | 29.1 |

- The extra falls with $m$: in bytes it fits $8n(a\,m + c)$ (hex20 $44^3$: $a$ = 4.9, $c$ = 27; hex8 $68^3$: $a$ = 4.7, $c$ = 107), so a per-family model $a\,m + c$ fits better than fixed blocks.
- All-hex8 costs more: its first transfer $P_0$ acts on the fine DOFs.
- Unlike GMG (`GMG_STEP`: 5.1 / 9.9), the low-memory step gains nothing with AMG: 9-block and low-memory totals are within 1% at $m = 25$.

**Device storage** (`p13b/storage.py`; preconditioner-owned bytes, leaves shared with the operator excluded):

| mesh | total | per fine DOF | AMG levels | AMG transfers | coarsest |
|---|---|---|---|---|---|
| hex20 $24^3$ / $44^3$ | 0.098 / 0.550 GB | 541 / 511 B | 0.042 / 0.236 GB | 0.044 / 0.241 GB | 0.2 / 4.5 MB |
| hex27 $24^3$ | 0.119 GB | 337 B | 0.042 | 0.044 | 0.2 MB |
| hex8 $48^3$ / $68^3$ | 0.402 / 1.079 GB | 1140 / 1095 B | 0.068 / 0.177 | 0.320 / 0.862 | 0 |

The rest of the hex20/hex27 total is the p-level (node-block Jacobi inverses, `NodalTransfer` tables, mask).

**SIMP study** (`p13b/simp.py`; the checked-in $32^3$ field and the 11a seeds as unstructured hex8 meshes, `s_K` = ρ³, `s_M` = ρ, σ from `gmg_study.json`; PCG to $10^{-8}$, 4 right-hand sides):

| field, σ | GMG (11a) | AMG, `max_coarse` 1000 | AMG, `max_coarse` 100 |
|---|---|---|---|
| simp32 at own λ₁ | 87 | 745–810 | 902–924 |
| simp32 at the solid's λ₁ | 33 | 70–72 | 72–75 |
| seed 4 at own λ₁ | 134–135 | 640–725 | 700–907 |
| seed 5 at own λ₁ | 107–109 | 136–137 | 151–152 |
| unfiltered binary at own λ₁ | 452 | 592–684 | 588–643 |

- AMG is worse than GMG on every field: GMG deflates the stiff islands; AMG's near-null space is the global rigid modes, and θ = 0 aggregates across void and solid.
- A strength threshold θ = 0.02 brings simp32 at its own λ₁ to 49–51 (levels 107811 → 11061 → 3399 → 1811, coarsest 986), but at the solid's λ₁ the coarse levels turn dense (1804 nonzeros per row at 1804 rows) and coarsening stalls at 1152; that build plus PCG did not finish in 16 min. On the 13a cases θ = 0.02 changes no count by more than 2 (h-sequences 14/14/14/14, plates 36/83/217) but raises AMG storage 34% (hex20 $44^3$, 0.74 GB) and 56% (hex8 $68^3$, 1.68 GB).
- Decision (without user input): keep θ = 0 and no `theta` keyword (it would expose a setting with a pathological setup). The θ = 0.02 result goes to `docs/open_items.md` for PRs 21–23, together with the per-island near-null space GMG uses.

**10M projections** (existing estimators plus the AMG terms above; hex20 $95^3$, 10.5M DOF):

| modes ($m$) | stored Ke, 9-block | stored, low-memory | on-the-fly, 9-block |
|---|---|---|---|
| 10 (15) | 48.5 GB | 48.3 GB | 38.1 GB |
| 20 (25) | 68.7 GB (65 GB device) | 68.5 GB | 58.3 GB |
| 30 (36) | 90.9 GB | 90.6 GB | 80.5 GB |

- GPU (76 GB usable pool): 10M hex20 with 20 modes fits with the 9-block step (about 65 GB on the device; the host term is not device memory); 30 modes does not. The roadmap's "10M fits" line and the AMG rows are rewritten with these numbers (re-derived by `amg_memory` in the implementation).
- Mac (64 GB unified): the optional 10M run uses 10 modes (48.5 GB), stored Ke, 9-block step.

## Design

### 1. Production hex8 Galerkin assembler (`pmg.py`)

`_hex8_galerkin(A, pattern=None) -> (Ab, T, corner, pattern)`:
- **`_pattern(op)`** (host, topology only): PR 12's `_tables` (corner ids, `NodalTransfer`, owner flags), the node-block pattern `indptr`, `indices` (int32, sorted) of $E^\top E$, per block an `(n_pad, 64)` int32 position table (padded elements point past the end and are dropped by the scatter), and the transpose permutation of the pattern's blocks. Returned as a small frozen host dataclass `Hex8Pattern`.
- **`_scatter_cells(data, op, b, sigma, m, d, own, pos)`** (jitted): one `lax.scan` over the block's chunks; each step builds the chunk's cells exactly as PR 12's `_cells` (owned diagonal, masks, `element_matrices`, the $P_\text{loc}^\top A_e P_\text{loc}$ contraction, now straight into `(c, 8, 8, 3, 3)`) and scatter-adds them (`mode="drop"`) into the carried `(nnzb, 3, 3)` values.
- Symmetrize on the device, `(data + data[tp].swapaxes(1, 2)) / 2` (exact: the pattern is symmetric by construction), copy to the host once, wrap as `scipy.sparse.bsr_array` with 3×3 blocks.
- `_cells` and the COO path are removed. `pmg` converts to CSC for `splu`.
- 13a's `_sa_levels` takes the BSR first level as is (`eliminate_zeros` on BSR drops all-zero blocks; strength and smoothing accept BSR), saving the CSR copy.
- Determinism: the scatter order on the device may vary, so values may differ from run to run by rounding; symmetry stays exact, and 13a's bitwise-determinism test applies to the SA loop on a fixed matrix (it does today).

### 2. Aggregate reuse

`amg(..., aggregates=None, return_aggregates=False)`:
- `return_aggregates=True` returns `(P, aggregates)`; `aggregates` is a frozen host dataclass `AMGAggregates` holding the `Hex8Pattern`, the tentative prolongators $T_l$ (after the zero-column drop) and coarse candidates $R_l$ per level, a copy of `free`, `max_coarse` and the corner count.
- With `aggregates=` the build skips the pattern, the corner tables, strength, aggregation and the tentative fit; it recomputes the cells, the smoothed prolongators, the Galerkin products and the device export from the current values. A different `free`, node count or `max_coarse` raises `ValueError`.
- Meaning: aggregates are structural under θ = 0, so a reused setup equals a fresh one (test 2 asserts both the roadmap's ≤ 20% and the equality). The option saves about 13% of a build and pins the hierarchy sizes, so PRs 21–23 can rebuild per design iteration without new shapes. `solve_modes` uses it across its σ rounds.

### 3. σ = 0 in `amg`

`amg(..., nonsingular=False)`: σ = 0 raises `SingularShiftError` as in 13a unless `nonsingular=True`, which states that the caller has proven $A_P$ nonsingular at σ = 0 (no rigid modes left by the BCs, every `s_K > 0`). `amg` checks `s_K > 0` itself (`ValueError` otherwise) and trusts the rest; the coarsest factor's finiteness check stays as the backstop. `pmg` keeps refusing σ = 0 (test-only). This closes the "p-MG refuses σ = 0" open item as the item proposed.

### 4. `solve_modes`

- `precond`: `"auto"` (GMG on voxel meshes, AMG otherwise), `"gmg"`, `"amg"`, `"jacobi"`. `"amg"` on a voxel mesh raises (`ValueError`: the stencil operator is not `Unstructured`; the SIMP study gives no reason to offer it). `precond_dtype="float32"` stays GMG-only (fp32 AMG is PR 18).
- `_gmg_sigma` becomes `_precond_sigma(op, free, R, build)`: `build(σ)` returns the preconditioner and raises `SingularShiftError` when σ = 0 is refused; the round logic is PR 11b's unchanged. GMG passes `lambda s: gmg(op, free, s)`; AMG passes a closure that calls `amg(op, free, s, mesh.points, nonsingular=(k == 0), aggregates=agg, return_aggregates=True)` and keeps `agg` for the next call (the final build reuses it too).
- `k == 0` is solid evidence: `rigid_modes(mesh, free)` returned no columns and did not raise (it rejects edge and vertex contacts); `solve_modes` passes no `s_K`, so every `s_K` is 1.
- `info["precond"]` is `"amg"`; `info["sigma"]`, `sigma_iterations` and `time["precond"]` as in 11b.
- Every existing unstructured `solve_modes` test, the m2 example and the README snippet now run AMG; they keep their assertions.

### 5. `memory.amg_memory(mesh, m)`

Same keys as `gmg_memory`: `storage`, `workspace`, `build`, `overhead`, `step`, `step_low`. Two families: all-hex8 meshes and the rest (hex20/hex27/mixed, with a p-level).
- **storage**: the p-level exactly from the mesh (node-block Jacobi inverses $72\,n_\text{nodes}$, `NodalTransfer` table widths from the element types, mask) plus the AMG part as a per-corner-DOF constant (hex20/hex27: AMG levels, transfers and coarsest ≈ 1760–1835 B per corner DOF at $24^3$–$44^3$; all-hex8: ≈ 1100–1140 B per DOF, mostly $P_0$). Constants are the envelope over the measured sizes with ≥ 5% margin, fitted on box meshes (documented as such).
- **step**, **step_low**: per family $8n(a\,m + c)$ envelopes over $24^3$–$68^3$ and $m$ = 15–25 (`AMG_STEP`).
- **workspace**: the standalone V-cycle temporaries, per family $8n(a\,m + c)$ (`AMG_WORK`).
- **build**: the setup host peak (assembler + SA loop + export), a per-corner-DOF constant fitted on the setup benchmark (Design 7).
- `solve_modes_memory` uses `amg_memory` for `precond="amg"` (and `"auto"` on unstructured meshes); the composition rule (`max(op build, precond build, step)`) is unchanged.

### 6. Memory ordering

Setup → free the assembled A → allocate LOBPCG:
- `amg` keeps no host matrix: the BSR values, SciPy levels and pyamg intermediates are locals freed on return (13a's leaves test pins the pytree; a new test checks that no `scipy.sparse` object survives `amg` via `gc`). The device copy of the hex8 values dies with the scatter's output once the host copy exists.
- `solve_modes` drops each σ-round hierarchy before the next build (11b), and the final `amg` returns before `lobpcg` allocates its blocks.
- `AMGAggregates` holds the $T_l$ (host, about 0.1 GB at 1M hex20); `solve_modes` drops it before `lobpcg`.
- Measured by test 10 (peak within 20% of `solve_modes_memory`, whose `max(build, step)` term is this ordering).

### 7. Benchmarks (`benchmarks/amg_study.py`, report-only)

- `setup()`: per size, time and host peak (fresh subprocess) of the assembler, the SA loop and the whole `amg`, hex20 $12^3$/$24^3$/$44^3$ and hex8 $24^3$/$48^3$/$68^3$; plus the reuse build time.
- `simp()`: the SIMP table above (AMG at θ = 0 only; the θ = 0.02 probe stays in the plan and `open_items.md`).
- `acceptance(n=44, n_modes=20)`: as `gmg_study.acceptance`, the cold `solve_modes` run (time, iterations, σ) and the peaks of two warm runs next to `solve_modes_memory`.
- `large(n=95, n_modes=10)`: the optional 10M run (cold, one process), recorded when run by hand (`--large`); not in the default run.
- The cost entry is re-run with the production assembler.

### 8. ROADMAP memory rows and the deferred GPU check

- Multigrid row: AMG device storage from `amg_memory` (≈ 0.55 GB at 1M hex20, ≈ 5.4 GB at 10M), the setup host peak from the benchmark.
- Preconditioner workspace row: AMG as a step extra (≈ 6 blocks with the 9-block step at $m = 25$, ≈ 11 with the low-memory step) instead of "≈ 4 blocks".
- "80 GB GPU: 10M fits" becomes "10M hex20 fits with 20 modes ($m = 25$, 9-block step): ≈ 65 GB on the device; 30 modes does not (≈ 87 GB)", with the figures re-derived by the shipped `solve_modes_memory`.
- Deferred GPU check entry "PR 13: 10M hex20 modal solve on one GPU" gets that configuration.

## Files

- `src/jaxmodal/precond/pmg.py`: `Hex8Pattern`, `_pattern`, `_scatter_cells`, `_hex8_galerkin(A, pattern=None)`; `_cells` and the COO path removed.
- `src/jaxmodal/precond/amg.py`: `AMGAggregates`, `aggregates`/`return_aggregates`/`nonsingular` keywords, BSR first level.
- `src/jaxmodal/eigen/api.py`: `"amg"` dispatch, `_precond_sigma`.
- `src/jaxmodal/memory.py`: `amg_memory`, `AMG_STEP`, `AMG_WORK`, constants; `solve_modes_memory` dispatch.
- `benchmarks/amg_study.py`, `benchmarks/results/amg_study.json`; `benchmarks/pmg_study.py` (new `_hex8_galerkin` return).
- Tests: `tests/test_amg.py`, `tests/test_pmg.py`, `tests/test_api.py`, `tests/test_gmg.py` (shared step helper moves to `tests/helpers.py` if reused).
- Docs as below.

## Tests

1. **Assembler against the oracle** (`test_pmg.py`). For hex20 $5\times4\times3$ with a sliding face and nodal masses, hex27 $3^3$, hex8 $5^3$ and a mixed hex8 + hex20 mesh (if the mesh builders give one; else two blocks of one type): `_hex8_galerkin(A)` equals $P^\top A_P P$ from PR 4's `assemble` and a dense $P$ from `NodalTransfer` to $10^{-12}$ relative; the pattern equals the structural corner adjacency (9 entries per adjacent corner pair, pinned by an independent count); exact symmetry; a chunk size that leaves padded elements; `pattern=` reuse gives the same matrix bitwise when nothing changes. Indices are int32.
2. **Aggregate reuse** (roadmap). hex20 $6^3$ and hex8 $8^3$ with a seeded random density (`s_K` = ρ³, `s_M` = ρ, ρ ∈ [0.5, 1]); aggregates from ρ, then a 20% seeded perturbation ρ': PCG iterations with reuse ≤ 1.2× a fresh setup on ρ' (roadmap), and the reused V-cycle equals the fresh one to $10^{-12}$ (θ = 0: aggregates are structural). Errors: another `free`, another mesh size, another `max_coarse` raise `ValueError`. A spy shows that strength, aggregation and `fit_candidates` are not called with reuse.
3. **σ = 0.** `amg(σ = 0)` raises `SingularShiftError`; with `nonsingular=True` on a clamped hex20 and hex8 box, PCG on $A_P(0)$ converges and the V-cycle is finite; `s_K` with a zero entry and `nonsingular=True` raises `ValueError`.
4. **`solve_modes` against `reference_modes`** (roadmap). hex20 (clamped, free with nodal masses, sliding) and pure hex8 (clamped, free), 6 modes, tol $10^{-10}$: λ to $10^{-8}$ and MAC ≥ $1 - 10^{-8}$ (PR 9 / 11b tolerances); `info["precond"] == "amg"`; the final build is at `info["sigma"]`; clamped bodies build first at σ = 0 (spy on `amg`).
5. **σ pre-estimate cases.** On the unstructured cases of the table above (hex20 $8^3$, hex8 $12^3$, beam, plate, free box, free beam, free plate; the slow ones marked `slow`): `_precond_sigma` with the AMG builder lands in $[\lambda_1(1 - 10^{-6}), 2\lambda_1]$; clamped bodies build once at σ = 0; free bodies never raise σ. The round-logic unit test runs against the generalized function.
6. **Dispatch and errors.** `"auto"` on a box mesh → `"amg"`; `"amg"` on a voxel mesh raises; `precond_dtype="float32"` with `"amg"` raises; a σ hint skips the pre-estimate (one build, at the hint); σ = 0 hint on a free body raises `SingularShiftError`; `"jacobi"` still works on unstructured meshes.
7. **`amg_memory` storage and workspace.** On hex20 $12^3$, hex27 $6^3$, hex8 $16^3$ and a sliding-mask case: `storage` ≥ the measured preconditioner-owned device bytes and ≤ 1.3× them; `workspace` ≥ `memory_analysis` of the V-cycle; tight at a mid size (≤ 1.25×). A non-box hex20 mesh (`distort`) stays inside the bound (documents the box calibration).
8. **AMG step constants.** The LOBPCG step with AMG ≤ `lobpcg_memory(..., precond_memory=amg_memory)`'s step (9-block and low-memory, hex20 and hex8); a `slow` tight test at hex20 $24^3$ and hex8 $48^3$ (where the constants bind) asserts the measured extra within 0.6 blocks below the model, as 11b's `test_gmg_step_tight`.
9. **Benchmark smokes** (`slow`): `amg_study.setup` at small sizes records times and peaks; `amg_study.simp` at a $8^3$ field runs and records counts for AMG and GMG.
10. **Mac acceptance** (roadmap; `slow`, Darwin only). `amg_study.acceptance(44)`: the cold run ≤ 900 s, all converged; the larger of two warm peaks within 20% of `solve_modes_memory` (estimate ≥ 2 GB). A smoke at $8^3$ checks the recorded keys and that the estimate is `solve_modes_memory`'s.
11. **No host matrices survive `amg`.** After `amg(...)` and `gc.collect()`, no new `scipy.sparse` arrays are alive (compare `gc.get_objects()` counts before and after).

The implementation re-measures every iteration bound and memory constant with the shipped code.

## Docs

- `docs/preconditioners.md`: the production assembler (pattern, fused scan, times and peaks); aggregate reuse (what is reused, why it equals a fresh setup under θ = 0, the saving); σ = 0 with `nonsingular`; `amg_memory`; the SIMP study (with the θ = 0.02 probe); the setup benchmark table; tests list.
- `docs/eigensolvers.md`: `solve_modes` dispatch (`"amg"`), the σ pre-estimate with AMG (table), the acceptance result in the Mac budget table (measured row for hex20).
- `README.md`: status (PR 13 complete, M3 done), the `solve_modes` snippet note that unstructured meshes use AMG, `solve_modes_memory` with AMG.
- `docs/ROADMAP.md`: PR 13b status line; the memory table rows and "10M fits" line (Design 8); the deferred GPU check configuration.
- `docs/open_items.md`:
  - "Voxel GMG on high-contrast SIMP fields": the AMG study answers the revisit (AMG is worse at θ = 0; θ = 0.02 helps at own λ₁ but turns the coarse levels dense); retargeted to PRs 21–23.
  - "p-MG refuses σ = 0": resolved for `amg` (`nonsingular`, used by `solve_modes`); `pmg` keeps refusing.
  - "AMG memory figures in the ROADMAP table": resolved (Design 8).
  - "Binary-field λ₁ miss": unchanged (AMG is no better there).
  - New, decision without user input: θ = 0 kept, no `theta` keyword.
  - New: AMG memory constants are calibrated on box meshes.

## Not covered

- fp32 AMG (PR 18).
- AMG on solid-shell meshes (PR 18) and flat-element fixes (M4).
- A `mode="otf"` option in `solve_modes` (the 10M Mac run uses stored Ke at 10 modes).
- High-contrast fixes (PRs 21–23).
