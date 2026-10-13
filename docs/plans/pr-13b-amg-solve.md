# PR 13b plan: AMG in `solve_modes`, production hex8 assembler, reuse, estimator, Mac acceptance

Status: draft (plan review round 2; round 1 findings applied, see the end).

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
- Projected estimate: `lobpcg_memory` (Jacobi, stored) 5.28 GB + AMG storage 0.55 GB + 6.0 step blocks (1.29 GB) ≈ 7.1 GB, 7% below the cold peak (1 − est/peak; 7.4 GB with the size envelope of Design 5, 3% below); the warm peaks are measured in the implementation.

**LOBPCG step with AMG** (`p13b/stepmem.py`; temporaries plus aliased state of the compiled step, extra fp64 blocks over the Jacobi step; standalone V-cycle temporaries for comparison):

| mesh | $m$ | 9-block extra | low-memory extra | V-cycle alone |
|---|---|---|---|---|
| hex20 $24^3$ / $44^3$ | 15 | 8.40 / 6.70 | 12.05 / 11.58 | 14.4 / 12.5 |
| hex20 $24^3$ / $44^3$ | 25 | 7.32 / 5.99 | 11.63 / 10.86 | 13.2 / 11.7 |
| hex27 $24^3$ | 15 / 25 | 5.64 / 5.00 | 10.40 / 9.76 | 11.6 / 11.0 |
| hex8 $48^3$ / $68^3$ | 15 | 12.74 / 11.86 | 17.17 / 16.61 | 18.7 / 17.7 |
| hex8 $48^3$ / $68^3$ | 25 | 9.70 / 9.01 | 14.15 / 13.79 | 15.6 / 14.9 |
| hex20 $12^3$ (small: the Jacobi step itself is 25–29 blocks) | 15 | 10.19 | 13.46 | 29.1 |

- The extra falls with $m$ and with $n$. Per family it fits $8n(a\,m + c) + 8mK$: the last term is n-independent (about 3 MB per column: the block-ELL scan-chunk gathers of the levels, $P$ and $P^\top$, bounded by chunk × width), which is why small meshes carry 10–40 extra blocks (hex20 $8^3$: 27.3 at $m = 15$; low-memory about 40 at hex20 $8^3$, hex8 $12^3$/$16^3$, hex27 $6^3$; reviewer probes `r13b/tests/step_small.py`, `r13b/perf/small.py`). A pure $8n(a\,m + c)$ envelope cannot be an upper bound at $12^3$ and tight at $44^3$ at once.
- All-hex8 costs more: its first transfer $P_0$ acts on the fine DOFs.
- Unlike GMG (`GMG_STEP`: 5.1 / 9.9), the low-memory step gains nothing with AMG: 9-block and low-memory totals are within 1% at $m = 25$.

**Device storage** (`p13b/storage.py`; preconditioner-owned bytes, leaves shared with the operator excluded):

| mesh | total | per fine DOF | AMG levels | AMG transfers | coarsest |
|---|---|---|---|---|---|
| hex20 $24^3$ / $44^3$ | 0.098 / 0.550 GB | 541 / 511 B | 0.042 / 0.236 GB | 0.044 / 0.241 GB | 0.2 / 4.5 MB |
| hex27 $24^3$ | 0.119 GB | 337 B | 0.042 | 0.044 | 0.2 MB |
| hex8 $48^3$ / $68^3$ | 0.402 / 1.079 GB | 1140 / 1095 B | 0.068 / 0.177 | 0.320 / 0.862 | 0 |
| hex20 $12^3$ (reviewer) | 18.5 MB AMG part | 2800 B per corner DOF | 8 MB | 6 MB | 4.5 MB |
| hex8 $8^3$ / $16^3$ / $24^3$ (reviewer) | | 774 / 1106 / 1187 B | | | |

The rest of the hex20/hex27 total is the p-level (node-block Jacobi inverses, `NodalTransfer` tables, mask). The dense coarsest factor does not scale with $n$ (it is set by where coarsening stops), so at small sizes a per-corner-DOF constant fitted at $24^3$–$44^3$ under-predicts (hex20 $12^3$: 53% above it).

**Whole `amg` with the production assembler** (performance reviewer's emulation, fresh subprocesses): hex20 $24^3$ / $44^3$ 0.29 / 1.61 GB, 1.9 / 5.2 s (6.2 / 5.9 kB per corner DOF); hex8 $48^3$ / $68^3$ 1.77 / 4.95 GB, 5.3 / 13.0 s (5.0 kB per DOF). The build term binds for all-hex8 at 10 modes (the 1M hex8 step is about 3.5 GB).

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
| 10 (15) | 48.5–51.2 GB | 48.3 GB | 38.1–40.9 GB |
| 20 (25) | 68.4–71.2 GB | 68.5 GB | 58.0–60.8 GB |
| 30 (36) | 89.6–94.7 GB | 90.6 GB | 80.5 GB |

Ranges: the $44^3$ fit to a $24^3$–$68^3$ envelope (performance reviewer, `r13b/perf/proj.py`); Design 5's model with the fixed per-column term lands near the lower end. The step model is fitted at $m$ = 15–25, so the $m = 36$ row is an extrapolation (enough for "does not fit").

- GPU (76 GB usable pool): only the fp64 modes of the `host` term live on the host; its freed-step-output block is device memory. Device estimate for 20 modes, stored Ke, 9-block: about 67–70 GB, and the 1M cold peak ran 3–7% above the estimate, so 71–74 GB against 76 GB: it fits, thinly. 30 modes does not (about 85–90 GB). Fallback if the GPU run does not fit: on-the-fly Ke (about 56–59 GB device), which needs `unstructured(mode="otf")` + `amg` + `lobpcg` directly (`solve_modes` has no `mode=`), or fewer modes. The 10M setup is host-side: about 16 GB host RAM (5.9 kB per corner DOF).
- Mac (64 GB unified): the optional 10M run uses 10 modes, stored Ke, 9-block step (about 49–51 GB, plus the cold excursion about 52–55 GB). `large()` prints `solve_modes_memory` first and stops above 52 GB.

## Design

### 1. Production hex8 Galerkin assembler (`pmg.py`)

`_hex8_galerkin(A, pattern=None) -> (Ab, pattern)`:
- **`_pattern(op) -> Hex8Pattern`** (host, topology only; a small frozen dataclass): PR 12's `_tables` output (corner ids, the `NodalTransfer` `T`, owner flags) plus `corner` (corner node ids in coarse order), the node-block pattern `indptr`, `indices` (int32, sorted) of $E^\top E$, per block an `(n_pad, 64)` int32 position table, and the transpose permutation of the pattern's blocks. Padded elements get position `nnzb` (past the end; never −1, which JAX would wrap to the last block before `mode="drop"`). `check_int32` covers `nnzb` and the scalar count $9\,\text{nnzb}$ (pyamg's CSR intermediates).
- **`_scatter_cells(data, op, b, sigma, m, d, own, pos)`** (jitted): one `lax.scan` over the block's chunks; each step builds the chunk's cells exactly as PR 12's `_cells` (owned diagonal, masks, `element_matrices`, the $P_\text{loc}^\top A_e P_\text{loc}$ contraction, now straight into `(c, 8, 8, 3, 3)`) and scatter-adds them (`mode="drop"`) into the carried `(nnzb, 3, 3)` values.
- Copy the values to the host once and symmetrize there with the transpose permutation, `(d + d[tp].swapaxes(1, 2)) / 2` (exact: IEEE addition commutes, so blocks $ij$ and $ji$ come out exact transposes); one device copy instead of three. Wrap as `scipy.sparse.bsr_array` with 3×3 blocks.
- Callers read `pattern.T` and `pattern.corner`; `_p_level` returns `((A, S), Ab, pattern)` (13a's trigger for slimming its tuple: a third consumer). `_cells` and the COO path are removed. `pmg` converts to CSC for `splu`.
- 13a's `_sa_levels` takes the BSR first level as is: BSR `eliminate_zeros` drops exactly the all-zero blocks (reviewer probe), the same block pattern as today's CSR copy, so 13a's pyamg-equality test is unaffected.
- Determinism: on CPU the scatter is deterministic (bitwise-identical rebuilds); on a GPU scatter-add order may vary by rounding. Tests that compare rebuilds use $10^{-15}$ relative, not bitwise; the GPU behaviour joins the deferred GPU list.

### 2. Aggregate reuse (an `amg` option, roadmap; for PRs 21–23)

`amg(..., aggregates=None, return_aggregates=False)`:
- `return_aggregates=True` returns `(P, aggregates)`; `aggregates` is a frozen host dataclass `AMGAggregates`: the `Hex8Pattern`, the tentative prolongators $T_l$ (after the zero-column drop) and coarse candidates $R_l$ per level, a copy of `free`, and a connectivity fingerprint (per block: element count and a hash of `dofs`).
- With `aggregates=` the build skips the pattern, the corner tables, strength, aggregation and the tentative fit; it recomputes the cells, the smoothed prolongators (`jacobi_prolongation_smoother` reads its strength argument only when filtering, so it gets `None`), the Galerkin products and the device export. It runs under the same local `np.random.seed(0)` and the same smoother call order: the spectral-radius estimate is the only RNG consumer, so skipping aggregation does not shift the stream (code comment). A different `free`, node count or connectivity fingerprint raises `ValueError`.
- Meaning:
  - **Values change, geometry fixed** (σ, `s_K`, `s_M`, materials): under θ = 0 first-level aggregates depend only on the pattern, the mask and $B$; coarser levels on the pattern of $P^\top A P$, which values change only through exact cancellations (none observed, even at uniform density). So a reused setup equals a fresh one (reviewer probes: identical $T_l$, strength nnz and $P$, difference 0.0, for σ = 0 vs > 0, ρ vs a 20% perturbation, uniform vs perturbed).
  - **Moved points** (FFD, PR 22): reuse keeps the old near-null space $B$ (points are not checked). It is still an SPD V-cycle and only approximates a fresh setup; documented, and test 2 bounds its iterations.
- The option saves about 13% of a build and pins the hierarchy sizes. `solve_modes` does not use it (under 1% of the acceptance run, and under θ = 0 its hierarchy shapes are identical anyway).

### 3. σ = 0 in `amg`

`amg(..., nonsingular=False)`: σ = 0 raises `SingularShiftError` as in 13a unless `nonsingular=True`, which states that the caller has proven $A_P$ nonsingular at σ = 0 (no rigid modes left by the BCs, every `s_K > 0`). `amg` checks `s_K > 0` itself (`ValueError` otherwise) and trusts the rest; the coarsest factor's finiteness check stays as the backstop (a singular coarsest matrix can factor with rounding-level pivots, so the backstop alone is not enough: hence the explicit flag). `pmg` keeps refusing σ = 0 (test-only). This closes the "p-MG refuses σ = 0" open item as the item proposed.

### 4. `solve_modes`

- `precond`: `"auto"` (GMG on voxel meshes, AMG otherwise), `"gmg"`, `"amg"`, `"jacobi"`. `"amg"` on a voxel mesh raises (`ValueError`: the stencil operator is not `Unstructured`; mirrors `"gmg"` on unstructured meshes; the SIMP study gives no reason to offer it). `precond_dtype="float32"` stays GMG-only (fp32 AMG is PR 18).
- `_gmg_sigma` becomes `_precond_sigma(op, free, R, build)`: `build(σ)` returns the preconditioner and raises `SingularShiftError` when σ = 0 is refused; the round logic is PR 11b's unchanged. GMG passes `lambda s: gmg(op, free, s)`, AMG `lambda s: amg(op, free, s, mesh.points, nonsingular=(k == 0))`.
- `k == 0` is solid evidence: `rigid_modes(mesh, free)` works per face-connected part, returned no columns and did not raise (it rejects edge and vertex contacts; a floating part gives k > 0); `solve_modes` passes no `s_K`, so every `s_K` is 1.
- `info["precond"]` is `"amg"`; `info["sigma"]`, `sigma_iterations` and `time["precond"]` as in 11b.
- The existing unstructured `solve_modes` tests, the m2 example and the README snippet now run AMG and keep their accuracy assertions; `test_dispatch` changes (its `"auto"` → `"jacobi"` assertion on a box mesh becomes `"amg"`; test 6).

### 5. `memory.amg_memory(mesh, m)`

Returns the four keys `lobpcg_memory` reads from a preconditioner estimate: `storage`, `build`, `step`, `step_low` (no `workspace` or `overhead`: nothing consumes them). Two families: all-hex8 meshes and the rest (hex20/hex27/mixed, with a p-level).
- **storage** = the p-level exactly from the mesh (node-block Jacobi inverses $72\,n_\text{nodes}$, `NodalTransfer` table widths from the element types, mask) + the AMG levels and transfers as a per-corner-DOF constant (envelope over hex20 $12^3$–$44^3$ / hex8 $16^3$–$68^3$ excluding the coarsest, ≥ 5% margin) + the coarsest factor $8\min(n_c, \texttt{max\_coarse})^2$ (a stall up to the 4·`max_coarse` cap can exceed it; documented). Calibrated on box meshes (documented).
- **step**, **step_low**: per family $8n(a\,m + c) + 8mK$ over Jacobi's step (`AMG_STEP`: $a$, $c$, $K$ per family and step kind), fitted as an envelope on hex20 $12^3$–$44^3$, hex27 $24^3$ and hex8 $12^3$–$68^3$ at $m$ = 15 and 25, so it bounds the small sizes and stays tight (≤ about 1 block) at 1M.
- **build**: the `amg` setup host peak, per family a per-corner-DOF constant fitted on the setup benchmark (hex20 about 6 kB, hex8 about 5 kB per corner DOF), checked on hex27 too.
- `solve_modes_memory` uses `amg_memory` for `precond="amg"` and `"auto"` on unstructured meshes; the composition rule (`max(op build, precond build, step)`) is unchanged.

### 6. Memory ordering

Setup → free the assembled A → allocate LOBPCG:
- `amg` keeps no host matrix and no setup device array: the device values are copied to the host and dropped inside `_hex8_galerkin`; the BSR values, SciPy levels and pyamg intermediates are locals freed on return. Test 11 checks both (`gc` for SciPy objects, `jax.live_arrays()` for device bytes).
- `solve_modes` drops each σ-round hierarchy before the next build (11b), and the final `amg` returns before `lobpcg` allocates its blocks.
- Measured by test 10 (peak within 20% of `solve_modes_memory`, whose `max(build, step)` term is this ordering) and pinned for the build by test 12.

### 7. Benchmarks (`benchmarks/amg_study.py`, report-only)

- `setup()`: per size, time and host peak (fresh subprocess) of the assembler, the SA loop and the whole `amg`, hex20 $12^3$/$24^3$/$44^3$, hex27 $24^3$ and hex8 $24^3$/$48^3$/$68^3$, plus the reuse build time. This replaces `pmg_study`'s `galerkin_1m` entry (its COO numbers are history; removed from `pmg_study`, the pmg docs point to `amg_study.setup`).
- `simp()`: the SIMP table above (AMG at θ = 0 only; the θ = 0.02 probe stays in the plan and `open_items.md`).
- `acceptance(n=44, n_modes=20)`: as `gmg_study.acceptance`, the cold `solve_modes` run (time, iterations, σ) and the peaks of two warm runs next to `solve_modes_memory`.
- `large(n=95, n_modes=10)`: the optional 10M run (cold, one process), run by hand (`--large`), not in the default run; prints `solve_modes_memory` first and stops above 52 GB.
- The cost entry is re-run with the production assembler.

### 8. ROADMAP memory rows and the deferred GPU check

All figures from the shipped `solve_modes_memory` / `amg_memory`:
- Multigrid row, unstructured columns (1M hex20, 10M hex20): AMG device storage (≈ 0.55 GB / ≈ 5.4 GB) and the setup host peak (≈ 1.6 GB / ≈ 16 GB). The voxel columns keep GMG's figures.
- Preconditioner workspace row, unstructured columns only: AMG as a step extra (about 6 blocks at $m = 25$ with the 9-block step; about 11 with the low-memory step, which therefore gains nothing).
- "80 GB GPU: 10M fits" becomes: 10M hex20 with 20 modes ($m = 25$, stored Ke, 9-block step) needs about 67–70 GB on the device (fits the 76 GB pool thinly); 30 modes does not; fallback on-the-fly Ke.
- Deferred GPU check entry "PR 13: 10M hex20 modal solve on one GPU" gets that configuration, the fallback and the host RAM (about 16 GB); a new entry: assembler scatter determinism and speed on the GPU.
- Decision without user input (logged): the table assumed $m = 30$ for 10M; with AMG that does not fit, so the GPU check runs 20 modes. The roadmap check names no mode count, so no criterion is relaxed.

## Files

- `src/jaxmodal/precond/pmg.py`: `Hex8Pattern`, `_pattern`, `_scatter_cells`, `_hex8_galerkin(A, pattern=None) -> (Ab, pattern)`, `_p_level`; `_cells` and the COO path removed.
- `src/jaxmodal/precond/amg.py`: `AMGAggregates`, `aggregates` / `return_aggregates` / `nonsingular` keywords, BSR first level.
- `src/jaxmodal/eigen/api.py`: `"amg"` dispatch, `_precond_sigma`.
- `src/jaxmodal/memory.py`: `amg_memory`, `AMG_STEP`, storage and build constants; `solve_modes_memory` dispatch.
- `benchmarks/amg_study.py`, `benchmarks/results/amg_study.json`; `benchmarks/pmg_study.py` (new `_hex8_galerkin` return, `galerkin_1m` removed), `benchmarks/results/pmg_study.json` entry removed.
- Tests: `tests/test_amg.py`, `tests/test_pmg.py`, `tests/test_api.py`; the step helper moves from `test_gmg.py` to `tests/helpers.py`.
- Docs as below.

## Tests

1. **Assembler against an independent oracle** (`test_pmg.py`). Reference: `p_ref(mesh)` (the element loop over reference nodes, independent of `_tables`) with PR 4's `assemble` and `a_p`. Cases: hex20 $5\times4\times3$ (consistent and lumped mass, sliding face, nodal masses), hex27 $3^3$, hex8 $5^3$, and `helpers.mixed_mesh` with nodal masses on the shared interface nodes and a mask that constrains some shared nodes (catches a doubled or missing owned diagonal). Each: `_hex8_galerkin(A)` equals $P^\top A_P P$ to $10^{-12}$ relative; the pattern equals the structural corner adjacency (9 entries per adjacent corner pair, by an independent count); exact symmetry; a chunk size that leaves padded elements, with the last block asserted exact (catches a −1 padding position); `pattern=` reuse gives the same matrix to $10^{-15}$; indices are int32.
2. **Aggregate reuse** (roadmap). hex20 $6^3$ and hex8 $8^3$, clamped plus a sliding face, seeded random density (`s_K` = ρ³, `s_M` = ρ, ρ ∈ [0.5, 1]); aggregates from ρ, then a 20% seeded perturbation ρ':
   - PCG iterations with reuse ≤ 1.2× a fresh setup on ρ' (roadmap);
   - the reused V-cycle equals the fresh one to $10^{-12}$ (the smoothed $P$ changes by 12–13% between ρ and ρ', so this catches stale cells, $P$ or Galerkin products);
   - σ = 0 aggregates reused at σ > 0 (with `nonsingular`) equal a fresh σ > 0 setup;
   - moved points (seeded 2% perturbation): reuse converges within 1.2× a fresh setup (documents the approximation);
   - a spy shows that `_pattern`, `_tables`, strength, aggregation and `fit_candidates` are not called with reuse;
   - errors: another `free`, another node count, another connectivity with the same node count (a renumbered element order) raise `ValueError`.
3. **σ = 0.** `amg(σ = 0)` raises `SingularShiftError`; with `nonsingular=True` on a clamped hex20 and hex8 box, PCG on $A_P(0)$ converges and the V-cycle is finite; `s_K` with a zero entry and `nonsingular=True` raises `ValueError`.
4. **`solve_modes` against `reference_modes`** (roadmap). Non-degenerate boxes (hex20 $5\times4\times3$, hex8 $6\times5\times4$; clamped, free with nodal masses, sliding), 6 modes, tol $10^{-10}$: λ to $10^{-8}$ and MAC ≥ $1 - 10^{-8}$ (PR 9 / 11b tolerances); `info["precond"] == "amg"`; the final build is at `info["sigma"]`. An `amg` spy asserts: clamped bodies build first at σ = 0 with `nonsingular=True`; free and sliding bodies call it only with σ > 0 and `nonsingular=False`.
5. **σ pre-estimate cases.** On hex20 $8^3$, hex8 $12^3$, beam, plate, free box, free beam, free plate (the slow ones marked `slow`): `_precond_sigma` with the AMG builder lands in $[\lambda_1(1 - 10^{-6}), 2\lambda_1]$; the spy assertions of test 4; free bodies' successive build shifts never increase. The round-logic unit test runs against the generalized function.
6. **Dispatch and errors** (updates `test_dispatch`). `"auto"` on a box mesh → `"amg"`; `"amg"` on a voxel mesh raises; `precond_dtype="float32"` with `"amg"` raises; a σ hint skips the pre-estimate (one build, at the hint); σ = 0 hint on a free unstructured body raises `SingularShiftError`; `"jacobi"` still works on unstructured meshes.
7. **`amg_memory` storage.** On hex20 $12^3$, hex27 $6^3$, hex8 $16^3$ and a sliding-mask case: `storage` ≥ the measured preconditioner-owned device bytes (leaves shared with the operator excluded) and ≤ 1.3× them; a distorted hex20 mesh (`distort`) stays inside the bound (documents the box calibration).
8. **AMG step constants.** The LOBPCG step with AMG ≤ `lobpcg_memory(..., precond_memory=amg_memory)`'s step for 9-block and low-memory at hex20 $8^3$ and $12^3$, hex27 $6^3$, hex8 $12^3$ (non-slow); a `slow` tight test at hex20 $24^3$ and hex8 $48^3$ asserts the measured extra within 1 block below the model (as 11b's `test_gmg_step_tight`).
9. **Benchmark smokes** (`slow`): `amg_study.setup` at small sizes records times and peaks; `amg_study.simp` at an $8^3$ field records counts for AMG and GMG.
10. **Mac acceptance** (roadmap; `slow`, Darwin only). `amg_study.acceptance(44)`: the cold run ≤ 900 s, all converged; the larger of two warm peaks within 20% of `solve_modes_memory` (estimate ≥ 2 GB). A smoke at $8^3$ checks the recorded keys and that the estimate is `solve_modes_memory`'s.
11. **Nothing from the setup survives `amg`.** With `return_aggregates=False`, after `amg(...)` and `gc.collect()`: no new `sp.sparray`/`sp.spmatrix` instances, and `jax.live_arrays()` bytes beyond the operator, mask and the returned pytree are 0.
12. **Build term** (`slow`, subprocess via `harness.measure`). The whole `amg` host peak at hex20 $24^3$, hex27 $12^3$ and hex8 $48^3$: `amg_memory(...)["build"]` ≥ measured and ≤ 1.3× measured.

The implementation re-measures every iteration bound and memory constant with the shipped code.

## Docs

- `docs/preconditioners.md`: the production assembler (pattern, fused scan, host symmetrization, times and peaks), replacing the pmg section's COO assembly steps and its `galerkin_1m` cost text; aggregate reuse (what is reused, why it equals a fresh setup for value changes, the approximation for moved points, the saving); σ = 0 with `nonsingular`; `amg_memory` (model, size range, box calibration); the SIMP study (with the θ = 0.02 probe); the setup benchmark table; tests list.
- `docs/eigensolvers.md`: `solve_modes` dispatch (`"amg"`), the σ pre-estimate with AMG (table); the memory section: `solve_modes_memory` composes `amg_memory`, the `step` bullet gains `AMG_STEP` (form, fitted range, $m = 36$ extrapolation, no low-memory gain), the measured-peak paragraph gains the acceptance peaks; the Mac budget table gets the measured hex20 row, and the hex20 setup sentence the measured setup times.
- `README.md`: status (PR 13 complete, M3 done), unstructured meshes use AMG in `solve_modes`, `solve_modes_memory` with AMG.
- `docs/ROADMAP.md`: PR 13b status line (noting the COO path replacement); the memory table rows and "10M fits" line (Design 8); the deferred GPU entries.
- `docs/open_items.md`:
  - "Voxel GMG on high-contrast SIMP fields": the AMG study answers the revisit (AMG is worse at θ = 0; θ = 0.02 helps at own λ₁ but turns the coarse levels dense); retargeted to PRs 21–23.
  - "p-MG refuses σ = 0": resolved for `amg` (`nonsingular`, used by `solve_modes`); `pmg` keeps refusing.
  - "AMG memory figures in the ROADMAP table": resolved (Design 8).
  - "Binary-field λ₁ miss": unchanged (AMG is no better there).
  - New, decisions without user input: θ = 0 kept, no `theta` keyword; the 10M GPU configuration (20 modes, $m = 25$, instead of the table's $m = 30$); the Mac 10M run at 10 modes; `nonsingular=True` trusts the caller apart from `s_K > 0`, and `solve_modes` derives it from `rigid_modes`; `"amg"` on voxel meshes raises; the COO assembler removed (pmg's Galerkin numbers change); reuse with moved points approximates.
  - New: AMG memory constants are calibrated on box meshes.

## Not covered

- fp32 AMG (PR 18).
- AMG on solid-shell meshes (PR 18) and flat-element fixes (M4).
- A `mode="otf"` option in `solve_modes` (the 10M Mac run uses stored Ke at 10 modes; the GPU fallback composes the building blocks).
- High-contrast fixes (PRs 21–23).

## Review log

Round 1 (5 reviewers, none satisfied, no BLOCKING): reuse removed from `solve_modes`; `amg_memory` trimmed to the consumed keys; `_hex8_galerkin` returns `(Ab, pattern)`; reuse validates connectivity and states the moved-points approximation; step model gains the n-independent term $8mK$ and the storage model an explicit coarsest term, with explicit test sizes; spy assertions on σ = 0 / `nonsingular` for free and sliding bodies; independent assembler oracle (`p_ref`, `mixed_mesh`, owner and padding cases); device-array survivor check; tested build term; GPU device arithmetic (only the modes are host) with a fallback; Mac 10M guard; host symmetrization; docs list completed and decisions logged.
