# PR 18 plan: performance pass, part c: fp32 AMG and σ-round reuse

Branch `pr-18c-fp32-amg` (this plan), from `main` after PR 18b. Roadmap: PR 18c's bullets
"fp32 option for p-MG and AMG (as in PR 11), including the solid-shell Schwarz patch inverses
(18a), modelled by the PR 13 estimator, and its validation" and "σ-round reuse of the
solid-shell AMG aggregates across `_precond_sigma`'s rebuilds", with the tests "With the fp32
preconditioner, the same convergence tolerance is reached" and "The PR 13 1M hex20 Mac
acceptance run with fp32 p-MG/AMG has peak within 20% of the PR 13 estimator";
`docs/ROADMAP.md:184` (the 18c estimators "validated against measured peaks where the
estimate is ≥ 2 GB"). The rest of the roadmap's PR 18c moves to new parts (below).

## Revisions

- **Revision 1:** first draft, from the prototypes in the session scratchpad `s18c/`
  (`proto.py`, `steps.py`, `reuse.py`).
- **Revision 2 (plan review round 1; all 5 not satisfied, one blocking):**
  - **Build term:** the shell `build` / `host_build` stay computed from the fp64 storage.
    The setup runs in fp64 on the host, and fp32 storage in the term undershot the measured
    fp32 build peak by 10–13% at 362²–512².
  - **fp32 memory constants:** one ratio per term times the fp64 term, instead of about 26
    refitted constants.
    - The ratios are the maxima over the fit grids.
    - The hex grid is widened to $m$ = 6–120.
    - Held-out points are checked out of the fit.
    - A hex fp32 polish term is added: the polish's PCG keeps fp64 vectors, so it exceeds
      the fp32 step.
  - **Aggregate lifetime:** the aggregates are dropped before any slice, as today. The
    plan's claim that slices reuse them is withdrawn.
  - **Level build:** levels are built directly in fp32 (`block_ell(..., dtype)`,
    `schwarz.build(..., dtype=)`). `gmg._to32` stays unchanged, with no shared helper.
  - **Tests:**
    - the V-cycle tolerances per case (thin shells round more), and the shell PCG cases
      moved off the round-off floor;
    - six existing tests that change;
    - an AMG copy of `test_fp32_slices`, the hex fp32 polish, a benchmark smoke, and
      `array_equal` for reuse.
  - **Docs:** a grep rule for every 18c / 18d / 18e retag, and the fp32 statements that
    become false.
  - **Benchmark run times** corrected; `steps` split in two; a `peaks` part added.

## The PR 18c split (decision without user input, logged)

The roadmap's PR 18c holds six independent items and, through the 18a plan
(`docs/plans/pr-18a-shell-amg.md:102–119`), about fifteen performance follow-ups. 18b showed
that one estimator calibration alone takes many review rounds. So 18c is split along the
code it touches, as 18a split PR 18:

| part | branch | contents |
|---|---|---|
| 18c | `pr-18c-fp32-amg` | fp32 AMG (hex and solid-shell, Schwarz inverses included), its `amg_memory` model and validation (convergence, the 1M hex20 fp32 acceptance, the 791k shell fp32 peak, fp32 build peaks), `precond_dtype="float32"` with AMG in `solve_modes`, σ-round reuse of the AMG aggregates (shell and hex), the 10M reach re-derived with fp32 AMG |
| 18d | `pr-18d-kernels` | Morton reordering and reordering tuning; kernel fusion checks (HLO, report only); the compile-time benchmark; the roofline tests (STREAM); `docs/performance.md`; the non-band follow-ups: `assembled._chunked` pipelining, caching `layout` for the gather path, `_amg_scan` keeping the mesh alive, `gmg_memory` recomputed per search step, the Metal `_default_budget`, $P_0^\top$ as a scatter over $P_0$'s ELL, 1.25× ELL width classes, the Schwarz scan remedy (chunks per step or a public `chunk_bytes`) if the GPU check needs it, the `scipy.ndimage.label` face-part finder (`docs/boundary_conditions.md:53`) |
| 18e | `pr-18e-band` | the old 18d (band follow-ups, `COUNT_STEP` refit), plus the band-path follow-ups from the old 18c: the refill through `device_put(V[:, i:i+c].T).T`, the verify skips before the gather (`open_items.md:167`, `:287`), merge and count copies (`:288`), host transposes and small compiles (`:289`) |
| 18f | `pr-18f-calculix` | the old 18e (CalculiX comparison) |

The band-path follow-ups go with the band PR because both rewrite the same functions
(`si_band`, `si_checked`, the LOBPCG growth). The dependency table becomes:
- 18c ← 11, 12, 13, 18b (unchanged);
- 18d ← 5, 6, 18b (no code dependency on 18c; it can start after 18b);
- 18e ← 14b, 15d, 18b (the old 18d's row);
- 18f ← 18e (the old "18e ← 18a–18d");
- 19 ← 17, 18a–18e (was 18a–18d).

`:627`'s "Split into five parts" becomes six (2026-10-19, this plan).

## Goal (18c)

1. `amg(..., dtype="float32")` runs the V-cycle in fp32 on every AMG path (hex8, the hex20 /
   hex27 p-level, solid shell, `ssh8` + `hex8`), with the same convergence as fp64.
2. `solve_modes(precond="amg", precond_dtype="float32")` (and `"auto"` resolving to AMG) uses
   it for LOBPCG's preconditioner and the polish; the memory model follows.
3. `solve_modes` builds the AMG aggregates once per call and reuses them in the later AMG
   builds before the slices (σ rounds, LOBPCG's build, band counts).

## Findings (prototypes and review round 1, CPU fp64 / fp32, M3 Max)

### Convergence (fp32 levels by casting a fp64 build)

PCG on $A_P$ (fp64) with the V-cycle, 4 masked random columns, max iterations, `amg` defaults:

| case | DOF | tol $10^{-8}$: fp64 / fp32 | tol $10^{-10}$: fp64 / fp32 |
|---|---|---|---|
| hex20 $16^3$, clamped, σ = 3e5 | 56,355 | 14 / 14 | 17 / 17 |
| `ssh8` plate $128^2$, $L/t$ = 10, two edges clamped, σ = 1 | 99,846 | 16 / 16 | 20 / 20 |
| `ssh8` plate $128^2$, $L/t$ = 1000 | 99,846 | 55–58 / 55–58 | 82–84 / 81–84 (both stop at 1.5–1.7e-8: the shell's round-off floor) |

Review round 1 (tests reviewer, 4 seeds) adds:
- **Plates $64^2$ off the floor:**
  - $L/t$ = 100 at $10^{-8}$: 18 / 18;
  - $L/t$ = 1000 at $10^{-6}$: 24–26 for both;
  - $L/t$ = 10 at $10^{-8}$: 15–16 for both.
- **$L/t$ = 1000 at $10^{-8}$** is below the floor: per column 73–77 for both, with true
  residuals 4–8e-8 for both.
- **The cycle itself:**
  - fp32 against fp64 differs by 1.5–4.5e-7 relative on hex8 / hex20 / hex27 / the block,
    with energy-normalized asymmetry ≤ 8e-8;
  - on the `ssh8` plate $32^2$ the difference grows with $(L/t)^2$: 3e-6 at $L/t$ = 10,
    2–5e-5 at 50, 2–4e-3 at 1000. The cause is the fine-level fp32 cycle; GMG's plate shows
    3e-4 (`test_gmg`).
- **Solve level (fp32 AMG in LOBPCG and the polish):**
  - hex20 $6\times5\times4$ at tol $10^{-10}$ in 25 iterations (as fp64);
  - stack ×3 $32^2$ at $10^{-10}$ in 80 (fp64 102; fp32 does not polish there, fp64 does);
  - the thin $32^2$ plate at $L/t$ = 1000, tol $10^{-6}$: 116 with the polish (fp64 118).
  All pass `check_reference`.

The fp32 cycle is a fixed linear operator up to fp32 round-off. PCG is not flexible, yet it
showed no loss of accuracy down to the fp64 floor.

### Storage (owned device bytes without the operator, float + int)

| case | fp64 | fp32 |
|---|---|---|
| hex20 $16^3$ | 25 + 6 MB | 13 + 6 MB |
| plate $128^2$ | 175 + 5 MB | 88 + 5 MB |

Float bytes halve except the fp64 parts (the free mask, the coarse Cholesky factor). The hex20
AMG is small next to its stored-Ke operator, which stays fp64 because LOBPCG shares it.

### LOBPCG step (extra over the Jacobi step, in blocks of $8nm$)

| case | $m$ | 9-block fp64 / fp32 | low fp64 / fp32 |
|---|---|---|---|
| hex20 $12^3$ | 15 / 25 / 60 | 10.2 / 9.0 / 7.3 → 7.4 / 6.5 / 5.0 | 13.5 / 12.3 / 11.2 → 11.4 / 10.5 / 9.7 |
| hex20 $24^3$ | 15 / 25 / 60 | 8.4 / 7.3 / 6.3 → 5.8 / 5.0 / 4.3 | 12.1 / 11.6 / 10.3 → 9.8 / 9.4 / 9.0 |
| hex8 $32^3$ | 15 / 25 / 60 | 13.3 / 10.7 / 8.3 → 8.2 / 6.9 / 5.7 | 17.3 / 14.5 / 12.1 → 12.2 / 10.8 / 9.6 |
| plate $128^2$ | 15 / 25 / 60 | 16.2 / 14.0 / 12.0 → 9.0 / 8.0 / 7.1 | 20.2 / 18.0 / 16.0 → 13.5 / 12.5 / 11.6 |

fp32 saves 1.3–7 blocks. Within one term the ratio fp32 / fp64 varies by only 0.04–0.10
across $m$ and $n$, which is what makes one ratio per term enough (below).

The **polish**'s PCG keeps fp64 vectors, so with fp32 AMG it exceeds the fp32 step on hex
(review round 1, totals in blocks):
- hex20 $14^3$: 28.5 against 26.9 at $m$ = 25, 38.1 against 37.2 at $m$ = 8;
- hex8 $24^3$: 27.5 against 27.3 at $m$ = 25.
In fp64 the hex polish stays below the step (`test_hex_polish_below_step`), so hex had no
polish term until now.

### Build peak (review round 1, fresh subprocesses)

The shell setup peak sits mostly in the host phase (`_shell_levels`, fp64 SciPy):
- flat $362^2$: 2.18 of 2.60 GB;
- block $64^3$: all of it.

With fp32 the build peak drops (4.57 against 5.14 GB at flat $512^2$), but by less than the
storage: 18b's `build` term with fp32 storage in it undershoots by 2.4 / 10 / 13% at flat
$256^2$ / $362^2$ / $512^2$. With the fp64 storage it stays above every fp32 peak (1.20× at
$512^2$, 1.23× at $362^2$).

### Reuse (warm, one build each)

| case | DOF | fresh | with `aggregates=` |
|---|---|---|---|
| plate $128^2$ / $256^2$ / $362^2$, $L/t$ = 1000 | 100k / 396k / 791k | 1.11 / 4.52 / 9.80 s | 1.02 / 4.12 / 8.86 s |
| stack ×3 $128^2$ | 200k | 2.30 s | 2.08 s |
| hex20 $24^3$ / $36^3$ | 182k / 596k | 0.50 / 1.66 s | 0.45 / 1.51 s |

- **Cost:** reuse saves about 10% of each later build; the Galerkin products, the Schwarz
  inversion and the export dominate.
- **Exactness:** λ, σ and the iteration counts are bitwise equal with and without reuse
  (review round 1):
  - hex20 $6\times5\times4$: 1 reuse;
  - free `ssh8` $32^2$ at $L/t$ = 50 / 1000: 3 / 4 reuses.
- **Aggregate size** (`_nbytes(aggs)`) per DOF: hex8 215, hex20 69–74, hex27 82, `ssh8`
  plate 187–190, block 211.

### Speed

fp32 AMG is a memory lever on hex: the p-level's fine apply stays fp64, and PCG time at hex20
$16^3$ is equal (1.0 s). On shells PCG is about 10% faster (2.8 against 3.1 s at $128^2$).
The benchmark records the V-cycle times.

## Design (18c)

### `_ap_apply` follows the vector's dtype (`operators/constraints.py`)

`_ap_apply` casts its result to `x.dtype`. The unstructured applies accumulate in fp64
(`matfree._assemble`), so an fp32 vector gives an fp64 product. Chebyshev's `fori_loop` carry
then changes dtype and fails to trace. With the cast, fp32 in gives fp32 out. The work is the
same as fp64, and the operator gets no fp32 copy. fp64 callers are unchanged, since the cast
is a no-op. No current caller passes fp32 into an `APOperator`: counting and shift-invert are
fp64. The voxel `Stencil` already follows the vector's dtype.

### `amg(..., dtype="float64")` (`precond/amg.py`, `precond/schwarz.py`)

- **Validation:** `dtype` is `"float64"` or `"float32"`, else `ValueError` (gmg's message).
- **Built directly in `dtype`:**
  - every assembled level's block-ELL values (`block_ell(A, b, dtype)`, existing);
  - the transfers (`block_ell(P, b, dtype)`);
  - the Schwarz packed inverses. `schwarz.build(A, blocks, sel, dtype=...)` (keyword: the
    fourth positional parameter is `chunk_bytes`) allocates the buffer in `dtype`, and
    `_invert` casts each chunk and checks finiteness after the cast.
  So no level exists in fp64 on the device.
- **Computed in fp64, then cast:**
  - the node-block Jacobi inverses and their mask (small);
  - the p-level `NodalTransfer` weights;
  - Chebyshev's `lmax` and `ratio`. `lmax_estimate` runs on the stored, fp32-valued
    operator with fp64 vectors, so the estimate is of the operator actually applied.
- **Stays fp64:**
  - the fine level's `APOperator` (`levels[0][0]` and `levels[0][1].A`; the operator LOBPCG
    shares, through the cast above);
  - the coarsest Cholesky factor (`Cholesky` casts its input and output);
  - `GMG.free`.
- `GMG(..., dtype="float32")` casts the residual at the cycle's entry and the result back.
- **No shared helper:** `amg` sets the dtype in its two level loops (`_level` and
  `_shell_amg`). `gmg._to32` (cast at the end, its fine level skipped by index) is unchanged.
- **`return_aggregates` / `aggregates=`:** the topology is dtype-independent. One caveat: the
  hex `Hex8Pattern` holds the p-level's `NodalTransfer` on the device in fp64. A fp32 cycle
  keeps its own fp32 copy, so while the aggregates are held (band mode) the fp64 transfer is
  extra: 16 B per DOF on hex20, 49 on hex27, about 49 MB at 1M hex27. Accepted and logged;
  the docs state it.
- **`pmg`** (test-only reference, `open_items.md` "`pmg` kept alongside `amg`") gets no
  dtype. The roadmap's "p-MG" is `amg`'s p-level, which is covered. Logged.

### `solve_modes` (`eigen/api.py`)

- **`resolve_precond`:** `precond_dtype="float32"` is accepted for `"amg"` too; `"jacobi"`
  still raises ("precond_dtype 'float32' needs precond 'gmg' or 'amg'").
- **Where fp32 applies** (as for GMG): LOBPCG's preconditioner, and so the polish, which runs
  PCG with it. These stay fp64:
  - the σ pre-estimate's builds;
  - band counts and slices (`make` is fp64; `docs/eigensolvers.md:82`).
  fp32 for the pre-estimate would gain nothing: the setup peak is fp64 either way.
- **`RESERVED["amg"]`** gains `"dtype"` (set through `precond_dtype`, as for GMG).
- **Docstring and `levers()`:** the `solve_modes` docstring (`:337–338`) names AMG, and
  `levers()` names `precond_dtype='float32'` for AMG as for GMG (with its $C_{max}$).
- **`_models` / `solve_modes_memory`:** pass `dtype=precond_dtype` to `amg_memory`. Today's
  `pc64` (`:477`), the fp64 model for the band-count / slice multigrid, is extended to AMG.

### σ-round reuse (`eigen/api.py`)

- **Mechanism:** `make` keeps the aggregates of the first AMG build in a closure variable,
  that build made with `return_aggregates=True`. While the variable is set, every later AMG
  build passes `aggregates=`:
  - the σ rounds of `_precond_sigma`;
  - LOBPCG's build;
  - the band counts.
  `lobpcg_precond`'s `return_aggregates=True` special case goes away.
- **Exact:** reuse keeps the topology (pattern, tentative prolongators and candidates, fibers,
  frames) and recomputes every value-dependent part ($A_b$, the smoothed $P$, the Galerkin
  products, the smoothers). Aggregation does not depend on σ:
  - hex: θ = 0, the structural pattern only;
  - shell: fibers and frames from the geometry and `Ab.indices`.
  A reused build at a new σ equals a fresh one (PR 13b's and 18a's reuse tests; bitwise in
  review round 1).
- **Lifetime:** the variable is cleared:
  - **plain solves:** right after LOBPCG's preconditioner is built, so the host holds no
    aggregates during the iterations;
  - **band mode:** where `aggs` is released today (`:555`, before the slices after a
    capacity stop, and `:589`, after the counts), so the slices never see them and
    `si_memory` / `_slice_capacity` need no new term;
  - **the no-chunk slice path** (`:594–611`): after its σ pre-estimate, before `si_band`, so
    reuse there covers the σ rounds only;
  - **counts:** they keep receiving the aggregates as today. The probe budget already
    subtracts `_nbytes(aggs)`.
- **SingularShiftError:** `build(0.0)` raises before any work (`pmg._check`). The first
  successful build, at the `sigma_estimate` bound, then makes the aggregates.
- **Memory:** in plain mode the σ rounds hold the aggregates during a reused build. That
  build skips aggregation, fibers and frames, which the fresh build holds as transients, so
  it stays inside the fresh `build` (15d's test 11 checks the count-time rebuild with the
  aggregates held for hex). 18c checks it for the shell family (test 9).

### `amg_memory(mesh, m, max_coarse=1000, degree=2, dtype="float64")` (`memory.py`)

- **Exact parts follow the dtype:**
  - hex p-level:
    - free mask 24 B per node, fp64 (`GMG.free`);
    - node-block Jacobi inverses and mask: $72 + 24$ B per node fp64, $36 + 12$ B fp32
      (120 → 72 B per node in all);
    - the `NodalTransfer` tables: 12 → 8 B per entry;
  - shell: `schwarz.nbytes(sizes, dtype)`;
  - the coarsest factor stays $8\,n_c^2$.
- **Fitted parts, fp32: one ratio per term (`FP32` in `memory.py`).** Each fp32 term is
  $\rho \times$ its fp64 term. $\rho$ is the maximum, over the fit grid, of measured fp32 /
  fp64 model. That makes it an upper bound on every fit point by construction.
  - **Storage:** $\rho$ per family (hex8, p, shell) on the fitted part, i.e. beyond the
    exact parts and the coarse factor. `AMG_STORAGE[fam]` times $n_c$; for the shell the
    `SHELL_STORAGE` sum. Grid: the 13b storage cases (hex20 $8^3$–$44^3$, hex27
    $6^3$–$24^3$, hex8 $12^3$–$68^3$) and 18b's `STORAGE` cases.
  - **Steps:** $\rho$ per key on the whole fp64 term:
    - hex: (hex8, p) × (9-block, low), the `AMG_STEP` term $8n(am + c) + 8mK$; the target
      is the extra over the Jacobi step, as for fp64;
    - shell: 9-block, low, polish, the `SHELL_STEP` term $8n(am + c) + K_0$; the target is
      net of `STEP_FIT`, the polish with the low $f$, as 18b.
  - **Hex polish (new, fp32 only):** `polish` = $\rho_\text{hp} \times$ the fp64 hex 9-block
    term. $\rho_\text{hp}$ is the maximum of the measured polish net of `STEP_FIT`'s low
    $f\cdot ws$, divided by that term. So `lobpcg_memory`'s existing `"polish" in pc`
    branch takes it. fp64 hex keeps no polish term (its polish stays below the step).
  - **Grids:**
    - hex: the 13b step sizes, at $m$ = 6, 15, 25, 60, 120, compile only above 400k DOF;
    - shell: 18b's `FIT` cases and $m$ grid ($m$ = 6–120, 25k–1.58M DOF) plus `EXTRA`.
    The shell measurement takes `shell_solve_study.measure`'s max over no basis and a basis
    of capacity $c\lceil 2m/c\rceil$. The hex one is `tests/helpers.step_bytes` (no basis,
    as the 13b fit), against `jacobi`'s step.
  - **Held out** (compile only, not fitted, each ≤ the fp32 model):
    - hex20 $52^3$ (1.7M DOF) and hex8 $80^3$, at $m$ = 25 and 120;
    - shell stack ×3 $256^2$ and T $96^2$, at $m$ = 120.
  - **Unchanged:**
    - the count terms (`AMG_COUNT`, `SHELL_STEP["count"]`): counts always build fp64;
    - `AMG_LOOP_K`: a loop-buffer allowance, kept as the fp64 upper bound;
    - `build` / `host_build`: computed from the fp64 storage whatever the dtype, because the
      setup runs in fp64 on the host (Findings: Build peak).
- **Gates (as 13b / 18b):**
  - every term ≥ measured on every fit and held-out point;
  - storage within 1.3× on the tight cases;
  - the step's upper ratio gate, model ≤ 1.3× measured, on flat and stack ×3 $128^2$ at
    $m$ = 25 / 120 and on hex20 $24^3$ at $m$ = 25.
  If a key fails the upper gate, that key alone gets a full LP refit with 18b's procedure
  (logged). The ratios are fitted after the implementation, from the benchmark.
- **Hex fp64 at $m$ = 120:** the review measured the fp64 hex low step 0.5–0.9 units of $8n$
  above the line through $m$ = 15 / 25, still inside the +1.5 pad. The fp64 constants stay as
  they are; the fp32 ratio is fitted on the widened grid against them.

### Benchmark: `benchmarks/amg_fp32_study.py` → `benchmarks/results/amg_fp32_study.json`

The file imports the helpers it needs and copies none: `amg_study.acceptance` (gaining
`dtype`), `shell_solve_study.measure` / `solve_peak` / `fit` (gaining `dtype`), and
`tests/helpers.step_bytes`. Parts (`--parts`), each ≤ 1 h, printing one line per case:

| part | time | contents |
|---|---|---|
| `accuracy` | ~10 min | PCG iterations and true residuals, fp64 against fp32, on the Findings cases plus hex27 $12^3$, hex8 $32^3$, stack ×3 $128^2$ and T $48^2$; V-cycle times, warm, best of 5, at $m$ = 25 (report only) |
| `storage` | ~15 min | owned bytes fp32 (float / int) on the storage grid |
| `steps_hex` | ~30 min | compile only, the hex grid and held-out points, fp32 and the Jacobi baseline |
| `steps_shell` | ~40 min | compile only, the shell grid and held-out points, fp32 |
| `fit` | seconds | the ratios and gates, into the JSON |
| `peaks` | ~5 min | fp32 build peaks in fresh subprocesses: 18b's `PEAKS` cases, the 13b setup cases (hex20 $12^3$ / $24^3$, hex27 $12^3$, hex8 $48^3$), and flat $256^2$ rebuilt with the aggregates held; gated ≤ `build` and on the 18b cases within 1.35× |
| `reuse` | ~5 min | fresh against reused build times, and `solve_modes` with build counts and `time["precond"]`, reuse on and off |
| `acceptance` | ~5 min | 1M hex20 with `precond_dtype="float32"`: cold run plus two warm peaks against `solve_modes_memory`; the fp64 numbers from `amg_study.json` beside them |
| `shell_peak` | ~5 min | 18b's 791k plate solve with `precond_dtype="float32"`, synchronous dispatch |
| `reach` | seconds | estimators only: the 10M hex20 and 791k shell $C_{max}$ with and without fp32 AMG (`strategy_study.memory_part` with the lever) |

`acceptance` and `shell_peak` record the JSON for the docs; tests 8 and 9 run the same
configurations, with nothing extra.

## Tests (18c)

Unless marked, the tests go in:
- `tests/test_amg.py` (hex);
- `tests/test_shell_solve.py` (shell);
- `tests/test_api.py` (dispatch).

1. **`_ap_apply` dtype** (`tests/test_operators.py`): `A(x32).dtype == float32`, equal to
   `A(x64)` to $10^{-6}$ relative (measured 3.5e-8); fp64 unchanged bit for bit.
2. **fp32 V-cycle against fp64:**
   - **Cases:** hex8 $8^3$, hex20 $6^3$, hex27 $4^3$, the mixed hex8 + hex20 mesh (with
     explicit BCs, as `test_amg_memory_build_mixed`'s fixture), and the block $8^3$
     (`ssh8` + `hex8`), with tolerances $10^{-5}$ (difference) and $10^{-6}$ (asymmetry).
     The `ssh8` plate $32^2$ at $L/t$ = 10 gets $2\cdot10^{-5}$ for both (measured
     3e-6 / 2e-6; thin shells round more, see test 3).
   - **Checks:**
     - `P32(r)` against `P64(r)`, relative Frobenius difference;
     - energy-normalized asymmetry of `P32`, as PR 11b's GMG test;
     - output dtype fp64;
     - the float leaves are fp32 except the fine `APOperator` (`levels[0][0]`,
       `levels[0][1].A`), `free` and `coarse.L`; the int leaves are unchanged.
   - **Schwarz:** on the shell cases, the fp32 packed inverses equal the fp64 ones cast,
     and the Schwarz apply on an fp32 input equals the fp64 apply to $10^{-6}$ (measured
     ≤ 1.2e-7).
3. **Convergence (roadmap):**
   - **PCG:** the fp32 iteration count is within ±1 of fp64 on hex20 $12^3$ at tol
     $10^{-8}$, and ≤ 1.05 × fp64 + 1 on the `ssh8` plate $64^2$ at $L/t$ = 100 (tol
     $10^{-8}$) and at $L/t$ = 1000 (tol $10^{-6}$). In every case the fp32 true residual is
     ≤ tol (the tolerance is reached).
   - **Solve level** (pre-estimate fp64, LOBPCG and the polish fp32):
     - hex20 $6\times5\times4$ (20 modes) matches `reference_modes` at tol $10^{-10}$ (PR 9
       tolerances, as the fp64 accuracy test);
     - the flat `ssh8` $32^2$ plate at $L/t$ = 1000, tol $10^{-6}$, converges;
     - stack ×3 $32^2$ matches `reference_modes` at tol $10^{-10}$, with no polish pin (fp32
       does not polish there).
4. **Options:**
   - `resolve_precond` accepts AMG fp32 and refuses Jacobi fp32;
   - `amg(dtype="float16")` raises;
   - `precond_options={"dtype": ...}` raises the reserved-key error;
   - the pre-estimate builds are fp64 and LOBPCG's build is fp32 (spy on `api.amg`'s
     `dtype`);
   - an AMG copy of `test_strategy.py::test_fp32_slices`: every count / slice `make` passes
     `dtype` float64, and `_slice_capacity` / `_count_block` receive the fp64 `amg_memory`.
5. **`amg_memory` storage fp32:** ≥ owned bytes on `STORAGE_CASES` and 18b's shell storage
   cases, within 1.3× on the tight cases; the exact parts equal the owned bytes of those
   leaves.
6. **Steps fp32:**
   - the hex step with fp32 AMG ≤ `lobpcg_memory`'s step on `test_amg_step`'s cases;
   - the shell fp32 step, low and polish ≤ the model at $m$ = 6 / 25 / 120 on 18b's
     `STEP_CASES`;
   - `test_hex_polish_below_step` with `dtype="float32"`: the fp32 polish ≤ the model's
     polish term, and that term is what `lobpcg_memory` takes;
   - slow tight pins: the hex20 $24^3$ ratio gate; shell ≤ 1.3× at $128^2$, $m$ = 25 / 120.
7. **`FP32` ratios:** each ratio equals the maximum the benchmark JSON records for its key.
   That pins the constants, as 18b's fit test.
8. **Acceptance (roadmap; `peak`, `slow`, darwin):**
   - `amg_study.acceptance(44, dtype="float32")`;
   - converged, cold run ≤ 15 min (the fp64 test's budget, 10× margin, kept as a comment);
   - `est ≥ 2e9`;
   - the larger warm peak within 20% of `solve_modes_memory(precond_dtype="float32")`;
   - that estimate is below the fp64 one on the same mesh.
   A `test_acceptance_smoke` parametrized by dtype checks the keys at $8^3$, with the
   estimate equal to `solve_modes_memory(precond_dtype=dtype)`.
9. **Shell peaks fp32 (`peak`, `slow`, darwin):**
   - the fp32 build peak in a fresh subprocess ≤ `build` on flat $362^2$ and block $32^3$;
   - the reused build with the aggregates held ≤ `build` on flat $256^2$;
   - the 791k plate solve with `precond_dtype="float32"` (synchronous dispatch), with
     `est ≥ 2e9` and within 20% of `solve_modes_memory`.
10. **σ-round reuse:**
    - `solve_modes` on a clamped hex20 box and on a free `ssh8` plate (σ = 0 refused, so the
      `sigma_estimate` start) runs aggregation once, by a spy on
      `amg.standard_aggregation` / `shell._fibers`;
    - every later build passes `aggregates=`;
    - λ, σ and the iteration counts are `array_equal` to a run with reuse disabled. The
      disabled run wraps `api.amg` so it drops `aggregates=` and still returns a tuple when
      asked.
    - Lifetime: in plain mode a weakref to the `AMGAggregates` is dead after the solve
      (`gc.collect()`); band mode's counts receive the same object.
11. **Benchmark smoke** (`slow`): `amg_fp32_study` parts `accuracy`, `storage`,
    `steps_hex` and `fit` on one tiny case each record their keys.
12. **Existing tests that change:**
    - `tests/test_api.py`: the `precond_dtype` error message.
    - `tests/test_strategy.py`:
      - `test_levers` (`:1111`): the AMG message now names `precond_dtype` (assert its
        $C_{max}$ instead of its absence);
      - `test_memory_no_lobpcg_build` (`:395`): the first σ-round build asks for aggregates
        (assert no build after the pre-estimate instead);
      - `test_capacity`'s `aggs_dead` (`:192`): unchanged; it pins the release before the
        slices.
    - `tests/test_band.py`: the `spy` (`:77–80`) and `no_build` (`:111–121`) fixtures stop
      using `"aggregates" in kw` as "the λ_c count build". They count builds after
      LOBPCG's, by σ value, so `:172–173`, `:258`, `:299`, `:322` and `:465` keep their
      meaning.
    - `tests/test_shell_solve.py::test_band`: unchanged (one `return_aggregates` build).

## Docs (18c)

**Retag rule:** every `PR 18c` / `18d` / `18e` mention in `README.md`, `docs/*.md` (not
`docs/plans/`), `src/`, `tests/` and `benchmarks/*.py` is retagged per the split table, and a
grep before merge checks it. The list below is what round 1 found; `benchmarks/strategy_study.py:227`
stays as history.

- **18c → 18d:**
  - `docs/mesh.md:38` (Morton);
  - `docs/operators.md:204` (`layout`) and `:251` (roofline);
  - `docs/preconditioners.md:570` (`layout`);
  - `docs/boundary_conditions.md:53` (`ndimage.label`);
  - `docs/conventions.md:89` (compile time);
  - ROADMAP `:771` (Pallas "if PR 18c's roofline") and `:785` (the Schwarz scan remedy);
  - the deferred GPU checks "PR 18c: matvec ≥ 50% of roofline" and "XLA kernel temporaries"
    (`:790–791`).
- **Stays 18c:** ROADMAP `:786` ("18c's fp32 Schwarz item").
- **18d → 18e:**
  - ROADMAP `:82`, `:95`, `:97`, `:177`;
  - `docs/eigensolvers.md:128`, `:154`, `:174`, `:180`, `:671`, `:773`, `:814`;
  - every "→ PR 18d" entry in `docs/open_items.md`.
- **18e → 18f:** the CalculiX mentions (ROADMAP `:666–670`, the memory file is outside the
  repo).
- **README `:5`:** "split into 18a–18e" → "18a–18f", with 18c's status.

**fp32 statements that become false:**
- `docs/eigensolvers.md`:
  - `:33` (the `"amg"` bullet's "fp64 only");
  - `:76` and the `precond_dtype` row `:82` (GMG and AMG; LOBPCG and the polish only; slices,
    counts and the pre-estimate fp64; on hex a memory lever only);
  - the levers list.
- `docs/open_items.md`: `:80` ("`gmg`'s and `amg`'s `dtype`") and `:174` ("LOBPCG's GMG or
  AMG").
- `docs/preconditioners.md`:
  - `:362–384`: `amg(...)`'s signature and a `dtype` bullet;
  - `:565`: `amg_memory`'s signature, the fp32 exact terms (72 B per node, 8 B per table
    entry), the `FP32` ratios, the hex fp32 polish term, the build term on fp64 storage,
    and the fit procedure with the measured gates;
  - the AMG and Schwarz sections: an "fp32 V-cycle" paragraph (what is fp32, what stays
    fp64, accuracy and times from the benchmark, the band-mode fp64 transfer copy);
  - reuse: σ rounds;
  - its "## Tests" list (`:621ff`).
- `docs/operators.md:55–58` and `:238`: on `Unstructured` operators `ap_operator` computes in
  fp64 and casts the result to `x.dtype` (PR 18c); the operator itself does not follow the
  dtype. The stale "(PR 13)" goes.
- `docs/eigensolvers.md`: the σ pre-estimate (aggregates built once), the memory-model
  section, the 10M paragraph and table (`:154–162`, from `reach`), and the tests list
  (`:895–980`).

**10M and memory statements:**
- ROADMAP:
  - `:92` ("the 700–1000 target needs PR 18c's fp32 AMG");
  - the memory table's 10M column: Multigrid, Preconditioner workspace, and the Max locked
    modes cell "with fp32 AMG from PR 18c (15d, fp64 AMG: 120 – 260)", replaced by the
    estimator's figure;
  - the 80 GB GPU paragraph (`:181`), with the fp32-AMG totals or "unchanged";
  - `:184`.
- **ROADMAP otherwise:**
  - `:18`: the status line, plus "(fp32 AMG: PR 18c)";
  - `:627` and the new 18c / 18d / 18e / 18f sections;
  - the dependency rows (split section);
  - deferred GPU checks: the fp32 AMG step ratios against the GPU's compiled temporaries.
- **`docs/open_items.md`:**
  - a new "PR 18c decisions without user input" entry: the split, pmg without fp32, the fine
    $A_P$ in fp64, fp32 in LOBPCG only, the build term on fp64 storage, the band-mode fp64
    transfer copy, the 10M reach correction, the reuse gain;
  - resolve `:241` ("18c's fp32 halves the patches", with the measured ratio) and `:255`
    (σ-round reuse, done).
- README: the `precond_dtype` sentence.
- The 18a plan's split table stays as history; this plan's table supersedes it.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **The 10M reach:**
  - The roadmap table's "≈ 700 (stored Ke) – 1000 (on the fly), with fp32 AMG" was an
    extrapolation.
  - The performance review estimates about 5 GB saved at chunk 5: 3.1–3.3 GB of storage
    and 2–2.5 low-step blocks. That is about 120–130 more columns (260 → about 390), not
    600.
  - The table gets the estimator's number. No test criterion depends on it.
- **fp32 polish:** PCG with the fp32 cycle reached the same floor as fp64. If a case
  stagnates, the polish's accept-or-revert keeps the unpolished chunk. Test 3 pins the
  thin-shell case.
- **PCG with a fp32 preconditioner is not flexible:** the cycle is linear, and symmetric only
  to fp32 round-off, which grows on thin shells (2–4e-3 at $L/t$ = 1000). Convergence there
  was unchanged; GMG has worked the same way since PR 11b.
- **The fine level stays fp64:** an fp32 $K_e$ copy would cost half of the operator's
  storage, to save a cast.
- **Ratios instead of a refit:** they waste at most about 1 block per term, against a
  1.3–7-block saving, and they stay upper bounds by construction on the grid. The held-out
  points guard the extrapolation.
- **GPU:** the fp32 ratios come from CPU `memory_analysis`, so they are a deferred GPU check.
