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

## The PR 18c split (decision without user input, logged)

The roadmap's PR 18c holds six independent items and, through the 18a plan
(`docs/plans/pr-18a-shell-amg.md:102–119`), about fifteen performance follow-ups. 18b showed
that one estimator calibration alone takes many review rounds. So 18c is split along the
code it touches, as 18a split PR 18:

| part | branch | contents |
|---|---|---|
| 18c | `pr-18c-fp32-amg` | fp32 AMG (hex and solid-shell, Schwarz inverses included), its `amg_memory` model and validation (convergence, the 1M hex20 fp32 acceptance, the 791k shell fp32 peak), `precond_dtype="float32"` with AMG in `solve_modes`, σ-round reuse of the AMG aggregates (shell and hex), the 10M reach re-derived with fp32 AMG |
| 18d | `pr-18d-kernels` | Morton reordering and reordering tuning; kernel fusion checks (HLO, report only); the compile-time benchmark; the roofline tests (STREAM); `docs/performance.md`; the non-band follow-ups: `assembled._chunked` pipelining, caching `layout` for the gather path, `_amg_scan` keeping the mesh alive, `gmg_memory` recomputed per search step, the Metal `_default_budget`, $P_0^\top$ as a scatter over $P_0$'s ELL, 1.25× ELL width classes, a public `chunk_bytes` if the GPU check needs it |
| 18e | `pr-18e-band` | the old 18d (band follow-ups, `COUNT_STEP` refit), plus the band-path follow-ups from the old 18c: the refill through `device_put(V[:, i:i+c].T).T`, the verify skips before the gather (`open_items.md:167`, `:287`), merge and count copies (`:288`), host transposes and small compiles (`:289`) |
| 18f | `pr-18f-calculix` | the old 18e (CalculiX comparison) |

The band-path follow-ups go with the band PR because both rewrite the same functions
(`si_band`, `si_checked`, the LOBPCG growth). The dependency table gains 18d ← 5, 6, 18c
(Morton reorders the operators 18c's tests use; no code dependency, ordering only), 18e ←
14b, 15d, 18b, 18f ← 18e. 18c ← 11, 12, 13, 18b (unchanged).

## Goal (18c)

1. `amg(..., dtype="float32")` runs the V-cycle in fp32 on every AMG path (hex8, the hex20 /
   hex27 p-level, solid shell, `ssh8` + `hex8`), with the same convergence as fp64.
2. `solve_modes(precond="amg", precond_dtype="float32")` (and `"auto"` resolving to AMG) uses
   it for LOBPCG's preconditioner and the polish; the memory model follows.
3. `solve_modes` builds the AMG aggregates once per call and reuses them in every later AMG
   build (σ rounds, LOBPCG's build, band counts and slices).

## Findings (prototypes, CPU fp64 / fp32, M3 Max)

### Convergence (`s18c/proto.py`; fp32 levels by casting a fp64 build)

PCG on $A_P$ (fp64) with the V-cycle, 4 masked random columns, max iterations, `amg` defaults:

| case | DOF | tol $10^{-8}$: fp64 / fp32 | tol $10^{-10}$: fp64 / fp32 |
|---|---|---|---|
| hex20 $16^3$, clamped, σ = 3e5 | 56,355 | 14 / 14 | 17 / 17 |
| `ssh8` plate $128^2$, $L/t$ = 10, two edges clamped, σ = 1 | 99,846 | 16 / 16 | 20 / 20 |
| `ssh8` plate $128^2$, $L/t$ = 1000 | 99,846 | 55–58 / 55–58 | 82–84 / 81–84 (both stop at 1.5–1.7e-8: the shell's round-off floor) |

The fp32 cycle is a fixed linear operator up to fp32 round-off. PCG is not flexible, yet it
showed no loss of accuracy down to the fp64 floor.

### Storage (owned device bytes without the operator, float + int)

| case | fp64 | fp32 |
|---|---|---|
| hex20 $16^3$ | 25 + 6 MB | 13 + 6 MB |
| plate $128^2$ | 175 + 5 MB | 88 + 5 MB |

Float bytes halve except the fp64 parts (the free mask, the coarse Cholesky factor). The hex20
AMG is small next to its stored-Ke operator, which stays fp64 because LOBPCG shares it.

### LOBPCG step (`s18c/steps.py`; extra over the Jacobi step, in blocks of $8nm$)

| case | $m$ | 9-block fp64 / fp32 | low fp64 / fp32 |
|---|---|---|---|
| hex20 $12^3$ | 15 / 25 / 60 | 10.2 / 9.0 / 7.3 → 7.4 / 6.5 / 5.0 | 13.5 / 12.3 / 11.2 → 11.4 / 10.5 / 9.7 |
| hex20 $24^3$ | 15 / 25 / 60 | 8.4 / 7.3 / 6.3 → 5.8 / 5.0 / 4.3 | 12.1 / 11.6 / 10.3 → 9.8 / 9.4 / 9.0 |
| hex8 $32^3$ | 15 / 25 / 60 | 13.3 / 10.7 / 8.3 → 8.2 / 6.9 / 5.7 | 17.3 / 14.5 / 12.1 → 12.2 / 10.8 / 9.6 |
| plate $128^2$ | 15 / 25 / 60 | 16.2 / 14.0 / 12.0 → 9.0 / 8.0 / 7.1 | 20.2 / 18.0 / 16.0 → 13.5 / 12.5 / 11.6 |

fp32 saves 2–7 blocks. The fp64 constants would stay upper bounds, but they would waste this
saving, which is the point of the option. So the step terms are refitted (below).

### Reuse (`s18c/reuse.py`; warm, one build each)

| case | DOF | fresh | with `aggregates=` |
|---|---|---|---|
| plate $128^2$ / $256^2$ / $362^2$, $L/t$ = 1000 | 100k / 396k / 791k | 1.11 / 4.52 / 9.80 s | 1.02 / 4.12 / 8.86 s |
| stack ×3 $128^2$ | 200k | 2.30 s | 2.08 s |
| hex20 $24^3$ / $36^3$ | 182k / 596k | 0.50 / 1.66 s | 0.45 / 1.51 s |

Reuse saves about 10% of each later build. The Galerkin products, the Schwarz inversion and the
export dominate. Small, but it is a few lines and exact (see Design), so it ships, and the
docs state the measured gain.

## Design (18c)

### `_ap_apply` follows the vector's dtype (`operators/constraints.py`)

`_ap_apply` casts its result to `x.dtype`. The unstructured applies compute in fp64 (the stored
$K_e$ and the otf constants are fp64), so an fp32 vector gives an fp64 product. Chebyshev's
`fori_loop` carry then changes dtype and fails to trace. With the cast, fp32 in gives fp32 out.
The work is the same as fp64, and the operator gets no fp32 copy. fp64 callers are unchanged
(the cast is a no-op). The voxel `Stencil` already follows the vector's dtype.

### `amg(..., dtype="float64")` (`precond/amg.py`, `precond/schwarz.py`)

- **Validation:** `dtype` is `"float64"` or `"float32"`, else `ValueError` (gmg's message).
- **What runs in fp32** (as PR 11b's GMG `_to32`):
  - every assembled level's block-ELL values (`block_ell(..., dtype)`);
  - the smoother data: node-block Jacobi inverses and their mask, Schwarz packed inverses,
    Chebyshev's `lmax` and `ratio`;
  - the transfers' float leaves (`ELLTransfer` values, the p-level `NodalTransfer` weights).
- **What stays fp64:**
  - the fine $A_P$ (the operator LOBPCG shares, through the cast above);
  - the coarsest Cholesky factor (`Cholesky` casts its input and output);
  - `GMG.free`.
  `GMG.dtype = "float32"` casts the residual at the cycle's entry and the result back.
- **Build order:** the setup runs in fp64 as now (pyamg, the Galerkin products, the inverses,
  the `lmax` estimates), and each level is cast as soon as it is built. So at most one level's
  fp64 device copy is alive next to the fp32 hierarchy. The exception is the Schwarz
  inverses: `schwarz.build(A, blocks, sel, dtype)` allocates its packed buffer in `dtype`, and
  `_invert` casts each chunk before writing it, so the largest level never exists in fp64.
  `schwarz.nbytes(sizes, dtype)` already takes the dtype.
- **One helper:** `gmg._to32` is generalized to a per-level cast (`_cast_level(A, S, dtype)`),
  used by `gmg` (all levels at the end, unchanged behaviour) and by `amg` (level by level).
- **`return_aggregates` / `aggregates=`:** dtype-independent (host topology). Aggregates from a
  fp64 build serve a fp32 one and the other way round.
- **`pmg`** (test-only reference, `open_items.md` "`pmg` kept alongside `amg`") gets no
  dtype. The roadmap's "p-MG" is `amg`'s p-level, which is covered. Logged.

### `solve_modes` (`eigen/api.py`)

- **`resolve_precond`:** `precond_dtype="float32"` is accepted for `"amg"` too; `"jacobi"`
  still raises ("precond_dtype 'float32' needs precond 'gmg' or 'amg'").
- **Where fp32 applies (as GMG today):** LOBPCG's preconditioner, and so the polish, which
  runs PCG with it. These stay fp64:
  - the σ pre-estimate's builds ("the pre-estimate builds stay fp64 for robustness", PR 13c);
  - band counts and slices (`make` is fp64; `docs/eigensolvers.md:82`).
- **`RESERVED["amg"]`** gains `"dtype"` (set through `precond_dtype`, as for GMG).
- **`levers()`:** names `precond_dtype='float32'` for AMG as for GMG (with its $C_{max}$).
- **`_models` / `solve_modes_memory`:** pass `dtype=precond_dtype` to `amg_memory`, and a fp64
  model for the band-count / slice multigrid (today's `pc64`, extended to AMG).

### σ-round reuse (`eigen/api.py`)

`make` holds the aggregates of the first AMG build in the call, made with
`return_aggregates=True`. Every later AMG build passes `aggregates=` instead: the σ rounds of
`_precond_sigma`, LOBPCG's build, the band counts and the slices.

- **Exact:** reuse keeps the topology (pattern, tentative prolongators and candidates, fibers,
  frames). Every value-dependent part is recomputed: $A_b$, the smoothed $P$, the Galerkin
  products and the smoothers. Aggregation does not depend on σ:
  - hex: θ = 0, so every connection is strong;
  - shell: fibers and frames come from the geometry.
  So a reused build at a new σ equals a fresh one (PR 13b's and 18a's reuse tests,
  $10^{-12}$).
- **Lifetime:**
  - Plain solves drop the aggregates once LOBPCG's preconditioner is built, so the host
    holds none during the iterations.
  - Band mode keeps them as today. Its counts and slices already reuse them, and they are
    counted in the probe budget (`_nbytes(aggs)`).
  - `lobpcg_precond`'s `return_aggregates=True` special case goes away: `make` holds the
    aggregates.
- **SingularShiftError:** `build(0.0)` raises before it returns aggregates. The first
  successful build (at the `sigma_estimate` bound) then makes them.
- **Memory:** while the aggregates are held, a later build allocates the reused build's
  peak. That build skips aggregation, fibers and frames, which are transients of the fresh
  build. 14b's test 11 checks this for hex (count-time rebuild with the aggregates held,
  within `build`). 18c adds the same check for the shell family (test 9).

### `amg_memory(mesh, m, max_coarse=1000, degree=2, dtype="float64")` (`memory.py`)

- **The exact parts follow the dtype:**
  - the hex p-level: free mask (fp64, 24 B per node), node-block Jacobi inverses and mask
    ($72 + 24$ B fp64 → $36 + 12$ B fp32 per node), and the `NodalTransfer` tables
    (int32 + float: 12 → 8 B per entry);
  - the shell family: `schwarz.nbytes(sizes, dtype)`;
  - the coarsest factor stays $8\,n_c^2$.
- **Fitted parts, fp32 constants:** fitted with the same procedure as each fp64 constant,
  on the same cases.
  - **Hex storage:** `AMG_STORAGE` per corner DOF. The fp32 value is the maximum over the
    13b cases (hex20 $8^3$–$44^3$, hex27 $6^3$–$24^3$, hex8 $12^3$–$68^3$) of the fp32
    owned bytes, minus the exact parts, per corner DOF, plus 2–4% as for fp64.
  - **Shell storage:** `SHELL_STORAGE` (per shell DOF, per fiber group, per hex-only DOF).
    Refitted as in 18b: an upper envelope of the fp32 owned bytes beyond the exact
    `schwarz.nbytes`, on 18b's `STORAGE` cases.
  - **Hex step:** `AMG_STEP[(fam, step, "float32")]`. The extra over the Jacobi step, $8n(am +
    c) + 8mK$, as the minimal linear-programming envelope of `memory_analysis` on the 13b
    step cases at $m$ = 15 and 25 (hex20 $8^3$–$44^3$, hex27 $6^3$–$24^3$, hex8 $12^3$–$68^3$;
    compile only above 400k DOF), plus 1.5 in $c$.
  - **Shell step:** `SHELL_STEP[(key, "float32")]` for `"9-block"`, `"low"` and `"polish"`.
    The term is $8n(am + c) + K_0$, fitted to `memory_analysis` minus `lobpcg_memory`'s
    `STEP_FIT` part (the polish with the low $f$). It is an LP upper envelope with $c, K_0 \ge
    0$, on 18b's `FIT` cases and $m$ grid ($m$ = 6–120, 25k–1.58M DOF, compile only), plus
    18b's `EXTRA` points, rounded up with $c$ + 0.5.
  - **Unchanged:**
    - the count terms (`AMG_COUNT`, `SHELL_STEP["count"]`), because counts always build
      fp64;
    - `AMG_LOOP_K`, a loop-buffer allowance; the fp64 value is kept as an upper bound;
    - `build` / `host_build`: the setup runs in fp64 on the host. The shell build term
      contains `storage`, which becomes the fp32 storage; test 9 checks that the fp32 build
      peak stays inside it.
- **Gates (as 13b / 18b):**
  - every fitted term ≥ measured on its fit points;
  - storage ≥ owned bytes on every case, with the tight cases within 1.3×;
  - the step's upper gate model ≤ 1.3× measured at large $n$ (18b's ratio gate) on flat and
    stack ×3 $128^2$, and on hex20 $24^3$ at $m$ = 25.
  The constants are fitted after the implementation, from the benchmark. The plan fixes
  the procedure, not the numbers (18b lesson).
- **The fit target is what consumes it:**
  - hex steps are added to `STEP_FIT`'s Jacobi-step model;
  - shell steps are net of `STEP_FIT`.
  The benchmark stores, for each point, the exact quantity that `lobpcg_memory` adds.

### Benchmark: `benchmarks/amg_fp32_study.py` → `benchmarks/results/amg_fp32_study.json`

Parts (`--parts`), each ≤ 1 h, printing one line per case:

- **`accuracy`** (about 10 min): PCG iterations, fp64 against fp32, at tol $10^{-8}$ /
  $10^{-10}$ on the prototype's cases plus hex27 $12^3$, hex8 $32^3$, stack ×3 $128^2$ and the
  T junction $48^2$. Also V-cycle time, warm, best of 5, at $m$ = 25 (report only).
- **`storage`** (about 15 min): owned bytes fp32 (float / int) on the 13b and 18b storage cases.
- **`steps`** (about 50 min, compile only): the hex and shell step terms at fp32 on the fit
  grids above. It reuses `shell_solve_study.measure` / `amg_study`'s step helpers with a
  `dtype` argument.
- **`fit`** (seconds): the LP fits and gates (`shell_solve_study.fit`, generalized to take the
  target array), written to the JSON.
- **`reuse`** (about 5 min): fresh against reused build times (the table above) and one
  `solve_modes` call per case, with build counts and `time["precond"]`, before and after.
- **`acceptance`** (about 25 min): `amg_study.acceptance(44, dtype="float32")`, the 1M hex20
  run with `precond_dtype="float32"` (cold run plus two warm peaks against
  `solve_modes_memory(precond_dtype="float32")`), and the fp64 run's numbers next to it from
  `amg_study.json`.
- **`shell_peak`** (about 10 min): 18b's 791k plate solve (`shell_solve_study.solve_peak`)
  with `precond_dtype="float32"`, synchronous dispatch, against `solve_modes_memory`.
- **`reach`** (seconds, estimators only): the 10M hex20 and 791k shell $C_{max}$ with and
  without fp32 AMG, by the strategy study's memory record (`strategy_study.memory_part`
  extended with the lever).

## Tests (18c)

Unless marked, in `tests/test_amg.py` (hex), `tests/test_shell_solve.py` (shell) or
`tests/test_api.py` (dispatch).

1. **`_ap_apply` dtype:** `A(x32).dtype == float32`, equal to `A(x64)` to fp32 round-off
   ($10^{-6}$ relative); fp64 unchanged bit for bit (`tests/test_operators.py`).
2. **fp32 V-cycle against fp64:**
   - hex8 $8^3$, hex20 $6^3$, hex27 $4^3$, the mixed mesh, the `ssh8` plate $32^2$ and the
     `ssh8` + `hex8` block $8^3$;
   - `P32(r)` equals `P64(r)` to $10^{-5}$ relative (Frobenius), output dtype fp64;
   - the float leaves are fp32 except `free` and `coarse.L`; the int leaves are unchanged;
   - symmetric to $10^{-6}$ (energy-normalized, as PR 11b's GMG test).
3. **Convergence (roadmap):** PCG iterations with fp32 within ±1 of fp64 at tol $10^{-8}$ on
   hex20 $12^3$ and the `ssh8` plate $64^2$ at $L/t$ = 1000. Also solve-level checks:
   - `solve_modes(precond_dtype="float32")` on hex20 $6\times5\times4$ (20 modes) matches
     `reference_modes` at tol $10^{-10}$ (PR 9 tolerances, as the fp64 accuracy test);
   - the flat `ssh8` $32^2$ plate at $L/t$ = 1000, tol $10^{-6}$, converges with the switch
     and polish (18b's thin-shell case);
   - one shell accuracy case (stack ×3 $32^2$) matches `reference_modes` at tol $10^{-10}$.
4. **`schwarz.build(dtype)`:** the packed inverses fp32 equal the fp64 ones cast, and the
   apply equals the fp64 apply to $10^{-6}$.
5. **Options:**
   - `resolve_precond` accepts AMG fp32 and refuses Jacobi fp32;
   - `amg(dtype="float16")` raises;
   - `precond_options={"dtype": ...}` raises the reserved-key error;
   - `levers()` names `precond_dtype='float32'` on an AMG mesh (`tests/test_options.py`);
   - the pre-estimate builds are fp64 and LOBPCG's build is fp32 (spy on `amg`'s `dtype`).
6. **`amg_memory` storage fp32:** ≥ owned bytes on `STORAGE_CASES` and 18b's shell storage
   cases, within 1.3× on the tight cases (as the fp64 tests); the exact parts equal the owned
   bytes of those leaves.
7. **Steps fp32:**
   - the hex step with fp32 AMG ≤ `lobpcg_memory`'s step on `test_amg_step`'s cases;
   - the shell fp32 step, low and polish ≤ the model at $m$ = 6 / 25 / 120 on 18b's
     `STEP_CASES`;
   - the slow tight pins: hex20 $24^3$, hex8 $48^3$ within 1 block below the constants; shell
     ≤ 1.3× at $128^2$, $m$ = 25 / 120.
8. **Acceptance (roadmap; `peak`, `slow`, darwin):** `amg_study.acceptance(44,
   dtype="float32")`. Converged, and the larger warm peak within 20% of
   `solve_modes_memory(precond_dtype="float32")`. The time budget is as the fp64 test
   (≤ 15 min cold).
9. **Shell build and solve peaks fp32 (`peak`, `slow`):**
   - the fp32 build peak in a fresh subprocess ≤ `build` on flat $256^2$ and block $32^3$;
   - the reused build with the aggregates held ≤ `build` on flat $256^2$;
   - the 791k plate solve with `precond_dtype="float32"` (synchronous dispatch) within 20% of
     `solve_modes_memory` (the ≥ 2 GB rule).
10. **σ-round reuse:**
    - a `solve_modes` call on a clamped hex20 box and on a free `ssh8` plate (σ = 0 refused,
      so the `sigma_estimate` start) runs aggregation once (a spy on
      `standard_aggregation` / `shell._fibers`), with every later build passing
      `aggregates=`;
    - λ, σ and the iteration counts equal those with reuse disabled (monkeypatched `make`)
      to $10^{-12}$;
    - plain mode holds no aggregates after LOBPCG's build (weakref on the `AMGAggregates`,
      `gc.collect()`);
    - band mode's counts receive the same object.
11. **Existing tests that change:**
    - `test_api`: the `precond_dtype` error message;
    - `test_options`: the reserved keys list;
    - `test_band` / `test_strategy`: the source of the aggregates (`lobpcg_precond` no longer
      takes `return_aggregates`);
    - `test_shell_solve`: the memory tests take a `dtype` parameter where cheap.

## Docs (18c)

- `docs/preconditioners.md`:
  - AMG gets an "fp32 V-cycle" paragraph (what is fp32, what stays fp64, the accuracy and
    timing from the benchmark);
  - the `amg_memory` sections (hex and shell) get the fp32 constants, the fit procedure and
    the measured gates;
  - the Schwarz section: fp32 packed inverses;
  - reuse: σ rounds.
- `docs/eigensolvers.md`:
  - the `precond_dtype` row (GMG and AMG, LOBPCG only, slices and counts fp64);
  - the levers list;
  - the σ pre-estimate (aggregates built once);
  - the memory-model section;
  - the 10M reach paragraph and table (`:154–162`, from the `reach` part).
- `docs/operators.md`: `ap_operator` follows the vector's dtype.
- `docs/ROADMAP.md`:
  - `:18` status;
  - the PR 18 split note (`:627`) and the new 18c / 18d / 18e / 18f sections;
  - the dependency table;
  - the memory table's 10M column (Multigrid, Preconditioner workspace, Max locked modes:
    the "≈ 700–1000 with fp32 AMG" estimate replaced by the estimator's figure);
  - `:184`;
  - deferred GPU checks: the fp32 AMG step constants against the GPU's compiled
    temporaries; retag "PR 18c: matvec ≥ 50% of roofline" and "XLA kernel temporaries" to
    18d.
- `docs/open_items.md`:
  - "PR 18c decisions without user input" (the split, pmg without fp32, the fine $A_P$ in
    fp64, fp32 in LOBPCG only, the 10M reach correction, the reuse gain);
  - retag the old 18c / 18d / 18e mentions (`:161`, `:165`, `:167`, `:183`, `:234`, `:239`,
    `:241`, `:245`, `:255`, `:287–289`) and the "→ PR 18d" entries to 18e.
- README: the status line and the `precond_dtype` sentence.
- The 18a plan's split table stays as history. This plan's table supersedes it.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **The 10M reach.** The roadmap table's "≈ 700 (stored Ke) – 1000 (on the fly), with fp32 AMG"
  was an extrapolation. From the prototype, fp32 AMG saves about 3.5 GB of storage and 2–4
  blocks of step at 10M. That is roughly 100–250 more columns, not 600. The table gets the
  estimator's number. No test criterion depends on it.
- **fp32 polish.** PCG with the fp32 cycle reached the same floor as fp64 in the prototype.
  If a case stagnates in implementation, the polish builds nothing new: it falls back to its
  existing accept-or-revert. Test 3 pins the thin-shell case.
- **PCG with a fp32 preconditioner is not flexible:** the cycle is linear and symmetric only
  to fp32 round-off. Measured harmless here; with GMG it has been so since PR 11b.
- **The fine level stays fp64:** an fp32 $K_e$ copy would cost half of the operator's
  storage, to save a cast.
- **GPU:** the fp32 step constants come from CPU `memory_analysis`, so they are a deferred
  GPU check.
