# PR 13c plan: every solver option reachable from `solve_modes`

Status: draft (plan review round 3 pending; log at the end).

User request (2026-10-13, after PR 13b): "Generally make the default settings a robust choice for most problems. But also expose all solver options to the user in some way." PR 13c is inserted before PR 14 so that PRs 14–15 follow the same pattern from the start. It is an API, validation and docs PR; the only numerics change is the Chebyshev loop form (Design 4), which leaves results equal to rounding.

## Facts

`solve_modes(mesh, materials, bcs, n_modes, orientations=None, *, mass, nodal_mass, tol, maxit, seed, chunk, guards, basis_dtype, low_memory, precond, sigma, precond_dtype)` already exposes every `lobpcg` option except `X0`, and the operator's `mass`/`nodal_mass`/`orientations`. Options reachable today only by composing the building blocks by hand:

| component | options not reachable from `solve_modes` | default |
|---|---|---|
| `unstructured` | `mode` ("stored" / "otf"), `path` (scatter / gather), `chunk` (elements per scan step), `range_nodes` | "stored", auto, auto, None |
| `stencil` | `slab` (cells per scan step) | auto |
| `gmg` | `degree`, `coarse_degree`, `ratio` | 2, 4, 30 |
| `amg` | `degree`, `ratio`, `max_coarse` | 2, 30, 1000 |
| `jacobi` | — | — |
| σ pre-estimate (`_precond_sigma`) | LOBPCG tol per round, iterations per round, rounds, the drop factor that triggers a rebuild, `sigma_estimate`'s `maxit` for the fallback start; its LOBPCG `seed` is fixed at 0 | 0.1, 20, 4, 3, 10 |
| `lobpcg` | `X0`: up to the block size $k$ = chunk + guards columns, seeds the first chunk only (other columns random), deflated against the rigid modes at the start (`_start`) | random (seeded) |

Unexposed on purpose (stated in the docs and `open_items.md`): `lobpcg`'s `R` and `precond` (set by `solve_modes`); `amg`'s `nonsingular` (a proof, derived from the rigid-mode count), `aggregates` / `return_aggregates` (state, building blocks); `gmg`'s `dtype` (the top-level `precond_dtype`; the pre-estimate builds stay fp64 for robustness); `pmg` (test-scale, superseded by `amg`); `block_jacobi` / `chebyshev` alone (dev building blocks: without a coarse level they gain nothing, PR 10 h-study).

Value checks today (correctness probe): `degree`, `coarse_degree`, `chunk`, `path`, `range_nodes`, `slab` raise on bad values; `ratio ≤ 1` (gmg, amg), `max_coarse` non-integer or ≤ 0 do not (the latter fails later as "coarsening stalled"), and the σ pre-estimate values have no builder at all (`rounds=0` returns `None` → `TypeError`).

Memory (performance probe, `memory_analysis`, fp64 blocks of $8nm$, $m = 15$): `_chebyshev` unrolls its loop in Python, and XLA keeps each step's temporaries: AMG V-cycle on hex20 $16^3$ 19.1 / 28.6 / 38.0 blocks at degree 2 / 4 / 6 (+4.7 per extra step; the fine level is the matrix-free $A_P$), GMG voxel $32^3$ +2.0 at degree 4. A `lax.fori_loop` over the `degree − 1` steps gives 19.1 / 18.2 / 18.2 with outputs equal to 12 digits. `max_coarse` enters `amg_memory` only through its hard-coded coarsest term $8\min(n_c, 1000)^2$ (0.13 GB at 4000, 3.2 GB at 20 000; +1.3 blocks of temporaries at 4000). `mode="otf"`: the AMG step extra over Jacobi is 8.6 blocks, as in stored mode, so `AMG_STEP` holds.

## Design

### 1. Option dicts and `X0` in `solve_modes`

New keywords, all `None` by default (= today's behaviour, pinned by test 1):
- `operator_options`: forwarded to the operator builder. Allowed keys: `mode`, `path`, `chunk`, `range_nodes` (unstructured meshes); `slab` (voxel meshes). The docs state that `operator_options["chunk"]` (elements per scan step) is unrelated to the top-level `chunk` (modes per locking chunk).
- `precond_options`: forwarded to the preconditioner builder for the pre-estimate builds and the final build. Allowed keys: `degree`, `coarse_degree`, `ratio` (gmg); `degree`, `ratio`, `max_coarse` (amg); none (jacobi).
- `sigma_options`: the pre-estimate's tuning, keys `tol`, `maxit`, `rounds`, `drop`, `estimate_maxit`. `_precond_sigma` gets them as keyword parameters with the current constants as defaults (`tol=0.1, maxit=20, rounds=4, drop=3.0, estimate_maxit=10`) and is called with `**sigma_options`; they are Python constants, so the compiled LOBPCG variants are unchanged. Rejected (`ValueError`) when `sigma` is passed or `precond="jacobi"`, which run no pre-estimate.
- `X0`: forwarded to the final `lobpcg` (shape and semantics as in Facts; with the default single chunk, the previous `V` is a valid warm start).
- `seed` now also seeds the pre-estimate LOBPCG (default 0 = today's value, so defaults are unchanged).

Keys are checked after `resolve_precond` (so `"auto"` takes the gmg keys on voxel meshes and the amg keys otherwise) and before any build (a typo fails before a long operator build). One helper `_check_options(name, opts, allowed)` raises `ValueError` naming the component, the unknown keys and the allowed ones; the message also names the reserved keys (one generic sentence: "solve_modes sets {reserved} itself or through its own keywords"; the reserved sets are the ones test 6 uses). The allowed sets are module constants; a test pins them against the builder signatures (test 6), so a later builder option fails CI until it is exposed or deliberately reserved.

### 2. Value validation

- `chebyshev`: `ratio > 1` (a concrete float; shared by gmg, amg, pmg), else `ValueError`.
- `amg`: `max_coarse` an int ≥ 1.
- `solve_modes`: `sigma_options` values: `tol > 0`; `maxit`, `rounds`, `estimate_maxit` ints ≥ 1; `drop > 1`.
- Other values stay with the builders (already checked).

### 3. `solve_modes_memory`

Takes `operator_options` and `precond_options`, validated with the same helper and key sets:
- `operator_options` go to `unstructured_memory` / `stencil_memory`, which take exactly those keys (`mode="otf"` gives the on-the-fly estimate, the ROADMAP's 10M fallback).
- `precond_options`: `max_coarse` goes to `amg_memory(mesh, m, max_coarse=1000)`, whose coarsest term becomes $8\min(n_c, \max(\texttt{max\_coarse}, 1000))^2$. The floor at the calibration value keeps the estimate an upper bound: below 1000 the large last level is stored as an ELL level with its own smoother and transfers, which the per-corner constant (calibrated at 1000) excludes. Without the floor, the model is 0.94× the owned bytes at hex20 $12^3$ with `max_coarse` 100 (correctness probe); with it, the model is ≥ 1.13× on every measured point, because the owned bytes below 1000 stay at or under those at 1000 (hex20 $12^3$: 18.8 against 21.0 MB). Above 1000, the dense term is loose (4.3× at 3000 on hex20 $12^3$), and the docs state this. With Design 4, `degree` and `coarse_degree` no longer change the V-cycle temporaries, so the default-calibrated constants bound them; `ratio` changes no memory. Remaining approximation, logged: `AMG_STEP` is calibrated at `max_coarse` 1000 (+1.3 blocks of temporaries measured at 4000).
- No `sigma_options` or `X0`: the pre-estimate peaks below the solve, and `X0` is the caller's array (one extra $(n, j)$ block, stated in the docs).

### 4. Chebyshev as a loop (`precond/chebyshev.py`)

`_chebyshev` runs its `degree − 1` steps in `lax.fori_loop` (carry `r, d, x, rho`; same body). Degree 1 skips the loop. Results equal the unrolled form to rounding (test 7); temporaries no longer grow with the degree, which makes `degree` / `coarse_degree` safe to expose against the memory estimate. `amg_study.cost` re-measures the V-cycle cost at 1M hex20 (expected unchanged; reported in the plan's implementation note and `preconditioners.md` if it moves).

The loop changes measured memory, so after the change:
- re-measure the GMG and AMG V-cycle temporaries (`memory_analysis`; default degrees, fp64 and fp32);
- if they moved, recalibrate `gmg_memory`'s workspace constant $c$ and `AMG_STEP`/`GMG_STEP`, and update `preconditioners.md` (the fp32 6.1 against 12.1 blocks, the GMG workspace $c$ = 15 / 7.5, the AMG temporaries 11.74 blocks, the Cost section);
- rerun the two-sided tests that pin these figures: `test_gmg.py` estimate ≤ 1.25× measured (line 667) and the fp32 ≤ 0.75× check (lines 588–615); the ±20 % acceptance peaks in `test_api.py` (line 514) and `test_amg.py` (line 637).

The estimate must stay an upper bound and within those tolerances; the new figures go into the docs.

A while-loop body cannot share buffers with the code outside it. Its n-independent buffers (the block-ELL scan-chunk gathers, about 0.95 MB per column) therefore add a fixed term at non-default degrees. Measured on hex20 $8^3$ with the low-memory step at degree 4: 1.049× the estimate; from $12^3$ up the loop is at most 0.72×. `AMG_STEP`'s K gets about 120k DOF-equivalents for this term, about 0.1 block at 1M, so the bound holds at every degree and size; this is noted in `memory.py` and `open_items.md`. Cost: the defaults are unchanged (AMG 205 against 208 ms, GMG 698 against 703 ms per V-cycle), and degree 4 runs 2–6 % slower with the loop than unrolled. This is stated next to the memory note in `preconditioners.md`.

### 5. Defaults audit (docs)

`docs/eigensolvers.md` gets an "Options" section: one table with every option of `solve_modes` (top-level keywords and dict keys), its component, default, and why it is the robust choice, citing the measurement (e.g. GMG/AMG over Jacobi: 14 against 330 PCG iterations; final pre-estimate σ within 1.001 λ₁ on every tabulated case except the unfiltered binary SIMP field, 390 λ₁; `degree` 2: cost ties with degree 4 (p-MG 14 iterations at degree 2, 9 at degree 4, equal time; `preconditioners.md` Smoother degree), and it is kept as the measured default of PRs 9–13 (re-checked against degree 4 with `amg_study.cost` after Design 4; changed only if degree 4 is measurably faster, which would be logged); `coarse_degree` 4: SIMP 116 → 87, plate 186 → 136; `max_coarse` 1000; `mode="stored"`: 0.196 s against 0.555 s per apply, `"otf"` for memory; tol $10^{-6}$; 9-block step; fp64 basis). Where a non-default is measured better for a class of problems, the table says so (`max_coarse` 2000 on single-layer plates: 15 against 19 iterations, a coarse-mesh-size effect, not a fix; the remedy for thin plates is the solid-shell element of PR 18; `mode="otf"` when the stored operator does not fit, e.g. 10M DOF on one 80 GB GPU, estimated about 60 against 71.5 GB, not measured; fp32 V-cycle on voxel meshes: 51 s against 61 s). No default changes: every current default is the robust option measured in PRs 9–13.

## Files

- `src/jaxmodal/eigen/api.py`: the keywords, key sets, `_check_options`, value checks, `_precond_sigma(op, free, R, build, *, seed=0, tol=0.1, maxit=20, rounds=4, drop=3.0, estimate_maxit=10)`.
- `src/jaxmodal/precond/chebyshev.py`: `fori_loop`, `ratio > 1`.
- `src/jaxmodal/precond/amg.py`: `max_coarse` check.
- `src/jaxmodal/memory.py`: `amg_memory(..., max_coarse=1000)`, `solve_modes_memory` options.
- Tests: `tests/test_api.py`, `tests/test_lobpcg.py` (memory composition), `tests/test_amg.py` (`amg_memory` with `max_coarse`), `tests/test_precond.py` (Chebyshev loop).

## Tests

1. **Defaults and forwarding** (spies on `unstructured`, `stencil`, `gmg`, `amg`, `lobpcg`, `sigma_estimate`): with all options `None`, the recorded args/kwargs of every call equal a snapshot of today's calls (equality, not containment); parametrized over every allowed key of every dict (one non-default value each), the spy shows that key with that value in every pre-estimate build and the final build (precond), the operator build (operator), or the final `lobpcg` (`X0`); `seed=3` reaches the pre-estimate LOBPCG.
2. **Effects against references** (hex20 box $6\times5\times4$, `xmin` clamped, against `reference_modes`, λ to $10^{-8}$, MAC ≥ $1 - 10^{-8}$): `operator_options={"mode": "otf"}`; `precond_options={"max_coarse": 100}` on hex20 $8^3$ ($n_c$ = 2187: `len(P.levels)` of the spied return is 2 at the default, 3 at 100; measured) with the same modes; GMG `{"degree": 3, "coarse_degree": 2}` on a voxel box; `X0` = the converged `V` converges in ≤ 2 iterations (default start: about 15), clamped and free (k = 6); an `X0` wider than the block size raises.
3. **`sigma_options` effects**, on PR 11b's scripted harness (`test_sigma_rounds`: monkeypatched `sigma_estimate`/`lobpcg`, a build that refuses σ = 0): probes `[30, 9, 2, 0.5]` with `{"rounds": 2}` build `[100, 30]` and return 9; the same probes with `{"drop": 10}` stop after the first probe (30 > 100/10); `{"drop": 2}` on `[40, 25]` builds `[100, 40]` and returns 25, where the default builds `[100]` and returns 40; the harness records `tol` and `maxit` of every LOBPCG round and `maxit` of `sigma_estimate` (`estimate_maxit`). On a real mesh: `{"maxit": 1}` gives `sigma_iterations == 1` on the clamped box (default 4); a free hex20 box with `{"estimate_maxit": 3}` passes `maxit=3` to `sigma_estimate` (spy).
4. **Validation**: an unknown key in each dict raises `ValueError` naming it and listing the allowed keys; `slab` on an unstructured mesh, `max_coarse` with gmg, any key with jacobi, `mass` in `operator_options` (message points to the keyword), `sigma_options` with `sigma=` or `precond="jacobi"` raise; bad values: `ratio` 1 and −1 (gmg, amg), `max_coarse` 0 and 2.5, each `sigma_options` key at its invalid boundary (`tol` 0, `maxit`/`rounds`/`estimate_maxit` 0, `drop` 1) raise `ValueError` matching the key.
5. **`solve_modes_memory`**: `operator_options={"mode": "otf"}` equals the composition with `unstructured_memory(..., mode="otf")`, and the measured AMG step with otf (hex20 $8^3$) is within it; `slab` on a voxel mesh likewise; `amg_memory`'s storage ≥ the owned device bytes (leaves de-duplicated, operator excluded) at `max_coarse` 100, 300, 1000 and 3000 on hex20 $12^3$, hex20 $8^3$ and hex8 $16^3$ (100 and 300 fail without the floor of Design 3), and its coarse term follows $\max(\texttt{max\_coarse}, 1000)$; `{"degree": 4}` is accepted, leaves the estimate unchanged, and the measured AMG step at degree 4 is within it, for the 9-block and low-memory steps, at hex20 $12^3$ (0.57 / 0.65 measured) and at hex20 $8^3$ (1.05 before the K term; covers the fixed loop term); unknown keys raise.
6. **Option sets pinned to signatures**: for `unstructured`, `stencil`, `gmg`, `amg`, `jacobi`, `lobpcg`, `_precond_sigma`: the keyword parameters of `inspect.signature` minus an explicit reserved set (positional inputs, top-level keywords, `dtype`, `nonsingular`, `aggregates`, `return_aggregates`, the `lobpcg` keywords `solve_modes` forwards) equal the allowed set.
7. **Chebyshev loop**: the `fori_loop` form equals an unrolled NumPy reference to $10^{-14}$ relative (degrees 1–5, block Jacobi on hex20 $4^3$); V-cycle temporaries (`memory_analysis`) at degree 4 ≤ those at degree 2 (AMG hex20 $16^3$: 18.2 against 19.1 blocks; GMG voxel $16^3$; at $8^3$ the loop's fixed buffers dominate the block units); `chebyshev(..., ratio=1.0)` raises; the fp32 GMG V-cycle output stays fp32 and within $10^{-5}$ of the unrolled form (the loop carry keeps the vector dtype).

## Docs

- `docs/eigensolvers.md`: the "Options" section (Design 5) with an example; the "Public API" signature and the `solve_modes_memory` signature updated; X0 semantics; the tests list.
- `docs/preconditioners.md` Smoother degree (line 328): reconcile with the Options table (cost tie, why degree 2).
- `docs/operators.md` (`mode`, `path`, `chunk`, `range_nodes`, `slab`) and `docs/preconditioners.md` (`degree`, `coarse_degree`, `ratio`, `max_coarse`; the Chebyshev loop note and memory consequence): one line each pointing to `eigensolvers.md#options`.
- `README.md`: one small, fast example (a $4^3$ box; `tests/test_readme.py` runs it) with `operator_options` / `precond_options`.
- `docs/ROADMAP.md`: a `### PR 13c` section (user request, inserted before PR 14), a row in the dependency table (13c depends on 13b), a sentence in PRs 14–15 that their new options follow the same pattern, the 13b deferred GPU item's otf fallback written as `operator_options={"mode": "otf"}`, and the status line.
- `docs/open_items.md`, decisions without user input: option dicts rather than flat keywords (the `chunk` clash; a stable signature); the unexposed options and why (Facts); `AMG_STEP` calibrated at `max_coarse` 1000; the coarse term floored at 1000 (loose above it).
- The `solve_modes` and `solve_modes_memory` docstrings.

## Not covered

- Passing a pre-built operator or preconditioner object to `solve_modes` (the building blocks compose; revisit if a user needs it).
- New defaults: none change.
- `solve_modes_memory` with `sigma_options` / `X0` (Design 3).

## Review log

Round 1 (5 reviewers, none satisfied; blocking from tests: two `sigma_options` tests could not fail): `rounds`/`drop`/`tol`/`maxit`/`estimate_maxit` now tested on the scripted harness plus real-mesh effects; option sets pinned to builder signatures; value validation added (`ratio`, `max_coarse`, `sigma_options`; the claim that builders validate everything was false); Chebyshev as `fori_loop` so the degree no longer grows memory (measured +4.7 blocks per step unrolled); `max_coarse` modelled in `amg_memory`; `X0` semantics stated and tested (first chunk, ≤ block size, deflated); `seed` forwarded to the pre-estimate; docs list completed (operators.md, preconditioners.md, both signatures, ROADMAP section and dependency row); defaults table cites final σ values and the measured better non-defaults.

Round 2 (simplicity and tests satisfied; docs, correctness and performance not): `amg_memory`'s coarse term floored at `max_coarse` 1000 (otherwise 0.94× owned bytes at 100); test 5 checks 100–3000 on three meshes; Design 4 re-measures and recalibrates the V-cycle memory constants and lists the two-sided tests it can move; the degree-2 rationale is a cost tie, not least work; fp32 GMG loop check in test 7; generic reserved-key message; drop-2 probes `[40, 25]`; plate `max_coarse` and 10M otf worded as a mesh-size effect and an estimate; Chebyshev degree checks moved to $12^3$–$16^3$; `AMG_STEP` K gets the loop's fixed buffers (step at degree 4 was 1.05× the estimate on hex20 $8^3$); the 2–6 % degree-4 slowdown is documented.
