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
- **Revision 3 (plan review round 2; all 5 not satisfied, none blocking):**
  - **Step terms:**
    - Fitted by 18b's linear-programming procedure, not by ratios. A whole-term ratio
      grows with $n$, because fp32 shrinks the constant more than the slope. Its grid
      maximum would undershoot the held-out points (hex20 low at 1.7M: 0.886 against 0.879
      at the grid top) and overshoot the small-$n$ gate.
    - Storage keeps one ratio per family; that term is proportional to $n_c$.
    - The "0.04–0.10 spread" claim is withdrawn.
  - **Reuse:**
    - A separate `reuse` builder is used only by `_precond_sigma`, LOBPCG's build and the
      counts. `make` and the slices are untouched, so a slice can never create aggregates.
    - Lifetime checks at the entry of LOBPCG's main solve and of `si_band`.
  - **Casts:** `chebyshev()` forces `lmax` and `ratio` to fp64, so `amg` replaces them
    after construction. The p-level (level 0) gets its smoother cast with `A` kept.
    `_level`'s finiteness check runs after the cast.
  - **Hex polish:** measured with `shell_solve_study.polish` in `steps_hex`.
  - **Tests:**
    - `test_band`'s λ_c-build marker, now a `_solve` wrapper;
    - `test_defaults_snapshot` added;
    - fp32 cases in the reuse and band-count tests;
    - the `solve_modes_memory` dtype passthrough;
    - the test 9 upper ratio;
    - markers.
  - **Docs:** the retag lists completed (open_items' 18c entries, the ROADMAP headings,
    `:18`, `:184`, `:627`); citations corrected (`ap_operator` is in
    `boundary_conditions.md`); the `si_slice` fp32 refusal now names AMG.
  - **Dependencies:** 18d ← 5, 6, 11, 13, 15d, 18b; 18f ← 18a–18e. The benchmark parts go
    from 10 to 8.
- **Revision 4 (plan review round 3; simplicity satisfied, tests 2 blocking, others not
  satisfied):**
  - **Test fixes (blocking):**
    - `test_band`'s λ_c-build marker wraps the `build` argument of `api._verify`, which is
      exactly the count build. The earlier `api.lobpcg` / `_solve` flag never fired in band
      mode.
    - Test 9's lower gate moves from block $32^3$ (today's fp64 estimate is already 1.58×
      the fp64 peak there) to block $64^3$, and the 1.35× gate applies only to 18b's gated
      cases.
  - **The p-level `NodalTransfer` stays fp64**, shared with the aggregates, and casts its
    outputs to the vector's dtype. This removes the unmodelled band-mode copy (0.19 / 0.58
    blocks at $m$ = 10 on hex20 / hex27) at the cost of 5 B/DOF.
  - **Fit:**
    - an asymptote row per $m$ at each family's largest fitted $n$, so the envelope bounds
      every larger $n$;
    - a rule for failing held-out points;
    - $m$ = 10 among the held-out points;
    - separate `AMG_STEP32` / `SHELL_STEP32` dicts;
    - `_envelope` called directly.
  - **`lobpcg_precond` removed:** LOBPCG's build is `reuse(...)`.
  - **`reach` dropped:** `strategy_study`'s memory part (estimators only) is re-run with
    fp32 AMG in its ladder, and `acceptance` records the 791k plate's $C_{max}$.
  - **Docs:** the open_items routing (18c / 18d / 18e), `open_items.md:239`,
    `preconditioners.md:422` / `:514`, and citations.
  - **Tests:** test 3's thin case gets a reference; markers; the probe-budget model for
    AMG; the voxel fp32 GMG unchanged; the cast finiteness checks.
  - **Dependencies:** 17 added to 18d's row (`assembled._chunked`).
- **Revision 5 (plan review round 4; simplicity satisfied, the others not, none blocking):**
  - **Casts:** `amg` ends with the existing `gmg._to32` and one finiteness check over the
    cast smoothers. The fine level (p-level, or the all-hex8 level 0 from `_p_level`) never
    passed `_level`'s check, so an fp32 overflow there went unseen. The per-loop casts and
    the p-level special case go.
  - **Fit:**
    - asymptote rows per (series, $m$), not per family; a rule for a non-monotone series;
    - the hex regressor $K/n$;
    - $m$ = 40 held out.
  - **`strategy_study --parts`,** so its memory part re-runs alone and replaces only its
    record.
  - **Tests:**
    - test 2 at `max_coarse=100` (the default gives one level on most cases);
    - the overflow scale relative to the inverses;
    - test 10's spy on "the first build that returns";
    - the band lifetime left to `test_capacity`;
    - stated meshes;
    - `amg_memory` dtype validation;
    - an `"auto"` → Jacobi fp32 refusal row kept.
  - **`levers()`** offers the polish lever wherever the model has a polish term.
  - **Docs:** `open_items.md:182` split; the eigensolvers acceptance table and shell
    section; ROADMAP `:18`'s phrase and `:784`.
- **Revision 6 (plan review round 5; simplicity satisfied; tests and correctness 1 blocking
  each):**
  - **Shell fit (blocking):** the targets rise with $n$, so asymptote rows go only on long
    series, valued by a $1/n$ extrapolation of the two largest $n$ (bounds rising series);
    single-size series get none.
  - **Test 3 (blocking):** the $L/t$ = 1000 plate clamps `xmin` and `xmax`, as measured;
    with `xmin` alone neither dtype converges to $10^{-6}$.
  - **10M:** both accountings; a `hex20_10M_gpu` row with a separate host budget.
    `strategy_study --parts memory` only.
  - **Fit grid:** hex27 $32^3$ fitted, $40^3$ held out; storage checked on the held-out
    builds.
  - **Tests:**
    - the overflow case at the default `max_coarse`;
    - `test_no_survivors` by dtype;
    - test 7 spelled out;
    - tests for `levers()` and `strategy_study`;
    - the aggregation spies replaced by an identity check;
    - the dtype spy folded into the snapshot.
  - **Docs:** the polish-term statements, ROADMAP `:18` / `:184` rewording, the 18e overlap,
    `AMG_LOOP_K`.

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
(`si_band`, `si_checked`, the LOBPCG growth). The later parts are renumbered rather than
appended, so the CalculiX comparison stays last and runs on the finished performance pass.
The dependency table becomes:
- 18c ← 11, 12, 13, 18b (unchanged);
- 18d ← 5, 6, 11, 13, 15d, 17, 18b: Morton touches the operators, and the follow-ups touch
  PR 11 / 13 / 15d / 17 (`assembled._chunked`) / 18a (through 18b) code. There is no code
  dependency on 18c, but both edit
  `precond/amg.py` / `schwarz.py`, so whichever merges second rebases. Likewise 18c and 18e
  both edit `eigen/api.py`'s band path (18c: the count builds' reuse; 18e: `si_band`);
- 18e ← 15d, 18b (the old 18d's row);
- 18f ← 18a–18e (the old "18e ← 18a–18d": CalculiX last);
- 19 ← 17, 18a–18e (was 18a–18d).

`:627`'s "Split into five parts … 18c and 18d after 18b; 18e last" becomes "six parts
(2026-10-19, the PR 18c plan) … 18c, 18d and 18e after 18b; 18f last".

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

fp32 saves 1.3–7 blocks.

The ratio fp32 / fp64 of a whole term is not constant (review round 2, compile only, against
the fp64 models). It grows with $n$: fp32 shrinks the $n$-independent part ($K$, $K_0$) much
more than the slope.
- **Hex, low step:**
  - hex8 $m$ = 25: 0.51 / 0.71 / 0.77 at 47k / 207k / 556k DOF (the same fixed meshes);
  - hex20 $m$ = 120: 0.70 / 0.80 / 0.85; slope ratio 0.90.
- **Shell, flat $128^2$ → $256^2$ at $m$ = 120:** 9-block 0.15 → 0.43, low 0.51 → 0.69,
  polish 0.73 → 0.83.

The hex20 9-block $K$ even rises: 71 MB fp32 against 33 MB fp64 at $m$ = 25. So the step
terms are refitted with their own constant (below), and only storage, which is proportional
to $n_c$, takes a ratio.

**fp64 hex at $m$ = 120 (found in review):** the fp64 hex8 9-block extra is 3.2% above
`AMG_STEP` at $40^3$ (1279 against 1240 MB, about 1% of the whole step). The fp64 hex fit
used $m$ = 15 / 25 only. Logged; the fp64 constants stay (18c does not refit fp64).

The **polish**'s PCG keeps fp64 vectors, so with fp32 AMG it exceeds the fp32 step on hex
(review round 1, totals in blocks):
- hex20 $14^3$: 28.5 against 26.9 at $m$ = 25, 38.1 against 37.2 at $m$ = 8;
- hex8 $24^3$: 27.5 against 27.3 at $m$ = 25.
In fp64 the hex polish stays below the step (`test_hex_polish_below_step`), so hex had no
polish term until now.

With `low_memory` on hex the fp32 polish sets the step at $m \ge 25$ (review round 3): hex20
$28^3$ fp32 polish 20.83 / 20.74 against fp64 low 21.49 / 20.66 blocks at $m$ = 25 / 120. So
there the fp32 step saving is 0–0.7 blocks; about 2 blocks only at small $m$ (2.15 at
hex20 $36^3$, $m$ = 10). The fp64 hex polish is also 0.04–0.8 blocks above the measured fp64
low step. It stays inside the low model's slack (1.1–2 blocks), which nothing pins yet
(below).

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
is a no-op. The only fp32 caller today is the voxel fp32 GMG's fine level (`gmg.py:460`),
whose `Stencil` already returns fp32, so the cast is a no-op there too.

### `amg(..., dtype="float64")` (`precond/amg.py`, `precond/schwarz.py`)

- **Validation:** `dtype` is `"float64"` or `"float32"`, else `ValueError` (gmg's message).
- **Built directly in `dtype`:**
  - every assembled level's block-ELL values (`block_ell(A, b, dtype)`, existing);
  - the transfers (`block_ell(P, b, dtype)`);
  - the Schwarz packed inverses. `schwarz.build(A, blocks, sel, dtype=...)` (keyword: the
    fourth positional parameter is `chunk_bytes`) allocates the buffer in `dtype`, and
    `_invert` casts each chunk and checks finiteness after the cast.
  So no assembled level exists in fp64 on the device. `lmax_estimate` runs on the stored,
  fp32-valued operator with fp64 vectors (the applies cast to the vector's dtype).
- **Cast at the end:** both `amg` paths (hex and `_shell_amg`) end with
  `levels = gmg._to32(levels) if dtype == "float32" else levels`. `_to32` is unchanged:
  - level 0 keeps its `A` (the fine `APOperator`);
  - every level's smoother `precond` (node-block Jacobi inverses and mask, or the Schwarz
    inverses, already fp32 so a no-op), `lmax` and `ratio` become fp32;
  - coarse `A` (already fp32) is a no-op.
  - **Why needed:** `chebyshev()` stores `lmax` / `ratio` as fp64 (`chebyshev.py:97–99`),
    and an fp32 cycle with fp64 scalars fails to trace (scan carry fp32 → fp64; review
    round 2).
  - **Fine level:** the p-level, or level 0 on an all-hex8 mesh, comes from `pmg._p_level`
    (shared with `pmg`), never from `_level`. Its Chebyshev `lmax` is therefore of the fp64
    smoother; the difference is about $10^{-7}$ relative, far below the 1.1 safety factor.
  - **Cost:** the fp64 node-block inverses (72 B per node per level) live until the end of
    the build, small next to storage and inside the host-dominated build peak.
  - **Finiteness:** one check right after `_to32`, over every level's cast smoother leaves.
    It raises `_level`'s message and covers the fine level, which `_level`'s own check never
    sees. `_invert`'s check after its cast stays.
- **Stays fp64:**
  - the fine level's `APOperator` (`levels[0][0]` and `levels[0][1].A`; the operator LOBPCG
    shares, through the cast above);
  - the coarsest Cholesky factor (`Cholesky` casts its input and output);
  - `GMG.free`;
  - the p-level `NodalTransfer` (hex20 / hex27). `__call__` and `restrict` cast `_gather`'s
    result to the input's dtype (`pmg.py:49–53`; a no-op in fp64), as `_ap_apply` does.
    - It is the object the `Hex8Pattern` aggregates already hold, so band mode keeps one
      copy, not two.
    - It costs 15 / 47 B per DOF fp64 on hex20 / hex27 (5 / 16 B more than fp32), small
      next to the fp64 fine apply's traffic.
- `GMG(..., dtype="float32")` casts the residual at the cycle's entry and the result back.
- **`return_aggregates` / `aggregates=`:** the topology is dtype-independent, and an fp32 build
  can reuse aggregates made by an fp64 build (the host aggregation is fp64 either way).
- **`pmg`** (test-only reference, `open_items.md` "`pmg` kept alongside `amg`") gets no
  dtype. The roadmap's "p-MG" is `amg`'s p-level, which is covered. Logged.

### `solve_modes` (`eigen/api.py`)

- **`resolve_precond`:** `precond_dtype="float32"` is accepted for `"amg"` too; `"jacobi"`
  still raises ("precond_dtype 'float32' needs precond 'gmg' or 'amg'").
- **Where fp32 applies** (as for GMG): LOBPCG's preconditioner, and so the polish, which runs
  PCG with it. These stay fp64:
  - the σ pre-estimate's builds;
  - band counts and slices (`make` is fp64; `docs/eigensolvers.md:169`).
  fp32 for the pre-estimate would gain nothing: the setup peak is fp64 either way.
- **`RESERVED["amg"]`** gains `"dtype"` (set through `precond_dtype`, as for GMG).
- **Docstring and `levers()`:** the `solve_modes` docstring (`:337–338`) names AMG, and
  `levers()` names `precond_dtype='float32'` for AMG as for GMG (with its $C_{max}$).
- **`_models` / `solve_modes_memory`:** pass `dtype=precond_dtype` to `amg_memory`. Today's
  `pc64` (`:477`), the fp64 model for the band-count / slice multigrid, is extended to AMG.
- **`amg_memory(dtype=...)`** rejects values other than `"float64"` / `"float32"`, as
  `gmg_memory`.
- **`levers()`'s polish lever** (`:487`) is offered whenever the preconditioner model has a
  `"polish"` term, so shell AMG as today, and now fp32 hex AMG too, where the polish sets
  the low step at $m \ge 25$. It no longer checks for `"ssh8" in mesh.blocks`.

### σ-round reuse (`eigen/api.py`)

- **Mechanism:** `make` is unchanged, and the slices keep calling it, so a slice build never
  asks for aggregates. A small `reuse` builder beside it holds `aggs` in the enclosing
  scope:
  ```python
  def reuse(s, **kw):  # sigma rounds, LOBPCG's build: one aggregation (exact, only values change)
      nonlocal aggs
      if precond != "amg":
          return make(s, **kw)
      if aggs is None:
          P, aggs = make(s, return_aggregates=True, **kw)
          return P
      return make(s, aggregates=aggs, **kw)
  ```
  Who uses what:
  - `_precond_sigma` gets `reuse`;
  - `lobpcg_precond` is removed. LOBPCG's build at both call sites (`:453`, `:526`) is
    `P = reuse(sigma, dtype=precond_dtype)` (`make` ignores `kw` for Jacobi), and band mode
    reads `aggs` directly;
  - the counts keep today's `make(s, aggregates=aggs)` lambda (`:587`).
  So the `return_aggregates=True` special case goes away.
  With a user `sigma` and no σ rounds, plain mode builds aggregates it drops at once (a
  fingerprint and the host tentatives): negligible, kept for one code path.
- **Exact:** reuse keeps the topology (pattern, tentative prolongators and candidates, fibers,
  frames) and recomputes every value-dependent part ($A_b$, the smoothed $P$, the Galerkin
  products, the smoothers). Aggregation does not depend on σ:
  - hex: θ = 0, the structural pattern only;
  - shell: fibers and frames from the geometry and `Ab.indices`.
  A reused build at a new σ equals a fresh one (PR 13b's and 18a's reuse tests; bitwise in
  review round 1).
- **Lifetime:** `aggs = None`, plain lines at:
  - **plain solves:** right after LOBPCG's preconditioner is built, so the host holds no
    aggregates during the iterations;
  - **band mode:** where `aggs` is released today (`:555`, before the slices after a
    capacity stop, and `:589`, after the counts);
  - **the no-chunk slice path** (`:594–611`): right after its `_precond_sigma` returns,
    before `si_band`, so reuse there covers the σ rounds only.
  Since the slices use `make`, they never hold aggregates, and `si_memory` /
  `_slice_capacity` need no new term. The counts keep receiving the aggregates as today, and
  the probe budget already subtracts `_nbytes(aggs)`.
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
    - the `NodalTransfer` tables: 12 B per entry in both (fp64, shared);
  - shell: `schwarz.nbytes(sizes, dtype)`;
  - the coarsest factor stays $8\,n_c^2$.
- **Storage, fp32: one ratio per family** (`STORAGE32 = {"hex8": ..., "p": ..., "shell": ...}`).
  - The fitted part (`AMG_STORAGE[fam]` × $n_c$; for the shell the `SHELL_STORAGE` sum) is
    multiplied by $\rho$. $\rho$ is the maximum over the grid of the fp32 owned bytes
    beyond the exact parts and the coarse factor, divided by the fp64 fitted part.
  - The term is proportional to $n_c$, with no constant, so a ratio is the right form.
  - Grid: the 13b storage cases (hex20 $8^3$–$44^3$, hex27 $6^3$–$24^3$, hex8
    $12^3$–$68^3$) and 18b's `STORAGE` cases.
- **Steps, fp32: refitted by 18b's procedure** (new dicts `AMG_STEP32[(fam, key)]` and
  `SHELL_STEP32[key]`, keys as the fp64 dicts plus `("hex8" | "p", "polish")`; the lookup
  picks the dict by dtype; the fp64 entries are unchanged).
  - The fit is 18b's least-sum upper envelope by linear programming
    (`shell_solve_study._envelope`, called directly; `fit` stays as it is). In blocks of
    $8nm$, with $c, K \ge 0$:
    - hex: $y = a + c/m + K/n$, rows $[1, 1/m, 1/n]$;
    - shell: $y = a + c/m + K_0/(8nm)$, as 18b.
  - **Extrapolation in $n$:** $K \ge 0$ alone does not bound larger $n$. If the LP puts
    more into $K$ than the true constant, the asymptote $a + c/m$ drops below the true one.
    - **Not monotone:** the targets need not fall with $n$. 18b's fp64 shell targets rise
      (flat $m$ = 120 9-block: −3.0, 6.1, 8.8, 9.4, 9.7 blocks from 25k to 1.58M DOF; block
      $m$ ≥ 25 too), and so do the fp32 ones (low 0.51 → 0.69 of fp64 at $128^2$ → $256^2$).
      So neither a row at $n_\text{max}$ nor every point as a row is right (review round 5:
      the latter puts the 25k plate's 61–108 blocks under every $n$).
    - **Asymptote rows, long series only:** one row $a + c/m \ge y_\infty$ per ($m$) on each
      series measured up to large $n$: hex8, hex20, hex27 (now to $32^3$, below), flat (to
      1.58M), block (to 837k).
      - $y_\infty = \max\big(y(n_2),\ (n_2 y_2 - n_1 y_1)/(n_2 - n_1)\big)$, the
        extrapolation in $1/n$ through the two largest $n_1 < n_2$ of that (series, $m$).
        This bounds a rising series too.
      - Single-size series (web, T, the stacks) get none: their small-$n$ constant would
        otherwise go into $a + c/m$ (measured: the 1.3× gates fail at 1.4–1.9×, and 3–4.7× at
        flat $512^2$). The flat and block rows and the held-out points cover them.
    - The result is rounded up with $c$ + 0.5. `fit` reports the rows' slack at 10M per
      family (about 0.3 blocks for hex20, roughly 6 fp32 columns at chunk 5).
    - **fp64 shell (logged, not refitted):** 18b's `SHELL_STEP` has the same unbounded
      extrapolation beyond 1.58M (1.002× at flat $512^2$, $m$ = 120, the target still
      rising). `fit` reports the fp64 $y_\infty$ rows against the fp64 model.
  - **Hex** ((hex8, p) × (9-block, low)): the term is $8n(am + c) + 8mK$; the target is the
    extra over the Jacobi step, as for fp64.
  - **Hex polish (new, fp32 only):** the term is $8n(am + c) + 8mK$, fitted to the measured
    polish net of `STEP_FIT`'s low $f\cdot ws$, as the shell polish. `lobpcg_memory`'s
    existing `"polish" in pc` branch takes it. A separate key, rather than folding the
    polish into the 9-block fit: `low_memory=True` solves take max(low, polish), so a
    polish folded into the 9-block term would be lost there. fp64 hex keeps no polish term
    (its polish stays below the step, `test_hex_polish_below_step`).
  - **Shell** (9-block, low, polish): the term is $8n(am + c) + K_0$; the target is net of
    `STEP_FIT`, the polish with the low $f$, as 18b.
  - **Grids:**
    - hex: the 13b step sizes plus hex27 $32^3$ (about 820k DOF, so hex27's series reaches
      large $n$), at $m$ = 6, 15, 25, 60, 120, compile only above 400k DOF.
      The step is measured with `tests/helpers.step_bytes` (no basis, as the 13b fit),
      against `jacobi`'s step; the polish with `shell_solve_study.polish`.
    - shell: 18b's `FIT` cases and $m$ grid ($m$ = 6–120, 25k–1.58M DOF) plus `EXTRA`,
      with `shell_solve_study.measure` (the max over no basis and a basis of capacity
      $c\lceil 2m/c\rceil$).
  - **Held out** (compile only, not fitted, each ≤ the fp32 model):
    - hex20 $52^3$ (1.7M DOF) and hex8 $80^3$, at $m$ = 10 (the 10M ladder's chunk 5), 25,
      40 (between the grid's 25 and 60: the band default chunk 40 plus guards) and 120;
    - hex27 $40^3$ (about 1.6M DOF, above its grid) at $m$ = 10 and 25;
    - shell stack ×3 $256^2$ and T $96^2$, at $m$ = 120.
    A failing held-out point moves into the fit, the constants are refitted, and a larger
    held-out point of that family is added (logged). The hex held-out builds also record
    the fp32 owned bytes, gated ≤ the storage model (exact parts plus $\rho$ × the fitted
    part), so the storage ratio is checked beyond its grid at no extra cost.
  - **fp64 hex polish against the fp64 low model:** `steps_hex` also records the fp64
    polish at the held-out points, and `fit` reports fp64 low model − fp64 polish
    (`low_memory` fp64 solves rely on it being ≥ 0). If negative, it is logged in
    open_items, not refitted in 18c.
  - **Unchanged:**
    - the count terms (`AMG_COUNT`, `SHELL_STEP["count"]`): counts always build fp64;
    - `AMG_LOOP_K`: a loop-buffer allowance, kept as the fp64 upper bound;
    - `build` / `host_build`: hex `AMG_BUILD` has no storage term; the shell's is computed
      from the fp64 storage whatever the dtype, because the setup runs in fp64 on the host
      (Findings: Build peak).
- **Gates (as 13b / 18b):**
  - every term ≥ measured on every fit and held-out point;
  - storage within 1.3× on the tight cases;
  - the step's upper ratio gate, model ≤ 1.3× measured, on flat and stack ×3 $128^2$ at
    $m$ = 25 / 120 and on hex20 $24^3$ at $m$ = 25;
  - model / measured reported at $m$ = 6, 10 and 120 on the largest hex grid mesh, since
    slack at $m$ = 10 eats into the 10M reach.
  On the hex polish, the fp32 polish must be ≤ `fl*ws` + its term on every grid and
  held-out point. The constants are fitted after the implementation, from the benchmark.
  The plan fixes the procedure, not the numbers.

### Benchmark: `benchmarks/amg_fp32_study.py` → `benchmarks/results/amg_fp32_study.json`

The file imports the helpers it needs and copies none:
- `amg_study.acceptance` (gaining `dtype`);
- `shell_solve_study.measure` / `polish` (they take a built P; the study builds it with
  `dtype`), `build` and `solve_peak` (gaining `dtype`: the build and the estimate), and
  `_envelope`. `shell_solve_study.fit` is not used;
- `tests/helpers.step_bytes`.
 Parts (`--parts`), each ≤ 1 h, printing one line per case:

| part | time | contents |
|---|---|---|
| `accuracy` | ~10 min | PCG iterations and true residuals, fp64 against fp32, on the Findings cases plus hex27 $12^3$, hex8 $32^3$, stack ×3 $128^2$ and T $48^2$; V-cycle times, warm, best of 5, at $m$ = 25; fresh against reused build times on flat $362^2$ and hex20 $36^3$ (report only) |
| `storage` | ~10 min | owned bytes fp32 (float / int) on the storage grid |
| `steps_hex` | ~20 min | compile only, the hex grid and held-out points, fp32 step and polish, the fp64 polish at the held-out points, and the Jacobi baseline |
| `steps_shell` | ~15 min | compile only, the shell grid and held-out points, fp32 |
| `fit` | seconds | the storage ratios, the LP fits and the gates, into the JSON |
| `peaks` | ~5 min | fp32 build peaks in fresh subprocesses: 18b's `PEAKS` cases, the 13b setup cases (hex20 $12^3$ / $24^3$, hex27 $12^3$, hex8 $48^3$), and flat $256^2$ rebuilt (fp64, as the σ rounds) with the aggregates held; gated ≤ `build` everywhere and ≥ `build` / 1.35 on 18b's gated cases (flat $362^2$, block $64^3$) |
| `acceptance` | ~10 min | the 1M hex20 (cold run plus two warm peaks) and 18b's 791k plate (synchronous dispatch) with `precond_dtype="float32"` against `solve_modes_memory`; the fp64 numbers from `amg_study.json` / `shell_solve_study.json` beside them; the 791k plate's band $C_{max}$ with and without fp32 AMG (estimators) |

`acceptance` records the JSON for the docs; tests 8 and 9 run the same configurations, with
nothing extra.

**10M reach:** `strategy_study.ladder` gains `"float32"` in `pcd` for AMG meshes (`:175`,
today voxel only).
- `strategy_study.py` gains `--parts memory`: it loads the existing JSON, replaces only
  `r["memory"]`, and adds `r["memory_commit"]` (the top-level `commit` stays the
  `constants` / `lobpcg` one). Estimators only, seconds; the 15d LOBPCG record (about
  14 min) is not re-measured. The default run is unchanged.
- **Two accountings at 10M (review round 5):** the `hex20_10M` case uses one unified
  76 GB budget (`host_budget=None`, the Mac model; `open_items.md:181`). There each column
  costs about 126 MB (42 MB device basis, 84 MB host), so fp32 AMG gains only about
  30–40 columns (stored, low, fp32 basis, chunk 5: 180 → 200–220). On a real 80 GB GPU
  node the host is separate. So the memory part adds a `hex20_10M_gpu` row with
  `budget=76e9, host_budget=0.8 × 512e9`. There fp64 AMG already reaches about 560
  (stored) / 810 (on the fly), and fp32 AMG about 650–680 / 900–930: the roadmap's
  700–1000 is roughly met. The 512 GB node RAM is an assumption, logged.
- `strategy_study.json`'s memory record and `eigensolvers.md:154–162`, which cite it, then
  get the fp32-AMG $C_{max}$. The target string at `strategy_study.py:227` ("700–1000 with fp32 AMG")
is replaced with that figure.

## Tests (18c)

Unless marked, the tests go in:
- `tests/test_amg.py` (hex);
- `tests/test_shell_solve.py` (shell);
- `tests/test_api.py` (dispatch).

1. **`_ap_apply` dtype** (`tests/test_operators.py`): `A(x32).dtype == float32`, equal to
   `A(x64)` to $10^{-6}$ relative (measured 3.5e-8); fp64 unchanged bit for bit; on a voxel
   `Stencil` the fp32 result equals the uncast apply bitwise (the cast is a no-op there, so
   the fp32 GMG is unchanged). `NodalTransfer` likewise: fp32 in, fp32 out, fp64 unchanged.
2. **fp32 V-cycle against fp64:**
   - **Cases**, all at `max_coarse=100` (the default 1000 leaves hex8 $8^3$, hex27 $4^3$,
     the block and the mixed mesh with no assembled level; 100 gives 2–3 levels, where the
     tolerances were measured): hex8 $8^3$, hex20 $6^3$, hex27 $4^3$, the mixed hex8 + hex20
     mesh (`mixed_mesh` with `orientations={1: R}` as `test_counting`, and a hand-built
     free mask: it has no node sets), and the block $8^3$ (`ssh8` + `hex8`), with tolerances $10^{-5}$ (difference) and $10^{-6}$ (asymmetry).
     The `ssh8` plate $32^2$ (`xmin` clamped) at $L/t$ = 10 gets $2\cdot10^{-5}$ for both (measured
     3e-6 / 2e-6; thin shells round more, see test 3).
   - **Checks:**
     - `P32(r)` against `P64(r)`, relative Frobenius difference;
     - energy-normalized asymmetry of `P32`, as PR 11b's GMG test;
     - output dtype fp64;
     - the float array leaves (`jax.Array` only; `BlockELL.padding` is a Python float) are
       fp32 except the fine `APOperator` (`levels[0][0]`,
       `levels[0][1].A`), the p-level `NodalTransfer`, `free` and `coarse.L`; the int
       leaves are unchanged.
   - **Schwarz:** on the shell cases, the fp32 packed inverses equal the fp64 ones cast
     (this pins the cast inside `_invert`).
   - **Finiteness after the cast:** the operator is scaled relative to itself, by
     $10^{-40}/\max|\mathrm{diag}|$, so the fp64 inverses are finite and above the fp32
     maximum $3.4\cdot10^{38}$ (asserted as a precondition). Then:
     - `schwarz.build(..., dtype="float32")` on the block operator raises;
     - `amg(..., dtype="float32")` on hex8 $8^3$ at the default `max_coarse` (one level, so
       its only smoother is the fine level's, which `_level` never checks) raises from the
       check after `_to32`. At `max_coarse=100` an assembled level would hold fp32
       subnormals that XLA CPU flushes to zero, which tests something else.
   - **No survivors:** `test_amg.py::test_no_survivors` is parametrized by dtype, so no fp64
     level copy or fp64 node-block inverse outlives an fp32 build.
3. **Convergence (roadmap):**
   - **PCG:** the fp32 iteration count is within ±1 of fp64 on hex20 $12^3$ at tol
     $10^{-8}$, and ≤ 1.05 × fp64 + 1 on the `ssh8` plate $64^2$ (σ = 1) at $L/t$ = 100
     (`xmin` clamped, tol $10^{-8}$; measured 18–19 for both) and at $L/t$ = 1000 (`xmin`
     and `xmax` clamped, as the Findings, tol $10^{-6}$; with `xmin` alone both dtypes
     stagnate at 2–3e-6). In every case both runs
     report `info["converged"]` and the fp32 true residual is ≤ tol.
   - **Solve level** (`slow`, as their fp64 counterparts `test_accuracy` /
     `test_polish_thin`; pre-estimate fp64, LOBPCG and the polish fp32):
     - hex20 $6\times5\times4$ (20 modes) matches `reference_modes` at tol $10^{-10}$ (PR 9
       tolerances, as the fp64 accuracy test);
     - the flat `ssh8` $32^2$ plate at $L/t$ = 1000, tol $10^{-6}$: `check_lam` against
       `reference_modes` and the residual thresholds, as `test_polish_thin`;
     - stack ×3 $32^2$ matches `reference_modes` at tol $10^{-10}$, with no polish pin (fp32
       does not polish there).
4. **Options:**
   - `resolve_precond` accepts AMG fp32 and refuses Jacobi fp32;
   - `amg(dtype="float16")` and `amg_memory(..., dtype="float16")` raise;
   - `tests/test_api.py`'s `bad` list keeps a refusal: `"auto"` on an `ssh8` + `hex20` mesh
     (resolves to Jacobi) with `precond_dtype="float32"` raises;
   - `precond_options={"dtype": ...}` raises the reserved-key error;
   - the pre-estimate builds are fp64 and LOBPCG's build is fp32: a `precond_dtype="float32"`
     case beside `test_defaults_snapshot` (test 12), not a separate spy;
   - `levers()`: a fp32 hex20 band under a tight budget warns with
     `polish_options={'on': False}`; fp64 hex does not (beside `test_polish_lever`);
   - `strategy_study` (`tests/test_strategy.py`): `ladder()` on a small hex20 box has
     `precond_dtype="float32"` rows, and `--parts memory` in a tmp dir replaces only
     `memory` (plus `memory_commit`) and keeps `constants` / `lobpcg` and `commit`;
   - `test_strategy.py::test_fp32_slices` parametrized by family (voxel GMG, hex AMG):
     every count / slice `make` passes `dtype` float64, and `_slice_capacity` /
     `_count_block` receive the fp64 multigrid model;
   - the LOBPCG-path band count (`verify=True`, no slices) builds fp64 with fp32 AMG;
   - `tests/test_band.py::test_fp64_count_model` parametrized by family (voxel GMG, hex20
     AMG): the probe budget uses the fp64 multigrid model;
   - `si_slice` refuses an fp32 AMG (`ValueError`, the message naming GMG and AMG);
   - `tests/test_lobpcg.py`'s `solve_modes_memory` loop (`:897`) gains
     `("amg", "float32")` on the hex20 box: equal to `lobpcg_memory` with
     `amg_memory(..., dtype="float32")`.
5. **`amg_memory` storage fp32:** ≥ owned bytes on `STORAGE_CASES` and 18b's shell storage
   cases, within 1.3× on the tight cases; the exact parts equal the owned bytes of those
   leaves.
6. **Steps fp32:**
   - the hex step with fp32 AMG ≤ `lobpcg_memory`'s step on `test_amg_step`'s cases;
   - the shell fp32 step, low and polish ≤ the model at $m$ = 6 / 25 / 120 on 18b's
     `STEP_CASES`;
   - a new `test_hex_polish_fp32` (`slow`, as `test_hex_polish_below_step`): the fp32 hex
     polish (measured as
     `shell_solve_study.polish`) ≤ `fl*ws` + the model's polish term, and `lobpcg_memory`
     takes that term (with and without `low_memory`). `test_hex_polish_below_step` stays
     fp64;
   - slow tight pins: the hex20 $24^3$ ratio gate; shell ≤ 1.3× at $128^2$, $m$ = 25 / 120.
7. **Constants pinned** (new; 18b had no such test): load `amg_fp32_study.json`; each
   `STORAGE32[fam]` equals the recorded maximum ratio, and each `AMG_STEP32` /
   `SHELL_STEP32` entry equals the recorded LP fit, rounded as stored ($c$ + 0.5 up).
8. **Acceptance (roadmap; `peak`, `slow`, darwin, as the fp64 `test_mac_acceptance`; not
   `serial`, which would run it twice in `scripts/test.sh`, and its wall-time budget has a
   10× margin, noted as a comment):**
   - `amg_study.acceptance(44, dtype="float32")`;
   - converged, cold run ≤ 15 min;
   - `est ≥ 2e9`;
   - the larger warm peak within 20% of `solve_modes_memory(precond_dtype="float32")`;
   - that estimate is below the fp64 one on the same mesh.
   A `test_acceptance_smoke` parametrized by dtype checks the keys at $8^3$, with the
   estimate equal to `solve_modes_memory(precond_dtype=dtype)`.
9. **Shell peaks fp32 (`peak`, `slow`, darwin):**
   - the fp32 build peak in a fresh subprocess ≤ `build` and ≥ `build` / 1.35 on flat
     $362^2$ and block $64^3$ (18b's gated cases; on block $32^3$ today's fp64 estimate is
     already 1.58× the fp64 peak);
   - the reused build (fp64, as the σ rounds) with the aggregates held, accounted as
     `test_amg_reuse_peak`: the peak above the operator and the aggregates, plus
     `_nbytes(agg)`, ≤ `build` on flat $256^2$;
   - the 791k plate solve with `precond_dtype="float32"` (synchronous dispatch), with
     `est ≥ 2e9` and within 20% of `solve_modes_memory`.
10. **σ-round reuse:**
    - `solve_modes` on a clamped hex20 $6\times5\times4$ box (parametrized by
      `precond_dtype`) and on a free `ssh8` $16^2$ plate at $L/t$ = 50 (σ = 0 refused, so
      the `sigma_estimate` start) aggregates once: a spy on `api.amg` records, by weakref,
      the aggregates returned by the first call that returns (`return_aggregates=True`;
      the free plate's first call, at σ = 0, raises before aggregating). Every later build
      passes `aggregates=` that same object. That `aggregates=` skips aggregation is 13b's
      and 18a's reuse tests; no aggregation spies;
    - λ, σ and the iteration counts are `array_equal` to a run with reuse disabled. The
      disabled run wraps `api.amg` so it drops `aggregates=` and still returns a tuple when
      asked.
    - **Lifetime** (the weakref, `gc.collect()` before each check):
      - plain mode: dead at the entry of the main `lobpcg` (picked out by `n_modes > 1`;
        the σ rounds pass 1);
      - band mode on the hex box, converged to the band end: the counts receive the same
        object;
      - after a capacity stop: `test_capacity`'s `aggs_dead` (fake builds) already pins it,
        with the σ-round aggregates too;
      - the no-chunk slice path: dead at the entry of `si_band`, with `fake_builds` as
        `test_memory_no_lobpcg_build` and `si_band` stubbed after the check; the σ-round
        builds after the first pass `aggregates=` (`fake_builds["builds"]`).
11. **Benchmark smoke** (`slow`): `amg_fp32_study` parts `accuracy`, `storage`,
    `steps_hex` and `fit` on one tiny case each record their keys. `steps_shell`, `peaks`
    and `acceptance` are left out: tests 6, 8 and 9 run their helpers.
12. **Existing tests that change:**
    - `tests/test_api.py`: the `bad` row at `:372` (AMG + fp32 expected to raise) is removed;
      the Jacobi row (`:373`) still matches "needs precond 'gmg'" in the new message.
    - `tests/test_strategy.py`:
      - `test_levers` (`:1111`): the AMG message now names `precond_dtype` (assert its
        $C_{max}$ instead of its absence);
      - `test_memory_no_lobpcg_build` (`:395`): for $f_1$ = 0 `not any(return_aggregates)`
        stays; for the interior case `return_aggregates` appears only inside
        `_precond_sigma`, and the existing `seen["before"] == sum(inside)` already gives no
        build after the pre-estimate;
      - `test_capacity`'s `aggs_dead` (`:192`): unchanged; it pins the release before the
        slices.
    - `tests/test_band.py`: the `spy` (`:77–80`) and `no_build` (`:111–121`) fixtures stop
      using `"aggregates" in kw` as "the λ_c count build", since the σ rounds and LOBPCG's
      build now pass it too. Both wrap `api._verify` and its `build` argument (the 13th
      positional, `api.py:584–588`), which is exactly the λ_c build. `spy` records the `s`
      that build is called with, and `no_build` replaces it with a failing one. This
      covers Jacobi too, so the `api.jacobi` patch goes. No flag, so nothing carries over
      between the two solves of `test_verify_false`. When `spy` and `no_build` stack,
      `no_build` wraps the current `api._verify`. So `:172–173`, `:260`, `:299`, `:326`
      and `:465` keep their meaning, and the comment at `:170` is reworded.
    - `tests/test_options.py::test_defaults_snapshot` (`:85`): the first AMG build carries
      `return_aggregates=True`, and LOBPCG's carries `aggregates=` and `dtype="float64"`.
      The snapshot strips the two aggregate keys and expects
      `dict(nonsingular=True, dtype="float64")` for LOBPCG's build, as the GMG case does.
      It asserts `return_aggregates` on the first call and `kw["aggregates"] is not None` on
      the second (the `calls` fixture records no return values). A `precond_dtype="float32"`
      case beside it expects the first build without `dtype` (fp64) and LOBPCG's with
      `dtype="float32"`.
    - `tests/test_shell_solve.py::test_band`: unchanged (one `return_aggregates` build); its
      docstring's "reuses LOBPCG's aggregates" becomes "the solve's".

## Docs (18c)

**Retag rule:** every `PR 18c` / `18d` / `18e` mention in `README.md`, `docs/*.md` (not
`docs/plans/`), `src/`, `tests/` and `benchmarks/*.py` is retagged per the split table, and a
grep before merge checks it. The retags run from the top down (18e → 18f first, then
18d → 18e, then each 18c mention to 18d, 18e or left on 18c), so no mention is retagged
twice. The list below is what
rounds 1–5 found; `benchmarks/strategy_study.py:227`'s target string is updated with the
re-run memory part's figure (Benchmark, 10M reach).

- **18c → 18d:**
  - `docs/mesh.md:38` (Morton);
  - `docs/operators.md:204` (`layout`) and `:251` (roofline);
  - `docs/preconditioners.md:570` (`layout`);
  - `docs/boundary_conditions.md:53` (`ndimage.label`);
  - `docs/conventions.md:89` ("PRs 5 and 18" → "PRs 5 and 18d", compile-time tests);
  - ROADMAP `:771` (Pallas "if PR 18c's roofline") and `:785` (the Schwarz scan remedy);
  - the deferred GPU checks "PR 18c: matvec ≥ 50% of roofline" and "XLA kernel temporaries"
    (`:790–791`).
- **Stays 18c:** ROADMAP `:786` ("18c's fp32 Schwarz item").
- **18d → 18e:**
  - ROADMAP `:82`, `:95`, `:97`, `:177`, `:184` ("18d (the COUNT_STEP refit)");
  - `docs/eigensolvers.md:128`, `:154`, `:174`, `:180`, `:671`, `:773`, `:814`;
  - every PR 18d mention in `docs/open_items.md` ("→ PR 18d", "deferred to PR 18d",
    "*PR 18d*"; e.g. `:137`, `:148`, `:203`, `:222`, `:282`, `:284–286`, `:307`), except
    `:182`, which is split: its band items (locked modes reused by `si_band`, the
    Weyl-sized store, `t_ref`) → 18e, and the `_amg_scan` cache and `gmg_memory` per search
    step → 18d (split table).
- **`docs/open_items.md`'s PR 18c mentions,** by the split table:
  - → 18e: `:161` (refill), `:167` (gather on a skip), `:287–289`;
  - → 18d: `:165` (`layout`), `:183` (Metal `_default_budget`; the XLA-temporaries GPU
    check), `:234` (`_chunked`), `:245` (`chunk_bytes`);
  - stay 18c and are resolved: `:241`, `:255`;
  - `:239` (the PR 18a split entry) is rewritten to the six parts, with 18c = fp32 AMG and
    reuse.
- **18e → 18f:** the CalculiX mentions (ROADMAP `:666–670`), and the memory file
  `project-calculix-benchmark.md` (outside the repo).
- **ROADMAP headings and rows:** the 18c–18f section headings, the dependency rows, `:18`
  (status) and `:627`.
- **README `:5`:** "split into 18a–18e" → "18a–18f", with 18c's status.

**Polish-term statements that become false** (the hex fp32 polish term, `levers()`):
- `docs/eigensolvers.md:412` ("exists only for the solid-shell AMG family") and `:466`
  ("Hex AMG … costs nothing there"): true for fp64 only; with fp32 the polish's PCG keeps
  fp64 vectors around the fp32 cycle;
- `docs/eigensolvers.md:154`: the levers list names `polish_options={'on': False}` (shell
  AMG and fp32 hex AMG);
- `docs/open_items.md:266` ("for the shell AMG family only");
- `docs/preconditioners.md:584` ("hex AMG has no polish term");
- `memory.lobpcg_memory`'s docstring (about `:157`, "polish for the shell AMG family only").

**fp32 statements that become false:**
- `docs/eigensolvers.md`:
  - `:33` (the `"amg"` bullet's "fp64 only");
  - `:76` and the `precond_dtype` row `:82` (GMG and AMG; LOBPCG and the polish only; slices,
    counts and the pre-estimate fp64; on hex a memory lever only);
  - the levers list (`:154`, "on GMG `precond_dtype`") and `:169` ("LOBPCG's GMG only");
  - `:709` and `shift_invert.py:215` (docstring) / `:246` (message): the MINRES refusal
    names "a GMG or AMG with fp32 levels".
- `docs/open_items.md`: `:80` ("`gmg`'s and `amg`'s `dtype`") and `:174` ("LOBPCG's GMG or
  AMG").
- `docs/preconditioners.md`:
  - `:362–384`: `amg(...)`'s signature and a `dtype` bullet;
  - `:565`: `amg_memory`'s signature, the fp32 exact terms (72 B per node; the transfer
    tables stay 12 B per entry), the `STORAGE32` ratios, the fp32 step fits and the hex fp32 polish term, the
    build term on fp64 storage, and the fit procedure with the measured gates;
  - the AMG and Schwarz sections: an "fp32 V-cycle" paragraph (what is fp32, what stays
    fp64 including the shared p-level transfer, accuracy and times from the benchmark; on
    hex with `low_memory` the step saving is 0–0.7 blocks at $m \ge 25$);
  - reuse: `:422`'s "`solve_modes` does not use it" becomes the σ rounds, LOBPCG's build and
    the counts; the shell **Reuse** paragraph (`:514`) the same;
  - its "## Tests" list (`:621ff`).
- `docs/boundary_conditions.md:41` (`ap_operator`) and `docs/operators.md`'s `unstructured`
  section (`:59ff`) / `:238`: on `Unstructured` operators `ap_operator` computes in fp64 and
  casts the result to `x.dtype` (PR 18c); the operator itself does not follow the dtype. The
  stale "(PR 13)" goes. `operators.md`'s validation list (`:301`) gains test 1.
- `docs/eigensolvers.md`:
  - `### σ pre-estimate` (`:488ff`): the aggregates are built once;
  - the memory-model section;
  - the 10M paragraph and table (`:154–162`, from the re-run `strategy_study` memory part);
  - "Mac acceptance budgets" (`:184–208`): `:186`'s "also with fp32 V-cycles" names
    AMG, and the table gains a row "hex20 1M, measured (PR 18c, fp32 AMG)" with its peak
    against the estimate;
  - "Measured on solid shells (PR 18b)" (`:470ff`): the 791k plate's fp32 peak against the
    estimate and its band $C_{max}$ with and without fp32 AMG;
  - the tests list (`:895–980`).

**10M and memory statements:**
- ROADMAP:
  - `:92` ("the 700–1000 target needs PR 18c's fp32 AMG");
  - the memory table's 10M column: Multigrid, Preconditioner workspace, and the Max locked
    modes cell "with fp32 AMG from PR 18c (15d, fp64 AMG: 120 – 260)", replaced by the
    estimator's figure;
  - the 80 GB GPU paragraph (`:181`), with the fp32-AMG totals or "unchanged";
  - `:184`: "18c (fp32 p-MG/AMG)" becomes "18c (fp32 AMG, its p-level included)", and the
    new 18c section's bullets from `:650` / `:658` the same (`pmg` gets no dtype).
- **ROADMAP otherwise:**
  - `:18`: the status line; its "(PR 13b) … fp32 V-cycles stay GMG-only" becomes "fp32
    V-cycles: GMG, and AMG since PR 18c". Its split sentence ("18c (kernels), 18d (band
    follow-ups) and 18e (CalculiX comparison)") is rewritten to the six parts, not retagged;
  - `:627` and the new 18c / 18d / 18e / 18f sections;
  - the dependency rows (split section);
  - deferred GPU checks: the fp32 AMG step constants against the GPU's compiled
    temporaries (new), and `:784`'s "the 10M hex20 band reach (120–260 locked modes with
    fp64 AMG)" gains the fp32-AMG figure.
- **`docs/open_items.md`:**
  - a new "PR 18c decisions without user input" entry: the split, pmg without fp32, the fine
    $A_P$ and the p-level transfer in fp64, fp32 in LOBPCG only, the build term on fp64
    storage, the 10M reach correction, the reuse gain, the fp64 hex step at $m$ = 120 and
    the fp64 hex polish against the low model, `AMG_LOOP_K` kept as the fp64 bound for
    fp32 (`:83`), the 10M two accountings and the assumed 512 GB node RAM, the shell
    $y_\infty$ rows and 18b's fp64 shell extrapolation beyond 1.58M;
  - resolve `:241` ("18c's fp32 halves the patches", with the measured ratio) and `:255`
    (σ-round reuse, done).
- README: a `precond_dtype="float32"` sentence in the AMG `solve_modes` paragraph (`:242`),
  and `:297` ("the voxel GMG (fp32 V-cycle …), p-MG and AMG") names AMG's fp32 cycle.
- The 18a plan's split table stays as history; this plan's table supersedes it.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **The 10M reach:**
  - The roadmap table's "≈ 700 (stored Ke) – 1000 (on the fly), with fp32 AMG" was an
    extrapolation.
  - The performance reviews estimate 2.6–3.3 GB of storage saved at chunk 5 (ρ about
    0.5–0.6: the int bytes do not halve, and the fp64 model is loose) plus 1–2 low-step
    blocks.
    - Unified 76 GB budget (today's `hex20_10M` record): about +30–40 columns.
    - Separate device budget (`hex20_10M_gpu`, 512 GB node assumed): about +90–120, to
      about 650–680 (stored) / 900–930 (on the fly).
    So the "120–260 on one 80 GB GPU" in `eigensolvers.md:162` and ROADMAP `:784` is the
    unified figure; on a GPU node the 700–1000 target is roughly met.
  - The docs and ROADMAP 10M cells get both re-run figures, each naming its accounting. No
    test criterion depends on them.
- **The pre-estimate phase:** it holds the fp64 AMG and, now, the aggregates, while
  `lobpcg_memory` takes the LOBPCG phase (fp32 storage) as the peak. The difference
  (about 0.3 GB at 1M, 3 GB at 10M) is ≥ 10× below basis plus step, so the LOBPCG phase
  stays the peak. Logged.
- **fp32 polish:** PCG with the fp32 cycle reached the same floor as fp64. If a case
  stagnates, the polish's accept-or-revert keeps the unpolished chunk. Test 3 pins the
  thin-shell case.
- **PCG with a fp32 preconditioner is not flexible:** the cycle is linear, and symmetric only
  to fp32 round-off, which grows on thin shells (2–4e-3 at $L/t$ = 1000). Convergence there
  was unchanged; GMG has worked the same way since PR 11b.
- **The fine level stays fp64:** an fp32 $K_e$ copy would cost half of the operator's
  storage, to save a cast.
- **Storage ratio, step refit:** storage is proportional to $n_c$, so a ratio bounds it on
  the grid by construction. The steps have a constant that fp32 shrinks unevenly, so they
  get 18b's LP envelope with an asymptote row at the largest $n$; the held-out points guard
  the extrapolation.
- **The p-level transfer in fp64:** 5 / 16 B per DOF more than fp32 (hex20 / hex27), against
  a second, unmodelled copy while band mode holds the aggregates.
- **fp64 hex step at $m$ = 120:** 3.2% above the model's extra at hex8 $40^3$ (about 1% of
  the step). Logged, not refitted in 18c.
- **GPU:** the fp32 constants come from CPU `memory_analysis`, so they are a deferred GPU
  check.
