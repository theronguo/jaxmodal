# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands), the Memory budget section and
the PR 15c plan's split (15d = the cost model, the memory budget, the strategy choice,
`verify=None` for bands and `solve_modes(..., band=(f1, f2))`).

Revision 6. Rounds 1–4:
- one decision, no seeding;
- memory-driven, no time crossover;
- no hybrid junction;
- a chunk ladder by reach;
- the basis rebuilt from a zeroed F-order host V;
- `n_modes`/200 store sizing;
- a 1.2 margin;
- bisection searches.

Round 5:
- **μ₁^ub** refined when the rule would fire on a loose estimate.
- **The filter output** is its own memory composition.
- **The store** after a capacity stop is sized from the locked count.
- **The ladder** uses a known depth (f₁ = 0 with `n_modes`), and its 10% threshold is measured.
- **Warnings:** a capacity fallback warns with the depth and the levers.
- **Caps:** C_max is capped at the free space, the capacity floor is 2c + 2b, and the host V is
  C + k.
- **Deviations logged:** δ_max ≤ 1 and the PR 14b behaviour changes.
- **Tests:** fixed (`n_modes=22`, `verify=True`, the ladder assertion from the rule, interval
  peaks).
- **Smaller changes:** the store check raises only, the bound is private, and one search helper
  serves all three searches.

## Scope

- `solve_modes(..., band=(f1, f2))` with $f_1 \ge 0$, plus `strategy` ("auto", "lobpcg",
  "shift_invert"), `memory_budget` (bytes) and `slice_options` (si_band's options).
- "auto": PR 14b's band mode from 0, with the ladder's chunk and the basis capped at
  $C_{max}$. It falls back to `si_band` (PR 15c) on the whole band in two cases:
  - LOBPCG stops at $C_{max}$ below $\lambda_{hi}$ ("capacity");
  - no chunk fits ("memory").
- The cost model, recorded in the docs and the benchmark as the evidence for the memory-driven
  rule: MINRES iterations against depth; LOBPCG's measured time from 0 to depth 1005 and per
  chunk size; the smallest band for which one slice would beat LOBPCG.
- Memory: `lobpcg_memory` and `si_memory` gain `device` and `host` entries. $C_{max}$ and the
  slice capacity are searched against the budget: one unified check on CPU, a device check and
  a host check on GPU.
- The $\lambda_{lo} < \mu_1$ rule (shift-invert from 0, filtered) and the up-front check of
  $\lambda_{hi}$ against an upper bound $\hat\lambda_\max$ of the spectrum.
- `si_band(verify=None)`: PR 14b's three-stage rule per leaf; si_band's store check.
- `si_band`'s capacity errors worded per case (PR 15c open item).
- The README example, the docs.

Notation:
- $\lambda_{lo} = (2\pi f_1)^2$ and $\lambda_{hi} = (2\pi f_2)^2$: the band edges;
- $\mu_1$: the lowest elastic eigenvalue;
- $\mu_1^{ub}$: an upper bound of $\mu_1$.

Out of scope (deferred, listed in `docs/open_items.md`):
- **PR 18, band follow-ups (features).**
  - **The hybrid junction.** After a capacity stop, keep LOBPCG's in-band locked modes and slice
    only above them. Rounds 2–3 found it needs a chosen junction gap, the lower neighbour
    plumbed out of si_checked and the span leaves, a projection dedupe, a combined count and a
    host term.
  - **`err.result`.**
  - **A memory-mapped host store** (the roadmap's np.memmap promise).
  - **Automatic levers in "auto":** `low_memory`, fp32 basis, on-the-fly Ke.
  - **`solve_modes_memory` for $f_1 > 0$** (a slice estimate).
  - **Enforcing δ_max ≤ 1 on slices from 0.**
  - **A time rule,** if a crossover is ever measured.
- **PR 18, performance:** PR 15c's "Late capacity errors and slice efficiency" items.
- **PR 18 and the deferred GPU checks:**
  - the depth-1000 remedies (15d warns only);
  - the GPU peaks of the new compositions;
  - the cost ratio on the GPU.

## Roadmap tests tagged [15d]

| Roadmap test | 15d test |
|---|---|
| [15a, 15c, 15d] matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 1–7, 9 |
| [15b, 15c, 15d] interior band between two well-separated modes: 0 modes, count < 0.5 (the planted-mode half is 15b/15c's) | 5 |
| [15a, 15c, 15d] clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` | 6 |
| [15d] f₁ below λ₁ solved as [0, f₂], filtered, matches `eigsh` | 7 |
| [15d] memory budget too small for the LOBPCG-from-0 basis → shift-invert | 2, 3 |
| [15a, 15d] MINRES iterations vs δ measured (15a), cost-model constants fitted | 13 (benchmark record) |
| PR 15 contents: capacity per slice from the memory model | 11, 17b |
| Band edges: an edge above the largest computed eigenvalue raises | 10 (up front); 15c tests 3, 11 (late) |

## Design

### API

```python
lam, V, info = solve_modes(mesh, materials, bcs, band=(f1, f2))           # auto
solve_modes(..., band=(f1, f2), strategy="shift_invert", memory_budget=8e9,
            slice_options=dict(tau=1e-8, block=6, verbose=True))
```

- `band=(f1, f2)`: $0 \le f_1 < f_2$ finite (Hz); `check_band` accepts $f_1 > 0$. $f_1 = 0$ runs
  the same code path.
- `strategy`:
  - "auto" (default).
  - "lobpcg": PR 14b's band mode, filtered, capped at $C_{max}$ like "auto". It raises
    `ValueError` (naming `memory_budget` and the levers):
    - up front, when no chunk (or the given `chunk`) fits;
    - at a capacity stop.
  - "shift_invert": `si_band` on the band.

  Without `band`, any `strategy` but "auto", a `memory_budget` or `slice_options` raises.
- `memory_budget`: bytes.
  - Default on GPU: `bytes_limit − bytes_in_use` at entry (`memory_stats()`), i.e. the usable
    pool under `XLA_PYTHON_CLIENT_MEM_FRACTION` minus what the caller holds.
  - Default on CPU: 0.8 × physical RAM (`os.sysconf`; unified memory on the Mac). That is 55.0e9
    bytes on the 64 GiB Mac, against PR 13b's separate 52 GB benchmark guard. It is not reduced
    by what the caller holds (documented).
  - The GPU host budget is 0.8 × physical RAM.
  - The searches compare 1.2 × the estimate with the budget. The estimators run up to 9–16% low
    (si_memory 1.09–1.11×, GMG LOBPCG 1.12×, a one-chunk zero-workspace solve 1.16×), and 1.2 is
    the roadmap's ±20% validation band. Test 17a reports the excess per column of the new V.
  - Recorded in `info["budget"]`.
- `slice_options`: dict with keys `tau`, `maxit_inner`, `block`, `max_slices`, `capacity`
  (overrides the sizing), `edges` and `verbose`, validated as si_band does.
- The other `solve_modes` arguments reach the solvers as follows:
  - `tol`, `verify`, `count_options`, `seed` → si_band;
  - `precond`, `precond_options` → si_band's `make(σ)`, always fp64. With
    `precond_dtype="float32"`, only LOBPCG's GMG is fp32, since MINRES needs a symmetric cycle;
  - `basis_dtype`, `low_memory`, `chunk` → LOBPCG and $C_{max}$. A given `chunk` replaces the
    ladder;
  - `sigma` → the preconditioners' shift. A hint never enters $\mu_1^{ub}$; it only selects the
    estimate path below.
- `n_modes`:
  - $f_1 = 0$: as PR 14b, the expected count. It sets the ladder's top chunk through
    `band_defaults` and is the known depth for the ladder.
  - $f_1 > 0$: the expected modes in the band. It sizes the slice store and the need cap and
    never reaches `band_defaults`: `_solve` gets `band_defaults(None, chunk)`, so chunk 40 by
    default and an initial basis of 4 chunks.
- Returns the band modes and `info`.
  - **The band filter.**
    - The upper end comes from LOBPCG's `n_band` (its own chain rule) or si_band's set.
    - The lower end keeps λ ≥ λ_lo(1 − 100t), extended down over chains with relative gaps
      < 100t (si_band's `_extend_chain`). t is LOBPCG's per-mode threshold or r_acc.
  - **Fields:**
    - `strategy` ∈ {"lobpcg", "shift_invert"};
    - `reason` ∈ {"forced", "fits", "capacity", "memory"};
    - `chunk`, `c_max` (per ladder chunk);
    - `stagnation_risk`, `budget`, `from_zero` (False when $f_1 = 0$ runs on LOBPCG), `mu1_ub`,
      `n_store` ($N_s$);
    - `count`: PR 14b's, or si_band's summary;
    - with slices: si_band's `slices`, `capacity`, `merged_edges`, `collapsed`,
      `edge_rr_failed`, `duplicates`, `r_acc`.

### Up-front checks (after the operator build, before the σ pre-estimate and any preconditioner)

- **Spectrum bound** $\hat\lambda_\max = \max_e \lambda_\max(K_e, M_e)$.
  - Why it bounds: for any $x$, $x^\top K x = \sum_e x_e^\top K_e x_e \le \max_e \lambda_e
    \sum_e x_e^\top M_e x_e \le \hat\lambda_\max\, x^\top M x$. Nodal masses, masking and rigid
    deflation only lower the Rayleigh quotient.
  - Exact: batched Cholesky and `eigvalsh` in `_spectrum_bound(op, mesh, ...)`.
    - Voxel: one reference element per phase on the host, times $\max_e s_{K,e}/s_{M,e}$. A
      cell with $s_M = 0$ and $s_K > 0$ gives ∞, so the check never raises; a void cell
      ($s_K = s_M = 0$) counts 0.
    - Unstructured: `element_matrices` (PR 12) on the built operator in device chunks, about
      200 µs per hex20 element (about 170 s at 10M; the time is recorded in the benchmark).
  - $\lambda_{hi} > \hat\lambda_\max$ raises `ValueError`. An edge in $(\lambda_\max,
    \hat\lambda_\max]$ still reaches si_band's capacity error or PR 14b's "more modes than the
    mesh provides".
  - The time is recorded in `info["time"]["bound"]`. The tests keep `lam_hat` = ‖K‖/‖M‖ for the
    eigsh tolerance.
- **$\lambda_{lo} < \mu_1$** on every slice path.
  - $\mu_1^{ub}$ is the min of the available upper bounds:
    - the σ pre-estimate (a Rayleigh quotient);
    - after LOBPCG ran, its lowest converged elastic Ritz value.

    With a `sigma=` hint or Jacobi there is no pre-estimate, so `sigma_estimate` (10 iterations)
    gives a first bound. When the rule would fire on that bound alone, it is refined: with a
    multigrid, one `lobpcg(op, free, 1, 0.1, R, P, maxit=20)` with the P already built at the
    hint (as `_precond_sigma`; measured within 1e-6 of μ₁); with Jacobi, `sigma_estimate` run to
    its stop rule (≤ 8% high measured; the 10-iteration bound was 28–11 400× high).
  - If $\lambda_{lo} < \mu_1^{ub}(1 + m)$ with $m = \max(10^{-2}, 300\,\text{tol})$, si_band runs
    on $[0, \lambda_{hi}]$ and the result is filtered; `info["from_zero"]` is True.
  - Why the margin: it keeps μ₁ out of si_band's widened first-slice window when
    $100\,r_{acc} < m/(1 + m)$. A wide first slice can exceed that (r_acc ∝ τ δ, δ ≈
    λ_hi/(2μ₁)); `_split` then recovers with extra runs, not a wrong result (documented).

### Strategy ("auto")

1. **Ladder.**
   - Candidates: the top chunk (`band_defaults`' chunk), halved while ≥ 5, always containing
     the top chunk.
   - For each, compute $C_{max}$ (below). A chunk fits if $C_{max} \ge$ up(4·chunk + k, c).
   - The choice:
     - with a known depth ($f_1 = 0$ and `n_modes` given): the largest fitting chunk with
       $C_{max} \ge$ up(1.25 n_modes + k, c);
     - otherwise: the largest fitting chunk whose $C_{max}$ ≥ 0.9 × the best $C_{max}$ among the
       fitting chunks (the 10% threshold is justified by benchmark part 2's time per mode
       against chunk; adjusted if the data say so).

     On small meshes $C_{max}$ is capped at the free space, so every chunk reaches it and the top
     chunk wins. Below the top chunk's need, the rule picks a small chunk (5 or 10) in practice:
     the reach gap between chunks is a few hundred columns at every mesh size (documented).
   - No fitting chunk: "memory". si_band runs on the whole band, and a warning names the levers
     and their $C_{max}$ from the cached estimators.
   - A given `chunk` is used if it fits, else "memory".
2. **LOBPCG** `_solve(..., lam_stop=λ_hi, max_capacity=C_max)`. In band mode `max_capacity` is
   required; solve_modes always passes it.
   - Basis growth: `new = min(max(1.5 old, need), C_max)`. The initial allocation is
     `min(up(n_modes + k, c), C_max)` with `_solve`'s n_modes (above).
   - At a growth, `B`, `args` and every other reference to B are dropped. B is reallocated with
     zeros and refilled with V's first `count` columns in chunks of c (a C-order host transient
     of c columns per chunk), cast to `basis_dtype`. It equals the old B exactly.
   - The host V is `np.zeros((n, C_max + k), order="F")`, allocated once. Calloc is lazy:
     untouched columns cost no pages. It is never concatenated.
   - The band-end check comes first. A lock needing count + k > $C_{max}$ stops the solve:
     `info["stopped"] = "capacity"`.
   - Reaching $\lambda_{hi}$ ("fits"): PR 14b's count at the cut above $\lambda_{hi}$, then the
     filter.
     - $f_1 = 0$ returns the view `V[:, :n_band]`. This supersedes 14b's compact copy: the
       touched beyond-band and last-chunk columns (≤ k + guards) stay resident, the rest were
       never touched.
     - An interior band copies its columns into an F-order output after B, the state and the
       preconditioner are freed.
   - Non-convergence ends the solve with `converged` False, as in PR 14b; there is no fallback.
3. **Capacity stop** ("capacity").
   - LOBPCG's basis, state, preconditioner and V are freed after reading $\mu_1^{ub}$ and the
     locked count L at or above λ_lo (for $f_1 = 0$, all locked).
   - The store reservation becomes $N_s = \max(N_s, \lceil 1.25 L\rceil)$, recorded in `n_store`.
   - A warning gives the locked depth and each lever's $C_{max}$.
   - si_band runs on $[\lambda_{lo}, \lambda_{hi}]$, or from 0 by the rule above.
4. **Stagnation.** After a capacity stop with ≥ 1000 locked modes, the warning says so and
   `stagnation_risk` is True: 15a measured a MINRES solve near depth 1005 as marginal at
   τ = 1e-8 on hex20/AMG. On the "memory" and forced paths the depth is unknown and no flag is
   set (documented); si_band's MINRES flags (τ_eff) still report it.

The strategy has no time-based choice. For every measured depth D and band size N, LOBPCG's
measured time from 0 to D + N is below one slice's cost,
$(84 + 1.8N)/b$ steps × $\alpha D(1 + D/D_0)$ iterations × $t_\text{minres}$. Example: on hex20
$16^3$/AMG at D ≈ 1000, LOBPCG from 0 took 6.1 min, against about 2.3 h for one slice's fixed
start. If benchmark part 2 shows otherwise, the finding is logged as a PR 18 item; 15d keeps the
memory-driven rule.

### Cost model (benchmark and docs only; no constants in code)

- **MINRES iterations per solve:** $\alpha D(1 + D/D_0)$.
  - Fit: log least squares on the frac-0.5 non-floor rows of PR 15a's table. The depth-1005
    row uses `recheck_1005`: it converged, while the frac-0.5 row stagnated at the same
    iteration count.
  - hex20/AMG: about α = 6.0, $D_0$ = 690, ratios 0.82–1.15 over depths 6–1005.
  - voxel/GMG (depths ≤ 201): α about 4.35, $D_0$ about 760, weakly constrained.
  - This replaces the roadmap's "∝ ln(1/τ)/δ plus an outlier term" (15a: depth drives the
    iterations).
- **LOBPCG from 0,** measured per chunk to about 1005 modes on hex20 $16^3$/AMG. The chunk
  iterations rise from 18 to 77, and $t_\text{warm}$ from 0.215 to 0.278 s: the projector adds
  +30% at depth 1000. The projector-to-operator ratio is per DOF, so it carries to larger meshes
  at the same depth.
- **Time per mode against chunk size** (chunks 5, 10, 20, 40 to about 200 modes): this justifies
  the ladder's 10% threshold.
- **The crossover table:** per depth D, the smallest N at which a slice beats LOBPCG to D + N.
  "None" is expected. Per mode, a slice costs about 25–110 LOBPCG modes.

### Memory

- **The split.** Both estimators gain `device` and `host` entries, each term tagged by memory
  space.
  - `lobpcg_memory`:
    - host = V, $8n(\text{n\_modes} + k)$, plus the AMG build (pyamg runs on the host) and the
      operator's host build;
    - device = the rest (basis, step, start, seed and slack blocks, kernel temporaries).
  - `si_memory`:
    - host = $8n\,\text{n\_deflate} + 8C^2 + 8n(\text{n\_modes} + 2)$ plus the host builds;
    - device = the rest.
  - Unified memory (CPU): device + host ≤ budget. GPU: device ≤ budget, host ≤ host budget.
- **One helper,** `_largest(fits, step, hi)`: the largest multiple of `step` ≤ hi with
  `fits(C)` True, by bisection (each `fits` is monotone). `probe_block`, $C_{max}$ and the slice
  capacity all use it. Inside a search, `op_memory`/`precond_memory` are wrapped in
  `functools.cache` (per call; amg_memory takes about 2.4 s at 10M).
- **$C_{max}(\text{chunk})$** = `_largest(fits, c, up(n_avail + k, c))`, n_avail = free −
  rigid. fits(C) requires 1.2 × max(A, B) ≤ budget, with:
  - A = `lobpcg_memory(..., n_modes=C, band=True)`, so V is C + k columns;
  - B, only for $f_1 > 0$, the filter after B is freed = overhead + op storage + rigid
    + $8n(C + k)$ + $8n\,C$.

  On GPU the split applies term by term.
- **Slice capacity $C$** = min(the budget cap, the need cap).
  - The budget cap: `_largest` over C ≥ 2c + 2b, with `si_memory(op_memory, n, C − W,
    n_modes=⌈1.25 m_s⌉, block, pc, n_rigid, n_deflate=W)` × 1.2 fitting. W = 2c, so the device
    basis is exactly C. The host term adds $2m_s$ columns (a span-rerun window and si_checked's
    [W | V] restart copy), $m_s = \max(1, (C - W - 84)/1.8)$. The store is reserved as
    $8n\,N_s$:
    - $N_s$ = `n_modes` if given, else 200, the roadmap's assumed band size;
    - after a capacity stop, at least 1.25 L.
  - The need cap: up(1.25 (1.8 $N_s$ + 84) + W, c).
  - si_slice already stops at the free space.
- **Limits:**
  - Below 96 columns (PR 15c's slice minimum at b = 6): a warning. This is common on "memory"
    with AMG, since a slice costs about the smallest LOBPCG chunk; the si_memory to LOBPCG
    chunk-5 ratio is recorded in the benchmark.
  - Below 2c + 2b: a `ValueError` naming the budget.
  - `slice_options["capacity"]` overrides the sizing.
- **si_band's store check:** when the host store would pass the remaining host room, a
  `ValueError` names `n_modes` and `memory_budget`.
- **Probe blocks:** PR 14b's count is unchanged; si_band's leaf counts as 15c.
- **`solve_modes_memory`:** unchanged; $f_1 > 0$ still raises. A slice estimate is deferred to
  PR 18, and the capacity search's `si_memory` dict is checked by the tests.

### `si_band(verify=None)`

`verify` ∈ {True, False, None}. None applies PR 14b's rule per leaf:
- skip above `MAX_DEGREE`;
- skip when the a-priori bound $2\,n_\text{blocks}\,d\,(b_p/b)\,t_\text{minres}$ exceeds the
  leaf's time (0.3 instead of 2 when τ_c is looser than the rule, as `_bound`);
- otherwise calibrate (PR 14b's `_calibrate`, with the leaf's count closure and preconditioner)
  and count iff the prediction ≤ the leaf's time.

Both skips happen before the pairs are gathered. A run leaf's time is its slice's; a span's is
its parent run's. $t_\text{minres}$ is the leaf's slice's measured seconds per MINRES iteration.
`_bound` and `_calibrate` move to `counting.py`, and the tests patch them there. Every leaf
records `predicted_s`, `predicted_kind` and `reason`.

### Errors

si_band's `_cap_error` is worded per case:
- "lo may be below lambda_1" only when no lower side closed and lo > 0;
- "hi above the spectrum" only when no upper side closed;
- lo = 0 names only hi.

## Changes to existing code

- `lobpcg._solve`:
  - band mode requires `max_capacity`: initial allocation, growth by free-and-refill from V,
    the count + k stop;
  - V is a preallocated zeroed F-order array, and the view is returned;
  - non-band mode is unchanged.
- `si_band`: `verify=None`, the store check, the `_cap_error` wording.
- `counting.py`: `_bound` and `_calibrate` moved from `api.py`.
- `api.py`:
  - `solve_modes`: `band=(f1, f2)`, `strategy`, `memory_budget`, `slice_options`, the ladder,
    the checks, μ₁^ub refinement, the fallback, the filter;
  - `check_band` for $f_1 > 0$;
  - the NotImplementedError removed;
  - `_spectrum_bound`.
- `memory.py`: term tags and `device`/`host` entries, `_largest` (also used by `probe_block`),
  the $C_{max}$ and capacity searches, the default budget.
- Direct band `_solve` callers in benchmarks and tests pass `max_capacity`.

## Benchmark (`benchmarks/strategy_study.py`, one invocation, ≤ 15 min, progress to a log)

1. **[15a, 15d] constants.** Refit α and $D_0$ per family on `shift_invert_study.json` by log
   least squares (no runs, seconds).
2. **[15d] crossover and ladder.** On hex20 $16^3$/AMG:
   - one LOBPCG band run to about 1005 modes (about 6 min), recording the iterations and
     $t_\text{warm}$ per chunk;
   - from it, the crossover table per depth against the MINRES rows, and the held-out "full"
     slice row predicted against measured (about 1.4×);
   - chunks 5, 10, 20 and 40 to about 200 modes (about 1 min each), recording the time per mode
     against chunk, which justifies or adjusts the 10% threshold.
3. **[15d] memory record** (estimators only, seconds with the cache).
   - The ladder (chunk 40 → 5: fits, $C_{max}$ including the 1.2 factor and the default budget)
     × {9-block, low} × {fp64, fp32 basis} × {stored, on-the-fly Ke}, plus {fp64, fp32 V-cycle}
     on voxel. At 10M hex20 on the 76 GB pool, and at 1M hex20 and 2M voxel on the Mac.
   - The si_memory to LOBPCG chunk-5 ratio at C = 48–144.
   - The spectrum-bound time at 10M (extrapolated per element).
   - fp32 AMG is not modelled until PR 18, so the roadmap's 10M "700–1000" (fp32 AMG) is
     recorded as fp64 plus the levers and marked for PR 18. The 2M Mac "> 2000" is checked
     with the fp32 V-cycle.

Results go to `benchmarks/results/strategy_study.json` with the git commit. Not run: the GMG
depth-500 row and the slice-model refit (logged). The measured peaks are tests 17a and 17b.

## Tests (`tests/test_strategy.py`; roadmap tags in the docstrings)

Method:
- **Exact preconditioner.** Real counts use the dense exact $(K + \sigma M)^{-1}$ (15c's
  `exact`), monkeypatched into `api.amg`/`api.gmg` and si_band's `make`. The fake follows the
  builders' contract: `SingularShiftError` at σ = 0 on free meshes, and
  `return_aggregates`/`aggregates`.
- **Count cost.** Each count's degree ≤ about 1000 is asserted as a precondition (≤ 15 s per
  count on "short").
- **`verify`.** Every test asserting counts passes `verify=True`, except test 14, which tests
  `verify=None`. Wiring-only tests use `count_options={"degree": 4, "n_probes": 2}`.
- **Budgets** come from the search functions (with the 1.2 factor), never hard-coded.
- **Timing.** Each test is timed and marked slow above about 20 s.
- **Accuracy.** Every test mapped to the eigsh item asserts the roadmap bound per mode with that
  mode's t, through test_shift_invert's `check_band` logic generalised.

1. **"fits"** ("short", hex20 $6\times5\times4$): an interior band of about 10 modes inside chunk
   0, with `n_modes=10`. Asserts: reason "fits", `from_zero` False, `info["chunk"] == 40` (the
   band count does not reach `band_defaults`), matches eigsh.
2. **"capacity"** ("short"), `chunk=5` (k = 10, c = 40), `n_modes=22`, band
   `edges(lr, 25, 46)`. The budget is at the top of the $C_{max} = 40$ window (81.06–82.74 MB
   measured with the 1.2 factor).
   - Preconditions: slice capacity ≥ 96 (C = 120 measured); LOBPCG's locked count < 25.
   - Asserts:
     - `stopped == "capacity"`;
     - si_band's (lo, hi) = (λ_lo, λ_hi) (spy);
     - no live array of LOBPCG's basis shape at si_band's entry (`jax.live_arrays()`);
     - matches eigsh, and the slice counts are < 0.5;
     - `mu1_ub` equals LOBPCG's lowest Ritz value;
     - `n_store` = max(22, ⌈1.25 L⌉);
     - the capacity warning names the depth.
   - At the same budget, `strategy="lobpcg"` raises at the cap, naming `memory_budget`.
3. **"memory"** (voxel $10^3$, GMG).
   - Budgets at 1.15× and 1.25× the chunk-5 need (31.42 MB) give "memory" and LOBPCG (chunk 5,
     $C_{max}$ = 40). This brackets the 1.2 factor.
   - Precondition: slice capacity at 1.15× ≥ 96 (exactly 96 measured).
   - With $\lambda_{lo} < \mu_1$: `from_zero` True, matches eigsh, `info["capacity"]` as
     searched.
   - With $\lambda_{lo}$ well above $\mu_1$: si_band gets (λ_lo, λ_hi) (spy), `from_zero` False.
   - The ladder: a budget between 1.2× the chunk-20 and chunk-40 needs gives the chunk the pure
     rule picks from the searched `c_max` table (5 on this mesh).
   - A pure-function table checks the rule:
     - the 0.9 × best reach;
     - the known-depth branch;
     - a top chunk below 5;
     - a given `chunk` not fitting.
4. **"shift_invert"**, with 15c test 2(a)'s configuration (modes 5–28, block 6, its seed) and
   `n_modes` given.
   - The budget comes from the search so that `capacity == 96`.
   - ≥ 2 runs; every leaf counted, < 0.5; the same set as "lobpcg".
   - One real-AMG copy, marked slow.
5. **Empty band:** strictly between two well-separated modes of the voxel "cube" (named gap).
   0 modes and count < 0.5 under both strategies; `from_zero` False.
6. **Clamped box**, `band=(0, f)`, "shift_invert": matches eigsh.
7. **$\lambda_{lo} < \mu_1$**, clamped and free, "shift_invert":
   - `from_zero` True, filtered, matches eigsh;
   - the margin: λ_lo = μ₁^ub(1 + 0.9e-2) gives True and (1 + 1.1e-2) gives False (μ₁^ub
     monkeypatched ≥ the true μ₁); tol = 1e-4 widens it to 3e-2;
   - correct with an inflated μ₁^ub;
   - the `sigma=` hint path on voxel $16^3$ with λ_lo ≈ 3μ₁: the refinement runs, `from_zero`
     False;
   - the Jacobi path on the 240-DOF "free" hex8;
   - a hint never enters `mu1_ub`.
8. **Filter table** (pure): the exact edge, inside and outside 100t, a chain crossing λ_lo,
   per-mode t, the upper end taken from `n_band`.
9. **Lower-edge clusters:** the voxel cube (doubles, triples) with λ_lo on a double and in the
   middle of a triple chain, under both strategies. All copies returned, `from_zero` False.
10. **Spectrum bound:**
    - equals a brute-force `scipy.linalg.eigh(Ke, Me)` loop over the elements of an
      unstructured graded `box_mesh` with nodal mass (its stiffest element in a forced partial
      last chunk) and of a two-phase scaled voxel whose stiffer phase is not phase 0;
    - ≥ the dense largest eigenvalue on both;
    - a cell with $s_M = 0$, $s_K > 0$ gives ∞; a void cell counts 0;
    - $\lambda_{hi} > \hat\lambda_\max$ raises before `_precond_sigma`, the preconditioner
      builders, `_solve` and `si_band` (spies);
    - just below the bound, no up-front raise.
11. **Capacity and $C_{max}$:**
    - one hand-computed device and host term per estimator;
    - `_largest` against brute force, including a tiny mesh where 8C² and the end phase matter,
      and the jump at cap > 0;
    - the need cap and the budget cap each binding;
    - $C_{max}$ capped at the free space;
    - unified and GPU (monkeypatched platform and `memory_stats`; AMG build in the host check);
    - below 2c + 2b raises naming the budget, below 96 warns;
    - an override above the need is kept;
    - `solve_modes_memory` with $f_1 > 0$ still raises.
12. **`_solve(max_capacity=...)`** directly:
    - the capacity sequence (growth straight to the cap);
    - a lock with count + k = $C_{max}$ is allowed, count + k = $C_{max}$ + 1 stops;
    - the initial allocation capped;
    - V F-order zeros of C_max + k columns;
    - after a refill, B equals V cast and its columns past count are 0;
    - `max_capacity` required in band mode.
13. **Cost-model record:** `strategy_study.json`'s constants equal a log least-squares refit of
    `shift_invert_study.json`'s rows (frac 0.5, non-floor, `recheck_1005`), with ratios in
    [0.75, 1.3]. The crossover and per-chunk entries are present.
14. **`si_band(verify=None)`:**
    - with `fake_band`, each branch: MAX_DEGREE skip; bound skip with 2 and with 0.3; calibrated
      and dear, skip; calibrated and cheap, count (`_calibrate` spied). A span leaf's time is
      its parent's, and the skips come before the gather (spy);
    - one real exact-preconditioner run: per-leaf `reason`/`predicted_kind`, `ran` True,
      estimate < 0.5;
    - `solve_modes(strategy="shift_invert")` at the default verify records them;
    - slow, real AMG on one leaf (hex20 $12^3$, as 14b's `test_calibration`): the bound ≤ the
      measured warm count time, and the calibrated prediction within [0.5, 2]×;
    - the store check (fake): raises past the host room.
15. **Capacity error wording** (`fake_band`): lower side closed, upper side closed, lo = 0.
16. **Options:**
    - `strategy`/`memory_budget`/`slice_options` without `band`, bad values, unknown
      `slice_options` keys, `f1 >= f2`;
    - forced "lobpcg" with no chunk fitting raises up front;
    - the default budgets (monkeypatched `os.sysconf`, a fake `memory_stats` with
      `bytes_in_use`);
    - spies: `slice_options`, `tol`, `count_options` and `precond_options` reach si_band and
      `make`;
    - `precond_dtype="float32"` on voxel through a slice path builds `make` in fp64;
    - a capacity stop with ≥ 1000 locked modes (monkeypatched) sets `stagnation_risk`;
    - LOBPCG not converged under "auto": no fallback, `converged` False.
17. **Mac peaks** (slow, darwin, fresh subprocess, MallocLargeCache=0, estimates ≥ 2 GB):
    - **(a)** `_solve(max_capacity=...)` on voxel about $48^3$, `low_memory=True`, chunk 10,
      n_modes about 100, so the old basis is ≥ 4c columns at a growth.
      - A spy around the growth (synchronised with `block_until_ready`) resets the interval
        footprint (`proc_reset_footprint_interval`) and reads its max.
      - Asserts: interval max − footprint before ≤ (new − old)·8n + c·8n, which catches an
        old+new transient of old·8n ≥ 4c·8n.
      - Asserts: the whole-run peak ≤ the budget and within 20% of the estimate.
      - Reports the excess per column.
    - **(b)** `solve_modes(strategy="shift_invert", band=..., memory_budget=B, verify=False)` on
      lumped voxel $80^3$, modes about 4–8, the capacity from the budget. The peak is ≤ B and
      within 20% of the composed estimate.

The README example (`band=(f1, f2)` on the "short" box mesh, about 10 modes) runs in
test_readme.py with every printed value asserted.

## Docs

- `docs/eigensolvers.md`:
  - a new section, "Interior bands and strategy":
    - the API and the checks;
    - the ladder, and the memory-driven rule with the measured evidence per D and N;
    - memory and capacity: small "memory" capacities, the store reservation;
    - `verify=None` for slices;
    - errors;
    - the stagnation warning;
    - the margin's recovery path;
    - δ_max on slices from 0;
  - §Frequency band [0, f_max], rewritten:
    - the Capacity bullet: free-and-refill growth, preallocated V;
    - the Returns bullet: a view, not a compact copy;
  - the Options block:
    - new rows for `strategy`, `memory_budget` and `slice_options`;
    - `band`, `n_modes`, `verify` and `precond_dtype` rewritten (the `band` row's "256 fp64
      columns up front" replaced);
  - the Validation list;
  - every mention of `15d`, `PR 15d` and `NotImplementedError`, and "the roadmap's crossover",
    in `docs/` and `README.md` rewritten (eigensolvers.md near lines 103, 122, 589–720).
- `README.md`: the `band=(f1, f2)` example, the options paragraph near line 222, the status.
- `docs/ROADMAP.md`:
  - status: the Status paragraph; PR 15d status; PR 15 complete and milestone M3b done; the
    dependency table (PR 18 gains 15d);
  - the strategy bullets: memory-driven; the ladder; "≲ 1200" replaced by the measured
    crossover; the 10M and 2M Mac figures from the record (fp32 AMG for PR 18);
  - a 15d note on $\alpha D(1 + D/D_0)$;
  - the δ_max ≤ 1 sentence: what holds (per-pair thresholds; slices from 0 can exceed tol;
    `r_acc` reported; enforcement → PR 18);
  - Memory budget: a shift-invert row; the Mac np.memmap bullet → PR 18;
  - PR 18 contents: the "band follow-ups (features)" and performance items above;
  - the deferred GPU checks: the default budget and the 1.2 margin, the GPU peaks of the new
    compositions, the cost ratio on the GPU, stagnation near depth 1000 at 10M.
- `docs/open_items.md`:
  - **closed:**
    - 14b's "no strategy selection or basis-fit check";
    - the 15c → 15d items done here: capacity from the budget, the `verify` callable (now
      `verify=None`), the edge rules, "15d words it per case" (the Late capacity errors
      bullet), the `verify` skips before the gather;
  - **superseded:** 14b's "Basis growth … no `max_modes` cap" and "Host V holds n_modes + chunk
    columns";
  - **partly addressed:**
    - "size the capacity with the window's chunk included" (the W = 2c reservation);
    - "span counts at small capacities" (the 96-column warning);
  - **retagged to PR 18 (no time-based cost model):**
    - "a done side gives up its half's gaps";
    - "requeue whole";
    - "σ = c_hi";
    - "stop si_checked before its count once r_acc widens the window";
  - the depth-1000 and σ-move items: warned, remedies to PR 18;
  - the decisions below, and the items not adopted.

## Decisions without user input (to log)

- **Memory-driven strategy, no time-based crossover.** LOBPCG to D + N is cheaper than any slice
  at every measured D and N (hex20 $16^3$ to depth 1005). This deviates from the roadmap's "cost
  model chooses" and "≲ 1200" (corrected). The roadmap's "slice widths and counting degree in
  the cost model" are replaced by capacity-limited widths and the per-leaf `verify=None`. A
  later crossover finding → PR 18. The GPU ratio is a deferred check.
- **Capacity stop:** si_band on the whole band, with a warning (depth, levers). The discarded
  LOBPCG run is bounded by its time to $C_{max}$. The hybrid junction → PR 18.
- **The ladder:**
  - from the top chunk down to 5;
  - with a known depth (f₁ = 0 with `n_modes`), the fastest chunk reaching it; otherwise the
    fastest within 10% of the best reach (threshold from benchmark part 2);
  - a given `chunk` replaces the ladder or gives "memory";
  - no automatic levers → PR 18.
- **Forced "lobpcg"** is capped by the budget and raises up front or at the cap.
- **Default budgets:**
  - the GPU free pool;
  - 0.8 × RAM on CPU, not reduced by the caller's holdings;
  - the GPU host budget 0.8 × RAM;
  - a 1.2 factor on every estimate, including the exact terms, at a cost of 17–20% of reach
    (revisited from test 17a's excess).
- **The LOBPCG growth** rebuilds B from the zeroed host V (PCIe about 3 s for 1000 columns at
  10M) instead of copying.
- **$f_1 = 0$ returns a view of V,** superseding 14b's compact copy.
- **Stagnation:**
  - warned only after a capacity stop at ≥ 1000 locked modes; the depth is unknown elsewhere;
  - remedies → PR 18;
  - the 10M criteria near index 1000 may need the user's input after the GPU checks.
- **Slice store and need:** `n_modes`, else 200, else after a capacity stop ≥ 1.25 L. There is
  never a Weyl estimate: anchored at μ₁ it is off by orders of magnitude (round-4 review).
  si_band raises when the store passes the host room. Below 96 columns, a warning.
- **The $\lambda_{lo} < \mu_1$ rule:**
  - the margin is max(1e-2, 300 tol);
  - μ₁^ub is the min of the available bounds, refined when only the 10-iteration estimate would
    fire it;
  - a `sigma=` hint never enters μ₁^ub.
- **δ_max ≤ 1 is not enforced** on slices from 0. Per-pair thresholds hold, and r_acc may
  exceed tol (reported); enforcement → PR 18.
- **`n_modes`** for an interior band is the expected band count; for $f_1 = 0$ it is a depth.
- **LOBPCG non-convergence** under "auto" gets no slice fallback (as PR 14b).
- **`solve_modes_memory`** is unchanged, and $f_1 > 0$ there → PR 18.
- **The spectrum bound** is exact, one tier, and runs after the operator build.
- **`err.result`** → PR 18; the error messages are worded per case now.
- **The slice cost form** is $\alpha D(1 + D/D_0)$; τ scaling is not modelled.
- **Not run:** the GMG depth-500 row and the slice-model refit.
- **np.memmap** → PR 18.
- **`precond_dtype="float32"`** applies to LOBPCG's GMG only; slices are always fp64.
- **Test tolerances:** slice tests run at tol 1e-6 with τ = 1e-8 (as in 15c), instead of the
  roadmap's r_acc = max(1e-10, …) test setting, for run time. The eigsh bound is asserted per
  mode.
