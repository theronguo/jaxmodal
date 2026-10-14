# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands), the Memory budget section and
the PR 15c plan's split (15d = the cost model, the memory budget, the strategy choice,
`verify=None` for bands and `solve_modes(..., band=(f1, f2))`).

Revision 5. Earlier rounds:
- round 1: one decision, no seeding, device/host terms, `slice_options`;
- round 2: memory-driven, no time crossover;
- round 3: no hybrid junction (PR 18), a chunk ladder, the basis rebuilt from the host V,
  F-order V, a 1.2 margin.

Round 4:
- the ladder picks by reach (C_max), not by "chunk 0 fits";
- no Weyl estimate for the slice store: `n_modes`, else the roadmap's 200 modes, with a runtime
  store check;
- LOBPCG always capped in band mode (forced "lobpcg" raises at the cap);
- the stop rule count + k > C_max;
- a zeroed V;
- the filter output after B is freed, and a view for f₁ = 0;
- bisection searches;
- a growth-footprint test and a slice-path peak test;
- test configurations fixed (chunk 5, the voxel mesh for "memory").

## Scope

- `solve_modes(..., band=(f1, f2))` with $f_1 \ge 0$, plus `strategy` ("auto", "lobpcg",
  "shift_invert"), `memory_budget` (bytes) and `slice_options` (si_band's options).
- "auto": PR 14b's band mode from 0 with the chunk of the ladder that reaches deepest in the
  budget, and the basis capped at $C_{max}$. It falls back to `si_band` (PR 15c) on the whole
  band in two cases:
  - LOBPCG stops at $C_{max}$ below $\lambda_{hi}$ ("capacity");
  - no chunk fits ("memory").
- The cost model, recorded in the docs and benchmark as the evidence for the memory-driven rule:
  - MINRES iterations against depth;
  - LOBPCG's measured time from 0 to depth 1005;
  - the smallest band for which one slice would beat LOBPCG.
- Memory:
  - `lobpcg_memory` and `si_memory` gain `device` and `host` entries;
  - $C_{max}$ and the slice capacity are bisected against the budget, with one unified check on
    CPU and a device and a host check on GPU.
- The $\lambda_{lo} < \mu_1$ rule (shift-invert from 0, filtered) and the up-front check of
  $\lambda_{hi}$ against an upper bound $\hat\lambda_\max$ of the spectrum.
- `si_band(verify=None)`: PR 14b's three-stage rule per leaf; si_band's store check.
- `si_band`'s capacity errors worded per case (PR 15c open item).
- `solve_modes_memory(..., band=(f1, f2))` with $f_1 > 0$, the README example, the docs.

Notation: $\lambda_{lo} = (2\pi f_1)^2$, $\lambda_{hi} = (2\pi f_2)^2$ (band edges); $\mu_1$
the lowest elastic eigenvalue, $\mu_1^{ub}$ an upper bound of it.

Out of scope (deferred, listed in `docs/open_items.md`):
- PR 18:
  - **the hybrid junction:** after a capacity stop, keep LOBPCG's in-band locked modes and slice
    only above them. Rounds 2–3 found it needs:
    - a chosen junction gap;
    - the lower neighbour plumbed out of si_checked and the span leaves;
    - a projection dedupe;
    - a combined count;
    - a host term.

    Today a capacity stop discards the LOBPCG run. That loss is bounded by LOBPCG's time to
    $C_{max}$, which costs less than slicing the band's first few modes (below);
  - PR 15c's "Late capacity errors and slice efficiency" items;
  - `err.result`;
  - a memory-mapped host store (the roadmap's np.memmap promise moves there);
  - automatic `low_memory` / fp32-basis / on-the-fly-Ke levers in "auto";
  - a user-facing slice memory estimate.
- PR 18 and the deferred GPU checks:
  - the depth-1000 remedies (15d flags the risk only);
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
  - "auto" (default);
  - "lobpcg": PR 14b's band mode, filtered. It is capped at $C_{max}$ like "auto", but a capacity
    stop raises `ValueError` naming `memory_budget` and the levers (`low_memory`, `basis_dtype`,
    on-the-fly Ke);
  - "shift_invert": `si_band` on the band.

  Without `band`, any `strategy` but "auto", a `memory_budget` or `slice_options` raises.
- `memory_budget`: bytes.
  - Default on GPU: `bytes_limit − bytes_in_use` at entry (`memory_stats()`). This is the usable
    pool under `XLA_PYTHON_CLIENT_MEM_FRACTION` minus what the caller holds.
  - Default on CPU: 0.8 × physical RAM (`os.sysconf`; unified memory on the Mac; 55.0e9 bytes on
    the 64 GiB Mac). It is not reduced by what the caller holds (documented).
  - The GPU host budget is 0.8 × physical RAM.
  - The searches compare 1.2 × the estimate with the budget. The estimators run up to 9–16% low
    (si_memory 1.09–1.11×, GMG LOBPCG 1.12×, a one-chunk zero-workspace solve 1.16×), and 1.2 is
    the roadmap's ±20% validation band.
  - Recorded in `info["budget"]`.
- `slice_options`: dict with keys `tau`, `maxit_inner`, `block`, `max_slices`, `capacity`
  (overrides the sizing), `edges` and `verbose` (validated as si_band does).
- How the other `solve_modes` arguments reach the solvers:
  - `tol`, `verify`, `count_options`, `seed` → si_band;
  - `precond`, `precond_options` → si_band's `make(σ)`, always fp64. With
    `precond_dtype="float32"`, only LOBPCG's GMG is fp32, since MINRES needs a symmetric cycle;
  - `basis_dtype`, `low_memory`, `chunk` → LOBPCG and $C_{max}$. A given `chunk` replaces the
    ladder;
  - `sigma` → LOBPCG's preconditioner and $\mu_1^{ub}$.
- `n_modes`: for $f_1 > 0$, the expected modes in the band (the slice store and need). For
  $f_1 = 0$, as PR 14b (the expected count, the top of the ladder via `band_defaults`).
- Returns the band modes and `info`.
  - The band filter:
    - Upper end: LOBPCG's `n_band` (its own chain rule) or si_band's set.
    - Lower end: λ ≥ λ_lo(1 − 100t), extended down over chains with relative gaps < 100t
      (si_band's `_extend_chain`), with t LOBPCG's per-mode threshold or r_acc.
  - `strategy` ∈ {"lobpcg", "shift_invert"};
  - `reason` ∈ {"forced", "fits", "capacity", "memory"};
  - `chunk`, `c_max` (per ladder chunk), `stagnation_risk`, `budget`, `from_zero`, `mu1_ub`;
  - `count`: PR 14b's, or si_band's summary;
  - with slices: si_band's `slices`, `capacity`, `merged_edges`, `collapsed`, `edge_rr_failed`,
    `duplicates`, `r_acc`.

### Up-front checks (before the σ pre-estimate and any build)

- **Spectrum bound** $\hat\lambda_\max = \max_e \lambda_\max(K_e, M_e)$.
  - The bound holds: for any $x$, $x^\top K x = \sum_e x_e^\top K_e x_e \le \max_e \lambda_e
    \sum_e x_e^\top M_e x_e \le \hat\lambda_\max\, x^\top M x$. Nodal masses, masking and rigid
    deflation only lower the Rayleigh quotient.
  - Computed exactly by batched Cholesky and `eigvalsh`:
    - voxel: one reference element per phase on the host, times $\max_e s_{K,e}/s_{M,e}$ (a void
      cell with $s_K = s_M = 0$ counts 0);
    - unstructured: `element_matrices` (PR 12) in device chunks, about 200 µs per hex20 element.
  - A singular $M_e$ with $s_K > 0$ gives ∞ (the check never raises).
  - $\lambda_{hi} > \hat\lambda_\max$ raises `ValueError`. An edge in $(\lambda_\max,
    \hat\lambda_\max]$ still reaches si_band's capacity error or PR 14b's "more modes than the
    mesh provides".
  - Time recorded in `info["time"]["bound"]`. The tests keep `lam_hat` = ‖K‖/‖M‖ for the eigsh
    tolerance.
- **$\lambda_{lo} < \mu_1$** on every slice path.
  - $\mu_1^{ub}$ is the min of the available upper bounds:
    - the σ pre-estimate (a Rayleigh quotient);
    - `sigma_estimate` (10 iterations) when a `sigma=` hint or Jacobi skips it;
    - after LOBPCG ran, its lowest converged elastic Ritz value (locked or not).
  - If $\lambda_{lo} < \mu_1^{ub}(1 + m)$ with $m = \max(10^{-2}, 300\,\text{tol})$, si_band runs
    on $[0, \lambda_{hi}]$ and the result is filtered; `info["from_zero"]` is True. The margin
    keeps $\mu_1$ out of si_band's widened first-slice window (lo(1 − 100 r_acc), r_acc ≥ tol).

### Strategy ("auto")

1. **Ladder.** The candidate chunks are the top chunk (`band_defaults`' chunk, 40 by default),
   halved down to 5. For each chunk, $C_{max}$ (below) is computed; a chunk fits if
   $C_{max} \ge$ up(4 chunk + k, c). The choice is the largest fitting chunk whose $C_{max}$
   ≥ 0.9 × the best $C_{max}$ among the fitting chunks: the fastest chunk that gives up at
   most 10% of the reach. With no fitting chunk, the reason is "memory": si_band on the whole
   band, and a warning names the levers and their $C_{max}$ from the estimators. A given `chunk`
   is used if it fits, else "memory".
2. **LOBPCG** `_solve(..., lam_stop=λ_hi, max_capacity=C_max)`. In band mode `max_capacity` is
   required (solve_modes always passes it).
   - Basis growth: `new = min(max(1.5 old, need), C_max)`. The initial allocation is
     `min(up(n_modes + k, c), C_max)`.
   - At a growth, `B`, `args` and every other reference to B are dropped. B is reallocated
     with zeros and refilled with V's first `count` columns in chunks of c (a C-order host
     transient of c columns per chunk), cast to `basis_dtype`, so it equals the old B exactly.
   - The host V is `np.zeros((n, C_max + k), order="F")`, allocated once. Calloc is lazy, so
     untouched columns cost no pages. It is never concatenated, and the result is the view
     `V[:, :end]`.
   - The band-end check comes first. A lock needing count + k > $C_{max}$ stops the solve:
     `info["stopped"] = "capacity"`.
   - Reaching $\lambda_{hi}$ ("fits"): PR 14b's count at the cut above $\lambda_{hi}$, then the
     filter. For $f_1 = 0$ it returns the view. For an interior band it copies its columns into
     an F-order output after B, the state and the preconditioner are freed.
   - Non-convergence ends the solve with `converged` False, as PR 14b: no fallback.
3. **Capacity stop** ("capacity"). LOBPCG's basis, state, preconditioner and V are freed after
   reading $\mu_1^{ub}$ and the top locked pair. si_band then runs on
   $[\lambda_{lo}, \lambda_{hi}]$, or from 0 by the rule above.
4. **Stagnation.** After a capacity stop, the band's top lies deeper than the locked count. If
   that count ≥ 1000, a warning is issued and `stagnation_risk` is True. On the "memory" and
   forced paths the depth is unknown and no flag is set (documented); si_band's MINRES
   stagnation flags (τ_eff) still report it. Basis: 15a measured a MINRES solve near depth 1005
   as marginal at τ = 1e-8 on hex20/AMG.

There is no time-based choice. For every measured depth D and band size N, LOBPCG's measured
time from 0 to D + N is below one slice's cost,
$(84 + 1.8N)/b$ steps × $\alpha D(1 + D/D_0)$ iterations × $t_\text{minres}$. Example: hex20
$16^3$/AMG at D ≈ 1000, LOBPCG from 0 measured 6.1 min, against about 2.3 h for one slice's
fixed start alone. If benchmark part 2 finds a finite crossover with a slice gain above 2×, a
time rule is added. Below that, the memory-driven rule stays (LOBPCG's single count and
robustness), and the outcome is logged.

### Cost model (benchmark and docs only; no constants in code)

- MINRES iterations per solve: $\alpha D(1 + D/D_0)$, fitted on the frac-0.5 non-floor rows of
  PR 15a's table. The depth-1005 row uses `recheck_1005` (converged; the frac-0.5 row
  stagnated at the same iteration count).
  - hex20/AMG: about α = 5.8, $D_0$ = 650, ratios 0.87–1.20 over depths 6–1005.
  - voxel/GMG (depths ≤ 201): α = 4.35, $D_0$ about 760, weakly constrained.
  - This measured form replaces the roadmap's "∝ ln(1/τ)/δ plus an outlier term" (15a: depth
    drives the iterations).
- LOBPCG from 0, measured per chunk to about 1005 modes on hex20 $16^3$/AMG:
  - iterations per chunk rise from 18 to 77;
  - $t_\text{warm}$ rises from 0.215 to 0.278 s (+30% projector growth at depth 1000 on this
    mesh).
- The crossover table, per depth D: the smallest N for which a slice beats LOBPCG to D + N
  ("none" expected). Per mode, a slice costs about 25–110 LOBPCG modes.
- Stated in the docs: the ratio is measured on 56k DOF. At 10M and depth about 1000, LOBPCG's
  projector grows further (about 4nDk flops per iteration), but MINRES grows faster (×350 from
  depth 6 to 1005).

### Memory

- **The split.** Both estimators gain `device` and `host` entries:
  - `lobpcg_memory`: host = $8n(\text{n\_modes} + k\cdot\text{band})$, the V. Device = total −
    host, its start, seed and slack blocks.
  - `si_memory`: host = $8n\,\text{n\_deflate} + 8C^2 + 8n(\text{n\_modes} + 2)$. Device = the
    rest; the end phase holds one device chunk.
  - Unified memory (CPU): device + host ≤ budget. GPU: device ≤ budget, host ≤ the host budget.
- **$C_{max}(\text{chunk})$.** The largest multiple of c whose estimate × 1.2 fits, found by
  bisection over multiples of c. The estimate is `lobpcg_memory(..., n_modes=C − k, band=True)`
  plus the interior band's filter output, which runs after B is freed and so enters the `max`
  with the phases, not the sum; on GPU it is host only. The cached models are evaluated once per
  width. Bisection is exact without affinity (the host term jumps at cap > 0).
- **Slice capacity $C$** = min(the budget cap, the need cap):
  - **the budget cap:** the largest multiple of c with `si_memory(op_memory, n, C − W,
    n_modes=⌈1.25 m_s⌉, block, pc, n_rigid, n_deflate=W)` × 1.2 fitting, bisected. W = 2c, so the
    device basis is exactly C. The host term adds $2m_s$ columns (a span-rerun window and
    si_checked's [W | V] restart copy), with $m_s = \max(1, (C - W - 84)/1.8)$. The store is
    reserved as $8n\,N_s$ with $N_s$ = `n_modes` if given, else 200 (the roadmap's assumed band
    size);
  - **the need cap:** up(1.25 (1.8 $N_s$ + 84) + W, c);
  - si_slice already stops at the free space (no third cap).
- **Capacity limits:**
  - below 96 columns (PR 15c's slice minimum at b = 6): a warning. This is the common outcome
    of "memory" on AMG meshes, since a slice costs about the smallest LOBPCG chunk (si_memory /
    LOBPCG chunk-5 ratio 0.76–1.06 at C = 48–144);
  - below 2b + c: a `ValueError` naming the budget;
  - `slice_options["capacity"]` overrides the sizing.
- **si_band's store check.** When the stored modes exceed the reserved $N_s$ by 50%, a warning
  names `n_modes`. When the host store would pass the remaining host budget, a `ValueError`
  names `n_modes` and `memory_budget`.
- **Probe blocks:** PR 14b's count is unchanged; si_band's leaf counts are as 15c.
- **`solve_modes_memory(..., band=(f1, f2))`** accepts $f_1 > 0$ and keeps PR 14b's LOBPCG shape.
  The capacity search returns its `si_memory` dict with C, which the tests check. No new
  `strategy`, `memory_budget` or `slice_options` there.

### `si_band(verify=None)`

`verify` ∈ {True, False, None}. None applies PR 14b's rule per leaf:
- skip above `MAX_DEGREE`;
- skip when the a-priori bound $2\,n_\text{blocks}\,d\,(b_p/b)\,t_\text{minres}$ exceeds the
  leaf's time (0.3 instead of 2 when τ_c is looser than the rule, as `_bound`);
- else calibrate (PR 14b's `_calibrate` with the leaf's count closure and preconditioner), and
  count iff the prediction ≤ the leaf's time.

A run leaf's time is its slice's; a span's is its parent run's. $t_\text{minres}$ is the leaf's
slice's measured seconds per MINRES iteration. `_bound` and `_calibrate` move to `counting.py`,
and the tests patch them there. Every leaf records `predicted_s`, `predicted_kind` and `reason`.

### Errors

si_band's `_cap_error` is worded per case:
- "lo may be below lambda_1" only when no lower side closed and lo > 0;
- "hi above the spectrum" only when no upper side closed;
- lo = 0 names only hi.

## Changes to existing code

- `lobpcg._solve`:
  - band mode requires `max_capacity`, which sets the initial allocation, the growth by
    free-and-refill from V, and the count + k stop;
  - V is `np.zeros` in F order, preallocated, with the view returned;
  - non-band mode is unchanged.
- `si_band`: `verify=None`, the store check, the `_cap_error` wording.
- `counting.py`: `_bound`, `_calibrate` moved from `api.py`.
- `api.py`:
  - `solve_modes`: `band=(f1, f2)`, `strategy`, `memory_budget`, `slice_options`, the ladder,
    the checks, the fallback, the filter;
  - `check_band` for $f_1 > 0$;
  - the NotImplementedError removed;
  - `spectrum_bound(mesh, materials, ...)`.
- `memory.py`:
  - the `device`/`host` entries;
  - cached models;
  - the bisection searches for $C_{max}$ and the capacity;
  - the default budget;
  - `solve_modes_memory` accepting $f_1 > 0$.
- Direct band `_solve` callers in benchmarks and tests pass `max_capacity`.

## Benchmark (`benchmarks/strategy_study.py`, one invocation, ≤ 15 min, progress to a log)

1. **[15a, 15d] constants:** refit α and $D_0$ per family on `shift_invert_study.json` (no runs,
   seconds).
2. **[15d] crossover:** one LOBPCG band run on hex20 $16^3$/AMG to about 1005 modes (about
   6 min measured), recording per-chunk iterations and $t_\text{warm}$. From it, the crossover
   table per depth against the MINRES rows. The held-out "full" slice row is predicted against
   measured (about 1.4×).
3. **[15d] memory record** (estimators only, seconds with the cache): the ladder (chunk 40 to 5:
   fits and $C_{max}$) × {9-block, low} × {fp64, fp32 basis} × {stored, on-the-fly Ke}, at:
   - 10M hex20 on the 76 GB pool;
   - 1M hex20 and 2M voxel on the Mac.

   It checks and updates the roadmap's 10M figures (700–1000) and the 2M Mac figure (> 2000).

Results go to `benchmarks/results/strategy_study.json` with the git commit.

Not run: the GMG depth-500 row and the slice-model refit. They refine predictions that decide
nothing (logged). The measured peaks are tests (17a, 17b), not benchmark parts.

## Tests (`tests/test_strategy.py`; roadmap tags in the docstrings)

Method:
- Real counts use the dense exact $(K + \sigma M)^{-1}$ (15c's `exact`) monkeypatched into
  `api.amg`/`api.gmg` and si_band's `make`. Each count's degree ≤ about 1000 is asserted as a
  precondition (≤ 15 s per count on "short").
- Wiring-only tests use `count_options={"degree": 4, "n_probes": 2}`.
- Budgets come from the search functions (with the 1.2 factor), never hard-coded.
- Each test is timed and marked slow above about 20 s.
- Every test mapped to the eigsh item asserts the roadmap bound per mode with that mode's t,
  through test_shift_invert's `check_band` logic generalised.

1. **"fits"** ("short", hex20 $6\times5\times4$): an interior band of about 10 modes inside chunk
   0. Asserted:
   - reason "fits", `from_zero` False;
   - `info["chunk"] == 40` with `n_modes` given;
   - matches eigsh.
2. **"capacity"** ("short"):
   - `chunk=5` (k = 10, c = 40), budget at the top of the $C_{max} = 40$ window (81.06–82.74 MB
     × 1.2 measured);
   - preconditions: slice capacity ≥ 96, LOBPCG's locked count < the band's top index;
   - band about modes 25–45;
   - asserted:
     - `stopped == "capacity"`;
     - si_band's (lo, hi) = (λ_lo, λ_hi) (spy);
     - at si_band's entry, no live array of LOBPCG's basis shape (`jax.live_arrays()`);
     - matches eigsh, slice counts < 0.5;
     - `mu1_ub` equals LOBPCG's lowest Ritz value and differs from the pre-estimate.
   - At the same budget, `strategy="lobpcg"` raises at the cap naming `memory_budget`.
3. **"memory"** (the voxel 10³ mesh, GMG):
   - budgets at 1.15× and 1.25× the chunk-5 estimate give "memory" and LOBPCG (brackets the
     1.2 factor);
   - precondition: the slice capacity at the 1.15× budget ≥ 96;
   - with $\lambda_{lo} < \mu_1$: `from_zero` True, matches eigsh, `info["capacity"]` as
     searched;
   - the ladder:
     - a budget between 1.2× the chunk-20 and chunk-40 needs gives `info["chunk"] == 20`;
     - a pure-function table checks the 0.9 × best-reach rule.
4. **"shift_invert"** with 15c test 2(a)'s configuration (modes 5–28, block 6, its seed):
   - `n_modes` given, so the store and need are fixed;
   - the budget from the search such that `capacity == 96`;
   - asserted: ≥ 2 runs, every leaf counted < 0.5, the same set as "lobpcg";
   - one real-AMG copy marked slow.
5. **Empty band:** strictly between two well-separated modes of the voxel "cube" (named gap).
   0 modes and count < 0.5 under both strategies, `from_zero` False.
6. **Clamped box**, `band=(0, f)`, "shift_invert": matches eigsh.
7. **$\lambda_{lo} < \mu_1$**, clamped and free, "shift_invert":
   - `from_zero` True, filtered, matches eigsh;
   - the margin: λ_lo = μ₁^ub(1 + 0.9e-2) gives True, (1 + 1.1e-2) gives False (μ₁^ub
     monkeypatched ≥ the true μ₁), and tol = 1e-4 widens it to 3e-2;
   - correct with an inflated μ₁^ub;
   - the `sigma=` hint and Jacobi paths (Jacobi on the 240-DOF "free" hex8).
8. **Filter table** (pure): the exact edge, inside and outside 100t, a chain crossing λ_lo, per-mode
   t, and the upper end taken from `n_band`.
9. **Lower-edge clusters:** the voxel cube (doubles, triples) with λ_lo on a double and mid
   triple chain, both strategies. All copies returned, `from_zero` False.
10. **Spectrum bound:**
    - ≥ the dense largest eigenvalue on:
      - an unstructured graded `box_mesh` with nodal mass, its stiffest element in a forced
        partial last chunk;
      - a two-phase scaled voxel whose stiffer phase is not phase 0;
    - equal on a one-element mesh without BCs;
    - a massless phase gives ∞, a void cell counts 0;
    - $\lambda_{hi} > \hat\lambda_\max$ raises before any build (spies on `_solve`, `si_band`,
      the builders, `_precond_sigma`);
    - just below does not raise up front.
11. **Capacity and $C_{max}$:**
    - one hand-computed device and host term per estimator;
    - the searches against brute-force loops, including a tiny mesh where 8C² and the end phase
      matter and the $C_{max}$ jump at cap > 0;
    - the need cap and the budget cap each binding;
    - unified and GPU (monkeypatched platform, `memory_stats`);
    - below 2b + c raises naming the budget, below 96 warns;
    - an override above the need is kept;
    - `solve_modes_memory` with $f_1 > 0$ keeps PR 14b's shape.
12. **`_solve(max_capacity=...)`** directly:
    - the capacity sequence (growth straight to the cap);
    - a lock with count + k = $C_{max}$ is allowed, count + k = $C_{max}$ + 1 stops;
    - the initial allocation is capped;
    - V is F order and zeros, with no concatenate (spy);
    - after a refill, B equals V cast and its columns past count are 0;
    - `_insert` (the old copy path) is not called at a growth (spy);
    - `max_capacity` is required in band mode.
13. **Cost-model record:** `strategy_study.json`'s constants equal a refit of
    `shift_invert_study.json` (frac 0.5, non-floor, `recheck_1005`), with ratios in
    [0.8, 1.25]; the crossover entries are present.
14. **`si_band(verify=None)`:**
    - `fake_band`: each branch (MAX_DEGREE skip; bound skip with 2 and with 0.3; calibrated and
      dear, skip; calibrated and cheap, count; `_calibrate` spied); a span leaf's time is its
      parent's;
    - one real exact-preconditioner run: per-leaf `reason`/`predicted_kind`, `ran` True,
      estimate < 0.5;
    - `solve_modes(strategy="shift_invert")` at the default verify records them;
    - slow, real AMG on one leaf (hex20 $12^3$, as 14b's `test_calibration`): the bound ≤ the
      measured warm count time, the calibrated prediction within [0.5, 2]×;
    - the store check (fake): warns past 1.5 $N_s$ and raises past the host room.
15. **Capacity error wording** (`fake_band`): lower side closed, upper side closed, lo = 0.
16. **Options:**
    - `strategy`/`memory_budget`/`slice_options` without `band`, bad values, unknown
      `slice_options` keys, `f1 >= f2`;
    - the default budgets (monkeypatched `os.sysconf`, a fake `memory_stats` with
      `bytes_in_use`);
    - spies: `slice_options`, `tol`, `count_options`, `precond_options` reach si_band and
      `make`;
    - `precond_dtype="float32"` on voxel through a slice path builds `make` in fp64;
    - a capacity stop with ≥ 1000 locked modes (monkeypatched) warns and sets
      `stagnation_risk`;
    - LOBPCG not converged under "auto": no fallback, `converged` False.
17. **Mac peaks** (slow, darwin, fresh subprocess, MallocLargeCache=0, estimates ≥ 2 GB):
    - **(a)** `_solve(max_capacity=...)` on voxel about $48^3$ with a small `n_modes`, so it
      grows after 2 chunks. The footprint increase across the growth is ≤ (new − old) · 8n plus
      a small slack (no old+new transient). The peak is ≤ the budget and within 20% of the
      estimate;
    - **(b)** `solve_modes(strategy="shift_invert", band=..., memory_budget=B)` on lumped voxel
      $80^3$, modes about 4–8, the capacity from the budget. The peak is ≤ B and within 20% of the
      composed estimate.

## Docs

- `docs/eigensolvers.md`:
  - a new section, "Interior bands and strategy":
    - the API and the checks;
    - the ladder and the memory-driven rule with the measured evidence (per D and N);
    - memory and capacity (the small "memory" capacities, the store reservation);
    - `verify=None` for slices, the errors, the stagnation flag;
  - the Options block: the new rows, plus `band`, `n_modes`, `verify` and `precond_dtype`
    rewritten;
  - the Validation list;
  - every `15d`, `PR 15d` and `NotImplementedError` mention, and "the roadmap's crossover", in
    `docs/` and `README.md` rewritten (eigensolvers.md near lines 103, 122, 589–720).
- `README.md`: the `band=(f1, f2)` example, the options paragraph near line 222, the status.
- `docs/ROADMAP.md`:
  - the Status paragraph, PR 15d status, PR 15 complete, the dependency table;
  - the strategy bullets: memory-driven, the ladder, "≲ 1200" replaced by the measured
    crossover;
  - the 10M figures and the 2M Mac "> 2000" figure from the record;
  - a 15d note on $\alpha D(1 + D/D_0)$;
  - a shift-invert row in Memory budget; the Mac np.memmap bullet → PR 18;
  - δ_max ≤ 1 for slices from 0 (σ inside the slice);
  - PR 18 contents: the deferred items above;
  - the deferred GPU checks: the default budget and the 1.2 margin, the GPU peaks of the new
    compositions, the cost ratio on the GPU, stagnation near depth 1000 at 10M.
- `docs/open_items.md`:
  - close 14b's "no strategy selection or basis-fit check";
  - close the 15c → 15d items done here: capacity from the budget, the `verify` callable (now
    `verify=None`), the edge rules;
  - give each 15c item that handed a choice to 15d's cost model its outcome:
    - "a done side gives up its half's gaps", "requeue whole" and "σ = c_hi": retagged to PR
      18, because there is no time-based cost model;
    - "span counts at small capacities": partly addressed by the 96-column warning;
  - the depth-1000 and σ-move items: flagged, remedies to PR 18;
  - the decisions below and the items not adopted.

## Decisions without user input (to log)

- **Memory-driven strategy, no time-based crossover.** LOBPCG to D + N is cheaper than any slice
  at every measured D and N (hex20 $16^3$ to depth 1005). This deviates from the roadmap's "cost
  model chooses" and "≲ 1200" (corrected). The rule for a crossover found later is above. The
  GPU ratio is a deferred check.
- **Capacity stop:** si_band on the whole band. The discarded LOBPCG run is bounded by its time
  to $C_{max}$. The hybrid junction → PR 18.
- **The ladder:**
  - from the top chunk down to 5, choosing the fastest chunk within 10% of the best reach, with
    a 4-chunk minimum basis;
  - no automatic `low_memory`, fp32 basis or on-the-fly Ke (they change time or accuracy; the
    warning names them) → PR 18.
- **Forced "lobpcg"** is capped by the budget and raises at the cap.
- **Default budgets:**
  - the GPU free pool;
  - 0.8 × RAM on CPU, not reduced by the caller's holdings;
  - the GPU host budget is 0.8 × RAM;
  - a 1.2 factor on every estimate.
- **The LOBPCG growth** rebuilds B from the zeroed host V (PCIe about 3 s for 1000 columns at
  10M) instead of copying.
- **Stagnation:**
  - flagged only after a capacity stop at ≥ 1000 locked modes; unknown depth elsewhere;
  - remedies → PR 18;
  - the 10M criteria near index 1000 may need the user's input after the GPU checks.
- **Slice store and need:** `n_modes` or 200 modes, never a Weyl estimate (anchored at μ₁ it is
  off by 0.2–10⁸×, measured), with si_band's runtime store check. A warning below 96 columns.
- **The $\lambda_{lo} < \mu_1$ margin** is max(1e-2, 300 tol). μ₁^ub is the min of the available
  bounds.
- **`n_modes`** for an interior band is the expected band count.
- **`solve_modes_memory`** gains only $f_1 > 0$ (no slice estimate → PR 18).
- **The spectrum bound** is exact, one tier.
- **`err.result`** → PR 18; the error messages are worded per case now.
- **The slice cost form** is $\alpha D(1 + D/D_0)$; τ scaling is not modelled.
- **Not run:** the GMG depth-500 row and the slice-model refit.
- **np.memmap** → PR 18.
- **`precond_dtype="float32"`** applies to LOBPCG's GMG only (slices always fp64).
- **Test tolerances:** slice tests run at tol 1e-6 with τ = 1e-8 (as 15c), instead of the
  roadmap's r_acc = max(1e-10, …) test setting, for run time. The eigsh bound is asserted per
  mode.
