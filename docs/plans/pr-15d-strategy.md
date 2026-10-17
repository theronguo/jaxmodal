# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands), the Memory budget section and
the PR 15c plan's split (15d = the cost model, the memory budget, the strategy choice,
`verify=None` for bands and `solve_modes(..., band=(f1, f2))`).

Revision 9. Rounds 1–7:
- one decision, no seeding;
- memory-driven, no time crossover;
- no hybrid junction;
- a chunk ladder by reach;
- the basis rebuilt from a zeroed F-order host V;
- `n_modes`/200 store sizing;
- bisection searches;
- μ₁^ub refinement;
- view returns;
- no up-front spectrum bound;
- the slice device/host window split;
- budgeted slice counts;
- `band_count(_steps=)`;
- the capped initial allocation.

Round 8:
- **The fit test** is applied per memory space, each with its own exact part; CPU uses the
  estimators' `total`.
- **`fallback_capacity` is dropped** (noisy, optimistic for $f_1 = 0$, undefined below the
  floor).
- **The growth rule** keeps the rounding to $c_L$.
- **The refill** uses `device_put(V[:, i:i+c].T)` (zero-copy) with the transpose in `_insert`.
- **`si_memory`'s `n_modes`** becomes the store's columns, so there is one store term and the
  Gram term uses the full Q width.
- **The count block's fit** is stated per space.
- **The exact terms** get 1.05 against column-proportional allocator growth, validated by 17a.
- **The μ₁ refinement** falls back to `sigma_estimate` at a singular hint.
- **A `verify=True` slice path** where one probe does not fit runs without counts.
- **Tests:**
  - 2(c)'s `n_modes`;
  - test 3's budgets defined as exact + s × modelled;
  - the count-block spy moved to test 3;
  - 17a at old = 14 $c_L$;
  - 17b at `verify=False` plus a new 17c count peak;
  - test_band.py's view and patch-target updates.
- **Docs:** the M3b example script and the remaining passages.

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
- Memory:
  - `lobpcg_memory`, `si_memory` and `count_memory` gain per-space `device`, `host`,
    `device_exact` and `host_exact` entries;
  - `lobpcg_memory` gains `capacity=`, and `si_memory` gains `host_window=`;
  - $C_{max}$, the slice capacity and the slice count block are searched against the budget
    under the fit test.
- The $\lambda_{lo} < \mu_1$ rule (shift-invert from 0, filtered).
- `si_band(verify=None)`: PR 14b's three-stage rule per leaf; `band_count(_steps=)`.
- `si_band`'s capacity errors worded per case (PR 15c open item).
- The milestone M3b example `examples/m3b_bands.py`, the README example, the docs.

Notation:
- $\lambda_{lo} = (2\pi f_1)^2$ and $\lambda_{hi} = (2\pi f_2)^2$: the band edges;
- $\mu_1$: the lowest elastic eigenvalue; $\mu_1^{ub}$: an upper bound of it;
- LOBPCG:
  - `chunk`: the locking chunk;
  - k = chunk + guards: the columns one lock may write;
  - $c_L$: `lobpcg._capacity`'s projector chunk (k in the low-memory step, else max(k, min(4k,
    64)));
  - $C_{max}$: its basis cap, a multiple of $c_L$;
  - L: the locked modes at or above $\lambda_{lo}$ at a capacity stop (all locked for
    $f_1 = 0$);
- slices:
  - b: si_band's block (6);
  - c = `_chunk(b)`: the Lanczos chunk (24 at b = 6);
  - C: the slice capacity;
  - W = 2c: the device window inside C;
  - $N_s$: the host store reservation in columns;
- n: DOFs; up(x, c): x rounded up to a multiple of c.

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
  - **An up-front spectrum bound** $\max_e \lambda_\max(K_e, M_e)$ (voxel per phase is cheap;
    unstructured costs about 170 s at 10M).
  - **The depth-1000 remedies** (15d warns only).
  - **A time rule,** if a crossover is ever measured.
- **PR 18, performance:** PR 15c's "Late capacity errors and slice efficiency" items, and
  `verify=False` predicting before the gather.
- **Deferred GPU checks:**
  - the default budget and the fit test's factors;
  - the GPU peaks of the new compositions, including growth fragmentation under the BFC
    allocator and the refill's PCIe time;
  - the cost ratio on the GPU;
  - stagnation near depth 1000 at 10M.

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
| Band edges: an edge above the largest computed eigenvalue raises | 10 (through solve_modes); 15c tests 3, 11 |
| Examples: each milestone adds its own example script | test_examples.py (m3b) |

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
    - The remaining 0.2 × RAM (13.7 GB) also holds macOS's large-block malloc cache: freed numpy
      blocks stay in the footprint, up to about 10 GB measured. The tests run with
      MallocLargeCache=0 and cannot see it (documented, logged).
  - The GPU host budget is 0.8 × physical RAM.
  - **The fit test,** per check: 1.05 × exact + 1.2 × (total − exact) ≤ budget.
    - The exact part is the column arrays: basis, V, the slice window and store, R and M R,
      stored Ke. Its 1.05 covers allocator growth proportional to the columns. Test 17a
      measures it under the preallocated V; eigensolvers.md records 0.15 × 8n × n_modes with
      PR 14b's concatenated V.
    - The modelled part is step, workspace, kernel temporaries, builds and preconditioners.
      These run up to 9–16% low (si_memory 1.09–1.11×, GMG LOBPCG 1.12×, a one-chunk
      zero-workspace solve 1.16×), and 1.2 is the roadmap's ±20% validation band.
    - Unified memory (CPU): one check on the estimator's `total` (the phase max taken across
      spaces) with exact = `device_exact + host_exact`.
    - GPU: `device` with `device_exact` against the budget, and `host` with `host_exact`
      against the host budget.
  - Recorded in `info["budget"]`.
- `slice_options`: dict with keys `tau`, `maxit_inner`, `block` (default 6, the roadmap's b),
  `max_slices`, `capacity` (overrides the sizing), `edges` and `verbose`, validated as si_band
  does.
- The other `solve_modes` arguments reach the solvers as follows:
  - `tol`, `verify`, `count_options`, `seed` → si_band. On slice paths the count `block`
    defaults to the budgeted one (below);
  - `precond`, `precond_options` → si_band's `make(σ)`, always fp64. With
    `precond_dtype="float32"`, only LOBPCG's GMG is fp32, since MINRES needs a symmetric cycle;
  - `basis_dtype`, `low_memory`, `chunk` → LOBPCG and $C_{max}$. A given `chunk` replaces the
    ladder;
  - `sigma` → the preconditioners' shift. A hint never enters $\mu_1^{ub}$.
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
    - λ is ascending, so the kept set is one range [i0, i1). Both strategies return the
      F-contiguous view `V[:, i0:i1]` of their F-order output; no copy is made. The columns
      below i0 stay resident behind the view; they were resident during the run, so the peak
      does not change (documented).
  - **Fields:**
    - `strategy` ∈ {"lobpcg", "shift_invert"};
    - `reason` ∈ {"forced", "fits", "capacity", "memory"};
    - `chunk`, and `c_max` (a dict {chunk: $C_{max}$});
    - `stagnation_risk`, `budget`, `mu1_ub`, `n_store` ($N_s$);
    - `from_zero`: True when si_band ran on $[0, \lambda_{hi}]$ (always the case for $f_1 = 0$
      on a slice path); False on LOBPCG and on slices from $\lambda_{lo}$;
    - `count`: PR 14b's, or si_band's summary;
    - with slices: si_band's `slices`, `capacity`, `merged_edges`, `collapsed`,
      `edge_rr_failed`, `duplicates`, `r_acc`.

### The $\lambda_{lo} < \mu_1$ check (before si_band's `make(σ)`)

Every slice path makes this check.

- **Bounds.** $\mu_1^{ub}$ is the min of the available upper bounds, each a Rayleigh quotient
  on R-deflated vectors:
  - the σ pre-estimate (`_precond_sigma`);
  - after LOBPCG ran, its lowest converged elastic Ritz value.
- **Refinement.** When neither exists ($f_1 > 0$ with a `sigma=` hint or Jacobi), one is
  computed directly:
  - multigrid: `lobpcg(op, free, 1, 0.1, R, P, maxit=20)` with the P built at the hint (as
    `_precond_sigma`; measured within about 5e-6 relative of μ₁). P is dropped before
    si_band's `make(σ)`. A hint whose `make` raises `SingularShiftError` (σ = 0 on a free
    body) falls back to `sigma_estimate`, as `_precond_sigma` does;
  - Jacobi: `sigma_estimate` run to its stop rule (≤ 8% high measured).
- **The rule.** If $\lambda_{lo} < \mu_1^{ub}(1 + m)$ with $m = \max(10^{-2}, 300\,\text{tol})$,
  si_band runs on $[0, \lambda_{hi}]$ and the result is filtered.
- **Why it is safe.** Every bound is a Rayleigh quotient, so a loose bound only fires the rule
  when it need not (time, not correctness). When the rule does not fire, $\lambda_{lo} >
  \mu_1(1 + m)$.
- **Why the margin.** It keeps μ₁ out of si_band's widened first-slice window
  [lo(1 − 100 r_acc), …] when $100\,r_{acc} < m/(1 + m)$. Two cases break that:
  - a wide first slice (r_acc ∝ τ δ, δ ≈ λ_hi/(2μ₁));
  - a chain extension over gaps < 100 r_acc reaching μ₁.

  `_split` then recovers with narrower runs if their r_acc shrinks. Otherwise it ends in
  `_cap_error("no progress")`, which names tau and the margin (documented).

**$\lambda_{hi}$ above the spectrum** is caught late, as before:
- LOBPCG's "more modes than the mesh provides" (PR 14b);
- si_band's capacity error (PR 15c), which says "hi above the spectrum" whenever no upper side
  closed, including on the "no progress" path. On slice paths this costs about $\log_2 S + 1$
  extra runs, each with a span count (documented).

### Strategy ("auto")

1. **Ladder.**
   - Candidates: the top chunk (`band_defaults`' chunk), halved while ≥ 5, always containing
     the top chunk.
   - For each, compute $C_{max}$ (below). A chunk fits if $C_{max} \ge \min$(up(4·chunk + k,
     $c_L$), the free-space cap); 4·chunk is `band_defaults`' initial basis.
   - The choice:
     - with a known depth ($f_1 = 0$ and `n_modes` given): the largest fitting chunk with
       $C_{max} \ge \min$(up(⌈1.25 n_modes⌉ + k, $c_L$), the free-space cap);
     - otherwise, or when no fitting chunk reaches the known depth: the largest fitting chunk
       whose $C_{max}$ ≥ 0.9 × the best $C_{max}$ among the fitting chunks. The 10% threshold
       is justified by benchmark part 2's time per mode against chunk, adjusted if the data say
       so.

     On small meshes $C_{max}$ is capped at the free space, so every chunk reaches it and the top
     chunk wins. Below the top chunk's need, the rule picks a small chunk (5 or 10) in practice:
     the reach gap between chunks is a few hundred columns at every mesh size (documented). The
     searches run on every band call: milliseconds with the cache, about 2.4 s for the AMG
     estimate at 10M.
   - No fitting chunk: "memory". si_band runs on the whole band, and a warning names the levers
     and their $C_{max}$ from the cached estimators.
   - A given `chunk` is used if it fits, else "memory".
2. **LOBPCG** `_solve(..., lam_stop=λ_hi, max_capacity=C_max)`. In band mode `max_capacity` is
   required; solve_modes always passes it.
   - The initial allocation is 14b's `_capacity(n_modes, chunk, k, low)` (0 for a single
     chunk), capped at $C_{max}$. When capped, `n_modes` becomes $C_{max} - k$, so the invariant
     n_modes = capacity − k that the count budget uses holds from the start, as after a growth.
   - Basis growth: `new = min(up(max(⌊1.5 old⌋, count + j + k + 1), c_L), C_max)`, as today plus
     the cap. Every width stays a multiple of $c_L$, which `_project`'s chunked slices need.
   - At a growth, `B`, `args` and every other reference to B are dropped. B is reallocated with
     zeros and refilled with exactly V's first `count` columns.
     - The refill runs in chunks of $c_L$: `device_put(V[:, i:i+c_L].T)` is C-contiguous and
       zero-copy, and the donated `_insert` transposes and casts to `basis_dtype`. A plain
       `.at[].set()` would copy B per chunk.
     - Each chunk is synchronised (`block_until_ready`) and dropped before the next, so queued
       transfers do not pile up.
     - The refilled B equals the old B exactly.
   - The host V is `np.zeros((n, C_max + k), order="F")`, allocated once. Calloc is lazy:
     untouched columns cost no pages. It is never concatenated.
   - The band-end check comes first. A lock needing count + k > $C_{max}$ stops the solve:
     `info["stopped"] = "capacity"`.
   - Reaching $\lambda_{hi}$ ("fits"): PR 14b's count at the cut above $\lambda_{hi}$, then the
     filter (a view). The count's probe-block budget passes `band=(0, f₂)` to
     `band_probe_block`, since `solve_modes_memory` rejects $f_1 > 0$; it only switches on band
     mode there.
   - Non-convergence ends the solve with `converged` False, as in PR 14b; there is no fallback.
3. **Capacity stop** ("capacity").
   - LOBPCG's basis, state, preconditioner, AMG aggregates and V are freed after reading
     $\mu_1^{ub}$ and L. The "memory" path holds no LOBPCG state; the σ pre-estimate's P and
     aggregates are dropped there too.
   - The store reservation becomes $N_s = \max(N_s, \lceil 1.25 L\rceil)$, recorded in `n_store`.
   - A warning gives the locked depth and each lever's $C_{max}$.
   - si_band runs on $[\lambda_{lo}, \lambda_{hi}]$, or from 0 by the rule above.
   - The slice-capacity limits (below) apply now. A capacity below the floor raises only after
     the LOBPCG run (documented).
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
  - The constants per family are recorded in `docs/eigensolvers.md` (§Measured MINRES cost).
  - This replaces the roadmap's "∝ ln(1/τ)/δ plus an outlier term" (15a: depth drives the
    iterations). Slice widths come from the capacity and counting from `verify=None`, not from
    this model.
- **LOBPCG from 0,** measured per chunk to about 1005 modes on hex20 $16^3$/AMG. The chunk
  iterations rise from 18 to 77, and $t_\text{warm}$ from 0.215 to 0.278 s: the projector adds
  +30% at depth 1000. The projector-to-operator ratio is per DOF, so it carries to larger meshes
  at the same depth.
- **Time per mode against chunk size** (chunks 5, 10, 20, 40 to about 200 modes), extrapolated
  to depth $C_{max}$ with the 1005-mode run's $t_\text{warm}$ rise (the projector costs about
  k/chunk × count per mode). This justifies the ladder's 10% threshold at the reach where it
  matters.
- **The crossover table:** per depth D, the smallest N at which a slice beats LOBPCG to D + N.
  "None" is expected. Per mode, a slice costs about 25–110 LOBPCG modes.

### Memory

- **The split.** `lobpcg_memory`, `si_memory` and `count_memory` tag each term by memory space
  and as exact or modelled. They return `device`, `host`, `device_exact` and `host_exact` next
  to today's `total`.
  - `lobpcg_memory(..., capacity=None)`:
    - with `capacity`, the basis is exactly that many columns and V is n_modes + k; without it,
      as today;
    - host = V (exact), plus the AMG build (pyamg runs on the host) and the operator's host
      build;
    - device = the rest: basis, R and stored Ke (exact), and step, start, seed and slack
      blocks, kernel temporaries.
  - `si_memory(op_memory, n_dof, capacity, n_modes, ..., n_deflate, host_window=None)`:
    - `n_modes` now means the host store's columns. The `end` phase keeps only its device
      chunk, since a slice's host V lands in the store: one store term;
    - device storage = operator + preconditioner + $v$·Q + R, M R, with Q = up(n_deflate, c) +
      capacity, as today;
    - host = $8n\,$(host_window, default n_deflate) + $8 Q^2$ (si_slice's G at full width) +
      $8n\,$n_modes, plus the host builds.
  - `count_memory`: operator storage, R, M R exact; the rest modelled; all device (W stays on
    the host in chunks).
- **One helper,** `_largest(fits, step, lo, hi)`: the largest multiple of `step` in [lo, hi]
  with `fits(C)` True, by bisection (each `fits` is monotone), as a Python `int`, or None when
  nothing fits. `probe_block`, $C_{max}$ and the slice capacity all use it. Inside a search,
  `op_memory`/`precond_memory` are wrapped in `functools.cache` (per call; amg_memory takes
  about 2.4 s at 10M).
- **$C_{max}(\text{chunk})$** = `_largest(fits, c_L, c_L, up(n_avail + k, c_L))` (the
  free-space cap), n_avail = free − rigid.
  - fits(C) applies the fit test to `lobpcg_memory(..., n_modes=C, capacity=C, band=True)`.
  - There is no output term, since the filter returns a view.
- **Slice capacity $C$** = min(the budget cap, the need cap).
  - The budget cap: `_largest` over C from up(2c + 2b, c). It applies the fit test to
    `si_memory(op_memory, n, C − W, n_modes=N_s, block, pc, n_rigid, n_deflate=W,
    host_window=2(C − 2b))`.
    - The device basis Q is exactly up(C, c).
    - host_window bounds the host window plus si_checked's [W | V] restart copy. The window
      never exceeds C − 2b columns, including a slice requeued whole.
    - The store reservation:
      - $N_s$ = `n_modes` if given, else 200, the roadmap's assumed band size;
      - after a capacity stop, at least 1.25 L.
  - The need cap: up(1.25 (1.8 $N_s$ + 84) + W, c).
  - si_slice already stops at the free space.
- **Slice counts.** si_band's counts (leaf, span, verification) run with `count_options["block"]`
  unless the user gave one.
  - The block is the largest $b_p$ that passes the fit test with `count_memory(b_p)` (which
    includes the operator, preconditioner, R and M R) plus the store and the host window as
    exact host terms. CPU: one check. GPU: `count_memory` against the device budget, the store
    and window against the host budget. Q is freed when counts run.
  - When one probe does not fit:
    - `verify=True`: a warning, then si_band runs with `verify=False` (no counts, no restarts),
      with the reason recorded;
    - `verify=None`: no counts, silently;
    - `verify=False`: unaffected.
- **Limits,** in order:
  1. `slice_options["capacity"]` overrides the sizing and skips both checks.
  2. Below up(2c + 2b, c), or nothing fits: a `ValueError` naming the budget.
  3. Below 96 columns (PR 15c's slice minimum at b = 6): a warning. This is common on "memory"
     with AMG, since a slice costs about the smallest LOBPCG chunk; the si_memory to LOBPCG
     chunk-5 ratio is recorded in the benchmark.
- **The store** is not checked against memory at run time. si_band gets `n_store` and warns once
  when the store first passes it (on the Mac an overrun swaps rather than failing fast).
- **Probe blocks:** PR 14b's count is unchanged, apart from the `band=(0, f₂)` budget.
- **`solve_modes_memory`:** unchanged; $f_1 > 0$ raises `ValueError` there (its own check,
  since `check_band` now accepts it). A slice estimate is deferred to PR 18.

### `si_band(verify=None)`

`verify` ∈ {True, False, None}. None applies PR 14b's rule per leaf:
- skip above `MAX_DEGREE`;
- skip when the a-priori bound $2\,n_\text{blocks}\,d\,(b_p/b)\,t_\text{minres}$ exceeds the
  leaf's time (0.3 instead of 2 when τ_c is looser than the rule, as `_bound`);
- otherwise calibrate (PR 14b's `_calibrate`, with the leaf's count closure and preconditioner)
  and count iff the prediction ≤ the leaf's time.

The details:
- **Order.** Both skips happen before the pairs are gathered.
- **Times.** A run leaf's time is its slice's; a span's is its parent run's. This reads the
  roadmap's "no more than the solve" per leaf, so a run and its span may each spend up to the
  run's time (logged). $t_\text{minres}$ is the leaf's slice's measured seconds per MINRES
  iteration.
- **`band_count(_steps=)`** is passed through to `_run`, so `_calibrate` works on leaves with
  lo > 0.
- **Location.** `_bound` and `_calibrate` move to `counting.py`. api.py and shift_invert.py call
  them as `counting._bound` / `counting._calibrate`, so patching the module reaches both.
- **Records.** Every leaf records `predicted_s`, `predicted_kind` and `reason` under every
  `verify` value. The roadmap's "the predicted cost is always reported" is the a-priori bound
  when not calibrated.

### Errors

si_band's `_cap_error` is worded per case:
- "lo may be below lambda_1" only when no lower side closed and lo > 0;
- "hi above the spectrum" whenever no upper side closed;
- lo = 0 names only hi;
- "no progress" names tau and, on slices from 0, the λ_lo margin.

## Changes to existing code

- `lobpcg._solve`:
  - band mode requires `max_capacity`;
  - the capped initial allocation with the n_modes invariant;
  - the capped growth rule, by synchronised free-and-refill from V;
  - the count + k stop;
  - V is a preallocated zeroed F-order array;
  - non-band mode is unchanged.
- `si_band`: `verify=None`, `n_store` and its warning, the `_cap_error` wording.
- `counting.py`: `_bound` and `_calibrate` moved from `api.py`; `band_count(_steps=)`.
- `api.py`:
  - `solve_modes`:
    - `band=(f1, f2)`, `strategy`, `memory_budget`, `slice_options`;
    - the ladder;
    - the μ₁ rule and its refinement;
    - the fallback with its frees;
    - the view filter;
    - the `band=(0, f₂)` probe budget;
    - the slice count block;
  - `check_band` for $f_1 > 0$;
  - the NotImplementedError removed.
- `memory.py`:
  - term tags and the per-space entries in `lobpcg_memory`, `si_memory`, `count_memory`;
  - `lobpcg_memory(capacity=)`, `si_memory(host_window=)` and its store meaning of `n_modes`;
  - `_largest`, also used by `probe_block`;
  - the $C_{max}$, capacity and count-block searches, the fit test, the default budget;
  - `solve_modes_memory`'s own $f_1 > 0$ check.
- Existing tests:
  - `tests/test_band.py`:
    - line 151 asserts the view (it asserted a copy);
    - the `_bound`/`_calibrate` monkeypatches (lines 283–384) target `counting`;
  - si_memory callers pass the store columns as `n_modes`.

## Benchmark (`benchmarks/strategy_study.py`, one invocation, ≤ 15 min, progress to a log)

1. **[15a, 15d] constants.** Refit α and $D_0$ per family on `shift_invert_study.json` by log
   least squares (no runs, seconds).
2. **[15d] crossover and ladder.** On hex20 $16^3$/AMG:
   - one LOBPCG band run to about 1005 modes (about 6 min), recording the iterations and
     $t_\text{warm}$ per chunk;
   - from it, the crossover table per depth against the MINRES rows, and the held-out "full"
     slice row predicted against measured (about 1.4×);
   - chunks 5, 10, 20 and 40 to about 200 modes (about 1 min each), recording the time per mode
     against chunk, extrapolated to depth with the first run's $t_\text{warm}$ rise.
3. **[15d] memory record** (estimators only, seconds with the cache).
   - The ladder (chunk 40 → 5: fits, $C_{max}$ under the fit test and the default budget, the
     slice capacity after a capacity stop) × {9-block, low} × {fp64, fp32 basis} × {stored,
     on-the-fly Ke}, plus {fp64, fp32 V-cycle} on voxel. At 10M hex20 on the 76 GB pool, and at
     1M hex20 and 2M voxel on the Mac.
   - The reach lost to the fit test's factors per target.
   - The si_memory to LOBPCG chunk-5 ratio at C = 48–144.
   - fp32 AMG is not modelled until PR 18, so the roadmap's 10M "700–1000" (fp32 AMG) is
     recorded as fp64 plus the levers and marked for PR 18. The 2M Mac "> 2000" is checked
     with the fp32 V-cycle.

Results go to `benchmarks/results/strategy_study.json` with the git commit. Not run: the GMG
depth-500 row and the slice-model refit (logged). The measured peaks are tests 17a–17c.

## Tests (`tests/test_strategy.py`; roadmap tags in the docstrings)

Method:
- **Exact preconditioner.** Real counts use the dense exact $(K + \sigma M)^{-1}$ (15c's
  `exact`), monkeypatched into `api.amg`/`api.gmg` and si_band's `make`. The fake follows the
  builders' contract: `SingularShiftError` at σ = 0 on free meshes, and
  `return_aggregates`/`aggregates` (a small class instance, so it can be weakly referenced).
- **Count cost.** Each count's degree ≤ about 1000 is asserted as a precondition (≤ 15 s per
  count on "short").
- **`verify`.** Every test asserting counts passes `verify=True`, except test 14, which tests
  `verify=None`. Wiring-only tests use `count_options={"degree": 4, "n_probes": 2}`.
- **Budgets and capacities** come from the search functions under the fit test, never
  hard-coded or recorded. A budget "s × the need" means 1.05 × exact + s × modelled at the given
  capacity.
- **Timing.** Each test is timed and marked slow above about 20 s.
- **Accuracy.** Every test mapped to the eigsh item asserts the roadmap bound per mode with that
  mode's t, through test_shift_invert's `check_band` logic generalised.
- **Warnings.** `pytest.warns` matches the warning under test; other warnings (below 96 columns)
  are allowed and listed in the docstring.

1. **"fits"** ("short", hex20 $6\times5\times4$): an interior band of about 10 modes inside chunk
   0, with `n_modes=10` and `verify=True`. Asserts:
   - reason "fits", `from_zero` False;
   - `info["chunk"] == 40` (the band count does not reach `band_defaults`);
   - `count["ran"]` True and the estimate < 0.5;
   - `V` is a view of an F-order array (`V.base is not None`, F-contiguous);
   - matches eigsh.
2. **"capacity"** ("short"), `chunk=5` (k = 10, $c_L$ = 40), `n_modes=22`, band
   `edges(lr, 25, 46)`, `slice_options={"capacity": 96}`. The budget is at the top of the
   $C_{max} = 40$ window (from the search).
   - Preconditions: λ_hi not reached; LOBPCG's locked count equals test 12's stop rule (35
     measured), so L = 10.
   - Asserts:
     - `stopped == "capacity"`;
     - si_band's (lo, hi) = (λ_lo, λ_hi) (spy);
     - at si_band's entry, no live array of LOBPCG's basis shape (`jax.live_arrays()`), and
       weakrefs to LOBPCG's host V and the aggregates are dead;
     - matches eigsh, and the slice counts are < 0.5;
     - `mu1_ub` ≤ LOBPCG's lowest Ritz value and equals the min of the bounds;
     - `n_store == 22` (n_modes binds);
     - the capacity warning names the depth.
   - At the same budget, `strategy="lobpcg"` raises at the cap, naming `memory_budget`.
   - **(b) ⌈1.25 L⌉ binds** (fake): `_solve` monkeypatched to return a capacity stop with 10
     locked modes below λ_lo and 20 in band, `n_modes=5`, si_band spied. Asserts
     `n_store == 25` (a mutant using all locked modes gives 38), and that the need cap and
     si_band's `n_store` received 25.
   - **(c) $f_1 = 0$** (fake): a capacity stop with 30 locked modes on `band=(0, f₂)`,
     `n_modes=5`. Asserts si_band's (0, λ_hi), `from_zero` True, `n_store == 38`.
3. **"memory"** (voxel $10^3$, GMG), `n_modes=10`.
   - Budgets of 1.15× and 1.25× the chunk-5 need at C = 40 give "memory" and LOBPCG (chunk 5,
     $C_{max}$ = 40). This brackets the 1.2 factor exactly.
     - The 1.25× run's band lies inside the first 30 modes, so it ends "fits".
     - Precondition: the searched slice capacity at 1.15× ≥ 72 (the floor).
   - "memory" warns naming the levers.
   - With $\lambda_{lo} < \mu_1$: `from_zero` True, matches eigsh, `info["capacity"]` equals the
     search's result.
   - **The count block** at 1.15×: the block si_band's counts receive equals the searched block
     (spy), with the block < `n_probes` as a precondition. A sentinel block from a monkeypatched
     search reaches the counts.
   - With $\lambda_{lo}$ well above $\mu_1$: si_band gets (λ_lo, λ_hi) (spy), `from_zero` False.
   - "auto" with `band=(0, f)` at the 1.15× budget: "memory", si_band on (0, λ_hi), matches
     eigsh.
   - The ladder, real run: a budget between the chunk-20 and chunk-40 needs gives the chunk the
     reach rule picks from the searched `c_max` table (5 on this mesh).
   - A pure-function table checks the rule:
     - the 0.9 × best reach;
     - the known-depth branch: the largest fitting chunk misses the depth and a smaller one
       reaches it (expects the smaller one);
     - the known-depth fallback when no chunk reaches the depth;
     - a top chunk below 5;
     - a given `chunk` not fitting;
     - a tiny mesh where up(4·chunk + k, $c_L$) exceeds the free-space cap.
4. **"shift_invert"**, with 15c test 2(a)'s configuration (modes 5–28, block 6, its seed) and
   `n_modes` given.
   - The budget comes from the search so that `capacity == 96`.
   - ≥ 2 runs; every leaf counted, < 0.5; the same set as "lobpcg".
   - One real-AMG copy, marked slow.
5. **Empty band:** strictly between two well-separated modes of the voxel "cube" (named gap).
   0 modes and count < 0.5 under both strategies; `from_zero` False.
6. **Clamped box**, `band=(0, f)`, "shift_invert": matches eigsh, `from_zero` True.
7. **$\lambda_{lo} < \mu_1$**, clamped and free, "shift_invert":
   - `from_zero` True, filtered, matches eigsh;
   - the margin: λ_lo = μ₁^ub(1 + 0.9e-2) gives True and (1 + 1.1e-2) gives False (μ₁^ub
     monkeypatched ≥ the true μ₁); tol = 1e-4 widens it to 3e-2;
   - correct with an inflated μ₁^ub;
   - the `sigma=` hint path on voxel $16^3$ with λ_lo ≈ 3μ₁, si_band stubbed by a spy: the
     multigrid refinement runs, `from_zero` False;
   - `sigma=0` on the free hex8: the refinement falls back to `sigma_estimate` (spy);
   - the Jacobi path on the 240-DOF "free" hex8 at λ_lo ≈ 3μ₁: the full `sigma_estimate` ran
     (spy), `from_zero` False;
   - a hint never enters `mu1_ub`.
8. **Filter table** (pure): the exact edge, inside and outside 100t, a chain crossing λ_lo,
   per-mode t, the upper end taken from `n_band`, the result a view.
9. **Lower-edge clusters:** the voxel cube (doubles, triples) with λ_lo on a double and in the
   middle of a triple chain, under both strategies. All copies returned, `from_zero` False.
10. **Edge above the spectrum** (through `solve_modes` on "tiny"): `f2` above the largest
    eigenvalue raises PR 14b's "more modes than the mesh provides" under "lobpcg". Under
    "shift_invert" it raises on the "no progress" path with "hi above the spectrum" in the
    message.
11. **Capacity and $C_{max}$:**
    - **Estimator terms:**
      - one hand-computed device and host term per estimator;
      - `lobpcg_memory(capacity=C)`'s basis exactly C;
      - si_memory's device basis exactly up(C, c)·8n, with `host_window` and the store only in
        the host term, and the Gram term $8Q^2$;
      - the per-space exact parts;
    - **The fit test:**
      - per space on GPU, `total` on CPU;
      - exact × 1.05, modelled × 1.2;
    - **`_largest`:**
      - against brute force, including a tiny mesh where $8Q^2$ and the end phase matter;
      - the jump at cap > 0;
      - the floor rounded up to c (72 at c = 24);
      - None when nothing fits;
    - **Caps:**
      - the need cap and the budget cap each binding;
      - $C_{max}$ capped at the free space, a multiple of $c_L$;
    - **Platforms:** unified and GPU (monkeypatched platform and `memory_stats`): a tight host
      budget where the host V binds lowers $C_{max}$; the AMG build counts in the host check;
    - **Limits:**
      - in order: an override skips both, below the floor or nothing fitting raises naming the
        budget, below 96 warns;
      - an override above the need is kept;
    - **The slice count block:**
      - the fit with the store and window per space;
      - the one-probe case under True, None and False;
    - **`solve_modes_memory`** with $f_1 > 0$ still raises `ValueError`.
12. **`_solve(max_capacity=...)`** directly:
    - the capacity sequence: every width a multiple of $c_L$, growth straight to the cap;
    - a lock with count + k = $C_{max}$ is allowed, count + k = $C_{max}$ + 1 stops;
    - the initial allocation capped, with n_modes = $C_{max} - k$;
    - V F-order zeros of C_max + k columns;
    - after a refill, B's first `count` columns equal V's cast, and the rest are 0;
    - `max_capacity` required in band mode;
    - test_band.py's growth tests (`test_growth_from_zero` and the capacity tests) still pass,
      with `max_capacity` passed through solve_modes.
13. **Cost-model record:** `strategy_study.json`'s constants equal a log least-squares refit of
    `shift_invert_study.json`'s rows (frac 0.5, non-floor, `recheck_1005`), with ratios in
    [0.75, 1.3]. The crossover and per-chunk entries are present.
14. **`si_band(verify=None)`:**
    - with `fake_band`, each branch: MAX_DEGREE skip; bound skip with 2 and with 0.3; calibrated
      and dear, skip; calibrated and cheap, count (`counting._calibrate` spied), on a leaf with
      lo > 0 (band_count's `_steps`). A span leaf's time is its parent's, and the skips come
      before the gather (spy);
    - `predicted_s`/`predicted_kind` present under True, False and None;
    - one real exact-preconditioner run: per-leaf `reason`/`predicted_kind`, `ran` True,
      estimate < 0.5;
    - `solve_modes(strategy="shift_invert")` at the default verify records them;
    - slow, real AMG on one leaf (hex20 $12^3$, as 14b's `test_calibration`): the bound ≤ the
      measured warm count time, and the calibrated prediction within [0.5, 2]×;
    - the store warning (fake): once, when the store passes `n_store`.
15. **Capacity error wording** (`fake_band`): lower side closed, upper side closed, lo = 0, no
    progress on a slice from 0, no progress with no upper side closed.
16. **Options:**
    - `strategy`/`memory_budget`/`slice_options` without `band`, bad values, unknown
      `slice_options` keys, `f1 >= f2`;
    - forced "lobpcg" with no chunk fitting raises up front;
    - the default budgets (monkeypatched `os.sysconf`, a fake `memory_stats` with
      `bytes_in_use`);
    - spies: `slice_options`, `tol`, `count_options` (a user `block` kept) and
      `precond_options` reach si_band and `make`;
    - `precond_dtype="float32"` on voxel through a slice path builds `make` in fp64;
    - a capacity stop with 999 locked modes (monkeypatched) leaves `stagnation_risk` False, and
      1000 sets it;
    - LOBPCG not converged under "auto": no fallback, `converged` False.
17. **Mac peaks** (slow, darwin, fresh subprocess, MallocLargeCache=0, estimates ≥ 2 GB, each
    ≤ about 5 min with a time precondition):
    - **(a)** `_solve(max_capacity=...)` on voxel about $48^3$, `low_memory=True`, chunk 10
      ($c_L$ = 15), `lam_stop` past about 300 modes, so the growths run 60 → 90 → 135 → 210 →
      315.
      - The bound applies to the growth 210 → 315 (old = 14 $c_L$), asserted as a precondition.
      - A spy around that growth resets the interval footprint
        (`proc_reset_footprint_interval`) and reads its max.
      - Asserts: interval max − footprint before < (new − old + old/2)·8n.
        - A correct synchronised refill holds 2–4.4 $c_L$ of transients (measured), against a
          slack of 7 $c_L$.
        - The old+new bug adds old = 14 $c_L$ columns.
        - The bound relies on XLA CPU zeros being lazily paged (docstring).
      - Asserts: (whole-run peak − 1.05 × exact) ≤ 1.2 × modelled, with the estimate at the
        final capacity and locked count (V is lazily paged). The excess per exact column is
        reported and ≤ 0.05.
    - **(b)** `solve_modes(strategy="shift_invert", band=..., memory_budget=B, verify=False,
      n_modes=6)` on lumped voxel $80^3$, modes about 4–8, the capacity from the budget.
      Asserts:
      - the peak ≤ B and ≤ the worst-case composition;
      - within 20% of the composition evaluated with the run's maximum window and stored count;
      - both reported.
    - **(c)** `band_count` on lumped voxel $80^3$ at the budgeted block, with
      `count_options={"degree": 8}` and the store and window held as host arrays. The peak is
      ≤ the fit-test composition and within 20% of it (the peak does not depend on degree).

The README example (`band=(f1, f2)` on the "short" box mesh, about 10 modes) runs in
test_readme.py with every printed value asserted.

`examples/m3b_bands.py` runs in `tests/test_examples.py`, on the "short" mesh:
- `band=(0, f)` and `band=(f1, f2)` under "auto";
- one forced `strategy="shift_invert"`;
- each ends in a check against eigsh.

## Docs

- `docs/eigensolvers.md`:
  - **§Public API:**
    - the signature line with `strategy`, `memory_budget`, `slice_options`;
    - step 1's band checks;
    - step 5's strategy dispatch;
    - the Returns bullet with the new `info` fields.
  - **A new section, "Interior bands and strategy":**
    - the API and the μ₁ check;
    - the ladder, and the memory-driven rule with the measured evidence per D and N;
    - memory and capacity:
      - the fit test and the macOS malloc-cache headroom;
      - small "memory" capacities, the store reservation and its warning;
      - the slice count block and its one-probe case;
      - the resident columns behind the returned view;
    - `verify=None` for slices;
    - errors, including late edge errors above the spectrum and their cost;
    - the stagnation warning;
    - the margin's recovery path and its "no progress" end;
    - δ_max on slices from 0.
  - **§Measured MINRES cost:** the $\alpha D(1 + D/D_0)$ constants per family.
  - **§Frequency band [0, f_max],** rewritten:
    - the Capacity bullet: capped growth, free-and-refill, preallocated V;
    - the Returns bullet and the copy sentence near line 829: a view, not a compact copy.
  - **"Edge above the spectrum"** (near line 664): late errors only, with their cost on slice
    paths.
  - **The estimator reference** (near lines 122, 348, 358 and 674):
    - `lobpcg_memory(capacity=)`, `si_memory(host_window=)` and its store `n_modes`;
    - the per-space entries;
    - the 0.15 × 8n × n_modes growth note re-measured;
    - `probe_block`/`band_probe_block` through `_largest`.
  - **The Options block:**
    - new rows for `strategy`, `memory_budget` and `slice_options` (block default 6);
    - rewritten rows:
      - `band`: the "256 fp64 columns up front" replaced;
      - `n_modes`, `verify`, `precond_dtype`;
      - `chunk` with `band` (near line 92): chunk 40 for $f_1 > 0$; the ladder replaces "pass
        `chunk` above 2M".
  - **The Validation list.**
  - **Every mention** of `15d`, `PR 15d` and `NotImplementedError`, and "the roadmap's
    crossover", in `docs/` and `README.md` rewritten (eigensolvers.md near lines 103, 122,
    589–720).
- `README.md`:
  - the `band=(f1, f2)` example;
  - the options paragraph near line 222;
  - the band paragraph near line 247 (interior bands, the strategy, the m3b example);
  - the status, naming M3b;
  - the closing "See docs/eigensolvers.md" list gains "interior bands and strategy".
- `docs/ROADMAP.md`:
  - **Status:**
    - the Status paragraph; PR 15d status; PR 15 complete and milestone M3b done (with
      `examples/m3b_bands.py`);
    - the dependency table (PR 18 gains 15d);
    - the PR 14 contents note "PR 15 adds the shift-invert branch" as done;
    - PR 15 Contents (line 545, "Slicing with the cost model … 15d: the cost model and
      strategy") and PR 15 Docs (line 550, "15d the memory model and calibration") as built.
  - **The δ_max ≤ 1 sentence** (Convergence/cluster bullet, line 80): what holds (per-pair
    thresholds; slices from 0 can exceed tol; `r_acc` reported; enforcement → PR 18).
  - **Frequency bands:**
    - the "Strategy: chosen by a cost model" header (line 88) and its bullets: memory-driven;
      the ladder; "≲ 1200" replaced by the measured crossover; the 10M and 2M Mac figures from
      the record (fp32 AMG for PR 18);
    - line 91, "`None` lets the cost model enable it": now PR 14b's rule per leaf;
    - line 93, "must expect stagnation": the warning and `stagnation_risk`, remedies → PR 18;
    - the Band edges bullet (line 95): "the cost model's cap" → `MAX_DEGREE`; the up-front
      bound deferred, late errors only;
    - line 99, the $f_1 < \lambda_1$ sentence: the margin, the LOBPCG Ritz bound and the
      refinement;
    - line 100, "Slice widths come from the cost model … counting degree": capacity-limited
      widths and the per-leaf rule;
    - a 15d note on $\alpha D(1 + D/D_0)$.
  - **Memory budget:** the "Max locked modes" row from the record under the fit test; a
    shift-invert row; the fit test; the Mac np.memmap bullet → PR 18.
  - **PR 18 contents:** the "band follow-ups (features)" and performance items above, including
    the depth-1000 remedies and the up-front spectrum bound.
  - **Deferred GPU checks:** the items in Out of scope.
- `docs/open_items.md`:
  - **Closed:**
    - 14b's "no strategy selection or basis-fit check";
    - 15a "Cost driver … 15d fits its cost model on depth";
    - 15a "`capacity` is required; 15d sets it";
    - 15c "τ_c kept … 15d's cost model prices counts" (settled by the per-leaf calibration);
    - 15c "`sigma` and `count_precond` are rejected … 15d exposes `make`'s options" (through
      `precond_options`);
    - the 15c → 15d items done here: capacity from the budget, the `verify` callable (now
      `verify=None`), "15d words it per case".
  - **Superseded:** 14b's "Basis growth … no `max_modes` cap" and "Host V holds n_modes + chunk
    columns"; 14b's "Band chunk default 40" annotated with the $f_1 > 0$ rule.
  - **Partly addressed:**
    - "`edge_unreached` is not evidence…" (line 147): the μ₁ rule done, the λ_hi bound → PR 18;
    - "A top remainder holding only known modes can raise" (line 169): the message fixed by the
      per-case wording, the cost → PR 18 with "halving before a capacity error";
    - "size the capacity with the window's chunk included" (the window reservation);
    - "span counts at small capacities" (the 96-column warning);
    - "a capacity error discards the band" (the window reservation; `err.result` → PR 18).
  - **Retagged to PR 18,** headings changed from "→ PR 15d" with a one-line 15d disposition per
    bullet:
    - "Late capacity errors and slice efficiency": "requeue whole", "stop si_checked before its
      count once r_acc widens the window", "halving before a capacity error";
    - the 15c decision "a done side gives up its half's gaps" (line 141);
    - the 15b "σ = $c_{hi}$" item in "Band-pass count and slice restarts" (its own line);
    - "the `verify` skips before the gather" for `verify=False` (None skips before it; False
      still predicts after it);
    - "Span counts at small capacities";
    - "Shift-invert cost near mode 1000" and "σ-move retries": warned, remedies to PR 18;
    - verification slices, per-step residuals, merge and count copies, host transposes.
  - **New:** the macOS malloc-cache headroom (measured up to about 10 GB, held by the 0.2 × RAM
    headroom; not visible to the MallocLargeCache=0 tests).
  - The decisions below, and the items not adopted.

## Decisions without user input (to log)

- **Memory-driven strategy, no time-based crossover.** LOBPCG to D + N is cheaper than any slice
  at every measured D and N (hex20 $16^3$ to depth 1005). This deviates from the roadmap's "cost
  model chooses" and "≲ 1200" (corrected). The roadmap's "slice widths and counting degree in
  the cost model" are replaced by capacity-limited widths and the per-leaf `verify=None`. A
  later crossover finding → PR 18. The GPU ratio is a deferred check.
- **Capacity stop:** si_band on the whole band, with a warning (depth, levers). The discarded
  LOBPCG run is bounded by its time to $C_{max}$; a slice capacity below the floor raises only
  after it. The hybrid junction → PR 18.
- **The ladder:**
  - from the top chunk down to 5;
  - with a known depth (f₁ = 0 with `n_modes`), the fastest chunk reaching it; otherwise, or if
    none reaches it, the fastest within 10% of the best reach (threshold from benchmark part 2);
  - a given `chunk` replaces the ladder or gives "memory";
  - no automatic levers → PR 18.
- **Forced "lobpcg"** is capped by the budget and raises up front or at the cap.
- **Default budgets:**
  - the GPU free pool;
  - 0.8 × RAM on CPU, not reduced by the caller's holdings; the rest holds the macOS malloc
    cache;
  - the GPU host budget 0.8 × RAM.
- **The fit test:** 1.05 on the exact column arrays, 1.2 on the modelled terms, per memory
  space. 1.2 on every term would cost 26–32% of reach at 10M; the per-target loss is in the
  benchmark record. The 1.05 is checked by 17a.
- **The LOBPCG growth** keeps the rounding to $c_L$ and rebuilds B from the zeroed host V by
  zero-copy chunks, synchronised (PCIe about 3 s for 1000 columns at 10M), instead of copying.
  Allocating B at $C_{max}$ up front on GPU, against fragmentation, is left to the GPU checks.
- **The initial allocation** is 14b's, capped, with n_modes = capacity − k.
- **Returned modes are a view** of the solver's F-order output for both strategies; the
  columns below the band stay resident. This supersedes 14b's compact copy.
- **No up-front spectrum bound:** the late errors meet the roadmap criterion, at about
  $\log_2 S + 1$ extra runs on slice paths; the bound → PR 18.
- **No run-time store check:** the $N_s$ reservation sizes the store; a one-time warning when
  it passes $N_s$; an overrun is an ordinary OOM.
- **Slice counts** run at a budgeted probe block on slice paths. When one probe does not fit,
  `verify=True` warns and runs without counts.
- **Stagnation:**
  - warned only after a capacity stop at ≥ 1000 locked modes; the depth is unknown elsewhere;
  - remedies → PR 18;
  - the 10M criteria near index 1000 may need the user's input after the GPU checks.
- **Slice store and need:** `n_modes`, else 200, else after a capacity stop ≥ 1.25 L. There is
  never a Weyl estimate: anchored at μ₁ it is off by orders of magnitude (round-4 review).
  Below 96 columns, a warning.
- **The slice host window** is reserved as 2(C − 2b) columns (window and restart copy), and the
  device window as 2c.
- **The $\lambda_{lo} < \mu_1$ rule:**
  - the margin is max(1e-2, 300 tol);
  - μ₁^ub is the min of the available bounds, computed directly when none exists (with the
    `sigma_estimate` fallback at a singular hint);
  - a `sigma=` hint never enters μ₁^ub.
- **`verify=None` per leaf:** each leaf's count is compared with its own slice's time (a span
  with its parent run's).
- **δ_max ≤ 1 is not enforced** on slices from 0. Per-pair thresholds hold, and r_acc may
  exceed tol (reported); enforcement → PR 18.
- **`n_modes`** for an interior band is the expected band count; for $f_1 = 0$ it is a depth.
- **`slice_options`** exposes `block` (default 6) and `edges` (internal in 15c), following the
  "every option reachable" rule.
- **LOBPCG non-convergence** under "auto" gets no slice fallback (as PR 14b).
- **`solve_modes_memory`** is unchanged, and $f_1 > 0$ there → PR 18. The interior count's probe
  budget passes `band=(0, f₂)`.
- **`err.result`** → PR 18; the error messages are worded per case now.
- **The slice cost form** is $\alpha D(1 + D/D_0)$; τ scaling is not modelled.
- **Not run:** the GMG depth-500 row and the slice-model refit.
- **np.memmap** → PR 18.
- **`precond_dtype="float32"`** applies to LOBPCG's GMG only; slices are always fp64.
- **Test tolerances:** slice tests run at tol 1e-6 with τ = 1e-8 (as in 15c), instead of the
  roadmap's r_acc = max(1e-10, …) test setting, for run time. The eigsh bound is asserted per
  mode.
- **Examples:** M3b gets `examples/m3b_bands.py`. M3 (PR 13b) had none; it is logged and not
  added retroactively.
