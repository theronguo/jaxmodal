# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands), the Memory budget section and
the PR 15c plan's split (15d = the cost model, the memory budget, the strategy choice,
`verify=None` for bands and `solve_modes(..., band=(f1, f2))`).

Revision 16. Rounds 1–8:
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
- the capped initial allocation;
- the per-space fit test.

Round 9:
- **LOBPCG's λ is sorted** (with V's columns) before the filter when chunks returned out of
  order.
- **The growth** drops the LOBPCG state before the refill and uses its own `_refill`.
- **The slice searches** use fp64 preconditioner models.
- **GPU host** is a max of the builds and V.
- **17a is split** into a direct `_grow` peak and a short real run with a slope check; the
  paging claim is corrected (XLA zeros eager, np.zeros lazy).
- **17b and 17c** are shortened to one slice and one probe block.
- **The benchmark** shares one AMG-estimate cache and drops the redundant chunk-40 run.
- **Test fixes:** μ₁ refinement checked against eigsh, both sides of the tol margin,
  test_band.py's `test_errors` row, pinned `slice_options` keys.
- **Docs:** the remaining passages.

Round 10:
- **The sort moves** to right after `_solve` returns, before `_verify`, μ₁^ub, L and the
  filter. One permutation covers λ, r, t and V's columns. V is permuted in place, so the
  result stays a view, and the cut is recomputed from the sorted values.
- **`_grow(V, count, new, c_L, dtype)`** no longer receives the old B. The caller drops every
  reference, and `_grow` waits for XLA CPU's asynchronous free before allocating.
- **Estimator keys** are `device_total`, `host_total`, `device_exact` and `host_exact`. The
  existing `host` component and `total` stay.
- **Defined:** L and the unlocked modes at a stop; the count-time AMG rebuild in the host max;
  the preconditioner in the device list; the cached AMG estimate.
- **Tests:** 17a1 transfers ownership and repeats the growth; 17a2 measures each growth's jump
  instead of a slope; 17b is pinned to depth about 6; tests 8 and 12 gained rows.
- **Docs NITs.**

Round 11:
- **The growth's free** is made synchronous: the caller calls `block_until_ready` on B's last
  readers before the drop. The planned footprint poll had no baseline and would time out under
  the default macOS allocator, so there is no wait at all.
- **The refill** reuses `_insert`; there is no `_refill`.
- **At a capacity stop,** the stopping chunk is removed from λ, r, t and V. The sort skips the
  cut recomputation there, so the fallback still runs. Otherwise, a sorted cut at the end sets
  `stopped`, not `reason`, and skips `_verify`.
- **Band mode returns the view `V[:, :end]`.**
- **`_amg_scan(mesh)`** is cached with `lru_cache(maxsize=1)`, keyed on the mesh only. There
  are no other estimator caches, and the host max has no aggregates term.
- **`verify=None`:**
  - decided once per leaf;
  - the falsy-verify sites made three-way;
  - a warm $t_\text{minres}$.
- **Tests:**
  - 17a1's order is pinned, and it runs the kept-alive mutant;
  - 17a2's jump bound is new·8n, plus an entry check that the free happened;
  - rows were added to tests 8 and 12;
  - the docs gaps (preconditioners.md, the Errors bullet, open_items line 139) are fixed.

Round 12:
- **Memory and search:**
  - the operator's whole `storage` is exact in all estimators;
  - the estimators are cached per call (`functools.cache` on `_models`' results), about 11
    distinct widths per band call;
  - the AMG count-time rebuild is measured with aggs held, with a fitted term as the
    fallback.
- **Refill:** in whole chunks through `_insert(B, jnp.asarray(...))`, with no recompile.
  `_precond_sigma(rounds=1)` gives μ₁^ub; there is one `_fits` helper; the ladder's fit goes
  through `_largest`'s lo.
- **The capacity stop** sits before the V write. The lower-end rule is defined for per-mode t.
- **`verify=None`:**
  - a warm $t_\text{minres}$ (`minres_first_s`);
  - `counting._degree`;
  - the decision carried over si_band's reruns.
- **Tests:**
  - 17a1 asserts on the minimum over runs, and its untestable mutant is dropped;
  - test 12's boundary row is reachable;
  - test 6 checks the view.
- **Docs:** si_checked, the "Edge above the spectrum" bullet, and smaller passages.

Round 13:
- **Returned modes** are a view only when the band starts at index 0. Otherwise they are an
  F-order copy taken after the solver state is freed, so modes below the band are not kept
  resident.
- **Named searches** (`c_max`, `slice_capacity`, `count_block`, `default_budget`, `_ladder`),
  fed by the per-call cached estimators. The build phases are assigned per space.
- **The count-degree helper's signature** is pinned.
- **The capacity stop** works through `end = count`.
- **The X0/state drop** has one code path.
- **Tests:**
  - warm-$t_\text{minres}$ rows;
  - golden degrees;
  - the non-monotone t row;
  - the fp32 refill;
  - a lower bound on 17a2's peak;
  - the 17c degree pair;
  - an estimator-cache spy.
- **Docs gaps:** `minres_first_s`, ROADMAP line 16 and 544, the preconditioners.md `build`
  bullet.

Round 14 (with the late round-13 correctness report):
- **The returned modes** are the solver's own V, compacted in place and resized. This needs no
  extra memory on any platform, and a copy is the fallback.
- **The lower-end rule** handles an empty kept set; i0 = 0 for $f_1 = 0$; no μ₁ refinement
  for $f_1 = 0$.
- **`_slice_degree`** is slice-only and mirrors `_count_slice`. `count`/`band_count` are
  unchanged, and the neighbour lookup is split into a λ pass and the column gather.
- **Reruns** carry the decision through `count["decision"]` / `_decision`.
- **Searches:**
  - internal, keyword-only;
  - `_c_max` computes its own lo;
  - `_ladder(reach, need)` with a per-chunk need;
  - `_count_block` takes `voxel`;
  - `pc64` is reused when fp64.
- **The redundant `block_until_ready`** is dropped, and the invariant is kept as a comment.
- **Tests:**
  - run leaves are faked at the si_slice level;
  - weakref ownership rows;
  - the empty-band row;
  - exact cache counts;
  - 17a2's `lam_stop` and environment;
  - 17c at degree 2 vs 8.
- **Docs:** the Returns fields, the probe budget's independence from `memory_budget`, and ROADMAP
  line 107 (it is eigensolvers.md line 107, the Capacity item; see round 15).

Round 15:
- **The resize** runs inline, in the one frame holding the only reference (`_solve` for its
  resize to `end`, `solve_modes` for the filter). numpy refuses it under any weakref or second
  name, so a helper taking V could never resize. The copy fallback now warns.
- **Tests** track identity by `id()`, never by a weakref or a stored array. Spies record ids,
  shapes and flags only. An extra-reference row checks the warning.
- **Stale text** from round 14 (`block_until_ready`, the band-mode view) is rewritten.
- **Simplifications:**
  - `_decision` is dropped, and a rerun re-decides;
  - `_count_nbs`/`_beyond` are unchanged;
  - `need` has no `up()`;
  - `_count_block` uses `_largest`;
  - `info["budget"]` is a pair;
  - `pc64 = pc` unless GMG is fp32.
- **`verify=None` inputs** are pinned: si_slice's new `time["total"]`, and the span's parent
  time and $t_\text{minres}$.
- **Tests:**
  - the cache-spy scenario;
  - the lever $C_{max}$ evaluation;
  - test_shift_invert.py:1826 (`verify=None` is now valid);
  - test_band.py's count spy, which held W.
- **Docs:** preconditioners.md line 479, eigensolvers.md lines 103 and 107, and the refill rates.

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
  - `lobpcg_memory`, `si_memory` and `count_memory` gain per-space `device_total`,
    `host_total`, `device_exact` and `host_exact` entries;
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
  - L: the locked modes the filter's lower-end rule keeps at a capacity stop (λ ≥
    λ_lo(1 − 100t) with chain extension; all locked for $f_1 = 0$). The stopping chunk's
    unlocked Ritz values are removed from λ (below), so they count towards neither L nor
    $\mu_1^{ub}$;
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
  - the growth refill's host transient (modelled at 1–2 host chunks);
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
  - **The fit test:** the estimate's exact part E and modelled part M pass when 1.05 E + 1.2 M ≤
    budget.
    - E is the column arrays: basis, V, the slice window and store, R and M R, the operator's `storage`. Its
      1.05 covers allocator growth proportional to the columns. Test 17a2 measures it as each
      growth's jump under the preallocated V. The 0.15 × 8n × n_modes in eigensolvers.md was
      measured on a plain chunked solve.
    - M is everything else: step, workspace, kernel temporaries, builds, preconditioners, the
      Gram matrix. These run up to 9–16% low (si_memory 1.09–1.11×, GMG LOBPCG 1.12×, a
      one-chunk zero-workspace solve 1.16×), and 1.2 is the roadmap's ±20% validation band.
    - Unified memory (CPU): one check, E = `device_exact + host_exact` and M = `total` − E.
      Exact terms are persistent, so they are inside every phase of `total`.
    - GPU: `device_total` with E = `device_exact` against the budget, and `host_total` with
      E = `host_exact` against the host budget.
  - Recorded in `info["budget"]` as the pair (budget, host_budget), host_budget None on CPU.
- `slice_options`: dict with keys `tau`, `maxit_inner`, `block` (default 6, the roadmap's b),
  `max_slices`, `capacity` (overrides the sizing), `edges` and `verbose`. They are validated as
  si_band does, and pinned in test_options.py against si_band's and si_slice's signatures.
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
    - **Order.** si_band's set is ascending. LOBPCG's `np.concatenate(lams)` is not
      guaranteed to be: a mode an earlier chunk missed can come back in a later chunk below
      that chunk's top.
      - The sort happens right after `_solve` returns. Everything that reads λ by index comes
        later: `_verify` (`lam[n_band - 1]`, `lam[n_band]`), $\mu_1^{ub}$, L and the filter.
      - If `np.diff(lam) < 0` anywhere, one stable `argsort` permutation is applied to λ, the
        per-mode r and t, and V's first `len(lam)` columns (the final chunk's included).
      - V is permuted in place, following the permutation's cycles with one n-vector buffer.
        Only moved columns are touched, and V stays the same owning array.
        This needs no copy (`V[:, order]` would allocate one; `np.take` would also lose the
        F-order).
      - At a capacity stop, λ holds only locked modes and no cut is used, so the sort runs but
        the cut and the end rule below are skipped; the fallback proceeds.
      - Otherwise, after a sort, the cut is recomputed with `lobpcg._band_count(lam, t, λ_hi)` on the
        sorted arrays, and `n_band` is replaced. If the cut reaches `len(lam)`, there is no
        mode above the cut to bound it. The solve is then reported as `converged` False with
        `info["stopped"] = "sorted cut at the end"`. `_verify` is not called, since it reads
        `lam[n_band]`; `count` gets `ran` False with that reason.
    - **The upper end** comes from LOBPCG's `n_band` (after the sort) or si_band's set.
    - **The lower end.** t is LOBPCG's per-mode threshold or r_acc.
      - For $f_1 = 0$, i0 = 0 explicitly, so a near-zero negative λ is kept.
      - Otherwise i0 is the smallest index with λ_i ≥ λ_lo(1 − 100 t_i), or len(λ) when none
        passes. With per-mode t the mask need not be monotone, so the smallest index is used.
      - Then extend down while 0 < i0 < n_band and λ[i0] − λ[i0−1] < 100 t[i0−1] λ[i0−1],
        with per-mode t as in `lobpcg._band_count`. With a scalar t this is si_band's
        `_extend_chain`.
      - [min(i0, n_band), n_band) is kept. It can be empty, for example when shift-invert from
        0 returns only μ₁ below the band.
      - The upper cut keeps `_band_count`'s existing `.sum()` mask. With per-mode t it can drop
        an in-tolerance mode inside the 100t window. This is pre-existing and logged for
        PR 18.
    - **Compacted in place.** The kept set of ascending λ is one range [i0, i1). Both
      strategies compact the solver's owning F-order V (LOBPCG's, or si_band's V_f) in place:
      - for j in range(i1 − i0): `V[:, j] = V[:, i0 + j]`. Ascending order is safe, since the
        destination never runs ahead of the source;
      - then `V.resize((n, i1 − i0))` (numpy's in-place realloc, `refcheck=True`). The result
        is the same owning, F-contiguous array, holding exactly the band's columns.
      - The result needs no extra memory on any platform, strategy or basis dtype (measured
        by round 14: 3.2 → 0.82 GB at 1M × 400 → 100 columns, in 0.02–0.07 s). A view would
        keep the i0 modes below the band resident, about 16 GB for a 1.6 GB result at 2M DOF
        with band modes 900–1000. A copy would need V + copy on the host, which is unbudgeted
        on GPU.
      - **Inline, in the owning frame.** numpy's `refcheck` refuses the resize under any other
        name, helper parameter, container or weakref (measured in round 15). So the
        compaction loop and `V.resize` are written inline:
        - in `solve_modes`, which tuple-unpacks the solver's result (`lam, V, info =
          lobpcg._solve(...)` / `si_band(...)`), so V is its only name;
        - and in `_solve`, for its own resize to `end`.

        A pure helper computes only (i0, i1). The loop is skipped when i0 = 0.
      - **Fallback.** If `resize` raises `ValueError`, the result is
        `np.array(V[:, :i1 − i0], order="F")` and a `RuntimeWarning` says "returned modes
        copied: another reference to V exists". `_solve`'s resize uses the same fallback. The
        fallback runs after the in-place compaction, so another holder would see shifted
        columns (documented; no code path keeps one, and tests assert the warning is absent).
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
- **Refinement.** For $f_1 = 0$ the rule always fires, so no refinement runs. When neither
  bound exists ($f_1 > 0$ with a `sigma=` hint or Jacobi), one is computed directly:
  - multigrid: `_precond_sigma(op, free, R, make, rounds=1)` itself. It builds at 0, falls
    back to `sigma_estimate` on `SingularShiftError` (a free body), and runs one 1-mode
    LOBPCG, which returns an upper bound (measured within about 5e-6 relative of μ₁). The hint
    is not used for the bound. P is dropped before si_band's `make(σ)`;
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
   - Candidates: the top chunk (`band_defaults`' chunk), halved with `//` while ≥ 5, always
     containing the top chunk.
   - For each, compute $C_{max}$ with `_c_max` (below). `_largest`'s lo = min(up(4·chunk +
     k, $c_L$), the free-space cap); 4·chunk is `band_defaults`' initial basis. None means the
     chunk does not fit, and `info["c_max"]` records None for it.
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
     searches run on every band call. Each bisection is 14–20 `fits` calls (log₂ of the range),
     about 70–90 per band call. Inside solve_modes, the estimators from `_models` are wrapped
     in `functools.cache`, so only the distinct widths cost anything: about 4 ladder k's, block
     6 and about 6 count blocks, so about 11. `amg_memory`'s
     mesh scan (corner nodes, the parent/child tables) moves into `_amg_scan(mesh)` with
     `functools.lru_cache(maxsize=1)`, returning (family, n_c, the p-level table bytes).
     - It is keyed on the mesh only, since `max_coarse` enters only the arithmetic after the
       scan. `Mesh` hashes by identity, and `maxsize=1` keeps at most one mesh alive. That
       mesh stays alive after `solve_modes` returns (about 0.2 GB at 10M hex20; documented).
     - A search therefore costs one scan, 2.3 s at 10.2M hex20 (measured), and milliseconds
       after it. The other estimators take 4–22 ms at 1–2M (measured) and are cached only
       per call (above).
     - The exception is `operator_options={"path": "gather"}`: its `layout` inverse map
       costs 3.2 s per `unstructured_memory` call at 10.2M. A band call then spends about
       35 s on estimates (about 11 distinct widths × 3.2 s with the per-call cache), plus
       the levers' warning. This is documented; `layout` is not cached separately, and the
       gather path is not the default.
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
   - **At a growth,** `B = _grow(V, count, new, c_L, dtype)`. `_grow` never receives the old
     B.
     - **The caller frees first.** The line `X0, state, args = state[0][:, j:], None, None`
       moves unconditionally to just after the V write, before the growth test; today X0 is
       taken after `_lock` (lobpcg.py:558). One code path, growth or not. On a growth the
       caller keeps `dtype = B.dtype` and sets `B = None`, dropping every other reference to the old B. This
       holds for the 9-block step too, so only X0 and X are alive during the refill.
     - **A synchronous free.** XLA CPU frees a dropped buffer only once every queued
       computation that reads it has finished (round 11: about 62 ms late with a step in
       flight, about 1 µs with nothing queued). At the growth nothing reading B is queued:
       `float(g)` and `np.asarray(rl[:j])` already wait for `_reortho`, B's only reader. A
       comment states this invariant; there is no extra call. `_grow` has no wait, no sleep
       and no footprint poll.
       - Under the default macOS allocator the old B stays in the malloc cache. It is not
         reused, since the new B is 1.5× larger, so for B below about 8 GB old and new are both in the footprint; larger
         freed blocks are unmapped (round 13). The
         cache plateaus at about 8 GB on the 64 GB Mac (round 12), inside the 0.2 × RAM
         headroom (documented).
       - The GPU BFC allocator's behaviour is a deferred GPU check.
     - **Refill.** B is reallocated with zeros (XLA CPU writes them eagerly) and refilled from
       V in whole chunks of $c_L$ up to up(count, $c_L$), by the existing donated
       `_insert(B, jnp.asarray(V[:, i:i+c]), jnp.int32(i))`. There is no separate `_refill`.
       - i + $c_L$ ≤ new always holds (new is a multiple of $c_L$ ≥ up(count, $c_L$)), so
         `dynamic_update_slice` never clamps.
       - The padding columns [count, count + j) hold the X that `_lock` rewrites, since V is
         written before the growth. The columns after them are V's zeros.
       - With one chunk shape, `_insert` compiles once per new B width, as `_lock` already
         does.
       - The transient is 2 chunks measured with fp64 on CPU (the host F→C copy and XLA's
         buffer), 1 with fp32; on GPU the host chunk and its device copy, plus half
         a chunk for an fp32 basis's `astype`: up to 2.5 chunks, about 13 GB at 10M with
         $c_L$ = 64. That fits well under the dropped 9-block step, about 61 GB at k = 48.
     - **Synchronised.** Each chunk is synchronised (`block_until_ready`) and dropped before the
       next, so queued transfers do not pile up.
     - The host-side refill transient (2 chunks on CPU) is measured on CPU only; on GPU it is
       modelled (one host chunk, up to 2 with the F→C copy) and listed in the Deferred GPU
       checks.
     - After the lock, B equals the old B with the lock applied, exactly.
   - The host V is `np.zeros((n, C_max + k), order="F")`, allocated once. Its calloc is lazy:
     untouched columns cost no pages. It is never concatenated. The k extra columns are
     needed: after the last lock count ≤ $C_{max}$, and a band-end final chunk writes up to
     count + chunk.
   - **Band mode returns its owning V,** resized in place to `end` columns (`V.resize`, inline
     in `_solve`, with the warning fallback above).
     Today's `np.ascontiguousarray(V[:, :end])` (lobpcg.py:563) is removed, since it would
     make a C-order copy.
   - **At a capacity stop** (checked right after the band-end break, lobpcg.py:546–548, and
     before line 549's V write):
     - `chunks[-1]["locked"] = 0`;
     - `jj = i = 0`, so `end = count`. The existing `np.concatenate(...)[:end]` slices drop
       the stopping chunk's λ, r and t, and the final chunk is not written to V;
     - `n_band` is `count`, and the caller does not use it;
     - `converged` stays True.
   - The band-end check comes first. A lock needing count + k > $C_{max}$ stops the solve:
     `info["stopped"] = "capacity"`.
   - Reaching $\lambda_{hi}$ ("fits"): PR 14b's count at the cut above $\lambda_{hi}$, then the
     filter. The count's probe-block budget passes `band=(0, f₂)` to `band_probe_block`, since
     `solve_modes_memory` rejects $f_1 > 0$; it only switches on band mode there.
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
- **Time per mode against chunk size** (chunks 5, 10, 20 to about 200 modes; chunk 40 from the
  first 200 modes of the 1005-mode run), extrapolated to depth $C_{max}$ with that run's
  $t_\text{warm}$ rise (the projector costs about k/chunk × count per mode). This justifies the
  ladder's 10% threshold at the reach where it matters.
- **The crossover table:** per depth D, the smallest N at which a slice beats LOBPCG to D + N.
  "None" is expected. Per mode, a slice costs about 25–110 LOBPCG modes.

### Memory

- **The split.** `lobpcg_memory`, `si_memory` and `count_memory` tag each term by memory space
  and as exact or modelled. They return `device_total`, `host_total`, `device_exact` and
  `host_exact` next to today's entries. The existing `host` component (`lobpcg_memory`'s
  transients and fp64 modes) and `total` keep their names and values.
  - `lobpcg_memory(..., capacity=None)`:
    - with `capacity`, the basis is exactly that many columns and V is n_modes + k; without it,
      as today;
    - `host_total` = max(operator host build, V + the AMG build, V + 8n·$c_L$ (the refill's
      host chunk, exact)), where V is exact.
      - pyamg runs on the host before `_solve` allocates V. `_verify`'s `build(lam_c)`
        rebuilds the hierarchy while V is held.
      - `_verify`'s rebuild is `make(s, aggregates=aggs)` and runs with the first build's
        aggs held. `amg_memory` has no aggregates term; its `build` is taken to bound
        `_nbytes(aggs)` + the rebuild's peak. Test 11 measures this on hex20 $16^3$ and
        $32^3$ and hex8 $32^3$ (`harness.measure`). If it fails at implementation, an
        `aggregates` term fitted on those meshes is added to `amg_memory` and to the host
        max;
    - `host` stays a component; `host_total` is the host-space peak (one docstring line);
    - `device_total` = basis, R, M R and the operator's whole `storage` (exact: layout
      arithmetic, not a fit), plus the preconditioner's storage, the
      overhead, the step, start, seed and slack blocks, and the kernel temporaries (all
      modelled).
  - `si_memory(op_memory, n_dof, capacity, n_modes, ..., n_deflate, host_window=None)`:
    - `n_modes` now means the host store's columns. The `end` phase keeps only its device
      chunk, since a slice's host V lands in the store: one store term;
    - device storage = operator + preconditioner + $v$·Q + R, M R, with Q = up(n_deflate, c) +
      capacity, as today;
    - `host` = $8n\,$(host_window, default n_deflate) + $8n\,$n_modes + $8 Q^2$ (an upper
      bound of si_slice's G, modelled), and `end` = $8n\,(2 + c)$, so `total` (CPU) and
      `host_total` (GPU) carry the same store and window terms;
    - `host_total` = `host` + max(the operator's host build, the preconditioner's host
      build). The builds run one at a time, while the window and store are held.
  - `count_memory`: operator storage, R, M R exact; the rest modelled; all device (W stays on
    the host in chunks).
- **One fit helper,** `_fits(est, budget, host_budget, host_extra=0)`: the fit test on one
  estimate, with `host_extra` exact host bytes added (the count block's store and window).
  `host_budget=None` means unified memory (CPU: one check on `total`). $C_{max}$, the slice
  capacity and the count block all use it.
- **Build phases by space:** the operator build and the AMG build live on the host. The GMG
  build (`GMG_BUILD`, a process peak of its JAX setup) lives on the device. `device_total`'s
  phase term is max(step, the device builds), and `host_total`'s is the host max above.
- **The searches,** internal to memory.py (leading underscore, so not public API), each
  taking solve_modes' cached estimators; every argument after `n_dof` is keyword-only:
  - `_c_max(op_memory, pc, n_dof, *, chunk, guards, low, basis_dtype, n_rigid, n_avail,
    budget, host_budget)` → int or None. It computes its own lo (above) and hi;
  - `_slice_capacity(op_memory, pc64, n_dof, *, n_store, block, n_rigid, budget,
    host_budget)` → int or None;
  - `_count_block(op_memory, pc64, n_dof, *, voxel, n_rigid, n_store, window, n_probes,
    budget, host_budget)` → int or None, through `_largest(fits, 1, 1, n_probes)`. `voxel`
    picks `count_memory`'s `COUNT_STEP` row;
  - `_default_budget()` → (budget, host_budget), with host_budget None on CPU. On GPU a user
    `memory_budget` sets the device budget, and the host budget stays 0.8 × RAM.
  - In api.py, `_ladder(reach, need)` → chunk or None is the pure rule of test 3:
    - `reach` is the {chunk: C_max or None} dict, also `info["c_max"]`;
    - `need` is None or a {chunk: threshold} dict, built by the caller from
      min(⌈1.25 n_modes⌉ + k, the free-space cap) per chunk (no rounding: $C_{max}$ is a
      multiple of $c_L$, so reach ≥ need is the same test).
  - solve_modes builds `op_memory, pc = (functools.cache(f) if f else f for f in
    _models(...))`. The slice searches reuse `op_memory`, with `pc64 = pc` unless the
    preconditioner is GMG in fp32 (the only model that depends on `precond_dtype`), else one
    cached fp64 model.
  - The levers' $C_{max}$ in the warnings is computed at the chosen chunk, or at chunk 5 on
    "memory".
- **One search helper,** `_largest(fits, step, lo, hi)`: the largest multiple of `step` in [lo, hi]
  with `fits(C)` True, by bisection (each `fits` is monotone), as a Python `int`, or None when
  nothing fits.
  - `probe_block`, $C_{max}$ and the slice capacity all use it. `probe_block` still raises
    `ValueError` when it gets None, as today.
- **$C_{max}(\text{chunk})$** = `_largest(fits, c_L, lo, up(n_avail + k, c_L))`, with lo =
  min(up(4·chunk + k, $c_L$), the free-space cap up(n_avail + k, $c_L$)), n_avail = free −
  rigid.
  - fits(C) applies the fit test to `lobpcg_memory(..., n_modes=C, capacity=C, band=True)`.
  - There is no output term, since the filter compacts V in place.
- **Slice capacity $C$** = `_largest` over [up(2c + 2b, c), the need cap], or the override.
  - The fit test runs on `si_memory(op_memory, n, C − W, n_modes=N_s, block, pc, n_rigid,
    n_deflate=W, host_window=2(C − 2b))`. pc is the fp64 preconditioner model (`_models(...,
    "float64", ...)`, as `band_probe_block`), since `make` is always fp64.
    - The device basis Q is exactly up(C, c).
    - host_window bounds the host window plus si_checked's [W | V] restart copy. The window
      never exceeds C − 2b columns, including a slice requeued whole.
    - The store reservation:
      - $N_s$ = `n_modes` if given, else 200, the roadmap's assumed band size;
      - after a capacity stop, at least 1.25 L. On a slice from 0 by the μ₁ rule, the store
        also holds the few modes in [μ₁, λ_lo(1 − 100t)), which L excludes. They lie within
        m of μ₁, and $N_s$ only drives a warning (documented).
  - The need cap: up(1.25 (1.8 $N_s$ + 84) + W, c).
  - si_slice already stops at the free space.
- **Slice counts.** si_band's counts (leaf, span, verification) run with `count_options["block"]`
  unless the user gave one.
  - The block is the largest $b_p$ passing the fit test with:
    - `count_memory(b_p)` (the operator, the fp64 preconditioner model, R and M R);
    - plus the store ($N_s$ columns) and the host window (2(C − 2b) columns), 8n per column, as
      exact host terms.
  - CPU: one check. GPU: `count_memory` against the device budget, the store and window against
    the host budget. Q is freed when counts run.
  - **One probe not fitting** is reachable: the count's 64-column W chunk keeps about 128
    vectors on the device at any block, which can exceed a floor-size slice's Q. Then:
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
  leaf's time (0.3 instead of 2 when τ_c is looser than the rule, as `_bound`).
  - $t_\text{minres}$ is warm: (MINRES time − the first step's) / (iterations − the first
    step's), so the first solve's compile does not inflate the bound.
  - si_slice gains an info field `minres_first_s`, the first step's MINRES seconds. A
    discarded σ-move solve counts in the first step. `info["time"]["minres"]` already includes
    it, while `minres_iterations` excludes it, so it leaves both the numerator and the
    denominator.
  - With one step, the cold mean is used;
- otherwise calibrate (PR 14b's `_calibrate`, with the leaf's count closure and preconditioner)
  and count iff the prediction ≤ the leaf's time. `_slice_degree` returns $g_\text{eff}$ as
  its gap, the value `_bound` reads.

The details:
- **Order.** The degree comes from λ only, through a new slice-only helper in shift_invert.py:
  `_slice_degree(lam_in, lo, hi, nb_lo, nb_hi, n_rigid, n_free, degree=None)` → (d, gap,
  lam_c).
  - It mirrors exactly what `_count_slice` (shift_invert.py:436–444) passes to `count` at
    lo = 0 and to `band_count` for lo > 0. `lam_in` is the in-band set, and nb_lo/nb_hi are
    the neighbours' λ.
  - A user `count_options["degree"]` is returned as d.
  - `count`, `band_count` and `_run` are unchanged.
  - The neighbours come from `_count_nbs`/`_beyond`, unchanged: they gather at most one
    column per side, and keep the "no converged neighbour" RuntimeError. Only the in-band
    gather (`W_in` in si_checked, `cols(inb)` in `_span_count`) moves after the decision, so
    both skips happen before it.
- **Restarts.** The rule is decided once per leaf, on its first count. si_checked's restarts
  follow that decision. The falsy-verify sites (`_predict=not verify` and `if not verify:`
  at shift_invert.py:548, 551, 719, 720) become explicit three-way tests, so None never acts
  as False.
- **Reruns.** si_band's widen-and-rerun loop (shift_invert.py:816–836) calls si_checked again
  for the same leaf. Each si_checked call decides on its own first count, so a rerun
  re-decides from its own slice (documented; it costs at most one more bound or calibration).
- **Times.**
  - si_slice gains `info["time"]["total"]`, its wall time from entry (line 230's `t0`) to
    return.
  - A run leaf's time is the sum of `time["total"]` over the si_slice calls of its si_checked
    call up to the deciding count (normally one); `make(σ)` is not included.
  - A span's time and $t_\text{minres}$ are its parent run's. si_checked records both in its
    `info["count"]` (`t_leaf`, `t_minres`), and si_band passes them to `_span_count` as two
    new keyword arguments.
  - This reads the roadmap's "no more than the solve" per leaf, so a run and its span may each
    spend up to the run's time (logged).
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
  - the capped growth rule through `_grow(V, count, new, c_L, dtype)` and `_insert`, with
    the caller's drop (`B = None`; a comment states that the V write and residual reads have
    already synchronised B's readers);
  - the capacity stop's return (the stopping chunk removed);
  - band mode returns its owning V resized in place to `end` (inline, warning fallback);
  - the count + k stop;
  - V is a preallocated zeroed F-order array;
  - non-band mode is unchanged.
- `si_band`: `verify=None`, `n_store` and its warning, the `_cap_error` wording.
- `si_checked`/`_span_count`: three-way `verify`; the in-band gather after the decision;
  `si_checked` records `t_leaf`/`t_minres`, and `_span_count` takes them; `si_slice`:
  `minres_first_s` and `time["total"]`; `_slice_degree`.
- `counting.py`: `_bound` and `_calibrate` moved from `api.py`; `band_count(_steps=)`.
- `api.py`:
  - `solve_modes`:
    - `band=(f1, f2)`, `strategy`, `memory_budget`, `slice_options`;
    - the ladder;
    - the μ₁ rule and its refinement;
    - the fallback with its frees;
    - the early sort (in-place permutation, the recomputed cut) and the in-place compacting
      filter;
    - the `band=(0, f₂)` probe budget;
    - the slice count block;
  - `check_band` for $f_1 > 0$ (messages "f1 >= 0", "f1 < f2");
  - the NotImplementedError removed.
- `memory.py`:
  - term tags and the per-space entries in `lobpcg_memory`, `si_memory`, `count_memory`;
  - `lobpcg_memory(capacity=)` and its host max;
  - `si_memory(host_window=)` and its store meaning of `n_modes`;
  - `_largest`, also used by `probe_block`;
  - the $C_{max}$, capacity and count-block searches, the fit test, the default budget;
  - `_amg_scan(mesh)` with `lru_cache(maxsize=1)`;
  - `solve_modes_memory`'s own $f_1 > 0$ check.
- Existing tests:
  - `tests/test_band.py`:
    - line 151 asserts an owning, F-contiguous result with exactly the band's columns (it
      asserted a copy);
    - the `spy` fixture (line 74) stores `np.array(W)`, not W itself, so it holds no reference
      to the returned V (otherwise the resize falls back and warns);
    - the `_bound`/`_calibrate` monkeypatches (lines 283–384) target `counting`;
    - `test_errors` (line 473): the interior-band row now succeeds or moves to test_strategy,
      and the `check_band` rows follow the new messages;
  - `tests/test_shift_invert.py:1826`: the `verify=None` row becomes `verify="yes"`, matching
    the new message "verify must be True, False or None";
  - si_memory callers pass the store columns as `n_modes`;
  - test_options.py pins the `slice_options` keys.

## Benchmark (`benchmarks/strategy_study.py`, one invocation, ≤ 15 min, progress to a log)

1. **[15a, 15d] constants.** Refit α and $D_0$ per family on `shift_invert_study.json` by log
   least squares (no runs, seconds).
2. **[15d] crossover and ladder.** On hex20 $16^3$/AMG:
   - one LOBPCG band run (chunk 40) to about 1005 modes (about 6 min), recording the iterations
     and $t_\text{warm}$ per chunk;
   - from it, the crossover table per depth against the MINRES rows, and the held-out "full"
     slice row predicted against measured (about 1.4×);
   - chunks 5, 10 and 20 to about 200 modes (47–85 s each, measured; the benchmark comes to about 11 min), recording the time per mode
     against chunk, extrapolated to depth with the first run's $t_\text{warm}$ rise.
3. **[15d] memory record** (estimators only). `_amg_scan`'s cache (above) is shared across
   part 3, which runs mesh by mesh, since the scan does not depend on stored/on-the-fly Ke,
   the basis dtype or m. About 2.3 s per mesh at 10M. Every estimator call passes `n_rigid` (0 clamped, 6 free), so
   no rigid-mode computation runs at 10M.
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
  hard-coded or recorded. A budget "s × the need" means 1.05 E + s M at the given capacity.
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
   - `V` owns its data (`V.base is None`), is F-contiguous, and has exactly 8n·(band count)
     bytes; `id(V)` equals the id a `_solve` wrapper recorded for the solver's V (compacted in
     place, since i0 > 0), and no "returned modes copied" warning is raised;
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
   - The ladder, real run: a budget in the middle between the chunk-20 and chunk-40 needs gives
     the chunk the reach rule picks from the searched `c_max` table (5 on this mesh; chunk 10
     reaches 0.81–0.87 of the best).
   - A pure-function table checks the rule:
     - the 0.9 × best reach;
     - the known-depth branch: the largest fitting chunk misses the depth and a smaller one
       reaches it (expects the smaller one);
     - the known-depth fallback when no chunk reaches the depth;
     - a top chunk below 5;
     - a tiny mesh where up(4·chunk + k, $c_L$) exceeds the free-space cap.
4. **"shift_invert"**, with 15c test 2(a)'s configuration (modes 5–28, block 6, its seed) and
   `n_modes` given.
   - The budget comes from the search so that `capacity == 96`.
   - ≥ 2 runs; every leaf counted, < 0.5; the same set as "lobpcg".
   - One real-AMG copy, marked slow.
5. **Empty band:** strictly between two well-separated modes of the voxel "cube" (named gap).
   0 modes and count < 0.5 under both strategies; `from_zero` False.
6. **Clamped box**, `band=(0, f)`, "shift_invert": matches eigsh, `from_zero` True; the
   returned V is si_band's V_f itself (a wrapper around si_band records `id(V_f)` only, never
   the array or a weakref), owning, F-contiguous, with exactly the band's columns; no copy
   warning.
7. **$\lambda_{lo} < \mu_1$**, clamped and free, "shift_invert":
   - `from_zero` True, filtered, matches eigsh;
   - the margin, with μ₁^ub monkeypatched ≥ the true μ₁:
     - λ_lo = μ₁^ub(1 + 0.9e-2) gives True and (1 + 1.1e-2) gives False;
     - at tol = 1e-4, 2.9e-2 gives True and 3.1e-2 gives False;
   - correct with an inflated μ₁^ub;
   - the `sigma=` hint path on voxel $16^3$ with λ_lo ≈ 3μ₁, si_band stubbed by a spy: the
     multigrid refinement runs, μ₁(eigsh) ≤ `mu1_ub` ≤ μ₁(1 + 1e-3), `from_zero` False;
   - `sigma=0` on the free hex8: the refinement falls back to `sigma_estimate` (spy);
   - the Jacobi path on the 240-DOF "free" hex8 at λ_lo ≈ 3μ₁: the full `sigma_estimate` ran
     (spy), μ₁ ≤ `mu1_ub` ≤ 1.1 μ₁, `from_zero` False;
   - a hint never enters `mu1_ub`.
8. **Filter table:** pure helpers, plus a `solve_modes` harness with `lobpcg._solve`
   monkeypatched to return a crafted (λ, F-order V, info with `n_band`, `n_modes`, `block`,
   residuals, thresholds), as in test 2(b), for the `lam_c`, `converged` and `stopped` rows:
   - the exact edge, inside and outside 100t, a chain crossing λ_lo;
   - a non-monotone per-mode-t row: index a passes with a large t, a+1 fails with a small t,
     a+2 passes; asserts i0 = a;
   - per-mode t, the upper end taken from `n_band`;
   - the result compacted in place: the harness's `_solve` builds V internally and records
     only `id(V)`, so solve_modes holds the only reference. Asserts `id(result)` equals the
     recorded id (a fallback copy is allocated while V lives, so its id differs), owning,
     F-contiguous, `nbytes` exact, the kept columns, no warning, at i0 = 0 and i0 > 0. A
     second row keeps an extra reference: the `RuntimeWarning` is raised and the copy has the
     right values. Spies in this test (on `_verify`/`count`) record only `lam`/`lam_c`, never
     W;
   - all modes below λ_lo, and an empty λ: an empty result, no IndexError;
   - out-of-order LOBPCG λ (e.g. [1, 2, 5, 6, 4.8] with lo between 4.8 and 5, and [1, 5, 2, 3,
     6]) sorted with their columns before the cut:
     - each mode's r and t follow its λ;
     - `id(result)` is the solver's V's id, compacted;
     - `count["lam_c"]` comes from the sorted λ (`_verify` spied);
     - the cut is recomputed: a mode moved below λ_hi is kept, and a cut at `len(lam)` gives
       `converged` False and `stopped` "sorted cut at the end", with `_verify` not called;
     - a capacity stop with λ out of order still falls back to si_band;
     - the in-place cycle permutation against `V[:, order]` on random permutations.
9. **Lower-edge clusters:** the voxel cube (doubles, triples) with λ_lo on a double and in the
   middle of a triple chain, under both strategies. All copies returned, `from_zero` False.
10. **Edge above the spectrum** (through `solve_modes` on "tiny"): `f2` above the largest
    eigenvalue raises PR 14b's "more modes than the mesh provides" under "lobpcg". Under
    "shift_invert" it raises on the "no progress" path with "hi above the spectrum" in the
    message.
11. **Capacity and $C_{max}$:**
    - **Estimator terms:**
      - one hand-computed device and host term per estimator;
      - `lobpcg_memory(capacity=C)`'s basis exactly C, and its `host_total` the max of the
        operator's host build and V + the AMG build; on hex20 $16^3$, $32^3$ and hex8 $32^3$, the
        measured peak of `amg(..., aggregates=aggs)` with aggs held, plus `_nbytes(aggs)`, ≤
        `amg_memory(...)["build"]` (slow, darwin, as test_amg's `test_amg_build_peak`;
        ratios 0.63–0.83 measured in round 13);
      - the per-call estimator cache (spy): on a "fits" run with `count_options={"block":
        b}` (so no probe-block search) and on a slice path with no lever warning, cache misses
        are counted through the cached wrappers (`band_probe_block` and the lever models
        excluded; the width is passed positionally). Each (estimator, width) pair misses
        exactly once per band call, and a second band call does not reuse the first call's
        cache;
      - the levers in the "memory" warning: each lever not already set is evaluated alone
        (`_c_max(..., low=True)`, `basis_dtype="float32"`, and on unstructured meshes in mode
        "stored" one extra cached `_models` with mode "otf"), checked by spy;
      - `_amg_scan` runs once per mesh (spy; tests touching it call `_amg_scan.cache_clear()`);
      - si_memory's device basis exactly up(C, c)·8n, with `host_window` and the store only in
        the host term, and the Gram term $8Q^2$;
      - the per-space exact parts;
    - **The fit test:**
      - per space on GPU, `total` on CPU;
      - E × 1.05, M × 1.2;
    - **`_largest`:**
      - against brute force, including a tiny mesh where $8Q^2$ and the end phase matter;
      - the jump at cap > 0;
      - the floor rounded up to c (72 at c = 24);
      - None when nothing fits, and `probe_block` raising on it;
    - **Caps:**
      - the need cap and the budget cap each binding;
      - $C_{max}$ capped at the free space, a multiple of $c_L$;
    - **Platforms:** unified and GPU (monkeypatched platform and `memory_stats`): a tight host
      budget where the host V binds lowers $C_{max}$; the AMG build counts in the host max;
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
    - the boundary: at chunk 5, $C_{max}$ = 40, count 30 (count + k = $C_{max}$) locks to 35,
      which catches a `>=` mutant, and the next lock stops. A monkeypatched `_iterate`
      returning one j = 1 chunk reaches count + k = $C_{max}$ + 1, which stops;
    - the initial allocation capped, with n_modes = $C_{max} - k$;
    - V F-order zeros of C_max + k columns;
    - after a refill and the lock, B's first count + j columns equal V's cast, and the rest
      are 0, for `basis_dtype` fp64 and fp32 (the only `astype` path);
    - the state freed before a growth, under the 9-block step. A spy on `_grow`'s entry
      asserts that no live array (`jax.live_arrays()`) has the old B's shape. The spy holds
      no reference to B, and the configuration is chosen so that the old width is > 0 and no
      other live block has it (asserted);
    - a capacity stop drops the stopping chunk: `len(lam) == count`, `len(info["residuals"])
      == len(info["thresholds"]) == count`, `chunks[-1]["locked"] == 0`, V's shape is
      (n, count);
    - band mode's return: V owning, F-contiguous, (n, end), with the same id as the
      preallocated V (a `np.zeros` spy records it), and no warning;
    - `max_capacity` required in band mode;
    - test_band.py's growth tests (`test_growth_from_zero` and the capacity tests) still pass,
      with `max_capacity` passed through solve_modes.
13. **Cost-model record:** `strategy_study.json`'s constants equal a log least-squares refit of
    `shift_invert_study.json`'s rows (frac 0.5, non-floor, `recheck_1005`), with ratios in
    [0.75, 1.3]. The crossover and per-chunk entries are present.
14. **`si_band(verify=None)`:**
    - `_slice_degree` against `_count_slice(..., _predict=True)`'s (degree, gap, lam_c) on a
      table of leaves (lo = 0 and lo > 0, a MAX_DEGREE row, a user `degree=` row); d is None
      iff `capped`. The lo = 0 rows use real M-orthonormal columns, since `count`'s predict
      path reads them;
    - the warm $t_\text{minres}$:
      - a pure row on crafted si_slice info (`minres_iterations` [120, 50, 50], `time["minres"]`,
        `minres_first_s`, a σ-move solve) asserts the exact warm value, the move counted in
        the first step;
      - a run leaf whose cold mean gives "bound skip" and whose warm value gives "calibrate"
        takes the calibrate branch;
      - `minres_first_s` is recorded, with the one-step fallback to the cold mean;
    - a widen-and-rerun call re-decides on its own first count (spy);
    - `time["total"]` recorded by si_slice; a span leaf receives its parent's `t_leaf` and
      `t_minres`;
    - **run leaves** are faked at the `si_slice` level (as `spy_slice`, test_shift_invert.py
      1060–1078), so si_checked's three-way code runs. The fake returns crafted info
      (`minres_iterations`, `time["minres"]`, `time["total"]`, `minres_first_s`,
      `sigma_moved`), and
      `_count_slice` runs either for real with the exact preconditioner or as a stub that
      honours `_predict`/`_steps` and returns `degree` and `time`. **Span leaves** and test 15
      keep `fake_band`. Each branch, run leaves: MAX_DEGREE skip; bound skip with 2 and with 0.3; calibrated
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
    - the pinned `slice_options` keys (test_options.py);
    - forced "lobpcg" with no chunk fitting raises up front;
    - a given `chunk` that does not fit, under "auto" (moved from test 3's pure table, since
      `_ladder` only sees the reach dict): "memory";
    - the default budgets (monkeypatched `os.sysconf`, a fake `memory_stats` with
      `bytes_in_use`);
    - spies: `slice_options`, `tol`, `count_options` (a user `block` kept) and
      `precond_options` reach si_band and `make`;
    - `precond_dtype="float32"` on voxel through a slice path: `make` and the slice searches'
      preconditioner models are fp64;
    - a capacity stop with 999 locked modes (monkeypatched) leaves `stagnation_risk` False, and
      1000 sets it;
    - LOBPCG not converged under "auto": no fallback, `converged` False.
17. **Mac peaks** (slow, darwin, fresh subprocess, MallocLargeCache=0, each ≤ about 5 min with
    the run time recorded in the docstring):
    - **(a1) The growth transient** (the paging check; test 12 is the reference check),
      `_grow` directly (seconds).
      - Setup: voxel about $48^3$, low-memory step, $c_L$ = 15, B at 14 $c_L$ columns, V with
        about 13.5 $c_L$ columns written, growing to 21 $c_L$.
      - The order is pinned: while B is still alive, the test reads "footprint before" and
        resets the interval footprint (`proc_reset_footprint_interval`). It then drops B,
        keeping only `dtype`, as `_solve` does, and calls `_grow`, then reads the interval max.
        The growth runs ≥ 10 times (B regrown each time). The assertion is on the minimum
        over the runs: spurious interval spikes reached about 9 $c_L$ (round 12), and noise
        only adds, while a kept-B bug shows in every run.
      - Asserts: interval max − footprint before < (new − old + old/2)·8n.
        - A correct refill pages new − old columns (XLA CPU zeros are eager; asserted as a
          precondition), plus the transient, about 1–3 $c_L$ (135–148 columns in all
          measured, against the 210-column threshold: a margin of about 62 columns).
        - The bug it guards against is B kept alive across the allocation, which adds old =
          14 $c_L$ columns. The test runs that mutant ≥ 3 times (a held reference) and asserts
          that its minimum exceeds the threshold (340–352 measured).
        - `_solve` has no extra sync call: its V write and residual reads already synchronise
          B's readers, so the in-flight case does not arise there (a comment states this).
          The test does its own `block_until_ready` before the drop, since it calls `_grow`
          directly.
    - **(a2) The fit test** on a short real run: `_solve(max_capacity=...)` on a non-cubic voxel
      $52\times48\times44$ (no cube multiplicities at the chunk boundary), low-memory, chunk 10,
      with `lam_stop` past mode 100 (at mode 95 the growth stopped at 90). The growth sequence
      60 → 90 → 135 is asserted as a precondition, with MallocLargeCache=0 set in the child's
      environment (as benchmarks/harness.py:83 does), not via `os.environ` inside it.
      A ≥ 2 GB estimate is a precondition too. Run time 94–181 s measured.
      - **Per-growth jump.** A spy on `_grow` reads the footprint at its entry, after the
        caller's drop, and again at its return. This checks that B is allocated at exactly new
        columns; the refill transient is 17a1's. Asserts:
        - jump ≤ 1.05 × new·8n;
        - the entry footprint ≤ the previous post-lock footprint − 0.9 × old·8n (the free
          happened). The post-lock footprint is read by a spy on `_lock`. Expected: about
          −old − k (the state dropped, X0 and ≤ j V columns added), so the 0.1·old allowance
          also covers compile-time host retention.
        A slope over the whole run is not used: residual noise of about ±100 columns gave
        slopes of 2.1–2.5 in round 10's measurements.
      - **Whole-run peak,** with the baseline read after setup (mesh, operator, preconditioner
        built). Asserts: peak − baseline ≤ 1.05 E + 1.2 M − (the setup terms with their own
        factors: operator storage, R and M R at 1.05; preconditioner storage and
        overhead at 1.2), at the final capacity and locked count (V is lazily paged). Also
        asserted: peak − baseline ≥ 0.8 × (E + M − the setup terms) at unit factors, so an
        over-conservative estimate, which would quietly send bands to shift-invert, fails.
    - **(b)** `solve_modes(strategy="shift_invert", band=..., memory_budget=B, verify=False,
      n_modes=3, sigma=...)` on lumped voxel $80^3$. The band is pinned to the lowest 2–3
      elastic modes, so the MINRES depth is about 6, for about 4 min. The μ₁ refinement (the
      `sigma=` hint path: a GMG build and a 1-mode LOBPCG) runs inside the measured peak; its
      share of the run time is recorded in the docstring.
      - B is chosen so that the searched capacity equals the need cap: one slice, asserted with
        `len(info["slices"]) == 1`.
      - Asserts:
        - the peak ≤ B and ≤ the worst-case composition;
        - within 20% of the composition evaluated with the run's maximum window and stored
          count;
        - both reported.
    - **(c)** `band_count` on lumped voxel $80^3$ at a budget giving a block below 50, with
      `count_options={"degree": 8, "n_probes": <the budgeted block>}` and the store and window
      held as host arrays.
      - Exactly one block runs (asserted).
      - The peak is ≤ the fit-test composition and within 20% of it. Two runs, at degree 2 and 8 (the run times
        recorded),
        give peaks within 5% of each other.

The README example (`band=(f1, f2)` on the "short" box mesh, about 10 modes) runs in
test_readme.py with every printed value asserted.

`examples/m3b_bands.py` runs in `tests/test_examples.py`, on the "short" mesh (about 40 s
measured):
- `band=(0, f)` and `band=(f1, f2)` under "auto";
- one forced `strategy="shift_invert"`;
- each ends in a check against eigsh.

## Docs

- `docs/eigensolvers.md`:
  - **§Public API:**
    - the signature line with `strategy`, `memory_budget`, `slice_options`;
    - step 1's band checks;
    - step 5's strategy dispatch;
    - the Errors bullet (line 39): the band, strategy and budget argument errors, forced
      "lobpcg" with no chunk fitting or at the cap, si_band's per-case capacity errors,
      `solve_modes_memory`'s $f_1 > 0$ error;
    - the Returns bullet with the new `info` fields.
  - **A new section, "Interior bands and strategy":**
    - the API and the μ₁ check;
    - the ladder, and the memory-driven rule with the measured evidence per D and N;
    - memory and capacity:
      - the fit test and the macOS malloc-cache headroom;
      - small "memory" capacities, the store reservation and its warning;
      - the slice count block and its one-probe case, and why it is reachable;
      - the in-place compaction of the returned modes (and the copy fallback);
      - the 0.8 × RAM default assumes an otherwise idle machine (the OS and other apps also
        use the 0.2 × RAM headroom);
      - the refill rate on CPU: about 9.5 GB/s for the warm `jnp.asarray` alone, about
        3.3 GB/s including first-touch and `_insert` (under 10 s per growth at 2M DOF and 1500
        columns), the host F→C copy dominating;
    - the order check before the filter;
    - `verify=None` for slices;
    - errors, including late edge errors above the spectrum and their cost;
    - the stagnation warning;
    - the margin's recovery path and its "no progress" end;
    - δ_max on slices from 0.
  - **§Measured MINRES cost:** the $\alpha D(1 + D/D_0)$ constants per family.
  - **§Frequency band [0, f_max],** rewritten:
    - the Capacity bullet (line 107): capped growth, free-and-refill (B dropped after the V
      write and residual reads have synchronised its readers, so XLA CPU frees it at once;
      whole-chunk refill), preallocated V;
    - line 103's `n_modes` sentence: for $f_1 = 0$ it is a lower bound on the returned count,
      for $f_1 > 0$ the store reservation's floor;
    - the sort right after `_solve` with the in-place column permutation;
    - the Returns bullet (line 108): the solver's V compacted in place to the band (an owning
      F-order array), and the band `info` fields `stopped` ("capacity", "sorted cut at the
      end") and `c_max`;
    - line 112: the count's probe budget on the "fits" path stays `solve_modes_memory(band=(0,
      f₂))` and ignores `memory_budget` (also annotated at open_items.md line 115);
    - the "Edge above the spectrum" bullet (line 106): LOBPCG raises only when $C_{max}$ is
      the free-space cap; otherwise a capacity stop, then si_band's late "hi above the
      spectrum" error, or the forced-"lobpcg" cap error;
    - line 320 ("V is a preallocated fp64 host array (n, n_modes)"): outside band mode; in
      band mode (n, $C_{max}$ + k), zeroed, F-order.
  - **§Checked slices** (lines 627, 633): `verify` three-way (None decided once per leaf,
    restarts follow it), the new skip reasons, `predicted_s` / `predicted_kind`; line 610:
    si_slice's `info` gains `minres_first_s` (the first step's MINRES seconds, a discarded
    σ-move solve included).
  - **The intro** (line 3): `band=(f1, f2)` and the interior-band section.
  - **The Validation list,** including line 829's "a copy, not a view".
  - **"Edge above the spectrum"** (near line 664): late errors only, with their cost on slice
    paths.
  - **The estimator reference** (near lines 112, 122, 348, 358, 562 and 674):
    - `lobpcg_memory(capacity=)` and its host max, `si_memory(host_window=)` and its store
      `n_modes`;
    - the per-space entries;
    - the 0.15 × 8n × n_modes growth note (line 358, measured on a plain chunked solve, not a
      concatenated V) kept, with 17a2's per-growth jump added next to it;
    - `count_memory` and `probe_block` (lines 554, 560), and the `count` options table's
      `block` row (line 517): the budgeted block on slice paths;
    - the si_band signature line (line 637) gains `verify=None` and `n_store`, and the
      Capacity-full paragraph (line 607) loses "15d checks edges up front";
    - `probe_block`/`band_probe_block` through `_largest`, and the `band=(0, f₂)` probe budget;
    - `solve_modes_memory(band=...)`: $f_1 > 0$ raises.
  - **The Options block:**
    - new rows for `strategy`, `memory_budget` and `slice_options` (block default 6);
    - rewritten rows:
      - `band`: the "256 fp64 columns up front" replaced;
      - `verify`, `precond_dtype`, and `count_options` (line 94: the block on slice paths);
      - the band row's `n_modes` sentence (there is no separate `n_modes` row);
      - `chunk` with `band` (near line 92): chunk 40 for $f_1 > 0$; the ladder replaces "pass
        `chunk` above 2M".
  - **Every mention** of `15d`, `PR 15d` and `NotImplementedError`, and "the roadmap's
    crossover", in `docs/` and `README.md` rewritten (eigensolvers.md near lines 103, 122,
    589–720).
- `docs/preconditioners.md` §"Memory: `amg_memory`" (lines 468–479): the mesh scan in `_amg_scan`,
  cached per mesh (`maxsize=1`), 2.3 s at 10.2M hex20; the gather path's `layout` cost (3.2 s
  per call at 10.2M). The `build` bullet (line 472): it also bounds the kept aggregates plus
  the rebuild with aggs held (test 11), or names the new `aggregates` term if that test fails.
  Line 479: keep "not in `total`", and add that the build does count in `host_total`, next to
  the host V, since both are alive at the AMG build.
- `README.md`:
  - the `band=(f1, f2)` example;
  - the options paragraph near line 222;
  - the band paragraph near line 247 (interior bands, the strategy);
  - the `examples/` sentence near line 259 gains `m3b_bands.py`;
  - the status, naming M3b;
  - the closing "See docs/eigensolvers.md" list gains "interior bands and strategy".
- `docs/ROADMAP.md`:
  - **Status:**
    - the Status paragraph; PR 15d status; PR 15 complete and milestone M3b done (with
      `examples/m3b_bands.py`);
    - the dependency table (PR 18 gains 15d);
    - line 107 (the Capacity bullet): "host `V` grows the same way (a host transient of old +
      new)" and "Pass `n_modes` … to avoid growth" → the preallocated V and the capped growth;
    - line 512 (PR 13c: slicing settings in the Options table): done through `slice_options`;
    - line 16:
      - PR 14b status ("No strategy selection yet", "grows 1.5×", "an edge above the
        spectrum raises"): a "(15d: strategy done; growth capped at $C_{max}$; LOBPCG
        raises only at the free-space cap, else a late error)" note;
      - PR 15a status ("`si_slice(...)` … internal until 15d"): "internal; reached through
        `solve_modes(band=...)` since 15d";
    - line 544 ("also used for [0, f_max] when the LOBPCG-from-0 basis does not fit"): done;
    - the PR 14 contents note "PR 15 adds the shift-invert branch" as done;
    - PR 15 Contents (line 545, "Slicing with the cost model … 15d: the cost model and
      strategy") and PR 15 Docs (line 550, "15d the memory model and calibration") as built;
    - the PR 15 Contents bullet at line 543 ("capacity per slice from the PR 15 memory
      model": from the budget search), the test bullet at line 561 ("the cost model switches
      to shift-invert": the memory-driven strategy), and the split sentence at line 541 (the
      cost model is evidence only);
    - line 97, "fixed block size b = 6" → "default b = 6".
  - **The δ_max ≤ 1 sentence** (Convergence/cluster bullet, line 80): what holds (per-pair
    thresholds; slices from 0 can exceed tol; `r_acc` reported; enforcement → PR 18).
  - **Frequency bands:**
    - the "Strategy: chosen by a cost model" header (line 88) and its bullets: memory-driven;
      the ladder; "≲ 1200" replaced by the measured crossover; the 10M and 2M Mac figures from
      the record (fp32 AMG for PR 18);
    - line 91, "`None` lets the cost model enable it": now PR 14b's rule per leaf;
    - lines 92–93, "Near mode index 1000 … PR 15 calibrates this … must expect stagnation": the
      15d note on $\alpha D(1 + D/D_0)$, the warning and `stagnation_risk`, remedies → PR 18;
    - the Band edges bullet (line 95): "the cost model's cap" → `MAX_DEGREE`; the up-front
      bound deferred, late errors only;
    - line 99, the $f_1 < \lambda_1$ sentence: the margin, the LOBPCG Ritz bound and the
      refinement;
    - line 100, "Slice widths come from the cost model … counting degree": capacity-limited
      widths and the per-leaf rule.
  - **Memory budget:** the "Max locked modes" row from the record under the fit test; a
    shift-invert row; the fit test; the Mac np.memmap bullet → PR 18.
  - **PR 18 contents:** the "band follow-ups (features)" and performance items above, including
    the depth-1000 remedies and the up-front spectrum bound.
  - **Deferred GPU checks:** the items in Out of scope.
- `docs/open_items.md`:
  - **Closed** (each moved to the Resolved entries at the end, with its 15d resolution, per
    open_items.md line 3):
    - 14b's "no strategy selection or basis-fit check";
    - 15a "`capacity` is required; 15d sets it";
    - 15c "`sigma` and `count_precond` are rejected … 15d exposes `make`'s options" (through
      `precond_options`);
    - the 15c → 15d items done here: capacity from the budget, the `verify` callable (now
      `verify=None`), "15d words it per case".
  - **15d clause resolved** (the decisions still hold; only their trailing 15d clause is
    rewritten): 15a "Cost driver … 15d fits its cost model on depth"; 15c "τ_c kept … 15d's
    cost model prices counts" (the per-leaf calibration).
  - **Annotated:** 15c "No provisional interior edges from a cost model" (line 139): `edges`
    is reachable through `slice_options`; its rationale is cited by the memory-driven
    decision.
  - **Superseded:**
    - 14b's "Basis growth … no `max_modes` cap" and "Host V holds n_modes + chunk columns";
    - 14b's "Band chunk default 40", annotated with the ladder for $f_1 = 0$ and chunk 40 for
      $f_1 > 0$;
    - 15a's "`si_memory` … takes the expected slice count `n_modes` for its end phase" (now the
      store's columns).
  - **Partly addressed:**
    - "`edge_unreached` is not evidence…" (line 147): the μ₁ rule done, the λ_hi bound → PR 18;
    - "A top remainder holding only known modes can raise" (line 169's last clause): the message fixed by the
      per-case wording, the cost → PR 18 with "halving before a capacity error";
    - line 167, "size the capacity with the window's chunk included" / "a capacity error
      discards the band": the window reservation; `err.result` → PR 18;
    - "Span counts at small capacities": the 96-column warning; the rest → PR 18, with its
      heading retagged "(PR 15c → PR 18)".
  - **Retagged to PR 18,** headings changed from "→ PR 15d" with a one-line 15d disposition per
    bullet:
    - "Late capacity errors and slice efficiency": "requeue whole"; "halving before a capacity
      error" (its "up-front $\hat\lambda_\max$ check removes the main case" now reads: the bound
      is deferred, so the case stays);
    - the 15c decision "a done side gives up its half's gaps" (line 141);
    - the 15c decision "A slice whose $r_{acc}$ widens its deflation window is rerun" (line
      153): "stop si_checked before its count";
    - the 15b "σ = $c_{hi}$" item in "Band-pass count and slice restarts" (its own line);
    - "the `verify` skips before the gather" for `verify=False` (None skips before it; False
      still predicts after it);
    - "Shift-invert cost near mode 1000" and "σ-move retries": warned, remedies to PR 18;
    - verification slices, per-step residuals, merge and count copies, host transposes.
  - **New:** the 0.8 × RAM default assumes an otherwise idle machine; PR 18 performance: the
    refill through `device_put(V[:, i:i+c].T).T` is 2.4× faster on CPU (round 13).
  - **New:** the macOS malloc-cache headroom (measured up to about 10 GB, plateauing near 8 GB
    under repeated growths, held by the 0.2 × RAM headroom; not visible to the
    MallocLargeCache=0 tests); PR 18 performance: cache `layout` for the gather path (about
    35 s of estimates per band call at 10.2M).
  - **Annotated:** "AMG build peak above 1M DOF" (line 68): the build also covers the
    count-time rebuild with aggs held (test 11).
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
  benchmark record. The 1.05 is checked by 17a2's per-growth jump.
- **The LOBPCG growth:**
  - keeps the rounding to $c_L$;
  - frees the state first;
  - drops B after the V write and residual reads, which already synchronise its readers, so
    the free is synchronous (no extra call; a comment states the invariant);
  - rebuilds B from the zeroed host V by whole chunks (one host copy each) through `_insert`, synchronised
    (at 10M the host F→C copy, about 16 s per 1000 columns, dominates PCIe; about 30–60 s per
    solve over all growths, negligible against the solve), instead of copying;
  - allocating B at $C_{max}$ up front on GPU, against fragmentation, is left to the GPU checks.
- **The initial allocation** is 14b's, capped, with n_modes = capacity − k.
- **LOBPCG's λ order:** when out of order (rare), sorted right after `_solve`, together with
  r, t and V's columns. The columns are permuted in place by cycles, so there is no copy, and
  the cut is recomputed. A sorted cut at the end of λ is reported as not converged.
- **The AMG estimate's mesh scan is cached** per mesh (`maxsize=1`).
- **The AMG count-time rebuild** is taken to be bounded by `amg_memory`'s `build`; if test 11
  shows otherwise, a fitted `aggregates` term is added.
- **L** counts the locked modes the filter's lower-end rule keeps (λ ≥ λ_lo(1 − 100t) with
  chain extension).
- **`verify=None` under restarts** is decided once per si_checked call, on its first count;
  restarts follow that decision, and a widen-and-rerun call re-decides.
- **Returned modes:** the solver's own F-order V, compacted in place and resized to the band,
  so modes below the band are not kept resident and no extra memory is needed. A copy is the
  fallback when another reference exists, with a `RuntimeWarning`. The resize runs inline in
  the owning frame. This refines 14b's compact copy.
- **The upper cut's per-mode-t mask** (`_band_count`'s `.sum()`) is kept as is: pre-existing,
  inside the 100t window; → PR 18.
- **No up-front spectrum bound:** the late errors meet the roadmap criterion, at about
  $\log_2 S + 1$ extra runs on slice paths; the bound → PR 18.
- **No run-time store check:** the $N_s$ reservation sizes the store; a one-time warning when
  it passes $N_s$; an overrun is an ordinary OOM.
- **Slice counts** run at a budgeted probe block on slice paths. When one probe does not fit,
  `verify=True` warns and runs without counts.
- **Slice searches** use the fp64 preconditioner model, since `make` is always fp64.
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
