# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands), the Memory budget section and
the PR 15c plan's split (15d = the cost model, the memory budget, the strategy choice,
`verify=None` for bands and `solve_modes(..., band=(f1, f2))`).

Revision 4.
- Round 1: no re-prediction per lock, no seeding of si_band, device and host memory terms,
  `slice_options`, capacity caps.
- Round 2: the strategy is memory-driven, with no time crossover.
- Round 3:
  - no hybrid junction: a capacity stop runs the slices on the whole band, and the junction is
    deferred to PR 18;
  - a chunk ladder (40 → 20 → 10 → 5) before the "memory" fallback, since chunk 40 does not fit at
    10M;
  - the basis rebuilt from the host V at a growth (no transient);
  - an F-order host V with no end copy;
  - a 1.2 margin;
  - cached, closed-form searches;
  - Mac peak and bound checks;
  - the crossover measured as total cost to depth 1005;
  - no cost constants in code.

## Scope

- `solve_modes(..., band=(f1, f2))` with $f_1 \ge 0$, plus `strategy` ("auto", "lobpcg",
  "shift_invert"), `memory_budget` (bytes) and `slice_options` (si_band's options).
- "auto": PR 14b's band mode from 0 with the largest chunk of the ladder 40, 20, 10, 5 whose
  chunk 0 fits the budget, and the basis capped at $C_{max}$. It falls back to `si_band` (PR 15c)
  on the whole band if LOBPCG stops at $C_{max}$ below $\lambda_2$ ("capacity"), or if no chunk
  of the ladder fits ("memory").
- The cost model, recorded in the docs and the benchmark as the evidence for the memory-driven
  rule:
  - MINRES iterations against depth;
  - LOBPCG's total time from 0 to depth 1005;
  - the smallest band for which one slice would beat LOBPCG.
- Memory:
  - `lobpcg_memory` and `si_memory` gain `device` and `host` entries;
  - $C_{max}$ and the slice capacity are searched against the budget: one unified check on CPU,
    a device and a host check on GPU.
- The $f_1 < \lambda_1$ rule (shift-invert from 0, filtered) and the up-front check of
  $\lambda_2$ against an upper bound $\hat\lambda_\max$ of the spectrum.
- `si_band(verify=None)`: PR 14b's three-stage rule per leaf.
- `si_band`'s capacity errors worded per case (PR 15c open item).
- `solve_modes_memory(..., band=(f1, f2), strategy=..., slice_options=...)`, the README
  example, the docs.

Out of scope (deferred, listed in `docs/open_items.md`):
- PR 18:
  - the hybrid junction: keep LOBPCG's in-band locked modes after a capacity stop and slice only
    above them. Rounds 2–3 found it needs a chosen junction gap, the lower neighbour plumbed out
    of si_checked and span leaves, a projection dedupe, a combined count and a host term. Its
    gain is the in-band locked modes, which matters once PR 18 raises $C_{max}$;
  - PR 15c's "Late capacity errors and slice efficiency" items;
  - `err.result`;
  - a memory-mapped host store (the roadmap's np.memmap promise moves there);
  - automatic `low_memory` / fp32-basis / on-the-fly-Ke levers in "auto".
- PR 18 and the deferred GPU checks: the depth-1000 remedies (15d flags the risk only); the GPU
  peaks of the new compositions; the cost ratio on the GPU.

## Roadmap tests tagged [15d]

| Roadmap test | 15d test |
|---|---|
| [15a, 15c, 15d] matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 1–7, 9 |
| [15b, 15c, 15d] interior band between two well-separated modes: 0 modes, count < 0.5 (the planted-mode half is 15b/15c's) | 5 |
| [15a, 15c, 15d] clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` | 6 |
| [15d] f₁ below λ₁ solved as [0, f₂], filtered, matches `eigsh` | 7 |
| [15d] memory budget too small for the LOBPCG-from-0 basis → shift-invert | 2, 3 |
| [15a, 15d] MINRES iterations vs δ measured (15a), cost-model constants fitted | 13 (benchmark record) |
| PR 15 contents: capacity per slice from the memory model | 11, 17 |
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
  - "lobpcg": PR 14b's band mode from 0, filtered, with the given chunk, uncapped;
  - "shift_invert": `si_band` on the band.

  Without `band`, any `strategy` but "auto", `memory_budget` or `slice_options` raises.
- `memory_budget`: bytes. Default on GPU: `bytes_limit − bytes_in_use` at entry
  (`memory_stats()`), the usable pool under `XLA_PYTHON_CLIENT_MEM_FRACTION` minus what the caller
  holds. On CPU: 0.8 × physical RAM (`os.sysconf`; unified memory on the Mac; the 52 GB
  benchmark guard on 64 GB), not reduced by what the caller holds (documented). The GPU host
  budget is 0.8 × physical RAM. The searches compare 1.2 × the estimate with the budget: the
  estimators run up to 9–16% low (si_memory 1.09–1.11×, GMG LOBPCG 1.12×, a one-chunk
  zero-workspace solve 1.16×), and 1.2 is the roadmap's ±20% validation band. Recorded in
  `info["budget"]`.
- `slice_options`: dict with keys `tau`, `maxit_inner`, `block`, `max_slices`, `capacity`
  (overrides the sizing), `edges`, `verbose` (validated as si_band does).
- How the other `solve_modes` arguments reach the slices:
  - `tol`, `verify`, `count_options`, `seed` → si_band;
  - `precond`, `precond_options` → si_band's `make(σ)`, always fp64. With
    `precond_dtype="float32"` only LOBPCG's GMG is fp32, since MINRES needs a symmetric cycle;
  - `basis_dtype`, `low_memory`, `chunk` → LOBPCG and $C_{max}$. A given `chunk` replaces the
    ladder;
  - `sigma` → LOBPCG's preconditioner and $\lambda_1^{ub}$.
- `n_modes`: for an interior band, the expected modes in the band. It sizes the slice store
  only; LOBPCG gets `band_defaults(None, chunk)`. For $f_1 = 0$ as PR 14b.
- Returns the band modes and `info`:
  - The band filter is si_band's `_extend_chain` with per-mode t: λ ∈ [λ₁(1 − 100t),
    λ₂(1 + 100t)], extended over chains with relative gaps < 100t. t is the LOBPCG threshold or
    r_acc.
  - `strategy` ∈ {"lobpcg", "shift_invert"};
  - `reason` ∈ {"forced", "fits", "capacity", "memory"};
  - `chunk`, `stagnation_risk`, `budget`, `from_zero`, `lambda1_ub`;
  - `count`: PR 14b's or si_band's summary;
  - with slices: si_band's `slices`, `capacity`, `merged_edges`, `collapsed`, `edge_rr_failed`,
    `duplicates`, `r_acc`.

### Up-front checks (before the σ pre-estimate and any build)

- **Spectrum bound** $\hat\lambda_\max = \max_e \lambda_\max(K_e, M_e)$.
  - Why it bounds the spectrum: for any $x$, $x^\top K x = \sum_e x_e^\top K_e x_e \le \max_e
    \lambda_e \sum_e x_e^\top M_e x_e \le \hat\lambda_\max\, x^\top M x$. Nodal masses, masking
    and rigid deflation only lower the Rayleigh quotient; ρ > 0 is enforced and lumped masses
    are positive.
  - One exact tier: batched Cholesky and `eigvalsh`.
    - Voxel meshes: one reference element per phase on the host. With per-cell scales the
      bound is $\lambda_\text{phase}\max_e s_{K,e}/s_{M,e}$.
    - Unstructured meshes: `element_matrices` (PR 12) in device chunks, about 200 µs per hex20
      element: about 16 s at 1M DOF, minutes at 10M.
    - Recorded in `info["time"]["bound"]`.
  - A singular $M_e$ (a massless phase, $s_M = 0$) gives ∞, and the check never raises.
  - $\lambda_2 > \hat\lambda_\max$ raises `ValueError`. An edge in $(\lambda_\max,
    \hat\lambda_\max]$ still reaches si_band's capacity error or PR 14b's "more modes than the
    mesh provides". The tests keep `lam_hat` = ‖K‖/‖M‖ for the eigsh tolerance.
- **$f_1 < \lambda_1$** applies on every slice path. $\lambda_1^{ub}$:
  - after LOBPCG locked modes ("capacity"): its lowest elastic mode;
  - otherwise: the σ pre-estimate, a Rayleigh quotient and so an upper bound. With a `sigma=`
    hint or Jacobi, `sigma_estimate` (10 iterations) gives it.

  If $\lambda_1 < \lambda_1^{ub}(1 + 10^{-2})$, si_band runs on $[0, \lambda_2]$ and the result
  is filtered; `info["from_zero"]` is True. The 1% margin keeps λ₁ out of si_band's widened
  first-slice window.

### Strategy ("auto")

1. **Chunk ladder.** Take the largest chunk in (40, 20, 10, 5) whose `lobpcg_memory` with a basis
   of 4 chunks fits the budget. If none fits, "memory": si_band on the whole band, with a warning
   naming the levers (`low_memory=True`, `basis_dtype="float32"`, on-the-fly Ke) and their reach
   from the estimators.
2. **LOBPCG** `_solve(..., lam_stop=λ2, max_capacity=C_max)`:
   - Growth `new = min(max(1.5 old, need), C_max)`; the initial allocation is
     `min(up(n_modes + k, c), C_max)`.
   - At a growth, B is freed first, then reallocated and refilled from the host V in chunks of c,
     cast to `basis_dtype`. B is exactly V's columns cast, so the rebuild is bitwise identical,
     and there is no old+new transient.
   - The host V is preallocated once at $C_{max} + k$ columns in F order (untouched columns cost
     no pages), never concatenated. The result is `V[:, :end]`, a view with no end copy.
   - The band-end check comes first. If the next lock would pass $C_{max}$, the solve stops with
     `info["stopped"] = "capacity"`.
   - Reaching $\lambda_2$ ("fits"): PR 14b's count at the cut above $\lambda_2$. Then the filter
     copies only the $[\lambda_1, \lambda_2]$ columns into an F-order output.
3. **Capacity stop** ("capacity"). LOBPCG's basis, state, preconditioner and V are freed after
   reading $\lambda_1^{ub}$. si_band runs on $[\lambda_1, \lambda_2]$, or from 0 by the rule
   above.
4. **Stagnation.** If the band's lower depth (λ₁'s index: LOBPCG's locked count when it reached
   λ₁, else the $a = 3/2$ Weyl bound) is ≥ 1000 on a slice path, a warning and
   `stagnation_risk` True. This follows 15a: a MINRES solve near depth 1005 is marginal at
   τ = 1e-8 on hex20/AMG.

There is no time-based choice. LOBPCG from 0 to depth D plus the band costs less than one slice's
fixed start (about 84 columns, 8–14 block steps of $\alpha D(1 + D/D_0)$ MINRES iterations) at
every measured depth. The benchmark records this.

### Cost model (benchmark and docs only; no constants in code)

- MINRES iterations per solve: $\alpha D(1 + D/D_0)$, fitted on the frac-0.5 non-floor rows of PR
  15a's table. hex20/AMG: about α = 5.8, $D_0$ = 650, ratios 0.87–1.20 over depths 6–1005.
  voxel/GMG (depths ≤ 201): α = 4.35, $D_0$ about 760, weakly constrained. This measured form
  replaces the roadmap's "∝ ln(1/τ)/δ plus an outlier term" (15a: depth drives the iterations).
- LOBPCG from 0: measured per chunk (iterations rising with depth, $t_\text{warm}$) to about
  1005 modes on hex20 $16^3$/AMG, the MINRES rows' mesh.
- The crossover: per depth D, the smallest band N for which one slice, $(84 + 1.8N)/b$ steps ×
  $\alpha D(1 + D/D_0)$ × $t_\text{minres}$, beats LOBPCG's measured time to D + N. "None"
  is expected. Per mode, a slice mode costs about 25–110 LOBPCG modes.
- Not modelled, stated in the docs: LOBPCG's projector growth at 10M and depth about 1000 (about
  4nDk flops per iteration). Beyond the measured mesh, the ratio is extrapolated.

### Memory

- **The split.** Both estimators gain `device` and `host` entries.
  - `lobpcg_memory`: host = $8n(\text{n\_modes} + k\cdot\text{band})$ (V); device = total − host
    (its start, seed and slack blocks stay on the device).
  - `si_memory`: host = $8n\,\text{n\_deflate} + 8C^2 + 8n(\text{n\_modes} + 2)$; device = the
    rest (the end phase holds one device chunk).
  - Unified memory (CPU): device + host ≤ budget. GPU: device ≤ budget and host ≤ the host
    budget.
- **$C_{max}$.** The largest multiple of LOBPCG's c with `lobpcg_memory(..., n_modes=C − k,
  band=True)` × 1.2 fitting; the host term is V at $C + k$ plus the filter's output, at most
  $C$ columns.
- **Slice capacity $C$** is the min of three caps.
  - The budget: the largest multiple of c with `si_memory(op_memory, n, C − W, n_modes=⌈1.25
    m_s⌉, block, pc, n_rigid, n_deflate=W)` × 1.2 fitting; W = 2c, so the device basis is
    exactly C. The host term adds the store $8n\hat N_\text{band}$ and $2m_s$ columns (a
    span-rerun window and si_checked's [W | V] restart copy). $m_s = \max(1, (C - W - 84)/1.8)$.
  - The need: up(1.25 (1.8 $\hat N_\text{band}$ + 84) + W, c).
  - The free space: up(n_free − n_rigid, c).
  - $\hat N_\text{band}$ is `n_modes` when given, else the $a = 3/2$ Weyl bound
    $(\lambda_2/\lambda_1^{ub})^{3/2}$ (with the locked count k₀ at λ₁^ub after a capacity stop:
    $k_0(\lambda_2/\lambda_1^{ub})^{3/2}$) minus the modes below λ₁. This is an overestimate,
    which is safe for sizing.
  - Below 96 columns (PR 15c's slice minimum at b = 6) a warning; below 2b + c a `ValueError`
    naming the budget. `slice_options["capacity"]` overrides all three caps.
  - Not modelled (documented): the merge's copies per merged-window column in the collapse case;
    macOS keeping freed large blocks resident.
- **Searches.** `op_memory` and `precond_memory` are evaluated once per width (cached in
  `_models`). Both totals are affine in C, so C is solved in closed form, then checked.
- **Probe blocks.** PR 14b's count is unchanged; si_band's leaf counts as 15c.
- **`solve_modes_memory(mesh, bcs, n_modes, band=(f1, f2), strategy="lobpcg",
  memory_budget=None, slice_options=None)`.**
  - "lobpcg" (the default) keeps PR 14b's return shape.
  - "shift_invert" returns the slice composition, with the capacity from the budget unless
    `slice_options["capacity"]` is given. `slice_options["block"]` reaches `si_memory`.
  - There is no "auto" mode: auto fits the budget by construction.

### `si_band(verify=None)`

`verify` ∈ {True, False, None}. None applies PR 14b's rule per leaf:
- skip above `MAX_DEGREE`;
- skip when the a-priori bound $2\,n_\text{blocks}\,d\,(b_p/b)\,t_\text{minres}$ exceeds the
  leaf's time (0.3 instead of 2 when τ_c is looser than the rule, as `_bound`);
- else calibrate (PR 14b's `_calibrate` with the leaf's count closure and preconditioner) and
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

- `lobpcg._solve`: `max_capacity` (initial allocation, growth by free-and-refill from V, the
  stop), F-order preallocated V, a view at the end.
- `si_band`: `verify=None`, `_cap_error` wording.
- `counting.py`: `_bound`, `_calibrate` moved from `api.py`.
- `api.py`:
  - `solve_modes`: `band=(f1, f2)`, `strategy`, `memory_budget`, `slice_options`, the ladder,
    the checks, the fallback, the filter;
  - `check_band` for $f_1 > 0$;
  - the NotImplementedError removed;
  - `spectrum_bound(mesh, materials, ...)`.
- `memory.py`:
  - `device` and `host` entries;
  - cached models;
  - the $C_{max}$ and capacity searches;
  - the default budget;
  - `solve_modes_memory(band=, strategy=, memory_budget=, slice_options=)`.

## Benchmark (`benchmarks/strategy_study.py`, one invocation, ≤ 25 min, progress to a log)

1. [15a, 15d] constants: refit α, $D_0$ per family on `shift_invert_study.json`. No runs,
   seconds.
2. [15d] crossover: one LOBPCG band run on hex20 $16^3$/AMG to about 1005 modes, recording
   per-chunk iterations and $t_\text{warm}$ (10–15 min). The crossover table per depth is
   computed against the MINRES rows; the held-out "full" slice row is predicted against measured
   (about 1.4×).
3. [15d] memory record at 10M hex20 (estimators only, seconds with the cache): the ladder
   chunk × {9-block, low} × {fp64, fp32 basis} × {stored, on-the-fly Ke} gives chunk 0 fits and
   $C_{max}$. It checks the roadmap's 700–1000 and updates its 10M figures.
4. [15d] Mac peak (about 8 min, MallocLargeCache=0): a capacity-stopped LOBPCG band on hex20
   (about 500k DOF), with a budget setting $C_{max}$ to about 8 chunks and at least one growth
   (estimate ≥ 2 GB). Measured peak ≤ budget, and peak against the estimate.

Results go to `benchmarks/results/strategy_study.json` with the git commit.

Not run: the GMG depth-500 row and the slice-model refit. They refine predictions that decide
nothing (logged).

## Tests (`tests/test_strategy.py`; roadmap tags in the docstrings)

Method:
- Real counts run with the dense exact $(K + \sigma M)^{-1}$ (15c's `exact`) monkeypatched into
  `api.amg`/`api.gmg` and si_band's `make`, with each count's degree asserted ≤ about 1000 as a
  precondition (≤ 15 s per count on "short").
- Wiring-only tests use `count_options={"degree": 4, "n_probes": 2}`.
- Budgets come from the estimators, never hard-coded.
- Each test is timed and marked slow above about 20 s.
- Every test mapped to the eigsh item asserts the roadmap bound per mode with that mode's t,
  through test_shift_invert's `check_band` logic generalised.

1. **"fits"** ("short", hex20 $6\times5\times4$): an interior band of about 10 modes inside chunk
   0. Asserts reason "fits", `from_zero` False, `info["chunk"] == 40` with `n_modes` given, and
   that it matches eigsh.
2. **"capacity"**:
   - setup: `chunk=8`, the budget at the top of the $C_{max} = 52$ window, with the slice capacity
     ≥ 96 asserted as a precondition;
   - band about modes 25–45: LOBPCG stops inside it, `stopped == "capacity"`, si_band runs on
     (λ₁, λ₂) (spy), it matches eigsh and the slice counts < 0.5;
   - λ₁ below LOBPCG's lowest mode: `lambda1_ub` from the locked modes.
3. **"memory"**:
   - budgets at 1.15× and 1.25× the chunk-5 estimate give "memory" and LOBPCG, which brackets the
     1.2 factor;
   - with f₁ < λ₁, `from_zero` is True and it matches eigsh;
   - the ladder: a budget between the chunk-20 and chunk-40 estimates gives `info["chunk"] == 20`.
4. **"shift_invert"** with 15c test 2(a)'s configuration (modes 5–28, block 6, its seed) and the
   budget set to give `capacity == 96`:
   - ≥ 2 runs, every leaf counted < 0.5, the same set as `strategy="lobpcg"`;
   - one real-AMG copy, marked slow.
5. **Empty band**: strictly between two well-separated modes of the voxel "cube" (named gap):
   0 modes, count < 0.5, both strategies, `from_zero` False.
6. **Clamped box**, `band=(0, f)`, "shift_invert": matches eigsh.
7. **$f_1 < \lambda_1$**, clamped and free, "shift_invert": `from_zero` True, filtered, matches
   eigsh.
   - The margin: λ_lo = λ₁^ub(1 + 0.5e-2) gives True and (1 + 2e-2) gives False, with λ₁^ub
     monkeypatched to ≥ the true λ₁.
   - Correct with an inflated λ₁^ub.
   - The `sigma=` hint and the Jacobi paths (Jacobi on the 240-DOF "free" hex8).
8. **Filter table** (pure): the exact edge, inside and outside 100t, a chain crossing $f_1$ and
   $f_2$, per-mode t.
9. **Lower-edge clusters**: the voxel cube (doubles, triples) with $f_1$ on a double and mid
   triple chain, both strategies, all copies returned, `from_zero` False.
10. **Spectrum bound**:
    - $\hat\lambda_\max \ge$ the dense largest eigenvalue on hex8, hex20, voxel, lumped and
      nodal-mass meshes, a graded `box_mesh` whose smallest element is not first, a two-phase
      voxel whose stiffer phase is not phase 0, a scaled voxel operator, and a forced small
      `element_matrices` chunk with the stiffest element in a partial last chunk;
    - equal on a one-element mesh without BCs; a massless phase gives ∞;
    - $\lambda_2 > \hat\lambda_\max$ raises before any build (spies on `_solve`, `si_band`,
      the builders, `_precond_sigma`);
    - $\lambda_2$ just below $\hat\lambda_\max$ does not raise up front.
11. **Capacity and $C_{max}$**:
    - one hand-computed device and host term for a fixed config;
    - each binding cap against a brute-force loop: the need, the free space (tiny mesh), the
      budget;
    - unified and GPU (monkeypatched platform, `memory_stats`); the 1.2 factor;
    - below 2b + c raises naming the budget, below 96 warns;
    - an override above the need is kept;
    - `solve_modes_memory` "lobpcg" (unchanged shape), "shift_invert", `slice_options["block"]`.
12. **`_solve(max_capacity=...)`** directly:
    - the capacity sequence, with growth straight to the cap;
    - a lock that exactly fills $C_{max}$ is allowed; one more column stops;
    - the initial allocation capped;
    - V F-order and preallocated (no concatenate); B after a refill equals V cast;
    - `stopped == "capacity"`.
13. **Cost-model record**: `strategy_study.json`'s constants equal a refit of
    `shift_invert_study.json`'s frac-0.5 non-floor rows (ratios in [0.8, 1.25]), and the
    crossover entries are present.
14. **`si_band(verify=None)`**:
    - `fake_band`: each branch (MAX_DEGREE skip; bound skip with 2 and with 0.3; calibrated and
      dear, skip; calibrated and cheap, count; `_calibrate` spied); a span leaf's time is its
      parent's;
    - one real exact-preconditioner run: per-leaf `reason`/`predicted_kind`, `ran` True,
      estimate < 0.5;
    - `solve_modes(strategy="shift_invert")` at the default verify records them;
    - slow, real AMG on one leaf (hex20 $12^3$, as 14b's `test_calibration`): the bound is ≤ the
      measured warm count time, and the calibrated prediction is within [0.5, 2]×.
15. **Capacity error wording** (`fake_band`): lower side closed, upper side closed, lo = 0.
16. **Options**:
    - invalid input raises: `strategy`/`memory_budget`/`slice_options` without `band`, bad
      values, unknown `slice_options` keys, `f1 >= f2`;
    - the default budgets (monkeypatched `os.sysconf`, fake `memory_stats` with `bytes_in_use`);
    - spies: `slice_options`, `tol`, `count_options`, `precond_options` reach si_band and
      `make`;
    - `precond_dtype="float32"` on voxel through a slice path builds `make` in fp64;
    - a slice path at depth ≥ 1000 (monkeypatched) warns and flags `stagnation_risk`.
17. **Mac peak** (slow, darwin): a capacity-capped LOBPCG band with one growth on voxel about
    $48^3$ (estimate ≥ 2 GB): the measured peak is ≤ the budget and within 20% of the estimate.

## Docs

- `docs/eigensolvers.md`:
  - a new section, "Interior bands and strategy": the API, the checks, the ladder and the
    memory-driven rule with the measured evidence, memory and capacity, `verify=None` for
    slices, errors, the stagnation flag;
  - the Options block: the new rows, plus `band`, `n_modes`, `verify` and `precond_dtype`
    rewritten;
  - the Validation list;
  - every mention of `15d`, `PR 15d`, `NotImplementedError` and "the roadmap's crossover" in
    `docs/` and `README.md` rewritten (eigensolvers.md near lines 103, 122, 589–720).
- `README.md`: the `band=(f1, f2)` example, the options paragraph near line 222, the status.
- `docs/ROADMAP.md`:
  - the Status paragraph, PR 15d status, PR 15 complete, the dependency table;
  - the strategy bullets: memory-driven, the ladder, "≲ 1200" replaced by the measured
    crossover, the 10M figures from the record;
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
  - name each 15c item that handed a choice to 15d's cost model with its outcome:
    - "a done side gives up its half's gaps", "requeue whole" and "σ = c_hi": retagged to
      PR 18, since there is no time-based cost model;
    - "span counts at small capacities": partly addressed by the 96-column warning;
  - the depth-1000 and σ-move items: flagged, remedies to PR 18;
  - the decisions below and the items not adopted.

## Decisions without user input (to log)

- **Strategy: memory-driven, no time-based crossover.** LOBPCG from 0 to the band costs less
  than one slice's fixed start at every measured depth. This deviates from the roadmap's "cost
  model chooses" and "≲ 1200" (corrected). The GPU ratio is a deferred check.
- **Capacity stop:** si_band on the whole band; the hybrid junction → PR 18.
- **The chunk ladder** (40, 20, 10, 5), a basis of 4 chunks required. No automatic `low_memory`,
  fp32 basis or on-the-fly Ke (they change time or accuracy; the warning names them) → PR 18.
- **Default budgets:** the GPU free pool; 0.8 × RAM on CPU, not reduced by the caller's
  holdings; the GPU host budget 0.8 × RAM; a 1.2 factor on every estimate.
- **The LOBPCG growth** rebuilds B from the host V (PCIe about 1 s at 10M) instead of copying.
- **Stagnation:** depth-1000 slices are flagged with a warning, remedies to PR 18. The 10M
  criteria near index 1000 may need the user's input after the GPU checks.
- **Slice capacity:** min(budget, need with a 25% margin, free space); a warning below 96.
- **The store** is sized by the $a = 3/2$ Weyl bound when `n_modes` is not given.
- **The $f_1 < \lambda_1$ rule:** $f_1$ within 1% above $\lambda_1^{ub}$ is treated as below
  it.
- **`n_modes`:**
  - for an interior band, it is the expected band count;
  - LOBPCG's chunk comes from the ladder or the `chunk` argument, never from `n_modes`.
- **`solve_modes_memory`** defaults to "lobpcg" (PR 14b's shape) and has no "auto" mode.
- **The spectrum bound** uses the exact tier only.
- **`err.result`** is deferred to PR 18; the error messages are worded per case now.
- **The slice cost form** is $\alpha D(1 + D/D_0)$; τ scaling and LOBPCG's projector growth are
  not modelled.
- **Not run:** the GMG depth-500 row and the slice-model refit.
- **Test tolerances:** slice tests run at tol 1e-6 with τ = 1e-8 (as 15c), instead of the
  roadmap's r_acc = max(1e-10, …) test setting, for run time. The eigsh bound is asserted per
  mode.
