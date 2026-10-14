# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands), the Memory budget section and
the PR 15c plan's split (15d = the cost model, the memory budget, the strategy choice,
`verify=None` for bands and `solve_modes(..., band=(f1, f2))`).

Revision 3. Round 1: no re-prediction per lock, no seeding of si_band, device and host memory
terms, counting cost, `slice_options`, capacity caps. Round 2:
- **No time crossover; the strategy is memory-driven.** The fitted cost model puts slices
  24–113× dearer per mode than LOBPCG at every measured depth (6–1005), and κ cannot close the
  gap. So LOBPCG runs as far as the budget allows, and slices take over above its capacity stop.
- **The junction is anchored on converged pairs,** not on LOBPCG's unconverged guard.
- **The host V is preallocated** at the cap.
- **The transient uses the max form.**
- **The budget leaves a 10% margin.**
- **The tests run on dense exact preconditioners.**
- **The benchmark is trimmed.**

## Scope

- `solve_modes(..., band=(f1, f2))` with $f_1 \ge 0$, plus `strategy` ("auto", "lobpcg",
  "shift_invert"), `memory_budget` (bytes) and `slice_options` (si_band's options).
- "auto": PR 14b's band mode from 0 with the LOBPCG basis capped by the budget; if the cap stops
  it below $\lambda_2$, `si_band` (PR 15c) solves the rest, joined to the locked modes at a
  counting edge between converged pairs. If even LOBPCG's chunk 0 does not fit, `si_band` on the
  whole band.
- The cost model (MINRES iterations against depth, LOBPCG time per mode) fitted and recorded in
  the docs. It is the evidence for the memory-driven rule, not a run-time decision.
- Memory:
  - `lobpcg_memory` and `si_memory` gain `device` and `host` entries;
  - the LOBPCG basis cap $C_{max}$ and the slice capacity are searched against the budget,
    one unified check on CPU and a device and a host check on GPU.
- The $f_1 < \lambda_1$ rule (shift-invert from 0, filtered) and the up-front check of
  $\lambda_2$ against an upper bound $\hat\lambda_\max$ of the spectrum.
- `si_band(verify=None)`: PR 14b's three-stage rule per leaf.
- `si_band`'s capacity errors worded per case (PR 15c open item).
- `solve_modes_memory(..., band=(f1, f2), strategy=...)`, the README example, the docs.

Out of scope (deferred, listed in `docs/open_items.md`):
- To PR 18:
  - PR 15c's "Late capacity errors and slice efficiency" items: verification slices,
    preconditioner reuse on a whole requeue, per-step residuals, merge copies, host transposes,
    `verify=False` input work, and stopping si_checked before its count on a window rerun;
  - `err.result` (the pairs found before a capacity error);
  - a memory-mapped host store, which moves the roadmap's np.memmap promise to PR 18.
- To PR 18 and the deferred GPU checks: the depth-1000 remedies (deflating converged modes from
  MINRES, a σ-aware multigrid, a lean LOBPCG fallback). 15d flags the risk only.
- To the deferred GPU checks: a measured peak of the new compositions ≥ 2 GB, and the
  per-mode LOBPCG/slice cost ratio on the GPU.

## Roadmap tests tagged [15d]

| Roadmap test | 15d test |
|---|---|
| [15a, 15c, 15d] matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 1–7, 9 |
| [15b, 15c, 15d] interior band between two well-separated modes: 0 modes, count < 0.5 (the planted-mode half is 15b/15c's) | 5 |
| [15a, 15c, 15d] clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` | 6 |
| [15d] f₁ below λ₁ solved as [0, f₂], filtered, matches `eigsh` | 7 |
| [15d] memory budget too small for the LOBPCG-from-0 basis → shift-invert | 2, 3 |
| [15a, 15d] MINRES iterations vs δ measured (15a), cost-model constants fitted | 13 |
| PR 15 contents: capacity per slice from the memory model | 11 |
| Band edges: an edge above the largest computed eigenvalue raises | 10 (up front); 15c tests 3, 11 (late) |

## Design

### API

```python
lam, V, info = solve_modes(mesh, materials, bcs, band=(f1, f2))           # auto
solve_modes(..., band=(f1, f2), strategy="shift_invert", memory_budget=8e9,
            slice_options=dict(tau=1e-8, block=6))
```

- `band=(f1, f2)`: $0 \le f_1 < f_2$ finite (Hz); `check_band` accepts $f_1 > 0$. $f_1 = 0$ runs
  the same code path (the filter keeps everything from 0).
- `strategy`:
  - "auto" (default);
  - "lobpcg": PR 14b's band mode from 0, filtered, no cap. It warns once when a basis growth
    passes the budget;
  - "shift_invert": `si_band` on the band, no LOBPCG.

  Without `band`, any `strategy` but "auto", `memory_budget` or `slice_options` raises.
- `memory_budget`: bytes.
  - Default on GPU: `bytes_limit − bytes_in_use` at entry, from `memory_stats()`. That is the
    usable pool under `XLA_PYTHON_CLIENT_MEM_FRACTION` (the roadmap's Memory budget) minus what
    the caller holds.
  - Default on CPU: 0.8 × physical RAM (`os.sysconf`; unified memory on the Mac; consistent with
    the 52 GB benchmark guard on 64 GB).
  - The GPU host budget is 0.8 × physical RAM.
  - The searches below compare 1.1 × the estimate with the budget (the estimators run up to
    7–11% low: lobpcg_memory at m = 25, the GMG peaks).
  - Recorded in `info["budget"]`.
- `slice_options`: dict with keys `tau`, `maxit_inner`, `block`, `max_slices`, `capacity`
  (overrides the sizing) and `edges`, validated as si_band does.
- The other `solve_modes` arguments reach the slices as follows:
  - `tol`, `verify`, `count_options` and `seed` → si_band;
  - `precond`, `precond_options` → si_band's `make(σ)`. It is always built in fp64; with
    `precond_dtype="float32"` only LOBPCG's GMG is fp32, since MINRES needs a symmetric cycle;
  - `basis_dtype`, `low_memory` → LOBPCG and $C_{max}$;
  - `sigma` → LOBPCG's preconditioner and $\lambda_1^{ub}$ (below).
- `n_modes`: for an interior band, the expected modes in the band (sizes the slice store; the
  LOBPCG chunk keeps its default 40, never `band_defaults(n_modes)`); for $f_1 = 0$ as PR 14b.
- Returns the band modes and `info`:
  - **The filter.** Clusters reaching into $[\lambda_1, \lambda_2]$ are kept whole at both edges:
    λ ∈ [λ₁(1 − 100t), λ₂(1 + 100t)], extended over chains with relative gaps < 100t. t is the
    per-mode threshold for LOBPCG modes and r_acc for slice modes.
  - **`strategy`:** "lobpcg", "shift_invert" or "hybrid".
  - **`choice`:** "forced", "lobpcg" (reached λ₂), "capacity" (LOBPCG stopped at $C_{max}$,
    slices above), "memory" (chunk 0 does not fit).
  - **Also:** `stagnation_risk`, `budget`, `from_zero`, `lambda1_ub`, `junction`.
  - **`count`:** PR 14b's, or si_band's summary, or after a switch `{"lobpcg": ...,
    "slices": ...}`.
  - **With slices:** si_band's `slices`, `capacity`, `merged_edges`, `collapsed`,
    `edge_rr_failed`, `duplicates`, `r_acc`.

### Up-front checks (before the σ pre-estimate and any build)

- **Spectrum bound** $\hat\lambda_\max = \max_e \lambda_\max(K_e, M_e)$.
  - **Why it bounds the spectrum:** for any $x$, $x^\top K x = \sum_e x_e^\top K_e x_e \le
    \max_e \lambda_e \sum_e x_e^\top M_e x_e \le \hat\lambda_\max\, x^\top M x$. Nodal masses,
    masking and rigid deflation only lower the Rayleigh quotient; ρ > 0 is enforced and lumped
    masses are positive.
  - **One tier, exact:** batched Cholesky and `eigvalsh`.
    - Voxel meshes: one reference element per phase on the host. With per-cell scales the bound
      is $\lambda_\text{phase} \max_e s_{K,e}/s_{M,e}$.
    - Unstructured meshes: `element_matrices` (PR 12) in device chunks, about 200 µs per hex20
      element: about 16 s at 1M DOF, a few minutes at 10M, small next to a band solve.
    - Recorded in `info["time"]["bound"]`.
  - **Singular $M_e$** (a massless phase, $s_M = 0$): $\hat\lambda_\max = \infty$, the check never
    raises.
  - **Outcome:** $\lambda_2 > \hat\lambda_\max$ raises `ValueError`. An edge in
    $(\lambda_\max, \hat\lambda_\max]$ still reaches si_band's capacity error or PR 14b's "more
    modes than the mesh provides".
- **$f_1 < \lambda_1$** applies to the paths that start on shift-invert: "shift_invert", and
  "memory".
  - $\lambda_1^{ub}$ is the σ pre-estimate, a Rayleigh quotient and so an upper bound. With a
    `sigma=` hint or Jacobi, `sigma_estimate` (10 iterations) gives it instead. It is recorded in
    `info["lambda1_ub"]`.
  - If $\lambda_1 < \lambda_1^{ub}(1 + 10^{-2})$, si_band runs on $[0, \lambda_2]$ and the
    result is filtered; `info["from_zero"]` is True. The 1% margin keeps λ₁ out of si_band's
    widened first-slice window.
  - "auto" with LOBPCG needs no rule: LOBPCG runs from 0.

### Strategy ("auto")

1. **Chunk 0 fits?** If `lobpcg_memory` for chunk 0 (chunk 40, its block and initial basis)
   exceeds the budget, use shift-invert on the whole band ("memory", the $f_1 < \lambda_1$ rule).
2. **LOBPCG** `_solve(..., lam_stop=λ2, max_capacity=C_max)`:
   - The basis grows as `new = min(max(1.5 old, need), C_max)`; the initial allocation is
     `min(up(n_modes + k, c), C_max)`.
   - The host V is preallocated once at $C_{max} + k$ columns (`np.empty`: unwritten pages cost
     nothing), so there is no concatenate transient.
   - The solve stops when the next lock needs more than $C_{max}$, with `info["stopped"] =
     "capacity"`, and returns its locked modes.
   - Band end copy: only the band columns are copied, not a full contiguous copy that is then
     filtered.
   - Reaching $\lambda_2$: PR 14b's count at the cut above $\lambda_2$, then the filter
     ("lobpcg").
3. **Capacity stop** ("capacity", strategy "hybrid"). LOBPCG returns; its basis, state and
   preconditioner are freed. Let $\lambda_j$ be the last locked mode.
   - **$\lambda_1 > \lambda_j$:** the locked modes are freed; si_band runs on
     $[\lambda_1, \lambda_2]$ with no junction.
   - **Otherwise:** si_band runs on $[\lambda_j, \lambda_2]$. Its first slice re-finds λ_j's chain
     (inside) and its lower neighbour ν, which si_band now exposes as `info["lower"]`: ν, the
     lowest inside value a, and the counting edge $c = (\nu + a)/2$.
     - **Junction** at c: the locked modes below c are kept and si_band's modes at or above c.
       A si_band mode below c, or ν itself, that matches no locked mode within
       100·max(t, r_acc) relative is a mode LOBPCG missed; it is kept if it lies in the band.
       A matched one is a duplicate and dropped.
     - **Count of [0, c]:** PR 14b's count after si_band, with lam_last = ν, lam_next = a, and
       W = the locked modes up to a's match plus any missed pair. It abuts si_band's own lower
       count at c, so no hole and no overlap.
     - The locked modes stay on the host until that count, and the slice capacity's host term
       holds them. With `verify=False` they are freed before si_band, except those in the band.
   - **Stagnation:** if the switch depth (j + n_rigid) is ≥ 1000, a warning is raised and
     `stagnation_risk` is set to True. The 15a measurement: a MINRES solve near depth 1005 is
     marginal at τ = 1e-8 on hex20/AMG.

There is no time-based choice: LOBPCG is cheaper per mode at every measured depth (below), and
computing modes below the band costs less than one slice mode per about 200 LOBPCG modes.

### Cost model (measured and recorded, not a run-time decision)

- **MINRES iterations per solve:** $\alpha D(1 + D/D_0)$, fitted on the frac-0.5 rows of PR 15a's
  table (floor rows excluded): hex20/AMG about α = 5.8, $D_0$ = 650, ratios 0.87–1.20 over
  depths 6–1005; voxel/GMG (depths ≤ 201) α = 4.35, $D_0$ about 760, weakly constrained. The
  roadmap's "∝ ln(1/τ)/δ plus an outlier term" is replaced by this measured form (15a: depth, not
  δ, drives the iterations).
- **LOBPCG per chunk:** iterations rising with depth (hex20 $12^3$: 18 → 37 over 7 chunks;
  voxel $24^3$: 17 → 34), $t_\text{warm}$ flat at these sizes.
- **Per-mode ratio:** a slice mode (≈ 1.8 columns plus 84 per slice, steps × MINRES
  iterations × $t_\text{minres}$) against a LOBPCG mode. Expected about 25–110×, so no
  crossover. The roadmap's "≲ 1200" sentence is corrected from the benchmark.

### Memory

- **The split:** both estimators gain `device` and `host` entries.
  - `lobpcg_memory`: host = $8n(\text{n\_modes} + k\cdot\text{band})$ (V);
    device = total − host (its start, seed and slack blocks stay on the device).
  - `si_memory`: host = $8n\,\text{n\_deflate} + 8C^2 + 8n(\text{n\_modes} + 2)$; device = the
    rest (the end phase holds one device chunk, $8n\,c$).
  - Unified memory (CPU): device + host ≤ budget. GPU: device ≤ budget and host ≤ host budget.
- **$C_{max}$:** the largest multiple of LOBPCG's c whose estimate fits the budget, with the
  growth composed by the max form. During a growth the step temporaries are freed and only the
  state (about 7 blocks) and the old basis are alive, so the device term is basis_new +
  max(step, $8n\,C_\text{old}$ + state), $C_\text{old} = C/1.5$. The host term is V at
  $C + k$ plus the band end copy, $8n\hat N_\text{band}$.
- **Slice capacity $C$:** the minimum of three caps.
  - **The budget:** the largest multiple of c with `si_memory(op_memory, n, C − W, n_modes=⌈1.25
    m_s⌉, block, pc, n_rigid, n_deflate=W)` fitting, with W = 2c, so the device basis is exactly C.
    The host term adds the kept locked modes, the store $8n\hat N_\text{band}$ and $2 m_s$ columns
    (a span-rerun window of up to $m_s$ and si_checked's [W | V] restart copy).
    $m_s = \max(1, (C - W - 84)/1.8)$.
  - **The need:** up(1.25 (1.8 $\hat N_\text{band}$ + 84) + W, c).
  - **The free space:** up(n_free − n_rigid, c). More columns are round-off noise.
  - **$\hat N_\text{band}$:** `n_modes` when given. Otherwise the $a = 3/2$ Weyl bound through
    the highest known mode: after a capacity stop $(j + n_r)(\lambda_2/\lambda_j)^{3/2} - j$;
    on the shift-invert paths $(\lambda_2/\lambda_1^{ub})^{3/2}$. This is an overestimate (fits
    on chunk modes underestimate by about 16%), which is safe for sizing.
  - **Limits:** below 96 columns (PR 15c's measured slice minimum at b = 6) a warning; below
    2b + c a `ValueError` naming the budget. `slice_options["capacity"]` overrides all three
    caps.
  - **Not modelled:** the merge's 4–5 copies per merged-window column in the collapse case, a
    known excess documented in the docs.
- **Probe blocks:** the counts take min(`band_probe_block(...)`, the budget's remainder).
- **`solve_modes_memory(mesh, bcs, n_modes, band=(f1, f2), strategy="lobpcg", memory_budget=None,
  capacity=None)`:** the default strategy keeps PR 14b's return shape, the top-level `total`.
  - "shift_invert" returns the slice composition, with `capacity` from the budget when not given.
  - "auto" returns the LOBPCG estimate at min(the band's basis, $C_{max}$) plus, under
    `"shift_invert"`, the slice estimate, and `total` = their max.

### `si_band(verify=None)`

`verify` ∈ {True, False, None}. None applies PR 14b's rule per leaf:
- skip above `MAX_DEGREE`;
- skip when the a-priori bound $2\,n_\text{blocks}\,d\,(b_p/b)\,t_\text{minres}$ exceeds the
  leaf's time; the factor is 0.3 instead of 2 when τ_c is looser than the rule, as `_bound`;
- otherwise calibrate (PR 14b's `_calibrate` with the leaf's count closure and preconditioner),
  and count iff the prediction ≤ the leaf's time.

A run leaf's time is its slice's; a span's is its parent run's. $t_\text{minres}$ is the leaf's
slice's measured seconds per MINRES iteration. `_bound` and `_calibrate` move to `counting.py`,
and the tests patch them there. Every leaf records `predicted_s`, `predicted_kind` and `reason`.

### Errors

si_band's `_cap_error` is worded per case (PR 15c open item):
- "lo may be below lambda_1" only when no lower side closed and lo > 0;
- "hi above the spectrum" only when no upper side closed;
- the lo = 0 case names only hi.

## Changes to existing code

- `lobpcg._solve`: `max_capacity` (growth rule, initial allocation, preallocated V, the stop),
  and the band end copy of the band columns only.
- `si_band`:
  - `verify=None`;
  - `info["lower"]` (ν, a, c of the first slice's lower edge);
  - `_cap_error` wording.
- `counting.py`: `_bound`, `_calibrate` moved from `api.py`.
- `api.solve_modes`:
  - `band=(f1, f2)`, `strategy`, `memory_budget`, `slice_options`;
  - the checks, the capacity stop and the junction;
  - `check_band` for $f_1 > 0$; the NotImplementedError removed.
- `memory.py`:
  - `device`/`host` entries;
  - $C_{max}$ and the slice capacity searches;
  - the default budget;
  - `solve_modes_memory(band=, strategy=, memory_budget=, capacity=)`.
- `eigen/api.py`: `spectrum_bound(mesh, materials, ...)` (one function).

## Benchmark (`benchmarks/strategy_study.py`, one invocation, ≤ 5 min, progress to a log)

- [15a, 15d] constants: refit α, $D_0$ per family on `shift_invert_study.json` (no runs, seconds).
- [15d] the per-mode ratio:
  - one LOBPCG band run to about 280 modes per mesh (hex20 $12^3$ AMG, about 31 s; voxel
    $24^3$ GMG, about 38 s), recording per-chunk iterations and $t_\text{warm}$;
  - the slice cost from the recorded si_band and slice rows of PR 15a/15c (no new slices);
  - predicted against measured on the held-out "full" slice row (ratio recorded, about 1.4×).
- Recorded in `benchmarks/results/strategy_study.json` with the git commit: the constants, the
  ratio per depth, the crossover ("none" expected), and the 10M estimates from
  `solve_modes_memory` (no run): $C_{max}$ with the default fp64 options and with fp32 basis plus
  low memory, against the roadmap's 700–1000.
- Not run: the GMG depth-500 row and the si_band slice-model refit at tol 1e-6. They only refine
  a prediction that decides nothing (logged).

## Tests (`tests/test_strategy.py`; roadmap tags in the docstrings)

Method:
- Real counts run with the dense exact $(K + \sigma M)^{-1}$ preconditioner (15c's `exact`),
  monkeypatched into `api.amg`/`api.gmg` and si_band's `make`.
- Where only the wiring matters: `count_options={"degree": 4, "n_probes": 2}`.
- Chunks and bands are chosen so that every real count's cut gap is ≳ 0.3, asserted as a
  precondition.
- Each test is timed on the implementation and marked slow above about 20 s.
- Budgets are computed from the estimators, never hard-coded: the windows on "short" are about
  1 MB wide over a fixed 55 MB.
- Every test mapped to the `eigsh` item asserts the roadmap bound per mode with that mode's t
  (LOBPCG threshold or r_acc), using test_shift_invert's `check_band` logic generalised.
- Across the junction, M-orthogonality is checked with the pairwise residual bound.

1. **"lobpcg" reached** ("short", hex20 $6\times5\times4$): an interior band of about 10 modes
   inside chunk 0. `choice` "lobpcg", `from_zero` False, `info["chunk"] == 40` with `n_modes`
   given; matches `eigsh`.
2. **Hybrid switch** ("capacity"): `chunk=8`, a budget whose $C_{max}$ stops LOBPCG inside the
   band. Asserted:
   - the junction (`info["junction"]`: ν, a, c match locked modes);
   - in-band locked modes returned once, `duplicates` 0;
   - both counts ran and are < 0.5;
   - matches `eigsh`.
   - Variant λ₁ > λ_j (band above the stop): a spy on si_band's (lo, hi) = (λ₁, λ₂), no [0, c]
     count.
3. **"memory":** a budget below chunk 0's need but above one slice, with the precondition asserted
   from the estimators. si_band on the whole band, `from_zero` per the rule, matches `eigsh`.
4. **"shift_invert"** with 15c test 2(a)'s configuration (modes 5–28, block 6, its seed) and the
   budget set by the brute-force capacity reference to give `capacity == 96`:
   - ≥ 2 runs, every leaf counted and < 0.5;
   - the same set as `strategy="lobpcg"` on that band.
   - One real-AMG copy marked slow.
5. **An empty band:** a band strictly between two well-separated modes of the voxel "cube" in
   test_shift_invert (named gap, `from_zero` False asserted). 0 modes and count < 0.5 under both
   strategies.
6. **Clamped box**, `band=(0, f)`, "shift_invert": matches `eigsh` (no lower neighbour).
7. **$f_1 < \lambda_1$** on clamped and free meshes, "shift_invert": `from_zero` True, filtered,
   matches `eigsh`.
   - λ_lo = λ₁^ub(1 + 0.5e-2) gives True, (1 + 2e-2) gives False (λ₁^ub monkeypatched).
   - With an inflated λ₁^ub still correct.
   - The `sigma=` hint and Jacobi paths give λ₁^ub from `sigma_estimate`. The Jacobi case runs on
     the 240-DOF "free" hex8 mesh.
8. **Junction faults** (fake si_band): a missed mode (ν matches no locked mode) is kept and in
   W; a duplicate below c is dropped; a mode in the band but below c from LOBPCG is kept once.
9. **Lower-edge clusters:** the voxel cube (doubles and triples) with $f_1$ on a double and on
   the middle of a triple chain, both strategies, all copies returned. Plus a pure-function table
   for the filter: the exact edge, inside and outside 100t, a chain crossing $f_1$, the same at
   $f_2$.
10. **Spectrum bound:**
    - $\hat\lambda_\max \ge$ the dense largest eigenvalue on hex8, hex20, voxel, lumped and
      nodal-mass meshes, a graded `box_mesh` whose smallest element is not first, a two-phase
      voxel mesh whose stiffer phase is not phase 0, and a scaled voxel operator;
    - equal on a one-element mesh without BCs;
    - a massless phase gives ∞;
    - $\lambda_2 > \hat\lambda_\max$ raises before any build (spies on `_solve`, `si_band`, the
      preconditioner builders and `_precond_sigma`).
11. **Capacity and $C_{max}$** against brute-force loops over the estimators:
    - each binding cap: the need on "short" with the default budget, the free space on a tiny
      mesh, and an explicit budget;
    - the host term with ≥ 30 kept locked modes moving the capacity by a chunk;
    - unified and GPU (monkeypatched platform, `memory_stats`);
    - the 1.1 factor;
    - below 2b + c raises naming the budget, below 96 warns;
    - `solve_modes_memory` for "lobpcg" (unchanged shape), "shift_invert" and "auto".
12. **`_solve(max_capacity=...)`** directly: the capacity sequence (growth straight to the cap),
    the initial allocation capped, V preallocated (no concatenate), `stopped == "capacity"`,
    `info["capacity"] == C_max`.
13. **Cost-model constants:** the code's constants equal the refit of `shift_invert_study.json`'s
    frac-0.5 non-floor rows, with ratios in [0.8, 1.25]. The held-out row is within 2×.
14. **`si_band(verify=None)`:**
    - with `fake_band`: each branch (MAX_DEGREE skip; bound skip with factor 2 and with 0.3;
      calibrated and too dear, skip; calibrated and cheap, count; `_calibrate` spied), and a span
      leaf's time is its parent's;
    - one real run (exact preconditioner, a cheap leaf): per-leaf `reason`/`predicted_kind`,
      `ran` True, estimate < 0.5;
    - `solve_modes(strategy="shift_invert")` at the default verify records `predicted_s` and
      `reason` per leaf.
15. **Capacity error wording** (`fake_band`): lower side closed, upper side closed, lo = 0.
16. **Options:**
    - `strategy`/`memory_budget`/`slice_options` without `band`, bad values, unknown
      `slice_options` keys, `f1 >= f2`;
    - `strategy="lobpcg"` past the budget warns and runs;
    - the default budgets (monkeypatched `os.sysconf` and a fake `memory_stats` with
      `bytes_in_use`);
    - spies show `slice_options`, `tol`, `count_options`, `precond_options` reaching si_band
      and `make`;
    - `precond_dtype="float32"` on voxel with a switch: `make` builds fp64;
    - a switch at depth ≥ 1000 (monkeypatched depth) warns and flags `stagnation_risk`.

## Docs

- `docs/eigensolvers.md`:
  - "Interior bands and strategy": API, checks, the memory-driven rule with the measured cost
    model as evidence, the junction, memory and capacity, `verify=None` for slices, errors, the
    stagnation flag;
  - the Options block (`strategy`, `memory_budget`, `slice_options`);
  - the Validation list;
  - every `15d`/`PR 15d`/`NotImplementedError` mention in `docs/` and `README.md` rewritten,
    including eigensolvers.md lines near 103, 122, 589–714.
- `README.md`: the `band=(f1, f2)` example; status.
- `docs/ROADMAP.md`:
  - the Status paragraph, PR 15d status, PR 15 complete, the dependency table;
  - the strategy bullets: memory-driven, the "≲ 1200" sentence replaced by the measured ratio,
    the 10M figures from the recorded estimates;
  - a 15d note on $\alpha D(1 + D/D_0)$;
  - a shift-invert row in Memory budget;
  - δ_max ≤ 1 for slices from 0 (σ inside the slice);
  - PR 18 contents: the deferred items above;
  - deferred GPU checks: the default budget, measured peaks ≥ 2 GB of the new compositions, the
    per-mode ratio on GPU, stagnation near depth 1000 at 10M.
- `docs/open_items.md`:
  - close 14b's "no strategy selection or basis-fit check";
  - close the 15c → 15d items done here: capacity from the budget, the `verify` callable (now
    `verify=None`), the edge rules, and the counting edge factor (moot: no time choice);
  - the depth-1000 and σ-move items: flagged, remedies to PR 18;
  - retag the rest to PR 18;
  - the decisions below and the items not adopted.

## Decisions without user input (to log)

- **No time-based crossover:** the strategy is memory-driven. The fitted cost model and the
  measured LOBPCG chunks put a slice mode at 25–110× a LOBPCG mode at every measured depth
  (6–1005), so "auto" uses LOBPCG as far as the budget allows. This deviates from the roadmap's
  "cost model chooses" and "≲ 1200" (corrected). The GPU ratio is a deferred check.
- **The capacity stop switches** to slices above the last locked mode, joined at si_band's lower
  counting edge between converged pairs. LOBPCG's unconverged guard is not used. The locked
  modes are kept until the [0, c] count.
- **No re-prediction and no Weyl fit:** the store is sized by the $a = 3/2$ Weyl bound, an
  overestimate.
- **Default budget:**
  - GPU: the free pool;
  - CPU: 0.8 × RAM;
  - GPU host: 0.8 × RAM;
  - every search uses a 1.1 factor on the estimates.
- **Depth-1000 slices** are flagged with a warning; the remedies go to PR 18. The 10M criteria
  near index 1000 may need the user's input after the GPU checks.
- **"lobpcg"** never switches; it warns when it grows past the budget.
- **Slice capacity** is min(budget, need with a 25% margin, free space), with a warning below
  96 columns.
- **$f_1$ within 1% above $\lambda_1^{ub}$** is treated as below it.
- **`n_modes`** for an interior band is the expected band count.
- **The spectrum bound** uses the exact tier only.
- **`err.result`** is deferred to PR 18; the error messages are worded per case now.
- **The slice cost-model form:** $\alpha D(1 + D/D_0)$; τ scaling is not modelled.
- **Not run:** the GMG depth-500 row and the slice-model refit. They inform no decision.
- **Test tolerances:** slice tests run at tol 1e-6 with τ = 1e-8 (as 15c), instead of the
  roadmap's r_acc = max(1e-10, …) test setting, for run time. The eigsh bound is asserted per
  mode as the roadmap states.
