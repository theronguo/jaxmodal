# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands), the Memory budget section and
the PR 15c plan's split (15d = the cost model, the memory budget, the strategy choice,
`verify=None` for bands and `solve_modes(..., band=(f1, f2))`).

Revision 2 (plan review round 1: one decision after chunk 0 instead of a re-prediction per lock;
no seeding of si_band, a cut in LOBPCG's gap instead; memory split into device and host terms
with the growth transients; the iteration model $\alpha D(1 + D/D_0)$ over all depths;
stagnation priced; counting cost in both predictions; the slice time measured on the device;
`slice_options`; capacity capped by the need and the free space; `err.result` deferred).

## Scope

- `solve_modes(..., band=(f1, f2))` with $f_1 \ge 0$, plus `strategy` ("auto", "lobpcg",
  "shift_invert"), `memory_budget` (bytes) and `slice_options` (si_band's options).
- The strategy: LOBPCG from 0 (PR 14b's band mode) for one chunk, then one decision: continue
  LOBPCG (filtered to the band) or switch to shift-invert slices (`si_band`, PR 15c) above a cut
  in LOBPCG's last gap. A capacity stop of LOBPCG switches the same way.
- The cost model: LOBPCG time from chunk 0's measured iterations and time; slice time from a
  measured MINRES iteration on the device and the iterations against depth fitted on PR 15a's
  table (plus one new voxel row); the predicted counting cost on both sides; stagnation risk.
- The memory budget: the LOBPCG basis cap $C_{max}$ (growth transient included), the slice
  capacity from `si_memory`, the host store; one check on unified memory (CPU, the Mac), a
  device and a host check on GPU.
- The $f_1 < \lambda_1$ rule (solve from 0 and filter) and the up-front check of $\lambda_2$
  against an upper bound $\hat\lambda_\max$ of the spectrum.
- `si_band(verify=None)`: PR 14b's three-stage rule per leaf.
- `si_band`'s capacity errors worded per case (PR 15c open item).
- `solve_modes_memory(..., band=(f1, f2), strategy=...)`, the README example, the docs.

Out of scope (deferred, listed in `docs/open_items.md`):
- PR 15c's "Late capacity errors and slice efficiency" items (verification slices,
  preconditioner reuse on a whole requeue, per-step residuals, merge copies, host transposes,
  `verify=False` input work, stopping si_checked before its count on a window rerun) and
  `err.result` (the pairs found before a capacity error) → PR 18.
- The depth-1000 remedies (deflating converged modes from MINRES, a σ-aware multigrid, a lean
  LOBPCG fallback) → PR 18 and the deferred GPU checks; 15d prices and flags the risk only.
- A memory-mapped host store for bands beyond host RAM → PR 18.

## Roadmap tests tagged [15d]

| Roadmap test | 15d test |
|---|---|
| [15a, 15c, 15d] matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 1, 2, 7 |
| [15b, 15c, 15d] interior band between two well-separated modes: 0 modes, count < 0.5 (the planted-mode half is 15b/15c's) | 3 |
| [15a, 15c, 15d] clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` | 4 |
| [15d] f₁ below λ₁ solved as [0, f₂], filtered, matches `eigsh` | 5 |
| [15d] memory budget too small for the LOBPCG-from-0 basis → shift-invert | 6, 7 |
| [15a, 15d] MINRES iterations vs δ measured (15a), cost-model constants fitted | 11, 12 |
| PR 15 contents: capacity per slice from the memory model | 9 |
| Band edges: an edge above the largest computed eigenvalue raises | 8 (up front); 15c tests 3, 11 (late) |

## Design

### API

```python
lam, V, info = solve_modes(mesh, materials, bcs, band=(f1, f2))           # auto
solve_modes(..., band=(f1, f2), strategy="shift_invert", memory_budget=8e9,
            slice_options=dict(tau=1e-8, block=6))
```

- `band=(f1, f2)`: $0 \le f_1 < f_2$ finite (Hz); `check_band` accepts $f_1 > 0$. $f_1 = 0$ runs
  the same code path as $f_1 > 0$ (the filter keeps everything from 0).
- `strategy`: "auto" (default), "lobpcg" (PR 14b's band mode from 0, filtered; never switches;
  over the budget it warns once up front and runs) or "shift_invert" (`si_band` on the band, no
  LOBPCG). Without `band`, any `strategy` but "auto", `memory_budget` or `slice_options` raises.
- `memory_budget`: bytes. Default: the device's `bytes_limit` on GPU (the usable pool under
  `XLA_PYTHON_CLIENT_MEM_FRACTION`, the roadmap's Memory budget), 0.8 × physical RAM on CPU
  (`os.sysconf`; unified memory on the Mac; consistent with the 52 GB benchmark guard on 64 GB).
  The host budget on GPU is 0.8 × physical RAM. Both are recorded in `info["budget"]`.
- `slice_options`: dict, keys `tau`, `maxit_inner`, `block`, `max_slices`, `capacity` (overrides
  the budget sizing), `edges` (validated as si_band does); `seed` comes from `solve_modes`'s.
- `n_modes`: for an interior band, the expected modes in the band (sizes the slice store; the
  LOBPCG chunk keeps its default 40, not `band_defaults(n_modes)`); for $f_1 = 0$ as PR 14b.
- Returns the band modes (clusters reaching into $[\lambda_1, \lambda_2]$ kept whole at both
  edges: λ ∈ [λ₁(1 − 100t), λ₂(1 + 100t)], extended over chains with relative gaps < 100t; t the
  per-mode threshold for LOBPCG modes, r_acc for slice modes) and `info` with:
  - `strategy` (used), `choice` (dict: `reason` "forced", "covered", "memory", "cost"; the
    predictor's inputs and outputs, below; `stagnation_risk`), `budget`, `from_zero`, `cut`;
  - `count`: PR 14b's (LOBPCG only), or `{"lobpcg": ..., "slices": ...}` after a switch;
  - after a switch or for "shift_invert": si_band's `slices`, `capacity`, `merged_edges`,
    `collapsed`, `edge_rr_failed`, `duplicates`, `r_acc`.

### Up-front checks

- **Spectrum bound** $\hat\lambda_\max = \max_e \lambda_\max(K_e, M_e)$: for any $x$,
  $x^\top K x = \sum_e x_e^\top K_e x_e \le \max_e \lambda_e \sum_e x_e^\top M_e x_e \le
  \hat\lambda_\max\, x^\top M x$ (nodal masses, masking and rigid deflation only lower the
  Rayleigh quotient; ρ > 0 is enforced; lumped masses are positive). Two tiers:
  - cheap: $\mu_e = \|L_e^{-1} K_e L_e^{-\top}\|_\infty \ge \lambda_e \ge \mu_e/\sqrt{n_e}$
    ($M_e = L_e L_e^\top$, $n_e$ the element DOFs). $\lambda_2 > \max_e \mu_e$ raises;
    $\lambda_2 \le \max_e \mu_e/\sqrt{n_e}$ passes;
  - exact (only between the two): batched `eigvalsh`; $\lambda_2 > \hat\lambda_\max$ raises.
  - Voxel meshes: one reference element per phase (host); with per-cell scales
    $\lambda_\text{phase} \max_e s_{K,e}/s_{M,e}$. Unstructured: `element_matrices` (PR 12) in
    device chunks. A singular $M_e$ (a massless phase, $s_M = 0$) gives $\hat\lambda_\max = \infty$:
    the check never raises. Measured cost recorded in `info["time"]["bound"]` (cheap tier about
    2.4 µs per hex20 element, the exact one about 200 µs).
  - Runs before the preconditioner build and the σ pre-estimate. An edge in
    $(\lambda_\max, \hat\lambda_\max]$ still reaches si_band's capacity error or PR 14b's "more
    modes than the mesh provides".
- **$f_1 < \lambda_1$** (paths that start on shift-invert: "shift_invert", or "auto" forced
  there before chunk 0): $\lambda_1^{ub}$ = the σ pre-estimate (a Rayleigh quotient, an upper
  bound); with a `sigma=` hint or Jacobi, `sigma_estimate` (10 iterations) gives it. If
  $\lambda_1 < \lambda_1^{ub}(1 + 10^{-2})$ (the margin keeps λ₁ out of si_band's widened
  first-slice window), si_band runs on $[0, \lambda_2]$ and the result is filtered;
  `info["from_zero"]` True. Under "auto" after chunk 0 the cut (below) covers it: LOBPCG already
  ran from 0.

### Strategy ("auto")

1. **Chunk 0 fits?** $C_0$ = the chunk-0 basis (`band_defaults` with chunk 40). If its
   `lobpcg_memory` exceeds the budget, the chunk is halved until it fits (floor 10); below the
   floor, shift-invert directly (`reason` "memory", $\hat N$ from the a-priori Weyl law
   $a = 3/2$ through $\lambda_1^{ub}$).
2. **Chunk 0.** `lobpcg._solve(..., lam_stop=λ2, max_capacity=C_max, decide=fn)`: two new
   arguments. `decide(lam, info)` is called once, after chunk 0's lock, and returns True to stop;
   `max_capacity` stops the solve when the next growth would pass it (`info["stopped"] =
   "capacity"`). Both return the locked modes and the gap above them (below). Chunk 0 past
   $\lambda_2$: no decision, `reason` "covered".
3. **The decision** (after chunk 0, $N$ locked elastic modes; a pure function of the recorded
   inputs, so tests recompute it):
   - $\hat N(\lambda) = c\,\lambda^a$: least squares of $\log k$ on $\log\lambda_k$ over the
     locked modes $k \ge \lceil N/2\rceil$ (the low modes deviate from Weyl), $a$ clipped to
     $[0.3, 1.5]$ (plates about 0.4, bulky 1.5); rigid modes not counted.
   - Memory: $\hat N(\lambda_2) + k + $ guards $> C_{max}$ → shift-invert ("memory").
   - Else cost: shift-invert iff $T_\text{si} < T_\text{lobpcg}$ (strict), "cost".
   - Stagnation: if a slice's depth $D$ (below) reaches $D_\text{stag} = 1000$ (hex20/AMG at
     τ = 1e-8, the recheck_1005 row; GMG takes the same until measured), LOBPCG is kept whenever
     it fits, whatever the times; when memory forces slices, a warning and
     `choice["stagnation_risk"]` True.
   - If the minimum slice (96 columns) does not fit the budget, LOBPCG continues regardless; its
     capacity stop then raises naming the budget.
4. **Switch** (the decision or the capacity stop). LOBPCG returns; its basis, state and
   preconditioner are freed. The cut is the middle of the gap above the last locked mode (to the
   next converged Ritz value; `_extent` guarantees ≥ 100t).
   - λ₁ ≥ cut: the locked modes are dropped; si_band on $[\lambda_1, \lambda_2]$.
   - λ₁ < cut: PR 14b's `_verify` at the cut counts $[0, \text{cut}]$ on all locked modes (the
     count a no-switch solve stopping there would run); then the locked modes outside
     $[\lambda_1, \text{cut}]$ are freed; si_band on $[\text{cut}, \lambda_2]$ (its first slice
     finds its own lower neighbour, the last locked mode again; si_band's modes below the cut are
     dropped, LOBPCG's above it never exist).
   - A si_band mode within 100 r_acc of the cut is a duplicate candidate: `info["cut_close"]`
     flags it (the gap is ≥ 100t; with r_acc ≤ t it cannot happen).
5. **No switch:** PR 14b's band mode to $\lambda_2$, its count at the cut above $\lambda_2$, the
   result filtered to the band.

`strategy="shift_invert"` skips 1–4: si_band on the band with $\hat N_\text{band}$ =
`n_modes` or the a-priori Weyl estimate.

### Cost model

Inputs after chunk 0: its iterations $i_0$ and warm iteration time $t_0$ (`t_iteration`), its
block $k$ and chunk, $\hat N$, and $t_\text{minres}$: 5 warm MINRES iterations (after one
compiling call) at width b on `ap_operator(op, free, -σ_mid)` with chunk 0's preconditioner
(AMG aggregates do not depend on σ under θ = 0; GMG's hierarchy has the same shapes), about
5–10 s at 1M DOF. Only the iteration constants come from the benchmark: they are
device-independent, so the GPU needs no refit, only the deferred check of predicted against
measured.

- $T_\text{lobpcg} = n_\text{ch}\, i_0\, t_0 (1 + \kappa \bar N / k) + T_\text{count}$:
  $n_\text{ch} = \lceil(\hat N(\lambda_2) - N)/\text{chunk}\rceil$ further chunks, $\bar N$ the
  middle of the remaining locked count (the projector's growth; κ from the benchmark). The
  count: PR 14b's degree and bound (`_bound`) at the predicted cut above $\lambda_2$, gap the
  Weyl mean gap $1/(a\hat N(\lambda_2))$.
- $T_\text{si}$: $N_b = \hat N(\lambda_2) - \hat N(\lambda_1)$ band modes; capacity $C$ (below)
  and window $W = 2c$, $C_\text{eff} = C - W$; $n_s = \lceil 1.8 N_b/(C_\text{eff} - 84)\rceil$
  slices, $1.8 N_b + 84 n_s$ columns (PR 15c's slice model; refit at tol 1e-6 in the benchmark),
  $(1.8 N_b/n_s + 84)/b$ steps per slice. Slice $i$ at depth
  $D_i = \hat N(\lambda_1) + (i + \tfrac12) N_b/n_s$ costs steps × $\alpha D_i(1 + D_i/D_0)$ ×
  $\ln(1/\tau)/\ln 10^8$ × $t_\text{minres}$; plus per leaf the band-pass count bound
  $2\,n_\text{blocks}\,d\,(b_p/b)\,t_\text{minres}$ with $d$ = `count_degree` at the Weyl mean gap
  (the counts' iterations per step taken equal to MINRES's), and the counting cost's lower-edge
  factor $(1 + a)/(2\sqrt a)$ (open item from 15b). The preconditioner build per slice and the
  Ritz work are under 1% (stated in the docs, not modelled).
- Counting cost under `verify`: True adds it; None adds min(count, solve) (counted only when not
  dearer than the solve); False adds 0.
- Iterations: $\alpha D(1 + D/D_0)$ per family, fitted on all depths of PR 15a's table (frac 0.5
  and 0.01 rows; hex20/AMG about α = 5.8, $D_0$ = 650, ratios 0.86–1.19 over depths 6–1005);
  GMG voxel: α on depths ≤ 201 plus one new row near depth 500 in the benchmark for $D_0$.
- `info["choice"]`: reason, $N$, $a$, $c$, $\hat N(\lambda_1)$, $\hat N(\lambda_2)$, $i_0$, $t_0$,
  $t_\text{minres}$, κ, α, $D_0$, $n_s$, steps, $D_i$, $C$, $C_{max}$, the predicted seconds of
  both strategies split into solve and count, `stagnation_risk`.

### Memory

Unified memory (CPU): device + host ≤ budget. GPU: device terms ≤ budget, host terms ≤ the host
budget.

- **$C_{max}$** (LOBPCG basis cap): the largest multiple of LOBPCG's $c$ with `lobpcg_memory(...,
  n_modes=C − k, band=True)` (its host term already holds V) + $8n\,C$ (the growth transient:
  old and new basis alive while copying; `_solve` grows straight to `max_capacity` when 1.5×
  would pass it, and only when old + new fits) + $8n\,\hat N_\text{band}$ (the end copy) ≤ budget.
  `_solve`'s band end copy takes the band columns only (no full contiguous copy followed by a
  filter).
- **Slice capacity** $C$ = min of
  - the budget: the largest multiple of $c$ with `si_memory(op_memory, n, C − W, n_modes=⌈1.25
    m_s⌉, block, pc, n_rigid, n_deflate=W)` (the device basis exactly $C$; W = 2c) on the device,
    and host $8n(\hat N_\text{band} + n_\text{kept} + 2 m_s)$ (store, kept locked modes, a
    span-rerun window of up to $m_s$ and si_checked's [W | V] restart copy),
    $m_s = \max(1, (C_\text{eff} - 84)/1.8)$;
  - the need: up(1.25(1.8 $\hat N_\text{band}$ + 84) + W, c);
  - the free space: up(n_free − n_rigid, c) (more columns are round-off noise).
  - Below 96 columns (PR 15c's measured slice minimum at b = 6) a warning; below 2b + c a
    `ValueError` naming the budget. `slice_options["capacity"]` overrides all three.
- `solve_modes_memory(mesh, bcs, n_modes, band=(f1, f2), strategy=..., memory_budget=None,
  capacity=None)` returns a dict for every strategy: "lobpcg" PR 14b's estimate, "shift_invert"
  the composition above (`capacity` from the budget when not given), "auto" both under
  `{"lobpcg": ..., "shift_invert": ...}`.

### `si_band(verify=None)`

`verify` ∈ {True, False, None}. None applies PR 14b's rule per leaf: skip above `MAX_DEGREE`;
skip when the a-priori bound $2\,n_\text{blocks}\,d\,(b_p/b)\,t_\text{minres}$ (0.3 instead of 2
when τ_c is looser than the rule, as `_bound`) exceeds the leaf's time; else calibrate (PR 14b's
`_calibrate` with the leaf's count closure and preconditioner) and count iff the prediction ≤
the leaf's time. A run leaf's time is its slice's; a span's its parent run's;
$t_\text{minres}$ is the leaf's slice's measured seconds per MINRES iteration. `_bound` and
`_calibrate` move to `counting.py`. Every leaf records `predicted_s`, `predicted_kind`, `reason`.

### Errors

si_band's `_cap_error` is worded per case (PR 15c open item): "lo may be below lambda_1" only
when no lower side closed and lo > 0; "hi above the spectrum" only when no upper side closed;
the lo = 0 case names only hi.

## Changes to existing code

- `lobpcg._solve`: `max_capacity`, `decide`, the gap above the last locked mode in `info`, the
  band end copy of the band columns only.
- `si_band`: `verify=None`; `_cap_error` wording.
- `counting.py`: `_bound`, `_calibrate` moved from `api.py`.
- `api.solve_modes`: `band=(f1, f2)`, `strategy`, `memory_budget`, `slice_options`, the checks,
  the decision, the switch; `check_band` for $f_1 > 0$; the NotImplementedError removed.
- `memory.py`: `solve_modes_memory(band=, strategy=, memory_budget=, capacity=)`, the
  capacity rule, the default budget.
- New `eigen/strategy.py`: the Weyl fit, the two predictions, the decision, the spectrum bound
  (pure functions).

## Benchmark (`benchmarks/strategy_study.py`, ≤ 1 h, progress per run to a log)

- Refit α, $D_0$ on `shift_invert_study.json` (no runs); one new voxel GMG MINRES row at depth
  about 500 ($32^3$, about 5 min) for GMG's $D_0$.
- Per mesh (hex20 $12^3$ AMG, voxel $24^3$ GMG): `solve_modes(band=(0, f))` for $N$ ≈ 40, 120,
  250 (κ); `si_band` with `verify=False` on two bands (depth ≈ 50 and 150, 10–20 modes: the
  slice model at tol 1e-6, predicted against measured). Counts are validated by PR 14b's
  calibration, not rerun. About 35 min in total.
- Held out: the "full" slice row of `shift_invert_study.json` (predicted against measured).
- Recorded: the predicted crossover depth per family on the Mac (checks the roadmap's "≲ 1200"),
  the git commit, `benchmarks/results/strategy_study.json`, cited in `docs/eigensolvers.md`.

## Tests (`tests/test_strategy.py`; fast unless marked; roadmap tags in the docstrings)

Tolerances: LOBPCG tol 1e-6 (eigenvalues to about $r^2$, `check_band`'s 10·TOL); slices
`tau=1e-8`, tol 1e-6 (as 15c); `verify=True` and `count["ran"]` asserted wherever a count is
asserted.

1. Auto, "covered": hex20 $6\times5\times4$ ("short"), a band of about 10 modes inside chunk 0:
   reason "covered", matches `eigsh`.
2. Auto, "cost" keeps LOBPCG: a band past chunk 0 (`chunk=8`, modes about 30–45): LOBPCG
   continued, recorded $T_\text{lobpcg} < T_\text{si}$, and both recomputed from
   `info["choice"]` by an independent hand formula; matches `eigsh`.
3. Auto, "cost" switches (monkeypatched constants making slices cheaper): the cut, si_band on
   [cut, λ₂], locked in-band modes returned once, `duplicates` 0, matches `eigsh`, both counts
   ran < 0.5.
4. `strategy="shift_invert"` with 15c test 2(a)'s configuration (modes 5–28, block 6, its seed),
   budget set by the brute-force capacity reference to give `capacity == 96`: ≥ 2 runs, matches
   `eigsh`, every leaf counted < 0.5; and the same set as `strategy="lobpcg"` on that band.
5. A band strictly between two well-separated modes (`test_shift_invert`'s free cube, its
   largest gap): 0 modes, count < 0.5, both strategies.
6. Clamped box, `band=(0, f)`, "shift_invert": matches `eigsh` (no lower neighbour).
7. $f_1$ below $\lambda_1$, clamped and free, "shift_invert": `from_zero` True, filtered, matches
   `eigsh`; with a monkeypatched inflated $\lambda_1^{ub}$ ($\lambda_1 < \lambda_{lo} <
   \lambda_1^{ub}$) still correct; with `precond="jacobi"` and with a `sigma=` hint; $f_1$
   exactly at the computed $f_1$ and just above (the margin); `from_zero` False in test 1.
8. Memory switch: `chunk=8`, a budget admitting at least one chunk more than chunk 0 while
   $\hat N(\lambda_2) > C_{max}$ (preconditions asserted: LOBPCG total > budget ≥ minimum slice):
   exactly one chunk ran, reason "memory", matches `eigsh`. A budget below chunk 0's own need
   (above one slice): shift-invert directly, "memory".
9. Capacity stop: a budget whose $C_{max}$ stops LOBPCG inside the band (decision forced to
   LOBPCG by a monkeypatched predictor): the switch at the cut, in-band locked modes counted by
   the [0, cut] count and returned once, matches `eigsh`.
10. Lower-edge clusters: the voxel cube (doubles and triples) with $f_1$ on a double and on the
    middle of a triple chain, both strategies: all copies returned; plus a pure-function table
    for the lower-edge filter (exact edge, inside and outside 100t, a chain crossing $f_1$).
11. Pure functions (`strategy.py`), table-driven: the Weyl fit (synthetic $\lambda_k =
    (k/c)^{1/a}$ recovers a = 1; a = 3 clipped to 1.5, a = 0.1 to 0.3; repeated values and rigid
    modes); $n_\text{ch}$; $n_s$, columns, steps, depths on two inputs; the switch at exactly
    $C_{max}$ and at $T_\text{si} = T_\text{lobpcg}$; stagnation (depth ≥ 1000 keeps LOBPCG if
    it fits, else the flag and warning).
12. Cost-model constants: the fit reproduces `shift_invert_study.json` (ratios within
    [0.8, 1.25] on the fitted rows); the held-out row within 2× (regression, recorded commit).
13. Spectrum bound: $\hat\lambda_\max \ge$ the dense largest eigenvalue on hex8/hex20/voxel/
    lumped/nodal-mass meshes, a graded `box_mesh` whose smallest element is not first, a
    two-phase voxel mesh whose stiffer phase is not phase 0 and a scaled voxel operator; equal on
    a one-element mesh without BCs; a massless phase gives ∞; both tiers exercised;
    $\lambda_2 > \hat\lambda_\max$ raises before any build (spies on `_solve`, `si_band`, the
    preconditioner builders and `_precond_sigma`).
14. Capacity, each binding case against a brute-force loop over `si_memory`: the need on "short"
    with the default budget (capacity < 300), the free space on a tiny mesh, an explicit budget;
    the host term with $n_\text{kept} \ge 30$ moving the capacity by a chunk; unified and GPU
    (monkeypatched platform); $C_{max}$ against a brute-force `lobpcg_memory` loop; below 2b + c
    raises naming the budget, below 96 warns.
15. `si_band(verify=None)` with `fake_band`: each branch (MAX_DEGREE skip; bound skip;
    calibrated and too dear, skip; calibrated and cheap, count; `_calibrate` spied), a span
    leaf's time is its parent's, per-leaf `predicted_s`/`reason`.
16. Capacity error wording (`fake_band`): each branch (lower side closed, upper side closed,
    lo = 0).
17. Options: `strategy`/`memory_budget`/`slice_options` without `band`, bad values, unknown
    `slice_options` keys, `f1 >= f2`, `strategy="lobpcg"` over the budget (warns, runs), the
    default budget (monkeypatched `os.sysconf` and a fake `memory_stats`),
    `slice_options` reaching si_band (spy).
18. `solve_modes_memory(band=..., strategy="auto")` equals the separate "lobpcg" and
    "shift_invert" estimates.

## Docs

- `docs/eigensolvers.md`: "Interior bands and strategy" (API, checks, the decision and cost
  model with the fitted constants, memory and capacity, `verify=None` for slices, errors, the
  stagnation flag); the Validation list; the stale "15d" sentences (the [0, f_max] section's
  "no strategy selection yet", the slicing section's "15d will ..." lines) rewritten.
- `README.md`: the `band=(f1, f2)` example; status.
- `docs/ROADMAP.md`: PR 15d status and PR 15 complete; a 15d note on the fitted
  $\alpha D(1 + D/D_0)$ form and the τ scaling; the crossover (≲ 1200) and 10M figures restated
  from the benchmark; a shift-invert row in the Memory budget section; δ_max ≤ 1 for slices from
  0 (si_band's slices from 0 keep δ ≤ 1 by construction: σ inside); the dependency table status;
  deferred GPU checks (default budget = the pool, predicted against measured times, stagnation
  near depth 1000 at 10M).
- `docs/open_items.md`: close 14b's "no strategy selection or basis-fit check", the 15c → 15d
  items done here (capacity from the budget, `verify` callable → `verify=None`, the edge rules,
  the counting edge factor, the depth-1000 and σ-move items: priced and flagged, remedies to
  PR 18); retag the rest to PR 18; the decisions below and the items not adopted.

## Decisions without user input (to log)

- One decision after chunk 0 (Weyl fit on its modes), not a re-prediction per lock; a capacity
  stop switches the same way. Chunk 0 is the price of the prediction (seconds at 1M DOF).
- A switch cuts in LOBPCG's gap above the last locked mode instead of seeding si_band with the
  locked modes: locked in-band modes are returned (verified by the [0, cut] count), si_band runs
  above the cut, no known-pair store and no host copy of the out-of-band locked modes.
- The default budget is the whole usable pool on GPU (the roadmap's estimates assume it) and
  0.8 × RAM on CPU; the GPU host budget 0.8 × RAM.
- Depth-1000 slices: priced (the $D_0$ term), LOBPCG preferred there when it fits, else a
  warning and `stagnation_risk`; the remedies go to PR 18 (the 10M criteria near index 1000 may
  need the user's input after the GPU checks).
- The Weyl exponent is clipped to [0.3, 1.5], fitted on the upper half of chunk 0's modes; κ and
  the iteration constants come from the benchmark; $t_0$ and $t_\text{minres}$ are measured per
  solve.
- "lobpcg" never switches; over the budget it warns and runs.
- If no slice fits, "auto" stays on LOBPCG and its capacity stop raises.
- Slice capacity: min(budget, need with a 25% margin, free space); a warning below 96 columns.
- $f_1$ within 1% above $\lambda_1^{ub}$ is treated as below it.
- `n_modes` for an interior band is the expected band count.
- `err.result` (the pairs before a capacity error) deferred to PR 18; the error messages are
  worded per case now.
- Test tolerances as stated above.
