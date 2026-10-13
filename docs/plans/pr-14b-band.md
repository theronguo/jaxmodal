# PR 14b plan: the [0, f_max] band mode with `verify`

Branch `pr-14b-band` from `main` (PR 14a merged). Roadmap: PR 14, items tagged [14b], and the
"Frequency bands" section (strategy, [0, f_max], band edges, completeness count).

Revision 2 (plan review round 1): the band-edge-above-spectrum error; a free a-priori pre-check
before any calibration; calibration at the real τ and γ class, timing the recurrence alone;
`n_modes` as the band's expected count (capacity), growth only past it; memory released and the
budget reduced by the host W; the final chunk re-orthonormalized; new wiring and detection
tests.

Revision 3 (round 2): the bound uses chunk 0's warm per-iteration time with a factor 2 (7 PCG
iterations at the default τ), the calibrated decision includes the build and calibration
already spent, an explicit `degree` is never capped, the band `chunk` follows `n_modes`, a count
that does not fit the budget is handled, test placements at the solver's own thresholds, tests
of the bound, the calibration and `count`'s `_steps`, and a cheap ≥ 2 GB memory test.

Revision 4 (round 3): `verify=True` keeps the modes when the count does not fit; the bound is
tested against warm step time and on the loose-τ branch; `t_iteration` = 0 without a warm
iteration; the slack ratio is measured in the implementation; the kept aggregates leave the
budget; test 7b pinned to about 200 steps.

## Scope

- `solve_modes(mesh, materials, bcs, n_modes=None, ..., band=None, verify=None,
  count_options=None)`: `n_modes` alone returns the lowest modes (unchanged); `band=(0, f_max)`
  returns every elastic mode with $f \le f_{max}$ (plus whole clusters reaching into the band),
  and `n_modes` is then an optional expected count (the basis capacity). `n_modes` stays the
  4th positional argument, so `solve_modes(mesh, mats, bcs, band=(0, f))` works. `band=(f1, f2)`
  with `f1 > 0` raises `NotImplementedError` ("interior bands: PR 15").
- The [0, f_max] LOBPCG mode: PR 9b's chunks and locking, stopping once a chunk has a converged
  mode beyond the band edge.
- Edge placement and the count's cut, as in the roadmap's "Band edges", including "a band edge
  above the largest computed eigenvalue raises an error".
- `verify=None/True/False`, the predicted counting cost always in `info["count"]`, the probe
  block from the solve's own memory estimate, the fp64 multigrid (or Jacobi) at $\lambda_c$.
- `count_options` forwarded to `count` (every count option reachable, PR 13c rule).
- `solve_modes_memory(..., band=..., n_modes=k)`: the band's estimate at an expected count.

Not in 14b (PR 15): shift-invert slices, interior bands, the choice between LOBPCG and
shift-invert. With LOBPCG-from-0 as the only strategy, 14b's cost function is the counting-cost
model `verify=None` needs; there is no basis-fit check (a band whose basis outgrows the device
fails with an out-of-memory error instead of switching strategy; `solve_modes_memory(band=...,
n_modes=k)` is the user's check until PR 15). Both are open items.

## Design

### 1. Band mode in `lobpcg._solve`

New internal keyword `lam_stop=None` ($(2\pi f_{max})^2$, the pencil's units). With it:

- `n_modes` is the expected count (default `4 * chunk`); the capacity is
  $\lceil (n_{modes} + m)/c\rceil c$ as today. `chunk` defaults to
  `min(40, ceil(1.25 n_modes) + 2, n_free - n_rigid - guards(40))` when `n_modes` is given
  (one chunk then normally holds the band and a mode beyond it; one chunk is faster than
  several, PR 9b: 2.4 s vs 4.9 s for 30 modes), and to `min(40, n_free - n_rigid -
  guards(40))` without it (block size $m = 48$, the width the memory model is calibrated at).
  Above about 2M DOF the default block may not fit the device (`solve_modes_memory(band=...)`
  reports it; the Options table says so). `info["chunk"]` reports the chunk used.
- Chunk 0's warm time per iteration (its iterations after the first, which compiles) is
  recorded as `info["t_iteration"]` (one `perf_counter` pair around the first iteration and the
  chunk); chunk 0 has no locked modes, so it times the applies without the projector. If
  chunk 0 ran a single iteration there is no warm time: `t_iteration` = 0, so the bound never
  skips and `verify=None` goes to the calibration; the reported `predicted_s` is then None
  (`predicted_kind: "unknown"`; `predicted_steps` still gives the cost).
- Every chunk is lockable (the guard error applies to every chunk).
- **Band rule** (host helper `_band_count(lam, t, lam_stop) -> i`, unit-tested on synthetic
  arrays): `i` = the number of Ritz values $\le$ `lam_stop`$\,(1 + 100\,t)$ (a neighbour within
  $100\,r_{acc}$ of the edge reaches into the band), extended over clusters (relative gap
  $< 100\,t$, as `_extent`). After a chunk converges with `j` columns (already extended over a
  boundary cluster): if `i < j` the solve stops (modes $1..i$ of the chunk are in the band,
  $i+1..j$ are converged modes beyond it); otherwise ($i \ge j$: every converged mode is in the
  band, guards may be too) the chunk is locked and the next one starts.
- **Edge above the spectrum:** before a new chunk, if `count + m > n_free - n_rigid`,
  `ValueError("the band needs more modes than the mesh provides at block size m (f_max = ...
  Hz, count modes locked, n_free = ...)")`; this covers an edge above the largest eigenvalue
  (the roadmap's error) and a band holding nearly every mode.
- **Capacity:** when `count + m` exceeds the capacity, `B` grows to $\lceil 1.5\times\rceil$
  (rounded to `c`): new zero array, jitted copy, old freed; the host `V` (preallocated at the
  capacity) grows the same way. Growth happens only past the user's expected count; it has a
  transient (old + new) and recompiles the step for the new shape (documented, open item).
- **Final chunk:** its `j` converged columns are re-orthonormalized against the basis
  (`_reortho`, as at a lock) and their residuals recomputed, so the count's W sketch check holds.
- Returns all converged modes (locked + the final chunk's `j`); `info["n_band"]` = the number in
  the band (internal; `solve_modes` splits on it). `lobpcg`'s public signature is unchanged.
- A failed chunk ends the solve as today (`converged` False); no count runs then
  (`info["count"]["ran"]` False, reason "solve not converged").

### 2. `solve_modes(..., band=(0, f_max))`

- Validation: `band` a pair of finite floats with `0 == f1 < f2` (`f1 > 0`: NotImplementedError);
  `band` or `n_modes` required; `verify` in (None, True, False); `count_options` keys checked
  against `COUNT_OPTIONS = ("n_probes", "seed", "tau", "degree", "block", "maxit")` before any
  build and pinned to `count`'s signature by the options test.
- Returns `lam`, `V` of the band modes only (possibly 0 columns) and `info` with `band`,
  `frequencies`, `residuals` / `thresholds` of the band modes, `n_beyond` (converged modes
  beyond the edge used by the count) and `count`.
- Edges (roadmap): `lam_last` = the last returned mode, or the band edge $(2\pi f_{max})^2$ when
  none is returned; `lam_next` = the first converged mode beyond. The count deflates by every
  converged mode (returned and beyond: the tail model needs all modes up to `lam_next`) and R.
- The solve's device hierarchy is released (`P = None`) before the λ_c build, and `B` is local
  to `_solve` (freed on return): one hierarchy alive at a time. The AMG aggregates
  (`return_aggregates=True`, host arrays and the pattern) are kept for the λ_c build.

### 3. Counting cost model and `verify`

One helper `_predict(...)` (the test seam) returns the decision inputs.

- `d` from `count_degree(g, ...)` (`None` above `MAX_DEGREE`), or `count_options["degree"]`,
  which is always used as given and never capped (as in `count`; the Options table says so).
  Probe block `b = probe_block(budget, n_probes, op_memory, n_dof, voxel, precond_memory,
  n_rigid)` unless `count_options["block"]`, with `budget` = the solve's
  `solve_modes_memory(...)["total"]` at the returned capacity minus $8\,n\,k_{all}$ (the host W:
  host and device share memory on the Mac; conservative on a GPU) and minus the kept AMG
  aggregates' host bytes. The λ_c multigrid is fp64
  (`precond_memory` at fp64 even when the solve ran an fp32 V-cycle). `n_blocks` =
  $\lceil n_{probes}/b\rceil$; `predicted_steps` = $n_{blocks}\,d$, reported in every mode.
  `probe_block` raising (the budget below `count_memory(1)`, possible when the solve ran an
  fp32 V-cycle and the λ_c multigrid is fp64): `verify=None` skips with reason "count does not
  fit the solve's memory"; `verify=True` also returns the modes with `ran: False`, the reason
  naming both sizes and `count_options={"block": 1}` (the user may accept a larger peak), and a
  warning (the check can only run after the solve; raising would discard it).
- **A-priori bound (free):** a count step runs ≥ 7 PCG iterations at width b (measured 7–14 at
  λ_c at the default τ, 14a), each at least one operator and one preconditioner apply; a LOBPCG
  iteration is two operator and one preconditioner apply at width m plus its dense work (at
  most about 1.5× the applies at m = 48), so a step costs ≥ about 2.3 (b/m) $t_{it}$.
  `predicted_s_bound` = $c$ · `predicted_steps` · (b/m) · $t_{it}$ with $c = 2$, or $c = 0.3$
  (one PCG iteration) when `count_options["tau"]` is looser than the rule; $t_{it}$ =
  `info["t_iteration"]` (chunk 0, warm). Round-2 probes put real step / bound at 4.5–6.8 on
  hex20 16³ AMG, b = 10 (0.50 s probe vs 0.70–0.82 s in the 14a table, with a different load);
  the implementation measures the ratio with the recurrence-only `time["step"]` on hex20 and
  voxel/GMG, and the docs give those measured values. Dropping the build and calibration keeps it a lower bound.
- `verify is False`: no build, no count; `info["count"]` = `{ran: False, degree, block,
  predicted_steps, predicted_s: predicted_s_bound, predicted_kind: "bound"}`.
- `verify is True`: build the fp64 preconditioner at $\lambda_c$ and run the count (no
  calibration); `predicted_s` = the bound, the actual time in `count`'s `time`. A capped `d`
  gives `count`'s NaN, unverifiable.
- `verify is None`: if `d` is None or `predicted_s_bound` > the solve time (σ pre-estimate +
  precond + LOBPCG), skip with `predicted_kind: "bound"` (no build). Otherwise build at
  $\lambda_c$ (AMG reuses the solve's aggregates, PR 13b: exact for a σ change under θ = 0) and
  calibrate: two `count` calls at the real `d`, τ and block with an internal `_steps` limit
  (1 step, which compiles the real γ class, then 3 steps), `t_step` = the second call's
  recurrence time / 3 and `t_block` its sampling and projection time. Predicted seconds =
  `build_s` + `calibration_s` + $n_{blocks}(d\,t_{step} + t_{block})$ (`predicted_kind:
  "calibrated"`: the whole counting cost, spent and to come); the count runs iff that is
  $\le$ the solve time (the roadmap rule).
- `count` change: `time["step"]` becomes the `_recurrence` time alone per step (the projections
  and the reduction go to `time["block"]`), and the internal `_steps` keyword caps the steps run
  without changing γ's padded class. `_steps` is listed as internal in the docstring and is not
  in `COUNT_OPTIONS`. The 14a measured table stays valid: its $t_{step}$ at b = 10 included the
  per-block projections spread over d ≥ 16 steps, a share below 1/d (the docs say so).
- `info["count"]`: `count`'s info plus `ran`, `reason`, `predicted_steps`, `predicted_s`,
  `predicted_kind`, `solve_s`, `build_s`, `calibration_s`.

### 4. `solve_modes_memory(..., band=None)`

With `band`, `n_modes` is the expected count and `n_modes` and `chunk` default as in
`solve_modes(band=...)`; the estimate is the plain estimate at that capacity (no growth when the count is right).
The count fits by construction (budget = this estimate minus the host W).

## Tests (`tests/test_band.py`)

1. **Band rule unit tests:** `_band_count` on synthetic λ, t: exact edge, neighbour inside
   $100\,t$, neighbour just outside, a chain of small gaps crossing the window, none beyond,
   all beyond (`i = 0`).
2. **No missed modes vs the dense inertia** ([14b]): reference box hex20 $6\times5\times4$
   (clamped) and a free hex8 box (rigid modes deflated): $f_{max}$ in a clear gap, at
   $\lambda_k/(1 + 10\,t_k)$ (mode $k$ returned through the neighbour window) and at
   $\lambda_k/(1 + 1000\,t_k)$ (not returned), with $t_k$ from a first solve's `thresholds`,
   and one needing 3 chunks with `chunk=8`; each edge chosen from the dense spectrum with a
   margin ≥ 10× from every $100\,t$ threshold, including $\lambda_{k-1}$ (asserted). Returned
   count = `count_below(K, M, λ_edge)` (minus the rigid count on the free box) plus the
   predicted window/cluster extension; returned λ match dense `eigh` to $10\,\mathrm{tol}$
   relative; `len(info["residuals"])` = the returned count; every converged column (returned
   and beyond, as passed to the count) is M-orthonormal to $10^{-10}$ (final-chunk reortho).
3. **Count wiring** ([14b]): `counting.count` wrapped by monkeypatch in test 2's runs: `lam_last`
   = the last returned λ (or the band edge), `lam_next` = the first converged beyond, W has every
   converged column, R passed on the free box, the λ_c builder called at
   $(\lambda_{last} + \lambda_{next})/2$; `info["count"]["lam_c"]` equals the dense value to
   $10\,\mathrm{tol}$ relative and `gap` to $10\,\mathrm{tol}/g$ relative.
4. **Below the first mode** ([14b]): $f_{max}$ = half the first elastic frequency, clamped and
   free: 0 modes, `verify=True` count < 0.5 and verifiable.
5. **Detection** ([14b]): the wrapped `count` receives W with one in-band mode (not next to the
   cut) removed: `verify=True` gives estimate ≥ 0.5 at the default seed.
6. **Double eigenvalue** ([14b]): square-section beam, `lam_stop` at the pair's value: both
   copies returned (also with `chunk` ending between them), and `verify=True` gives < 0.5,
   verifiable.
7. **verify** ([14b]): `True` runs (no calibration); `False` neither builds nor counts and
   reports `predicted_steps` = $n_{blocks} d$ and the bound; `None` with `_predict` patched to
   both sides runs / skips, and skips without building when the bound exceeds the solve
   (builder patched to raise); `predicted_s` = `build_s + calibration_s` +
   $n_{blocks}(d\,t_{step} + t_{block})$ from the info values; a capped `d`
   (`counting.MAX_DEGREE` monkeypatched to 4, so the rule returns None): `None` skips without
   building, `True` returns NaN unverifiable; the bound equals $c$ · `predicted_steps` · (b/m)
   · `t_iteration` with $c = 2$ at the default τ and $c = 0.3$ with `count_options={"tau":
   1e-4}`; `t_iteration` = 0 forced (patched) makes `None` calibrate; a budget patched below
   `count_memory(1)`: `None` skips with the reason, `True` returns the modes with `ran: False`
   and warns.
7b. **Calibration accuracy and the bound** (slow): hex20 $12^3$, `_predict` patched to force
   the run, `count_options={"degree": 200, "block": 10, "n_probes": 10}` (one block, about 200
   steps, far more than the 3 calibration steps): the calibrated `predicted_s` minus the spent
   part is within $[0.5, 2]\times$ the actual count time, and the bound ≤ $n_{blocks}\,d\,
   t_{step}$ with the calibrated warm $t_{step}$.
7c. **`count` changes** (`tests/test_counting.py`): `_steps=3` runs exactly 3 steps;
   `_steps >= d` gives the same estimate as no limit; `time["step"]` and `time["block"]` exist
   and `step·steps + block ≤ total`.
8. **Growth:** `chunk=8`, `n_modes=8`, $f_{max}$ needing ≥ 4 chunks: the capacity grows (in
   `info`), and λ and V equal a full-capacity run with the same chunk and seed to $10^{-12}$; the
   same with `basis_dtype="float32"` (dtype kept); the count runs on the grown solve.
9. **Edge above the spectrum:** hex8 $2\times2\times2$ clamped with $f_{max}$ = 2× the largest
   dense frequency raises the ValueError; tiny-mesh chunk default works; `n_modes=5` gives
   `info["chunk"]` = 9.
10. **Probe block and options:** `info["count"]["block"]` = `probe_block(budget)` with
    `count_memory(b) <= budget < count_memory(b + 1)` (or b = n_probes); `count_options={"block":
    3}` overrides; unknown keys raise before any build.
11. **Errors:** neither `n_modes` nor `band`, `f1 > 0`, $f_{max} \le 0$, non-finite band, bad
    `verify`, bad `count_options`.
12. **Memory:** `solve_modes_memory(band=..., n_modes=k)` equals the plain estimate at the band
    capacity (exact); slow (macOS): the measured process peak of a band solve with
    `verify=True, count_options={"degree": 3}` (the peak is set by b, not d; keeps the run
    short) within 20% of the estimate at a size where it is ≥ 2 GB.
13. **Voxel:** test 2's clear-gap case on a voxel box with GMG.
14. **Example:** README `band` example (test_readme).

## Docs

- `docs/eigensolvers.md`: "Frequency band [0, f_max]" (mode, edges, the error, capacity and
  growth, verify and the cost model, info keys), the `solve_modes` signature, the Options table
  (`band`, `verify`, `count_options`, the band `chunk` default and its 2M DOF note), the memory
  section (`solve_modes_memory(band=...)`), `count`'s `time` keys, the validation list for
  `tests/test_band.py`.
- README: a band example; status line; the sentence on what eigensolvers.md covers.
- ROADMAP: status entry; mark the [14b] items done (the split note exists).
- `docs/open_items.md`: no strategy selection and no basis-fit check (PR 15); growth transient
  and recompiles, no `max_modes` cap (the free-DOF error and the expected count instead); band
  `chunk` default 40 and the 2M DOF limit; the a-priori bound's assumptions (factor 2, measured
  slack as measured); an explicit `degree` bypasses the cap; `verify=True` keeps the modes and
  warns when the count does not fit; budget = solve estimate minus host W and the kept
  aggregates; the final chunk converges modes beyond the band (up to `chunk`; a
  stop callback could save iterations); the τ_c question carried over from 14a.
