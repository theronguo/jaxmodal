# PR 14b plan: the [0, f_max] band mode with `verify`

Branch `pr-14b-band` from `main` (PR 14a merged). Roadmap: PR 14, items tagged [14b], and the
"Frequency bands" section (strategy, [0, f_max], band edges, completeness count).

Revision 2 (plan review round 1): the band-edge-above-spectrum error; a free a-priori pre-check
before any calibration; calibration at the real τ and γ class, timing the recurrence alone;
`n_modes` as the band's expected count (capacity), growth only past it; memory released and the
budget reduced by the host W; the final chunk re-orthonormalized; new wiring and detection
tests.

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

- `n_modes` is the capacity (expected count; default `4 * chunk`); `chunk` defaults to
  `min(40, n_free - n_rigid - guards(40))` in band mode (block size $m = 48$ at the default,
  the width the memory model is calibrated at; smaller on tiny meshes). One chunk is faster
  than several (PR 9b: 2.4 s vs 4.9 s for 30 modes), so a user who knows the count passes
  `chunk` and `n_modes`. Above about 2M DOF the default block may not fit the device
  (`solve_modes_memory(band=...)` reports it; the Options table says so).
- Every chunk is lockable (the guard error applies to every chunk).
- **Band rule** (host helper `_band_count(lam, t, lam_stop) -> i`, unit-tested on synthetic
  arrays): `i` = the number of Ritz values $\le$ `lam_stop`$\,(1 + 100\,t)$ (a neighbour within
  $100\,r_{acc}$ of the edge reaches into the band), extended over clusters (relative gap
  $< 100\,t$, as `_extent`). After a chunk converges with `j` columns (already extended over a
  boundary cluster): if `i < j` the solve stops (modes $1..i$ of the chunk are in the band,
  $i+1..j$ are converged modes beyond it); otherwise ($i \ge j$: every converged mode is in the
  band, guards may be too) the chunk is locked and the next one starts.
- **Edge above the spectrum:** before a new chunk, if `count + m > n_free - n_rigid`,
  `ValueError("band edge f_max = ... Hz is above the largest computable eigenvalue (count
  modes locked, n_free = ...)")`.
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
- The solve's preconditioner is released (`P = None`) before the λ_c build, and `B` is local to
  `_solve` (freed on return): one hierarchy alive at a time.

### 3. Counting cost model and `verify`

One helper `_predict(...)` (the test seam) returns the decision inputs.

- `d` from `count_degree(g, ...)` (or `count_options["degree"]`); `None` above `MAX_DEGREE`.
  Probe block `b = probe_block(budget, n_probes, op_memory, n_dof, voxel, precond_memory,
  n_rigid)` unless `count_options["block"]`, with `budget` = the solve's
  `solve_modes_memory(...)["total"]` at the returned capacity minus $8\,n\,k_{all}$ (the host W:
  host and device share memory on the Mac; conservative on a GPU). The λ_c multigrid is fp64
  (`precond_memory` at fp64 even when the solve ran an fp32 V-cycle). `n_blocks` =
  $\lceil n_{probes}/b\rceil$; `predicted_steps` = $n_{blocks}\,d$, reported in every mode.
- **A-priori bound (free):** a count step runs ≥ 7 PCG iterations at width b (measured 7–14 at
  λ_c, 14a), each at least one operator and one preconditioner apply; a LOBPCG iteration does
  one preconditioner and two operator applies at width m. So `predicted_s_bound` =
  `predicted_steps` · (b / m) · $t_{it}$, with $t_{it}$ the solve's mean time per LOBPCG
  iteration (per-column scaling favours the count at small b, so this stays a lower bound up to
  LOBPCG's dense $O(n m^2)$ work, covered by the ≥ 7× PCG margin).
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
  $n_{blocks}(d\,t_{step} + t_{block})$ (`predicted_kind: "calibrated"`); the count runs iff
  that is $\le$ the solve time.
- `count` change: `time["step"]` becomes the `_recurrence` time alone per step (the projections
  and the reduction go to `time["block"]`), and the internal `_steps` keyword caps the steps run
  without changing γ's padded class.
- `info["count"]`: `count`'s info plus `ran`, `reason`, `predicted_steps`, `predicted_s`,
  `predicted_kind`, `solve_s`, `calibration_s`.

### 4. `solve_modes_memory(..., band=None)`

With `band`, `n_modes` (required) is the expected count and `chunk` defaults to the band
default; the estimate is the plain estimate at that capacity (no growth when the count is right).
The count fits by construction (budget = this estimate minus the host W).

## Tests (`tests/test_band.py`)

1. **Band rule unit tests:** `_band_count` on synthetic λ, t: exact edge, neighbour inside
   $100\,t$, neighbour just outside, a chain of small gaps crossing the window, none beyond,
   all beyond (`i = 0`).
2. **No missed modes vs the dense inertia** ([14b]): reference box hex20 $6\times5\times4$
   (clamped) and a free hex8 box (rigid modes deflated): $f_{max}$ in a clear gap, at
   $\lambda_k(1 - 20\,\mathrm{tol})$ (mode $k$ returned through the neighbour window) and at
   $\lambda_k(1 - 10^4\,\mathrm{tol})$ (not returned), and one needing 3 chunks with `chunk=8`;
   each $f_{max}$ chosen from the dense spectrum with a margin ≥ 10× from every $100\,t$
   threshold (asserted). Returned count = `count_below(K, M, λ_edge)` (minus the rigid count on
   the free box) plus the predicted window/cluster extension; returned λ match dense `eigh`
   to $10\,\mathrm{tol}$ relative.
3. **Count wiring** ([14b]): `counting.count` wrapped by monkeypatch in test 2's runs: `lam_last`
   = the last returned λ (or the band edge), `lam_next` = the first converged beyond, W has every
   converged column, R passed on the free box, the λ_c builder called at
   $(\lambda_{last} + \lambda_{next})/2$; `info["count"]["lam_c"]` and `gap` equal the dense
   values.
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
   (builder patched to raise); `predicted_s` = $n_{blocks}(d\,t_{step} + t_{block})$ from the
   info values; a capped `d` (`count_options={"degree": MAX_DEGREE + 1}` or a forced tiny gap):
   `None` skips, `True` returns NaN unverifiable.
8. **Growth:** `chunk=8`, `n_modes=8`, $f_{max}$ needing ≥ 4 chunks: the capacity grows (in
   `info`), and λ and V equal a full-capacity run with the same chunk and seed to $10^{-12}$; the
   same with `basis_dtype="float32"` (dtype kept); the count runs on the grown solve.
9. **Edge above the spectrum:** hex8 $2\times2\times2$ clamped with $f_{max}$ = 2× the largest
   dense frequency raises the ValueError; tiny-mesh chunk default works.
10. **Probe block and options:** `info["count"]["block"]` = `probe_block(budget)` with
    `count_memory(b) <= budget < count_memory(b + 1)` (or b = n_probes); `count_options={"block":
    3}` overrides; unknown keys raise before any build.
11. **Errors:** neither `n_modes` nor `band`, `f1 > 0`, $f_{max} \le 0$, non-finite band, bad
    `verify`, bad `count_options`.
12. **Memory:** `solve_modes_memory(band=..., n_modes=k)` equals the plain estimate at the band
    capacity (exact); slow (macOS): the measured process peak of a band solve with
    `verify=True` within 20% of the estimate at a size where it is ≥ 2 GB.
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
  `chunk` default 40 and the 2M DOF limit; the a-priori bound's assumptions; budget = solve
  estimate minus host W; the final chunk converges modes beyond the band (up to `chunk`; a
  stop callback could save iterations); the τ_c question carried over from 14a.
