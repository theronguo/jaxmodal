# PR 14b plan: the [0, f_max] band mode with `verify`

Branch `pr-14b-band` from `main` (PR 14a merged). Roadmap: PR 14, items tagged [14b], and the
"Frequency bands" section (strategy, [0, f_max], band edges, completeness count).

## Scope

- `solve_modes(mesh, materials, bcs, n_modes=None, ..., band=None, verify=None,
  count_options=None)`: exactly one of `n_modes` and `band`. `band=(0, f_max)` returns every
  elastic mode with $f \le f_{max}$ (plus whole clusters reaching into the band). `band=(f1, f2)`
  with `f1 > 0` raises `NotImplementedError` ("interior bands: PR 15").
- The [0, f_max] LOBPCG mode: PR 9b's chunks and locking, stopping once a chunk has a converged
  mode beyond the band edge (and the edge lies in a gap).
- Edge placement and the count's cut, as in the roadmap's "Band edges".
- `verify=None/True/False` with the predicted counting cost in `info["count"]`, the probe block
  from the solve's own memory estimate, the fp64 multigrid (or Jacobi) built at $\lambda_c$.
- `count_options` forwarded to `count` (every count option reachable, PR 13c rule).
- `solve_modes_memory(..., band=..., n_modes=...)`: the band's estimate at an expected count.

Not in 14b (PR 15): shift-invert slices, interior bands, strategy selection between LOBPCG and
shift-invert. With LOBPCG-from-0 as the only strategy, the 14b "cost function" is the counting
cost model that `verify=None` needs; the solve-cost half and the selection arrive with the
second strategy in PR 15 (open item).

## Design

### 1. Band mode in `lobpcg._solve`

New internal keyword `lam_stop=None` (λ in the pencil's units, $(2\pi f_{max})^2$). With it:

- `n_modes` is the initial capacity hint (default `4 * chunk`); `chunk` defaults to 40 in band
  mode (`guards` by the usual rule, $m = 48$, the width the memory model is calibrated at). One
  chunk is faster than several (PR 9b: 2.4 s vs 4.9 s for 30 modes), so a user who knows the
  count passes `chunk`; the default trades time for not needing it.
- Every chunk is lockable (`lock=True`, so the guard error applies to every chunk).
- After a chunk converges with `j` columns (already extended over a boundary cluster), the
  band rule: `i` = the number of its Ritz values $\le$ `lam_stop`$\,(1 + 100\,t_i)$ (a neighbour
  within $100\,r_{acc}$ of the edge reaches into the band), extended over clusters with
  `_extent` (relative gap $< 100\,t$). If `i < j`, the solve stops: modes $1..i$ of this chunk are
  in the band, $i+1..j$ are converged modes beyond it. Otherwise the chunk is locked and the
  next one starts. (`i \le j` always: `j` already ends in a gap.)
- Capacity: when `count + m` exceeds it, `B` grows to $\lceil 1.5\times\rceil$ (rounded to `c`): a new
  zero array, a jitted copy, the old one freed; the host `V` grows the same way (transient peak
  2.5× the old basis, documented; the memory estimate takes the expected count).
- Returns all converged modes (locked + the final chunk's `j`), and `info["n_band"]` = the
  number in the band. `lobpcg`'s public signature is unchanged.
- A failed chunk ends the solve as today (`converged` False); no count runs then (flagged).

### 2. `solve_modes(..., band=(0, f_max))`

- Validation: `band` a pair of finite floats, `0 == f1 < f2` (`f1 > 0`: `NotImplementedError`),
  exactly one of `n_modes`/`band`; `verify` in (None, True, False); `count_options` keys checked
  against `COUNT_OPTIONS = ("n_probes", "seed", "tau", "degree", "block", "maxit")`, pinned to
  `count`'s signature by the existing options test.
- Solve with `_solve(..., lam_stop=(2π f_max)^2)`; returns `lam`, `V` of the band modes only
  (possibly 0 columns) and `info` with `n_band`, `band`, `frequencies` and `count`.
- Edges (roadmap): `lam_last` = the last returned mode, or the band edge $(2\pi f_{max})^2$ when
  none is returned; `lam_next` = the first converged mode beyond. The count deflates by every
  converged mode (returned and beyond, so the tail model sees all modes up to `lam_next`), plus R.

### 3. Counting cost model and `verify`

- `d` from `count_degree(g, ...)` (host, cheap; `None` above `MAX_DEGREE`). `n_blocks` =
  $\lceil n_{probes}/b\rceil$.
- Probe block `b = probe_block(budget, n_probes, op_memory, n_dof, voxel, precond_memory,
  n_rigid)` with `budget` = the solve's own `solve_modes_memory(...)["total"]` at the returned
  count: the count never needs more device memory than the solve did.
- `verify is False`: no count, no multigrid build; `info["count"]` = `{ran: False, degree,
  block, steps: n_blocks * d}` (the predicted cost in recurrence steps; seconds need the
  calibration below, so they are None).
- Otherwise: build the fp64 preconditioner at $\lambda_c$ (the solve's builder: `gmg` / `amg`
  with `nonsingular` from the rigid count / `jacobi`), then a calibration run: `count(...,
  degree=2, n_probes=b, block=b)` twice (the first compiles), `t_step` from the second's
  `time["step"]`. Predicted seconds = $n_{blocks}\,d\,t_{step}$ plus the build and calibration time
  already spent.
- `verify is None`: run the count iff `d` is not None and predicted seconds $\le$ the solve time
  (`info["time"]` LOBPCG + precond + σ pre-estimate). `verify is True`: always run (a capped
  `d` gives `count`'s NaN, unverifiable).
- `info["count"]`: `count`'s info plus `ran`, `predicted_s`, `solve_s`, `calibration_s`,
  `estimate`.

### 4. `solve_modes_memory(..., band=None)`

With `band`, `n_modes` is the expected count (required) and `chunk` defaults to 40, the
capacity the band mode would allocate after its growth steps is used (worst case including the
1.5× transient). The count phase fits by construction (budget = this estimate).

## Tests (`tests/test_band.py`)

1. **No missed modes vs the dense inertia** ([14b]): reference box hex20 $6\times5\times4$
   (clamped) and a free hex8 box (rigid modes deflated), each at 3 values of $f_{max}$ (one inside
   a cluster, one in a clear gap, one high enough for 3 chunks with `chunk=8`):
   `count_below(K, M, λ_edge)` ≤ returned ≤ `count_below(K, M, λ_edge (1 + 100 tol))` plus the
   cluster extension, returned λ match dense `eigh` to $10\,\mathrm{tol}$ relative, and the
   returned set equals the dense set selected by the same edge rule.
2. **Below the first mode** ([14b]): $f_{max}$ = half the first elastic frequency returns 0 modes,
   `verify=True` count < 0.5 and verifiable.
3. **Double eigenvalue** ([14b]): square-section beam, $\lambda_c$ set to the double pair's value
   (between its two copies, equal to rounding): both copies returned; with `chunk` ending
   between the copies the cluster rule still returns both.
4. **verify** ([14b]): `True` runs the count (`ran`), `False` does not (no calibration, `steps`
   reported); `None` follows the prediction: monkeypatch the calibrated `t_step` to 0 → runs, to
   1e9 → skips.
5. **Multi-chunk and growth:** `chunk=8`, initial capacity forced small, $f_{max}$ needing ≥ 4
   chunks: capacity grows, result equals the single-chunk band solve (λ to $10^{-8}$, subspace by
   MAC ≥ 1 − 1e-8).
6. **Probe block:** `info["count"]["block"]` equals `probe_block` at the solve estimate;
   `count_options={"block": 3}` overrides it; unknown keys raise.
7. **Errors:** both/neither of `n_modes`/`band`, `f1 > 0` (`NotImplementedError`), $f_{max} \le 0$,
   non-finite band, bad `verify`, bad `count_options`.
8. **Memory:** `solve_modes_memory(band=..., n_modes=k)` ≥ `solve_modes_memory(n_modes=k)` and
   includes the growth transient; the measured process peak of a band solve with growth within
   the existing ±20% band (slow, macOS), at a size ≥ 1 GB.
9. **Voxel:** test 1 on a voxel box with GMG (clamped).
10. **Example:** README `band` example runs (test_readme).

## Docs

- `docs/eigensolvers.md`: "Frequency band [0, f_max]" (mode, edges, verify and the cost model,
  info keys), `solve_modes` signature and the Options table (`band`, `verify`, `count_options`,
  band `chunk` default 40), memory estimate with band.
- README: a band example; status line. ROADMAP: status entry, PR 14 split note.
- `docs/open_items.md`: no strategy selection yet (PR 15); predicted seconds need calibration
  (verify=False reports steps only); band `chunk` default 40; capacity growth 1.5× with a
  transient; budget = the solve's estimate; the τ_c question carried over from 14a.
