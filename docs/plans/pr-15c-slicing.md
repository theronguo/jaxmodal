# PR 15c plan: slicing an interior band

Branch `pr-15c-slicing` from `main` (PR 15b merged). Roadmap: PR 15 ("Interior bands", the
[15c] items), the "Frequency bands" section (strategy, band edges, interior bands) and the split
in the PR 15a plan (15c = slicing with the cost model, the strategy choice and `band=(f1, f2)`).

## Split of the remaining PR 15 work

15c as planned in 15a holds two separable parts. This plan splits it (the 15a split's rule: every
roadmap test assigned, none dropped):

- **15c (this plan): the slicing engine** `si_band`: a band $[\lambda_{lo}, \lambda_{hi}]$ solved by
  one or more `si_checked` slices (15b) within a given Lanczos capacity, a full slice split at its
  largest gap, deflation between slices, duplicate removal, the final edge re-check with
  Rayleigh–Ritz over merged edges, the slice-collapse flag, per-slice counts.
- **15d: strategy and API**: the cost model (fitted on 15a's measured MINRES cost against depth,
  14b's LOBPCG step time, the counting cost), the memory budget (capacity from `si_memory`, the
  LOBPCG-from-0 basis limit) and the strategy choice, `solve_modes(..., band=(f1, f2))` with the
  f₁ < λ₁ rule (solve [0, f₂] and filter), (0, f_max) forced onto shift-invert through
  `solve_modes`, `solve_modes_memory` for bands, the README example.

Roadmap PR 15 tests tagged [15c] (or [15a, 15c], [15b, 15c]):

| Roadmap test | Part |
|---|---|
| [15a, 15c] matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 15c tests 1, 2, 4 (multi-slice bands); 15d through `solve_modes` |
| [15b, 15c] interior band between two well-separated modes: 0 modes, count < 0.5; planted mode left out: count ≥ 0.5 | 15c test 8 (through `si_band`); 15d test through `band=` |
| [15a, 15c] clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` | 15c test 6 (`si_band` from 0); 15d through `solve_modes` |
| [15c] f₁ below λ₁ solved as [0, f₂], filtered, matches `eigsh` | 15d (the rule uses the σ pre-estimate) |
| [15c] forced edge failing the final r_acc re-check widened or merged, matches `eigsh`; τ_eff forced to 1e-6 near a dense cluster sets the slice-collapse flag | 15c test 5 |
| [15c] each slice's count < 0.5 | 15c tests 2, 4 |
| [15c] memory budget too small for LOBPCG from 0 → shift-invert | 15d |
| [15c] voxel cube band across ≥ 2 slices, cluster near a slice edge, tol 1e-6: no duplicates, count and multiplicities as `eigsh`, λ accuracy | 15c test 4 |
| [15c] band edge exactly on a double eigenvalue returns both copies, d ≈ 25/g | 15c test 7 |
| [15a, 15c] MINRES iterations vs δ measured (15a), cost-model constants fitted | 15d |
| PR 15 contents: capacity per slice from the memory model; a full slice split at its largest gap; a full slice without a gap ≥ 100 r_acc raises (test: a forced small capacity splits the slice and matches `eigsh`) | 15c tests 2, 3 (capacity given); 15d (capacity from `si_memory`) |
| PR 15 contents: removal of duplicate modes at slice edges | 15c tests 4, 9 |

## Facts

- `si_slice` (15a) returns the converged pairs of its `inside` set: $[\lambda_{lo}(1 - 100r),
  \lambda_{hi}(1 + 100r)]$ extended over chains of relative gaps $< 100r$ (clusters whole), $r$ =
  its $r_{acc}$; a converged neighbour beyond each edge (none below at $\lambda_{lo} = 0$); W is
  deflated and never returned. On `capacity_full` it returns the converged inside pairs found so
  far, `edge_unreached` naming an edge no converged Ritz value ever passed.
- `si_checked` (15b) adds the count (band-pass for $\lambda_{lo} > 0$, the step from 0) with up to
  2 restarts; the caller's W modes inside $[\lambda_{lo}, \lambda_{hi}]$ or the returned span count
  as found. A slice that is not converged is returned with reason "slice not converged" and no
  count.
- Measured (15a, `docs/eigensolvers.md`): about 6.6 Lanczos columns per mode on a full slice; the
  MINRES iterations per step follow the depth of σ, not δ or the slice width. So for a fixed band
  the Lanczos cost is about (columns) × (iterations per step), nearly independent of how the band
  is cut, plus a per-slice overhead (the start and the neighbours, a few blocks), and the
  reorthogonalization ($4 n L b$ flops per step at basis $L$) stays below 1% of a step's MINRES
  ($\gtrsim 10^3$ iterations × $b$ operator and V-cycle applies at depth ≳ 100). Each slice also
  costs one count of degree $\approx 23/g_\text{eff}$ at its own edges. Fewer, wider slices are
  therefore cheaper: the slice width is limited by memory (the capacity), and 15d's cost model
  needs no provisional interior edges.

## Design

### `si_band`

```python
lam, V, info = si_band(op, free, lo, hi, make, *, capacity, R=None, verify=True,
                       edges=None, count_options=None, tol=1e-6, tau=1e-8, block=6,
                       seed=0, verbose=False, max_slices=64, **slice_options)
```

Internal (module `eigen/shift_invert.py`, as `si_slice` and `si_checked`; 15d calls it from
`solve_modes`). `make(sigma)` returns the SPD preconditioner at σ (the slice's midpoint; one
hierarchy alive at a time, as `solve_modes`'s `make`). `capacity`: Lanczos columns per slice (15d
sets it from the budget). `verify`: run each slice's count (`si_checked`), or not (`si_checked`
gains `verify=False`: the slice alone, `info["count"]` = `{ran: False, reason: "not requested",
passed: False, verifiable: False}`, no restarts). `edges`: optional provisional interior edges
(tests and callers that know the spectrum); default one slice. `max_slices`: a backstop.

**Work list.** Slices are processed from a list ordered by λ, lowest first, starting with
$[\lambda_{lo}, e_1], [e_1, e_2], \dots, [e_k, \lambda_{hi}]$. A store holds every accepted pair
(λ, v, the slice that found it, its residual and threshold). For slice $[a, b]$:

1. **Deflation.** W = the stored pairs inside the slice's window $[a(1 - 100r_s), b(1 + 100r_s)]$
   ($r_s$ = the largest $r_{acc}$ so far, tol at first) extended over chains of relative gaps
   $< 100 r_s$: the modes it would otherwise find again (a split parent's converged modes, a
   finished neighbour's cluster reaching over the edge). Modes outside the window are not
   deflated, so the slice's own neighbours stay nearby eigenvalues it can converge quickly (a
   deflated neighbour would push si_slice's stop rule to the next undeflated eigenvalue).
2. **Run.** `si_checked(op, free, a, b, make((a + b) / 2), capacity=capacity, W=W, lam_W=...,
   R=R, verify=verify, ...)`; the preconditioner is released after the slice.
3. **Outcome.**
   - Converged: its pairs join the store (the deflated W modes are not returned, so no duplicate
     arises from them).
   - `capacity_full`: its converged pairs join the store (they passed si_slice's final residual
     check), and the slice is **split** at the largest relative gap
     $(\mu_{i+1} - \mu_i)/\mu_{i+1}$ between consecutive converged values in $(a, b)$ (the
     returned pairs and the W modes inside), provided the gap is $\ge 100\,r_{acc}$ of the slice:
     the edge $s = (\mu_i + \mu_{i+1})/2$ replaces $[a, b]$ by $[a, s], [s, b]$ in the list. This
     is the roadmap's edge snapping (an edge in the largest gap among converged Ritz values of its
     window, the window being the full slice). Fewer than two converged values or no gap
     $\ge 100\,r_{acc}$: `ValueError("slice [a, b] fills capacity C without a gap >= 100 r_acc
     among its converged modes: raise capacity")`. A child holds fewer unknown modes than its
     parent (the converged ones are deflated) and σ moves to its midpoint; `max_slices` bounds
     the splits (`RuntimeError` past it).
   - `edge_unreached` with capacity full: "hi" on the highest slice raises `ValueError("band
     edge above the spectrum")` (roadmap); "lo" on the lowest slice retries that slice once as
     $[0, b]$ (λ_lo below λ₁: the [0, f₂] rule as a fallback; 15d applies it up front from the
     pre-estimate), its pairs filtered to the band at the end. Elsewhere it is a split.
   - Count flagged (after 15b's restarts) or unverifiable: the pairs join the store and the band
     is flagged (`info["count"]["passed"]` False); no split (a rerun would not change it, 15b).
   - Inner solves failed (si_slice reason): `converged` False, the band stops with the pairs
     found, reason set.

**Duplicates.** A pair can still be found twice when a cluster chain reaches past a slice's
deflation window. After all slices, the store is scanned in slice order: a pair $(\mu, v)$ from
slice $j$ is compared with the kept pairs of other slices within $100\,r$ relative of μ ($r$ =
the band-wide $r_{acc}$); with $Q$ those kept vectors, $p = v - Q Q^\top M v$, and $v$ is dropped
when $\|p\|_M < 0.5$ (roadmap: a new vector has $\approx 1$, a duplicate $\lesssim 2 r\lambda/g$).
The kept vectors are the original Ritz vectors; the projection only detects.

**Edge re-check.** $r_{acc}$ (band) = the maximum over slices (si_slice's `r_acc`, already over
the pairs and neighbours). Each interior edge $s$ is re-checked: with $\mu_-$ the largest kept
eigenvalue below $s$ and $\mu_+$ the smallest above, the edge passes when
$(\mu_+ - \mu_-)/\mu_+ \ge 100\,r_{acc}$. A failing edge is **merged**: the window = the chain
of kept pairs with relative gaps $< 100\,r_{acc}$ through $\mu_-, \mu_+$; Rayleigh–Ritz of
$(K, M)$ on those vectors (host: $G_K = V_w^\top K V_w$, $G_M = V_w^\top M V_w$ by
`apply_K`/`apply_M` on the window, `scipy.linalg.eigh(G_K, G_M)`), the window's pairs replaced by
the Ritz pairs, residuals recomputed; a Ritz pair above its threshold (the larger of the two
slices' thresholds at μ) sets `info["edge_rr_failed"]`. A merged edge is listed in
`info["merged_edges"]`; when every interior edge (at least one) is merged the slices have
collapsed into one: `info["collapsed"]` True (roadmap flag). The counts stay valid: Rayleigh–Ritz
rotates within the span the counts deflated.

**Result.** `lam`, `V`: the kept pairs, ascending (host fp64, M-orthonormal: slices deflate each
other's windows, duplicates are dropped, merged windows are M-orthonormalized by the Ritz step);
with the [0, b] fallback, filtered to $[\lambda_{lo}(1 - 100r), \lambda_{hi}(1 + 100r)]$ with
clusters whole. `info`: `converged` (every slice converged, no raise), `reason`, `r_acc`,
`tau_eff` (max), `residuals`, `thresholds`, `slices` (per slice: `interval`, `sigma`, `steps`,
`basis`, `n_new`, `capacity_full`, `count` estimate and reason, `restarts`, `time`), `splits`,
`merged_edges`, `collapsed`, `edge_rr_failed`, `duplicates` (dropped), `count` (`ran`, `passed` =
every slice passed, `flagged`, `estimates` per slice, `reasons`), `time` (total, per phase).
`verbose`: one line per slice (interval, σ, outcome, modes, count) plus `si_slice`'s per-step
lines.

### Changes to existing code

- `si_checked(..., verify=True)`: `verify=False` runs the slice alone (no count, no restarts).
- Nothing else in 15a/15b changes.

## Tests (`tests/test_shift_invert.py`, mesh "short" = hex20 6×5×4 unless noted, AMG; references
from the cached dense `eigh`; tol $10^{-10}$, τ $= 10^{-8}$ unless noted; counts with the exact
dense preconditioner where a test checks counts, as in 15b)

1. **One slice:** `si_band` on modes 13–18's band (the 15b band) equals `si_checked`'s result;
   `splits` 0, one slice, count passed.
2. **Capacity split (roadmap):** a band of modes 8–20 with capacity 4 blocks: at least one split,
   every slice converged, the union matches the reference (count, λ to max(1e-8, 10 ε λ̂_max/λ)
   for gaps ≥ 1e-2, else 10 r_acc), $\|V^\top M V - I\| \le 10^{-10}$, no duplicates, each slice's
   count < 0.5 (roadmap "each slice's count").
3. **No gap raises:** a fake `si_checked` returning `capacity_full` with converged values in one
   exact cluster raises the capacity error; fewer than two converged values raises; `max_slices`
   exceeded raises; "hi" unreached on the top slice raises "above the spectrum".
4. **Voxel cube across slices (roadmap):** the free voxel cube (15a's, with R), a band holding
   its double and triple clusters, `edges` forcing an interior edge within $100\,r$ of a cluster
   (inside the chain), tol $10^{-6}$: no duplicates, count and multiplicities as the reference,
   λ within max(1e-8, 10 ε λ̂_max/λ) for clusters with relative gap ≥ 1e-2, else 10 tol; each
   slice's count < 0.5. Slow if over the fast budget.
5. **Edge re-check and collapse (roadmap):** (a) `edges` at a near-degenerate gap with inner
   solves stopped at $10^{-6}$ in the slices (15a's monkeypatch: τ_eff = 1e-6) so that the
   band-wide $100\,r_{acc}$ exceeds the edge's gap: the edge is merged, `collapsed` True (one
   interior edge), the result matches the reference within 10 r_acc; (b) fake slices (reference
   pairs with a rotated cluster split across the edge): Rayleigh–Ritz restores the exact pairs to
   1e-12; a window whose Ritz residual exceeds the threshold sets `edge_rr_failed`; two edges of
   which one passes: `merged_edges` lists one, `collapsed` False.
6. **From 0 (roadmap, the 15c part):** clamped `si_band(0, hi)` over modes 0–5 matches the
   reference; the count is the step count.
7. **Edge on a double eigenvalue (roadmap):** voxel cube, $\lambda_{hi}$ set exactly on a double
   eigenvalue: both copies returned, the count's degree within [19, 24]/g of the edge gap
   (count_degree's rule; roadmap "d ≈ 25/g").
8. **Empty band and planted mode (roadmap, end to end):** a band in the middle third of the
   largest gap: 0 modes, count < 0.5; with the density scaled to plant a mode in it (15b test 7's
   operator) and `si_slice` patched to drop that mode on every run: the count stays ≥ 0.5, the
   band is flagged after 2 restarts.
9. **Duplicates:** fake slices returning the same mode twice (and a near copy rotated by
   $10^{-3}$ within a cluster): one kept, `duplicates` 1; a genuinely new vector in the same
   window (projected norm ≈ 1) is kept.
10. **Edge fallback and errors:** "lo" unreached on the lowest slice reruns it from 0 and filters
    (fake); `verify=False` runs no count (`si_checked` called with it, `count["ran"]` False); W
    windows: a split child's W holds exactly the parent's converged modes in its window (spy);
    argument errors (lo ≥ hi, edges outside (lo, hi) or unsorted, capacity).

Budget: tests 1–3, 5b, 6, 8–10 fast (seconds each with the cached references; fakes where the
logic is the subject); 2, 4, 5a and 7 run real slices (tens of seconds each) and are marked slow
only if they exceed 60 s.

## Docs

- `docs/eigensolvers.md`, "Shift-invert slices": a "Slicing a band" subsection (the work list,
  deflation windows, splits at the largest gap and why the width is memory-limited, duplicates,
  the edge re-check with Rayleigh–Ritz, the collapse flag, `info`); Validation list.
- `docs/ROADMAP.md`: the 15c/15d split note in PR 15, the [15c]/[15d] tags, the Dependencies
  table ("15c Slicing | 15b", "15d Band strategy, `band=(f1, f2)` | 14b, 15c"), the PR 15c status
  entry.
- `docs/open_items.md`: decisions below.
- README: status line (15c; the `band=(f1, f2)` example with 15d).

## Decisions without user input (to log)

- 15c split into 15c (slicing engine) and 15d (strategy, API, memory), all roadmap tests assigned.
- No provisional interior edges from the cost model: slices come from splitting full slices at
  their largest converged gap (the measured cost favours the widest slice that fits); `edges` is
  an internal option.
- Deflation windows: a slice deflates only the stored modes in its own window, so its neighbours
  stay close.
- The [0, b] fallback when λ_lo is unreached on the lowest slice (15d applies the roadmap rule up
  front).
- A flagged or unverifiable slice count is not split or rerun beyond 15b's restarts.
