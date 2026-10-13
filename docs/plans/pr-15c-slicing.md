# PR 15c plan: slicing an interior band

Branch `pr-15c-slicing` from `main` (PR 15b merged). Roadmap: PR 15 ("Interior bands", the
[15c] items), the "Frequency bands" section (strategy, band edges, interior bands) and the split
in the PR 15a plan (15c = slicing with the cost model, the strategy choice and `band=(f1, f2)`).

Revision 2 (plan review round 1): a full slice keeps its converged span and requeues only its
outer remainders (no binary split); `capacity` is the total basis per slice, deflation included;
`edge_unreached` no longer raises or reruns (no [0, b] fallback); one window routine for
duplicates and Rayleigh–Ritz over every cluster held by two slices, applied at the boundaries
between the slices' kept sets; a `verify` hook for 15d; the measured cost model of a slice;
measured test capacities; the roadmap text and the stale 15c pointers in the docs list.

## Split of the remaining PR 15 work

15c as planned in 15a holds two separable parts. This plan splits it (the 15a split's rule: every
roadmap test assigned, none dropped):

- **15c (this plan): the slicing engine** `si_band`: a band $[\lambda_{lo}, \lambda_{hi}]$ solved by
  `si_checked` slices (15b) within a given basis capacity, full slices reduced to their outer
  remainders, deflation between slices, the merge of clusters held by two slices (duplicates,
  Rayleigh–Ritz, the final re-check against the band-wide $r_{acc}$, the slice-collapse flag),
  per-slice counts.
- **15d: strategy and API**: the cost model (the slice model measured below, 15a's MINRES cost
  against depth, 14b's LOBPCG step time, the counting cost), the memory budget (capacity from
  `si_memory`, the LOBPCG-from-0 basis limit, the host term of the found modes) and the strategy
  choice, `verify=None` for bands (through the hook below), `solve_modes(..., band=(f1, f2))`
  with the f₁ < λ₁ rule (solve [0, f₂] and filter) and the check of $\lambda_{hi}$ against the
  spectrum up front, (0, f_max) forced onto shift-invert through `solve_modes`,
  `solve_modes_memory` for bands, the README example, the roadmap's PR 15 docs item (memory model
  for many modes, cost-model calibration).

Roadmap PR 15 tests tagged [15c] (or [15a, 15c], [15b, 15c]):

| Roadmap test | Part |
|---|---|
| [15a, 15c] matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 15c tests 1, 2, 4 (multi-slice bands); 15d through `solve_modes` |
| [15b, 15c] interior band between two well-separated modes: 0 modes, count < 0.5; planted mode left out: count ≥ 0.5 | 15c test 8 (through `si_band`); 15d test through `band=` |
| [15a, 15c] clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` | 15c test 6 (`si_band` from 0); 15d through `solve_modes` |
| [15c] f₁ below λ₁ solved as [0, f₂], filtered, matches `eigsh` | 15d (the rule uses the σ pre-estimate; 15c has no fallback) |
| [15c] forced edge failing the final r_acc re-check widened or merged, matches `eigsh`; τ_eff forced to 1e-6 near a dense cluster sets the slice-collapse flag | 15c test 5 |
| [15c] each slice's count < 0.5 | 15c tests 2, 4 (every counted leaf) |
| [15c] memory budget too small for LOBPCG from 0 → shift-invert | 15d |
| [15c] voxel cube band across ≥ 2 slices, cluster near a slice edge, tol 1e-6: no duplicates, count and multiplicities as `eigsh`, λ accuracy | 15c test 4 |
| [15c] band edge exactly on a double eigenvalue returns both copies, d ≈ 25/g | 15c test 7 |
| [15a, 15c] MINRES iterations vs δ measured (15a), cost-model constants fitted | 15d (on the slice model below and 15a's study) |
| PR 15 contents: capacity per slice from the memory model; a full slice split at its largest gap; a full slice without a gap ≥ 100 r_acc raises (test: a forced small capacity splits the slice and matches `eigsh`) | 15c tests 2, 3 (capacity given); 15d (capacity from `si_memory`) |
| PR 15 contents: removal of duplicate modes at slice edges | 15c tests 5, 9 |

## Facts

- `si_slice` (15a) returns the converged pairs of its `inside` set: $[\lambda_{lo}(1 - 100r),
  \lambda_{hi}(1 + 100r)]$ extended over chains of relative gaps $< 100r$ (clusters whole), $r$ =
  its $r_{acc}$; a converged neighbour beyond each edge (none below at $\lambda_{lo} = 0$); W is
  deflated and never returned. Its device basis is $Q$: $(n, \lceil n_W/c\rceil c + \lceil
  \texttt{capacity}/c\rceil c)$. On `capacity_full` it returns the converged inside pairs found so
  far with `converged` False and `reason` "capacity full" (distinct from "inner solves failed");
  `edge_unreached` names the edges no converged Ritz value has passed, which is the normal state
  of a slice too wide for its capacity (convergence spreads outward from σ; every full run in the
  round-1 probes reported "both", "lo" or "hi" with both edges deep inside the spectrum), so it is
  no evidence of an edge outside the spectrum.
- `si_checked` (15b) adds the count with up to 2 restarts; the caller's W modes inside
  $[\lambda_{lo}, \lambda_{hi}]$ or the returned span count as found; its count edges come from
  the found sets (`_beyond`), not from the nominal slice edges. An unconverged slice gets no count
  ("slice not converged").
- **Slice cost, measured** (plan review round 1; hex20 8×6×5 clamped, 3720 free DOF, AMG,
  τ = $10^{-8}$, capacity never full, each later slice deflating the earlier slices' modes):

  | band | slices | tol | Lanczos columns | MINRES block iterations | time |
  |---|---|---|---|---|---|
  | modes 20–60 | 1 | 1e-8 | 156 | 6660 | 45 s |
  | modes 20–60 | 2 | 1e-8 | 240 | 10920 | 55 s |
  | modes 20–60 | 4 | 1e-8 | 402 | 18780 | 83 s |
  | modes 10–50 | 1 | 1e-8 | 150 | 25.3k column its | 21 s |
  | modes 10–50 | 2 | 1e-8 | 234 | 42.4k | 33 s |
  | modes 10–50 | 4 | 1e-8 | 414 | 93.8k | 87 s |

  Columns ≈ 1.8 per mode + about 84 per slice (tol 1e-8); at tol $10^{-10}$ single slices need
  54 columns for 2 modes, 90 for 12 and 156 for 40: a slice costs about 50–85 columns (8–14
  Lanczos steps) before it converges anything. 15a's "6.6 columns per mode" was one 20-mode slice,
  overhead included. MINRES iterations per step follow the depth of σ (15a), so the cost of a band
  is about (columns) × (iterations per step): one slice is 2.8× cheaper than 2 and 3.7× cheaper
  than 4 here. Reorthogonalization ($4nLb$ flops per step) is about $10^{-4}$ of a step's MINRES at
  depth 1000 and $L = 1300$, 0.2% at depth 100, a few % only at depth ≤ 20. So the widest slice
  that fits the memory is the cheapest, splits are worth it only when memory forces them, and each
  extra slice costs its 8–14-step overhead (35–55 h near depth 1000 at 1M DOF on the Mac).
- **Small capacities, measured** (round 1, mesh "short", tol $10^{-10}$, band of modes 8–21,
  [10.46, 22.63], 14 modes): capacity 24 converges nothing; 48 converges 3 pairs on the first run
  and later full slices converge 0 new pairs; 72 finishes with splits (5 runs, all 14 modes,
  error 1.4e-13). Two converged adjacent slices are M-orthogonal to 1e-13 (an edge in a 10% gap).

## Design

### `si_band`

```python
lam, V, info = si_band(op, free, lo, hi, make, *, capacity, R=None, verify=True,
                       edges=None, count_options=None, tol=1e-6, block=6, seed=0,
                       verbose=False, max_slices=64, **slice_options)
```

Internal (module `eigen/shift_invert.py`, as `si_slice` and `si_checked`; 15d calls it from
`solve_modes`). Explicit keywords are those `si_band` reads itself; `**slice_options` (`tau`,
`maxit_inner`, ...) pass to `si_checked`/`si_slice`.

- `make(sigma)`: the SPD preconditioner at σ (the slice's midpoint), as `solve_modes`'s `make`.
  The reference is dropped (`P = None`) as soon as the slice returns, before the next `make`, so
  one hierarchy is alive at a time. The count reuses it (15b's default; decision below).
- `capacity`: the total device basis per slice, deflation included: slice $j$ runs with
  $\texttt{capacity}_j = \lceil \texttt{capacity}/c\rceil c - \lceil n_{W,j}/c\rceil c$ ($c$ =
  the projector chunk), so every slice's $Q$ stays within the first run's (15b's restart rule then
  keeps restarts within it too). $\texttt{capacity}_j < 2b$ raises the capacity error.
- `verify`: True (count every leaf), False (none), or a callable `verify(info) -> bool` that
  `si_checked` calls after the slice converged and before counting, with the slice's info plus
  `count_degree` and `count_gap`, the degree and governing gap the count would use (from
  `band_edges`/`count_degree`, host-only and cheap). 15d's `verify=None` is such a callable
  (predicted count cost against the slice's solve time, as 14b). The predicted degree and gap are
  recorded in `info["count"]` whether or not the count runs.
- `edges`: optional provisional interior edges (tests, and callers that know the spectrum),
  used as given: a cluster straddling one is held whole by the first slice processed (si_slice's
  chains) and deflated by the next (windows below), which is the roadmap's snapping to a gap
  without a search. Default: one slice.
- `seed`: slice $i$ (in processing order) uses `seed + i` for its start block (si_checked adds the
  restart index), so no slice reuses another's start block; the count seed likewise.
- `max_slices`: a backstop (`RuntimeError`).

**Work list.** Slices are processed from a list ordered by λ, lowest first. A store holds every
accepted pair (λ, v, its leaf, residual, threshold). For slice $[a, b]$:

1. **Deflation window.** W = the stored pairs in $[a(1 - 100 r_w), b(1 + 100 r_w)]$ extended over
   chains of relative gaps $< 100 r_w$, with $r_w = 10\max(r_s, \texttt{tol})$ and $r_s$ the
   largest $r_{acc}$ so far: the modes the slice would otherwise find again. The factor 10 covers a
   later slice's larger $r_{acc}$; extra deflation only moves si_slice's own neighbour outward,
   and `_beyond` uses the deflated W modes as the count's neighbours. Modes far outside the window
   stay undeflated, so the slice's neighbours remain nearby eigenvalues.
2. **Run** `si_checked(op, free, a, b, P, capacity=capacity_j, W=W, lam_W=..., R=R,
   verify=verify, seed=seed + i, ...)`.
3. **Outcome** (by si_slice's `reason`, not by `converged`, which is False for both failures):
   - **Converged:** a leaf; its pairs join the store (the W modes are not returned, so they are
     not found twice).
   - **Capacity full:** its converged pairs join the store (they passed si_slice's final residual
     check). The converged values $\mu_1 < \dots < \mu_m$ (the returned pairs and the W modes in
     $(a, b)$) form a span around σ. With $r$ = the slice's $r_{acc}$:
     - the lower cut $s_{lo}$ = the midpoint of the largest relative gap
       $(\mu_{i+1} - \mu_i)/\mu_{i+1} \ge 100r$ among consecutive values below σ, or none;
       likewise the upper cut $s_{hi}$ above σ (if σ falls in a gap ≥ 100r with values on both
       sides, it may serve as one cut);
     - the slice is replaced by the remainders $[a, s_{lo}]$ and $[s_{hi}, b]$ (each only if its
       cut exists) and the **span leaf** $[s_{lo}, s_{hi}]$ (with $a$ or $b$ for a missing cut): no
       Lanczos run, its pairs are the stored ones inside; its count (when verify) is
       `band_count` on them with the nearest stored pairs beyond $s_{lo}$, $s_{hi}$ as out-of-band
       pairs (they converged in the same run; λ_lo = 0 uses `count`), at the remainders'
       preconditioner `make(c_hi)` built for it. This is the roadmap's split at the largest gap
       (one per side), without a second run of the converged middle.
     - **No progress** (no new converged pair) or **no cut** on either side: `ValueError("slice
       [a, b] fills capacity C without converging a gap >= 100 r_acc: raise capacity (measured: a
       slice needs about 50-85 columns before it converges anything); lo may be below lambda_1
       or hi above the spectrum")`. The remainders hold fewer unknown modes than the parent (the
       span is done) and their σ sits in their unknown region, so the list terminates; with
       `max_slices` as the backstop.
   - **Count flagged** (after 15b's restarts) or **unverifiable**: a leaf; its pairs join the store
     and the band count is not passed; no split.
   - **Inner solves failed:** the band stops with the pairs found (`converged` False, `reason`
     from the slice, prefixed with its interval).

**Merge of clusters held by two leaves** (one routine for duplicates, the roadmap's Rayleigh–Ritz
on merged clusters and the final re-check). After all slices, with $r_{acc}$ (band) = the maximum
over slices: the kept pairs are sorted and split into windows, chains of relative gaps
$< 100 r_{acc}$. A window holding pairs of one leaf is final. A window holding pairs of two or more
leaves (a cluster across the boundary between leaves' kept sets, which is where a nominal edge
"fails" the re-check, or a pair found twice) is merged:

1. In leaf processing order, $Q$ = the first leaf's vectors (M-orthonormal); for each next leaf's
   vectors $U$: $P = U - QQ^\top M U$, `eigh`$(P^\top M P)$, keep the directions $U y_k$ with
   eigenvalue $\ge 0.25$ ($\|p\|_M \ge 0.5$: the roadmap's duplicate rule applied to the subspace,
   so a 45°-rotated copy of a cluster pair is recognized), append them to $Q$
   (M-orthonormalized); the dropped count is `info["duplicates"]`.
2. Rayleigh–Ritz of $(K, M)$ on $Q$: $G_K = Q^\top K Q$, $G_M = Q^\top M Q$ (`apply_K`/`apply_M`
   on the window, a few columns), `scipy.linalg.eigh(G_K, G_M)`; the window's pairs are replaced
   by the Ritz pairs and their residuals recomputed against the threshold (the largest of the
   window's thresholds).
3. A Ritz pair above its threshold: the window is widened once to the next gap $\ge 100 r_{acc}$
   on each side (adding the adjacent pairs) and steps 1–2 repeat (roadmap: "Rayleigh–Ritz over that
   edge window is repeated"); still above: `info["edge_rr_failed"]` True.

`info["merged_edges"]` lists the boundaries between consecutive leaves whose window was merged;
`info["collapsed"]` is True when every boundary between leaves (at least one) is merged (the
roadmap's slice-collapse flag: τ_eff made $100 r_{acc}$ exceed the gaps at all of them). The
counts stay valid: each leaf's count deflated its W and its returned span, and the merge keeps the
union span (it drops only directions with $\|p\|_M < 0.5$, copies of kept ones).

**Result.** `lam`, `V`: the kept pairs, ascending, assembled once into a preallocated host fp64
array (per-leaf blocks freed as copied), M-orthonormal within a leaf exactly and across leaves to
the merge's Rayleigh–Ritz or to $O(r/g)$ at boundaries with gaps $g \ge 100 r$ (1e-13 measured).
`info`: `converged` (every leaf converged, no inner failure), `reason`, `r_acc`, `tau_eff`
(max), `residuals`, `thresholds`, `slices` (each slice's `si_checked` info as is, with the vector
fields `neighbours` removed from it and its `runs`, plus `interval`, `kind` "run" or "span",
`leaf` (False for a full slice replaced by remainders), `cuts`, `n_new`), `splits`,
`merged_edges`, `collapsed`, `edge_rr_failed`, `duplicates`, `count` (si_checked's shape over the
leaves: `ran` any leaf counted, `passed` every leaf counted and passed, `verifiable` all counted
leaves verifiable, `flagged` any, `estimate` the max, `estimates` per leaf, `reason` None or the
first failing leaf's reason prefixed with its interval), `time` (total, slices, merge). Host
memory: the store $8nN$ ($N$ found modes) plus a window copy $8n\,n_{W}$ and the slice's $V$
while it runs, plus 15b's restart copy; 15d's budget subtracts it (as 14b does for the
converged modes). `verbose`: one line per slice (interval, σ, capacity_j, outcome, new modes,
count) plus `si_slice`'s per-step lines.

### Changes to existing code

- `si_checked(..., verify=True)`: False runs the slice alone (no count, no restarts); a callable
  decides after the slice converged; `info["count"]` always holds `count_degree` and `count_gap`
  (the prediction), and `ran` False with reason "not requested" when skipped.
- Nothing else in 15a/15b changes.

## Tests (`tests/test_shift_invert.py`; mesh "short" = hex20 6×5×4 unless noted; references from
the cached dense `eigh`; tol $10^{-10}$, τ $= 10^{-8}$ unless noted. Tests that check counts pass
a `make(σ)` returning the exact dense $(K + \sigma M)^{-1}$ on the free DOFs (SPD for σ > 0), so
each slice's count runs a few PCG iterations; others use AMG. Capacities are measured on the
implementation and recorded as constants with the observed number of slices.)

1. **One slice:** `si_band` on modes 13–18's band (15b's) with a large capacity equals
   `si_checked`'s result; one leaf, `splits` 0, count passed.
2. **Capacity split (roadmap):** the band of modes 8–21 with capacity 72 (12 blocks; round-1
   probe: splits, all 14 modes): at least one full slice replaced by remainders and a span leaf,
   every leaf converged, the result matches the reference (count, λ to max(1e-8, 10 ε λ̂_max/λ)
   for gaps ≥ 1e-2, else 10 r_acc), $\|V^\top M V - I\| \le 10^{-10}$, no duplicates, every
   counted leaf's count < 0.5 (roadmap "each slice's count"), the recorded number of slices; a
   spy checks $\lceil \texttt{capacity}_j/c\rceil c + \lceil n_{W,j}/c\rceil c \le \lceil
   \texttt{capacity}/c\rceil c$ and that a remainder's W holds the stored modes of its window.
3. **Fakes for the outcome rules:** a full slice with no new converged pair raises the capacity
   error; a full slice whose converged values sit in one exact cluster (no cut) raises; a full
   slice with `edge_unreached` "both" and valid cuts splits (no raise); `max_slices` exceeded
   raises; an inner failure stops the band with the pairs found, `converged` False and the reason;
   an unverifiable (capped) and a flagged leaf are not split, `count["passed"]` False, `flagged` as
   reported; a replaced parent's missing count does not fail `passed`; the span leaf's count calls
   `band_count` with the stored pairs and the nearest pairs beyond its cuts.
4. **Voxel cube across slices (roadmap):** the free voxel cube (15a's, with R), a band holding
   its double and triple clusters, `edges` with one interior edge inside a cluster's chain, tol
   $10^{-6}$: the cluster is held whole by the lower slice and deflated by the upper (`collapsed`
   False, no merge, `duplicates` 0), count and multiplicities as the reference, λ within
   max(1e-8, 10 ε λ̂_max/λ) for clusters with relative gap ≥ 1e-2, else 10 tol, M-orthonormal,
   each leaf's count < 0.5. Timed first; slow if over 60 s.
5. **Re-check and collapse (roadmap):** (a) a near-degenerate pair of "short" (relative gap $g$,
   e.g. modes 23–24 at 4.6e-4) with an interior edge (`edges`) between its copies; inner solves
   stopped at $10^{-6}$ (15a's monkeypatch, τ_eff = 1e-6) in the second slice only, on a wide
   enough slice that its $100 r_{acc} > g$ (asserted from the run's info before the merge
   assertions); the first slice stops with $r_{acc} \approx$ tol and keeps one copy: the window
   across the boundary holds vectors of both leaves, it is merged, `merged_edges` lists the
   boundary, `collapsed` True (the only boundary), the result matches the reference within
   10 r_acc and is M-orthonormal; (b) fake leaves with reference pairs: a cluster split across
   the boundary with rotated vectors is restored by the Rayleigh–Ritz to 1e-12; a window whose
   Ritz residual stays above the threshold after widening sets `edge_rr_failed`; two boundaries of
   which one merges: `collapsed` False.
6. **From 0 (roadmap, the 15c part):** clamped `si_band(0, hi)` over modes 0–5 matches the
   reference; the count is the step count.
7. **Edge on a double eigenvalue (roadmap):** voxel cube, $\lambda_{hi}$ set exactly on a double
   eigenvalue and $\lambda_{lo}$ chosen so the upper edge governs: both copies returned, the leaf's
   count `gap` = its upper edge gap and $19 \le d \cdot$ `gap` $\le 24$ (count_degree's rule;
   roadmap "d ≈ 25/g").
8. **Empty band and flagged leaf (roadmap, end to end):** a band in the middle third of the
   largest gap: 0 modes, count < 0.5; a fake `si_checked` returning a flagged count: the band's
   `count["flagged"]` True, `passed` False, `reason` naming the interval (15b tests the restarts).
9. **Merge routine:** fake leaves: the same mode returned twice → one kept, `duplicates` 1; a
   45°-rotated double (both copies from the second leaf, one copy kept from the first) → 2
   vectors kept, exact pairs, M-orthonormal; a near copy $\cos\theta\,v_1 + \sin\theta\,v_2$
   ($\theta = 10^{-3}$, $v_1, v_2$ kept) → dropped, no mode lost; a genuinely new vector in the
   window (projected norm ≈ 1) → kept; M-orthonormality asserted after each.
10. **Hook and errors:** `verify=False` (`si_checked` called with it, `count["ran"]` False, the
    predicted `count_degree` present); a callable declining one leaf and accepting another; seeds
    `seed + i` per slice (spy); argument errors (lo ≥ hi, edges outside (lo, hi) or unsorted,
    capacity below 2b after deflation).

Budget: tests 1, 3, 5b, 6, 8–10 fast (seconds; fakes where the logic is the subject); 2, 5a and 7
run real slices (round-1 probes: 3–26 s each) and stay fast if under 60 s; test 4 is timed first.

## Docs

- `docs/eigensolvers.md`, "Shift-invert slices": a "Slicing a band" subsection (the work list,
  deflation windows, capacity per slice, full slices reduced to remainders and a span leaf, the
  measured slice cost model and why the width is memory-limited, the merge routine, the re-check
  and collapse flag, the `verify` hook, `info`, host memory); the Validation list; the stale "15c"
  pointers that now mean 15d (strategy choice, `band=(f1, f2)`, capacity from `si_memory`, the cost
  model on depth, "the choice 15c makes": lines near 122, 589, 607, 614, 643, 663, 666, 676).
- `docs/ROADMAP.md`: the PR 15 split sentence rewritten for 15c/15d; the [15c] tags split into
  [15c]/[15d]; the "Slice widths come from the cost model …" bullet and the Contents bullet
  "Slicing with the cost model" amended with a note (widths limited by the capacity; edges from
  splits of full slices at their largest converged gap, which is the snapping rule; leaves keep
  si_slice's widened inside sets); the PR 15 docs item moved to 15d; the Dependencies table
  ("15c Slicing | 15a, 15b", "15d Band strategy, `band=(f1, f2)` | 14b, 15c"); the PR 15c status.
- `docs/open_items.md`: the decisions below; the "Shift-invert cost near mode 1000" and "σ-move
  retries" items and the 15c references in the PR 15a list retagged to 15d.
- README: status line (15c; the `band=(f1, f2)` example with 15d).

## Decisions without user input (to log)

- 15c split into 15c (slicing engine) and 15d (strategy, API, memory, `verify=None`), all roadmap
  tests assigned.
- No provisional interior edges from the cost model: the measured slice cost (1.8 columns per
  mode plus 50–85 per slice) makes the widest slice that fits the cheapest; slices come from full
  slices reduced to their remainders at the largest converged gap on each side of σ, the
  converged span kept as a leaf without a second run. `edges` is an internal option, used as
  given (si_slice's chains and the deflation windows assign a straddling cluster whole).
- Leaves keep si_slice's widened inside sets (margins and straddling chains) rather than only the
  pairs inside their own interval; deflation windows and the merge routine handle the overlap.
- Deflation windows at $10\max(r_s, \texttt{tol})$; `capacity` is the total basis per slice.
- `edge_unreached` is not evidence of an edge outside the spectrum: no fallback in 15c; 15d
  applies the f₁ < λ₁ rule and the λ_hi check up front, and the capacity error names both causes.
- The merge routine applies the roadmap's 0.5 projected-norm rule per direction of a leaf's
  window subspace, then Rayleigh–Ritz; a failing window is widened once, then flagged.
- The count reuses the slice's preconditioner (built at the slice's σ, not at $c_{hi}$; only the
  PCG iterations depend on it, recorded per leaf); a span leaf builds `make(c_hi)`.
- A flagged or unverifiable leaf is not split or rerun beyond 15b's restarts.
