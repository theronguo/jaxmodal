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

Revision 3 (plan review round 2): the span of a full slice is decided by si_slice's own
completeness rule on the run's final Ritz values (every Ritz value in it and a neighbour beyond
each cut converged), not by gaps among converged values, which lost modes in the round-2
prototypes; a side whose edge is not reached and has no cut becomes a remainder holding the
converged pairs (binary split on that side); a flagged span leaf is requeued as a run; the span
leaf's count reuses the parent's preconditioner through a helper factored out of `si_checked`;
`verify` is a bool in 15c (the callable moves to 15d with its only user); an edge above the
spectrum has a defined error path; deflation chains at the slice scale; seed stride; a chunked
merge; test 2 on configurations measured under these rules; corrected Facts.

Revision 4 (plan review round 3): si_slice's completeness check takes the deflated W modes as
converged candidates (a W mode can be the neighbour beyond an edge), and every deflation window
holds the nearest stored pair beyond each edge, so a remainder whose modes are all known closes on
known neighbours instead of re-converging distant ones (the round-3 prototypes raised "no
progress" on both planned split tests without it, and passed with it); the span grows from the
converged value nearest σ, its sides are the halves of the converged run, and each cut is the
largest qualifying gap within its half (keeps the remainder's capacity, bounds the count degree);
progress is a new pair inside the slice or a newly completed side; a remainder runs at its
midpoint; one capacity message for all three paths of an edge outside the spectrum; test
configurations chosen on the implementation and checked at ± one chunk and a second seed; the
reference docs of `si_slice`/`si_checked` and the roadmap note in the docs list.

Revision 5 (plan review round 4): a MINRES false-stagnation fix (a seed-dependent "inner solves
failed" of plain si_slice, measured, diagnosed and prototyped below); cuts only between run
values inside the slice (a cut on the edge gave zero-width spans and identical reruns); progress
defined
against the parent; static run halves; a W neighbour sets si_slice's `seen`; a capped deflation
margin; test 2's robustness property restated (match or raise, never a short result) with the
measured candidates; the roadmap stop-rule text and si_checked's neighbour sentence in the docs
list.

## Split of the remaining PR 15 work

15c as planned in 15a holds two separable parts. This plan splits it (the 15a split's rule: every
roadmap test assigned, none dropped):

- **15c (this plan): the slicing engine** `si_band`: a band $[\lambda_{lo}, \lambda_{hi}]$ solved by
  `si_checked` slices (15b) within a given basis capacity, full slices reduced to a verified span
  and remainders, deflation between slices, the merge of clusters held by two runs (duplicates,
  Rayleigh–Ritz, the final re-check against the band-wide $r_{acc}$, the slice-collapse flag),
  per-leaf counts.
- **15d: strategy and API**: the cost model (the slice model measured below, 15a's MINRES cost
  against depth, 14b's LOBPCG step time, the counting cost, the wide-slice count factor below),
  the memory budget (capacity from `si_memory`, the LOBPCG-from-0 basis limit, the host term of
  the found modes) and the strategy choice, `verify=None` for bands (a per-leaf callable on the
  predicted count cost, recorded by 15c), `solve_modes(..., band=(f1, f2))` with the f₁ < λ₁
  rule (solve [0, f₂] and filter) and the check of $\lambda_{hi}$ against the spectrum bound up
  front, (0, f_max) forced onto shift-invert through `solve_modes`, `solve_modes_memory` for
  bands, the README example, the roadmap's PR 15 docs item (memory model for many modes,
  cost-model calibration).

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
| Band edges: an edge above the largest computed eigenvalue raises | 15c tests 3, 11 (in `si_band`); 15d (the up-front bound) |

## Facts

- `si_slice` (15a) declares a slice done when every Ritz value of its `inside` set ($[\lambda_{lo}(1
  - 100r), \lambda_{hi}(1 + 100r)]$ extended over chains of relative gaps $< 100r$, $r$ = its
  $r_{acc}$) and the nearest Ritz value beyond each edge (none below at $\lambda_{lo} = 0$) have
  converged (`_slice_check`); unconverged Ritz values are the sign of modes not yet found. It
  returns the converged inside pairs and the neighbours; W is deflated and never returned. Its
  device basis is $Q$: $(n, \lceil n_W/c\rceil c + \lceil \texttt{capacity}/c\rceil c)$. On
  `capacity_full` it returns the converged inside pairs found so far with `converged` False and
  `reason` "capacity full" (distinct from "inner solves failed"); `edge_unreached` names the
  edges whose neighbour never converged, the normal state of a slice too wide for its capacity
  (every full run in the round-1 probes reported it with both edges deep inside the spectrum), so
  it is no evidence of an edge outside the spectrum.
- Converged values of a full run can have unconverged Ritz values (missing modes) between them:
  in the round-2 prototypes a near-degenerate pair (4.697, 4.702) had one copy converged, and a
  cut placed by gaps among converged values alone dropped the other; one run of modes 20–29
  converged only 24.274 and 24.285 above σ while 22.48–22.76 below σ had not converged.
- **MINRES false stagnation** (round 4, tests prototype; reproduced): plain `si_slice` on
  "short" [32.120, 39.101], capacity 72, seed 0 (σ = 35.61, nearest eigenvalue 35.561) stops with
  "inner solves failed": at step 2 one column is flagged stagnated after 90 iterations with
  explicit residual 1.3 (τ_eff 1.3), while the other five converge in 300–310. With the
  stagnation exit disabled that column converges in 300 like the others (11 steps, τ_eff 1e-8).
  Its history (an unrolled MINRES on that column): explicit residual 1.29–1.33 ‖b‖ and the
  recursive one 0.95–0.87 for the first 100 iterations (a plateau before any progress over
  x = 0, the Krylov space still resolving the eigenvalues near σ), then 0.64 at 120, 9e-3 at 210,
  5e-9 at 300. `minres` (`linalg/krylov.py`) declares stagnation after STALL_C = 5 checks (50
  iterations) without a new explicit minimum, which fires inside the plateau. Seeds 1, 2 or
  σ = 35.0 run normally. 15c creates many slices at arbitrary midpoints, so the false exit would
  stop bands. Prototype fix (below): no stagnation while the best explicit residual is still
  ≥ ‖b‖; the repro then converges (11 steps, τ_eff 1e-8) and all 23 `tests/test_krylov.py`
  tests pass unchanged (σ-on-an-eigenvalue stagnation, the 3-eigenvalue freeze, stagnation
  before maxit).
- `si_checked` (15b) adds the count with up to 2 restarts (seeds `seed + j`); the caller's W modes
  inside $[\lambda_{lo}, \lambda_{hi}]$ or the returned span count as found; its count edges come
  from the found sets (`_beyond`). An unconverged slice gets no count ("slice not converged").
- **Slice cost, measured** (plan review round 1; hex20 8×6×5 clamped, 3720 free DOF, AMG,
  τ = $10^{-8}$, tol $10^{-8}$, capacity never full, each later slice deflating the earlier
  slices' modes; MINRES in column iterations, block iterations × b = 6):

  | band | slices | Lanczos columns | MINRES column iterations | time |
  |---|---|---|---|---|
  | modes 20–60 | 1 | 156 | 40.0k | 45 s |
  | modes 20–60 | 2 | 240 | 65.5k | 55 s |
  | modes 20–60 | 4 | 402 | 112.7k | 83 s |
  | modes 10–50 | 1 | 150 | 25.3k | 21 s |
  | modes 10–50 | 2 | 234 | 42.4k | 33 s |
  | modes 10–50 | 4 | 414 | 93.8k | 87 s |

  Columns ≈ 1.8 per mode + about 84 per slice (every row within 4%); at tol $10^{-10}$ single
  slices need 54 columns for 2 modes, 90 for 12 and 156 for 40: a slice costs about 50–85 columns
  (8–14 Lanczos steps) before it converges anything. 15a's "6.6 columns per mode" was one 20-mode
  slice, overhead included. MINRES iterations per step follow the depth of σ (15a), so the cost of
  a band is about (columns) × (iterations per step): one slice is about 1.65× cheaper than 2 and
  2.8–3.7× cheaper than 4 here. Reorthogonalization ($4nLb$ flops per step) is about $10^{-4}$ of
  a step's MINRES at depth 1000 and $L = 1300$, 0.2% at depth 100, a few % only at depth ≤ 20. So
  the widest slice that fits the memory is the cheapest, splits are worth it only when memory
  forces them, and each extra slice costs its 8–14-step overhead (35–55 h near depth 1000 at 1M
  DOF on the Mac).
- **Small capacities, measured** (rounds 2–3, mesh "short", tol $10^{-10}$, block 6, chunk
  $c = 24$, emulations of the rules with real si_slice runs): the band of modes 8–21
  ([10.46, 22.63], 14 modes) at capacity 72 converges 8 pairs, then a remainder deflating 4 of
  them runs at $72 - 24 = 48$ and raises "no progress" under every variant; capacities 80–120
  round to ≥ 96 and need one slice. Capacities act in multiples of $c$. Under revision 3's rules
  the band of modes 5–28 at capacity 96 and modes 8–21 at block 2, capacity 40, raised "no
  progress" in a remainder holding only a known mode (19.39): its neighbours (16.34, 22.48, each
  half of a pair 1.3e-3 apart) were stored pairs outside its window, reappearing as unconverged
  Ritz values; with a W neighbour counted as converged (round 3, correctness prototype) they
  finish in 2 and 3 runs, all modes, error ≤ 2e-13. The outcome of small-capacity configurations
  flips with rule details, so test configurations are chosen on the implementation.
  Two converged adjacent slices are M-orthogonal to 1e-13 (an edge in a 10% gap).

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

- `make(sigma)`: the SPD preconditioner at σ, as `solve_modes`'s `make`. The reference is dropped
  (`P = None`) once the slice and its span count (below) are done, before the next `make`, so one
  hierarchy is alive at a time. Counts reuse it (15b's default; decision below).
- `capacity`: the total device basis per slice, deflation included: slice $j$ runs with
  $\texttt{capacity}_j = \lceil \texttt{capacity}/c\rceil c - \lceil n_{W,j}/c\rceil c$ ($c$ =
  the projector chunk), so every slice's $Q$ stays within the first run's (15b's restart rule then
  keeps restarts within it too). $\texttt{capacity}_j < 2b$ raises the capacity error.
- `verify`: True (count every leaf) or False (none). With False, `info["count"]` still records the
  degree and gap each leaf's count would use (`degree`, `gap`, from `band_edges`/`count_degree`,
  host-only); they exist for 15d's `verify=None` (a callable deciding per leaf on the predicted
  count cost against the slice's solve time, as 14b).
- `edges`: optional provisional interior edges (tests, and callers that know the spectrum),
  used as given: a cluster straddling one is held whole by the first slice processed (si_slice's
  chains) and deflated by the next (windows below), which is the roadmap's snapping to a gap
  without a search. Default: one slice.
- `seed`: slice $i$ (in processing order) uses `seed + (RESTARTS + 1) i` for its start block and
  the count seed likewise, so no slice reuses another's start block or probes (si_checked adds
  the restart index $j \le$ RESTARTS).
- `max_slices`: a backstop (`RuntimeError`).

**Work list.** Slices are processed from a list ordered by λ, lowest first. A store holds every
accepted pair (λ, v, its run, residual, threshold). For slice $[a, b]$ with σ:

1. **Deflation window.** With $r_w = \max(r_s, \texttt{tol})$ ($r_s$ the largest $r_{acc}$ so
   far) and the margin $m_w = \min(1000 r_w, 10^{-2})$ (si_slice's inside set spills by $100 r
   \le 10^{-2}$ at the τ_eff stop): W = the stored pairs in $[a(1 - m_w), b(1 + m_w)]$ extended
   over chains of
   relative gaps $< 100 r_w$ (the slice's own scale), plus the nearest stored pair (with its
   chain) beyond each extended end: the modes the slice would otherwise find again, and known
   neighbours. The margin covers a later slice's larger $r_{acc}$. si_slice counts W modes as
   converged candidates (change below), so a known neighbour closes a side without being
   re-converged, and `_beyond` uses W modes as the count's neighbours. Soundness: a W neighbour
   is no evidence that the slice's own Krylov space reached the edge, but an unknown mode between
   the edge and it is not deflated and appears as a nearer Ritz value, which must converge before
   the side closes (si_slice's own rule); a mode the Krylov space has not yet resolved at all is
   the same risk as in any slice, and the leaf's count is the backstop.
2. **Run** `si_checked(op, free, a, b, P, capacity=capacity_j, W=W, lam_W=..., R=R,
   verify=verify, seed=..., ...)` at σ = the midpoint of $[a, b]$ (si_slice's default).
3. **Outcome** (by si_slice's `reason`, not by `converged`, which is False for both failures):
   - **Converged:** a leaf; its pairs join the store.
   - **Capacity full:** its converged pairs join the store (they passed si_slice's final residual
     check). si_slice's info carries the final Ritz values with the W modes merged in and one
     converged flag per value (a small 15a addition, `info["ritz"]` = (μ, converged); W modes
     True, values never evaluated False). The **run** is the maximal sequence of consecutive
     converged values containing the converged value nearest σ (a chain $< 100r$ is taken whole,
     $r$ = the slice's $r_{acc}$); the cut candidates are the gaps between consecutive run values
     both inside $(a, b)$, $m_0 < \dots < m_{k-1}$, gap $i$ between $m_i$ and $m_{i+1}$; gaps
     $i \le \lfloor (k - 2)/2 \rfloor$ are the lower half, the others the upper half (static; for
     $k = 2$ the single gap is the lower side's). A gap touching a W neighbour beyond an edge is
     never a cut (that side is complete), so a cut lies strictly inside $(a, b)$ with values on
     both sides of it. On each side:
     - **complete:** the run reaches a converged value beyond the slice's edge (si_slice's own
       neighbour or a W mode; the $a$ side when $a = 0$): no remainder; the span extends to the
       edge, with that value as its out-of-band pair;
     - **cut:** otherwise, the largest relative gap $(m_{i+1} - m_i)/m_{i+1} \ge 100r$ in that
       side's half; the cut $s$ is its midpoint, the run value beyond it is the span's out-of-band
       pair, and $[a, s]$ (or $[s, b]$) is a remainder: a run slice deflating the stored pairs of
       its window (the run values between the cut and the run's end, at most half the run, are
       now W). The half keeps most of the run in the span (the remainder's capacity_j) while the
       largest gap in it keeps the count's degree ∝ 1/g low;
     - **no cut:** that side's interval up to the other side's cut or edge becomes one remainder
       (a run with the converged pairs deflated: a binary split; when the other side is complete
       that remainder is $[a, b]$ itself).
     The span leaf $[s_{lo}, s_{hi}]$ exists when both sides are complete or cut: no Lanczos run,
     its pairs are the stored ones inside, it counts as converged. When verify, its count runs
     before `P` is released (`_count_slice` below, with the out-of-band pairs above; λ_lo = 0 uses
     `count`). A span leaf whose count is ≥ 0.5 (verifiable) is requeued as a run slice with W =
     its pairs (15b's restarts then apply); an unverifiable one stays a leaf, not passed.
     - **Raise** when the run made no progress (no new pair inside $[a, b]$, which W deflates so
       it cannot be a stored one, and no side complete that was not complete in the run of the
       slice it came from; a first slice has none complete), or neither side is complete or cut:
       `ValueError("slice [a, b] fills capacity C without converging a gap >= 100 r_acc on
       either side: raise capacity (measured: a slice needs about 50-85 columns before it
       converges anything); lo may be below lambda_1 or hi above the spectrum")`. The same
       message covers $\texttt{capacity}_j < 2b$ and si_slice's free-space stop. Each requeue is
       a strict sub-interval of its parent (cuts lie strictly inside) or the same interval with
       more stored pairs or a side newly complete, so the list terminates; `max_slices` is the
       backstop.
   - **Count flagged** (after 15b's restarts) or **unverifiable**: a leaf; its pairs join the store
     and the band count is not passed; no split.
   - **Inner solves failed:** the band stops with the pairs found (`converged` False, `reason`
     from the slice, prefixed with its interval).

**Edge above the spectrum.** A side is complete only with a converged neighbour beyond it, so
with $\lambda_{hi}$ above the largest eigenvalue the top side never closes: each top remainder
deflates the modes found below and eventually converges nothing new, which raises the capacity
error naming "hi above the spectrum" (or si_slice's free-space stop when the basis reaches the
free DOF count, which si_band reports with the same message). 15d checks $\lambda_{hi}$ against
the λ̂_max bound up front; an edge between λ_max and λ̂_max is caught here, at the cost of the
top remainders' full runs. A lower edge in $(0, \lambda_1)$ is the mirror case (15d's f₁ < λ₁
rule guards it up front).

**Merge of clusters held by two runs** (one routine for duplicates, the roadmap's Rayleigh–Ritz
on merged clusters and the final re-check). After all slices, with $r_{acc}$ (band) = the maximum
over slices: the kept pairs are sorted and split into windows, chains of relative gaps
$< 100 r_{acc}$. A window holding pairs of one run is final. A window holding pairs of two or
more runs (a cluster across the boundary between leaves, where a nominal edge "fails" the
re-check, or a pair found twice) is merged:

1. In run order, $Q$ = the first run's vectors (M-orthonormal); for each next run's vectors $U$:
   $P = U - QQ^\top M U$, `eigh`$(P^\top M P)$, keep the directions $P y_k$ with eigenvalue
   $\ge 0.25$ ($\|p\|_M \ge 0.5$: the roadmap's duplicate rule applied to the subspace, so a
   45°-rotated copy of a cluster pair is recognized), append them to $Q$ (M-orthonormalized);
   the dropped count is `info["duplicates"]`.
2. Rayleigh–Ritz of $(K, M)$ on $Q$: $G_K = Q^\top K Q$, $G_M = Q^\top M Q$,
   `scipy.linalg.eigh(G_K, G_M)`; the window's pairs are replaced by the Ritz pairs and their
   residuals recomputed against the threshold (the largest of the window's thresholds).
3. A Ritz pair above its threshold: the window is widened once to the next gap $\ge 100 r_{acc}$
   on each side (adding the adjacent pairs) and steps 1–2 repeat (roadmap: "Rayleigh–Ritz over
   that edge window is repeated"); still above: `info["edge_rr_failed"]` True.

All products stream in chunks of $c$ columns (`apply_K`/`apply_M` per chunk, $Q^\top M U$ and the
Gram matrices accumulated on the host, as 15a's `_form_v` and 15b's `_project`): in the collapse
case a window can hold every mode of the band, and the merge's device peak stays one chunk.

`info["merged_edges"]` lists the boundaries between consecutive leaves whose window was merged;
`info["collapsed"]` is True when every boundary between leaves (at least one) is merged (the
roadmap's slice-collapse flag: τ_eff made $100 r_{acc}$ exceed the gaps at all of them). The
counts stay valid: each leaf's count deflated its W and its returned span, and the merge keeps the
union span (it drops only directions with $\|p\|_M < 0.5$, copies of kept ones).

**Result.** `lam`, `V`: the kept pairs, ascending, assembled once into a preallocated host fp64
array (per-run blocks freed as copied), M-orthonormal within a run exactly and across runs to
the merge's Rayleigh–Ritz or to $O(r/g)$ at boundaries with gaps $g \ge 100 r$ (1e-13 measured).
`info`: `converged` (every leaf converged, no inner failure), `reason`, `r_acc`, `tau_eff`
(max), `residuals`, `thresholds`, `slices` (one entry per processed slice in order: a run's
`si_checked` info as is, with the vector fields `neighbours` and `ritz` removed from it and its
`runs`, plus `interval`, `kind` "run", `leaf` (False for a full slice replaced by its span and
remainders), `n_new`; a span leaf `{interval, kind: "span", leaf: True, parent, count}` with
`count` in si_checked's count shape), `merged_edges`, `collapsed`, `edge_rr_failed`,
`duplicates`, `count` (si_checked's shape over the leaves, span leaves included: `ran` any leaf
counted, `passed` every leaf counted and passed, `verifiable` all counted leaves verifiable,
`flagged` any, `estimate` the max, `estimates` per leaf, `reason` None or the first failing
leaf's reason prefixed with its interval), `time` (total, slices, merge). Host memory: the store
$8nN$ ($N$ found modes) plus a window copy $8n\,n_{W}$ and the slice's $V$ while it runs, plus
15b's restart copy; 15d's budget subtracts it (as 14b does for the converged modes). `verbose`:
one line per slice (interval, σ, capacity_j, outcome, new modes, span and cuts, count) plus
`si_slice`'s per-step lines.

### Changes to existing code

- `si_slice`: `_slice_sets`/`_slice_check` take `lam_W`: the W values join the Ritz values as
  converged candidates (their residuals are not evaluated), so the inside set and the neighbour
  beyond each edge are formed over Ritz values and W modes; a W neighbour is converged by
  definition, while an unknown mode between the edge and it still appears as a nearer Ritz value
  that must converge, so the completeness guarantee is unchanged. W candidates add no thresholds
  to $r_{acc}$ and are never returned; the gap ≥ 100 r_acc condition applies to a W neighbour as
  to a Ritz one. A side closed by a W neighbour sets `seen` (so `edge_unreached` does not name
  it). `info["neighbours"]` keeps only si_slice's own (Ritz) neighbours (`_beyond` already takes
  the nearer of those and W). `info["ritz"]` = (μ with W merged, converged flags), host arrays
  already computed in its loop. This also helps 15b's restarts, whose W holds the found in-band
  modes.
- `minres` (`linalg/krylov.py`): a column stagnates only when its explicit residual has made no
  new minimum for STALL_C checks **and** its best explicit residual is below $\|b\|$ (some
  progress over $x = 0$). A plateau before any progress is the indefinite Krylov space still
  resolving the eigenvalues near σ, not a stall; such a column runs on to its inner cap at worst
  (si_slice's first-step cap `maxit_inner`, then $10 n_1$), and a capped column still counts as
  bad for si_slice's σ-move and τ_eff rules. A recursive-residual criterion was tried first and
  does not separate the cases: the recursive residual plateaus with the explicit one (0.95 →
  0.87 over 100 iterations). The existing stagnation tests (σ on an eigenvalue, the
  3-eigenvalue freeze, stagnation before maxit) pass unchanged; si_slice's σ-move rule is
  unchanged.
- `si_checked(..., verify=True)`: False runs the slice alone (no count, no restarts);
  `info["count"]` then holds `ran` False, reason "not requested" and the predicted `degree`,
  `gap`. Its count block (band_count with the nearest out-of-band pair on each side, or `count`
  at λ_lo = 0) moves into `_count_slice(op, free, V_c, lam_c, lo, hi, nb_lo, nb_hi, **kw)`, used
  by `si_checked` and by `si_band`'s span leaves.
- Nothing else in 15a/15b changes. An inner failure still stops the band (no seed retry): its
  measured cause is fixed at the source.

## Tests (`tests/test_shift_invert.py`; mesh "short" = hex20 6×5×4 unless noted; references from
the cached dense `eigh`, extended to 60 modes for "short" (2 s); tol $10^{-10}$, τ $= 10^{-8}$
unless noted. Tests that check counts pass a `make(σ)` returning the exact dense
$(K + \sigma M)^{-1}$ on the free DOFs (SPD for σ > 0), so each count runs a few PCG iterations;
others use AMG. Capacities and run counts are measured on the implementation and recorded as
constants.)

1. **One slice:** `si_band` on modes 13–18's band (15b's) with a large capacity equals
   `si_checked`'s result; one leaf, no split, count passed.
2. **Capacity split (roadmap), real runs:** (a) a block-6 configuration (candidate: the band of
   modes 5–28, capacity 96: 2–3 runs, seed-stable in the round-4 emulations); (b) a block-2
   configuration (candidate: modes 8–21, capacity 48, 3 runs, seed-stable; 40, which lost modes 20
   and 21 under revision 2's rule, splits with 4–6 runs by seed). The configurations are chosen
   on the implementation. Robustness: at ± one chunk of capacity and with a second seed the band
   matches the reference (splits allowed) or raises the capacity error, never a short or wrong
   result (a chunk is about a quarter of a slice's need here, so −1 chunk may raise). At the
   chosen capacity and seed: at least one full slice replaced by a span and
   remainders, every leaf converged (span leaves included), the result matches the reference
   (count, λ to max(1e-8, 10 ε λ̂_max/λ) for gaps ≥ 1e-2, else 10 r_acc), $\|V^\top M V - I\|
   \le 10^{-10}$, no duplicates, every counted leaf's count < 0.5 (roadmap "each slice's count"),
   the recorded number of runs (per seed); a spy checks $\lceil \texttt{capacity}_j/c\rceil c +
   \lceil n_{W,j}/c\rceil c \le \lceil \texttt{capacity}/c\rceil c$, that a remainder's W holds
   the stored modes of its window and its known neighbours, and records the smallest remainder
   capacity_j (measured 0.75–0.83 of the capacity; asserted ≥ half for the chosen
   configurations only, not a design bound). (c) modes 8–21 at block 6, capacity 72: raises the
   capacity error (5–12 runs by seed; only the raise is asserted, the time recorded). (d) The
   MINRES regression: plain `si_slice` on [32.120, 39.101], capacity 72, seed 0 converges (it
   stopped with "inner solves failed" before the fix); plus a `tests/test_krylov.py` case: an
   indefinite system with an initial plateau above $\|b\|$ longer than STALL_C checks (a
   constructed diagonal spectrum, or the slice's step-2 column rebuilt from the run) converges
   and is not flagged.
3. **Fakes for the outcome rules** (a fake `si_checked` returning chosen pairs, Ritz values and
   converged flags): an unconverged Ritz value between two converged values below σ puts the cut
   above it (no mode claimed past it); an unconverged value nearest σ with new converged pairs
   elsewhere grows the run from the converged value nearest σ (no raise); a W mode next to an
   unconverged Ritz value: the run stops at the unconverged value; the halves are the static
   split (k = 2: the gap is the lower side's); the cut is the largest gap within its half, not
   the run's overall largest; a qualifying gap whose midpoint is on the edge (a W neighbour
   beyond it) is not a cut; a rerun of the same interval with no new pair and no newly complete
   side raises; a complete side (neighbour converged) closes the span at
   the edge with that neighbour as `lam_out`, likewise with a W mode as the neighbour; an
   unreached side without a cut becomes one remainder holding its converged pairs as W; one side
   complete and the other without a cut reruns $[a, b]$ with W; a remainder holding only W modes
   with known neighbours converges at once and becomes a leaf; a full slice with no new pair
   inside $[a, b]$ and no newly completed side raises;
   neither side complete or cut raises; a span leaf whose count is flagged is requeued as a run
   with W = its pairs; `edge_unreached` "both" with valid cuts splits (no raise); `max_slices`
   exceeded raises; an inner failure stops the band with the pairs found, `converged` False and
   the reason; an unverifiable (capped) and a flagged run leaf are not split, `count["passed"]`
   False, `flagged` as reported; a replaced parent's missing count does not fail `passed`; the
   span leaf's count calls `_count_slice` with the stored pairs and the out-of-band pairs beyond
   its cuts, with the parent's preconditioner; the aggregate `count` includes span leaves; a top
   side never closing (no neighbour beyond hi) raises with "above the spectrum", a bottom side
   never closing (lo > 0, nothing below) with "below lambda_1".
   **si_slice with W neighbours** (real, "short"): 15b's modes 13–18 slice with W = modes 11–12
   and 19–20 (the neighbours) converges with the W modes as neighbours (`neighbours` None on both
   sides, `_beyond` picks W, `edge_unreached` None), the same pairs as without W to 1e-10, in no
   more steps; an unknown mode between the edge and a W neighbour (W = mode 20 only, mode 19
   unknown) is still converged as the Ritz neighbour.
4. **Voxel cube across slices (roadmap):** the free voxel cube (15a's, with R), a band holding
   its double and triple clusters, `edges` with one interior edge inside a cluster's chain, tol
   $10^{-6}$: the cluster is held whole by the lower slice and deflated by the upper (`collapsed`
   False, no merge, `duplicates` 0), count and multiplicities as the reference, λ within
   max(1e-8, 10 ε λ̂_max/λ) for clusters with relative gap ≥ 1e-2, else 10 tol, M-orthonormal,
   each leaf's count < 0.5. Timed first; slow if over 60 s.
5. **Re-check and collapse (roadmap):** (a) modes 23–24 of "short" (relative gap 4.6e-4), the
   interior edge $s$ (`edges`) at their midpoint; slice 1 $[(\lambda_{21} + \lambda_{22})/2, s]$
   at τ = 1e-8 (2 modes, $r_{acc} \approx$ 4e-9, one copy kept), slice 2 $[s, 50]$ with inner
   solves stopped at $10^{-6}$ (15a's monkeypatch, applied to the second `si_checked` call only)
   so τ_eff = 1e-6 and $100 r_{acc} \approx 6\cdot10^{-4} > g$ (asserted from the run's info
   before the merge assertions; round 2: hi = 45 gives 5.2e-4, 40 too little); slice 2 returns
   mode 23 again: the window across the boundary holds vectors of both runs, it is merged,
   `duplicates` 1, `merged_edges` lists the boundary, `collapsed` True (the only boundary), the
   result matches the reference within 10 r_acc and is M-orthonormal; (b) fake leaves with
   reference pairs: a cluster split across the boundary with rotated vectors is restored by the
   Rayleigh–Ritz to 1e-12; a window whose Ritz residual stays above the threshold after widening
   sets `edge_rr_failed`; two boundaries of which one merges: `collapsed` False; a window wider
   than $c$ goes through the chunked products (matches a dense RR).
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
   45°-rotated double (both copies from the second run, one copy kept from the first) → 2
   vectors kept, exact pairs, M-orthonormal; a near copy $\cos\theta\,v_1 + \sin\theta\,v_2$
   ($\theta = 10^{-3}$, $v_1, v_2$ kept) → dropped, no mode lost; a genuinely new vector in the
   window (projected norm ≈ 1) → kept; M-orthonormality asserted after each.
10. **Options and errors:** `verify=False` (`si_checked` called with it, `count["ran"]` False,
    the predicted `degree` and `gap` present and equal to a counted run's); seeds
    `seed + (RESTARTS + 1) i` per slice (spy); argument errors (lo ≥ hi, edges outside (lo, hi)
    or unsorted, capacity below 2b after deflation).
11. **Edge above the spectrum, real:** a tiny clamped mesh (free DOF below the capacity), hi
    above its largest eigenvalue: raises the capacity error naming "hi above the spectrum"; the
    path taken (no progress, capacity_j < 2b or the free-space stop) is recorded in the test.

Budget: tests 1, 3, 5b, 6, 8–11 fast (seconds; fakes where the logic is the subject); 2, 5a and 7
run real slices (round-2 probes: 4–13 s per run) and stay fast if under 60 s each; test 4 is
timed first.

## Docs

- `docs/eigensolvers.md`, "Shift-invert slices": a "Slicing a band" subsection (the work list,
  deflation windows, capacity per slice, full slices reduced to a span decided by si_slice's
  completeness rule and remainders, the measured slice cost model and why the width is
  memory-limited, the merge routine, the re-check and collapse flag, `verify`, the edge above the
  spectrum and its cost, `info`, host memory); the existing reference text: si_slice's
  completeness rule with W modes as converged candidates and its info list gaining `ritz`,
  si_checked's signature gaining `verify`, its `count` info gaining the predicted `degree`/`gap`
  and reason "not requested", one line on `_count_slice` as the shared count step, si_checked's
  "no converged neighbour … raises" sentence (si_slice's or a W mode); the MINRES stagnation
  rule in docs/preconditioners.md and the "Inner tolerance" bullet of eigensolvers.md; the
  Validation list (the new si_slice W-neighbour and si_checked verify tests and the si_band
  tests); the stale "15c" pointers that now mean 15d
  (strategy choice, `band=(f1, f2)`, capacity from `si_memory`, the cost model on depth, "the
  choice 15c makes": lines near 122, 589, 607, 614, 643, 663, 666, 676).
- `docs/ROADMAP.md`: the Interior-bands stop bullet amended ("PR 15c: a known deflated mode
  beyond the edge, at a gap ≥ 100 r_acc, also serves as the neighbour; an unknown mode between
  still appears as a nearer Ritz value that must converge"); the PR 15 split sentence rewritten
  for 15c/15d; the [15c] tags split into
  [15c]/[15d]; the "Slice widths come from the cost model …" bullet and the Contents bullet
  "Slicing with the cost model" amended with a note (widths limited by the capacity; edges come
  from full slices: the span si_slice's completeness rule verifies on the run's final Ritz
  values, cut at the largest gap ≥ 100 r_acc within each half of its converged run (the snapping
  rule), a binary split on a side without a cut; leaves keep si_slice's widened inside sets),
  worded as in the open_items decision; the PR 15 docs item moved to 15d; the Dependencies table
  ("15c Slicing | 15a, 15b", "15d Band strategy, `band=(f1, f2)` | 14b, 15c"); the PR 15c status.
- `docs/open_items.md`: the decisions below; the "Shift-invert cost near mode 1000" and "σ-move
  retries" items and the 15c references in the PR 15a list retagged to 15d; in the PR 15b list,
  "σ = c_hi" restated (15c's slices are as wide as memory allows, so the lower edge's
  $(1 + a)/(2\sqrt a)$ degree factor is paid by wide leaves; recorded per leaf, an item for 15d's
  cost model), "15c may pass `count_precond`" and "15c decides whether a flagged slice splits"
  pointed at the decisions below.
- README: status line (15c; the `band=(f1, f2)` example with 15d).

## Decisions without user input (to log)

- 15c split into 15c (slicing engine) and 15d (strategy, API, memory, `verify=None`), all roadmap
  tests assigned.
- No provisional interior edges from the cost model: the measured slice cost (1.8 columns per
  mode plus 50–85 per slice) makes the widest slice that fits the cheapest; slices come from full
  slices reduced to their span and remainders, the converged span kept as a leaf without a second
  run. `edges` is an internal option, used as given (si_slice's chains and the deflation windows
  assign a straddling cluster whole).
- Edges come from full slices: the span si_slice's completeness rule verifies on the run's final
  Ritz values, cut at the largest gap ≥ 100 r_acc within each half of its converged run (the
  roadmap's snapping rule), a binary split on a side without a cut; a flagged span leaf is rerun.
- The cut is the largest qualifying gap within the half of the converged run on its side, not
  the run's overall largest (the roadmap says "largest gap"): a cut deep in the run would move
  up to half the run into the remainder's W, lowering its capacity_j (one chunk per $c$ pairs;
  it caused the round-2 "no progress"), while the outermost gap can be tiny and the count's
  degree is ∝ 1/g. The half bounds the first, the largest gap in it the second.
- si_slice counts deflated W modes as converged candidates (a W mode can close a side), and
  deflation windows include the nearest stored pair beyond each edge.
- Remainders run at their midpoint (si_slice's default σ); placing σ from stored values pulled
  it to an edge in the round-3 prototypes.
- Leaves keep si_slice's widened inside sets (margins and straddling chains) rather than only the
  pairs inside their own interval; deflation windows and the merge routine handle the overlap.
- Deflation windows: margins at $\min(1000 r_w, 10^{-2})$, chains at $100 r_w$, $r_w =
  \max(r_s, \texttt{tol})$; `capacity` is the total basis per slice.
- MINRES does not stagnate before its explicit residual drops below $\|b\|$ (a plateau, not a
  stall), fixed in 15c where it surfaced; an inner failure stops the band without a seed retry.
- `edge_unreached` is not evidence of an edge outside the spectrum: no fallback in 15c; 15d
  applies the f₁ < λ₁ rule and the λ_hi bound up front; an edge in (λ_max, λ̂_max] raises here,
  and the capacity error names both causes.
- The merge routine applies the roadmap's 0.5 projected-norm rule per direction of a run's
  window subspace, then Rayleigh–Ritz; a failing window is widened once, then flagged.
- Counts reuse the preconditioner of the slice they belong to (a span leaf: its parent's, before
  it is released; built at the slice's σ, not at $c_{hi}$; only PCG iterations depend on it,
  recorded per leaf).
- A flagged or unverifiable run leaf is not split or rerun beyond 15b's restarts.
- `verify` is a bool in 15c; the per-leaf callable comes with 15d's `verify=None`.
