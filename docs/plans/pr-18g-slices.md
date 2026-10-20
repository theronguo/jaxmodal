# PR 18g: slice efficiency (plan)

Branch `pr-18g-slices`. Roadmap section "PR 18g: Slice efficiency" (`docs/ROADMAP.md`), the open items "Slice efficiency (PR 15c → PR 18e → PR 18f → PR 18g)" (`docs/open_items.md`).

## Goal

1. **A slice's own modes below σ** (the main item). PR 18f deflates each slice's MINRES by the stored modes below σ, but a slice's own modes below σ stay undeflated, and they are the nearest, so the most expensive. Probe (2026-10-20, scratch `p18g/own.py`): the deep band's slice of PR 18f (hex20 $16^3$, AMG, σ = 443.93, the cut at 434.72, 957 known modes below it, 22 of the slice's own modes in [cut, σ)), one solve of 6 columns at τ = $10^{-8}$ with the exact eigsh vectors:

   | $Z$ | $n_z$ | MINRES iterations | s per iteration |
   |---|---|---|---|
   | the 957 known (PR 18f) | 957 | 840 | 0.0681 |
   | + the 11 farthest own | 968 | 640 | 0.0679 |
   | + the 11 nearest own | 968 | 500 | 0.0682 |
   | + all 22 own | 979 | 340 | 0.0682 |

   So deflating the own modes cuts the later steps' solves to about 0.4× at the same cost per iteration. PR 18f's slice ran 810–920 iterations per step over 37 steps (1839 s, 36 s per mode).
2. **The deferred slice items** (PR 15c/15d → 18g): verification slices (remainders that find at most one new mode in 10–15 steps), a top remainder holding only known modes, the whole-slice requeue. Measured first, changed only where the study shows a gain (below).

## Design

### 1. Own-mode deflation in si_slice

- **When:** after each kept Lanczos step, the step's accepted inside pairs (`last[2]`: converged by the stop rule's thresholds) with $\mu < \sigma$ are the own pairs. When their number grows, the deflation block is rebuilt before the next step's solve; otherwise it is kept. The Ritz vectors are formed from the current basis (`_form_v(Q, L0, Yr[:, own], c)`, host fp64, masked like `Q`).
- **Exact:** the A-orthogonal deflation is exact for any $Z$ (PR 18f), so the Lanczos basis, the Ritz values and the returned pairs are those of the undeflated run up to the inner tolerance; only the MINRES iterations change. A Ritz vector that later moves (it was accepted, so it is converged to its threshold) only weakens the deflation, never the solve.
- **Cap:** `si_slice(..., max_deflate=None)`: the total deflation columns, known plus own. None (the default) means `len(Z)`: no own columns beyond the given ones, today's behaviour. With a cap, the block is the own pairs (nearest σ first, at most `OWN_MAX` = $2c$) plus the nearest known columns that still fit: the known `Z` is ordered by ascending eigenvalue (farthest first) and its first columns are dropped. si_band passes its `max_deflate` (solve_modes: the plan's $d_Z$), so a slice whose store has fewer than $d_Z$ modes below σ (e.g. the first slice from 0 or from lo) deflates its own modes in the room left.
- **Rebuild:** the old device block and $H$ are dropped (`Zd = H = None`) before the new upload (`_zblock`) and setup (`_deflate_setup`): no two blocks coexist. Cost: one upload and $\lceil n_z/c\rceil$ operator applies of width $c$ (1.5 s at $n_z$ = 957, 56k DOF, against about 50 s per step there); at most once per step that accepted a new own pair. A σ move recomputes $H$ as today. One compile per padded width (as PR 18f).
- **si_band's `_z_cols`** returns the nearest $n$ indices ordered by ascending eigenvalue (today: ascending store index), so si_slice can drop the farthest first.
- **si_checked** restarts pass the same known `Z` and cap; each run builds its own pairs afresh.
- **info:** `deflated` stays `len(Z)` (the known columns given); new `own_deflated` (the own columns in the last step's block) and `z_columns` (per kept step, the block's columns used by its solve).

### 2. Memory

- Device: unchanged, the block stays within the cap ($d_Z$ in `_slice_plan`).
- Host: the own vectors, $8n \cdot \min(\texttt{OWN\_MAX}, d_Z)$, added to `si_memory`'s host term when `n_z` > 0 (exact). `_slice_plan` needs no other change.

### 3. The deferred slice items: measured

`benchmarks/slice_study.py`, part `many`: hex20 $8\times6\times5$ clamped, AMG, the band of modes 5–200 at capacity 96 (the PR 15c code review case), `si_band` through `solve_modes(strategy="shift_invert", slice_options={"capacity": 96})`: per slice the interval, kind, new modes, steps, MINRES iterations and seconds; three runs:

1. `max_deflate` = 0 (PR 18e behaviour: no deflation);
2. the default (PR 18f deflation plus own modes);
3. 2 plus the **remainder σ rule** below.

**Remainder σ rule** (candidate): a work item with exactly one done edge (a remainder of a split) and stored modes inside places σ at the midpoint of its unexplored part: for $[a, x]$ with $x$ done, $\sigma = (a + l)/2$ with $l$ the smallest stored value in $(a, x]$; for $[x, e]$ with $x$ done, $\sigma = (u + e)/2$ with $u$ the largest stored value in $[x, e)$; without stored values inside, the midpoint as today. Adopted when run 3 against run 2 lowers the band's total MINRES iterations by $\ge 15\%$ with no more slices and every mode matching eigsh; otherwise not implemented, and the open items are closed with the study's numbers. The top-remainder and requeue items are recorded from the same runs (how often, at what cost) and closed or kept with the numbers.

### 4. The deep band

Part `deep` of the study: PR 18f's band (`deep_band_study.band`, the cached LOBPCG run, hex20 $16^3$, modes 951–1008) with own deflation: per step the MINRES iterations and $z$ columns, the slice's seconds per mode, against eigsh. About 40 min (LOBPCG 396 s, the slice).

## Acceptance

- The deep band: every mode against eigsh ($\le 10^{-12}$ relative), the slice's MINRES iterations summed over its steps $\le 0.7\times$ PR 18f's, seconds per mode $\le 0.7\times$ (36 s).
- The `many` band: every mode against eigsh in each run; the run-2 total MINRES iterations below run 1's.

## Tests

1. `tests/test_deflation.py`: a slice above 30 modes of "hex20" with `Z` = the modes below lo and `max_deflate` = 30 + 12: the pairs match the `max_deflate=None` run and eigsh; `z_columns` grows over the steps; the summed MINRES iterations after the first own rebuild are below the run without own columns; `own_deflated` > 0.
2. The cap: `max_deflate` = len(Z) + 2 with a spy on `_zblock`: each block holds at most the cap, the own columns first-built nearest σ, the farthest known columns dropped (by eigenvalue).
3. The rebuild frees the previous block (weakref on `_zblock`'s output, as `test_zd_freed`).
4. `_z_cols` ordering by eigenvalue (updated `test_z_cols`), si_band passing `max_deflate` to each slice (fake slices).
5. `si_memory`'s host term with `n_z` > 0 (`test_memory_formulas`).
6. If the remainder σ rule is adopted: si_band passes the rule's σ per work item (fake slices, crafted stores), and the `many` band end to end against eigsh (slow).

## Docs

- `docs/eigensolvers.md`: the MINRES deflation section (own pairs, the cap, the rebuild), si_slice's options and info (`max_deflate`, `own_deflated`, `z_columns`), the slice memory model (the host term), the measured section (the study's tables), the test list.
- `docs/ROADMAP.md`: the 18g section's status and results, the status paragraph at merge; `docs/open_items.md`: the slice-efficiency items closed or updated with the numbers, decisions without user input; README if user-visible behaviour changes (none planned: options stay internal to si_band/si_slice).

## Decisions without user input

- Own pairs only once accepted (not merely small residuals): the deflation stays tied to the stop rule's thresholds.
- `OWN_MAX` = $2c$ bounds the host vectors; the probe's slice had 22 own modes below σ.
- The remainder rule is decided by the study's numbers, not by argument.
