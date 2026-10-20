# PR 18g: slice efficiency (plan)

Branch `pr-18g-slices`. Roadmap section "PR 18g: Slice efficiency" (`docs/ROADMAP.md`), the open items "Slice efficiency (PR 15c → PR 18e → PR 18f → PR 18g)" (`docs/open_items.md`). Rev 2 (after plan review round 1).

## Goal

1. **A slice's own modes** (the main item). PR 18f deflates each slice's MINRES by the stored modes below σ. The slice's own modes, the nearest to σ and so the most expensive, stay undeflated. Probe (2026-10-20; folded into `benchmarks/slice_study.py`, part `probe`): the deep band's slice of PR 18f (hex20 $16^3$, AMG, σ = 443.93, the cut at 434.72, 957 known modes below it, 22 own modes in [cut, σ), 29 in (σ, hi]), one solve of 6 columns at τ = $10^{-8}$, the vectors from the cached LOBPCG modes:

   | $Z$ | $n_z$ | MINRES iterations | s per iteration |
   |---|---|---|---|
   | the 957 known (PR 18f) | 957 | 840 | 0.068 |
   | + the 11 farthest own below σ | 968 | 640 | 0.068 |
   | + the 11 nearest own below σ | 968 | 500 | 0.068 |
   | + all 22 own below σ | 979 | 340 | 0.068 |
   | + the 29 own above σ only | 986 | 460 | 0.068 |
   | + all 51 own, both sides | 1008 | 120 | 0.068 |
   | + the 22 below, perturbed $10^{-3}$ | 979 | 340 | 0.069 |

   Deflating the own modes on both sides of σ cuts a solve to 0.14×, at the same cost per iteration. Vectors with errors of $10^{-3}$ deflate as well as exact ones. PR 18f's slice ran 810–920 iterations per step over 37 steps, about 31,600 in total (1839 s, 36 s per mode).
2. **The deferred slice items** (PR 15c/15d → 18g): verification slices (remainders that find at most one new mode in 10–15 steps), a top remainder holding only known modes ("no progress"), the whole-slice requeue. They are measured first and changed only where the study shows a gain (below).

## Design

### 1. Own-mode deflation in si_slice

- **Slots:** `si_slice(..., own=0)`: the number of own columns, appended after the known `Z` in one block of fixed padded width $\lceil (\mathrm{len}(Z) + \mathrm{own})/c \rceil c$. The block is allocated and the known columns uploaded once, at setup, as today. The own slots start zero, so `own = 0` is today's behaviour. `own` must be an int ≥ 0 (ValueError otherwise). With `own > 0` the deflated MINRES runs from the first step, also for `Z=None`: zero own columns give $H = 0$, so the first solves are plain MINRES. No switch happens mid-run.
- **Which pairs:** after each kept step that does not end the loop, the own pairs are the candidates `_slice_check` evaluated (`cres`: inside pairs and the neighbours, both sides of σ) with relative residual $\le$ `OWN_RES` = $10^{-3}$, nearest $|\mu - \sigma|$ first, at most `own`. They are selected before acceptance: the deflation is exact for any $Z$ (PR 18f; checked again in review: $b - Ax = P_Z(b - A\tilde x)$ for any $Z$, and a rank-one deflation cannot push the spectrum outside $A$'s range). So the own set only changes the MINRES iterations, never the Lanczos basis beyond the inner tolerance, the Ritz values or the returned pairs. A pair accepted later, or never, costs nothing in correctness. A below-lo neighbour that is also a stored known column, or a near duplicate, drops out in the pinv (rcond $10^{-10}$, as PR 18f).
- **Rebuild:** the own vectors are formed on the device ($Q_L Y_{\mathrm{own}}$ through `_combine`, masked like $Q$) and written into the block's own slots in place (the unused slots zeroed). Then $H$ is recomputed (`_deflate_setup` on the real columns). This happens every such step with a non-empty own set, because the Ritz vectors improve each step. No host vectors, no re-upload of the known columns, no second block, one compile (fixed width). Cost per rebuild: $\lceil n_z/c \rceil$ operator applies of width $c$, measured 1.5 s at $n_z$ = 957 against about 50 s per step in the deep slice. A σ move (only before the first kept step, so before any own column) recomputes $H$ as today; later rebuilds use the current A and σ.
- **si_band:** with `max_deflate` = $d_Z$ > 0 it passes `own = min(OWN_MAX, max_deflate)` and the `max_deflate − own` nearest known columns (`_z_cols`, unchanged). The block stays within the plan's $d_Z$. `OWN_MAX` = 64: the deep slice had 51 candidates, and 64 columns add about 7% to the block of 957 there. Known columns given up for own slots are the farthest; the probe shows the own ones are worth far more.
- **si_checked** restarts pass the same `Z` and `own`. Each run forms its own columns afresh. A restart puts the earlier runs' found modes into W, so they are neither own nor known columns in the restart: less deflation, not wrong answers. Restarts are rare; noted in the docs.
- **info:** `deflated` stays the known columns given. New `own_columns` holds, per kept step, the own columns its solve used.

### 2. Memory

Unchanged. The device block is `len(Z) + own` ≤ $d_Z$ columns, as `_slice_plan` already sizes it (the `max_deflate` docstrings say known plus own). Forming the own vectors uses one `_combine` output of width `own` next to Q, below the final stage's `_form_v` chunk plus residual pass already in the model. The host has no new arrays.

### 3. The deferred slice items: measured

`benchmarks/slice_study.py` (results in `benchmarks/results/slice_study.json`; `--parts probe,many,deep`, each part within 1 h):

- **Part `probe`:** the table above, from `deep_band_study`'s LOBPCG cache. About 10 min (one band LOBPCG of 396 s on a cold cache).
- **Part `many`:** hex20 $8\times6\times5$ clamped, AMG, the band of modes 5–200 at capacity 96 (the PR 15c code review case), through `solve_modes(strategy="shift_invert", slice_options={"capacity": 96})`. Per slice it records the interval, kind (first, remainder, verification, requeue), new modes, steps, MINRES iterations and seconds. Runs:
  1. own deflation off (`OWN_MAX` = 0 patched in the study: PR 18f behaviour);
  2. the default.

  A few minutes per run (PR 15c's case: 37 slices of about 10 s each).
- **Remainder σ rule**, implemented and run (run 3) only if run 2's remainders with one done edge take ≥ 30% of the band's MINRES iterations. A work item whose own edges include exactly one done edge (`done ∩ {a, e}`; si_band's `done` also carries the parent's edges) and that has stored modes inside places σ at the midpoint of its unexplored part:
  - for $[a, x]$ with $x$ done, $\sigma = (a + l)/2$ with $l$ the smallest stored value in $(a, x]$;
  - for $[x, e]$ with $x$ done, $\sigma = (u + e)/2$ with $u$ the largest stored value in $[x, e)$;
  - without stored values inside, the midpoint as today.

  si_band builds the preconditioner at that σ (`make(σ)`, keeping the `p_sig` reuse), takes `_z_cols` at that σ, and passes `sigma` to si_checked internally; the user ban on `sigma` in `slice_options` stays. The rule is kept if it lowers the total MINRES iterations by ≥ 15% against run 2, with no more slices and every mode matching eigsh. Otherwise it is removed and the item is closed with the numbers.
- **Top remainder and requeue:** counted in runs 1–2, with the slices and the share of MINRES iterations each takes. Each item is closed if it takes < 10% of the band's iterations; otherwise it is kept in `docs/open_items.md` with the numbers.

### 4. The deep band

Part `deep`: PR 18f's band (`deep_band_study.band`, the cached LOBPCG run, hex20 $16^3$, modes 951–1008) with own deflation. Per step it records the MINRES iterations and own columns, plus the slice's seconds per mode, all against eigsh. About 40 min. PR 18f's slice log (scratch `p18f/study_band.log`) accepted no pair in steps 1–7, which is why the own set is gated by residual, not by acceptance.

## Acceptance

- **The deep band:**
  - every mode against eigsh ($\le 10^{-12}$ relative);
  - the slice's MINRES iterations summed over its steps $\le 0.7\times$ PR 18f's (about 31,600, so ≤ 22,100);
  - seconds per mode $\le 0.75\times$ (36 s, so ≤ 27 s).

  Predicted: about 0.5×, assuming the candidates reach $10^{-3}$ by about step 8 (when PR 18f accepted its first pairs at its thresholds, so earlier for the looser gate; the study logs it). Until then the steps run at PR 18f's cost; the steps after are at 0.15–0.4× (the probe's both-sides and below-only rows). The seconds include the rebuilds (1.5 s per step). If the target is missed but the sum is ≤ 0.85×, the feature is kept and the numbers recorded, with the gate (`OWN_RES`, candidates beyond the neighbours) logged as an open item. Above 0.85×, the gate is revisited before merging.
- **The `many` band:** every mode against eigsh in each run, and the run-2 total MINRES iterations below run 1's.

## Tests

`tests/test_deflation.py`, hex20 slice `edges(lr, 30, 42)` (12 inside, 7 own below σ, 16 steps, about 10 s), `Z` = the modes below lo, slow where it solves:

1. **Exactness and gain:** `own` = 12 against `own` = 0:
   - eigenvalues match to $10^{-10}$ relative and subspaces to small angles, and both match eigsh;
   - `steps` and `basis` are equal;
   - `own_columns` starts at 0 and grows;
   - the mean MINRES iterations per step over the steps after the first own column are below the `own` = 0 run's over the same steps (about 60 against 110 measured with exact vectors).
2. **The own set:** a spy on the own-slot write (and on `_deflate_setup`):
   - every own column's Rayleigh quotient matches an evaluated candidate with residual ≤ `OWN_RES`;
   - the count is ≤ `own`; the block width never changes;
   - with `own` = 2, the two kept are the nearest $|\mu-\sigma|$ (some above σ when the candidates are).
3. **`Z=None`, `own` > 0:** the pairs match eigsh and the plain run, `own_columns` > 0, later steps need fewer iterations. **`own` = 0:** no own write happens and `own_columns` is all 0. **`own` < 0** or a non-int raises ValueError.
4. **Rebuilds:**
   - `_zblock` is called once;
   - `_deflate_setup` is called once at setup plus once per non-final step with a non-empty own set (kills both "never" and "every step including the last");
   - no rebuild after the final step.
5. **σ move:** extend `test_slice_deflated_move` with `own` > 0; every `_deflate_setup` after the move sees the moved operator (spy on the shift).
6. **si_checked:** extend `test_checked_keeps_z`: each run gets the same `Z` and `own`, and each restart's first block has zero own columns.
7. **si_band** (fake slices): with `max_deflate` = $d$ ≥ `OWN_MAX`, each slice gets `own` = `OWN_MAX` and the $d$ − `OWN_MAX` nearest known columns; with $d$ < `OWN_MAX`, `own` = $d$ and no known columns; with $d$ = 0, `own` = 0.
8. If the remainder σ rule is adopted: si_band passes the rule's σ per work item (fake slices, crafted stores, a remainder of a remainder whose `done` carries a parent edge), and the `many` band end to end against eigsh (slow).

Mutations to try in code review: the residual gate off (all candidates), the gate on acceptance only, the nearest ordering reversed, the slots not zeroed, a rebuild on the last step, stale A after a move, `own` ignored in si_band.

## Docs

- **README:**
  - line 5 (status, order: 18g, then 18h);
  - line 287 ("deflated by the known modes below its shift" → known and own modes, the numbers).
- **`docs/eigensolvers.md`:**
  - the MINRES deflation section (own columns, the gate, the slots, the rebuild, the restart note);
  - si_slice's options and info (`own`, `own_columns`);
  - `max_deflate` (known plus own);
  - the measured section (the study's tables: the probe, `many`, `deep`), replacing PR 18f's deep-band numbers where superseded;
  - the stale "PR 18g" forward references (lines about 153, 157, 196, 829, 887);
  - the test list.
- **Docstrings:**
  - si_slice (`own`, `own_columns`);
  - si_band and solve_modes (`max_deflate` counts known plus own; `own` stays internal: si_band rejects it in `slice_options`, as it does `Z`);
  - `_slice_plan` and `si_memory` (the block is known plus own).
- **`docs/ROADMAP.md`:**
  - the 18g section's Contents and Tests rewritten to this scope, then its status and results;
  - the status paragraph and the order lines (about 18, 92, 94);
  - "Deferred GPU checks": the own rebuild's $H$ setup per step at 10M DOF (applies plus the $n_z \times n_z$ pinv on the host), the incremental $E$ update if it shows.
- **`docs/open_items.md`:**
  - the slice-efficiency items closed or updated with the numbers;
  - the PR 18f residual-risk notes (about lines 155 and 335) updated;
  - the decisions below.

## Decisions without user input

- Own columns are gated by residual ($10^{-3}$), not by acceptance, on both sides of σ: the probe shows both sides matter (0.14× against 0.4×) and perturbed vectors deflate as well; the deflation is exact for any vectors.
- Fixed own slots (`OWN_MAX` = 64) inside the plan's $d_Z$, taken from the farthest known columns: one block, one compile, no memory-model change, no host vectors.
- $H$ is recomputed in full per rebuild (1.5 s against about 50 s per step); an incremental $E$ (only the own rows) waits for a measured need (GPU checks).
- The remainder σ rule runs only if remainders are costly (≥ 30% of iterations), and is kept by the numbers (≥ 15% fewer), not by argument.
