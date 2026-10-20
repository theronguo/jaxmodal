# PR 18g: slice efficiency (plan)

Branch `pr-18g-slices`. Roadmap section "PR 18g: Slice efficiency" (`docs/ROADMAP.md`), the open items "Slice efficiency (PR 15c → PR 18e → PR 18f → PR 18g)" (`docs/open_items.md`). Rev 3 (after plan review rounds 1–2).

## Goal

1. **A slice's own modes** (the main item). PR 18f deflates each slice's MINRES by the stored modes below σ. The slice's own modes are the nearest to σ, and so the most expensive, and they stay undeflated.

   **Probe** (2026-10-20; folded into `benchmarks/slice_study.py`, part `probe`): the deep band's slice of PR 18f (hex20 $16^3$, AMG, σ = 443.93). The cut is at 434.72, with 957 known modes below it, 22 own modes in [cut, σ) and 29 in (σ, hi]. One solve of 6 columns at τ = $10^{-8}$, the vectors from the cached LOBPCG modes:

   | $Z$ | $n_z$ | MINRES iterations |
   |---|---|---|
   | the 957 known (PR 18f) | 957 | 840 |
   | + the 11 farthest own below σ | 968 | 640 |
   | + the 11 nearest own below σ | 968 | 500 |
   | + all 22 own below σ | 979 | 340 |
   | + the 29 own above σ only | 986 | 460 |
   | + all 51 own, both sides | 1008 | 120 |
   | the nearest 436 known | 436 | 3540 |
   | 436 known + 51 own | 487 | 520 |
   | the nearest 64 known | 64 | 8200 |
   | 13 known + 51 own | 64 | 2030 |

   The cost per iteration is the same in every row (0.068 s). Own columns on both sides of σ are worth far more than the farthest known ones. PR 18f's slice ran 810–920 iterations per step over 37 steps, about 31,600 in total (1839 s, 36 s per mode).

   **Before acceptance** (review round 2, the hex20 test slice `edges(lr, 30, 42)`, 16 steps, 110 iterations per step with the 30 known modes; own columns formed from `_slice_check`'s evaluated candidates each step, nearest $|\mu-\sigma|$ first, at most 12):

   | own columns from | summed iterations | ratio |
   |---|---|---|
   | none | 1720 | 1 |
   | every candidate | 780 | 0.45 |
   | candidates with residual ≤ $10^{-2}$ | 960 | 0.56 |
   | ≤ $10^{-3}$ | 1010 | 0.59 |
   | ≤ $10^{-4}$ | 1130 | 0.66 |

   With every candidate, the gain starts at step 2 (110, 80, 60, 50, then 40 per step). Unconverged Ritz vectors near σ jointly span the near-σ subspace well, and ghost Ritz values (residual about 1) did no harm. A below-lo neighbour that is a stored known mode (step 13, residual about $10^{-4}$) raised that solve from 40 to 70: a near duplicate of a $Z$ column that pinv (rcond $10^{-10}$) keeps, which makes $E$ ill-conditioned (review probe: cond($E$) 2.5e9 at a $10^{-3}$ error).
2. **The deferred slice items** (PR 15c/15d → 18g): verification slices (remainders that find at most one new mode in 10–15 steps), a top remainder holding only known modes ("no progress"), the whole-slice requeue. In this PR they are measured; changes go to a follow-up if the numbers ask for one (below).

## Design

### 1. Own-mode deflation in si_slice

- **Slots:** `si_slice(..., own=0)` takes `own`, the number of own columns. The block is `[Z | own slots]`. Each part is zero-padded to a multiple of $c$, so the own region starts on a chunk boundary. The block is allocated once at setup, the known columns uploaded once as today, the slots initially zero. `own = 0` is today's behaviour. `own` must be an int ≥ 0, not a bool (ValueError, as `max_deflate`'s check). With `own > 0` and `Z=None`, the block exists from the start and has no real column until the first own write. The first solve is the plain MINRES (`solve` checks the current real column count, so no H setup runs on an empty block).
- **Which pairs:** after each kept step, the own set is the Ritz indices `_slice_check` evaluated (`cres`: the inside pairs and the Ritz neighbours; WMODE and None skipped). With `Z` given, only the candidates with $\mu \ge$ lo count: the below-lo neighbour is in practice a stored known column, a near duplicate. They are ordered nearest $|\mu - \sigma|$ first, at most `own`. There is no residual gate (the table above). The deflation is exact for any $Z$ (PR 18f; checked again in review: $b - Ax = P_Z(b - A\tilde x)$, and $P_Z A$ stays symmetric). So the own set only changes the MINRES iterations, not the returned pairs. Own Ritz vectors are mutually $M$-orthogonal and $M$-orthogonal to W, so they do not duplicate each other or the window.
- **Rebuild:** when a step has a non-empty own set and the loop goes on (`not done and k + b <= limit`), the slots are rewritten. The own vectors $Q_L Y_{\mathrm{own}}$ are formed $c$ columns at a time (`_combine` at width $c$, masked like Q) and written into the slots through the donated `_insert`, in place: no copy of the block. The slots past the own count are rewritten as zeros. Then $H$ is recomputed on the real columns (`_deflate_setup`). The block's padded width is fixed, so there is one compile. A σ move happens only before the first kept step, so before any own column, and recomputes $H$ as today; later rebuilds use the current A.
- **Rebuild cost:** $\lceil n_z/c\rceil$ applies of width $c$ plus the $n_z \times n_z$ pinv. That is 1.5 s at $n_z$ = 957 and 56k DOF: 3% of an early deep step (50 s), about 20% of a late one (8 s), 5–7% of the deep slice estimated. Part `deep` records it per step. An incremental $E$ (only $Z^T A Z_{\mathrm{own}}$, own/$c$ applies) is the follow-up if it exceeds 10% of the slice's seconds.
- **si_band:** with `max_deflate` = $d$ > 0, it passes `own = min(OWN_MAX, d // 2)` (`OWN_MAX` = 64) and the `d − own` nearest known columns (`_z_cols`, unchanged). The block stays within the plan's $d_Z$. Half of $d$ stays for known columns, so a small $d$ still deflates the nearest known modes from step 1. In the deep band, $d$ = 1157 and $d$ − 64 ≥ 957: every known column stays and 64 own slots are added. si_band rejects `own` in `slice_options`, as it does `Z`.
- **solve_modes' `stagnation_risk`** (`eigen/api.py:812`) compares the known columns actually deflated, $d - \mathrm{own}(d)$, with $j$ (today: $d < j$), and the warning's text says so.
- **si_checked** restarts pass the same `Z` and `own`. Each run forms its own columns afresh. A restart puts the earlier runs' found modes into W, so the restart's own candidates exclude them: less deflation, never wrong answers. Restarts are rare; noted in the docs.
- **info:** `deflated` stays the known columns given. New `own_columns` holds, per kept step, the own columns its solve used (before pinv).

### 2. Memory

Unchanged. The block is `len(Z) + own` ≤ $d_Z$ columns, padded per part, as `_slice_plan` sizes it. The docstrings are updated to say known plus own, and the padding term goes up by at most $c$ columns, which goes into `si_memory`'s block term. Forming the own vectors uses one `_combine` chunk of width $c$, the same as the modelled ritz phase. The writes are in place. There are no host arrays. `test_z_peak` (peak) gets an `own > 0` case.

### 3. The deferred slice items: measured

`benchmarks/slice_study.py` (results in `benchmarks/results/slice_study.json`; `--parts probe,many,deep`; each part within 1 h, `probe` only on a warm LOBPCG cache, not in the default parts):

- **Part `probe`:** the tables above (the deep slice's solves; the test slice's own-set variants, about 1 min).
- **Part `many`:** hex20 $8\times6\times5$ clamped, AMG, the band of modes 5–200 at capacity 96 (the PR 15c code review case), through `solve_modes(strategy="shift_invert", slice_options={"capacity": 96})`. Per slice it records the interval, kind (first, remainder with one done edge, verification, requeue, top remainder), new modes, steps, MINRES iterations and seconds. Two runs: own deflation off (`OWN_MAX` = 0 patched in the study: PR 18f behaviour, not an option) and the default. A few minutes per run.
- **Decisions from run 2:** the share of the band's MINRES iterations each kind takes. A kind under 10% is closed in `docs/open_items.md` with the numbers. One at 10% or more stays open with the numbers and a candidate (for one-done-edge remainders: σ at the midpoint of the unexplored part) for a follow-up PR.

### 4. The deep band

Part `deep`: PR 18f's band (`deep_band_study.band`, the cached LOBPCG run, hex20 $16^3$, modes 951–1008) with own deflation. Per step it records the MINRES iterations, own columns and rebuild seconds, plus the slice's seconds per mode, all against eigsh. About 40 min.

## Acceptance

- **The deep band, binding:**
  - every mode against eigsh ($\le 10^{-12}$ relative);
  - the slice's MINRES iterations summed over its steps $\le 0.7\times$ PR 18f's (about 31,600, so ≤ 22,100). The iteration count does not depend on machine load.
- **The deep band, reported:** seconds per mode against PR 18f's 36 s (expected ≤ 27 s).
- **Predicted:** 0.45–0.55×. The review combined PR 18f's per-step candidate curve with the probe's ratios (11 own → 0.6, 22 → 0.4, 51 → 0.14); the test slice gave 0.45× with every candidate. If the binding target is missed, the own set is revisited (the measurements per step say where) before merging.
- **The `many` band:** every mode against eigsh in each run, and the run-2 total MINRES iterations below run 1's.

## Tests

`tests/test_deflation.py`, the hex20 slice `edges(lr, 30, 42)` (12 inside, 7 own below σ, 16 steps, about 12 s), `Z` = the modes below lo, slow where it solves. One cached, spied run with `own` = 12 serves tests 1, 2 and 4. The spies are on `_slice_check` (`mu`, `cres` per step), on `_own_write` (the small helper that forms and writes the slots) and on `_deflate_setup`.

1. **Exactness and gain:** `own` = 12 against `own` = 0:
   - eigenvalues match to $10^{-10}$ relative and subspaces to small angles, and both match eigsh;
   - |Δ`steps`| ≤ 1 (measured equal);
   - `own_columns[0]` = 0, some step reaches 12, none above 12;
   - the mean MINRES iterations per step over steps 2 onward are below the `own` = 0 run's over the same steps (about 50 against 110 measured).
2. **The own set, exact:** at each rebuild, the written set equals the first min(12, #candidates) of the step's evaluated Ritz candidates with $\mu \ge$ lo, by $|\mu-\sigma|$. Each written column's Rayleigh quotient matches its μ to $10^{-10}$ relative (kills a stale-$Y$ mutant), and the slots past the count are zero. Preconditions, so the mutants stay killable:
   - some step has more candidates than accepted pairs (kills "accepted only");
   - some step has more than 12 candidates (kills "cap ignored" and "ordering reversed");
   - some written column has μ > σ (kills "below σ only");
   - some step's below neighbour is a Ritz index below lo (kills "below-lo kept").
3. **`Z=None`, `own` > 0:** the pairs match eigsh and the plain run, and `own_columns` > 0. Later steps need fewer iterations than the plain run's. Here the below neighbour may be an own column.
   - **`own` = 0:** `_own_write` is never called and `own_columns` is all 0.
   - **`own` < 0, a float or a bool:** ValueError.
4. **Rebuilds:**
   - `_zblock` is called once;
   - `_own_write` and the following `_deflate_setup` run once per step with a non-empty own set and `not done and k + b <= limit` (measured 13 in the test run: kills "never", "every step" and "on the last step");
   - a capacity-full case (a small capacity) has no write on its last step.
5. **σ move:** `test_slice_deflated_move` with `own` > 0: the move happens, `own_columns[0]` = 0, and the pairs match.
6. **si_checked:** extend `test_checked_keeps_z`: each run gets the same `Z` and `own`, and the restart's `own_columns` is > 0 at some step.
7. **si_band and solve_modes** (fake slices):
   - with `max_deflate` = $d$, each slice gets `own` = min(64, $d$//2) and the $d$ − own nearest known columns ($d$ = 3: own 1, known [0, 2, 2] in `test_band_z`; $d$ = 0: own 0);
   - `own` in `slice_options` raises;
   - `stagnation_risk` uses $d$ − own (`test_stagnation_risk`'s cases recomputed);
   - `test_deep_band`'s "max_deflate ≥ j" comment and check use $d$ − own.
8. **Memory:** `test_z_peak` with `own` > 0 (peak), and `test_memory_formulas` for the per-part padding.

Mutations to try in code review: accepted pairs only, below σ only, the below-lo neighbour kept, the cap ignored, the ordering reversed, a stale $Y$, the slots not zeroed, a rebuild on the last step or on every step, `own` ignored in si_band, `stagnation_risk` with $d$.

## Docs

- **README:**
  - line 5: add the 18g done entry (own-mode deflation, the measured gain), "next is 18h", order from here: 18h (CalculiX), then 20–23a, then 19, 23b;
  - line 287: "deflated by the known modes below its shift" becomes known and own modes, and the 52× figure is updated.
- **`docs/eigensolvers.md`:**
  - 141: the "auto" strategy deflates by known and own modes;
  - 153, 157: the 52× and 36 s numbers;
  - 189: the own-modes residual risk, resolved;
  - 196: $d_Z$ = known plus own;
  - 829: si_band's Z per item plus own;
  - 880–887: the PR 18f deep-band paragraph, its "PR 18g" note, and the `slice_study.py` link with `--parts`;
  - the MINRES deflation section: own columns, the candidates, the slots, the rebuild, the restart note;
  - si_slice's options and info (`own`, `own_columns`);
  - the measured section (the study's tables);
  - the test list.
- **Docstrings:**
  - si_slice: `own`, `own_columns`;
  - si_band and solve_modes: `max_deflate` counts known plus own, `own` is rejected;
  - `_slice_plan` and `si_memory`: the block is known plus own;
  - `stagnation_risk`.
- **`docs/ROADMAP.md`:**
  - line 18: the 18g merged entry, order from here 18h, 20, …;
  - lines 92 and 94: the 52× ratio and the per-step iterations, from the deep part;
  - the 18g section's Contents and Tests rewritten to this scope, then its status and results;
  - "Deferred GPU checks": the rebuild's applies and host pinv at 10M DOF;
  - the order line 879 and the dependency rows 869–871 need no change.
- **`docs/open_items.md`:**
  - 348–354: the section and each item closed with the numbers, or kept for a follow-up;
  - 395: the "→ PR 18g" note;
  - 155 and 335: the PR 18f residual-risk notes;
  - the decisions below.
- **Memory:** the autonomy line (18g merged, next 18h).

## Decisions without user input

- Own columns from every evaluated candidate, both sides of σ, no residual gate. The measured gain is 0.45× against 0.59× at $10^{-3}$, and the deflation is exact for any vectors. With known columns given, the below-lo neighbour is excluded (a near duplicate slows MINRES).
- Own slots: min(64, $d_Z$//2) inside the plan's $d_Z$, appended after the known columns. Half stays known so a small $d_Z$ still deflates from step 1. One block, one compile, in-place writes, no host vectors.
- $H$ is recomputed in full per rebuild; an incremental $E$ waits for the measured share (> 10%).
- The remainder σ rule and the other deferred items are measured here and changed in a follow-up only if a kind takes ≥ 10% of the iterations.
