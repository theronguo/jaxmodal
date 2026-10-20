# PR 18g: slice efficiency (plan)

Branch `pr-18g-slices`. Roadmap section "PR 18g: Slice efficiency" (`docs/ROADMAP.md`), the open items "Slice efficiency (PR 15c → PR 18e → PR 18f → PR 18g)" (`docs/open_items.md`). Rev 6 (after plan review rounds 1–5).

## Goal

1. **A slice's own modes** (the main item). PR 18f deflates each slice's MINRES by the stored modes below σ. The slice's own modes are the nearest to σ, and so the most expensive, and they stay undeflated.

   **Probe** (2026-10-20, scratch `p18g/own.py` and review round 1–2 probes): the deep band's slice of PR 18f (hex20 $16^3$, AMG, σ = 443.93). The cut is at 434.72, with 957 known modes below it, 22 own modes in [cut, σ) and 29 in (σ, hi]. One solve of 6 columns at τ = $10^{-8}$, the vectors from the cached LOBPCG modes:

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

   **Before acceptance** (review rounds 2–5, the hex20 test slice `edges(lr, 30, 42)`, 16 steps, 110 iterations per step with the 30 known modes; own columns formed from `_slice_check`'s evaluated candidates each step, nearest $|\mu-\sigma|$ first, at most 12):

   | own columns from | summed iterations | ratio |
   |---|---|---|
   | none | 1720 | 1 |
   | every candidate | 780 | 0.45 |
   | $\mu \ge$ lo, the Schur screen after the cap (chosen; round 5, two runs) | 770–780 | 0.45 |
   | the same, the screen scaled to the deep slice's half-width (×6.9) | 830 | 0.48 |
   | candidates with residual ≤ $10^{-2}$ | 960 | 0.56 |
   | ≤ $10^{-3}$ | 1010 | 0.59 |
   | ≤ $10^{-4}$ | 1130 | 0.66 |

   Identical runs differ by about ±20 iterations. In rounds 4–5 the true residual matched MINRES's to 2 digits (worst 0.99τ). With every candidate, the gain starts at step 2 (110, 80, 60, 50, then 40 per step). Unconverged Ritz vectors near σ jointly span the near-σ subspace well, and ghost Ritz values (residual about 1) did no harm. A below-lo neighbour that is a stored known mode (step 13, residual about $10^{-4}$) raised that solve from 40 to 70: a near duplicate of a $Z$ column that pinv (rcond $10^{-10}$) keeps, which makes $E$ ill-conditioned (review probe: cond($E$) 2.5e9 at a $10^{-3}$ error).
2. **The deferred slice items** (PR 15c/15d → 18g): verification slices (remainders that find at most one new mode in 10–15 steps), a top remainder holding only known modes ("no progress"), the whole-slice requeue. In this PR they are measured; changes go to a follow-up if the numbers ask for one (below).

## Design

### 1. Own-mode deflation in si_slice

- **Slots:** `si_slice(..., own=0)` takes `own`, the number of own columns. The block is `[Z | own slots]`, each part zero-padded to a multiple of $c$, so the own region starts on a chunk boundary. The block is allocated once at setup, the known columns are uploaded once as today, and the slots start at zero. `own = 0` is today's behaviour. `own` must be an int ≥ 0, not a bool (ValueError, as `max_deflate`'s check). When the block exists (`len(Z) + own > 0`), every solve uses `_dminres`. $H$ starts as $H_K$, the pinv of the known part padded with zero own rows: zeros for `Z=None`, where `_dminres` with $H = 0$ is plain MINRES (a few block products on the early steps).
- **$E$ and $H$ over the padded width:** `_deflate_setup`'s `nz` trim goes. `_zaz(A, Zd, c, i0=0)` gets a start chunk: setup forms all block columns of $E = Z^T A Z$, a rebuild only the own region's (`i0` = its first chunk; $\lceil \mathrm{own}/c\rceil$ applies, 3 instead of 43 in the deep slice). The host keeps $E$, mirrors the new block column into its rows, and takes $H = \mathrm{pinv}(\mathrm{sym}(E))$ over the padded width with rcond $10^{-10}$ (`_own_h`, a host helper; $O(n_z^3)$, 0.1 s at $10^3$ columns). Zero slots give zero rows and columns of $E$, which pinv drops. A σ move (only before the first kept step, so before any own column) recomputes $E_{KK}$ and $H_K$ as today's setup does; later rebuilds use the current A.
- **Which pairs** (`_own_set(mu, cres, lo, sigma, own)`, a pure helper): the Ritz indices in `_slice_check`'s `cres` (the inside pairs and the Ritz neighbours; WMODE and None never enter `cres`; every key is an evaluated candidate). It keeps those with $\mu \ge$ lo, always: the below-lo neighbour is in practice a stored known column, a near duplicate. Then it orders them nearest $|\mu - \sigma|$ first (ties by ascending μ, stable) and caps at `own`.
- **Screen** (in `_own_h`, before the pinv): column $i$ is kept only if $|s_i| \ge \rho_i^2 |\mu_i|$. Here $s_i = E_{ii} - e_i^T E_{KK}^{+} e_i$ is its Schur complement against the known part, with $E_{KK}^+$ the known block of the kept $H_K$ (no second pinv) ($= \mu_i - \sigma$ without known overlap), and $\rho_i$ is its relative residual from `cres` (`_resid`'s $\|Kz - \mu Mz\|/(|\mu|\,\|Mz\|)$). This bounds the column's rank-one term in $P_Z A$ at $O(|\mu|)$.
  - What it drops: a ghost with a large residual near σ, and a mixed column (part known mode, part a direction with $y^T A y \approx 0$). Review probes: 20000 capped iterations with either; for the mixed columns $s/E_{ii}$ = 2e-7 to 2e-3, against 0.98–1.02 for noisy genuine own columns.
  - How: every inactive slot (screened out, or past the own count) gets zero rows and columns in $E$ before the pinv, so it is inert (zero rows of $H$), with no device write. A screened-out slot is not refilled from the next candidate (a follow-up if the deep part shows many drops).
  - Per column, not jointly: two columns can each pass and still together contain part of a known mode (a review probe: 2 of 952 synthetic cases, mild, cond($E$) about 900). The fallback covers that.
  - Its triggers are synthetic so far: on the test slice it drops only high-residual ghosts (ρ ≥ 0.08), never a converged column ($s/(\mu-\sigma)$ = 1.000), at a cost of 770–780 against 710–730 summed with no screen. If the deep part never drops a candidate, removing it is a follow-up option.
- **Exactness:** the deflation is exact for any $Z$ and any symmetric $H$ (PR 18f; rechecked in review: $b - Ax = P_Z(b - A\tilde x)$, $P_Z A$ symmetric; a truncated pinv keeps $HEH = H$). So the own set changes only the MINRES iterations, not the returned pairs. The own Ritz vectors lie in the Lanczos space, which is $M$-orthogonal to W by construction, and are mutually $A$-orthogonal ($Z_{\mathrm{own}}^T A Z_{\mathrm{own}} = \mathrm{diag}(\mu - \sigma)$, from $G = Q_L^T K Q_L$). Inside candidates near lo can still overlap known columns outside W, so $E$ is not always well conditioned: the screen's and the fallback's job.
- **Fallback** (`_own_failed(res, stagnated, capped, true_res, tau, tau_eff)`, a pure predicate). It applies only to a solve whose own count after the screen is above 0; a solve with every own column screened out runs as a known-only solve. Such a solve fails if any of these holds:
  - a residual is non-finite;
  - a column is capped;
  - a column stagnated above the current `tau_eff`;
  - the true residual, per column $\|b_j - Ax_j\|/\|b_j\|$ with the max over columns (one apply of width $b$ per solve), is above $10\max(\tau, \tau_{\mathrm{eff}}, \mathrm{TAU\_MIN})$. That is, the true residual agrees with MINRES's to 10×; each term keeps a tolerated case from tripping it (τ_eff for stagnation already tolerated, TAU_MIN for a τ below the fp64 floor).

  Stagnation the slice already tolerates does not count. Solves with own columns are capped at $\min(\mathrm{cap}, 2\max(\mathrm{iters}))$, so a failure costs at most twice a step's solve. The cap is safe: `iters` always holds step 1, which has no own columns and is the largest count; τ_eff only rises, lowering counts; the σ move comes before any own solve; and PR 18f's deep slice drifted only 810→920 over 37 steps.

  The check runs right after the solve, before the non-finite break, `bad` and `tau_eff`. On a failure:
  - $H = H_K$ and own writes stop for the rest of the run;
  - the solve is redone at the normal cap, and its result, `capped`, `inner_maxit` and seconds are used;
  - the failed solve's iterations go to `own_dropped`, its seconds to `time["minres"]`; neither goes to `iters` or `t_its` (so `t_minres` leaves it out).

  So a failed own deflation never raises `tau_eff` or fails a slice that works without it.
- **Rebuild:** when the loop goes on (`not done`), own writes are on, and the own set is non-empty, `_own_write` rewrites the slots. The own vectors $Q_L Y_{\mathrm{own}}$ are formed $c$ columns at a time (`_combine` at width $c$, masked like Q) and written through the donated `_insert`, in place, with no copy of the block. Slots past the own count keep stale vectors, masked in $E$ (above). No other live reference to the old `Zd` may remain, since donation deletes it. Then `_own_h` updates $E$ and $H$. The padded width is fixed, so there is one compile. A capacity-full exit after the rebuild wastes at most one rebuild.
- **Rebuild cost:** own/$c$ applies plus the host pinv, under 0.5 s in the deep slice (the full setup was 1.6 s at $n_z$ = 1005). Part `deep` records it per step.
- **si_band:** with `max_deflate` = $d$ > 0, it passes `own = min(OWN_MAX, d // 2)` and the `d − own` nearest known columns (`_z_cols`, unchanged). The block stays within the plan's $d_Z$. Half of $d$ stays for known columns, so a small $d$ still deflates the nearest known modes from step 1. `OWN_MAX` = 64 is at least the deep slice's 51 own candidates; its columns cost per MINRES iteration about what known columns do (64 add about 2% there). In the deep band, $d$ = 1157 and $d$ − 64 ≥ 957: every known column stays and 64 own slots are added. si_band raises "si_band sets" for `own` in `slice_options`, as for `Z`. solve_modes' SLICE_OPTIONS whitelist rejects it too, since `own` is not added there.
- **solve_modes' `stagnation_risk`** (`eigen/api.py:812`) compares the known columns actually deflated, $d - \mathrm{own}(d)$, with $j$ (today: $d < j$). The warning text becomes "deflation limited to $d - \mathrm{own}$ of $j$ known modes". A junction with $j \le d < j + \mathrm{own}$ (e.g. a memory-capped $d = j$) now warns where PR 18f did not: there the own slots take the farthest known columns. The probe favours that trade (13 known + 51 own: 2030 iterations against 8200 with 64 known), and the warning stays truthful about the known part.
- **si_checked** restarts pass the same `Z` and `own`. Each run forms its own columns afresh. A restart puts the earlier runs' found modes into W, so the restart's own candidates exclude them: less deflation, never wrong answers. Restarts are rare; this is noted in the docs.
- **info:**
  - `deflated` stays the known columns given;
  - `own_columns` is a list with one int per kept step: the own columns its solve used (after the screen, before pinv; 0 on the fallback's redone solve and after);
  - `own_dropped` is `(step, iterations)` of the failed solve, or None; `step` is 0-based, the index into `own_columns` (as `minres_iterations`).

### 2. Memory

Unchanged in kind. The block is `len(Z) + own` ≤ $d_Z$ real columns. Padding per part adds at most $c$ columns, which go into `si_memory`'s block term ($z_p$). The docstrings say known plus own. Forming the own vectors uses one `_combine` chunk of width $c$, as the modelled ritz phase does. The true-residual apply has width $b$, inside the MINRES phase. The writes are in place. The host keeps $E$ ($n_z^2$, as the setup today). `test_z_peak` (peak) gets an `own > 0` case.

### 3. The deferred slice items: measured

`benchmarks/slice_study.py` (results in `benchmarks/results/slice_study.json`; `--parts many,deep`; each part within 1 h):

- The probe tables above were measured during planning (review scratch); they back decisions already made and are recorded in the docs as such, with no study part of their own (benchmark-cost rule).
- **Part `many`:** hex20 $8\times6\times5$ clamped, AMG, the band of modes 5–200 at capacity 96 (the PR 15c code review case), through `solve_modes(strategy="shift_invert", slice_options={"capacity": 96})`. Per slice it records the interval, kind (first, remainder with one done edge, verification, requeue, top remainder), new modes, steps, MINRES iterations and seconds. Two runs: own deflation off (`OWN_MAX` = 0 patched in the study: PR 18f behaviour, not an option) and the default. A few minutes per run.
- **Decisions from run 2:** the share of the band's MINRES iterations each kind takes. A kind under 10% is closed in `docs/open_items.md` with the numbers. One at 10% or more stays open with the numbers and a candidate (for one-done-edge remainders: σ at the midpoint of the unexplored part) for a follow-up PR.

### 4. The deep band

Part `deep`: PR 18f's band (`deep_band_study.band`, the cached LOBPCG run, hex20 $16^3$, modes 951–1008) with own deflation. Per step it records the MINRES iterations, own columns, the candidates the screen dropped, rebuild seconds and any fallback, plus the slice's seconds per mode, all against eigsh. About 20 min at the predicted gain (under 1 h without it).

## Acceptance

- **The deep band, binding:**
  - every mode against eigsh ($\le 10^{-12}$ relative);
  - the slice's MINRES iterations summed over its steps, including the discarded σ-move and fallback solves, $\le 0.7\times$ PR 18f's (about 31,600, so ≤ 22,100). The iteration count does not depend on machine load.
- **The deep band, reported:** seconds per mode against PR 18f's 36 s (expected ≤ 27 s: MINRES was about 90% of PR 18f's 1839 s; Lanczos, the Ritz checks and the rebuilds are not reduced).
- **Predicted:** 0.45–0.55×. The review combined PR 18f's per-step candidate curve with the probe's ratios (11 own → 0.6, 22 → 0.4, 51 → 0.14); the test slice gave 0.45× with the chosen rule (0.48× with the screen scaled to the deep slice). If the binding target is missed, or a fallback fires, the own set (or a re-enable after a fallback) is revisited (the measurements per step say where) before merging.
- **The `many` band:** every mode against eigsh in each run, and the run-2 total MINRES iterations below run 1's.

## Tests

**Fast tier** (`tests/test_deflation.py`, no solve):

1. **`_own_set`** on synthetic `mu`/`cres`. The cases:
   - a below-lo candidate nearest σ, and μ exactly at lo (kept);
   - more than `own` candidates, on both sides of σ;
   - equal $|\mu-\sigma|$ on both sides (the tie order);
   - no candidate at all (empty set);
   - a WMODE side.

   The asserts: the exact set and its order. This kills "accepted only", "below σ only", "below-lo kept", "`>` lo", "cap ignored" and "ordering reversed".
2. **`_own_h`** on a small dense pencil against NumPy:
   - $H$ equals `pinv(sym(Z^T A Z))` over the padded width with a gap between the parts (kills the prefix trim and a transposed or unmirrored block column);
   - the screen drops a mixed column (part known mode, part $y^T A y \approx 0$) and a near-σ ghost, and keeps noisy genuine own columns, with clear margins on both sides;
   - the inactive slots (screened out, past the count, stale vectors) have zero rows of $H$;
   - a set that is non-empty but fully screened out gives $H = H_K$ and a kept count of 0;
   - "screen removed" fails.
3. **`_own_failed`:** one case per trigger (non-finite, capped, stagnated above `tau_eff`, true residual above the threshold). The negatives:
   - stagnated at or below `tau_eff`;
   - a true residual within $10\tau$;
   - `tau_eff` > 10τ with a true residual between 10τ and 10·`tau_eff` (kills dropping `tau_eff`);
   - τ < TAU_MIN with a true residual between 10τ and 10·TAU_MIN (kills dropping the floor).

   This kills a predicate missing any trigger or term.

**Slow tier:** the hex20 slice `edges(lr, 30, 42)` (12 inside, 7 own below σ, 16 steps, about 10 s; with `Z=None` about 22 s). `Z` is the 30 modes below lo, except in tests 6 and 8. One cached, spied run with `own` = 12 serves tests 4, 5 and 7. The spies:
- `_slice_check`: `mu` and `cres` per step;
- `_own_write`: copies the block to the host at once, since donation deletes it;
- `_own_h`: $E$, $H$ and the kept columns.

Numbers measured in review (`p2-performance/gate.log`, `p3-performance/gate3.log`, `p4-performance/tr.log`, `p5-tests/proto.py`): the mean is about 50 against 110 iterations per step from step 2 on, and every step has a non-empty own set.

4. **Exactness and gain:** `own` = 12 against `own` = 0:
   - eigenvalues match to $10^{-10}$ relative and subspaces to small angles, and both match eigsh;
   - |Δ`steps`| ≤ 1 (measured equal);
   - `own_columns[0]` = 0, and `own_dropped` is None;
   - the mean MINRES iterations per step from step 2 on are below the `own` = 0 run's over the same steps.
5. **The wiring:** at each rebuild:
   - the written set equals `_own_set` on the spied inputs, and some step writes 12 columns;
   - each written column's Rayleigh quotient matches its μ to $10^{-10}$ relative (kills a stale $Y$);
   - every real kept column $z$ (known and own) satisfies $\|P_Z A z\| / \|A z\| \le \max(10^{-8}, 10\,\epsilon\,\mathrm{cond}(E))$ through `_zop` with the spied $H$ (kills a wrong offset and an $H$ missing own slots).

   Precondition: some step has more than 12 candidates with $\mu \ge$ lo.
6. **`Z=None`, `own` > 0:** the pairs match eigsh, and some entry of `own_columns` is > 0. **`own` = 0:** `_own_write` is never called and `own_columns` is all 0. **`own` < 0, a float or a bool:** ValueError.
7. **Rebuilds:** `_own_write` runs exactly at the non-final steps with a non-empty own set (the count from the spies, here steps − 1), never after the last step.
8. **Fallback, end to end:** a spy on `minres` forces "stagnated, residual $10^{-3}$" on the first solve after the 3rd rebuild (step index 3; σ cannot move after the first kept step), as `test_slice_deflated_move`'s stall does. The redo runs for real. A real ill-conditioned column is test 2's job: an injected column is neutralised by the real Ritz own columns (round 5 probe `p5-tests/t8*.py`). The asserts:
   - `own_dropped` = (3, the stalled solve's iterations);
   - later entries of `own_columns` are 0;
   - after the fallback, every known column satisfies the `_zop` check of test 5 with the solve's $H$ (kills "old $H$ kept");
   - `tau_eff` is not raised: the forced $10^{-3}$ would raise it without the fallback;
   - the pairs match eigsh.

   Also once with `Z=None`.
9. **σ move:** `test_slice_deflated_move` with `own` > 0:
   - the move happens, and `own_columns[0]` = 0;
   - the setup after the move sees the moved σ (the existing `len(sig) == 2` stays: setup and move);
   - at a rebuild after the move, test 5's `_zop` check holds on all real columns (kills a stale $E_{KK}$);
   - the pairs match.
10. **si_checked:** extend `test_checked_keeps_z`: each run gets the same `Z` and the same `own`, and some entry of the restart's `own_columns` is > 0.
11. **si_band and solve_modes** (fake slices), with `max_deflate` = $d$:
    - each slice gets `own` = min(64, $d$//2) and the $d$ − own nearest known columns: $d$ = 3 gives own 1 and known [0, 2, 2] in `test_band_z` (`stored[-(d - own):]`); $d$ = 200 gives own 64 and known min(136, stored); $d$ = 0 gives own 0;
    - `own` in `slice_options` raises "si_band sets";
    - `test_stagnation_risk` with $d$ − own: "full" ($d = j$) now expects True and the text "deflation limited to {d − own} of {j}"; a new "full_own" case ($d = j$ + own($d$)) expects False; "below" asserts risk = ($d$ − own($d$) < $j$) from the recorded $d$;
    - `fake_band` records `own` per call;
    - `test_deep_band`'s "max_deflate ≥ j" check uses $d$ − own.
12. **Memory:** `test_z_peak` with `own` > 0 (peak), `test_zd_freed` with `own` > 0 (the last live block is dead after return), and `test_memory_formulas` for the per-part padding.

Mutations to try in code review:
- the fast-tier mutants above;
- a stale $Y$;
- a wrong slot offset;
- a rebuild on every step;
- the fallback removed, raising `tau_eff`, keeping the failed solve in `iters`, or the redo at the own cap;
- a stale $E_{KK}$ after a move;
- `own` ignored in si_band;
- `stagnation_risk` with $d$.

## Docs

- **README:**
  - line 5: add the 18g done entry (own-mode deflation, the measured gain), "next is 18h", order from here: 18h (CalculiX), then 20–23a, then 19, 23b;
  - line 287: "deflated by the known modes below its shift" becomes known and own modes, `max_deflate` caps both, and the 52× figure is updated.
- **`docs/eigensolvers.md`:**
  - 101: the `slice_options` table's `max_deflate` caps known plus own;
  - 141: the "auto" strategy deflates by known and own modes;
  - 153, 157: the 52× and 36 s numbers;
  - 187: the junction hand-over ("the nearest $d_Z$ stored modes below its σ" → $d_Z$ − own known plus own slots);
  - 189: the warning text ($d_Z - \mathrm{own}$ of $j$), the "up to $d_Z$" clause, and the own-modes residual risk resolved;
  - 196: $d_Z$ = known plus own;
  - 786: si_checked passes `Z` and `own` to every run, with the restart note;
  - 829: si_band's Z per item plus own;
  - 844: si_memory's $n_z$ = known plus own, padded per part ($z_p$);
  - 880–887: the PR 18f deep-band paragraph, its "PR 18g" note, and the `slice_study.py` link with `--parts`;
  - the MINRES deflation section: own columns, `_own_set`, the slots, $E$/$H$ over the padded width, the screen (per column), the rebuild, the fallback and why (exact, but conditioning not guaranteed on an indefinite $A$), why the own cap is safe, the restart note;
  - si_slice's options and info (`own`, `own_columns`, `own_dropped`), and 754: `t_minres` leaves out a failed own solve;
  - a new paragraph "Slice efficiency (PR 18g)" (anchor `slice-efficiency`) after `#deep-bands-deflated`: the planning probes (as measured during planning), the `many` and `deep` tables, linked from the open-items closures;
  - the test list.
- **Docstrings:**
  - si_slice: `own`, `own_columns`, `own_dropped`;
  - si_checked (shift_invert.py:587ff): `own` passed to every run;
  - si_band and solve_modes: `max_deflate` counts known plus own; `own` raises "si_band sets" in si_band and is not in solve_modes' SLICE_OPTIONS;
  - `_slice_plan` and `si_memory`: the block is known plus own;
  - `_zaz` (`i0`), `_deflate_setup` (no trim), `_own_set`, `_own_h`, `_own_failed`, `_own_write`;
  - `stagnation_risk`.
- **`docs/ROADMAP.md`:**
  - line 18: the 18g merged entry, order from here 18h, 20, …;
  - lines 92 and 94: the 52× ratio and the per-step iterations, from the deep part;
  - the 18g section (689ff):
    - "Contents (plan `docs/plans/pr-18g-slices.md`)" and Tests rewritten to this scope: own columns on both sides of σ;
    - "σ toward the unconverged neighbour" mapped to the remainder-σ candidate of §3;
    - then the status and results;
  - "Deferred GPU checks": the rebuild seconds against the step's MINRES seconds at 10M DOF, the host pinv per rebuild at several thousand $n_z$, and the true-to-reported residual ratio (the fallback's threshold);
  - line 828 (the $d_Z$ cap at 10M): $d_Z$ is known plus own (at most 64 own);
  - the order line 879 and the dependency rows 869–871 need no change.
- **`docs/open_items.md`:**
  - 175: the warning now also at $j \le d < j + \mathrm{own}$;
  - 329: "the nearest stored modes below σ first when memory limits $d_Z$": half of $d_Z$ now goes to own slots;
  - 348–354: the section and each item closed with the numbers, or kept for a follow-up;
  - 395: the "→ PR 18g" note;
  - 155: the 52× figure; 334: the "810–920 deflated MINRES iterations per step" (kept as PR 18f's figure, labelled so); 335: the PR 18f residual-risk note;
  - the decisions below.
- **Memory:** the autonomy line (18g merged, next 18h).

## Decisions without user input

- Own columns from every evaluated candidate with $\mu \ge$ lo, on both sides of σ:
  - no residual gate: the measured gain is 0.45× against 0.59× at $10^{-3}$;
  - the below-lo neighbour is excluded: it is a near duplicate of a known column and slows MINRES;
  - the Schur screen $|s_i| \ge \rho_i^2|\mu_i|$, per column, drops ghosts at σ and mixed columns before they reach a solve. Its triggers are synthetic so far; it costs about 0.03× on the test slice and is removable if the deep part never drops a candidate;
  - a capped, tolerance-aware fallback switches own deflation off for the run when a solve with own columns still fails. Exactness holds; conditioning does not.
- No re-enable after a fallback: none fired in review. If the deep part shows one, re-enabling at the next rebuild is the follow-up.
- Own slots: min(64, $d_Z$//2) inside the plan's $d_Z$, appended after the known columns.
  - Half stays known, so a small $d_Z$ still deflates from step 1.
  - At $j \le d < j + \mathrm{own}$ the slots take the farthest known columns, and `stagnation_risk` (the known part: $d - \mathrm{own} < j$) warns. The probe favours own columns there.
- One block, one compile, in-place writes, no host vectors. $E$ is kept on the host; a rebuild forms only the own region's block column (`_zaz` from `i0`).
- The planning probes are not a study part: the decisions are made, and the numbers are recorded as measured during planning.
- Inactive slots are masked in $E$ (no zero writes); a screened-out slot is not refilled.
- The remainder σ rule and the other deferred items are measured here, and changed in a follow-up only if a kind takes ≥ 10% of the iterations.
