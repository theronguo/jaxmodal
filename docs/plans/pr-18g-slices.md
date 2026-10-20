# PR 18g: slice efficiency (plan)

Branch `pr-18g-slices`. Roadmap section "PR 18g: Slice efficiency" (`docs/ROADMAP.md`), the open items "Slice efficiency (PR 15c → PR 18e → PR 18f → PR 18g)" (`docs/open_items.md`). Rev 4 (after plan review rounds 1–3).

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

- **Slots:** `si_slice(..., own=0)` takes `own`, the number of own columns. The block is `[Z | own slots]`, each part zero-padded to a multiple of $c$, so the own region starts on a chunk boundary. The block is allocated once at setup, the known columns are uploaded once as today, and the slots start at zero. `own = 0` is today's behaviour. `own` must be an int ≥ 0, not a bool (ValueError, as `max_deflate`'s check). With `own > 0` and `Z=None`, the block has no real column until the first own write, and the solves until then are plain MINRES. `solve` checks the current own count; no H setup runs on an empty block.
- **$E$ and $H$ over the padded width:** `_deflate_setup` drops its `nz` trim. $E = Z^T A Z$ is formed over the whole padded block, and $H = \mathrm{pinv}(\mathrm{sym}(E))$ with rcond $10^{-10}$. Zero slots give zero rows and columns of $E$, which pinv drops, so the gap between the parts does no harm. The known part $E_{KK}$ is kept on the host from setup. A rebuild forms only the own columns' block column $Z^T (A Z_{\mathrm{own}})$, which takes $\lceil \mathrm{own}/c \rceil$ applies of width $c$ (3 instead of 43 in the deep slice), mirrors it, and re-runs the pinv (host, $O(n_z^3)$: 0.1 s at $10^3$ columns).
- **Which pairs** (`_own_set(mu, cres, lo, sigma, own)`, a pure helper): it takes the Ritz indices in `_slice_check`'s `cres` (the inside pairs and the Ritz neighbours; the WMODE and None sides are not in `cres`). It keeps those with $\mu \ge$ lo (always: the below-lo neighbour is in practice a stored known column, a near duplicate) and $|\mu - \sigma| \ge \rho^2 |\mu|$, with ρ the candidate's relative residual from `cres`. They are ordered nearest $|\mu - \sigma|$ first and capped at `own`. The second condition bounds a candidate's rank-one term in $P_Z A$ at $O(|\mu|)$. A ghost with a large residual and μ within $10^{-7}$ of σ otherwise blows $P_Z A$ up: 20000 capped iterations in a review probe. On the test slice the filter costs nothing measurable (750 against 730 summed). There is no residual gate otherwise (the table above).
- **Exactness:** the deflation is exact for any $Z$ and any symmetric $H$ (PR 18f; rechecked in review: $b - Ax = P_Z(b - A\tilde x)$, $P_Z A$ symmetric). So the own set changes only the MINRES iterations, not the returned pairs. Own Ritz vectors are mutually $M$-orthogonal and $M$-orthogonal to W ($Z_{\mathrm{own}}^T A Z_{\mathrm{own}} = \mathrm{diag}(\mu - \sigma)$). Inside candidates near lo can still overlap known columns outside W, so $E$ is not always well conditioned. That case is the fallback's job.
- **Fallback:** the conditioning of $P_Z A$ is not guaranteed for non-eigen vectors on an indefinite $A$. A mixed column, part known mode and part a direction with $y^T A y \approx 0$, passes the filter and still stalls MINRES (review probe). So a solve with own columns is checked: any column stagnated or capped, or the true residual $\|b - AX\|/\|b\|$ above $10\tau$ (one apply of width $b$ per solve). If it fails, the own slots are zeroed, $H$ is recomputed on the known part, the solve is redone (the failed solve's iterations kept in `own_dropped`, like a move's), and own deflation stays off for the rest of the run. This happens before `bad` and `tau_eff` are evaluated, so a failed own deflation never raises `tau_eff` or fails a slice that works without it. info `own_dropped`: (step, discarded iterations), or None.
- **Rebuild:** when the loop goes on (`not done`) and the own set is non-empty, the slots are rewritten. The own vectors $Q_L Y_{\mathrm{own}}$ are formed $c$ columns at a time (`_combine` at width $c$, masked like Q) and written into the slots through the donated `_insert`, in place with no copy of the block. The slots past the own count are rewritten as zeros. No other live reference to the old `Zd` may remain, since donation deletes it. The padded width is fixed, so there is one compile. A capacity-full exit after the rebuild wastes at most one rebuild. A σ move happens only before the first kept step, so before any own column, and recomputes $H$ as today; later rebuilds use the current A.
- **Rebuild cost:** own/$c$ applies plus the host pinv, under 0.5 s in the deep slice (the full setup was 1.6 s at $n_z$ = 1005). Part `deep` records it per step.
- **si_band:** with `max_deflate` = $d$ > 0 it passes `own = min(OWN_MAX, d // 2)` and the `d − own` nearest known columns (`_z_cols`, unchanged). The block stays within the plan's $d_Z$. Half of $d$ stays for known columns, so a small $d$ still deflates the nearest known modes from step 1. `OWN_MAX` = 64 is at least the deep slice's 51 own candidates, and its columns cost per MINRES iteration about what known columns do (64 add about 2% there). In the deep band $d$ = 1157 and $d$ − 64 ≥ 957: every known column stays and 64 own slots are added. si_band rejects `own` in `slice_options`, as it does `Z` (the SLICE_OPTIONS key check).
- **solve_modes' `stagnation_risk`** (`eigen/api.py:812`) compares the known columns actually deflated, $d - \mathrm{own}(d)$, with $j$ (today: $d < j$). The warning text becomes "deflation limited to $d - \mathrm{own}$ of $j$ known modes". A junction with $j \le d < j + \mathrm{own}$ (e.g. a memory-capped $d = j$) now warns where PR 18f did not: there the own slots take the farthest known columns. The probe favours that trade (13 known + 51 own: 2030 iterations against 8200 with 64 known), and the warning stays truthful about the known part.
- **si_checked** restarts pass the same `Z` and `own`; each run forms its own columns afresh. A restart puts the earlier runs' found modes into W, so the restart's own candidates exclude them: less deflation, never wrong answers. Restarts are rare; noted in the docs.
- **info:** `deflated` stays the known columns given. New `own_columns` holds, per kept step, the own columns its solve used (after the filter, before pinv; 0 after a fallback). New `own_dropped` holds the fallback's step.

### 2. Memory

Unchanged in kind. The block is `len(Z) + own` ≤ $d_Z$ real columns. Padding per part adds at most $c$ columns, which go into `si_memory`'s block term ($z_p$). The docstrings say known plus own. Forming the own vectors uses one `_combine` chunk of width $c$, as the modelled ritz phase does. The fallback's true-residual apply has width $b$, inside the MINRES phase. The writes are in place. The host gains $E$ ($n_z^2$, as the setup today). `test_z_peak` (peak) gets an `own > 0` case.

### 3. The deferred slice items: measured

`benchmarks/slice_study.py` (results in `benchmarks/results/slice_study.json`; `--parts probe,many,deep`; each part within 1 h):

- **Part `probe`:** the test slice's own-set variants (the second table above, about 1 min; it backs the no-gate and filter decisions). The deep slice's solves stay in the plan as measured in review (they need the warm LOBPCG cache).
- **Part `many`:** hex20 $8\times6\times5$ clamped, AMG, the band of modes 5–200 at capacity 96 (the PR 15c code review case), through `solve_modes(strategy="shift_invert", slice_options={"capacity": 96})`. Per slice it records the interval, kind (first, remainder with one done edge, verification, requeue, top remainder), new modes, steps, MINRES iterations and seconds. Two runs: own deflation off (`OWN_MAX` = 0 patched in the study: PR 18f behaviour, not an option) and the default. A few minutes per run.
- **Decisions from run 2:** the share of the band's MINRES iterations each kind takes. A kind under 10% is closed in `docs/open_items.md` with the numbers. One at 10% or more stays open with the numbers and a candidate (for one-done-edge remainders: σ at the midpoint of the unexplored part) for a follow-up PR.

### 4. The deep band

Part `deep`: PR 18f's band (`deep_band_study.band`, the cached LOBPCG run, hex20 $16^3$, modes 951–1008) with own deflation. Per step it records the MINRES iterations, own columns, rebuild seconds and any fallback, plus the slice's seconds per mode, all against eigsh. About 40 min.

## Acceptance

- **The deep band, binding:**
  - every mode against eigsh ($\le 10^{-12}$ relative);
  - the slice's MINRES iterations summed over its steps $\le 0.7\times$ PR 18f's (about 31,600, so ≤ 22,100). The iteration count does not depend on machine load.
- **The deep band, reported:** seconds per mode against PR 18f's 36 s (expected ≤ 27 s).
- **Predicted:** 0.45–0.55×. The review combined PR 18f's per-step candidate curve with the probe's ratios (11 own → 0.6, 22 → 0.4, 51 → 0.14); the test slice gave 0.45× with every candidate. If the binding target is missed, the own set is revisited (the measurements per step say where) before merging.
- **The `many` band:** every mode against eigsh in each run, and the run-2 total MINRES iterations below run 1's.

## Tests

**Fast tier** (`tests/test_deflation.py`): `_own_set` on synthetic `mu`/`cres`. The cases:
- a below-lo candidate nearest σ;
- more than `own` candidates, on both sides of σ;
- a ghost with a large residual within $\rho^2|\mu|$ of σ;
- a WMODE side.

The asserts: the exact set and its order. This kills the "accepted only", "below σ only", "below-lo kept", "filter removed", "cap ignored" and "ordering reversed" mutants for certain.

**Slow tier:** the hex20 slice `edges(lr, 30, 42)` (12 inside, 7 own below σ, 16 steps, about 10 s; with `Z=None` about 22 s), `Z` = the 30 modes below lo. One cached, spied run with `own` = 12 serves tests 1, 2 and 4. The spies are on `_slice_check` (`mu`, `cres` per step), `_own_write` (the helper that forms and writes the slots; the spy copies the block to the host at once, since donation deletes it) and `_deflate_setup`. Numbers measured in review (`p2-performance/gate.log`, `p3-performance/gate3.log`): mean about 50 against 110 iterations per step from step 2 on; every step with a non-empty own set.

1. **Exactness and gain:** `own` = 12 against `own` = 0:
   - eigenvalues match to $10^{-10}$ relative and subspaces to small angles, and both match eigsh;
   - |Δ`steps`| ≤ 1 (measured equal);
   - `own_columns[0]` = 0;
   - `own_dropped` is None;
   - the mean MINRES iterations per step from step 2 on are below the `own` = 0 run's over the same steps.
2. **The wiring:** at each rebuild:
   - the written set equals `_own_set` on the spied inputs, and some step writes 12 columns;
   - each written column's Rayleigh quotient matches its μ to $10^{-10}$ relative (kills a stale $Y$);
   - the known region still equals `Z` (kills a wrong offset);
   - every real column $z$ (known and own) satisfies $\|P_Z A z\| / \|A z\| \le 10^{-8}$ through `_zop` (kills the prefix trim and an H missing own slots).

   Precondition: some step has more candidates with $\mu \ge$ lo than 12.
3. **`Z=None`, `own` > 0:** the pairs match eigsh, `own_columns` > 0, and `_deflate_setup` is not called before the first own write. **`own` = 0:** `_own_write` is never called and `own_columns` is all 0. **`own` < 0, a float or a bool:** ValueError.
4. **Rebuilds:** `_zblock` is called once. `_own_write` runs exactly at the non-final steps with a non-empty own set (the count from the spies, here steps − 1), never after the last step.
5. **Fallback:** a spy on `_own_write` injects a crafted column: half a known mode below lo plus a vector with $y^T A y \approx 0$ (review's `schur.py` construction). Then:
   - the fallback fires, and `own_dropped` is that step;
   - `own_columns` is 0 afterwards;
   - `tau_eff` is not raised;
   - the pairs match eigsh.

   Also once with `Z=None`.
6. **σ move:** `test_slice_deflated_move` with `own` > 0:
   - the move happens and `own_columns[0]` = 0;
   - every `_deflate_setup` after the move sees the moved σ (the existing `len(sig) == 2` becomes "the first is the original σ, all later the moved one");
   - the pairs match.
7. **si_checked:** extend `test_checked_keeps_z`: each run gets the same `Z` and the same `own`, and the restart's `own_columns` is > 0 at some step.
8. **si_band and solve_modes** (fake slices), with `max_deflate` = $d$:
   - each slice gets `own` = min(64, $d$//2) and the $d$ − own nearest known columns: $d$ = 3 gives own 1 and known [0, 2, 2] in `test_band_z` (`stored[-(d - own):]`); $d$ = 200 gives own 64 and known min(136, stored); $d$ = 0 gives own 0;
   - `own` in `slice_options` raises;
   - `test_stagnation_risk` is recomputed with $d$ − own, including the new warning text;
   - `test_deep_band`'s "max_deflate ≥ j" check uses $d$ − own.
9. **Memory:** `test_z_peak` with `own` > 0 (peak), and `test_memory_formulas` for the per-part padding.

Mutations to try in code review:
- in `_own_set`: the mutants above;
- the prefix trim kept;
- a stale $Y$;
- a wrong slot offset;
- a rebuild on every step;
- the fallback removed, or raising `tau_eff`;
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
  - 189: the warning text ($d_Z - \mathrm{own}$ of $j$), the "up to $d_Z$" clause, and the own-modes residual risk resolved;
  - 196: $d_Z$ = known plus own;
  - 786: si_checked passes `Z` and `own` to every run, with the restart note;
  - 829: si_band's Z per item plus own;
  - 844: si_memory's $n_z$ = known plus own, padded per part ($z_p$);
  - 880–887: the PR 18f deep-band paragraph, its "PR 18g" note, and the `slice_study.py` link with `--parts`;
  - the MINRES deflation section: own columns, `_own_set` and its filter, the slots, $E$/$H$ over the padded width, the rebuild, the fallback and why (exact, but conditioning not guaranteed on an indefinite $A$), the restart note;
  - si_slice's options and info (`own`, `own_columns`, `own_dropped`);
  - the measured section (the study's tables);
  - the test list.
- **Docstrings:**
  - si_slice: `own`, `own_columns`, `own_dropped`;
  - si_checked (shift_invert.py:587ff): `own` passed to every run;
  - si_band and solve_modes: `max_deflate` counts known plus own, `own` is rejected by the key check;
  - `_slice_plan` and `si_memory`: the block is known plus own;
  - `_deflate_setup`: no trim;
  - `stagnation_risk`.
- **`docs/ROADMAP.md`:**
  - line 18: the 18g merged entry, order from here 18h, 20, …;
  - lines 92 and 94: the 52× ratio and the per-step iterations, from the deep part;
  - the 18g section (691ff):
    - Contents and Tests rewritten to this scope: own columns on both sides of σ;
    - "σ toward the unconverged neighbour" mapped to the remainder-σ candidate of §3;
    - then the status and results;
  - "Deferred GPU checks": the rebuild seconds against the step's MINRES seconds at 10M DOF, and the host pinv at several thousand $n_z$;
  - the order line 879 and the dependency rows 869–871 need no change.
- **`docs/open_items.md`:**
  - 175: the warning now also at $j \le d < j + \mathrm{own}$;
  - 348–354: the section and each item closed with the numbers, or kept for a follow-up;
  - 395: the "→ PR 18g" note;
  - 155 and 335: the PR 18f residual-risk notes;
  - the decisions below.
- **Memory:** the autonomy line (18g merged, next 18h).

## Decisions without user input

- Own columns from every evaluated candidate with $\mu \ge$ lo, both sides of σ:
  - no residual gate, because the measured gain is 0.45× against 0.59× at $10^{-3}$;
  - the below-lo neighbour is excluded, because it is a near duplicate of a known column and slows MINRES;
  - the filter $|\mu-\sigma| \ge \rho^2|\mu|$ guards against ghosts at σ;
  - a fallback switches own deflation off for the run when a solve with own columns fails its check. Exactness holds, conditioning does not.
- Own slots: min(64, $d_Z$//2) inside the plan's $d_Z$, appended after the known columns. Half stays known, so a small $d_Z$ still deflates from step 1. At $j \le d < j + \mathrm{own}$ the slots take the farthest known columns, and `stagnation_risk` (known part, $d - \mathrm{own} < j$) warns; the probe favours own columns there.
- One block, one compile, in-place writes, no host vectors. $E_{KK}$ is kept on the host, and a rebuild forms only the own columns' block of $E$.
- The remainder σ rule and the other deferred items are measured here, and changed in a follow-up only if a kind takes ≥ 10% of the iterations.
