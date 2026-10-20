# PR 18 plan: performance pass, part f: deep bands

Branch `pr-18f-deep-bands` from `main` (0dc91a5, PR 18e merged). Roadmap: "PR 18f: Deep bands".

## Revisions

- Rev 1: first draft.
- Rev 2 (plan review round 1: none satisfied):
  - **Deflation:**
    - Z-only form: no stored $AZ$, one extra $A$ apply per iteration, half the memory.
    - $H = E^+$ by `np.linalg.pinv(hermitian=True)`.
    - Zero-norm columns are guarded.
  - **Memory:** a capacity floor from the modes still to compute, and the need cap from $N_s - D$.
  - **Junction:** the cut, the known leaf in solve_modes, the rebuild before the count.
  - **Levers:** maximum reach for an unreached known depth, and otf back to stored.
  - **Options, tests, benchmark, docs:** per the reviewers.
- Rev 3 (plan review round 2: none satisfied):
  - **Junction, simpler:**
    - The known block is truncated at the cut. The first slice recomputes the at most $w$ locked modes above it.
    - The known columns form the result's leading columns, so the assembly appends in place: no holes, no `_permute` in the assembly, no `r_s` seed, no top remainder of known modes only.
    - `known` is passed as a list that si_band empties (a tuple's reference defeats `resize`).
    - The in-place path requires an owned F-order block.
  - **Deflation:**
    - $R$ is not in $Z$, so `deflated` = `len(Z)`.
    - There is one helper, `_dminres`, plus `_deflate_setup`, and a shared `_upload` for $W$ and $Z$.
    - A column is solved directly when $\|P_Z b\| \le \tau\|b\|$.
    - $E$ is accumulated in chunks of $4c$.
    - $P_D$ is renamed $P_Z$.
  - **Memory:**
    - `_slice_plan` maximises $d$ first, at a low capacity, then the capacity beside it. Rev 2's floor gave $d = 0$ whenever the budget binds.
    - The 1M claim is corrected with the reviewer's table.
    - $H$ and the $E$ chunk count as device terms.
  - **Levers:**
    - Rule 2 takes the maximum reach among the states whose junction deflates fully, else the maximum reach.
    - otf → stored compares the undeflated counts.
  - **`stagnation_risk`:** also without a junction, and with `max_deflate` below $D$.
  - **Options:** si_band's `max_deflate` defaults to 0; solve_modes sets it. Bools are rejected.
  - **Tests:**
    - `id`, not weakrefs; spies copy what they capture.
    - The x reference at a MINRES tolerance of $10^{-12}$ in a real gap.
    - Test 7 counts the host copies of $Z$.
    - The partial case runs at the si_band level.
    - The branches listed by the reviewers are covered.
    - The full list of changed tests.
  - **Benchmark:** no undeflated reruns (cited); a band of 20 modes at depth 1000; single solves as their own part; times estimated.
  - **Goal:** iterations still grow with depth; the acceptance is a ratio.
  - **Docs:** the reviewers' added locations and a catch-all grep.
- Rev 4 (plan review round 3: performance satisfied):
  - **Junction:**
    - The count runs before the truncation, on $j + 1$ columns.
    - $j = 1$ copies: an $(n, 1)$ block is also C-contiguous, so `resize` would scramble it.
    - The in-place condition is now stated operationally.
    - solve_modes drops its own `V` before calling si_band.
    - The cut's threshold includes tol.
  - **The known leaf is recorded by si_band** (passed in `known`), not inserted by solve_modes.
  - **Rule 2 removed:** unreached known depths fall through to rule 3's 5% band, with no `_slice_plan` inside `_choose`.
  - **No solved-column special case:** `_minres` already handles it.
  - **A C-order `known` block raises.**
  - **Setup phase modelled** (`op_memory(4c)`).
  - **otf → stored:** keeps otf when the stored plan does not fit, and counts the aggregates.
  - **Tests:**
    - corrected values (`test_filter_capacity_sorted`);
    - stubs and spies (`stub_band` keys, `_slice_plan` spy, `fake_band`);
    - the `max_deflate` min;
    - duplicate branches;
    - test 7's sensitivity assert;
    - the benchmark acceptance at `recheck_1005`'s σ with $u$ = 0.
  - **Docs:** `:182`, `:155`, the benchmark paragraph, signatures and notation.

## Split (decision without user input)

The roadmap's 18f holds four groups. Two of them, the junction and the deflated MINRES, are what the depth-1000 remedy needs; the levers after a capacity stop depend on them. The fourth group, the slice-efficiency items of PR 15c/15d, is independent of depth and moves to its own PR:

| PR | branch | contents |
|---|---|---|
| 18f | `pr-18f-deep-bands` | this plan: the LOBPCG/slice junction, MINRES deflated by the known modes below σ, its memory, levers after a capacity stop, the σ-move retry policy (decided by the study), the depth study |
| 18g | `pr-18g-slices` | new: the slice-efficiency items (verification slices, σ toward the unconverged neighbour, the top-remainder neighbour, a slice's own modes below σ), with its own plan. They are measured first, so items may close without code. |
| 18h | `pr-18h-calculix` | the old 18g (CalculiX comparison), unchanged |

Dependencies: 18g ← 18f; 18h ← 18a–18g; 19 ← 17, 18a–18g. Order: 18f → 18g → 18h → 20 → 21 → 22 → 23a → 19 → 23b (all of 18 runs on the Mac CPU).

## Probe (planning, scratchpad `p18f/adefl_probe.py`, not committed)

Setup:
- Hex20 $16^3$ (56k DOF, clamped `xmin`), AMG at σ, 6 columns, τ = $10^{-8}$.
- σ mid-gap above the first `depth` modes.
- The reference modes of PR 18e's probe. Their λ-scaled residuals reach $5.6\cdot10^{-7}$, about LOBPCG's accuracy at tol $10^{-6}$.
- The A-orthogonal deflation (below) with $Z$ = the known modes below σ, except the $u$ nearest below σ. Those stay unknown: they stand for a slice's own modes between its lower edge and σ.

| depth | none (18e probe) | $u$ = 0 | $u$ = 10 | $u$ = 30 | $u$ = 60 |
|---|---|---|---|---|---|
| 197 | 1500 its, 61.9 s | 140, 6.7 s | 240, 11.0 s | 400, 17.9 s | 560, 24.5 s |
| 503 | 4960 its, 203.7 s | 220, 11.0 s | — | 650, 31.2 s | 980, 46.1 s |
| 987 (plan review round 2) | 14370 (`shift_invert_study` `recheck_1005`) | 320 | — | 1080 at $u$ = 40 | 1560 at $u$ = 80 |

**Accuracy.** Every solve's true residual $\|b - Ax\|/\|b\|$ is at most $10^{-8}$, even though the deflated modes carry residuals of $5.6\cdot10^{-7}$.

**Partial deflation** at depth 503, $u$ = 30:

| deflated | iterations |
|---|---|
| nearest 100 | 3250 |
| nearest 250 | 2090 |
| lowest 250 | 2460 |
| all 473 | 650 |

**Cost per iteration.** At depth 987 (56k DOF) one iteration takes 0.059 s in the Z-only form, 0.055 s in the $AZ$ form and 0.043 s undeflated.

**What this shows.**
- The iterations still grow with depth. The rest of the cost is the slice's own $u$ modes below σ, an 18g item.
- The deflation still cuts them by more than 10×.

At hex20 $44^3$ (1.08M DOF, M3 Max):
- One $A$ apply at $b$ = 6 takes 0.12 s, the AMG V-cycle 0.64 s.
- The deflation update ($Z^\top y$ and $Z\,t$) runs at 0.20 s per 1000 columns, memory-bound.
- $Z^\top(AZ_j)$ at $d$ = 1000 takes 20 s in chunks of 6 and 5.3 s in chunks of 96. The $d$ applies of $A$ add about 20 s, so a setup costs about 25 s per σ (70 s at $d$ = 2000), negligible against a 1M slice.

## Goal (18f)

Bands past LOBPCG's basis cap continue from the locked modes instead of from 0. Every slice's MINRES deflates the known modes below σ, cutting MINRES iterations at depth 1000 by more than 10× (to at most 0.1× undeflated). The levers stop discarding LOBPCG reach. Every pair still passes the same acceptance tests, so the results are unchanged up to the inner-solve accuracy. The band's slices differ: they now start at the junction cut.

## Design (18f)

### 1. Deflated MINRES (`eigen/shift_invert.py`)

**Notation.**
- $A = K - \sigma M$, masked: `ap_operator(op, free, -σ)`.
- $Z$ is the device block of the $n_z$ MINRES deflation vectors, the known modes below σ.
- $Z$ is distinct from two other blocks:
  - the Lanczos deflation window $W$ inside the basis $Q$, which si_slice projects out of the Krylov space as before;
  - the rigid modes $R$, which stay as today. Deflating them too would be negligible: at most 6 columns.
- $E = Z^\top A Z$ is symmetric, $n_z \times n_z$, and $H = E^+$.

**The deflated system.** With

$$P_Z = I - A Z H Z^\top, \qquad P_Z A = A - AZ\,H\,(AZ)^\top \ (\text{symmetric}),$$

MINRES (`linalg.minres`, unchanged, with si_slice's SPD preconditioner $P$) solves $P_Z A\,\tilde x = P_Z b$. The solution is

$$x = Z H Z^\top b + P_Z^\top \tilde x.$$

**Exact for any symmetric $H$.**
- $A P_Z^\top = P_Z A$, so $b - Ax = P_Z b - P_Z A\tilde x$.
- The residual MINRES checks is therefore the true residual, in exact arithmetic. This needs neither accurate eigenvectors nor $H = E^{-1}$.

**For $H = E^+$:**
- $HEH = H$ (a Moore–Penrose identity), so $P_Z^2 = P_Z$: $P_Z$ is a projector, also for a singular $E$.
- $Z^\top P_Z b = (I - EH)Z^\top b$ vanishes on the range of $E$.
- The null-space drift in $\tilde x$ is harmless, since $P_Z^\top Z = Z(I - HE)$ vanishes on the range of $E$ too.

**Exact eigenvectors.** Then $P_Z A = \Pi^\top A \Pi$ with $\Pi = I - ZZ^\top M$, 18e's probe operator. The preconditioned operators $P\,P_ZA$ and $\Pi P \Pi^\top A$ share their nonzero eigenvalues.

Plan review rounds 1 and 2 checked all of this on dense pencils with inexact, duplicated and near-duplicated $Z$. Even with $|H|$ near $10^9$, true and MINRES residuals agreed to about $10^{-12}$.

**Z only, no stored $AZ$.**
- operator: $P_Z A v = A\,(v - Z H Z^\top A v)$, two $A$ applies;
- right-hand side: $P_Z b = b - A(ZHZ^\top b)$;
- solution: $x = ZHZ^\top b + \tilde x - ZHZ^\top(A\tilde x)$.

In exact arithmetic this equals the $AZ$ form, at $8n\,n_z$ device bytes instead of $16n\,n_z$. The price is one $A$ apply per iteration: +7.5% at 56k DOF, about +11% at 1M with AMG.

**$H$.**
- `np.linalg.pinv((E + Eᵀ)/2, rcond=1e-10, hermitian=True)`, symmetric by construction.
- Duplicate store columns (two runs' copies of one mode, merged only at the end) drop out.
- $E$ is accumulated as $Z^\top (A Z_j)$ in chunks $Z_j$ of $4c$ columns, $c$ = `_chunk(block)`. Each chunk reads all of $Z$, so larger chunks are faster: at 1M, 5.3 s instead of 20 s for chunks of 6.

**Tolerance.**
- MINRES runs at the per-column tolerance $\tau\,\|b\|/\|P_Z b\|$ (`minres` broadcasts an array `tol`). The ratio $\|P_Z b\|/\|b\|$ can lie on either side of 1.
- The returned relative residuals are scaled back by $\|P_Z b\|/\|b\|$. So `tau`, $\tau_\text{eff}$, the stagnation test and the σ-move threshold keep their meaning.
- No special case for columns already solved by $ZHZ^\top b$: the tolerance is $\tau\|b\|/\max(\|P_Z b\|, \text{tiny})$, and `_minres` already converges a zero column at once and a column with tolerance ≥ 1 at its first check.

**si_slice.**
- New argument `Z`: a sequence of host 1-D columns, never returned. si_band's store columns are such views.
- `_upload(cols, n, c)` (shared with $W$) fills a zero device block of $\lceil n_z/c\rceil c$ columns in chunks of $c$ via `_insert`, with no full host or device copy. A reviewer measured +0.39 GB maxrss for a 3.84 GB block.
- `_deflate_setup(A, Zd, c)` returns $H$.
- `_dminres(A, rhs, P, Zd, H, tol, maxit)` does the right-hand side, MINRES on the `Partial` operator, the recombination and the residual scaling.
- `Zd` and `H` are jit arguments, never closure constants. (A reviewer's closure capture duplicated them as 8.6–17 GB of constants.)
- On a σ move $H$ is recomputed: $A$ changes, $Z$ is kept.
- `info["deflated"]` is `len(Z)`, 0 without `Z`.
- The setup's time joins `time["setup"]`; a σ move's recompute joins the move's.

**si_checked** passes `Z` through `slice_options` unchanged. Its restarts keep `Z` (same σ), and each run recomputes $H$.

### 2. Slicing with deflation and known pairs (`si_band`)

**Z per work item.**
- $Z$ is the stored modes with $\lambda < \sigma$ (σ the item's midpoint), the nearest `max_deflate` of them.
- `max_deflate` is a new argument, an int ≥ 0, default 0. solve_modes sets it from `_slice_plan`, so direct si_band calls stay unbudgeted and unchanged.
- $Z$ is built per item as a list of store columns (views) and dropped after the item.
- Window modes below σ are included. They are also in the Lanczos window $W$, and the two roles do not interact: $W$ projects the Krylov space, $Z$ the inner solve.
- `"Z"` joins the slice options si_band sets itself (the reject set).

**`known=[lam, V, residuals, thresholds, leaf]`** (new argument, None by default):
- The locked pairs below the cut, ascending. `V` is the host block, fp64 and F-order (else a `ValueError`).
- `leaf`: None, or `(interval, count)`, the known leaf's interval and count record from solve_modes (section 3). si_band puts `{interval, kind: "known", leaf: True, count}` first in `slices` and the pair into `leaves` before the loop, so `info["count"]` covers it on the normal and the error exit alike.
- si_band takes the list's items and empties it (`known.clear()`), so the caller's list holds no reference.
- They enter the store as run −1 (views of `V`'s columns). Their windows close the first slice's lower side as deflated $W$ modes, as any stored pair would.
- Nothing else changes: the work starts at `lo`, and `r_s` comes from the slices as today.

**Assembly in place.** The known pairs all lie below the cut, and the slices start at or above it. So in the merge's ascending layout the known columns are normally the first $j$ parts, unmerged and in order. The in-place path applies when `order[:j]` is `arange(j)`, every window covering those columns holds run −1 only, $j \ge 2$ and `V` owns its data. ($j = 1$: an $(n, 1)$ block is also C-contiguous, and `resize` lays the grown array out in C order, scrambling it without error.) Then:
1. the store's views of `V` are dropped and `gc.collect()` runs;
2. `V.resize((n, j + n_\text{rest}))` keeps the leading columns, since F-order data is column after column;
3. the other parts are written from column $j$ on.

Otherwise the assembly copies as today, from `V` and the store by index, with the "returned modes copied" `RuntimeWarning`:
- a merged window holding a known column: a mode LOBPCG missed below the cut, or a slice whose $r_\text{acc}$ is large enough to chain across the cut's gap;
- $j = 1$;
- a resize failure (another reference, or a borrowed block).

A C-order block is rejected up front: `resize` on it scrambles columns without error.

The `err.result` path assembles the same way. A reviewer measured an 8 GB F-order array grown by 10% with maxrss unchanged.

### 3. The junction (`solve_modes`, `eigen/api.py`)

After a capacity stop (`strategy="auto"`):

1. **Sort.** The locked modes are sorted as on the "fits" path (`_permute(V, order)` also when stopped), with `info`'s residuals and thresholds. The aggregates are kept until the count is done.
2. **The cut** `_cut(lam, t, w)`, with $w = \min(m, c)$ ($m$ = LOBPCG's block `info["block"]`, $c$ = `_chunk` of the slice block, 24 at $b$ = 6; $D$ = the number of locked modes at the stop):
   - It takes the midpoint of the largest relative gap $(\lambda_{i+1} - \lambda_i)/\lambda_{i+1} \ge 100\max(t_\text{top}, \text{tol})$ among the top $w + 1$ locked modes. The tol term keeps the cut wider than the assembly's window chain ($100\,r_s$), so a slice mode does not merge with a known column (that would copy, at a host peak of about $2j$). $t_\text{top}$ is their thresholds, not all $D$: the lowest modes' round-off floors must not block it.
   - The midpoint must lie below hi, since in band mode a locked mode can reach $\lambda_{hi}(1 + 100t)$.
   - $j$ is the number of modes below the cut. At most $w$ locked modes lie above it, and the first slice recomputes them; a missed mode is most likely near the top, where the next chunk would have returned it.
   - With no qualifying gap (or $D < 2$) there is no junction. LOBPCG's modes are freed as today, and the band runs from its start, deflated by its own finds.
3. **otf back to stored** (section 5).
4. **The count**, only when the cut lies above lo. Below lo the known pairs are not returned, so their completeness is not part of the band.
   - `_verify` runs as on the "fits" path on `lam[:j+1]` and `V[:, :j+1]` (`count` needs every mode up to `lam_next` in $W$), with `n_band` = $j$, so it counts over [0, cut] with the first locked mode above the cut as `lam_next`.
   - It takes LOBPCG's `t_iteration`, block and solve time, with `reuse` and `probe_budget` as there and the final `op_opts`.
5. **Truncate**, with no jax op after the count (the rule at api.py:779–788): `gc.collect()`, then `V.resize((n, j))`. The untouched columns are lazily paged zeros. A failed resize keeps `np.array(V[:, :j], order="F")` with the warning. solve_modes then builds `known = [lam[:j], V, r, t, leaf]` and drops its own names (`V = None`), so si_band's list holds the only reference.
6. **The leaf.**
   - solve_modes passes `((lo_b, cut), count)` as `known`'s leaf (lo_b = the band's lower edge), with `count` in `_count_out`'s keys (`passed` and `verifiable` False by default, `estimate`, `estimates`, `flagged`, `restarts` 0, `ran`, `reason`, `t_leaf`). si_band records it (section 2).
   - A flagged count is not rerun: rerunning a span of $j$ modes would deflate more than any slice's capacity. The leaf stays flagged, as on the "fits" path.
   - This is a regression against today's capacity-stop path, which slices $(\text{lo}, \text{cut})$ again and would find a mode LOBPCG missed there. It is logged.
7. **si_band**, called with:
   - `known`, lo = max(lo_b, cut), `edges` at or below the cut dropped;
   - `max_deflate` = the user's cap or $d$ (section 4), whichever is smaller.

   The store is $N_s = j + N_\text{new}$ with $N_\text{new} = \lceil 1.25\,\max(n_\text{modes} - L_j, 0)\rceil$ when `n_modes` is given, else 200. $L_j$ is the number of known modes in the band. On `edge_rr_failed`, `complete_to` becomes lo_b; otherwise it is si_band's (at least the cut).
8. **Info.**
   - `info["junction"] = {"cut", "known": j}` (None without a junction).
   - `from_zero` and `mu1_ub` keep their meaning, applied to lo_b.
   - `merged_edges` stays si_band's: the cut is not among its bounds (logged).
   - `stagnation_risk` = depth ≥ `STAGNATION_DEPTH` and (no junction or $\min(\texttt{max\_deflate}, d) < j$). $j$ is a lower proxy: later items also deflate the band's own finds below σ.
   - The capacity-stop warning keeps "N modes locked". It says the band continues from the cut with shift-invert slices. Under the risk it adds "MINRES may stagnate: deflation limited to $d$ of $j$ known modes" (or "no junction").

The "memory" path (no chunk fits) and forced "shift_invert" have no junction. Their slices from 0 deflate what earlier slices found, up to $d$. A band with $f_1 > 0$ that starts at $\lambda_{lo}$ deflates only the band's modes found below each σ (logged: the modes below $f_1$ are not computed for it).

### 4. Memory (`memory.py`)

**`si_memory(..., n_z=0)`:** the columns of $Z$, distinct from `n_deflate` (the Lanczos window $W$ inside $Q$). When $n_z > 0$ it adds:
- device storage: $8n\lceil n_z/c\rceil c$ and $8 n_z^2$ (`H`);
- the setup phase: input and output chunks $2 \cdot 8n \cdot 4c$, $8 n_z \cdot 4c$ and `op_memory(4c)["workspace"]`;
- the MINRES phase: one $n \times b$ block (the second $A$ apply's output) plus $n_z \times b$;
- host: $16 n_z^2$ ($E$ and $H$).

The storage and host terms are exact (`device_exact`, `host_exact`), the blocks modelled. The host store's `n_modes` must include the known columns, so on unified memory $Z$ costs twice: once in the store, once on the device.

$\lceil x\rceil_c$ rounds up to a multiple of $c$.

**`_slice_plan(op_memory, pc64, n, *, n_store, n_new, block, n_rigid, budget, host_budget, capacity=None)` → (C, d)**, where $n_\text{store} = N_s$ (the host store's columns, the cap on $d$) and $n_\text{new}$ the new modes (the need cap's count):
1. $d^*$ is the largest $d \le n_\text{store}$ fitting at the low capacity $C_\text{lo} = \min(\lceil 2\,\texttt{SLICE\_MIN}\rceil_c, C_0)$, by `_largest`. $C_0$ is `_slice_capacity` at $n_z$ = 0; None raises the slice-floor error, as today.
2. $C$ is the largest capacity fitting beside $d^*$, capped at the need cap $\lceil 1.25\,(1.8\,n_\text{new} + 84) + 2c\rceil_c$.

Why deflation comes first:
- A capacity column and a deflation column cost about the same.
- Halving the capacity adds slices, each with the fixed 84 columns. Three slices of 288 cost about 20% more Lanczos columns than one of 720.
- An undeflated mode costs MINRES 6.5–12 iterations per step (the probes).

With a given `capacity`, $d$ is the largest fitting beside it; `_largest`'s None becomes $d = 0$. `_slice_capacity` gains `n_z` and takes its need cap from `n_new`. On the paths without a junction, `n_new` = `n_store` as today, and with $d = 0$ the plan gives today's $C$.

**The assembly:** in place (section 2), so the host peak stays the store plus one slice block.

**At 1M DOF** (a reviewer's emulation, hex20 $44^3$, 0.8 × 64 GB, stored operator):

| $D$ | $N_\text{new}$ | $d$ at $C$ = 192 | at $C$ = 288 | at $C$ = 720 |
|---|---|---|---|---|
| 2000 | 50 | 2050 | 2050 | — |
| 2000 | 250 | 2250 | 2250 | 1176 |
| 2600 | 50 | — | 2064 | — |
| 2600 | 250 | 2160 | 1872 | 576 |

- LOBPCG's $C_\text{max}$ at 1M is 2112–2720 (chunks 40 to 5). So a junction there happens only at $D \approx 2000$–2700, where $d < D$ for large $N_\text{new}$.
- A depth-1000 band at 1M never reaches the junction: LOBPCG fits.
- **Zero-copy lever, logged as the 1M limit (deferred).** On the CPU, `np.asarray(jax_array)` shares a jax array's buffer. So holding the known block once on the device as $(D, n)$, with the host store as its `.T` view, would roughly double $d$.

### 5. Levers after a capacity stop (`eigen/api.py`)

**Rule 2 removed** (`_choose`'s `if needs: return None`): a known depth that no state reaches falls through to rule 3, the cheapest state within 5% of the best used reach.
- Rule 2 existed because a capacity stop discarded its LOBPCG work. With the junction, reach saves slice modes.
- Maximum reach is not taken: each mode of reach also costs two columns at the junction (the store and $Z$), so beyond full deflation more reach can cost more MINRES than it saves (a reviewer's 1M emulation). The 5% band bounds the reach given up; the trade-off at the 1M junction is logged with the deferred zero-copy item.
- This resolves the 18e decision "Rule 2 can forgo levers".

**otf back to stored**, when the levers set otf, before the count (section 3):
- `_slice_plan` runs with both models, with the AMG aggregates still held for the count (`_nbytes(aggs)`) as host extra. The operator is rebuilt stored when $d_s \ge d_o - c$ and $C_s \ge C_o - c$; a stored plan that does not fit (the slice-floor error) keeps otf.
- Iterations follow the undeflated count $u + j - d$ ($u$: a slice's own modes below σ), not $d$. otf costs about +50% per deflated MINRES iteration (two applies, +32% each at 1M), so the rule allows at most one slice chunk $c$ of loss.
- The otf operator is freed first, and `op` is rebound so that `make` and `reuse` see it. `op_memory` and the count block are recomputed with the stored model. The aggregates stay valid, since they depend on the mesh graph only.
- `info["levers"]` stays LOBPCG's choice.

**A LOBPCG rerun with levers: closed.** The junction keeps the locked modes, and the levers already raise the reach up front. A rerun would repay the depth already reached. Unknown depths with a fitting chunk still get no levers (logged).

### 6. Options and closures

- **`slice_options={"max_deflate": n}`** (SLICE_OPTIONS):
  - an int ≥ 0; bools are rejected;
  - solve_modes takes it out of the options forwarded to si_band and passes $\min(n, d)$;
  - 0 turns the deflation off (comparisons, fallbacks).
- **The σ-move retry policy:** decided by the study. If the depth-1000 band shows no depth-driven σ move with deflation, the item closes. Otherwise it goes to 18g with the measurement.
- **Closed:**
  - fp32 storage of $Z$ on the Mac: exactness holds for any stored $Z$, but fp32 changes the deflation's effect, and the halved memory is a GPU-side question. A deferred GPU check.
  - Deflating modes above σ: after preconditioning they lie far on the positive side, and there is no measured gain.
  - A slice-level band from 0 for $f_1 > 0$ bands without LOBPCG: it would compute every mode below $f_1$ by slices.
- **Deferred (logged):** the zero-copy known block (section 4), the 1M limit.

## Tests (18f)

References: SciPy `eigsh` and dense NumPy (`tests/test_shift_invert.py`, `tests/test_strategy.py`'s helpers `prob`, `check`, `run`, `fake_builds`, `capacity_stop`, `stub_band`, `harness`).

Spies record `id(V)`, never a weakref: numpy refuses `resize` on an array that has one. Any spy that captures `known`, store or $Z$ columns copies them with `np.array`.

1. **`_dminres`** (dense references, "tiny" and hex20 $4^3$ clamped, $K$ and $M$ assembled; σ in a relative gap ≥ $10^{-2}$ chosen from the spectrum):
   - **Accuracy.** $Z$ is the eigsh modes below σ. MINRES at tolerance $10^{-12}$ gives $x$ to `np.linalg.solve(K - σM, b)` within $10^{3}\kappa\cdot10^{-12}$. At $10^{-8}$ the true residual is ≤ $1.01\,\tau$, both for that $Z$ and with $10^{-3}$ noise added (any $Z$ is exact).
   - **Any symmetric $H$.** With a random symmetric $H$ in place of $E^+$, through the real recombination: $b - Ax = P_Z b - P_Z A\tilde x$ to $10^{-12}$ relative.
   - **Duplicates:** an exact duplicate column ($E$ singular: `inv` in place of `pinv` fails), a dropped near-duplicate (δ = $10^{-6}$, $|H|$ of order 1) and a kept one (δ = $10^{-3}$, $|H| \gg 1$): each leaves the true residual ≤ $1.01\,\tau$. $H$ is symmetric when $E$ is slightly nonsymmetric.
   - **Tolerance scaling,** with $b = M(Zc + 0.01\,\|Zc\|_M\,\text{noise})$ (noise of unit $M$-norm), so $\|P_Z b\|/\|b\| \le 0.1$:
     - a `minres` spy sees $\tau\|b\|/\|P_Z b\|$ per column;
     - the returned residuals are within 2× of `norm(b - A x) / norm(b)`, where unscaled values would be off by ≥ 10×.
   - **Solved columns.** $b = AZc$ ($P_Z b$ = 0 up to round-off for any $Z$) and an exactly zero $P_Z b$ (monkeypatched): residual ≤ τ, no NaN.
2. **si_slice with `Z`** (slow, hex20 8×6×5, a slice above 30 modes):
   - the pairs agree with the `Z=None` run to the acceptance threshold, and with eigsh;
   - `info["deflated"]` = 30;
   - the first-step MINRES iterations are ≤ 0.6× the `Z=None` run's (a reviewer measured 0.39×);
   - one case passes the window modes below σ both as $W$ and in `Z`, checked against eigsh;
   - a monkeypatched first solve that stagnates moves σ, and a spy on `_deflate_setup` sees both σ values;
   - an si_checked restart (count monkeypatched to flag once) keeps `Z`.
3. **si_band's Z selection** (a recording si_checked that saves `len(Z)` and the λ of the columns passed):
   - stored modes crafted just below and just above an item's midpoint: only those below are passed, so `<` against `≤` and above-σ mutations fail;
   - the columns are views of the store;
   - `max_deflate` = 3 gives the 3 nearest, 0 (the default) gives none, None is rejected;
   - `Z` in `slice_options` raises.
4. **`known` in si_band** (fakes and `test_band_merge`'s setup):
   - the known pairs come back first in `lam`, with LOBPCG's residuals and thresholds;
   - `known` is empty after the call; a C-order or fp32 block raises;
   - the leaf: `slices[0]` is the known entry, and `info["count"]` covers it on the normal exit and in `err.result`;
   - `err.result` after the first slice raises holds the known pairs, assembled in place (`id`);
   - **in place:** `id(V_f)` = `id` of the given block, with no `RuntimeWarning`, and every column of `V_f` equals the expected vector (known, then the slices', as `assert_compacted`);
   - **fallbacks, each copying with the warning and the same contents:**
     - a second reference held;
     - a slice that re-finds a known mode (a merged window holding a known column);
   - `fake_band` (`test_shift_invert.py:1175`) is updated where the `max_deflate` or `known` kwarg reaches it, and records `len(Z)`, not `Z` (views would defeat the in-place path).
5. **`_cut`** (crafted λ, t, $w$, hi):
   - the largest gap in the top $w + 1$, with a larger gap just outside that window ignored (off-by-one);
   - the relative-gap denominator ($\lambda_{i+1}$) and the exact $\ge 100\max(t_\text{top}, \text{tol})$ boundary (both terms deciding a case) deciding a case;
   - a large $t$ on a low mode not blocking the cut;
   - a midpoint at or above hi excluded, the next-largest qualifying gap taken;
   - $w$ = 5 and $w$ = 10 picking different gaps on one spectrum (a wrong $w$ fails);
   - dyadic $t$ ($2^{-9}$) for the exact boundary;
   - None for no gap and for $D < 2$.
6. **Junction end to end** (`test_capacity` updated: "short", chunk 5, 35 locked inside the band 25–45):
   - si_band receives `known` with the $j$ pairs below the cut (the 0.0898 gap, 5 locked modes above it ≤ $w$ = 10; $j$ = 30, asserted from `_cut`) and lo = the cut;
   - the LOBPCG `V`'s `id` is si_band's known block, it has $j$ columns, and the returned `V`'s `id` is the same (truncate, append, compaction in place). `test_capacity`'s own `solve` spy drops `seen["V"] = weakref.ref(...)` and `v_dead` for these `id` checks;
   - si_band's `max_deflate` equals `_slice_plan`'s $d$;
   - `aggs_dead` still holds at si_band's entry (the count ran before it);
   - the known leaf is `slices[0]` and passes, and `info["count"]` covers it;
   - $N_s = j + \lceil 1.25 \cdot (22 - L_j)\rceil$ (the value computed in the test from $L_j$);
   - `info["junction"]`;
   - `seen["interval"]` = (cut, hi);
   - the band against eigsh (`check`), with every returned $\lambda \ge \lambda_{lo}(1 - 100t)$ and $V$ holding exactly the band's columns.

   Updated for the junction:
   - `test_capacity_store`: interval (cut, hi) with the cut in the gap above the 20th locked mode; $N_s$ from that $j$. New cases: `n_modes` 30, and none (200).
   - `test_error_result_capacity` (:915): $N_s$ from $j$; max_slices = 1 still raises, now with the first slice at the cut.
   - `test_filter_capacity_sorted` (:840): λ becomes [1, 5, 1.2, 1.4] with lo = 1.1, hi = 9 (gaps 0.167, 0.143, 0.72): cut 3.2, and the count runs on the sorted [1, 1.2, 1.4] with `lam_next` = 5 (`rec["verify"]` asserted sorted).

   Other cases:
   - **No gap:** `harness` with λ spaced 0.5% apart and $t = 10^{-3}$; `info["junction"]` is None and the band runs from its start.
   - **Cut ≤ lo** (locked modes all below the band): no leaf, `_verify` not called, si_band from lo with `known`.
   - **A held V at the truncation** (`harness` with `hold=`, as `test_filter_held`): the copy with the warning under `pytest.warns`.
   - **A flagged leaf** (`_verify` stubbed to estimate 0.6): no rerun; the leaf and `info["count"]` are flagged.
   - **`edge_rr_failed`** (stub_band): `complete_to` = lo_b.
7. **Memory:**
   - `test_memory_formulas` (`test_shift_invert.py:588`) gains the exact `n_z` terms and `device_exact` / `host_exact`.
   - `_slice_plan` on crafted fit functions, checked against a brute-force search over $(C, d)$ as `test_slice_capacity` does:
     - full deflation;
     - $d^* < n_\text{store}$ at $C_\text{lo}$;
     - $C_0 < C_\text{lo}$;
     - a given capacity, with None → $d = 0$;
     - $d = 0$ equal to today's $C$.
   - **A process peak** (slow, `peak`, macOS, PEAK_SETUP style in a subprocess):
     - one si_slice step on voxel $48^3$ / GMG (353k DOF) with `Z` = 800 random masked columns, held as a host block; the slice lies below $\lambda_1$ (so $A$ and $E$ are SPD), with `maxit_inner` 200 and a loose `tau` (the peak, not convergence, is measured);
     - the peak is within 20% of `si_memory(..., n_modes=800, n_z=800)`'s total (about 4.6 GB plus the setup);
     - the sensitivity: peak − `si_memory(n_z=0)` total ≥ 0.8 × (`si_memory(n_z=800)` − `si_memory(n_z=0)`) total (the $n_z$ terms are about 43% of the total, 2.3 of 5.3 GB); about 1 min (memory-bound MINRES, capped);
     - the capacity is at its floor, so the run is about one step.
8. **Levers:**
   - `test_choose`'s rule-2 cases expect rule 3's pick (the cheapest within 5% of the best reach);
   - `test_levers_unmargined`'s 1.2e8 sub-case (:1422; today rule 2 / "memory") expects the new pick;
   - the 18e "Rule 2 can forgo levers" case (slow, voxel $10^3$, `n_modes` 12) now gets rule 3's pick and returns all 12, checked against eigsh;
   - **otf back to stored** (fake builds, `_choose` monkeypatched to otf, a capacity stop):
     - a second operator build without `mode` when the stored plan qualifies, with the otf operator dead (weakref on the operator, not on V) before it;
     - none at $d_s = d_o - c - 1$, and a rebuild at $d_o - c$ (the boundary);
   - `test_levers_applied`: the rebuilt build and the capacity recorded after it (`builds`, `caps` updated);
   - `slice_options={"max_deflate": n}` reaches si_band as $\min(n, d)$: 0, a cap below $d$ (as given) and one above (as $d$), so a min/max swap fails; a bool, a non-int or a negative value raises.
9. **`stagnation_risk`** (`test_stagnation_risk`, :2308; `_slice_plan` monkeypatched):
   - depth 1000 with $d < j$: True, and the warning names "deflation limited to";
   - $d = j$: False;
   - `max_deflate` below $j$: True;
   - no junction: True;
   - depth 999 with $d < j$: False.
10. **Deep band** (slow, about 3 min target, measured in implementation): hex20 $10^3$ clamped, `chunk=40` pinned, a budget that capacity-stops LOBPCG near depth 250, the band from mode 260 to 300, against eigsh at the band's centre ($k$ = 60):
    - every band mode is found, and the leaves are counted (`verify=True`);
    - a recording spy on si_checked: each item's `len(Z)` equals the store's modes below its σ, capped by $d$ from `_slice_plan` at the run's budget (asserted to be the uncapped count here);
    - one first MINRES solve at the first slice's σ, deflated against plain (`_dminres` and `linalg.minres` on copies of the run's store), with the iteration ratio asserted ≤ 0.5 and printed.

    The partial case runs at the si_band level on the first run's copied known pairs, a band of 10 modes with `max_deflate` = 100: every band mode found.
11. **Changed spies:**
    - `test_memory` (3 cases), `test_fp32_slices` and `test_memory_bracket`: a spy on `_slice_plan`'s returned $(C, d)$; `info["capacity"]` and the count's window are asserted against that $C$, and $C_0 \ge 72$ separately;
    - `stub_band` (and through it `harness`) returns `slices=[]` and `merged_edges=[]`, for the junction's tests (test_capacity_store, test_filter_capacity_sorted, test_stagnation_risk, test_levers_applied, test_levers_unmargined, test_levers_rows);
    - `_slice_capacity`'s `n_new` defaults to `n_store`, so its direct calls (`test_strategy.py:480, :1613, :2471`) stay unchanged;
    - `test_options.py:553–560`: `band_set` gains `known`, `slice_set` gains `Z`; `max_deflate` is a SLICE_OPTION and an si_band parameter, so the existing equality pins it unchanged.
12. All other band, strategy and shift-invert tests pass unchanged, except the cases named above (each named in its commit).

## Benchmark (18f)

`benchmarks/deep_band_study.py` (results `benchmarks/results/deep_band_study.json`, per-step progress logged to a file, `--parts band,solves`). Each part is under 1 h, and the estimates come from the plan review measurements.

- **`band`** (about 40 min): hex20 $16^3$, `chunk=40`, `verify=False` (eigsh is the acceptance check; the counts' predicted degree and gap recorded, as si_band does under `verify=False`).
  - The budget capacity-stops LOBPCG near depth 970 (about 6 min).
  - Then the band of modes 980–1000 runs end to end: from the cut, about 30 modes, about 140 columns, about 23 steps of about 60 s.
  - Recorded per slice: MINRES iterations per step, $n_z$, $d$ against $j$, σ moves, `inner_maxit` hits, time.
  - The frequencies are checked against eigsh at the band's centre ($k$ = 60, `maxiter` set; a few minutes).
- **`solves`** (about 20 min): first MINRES solves with $Z$ from a LOBPCG run to 1010 modes cached to disk (about 8.5 min, rerun only without the cache), at depths 500 and 1000:
  - deflated;
  - with the nearest half of $Z$ (the memory-bound case; about 5 min at depth 1000);
  - the deflation's per-iteration overhead.

  The undeflated references are cited, not rerun: 4960 iterations at depth 503 (18e probe) and 14,370 at depth 1005 (`shift_invert_study` `recheck_1005`).
- **Acceptance:** on the `solves` part at `recheck_1005`'s σ and right-hand-side width, with $u$ = 0 (every mode below σ deflated): deflated ≤ 0.1× its 14,370 iterations (the probe: 0.022 at $u$ = 0, 0.075 at 40, 0.11 at 80; the half-Z case reported); the band against eigsh.

## Docs (18f)

- `docs/eigensolvers.md`:
  - Interior bands and strategy:
    - `:141`: the capacity stop → the junction, and the `stagnation_risk` definition;
    - `:145–151` "Why memory-driven": the slice per-mode row and the crossover claim marked undeflated, with the study's figure;
    - `:157`: "33–1260× cheaper" marked undeflated, with the study's figure;
    - `:166`: rule 2 removed (unreached known depths take rule 3's pick), and the residual risks;
    - `:182`: the slice capacity → `_slice_plan` ($C$ and $d$), the need cap from $n_\text{new}$, `n_z` in the fit;
    - `:175`: the 10M reach discussion gains the deflation cap $d < D$;
    - `:183`: the store $N_s$;
    - `:187`: `from_zero` and `mu1_ub` after a junction;
    - `:193`: the filter, sort and compaction after a junction;
    - `:197` Errors: forced "lobpcg" for an unreached known depth, the `slice_options` key `max_deflate`, the discarded LOBPCG run → kept;
    - `:199`: `info["junction"]`.
  - Options table: `slice_options` `max_deflate`.
  - Shift-invert slices (`:716–753`): the MINRES deflation ($Z$, distinct from the Lanczos window $W$ and from $R$), `Z` in the argument table (`:740–753`), `info["deflated"]` in the info list, and the exactness and projector arguments with the two preconditioned operators written out. Unlike `lobpcg`, which stores $R$ and $MR$, the Z-only form stores no $AZ$.
  - Slicing a band (`:765`, the signature at `:767` with `known=None, max_deflate=0`, `:798`): $Z$ per item, `known`, `max_deflate`, the in-place assembly, the "known" slice kind and the known leaf in `count`.
  - Slice memory model (`:802`, the signature at `:804`): `n_z=0`, `_slice_plan`, the 1M table.
  - Measured MINRES cost:
    - `:831–832` ("Depth sets the cost", the $\alpha D(1 + D/D_0)$ model): marked undeflated, before PR 18f;
    - `:835` "Depth 1000 is marginal" → the deflated study figures;
    - a `deep_band_study.py` paragraph beside `:815` (script, `--parts band,solves`, results JSON, setup).
  - Test sections: `:997` (a capacity stop handing the whole band to si_band → the junction), `:1002` (the stagnation warning, new rule), `:1003` (rule 2 at $n_\text{modes}$ 150 → the new rule), plus the new tests.
- `README.md`:
  - the status sentence: "split into 18a–18h", 18f's status, the order;
  - the interior-band paragraph (`:287`): the junction and the deflation, with the "33× … 1260×" ratio marked as undeflated (or replaced by the study's).
- `docs/ROADMAP.md`:
  - the status line, and `:626` ("seven parts … 18g last" → eight parts, 18h last);
  - the 18f section rewritten to its contents, with the acceptance as run;
  - a new 18g (slice efficiency, plus a slice's own modes below σ), and the old 18g → 18h;
  - the dependency rows and the order;
  - `:91`, `:92` (33×/1260×, no crossover): marked undeflated, with the study's figures;
  - `:95` (M3b: the depth-1000 mid-gap solve, now with the deflated figure);
  - `:808` (the PR 15d GPU check "MINRES near depth 1000" → deflated, with the cap);
  - deferred GPU checks:
    - the deflation cap $d$ at 10M DOF, where $Z$ costs 84 MB per column (twice on unified memory), so $d < D$: partial-deflation MINRES near depth 1000;
    - fp32 $Z$;
    - the deflation overhead at 10M.
- `docs/open_items.md`:
  - "Shift-invert cost near mode 1000": stays in place, annotated "PR 18f: resolved on the Mac (the study's figures); the memory cap $d$ at 1M and 10M stays open (the zero-copy lever)";
  - `:175` (the 10M criteria): annotated with $d < D$;
  - "σ-move retries": by the study's result;
  - `:155` ("Memory-driven strategy, no time-based crossover … 33× … 1260×"): annotated "PR 18f: the ratios are undeflated; the deflated figures are in deep_band_study";
  - the "Slice efficiency (PR 15c → PR 18e → PR 18f)" heading at `:332` and the bodies `:336–338`, `:375` ("→ PR 18f") → 18g; the headings at `:340` and `:344` by the study's result;
  - annotated in place with "PR 18f: …":
    - `:156` (the discarded LOBPCG run);
    - `:172` (the old $N_s$);
    - `:183` (locked modes discarded);
    - `:313` (the 18e levers decision: rule 2, otf carried into the slices);
    - `:321` ("Rule 2 can forgo levers": resolved by removing rule 2, annotated in place);
    - `:312` and `:367` (the order);
  - a "PR 18f decisions" section (section 7 below).
- **Catch-all:** before the merge, `git grep -n -e 18f -e 18g -e CalculiX -e "capacity stop" -e freed -e discard -e "rule 2" -e 1260 -e crossover -e stagnation_risk -e STAGNATION_DEPTH -e N_s -e n_store -e "depth 1000" -- docs README.md src`, with every hit checked against the new behaviour.
- Docstrings: `si_slice` (`Z`), `si_band` (`known`, `max_deflate`, the assembly), `solve_modes` (the junction, `max_deflate`, `info["junction"]`), `_choose`, `si_memory` (`n_z` against `n_deflate`), `_slice_capacity`, `_slice_plan`, `_cut`, `_dminres`.
- Memory:
  - MEMORY.md: the autonomy line (18f merged, next PR 18g), the scale line's order with 18h, and the CalculiX line → 18h;
  - `feedback-autonomy.md`: a PR 18f entry (plan and code review rounds, merge);
  - `project-calculix-benchmark.md`: the description and body → 18h, with the renumbering;
  - `project-scale-questions.md`: the order.

## 7. Risks and decisions without user input (logged in `docs/open_items.md`)

- The split (18g slice efficiency, 18h CalculiX) and the order.
- The A-orthogonal deflation in its Z-only form: exact for any $Z$ and symmetric $H$, at half the memory of the $AZ$ form, for one $A$ apply per iteration (+7.5–11%).
- Deflation before capacity in `_slice_plan` (at $C_\text{lo} = 2\,\texttt{SLICE\_MIN}$), and the nearest modes first when memory limits $d$.
- The cut rule, the truncation at the cut (at most $w$ modes recomputed), the known leaf without a rerun when it flags (a regression against re-slicing $(\text{lo}, \text{cut})$), and no junction without a qualifying gap.
- `merged_edges` without the cut.
- Rule 2 removed (rule 3's 5% band for unreached known depths); otf back to stored within one chunk of $d$; the LOBPCG rerun closed; no levers for unknown depths with a fit.
- The σ-move retry policy decided by the study.
- **Residual risks:**
  - $f_1 > 0$ bands without LOBPCG deflate only their own modes;
  - at 1M the junction comes at $D \approx 2000$–2700 with $d < D$ for large $N_\text{new}$ (the zero-copy lever, deferred);
  - at 10M the deflation is memory-capped (a deferred GPU check);
  - the iterations still grow with depth through a slice's own modes below σ (→ 18g);
  - a deflated MINRES iteration costs more (two applies and the $Z$ update: about 1.4× at $d$ = 1000, 1.9× at 2600 at 1M), while `t_leaf` shrinks. `counting._bound` stays a lower bound only through its 3.5× slack for slices, and `verify=None` skips more counts. This is documented, and the deep-band test runs with `verify=True`.
- The roadmap's acceptance "depth-500 and depth-1000 bands against eigsh": the depth-1000 band runs end to end in the study, the test at about depth 260 (the suite's time), and depth 500 as single solves.
