# PR 18 plan: performance pass, part f: deep bands

Branch `pr-18f-deep-bands` from `main` (0dc91a5, PR 18e merged). Roadmap: "PR 18f: Deep bands".

## Revisions

- Rev 1: first draft.
- Rev 2 (plan review round 1: none satisfied):
  - **Deflation:**
    - Z-only form: no stored $AZ$, one extra $A$ apply per iteration, half the memory, so full deflation fits at the junction on 1M DOF (a reviewer's emulation: $d \approx D/2$ with $AZ$ stored).
    - $H = E^+$ by `np.linalg.pinv(hermitian=True)`.
    - Zero-norm columns are guarded.
    - $R$ is not deflated alone (negligible: at most 6 columns).
  - **Memory:**
    - The capacity floor comes from the modes still to compute, not `SLICE_MIN`.
    - The need cap is taken from $N_s - D$.
    - The assembly grows the known block in place instead of copying it (the copy needed $2\times$ the store at the end).
  - **Junction:**
    - The cut lies in $(\text{lo}, \text{hi})$ with at most $\min(m, c)$ known modes above it, and a single tier.
    - The known leaf lives in solve_modes, and si_band takes the pairs only.
    - The leaf record has `_count_out`'s keys.
    - The rebuild comes before the count.
    - `gc.collect()` before the resize, with a view as the fallback.
    - $r_s$ is seeded from the known thresholds.
  - **Levers:** a known depth that no state reaches takes the maximum reach (cheapest among equal reaches; the 5% band would give up more than any lever costs). The otf → stored rule needs $d_\text{stored} \ge 0.95\,d_\text{otf}$.
  - **Options:** `slice_options={"max_deflate": int}` instead of a bool. No `slice_operator` or `time["deflate"]` key. `info["junction"] = {cut, known}`.
  - **`stagnation_risk`:** on $d < D$.
  - **Tests:**
    - pinned chunks, and `deflated` asserted from the plan;
    - a partial-deflation case;
    - single-solve MINRES comparisons;
    - an si_slice-level process peak at about 2 GB;
    - mutation-sensitive tolerance tests;
    - the W/Z overlap;
    - `_cut` edges;
    - the $N_s$ formula exercised;
    - the full list of changed existing tests.
  - **Benchmark:** a depth-1000 band end to end, as the roadmap asks.
  - **Docs:** the reviewers' list of stale locations.

## Split (decision without user input)

The roadmap's 18f holds four groups. Two of them, the junction and the deflated MINRES, are what the depth-1000 remedy needs; the levers after a capacity stop depend on them. The fourth group, the slice-efficiency items of PR 15c/15d, is independent of depth and moves to its own PR:

| PR | branch | contents |
|---|---|---|
| 18f | `pr-18f-deep-bands` | this plan: the LOBPCG/slice junction, MINRES deflated by the known modes below σ, its memory, levers after a capacity stop, the σ-move retry policy (decided by the study), the depth study |
| 18g | `pr-18g-slices` | new: the slice-efficiency items (verification slices, σ toward the unconverged neighbour, the top-remainder neighbour); its own plan, measured first, so items may close without code |
| 18h | `pr-18h-calculix` | the old 18g (CalculiX comparison), unchanged |

Dependencies: 18g ← 18f; 18h ← 18a–18g; 19 ← 17, 18a–18g. Order: 18f → 18g → 18h → 20 → 21 → 22 → 23a → 19 → 23b (all of 18 runs on the Mac CPU).

## Probe (planning, scratchpad `p18f/adefl_probe.py`, not committed)

Hex20 $16^3$ (56k DOF, clamped `xmin`), AMG at σ, 6 columns, τ = $10^{-8}$, σ mid-gap above depth $D$, the reference modes of PR 18e's probe (λ-scaled residuals up to $5.6\cdot10^{-7}$, about LOBPCG's accuracy at tol $10^{-6}$). The A-orthogonal deflation (below) with $Z$ = the known modes below σ, except the $u$ nearest below σ, which stay unknown (a slice's own modes between its lower edge and σ):

| depth | none (18e probe) | $u$ = 0 | $u$ = 10 | $u$ = 30 | $u$ = 60 |
|---|---|---|---|---|---|
| 197 | 1500 its, 61.9 s | 140, 6.7 s | 240, 11.0 s | 400, 17.9 s | 560, 24.5 s |
| 503 | 4960 its, 203.7 s | 220, 11.0 s | — | 650, 31.2 s | 980, 46.1 s |

Every solve's true residual $\|b - Ax\|/\|b\|$ is at most $10^{-8}$ (7.9e-9–9.9e-9), although the deflated modes carry residuals of $5.6\cdot10^{-7}$: the A-orthogonal form is exact for any $Z$, so the 18e caveat about the coupling $Z^\top A\Pi$ does not arise. Partial deflation at depth 503, $u$ = 30: the nearest 100 below the unknown ones give 3250 iterations, the nearest 250 give 2090, the lowest 250 give 2460 and all 473 give 650. The gain comes from deflating essentially all of them; when memory limits the count, the nearest are the better choice.

Plan review round 1 measured at hex20 $44^3$ (1.08M DOF, M3 Max). One $A$ apply at $b$ = 6 takes 0.12 s, the AMG V-cycle 0.64 s. The deflation update ($Z^\top y$ and $Z\,t$) runs at 0.20 s per 1000 columns, memory-bound at about 85 GB/s. $E = Z^\top AZ$ takes 7.1 s at 1300 columns, and $AZ$ in chunks of 6 about 26 s per slice.

## Goal (18f)

Bands past LOBPCG's basis cap continue from the locked modes instead of from 0. Every slice's MINRES deflates the known modes below σ, so MINRES iterations stop growing with depth (the 1000-mode remedy, on the Mac). The levers stop discarding LOBPCG reach. Every pair still passes the same acceptance tests, so the results are unchanged up to the inner-solve accuracy. The slices that the band runs differ: they start at the junction cut.

## Design (18f)

### 1. Deflated MINRES (`eigen/shift_invert.py`)

Notation: $A = K - \sigma M$ (masked, `ap_operator(op, free, -σ)`). $Z$ is the device block of the $n_Z$ MINRES deflation vectors: the rigid modes $R$ and the known modes below σ. It is distinct from the Lanczos deflation window $W$ inside the basis $Q$, which si_slice projects out of the Krylov space as before. $E = Z^\top A Z$ (symmetric, $n_Z \times n_Z$) and $H = E^+$. With

$$P_D = I - A Z H Z^\top, \qquad P_D A = A - AZ\,H\,(AZ)^\top \ (\text{symmetric}),$$

MINRES (`linalg.minres`, unchanged, with si_slice's SPD preconditioner $P$) solves $P_D A\,\tilde x = P_D b$, and the solution is $x = Z H Z^\top b + P_D^\top \tilde x$. For any symmetric $H$, $A P_D^\top = P_D A$, so $b - Ax = P_D b - P_D A\tilde x$: the residual MINRES checks is the true residual, exactly in exact arithmetic. Neither eigenvector accuracy nor $H = E^{-1}$ is needed for this. $H = E^{-1}$ makes $P_D$ a projector, so $Z^\top P_D b = 0$ and the system is consistent. The null-space drift in $\tilde x$ is harmless, since $P_D^\top Z = 0$. (Plan review round 1 checked all of this on a dense $n$ = 300 pencil with inexact and duplicated $Z$ and a random $H$: identity to $5\cdot10^{-15}$.) With exact eigenvectors $P_D A = \Pi^\top A \Pi$, $\Pi = I - ZZ^\top M$ (18e's probe operator), and the preconditioned operators $P\,P_DA$ and $\Pi P \Pi^\top A$ share their nonzero eigenvalues.

- **Z only, no stored $AZ$** (as `lobpcg`'s deflation stores only $R$ and $MR$):
  - operator: $P_D A v = A\,(v - Z H Z^\top A v)$, two $A$ applies;
  - right-hand side: $P_D b = b - A(ZHZ^\top b)$;
  - solution: $x = ZHZ^\top b + \tilde x - ZHZ^\top(A\tilde x)$.
  
  In exact arithmetic this equals the $AZ$ form, at $8n\,n_Z$ device bytes instead of $16n\,n_Z$. The cost is one $A$ apply per iteration (+16% of an iteration at 1M DOF with AMG), with the same deflation traffic ($Z$ read twice instead of $Z$ and $AZ$ once each: 0.252 against 0.262 s at 1300 columns).
- **$H$:** `np.linalg.pinv((E + Eᵀ)/2, rcond=1e-10, hermitian=True)`, symmetric by construction. Duplicate store columns (two runs' copies of one mode, merged only at the end) drop out. $E$ is accumulated chunk by chunk as $Z^\top (A Z_j)$, in chunks of $b$ columns (inside the modelled MINRES workspace).
- **Tolerance:** MINRES runs at the per-column tolerance $\tau\,\|b\|/\|P_D b\|$ (`minres` broadcasts an array `tol`; $\|P_D b\|/\|b\|$ can lie on either side of 1). Its relative residuals are scaled back by $\|P_D b\|/\|b\|$, so `tau`, $\tau_\text{eff}$, the stagnation test and the σ-move threshold keep their meaning. A column with $P_D b = 0$ is solved exactly by $ZHZ^\top b$: its tolerance is set to 1 and it counts as converged with residual 0.
- **si_slice:** a new argument `Z`, a sequence of host 1-D columns (never returned; si_band's store columns are such views). si_slice uploads $R$ and then `Z` into a zero device block of $\lceil (k + n)/c \rceil c$ columns ($n$ = `len(Z)`), in chunks of $c$ columns with `_insert` (no full host or device copy). The two helpers:
  - `_deflate_setup(A, Zd)` returns $H$, from $E$ accumulated in chunks;
  - `_deflated(A, Zd, H)` is a `Partial` operator: $y = Av$, then $A\,(v - Zd\,(H\,(Zd^\top y)))$.
  
  `Zd` and `H` are jit arguments, never closure constants (a reviewer's closure capture duplicated them as 8.6–17 GB of constants). The right-hand side and the recombination are `_deflate_rhs` and `_deflate_x` (one line each, tested). On a σ move $H$ is recomputed ($A$ changes; $Z$ is kept). `info["deflated"]` reports $k + n$; without `Z` (or with an empty one) nothing is deflated and it is 0. Deflating $R$ alone would be negligible (at most 6 columns). The setup's time joins `time["setup"]`, a σ move's recompute the move's.
- **si_checked** passes `Z` through `slice_options` unchanged. Its restarts keep `Z` (same σ). Each run recomputes $H$ (rare).

### 2. Slicing with deflation and known pairs (`si_band`)

- **Z per work item:** the stored modes with $\lambda < \sigma$ (σ the item's midpoint), the nearest `max_deflate` of them (new argument: None = all, 0 = none), passed as a list of store columns (views). Window modes below σ are included: they are also in the Lanczos window $W$, and the two roles do not interact (si_slice's $W$ projects the Krylov space, $Z$ the inner solve). `"Z"` joins the slice options si_band sets itself (the reject set).
- **`known=(lam, V, residuals, thresholds)`** (new argument; None by default): LOBPCG's locked pairs, ascending. `V` is the host F-order block, which si_band takes over: the caller keeps no reference. They are stored as run −1, so `_merge` treats them as a run of their own, and $r_s$ starts at their largest threshold. Nothing else in si_band changes. The work starts at `lo` (solve_modes passes the cut there, section 3), and the known pairs below it close the first slice's lower side as deflated $W$ modes, as any stored pair would.
- **Assembly without a second copy of the known block:** `V_f` is the known block grown in place. Before the merge's final layout, the store's views of `V` are dropped, `gc.collect()` runs, and `V.resize((n, D + n_\text{new}))` keeps the first $D$ columns. The slice and merged columns are written into the tail. A merged window that held known columns writes its first columns into those slots, and any left over become holes. One `_permute` over the full permutation (kept columns ascending, holes last) and a final shrink give ascending order. A reviewer measured an 8 GB F-order array grown by 10%: columns preserved, maxrss unchanged. If the resize fails (another reference), the assembly copies as today, with the existing "returned modes copied" `RuntimeWarning`. The peak is then the existing "1× + a block".

### 3. The junction (`solve_modes`, `eigen/api.py`)

After a capacity stop (`strategy="auto"`):

1. **Sort and keep:** the locked modes are sorted as on the "fits" path (`_permute(V, order)` also when stopped), with `info`'s residuals and thresholds. After `gc.collect()`, `V.resize((n, D))` ($D$ = locked count). The untouched columns are lazily paged zeros, so a failed resize keeps the view `V[:, :D]`, and the assembly then copies. The aggregates are kept until the count is done.
2. **The cut** `_cut(lam, t, w, lo, hi)`: the midpoint of the largest relative gap $(\lambda_{i+1} - \lambda_i)/\lambda_{i+1} \ge 100\max t$ among the top $w + 1$ locked modes, $w = \min(m, c)$ ($m$ = LOBPCG's block, $c$ = the slice chunk width). The gap must have its midpoint below hi: in band mode a locked mode can reach $\lambda_{hi}(1 + 100t)$. Among the qualifying gaps, those with a midpoint above lo come first. At most $w$ known modes then lie above the cut, inside the first slice's window. A missed mode is most likely near the top, where the next chunk would have returned it. No qualifying gap (or $D < 2$): no junction (`known` None, today's path, still deflated by its own finds).
3. **otf back to stored** (section 5), then **the count:** `_verify` on the locked modes, as on the "fits" path. It counts over $[0, \lambda_c]$, with `n_band` the modes below the cut and the first one above it as `lam_next`, and takes LOBPCG's `t_iteration`, block and solve time, with `reuse` and `probe_budget` as there (with the final `op_opts`). Only when the cut lies above lo: below lo the known pairs are not returned, so their completeness is not part of the band.
4. **The leaf:** solve_modes wraps `_verify`'s record as `_count_out` does: `passed` and `verifiable` False by default, `estimate`, `estimates`, `flagged` = verifiable and estimate ≥ 0.5, `restarts` 0, `ran`, `reason` and `t_leaf`. It inserts `{interval: (lo_b, cut), kind: "known", leaf: True, count}` at `info["slices"][0]` (lo_b = si_band's lower edge) and recomputes `info["count"] = _band_count(...)` over the leaf entries. A flagged count is not rerun: rerunning a span of $D$ modes would deflate more than any slice's capacity. The leaf stays flagged, as on the "fits" path. This is a regression against today's capacity-stop path, which slices $(\text{lo}, \text{cut})$ again and would find a mode LOBPCG missed there. It is logged.
5. **si_band** with `known`, lo = max(lo_b, cut), `edges` at or below the cut dropped, and the deflation cap from section 4. The store is $N_s = D + N_\text{new}$, with $N_\text{new} = \lceil 1.25\,\max(n_\text{modes} - L, 0)\rceil$ when `n_modes` is given, else 200 ($L$ the in-band locked count, as today). After si_band, `complete_to` on `edge_rr_failed` becomes lo_b (a merged window can straddle the cut). Otherwise it is si_band's (at least the cut, since the known leaf covers below it).
6. **Info:** `info["junction"] = {"cut", "known": D}` (None without a junction). `from_zero` and `mu1_ub` keep their meaning: the band's lower edge rule, now applied to lo_b, and LOBPCG's bound. `stagnation_risk` = depth ≥ `STAGNATION_DEPTH` and $d < D$. The capacity-stop warning says that the band continues from the cut with shift-invert slices, and, under that risk, adds "MINRES may stagnate: deflation limited to $d$ of $D$ known modes".

The "memory" path (no chunk fits) and forced "shift_invert" have no junction. Their slices from 0 deflate what earlier slices found (section 2). A band with $f_1 > 0$ that starts at $\lambda_{lo}$ deflates only the band's modes found below each σ (logged: the modes below $f_1$ are not computed for it).

### 4. Memory (`memory.py`)

- **`si_memory(..., n_z=0)`:** the columns of $Z$ beside $R$ (distinct from `n_deflate`, the Lanczos window $W$ inside $Q$). Device storage gains $8n\lceil (n_z + k)/c\rceil c$ when $n_z > 0$ ($R$'s $k$ columns copied in). Host gains $16 n_z^2$ ($E$ and $H$). The MINRES phase gains one $n \times b$ block (the second $A$ apply's output), plus $n_z \times b$. Exact terms in `device_exact` and `host_exact`, the block modelled.
- **`_slice_plan(op_memory, pc64, n, *, n_new, n_store, n_exp, block, n_rigid, budget, host_budget, capacity=None)` → (C, d):**
  1. The capacity floor is $C_f = \min(C_0, \lceil 1.25\,(1.8\,N_\text{new} + 84) + 2c\rceil_c)$. $C_0$ is `_slice_capacity` at $n_z$ = 0 (None: the slice-floor error, as today), and the second term is today's need cap, computed from the new modes only. A capacity below that would split the band into more slices, each paying the fixed 50–85 columns, and a capacity column costs one deflation column.
  2. $d$ is the largest $d \le n_\text{exp}$ (= $N_s$) with $C(d) \ge C_f$, by `_largest`.
  3. $C = C(d)$, capped at the need cap.
  
  With a given `capacity`, $d$ is the largest fitting beside it. `_slice_capacity` gains `n_z` and takes its need cap from `n_new`, which equals today's `n_store` on the other paths. On those paths ($D$ = 0) `_slice_plan` therefore gives today's $C$ when $d$ fits fully.
- **The assembly:** in place (section 2), so the host peak stays the store plus one slice block. The fit test needs no new term; test 7 checks the in-place path.
- At 1M DOF this fits: $D$ = 2600 at the budget LOBPCG stopped at gives $d = D$ with $C \ge C_f$ (a reviewer's emulation, with the Z-only form).

### 5. Levers after a capacity stop (`eigen/api.py`)

- **Rule 2 becomes the maximum reach:** a known depth that no state reaches (margined or not) takes the state with the largest used reach, with ties to the cheaper by `key`. Rule 2 existed because a predicted capacity stop discarded its LOBPCG work. With the junction, every mode of reach saves a slice mode, and a slice mode costs about 70 LOBPCG modes even deflated (the 18e ratio of 556 at depth 503, over the probe's 8×). So no lever's factor (≤ 13×) justifies giving up reach, and rule 3's 5% band would give up about 130 modes at $D$ = 2600. This resolves the 18e decision "Rule 2 can forgo levers". Rule 3 (unknown depth, no lever-free fit) keeps its 5% band: there the reach may not be used at all.
- **otf back to stored:** when the levers set otf, before the count (section 3), `_slice_plan` runs with the stored model (`op_opts` without `mode`). If $C \ge C_f$ and $d \ge 0.95\,d_\text{otf}$, the operator is rebuilt stored. The otf one is freed first, `op` is rebound so that `make` and `reuse` see it, and the aggregates stay valid, since they depend on the mesh graph only. The 5% mirrors the other rules, and otf costs about +32% per MINRES iteration at 1M DOF. `info["levers"]` stays LOBPCG's choice.
- **A LOBPCG rerun with levers: closed.** The junction keeps the locked modes, and the levers already raise the reach up front. A rerun would repay the depth already reached. Unknown depths with a fitting chunk still get no levers (logged).

### 6. Options and closures

- `slice_options={"max_deflate": n}` (SLICE_OPTIONS, an int ≥ 0): solve_modes passes $\min(n, d)$ to si_band, and 0 turns the deflation off (comparisons, fallbacks).
- **The σ-move retry policy:** decided by the study. If the depth-1000 band shows no depth-driven σ move with deflation, the item closes. Otherwise it goes to 18g with the measurement.
- **Closed:**
  - fp32 storage of $Z$ (on the Mac; exactness needs the stored $Z$ to be what $A$ is applied to, which holds for any stored $Z$, but fp32 changes the deflation's effect, and the halved memory is a GPU-side question: a deferred GPU check);
  - deflating modes above σ (they lie far on the positive side after preconditioning; no measured gain);
  - a slice-level band from 0 for $f_1 > 0$ bands without LOBPCG (it computes every mode below $f_1$ by slices);
  - zero-copy device views of the store on the CPU (a reviewer saw `jax.device_put` of a C-contiguous transposed view alias the buffer; it depends on JAX internals: a deferred lever, logged).

## Tests (18f)

References: SciPy `eigsh` and dense NumPy (`tests/test_shift_invert.py`, `tests/test_strategy.py`'s helpers `prob`, `check`, `run`, `fake_builds`, `capacity_stop`, `stub_band`, `harness`).

1. **The deflated solve** (dense references, "tiny" and hex20 $4^3$ clamped; $K$, $M$ assembled):
   - $x$ from `_deflate_rhs`, `minres` on `_deflated` and `_deflate_x`, against `np.linalg.solve(K - σM, b)` to $10^{-8}$ relative. $Z$ is the 10 eigsh modes below σ, then the same with $10^{-3}$ noise added to $Z$, which still meets $10^{-8}$ (any $Z$ is exact).
   - With a random symmetric $H$ in place of $E^+$, through the real recombination: $b - Ax = P_D b - P_D A\tilde x$ to $10^{-12}$ relative.
   - A duplicated column of $Z$ leaves $x$ unchanged.
   - $H$ is symmetric when $E$ is slightly nonsymmetric.
   - Tolerance scaling, with $b = M(Zc + 0.01\,\text{noise})$ (so $\|P_D b\|/\|b\| \le 0.1$):
     - a `minres` spy sees $\tau\|b\|/\|P_D b\|$ per column;
     - the returned residuals are within 2× of `norm(b - A x) / norm(b)`, where the unscaled values would be off by ≥ 10×;
     - a zero column ($b \in \text{range}(MZ)$ exactly) counts as converged.
2. **si_slice with `Z`** (hex20, a slice above 30 modes): the pairs agree with the `Z=None` run to the acceptance threshold, and with eigsh. `info["deflated"]` = 30. The first-step MINRES iterations are ≤ 0.6× the `Z=None` run's. One case passes the window modes below σ both as $W$ and in `Z`, checked against eigsh. A monkeypatched first solve that stagnates moves σ, and a spy on `_deflate_setup` sees both σ values.
3. **si_band's Z selection** (a recording si_checked):
   - stored modes crafted just below and just above an item's midpoint: only those below are passed, so `<` against `≤` and above-σ mutations fail;
   - the columns are views of the store (`np.shares_memory`);
   - `max_deflate` = 3 gives the 3 nearest, 0 gives none;
   - `Z` in `slice_options` raises.
4. **`known` in si_band** (fakes and `test_band_merge`'s setup):
   - the known pairs come back in `lam` with LOBPCG's residuals and thresholds;
   - `_merge` sees run −1 against a slice that re-found a known mode;
   - $r_s$ starts at the known thresholds;
   - `err.result` after the first slice raises holds the known pairs;
   - the in-place assembly: `V_f` shares memory with the given block, with no `RuntimeWarning`, and with a second reference held, the copy plus the warning.
   - `fake_band` (`test_shift_invert.py:1175`) is updated where the new `deflated` key or the `Z` kwarg reaches it.
5. **`_cut`** (crafted λ, t, $w$, lo, hi):
   - the largest gap in the top $w + 1$, with a larger gap just outside that window ignored (off-by-one);
   - the relative-gap denominator ($\lambda_{i+1}$) and the exact $\ge 100\max t$ boundary deciding a case;
   - a midpoint at or above hi excluded;
   - a gap above lo preferred to a larger one below it;
   - None for no gap and for $D < 2$.
6. **Junction end to end** (`test_capacity` updated: "short", chunk 5, 35 locked inside the band 25–45):
   - si_band receives `known` with 35 pairs and lo = the cut from `_cut`;
   - the LOBPCG `V` is the known block (`v_dead` flips) and has 35 columns;
   - `aggs_dead` still holds at si_band's entry (the count ran before it);
   - the known leaf is `slices[0]` and passes, and `info["count"]` covers it;
   - $N_s = 35 + \lceil 1.25 \cdot (22 - 10)\rceil = 50$;
   - `info["junction"]`;
   - the band against eigsh (`check`), with every returned $\lambda \ge \lambda_{lo}(1 - 100t)$ and $V$ holding exactly the band's columns.
   
   Updated for the junction:
   - `test_capacity_store`: interior $30 + \lceil 1.25 \max(5 - 20, 0)\rceil$ = 30, zero 30 + 0. A new interior case with `n_modes` 30 gives 30 + 13 = 43, and one without `n_modes` gives 30 + 200 = 230.
   - `test_error_result_capacity` (:915): $N_s$ = 50.
   - `test_filter_capacity_sorted` (:840): the count now runs on the sorted λ, cut 4 from the 3 → 5 gap.
   
   The no-gap case uses `harness` with λ spaced 0.5% apart and $t = 10^{-3}$: `info["junction"]` is None and the band runs from its start. A cut ≤ lo case (locked modes all below the band): no leaf, si_band from lo with `known`.
7. **Memory:**
   - `test_memory_formulas` (`test_shift_invert.py:588`) gains the exact `n_z` terms: $8n\lceil (n_z + k)/c\rceil c$ device, $16 n_z^2$ host, the MINRES block, and `device_exact` / `host_exact`;
   - `_slice_plan` on crafted fit functions: full deflation, a bisected $d$ at the floor $C_f$, a given capacity, $D$ = 0 equal to today's $C$;
   - a process peak (slow, `peak`, macOS, PEAK_SETUP style in a subprocess): one si_slice step on voxel $48^3$ / GMG with `Z` = 800 random masked columns (353k DOF, $Z$ about 2.3 GB, any $Z$ is exact), with the deflation terms ≥ 50% of the estimate and the peak within 20% of `si_memory`'s total.
8. **Levers:**
   - `test_choose`'s rule-2 cases now expect the maximum-reach pick, with a tie case;
   - `test_levers_unmargined`'s 1.2e8 sub-case (:1422; today rule 2 / "memory") expects the maximum-reach state;
   - the 18e "Rule 2 can forgo levers" case (voxel $10^3$, `n_modes` 12) now gets {precond, basis, low} and returns all 12, checked against eigsh;
   - otf back to stored (fake builds, `_choose` monkeypatched to otf, a capacity stop): a second operator build without `mode` when the stored plan qualifies, with the otf operator dead (weakref) before it; none when $d_\text{stored} < 0.95\,d_\text{otf}$;
   - `slice_options={"max_deflate": 0}` reaches si_band as 0, and a non-int or a negative value raises.
9. **`stagnation_risk`** (`test_stagnation_risk`, :2308): `_slice_plan` monkeypatched to $d < D$ (risk True, the warning names "deflation limited to") and $d = D$ (False) at depth 1000; 999 stays False.
10. **Deep band** (slow, about 3 min target, measured in implementation): hex20 $10^3$ clamped, `chunk=40` pinned, a budget capacity-stopping LOBPCG near depth 250, and the band from mode 260 to mode 300 against eigsh at the band's centre ($k$ = 60):
    - every band mode found;
    - each slice's `deflated` equals $k$ + the stored modes below its σ, capped by $d$ from `_slice_plan` at the run's budget (asserted to be $D$ + found here);
    - one first MINRES solve at the first slice's σ, deflated against plain (`linalg.minres` directly on the run's store), with the iteration ratio asserted ≤ 0.5 and printed.
    
    A second run with `max_deflate` = 100 (partial deflation) also finds every band mode.
11. `test_options.py:553–560`: `band_set`, `slice_set` and the equality gain `known`, `max_deflate`, `Z` and the `max_deflate` slice option.
12. All other band, strategy and shift-invert tests pass unchanged, except the pinned cases above (each named in its commit).

## Benchmark (18f)

`benchmarks/deep_band_study.py` (results `benchmarks/results/deep_band_study.json`, ≤ 1 h, per-step progress logged to a file):
- hex20 $16^3$, `chunk=40`, a budget that capacity-stops LOBPCG near depth 950, and the band of modes 980–1020 end to end, recording:
  - per slice: MINRES iterations per step, $n_Z$, $d$ against $D$, σ moves, time;
  - frequencies against eigsh at the band's centre ($k$ = 80, `maxiter` set).
- At depths 500 and 1000: one first MINRES solve at the band slice's σ with and without deflation ($Z$ from LOBPCG's own modes, maxit 20000 for the undeflated; the depth-1000 modes cached), and one with half of $Z$ (the memory-bound case), plus the deflation's per-iteration overhead.
- If the end-to-end band overruns the hour, it runs at depth 500 instead, logged as a deviation.

## Docs (18f)

- `docs/eigensolvers.md`:
  - Interior bands and strategy:
    - `:141`: the capacity stop → the junction;
    - `:145–151` "Why memory-driven": the slice per-mode row and the crossover claim marked undeflated, with the study's figure;
    - `:166`: rule 2 → the maximum reach, and the residual risks;
    - `:183`: the store $N_s$;
    - `:187`: `from_zero` and `mu1_ub` after a junction;
    - `:197` Errors: forced "lobpcg" for an unreached known depth, the `slice_options` keys (`max_deflate`), the discarded LOBPCG run → kept;
    - `:199`: `info["junction"]`.
  - Options table: `slice_options` `max_deflate`.
  - Shift-invert slices (`:716–753`): the MINRES deflation ($Z$, defined against the Lanczos window $W$), `info["deflated"]`, the exactness argument with the two preconditioned operators written out.
  - Slicing a band (`:765`): $Z$ per item, `known`, `max_deflate`, the in-place assembly.
  - Slice memory model (`:802`): `n_z`, `_slice_plan`.
  - Measured MINRES cost: `:824–828` ("Depth sets the cost", the $\alpha D(1 + D/D_0)$ model) marked undeflated, before PR 18f; `:835` "Depth 1000 is marginal" → the deflated study figures.
  - Test sections: `:997` (a capacity stop handing the whole band to si_band → the junction), `:1002` (the stagnation warning now tied to $d < D$), `:1003` (rule 2 at $n_\text{modes}$ 150 → maximum reach), plus the new tests.
- `README.md`:
  - the status sentence: "split into 18a–18h", 18f's status, the order;
  - the interior-band paragraph (`:287`): the junction and the deflation, with the "33× … 1260×" ratio marked as undeflated (or replaced by the study's).
- `docs/ROADMAP.md`:
  - the status line, and `:626` ("split into eight parts … 18h last");
  - the 18f section rewritten to its contents, with the acceptance as run;
  - a new 18g (slice efficiency), and the old 18g → 18h;
  - the dependency rows and the order;
  - `:95` (M3b: the depth-1000 mid-gap solve, now with the deflated figure) and `:808` (the PR 15d GPU check "MINRES near depth 1000" → deflated, with the cap);
  - deferred GPU checks:
    - the deflation cap $d$ at 10M DOF, where $Z$ costs 84 MB per column and 1000 modes need 84 GB, so $d < D$, partial-deflation MINRES near depth 1000;
    - fp32 $Z$;
    - the deflation overhead at 10M.
- `docs/open_items.md`:
  - "Shift-invert cost near mode 1000": resolved on the Mac (PR 18f), with the 10M part kept open (the cap $d$), and `:175` (10M criteria) annotated;
  - "σ-move retries": by the study's result;
  - "Slice efficiency" → 18g; `:375` (Resolved "Late capacity errors…": "→ PR 18f" → 18g);
  - annotated in place with "PR 18f: …":
    - `:156` (the discarded LOBPCG run);
    - `:172` (the old $N_s$);
    - `:183` (locked modes discarded);
    - `:313` (the 18e levers decision: rule 2, otf carried into the slices);
    - `:321` ("Rule 2 can forgo levers": resolved by the maximum-reach rule);
    - `:312` and `:367` (the order);
  - a "PR 18f decisions" section (section 7 below).
- Docstrings: `si_slice` (`Z`), `si_band` (`known`, `max_deflate`, the assembly), `solve_modes` (the junction, `max_deflate`, `info["junction"]`), `_choose`, `si_memory` (`n_z` against `n_deflate`), `_slice_capacity`, `_slice_plan`, `_cut`.
- Memory:
  - MEMORY.md: the autonomy line, the scale line's order with 18h, and the CalculiX line → 18h;
  - `feedback-autonomy.md`: a PR 18f entry (plan and code review rounds, merge);
  - `project-calculix-benchmark.md`: the description and body → 18h, with the renumbering;
  - `project-scale-questions.md`: the order.

## 7. Risks and decisions without user input (logged in `docs/open_items.md`)

- The split (18g slice efficiency, 18h CalculiX) and the order.
- The A-orthogonal deflation in its Z-only form: exact for any $Z$ and $H$, at half the memory of the $AZ$ form, for one $A$ apply per iteration (+16% at 1M DOF).
- Deflation over capacity above the need-based floor $C_f$, and the nearest modes first when memory limits $d$.
- The cut rule, the known leaf without a rerun when it flags (a regression against re-slicing $(\text{lo}, \text{cut})$), and no junction without a qualifying gap.
- Maximum reach for unreached known depths, otf back to stored at $d \ge 0.95\,d_\text{otf}$, the LOBPCG rerun closed, and no levers for unknown depths with a fit.
- The σ-move retry policy decided by the study.
- Residual risks: $f_1 > 0$ bands without LOBPCG deflate only their own modes; at 10M DOF the deflation is memory-capped (a deferred GPU check).
- The roadmap's acceptance "depth-500 and depth-1000 bands against eigsh": the depth-1000 band runs end to end in the study, the test at about depth 260 (the suite's time), and depth 500 as single solves.
