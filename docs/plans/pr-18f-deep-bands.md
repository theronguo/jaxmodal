# PR 18 plan: performance pass, part f: deep bands

Branch `pr-18f-deep-bands` from `main` (0dc91a5, PR 18e merged). Roadmap: "PR 18f: Deep bands".

## Revisions

- Rev 1: first draft.

## Split (decision without user input)

The roadmap's 18f holds four groups. Two of them, the junction and the deflated MINRES, are what the depth-1000 remedy needs; the levers after a capacity stop depend on them. The fourth group, the slice-efficiency items of PR 15c/15d, is independent of depth and moves to its own PR:

| PR | branch | contents |
|---|---|---|
| 18f | `pr-18f-deep-bands` | this plan: the LOBPCG/slice junction, MINRES deflated by the known modes below σ, its memory, levers after a capacity stop, the σ-move retry policy (closed), the depth study |
| 18g | `pr-18g-slices` | new: the slice-efficiency items (verification slices, σ toward the unconverged neighbour, the top-remainder neighbour); its own plan, measured first, so items may close without code |
| 18h | `pr-18h-calculix` | the old 18g (CalculiX comparison), unchanged |

Dependencies: 18g ← 18f; 18h ← 18a–18g; 19 ← 17, 18a–18g. Order: 18f → 18g → 18h → 20 → 21 → 22 → 23a → 19 → 23b (all of 18 runs on the Mac CPU).

## Probe (planning, scratchpad `p18f/adefl_probe.py`, not committed)

Hex20 $16^3$ (56k DOF, clamped `xmin`), AMG at σ, 6 columns, τ = $10^{-8}$, σ mid-gap above depth $d$, the reference modes of PR 18e's probe (λ-scaled residuals up to $5.6\cdot10^{-7}$, LOBPCG accuracy at tol $10^{-6}$). The **A-orthogonal deflation** (below) with $Z$ = the known modes below σ, except the $u$ nearest below σ, which stay unknown (a slice's own modes between $a$ and σ):

| depth | none (18e probe) | $u$ = 0 | $u$ = 10 | $u$ = 30 | $u$ = 60 |
|---|---|---|---|---|---|
| 197 | 1500 its, 61.9 s | 140, 6.7 s | 240, 11.0 s | 400, 17.9 s | 560, 24.5 s |
| 503 | 4960 its, 203.7 s | 220, 11.0 s | — | 650, 31.2 s | 980, 46.1 s |

Every solve's true residual $\|b - Ax\|/\|b\|$ is at most $10^{-8}$ (7.9e-9–9.9e-9), although the deflated modes carry residuals of $5.6\cdot10^{-7}$: the A-orthogonal form is exact for any $Z$ (below), so the 18e caveat about the residual coupling $Z^\top A\Pi$ does not arise. Forming $AZ$ took 0.3–0.4 s. Per iteration the deflation adds 22% at $d$ = 503 (41 → 50 ms). **Partial deflation** at depth 503, $u$ = 30: the nearest 100 below the unknown ones give 3250 iterations, the nearest 250 give 2090, the lowest 250 give 2460 and all 473 give 650. So the gain comes from deflating essentially all of them; when memory limits $d$, the nearest ones are the better choice.

## Goal (18f)

Bands past LOBPCG's basis cap continue from the locked modes instead of from 0. Every slice's MINRES deflates the known modes below σ, so MINRES iterations stop growing with depth (the 1000-mode remedy). The levers stop discarding LOBPCG reach. Every pair still passes the same acceptance tests, so the results are unchanged up to the inner-solve accuracy. The slices that the band runs differ: they start at the junction cut.

## Design (18f)

### 1. Deflated MINRES (`eigen/shift_invert.py`)

$A = K - \sigma M$ (masked, `ap_operator(op, free, -σ)`), $Z$ the device block of deflation vectors (the rigid modes $R$ and known modes below σ), $AZ = A Z$, $E = Z^\top A Z$ (symmetric, $d \times d$), $G = E^+$ (symmetric). Then

$$P_D = I - AZ\,G\,Z^\top,\qquad P_D A = A - AZ\,G\,(AZ)^\top \ (\text{symmetric}),$$

MINRES (`linalg.minres`, unchanged, with si_slice's SPD preconditioner) solves $P_D A\,\tilde x = P_D b$, and the solution is $x = Z G Z^\top b + P_D^\top \tilde x$. For **any** symmetric $G$, $A P_D^\top = P_D A$, so $b - Ax = P_D b - P_D A\tilde x$: the residual MINRES checks is the true residual, exactly in exact arithmetic. The exactness holds for any $Z$ and any $G$: neither eigenvector accuracy nor $G = E^{-1}$ is needed. $G = E^{-1}$ makes $P_D$ the projector that removes the deflated directions (the convergence gain). With exact eigenvectors $P_D A = \Pi^\top A\Pi$, $\Pi = I - ZZ^\top M$ (18e's probe operator), and plain $P$ gives the same spectrum as $\Pi P\Pi^\top$ ($XY$ and $YX$ share their nonzero eigenvalues).

- **$G$:** `eigh` of $(E + E^\top)/2$ on the host, eigenvalues with $|e_i| < 10^{-10}\max|e|$ dropped (a pseudo-inverse). Duplicate store columns (two runs' copies of one mode, merged only at the end) then make $Z$ rank-deficient without harm.
- **Tolerance:** MINRES runs at a per-column tolerance $\tau\,\|b\|/\|P_D b\|$ (`minres` broadcasts an array `tol`), and its relative residuals are scaled back by $\|P_D b\|/\|b\|$, so `tau`, $\tau_\text{eff}$, the stagnation test and the σ-move threshold keep their meaning.
- **si_slice:** a new argument `Z` (host columns: a 2-D array or a sequence of 1-D arrays; never returned). si_slice uploads `Z` into a zero device block of $\lceil (k + d)/c \rceil c$ columns, $R$ first, in chunks of $c$ columns with `_insert` (no full host or device copy). $AZ$ is computed in chunks of $b$ columns (inside the modelled MINRES workspace), and $E$ and $G$ on the host. `_deflated(A, Z, AZ, G)` is the operator: one `Partial`, $y = A v$, then $y - AZ\,(G\,(Z^\top y))$. On a σ move, $AZ$, $E$ and $G$ are recomputed (σ changes; $Z$ is kept). `info["deflated"]` reports $k + d$. Without `Z` (or with an empty one) nothing is deflated and it is 0: $R$ alone would gain nothing, since $R^\top b = 0$ for $b = My$, $y \perp_M R$. `info["time"]["deflate"]` holds the upload, $AZ$ and $G$.
- **si_checked** passes `Z` through `slice_options` unchanged. Its restarts keep the same `Z` (same σ). Each run re-forms $AZ$ (d / b applies; restarts are rare).

### 2. Slicing with deflation and known pairs (`si_band`)

- **Z per work item:** the stored modes with $\lambda < \sigma$ (σ the item's midpoint), the nearest `max_deflate` of them (new argument; None = all, 0 = none), passed as a list of store columns (views, no copy), plus $R$ inside si_slice. Window modes below σ are included: they make $P_D A$ definite near σ on the deflated side, and the store holds them anyway. `"Z"` joins the slice options si_band sets itself (the reject set).
- **`known`** (new argument, None or a dict `lam`, `V`, `residuals`, `thresholds`, `cut`, `count`): pairs to start from. They are LOBPCG's locked modes, ascending, with `V` the host F-order block whose columns the store keeps as views. They are stored as run −1, so `_merge` treats them as a run of their own. `cut` is where the slices take over. The work starts at $\max(\text{lo}, \text{cut})$, with the cut in the first item's done edges, and interior `edges` at or below the cut are dropped. When cut > lo, the leaf $((\text{lo}, \text{cut}), \text{count})$ joins `leaves` and `slices` gets `{interval, kind "known", leaf True, count}`. Its count is the caller's, run before si_band (below). When cut ≤ lo, the known pairs serve only as deflation vectors and lower neighbours. `complete_to` and the err paths are unchanged: the stored pairs include the known ones, and the filter in solve_modes drops those below $f_1$.
- The known block is copied into `V_f` at assembly. It is freed only once all its columns are copied, so for one moment the assembly holds the store plus the known block (budgeted, section 4).

### 3. The junction (`solve_modes`, `eigen/api.py`)

After a capacity stop (`strategy="auto"`):

1. **Sort and keep:** the locked modes are sorted as on the "fits" path (`_permute(V, order)` also when stopped). `V.resize((n, D))` drops the unused columns ($D$ = locked count; `V` has no other reference). `info`'s residuals and thresholds are kept with them. The aggregates are kept until the count is done.
2. **The cut** `_cut(lam, t, m)`: the midpoint of the largest relative gap $(\lambda_{i+1} - \lambda_i)/\lambda_{i+1} \ge 100\max t$ among the top $m + 1$ locked modes ($m$ = LOBPCG's block: a missed mode is most likely near the top, where the next chunk would have returned it). If that span has none, the largest gap over all locked modes that is at least $100\max t$. If there is none, or $D < 2$, there is no junction, and the band runs as today but deflated (`info["junction"]` None).
3. **The count:** `_verify` on the locked modes, as on the "fits" path: the count over $[0, \lambda_c]$, with `n_band` the modes below the cut and the first one above it as `lam_next`; LOBPCG's `t_iteration`, block and solve time; `reuse` and `probe_budget` as there. Its record is the known leaf's count: verify True / False / None as on the "fits" path. A flagged count (≥ 0.5) is not rerun: rerunning a span of $D$ modes would deflate more than any slice's capacity, so the leaf stays flagged, as the "fits" path reports it.
4. **otf back to stored** (section 5), then si_band with `known`. The deflation cap comes from section 4. The store is $N_s = D + \lceil 1.25\,\max(n_\text{modes} - L, 0)\rceil$ with `n_modes`, else $D + 200$ ($L$ the in-band locked count, as today).
5. **Info:** `info["junction"] = {"cut", "cut_hz", "known": D, "count"}`. `stagnation_risk` becomes True only when the depth is ≥ `STAGNATION_DEPTH` and the deflation cap is below it. The capacity-stop warning says that "the band continues from the cut with shift-invert slices", and adds "(MINRES may stagnate: deflation limited to $d$ of $D$ modes)" in that case.

The "memory" path (no chunk fits) and forced "shift_invert" have no junction. Their slices from 0 still deflate what earlier slices found (section 2). A band with $f_1 > 0$ that starts at $\lambda_{lo}$ deflates only the band's own modes found below each σ (a logged limit: the modes below $f_1$ are not computed for it).

### 4. Memory (`memory.py`)

- **`si_memory(..., n_known=0)`:** device storage gains $2 \cdot 8n \lceil (n_\text{known} + k)/c\rceil c$ when $n_\text{known} > 0$ ($Z$ and $AZ$, with $R$'s $k$ columns copied in), host gains $16\,n_\text{known}^2$ ($E$ and $G$), and the MINRES phase gains one $n \times b$ block (the deflation's temporaries). All of these are exact or modelled terms of the fit test, as their kind.
- **`_slice_plan(op_memory, pc64, n, *, n_store, n_exp, block, n_rigid, budget, host_budget, capacity=None)` → (C, d):** with $C_0$ = `_slice_capacity` at $n_\text{known}$ = 0 (None: the slice-floor error, as today) and the floor $C_\min = \min(\texttt{SLICE\_MIN}, C_0)$, take $C(n_\text{exp})$ if it is at least $C_\min$, else the largest $d \le n_\text{exp}$ with $C(d) \ge C_\min$, bisected. Deflation therefore wins over capacity above $C_\min$: per the probe it cuts MINRES 4–8× at depth, while a capacity above 96 columns only lowers the slices' fixed cost (a slice converges after about 50–85 columns). With a given `capacity`, $d$ is the largest that fits beside it. $n_\text{exp} = N_s$, all stored modes. `_slice_capacity` gains `n_known` (passed to `si_memory`). Its need cap is unchanged.
- **The known block at assembly:** the host term adds $8nD$ through `n_store` (the store then holds $D$ + new, and the assembly copy is the existing "1× + a block", the block now $D$ columns). Section 7 checks it against a process peak.

### 5. Levers after a capacity stop (`eigen/api.py`)

- **Rule 2 becomes the reach rule:** a known depth that no state reaches (margined or not) now takes rule 3's choice, the cheapest state within 5% of the best used reach (`_choose` loses `if needs: return None`). Rule 2 existed because a predicted capacity stop discarded its LOBPCG work. The junction keeps that work, so more reach means fewer slice modes. LOBPCG costs far less per mode than slices (about 36 column applies per mode in a 40-mode chunk, against thousands of MINRES column applies per slice mode at depth), so even otf's 3× pays for itself. This resolves the open item "Rule 2 can forgo levers".
- **otf back to stored:** when the levers set otf (`"operator_options" in choice`), after the capacity stop and before the count, `_slice_plan` runs with the stored model (`op_opts` without `mode`). If it gives at least the otf plan's $(C, d)$, the operator is rebuilt stored (the otf one freed first; the aggregates stay valid, since they depend on the mesh graph only) and `info["slice_operator"]` = "stored". Otherwise otf is kept (`"otf"`; on paths without the lever, the given mode).
- **A LOBPCG rerun with levers: closed.** The junction keeps the locked modes, and the levers already raise the reach up front (rule 1, rule 3, and now the reach rule). A rerun would repay the depth already reached. Unknown depths with a fitting chunk still get no levers (logged: unknown cost against an unknown depth).

### 6. Options and closures

- `slice_options={"deflate": False}` (SLICE_OPTIONS, a bool, default True) sets `max_deflate=0`. It is for comparisons and fallbacks.
- **The σ-move retry policy: closed.** Its cost came from depth-driven stagnation, which the deflation removes. The study (section 8) records the σ moves at depths 500 and 1000. If one is depth-driven there, the item goes to 18g.
- **Also closed:** $Z$ and $AZ$ in fp32 (the exactness needs $AZ = AZ$ exactly; the halved memory is a GPU-side question, a deferred GPU check); deflating modes above σ (positive and far from σ after preconditioning, no measured gain); a slice-level band from 0 for $f_1 > 0$ bands without LOBPCG (it computes every mode below $f_1$ by slices, more than the depth it saves).

## Tests (18f)

References: SciPy `eigsh` and dense NumPy (`tests/test_shift_invert.py`, `tests/test_strategy.py` helpers `prob`, `check`, `run`, `fake_builds`, `capacity_stop`, `stub_band`).

1. **`_deflated` (dense references, "tiny" and hex20 $4^3$ clamped):**
   - $x$ against `np.linalg.solve(K - σM, b)` to $10^{-8}$ relative, with σ above 10 modes and $Z$ the 10 eigsh modes;
   - with $G$ a random symmetric matrix instead of $E^+$, the residual identity $b - Ax = P_D b - P_D A\tilde x$ to $10^{-12}$ relative;
   - $Z$ with a duplicated column: the pseudo-inverse drops one direction and $x$ is unchanged;
   - the tolerance scaling: the returned residuals are relative to $\|b\|$ (equal to `norm(b - A x) / norm(b)` within the check interval).
2. **si_slice with `Z`:** on a hex20 slice above 30 modes, the pairs agree with the `Z=None` run and with eigsh (si_slice-level checks), with `info["deflated"]` = 30 and the first-step MINRES iterations below the `Z=None` run's. A monkeypatched first solve that stagnates makes σ move, and $AZ$ is recomputed (a spy on the $AZ$ builder sees two σ values). A 1-D column list and a 2-D array give identical results.
3. **si_band's Z selection (fakes, a recording si_checked):**
   - $Z$ = the stored modes below each item's midpoint;
   - `max_deflate` = 3 gives the 3 nearest;
   - 0 gives none, and $R$ is not counted;
   - `Z` in `slice_options` raises.
4. **`known` in si_band (fakes):**
   - the work starts at the cut, which is in the done edges;
   - edges at or below the cut are dropped;
   - the known leaf is in `leaves` and `slices`;
   - cut ≤ lo gives no leaf but a stored pair;
   - the known pairs come back in `lam`, and `_merge` sees run −1 against a slice that re-found one (`test_band_merge`'s setup);
   - `err.result` after the first slice raises holds the known pairs, with `complete_to` = the cut.
5. **`_cut` (crafted λ, t, m):**
   - the largest gap in the top $m + 1$;
   - the fallback to all locked modes;
   - None when no gap reaches $100\max t$;
   - $D < 2$.
6. **Junction end to end** (`test_capacity` updated: "short", chunk 5, 35 locked inside the band 25–45):
   - si_band receives `known` with 35 pairs and the cut from `_cut`, and the first item starts at the cut;
   - the LOBPCG `V` is the store's base (no longer freed) and is resized to 35 columns;
   - the count leaf passes;
   - `info["junction"]`;
   - the band against eigsh (`check`).
   
   `test_capacity_store` gets the new $N_s$ ($D + \lceil 1.25 \max(n - L, 0)\rceil$: 30 + 0 interior, 30 + 0 zero), and si_band's interval stays the band's. The `harness` capacity case (`test_strategy.py:840`) gets the junction fields. Without a gap (`capacity_stop` with a crafted cluster), `info["junction"]` is None and the band runs from the band's start.
7. **Memory:**
   - `si_memory`'s `n_known` terms (exact: $2 \cdot 8n\lceil d/c\rceil c$ device, $16d^2$ host);
   - `_slice_plan` on crafted fit functions: full deflation, bisected $d$ at the floor, a given capacity;
   - the existing `si_memory` checks against `memory_analysis` (`test_si_memory*`) gain one deflated case (`n_known` 64), with the device step within the existing tolerance;
   - one process-peak case (slow, `peak`, macOS): a junction band on hex20 $12^3$ at a budget forcing the stop, the peak within 20% of `si_memory`'s total at the chosen (C, d) plus the store.
8. **Levers:**
   - `test_choose`'s rule-2 cases now expect rule 3's pick;
   - `test_levers_known`'s rule-2 case gets the reach rule's state;
   - the "Rule 2 can forgo levers" case (voxel $10^3$, `n_modes` 12) now gets {precond, basis, low} and LOBPCG returns all 12;
   - otf back to stored (fake builds, `_choose` monkeypatched to otf, a capacity stop): a second operator build in stored mode when its plan fits, and none when the stored plan misses (`info["slice_operator"]`).
9. **Deep band** (slow, about 3 min target, measured in implementation): hex20 $10^3$ clamped, LOBPCG capacity-stopped near depth 250 by the budget, `band` from mode 260 to mode 300 against eigsh at the band's centre ($k$ = 60):
   - every band mode found;
   - `info["slices"]` with `deflated` ≥ 250;
   - the first slice's first-step MINRES iterations recorded;
   - a `deflate: False` run of the first slice only (`max_slices=1`, `err.result`) records more iterations (asserted, with the ratio printed).
10. All other band, strategy and shift-invert tests pass unchanged, except the pinned cases above (each named in its commit).

## Benchmark (18f)

`benchmarks/deep_band_study.py` (results `benchmarks/results/deep_band_study.json`, ≤ 1 h, per-step progress logged):
- hex20 $16^3$, a budget that capacity-stops LOBPCG near depth 450, and the band of modes 480–520 end to end: per-slice MINRES iterations, `deflated`, σ moves, time, and frequencies against eigsh at the band's centre;
- at depths 500 and 1000: one first MINRES solve at the band slice's σ with and without deflation (Z from LOBPCG's own modes, the probe's measurement now at LOBPCG accuracy);
- the deflation's per-iteration overhead.

## Docs (18f)

- `docs/eigensolvers.md`:
  - Interior bands and strategy (`:141`: the capacity stop → the junction; `:166`: rule 2 → the reach rule, the residual risks updated; `:183`: the store $N_s$; `:197` Errors: "discarded LOBPCG run" → kept; `:199`: `info["junction"]`, `slice_operator`, `deflate`);
  - the options table (`slice_options` `deflate`);
  - Shift-invert slices (`:716–750`: the deflated MINRES, `Z`, `info["deflated"]`, `time["deflate"]`, the exactness argument);
  - Slicing a band (`:765`: Z per item, `known`, `max_deflate`, the known leaf);
  - Slice memory model (`:802`: `n_known`, `_slice_plan`);
  - Measured MINRES cost (`:835` "Depth 1000 is marginal" → the deflated figures from the study, with the 18e/18f probes);
  - the test sections.
- `README.md`: the status and order sentence, and the interior-band paragraph (junction, deflation).
- `docs/ROADMAP.md`:
  - status line;
  - the 18f section rewritten to its contents;
  - new 18g (slice efficiency); old 18g → 18h;
  - the dependency rows and the order;
  - deferred GPU checks: fp32 $Z$ / $AZ$, the deflation overhead at 10M DOF.
- `docs/open_items.md`:
  - "Shift-invert cost near mode 1000", "σ-move retries" and "Rule 2 can forgo levers" → Resolved, with outcomes;
  - "Slice efficiency" → 18g;
  - a "PR 18f decisions" section;
  - the 18e split entry's order updated.
- Docstrings: `si_slice` (`Z`), `si_band` (`known`, `max_deflate`), `solve_modes` (junction, `deflate`), `_choose`, `si_memory`, `_slice_plan`.
- Memory: MEMORY.md (autonomy line, order, CalculiX → 18h), `project-calculix-benchmark.md`, `project-scale-questions.md`.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- The split (18g slice efficiency, 18h CalculiX) and the order.
- The A-orthogonal deflation over the M-projected one. It is exact for any $Z$ and needs only $Z$ and $AZ$: the same memory, with no coupling term.
- Deflation over capacity above `SLICE_MIN`, and the nearest modes first when memory limits $d$.
- The cut rule and the known leaf (no rerun when flagged).
- Rule 2 → the reach rule, otf back to stored when the stored plan is no worse, the LOBPCG rerun closed, and no levers for unknown depths with a fit.
- The σ-move retry policy closed (re-opened to 18g if the study shows depth-driven moves).
- The residual risk: $f_1 > 0$ bands without LOBPCG deflate only their own modes.
