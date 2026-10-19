# PR 18 plan: performance pass, part b: solid-shell AMG in `solve_modes`

Branch `pr-18b-shell-solve` (this plan), from `main` after PR 18a. Roadmap: PR 18b
("`"auto"` → AMG on `ssh8` meshes, the σ pre-estimate on shells, band-mode reuse, the
`amg_memory` shell family and its calibration"; test: "`solve_modes` with AMG on a solid-shell
mesh matches the PR 8 `eigsh` to PR 9 tolerances"; `docs/ROADMAP.md:184`: the shell estimator
"validated against measured peaks where the estimate is ≥ 2 GB"), the 18a plan's "Carried
into 18b's plan" list (`docs/plans/pr-18a-shell-amg.md:199–238`), and the "Thin plates and
slender beams" entry of `docs/open_items.md`.

## Revisions

- **Revision 1:** first draft, from the sizing study (scripts in the session scratchpad `s18b/`).
- **Revision 2 (plan review round 1):** stagnation diagnosed as drift in the implicit
  products; an explicit-product regime; a rigid-mode leak on free bodies found (a bug on
  `main`) and fixed by re-deflation in the recompute branch; the polish as a one-step
  finisher; PR 9 tolerance tests; per-component memory tests; docs lists.
- **Revision 3 (plan review round 2):**
  - **The regime switches the chunk to the low-memory step** instead of recomputing in the
    9-block step. W's K and M products drift too, and the 9-block step's `lax.cond` with a
    re-deflation added 1–3 blocks to its compiled peak (`test_step_memory` would fail). The low
    step is explicit in every product, its peak is below the 9-block one, and it does not
    leak. Free bodies now converge without a polish. The re-deflation in the recompute branch
    is dropped (a single deflation of X and P at the switch).
  - **Polish:** X and λ are kept until `_begin` succeeds (one fallback rule); `polish` returns
    `(Y, iterations)`; a one-step progress rule including the pre-polish ratio; `maxit`
    default 500; A_P built once; the memory term a margin over the step, not a fit.
  - **Failed exits:** an explicit Rayleigh–Ritz (λ ascending, each an upper bound for its
    index; r and t from it), not per-column Rayleigh quotients.
  - **Constants:** host build refitted as an upper envelope (β 150 MB, γ 1.4 / 2.5 kB);
    $\alpha_T$ = 880 B; tightness tests only where the constants are tight.
  - **Tests:** 7 accuracy cases; $\sin\theta$ bounded by the solver's and the reference's
    residuals (the "reference sharper" assertion fails after a polish); r_M sparse; a 791k
    `peak` test for the ≥ 2 GB rule; the existing tests that change, completed.
  - **Docs:** file:line list corrected and completed; carried items completed.
- **Revision 4 (plan review round 3; correctness and docs satisfied):**
  - **One polish step**, not a loop (every measured case accepts in one): its trigger and
    $w_0$ from explicit residuals of the stalled X, the better of the two kept (superseded in rev 6: accept-or-revert); `_iterate`
    returns only $(X, \lambda)$, so no state rebuild.
  - **Switch details:** the progress window resets at the switch; the low step runs with
    `c = k` (`STEP_FIT["low"]` exact); `_to_low` donates; one extra compile.
  - **Failed-exit Rayleigh–Ritz** not on `stop` exits (the σ estimates: it changed σ and an
    AMG solve's iterations, 109 → 159).
  - **Polish memory:** an `optimization_barrier` in `_pcg`'s loop stops XLA from hoisting the
    V-cycle's unpacked inverses out of the loop (162 → 75.6 blocks at flat $64^2$, PCG 2.3×
    faster there); then a fitted `pc["polish"]` per family instead of "+3 blocks", so hex and
    `low_memory` estimates do not inflate.
  - **Constants:** $\alpha_T$ split (880 B shell DOFs, 650 B hex-only DOFs), $\alpha_h$ =
    2.6 kB, $\gamma_h$ = 3.3 kB (block $48^3$ / $64^3$ measured), block $64^3$ calibrated and
    peak-tested.
  - **Tests:** Rayleigh-quotient checks instead of Ky Fan; $r_{M,\text{ref}}$ bounded by the
    `check_lam` floor; a live-bytes check at the polish; `test_compiles`, `test_mac_acceptance`
    listed; overlaps dropped; `switched is None` in the AMG / GMG / p-MG solve helpers.
- **Revision 5 (plan review round 4; simplicity, tests and docs satisfied):**
  - **`_project` clamp-safe:** in band mode the capacity is a multiple of the 9-block `c`,
    not of `k`, so the switched low step's `c = k` chunks could read past it
    (`dynamic_slice` clamps, and columns were subtracted twice); the chunk start is clamped
    and columns are masked to `[i c, count)`.
  - **The barrier at all three sites:** `_pcg`, `_minres` and `_recurrence`'s body (shell
    MINRES 456 → 187 MB, the recurrence 421 → 158 MB at flat $64^2$); hex counts unchanged.
  - **`_final_rr` = `_start` + `_rr_start`** (deflated, the polish's own path), with the
    acceptance check right after it; the polish triggers on any $r_i > t_i$ up to $j$.
  - **`_precond_sigma`:** gets the switch and the final Rayleigh–Ritz (no `stop`; its
    upper-bound docstring now holds), stated and tested.
  - **Smaller:** the polish memory term only for the shell family (hex: below the modelled
    step, tested);
    `info["polishes"]` dropped; the polish term counts the output block; `X = X0 = None`
    after `_begin`; `t_iteration` over 9-block iterations only; the 791k build and solve
    peaks from one subprocess; tests reworked (exact dense inverse-iteration fake, a degrading
    fake, live bytes relative to a baseline, one parametrized `switched is None` test).
- **Revision 6 (plan review round 5; correctness, simplicity, docs, performance satisfied):**
  - **Tests that failed as specified:** the dense polish tests start from exact eigenvectors
    plus high-frequency noise below tol (one inverse-iteration step does not remove
    low-frequency noise: r/t stayed about $10^3$ with $10^{-3}$ noise; measured 7.8 → 0.002
    with the new start); `_precond_sigma`'s bound tested up to the `check_lam` floor (a
    converged round keeps the step's λ, 1.8e-6 below `eigsh` at $L/t$ = 1000).
  - **Simpler:** accept-or-revert instead of keeping the better X; one info key `polish`
    (PCG iterations or None, as `switched`); `EXPLICIT` renamed `SWITCH`; no switch when a
    `stop` callback is given (`sigma_estimate` bitwise unchanged); the projector masks the
    coefficients, not a slice of B.
  - **Stated:** the flags of a `STALL` / `maxit` exit accepted by the final Rayleigh–Ritz;
    the Ritz upper bound exact for the first chunk, up to the locked basis' accuracy later;
    the polish fit includes the web, block and stack ×10 meshes (polish above the step
    there by up to 13.7 blocks at web $64^2$, $m$ = 8).
  - **Tests and docs lists completed** (the existing tests a switch reaches, measured).
- **Revision 7 (plan review round 6; correctness, simplicity, tests, docs satisfied):**
  - **Memory form:** the shell step, step_low, polish and count are $8n(am + c) + K_0$, with
    $K_0$ independent of $m$ (the chunk-bounded gathers: about 120 MB at flat $64^2$ for
    $m$ = 8 and 25 alike); the hex family's $8mK$ under-estimated 25k DOF by 19–35 blocks at
    $m$ = 8 when fitted from 100k. Fitted on 25k–396k (count 25k–200k); tightness gated in rev 10.
  - **`stop` given:** no switch and no `_final_rr` on any exit (`_precond_sigma`'s fallback
    `sigma_estimate(maxit=10)` ends by `maxit`; with the RR its σ moved).
  - **Implementation notes:** `_explicit` keeps both branches, keyed on the current step kind;
    the next chunk's start from `_iterate`'s X before `_reortho`; only X and λ live at the
    polish; the switch check after the convergence check; the Ritz bound stated correctly.
  - **Tests:** three existing tests added (`test_defaults_snapshot`, `test_option_sets_pinned`,
    `test_solve_boundary_one`); the cluster test made discriminating; boundary groups in test
    2; the live-bytes slack 0.5 block.
- **Revision 8 (plan review round 7; correctness, simplicity, docs satisfied):**
  - **Fit range in $m$:** the defaults reach $m$ = 60 (50 modes) and 120 (100 modes), where
    the $m$ = 8 / 15 / 25 envelope under-estimated by up to 2.6 blocks; the step, low step
    and polish are fitted at $m$ = 6 / 8 / 15 / 25 / 60 / 120 and stated valid for
    $6 \le m \le 120$.
  - **Tests that did not discriminate:** the cluster-extension start (high-frequency noise
    on every multiplet column, `want` 1 or 4 on the cube pencil; pre-polish asserts); the
    `_precond_sigma` check made structural (main and the fix differ by 2e-8 there, at the
    Rayleigh-quotient noise level).
  - **Simpler:** one storage constant per hex-only DOF ($n_\text{hex} = 3 n_h$ made two
    collinear).
- **Revision 9 (plan review round 8; correctness, simplicity, tests, docs satisfied):**
  - **Fit procedure:** web $64^2$ at $m \ge 60$ (a size-specific allocation, flat at 37.9
    blocks from $m$ = 60 to 120) left out of the 9-block fit, which it inflated by 5–9 blocks
    on every case from 100k up; covered by a stated small-mesh margin instead; an upper
    tightness gate at large $n$; the benchmark calibrates every fit case at every $m$.
  - **Smaller:** the memory tightness tests merged into one parametrized test per (mesh,
    $m$); the last-iteration switch rule and the Rayleigh-quotient sanity check dropped;
    `t_iteration` tested through its counter; the cluster test's noise size stated
    (absolute $10^{-2}$).
- **Revision 10 (plan review round 9; correctness, simplicity, docs satisfied):**
  - **Gates set from the measured slack:** the large-$n$ upper gate at + 5 blocks (measured
    with the rev 9 fit: 2.3–4.5; the outlier-inflated fit: 6.2–8.1); the per-$m$
    "within 1 block of the binding case" gate dropped (the LP touches about 3 of 6 $m$
    values; the lower bound and the upper gate cover it); the small-mesh margin as an
    absolute ≤ 128 MB (measured 3.9 blocks, about 100 MB).
  - **Smaller tests:** the memory test with a basis only at $m$ = 6 / 25 / 120 (the full grid
    stays in the benchmark JSON); the `_to_low` unit test keeps only its basis half; the
    revert fakes parametrized; `t_iteration` checked finite and > 0 only.
- **Revision 11 (plan review round 10; correctness, simplicity, docs, performance satisfied):**
  - **Fit target fixed:** `lobpcg_memory` adds $A b + f\,\text{ws}$ (`STEP_FIT`) to
    `pc["step"]`, so the shell `pc["step"]` / `pc["step_low"]` / `pc["polish"]` are fitted to
    the measured total minus that term (with the modelled ws), not to the totals. Refit on the
    round 8/9 data: $a$ = 5.67, $c$ = 15.9, $K_0$ ≈ 74 MB; no point needs
    leaving out (web $64^2$ at $m$ = 120: +7.2 blocks; the outlier was the Jacobi step's, which
    measures the same 37.9 blocks), so the exclusion and the 128 MB small-mesh margin are
    dropped; large-$n$ slack 0–2.4 blocks. The gates are checked at the `lobpcg_memory` level
    and set after the shipped refit (procedure below), with an upper gate on the polish too.
  - **Rounding margin:** constants rounded up and $c$ raised by 0.5 (the hex family's +1.5).
  - **Correctness notes:** the recurrence barrier binds a new name (`Q`; `P` is a closure
    there), `_minres`'s names per site; `t_iteration` counts the solve's own step (all low
    iterations with `low_memory=True`); the clamp's band-only cause stated; the revert's
    bitwise claim qualified by the guard error.
  - **Smaller tests:** the band clamp solve folded into test 4's band run (a spy there); the
    σ-bound half of the `_precond_sigma` test and test 4's Rayleigh-quotient check dropped.

## Goal (18b)

- `solve_modes(precond="auto")` uses AMG on meshes whose blocks are all `ssh8`, or `ssh8` and
  `hex8`; `precond="amg"` is accepted there. `ssh8` with `hex20`/`hex27` keeps Jacobi under
  `"auto"`, and `precond="amg"` (also in `solve_modes_memory`) raises up front with `amg()`'s
  message.
- The roadmap test: on solid-shell meshes `solve_modes` with AMG matches the PR 8 reference
  (`reference_modes`, `eigsh` shift-invert) to the PR 9 tolerances at tol $10^{-10}$:
  `check_lam` (λ relative error ≤ $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$ for
  relative gaps ≥ $10^{-2}$, $10\,$tol cluster-wise otherwise) and principal angles per
  eigenvalue group.
- **Thin shells converge.** Today LOBPCG stagnates on solid shells: flat plates from about
  $L/t = 300$ at tol $10^{-6}$ and $L/t \approx 100$ at $10^{-10}$, 3-layer stacks at $10^{-10}$
  (190 / 250 iterations at $L/t$ = 50 / 100), free bodies with leaked rigid modes. 18b adds the low-memory
  switch and a polish stage that reach the floor.
- `amg_memory` covers `ssh8` meshes (storage, build, step, count), so `solve_modes_memory`, band
  mode, the slice and count estimators and the strategy work on shells.
- Band mode reuses the AMG aggregates on shells as on hex meshes (`return_aggregates`, already
  generic).

## Findings (sizing study and review rounds 1–10, CPU fp64, M3 Max)

### `solve_modes` with shell AMG works as is below $L/t \approx 200$

With the dispatch patched (prototype), 20 modes, tol $10^{-6}$, `maxit` 3000 (Jacobi exits:
`STALL`), otherwise defaults:

| case | DOF | AMG: its / time | Jacobi: its / time | λ error vs `eigsh` |
|---|---|---|---|---|
| flat $32^2$, $L/t = 50$, SS | 6.5k | 16 / 3.3 s | 840 / 4.4 s | $2\cdot10^{-11}$ |
| free $24^2$, $L/t = 50$ | 3.8k | 17 / 3.0 s | 1307 / 9.0 s | (rigid modes deflated) |
| T-junction $24^2$, $L/t = 50$, clamped | 5.7k | 16 / 2.6 s | 998 / 5.0 s | $7\cdot10^{-12}$ |
| flat $64^2$, $L/t = 100$ | 25k | 17 / 4.7 s | 2392, not converged / 30 s | $5\cdot10^{-10}$ |
| curved $32^2$, $L/t = 50$ | 6.5k | 19 / 3.3 s | 1145 / 6.0 s | $1\cdot10^{-11}$ |
| stack $32^2\times3$, $L/t = 100$ | 13k | 19 / 3.8 s | 3000, not converged / 27 s | $4\cdot10^{-9}$ |
| web $32^2$ (`ssh8` + `hex8`), $L/t = 50$ | 7.1k | 19 / 3.7 s | 844 / 4.7 s | $6\cdot10^{-12}$ |

The σ pre-estimate (`_precond_sigma`) needs no change: 6–9 iterations with σ = 0 first on
supported shells, 34 on the free plate (σ = 0 refused, `sigma_estimate` first), and it lands
within 0.3% of the Kirchhoff $\lambda_1$ at $L/t = 1000$.

### Why LOBPCG stagnates on thin shells

$\kappa = \lambda_\max/\lambda_1$ grows like $(L/t)^4$ (thickness modes) and does not depend on
$h$: $4.8\cdot10^7$, $7.3\cdot10^8$, $3.7\cdot10^9$, $4.5\cdot10^{11}$ at $L/t$ = 100, 200,
300, 1000. LOBPCG's floor thresholds $t_i$ (`C_FLOOR` term) are $3\cdot10^{-4}$ (mode 1) to
$1.3\cdot10^{-6}$ (mode 20) at $L/t = 1000$; exact eigenvectors have residuals 0.06–0.2× $t_i$.
LOBPCG exits by `STALL` (50 iterations without progress) with worst $r_i/t_i$ up to
$8\cdot10^5$ although the Ritz values are within $3\cdot10^{-5}$.

1. **Drift in the implicit products.** The 9-block step updates $KX$, $MX$, $KP$, $MP$ and
   $KW$, $MW$ (through the orthogonalizations) by linear combinations; $KX$, $MX$, $KP$, $MP$
   are recomputed every `RECOMPUTE` = 20 iterations. At κ ~ $10^8$–$10^{11}$ the cancellation
   error swamps the residual. Worst $r_i/t_i$ on exit, `RECOMPUTE` patched (review round 1):

   | case | today (20) | every iteration (1) | low-memory step (all explicit) |
   |---|---|---|---|
   | $32^2$, $L/t = 1000$, tol $10^{-6}$ | 210 its, $7.9\cdot10^5$ | 134 its, 35 | 261 its, 58 |
   | $32^2$, $L/t = 300$, tol $10^{-6}$ | 190 its, $2.5\cdot10^4$ | 97 its, 6.4 | |
   | $32^2$, $L/t = 100$, tol $10^{-10}$ | 170 its, $1.7\cdot10^5$ | 35 its, converged | |

   Making $KW$, $MW$ explicit too converges the free bodies and $L/t = 300$ without a polish
   (review round 2). The low-memory step forms every product explicitly.
2. **Rigid-mode leak (free bodies, a bug on `main`).** The 9-block step deflates $W$ against
   $R$ once, before its orthogonalize-and-SVQB passes, and never re-deflates $X$ or $P$; with
   the drift, Rayleigh–Ritz pulls rigid components in (Ritz value about 0). Today, tol
   $10^{-10}$: free $24^2$ ($L/t = 50$) and the free cylinder return 6 near-zero "elastic"
   modes, $\|R^\top MV\|$ = 0.96 / 0.90, not converged. The low-memory step does not leak (also
   over 162 stalled iterations at $L/t = 300$: $3.6\cdot10^{-15}$).
3. **The κ-amplified remainder.** Even fully explicit, at $L/t \ge 1000$ the residual stays a
   few times above the floor: the iterates keep a high-frequency component that the λ-scaled
   2-norm residual amplifies by about κ. One block inverse-iteration step
   $Y = A_P(\sigma)^{-1} M X$ (PCG with the V-cycle), deflation, SVQB and Rayleigh–Ritz damps it
   by about $\lambda/\lambda_\max$. A LOBPCG restart after it brings the component back, and
   LOBPCG with the inner PCG as its preconditioner breaks down ($W \approx X$), so the polish
   is pure inverse iteration.

### Switch to the low-memory step, then polish (prototype `s18b2/harness.py`, `SWITCH=1`)

The 9-block step runs as today; after 5 iterations without progress the chunk deflates X and
P once and continues with the low-memory step; on a `STALL` exit with some wanted
$r_i > t_i$ one polish step (PCG tol $10^{-11}$ in this run), then explicit residuals.
20 modes, AMG:

| case | tol | today | switch at it | switch: its, worst $r/t$ | + polish |
|---|---|---|---|---|---|
| $32^2$, $L/t = 50$ | $10^{-10}$ | 24, converged | never | identical | none |
| $64^2$, $L/t = 100$ | $10^{-6}$ | 17, converged | never | identical | none |
| $32^2$, $L/t = 100$ | $10^{-10}$ | 170, stalled | 23 | 30, converged | none |
| $32^2$, $L/t = 300$ | $10^{-6}$ | 190, stalled | 21 | 111, converged | none |
| $32^2$, $L/t = 1000$ | $10^{-6}$ / $10^{-10}$ | 210, stalled | 27 | 109, stalled, 1.78 | 1 step (0.042), 224 PCG its, 9.2 s |
| $128^2$, $L/t = 1000$ | $10^{-6}$ | 130, stalled | 21 | 122, stalled, 0.92 (a column) | 1 step (0.043), 122 PCG its, 54 s (whole solve with build and compile, before the barrier; re-measured in the benchmark) |
| stack $32^2\times3$, $L/t = 100$ | $10^{-10}$ | 250, stalled | 26 | 105, converged | none |
| $32^2$, $L/t = 1000$, 30 modes, chunks of 10 | $10^{-6}$ | | 28 / 24 / 24 | stalled | 1 step per chunk (0.039, 0.038, 0.037) |
| free $24^2$, $L/t = 50$ | $10^{-10}$ | 130, leaked | yes | converged, `check_lam` passes (1.7e-10) | none |
| free cylinder | $10^{-10}$ | 170, leaked | yes | converged, `check_lam` passes (1.7e-8) | none |
| $32^2$, $L/t = 50$, Jacobi | $10^{-6}$ | 840, converged | 667 | 836, converged | (no polish) |

- Converging AMG and GMG solves never switch (review round 2: hex20 boxes clamped and free,
  chunked; voxel GMG clamped and free; also every hex AMG test in `test_amg`, `test_api`,
  `test_pmg`, `test_gmg`). Slow Jacobi solves can switch late (above: harmless, a little
  faster).
- The low step costs 1.4–1.9× per iteration, only after the switch.
- The polished λ error against `eigsh` at $L/t = 1000$ is $4.7\cdot10^{-6}$, inside
  `check_lam`'s floor $10\varepsilon\lambda_\max/\lambda_i$ (`eigsh` is itself floor-limited).
- Before the switch, the same polish needed 4–6 steps at $L/t = 1000$ (166 s at $128^2$).
- Today a stalled exit returns λ below the true value (956.6276 vs 956.6574, $L/t = 1000$):
  `_explicit` recomputes $r$, $t$ but not λ.

PCG tolerance (32², $L/t = 1000$, tol $10^{-6}$, round 1): $10^{-10}$, $10^{-8}$, $10^{-7}$ give
identical polish ratios at 196 / 142 / 118 PCG its, $10^{-6}$ needs one more step. At tol
$10^{-10}$ (32², $L/t = 100$, without the switch) PCG $10^{-8}$ stalls, $10^{-10}$ and
$10^{-12}$ converge. Hence $\min(10^{-7}, 0.1\,\mathrm{tol})$. One PCG iteration at width $m$
costs about one LOBPCG step (0.17 vs 0.19 s at $128^2$, $m = 25$).

### Storage (device bytes, `amg` defaults, fp64)

Per component (`nbytes` of distinct leaves; level matrices without the operator):

| case | DOF | L0 Schwarz | `schwarz.nbytes` of the L0 patches | $T_0$ B/DOF | coarse levels per shell fiber / hex-only node | total B/DOF |
|---|---|---|---|---|---|---|
| flat $64^2$ | 25k | 10.22 MB | 10.22 MB (+16 B) | 700 | 4.69 kB | 1997 |
| flat $128^2$ | 100k | 40.90 MB | equal | 571 | 5.51 kB | 1910 |
| flat $256^2$ | 396k | 163.6 MB | equal | 529 | 5.80 kB | 1934 |
| stack $128^2\times3$ | 200k | 157.3 MB | equal | 540 | 5.51 kB | 1797 |
| stack $64^2\times10$ | 139k | 290.4 MB | equal | 549 | 4.69 kB | 2801 |
| T 3/3 layers $48^2$ | 45k | 43.0 MB | equal | 614 | 3.97 kB | 1988 |
| block $32^3$ + skin | 111k | 4.6 MB | equal | 563 | 1.70 kB (hex) | 1169 |
| block $24^3$ + skin | 49k | 2.3 MB | equal | 618 | 1.76 kB (hex) | 1225 |

- Level 0's Schwarz equals `schwarz.nbytes` of the level-0 patch sizes (up to 64 B), which the
  mesh alone gives (`_fibers` and `_columns` read only the `ssh8` connectivity).
- $T_0$ (rectangular block-ELL $P_0$ and $P_0^\top$): 501–700 B per fine DOF (518 / 526 / 501
  at $16^2$ / $24^2$ / $32^2$, 700 at $64^2$, 571 at $128^2$; not monotone: ELL padding).
- The coarse levels scale with fiber groups, not DOFs (the 10-layer stack equals the flat plate
  of the same in-plane size): 4.0–5.8 kB per shell fiber, growing with the level count by
  shrinking increments (+0.82, then +0.29 kB per 4× DOFs), and about 1.7 kB per hex-only node.
- The rest (free masks, Chebyshev scalars) is about 8 B per DOF.

### Step, polish, build and count

Blocks are $8nm$ bytes; "extra" is over the Jacobi step, "total" includes the step's state
in the step bullet; the polish table's "9-block total" is a separate measurement (review
round 3, scripts `plan18b-r3-performance/`) whose terms differ (66.5 vs 43.4 at flat $64^2$),
so the polish-vs-step comparison is re-measured with one definition (`memory_analysis`
temporaries + arguments + outputs) in the calibration. The polish columns omit `_pcg`'s output block (+1, review
round 4).

- **LOBPCG step extra** (`memory_analysis`, $m = 15$, no basis; with a basis the AMG steps are
  unchanged, review round 2): 9-block 27.4 / 16.4 / 12.5 (flat $64^2$ / $128^2$ / $256^2$),
  12.4 (stack ×3), 11.4 (stack ×10), 21.6 (T 3/3), 18.2 (block $32^3$); low 40.3 / 20.4 /
  17.5, 16.4, 15.8, 24.9, 22.1. The $n$-independent part is a constant $K_0$ of about 120–150 MB
  (independent of $m$, review round 6; the coarse levels' chunk-bounded Schwarz gathers and unpacked inverses, which
  also covers 18a's Schwarz apply temporaries $2\,\texttt{CHUNK\_BYTES}\,k/24$), so the
  rev 4 fit used the cases from 100k DOF up (rev 7: $K_0$, fitted from 25k). Degree 3: within ±1.1 blocks of degree 2. With one definition (`step_bytes`,
  review round 8) the low step exceeds the 9-block step only on web $64^2$ at $m \le 8$ (116
  vs 108 blocks at $m$ = 8, 149 vs 138 at $m$ = 6); flat $64^2$ at $m$ = 15: 9-block 65.9, low
  63.9. A switching solve needs the larger: `lobpcg_memory` takes the max.
- **Polish totals** (`_pcg(ap_operator, V-cycle, b)` temporaries + X + b, review round 3):
  XLA hoists the V-cycle's preprocessing (the unpacked Schwarz inverses) out of PCG's
  `while_loop`, so it stays allocated for the whole solve: at flat $64^2$, $m = 15$, one
  V-cycle needs 50.4 blocks, inside a loop 131, the polish 162 against the 9-block total
  66.5. An `optimization_barrier((P, r))` before `z = P(r)` in the loop body stops it:

  | case | $m$ | 9-block total | polish, today | polish, with the barrier |
  |---|---|---|---|---|
  | flat $64^2$ (25k) | 15 | 66.5 | 162 | 75.6 |
  | flat $128^2$ (100k) | 8 / 15 / 25 | | +2.65 over the step | 38.7 / 32.1 / 29.1 (1.8–1.9 below) |
  | flat $256^2$ (396k) | 25 | 27.4 | +0.07–0.4 | |
  | hex8 $40^3$ (AMG) | 15 | | below | 27.2 (1.7–2 below) |

  PCG with the barrier: 100 iterations 3.50 s instead of 8.02 s at $64^2$, 12.69 vs 12.79 s at
  $128^2$. The polish's excess is $n$-independent: $K_{0,P}$ about 150–190 MB (independent of $m$, as
  the step's; review rounds 6–7).
- **Host setup peak** (fresh subprocess, above the built operator, CPU: includes the device
  storage): flat 0.46 / 1.33 / 2.61 / 5.14 GB at 100k / 396k / 791k / 1.58M DOF, stack ×3
  $128^2$ 3.8 kB/DOF, stack ×10 $64^2$ 4.8 kB, T 3/3 $48^2$ 283 MB (45k DOF), block $32^3$ /
  $48^3$ 3.7 / 3.9 kB, block $64^3$ (837k DOF) 3.28 GB. Peak minus storage is not linear in
  $n$: 2.67 / 1.42 / 1.32 kB per DOF at 100k / 396k / 1.58M on flat plates (a fixed part of
  about 150 MB), growing by 2.77–2.78 kB per hex DOF from block $32^3$ to $64^3$. Coarse levels
  per hex-only node: 2.18 kB on blocks $48^3$ / $64^3$ (three levels), 1.70–1.76 kB on the
  two-level $24^3$ / $32^3$. Flat $362^2$ (791k): 5.93 kB per fiber, $T_0$ 533 B/DOF.
- **Count:** prototype-measured at flat $64^2$ (the PR 14 recurrence with the V-cycle, both
  barriers: $K_0$ about 138 MB, independent of $b$); fitted in implementation.


## Design (18b)

### Dispatch (`eigen/api.py`)

- `resolve_precond`: `"auto"` → `"gmg"` on voxel meshes; on unstructured meshes `"amg"` unless
  the mesh has `ssh8` with `hex20`/`hex27` (`"jacobi"` there, as now). `precond="amg"` on such
  a mesh raises `amg()`'s "solid-shell AMG supports ssh8 and hex8 blocks" before any build (in
  `solve_modes` and `solve_modes_memory`).
- `no_shell` is deleted: defined at `precond/pmg.py:214–216`, called at `api.py:41` and
  `memory.py:347`, imported at `api.py:18` and `memory.py:27`; `pmg` keeps its own refusal
  (`pmg.py:253`).
- The `make` / `lobpcg_precond` closures, `_precond_sigma`, band-mode aggregate reuse and the
  slices need no change: `amg` takes `nonsingular`, `return_aggregates` and `aggregates` on
  shells already (18a).

### Low-memory switch (`eigen/lobpcg.py`)

- `SWITCH = 5` (stuck iterations before the switch). In `_iterate` (9-block mode), when
  `it - it_best >= SWITCH` for the first time in a chunk, `_to_low(X, P, ...)` (one jit,
  donating X and P; the 9-block state's other buffers are dropped first) deflates X and P
  against $R$ and the locked basis (`_deflate`) and returns the low state `(X, P, actP, λ)`;
  the rest of the chunk runs `lobpcg_step_low` with `c = k` (the low step's own width, so
  `STEP_FIT["low"]` holds). The progress window (`best`, `best_l`, `it_best`) resets at the
  switch, so the low step gets a full `STALL` window (new since the prototype: the "switch:
  its" column above is re-measured). Sticky: returning to the recurrences re-stalls
  (measured trigger form). `_to_low` deflates X and P but does not re-orthonormalize X: the
  leak builds up only after the switch point (measured at the switch on free $24^2$, $L/t$ 50
  and 1000, and the cylinder: $\|R^\top MX\|$ 0 to 4 digits, $\operatorname{diag}(X^\top MX)$
  = $1 - 10^{-15}$). No switch and no `_final_rr` when a `stop` callback is given, on any exit
  (`_precond_sigma`'s fallback `sigma_estimate(maxit=10)` ends by `maxit`, since `stop`
  needs 11 Ritz sums: with the RR its σ moved in the 12th digit at $L/t$ = 50 and the main
  solve went 150 → 230 iterations at $L/t$ = 1000; review round 6).
- **Where the check sits:** in `_iterate` after the progress update and after the
  convergence check (a check that passes on the same iteration wins). `_explicit` (the
  in-loop convergence check, which writes the explicit products back into the state) keeps
  both branches, keyed on the current step kind (`low or switched`), so a never-stuck 9-block
  solve stays bitwise unchanged.
- **`_project` clamp-safe** (review round 4): the capacity is a multiple of the 9-block `c`
  (`_capacity`, `_grow`); only in band mode can the post-lock count (`count + j`, with
  $\text{cap} - \text{count} \ge k - j$) make $\lceil\text{count}/k\rceil k$ exceed it
  (non-band chunked solves keep $\lceil\text{count}/k\rceil k < \text{cap}$), so there and a `c = k` chunk
  would start past the end (`dynamic_slice` clamps the start; columns already projected were
  subtracted again). The body slices at $s = \min(ic, \text{cap} - c)$ and masks the $c\times k$
  coefficients `Bc.T @ MY` to the columns in $[ic, \text{count})$ (no new $n\times c$
  temporary; measured: `test_step_memory`'s blocks identical); unchanged results for every `c` that divides the capacity.
- `t_iteration` (`t_warm` over its own counter of warm iterations of the solve's own step,
  not `it`; the docstring states it may average few iterations on an early switch) counts
  warm iterations of the solve's own step (all low iterations with `low_memory=True`, as
  today); after a switch the first low iteration (`switched` + 1, which compiles the low step;
  `_to_low` runs after iteration `switched` is timed) and the low iterations after it are
  left out of both sums (it feeds `counting.py:155`'s bound, made for the 9-block step).
- `lobpcg_step` and `STEP_FIT["9-block"]` are unchanged: a 9-block solve that never gets
  stuck for 5 iterations is bitwise unchanged.
- `rec["switched"]`: the number of the 9-block iteration after which `_to_low` ran, 1-based
  like `iterations` (the low steps start at `switched + 1`; None if never), in
  `info["chunks"]`; pinned in test 3.
- `_to_low` projects with the solve's `c` (the capacity is a multiple of it); its peak (X, P,
  MX, an $n\times c$ slice, B) is below the 9-block step's.

### `_iterate`'s exits (`eigen/lobpcg.py`)

`_iterate` returns `(X, λ, r, t, j, rec)` instead of a state (`_solve` uses only X and λ after
it; `state = (X,) + state[1:]` in `_solve` becomes `Xi = ...`, a name of its own: the next
chunk's start `X0 = Xi[:, j:]` is taken before `X, g = _reortho(...)`, which zeroes columns
≥ j; only the exit path's `_explicit` call is replaced by `_final_rr` / `_confirm`):

- **converged:** as today (`_confirm` products; acceptance after the explicit check).
- **any exit with a `stop` callback** (σ estimates): as today (`_confirm` residuals, λ
  unchanged; no switch, no `_final_rr`).
- **`STALL` or `maxit`, no `stop`:** `_final_rr(X)` = `_start(X)` (deflation against $R$ and the basis,
  then SVQB) and `_rr_start`: X rotated, λ the Ritz values $\mu_i$ of the deflated pencil
  (ascending). With $\nu_i$ the eigenvalues of the deflated pencil, interlacing gives
  $\lambda_i \le \nu_i \le \lambda_{\text{count}+i}$ and Courant–Fischer $\mu_i \ge \nu_i$: so
  $\mu_i \ge \lambda_i$ always, and $\mu_i \ge \lambda_{\text{count}+i}$ (the bound for the
  wanted index) exactly for the first chunk and later up to the locked basis' accuracy, also
  when X held leaked rigid components. Explicit $r$, $t$ (`_confirm`).
  `_final_rr` and the polish's `_start` use the solve's `c` from `_capacity` (reusing `_begin`'s
  compiled `_start`), while the switched low step uses $c = k$: both widths reach `_project`
  (its docstring says so; the clamp handles both, so they are not unified). If SVQB drops a column, λ and X stay as
  they were (today's behaviour). The polish's step 4 acceptance runs on this result first (a
  `STALL` exit whose Ritz vectors already pass converges, `stagnated` True; a `maxit` exit
  that passes converges, `stagnated` by the rule below). Then, on a `STALL` exit with
  `polish` given and some $r_i > t_i$ up to $j$, the polish below. Then, with `lock` and
  $r_i^2 \le t_i$ for the wanted columns, the guard check (`_extent(..., failed=True)`). A
  `fix:` commit of its own (all solvers, hex too). `sigma_estimate` (the `stop` exit) is
  unchanged; `_precond_sigma`'s own rounds (no `stop`) get it on their `maxit` exits, so their λ is a Ritz
  value (an upper bound up to the rounding floor; converged rounds keep the step's value,
  measured 1.8e-6 below `eigsh` at $L/t$ = 1000 on free $24^2$, inside the floor $2\cdot10^{-3}$),
  and its rounds can switch (a 4-iteration stuck window measured on free $24^2$ at $L/t$ =
  1000, one short of `SWITCH`).
- `_solve` drops the start block after `_begin` (`X = X0 = None`), so only the chunk's own
  state is live during `_iterate` and at the polish (one block less at the step peak too).

### Polish: one inverse-iteration step on a stagnated chunk (`eigen/lobpcg.py`)

- `lobpcg(..., polish=None)`, also `_solve`. `polish`: a host callable `X -> (Y, iterations)`
  with $Y \approx A_P(\sigma)^{-1} M_m X$ (masked), $A_P$ SPD.
- The polish runs in `_solve` after `_iterate` returns (so `_iterate`'s state is gone).
  From `_final_rr`'s X, λ, r, t, one step:
  1. $Y$, its = `polish(X)` on the whole block ($m$ columns); only X and λ of the chunk are
     live (`_iterate` returned no state, the start block was dropped, and `_final_rr`'s and
     `_confirm`'s KX, MX are deleted before the call);
  2. `_start(Y)` (deflation against $R$ and the basis, SVQB) and `_rr_start`; if SVQB drops a
     column or Y is not finite, the polish is discarded;
  3. explicit residuals and thresholds of the new X (`_confirm`);
  4. if every wanted $r_i \le t_i$: $j$ = max($j$, `_extent(...)`) on the new Ritz values;
     accept (the chunk converges) when every $r_i \le t_i$ up to $j$;
  5. otherwise revert to `_final_rr`'s X, λ, r, t (a polish that does not accept leaves the
     result bitwise equal to `on=False`, unless step 4's `_extent` raises). A guard error from step 4's `_extent` on the
     polished Ritz values raises, as on the converged path.
- No LOBPCG step follows a polish. A `maxit` exit is not polished (`maxit` is the user's
  budget; documented). Multi-step polishing is not built (every measured case accepts in one
  step after the switch; before the switch 4–6 steps were needed; logged).
- **Flags:** a polished chunk that accepts is `converged` True. `stagnated` keeps its meaning:
  True for a `STALL` exit (polished or not) and for a floor acceptance with some $r_i >$ tol.
- `info["chunks"][i]["polish"]`: the PCG iterations of the polish, None if none ran (as
  `switched`; whether it accepted is `converged`).
- `solve_modes` passes `polish` for `precond="amg"` (shell and hex meshes) on every LOBPCG run
  except `_precond_sigma`'s 1-mode estimate. The closure builds $A_P$ = `ap_operator(op, free,
  σ)` once and uses LOBPCG's own `P` and σ: `X -> pcg(A_P, m M X, P, ptol, maxit)` returning
  $Y$ and the max iterations over columns. Band mode and the chunked driver get it too. Not
  offered: Jacobi (its PCG needs thousands of iterations), GMG and p-MG (no stagnating case
  measured; logged). Hex AMG gets it because the rule is one branch (`precond == "amg"`),
  not from a measured need; it costs nothing there (test 6).
- `solve_modes(..., polish_options=None)` and `solve_modes_memory(..., polish_options=None)`:
  `on` (default True), `tol` (default $\min(10^{-7}, 0.1\,\mathrm{tol})$), `maxit` (default
  500). Validated like the other option dicts (unknown keys, `on` a bool, `tol` > 0, `maxit` ≥
  1 int); with a non-AMG `precond` (also "auto" resolved to jacobi / gmg) it raises
  ("polish_options needs precond='amg'", with the resolved label). `on=False` passes
  `polish=None` and drops the memory term. Ignored where no LOBPCG runs (band mode with
  `strategy="shift_invert"`, or `"auto"` falling back to slices), as `low_memory` is.
- **Barrier** at the three loops that apply the preconditioner: `_pcg` and `_minres`
  (`linalg/krylov.py`: `Q, r = lax.optimization_barrier((P, r)); z = Q(r)` in `_pcg`'s loop
  body, `Q, r2 = ...; y = Q(r2)` in `_minres`') and `counting._recurrence`'s body
  (`Q, w = lax.optimization_barrier((P, w))`, then `_pcg(A, Q, ...)`: `P` is a closure
  there, so a new name; the inner barrier alone leaves the recurrence unchanged). XLA then keeps the
  V-cycle's preprocessing inside the loop. Review round 4, flat $64^2$: shell MINRES ($b$ = 6)
  456 → 187 MB, 100 iterations 2.27 → 1.91 s; the recurrence 421 → 158 MB ($b$ = 4), 484 →
  218 MB ($b$ = 16); PCG as in the table above. Hex AMG, voxel GMG, p-MG: neutral or faster
  (p-MG temps 191 → 98 blocks), Jacobi bitwise equal; the recurrence temps of
  `test_step_memory`'s cases bitwise unchanged, elsewhere ≤ the count model (voxel GMG
  $32^3$ $b$ = 16 +1.9%, ratio to `GMG_COUNT` ≤ 0.97 up to $64^3$).
- **Cost:** one PCG solve at width $m$ (47–118 PCG iterations measured, each about one LOBPCG step);
  at most `maxit` = 500. The `STALL` = 50 iterations before it stay (an earlier trigger logged,
  not built).

### Memory (`memory.py`)

- `lobpcg_memory`: step = max(mode step, polish), and in 9-block mode also the low step (the
  switch): 9-block mode max(9-block, low, polish), `low_memory=True` max(low, polish). The low
  term wins only for hex8 AMG below about 240k DOF and the p family below about 56k (never for
  GMG or Jacobi; review round 3); it applies to every preconditioner.
- polish $= f\cdot\text{ws} + \texttt{pc["polish"]}$ when the preconditioner estimate has a
  `polish` key (the AMG shell family only): `pc["polish"]` $= 8n(a_P m + c_P) + K_{0,P}$,
  fitted to the measured polish (the `_pcg` temporaries with the barrier plus its output
  block, X and b) minus $f\cdot\text{ws}$ (modelled ws, the mode's `STEP_FIT` $f$), fitted at $m$ = 6 / 8 /
  15 / 25 / 60 / 120, valid for $6 \le m \le 120$ (upper envelope; flat $64^2$–$256^2$, stack ×3 and ×10, web $64^2$, block $32^3$: the
  polish exceeds the larger step by up to 13.7 blocks on web $64^2$ at $m$ = 8; fitted on
  25k–396k, valid from 25k DOF, as the step fit). The envelope's slack: up to about 12
  blocks above the steps at $m$ = 6 near 100k (about 57 MB; 77 MB at $m$ = 8) (every shell AMG estimate carries it, since
  the polish is on by default), negligible at 791k (recorded in the 791k test). Hex AMG has no key: its measured polish is within ±1.3 blocks of
  the measured steps and 10–29 blocks below the modelled step (hex20 $14^3$, hex8 $24^3$,
  $40^3$; tested). With `polish_options={"on": False}` the key is
  dropped.

### `amg_memory` shell family (`memory.py`)

Meshes with `ssh8` blocks (and `hex8`), at the defaults' structure:

- **storage** $= \texttt{schwarz.nbytes}(\text{level-0 patch sizes}) + \alpha_{T,s} n_\text{shell}
  + \alpha_s n_s + \alpha_h n_\text{hex} + 8\min(n_\text{coarse},
  \max(\texttt{max\_coarse}, 1000))^2$, with $n_\text{shell}$ / $n_\text{hex}$ the DOFs of nodes
  in / not in `ssh8` elements, $n_s$ the shell fiber groups, and the coarse term as the hex family's. The level-0 sizes come from a mesh scan
  (`_shell_scan(mesh)`, cached like `_amg_scan`): `_fibers(E, n_nodes)` and `_columns(E, group,
  n_nodes)` on the mesh's `ssh8` connectivity (`amg` calls the same functions on the operator's
  tables). Constants (named like `AMG_STEP` / `AMG_BUILD`; the docs list them by name): $\alpha_{T,s}$ = 880 B (700 measured max, non-monotone, 1.25×; includes the
  8 B/DOF rest), $\alpha_s$ = 6.5 kB (5.93 kB at 791k; increments shrink), $\alpha_h$ =
  1.52 kB per hex-only DOF (level-0 650 B + coarse 2.6 kB per node / 3: the two are
  collinear, $n_\text{hex} = 3 n_h$, as the hex family's single `AMG_STORAGE["hex8"]`; the
  per-component numbers stay in the docs table); fitted per leaf group, not on totals (single-layer plates have $n_s = n/6$, so a total fit is collinear). The
  calibration adds the sweep-built cylinder, the L-fold (non-box meshes) and block $64^3$.
- **build** $=$ **host_build** $= \beta + \text{storage} + \gamma_s n_\text{shell} +
  \gamma_h n_\text{hex}$, the PR 13 convention (process peak above the operator; on a GPU the
  stored part is device memory, conservative there; `pc_dev` = 0: the device inversion
  temporaries are a few `CHUNK_BYTES`). β, γ fitted as the upper envelope (LP, as `AMG_STEP`)
  of peak minus measured storage over every case from 45k to 1.58M: about β = 150 MB,
  $\gamma_s$ = 1.4 kB (1.28–1.30 measured above 396k), $\gamma_h$ = 3.3 kB (2.78 measured; the
  hex family's growth toward 10M). Model / measured build: block $64^3$ 1.27× (gate 1.35×),
  flat $362^2$ 1.20× (gate 1.3×).
- **step**, **step_low** $= 8n(a m + c) + K_0$, with $K_0$ in bytes independent of $m$ (the
  chunk-bounded gathers and unpacked inverses of the coarse levels: about 120 MB at flat
  $64^2$ for $m$ = 8 and 25 alike; the hex family's $8mK$ form fits there because its $K$ is
  small, review round 6), fitted at $m$ = 6 / 8 / 15 / 25 / 60 / 120 (valid for $6 \le m \le
  120$: 50 and 100 modes give 60 and 120; the $m \le 25$ fit under-estimated $m$ = 120 by
  2.6 blocks at 25k, review round 7) on flat $64^2$–$256^2$, web $64^2$,
  block $32^3$, stack ×10 $64^2$ and ×3 $128^2$ (25k–396k; upper envelope, with and without a
  basis), every case at every $m$, valid from 25k DOF; `AMG_LOOP_K` re-measured for degree
  > 2. **Fit target:** `lobpcg_memory` adds $A b + f\,\text{ws}$ (`STEP_FIT`, modelled ws)
  to `pc["step"]` (`memory.py:165`), so `pc["step"]` / `pc["step_low"]` are fitted to the
  measured `step_bytes` minus that term, and the gates compare `lobpcg_memory`'s step with
  the measured total (review round 10: fitting the totals double-counted the Jacobi step,
  16 + 2.5 ws blocks). Every point is in the fit. Web $64^2$ (26k) at $m \ge 60$, flat at
  37.9 blocks from $m$ = 60 to 120, is the Jacobi step's (Jacobi alone measures the same)
  and lands +7.2 blocks over in the refit on the round 8/9 data ($a$ = 5.67, $c$ = 15.9,
  $K_0$ ≈ 74 MB); large-$n$ slack 0–2.4 blocks (flat $128^2$, stack ×3 $128^2$ at $m$ = 25 /
  120). The shipped constants are rounded up, with $c$ raised by 0.5 against rounding (the
  hex family's +1.5, `memory.py:296–298`). $c$ is a
  fit parameter and may be negative ($c$ and
  $K_0/n$ both multiply $1/m$); the model holds only for $m \ge 6$ together with $K_0$, as the
  constants' comment states.
- **count** $= 8n(a b + c) + K_0$ from `_recurrence` (with both barriers) with the shell
  V-cycle, $b$ = 4–32, fitted on flat $64^2$, $128^2$ and stack ×3 $128^2$ (25k–200k; about
  138 MB independent of $b$ at flat $64^2$), and the same constants checked against the shell MINRES temps
  (`si_memory`'s minres phase is SI_MINRES·b + ws + `pc["count"]`).
- `amg_memory` picks the shell family when `ssh8` is present; the hex family is unchanged.

### Benchmark

`benchmarks/shell_solve_study.py` → `benchmarks/results/shell_solve_study.json` (per-step
progress printed; `--parts` selects sections). Estimated 45–55 min:
- accuracy / time (about 12 min): the tables above with the shipped code (AMG; Jacobi where it
  converges: iteration counts), the switch and
  polish cases at $L/t$ = 300 and 1000 ($32^2$, $128^2$), the PR 9 tolerance cases, the SIMP
  GMG stagnation case of `open_items.md:32–34` (whether the switch changes it);
- calibration (about 30 min): storage per component (with the cylinder, L-fold and block
  $64^3$), step / step_low and the polish peak at $m$ = 6 / 8 / 15 / 25 / 60 / 120 on every fit
  case (compile-only, a few minutes more), plus compile-only steps at $m$ = 6 and 120 on the
  flat $362^2$ plate (791k; model / measured recorded: $c$ < 0 is checked only to 396k by
  the fit), count at
  $b$ = 4–32, host peaks in fresh subprocesses up to the 1.58M plate and block $64^3$ (the
  ≥ 2 GB check of `ROADMAP.md:184`: model / measured recorded), shell MINRES temps at $b$ =
  6 on flat $64^2$ / $128^2$.

## Tests (18b)

Shell tests go to a new `tests/test_shell_solve.py`, LOBPCG unit tests to
`tests/test_lobpcg.py`. Slow marks for anything above about 5 s; `peak` + slow + the darwin
skip for the build-peak tests; no wall-time asserts (no `serial`).

1. **Dispatch:**
   - `resolve_precond` gives `"amg"` on `ssh8` and `ssh8` + `hex8`, `"jacobi"` on `ssh8` +
     `hex20`; `precond="amg"` there raises before any build in `solve_modes` and
     `solve_modes_memory` (spy: `amg` and `lobpcg` not called);
   - `info["precond"]`, the "auto (resolved to amg)" label in option errors;
   - a wiring spy: `polish` reaches `lobpcg` with `"amg"` and is None with `"jacobi"`,
     `"gmg"`, `"pmg"`, with `polish_options={"on": False}`, and for `_precond_sigma`.
2. **Accuracy against `reference_modes`** (the roadmap test), 20 modes, tol $10^{-10}$,
   `maxit` 3000, `precond="auto"`, on flat SS $32^2$, free $24^2$ (rigid deflated), 3-layer
   stack $32^2$ ($L/t$ 100, slow), curved panel $32^2$, the free cylinder, T-junction $24^2$,
   the hex8-web plate $32^2$, a plies laminate $32^2$ ($L/t$ 50 unless stated):
   - `converged`, `info["precond"] == "amg"`, `check_lam` (all floor-accepted, `stagnated`
     not asserted);
   - eigenvalue groups from reference gaps < $10^{-2}$ (the `check_lam` rule; the reference
     has $n + 4$ modes); a group straddling mode 20 compares the returned columns with the
     whole reference cluster (`max_sine` takes a wider reference block), none dropped;
   - per eigenvalue group $\sin\theta \le 10\,(r_M + r_{M,\text{ref}})\lambda/\text{gap}$
     and $r_{M,\text{ref},i} \le 10\varepsilon\lambda_\max/\lambda_i$ (measured: ≥ 17× margin);
     sparse variants `r_M_sparse` / `max_sine_sparse` in `helpers.py` (`splu` of $M_f$;
     $\|L^\top E\|_2 = \sqrt{\lambda_\max(E^\top M E)}$), the dense ones unchanged;
   - V M-orthonormal; on the free cases $\|R^\top MV\| \le 10^{-10}$ (fails on `main`: 0.96);
   - `polish` None on every chunk of all eight; `chunks[0]["switched"]` not None on the stack and the
     free cases; `info["iterations"]` ≤ 30 on the curved panel (replaces the Jacobi
     wall-time comparison; measured before pinning).
3. **Low-memory switch:**
   - $32^2$, $L/t = 100$, tol $10^{-10}$: converges, `polish` None, `switched` not None;
   - one parametrized test: representative AMG / GMG / p-MG solves (`test_gap_band`'s
     pencil, the clamped hex20 box at tol $10^{-8}$, a voxel GMG and a p-MG case) have
     `switched is None` on every chunk and λ bitwise equal to the same solve with
     `SWITCH` monkeypatched to infinity (measured: no converging AMG / GMG / p-MG chunk
     switches; 11 of 516 chunks switch, all Jacobi);
   - `_project` unit test: a capacity of 10, `c` = 4, count 8, 9 and 10: $B^\top M Z$ ≤
     $10^{-13}$ (fails today at 9 and 10: 1.44), and bitwise unchanged where `c` divides the
     capacity (the switched solve's wiring: a spy on test 4's band run);
   - `_to_low` unit test (fp64 basis): an injected locked-basis component ($10^{-6}$) in X
     and P is gone ($\le 10^{-13}$) and the low step continues from it (the rigid half is
     covered by test 2's free cases);
   - `test_compiles`: a switch adds exactly one low-step compile (and one `_to_low`) per
     shape, none on a second switching solve of the same shape (the 5th solve there switches
     at iteration 130; the expected list stated in the test).
4. **Polish:**
   - $32^2$, $L/t = 1000$, tol $10^{-6}$ (module fixture, also used by test 5): with
     `on=False` `converged` False, `stagnated` True; with the default `converged` and
     `stagnated` True, `polish` in 50–400, the iteration count equal to the unpolished run's; residuals recomputed from the assembled K, M ≤ the
     thresholds and `check_lam`; a `pcg`
     spy: one call with `maxit` 500 and `tol` $10^{-7}$ (the defaults);
   - free thin plate ($24^2$, $L/t = 1000$, tol $10^{-6}$): `check_lam`,
     $\|R^\top M V\| \le 10^{-10}$;
   - chunked ($n$ = 30, chunks of 10, fp64 and fp32 basis) and band mode with `chunk=10` on
     the thin plate, one test (slow): every chunk converged, `polish` ≤ 500 where not None
     and not None on the fp64 chunks (measured), no mode twice, $V^\top MV - I$ within the PR 9 bounds, `check_lam` (fp32 and band
     measured before pinning); `chunk` chosen so a switched band chunk starts with count + k >
     cap (monkeypatched `SWITCH` = 1 if needed), and a spy asserts the clamp condition
     $\lceil \text{count}/k\rceil k > \text{cap}$ on it, so the run cannot silently miss it;
   - live bytes: `gc.collect()` and a baseline of `jax.live_arrays()` bytes just before the
     `lobpcg` call; a fake `polish` runs `gc.collect()` and records the bytes on entry on a
     stalling diagonal case with `R=None` and the zero preconditioner ($n$ large enough that a
     block dominates, $k \ge 4$, so the $n\times 1$ mask fits in the slack); entry − baseline ≤ basis + X + 0.5 block (a
     leaked KX, MX or P fails);
   - dense unit tests on a PR 9 pencil, `STALL` forced (a small monkeypatched `STALL` and a
     zero preconditioner), `X0` = the $m$ lowest exact eigenvectors + $\delta\,V_\text{hi} G /
     \sqrt{n_\text{hi}}$ ($V_\text{hi}$ the upper half of the dense eigenvectors, $G$ Gaussian,
     $\delta = 10^{-2}\,\mathrm{tol}$; review round 5: r/t 7.8 → 0.002 after one exact step,
     where $10^{-3}$ noise of all frequencies stays at about $10^3$), `C_FLOOR` unchanged; all
     fakes share this start:
     - an exact dense inverse-iteration fake, `Y = solve(K_f + σM_f, M_f X)`, accepts;
     - revert, parametrized over a degrading fake (X + O(1) noise) and a rank-deficient Y:
       λ, V bitwise equal to the `on=False` run, no raise;
     - a `maxit` exit never calls the fake;
     - one case with `low_memory=True`;
     - cluster extension by the polish: `test_multiplicities`' cube pencil (R, zero
       preconditioner, `STALL` = 3, tol $10^{-6}$), `want` = 1 (cluster [0, 1]) or 4 (cluster
       [2, 3, 4]; a 3.3% gap above it against a 1.3% Ritz shift), high-frequency noise
       ($\delta V_\text{hi} G/\sqrt{n_\text{hi}}$) of absolute size $\delta = 10^{-2}$ (not the
       shared start's $10^{-2}\,$tol, which converges at iteration 1; measured on `main`,
       review round 8) on every column of the multiplet (noise on the outside partners only
       converges at once, review round 7); pre-polish asserts with `on=False`: `converged`
       False and an `_extent` spy sees no call with `failed=False` (the guard check's
       `failed=True` call excluded); with an oracle fake
       returning the exact block eigenvectors: `chunks[0]["locked"]` > `want`;
     - the failed-exit guard check runs only after a polish that did not accept;
   - `t_iteration`: with `SWITCH` monkeypatched to 2 (so `switched` ≥ 2 and at least one
     warm 9-block iteration), finite and > 0; with `low_memory=True`, finite and > 0;
   - option errors (`on` not a bool, `tol` ≤ 0, non-int `maxit`, unknown keys, with
     `precond="jacobi"`, with "auto" on `ssh8` + `hex20` (label "auto (resolved to
     jacobi)")).
5. **Failed-exit Rayleigh–Ritz** (on the test 4 fixture's `on=False` run): λ ascending;
   $\max_i |\lambda_i - v_i^\top K v_i / v_i^\top M v_i|/\lambda_i \le 10^{-6}$ with the
   solver's matrix-free op (measured $3.3\cdot10^{-8}$; `main` $6.4\cdot10^{-5}$); off-diagonal
   of $V^\top K V$ over its diagonal ≤ $10^{-7}$ (measured $1.8\cdot10^{-9}$; `main`
   $3.2\cdot10^{-7}$); residuals equal (rtol $10^{-5}$, as `test_stagnation`) to ones
   recomputed with the returned λ and the solver's matrix-free op (assembled K, M differ by
   about $10^{-3}$ at this r/t); an SVQB drop on `_final_rr`'s own `_start` call only (a
   call counter, so `_begin` does not raise) leaves λ, X as before.
   `sigma_estimate`: spies assert neither `_final_rr` nor `_to_low` runs with `stop` given,
   including `_precond_sigma`'s fallback with `estimate_maxit=10` (a `maxit` exit).
   `_precond_sigma` on the free $24^2$ plate at $L/t$ = 1000 (slow; the test 4 free-plate
   fixture; the σ upper bound up to the floor is a docs claim, `eigensolvers.md:465`, not
   asserted: it passes on `main` too): a spy
   sees `_final_rr` run on every `maxit` round of `_precond_sigma` (at least one; two
   measured) and that round's returned λ bitwise equal to its output (the structural half,
   which fails on `main`; a Rayleigh-quotient comparison cannot discriminate: main and the
   fix differ by $2\cdot10^{-8}$ here, at its noise level). `test_stagnation`'s maxit-exit residual
   check still holds (its `_confirm` counting spy adjusted).
6. **Memory:**
   - `lobpcg_memory`: step = max(9-block, low, polish) and max(low, polish); no polish term
     for Jacobi, GMG, p-MG, hex AMG or with `on=False`; one case where the low term wins
     (hex8 AMG, 100k; model-only, no build or solve); on flat $64^2$ at $m$ = 15 the step
     equals the polish term (measured 75.6 vs 66.5 blocks);
   - one parametrized test per (mesh, $m$), building the AMG once and checking the step,
     step_low and polish models against their `memory_analysis` values (polish: the `_pcg`
     temporaries with the barrier + output + X + b), on flat $64^2$ and $128^2$, web $64^2$,
     block $32^3$, stack ×10 and stack ×3 $128^2$ at $m$ = 6 / 25 / 120 (slow above 100k), with
     a basis (the larger case; the AMG steps are unchanged by it), plus T and a degree-3 case
     at $m$ = 15 (the full grid stays in the benchmark JSON), each compared at the
     `lobpcg_memory` level ($A b + f\,\text{ws}$ + the `pc` term against the measured total):
     model ≥ measured everywhere; at large $n$ (flat $128^2$ and stack ×3 $128^2$, $m$ = 25 and
     120) the 9-block and polish models ≤ measured + $G$ blocks, $G$ set after the shipped
     refit between its largest slack there (0–2.4 blocks on the round 8/9 data, before the
     rounding margin) and what a deliberately broken fit gives (fitting the totals: 6.2–8.1),
     stated with both numbers in the test; hex AMG: the modelled step ≥ the measured polish on hex20
     $14^3$, hex8 $24^3$ and $40^3$ at $m$ = 8 / 25 (measured: 10–29 blocks above);
   - shell MINRES temps ($b$ = 6, flat $64^2$ and $128^2$) ≤ `si_memory`'s minres phase;
     `test_counting::test_step_memory` and the slice-memory tests re-run (barriers), and the
     `si_slice` tests (shift_invert `_extend` calls the changed `_project` through
     `_deflate`; its width `L0 + capacity` is a multiple of `_chunk(block)`, so bitwise
     unchanged);
   - `amg_memory` shell storage ≥ the owned device bytes on every case (flat, stack ×3 and
     ×10, T, web, block, cylinder, L-fold, a $48^2$ plate); the scan's sorted level-0 patch
     sizes equal the built hierarchy's; total within 1.3× on flat $128^2$ and block $32^3$
     (slow); `_shell_scan` cached (as `test_amg_scan_cached`);
   - count ≥ the measured recurrence at $b$ = 4, 16 and 32 on flat $64^2$ and $128^2$,
     within 1.3× at 16 on $128^2$;
   - build ≥ the host peak in fresh subprocesses on flat $128^2$, T 3/3 and block $32^3$;
     block $64^3$ (837k): ≥ the peak and ≤ 1.35× (model 1.27×); flat $362^2$ (791k), one
     subprocess running `solve_modes` (20 modes, tol $10^{-6}$, $L/t$ ≤ 100: no switch, no
     polish; runtime measured before pinning, a few minutes as the 13b precedent): storage ≥ the owned device
     bytes (the out-of-fit size), the build model ≥ the peak after the AMG build (above the
     operator) and ≤ 1.3× it (model 1.20×), the final process peak within ±20% of
     `solve_modes_memory` (measured before pinning; `peak`, slow, darwin; the 13b precedent;
     the ≥ 2 GB rule of `ROADMAP.md:184`);
   - `_pc_builds` composition: `solve_modes_memory` on a shell mesh sums as on hex meshes;
   - storage tests on every platform (no darwin skip; the peak tests keep theirs).
7. **Band mode** on a shell: `band=(0, f)` on the flat $32^2$ plate matches the reference
   (every mode in the band) with the aggregates reused (spy: one fresh `amg` setup with
   `return_aggregates`, then `aggregates=`).
8. **M4 and existing tests that change:**
   - `examples/m4_shell.py` (its `maxit=2000` was sized for Jacobi: dropped) and its test:
     under `"auto"` now AMG;
   - `test_m4_reference` (`tests/test_shell_ops.py:169`): AMG; the "reference sharper"
     assertion replaced as in test 2 (it fails under AMG and, with the switch, under Jacobi);
     the explicit `precond="jacobi"` run keeps Jacobi tested;
   - `test_m4_bands`: AMG; its shift-invert slices and count now use shell AMG; its
     Jacobi-sized `maxit=2000` (`tests/test_shell_ops.py:160`) dropped, as for
     `m4_shell.py`, and `test_auto_jacobi_mixed`'s (`:208`) kept (it stays Jacobi);
   - `test_auto_jacobi_mixed`, `test_auto_labels` (under AMG `precond_options={"degree": 3}`
     and `sigma_options={"tol": 1e-3}` become valid: new unknown keys), `test_refusals` (three
     parts), `test_solve_modes_memory`, the `AMG` message constant (`test_shell_ops.py:32`), the
     `amg_memory` refusal tests, `test_stagnation` (test 5), `test_compiles` (test 3);
   - `tests/test_options.py::test_defaults_snapshot` (the AMG box solve now passes `polish`:
     assert it callable, compare the rest) and `test_option_sets_pinned` (`"polish"` in
     `LOBPCG_FORWARDED` and in the exempt set, built internally like `precond`;
     `POLISH_OPTIONS == {"on", "tol", "maxit"}` pinned); `tests/test_strategy.py::
     test_solve_boundary_one` (its `_iterate` wrapper unpacks the new tuple);
   - re-run, may move: the hex AMG estimate tests (`test_mac_acceptance` runs hex20 $44^3$,
     1.08M DOF, where the low term does not win: a re-run only; the hex8 AMG peak tests below
     240k DOF may rise by up to 4%), `test_api.py::test_sigma_estimate`, `::test_amg_sigma`
     and `test_options.py::test_sigma_options_real` (`_precond_sigma`'s `maxit` rounds go
     through the final RR and can switch), `test_counting.py::test_sampling_memory` (Jacobi
     `_pcg` temps under the barrier), the `test_shift_invert` slice-peak tests (the 0.8 lower
     bound, GMG MINRES under the barrier),
     free-body pinned results (the failed-exit RR), `test_counting::test_step_memory`; the
     chunks a switch reaches (prototype regression log, round 5: `test_floor[clamped]` at
     230, whose history-minimum ratio assert 0.1–10 may move, `test_multiplicities[beam]`
     and `[cube]`, `test_cluster_at_chunk_boundary`, `test_near_singular_mass`,
     `test_guard_error` at 406, plus the failed-exit RR before its guard check),
     `test_failed_chunk` (failed-exit λ), `tests/test_readme.py::test_readme_snippets` (the
     laminate snippet under AMG, `maxit` dropped);
   - `helpers.py`: `r_M_sparse` / `max_sine_sparse` equal the dense versions (rtol
     $10^{-10}$) on a small pencil;
   - `test_step_memory` (`tests/test_lobpcg.py:511`), `test_estimate_chunks` and the
     `STEP_FIT` pins: unchanged by design, run as the check.
9. **Benchmark smoke** (slow): `shell_solve_study --parts` at small sizes records counts,
    times and the calibration entries.

## Docs (18b)

- `docs/eigensolvers.md`:
  - `:25` (`solve_modes` signature: `polish_options`), `:70–75` (the option-dict bullets: a
    `polish_options` bullet, `on` the only off switch, and its checks in the validation
    sentence; ignored where no LOBPCG runs, as `low_memory`), `:31` (step 4 `"auto"`: the `ssh8` +
    `hex20`/`hex27` Jacobi exception and the up-front raise), `:33` (the AMG bullet: the
    `polish` closure), `:36` (step 5: `polish` forwarded);
  - `:41` (the solid-shell bullet: Jacobi stagnation text and M4 timings rewritten for AMG;
    the `precond="amg"` / `amg_memory` refusal sentence dropped, the `pmg` refusal kept and the
    `ssh8` + `hex20`/`hex27` up-front raise added;
    naming the mesh and count of each Jacobi number: 840 LOBPCG iterations here, 1644 /
    1289 at `preconditioners.md:55`; `:40` names `ssh8` among the remedies), `:79` (the `"auto"` rules) and `:88`
    (the `max_coarse` row: the remedy now solid shells with AMG, "since PR 18b");
  - `:119` (`t_iteration`: warm iterations of the solve's own step; switched low
    iterations left out), `:313` (`lobpcg` signature:
    `polish`), `:315` (`info` / `chunks` keys: `polish`, `switched`; λ, r, t on a failed exit from the final
    Rayleigh–Ritz), `:316` (deflation: plus the one at the switch), `:320` (`low_memory`: a
    9-block solve can end in the low step);
  - `:361` (RECOMPUTE step 7: the switch to the low step after 5 stuck iterations), `:374`
    (Failed chunk: λ, V, r, t from `_final_rr`; "a no-progress or `maxit` exit ends the solve"
    becomes: a `STALL` / `maxit` exit that passes after `_final_rr` or an accepted polish locks
    and continues, otherwise it ends the solve), `:376` (Projector: the clamped start and the
    masked coefficients, `c = k` after a switch), `:391` (Low-memory mode: also the switch
    target; a pointer);
  - new subsections under Convergence: "Low-memory switch" and "Polish" (the three causes, κ ∝
    $(L/t)^4$, the measured tables from `benchmarks/shell_solve_study.py` and its JSON, the
    defaults; inverse iteration accepts on residuals and
    does not recover a mode missing from the block, so the completeness checks still apply; a
    `maxit` exit is not polished; a switch costs one extra compile);
  - `:333`/`:337` (one compiled variant per shape: plus the low step and `_to_low` after a
    switch), `:404–415` (Memory model: step = max(9-block, low, polish); with shell AMG
    `low_memory` saves less, the polish's PCG being above the low step on small meshes;
    `:408` / `:411` the AMG step term per family: the shell family's $K_0$ form, a pointer to
    preconditioners.md; `:415` the `solve_modes_memory` signature: `polish_options`), `:429`
    (the "no progress for 50 iterations" row: the window resets at the switch, up to 5
    iterations in the 9-block step then 50 in the low step), `:419–430` (Convergence
    exit table: rows for "`STALL` / `maxit`, final Rayleigh–Ritz passes" and "polished,
    accepted", `converged` and `stagnated`), `:462` (σ pre-estimate step 2: Ritz values on
    `maxit` rounds, the bound up to the floor; rounds can switch), `:465` (the tests' σ
    range: the shell bound down to the `check_lam` floor), `:615` (`count_memory`: the
    hex family unchanged, the shell family in preconditioners.md), `:735` (`si_memory`'s minres
    phase: checked on shells), `:496–511` (the Jacobi hex
    table: re-measured or marked "before 18b");
  - the Options table row (`polish_options`), the tests list (`:813` Stagnation, `:820`
    recompute schedule, `:825` Compiles, `:849` the options snapshot (`polish` checked as
    callable), `:852` the option-error tests (`polish_options`); `tests/test_shell_solve.py`), the `tests/test_counting.py`
    κ ≥ $10^7$ bullet at `:873` (its thin-plate sentence: counting on `ssh8` with AMG is
    covered by test 7, at κ ≈ $3\cdot10^6$; the same wording at `open_items.md:104`),
    "Recurrence and solves" `:561–584` (one line: the barrier in the recurrence's loop),
    `:85` (the options table's `degree` row: degree 3 for curved shells, a pointer).
- `docs/preconditioners.md`: `:55`, `:474` ("still refuse AMG on `ssh8` until PR 18b"),
  `:510`, `:565`, `:570`, `:615` (the `test_refusals` sentence; `tests/test_shell_solve.py`):
  the `amg_memory` shell family (formulas, the constants listed by name, measured tables, the
  `polish` key), the "AMG on solid shells" `solve_modes` sentence, the degree-3 note
  (`:548`; `precond_options={"degree": 3}` for curved panels, as 18a promised,
  `open_items.md:251`); the shell-family tables from `benchmarks/shell_solve_study.py`; the
  validity ranges next to the constants ($6 \le m \le 120$, $4 \le b \le 32$, from 25k DOF;
  $c$ a fit parameter that may be negative; the fit target net of `STEP_FIT`; `preconditioners.md:371`: the default for unstructured meshes, solid shells since 18b, Jacobi with `ssh8` + `hex20`/`hex27`);
  `:88` and `:97` (PCG and MINRES "Memory" bullets: the barrier, also in the count
  recurrence).
- `docs/open_items.md`: `:16–22` (the solid-shell part moves to a Resolved entry at the end,
  the file's convention; the hex / voxel thin-plate part keeps its entry, retitled "(PR 11a; solid shells resolved
  in 18b)"), `:32–34` (SIMP GMG stagnation: the benchmark's result with the switch),
  `:84` (options kept internal: `lobpcg`'s `R`, `precond` and `polish`, set by
  `solve_modes` and tuned through `polish_options`), `:104` (counting on `ssh8`: test 7's band), `:210`, `:211` (the 4 mm laminate), `:243`,
  `:244` (`ssh8` + `hex20`/`hex27`: `"auto"` keeps Jacobi, `precond="amg"` raises up front),
  `:245`, `:251`; the 18b decisions (Risks below).
- `docs/ROADMAP.md`: `:18` (status line), the PR 18b section (status, a Docs bullet as 18c's,
  a Contents bullet for the
  switch, the polish, the failed-exit Rayleigh–Ritz and the barriers in `_pcg`, `_minres`
  and `_recurrence`), `:184` (the
  ≥ 2 GB validation: the 791k and block $64^3$ tests and the benchmark), `:475`, `:494`,
  `:580` and `:590` (M4 under AMG), PR 18b Tests bullets (thin-shell convergence at $L/t$ =
  1000, $\|R^\top MV\| \le 10^{-10}$ on free shells, the 791k / block $64^3$ peak checks), `:784` (the 18a host-peak check: CPU calibration done, the
  GPU part kept as one entry); Deferred GPU checks: the shell step / polish fits and the
  barrier's effect on a GPU.
- Code: `eigen/api.py:311–317` (`precond`, `polish_options`; the allowed-keys list gains
  `POLISH_OPTIONS`), `memory.py:396–410`
  (`solve_modes_memory` docstring: `polish_options`), the `pcg` / `_minres` comments in
  `linalg/krylov.py` and `counting._recurrence` (the barrier), `memory.py:144–152`
  (`lobpcg_memory` docstring: the max) and `:338–345` (`amg_memory`), `eigen/lobpcg.py:18`
  (RECOMPUTE), `:23` (STALL: before the polish), the new `SWITCH` comment, `:113–114`
  (`_project`: clamped start, masked coefficients), `:403` (`_iterate` returns `(X, λ, r, t,
  j, rec)`), `eigen/api.py:226` (`_precond_sigma`: an upper bound up to the floor), `eigen/lobpcg.py:368` and the
  `lobpcg` docstring (`polish`, `low_memory` and the switch, flags), the shell constants'
  comment in `memory.py` ($c$ a fit parameter, valid $6 \le m \le 120$ and from 25k DOF,
  fitted net of `STEP_FIT`), the `_solve` (`:470`, `polish`) and
  `_explicit` (`:458`, `low` means low or switched) docstrings,
  `examples/m4_shell.py:2`.
- `docs/elements.md:218`; `README.md:5` (status), `:82` ("`solve_modes` (Jacobi-preconditioned
  …"), `:91` and `:95` (the laminate snippet's thickness comment and the snippet: `maxit=2000` dropped; the 4 mm plate can come back),
  `:242` (AMG on unstructured meshes: solid shells since 18b), `:252` (options reachable
  through dicts: `polish_options`), `:301` (eigensolvers.md contents: low-memory switch,
  polish); `docs/verification.md:239` (AMG reaches the floor).

### Carried items (18a plan `:199–238`)

- **Done here:** the dispatch, the σ pre-estimate check, band-mode reuse, the `amg_memory`
  shell family, the storage checks on every platform, the accuracy cases (plies included).
- **Changed:**
  - the storage form: $\alpha_{T,s}$ per shell DOF and one $\alpha_h$ per hex-only DOF instead of
    $\alpha_P n_{c,\text{shell}}$; the step's level-0 chunk temporaries fitted ($8n(am+c) +
    K_0$) instead of exact from `schwarz.nbytes` (the $K_0$ term covers the Schwarz apply
    temporaries $2\,\texttt{CHUNK\_BYTES}\,k/24$); the hex-constant extrapolation decision
    kept for the hex family only;
  - the 1.3× tightness gate: on totals (flat $128^2$, block $32^3$) instead of "the n-scaling
    part", with hex-dominant meshes no longer upper-bound only (block $32^3$: model / measured
    total 1.26× with the per-DOF constants);
  - the build convention: `pc_dev` = 0 instead of the device inversion temporaries (about
    7 × `CHUNK_BYTES`, negligible against the step);
  - the Jacobi stagnation test: Jacobi converges at $L/t = 50$ (840–1307 iterations), so the
    case moves to the curved panel and counts AMG iterations (test 2);
  - the Jacobi wall-time comparison (`serial`): replaced by test 2's count (wall time on a
    loaded machine is the flaky part).
- **Dropped:** the degree-3 curved fallback as a default (degree 2 suffices on every measured
  shell; degree 3 stays an option, documented and tested by the step model).

## Measured (prototypes, 18b planning)

Scripts in the session scratchpad: `s18b/` (`proto.py`, `thin2.py`, `floor.py`, `kappa.py`,
`scilob.py`, `polish.py`, `storage.py`, `step.py`, `peaks.py`), `s18b2/` (`harness.py` the
patches: `SWITCH`, `TRIG`, `DEFL`, `POLISH`; `plate.py`, `free2.py`, `acc2.py`, `acc3.py`;
logs `run3.log`, `run4.log`), and the reviewers' folders `plan18b-r1-*`, `plan18b-r2-*`,
`plan18b-r3-*` (the barrier: `plan18b-r3-performance/polmem*.py`), `plan18b-r4-*` (the
clamp: `plan18b-r4-correctness/clamp.py`; MINRES and the recurrence:
`plan18b-r4-performance/mr*.py`, `rec*.py`), `plan18b-r5-*` (the dense polish start:
`plan18b-r5-tests/inv2.py`; hex / shell polish: `plan18b-r5-performance/hexpol.py`,
`shpol.py`).

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **Eigensolver behaviour changes in 18b** (the shell solve needs them; they are not deferred
  to a later part): the low-memory switch, the polish, the failed-exit Rayleigh–Ritz and the
  barriers. Without them the roadmap test fails at tol $10^{-10}$ and "auto" AMG
  stagnates on typical thin shells. Unchanged: 9-block solves that never get stuck for 5
  iterations (every measured converging AMG / GMG / p-MG solve) and `sigma_estimate`. Changed:
  `_precond_sigma`'s own `maxit` rounds (Ritz values: an upper bound up to the floor; they
  can switch),
  slow or stalling Jacobi and GMG solves that get stuck (late switch; Jacobi measured
  harmless; the SIMP GMG case re-measured), every `STALL` / `maxit` exit (Ritz λ; a `STALL`
  exit can now converge without a polish), PCG, MINRES and the count recurrence (same
  arithmetic, different scheduling).
- **The rigid-mode leak on `main`** (free thin shells return rigid modes as elastic): fixed by
  the switch (the low step does not leak; one deflation at the switch). A `fix:` commit of its
  own, with the before / after table.
- **Switch, not recompute:** recomputing in the 9-block step left W's products implicit and
  raised the step's compiled peak; the low step costs 1.4–1.9× per iteration after the switch.
  `SWITCH` = 5, sticky.
- **Polish:** one step, `"amg"` only (shell and hex); GMG and p-MG unmeasured; pure inverse
  iteration (a LOBPCG restart and LOBPCG with an inner-PCG preconditioner rejected);
  multi-step not built; defaults `tol` $\min(10^{-7}, 0.1\,\mathrm{tol})$, `maxit` 500; a
  `maxit` exit is not polished; no polish in `_precond_sigma`; `polish_options` with a non-AMG
  `precond` raises instead of being ignored.
- **Flags:** a polished accepted chunk is `converged` and `stagnated`; a non-accepting polish
  reverts (accept-or-revert).
- **Not split into its own part:** the polish is needed only from $L/t \approx 1000$ (no
  accuracy case polishes), but a 1 mm plate of 1 m span at the default tol $10^{-6}$ fails
  without it, and its pieces (closure, options, memory key) are small; kept in 18b.
- **`STALL` = 50 kept**; an earlier trigger logged, not built.
- **No switch and no final Rayleigh–Ritz when a `stop` callback is given** (`sigma_estimate`
  bitwise unchanged, also its `maxit` exits).
- **`t_iteration`** counts warm iterations of the solve's own step, not switched low
  iterations (the input of `counting.py:155`'s
  time bound).
- **Floor-limited accuracy:** at $L/t = 1000$ the λ error is about $5\cdot10^{-6}$ (residual
  floor $\varepsilon\kappa \approx 10^{-4}$); stated in the docs.
- **`ssh8` + `hex20`/`hex27`** stays Jacobi under `"auto"` (nonconforming meshes, 18a).
- **`no_shell` removed.**
- **Barrier** in `_pcg`, `_minres` and the count recurrence (all three loops the shell solve
  uses); same arithmetic, different scheduling.
- **`_project` clamp-safe** for chunk widths that do not divide the capacity (also used by
  shift_invert's `_extend`, bitwise unchanged there).
- **Memory:** `lobpcg_memory`'s max with the low step for every preconditioner; the polish term
  for the AMG shell family only, on by default, so every shell AMG estimate carries its
  envelope slack (up to about 12 blocks at $m$ = 6–8 near 100k); storage per shell DOF, fiber
  group and hex-only DOF
  (instead of 18a's $\alpha_P n_{c,\text{shell}}$); the step's level-0 Schwarz temporaries
  fitted, not exact from `schwarz.nbytes`; the shell step, polish and count as $8n(am + c) +
  K_0$ (not the hex $8mK$), fitted on 25k–396k (count 25k–200k), valid from 25k DOF, the
  step and polish for $6 \le m \le 120$, the count for $4 \le b \le 32$; the `pc` terms
  fitted net of `STEP_FIT`, every point in the fit, constants rounded up with $c$ + 0.5;
  gates at the `lobpcg_memory` level: the lower bound and a large-$n$ upper gate on the
  9-block and polish terms set after the refit, no per-case tightness gate; $n$ > 791k at
  small $m$ extrapolated; storage
  tightness gated on totals; block $64^3$ build gate 1.35×; `pc_dev` = 0; the constants carry margins for
  levels beyond the measured sizes (10M extrapolation from 1.58M); the count constants are
  fitted in implementation; the deferred GPU checks of 18a stay.
- **Tests:** the memory test at $m$ = 6 / 25 / 120 only (the full grid in the benchmark
  JSON); the Jacobi wall-time test replaced by an iteration count on the curved panel;
  `test_m4_reference`'s "reference sharper" assertion replaced by the residual-bounded
  $\sin\theta$ check; the degree-3 fallback
  dropped as a default.
