# PR 18 plan: performance pass, part e: band follow-ups

Branch `pr-18e-band` from `main` (0948d20, PR 18d and the roadmap reorder merged). Roadmap: "PR 18e: Band follow-ups".

## Revisions

- Rev 1: first draft.
- Rev 2 (plan review round 1: none satisfied):
  - Levers: a lever is applied only when it helps (raises the best reach by ≥ 5% or clears the trigger); one fallback rule (revert all, case (a) and (b) alike); otf kept for the slices only when the stored slice capacity does not fit; one lever table read by the hint and the application; `info["levers"]` a dict; the keyword/function clash; the re-bound estimators; the halved-chunk limitation logged; the existing tests that pin the fallback get `levers=False`.
  - `err.result`: errors set in place inside si_band's loop (no `_finish` helpers), `info["complete_to"]`, references cleared before the compaction, P released; the forced-"lobpcg" result dropped.
  - Per-slice τ: $\hat\delta$ from $a_\text{eff} = a/2$ ($\lambda_1^{est}/2$ from 0), floored at 1, a pure function of the work item; `TAU_LOW` = `TAU_MIN` = $10^{-10}$ (a reviewer measured a $4\cdot10^{-11}$ floor at depth 10); the measured cost (+8–12%).
  - Performance: transposed `_form_v` / `_apply_km`, the merge rewrite and the `_chunks` host pad closed (gain under 1% of a slice, or an unmodelled rare peak); the gather skip only in `_span_count`; the transposed send shared by `_grow` and `_chunks`; the NumPy sketch kept.
  - `COUNT_STEP`: slope and intercept fitted at fixed $n$, the validity range stated, a deferred GPU check; one ≥ 1M case per family in the test.
  - 18f: the probe's caveats (mode accuracy, device memory) handed to its plan.
  - Docs: README:287, ROADMAP :95 / :846, eigensolvers :100 / :113 / :128 / :183 / :416 / :675 / :732 / :750 / :774–775 and the test sections, open_items :99 / :143, `memory.py` comments, memory files.

## Split (decision without user input)

The roadmap's 18e lists about fifteen items. A feasibility probe during planning (below) showed that the depth-1000 MINRES problem has a strong remedy that needs the LOBPCG modes a capacity stop now discards, so the deep-band items move to a new PR:

| PR | branch | contents |
|---|---|---|
| 18e | `pr-18e-band` | this plan: automatic memory levers, `err.result`, the per-slice τ ($\delta_\max$ enforcement), band-path performance items, the `COUNT_STEP` refit, and the closures below |
| 18f | `pr-18f-deep-bands` | new: the hybrid LOBPCG/slice junction (the locked modes kept as known pairs), MINRES deflation by the known modes below σ (the depth-1000 remedy), the σ-move retry policy, levers kept for a deeper capacity stop; its own plan after 18e merges |
| 18g | `pr-18g-calculix` | the old 18f (CalculiX comparison), unchanged |

Dependencies: 18f ← 18e; 18g ← 18a–18f; 19 ← 17, 18a–18f (as today with 18f added; 20+ unchanged). Order: 18e → 18f → 18g → 20 → 21 → 22 → 23a → 19 → 23b (the user's order of 2026-10-19 with 18f inserted; 18f runs and is tested on the Mac CPU, so "everything hardcore GPU comes last" still holds).

**The probe** (scratchpad `p18e/deflate_probe.py`, not committed; hex20 $16^3$, 56k DOF, AMG at σ, 6 columns, τ = $10^{-8}$, σ mid-gap above depth $d$; modes from `reference_modes`): MINRES on $K - \sigma M$ against MINRES on the deflated $\Pi^\top(K - \sigma M)\Pi$, $\Pi = I - WW^\top M$, preconditioner $\Pi P \Pi^\top$, $W$ the modes below σ:

| depth | none | nearest 20 | all below σ |
|---|---|---|---|
| 197 | 1500 its, 61.9 s | 910 its, 39.0 s | 140 its, 7.7 s |
| 503 | 4960 its, 203.7 s | 2940 its, 125.1 s | 220 its, 14.5 s |

All converged to τ (the deflated solution completed by $\sum_w w\,(w^\top b)/(\lambda_w - \sigma)$). Deflating every mode below σ makes the operator nearly definite on the complement. For 18f's plan (plan review round 1, performance): the reference modes have λ-scaled residuals up to $5.6\cdot10^{-7}$, and with partial deflation (nearest 50 / 100) the completed solution's true residual stalls near $4\cdot10^{-7}$; so 18f must probe with LOBPCG-accuracy modes, treat the residual coupling $W^\top A \Pi$ (a correction or a coupled solve), and budget the device memory of $W$ and $MW$ ($16\,n\,d$ B in fp64: the memory whose shortage caused the capacity stop).

## Goal (18e)

Close the PR 15d band follow-ups that do not need the junction, and the band-path performance items, without changing any solver result except where stated (levers change the options a band solve runs with; the per-slice τ tightens inner solves).

## Design (18e)

### 1. Automatic memory levers (`eigen/api.py`)

Today the ladder (`_ladder`) picks a chunk from the per-chunk $C_{max}$, and the levers are only named in warnings and errors (`levers()`, api.py:497). LOBPCG from 0 is 33–1260× cheaper than slices at every measured depth (`strategy_study.json`), so a lever that keeps a band on LOBPCG pays even at 2–4× per iteration.

- **Keyword** `levers=True` (bool; `False` keeps the options as given, today's behaviour). Validated up front; `levers=False` without `band` raises with the other band-only options ("strategy, memory_budget, slice_options and levers need band"). `solve_modes_memory` takes no `levers` (it estimates the options it is given; a test pins the `TypeError`).
- **One lever table** `LEVERS`: `(option, value, applicable(mesh, precond, tol, options))` for, in order:
  1. `precond_dtype="float32"` (GMG or AMG): LOBPCG and the polish only; with AMG the same PCG counts, V-cycle 0.82–0.91× the fp64 time; faster on voxel meshes (`eigensolvers.md:82`).
  2. `basis_dtype="float32"` when `tol >= 1e-6` (the fp32 basis floors locked residuals near $2^{-23} \approx 1.2\cdot10^{-7}$, `eigensolvers.md:323`; a 10× margin).
  3. `low_memory=True` (1.4–1.9× per iteration, `eigensolvers.md:324`).
  4. `operator_options={"mode": "otf"}` on an unstructured mesh without `ssh8` blocks (2–4× per apply, `operators.md:173`).
  5. `polish_options={"on": False}` (AMG): hint only, never applied (a stalled chunk then fails).
  The hint function (renamed `_lever_hint(c)`, so it does not shadow the keyword) and the application both read `LEVERS`; each entry counts only while its option is still at the default.
- **Trigger** (strategy "auto" or "lobpcg", with the options as given): (a) no chunk fits (`ch is None`), or (b) a known depth ($f_1 = 0$ with `n_modes`) that no fitting chunk reaches (no $c$ with $C_{max}(c) \ge$ `need[c]`).
- **Application:** in table order, a lever is tried on the current state: `need`, `reach` and the ladder are recomputed with it; it is kept if it clears the trigger or raises the best reach $\max_c C_{max}(c)$ (0 when none fits) by at least 5%, else skipped (a reviewer's probe: `low_memory` lowers $C_{max}$ with AMG on hex20 at 3 GB, 512 → 480; otf lowers it with fp32 AMG, 640 → 448). It stops when the trigger clears.
- **Fallback (one rule):** if the trigger still holds after the table, every lever is reverted and today's path runs with the options as given (case (a): slices, "memory"; case (b): today's "no chunk reaches the need" chunk). A deeper capacity stop is worth nothing in 18e (the slices solve the whole band again); keeping levers for it is 18f's (with the junction). One exception, for robustness: on the slice paths ("memory", forced "shift_invert", after a capacity stop) when `_slice_capacity` returns None with the stored operator and a slice fits with otf, the operator is built in otf mode (`info["levers"]` lists it); the "smallest slice does not fit" error otherwise.
- **Residual risk (logged):** levers kept for a known depth stay on if `n_modes` was an underestimate and a capacity stop follows (otf then slows the slices). **Known limitation (logged):** the ladder is evaluated per lever state, so a cheaper (larger chunk, lever) state reaching the need is not preferred over a halved chunk without levers (chunk 5 costs 2.7× chunk 40 per mode); a per-state cost choice is not made.
- **Order of work in `solve_modes`:** the budget, `free`, the rigid count $k$ (`rigid_modes(mesh, free).shape[1]`, built once and reused for R), $n$ = `3 * len(mesh.points)`, the ladder and the levers come before the operator build; the operator is built once with the final `operator_options`. The estimator closures (`models`, `c_max`, `op_memory`, `pc`, `pc64`, `probe_budget`, `_slice_capacity`, `_count_block`, `_lever_hint`, `info["c_max"]`) are rebuilt from the final options (they bind their defaults today, api.py:485–495).
- **`info["levers"]`:** a dict of the applied options, e.g. `{"precond_dtype": "float32", "low_memory": True}`, `{}` when none (every band path).

### 2. `err.result` (`eigen/shift_invert.py`, `eigen/api.py`)

A capacity error today discards every pair found (open item "A capacity error discards the band": 37 slices, 25 min, then the raise).

- **si_band:** the four raises in the loop (`max_slices` `RuntimeError` :861, the deflated-modes `ValueError` :878, `_cap_error` :925 and :929) become `err = <the exception>; converged, reason = False, str(err); break` (the :878 site is in the inner loop: one `if err: break` after it), with P and the slice's V released first. The existing post-loop merge and assembly run unchanged on the stored pairs; then `if err is not None: err.result = (lam_f, V_f, info); raise err` with no other local holding `V_f`. Other exceptions (a refill breakdown, "no converged neighbour") propagate as today.
- **`info["complete_to"]`** (λ): every stored pair below it is leaf-verified (slices run lowest first): the smallest lower edge among the failing slice and the work items left; `hi` on a normal exit. solve_modes adds `info["complete_to_hz"]`.
- **solve_modes:** `except Exception as e:` with `getattr(e, "result", None)`: unpack `lam, V, info`, set `e.result = None` and `traceback.clear_frames(e.__traceback__)` (so the in-place compaction's `V.resize` sees no other reference), run the existing tail (filter, compaction, info keys of a normal exit), then `e.result = (lam, V, info)` and re-raise. The forced-"lobpcg" errors get no result (the user chose a strategy that raises; "auto" returns the modes through the slices).
- **Decision (logged):** raise with the result attached rather than return with `converged=False`: a capacity error ends a band that cannot continue at this capacity, and raising keeps it loud; LOBPCG non-convergence keeps returning.

### 3. Per-slice τ: $\delta_\max$ enforcement (`eigen/shift_invert.py`)

A pair is accepted at $r_i \le \max(\mathrm{tol}, 10\,\tau_\text{eff}\,\delta_i)$, $\delta_i = |\lambda_i - \sigma|/|\lambda_i|$, and $r_{acc}$ is the largest threshold over the inside pairs and the slice's neighbours. With σ the midpoint of $[a, e]$, the lowest pair or the below neighbour has $\delta$ unbounded as $a \to 0$: slices from 0 report $r_{acc}$ above tol (open item "δ_max ≤ 1 is not enforced"), and a large $r_{acc}$ widens the chain gap $100\,r_{acc}$ that edges need (`eigensolvers.md:175`). The roadmap's remedy (cap $\delta_\max \le 1$ when choosing slices) splits slices from 0 into narrow pieces at full slice cost each; tightening the inner tolerance is cheaper.

- `si_band` gains `lam1=None` (an estimate of $\lambda_1$). Each work item $[a, e]$ (slices, widen-and-reruns of the same interval, span reruns: all by the same formula, no inherited state), unless the caller passed `tau`:

$$a_\text{eff} = \tfrac12 \begin{cases} a & a > 0 \\ \lambda_1^{est} & a = 0 \end{cases}, \qquad \hat\delta = \max\Bigl(\frac{\sigma - a_\text{eff}}{a_\text{eff}},\ 1\Bigr), \qquad \tau_s = \max\Bigl(\tau_\text{low},\ \min\bigl(\tau_0,\ \mathrm{tol}/(10\,\hat\delta)\bigr)\Bigr),$$

  $\tau_0 = 10^{-8}$, $\tau_\text{low}$ = `TAU_MIN` = $10^{-10}$ (the existing floor constant; a reviewer measured MINRES stagnating at $4\cdot10^{-11}$ with τ = $10^{-12}$ at depth 10). The factor ½ covers a below neighbour down to $a/2$ (or $\lambda_1/2$) and an upper-bound estimate within 2×; the floor 1 covers the above neighbour ($\delta < 1$). Without `lam1` a slice from 0 keeps $\tau_0$. At $\tau_\text{low}$, $r_{acc}$ can exceed tol for $\hat\delta > \mathrm{tol}/10^{-9}$ (1000 at tol $10^{-6}$; reported as today). The counts' τ (`count_options`) is unchanged.
- `solve_modes` passes `lam1=info["mu1_ub"]`. On slices from $f_1 = 0$ from the start (none today) the existing pre-estimate runs for this as for $f_1 > 0$ (`_precond_sigma`, Jacobi `sigma_estimate`; its hierarchy freed before the slices). `sigma_estimate` stops at a 5% decrease over 10 iterations, so its bound can be looser than 2× on hard meshes; the test meshes measure it.
- `info["slices"][i]["tau"]` records each run's τ.
- **Cost** (reviewer's measurement, hex20 $16^3$, AMG): τ $10^{-8} \to 10^{-10}$ adds 8–12% MINRES iterations at depths 10–100. Lanczos steps can also grow (low pairs must reach tol, not $10\,\tau\delta$); test 7 records steps, basis and full-capacity counts.

### 4. Band-path performance (`eigen/lobpcg.py`, `eigen/shift_invert.py`, `eigen/counting.py`)

- **Transposed send** (`_grow` and `counting._chunks`): an F-order host chunk $V[:, i:i+c]$ is sent as `jax.device_put(chunk.T)` (a C-order $(c, n)$ view, no host transposing copy) and transposed inside the jitted consumer (`_insert_t` for the refill; `_chunk_step` for the projections). `_grow`: 2.4× faster on CPU (PR 15d review); `_chunks` runs on every projection pass of the counts, the same copy. Device arrays are passed as today.
- **No gather on a skipped span count:** `_span_count` predicts with a zero-cost placeholder `np.broadcast_to(0.0, (n, len(inb)))` (the predict path checks shapes only) and calls `cols(inb)` (a host copy of up to one slice's V, about 5 GB at 1M DOF) only when `_decide` says go (calibration or count), once. `si_checked`'s `W_in` is already a view under si_band (`_window` is contiguous in the sorted store); `count` and `band_count` are unchanged.
- **NumPy sketch:** `_check_ortho` draws $z$ with `np.random.default_rng((seed % 2**64, 2**32 - 1))` instead of `jax.random.normal` of shape $(k, 3)$ (one compile per W width, 60–72 ms each; shorter code). The values change, the check's tolerance does not.
- **Preconditioner reuse on a whole requeue:** si_band keeps the previous slice's P when the next work item has the same σ (a slice requeued whole); the release line `P = V = inf = None` then keeps P.

### 5. `COUNT_STEP` refit (`memory.py`)

`count_memory`'s unstructured step $8n(a\,b + c) + w$, $(a, c) = (10.3, 3.0)$, runs low above its fit range (measured/model 1.015 at 0.54M, 1.035 at 0.97M, 1.045 at 1.5M `ssh8` DOF with Jacobi; hex20 $36^3$ 1.023; open_items :203).

- Measure `memory_analysis` of `_recurrence` (`_recurrence_temps` of `tests/test_counting.py`; `shell_solve_study.count` for `ssh8`) on `ssh8` plates near 0.8M, 1.2M and 1.6M DOF and hex20 $36^3$ (0.6M) and $44^3$ (1.08M), $b$ = 16 and 32 at each $n$ (slope and intercept apart), Jacobi and multigrid; refit $(a, c)$ as the 16b fit (upper envelope, $c \ge 0$, rounded up by 0.5) over the old and new points. Compile-only plus one AMG setup per mesh: about 10 s and 4.5 GB each at 1.08M hex20 (reviewer probe), well inside the 1 h rule. If the excess grows with $n$ beyond what $(a, c)$ absorbs, the step gains an $n$-dependent term instead, fitted the same way.
- Valid to 1.6M DOF unstructured (stated in the `memory.py` comment and `eigensolvers.md`); a `memory_analysis` check at 10M on the GPU joins the Deferred GPU checks.
- The refit raises the count estimate a few percent: the tests that pin a probe block or the step's tightness (`test_step_memory`'s 0.5 / 0.6 × model, `test_shell_count`'s 1.3×, `test_band.py:476/535`, `test_strategy.py:1007/1278/2126`) are rerun and adjusted by their own formulas, each change named in the commit.

### 6. Closed without change (logged in `docs/open_items.md` with the reason)

- **Up-front spectrum bound $\hat\lambda_\max$:** an edge above the spectrum is only reachable on meshes small enough that the band covers the top of the spectrum, where the late error costs seconds; the cheap rigorous bound (max over elements of $\lambda_\max(K_e, M_e)$) is typically 2–4× above $\lambda_\max$, too loose to close a side. Stays a late error, now with `err.result`.
- **Halving before a capacity error:** kept (robustness over the error's cost); with `err.result` the pairs found are no longer lost.
- **$f_1 > 0$ in `solve_modes_memory`:** an interior band's memory is set by `solve_modes` itself (the ladder and the slice capacity are sized to `memory_budget`); an estimate would need the depth, unknown before the solve. The `ValueError` names `band=(0, f2)` with `n_modes` = the expected depth as the LOBPCG estimate.
- **`np.memmap` for the locked modes:** the fit test bounds the host V by the host budget, so LOBPCG never locks past RAM; at 10M DOF and 1000 modes the fp64 host V is 80 GB, below a GPU node's RAM. The ROADMAP line is removed.
- **Transposed `_form_v` / `_apply_km`:** 0.33 s per call at 1M DOF against slices of minutes (`_form_v` once per run; `_apply_km` only in merges).
- **Merge copies ($Q$, $MQ$ only):** merges are rare (windows held by two runs; the collapse case only would gain), the merge peak is not budgeted by any estimator, and the rewrite costs one K apply per column. Revisit if a measured case shows the merge sets the process peak.
- **`_chunks` padding compiles:** 6–15 ms per new width against counts of seconds to minutes; a host pad would add an unmodelled $(n, 64)$ host block.
- **σ between the band-pass edges, span counts at small capacities, per-step residuals of inside candidates, `t_ref` including compilation:** `verify=None` prices each count per leaf, a capacity below 96 warns, `info["time"]["ritz"]` records the residual cost (no measured case where it matters), and compile time in `t_ref` only makes `verify=None` slightly more permissive.
- **Verification slices, σ toward the unconverged neighbour, the top-remainder neighbour, the σ-move retry policy, the stagnation remedies:** 18f.

## Tests (18e)

References: SciPy `eigsh` through `tests/test_strategy.py`'s helpers (`prob`, `check`, `run`, `fake_builds`, `need`, `lob_est`), `tests/test_shift_invert.py`'s references, and the old code paths.

**Existing tests that pin the slice fallback** get `levers=False` and keep their purpose: `test_memory_bracket`, `test_memory` (3 cases), `test_memory_block_sentinel`, `test_levers`, `test_given_chunk_no_fit`, `test_memory_no_lobpcg_build`, `test_forced_lobpcg_no_fit` (`tests/test_strategy.py`). `test_pre_estimate_zero` and `test_memory_no_lobpcg_build` (f1 = 0 runs no pre-estimate / builds nothing before si_band) are rewritten: f1 = 0 from the start now runs the pre-estimate once, for `lam1`.

1. **Levers, known depth, fake builds:** the budget is found on the estimators in $[\text{need(precond + low)}, \text{need(precond)})$ on "short" (the precond lever may gain under 5% there and is then skipped; the expected dict is computed by the same rule from the estimators): `info["levers"]` as expected, `reason` "fits", `check` against `eigsh`; the same budget with `levers=False`: today's chunk, `{}`.
2. **Levers, real builds** (no fake builds, "short", `tol=1e-6`, f1 = 0 with `n_modes`): a budget where the trigger clears only with otf: `info["levers"]` includes `operator_options`, a spy on `unstructured` sees one build, in otf mode; the band matches `eigsh` (`check`). A budget where no lever state clears the trigger: `{}`, one stored-mode build, today's chunk, the result against `eigsh`.
3. **Lever skipped when it does not help:** a state where `low_memory` lowers or keeps $C_{max}$ (AMG on hex20, the reviewer's case): not in `info["levers"]`.
4. **Levers, no fit:** a budget below every lever state, `slice_options={"capacity": 96}`: "memory", `{}`, the slice operator stored; the warning names the five hints on "short". The otf exception: a budget where `_slice_capacity` is None with the stored model and not with otf (fake builds): otf built, listed.
5. **Applicability:** `low_memory=True` given: not listed; `tol=1e-7`: no basis lever, `tol=1e-6`: basis lever applied; Jacobi mesh ("tiny"): no precond lever; `ssh8` (`T.plate(8, 50)` with stubbed `_solve` and si_band): no otf lever; forced "lobpcg" with no fit: levers tried, reverted, the error names all hints; `info["levers"]` present on every band path. Non-bool `levers` raises; `levers=False` without `band` raises; `solve_modes_memory(..., levers=True)` raises `TypeError`.
6. **`err.result` (si_band, fakes of `test_shift_invert.py`):**
   - "nothing converged" after a stored slice (`edges=[c]`: slice 1 converges, slice 2 raises);
   - "no progress" (`:1430` case);
   - the deflated-modes `ValueError` (`:1852` case, one slice stored);
   - `max_slices` with the full-slice fake (modes 7–11 stored);
   - one case with a merged window before the raise.
   Each: `e.result`'s λ equals the stored set, `info["converged"]` False, `reason` the message, `complete_to` as defined, `count["estimates"]` only the leaves run; V's orthogonality by the normal-exit tests' criterion.
7. **`err.result` (solve_modes):** `test_edge_above_spectrum`'s band ("tiny", "no progress") with `edges` so lower slices store pairs first, including pairs below `lo`: `e.result` is filtered to the band (out-of-band pairs removed), its key set equals a normal exit's, the modes match `eigsh`, no "returned modes copied" warning (in-place compaction), `complete_to_hz` present.
8. **Per-slice τ:**
   - `si_band` from 0 on "short" ($\lambda_{60}/\lambda_1 \approx 78$; one slice, `lam1` from `eigsh`): `slices[0]["tau"]` equals the formula, `r_acc <= tol`, all pairs against `eigsh`; with `slice_options={"tau": 1e-8}`, `r_acc > tol` (today's behaviour, pinning the effect); steps, basis and capacity-full recorded for both.
   - An interior slice with $\hat\delta = 1$ keeps $10^{-8}$ at tol $10^{-6}$; the clamp at `TAU_MIN` with `tol=1e-10` (formula and a run's `tau`, not `tau_eff`).
   - With `fake_band`: a widen-and-rerun and a span rerun (`test_band_span_rerun`'s setup) receive the formula value of their own interval; no `lam1` keeps $\tau_0$ on a slice from 0.
   - `solve_modes` passes `lam1 == info["mu1_ub"]` on the capacity-stop, "memory" ($f_1 > 0$), from-0 and Jacobi paths.
9. **Transposed send:** `_grow` returns the same B as the old path exactly (fp64 and fp32 bases, V with fewer than `up(count, c)` columns: a partial last chunk); `_project` over host W of widths 5, 37, 64, 100 equals the old path to $10^{-14}$ relative.
10. **No gather on skips:** monkeypatched `_decide_slice` wrapping the gather: `verify=False` and a `verify=None` skip by the bound gather nothing; a `verify=None` calibration plus count gathers once; `info["count"]` equals a stored dict of the same run's degree, gap and reason.
11. **Sketch:** `_check_ortho` accepts M-orthonormal W (host and device, fp64 and fp32) and rejects a perturbed one, for seeds 0, −1, $2^{40}$.
12. **Preconditioner reuse:** the requeue-whole case (`test_shift_invert.py:1435`) with a counting `make`: one call per distinct consecutive σ, a rebuild on the next σ, and the previous P dead (weakref) before the rebuild.
13. **`COUNT_STEP`** (slow, `peak`): `count_memory`'s step (without its 2 b blocks) ≥ the measured recurrence and ≤ 1.2× of it at $b$ = 16 and 32, multigrid, on one ≥ 1M `ssh8` and one ≥ 1M hex20 case (extends `test_shell_count` and `test_step_memory`).
14. All other band, strategy and shift-invert tests pass unchanged except where a lever, the τ rule or the refit changes a pinned value (each named in its commit).

## Docs (18e)

- `docs/eigensolvers.md`:
  - options table: `levers` (new row); `slice_options` `tau` (`:100`) and si_slice's `tau` row (`:732`) with the per-slice default.
  - the ladder paragraph (`:154`, "never set automatically (PR 18e)") → the lever table, trigger, 5% rule, fallback, otf exception, limitation; Interior bands and strategy (`:140–141`); the `info` keys paragraph (`:183`, `levers`, `complete_to`).
  - the $\mu_1$ rule paragraph (`:175`) → the per-slice τ and `lam1`.
  - Errors (`:181`): the `levers` argument errors, `err.result` with `complete_to`, the spectrum-bound closure; si_band's Errors and Termination bullets (`:774–775`); the si_band paragraph (`:750`: `lam1`, `slices[i]["tau"]`); Edge above the spectrum (`:777`) reworded as a decision.
  - `solve_modes_memory` (`:128`, `:675`): the $f_1 > 0$ closure; the count memory model (`:665–675`) with the refit and its range.
  - the refill (`:113`, `:416`): the host transient is unchanged (one chunk), stated.
  - test sections (`:960–987`: Slicing a band `:967`, test_strategy `:977`, `_solve` refill `:982`, Errors and options `:985`): levers, `err.result`, per-slice τ, transposed send, gather skip, sketch, P reuse, `COUNT_STEP`.
- Docstrings: `solve_modes` (`levers`, `info["levers"]`, `complete_to`), `si_band` (`lam1`, the per-slice τ, `err.result`, `slices[i]["tau"]`, `complete_to`), `_check_ortho`, `_span_count`, `count_memory`; the `COUNT_STEP` comment (`memory.py:569–574`, the fit range).
- `README.md`: `:5` status (18e, the split into 18e–18g); `:287` the interior-band paragraph (levers first, `levers=False`, `err.result`).
- `docs/ROADMAP.md`: `:82` (the δ rule), `:95` (remedies → 18f), `:97` (late errors only, closed), `:177` (np.memmap line removed), `:184` (18e's refit), the 18e section rewritten to its contents, a new 18f section, 18f → 18g, the dependency rows (`:844–846`, 19 ← 18a–18f), Deferred GPU checks (the 10M `COUNT_STEP` check), `:18` status, `:627` the split, `:854` the order.
- `docs/open_items.md`: `:99` (Memory constants: the refit range), `:143` (the $\lambda_{hi}$ bound), the PR 15d decisions (`:155–182`) and the PR 15c sections (`:317–347`) item by item (done in 18e, closed with the reason, or → 18f; `:328`'s merge figure corrected when closed), `:203` and `:222` the refit outcome, `:362` the order, and a "PR 18e decisions" section.
- Memory: MEMORY.md (the autonomy line, the scale line's order, the CalculiX line), `project-calculix-benchmark.md` (18f → 18g), `project-scale-questions.md` (the order).

## Risks and decisions without user input (logged in `docs/open_items.md`)

- The split (18f deep bands, 18g CalculiX) and the order.
- Levers on by default, applied only when they help and when a known depth is not reached or nothing fits; otherwise reverted. Polish never off automatically. The residual risk (an `n_modes` underestimate keeps the levers into slices) and the halved-chunk limitation.
- `err.result` on a raise rather than a return; forced "lobpcg" gets none.
- The per-slice τ instead of the roadmap's $\delta_\max \le 1$ slice cap: the same goal ($r_{acc} \le$ tol) at +8–12% MINRES iterations on the affected slices, with `TAU_MIN` as the floor.
- The closures in section 6.
- The NumPy sketch in `_check_ortho`.
