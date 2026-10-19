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
- Rev 3 (plan review round 2: none satisfied):
  - Levers: the greedy pass and its 5% rule replaced by an enumeration of the applicable lever subsets (≤ 16) with a static cost (a performance reviewer's replay of `strategy_study.json`: the greedy pass reverted to slices where {precond, basis, low} reached the need, and kept `low_memory` where otf alone sufficed; reach is not monotone in the levers); the trigger includes "only a halved chunk reaches the need" (the halved-chunk limitation gone); a pure `_choose` helper tested with crafted reaches; the otf exception on the slice paths moved to 18f; polish off out of the table; the closures' frozen defaults removed instead of a rebuild step.
  - Per-slice τ: `lam1` dropped; closed form $\hat\delta = e/a$; a slice from 0 runs at `TAU_MIN`; the σ-move test keeps the untightened τ; the cost's scope (every slice at tol ≤ $10^{-8}$).
  - `err.result`: `complete_to` defined for every exit; the solve_modes tail runs once for both exits.
  - Performance: `_chunks` yields one orientation, the step written $Y - ((W_t\,MY)^\top W_t)^\top$ (3.5× on CPU; the naive form gains 3%); the `_grow` figure corrected; the lazy gather in `_span_count`.
  - `COUNT_STEP`: fitted on Jacobi points, multigrid as a check of the sum.
  - Tests and docs per the reviewers' lists.

## Split (decision without user input)

The roadmap's 18e lists about fifteen items. A feasibility probe during planning (below) showed that the depth-1000 MINRES problem has a strong remedy that needs the LOBPCG modes a capacity stop now discards, so the deep-band items move to a new PR:

| PR | branch | contents |
|---|---|---|
| 18e | `pr-18e-band` | this plan: automatic memory levers, `err.result`, the per-slice τ ($\delta_\max$ enforcement), band-path performance items, the `COUNT_STEP` refit, and the closures below |
| 18f | `pr-18f-deep-bands` | new: the hybrid LOBPCG/slice junction (the locked modes kept as known pairs), MINRES deflation by the known modes below σ (the depth-1000 remedy), levers after a capacity stop (a LOBPCG rerun with levers when they raise $C_{max}$ well past the stop; otf for the slices when the stored slice operator does not fit, the stored one back when it does), and the items of the last bullet of section 6; its own plan after 18e merges |
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

Today the ladder (`_ladder`) picks a chunk from the per-chunk $C_{max}$, and the levers are only named in warnings and errors (`levers()`, api.py:497). LOBPCG from 0 is 33–1260× cheaper than slices at every measured depth (`strategy_study.json`), so a lever state that keeps a band on LOBPCG pays even at 2–4× per iteration.

- **Keyword** `levers=True` (bool; `False` pins the options as given, today's behaviour). Validated up front; `levers=False` without `band` raises with the other band-only options ("strategy, memory_budget, slice_options and levers need band"). `solve_modes_memory` takes no `levers`.
- **Lever table** `LEVERS`, rows `(option, value, cost, applicable)`:

  | option | value | cost factor | applicable |
  |---|---|---|---|
  | `precond_dtype` | `"float32"` | 0.9 | GMG or AMG (LOBPCG and polish; V-cycle 0.82–0.91× the fp64 time, `eigensolvers.md:82`) |
  | `basis_dtype` | `"float32"` | 1.0 | `tol >= 1e-6` (the fp32 basis floors locked residuals near $1.2\cdot10^{-7}$, `eigensolvers.md:323`) |
  | `low_memory` | `True` | 1.6 | always (1.4–1.9× per iteration, `eigensolvers.md:324`) |
  | `operator_options["mode"]` | `"otf"` | 3.0 | an unstructured mesh without `ssh8` blocks (2–4× per apply, `operators.md:173`) |

  A row counts only while its option is at the default (`precond_dtype`, `basis_dtype` and `low_memory` given explicitly at their default cannot be told apart and count; `operator_options` with any `"mode"` given counts as given). `levers=False` pins everything. The hint (the inner `levers()` renamed `_lever_hint(c)`, so it does not shadow the keyword) reads the same table and appends the polish-off line (AMG) as today; polish off is never applied (a stalled chunk then fails).
- **Trigger** (strategy "auto" or "lobpcg", the options as given): (a) no chunk fits, or (b) a known depth ($f_1 = 0$ with `n_modes`) that the top chunk does not reach (today the ladder then halves or takes the "no chunk reaches the need" chunk). Otherwise nothing changes.
- **Choice** (`_choose(reach, need)`, a pure function): every subset $S$ of the applicable rows (at most 16) is evaluated: per chunk $c$ of the ladder, $C_{max}(S, c)$ from the estimators with $S$ applied (all 16 × 4 states cost 0.11 s on hex20 $44^3$, a reviewer's measurement). Cost of $(S, c)$ = chunk factor (1.0 / 1.7 / 2.2 / 2.7 at chunks 40 / 20 / 10 / 5 and their halvings, `eigensolvers.md:154`) × $\prod_{r \in S}$ cost$_r$.
  - (b): among the $(S, c)$ with $C_{max} \ge$ need, the cheapest; on a tie fewer levers. None reaches: no levers, today's ladder.
  - (a): among the $S$ with some chunk fitting, the largest best reach $\max_c C_{max}(S, c)$ (the depth is unknown; a capacity stop costs 33× or more), on a tie the cheapest; within $S$ today's ladder rule picks the chunk. None fits: no levers, today's "memory" path.
  The reach is not monotone in the levers (`low_memory` lowers $C_{max}$ with fp32 levers on hex20 $44^3$, 2624 → 2592), so no greedy order is used.
- **Residual risks (logged):** in case (a) or after an `n_modes` underestimate, a capacity stop can follow with otf kept for the slices (2–4× per apply where the stored operator might fit; the remedy, back to stored, is 18f's). Bands of unknown depth with a fitting chunk ($f_1 > 0$, or $f_1 = 0$ without `n_modes`) get no levers (a deeper capacity stop is worth nothing until 18f's junction; the LOBPCG rerun with levers is 18f's). The cost factors are static (measured on the Mac CPU; GPU factors are a deferred check).
- **Order in `solve_modes`:** the budget, `free`, the rigid count $k$ (`rigid_modes(mesh, free)`, built once and reused for R), $n$ = `3 * len(mesh.points)`, the ladder and the choice come before the operator build; the operator is built once with the final options. The estimator closures (`models`, `c_max`, `op_memory`, `pc`, `pc64`, `probe_budget`, `_slice_capacity`, `_count_block`, `_lever_hint`) lose their frozen default arguments (`low=low_memory`, `dtype=basis_dtype`, `ops=op_opts`, `po=po_opts`, api.py:485–495) and take the state as arguments where `_choose` evaluates subsets; `op_memory`, `pc`, `pc64` are reassigned once after the choice. `info["c_max"]` is the chosen state's.
- **`info["levers"]`:** a dict of the applied options, e.g. `{"precond_dtype": "float32", "low_memory": True}`, `{}` when none (every band path).
- **Errors:** the "smallest slice does not fit" error lists otf among its hints (no automatic otf on the slice paths in 18e). Forced "lobpcg" with no state fitting: the error names all hints, as today.

### 2. `err.result` (`eigen/shift_invert.py`, `eigen/api.py`)

A capacity error today discards every pair found (open item "A capacity error discards the band": 37 slices, 25 min, then the raise).

- **si_band:** the four raises in the loop (`max_slices` `RuntimeError` :861, the deflated-modes `ValueError` :878, `_cap_error` :925 and :929) become `err = <the exception>; converged, reason = False, str(err); break` (the :878 site is in the inner loop: one `if err: break` after it), with P and the slice's V released first. The existing post-loop merge and assembly run unchanged on the stored pairs; then `if err is not None: err.result = (lam_f, V_f, info); del V_f; raise err`. Other exceptions (a refill breakdown, "no converged neighbour") propagate as today.
- **`info["complete_to"]`** (λ): every stored pair below it is leaf-verified (slices run lowest first). On a raise, a "slice not converged" break, or an `rr_failed` exit: the smallest lower edge among the failing slice and the work items left; on a converged exit `hi`. solve_modes adds `info["complete_to_hz"]`.
- **solve_modes:** `except (RuntimeError, ValueError) as e:` re-raises at once without `e.result`; else unpacks `lam, V, info = e.result`, sets `e.result = None`, `traceback.clear_frames(e.__traceback__)` (a reviewer verified that the in-place `V.resize` then succeeds), keeps `err = e`, and falls through the existing tail (filter, compaction, info keys); at the end `if err: err.result = (lam, V, info); raise err`. Forced-"lobpcg" errors get no result (the user chose a strategy that raises; "auto" returns the modes through the slices).
- **Decision (logged):** raise with the result attached rather than return with `converged=False`: a capacity error ends a band that cannot continue at this capacity, and raising keeps it loud; LOBPCG non-convergence keeps returning.

### 3. Per-slice τ: $\delta_\max$ enforcement (`eigen/shift_invert.py`)

A pair is accepted at $r_i \le \max(\mathrm{tol}, 10\,\tau_\text{eff}\,\delta_i)$, $\delta_i = |\lambda_i - \sigma|/|\lambda_i|$, and $r_{acc}$ is the largest threshold over the inside pairs and the slice's neighbours. With σ the midpoint of $[a, e]$, the lowest pair or the below neighbour has $\delta$ unbounded as $a \to 0$: slices from 0 report $r_{acc}$ above tol (open item "δ_max ≤ 1 is not enforced"), and a large $r_{acc}$ widens the chain gap $100\,r_{acc}$ that edges need (`eigensolvers.md:175`). The roadmap's remedy (cap $\delta_\max \le 1$ when choosing slices) splits slices from 0 into narrow pieces at full slice cost each; tightening the inner tolerance is cheaper.

- Each work item $[a, e]$ (slices, widen-and-reruns, span reruns: the same formula on its own interval, no inherited state), unless the caller passed `tau`:

$$\tau_s = \begin{cases} \max\bigl(\tau_\min,\ \min(\tau_0,\ \mathrm{tol}\,a/(10\,e))\bigr) & a > 0 \\ \tau_\min & a = 0 \end{cases}$$

  $\tau_0 = 10^{-8}$, $\tau_\min$ = `TAU_MIN` = $10^{-10}$ (a reviewer measured MINRES stagnating at $4\cdot10^{-11}$ with τ = $10^{-12}$ at depth 10). This is $\mathrm{tol}/(10\,\hat\delta)$ with $\hat\delta = (\sigma - a/2)/(a/2) = e/a$: the bound for a below neighbour down to $a/2$ ($\ge 1$ also covers the above neighbour). A slice from 0 has no such bound and runs at $\tau_\min$; $r_{acc}$ can then exceed tol when $\lambda_e/\lambda_1 > \mathrm{tol}/10^{-9}$ (1000 at tol $10^{-6}$; reported as today). A below neighbour under $a/2$ (a sparse spectrum below an interior band) can also give $r_{acc} >$ tol: $r_{acc} \le$ tol is the goal, not a guarantee (logged).
- The σ-move test keeps the untightened τ: `si_slice` gains `stag_tau` (si_band passes $\tau_0$, or the caller's `tau`), so `stag = STAGNATION * max(stag_tau, TAU_MIN)` (shift_invert.py:329) is unchanged; si_band passes `tau=τ_s` without forwarding `tau` again in `**slice_options`.
- `info["slices"][i]["tau"]` records each run's τ.
- **Cost** (reviewer's measurements, hex20 $12^3$ / $16^3$, AMG): τ $10^{-8} \to 10^{-10}$ adds 8–12% MINRES iterations; one slice from 0 to mode 60 at tol $10^{-6}$: 78.0 s → 84.1 s, $r_{acc}$ $3.8\cdot10^{-6} \to 1.0\cdot10^{-6}$, Lanczos steps and basis unchanged. **Scope:** at tol $10^{-6}$ only slices with $e/a > 10$ (in practice from 0 or the first few) change; at tol $\le 10^{-8}$ every slice drops to $\tau \le$ tol/10 (+8% on every slice; `test_strategy`'s TOL is $10^{-8}$).

### 4. Band-path performance (`eigen/lobpcg.py`, `eigen/shift_invert.py`, `eigen/counting.py`)

- **Transposed send.** `_grow`: an F-order host chunk $V[:, i:i+c]$ is sent as `jax.device_put(chunk.T)` (a C-order $(c, n)$ view; on CPU the send aliases V, no copy) and transposed inside the jitted `_insert_t` (1.5–2.4× on CPU, machine-dependent). `counting._chunks` always yields $(c, n)$ chunks $W_t$ (host chunks by the view's send, device chunks transposed inside the jit; padding along axis 0), and its consumers use the transposed forms: `_chunk_step` $Y - ((W_t\,MY)^\top W_t)^\top$ (19.7 ms against 54 ms for the naive $Y - W_t^\top (W_t\,MY)$ at $n$ = 1M, $c$ = 64, $b$ = 16, a reviewer's measurement: the counts' projection pass 3.5× faster) and `_check_ortho`'s two products likewise. A C-order host W (si_checked's restart from `np.hstack`) gains nothing and loses nothing.
- **Lazy gather in `_span_count`:** `cols(inb)` (a host copy of up to one slice's V, about 5 GB at 1M DOF) becomes a memoized getter: the predict path gets the zero-stride placeholder `np.broadcast_to(0.0, (n, len(inb)))` (it reads shapes only); the first calibration (the `cal` closure, shift_invert.py:476, called from `counting._decide` before `go` is known) or count gathers once and the count reuses it. `count` and `band_count` are unchanged.
- **NumPy sketch:** `_check_ortho` draws $z$ with `np.random.default_rng((seed % 2**64, 2**32 - 1))` instead of `jax.random.normal` (one compile per W width, 60–72 ms each). The values change, the tolerance does not.
- **Preconditioner reuse on a whole requeue:** si_band keeps the previous slice's P when the next work item has the same σ (one AMG setup, about 4 s at 1M DOF; the one-alive rule kept): the release line `P = V = inf = None` keeps P then.

### 5. `COUNT_STEP` refit (`memory.py`)

`count_memory`'s unstructured step $8n(a\,b + c) + w$, $(a, c) = (10.3, 3.0)$, is the Jacobi step (multigrid adds `pc["count"]`); it runs low above its fit range (measured/model 1.015 at 0.54M, 1.035 at 0.97M, 1.045 at 1.5M `ssh8` DOF with Jacobi; hex20 $36^3$ 1.023; open_items :203).

- Measure `memory_analysis` of `_recurrence` (`_recurrence_temps` of `tests/test_counting.py`; `shell_solve_study.count` for `ssh8`) on `ssh8` plates near 0.8M, 1.2M and 1.6M DOF and hex20 $36^3$ (0.6M) and $44^3$ (1.08M), $b$ = 16 and 32 at each $n$. Fit $(a, c)$ on the Jacobi points (old and new; upper envelope, $c \ge 0$, rounded up by 0.5, as 16b); the multigrid points check the sum with `pc["count"]`. About 20 compile-only runs plus one AMG setup per mesh (about 10 s and 4.5 GB each at 1.08M hex20), well inside the 1 h rule. If the excess grows with $n$ beyond what $(a, c)$ absorbs, the step gains an $n$-dependent term, fitted the same way.
- Valid to 1.6M DOF unstructured (the `memory.py` comment and `eigensolvers.md`); a `memory_analysis` check at 10M on the GPU joins the Deferred GPU checks.
- The refit raises the count estimate a few percent: the tests that pin a probe block or the step's tightness (`test_step_memory`'s 0.5 / 0.6 × model, `test_shell_count`'s 1.3×, `test_band.py:476/535`, `test_strategy.py:1007/1278/2126`) are rerun and adjusted by their own formulas, each change named in the commit.

### 6. Closed without change (logged in `docs/open_items.md` with the reason)

- **Up-front spectrum bound $\hat\lambda_\max$:** an edge above the spectrum is only reachable on meshes small enough that the band covers the top of the spectrum, where the late error costs seconds; the cheap rigorous bound (max over elements of $\lambda_\max(K_e, M_e)$) is typically 2–4× above $\lambda_\max$, too loose to close a side. Stays a late error, now with `err.result`.
- **Halving before a capacity error:** kept (robustness over the error's cost); with `err.result` the pairs found are no longer lost.
- **$f_1 > 0$ in `solve_modes_memory`:** an interior band's memory is set by `solve_modes` itself (the ladder and the slice capacity are sized to `memory_budget`); an estimate would need the depth, unknown before the solve. The `ValueError` names `band=(0, f2)` with `n_modes` = the expected depth as the LOBPCG estimate.
- **`np.memmap` for the locked modes:** the fit test bounds the host V by the host budget, so LOBPCG never locks past RAM; at 10M DOF and 1000 modes the fp64 host V is 80 GB, below a GPU node's RAM. The ROADMAP line is removed.
- **Transposed `_form_v` / `_apply_km`:** 0.33 s per call at 1M DOF against slices of minutes (`_form_v` once per run; `_apply_km` only in merges).
- **Merge copies ($Q$, $MQ$ only):** merges are rare (windows held by two runs; the collapse case only would gain), the merge peak is not budgeted by any estimator, and the rewrite costs one K apply per column. Revisit if a measured case shows the merge sets the process peak.
- **`_chunks` padding compiles:** 6–15 ms per new width against counts of seconds to minutes; a host pad would add an unmodelled $(64, n)$ host block.
- **σ between the band-pass edges, span counts at small capacities, per-step residuals of inside candidates, `t_ref` including compilation:** `verify=None` prices each count per leaf, a capacity below 96 warns, `info["time"]["ritz"]` records the residual cost (no measured case where it matters), and compile time in `t_ref` only makes `verify=None` slightly more permissive.
- **To 18f:** verification slices, σ toward the unconverged neighbour, the top-remainder neighbour, the σ-move retry policy, the stagnation remedies, levers after a capacity stop.

## Tests (18e)

References: SciPy `eigsh` through `tests/test_strategy.py`'s helpers (`prob`, `check`, `run`, `fake_builds`, `need`, `lob_est`), `tests/test_shift_invert.py`'s references, NumPy references for the kernels.

**Existing tests that pin the slice fallback** get `levers=False` and keep their purpose: `test_memory_bracket`, `test_memory` (3 cases), `test_memory_block_sentinel`, `test_levers`, `test_given_chunk_no_fit`, `test_memory_no_lobpcg_build`, `test_forced_lobpcg_no_fit` (`tests/test_strategy.py`).

1. **`_choose` (pure, crafted reaches):** a table `reach[(S, c)]` and a need:
   - (b): the cheapest state reaching the need wins over a larger-reach one; a two-lever state that reaches where neither lever alone does; a state with an extra lever that adds nothing loses to the same state without it; a halved chunk without levers (cost 2.7) loses to chunk 40 with fp32 precond (0.9), and to chunk 40 with `low_memory` (1.6); a non-monotone table (adding `low_memory` lowers the reach); ties → fewer levers; nothing reaches → `{}`.
   - (a): the largest best reach wins; ties → cheaper; nothing fits → `{}`.
   Literal expected dicts throughout.
2. **Levers end to end, fake builds** ("short", `n_modes` 30): budgets found by bisection on the estimators (as `chunk_need` / `voxel_budget`):
   - trigger (b) from the top chunk (a reviewer found 1.214e8: no levers 40 < need, precond 160): `info["levers"] == {"precond_dtype": "float32"}` (a literal; its preconditions on the estimators asserted), `reason` "fits", `info["c_max"]` the chosen state's, `check` against `eigsh`; with `levers=False`: today's chunk, `{}`;
   - a budget in the precond+low window (about 1.17e8–1.21e8 at `tol=1e-7`, no basis lever): `{"precond_dtype": "float32", "low_memory": True}`;
   - `_choose` monkeypatched to return otf: the operator spy sees one build, in otf mode.
3. **Levers, real builds** ("short", AMG, `tol=1e-7`, f1 = 0 with `n_modes`): the precond+low budget of test 2 with real builds: `info["levers"]` as there, one AMG build in fp32, the band against `eigsh` (`check`). An `n_modes` overestimate (need above every state's reach, band inside the 60 reference modes): `{}`, today's chunk, the result against `eigsh`.
4. **Levers, no fit:** a budget below every state, `slice_options={"capacity": 96}`: "memory", `{}`; the warning names the hints on "short"; the "smallest slice does not fit" error lists otf.
5. **Applicability:** `low_memory=True` given: not in any subset; `tol=1e-7`: no basis row, `tol=1e-6`: basis row present (orthonormality checked at $10^{-5}$ when `basis_dtype` is in `info["levers"]`, as `test_fp32_basis`); Jacobi ("tiny"): no precond row; `ssh8` (`T.plate(8, 50)` with stubbed `_solve` and si_band): no otf row; `operator_options={"mode": "stored"}` given: no otf row; forced "lobpcg" with no state fitting: the error names all hints, `{}` tried and reverted; `info["levers"]` present on every band path. Non-bool `levers` raises; `levers=False` without `band` raises.
6. **`err.result` (si_band, fakes of `test_shift_invert.py`):**
   - "nothing converged" after a stored slice (`edges=[c]`: slice 1 converges, slice 2 raises);
   - "no progress" (`:1430` case);
   - the deflated-modes `ValueError` (`:1852` case, one slice stored);
   - `max_slices` with the full-slice fake (modes 7–11 stored);
   - one case with a merged window before the raise.
   Each: `e.result`'s λ equals the stored set, `info["converged"]` False, `reason` the message, `complete_to` as defined, `count["estimates"]` only the leaves run; V's orthogonality by the normal-exit tests' criterion. Plus the normal return after a non-converged slice following a stored slice: `complete_to` = that slice's lower edge; a converged band: `complete_to == hi`.
7. **`err.result` (solve_modes):**
   - `test_edge_above_spectrum`'s band ("tiny", "no progress") with $f_1$ = `freq(1.005 * lr[0])` (within `MU1_MARGIN`, so the band runs from 0, $\lambda_1$ stored and filtered, $i_0 \ge 1$) and `edges` in $(0, hi)$: `e.result` filtered to the band, key set equals a normal exit's, modes against `eigsh`, no "returned modes copied" warning (in-place compaction), `complete_to_hz` present;
   - one case after an "auto" capacity stop (`stopped`, `time["lobpcg"]`, $L > 0$ in the result's info);
   - forced "lobpcg" errors carry no `result`.
8. **Per-slice τ:**
   - `si_band` from 0 on "short" at `tol=1e-6`, capacity 600 (one slice): `slices[0]["tau"] == TAU_MIN`, $r_{acc} \le$ tol, all pairs against `eigsh` (si_band-level checks); with `slice_options={"tau": 1e-8}`, $r_{acc} >$ tol (pins the effect); steps, basis and capacity-full recorded for both; `sigma_moved` equal in both.
   - interior slices at tol $10^{-6}$: $e/a \le 10$ keeps $10^{-8}$, $e/a > 10$ tightens to the formula; tol $10^{-10}$ clamps at `TAU_MIN` (formula and a run's `tau`, not `tau_eff`).
   - with `fake_band`: a widen-and-rerun and a span rerun (`test_band_span_rerun`'s setup) receive the formula value of their own interval; `si_slice` receives `stag_tau` = $\tau_0$ throughout.
9. **Transposed send** (NumPy references): `_grow`'s B against `np.zeros` with V's first `up(count, c)` columns written in (fp64 and fp32 bases, a partial last chunk); `_project` over host W of widths 5, 37, 64, 100 (F-order and C-order) and a mixed host/device tuple against the dense $Y - WW^\top MY - RR^\top MY$ to $10^{-14}$ relative.
10. **Lazy gather:** `si._span_count` wrapped to count its `cols` calls of width `len(inb)`: `verify=False` and a `verify=None` skip by the bound gather nothing (the predict path receives a zero-stride array); a `verify=None` calibration plus count gathers once; `info["count"]` equals a stored dict of the same run's degree, gap and reason.
11. **Sketch:** `_check_ortho` accepts M-orthonormal W (host F- and C-order, device, fp64 and fp32) and rejects a perturbed one, for seeds 0, −1, $2^{40}$ (this also exercises the transposed `_chunks`).
12. **Preconditioner reuse:** the code-review-round-18 case of `test_band_rules` (call 2 a remainder with a new σ, call 3 requeued whole with the same σ), `si.si_band` called directly with a `make` returning a weak-referenceable object: `make` called once per distinct consecutive σ, `calls[1]["P"] is calls[2]["P"]`, and the previous P dead (weakref) before a rebuild.
13. **`COUNT_STEP`** (slow, and `peak` for scheduling: the compile of a 1M recurrence peaks near 4.5 GB, as `test_step_memory`'s large cases): `count_memory`'s step (without its 2 b blocks) ≥ the measured recurrence and ≤ 1.2× of it, Jacobi and multigrid, $b$ = 16 and 32, on one ≥ 1M `ssh8` and one ≥ 1M hex20 case (extends `test_shell_count` and `test_step_memory`).
14. All other band, strategy and shift-invert tests pass unchanged except where a lever, the τ rule or the refit changes a pinned value (each named in its commit).

## Docs (18e)

- `docs/eigensolvers.md`:
  - options table: `levers` (new row); `slice_options` `tau` (`:100`) and si_slice's `tau` and `stag_tau` rows (`:732`) with the per-slice default.
  - the ladder paragraph (`:154`, "never set automatically (PR 18e)") → the lever table, trigger, choice and cost factors, residual risks; Interior bands and strategy (`:140–141`); the `info` keys paragraph (`:183`, `levers`, `complete_to`).
  - the $\mu_1$ rule paragraph (`:175`) → the per-slice τ, its scope and cost.
  - Errors (`:181`): the `levers` argument errors, `err.result` with `complete_to`, the spectrum-bound closure; si_band's Errors and Termination bullets (`:774–775`); the si_band paragraph (`:750`: `slices[i]["tau"]`, `complete_to`); Edge above the spectrum (`:777`) reworded as a decision.
  - "Depth 1000 is marginal and expensive" (`:818`): remedies → 18f.
  - `solve_modes_memory` (`:128`, `:675`): the $f_1 > 0$ closure; the count memory model (`:665–675`) with the refit and its range.
  - "Π streamed" (`:618`): chunks sent transposed, the sketch drawn by NumPy `default_rng`.
  - the refill (`:113`, `:416`): the host transient is unchanged (one chunk), stated.
  - test sections: `:908` (the shell Memory bullet: `test_shell_count`'s ≥ 1M case), `:929–940` (the counting Memory bullet: the ≥ 1M `COUNT_STEP` cases; the sketch; `_chunks`), `:960–987` (Slicing a band `:967`, test_strategy `:977`, `_solve` refill `:982`, Errors and options `:985`): levers, `err.result`, per-slice τ, transposed send, lazy gather, P reuse.
- Docstrings: `solve_modes` (`levers`, `info["levers"]`, `complete_to`), `si_band` (the per-slice τ, `err.result`, `slices[i]["tau"]`, `complete_to`), `si_slice` (`stag_tau`), `_choose`, `_check_ortho`, `_span_count`, `count_memory`; the `COUNT_STEP` comment (`memory.py:569–574`, Jacobi fit, the fit range).
- `README.md`: `:5` status (18e, the split into 18e–18g); `:287` the interior-band paragraph (levers first, `levers=False`, `err.result`).
- `docs/ROADMAP.md`:
  - `:18` status, `:82` (the δ rule → per-slice τ), `:95` (remedies → 18f), `:97` (late errors only, closed), `:177` (np.memmap line removed), `:184` (the refit: valid to 1.6M unstructured, the 10M check deferred);
  - the 18e section rewritten to its contents; a new 18f section with the Split table's 18f row in full; the old 18f section renumbered 18g; `:627` "seven parts … 18g last";
  - dependency rows (`:844–846`): "18e Band follow-ups | 15d, 18b", "18f Deep bands | 18e", "18g CalculiX comparison | 18a–18f", "19 Multi-GPU | 17, 18a–18f";
  - Deferred GPU checks: the 10M `COUNT_STEP` check, the lever cost factors on the GPU;
  - `:854` the order.
- `docs/open_items.md`: `:99` (Memory constants: the refit range), `:143` (the $\lambda_{hi}$ bound), the PR 15d decisions (`:155–183`, including `:183`'s `levers()` note) item by item; the PR 15c and 15a → 18e sections (`:317`, `:331`, `:335`, `:339`), their headings retargeted (→ 18f or closed, `:328`'s merge figure corrected when closed); `:203` and `:222` the refit outcome; `:362` the order; a "PR 18e decisions" section.
- Memory: MEMORY.md (the autonomy line, the scale line's order, the CalculiX line), `feedback-autonomy.md` (an 18e entry: split, review rounds, merge), `project-calculix-benchmark.md` (description and body: 18g), `project-scale-questions.md` (the order).

## Risks and decisions without user input (logged in `docs/open_items.md`)

- The split (18f deep bands, 18g CalculiX) and the order.
- Levers on by default, chosen by enumeration with static cost factors when nothing fits or the top chunk misses a known depth; polish never off automatically; the residual risks of section 1 (otf kept into slices after a capacity stop, no levers for unknown depths with a fit).
- `err.result` on a raise rather than a return; forced "lobpcg" gets none.
- The per-slice τ instead of the roadmap's $\delta_\max \le 1$ slice cap: $r_{acc} \le$ tol on slices with $e/a$ bounded, at +8–12% MINRES iterations on the affected slices (every slice at tol ≤ $10^{-8}$), with `TAU_MIN` as the floor; slices from 0 at `TAU_MIN` without a guarantee.
- The closures in section 6.
- The NumPy sketch in `_check_ortho`.
