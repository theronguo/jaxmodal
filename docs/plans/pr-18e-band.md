# PR 18 plan: performance pass, part e: band follow-ups

Branch `pr-18e-band` from `main` (0948d20, PR 18d and the roadmap reorder merged). Roadmap: "PR 18e: Band follow-ups".

## Revisions

- Rev 1: first draft.

## Split (decision without user input)

The roadmap's 18e lists about fifteen items. A feasibility probe during planning (below) showed that the depth-1000 MINRES problem has a strong remedy that needs the LOBPCG modes a capacity stop now discards, so the two deep-band items move to a new PR together:

| PR | branch | contents |
|---|---|---|
| 18e | `pr-18e-band` | this plan: automatic memory levers, `err.result`, the per-slice τ ($\delta_\max$ enforcement), the band-path performance items, the `COUNT_STEP` refit, and the closures below |
| 18f | `pr-18f-deep-bands` | new: the hybrid LOBPCG/slice junction (the locked modes kept as known pairs) and MINRES deflation by the known modes below σ (the depth-1000 remedy), the σ-move retry policy; its own plan after 18e merges |
| 18g | `pr-18g-calculix` | the old 18f (CalculiX comparison), unchanged |

Dependencies: 18f ← 18e; 18g ← 18a–18f. Order: 18e → 18f → 18g → 20 → 21 → 22 → 23a → 19 → 23b (the user's order of 2026-10-19 with 18f inserted; "everything hardcore GPU comes last" still holds: 18f runs and is tested on the Mac CPU).

**The probe** (`scratchpad`, not committed; hex20 $16^3$, 56k DOF, AMG at σ, 6 columns, τ = $10^{-8}$, σ mid-gap above depth $d$; references from `reference_modes`): MINRES on $K - \sigma M$ against MINRES on the deflated operator $\Pi^\top(K - \sigma M)\Pi$, $\Pi = I - WW^\top M$, with the preconditioner $\Pi P \Pi^\top$ and $W$ the exact modes below σ:

| depth | none | nearest 20 | all below σ |
|---|---|---|---|
| 197 | 1500 its, 61.9 s | 910 its, 39.0 s | 140 its, 7.7 s |
| 503 | 4960 its, 203.7 s | 2940 its, 125.1 s | 220 its, 14.5 s |

All solves converged (explicit residual below τ, the deflated solution completed by the exact part $\sum_w w\,(w^\top b)/(\lambda_w - \sigma)$). Deflating every mode below σ makes the operator nearly definite on the complement: 11× / 23× fewer iterations (8× / 14× less time with the projections) at depths 197 / 503, close to a shallow σ (depth 20: 100–120 iterations). The nearest 20 alone give 1.6–1.7×. This is the roadmap's own option ("deflate the converged modes below σ from MINRES") and it needs exactly the modes LOBPCG holds at a capacity stop, hence 18f.

## Goal (18e)

Close the PR 15d band follow-ups that do not need the junction, and the band-path performance items, without changing any solver result except where stated (levers change the options a band solve runs with; the per-slice τ tightens inner solves).

## Design (18e)

### 1. Automatic memory levers (`eigen/api.py`)

Today the ladder (`_ladder`) picks a chunk from the per-chunk $C_{max}$; the levers (`levers()`) are only named in warnings and errors. LOBPCG from 0 is 33–1260× cheaper than slices at every measured depth (`strategy_study.json`), so a lever that keeps a band on LOBPCG pays even at 2–4× per iteration.

- New keyword `levers=True` (bool; `False` keeps the options as given, today's behaviour). Validated up front (`ValueError` for a non-bool); `levers=False` without `band` raises with the other band-only options ("strategy, memory_budget, slice_options and levers need band").
- **Trigger:** under `strategy` "auto" or "lobpcg", after the ladder without levers: (a) no chunk fits (`ch is None`), or (b) a known depth ($f_1 = 0$ with `n_modes`, `need` set) that no fitting chunk reaches (no $c$ with $C_{max}(c) \ge$ `need[c]`). A band of unknown depth that fits gets no lever (a capacity stop is the fallback; 18f makes it cheap).
- **Order** (each applied only when still at its default and applicable; cumulative; after each, `need`, `reach` and the ladder are recomputed; stop at the first state where the trigger no longer holds):
  1. `precond_dtype="float32"` (GMG or AMG, not Jacobi): LOBPCG and the polish only; the same PCG counts with AMG, V-cycle 0.82–0.91× the fp64 time; faster on voxel meshes (`eigensolvers.md` options table).
  2. `basis_dtype="float32"` when `tol >= 1e-6` (the fp32 basis floors the locked residuals near $2^{-23} \approx 1.2\cdot10^{-7}$; a 10× margin).
  3. `low_memory=True` (1.4–1.9× per iteration, `eigensolvers.md` low-memory table).
  4. `operator_options={"mode": "otf"}` on an unstructured mesh without `ssh8` blocks (2–4× per apply, `operators.md:173`).
  `polish_options={"on": False}` is never applied (a stalled chunk then fails); it stays named in warnings.
- If the trigger still holds after the last lever: case (a) falls back to slices ("memory") with **all levers reverted** (slices are fp64 throughout, and a slower otf operator would only slow them); case (b) keeps all applicable levers (each lowers the estimate, so the last state reaches furthest, and a deeper capacity stop leaves fewer modes to the slices) and picks the chunk by today's "no chunk reaches the need" rule.
- The ladder and levers need only the estimators, `free` and the rigid count, so they move before the operator build (`n_avail` from `rigid_modes(mesh, free)` as today; the rigid basis is built once). The operator is built once, with the final `operator_options`.
- `info["levers"]`: the applied levers as strings (e.g. `["precond_dtype='float32'", "low_memory=True"]`), `[]` when none. The warnings and errors keep naming the levers not set (`levers()` reads the final options).
- `solve_modes_memory` is unchanged (it estimates the options it is given).

### 2. `err.result` (`eigen/shift_invert.py`, `eigen/api.py`)

A capacity error today discards every pair found (open item "A capacity error discards the band": 37 slices, 25 min, then the raise).

- `si_band`: the post-loop stage (merge, assembly, `info`) moves into a helper `_finish(...)` used by the normal exit and by the error exits: the two `_cap_error` raises ("nothing converged", "no progress"), the deflated-modes capacity `ValueError` and the `max_slices` `RuntimeError`. Each error gets `err.result = (lam, V, info)` of the pairs stored so far, merged as on a normal exit, with `info["converged"] = False` and `info["reason"]` the error message; the leaves counted so far are in `info["count"]`. The slice's own unstored state is dropped (P released first, as today).
- `solve_modes`: catches an exception carrying `result` from `si_band`, applies the band filter and in-place compaction of the normal exit (the tail moves into a helper `_finish_band(lam, V, info, i0, i1)`), sets `err.result` to the filtered `(lam, V, info)` with the `info` keys of a normal exit, and re-raises. The forced-"lobpcg" cap error gets `err.result` from LOBPCG's locked modes (sorted, filtered to the band like the "fits" path, `count` = `dict(ran=False, reason="capacity")`).
- Documented in `eigensolvers.md` (Errors) with the example `except ValueError as e: lam, V, info = e.result`.

### 3. Per-slice τ: $\delta_\max$ enforcement (`eigen/shift_invert.py`)

A pair is accepted at $r_i \le \max(\mathrm{tol}, 10\,\tau_\text{eff}\,\delta_i)$, $\delta_i = |\lambda_i - \sigma|/|\lambda_i|$. On a slice $[a, e]$ with σ the midpoint, the lowest pair has $\delta \approx (\sigma - a)/a$, unbounded as $a \to 0$: slices from 0 report $r_{acc}$ above tol (open item "δ_max ≤ 1 is not enforced"), and a large $r_{acc}$ widens the chain gap $100\,r_{acc}$ that edges need (`eigensolvers.md:175`). The roadmap's remedy (cap $\delta_\max \le 1$ when choosing slices) would split slices from 0 into narrow pieces at full slice cost each; the inner tolerance is cheaper: MINRES iterations grow with $\ln(1/\tau)$ only.

- `si_band` gains `lam1=None` (an estimate of $\lambda_1$). Per slice $[a, e]$, unless the caller passed `tau` in its options: $a_\text{eff} = a$ for $a > 0$, else $\lambda_1^{est}/2$ (the factor 2 covers an upper-bound estimate; without `lam1` the slice keeps the default τ), $\hat\delta = (\sigma - a_\text{eff})/a_\text{eff}$, and

$$\tau_s = \max\Bigl(\tau_\text{low},\ \min\bigl(\tau_0,\ \mathrm{tol}/(10\,\hat\delta)\bigr)\Bigr),$$

  with $\tau_0 = 10^{-8}$ (the default) and $\tau_\text{low} = 10^{-12}$ (`TAU_LOW`, above the measured fp64 floor $3\cdot10^{-14}$–$9\cdot10^{-13}$ to depth 503). At $\tau_\text{low}$, $r_{acc}$ can still exceed tol (reported, as today). A widen-and-rerun keeps the slice's τ. The span counts and their reruns use the parent's τ.
- `solve_modes` passes `lam1=info["mu1_ub"]` (LOBPCG's first Ritz value or the pre-estimate; on slices from 0 from the start without either, the σ pre-estimate is run for this as for $f_1 > 0$; Jacobi: `sigma_estimate`).
- `info["slices"][i]["tau"]` records each run's τ.
- Cost: τ $10^{-8} \to 10^{-10}$ adds about 25% MINRES iterations on the affected slices (to measure in the test below; the ratio is recorded in `eigensolvers.md`).

### 4. Band-path performance (`eigen/lobpcg.py`, `eigen/shift_invert.py`, `eigen/counting.py`)

- **Refill:** `_grow` writes each chunk through `jax.device_put(V[:, i:i+c].T)` and a jitted transposed insert (`_insert_t`: `dynamic_update_slice` of `Xt.T`), so the F-order host chunk is sent without a host transposing copy (2.4× faster on CPU, PR 15d review measurement; re-measured at 1M DOF in the test log).
- **Verify skips before the gather:** `count` and `band_count` accept `W=None` / `V=None` with `_predict=True` (only the eigenvalues enter the degree). `_decide_slice` takes a gather callable: the prediction runs without arrays; the calibration and the count call the gather once. `si_checked` builds `W_in` (a fancy-index copy when not contiguous) and `_span_count` builds `cols(inb)` only through it, so `verify=False` and a `verify=None` skip by the bound gather nothing.
- **Merge copies:** `_merge` keeps $Q$ and $MQ$ only: $G_K = Q^\top K Q$ is formed by device chunks of $K Q_c$, and the residuals recompute $KX$, $MX$ per device chunk of $X = QZ$. Host memory per window column falls from about $(4\text{–}5)\cdot 8n$ to about $(3\text{–}3.5)\cdot 8n$ at two extra K applies per column (merges are rare: windows held by two runs).
- **Host transposes:** `_form_v` and `_apply_km` get transposed jitted kernels returning $(w, n)$ C-order device arrays, read back as F-order views (`np.asarray(Xt).T`), and `_apply_km` sends its input the same way: no transposing copy on either side (measured at 1M DOF: `_form_v` 1.36 → 1.03 s, PR 15c review).
- **Small compiles:** `_check_ortho` draws its sketch with NumPy (`np.random.default_rng((seed, 2**32 - 1))`), not `jax.random.normal` of shape $(k, 3)$ (one compile per W width); `_chunks` pads a host array's last chunk on the host (one compiled width, no `jnp.pad` compile per width; device arrays keep `jnp.pad`). The sketch values change (a different generator), the check's tolerance does not.
- **Preconditioner reuse on a whole requeue:** `si_band` keeps the previous slice's preconditioner when the next slice has the same σ (a slice requeued whole), instead of rebuilding it.

### 5. `COUNT_STEP` refit (`memory.py`)

`count_memory`'s unstructured step $8n(a\,b + c) + w$, $(a, c) = (10.3, 3.0)$, runs low above its fit range (measured/model 1.015 at 0.54M, 1.035 at 0.97M, 1.045 at 1.5M `ssh8` DOF with Jacobi; hex20 $36^3$ 1.023; open_items PR 16 Reader entry).

- Measure `memory_analysis` of `_recurrence` (the existing `_recurrence_temps` helper of `tests/test_counting.py`, and `shell_solve_study.count` for `ssh8`) on `ssh8` plates at about 0.8M, 1.2M and 1.6M DOF and hex20 boxes $36^3$ (0.6M) and $44^3$ (1.08M), $b$ = 4, 16, 32, 50, Jacobi and the multigrid; add them to the fit's points and refit $a$ (upper envelope, $c \ge 0$, rounded up by 0.5 as the 16b fit) so that the model bounds every point (old and new) and stays within 1.2× of the measurement at $b \ge 16$ on the ≥ 1M cases. The stencil entry is unchanged (its fit range reaches 2M).
- The refit raises the count's estimate a few percent, so `probe_block` and `band_probe_block` may return a smaller block near a budget's edge: the tests that pin a block (listed after the measurement in the implementation commit) are rerun and adjusted by their own formulas, not by hand.

### 6. Closed without change (logged in `docs/open_items.md` with the reason)

- **Up-front spectrum bound $\hat\lambda_\max$:** an edge above the spectrum is only reachable on meshes small enough that the band covers the top of the spectrum, where the late error costs seconds; the cheap rigorous bound (max over elements of $\lambda_\max(K_e, M_e)$) is typically 2–4× above $\lambda_\max$, too loose to close a side. Stays a late error (`eigensolvers.md:777`).
- **Halving before a capacity error:** kept (robustness over the error's cost); with `err.result` the pairs found are no longer lost.
- **$f_1 > 0$ in `solve_modes_memory`:** an interior band's memory is set by `solve_modes` itself (the ladder and the slice capacity are sized to `memory_budget`); an estimate would need the depth, unknown before the solve. The `ValueError` message names `band=(0, f2)` with `n_modes` = the expected depth as the LOBPCG estimate.
- **`np.memmap` for the locked modes:** the fit test bounds the host V by the host budget, so LOBPCG never locks past RAM; at 10M DOF and 1000 modes the fp64 host V is 80 GB, below a GPU node's RAM. Not needed; the ROADMAP line is removed.
- **σ between the band-pass edges, span counts at small capacities, per-step residuals of inside candidates, `t_ref` including compilation:** `verify=None` prices each count per leaf, a capacity below 96 warns, `info["time"]["ritz"]` records the residual cost (no measured case where it matters), and the compile time in `t_ref` only makes `verify=None` slightly more permissive. No change.
- **Verification slices, "move σ toward the unconverged neighbour", the top-remainder neighbour:** 18f (the junction changes which remainders exist); the σ-move retry policy and the stagnation warning likewise.

## Tests (18e)

References: SciPy `eigsh` through the existing helpers of `tests/test_strategy.py` (`prob`, `check`), `tests/test_shift_invert.py`'s references, the old code paths for equality.

1. **Levers, known depth** (`tests/test_strategy.py`, fake builds as test 3): a budget where the no-lever ladder does not reach the need but `precond_dtype="float32"` plus `low_memory=True` does (found by the estimators): `info["levers"] == ["precond_dtype='float32'", "low_memory=True"]`, `reason` "fits", the band matches `eigsh` (`check`). The same budget with `levers=False`: today's chunk and no levers.
2. **Levers, no fit:** a budget below every lever state: "memory", `info["levers"] == []`, the slice operator built with the user's options (stored mode), the warning names all four levers.
3. **Levers respect user options and tol:** `low_memory=True` given: not in `info["levers"]`; `tol=1e-7`: no basis lever; Jacobi: no precond lever; `ssh8`: no otf lever; forced "lobpcg" applies them before raising (the error names the levers left).
4. **Levers argument checks:** non-bool raises; `levers=False` without `band` raises.
5. **`err.result` (si_band):** the existing "no progress" and "nothing converged" cases of `test_shift_invert.py` (around :1430): `e.result` holds the pairs stored before the raise (λ against the reference to tol, V M-orthonormal to $10^{-8}$), `converged` False, `reason` the message; `max_slices` likewise.
6. **`err.result` (solve_modes):** a band whose top edge lies above the spectrum on the 54-DOF mesh (`eigensolvers.md:777`) under "shift_invert": the error's `result` is filtered to the band, its modes match `eigsh`; forced "lobpcg" at the cap (test 2's budget): `result` holds the locked in-band modes, `count["reason"] == "capacity"`.
7. **Per-slice τ:** `si_band` from 0 on a small hex20 mesh with $\lambda_{hi}/\lambda_1 \approx 100$ (one slice, `lam1` from `eigsh`): $\tau_s$ as the formula, `r_acc <= tol`, all pairs against `eigsh`; with `slice_options={"tau": 1e-8}` the same slice reports `r_acc > tol` (today's behaviour, pinning the effect); an interior slice with small $\hat\delta$ keeps $10^{-8}$; $\tau_s$ clamps at `TAU_LOW`; the MINRES iteration ratio is printed and recorded.
8. **Refill:** `_grow` returns the same B as the old path (exact equality) for fp64 and fp32 bases and a partial last chunk.
9. **No gather on skips:** `si_band` with `verify=False` and with `verify=None` skipping by the bound: a spy on the gather callable records no calls; `info["count"]` equals today's (degree, gap, reason).
10. **Merge:** `_merge` on two overlapping runs of a known cluster returns the same θ, residuals and duplicates as the old implementation (stored in the test from the old code path at commit time) to $10^{-12}$; peak host bytes by `tracemalloc`-free counting of the allocated arrays (the column count formula).
11. **Transposes and compiles:** `_form_v`, `_apply_km` equal to the old results exactly; `_check_ortho` and `_chunks` over widths 5, 37, 64, 100 compile nothing after the first (counted with `jax.log_compiles` and `caplog`).
12. **Preconditioner reuse:** the existing requeue-whole case (`test_shift_invert.py:1435`): `make` called once for the two runs of the same interval.
13. **`COUNT_STEP`** (slow): `count_memory`'s step (without its 2 b blocks) ≥ the measured recurrence and ≤ 1.2× of it at $b$ = 16 and 32 on the ≥ 1M `ssh8` and hex20 cases, Jacobi and multigrid (extends `test_shell_count` and `test_step_memory`).
14. The existing band, strategy and shift-invert tests pass unchanged except where a lever, the τ rule or the refit changes a pinned value (each such change named in the commit).

## Docs (18e)

- `docs/eigensolvers.md`: the ladder paragraph (`:154`, "never set automatically (PR 18e)") → the lever rule, order and `info["levers"]`; Interior bands and strategy (`:140–141`) with the levers; the $\mu_1$ rule paragraph (`:175`, "$\delta_\max \le 1$ is not enforced") → the per-slice τ; Errors (`:181`) with `err.result` and the spectrum-bound closure; Slicing a band (`:777`, "no up-front spectrum bound (PR 18e)") reworded as a decision; the merge memory paragraph with the $Q$, $MQ$ figures; `solve_modes`' docstring and the options table (`levers`); the count memory model paragraph (`:665–675`) with the refit range.
- `docs/ROADMAP.md`: `:82` (the δ rule), `:97` (late errors only, closed), `:177` (np.memmap line removed), `:184` (18e's refit), the 18e section rewritten to its contents, new 18f section, 18f → 18g, dependency rows, `:18` status, `:627` the split, `:854` the order.
- `docs/open_items.md`: the PR 15d decisions (`:155–182`) and the PR 15c sections (`:317–347`) updated item by item (done in 18e, closed with the reason, or → 18f); a "PR 18e decisions" section; `:222` and `:203` the refit outcome; `:362` the order.
- `README.md:5`: status (18e, the split into 18e–18g).
- Memory: MEMORY.md and `project-calculix-benchmark.md` (18f → 18g), the autonomy line.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- The split (18f deep bands, 18g CalculiX) and the order.
- Levers on by default (`levers=True`): a band that today raises (forced) or falls to slices can now run slower LOBPCG options instead; LOBPCG stays far cheaper than slices at every measured depth. Polish never off automatically. Levers apply only when a known depth is not reached or nothing fits.
- The per-slice τ instead of the roadmap's $\delta_\max \le 1$ slice cap: same goal ($r_{acc} \le$ tol) at about +25% MINRES iterations on the affected slices instead of extra slices.
- The closures in section 6.
- The new sketch generator in `_check_ortho` (results of the check unchanged within its tolerance).
