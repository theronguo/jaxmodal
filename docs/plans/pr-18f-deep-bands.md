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
- Rev 5 (plan review round 4: docs, performance and correctness satisfied):
  - **No truncation.** LOBPCG's untruncated block (lazy zero tail) goes to si_band, whose one `resize` sizes the result, so the $j = 1$ case and the truncation fallback are gone.
  - **The in-place check runs on the final parts:** a known prefix of length $p \le j$, with merged windows at the cut only shortening it.
  - **$E$/$H$:**
    - $E$ is trimmed before `pinv`;
    - the setup runs in chunks of $c$, modelled with the operator's vectors.
  - **$N_\text{new}$** is floored by the recomputed $D - j$.
  - **Tests:** the selection helper `_z_cols`, the Moore–Penrose asserts, corrected expected values, the otf branches, the spies at call time.
  - **Docs:** `:18`, `:115`, `:153`, $d_Z$, `:342`/`:346`.
- Rev 6 (plan review round 5: docs satisfied):
  - **All locked modes below the band:** $j = D$, no cut, none dropped from deflation. The benchmark stops LOBPCG inside the band, so it measures the cut path.
  - **In place:**
    - not for a C-contiguous block (copy), so a 1-column block is safe in direct calls;
    - one entry `assert` on `known`.
  - **`max_deflate`** bounds $d$ inside `_slice_plan`, so 0 gives today's $C$.
  - **Host extra:** the locked columns above the cut.
  - **The deflated operator** is a module-level function (one compile per padded $n_z$).
  - **otf → stored kept** (`ROADMAP.md:685`), although a reviewer proposed deferring it.
  - **Tests:**
    - test 7 measures two peaks;
    - test 6's stub cases are fixed;
    - `stub_band` gets `edge_rr_failed`;
    - `_cut(lam, t, w, hi, tol)`;
    - test 10's spy dropped.
  - **Docs:** the revision order, `:1004`, `:175`, `:312`.
- Rev 7 (plan review round 6: performance satisfied):
  - **The junction has two cases:**
    - (a) a cut with its midpoint in $(\text{lo}_b, \text{hi})$, counted, with a leaf;
    - (b) otherwise, every locked mode below $\text{lo}_b(1 - 100\max(t_i, \text{tol}))$ known: none dropped from deflation, and no junction only for $j = 0$.
  - **otf → stored** on $d$ alone.
  - **The entry assert** excludes C-contiguous blocks.
  - **`_z_cols`** is tested once.
  - **Setup chunks** by `dynamic_slice_in_dim`.
  - **`probe_budget`** holds $8nD$; `no_count` is prefixed, not overwritten.
  - **The benchmark band** becomes modes 970–1000, with the stop near 995 and cut > lo_b asserted.
  - **Tests:**
    - the sorted known block at the stop;
    - `host_extra`;
    - the `n_new` need cap;
    - case (b) with the real si_band.
  - **Docs:** `info["junction"]` and `stagnation_risk` for case (b), $d_Z$.
- Rev 8 (plan review round 7: simplicity, docs and correctness satisfied):
  - **The junction stop** of the benchmark band at a real chunk boundary; test 10 prints its ratio only.
  - **`max_deflate`** enters only `_slice_plan`; $d$ bisected with step 1.
  - **$N_\text{new}$** without `n_modes`: $\max(200, \lceil 1.25(D - j)\rceil)$, `_n_new`.
  - **The entry assert:** `max_capacity` + $m$ columns, `V.shape[1] >= j + 1`.
  - **The assembly** grows `V` per part (numpy zero-fills a grown tail).
  - **otf → stored** rebinds `op_opts`; the final plan runs without the aggregates.
  - **Case (b) test** construction fixed; `from_zero` with no gap logged.
- Rev 9 (plan review round 8: docs and simplicity satisfied):
  - **The stopping chunk is kept** at a capacity stop (converged and confirmed, written to `V`, not locked); the benchmark predicts $D$ from the chunk boundaries and the $c_L$ grid.
  - **The assembly** keeps `V` at 2 or more columns until the final resize (an $(n, \le 1)$ array is also C-contiguous) and grows in steps of $c$.
  - **`_slice_plan`:** `max_deflate` defaults to 0; $C_0$ takes `n_new`, step 2 searches $[C_\text{lo}, \text{need cap}]$.
  - **Acceptance** requires no stagnation and the explicit residual, with `shift_invert_study`'s setup.
  - **Tests:** $p$ = 0 and 1 in place, a need cap below 192, `no_count` with the real si_band, `n_band` recorded.
- Rev 10 (plan review round 9: correctness, tests and performance satisfied):
  - **The assembly** writes into `V` from column $p$ with no shrink first (a macOS shrink frees nothing and the next grow moves the block).
  - **$D$** = the locked modes plus the kept chunk (`chunks[-1]["kept"]`), used for every "locked modes" count.
  - **The benchmark band** runs with `levers=False` and checks $D$, the cut and $d \ge j$ before LOBPCG.
  - **Docs:** `eigensolvers.md:999`, `lobpcg.py:567`/`:666`, `api.py:648`, the catch-all, and the kept chunk logged.
  - **Tests:** the kept chunk's updated tests named; test 6 at $D$ = 40 with 10 above the cut; test 10's $D$ below the band; `no_count`'s `_verify` stub.
- Rev 11 (plan review round 10: simplicity, docs, tests and performance satisfied):
  - **The assembly's failures** are all found before any write: a probe grow by $c$ (numpy raises on a second reference only at a size change).
  - **$D$** = `len(lam)`; no `kept` key.
  - **MINRES phase** models two $n \times b$ temporaries.
  - **Tests:** test 10 with `levers=False`; `no_count`'s expected reason built from `info["junction"]`; `test_mac_growth_peak` checked; test 6's off-by-one wording.
- Rev 12 (plan review round 11: all five satisfied; minors applied):
  - **The assembly:** one "no shrink" sentence, the entry assert in its own bullet, the probe grow as the ownership check, the fallback reading known columns from `V`, the last item's `Z` list dropped before the probe.
  - **Tests:** two grows after the probe, a `break`-path in-place case, `no_count`'s stub with `passed=False`.
  - **Docs:** `eigensolvers.md:184`, `:757–759` (si_checked), `:166`'s otf and "until 18f" clauses; the Rev 10 entry restored.

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
- $Z^\top(AZ_j)$ at $n_z$ = 1000 takes 20 s in chunks of 6 and 5.3 s in chunks of 96. The $d$ applies of $A$ add about 20 s, so a setup costs about 25 s per σ (70 s at $n_z$ = 2000), negligible against a 1M slice.

## Goal (18f)

Bands past LOBPCG's basis cap continue from the locked modes instead of from 0. Every slice's MINRES deflates the known modes below σ, cutting MINRES iterations at depth 1000 by more than 10× (to at most 0.1× undeflated). The levers stop discarding LOBPCG reach. Every pair still passes the same acceptance tests, so the results are unchanged up to the inner-solve accuracy. The band's slices differ: they now start at the cut (case (a)) or at $\text{lo}_b$ (case (b)).

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
- `np.linalg.pinv((E + Eᵀ)/2, rcond=1e-10, hermitian=True)` on $E$ trimmed to $n_z \times n_z$ (the padding dropped), symmetric by construction; $H$ is padded back with zeros on the device.
- Duplicate store columns (two runs' copies of one mode, merged only at the end) drop out.
- $E$ is accumulated as $Z^\top (A Z_j)$ in chunks $Z_j$ of $c$ columns, $c$ = `_chunk(block)`, taken with `jax.lax.dynamic_slice_in_dim` at a `jnp.int32` offset (as si_slice does for the right-hand side), so the setup compiles once per padded $n_z$. Each chunk reads all of $Z$, so larger chunks are faster (at 1M, 5.3 s at 96 columns against 20 s at 6), but a chunk of $4c$ can become the binding memory phase (at 1M and on a 10M GPU), costing deflation columns worth far more than the saved seconds; $c$ never binds.

**Tolerance.**
- MINRES runs at the per-column tolerance $\tau\,\|b\|/\|P_Z b\|$ (`minres` broadcasts an array `tol`). The ratio $\|P_Z b\|/\|b\|$ can lie on either side of 1.
- The returned relative residuals are scaled back by $\|P_Z b\|/\|b\|$ (`_div`, so $\|b\| = 0$ gives 0). So `tau`, $\tau_\text{eff}$, the stagnation test and the σ-move threshold keep their meaning.
- No special case for columns already solved by $ZHZ^\top b$: the tolerance is $\tau\|b\|/\max(\|P_Z b\|, \text{tiny})$, and `_minres` already converges a zero column at once and a column with tolerance ≥ 1 at its first check.

**si_slice.**
- New argument `Z`: a sequence of host 1-D columns, never returned. si_band's store columns are such views.
- `_upload(cols, n, c)` (shared with $W$) copies the columns into fresh `np.zeros` chunks (never passing a store view to jax: jaxlib's held host view would block the final `resize`) and fills a zero device block of $\lceil n_z/c\rceil c$ columns in chunks of $c$ via `_insert`, with no full host or device copy. A reviewer measured +0.39 GB maxrss for a 3.84 GB block.
- `_deflate_setup(A, Zd, c)` returns $H$.
- `_dminres(A, rhs, P, Zd, H, tol, maxit)` does the right-hand side, MINRES on the `Partial` operator, the recombination and the residual scaling. The operator is a module-level function wrapped in `Partial`, so jitted `_minres` compiles once per padded $n_z$, not once per call.
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
- The $j$ locked pairs below the cut, ascending. `V` is LOBPCG's untruncated host block, F-order, with `max_capacity` + $m$ columns ($m$ = LOBPCG's block `info["block"]`): the columns past the locked ones are lazily paged zeros, and the at most $w$ locked columns above the cut (about 0.2 GB at 1M) stay until the assembly. si_band uses `V[:, :j]`.
- `leaf`: None, or `(interval, count)`, the known leaf's interval and count record from solve_modes (section 3). si_band puts `{interval, kind: "known", leaf: True, count}` first in `slices` and the pair into `leaves` before the loop, so `info["count"]` covers it on the normal and the error exit alike.
- si_band takes the list's items and empties it (`known.clear()`), so the caller's list holds no reference.
- They enter the store as run −1 (views of `V`'s columns). Their windows close the first slice's lower side as deflated $W$ modes, as any stored pair would.
- Nothing else changes: the work starts at `lo`, and `r_s` comes from the slices as today. The known leaf's lower edge stays out of `bounds` (so `collapsed` keeps its meaning), and `complete_to`'s docstring covers the known leaf.

**Assembly in place.** The known pairs all lie below si_band's lo = max(lo_b, cut), and the slices start at or above it. The check runs on the final `parts`, after the merge loop (a widened merge can pull the top known-only window in):
- $p$ is the length of the leading prefix of index parts whose store indices concatenate to `arange(p)` (known columns in place, $p \le j$);
- the in-place path applies when no later index part holds a known column (index $< j$) (the entry assert below already excludes a C-contiguous block, which `resize` would lay out in C order; ownership is checked by the probe grow below, not separately). A merged window's known inputs were already copied by `cols()`, so a slice that re-finds a known mode, or a chain across the cut, only shortens $p$.
- `known` is internal: one entry `assert` that `known` is a list, `V` is fp64, F-contiguous and not C-contiguous (LOBPCG's band block has `max_capacity` + $m$ columns, $m$ = `info["block"]`, at least 2), `V.shape[0] == n` and `V.shape[1] >= j + 1` (the count reads `V[:, :j+1]`), and `lam` is ascending with `lam[-1] < lo`.

Then:
1. the store entries of the known block (indices $< j$) are set to None and every other name for it (the per-slice `V`, the unpacked `known` items, the last item's `Z` list, which a `break` exit leaves bound) dropped, leaving the one name used for the resize; the slice entries stay until written; `gc.collect()` runs;
2. the parts are written straight into `V`'s columns from $p$ on, each part's store entries set to None after its write. `V` already has `max_capacity` + $m$ columns (the $D$ touched ones, then lazy zero pages); only a part that does not fit grows it, by `resize` in steps of $c$ columns (numpy zero-fills a grown tail, so one up-front grow would hold the slice columns twice), and one final `resize` sets the result width. Each resize keeps the leading columns, since F-order data is column after column. No shrink comes first: on macOS a shrink of a touched block frees nothing and the next grow moves it, holding $8n(D - p)$ twice (a reviewer's measurement: maxrss 4.68 GB against 2.76 GB in place), and `V` never drops to an $(n, \le 1)$ C-contiguous shape before the final resize. Overwriting the known columns from $p$ on is safe under the in-place condition: no later index part refers to them, merged parts hold `cols()` copies, and the locked columns above the cut are not in the store;
3. so the host peak is $V$ plus the unwritten store plus one part; a large `realloc` is an `mremap` on Linux (no copy); measured on macOS: resizes of about 10 ms each (no copy), maxrss after reading every column about 5% above a plain fill (no doubling); macOS maxrss undercounts realloc-grown pages until they are read, so a peak check reads all of `V` first.

Every failure is found before any write: the known-index check above, then (after step 1) one probe grow `V.resize((n, V.shape[1] + c))`, the first of the steps-of-$c$ grows ($8nc$ committed), which raises for another reference or a borrowed block (numpy raises only on a size change, so a later grow or the final resize would fail after columns were overwritten). On a failure the assembly copies as today, with the "returned modes copied" `RuntimeWarning`: a known index $s < j$ is read from `V[:, s]` (step 1 may already have set its store entry to None; nothing is written before the probe), a slice index from the store.

The `err.result` path assembles the same way; si_band holds the known block under its own name (not the per-slice `V`) and drops it beside `del V_f`, with `traceback.clear_frames` in solve_modes as today.

### 3. The junction (`solve_modes`, `eigen/api.py`)

**The stopping chunk is kept (`lobpcg.py:666`).** At a capacity stop the chunk has converged, passed `_reortho` and `_confirm`; only B has no room to lock it. It is kept like the final chunk (`Xi = X`, `jj = j`, `stopped = "capacity"`; `i` is no longer zeroed, so past the `i < j` break $i = j$ and `n_band` = count + $j$ = $D$), except that `chunks[-1]["locked"]` stays 0; $D$ = `len(lam)` (api.py's `depth`). The chunk is written to `V` at `:685`; $\text{count} + j \le C_\text{max} + m$ fits `V`, so it costs no memory. Without it the slices would recompute up to a chunk of modes at the deepest point (about 12 deflated slice steps at 1M). $D$ includes it; the tests follow: `test_solve_boundary` (:983: `len(lam)`, residuals and thresholds 35 → 40, columns 35:40 non-zero and M-orthonormal against the locked ones, `V` zero past 40, `chunks[-1]["locked"] == 0` kept, docstring), `test_solve_capacity_sequence` (:978: 155 → 160, docstring), `test_solve_boundary_one` (31 → 32), `test_mac_growth_peak` (:2396, its model's `r["locked"] = len(lam)` now includes the kept chunk, checked), `test_capacity` (:215–218: "40 modes locked", `s["locked"] == 40`, `lr[39] < hi`, docstring) and `test_error_result_capacity` (:922: "40 modes locked"); and `eigensolvers.md:114` ("the stopping chunk dropped") changes.

After a capacity stop (`strategy="auto"`), with $\text{lo}_b$ = the band's lower edge (0 when `from_zero`, else $\lambda_{lo}$):

1. **Sort.** The locked modes are sorted as on the "fits" path (`_permute(V, order)` also when stopped), with `info`'s residuals and thresholds. The aggregates are kept until the count is done.
2. **The known pairs**, in two cases ($D$ = LOBPCG's converged modes at the stop: the locked ones plus the kept stopping chunk; "locked modes" in sections 3, 5 and 7 means these $D$, as do the warning's N in "N modes locked", `STAGNATION_DEPTH`'s count and `info["junction"]`):
   - **(a) A cut above lo.** `_cut(lam, t, w, lo_b, hi, tol)` returns a cut, searching only gaps whose midpoint lies in $(\text{lo}_b, \text{hi})$: the known pairs are the $j$ below it, counted and with a leaf (steps 4 and 6). Here $w = \min(m, c)$ ($m$ = LOBPCG's block `info["block"]`, $c$ = `_chunk` of the slice block, 24 at $b$ = 6).
     - The cut is the midpoint of the largest relative gap $(\lambda_{i+1} - \lambda_i)/\lambda_{i+1} \ge 100\max(t_\text{top}, \text{tol})$ among the top $w + 1$ locked modes. $t_\text{top}$ is the maximum of their thresholds, not of all $D$: the lowest modes' round-off floors must not block it.
     - The tol term keeps the cut wider than the assembly's window chain ($100\,r_s$), so a slice mode rarely merges with a known column (which only shortens the in-place prefix, section 2).
     - The midpoint must lie below hi, since in band mode a locked mode can reach $\lambda_{hi}(1 + 100t)$.
     - At most $w$ locked modes lie above the cut, and the first slice recomputes them. A missed mode is most likely near the top, where a later chunk would have returned it.
   - **(b) Otherwise:** no qualifying gap with its midpoint in $(\text{lo}_b, \text{hi})$ (e.g. every mode below the band), or $D < 2$. The known pairs are every locked mode with $\lambda_i < \text{lo}_b(1 - 100\max(t_i, \text{tol}))$, $j$ of them, with no leaf and no count.
     - They are never returned, and completeness below $\text{lo}_b$ is not part of the band. $Z$ is exact for any $Z$, so none is dropped from deflation: a dropped mode costs 14–19 MINRES iterations per step at depth 500–1000.
     - With $j = 0$ (e.g. `from_zero`) there is no junction: LOBPCG's modes are freed as today.
3. **otf back to stored** (section 5).
4. **The count**, in case (a) only. Below $\text{lo}_b$ the known pairs are not returned, so their completeness is not part of the band.
   - `_verify` runs as on the "fits" path on `lam[:j+1]` and `V[:, :j+1]` (`count` needs every mode up to `lam_next` in $W$), with `n_band` = $j$, so it counts over [0, cut] with the first locked mode above the cut as `lam_next`.
   - It takes LOBPCG's `t_iteration`, block and solve time, with `reuse` and `probe_budget` as there and the final `op_opts`; `probe_budget`'s held host term is $8nD$ (all touched columns) plus the aggregates, not $8n(j+1)$.
   - When the slice counts do not fit (`no_count`), solve_modes prefixes `no_count` to `info["count"]["reason"]` instead of overwriting it, so a failed known leaf keeps its reason.
5. **Hand over**, with no jax op after the count (the rule at api.py:779–788): solve_modes builds `known = [lam[:j], V, r[:j], t[:j], leaf]` with the untruncated `V` and drops its own names (`V = None`), so si_band's list holds the only reference. No truncation: si_band's one resize sizes the result.
6. **The leaf.**
   - solve_modes passes `((lo_b, cut), count)` as `known`'s leaf (lo_b = the band's lower edge), with `count` in `_count_out`'s keys (`passed` and `verifiable` False by default, `estimate`, `estimates`, `flagged`, `restarts` 0, `ran`, `reason`, `t_leaf`). si_band records it (section 2).
   - A flagged count is not rerun: rerunning a span of $j$ modes would deflate more than any slice's capacity. The leaf stays flagged, as on the "fits" path.
   - This is a regression against today's capacity-stop path, which slices $(\text{lo}, \text{cut})$ again and would find a mode LOBPCG missed there. It is logged.
7. **si_band**, called with:
   - `known`, lo = max(lo_b, cut) (lo_b in case (b)), `edges` at or below lo dropped;
   - `max_deflate` = $d$ from `_slice_plan` (section 4), which already includes the user's cap.

   The store is $N_s = j + N_\text{new}$ with $N_\text{new} = \lceil 1.25\,\max(n_\text{modes} - L_j, D - j)\rceil$ when `n_modes` is given, else $\max(200, \lceil 1.25\,(D - j)\rceil)$ (the pure helper `_n_new(n_modes, L_j, D, j)`). $L_j$ is the number of known modes in the band, and $D - j$ (the recomputed locked modes) floors it, as today's $\max(n_\text{modes}, \lceil 1.25 L\rceil)$ does. On `edge_rr_failed`, `complete_to` becomes lo_b; otherwise it is si_band's (at least the cut).
8. **Info.**
   - `info["junction"] = {"cut", "known": j}`: `cut` None in case (b); `info["junction"]` None without a junction ($j = 0$, and the "memory" and forced "shift_invert" paths).
   - `from_zero` and `mu1_ub` keep their meaning, applied to lo_b.
   - `merged_edges` stays si_band's: the cut is not among its bounds (logged).
   - `stagnation_risk` = $D$ ≥ `STAGNATION_DEPTH` and (no junction or $d < j$). $j$ is a lower proxy: later items also deflate the band's own finds below σ.
   - The capacity-stop warning keeps "N modes locked". It says the band continues from the cut with shift-invert slices (case (b): "continues from lo with j locked modes known"). Under the risk it adds "MINRES may stagnate: deflation limited to $d$ of $j$ known modes" (or "no junction").

The "memory" path (no chunk fits) and forced "shift_invert" have no junction. Their slices from 0 deflate what earlier slices found, up to $d$. A band with $f_1 > 0$ that starts at $\lambda_{lo}$ deflates only the band's modes found below each σ (logged: the modes below $f_1$ are not computed for it).

### 4. Memory (`memory.py`)

$\lceil x\rceil_c$ rounds up to a multiple of $c$.

**`si_memory(..., n_z=0)`:** the columns of $Z$, distinct from `n_deflate` (the Lanczos window $W$ inside $Q$). When $n_z > 0$ it adds:
- device storage: $8n\lceil n_z/c\rceil c$ and $8 n_z^2$ (`H`);
- the setup phase: `op_memory(c)["vectors"]` + `op_memory(c)["workspace"]` + $8 n_z c$;
- the MINRES phase: two $n \times b$ blocks (the Z-only operator's temporaries) plus $n_z \times b$;
- host: $16 n_z^2$ ($E$ and $H$).

The storage and host terms are exact (`device_exact`, `host_exact`), the blocks modelled. The host store's `n_modes` must include the known columns, so on unified memory $Z$ costs twice: once in the store, once on the device.

**`_slice_plan(op_memory, pc64, n, *, n_store, n_new, block, n_rigid, budget, host_budget, max_deflate=0, capacity=None, host_extra=0)` → (C, d)**, where $n_\text{store} = N_s$ (the host store's columns), $d_\text{max} = \min(n_\text{store}, \texttt{max\_deflate})$ (the user's cap enters the plan, so `max_deflate=0` gives today's $C$) and $n_\text{new}$ the new modes (the need cap's count):
1. $d^*$ is the largest $d \le d_\text{max}$ fitting at the low capacity $C_\text{lo} = \min(\lceil 2\,\texttt{SLICE\_MIN}\rceil_c, C_0)$, by `_largest(fits, 1, 0, d_max)` (step 1, so $d_\text{max}$ need not be a multiple of $c$; $d = 0$ fits by $C_0$). $C_0$ is `_slice_capacity(..., n_z=0, n_new=n_new)` (so $C_\text{lo} \le C_0 \le$ the need cap); None raises the slice-floor error, as today.
2. $C$ is the largest capacity in $[C_\text{lo}, \text{need cap}]$ fitting beside $d^*$, capped at the need cap $\lceil 1.25\,(1.8\,n_\text{new} + 84) + 2c\rceil_c$.

Why deflation comes first:
- A capacity column and a deflation column cost about the same.
- Halving the capacity adds slices, each with the fixed 84 columns. Three slices of 288 cost about 20% more Lanczos columns than one of 720.
- An undeflated mode costs MINRES 6.5–12 iterations per step (the probes).

With a given `capacity`, $d$ is the largest fitting beside it; `_largest`'s None becomes $d = 0$. solve_modes calls only `_slice_plan`. `_slice_capacity` gains `n_z` and `n_new` (the need cap's count, default `n_store`). On the paths without a junction, `n_new` = `n_store` as today, and with $d = 0$ the plan gives today's $C$.

**The assembly:** in place (section 2), so the host peak stays the store plus one slice block.

At a junction, the touched locked columns above the cut, $8n(D - j)$, are passed as host extra to `_slice_plan` and `_count_block`.

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
- Kept although a reviewer proposed deferring it: the roadmap's 18f lever item asks for it (`ROADMAP.md:685`: "otf for the slices only when the stored slice operator does not fit"), and otf costs about +50% per deflated iteration.
- `_slice_plan` runs with both models ($d_s$, $d_o$: its $d$ with the stored and the otf model), with the AMG aggregates still held for the count (`_nbytes(aggs)`) as host extra. The operator is rebuilt stored when $d_s \ge d_o - c$; a stored plan that does not fit (the slice-floor error) keeps otf.
- Iterations follow the undeflated count $u + j - d$ ($u$: a slice's own modes below σ), not $d$. otf costs about +50% per deflated MINRES iteration (two applies, +32% each at 1M), so the rule allows at most one slice chunk $c$ of loss.
- The otf operator is freed first, and `op` and `op_opts` are rebound (`op_opts["otf"]` False) so that `make`, `reuse` and the count see them. The decision only picks the operator: the final `_slice_plan` runs after the count, without the aggregates' host extra (freed), so its $d$ is at least $d_s$. `op_memory` and the count block are recomputed with the stored model. The aggregates stay valid, since they depend on the mesh graph only.
- `info["levers"]` stays LOBPCG's choice.

**A LOBPCG rerun with levers: closed.** The junction keeps the locked modes, and the levers already raise the reach up front. A rerun would repay the depth already reached. Unknown depths with a fitting chunk still get no levers (logged).

### 6. Options and closures

- **`slice_options={"max_deflate": n}`** (SLICE_OPTIONS):
  - an int ≥ 0; bools are rejected;
  - solve_modes takes it out of the options forwarded to si_band and passes it to `_slice_plan`, whose $d$ goes to si_band;
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
   - **Accuracy.** $Z$ is the eigsh modes below σ. MINRES at tolerance $10^{-12}$ gives $x$ to `np.linalg.solve(K - σM, b)` within $10^{3}\kappa\cdot10^{-12}$ (κ = cond₂(K − σM) on the free DOFs). At $10^{-8}$ the true residual is ≤ $1.01\,\tau$, both for that $Z$ and with $10^{-3}$ noise added (any $Z$ is exact).
   - **Any symmetric $H$.** With a random symmetric $H$ in place of $E^+$, through the real recombination: $b - Ax = P_Z b - P_Z A\tilde x$ to $10^{-12}$ relative.
   - **Duplicates:** an exact duplicate column ($E$ singular), a dropped near-duplicate (δ = $10^{-6}$, $|H|$ of order 1) and a kept one (δ = $10^{-3}$, $|H| \gg 1$): each leaves the true residual ≤ $1.01\,\tau$, and on the exact and dropped cases $H$ is finite with $HEH \approx H$ and $EHE \approx E$ (relative $10^{-10}$; `inv` fails these). $H$ is symmetric when $E$ is slightly nonsymmetric.
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
   - the selection is a small pure helper `_z_cols(lam_s, sigma, n)` (indices of the nearest $n$ with $\lambda < \sigma$), unit-tested with λ just below, exactly at and just above σ, so `<` against `≤` and above-σ mutations fail; the si_band level checks only that the columns are store views and the `max_deflate` cap;
   - the columns are views of the store;
   - `max_deflate` = 3 gives the 3 nearest, 0 (the default) gives none, None is rejected;
   - `Z` in `slice_options` raises.
4. **`known` in si_band** (fakes and `test_merge`'s `two_runs` setup, test_shift_invert.py:1899):
   - the known pairs come back first in `lam`, with LOBPCG's residuals and thresholds;
   - the first slice's `lam_W` (recorded λ only) holds the top known λ, so the known windows close its lower side;
   - `known` is empty after the call;
   - the leaf (also a flagged one): `slices[0]` is the known entry, and `info["count"]` covers it on the normal exit and in `err.result`;
   - `err.result` after the first slice raises holds the known pairs, assembled in place (`id`);
   - **in place:** `id(V_f)` = `id` of the given block, with no `RuntimeWarning`, and every column of `V_f` equals the expected vector (known, then the slices', as `assert_compacted`); one in-place case ends on a `break` path (a non-converged slice; `id` kept, no warning), one in-place case has slice parts wider than the given block plus $c$, so at least two grows after the probe run (same `id`, no warning, contents as `assert_compacted`);
   - **fallbacks, each copying with the warning and the same contents:**
     - a second reference held (found by the probe grow before any write: the holder's array unchanged);
     - a borrowed block (`big[:, :D]`);
     - a known index part after a merged window (crafted);
   - **shortened prefix, still in place** (no warning, `id` kept, contents as `assert_compacted`): a slice that re-finds the top known mode, a widened merge that pulls in the top known-only window, and $p$ = 1 ($j$ = 2, the top re-found) and $p$ = 0 (the lowest known window swallowed);
   - `fake_band` (`test_shift_invert.py:1175`) is updated where the `max_deflate` or `known` kwarg reaches it, and records `len(Z)`, not `Z` (views would defeat the in-place path).
5. **`_cut`** (crafted λ, t, $w$, lo_b, hi, tol):
   - the largest gap in the top $w + 1$, with a larger gap just outside that window ignored (off-by-one);
   - the relative-gap denominator ($\lambda_{i+1}$) and the exact $\ge 100\max(t_\text{top}, \text{tol})$ boundary (both terms deciding a case);
   - a large $t$ on a low mode not blocking the cut;
   - a midpoint at or above hi excluded, the next-largest qualifying gap taken;
   - $w$ = 5 and $w$ = 10 picking different gaps on one spectrum (a wrong $w$ fails);
   - a larger gap with its midpoint at or below lo_b ignored (the next above lo_b taken, else None);
   - dyadic $t$ ($2^{-9}$) for the exact boundary;
   - None for no gap and for $D < 2$.
6. **Junction end to end** (`test_capacity` updated: "short", chunk 5, $D$ = 40 (35 locked and the kept chunk) inside the band 25–45):
   - si_band receives `known` with the $j$ pairs below the cut (the 0.0898 gap at the bottom of the top $w + 1$ = 11, so a window one too small picks 0.0427 and fails (test 5 covers one too large); 10 modes above it = $w$; $j$ = 30, asserted from `_cut`; $N_s$ = 30 + ⌈1.25 max(22 − 5, 10)⌉ = 52) and lo = the cut;
   - the LOBPCG `V`'s `id` is si_band's known block, si_band gets $j$ known pairs, and the returned `V`'s `id` is the same (one resize in si_band, then solve_modes' compaction in place). `test_capacity`'s own `solve` spy drops `seen["V"] = weakref.ref(...)` and `v_dead` for these `id` checks;
   - si_band's `max_deflate` equals `_slice_plan`'s $d$;
   - `aggs_dead` still holds at si_band's entry (the count ran before it);
   - the known leaf is `slices[0]` and passes, and `info["count"]` covers it;
   - $N_s = j + \lceil 1.25 \max(22 - L_j, D - j)\rceil$ (computed in the test);
   - `info["junction"]`;
   - `seen["interval"]` = (cut, hi);
   - the band against eigsh (`check`), with every returned $\lambda \ge \lambda_{lo}(1 - 100t)$ and $V$ holding exactly the band's columns.

   Updated for the junction:
   - `test_capacity_store`: interval (cut, hi) with the cut in the gap above the 20th locked mode; $j$ = 20 and $D - j$ = 10 (the `_slice_plan` spy and a `_count_block` spy assert `host_extra` = $8n \cdot 10$), so `n_modes` 5 gives $N_s = 20 + 13 = 33$ (interior and zero); new cases: `n_modes` 30 ($N_s = 20 + \lceil 1.25\max(30 - L_j, 10)\rceil$, $L_j$ asserted per case) and none ($20 + 200$); `_n_new` alone: none with $D - j$ = 10 → 200, with $D - j$ = 400 → 500, and the `n_modes` floor.
   - `test_error_result_capacity` (:915): $N_s$ from $j$; max_slices = 1 still raises, now with the first slice at the cut. Its `check_partial` $10^{-8}$ M-orthogonality between LOBPCG and slice modes has a measured margin of about 3× (3.7e-9 at tol $10^{-8}$), noted in the test.
   - `test_filter_capacity_sorted` (:840): λ becomes [1, 5, 1.2, 1.4] with lo = 1.1, hi = 9 (gaps 0.167, 0.143, 0.72): cut 3.2, and the count runs on the sorted `lam[:j+1]` = [1, 1.2, 1.4, 5] (`n_band` 3, recorded by the harness's `_verify` stub beside `lam` and asserted; `lam_next` = 5); the existing `not rec["verify"]` assert flips. With per-mode $t$ = [1e-8, 2e-8, 3e-8, 4e-8] and the stub_band spy copying what it receives: `known`'s `V[0, :3]` == [1, 3, 4] (the harness fills column $i$ with $i + 1$), thresholds [1e-8, 3e-8, 4e-8] and the residuals in the same order (a skipped `_permute` when stopped fails).

   Other cases:
   - **No gap:** `harness` with λ spaced 0.5% apart and $t = 10^{-3}$: case (b), $j$ = #($\lambda < \text{lo}_b(1 - 100\max(t, \text{tol}))$) asserted (a mutation to `< lo_b` or to tol only fails); with `from_zero` ($j = 0$) `info["junction"]` is None and the band runs from 0.
   - **Case (b)** (`capacity_stop("short", locked)` with the real si_band and `fake_band` recording `len(Z)`; `_slice_plan` monkeypatched to $d \ge D$), two cases: all locked modes below the band, and exactly one locked mode $D-1$ inside it with f1 chosen so that lo lies in $((\lambda_{D-2} + \lambda_{D-1})/2, \lambda_{D-1})$ ($j = D - 1$, mode $D-1$ recomputed and asserted). The case-(b) warning text is matched. Common asserts: `info["junction"]` = {cut: None, known: $j$}, $j$ = the modes below $\text{lo}_b(1 - 100\max(t_i, \text{tol}))$ (asserted per case), no leaf, `_verify` not called, si_band from lo, and the first item's `len(Z)` = $j$.
   - **A flagged leaf** (`_verify` stubbed to estimate 0.6): `_verify` called once (no rerun) and the leaf the stub captured is flagged; si_band's recording of a flagged leaf into `info["count"]` is test 4's.
   - **`edge_rr_failed`** (stub_band): `complete_to` = lo_b.
   - **`no_count`** (`capacity_stop` with the real si_band and `fake_band`, its own `api._verify` stub returning `passed=False` and `reason="faked"`, `_count_block` → None, `verify=True`): the reason starts with `no_count` and contains `_band_count`'s reason for the known leaf, built in the test from lo_b and `info["junction"]["cut"]` in `_band_count`'s format with `": faked"` (the known leaf first, `_count_out`'s keys filled by solve_modes).
   - **`probe_budget`** (test_capacity_store, a `mem.band_probe_block` spy): the held term is $8nD$ plus the aggregates.
   - **edges** (`slice_options={"edges": [x_below_cut, x_above_cut]}`): si_band receives only the one above the cut.
   - **No-junction paths:** `info["junction"]` None in test_memory ("memory") and test_slice_reason (forced "shift_invert"), and si_band's `max_deflate` equal to `_slice_plan`'s $d$ (test_memory's spy).
7. **Memory:**
   - `test_memory_formulas` (`test_shift_invert.py:588`) gains the exact `n_z` terms and `device_exact` / `host_exact`.
   - `_slice_plan` on crafted fit functions, checked against a brute-force search over $(C, d)$ as `test_slice_capacity` does:
     - full deflation;
     - $d^* < n_\text{store}$ at $C_\text{lo}$;
     - $d_\text{max}$ not a multiple of $c$ (e.g. `max_deflate` 37 with $c$ = 16): $d$ = 37 when it fits;
     - $C_0 < \lceil 2\,\texttt{SLICE\_MIN}\rceil_c$;
     - a need cap below $\lceil 2\,\texttt{SLICE\_MIN}\rceil_c$ ($n_\text{new}$ = 5, $c$ = 24: cap 168): $C$ = 168;
     - a given capacity, with None → $d = 0$;
     - $d = 0$ equal to today's $C$;
     - $n_\text{new} < n_\text{store}$ with the budget above the need cap: $C$ = the $n_\text{new}$ cap;
     - `max_deflate=0` through solve_modes giving today's $C$.
   - **A process peak** (slow, `peak`, macOS, PEAK_SETUP style in a subprocess; up to 4 Lanczos steps at the 72-column floor, 30–40 s each: about 3 min):
     - one si_slice step on voxel $48^3$ / GMG (353k DOF) with `Z` = 800 random masked columns, held as a host block; the slice lies below $\lambda_1$ (so $A$ and $E$ are SPD), with `maxit_inner` 200 and a loose `tau` (the peak, not convergence, is measured);
     - the peak is within 20% of `si_memory(..., n_modes=800, n_z=800)`'s total (about 4.6 GB plus the setup);
     - the sensitivity: a second subprocess with `Z=None` and the same setup (and the same 800-column host block); $0.8\,\Delta \le \text{peak}_{800} - \text{peak}_0 \le 1.2\,\Delta$ with $\Delta$ = `si_memory(n_modes=800, n_z=800)` − `si_memory(n_modes=800, n_z=0)` totals (about 2.3 of 5.3 GB); about 6 min for both. It is run beside the other peak tests in implementation, and marked `serial` if loaded readings drop (as `test_mac_count_peak`);
     - the capacity is at its floor;
     - a weakref on si_slice's device `Zd` is dead after it returns (consecutive items and restarts re-upload it).
8. **Levers:**
   - `test_choose`'s rule-2 cases expect rule 3's pick (the cheapest within 5% of the best reach);
   - `test_levers_unmargined`'s 1.2e8 sub-case (:1422; today rule 2 / "memory"): the expected state, warning and `reason` computed with `state_reach` as the other lever tests ("capacity" if rule 3 has a state with a fitting chunk, else "memory");
   - new (from open_items :321): the "Rule 2 can forgo levers" case (slow, voxel $10^3$, `n_modes` 12, the tight budget from a `state_reach` bisection) gets rule 3's pick and returns all 12, checked against eigsh;
   - **otf back to stored** (fake builds, `_choose` monkeypatched to otf, a capacity stop):
     - a second operator build without `mode` when the stored plan qualifies, with the otf operator dead (weakref on the operator, not on V) before it;
     - none at $d_s = d_o - c - 1$, and a rebuild at $d_o - c$ (the boundary); the `_slice_plan` stub tells the models apart by `op_memory(6)` (as `test_levers_applied`'s `cap` spy) and the call order is asserted;
     - a stored plan raising the slice-floor error: otf kept, one build;
   - `test_levers_applied`: the rebuilt build and the capacity recorded after it (`builds`, `caps` updated);
   - `slice_options={"max_deflate": n}` reaches `_slice_plan`, and si_band receives its $d$: 0, a cap below the memory's $d$ (the cap) and one above (the memory's $d$); a bool, a non-int or a negative value raises.
9. **`stagnation_risk`** (`test_stagnation_risk`, :2308; `_slice_plan` monkeypatched):
   - depth 1000 with $d < j$: True, and the warning names "deflation limited to";
   - $d = j$: False;
   - `max_deflate` below $j$ (through the real `_slice_plan`'s cap): True;
   - no junction (f1 = 0 and $t = 10^{-3}$: no qualifying gap, $j = 0$, `info["junction"]` None): True;
   - all below the band, $d \ge D$: False;
   - depth 999 with $d < j$: False.
10. **Deep band** (slow, about 3 min target, measured in implementation): hex20 $10^3$ clamped, `chunk=40` pinned, `levers=False`, a budget that capacity-stops LOBPCG near depth 250 with $D$ (including the kept chunk, up to 40 modes) below $\text{lo}_b(1 - 100\max(t, \text{tol}))$, $D$ asserted before the case-(b) asserts, the band from mode 260 to 300, against eigsh at the band's centre ($k$ = 60):
    - every band mode is found, and the leaves are counted (`verify=True`);
    - case (b): `info["junction"]` = {cut: None, known: $j$}, $j$ asserted;
    - each slice's `deflated` > 0 (the selection itself is pinned by test 3, the cap by tests 7 and 8), and the first slice's deflated/plain MINRES iteration ratio printed.
11. **Changed spies:**
    - `test_memory` (3 cases), `test_fp32_slices` and `test_memory_bracket`: a spy on `_slice_plan`'s returned $(C, d)$; `info["capacity"]` and the count's window are asserted against that $C$, and $C_0 \ge 72$ separately;
    - `stub_band` (and through it `harness`) returns `slices=[]`, `merged_edges=[]` and `edge_rr_failed` (False, True for the `edge_rr_failed` case), for the junction's tests (test_capacity_store, whose `_slice_capacity` spy becomes the `_slice_plan` spy, test_filter_capacity_sorted, test_stagnation_risk, test_levers_applied, test_levers_unmargined, test_levers_rows);
    - `_slice_capacity`'s `n_new` defaults to `n_store`, so its direct calls (`test_strategy.py:480, :1613, :2471`) stay unchanged;
    - `test_options.py:553–560`: `band_set` gains `known`, `slice_set` gains `Z`; `max_deflate` is a SLICE_OPTION and an si_band parameter, so the existing equality pins it unchanged.
12. All other band, strategy and shift-invert tests pass unchanged, except the cases named above (each named in its commit).

## Benchmark (18f)

`benchmarks/deep_band_study.py` (results `benchmarks/results/deep_band_study.json`, per-step progress logged to a file, `--parts band,solves`). Each part is under 1 h, and the estimates come from the plan review measurements.

- **`band`** (about 45 min, run after `solves`): hex20 $16^3$, `chunk=40`, `verify=False` (eigsh is the acceptance check; the counts' predicted degree and gap recorded, as si_band does under `verify=False`).
  - $D$ lies on a chunk boundary (at chunk 40 on this mesh: …, 924, 964, 1005, 1045; strategy_study), and $C_\text{max}$ is a multiple of $c_L$ (64 here, block $k$ = 48). The script predicts $D$ from the cached run's `info["chunks"]`: with $C_\text{max}$ = 960 the lock at 924 fails ($924 + 48 > 960$) and the kept chunk gives $D = 964$. The band part runs with `levers=False` (levers change $k$, $c_L$ or the boundaries); the budget is picked through `_c_max` for that $C_\text{max}$, the predicted $D$ and cut (from the cached λ and thresholds, cut > lo_b) and `_slice_plan`'s $d \ge j$ (full deflation) checked before LOBPCG runs (about 8 min), and $D$ and the cut asserted again after it (chunk outcomes at the smaller $C_\text{max}$ may differ from the cached run's). lo is about 12 modes below $D$ (mode ~952), hi at the next boundary (mode ~1005, within the cached reach).
  - Before the slices, the study asserts the capacity stop, $D \in (\text{lo}, \text{hi})$ and cut > lo_b, so the cut path is measured. (Test 10 covers case (b) and asserts `info["junction"]` = {cut: None, known: $j$}.)
  - The slices from the cut: about 55 modes with at most about 12 recomputed (the locked modes between lo and $D$), about 180 columns, about 30 steps of about 50 s (about 25 min).
  - Recorded per slice: MINRES iterations per step, $n_z$, $d$ against $j$, σ moves, `inner_maxit` hits, time.
  - The frequencies are checked against eigsh at the band's centre ($k$ = band width + 20, `maxiter` set; a few minutes).
- **`solves`** (about 20 min): first MINRES solves with $Z$ from a LOBPCG run at chunk 40 to the 1045 boundary cached to disk (about 8 min, rerun only without the cache; its chunk boundaries also place the `band` part), at depths 500 and 1000:
  - deflated;
  - with the nearest half of $Z$ (the memory-bound case; about 5 min at depth 1000);
  - the deflation's per-iteration overhead.

  The undeflated references are cited, not rerun: 4960 iterations at depth 503 (18e probe) and 14,370 at depth 1005 (`shift_invert_study` `recheck_1005`).
- **Acceptance:** on the `solves` part at `recheck_1005`'s σ and right-hand-side width, with $u$ = 0 (every mode below σ deflated): deflated ≤ 0.1× its 14,370 iterations, with no column stagnated and the explicit $\max\|b - Ax\|/\|b\| \le 1.01\,\tau$ (test 1's bound, computed as `shift_invert_study.inner` records it); the solve reuses `shift_invert_study`'s `setup` (AMG at σ), `rhs` (6 columns, $Mq$), τ = $10^{-8}$, maxit 20000 and the stored σ 452.388, so only the deflation differs (the probe: 0.022 at $u$ = 0, 0.075 at 40, 0.11 at 80; the half-Z case reported); the band against eigsh.

## Docs (18f)

- `docs/eigensolvers.md` (the deflation cap is written $d_Z$ there: $d$ already means the count degree at `:119` and the depth at `:815`):
  - `:114` (band mode, Capacity): a capacity stop keeps the stopping chunk (written to `V`, not locked);
  - Interior bands and strategy:
    - `:115` (`stopped`): a capacity stop continues from the junction cut with slices;
    - `:141`: the capacity stop → the junction, and the `stagnation_risk` definition;
    - `:145–153` "Why memory-driven" (including "No crossover exists"): the slice per-mode row and the crossover claim marked undeflated, with the study's figure;
    - `:157`: "33–1260× cheaper" marked undeflated, with the study's figure;
    - `:166`: rule 2 removed (unreached known depths take rule 3's pick), and the residual risks: otf back to stored before the slices (when $d_s \ge d_o - c$), and the "until 18f's junction" clause resolved;
    - `:182`: the slice capacity → `_slice_plan` ($C$ and $d$), the need cap from $n_\text{new}$, `n_z` in the fit;
    - `:175`: the 10M reach discussion gains the deflation cap $d_Z < D$;
    - `:183`: the store $N_s$;
    - `:184`: at a junction the count block also leaves room for the $8n(D - j)$ locked columns above the cut;
    - `:757`–`:759` (si_checked): `Z` passed through, kept on restarts (same σ), $H$ recomputed per run;
    - `:187`: `from_zero` and `mu1_ub` after a junction;
    - `:193`: the filter, sort and compaction after a junction;
    - `:197` Errors: forced "lobpcg" for an unreached known depth, the `slice_options` key `max_deflate`, the discarded LOBPCG run → kept;
    - `:199`: `info["junction"]`; after a junction `info["count"]` is si_band's, including the known leaf, and `no_count` is prefixed to its reason.
  - Options table: `slice_options` `max_deflate`.
  - Shift-invert slices (`:716–753`): the MINRES deflation ($Z$, distinct from the Lanczos window $W$ and from $R$), `Z` in the argument table (`:740–753`), `info["deflated"]` in the info list, and the exactness and projector arguments with the two preconditioned operators written out. Unlike `lobpcg`, which stores $R$ and $MR$, the Z-only form stores no $AZ$.
  - Slicing a band (`:765`, the signature at `:767` with `known=None, max_deflate=0`, `:798`): $Z$ per item, `known`, `max_deflate`, the in-place assembly, the "known" slice kind and the known leaf in `count`.
  - Slice memory model (`:802`, the signature at `:804`): `n_z=0`, `_slice_plan`, the 1M table.
  - Measured MINRES cost:
    - `:831–832` ("Depth sets the cost", the $\alpha D(1 + D/D_0)$ model): marked undeflated, before PR 18f;
    - `:835` "Depth 1000 is marginal" → the deflated study figures;
    - a `deep_band_study.py` paragraph beside `:815` (script, `--parts band,solves`, results JSON, setup).
  - Test sections: `:997` (a capacity stop handing the whole band to si_band → the junction), `:1002` (the stagnation warning, new rule), `:1004` (the `err.result` tests: the known pairs first, assembled in place), `:999` (`_solve(max_capacity)`: the stopping chunk kept, written to `V` and M-orthonormal against the locked ones, the columns past it zero), `:1003` (rule 2 at $n_\text{modes}$ 150 → the new rule), plus the new tests.
- `README.md`:
  - the status sentence: "split into 18a–18h", 18f's status, the order;
  - the interior-band paragraph (`:287`): the junction and the deflation, with the "33× … 1260×" ratio marked as undeflated (or replaced by the study's).
- `docs/ROADMAP.md`:
  - the status paragraph `:18` (the PR 18 split list → 18a–18h: 18f deep bands, 18g slices, 18h CalculiX), and `:626` ("seven parts … 18g last" → eight parts, 18h last);
  - the 18f section rewritten to its contents, with the acceptance as run;
  - a new 18g (slice efficiency, plus a slice's own modes below σ), and the old 18g → 18h;
  - the dependency rows and the order;
  - `:91`, `:92` (33×/1260×, no crossover): marked undeflated, with the study's figures;
  - `:94` (M3b's 10³–10⁵ MINRES iterations per step): marked undeflated, with the deflated study figure;
  - `:95` (M3b: the depth-1000 mid-gap solve, now with the deflated figure);
  - `:808` (the PR 15d GPU check "MINRES near depth 1000" → deflated, with the cap);
  - deferred GPU checks:
    - the deflation cap $d_Z$ at 10M DOF, where $Z$ costs 84 MB per column (twice on unified memory), so $d_Z < D$: partial-deflation MINRES near depth 1000;
    - fp32 $Z$;
    - the deflation overhead at 10M.
- `docs/open_items.md`:
  - "Shift-invert cost near mode 1000": stays in place, annotated "PR 18f: resolved on the Mac (the study's figures); the deflation cap $d_Z$ at 1M and 10M stays open (the zero-copy lever)";
  - `:175` (remedies → PR 18f): "PR 18f: implemented; at 10M $d_Z < D$";
  - `:342` ("about 4 h per slice step at 1M"): annotated as undeflated;
  - "σ-move retries" (`:344`, body `:346`): by the study's result;
  - `:155` ("Memory-driven strategy, no time-based crossover … 33× … 1260×"): annotated "PR 18f: the ratios are undeflated; the deflated figures are in deep_band_study";
  - the "Slice efficiency (PR 15c → PR 18e → PR 18f)" heading at `:332` and the bodies `:336–338`, `:375` ("→ PR 18f") → 18g; the headings at `:340` and `:344` by the study's result;
  - annotated in place with "PR 18f: …":
    - `:156` (the discarded LOBPCG run);
    - `:172` (the old $N_s$);
    - `:183` (locked modes discarded);
    - `:313` (the 18e levers decision: rule 2, otf carried into the slices);
    - `:321` ("Rule 2 can forgo levers": resolved by removing rule 2, annotated in place);
    - `:312` (the 18e split: the slice items → 18g, and the order) and `:367` (the order);
  - a "PR 18f decisions" section (section 7 below).
- **Catch-all:** before the merge, `git grep -n -e 18f -e 18g -e CalculiX -e "capacity stop" -e freed -e discard -e "rule 2" -e 1260 -e crossover -e stagnation_risk -e STAGNATION_DEPTH -e N_s -e n_store -e "depth 1000" -e "chunk dropped" -e "dropped unsorted" -- docs README.md src`, with every hit checked against the new behaviour.
- Docstrings and comments: `lobpcg._solve` (`lobpcg.py:567`, "the stopping chunk dropped") and the comments at `lobpcg.py:666` ("chunk dropped") and `api.py:648` ("dropped unsorted"), `si_slice` (`Z`), `si_band` (`known`, `max_deflate`, the assembly), `solve_modes` (the junction, `max_deflate`, `info["junction"]`), `_choose`, `si_memory` (`n_z` against `n_deflate`), `_slice_capacity`, `_slice_plan`, `_cut`, `_dminres`.
- Memory:
  - MEMORY.md: the autonomy line (18f merged, next PR 18g), the scale line's order with 18h, and the CalculiX line → 18h;
  - `feedback-autonomy.md`: a PR 18f entry (plan and code review rounds, merge);
  - `project-calculix-benchmark.md`: the description and body → 18h, with the renumbering;
  - `project-scale-questions.md`: the order.

## 7. Risks and decisions without user input (logged in `docs/open_items.md`)

- The split (18g slice efficiency, 18h CalculiX) and the order.
- A capacity stop keeps the stopping chunk (converged and confirmed, not locked), so $D$ includes it and the slices do not recompute it.
- The A-orthogonal deflation in its Z-only form: exact for any $Z$ and symmetric $H$, at half the memory of the $AZ$ form, for one $A$ apply per iteration (+7.5–11%).
- Deflation before capacity in `_slice_plan` (at $C_\text{lo}$ (section 4)), and the nearest modes first when memory limits $d$.
- The two cases: a cut above lo (the known pairs end there; at most $w$ locked modes recomputed), else every locked mode below $\text{lo}_b(1 - 100\max(t_i, \text{tol}))$ known (none dropped from deflation); the known leaf without a rerun when it flags (a regression against re-slicing $(\text{lo}, \text{cut})$), and no junction for $j = 0$.
- `merged_edges` without the cut.
- A `from_zero` band with no qualifying gap among the top $w+1$ modes has $j = 0$ and loses all $D$ locked modes (sliced again from 0).
- Rule 2 removed (rule 3's 5% band for unreached known depths); otf back to stored within one chunk of $d$; the LOBPCG rerun closed; no levers for unknown depths with a fit.
- The σ-move retry policy decided by the study.
- **Residual risks:**
  - $f_1 > 0$ bands without LOBPCG deflate only their own modes;
  - at 1M the junction comes at $D \approx 2000$–2700 with $d < D$ for large $N_\text{new}$ (the zero-copy lever, deferred);
  - at 10M the deflation is memory-capped (a deferred GPU check);
  - the iterations still grow with depth through a slice's own modes below σ (→ 18g);
  - a deflated MINRES iteration costs more (two applies and the $Z$ update: about 1.4× at $d$ = 1000, 1.9× at 2600 at 1M), while `t_leaf` shrinks. `counting._bound` stays a lower bound only through its 3.5× slack for slices, and `verify=None` skips more counts. This is documented, and the deep-band test runs with `verify=True`.
- The roadmap's acceptance "depth-500 and depth-1000 bands against eigsh": the depth-1000 band runs end to end in the study, the test at about depth 260 (the suite's time), and depth 500 as single solves.
