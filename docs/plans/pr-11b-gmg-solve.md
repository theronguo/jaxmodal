# PR 11b plan: GMG in `solve_modes`, σ pre-estimate, fp32 V-cycle, estimator, Mac acceptance

PR 11b completes PR 11 (M3) on top of 11a's `gmg(op, free, sigma)`. It makes the voxel GMG the default preconditioner of `solve_modes`. It also adds:

- a σ pre-estimate that works on slender and high-contrast bodies (`docs/open_items.md`);
- an fp32 V-cycle;
- the GMG memory estimator and a one-call `solve_modes_memory`;
- the 2M-DOF Mac acceptance;
- the deferred GPU checks.

Roadmap tests in this PR:

| roadmap test | here |
|---|---|
| σ pre-estimate within 2× of λ₁; a user hint skips it | tests 1, 2 |
| `solve_modes` on an odd voxel mesh matches `eigsh` (PR 9 tolerances) | test 3 |
| fp32 GMG: same tol as fp64, ≤ 20% more LOBPCG iterations | test 6 |
| Mac acceptance: 2M DOF, 10 modes, tol $10^{-6}$, ≤ 10 min, peak within 20% of the estimators, fp64 and fp32 V-cycles | test 10 |
| deferred GPU checks | ROADMAP list (no test) |

## Measured facts (prototypes in `scratchpad/p11b/`, reviewer probes in `scratchpad/r11b_corr1/`, `rev11b_p1/`, `rev11b_perf/`, `c11b-plan1/`; jax 0.11.2, M3 Max CPU)

Steel, unit box, `xmin` clamped unless noted. λ₁ is the SciPy reference (`benchmarks/gmg_study.lam1`).

**σ sensitivity.** LOBPCG with GMG built at σ = f·λ₁ (10 modes, tol $10^{-6}$), iterations:

| case | f = 0 | 1e-3 | 0.3 | 1 | 3 | 10 | 100 | 1e4 |
|---|---|---|---|---|---|---|---|---|
| box $16^3$ | 15 | 15 | 15 | 15 | 16 | 17 | 30 | 234 |
| free box $16^3$ (6 rigid modes deflated) | refused | 17 | 19 | 20 | 23 | 32 | 95 | not conv. |
| plate $32^2\times1$ | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 32 |
| beam $64\times2\times2$ | 15 | 15 | 15 | 15 | 15 | 15 | 16 | 19 |
| aniso $h_z = h/10$, $32^3$ | 109 | 108 | 109 | 108 | 109 | 108 | 109 | 100 |

- Any σ in $[0, 3\lambda_1]$ is as good as $\lambda_1$; σ ≥ 10λ₁ costs iterations on bulky and free bodies.
- `sigma_estimate` (Jacobi) is up to 3800× above λ₁ on thin plates, slender beams, anisotropic grids and SIMP fields, and within 2.5× on small plates and stubby beams (11a study). At 2M DOF it takes 73 s (252 iterations), more than the solve.
- Capped at 3 or 10 iterations it is $10^2$–$10^7$× above λ₁.

**GMG-refined pre-estimate** (the rule below; probes `rev11b_p1/sigplan.py`, `r11b_corr1/sigp.py`, `c11b-plan1/sigv*.py`):

| case | start σ/λ₁ | σ/λ₁ after each round | LOBPCG iterations |
|---|---|---|---|
| box $8^3$ | 0 | 1.00 | 4 |
| odd $9\times8\times8$ | 0 | 1.00 | 4 |
| plate $32^2\times1$ / $64^2\times1$ | 0 | 1.00 / 1.00 | 14 / 20 |
| beam $64\times2\times2$ / $128\times1\times1$ | 0 | 1.00 / 1.00 | 9 / 20 |
| aniso $h_z = h/10$, $16^3$ | 0 | 1.00 | 20 |
| SIMP field (checked-in), $32^3$ | 0 | 1.00 | 20 |
| clamped beam with a 200× tip nodal mass | 0 | 1.00 | 6 |
| free box $8^3$ | 1.82 | 1.00 | 6 |
| free plate $16^2\times1$ | 81 | 1.00, 1.00 | 11 + 7 |
| free beam $128\times1\times1$ | $5.1\cdot10^4$ | **2.7**, 1.00 | 40 in total |
| free aniso $h_z = h/10$ | $8.9\cdot10^3$ | **6.83**, 1.00, 1.00 | |
| free SIMP field | 277 | 1.05, 1.00 | |
| two disjoint free boxes (12 rigid modes) | 1.75 | 1.00 | |
| clamped body + floating part (6 rigid modes) | 14 | 1.00, 1.00 | |
| unfiltered binary SIMP, $32^3$ | 0 | 390 | 20 |

- From σ₀ > 0 one round is not always enough: the free beam and the free anisotropic grid land at 2.7 and 6.8 λ₁ after round 1. Test 1 includes the free beam, so dropping the rounds fails it.
- The binary field's λ₁ is a void-region mode that 20 iterations from a random start miss (100 find it). 10-mode LOBPCG does not converge on that field at any σ either (below). It is a documented limitation, not a test case.
- **Cost at 2M DOF (clamped, one round):**
  - `gmg(op, free, 0.0)` 5–6 s cold;
  - 5 LOBPCG iterations 8–9 s cold (6.6 s warm);
  - 13–15 s in total, against 73 s for Jacobi `sigma_estimate`.
- **Free bodies** add `sigma_estimate(maxit=10)` and up to 3 more rounds (build 2–6 s plus ≤ 20 iterations at about 1.3 s each). The worst case is about 2 min at 2M.

**Mac acceptance prototype** ($87^3$, 2.04M DOF, 10 modes, tol $10^{-6}$, 9-block step, fp64 basis, cold process, wall time including compilation; `big2.py`):

| V-cycle | σ pre-estimate | final GMG build | LOBPCG (iterations) | total |
|---|---|---|---|---|
| fp64 | 13.2 s | 2.0 s | 42.9 s (16) | 58.8 s |
| fp32 | 13.2 s | 2.3 s | 38.7 s (16) | 54.9 s |

The budget model assumed $N_\text{it}$ = 150 and $c_V$ = 5.5, about 8 min (`docs/eigensolvers.md`). Measured: $N_\text{it}$ = 16, $c_V$ = 5.3 (fp64) and 4.3 (fp32, corrected prototype; the fp32 run above used an fp64 $m_e$). The 10-min budget stays as the roadmap fixes it, and the model's table gets the measured numbers.

**V-cycle, warm, best of 5** ($87^3$, $m = 15$; `rev11b_perf/fp32.py`):

| V-cycle | time | temporaries |
|---|---|---|
| fp64 | 1.76 s ($5.3\,t_A$) | 12.11 fp64 blocks |
| fp32 | 1.41 s ($4.3\,t_A$; −20%) | 7.80 fp64 blocks |

- The fp32 $A_P$ apply alone takes 0.19 s against 0.33 s.
- These fp32 figures use the corrected prototype: the k = 1 smoother mask is cast to fp32, and the consistent $M_e$ is cast to the vector dtype. With an fp64 $M_e$ the mass GEMM ran in fp64: 1.48 s and 8.45 blocks.
- The 11a docs' 1.92 s (mean of 3, `gmg_study.json` "large") and this best of 5 measure the same thing. The implementation re-measures the "large" entry with best of 5 and quotes one method throughout.

**Peak memory** (`harness.measure`, warm subprocess, `lobpcg(..., maxit=20)` after the build, $87^3$, block 15; the GMG solve converges at 16, so this is the whole solve; repeated runs):

| preconditioner | measured peaks | estimate |
|---|---|---|
| Jacobi | 4.53–4.92 GB | 5.27 GB (`lobpcg_memory` today) |
| GMG fp64 | 6.12, 6.73 GB | about 6.83 GB (+1.5% to +12%) |
| GMG fp32 (corrected prototype) | 5.05, 5.08, 5.09 GB | about 5.77 GB (+14%) |

- The projected estimates are the Jacobi estimate plus GMG storage (0.31 / 0.16 GB) plus the extra step blocks with GMG. Those come from `memory_analysis` of the step with GMG (`p11b/stepmem.py`):

  | step | Jacobi | GMG fp64 | GMG fp32 |
  |---|---|---|---|
  | 9-block, $87^3$ | 16.00 | 21.11 (+5.1) | 17.43 (+1.4) |
  | 9-block, $64^3$ | 18.82 | 23.51 (+4.7) | 19.83 (+1.0) |
  | low-memory, $87^3$ | 9.42 | 19.11 (+9.7) | 14.94 (+5.5) |
  | low-memory, $64^3$ | 11.83 | 21.51 (+9.7) | 18.37 (+6.5) |

  These are temporaries plus aliased state, in fp64 blocks of $(n, 15)$.
- A stencil-specific workspace factor $f$ would not tighten the Jacobi part: the fitted $f$ is 0 at $87^3$, 0.7 at $64^3$ and 2.2 at $32^3$, so the envelope stays about 2.2–2.5.
- The roadmap band, $|\text{est} - \text{peak}| \le 0.2\,\text{peak}$ (as in the PR 6 and PR 9 tests), allows 5.39–8.08 GB for fp64 and 4.51–6.77 GB for fp32. Both projections sit well inside.
- With `maxit = 3` the peaks are about 0.5 GB lower (4.47 / 6.21 / 5.11 GB). Test 10 measures the whole `solve_modes` run.

**fp32 V-cycle prototype** (hierarchy cast to fp32; coarsest Cholesky and deflation in fp64; `lob32.py`, `c11b-plan1/lob32b.py`):
- LOBPCG iterations, fp32 / fp64: box $16^3$ 15/15, box $32^3$ 15/16, plate $32^2\times1$ 21/21, odd $33\times8\times8$ 14/14.
- λ agree to ≤ $3\cdot10^{-11}$ relative.
- The cycle differs from fp64 by $3\cdot10^{-7}$ relative, and $x^\top P_{32}x$ stays positive, including on $A_P x$ and $A_P^2 x$ directions.
- Asymmetry $|u^\top Pv - v^\top Pu|/\sqrt{u^\top Pu\; v^\top Pv} \le 6\cdot10^{-9}$.
- The stencil allocates its accumulators in fp64 and promotes fp32 inputs, so `apply_K`/`apply_M` must follow the vector's dtype (below). With that patch, fp32 applies match fp64 to ≤ $1.7\cdot10^{-7}$ (two phases, masked, nodal masses, consistent and lumped).

**SIMP fields under LOBPCG** (observation, cause unknown). On the checked-in SIMP field at tol $10^{-6}$ for 10 modes:
- fp64 GMG at λ₁ stagnates at 208 iterations, or at 157 with 15 guards;
- fp32 GMG converges at 241;
- Jacobi does not converge in 400.

Logged in `docs/open_items.md` with the cause open (search-direction quality, the floor-acceptance logic, or the pencil's conditioning). Revisit in PR 21, whose modified interpolation removes the spurious island modes. Not an 11b test.

## Design

### `solve_modes`

```python
lam, V, info = solve_modes(mesh, materials, bcs, n_modes, orientations=None, *, mass="consistent",
                           nodal_mass=None, tol=1e-6, maxit=500, seed=0, chunk=None, guards=None,
                           basis_dtype="float64", low_memory=False, precond="auto", sigma=None,
                           precond_dtype="float64")
```

- **`precond`**:
  - `"auto"` (default) means `"gmg"` for a voxel mesh and `"jacobi"` otherwise.
  - `"gmg"` on a non-voxel mesh raises `ValueError`, as does an unknown name.
  - PRs 12 and 13 add `"pmg"`/`"amg"` and switch `"auto"` for unstructured meshes.
  - It is a plain `if` on the name; a registry waits until a third preconditioner exists.
- **`sigma`**:
  - `None` (default): GMG runs the pre-estimate below.
  - A float ≥ 0: a hint. GMG is built at it directly, skipping the pre-estimate.
  - Jacobi uses `jacobi(op, free, sigma or 0.0)` and never runs the pre-estimate (PR 9a: σ changes Jacobi's iterations by < 3%).
  - A negative or non-finite σ raises.
- **`precond_dtype`**: `"float64"` or `"float32"`, forwarded to `gmg(..., dtype=...)`. With Jacobi only `"float64"` is accepted. PR 13 adds fp32 for p-MG and AMG.
- **`info`** gains:
  - `precond`: the name used;
  - `sigma`: the shift the preconditioner was built at (0.0 for Jacobi without a hint);
  - `sigma_iterations`: the pre-estimate's LOBPCG iterations summed over rounds, 0 when skipped;
  - `time["precond"]`: wall time of the rigid modes, the pre-estimate and the final build. `time["setup"]` stays `lobpcg`'s.
- **Order:**
  1. `dof_mask`, then the operator and the rigid modes (as in 9c).
  2. The preconditioner (pre-estimate, then the final `gmg` at σ).
  3. `lobpcg(..., precond=P)`.

### σ pre-estimate (`_gmg_sigma(op, free, R)` in `eigen/api.py`, about 15 lines)

`sigma_estimate` is unchanged. The refinement calls `lobpcg` directly:

```python
lam, _, info = lobpcg(op, free, 1, 0.1, R, P, maxit=20)  # block 6; never raises on non-convergence
```

1. **Start.**
   - `P = gmg(op, free, 0.0)`, which succeeds when the singular guard proves $A_P$ nonsingular (clamped bodies), giving σ = 0.
   - On `SingularShiftError`, start from σ₀ = `sigma_estimate(op, free, R, maxit=10)[0]` (an upper bound) and build `P = gmg(op, free, σ₀)`.
2. **Round.** `new = lam[0]`, the Rayleigh quotient, always ≥ λ₁. Then `σ = new` if σ = 0, else `min(σ, new)`.
3. **Stop.**
   - From σ = 0, stop after one round: GMG at σ = 0 approximates $K^{-1}$ on the free DOFs, the best preconditioner for λ₁.
   - From σ₀ > 0, stop when `new > previous / 3`, or after 4 rounds.
   - Otherwise set `P = None` (one hierarchy alive at a time), rebuild at σ, and go to step 2.
4. Return σ and the summed iterations. `solve_modes` then sets `P = None` and builds the final `gmg(op, free, σ, dtype=precond_dtype)`. Rebuilds on the same grid do not recompile (11a).

The residual tolerance 0.1 is `lobpcg`'s own λ-scaled residual. With Jacobi a residual stop was rejected (a stop at 0.5 gave 432× λ₁, `docs/eigensolvers.md`). With GMG the residual reaches 0.1 only near an eigenpair; the probes stop early (4–14 iterations) only where σ is within 1% of λ₁, and otherwise run to 20. The binary field is the known miss.

**`SingularShiftError(ValueError)`** is a new exception in `precond/gmg.py`, exported from `jaxmodal.precond`.
- Raised by every σ = 0 refusal in `_check_sigma0`: zero-stiffness cells, rigid modes left, and meshes `rigid_modes` rejects (chained).
- The finite backstops ("smoother blocks are not finite", "coarsest factor is not finite") stay plain `ValueError` and propagate.
- `_gmg_sigma` catches only `SingularShiftError`.

### fp32 V-cycle: `gmg(op, free, sigma, *, degree=2, coarse_degree=4, ratio=30.0, dtype="float64")`

- **Build.** The build runs in fp64 as today (the Galerkin sums, smoother inverses and `lmax_estimate` need it). With `dtype="float32"` the levels are then rebuilt in fp32:
  - Each coarse level's `BlockELL` gets fp32 values: `A32`.
  - Each smoother becomes `dataclasses.replace(S, A=A32, precond=D32, lmax=f32(lmax), ratio=f32(ratio))`. `D32` is the Partial with fp32 inverse blocks, and for k = 1 also the fp32 mask: `Partial(_block_jacobi, inv32, m.astype(f32))`. An fp64 `m` would promote every smoother output to fp64 and silently undo the cycle.
  - The transfer weights are cast to fp32.
  - Level 0 keeps the fp64 `APOperator` as `levels[0][0]` and as `S.A` (one object, no copy of the caller's Stencil).

  So `levels[l][0] is levels[l][1].A` holds on every level. The fp64 originals are dropped before `gmg` returns, so steady storage is the fp32 copy.
- **Kept in fp64:**
  - the coarsest Cholesky factor (the cycle casts the coarse right-hand side to fp64 and back);
  - the deflation arrays (small, applied outside the cycle in fp64);
  - `free`;
  - the fine operator's leaves.
- **Dtype-following applies.** `Stencil.apply_K`/`apply_M` cast the operator's float leaves to `v.dtype` inside the jitted apply, and cast `reference_element`'s $K_e$/$m_e$ to `v.dtype`; the quadrature tables are NumPy fp64 constants, so $m_e$ would otherwise stay fp64 and the mass GEMM would run in fp64. They allocate the `_slabs`/phase-scan accumulators in that dtype. The lumped `apply_M` casts the (fp64) diagonal once. `APOperator.__call__` casts its mask, σ and $s$ likewise.
  - For fp64 vectors every cast is a no-op, so fp64 results are bitwise unchanged; no current caller passes fp32 (LOBPCG upcasts fp32 basis slices).
  - Diagonals and `block_diag_K` stay fp64.
  - The unstructured operator is unchanged (PR 13).
- **Cycle** (`_vcycle_apply`):
  1. Mask and pre-deflate in fp64.
  2. Cast to `P.dtype` and run the V-cycle.
  3. Cast the correction back and post-deflate in fp64.

  `GMG` gets a static field `dtype`. Pre- and post-smoothing are the same operations; the fp32 rounding asymmetry (≤ $6\cdot10^{-9}$ in the energy-normalized measure) is harmless to LOBPCG. PCG to $10^{-8}$ with an fp32 cycle is not promised (documented).

### Memory estimator (`memory.py`)

**`gmg_memory(mesh, m, dtype="float64", s_K=None)`** returns `storage`, `workspace`, `build` and `overhead`, with the operator estimators' keys and composition rule.

- **`storage`** is an upper bound from shape arithmetic. It shares no hierarchy code with `gmg.py` (only `_islands`, for the island term). The shape is the grid `gmg` builds from: `mesh.voxel.shape` cropped of the empty trailing layers along the scan axis, as `grid_ids` does. That crop decides which axes are stopped and the group size k.
  - **Fine level:** exact from the mesh. Smoother inverse blocks: $9k$ entries per node for the group size $k$ (the product of the stopped axes' node counts), in `dtype`. Plus the group index arrays, the k = 1 mask, and `free`.
  - **Coarse levels:** each axis of the bounding box is halved while it has more than 2 elements (11a's rule), and every level is counted as a full box. Per level:
    - ELL values: rows padded like `_ell_rows`, $27\cdot9$ entries, in `dtype`;
    - int32 ELL columns;
    - smoother blocks and masks as on the fine level;
    - transfer index arrays (int32) and int64 tables with `dtype` weights.
  - **Coarsest factor:** $(3n_c)^2\cdot 8$ (≤ 27 nodes).
  - **Islands:** with `s_K` given, the island deflation from `_islands` and the padded row count (104 B per row, $96^2\cdot8$ for $E^+$).
  - The measure is the unique device buffers of the built `GMG` by identity, excluding the operator's own leaves (counted by `stencil_memory`). The Chebyshev and APOperator scalars are negligible and covered by the bound.
- **`workspace`** $= c_\text{dt}\cdot 8\,n_\text{dof}\,m$: the V-cycle temporaries alone, for PCG users. $c_\text{dt}$ is an upper envelope of `memory_analysis` of the jitted cycle at $16^3$–$64^3$ ($c$ = 12.1 for fp64 and 8.45 for fp32 at $87^3$). With islands, add $3\,n_\text{rows}/n_\text{dof}$ blocks.
- **`build`:** max(the per-node setup fit (+1.12 GB at $87^3$), fp64 storage + `dtype` storage). The cast runs after the fp64 build's sweep temporaries are freed, so the fp32 build holds at most both hierarchies. Test 10's band confirms it.

**`lobpcg_memory(..., precond_memory=None)`** takes a callable $m \mapsto$ preconditioner estimate (`lambda m: gmg_memory(mesh, m)`).
- The step term becomes $(A + A_P)\,b + f\,w$. $A_P$ is a per-(step type, dtype) table `GMG_STEP` of extra blocks, an upper envelope from `memory_analysis` of the 9-block and low-memory steps with GMG at $32^3$–$87^3$ (table above). It is one fit, not a fit times a fit. The fit is checked once against a harness peak difference (GMG minus Jacobi) at $64^3$, so that an underestimate of the GMG part cannot hide behind the Jacobi part's slack.
- The preconditioner's `storage` is added to the total, and its `build` enters max(build, step).
- `precond_memory=None` is Jacobi (already inside $A$), unchanged.

**`solve_modes_memory(mesh, bcs, n_modes, *, mass="consistent", chunk=None, guards=None, basis_dtype="float64", low_memory=False, precond="auto", precond_dtype="float64", n_rigid=None)`** is the one-call estimate of a `solve_modes` run.
- It is composed from `stencil_memory` or `unstructured_memory`, `gmg_memory` (no islands: `solve_modes` builds uniform `s_K`) and `lobpcg_memory`.
- `n_rigid=None` computes it with `rigid_modes(mesh, dof_mask(mesh, bcs))`; pass it to skip that host work at large sizes.
- It returns `lobpcg_memory`'s dict.
- The pre-estimate's GMG is freed before the final build, and its 6-column LOBPCG is smaller than the solve's, so it adds nothing to the peak (test 10 measures the whole clamped run; the free path frees P before each rebuild).

### Docs

- **`docs/eigensolvers.md`:**
  - `solve_modes`: the signature, the dispatch, `precond`/`sigma`/`precond_dtype`, and the new `info` keys. Replace step 4 "`lobpcg` with the default Jacobi". Rewrite the "Slender models under Jacobi … thin plates are out of reach" bullet: voxel meshes now get GMG, unstructured meshes wait for PRs 12–13.
  - The σ pre-estimate section:
    - the GMG-refined rule and its table;
    - why the residual stop is safe with GMG while the Jacobi text ("no stop criterion here … 432×") still holds for Jacobi;
    - the free-body cost;
    - the binary-field miss;
    - the sentence "lobpcg therefore never runs the pre-estimate … whose solve_modes dispatch runs it" updated to the shipped dispatch.
  - Mac acceptance: the measured runs. In the model table, $N_\text{it}$ = 16 and $c_V$ = 5.3 / 4.3 (re-measured on the shipped code) replace 150 and 5.5.
  - The memory model: `precond_memory`, `GMG_STEP`, `solve_modes_memory`.
- **`docs/preconditioners.md`:**
  - the fp32 V-cycle: what is cast, the fp64 coarse solve and deflation, the measured iterations, time (1.76 → 1.48 s) and temporaries, and that PCG to $10^{-8}$ is not promised;
  - `gmg_memory`;
  - the 2M acceptance;
  - correct "18 at $87^3$" to the shipped 16.
- **`docs/operators.md`:** the stencil's dtype-following `apply_K`/`apply_M`.
- **`README.md`:** voxel meshes get GMG automatically. Show a voxel `solve_modes` call with `precond_dtype` and `solve_modes_memory`. Update the status line.
- **`docs/ROADMAP.md`:**
  - the PR 11b status, and the "Plans:" line gains `pr-11b-gmg-solve.md`;
  - the deferred GPU checks, with `solve_modes_memory`'s 100M figure (low-memory step, fp32 V-cycle, $m = 8$). Expected outcome: it does not fit one 80 GB GPU (the step alone is about 70 GB), so the 100M solve comes after PR 19;
  - new deferred items: fp32 V-cycle speed on the GPU (16% on CPU), the σ = 0 guard's host `rigid_modes` at 100M (about 27 s and 20 GB host, scaled from 2M), and the per-island $A_P$ applies of the deflation setup.
- **`docs/open_items.md`:**
  - Move the σ pre-estimate entry to Resolved, noting the binary-field exception (390×).
  - New entries:
    - the binary-field λ₁ miss;
    - SIMP LOBPCG stagnation at tol $10^{-6}$, cause open;
    - phase-contrast models: islands are detected from `s_K` only, so a multi-phase voxel model with $10^3$–$10^5$ material contrast gets no island deflation.

## Files

- `src/jaxmodal/eigen/api.py`: `solve_modes` options, `_gmg_sigma`.
- `src/jaxmodal/precond/gmg.py`: `dtype`, the fp32 level rebuild, `SingularShiftError`, the cycle casts.
- `src/jaxmodal/precond/__init__.py`: export `SingularShiftError`.
- `src/jaxmodal/operators/voxel.py`, `operators/constraints.py`: dtype-following applies.
- `src/jaxmodal/memory.py`: `gmg_memory`, `lobpcg_memory(precond_memory=)`, `GMG_STEP`, `solve_modes_memory`.
- Tests: `tests/test_api.py`, `tests/test_gmg.py`, `tests/test_stencil.py`, `tests/test_lobpcg.py` (memory).
- `benchmarks/gmg_study.py`: an `acceptance(n)` entry (the 2M solve, fp64 and fp32: times, iterations, peaks, estimates) into `results/gmg_study.json`.
- The docs above.

## Tests

Runtime budget: the fast suite (`-m "not slow and not gpu"`) stays under 5 min idle (ROADMAP rule). The new fast tests add at most about 60 s; the larger cases are marked slow.

1. **σ pre-estimate within 2× of λ₁** (`test_api.py`, parametrized, `_gmg_sigma` against `reference_modes`): $\lambda_1(1 - 10^{-8}) \le \tilde\sigma \le 2\lambda_1$.
   - Fast cases: box $8^3$, odd $9\times8\times8$, plate $32^2\times1$, beam $64\times2\times2$, free box $8^3$, free plate $16^2\times1$, free beam $128\times1\times1$ (2.7 λ₁ after one round, so it needs the rounds).
   - Slow cases: aniso $h_z = h/10$ at $16^3$, the checked-in SIMP field.
   - A spy on `gmg` (wrapping the calls inside `_gmg_sigma`) asserts the σ sequence. Clamped cases build exactly once, at 0.0. Free cases build at σ₀ > 0 first, and each later build σ is ≤ the previous one. Test 3 asserts, through `solve_modes`, that the final build is at `info["sigma"]` (clamped sequence [0.0, σ̃]).
   - `sigma_iterations` ≤ 20 × rounds.
2. **Hint skips the pre-estimate.**
   - `solve_modes(..., sigma=s)` gives `info["sigma"] == s` and `info["sigma_iterations"] == 0`; a spy shows `_gmg_sigma` is not called and `gmg` is built once, at `s`; the modes match the default run to the solve tolerance.
   - `sigma=0.0` on a free body raises `SingularShiftError`: the hint is not overridden.
3. **`solve_modes` on an odd voxel mesh matches `eigsh`.**
   - $9\times6\times5$ voxel mesh, clamped `xmin`, 6 modes, tol $10^{-10}$, `precond="auto"`. λ relative error ≤ $10^{-8}$ and MAC ≥ $1 - 10^{-8}$ against `reference_modes` (PR 9 tolerances); `info["precond"] == "gmg"`.
   - The same for a free odd voxel mesh (σ₀ path, rigid deflation).
   - The same for an occupancy-masked odd voxel mesh with nodal masses.
4. **Dispatch, options, errors.**
   - `"auto"` picks Jacobi for a hex20 box; `"gmg"` there raises.
   - `"jacobi"` on a voxel mesh gives the 9c path (`info["sigma"] == 0.0`).
   - Unknown names, `precond_dtype="float32"` with Jacobi, and a negative or NaN σ all raise.
   - `info` has the new keys, including `time["precond"]`.
   - **Exception narrowness:**
     - `gmg(..., 0.0)` on a free body raises `SingularShiftError`, while a backstop (patched non-finite smoother blocks) raises a `ValueError` that is not one.
     - With `gmg` monkeypatched to raise a plain `ValueError` at σ = 0, `solve_modes` propagates it.
5. **Dtype-following applies** (`test_stencil.py`):
   - fp32 `apply_K`/`apply_M` (consistent and lumped, two phases, occupancy-masked, nodal masses) return fp32 and match fp64 to $10^{-6}$ relative; `ap_operator` on fp32 returns fp32.
   - Compiled `memory_analysis` temporaries of fp32 `apply_K` and consistent `apply_M` are ≤ 0.6× fp64 (measured 0.50). This kills "compute in fp64, cast the result" and an fp64 $m_e$. The lumped `apply_M` has no output-sized temporaries, so for it only the dtype and accuracy are checked.
6. **fp32 GMG** (`test_gmg.py`):
   - Leaves: ELL values, smoother inverses, transfer weights and Chebyshev scalars are fp32. `levels[l][0] is levels[l][1].A` on every level. The only fp64 leaves left are the operator's own arrays (identical objects), `free`, `L` and `defl`.
   - The cycle output is fp64, within $10^{-5}$ relative of the fp64 cycle, symmetric to $10^{-5}$, and positive on masked inputs.
   - The jitted fp32 cycle's temporaries are ≤ 0.75× fp64 (measured 0.64 with the corrected prototype, 0.70 with an fp64 $m_e$).
   - LOBPCG (10 modes, tol $10^{-6}$) with fp32 GMG converges with ≤ 1.2× the fp64 iterations, with λ within $10^{-9}$ relative. Box $16^3$ is fast (15/15). Box $32^3$, plate $32^2\times1$ and odd $33\times8\times8$ are slow (15/16, 21/21, 14/14).
7. **`gmg_memory` storage bound.** On box $16^3$, an occupancy-masked odd mesh, the island case (with `s_K`) and plate $32^2\times1$, in fp64 and fp32:
   - the bound is ≥ the unique buffer bytes of the built `GMG` (excluding the operator's leaves);
   - on the full boxes it is ≤ (measured ratio + 0.05)×, with the measured ratio (expected close to 1, from row padding only) recorded at implementation. That way dropping one term (the fine smoother blocks are about 20% of the bytes) fails.
   - A masked mesh cropped along the scan axis (`voxel_mesh((10, 10, 10))` with only z < 2 active, so k = 3 on the fine level) is included.
8. **Workspace and step fits are upper bounds, and tight.**
   - V-cycle `memory_analysis` (fp64, fp32; $m$ = 15) at $16^3$ (fast) and $32^3$ (slow) is ≤ `gmg_memory["workspace"]`, and within 1.25× of it at $32^3$.
   - The LOBPCG step with GMG (9-block, low-memory) is ≤ the `lobpcg_memory` step term.
9. **Estimator composition** (fast, exact). `lobpcg_memory` with `precond_memory` minus without equals the preconditioner storage plus the change in max(build, step) per the documented rule. `solve_modes_memory(...)["total"]` equals the hand-composed sum from the three estimators, for GMG fp64, GMG fp32 and Jacobi, clamped and free.
10. **Mac acceptance** (slow, macOS; `test_api.py`). $87^3$ voxel box (2.04M DOF), steel, `xmin` clamped, `solve_modes(..., 10, tol=1e-6)` with defaults (GMG, pre-estimate, 9-block, fp64 basis), once with `precond_dtype="float64"` and once with `"float32"`.
    - **Wall time:** one cold subprocess per dtype (`harness.measure` with an empty setup; it drops the compilation cache, so compilation counts). `time_s` ≤ 600 s, and all modes converged.
    - **Peak:** one warm subprocess per dtype (setup = a full run, freed). $|\text{est} - \text{peak}| \le 0.2\,\text{peak}$ with est = `solve_modes_memory(mesh, bcs, 10, precond_dtype=...)["total"]`, and est ≥ 2 GB.
    - Expected: 59 / 55 s, peaks 6.1–6.7 / 5.1 GB, estimates about 6.8 / 5.8 GB.
11. **Gmg-study acceptance entry smoke** (slow): `gmg_study.acceptance(n=8)` runs and records the times, iterations, peak and estimate keys.

## Not covered

- p-MG and AMG fp32, and the unstructured operator's dtype-following (PR 13).
- A 100M solve and the GPU checks (deferred list).
- The SIMP LOBPCG stagnation, the binary-field λ₁ miss, and island detection from phases (`open_items.md`).
