# PR 11b plan: GMG in `solve_modes`, σ pre-estimate, fp32 V-cycle, estimator, Mac acceptance

PR 11b completes PR 11 (M3) on top of 11a's `gmg(op, free, sigma)`. It makes the voxel GMG the default preconditioner of `solve_modes` and adds the following:

- a σ pre-estimate that works on slender and high-contrast bodies (`docs/open_items.md`);
- an fp32 V-cycle;
- the GMG memory estimator and a one-call `solve_modes_memory`;
- the 2M-DOF Mac acceptance;
- the deferred GPU checks.

Roadmap tests in this PR:

| roadmap test | here |
|---|---|
| σ pre-estimate within 2× of λ₁; a user hint skips it | test 1, 2 |
| `solve_modes` on an odd voxel mesh matches `eigsh` (PR 9 tolerances) | test 3 |
| fp32 GMG: same tol as fp64, ≤ 20% more LOBPCG iterations | test 6 |
| Mac acceptance: 2M DOF, 10 modes, tol $10^{-6}$, ≤ 10 min, peak within 20% of the estimators, fp64 and fp32 V-cycles | test 9 |
| deferred GPU checks | ROADMAP list (no test) |

## Measured facts (prototypes in `scratchpad/p11b/`; jax 0.11.2, M3 Max CPU)

Steel, unit box, `xmin` clamped unless noted. λ₁ is the SciPy reference (`benchmarks/gmg_study.lam1`).

**σ sensitivity.** LOBPCG with GMG built at σ = f·λ₁ (10 modes, tol $10^{-6}$), in iterations:

| case | f = 0 | 1e-3 | 0.3 | 1 | 3 | 10 | 100 | 1e4 |
|---|---|---|---|---|---|---|---|---|
| box $16^3$ | 15 | 15 | 15 | 15 | 16 | 17 | 30 | 234 |
| free box $16^3$ (6 rigid modes deflated) | refused | 17 | 19 | 20 | 23 | 32 | 95 | not conv. |
| plate $32^2\times1$ | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 32 |
| beam $64\times2\times2$ | 15 | 15 | 15 | 15 | 15 | 15 | 16 | 19 |
| aniso $h_z = h/10$, $32^3$ | 109 | 108 | 109 | 108 | 109 | 108 | 109 | 100 |

- Any σ in $[0, 3\lambda_1]$ is as good as $\lambda_1$; σ ≥ 10λ₁ costs iterations on bulky and free bodies.
- `sigma_estimate` (Jacobi) is 46–3800× above λ₁ on plates, beams, anisotropic grids and SIMP fields (11a study). At 2M DOF it takes 73 s (252 iterations), more than the LOBPCG solve itself.
- Capping it at 3 or 10 iterations gives $10^2$–$10^7$×.

**GMG-refined pre-estimate** (prototype `sig3.py`):
- Build `gmg(op, free, 0.0)`, if the singular guard accepts σ = 0. Otherwise build `gmg(op, free, σ₀)`, with σ₀ = `sigma_estimate(..., maxit=10)` (an upper bound).
- Then run 1-mode LOBPCG with that GMG (block 6, maxit 20, residual tol 0.1). σ = its Rayleigh quotient, which is ≥ λ₁.
- From σ₀ > 0, rebuild at σ and repeat while σ falls below a third of the previous value (at most 4 rounds).

| case | start | rounds | σ/λ₁ | time |
|---|---|---|---|---|
| box $16^3$ | 0 | 1 | 1.00 | 3.7 s |
| free box $16^3$ | 5.1λ₁ | 2 | 1.00 | 1.4 s |
| plate $64^2\times1$ | 0 | 1 | 1.00 | 4.9 s |
| beam $128\times1\times1$ | 0 | 1 | 1.00 | 4.6 s |
| SIMP field (checked-in), $32^3$ | 0 | 1 | 1.00 | 4.9 s |
| SIMP seed 4 | 0 | 1 | 1.00 | 2.0 s |
| aniso $h_z = h/10$ | 0 | 1 | 1.00 | 2.2 s |
| free plate $32^2\times1$ | 1800λ₁ | 2 | 1.00 | 2.6 s |
| free beam $64\times2\times2$ | 1000λ₁ | 2 | 1.00 | 4.6 s |
| unfiltered binary SIMP, $32^3$ | 0 | 1 | 390 | 2.9 s |

- The binary field's λ₁ is a void-region mode that 20 iterations from a random start miss. 100 iterations find it.
- 10-mode LOBPCG does not converge on that field at any σ, with GMG or with Jacobi (below). It is a documented limitation, not a test case.
- Times are at prototype sizes. At 2M DOF the pre-estimate costs 13 s (σ = 0 build 5.1 s, 5 iterations 8.1 s) against 73 s for Jacobi `sigma_estimate`.

**Mac acceptance prototype** ($87^3$, 2.04M DOF, 10 modes, tol $10^{-6}$, 9-block step, fp64 basis, cold process, wall time including compilation):

| V-cycle | σ pre-estimate | final GMG build | LOBPCG (iterations) | total |
|---|---|---|---|---|
| fp64 | 13.2 s | 2.0 s | 42.9 s (16) | 58.8 s |
| fp32 | 13.2 s | 2.3 s | 38.7 s (16) | 54.9 s |

The budget model assumed 150 LOBPCG iterations and about 8 min (`docs/eigensolvers.md`). The 10-min budget stays as the roadmap fixes it; the model's table is updated with the measured numbers.

**Peak memory** (`harness.measure`, `lobpcg(..., maxit=3)` after the build, $87^3$, block 15):

| preconditioner | peak | `lobpcg_memory` estimate without preconditioner |
|---|---|---|
| Jacobi | 4.47 GB | 5.27 GB |
| GMG fp64 | 6.21 GB | not applicable |
| GMG fp32 | 5.11 GB | not applicable |

GMG adds 1.75 GB (fp64) or 0.64 GB (fp32) over Jacobi:
- 0.31 GB of storage (11a);
- the V-cycle workspace inside the step: 12.1 fp64 blocks for the V-cycle alone (fp32: 8.4), partly overlapping the step's own temporaries.

**fp32 V-cycle prototype** (cast the built hierarchy to fp32; coarsest Cholesky and deflation in fp64):
- LOBPCG iterations, fp64 → fp32: box $32^3$ 16 → 15, plate $32^2\times1$ 21 → 21, odd $33\times8\times8$ 14 → 14.
- At $87^3$, m = 15: V-cycle 3.19 → 2.61 s, temporaries 12.1 → 8.4 fp64 blocks.
- The fp32 cycle differs from the fp64 one by $3\cdot10^{-7}$ relative, and its asymmetry is $10^{-6}$ relative. LOBPCG tolerates both (it uses the preconditioner only to build its search directions).
- The stencil allocates its accumulators in fp64 and promotes fp32 inputs, so the prototype makes `apply_K`/`apply_M` follow the vector's dtype (below).

**SIMP fields under LOBPCG.**
- On the checked-in SIMP field, LOBPCG for 10 modes at tol $10^{-6}$ reports stagnation at:
  - 208 iterations with fp64 GMG at λ₁;
  - 157 with 15 guards.
- With fp32 GMG it converges in 241 iterations. Jacobi does not converge in 400.
- The likely cause is the fp64 residual floor: $\varepsilon\,\lambda_\max/\lambda_1 \approx 10^{-6}$ for this $10^9$-contrast pencil, whose $\lambda_1$ is a spurious island mode.
- Logged in `docs/open_items.md` (revisit in PR 21, whose modified interpolation removes the spurious modes). Not an 11b test.

## Design

### `solve_modes`

```python
lam, V, info = solve_modes(mesh, materials, bcs, n_modes, orientations=None, *, mass="consistent",
                           nodal_mass=None, tol=1e-6, maxit=500, seed=0, chunk=None, guards=None,
                           basis_dtype="float64", low_memory=False, precond="auto", sigma=None,
                           precond_dtype="float64")
```

- **`precond`**:
  - `"auto"` (default) is `"gmg"` for a voxel mesh and `"jacobi"` otherwise.
  - `"gmg"` on a non-voxel mesh raises `ValueError`.
  - Any other value raises.
  - PRs 12 and 13 add `"pmg"` and `"amg"`, and switch `"auto"` for unstructured meshes.
  - Dispatch is a plain `if` on the name; a registry waits until a third preconditioner exists.
- **`sigma`**:
  - `None` (default): GMG runs the pre-estimate below.
  - A float ≥ 0: a hint. GMG is built at it directly, and the pre-estimate is skipped.
  - Jacobi uses `jacobi(op, free, sigma or 0.0)` and never runs the pre-estimate (PR 9a: σ changes Jacobi's iterations by < 3%).
  - A negative or non-finite σ raises.
- **`precond_dtype`**: `"float64"` or `"float32"`, forwarded to `gmg(..., dtype=...)`. With Jacobi only `"float64"` is accepted (others raise). PR 13 adds fp32 for p-MG and AMG.
- **`info`** gains:
  - `precond`: the name used;
  - `sigma`: the shift the preconditioner was built at (`None` for Jacobi without a hint);
  - `sigma_iterations`: LOBPCG iterations of the pre-estimate, 0 when it is skipped;
  - `setup_s`: wall time from the start of `solve_modes` to the start of `lobpcg`, covering the build, rigid modes, the pre-estimate and the final GMG.
- Order:
  1. `dof_mask`, then the operator, then the rigid modes (as in 9c).
  2. The preconditioner (pre-estimate, then the final `gmg` at σ).
  3. `lobpcg(..., precond=P)`.

  The pre-estimate GMG is freed before the final build (only one hierarchy is alive).

### σ pre-estimate: `sigma_estimate(op, free, R=None, maxit=300, seed=0, precond=None)`

The existing function gains `precond`:
- With `precond=None`, it behaves as today (Jacobi, plateau stop).
- With a preconditioner, it runs 1-mode LOBPCG (block 6) with it, stopping at residual ≤ 0.1 or `maxit` (callers pass 20), and returns `(λ̃₁, iterations)`. λ̃₁ ≥ λ₁ always holds (a Rayleigh quotient).

The plateau stop stays Jacobi-only: with GMG, λ̃ can sit at a plateau for a few iterations before dropping (binary field: 7 iterations at 390λ₁). The residual stop is safe with GMG because it reaches 0.1 only near an eigenpair. The prototype stops at 4–8 iterations only on the cases where λ̃ is already within 1% of λ₁.

The round logic lives in `eigen/api.py` (`_gmg_sigma(op, free, R)`, about 15 lines):
1. **Start.** Build `gmg(op, free, 0.0)`; this succeeds when the singular guard proves $A_P$ nonsingular (clamped bodies). On `ValueError` from the guard, start instead from σ₀ = `sigma_estimate(op, free, R, maxit=10)[0]` and build `gmg(op, free, σ₀)`.
2. **Refine.** σ = `sigma_estimate(op, free, R, maxit=20, precond=P)[0]`.
3. **Stop.** From σ = 0, stop after one round (GMG at σ = 0 is $K^{-1}$ on the free DOFs, the best preconditioner for λ₁). From σ₀ > 0, rebuild at σ and repeat until σ > previous/3, at most 4 rounds.

To tell the guard's refusal apart from other errors, `gmg` raises a `SingularShiftError(ValueError)` subclass at σ = 0; only that one is caught. Rebuilds on the same grid do not recompile (11a). Total pre-estimate iterations go to `info["sigma_iterations"]`.

### fp32 V-cycle: `gmg(op, free, sigma, *, degree=2, coarse_degree=4, ratio=30.0, dtype="float64")`

- The build runs in fp64 as today (the Galerkin sums, smoother inverses and `lmax_estimate` need it). With `dtype="float32"` it then casts to fp32:
  - the coarse levels' block-ELL values;
  - every level's smoother inverse blocks;
  - the Chebyshev `lmax` and `ratio`;
  - the transfer weights.

  The fp64 originals are dropped, so steady storage is the fp32 copy; the build peak briefly holds both, and the estimator's `build` term includes it.
- Kept in fp64:
  - the coarsest Cholesky factor (the cycle casts the coarse right-hand side to fp64 and back);
  - the deflation arrays (small, applied outside the cycle in fp64);
  - the fine $A_P$, which is the matrix-free operator. Its stored leaves stay fp64, and it runs in the vector's precision (next bullet).
- **Dtype-following applies.** `Stencil.apply_K`/`apply_M` cast the operator's float leaves to `v.dtype` inside the jitted apply and allocate their accumulators in that dtype. `APOperator.__call__` casts its mask, σ and $s$ likewise. For fp64 vectors every cast is a no-op, so fp64 results are unchanged (existing tests). For fp32 vectors the slab GEMMs run in fp32.
  - Diagonals and `block_diag_K` stay fp64.
  - The unstructured operator is unchanged (PR 13 owns fp32 for p-MG and AMG).
- **The cycle** (`_vcycle_apply`):
  1. Mask and pre-deflate in fp64.
  2. Cast to `P.dtype` and run the V-cycle.
  3. Cast the correction back and post-deflate in fp64.

  `GMG` gets a static field `dtype`. The cycle is applied symmetrically (same operations pre and post). Its fp32 rounding asymmetry, about $10^{-6}$ relative, is within what LOBPCG and PCG to $10^{-6}$ tolerate. PCG to $10^{-8}$ with an fp32 cycle is not promised (documented).

### Memory estimator (`memory.py`)

**`gmg_memory(mesh, m, dtype="float64", s_K=None)`** returns `storage`, `workspace`, `build` and `overhead` (the operator estimators' keys and composition rule).

- **`storage`** is computed exactly from the hierarchy shapes, with only boolean masks on the host. Each level's active-cell mask is coarsened by the 11a rule, and its node count is the OR of the 8 shifted masks; there is no `voxel_mesh` per level, so the estimate stays cheap at 100M cells. It sums:
  - ELL values: rows padded to `_ell_rows`, $27\cdot9$ entries, in `dtype`;
  - ELL columns: int32;
  - smoother inverse blocks: $9k$ entries per node for group size $k$, in `dtype`, plus their index arrays;
  - transfer index arrays and weights;
  - the coarsest dense factor: $(3n_c)^2\cdot8$;
  - the fine `free` vector;
  - with `s_K` given, the island deflation, from `_islands` and the padded row count: about 104 B per row plus 96²·8 for $E^+$.

  Test 7 checks it equals the built `GMG`'s leaf bytes exactly.
- **`workspace`**: the V-cycle temporaries at width $m$. A fit $c_\text{dt}\cdot n_\text{dof}\cdot m\cdot 8 + w_\text{op}$ with $c$ from `memory_analysis` of the jitted cycle at three sizes, as for the stencil (12.1 fp64 blocks for fp64 and 8.4 for fp32 at $87^3$, $m = 15$; the fit is an upper envelope). With islands, add 3 padded-rows/$n_\text{dof}$ blocks (11a review).
- **`build`**: the setup peak, a per-node fit of the 11a measurement (+1.12 GB at $87^3$) plus, for fp32, the fp32 copy of the storage.

**`lobpcg_memory(..., precond_memory=None)`**: a new keyword taking a callable $m \mapsto$ preconditioner estimate (`lambda m: gmg_memory(mesh, m)`).
- It adds the preconditioner's `storage` to the total and its `build` into max(build, step).
- `step` gains $g\cdot$ workspace, with $g$ fitted from `memory_analysis` of the 9-block and low-memory steps with GMG as the preconditioner (the V-cycle's temporaries partly overlap the step's own). Fit at $16^3$–$64^3$, upper envelope, recorded like `STEP_FIT`.
- Jacobi's own storage, one vector, is already inside $A$.

**`solve_modes_memory(mesh, bcs, n_modes, *, mass="consistent", chunk=None, guards=None, basis_dtype="float64", low_memory=False, precond="auto", precond_dtype="float64", s_K=None)`**: the one-call estimate of a `solve_modes` run, composed from `stencil_memory` / `unstructured_memory`, `gmg_memory` and `lobpcg_memory`, with `n_rigid` from `rigid_modes(mesh, dof_mask(mesh, bcs))`.
- Returns `lobpcg_memory`'s dict.
- The pre-estimate's GMG is freed before the final build and its 1-mode LOBPCG is smaller than the solve, so it adds nothing to the peak (test 9 measures the whole run).

### Docs

- `docs/eigensolvers.md`:
  - the `solve_modes` signature, dispatch, `precond`/`sigma`/`precond_dtype` and the new `info` keys;
  - the σ pre-estimate section: the GMG-refined rule, the table above, the binary-field limitation;
  - the Mac acceptance section with the measured runs (the model's $N_\text{it}$ = 150 replaced by the measured 16);
  - the memory model with `precond_memory`, and `solve_modes_memory`.
- `docs/preconditioners.md`:
  - the fp32 V-cycle (what is cast, the fp64 coarse solve and deflation, the measured iterations and time, PCG to $10^{-8}$ not promised);
  - `gmg_memory`;
  - the measured 2M acceptance.
- `docs/operators.md`: dtype-following `apply_K`/`apply_M` for the stencil.
- `README.md`: the quick-start notes that voxel meshes get GMG automatically, shows a `solve_modes` voxel call with `precond_dtype`, and points to `solve_modes_memory`. Update the status line.
- `docs/ROADMAP.md`:
  - the PR 11b status;
  - the deferred GPU checks, with the 100M estimate of `solve_modes_memory` (low-memory step, fp32 V-cycle) recorded next to them.
- `docs/open_items.md`:
  - the σ pre-estimate entry resolved;
  - new entries for the binary-field λ₁ miss and for SIMP LOBPCG stagnation at tol $10^{-6}$.

## Files

- `src/jaxmodal/eigen/api.py`: `solve_modes` options, `_gmg_sigma`.
- `src/jaxmodal/eigen/lobpcg.py`: `sigma_estimate(..., precond=None)` with the residual stop.
- `src/jaxmodal/precond/gmg.py`: `dtype`, the fp32 cast, `SingularShiftError`, the cycle casts.
- `src/jaxmodal/precond/__init__.py`: export `SingularShiftError`.
- `src/jaxmodal/operators/voxel.py`, `operators/constraints.py`: dtype-following applies.
- `src/jaxmodal/memory.py`: `gmg_memory`, `lobpcg_memory(precond_memory=)`, `solve_modes_memory`, fits.
- `tests/test_api.py`, `tests/test_gmg.py`, `tests/test_stencil.py`, `tests/test_lobpcg.py` (memory).
- `benchmarks/gmg_study.py`: an `acceptance` entry (the 2M solve, fp64 and fp32: times, iterations, peaks, estimates) written to `results/gmg_study.json`.
- Docs listed above.

## Tests

1. **σ pre-estimate within 2× of λ₁** (`test_api.py`, parametrized). Cases: box $8^3$ clamped, free box $8^3$, odd $9\times8\times8$, plate $32^2\times1$, beam $64\times2\times2$, aniso $h_z = h/10$ ($16^3$), the checked-in SIMP field ($32^3$, slow), free plate $16^2\times1$.
   - Against `reference_modes`: $\lambda_1 \le \tilde\sigma \le 2\lambda_1$.
   - The prototype gives 1.00 on all of them.
   - The SIMP case is marked slow (its reference takes about 20 s).
2. **Hint skips the pre-estimate.** `solve_modes(..., sigma=s)` gives `info["sigma"] == s` and `info["sigma_iterations"] == 0`, and the modes match the default run to the solve tolerance.
   - A spy on `sigma_estimate` (monkeypatch) confirms it is not called.
   - With `sigma=0.0` on a free body the guard raises; the hint is not overridden.
3. **`solve_modes` on an odd voxel mesh matches `eigsh`.** $9\times6\times5$ voxel mesh, clamped `xmin`, 6 modes. Compare λ and MAC against `reference_modes` to the PR 9 tolerances (relative λ error ≤ $10^{-8}$ at tol $10^{-10}$, MAC ≥ $1 - 10^{-8}$), with `precond="auto"` (GMG); `info["precond"] == "gmg"`.
   - A free odd voxel mesh is run the same way: rigid deflation plus pre-estimate from σ₀ > 0.
   - Also an occupancy-masked voxel mesh with nodal masses.
4. **Dispatch and options.**
   - `precond="auto"` picks Jacobi for a hex20 box, and `"gmg"` there raises.
   - `precond="jacobi"` on a voxel mesh gives the 9c result; unknown names raise.
   - `precond_dtype="float32"` with Jacobi raises; a negative or NaN σ raises.
   - `info` has all new keys.
5. **Dtype-following applies** (`test_stencil.py`). fp32 `apply_K`/`apply_M` (consistent and lumped, two phases, occupancy-masked, nodal masses) return fp32 and match fp64 to $10^{-6}$ relative. `ap_operator` on fp32 returns fp32.
6. **fp32 GMG** (`test_gmg.py`):
   - The fp32 `GMG` leaves (ELL values, smoother inverses, transfer weights, Chebyshev scalars) are fp32, and the Cholesky factor and deflation are fp64.
   - The V-cycle output is fp64 and within $10^{-5}$ relative of the fp64 cycle.
   - It is symmetric to $10^{-5}$ and positive on masked inputs.
   - LOBPCG (10 modes, tol $10^{-6}$) with fp32 GMG converges with at most 1.2× the fp64 iterations on box $32^3$, plate $32^2\times1$ and odd $33\times8\times8$ (measured 15/16, 21/21, 14/14). Its λ agree with the fp64 run to $10^{-9}$ relative.
7. **`gmg_memory` storage is exact.** It equals the sum of the built `GMG`'s leaf bytes on:
   - box $16^3$;
   - an occupancy-masked odd mesh;
   - the island case (with `s_K`);
   - plate $32^2\times1$.

   Each in fp64 and fp32.
8. **Workspace and step fits are upper bounds.**
   - `memory_analysis` of the jitted V-cycle (fp64, fp32; m = 1, 6, 15) at $16^3$ and $32^3$ is ≤ `gmg_memory(...)["workspace"]`, and within 2× of it at $32^3$.
   - The LOBPCG step with GMG is ≤ the `lobpcg_memory` step term (9-block and low-memory).
9. **Mac acceptance** (slow, macOS; `test_api.py`). $87^3$ voxel box (2.04M DOF), steel, `xmin` clamped, `solve_modes(..., 10, tol=1e-6)` with defaults (GMG, pre-estimate, 9-block, fp64 basis), once with `precond_dtype="float64"` and once with `"float32"`.
   - **Wall time:** one cold subprocess per dtype (`harness.measure` with an empty setup, so compilation counts). `time_s` ≤ 600 s; all modes converged.
   - **Peak:** one warm subprocess per dtype (setup = a full warm-up run, freed), so the measured peak is the solve's. It must be within 20% of `solve_modes_memory(mesh, bcs, 10, precond_dtype=...)["total"]`, and the estimate must be ≥ 2 GB.
   - The prototype gives 59 s / 55 s and peaks of 6.21 / 5.11 GB.
10. **Gmg-study acceptance entry smoke** (slow): `gmg_study.acceptance(n=8)` runs and records times, iterations, peak and estimate keys.

## Not covered

- p-MG and AMG fp32 (PR 13).
- The unstructured operator's dtype-following (PR 13).
- A 100M solve and the GPU checks (deferred list).
- The SIMP LOBPCG floor and the binary-field λ₁ miss (`open_items.md`).
