# Open items

Decisions taken without the user's input, and known weaknesses deferred to later PRs. Each entry says what is open, the current default, and where it is revisited. Resolved entries move to the end with their resolution.

## Open

### Voxel GMG on high-contrast SIMP fields (PR 11a → PR 13b, PRs 21–23)

At a SIMP field's own $\lambda_1$ ($s_K = \rho^3$, $s_M = \rho$, $\rho_\min = 10^{-3}$) the lowest modes are spurious island and void-region modes. With island deflation and coarse Chebyshev degree 4, PCG to $10^{-8}$ takes 87 iterations on the checked-in field (the roadmap test, ≤ 100), 109–135 on other seeds of the same recipe and 452 on an unfiltered binary field (`benchmarks/results/gmg_study.json`). Every case converges; only speed differs.

- **Default:** ship as is (user, 2026-10-12: "As long as it should work for general problems now it's good").
- **Candidate fix:** operator-dependent transfers. A SciPy prototype with Jacobi-smoothed, phase-aware prolongation plus deflation reached 34 (checked-in field) and 55 (seed 4), but needs coarse stencils wider than 27 points, which algebraic multigrid has.
- **PR 13b's AMG study** (`amg_study.simp`, the fields as unstructured hex8 meshes): AMG is worse than GMG on the filtered fields at θ = 0 (checked-in field at its own λ₁: 789–911 against 87; at the solid's λ₁ 70–75 against 33; seeds 4 / 5: 646–761 / 136–152 against 133–135 / 107–109) and level on the binary field (446–451 against 413–450; `benchmarks/results/amg_study.json`). AMG's near-null space is the global rigid modes and θ = 0 aggregates across void and solid, while GMG deflates the islands. A strength threshold θ = 0.02 brings the checked-in field at its own λ₁ to 49–51, but at the solid's λ₁ its coarse levels turn dense (1804 nonzeros per row) and coarsening stalls at 1152 rows (build plus PCG unfinished after 16 min); it also raises AMG storage 34–56% on the 13a cases. **Decision without user input (PR 13b):** keep θ = 0 and no `theta` keyword (it would expose a setting with a pathological setup).
- **Revisit:** PRs 21–23: modified mass interpolation removes the spurious modes; a per-island near-null space (as GMG's deflation) or a safe strength threshold for AMG; PR 23 re-runs the contrast test on a real design.

### Thin plates and slender beams (PR 11a → M4 / PR 18)

At σ = $\lambda_1$ the GMG iterations grow like $h^{-1}$ on one-element-thick plates (23 / 37 / 71 / 136 from $16^2$ to $128^2$): coarse hex8 layers lock in bending. Default: documented limitation; the roadmap single-layer test uses $32^2\times1$. Revisit in M4 (solid-shell element).

p-MG (PR 12) has the same cause on flat elements: its hex8 coarse space locks in bending when the element aspect ratio $a \gg 1$. Single-layer hex20 plates $16^2\times1$ take 15 / 21 / 63 iterations for $a$ = 1 / 2 / 4 and stagnate at $a = 10$ (620–720); a box with $h_z = h/a$ takes 14 / 17 / 28 / 76; the hex27 plate $32^2\times1$ at $a = 10$ exceeds 1000. Slenderness with $a = 1$ is fine (beam $20\times2\times2$: 16).

AMG on the hex8 level (PR 13a, `max_coarse` = 100) makes it worse: plates 36 / 85 / 229 / > 1000 for $a$ = 1 / 2 / 4 / 10, boxes 14 / 20 / 52 / 244. The deeper AMG levels lose plate bending (the $a = 1$ plate takes 15 / 19 / 36 with zero / one / two AMG levels); the beam stays at 16. **Decision without user input (PR 13a):** neither AMG on the high-order level nor a bending-aware near-null space is attempted; both are a new setup design rather than a hex8-level AMG. Retargeted to M4 / PR 18 (solid-shell elements).

### Anisotropic voxel spacing (PR 11a)

Coarsening is by element count, not spacing: $h_z = h/10$ needs 106 iterations. Default: documented limitation. Spacing-aware semi-coarsening needs line smoothers; no PR owns it yet.

### Binary-field λ₁ miss of the σ pre-estimate (PR 11b)

On an unfiltered binary SIMP field ($32^3$) the GMG-refined pre-estimate lands at 390 λ₁: λ₁ is a void-region mode that 20 iterations from a random start miss (100 find it), and 10-mode LOBPCG does not converge on that field at any σ either. Default: documented limitation (filtered fields and every other tested case are within 1.05 λ₁). PR 13b's SIMP study gives no help: AMG is level with GMG on that field (446–451 against 413–450 PCG iterations). Revisit with PRs 21–23's modified mass interpolation, which removes the void modes.

### SIMP LOBPCG stagnation at tol $10^{-6}$ (PR 11b)

On the checked-in SIMP field, 10 modes at tol $10^{-6}$: fp64 GMG at λ₁ stagnates at 208 iterations (157 with 15 guards), the fp32 cycle converges at 241, Jacobi does not converge in 400. Cause open: search-direction quality, the floor-acceptance logic or the pencil's conditioning. Default: documented; not a test. Revisit in PR 21 (modified interpolation removes the spurious island modes).

### Island deflation from phases (PR 11b)

Stiff islands are detected from `s_K` only, so a multi-phase voxel model with $10^3$–$10^5$ material contrast between phases (uniform `s_K`) gets no island deflation. Default: documented. Revisit when a multi-phase contrast case appears (PR 23).

### GMG intermittent extra peak in LOBPCG (PR 11b)

With GMG the LOBPCG process peak is bimodal: in some runs, from about iteration 5, it is about one V-cycle temporary above the step's `memory_analysis` (+12 blocks at $48^3$), in other identical runs not; synchronous dispatch makes it rarer. So `lobpcg_memory` with `gmg_memory` is not a strict bound at mid sizes: 11% low at $64^3$ (fp64 cycle), 4% high at $48^3$ (worst of three runs), 10–12% high at $87^3$; all inside the roadmap's ±20% band. Default: keep the `memory_analysis` envelope `GMG_STEP` and document it. Revisit on the GPU (PR 24: the BFC allocator behaves differently) or if a size falls outside the band.

### Run-to-run LOBPCG peak without GMG (found during PR 12, decision without user input)

The chunked 300k-DOF diagonal-pencil peak (`test_estimate_chunks`) varies between identical runs too: 0.95–1.07× `lobpcg_memory` (fp64 basis, 1.24–1.39 against 1.30 GB; fp32 up to 1.04×), and peaks above the estimate, which earlier runs did not show, became the common case on 2026-10-13 with no package change (it fails on `main` as well). Unchunked (30 modes, $m = 36$) it is 1.00–1.16×. Synchronous CPU dispatch makes the excursions smaller but does not remove them, as for GMG. All inside the roadmap's ±20% band. Default, as for GMG: keep the estimator and test against 0.8–1.1× the estimate instead of a strict bound (3% headroom over the worst run; if it trips again and repeated runs show the base level unchanged at about 0.95×, with only the intermittent excursion larger, widen to the roadmap's 1.2×; otherwise investigate the estimator or the step, since an extra fp64 basis copy lands at about 1.17–1.22×). Revisit with the GMG item (PR 24) or if a size falls outside the band.

### `splu` ordering for the test-only coarse solve (PR 12, decision without user input)

Geometric nested dissection on the corner coordinates factors 4–8× faster than COLAMD (hex20 $24^3$: 3.5 s / 0.69 GB against 14 s / 1.5 GB), but `Unstructured` carries no coordinates. Default: keep COLAMD instead of adding a `points` argument to a test-only path. `amg` (PR 13a) needs no host direct solve (device Cholesky of a coarsest matrix ≤ 4·`max_coarse`), so this stays as is unless `pmg` leaves the tests.

### `pmg` kept alongside `amg` (PR 13a, decision without user input)

`pmg` (exact host `splu` coarse solve) stays as the test-only reference next to `amg`: its tests and study pin the p-level, and `amg`'s no-AMG-level path is checked against it. Default: keep; remove if it ever becomes a maintenance burden.

### Own SA setup loop instead of `smoothed_aggregation_solver` (PR 13a, decision without user input)

The roadmap says "pyamg smoothed-aggregation setup". `amg` runs its own short loop over pyamg's primitives, because `fit_candidates` keeps zero tentative columns and pyamg's solver hides the resulting zero rows with a pseudo-inverse coarse solve, while the roadmap requires the columns dropped (dense Cholesky). The first-level prolongator equals pyamg's exactly (test). Default: keep.

### Masked near-null space (PR 13a, decision without user input)

$B$ has the rows of constrained corner DOFs zeroed, as the roadmap says. Unmasked $B$ gave the same iteration counts but makes the constrained rows of the all-hex8 $P_0$ nonzero (0.19 on hex8 $8^3$ with sliding faces), so the fine mask would no longer be the only mask needed. Default: masked.

### AMG memory constants calibrated on box meshes (PR 13b, decision without user input)

`amg_memory`'s per-corner-DOF storage constant, its step constants `AMG_STEP` and its setup term are fitted on box meshes (hex20 $8^3$–$44^3$, hex27 $6^3$–$24^3$, hex8 $12^3$–$68^3$; a distorted hex20 box stays inside the bound). Other geometries aggregate differently. Default: document; revisit if a real mesh falls outside the roadmap's ±20% band.

### AMG build peak above 1M DOF (PR 13b, decision without user input)

The measured setup peak per corner DOF grows about 0.65 kB per decade of $n_c$ (hex20 3.7 / 4.3 / 4.8 kB at $12^3$ / $24^3$ / $44^3$); `AMG_BUILD` extrapolates that trend to 10M DOF (14.6 GB hex20, 46.6 GB hex8). hex27 has two points ($12^3$, $24^3$) that grow about 1.3 kB per decade; its 6.0 kB lies between the hex20 slope (about 5.75 kB at 10M DOF) and its own (about 6.6 kB). Check it with the first 10M setup (the deferred GPU check runs it on the host).

### PR 13b decisions without user input

- **10M GPU configuration:** the memory table assumed $m = 30$ for 10M; with AMG 30 modes do not fit the 76 GB pool (about 89–94 GB device), so the deferred GPU check runs 20 modes ($m = 25$, stored Ke, 9-block step; 71.5 GB by the shipped `solve_modes_memory`, fits thinly; 30 modes: 92.6 GB), with on-the-fly Ke (about 60 GB) as fallback. The roadmap check names no mode count, so no criterion is relaxed.
- **Mac 10M run:** 10 modes, stored Ke, 9-block step; `amg_study.large` stops above 52 GB; the shipped model gives 52.2 GB, so the run was skipped (the 10M acceptance is the deferred GPU check).
- **`nonsingular=True` trusts the caller** apart from `s_K > 0`; `solve_modes` derives it from `rigid_modes` (no rigid modes left, no `s_K` passed). `pmg` keeps refusing σ = 0 (test-only).
- **`"amg"` on voxel meshes raises:** GMG is better on every voxel case measured (SIMP study above).
- **The COO assembler is gone:** `pmg`'s Galerkin matrix now comes from the production assembler (same values to rounding, BSR instead of CSR).
- **Aggregate reuse with moved points approximates:** it keeps the old near-null space; only connectivity and `free` are checked.

### PR 13c decisions without user input

- **Option dicts, not flat keywords:** `operator_options`, `precond_options`, `sigma_options` keep the `solve_modes` signature stable as builders gain options, and avoid the clash between the operator's `chunk` (elements per scan step) and the solver's `chunk` (modes per locking chunk).
- **Options kept internal:** `lobpcg`'s `R` and `precond` (set by `solve_modes`); `amg`'s `nonsingular` (a proof, derived from the rigid-mode count) and `aggregates` / `return_aggregates` (state: compose the building blocks); `gmg`'s `dtype` (the top-level `precond_dtype`; the pre-estimate builds stay fp64 for robustness); `pmg` (test scale, superseded by `amg`); `block_jacobi` / `chebyshev` alone (without a coarse level they gain nothing, PR 10). Passing a pre-built operator or preconditioner to `solve_modes` is not supported; revisit if a user needs it.
- **No default changes:** every default is the robust option measured in PRs 9–13 (table in `docs/eigensolvers.md#options`); smoother degree 2 is the fastest in a full solve (7.0 / 8.8 / 10.3 s at degree 2 / 3 / 4) and stays.
- **`amg_memory` at other `max_coarse`:** the coarse term is floored at the calibration value 1000 (an upper bound at 100–3000, loose above 1000: 4.3× at 3000); `AMG_STEP` stays calibrated at 1000, so above it the step term is not a bound (+1.3 blocks of temporaries at 4000).
- **Chebyshev loop terms:** degree > 2 adds `AMG_LOOP_K` = 120k DOF-equivalents to `AMG_STEP`'s $K$ (fixed loop buffers, +77k–103k measured) and `GMG_LOOP` = 1.2 / 0.7 fp64 blocks (fp64 / fp32 V-cycle) to `gmg_memory` (+1.07 / 0.60 measured), calibrated on boxes like the other constants. `GMG_STEP` re-fitted after the loop lowered the default fp64 steps (4.3 / 9.0, was 5.1 / 9.9); `GMG_WORK` kept (now 7–39% above the measured temporaries at $32^3$–$64^3$).

### PR 14a decisions without user input

- **PR 14 split:** 14a ships the counting estimator (`count`); 14b ships the $[0, f_{max}]$ driver, the budget logic and `verify`. The roadmap tests are tagged [14a]/[14b]; no criterion is dropped.
- **Shift $\sigma = \lambda_c$:** the spectral map $X = 2\lambda_c T - I$ puts the cut at $x = 0$ and needs no new operator; the preconditioner is built at $\lambda_c$ like any `solve_modes` shift.
- **Probes from quadrature factors:** $M_e = B_e B_e^T$ with $B_e = N^T \operatorname{diag}\sqrt{w \det J}$ (exact, no Cholesky); lumped mass uses $\sqrt{\operatorname{diag} M}$. Constrained DOFs carry diagonal 1 in $M_P$ and are projected out of every probe (the roadmap's scaled diagonal is not needed: $g$ vanishes there, so $y$ is the same).
- **Weyl exponent $a = 3/2$ fixed** in the tail-bias bound (3D); a 2D or 1D-like body (slender beam, thin plate) has a slower-growing count, so the bound stays conservative.
- **Degree search:** doubling then bisection on the exact $q(x_l)^2 \ge 0.95$ and tail-bias $\le 0.05$ tests; `MAX_DEGREE` = 100 000 (beyond it `count_degree` returns `(None, nan)` and `count` is not run: the estimate is NaN, `verifiable` False; 14b must then move the cut or solve more modes).
- **Tail-bias grid $64d$ instead of the plan's $8d$:** the trapezoid on $8d$ points was 2.5% off `quad` (second order); $64d$ gives 0.04% and costs 1–2 s for a degree search near `MAX_DEGREE`. The ripple guard stays on $8d$.
- **Jacobi default inner preconditioner** for `count`'s `precond=None` (robust, no setup); callers (PR 14b's `verify`) pass the fp64 AMG/GMG built at $\lambda_c$.
- **Inner tolerance rule $\tau_c = 0.01/(d\sqrt{n_{free}})$ kept although far too strict (needs your input):** the counting study measures $|\Delta s| \le \tau s$ independent of $d$ (20–400), so at the default $\tau_c$ the error is about $10^{-6}$, five orders below the 0.5 threshold. Loosening it (e.g. to $\tau = 10^{-4}$, error $\le 10^{-4}$ with $s \approx 1$) would save 1.8–3.9 PCG iterations per decade, about a third of the PCG iterations on the measured meshes, but the roadmap fixes $\tau_c \lesssim 0.01/(d\sqrt{n})$ and roadmap criteria are not relaxed without you. Default: keep (still kept in PR 14b; decide before PR 15).
- **Counting cost:** at 200 modes, counting at a typical cut ($g$ = 0.002–0.04, $d$ = 500–11 000) is predicted at 10²–10³ solve times (median 160–250×); only cuts with $g \gtrsim 0.6$ cost a few solves (1.2–3.6×). `verify=None` in 14b will therefore rarely enable counting; 14b should place cuts at the largest gap near the band edge.
- **Early stop and NaN:** a block whose PCG solve fails (stagnation or `maxit`) stops the recurrence; the estimate is NaN and `verifiable` is False, so 14b never trusts a partial count.
- **Detection test:** 100 sets of 50 probes from one 5000-probe count; each probe has its own `fold_in` key, so the sets are i.i.d. and equal 100 independent 50-probe counts (the roadmap's fixed list of 100 seeds), and results do not depend on the block size (default one block; 14b sizes it with `probe_block`).
- **Orthonormality check:** $W^T M W = I$ is checked by a streamed random sketch (3 vectors) at `ORTHO_TOL` (1e-6 fp64, 1e-4 fp32), not a full Gram matrix.
- **Memory constants:** `COUNT_STEP`, `GMG_COUNT`, `AMG_COUNT` are fitted on box meshes (hex20 $12^3$–$28^3$, voxel $16^3$–$64^3$); the hex20 $8^3$ case (one chunk holds the mesh) is outside the fit.
- **No thin-plate counting test:** AMG is weak on flat elements (see "Thin plates and slender beams"); the slender hex20 cantilever (κ 2.2e9) covers the high-κ case.

### PR 14b decisions without user input

- **No strategy selection or basis-fit check:** LOBPCG from 0 is the only band strategy, so there is nothing to select; a band whose basis outgrows the device fails with an out-of-memory error. PR 15 adds shift-invert slices, the cost model's choice and the memory check.
- **Basis growth:** past its capacity (default $4\cdot$`chunk`, or `n_modes`) the locked basis grows 1.5× (at least room for the next lock) at lock time: a transient of old + new arrays, and the LOBPCG step recompiles for the new shape. No `max_modes` cap; passing `n_modes` near the expected count avoids growth.
- **Band chunk default 40** (block 48, as the 2M-DOF budget measured in PR 11b) without `n_modes`; with `n_modes`, $\min(40, \lceil 1.25\,n\rceil + 2)$.
- **The a-priori bound's factor $c = 2$** assumes a count step costs at least 7 PCG iterations at width $b$ and a LOBPCG iteration at most about 3 applies at width $m$. Measured slack: the real step is 4.7–6.4× the bound's step at $m = 48$ (7× in `test_calibration`), but only 1.3× on the 2.7k-DOF box at $m = 11$, where LOBPCG's fixed costs dominate; the margin is only 1.3× there, so on tiny solves the bound may exceed the real cost on another machine or mesh and skip a count that would have fit (never a wrong count).
- **Two seams `_bound` and `_calibrate`** instead of one `_predict`: the tests patch each separately (the bound to force a calibration, the calibration to skip the count).
- **An explicit `count_options["degree"]` bypasses `MAX_DEGREE`** (the user's choice; only the degree rule's $d$ is capped, and `count` returns NaN only on that path).
- **`verify=True` keeps the modes when the count does not fit** (budget below one probe): it returns them with `ran` False, the reason and a warning instead of raising, because the check runs after the solve and raising would discard it.
- **Probe budget = the solve's estimate − host modes − kept aggregates:** `solve_modes_memory(band=...)` at the final capacity, minus $8\,n\,k$ for the converged modes on the host (shared memory on the Mac) and the AMG aggregates; the λ_c multigrid is counted at fp64.
- **The final chunk converges modes beyond the band** (`n_beyond`): they are needed for $\lambda_\text{next}$ and are deflated in the count, but not returned.
- **Host `V` holds `n_modes` + chunk columns** in band mode (one more block than a plain solve); `lobpcg_memory(band=True)` adds one block.
- **Band peak test:** voxel $48^3$, `n_modes=10` (chunk 15, block 20, inside the memory model's 15–48 fit) measured 2.83 GB against the 2.62 GB estimate, 7% under (within the 20% criterion).
- **The final chunk's re-orthonormalization is defensive:** at test sizes $\max|W^\top M W - I|$ is the same with and without it (3.8e-15 vs 4.2e-15 fp64, 7.5e-9 vs 7.7e-9 fp32, 3 chunks); it guards against the drift PR 9b measured after about 20 chunks.
- **$\tau_c$:** the band count uses the roadmap's $\tau_c$; decided in PR 15c (below).

### PR 15a decisions without user input

- **PR 15 split into 15a / 15b / 15c** (the one-slice solver; the band-pass count and restarts; slicing, strategy choice and `band=(f1, f2)`; PR 15c split again into 15c, the slicing engine, and 15d, the strategy and API), every roadmap PR 15 test assigned to one part (`docs/plans/pr-15a-shift-invert.md`).
- **Ritz pairs by Rayleigh–Ritz on (K, M)** over the shift-invert Krylov basis, not from the inexact projected $T$ (whose residuals exceeded $10\tau\delta$ and grew with the basis); the roadmap's "final Rayleigh–Ritz" is the last step's, with true residuals recomputed on the compact V. No implicit restarts: 15c splits a full slice.
- **Per-pair thresholds** $t_i = \max(tol, 10\tau_\text{eff}\delta_i, \text{round-off floor})$ and $r_{acc} = \max t_i$ over the in-slice pairs and the neighbours (the roadmap's $r_{acc}$ plus LOBPCG's floor term and the neighbours, so never below the roadmap's value).
- **σ at the slice midpoint** (or given), moved once by 1% of the width when the first solve stagnates above $10^3\max(\tau, 10^{-10})$ (the plan said $10^3\tau$; the floor term keeps a τ below the fp64 MINRES floor from moving σ); $\tau_\text{eff} > 10^{-4}$ or a non-finite MINRES residual ends the slice as "inner solves failed".
- **τ_eff** = the running max of the stagnated or capped columns' final residuals; later inner solves capped at 10× the first step's iterations (the first step at the σ used).
- **`capacity` is required** (no default allocation); 15d sets it from `si_memory` and the budget. The basis also stops at the free space left by R and W (beyond it new columns are round-off noise).
- **`lam` holds the converged `inside` set**, which extends 100 $r_{acc}$ past each edge and over clusters straddling one (the plan's "in $[\lambda_{lo}, \lambda_{hi}]$" read with its own 100 r window); 15c's duplicate removal handles the overlap.
- **`si_memory`** counts the build on top of the full storage plus a host term (as `lobpcg_memory`), and takes the expected slice count `n_modes` for its end phase.
- **Cost driver:** the depth of σ in the spectrum, not δ, sets the MINRES iterations (measured); 15d fits its cost model on depth (roadmap note on the two cost-model bullets).
- **Largest-size benchmark cut (user: "Run something more reasonable", 2026-10-13).** The plan's 1M and 2M τ = 1e-14 floor solves run near mode 6, not 100 (about 2 min each), and the 2M voxel slice sits near mode 6 too. The 1M hex20 slice near mode 100 was measured before the cut (converged, 9 steps of about 780 MINRES iterations, 1.7 h). `shift_invert_study.py --parts` runs one part at a time and every slice prints a progress line per step.

### PR 15c decisions without user input

- **Counting tolerance $\tau_c$ kept at the roadmap rule** $0.01/(d\sqrt{n_{free}})$ (user, 2026-10-14: "Just decide for yourself"). Loosening it to $10^{-4}$ would save about a third of the PCG iterations of a count (PR 14a's study) but leave counts at 100× or more the solve at typical gaps, so it would not change which counts run, while the strict rule keeps the error about $10^{-6}$, far below the 0.5 threshold. `count_options["tau"]` sets it per call; 15d's cost model prices counts at the τ in use.
- **PR 15c split into 15c (slicing engine `si_band`) and 15d (cost model, memory budget, strategy, `verify=None`, `band=(f1, f2)`)**, every roadmap PR 15 test assigned ([15c] / [15d] tags in the roadmap).
- **No provisional interior edges from a cost model:** the measured slice cost (about 1.8 Lanczos columns per mode plus about 84 per slice, a slice spending 50–85 before it converges anything) makes the widest slice that fits the cheapest; slices come from full slices reduced to their span and remainders, the converged span kept as a leaf without a second run. `edges` is an internal option, used as given.
- **Edges come from full slices:** the span si_slice's completeness rule verifies on the run's final Ritz values, cut at the largest gap $\ge 100\,r_{acc}$ within each half of its converged run (the roadmap's snapping rule), a binary split on a side without a cut; a flagged span is rerun. The cut is the largest gap within the half on its side, not the run's overall largest (the roadmap says "largest gap"): a cut deep in the run moves up to half the run into the remainder's W, lowering its capacity, while the outermost gap can be tiny and the count's degree is $\propto 1/g$.
- **si_slice counts deflated W modes as converged candidates** (a W mode can close a side), and deflation windows include the nearest stored pair beyond each edge. Consequence: a mode across a boundary is deflated by the next slice instead of found twice, so the roadmap's collapse test sees a merged window without a duplicate (the plan expected one); duplicate removal is tested on fakes.
- **Remainders run at their midpoint** (si_slice's default σ).
- **Leaves keep si_slice's widened inside sets** (margins, straddling chains) rather than only the pairs inside their interval; deflation windows and the merge handle the overlap, so `lam` can hold pairs just outside the band.
- **Deflation windows:** margins $\min(1000 r_w, 10^{-2})$, chains at $100 r_w$, $r_w = \max(r_s, \texttt{tol})$; `capacity` is the total basis per slice, deflation included.
- **MINRES does not stagnate before its best explicit residual drops below $\|b\|$** (a plateau, not a stall), fixed where it surfaced (PR 15c); an inner failure stops the band without a seed retry.
- **`edge_unreached` is not evidence of an edge outside the spectrum:** no fallback in 15c; 15d applies the $f_1 < \lambda_1$ rule and the $\lambda_{hi}$ bound up front; an edge in $(\lambda_\max, \hat\lambda_\max]$ raises the capacity error, which names both causes.
- **The merge** applies the roadmap's 0.5 projected-norm rule per direction of a run's window subspace, then Rayleigh–Ritz; a failing window is widened once (by the neighbouring windows), then flagged (`edge_rr_failed`).
- **Counts reuse the preconditioner of their slice** (a span leaf: its parent's, before it is released; built at the slice's σ, not at $c_{hi}$; only PCG iterations depend on it).
- **A flagged or unverifiable run leaf is not split or rerun** beyond 15b's restarts.
- **`verify` is a bool in 15c**; the per-leaf callable comes with 15d's `verify=None`.
- **V is M-orthogonal to residual level across runs** that do not deflate each other (code review round 1: $1.6\cdot10^{-9}$ on modes 5–40 of hex20 $6\times5\times4$, about $4\cdot10^{-9}$ on hex20 $8\times6\times5$ modes 5–120, against $\le 10^{-13}$ within windows); the bound $(\|x_j\|\|r_i\| + \|x_i\|\|r_j\|)/|\lambda_i - \lambda_j|$ is exact and tested. No final M-orthonormalization across runs: it needs another streamed pass over V and changes the residuals by the same order.
- **A slice whose $r_{acc}$ widens its deflation window is rerun** with the extra stored pairs deflated (code review round 1: a slice with loose inner solves found a stored triple again, and the next slice raised on its non-orthonormal W). Rare (raised $\tau_\text{eff}$ only); the alternative, deduplicating at insertion, leaves kept vectors only residual-orthogonal to the stored ones near them.
- **Slow tests beyond the plan's fast budget:** the block-2 capacity split (60 s) and the capacity-72 raise (43–97 s, load-dependent) are marked slow; the fast tier keeps the block-6 split, the fakes and the unit tests of `_split`, `_window` and the merge rules.
- **Cross-run orthogonality inside a deflation window** (code review round 2): a window can hold stored pairs of two runs that did not deflate each other (the margin grows when $r_s$ rises between a parent and its remainder); their inner product is residual-level, $\lesssim 2r/g$ ($r$ the pairs' λ-scaled residuals, $g$ their relative gap), and at the default tol $10^{-6}$ and gaps near $10^{-2}$ it could in principle exceed si_slice's W check ($10^{-6}$ by sketch), which would raise. Not seen in any run (actual residuals sit well below tol); revisit if a slice raises on its W.
- **`sigma` and `count_precond` are rejected by `si_band`**: σ is each slice's midpoint and `make(sigma)` gives the slices' and counts' preconditioners, so 15d exposes `make`'s options, not these two.

### MINRES plateaus below $\|b\|$ (PR 15c)

PR 15c's stagnation rule skips plateaus only while the best explicit residual is $\ge \|b\|$. A plateau below $\|b\|$ in the 2-norm (another norm ratio, a better preconditioner) would still stop the slice as "inner solves failed", and the band with it. Not measured so far. A later fallback: continue a stagnated column once from its best iterate. While a column plateaus above $\|b\|$, `minres` returns its best *checked* iterate, which can be worse than $x = 0$ ($1.22\,\|b\|$ in the test case); a capped column counts as bad for si_slice's σ-move and τ_eff rules either way. Revisit if a band stops with "inner solves failed".

### Late capacity errors and slice efficiency (PR 15c → PR 15d)

Code review round 1 (hex20 $8\times6\times5$, AMG; scratch measurements):

- **A capacity error discards the band.** Modes 5–200 at capacity 96 ran 37 slices (about 25 min) and then raised at a remainder. At 1M DOF a slice step can take hours. Remainders run at capacity 72 there because a 2-mode window costs a whole projector chunk (24 columns). Options for 15d: size the capacity with the window's chunk included; return the pairs found with `converged` False and the reason, as an inner failure does; or attach them to the error.
- **Verification slices.** 11 of those 37 slices found at most one new mode: their pairs were already stored by the parent, and they ran 10–15 steps to close their edges. MINRES is about 92% of slice time. 15d's cost model should price this; closing such remainders more cheaply is an option.
- **A top remainder holding only known modes can raise.** Its upper neighbour must still converge at σ = its midpoint. Seen on the cube band [6.855, 39.384] at capacity 96 ("no progress" after all band modes were found). Option: σ toward an edge whose neighbour is missing.

### Span counts at small capacities (PR 15c → PR 15d)

A small capacity converges few modes per slice, so a span's cut can only fall in a small gap, and the count's degree grows as $1/g$: modes 8–21 of hex20 $6\times5\times4$ at block 2 and capacity 24 needed a degree of about 18 000 for the first span (about 6 min, against seconds for the slices), and the band then raised the capacity error ("no progress"), so the count was wasted. Default: the count runs (`verify=True`); 15d's per-leaf `verify=None` decides on the predicted cost (`degree` is recorded with `verify=False`), and its capacity choice should avoid such slices.

### Shift-invert cost near mode 1000 (PR 15a → PR 15d)

With σ near mode 1000 (hex20 $16^3$, AMG), a mid-gap MINRES solve takes about 14k iterations (4970–5260 at depth 500); at 1M DOF on the Mac that is about 4 h per slice step. Of two runs at the same settings one had a column stagnate (last iterate $6.8 \times 10^{-3}$) and one converged ($9.8 \times 10^{-9}$), the difference most likely from round-off (the recompiled loop) (the best-iterate fix leaves the iterates and exits unchanged): at that depth the inner solve is marginal, and σ near an eigenvalue stagnated there too. So the limit is both cost and possible failure. This matters for the 10M GPU goal, where LOBPCG's basis runs out near modes 700–1000 and the roadmap sends those bands to shift-invert. 15d's cost model must price it; options if it is too slow: deflate the converged modes below σ from MINRES, a stronger (higher-degree or σ-aware) multigrid, or a memory-lean LOBPCG fallback. May need the user's input if no option meets the roadmap's 10M band criteria.

### σ-move retries (PR 15a → PR 15d)

A slice whose first solve stagnates moves σ once and repeats the solve; when the stagnation is depth-driven, the retry costs a second full solve before the slice stops. 15d's strategy should not place slices where the cost model predicts stagnation, so the retry only handles σ on an eigenvalue.

### Band-pass count and slice restarts (PR 15b, decisions without user input)

- **Tail model fitted to the band.** No mode index is known inside a band, so the Weyl count $N \propto \lambda^{3/2}$ is fitted to the in-band modes and the nearest pair on each side, with $N(0) = 0$ (rigid modes deflated). A band with few modes gives a rough fit; the 3/2 exponent stays the upper bound, and the tail binds only with very many modes below.
- **Only the nearest out-of-band pair per side is deflated.** Further modes add at most $q^2 \approx 7\cdot10^{-4}$ (at $g/2$ past an edge), inside the 0.05 bias. Adjacent slices' modes are not all deflated (memory and transfer cost).
- **σ = $c_{hi}$, the upper counting edge.** One recurrence; the lower edge costs $(1 + a)/(2\sqrt a)$ more degree (2× at $a = 0.07$). A σ between the edges would balance the edges. 15c's slices are as wide as memory allows, so wide leaves pay the lower edge's factor; it is recorded per leaf (`degree`, `gap`) and is an item for 15d's cost model.
- **The slice's preconditioner is reused for the count** (built at the slice's σ, SPD): no second multigrid in memory; it costs PCG iterations when σ is far from $c_{hi}$ (wide slices). PR 15c: counts reuse the slice's preconditioner (PR 15c decisions above).
- **Restarts** deflate the found in-band modes, use fresh seeds for the start block and the probes, and stay within the first run's device memory (capacity reduced); at most 2. Unverifiable counts and unconverged slices are not restarted: a rerun would not change them. PR 15c: a flagged slice is not split (PR 15c decisions above).
- **The planted-mode test** scales the mass by $s$ to move one eigenvalue into a gap, instead of editing a mesh.
- **Progress** comes from a per-step `jax.debug.callback` (scalars only, rate-limited on the host); the verbose loop is a separate compile.
- **Slow tests.** The band expected-value tests (missing, free) and detection tests (missing, complete) take 21–111 s and are marked slow, above the plan's fast budget; they run in the full suite.

### End-to-end `jax.grad` and eigenvector sensitivities (user, 2026-10-13)

Asked whether `solve_modes` should support `jax.grad` end to end (a custom VJP wrapping PR 21's sensitivities) or eigenvector sensitivities. User: "No idea. Just follow the roadmap". Default: the roadmap as written (principle 5: eigenvalue sensitivities by a separate element-kernel pass in PR 21; solvers and preconditioners never differentiated; no eigenvector gradients). Revisit if an optimization use case needs mode-shape objectives.

### GMG restriction on the GPU (PR 11 deferred GPU checks)

Restriction is the transpose of gathers, i.e. scatter-adds (6× the prolongation time on CPU at $87^3$). Default: keep; listed in the deferred GPU checks with "rewrite as gathers if it dominates".

## Resolved

- **MINRES returned its last iterate (PR 15a review round 4).** After a stall the returned residual had drifted up to 40× above the minimum (σ on an eigenvalue), which inflated τ_eff. `minres` now keeps and returns the best iterate (one more block). The fix changes only the returned iterate and its reported residual, not the iterates, flags or counts. The 1M / 2M floors at depth 6 are $7.0 \times 10^{-14}$ / $3.3 \times 10^{-14}$; the 1M attempt at depth 100 that ended at $2.3 \times 10^{-3}$ stays unexplained (not rerun).

- **p-MG refuses σ = 0 (PR 12 → PR 13b, decision without user input).** `amg(..., nonsingular=True)` accepts σ = 0 when the caller states that $A_P(0)$ is nonsingular (it checks `s_K > 0` itself); `solve_modes` passes it when no rigid modes are left, so clamped bodies start the σ pre-estimate at σ = 0. `pmg` keeps refusing σ = 0 (test-only).

- **AMG memory figures in the ROADMAP table (PR 13a → PR 13b).** `amg_memory` models the AMG as device storage plus a LOBPCG step extra (about 6 blocks at $m = 25$ with the 9-block step; about 11 with the low-memory step, which therefore gains nothing); the Multigrid row, the Preconditioner workspace row and the 10M line are re-derived from it (`docs/ROADMAP.md`). The two host copies of the hex8 matrix are gone with the COO assembler.

- **p-MG on curved elements: approximate rigid rotations (PR 12 → PR 13a, decision without user input).** PR 12 asked PR 13's near-null space to restrict the fine rigid modes. `amg` evaluates the rigid modes at the corner coordinates instead: the curved $6^3$ case converges in 14, the same as the straight one, and restriction would need a fine rigid-mode build plus a per-corner least-squares fit.

- **σ pre-estimate on slender and high-contrast bodies (PR 11b, 2026-10-12).** `sigma_estimate` was up to 3800× above λ₁ on thin plates, slender beams, anisotropic grids and SIMP fields. Resolved by the GMG-refined pre-estimate (`docs/eigensolvers.md`): within 1.05 λ₁ on every tested case except the unfiltered binary field (390×, open above).

- **SIMP roadmap test at which σ (PR 11a, 2026-10-12).** I had kept the test at the solid box's σ and documented the field's own $\lambda_1$ (723 iterations) as a limitation. User: "Make it a blocking item". Resolved by island deflation and coarse degree 4: 87 at the own $\lambda_1$.
