# Open items

Decisions taken without the user's input, and known weaknesses deferred to later PRs. Each entry says what is open, the current default, and where it is revisited. Resolved entries move to the end with their resolution.

## Open

### Voxel GMG on high-contrast SIMP fields (PR 11a → PR 13b, PRs 21–23)

At a SIMP field's own $\lambda_1$ ($s_K = \rho^3$, $s_M = \rho$, $\rho_\min = 10^{-3}$) the lowest modes are spurious island and void-region modes. With island deflation and coarse Chebyshev degree 4, PCG to $10^{-8}$ takes 87 iterations on the checked-in field (the roadmap test, ≤ 100), 109–135 on other seeds of the same recipe and 452 on an unfiltered binary field (`benchmarks/results/gmg_study.json`). Every case converges; only speed differs.

- **Default:** ship as is (user, 2026-10-12: "As long as it should work for general problems now it's good").
- **Candidate fix:** operator-dependent transfers. A SciPy prototype with Jacobi-smoothed, phase-aware prolongation plus deflation reached 34 (checked-in field) and 55 (seed 4), but needs coarse stencils wider than 27 points, which algebraic multigrid has.
- **Revisit:** PR 13b's study runs `amg` on the checked-in field as an unstructured hex8 mesh and reports the count against GMG's (decision without user input, PR 13a); any fix beyond that moves to PRs 21–23 (modified mass interpolation removes the spurious modes; PR 23 re-runs the contrast test on a real design).

### Thin plates and slender beams (PR 11a → M4 / PR 18)

At σ = $\lambda_1$ the GMG iterations grow like $h^{-1}$ on one-element-thick plates (23 / 37 / 71 / 136 from $16^2$ to $128^2$): coarse hex8 layers lock in bending. Default: documented limitation; the roadmap single-layer test uses $32^2\times1$. Revisit in M4 (solid-shell element).

p-MG (PR 12) has the same cause on flat elements: its hex8 coarse space locks in bending when the element aspect ratio $a \gg 1$. Single-layer hex20 plates $16^2\times1$ take 15 / 21 / 63 iterations for $a$ = 1 / 2 / 4 and stagnate at $a = 10$ (620–720); a box with $h_z = h/a$ takes 14 / 17 / 28 / 76; the hex27 plate $32^2\times1$ at $a = 10$ exceeds 1000. Slenderness with $a = 1$ is fine (beam $20\times2\times2$: 16).

AMG on the hex8 level (PR 13a, `max_coarse` = 100) makes it worse: plates 36 / 85 / 229 / > 1000 for $a$ = 1 / 2 / 4 / 10, boxes 14 / 20 / 52 / 244. The deeper AMG levels lose plate bending (the $a = 1$ plate takes 15 / 19 / 36 with zero / one / two AMG levels); the beam stays at 16. **Decision without user input (PR 13a):** neither AMG on the high-order level nor a bending-aware near-null space is attempted; both are a new setup design rather than a hex8-level AMG. Retargeted to M4 / PR 18 (solid-shell elements).

### Anisotropic voxel spacing (PR 11a)

Coarsening is by element count, not spacing: $h_z = h/10$ needs 106 iterations. Default: documented limitation. Spacing-aware semi-coarsening needs line smoothers; no PR owns it yet.

### Binary-field λ₁ miss of the σ pre-estimate (PR 11b)

On an unfiltered binary SIMP field ($32^3$) the GMG-refined pre-estimate lands at 390 λ₁: λ₁ is a void-region mode that 20 iterations from a random start miss (100 find it), and 10-mode LOBPCG does not converge on that field at any σ either. Default: documented limitation (filtered fields and every other tested case are within 1.05 λ₁). Revisit with PR 13b's SIMP study (`amg` on the field) and PRs 21–23's modified mass interpolation, which removes the void modes.

### SIMP LOBPCG stagnation at tol $10^{-6}$ (PR 11b)

On the checked-in SIMP field, 10 modes at tol $10^{-6}$: fp64 GMG at λ₁ stagnates at 208 iterations (157 with 15 guards), the fp32 cycle converges at 241, Jacobi does not converge in 400. Cause open: search-direction quality, the floor-acceptance logic or the pencil's conditioning. Default: documented; not a test. Revisit in PR 21 (modified interpolation removes the spurious island modes).

### Island deflation from phases (PR 11b)

Stiff islands are detected from `s_K` only, so a multi-phase voxel model with $10^3$–$10^5$ material contrast between phases (uniform `s_K`) gets no island deflation. Default: documented. Revisit when a multi-phase contrast case appears (PR 23).

### GMG intermittent extra peak in LOBPCG (PR 11b)

With GMG the LOBPCG process peak is bimodal: in some runs, from about iteration 5, it is about one V-cycle temporary above the step's `memory_analysis` (+12 blocks at $48^3$), in other identical runs not; synchronous dispatch makes it rarer. So `lobpcg_memory` with `gmg_memory` is not a strict bound at mid sizes: 11% low at $64^3$ (fp64 cycle), 4% high at $48^3$ (worst of three runs), 10–12% high at $87^3$; all inside the roadmap's ±20% band. Default: keep the `memory_analysis` envelope `GMG_STEP` and document it. Revisit on the GPU (PR 24: the BFC allocator behaves differently) or if a size falls outside the band.

### Run-to-run LOBPCG peak without GMG (found during PR 12, decision without user input)

The chunked 300k-DOF diagonal-pencil peak (`test_estimate_chunks`) varies between identical runs too: 0.95–1.07× `lobpcg_memory` (fp64 basis, 1.24–1.39 against 1.30 GB; fp32 up to 1.04×), and peaks above the estimate, which earlier runs did not show, became the common case on 2026-10-13 with no package change (it fails on `main` as well). Unchunked (30 modes, $m = 36$) it is 1.00–1.16×. Synchronous CPU dispatch makes the excursions smaller but does not remove them, as for GMG. All inside the roadmap's ±20% band. Default, as for GMG: keep the estimator and test against 0.8–1.1× the estimate instead of a strict bound (3% headroom over the worst run; if it trips again and repeated runs show the base level unchanged at about 0.95×, with only the intermittent excursion larger, widen to the roadmap's 1.2×; otherwise investigate the estimator or the step, since an extra fp64 basis copy lands at about 1.17–1.22×). Revisit with the GMG item (PR 24) or if a size falls outside the band.

### p-MG refuses σ = 0 (PR 12, decision without user input)

`pmg` raises `SingularShiftError` at σ = 0 because it has no mesh to prove $A_P$ nonsingular. Cost: a σ pre-estimate started at σ = 0 needs fewer rounds and iterations on clamped bodies (cube: 1 round / 4 iterations against 2 / 10; plate $a = 4$: 1 / 20 against 3 / 60). Default: σ > 0 required. `amg` (PR 13a) follows the same rule. Revisit in PR 13b, which owns the pre-estimate: accept σ = 0 when `solve_modes` has found no rigid modes and every `s_K > 0`.

### `splu` ordering for the test-only coarse solve (PR 12, decision without user input)

Geometric nested dissection on the corner coordinates factors 4–8× faster than COLAMD (hex20 $24^3$: 3.5 s / 0.69 GB against 14 s / 1.5 GB), but `Unstructured` carries no coordinates. Default: keep COLAMD instead of adding a `points` argument to a test-only path. `amg` (PR 13a) needs no host direct solve (device Cholesky of a coarsest matrix ≤ 4·`max_coarse`), so this stays as is unless `pmg` leaves the tests.

### `pmg` kept alongside `amg` (PR 13a, decision without user input)

`pmg` (exact host `splu` coarse solve) stays as the test-only reference next to `amg`: its tests and study pin the p-level, and `amg`'s no-AMG-level path is checked against it. Default: keep; remove if it ever becomes a maintenance burden.

### Own SA setup loop instead of `smoothed_aggregation_solver` (PR 13a, decision without user input)

The roadmap says "pyamg smoothed-aggregation setup". `amg` runs its own short loop over pyamg's primitives, because `fit_candidates` keeps zero tentative columns and pyamg's solver hides the resulting zero rows with a pseudo-inverse coarse solve, while the roadmap requires the columns dropped (dense Cholesky). The first-level prolongator equals pyamg's exactly (test). Default: keep.

### Masked near-null space (PR 13a, decision without user input)

$B$ has the rows of constrained corner DOFs zeroed, as the roadmap says. Unmasked $B$ gave the same iteration counts but makes the constrained rows of the all-hex8 $P_0$ nonzero (0.19 on hex8 $8^3$ with sliding faces), so the fine mask would no longer be the only mask needed. Default: masked.

### AMG memory figures in the ROADMAP table (PR 13a → PR 13b)

Measured at 1M hex20: AMG device storage 0.48 GB (the table says ≈ 0.4 GB; about 5 GB at 10M hex20) and V-cycle temporaries of 11.7 fine blocks (the table budgets ≈ 4 blocks: about 2.6 GB at 1M, $m = 25$, and about 29 GB at 10M, $m = 30$). With those, "80 GB GPU: 10M fits" becomes about 84 GB with the 9-block step and about 72 GB with the low-memory step; both are upper bounds, since inside the LOBPCG step the preconditioner's extra is smaller (GMG: `GMG_STEP` = 5.1 blocks against about 12 standalone). PR 13b's estimator models p-MG + AMG as a step extra and re-derives the three rows; 13b decides between the low-memory step and a smaller $m$ for the 10M GPU solve.

Also for 13b's memory ordering: during setup `amg` holds two host copies of the hex8 Galerkin matrix (the caller's and `_sa_levels`' zero-free copy, about 0.25 GB each at 1M hex20), and all-hex8 meshes build an unused identity `NodalTransfer`; both are transient and small next to the 2.4 GB Galerkin peak.

### GMG restriction on the GPU (PR 11 deferred GPU checks)

Restriction is the transpose of gathers, i.e. scatter-adds (6× the prolongation time on CPU at $87^3$). Default: keep; listed in the deferred GPU checks with "rewrite as gathers if it dominates".

## Resolved

- **p-MG on curved elements: approximate rigid rotations (PR 12 → PR 13a, decision without user input).** PR 12 asked PR 13's near-null space to restrict the fine rigid modes. `amg` evaluates the rigid modes at the corner coordinates instead: the curved $6^3$ case converges in 14, the same as the straight one, and restriction would need a fine rigid-mode build plus a per-corner least-squares fit.

- **σ pre-estimate on slender and high-contrast bodies (PR 11b, 2026-10-12).** `sigma_estimate` was up to 3800× above λ₁ on thin plates, slender beams, anisotropic grids and SIMP fields. Resolved by the GMG-refined pre-estimate (`docs/eigensolvers.md`): within 1.05 λ₁ on every tested case except the unfiltered binary field (390×, open above).

- **SIMP roadmap test at which σ (PR 11a, 2026-10-12).** I had kept the test at the solid box's σ and documented the field's own $\lambda_1$ (723 iterations) as a limitation. User: "Make it a blocking item". Resolved by island deflation and coarse degree 4: 87 at the own $\lambda_1$.
