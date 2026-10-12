# Open items

Decisions taken without the user's input, and known weaknesses deferred to later PRs. Each entry says what is open, the current default, and where it is revisited. Resolved entries move to the end with their resolution.

## Open

### Voxel GMG on high-contrast SIMP fields (PR 11a → PR 13, PRs 21–23)

At a SIMP field's own $\lambda_1$ ($s_K = \rho^3$, $s_M = \rho$, $\rho_\min = 10^{-3}$) the lowest modes are spurious island and void-region modes. With island deflation and coarse Chebyshev degree 4, PCG to $10^{-8}$ takes 87 iterations on the checked-in field (the roadmap test, ≤ 100), 109–135 on other seeds of the same recipe and 452 on an unfiltered binary field (`benchmarks/results/gmg_study.json`). Every case converges; only speed differs.

- **Default:** ship as is (user, 2026-10-12: "As long as it should work for general problems now it's good").
- **Candidate fix:** operator-dependent transfers. A SciPy prototype with Jacobi-smoothed, phase-aware prolongation plus deflation reached 34 (checked-in field) and 55 (seed 4), but needs coarse stencils wider than 27 points: PR 13 (algebraic multigrid).
- **Revisit:** PR 13; PRs 21–23 (modified mass interpolation removes the spurious modes; PR 23 re-runs the contrast test on a real design).

### Thin plates and slender beams (PR 11a → PR 13, M4)

At σ = $\lambda_1$ the GMG iterations grow like $h^{-1}$ on one-element-thick plates (23 / 37 / 71 / 136 from $16^2$ to $128^2$): coarse hex8 layers lock in bending. Default: documented limitation; the roadmap single-layer test uses $32^2\times1$. Revisit in PR 13 (algebraic coarsening) and M4 (solid-shell element).

### Anisotropic voxel spacing (PR 11a)

Coarsening is by element count, not spacing: $h_z = h/10$ needs 106 iterations. Default: documented limitation. Spacing-aware semi-coarsening needs line smoothers; no PR owns it yet.

### Binary-field λ₁ miss of the σ pre-estimate (PR 11b)

On an unfiltered binary SIMP field ($32^3$) the GMG-refined pre-estimate lands at 390 λ₁: λ₁ is a void-region mode that 20 iterations from a random start miss (100 find it), and 10-mode LOBPCG does not converge on that field at any σ either. Default: documented limitation (filtered fields and every other tested case are within 1.05 λ₁). Revisit with PR 13's operator-dependent transfers and PRs 21–23's modified mass interpolation, which remove the void modes.

### SIMP LOBPCG stagnation at tol $10^{-6}$ (PR 11b)

On the checked-in SIMP field, 10 modes at tol $10^{-6}$: fp64 GMG at λ₁ stagnates at 208 iterations (157 with 15 guards), the fp32 cycle converges at 241, Jacobi does not converge in 400. Cause open: search-direction quality, the floor-acceptance logic or the pencil's conditioning. Default: documented; not a test. Revisit in PR 21 (modified interpolation removes the spurious island modes).

### Island deflation from phases (PR 11b)

Stiff islands are detected from `s_K` only, so a multi-phase voxel model with $10^3$–$10^5$ material contrast between phases (uniform `s_K`) gets no island deflation. Default: documented. Revisit when a multi-phase contrast case appears (PR 13 or PR 23).

### GMG intermittent extra peak in LOBPCG (PR 11b)

With GMG the LOBPCG process peak is bimodal: in some runs, from about iteration 5, it is about one V-cycle temporary above the step's `memory_analysis` (+12 blocks at $48^3$), in other identical runs not; synchronous dispatch makes it rarer. So `lobpcg_memory` with `gmg_memory` is not a strict bound at mid sizes: 11% low at $64^3$ (fp64 cycle), 4% high at $48^3$ (worst of three runs), 10–12% high at $87^3$; all inside the roadmap's ±20% band. Default: keep the `memory_analysis` envelope `GMG_STEP` and document it. Revisit on the GPU (PR 24: the BFC allocator behaves differently) or if a size falls outside the band.

### GMG restriction on the GPU (PR 11 deferred GPU checks)

Restriction is the transpose of gathers, i.e. scatter-adds (6× the prolongation time on CPU at $87^3$). Default: keep; listed in the deferred GPU checks with "rewrite as gathers if it dominates".

## Resolved

- **σ pre-estimate on slender and high-contrast bodies (PR 11b, 2026-10-12).** `sigma_estimate` was up to 3800× above λ₁ on thin plates, slender beams, anisotropic grids and SIMP fields. Resolved by the GMG-refined pre-estimate (`docs/eigensolvers.md`): within 1.05 λ₁ on every tested case except the unfiltered binary field (390×, open above).

- **SIMP roadmap test at which σ (PR 11a, 2026-10-12).** I had kept the test at the solid box's σ and documented the field's own $\lambda_1$ (723 iterations) as a limitation. User: "Make it a blocking item". Resolved by island deflation and coarse degree 4: 87 at the own $\lambda_1$.
