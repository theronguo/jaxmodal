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

### σ pre-estimate on slender and high-contrast bodies (PR 11b)

`sigma_estimate` is 30–3800× above $\lambda_1$ on thin plates, slender beams, anisotropic grids and SIMP fields, so the roadmap check "within 2× of $\lambda_1$" would fail there. Default: none yet; 11b's plan must choose a better pre-estimate (e.g. GMG-preconditioned LOBPCG).

### GMG restriction on the GPU (PR 11 deferred GPU checks)

Restriction is the transpose of gathers, i.e. scatter-adds (6× the prolongation time on CPU at $87^3$). Default: keep; listed in the deferred GPU checks with "rewrite as gathers if it dominates".

## Resolved

- **SIMP roadmap test at which σ (PR 11a, 2026-10-12).** I had kept the test at the solid box's σ and documented the field's own $\lambda_1$ (723 iterations) as a limitation. User: "Make it a blocking item". Resolved by island deflation and coarse degree 4: 87 at the own $\lambda_1$.
