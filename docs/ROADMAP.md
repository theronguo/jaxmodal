# Roadmap

Large-scale 3D linear modal analysis (Kv = λMv) in JAX. The solver uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. FFD shape optimization and topology optimization come later.

**Targets:**
- Lowest modes and modes in frequency bands.
- Frequency accuracy of 0.1% (relative error in f, i.e. about 0.2% in λ) against a converged reference, for hex20/hex27 and solid-shell meshes.
- 10M DOFs, unstructured, on a single GPU.
- 100M DOFs on voxel meshes with a small block size (see the memory budget).
- Multi-GPU later.
- An FP64-capable GPU (A100/H100 class) for production runs.
- Until a GPU is available, development and all tests run on an Apple M3 Max (64 GB, CPU backend, fp64).

Package name: `jaxmodal`. License: MIT.

Status: planning. Review rounds 1–29 incorporated. Implementation starts after three consecutive review rounds with all 5 reviewers satisfied.

## Principles

1. **Validate every fast path against a slow one.** References are SciPy, dense NumPy, or analytic or published values. Every acceptance criterion is quantitative. "a equals b to tol" means ‖a − b‖_∞ ≤ tol·‖b‖_∞ (for pointwise fields such as stress, the max over all points); absolute tolerances are used only for dimensionless quantities (e.g. VᵀMV − I).
2. **One small interface.** Every discretization (unstructured, voxel) provides `apply_K`, `apply_M`, `diag_K`, `diag_M` and the rigid body modes (the interface is complete after PR 7, and for voxel meshes after PRs 6 and 7). Each operator applies natively to blocks of shape `(n_dof, m)`, not via `vmap`. Solvers only see this interface plus `apply_P`.
3. **Fixed shapes; everything else is a `jit` argument.**
   - Topology preprocessing (connectivity, DOF maps, sets) runs in NumPy.
   - Every array is a pytree `jit` argument: int32 topology, geometry, densities, material parameters, nodal masses, stored Ke, multigrid/AMG matrices and rigid modes. Nothing is closed over.
   - Only shapes, chunk sizes, element types, block sizes, Chebyshev smoother degrees, multigrid level counts and mode/option flags (stored vs on-the-fly, lumped mass, scatter vs gather, LOBPCG step variant) are static. The counting-polynomial degree is a dynamic `fori_loop` bound; its Jackson/Chebyshev coefficients are computed in-loop from (k, d).
   - The one allowed closure exception is the test-only `pure_callback` coarse solve in PR 12, with an explicit `vmap_method` and fp64 `ShapeDtypeStruct` results.
   - Iterative solvers keep fixed shapes. Dropped or converged columns are masked, not removed.
4. **Extensible element types.**
   - Each element type is one registry entry: node count, shape functions, quadrature rule and reference faces.
   - Per-format node-ordering maps live in `mesh/io`, and the registry refers to them.
   - Only hex types are implemented now; wedges and tets can be added later without touching operators or solvers.
5. **Differentiable from day one.**
   - Element kernels are pure JAX functions of `(X_e, C_e, ρ_e)`: energies and bilinear forms uᵀKe w and uᵀMe w.
   - Material constructors are traceable.
   - Sensitivities are a separate pass over the element kernels. Solvers and preconditioners are never differentiated.
6. **Memory-aware.**
   - Elements (and voxel z-slabs) are processed in fixed-size chunks with `scan`/`fori_loop`, the last chunk padded, accumulating into a donated output carried through the loop.
   - Indices are int32, with asserts that N_dof and every flattened index product used by an int32 gather/scatter (e.g. n_el × n_dof_e) stay below 2³¹. The asserts run in NumPy before casting.
   - The deflation projector (VᵀMX, V·c, and Π_V in counting) runs over fixed-size column chunks of the preallocated basis inside `fori_loop` (bound = current lock count) with `dynamic_slice`, upcasting one fp32 chunk at a time. All `dynamic_slice`/`dynamic_update_slice` start indices are `jnp.int32`, also under x64. MX is passed in, not recomputed. The basis capacity is a multiple of the chunk size and unused columns are zero, so `dynamic_slice` never clamps onto earlier columns. On the write side, `dynamic_update_slice` would also clamp, so the capacity is ≥ max lock count + insert width (rounded up to a multiple of the chunk size), and the host loop asserts count + width ≤ capacity before every insert (raising, or reallocating with one recompile). Unchunked mixed-precision products make XLA materialize a full fp64 copy of V.
   - Estimators include chunk temporaries and are checked against measured peaks.
7. **Conventions** live in `docs/conventions.md` and are fixed in PR 0:
   - VTK node ordering internally, converted at import.
   - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear strains.
   - fp64 by default, enabled through an explicit config call (no import side effects; PR 0 test: importing jaxmodal leaves `jax_enable_x64` unchanged).
   - Loops that are differentiated (operator chunk loops) use static trip counts (`scan`, or `fori_loop` with Python-int bounds); traced bounds appear only in solvers, which are never differentiated.
   - Index arrays are always int32.
   - Iteration vectors, returned modes, Gram matrices and Rayleigh–Ritz are always fp64. fp32 is allowed only inside preconditioners and for the device-side deflation basis. Returned locked modes come from an fp64 host copy.
   - m always denotes the LOBPCG block size (nev plus guard vectors).

## Key design decisions

- **Integration:** full integration only (hex8 2³, hex20/hex27 3³) for K and consistent M. No reduced integration, because its zero-energy modes would pollute the spectrum.
- **Mass:**
  - Consistent mass is the default everywhere. Lumping is opt-in via `mass="lumped"`: row-sum for hex8/hex27, HRZ for hex20. Row-sum lumping gives non-positive corner masses for hex20. Solid-shell elements are consistent-mass only.
  - An optional nodal (point) mass vector is added to M.
- **Hex8:** plain, fully integrated hex8 is accepted. It shear-locks in bending, so bending accuracy comes from refinement or from hex20/hex27.
- **Dirichlet BCs:** zero displacement per component in global axes on node sets. There are no skew constraints.
  - Eigen operators are purely masked: `apply_K(v) = m ⊙ K(m ⊙ v)`, the same for M, with all iterates masked.
  - Every preconditioner and linear solve instead uses A_P = m A m + s·(I − diag m), with s the mean of diag(A) over free DOFs, which puts a scaled diagonal on the constrained DOFs (independent of the unit system). Preconditioner output is masked.
- **Rigid body modes:**
  - Computed analytically per connected part, with parts defined by face adjacency; edge- or vertex-only contacts (hinges) raise an error.
  - Under Dirichlet sets, the remaining modes are R·null((I − m)R), computed via an SVD of (I − m)R with unit-normalized columns (rank tolerance σ_i ≤ 1e-8·σ_max, unit-independent), then M-orthonormalized and deflated with P = I − R Rᵀ M.
  - The AMG near-null-space is m ⊙ R: all 6 modes per part with constrained rows zeroed. It is not the reduced deflation basis, which can be empty for clamped models.
- **Preconditioning:**
  - Preconditioners approximate A = K + σM with σ > 0 (default ≈ λ_target, estimated before the solve by a few Jacobi-preconditioned LOBPCG iterations or given as a user hint), used as A_P.
  - They must be fixed, linear and SPD: symmetric pre- and post-smoothing, no inner Krylov.
  - The Chebyshev λ_max estimate uses a safety factor of 1.1.
- **Eigensolver:** generalized LOBPCG following Hetmaniuk–Lehoucq / Duersch et al. 2018:
  - SVQB orthonormalization with a drop tolerance; dropped directions are masked.
  - In Rayleigh–Ritz, masked rows and columns are zeroed off the diagonal, with B = 1 and A = 10·λ̂_max on the diagonal. They never appear as spurious low Ritz values.
  - P is dropped (masked) when ill-conditioned.
  - Implicit product updates with periodic explicit recompute, selected by a traced flag (`lax.cond`) inside the step variants, so it adds no compiled variant.
  - The deflation projector is re-applied every iteration.
  - The locked buffer is preallocated, with a count. A Python outer loop calls at most two compiled step variants (with and without P); each step call donates the iteration state. Lock insertion into the preallocated basis is a jitted `dynamic_update_slice` with the basis donated (in place).
  - A low-memory 6-block mode stores X, W, P and three product temporaries, recomputing the K and M products as needed (about 3 extra K and M applies per iteration; PR 9 measures the cost).
- **Convergence:**
  - Measured by the λ-scaled residual r = ‖Kx − λMx‖ / (|λ|‖Mx‖). This gives sin θ ≲ r·λ/gap and δλ/λ ≲ r²·λ/gap (heuristic, up to M conditioning; the rigorous form with the M⁻¹-norm residual is used in PR 9). The normwise backward error is only reported.
  - Default tol = 1e-6.
  - Accept r ≤ max(tol, c·ε·(‖K‖ + |λ|‖M‖)‖x‖ / (|λ|‖Mx‖)), with c = 10 and ‖K‖, ‖M‖ estimated by power iteration. For accurate eigenvectors it does not blow up as M becomes singular. A stagnation exit is flagged in `info`.
  - A cluster (for locking) is a set of eigenvalues whose relative gaps are < 100·tol. For shift-invert, tol here and in all slice-edge and band-edge rules is one band-wide r_acc = max over slices of max(tol, 10·τ_eff·δ_max), where each pair is accepted at r_i ≤ max(tol, 10·τ_eff·δ_i) and r_acc is the max of these thresholds over returned pairs and the edge neighbours used for counting; the cost model caps δ_max ≤ 1 when choosing slices. The PR 9 test threshold of 1e-4 differs on purpose: it only decides whether to compare by subspace.
  - Block size m = nev + max(5, 0.2·nev).
  - Chunk and lock cutoffs must fall in a gap.
- **Precision:**
  - The attainable residual floor is about ε·κ, with κ = λ_max/λ_min: roughly 1e-8–1e-6 in fp64, fine for 0.1% targets but O(1) in fp32.
  - Hence fp32 never goes into X, R, P or the returned modes.
  - PR 9 measures the floor.
- **Frequency bands (PRs 14–15):**
  - **Strategy:** chosen by a cost model.
    - Chunked LOBPCG from 0 (then filtered to the band) is used while it is cheaper and its deflation basis (every mode below the band's upper edge) fits in memory; otherwise shift-invert slices are used.
    - On the Mac test sizes the predicted cost crossover (not memory) puts LOBPCG up to mode index ≲ 1200. At 10M hex20 on 80 GB the basis limit (≈ 900–1200 modes, see Memory budget) sends bands near index 1000 to shift-invert. At 100M, bands need multi-GPU.
    - The stochastic count is controlled by `verify` (implemented in PR 14) for every strategy: `None` (default) lets the cost model enable it only when it is predicted to cost no more than the solve; `True`/`False` force it. The predicted cost is always reported in `info`.
    - Near mode index 1000 (relative gap g ≈ 7e-4), shift-invert costs ~10³–10⁵ MINRES iterations per Lanczos step (worst case ∝ ln(1/τ)/δ, δ the relative distance from σ to the nearest eigenvalue; Krylov deflation of the few eigenvalues nearest σ moves it toward the low end; PR 15 calibrates this), and counting costs d ≈ 25/g ≈ 3.5e4 solves per probe block.
  - **[0, f_max]:** chunked LOBPCG until λ > λ_c, with the cutoff λ_c = (2πf_max)².
  - **Band edges:** every cluster (relative gap < 100·tol) that reaches into the band is returned whole. The counting edge (and σ for the [0, f_max] count) is placed at the middle of the gap between the last returned mode and the first computed mode beyond it, and g is that gap. The same applies to f₁ and f₂; an edge at λ = 0 has no counting edge (the step polynomial already covers it). When no mode is returned, each counting edge sits midway between its band edge (2πf)² and the nearest converged neighbour outside the band, and g is the smaller of the edge gaps.
  - **Interior bands [f₁, f₂]:**
    - Block shift-invert Lanczos on (K − σM)⁻¹M with σ in the band, fixed block size b = 6 (≥ the largest expected multiplicity, so repeated eigenvalues are found), the same column masking as LOBPCG, full M-reorthogonalization, and a final Rayleigh–Ritz with true residuals on (K, M).
    - Inner MINRES with absolute-value preconditioning by the SPD multigrid of K + σM (Vecharynski–Knyazev 2013), with a fixed inner tolerance τ = 1e-8 on the explicitly computed 2-norm residual ‖r_in‖₂ ≤ τ‖Mq‖₂ (checked at exit). MINRES keeps iterating until the explicit 2-norm residual meets τ or stagnates (no new minimum of the explicit residual over 5 consecutive checks, one check every 10 iterations; a hard iteration cap sets an `info` flag). If it stagnates above τ, τ_eff is raised to the measured floor, carried into r_acc and flagged in `info`; PR 15 measures this floor. The constrained diagonal of K − σM is mean diag(K) + σ·mean diag(M) over free DOFs (positive). The attainable outer λ-scaled residual is then ≈ τ·δ_max, with δ_max = max over the slice of |λ − σ|/|λ|; acceptance uses r_acc = max(tol, 10·τ_eff·δ_max), with τ_eff = τ unless raised below.
    - Each slice's Lanczos run stops when every Ritz value in its interval meets the slice's running r_acc (from its current δ_max and τ_eff) and at least one converged Ritz value lies beyond each slice edge, in a gap ≥ 100·r_acc. An edge at λ = 0 (a clamped [0, f_max] run on shift-invert) needs no neighbour beyond it, since no eigenvalue lies below 0 (free bodies are deflated first). After all slices finish, every edge is re-checked against the final band-wide r_acc; an edge that fails is widened or merged into the neighbouring slice. If (2πf₁)² is below the Rayleigh-quotient upper bound of λ₁ from the σ pre-estimate, the band is solved as [0, f₂] and filtered to [f₁, f₂], so a missing lower neighbour never blocks the stop. Block size b = 6 finds larger exact multiplicities only slowly (through rounding); the count catches a shortfall.
    - Slice widths come from the cost model (MINRES iterations ∝ ln(1/τ)/δ, with δ the relative distance from σ to the nearest eigenvalue, plus an outlier term for the few eigenvalues nearest σ; calibrated in PR 15). The cost model also counts deflation-basis memory (n_dof × number of vectors) and the predicted counting cost, and rejects strategies over the memory budget. Slice edges, like lock cutoffs, are placed in the largest relative gap (≥ 100·tol) within the cost model's allowed window, so each cluster belongs whole to one slice: the cost model sets provisional edges, each edge is snapped to the largest gap among converged Ritz values in its window, and only then is the stop rule evaluated. The cost model includes each slice's counting degree (same edge rule, g = that slice-edge gap). If τ_eff makes 100·r_acc exceed every gap in the window so that slices collapse into one, `info` is flagged. Before merging, each slice keeps only the Ritz pairs inside its own interval that meet r_acc. Remaining duplicates at slice edges are removed by M-orthogonalization, dropping a vector whose projected M-norm (M-normalized) is < 0.5 (a new vector has ≈ 1, a duplicate ≲ 2·r·λ/gap). The projected norm only detects duplicates; kept vectors are the original Ritz vectors. If vectors from both slices are kept for one cluster, Rayleigh–Ritz is run on the merged cluster; kept vectors must meet r_acc, otherwise Rayleigh–Ritz over that edge window is repeated and `info` is flagged. If the window has no gap ≥ 100·r_acc, it is widened or merged into the neighbouring slice.
    - Neither the Lanczos nor the counting steps need M⁻¹ beyond the one M-solve per probe. Consistent M costs nothing else extra there.
    - MINRES needs a preconditioner that is symmetric up to rounding (symmetric smoother and transfer pairs). fp32 V-cycles are used only inside LOBPCG and PCG, never inside MINRES (fp32 asymmetry would floor τ near 1e-6).
  - **Completeness count (stochastic, at scale):**
    - Estimate s = ‖Π_V q(T) Π_V y‖²_M, with:
      - T = (K + σM)⁻¹M with σ ≈ λ_c ([0, f_max]) or σ at the band's upper edge (interior band);
      - Π_V = I − VVᵀM, where V includes the computed modes and R.
    - Probes are processed in blocks whose size the PR 14 memory model sets and counts (recurrence, PCG and preconditioner workspace per probe). y = M_P⁻¹g with g ~ N(0, M) masked. g comes from element-wise Cholesky factors of Me plus √m_i·ξ_i for nodal masses, followed by one Jacobi-PCG solve per probe with the masked M_P (scaled diagonal on constrained DOFs), to the same tolerance τ_c as below.
    - q is a Chebyshev–Jackson step polynomial ([0, λ_c]) or band-pass polynomial (interior band, deflated by the modes found in the band and the converged out-of-band neighbours within the polynomial's transition width of each band edge).
    - The degree d is chosen numerically so that:
      - q² ≥ 0.95 on the in-band interval up to the last returned mode, [0, λ_c(1 − g/2)] (up to the band edge (2πf)² when none is returned); this sets d ≈ 25/g;
      - the Jackson-tail bias Σ q(λ_i)² over the out-of-band spectrum is ≤ 0.05, estimated with a Weyl-type density N(λ) ∝ λ^a, with a fitted to the computed modes (3/2 for bulky solids, smaller for thin structures). (Bounding n_dof·max q² instead would need d ≈ 75/g or more.)
    - A missed mode inside the gap next to the cutoff is detected with lower probability.
    - Inner PCG tolerance τ_c ≲ 0.01/(d·√n_dof). PR 14 measures how solve errors actually propagate through the recurrence (worst case O(d²), not O(d)). Counting solves use the fp64 V-cycle. If τ_c is below the attainable PCG accuracy, `info` marks the count as unverifiable instead of passing it. At 10M DOFs near index 1000 (τ_c ≈ 1e-10) this is expected to happen often.
    - One missing mode makes the estimate behave like χ²ₙ/n. With 50 probes, P(missed mode undetected) ≈ 0.2%.
    - A slice passes when its estimate is < 0.5.
  - **Exact count:** at reference sizes (≤ ~15k DOF), from the inertia of dense `scipy.linalg.ldl` of K − σM restricted to the free DOFs (the constrained diagonal of A_P could otherwise add spurious negative pivots).
  - **Assumed band position:** interior bands sit at mode indices up to ~1000 and contain ≤ 200 modes. This is a working assumption (the user doesn't know typical bands yet); PR 14 reports the counting cost as a function of mode index, so the assumption can be revisited. This bounds the polynomial degree, and so the cost of counting.
- **Preconditioners:**
  - Voxel meshes: geometric multigrid with semi-coarsening: an axis stops coarsening once it has ≤ 2 elements.
  - Unstructured meshes: p-multigrid (hex20/hex27 → hex8), then smoothed-aggregation AMG. AMG setup runs in pyamg on the host; the V-cycle runs on the device with a self-written block-ELL SpMV.
  - No `jax.experimental.sparse` in hot paths.
- **Shells:** solid-shell elements (3 DOFs per node) following Klinkel–Gruttmann–Wagner (1999):
  - ANS for transverse shear (Bathe–Dvorkin edge midpoints: γ_ξζ at (ξ, η) = (0, ±1), γ_ηζ at (±1, 0)).
  - ANS for the thickness strain at the 4 corner points.
  - EAS built with the center Jacobian J₀, by default with 5 parameters: 1 thickness (linear in ζ) and 4 Simo–Rifai membrane parameters (a project choice within the KGW framework). ∫_□ M(ξ) dξ = 0 in the parent domain holds for the patch test.
  - Laminates with n Gauss points per ply.
- **Sensitivities:**
  - Formula: dλ/dp = vᵀ(∂K/∂p − λ∂M/∂p)v / vᵀMv.
  - Computed as a chunked `vmap(grad(element energy))` pass with v held fixed, in fp64.
  - Accuracy: the PR 21 finite-difference tests use dense `eigh` modes on coarse meshes (κ ≲ 1e4, accurate to ~1e-12); the LOBPCG-based gradient tests use r ≤ 1e-8. Optimization runs use r ≤ max(1e-6, measured floor), and the sensitivity error is O(r·λ/gap).
  - Clusters use the sensitivity matrix f_sk = v_sᵀ(K′ − ½(λ_s + λ_k)M′)v_k over an M-orthonormal cluster basis, built from the bilinear forms (symmetric; equal to the cluster-mean form for an exact cluster). The optimization cluster tolerance is a relative gap of 1e-2 (a working choice).
  - In MMA, each cluster uses the Seyranian / Du & Olhoff conditions:
    - diagonal constraints β ≤ λ_s + f_ss·Δρ, with λ_s the computed eigenvalue;
    - off-diagonal constraints |f_sk·Δρ| ≤ ε_off (s < k), as paired inequalities, with ε_off = 1e-2·|β| for the current iterate's β (so the constraint stays linear in the MMA subproblem). By Gershgorin, to first order, β can overshoot the true λ_min by at most (N − 1)·ε_off.
  - Per-mode diagonal constraints alone are invalid, because f_kk depends on the basis.
  - Deflation and masking leave the formula unchanged.
- **Scaling:** β, KS and MMA work on λ/λ_ref (λ_ref = lowest elastic λ of the initial design), so p (KS parameter p = 50, fixed per example and recorded in `docs/optimization.md`) and the MMA constants are dimensionless. Every MMA constraint is normalized: Σρ̄_e v_e/V* − 1 ≤ 0, mass/M₀ − 1 ≤ 0, eigenvalue constraints in λ/λ_ref, detJ constraints in detJ/detJ₀. FFD control-point boxes are fractions of the lattice size. β starts at 0.99·λ_min/λ_ref of the initial design (> 0) and has finite MMA box bounds [0, 100]; reaching the upper bound is an error.
- **Optimizer:** MMA implemented in JAX (~200 LOC). No OC, because eigenvalue sensitivities change sign.
- **Scatter vs gather:** XLA:CPU scatter is effectively serial. PR 5 benchmarks scatter against an inverse-map gather on each backend, and the default may differ per backend.
  - The gather runs on an RCM-ordered mesh (RCM is built in PR 5), chunked by node range.
  - Element contributions go into a padded, fixed-size element window per range, which is counted in the estimator.
- **Dependencies:**
  - Python 3.12.
  - Core: `jax` (pinned minimum version), `numpy`, `scipy`, `meshio`, `pyamg`, all with macOS arm64 wheels.
  - `pymetis` is an optional extra (only PR 19 needs it), pinned to a version with macOS arm64 wheels.
  - Versions are locked with `uv.lock`.
  - No SLEPc, CHOLMOD or pypardiso.
  - meshio's `.inp` reader doesn't handle parts or assemblies; a small custom reader is added only if needed.

## Memory budget

All sizes are fp64; GB = 10⁹ bytes.

| Item | 1M hex20 (Mac test) | 2M voxel (Mac test) | 10M hex20 | 100M voxel |
|---|---|---|---|---|
| One vector | 0.008 GB | 0.016 GB | 0.08 GB | 0.8 GB |
| Operator | Ke ≈ 1.2 GB (symmetric) | < 0.1 GB | Ke ≈ 12.2 GB (symmetric) or on-the-fly | < 1 GB |
| LOBPCG, 9 blocks × m | m = 25: ≈ 1.8 GB | m = 15: ≈ 2.2 GB | m = 30: ≈ 22 GB | m = 10: ≈ 72 GB (multi-GPU only) |
| LOBPCG, 6 blocks | — | — | m = 30: ≈ 14 GB | m = 8: ≈ 38 GB |
| Rayleigh–Ritz update temporaries (≈ 2 blocks × m; also hosts the (chunk + m)-block deflation-projector temporaries, chunk ≤ m) | ≈ 0.4 GB | ≈ 0.5 GB | ≈ 5 GB | m = 8: ≈ 13 GB |
| Preconditioner workspace (≈ 4 blocks × m) | ≈ 0.8 GB | ≈ 1 GB | ≈ 10 GB | m = 8: ≈ 26 GB (fp32: 13 GB) |
| Multigrid | AMG ≈ 0.4 GB (device); pyamg peak 2–3 GB (host) | ≈ 0.2 GB | AMG ≈ 3–4 GB; pyamg peak ≈ 20+ GB (host) | ≈ 9.9 GB incl. int32 indices (fp32: ≈ 5.2 GB) |
| Gather path (if chosen) | chunked by node range | — | chunked by node range | — |
| Locked modes / deflation basis | small | small | n × 0.08 GB (fp32: half) | n × 0.8 GB (fp32: half) |
| Max locked modes (fp32 basis; 6 blocks, fp32 preconditioner and multigrid) | > 2000 | > 2000 | ≈ 900 (stored Ke) – 1200 (on the fly), with fp32 AMG from PR 18 | ≲ 8 |

- **Mac (64 GB, unified memory):**
  - Both test columns fit easily.
  - 10M is an optional, tight benchmark. Order the work as: AMG setup → free the assembled A → allocate LOBPCG.
  - An `np.memmap` is used only if the locked modes exceed RAM.
- **80 GB GPU:**
  - GPU runs set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (the JAX default is 0.75). `XLA_PYTHON_CLIENT_PREALLOCATE=false` is not used, because it fragments memory. The usable pool is then ≈ 76 GB; the CUDA context and library workspaces sit in the remaining 4 GB. The estimators budget against the usable pool, not 80 GB. This is conservative: an "80 GB" card has ≈ 85·10⁹ bytes.
  - 10M fits.
  - 100M is tight: ≈ 69.2 GB with 6 blocks and m = 8 (nev = 3 per chunk): 38.4 for the blocks, 12.8 for Rayleigh–Ritz, 12.8 for the fp32 preconditioner workspace and 5.2 for the fp32 multigrid (values plus int32 indices). The operator, a bool mask, fp64 diagonals, s_K/s_M and nodal masses take ≈ 3–4 GB more. The remaining ≈ 3–4 GB holds the fp32 deflation basis, so ≲ 8 modes in total at 100M, including any rigid modes R.
  - Larger blocks at 100M need multi-GPU (PR 19).
- The PR 5/6/9/11/13/14/15/16/18 estimators reproduce this table and are validated against measured peaks where the estimate is ≥ 2 GB.

## Development environment

- **Development:** Apple M3 Max with 16 cores and 64 GB of unified memory, using the JAX CPU backend in fp64. The acceptance tests run at 1M (hex20) and 2M (voxel) DOFs. 10M is an optional benchmark.
- **No Mac GPU:** the GPU isn't usable for this project. Metal GPUs have no fp64 hardware, and the `jax-metal` plugin is experimental with incomplete op coverage.
- **Local checks:** `scripts/check.sh` runs lint plus the fast suite, `pytest -m "not slow and not gpu"`, which must finish in under 5 minutes on the M3. Remote CI comes later.
- **Merge gate:** `pytest -m "not gpu"`, slow tests included.
- **GPU checks:** GPU acceptance criteria go into the "Deferred GPU checks" list and are run in PR 24 whenever a CUDA machine becomes available. They don't gate other milestones.
- **Mac acceptance budgets:** the numbers below are provisional; PR 9 finalizes them once matvec throughput has been measured, and later PRs use the PR 9 values.
- **Benchmarks:**
  - Each measurement runs in a fresh subprocess and reports peak RSS minus baseline RSS.
    - `memory_stats()` is unavailable on CPU.
    - `ru_maxrss` is in bytes on macOS and KiB on Linux; the harness branches by platform. GPU runs use the `memory_stats()` peak.
    - For the 10M runs, the peak `phys_footprint` (`proc_pid_rusage`, field `ri_lifetime_max_phys_footprint`) is also recorded, because memory compression can make RSS undercount.
  - JSON results go to `benchmarks/results/` and record:
    - jax and jaxlib versions;
    - STREAM triad bandwidth, both jitted and a small multithreaded C triad using pthreads (Apple clang has no OpenMP by default), skipped and recorded as skipped when no C compiler is found;
    - XLA:CPU intra-op thread pool size (unless pinned) and P/E-core setup;
    - measured fp64 peak;
    - time, peak memory and compile time.
- **Compilation cache:**
  - The persistent cache is enabled in tests, with `jax_persistent_cache_min_compile_time_secs = 0` and `jax_persistent_cache_min_entry_size_bytes = -1` (no size restriction). PR 0 checks and records whether it actually stores entries on CPU. If it doesn't, the cache is disabled and the tests don't depend on it.
  - Compile-time tests (PRs 5 and 18; PR 9 only counts compiles) run in a fresh subprocess with the cache disabled and time `jit(f).lower(...).compile()`.
- **Multi-device tests:**
  - They live in `tests/multidevice/`, which is in `collect_ignore`. `jax_num_cpu_devices` is used if the pinned jax supports it, set before the backend initializes.
  - A wrapper test re-runs them in a fresh subprocess with `XLA_FLAGS=--xla_force_host_platform_device_count=…` set before jax is imported.
- **Data:**
  - Tiny test meshes live in `tests/data/`.
  - Large benchmark meshes are generated by scripts in `benchmarks/meshes/` and git-ignored.
- **Solver output:** solvers return an `info` dict (iterations, residual history, timings, stagnation flags). No logging framework.

## Package layout

```
src/jaxmodal/
  config.py    dtype / x64 setup
  memory.py    memory estimators
  mesh/        data model, io (incl. node-ordering maps), voxel generator, reordering, partition
  materials/   anisotropic C, Bond rotations, orientations
  elements/    registry, quadrature, shape functions, hex8/20/27, solid-shell, element-wise Me Cholesky (for probes)
  operators/   dofmap, assembled (oracle), matrix-free, voxel stencil, constraints (masking, A_P, rigid modes, deflation)
  linalg/      pcg, minres, block-ELL spmv, dense inertia
  precond/     block jacobi, chebyshev, gmg, pmg, amg
  eigen/       lobpcg, counting, shift-invert lanczos, api (solve_modes), reference (scipy)
  post/        normalization, MAC, effective mass, stresses, export
  opt/         sensitivities, mma, ffd, simp, filters
tests/  benchmarks/  examples/  docs/  scripts/
```

Each PR ships its docs (with LaTeX math) in `docs/`, listed below as **Docs**. PRs are numbered in execution order; the dependency table at the end is authoritative.

## M0: Foundations

### PR 0: Project scaffold
- Contents:
  - pyproject, `uv.lock` and lint config.
  - pytest markers `slow` and `gpu`.
  - `scripts/check.sh`.
  - `config.py` (an explicit x64 switch).
  - Persistent compilation cache.
  - A benchmark harness (subprocess RSS, STREAM triad, fp64 peak, versions).
  - LICENSE (MIT), version 0.x.
- Docs: `README.md`, `docs/conventions.md`.
- Tests:
  - An fp32 path stays fp32 with x64 enabled.
  - Importing jaxmodal leaves `jax_enable_x64` unchanged.
  - Index arrays from `argsort`, `nonzero` and `arange` are cast back to int32.
  - The harness reports a stable RSS delta for a known allocation.
  - Buffer donation works on CPU: there is no "donated buffers were not usable" warning, and `x.is_deleted()` is true for the donated input. If the pinned jaxlib doesn't donate on CPU, the minimum version is raised; failing that, the estimators budget without donation.
  - Whether the persistent cache stores entries on CPU is recorded; if not, the cache is disabled and cache-dependent checks are skipped.

### PR 1: Mesh data model and parser
- Contents:
  - `Mesh` dataclass: coordinates, element blocks keyed by type, node and element sets, material and orientation IDs.
  - meshio readers for Abaqus `.inp` (flat), Gmsh and VTU.
  - Per-format node-ordering maps for hex20/hex27.
  - Structured voxel mesh generator, with an occupancy mask that drops inactive elements and nodes (e.g. a voxelized sphere). The `Mesh` carries grid size and spacing for voxel dispatch.
- Docs: `docs/mesh.md`.
- Tests:
  - Read → write → read round-trips.
  - Orderings match hand-built references for every format.
  - hex8 corner Jacobians are positive.
  - Voxelized sphere: element and node counts and volume match a NumPy count; no orphan nodes.

### PR 2: Materials
- Contents:
  - Traceable 6×6 C constructors: isotropic, transversely isotropic, orthotropic (ν_ij/E_i = ν_ji/E_j convention), full anisotropic.
  - Positive-definiteness check, relative: λ_min(C) > 1e-12·λ_max(C).
  - Stress and strain Bond matrices.
  - Orientations: local material axes per element, from Euler angles or two direction vectors (test: both give the same rotation for equivalent input).
  - Density.
- Docs: `docs/materials.md`.
- Tests:
  - Bond rotation equals `einsum` tensor rotation.
  - Isotropic C is invariant under rotation; transversely isotropic C is invariant under rotation about its axis.
  - C stays positive definite after rotation; a non-PD C is rejected.
  - The orthotropic constructor's compliance matrix equals the hand-built S from (E_i, ν_ij, G_ij) to 1e-14.
  - `jax.grad` with respect to angles and constants matches finite differences.

## M1: Elements and operators

### PR 3: Element registry and hex elements
- Contents:
  - Registry.
  - Gauss quadrature.
  - hex8, hex20 (serendipity) and hex27 shape functions and derivatives.
  - B-matrix.
  - Batched Ke and Me (consistent and lumped), plus the bilinear forms.
  - Jacobian check at quadrature points.
- Docs: `docs/elements.md`.
- Tests:
  - Partition of unity.
  - Linear completeness.
  - Exactly 6 zero eigenvalues of Ke (|λ| < 1e-10·λ_max, with the 7th above 1e-6·λ_max), on distorted elements too: aspect ratio ≤ 2, ν ≤ 0.49, random nodal perturbations ≤ 20% of the edge for hex8 and ≤ 10% for hex20/hex27 (midside nodes included), with detJ > 0 at nodes and quadrature points. The 7th eigenvalue is a quantity of element shape, ~1e-4 of λ_max for hex20/hex27 even undistorted at aspect ratio 2, which is why the gate is 1e-6.
  - On mildly distorted elements (bound stated in the docs, e.g. midside offsets ≤ 5% of the edge length for hex27 row-sum), lumped masses are all > 0. Lumping raises an error when any lumped nodal mass is ≤ 0. Total mass equals ρV per direction, with V the quadrature volume.
  - Matches a NumPy reference implementation to 1e-12.
  - `jax.grad` of uᵀKe(X)u matches finite differences (rel. err. < 1e-6).
  - The Jacobian check raises on an inverted element.

### PR 4: DOF map, assembled reference operator, masking
- Contents:
  - Global DOF numbering (int32).
  - Per-group element-to-DOF arrays.
  - SciPy CSR K and M via COO; M consistent or lumped (diagonal from the PR 3 lumped Me).
  - Nodal mass vector.
  - Masking and A_P.
- Docs: `docs/operators.md`.
- Tests:
  - Distorted multi-element patch test, with linear displacement imposed by lifting inside the test:
    - Straight-edged distortion (midside nodes at edge midpoints): constant stress to 1e-10 relative to ‖σ‖.
    - Curved hex20/hex27 (midside offset amplitude stated): constant stress to 1e-10 relative to ‖σ‖ at every quadrature point. 3³ Gauss integrates cof(J)ᵀ∇̂N exactly here: for triquadratic geometry the integrand has degree ≤ 5 per direction.
  - Clamped model: the free-DOF block of the masked K and M equals SciPy's reduced matrices, so there are no spurious eigenvalues.
  - A_P is SPD on a clamped model, and Jacobi built on it has no zero divisions.
  - Nodal masses add to the diagonal of the SciPy M exactly.
  - Rigid translation mass 1ᵀM_x1 = ρV + Σ nodal masses, for consistent and lumped M, to 1e-12 relative.
  - The int32 asserts raise on fake sizes just above 2³¹.

### PR 5: Matrix-free unstructured operator
- Contents:
  - Two modes: stored Ke, and on-the-fly computation from precomputed invJ·detJ per quadrature point.
  - Optional per-element stiffness and mass scaling arrays s_K and s_M, so density design on unstructured meshes stays possible.
  - Chunked, block-native apply.
  - Scatter (`segment_sum`) vs chunked inverse-map gather, benchmarked per backend.
  - Consistent and lumped `apply_M`; `diag_K`, `diag_M`, nodal masses.
  - Memory estimator.
  - Node-local numbering.
  - RCM node reordering (`scipy.sparse.csgraph.reverse_cuthill_mckee`) in `mesh/`.
- Docs: `docs/operators.md`.
- Tests:
  - Stored and on-the-fly, each with scatter and with gather, on an RCM-reordered mesh with random s_K and s_M, match the scaled PR 4 SciPy K and M (consistent and lumped, permuted) to 1e-12 relative, for m = 1 and a block m > 1.
  - `diag_K`/`diag_M` equal the SciPy diagonals to 1e-12.
  - In stored-Ke mode with the per-backend default path (and the gather path), the estimator is within 20% of the measured peak at 2M hex20 DOFs.
  - CPU scatter vs gather benchmark; sets the Mac default.
  - Deferred GPU check: scatter vs gather benchmark; sets the GPU default.
  - With the chunk size fixed and the cache disabled, compile time (hex20, lowered with `ShapeDtypeStruct` inputs) at 1M elements is ≤ 1.5× that at 10k elements.

### PR 6: Voxel stencil operator
- Contents:
  - K matvec over z-slabs: `scan` + `dynamic_slice` with a 1-layer halo; per slab, 8 shifted corner slices → einsum with the reference Ke → shifted adds into a donated output.
  - Per-element scaling arrays s_K and s_M, as in PR 5.
  - Occupancy-masked voxel meshes: inactive elements contribute zero; a grid↔mesh node map handles dropped nodes.
  - `apply_M` (consistent and lumped), `diag_K`, `diag_M`, nodal masses.
  - Memory estimator, including slab temporaries.
  - Restriction: one material and orientation per material ID (phase).
- Docs: `docs/operators.md`.
- Tests:
  - `apply_K`, `apply_M` (consistent and lumped), `diag_K`, `diag_M` and nodal masses match PR 5 on the same mesh, for a full block and for the PR 1 voxelized sphere.
  - Two materials or orientations in one phase raise an error.
  - `jax.grad` with respect to s_K and s_M matches finite differences.
  - Mac acceptance: for a matvec plus six preallocated (n_dof, 24) fp64 arrays (≈ 2.3 GB) at 2M DOFs, the estimate is within 20% of the measured peak.

### PR 7: Rigid body modes and deflation
- Contents:
  - Rigid modes per connected part, with rotations about each part's centroid.
  - Automatic remaining modes under Dirichlet sets.
  - M-orthonormalization.
  - Projector P = I − R Rᵀ M.
- Docs: `docs/boundary_conditions.md`.
- Tests:
  - ‖KR‖ ≤ 1e-12·‖K‖‖R‖.
  - ‖RᵀMR − I‖ ≤ 1e-12.
  - Free block: u_x = 0 on x = 0 leaves 3 modes; adding u_y = 0 on y = 0 leaves 1 (T_z); adding u_z = 0 on z = 0 leaves 0.
  - Two disconnected parts give 12 modes; two voxels joined only along an edge raise the hinge error.
  - Projected random vectors are M-orthogonal to R.

## M2: First modes end-to-end

### PR 8: Reference eigensolver and verification suite
- Contents:
  - SciPy `eigsh` shift-invert (negative shift) on assembled matrices, with its largest supported size stated.
  - Dense inertia via `scipy.linalg.ldl` (≤ ~15k DOF) (test: on a clamped and a free model, the `ldl` count of K − σM over free DOFs equals the dense `eigh` count below σ, for σ mid-gap between several consecutive eigenvalue pairs).
  - In-test oracles:
    - A rectangular block with sliding BCs (exact separable solution, including orthotropic with material axes aligned to the block axes); also run with lumped mass, with fitted rate ≥ 1.8 (hex8, hex27 row-sum, hex20 HRZ).
    - Rayleigh–Ritz (Visscher et al. 1991) for a free anisotropic parallelepiped, here with a Legendre basis, dense `eigh`, converged when successive degrees change every compared eigenvalue (modes 7 to the highest compared mode, sorted) by a relative amount < 1e-6.
    - The Srinivas–Rao exact 3D solution for thick orthotropic and cross-ply simply supported plates.
  - Published benchmarks:
    - NAFEMS FV52 (simply supported thick plate). The docs state where u_z = 0 is applied; modes 1–3 are in-plane rigid modes, which exercises the automatic partial rigid modes.
    - NAFEMS FV12 (free thin plate), run with hex20/hex27 only, because hex8 locks at L/t = 200. The in-plane refinement is stated (20×20 hex20 is ≈ 0.6% above converged for modes 12–13, the worst; free-edge layers slow convergence). The measured values are recorded in `docs/verification.md`.
    - The Lamb sphere on voxel meshes.
- Docs:
  - `docs/eigensolvers.md`.
  - `docs/verification.md`, with exact setups, sources and reference values. It also defines simply supported BCs on solids face-wise, over the whole edge face through the thickness (used again in PR 16).
- Tests:
  - **0.1% target:** sliding block, Visscher Ritz and Srinivas–Rao within 0.1% (relative error in f) at stated hex20/hex27 meshes.
  - **NAFEMS:** a mode is gated against its target (1%) only when a fine hex27 run is within 0.5% of it. The fine hex27 run is refinement-converged (edge-graded in-plane mesh; the last refinement changes f by < 0.1%), with its mesh stated in `docs/verification.md`. Otherwise it is gated against the fine hex27 run (1%) and the gap to the target is recorded in `docs/verification.md` (e.g. FV12 modes 10–13 sit ≈ 1% below the quoted targets). For FV52, the docs fix the u_z = 0 placement that the fine hex27 run shows reproduces mode 4.
  - **Convergence rates:**
    - Measured against the exact sliding-block and Srinivas–Rao solutions on affine meshes. Element faces conform to ply interfaces. Fitted rate ≥ 1.8 for hex8 (sliding block and thick Srinivas–Rao, L/t = 10, only, because of locking) and ≥ 3.6 for hex20/hex27, over ≥ 3 levels (finest error ≥ 100× the `eigsh` floor ≈ ε·κ) in the asymptotic range. Richardson extrapolation is used only to estimate errors, never to fit rates.
    - Voxel Lamb sphere, against the exact solution for the sphere of equal (voxelized) volume, over ≥ 5 levels fixed in the docs: the max relative error over the two lowest 5-fold clusters (selected by cluster, not by index; torsional and spheroidal modes are told apart by their radial kinetic-energy fraction, ≈ 0 for torsional) drops by ≥ 2× from the coarsest to the finest level (staircase errors oscillate, so the least-squares rate is reported only), with split clusters compared by their mean.
  - **Mode matching:** free bodies (Visscher, Lamb, FV12) are compared from mode 7 on, FV52 from mode 4 on. Modes are paired with Srinivas–Rao by M-weighted MAC (defined in PR 17, computed in-test here) and by the (m, n, family) labels; degenerate sets (analytic relative gap < 1e-2, e.g. (m, n)/(n, m) on a square plate) are paired by subspace (summed MAC over the set).
  - **Upper bound:** consistent-mass eigenvalues converge from above on nested meshes of parallelepiped elements, with exact geometry and conforming essential BCs; checked as λ_k(fine) ≤ λ_k(coarse) + 10·ε·λ̂_max, from mode 7 for free bodies and mode 4 for FV52 (not for the voxel Lamb sphere, whose geometry is not nested).

### PR 9: Generalized LOBPCG and public API
- Contents:
  - LOBPCG as under Key design decisions, including fixed shapes, the 6-block mode and floor-aware acceptance.
  - Jacobi on A_P, with the σ pre-estimate (a few Jacobi-LOBPCG iterations or a user hint; test: the pre-estimate is within 2× of λ₁ and a user hint skips it).
  - Chunked mode computation; the deflation basis is optionally fp32, returned modes are fp64.
  - LOBPCG workspace estimator.
  - Public API: `solve_modes(mesh, materials, bcs, n_modes, ...) → (λ, V, info)`.
  - `examples/` script per milestone, also run as a smoke test.
  - Fixes the Mac acceptance budgets.
- Docs: `docs/eigensolvers.md` (algorithm, convergence theory, memory model, measured floor).
- Tests, at reference sizes against PR 8, run at tol = 1e-10 with an fp64 deflation basis:
  - Relative λ error ≤ max(1e-8, 10·ε·λ̂_max/λ), with λ̂_max from the power iteration (dimensionless, so independent of units), for modes with relative gap ≥ 1e-2 (clusters below that gap are checked by subspace, as below).
  - Principal angles: sin θ ≤ 10·r_M·λ/gap, with the reference residual required to be ≥ 10× smaller than r_M. Angles are in the M-inner product, the gap is absolute, and r_M is the λ-scaled residual in the M⁻¹-norm with x M-normalized (dense at reference sizes). Clusters with relative gap < 1e-4 are compared by subspace.
  - Exact multiplicities on a square-section beam and a cube.
  - Free-free: ‖RᵀMX‖ ≤ 1e-10.
  - Modes with relative gap in [1e-4, 1e-2) meet λ error ≤ 10·tol, cluster-wise (sorted).
  - tol below the floor sets the stagnation flag and exits before the iteration cap; a rank-deficient starting block (duplicated columns) is masked by SVQB and still converges; a forced ill-conditioned P is dropped.
  - A cluster larger than the guard vectors at a chunk boundary raises an error that names the required guard count.
  - Near-singular M (void mass ≈ 1e-12 and void stiffness ≈ 1e-9 relative to solid, SIMP-like): converges to max(1e-8, floor) within 500 iterations.
  - The λ-error and principal-angle tests above run in both the 9-block and the 6-block mode; the 6-block λ match the 9-block λ to max(1e-8, 10·ε·λ̂_max/λ).
  - The workspace estimator (9- and 6-block modes) is within 20% of the measured peak at 1M DOFs with m = 48 (estimate ≥ 2 GB in both modes).
  - With an fp32 basis, the compiled projector (MX passed in) has `memory_analysis().temp_size_in_bytes` ≤ 1.01·(chunk + m)·n_dof·8 B at n_dof ≥ 1e5 in the fused per-chunk form (Y −= V_c(V_cᵀMX)), unchanged between basis capacities differing by ≥ 4×; a lock count that is not a multiple of the chunk size gives the same result as the unchunked fp64 projector.
  - Chunked solve with an fp32 basis (≥ 3 chunks, tol = 1e-6): λ matches the fp64-basis run to ≤ max(1e-8, 10·ε·λ̂_max/λ) for modes with relative gap ≥ 1e-2 (10·tol otherwise), ‖VᵀMV − I‖ ≤ 1e-5 over all returned modes (no duplicates), and the count matches the PR 8 dense inertia.
  - Lock insertion updates the basis in place: the donated input `is_deleted()` and the buffer pointer is unchanged.
  - Inserting into a nearly full basis (count + width > capacity) trips the host assert, and the earlier columns stay bit-identical.
  - The compiled step function's `memory_analysis().temp_size_in_bytes` is unchanged between basis capacities differing by ≥ 4×.
  - The floor (attainable λ-scaled residual vs ε·λ̂_max/λ) is measured at reference sizes and recorded in `docs/eigensolvers.md`.
  - The 6-block mode's extra cost (time per iteration vs 9-block) is measured and recorded in `docs/eigensolvers.md`.
  - The step function's `memory_analysis` temp (maximum over both `lax.cond` branches, including the explicit recompute) is within the PR 9 estimator.
  - The compiled step function compiles at most twice per solve (counted with `jax.monitoring` compile events) (the σ pre-estimate is a separate Jacobi-only run, counted separately).
- **Milestone:** first end-to-end modes in pure JAX.

## M3: Preconditioning (core of the project)

### PR 10: Linear algebra kernels and smoothers
- Contents:
  - PCG and MINRES.
  - Block-ELL SpMV, with the padding ratio reported and a sliced-ELL fallback.
  - 3×3 node-block Jacobi, with the node blocks computed from the element kernels.
  - Chebyshev smoother with eigenvalue estimation (safety factor 1.1).
  - All applied to A_P.
- Docs: `docs/preconditioners.md`.
- Tests:
  - PCG and MINRES match `spsolve` to 1e-10; MINRES also on an indefinite K − σM with σ inside the spectrum.
  - Block-ELL and sliced-ELL match SciPy CSR to 1e-12.
  - Node blocks equal the 3×3 diagonal blocks of the PR 4 SciPy A_P to 1e-12.
  - The estimated λ_max(D⁻¹A_P) × 1.1 is ≥ the `eigsh` value and ≤ 1.5× it; the Chebyshev smoother is A_P-symmetric to 1e-12.
  - Iteration count is defined as PCG iterations on A_P, at fixed σ, to a relative residual of 1e-8. A study vs h is written to `benchmarks/results/`.

### PR 11: Geometric multigrid for voxel meshes
- Contents:
  - Trilinear transfer operators, with semi-coarsening (an axis stops coarsening at ≤ 2 elements). An axis with an odd element count gets one coarse cell spanning a single fine element; transfers use the coarse nodal positions.
  - `solve_modes` dispatches voxel meshes to this operator and GMG, including the σ pre-estimate.
  - Galerkin coarse operators of A_P on levels ≥ 1, built element-wise as Pᵀ(Σ_e m_e⊙(s_K,e·Ke + σ·s_M,e·Me)⊙m_e)P on the nested voxel hierarchy (Me consistent or lumped, matching the fine operator), plus σ·Pᵀ(m⊙m_nodal⊙m)P for nodal masses and Pᵀ s·(I − diag m)P for the constrained DOFs. Coarse DOFs whose support holds no active fine DOF (occupancy-masked meshes) are masked the same way, with diagonal s, so the coarsest Cholesky stays SPD. The fine operator is never assembled. Stored in block-ELL.
  - Symmetric Chebyshev smoothing.
  - Dense Cholesky on the coarsest level.
  - fp32 option for the V-cycle (levels, smoother workspace and transfers in fp32, applied symmetrically; the coarsest Cholesky stays fp64), modelled by the memory estimator.
  - Memory estimator.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Iterations grow ≤ 20% over 3 refinements.
  - `solve_modes` on a voxel mesh with an odd element count along one axis matches the PR 8 `eigsh` (PR 9 tolerances).
  - On a small occupancy-masked, clamped voxel mesh with nodal masses and an odd element count along one axis, each coarse level equals Pᵀ A_P P from the PR 4 SciPy matrices (masked coarse DOFs with diagonal s) to 1e-12.
  - LOBPCG with the fp32 GMG reaches the same tolerance as with fp64 GMG at a reference size, with ≤ 20% more iterations.
  - A checked-in synthetic SIMP-like density field (ρ ∈ [1e-3, 1], p = 3) converges to 1e-8 within 100 PCG iterations.
  - A single-layer mesh (n_z = 1) converges to 1e-8 within 100 PCG iterations.
  - A free-free voxel body (σ > 0) converges to 1e-8 within 100 PCG iterations; the coarsest-level Cholesky succeeds.
  - Mac acceptance: a 2M-DOF voxel modal solve, 10 modes, wall time ≤ 10 min, peak within 20% of the PR 6 + PR 9 + PR 11 estimators, with fp64 and with fp32 V-cycles.
  - Deferred GPU checks: a 100M matvec plus the LOBPCG and preconditioner workspace fits in the usable pool (PR 9 and PR 11 estimators); a 100M voxel modal solve.

### PR 12: p-multigrid for unstructured meshes
- Contents:
  - hex20/hex27 → hex8 transfer operators, with symmetric Chebyshev smoothing on the high-order level.
  - The hex8 coarse operator is the Galerkin product Pᵀ A_P P (also the input to PR 13 AMG). It is solved exactly by SciPy `splu` via `pure_callback`. This is test-only and is replaced by AMG in PR 13.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Iterations on hex20 grow ≤ 20% over 3 refinements.
  - Prolongation (hex20→hex8 and hex27→hex8) reproduces a trilinear field exactly (1e-12); Pᵀ A_P P matches a SciPy triple product to 1e-12.

### PR 13: AMG on the hex8 level
- Contents:
  - pyamg smoothed-aggregation setup on the assembled hex8 Galerkin A_P from PR 12, with the masked rigid modes as the near-null-space.
  - Hierarchy exported to block-ELL, with a V-cycle on the device. Zero tentative-prolongator columns (aggregates of fully constrained DOFs) are dropped, so every coarse matrix stays SPD and the coarsest dense Cholesky succeeds.
  - Combined with PR 12; pure hex8 meshes skip p-MG and run AMG on the assembled A_P (solid-shell meshes do the same from PR 18). `solve_modes` dispatches unstructured meshes to this preconditioner.
  - Memory ordering: setup → free the assembled A → allocate LOBPCG.
  - Option to reuse aggregates and recompute only the values (test: on a perturbed-density model, PCG iterations within 20% of a fresh setup).
  - Memory estimator.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Setup time and host-memory benchmarks (report-only).
  - A clamped hex20 model where aggregates straddle the Dirichlet boundary sets up and the coarsest Cholesky succeeds.
  - PCG iterations on hex20 (p-MG + AMG) grow ≤ 20% over 3 refinements.
  - `solve_modes` on a hex20 and a pure hex8 mesh matches the PR 8 `eigsh` (PR 9 tolerances).
  - Mac acceptance: a 1M-DOF hex20 modal solve, 20 modes, wall time ≤ 15 min, peak within 20% of the PR 5 + PR 9 + PR 13 estimators (PR 13 covers the p-MG workspace).
  - Optional Mac benchmark: 10M DOFs.
  - Deferred GPU check: 10M DOFs on one GPU.

## M3b: Frequency bands

### PR 14: Eigenvalue counting and [0, f_max]
- Contents:
  - Deflated stochastic count with M-weighted probes and a Chebyshev–Jackson step polynomial (50 probes, fixed seed list).
  - Element-wise Me Cholesky factors for probe sampling.
  - Probe-block memory model (recurrence, PCG and preconditioner workspace and projector temporaries (chunk + block) per probe), used to set the probe block size.
  - `verify=None/True/False`, with the predicted counting cost in `info`.
  - The cost function and LOBPCG-from-0 selection; PR 15 adds the shift-invert branch and slicing.
  - [0, f_max] mode reusing PR 9 chunking, stopping at λ > λ_c.
  - `solve_modes(..., band=(0, f_max))`.
- Docs: `docs/eigensolvers.md` (estimator, variance, degree and cost).
- Tests:
  - With one mode deliberately left out of V (not the one next to the cutoff), the count detects it for ≥ 97 of a fixed list of 100 seeds. The test is deterministic.
  - With V complete, the estimate (including Jackson bias) is < 0.5.
  - A κ ≥ 1e7 case, sized so the count is verifiable (asserted): with V complete, the estimate is < 0.5.
  - A case with τ_c forced below the measured PCG floor sets the unverifiable flag; the measured error propagation vs d is recorded.
  - `verify=True/False` force the count on/off; `verify=None` follows the predicted cost.
  - Element-wise Me Cholesky: LLᵀ = Me to 1e-12.
  - The probe-block memory model is within 20% of the measured peak at a size where the estimate is ≥ 2 GB.
  - Counting cost (polynomial degree, wall time) vs mode index is reported.
  - No modes are missed compared with the dense inertia at reference sizes.
  - (0, f_max) with f_max below the first elastic frequency returns 0 modes and a count < 0.5; λ_c set inside a double eigenvalue returns both copies.
  - The probe covariance Σ_e L_e L_eᵀ assembled (plus nodal masses, masked) equals the masked SciPy M to 1e-12.

### PR 15: Interior bands
- Contents:
  - Lanczos basis capacity per slice from the PR 15 memory model; a slice that fills it before the stop rule holds is split at its largest gap (test: a forced small capacity splits the slice and still matches `eigsh`).
  - Block shift-invert Lanczos (b = 6) with MINRES inner solves, also used for [0, f_max] when the LOBPCG-from-0 basis does not fit.
  - Slicing with the cost model.
  - Removal of duplicate modes at slice edges.
  - Band-pass count deflated by the modes found in the band and the converged neighbours just outside each band edge. A slice count ≥ 0.5 sets an `info` flag and restarts that slice's Lanczos run with a fresh random block, deflated by the modes already found; after 2 failed restarts the slice is returned with the count flagged.
  - Test tolerances: outer Lanczos residual r_acc = max(1e-10, 10·τ_eff·δ_max) with inner MINRES τ = 1e-8; cluster, slice-edge and band-edge gaps use 100·r_acc.
  - `band=(f1, f2)`.
- Docs: `docs/eigensolvers.md` (memory model for many modes, cost-model calibration, measured MINRES floor).
- Tests:
  - Matches SciPy `eigsh(sigma=...)` to relative error ≤ max(1e-8, 10·ε·λ̂_max/λ) for clusters with relative gap ≥ 1e-2 (others within 10·r_acc, multiplicities equal).
  - An interior band strictly between two well-separated modes returns 0 modes with count < 0.5; with a mode planted in that band and left out of V, the count is ≥ 0.5.
  - A clamped model with band (0, f_max) forced onto shift-invert stops and matches `eigsh` (the λ = 0 edge needs no neighbour).
  - A band with f₁ below λ₁ is solved as [0, f₂], filtered, and matches `eigsh`.
  - A slice with one in-band mode removed after convergence gives a count ≥ 0.5, sets the `info` flag and recovers the mode on restart; a slice whose count stays ≥ 0.5 (mode forcibly withheld) stops after 2 restarts with the count flagged.
  - A forced edge failing the final r_acc re-check is widened or merged, and the result still matches `eigsh`; τ_eff forced to 1e-6 near a dense cluster sets the slice-collapse flag.
  - Each slice's count estimate is < 0.5.
  - With one in-band mode (not next to a band edge) left out of V, the band-pass count detects it for ≥ 97 of a fixed list of 100 seeds. With V complete, with τ_c = 1e-12, switching off neighbour deflation raises the estimate (same probes y_j) by the probe mean of Σ_i q(μ_i)²(v_iᵀM y_j)² over the converged out-of-band neighbours v_i, to 1e-8 relative; with it, the estimate is < 0.5.
  - On a model with known repeated interior eigenvalues (voxel cube double/triple modes), each cluster is found with its exact multiplicity.
  - With a memory budget too small for the LOBPCG-from-0 basis, the cost model switches to shift-invert.
  - On the voxel cube, a band forced across ≥ 2 slices with a cluster near a slice edge, at the default tol = 1e-6: no duplicates, same count and multiplicities as `eigsh`, and λ within max(1e-8, 10·ε·λ̂_max/λ) for clusters with relative gap ≥ 1e-2, otherwise within 10·tol.
  - A band edge set exactly on a double eigenvalue returns both copies, and d stays ≈ 25/g.
  - MINRES iterations vs δ are measured and the cost-model constants fitted; the fp64 V-cycle MINRES true-residual floor is measured; both are recorded in `docs/eigensolvers.md`.
  - A case with τ forced below the measured floor raises τ_eff to the floor, carries it into r_acc and sets the `info` flag.
  - r_acc and the measured floor at the largest Mac size are recorded in `docs/eigensolvers.md`.
  - The memory model (deflation basis plus Lanczos and MINRES workspace) is within 20% of the measured peak at a size where the estimate is ≥ 2 GB.

## M4: Solid-shell element

### PR 16: Solid-shell element (KGW 1999)
- Contents:
  - Element blocks are marked as solid-shell by a per-block element type (same 8-node topology as hex8).
  - Thickness direction from a user-specified stack direction or a topological sweep; an error on ambiguity (test: a stacked plate is detected, an ambiguous cube block raises).
  - Until PR 18, `solve_modes` on solid-shell meshes uses Jacobi and is limited to the small M4 example; PR 16 accuracy tests use the PR 8 `eigsh`.
  - ANS and 5-parameter EAS as under Key design decisions.
  - Static condensation, with condensed Ke stored (solid-shell blocks support stored Ke only; on-the-fly mode raises an error, tested). This adds to the PR 5 estimator (test: stored condensed-Ke matvec on a solid-shell mesh sized so the estimate is ≥ 2 GB, estimate within 20% of the measured peak).
  - Laminates with ply angles in the local shell frame.
  - Stress recovery function (ANS strains, recovered EAS parameters) in the element registry; `mass="lumped"` on solid-shell raises an error (tested).
  - Mass from the displacement interpolation. Consistent mass only; lumping is not supported for solid-shell elements.
  - Simply supported BCs on solids as defined in PR 8.
- Docs: `docs/elements.md`, `docs/operators.md` (condensed Ke), `docs/verification.md` (reference levels, elements per ply).
- Tests:
  - Stored condensed-Ke `apply_K`, `apply_M`, `diag_K` and `diag_M` match the PR 4 SciPy K and M to 1e-12 relative, on a solid-shell mesh and on a mesh mixing hex8 and solid-shell blocks, for m = 1 and m > 1, with random s_K/s_M.
  - Jacobi `solve_modes` on the M4 example matches the PR 8 `eigsh` to the PR 9 tolerances.
  - Postprocessor stress recovery (PR 17) on a solid-shell mesh under imposed linear displacement equals Cε to 1e-10 (owned by whichever of PR 16/17 merges last).
  - Flat patch test with lifting: constant stress to 1e-10 relative to ‖σ‖, also through the solid-shell recovery function (ANS strains, recovered EAS parameters).
  - Exactly 6 zero eigenvalues after condensation, on an element with h/t ≤ 2: six ≤ 1e-10·λ_max, the 7th > 1e-8·λ_max.
  - MacNeal–Harder (against the MacNeal–Harder 1985 reference values):
    - Normalized tip deflection ≥ 0.95 on the regular mesh for extension, in-plane shear and out-of-plane shear.
    - On the parallelogram mesh: ≥ 0.9 for extension and out-of-plane shear.
    - Parallelogram in-plane shear, and twist on both meshes, are report-only.
    - Trapezoidal in-plane bending is report-only, because all 4-node membrane elements that pass the patch test lock there (MacNeal 1987).
  - Discretization-error tests use the PR 8 `eigsh` shift-invert solve at reference sizes (Jacobi LOBPCG is impractical at κ ≈ 1e12).
  - **0.1% target (measured against 3D elasticity):**
    - Srinivas–Rao at L/t = 10 and 100 with ≥ k stacked elements per ply (k stated in the docs).
    - One layered element per laminate is reported, not gated.
    - A clamped cylinder (R/t and L/R stated) against a hex27 reference whose error, estimated from ≥ 3 nested levels (Richardson, estimate only), is ≤ 0.02% (1/5 of the gate), within the PR 8 `eigsh` limit; levels and estimate recorded in `docs/verification.md`.
  - No locking (simply supported square plate, first 6 bending modes, picked by z-reflection antisymmetry, same in-plane mesh and layer count at both L/t): at L/t = 10 and 1000, the discretization error is measured against a converged solid-shell run at the same L/t. The mesh keeps that error ≫ 1e-4, since κ ≈ 1e12 at L/t = 1000 limits eigenvalue accuracy to about 1e-4. The error at L/t = 1000 is ≤ 1.5× the error at L/t = 10.
  - Distorted meshes (parallelogram only): a skew plate meshed with parallelograms at a stated skew angle and in-plane density; clamped edges, L/t stated; frequencies within 1% of a hex27 reference of the same skew plate whose estimated error (≥ 3 nested levels) is ≤ 0.2%.
  - `jax.grad` matches finite differences.

## M5: Postprocessing and performance

### PR 17: Postprocessor
- Contents:
  - Mode normalization.
  - MAC, M-weighted: (φ_iᵀMψ_j)²/((φ_iᵀMφ_i)(ψ_jᵀMψ_j)); also used for the PR 8 pairing.
  - Participation factors and effective modal mass (with r masked and rigid modes included in the mode set for the sum rule).
  - Strain and stress recovery per mode via each element type's recovery function in the registry (hex here; the solid-shell one, with ANS strains and recovered EAS parameters, ships with PR 16 and is picked up when both are merged).
  - VTU/XDMF export and a frequency table (effective masses summed per cluster, which is basis-invariant).
- Docs: `docs/postprocessing.md`.
- Tests:
  - On a small assembled model with all modes computed densely, Σ effective mass = rᵀMr over the free DOFs to 1e-10 relative.
  - meshio round-trip of the export (XDMF written with XML data, so no h5py).
  - Stress recovery under an imposed linear displacement equals Cε at every quadrature point to 1e-10; on a solid-shell mesh too once PR 16 is merged (the PR 16 test, owned by whichever merges last).
  - Rotating the basis of a double mode leaves the per-cluster effective mass unchanged to 1e-12.
  - Normalized modes satisfy VᵀMV = I to 1e-12; the M-weighted MAC of a set with itself is the identity to 1e-12 and matches a NumPy MAC.

### PR 18: Performance pass
- Contents:
  - Morton reordering and reordering tuning.
  - Kernel fusion checks (HLO inspection, report-only, recorded in `docs/performance.md`).
  - fp32 option for p-MG and AMG (as in PR 11), modelled by the PR 13 estimator, and its validation.
  - Compile-time benchmark: full `solve_modes` compile time with GMG/AMG vs size, recorded in `docs/performance.md`.
  - AMG on the assembled A_P for solid-shell meshes, and iteration counts vs L/t.
- Docs: `docs/performance.md`.
- Tests:
  - Roofline = min(bandwidth × arithmetic intensity, measured fp64 peak). The voxel stencil and stored-Ke matvecs are measured against jitted STREAM bandwidth. On the Mac (serial XLA:CPU scatter) the percentage is reported only; ≥ 50% at m = 1 is a deferred GPU check. For m > 1, the percentage is reported.
  - With the fp32 preconditioner, the same convergence tolerance is reached.
  - The PR 13 1M hex20 Mac acceptance run with fp32 p-MG/AMG has peak within 20% of the PR 13 estimator.
  - After Morton reordering, apply_K and apply_M equal the permuted original to 1e-12.
  - Solid-shell AMG iterations grow ≤ 2× from L/t = 10 to L/t = 1000, or a fallback is documented.

### PR 19: Partitioning and multi-GPU
- Status: an epic, to be split before starting.
- Contents:
  - METIS partitions padded to uniform shard sizes.
  - `shard_map` operator with `ppermute` halos.
  - `psum` Gram matrices.
  - Distributed GMG first.
  - Developed on the Mac with forced host devices, in a separate pytest process.
- Docs: `docs/parallel.md`.
- Tests:
  - Results match single-device runs.
  - Deferred GPU check: weak scaling.

## M6: Optimization

Working assumptions, still open: topology optimization at ≤ 10M voxel elements, clamped, densities as the only design variables. The user will decide later. Principle 5 keeps the alternatives open.

### PR 20: MMA
- Contents: MMA in JAX.
- Docs: `docs/optimization.md`.
- Tests:
  - Matches reference iterates on Svanberg's toy problem to 1e-8, which requires a faithful `subsolv` (including epsimin and the epsi schedule), written from Svanberg's published notes (2007), not translated from GPL code. The iterates are generated once offline with a reference MMA port and committed as data, so there is no GPL dependency.
  - Solves Svanberg's test problems.

### PR 21: Eigenvalue sensitivities
- Contents:
  - Element-kernel sensitivity pass.
  - Cluster detection and the cluster sensitivity matrix.
  - Bound formulation (max β s.t. λ_j ≥ β, Du & Olhoff 2007), with per-mode bounds up to a gap cutoff J, extended to the end of the cluster containing mode J plus 2 guard modes (failing only when the guards run out). The constraint vector has fixed capacity J_max² with J_max = J + guards, padded with inactive constraints (−1 ≤ 0, zero gradient), so cluster changes do not recompile.
  - KS aggregation over complete clusters only: n is extended to the end of the cluster at the cutoff, with 2 guard modes beyond; the run fails only when the guard modes run out.
- Docs: `docs/optimization.md`.
- Tests:
  - At ~5k DOFs (dense `eigh` still feasible), the gradient from LOBPCG modes (r ≤ 1e-8) matches the dense-`eigh` gradient within 10·r_M·λ/gap·‖g‖_∞ (r_M as in PR 9), for modes with relative gap ≥ 1e-2.
  - KS aggregate and bound-constraint functions: central FD against dense `eigh` (relative gap ≥ 1e-2, error ≤ 1e-6·‖g‖_∞).
  - Cluster detection groups the double eigenvalue of the double-eigenvalue model below; the KS cutoff extends over it. With J set so the cutoff lands inside that double eigenvalue, the bound constraint set is extended over both copies. At a near cluster (relative gap ≈ 1e-3), F is symmetric and matches central FD of the reduced matrix B^{-1/2}AB^{-1/2} (A = V_cᵀKV_c, B = V_cᵀMV_c) to 1e-6·‖F‖. Running out of guard modes raises an error; the bound-formulation function raises when its β input reaches the upper bound of 100. V_c is held fixed at the base design's eigenvectors in the near-cluster FD.
  - Changing the cluster structure between calls (padded constraints) does not increase the compile count.
  - Central finite differences with a step-size sweep against dense `eigh` on coarse meshes (κ ≲ 1e4, at most a few hundred elements), for coordinate, density, material and nodal-mass parameters. Error ≤ 1e-6·‖g‖_∞, with each parameter type tested separately. The test eigenvalues have relative gap ≥ 1e-2; the test checks per-mode gradients and bypasses cluster detection. Density parameters enter through linear scaling (s_K = s_M = ρ); the PR 23 interpolations are tested there.
  - Double eigenvalue (a mode of multiplicity exactly 2, checked from dense `eigh` gaps, e.g. a square plate or a cantilevered cube): one-sided directional differences along a random ±1 direction d (a uniform d gives Σ_e d_e f^e = 0 when s_K = s_M = ρ) match the eigenvalues of Σ_e d_e f^e. Both pairs are sorted before comparing, with ‖d‖_∞ = 1 and error ≤ 1e-4·‖Σ_e d_e f^e‖₂ and an O(h) step sweep (the test passes at the best h of the sweep).

### PR 22: FFD shape parametrization
- Contents:
  - Bernstein lattice → node coordinates, in JAX.
  - Nodes outside the lattice, the two outer control-point layers (C¹ continuity at the lattice boundary; so ≥ 5 control points per direction) and BC nodes are frozen. BC regions must lie on or outside the lattice boundary; this is checked (test: frozen nodes and the two outer control-point layers have zero gradient and do not move).
  - Constraint: a KS/p-norm aggregate of detJ/detJ₀ ≥ ε, with ε = 0.1 and p = 50 (recorded in `docs/optimization.md`). Both KS aggregates use a shifted logsumexp.
  - Example (unimodal fundamental mode with a large initial gap, e.g. first bending of a cantilever plate with L/t ≤ 20, checked by the PR 21 gap): maximize the fundamental frequency subject to mass ≤ M₀, using MMA with the PR 21 bound formulation (J ≥ 2), at small size with a Jacobi preconditioner.
- Docs: `docs/optimization.md`.
- Tests:
  - Control-point gradients of the objective and of the detJ aggregate pass the PR 21 finite-difference criterion.
  - A BC region inside the lattice raises an error.
  - The example converges: relative objective change < 1e-4 over 5 iterations, λ₁ increased, no inverted elements.
  - Rescaling the example from m to mm (E and ρ converted) leaves the normalized iterates unchanged to 1e-8 relative; a power-of-two rescaling (length × 2¹⁰, E and ρ scaled by powers of two) leaves them unchanged to 1e-12.

### PR 23: Topology optimization
- Contents:
  - Densities:
    - Design ρ ∈ [0, 1].
    - Filtered ρ̃.
    - Physical ρ̄ = ρ_min + (1 − ρ_min)·H_η(ρ̃), with ρ_min = 1e-3. Without projection, H_η is the identity (η is the projection sharpness, with threshold 0.5; β is the bound variable).
  - SIMP: E = ρ̄^p E₀ with p = 3.
  - Mass interpolation m(ρ̄) = ρ̄ for ρ̄ > 0.1, else c₁ρ̄⁶ + c₂ρ̄⁷ (c₁ = 6e5, c₂ = −5e6; C¹ at 0.1).
  - Volume constraint Σ ρ̄_e v_e ≤ V* (ρ̄ includes the ρ_min floor; V* is matched to the paper accordingly).
  - Non-structural mass via nodal masses (in the single-layer benchmark split equally between top and bottom nodes).
  - Filters:
    - Voxel meshes: radial cone density filter by direct convolution, normalized at boundaries.
    - The benchmark also uses a density filter, even if the paper used a sensitivity filter, so that gradients stay consistent. The 5% tolerance on ω₁ absorbs the difference.
    - Unstructured meshes: a Helmholtz PDE filter with PCG. Its VJP is Tᵀ K_f⁻¹ T V⁻¹ c (T the element–node transfer, V the element volumes), still one symmetric `custom_linear_solve` with K_f, so the solver is not differentiated. The PR 23 finite-difference test runs it to a 1e-12 tolerance.
  - Optional Heaviside projection with η-continuation: H_η(x) = (tanh(η/2) + tanh(η(x − 1/2)))/(2·tanh(η/2)), so H_η(0) = 0 and H_η(1) = 1.
  - Voxel sensitivity pass (reference Ke and Me scaled per element by s_K, s_M) feeding the cone-filter VJP.
  - Bound formulation with MMA.
- First step of the PR: choose the Du & Olhoff 2007 example (and its volume fraction) and record it in `docs/verification.md`. The benchmark's pass criteria below are then fixed to that example.
- Docs: `docs/optimization.md`; `docs/verification.md` gets the full Du & Olhoff setup (table and figure numbers).
- Tests:
  - Mass interpolation unit test: value and slope continuous at 0.1, positive on (ρ_min, 0.1), autodiff matches FD inside (ρ_min, 0.1).
  - End-to-end finite differences, run on (a) a voxel mesh with the cone filter through the voxel sensitivity pass and (b) an unstructured mesh with the Helmholtz filter, through filter → projection → interpolation → KS/bound, to the PR 21 criterion (central FD with step sweep, ≤ 1e-6·‖g‖_∞), with test densities in ρ̄ ∈ [0.2, 1] (so κ stays small) and away from ρ̄ = 0.1 ± h, where the mass interpolation is only C¹, and relative gap ≥ 1e-2 for per-mode bound terms.
  - Re-run of the PR 11 SIMP-contrast test on a PR 23 design.
  - Both filters preserve a constant density field (1e-12, Helmholtz PCG at 1e-12 tolerance); the Helmholtz filter VJP matches the dense transpose Fᵀc to 1e-10.
  - MMA with the PR 21 cluster constraints on a forced double eigenvalue (the bimodal example chosen in the PR's first step, or the Du & Olhoff clamped–clamped beam), without projection (so no η-continuation): for k ≥ k₀ = 10 (fixed in the test), β_k ≥ max_{k₀≤j<k} β_j − 1e-2·|β_k| (MMA is not monotone; the tolerance absorbs asymptote adaptation), and over the last 5 iterates the 2-D cluster subspace is stable (cos² of both principal angles between consecutive iterates ≥ 0.9, in the M of the later iterate) and no third mode joins the cluster (λ₃/λ₂ − 1 > 1e-2). Individual modes inside the cluster are not tracked, since any basis of it is valid.
  - Benchmark: the chosen Du & Olhoff 2007 example, with the paper's filter radius, and projection only if the paper uses it. The setup:
    - one element layer of thickness t ≤ h;
    - u_z = 0 on all nodes;
    - plane-stress-equivalent material E′ = E(1 + 2ν)/(1 + ν)², ν′ = ν/(1 + ν).
  - Pass criteria for the benchmark:
    - ω₁ within 5% of the published value.
    - If the paper's example is bimodal: relative gap λ₂/λ₁ − 1 ≤ 1%.
    - The lowest eigenvalue restricted to the z-antisymmetric subspace (LOBPCG on the operator wrapped by P = [I; −I]/√2 over top/bottom node pairs) lies above the highest reported mode or cluster.
    - No localized modes: for each reported mode or cluster (relative gap < 1e-2), with an M-orthonormal basis V_c, the largest eigenvalue of V_cᵀ M_void V_c is < 5%, where M_void is M restricted to elements with ρ̄ < 0.1. This is the worst kinetic-energy fraction over the cluster subspace, so it does not depend on the basis returned.

## GPU validation

### PR 24: GPU validation (whenever a CUDA machine is available)
- Runs every item in the deferred list below.
- Records results in `benchmarks/results/`.
- Adds Pallas kernels if PR 18's roofline criterion calls for them.

**Deferred GPU checks:**
- PR 11: 100M matvec + LOBPCG and preconditioner workspace fits in the usable pool (≈ 76 GB).
- PR 11: 100M voxel modal solve.
- PR 13: 10M hex20 modal solve on one GPU.
- PR 5: scatter vs gather benchmark on GPU; sets the GPU default.
- PR 18: matvec ≥ 50% of roofline at m = 1.
- PR 19: weak scaling.

## Dependencies

| PR | Depends on |
|---|---|
| 0 Scaffold | — |
| 1 Mesh | 0 |
| 2 Materials | 0 |
| 3 Elements | 1, 2 |
| 4 DOF map, assembled, masking | 3 |
| 5 Matrix-free unstructured | 4 |
| 6 Voxel stencil | 4, 5 |
| 7 Rigid modes, deflation | 4 |
| 8 Reference eigensolver | 4, 7 |
| 9 LOBPCG, API | 5, 7, 8 |
| 10 Linalg kernels, smoothers | 5 |
| 11 Voxel GMG | 6, 9, 10 |
| 12 p-MG | 5, 9, 10 |
| 13 AMG | 12 |
| 14 Counting, [0, f_max] | 9, 10, 11 (preconditioned K + σM solves on voxel meshes; 13 optional; without PR 13, unstructured counting uses Jacobi PCG) |
| 15 Interior bands | 14; 13 for unstructured meshes |
| 16 Solid-shell | 3, 5, 8, 9 |
| 17 Postprocessor | 9 |
| 18 Performance | 11, 13, 16 |
| 19 Multi-GPU | 18 |
| 20 MMA | 0 |
| 21 Sensitivities | 9 |
| 22 FFD | 20, 21 |
| 23 Topology opt | 11, 20, 21 |
| 24 GPU validation | CUDA hardware; the PRs in the deferred list |

- **Critical path:** 0 → 1 → 3 → 4 → 7 → 8 → 9, with PR 2 in parallel with 1 and PR 5 in parallel with 7–8. This gives first modes in pure JAX.
- **Biggest risk:** M3 (preconditioning).
- **Parallel work:** PRs 2, 6 and 20 can proceed alongside the critical path. PRs 16 and 17 can proceed alongside M3.
