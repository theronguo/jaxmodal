# Roadmap

Large-scale 3D linear modal analysis (Kv = λMv) in JAX. The solver uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. FFD shape optimization and topology optimization come later.

**Targets:**
- Lowest modes and modes in frequency bands.
- Frequency accuracy of 0.1% against a converged reference, for hex20/hex27 and solid-shell meshes.
- 10M DOFs, unstructured, on a single GPU.
- 100M DOFs on voxel meshes with a small block size (see the memory budget).
- Multi-GPU later.
- An FP64-capable GPU (A100/H100 class) for production runs.
- Until a GPU is available, development and all tests run on an Apple M3 Max (64 GB, CPU backend, fp64).

Package name: `jaxmodal`. License: MIT.

Status: planning. Review rounds 1–6 incorporated. Implementation starts after a review round with all 5 reviewers satisfied.

## Principles

1. **Validate every fast path against a slow one.** References are SciPy, dense NumPy, or analytic or published values. Every acceptance criterion is quantitative.
2. **One small interface.** Every discretization (unstructured, voxel) provides `apply_K`, `apply_M`, `diag_K`, `diag_M` and the rigid body modes. Each operator applies natively to blocks of shape `(n_dof, m)`, not via `vmap`. Solvers only see this interface plus `apply_P`.
3. **Fixed shapes; everything else is a `jit` argument.**
   - Topology preprocessing (connectivity, DOF maps, sets) runs in NumPy.
   - Every array is a pytree `jit` argument: int32 topology, geometry, densities, material parameters, nodal masses, stored Ke, multigrid/AMG matrices and rigid modes. Nothing is closed over.
   - Only shapes, chunk sizes, element types, block sizes, Chebyshev smoother degrees and multigrid level counts are static. The counting-polynomial degree is a dynamic `fori_loop` bound; its Jackson/Chebyshev coefficients are computed in-loop from (k, d).
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
   - Elements (and voxel z-slabs) are processed in fixed-size chunks with `lax.map`/`scan`, the last chunk padded, adding into a donated output.
   - Indices are int32, with asserts that N_dof and every flattened index product (e.g. n_el × n_dof_e, N_dof × m) stay below 2³¹. The asserts run in NumPy before casting.
   - Estimators include chunk temporaries and are checked against measured peaks.
7. **Conventions** live in `docs/conventions.md` and are fixed in PR 0:
   - VTK node ordering internally, converted at import.
   - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear strains.
   - fp64 by default, enabled through an explicit config call (no import side effects).
   - Index arrays are always int32.
   - Iteration vectors, returned modes, Gram matrices and Rayleigh–Ritz are always fp64. fp32 is allowed only inside preconditioners and for the device-side deflation basis. Returned locked modes come from an fp64 host copy.
   - m always denotes the LOBPCG block size (nev plus guard vectors).

## Key design decisions

- **Integration:** full integration only (hex8 2³, hex20/hex27 3³) for K and consistent M. No reduced integration, because its zero-energy modes would pollute the spectrum.
- **Mass:**
  - Consistent mass is the default everywhere. Lumping is opt-in via `mass="lumped"`: row-sum for hex8/hex27, HRZ for hex20. Row-sum lumping gives non-positive corner masses for hex20.
  - An optional nodal (point) mass vector is added to M.
- **Hex8:** plain, fully integrated hex8 is accepted. It shear-locks in bending, so bending accuracy comes from refinement or from hex20/hex27.
- **Dirichlet BCs:** zero displacement per component in global axes on node sets. There are no skew constraints.
  - Eigen operators are purely masked: `apply_K(v) = m ⊙ K(m ⊙ v)`, the same for M, with all iterates masked.
  - Every preconditioner, linear solve and inertia count instead uses A_P = m A m + (I − diag m), which puts a unit diagonal on the constrained DOFs. Preconditioner output is masked.
- **Rigid body modes:**
  - Computed analytically per connected part.
  - Under Dirichlet sets, the remaining modes are R·null((I − m)R), computed via an SVD, then M-orthonormalized and deflated with P = I − R Rᵀ M.
  - The AMG near-null-space is m ⊙ R: all 6 modes per part with constrained rows zeroed. It is not the reduced deflation basis, which can be empty for clamped models.
- **Preconditioning:**
  - Preconditioners approximate A = K + σM with σ > 0 (default ≈ estimated λ_target), used as A_P.
  - They must be fixed, linear and SPD: symmetric pre- and post-smoothing, no inner Krylov.
  - The Chebyshev λ_max estimate uses a safety factor of 1.1.
- **Eigensolver:** generalized LOBPCG following Hetmaniuk–Lehoucq / Duersch et al. 2018:
  - SVQB orthonormalization with a drop tolerance; dropped directions are masked.
  - In Rayleigh–Ritz, masked rows and columns are zeroed off the diagonal, with B = 1 and A = 10·λ̂_max on the diagonal. They never appear as spurious low Ritz values.
  - P is dropped (masked) when ill-conditioned.
  - Implicit product updates with periodic explicit recompute.
  - The deflation projector is re-applied every iteration.
  - The locked buffer is preallocated, with a count. A Python outer loop calls at most two compiled step variants (with and without P).
  - A low-memory 6-block mode recomputes KX and MX instead of storing them.
- **Convergence:**
  - Measured by the λ-scaled residual r = ‖Kx − λMx‖ / (|λ|‖Mx‖). This gives sin θ ≲ r·λ/gap and δλ/λ ≲ r²·λ/gap. The normwise backward error is only reported.
  - Default tol = 1e-6.
  - Accept r ≤ max(tol, c·ε·(‖K‖ + |λ|‖M‖)‖x‖ / (|λ|‖Mx‖)), with c = 10 and ‖K‖, ‖M‖ estimated by power iteration. This stays bounded for near-singular M. A stagnation exit is flagged in `info`.
  - A cluster (for locking) is a set of eigenvalues whose relative gaps are < 100·tol. The PR 9 test threshold of 1e-4 differs on purpose: it only decides whether to compare by subspace.
  - Block size m = nev + max(5, 0.2·nev).
  - Chunk and lock cutoffs must fall in a gap.
- **Precision:**
  - The attainable residual floor is about ε·κ, with κ = λ_max/λ_min: roughly 1e-8–1e-6 in fp64, fine for 0.1% targets but O(1) in fp32.
  - Hence fp32 never goes into X, R, P or the returned modes.
  - PR 9 measures the floor.
- **Frequency bands (PRs 14–15):**
  - **Strategy:** chosen by a cost model.
    - Bands whose upper mode index is ≲ 1200 are computed by chunked LOBPCG from 0, then filtered to the band.
    - The stochastic count is on by default for shift-invert bands. For LOBPCG-from-0 bands it is opt-in (`verify=True`, implemented in PR 14), because it can cost more than the solve itself.
    - Higher bands use shift-invert. Near mode index 1000 (relative gap ~7e-4), shift-invert costs ~10³ MINRES iterations per Lanczos step, and counting costs d ~ 10³–10⁴ solves per probe block.
  - **[0, f_max]:** chunked LOBPCG until λ > λ_max.
  - **Interior bands [f₁, f₂]:**
    - Shift-invert Lanczos on (K − σM)⁻¹M with σ in the band, full M-reorthogonalization, and a final Rayleigh–Ritz with true residuals on (K, M).
    - Inner MINRES with absolute-value preconditioning by the SPD multigrid of K + σM (Vecharynski–Knyazev 2013), with a fixed inner tolerance τ = 1e-8.
    - Slice widths come from the cost model (iterations ∝ 1 / relative distance to the slice edge). Duplicate modes at slice edges are removed by M-orthogonalization.
    - Neither the Lanczos nor the counting steps need M⁻¹. Consistent M costs nothing extra there.
  - **Completeness count (stochastic, at scale):**
    - Estimate s = ‖Π_V q(T) Π_V y‖²_M, with:
      - T = (K + σM)⁻¹M and σ ≈ λ_cutoff;
      - Π_V = I − VVᵀM, where V includes the computed modes and R.
    - Probes: y = M⁻¹g with g ~ N(0, M). g comes from element-wise Cholesky factors of Me, plus one Jacobi-PCG M-solve per probe.
    - q is a Chebyshev–Jackson step polynomial ([0, λ_max]) or band-pass polynomial (interior band, deflated by the modes found in the band).
    - The degree d (∝ λ/gap at the cutoff) is chosen numerically so that the Jackson-tail bias n_dof·max q² over the out-of-band spectrum is ≤ 0.05, and q² ≥ 0.95 over the whole in-band interval.
    - Inner PCG tolerance τ ≲ 0.01/(d·√n_dof).
    - One missing mode makes the estimate behave like χ²ₙ/n. With 50 probes, P(missed mode undetected) ≈ 0.2%.
    - A slice passes when its estimate is < 0.5.
  - **Exact count:** at reference sizes (≤ ~15k DOF), from the inertia of dense `scipy.linalg.ldl`.
  - **Assumed band position:** interior bands sit at mode indices up to ~1000 and contain ≤ 200 modes. This is a working assumption (the user doesn't know typical bands yet); PR 14 reports the counting cost as a function of mode index, so the assumption can be revisited. This bounds the polynomial degree, and so the cost of counting.
- **Preconditioners:**
  - Voxel meshes: geometric multigrid with semi-coarsening: an axis stops coarsening once it has ≤ 2 elements.
  - Unstructured meshes: p-multigrid (hex20/hex27 → hex8), then smoothed-aggregation AMG. AMG setup runs in pyamg on the host; the V-cycle runs on the device with a self-written block-ELL SpMV.
  - No `jax.experimental.sparse` in hot paths.
- **Shells:** solid-shell elements (3 DOFs per node) following Klinkel–Gruttmann–Wagner (1999):
  - ANS for transverse shear (sampled at edge midpoints).
  - ANS for the thickness strain at the 4 corner points.
  - EAS built with the center Jacobian J₀, by default with 5 parameters: 1 thickness (linear in ζ) and 4 Simo–Rifai membrane parameters. ∫M dV = 0 holds for the patch test.
  - Laminates with n Gauss points per ply.
- **Sensitivities:**
  - Formula: dλ/dp = vᵀ(∂K/∂p − λ∂M/∂p)v / vᵀMv.
  - Computed as a chunked `vmap(grad(element energy))` pass with v held fixed, in fp64.
  - Accuracy: r ≤ 1e-8 in the PR 21 tests. Optimization runs use r ≤ max(1e-6, measured floor), and the sensitivity error is O(r·λ/gap).
  - Clusters use the sensitivity matrix f_sk = v_sᵀ(K′ − λM′)v_k over an M-orthonormal cluster basis, built from the bilinear forms. The optimization cluster tolerance is a relative gap of 1e-2, as in Du & Olhoff.
  - In MMA, each cluster uses the Seyranian / Du & Olhoff conditions:
    - diagonal constraints β ≤ λ̄ + f_ss·Δρ;
    - off-diagonal constraints |f_sk·Δρ| ≤ ε_off (s < k), as paired inequalities, with ε_off = 1e-2·|β|. By Gershgorin, β can overshoot the true λ_min by at most (N − 1)·ε_off.
  - Per-mode diagonal constraints alone are invalid, because f_kk depends on the basis.
  - Deflation and masking leave the formula unchanged.
- **Optimizer:** MMA implemented in JAX (~200 LOC). No OC, because eigenvalue sensitivities change sign.
- **Scatter vs gather:** XLA:CPU scatter is effectively serial. PR 5 benchmarks scatter against an inverse-map gather on each backend, and the default may differ per backend.
  - The gather runs on an RCM-ordered mesh (RCM is built in PR 5), chunked by node range.
  - Element contributions go into a padded, fixed-size element window per range, which is counted in the estimator.
- **Dependencies:**
  - Python 3.12.
  - Core: `jax` (pinned minimum version), `numpy`, `scipy`, `meshio`, `pyamg`, all with macOS arm64 wheels.
  - `pymetis` is an optional extra (only PR 19 needs it).
  - Versions are locked with `uv.lock`.
  - No SLEPc, CHOLMOD or pypardiso.
  - meshio's `.inp` reader doesn't handle parts or assemblies; a small custom reader is added only if needed.

## Memory budget

All sizes are fp64.

| Item | 1M hex20 (Mac test) | 2M voxel (Mac test) | 10M hex20 | 100M voxel |
|---|---|---|---|---|
| One vector | 0.008 GB | 0.016 GB | 0.08 GB | 0.8 GB |
| Operator | Ke ≈ 1.2 GB (symmetric) | < 0.1 GB | Ke ≈ 11.7 GB (symmetric) or on-the-fly | < 1 GB |
| LOBPCG, 9 blocks × m | m = 25: ≈ 1.8 GB | m = 15: ≈ 2.2 GB | m = 30: ≈ 22 GB | m = 10: ≈ 72 GB |
| LOBPCG, 6 blocks | — | — | m = 30: ≈ 14 GB | m = 8: ≈ 38 GB |
| Rayleigh–Ritz update temporaries (≈ 2 blocks × m) | ≈ 0.4 GB | ≈ 0.5 GB | ≈ 5 GB | m = 8: ≈ 13 GB |
| Preconditioner workspace (≈ 4 blocks × m) | ≈ 0.8 GB | ≈ 1 GB | ≈ 10 GB | m = 8: ≈ 26 GB (fp32: 13 GB) |
| Multigrid | AMG ≈ 0.4 GB (device); pyamg peak 2–3 GB (host) | ≈ 0.2 GB | AMG ≈ 3–4 GB; pyamg peak ≈ 20+ GB (host) | ≈ 9.3 GB |
| Gather path (if chosen) | chunked by node range | — | chunked by node range | — |
| Locked modes / deflation basis | small | small | n × 0.08 GB (fp32: half) | n × 0.8 GB (fp32: half) |

- **Mac (64 GB, unified memory):**
  - Both test columns fit easily.
  - 10M is an optional, tight benchmark. Order the work as: AMG setup → free the assembled A → allocate LOBPCG.
  - An `np.memmap` is used only if the locked modes exceed RAM.
- **80 GB GPU:**
  - 10M fits.
  - 100M is tight: ≈ 70 GB with 6 blocks and m = 8 (nev = 3 per chunk): 38 for the blocks, 13 for Rayleigh–Ritz, 13 for the fp32 preconditioner workspace and 4.7 for the fp32 multigrid. The remaining ≈ 10 GB holds the fp32 deflation basis, so ≲ 20 modes in total at 100M.
  - Larger blocks at 100M need multi-GPU (PR 19).
- The PR 5/6/9/11/13 estimators reproduce this table and are validated against measured peaks where the estimate is ≥ 2 GB.

## Development environment

- **Development:** Apple M3 Max with 16 cores and 64 GB of unified memory, using the JAX CPU backend in fp64. The acceptance tests run at 1M (hex20) and 2M (voxel) DOFs. 10M is an optional benchmark.
- **No Mac GPU:** the GPU isn't usable for this project. Metal GPUs have no fp64 hardware, and the `jax-metal` plugin is experimental with incomplete op coverage.
- **Local checks:** `scripts/check.sh` runs lint plus the fast suite, `pytest -m "not slow and not gpu"`, which must finish in under 5 minutes on the M3. Remote CI comes later.
- **Merge gate:** `pytest -m "not gpu"`, slow tests included.
- **GPU checks:** GPU acceptance criteria go into the "Deferred GPU checks" list and are run in PR 24 whenever a CUDA machine becomes available. They don't gate other milestones.
- **Mac acceptance budgets:** the initial numbers below are fixed in PR 9, once matvec throughput has been measured.
- **Benchmarks:**
  - Each measurement runs in a fresh subprocess and reports peak RSS minus baseline RSS.
    - `memory_stats()` is unavailable on CPU.
    - `ru_maxrss` is in bytes on macOS.
    - For the 10M runs, the peak `phys_footprint` (`proc_pid_rusage`, field `ri_lifetime_max_phys_footprint`) is also recorded, because memory compression can make RSS undercount.
  - JSON results go to `benchmarks/results/` and record:
    - jax and jaxlib versions;
    - STREAM triad bandwidth, both jitted and NumPy/C multithreaded, with the method stated;
    - XLA thread count (the default Eigen pool unless pinned) and P/E-core setup;
    - measured fp64 peak;
    - time, peak memory and compile time.
- **Compilation cache:**
  - The persistent cache is enabled in tests, with `persistent_cache_min_compile_time_secs = 0`. PR 0 checks that it actually stores entries on CPU.
  - Compile-time tests (PRs 5, 9, 18) run in a fresh subprocess with the cache disabled and time `jit(f).lower(...).compile()`.
- **Multi-device tests:**
  - They live in `tests/multidevice/`, which is in `collect_ignore`. `jax_num_cpu_devices` is used if the pinned jax supports it.
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
  - Index arrays from `argsort`, `nonzero` and `arange` are cast back to int32.
  - The harness reports a stable RSS delta for a known allocation.
  - Buffer donation works on CPU: there is no "donated buffers were not usable" warning, and `x.is_deleted()` is true for the donated input.
  - The persistent cache stores entries on CPU.

### PR 1: Mesh data model and parser
- Contents:
  - `Mesh` dataclass: coordinates, element blocks keyed by type, node and element sets, material and orientation IDs.
  - meshio readers for Abaqus `.inp` (flat), Gmsh and VTU.
  - Per-format node-ordering maps for hex20/hex27.
  - Structured voxel mesh generator.
- Docs: `docs/mesh.md`.
- Tests:
  - Read → write → read round-trips.
  - Orderings match hand-built references for every format.
  - hex8 corner Jacobians are positive.

### PR 2: Materials
- Contents:
  - Traceable 6×6 C constructors: isotropic, transversely isotropic, orthotropic (ν_ij/E_i = ν_ji/E_j convention), full anisotropic.
  - Positive-definiteness check.
  - Stress and strain Bond matrices.
  - Density.
- Docs: `docs/materials.md`.
- Tests:
  - Bond rotation equals `einsum` tensor rotation.
  - Isotropic C is invariant under rotation.
  - C stays positive definite after rotation.
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
  - Exactly 6 zero eigenvalues of Ke, on distorted elements too.
  - Lumped masses all > 0, and total mass equals ρV per direction.
  - Matches a NumPy reference implementation.
  - `jax.grad` of uᵀKe(X)u matches finite differences (rel. err. < 1e-6).

### PR 4: DOF map, assembled reference operator, masking
- Contents:
  - Global DOF numbering (int32).
  - Per-group element-to-DOF arrays.
  - SciPy CSR K and M via COO.
  - Nodal mass vector.
  - Masking and A_P.
- Docs: `docs/operators.md`.
- Tests:
  - Distorted multi-element patch test, with linear displacement imposed by lifting inside the test:
    - Straight-edged distortion (midside nodes at edge midpoints): constant stress to 1e-10.
    - Curved hex20/hex27 (midside offset amplitude stated): constant stress to 1e-10 at every quadrature point. 3³ Gauss integrates cof(J)ᵀ∇̂N exactly here: for triquadratic geometry the integrand has degree ≤ 5 per direction.
  - Clamped model: no spurious eigenvalues compared with SciPy on the reduced matrices.
  - A_P is SPD on a clamped model, and Jacobi built on it has no zero divisions.

### PR 5: Matrix-free unstructured operator
- Contents:
  - Two modes: stored Ke, and on-the-fly computation from precomputed invJ·detJ per quadrature point.
  - Chunked, block-native apply.
  - Scatter (`segment_sum`) vs chunked inverse-map gather, benchmarked per backend.
  - `diag_K`, `diag_M`, nodal masses.
  - Memory estimator.
  - Node-local numbering.
  - RCM node reordering (`scipy.sparse.csgraph.reverse_cuthill_mckee`) in `mesh/`.
- Docs: `docs/operators.md`.
- Tests:
  - Both modes match PR 4 to 1e-12 relative.
  - The estimator is within 20% of the measured peak at 2M hex20 DOFs.
  - With the chunk size fixed and the cache disabled, compile time at 1M elements is ≤ 1.5× that at 10k elements.

### PR 6: Voxel stencil operator
- Contents:
  - K matvec over z-slabs: `scan` + `dynamic_slice` with a 1-layer halo; per slab, 8 shifted corner slices → einsum with the reference Ke → shifted adds into a donated output.
  - Separate per-element stiffness and mass scaling arrays, s_K and s_M.
  - `apply_M` (consistent and lumped), `diag_K`, `diag_M`, nodal masses.
  - Memory estimator, including slab temporaries.
  - Restriction: one material and orientation per phase.
- Docs: `docs/operators.md`.
- Tests:
  - Matches PR 5 on the same mesh.
  - `jax.grad` with respect to s_K and s_M matches finite differences.
  - Mac acceptance: for a matvec plus six preallocated (n_dof, 8) fp64 arrays at 2M DOFs, the estimate is within 20% of the measured peak.
  - Deferred GPU check (uses the PR 9 and PR 11 estimators): a 100M matvec plus the LOBPCG and preconditioner workspace fits in 80 GB.

### PR 7: Rigid body modes and deflation
- Contents:
  - Rigid modes per connected part.
  - Automatic remaining modes under Dirichlet sets.
  - M-orthonormalization.
  - Projector P = I − R Rᵀ M.
- Docs: `docs/boundary_conditions.md`.
- Tests:
  - ‖KR‖ ≤ 1e-12·‖K‖‖R‖.
  - ‖RᵀMR − I‖ ≤ 1e-12.
  - Free block: u_x = 0 on x = 0 leaves 3 modes; adding u_y = 0 on y = 0 leaves 1 (T_z); adding u_z = 0 on z = 0 leaves 0.
  - Two disconnected parts give 12 modes.
  - Projected random vectors are M-orthogonal to R.

## M2: First modes end-to-end

### PR 8: Reference eigensolver and verification suite
- Contents:
  - SciPy `eigsh` shift-invert (negative shift) on assembled matrices, with its largest supported size stated.
  - Dense inertia via `scipy.linalg.ldl` (≤ ~15k DOF).
  - In-test oracles:
    - A rectangular block with sliding BCs (exact separable solution, including orthotropic).
    - Visscher Ritz method for a free anisotropic parallelepiped: a Legendre basis, dense `eigh`, converged when successive degrees change λ by < 1e-6.
    - The Srinivas–Rao exact 3D solution for thick orthotropic and cross-ply simply supported plates.
  - Published benchmarks:
    - NAFEMS FV52 (simply supported thick plate). The docs state where u_z = 0 is applied; modes 1–3 are in-plane rigid modes, which exercises the automatic partial rigid modes.
    - NAFEMS FV12 (free thin plate), run with hex20/hex27 only, because hex8 locks at L/t = 200.
    - The Lamb sphere on voxel meshes.
- Docs:
  - `docs/eigensolvers.md`.
  - `docs/verification.md`, with exact setups, sources and reference values.
- Tests:
  - **0.1% target:** sliding block, Visscher Ritz and Srinivas–Rao within 0.1% at stated hex20/hex27 meshes.
  - **NAFEMS:** within 1% of target, after first checking the gap between the target and a fine hex27 run.
  - **Convergence rates:**
    - Measured against the exact sliding-block and Srinivas–Rao solutions on affine meshes. Fitted rate ≥ 1.8 for hex8 and ≥ 3.6 for hex20/hex27, over ≥ 3 levels in the asymptotic range. Richardson extrapolation is used only to estimate errors, never to fit rates.
    - Voxel Lamb sphere: fitted rate ≥ 0.8 over ≥ 4 levels, with split clusters compared by their mean.
  - **Mode matching:** free bodies (Visscher, Lamb, FV12) are compared from mode 7 on. Modes are paired with Srinivas–Rao by MAC and by the (m, n, family) labels.
  - **Upper bound:** consistent-mass eigenvalues converge from above on meshes of parallelepiped elements.

### PR 9: Generalized LOBPCG and public API
- Contents:
  - LOBPCG as under Key design decisions, including fixed shapes, the 6-block mode and floor-aware acceptance.
  - Jacobi on A_P.
  - Chunked mode computation; the deflation basis is optionally fp32, returned modes are fp64.
  - LOBPCG workspace estimator.
  - Public API: `solve_modes(mesh, materials, bcs, n_modes, ...) → (λ, V, info)`.
  - `examples/` script per milestone, also run as a smoke test.
  - Fixes the Mac acceptance budgets.
- Docs: `docs/eigensolvers.md` (algorithm, convergence theory, memory model, measured floor).
- Tests, at reference sizes against PR 8, run at tol = 1e-10:
  - Relative λ error ≤ 1e-8.
  - Principal angles: sin θ ≤ 10·r_M·λ/gap. Angles are in the M-inner product, the gap is absolute, and r_M is the residual in the M⁻¹-norm (dense at reference sizes). Clusters with relative gap < 1e-4 are compared by subspace.
  - Exact multiplicities on a square-section beam and a cube.
  - Free-free: ‖RᵀMX‖ ≤ 1e-10.
  - Near-singular M (void mass ≈ 1e-12): converges to tol = 1e-8 within 500 iterations.
  - The workspace estimator (9- and 6-block modes) is within 20% of the measured peak at 1M DOFs with m = 40.
  - The compiled step function compiles at most twice per solve.
- **Milestone:** first end-to-end modes in pure JAX.

## M3: Preconditioning (core of the project)

### PR 10: Linear algebra kernels and smoothers
- Contents:
  - PCG and MINRES.
  - Block-ELL SpMV, with the padding ratio reported and a sliced-ELL fallback.
  - 3×3 node-block Jacobi.
  - Chebyshev smoother with eigenvalue estimation (safety factor 1.1).
  - All applied to A_P.
- Docs: `docs/preconditioners.md`.
- Tests:
  - PCG and MINRES match `spsolve` to 1e-10.
  - Block-ELL matches SciPy CSR to 1e-12.
  - Iteration count is defined as PCG iterations on A_P, at fixed σ, to a relative residual of 1e-8. A study vs h is written to `benchmarks/results/`.

### PR 11: Geometric multigrid for voxel meshes
- Contents:
  - Trilinear transfer operators, with semi-coarsening (an axis stops coarsening at ≤ 2 elements).
  - Galerkin coarse operators stored in block-ELL on levels ≥ 1.
  - Symmetric Chebyshev smoothing.
  - Dense Cholesky on the coarsest level.
  - Memory estimator.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Iterations grow ≤ 20% over 3 refinements.
  - A checked-in synthetic SIMP-like density field (ρ ∈ [1e-3, 1], p = 3) converges to 1e-8 within 100 PCG iterations.
  - A single-layer mesh (n_z = 1) works.
  - Mac acceptance: a 2M-DOF voxel modal solve, 10 modes, wall time ≤ 10 min.
  - Deferred GPU check: a 100M voxel modal solve.

### PR 12: p-multigrid for unstructured meshes
- Contents:
  - hex20/hex27 → hex8 transfer operators, with symmetric Chebyshev smoothing on the high-order level.
  - The hex8 coarse level is solved exactly by SciPy `splu` via `pure_callback`. This is test-only and is replaced by AMG in PR 13.
- Docs: `docs/preconditioners.md`.
- Tests: iterations on hex20 grow ≤ 20% over 3 refinements.

### PR 13: AMG on the hex8 level
- Contents:
  - pyamg smoothed-aggregation setup on the assembled hex8 A_P, with the masked rigid modes as the near-null-space.
  - Hierarchy exported to block-ELL, with a V-cycle on the device.
  - Combined with PR 12.
  - Memory ordering: setup → free the assembled A → allocate LOBPCG.
  - Option to reuse aggregates and recompute only the values.
  - Memory estimator.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Setup time and host-memory benchmarks.
  - Mac acceptance: a 1M-DOF hex20 modal solve, 20 modes, wall time ≤ 15 min, peak within 20% of the estimator.
  - Optional Mac benchmark: 10M DOFs.
  - Deferred GPU check: 10M DOFs on one GPU.

## M3b: Frequency bands

### PR 14: Eigenvalue counting and [0, f_max]
- Contents:
  - Deflated stochastic count with M-weighted probes and a Chebyshev–Jackson step polynomial (50 probes, fixed seed list).
  - Cost-model strategy selection.
  - [0, f_max] mode reusing PR 9 chunking, stopping at λ > λ_max.
  - `solve_modes(..., band=(0, f_max))`.
- Docs: `docs/eigensolvers.md` (estimator, variance, degree and cost).
- Tests:
  - With one mode deliberately left out of V, the count detects it for ≥ 97 of a fixed list of 100 seeds. The test is deterministic.
  - With V complete, the estimate (including Jackson bias) is < 0.5.
  - A κ ≥ 1e7 case checks that inner-solve error and bias stay within the budget.
  - Counting cost (polynomial degree, wall time) vs mode index is reported.
  - No modes are missed compared with the dense inertia at reference sizes.

### PR 15: Interior bands
- Contents:
  - Shift-invert Lanczos with MINRES inner solves.
  - Slicing with the cost model.
  - Removal of duplicate modes at slice edges.
  - Band-pass count deflated by the modes found in the band.
  - `band=(f1, f2)`.
- Docs: `docs/eigensolvers.md` (memory model for many modes).
- Tests:
  - Matches SciPy `eigsh(sigma=...)` to relative error ≤ 1e-8.
  - Each slice's count estimate is < 0.5.

## M4: Solid-shell element

### PR 16: Solid-shell element (KGW 1999)
- Contents:
  - Thickness direction from a user-specified stack direction or a topological sweep; an error on ambiguity.
  - ANS and 5-parameter EAS as under Key design decisions.
  - Static condensation, with condensed Ke stored. This adds to the PR 5 estimator.
  - Laminates with ply angles in the local shell frame.
  - Mass from the displacement interpolation.
  - Simply supported BCs on solids are defined in the docs.
- Docs: `docs/elements.md`.
- Tests:
  - Flat patch test with lifting.
  - Exactly 6 zero eigenvalues after condensation.
  - MacNeal–Harder (against the MacNeal–Harder 1985 reference values):
    - Normalized tip deflection ≥ 0.95 on the regular mesh and ≥ 0.9 on the parallelogram mesh, for extension, in-plane shear and out-of-plane shear.
    - Twist is report-only.
    - Trapezoidal in-plane bending is report-only, because all 4-node membrane elements that pass the patch test lock there (MacNeal 1987).
  - **0.1% target (measured against 3D elasticity):**
    - Srinivas–Rao at L/t = 10 and 100 with ≥ k stacked elements per ply (k stated in the docs).
    - One layered element per laminate is reported, not gated.
    - A clamped cylinder (R/t and L/R stated) against a fine hex27 reference.
  - No locking: at L/t = 10 and 1000, the discretization error is measured against a converged solid-shell run at the same L/t. The error at L/t = 1000 is ≤ 1.5× the error at L/t = 10.
  - Distorted meshes (parallelogram only): frequencies within 1% of the undistorted mesh at the same density.
  - `jax.grad` matches finite differences.

## M5: Postprocessing and performance

### PR 17: Postprocessor
- Contents:
  - Mode normalization.
  - MAC.
  - Participation factors and effective modal mass.
  - Strain and stress recovery per mode.
  - VTU/XDMF export and a frequency table.
- Docs: `docs/postprocessing.md`.
- Tests:
  - On a small assembled model with all modes computed densely, Σ effective mass = rᵀMr over the free DOFs.
  - meshio round-trip of the export.

### PR 18: Performance pass
- Contents:
  - Morton reordering and reordering tuning.
  - Kernel fusion checks.
  - fp32 preconditioner validation.
  - Compile-time benchmark.
  - AMG iteration counts vs L/t for solid-shell models.
- Docs: `docs/performance.md`.
- Tests:
  - Roofline = min(bandwidth × arithmetic intensity, measured fp64 peak). The matvec reaches ≥ 50% of it at m = 1; for m > 1, the percentage of roofline is reported.
  - With the fp32 preconditioner, the same convergence tolerance is reached.
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
  - Matches reference iterates on Svanberg's toy problem to 1e-8. The iterates are generated once offline with a reference MMA port and committed as data, so there is no GPL dependency.
  - Solves Svanberg's test problems.

### PR 21: Eigenvalue sensitivities
- Contents:
  - Element-kernel sensitivity pass.
  - Cluster detection and the cluster sensitivity matrix.
  - Bound formulation (max β s.t. λ_j ≥ β, Du & Olhoff 2007), with per-mode bounds up to a gap cutoff J.
  - KS aggregation over complete clusters only: compute n + 2 modes and assert a gap at the cutoff.
- Docs: `docs/optimization.md`.
- Tests:
  - Central finite differences with a step-size sweep against dense `eigh` solved to ~1e-13 at small sizes, for coordinate, density, material and nodal-mass parameters. Error ≤ 1e-6·‖g‖_∞. The test eigenvalues have relative gap ≥ 1e-3. Density parameters enter through linear scaling (s_K = s_M = ρ); the PR 23 interpolations are tested there.
  - Double eigenvalue (symmetric cube or plate): one-sided directional differences along d match the eigenvalues of Σ_e d_e f^e. Both pairs are sorted before comparing, with error ≤ 1e-4·‖g‖_∞ and an O(h) step sweep.

### PR 22: FFD shape parametrization
- Contents:
  - Bernstein lattice → node coordinates, in JAX.
  - Nodes outside the lattice, the outer control-point layer (for continuity at the lattice boundary) and BC nodes are frozen.
  - Constraint: a KS/p-norm aggregate of detJ/detJ₀ ≥ ε.
  - Example: maximize the fundamental frequency subject to mass ≤ M₀, using MMA.
- Docs: `docs/optimization.md`.
- Tests:
  - Control-point gradients pass the PR 21 finite-difference criterion.
  - The example converges: relative objective change < 1e-4 over 5 iterations, λ₁ increased, no inverted elements.

### PR 23: Topology optimization
- Contents:
  - Densities:
    - Design ρ ∈ [0, 1].
    - Filtered ρ̃.
    - Physical ρ̄ = ρ_min + (1 − ρ_min)·H_β(ρ̃), with ρ_min = 1e-3. Without projection, H_β is the identity.
  - SIMP: E = ρ̄^p E₀ with p = 3.
  - Mass interpolation m(ρ̄) = ρ̄ for ρ̄ > 0.1, else c₁ρ̄⁶ + c₂ρ̄⁷ (c₁ = 6e5, c₂ = −5e6; C¹ at 0.1).
  - Volume constraint Σ ρ̄_e v_e ≤ V*.
  - Non-structural mass via nodal masses.
  - Filters:
    - Voxel meshes: radial cone density filter by direct convolution, normalized at boundaries.
    - The benchmark also uses a density filter, even if the paper used a sensitivity filter, so that gradients stay consistent. The 5% tolerance on ω₁ absorbs the difference.
    - Unstructured meshes: a Helmholtz PDE filter with PCG.
  - Optional Heaviside projection with β-continuation.
  - Bound formulation with MMA.
- First step of the PR: choose the Du & Olhoff 2007 example (and its volume fraction) and record it in `docs/verification.md`. The benchmark's pass criteria below are then fixed to that example.
- Docs: `docs/optimization.md`; `docs/verification.md` gets the full Du & Olhoff setup (table and figure numbers).
- Tests:
  - End-to-end finite differences through filter → projection → interpolation → KS/bound, with test densities away from ρ̄ = 0.1 ± h, where the mass interpolation is only C¹.
  - Re-run of the PR 11 SIMP-contrast test on a PR 23 design.
  - MMA with the PR 21 cluster constraints on a forced double eigenvalue: β is non-decreasing (to 1e-3 relative) after iteration k₀, with no mode-switch oscillation.
  - Benchmark: the chosen Du & Olhoff 2007 example, with the paper's filter radius, and projection only if the paper uses it. The setup:
    - one element layer of thickness t ≤ h;
    - u_z = 0;
    - plane-stress-equivalent material E′ = E(1 + 2ν)/(1 + ν)², ν′ = ν/(1 + ν).
  - Pass criteria for the benchmark:
    - ω₁ within 5% of the published value.
    - If the paper's example is bimodal: relative gap λ₂/λ₁ − 1 ≤ 1%.
    - Modes antisymmetric in z lie above the computed band.
    - No localized modes: in each reported mode, the kinetic-energy fraction in elements with ρ̄ < 0.1 is < 5%.

## GPU validation

### PR 24: GPU validation (whenever a CUDA machine is available)
- Runs every item in the deferred list below.
- Records results in `benchmarks/results/`.
- Adds Pallas kernels if PR 18's roofline criterion calls for them.

**Deferred GPU checks:**
- PR 6: 100M matvec + LOBPCG and preconditioner workspace fits in 80 GB.
- PR 11: 100M voxel modal solve.
- PR 13: 10M hex20 modal solve on one GPU.
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
| 14 Counting, [0, f_max] | 9, 10, 11 (13 optional) |
| 15 Interior bands | 14; 13 for unstructured meshes |
| 16 Solid-shell | 3, 5, 8 |
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
