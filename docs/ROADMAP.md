# Roadmap

Large-scale 3D linear modal analysis (Kv = λMv) in JAX. The solver uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. FFD shape optimization and topology optimization come later.

**Targets:**
- Lowest modes and modes in frequency bands.
- Frequency accuracy of 0.1% against a converged reference, for hex20/hex27 and solid-shell meshes.
- 10M DOFs, unstructured, on a single GPU.
- 100M DOFs on voxel meshes with a small number of modes per chunk (see the memory budget).
- Multi-GPU later.
- An FP64-capable GPU (A100/H100 class) for production runs.
- Until a GPU is available, development and all tests run on an Apple M3 Max (64 GB, CPU backend, fp64).

Package name: `jaxmodal`. License: MIT.

Status: planning. Review rounds 1–2 are incorporated. Items marked *(default, pending user)* are provisional.

## Principles

1. **Validate every fast path against a slow one.** References are SciPy, dense NumPy, or analytic or published values. Every acceptance criterion is quantitative.
2. **One small interface.** Every discretization (unstructured, voxel) provides `apply_K`, `apply_M`, `diag_K`, `diag_M` and the rigid body modes. Each operator applies natively to blocks of shape `(n_dof, m)`, not via `vmap`. Solvers only see this interface plus `apply_P`.
3. **Fixed shapes; everything else is a `jit` argument.**
   - Topology preprocessing (connectivity, DOF maps, sets) runs in NumPy.
   - Every array is a pytree `jit` argument: int32 topology, geometry, densities, material parameters, stored Ke, multigrid/AMG matrices and rigid modes. Nothing is closed over, so there are no huge HLO constants.
   - Only shapes, chunk sizes, element types and `n_modes` are static.
4. **Extensible element types.**
   - Each element type is one registry entry: node count, shape functions, quadrature rule and reference faces.
   - Per-format node-ordering maps live in `mesh/io` (PR 1), and the registry refers to them.
   - Only hex types are implemented now; wedges and tets can be added later without touching operators or solvers.
5. **Differentiable from day one.**
   - Element kernels are pure JAX functions of `(X_e, C_e, ρ_e)`: energies and bilinear forms uᵀKe w and uᵀMe w.
   - Material constructors are traceable.
   - Sensitivities are a separate pass over the element kernels. Solvers and preconditioners are never differentiated.
6. **Memory-aware.**
   - Elements are processed in fixed-size chunks (`lax.map`/`scan`), with the last chunk padded.
   - Connectivity is int32, with an assert that N_dof < 2³¹.
   - Large buffers are donated.
   - Estimators include chunk temporaries (chunk × n_dof_e × m × 8 B) and are checked against measured peaks.
7. **Conventions** live in `docs/conventions.md` and are fixed in PR 0:
   - VTK node ordering internally, converted at import.
   - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear strains.
   - fp64 by default, enabled through an explicit config call (no import side effects).
   - Index arrays are always int32.
   - Iteration vectors, Gram matrices and Rayleigh–Ritz are always fp64. fp32 is allowed only inside preconditioners and for storing locked modes.

## Key design decisions

- **Integration:** full integration only (hex8 2³, hex20/hex27 3³) for K and consistent M. No reduced integration, because its zero-energy modes would pollute the spectrum.
- **Mass:**
  - Consistent mass is the default.
  - Lumped mass is optional: row-sum for hex8/hex27, HRZ for hex20. Row-sum lumping gives non-positive corner masses for hex20.
- **Hex8:** plain, fully integrated hex8 is accepted. It shear-locks in bending, so bending accuracy comes from refinement or from hex20/hex27. Incompatible modes are a possible later addition.
- **Dirichlet BCs** (axis-aligned only; inclined symmetry planes raise an error) *(default, pending user)*:
  - Eigen operators are purely masked: `apply_K(v) = m ⊙ K(m ⊙ v)`, the same for M, with all iterates masked.
  - Every preconditioner, linear solve and inertia count instead uses A_P = m A m + (I − diag m): a unit diagonal on constrained DOFs, so the matrix is not singular. Preconditioner output is masked.
- **Rigid body modes:**
  - Computed analytically per connected part.
  - Under Dirichlet or symmetry sets, the remaining rigid modes are computed automatically as R·null((I − m)R) via an SVD, then M-orthonormalized and deflated with P = I − R Rᵀ M.
  - A user override is optional. No penalty.
- **Preconditioning:** preconditioners approximate A = K + σM with σ > 0 (default ≈ estimated λ_target), used as A_P. A is SPD on the free DOFs in both the free-free and Dirichlet cases.
- **Eigensolver:** generalized LOBPCG following Hetmaniuk–Lehoucq / Duersch et al. 2018:
  - SVQB orthonormalization with a drop tolerance.
  - P is dropped when ill-conditioned.
  - Implicit product updates with periodic explicit recompute.
  - The deflation projector is re-applied every iteration.
- **Convergence:**
  - Measured by the λ-scaled residual r = ‖Kx − λMx‖ / (|λ|‖Mx‖) ≤ tol, with default tol = 1e-6.
  - This gives sin θ ≲ tol·λ/gap and δλ/λ ≲ tol²·λ/gap. The normwise backward error doesn't certify low modes at κ = λ_max/λ_min ≈ 1e8–1e10, so it is only reported.
  - A cluster is a set of eigenvalues whose relative gaps are < 100·tol.
  - Block size = nev + max(5, 0.2·nev).
  - Chunk and lock cutoffs must fall in a gap.
- **Precision:**
  - The attainable residual floor is about ε·κ: roughly 1e-7–1e-6 in fp64, fine for 0.1% targets but O(1) in fp32. Hence fp32 never goes into X, R or P.
  - PR 9 measures the floor. The 1e-8 eigenvalue agreement criterion applies only at reference sizes, where κ is small.
- **Frequency bands (PR 21):**
  - **[0, f_max]:** chunked LOBPCG until λ > λ_max.
  - **Interior bands [f₁, f₂]:**
    - Shift-invert Lanczos on (K − σM)⁻¹M with σ in the band, full M-reorthogonalization, and a final Rayleigh–Ritz with true residuals on (K, M).
    - Inner MINRES with absolute-value preconditioning by the SPD multigrid of K + σM (Vecharynski–Knyazev 2013).
    - Fixed inner tolerance τ = 1e-8.
    - Slice widths come from the iteration cost model (∝ 1 / relative distance to the slice edge). Duplicate modes at slice edges are removed by M-orthogonalization.
  - **Completeness:**
    - At scale, a deflated stochastic count tr(step(T)) − rank(V) on T = (K + σM)⁻¹M, with PCG + multigrid inner solves, a fixed seed and probe count, and a stated confidence level *(default, pending user)*.
    - At reference sizes (≤ ~15k DOF), an exact count from the inertia of dense `scipy.linalg.ldl`.
  - **Mass in band mode:** lumped M is allowed *(default, pending user)*.
  - The same machinery is the fallback when nev ≳ 100 or LOBPCG preconditioning is poor.
- **Preconditioners:**
  - Voxel meshes: geometric multigrid.
  - Unstructured meshes: p-multigrid (hex20/hex27 → hex8), then smoothed-aggregation AMG. AMG setup runs in pyamg on the host; the V-cycle runs on the device with a self-written block-ELL SpMV.
  - No `jax.experimental.sparse` in hot paths.
- **Shells:** solid-shell elements (3 DOFs per node) following Klinkel–Gruttmann–Wagner (1999):
  - ANS for transverse shear.
  - ANS for the thickness strain at the 4 corner points.
  - EAS built with the center Jacobian J₀: one thickness parameter (linear in ζ) is mandatory, 4 Simo–Rifai membrane parameters are optional, and ∫M dV = 0 holds for the patch test.
  - Laminates with n Gauss points per ply.
- **Sensitivities:**
  - Formula: dλ/dp = vᵀ(∂K/∂p − λ∂M/∂p)v / vᵀMv.
  - Computed as a chunked `vmap(grad(element energy))` pass with v held fixed, independent of the operator mode.
  - Valid only for simple eigenvalues, with v converged to r ≤ 1e-8.
  - Clusters use the sensitivity matrix f_sk = v_sᵀ(K′ − λM′)v_k, built from the bilinear forms.
  - Deflation and masking leave the formula unchanged.
- **Optimizer:** MMA implemented in JAX (~200 LOC). No OC, because eigenvalue sensitivities change sign.
- **Scatter vs gather:** XLA:CPU scatter is effectively serial. PR 5 benchmarks scatter against an inverse-map gather on each backend, and the default may differ per backend.
- **Dependencies:**
  - Core: `jax` (pinned minimum version), `numpy`, `scipy`, `meshio`, `pyamg`, `pymetis`, all with macOS arm64 wheels.
  - Versions are locked with `uv.lock`.
  - SLEPc/MUMPS is an optional conda extra. scikit-sparse/CHOLMOD is optional. pypardiso is not used (no arm64 build).
  - meshio's `.inp` reader doesn't handle parts or assemblies; a small custom reader is added only if needed.

## Memory budget

All sizes are fp64 unless noted.

| Item | 10M DOF unstructured hex20 (~0.8M elements) | 100M DOF voxel hex8 (~33M elements) |
|---|---|---|
| One vector | 0.08 GB | 0.8 GB |
| Operator | stored Ke ≈ 11.7 GB (symmetric) or on-the-fly | reference Ke + s_K, s_M < 1 GB |
| LOBPCG, 9 blocks × m | m = 30: ≈ 22 GB | m = 10: ≈ 72 GB; doesn't fit with the operator and multigrid |
| LOBPCG, 6 blocks (lumped M or recomputed products) | m = 30: ≈ 14 GB | m = 8: ≈ 38 GB |
| Multigrid | hex8-level AMG ≈ 3–4 GB (device); pyamg setup peak ≈ 20+ GB (host) | Galerkin levels ≈ 9.3 GB (4.7 GB in fp32) |
| Locked modes | n_locked × 0.08 GB (fp32: half) | n_locked × 0.8 GB (fp32: half) |

- **80 GB GPU:** both columns fit. At 100M DOFs, modes are computed in chunks of m = 8 with 6 fp64 blocks. Locked modes are stored in fp32 or in host memory; host storage adds a host→device transfer every iteration.
- **64 GB Mac (unified memory, so host storage saves nothing):**
  - The 10M column is tight. Order the work as: AMG setup → free the assembled A → allocate LOBPCG.
  - Locked modes go in fp32 or an `np.memmap`.
  - The 100M column does not fit.
- The PR 5/6/9 estimators reproduce this table and are validated against measured peaks.

## Development environment

- **Development:** Apple M3 Max with 16 cores and 64 GB of unified memory, using the JAX CPU backend in fp64. The 10M-DOF unstructured case runs here; it fits but is slow.
- **No Mac GPU:** the GPU isn't usable for this project. Metal GPUs have no fp64 hardware, and the `jax-metal` plugin is experimental with incomplete op coverage.
- **Local checks:** `scripts/check.sh` runs lint plus the fast suite, `pytest -m "not slow and not gpu"`, which must finish in under 5 minutes on the M3. Remote CI comes later.
- **Merge gate:** `pytest -m "not gpu"`, slow tests included.
- **GPU checks:** GPU acceptance criteria go into the "Deferred GPU checks" list below and are run in PR 23 once a CUDA machine is available. Each affected PR has a Mac stand-in criterion *(default, pending user)*.
- **Benchmarks:** JSON results are committed to `benchmarks/results/` and record:
  - jax and jaxlib versions;
  - the measured STREAM bandwidth, which is the CPU roofline;
  - time, peak memory and compile time.
- **Compilation cache:** the persistent compilation cache is enabled in tests.
- **Multi-GPU development:** `shard_map` code is developed on the Mac with `--xla_force_host_platform_device_count`.
- **Data:**
  - Tiny test meshes live in `tests/data/`.
  - Large benchmark meshes are generated by scripts in `benchmarks/meshes/` and git-ignored.
- **Solver output:** solvers return an `info` dict (iterations, residual history, timings). No logging framework.

## Package layout

```
src/jaxmodal/
  config.py    dtype / x64 setup
  mesh/        data model, io (incl. node-ordering maps), voxel generator, reordering, partition
  materials/   anisotropic C, Bond rotations, orientations
  elements/    registry, quadrature, shape functions, hex8/20/27, solid-shell
  operators/   dofmap, assembled (oracle), matrix-free, voxel stencil, constraints (masking, A_P, rigid modes, deflation)
  linalg/      pcg, minres, block-ELL spmv, dense inertia
  precond/     block jacobi, chebyshev, gmg, pmg, amg
  eigen/       lobpcg, bands (counting, shift-invert Lanczos), api (solve_modes), backends/{scipy, slepc}
  post/        normalization, MAC, effective mass, stresses, export
  opt/         sensitivities, mma, ffd, simp, filters
tests/  benchmarks/  examples/  docs/  scripts/
```

Each PR ships its docs (with LaTeX math) in `docs/`, listed below as **Docs**.

## M0: Foundations

### PR 0: Project scaffold
- Contents:
  - pyproject, `uv.lock` and lint config.
  - pytest markers `slow` and `gpu`.
  - `scripts/check.sh`.
  - `config.py` (an explicit x64 switch).
  - Persistent compilation cache.
  - A benchmark harness that records time, peak memory, compile time, versions and STREAM bandwidth.
  - LICENSE (MIT), version 0.x.
- Docs: `README.md`, `docs/conventions.md`.
- Tests:
  - Dtype-leak test: an fp32 path stays fp32 with x64 enabled.
  - Index arrays from `argsort`, `nonzero` and `arange` are cast back to int32.

### PR 1: Mesh data model and parser
- Contents:
  - `Mesh` dataclass: coordinates, element blocks keyed by type, node and element sets, material and orientation IDs.
  - meshio readers for Abaqus `.inp` (flat), Gmsh and VTU.
  - Per-format node-ordering maps for hex20/hex27. Gmsh, Abaqus and VTK orderings differ.
  - Structured voxel mesh generator.
- Docs: `docs/mesh.md`.
- Tests:
  - Read → write → read round-trips.
  - Orderings match hand-built references for every format.
  - hex8 corner Jacobians are positive. The full quadrature-point check is in PR 3.

### PR 2: Materials
- Contents:
  - Traceable 6×6 C constructors: isotropic, transversely isotropic, orthotropic (ν_ij/E_i = ν_ji/E_j convention), full anisotropic.
  - Positive-definiteness check.
  - Stress and strain Bond matrices for rotation into local axes.
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
  - Batched Ke and Me (consistent, and lumped as described under Mass), plus the bilinear forms.
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
  - Masking and A_P.
- Docs: `docs/operators.md`.
- Tests:
  - Distorted multi-element patch test: constant stress to 1e-10.
  - Clamped model: no spurious eigenvalues compared with SciPy on the reduced matrices.
  - A_P is SPD on a clamped model, and Jacobi built on it has no zero divisions.

### PR 5: Matrix-free unstructured operator
- Contents:
  - Two modes: stored Ke, and on-the-fly computation from precomputed invJ·detJ per quadrature point.
  - Chunked, block-native (n_dof, m) apply.
  - Scatter (`segment_sum`) vs inverse-map gather, benchmarked per backend.
  - `diag_K`, `diag_M`.
  - Memory estimator.
  - Node-local numbering, so partitioning is possible later.
- Docs: `docs/operators.md`.
- Tests:
  - Both modes match PR 4 to 1e-12 relative. Tests use tolerances, never bitwise equality.
  - The estimator is within 20% of the measured peak.
  - Compile time doesn't depend on mesh size at a fixed shape.

### PR 6: Voxel stencil operator
- Contents:
  - K matvec as 8 shifted corner slices → einsum with the reference Ke → 8 shifted adds.
  - Separate per-element stiffness and mass scaling arrays, s_K and s_M.
  - `apply_M` (consistent and lumped), `diag_K`, `diag_M`.
  - Restriction: one material and orientation per phase.
- Docs: `docs/operators.md`.
- Tests:
  - Matches PR 5 on the same mesh.
  - `jax.grad` with respect to s_K and s_M matches finite differences.
  - Mac stand-in: estimator within 20% of the measured peak at 1M and 10M DOFs, extrapolated to 100M.
  - Deferred GPU check: a 100M-DOF matvec plus the LOBPCG workspace (6 blocks, m = 8) fits in 80 GB.

### PR 7: Rigid body modes and deflation
- Contents:
  - Rigid modes per connected part.
  - Automatic remaining modes under Dirichlet or symmetry sets: R·null((I − m)R).
  - M-orthonormalization.
  - Projector P = I − R Rᵀ M.
  - Optional user override.
- Docs: `docs/boundary_conditions.md`.
- Tests:
  - ‖KR‖ ≤ 1e-12·‖K‖‖R‖.
  - ‖RᵀMR − I‖ ≤ 1e-12.
  - A free block with 1, 2 or 3 axis-aligned symmetry planes leaves 3, 1 or 0 rigid modes.
  - Two disconnected parts give 12 rigid modes.
  - Projected iterates stay M-orthogonal to R.

## M2: First modes end-to-end

### PR 8: Reference eigensolver and verification suite
- Contents:
  - SciPy `eigsh` shift-invert (negative shift) on assembled matrices, with its largest supported size stated.
  - Dense inertia via `scipy.linalg.ldl` (≤ ~15k DOF).
  - In-test oracles:
    - A rectangular block with sliding BCs (exact separable solution, including orthotropic).
    - Visscher's xyz-polynomial Ritz method (dense `eigh`, converged to 1e-6) for a free anisotropic parallelepiped with arbitrary C.
    - The Srinivas–Rao exact 3D solution for thick orthotropic simply supported plates.
  - Published benchmarks:
    - NAFEMS FV52 (simply supported thick plate).
    - NAFEMS FV12 (free thin plate, exercises deflation), with hex20/hex27 only, because hex8 locks at L/t = 200.
    - The Lamb sphere on voxel meshes.
- Docs:
  - `docs/eigensolvers.md`.
  - `docs/verification.md`, with the exact setups, sources and reference values.
- Tests:
  - **0.1% target:** sliding block and Visscher Ritz within 0.1% at stated hex20/hex27 meshes.
  - **Published benchmarks:** NAFEMS within 1% of the published targets.
  - **Convergence rates:**
    - O(h^{2p}) on affine meshes against a Richardson-extrapolated reference: hex8 h², hex20/hex27 h⁴. Serendipity hex20 loses order on non-affine meshes (Arnold–Boffi–Falk 2002).
    - Voxel meshes: fitted rate ≥ 0.8.
    - The Lamb sphere's split degenerate clusters are compared by their mean.
  - **Upper bound:** consistent-mass eigenvalues converge from above, checked only on exact polyhedral geometry.

### PR 9: Generalized LOBPCG and public API
- Contents:
  - LOBPCG with the convergence and cluster rules under Key design decisions.
  - Jacobi on A_P.
  - Chunked mode computation, with locked modes stored in fp32 or an `np.memmap`.
  - LOBPCG workspace estimator.
  - Public API: `solve_modes(mesh, materials, bcs, n_modes, ...) → (λ, V, info)`.
  - `examples/` script per milestone, also run as a smoke test.
- Docs: `docs/eigensolvers.md` (algorithm, convergence theory, memory model, measured residual floor).
- Tests, at reference sizes against PR 8:
  - Relative λ error ≤ 1e-8.
  - Principal angles ≤ 1e-6, with clusters compared by subspace.
  - Exact multiplicities on a square-section beam and a cube.
  - Free-free: ‖RᵀMX‖ ≤ 1e-10.
  - Near-singular M (void mass ≈ 1e-12) converges.
  - The workspace estimator is within 20% of the measured peak.
- **Milestone:** first end-to-end modes in pure JAX.

## M3: Preconditioning (core of the project)

### PR 10: Linear algebra kernels and smoothers
- Contents:
  - PCG and MINRES.
  - Block-ELL SpMV, including the padding ratio and a sliced-ELL fallback.
  - 3×3 node-block Jacobi.
  - Chebyshev smoother with eigenvalue estimation.
  - All applied to A_P.
- Docs: `docs/preconditioners.md`.
- Tests:
  - PCG and MINRES match `spsolve` to 1e-10.
  - Block-ELL matches SciPy CSR to 1e-12.
  - Iteration count is defined as PCG iterations on A_P to a relative residual of 1e-8. A study vs h is written to `benchmarks/results/`.

### PR 11: Geometric multigrid for voxel meshes
- Contents:
  - Trilinear transfer operators.
  - Galerkin coarse operators stored in block-ELL on levels ≥ 1.
  - Chebyshev smoothing.
  - Dense Cholesky on the coarsest level.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Iterations grow ≤ 20% over 3 refinements.
  - A synthetic SIMP-like density field (checked in; ρ ∈ [1e-3, 1], p = 3, giving a stiffness contrast of 1e-9) converges, with iteration counts reported. This is re-run on a PR 20 design later.
  - Mac stand-in: a 10M-DOF voxel modal solve, with wall time recorded.
  - Deferred GPU check: a 100M-DOF voxel modal solve.

### PR 12: p-multigrid for unstructured meshes
- Contents: hex20/hex27 → hex8 transfer operators, with a Chebyshev smoother on the high-order level.
- Docs: `docs/preconditioners.md`.
- Tests: iterations on hex20 grow ≤ 20% over 3 refinements.

### PR 13: AMG on the hex8 level
- Contents:
  - pyamg smoothed-aggregation setup on the assembled hex8 A_P, with rigid modes as the near-null-space.
  - Hierarchy exported to block-ELL, with a V-cycle on the device.
  - Combined with PR 12.
  - Memory ordering: setup → free the assembled A → allocate LOBPCG.
  - Option to reuse aggregates and recompute only the values.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Setup time and host-memory benchmarks.
  - Mac: a 10M-DOF hex20 modal solve with 20 modes, peak ≤ 50 GB, wall time ≤ 3 h *(default, pending user)*.
  - Deferred GPU check: the same run on one GPU.

## M3b: Frequency bands

### PR 21a: Eigenvalue counting and [0, f_max]
- Contents:
  - A deflated stochastic count on T = (K + σM)⁻¹M, with a fixed seed and probe count.
  - A [0, f_max] mode that reuses PR 9 chunking and stops at λ > λ_max.
  - `solve_modes(..., band=(0, f_max))`.
- Docs: `docs/eigensolvers.md`.
- Tests:
  - The count matches the dense inertia within ±1 with 99% confidence on the verification suite, including a κ ≥ 1e7 case.
  - No modes are missed at reference sizes.

### PR 21b: Interior bands
- Contents:
  - Shift-invert Lanczos with MINRES inner solves.
  - Slicing with the cost model.
  - Removal of duplicate modes at slice edges.
  - `band=(f1, f2)`.
- Docs: `docs/eigensolvers.md` (including the memory model for many modes).
- Tests:
  - Matches SciPy `eigsh(sigma=...)` to relative error ≤ 1e-8.
  - The per-slice count check passes.

## M4: Solid-shell element

### PR 14: Solid-shell element (KGW 1999)
- Contents:
  - Thickness direction from a user-specified stack direction or a topological sweep through connected layers; an error on ambiguity.
  - ANS and EAS as specified under Key design decisions.
  - Static condensation, with condensed Ke stored. This adds to the PR 5 estimator.
  - Laminates with ply angles in the local shell frame.
  - Mass from the displacement interpolation.
  - Simply supported BCs on solids are defined in the docs.
- Docs: `docs/elements.md`.
- Tests:
  - Patch test.
  - Exactly 6 zero eigenvalues after condensation.
  - MacNeal–Harder in-plane bending.
  - No locking: the error at L/t = 1000 is ≤ 1.5× the error at L/t = 10 on the same mesh.
  - A clamped cylinder against a fine hex27 reference.
  - Distorted meshes.
  - Laminates:
    - Navier/Reddy CLPT for simply supported cross-ply at L/t ≥ 100.
    - Srinivas–Rao for thick plates.
    - Both within 1%.
  - `jax.grad` matches finite differences.

## M5: Postprocessing and performance

### PR 15: Postprocessor
- Contents:
  - Mode normalization.
  - MAC.
  - Participation factors and effective modal mass.
  - Strain and stress recovery per mode.
  - VTU/XDMF export and a frequency table.
- Docs: `docs/postprocessing.md`.
- Tests:
  - On a small assembled model with all modes computed densely, Σ effective mass = rᵀMr over the free DOFs. In free-free models the rigid modes carry all of it.
  - meshio round-trip of the export.

### PR 16: Performance pass
- Contents:
  - RCM/Morton reordering.
  - Kernel fusion checks.
  - fp32 preconditioner validation.
  - Compile-time benchmark.
  - AMG iteration counts vs L/t for solid-shell models.
  - Pallas kernels are deferred to the GPU phase, and only if the matvec reaches < 50% of the bandwidth roofline.
- Docs: `docs/performance.md`.
- Tests:
  - Matvec ≥ 50% of the measured STREAM bandwidth on the Mac.
  - With the fp32 preconditioner, the same convergence tolerance is reached.
  - Solid-shell AMG iterations grow ≤ 2× from L/t = 10 to L/t = 1000, or a fallback is documented.

### PR 17: Partitioning and multi-GPU
- Status: an epic, to be split before starting.
- Contents:
  - METIS partitions padded to uniform shard sizes.
  - `shard_map` operator with `ppermute` halos.
  - `psum` Gram matrices in LOBPCG.
  - Distributed GMG first.
  - Developed on the Mac with forced host devices.
- Docs: `docs/parallel.md`.
- Tests:
  - Results match single-device runs.
  - Deferred GPU check: weak scaling.

## M6: Optimization

Working assumptions, still open: topology optimization at ≤ 10M voxel elements, clamped, densities as the only design variables. The user will decide later. Principle 5 keeps the alternatives open.

### PR 22: MMA
- Contents: MMA in JAX.
- Docs: `docs/optimization.md`.
- Tests:
  - Matches the iterates of a reference Python MMA port on Svanberg's toy problem to 1e-8.
  - Solves Svanberg's test problems.

### PR 18: Eigenvalue sensitivities
- Contents:
  - Element-kernel sensitivity pass, as under Key design decisions.
  - Cluster detection and the cluster sensitivity matrix.
  - Bound formulation (max β s.t. λ_j ≥ β, Du & Olhoff 2007), with a bound constraint on every mode up to a gap cutoff J.
  - KS aggregation over complete clusters only: compute n + 2 modes and assert a gap at the cutoff.
- Docs: `docs/optimization.md`.
- Tests:
  - Central finite differences with a step-size sweep against dense `eigh`/`eigsh` solved to ~1e-13 at small sizes: rel. err. < 1e-6 for coordinate, density and material parameters.
  - Directional finite-difference check on a double eigenvalue (symmetric cube or plate) using the eigenvalues of the cluster matrix.

### PR 19: FFD shape parametrization
- Contents:
  - Bernstein lattice → node coordinates, in JAX.
  - Nodes outside the lattice and BC nodes are frozen.
  - Constraint: a KS/p-norm aggregate of detJ/detJ₀ ≥ ε.
  - Example: maximize the fundamental frequency at constant mass, using MMA.
- Docs: `docs/optimization.md`.
- Tests:
  - Control-point gradients pass the PR 18 finite-difference criterion.
  - The example converges with no inverted elements.

### PR 20: Topology optimization
- Contents:
  - SIMP: E = E_min + ρ^p(E₀ − E_min), with ρ ∈ [1e-3, 1].
  - Mass interpolation m = ρ for ρ > 0.1, else c₁ρ⁶ + c₂ρ⁷ (c₁ = 6e5, c₂ = −5e6; C¹ at 0.1), per Du & Olhoff 2007.
  - Filters:
    - Voxel meshes: a separable product of 1D hat filters, normalized at boundaries.
    - Unstructured meshes: a Helmholtz PDE filter with PCG.
  - Heaviside projection with β-continuation.
  - Bound formulation with MMA.
- Docs: `docs/optimization.md`; `docs/verification.md` gets the full Du & Olhoff setup (domain, mesh, volume fraction, non-structural mass, p, filter radius, ρ_min, E, ν, ρ, with table and figure numbers).
- Tests: reproduces a Du & Olhoff 2007 beam example as one element layer with u_z = 0 and a plane-stress-equivalent material (E′ = E(1 + 2ν)/(1 + ν)², ν′ = ν/(1 + ν)):
  - ω₁ within 5% of the published value.
  - λ₁ ≈ λ₂ if the chosen example is bimodal in the paper; check this against the paper.
  - No localized modes (MAC and energy-fraction check).

## GPU validation

### PR 23: GPU validation (when a CUDA machine is available)
- Runs every item in the deferred list below.
- Records results in `benchmarks/results/`.
- Adds Pallas kernels if PR 16's roofline criterion calls for them.

**Deferred GPU checks:**
- PR 6: 100M matvec + LOBPCG workspace fits in 80 GB.
- PR 11: 100M voxel modal solve.
- PR 13: 10M hex20 modal solve on one GPU.
- PR 17: weak scaling.

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
| 10 Linalg kernels, smoothers | 5, 6 |
| 11 Voxel GMG | 6, 9, 10 |
| 12 p-MG | 5, 9, 10 |
| 13 AMG | 12 |
| 14 Solid-shell | 3, 5, 8 |
| 15 Postprocessor | 9 |
| 16 Performance | 11, 13, 14 |
| 17 Multi-GPU | 16 |
| 21a Counting, [0, f_max] | 9, 10, 11 (13 optional) |
| 21b Interior bands | 21a |
| 22 MMA | 0 |
| 18 Sensitivities | 9 |
| 19 FFD | 18, 22 |
| 20 Topology opt | 11, 18, 22 |
| 23 GPU validation | CUDA hardware; the PRs in the deferred list |

- **Critical path:** 0 → 1 → 3 → 4 → 5/7 → 8 → 9 gives first modes in pure JAX.
- **Biggest risk:** M3 (preconditioning).
- **Off the critical path:** PRs 6, 14, 15 and 22 can be developed alongside it.
- **Numbering:** PR numbers are identifiers, not a strict order; follow this table.
