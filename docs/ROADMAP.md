# Roadmap

Large-scale 3D linear modal analysis (Kv = λMv) in JAX. The solver uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. FFD shape optimization and topology optimization come later.

**Targets:**
- Lowest modes and modes in frequency bands.
- 10M DOFs, unstructured, on a single GPU.
- 100M DOFs on voxel meshes with a small number of modes per chunk (see the memory budget).
- Multi-GPU later.
- An FP64-capable GPU (A100/H100 class) for production runs; dtype is configurable.
- Until a GPU is available, development and all tests run on an Apple M3 Max (64 GB, CPU backend, fp64).

Package name: `jaxmodal` (free on PyPI as of 2026-10-07). License: MIT.

Status: planning; review round 1 and the user's answers incorporated.

## Principles

1. **Validate every fast path against a slow one.** References are SciPy (assembled matrices, `eigsh`, `spsolve`), dense NumPy, or analytic or published values. Every acceptance criterion is quantitative.
2. **One small interface.** Every discretization (unstructured, voxel) provides `apply_K`, `apply_M`, `diag_K`, `diag_M` and the rigid body modes. Solvers only see this interface plus `apply_P`.
3. **Fixed array shapes, element groups by type.**
   - Topology preprocessing (connectivity, DOF maps, sets) runs in NumPy.
   - Geometry, densities and material parameters are JAX arrays passed as `jit` arguments, never closed over. A design change therefore needs no NumPy step and no re-JIT.
4. **Extensible element types.** Each element type is one registry entry: node count, shape functions, quadrature rule, reference faces and node-ordering maps per file format. Only hex types are implemented now; wedges and tets can be added later without touching operators or solvers.
5. **Differentiable from day one.** Element kernels are pure JAX functions of `(X_e, C_e, ρ_e)`, and material constructors are traceable. M6 then needs no rework.
6. **Memory-aware.**
   - Elements are processed in fixed-size chunks (`lax.map`/`scan`).
   - Connectivity is int32, with an assert that N_dof < 2³¹.
   - Large buffers are donated to `jit`.
   - Memory estimates are checked against measured peaks.
7. **Conventions** live in `docs/conventions.md` and are fixed in PR 0:
   - VTK node ordering internally (converted at import).
   - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear strains.
   - fp64 by default, enabled through an explicit config call (no import side effects).
   - Gram matrices and Rayleigh–Ritz always run in fp64. fp32 is allowed only inside preconditioners and for storage.

## Key design decisions

- **Integration:** full integration only (hex8 2³, hex20/hex27 3³) for K and consistent M. No reduced integration, because its zero-energy modes would pollute the spectrum.
- **Mass:**
  - Consistent mass is the default; it gives eigenvalues that are upper bounds.
  - Lumped mass is optional: row-sum for hex8/hex27, HRZ for hex20. Row-sum lumping gives non-positive corner masses for hex20.
- **Hex8:** plain, fully integrated hex8 is accepted. Its bending accuracy comes from refinement or from hex20. Incompatible modes are a possible later addition.
- **Dirichlet BCs:**
  - Applied by masking: `apply_K(v) = m ⊙ K(m ⊙ v)`, the same for M, with all iterates masked. Constrained DOFs leave the search space.
  - No unit diagonals in both K and M; those create spurious λ = 1 modes.
- **Free-free:**
  - Analytic rigid body modes (rotations about the center of mass), M-orthonormalized and deflated with P = I − R Rᵀ M. No penalty.
  - When Dirichlet sets exist, deflation applies only to the remaining rigid modes the user lists (e.g. symmetry planes).
  - Disconnected parts are detected; each contributes 6 rigid modes.
- **Preconditioning:** preconditioners approximate A = K + σM with σ > 0 (default ≈ estimated λ_target). A is SPD in both the free-free and Dirichlet cases. The eigenproblem itself is not shifted.
- **Eigensolver:** generalized LOBPCG following Hetmaniuk–Lehoucq / Duersch et al. 2018:
  - SVQB orthonormalization with a drop tolerance.
  - P is dropped when ill-conditioned.
  - Implicit product updates with periodic explicit recompute.
  - The deflation projector is re-applied every iteration.
  - Convergence is measured by the backward error ‖Kx − λMx‖ / ((‖K‖ + |λ|‖M‖)‖x‖), with norms estimated by power iteration.
  - Block size = nev + max(5, 0.2·nev) guard vectors. Locking never splits an unresolved cluster.
- **Frequency bands (PR 21):**
  - [0, f_max]: chunked LOBPCG until λ > λ_max. Completeness is checked with a matrix-free stochastic eigenvalue count (Chebyshev/KPM) and, at reference sizes, the LDLᵀ inertia (Sturm count).
  - Interior bands [f₁, f₂]: shift-invert Lanczos/Krylov–Schur on (K − σM)⁻¹M with σ inside the band. The inner solve uses MINRES, since K − σM is indefinite, with the SPD multigrid of K + |σ|M as preconditioner. Wide bands are split into slices.
  - The same machinery is the fallback when nev ≳ 100 or LOBPCG preconditioning is poor.
- **Precision:**
  - Round-off error in the smallest eigenvalues scales like ε·λ_max/λ_min, with λ_max/λ_min ≈ 1e8–1e10. That gives about 1e-7 relative error in fp64 (fine), but O(1) in fp32 (useless).
  - Hence fp64 for everything except preconditioner internals and storage.
  - Discretization error (target ≈ 0.1%) dominates round-off in fp64. PR 9 measures the attainable residual floor.
- **Preconditioners:**
  - Voxel meshes: geometric multigrid.
  - Unstructured meshes: p-multigrid (hex20/hex27 → hex8), then smoothed-aggregation AMG. AMG setup runs in pyamg on the host; the V-cycle runs on the GPU with a self-written block-ELL SpMV.
  - No `jax.experimental.sparse` in hot paths.
- **Shells:** solid-shell elements (3 DOFs per node), following Klinkel–Gruttmann–Wagner (1999):
  - ANS for transverse shear.
  - ANS for the thickness strain at the 4 corner points.
  - EAS built with the center Jacobian J₀.
  - Laminates with n Gauss points per ply.
- **Sensitivities:**
  - Formula: dλ/dp = vᵀ(∂K/∂p − λ∂M/∂p)v / vᵀMv, computed via `jax.grad` of the element energies with v held fixed.
  - Valid only for simple eigenvalues with a converged v (backward error ≤ 1e-8).
  - Deflation and masking leave the formula unchanged.
  - Repeated eigenvalues are handled with the bound formulation; KS aggregation only over complete clusters.
- **Optimizer:** MMA implemented in JAX (~200 LOC), checked against Svanberg's test problems. No OC, because eigenvalue sensitivities change sign.
- **Dependencies:**
  - Core: `jax`, `numpy`, `scipy`, `meshio`, `pyamg`, `pymetis`.
  - SLEPc is an optional extra installed via conda, excluded from the default checks.
  - meshio's `.inp` reader doesn't handle parts or assemblies; a small custom reader is added only if needed.

## Memory budget (single 80 GB GPU; the 64 GB Mac fits the 10M column, but slower)

| Item | 10M DOF unstructured hex20 (~0.8M elements) | 100M DOF voxel hex8 (~33M elements) |
|---|---|---|
| One fp64 vector | 0.08 GB | 0.8 GB |
| Operator | stored Ke ≈ 12 GB (symmetric) or on-the-fly | reference Ke + densities < 1 GB |
| LOBPCG: 9 blocks × m vectors | m = 30: ≈ 22 GB | m = 10: ≈ 72 GB, so it doesn't fit |
| LOBPCG reduced: 6 blocks (lumped M or recomputed products) | m = 30: ≈ 14 GB | m = 10: ≈ 48 GB; fp32 storage: ≈ 24 GB |
| Multigrid hierarchy | ≈ 0.15 × operator | ≈ 0.15 × fine-level vectors and stencils |

- At 100M DOFs, modes are computed in chunks of m ≈ 8–10. Converged modes are deflated and stored in fp32 or in host memory.
- The PR 5/6/9 estimators reproduce this table and are validated against measured peaks.

## Development environment

- **Development:** Apple M3 Max with 16 cores and 64 GB of unified memory, using the JAX CPU backend in fp64. Everything runs here, including the 10M-DOF unstructured case, which fits in memory but is slow.
- **No Mac GPU:** the GPU isn't usable for this project. Metal GPUs have no fp64 hardware, and the `jax-metal` plugin is experimental with incomplete op coverage.
- **Local checks:** `scripts/check.sh` runs lint plus `pytest -m "not slow and not gpu"`. Remote CI comes later.
- **GPU work:** `gpu` tests and the 100M benchmarks wait for a CUDA machine (TBD, e.g. a cloud H100); until then they are skipped. Benchmark JSON is committed to `benchmarks/results/`.
- **Data:**
  - Tiny test meshes live in `tests/data/`.
  - Large benchmark meshes are generated by scripts in `benchmarks/meshes/` and git-ignored.
- **Solver output:** solvers return an `info` dict (iterations, residual history, timings). No logging framework.

## Package layout

```
src/jaxmodal/
  config.py    dtype / x64 setup
  mesh/        data model, io, voxel generator, reordering, partition
  materials/   anisotropic C, Bond rotations, orientations
  elements/    quadrature, shape functions, hex8/20/27, solid-shell
  operators/   dofmap, assembled (oracle), matrix-free, voxel stencil, constraints (masking, rigid modes, deflation)
  linalg/      pcg, block-ELL spmv
  precond/     jacobi, chebyshev, gmg, pmg, amg
  eigen/       lobpcg, bands (counting, shift-invert Lanczos, MINRES), api (solve_modes), backends/{scipy, slepc}
  post/        normalization, MAC, effective mass, stresses, export
  opt/         sensitivities, mma, ffd, simp, filters
tests/  benchmarks/  examples/  docs/  scripts/
```

Each PR ships its docs (with LaTeX math) in `docs/`, listed below as **Docs**.

## M0: Foundations

### PR 0: Project scaffold
- Contents:
  - pyproject and lint config.
  - pytest markers `slow` and `gpu`.
  - `scripts/check.sh`.
  - `config.py` (an explicit x64 switch).
  - A benchmark harness that records time, peak memory and compile time.
  - LICENSE, version 0.x.
- Docs: `README.md`, `docs/conventions.md` (node ordering, Voigt order, dtype policy, units).
- Tests:
  - Dtype-leak test: an fp32 path stays fp32 with x64 enabled.
  - `check.sh` passes.

### PR 1: Mesh data model and parser
- Contents:
  - `Mesh` dataclass: coordinates, element blocks keyed by type, node and element sets, material and orientation IDs.
  - meshio readers for Abaqus `.inp` (flat), Gmsh and VTU, with node-order conversion for hex20/hex27. Gmsh, Abaqus and VTK orderings differ.
  - Structured voxel mesh generator.
- Docs: `docs/mesh.md`.
- Tests:
  - Read → write → read round-trips.
  - hex20/hex27 ordering matches hand-built references for every format.
  - hex8 corner Jacobians are positive. The full quadrature-point check is in PR 3.

### PR 2: Materials
- Contents:
  - Traceable 6×6 C constructors: isotropic, transversely isotropic, orthotropic (ν_ij/E_i = ν_ji/E_j convention), full anisotropic.
  - Positive-definiteness check.
  - Stress and strain Bond matrices for rotation into local axes, per element or quadrature point.
  - Density.
- Docs: `docs/materials.md`.
- Tests:
  - Bond rotation equals `einsum` tensor rotation.
  - Isotropic C is invariant under rotation.
  - C stays positive definite after rotation.
  - `jax.grad` with respect to angles and constants matches finite differences.

## M1: Elements and operators

### PR 3: Hex element library
- Contents:
  - Gauss quadrature.
  - hex8, hex20 (serendipity) and hex27 shape functions and derivatives.
  - B-matrix.
  - Batched Ke and Me (consistent, and lumped as described under Mass).
  - Jacobian check at quadrature points.
- Docs: `docs/elements.md`.
- Tests:
  - Partition of unity.
  - Linear completeness.
  - Exactly 6 zero eigenvalues of Ke, on distorted elements too.
  - Lumped masses all > 0, and total mass equals ρV per direction.
  - Matches a NumPy reference implementation.
  - `jax.grad` of uᵀKe(X)u matches finite differences (rel. err. < 1e-6).

### PR 4: DOF map, assembled reference operator, Dirichlet masking
- Contents:
  - Global DOF numbering (int32).
  - Per-group element-to-DOF arrays.
  - SciPy CSR K and M via COO.
  - Masking.
- Docs: `docs/operators.md`.
- Tests:
  - Distorted multi-element patch test: constant stress to 1e-10.
  - Clamped model: no spurious eigenvalues compared with SciPy on the reduced matrices.

### PR 5: Matrix-free unstructured operator
- Contents:
  - Two modes: stored Ke, and on-the-fly computation from precomputed invJ·detJ per quadrature point.
  - Chunked gather → kernel → `segment_sum`.
  - `diag_K`, `diag_M`.
  - Memory estimator.
  - Benchmark of scatter-add vs a deterministic inverse-map gather.
  - Node-local numbering, so partitioning is possible later.
- Docs: `docs/operators.md`.
- Tests:
  - Both modes match PR 4 to 1e-12 relative. Tests use tolerances, never bitwise equality.
  - The estimator is within 20% of the measured peak.

### PR 6: Voxel stencil operator
- Contents:
  - K matvec as 8 shifted corner slices → einsum with density-scaled reference Ke → 8 shifted adds. No gather, no convolution.
  - `apply_M` (consistent and lumped), `diag_K`, `diag_M`.
  - Density scaling of K and M.
  - Restriction: one material and orientation per phase.
- Docs: `docs/operators.md`.
- Tests:
  - Matches PR 5 on the same mesh.
  - GPU benchmark: a 100M-DOF matvec plus the LOBPCG workspace for m = 10 (fp32 storage) fits in 80 GB.

### PR 7: Rigid body modes and deflation
- Contents:
  - Analytic rigid body modes about the center of mass, M-orthonormalized.
  - Detection of disconnected parts.
  - Projector P = I − R Rᵀ M.
  - User-listed rigid subsets when Dirichlet sets exist.
- Docs: `docs/boundary_conditions.md`.
- Tests:
  - ‖KR‖ ≤ 1e-12·‖K‖‖R‖.
  - ‖RᵀMR − I‖ ≤ 1e-12.
  - Projected iterates stay M-orthogonal to R.

## M2: First modes end-to-end

### PR 8: Reference eigensolver and verification suite
- Contents:
  - SciPy `eigsh` shift-invert with a negative shift on assembled matrices. Its largest supported size is stated; above it, acceptance relies on backward-error residuals.
  - Verification problems:
    - NAFEMS FV52 (simply supported thick plate) and FV12 (free thin plate, exercises deflation). Reference values are checked against the NAFEMS source.
    - A rectangular block with sliding BCs (exact separable solution, including orthotropic).
    - RUS free anisotropic parallelepiped (Visscher et al. 1991).
    - A Lamb sphere for voxel meshes.
  - SLEPc backend is deferred and optional.
- Docs: `docs/eigensolvers.md`, `docs/verification.md`.
- Tests:
  - Eigenvalue error converges at O(h^{2p}) (hex8 h², hex20/hex27 h⁴) against a Richardson-extrapolated 3D reference; voxel meshes O(h).
  - Analytic and NAFEMS values agree within 1% at the stated meshes.
  - Consistent-mass eigenvalues converge from above.

### PR 9: Generalized LOBPCG and public API
- Contents:
  - LOBPCG as specified under Key design decisions.
  - Jacobi preconditioner, so the suite converges in reasonable time.
  - Chunked mode computation with fp32 or host storage of locked modes.
  - Public API: `solve_modes(mesh, materials, bcs, n_modes, ...) → (λ, V, info)`.
  - `examples/` script per milestone, also run as a smoke test.
- Docs: `docs/eigensolvers.md` (algorithm, memory model, precision policy and attainable residual floor).
- Tests:
  - Relative λ error ≤ 1e-8 vs PR 8.
  - Eigenspaces match by principal angles ≤ 1e-6. Clusters are compared by subspace, not by per-mode MAC.
  - Exact multiplicities on a square-section beam and a cube.
  - Free-free: ‖RᵀMX‖ ≤ 1e-10 and no rigid modes returned.
- **Milestone:** first end-to-end modes in pure JAX.

## M3: Preconditioning (core of the project)

### PR 10: PCG, simple preconditioners and smoothers
- Contents:
  - PCG (also needed by the Helmholtz filter and the fallback solver).
  - Jacobi.
  - 3×3 node-block Jacobi.
  - Chebyshev smoother with eigenvalue estimation.
  - All applied to A = K + σM.
- Docs: `docs/preconditioners.md`.
- Tests:
  - PCG matches `spsolve` to 1e-10.
  - Iteration-count study vs h, published to `benchmarks/results/`.

### PR 11: Geometric multigrid for voxel meshes
- Contents:
  - Trilinear transfer operators.
  - Galerkin coarse operators, stored assembled in block-ELL format on levels ≥ 1.
  - Chebyshev smoothing.
  - Dense Cholesky on the coarsest level.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Iterations grow ≤ 20% over 3 refinements.
  - Robustness on a real SIMP design at a 1e-9 density contrast (iteration counts reported). Robustness of Galerkin coarsening under contrast is a hypothesis this test checks.
  - GPU: a 100M-DOF voxel modal solve with m = 8–10 per chunk.

### PR 12: p-multigrid for unstructured meshes
- Contents: hex20/hex27 → hex8 transfer operators, with a Chebyshev smoother on the high-order level.
- Docs: `docs/preconditioners.md`.
- Tests: iteration counts on hex20 grow ≤ 20% over 3 refinements.

### PR 13: AMG on the hex8 level
- Contents:
  - pyamg smoothed-aggregation setup on the assembled hex8 A, with rigid body modes as the near-null-space.
  - Hierarchy exported to block-ELL on the GPU, with a V-cycle there.
  - Combined with PR 12.
  - Option to reuse aggregates and recompute only the values, for shape optimization.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Setup time and host-memory benchmarks.
  - GPU: a 10M-DOF hex20 modal solve on one GPU, with iteration counts reported.

## M3b: Frequency bands

### PR 21: Frequency-band eigensolver
- Contents:
  - Stochastic eigenvalue count in [a, b] via Chebyshev/KPM.
  - [0, f_max] mode with chunked LOBPCG, locked modes stored on the host.
  - Interior-band shift-invert Lanczos/Krylov–Schur with MINRES inner solves and spectrum slicing.
  - `solve_modes(..., band=(f1, f2))`.
- Docs: `docs/eigensolvers.md` (bands, counting, slicing).
- Tests:
  - Band eigenvalues match SciPy `eigsh(sigma=...)` to relative error ≤ 1e-8.
  - No missed modes compared with the LDLᵀ inertia count at reference sizes.
  - KPM count within ±1 of the true count on the verification suite.
  - Memory model for many modes (e.g. 1000 modes at 10M DOFs = 80 GB fp64, so they are kept on the host).

## M4: Solid-shell element

### PR 14: Solid-shell element (KGW 1999: ANS + EAS, 3 DOFs per node)
- Depends on PRs 3 and 8 for the element and frequency tests. The preconditioner check runs after PR 13.
- Contents:
  - Thickness direction from a user-specified stack direction or a topological sweep through connected layers; an error on ambiguity.
  - ANS and EAS as specified under Key design decisions.
  - Static condensation, with condensed Ke stored. This adds to the PR 5 memory estimator.
  - Laminates with ply angles in the local shell frame.
  - Mass from the displacement interpolation.
  - Simply supported BCs on solids are defined in the docs.
- Docs: `docs/elements.md` (solid-shell).
- Tests:
  - Patch test.
  - Exactly 6 zero eigenvalues after condensation.
  - Thin-plate frequencies without locking up to L/t = 1000.
  - Curved shell (clamped cylinder frequencies) and distorted meshes.
  - Laminates within 1% of CLPT/Leissa values.
  - `jax.grad` matches finite differences.
  - After PR 13: AMG iteration counts vs L/t reported. A fallback is listed if they degrade.

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
  - A Pallas kernel only if the matvec reaches < 50% of the bandwidth roofline (fp32 first).
- Docs: `docs/performance.md`.
- Tests: speedups on the 10M unstructured and 100M voxel benchmarks are recorded in `benchmarks/results/`.

### PR 17: Partitioning and multi-GPU
- Status: an epic, to be split before starting.
- Contents:
  - METIS partitions padded to uniform shard sizes.
  - `shard_map` operator with `ppermute` halos.
  - `psum` Gram matrices in LOBPCG.
  - Distributed GMG first.
- Docs: `docs/parallel.md`.
- Tests: results match single-GPU runs; weak-scaling plot.

## M6: Optimization

Working assumptions, still open: topology optimization at ≤ 10M voxel elements, clamped, densities as the only design variables. The user will decide later. Earlier PRs keep the alternatives open: free-free and material or fiber-angle design variables are already differentiable through Principles 4–5.

### PR 18: Eigenvalue sensitivities and MMA
- Contents:
  - Sensitivity formula and its preconditions, as under Key design decisions.
  - On-the-fly operators as the differentiable path; stored Ke is rebuilt each design iteration.
  - Cluster detection.
  - Bound formulation (max β s.t. λ_j ≥ β, Du & Olhoff 2007).
  - KS over complete clusters only: compute n + 2 modes and assert a gap at the cutoff.
  - MMA in JAX.
- Docs: `docs/optimization.md`.
- Tests:
  - Central finite differences with a step-size sweep: rel. err. < 1e-6 for coordinate, density and material parameters.
  - Directional finite-difference check on a double eigenvalue (symmetric cube or plate) using Seyranian subspace derivatives.
  - MMA solves Svanberg's test problems.

### PR 19: FFD shape parametrization
- Contents:
  - Bernstein lattice → node coordinates, in JAX.
  - Nodes outside the lattice and BC nodes are frozen.
  - Constraint: min det J at quadrature points ≥ ε · initial.
  - Example: maximize the fundamental frequency at constant mass, using MMA.
- Docs: `docs/optimization.md`.
- Tests:
  - Control-point gradients pass the PR 18 finite-difference criterion.
  - The example converges with no inverted elements.

### PR 20: Topology optimization
- Contents:
  - SIMP: E = E_min + ρ^p(E₀ − E_min).
  - Mass interpolation m = ρ for ρ > 0.1, else c₁ρ⁶ + c₂ρ⁷ (c₁ = 6e5, c₂ = −5e6), per Du & Olhoff 2007.
  - Filters: separable convolution on voxel meshes; Helmholtz PDE filter with PCG on unstructured meshes.
  - Heaviside projection with β-continuation.
  - Bound formulation solved with MMA.
- Docs: `docs/optimization.md`.
- Tests:
  - Reproduces the Du & Olhoff 2007 clamped-clamped beam with non-structural mass, extruded to thin 3D:
    - bimodal optimum (λ₁ ≈ λ₂);
    - ω₁ within 5% of the published value;
    - no localized modes, checked by MAC and an energy-fraction check.

## Dependencies

| PR | Depends on |
|---|---|
| 0 Scaffold | — |
| 1 Mesh | 0 |
| 2 Materials | 0 |
| 3 Hex elements | 1, 2 |
| 4 DOF map, assembled, masking | 3 |
| 5 Matrix-free unstructured | 4 |
| 6 Voxel stencil | 4 (compared with 5) |
| 7 Rigid modes, deflation | 4 |
| 8 Reference eigensolver | 4, 7 |
| 9 LOBPCG, API | 5, 7, 8 |
| 10 PCG, smoothers | 5, 6 |
| 11 Voxel GMG | 6, 9, 10 |
| 12 p-MG | 5, 9, 10 |
| 13 AMG | 12 |
| 14 Solid-shell | 3, 8 (element tests); 13 (preconditioner check) |
| 15 Postprocessor | 9 |
| 16 Performance | 11, 13 |
| 17 Multi-GPU | 16 |
| 18 Sensitivities, MMA | 9 |
| 19 FFD | 18 |
| 20 Topology opt | 11, 18 |
| 21 Frequency bands | 9, 10, 11 or 13 |

- **Critical path:** 0 → 1 → 3 → 4 → 5/7 → 8 → 9 gives first modes in pure JAX.
- **Biggest risk:** M3 (preconditioning). PR 11 gives an early 100M voxel result before the unstructured multigrid work.
- **Off the critical path:** PRs 2, 6, 14 (element part) and 15 can be developed alongside it.
- **Numbering:** PR numbers are identifiers, not a strict order; follow the dependency table.
