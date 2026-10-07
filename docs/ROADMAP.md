# Roadmap

Large-scale 3D linear modal analysis (Kv = λMv) in JAX: matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, linear anisotropic elasticity. Later: FFD shape optimization and topology optimization.

Targets: 10M DOFs on a single GPU (100M stretch, realistic on voxel meshes); multi-GPU later. FP64-capable GPU (A100/H100 class) assumed; dtype configurable.

## Principles

1. **Validate every fast path against a slow one.** Each new operator, preconditioner and eigensolver is tested against a simpler reference: SciPy for assembled operators and eigensolves, dense NumPy for single elements.
2. **One small interface.** The solver layer only sees `apply_K`, `apply_M`, `diag_K` and `apply_P`. Voxel and unstructured meshes are two implementations of that interface.
3. **Fixed array shapes, element groups by type.** Mesh preprocessing runs in NumPy; everything inside JAX works on fixed-shape arrays.
4. **Conventions settled in PR 1 and never changed:**
   - VTK node ordering internally, converted at import.
   - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear strains.
   - fp64 by default, with dtype configurable.

## Key design decisions

- **Free-free:** analytic rigid body modes (3 translations, 3 rotations), M-orthonormalized and deflated with the projector P = I − R Rᵀ M. No penalty, no shift. Dirichlet sets are handled by masking.
- **Shells:** solid-shell (continuum shell, 3 DOFs/node, ANS + EAS) instead of classical 6-DOF shells. This keeps one DOF type and one preconditioner family.
- **Eigensolver:** generalized LOBPCG in JAX. The SciPy `eigsh` shift-invert backend (optionally SLEPc) is the reference.
- **Preconditioners:** geometric multigrid for voxel meshes. For unstructured meshes, p-multigrid (hex20/27 → hex8) followed by AMG (setup with pyamg on the host, V-cycle on the GPU).
- **Sensitivities:** dλ/dp = vᵀ(∂K/∂p − λ∂M/∂p)v via `jax.grad` of the element energies with v held fixed. No differentiation through the eigensolver.

## Package layout

```
src/<pkg>/
  mesh/        data model, io, voxel generator, reordering, partition
  materials/   anisotropic C, orientations
  elements/    quadrature, shape functions, hex8/20/27, solid-shell
  operators/   dofmap, assembled (oracle), matrix-free, voxel stencil, constraints
  precond/     jacobi, chebyshev, gmg, pmg, amg
  eigen/       lobpcg, deflation, backends/{scipy, slepc}
  post/        normalization, MAC, effective mass, stresses, export
  opt/         sensitivities, ffd, simp, filters, optimizers
tests/  benchmarks/  examples/  docs/
```

Each PR ships its own docs (with LaTeX math) in `docs/`, listed below as **Docs**.

## M0: Foundations

### PR 0: Project scaffold
- Contents:
  - pyproject and lint config.
  - pytest with tests marked `slow`/`gpu`.
  - CPU CI.
  - fp64 enabled by default.
  - A benchmark harness that records time and peak memory.
- Docs: `README.md` (overview, install, test, links to `docs/`).
- Done when CI is green on an empty test suite and the benchmark runner prints its JSON output.

### PR 1: Mesh data model and parser
- Contents:
  - A `Mesh` dataclass: coordinates, element blocks keyed by type, node and element sets, material and orientation IDs.
  - Readers via meshio for Abaqus `.inp`, Gmsh and VTU, with node-order conversion for hex20/hex27.
  - Inverted-element detection via Jacobian sign checks.
  - A structured voxel mesh generator.
- Docs: `docs/mesh.md` (data model, node-ordering conventions, supported formats).
- Done when:
  - Read → write → read round-trips.
  - Hex20 ordering matches a hand-built reference.
  - Every element has det J > 0.

### PR 2: Materials
- Contents:
  - 6×6 C constructors: isotropic, transversely isotropic, orthotropic, full anisotropic.
  - Positive-definiteness check.
  - Rotation into local axes via the Bond matrix, per element or per quadrature point.
  - Density.
- Docs: `docs/materials.md` (Voigt convention, constitutive matrices, Bond rotation).
- Done when:
  - A rotated orthotropic C matches tensor rotation done with `einsum`.
  - C stays positive definite after rotation.

## M1: Elements and operators

### PR 3: Hex element library
- Contents:
  - Gauss quadrature.
  - Shape functions and derivatives for hex8, hex20 (serendipity) and hex27.
  - B-matrix.
  - Batched `Ke` and `Me` (consistent and lumped) via `vmap`.
- Docs: `docs/elements.md` (shape functions, quadrature rules, Ke/Me formulas).
- Done when:
  - Shape functions sum to 1 and reproduce linear fields (the patch test passes).
  - Ke has exactly 6 zero eigenvalues.
  - Mass sums to ρV.
  - Single-element results match a NumPy implementation.

### PR 4: DOF map and assembled reference operator
- Contents:
  - Global DOF numbering and element-to-DOF index arrays per element group.
  - Assembled SciPy CSR K and M via COO.
  - Dirichlet masking.
- This is the reference "oracle" for all later operator PRs.
- Docs: `docs/operators.md` (DOF map, assembly).
- Done when the 3D patch test passes and cantilever static deflection converges to beam theory.

### PR 5: Matrix-free unstructured operator
- Contents:
  - Gather → element kernel → `segment_sum`, in three modes: JAX BCSR, stored element-by-element Ke, and on-the-fly recomputation.
  - `diag_K` extraction.
  - A memory-budget estimator.
- Docs: `docs/operators.md` (operator modes, memory model).
- Done when:
  - All three modes match PR 4 to ~1e-12.
  - Matvec throughput is benchmarked against DOF count.

### PR 6: Voxel stencil operator
- Contents:
  - Matvec as a 27-point stencil using a single reference Ke.
  - Per-element density scaling.
  - No gather/scatter.
- Docs: `docs/operators.md` (stencil formulation).
- Done when:
  - It matches PR 5 on the same voxel mesh.
  - A 100M-DOF matvec fits in GPU memory (benchmark).

### PR 7: Boundary conditions and rigid body deflation
- Contents:
  - Analytic rigid body modes, M-orthonormalized.
  - Projector P = I − R Rᵀ M.
  - Dirichlet sets.
  - Penalty and MPC hooks, reserved for later coupling needs.
- Docs: `docs/boundary_conditions.md`.
- Done when:
  - K·R ≈ 0.
  - Rᵀ M R = I.
  - Projected iterates stay M-orthogonal to R.

## M2: First modes end-to-end

### PR 8: Reference eigensolver backends
- Contents:
  - SciPy `eigsh` shift-invert with a negative shift, on assembled matrices.
  - Optional SLEPc backend.
  - A verification suite: cantilever and free-free beam analytic frequencies, plus NAFEMS free-vibration benchmarks.
- Docs: `docs/eigensolvers.md` (shift-invert, verification problems).
- Done when frequencies converge at the expected rate under h- and p-refinement.

### PR 9: Generalized LOBPCG in JAX
- Contents:
  - M-orthonormalization and soft locking.
  - Hooks for deflation and preconditioning.
  - Block chunking to limit memory at 100M DOFs.
  - fp64 Rayleigh–Ritz, with mixed precision as an option.
- Docs: `docs/eigensolvers.md` (LOBPCG algorithm, memory model).
- Done when:
  - It matches PR 8 to tolerance on the PR 8 suite.
  - Free-free runs return no rigid modes.
- **Milestone:** first end-to-end modes in pure JAX.

## M3: Preconditioning (core of the project)

### PR 10: Simple preconditioners and smoothers
- Contents: Jacobi, 3×3 node-block Jacobi, Chebyshev polynomial smoother with eigenvalue estimation.
- Docs: `docs/preconditioners.md`.
- Done when an iteration-count study shows the expected h-dependence. These are the smoothers for every multigrid PR.

### PR 11: Geometric multigrid for voxel meshes
- Contents:
  - Trilinear transfer operators.
  - Galerkin coarse operators, which stay robust under the density contrast of topology optimization.
  - Chebyshev smoothing.
  - Dense direct solve on the coarsest level.
- Docs: `docs/preconditioners.md` (GMG).
- Done when:
  - Iteration counts stay roughly flat with refinement.
  - It remains robust at a 1e-9 density contrast.
  - A 100M-DOF voxel modal solve runs.

### PR 12: p-multigrid for unstructured meshes
- Contents: hex20 and hex27 → hex8 transfer operators, with a Chebyshev smoother on the high-order level.
- Docs: `docs/preconditioners.md` (p-MG).
- Done when convergence is p-robust on hex20 meshes.

### PR 13: AMG on the hex8 level
- Contents:
  - pyamg smoothed-aggregation setup on the host, with rigid body modes as the near-null-space.
  - Hierarchy exported to JAX BCSR.
  - V-cycle on the GPU.
  - Combined with PR 12 into one preconditioner.
- Docs: `docs/preconditioners.md` (AMG).
- Done when:
  - A 10M-DOF unstructured hex20 modal solve runs on one GPU.
  - Iteration counts are reported.

## M4: Solid-shell element

### PR 14: Solid-shell element (ANS + EAS, 3 DOFs per node)
- Contents:
  - Thickness-direction detection.
  - ANS treatment of transverse shear and thickness strains.
  - EAS with per-element static condensation. Condensed element matrices are stored, since recomputing them every matvec is expensive.
  - Layered anisotropic laminates.
- Docs: `docs/elements.md` (solid-shell: ANS, EAS, condensation, laminates).
- Done when:
  - Thin-plate frequencies converge without locking up to L/t ≈ 1000.
  - It matches an analytic simply supported plate.
  - It matches hex20 results on a thick plate.
  - The p-multigrid/AMG pipeline still converges.

## M5: Postprocessing and performance

### PR 15: Postprocessor
- Contents:
  - Mode normalization (mass or max-norm).
  - MAC matrix.
  - Participation factors and effective modal mass.
  - Strain and stress recovery per mode.
  - VTU/XDMF export of modes and a frequency table.
- Docs: `docs/postprocessing.md`.
- Done when:
  - Effective mass sums to total mass in the limit.
  - Modes are visible in ParaView.

### PR 16: Performance pass
- Contents:
  - Node and element reordering (RCM or Morton) for locality.
  - Kernel fusion and checks for JIT recompilation.
  - Validation of the fp32 preconditioner.
- Docs: `docs/performance.md` (benchmarks, precision policy).
- Done when the published 10M unstructured and 100M voxel benchmarks show measured speedups.

### PR 17: Partitioning and multi-GPU (can be deferred)
- Contents:
  - METIS partitioning and ghost-node maps.
  - A `shard_map` operator with halo exchange.
  - Distributed reductions in LOBPCG.
  - Multigrid distributed first for voxel meshes.
- Docs: `docs/parallel.md`.
- Done when results match single-GPU runs and a weak-scaling plot exists.

## M6: Optimization

### PR 18: Eigenvalue sensitivities
- Contents:
  - dλ/dp = vᵀ(∂K/∂p − λ∂M/∂p)v, computed via `jax.grad` of the element energies with v held fixed.
  - Detection of repeated eigenvalues.
  - KS aggregation over the lowest modes.
- Docs: `docs/optimization.md` (sensitivities, repeated eigenvalues).
- Done when gradients match finite differences for coordinate, density and material parameters.

### PR 19: FFD shape parametrization
- Contents:
  - A Bernstein-basis lattice mapping control points to node coordinates.
  - Mesh-quality guards.
  - An example: maximize the fundamental frequency at constant mass.
- Docs: `docs/optimization.md` (FFD).
- Done when:
  - Gradients with respect to control points pass the finite-difference check.
  - The example optimization converges.

### PR 20: Topology optimization
- Contents:
  - SIMP/RAMP interpolation, with the Pedersen mass interpolation to avoid spurious localized modes.
  - A Helmholtz density filter, reusing the matrix-free machinery.
  - Heaviside projection.
  - MMA or OC optimizer.
  - An example: maximize the first eigenvalue of a voxel design.
- Docs: `docs/optimization.md` (SIMP, filters, optimizers).
- Done when:
  - It reproduces a known frequency-maximization benchmark qualitatively.
  - Spurious localized modes don't appear.

## Dependencies

```
0 → 1 → 2 → 3 → 4 → 5 → 7 → 8 → 9 ──→ 10 → 11 (voxel GMG) ─┐
                     ↘ 6 ───────────↗      ↘ 12 → 13 (AMG) ──┼→ 16 → 17
                5 → 14 (solid-shell, parallel to M3)          │
                9 → 15 (postproc, any time after 9)           │
                9 + 5/6 → 18 → 19 (shape), 18 + 11 → 20 (topo)┘
```

- **Critical path:** PRs 0–9 give first modes in pure JAX.
- **Biggest risk:** M3. Start PR 10 as soon as PR 9 lands. PR 11 then gives an early 100M voxel result, before the unstructured multigrid work.
- **Parallel work:** PRs 14 and 15 can be developed alongside M3.
