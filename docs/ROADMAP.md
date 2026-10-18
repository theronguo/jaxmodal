# Roadmap

Large-scale 3D linear modal analysis (Kv = λMv) in JAX. The solver uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. FFD shape optimization and topology optimization come later.

**Targets:**
- Lowest modes and modes in frequency bands.
- Frequency accuracy of 0.1% (relative error in f, i.e. about 0.2% in λ) against a converged reference, for hex20/hex27 and solid-shell meshes.
- 10M DOFs, unstructured, on a single GPU.
- 100M DOFs on voxel meshes with a small block size (see the memory budget).
- Multi-GPU later (PR 19), multi-node included (user, 2026-10-17): first hardware is 2× A100, possibly on different nodes.
- Topology optimization as large as possible on multi-GPU, multi-node hardware (user, 2026-10-17).
- Checkpoint/restart for long runs under HPC time limits (user, 2026-10-17).
- An FP64-capable GPU (A100/H100 class) for production runs.
- Until a GPU is available, development and all tests run on an Apple M3 Max (64 GB, CPU backend, fp64).

Package name: `jaxmodal`. License: MIT.

Status: final. Review rounds 1–30 incorporated (round 30: all 5 reviewers satisfied). Implementation proceeds PR by PR, each with its own reviewed plan in `docs/plans/`. PR 0 merged: jax pinned at ≥ 0.11.2; the persistent cache and CPU donation work; the `pymetis` extra is deferred to PR 19 (minimum 2025.1.1). PR 1 merged: own flat-`.inp` reader, Gmsh 4.1 ASCII only, VTU is the only written format; host topology is int64 until PR 4's `check_int32` and cast. PR 2 merged: phases are (material, orientation) label pairs; `phases` (host) is split from the traceable `phase_table`, so PR 21 can differentiate it; PR 6 builds one reference Ke per phase. PR 3 merged: the element type is a static name and per-type tables are embedded constants (per-element arrays stay arguments); element DOF order is $3a + i$; `check_elements` is the host Jacobian/lumping check (PR 4 and PR 9 call it); `jacobian` gives PR 5's invJ and w·detJ. PR 4 merged: DOF $3n + i$; BCs are `{node_set: "xyz"}` (parsed only by `dof_mask`; PR 9 takes the same dict); masks are bool, True = free; `nodal_mass` is `(n_nodes,)`, isotropic (×3 per DOF); `assemble` returns symmetrized SciPy CSR with int32 indices (oracle, x64 only, ≤ ~10⁵ DOF on bulky meshes; thin meshes further, PR 16c); `box_mesh` (hex8/hex20/hex27) and `voxel_mesh` have face sets `xmin`…`zmax`; later PRs cast their own device index arrays through `check_int32`; `a_p` takes an explicit `s` for indefinite $K - \sigma M$. PR 5 merged: `unstructured(...)` returns a pytree operator whose methods `apply_K`, `apply_M`, `diag_K`, `diag_M`, `scaled` are the solver interface (PR 6's `stencil(...) -> Stencil`, with estimator `stencil_memory`, provides the same five); unused leaves are `None` (stored mode has no `C`: rebuild to change materials); Ke is packed upper-triangle and built on the device; the CPU default path is scatter (the apply is kernel-bound, scatter is a small part); gather windows do not depend on element order; `rcm`/`renumber` live in `mesh/`; the estimator returns storage/vectors/workspace/build/overhead (later PRs add `storage + workspace`, count `overhead` once). PR 6 merged: `stencil(...) -> Stencil` (module `operators/voxel.py`) has PR 5's interface (five methods plus `n_dof`); it gathers through int32 node/cell id grids with fill/drop sentinels (no full-grid vectors), scans slabs along the longest axis (≈ 64k cells), runs one 24×24 GEMM per phase (phase scan, workspace flat in P), and keeps `C`, `rho`, `spacing` as leaves; `reference_element` and the grids (`node`, `cell`, `axes`) are the inputs for PR 11 and PR 23; shared operator helpers live in `operators/common.py`. PR 7 merged: `parts(mesh)` (in `mesh/`, no JAX) finds face-connected parts and rejects edge, vertex and nonconforming contacts; `rigid_modes(mesh, free=None, max_parts=10)` returns dense host modes (per-part null space under BCs, constrained rows exactly 0); `m_orthonormalize(V, MV)` is CholQR2 with V donated (3 blocks peak), and `project(R, X, MX)` is the fp64 reference for PR 9's chunked projector (PR 9b keeps R as a separate fp64 block instead of the first basis columns); the rigid-mode setup runs before PR 9 allocates the solver blocks; BC/mask/A_P docs moved to `docs/boundary_conditions.md`. PR 8 is split into 8a/8b/8c. PR 8a merged: `reference_modes(K, M, n_modes, free, sigma, points)` (`eigen/`, SciPy) runs `eigsh` shift-invert at a negative shift (two-pass σ rule) and certifies every result by residuals, M-orthonormality and a sparse Sylvester count, raising instead of returning missed modes; with `points` it orders by geometric nested dissection (about 100k free DOF on bulky meshes); `count_below` (`linalg/`) is the dense `sytrf` inertia (≤ 20k free DOF); `restrict` gives the free blocks; the sliding-block oracle is `tests/exact.py`. PR 8b merged: in-test oracles `plate_modes` (Srinivas–Rao, 1D GLL Ritz in $z$ per $(m, n)$, truncation by a check), `visscher` (Legendre Ritz, parity split) and `mac` (M-weighted) in `tests/exact.py`; `SIMPLY_SUPPORTED` defines the face-wise simply supported BCs PR 16 reuses; degenerate sets pair by summed MAC after a 45° rotation; the hex8 Srinivas–Rao rate is taken on the single-ply orthotropic plate (the cross-ply is pre-asymptotic for hex8 at affordable sizes). PR 8c merged: `box_mesh(..., coords=...)` takes per-axis element boundaries (graded meshes, midside nodes at element midpoints); the σ rule's pass 2 runs at $-0.3\tilde\lambda_k$, replacing 8a's $-0.01$, whose rigid-mode amplification failed the residual check on a 59k-DOF free voxel sphere; FV52 uses $u_z = 0$ on the whole edge faces (the NAFEMS line support is singular); `lamb_sphere` (pole-free spheroidal form) is in `tests/exact.py`; the FV fine values and Lamb cluster-mean errors are recorded test constants. PR 9 is split into 9a (LOBPCG core, Jacobi, σ pre-estimate), 9b (chunks and locking, fp32 basis, 6-block mode, workspace estimator and memory model) and 9c (`solve_modes`, examples, Mac budgets); PR 10 needs none of PR 9, PR 12 needs 9a, PRs 11 and 13 need 9c and 9b's estimator, and PR 14 and M2 need 9b and 9c. PR 9a merged: `lobpcg(op, free, n_modes, tol, R, precond, X0, maxit, seed)` (`eigen/lobpcg.py`) runs one module-level jitted step that takes the operator, mask, preconditioner (a pytree callable), R, M R and the constants as arguments, so new data never recompiles; P starts as masked zeros (one compiled variant) and is kept M-orthogonal to X in coefficient space (Duersch), with a whole-block pivot-test fallback; outcomes are reported by `converged`/`stagnated` in `info` and never raise; the default Jacobi uses σ = 0 and `lobpcg` never runs `sigma_estimate` (σ changes iterations by < 3%; the pre-estimate is for PRs 11–13); the Rayleigh–Ritz mask value is 10·max|λ| + ‖K‖/‖M‖, so the solver needs no λ̂_max; the measured step memory is 16 blocks, not 9 + 2 (9b's estimator models it). PR 9b merged: chunked locking with a preallocated (fp64 or fp32) device basis and a chunked fp64-upcast projector; columns are re-orthonormalized at lock (two M-projections plus Cholesky; a single-pass basis drifted into span(B) after ~20 chunks); the guard error is raised only for chunks that will be locked; `low_memory=True` (11 blocks, not 6); `V` is a preallocated fp64 host array; `memory.lobpcg_memory` (step $= A\,b + f\,w$ envelope fit, host term +4 blocks when chunked) is 1–4% above the measured peaks at $m = 48$ and 5% above chunked (4–7% below at $m = 25$ on hex20 1M, and up to 7% below for many chunks with a zero-workspace operator). PR 9b merged: `lobpcg(..., *, chunk, guards, basis_dtype, low_memory)` locks converged chunks into a preallocated `(n, capacity)` basis (donated in-place insert, chunked fp64-upcast projector with one extra M apply per iteration, fp32 option with a $2u_{32}$ threshold term), extends a lock over a boundary cluster or raises the guard error, stops at a failed chunk, and returns `V` as NumPy fp64; `lobpcg_step_low` is the low-memory step (11 blocks, not 6); `memory.lobpcg_memory` is the estimator; six deviations are listed in the PR 9 note. PR 9c merged (milestone M2): `solve_modes` (`eigen/api.py`, exported from `jaxmodal.eigen`) runs `dof_mask` first, dispatches voxel meshes to `stencil` and others to `unstructured` (the builders run the host checks), deflates the rigid modes the BCs leave and returns them M-orthonormal in `info["rigid"]`, and adds `operator`, `n_rigid` and `frequencies` to `info`; `nodal_mass` is forwarded; the Mac budgets stay 10 and 15 min, now final, with the run specified and the throughput model in `docs/eigensolvers.md`; `examples/m0`–`m2` run as a smoke test. PR 10 merged: `pcg`/`minres` (`linalg/krylov.py`) solve `(n, k)` blocks column-wise with explicit-residual exits (PCG restarts from the explicit residual and reports `stagnated` after 5 restarts; MINRES checks every 10 iterations, `stagnated` after 5 checks without a new minimum) and `info` dicts of device arrays; `ap_operator` (`operators/constraints.py`) is the matrix-free $A_P$, default $s$ = mean free diag K + |σ| diag M, so `ap_operator(op, free, -sigma_b)` is the shift-invert operator with SPD preconditioners built at $+\sigma_b$ (a concrete σ < 0 raises in `jacobi`/`block_jacobi`); `jacobi` moved to `precond/`; `block_diag_K()` is the sixth operator method; `block_jacobi` inverts by batched Cholesky through the internal `_inverse_blocks(B, free)` that PRs 11 and 13 reuse for level blocks; `chebyshev(A, precond, lmax, degree=2, ratio=30)` applies the 1.1 safety factor inside, `lmax_estimate` is 20 seeded D⁻¹-weighted Lanczos steps; `block_ell(A, b, dtype, sliced, chunk)` is a row-chunked block-ELL (sliced into power-of-2 width buckets when plain padding > 1.5, e.g. hex20; constrained nodes are width-1 rows); the h-study (`benchmarks/pcg_study.py`) gives $h^{-1}$ growth for Jacobi, block Jacobi and Chebyshev 2. PR 11 is split into 11a (GMG core) and 11b (`solve_modes` dispatch, σ hint, fp32 V-cycle, memory estimator, Mac acceptance, deferred GPU checks). PR 11a merged: `gmg(op, free, sigma, degree=2, coarse_degree=4, ratio=30)` (`precond/gmg.py`) builds a voxel hierarchy by halving every axis with more than 2 elements (weights from carried physical positions), Galerkin levels of $A_P$ with nodal masses and the constrained diagonal folded per owning element, a chunked device setup loop that writes parent cells, smoother blocks and 27-slot block-ELL values in one sweep (+1.12 GB at 2M DOF), stopped-axis group-block Chebyshev smoothers (degree 2 fine, 4 coarse), deflation of the rigid modes of stiff islands (SIMP) and an fp64 dense coarsest Cholesky; coarse DOFs with a zero prolongation column are dropped instead of kept with diagonal s; at σ = 0 it refuses meshes it cannot prove nonsingular (zero-stiffness cells, rigid modes, meshes `rigid_modes` rejects); at σ = $\lambda_1$ (SciPy reference) iterations are 13 on boxes $8^3$–$64^3$, 87 on the SIMP field at its own (island-mode) $\lambda_1$ and 33 at the solid's, and grow on thin plates ($h^{-1}$: 71 at $64^2\times1$, 136 at $128^2\times1$), at spacing ratio 10 (106) and on other SIMP fields at their own $\lambda_1$ (109–135; unfiltered binary 452), documented limitations listed in `docs/open_items.md`; the V-cycle costs $5.2\,t_A$ at 2M DOF; `sigma_estimate` overestimates $\lambda_1$ by up to 3800× on thin, slender, anisotropic and SIMP cases, which 11b's σ pre-estimate must address. PR 11b merged (PR 11 complete; M3 continues with PRs 12–13): `solve_modes(..., precond="auto"|"gmg"|"jacobi", sigma=None, precond_dtype="float64"|"float32")` uses GMG on voxel meshes and Jacobi otherwise; without a hint σ comes from a GMG-refined pre-estimate (start at σ = 0 when the guard proves $A_P$ nonsingular, else at `sigma_estimate(maxit=10)`, then 1-mode GMG-LOBPCG rounds at tol 0.1, ≤ 20 iterations each), within 1.05 λ₁ on every tested case except an unfiltered binary field (390×, open item); `gmg` raises `SingularShiftError` (a `ValueError`) for its σ = 0 refusals; `gmg(..., dtype="float32")` casts the coarse levels, smoothers and transfers after the fp64 build (the fine operator, coarsest factor and deflation stay fp64), and the stencil and $A_P$ applies follow the vector's dtype; `memory.gmg_memory` (storage bound by shape arithmetic, V-cycle workspace, setup, extra step blocks `GMG_STEP`), `lobpcg_memory(..., precond_memory=)` and `solve_modes_memory`; the 2M-DOF Mac acceptance (10 modes, tol $10^{-6}$) runs in 61 s (fp64 V-cycle) and 51 s (fp32), 16 LOBPCG iterations, with warm peaks 6.23 / 4.86 GB against estimates 6.83 / 5.44 GB. PR 12 merged (M3 continues with PR 13): `pmg(op, free, sigma, degree=2, ratio=30)` (`precond/pmg.py`) returns a `GMG`: PR 11a's V-cycle now takes any pytree coarse solve (`GMG.coarse`: `Cholesky` for `gmg`, `HostLU` for `pmg`) and any transfer with a `restrict` method. The p-level transfer is a gather over per-node trilinear corner weights, restriction a gather over the inverse table; the hex8 Galerkin matrix is built per element on the device (`matfree.element_matrices`) and summed on the host into a SciPy CSR with a structural pattern, symmetrized on it (`_hex8_galerkin(A)`, PR 13's input; 1.4 s stored / 2.7 s otf and a 2.5 GB host peak at 1M hex20 DOF); the coarse solve is host `splu` through `pure_callback` (test-only; the jit caches hold the factor until `jax.clear_caches()`). σ > 0 is required (σ = 0 raises `SingularShiftError`). Measured: 14 PCG iterations on hex20 boxes $3^3$–$24^3$ and 13 on hex27, 14–19 on beam, free, curved, nodal-mass and `s_K`-contrast cases; LOBPCG 22–34× fewer iterations than Jacobi. Flat elements (aspect $a \ge 4$) degrade (open item, PR 13 / M4); σ = 0 refusal, COLAMD and curved-element rotations are logged in `docs/open_items.md`. PR 13 is split into 13a (AMG core) and 13b (production hex8 assembler, aggregate reuse, `solve_modes` dispatch and σ pre-estimate, memory estimator and ordering, setup benchmarks, SIMP study, Mac acceptance, optional 10M, deferred GPU check). PR 13a merged: `amg(op, free, sigma, points, degree=2, ratio=30, max_coarse=1000)` (`precond/amg.py`) returns one flat `GMG`: PR 12's p-level (none on all-hex8 meshes, whose matrix-free fine level takes the first AMG prolongator), smoothed-aggregation levels on the hex8 Galerkin matrix in block-ELL (3×3 blocks on the first level, scalar below), rectangular block-ELL transfers (gathers only), Chebyshev on every level and a dense Cholesky coarsest solve; near-null space: the masked rigid modes at the corner coordinates (`points` in operator node order); setup: an own loop over pyamg's primitives (deviation: `smoothed_aggregation_solver` keeps zero tentative columns; the first-level P equals pyamg's exactly) with the zero-column drop, a local fixed seed and a coarsest cap; `block_ell` takes rectangular matrices and bounds a bucket's scan by its width. Measured: 14 PCG iterations on hex20 boxes $3^3$–$24^3$ (15 at $3^3$), 13 on hex27 and hex8 (to 353k DOF), 14–20 on beam, free, curved, nodal and checkerboard; at 1M hex20 the V-cycle costs 5.4 $t_A$ with 0.48 GB of AMG device storage and 11.7 blocks of temporaries (13b re-derives the memory table's AMG, workspace and 10M rows); flat elements degrade further (open item, M4 / PR 18). PR 13b merged (PR 13 complete; milestone M3 done): `solve_modes(..., precond="auto")` uses AMG on unstructured meshes ("amg" on a voxel mesh raises; fp32 V-cycles stay GMG-only), with the σ pre-estimate refined by AMG rounds (`_precond_sigma`; σ = 0 start via `amg(..., nonsingular=True)`, derived from `rigid_modes`); PR 12's COO Galerkin path is replaced by a production assembler (`Hex8Pattern`: the $E^\top E$ node-block pattern, a fused device scatter into $(nnzb, 3, 3)$, host symmetrization, SciPy BSR; 0.72 s and 0.42 GB at 1M hex20); `amg(..., return_aggregates=True)` / `aggregates=` reuse the pattern and tentative prolongators (equal to a fresh setup for value changes under θ = 0, approximate for moved points; saves about 6–11% at $24^3$ and up); `memory.amg_memory` (storage, host build, `AMG_STEP`) feeds `solve_modes_memory`. Measured: the 1M hex20 Mac acceptance (20 modes) runs in 72 s with 19 LOBPCG iterations, warm peaks 7.46 / 7.43 GB against the 7.45 GB estimate; on SIMP fields AMG trails GMG (θ = 0 kept, open item); the optional Mac 10M run is skipped (estimate 52.2 GB above the 52 GB guard); 10M is the deferred GPU check. PR 13c merged (user request, inserted before PR 14): every option of the operator (`operator_options`: `mode`, `path`, `chunk`, `range_nodes` / `slab`), the preconditioner (`precond_options`: GMG `degree`, `coarse_degree`, `ratio`; AMG `degree`, `ratio`, `max_coarse`), the σ pre-estimate (`sigma_options`: `tol`, `maxit`, `rounds`, `drop`, `estimate_maxit`) and `lobpcg` (`X0`; `seed` now also seeds the pre-estimate) is reachable from `solve_modes` and `solve_modes_memory`, keys checked before any build and pinned to the builder signatures by a test; values are validated (`ratio > 1`, `max_coarse` an int ≥ 1, the `sigma_options` bounds); no default changes, and `docs/eigensolvers.md#options` justifies each; `_chebyshev` runs as a `lax.fori_loop`, so the smoother degree no longer grows the V-cycle temporaries (AMG 38.0 → 18.2 blocks at degree 6); `amg_memory(..., max_coarse, degree)` (coarse term floored at 1000, loop term for degree > 2) and `gmg_memory(..., degree)` model the options, and `GMG_STEP` is re-fitted (4.3 / 9.0 fp64 blocks). PR 14 is split into 14a (the count as a building block) and 14b (the $[0, f_{max}]$ mode, `verify`, cost function). PR 14a merged: `count(op, free, W, lam, lam_last, lam_next, *, R, precond, n_probes=50, seed, tau, degree, block, maxit)` (`eigen/counting.py`) estimates the modes missing from $W$ below $\lambda_c$ with a Jackson–Chebyshev step of $(K + \lambda_c M)^{-1}M$ (one `lax.while_loop` recurrence per probe block, early stop on a failed solve, NaN and `verifiable` False), probes $g \sim N(0, M)$ from exact element quadrature factors drawn per chunk/slab inside the scatter loop (`operators/sampling.py`; per-probe `fold_in` keys, so probes do not depend on the block), $W$ streamed in 64-column chunks (never all on the device) with a sketch check of $W^\top MW = I$; the degree rule (`count_degree`: $q^2 \ge 0.95$ at $\lambda_c(1 - g/2)$ and Weyl tail bias ≤ 0.05, $a = 3/2$) gives $d \approx 23/g$, capped at $10^5$; `memory.count_memory` / `probe_block` model the sampling, step (`COUNT_STEP`, `GMG_COUNT`, `AMG_COUNT`) and projector phases. Measured (`benchmarks/counting_study.py`): solve errors propagate as $|\Delta s| \le \tau s$ independent of $d$, so the roadmap's $\tau_c$ is about $10^5$× stricter than needed (kept; open item); a count at a typical cut of a 200-mode solve is predicted at 160–250× the solve (only cuts with $g \gtrsim 0.6$ cost a few solves, 1.2–3.6×), so 14b's `verify=None` will rarely count. PR 14b merged: `solve_modes(..., band=(0, f_max), verify, count_options)` (`n_modes` then optional, the expected count) runs chunked LOBPCG from 0 with every chunk lockable and stops at the first chunk whose converged Ritz values reach past $\lambda_{stop}(1 + 100t)$, clusters kept whole (`lobpcg._band_count`); the locked basis grows 1.5× (at least room for the next lock) at lock time past its capacity; an edge above the spectrum raises (15d: growth capped at $C_{max}$ from `memory_budget`, free-and-refill; LOBPCG raises only at the free-space cap, else a late error). The count cuts between the last returned and the first converged mode beyond (the band edge when none is returned), deflating all converged modes, with the fp64 multigrid at $\lambda_c$ (AMG reuses the solve's aggregates) and the probe block from `solve_modes_memory(band=...)` minus the host modes. `verify=None` skips on `MAX_DEGREE`, then on a free a-priori bound ($2\,n_\text{blocks}\,d\,(b/m)\,t_\text{it}$; measured 4.7–6.4× below the real step at $m = 48$), then calibrates two one-block counts (one run of `test_calibration`: 14.66 s predicted vs 14.65 s) and counts iff build + calibration + predicted count ≤ the solve. No strategy selection yet: LOBPCG from 0 is the only band strategy (15d: strategy done). PR 15 is split into 15a (one shift-invert slice), 15b (band-pass count, restarts) and 15c (slicing, strategy, `band=(f1, f2)`; the PR 15c plan split it again into 15c, the slicing engine, and 15d, the strategy and API). PR 15a merged: `si_slice(op, free, lo, hi, precond, *, capacity, sigma, R, W, tol, tau, block, maxit_inner, seed, verbose)` (`eigen/shift_invert.py`, internal; reached through `solve_modes(band=...)` since 15d) runs block shift-invert Lanczos with inner MINRES on `ap_operator(op, free, -sigma)` and an SPD multigrid at $+\sigma$, full M-reorthogonalization into a preallocated basis (W deflated, never returned), Rayleigh–Ritz on (K, M) (not the inexact T), per-pair thresholds $t_i = \max(tol, 10\tau_\text{eff}\delta_i, \text{floor})$ and $r_{acc} = \max t_i$ over the pairs and neighbours, the stop rule with clusters whole and a converged neighbour beyond each edge, one σ move when the first solve stagnates, `capacity_full` / `edge_unreached` for 15c, and V formed on the host in chunks; `memory.si_memory` is within 20% of measured peaks. Measured (`benchmarks/shift_invert_study.py`): MINRES iterations follow the depth, not δ or the mesh size; slices converge at 1M hex20 near mode 100 (1.7 h) and at 2M voxel near mode 6 (4.6 min); the floor at those sizes is $7.0 \times 10^{-14}$ / $3.3 \times 10^{-14}$; at depth 1000 a mid-gap solve takes about 14k iterations and may stagnate (open item for 15c). `minres` now returns its best iterate (the 2-norm residual is not monotone; the last iterate had drifted up to 40× above the minimum with σ on an eigenvalue). PR 15b merged: `band_count(op, free, V, lam, lo, hi, V_out, lam_out, ...)` (`eigen/counting.py`) counts the modes missing from an interior band with a Jackson band-pass (the difference of two steps, one recurrence at $\sigma = c_{hi}$) between counting edges midway to the nearest known out-of-band eigenvalues (`band_edges`); the degree rule checks both edges ($d\,g_\text{eff} \approx 23$, the lower edge costing $(1 + a)/(2\sqrt a)$ more, $a = c_{lo}/c_{hi}$) and adds the lower tail to the bias, with the Weyl model fitted to the band's modes and nearest neighbours; only the nearest out-of-band pair on each side is deflated (the rest sit inside the bias). The internal `si_checked` runs `si_slice` and the count (band-pass for $\lambda_{lo} > 0$, the step from 0) with up to 2 restarts (fresh seeds, the found modes deflated, the capacity reduced to stay within the first run's memory), reusing the slice's preconditioner; unverifiable counts and unconverged slices are flagged, not restarted. `count` takes a tuple $W$ and `verbose` (a per-step `jax.debug.callback`, at most one line per 10 s; also `count_options={"verbose": True}`). PR 15c merged: the internal `si_band` slices a band within a basis capacity per slice (deflation included): slices run lowest first, each deflating the stored modes of its window and the nearest known pair beyond each end; a full slice keeps the span si_slice's completeness rule verifies on its last Ritz values (a leaf without a run, counted with the parent's preconditioner) and requeues its remainders, cut at the largest gap ≥ 100 r_acc within each half of its converged run (a binary split on a side without a cut); clusters held by several runs are merged (duplicates by the projected M-norm per direction, Rayleigh–Ritz, the window widened once, the collapse flag). `si_slice` takes `lam_W` (known modes as converged candidates of its stop rule) and returns the last Ritz values (`info["ritz"]`); `si_checked` takes `verify`; MINRES no longer stagnates before its best residual drops below ‖b‖. The strategy choice, the cost model and `band=(f1, f2)` move to PR 15d. PR 15d merged: `solve_modes(..., band=(f1, f2), strategy, memory_budget, slice_options)`. The strategy is memory-driven, not by a time crossover: LOBPCG to the band was cheaper than a slice at every measured depth (hex20 $16^3$, AMG: 33× at depth 6 to 1260× at depth 1005; `benchmarks/strategy_study.py`). `"auto"` runs LOBPCG from 0 with its basis capped at $C_{max}$, the largest basis passing the fit test ($1.05\times$ the exact column arrays + $1.2\times$ the modelled terms ≤ the budget, per memory space; default the free GPU pool or $0.8\times$ RAM), on a chunk ladder (top chunk halved down to 5; the fastest reaching a known depth, else within 10% of the best reach); no chunk fitting gives slices from the start ("memory"), a capacity stop hands the whole band to `si_band` ("capacity", with a stagnation warning from 1000 locked modes). The growth frees and refills the basis from a preallocated host V; the returned V is compacted in place. Slices get a searched capacity (window and store reserved; the store $N_s = \max(n_\text{modes}$ or 200$, \lceil 1.25\,L\rceil)$), a budgeted count block and per-leaf `verify=None` (`counting._decide`); $\lambda_{lo} < \mu_1^{ub}(1 + \max(10^{-2}, 300\,tol))$ starts them from 0. The MINRES cost model $\alpha D(1 + D/D_0)$ is fitted per family (hex20 5.95 / 691, voxel 4.31 / 814; ratios 0.82–1.15) as evidence. `examples/m3b_bands.py`; PR 15 and milestone M3b are complete. PR 16a merged (plan `docs/plans/pr-16-solid-shell.md`, split 16a element / 16b operator and API / 16c accuracy suite): the `ssh8` solid shell (KGW 1999: Bathe–Dvorkin and Betsch–Stein ANS, 5-parameter EAS condensed by a 5×5 Cholesky, 2×2 in-plane × 2 Gauss points per ply), one-element kernels `shell_stiffness`, `shell_mass`, `shell_recover` and `RECOVER`; `Laminate`/`Ply` and `ply_table`; `solid_shell` (thickness by `direction` or a topological sweep); `assemble` on `ssh8`; `check_elements` through the thickness. MacNeal–Harder regular 0.988 / 0.972 / 0.979, parallelogram extension and out-of-plane 0.994 / 0.976; patch tests exact on parallelogram patches and for membrane and $\sigma_{33}$ states on distorted flat ones. PR 16b merged: the stored `ssh8` operator (packed condensed $K_e$ and $m_e$, 336 floats per element; otf refused), the exact probe factor $\sqrt{s_M}\,\mathrm{chol}(m_e)$, the build envelope (57.5 KB per element, fitted at 32 plies, after `shell_mass` scans its plies), `solve_modes` with Jacobi by default on `ssh8` (`"amg"` refused until PR 18; "auto (resolved to X)" in option errors), `.inp` SC8R, `examples/m4_shell.py` (232 Jacobi iterations at $L/t = 20$; `test_m4_reference` matches `reference_modes` to the PR 9 tolerances). PR 16c merged (PR 16 and milestone M4 complete; plan `docs/plans/pr-16c-accuracy.md`): `benchmarks/shell_accuracy.py` (parts `srao` / `cyl` / `skew`, 23 / 39 / 4 min) records `ssh8` against Srinivas–Rao (worst $5.8\cdot10^{-4}$; $k$ = 12 / 24 elements per ply at $L/t = 10$, 3 / 4 at 100), the clamped cylinder $R/t = 50$, $L/R = 2$ ($512\times96\times1$, worst $7.6\cdot10^{-4}$, 4 symmetry classes) and the 30° clamped skew plate $L/t = 20$ ($192^2\times1$, worst $7.4\cdot10^{-3}$) against graded hex27 references at $n_z = 3$ (errors $\le 1.2\cdot10^{-4}$ / $1.1\cdot10^{-3}$: Richardson capped at rate 4 ×1.25 with a one-level-back check and a crossing guard, plus a thickness term at the corner rate 1.42 extrapolated in $h$); skew modes are matched within their mirror class (two pairs swap between levels); no locking: bounded error ratio 1.37 ($L/t$ 1000 vs 10; hex8 168); `tests/test_shell_accuracy.py` gates the record (builder digests) and recomputes one key per code path. The PR 8 `eigsh` reaches 385k thin hex27 DOF (25.7 GB).

## Principles

1. **Validate every fast path against a slow one.** References are SciPy, dense NumPy, or analytic or published values. Every acceptance criterion is quantitative. "a equals b to tol" means ‖a − b‖_∞ ≤ tol·‖b‖_∞ (for pointwise fields such as stress, the max over all points); absolute tolerances are used only for dimensionless quantities (e.g. VᵀMV − I).
2. **One small interface.** Every discretization (unstructured, voxel) provides `apply_K`, `apply_M`, `diag_K`, `diag_M`, `block_diag_K` (the 3×3 diagonal node blocks of K, from PR 10) and the rigid body modes (the interface is complete after PR 7, and for voxel meshes after PRs 6 and 7; `block_diag_K` after PR 10). The rigid modes depend only on the mesh and the mask, so one mesh-level function, `rigid_modes(mesh, free)`, serves both discretizations. Each operator applies natively to blocks of shape `(n_dof, m)`, not via `vmap`. Solvers only see this interface plus `apply_P`.
3. **Fixed shapes; everything else is a `jit` argument.**
   - Topology preprocessing (connectivity, DOF maps, sets) runs in NumPy.
   - Every array is a pytree `jit` argument: int32 topology, geometry, densities, material parameters, nodal masses, stored Ke, multigrid/AMG matrices and rigid modes. Nothing is closed over.
   - Only shapes, chunk sizes, element types, block sizes, Chebyshev smoother degrees, multigrid level counts and mode/option flags (stored vs on-the-fly, lumped mass, scatter vs gather, LOBPCG step variant) are static. The counting-polynomial degree is a dynamic `fori_loop` bound; its Jackson/Chebyshev coefficients are computed in-loop from (k, d).
   - The one allowed closure exception is the test-only `pure_callback` coarse solve in PR 12, with an explicit `vmap_method` and fp64 `ShapeDtypeStruct` results.
   - Iterative solvers keep fixed shapes. Dropped or converged columns are masked, not removed.
4. **Extensible element types.**
   - Each element type is one registry entry: node count, shape functions, quadrature rule and reference faces.
   - Any per-format node permutation lives in `mesh/io`; none is needed today (meshio permutes Gmsh, Abaqus C3D20 = VTK).
   - Only hex types are implemented now; wedges and tets can be added later without touching operators or solvers.
5. **Differentiable from day one.**
   - Element kernels are pure JAX functions of `(X_e, C_e, ρ_e)`: energies and bilinear forms uᵀKe w and uᵀMe w.
   - Material constructors are traceable.
   - Sensitivities are a separate pass over the element kernels. Solvers and preconditioners are never differentiated.
6. **Memory-aware.**
   - Elements (and voxel slabs along the longest grid axis) are processed in fixed-size chunks with `scan`/`fori_loop`, the last chunk padded, accumulating into a donated output carried through the loop.
   - Indices are int32, with asserts that N_dof and every flattened index product used by an int32 gather/scatter (e.g. n_el × n_dof_e) stay below 2³¹. The asserts run in NumPy before casting.
   - The deflation projector (VᵀMX, V·c, and Π_V in counting) runs over fixed-size column chunks of the preallocated basis inside `fori_loop` (bound = current lock count) with `dynamic_slice`, upcasting one fp32 chunk at a time. All `dynamic_slice`/`dynamic_update_slice` start indices are `jnp.int32`, also under x64. MX is passed in, not recomputed. The basis capacity is a multiple of the chunk size and unused columns are zero, so `dynamic_slice` never clamps onto earlier columns. On the write side, `dynamic_update_slice` would also clamp, so the capacity is ≥ max lock count + insert width (rounded up to a multiple of the chunk size), and the host loop asserts count + width ≤ capacity before every insert (raising, or reallocating with one recompile). Unchunked mixed-precision products make XLA materialize a full fp64 copy of V.
   - Estimators include chunk temporaries and are checked against measured peaks.
7. **Conventions** live in `docs/conventions.md` and are fixed in PR 0:
   - VTK node ordering internally, converted at import.
   - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear strains.
   - fp64 by default, enabled through an explicit config call (no import side effects; PR 0 test: importing jaxmodal leaves `jax_enable_x64` unchanged).
   - Loops that are differentiated (operator chunk loops) use static trip counts (`scan`, or `fori_loop` with Python-int bounds); traced bounds appear only in solvers, which are never differentiated.
   - Index arrays passed to JAX are int32. Host-side NumPy topology is int64; it is cast through `check_int32` (PR 4) before going to the device.
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
  - Under Dirichlet sets, the remaining modes are R·null((I − m)R), computed via an SVD of (I − m)R with unit-normalized columns (rank tolerance σ_i ≤ 1e-8·σ_max, unit-independent), per part since (I − m)R is block-diagonal by part, then M-orthonormalized (CholQR2) and deflated with P = I − R Rᵀ M. R is dense, so more than `max_parts` parts (default 10, e.g. floating voxel islands) raise. It assumes s_K > 0 on every element (void elements would add zero modes that R does not contain).
  - The AMG near-null-space is m ⊙ R: all 6 modes per part with constrained rows zeroed. It is not the reduced deflation basis, which can be empty for clamped models.
- **Preconditioning:**
  - Preconditioners approximate A = K + σM with σ > 0 (default ≈ λ_target, estimated before the solve by the GMG-refined pre-estimate of PR 11b or given as a user hint), used as A_P. Exception: PR 9's default Jacobi uses σ = 0 (iterations change < 3%).
  - They must be fixed, linear and SPD: symmetric pre- and post-smoothing, no inner Krylov.
  - The Chebyshev λ_max estimate uses a safety factor of 1.1.
- **Eigensolver:** generalized LOBPCG following Hetmaniuk–Lehoucq / Duersch et al. 2018:
  - SVQB orthonormalization with a drop tolerance; dropped directions are masked.
  - In Rayleigh–Ritz, masked rows and columns are zeroed off the diagonal, with B = 1 and A = 10·max|λ| + ‖K‖/‖M‖ on the diagonal (by Courant–Fischer above every wanted Ritz value; no λ_max estimate needed). They never appear as spurious low Ritz values.
  - P is dropped (masked) when ill-conditioned.
  - Implicit product updates with periodic explicit recompute, selected by a traced flag (`lax.cond`) inside the step variants, so it adds no compiled variant.
  - The deflation projector is re-applied every iteration.
  - The locked buffer is preallocated, with a count. A Python outer loop calls at most two compiled step variants (the 9-block step, 16 blocks measured, and the low-memory step, 11 blocks measured); each step call donates the iteration state. Lock insertion into the preallocated basis is a jitted `dynamic_update_slice` with the basis donated (in place).
  - A low-memory step (the "6-block mode") keeps only X and P and recomputes every K and M product when needed: 6 applies per iteration against 2, and 11 blocks measured, not 6 (PR 9b; see the PR 9 note).
- **Convergence:**
  - Measured by the λ-scaled residual r = ‖Kx − λMx‖ / (|λ|‖Mx‖). This gives sin θ ≲ r·λ/gap and δλ/λ ≲ r²·λ/gap (heuristic, up to M conditioning; the rigorous form with the M⁻¹-norm residual is used in PR 9). The normwise backward error is only reported.
  - Default tol = 1e-6.
  - Accept r ≤ max(tol, c·ε·(‖K‖ + |λ|‖M‖)‖x‖ / (|λ|‖Mx‖)), with c = 10 and ‖K‖, ‖M‖ estimated by power iteration. For accurate eigenvectors it does not blow up as M becomes singular. A stagnation exit is flagged in `info`.
  - A cluster (for locking) is a set of eigenvalues whose relative gaps are < 100·tol. For shift-invert, tol here and in all slice-edge and band-edge rules is one band-wide r_acc = max over slices of max(tol, 10·τ_eff·δ_max), where each pair is accepted at r_i ≤ max(tol, 10·τ_eff·δ_i) and r_acc is the max of these thresholds over returned pairs and the edge neighbours used for counting; the cost model caps δ_max ≤ 1 when choosing slices (15d: per-pair thresholds hold, slices from 0 can exceed tol, `r_acc` is reported; enforcement → PR 18). The PR 9 test threshold of 1e-4 differs on purpose: it only decides whether to compare by subspace.
  - Block size m = nev + max(5, 0.2·nev).
  - Chunk and lock cutoffs must fall in a gap.
- **Precision:**
  - The attainable residual floor is about ε·κ, with κ = λ_max/λ_min: roughly 1e-8–1e-6 in fp64, fine for 0.1% targets but O(1) in fp32.
  - Hence fp32 never goes into X, R, P or the returned modes.
  - PR 9 measures the floor.
- **Frequency bands (PRs 14–15):**
  - **Strategy:** memory-driven (PR 15d; the roadmap's cost-model choice was replaced after measurement).
    - Chunked LOBPCG from 0 (then filtered to the band) is used while its deflation basis (every mode below the band's upper edge) fits the memory budget; otherwise shift-invert slices are used. The chunk comes from a ladder (the top chunk halved down to 5, the fastest whose basis cap reaches a known depth, else within 10% of the best reach).
    - Measured (hex20 $16^3$, AMG, Mac CPU): LOBPCG from 0 is cheaper than a slice at every depth to 1005 (33× at depth 6, 1260× at depth 1005), so there is no time crossover. On the Mac budget the basis cap is 2112–2720 columns at 1M hex20 and 768–1440 at 2M voxel (fp64; up to 3710 / 2010 with the memory levers); at 10M hex20 on 76 GB, 120 (fp64 AMG) to 260 (fp32 basis, low memory, on the fly), so the 700–1000 target needs PR 18's fp32 AMG. At 100M, bands need multi-GPU.
    - The stochastic count is controlled by `verify` (implemented in PR 14) for every strategy: `None` (default) enables it only when it is predicted to cost no more than the solve (PR 14b's rule, per leaf on slices); `True`/`False` force it. The predicted cost is always reported in `info`.
    - Near mode index 1000 (relative gap g ≈ 7e-4), shift-invert costs ~10³–10⁵ MINRES iterations per Lanczos step (worst case ∝ ln(1/τ)/δ, δ the relative distance from σ to the nearest eigenvalue; Krylov deflation of the few eigenvalues nearest σ moves it toward the low end; PR 15 calibrates this), and counting costs d ≈ 25/g ≈ 3.5e4 solves per probe block.
      - Measured (PR 15a, `docs/eigensolvers.md#measured-minres-cost-floor-and-slices`): the iterations grow roughly linearly with the depth of σ in the spectrum, the rate rising with it (hex20 with AMG: 5–6.7 per mode below depth 100, 7.7 / 8.5 / 9.9 / 14.3 at depths 100 / 201 / 503 / 1005; voxel with GMG 4–5.5), barely with δ (≤ 20% over four decades), and not with the mesh size (1M and 2M DOF match $16^3$ / $32^3$). At depth 1000 a mid-gap solve takes about 14k iterations and is marginal (of two runs, one stagnated and one converged, by round-off), about 4 h per slice step at 1M DOF on the Mac. 15d fits $\alpha D(1 + D/D_0)$ on depth (evidence only), warns after a capacity stop at ≥ 1000 locked modes (`stagnation_risk`); remedies → PR 18. The floor at the largest Mac sizes (depth 6) is $7.0 \times 10^{-14}$ (1M hex20) and $3.3 \times 10^{-14}$ (2M voxel).
  - **[0, f_max]:** chunked LOBPCG until λ > λ_c, with the cutoff λ_c = (2πf_max)².
  - **Band edges:** every cluster (relative gap < 100·tol) that reaches into the band is returned whole. The counting edge (and σ for the [0, f_max] count) is placed at the middle of the gap between the last returned mode and the first computed mode beyond it, and g is that gap. The same applies to f₁ and f₂; an edge at λ = 0 has no counting edge (the step polynomial already covers it). When no mode is returned, each counting edge sits midway between its band edge (2πf)² and the nearest converged neighbour outside the band, and g is the smaller of the edge gaps. A neighbour within 100·r_acc of a band edge is returned (it reaches into the band); if d still exceeds `MAX_DEGREE`, the count is flagged unverifiable. A band edge above the largest computed eigenvalue raises an error (15d: late errors only; the up-front spectrum bound → PR 18).
  - **Interior bands [f₁, f₂]:**
    - Block shift-invert Lanczos on (K − σM)⁻¹M with σ in the band, default block size b = 6 (≥ the largest expected multiplicity, so repeated eigenvalues are found), the same column masking as LOBPCG, full M-reorthogonalization, and a final Rayleigh–Ritz with true residuals on (K, M).
    - Inner MINRES with absolute-value preconditioning by the SPD multigrid of K + σM (Vecharynski–Knyazev 2013), with a fixed inner tolerance τ = 1e-8 on the explicitly computed 2-norm residual ‖r_in‖₂ ≤ τ‖Mq‖₂ (checked at exit). MINRES keeps iterating until the explicit 2-norm residual meets τ or stagnates (no new minimum of the explicit residual over 5 consecutive checks, one check every 10 iterations; a hard iteration cap sets an `info` flag; PR 15c: no stagnation while the best explicit residual is ≥ ‖b‖, a plateau before any progress, measured on a slice step). If it stagnates above τ, τ_eff is raised to the measured floor, carried into r_acc and flagged in `info`; PR 15 measures this floor. The constrained diagonal of K − σM is mean diag(K) + σ·mean diag(M) over free DOFs (positive). The attainable outer λ-scaled residual is then ≈ τ·δ_max, with δ_max = max over the slice of |λ − σ|/|λ|; acceptance uses r_acc = max(tol, 10·τ_eff·δ_max), with τ_eff = τ unless raised below.
    - Each slice's Lanczos run stops when every Ritz value in its interval meets the slice's running r_acc (from its current δ_max and τ_eff) and at least one converged Ritz value lies beyond each slice edge, in a gap ≥ 100·r_acc (PR 15c: a known deflated mode beyond the edge, at a gap ≥ 100 r_acc, also serves as the neighbour; an unknown mode between still appears as a nearer Ritz value that must converge). An edge at λ = 0 (a clamped [0, f_max] run on shift-invert) needs no neighbour beyond it, since no eigenvalue lies below 0 (free bodies are deflated first). After all slices finish, every edge is re-checked against the final band-wide r_acc; an edge that fails is widened or merged into the neighbouring slice. If (2πf₁)² is below the Rayleigh-quotient upper bound of λ₁ from the σ pre-estimate, the band is solved as [0, f₂] and filtered to [f₁, f₂], so a missing lower neighbour never blocks the stop (15d: with the margin max(1e-2, 300 tol), the bound the min of LOBPCG's first Ritz value and the pre-estimate, and the pre-estimate run on the slice paths). Block size b = 6 finds larger exact multiplicities only slowly (through rounding); the count catches a shortfall.
    - (15d: slice widths are capacity-limited and the counting degree enters through the per-leaf `verify=None`; see the PR 15c note below.) Slice widths come from the cost model (MINRES iterations ∝ ln(1/τ)/δ, with δ the relative distance from σ to the nearest eigenvalue, plus an outlier term for the few eigenvalues nearest σ; calibrated in PR 15). The cost model also counts deflation-basis memory (n_dof × number of vectors) and the predicted counting cost, and rejects strategies over the memory budget. Slice edges, like lock cutoffs, are placed in the largest relative gap (≥ 100·tol) within the cost model's allowed window, so each cluster belongs whole to one slice: the cost model sets provisional edges, each edge is snapped to the largest gap among converged Ritz values in its window, and only then is the stop rule evaluated. The cost model includes each slice's counting degree (same edge rule, g = that slice-edge gap). If τ_eff makes 100·r_acc exceed every gap in the window so that slices collapse into one, `info` is flagged. Before merging, each slice keeps only the Ritz pairs inside its own interval that meet r_acc. Remaining duplicates at slice edges are removed by M-orthogonalization, dropping a vector whose projected M-norm (M-normalized) is < 0.5 (a new vector has ≈ 1, a duplicate ≲ 2·r·λ/gap). The projected norm only detects duplicates; kept vectors are the original Ritz vectors. If vectors from both slices are kept for one cluster, Rayleigh–Ritz is run on the merged cluster; kept vectors must meet r_acc, otherwise Rayleigh–Ritz over that edge window is repeated and `info` is flagged. If the window has no gap ≥ 100·r_acc, it is widened or merged into the neighbouring slice.
      - PR 15c note: slice widths are limited by the capacity, not set by the cost model: the measured slice cost (about 1.8 Lanczos columns per mode plus about 84 per slice, a slice spending 50–85 before it converges anything; MINRES iterations set by the depth of σ) makes the widest slice that fits the cheapest. Edges come from full slices: the span si_slice's completeness rule verifies on the run's final Ritz values, cut at the largest gap ≥ 100·r_acc within each half of its converged run (the snapping rule), a binary split on a side without a cut. Leaves keep si_slice's widened inside sets; deflation windows holding the stored neighbours prevent most duplicates (a slice whose r_acc widens its window is rerun), and the merge (duplicates, Rayleigh–Ritz, re-check) is the backstop; across runs that do not deflate each other, V is M-orthogonal to residual level (the exact bound $(\|x_j\|\|r_i\| + \|x_i\|\|r_j\|)/|\lambda_i - \lambda_j|$, up to about $4\cdot10^{-9}$ measured) (`docs/open_items.md`, PR 15c).
    - Neither the Lanczos nor the counting steps need M⁻¹ beyond the one M-solve per probe. Consistent M costs nothing else extra there.
    - MINRES needs a preconditioner that is symmetric up to rounding (symmetric smoother and transfer pairs). fp32 V-cycles are used only inside LOBPCG and PCG, never inside MINRES (fp32 asymmetry would floor τ near 1e-6).
  - **Completeness count (stochastic, at scale):**
    - Estimate s = ‖Π_V q(T) Π_V y‖²_M, with:
      - T = (K + σM)⁻¹M with σ ≈ λ_c ([0, f_max]) or σ at the band's upper edge (interior band);
      - Π_V = I − VVᵀM, where V includes the computed modes and R.
    - Probes are processed in blocks whose size the PR 14 memory model sets and counts (recurrence, PCG and preconditioner workspace per probe). y = M_P⁻¹g with g ~ N(0, M) masked. g comes from element-wise quadrature factors of Me (B_e = Nᵀ·diag(√(w det J)), exact: B_e B_eᵀ = m_e; PR 14a) plus √m_i·ξ_i for nodal masses (lumped mass: √(diag M)·ξ), followed by one Jacobi-PCG solve per probe with the masked M_P (unit diagonal on constrained DOFs, where g vanishes), to the same tolerance τ_c as below.
    - q is a Chebyshev–Jackson step polynomial ([0, λ_c]) or band-pass polynomial (interior band, deflated by the modes found in the band and the converged out-of-band neighbours within the polynomial's transition width of each band edge).
    - The degree d is chosen numerically so that:
      - q² ≥ 0.95 on the in-band interval up to the last returned mode, [0, λ_c(1 − g/2)] (up to the band edge (2πf)² when none is returned); this sets d ≈ 25/g (measured in PR 14a: d ≈ 23/g);
      - the Jackson-tail bias Σ q(λ_i)² over the out-of-band spectrum is ≤ 0.05, estimated with a Weyl-type density N(λ) ∝ λ^a with a = 3/2 fixed (PR 14a: the 3D exponent is an upper bound on fits to computed modes, bulky 1.1, plates 0.4–0.5, beams 0.3–1.3, so it is stricter than a fitted a). (Bounding n_dof·max q² instead would need d ≈ 75/g or more.)
    - A missed mode inside the gap next to the cutoff is detected with lower probability.
    - Inner PCG tolerance τ_c ≲ 0.01/(d·√n_dof). PR 14 measures how solve errors actually propagate through the recurrence (worst case O(d²), not O(d)). Counting solves use the fp64 V-cycle. If τ_c is below the attainable PCG accuracy, `info` marks the count as unverifiable instead of passing it. At 10M DOFs near index 1000 (τ_c ≈ 1e-10) this is expected to happen often.
    - One missing mode makes the estimate behave like χ²ₙ/n. With 50 probes, P(missed mode undetected) ≈ 0.2%.
    - A slice passes when its estimate is < 0.5.
  - **Exact count:** at reference sizes (≤ 20k free DOF), from the inertia of the dense LAPACK `sytrf` (Bunch–Kaufman, the routine `scipy.linalg.ldl` calls, in place on one dense copy) of K − σM restricted to the free DOFs (the constrained diagonal of A_P could otherwise add spurious negative pivots).
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
- **Scatter vs gather:** PR 5 benchmarks scatter against an inverse-map gather on each backend, and the default may differ per backend. On the Mac CPU the apply is kernel-bound (the $K_e$ unpack and contraction), and the scatter is a small part of it, so scatter wins and is the CPU default (`docs/operators.md`); the GPU default is set by the deferred GPU check.
  - The gather is chunked by node range and works for any element order; RCM (or grid order) keeps the windows narrow.
  - Element contributions go into a padded, fixed-size element window per range, which is counted in the estimator.
- **Dependencies:**
  - Python 3.12.
  - Core: `jax` (pinned minimum version), `numpy`, `scipy`, `meshio`, `pyamg`, all with macOS arm64 wheels.
  - `pymetis` is an optional extra, added in PR 19 (its only user) with minimum 2025.1.1, the oldest cp312 macOS arm64 wheel.
  - Versions are locked with `uv.lock`.
  - No SLEPc, CHOLMOD or pypardiso.
  - meshio's `.inp` reader is unusable (see `docs/plans/pr-01-mesh.md`); PR 1 adds a flat-`.inp` reader. Gmsh input must be 4.1 ASCII.

## Memory budget

All sizes are fp64; GB = 10⁹ bytes.

| Item | 1M hex20 (Mac test) | 2M voxel (Mac test) | 10M hex20 | 100M voxel |
|---|---|---|---|---|
| One vector | 0.008 GB | 0.016 GB | 0.08 GB | 0.8 GB |
| Operator | Ke ≈ 1.2 GB (symmetric) | < 0.1 GB (≈ 8 MB id grids and phase; 35 MB with s_K/s_M/nodal, PR 6 estimator) | Ke ≈ 12.2 GB (symmetric) or on-the-fly | ≈ 0.4 GB (id grids, phase; s_K/s_M/nodal ≈ 1.3 GB are in the 3–4 GB allowance below) |
| LOBPCG, 9-block step (16 blocks × m, measured) | m = 25: ≈ 3.2 GB | m = 15: ≈ 3.8 GB | m = 30: ≈ 38 GB | m = 10: ≈ 128 GB (multi-GPU only) |
| LOBPCG, low-memory step (11 blocks × m; 9 with the voxel stencil on CPU) | — | — | m = 30: ≈ 26 GB | m = 8: ≈ 58–70 GB |
| Rayleigh–Ritz update and deflation-projector temporaries | included in the LOBPCG rows | included | included | included |
| Preconditioner workspace (GMG ≈ 4–5 blocks × m; AMG ≈ 6 blocks at m = 25, ≈ 11 with the low-memory step, `AMG_STEP`) | ≈ 1.4 GB | ≈ 1 GB | ≈ 14 GB (m = 25) | m = 8: ≈ 26 GB (fp32: 13 GB) |
| Multigrid | AMG ≈ 0.73 GB (device, `amg_memory`; 0.48 GB measured); host build ≈ 1.5 GB | ≈ 0.2 GB | AMG ≈ 7.1 GB; host build ≈ 15 GB (extrapolated) | ≈ 9.9 GB incl. int32 indices (fp32: ≈ 5.2 GB) |
| Gather path (if chosen) | ≤ 2 vectors × m (the $R$ cap): ≤ 0.4 GB at m = 25 | — | ≤ 4.8 GB at m = 30 | — |
| Locked modes / deflation basis | small | small | n × 0.08 GB (fp32: half) | n × 0.8 GB (fp32: half) |
| Max locked modes (fp32 basis; low-memory step, fp32 preconditioner and multigrid) | > 2000 (15d, under the fit test: 2720 fp64 at chunk 5, 3710 with the levers) | > 2000 (15d: 1440 fp64, 2010 with fp32 V-cycle and basis, low memory) | ≈ 700 (stored Ke) – 1000 (on the fly), with fp32 AMG from PR 18 (15d, fp64 AMG: 120 stored – 260 on the fly with the levers) | none on one GPU unless the GPU step holds ≤ 7.5 blocks (see below) |
| Shift-invert slice capacity (15d, fit test, store 200) | 624 | 624 | 96 (on the fly; none stored) | — |

The LOBPCG rows use PR 9b's measured steps: 16 blocks for the 9-block step (new X and P coexist with the old blocks and W) and 11 for the low-memory step (XLA copies the donated X and P early), both including the Rayleigh–Ritz update and deflation-projector temporaries. `memory.lobpcg_memory` reproduces them (1–4% above the measured peak at 1.08M DOF, one chunk, $m = 48$; 4–7% below at $m = 25$ on hex20, 4% above on the 2M voxel at $m = 15$; for a chunked 300k-DOF solve between 5% above and 7% below, varying between runs).

- **Mac (64 GB, unified memory):**
  - Both test columns fit easily.
  - 10M is an optional, tight benchmark. Order the work as: AMG setup → free the assembled A → allocate LOBPCG.
  - An `np.memmap` is used only if the locked modes exceed RAM (→ PR 18).
  - Band solves are sized by the fit test (PR 15d): 1.05 × the exact column arrays + 1.2 × the modelled terms ≤ `memory_budget` (default 0.8 × RAM; the rest holds the OS and the macOS malloc cache).
- **80 GB GPU:**
  - GPU runs set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (the JAX default is 0.75). `XLA_PYTHON_CLIENT_PREALLOCATE=false` is not used, because it fragments memory. The usable pool is then ≈ 76 GB; the CUDA context and library workspaces sit in the remaining 4 GB. The estimators budget against the usable pool, not 80 GB. This is conservative: an "80 GB" card has ≈ 85·10⁹ bytes.
  - 10M hex20 with AMG fits with 20 modes ($m = 25$, stored Ke, 9-block step): `solve_modes_memory` gives 71.5 GB in total (device about 70 GB, the fp64 modes live on the host) against the 76 GB pool (thin); 30 modes (92.6 GB) do not. Fallback: on-the-fly Ke (about 60 GB). The AMG setup needs about 15 GB of host RAM (extrapolated).
  - 100M on one GPU is not expected to fit (PR 9b). At m = 8 (nev = 3 per chunk) a block is 6.4 GB. The low-memory step measured 11 blocks with the hex20 operator and 9.0 blocks with the voxel stencil on CPU (5.4M DOF): 57.6 GB, plus 12.8 for the fp32 preconditioner workspace, 5.2 for the fp32 multigrid and ≈ 3–4 GB for the operator, mask, diagonals, s_K/s_M and nodal masses, ≈ 79–80 GB against the ≈ 76 GB pool, before any deflation basis. A single-GPU 100M solve fits only if the step holds ≤ 7.5 blocks on the GPU (XLA:GPU's buffer assignment may differ); otherwise 100M needs multi-GPU (PR 19). Free bodies add R and M R in fp64 (9.6 GB for 6 rigid modes), so they need multi-GPU either way.
  - Larger blocks at 100M need multi-GPU (PR 19).
- The PR 5/6/9/11/13/14/15/16/18 estimators reproduce this table and are validated against measured peaks where the estimate is ≥ 2 GB.

## Development environment

- **Development:** Apple M3 Max with 16 cores and 64 GB of unified memory, using the JAX CPU backend in fp64. The acceptance tests run at 1M (hex20) and 2M (voxel) DOFs. 10M is an optional benchmark.
- **No Mac GPU:** the GPU isn't usable for this project. Metal GPUs have no fp64 hardware, and the `jax-metal` plugin is experimental with incomplete op coverage.
- **Local checks:** `scripts/check.sh` runs lint plus the fast suite, `pytest -m "not slow and not gpu"`, which must finish in under 5 minutes on the M3. Remote CI comes later.
- **Merge gate:** `pytest -m "not gpu"`, slow tests included.
- **GPU checks:** GPU acceptance criteria go into the "Deferred GPU checks" list and are run in PR 24 whenever a CUDA machine becomes available. They don't gate other milestones.
- **Mac acceptance budgets:** final (PR 9c): 10 min for PR 11's 2M voxel solve and 15 min for PR 13's 1M hex20 solve, with the run specified as tol $10^{-6}$, the 9-block step, an fp64 basis, and wall time including setup, the σ pre-estimate and compilation. The throughput model behind them is in `docs/eigensolvers.md` (Mac acceptance budgets); PR 11 or 13 revises its budget with that model if its measured LOBPCG iterations exceed the assumed 150 or its V-cycle costs more than assumed.
- **Benchmarks:**
  - Each measurement runs in a fresh subprocess and reports the peak minus baseline (macOS: `peak_bytes`, see below).
    - `memory_stats()` is unavailable on CPU.
    - `ru_maxrss` is in bytes on macOS and KiB on Linux; the harness branches by platform. GPU runs use the `memory_stats()` peak.
    - On macOS every measurement records `peak_bytes` = max(interval `phys_footprint` delta, `ru_maxrss` delta) as the primary number: the interval peak is read after `proc_reset_footprint_interval` (a private libSystem symbol), with `MallocLargeCache=0` (undocumented) in the child. Footprint counts compressed pages, which RSS can undercount; `ru_maxrss` catches reused small malloc blocks, which footprint misses. Details in `docs/conventions.md`.
  - JSON results go to `benchmarks/results/` and record:
    - jax and jaxlib versions;
    - STREAM triad bandwidth, both jitted and a small multithreaded C triad using pthreads (Apple clang has no OpenMP by default), skipped and recorded as skipped when no C compiler is found;
    - XLA:CPU intra-op thread pool size (unless pinned) and P/E-core setup;
    - measured fp64 peak;
    - time and peak memory; compile time is recorded by the PRs that measure it (5, 18).
- **Compilation cache:**
  - The persistent cache is enabled in tests, with `jax_persistent_cache_min_compile_time_secs = 0` and `jax_persistent_cache_min_entry_size_bytes = -1` (no size restriction). It stores entries on CPU with jax 0.11.2 (PR 0). PR 0's cache test fails rather than skips if a later jax stops storing entries; the remedy is removing the `enable_cache` call in `tests/conftest.py`. Nothing depends on the cache for correctness.
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
  mesh/        data model, io, voxel and box generators, reordering, partition
  materials/   anisotropic C, Bond rotations, orientations
  elements/    registry, quadrature, shape functions, hex8/20/27, solid-shell
  operators/   dofmap, assembled (oracle), matrix-free, voxel stencil, constraints (masking, A_P, rigid modes, deflation), sampling (probes with covariance M from quadrature factors of Me)
  linalg/      pcg, minres, block-ELL spmv, dense inertia
  precond/     block jacobi, chebyshev, gmg, pmg, amg
  eigen/       lobpcg, counting, shift-invert lanczos, api (solve_modes), reference (scipy)
  post/        normalization, MAC, effective mass, stresses, export
  opt/         sensitivities, mma, ffd, simp, filters
tests/  benchmarks/  examples/  docs/  scripts/
```

Each PR ships its docs (with LaTeX math) in `docs/`, listed below as **Docs**. Every PR that changes the public API or installation also updates `README.md`. PRs are numbered in execution order; the dependency table at the end is authoritative.

## M0: Foundations

### PR 0: Project scaffold
- Contents:
  - pyproject, `uv.lock` and lint config.
  - pytest markers `slow` and `gpu`.
  - `scripts/check.sh`.
  - `config.py` (an explicit x64 switch).
  - Persistent compilation cache.
  - A benchmark harness (subprocess peak memory and RSS, STREAM triad, fp64 peak, versions).
  - LICENSE (MIT), version 0.x.
- Docs: `README.md`, `docs/conventions.md`.
- Tests:
  - An fp32 path stays fp32 with x64 enabled.
  - Importing jaxmodal leaves `jax_enable_x64` unchanged.
  - Index arrays from `argsort`, `nonzero` and `arange` are cast back to int32.
  - The harness reports a stable peak and RSS delta for a known allocation.
  - Buffer donation works on CPU: there is no "donated buffers were not usable" warning, and `x.is_deleted()` is true for the donated input. If the pinned jaxlib doesn't donate on CPU, the minimum version is raised; failing that, the estimators budget without donation.
  - The persistent cache stores entries on CPU (it does on jax 0.11.2); the test fails rather than skips if it stops, and the remedy is removing the `enable_cache` call in conftest.

### PR 1: Mesh data model and parser
- Contents:
  - `Mesh` dataclass: coordinates, element blocks keyed by type, node and element sets, material and orientation labels.
  - meshio readers for Gmsh (4.1 ASCII) and VTU, plus an own flat-`.inp` reader; VTU writer.
  - Node orderings: VTK internally; only meshio's Gmsh permutation is needed (Abaqus C3D20 = VTK).
  - Structured voxel mesh generator, with an occupancy mask that drops inactive elements and nodes (e.g. a voxelized sphere). The `Mesh` carries grid shape, spacing, origin and grid↔mesh node/element maps for voxel dispatch.
- Docs: `docs/mesh.md`.
- Tests:
  - VTU read → write → read round trip.
  - Orderings match hand-built references for every format.
  - hex8 corner Jacobians are positive.
  - Voxelized sphere: element and node counts and volume match a NumPy count; no orphan nodes.

### PR 2: Materials
- Contents:
  - Traceable 6×6 C constructors: isotropic, transversely isotropic, orthotropic (ν_ij/E_i = ν_ji/E_j convention), full anisotropic.
  - Positive-definiteness check, relative: λ_min(C) > 1e-12·λ_max(C).
  - Stress and strain Bond matrices.
  - Orientations: local material axes per orientation label, from Euler angles or two direction vectors (test: both give the same rotation for equivalent input); per-element R fields through the batched functions.
  - Phases: mesh labels → (index, pairs) on the host; a traceable per-phase table of rotated C and ρ.
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
  - The lumped-mass and Jacobian raises live in the host check `check_elements(mesh, mass)`; the traceable kernels never raise.

### PR 4: DOF map, assembled reference operator, masking
- Contents:
  - Global DOF numbering (int32).
  - Per-group element-to-DOF arrays.
  - SciPy CSR K and M via COO; M consistent or lumped (diagonal from the PR 3 lumped Me). Assembly calls `check_elements(mesh, mass)` first.
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
  - Two modes: stored Ke, and on-the-fly computation from precomputed invJ and w·detJ per quadrature point (10 floats, from PR 3's `jacobian`).
  - Optional per-element stiffness and mass scaling arrays s_K and s_M, so density design on unstructured meshes stays possible; add s_K, s_M to PR 4's `assemble` (the oracle).
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
  - K matvec over slabs along the longest grid axis: `scan` + `dynamic_slice` of int32 node/cell id grids with a 1-layer halo; per slab, gather the nodes (fill), 8 shifted corner slices → one corner-DOF-major GEMM per phase with the reference Ke → shifted adds → scatter-add (drop) into the output carried through the scan (measured layout: `docs/operators.md`).
  - Interface: `stencil(...) -> Stencil` with PR 5's methods (`apply_K`, `apply_M`, `diag_K`, `diag_M`, `scaled`); estimator `stencil_memory`.
  - Per-element scaling arrays s_K and s_M, as in PR 5.
  - Occupancy-masked voxel meshes: inactive elements contribute zero; PR 1's grid↔mesh node map handles dropped nodes, and PR 6 checks it against the geometry.
  - `apply_M` (consistent and lumped), `diag_K`, `diag_M`, nodal masses.
  - Memory estimator, including slab temporaries.
  - Restriction: one reference Ke per phase, a (material, orientation) pair (PR 2).
- Docs: `docs/operators.md`.
- Tests:
  - `apply_K`, `apply_M` (consistent and lumped), `diag_K`, `diag_M` and nodal masses match PR 5 on the same mesh, for a full block and for the PR 1 voxelized sphere.
  - A two-phase voxel mesh (two materials, or one material in two orientations; labels set with `dataclasses.replace(mesh, material=..., orientation=...)`) matches PR 5.
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
Split into three parts with unchanged acceptance criteria: **8a** the reference eigensolver, the dense inertia and the sliding block; **8b** Visscher Ritz, Srinivas–Rao, simply supported BCs and MAC pairing; **8c** NAFEMS FV52/FV12 and the Lamb sphere. PR 9 strictly needs only 8a; the dependency table and the critical path are unchanged.
- Contents:
  - SciPy `eigsh` shift-invert (negative shift) on assembled matrices, with its largest supported size stated.
  - Sizes: **reference size** means ≤ 20k free DOF (`MAX_DENSE`), where dense `eigh`, M⁻¹ and the dense inertia are affordable (PR 9's dense r_M and inertia comparisons, PR 14's exact count). **The PR 8 `eigsh`** (`reference_modes`) alone reaches about 100k free DOF on bulky meshes with nested dissection (routine peak about 7 GB and 2 min at 107k with rigid modes; about 35k without; further on plates, e.g. 385k thin hex27 DOF at 25.7 GB in PR 16c), see `docs/eigensolvers.md`.
  - Dense inertia via LAPACK `sytrf` (≤ 20k free DOF) (test: on a clamped and a free model, the `sytrf` count of K − σM over free DOFs equals the dense `eigh` count below σ, for σ mid-gap between several consecutive eigenvalue pairs).
  - In-test oracles:
    - A rectangular block with sliding BCs (exact separable solution, including orthotropic with material axes aligned to the block axes); also run with lumped mass, with fitted rate ≥ 1.8 (hex8, hex27 row-sum, hex20 HRZ).
    - Rayleigh–Ritz (Visscher et al. 1991) for a free anisotropic parallelepiped, here with a Legendre basis, dense `eigh`, converged when successive degrees change every compared eigenvalue (modes 7 to the highest compared mode, sorted) by a relative amount < 1e-6.
    - The Srinivas–Rao exact 3D solution for thick orthotropic and cross-ply simply supported plates.
  - Published benchmarks:
    - NAFEMS FV52 (simply supported thick plate). The docs state where u_z = 0 is applied; modes 1–3 are in-plane rigid modes, which exercises the automatic partial rigid modes.
    - NAFEMS FV12 (free thin plate), run with hex20/hex27 only, because hex8 locks at L/t = 200. The in-plane refinement is stated (20×20 hex20 is ≈ 0.5% above converged for modes 12–13, the worst; free-edge layers slow convergence). The measured values are recorded in `docs/verification.md`.
    - The Lamb sphere on voxel meshes.
- Docs:
  - `docs/eigensolvers.md`.
  - `docs/verification.md`, with exact setups, sources and reference values. It also defines simply supported BCs on solids face-wise, over the whole edge face through the thickness (used again in PR 16).
- Tests:
  - **0.1% target:** sliding block, Visscher Ritz and Srinivas–Rao within 0.1% (relative error in f) at stated hex20/hex27 meshes.
  - **NAFEMS:** a mode is gated against its target (1%) only when a fine hex27 run is within 0.5% of it. The fine hex27 run is refinement-converged (edge-graded in-plane mesh; the last refinement changes f by < 0.1%), with its mesh stated in `docs/verification.md`. Otherwise it is gated against the fine hex27 run (1%) and the gap to the target is recorded in `docs/verification.md` (e.g. FV12 modes 10–13 sit ≈ 1% below the quoted targets). For FV52, no u_z = 0 placement reproduces mode 4 (the converged 3D value with u_z = 0 on the edge faces is 3.7% below the target; the NAFEMS bottom-edge line support does not converge); edge faces are used, other modes gated as the rule says.
  - **Convergence rates:**
    - Measured against the exact sliding-block and Srinivas–Rao solutions on affine meshes. Element faces conform to ply interfaces. Fitted rate ≥ 1.8 for hex8 (sliding block and thick Srinivas–Rao, L/t = 10, only, because of locking) and ≥ 3.6 for hex20/hex27, over ≥ 3 levels (finest error ≥ 100× the `eigsh` floor ≈ ε·κ) in the asymptotic range. Richardson extrapolation is used only to estimate errors, never to fit rates.
    - Voxel Lamb sphere, against the exact solution for the sphere of equal (voxelized) volume, over ≥ 5 levels fixed in the docs: the max relative error over the two lowest 5-fold clusters (selected by cluster, not by index; torsional and spheroidal modes are told apart by their radial kinetic-energy fraction, ≈ 0 for torsional) drops by ≥ 2× from the coarsest to the finest level (staircase errors oscillate, so the least-squares rate is reported only), with split clusters compared by their mean.
  - **Mode matching:** free bodies (Visscher, Lamb, FV12) are compared from mode 7 on, FV52 from mode 4 on. Modes are paired with Srinivas–Rao by M-weighted MAC (defined in PR 17, computed in-test here) and by the (m, n, family) labels; degenerate sets (analytic relative gap < 1e-2, e.g. (m, n)/(n, m) on a square plate) are paired by subspace (summed MAC over the set).
  - **Upper bound:** consistent-mass eigenvalues converge from above on nested meshes of parallelepiped elements, with exact geometry and conforming essential BCs; checked as λ_k(fine) ≤ λ_k(coarse) + 10·ε·λ̂_max, from mode 7 for free bodies and mode 4 for FV52 (not for the voxel Lamb sphere, whose geometry is not nested).

### PR 9: Generalized LOBPCG and public API
- Contents:
  - LOBPCG as under Key design decisions, including fixed shapes, the 6-block mode and floor-aware acceptance.
  - Jacobi on A_P, with the σ pre-estimate (a few Jacobi-LOBPCG iterations or a user hint; test: the pre-estimate is within 2× of λ₁; the user-hint test is in PR 11).
  - Chunked mode computation; the deflation basis is optionally fp32, returned modes are fp64.
  - LOBPCG workspace estimator.
  - Public API: `solve_modes(mesh, materials, bcs, n_modes, orientations=None, ...) → (λ, V, info)`. It calls `check_materials` and `check_elements` before tracing.
  - `examples/` script per milestone, also run as a smoke test.
  - Fixes the Mac acceptance budgets.
- Docs: `docs/eigensolvers.md` (algorithm, convergence theory, memory model, measured floor).
- Tests, at reference sizes against PR 8, run at tol = 1e-10 with an fp64 deflation basis:
  - Relative λ error ≤ max(1e-8, 10·ε·λ̂_max/λ), with λ̂_max the λ_max from SciPy `eigsh(which="LA")` on the free blocks (tests only; dimensionless, so independent of units; used with this meaning below), for modes with relative gap ≥ 1e-2 (clusters below that gap are checked by subspace, as below).
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
  - The compiled step function compiles at most twice per solve (counted with `jax.monitoring` compile events filtered by the step's `fun_name`, cache disabled) (the σ pre-estimate is a separate Jacobi-only run, counted separately).
- **Note (split and deviations).** PR 9 is split into 9a (LOBPCG core, merged), 9b (merged: chunks, locking, fp32 basis, low-memory step, estimator) and 9c (merged: `solve_modes`, examples, Mac budgets); see `docs/plans/pr-09a-lobpcg.md`, `docs/plans/pr-09b-chunks.md` and `docs/plans/pr-09c-solve-modes.md`. PR 9b deviates in six points:
  1. The low-memory ("6-block") step holds 11 blocks with a real operator (7 with a diagonal one; 9 with the voxel stencil on CPU), not 6: its state is (X, P) and every product is recomputed, but an apply inside the step costs about 4 blocks. The Memory budget rows, the 10M basis limit and the 100M paragraph are updated accordingly.
  2. The rigid modes stay a separate fp64 block with M R precomputed, not the first basis columns: projecting against the locked basis costs one extra M apply per iteration (storing M V would double the basis), and single-chunk free-body solves pay nothing.
  3. A chunk that does not converge (no-progress or `maxit` exit) ends the solve; the locked modes plus that chunk's wanted columns are returned with `converged` False. A floor-accepted chunk counts as converged.
  4. `V` is returned as a NumPy fp64 array (host) in every solve.
  5. The cluster rule compares the relative gap $(\lambda_{j+1} - \lambda_j)/|\lambda_j|$ with $100\,t_j$ ($t_j \ge$ tol is the acceptance threshold), evaluated once the wanted residuals are $\le \sqrt{t}$; at a failed exit the boundary is re-checked with the guards' Ritz-value uncertainty $\min(10r^2, 10^{-3})$, so a split cluster gives the guard error rather than a silent failure. Only chunks that will be locked raise the guard error; the final chunk (every single-chunk solve) returns as in 9a.
  6. With an fp32 basis the chunks after the first accept at $\max(\mathrm{tol}, 2u_{32}) \approx 1.2\cdot10^{-7}$: the rounded locked modes leave a residual floor near $10^{-8}$.
- **Milestone:** first end-to-end modes in pure JAX (reached with PR 9c; `examples/m0_foundations.py`, `m1_operators.py` and `m2_modes.py` cover M0–M2, and each later milestone adds its own example script to `examples/`, run by `tests/test_examples.py`).

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
- Split: 11a is the GMG core (transfers, Galerkin levels, smoothers, V-cycle, singular guard; tests: Galerkin, h-independence, SIMP, single layer, free body, odd axis in PCG, LOBPCG with GMG against `reference_modes`); 11b is the `solve_modes` dispatch, σ hint, fp32 V-cycle, memory estimator, Mac acceptance and the deferred GPU checks (tests: σ pre-estimate and hint, `solve_modes` odd-axis, fp32, Mac acceptance). Plans: `docs/plans/pr-11a-gmg.md`, `docs/plans/pr-11b-gmg-solve.md`.
- Contents:
  - Trilinear transfer operators, with semi-coarsening (an axis stops coarsening at ≤ 2 elements). An axis with an odd element count gets one coarse cell spanning a single fine element; transfers use the coarse nodal positions.
  - `solve_modes` dispatches voxel meshes to this operator and GMG, including the σ pre-estimate.
  - Galerkin coarse operators of A_P on levels ≥ 1, built element-wise as Pᵀ(Σ_e m_e⊙(s_K,e·Ke + σ·s_M,e·Me)⊙m_e)P on the nested voxel hierarchy (Me consistent or lumped, matching the fine operator), plus σ·Pᵀ(m⊙m_nodal⊙m)P for nodal masses and Pᵀ s·(I − diag m)P for the constrained DOFs. Coarse DOFs with a zero prolongation column (occupancy-masked meshes) are dropped (11a; they would be decoupled rows), so the coarsest Cholesky stays SPD. The fine operator is never assembled. Stored in block-ELL.
  - Symmetric Chebyshev smoothing over stopped-axis group blocks (point smoothing fails on semi-coarsened plates).
  - Dense Cholesky on the coarsest level.
  - fp32 option for the V-cycle (levels, smoother workspace and transfers in fp32, applied symmetrically; the coarsest Cholesky stays fp64), modelled by the memory estimator.
  - Memory estimator.
- Docs: `docs/preconditioners.md`.
- Tests:
  - The σ pre-estimate is within 2× of λ₁, and a user hint skips it (moved from PR 9). 11b: `sigma_estimate` is up to 3800× above λ₁ on thin plates, slender beams, anisotropic grids and the SIMP field (`benchmarks/results/gmg_study.json`), so 11b uses a GMG-refined pre-estimate instead (within 1.05 λ₁ on the tested cases; an unfiltered binary field is a documented miss, `docs/open_items.md`).
  - Iterations grow ≤ 20% over 3 refinements.
  - `solve_modes` on a voxel mesh with an odd element count along one axis matches the PR 8 `eigsh` (PR 9 tolerances).
  - On a small occupancy-masked, clamped voxel mesh with nodal masses and an odd element count along one axis, each coarse level equals Pᵀ A_P P from the PR 4 SciPy matrices (coarse DOFs with zero P columns dropped) to 1e-12.
  - LOBPCG with the fp32 GMG reaches the same tolerance as with fp64 GMG at a reference size, with ≤ 20% more iterations.
  - A checked-in synthetic SIMP-like density field (ρ ∈ [1e-3, 1], p = 3) converges to 1e-8 within 100 PCG iterations (11a, blocking per user decision: at the field's own $\lambda_1$, an island mode of the $\rho^3$/$\rho$ interpolation, 87 with island deflation; 33 at the solid box's $\lambda_1$; other seeds 109–135 and an unfiltered field 452 are documented limitations, see `docs/open_items.md`).
  - A single-layer mesh (n_z = 1) converges to 1e-8 within 100 PCG iterations (11a: $32^2\times1$ at σ = $\lambda_1$, 37; thin plates grow like $h^{-1}$, 136 at $128^2\times1$, a documented limitation for PR 13 and M4).
  - A free-free voxel body (σ > 0) converges to 1e-8 within 100 PCG iterations; the coarsest-level Cholesky succeeds.
  - Mac acceptance: a 2M-DOF voxel modal solve, 10 modes, tol $10^{-6}$, 9-block step, fp64 basis, wall time (setup, σ pre-estimate and compilation included) ≤ 10 min, peak within 20% of the PR 6 + PR 9 + PR 11 estimators, with fp64 and with fp32 V-cycles.
  - Deferred GPU checks: the low-memory LOBPCG step with the 100M voxel stencil at m = 8 holds ≤ 7.5 blocks (expected to fail: 9.0 on CPU, PR 9b), so a 100M matvec plus the LOBPCG and preconditioner workspace fits in the usable pool (PR 9 and PR 11 estimators); a 100M voxel modal solve (on one GPU only if that holds, else after PR 19); GMG restriction (scatter-adds) against prolongation time on the GPU.

### PR 12: p-multigrid for unstructured meshes
- Contents:
  - hex20/hex27 → hex8 transfer operators, with symmetric Chebyshev smoothing on the high-order level.
  - The hex8 coarse operator is the Galerkin product Pᵀ A_P P (also the input to PR 13 AMG). It is solved exactly by SciPy `splu` via `pure_callback`. This is test-only and is replaced by AMG in PR 13.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Iterations on hex20 grow ≤ 20% over 3 refinements.
  - Prolongation (hex20→hex8 and hex27→hex8) reproduces a trilinear field exactly (1e-12); Pᵀ A_P P matches a SciPy triple product to 1e-12.

### PR 13: AMG on the hex8 level
- Split (PR 13a plan, `docs/plans/pr-13a-amg.md`): 13a is the AMG core (setup, device export, the straddling-boundary and h-independence tests); 13b is the production assembler, aggregate reuse, `solve_modes` dispatch with the σ pre-estimate (σ = 0 included), the memory estimator and ordering, the setup benchmarks, the SIMP study, the `solve_modes` tests, the Mac acceptance, the optional 10M benchmark and the deferred GPU check. Deviation: the setup is an own loop over pyamg's primitives, not `smoothed_aggregation_solver`, because the latter keeps zero tentative columns (the first-level prolongator equals pyamg's). 13b note: PR 13a measured 0.48 GB of AMG device storage and 11.7 fine blocks of V-cycle temporaries at 1M hex20, so the memory table's Multigrid row (≈ 0.4 GB), Preconditioner workspace row (≈ 4 blocks) and the "80 GB GPU: 10M fits" line are re-derived by 13b's estimator (`docs/open_items.md`).
- Contents:
  - pyamg smoothed-aggregation setup on the assembled hex8 Galerkin A_P from PR 12, with the masked rigid modes as the near-null-space.
  - Hierarchy exported to block-ELL, with a V-cycle on the device. Zero tentative-prolongator columns (aggregates of fully constrained DOFs) are dropped, so every coarse matrix stays SPD and the coarsest dense Cholesky succeeds.
  - Combined with PR 12; pure hex8 meshes skip p-MG and run AMG on the assembled A_P (solid-shell meshes do the same from PR 18). `solve_modes` dispatches unstructured meshes to this preconditioner.
  - Memory ordering: setup → free the assembled A → allocate LOBPCG.
  - PR 4's `assemble` is the oracle only (≈ 120 GB at 10M hex8 DOF); the production hex8 A_P needs its own chunked assembler into a precomputed CSR pattern (≈ 1 GB per matrix per 1M hex8 DOF). Done in 13b: a fused device scatter into a precomputed 3×3 block pattern (BSR), 1.48 s and 1.42 GB process peak at 1M hex8 DOF.
  - Option to reuse aggregates and recompute only the values (test: on a perturbed-density model, PCG iterations within 20% of a fresh setup).
  - Memory estimator.
- Docs: `docs/preconditioners.md`.
- Tests:
  - Setup time and host-memory benchmarks (report-only).
  - A clamped hex20 model where aggregates straddle the Dirichlet boundary sets up and the coarsest Cholesky succeeds.
  - PCG iterations on hex20 (p-MG + AMG) grow ≤ 20% over 3 refinements.
  - `solve_modes` on a hex20 and a pure hex8 mesh matches the PR 8 `eigsh` (PR 9 tolerances).
  - Mac acceptance: a 1M-DOF hex20 modal solve, 20 modes, tol $10^{-6}$, 9-block step, fp64 basis, wall time (setup, σ pre-estimate and compilation included) ≤ 15 min, peak within 20% of the PR 5 + PR 9 + PR 13 estimators (PR 13 covers the p-MG workspace).
  - Optional Mac benchmark: 10M DOFs.
  - Deferred GPU check: 10M DOFs on one GPU.
  - 13b: the optional Mac 10M run (`amg_study.py --large`, 10 modes) is skipped: its estimate (52.2 GB) is above the 52 GB guard.

### PR 13c: Solver options (user request, after PR 13b)
- Request (2026-10-13): "Generally make the default settings a robust choice for most problems. But also expose all solver options to the user in some way."
- Contents:
  - `solve_modes(..., X0, operator_options, precond_options, sigma_options)`; keys checked against the builders' options before any build; values validated; `solve_modes_memory(..., operator_options, precond_options)`.
  - The Chebyshev smoother as a `lax.fori_loop` (memory flat in the degree), `amg_memory` / `gmg_memory` terms for the options.
  - The Options table in `docs/eigensolvers.md`: every option, its default and why it is the robust choice.
- Tests: the defaults unchanged (call snapshot), every key forwarded, options against `reference_modes`, the pre-estimate options on scripted probes, validation, memory composition and bounds, the option sets pinned to the builder signatures, the loop against the unrolled iteration.
- PRs 14 and 15 follow the same pattern: their new options (`band`, `verify`, probe and slicing settings) are reachable from `solve_modes`, validated, and listed in the Options table (done: `count_options`, `slice_options`, `strategy`, `memory_budget`).

## M3b: Frequency bands

### PR 14: Eigenvalue counting and [0, f_max]
- Split (PR 14a plan, `docs/plans/pr-14a-counting.md`): **14a** is the count as a building block (`jaxmodal.eigen.count`, probe sampling, step polynomial and degree rule, τ_c and the unverifiable flag, the probe-block memory model `count_memory` / `probe_block`, the counting benchmark); **14b** is `solve_modes(..., band=(0, f_max), verify, count_options)`, the [0, f_max] LOBPCG mode and edge placement, the cost function, the probe block from the solve's estimate and the fp64 multigrid at λ_c. Tests are tagged [14a] / [14b].
- Contents:
  - Deflated stochastic count with M-weighted probes and a Chebyshev–Jackson step polynomial (50 probes, fixed seed list).
  - Element-wise Me quadrature factors for probe sampling (exact factors $B_e = N^\top\mathrm{diag}(\sqrt{w\det J})$ instead of Cholesky; PR 14a).
  - Probe-block memory model (recurrence, PCG and preconditioner workspace and projector temporaries (chunk + block) per probe), used to set the probe block size.
  - `verify=None/True/False`, with the predicted counting cost in `info`.
  - The cost function and LOBPCG-from-0 selection; PR 15 adds the shift-invert branch and slicing (done in 15c / 15d). (PR 14b implements the counting cost for `verify=None`; with one strategy there is nothing to select, so the selection moves to PR 15.)
  - [0, f_max] mode reusing PR 9 chunking, stopping at λ > λ_c.
  - `solve_modes(..., band=(0, f_max))`.
- Docs: `docs/eigensolvers.md` (estimator, variance, degree and cost).
- Tests:
  - [14a] With one mode deliberately left out of V (not the one next to the cutoff), the count detects it for ≥ 97 of a fixed list of 100 seeds. The test is deterministic.
  - [14a] With V complete, the estimate (including Jackson bias) is < 0.5.
  - [14a] A κ ≥ 1e7 case, sized so the count is verifiable (asserted): with V complete, the estimate is < 0.5.
  - [14a] A case with τ_c forced below the measured PCG floor sets the unverifiable flag; the measured error propagation vs d is recorded.
  - [14b] `verify=True/False` force the count on/off; `verify=None` follows the predicted cost.
  - [14a] Element-wise Me quadrature factors: F_e F_eᵀ = Me to 1e-12.
  - [14a] The probe-block memory model is within 20% of the measured peak at a size where the estimate is ≥ 2 GB.
  - [14a] Counting cost (polynomial degree, wall time) vs mode index is reported.
  - [14b] No modes are missed compared with the dense inertia at reference sizes.
  - [14b] (0, f_max) with f_max below the first elastic frequency returns 0 modes and a count < 0.5; λ_c set inside a double eigenvalue returns both copies.
  - [14a] The probe covariance Σ_e F_e F_eᵀ assembled (plus nodal masses, masked) equals the masked SciPy M to 1e-12.

### PR 15: Interior bands
- Split (PR 15a plan, `docs/plans/pr-15a-shift-invert.md`): **15a** is the block shift-invert Lanczos solver for one slice (`eigen/shift_invert.py`: inner MINRES rules, τ_eff, r_acc, the slice stop rule), the measured MINRES cost and floor (`benchmarks/shift_invert_study.py`) and the slice memory model `si_memory`; **15b** is the band-pass count (two-edge polynomial, neighbour deflation) and the slice restarts; **15c** is slicing with the cost model (widths, edge snapping, capacity splits, edge re-check, duplicate removal, slice collapse), the strategy choice and `solve_modes(..., band=(f1, f2))` (the cost model ended as evidence only: the strategy is memory-driven). Tests are tagged [15a] / [15b] / [15c]; none is dropped. The PR 15c plan (`docs/plans/pr-15c-slicing.md`) splits 15c again: **15c** is the slicing engine `si_band` (capacity splits, edge snapping, deflation between slices, edge re-check, duplicate removal, slice collapse, per-leaf counts); **15d** is the cost model, the memory budget, the strategy choice, `verify=None` for bands and `solve_modes(..., band=(f1, f2))`. The [15c] tags below are split into [15c] / [15d].
- Contents:
  - Lanczos basis capacity per slice from the PR 15 memory model (15d: searched from the budget); a slice that fills it before the stop rule holds is split at its largest gap; a full slice without a gap ≥ 100·r_acc raises (test: a forced small capacity splits the slice and still matches `eigsh`).
  - Block shift-invert Lanczos (b = 6) with MINRES inner solves, also used for [0, f_max] when the LOBPCG-from-0 basis does not fit (done, 15d).
  - Slicing with the cost model (15c: the slicing engine, widths limited by the capacity; 15d: the cost model as evidence and the memory-driven strategy; built).
  - Removal of duplicate modes at slice edges.
  - Band-pass count deflated by the modes found in the band and the converged neighbours just outside each band edge. A slice count ≥ 0.5 sets an `info` flag and restarts that slice's Lanczos run with a fresh random block, deflated by the modes already found; after 2 failed restarts the slice is returned with the count flagged.
  - Test tolerances: outer Lanczos residual r_acc = max(1e-10, 10·τ_eff·δ_max) with inner MINRES τ = 1e-8; cluster, slice-edge and band-edge gaps use 100·r_acc.
  - `band=(f1, f2)`.
- Docs: `docs/eigensolvers.md` (memory model for many modes, cost-model calibration, measured MINRES floor; 15a measured the floor, 15d the memory model and calibration; built).
- Tests:
  - [15a, 15c, 15d] Matches SciPy `eigsh(sigma=...)` to relative error ≤ max(1e-8, 10·ε·λ̂_max/λ) for clusters with relative gap ≥ 1e-2 (others within 10·r_acc, multiplicities equal).
  - [15b, 15c, 15d] An interior band strictly between two well-separated modes returns 0 modes with count < 0.5; with a mode planted in that band and left out of V, the count is ≥ 0.5.
  - [15a, 15c, 15d] A clamped model with band (0, f_max) forced onto shift-invert stops and matches `eigsh` (the λ = 0 edge needs no neighbour).
  - [15d] A band with f₁ below λ₁ is solved as [0, f₂], filtered, and matches `eigsh`.
  - [15b] A slice with one in-band mode removed after convergence gives a count ≥ 0.5, sets the `info` flag and recovers the mode on restart; a slice whose count stays ≥ 0.5 (mode forcibly withheld) stops after 2 restarts with the count flagged.
  - [15c] A forced edge failing the final r_acc re-check is widened or merged, and the result still matches `eigsh`; τ_eff forced to 1e-6 near a dense cluster sets the slice-collapse flag.
  - [15c] Each slice's count estimate is < 0.5.
  - [15b] With one in-band mode (not next to a band edge) left out of V, the band-pass count detects it for ≥ 97 of a fixed list of 100 seeds. With V complete, with τ_c = 1e-12, switching off neighbour deflation raises the estimate (same probes y_j) by the probe mean of Σ_i q(μ_i)²(v_iᵀM y_j)² over the converged out-of-band neighbours v_i, to 1e-8 relative; with it, the estimate is < 0.5.
  - [15a] On a model with known repeated interior eigenvalues (voxel cube double/triple modes), each cluster is found with its exact multiplicity.
  - [15d] With a memory budget too small for the LOBPCG-from-0 basis, the memory-driven strategy switches to shift-invert.
  - [15c] On the voxel cube, a band forced across ≥ 2 slices with a cluster near a slice edge, at the default tol = 1e-6: no duplicates, same count and multiplicities as `eigsh`, and λ within max(1e-8, 10·ε·λ̂_max/λ) for clusters with relative gap ≥ 1e-2, otherwise within 10·tol.
  - [15c] A band edge set exactly on a double eigenvalue returns both copies, and d stays ≈ 25/g.
  - [15a, 15d] MINRES iterations vs δ are measured and the cost-model constants fitted; the fp64 V-cycle MINRES true-residual floor is measured; both are recorded in `docs/eigensolvers.md`.
  - [15a] A case with τ forced below the measured floor raises τ_eff to the floor, carries it into r_acc and sets the `info` flag.
  - [15a] r_acc and the measured floor at the largest Mac size are recorded in `docs/eigensolvers.md`.
  - [15a] The memory model (deflation basis plus Lanczos and MINRES workspace) is within 20% of the measured peak at a size where the estimate is ≥ 2 GB.

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
  - Simply supported BCs on solids as defined in PR 8b.
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
  - Discretization-error tests use the PR 8 `eigsh` shift-invert solve (within its size limit, `docs/eigensolvers.md`) (Jacobi LOBPCG is impractical at κ ≈ 1e12).
  - **0.1% target (measured against 3D elasticity):**
    - Srinivas–Rao at L/t = 10 and 100 with ≥ k stacked elements per ply (k stated in the docs).
    - One layered element per laminate is reported, not gated.
    - A clamped cylinder (R/t and L/R stated) against a hex27 reference whose error, estimated from ≥ 3 nested levels (Richardson, estimate only), is ≤ 0.02% (1/5 of the gate), within the PR 8 `eigsh` limit; levels and estimate recorded in `docs/verification.md`.
  - No locking (simply supported square plate, first 6 bending modes, picked by z-reflection parity (u_z even, u_x and u_y odd), same in-plane mesh and layer count at both L/t): at L/t = 10 and 1000, the discretization error is measured against a converged solid-shell run at the same L/t. The mesh keeps that error ≫ 1e-4, since κ ≈ 1e12 at L/t = 1000 limits eigenvalue accuracy to about 1e-4. The error at L/t = 1000 is ≤ 1.5× the error at L/t = 10.
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
  - Band follow-ups from PR 15d (`docs/open_items.md`): automatic memory levers, the up-front spectrum bound ($\hat\lambda_\max$), $\delta_\max \le 1$ enforcement on slices from 0, MINRES remedies near depth 1000 (stagnation), halving before a capacity error, `err.result`, the hybrid LOBPCG/slice junction, $f_1 > 0$ in `solve_modes_memory`, np.memmap, a measured time crossover if any (GPU); performance: the refill through `device_put(V[:, i:i+c].T).T` (2.4× faster on CPU), caching `layout` for the gather path, the verify skips before the gather.
  - AMG on the assembled A_P for solid-shell meshes, and iteration counts vs L/t (test: `solve_modes` with AMG on a solid-shell mesh matches the PR 8 `eigsh` to PR 9 tolerances).
  - `COUNT_STEP` refit at ≥ 1M DOF (16b, re-deferred by 16c; `docs/open_items.md`, PR 16 and the 14a "Memory constants" fit range): measured/model reaches 1.045 at 1.5M `ssh8` DOF (test: the count-step estimate within its band on a ≥ 1M `ssh8` and a ≥ 1M hex20 case).
- Docs: `docs/performance.md`.
- Tests:
  - Roofline = min(bandwidth × arithmetic intensity, measured fp64 peak). The voxel stencil and stored-Ke matvecs are measured against jitted STREAM bandwidth. On the Mac (kernel-bound stored-Ke apply, PR 5) the percentage is reported only; ≥ 50% at m = 1 is a deferred GPU check. For m > 1, the percentage is reported.
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
  - Multi-node (user, 2026-10-17): `jax.distributed.initialize`, one process per GPU; distributed setup (partitioning, padding, halo maps, GMG levels) without building the global mesh on one host; int32 index headroom per shard; GMG coarse-level agglomeration onto fewer devices; on-the-fly Ke for unstructured meshes where stored Ke does not fit.
  - Checkpoint/restart (user, 2026-10-17): sharded save/load of the LOBPCG block and solver state, so a run cut off by an HPC time limit resumes from the last checkpoint; optional wall-clock budget for long solves that writes a checkpoint and returns before the limit.
- Docs: `docs/parallel.md`.
- Tests:
  - Results match single-device runs.
  - Multi-process CPU runs on the Mac (several `jax.distributed` processes on localhost) match single-process runs.
  - A run stopped at a checkpoint and restarted (also with a different process count where supported) matches an uninterrupted run.
  - Deferred GPU checks: weak scaling; 2× A100 across two nodes.

## M6: Optimization

Topology optimization targets the largest size the hardware allows, multi-GPU and multi-node (user, 2026-10-17; replaces the earlier ≤ 10M voxel-element working assumption), so PR 23 builds on PR 19. Still open: clamped supports and densities as the only design variables. The user will decide later. Principle 5 keeps the alternatives open.

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
  - Control-point gradients of the objective, the mass constraint and the detJ aggregate pass the PR 21 finite-difference criterion.
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
    - Unstructured meshes: a Helmholtz PDE filter with PCG, length parameter r = R/(2√3) for cone radius R (recorded in `docs/optimization.md`). Its VJP is Tᵀ K_f⁻¹ T V⁻¹ c (T the element–node transfer, V the element volumes), still one symmetric `custom_linear_solve` with K_f, so the solver is not differentiated. The PR 23 finite-difference test runs it to a 1e-12 tolerance.
  - Optional Heaviside projection with η-continuation: H_η(x) = (tanh(η/2) + tanh(η(x − 1/2)))/(2·tanh(η/2)), so H_η(0) = 0 and H_η(1) = 1.
  - Voxel sensitivity pass (reference Ke and Me scaled per element by s_K, s_M) feeding the cone-filter VJP.
  - Bound formulation with MMA.
  - Sharded on PR 19: densities, sensitivities, the filter (halo exchange for the cone filter, the distributed operator for the Helmholtz filter) and MMA (element-wise updates, `psum` for the constraint sums).
  - Optimizer checkpoint/restart: densities, MMA state (asymptotes, previous iterates), iteration count, continuation parameters and the last modes (warm start X0), saved on PR 19's sharded checkpoints.
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
- PR 11: the low-memory LOBPCG step with the 100M voxel stencil at m = 8 holds ≤ 7.5 blocks (9.0 on CPU), so the matvec + LOBPCG and preconditioner workspace fits in the usable pool (≈ 76 GB).
- PR 11: 100M voxel modal solve (one GPU only if the check above holds, else after PR 19).
- PR 11: GMG restriction (scatter-adds, 6× prolongation on CPU) against prolongation time; rewrite as gathers if it dominates.
- PR 11b: `solve_modes_memory` for a 100M voxel solve (321³ cube, low-memory step, fp32 V-cycle, 10 modes in chunks of 3 with 5 guards, $m = 8$) is 169 GB, the step alone 100 GB (11 + 4.5 blocks of 6.4 GB); expected: it does not fit one 80 GB GPU, so the 100M solve comes after PR 19.
- PR 11b: fp32 V-cycle speed on the GPU (39% faster than fp64 on CPU at $87^3$).
- PR 11b: the σ = 0 guard's host `rigid_modes` at 100M (about 27 s and 20 GB host, scaled from 2M); move it to the device or skip it with a σ hint if it dominates.
- PR 11b: the per-island $A_P$ applies of the deflation setup (0.2 s per island at 2M on CPU).
- PR 13: 10M hex20 modal solve on one GPU: 20 modes, $m = 25$, stored Ke, 9-block step (71.5 GB total by `solve_modes_memory`, device about 70 GB, against the 76 GB pool); fallback on-the-fly Ke (about 60 GB, estimated) via `solve_modes(..., operator_options={"mode": "otf"})`; the host-side AMG setup needs about 15 GB of RAM (extrapolated from 1M).
- PR 13b: the hex8 assembler's scatter-add on the GPU: determinism (summation order at rounding level) and speed against the CPU's 0.72 s at 1M hex20.
- PR 5: scatter vs gather benchmark on GPU; sets the GPU default and re-tunes the $R$ rule (the CPU cap of 2 vectors gives $R = 32768$ at 1M DOF).
- PR 15d: the LOBPCG / slice cost ratio on the GPU (memory-driven strategy kept unless a crossover appears); allocating the band basis at $C_{max}$ up front against fragmentation; the per-space fit test against the GPU pool and host RAM; the 10M hex20 band reach (120–260 locked modes with fp64 AMG) and its MINRES behaviour near depth 1000; the slice and count peaks against `si_memory` / `count_memory` on the device.
- PR 18: matvec ≥ 50% of roofline at m = 1.
- PR 19: weak scaling.
- PR 24: harness Linux and GPU peak paths, checked with the PR 0 harness test cases.

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
| 13c Solver options | 13 |
| 14a Counting | 9, 10, 11 (preconditioned K + σM solves on voxel meshes; 13 optional; without PR 13, unstructured counting uses Jacobi PCG) |
| 14b [0, f_max], verify | 14a |
| 15a Shift-invert slice | 10, 13 |
| 15b Band-pass count | 14a, 15a |
| 15c Slicing | 15a, 15b |
| 15d Band strategy, `band=(f1, f2)` | 14b, 15c |
| 16a Solid-shell element | 3, 4 |
| 16b Solid-shell operator and API | 5, 9, 16a |
| 16c Solid-shell accuracy suite | 8, 16b |
| 17 Postprocessor | 9 |
| 18 Performance | 11, 13, 15d, 16c |
| 19 Multi-GPU | 18 |
| 20 MMA | 0 |
| 21 Sensitivities | 9 |
| 22 FFD | 20, 21 |
| 23 Topology opt | 11, 19, 20, 21 |
| 24 GPU validation | CUDA hardware; the PRs in the deferred list |

- **Critical path:** 0 → 1 → 3 → 4 → 7 → 8 → 9, with PR 2 in parallel with 1 and PR 5 in parallel with 7–8. This gives first modes in pure JAX.
- **Biggest risk:** M3 (preconditioning).
- **Parallel work:** PRs 2, 6 and 20 can proceed alongside the critical path. PRs 16 and 17 can proceed alongside M3.
