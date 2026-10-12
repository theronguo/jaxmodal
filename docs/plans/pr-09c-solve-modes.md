# PR 9c plan: `solve_modes`, examples, Mac acceptance budgets

Branch `pr-09c-solve-modes`, from `main` after PR 9b (split: [`pr-09a-lobpcg.md`](pr-09a-lobpcg.md)).

## Scope

The PR 9 roadmap items of the 9c row:
- Contents 5: the public API `solve_modes(mesh, materials, bcs, n_modes, orientations=None, ...) → (λ, V, info)`, calling `check_materials` and `check_elements` before tracing (through the operator builders, which run them first, host-side; decision 5);
- Contents 6: an `examples/` script per milestone, also run as a smoke test;
- Contents 7: finalized Mac acceptance budgets;
- the M2 milestone ("first end-to-end modes in pure JAX").

No PR 9 acceptance test is left for 9c: 9a and 9b cover them all. 9c adds the tests below.

**Deviations from the roadmap:** none. Decisions the roadmap leaves open:
1. **Rigid modes are deflated and returned separately.** `solve_modes` builds the rigid modes that the BCs leave (`rigid_modes(mesh, free)`, PR 7), M-orthonormalizes them with the operator's mass (nodal masses included), and passes them as `R`. It returns the lowest `n_modes` *elastic* modes; `info["rigid"]` is the M-orthonormal rigid basis $(n_\mathrm{dof}, k)$ as a NumPy fp64 array ($k \le 6\cdot$`MAX_PARTS` columns), for PR 17's effective-mass sum rule ("rigid modes included in the mode set"), and `info["n_rigid"]` $= k$. This matches `lobpcg` and PR 14's counting, which also starts above the rigid modes.
2. **Operator dispatch:** a mesh with a voxel grid (`mesh.voxel is not None`) gets the voxel stencil (PR 6), any other mesh the unstructured operator (PR 5) in its default stored mode. `info["operator"]` is `"stencil"` or `"unstructured"`. PRs 11 and 13 add their preconditioners to this dispatch; 9c uses `lobpcg`'s default Jacobi.
3. **No top-level export:** `solve_modes` lives in `eigen/api.py` (package layout) and is exported from `jaxmodal.eigen`. `import jaxmodal` stays free of jax imports.
4. **Frequencies:** `info["frequencies"]` $= \sqrt{\max(\lambda, 0)}/(2\pi)$ (Hz for SI inputs), for the examples and PR 14's `band=(0, f_max)`.
5. **Host checks:** `unstructured` already runs `phases`, `check_materials` and `check_elements` first (host-side, before any build or trace), and `stencil` runs `check_materials` and its grid check (which implies $\det J > 0$). `solve_modes` does not repeat them; it only runs `dof_mask` before the build, so a bad BC dict fails before the 1–2 s operator build. Test 7 pins the order.
6. **Options:** `nodal_mass` (roadmap key decisions: "an optional nodal (point) mass vector is added to M"; PRs 11, 14, 21, 23 build on it) is forwarded to the operator. Other operator options (`mode="otf"`, `path`, `chunk`/`range_nodes`, `slab`) are not exposed yet (Not covered).

## Measured facts (prototype `scratchpad/p9c/`; jax 0.11.2, fp64, M3 Max CPU)

Isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$; 10 modes, tol $10^{-10}$, Jacobi; reference: `reference_modes` on the assembled matrices.

| mesh | BCs | operator | rigid | iterations | time | max rel. λ error |
|---|---|---|---|---|---|---|
| hex20 box $6\times5\times4$ | `xmin` clamped | unstructured | 0 | 236 | 1.9 s | $2.5\cdot10^{-13}$ |
| hex20 box $6\times5\times4$ | free | unstructured | 6 | 271 | 1.2 s | $2.0\cdot10^{-14}$ |
| voxel $10\times8\times6$ | `zmin` clamped | stencil | 0 | 120 | 1.8 s | $5.8\cdot10^{-14}$ |
| voxel $10\times8\times6$ | free | stencil | 6 | 125 | 1.1 s | $2.6\cdot10^{-14}$ |

- Host overhead at the acceptance sizes (review probe): `check_elements` 0.05 s (hex20 $44^3$) and 0.09 s (voxel $87^3$); operator build 2.0 s and 0.2 s; `rigid_modes` 0.1 s and 0.5 s; `m_orthonormalize` 0.1 s and 0.2 s. Under 3 s in total.
- Edge cases (correctness review, tol $10^{-10}$, against `reference_modes` or dense `eigh`): sliding BCs with 3, 1 and 3 rigid modes, two disjoint voxel blocks (6 rigid with one clamped, 12 free, consistent and lumped), lumped voxel and HRZ hex27: max rel. λ error $\le 7\cdot10^{-14}$, and the reference's zero-eigenvalue count equals `n_rigid` in every case. Voxel masks whose parts touch only at an edge or vertex raise from `parts` (PR 7 decision), and more than `MAX_PARTS` = 10 parts raise from `rigid_modes`.
- On the two-block free voxel with lumped mass (12 rigid, 1,764 free DOF), `reference_modes` raises "eigsh missed modes" at request sizes 20 and 26 (PR 8a's certification), while `solve_modes` matches dense `eigh` to $3\cdot10^{-13}$: multi-part tests use the dense oracle.
- **AMG setup** (performance review, `scratchpad/p9c-rev/amg.py`): pyamg smoothed aggregation on the PR 13 acceptance hex8 $A_P$ ($44^3$, 273k DOF, 21M nonzeros, masked rigid modes as B) takes 1.9 s (3 levels, operator complexity 1.15); `assemble` 2.6 s; about 3 GB peak.
- **Coarse SpMV** (same review, `ell.py`): a JAX block-ELL SpMV with 27 neighbours of 3×3 blocks at 273k DOF takes 39 ms ($m = 25$) and 29 ms ($m = 15$).
- **Throughput at the acceptance sizes** (steel, Jacobi; `scratchpad/p9c/budget.py`):

  | case | DOF | m | apply_K | apply_M | 9-block step | low-memory step | setup |
  |---|---|---|---|---|---|---|---|
  | hex20 $44^3$, clamped, 20 modes | 1.08M | 25 | 140 ms | 167 ms | 730 ms | 1165 ms | 2.6 s |
  | voxel $87^3$, 10 modes | 2.04M | 15 | 160 ms | 162 ms | 786 ms | 1248 ms | 1.0 s |

## Mac acceptance budgets

The roadmap's provisional figures are 10 min (PR 11: 2M voxel, 10 modes) and 15 min (PR 13: 1M hex20, 20 modes). **Decision: they become final, unchanged, and the acceptance run is specified:** tol $= 10^{-6}$, 9-block step, fp64 basis, wall time including setup, the σ pre-estimate and compilation. The fp32 V-cycle run of PR 11 has the same budget.

Rationale (model for a preconditioned solve, from the throughput above):

$$T \approx T_\text{setup} + T_\sigma + T_\text{compile} + N_\text{it}\,(t_\text{step} + c_V\,t_A + t_\text{axpy}),$$

with $t_A = t_K + t_M$ (the preconditioners approximate $K + \sigma M$, $\sigma > 0$, and there is no fused apply), $c_V$ the fine-level applies per V-cycle counted in $t_A$ units (Chebyshev degree 2 pre- and post-smoothing plus the residual: 5; coarse levels: voxel GMG level 1 about 0.5, hex20 p-MG plus hex8 AMG about 1, from the coarse SpMV above), $t_\text{axpy} \approx 0.1$ s the smoother's vector updates, and $N_\text{it} = 150$ LOBPCG iterations, an assumption that PRs 11 and 13 must check (their PCG criteria, 100 iterations to $10^{-8}$, are a different quantity).

| case | per iteration | $N_\text{it}$ = 150 | setup + σ + compile | total | budget | margin |
|---|---|---|---|---|---|---|
| voxel 2M, 10 modes | $0.79 + 5.5 \cdot 0.32 + 0.1 = 2.65$ s | 6.6 min | about 1.5 min | about 8 min | 10 min | 1.25× |
| hex20 1M, 20 modes | $0.73 + 6 \cdot 0.31 + 0.1 = 2.69$ s | 6.7 min | about 1.5 min (AMG 1.9 s, hex8 assembly 2.6 s, PR 12 Galerkin product, σ, compile) | about 8 min | 15 min | 1.9× |

Tightening to the model would leave no margin for the two unmeasured inputs ($N_\text{it}$, the V-cycle cost). If PR 11 or 13 measures $N_\text{it} > 150$ or a costlier V-cycle, it revisits the budget there with this model; a fused $K + \sigma M$ apply ($t_A \approx t_K$) is the first lever. The model and values go into `docs/eigensolvers.md`; the roadmap drops "provisional" and adds the run specification to the PR 11 and PR 13 Mac acceptance lines. The peak criteria (within 20% of the estimators) are unchanged.

## Files

- `src/jaxmodal/eigen/api.py`: `solve_modes`.
- `src/jaxmodal/eigen/__init__.py`: export.
- `tests/test_api.py`: the tests below.
- `examples/m0_foundations.py`, `examples/m1_operators.py`, `examples/m2_modes.py`.
- `tests/test_examples.py`: the smoke test.
- Docs: `README.md`, `docs/eigensolvers.md`, `docs/ROADMAP.md`.

## API

```python
solve_modes(mesh, materials, bcs, n_modes, orientations=None, *, mass="consistent",
            nodal_mass=None, tol=1e-6, maxit=500, seed=0, chunk=None, guards=None,
            basis_dtype="float64", low_memory=False) -> (lam, V, info)
```

1. `free = dof_mask(mesh, bcs)` (raises for unknown sets or bad components), before any build.
2. Operator: `stencil(mesh, materials, orientations, mass=mass, nodal_mass=nodal_mass)` if `mesh.voxel is not None`, else `unstructured(mesh, materials, orientations, mass=mass, nodal_mass=nodal_mass)`. Both run the material (and element or grid) checks first, host-side. `info["operator"]` is set in the same branch as the call.
3. `R = rigid_modes(mesh, free)`; with $k > 0$ columns, `R = jnp.asarray(R)` and `R = m_orthonormalize(R, op.apply_M(R))` (it donates its first argument: a device array, distinct from `op.apply_M(R)`), else `None`.
4. `lam, V, info = lobpcg(op, free, n_modes, tol, R, maxit=maxit, seed=seed, chunk=chunk, guards=guards, basis_dtype=basis_dtype, low_memory=low_memory)`.
5. `info` gains `operator`, `n_rigid`, `rigid` (NumPy, or a $(n_\mathrm{dof}, 0)$ array) and `frequencies`. Returns `lam` (elastic modes, ascending), `V` (NumPy fp64, M-orthonormal, constrained rows 0) and `info`.

Errors (`ValueError` unless stated): bad `bcs`; the builders' material, element, grid and nodal-mass checks; voxel or unstructured meshes whose parts touch only at an edge or vertex (`parts`), and more than 10 parts (`rigid_modes`; remove floating islands); `lobpcg`'s, e.g. `n_modes` + guards above the free DOFs minus the rigid count; `RuntimeError`/`FloatingPointError` breakdowns as `lobpcg`. With 9c's Jacobi, the default `maxit=500` suits compact models; slender ones (the M2 beam) need a larger `maxit` until PRs 11 and 13.

## Tests (`tests/test_api.py`)

Reference: `reference_modes(K, M, n_modes + n_rigid + 6, free, points=mesh.points)` (6 extra so a cluster at the end keeps its partners, the 9a convention) on `assemble(mesh, materials, orientations, mass=..., nodal_mass=...)`; its first `n_rigid` eigenvalues are asserted $\le 10^{-8}\lambda_{n_\mathrm{rigid}+1}$ and eigenvalue `n_rigid + 1` is not, which pins the rigid count independently of `solve_modes`. λ tolerance: $\max(10^{-8}, 10\varepsilon\hat\lambda_\max/\lambda)$ for modes with relative gap ≥ $10^{-2}$, $10\,\mathrm{tol}$ cluster-wise otherwise (the 9a helper from `tests/test_lobpcg.py`), tol $10^{-10}$. Builder calls are observed with monkeypatch spies on `api.unstructured`, `api.stencil` and `api.lobpcg` that record their arguments and forward to the real functions.

1. **Unstructured, clamped:** hex20 box $6\times5\times4$, `xmin` clamped, 10 modes: λ as above, `V` M-orthonormal to $10^{-10}$ with constrained rows 0, `info["operator"] == "unstructured"`, `unstructured` called once and `stencil` never, `n_rigid == 0` and `info["rigid"].shape == (n_dof, 0)`, `frequencies` $= \sqrt\lambda/2\pi$.
2. **Free-free with nodal masses:** the same box, `bcs={}`, random positive nodal masses (about 1% of the element mass per node): `n_rigid == 6`, λ match the reference (with the same nodal masses) above its 6 zero modes, `info["rigid"]` M-orthonormal and $\|R^\top M V\| \le 10^{-10}$ with the full mass.
3. **Sliding BC:** `{"xmin": "x"}`: `n_rigid == 3` ($T_y$, $T_z$, $R_x$), λ match the reference above its 3 zero modes.
4. **Voxel:** voxel $10\times8\times6$, `zmin` clamped and free: `info["operator"] == "stencil"`, `stencil` called once and `unstructured` never, λ as above (rigid 0 and 6).
5. **Anisotropic, rotated, hex27, lumped:** hex27 box, a transversely isotropic material with orientation label 1 on half the elements (`from_euler`), `mass="lumped"`: λ against the assembled reference with the same orientations and lumped mass; the identity-orientation reference differs from it by more than 100× the test tolerance (so forwarding is observable).
6. **Forwarding:** on test 1's box, two calls with 12 modes: (a) `tol=1e-9`, `maxit=900`, `seed=3`, `chunk=4`, `guards=2`; (b) `tol=1e-6`, `chunk=4`, `basis_dtype="float32"`, `low_memory=True`. The `lobpcg` spy asserts each value arrives; λ match the reference (9a rule at the call's tol) and `len(info["chunks"]) == 3`.
7. **Host checks before the build and before tracing**, each `pytest.raises(ValueError, match=<the specific message>)`:
   - bad BCs (an unknown node set, a bad component string): the builder spies raise `AssertionError("operator built before the BC check")` if called;
   - a material label missing from `materials`, a non-positive density, an inverted element (two nodes swapped): raised by the builders' checks, with the `lobpcg` spy raising `AssertionError` if reached.

**Examples smoke test** (`tests/test_examples.py`): each script runs with `sys.executable` in a subprocess, with an output directory argument (`tmp_path`), exits 0 and prints its final check line; fast suite (each < 20 s including import and compile).

**Mutants**, each killed by the test given:

| mutant | killed by |
|---|---|
| rigid modes not deflated | 2, 3 (the zero modes come back: λ error 1.0) |
| rigid modes not M-orthonormalized (unit columns) | 2, 3 (`lobpcg` raises on $R^\top M_mR \ne I$) |
| `free=None` passed to `rigid_modes` (all 6 modes) | 1, 3 (`lobpcg` raises on the R check: the unmasked modes are not M-orthonormal on the free DOFs); equivalent on free-free |
| voxel mesh sent to `unstructured` | 4 (spies) |
| orientations not forwarded | 5 |
| `mass` not forwarded | 5 |
| `nodal_mass` not forwarded | 2 |
| R orthonormalized without the nodal masses | 2 ($R^\top MV$, `lobpcg`'s R check) |
| `dof_mask` after the operator build | 7 (spies) |
| `tol`/`maxit`/`seed`/`chunk`/`guards`/`basis_dtype`/`low_memory` not forwarded | 6 (spy) |
| `n_rigid` or `rigid` wrong | 1, 2, 3 |
| frequencies without the $2\pi$ | 1 |
| frequencies without $\max(\lambda, 0)$ | none: equivalent (returned λ are elastic, positive) |

## Examples (`examples/`)

Each script takes an optional output directory (default: a temporary directory), runs in a few seconds, and ends with a check that raises on failure.
- **`m0_foundations.py`** (M0): read the checked-in Gmsh pair (`tests/data/gmsh_pair.msh`, path resolved from `__file__`), build a voxel sphere and a box mesh, write VTU, define an isotropic and a rotated transversely isotropic material, `phases`/`phase_table`. Check: the written VTU reads back with the same points and cells.
- **`m1_operators.py`** (M1): element matrices, assembled K and M, the matrix-free operator and the voxel stencil, rigid modes and the deflation projector. Check: `apply_K` equals the assembled product to $10^{-12}$ (relative) for both operators.
- **`m2_modes.py`** (M2): a steel cantilever (hex20 $20\times2\times2$, $1 \times 0.1 \times 0.1$ m, clamped at `xmin`, 6 modes, tol $10^{-6}$, `maxit=1000`) and a free-free voxel aluminium cube via `solve_modes`, printing frequencies in Hz. Checks: the first bending frequency within 1% of Euler–Bernoulli, $f_1 = \frac{1.8751^2}{2\pi L^2}\sqrt{EI/(\rho A)}$ (measured 83.48 Hz, ratio 0.9991, 403 iterations, 2.8 s), and the λ match `reference_modes` by the 9a rule at tol $10^{-6}$ ($10\,$tol cluster-wise: the square section makes the bending modes pairs, the cube has triples). Jacobi on this slender beam needs more than 500 iterations at tol $10^{-8}$ (hits `maxit` at $20\times2\times2$ and $40\times4\times4$), hence tol $10^{-6}$ and the larger `maxit`.

## Docs

- `docs/eigensolvers.md`: a `solve_modes` section (signature, dispatch, rigid deflation and `info["rigid"]`, `nodal_mass`, `info` additions, errors including hinged parts and `MAX_PARTS`, the `maxit` note for slender models under Jacobi) and a "Mac acceptance budgets" subsection (the run specification, the model and the values).
- `docs/ROADMAP.md`: "provisional" dropped from the budget line (Development environment) with a pointer to the model; the PR 11 and PR 13 Mac acceptance lines get the run specification (tol $10^{-6}$, 9-block, fp64 basis, wall time including setup, σ pre-estimate and compilation); status line; the PR 9 split note marks 9c merged; M2 milestone reached; a note that later milestones add their own example scripts.
- `README.md`: a `solve_modes` usage snippet, the examples, and the status line.

## Not covered

- Preconditioner choice and the σ pre-estimate in the dispatch (PRs 11 and 13).
- Operator options beyond `mass` and `nodal_mass` (`mode="otf"`, `path`, `chunk`/`range_nodes`, `slab`): added when a later PR needs them (e.g. PR 13/18 for on-the-fly figures), backward compatible.
- A `solve_modes` memory estimate: PRs 11 and 13 combine the operator, LOBPCG and preconditioner estimators for their Mac acceptance.
- Exporting mode shapes to VTU (PR 17's postprocessor).
