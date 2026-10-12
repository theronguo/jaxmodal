# PR 9c plan: `solve_modes`, examples, Mac acceptance budgets

Branch `pr-09c-solve-modes`, from `main` after PR 9b (split: [`pr-09a-lobpcg.md`](pr-09a-lobpcg.md)).

## Scope

The PR 9 roadmap items of the 9c row:
- Contents 5: the public API `solve_modes(mesh, materials, bcs, n_modes, orientations=None, ...) → (λ, V, info)`, calling `check_materials` and `check_elements` before tracing;
- Contents 6: an `examples/` script per milestone, also run as a smoke test;
- Contents 7: finalized Mac acceptance budgets;
- the M2 milestone ("first end-to-end modes in pure JAX").

No PR 9 acceptance test is left for 9c: 9a and 9b cover them all. 9c adds the tests below.

**Deviations from the roadmap:** none. Decisions the roadmap leaves open:
1. **Rigid modes are deflated, not returned.** `solve_modes` builds the rigid modes that the BCs leave (`rigid_modes(mesh, free)`, PR 7), M-orthonormalizes them with the operator's mass, and passes them as `R`. It returns the lowest `n_modes` *elastic* modes; `info["n_rigid"]` is the number deflated. This matches `lobpcg` and PR 14's counting, which also starts above the rigid modes.
2. **Operator dispatch:** a mesh with a voxel grid (`mesh.voxel is not None`) gets the voxel stencil (PR 6), any other mesh the unstructured operator (PR 5) in its default stored mode. `info["operator"]` is `"stencil"` or `"unstructured"`. PRs 11 and 13 add their preconditioners to this dispatch; 9c uses `lobpcg`'s default Jacobi.
3. **No top-level export:** `solve_modes` lives in `eigen/api.py` (package layout) and is exported from `jaxmodal.eigen`. `import jaxmodal` stays free of jax imports.
4. **Frequencies:** `info["frequencies"]` $= \sqrt{\max(\lambda, 0)}/(2\pi)$ (Hz for SI inputs), for the examples and PR 14's `band=(0, f_max)`.

## Measured facts (prototype `scratchpad/p9c/`; jax 0.11.2, fp64, M3 Max CPU)

Isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$; 10 modes, tol $10^{-10}$, Jacobi; reference: `reference_modes` on the assembled matrices.

| mesh | BCs | operator | rigid | iterations | time | max rel. λ error |
|---|---|---|---|---|---|---|
| hex20 box $6\times5\times4$ | `xmin` clamped | unstructured | 0 | 236 | 1.9 s | $2.5\cdot10^{-13}$ |
| hex20 box $6\times5\times4$ | free | unstructured | 6 | 271 | 1.2 s | $2.0\cdot10^{-14}$ |
| voxel $10\times8\times6$ | `zmin` clamped | stencil | 0 | 120 | 1.8 s | $5.8\cdot10^{-14}$ |
| voxel $10\times8\times6$ | free | stencil | 6 | 125 | 1.1 s | $2.6\cdot10^{-14}$ |

- `check_elements` on hex20 $44^3$ (1.08M DOF) takes 0.05 s, and building the operator 1.7 s. `unstructured` runs both host checks again internally; the duplicate costs nothing measurable.
- **Throughput at the acceptance sizes** (steel, Jacobi; `scratchpad/p9c/budget.py`):

  | case | DOF | m | apply_K | apply_M | 9-block step | low-memory step | setup |
  |---|---|---|---|---|---|---|---|
  | hex20 $44^3$, clamped, 20 modes | 1.08M | 25 | 140 ms | 167 ms | 730 ms | 1165 ms | 2.6 s |
  | voxel $87^3$, 10 modes | 2.04M | 15 | 160 ms | 162 ms | 786 ms | 1248 ms | 1.0 s |

## Mac acceptance budgets

The roadmap's provisional figures are 10 min (PR 11: 2M voxel, 10 modes) and 15 min (PR 13: 1M hex20, 20 modes). Model for a preconditioned solve, from the throughput above:

$$T \approx T_\text{setup} + N_\text{it}\,(t_\text{step} + c_V\,t_K),$$

with $c_V = 8$ fine-level K applies per V-cycle (Chebyshev degree 2 pre- and post-smoothing plus the residual, coarse levels about 1/7 more for 3D coarsening by 2) and $N_\text{it} = 150$ LOBPCG iterations (PR 11's PCG criteria allow 100 iterations per solve; LOBPCG with a multigrid preconditioner needs a comparable count).
- **Voxel, 2M DOF, 10 modes:** $150\,(0.79 + 8 \cdot 0.16)$ s $= 310$ s, plus GMG setup (Galerkin levels, dense coarsest Cholesky; < 1 min) and compile: about 6 min. **Budget: 8 min.**
- **hex20, 1M DOF, 20 modes:** $150\,(0.73 + 8 \cdot 0.14)$ s $= 278$ s, plus pyamg setup on the hex8 Galerkin $A_P$ (273k DOF; about 2 min) and compile: about 7.5 min. **Budget: 12 min.**

Both keep about 1.3–1.6× margin over the model. They replace the provisional values in the roadmap (Development environment, PR 11 and PR 13 Mac acceptance) and are recorded in `docs/eigensolvers.md`. The peak criteria (within 20% of the estimators) are unchanged.

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
            tol=1e-6, maxit=500, seed=0, chunk=None, guards=None,
            basis_dtype="float64", low_memory=False) -> (lam, V, info)
```

1. `index, pairs = phases(mesh)`; `check_materials(pairs, materials, orientations)`; `check_elements(mesh, mass)`; `free = dof_mask(mesh, bcs)` (raises for unknown sets or bad components). All host-side, before any operator is built or anything traced.
2. Operator: `stencil(mesh, materials, orientations, mass=mass)` if `mesh.voxel is not None`, else `unstructured(mesh, materials, orientations, mass=mass)`.
3. `R = rigid_modes(mesh, free)`; with $k > 0$ columns, `R = m_orthonormalize(R, op.apply_M(R))`, else `None`.
4. `lam, V, info = lobpcg(op, free, n_modes, tol, R, maxit=maxit, seed=seed, chunk=chunk, guards=guards, basis_dtype=basis_dtype, low_memory=low_memory)`.
5. `info` gains `operator`, `n_rigid` and `frequencies`. Returns `lam` (elastic modes, ascending), `V` (NumPy fp64, M-orthonormal, constrained rows 0) and `info`.

Errors: as `lobpcg` (e.g. `n_modes` larger than the free DOFs minus the rigid count) plus the host checks above.

## Tests (`tests/test_api.py`)

Reference: `reference_modes(K, M, n_modes + n_rigid, free)` on `assemble(mesh, materials, orientations, mass=...)`; λ tolerance $\max(10^{-8}, 10\varepsilon\hat\lambda_\max/\lambda)$ for modes with relative gap ≥ $10^{-2}$, $10\,\mathrm{tol}$ cluster-wise otherwise (the 9a rule, helper reused from `tests/test_lobpcg.py`), tol $10^{-10}$.

1. **Unstructured, clamped:** hex20 box $6\times5\times4$, `xmin` clamped, 10 modes: λ as above, `V` M-orthonormal to $10^{-10}$ with constrained rows 0, `info["operator"] == "unstructured"`, `n_rigid == 0`, `frequencies` $= \sqrt\lambda/2\pi$.
2. **Free-free:** the same box, `bcs={}`: `n_rigid == 6`, λ match the reference's modes 7–16, $\|R^\top M V\| \le 10^{-10}$ with $R$ from `rigid_modes`.
3. **Sliding BC:** `{"xmin": "x"}`: `n_rigid == 3` ($T_y$, $T_z$, $R_x$), λ match the reference above its 3 zero modes.
4. **Voxel:** voxel $10\times8\times6$, `zmin` clamped and free: `info["operator"] == "stencil"`, λ as above (rigid 0 and 6).
5. **Anisotropic, rotated, hex27, lumped:** hex27 box, a transversely isotropic material with orientation label 1 on half the elements (`from_euler`), `mass="lumped"`: λ against the assembled reference with the same orientations and lumped mass.
6. **Forwarding:** chunked (`chunk=4`, 12 modes, 3 chunks) and `low_memory=True` on test 1's box: λ as above, `len(info["chunks"]) == 3`.
7. **Host checks before tracing:** each raises `ValueError` without building an operator (`monkeypatch` `api.unstructured` and `api.stencil` to fail if called): a material label missing from `materials`, a non-positive density, an inverted element (two nodes swapped), an unknown node set in `bcs`, a bad component string.

**Examples smoke test** (`tests/test_examples.py`): each script runs in a subprocess with an output directory argument (`tmp_path`), exits 0 and prints its final check line; fast suite (each < 20 s).

**Mutants**, each killed by the test given:

| mutant | killed by |
|---|---|
| rigid modes not deflated | 2 (λ shift: the zero modes come back), 3 |
| rigid modes not M-orthonormalized (unit columns) | 2 (`lobpcg` raises on $R^\top MR \ne I$) |
| `free=None` passed to `rigid_modes` (all 6 modes) | 1 (clamped: 6 spurious deflations shift λ), 3 |
| voxel mesh sent to `unstructured` | 4 (`operator`) |
| orientations not forwarded | 5 |
| `mass` not forwarded | 5 |
| host checks after the operator build | 7 |
| `chunk`/`low_memory` not forwarded | 6 |
| frequencies without the $2\pi$ | 1 |

## Examples (`examples/`)

Each script takes an optional output directory (default: a temporary directory), runs in a few seconds, and ends with a check that raises on failure.
- **`m0_foundations.py`** (M0): read the checked-in Gmsh pair (`tests/data/gmsh_pair.msh`), build a voxel sphere and a box mesh, write VTU, define an isotropic and a rotated transversely isotropic material, `phases`/`phase_table`. Check: the written VTU reads back with the same points and cells.
- **`m1_operators.py`** (M1): element matrices, assembled K and M, the matrix-free operator and the voxel stencil, rigid modes and the deflation projector. Check: `apply_K` equals the assembled product to $10^{-12}$ (relative) for both operators.
- **`m2_modes.py`** (M2): a steel cantilever (hex20 $20\times2\times2$, $1 \times 0.1 \times 0.1$ m, clamped at `xmin`, 6 modes, tol $10^{-6}$, `maxit=3000`) and a free-free voxel aluminium cube via `solve_modes`, printing frequencies in Hz. Checks: the first bending frequency within 1% of Euler–Bernoulli, $f_1 = \frac{1.8751^2}{2\pi L^2}\sqrt{EI/(\rho A)}$ (measured 83.48 Hz, ratio 0.9991, 403 iterations, 2.8 s), and the λ match `reference_modes` (tolerance fixed from the measured error at implementation). Jacobi on this slender beam needs more than 500 iterations at tol $10^{-8}$ (hits `maxit` at $20\times2\times2$ and $40\times4\times4$), hence tol $10^{-6}$ and the larger `maxit`.

## Docs

- `docs/eigensolvers.md`: a `solve_modes` section (signature, dispatch, rigid deflation, `info` additions, errors) and a "Mac acceptance budgets" subsection (the model and the values).
- `docs/ROADMAP.md`: the provisional-budget line (Development environment) and the PR 11/13 Mac acceptance times replaced by the PR 9 values; status line; the PR 9 split note marks 9c merged; M2 milestone reached.
- `README.md`: a `solve_modes` usage snippet, the examples, and the status line.

## Not covered

- Preconditioner choice and the σ pre-estimate in the dispatch (PRs 11 and 13).
- A `solve_modes` memory estimate: PRs 11 and 13 combine the operator, LOBPCG and preconditioner estimators for their Mac acceptance.
- Exporting mode shapes to VTU (PR 17's postprocessor).
