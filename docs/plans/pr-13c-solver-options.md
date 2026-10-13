# PR 13c plan: every solver option reachable from `solve_modes`

Status: draft (plan review round 1 pending).

User request (2026-10-13, after PR 13b): "Generally make the default settings a robust choice for most problems. But also expose all solver options to the user in some way." PR 13c is inserted before PR 14 so that PRs 14–15 follow the same pattern from the start. It is an API and docs PR: no numerics change at the defaults.

## Facts

`solve_modes(mesh, materials, bcs, n_modes, orientations=None, *, mass, nodal_mass, tol, maxit, seed, chunk, guards, basis_dtype, low_memory, precond, sigma, precond_dtype)` already exposes every `lobpcg` option except `X0`, and the operator's `mass`/`nodal_mass`. Options that are reachable today only by composing `unstructured`/`stencil` + `gmg`/`amg` + `lobpcg` by hand:

| component | options not reachable from `solve_modes` | default |
|---|---|---|
| `unstructured` | `mode` ("stored" / "otf"), `path` (scatter / gather), `chunk` (elements per scan step), `range_nodes` | "stored", auto, auto, None |
| `stencil` | `slab` (cells per scan step) | auto |
| `gmg` | `degree`, `coarse_degree`, `ratio` | 2, 4, 30 |
| `amg` | `degree`, `ratio`, `max_coarse` | 2, 30, 1000 |
| `jacobi` | — | — |
| σ pre-estimate (`_precond_sigma`) | LOBPCG tol per round, iterations per round, rounds, the drop factor that triggers a rebuild, `sigma_estimate`'s `maxit` for the fallback start | 0.1, 20, 4, 3, 10 |
| `lobpcg` | `X0` (initial block; a warm start for optimization loops) | random (seeded) |

`unstructured`'s `chunk` and `lobpcg`'s `chunk` share a name, so the operator options cannot simply become more top-level keywords.

`solve_modes_memory` composes `unstructured_memory(mesh, m, mode, mass, path, chunk, range_nodes)` or `stencil_memory(mesh, m, mass, slab)` with `gmg_memory(mesh, m, dtype)` / `amg_memory(mesh, m)`; neither preconditioner estimator models non-default smoother degrees or `max_coarse`.

## Design

### 1. Option dicts in `solve_modes`

New keywords, all `None` by default (= today's behaviour):
- `operator_options`: forwarded to the operator builder. Allowed keys: `mode`, `path`, `chunk`, `range_nodes` (unstructured meshes); `slab` (voxel meshes).
- `precond_options`: forwarded to the preconditioner builder, both for the pre-estimate builds and the final build. Allowed keys: `degree`, `coarse_degree`, `ratio` (gmg); `degree`, `ratio`, `max_coarse` (amg); none (jacobi). `dtype` stays the top-level `precond_dtype`, and `amg`'s `nonsingular` stays derived from the rigid-mode count (it is a proof, not a preference); `aggregates`/`return_aggregates` are state, not settings, and stay with the building blocks (docs show the composition).
- `sigma_options`: the pre-estimate's tuning, keys `tol`, `maxit`, `rounds`, `drop`, `estimate_maxit` with the defaults above. Ignored (and rejected if given) when `sigma` is passed or `precond="jacobi"`, which run no pre-estimate.
- `X0`: forwarded to `lobpcg`.

One helper validates a dict against the allowed key set of the resolved component and raises `ValueError` naming the component, the unknown keys and the allowed ones (a typo must not be silently ignored). Values are validated by the builders themselves (they already raise on bad values). `_precond_sigma` takes the merged `sigma_options` instead of its constants.

### 2. `solve_modes_memory`

Takes `operator_options` and `precond_options` with the same validation:
- `operator_options` are forwarded to `unstructured_memory` / `stencil_memory`, which already take exactly those keys (so `mode="otf"` gives the on-the-fly estimate, the 10M fallback in the ROADMAP).
- `precond_options` are validated but the estimators model only the defaults; non-default values are documented as "estimated at the default smoother degrees and `max_coarse`" (logged in `docs/open_items.md`). Rationale: a degree changes the V-cycle temporaries by a fraction of a block and `max_coarse` only the coarsest factor ($8\,n_\text{coarse}^2$, which `amg_memory` could take later), not worth new calibrations now.

### 3. Defaults audit

The docs get one table (`docs/eigensolvers.md`, "Options") listing every option of `solve_modes` with its default and the reason it is the robust choice, citing the measurements behind it (e.g. GMG/AMG over Jacobi: 14 against 300+ PCG iterations; σ pre-estimate within 1.05 λ₁ on every tested case but one; `degree` 2: least workspace at equal cost; `coarse_degree` 4: SIMP 116 → 87; `max_coarse` 1000; `mode="stored"`: fastest, `"otf"` for memory; tol $10^{-6}$; 9-block step; fp64 basis). No default changes: every current default is already the robust option measured in PRs 9–13. Where a default trades robustness for speed, the table says so.

## Files

- `src/jaxmodal/eigen/api.py`: the keywords, the validation helper, `_precond_sigma(op, free, R, build, opts)`.
- `src/jaxmodal/memory.py`: `solve_modes_memory` options.
- `tests/test_api.py`, `tests/test_lobpcg.py` (memory composition).
- `docs/eigensolvers.md` (options table and the defaults audit), `README.md` (one example with options), `docs/ROADMAP.md` (PR 13c row and status), `docs/open_items.md` (decisions below).

## Tests

1. Forwarding, by spies on the builders (`unstructured`, `stencil`, `gmg`, `amg`, `lobpcg`): every allowed key reaches the right builder with its value, for every pre-estimate build and the final one; defaults unchanged when the dicts are `None` (the spy sees exactly today's arguments).
2. Effect, against references: `operator_options={"mode": "otf"}` gives the same λ as the default to $10^{-10}$ relative and MAC ≥ $1 - 10^{-8}$ against `reference_modes` (hex20 box, clamped); `precond_options={"max_coarse": 100}` builds more AMG levels than the default (the spy records the returned hierarchy's level count) and gives the same modes; `{"degree": 3}` on GMG converges with the same modes; `X0` equal to the converged `V` converges in ≤ 2 iterations.
3. `sigma_options`: `{"rounds": 1}` stops after one round (spy count); `{"tol": 1e-3, "maxit": 50}` gives σ within $10^{-3}$ relative of λ₁ (SciPy) on the clamped box; given with `sigma=` or `precond="jacobi"` it raises.
4. Validation: an unknown key in each dict raises `ValueError` naming the key and listing the allowed ones; a voxel-only key (`slab`) on an unstructured mesh, an amg-only key (`max_coarse`) with gmg, any key with jacobi raise; values outside the builders' range raise from the builder.
5. `solve_modes_memory`: `operator_options={"mode": "otf"}` equals the composition with `unstructured_memory(..., mode="otf")`; `slab` on a voxel mesh likewise; unknown keys raise.

## Docs

- `docs/eigensolvers.md`: "Options" section with the full table (option, component, default, why robust, measured basis), the option dicts, an example; the tests list.
- `README.md`: the solve example gains one line with `operator_options` / `precond_options`.
- `docs/ROADMAP.md`: PR 13c row (user request, inserted before PR 14) and status line.
- `docs/open_items.md`, decisions without user input: option dicts rather than flat keywords (name clash of `chunk`, keeps the signature stable as components gain options); `nonsingular`, `dtype` and aggregate reuse not in the dicts; memory estimates at default precond settings.

## Not covered

- Passing a pre-built operator or preconditioner object to `solve_modes` (the building blocks already compose; revisit if a user needs it).
- New defaults: none change.
