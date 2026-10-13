# Eigensolvers

`jaxmodal.eigen.reference_modes` (module `eigen/reference.py`) is the SciPy reference eigensolver on assembled matrices; `jaxmodal.linalg.count_below` (module `linalg/inertia.py`) is the exact dense inertia count (PR 8a). They are oracles for the matrix-free LOBPCG and for later PRs; both run on the host on the SciPy matrices from `assemble`. `jaxmodal.eigen.lobpcg` (module `eigen/lobpcg.py`, PRs 9a and 9b) is the matrix-free solver, with `jacobi` (defined in `precond/jacobi.py` since PR 10, re-exported here), `sigma_estimate`, chunked solves and the estimator `jaxmodal.memory.lobpcg_memory`; see [LOBPCG](#lobpcg). `jaxmodal.eigen.solve_modes` (PR 9c) is the public entry point on top of it; see [Public API](#public-api-solve_modes). `jaxmodal.eigen.count` (PR 14a) is the stochastic completeness count; see [Completeness count](#completeness-count).

```python
from jaxmodal.eigen import reference_modes
from jaxmodal.linalg import count_below
from jaxmodal.operators import assemble, dof_mask

K, M = assemble(mesh, materials)
free = dof_mask(mesh, bcs)
lam, V = reference_modes(K, M, 20, free, points=mesh.points)  # lam (20,) ascending, V (n_dof, 20)
n = count_below(K, M, sigma, free)  # number of eigenvalues of (K_ff, M_ff) below sigma
```

## Public API (`solve_modes`)

```python
from jaxmodal.eigen import solve_modes

lam, V, info = solve_modes(mesh, materials, {"xmin": "xyz"}, 20, orientations, mass="lumped")
f = info["frequencies"]  # Hz for SI inputs
```

`solve_modes(mesh, materials, bcs, n_modes=None, orientations=None, *, mass="consistent", nodal_mass=None, tol=1e-6, maxit=500, seed=0, chunk=None, guards=None, basis_dtype="float64", low_memory=False, precond="auto", sigma=None, precond_dtype="float64", X0=None, operator_options=None, precond_options=None, sigma_options=None, band=None, verify=None, count_options=None)` (module `eigen/api.py`, PRs 9c, 11b, 13b, 13c and 14b) is the modal-analysis entry point. It returns the lowest `n_modes` elastic modes, or with `band=(0, f_max)` every elastic mode up to $f_{max}$ with an optional completeness check ([Frequency band](#frequency-band-0-f_max)). Every option of the operator, the preconditioner, the σ pre-estimate, `lobpcg` and the count is reachable from it ([Options](#options)):

1. `n_modes` / `band`, `verify` and the option dicts' keys are checked first ([Options](#options)), then `dof_mask(mesh, bcs)`, so a typo or a bad BC dict fails before the operator build ([`boundary_conditions.md`](boundary_conditions.md)).
2. **Dispatch:** a voxel mesh (`mesh.voxel is not None`) gets `stencil`, any other mesh `unstructured` (stored mode unless `operator_options` sets `mode`), with `orientations`, `mass`, `nodal_mass` and `operator_options` forwarded (`nodal_mass` `(n_nodes,)` is added to M, isotropically). Both builders run the host checks (`check_materials`; `check_elements` or the voxel grid check) before any trace. `info["operator"]` is `"stencil"` or `"unstructured"`.
3. **Rigid deflation:** `rigid_modes(mesh, free)` gives the rigid modes the BCs leave (per face-connected part), M-orthonormalized with the operator's own mass (nodal masses included) and passed to `lobpcg` as `R`. The lowest `n_modes` *elastic* modes come back; `info["rigid"]` is the M-orthonormal rigid basis $(n_\mathrm{dof}, k)$, NumPy fp64, and `info["n_rigid"]` $= k$ (for PR 17's effective-mass sum rule).
4. **Preconditioner** (PRs 11b, 13b), by `precond`:
   - `"auto"` (default): `"gmg"` on a voxel mesh, `"amg"` otherwise.
   - `"gmg"`: the voxel geometric multigrid ([`preconditioners.md`](preconditioners.md#voxel-geometric-multigrid-gmg)) built at σ: the hint `sigma` (a finite float ≥ 0) if given, else the [multigrid-refined pre-estimate](#σ-pre-estimate). `precond_dtype="float32"` runs its V-cycle in fp32. `"gmg"` on a non-voxel mesh raises `ValueError`.
   - `"amg"`: the unstructured AMG ([`preconditioners.md`](preconditioners.md#unstructured-amg-amg)), `amg(op, free, σ, mesh.points, nonsingular=(k == 0))`, at the hint or the pre-estimate as for GMG. With no rigid modes left ($k = 0$: `rigid_modes` found none per face-connected part, and `solve_modes` passes no `s_K`, so every `s_K` is 1) $A_P(0)$ is nonsingular, and `nonsingular=True` lets the pre-estimate start at σ = 0. fp64 only; `"amg"` on a voxel mesh raises `ValueError` (GMG is better there: [SIMP study](preconditioners.md#unstructured-amg-amg)).
   - `"jacobi"`: `jacobi(op, free, sigma or 0.0)`; no pre-estimate (σ changes Jacobi's iterations by < 3%), and only `precond_dtype="float64"`.
   - A negative or non-finite `sigma`, an unknown name or dtype raise `ValueError`; `sigma=0.0` on a body that `gmg` cannot prove nonsingular, or that has rigid modes (AMG), raises `SingularShiftError` (the hint is never replaced).
5. `lobpcg` with that preconditioner; the solver options and `X0` are forwarded unchanged ([LOBPCG](#lobpcg)). With `band`, its band mode, then the count ([Frequency band](#frequency-band-0-f_max)).

- **Returns:** `lobpcg`'s `lam`, `V` and `info`, plus `operator`, `n_rigid`, `rigid`, `frequencies` $= \sqrt{\max(\lambda, 0)}/(2\pi)$, `precond` (the name used), `sigma` (the shift the preconditioner was built at; 0.0 for Jacobi without a hint), `sigma_iterations` (the pre-estimate's LOBPCG iterations, 0 when skipped) and `time["precond"]` (rigid modes, pre-estimate and final build; `time["setup"]` stays `lobpcg`'s).
- **Errors:** those of `dof_mask`, the builders and `lobpcg`, and from the rigid-mode setup: parts that touch only at an edge or vertex (hinged) or nonconformingly raise from `parts`, and more than `MAX_PARTS` = 10 parts raise from `rigid_modes` (R is dense).
- **Slender models:** voxel meshes get GMG, whose iterations stay low on slender beams and grow like $h^{-1}$ on one-element-thick plates; unstructured meshes get AMG, low on slender beams and bulky bodies but weak on flat elements (a single-layer hex20 plate needs about 36 PCG iterations at aspect 1 and does not converge at aspect 10; solid-shell elements, M4). See `preconditioners.md`.

Measured (isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$; 10 modes, tol $10^{-10}$; M3 Max CPU):

| mesh | BCs | operator | rigid | iterations | time | max rel. λ error |
|---|---|---|---|---|---|---|
| hex20 box $6\times5\times4$ | `xmin` clamped | unstructured + AMG | 0 | 21 (+4 σ) | 0.18 s (warm) | $1.6\cdot10^{-13}$ |
| hex20 box $6\times5\times4$ | free | unstructured + AMG | 6 | 23 (+10 σ) | 0.24 s (warm) | $1.8\cdot10^{-14}$ |
| voxel $10\times8\times6$ | `zmin` clamped | stencil + GMG | 0 | 22 (+4 σ) | 0.16 s (warm) | $6.6\cdot10^{-14}$ |
| voxel $10\times8\times6$ | free | stencil + GMG | 6 | 22 (+5 σ) | 0.18 s (warm) | $5.5\cdot10^{-14}$ |

The host steps at the acceptance sizes (hex20 $44^3$, voxel $87^3$) take under 3 s together: element check 0.05–0.09 s, operator build 2.0 / 0.2 s, `rigid_modes` 0.1 / 0.5 s, `m_orthonormalize` 0.1 / 0.2 s.

### Options

The defaults are the robust choice for most problems, measured in PRs 9–13; every other option of the building blocks is reachable through three dicts and `X0`. `None` (the default) means the builder's own defaults.

```python
lam, V, info = solve_modes(
    mesh,
    materials,
    {"xmin": "xyz"},
    20,
    operator_options={"mode": "otf"},  # on-the-fly operator: less memory
    precond_options={"degree": 3, "max_coarse": 2000},
    sigma_options={"rounds": 2},
)
```

- `operator_options`: forwarded to `unstructured` (keys `mode`, `path`, `chunk`, `range_nodes`) or `stencil` (`slab`) ([`operators.md`](operators.md)). `operator_options["chunk"]` is elements per scan step; the top-level `chunk` is modes per locking chunk.
- `precond_options`: forwarded to every pre-estimate build and the final build: `gmg` (`degree`, `coarse_degree`, `ratio`), `amg` (`degree`, `ratio`, `max_coarse`), `jacobi` (none) ([`preconditioners.md`](preconditioners.md)). With `"auto"` the keys of the resolved name apply.
- `sigma_options`: the [pre-estimate](#σ-pre-estimate)'s tuning, `tol`, `maxit` (per round), `rounds`, `drop` (rebuild while σ drops by more than this factor), `estimate_maxit` (`sigma_estimate`'s cap for the fallback start). They raise `ValueError` with a `sigma` hint or `precond="jacobi"`, which run no pre-estimate. `seed` also seeds the pre-estimate.
- `X0`: an $(n_\mathrm{dof}, j)$ start block, $j \le$ the block size (chunk + guards); it seeds the first chunk (other columns random) and is deflated against the rigid modes. With the default single chunk, a previous `V` is a valid warm start (1 iteration instead of 16, hex20 $8^3$, 10 modes; `benchmarks/results/options_study.json`). It is the caller's array: one extra $(n, j)$ block, not in `solve_modes_memory`.

An unknown key raises `ValueError` naming it and the allowed keys; values are checked by the builders (`ratio > 1`, `max_coarse` an int ≥ 1, `degree` an int ≥ 1, ...) and by `solve_modes` for `sigma_options` (`tol > 0`; `maxit`, `rounds`, `estimate_maxit` ints ≥ 1; `drop > 1`). Kept internal on purpose ([`open_items.md`](open_items.md)): `lobpcg`'s `R` and `precond` (set by `solve_modes`), `amg`'s `nonsingular` (derived from the rigid-mode count) and `aggregates` / `return_aggregates` (state; compose the building blocks), `gmg`'s `dtype` (the top-level `precond_dtype`; the pre-estimate builds stay fp64), `pmg` (test scale, superseded by `amg`), `block_jacobi` / `chebyshev` alone (without a coarse level they gain nothing, PR 10).

| option | default | why it is the robust choice |
|---|---|---|
| `precond` | `"auto"`: GMG on voxel meshes, AMG otherwise | 16 against 283 LOBPCG iterations for Jacobi (hex20 $8^3$, 10 modes, tol $10^{-6}$, `benchmarks/results/options_study.json`). Known limits ([`preconditioners.md`](preconditioners.md#unstructured-amg-amg)): AMG does not converge on flat elements (hex20 plate $16^2\times1$ at aspect $a = 10$: > 1000 PCG iterations; solid-shell elements, M4 / PR 18) and is slow on SIMP fields at their own $\lambda_1$ (790–910 PCG iterations against GMG's 87; use a voxel mesh and GMG) |
| `sigma` | refined pre-estimate | final σ within $1.001\lambda_1$ on every tabulated case except the unfiltered binary SIMP field ($390\lambda_1$; LOBPCG does not converge on that field at any σ, [σ pre-estimate](#σ-pre-estimate)) |
| `precond_dtype` | `"float64"` | always converges; `"float32"` (GMG only) is faster on voxel meshes (51 against 61 s at $87^3$) |
| `tol` | $10^{-6}$ | the roadmap's acceptance tolerance (λ-scaled residual); the Rayleigh quotient's λ error is of order residual² |
| `chunk`, `guards` | one chunk, `guards` $\max(5, \lfloor\mathrm{chunk}/5\rfloor)$ | the 9-block step; chunks bound memory for many modes |
| `basis_dtype`, `low_memory` | `"float64"`, `False` | full accuracy; fp32 basis and the low-memory step trade accuracy or time for memory |
| `degree` | 2 | fastest: warm LOBPCG 7.0 / 8.8 / 10.3 s at degree 2 / 3 / 4 for 17 / 16 / 15 iterations (AMG, hex20 $24^3$) and 2.9 / 3.5 / 3.9 s for 16 / 15 / 14 (GMG fine level, voxel $32^3$); 20 modes, `benchmarks/results/options_study.json` |
| `coarse_degree` (GMG) | 4 | SIMP 116 → 87 and plate 186 → 136 iterations over degree 2 |
| `ratio` | 30 | the Chebyshev smoothing interval $[\lambda_\max/30, \lambda_\max]$ |
| `max_coarse` (AMG) | 1000 | dense coarsest factor of 8 MB. Known limitation: on single-layer plates 2000 gives 15 against 19 iterations, a coarse-mesh-size effect rather than a fix (flat elements; remedy: solid-shell elements, M4 / PR 18); above 1000 the memory estimate's step term is not a bound (+1.3 blocks at 4000) |
| `mode` | `"stored"` | 0.196 s against 0.555 s per apply (hex20 1M DOF); `"otf"` when the stored operator does not fit (e.g. 10M DOF on one 80 GB GPU, estimated about 60 against 71.5 GB, not measured) |
| `path`, `chunk`, `range_nodes`, `slab` | auto per backend | measured layout defaults ([`operators.md`](operators.md)) |
| `sigma_options` | `tol` 0.1, `maxit` 20, `rounds` 4, `drop` 3, `estimate_maxit` 10 | [σ pre-estimate](#σ-pre-estimate) |
| `band` | `None` (the lowest `n_modes`) | `(0, f_max)`: every mode up to $f_{max}$ ([Frequency band](#frequency-band-0-f_max)). Without `n_modes` the expected count is $4\cdot$`chunk` = 160: the basis holds 256 fp64 columns up front, 2 GB per million DOF; pass `n_modes` for small bands on large meshes |
| `chunk` with `band` | $\min(40, \lceil 1.25\,n_\text{modes}\rceil + 2)$, 40 without `n_modes`, at most $n_\text{free} - n_\text{rigid}$ minus the guards (8 by default) | one chunk normally holds the band and a mode beyond it (one chunk is faster than several, PR 9b); 40 (block 48) is the width the memory model is calibrated at. Above about 2M DOF a block of 48 may not fit: check `solve_modes_memory(band=...)` and pass `chunk` |
| `verify` | `None`: count when predicted to cost no more than the solve | the roadmap's cost rule; `True` forces the count, `False` reports only its predicted cost |
| `count_options` | `count`'s defaults; `block` from the solve's memory estimate | `n_probes`, `seed`, `tau`, `degree`, `block`, `maxit` ([Completeness count](#completeness-count)). An explicit `degree` is used as given, never capped at `MAX_DEGREE` |

### Frequency band [0, f_max]

```python
lam, V, info = solve_modes(mesh, materials, {"xmin": "xyz"}, band=(0, 500.0))  # f <= 500 Hz
info["count"]["passed"]  # when the count ran: no mode below the cut is missing
```

`band=(0, f_max)` (Hz for SI inputs; interior bands $f_1 > 0$ raise `NotImplementedError`, PR 15) returns every elastic mode with $f \le f_{max}$, possibly none, plus whole clusters reaching into the band. `n_modes` is then optional: the expected count, which sizes the locked basis (default $4\cdot$`chunk`) and the default `chunk`.

- **Band mode** (`lobpcg._solve(..., lam_stop=(2π f_max)²)`): [chunks and locking](#chunks-and-locking) as for `n_modes`, every chunk lockable. After a chunk converges with $j$ columns (already extended over a boundary cluster), $i$ = the number of its Ritz values $\le \lambda_{stop}(1 + 100\,t)$ (a neighbour within the 100 $t$ window reaches into the band, as in the roadmap's band edges), extended over clusters (relative gaps $< 100\,t$). If $i < j$ the solve stops: modes $1..i$ are in the band, $i+1..j$ are converged modes beyond it. Otherwise all $j$ are in the band, the chunk is locked and the next starts. The final chunk's converged columns are re-orthonormalized against the basis as at a lock, so all converged modes are M-orthonormal for the count.
- **Edge above the spectrum:** a new chunk that would need more than $n_\text{free} - n_\text{rigid}$ columns raises `ValueError` ("the band needs more modes than the mesh provides"): $f_{max}$ above the largest eigenvalue, or a band holding nearly every mode.
- **Capacity:** past the expected count the device basis grows to $1.5\times$, at least room for the next lock (new array, copy, old freed: a transient of old + new, and the step recompiles for the new shape) and the host `V` grows the same way (a host transient of old + new). `info["capacity"]` reports the final basis columns. Pass `n_modes` near the expected count to avoid growth.
- **Returns** `lam`, `V` (the band modes, a compact copy: the solve's host buffer is freed) and `info` with `band`, `chunk`, `capacity`, `n_beyond` (converged modes beyond the edge), `residuals` / `thresholds` / `frequencies` of the band modes, `t_iteration` and `count`. A failed chunk ends the solve (`converged` False) and no count runs (`verify=True` warns); `lam`, `V` then hold the failed chunk's Ritz pairs at or below the edge, unverified and possibly incomplete.

**Count and `verify`.** The cut follows the roadmap's band edges: $\lambda_\text{last}$ = the last returned mode (the band edge $(2\pi f_{max})^2$ when none is returned), $\lambda_\text{next}$ = the first converged mode beyond, $\lambda_c$ their midpoint. `count` deflates every converged mode (returned and beyond: the tail model needs all modes up to $\lambda_\text{next}$) and the rigid modes, with the fp64 multigrid (or Jacobi) built at $\lambda_c$ (AMG reuses the solve's aggregates, exact for a shift change). The solve's preconditioner is released first.

- **Degree and block.** $d$ from `count_degree` (or `count_options["degree"]`); the probe block $b$ = `probe_block` with the budget = `solve_modes_memory(...)` of this solve minus the host copy of the converged modes ($8\,n\,k$; host and device share memory on the Mac) and the kept AMG aggregates, the λ_c multigrid counted at fp64. A budget below one probe (possible after an fp32 V-cycle solve): `verify=None` skips with the reason, `verify=True` returns the modes with `ran` False, the reason and a warning (the check runs after the solve; raising would discard it). `count_options={"block": 1}` runs it anyway with a larger peak.
- **A-priori bound** (free, before any build): a count step runs at least 7 PCG iterations at width $b$ (7–14 measured at $\lambda_c$, PR 14a), each at least one operator and one preconditioner apply; a LOBPCG iteration costs at most about 3 of these applies at width $m$ (two operator and one preconditioner apply plus its dense work, at most about 1.5× the applies at $m = 48$). So a step costs at least about $2.3\,(b/m)\,t_\text{it}$, and the bound is
  $$t_\text{bound} = c\; n_\text{blocks}\, d\; \frac{b}{m}\; t_\text{it},\qquad c = 2\ (0.3\ \text{when}\ \tau\ \text{is looser than the rule: one PCG iteration}),$$
  with $t_\text{it}$ = `info["t_iteration"]`, chunk 0's mean step time over its iterations after the compiling first one (the step calls only; chunk 0 has no projector). Measured ([`band_study.py`](../benchmarks/band_study.py)): the real warm step is 6.4× the bound's step on hex20 $16^3$ with AMG and 4.7× on voxel $32^3$ with GMG (steel, $b = 10$, $m = 48$; `benchmarks/results/band_study.json`), and about 7× in `test_calibration` (hex20 $12^3$, $d = 50$); on the 2.7k-DOF reference box at $m = 11$, where LOBPCG's fixed costs dominate, it is 1.3. Without a warm iteration (chunk 0 converged at once) there is no bound (`predicted_s` None, `predicted_kind` "unknown").
- **`verify=False`:** no build, no count; `info["count"]` reports `degree`, `block`, `predicted_steps` $= n_\text{blocks}\,d$ and the bound.
- **`verify=True`:** builds at $\lambda_c$ and runs the count (no calibration). A rule degree above `MAX_DEGREE` gives `count`'s NaN, unverifiable (no build).
- **`verify=None`:** skips when $d$ exceeds `MAX_DEGREE` or the bound exceeds the solve time (σ pre-estimate, preconditioner and LOBPCG; no build). Otherwise it builds at $\lambda_c$ and calibrates: two one-block counts at the real $d$, $\tau$ and $b$ capped at 1 step (compiles the real polynomial size class) and 3 steps (timed): $t_\text{step}$ = the recurrence time per step, $t_\text{block}$ = sampling and projections. The predicted counting cost is $t_\text{build} + t_\text{calibration} + n_\text{blocks}(d\,t_\text{step} + t_\text{block})$, and the count runs iff it is at most the solve time.

`info["count"]` is `count`'s info (when it ran) plus `ran`, `reason`, `estimate`, `lam_c`, `gap`, `degree`, `block`, `predicted_steps`, `predicted_s`, `predicted_kind` ("bound", "calibrated" or "unknown"), `solve_s`, `build_s`, `calibration_s` (and `calibrated_step`, `calibrated_block`). PR 14a's measurements apply: a typical cut costs 10²–10³ solves, so `verify=None` counts mainly when the band edge sits in a large gap ($g \gtrsim 0.6$) or the band is small.

No strategy selection yet: LOBPCG from 0 is the only band strategy, and a band whose basis outgrows the device fails with an out-of-memory error ([shift-invert slices](#shift-invert-slices) exist since PR 15a; PR 15c adds slicing and the choice). `solve_modes_memory(mesh, bcs, k, band=(0, f_max))` estimates a band solve expecting $k$ modes. `memory.band_probe_block(mesh, bcs, n_modes, n_probes, host_bytes, *, n_rigid, **options)` is the count's budget rule: the largest probe block within that estimate minus `host_bytes`, with the λ_c multigrid at fp64; it raises below one probe.

### Mac acceptance budgets

PRs 11 and 13 accept on wall-time budgets: **10 min** for a 2M-DOF voxel solve, 10 modes (PR 11, also with fp32 V-cycles), and **15 min** for a 1M-DOF hex20 solve, 20 modes (PR 13). The run is specified as tol $= 10^{-6}$, the 9-block step, an fp64 basis, and wall time including setup, the σ pre-estimate and compilation. The peak criteria (within 20% of the estimators) are separate.

Throughput at the acceptance sizes (steel, Jacobi; setup is `lobpcg`'s own, clamped: a free body adds about 1 s at 2M DOF for $R$ and $MR$, and the operator build is separate):

| case | DOF | $m$ | $t_K$ | $t_M$ | 9-block step | low-memory step | setup |
|---|---|---|---|---|---|---|---|
| hex20 $44^3$, clamped, 20 modes | 1.08M | 25 | 140 ms | 167 ms | 730 ms | 1165 ms | 2.6 s |
| voxel $87^3$, 10 modes | 2.04M | 15 | 160 ms | 162 ms | 786 ms | 1248 ms | 1.0 s |

Model for a preconditioned solve:

$$T \approx T_\text{setup} + T_\sigma + T_\text{compile} + N_\text{it}\,(t_\text{step} + c_V\,t_A + t_\text{axpy}),$$

with $t_A = t_K + t_M$ (the preconditioners approximate $K + \sigma M$, $\sigma > 0$, with no fused apply); $c_V$ the fine-level applies per V-cycle in $t_A$ units (Chebyshev degree 2 pre- and post-smoothing plus the residual: 5; coarse levels about 0.5 for the voxel GMG and about 1 for hex20 p-MG plus hex8 AMG); $t_\text{axpy} \approx 0.1$ s the smoother's vector updates; and $N_\text{it}$ the LOBPCG iterations, assumed 150 before PR 11 measured them.

| case | per iteration | $N_\text{it} = 150$ | setup + σ + compile | total | budget | margin |
|---|---|---|---|---|---|---|
| voxel 2M, 10 modes (assumed) | $0.79 + 5.5 \cdot 0.32 + 0.1 = 2.65$ s | 6.6 min | about 1.5 min | about 8 min | 10 min | 1.25× |
| voxel 2M, measured (PR 11b), fp64 / fp32 V-cycle | 2.7 / 2.1 s ($c_V$ = 5.2 / 3.2) | $N_\text{it}$ = 16: 43 / 33 s | 17.5 / 18.1 s | 61 / 51 s | 10 min | 10× / 12× |
| hex20 1M, 20 modes | $0.73 + 6 \cdot 0.31 + 0.1 = 2.69$ s | 6.7 min | about 1.5 min | about 8 min | 15 min | 1.8× |
| hex20 1M, measured (PR 13b, AMG) | 2.6 s ($c_V$ = 5.6) | $N_\text{it}$ = 19: 49 s | 21 s | 72 s | 15 min | 12× |

The measured hex20 run (`amg_study.acceptance`, one cold process, compilation included): operator and `lobpcg` setup 2.7 s; `time["precond"]` 18.1 s, i.e. the σ pre-estimate (the σ = 0 build plus 6 LOBPCG iterations; σ = 1.0000002 λ₁) and the final `amg` (about 4 s each: the production hex8 assembler 0.7 s, the SA loop 1.8 s, the rest the device export and the fine level); LOBPCG 49 s for 19 iterations. Warm peaks 7.46 / 7.43 GB against the 7.45 GB `solve_modes_memory` estimate; cold 7.67 GB. The budgets keep the roadmap's figures rather than the model's, which leaves margin for the two unmeasured inputs ($N_\text{it}$, the V-cycle cost). PR 13b measured $N_\text{it}$ = 19, so the hex20 budget stands; a fused $K + \sigma M$ apply ($t_A \approx t_K$) remains the first lever if a later case needs one. PR 11b measured the voxel case (`docs/preconditioners.md`, Acceptance): GMG needs 16 iterations, not 150, and the σ pre-estimate (13–15 s at 2M DOF) replaces the 73 s Jacobi `sigma_estimate`.

## Reference solver (`reference_modes`)

`reference_modes(K, M, n_modes, free=None, sigma=None, points=None)` returns the lowest `n_modes` eigenpairs of $K_f v = \lambda M_f v$, where $K_f$, $M_f$ are the free blocks: $\lambda$ ascending, and $V$ of shape `(n_dof, n_modes)` with constrained rows exactly 0 and $V^\top M V = I$. PR 9 compares its masked LOBPCG modes against these padded modes.

- **Restriction, not masking** (`restrict`, [`boundary_conditions.md`](boundary_conditions.md)): the masked operators keep the constrained DOFs as zero rows, which would add spurious zero eigenvalues and spurious zero pivots.
- **Inputs.** `1 <= n_modes` and `n_modes + 1 < n_free` (ARPACK needs $k < n$, and one extra mode is computed for the check); `sigma` is `None` or a finite value $< 0$; `points` is `None` or the `(n_nodes, 3)` `mesh.points` that `K`, `M` were assembled from (DOF $3n + i$). Anything else raises `ValueError`. Points of another mesh of the same size cannot be detected; they only slow the factorization and never change the result.

### Shift-invert at a negative shift

`eigsh(K_f, n_modes + 1, M=M_f, sigma=σ, OPinv=…, tol=0, v0=…)` with $\sigma < 0$, so $K_f - \sigma M_f$ is SPD and its factorization needs no pivoting. The start vector is seeded (`default_rng(0)`), so runs are reproducible.

**ARPACK fails silently when $|\sigma|$ is far above the wanted eigenvalues.** Shift-invert converges at a ratio of about $(\lambda_{k+1} - \lambda_k)/|\sigma|$ when $|\sigma| \gg \lambda_k$. In that regime `eigsh` can return wrong eigenpairs without raising: higher modes in their place, or missing rigid modes. Measured, with $\bar s = \mathrm{mean}\,\mathrm{diag} K_f / \mathrm{mean}\,\mathrm{diag} M_f$:

- a free consistent hex20 box at $\sigma = -\bar s$ (about $160\lambda_7$): relative errors of 1.3, rigid modes missing;
- a free hex20 plate with $L/t = 100$ at $\sigma = -10^{-2}\bar s$: errors of 2.1;
- the free hex27 plate $8\times8\times1$ with $L/t = 200$ at $\sigma = -10^{-2}\bar s$ ($8\cdot10^6\lambda_7$): wrong pairs after 40 s (`test_arpack_failure` records it). At $-10^{-3}\bar s$ and $-10^{-4}\bar s$ the same plate comes out right.

On thin plates $\lambda_7/\bar s$ is tiny ($2\cdot10^{-8}$ at $L/t = 100$, $1.2\cdot10^{-9}$ at $L/t = 200$; a free hex20 box has $6\cdot10^{-3}$), so no fixed fraction of $\bar s$ works. Too close to 0 fails too on a free body: $K_f - \sigma M_f$ is nearly singular along the rigid modes, and the elastic eigenvalues lose about $\varepsilon\lambda_7/|\sigma|$.

**σ rule** (`sigma=None`):

1. **Pass 1** at $\sigma_0 = -\sqrt\varepsilon\,\bar s$ (unit-independent). It needs only rough $\tilde\lambda$, and works while $|\sigma_0|$ is not far above $\lambda_k$ ($k$ = `n_modes + 1`); measured on plates down to $\lambda_7/\bar s = 10^{-9}$ ($L/t = 1000$).
2. **Pass 2**, if pass 1 finds near-zero eigenvalues ($\tilde\lambda_1 < 100|\sigma_0|$: rigid modes, a nearly singular $K_f$, or a thin structure whose low modes lie below $100|\sigma_0|$), at $\sigma = -0.3\,\tilde\lambda_k$. Then $|\sigma| < \lambda_k$, and the rigid-mode amplification is at most $\lambda_k/|\sigma| \approx 3$ (see below). Only if every wanted mode is rigid-mode noise ($\tilde\lambda_k \le 100\,\varepsilon\,\bar s$, which then raises `ValueError` "no gap") does pass 1 stand; on very thin structures that request can instead fail loudly (`ArpackNoConvergence`, or the residual check).

For SPD $K_f$ (clamped, sliding), pass 1 is as accurate as a pass near $-\lambda_k$.

**Why $-0.3\tilde\lambda_k$ (PR 8c).** In shift-invert the shifted rigid modes dominate the operator ($1/|\sigma|$ against $1/(\lambda - \sigma)$), and ARPACK's accuracy is relative to the largest Ritz value, so the elastic residuals grow by about $\lambda/|\sigma|$. PR 8a used $-0.01\tilde\lambda_k$ (a factor of about 100), which passed its small tests but failed on the free voxel sphere with 32 cells per diameter (59.4k DOF, `n_modes = 18`): residual 172ε against the 100ε bound, so `reference_modes` raised. The failure depends on size and $k$ (spheres with 12, 16, 20, 32 cells: 24, 14–19, 49, 172ε at 18 modes; 50ε at 32 cells with 22 modes). Pass-2 factors 0.01, 0.1, 0.3 and 1 give 172, 8.0, 3.6 and 3.7ε on that sphere (one pass-2 solve takes 12.0–13.1 s, about 4% more at 0.3 than at 0.01; the whole call with certification about 34 s), and 5.5, 1.5, 1.1, 1.5ε on a free hex20 $12^3$ cube. 0.3 keeps $|\sigma| < \lambda_k$, away from the opposite failure below (σ far below $\lambda_k$).

The rule is validated against dense `eigh` up to $\kappa = \lambda_\max/\lambda_7 \approx 8\cdot10^9$, and certifies on the free hex27 plate $8\times8\times1$ with spacing $(1, 1, 0.005)$ ($L/t = 1600$, $\kappa = 3\cdot10^{13}$) for `n_modes` 7, 10, 12, 17, 21 and 25; there the floor $f_\lambda = 10\,\varepsilon\,\kappa \approx 0.07$ for the lowest elastic mode, and the measured error against dense `eigh` is $0.09 f_\lambda$ (lowest 12; the same under PR 8a's $-0.01$ and the current $-0.3$). (Without the thin-structure case of pass 2, pass 1 at $|\sigma_0| \approx 10^5\lambda_k$ fails the residual check there, at 700–7800ε.)

**Explicit `sigma`** skips the rule (one pass), e.g. $-0.3\lambda_k$ from a coarser run, for PR 16's solid shells ($\kappa \approx 10^{12}$) if the rule ever fails there. A $\sigma$ far below $\lambda_k$ raises the residuals (45–90ε on the $L/t = 200$ plates at $-10^{-2}\bar s$) and can trip the residual check, loudly.

### Certification

Every result is checked; a failure raises `RuntimeError` instead of returning wrong modes. With the computed $\lambda_1 \le \ldots \le \lambda_{n+1}$ ($n$ = `n_modes`):

1. **Orthonormality:** $\|V^\top M_f V - I\|_\max \le 10^{-10}$.
2. **Residuals:** every mode has $\|K_f v - \lambda M_f v\|_\infty \le 100\,\varepsilon\,(\|K_f\|_\infty + |\lambda|\,\|M_f\|_\infty)\,\|v\|_\infty$ (measured maximum $10.65\varepsilon$). With accurate, M-orthonormal eigenvectors, a count equal to $j$ proves that the first $j$ vectors span the lowest invariant subspace; together, 1 and 2 rule out a ghost copy standing in for a missing mode.
3. **Count.** Let $j$ be the largest index $\le n$ with $\lambda_{j+1} - \lambda_j > \tau = \max(10^{-6}|\lambda_{n+1}|,\ 100\,\varepsilon\,\bar s)$: the last gap at or below the requested count. Rigid-mode noise ($\lesssim 4\varepsilon\bar s$), exact FE degeneracies and the splitting of degenerate pairs by `eigsh` (up to $2.6\cdot10^{-5}$ relative, $\le 1.7\varepsilon\bar s$ absolute, on very thin square plates) count as clusters; real gaps are far wider. $K_f - \mu M_f$ at $\mu = (\lambda_j + \lambda_{j+1})/2$ is factored, and by Sylvester's law of inertia the number $c$ of negative pivots is the number of eigenvalues below $\mu$; $c \ne j$ raises `eigsh missed modes: c eigenvalues below μ, j found`. No gap at all (e.g. `n_modes=3` on a free body, inside the rigid cluster) raises `ValueError` asking for more modes.

A request that ends inside a cluster of near-singular soft modes (soft springs, near-mechanisms, $\lambda \lesssim 10^{-9}\bar s$) can fail the residual check, loudly; ask for modes past the cluster.

When `n_modes` ends inside a cluster, the count certifies modes $1..j$ only; the tail cluster's members are still certified as eigenpairs by the residuals, and a missed member is caught if $\lambda_{n+1}$ lies past the gap.

**Sparse inertia from SuperLU.** The count reuses the solver's factorization: `splu(A[p][:, p], permc_spec=…, diag_pivot_thresh=0, options=dict(SymmetricMode=True))` keeps `perm_r == perm_c`, so $PAP^\top = LU$ with $U = DL^\top$, and $c$ = the number of negative entries of $\mathrm{diag}\,U$. If SuperLU ever pivots off the diagonal (`perm_r != perm_c`), the count is unavailable and the routine raises. This is an unpivoted $LDL^\top$ of an indefinite matrix: element growth is not controlled, and `perm_r == perm_c` guards the structure, not the growth. A wrong count is possible in principle and would almost always show up as a loud mismatch. It matched dense `eigh` in every case tried (free box, clamped hex27, plates; consistent and lumped; both orderings; up to 747 negative pivots).

### Ordering

- **With `points`: geometric nested dissection.** The free DOFs are bisected recursively at the median of the widest coordinate axis (`c <= median`, which keeps whole node planes on one side so the separator is one clean layer; if all DOFs tie at or below the median, `c < median`; if that is empty too, the subset becomes a leaf). The separator is the left DOFs coupled to the right side, read from the CSR rows of $|K_f|$. Children are ordered first, then the separator, down to leaves of `ND_LEAF = 64` DOFs; SuperLU then factors with `permc_spec="NATURAL"`. The ordering takes 0.04–0.15 s. A rank split (equal halves) cuts through node planes and costs 30–65% more factorization time.
- **Without `points`:** SuperLU's `MMD_AT_PLUS_A`, fine for plates, slower and denser for bulky meshes.

One factorization of the prototype's boxes (harness, SuperLU fill or peak; the DOF counts differ slightly from the cost table below, which uses other BCs):

| mesh | DOF (prototype) | `MMD_AT_PLUS_A` | geometric ND |
|---|---|---|---|
| hex20 $12^3$ | 22k | 4.7 s, 50M fill | 1.4 s, 28M |
| hex20 $14^3$ | 34.6k | 14.9 s, 106M | 4.7 s, 55M |
| hex8 $24^3$ | 43k | 15.8 s, 117M | 3.4 s, 54M |
| hex20 $16^3$ | 51k | 34.5 s, 2.12 GB | 7.2 s, 1.15 GB |
| hex8 $30^3$ | 84k | 118 s, 5.53 GB | 11.6 s, 1.57 GB |
| hex20 $20^3$ | ≈ 100k | — | 26.2 s, 2.89 GB |
| hex27 plate $40\times40\times1$ | 59k | 1.5 s, 34M | 1.7 s, 46M |

### Accuracy

The attainable accuracy is the floor $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$ ($\hat\lambda_\max$: the largest eigenvalue, from dense `eigh` or `eigsh(which="LA")`; the verification tests use `tol=1e-3`, enough for an order of magnitude) (relative), i.e. about $\varepsilon\kappa$ for the lowest elastic mode. Against dense `scipy.linalg.eigh` of $(K_f, M_f)$, up to 20 modes on free hex20 boxes (consistent and lumped), clamped hex27, sliding hex20, and free hex20/hex27 plates with $L/t = 100, 200$ ($\kappa \le 7.8\cdot10^9$):

- elastic modes: $|\lambda - \lambda_\text{dense}| \le 0.99 f_\lambda\lambda$ (each side of the comparison is floor-limited);
- rigid modes: $|\lambda| \le 0.04\cdot10\varepsilon\lambda_\max$ (dense `eigh`: up to $0.22$);
- $\|V^\top M V - I\| \le 8\cdot10^{-15}$;
- residuals $1.9$–$4.9\varepsilon$ on boxes, $4.3$–$10.65\varepsilon$ on plates (relative as in the check; on the full $K$ the constrained rows carry reaction forces).

The tests (`tests/test_reference.py`) assert $2 f_\lambda$, $10^{-12}$ and $100\varepsilon$.

### Cost and size

2 factorizations for SPD $K_f$ (one solve and the count), 3 with rigid modes (pass 1, pass 2, the count); the shifts differ, so nothing is shared. The routine peaks above one factorization, because the count's `lu.U` builds a full CSC copy of $U$ next to the factor (SciPy has no pivot accessor). The whole routine, `n_modes = 20`, orthotropic material, measured with the benchmark harness (peak over the setup):

| mesh, BCs | ordering | free DOF | time | of which the check | peak |
|---|---|---|---|---|---|
| hex20 $10^3$, free | ND | 14.9k | 3.3 s | 0.7 s | 0.42 GB |
| hex20 $16^3$, clamped | ND | 53.9k | 19.7 s | 8.4 s | 2.57 GB |
| hex20 $16^3$, free | ND | 56.4k | 36.3 s | 9.8 s | 2.73 GB |
| hex8 $30^3$, clamped | ND | 86.5k | 28.2 s | 12.2 s | 3.39 GB |
| hex8 $30^3$, free | ND | 89.4k | 49.9 s | 13.6 s | 3.56 GB |
| hex20 $20^3$, clamped | ND | 103k | 67.7 s | 30.4 s | 6.48 GB |
| hex20 $20^3$, free | ND | 107k | 120 s | 34.5 s | 6.98 GB |
| hex20 $14^3$, clamped | MMD | 36.5k | 29.6 s | 13.5 s | 2.49 GB |
| hex20 $14^3$, free | MMD | 38.5k | 53.6 s | 15.8 s | 2.78 GB |
| hex27 plate $40\times40\times1$, $L/t = 200$, free | ND | 59.0k | 7.7 s | 1.9 s | 1.26 GB |
| the same plate | MMD | 59.0k | 5.7 s | 1.4 s | 0.90 GB |

The factorizations dominate; the check is about 20–45% of the time. The largest supported size is about 100k free DOF on bulky meshes with `points` (2 minutes, 7 GB), and about 35k without. Plate-like meshes factor much faster and go further; on the plate MMD is as good as ND.

## LOBPCG

```python
from jaxmodal.eigen import jacobi, lobpcg, sigma_estimate

lam, V, info = lobpcg(op, free, 20, tol=1e-8, R=R)  # R: M-orthonormal rigid modes or None
lam, V, info = lobpcg(op, free, 20, precond=jacobi(op, free, sigma))  # any pytree callable
sigma, its = sigma_estimate(op, free, R)  # lam_1 upper bound, for preconditioners of K + sigma M
lam, V, info = lobpcg(op, free, 200, R=R, chunk=40, basis_dtype="float32", low_memory=True)
```

`lobpcg(op, free, n_modes, tol=1e-6, R=None, precond=None, X0=None, maxit=500, seed=0, *, chunk=None, guards=None, basis_dtype="float64", low_memory=False)` computes the lowest `n_modes` eigenpairs of the masked pencil $K_m v = m \odot K(m \odot v)$, $M_m$ likewise, for any operator with `apply_K`, `apply_M`, `diag_K`, `diag_M` and `n_dof` (PR 5 or PR 6).

- **Returns:** `lam` `(n_modes,)` ascending (NumPy); `V` `(n_dof, n_modes)`, a NumPy fp64 array on the host, M-orthonormal with constrained rows exactly 0; and `info` with `iterations` (total), `converged` (every chunk), `stagnated` (any chunk), `residuals` and `thresholds` (per returned mode, from explicitly recomputed products, on every exit; for locked modes, of the re-orthonormalized columns), `p_drops`, `refills`, `norm_K`, `norm_M`, `history` (max wanted residual per iteration, chunks concatenated), `chunks` (per chunk: `want`, `locked` (columns accepted; the final chunk's are returned, never inserted into the basis), `iterations`, `converged`, `stagnated`) and `time` (`setup`, `iterations`, in seconds). After a failed chunk fewer than `n_modes` come back (see [Chunks and locking](#chunks-and-locking)).
- **Deflation basis `R`:** `(n_dof, k)`, M-orthonormal (rigid modes after `m_orthonormalize` with the same operator). $MR = M_m R$ is computed once. The start block and every new direction are projected, $W \leftarrow W - R\,(MR^\top W)$, so no extra M apply is needed per iteration. Without `R`, a free body returns its rigid modes as λ ≈ 0; their λ-scaled residual is meaningless, so the solve ends at the no-progress exit (`converged` False, `stagnated` True).
- **Preconditioner `precond`:** a pytree callable `(n, j) -> (n, j)` (e.g. `jax.tree_util.Partial`), fixed, linear and SPD on the free DOFs, with masked output. It is a step argument, so new preconditioner data does not recompile. The default is `jacobi(op, free)`.
- **Chunks and block size:** `chunk` wanted modes per chunk (default `n_modes`, one chunk) and `guards` extra columns (default $\max(5, \lfloor \text{chunk}/5\rfloor)$), block size $m$ = chunk + guards for every chunk. `maxit` ($\ge 1$) counts per chunk; `X0` seeds the first chunk.
- **`basis_dtype`:** `"float64"` or `"float32"`, the device dtype of the locked basis. With fp32 the chunks after the first accept at $\max(\mathrm{tol}, 1.2\cdot10^{-7})$; use fp64 for a tighter tol.
- **`low_memory`:** the low-memory step (11 instead of 16 blocks, 1.4–1.9× the time per iteration).
- **Errors:**
  - `ValueError` if `n_modes` or `maxit` is below 1, if `chunk` is not in $[1, n_\mathrm{modes}]$, `guards` is below 1 or `basis_dtype` is neither fp64 nor fp32, if $m$ exceeds the free DOFs minus the rank of $R$, for the guard error (a cluster at a chunk boundary fills the block), if $\|R^\top M_mR - I\|_\max > 10^{-8}$ (e.g. R orthonormalized with another mass), or for a bad mask or `X0`;
  - `RuntimeError` if the start block stays rank-deficient after 3 refills, or if a column to be locked lies in the span of the locked modes;
  - `FloatingPointError` for a non-finite step.

  Non-convergence never raises: check `info`.
- **`jacobi(op, free, sigma=0.0)`:** Jacobi on $A_P = K + \sigma M$, i.e. $v \mapsto m \odot v / (\mathrm{diag}K + \sigma\,\mathrm{diag}M)$, with output 0 on constrained DOFs. The constrained diagonal $s$ of $A_P$ ([`boundary_conditions.md`](boundary_conditions.md)) never reaches the output. It lives in `jaxmodal.precond` (PR 10, with `block_jacobi` and Chebyshev, [`preconditioners.md`](preconditioners.md)); `jaxmodal.eigen.jacobi` is the same function. A concrete $\sigma < 0$ raises `ValueError`.

### Algorithm

Hetmaniuk–Lehoucq / Duersch et al. (2018), with fixed shapes: every block is $(n, m)$, and inactive columns are exact zeros tracked by boolean masks.

- **One compiled step.** `lobpcg_step(state, op, mask, precond, R, MR, B, count, consts, recompute, *, c)` is a module-level `jax.jit` (`lobpcg_step_low` for the [low-memory mode](#low-memory-mode)). `B` and `count` are the locked basis and its traced count, `c` the static projector chunk width; a single-chunk solve passes a basis with no columns, which compiles the projector out.
  - The state $(X, KX, MX, P, KP, MP, \mathrm{act}_P, \lambda)$ is donated.
  - `consts` $= (\|K\|, \|M\|, \mathrm{tol}, c)$ and `recompute` are traced.
  - Nothing is closed over: closing over the operator would embed it in the program (6.4 MB of HLO for a diagonal operator at 100k DOF, against 0.05 MB with arguments; 2.2 GB more compile memory with a real operator at 182k) and recompile on every solve.
  - $P$ starts as a masked zero block, so there is one compiled variant per shape, and a second solve with new data (`op.scaled`, a new preconditioner, a new tol) compiles nothing.
- **Setup.**
  - $\|K_m\|_2$ and $\|M_m\|_2$ come from 30 power iterations each.
  - The start block (random, or `X0` padded with random columns) is masked, deflated and SVQB-orthonormalized. Columns SVQB drops (a rank-deficient `X0`) are refilled with random vectors, up to 3 times.
  - Rayleigh–Ritz on $X$ completes the setup.
- **SVQB.** Inactive rows and columns of $G = V^\top MV$ are zeroed, so they form the exact $\theta = 0$ eigenspace and kept directions carry nothing from them. The rest is scaled to unit diagonal, and directions with $\theta \le 10^{-14}\theta_\max$ are dropped (exact zero columns).
- **Each iteration:**
  1. Residuals give the active columns $a = r > t$. This is soft locking: converged columns get no new direction but stay in $X$.
  2. $W = T(R_X) \odot a$ is deflated against the locked basis (when count > 0, one extra M apply) and $R$, then $KW$ and $MW$ are applied. Without locked modes these are the only two width-$m$ applies per iteration.
  3. $W$ is M-orthogonalized against $[X, P]$, then SVQB, then orthogonalized again, then SVQB.
  4. Rayleigh–Ritz on $[X, W, P]$, with the Grams built pairwise from $m\times m$ products.
     - Masked rows and columns are zeroed off the diagonal, with $G_M = 1$ and $G_K = \beta = 10\max_i|\lambda_i| + \|K\|/\|M\|$ on it.
     - $X$ is fully active and M-orthonormal, so by Courant–Fischer the lowest $m$ active Ritz values are $\le \max_i\lambda_i < \beta$. Masked slots never enter, and no $\hat\lambda_\max$ estimate is needed.
     - **Pivot test:** if the Cholesky of $G_M$ is not finite or $\min\mathrm{diag}(L)^2 \le 10^{-10}$, the small problem is redone with the P block masked: a `jnp.where` between two small solutions. This bounds $\lambda_\min(G_M)$ only from above, not the condition number; in practice the pivots are about 1.
  5. The new P coefficients are built as in Duersch:
     - start from the Ritz coefficients with the $X$ rows zeroed;
     - $G_M$-orthogonalize them against $C_X$ twice, in the small space;
     - mask columns whose $G_M$-norm fell to $\le 10^{-8}$ of their norm before (the **noise rule**; without it a noise column would carry about $10^{-4}$ of non-orthogonality into $P$);
     - SVQB in coefficient space.

     `p_drops` counts the iterations in which the pivot fallback, the noise rule or this SVQB lost a P column. It can also fire under heavy soft locking, when the active W and P columns span fewer than $m$ directions; it is a diagnostic, not a failure.
     
     $[X', P']$ is then M-orthonormal without any n-sized orthogonalization of $P$ (measured: $\max|G_M - I| \le 7\cdot10^{-8}$).
  6. **Update.** $X' = XC_x + WC_w + PC_p$, and $P'$ likewise, with the same GEMMs for the K and M products. It is sequenced with `optimization_barrier`: XLA does not fuse a sum of GEMMs, and this saves 2 blocks.
  7. Every `RECOMPUTE` = 20 iterations, a traced `lax.cond` replaces the implicitly updated products by explicit applies.
- **Exit check.** When the wanted pairs meet their thresholds, $KX$ and $MX$ are recomputed explicitly and written back into the state. The solve exits only if the check still passes.

### Chunks and locking

A solve of many modes runs in chunks of `chunk` wanted modes with one block shape; converged chunks are locked into a basis $V$ that later chunks are deflated against.

- **Driver.** For each chunk, `want` = min(chunk, modes still missing). The start block is the previous chunk's unlocked Ritz vectors, padded with random columns, deflated against $V$ and $R$ and SVQB-orthonormalized (the setup's refill rule). The iteration loop is the one above, on the first $j$ columns.
- **Cluster rule.** A lock must end in a gap. Once the wanted residuals are $\le \sqrt{t}$ (Ritz values are then accurate to about $r^2 \le t$, well inside the window), $j$ extends from `want` while
  $$\frac{\lambda_{j+1} - \lambda_j}{|\lambda_j|} < 100\,t_j,$$
  with $t_j \ge$ tol the acceptance threshold (so floor-accepted pairs and the fp32 term count). The rule runs every iteration; $j$ only grows, and the progress test restarts when it does. The extended columns must converge too.
- **Guard error.** If $j$ reaches $m$, the cluster's end is not visible inside the block: `ValueError` "a cluster at the chunk boundary (from mode i) fills the block: it needs more than g guard vectors; pass guards >= g + 1 or a chunk size that ends in a gap". A cluster split by too few guards converges very slowly (one guard on a triple: the guard's Ritz value is still $10^{-7}$ off after 500 iterations), so at a failed exit with the wanted residuals $\le \sqrt t$ the boundary is re-checked with the guards' Ritz-value uncertainty, gap $< \max(100\,t_j, \min(10\,r_{j+1}^2, 10^{-3}))$, and the guard error is raised if the cluster then fills the block. Only chunks that will be locked raise it: the final chunk (and so every single-chunk solve) keeps `want` when the cluster fills the block, since its tail past `n_modes` is cut anyway, as in 9a.
- **Locking.** A converged chunk (including floor acceptance) is locked: columns $1..j$ go into $V$, and $X_{:, j+1..m}$ seeds the next chunk. Before insertion they are M-orthogonalized twice against the basis and M-orthonormalized by Cholesky, and their residuals recomputed (4 M applies and one K apply on $m$ columns per lock), which keeps $V$ M-orthonormal to rounding: the in-loop deflation is a single pass, and when the preconditioned residual is dominated by the locked directions (Jacobi on a pencil whose low modes have small diagonals, up to $\|K\|/\lambda$ times) a basis off by $10^{-9}$ leaves enough of them in $W$ to pull the block into span($V$) after 9–22 chunks in the measured cases. A column whose projected M-norm² falls below 0.5 lies in span($V$) and raises `RuntimeError`. A cluster that straddles `n_modes` is computed whole and the return is cut at `n_modes`.
- **Failed chunk.** A no-progress or `maxit` exit ends the solve: the locked modes plus that chunk's `want` Ritz vectors (upper bounds, M-orthonormal; residuals from the explicit confirmation) are returned with `converged` False. Locking unconverged vectors would corrupt the deflation of every later chunk.
- **Basis.** `B` is a zero-initialized device array $(n, \text{capacity})$ in `basis_dtype` (column layout; the transposed layout was 2–3× slower), with capacity $\lceil (n_\mathrm{modes} + m)/c\rceil\,c$, no basis for a single chunk. A lock writes $m$ columns (X with the unlocked columns zeroed) with a jitted `dynamic_update_slice`, `B` donated (in place). The host checks count + m ≤ capacity before writing (`dynamic_update_slice` would clamp the start onto earlier columns) and raises `RuntimeError`; the capacity rule never triggers it.
- **Projector.** $Y \leftarrow Y - V_c(V_c^\top MY)$ in a `fori_loop` over $\lceil \text{count}/c \rceil$ column chunks of width $c$, each `dynamic_slice`d and upcast to fp64, with $MY$ of the undeflated block passed in. Columns past the count are zero, so the last chunk reads past it exactly. The temp is exactly $(c + m)\,n \cdot 8$ B, independent of the capacity. $c = \max(m, \min(4m, 64))$ for the 9-block step (at small $m$ the projector re-reads Y and MY per chunk: 41 / 33 / 26 ms for $c = m, 2m, 4m$ at $m = 15$, 200 locked, 182k DOF; 221 / 167 / 105 ms with 1000 locked; the step stays at 16 blocks for $c \le 4m$) and $c = m$ for the low-memory step ($4m$ would cost 2.3 blocks). One classical pass suffices for an fp64 basis: the coefficients come from the original $MY$, and $X$ does not drift toward $V$ during a chunk (Rayleigh–Ritz gains locked content only to second order; $\max|V^\top MX|$ stays at $10^{-15}$ over 800 iterations).
- **fp32 basis.** The rounded locked modes are M-orthonormal only to about $u_{32} = 2^{-24}$, and one pass leaves about $u_{32}$ of locked content in $W$: chunks after the first floor near $10^{-8}$ (measured $6\cdot10^{-9}$–$10^{-8}$). Their threshold is $t_i = \max(\mathrm{tol}, \text{floor}_i, 2u_{32})$, about $1.2\cdot10^{-7}$, so they floor-accept (`converged` and `stagnated`) instead of stopping the solve. This assumes $\|Mv_i\|/\|Mx\| = O(1)$; with strongly heterogeneous density (e.g. SIMP voids) the floor can be higher, and the chunk then stops loudly. Use tol $\ge 10^{-6}$ with fp32.
- **Returned modes.** `V` is a preallocated fp64 host array $(n, n_\mathrm{modes})$: each lock copies its columns into it before inserting them into the basis, and the final chunk (never inserted) adds its columns at the end. No device copy of the basis is made.

| case (hex20 $6^3$, 30 modes, chunks of 10, $m = 15$) | basis | chunk iterations | max rel. λ error | $\max\lvert V^\top MV - I\rvert$ |
|---|---|---|---|---|
| clamped, tol $10^{-10}$ | fp64 | 313 / 286 / 268 | $7\cdot10^{-13}$ | $6\cdot10^{-15}$ |
| free, tol $10^{-10}$ | fp64 | 349 / 363 / 284 | $3\cdot10^{-14}$ | $4\cdot10^{-15}$ |
| clamped, tol $10^{-6}$ | fp32 | 197 / 175 / 163 | $1\cdot10^{-12}$ | $9\cdot10^{-9}$ |
| clamped, tol $10^{-10}$ | fp32 | 313 / 191 / 181, chunks 2–3 floor-accepted | $7\cdot10^{-13}$ | $1\cdot10^{-8}$ |

A single chunk of 30 modes ($m = 36$) needs 266 iterations, 2.4 s against 4.9 s for the 3 chunks: chunking trades time for memory.

### Low-memory mode

`lobpcg_step_low` keeps only $(X, P, \mathrm{act}_P, \lambda)$ and recomputes every product when it is needed, in phases separated by `optimization_barrier`s: $KX$, $MX$ (residuals of the input $X$, $W$, Gram blocks; KX dies); deflation; $MP$ and two classical M-orthogonalizations of $W$ against $X$ and $P$ (MX, MP die); $MW$ and SVQB; $KW$; $KP$; Rayleigh–Ritz and the sequenced update of $X$ and $P$. Every Gram block is formed from explicit products, so there is no recompute schedule. It needs 6 applies per iteration instead of 2 (plus the deflation's), and it reports the residuals of the input $X$; the exit check confirms the updated $X$ explicitly before accepting.

- **Memory:** 7 blocks with a diagonal operator; 11.0–11.7 with the hex20 operator at 182k–1.08M DOF ($m$ = 15, 48); 9.0–9.9 with the voxel stencil ($m = 8$, 1.6M–5.4M DOF, CPU). Not 6: XLA copies the donated $X$ and $P$ early so their buffers can take the new ones, and an apply inside the step costs about 4 blocks (input copy, accumulator, output, chunk workspace). Variants without donation, with fewer barriers or fused masks all measured 11–12 blocks.
- **Accuracy:** same iterations ±2 as the 9-block step, λ matching it to $\le 3\cdot10^{-13}$ (12 modes, tol $10^{-10}$, the four accuracy cases).
- **Cost:** see the table below.

| hex20 $24^3$ (182k DOF), warm | 9-block | low-memory |
|---|---|---|
| $m = 15$, no locked modes | 75 ms | 140 ms |
| $m = 15$, 200 locked | 123 ms | 201 ms |
| $m = 48$, no locked modes | 223 ms | 349 ms |
| $m = 48$, 200 locked | 312 ms | 446 ms |

### Memory model

`jaxmodal.memory.lobpcg_memory(op_memory, n_dof, n_modes, chunk=None, guards=None, low_memory=False, basis_dtype="float64", n_rigid=0, precond_memory=None)` takes `op_memory`, a callable $m \mapsto$ the operator estimate (e.g. `lambda m: unstructured_memory(mesh, m)`), and `precond_memory`, a callable $m \mapsto$ the preconditioner estimate (e.g. `lambda m: gmg_memory(mesh, m)`; `None` is the default Jacobi, already inside $A$). It returns, in bytes, with $b = 8\,n\,m$ one block:

- **`step`** $= A\,b + f\,w + A_P\,b$, with $w$ the operator's `workspace` at width $m$: $(A, f) = (16, 2.5)$ for the 9-block step and $(11, 1.75)$ for the low-memory step. $A$ is the large-n step (it includes the operator's own vectors); $f$ is the upper envelope of `memory_analysis` over hex8/hex20/hex27 at 6.6k–1.08M DOF and $m$ = 7–48 (two applies' workspace can be live at once in the 9-block step). The margin is about 3% at 1M DOF, where $w$ is a fraction of a block, and up to about 2× at 10k–100k DOF, where $w$ is several blocks. With a diagonal operator the steps hold 16 and 7 blocks (pinned by tests). $A_P\,b$ is the preconditioner's `step` (9-block) or `step_low` entry: 0 for Jacobi; for AMG $8n(a\,m + c) + 8mK$ (`AMG_STEP`, below); for GMG the extra blocks over Jacobi, `GMG_STEP` = 4.3 / 0 (9-block, fp64 / fp32 V-cycle) and 9.0 / 4.5 (low-memory; re-fitted in PR 13c after the Chebyshev loop lowered the fp64 steps by about 1 block), the upper envelope of `memory_analysis` of the steps at $16^3$–$87^3$, $m = 15$, plus `GMG_LOOP` blocks for a fine-level `degree` > 2 (at $32^3$ and below the stencil's slab workspace dominates and the 9-block step adds nothing; the fp32 cycle fits under the 9-block step's existing peak at every size). **Intermittent extra peak with GMG:** the measured process peak with GMG is bimodal: in some runs, from about iteration 5 on, it is about one V-cycle temporary above `memory_analysis` (+12 blocks at $48^3$, $m = 15$; at $64^3$ GMG − Jacobi is 0.68 GB after 1 iteration, matching `memory_analysis`, and 1.6 GB after 20), in other runs of the same configuration not; synchronous CPU dispatch makes it rarer but does not remove it. Adding it everywhere would put $87^3$ at +36%, so `GMG_STEP` stays the `memory_analysis` envelope: `solve_modes`-sized totals (PR 11b figures, before the 13c re-fit, which lowers estimate and measurement alike) are 11% low at $64^3$ with the fp64 cycle (3.37 against 3.78 GB; fp32 +4%), 4% high at $48^3$ (worst of three runs) and 10–12% high at $87^3$, inside the ±20% band (`docs/open_items.md`).
- **`basis`** $= n \cdot \text{capacity} \cdot (8$ or $4)$; **`rigid`** $= 2 \cdot 8\,n\,n_\mathrm{rigid}$ ($R$ and $MR$); **`host`** $= b + 8\,n\,n_\mathrm{modes}$ (the asynchronously freed step output and the fp64 modes on the host), plus 4 blocks for a chunked solve (transients between chunks: the start block, the seed, allocator slack; up to about 6.5 measured, varying between runs).
- **AMG step extra** (`AMG_STEP`, PR 13b): per family (all-hex8; the rest) and step kind, $8n(a\,m + c) + 8mK$ with $(a, c, K)$ = (4.80, 42.0, 163 889) / (9.57, 28.5, 227 703) for hex20/hex27 (9-block / low-memory) and (4.94, 115.4, 71 400) / (9.17, 107.2, 384 682) for hex8. The last term is n-independent (about 1–3 MB per column: the block-ELL scan-chunk gathers of the AMG levels and transfers), which is why small meshes carry 10–40 extra blocks. The constants are the minimal linear-programming envelope of `memory_analysis` at hex20 $8^3$–$44^3$, hex27 $6^3$–$24^3$, hex8 $12^3$–$68^3$, $m$ = 15 and 25 (plus 1.5 in $c$ against rounding); at 1M the gap is 0.6–1 block. At $m$ = 36 (30 modes) the model extrapolates. Unlike GMG, the low-memory step gains nothing with AMG (about 6 extra blocks at $m = 25$ with the 9-block step, about 11 with the low-memory one at 1M hex20).
- **`precond`**: the preconditioner's `storage` (0 for Jacobi).
- **`total`** = overhead + operator storage + precond + basis + rigid + host + max(operator build, precond build, step).

`jaxmodal.memory.solve_modes_memory(mesh, bcs, n_modes, *, mass="consistent", chunk=None, guards=None, basis_dtype="float64", low_memory=False, precond="auto", precond_dtype="float64", n_rigid=None, operator_options=None, precond_options=None)` is the one-call estimate of a `solve_modes` run with the same options (keys checked as in `solve_modes`; `operator_options` go to the operator estimator, `max_coarse` and `degree` to `amg_memory`, `degree` to `gmg_memory`; `coarse_degree` and `ratio` change no memory): `stencil_memory` or `unstructured_memory`, `gmg_memory` for GMG (no islands: `solve_modes` builds uniform `s_K`), `amg_memory` for AMG ([`preconditioners.md`](preconditioners.md#memory-amg_memory)) and `lobpcg_memory`, composed; it returns `lobpcg_memory`'s dict. `n_rigid=None` counts the rigid modes with `rigid_modes(mesh, dof_mask(mesh, bcs))` on the host; pass it to skip that at large sizes. The pre-estimate's hierarchy is freed before the final build, and its 6-column LOBPCG is smaller than the solve's, so it adds nothing to the peak.

Measured process peak at hex20 $44^3$ (1.08M DOF), 40 modes in one chunk with 8 guards ($m = 48$), 3 iterations: 8.74 GB against the estimate 8.87 GB (9-block), 6.52 against 6.75 GB (low-memory). At the PR 13 acceptance width ($m = 25$, 10 iterations, 9-block) the estimate is 4–7% low: 5.50 against 5.28 GB clamped, 5.63–5.79 against 5.39 GB free; at the PR 11 voxel size ($87^3$, $m = 15$, free, 9-block) 5.26 against 5.47 GB. Chunked, with a diagonal pencil (300k DOF, 100 modes in chunks of 10), the peak varies between runs: 1.24–1.39 GB against 1.30 GB (fp64 basis), 1.10–1.20 GB against 1.15 GB (fp32); the test allows 0.8–1.1× the estimate ([`open_items.md`](open_items.md)). The step term is an envelope fit: up to about 2× high at 10k–100k DOF, and for an operator without workspace (diagonal) a single chunk can come out up to about 14% low (30 modes, $m = 36$, peak up to 1.16× the estimate; varying between runs, [`open_items.md`](open_items.md)), and many chunks up to 7% low (500k DOF, 400 modes in chunks of 10: allocator growth of about $0.15 \cdot 8\,n\,n_\mathrm{modes}$; real operators usually add slack through the workspace term, but at narrow blocks such as hex20 at $m = 25$ the fit is 4–7% low, as measured above).

### Convergence

- **Residual:** the λ-scaled $r_i = \|K_mx_i - \lambda_iM_mx_i\| / (|\lambda_i|\,\|M_mx_i\|)$.
- **Threshold:** $t_i = \max(\mathrm{tol},\ c\,\varepsilon\,(\|K\| + |\lambda_i|\,\|M\|)\,\|x_i\| / (|\lambda_i|\,\|M_mx_i\|))$ with $c = 10$. It stays bounded as M becomes singular.
- **Progress** is a new minimum of the max wanted residual by a factor 0.9, or a decrease of the sum of the wanted Ritz values by more than a relative $10^{-10}$. Both clauses are needed: while λ̃ falls, the λ-scaled residual rises. On a 1-mode hex20 $6^3$ run, the residual alone had no new minimum for 85 iterations.

| exit | `converged` | `stagnated` |
|---|---|---|
| all wanted $r_i \le$ tol (explicitly confirmed) | True | False |
| all wanted $r_i \le t_i$, some above tol: accepted at the floor | True | True |
| no progress for 50 iterations | False | True |
| `maxit` reached | False | False |

**Theory.** For M-normalized $x$ and the M⁻¹-norm residual $r_M = \|Kx - \lambda Mx\|_{M^{-1}}/|\lambda|$, with the absolute gap to the rest of the spectrum:

$$\sin\theta \le \frac{r_M\,\lambda}{\mathrm{gap}}, \qquad \frac{|\delta\lambda|}{\lambda} \le \frac{r_M^2\,\lambda}{\mathrm{gap}}$$

These are Davis–Kahan and Kato–Temple in the M-inner product. The 2-norm $r$ is the cheap proxy; it equals $r_M$ up to the conditioning of M.

### σ pre-estimate

`sigma_estimate(op, free, R=None, maxit=300, seed=0)` returns `(sigma, iterations)`.
- It runs the same loop with 1 mode ($m = 6$) and Jacobi at σ = 0, and returns the Rayleigh quotient λ̃₁ ≥ λ₁.
- It stops when λ̃₁ has decreased by less than 5% over the last 10 iterations.
- The λ-scaled residual is no stop criterion here: it is small while λ̃ is huge (a stop at 0.5 gave 432× λ₁). A fixed count is no good either: 20 iterations give 4–98× λ₁.
- A free body needs `R`; without it σ is a rigid mode.

| mesh | DOF | iterations | λ̃₁/λ₁ |
|---|---|---|---|
| hex20 $8^3$ clamped | 8019 | 89 | 1.29 |
| hex20 $8^3$ free | 8019 | 102 | 1.04 |
| hex8 $16^3$ clamped | 14739 | 59 | 1.27 |
| hex27 $6^3$ clamped | 6591 | 76 | 1.01 |
| hex20 $12^3$ clamped | 24843 | 106 | 1.32 |
| hex8 $24^3$ free | 46875 | 82 | 1.02 |

The ratio grows slowly with size. That costs 60–110 iterations at $m = 6$, about 10–15% of a solve.

**The default Jacobi uses σ = 0.** For Jacobi, σ·diag M/diag K ≤ 4·10⁻⁴, and the iterations to $10^{-8}$ barely change: 330 / 338 / 334 for σ = 0 / λ₁ / λ₁₀ on hex20 $8^3$, and 231 / 231 / 229 on hex8 $16^3$. `lobpcg` therefore never runs a pre-estimate, and `solve_modes` runs one only for GMG and AMG (below).

**Multigrid-refined pre-estimate** (PR 11b, generalized in 13b: `_precond_sigma(op, free, R, build, *, seed=0, tol=0.1, maxit=20, rounds=4, drop=3.0, estimate_maxit=10)` in `eigen/api.py`, tuned through `sigma_options`, run by `solve_modes` for `precond="gmg"` and `"amg"` without a hint; `build(σ)` is `gmg(op, free, σ)` or `amg(op, free, σ, mesh.points, nonsingular=(k == 0))`, and raises `SingularShiftError` where σ = 0 is refused). `sigma_estimate` is up to 3800× above λ₁ on thin plates, slender beams, anisotropic grids and SIMP fields, takes 73 s (252 iterations) at 2M DOF, and capped at 3–10 iterations is $10^2$–$10^7$× above. GMG at any σ in $[0, 3\lambda_1]$ preconditions as well as at λ₁ (σ ≥ 10λ₁ costs iterations on bulky and free bodies), so the estimate refines itself with GMG:

1. **Start:** `build(0.0)` when it accepts σ = 0 (GMG: the singular guard proves $A_P$ nonsingular; AMG: no rigid modes). On `SingularShiftError` (free or sliding bodies, zero-stiffness cells), start from the upper bound σ₀ = `sigma_estimate(op, free, R, maxit=10)` and `build(σ₀)`.
2. **Round:** 1-mode `lobpcg(op, free, 1, 0.1, R, P, maxit=20)` gives the Rayleigh quotient $\tilde\lambda \ge \lambda_1$; σ becomes $\tilde\lambda$ (from 0) or $\min(\sigma, \tilde\lambda)$.
3. **Stop:** from σ = 0 after one round (GMG at σ = 0 approximates $K^{-1}$ on the free DOFs, the best preconditioner for λ₁); from σ₀ when $\tilde\lambda$ > previous/3, or after 4 rounds. Otherwise rebuild at σ (the old hierarchy freed first) and repeat.

The tests accept σ in $[\lambda_1(1 - 10^{-6}), 2\lambda_1]$: the Rayleigh quotient is $\ge \lambda_1$ in exact arithmetic, but on the SIMP field it lands at $(1 - 4\cdot10^{-8})\lambda_1$ of the SciPy reference, within that reference's accuracy on such an ill-conditioned field.

The residual stop at 0.1 is safe here though not with Jacobi (above): with GMG the residual reaches 0.1 only near an eigenpair, so the probes stop early (4–14 iterations) only where σ is within 1% of λ₁, and otherwise run to 20.

| case | start σ/λ₁ | σ/λ₁ after each round |
|---|---|---|
| boxes, odd grid, plates $32^2$ / $64^2\times1$, beams $64\times2\times2$ / $128\times1\times1$, aniso $h_z = h/10$, SIMP field, clamped beam with a 200× tip mass | 0 | 1.00 |
| free box $8^3$ | 1.82 | 1.00 |
| free plate $16^2\times1$ | 81 | 1.00, 1.00 |
| free beam $128\times1\times1$ | $5.1\cdot10^4$ | 2.7, 1.00 |
| free aniso $h_z = h/10$ | $8.9\cdot10^3$ | 6.83, 1.00, 1.00 |
| free SIMP field | 277 | 1.05, 1.00 |
| unfiltered binary SIMP field $32^3$ | 0 | 390 |

With AMG (PR 13b; λ₁ from SciPy; `amg` at σ = 0 needs no change beyond lifting the refusal):

| case | DOF | start σ/λ₁ | σ/λ₁ after each round | LOBPCG iterations |
|---|---|---|---|---|
| hex20 $8^3$ / hex8 $12^3$ / hex27 $6^3$ | 8k / 7k / 7k | 0 | 1.000 | 4 / 5 / 5 |
| beam $40\times2\times2$, plate $16^2\times1$ | 4k / 6k | 0 | 1.000 | 7 / 7 |
| plate $16^2\times1$, $a = 4$ | 6k | 0 | 1.001 | 20 |
| checkerboard `s_K` $10^{-3}$, nodal masses ($6^3$) | 4k | 0 | 1.000 | 7 / 5 |
| free hex20 $8^3$ | 8k | 5.5 | 1.000, 1.000 | 8 + 5 |
| free beam $40\times2\times2$ | 4k | 1578 | 1.000, 1.000 | 20 + 6 |
| free plate $16^2\times1$ | 6k | 712 | 1.001, 1.000 | 20 + 7 |
| free hex8 $12^3$ | 7k | 2.8 | 1.000 | 7 |
| hex20 $44^3$ (acceptance) | 1.08M | 0 | 1.000 | 6 |

- **Cost** at 2M DOF (clamped, one round): the σ = 0 build 5–6 s plus 5 LOBPCG iterations 7–9 s, against 73 s for Jacobi `sigma_estimate`. Free bodies add `sigma_estimate(maxit=10)` and up to 3 more rounds (worst case about 2 min at 2M).
- **Known miss:** on an unfiltered binary SIMP field λ₁ is a void-region mode that 20 iterations from a random start miss (100 find it), so σ lands at 390 λ₁ (`docs/open_items.md`); 10-mode LOBPCG does not converge on that field at any σ either.

### Measured behaviour

All runs: isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$ on the box $(1, 0.83, 0.67)$, `xmin` clamped unless free, 12 modes, Jacobi, M3 Max CPU, warm.

| mesh | DOF | tol $10^{-6}$ | tol $10^{-10}$ | max rel. λ error |
|---|---|---|---|---|
| hex20 $6^3$ clamped | 3675 | 197 its | 320 its (1.5 s) | $1\cdot10^{-13}$ |
| hex20 $6^3$ free (rigid deflation) | 3675 | 209 | 346 (1.5 s) | $4\cdot10^{-14}$ |
| hex27 $4^3$ clamped | 2187 | 109 | 169 (0.5 s) | $2\cdot10^{-13}$ |
| hex8 $10^3$ clamped | 3993 | 103 | 159 (0.7 s) | $1\cdot10^{-12}$ |
| hex20 $8^3$ clamped | 8019 | 281 | 439 (2.1 s) | $1\cdot10^{-12}$ |

- **Iterations:** these grow roughly like $h^{-1}$, as expected for Jacobi ($\sqrt\kappa$). No P drops occurred in these runs.
- **Floor:** at tol $10^{-16}$ on hex20 $4^3$ the solve accepts at the floor threshold (`converged` and `stagnated`). Over seeds 0–3, the lowest max wanted residual is $1.2$–$3.4\cdot10^{-12}$ clamped and $3.4$–$4.4\cdot10^{-13}$ free, that is $0.5$–$1.9\times\varepsilon\lambda_\max/\lambda_1$.
- **Near-singular M:** hex20 $6^3$ with 30% void elements ($E\cdot10^{-9}$, $\rho\cdot10^{-12}$) converges to $10^{-8}$ in 337–374 iterations (3 seeds), with λ errors of $10^{-12}$.
- **Thin plates are out of reach for Jacobi.**
  - A clamped hex20 plate $10\times10\times1$ with $L/t = 100$ has $\lambda_\max/\lambda_1 = 7\cdot10^{13}$. After 6000 iterations λ̃₁ is still $8\cdot10^4$ times too high.
  - λ̃₁ decreases monotonically. The λ-scaled residual grows (3 → 70), because λ̃ shrinks faster than the absolute residual.
  
  Plates need the preconditioners of PRs 10–13.
- **Cost per iteration** (hex20 $24^3$, 182k DOF):
  - 72 ms at $m = 15$ (K and M applies about 40 ms) and 216 ms at $m = 48$ (about 90 ms). The rest is GEMMs at about 450 GFLOP/s.
  - Recompute iterations cost 150 / 390 ms.
  - The setup (power iterations, start block, R check) takes about 0.5 s warm (≈ 7 iterations) and 1.1 s with the first-call compiles.
- **Step memory:**
  - With a diagonal operator, so that no operator workspace counts, the compiled step holds temp + donated state = 16.0 blocks of $(n, m)$ fp64. This is pinned by a test; without the barriers it is 18.
  - The roadmap's original 9 + 2 blocks were not reached in any measured variant. A donated buffer takes its output only after every read of it, so the new $X$ and $P$ coexist with the old ones and $W$.
  - With a real operator the operator workspace adds to it: 55.6 blocks at hex20 $8^3$ (8k DOF), 16.05 at 182k and 16.00 at 1.08M, so the step reaches its 16 blocks by about 180k DOF. See [Memory model](#memory-model).

## Completeness count

`jaxmodal.eigen.count` (module `eigen/counting.py`, PR 14a) estimates how many eigenvalues below a cut $\lambda_c$ are missing from a set of computed modes $W$. It is the building block of PR 14b's `verify` option of `solve_modes(..., band=(0, f_max))`; on its own it takes the modes and their eigenvalues:

```python
from jaxmodal.eigen import count

est, info = count(op, free, W, lam, lam_last, lam_next, R=R, precond=P)
info["passed"]  # verifiable and est < 0.5: no mode below lam_c is missing
```

### Estimator

- **The map.** With $\sigma = \lambda_c = (\lambda_\text{last} + \lambda_\text{next})/2$ (the last mode below the cut and the first above it), $T = (K + \sigma M)^{-1}M$ on the free DOFs and $X = 2\sigma T - I$, a mode $\lambda \ge 0$ becomes an eigenvalue $x = (\sigma - \lambda)/(\sigma + \lambda) \in (-1, 1]$ of $X$. No spectral bound is needed ($\lambda = 0 \mapsto 1$, $\lambda \to \infty \mapsto -1$). The cut sits at $x = 0$, where $|dx/d\ln\lambda| = 1/2$ is largest, so the relative gap $g = (\lambda_\text{next} - \lambda_\text{last})/\lambda_c$ becomes $[-g/4, g/4]$.
- **The estimate.** With $\Pi = I - WW^\top M - RR^\top M$ (computed modes and rigid modes, M-orthonormal) and probes $y_j = M_P^{-1}g_j$, $g_j \sim N(0, M)$ masked (so $\mathrm{Cov}(y) = M_{ff}^{-1}$),
  $$s_j = \|\Pi\,q(T)\,\Pi y_j\|_M^2,\qquad E\,s_j = \sum_{i \notin W} q(\lambda_i)^2 .$$
  The estimate is the mean of $s_j$ over the probes (default 50). One missing in-band mode ($q^2 \ge 0.95$) gives $s \gtrsim 0.95\,\chi^2_{50}/50$, and $P(\chi^2_{50}/50 < 0.5/0.95) = 0.23\%$: the count passes when the estimate is $< 0.5$.
- **Probes** come from `mass_sample_keys` ([`operators.md`](operators.md#probe-sampling)); probe $j$ uses `fold_in(key(seed), j)`, so a probe does not depend on the block it runs in. $y = M_P^{-1}g$ is one Jacobi PCG on $M_P = mMm + (1 - m)$ to $\tau_c$ (exact for lumped mass). The constrained diagonal is 1 rather than the roadmap's scaled diagonal: $g$ vanishes there, so $y$ is the same.

### Step polynomial and degree

$q(x) = \sum_{k=0}^d \gamma_k T_k(x)$ is the Jackson-damped Chebyshev series of the indicator of $x \ge 0$ (Weiße et al. 2006): $\gamma_k = c_k g_k$ with $c_0 = 1/2$, $c_k = 2\sin(k\pi/2)/(k\pi)$ and
$$g_k = \frac{(d - k + 1)\cos(k\alpha) + \sin(k\alpha)\cot\alpha}{d + 1},\qquad \alpha = \frac{\pi}{d + 1},$$
so $0 \le q \le 1$. `count_degree` returns the smallest $d$ (doubling, then bisection) with

- $q^2 \ge 0.95$ (`Q_MIN`) at the in-band edge $\lambda_c(1 - g/2)$, by the exact $O(d)$ cosine sum, and on a DCT grid of $8d$ points over the band (a guard against ripples);
- a Jackson-tail bias $\int q^2\,dN \le 0.05$ (`BIAS`) over the modes above $\lambda_\text{top} = \max(\texttt{lam})$, with the Weyl model $N(\lambda) = n_\text{top}(\lambda/\lambda_\text{top})^{3/2}$ (`WEYL_A`), $n_\text{top}$ = the computed and rigid modes, up to $N = n_\text{free}$; evaluated by the trapezoid rule on a finer DCT grid of $64d$ points with both ends exact (second order: within 0.04% of `scipy.integrate.quad` where $dN/d\theta$ grows fastest; $8d$ points were 2.5% off).

The exponent $3/2$ is fixed: it is the 3D value and an upper bound on fits to computed modes (bulky 1.1, plate 0.4–0.5, beam 0.3–1.3), stricter than a fitted exponent. The in-band condition sets the degree, $d \approx 23/g$:

| g | 0.5 | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|
| d | 39 | 108 | 222 | 451 | 1136 | 2279 |

The tail binds only at large gaps with thousands of modes below the cut ($g = 0.5$, $n_\text{top} = 5000$). Above `MAX_DEGREE` = $10^5$ ($23/g$ larger) the count is not run: `estimate` NaN, `verifiable` False, `degree` $= \lceil 23/g\rceil$.

### Recurrence and solves

- **Recurrence.** $w_0 = \Pi y$, $w_1 = Xw_0$, $w_k = 2Xw_{k-1} - w_{k-2}$, $u = \sum_k \gamma_k w_k$, one jitted `lax.while_loop` (`_recurrence`). $Xw = 2\sigma\,Tw - w$ with $Tw$ = PCG on `ap_operator(op, free, σ)` with the caller's SPD preconditioner (default `jacobi(op, free, σ)`; PR 14b passes the fp64 multigrid at $\lambda_c$). $\gamma$ is zero-padded to the next power of two and $d$ is a traced loop bound: one compile per block width and size class.
- **Π streamed.** $W$ stays where the caller keeps it (the host copy `lobpcg` returns, fp64 or fp32): $\Pi$ moves it to the device in column chunks of 64 (upcast to fp64), at the start and the end of each block. No $n \times k$ device copy exists. $W$'s M-orthonormality is checked by a streamed sketch, $e = W^\top M(Wz) - z$ with 3 seeded Gaussian columns, $\|e\| \le 10^{-6}\|z\|$ ($10^{-4}$ for fp32 $W$).
- **Tolerance.** $\tau_c = 0.01/(d\sqrt{n_\text{free}})$. Errors propagate linearly: $|\Delta s_j| \lesssim 2\sqrt{s_j}\,\varepsilon + \varepsilon^2$, with $\varepsilon$ the accumulated M-norm error ($\lesssim \tau d\sqrt n$ in the worst case). PCG stops on the 2-norm residual, so the worst case carries $\sqrt{\kappa(M)}$, $\kappa(D^{-1/2}MD^{-1/2}) \approx 145$ on hex20: $|\Delta s| \lesssim 0.02\sqrt{\kappa(M)\,s} \approx 0.24\sqrt s$. Measured errors are far smaller ([measured](#measured-cost-and-errors)).
- **The flag.** A probe-y or recurrence PCG column that stagnates or reaches `maxit` unconverged makes the count unverifiable: the recurrence stops at once, the remaining blocks are skipped, `estimate` is NaN and `probes` is NaN for the stopped and skipped blocks. $\tau_c$ below the attainable PCG accuracy appears as the stagnation exit.

`info` holds `probes` ($s_j$), `std`, `degree`, `gap`, `lam_c`, `tau`, `bias`, `block`, `steps` (recurrence steps run), `pcg_iterations` (mean per solve, max), `verifiable`, `passed` and `time`: `sampling` (the y solves), `recurrence` (the recurrences alone), `projection` (projections and the reduction), `step` (recurrence per step), `block` ((sampling + projection) per block run) and `total`; only `total` when the degree cap skips the count, with `tau` then None unless passed. Before PR 14b `step` included the projections; the [measured table](#measured-cost-and-errors) stays valid (they are spread over $d \ge 16$ steps, a share below $1/d$). The internal `_steps` keyword (PR 14b's calibration) caps the steps per block without changing the compiled polynomial size.

| keyword | default | why |
|---|---|---|
| `n_probes` | 50 | $P(\text{miss}) = 0.23\%$ for one missing mode with $q^2 \ge 0.95$ |
| `seed` | 0 | a fixed probe list: deterministic counts |
| `tau` | $0.01/(d\sqrt{n_\text{free}})$ | the error model above; the measured errors stay about five orders of magnitude below the 0.5 threshold (roadmap bound, kept; see below) |
| `degree` | the degree rule | the smallest $d$ meeting `Q_MIN` and `BIAS` |
| `block` | `n_probes` | one block; PR 14b sets it from `probe_block` and the solve's memory estimate |
| `maxit` | 1000 | per PCG solve; a solve needing more is flagged |
| `precond` | `jacobi(op, free, λ_c)` | building-block default; the multigrid at $\lambda_c$ is PR 14b's |

Kept internal on purpose: `Q_MIN` and `BIAS` (acceptance criteria), `WEYL_A` (the 3D bound) and `MAX_DEGREE` (the cost cap; an explicit `degree` bypasses it).

### Memory model

`memory.count_memory(op_memory, n_dof, b, *, voxel, precond_memory, n_rigid, chunk, w_dtype)` returns the storage (operator, preconditioner, $R$ and $MR$) and three phases, total = overhead + storage + max(phases):

- **step:** the recurrence's `memory_analysis` temporaries, operator workspace $w$ + $8n(10.2\,b + c)$ with $c = 3$ (unstructured) or 1 (stencil); on the stencil also at least $2w + 8n(2.2\,b + 1)$ (the slab workspace appears twice while a slab covers most of the grid); plus the preconditioner's `count` term (`gmg_memory`: $8n(7.8\,b + 10)$; `amg_memory`: $8n(11.5\,b + 45)$) and $w_0$, $u$ (2 blocks). Upper envelopes of hex20 $12^3$–$28^3$ and voxel $16^3$–$64^3$, $b$ = 4–32; below about 20k hex20 DOF one chunk holds the whole mesh and the fixed buffers are larger.
- **sampling:** the y PCG, $w + 8n \cdot 8b$; the element inputs are drawn inside the scatter loop, so sampling stays below the step.
- **projector:** two chunks of $W$ in flight (asynchronous transfers; $n \cdot 64$ entries each, plus the fp64 upcast for fp32 $W$), three blocks (Y is donated) and the M apply.

`memory.probe_block(budget, n_probes, op_memory, n_dof, **kw)` is the largest block with total ≤ budget (raises below one probe). `solve_modes(band=..., verify=...)` sizes it from the solve's own estimate ([Frequency band](#frequency-band-0-f_max)), so the count fits by construction.

`solve_modes_memory(mesh, bcs, n_modes=None, ..., band=None)`: with `band`, `n_modes` is the expected count and `n_modes` and `chunk` default as in `solve_modes(band=...)`; the estimate is `lobpcg_memory(..., band=True)` at that capacity, one host block above the plain estimate (`V` keeps room for the final chunk). Past the expected count the basis grows ([capacity](#frequency-band-0-f_max)), which the estimate does not cover.

### Measured cost and errors

`benchmarks/counting_study.py` (results in `benchmarks/results/counting_study.json`; Apple M-series CPU, fp64, steel cantilever clamped at `xmin`, 200 modes from `solve_modes`):

**Degree and cost against the cut.** Every cut between modes $k$ and $k + 1$ of the two 200-mode solves:

| mesh | DOF | solve | $d\,g$ (median) | cuts not countable | $t_\text{step}$ ($b = 10$) | PCG / solve | predicted count / solve |
|---|---|---|---|---|---|---|---|
| hex20 $16^3$, AMG at $\lambda_c$ | 56k | 36.5 s | 22.8 | 53 of 199 | 0.70–0.82 s | 12–14 | median 250×, 3.6× at $g = 0.6$ |
| voxel $32^3$, GMG at $\lambda_c$ | 108k | 59.8 s | 22.8 | 52 of 199 | 0.80–0.96 s | 12–14 | median 160×, 2.4× at $g = 0.6$ |

Of the cuts not countable, 50 per mesh fall inside degenerate pairs (symmetric cross-section, $g \le 10^{-12}$); the other 3 (hex20) / 2 (voxel) are near-degenerate, $g = 1.3 \cdot 10^{-5}$–$1.7 \cdot 10^{-4}$, below the cap's $23/10^5$. 14b places cuts only between clusters and away from such gaps. The predicted count time is $\lceil 50/10 \rceil\,d\,t_\text{step}$. The degree rule $d \approx 23/g$ holds for $g \le 0.5$ ($d\,g$ = 22.8 median); at large gaps the map's curvature lowers it ($d\,g$ = 19 at $g = 0.6$, 16 at $g = 1.0$). The step time does not depend on the mode index (the multigrid at $\lambda_c$ needs 12–14 iterations from index 10 to 199). The typical relative gap between consecutive modes at index 10–200 is $g \approx 0.002$–$0.04$, i.e. $d \approx 500$–$11\,000$. Counting at an arbitrary cut therefore costs 10²–10³ solves; only cuts with $g \gtrsim 0.6$ cost a few solves (3.6× / 2.4× at $g = 0.6$, 1.8× / 1.2× at $g = 1.0$). So `verify=None` (PR 14b) will enable counting rarely, as the roadmap's cost rule intends.

**Error propagation.** $|s_j(\tau) - s_j(\text{exact})|$ (max over 10 probes, AMG at $\lambda_c$ against exact dense solves) on the reference box and a graded box ($1.3^i$ spacing in $x$, upper half 100× denser; scaled $\kappa(M)$ 144 and 146, 1962 free DOF) for $d$ = 20, 100, 400 and $\tau = 10^{-4}$–$10^{-10}$:

- with one mode missing ($s \approx 2$) the error is $0.4$–$1.0\,\tau s$, **independent of $d$**: no $O(d)$ or $O(d^2)$ growth (the Chebyshev recurrence is stable, and the solve errors do not accumulate in the tail);
- with $W$ complete ($s \le 1.3 \cdot 10^{-4}$) it is below $10^{-8}$ at every $\tau$ ($\tau = 10^{-4}$ included);
- AMG needs 7–18 iterations on the reference box and 19–43 on the graded one; every count is verifiable.

At the default $\tau_c = 0.01/(d\sqrt{n_\text{free}})$ ($2.3 \times 10^{-6}$ at $d = 100$ here) the error is about $10^{-6}$, five orders of magnitude below the 0.5 threshold; the rule is far stricter than needed on these meshes, but it is the roadmap's bound ($\tau_c \lesssim 0.01/(d\sqrt{n})$) and is kept (open item: 14b may loosen it to save PCG iterations, about 1.8 (reference) and 3.9 (graded) iterations per decade, a third of the PCG iterations from $\tau_c$ to $10^{-4}$ on these meshes).

**Floor.** The reference box's Jacobi count (degree 1) is verifiable down to $\tau = 10^{-14}$ and stagnates at $10^{-15}$ (294 iterations; `test_floor` with its own probes reaches $10^{-15}$); the stagnation exit, not `maxit`, flags it.

## Shift-invert slices

`si_slice` (module `eigen/shift_invert.py`, PR 15a) computes every eigenpair in one slice $[\lambda_{lo}, \lambda_{hi}]$ ($\lambda = (2\pi f)^2$, $0 \le \lambda_{lo} < \lambda_{hi}$) by block shift-invert Lanczos. It is internal: PR 15b adds the band-pass count and restarts, and PR 15c the slicing, the strategy choice and `solve_modes(..., band=(f1, f2))`, which exposes its options.

```python
from jaxmodal.eigen.shift_invert import si_slice

P = amg(op, free, sigma, mesh.points)  # SPD multigrid of K + sigma M (gmg / jacobi also)
lam, V, info = si_slice(op, free, lo, hi, P, capacity=120, sigma=sigma, R=R, W=W)
```

### Algorithm

- **Shift and operator.** σ defaults to the midpoint $(\lambda_{lo} + \lambda_{hi})/2$. One step applies $T = (K - \sigma M)^{-1}M$ on the free DOFs to the newest block $Q_j$ ($n \times b$, $b$ = `block` = 6): `minres(ap_operator(op, free, -sigma), m * M(Q_j), precond, tau, maxit)`. The constrained diagonal is `ap_operator`'s default at $-\sigma$, mean diag K + σ mean diag M over the free DOFs (positive). `precond` is SPD, built by the caller at $+\sigma$ (absolute-value preconditioning, Vecharynski–Knyazev 2013); a GMG with fp32 levels raises (its asymmetry floors MINRES near $10^{-6}$). σ is a traced leaf of the operator, so one compiled MINRES serves every σ.
- **Inner tolerance.** MINRES stops on the explicit 2-norm residual $\|r_{in}\|_2 \le \tau\|Mq\|_2$ (τ = `tau` = $10^{-8}$) or stagnates (no new minimum over 5 checks, one every 10 iterations). $\tau_\text{eff}$ is $\tau$ raised to the running maximum of the final residuals of columns that stagnated or hit the cap (`info["tau_raised"]`). Inner cap: `maxit_inner` for the first step, then $\min(\texttt{maxit\_inner},\ 10\,n_1)$ with $n_1$ the first step's iterations (the depth of σ, which sets the iterations, is fixed within a slice); a capped column sets `info["inner_maxit"]`. $\tau_\text{eff} > 10^{-4}$ stops the slice: `converged` False, `reason` "inner solves failed".
- **σ on an eigenvalue.** If the first block stagnates above $10^3\max(\tau, 10^{-10})$ (last-iterate residuals of 0.1 and more were measured with σ on an eigenvalue before the best-iterate fix), σ moves once by 1% of the slice width toward $\lambda_{hi}$ and the start repeats (`info["sigma_moved"]`; `info["sigma"]` is the value used; the inner cap is based on the repeated start). The $10^{-10}$ floor of the trigger sits above the fp64 MINRES floor (measured $2\cdot10^{-14}$–$10^{-12}$ at depths 6–500), so a τ below the floor raises $\tau_\text{eff}$ instead of moving σ. A non-finite MINRES residual also stops the slice as "inner solves failed".
- **Basis.** A preallocated fp64 device array holds $[W \mid Q_1 \mid Q_2 \mid \dots]$, the W part padded to the projector chunk $c = \max(b, \min(4b, 64))$, the Lanczos part `capacity` columns (required, rounded up to a multiple of $c$); the basis also stops within $b$ columns of the free space left by R and W ($n_\text{free} - n_R - n_W$; the Rayleigh–Ritz is then exact when $b$ divides it; below $2b$ it raises). Each new block is projected twice against R and every filled column (`lobpcg._deflate`, chunked fp64), M-orthonormalized by SVQB, and a column lost to rank loss is replaced by a fresh random vector (at most 3 refills). Then $G_{:,new} = Q_L^\top K Q_{new}$ over the Lanczos columns only: W stays out of the projection, so its modes never return.
- **Ritz pairs.** $\mu, Y$ = `eigh` of the symmetric $G$ (host): the Rayleigh–Ritz of $(K, M)$ on the Lanczos basis. The inexact solves only shape the subspace; Ritz values come from exact K products, so they are not limited by τ (Ritz values of the projected inexact $T$ gave residuals above $10\tau\delta$ that grew with the basis).
- **Thresholds.** Pair $i$ is accepted at $r_i \le t_i = \max(tol,\ 10\,\tau_\text{eff}\,\delta_i,\ \text{the round-off floor})$, $\delta_i = |\mu_i - \sigma|/|\mu_i|$, $r_i$ LOBPCG's λ-scaled residual and its floor term ([convergence](#convergence)). $r_{acc} = \max t_i$ over the in-slice pairs and the neighbours, never below the roadmap's $\max(tol, 10\,\tau_\text{eff}\,\delta_\max)$.
- **Sets and stop rule.** `_slice_sets(mu, lo, hi, r)`: `inside` = the Ritz values in $[\lambda_{lo}(1 - 100r), \lambda_{hi}(1 + 100r)]$ extended over chains of relative gaps $< 100r$ (clusters stay whole); `below` / `above` = the nearest Ritz value beyond on each side (gap $\ge 100r$ by construction; no `below` at $\lambda_{lo} = 0$). `_slice_check` forms the sets with the previous step's $r_{acc}$ (tol at first), computes residuals of the candidates (inside plus neighbours) in chunks of $b$ candidates (the basis read in $c$-column chunks), and re-forms the sets with $r = r_{acc}$ while $r_{acc} > r$, computing residuals only for new candidates. `done` = every candidate converged, every needed neighbour present, and $r_{acc} \le r$. Spurious interior Ritz values carry large residuals and stay unconverged until the subspace resolves them.
- **Capacity full.** When the next block does not fit before `done` (in the capacity or the free space), the slice returns its converged inside pairs with `converged` False, `capacity_full` True, `reason` "capacity full" and `edge_unreached` = "lo", "hi", "both" or None: the edges beyond which no converged Ritz value ever appeared ($\lambda_{lo}$ below $\lambda_1$ or $\lambda_{hi}$ above the spectrum, for 15c's [0, f₂] rule and above-spectrum error).
- **Final stage.** V of the accepted pairs and converged neighbours is formed from the last $Y$ in chunks of $c$ columns, each copied to the host (`_form_v`); the basis is released, and the residuals and thresholds are recomputed from explicit K and M products of V (a pair above its $t_i$ there sets `converged` False).

`lam` (ascending) and `V` (host fp64, $n \times k$, M-orthonormal, constrained rows 0) are the converged `inside` pairs: the slice's modes plus those within $100\,r_{acc}$ of an edge and whole clusters straddling one (15c's duplicate removal handles the overlap between slices). `info`: `converged`, `reason` (None, "capacity full", "inner solves failed"), `sigma`, `sigma_moved`, `move_iterations` (the discarded solve's iterations, or None), `steps`, `basis` (Lanczos columns used), `minres_iterations` (per step, max over columns), `inner_cap`, `inner_maxit`, `tau_eff`, `tau_raised`, `r_acc`, `delta_max`, `residuals`, `thresholds`, `neighbours` (`{"lo": (lam, v) or None, "hi": ...}`, for 15b's deflation and 15c's edges), `capacity_full`, `edge_unreached`, `time` (setup, minres, ortho, ritz).

| keyword | default | why |
|---|---|---|
| `capacity` | required | the basis size; 15c sets it from `si_memory` and the budget |
| `sigma` | midpoint | the smallest worst-case δ over the slice, so the fewest steps; the depth of σ, not δ, sets the per-step MINRES cost ([measured](#measured-minres-cost-floor-and-slices)) |
| `R`, `W` | None | rigid modes; known modes to deflate (restarts, neighbouring slices; host array, M-orthonormality checked by a sketch) |
| `tol` | $10^{-6}$ | as `lobpcg` |
| `tau` | $10^{-8}$ | the roadmap's inner tolerance; $r_{acc} \approx 10\tau\delta$ |
| `block` | 6 | as LOBPCG's minimum chunk |
| `maxit_inner` | 20000 | a backstop; later steps are capped at 10× the first |
| `seed` | 0 | the start block |
| `verbose` | False | one progress line per step (basis, MINRES iterations and residual, τ_eff, accepted pairs, time) and on a σ move |

### Slice memory model

`memory.si_memory(op_memory, n_dof, capacity, n_modes, block=6, precond_memory=None, n_rigid=0, n_deflate=0)`: storage = operator + preconditioner + the basis $8n(\lceil n_\text{deflate}/c\rceil c + \text{capacity})$ + R and MR; host = $8n\,n_\text{deflate}$ (W) + $8\,\text{capacity}^2$ (G); phases at width $b$:

- **minres:** operator workspace + 17.5 blocks + the preconditioner's `count` term: 15.5 blocks for the jitted MINRES loop (`memory_analysis` fit with Jacobi: 13.9–15.4 on hex20 $20^3$–$32^3$ and voxel $48^3$–$80^3$, including the best iterate kept per column; up to 18.3 on small voxel grids of 15k–108k DOF, far below the GB range) plus 2 for the right-hand side and the solution; a multigrid adds 0.5–0.96 of its `count` term;
- **reortho:** workspace + 4 blocks (the chunked projector);
- **ritz:** workspace + 6 blocks + one $c$-column chunk of the basis (residuals at width $b$);
- **end:** host V of `n_modes` + 2 columns and one device chunk, $8n(n_\text{modes} + 2 + c)$, while the basis is alive.

total = overhead + storage + host + max(operator build, preconditioner build, phases), `lobpcg_memory`'s conservative form. `n_modes` is the expected slice count (15c estimates it); a slice returning more exceeds the end term.

### Measured MINRES cost, floor and slices

`benchmarks/shift_invert_study.py` (results in `benchmarks/results/shift_invert_study.json`; Apple M-series CPU, fp64, 6 random columns $b = Mq$, τ = $10^{-8}$; σ in the gap above mode $d$ (the depth) at fractions 0.5, $10^{-2}$, $10^{-4}$ of the gap; `--parts` runs one part).

**MINRES iterations against depth and δ** (max over the 6 columns; floor: τ = $10^{-14}$ at the gap midpoint, the smallest explicit relative residual reached):

| depth | hex20 $16^3$, AMG (56k DOF) | floor | voxel $32^3$, fp64 GMG (108k DOF) | floor |
|---|---|---|---|---|
| 6 | 40 / 50 / 60 | 3.3e-14 | 30 / 30 / 40 | 1.9e-14 |
| 20 | 100 / 110 / 120 | 4.7e-14 | 80 / 80 / 90 | 2.5e-14 |
| 55 | 330 / 350 / 380 | 1.3e-13 | 230 / 250 / 270 | 6.8e-14 |
| 100 (voxel 102) | 770 / 810 / 870 | 3.3e-13 | 510 / 540 / stag. 1e-2 | 4.5e-13 |
| 201 | 1710 / 1840 / stag. 3e-2 | 4.3e-13 | 1100 / 1130 / 1210 | 2.6e-13 |
| 503 | 4970 / 5260 / stag. 5e-2 | 9.1e-13 | | |
| 1005 | 14370 (rerun; stag. 6.8e-3 before) / stag. 3e-2 / stag. 3.5e-2 | stag. 6.8e-3 | | |

(the three entries: σ at 0.5, $10^{-2}$, $10^{-4}$ of the gap, δ about $5\cdot10^{-4}$–$0.2$, $10^{-5}$–$5\cdot10^{-3}$, $10^{-7}$–$5\cdot10^{-5}$, falling with the depth; "stag. x": stagnated, x the last iterate's residual (measured before PR 15a's best-iterate fix, which leaves the iterates, flags and counts unchanged and only returns the best checked iterate).)

- **Depth sets the cost, δ barely does.** Iterations grow roughly linearly with the depth, the rate rising with it (hex20: 5–6.7 per mode below depth 100, then 7.7 / 8.5 / 9.9 / 14.3 at depths 100 / 201 / 503 / 1005; voxel 4–5.5), and δ over four decades changes them by at most 20% from depth 20 on (40 to 60 at depth 6). 15c's cost model uses depth.
- **The floor** rises from 3e-14 at depth 6 to 9e-13 at depth 503, far below τ = $10^{-8}$.
- **σ very close to an eigenvalue** (δ below about $10^{-6}$) at depth ≥ 100 can stagnate, at $10^{-2}$–$5\cdot10^{-2}$ (4 of 6 such solves; hex20 depth 100 at δ = 1.1e-6 and voxel depth 201 converged); a first solve like that makes the slice move σ once (later solves raise τ_eff). Its midpoint σ has δ near the gap's half width, far from this regime.
- **Depth 1000 is marginal and expensive:** on hex20 $16^3$ the midpoint solve ran 14370 iterations twice: once a column stagnated (last iterate $6.8 \times 10^{-3}$), once (`--parts recheck_1005`, after the best-iterate fix) every column converged ($9.8 \times 10^{-9}$). The fix cannot cause that (it leaves the iterates and exits unchanged), so the outcome there most likely depends on round-off (the recompiled loop). Either way a slice step costs about 14k iterations, about 4 h at 1M DOF on the Mac; 15c must price it and expect stagnation ([open item](open_items.md)).
- **Mesh size barely matters** (AMG and GMG are h-robust): at 1M DOF the iterations at depth 100 match $16^3$, and at 2M DOF those at depth 6 match $32^3$ (below).

**Slices at the largest Mac sizes** (capacity 10 blocks, tol $10^{-6}$; edges at the coarse mesh's widest gaps):

| mesh | depth | modes | steps | MINRES its per step | $r_{acc}$ | $\tau_\text{eff}$ | $\delta_\max$ | time |
|---|---|---|---|---|---|---|---|---|
| hex20 $44^3$, AMG (1.08M DOF) | about 100 | 5 | 9 | 750–790 | $10^{-6}$ | $10^{-8}$ | 0.010 | 1.7 h |
| voxel $86^3$, fp64 GMG (1.98M DOF) | about 6 | 3 | 9 | 30–40 | $10^{-6}$ | $10^{-8}$ | 0.105 | 4.6 min |

Both converged with σ unmoved and τ_eff = τ. About 1 s per MINRES iteration at these sizes, so a deep slice costs hours on the Mac. In every measured case LOBPCG from 0 was faster (15× at depth 50, below; PR 11b's 2M voxel solve takes 61 s for 10 modes against this slice's 4.6 min for 3): slices pay where LOBPCG's basis no longer fits the memory, the choice 15c makes.

**Floor at the largest Mac sizes** (τ = $10^{-14}$, 6 columns, σ mid-gap after mode 6 of the coarse mesh): $7.0 \times 10^{-14}$ on hex20 $44^3$ (1.08M DOF, 110 iterations) and $3.3 \times 10^{-14}$ on voxel $86^3$ (1.98M DOF, 100 iterations), against $3.3 \times 10^{-14}$ / $1.9 \times 10^{-14}$ at $16^3$ / $32^3$: the floor does not grow with the mesh size. An earlier attempt at the 1M slice's σ (depth about 100) with the pre-fix MINRES ended at $2.3 \times 10^{-3}$ after 980 iterations, though the slice's τ = $10^{-8}$ solves at that σ converged in 780 (other right-hand sides); unexplained, not rerun (the best-iterate fix would lower the reported residual, not the stall).

**A full slice against LOBPCG from 0** (hex20 $12^3$, 25k DOF, modes 51–70 at tol $10^{-8}$, AMG): 22 steps, 138 Lanczos columns (6.6 per mode), 270–280 MINRES iterations per step, $r_{acc} = 1.7 \times 10^{-8}$, 132 s; it returns 21 modes, because mode 70 is a double eigenvalue straddling $\lambda_{hi}$ (kept whole), all within $6.2 \times 10^{-15}\,\lambda_{hi}$ of the reference. `solve_modes(band=(0, f_hi))` gets the same band in 8.5 s (53 iterations): at depth 50 LOBPCG from 0 wins by 15×, as the roadmap's crossover expects.

## Sizes

- **Reference size** means $\le$ `MAX_DENSE` = 20k free DOF: dense `eigh`, $M^{-1}$ and `count_below` are affordable (PR 9's dense $r_M$ and inertia comparisons, PR 14's exact count and the dense estimator of PR 14a's tests).
- **The PR 8 `eigsh`** means `reference_modes` alone: about 100k free DOF on bulky meshes with `points`, about 35k without, further on plates (used by PRs 11, 13, 16, 18).

## Dense inertia count (`count_below`)

`count_below(K, M, sigma, free=None)` returns the exact number of eigenvalues of $(K_f, M_f)$ below $\sigma$ (any finite float; $M_f$ positive semidefinite), by Sylvester's law of inertia on $A = K_f - \sigma M_f$:

- **Factorization.** `A = (K_f - sigma * M_f).toarray(order="F")`, then LAPACK `dsytrf` (Bunch–Kaufman $A = LDL^\top$, the routine `scipy.linalg.ldl` calls) with the queried `lwork` and `overwrite_a`: in place on one dense copy. f2py overwrites only Fortran-ordered input; C order would double the memory, and `scipy.linalg.ldl` returns three $n^2$ arrays. Without the queried `lwork` the unblocked code takes 123 s at 15k instead of 7.5 s. `lower=1` reads one triangle, so round-off asymmetry does not matter.
- **Counting.** $D$ has $1\times1$ and $2\times2$ blocks (`ipiv < 0` marks the rows of $2\times2$ blocks). Bunch–Kaufman chooses a $2\times2$ pivot only when $|a_{kk}|\,\sigma_r < \alpha\lambda^2$ and $|a_{rr}| < \alpha\sigma_r$, $\alpha = (1 + \sqrt{17})/8$, on the entries that form the block. So $|a_{kk}a_{rr}| < \alpha^2\lambda^2 < \lambda^2$ and $\det < 0$: each $2\times2$ block has exactly one negative eigenvalue. The count is the number of negative $1\times1$ pivots plus half the number of $2\times2$ rows.
- **Errors.** An exactly zero pivot ($\sigma$ is an eigenvalue to working precision) raises `ValueError`, as does $n_\text{free} >$ `MAX_DENSE` = 20 000 (3.2 GB, 16 s; 1.9 s at 8k). Counts are exact for $\sigma$ in gaps of relative width $\ge 10^{-3}$; a $\sigma$ inside a near-degenerate pair or the rigid cluster gives an ambiguous count.

**Why two inertia counts.** `count_below` is dense and pivoted (Bunch–Kaufman is backward stable), so it is exact for well-separated $\sigma$: the roadmap's exact count at reference sizes (PRs 9, 14). The check inside `reference_modes` is sparse and unpivoted; it reuses the solver's factorization and runs to about 100k DOF, and a failure is loud.

## Validation

`tests/test_reference.py`, against dense `scipy.linalg.eigh` of $(K_f, M_f)$:

- **Accuracy**, `n_modes = 20`: a free orthotropic hex20 box (consistent and lumped, with nodal masses including zeros: the case that needs pass 2), a clamped curved hex27, the mixed hex8 + hex20 mesh with `xmin: x` (3 rigid modes), and a free hex27 plate $8\times8\times1$ with $L/t = 200$. Rigid modes $|\lambda| \le 10\varepsilon\lambda_\max$, elastic $\le 2 f_\lambda$, $\|V^\top MV - I\| \le 10^{-12}$, constrained rows 0, residuals $\le 100\varepsilon$; an explicit $\sigma = -0.3\lambda_{n+1}$ and the run without `points` give the same $\lambda$. `_order` returns a permutation, also in the tie fallback.
- **Certification**, each check pinned by its message: a dropped mode (count), the real ARPACK failure above (`slow`), a ghost copy (orthonormality), a vector perturbed by $10^{-6}$ M-orthogonally to the others (residuals), a patched SuperLU with `perm_r != perm_c`.
- **Clusters:** the first four `n_modes` ending inside an exact multiplicity of a free isotropic hex20 $4^3$ cube certify and match up to the last gap; `n_modes = 3` on a free box raises; on the $L/t = 1600$ plate, `n_modes` 7, 10, 12, 17, 21, 25 certify under the automatic σ rule (a purely relative gap threshold falsely raises at 10, 12, 17).
- **Inputs** raise `ValueError`; `restrict` matches dense slicing.
- **`count_below`** matches the dense `eigh` count at mid-gap $\sigma$ (relative gap $\ge 10^{-3}$, including the first gap above the rigid cluster) on a clamped hex27 and a free hex20 box, consistent and lumped; these cases have $2\times2$ pivots.

`tests/test_lobpcg.py`, against `reference_modes` and dense linear algebra:

- **Accuracy** at tol $10^{-10}$, 12 modes, on clamped hex20 $6^3$, free hex20 $6^3$ (rigid deflation, $\|R^\top MV\| \le 10^{-10}$), clamped hex27 $4^3$ and clamped hex8 $10^3$.
  - Modes with relative gap $\ge 10^{-2}$: λ error $\le \max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$, with $\lambda_\max$ from `eigsh(which="LA")`. Gaps in $[10^{-4}, 10^{-2})$: $\le 10\,$tol, cluster-wise.
  - Principal sines $\le 10\,r_M\lambda/\mathrm{gap}$, with the dense $r_M$. The reference's $r_M$ is $\ge 10\times$ smaller. Sines are computed stably, as $\sigma_\max(L^\top(Q - Q_rQ_r^\top MQ))$; $\sqrt{1-\cos^2}$ bottoms out near $10^{-8}$. Clusters below a relative gap of $10^{-4}$ are compared by subspace.
  - $\|V^\top MV - I\| \le 10^{-10}$, constrained rows 0, and `info["residuals"]` equal to residuals recomputed with the operator.
  - No P drops, and `norm_K`, `norm_M` within $[0.8, 1]\times$ the exact 2-norms (`eigsh`).
- **Exact multiplicities** on a free square-section beam and a free cube. A beam of section $1\times1.002$ gives bending pairs in the $[10^{-4}, 10^{-2})$ band.
- **Stagnation:**
  - tol $10^{-16}$ gives floor acceptance;
  - with the floor term patched to 0, the no-progress exit;
  - a 1-mode run is not flagged.
- **Robustness:**
  - a start block with duplicated columns (refill);
  - a forced $P \approx XC$ at noise $10^{-12}$ (the Cholesky fails) and $10^{-6}$ (finite, pivot below `PIVOT`): P is dropped, the new $X$ stays M-orthonormal;
  - the recompute schedule (iterations 20, 40) and explicit products after a recompute step;
  - the P invariant after 30 steps ($\le 10^{-6}$);
  - `_p_coeffs` masks exactly a noise column;
  - near-singular M within 500 iterations;
  - a NaN preconditioner raises `FloatingPointError`.
- **Compiles:** one step compile per solve, and 0 for a second solve with a new operator, preconditioner and tol (subprocess, cache off, `jax.monitoring`, filtered by the exact `fun_name`); a 3-chunk solve compiles the step once and a second one with new data nothing; the low-memory step is one more variant.
- **Memory:** with a diagonal operator, step memory $\le 16.5$ blocks (9-block) and $\le 7.5$ (low-memory), the same to 0.01 blocks for capacities $c\lceil 2m/c\rceil$ and 4× that, fp64 or fp32, and without a basis. With the hex20 operator ($24^3$ at $m$ = 15, 48; $8^3$ at $m = 15$) and the held-out hex8 $20^3$ and hex27 $12^3$ ($m = 24$) the step is within `lobpcg_memory`'s step term, in both modes. At 1.08M DOF (slow, macOS, one chunk) the measured peak is within 20% of the estimate (≥ 2 GB); a chunked diagonal-pencil solve (slow, macOS, fp64 and fp32 basis) peaks below the estimate.
- **Chunks** (PR 9b), against `reference_modes`:
  - 30 modes in 3 chunks of 10, clamped and free hex20 $6^3$, both modes: the accuracy checks above over all 30 modes, every chunk converged and none stagnated;
  - the four accuracy cases in the low-memory mode, λ matching the 9-block λ to $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$; the P invariant also holds for the low-memory step;
  - a free cube (multiplicities 2, 3, 3, 2, 1, 3, 3), 12 modes in chunks of 4: locks of 5, 5 and 4 modes, the return cut at 12 (`V` has 12 columns), exact multiplicities; with one guard the guard error ("guards >= 2"), but not for a single chunk (4 modes, one guard: no error, as in 9a);
  - `_capacity`: the worst-case lock writes stay within the capacity;
  - an fp32 basis at tol $10^{-6}$ (λ within the rule above, $\|V^\top MV - I\| \le 10^{-5}$, the count below the λ₃₀/λ₃₁ midpoint equal to `count_below`'s and to 30) and at tol $10^{-10}$ (chunks 2–3 floor-accept, chunk 1 does not);
  - the projector against the one-pass fp64 one to $10^{-14}$ (count $3c + 5$, fp64 and fp32 basis) and its temp $\le 1.01\,(c + m)\,n \cdot 8$ B, the same for capacities $2c$ and $8c$;
  - in-place insertion (`is_deleted`, same buffer pointer) and a full basis (raises before writing; the basis is intact);
  - `_reortho` (M-orthogonal to a basis whose count is not a multiple of the chunk width, inputs with norms ≠ 1): zero tail, M-orthonormal columns M-orthogonal to the basis ($10^{-13}$), column order kept, two projection passes needed for columns $10^{-6}$ off the span, a column inside the basis span flagged by its projected norm, and the solve raising on it (monkeypatched);
  - 300 modes in 30 chunks on a diagonal pencil (20k DOF, Jacobi residuals up to $10^6\times$ along the locked coordinates; slow): converged, $\|V^\top MV - I\| \le 10^{-13}$, λ exact to $10^{-9}$ (without the re-orthonormalization at lock chunk 9 fails);
  - a failed second chunk (`X0` = reference modes, `maxit` 5): 20 modes, chunks `[True, False]`, all 20 M-orthonormal;
  - `ValueError` for `chunk` 0 or above `n_modes`, `guards` 0, `basis_dtype="float16"` and chunk + guards above the free DOFs.
- **Floor** (slow): within $[0.1, 10]\times\varepsilon\lambda_\max/\lambda_1$.
- **Jacobi, σ and errors:** Jacobi against the assembled diagonals; σ within $[\lambda_1, 2\lambda_1]$; `ValueError` for a too-large block, a bad `X0`, `n_modes` or `maxit` below 1, and a non-M-orthonormal R; `RuntimeError` after 3 refills; the `maxit` and no-progress exits run the explicit confirmation (counted).

`tests/test_api.py`, `solve_modes` against `reference_modes` (10 modes, tol $10^{-10}$, the λ rule above; the rigid count pinned by the reference's zero eigenvalues):

- PR 11b: the GMG-refined σ pre-estimate within $[\lambda_1(1 - 10^{-6}), 2\lambda_1]$ of the SciPy $\lambda_1$ on boxes, an odd grid, a thin plate, beams, free boxes, plates and the free $128\times1\times1$ beam (slow: anisotropic spacing and the SIMP field), with a spy on the builds (clamped: once at 0; free: from σ₀ > 0, never rising); the round logic on scripted probes (rebuild while σ drops by more than 3×, the minimum kept, at most 4 rounds); free cases rebuild only after a drop by more than 3×; the final LOBPCG gets the GMG preconditioner in the requested dtype, and Jacobi built at a σ hint; a non-`SingularShiftError` at σ = 0 propagates without the `sigma_estimate` fallback; the hint skipping the pre-estimate (one build at the hint, same modes) and `sigma=0` on a free body raising; GMG solves on odd $9\times6\times5$ meshes (clamped, free, masked with nodal masses) within $10^{-8}$ in λ and MAC $\ge 1 - 10^{-8}$, the final build at `info["sigma"]`; the dispatch and option errors; that only the σ = 0 refusals are `SingularShiftError` (a backstop and a plain `ValueError` propagate); the $87^3$ Mac acceptance (slow, macOS) and its benchmark entry's smoke run;
- PR 13b: AMG solves (hex20 $5\times4\times3$, hex8 $6\times5\times4$; clamped, free with nodal masses, sliding) within $10^{-8}$ in λ and MAC $\ge 1 - 10^{-8}$, with a spy on the successful `amg` builds (clamped: the first at σ = 0 with `nonsingular=True`; free and sliding, with rigid modes left: σ > 0, `nonsingular=False`; the final build at `info["sigma"]`); the AMG-refined pre-estimate within $[\lambda_1(1 - 10^{-6}), 2\lambda_1]$ on 7 cases (boxes, beam, plates, free; slow: the free plate); a hint skipping it; `"auto"` → AMG on unstructured meshes, the `"amg"`-on-voxel and fp32-without-GMG errors; `solve_modes_memory` on a hex20 box equal to `lobpcg_memory` with `amg_memory` (default) and without a preconditioner term (`precond="jacobi"`; `tests/test_lobpcg.py`); the 1M hex20 Mac acceptance (slow, macOS, `tests/test_amg.py`: ≤ 15 min, peak within 20% of `solve_modes_memory`) and its smoke run;

`tests/test_options.py` (PR 13c):

- the calls with every option `None` equal a snapshot of the pre-13c calls (AMG and GMG); each operator and preconditioner key reaches the operator build or every pre-estimate and final build with its value; `X0` reaches the final `lobpcg`, `seed` the pre-estimate and `sigma_estimate`;
- against `reference_modes` (λ to $10^{-8}$, MAC $\ge 1 - 10^{-8}$): `mode="otf"`, `max_coarse` 100 (one more AMG level on hex20 $8^3$), GMG `degree` 3 / `coarse_degree` 2; `X0` = the converged `V` in ≤ 2 iterations (clamped, free); a too-wide `X0` raises;
- `sigma_options` on scripted probes (`rounds` 2, `drop` 10 and 2, `tol` / `maxit` / `estimate_maxit` / `seed` recorded) and on real meshes (`maxit` 1 gives 1 iteration; `estimate_maxit` reaches `sigma_estimate`);
- unknown, misplaced (`slab` on unstructured, `max_coarse` with GMG, any key with Jacobi, `mass`) keys and invalid values (bool counts included; NumPy integers pass) raise `ValueError` naming them, before any build;
- `solve_modes_memory` with options equals the composition of the estimators with the same options and checks the count values; the loop terms start at degree 3; `amg_memory`'s storage bounds the owned bytes at `max_coarse` 100–3000 on three meshes; the LOBPCG steps at AMG degree 4 (hex20 $8^3$, $12^3$, hex8 $12^3$) and GMG degree 4 / `coarse_degree` 8 (voxel $8^3$, $16^3$ fp64 and fp32, $48^3$ slow) within the estimate; `GMG_LOOP` and `AMG_LOOP_K` within 0.8–1× (fp32 0.7–1×) and 0.6–1× of the measured degree-4 minus degree-2 increase (voxel $16^3$, $48^3$ slow; hex20 $8^3$);
- the builder signatures minus the reserved parameters equal the allowed keys (a new builder option fails until exposed or reserved);
- the Chebyshev loop equals the unrolled iteration to $10^{-14}$ (degrees 1–5), keeps fp32 in the fp32 V-cycle, and its AMG V-cycle temporaries at degree 4 stay ≤ degree 2 (hex20 $16^3$);
- the `benchmarks/options_study.py` smoke run (slow).

`tests/test_api.py` also covers:

- clamped hex20 box (unstructured dispatch, `V` M-orthonormal with constrained rows 0, `frequencies`); free with random nodal masses (6 rigid modes, `info["rigid"]` M-orthonormal and M-orthogonal to `V` under the full mass; the nodal masses change λ observably); sliding `xmin: x` (3 rigid modes);
- voxel, clamped and free (stencil dispatch, counted by spies on the builders);
- hex27 with a rotated transversely isotropic phase on half the elements and HRZ lumped mass (the rotation changes λ observably);
- every `lobpcg` option forwarded (two option sets, 3 chunks each);
- bad BCs raise before any operator is built; a missing material, $\rho \le 0$ and an inverted hex20 raise in the builders before `lobpcg`.

`tests/test_counting.py` (PR 14a; reference mesh hex20 $6\times5\times4$ with spacing $(1/6, 1/5, 1/4)$, $E = 3$, $\nu = 0.3$, $\rho = 2$, xmin clamped, 1962 free DOF, cut after mode 6 at $g = 0.41$, exact dense $A_P^{-1}$ as the preconditioner unless the test is about it):

- **Factors and covariance:** $FF^\top = M_e$ from `element_matrices` to $10^{-12}$ (hex8, hex20, hex27, padded elements 0) and $BB^\top = m_e$ of `reference_element`; $SS^\top$ masked equals the masked SciPy M to $10^{-12}$ (hex20 with nodal masses, two phases and $s_M$; mixed hex8/hex20; lumped; voxel with nodal masses; voxel lumped); a probe's column depends only on its key.
- **Polynomial:** the DCT grid against the direct sum; $-10^{-15} \le q \le 1 + 10^{-15}$; `count_degree` meets $q(x_l)^2 \ge 0.95$ at $d$ and fails at $d - 1$, $d g \in [19, 24]$ for $g$ = 0.5–0.01; a bias-limited case ($g = 0.5$, $n_\text{top} = 5000$); `tail_bias` against `scipy.integrate.quad` to $10^{-3}$; the `MAX_DEGREE` exit values.
- **Expected value:** the mean of 2000 probes within 5 standard errors of $\sum_{i\notin W} q(\lambda_i)^2$ from `scipy.linalg.eigh`: $W$ complete, mode 3 left out (consistent, lumped, nodal masses), a free body with $R$ and with $R$ omitted (about 6 more).
- **Against the dense estimator** on the same probes (hex20, lumped hex20, voxel $4^3$): $10^{-8}$ relative (mode left out) and $10^{-10}$ absolute ($W$ complete) at $\tau = 10^{-12}$; within the linear error bound at the default $\tau$.
- **Detection:** one 5000-probe count in 100 sets of 50, mode 3 left out: ≥ 97 sets ≥ 0.5 (measured 100); $W$ complete (slow): all sets < 0.5.
- **$\kappa \ge 10^7$:** hex20 cantilever $64\times1\times1$ ($\kappa = 2.2\cdot10^9$), AMG at $\lambda_c$ (9 PCG iterations per solve), default $\tau$: verifiable, estimate < 0.5. A thin plate is not used: AMG on its flat elements (aspect 8) needs 400–480 iterations per solve and stagnates at $\tau \le 10^{-10}$.
- **Flags:** the floor measured in the test (`count(degree=1)` on the same probes, Jacobi; $10^{-15}$), then $\tau$ = floor/10 and $10^{-17}$: NaN, unverifiable, `probes` NaN, early stop; `maxit` in the recurrence (lumped, `steps` ≥ 1) and in the y solve (exact preconditioner, `steps` 0); a failed y solve (monkeypatched `_solve_y`: `steps` 0); the default τ rule.
- **Memory:** the recurrence's temporaries within `count_memory`'s step (hex20 $12^3$ with Jacobi and AMG, voxel $16^3$ with Jacobi and GMG, $b$ = 4, 16, 32; Jacobi ≥ 0.6× tight; the multigrid increment over Jacobi 0.7 (AMG) / 0.45 (GMG) to 1.0 of its `count` term); the sampling kernels within the sampling phase, itself below the step; `probe_block` at the budget boundary; the process peak of a 3-block count at ≥ 2 GB within 20% with a host $W$ (voxel $60^3$, $b = 24$: 2.29 GB against 2.23 GB; slow, macOS).
- **Blocks and compiles:** $b = 7$ and $b = 50$ give the same $s_j$; degrees 100 and 120 share one compile; an fp32 $W$ changes the estimate by ≤ $10^{-4}$ and not the verdict; input errors; the benchmark smoke run (slow).

`tests/test_examples.py` runs each script in `examples/` in a subprocess; it must exit 0 and print its check line.

The discretization itself is verified in [`verification.md`](verification.md).

`tests/test_band.py` (PR 14b), against dense `scipy.linalg.eigh` and its inertia:

- **Band rule:** `_band_count` on synthetic spectra (exact edge, inside / outside the 100 $t$ window, a cluster chain, all and none in the band).
- **No missed modes:** the clamped hex20 reference box and a free hex8 box (rigid modes deflated), $f_{max}$ in a clear gap (`verify=True`), 10× inside and outside the 100 $t$ window below mode 4 (thresholds from a first solve), 19 modes in chunks of 8, and a voxel box with GMG: the returned count equals the dense count below the edge, λ to $10\,\mathrm{tol}$; the count receives every converged column (M-orthonormal to $10^{-10}$), the roadmap's edges, $\lambda_c$ and $g$ to the dense values, R on the free box, and the preconditioner built at $\lambda_c$.
- **Roadmap cases:** $f_{max}$ at half the first frequency (clamped and free: 0 modes, count verifiable < 0.5); a double eigenvalue (square beam) returns both copies, also with `chunk=1` ending between them, count < 0.5; W missing mode 1 (not next to the cut) gives an estimate ≥ 0.5.
- **verify:** `False` builds and counts nothing and reports $n_\text{blocks}\,d$ and the bound ($c = 2$, 0.3 with a looser `tau`); `True` runs without calibration; `None` runs or skips on the patched calibration, skips without building when the bound exceeds the solve, and reports $t_\text{build} + t_\text{cal} + n_\text{blocks}(d\,t_\text{step} + t_\text{block})$; no warm iteration time → `predicted_s` None and calibration; `MAX_DEGREE` patched to 4 → `None` skips unbuilt, `True` returns NaN; a budget below one probe → `None` skips, `True` keeps the modes and warns; a failed solve (`maxit` 3, one chunk and chunks of 8) → `converged` False, no count, a warning under `True`. Slow: the calibrated prediction within $[0.5, 2]\times$ the actual count (hex20 $12^3$, 50 steps; one run: 14.66 against 14.65 s, the bound 2.20 s) and the bound below $d\,t_\text{step}$.
- **Growth, edge, block, errors, memory:** chunk 8 from an expected count of 8 grows the basis and equals a full-capacity run to $10^{-12}$ (fp64 and fp32 basis); $f_{max}$ above the spectrum of a 54-DOF mesh raises; the chunk defaults; the probe block is the largest with `count_memory` ≤ the budget, which is the solve's estimate minus the host modes (and with AMG the kept aggregates), at the grown capacity after growth (also from no basis: chunk 8 above an expected count of 4), with the λ_c multigrid at fp64 after an fp32 V-cycle solve; the returned `V` is a copy, not a view of the solve's buffer; argument errors; `solve_modes_memory(band=...)` = `lobpcg_memory(..., band=True)`; slow (macOS): the measured peak of a band solve with `verify=True` within 20% of the estimate (voxel $48^3$, 2.62 GB estimated, 2.83 GB measured).


`tests/test_shift_invert.py` (PR 15a; references `eigsh(sigma=0)` on the assembled free blocks or dense `eigh` below 5k free DOF, cached per mesh; `count_below` for slice counts; tol $10^{-10}$, τ $= 10^{-8}$; edges at gap midpoints moved out to relative gaps $\ge 10^{-2}$):

- **Slices against the reference:** an interior slice around modes 31–45 on hex20 $8\times6\times5$ (AMG, `max_coarse` 100) and voxel $10^3$ (GMG) (slow); a slice from $\lambda = 0$ (no lower neighbour); a free hex8 box with R (above the rigid modes and from 0, where undeflated rigid Ritz values would land inside); the free voxel cube's double and triple clusters with exact multiplicity; an empty slice inside a gap (with `verbose`, one progress line per step). Each checks: converged in under half the capacity's steps, the dense count, λ to $\max(10^{-8}, 10\varepsilon\hat\lambda_\max/\lambda)$ (gap $\ge 10^{-2}$) or $10\,r_{acc}$, $\|V^\top MV - I\| \le 10^{-10}$, $r_i \le t_i \le r_{acc}$, a converged neighbour beyond each edge (true residual $\le r_{acc}$), and $r_{acc} = \max t_i$ over the pairs and neighbours (raised by the neighbours on the interior slices). An edge between the copies of a double eigenvalue: converged, both copies, λ to $10^{-8}$.
- **σ on an eigenvalue** (slow): MINRES at mode 21 stagnates (pre-check); σ moves once (`move_iterations` = the discarded solve's iterations), the inner cap comes from the repeated start, the slice matches; with the move patched to 0 the slice stops ("inner solves failed").
- **τ_eff:** τ $= 10^{-16}$ raises τ_eff to the largest stagnated residual without moving σ (`move_iterations` None) and the slice converges; the inner caps (`maxit_inner` first, then 10× the first step); `maxit_inner` = 20 sets `inner_maxit` and stops the slice; inner solves stopped at $10^{-6}$ (reported stagnated) give $t_i = 10\,\tau_\text{eff}\,\delta_i$ with residuals above tol, all within $t_i$, λ within $10\,r_{acc}$; a NaN MINRES residual stops the slice; a final residual pass above the thresholds makes it unconverged.
- **Sets and stop rule** on synthetic values: values reached only through the 100r margins or only through chains (past $\lambda_{hi}(1 + 100r)$), lo = 0, a missing neighbour, an empty slice; `_slice_check` with a fake residual function: unconverged pairs and neighbours, a neighbour merged in the same call (residuals asked only for new indices), a chain of re-forms.
- **Capacity:** capacities 48 and 72 on a 12-mode slice (2 and 11 converged pairs returned, `edge_unreached` "both"); $\lambda_{hi}$ above the spectrum ("hi"); $\lambda_{lo} = \lambda_1/4$ ("lo"); a capacity above the 54 free DOFs stops at the full space (54 = 9 blocks) with the exact top modes.
- **W, refill, errors:** half of a slice's modes as W: the other half returned, M-orthogonal to W; a start block with two equal columns is refilled; argument errors (edges, fp32 GMG, block, capacity, R and W shapes, a non-M-orthonormal W, a free space below 2 blocks).
- **Memory:** `si_memory`'s composition (storage, host, end phase, total); process peaks within 20% at estimates ≥ 2 GB (slow, macOS): MINRES-dominated (voxel $80^3$, capacity 4 blocks; 3.10 GB measured against 2.80 GB), basis-dominated (voxel $48^3$, capacity 744; 3.03 against 2.79 GB), and `_form_v` alone on an 800k × 240 basis of ones (2.96 against 2.66 GB; values do not matter for memory, and `jax.random.normal` alone peaks at 2.5× the basis).
- The benchmark smoke run (slow).
