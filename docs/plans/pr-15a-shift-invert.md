# PR 15a plan: block shift-invert Lanczos on one slice

Branch `pr-15a-shift-invert` from `main` (PR 14b merged). Roadmap: PR 15 ("Interior bands"),
the "Frequency bands" section (interior bands, band edges, completeness count) and the
"Preconditioners" rule that fp32 V-cycles never run inside MINRES.

## Split of PR 15

PR 15 is three PRs, in dependency order; every PR 15 roadmap test is assigned to one of them
and none is dropped:

- **15a (this plan):** the block shift-invert Lanczos solver for one slice
  $[\lambda_{lo}, \lambda_{hi}]$ (`eigen/shift_invert.py`), its inner MINRES rules ($\tau$,
  $\tau_\text{eff}$, the floor), $r_{acc}$, the slice stop rule, the measured MINRES cost and
  floor (benchmark), the slice memory model.
- **15b:** the band-pass count (two-edge polynomial, neighbour deflation) and the slice restarts
  (count ≥ 0.5 → fresh block deflated by the found modes, at most 2 restarts).
- **15c:** slicing with the cost model (widths, edge snapping, capacity splits, edge re-check,
  duplicate removal, slice collapse), the strategy choice between LOBPCG from 0 and
  shift-invert (cost and memory), `solve_modes(..., band=(f1, f2))` and the f₁-below-λ₁ rule.

Test assignment (roadmap PR 15 tests):

| Roadmap test | Part |
|---|---|
| matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 15a (one slice), again in 15c (bands) |
| clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` (λ = 0 edge) | 15a (slice with λ_lo = 0), 15c (through `solve_modes`) |
| repeated interior eigenvalues of the voxel cube with exact multiplicity | 15a |
| MINRES iterations vs δ measured, cost constants fitted; fp64 V-cycle MINRES floor measured, recorded | 15a (measured), 15c (cost model fitted on them) |
| τ forced below the floor raises τ_eff, carried into r_acc, `info` flag | 15a |
| r_acc and the floor at the largest Mac size recorded | 15a |
| memory model (deflation basis, Lanczos, MINRES workspace) within 20% at an estimate ≥ 2 GB | 15a |
| interior band with 0 modes, count < 0.5; planted mode left out → count ≥ 0.5 | 15b (count), 15c (end to end) |
| slice with a removed mode: count ≥ 0.5, flag, recovered on restart; withheld: 2 restarts then flagged | 15b |
| detection ≥ 97/100 seeds; neighbour-deflation identity to 1e-8; complete V < 0.5 | 15b |
| each slice's count < 0.5 | 15c |
| capacity split at the largest gap; full slice without a gap raises | 15c |
| edge failing the final r_acc re-check widened or merged; τ_eff = 1e-6 sets the slice-collapse flag | 15c |
| f₁ below λ₁ solved as [0, f₂] and filtered | 15c |
| memory budget too small for LOBPCG from 0 → shift-invert | 15c |
| voxel cube band across ≥ 2 slices, cluster near a slice edge, tol 1e-6 | 15c |
| band edge on a double eigenvalue returns both copies, d ≈ 25/g | 15c |

## Facts (prototypes in the scratchpad, `p15/`)

- MINRES on $A_P(-\sigma) = K - \sigma M$ (masked) with the SPD AMG of $K + \sigma M$
  (`ap_operator(op, free, -sigma)`, `amg(op, free, sigma, ...)`), 6 columns, τ = 1e-8, hex20
  $12^3$ clamped (24.8k DOF): about 40 iterations with σ near mode 6, 270–300 near mode 51,
  nearly independent of δ = min |λ − σ|/σ from 0.2 down to 4e-6. The iteration count is set by
  the depth of σ in the spectrum (the number of negative eigenvalues of $K - \sigma M$), not by δ,
  until σ hits an eigenvalue: σ exactly on the double mode 21–22 stagnated at a residual of 0.1.
- Block shift-invert Lanczos prototype (b = 6, full M-reorthogonalization, Ritz values from the
  projected pencil, true residuals): hex20 8×6×5 clamped, slice holding modes 31–45 at tol 1e-10:
  18 steps, 114 basis columns (7.6 per mode), 280 MINRES iterations per step, all 15 modes to
  8e-15 relative against `eigsh`. Voxel 10³ (GMG): 18 steps, 204 iterations per step; the slice
  edge fell inside a cube degeneracy and the cluster came back whole (16 modes).
- `linalg.minres` (PR 10) checks the explicit 2-norm residual every 10 iterations, reports
  `converged`, `stagnated` (5 checks without a new minimum) and per-column `residuals`; its tol is
  relative to $\|b\|_2$ with $b = Mq$, the roadmap's $\|r_{in}\|_2 \le \tau \|Mq\|_2$.
- LOBPCG's λ-scaled residual and threshold (`lobpcg._resid`): $r = \|Kx - \lambda Mx\|/(|\lambda|
  \|Mx\|)$, threshold $\max(tol, c\,\varepsilon(\|K\| + |\lambda|\|M\|)\|x\|/(|\lambda|\|Mx\|))$.
  The chunked projector `lobpcg._deflate` (R plus a preallocated basis, fp64 upcast in chunks of
  c columns) is reused for the Lanczos reorthogonalization.

## Design

### 1. `eigen/shift_invert.py`: `si_slice`

```python
lam, V, info = si_slice(op, free, lo, hi, precond, *, sigma=None, R=None, W=None, tol=1e-6,
                        tau=1e-8, block=6, capacity=None, maxit_inner=20000, max_steps=None,
                        seed=0)
```

Internal (exported from `jaxmodal.eigen` for the tests and PR 15c; `solve_modes` gains it in
15c). `lo`, `hi` are eigenvalues ($\lambda = (2\pi f)^2$), `0 <= lo < hi`. `precond` is the SPD
preconditioner of $K + \sigma M$ (fp64 GMG/AMG or Jacobi at +σ; a GMG with fp32 levels raises:
fp32 asymmetry floors MINRES near 1e-6). `R`: the M-orthonormal rigid modes (deflated, as in
LOBPCG). `W`: known M-orthonormal modes to deflate (restarts in 15b, neighbouring slices in 15c);
host array, copied into the front of the basis.

- **Shift.** σ defaults to the midpoint $(\lambda_{lo} + \lambda_{hi})/2$. If the first block's
  MINRES stagnates with a residual above $10^3\tau$ (σ on an eigenvalue, as the double mode in the
  prototype), σ moves by 1% of the slice width toward $\lambda_{hi}$ and the start repeats once
  (`info["sigma_moved"]`); a second stagnation continues with τ_eff (below).
- **Operator.** $T = (K - \sigma M)^{-1} M$ on the free DOFs: one step solves
  `minres(ap_operator(op, free, -sigma), m * M(Q_j), precond, tau, maxit_inner)` for the newest
  block $Q_j$ (n × b), masks the result and deflates it.
- **Basis.** A preallocated fp64 device array `Q` of `capacity` columns (a multiple of the
  projector chunk c = max(b, min(4b, 64))) holds $[W \mid Q_1 \mid Q_2 \mid \dots]$; the count of
  filled columns is a traced int (one compiled step for all steps, as LOBPCG's locked basis).
  Default capacity: `len(W) + b * max_steps` with `max_steps` default 100 (600 Lanczos columns);
  15c sets it from the memory model.
- **Step (jitted).** $Z \leftarrow T Q_j$ (MINRES, outside the jit), then
  $H_{:, j} = Q^\top M Z$ for all filled blocks (one M apply on Z, one chunked product), two
  passes of `lobpcg._deflate` against R and the filled basis, then M-orthonormalization by SVQB
  (`lobpcg._svqb_t` on the Gram matrix). A column dropped by SVQB (rank loss: an invariant
  subspace found, or a converged direction) is replaced by a fresh random masked vector, deflated
  and orthonormalized, so the block keeps width b (the roadmap's "same column masking as
  LOBPCG"). $H$ (host, (steps·b)²) is the projected $Q^\top M T Q$, symmetrized.
- **Ritz values.** $\theta$ = eigenvalues of $H$; $\mu = \sigma + 1/\theta$ for $|\theta| >
  10^{-14}\max|\theta|$. Candidates: the Ritz values in $[\lambda_{lo}(1 - 100 t),
  \lambda_{hi}(1 + 100 t)]$ plus the nearest one beyond each edge (t = the running $r_{acc}$).
  Their Ritz vectors $X = QY$ (one chunked product) get true residuals and thresholds with
  `lobpcg._resid` (one K and one M apply on the candidates; $\|K\|$, $\|M\|$ from
  `lobpcg._norms` once per slice).
- **Accuracy target.** $r_{acc} = \max(tol, 10\,\tau_\text{eff}\,\delta_{max})$ with
  $\delta_{max} = \max |\mu - \sigma|/|\mu|$ over the in-slice candidates; $\tau_\text{eff} =
  \tau$ unless a MINRES column ends stagnated or at `maxit_inner` above τ: then $\tau_\text{eff}$
  = the largest relative residual such a column reached (the measured floor of that solve),
  `info["tau_raised"]` True (`info["inner_maxit"]` for the cap). A converged column never lowers
  τ_eff. A Ritz pair is converged when its residual ≤ max($r_{acc}$, its `_resid` floor).
- **Stop rule** (checked after every step, the roadmap's slice stop rule): every candidate with
  $\mu$ in $[\lambda_{lo}, \lambda_{hi}]$ is converged, the in-slice set is extended over clusters
  (relative gap < $100\,r_{acc}$, `lobpcg._band_count`'s window rule on both edges), and at
  least one converged Ritz value lies beyond each edge, past a gap ≥ $100\,r_{acc}$ from the
  extended set. An edge at $\lambda_{lo} = 0$ needs no neighbour (free bodies arrive with R
  deflated). With no Ritz value in the slice, the rule is the two neighbours alone (an empty
  slice).
- **Capacity full / step cap.** When the next block does not fit (or `max_steps` is reached)
  before the stop rule holds, the slice returns its converged in-slice pairs with
  `converged` False and `info["capacity_full"]` True (15c splits the slice at its largest gap).
- **Final Rayleigh–Ritz.** On the converged in-slice vectors plus the converged neighbours: the
  projected $X^\top K X$ (X M-orthonormal), a small `eigh`, the rotated vectors, explicit K and M
  products, residuals and thresholds recomputed (the roadmap's "final Rayleigh–Ritz with true
  residuals on (K, M)"). A pair that falls above $r_{acc}$ there sets `converged` False.
- **Returns** `lam` (ascending, in the slice after cluster extension), `V` (host fp64, n × k, a
  compact array) and `info`: `converged`, `sigma`, `steps`, `basis` (columns used),
  `minres_iterations` (per step: max over columns), `tau_eff`, `tau_raised`, `inner_maxit`,
  `sigma_moved`, `r_acc`, `delta_max`, `residuals`, `thresholds`, `neighbours` (`lam`, `V` of the
  converged neighbours beyond each edge: 15b deflates them), `capacity_full`, `time`
  (setup, minres, ortho, ritz).

### 2. Benchmark (`benchmarks/shift_invert_study.py` → `results/shift_invert_study.json`)

- **MINRES iterations vs δ and depth:** hex20 $16^3$ clamped (AMG) and voxel $32^3$ clamped
  (GMG fp64), 6 random columns $b = Mq$, τ = 1e-8, σ at fractions 0.5, 0.1, 1e-2, 1e-3, 1e-4 of
  the gap after modes 5, 20, 50, 100, 200 (reference eigenvalues from `solve_modes`, gaps of
  relative width ≥ 1e-3, so the fraction sets δ); iterations (max over columns), δ, and the
  number of eigenvalues below σ. This is the data 15c fits the cost model to.
- **Floor:** the same solves at τ = 1e-14 report the smallest explicit relative residual MINRES
  reaches (the fp64 V-cycle MINRES true-residual floor) per depth and mesh.
- **Largest Mac size:** hex20 1M DOF (the PR 13b acceptance mesh) with AMG, σ near mode 100:
  the floor (τ = 1e-14, one 6-column solve) and $r_{acc}$ for a slice there
  ($10\,\tau_\text{eff}\,\delta_{max}$ with the slice's δ_max from the reference eigenvalues of
  a short `solve_modes` run).
- **A full slice:** hex20 $12^3$, modes 51–70 at tol 1e-8: steps, basis columns per mode, MINRES
  iterations per step, seconds, and the same band by LOBPCG from 0 (`solve_modes(band=...)`),
  as a first data point for 15c's crossover.

### 3. Slice memory model (`memory.si_memory`)

```python
si_memory(op_memory, n_dof, capacity, block=6, precond_memory=None, n_rigid=0)
```

storage = operator + preconditioner storage (fp64) + basis $8\,n\cdot$capacity + R and MR;
phases: MINRES (x, r1, r2, y, w, w2, b and the explicit check: about 9 vectors per column, plus
the operator and V-cycle workspace at width b), reortho (the chunked projector at width b, as
`lobpcg_memory`'s projector term), Ritz (candidates X, KX, MX: 3 vectors per candidate, at most
`2 b + slice modes`); total = overhead + storage + max(phases). The MINRES per-column constant
is fitted from `memory_analysis` of the jitted MINRES loop on the stencil and unstructured
operators (as COUNT_STEP in 14a), envelope over b = 6 and 12. 15c composes it into
`solve_modes_memory`.

## Tests (`tests/test_shift_invert.py`)

References: SciPy `eigsh(K_f, k, M_f, sigma=σ)` on the assembled free blocks (and dense `eigh`
on meshes ≤ 5k free DOF), the dense `sytrf` inertia (`count_below`) for slice counts.

1. **One interior slice vs `eigsh`** (hex20 8×6×5 clamped, AMG; voxel 10³ clamped, GMG):
   slice edges at the midpoints of gaps ≥ 1e-2 around modes 31–45, tol 1e-10: the count equals
   `count_below(hi) - count_below(lo)`; λ within max(1e-8, 10 ε λ̂_max/λ) of `eigsh` for clusters
   with relative gap ≥ 1e-2, within 10 r_acc otherwise; V M-orthonormal to 1e-10; true residuals
   ≤ r_acc; `neighbours` hold one converged value beyond each edge.
2. **λ = 0 edge** (the roadmap's clamped (0, f_max) on shift-invert, at slice level): clamped
   hex20, `lo = 0`, `hi` in a gap after mode 12: the 12 modes vs `eigsh`, no neighbour below,
   one above.
3. **Free body:** free hex8 box with R deflated, slice after the rigid modes: no rigid mode
   returned, the elastic modes vs dense `eigh`.
4. **Repeated eigenvalues** (voxel cube 8³, free with R deflated, or clamped on one face with
   the x–y symmetry): slices containing a double and a triple cluster return each with its exact
   multiplicity (dense `eigh`: cluster sizes from relative gaps < 1e-8).
5. **Empty slice:** a slice strictly inside a gap: 0 modes, `converged`, both neighbours.
6. **Edge cluster:** a slice edge between the two copies of a double eigenvalue: both copies
   returned (cluster extension).
7. **σ on an eigenvalue:** σ passed exactly at a reference eigenvalue: `sigma_moved` True and the
   slice still matches `eigsh`.
8. **τ below the floor:** τ = 1e-16: `tau_raised` True, τ_eff ≥ the floor the benchmark measured
   (> 1e-16), `r_acc` = max(tol, 10 τ_eff δ_max) as reported, and the slice still converges at
   that r_acc.
9. **Capacity full:** capacity of 2 blocks on test 1's slice: `converged` False,
   `capacity_full` True, the converged pairs returned are correct.
10. **Deflation W:** test 1 with half the slice's modes passed as W: the other half returned, none
    duplicated (the W columns are M-orthogonal to V to 1e-10).
11. **Column refill:** a start block with two identical columns (forced through `seed`/a test
    seam): the SVQB drop refills them; the slice still matches.
12. **Errors:** lo < 0, lo ≥ hi, a non-SPD-signature precond dtype (fp32 GMG), bad `block` or
    `capacity`, mismatched W / R shapes.
13. **Memory model:** `si_memory` storage equals the leaf bytes of the operator, preconditioner
    and basis; slow (macOS): the measured peak of a voxel slice with a large capacity within 20%
    of the estimate at ≥ 2 GB.
14. **Benchmark smoke:** `shift_invert_study` at a tiny size runs (marked slow), so the
    recorded figures stay reproducible.

## Docs

- `docs/eigensolvers.md`: a "Shift-invert slices" section (the operator, MINRES and the SPD
  preconditioner, τ, τ_eff and the floor, r_acc, the stop rule, capacity, the final
  Rayleigh–Ritz, `info` keys), the measured MINRES cost vs δ and depth, the floor and r_acc at
  1M, the slice memory model; the validation list for `tests/test_shift_invert.py`.
- README: status line (15a). The `solve_modes` example comes with 15c.
- ROADMAP: the PR 15 split note and the [15a]/[15b]/[15c] tags; the 15a status entry.
- `docs/open_items.md`: the split; σ moved by 1% once; τ_eff from the achieved residual; the
  measured cost driver (depth, not δ: 15c's cost model takes depth as its main variable);
  `max_steps` 100 default; the column-refill rule.

## Decisions without user input (to log)

- PR 15 split into 15a/15b/15c, all roadmap tests assigned.
- Ritz values from the shift-inverted projection ($\mu = \sigma + 1/\theta$), confirmed by true
  residuals and a final Rayleigh–Ritz on (K, M); no implicit restarts (the basis grows to the
  capacity; 15c splits a full slice).
- σ at the slice midpoint, moved once by 1% when the first solve stagnates.
- τ_eff = the largest residual a stagnated or capped MINRES column reached.
- Default capacity 100 steps (600 columns) without a memory model input.
