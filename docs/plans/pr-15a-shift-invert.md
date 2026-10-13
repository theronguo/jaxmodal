# PR 15a plan: block shift-invert Lanczos on one slice

Branch `pr-15a-shift-invert` from `main` (PR 14b merged). Roadmap: PR 15 ("Interior bands"),
the "Frequency bands" section (interior bands, band edges, completeness count) and the
"Preconditioners" rule that fp32 V-cycles never run inside MINRES.

Revision 2 (plan review round 1): Ritz pairs by Rayleigh–Ritz on (K, M) (the inexact projected
T cannot reach r_acc), the roadmap's per-pair thresholds with r_acc over pairs and neighbours,
τ_eff defined and a failed-solve stop, W kept out of the projection, `capacity` as the only
(required) limit, the stop rule as a tested helper with unreached-edge reporting, chunked Ritz
residuals, two memory test regimes, the benchmark to depth 1000 with SciPy references and a real
1M slice, roadmap tolerances on every accuracy test, a τ_eff seam test.

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
- Block shift-invert Lanczos prototype (b = 6, full M-reorthogonalization, Rayleigh–Ritz on
  (K, M) over the basis, true residuals): hex20 8×6×5 clamped, slice holding modes 31–45 at tol 1e-10:
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
lam, V, info = si_slice(op, free, lo, hi, precond, *, capacity, sigma=None, R=None, W=None,
                        tol=1e-6, tau=1e-8, block=6, maxit_inner=20000, seed=0)
```

Internal: imported from `jaxmodal.eigen.shift_invert` by the tests and PR 15b/15c, not exported
from `jaxmodal.eigen`; `solve_modes` exposes slices (and every option, PR 13c rule) in 15c.
`lo`, `hi` are eigenvalues ($\lambda = (2\pi f)^2$), `0 <= lo < hi`. `precond` is the SPD
preconditioner of $K + \sigma M$ (fp64 GMG/AMG or Jacobi built at +σ by the caller); a GMG with
fp32 levels raises (fp32 asymmetry floors MINRES near 1e-6). `R`: the M-orthonormal rigid modes.
`W`: known M-orthonormal modes to deflate (restarts in 15b, neighbouring slices in 15c), a host
array; its M-orthonormality is checked once by a sketch (`counting._check_ortho`). `capacity`
(required, Lanczos columns, rounded up to a multiple of the projector chunk c) is the one
resource limit: there is no default allocation and no step cap besides it; 15c sets it from
`si_memory` and the budget, tests pass it.

- **Shift.** σ defaults to the midpoint $(\lambda_{lo} + \lambda_{hi})/2$; an explicit `sigma`
  follows the same rules. If the first block's MINRES stagnates with a residual above $10^3\tau$
  (σ on an eigenvalue: measured residuals 0.1–3e8 there), σ moves by 1% of the slice width
  toward $\lambda_{hi}$ and the start repeats once (`info["sigma_moved"]`, `info["sigma"]` the
  value used).
- **Operator.** $T = (K - \sigma M)^{-1} M$ on the free DOFs: one step solves
  `minres(ap_operator(op, free, -sigma), m * M(Q_j), precond, tau, maxit)` for the newest block
  $Q_j$ (n × b; σ is a traced leaf of `APOperator`, so one compiled MINRES serves every σ), masks
  the result and deflates it. The constrained diagonal is `ap_operator`'s default at −σ: mean
  diag K + σ mean diag M over the free DOFs (the roadmap's rule). Inner cap: `maxit` =
  `maxit_inner` for the first step, then min(`maxit_inner`, 10 × the first step's iterations)
  (the depth of σ, which sets the iterations, does not change within a slice); a capped column
  counts as stagnated for τ_eff and sets `info["inner_maxit"]`.
- **Basis.** A preallocated fp64 device array `Q` of `len(W) + capacity` columns holds
  $[W \mid Q_1 \mid Q_2 \mid \dots]$; the count of filled columns is a traced int (one compiled
  step, as LOBPCG's locked basis); the start block is deflated against R and W.
- **Step (jitted).** After MINRES: two passes of `lobpcg._deflate` against R and all filled
  columns (W and the Lanczos blocks), then M-orthonormalization by SVQB (`lobpcg._svqb_t`). A
  column dropped by SVQB (rank loss) is replaced by a fresh random masked vector, deflated and
  orthonormalized, so the block keeps width b (the roadmap's "same column masking as LOBPCG").
  Then one K apply on the new block and $G_{:, new} = Q_L^\top K Q_{new}$ over the Lanczos
  columns $Q_L$ only (W stays out of the projection, so its modes never return as Ritz pairs):
  $G$ (host, symmetric) is the projected stiffness on the M-orthonormal Lanczos basis.
- **Ritz pairs: Rayleigh–Ritz on (K, M).** $\mu$, $Y$ = `eigh(sym(G))` (Q is M-orthonormal, so
  this is the Rayleigh–Ritz of the pencil on span(Q_L)). The inexact solves only shape the
  subspace; Ritz values and vectors come from exact K products, so their accuracy is not limited
  by τ (correctness review: θ of the inexact projected T gave residuals above $10\tau\delta$ that
  grew with the basis; Rayleigh–Ritz on (K, M) over the same basis gave 5e-10 at τ = 1e-8).
  Candidates: the Ritz values in $[\lambda_{lo}(1 - 100 r), \lambda_{hi}(1 + 100 r)]$ plus the
  nearest one beyond each edge (r = the running $r_{acc}$). Their residuals are computed in
  chunks of c candidate columns at one compiled width (zero-padded, masked): $X_c = Q_L Y_c$, one
  K and one M apply, `lobpcg._resid`, discarded. Spurious interior Ritz values of the Rayleigh–
  Ritz carry large residuals and simply stay unconverged until the subspace resolves them.
- **Thresholds and r_acc (the roadmap's per-pair rule).** Pair i is accepted at $r_i \le t_i =
  \max(tol,\ 10\,\tau_\text{eff}\,\delta_i,\ \text{its \_resid floor})$ with $\delta_i = |\mu_i -
  \sigma|/|\mu_i|$. $r_{acc} = \max t_i$ over the in-slice pairs and the neighbours used by the
  stop rule (`info["r_acc"]`; 15b/15c take the band-wide max). $\tau_\text{eff} = \max(\tau,$ the
  running max over all steps of `minres`'s final relative residual for columns that stagnated or
  hit the cap$)$, `info["tau_raised"]`. If τ_eff exceeds $10^{-4}$ (an inner solve that failed,
  e.g. σ on an eigenvalue after the move), the slice stops with `converged` False and
  `info["reason"]` = "inner solves failed".
- **Stop rule: `_slice_stop(mu, r, t, lo, hi, r_acc)`**, a pure host helper (unit-tested on
  synthetic arrays, as 14b's `_band_count`) returning `(done, inside, below, above)` index sets:
  `inside` = the Ritz values in $[\lambda_{lo}, \lambda_{hi}]$ extended over clusters on both
  edges (relative gaps < $100\,r_{acc}$); `below`/`above` = the nearest converged Ritz value
  beyond each edge past a gap ≥ $100\,r_{acc}$ from `inside` (`below` empty and not needed when
  $\lambda_{lo} = 0$); `done` = every `inside` pair converged and both neighbours found. An empty
  slice needs only the neighbours. 15c's edge snapping passes windows through this helper without
  touching the step loop.
- **Capacity full.** When the next block does not fit before `done`, the slice returns its
  converged `inside` pairs with `converged` False, `info["capacity_full"]` True and
  `info["edge_unreached"]` = "lo", "hi", "both" or None (no converged Ritz value ever appeared
  beyond that edge: λ_lo below λ₁ or λ_hi above the spectrum, which 15c handles by its [0, f₂]
  rule or the above-spectrum error instead of splitting).
- **Final Rayleigh–Ritz.** The compact X of the converged `inside` pairs and neighbours is
  formed in chunks, Q is released, then $X^\top K X$ (chunked), a small `eigh`, the rotated
  vectors, explicit K and M products and the residuals and thresholds recomputed (the roadmap's
  "final Rayleigh–Ritz with true residuals on (K, M)"). A pair above its $t_i$ there sets
  `converged` False.
- **Returns** `lam` (ascending, `inside`), `V` (host fp64, n × k, compact) and `info`:
  `converged`, `reason`, `sigma`, `sigma_moved`, `steps`, `basis` (Lanczos columns used),
  `minres_iterations` (per step, max over columns), `tau_eff`, `tau_raised`, `inner_maxit`,
  `r_acc`, `delta_max`, `residuals`, `thresholds` (the $t_i$), `neighbours` =
  `{"lo": (lam, V) or None, "hi": (lam, V) or None}` (None at a λ = 0 edge or when unreached;
  15b deflates them, 15c uses them at slice edges), `capacity_full`, `edge_unreached`, `time`
  (setup, minres, ortho, ritz).

### 2. Benchmark (`benchmarks/shift_invert_study.py` → `results/shift_invert_study.json`)

References from SciPy `reference_modes` / `eigsh(sigma=...)` on the assembled free blocks (LU at
≤ 110k DOF is cheap; `count_below` or the Sylvester count gives the depth), not long
`solve_modes` runs.

- **MINRES iterations vs depth and δ:** hex20 $16^3$ clamped (56k DOF, AMG) and voxel $32^3$
  clamped (108k DOF, GMG fp64), 6 random columns $b = Mq$, τ = 1e-8; σ in the gap after modes
  5, 20, 50, 100, 200 (both meshes) and 500, 1000 (hex20 $16^3$), at fractions 0.5, 1e-2, 1e-4
  of the gap (so δ spans four decades at each depth). Recorded per solve: iterations (max over
  columns), δ, depth (eigenvalues below σ), seconds and seconds per iteration. 15c fits its cost
  model (iterations and seconds against depth, with δ as a secondary term) on this.
- **Floor:** τ = 1e-14 at one fraction (0.5) per depth: the smallest explicit relative residual
  reached (the fp64 V-cycle MINRES true-residual floor), iterations and seconds.
- **Largest Mac sizes:** hex20 1M DOF with AMG (the PR 13b acceptance mesh) and voxel 2M DOF
  with fp64 GMG (the PR 11b acceptance mesh): the floor near mode 100 (one 6-column τ = 1e-14
  solve each, σ from the same geometry's $12^3$ / $32^3$ eigenvalues scaled, depth confirmed by
  the slice), and one real short slice on the hex20 1M mesh holding about 3 modes near mode 100
  (edges from the coarse mesh's eigenvalues; the stop rule extends clusters), recording its
  `r_acc`, `tau_eff`, `delta_max`, steps and seconds.
- **A full slice vs LOBPCG from 0:** hex20 $12^3$, modes 51–70 at tol 1e-8: steps, Lanczos
  columns per mode, MINRES iterations per step, seconds; the same band by `solve_modes(band=...)`
  (a first data point for 15c's crossover).
- **Run time:** about 1.5–2 h in total (grid about 45 min, 1M and 2M floors about 50 min, the 1M
  slice about 30 min); the JSON records each part's wall time.

### 3. Slice memory model (`memory.si_memory`)

```python
si_memory(op_memory, n_dof, capacity, block=6, precond_memory=None, n_rigid=0, n_deflate=0)
```

storage = operator + preconditioner storage (fp64) + basis $8\,n\,(n_\text{deflate} +
\text{capacity})$ + R and MR; phases (each at a fixed width, so constants): MINRES (its state
x, r1, r2, y, w, w2, b, the A apply and the explicit check: a per-column constant fitted from
`memory_analysis` of the jitted MINRES loop on the stencil and unstructured operators, envelope
over b = 6 and 12, plus the operator and V-cycle workspace at width b), reortho (the chunked
projector at width b, as `lobpcg_memory`'s projector term), Ritz (one chunk of c candidates:
X, KX, MX, plus the operator workspace at width c); the final Rayleigh–Ritz holds the compact X
after Q is released (at most the basis); total = overhead + storage + max(phases). The phase
split is returned. 15c composes it into `solve_modes_memory`.

## Tests (`tests/test_shift_invert.py`)

References: SciPy `eigsh(K_f, k, M_f, sigma=σ)` on the assembled free blocks and dense `eigh` on
meshes ≤ 5k free DOF (cached per mesh with `functools.cache`), the dense `sytrf` inertia
(`count_below`) for slice counts. Every slice accuracy test runs at the roadmap's test
tolerances, tol = 1e-10 and τ = 1e-8 (r_acc = max(1e-10, 10 τ_eff δ)), and checks each pair
against the solver's own rule $r_i \le t_i$ (with $t_i \le r_{acc}$). λ̂_max = `norm_K / norm_M`
from `lobpcg._norms`.

1. **One interior slice vs `eigsh`** (hex20 8×6×5 clamped, AMG; voxel 10³ clamped, GMG):
   slice edges at the midpoints of gaps ≥ 1e-2 around modes 31–45: `converged`, not
   `capacity_full`, fewer than half the capacity's steps; the count equals
   `count_below(hi) - count_below(lo)`; λ within max(1e-8, 10 ε λ̂_max/λ) of `eigsh` for clusters
   with relative gap ≥ 1e-2, within 10 r_acc otherwise, multiplicities equal; V M-orthonormal to
   1e-10; $r_i \le t_i$; one converged neighbour beyond each edge.
2. **λ = 0 edge** (the roadmap's clamped (0, f_max) on shift-invert, at slice level): clamped
   hex20, `lo = 0`, `hi` in a gap after mode 12: `converged`, not `capacity_full`, few steps,
   `neighbours["lo"]` None, the 12 modes vs `eigsh`.
3. **Free body:** free hex8 box with R deflated, a slice after the rigid modes: no rigid mode
   returned ($\max|V^\top M R| \le 10^{-10}$), the elastic modes vs dense `eigh`.
4. **Repeated eigenvalues:** the free voxel cube 8³ with R deflated (PR 9b's multiplicity
   pattern 2, 3, 3, 2, 1, 3, 3 from dense `eigh`, clusters at relative gaps < 1e-8): one slice
   holding a double cluster, one holding a triple cluster (named from the dense spectrum), each
   returned with its exact multiplicity.
5. **Empty slice:** a slice strictly inside a gap: 0 modes, `converged`, not `capacity_full`,
   both neighbours.
6. **Edge cluster:** a slice edge between the two copies of a double eigenvalue: both copies
   returned (cluster extension).
7. **σ on an eigenvalue:** `sigma` passed exactly at a reference eigenvalue: `sigma_moved` True,
   `info["sigma"]` ≠ the passed σ, and the slice matches `eigsh`. With the move patched to 0: the
   slice stops with `converged` False and reason "inner solves failed" (no modes at a huge
   r_acc).
8. **τ_eff from the solves (spy on `minres`):** τ = 1e-16: `tau_raised`, τ_eff > 1e-16 and equal
   to the running max of the captured stagnated columns' final residuals, `r_acc` = max $t_i$ as
   reported, the slice converges.
9. **τ_eff carried into r_acc (seam):** `minres` wrapped to stop at relative residual 1e-6 and
   report `stagnated` (as 15c's τ_eff = 1e-6 case): τ_eff = 1e-6 (to the captured value),
   `tau_raised`, $r_{acc} = 10\,\tau_\text{eff}\,\delta_{max}$ (above tol), every $t_i$ =
   max(tol, 10 τ_eff δ_i, floor), at least one returned residual above tol, all ≤ their $t_i$, λ
   within 10 r_acc of `eigsh`.
10. **Stop rule (`_slice_stop`, synthetic arrays):** a neighbour within 100 r_acc of the extended
    set does not count; a cluster chain across each edge extends `inside`; an unconverged
    neighbour does not count; λ_lo = 0 needs no lower neighbour; an empty slice; an unconverged
    inside pair blocks `done`.
11. **Capacity full and unreached edges:** capacity of 2 blocks on test 1's slice: `converged`
    False, `capacity_full`, `edge_unreached` None or the edge lacking a neighbour, and the
    converged pairs returned match `eigsh`; on the tiny mesh, `hi` above the spectrum with a
    small capacity: `edge_unreached` = "hi".
12. **Deflation W:** test 1's slice (on a short 10-mode slice) with half its modes as W: the other
    half returned, W never returned ($\max|W^\top M V| \le 10^{-10}$), returned ∪ W = the `eigsh`
    set.
13. **Column refill:** a start block with two identical columns (a test seam): the SVQB drop
    refills them; the slice still matches.
14. **Errors:** lo < 0, lo ≥ hi, a precond with fp32 levels, bad `block` or `capacity`,
    mismatched W / R shapes, a non-orthonormal W.
15. **Memory model:** `si_memory` storage equals the leaf bytes of the operator, preconditioner
    and basis. Slow (macOS), two cases with estimates ≥ 2 GB, each within 20% of the measured
    peak, the phase split printed: (a) MINRES-dominated: voxel about 1.5M DOF, fp64 GMG,
    capacity 4 blocks, σ near mode 5 (short solves); (b) basis-dominated: a large capacity, a
    few steps.
16. **Benchmark smoke:** `shift_invert_study` at a tiny size runs (marked slow).

Tests 9, 11–13 run on short slices (about 10 modes); if the fast suite exceeds about 60 s, the
modes 31–45 slices are marked slow.

## Docs

- `docs/eigensolvers.md`: a "Shift-invert slices" section (the operator and the constrained
  diagonal, MINRES with the SPD preconditioner, τ, τ_eff and the inner cap, the per-pair
  thresholds and r_acc, Rayleigh–Ritz on (K, M), the stop rule, capacity and unreached edges,
  the final Rayleigh–Ritz, `info` keys), the measured MINRES cost vs depth and δ, the floor and
  the 1M slice's r_acc (hex20 AMG and voxel GMG), the slice memory model; the validation list
  for `tests/test_shift_invert.py`.
- README: status line (15a). The `solve_modes` example comes with 15c.
- ROADMAP: the PR 15 split note, the [15a]/[15b]/[15c] tags, Dependencies rows ("15a
  Shift-invert slice | 10, 13", "15b Band-pass count | 14a, 15a", "15c Slicing, bands, strategy
  | 14b, 15a, 15b"), a note on the two Frequency-bands bullets (PR 15a measured: the depth of σ,
  not δ, drives the MINRES iterations; 15c fits its cost model on depth), the 15a status entry.
- `docs/open_items.md`: the decisions below.

## Decisions without user input (to log)

- PR 15 split into 15a/15b/15c, all roadmap tests assigned.
- Ritz pairs by Rayleigh–Ritz on (K, M) over the shift-invert Krylov basis (not the inexact
  projected T), confirmed by a final Rayleigh–Ritz; no implicit restarts (15c splits a full
  slice).
- σ at the slice midpoint (or given), moved once by 1% when the first solve stagnates; τ_eff >
  1e-4 ends the slice as failed.
- τ_eff = the running max of stagnated or capped columns' final residuals; later steps capped at
  10× the first step's iterations.
- `capacity` required (no default allocation); 15c sets it from `si_memory`.
- The measured cost driver: depth, not δ (roadmap note; 15c's cost model on depth).
