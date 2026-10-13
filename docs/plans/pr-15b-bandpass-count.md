# PR 15b plan: band-pass count and slice restarts

Branch `pr-15b-bandpass-count` from `main` (PR 15a merged). Roadmap: PR 15 ("Interior bands",
the [15b] items), the "Frequency bands" section (band edges, completeness count) and the split
in the PR 15a plan (15b = the band-pass count with neighbour deflation, and the slice restarts).

Revision 2 (plan review round 1): out-of-band pairs as an array (the caller's W, e.g. adjacent
slices' modes, plus the neighbours; the edges use the nearest), `band_edges` as a helper and
`precond` as in `count` (the slice's preconditioner by default), `si_checked`'s count results in
one `info["count"]` dict shaped like 14b's, restarts within the first run's memory, the
lower-edge degree formula, a segmented recurrence with progress lines, W streamed from several
host arrays, no `neighbours=` option (test 8 calls the private loop), test 4 cut to one case
plus a free body with R, concrete ranges and budgets in the tests.

Revision 3 (round 2): only the nearest out-of-band pair on each side is deflated (from
`si_slice`'s neighbours and the caller's W), so no large deflation set is streamed; progress
lines by a host callback inside the existing loop (no segmentation, no carry or memory change);
`verbose` in `solve_modes`'s `count_options`; the result of an unconverged restart; `gap` = the
governing gap; `n_top` in `info`; `lam_W` required with W.

Revision 4 (round 3): the progress callback every step (throttled on the host to one line per
10 s), `si_checked(verbose)` reaching the count, docs for `count`'s tuple W and `verbose`.

## Scope

- **In:** the band-pass count `band_count` (two counting edges, deflation of the known
  out-of-band neighbours, the degree rule with a lower edge and a lower tail), `band_edges`, and
  `si_checked`: one slice by `si_slice`, then its count, with at most 2 restarts while the count
  stays ≥ 0.5. Building blocks for 15c.
- **Out (15c):** slicing, the cost model and `verify` for bands (15c calls `si_slice` directly
  when the count is off, and predicts the count's cost from `band_edges` and `count_degree`
  before running it, as 14b's `_verify`), `solve_modes(..., band=(f1, f2))`, "each slice's
  count < 0.5" over a multi-slice band.

Roadmap [15b] tests (all here, none dropped):

| Roadmap test | Test below |
|---|---|
| [15b, 15c] interior band between two well-separated modes: 0 modes, count < 0.5; a mode planted in it and left out of V: count ≥ 0.5 | 7 (15b part; end to end through `band=` in 15c) |
| [15b] slice with one in-band mode removed after convergence: count ≥ 0.5, flag, mode recovered on restart; mode forcibly withheld: 2 restarts, count flagged | 9 |
| [15b] one in-band mode (not next to an edge) left out: detected for ≥ 97 of 100 seeds | 6 |
| [15b] V complete, τ_c = 1e-12: neighbour deflation off raises the estimate by the probe mean of $\sum_i q(\mu_i)^2 (v_i^\top M y_j)^2$ to 1e-8 relative; on: estimate < 0.5 | 8 |

## Facts

- PR 14a's `count` estimates $s = \frac1{n_p}\sum_j \|\Pi_V\, q(T)\, \Pi_V y_j\|_M^2$ with $T =
  (K + \lambda_c M)^{-1} M$, $x = 2\lambda_c T - I$, whose eigenvalue for mode λ is $x(\lambda) =
  (\lambda_c - \lambda)/(\lambda_c + \lambda) = \cos\theta$, $\theta = 2\arctan\sqrt{\lambda/\lambda_c}$
  (x decreases as λ grows). q is the Jackson-damped Chebyshev series of the indicator of
  $x \ge 0$ (λ ≤ λ_c). The probe loop (sampling, projection, recurrence, reduction, flags,
  timing) depends on the polynomial only through its coefficients γ.
- The indicator of $x \ge x_0$, $x_0 = \cos t$, has $c_0 = t/\pi$, $c_k = 2\sin(kt)/(k\pi)$
  (14a's series at $t = \pi/2$). The band $c_{lo} \le \lambda \le c_{hi} = \lambda_c$ is
  $x \in [0, x_0]$, $x_0 = (1 - a)/(1 + a)$, $a = c_{lo}/c_{hi}$, so its indicator is
  $1\{x \ge 0\} - 1\{x \ge x_0\}$ and, damping being linear, the Jackson band-pass is
  $\gamma = \gamma(0) - \gamma(x_0)$. The Jackson kernel is positive, so $0 \le q \le 1$.
- Resolution in θ: $d\theta/du = 1/(\sqrt u (1 + u))$, $u = \lambda/\lambda_c$: 1/2 at the upper
  edge, $\sqrt a/(1 + a)$ per unit relative gap at the lower edge. Measured (correctness and
  performance reviews, the rule below): $d\,g_{hi} \approx 22.8$ when the upper edge is tighter
  (the step's value, for every a), and $d\,g_{lo} \approx 23\,(1 + a)/(2\sqrt a)$ when the lower
  edge is tighter: 22.9, 24.2, 28.6, 39.8, 115 at a = 0.9, 0.5, 0.25, 0.1, 0.01. Pass/fail is
  monotone in d (bisection is safe). The roadmap places σ at the upper edge; kept (logged),
  with the formula in the docs so 15c's cost model sees the cost of wide slices (it calls
  `count_degree`, never 23/g).
- Out-of-band weight at d = 23/g (performance review): $q^2 \approx 6.7\text{e-}4$ at a
  neighbour g/2 past its counting edge, 3.9e-6 at g, 2.6e-7 at 1.5g. One deflated neighbour per
  edge typically removes everything above about 4e-6 (when the next mode is ≥ g past the
edge); a mode close behind the neighbour or a degenerate twin of a neighbour left in adds at most
  about 7e-4. On the 14a reference spectrum (bands at modes 11–16, 31–40, 61–66, 101–110, an
  empty band) the degree was always set by the $q^2$ rule, and the exact out-of-band sum with
  the neighbours deflated was ≤ 1e-3 (correctness review). So the roadmap's "converged
  out-of-band neighbours within the polynomial's transition width" is met by the nearest pair on
  each side; a further known pair left in adds about 4e-6 when ≥ g past the edge, at most about
  7e-4 when close behind the nearest pair (e.g. a split double mode), which the tail model and
  the 0.05 budget absorb, at the cost of streaming them through every projection.
- `si_slice` returns lam (ascending, the converged pairs in the slice widened over clusters), V
  (host, M-orthonormal), and `info["neighbours"] = {"lo": (λ, v) or None, "hi": ...}`: the
  nearest converged Ritz pair beyond each edge that is not in its W (none below for lo = 0).
  Pairs in W are deflated from the Krylov basis and never return; so when an adjacent slice's
  mode (in W) is the true nearest eigenvalue beyond an edge, `info["neighbours"]` holds the one
  after it.
- With exact eigenvectors, Π_V commutes with T, so $s_j = \sum_{i \notin V} q(\lambda_i)^2
  (v_i^\top M y_j)^2$; dropping a neighbour v from V adds exactly $q(\mu)^2 (v^\top M y_j)^2$
  (the cross term vanishes because $\Pi_V q(T) \Pi_V y_j \perp_M v$): test 8's identity.

## Design

### 1. Polynomial and degree (`eigen/counting.py`)

- `jackson_step(d, x0=0.0)`: the Jackson-damped indicator of $x \ge x_0$ (one function; callers
  unchanged).
- `count_degree(gap, lam_top, n_top, n_free, lower=None)`: `lower = (lam_in, lam_bot)` in units
  of λ_c (the lowest in-band λ: the lowest returned mode, or the band edge when none; and the
  nearest out-of-band λ below) switches to the band-pass $\gamma(0) - \gamma(x_0)$, $c_{lo} =
  (\lambda_{in} + \lambda_{bot})/2$. The rule (roadmap): the smallest d with $q^2 \ge 0.95$ at
  both in-band ends (θ of `lam_in` and of $1 - g/2$) and on the DCT grid between them, and a
  tail bias ≤ 0.05; doubling then bisection as now. Early exit and the degree reported when
  capped: $\lceil D_{rule} / g_{eff} \rceil$ with $g_{eff} = \min(g, g_{lo}\cdot 2\sqrt a/(1 + a))$.
- Tail bias with the roadmap's Weyl density $N(\lambda) \propto \lambda^{3/2}$: `tail_bias` (the
  upper tail, `lam_top` to $\lambda_{max}$, $N(\lambda_{max}) = n_{free}$; signature unchanged)
  is built on `_bias(gamma, l0, l1, n_ref, lam_ref)` $= \int_{l_0}^{l_1} q^2\, dN$ with $N =
  n_{ref}(\lambda/\lambda_{ref})^a$; the lower tail is `_bias(gamma, 0, lam_bot, n_top,
  lam_top)` (rigid modes sit at 0 and are deflated; N(0) = 0). The bias is the sum.
- **Normalization for bands.** A slice does not know its mode index, so `n_top` cannot be the
  count up to `lam_top` as in 14a. It is fitted to the contiguous core: the in-band modes and the
  nearest out-of-band pair on each side, $n$ eigenvalues from $\lambda_{bot}$ to $\lambda_{top}$:
  $C = (n - 1)/(\lambda_{top}^a - \lambda_{bot}^a)$, `n_top` $= C\lambda_{top}^a$. Further
  deflated pairs are not used in the fit (they need not be contiguous); deflating them only
  lowers the estimate. Underestimating the density only raises false alarms (the bias adds to
  the estimate; it never hides a missing mode); measured above, the bias is ≈ 0 on real spectra.
  An empty band fits C to the one interval between its neighbours (logged).

### 2. `band_edges` and `band_count` (`eigen/counting.py`, exported from `jaxmodal.eigen`)

```python
c_lo, c_hi, g_lo, g_hi = band_edges(lam, lo, hi, lam_out)
est, info = band_count(op, free, V, lam, lo, hi, V_out, lam_out, *, R=None, precond=None,
                       n_probes=50, seed=0, tau=None, degree=None, block=None, maxit=1000,
                       verbose=False)
```

- V (n, k ≥ 0), lam: the in-band modes (as `si_slice` returns them; the out-of-band pairs are
  added by `band_count`, not passed in V); lo < hi: the band edges (eigenvalues, lo > 0; a band
  from 0 uses `count`). V_out, lam_out: the known converged out-of-band pairs nearest the band (`si_slice`'s
  neighbours, the nearest pair of the caller's W beyond each edge), a host array (n, j) and
  its eigenvalues (j,); 2 columns from `si_checked`, any j accepted. None may lie in
  $[\lambda_{in,lo}, \lambda_{in,hi}]$, and at least one must lie on each side (otherwise the
  edge cannot be placed: raises). All of them are deflated.
- `band_edges` (roadmap "Band edges"): $\lambda_{in,lo}$ = min(lam) (lo when lam is empty),
  $\lambda_{in,hi}$ = max(lam) (hi when empty); $\lambda_{nb,lo}$, $\lambda_{nb,hi}$ = the
  nearest lam_out below and above; $c_{lo} = (\lambda_{nb,lo} + \lambda_{in,lo})/2$, $g_{lo} =
  (\lambda_{in,lo} - \lambda_{nb,lo})/c_{lo}$, the same at the top; σ = λ_c = $c_{hi}$. So
  $q^2 \ge 0.95$ holds on $[\lambda_{in,lo}, \lambda_{in,hi}]$, the in-band interval up to the
  returned modes (up to the band edges when none). 15c uses it with `count_degree` to predict
  the cost before building anything.
- W = (V, V_out): `_chunks`, `_project` and `_check_ortho` take a tuple of host arrays and
  stream across it (a few lines), so V is never copied to append V_out; `count` accepts a tuple
  W too (the total columns checked against len(lam)). The R term as in `count`.
- `precond`: the SPD preconditioner of $K + c_{hi} M$ (default `jacobi(op, free, c_hi)`), as
  `count`'s `precond` at λ_c.
- τ_c, blocks, flags, the degree cap (unverifiable, not run), NaN on a failed solve: as
  `count`. `info` as `count`'s plus `edges` ($c_{lo}$, $c_{hi}$) and `gaps` ($g_{lo}$, $g_{hi}$)
  (`lam_c` = $c_{hi}$; `gap` = the governing gap $g_{eff}$, the roadmap's "smaller of the edge
  gaps" in θ-resolution terms), and `n_top` (the fitted model count at `lam_top`).
- **Refactor.** `count`'s probe loop moves to a private `_estimate(op, m, Ws, R, MR, precond,
  gamma, d, lam_c, tau, n_probes, block, seed, maxit, verbose, _steps)` returning the estimate
  and the shared info; `count` and `band_count` validate, place the edges, pick d and call it.
  No behaviour change for `count`.
- **Progress.** With `verbose` (new on `count` too; a static flag of `_recurrence`), the loop
  body calls `jax.debug.callback` every step with scalars only (k, iteration sum, iteration
  max; d and t0 bound on the host by `functools.partial`); the host side prints steps done of d, PCG iterations and
  elapsed time when at least `PRINT_S = 10` s passed since its last line, and each block prints
  one line. The carry and the loop are unchanged, so the step-phase memory (`count_memory`) and
  the results are unchanged. At 1M DOF a step takes about 65–85 s (50 probes), so this prints
  one line per step there; on test meshes the throttle keeps it quiet. A verbose count compiles
  its own loop variant once (seconds; noted in the docs). `verbose` joins `solve_modes`'s
  `COUNT_OPTIONS` (rule: every option reachable; 14b band counts get progress too).
- Memory: `count_memory` / `probe_block` apply unchanged (V_out adds 2 host columns from `si_checked`).

### 3. `si_checked` (`eigen/shift_invert.py`, internal like `si_slice`)

```python
lam, V, info = si_checked(op, free, lo, hi, precond, *, capacity, W=None, lam_W=None,
                          count_precond=None, count_options=None, seed=0, verbose=False,
                          **slice_options)
```

- W (host, M-orthonormal, the caller's known modes, e.g. adjacent slices') needs `lam_W` of
  matching length (raises otherwise). Run `si_slice` (W, seed, verbose and every other option
  pass through). If it converged, count with V = all in-band modes found and V_out = the nearest
  out-of-band pair on each side among `si_slice`'s neighbours and W (when an adjacent slice's
  mode is nearer than the neighbour, it places that edge): `band_count(op, free, V, lam, lo,
  hi, V_out, lam_out)` for lo > 0; for lo = 0, `count(op, free, (V, v_up), [lam, λ_up],
  lam_last, λ_up)` with lam_last = `lam.max() if len(lam) else hi` and v_up the nearest pair
  above (the step polynomial; roadmap: an edge at λ = 0 has no counting edge; the deflation set
  is then contiguous from 0, as `count`'s tail model needs). `count_precond` defaults to the
  slice's `precond` (the SPD fp64 multigrid of $K + \sigma M$, σ the midpoint, a good
  preconditioner of $K + c_{hi} M$: no second build, no second preconditioner alive); 15c may
  pass one built at $c_{hi}$. `count_options` (n_probes, seed, tau, degree, block, maxit) pass
  through.
- **Restart rule (roadmap).** A verifiable count ≥ 0.5 flags the slice and restarts it:
  `si_slice` with a fresh random block (seed + j for restart j) and W extended by the in-band
  modes found so far (lam_W too; this extended W is only for `si_slice`, the count keeps
  V = the found in-band modes and V_out as above). Only in-band modes are added: the neighbours must reappear as
  Ritz values to anchor the stop rule (deflated, the rule would wait for the next modes beyond
  each edge). The restart's capacity is `capacity` − L0_extra, L0_extra the extra deflation
  columns rounded up to the chunk (`si_slice`'s Q is (n, L0 + capacity)), so its device peak
  stays within the first run's; if that leaves < 2 block columns, the slice is returned flagged
  without the restart (`reason`). New in-band pairs are merged with the found ones and re-sorted
  ascending (residuals and thresholds permuted alongside); they are M-orthogonal to the found
  ones by construction. The neighbours are the restart's. The count is repeated with fresh
  probes (count seed + j: a false alarm from one probe draw does not repeat with V unchanged).
  After 2 restarts with the count still ≥ 0.5 the slice is returned with the count flagged.
  An unverifiable count (degree cap, solve floor) and a slice that did not converge are not
  restarted: they are returned flagged (`passed` False). A restart that does not converge
  ends the loop: the found modes merged with its converged in-band pairs are returned,
  `converged` False, `count["reason"]` "slice not converged", `flagged` True, `restarts` = j,
  and the count fields of the last count that ran.
- `info`: the last run's `si_slice` info (r_acc and tau_eff the max over runs; residuals and
  thresholds of all returned pairs; `converged` of the last run) plus `runs` (each run's info)
  and `count`, one dict shaped like 14b's `info["count"]`: `count`'s / `band_count`'s info of
  the last count plus `ran`, `reason` (None, "slice not converged", "restart capacity", or the
  count's degree-cap reason), `estimate` (the last), `estimates` (all), `flagged` (some
  verifiable count ≥ 0.5: the roadmap's flag) and `restarts`. A failed check is `flagged and not
  passed`. 15c adds the predicted-cost keys.
- `verbose`: also sets the count's `verbose` unless `count_options` gives it. `si_slice`'s lines, the count's progress lines, and one line per count (estimate,
  degree, steps, time, action: pass, restart j, flagged).
- Memory: `si_slice` releases its basis before returning, the count reuses the slice's
  preconditioner, so the peak is max(slice, count) with one preconditioner; restarts stay
  within the first run's device peak (above).

## Tests

In `tests/test_counting.py` (band count; its reference `ref()` mesh: hex20 6×5×4 clamped,
dense eigenpairs, `exact_precond`) and `tests/test_shift_invert.py` (`si_checked`). Fast-suite
budget: the new tests ≤ 60 s together outside test 6's complete case (slow, as 14a).

1. `jackson_step(d, 0)` unchanged (equals the stored 14a coefficients); the undamped
   coefficients of $x \ge x_0$ against scipy quad of the indicator's Chebyshev projection; the
   band-pass grid in [−ε, 1 + ε].
2. Band degree rule: for a = 0.25, 0.5, 0.9 with each edge the tighter one: $q^2 \ge 0.95$ at
   both in-band ends for d, and d − 1 fails the rule; upper edge tighter:
   $d\,g_{hi} \in [19, 24]$; lower edge tighter: $d\,g_{lo}\cdot 2\sqrt a/(1 + a) \in [19, 24]$.
   A synthetic dense out-of-band spectrum (large n_top, as 14a's test) makes the bias set d.
   `band_edges` on a few cases (empty band, straddling cluster, W nearer than the neighbour);
   `info["n_top"]` equals $C\lambda_{top}^a$ of the contiguous core for a known case.
3. Lower tail against scipy quad of $q^2\, dN/d\lambda$ over $(0, \lambda_{bot})$, 1e-3
   relative (as 14a's upper-tail test).
4. Expected value (as 14a test 4): the mean of 2000 probes against the exact sum
   $\sum_{i \notin W \cup R} q(\lambda_i)^2$ within 5 standard errors: one interior mode missing
   (clamped, consistent mass), and a free hex20 box with R (band above the rigid modes).
5. Against the dense estimator (as 14a test 5): $s_j$ against the dense band-pass recurrence
   with exact solves on the same probes to 1e-8 relative (missing) / 1e-10 absolute (complete),
   τ_c = 1e-12 (the 14a helper takes γ instead of d).
6. **[15b] Detection:** a band holding ≥ 4 modes, one interior mode (not next to an edge) left
   out: one 5000-probe count as 100 sets of 50 (14a's scheme, deterministic), ≥ 97 sets ≥ 0.5;
   V complete: every set < 0.5 (slow, as 14a).
7. **[15b] Empty band:** the largest relative gap $(\lambda_m, \lambda_{m+1})$ among the first
   40 modes; the band is its middle third. `si_checked` returns 0 modes and a count < 0.5.
   Planted: the same band on the model with density × s, $s = \lambda_{m+1}/\lambda_\text{centre}$
   (every eigenvalue scales by 1/s, so mode m + 1 lands at the band centre); assert that the
   scaled modes m and m + 2 lie outside the band; `band_count` with V empty and those two dense
   neighbours gives ≥ 0.5.
8. **[15b] Neighbour deflation:** V complete (dense modes, τ_c = 1e-12, exact preconditioner):
   `band_count`'s estimate is < 0.5; `_estimate` with the same γ, d, λ_c, seed and W without the
   two neighbours rises by the probe mean of $\sum_{i \in nb} q(\mu_i)^2 (v_i^\top M y_j)^2$ to
   1e-8 relative ($y_j$ from the same keys, dense solves).
9. **[15b] Restarts** (`monkeypatch` of `shift_invert.si_slice`, hex20 6×5×4 clamped, a slice
   of ≥ 4 modes with edge gaps ≥ 0.1, exact dense preconditioner, so d ≲ 250): (a) the first
   run's result with one in-band mode removed: count ≥ 0.5, `flagged`, one restart recovers it,
   final count < 0.5, λ against the dense reference (15a tolerances) with multiplicities, V
   M-orthonormal; (b) the mode withheld on every run: 3 `si_slice` calls, `restarts == 2`,
   every estimate ≥ 0.5, `flagged` and not `passed`; the restart's W holds the found modes, its
   capacity is reduced by the extra deflation columns, and the seeds differ.
10. `si_checked` with lo = 0 (clamped) uses the step count: < 0.5, `count["lam_c"]` midway
    between the last mode and the neighbour. W nearer than the neighbour: the adjacent mode
    beyond `hi` passed in W (dense), V complete: count < 0.5, the upper edge placed at that
    mode.
11. Not restarted: an unverifiable count (monkeypatched `band_count` returning NaN,
    unverifiable), an unconverged slice (capacity full) and a restart without capacity return
    flagged or unrun with `restarts == 0` and the reason; verbose line counts (`capsys`) for
    `si_checked` and for `count` (`PRINT_S` monkeypatched to 0; `count_options={"verbose":
    True}` through `solve_modes` with `band=(0, f_max)`), and the same estimate and probes with
    and without `verbose` to 1e-12 relative (different compiled programs). A restart that does not converge (monkeypatched capacity full on the
    restart): merged modes, `converged` False, the reason, `flagged`, `restarts == 1`. W
    without `lam_W` or with a mismatched length raises.
12. `band_count` errors: no out-of-band pair on a side, lo = 0, an out-of-band pair inside the
    in-band interval, bad shapes; info fields (`edges`, `gaps`, `gap`, `degree`); `count` with a
    tuple W gives the same estimate as with its concatenation.

Run: these tests after each commit, then the full `pytest -m "not gpu"` (slow included).

## Benchmark

None: no [15b] criterion measures cost, and the count's cost per step is 14a's (the band-pass
changes only γ; PCG on $K + cM$ with the multigrid is index-independent, 12–14 iterations
measured to index 199). A host-only table of the band degree against $(g_{lo}, g_{hi}, a)$,
including tight lower edges, goes into the docs for 15c's cost model (seconds, no solves).

## Docs

- `docs/eigensolvers.md`, "Completeness count": `band_count` and `band_edges` next to `count`,
  the band-pass polynomial, its edges, the degree rule with two edges and the lower-edge
  formula, the lower tail and the band normalization, the out-of-band deflation and its
  justification, progress lines, the degree table; "Shift-invert slices": `si_checked`, the
  restart rule, its `info["count"]` layout; "Validation": the new tests in both files.
- `docs/eigensolvers.md`, `count`'s description: W may be a tuple of arrays; `verbose`.
- `README.md`: status line; a `band_count` line next to `count`.
- `docs/eigensolvers.md` options table: `count_options` gains `verbose`.
- `docs/ROADMAP.md`: PR 15b status.
- `docs/open_items.md`: the decisions below.

## Decisions without user input (to log)

- The band tail model is fitted to the contiguous core (no mode index in a slice).
- The nearest out-of-band pair per side (from `si_slice` or the caller's W) meets "the
  converged neighbours within the transition width" (numbers above).
- `verbose` progress by `jax.debug.callback` inside the loop, throttled to one line per 10 s;
  `verbose` added to `COUNT_OPTIONS`.
- `band_count` needs lo > 0; bands from 0 use `count`. The lower tail has N(0) = 0 with the
  rigid modes deflated.
- σ of the band count at the upper counting edge (roadmap); the lower-edge cost
  $23(1 + a)/(2\sqrt a)/g_{lo}$ grows for wide slices (115/g at a = 0.01), which 15c's cost
  model sees through `count_degree`. Centring the map would cut it (logged as an option).
- The count reuses the slice's preconditioner by default.
- Restarts deflate the in-band modes only, within the first run's memory; recounts use fresh
  probes (seed + j for slice and count).
- Unverifiable counts and unconverged slices are not restarted.
- Test 7 plants the mode by scaling the density.
