# PR 15b plan: band-pass count and slice restarts

Branch `pr-15b-bandpass-count` from `main` (PR 15a merged). Roadmap: PR 15 ("Interior bands",
the [15b] items), the "Frequency bands" section (band edges, completeness count) and the split
in the PR 15a plan (15b = the band-pass count with neighbour deflation, and the slice restarts).

## Scope

- **In:** the band-pass count `band_count` (two counting edges, neighbour deflation, the degree
  rule with a lower edge and a lower tail), and `si_checked`: one slice by `si_slice`, then its
  count, with at most 2 restarts while the count stays ≥ 0.5. Both are building blocks for 15c.
- **Out (15c):** slicing, the cost model, `solve_modes(..., band=(f1, f2))`, `verify` for bands,
  the strategy choice, "each slice's count < 0.5" over a multi-slice band.

Roadmap [15b] tests (all here, none dropped):

| Roadmap test | Test below |
|---|---|
| [15b, 15c] interior band between two well-separated modes: 0 modes, count < 0.5; a mode planted in it and left out of V: count ≥ 0.5 | 7 (15b part; the end-to-end part through `band=` is 15c) |
| [15b] slice with one in-band mode removed after convergence: count ≥ 0.5, flag, mode recovered on restart; mode forcibly withheld: 2 restarts, count flagged | 9 |
| [15b] one in-band mode (not next to an edge) left out: detected for ≥ 97 of 100 seeds | 6 |
| [15b] V complete, τ_c = 1e-12: neighbour deflation off raises the estimate by the probe mean of $\sum_i q(\mu_i)^2 (v_i^\top M y_j)^2$ to 1e-8 relative; on: estimate < 0.5 | 8 |

## Facts

- PR 14a's `count` estimates $s = \frac1{n_p}\sum_j \|\Pi_V\, q(T)\, \Pi_V y_j\|_M^2$ with $T =
  (K + \lambda_c M)^{-1} M$, $x = 2\lambda_c T - I$, whose eigenvalue for mode λ is $x(\lambda) =
  (\lambda_c - \lambda)/(\lambda_c + \lambda) = \cos\theta$, $\theta = 2\arctan\sqrt{\lambda/\lambda_c}$.
  q is the Jackson-damped Chebyshev series of the indicator of $x \ge 0$ (λ ≤ λ_c):
  $c_0 = 1/2$, $c_k = 2\sin(k\pi/2)/(k\pi)$, times the Jackson factors $g_k$. The probe loop
  (sampling, projection, recurrence, reduction, flags, timing) does not depend on the
  polynomial beyond its coefficients γ.
- The indicator of $x \ge x_0$, $x_0 = \cos t$, has $c_0 = t/\pi$, $c_k = 2\sin(kt)/(k\pi)$.
  The band-pass indicator of $[x_0, 0]$ (λ between the lower counting edge $c_{lo}$ and
  $\lambda_c = c_{hi}$) is the difference of two such series, so the Jackson band-pass q is the
  difference of two Jackson step polynomials with the same damping: $\gamma = \gamma(0) -
  \gamma(x_0)$, $x_0 = (1 - a)/(1 + a)$, $a = c_{lo}/c_{hi}$. Near the upper edge the lower step
  is far past its transition, so the band-pass behaves like the step there: d ≈ 23/g still
  holds for the tighter edge.
- Resolution in θ: at the upper edge $d\theta/du = 1/2$ ($u = \lambda/\lambda_c$), at the lower
  edge $\sqrt a/(1+a)$ per unit relative gap, which is ≥ 0.4 for a ≥ 1/4 (≈ 1/2 for narrow
  bands). The roadmap places σ at the band's upper edge; this loses at most 20% resolution at
  the lower edge for bands up to a factor 4 wide, so it is kept.
- `si_slice` returns lam (the converged pairs in the slice, widened over clusters), V (host,
  M-orthonormal), and `info["neighbours"] = {"lo": (λ, v) or None, "hi": ...}`: one converged
  Ritz pair beyond each edge (none below for lo = 0). Pairs in W are deflated from the Krylov
  basis and never return.
- With exact eigenvectors, Π_V commutes with T, so $s_j = \sum_{i \notin V} q(\lambda_i)^2
  (v_i^\top M y_j)^2$; dropping a neighbour v from V adds exactly $q(\mu)^2 (v^\top M y_j)^2$
  (the cross term vanishes because $\Pi_V q(T) \Pi_V y_j \perp_M v$). This is test 8's identity.

## Design

### 1. Polynomial and degree (`eigen/counting.py`)

- `jackson_coef(d, x0=0.0)`: the Jackson-damped indicator of $x \ge x_0$ (above);
  `jackson_step(d)` stays as `jackson_coef(d)` (tests and callers unchanged).
- `count_degree(gap, lam_top, n_top, n_free, lower=None)`: `lower = (lam_in, lam_bot)` in units
  of λ_c (the lowest returned mode, or the band edge when none, and the lowest deflated
  eigenvalue, the lower neighbour) switches to the band-pass polynomial. The rule (roadmap):
  the smallest d with $q^2 \ge 0.95$ at both in-band ends ($\theta$ of `lam_in` and of
  $1 - g/2$) and on the DCT grid between them, and a tail bias ≤ 0.05. The early exit uses the
  smaller of the two edge gaps. Doubling then bisection as now.
- Tail bias with the roadmap's Weyl density $N(\lambda) \propto \lambda^{3/2}$: the existing
  `tail_bias` (upper tail from `lam_top` to $\lambda_{max}$, $N(\lambda_{max}) = n_{free}$) is
  built on a helper `_bias(gamma, l0, l1, n_ref, lam_ref)` $= \int_{l_0}^{l_1} q^2\, dN$ with
  $N = n_{ref} (\lambda/\lambda_{ref})^{a}$; the lower tail is `_bias(gamma, 0, lam_bot, n_top,
  lam_top)` (rigid modes are at 0 and deflated; the model has N(0) = 0). The bias is the sum.
- **Normalization for bands.** A slice does not know its mode index, so `n_top` cannot be the
  count up to `lam_top` as in 14a. It is fitted to the deflated set: $C = (n_W - 1) /
  (\lambda_{top}^a - \lambda_{bot}^a)$ over the $n_W$ deflated eigenvalues (in-band modes and
  the two neighbours, contiguous), `n_top` $= C \lambda_{top}^a$. Underestimating the density
  only raises false alarms (the bias adds to the estimate; it never hides a missing mode), and
  the neighbours, which carry most of the out-of-band weight near each edge, are deflated. An
  empty band fits C to the one interval between its neighbours (logged in `open_items.md`).

### 2. `band_count` (`eigen/counting.py`, exported from `jaxmodal.eigen`)

```python
est, info = band_count(op, free, V, lam, lo, hi, nb_lo, nb_hi, *, R=None, build=None,
                       neighbours=True, n_probes=50, seed=0, tau=None, degree=None,
                       block=None, maxit=1000)
```

- V (n, k ≥ 0), lam: the in-band modes (as `si_slice` returns them); lo < hi: the band edges
  (eigenvalues, lo > 0: a band from 0 uses `count`); `nb_lo`, `nb_hi`: (λ, v) of the converged
  neighbours (`si_slice`'s `info["neighbours"]`), required (a missing one raises: the edge
  cannot be placed).
- Edges (roadmap "Band edges"): $\lambda_{in,lo}$ = min(lam) (lo when lam is empty),
  $\lambda_{in,hi}$ = max(lam) (hi when empty); $c_{lo} = (\lambda_{nb,lo} + \lambda_{in,lo})/2$,
  $g_{lo} = (\lambda_{in,lo} - \lambda_{nb,lo})/c_{lo}$, the same at the top; σ = λ_c = $c_{hi}$.
  So $q^2 \ge 0.95$ holds on $[\lambda_{in,lo}, \lambda_{in,hi}]$, the in-band interval up to
  the returned modes (up to the band edges when none). Requires $\lambda_{nb,lo} <
  \lambda_{in,lo} \le \lambda_{in,hi} < \lambda_{nb,hi}$.
- W = [v_lo, V, v_hi] (`neighbours=True`) or V alone (`False`: the diagnostic of test 8; the
  degree and tail model still use all $n_W$ eigenvalues, so both runs use the same q unless
  `degree` is passed).
- `build(shift)`: the SPD preconditioner of $K + c_{hi} M$ (default `jacobi(op, free, shift)`,
  as `count`'s default). The caller cannot build it before the edges are known; 14b's
  `_verify` uses the same builder convention.
- τ_c, blocks, flags, the degree cap (unverifiable, not run), NaN on a failed solve: as
  `count`. `info` as `count`'s plus `edges` ($c_{lo}$, $c_{hi}$), `gaps` ($g_{lo}$, $g_{hi}$)
  and `neighbours`.
- **Refactor.** `count`'s validation, degree and probe loop are split: a private
  `_estimate(op, free, W, R, gamma, d, lam_c, tau, precond, ...)` runs the probe loop and fills
  the shared info; `count` and `band_count` validate, place the edges, pick d and call it. No
  behaviour change for `count` (its tests unchanged).
- Memory: W has 2 more columns than the in-band modes and is streamed in chunks, so
  `count_memory` / `probe_block` apply unchanged (host W grows by 2 columns).

### 3. `si_checked` (`eigen/shift_invert.py`, internal like `si_slice`)

```python
lam, V, info = si_checked(op, free, lo, hi, precond, *, capacity, build=None,
                          count_options=None, W=None, seed=0, verbose=False, **slice_options)
```

- Run `si_slice` (all its options pass through). If it converged, count: `band_count` with
  `info["neighbours"]` for lo > 0; `count(op, free, [V, v_hi], [lam, λ_hi], max(lam) or hi,
  λ_hi)` for lo = 0 (the step polynomial; roadmap: an edge at λ = 0 has no counting edge).
  `count_options` (n_probes, seed, tau, degree, block, maxit) pass through; `build` as above.
- **Restart rule (roadmap).** A verifiable count ≥ 0.5 sets `info["count_flag"]` and restarts
  the slice: `si_slice` with a fresh random block (seed + j for restart j) and W extended by the
  in-band modes found so far. Only the in-band modes are deflated: the neighbours must
  reappear as Ritz values to anchor the stop rule (deflated, the rule would wait for the next
  modes beyond each edge). The new in-band pairs are appended (they are M-orthogonal to the
  deflated ones by construction), the neighbours replaced by the restart's, and the count
  repeated with fresh probes (count seed + j: a false alarm from one probe draw does not repeat
  with V unchanged). After 2 restarts whose count is still ≥ 0.5 the slice is returned with
  `info["count_failed"]`. An unverifiable count (degree cap, solve floor) and a slice that did
  not converge are not restarted: they return flagged (`count_passed` False).
- `info`: the last run's `si_slice` info (r_acc and tau_eff as the max over runs, residuals and
  thresholds of all returned pairs) plus `runs` (each run's info), `count` (the last estimate),
  `counts` (all estimates), `count_info` (the last), `count_flag`, `count_failed`,
  `count_passed`, `restarts`.
- `verbose`: `si_slice`'s lines plus one line per count (estimate, degree, steps, time, action).
- Memory: `si_slice` releases its basis before returning (15a), so the count runs after it with
  V and the neighbours on the host; the peak is max(slice, count), as `si_memory` and
  `count_memory` give. A restart holds the found modes in its W (part of `si_memory`'s W term).

## Tests

In `tests/test_counting.py` (band count, its reference `ref()` mesh: hex20 6×5×4 clamped,
dense eigenpairs, `exact_precond`) and `tests/test_shift_invert.py` (`si_checked`).

1. `jackson_coef(d, 0)` equals `jackson_step(d)`; the undamped coefficients of $x \ge x_0$
   against quad of the indicator's Chebyshev projection; the band-pass grid stays in
   [−ε, 1 + ε].
2. Band degree rule: for (g_lo, g_hi) with each edge the tighter one and a ≈ 0.5 and 0.9,
   $q^2 \ge 0.95$ at both in-band ends for d and fails for d − 1; d·min(g) in a reported range
   (the step's 19–24 for g_hi tighter); a dense out-of-band spectrum makes the bias set d.
3. Lower tail against scipy quad of $q^2\, dN/d\lambda$ over $(0, \lambda_{bot})$ (as the 14a
   upper-tail test, 1e-3 relative).
4. Expected value (as 14a test 4): the mean of 2000 probes against the exact sum
   $\sum_{i \notin W} q(\lambda_i)^2$ within 5 standard errors, for V complete, one interior mode
   missing, and neighbours off; consistent and lumped mass.
5. Against the dense estimator (as 14a test 5): $s_j$ against the dense band-pass recurrence
   with exact solves on the same probes to 1e-8 relative (missing) / 1e-10 absolute (complete),
   τ_c = 1e-12 (the 14a helper takes γ instead of d).
6. **[15b] Detection:** a band holding several modes, one interior mode (not next to an edge)
   left out: one 5000-probe count as 100 sets of 50 (the 14a scheme, deterministic), ≥ 97 sets
   ≥ 0.5; V complete: every set < 0.5 (slow, as 14a).
7. **[15b] Empty band:** in the largest relative gap among the first 40 modes, a band in the
   middle third of the gap: `si_checked` returns 0 modes and a count < 0.5. Planted: the same
   band on the model with the density scaled by s so that the mode above the gap moves to the
   band centre (all eigenvalues scale by 1/s; the case is chosen so that both of its
   neighbours stay outside the band): `band_count` with V empty and the dense neighbours gives
   ≥ 0.5.
8. **[15b] Neighbour deflation:** V complete (dense modes, τ_c = 1e-12, exact preconditioner):
   the estimate is < 0.5; with `neighbours=False` and the same degree, the estimate rises by
   the probe mean of $\sum_{i \in nb} q(\mu_i)^2 (v_i^\top M y_j)^2$ to 1e-8 relative ($y_j$
   from the same keys, dense solves).
9. **[15b] Restarts** (`monkeypatch` of `shift_invert.si_slice`): (a) the first run's result
   with one in-band mode removed: count ≥ 0.5, `count_flag`, one restart recovers it, final
   count < 0.5, λ against the dense reference (15a tolerances) with multiplicities, V
   M-orthonormal; (b) the mode withheld on every run: 3 `si_slice` calls, `restarts == 2`,
   every count ≥ 0.5, `count_failed`, `count_passed` False; the restart W holds the found
   modes and the seeds differ.
10. `si_checked` with lo = 0 (clamped) uses the step count: < 0.5, `count_info["lam_c"]`
    midway between the last mode and the neighbour.
11. Not restarted: an unverifiable count (a monkeypatched `band_count` returning NaN,
    unverifiable) and an unconverged slice (capacity full) return flagged with
    `restarts == 0`; verbose line count (`capsys`).
12. `band_count` errors: missing neighbour, lo = 0, neighbour inside the band, bad shapes; info
    fields (`edges`, `gaps`, `degree`).

Run: these tests after each commit, then the full `pytest -m "not gpu"` (slow included).

## Benchmark

None: no [15b] criterion measures cost, and the count's cost per step is 14a's (the band-pass
changes only γ). A host-only table of the band degree d against (g_lo, g_hi, a) goes into the
docs for 15c's cost model (seconds to compute, no solves).

## Docs

- `docs/eigensolvers.md`, "Completeness count": the band-pass polynomial, its edges, the
  degree rule with two edges, the lower tail and the band normalization, neighbour deflation,
  the degree table; "Shift-invert slices": `si_checked` and the restart rule.
- `README.md`: status line; a `band_count` line next to `count`.
- `docs/ROADMAP.md`: PR 15b status.
- `docs/open_items.md`: the decisions below.

## Decisions without user input (to log)

- The band tail model is fitted to the deflated set (no mode index in a slice).
- Restarts deflate the in-band modes only; recounts use fresh probes (seed + j).
- σ of the band count at the upper counting edge (roadmap), resolution loss at the lower edge
  ≤ 20% for bands up to a factor 4 wide.
- The count preconditioner comes from a builder at $c_{hi}$ (default Jacobi); 15c passes the
  multigrid builder.
- Unverifiable counts and unconverged slices are not restarted.
