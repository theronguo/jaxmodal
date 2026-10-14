# PR 15d plan: band strategy and `band=(f1, f2)`

Branch `pr-15d-strategy` from `main` (PR 15c merged). Roadmap: PR 15 (the [15d] items), the
"Frequency bands" section (strategy, band edges, interior bands) and the PR 15c plan's split
(15d = the cost model, the memory budget, the strategy choice, `verify=None` for bands and
`solve_modes(..., band=(f1, f2))`).

## Scope

- `solve_modes(..., band=(f1, f2))` with $f_1 > 0$ (interior bands), plus `strategy`
  ("auto", "lobpcg", "shift_invert") and `memory_budget` (bytes).
- The strategy: LOBPCG from 0 (PR 14b's band mode, filtered to the band) while it is predicted
  cheaper and its basis fits the budget; otherwise shift-invert slices (`si_band`, PR 15c).
- The cost model: LOBPCG time from chunk 0's measured iteration time, the slice model from PR
  15c and the MINRES iterations against depth from PR 15a's study, fitted in a new benchmark.
- The memory budget: the LOBPCG basis limit, the slice capacity from `si_memory`, the host
  store of the found modes.
- The $f_1 < \lambda_1$ rule (solve $[0, f_2]$ and filter) and the up-front check of $\lambda_2$
  against an upper bound $\hat\lambda_\max$ of the spectrum.
- `verify=None` for bands: a per-leaf decision on the predicted count cost.
- `solve_modes_memory(..., band=(f1, f2), strategy=...)`, the README example, the docs.

Out of scope (deferred, listed in `docs/open_items.md`): the slice-efficiency items of PR 15c's
"Late capacity errors and slice efficiency" list (verification slices, preconditioner reuse on
a whole requeue, per-step residuals, merge copies, host transposes, `verify=False` input work,
stopping si_checked before its count on a window rerun) go to PR 18 (performance); the depth-1000
options (deflating converged modes from MINRES, a σ-aware multigrid, a lean LOBPCG fallback) go
to PR 18 and the deferred GPU checks.

## Roadmap tests tagged [15d]

| Roadmap test | 15d test |
|---|---|
| [15a, 15c, 15d] matches `eigsh(sigma=...)` (gap ≥ 1e-2 to max(1e-8, 10 ε λ̂_max/λ), else 10 r_acc, multiplicities) | 1, 2 |
| [15b, 15c, 15d] interior band between two well-separated modes: 0 modes, count < 0.5 (the planted-mode half is 15b/15c's, through `band_count`/`si_band`) | 3 |
| [15a, 15c, 15d] clamped (0, f_max) forced onto shift-invert stops and matches `eigsh` | 4 |
| [15d] f₁ below λ₁ solved as [0, f₂], filtered, matches `eigsh` | 5 |
| [15d] memory budget too small for the LOBPCG-from-0 basis → shift-invert | 6, 7 |
| [15a, 15d] MINRES iterations vs δ measured (15a), cost-model constants fitted | 11 (benchmark + recorded fit) |
| PR 15 contents: capacity per slice from the memory model | 9 |
| Band edges: an edge above the largest computed eigenvalue raises | 8 (up front); 15c tests 3, 11 (late) |

## Design

### API

```python
lam, V, info = solve_modes(mesh, materials, bcs, band=(f1, f2))           # auto
solve_modes(..., band=(f1, f2), strategy="shift_invert", memory_budget=8e9)
```

- `band=(f1, f2)`: $0 \le f_1 < f_2$ finite (Hz). $f_1 = 0$ keeps PR 14b's behaviour under
  `strategy="auto"` unless the budget forces shift-invert. `check_band` accepts $f_1 > 0$.
- `strategy`: "auto" (default), "lobpcg" (PR 14b's band mode from 0, filtered; never switches)
  or "shift_invert" (`si_band` on the band). Without `band`, any `strategy` but "auto" raises.
- `memory_budget`: bytes for the solve; default 0.75 × the device's `bytes_limit` on GPU
  (`jax.devices()[0].memory_stats()`), 0.75 × physical RAM on CPU (`os.sysconf`, unified memory on
  the Mac). Shift-invert sizes its slice capacity from it; "auto" switches on it.
- `n_modes` stays the expected count (PR 14b): the LOBPCG basis and the expected slice
  modes. `verify` and `count_options` as in PR 14b.
- Returns the band modes (clusters reaching into $[\lambda_1, \lambda_2]$ kept whole, as PR 14b's
  upper edge, now also at the lower edge) and `info` with `strategy` (the one used), `choice`
  (the reason: "forced", "memory", "cost"; the predicted seconds of both strategies), `band`,
  `count` (PR 14b's for LOBPCG; si_band's summary for slices), and for slices `slices`,
  `capacity`, `merged_edges`, `collapsed`, `edge_rr_failed`, `duplicates` (si_band's info).

### Up-front checks

- **Spectrum bound** $\hat\lambda_\max = \max_e \lambda_\max(K_e, M_e)$: for any $x$,
  $x^\top K x = \sum_e x_e^\top K_e x_e \le \max_e \lambda_e \sum_e x_e^\top M_e x_e \le
  \hat\lambda_\max\, x^\top M x$ (nodal masses and masking only lower the Rayleigh quotient). Voxel
  meshes: one reference element per phase (host, exact). Unstructured: `element_matrices` (PR 12)
  and the element masses in device chunks, a batched Cholesky of $M_e$ and `eigvalsh` (lumped:
  diagonal $M_e$). $\lambda_2 > \hat\lambda_\max$ raises `ValueError` before any solve.
  An edge in $(\lambda_\max, \hat\lambda_\max]$ still reaches si_band's capacity error
  (PR 15c) or PR 14b's "more modes than the mesh provides".
- **$f_1 < \lambda_1$:** $\lambda_1^{ub}$ = the σ pre-estimate (an upper bound of $\lambda_1$; with
  Jacobi `sigma_estimate`). If $\lambda_1 = (2\pi f_1)^2 < \lambda_1^{ub}$ the band is solved as
  $[0, \lambda_2]$ (si_band from 0, or LOBPCG) and filtered; `info["from_zero"]` True.

### Strategy ("auto")

1. Run PR 14b's band mode with the LOBPCG basis capped at the budget: `lobpcg._solve(...,
   lam_stop=λ2, max_capacity=C_max)`, $C_{max}$ the largest basis whose `solve_modes_memory`
   total (plus the host V) fits `memory_budget`; past it the basis does not grow and the solve
   returns its locked modes with `info["stopped"] = "capacity"`.
2. After each lock (`on_lock` callback, the locked $\lambda$ so far), predict:
   - $\hat N(\lambda_2)$: a Weyl fit $N \propto \lambda^a$ to the locked modes (as 15b's tail model;
     $a$ fitted, clipped to $[0.3, 1.5]$);
   - $T_\text{lobpcg}$: the remaining chunks $\lceil (\hat N - N)/m\rceil$ times chunk 0's
     iterations times $t_\text{it}(N)$, $t_\text{it}(N) = t_\text{it,0}(1 + \kappa N/m)$ (the
     projector's growth; κ fitted);
   - $T_\text{si}$: slices × steps × MINRES iterations × $t_\text{minres}$: modes in the band
     $\hat N(\lambda_2) - \hat N(\lambda_1)$; columns per slice $1.8\,n + 84$ (PR 15c), so
     $\lceil\text{columns}/C\rceil$ slices at capacity $C$ and steps = columns $/b$; MINRES
     iterations $\alpha\,D^{p}$ at depth $D = \hat N$ at the band middle (α, p fitted per
     preconditioner family on PR 15a's table); $t_\text{minres} = \beta\, (b/m)\, t_\text{it,0}$
     (β fitted).
   Switch when $\hat N(\lambda_2)$ exceeds $C_{max}$ or $T_\text{si} < T_\text{lobpcg}$. A
   switch keeps the locked modes; a stop at $C_{max}$ also switches.
3. On a switch: `si_band` on $[\lambda_1, \lambda_2]$ (or $[0, \lambda_2]$ under the
   $f_1 < \lambda_1$ rule), seeded with the locked modes as stored pairs (`known=(lam, V)`, a new
   si_band argument: they enter the store as run −1, are deflated by every window that holds
   them and are never returned). The nearest locked mode below $\lambda_1$ closes the first
   slice's lower side. Locked modes inside the band are re-verified by the slices (deflated;
   rare: the prediction failed late).
4. No switch: PR 14b's count at the cut above $\lambda_2$ verifies $[0, \lambda_2]$; the result is
   filtered to the band.

`strategy="shift_invert"` skips 1–2 (no LOBPCG; `n_modes` or $\hat N$ from the Weyl law with
$a = 3/2$ through $\lambda_1^{ub}$ sizes the store term). The LOBPCG chunk 0 of "auto" is the
price of the prediction (40 modes at most; PR 14b's band_study: seconds at 1M DOF).

### Capacity and memory

- Slice capacity: the largest multiple of $c$ with `si_memory(op_memory, n, capacity,
  n_modes=⌈1.25 m_s⌉, block, pc, n_rigid, n_deflate=2c)` + host store $8n(\hat N_\text{band} +
  n_\text{known})$ ≤ `memory_budget`, $m_s$ the expected modes per slice
  ($(\text{capacity} - 84)/1.8$). Below $2b$ + one window chunk: `ValueError` naming the budget.
- `solve_modes_memory(mesh, bcs, n_modes, band=(f1, f2), strategy=..., capacity=None,
  memory_budget=None)`: "lobpcg" as PR 14b; "shift_invert" the `si_memory` composition above
  (the capacity from the budget when not given). "auto" returns both estimates.

### `verify` for slices

si_band's `verify` takes a callable `verify(leaf) -> bool` besides bools (leaf: degree, gap,
the slice's MINRES time per iteration and its time). `verify=None` passes PR 14b's three-stage
rule per leaf: skip above `MAX_DEGREE`; skip when the bound $2\,n_\text{blocks}\,d\,(b_p/b)\,
t_\text{minres}$ exceeds the slice's time; else calibrate (PR 14b's `_calibrate`, the slice's
preconditioner) and count iff the prediction ≤ the slice's time. Skipped leaves record
`predicted_s` and `reason`.

### Errors

- si_band's capacity errors (`_cap_error`) carry the band so far: `err.result = (lam, V, info)`
  (the stored pairs, merged as at the end, `converged` False, the reason); `solve_modes`
  re-raises with it. The message is worded per case (PR 15c open item): "lo may be below
  lambda_1" only when no lower side closed, "hi above the spectrum" only when no upper side did.

## Benchmark (`benchmarks/strategy_study.py`, ≤ 1 h, progress per run)

- Fit α, p per family (AMG hex20, GMG voxel) on `shift_invert_study.json` (no new runs).
- Measure, per mesh (hex20 $12^3$ AMG, voxel $24^3$ GMG): `solve_modes(band=(0, f))` for three
  $f$ ($N$ ≈ 40, 120, 250: $t_\text{it}(N)$, κ) and `si_band` on two bands (depth ≈ 50 and 150,
  10–20 modes: β and the slice model); record predicted against measured seconds.
- Results in `benchmarks/results/strategy_study.json`, cited in `docs/eigensolvers.md`.

## Tests (`tests/test_strategy.py`; fast unless marked)

1. Auto on hex20 $6\times5\times4$ (reference by `reference_modes`): an interior band of about
   10 modes chooses LOBPCG ("cost"), matches `eigsh` (roadmap tolerances, multiplicities), clusters
   at both edges whole.
2. `strategy="shift_invert"` on the same band with a budget giving two or more slices: matches
   `eigsh`, every counted leaf < 0.5, `info["capacity"]` from the budget.
3. A band strictly between two well-separated modes (voxel cube): 0 modes, count < 0.5, under
   both strategies.
4. Clamped box, `band=(0, f)` with `strategy="shift_invert"`: matches `eigsh` (no lower
   neighbour).
5. $f_1$ below $\lambda_1$: `from_zero` True, filtered, matches `eigsh` (both strategies).
6. `memory_budget` below the LOBPCG basis for the band but above one slice: auto switches
   ("memory") after chunk 0, matches `eigsh`.
7. A budget whose cap stops LOBPCG inside the band (prediction overridden by a fake `on_lock`):
   the switch seeds si_band with the locked modes, no duplicates, matches `eigsh`.
8. $\lambda_2 > \hat\lambda_\max$ raises before any solve (no LOBPCG, no slice); $\hat\lambda_\max
   \ge$ the dense largest eigenvalue on hex8/hex20/voxel/lumped/nodal-mass meshes, and equals it
   on a one-element mesh.
9. Capacity: `si_memory` total of the chosen capacity ≤ budget < that of the next chunk; a
   budget below the minimum raises naming it.
10. `verify` for slices: True/False forced per leaf; None skips a leaf whose bound exceeds its
    slice time (faked times) and counts a cheap one; the per-leaf `predicted_s`, `reason`.
11. Cost model: the fitted constants reproduce the recorded study (MINRES its within 30% of
    the table below depth 503; predicted seconds within 2× of measured in
    `strategy_study.json`).
12. A capacity error carries the band found so far (`err.result`, `converged` False).
13. Options: `strategy` without `band`, bad `memory_budget`, `f1 >= f2`, `strategy="lobpcg"`
    with an interior band (filtered result) raise or behave as documented.
14. (slow) `solve_modes_memory(band=..., strategy="shift_invert")` within 20% of the measured
    peak of a voxel $86^3$ band near depth 6 (estimate ≥ 2 GB, a few minutes).

## Docs

- `docs/eigensolvers.md`: "Interior bands and strategy" (API, checks, the strategy and cost
  model with the fitted constants, capacity and memory, `verify` for slices, errors); the
  Validation list; the [0, f_max] section's "no strategy selection yet" sentence removed.
- `README.md`: the `band=(f1, f2)` example; status.
- `docs/ROADMAP.md`: PR 15d status; PR 15 complete.
- `docs/open_items.md`: decisions without user input (below), the deferred items.

## Decisions without user input (to log)

- The strategy predicts from chunk 0 of LOBPCG (Weyl fit to its modes), not a priori: the
  exponent varies from 0.4 (plates) to 1.5 (bulky), so a fixed Weyl law misjudges the basis.
- The default budget is 0.75 of device memory (GPU) or physical RAM (CPU).
- A late switch re-verifies the locked in-band modes by slices instead of mixing a step count
  for the LOBPCG part with the slices' band-pass counts.
- Capacity errors keep raising (bad bands are user errors) but carry the pairs found.
