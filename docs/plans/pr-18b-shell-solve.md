# PR 18 plan: performance pass, part b: solid-shell AMG in `solve_modes`

Branch `pr-18b-shell-solve` (this plan), from `main` after PR 18a. Roadmap: PR 18b
("`"auto"` → AMG on `ssh8` meshes, the σ pre-estimate on shells, band-mode reuse, the
`amg_memory` shell family and its calibration"; test: "`solve_modes` with AMG on a solid-shell
mesh matches the PR 8 `eigsh` to PR 9 tolerances"), the 18a plan's "Carried into 18b's plan"
list, and the "Thin plates and slender beams" entry of `docs/open_items.md`.

## Revisions

- **Revision 1:** first draft, from the sizing study (scripts in the session scratchpad `s18b/`).
- **Revision 2 (plan review round 1):**
  - **Diagnosis corrected.** The thin-shell stagnation comes mostly from drift in LOBPCG's
    implicitly updated products ($KX$, $MX$, $KP$, $MP$; explicit only every `RECOMPUTE` = 20
    iterations), not from preconditioner noise. New: an explicit-product regime once a chunk
    stops progressing. With it the polish is a one-step finisher.
  - **Rigid-mode leak found (a bug on `main`).** On free thin shells the 9-block step lets the
    rigid modes back into the block (|RᵀMV| up to 0.96); `solve_modes` returns them as elastic
    modes (flagged not converged). Fix: every recompute also re-deflates $X$ and $P$ against
    $R$ and the locked basis. The reviewers' "free $24^2$, stack and cylinder stall at
    $10^{-10}$" were this leak and the drift.
  - **Polish:**
    - only X is kept across the PCG (the state is freed), and the memory model takes
      `max(step, polish)`;
    - the PCG tolerance follows `tol` ($\min(10^{-7}, 0.1\,\mathrm{tol})$), and the default
      step cap is 5;
    - the polish runs before the failed-exit guard check;
    - a needed refill ends it quietly;
    - λ on a non-converged exit is the Rayleigh quotient;
    - it is offered with `"amg"` only.
  - **Tests:**
    - the PR 9 tolerances are `check_lam` plus principal angles (`r_M`, `max_sine`), not MAC;
    - the polish and explicit regime are pinned by counts and spies;
    - per-component storage tests and polish memory tests are added;
    - the existing tests that change are listed;
    - the thin hex20 bullet is dropped (it does not stall: `maxit` exits first).
  - **Memory model:** the count gets an $n$-independent term; β and γ values, step-fit range
    ≥ 100k, measured-maximum wording for $\alpha_T$, a GPU host-build deferred check.
  - **Docs:** a file:line list and a carried-items list; the benchmark gets a time budget.

## Goal (18b)

- `solve_modes(precond="auto")` uses AMG on meshes whose blocks are all `ssh8`, or `ssh8` and
  `hex8`; `precond="amg"` is accepted there. `ssh8` with `hex20`/`hex27` keeps Jacobi under
  `"auto"`, and `precond="amg"` raises up front with `amg()`'s message.
- The roadmap test: on solid-shell meshes `solve_modes` with AMG matches the PR 8 `eigsh` to
  the PR 9 tolerances at tol $10^{-10}$: `check_lam` (λ relative error ≤
  $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$ for relative gaps ≥ $10^{-2}$, $10\,$tol
  cluster-wise otherwise) and principal angles $\sin\theta \le 10\, r_M \lambda/\text{gap}$
  per group (as `test_m4_reference`).
- **Thin shells converge.** Today LOBPCG stagnates on solid shells from about $L/t = 300$ at
  tol $10^{-6}$ (and from $L/t \approx 100$ at $10^{-10}$). 18b adds the explicit-product regime
  and a polish stage that reach the floor.
- `amg_memory` covers `ssh8` meshes (storage, build, step, count, polish), so
  `solve_modes_memory`, band mode, the slice and count estimators and the strategy work on
  shells.
- Band mode reuses the AMG aggregates on shells as on hex meshes (`return_aggregates`, already
  generic).

## Findings (sizing study and review round 1, CPU fp64, M3 Max)

### `solve_modes` with shell AMG works as is below $L/t \approx 200$

With the dispatch patched (prototype), 20 modes, tol $10^{-6}$, defaults:

| case | DOF | AMG: its / time | Jacobi: its / time | λ error vs `eigsh` |
|---|---|---|---|---|
| flat $32^2$, $L/t = 50$, SS | 6.5k | 16 / 3.3 s | 840 / 4.4 s | $2\cdot10^{-11}$ |
| free $24^2$, $L/t = 50$ | 3.8k | 17 / 3.0 s | 1307 / 9.0 s | (rigid filter) |
| T-junction $24^2$, $L/t = 50$, clamped | 5.7k | 16 / 2.6 s | 998 / 5.0 s | $7\cdot10^{-12}$ |
| flat $64^2$, $L/t = 100$ | 25k | 17 / 4.7 s | 2392, not converged / 30 s | $5\cdot10^{-10}$ |
| curved $32^2$, $L/t = 50$ | 6.5k | 19 / 3.3 s | 1145 / 6.0 s | $1\cdot10^{-11}$ |
| stack $32^2\times3$, $L/t = 100$ | 13k | 19 / 3.8 s | 3000, not converged / 27 s | $4\cdot10^{-9}$ |
| web $32^2$ (`ssh8` + `hex8`), $L/t = 50$ | 7.1k | 19 / 3.7 s | 844 / 4.7 s | $6\cdot10^{-12}$ |

The σ pre-estimate (`_precond_sigma`) needs no change: 6–9 iterations with σ = 0 first on
supported shells, 34 on the free plate (σ = 0 refused, `sigma_estimate` first), and it lands
within 0.3% of the Kirchhoff $\lambda_1$ at $L/t = 1000$.

### LOBPCG stagnates on thin shells: drift in the implicit products

$\kappa = \lambda_\max/\lambda_1$ grows like $(L/t)^4$ (thickness modes) and does not depend on
$h$: $4.8\cdot10^7$, $7.3\cdot10^8$, $3.7\cdot10^9$, $4.5\cdot10^{11}$ at $L/t$ = 100, 200,
300, 1000. LOBPCG's floor thresholds $t_i$ (`C_FLOOR` term) are $3\cdot10^{-4}$ (mode 1) to
$1.3\cdot10^{-6}$ (mode 20) at $L/t = 1000$; exact eigenvectors (`eigsh` shift-invert) have
residuals 0.06–0.2× $t_i$. Flat SS $32^2$, 20 modes, AMG: today converged at $L/t$ = 100 / 200
(tol $10^{-6}$), stagnated (`STALL` exit) at $L/t \ge 300$ with worst $r_i/t_i$ up to
$8\cdot10^5$ although the Ritz values are within $3\cdot10^{-5}$. Jacobi stagnates the same way.

**Cause (review round 1).** The 9-block step updates $KX$, $MX$, $KP$, $MP$ by recurrence and
recomputes them every `RECOMPUTE` = 20 iterations. At κ ~ $10^8$–$10^{11}$ the drift between
recomputes swamps the residual. Worst $r_i/t_i$ on exit (`RECOMPUTE` patched):

| case | today (20) | every iteration (1) | low-memory (fully explicit) |
|---|---|---|---|
| $32^2$, $L/t = 1000$, tol $10^{-6}$ | 210 its, $7.9\cdot10^5$ | 134 its, 35 | 261 its, 58 |
| $32^2$, $L/t = 300$, tol $10^{-6}$ | 190 its, $2.5\cdot10^4$ | 97 its, 6.4 | |
| $32^2$, $L/t = 100$, tol $10^{-10}$ | 170 its, $1.7\cdot10^5$ | 35 its, converged | |
| $128^2$, $L/t = 1000$, after 40 its | $1.1\cdot10^8$ | 24 | |

Recomputing every iteration everywhere costs +16% per iteration with AMG at 100k DOF and about
2× with Jacobi. Recomputing only once a chunk stops progressing (**trigger**: 5 iterations
without progress, then every iteration until the chunk ends) does as well and leaves
converging solves unchanged. The other trigger tried, "near" (wanted $r_i^2 \le t_i$), alters
converging solves' last iterations:

| case | today | trigger: 5 its stuck | trigger: near |
|---|---|---|---|
| $32^2$, $L/t = 1000$, $10^{-6}$ | 210, stagnated, $7.9\cdot10^5$ | 112, stagnated, 37 | 131, stagnated, 28 |
| $32^2$, $L/t = 300$, $10^{-6}$ | 190, stagnated, $2.5\cdot10^4$ | 108, stagnated, 2.9 | 99, stagnated, 2.7 |
| $32^2$, $L/t = 100$, $10^{-10}$ | 170, stagnated, $1.7\cdot10^5$ | 33, converged | 30, converged |
| $32^2$, $L/t = 50$, $10^{-10}$ / $10^{-6}$ | 24 / 16, converged | identical | 24 / 16, ratios changed |
| $64^2$, $L/t = 100$, $10^{-6}$ | 17, converged | identical | 17, ratios changed |

Even fully explicit, the residual stays 3–58× above the floor at $L/t \ge 300$ (the low-memory
step too): the iterates keep a high-frequency component that the λ-scaled 2-norm residual
amplifies by about κ. One block inverse-iteration step $Y = A_P(\sigma)^{-1} M X$ (PCG with the
V-cycle), deflation, SVQB and Rayleigh–Ritz damps it by about $\lambda/\lambda_\max$. A LOBPCG
restart after it brings the component back, and LOBPCG with the inner PCG as its preconditioner
breaks down ($W \approx X$), so the polish stays pure inverse iteration.

### Free bodies: the rigid modes leak back

The 9-block step deflates $W$ against $R$ once, before its two orthogonalize-and-SVQB passes;
$X$ and $P$ are never re-deflated, and Rayleigh–Ritz pulls any rounding-level rigid component
in (its Ritz value is about 0, below every elastic one). Tol $10^{-10}$, 20 modes, AMG:

| case | today | regime | regime + polish | regime + re-deflation + polish | low-memory, today |
|---|---|---|---|---|---|
| free $24^2$, $L/t = 50$ | 130 its, λ₀ ≈ 0, $\|R^\top MV\|$ = 0.96 | 71, $4\cdot10^{-14}$, worst 5.3 | converged, 1 step | converged (96 + 1 step), $7\cdot10^{-16}$ | converged, 26 its |
| cylinder (free) | 170, λ₀ ≈ 0, 0.90 | 170, 0.93 | not accepted (progress stop) | converged (123 + 1 step), $2\cdot10^{-16}$ | converged, 30 its |

λ errors after the fix: $2.8\cdot10^{-7}$ (free plate) and $7.4\cdot10^{-7}$ (cylinder) against
`eigsh`; the test checks them with `check_lam` (whose floor $10\varepsilon\lambda_\max/\lambda$
is $1.5\cdot10^{-8}$ / $4.6\cdot10^{-6}$ for mode 1). Today both return 6 near-zero "elastic"
modes. With the regime, the 3-layer stacks ($L/t$ = 50 / 100), which stalled at 190 / 250
iterations, converge in 29 / 51 without a polish.

### Explicit regime plus polish (prototype)

On a `STALL` exit, polish steps with explicit residuals after each, accept when every wanted
$r_i \le t_i$; PCG tol $\min(10^{-7}, 0.1\,\mathrm{tol})$, maxit 1000:

| case | tol | LOBPCG (trigger: 5 stuck) | polish steps (worst $r_i/t_i$) | PCG its | total |
|---|---|---|---|---|---|
| $32^2$, $L/t = 300$ | $10^{-6}$ | 108, stagnated | 1 (0.097) | 21 | 5.7 s |
| $32^2$, $L/t = 1000$ | $10^{-6}$ | 112, stagnated | 1 (0.057) | 118 | 7.0 s |
| $32^2$, $L/t = 1000$ | $10^{-10}$ | 112, stagnated | 1 (0.041) | 221 | 8.2 s |
| $32^2$, $L/t = 100$ | $10^{-10}$ | 33, converged | none | – | 3.8 s |
| $32^2$, $L/t = 1000$, 30 modes in chunks of 10 | $10^{-6}$ | 90 / 163 / 129, stagnated | 1 each (0.039, 0.056, 0.10) | 118 / 84 / 47 | 12 s |
| $128^2$, $L/t = 300$ | $10^{-6}$ | 72, converged | none | – | 22 s |
| $128^2$, $L/t = 1000$ | $10^{-6}$ | 79, stagnated | 1 (0.083) | 71 | 37 s |

(Before the explicit regime the same polish needed 4–6 steps at $L/t = 1000$, 166 s at
$128^2$.) The polished λ error against `eigsh` is $4.5\cdot10^{-6}$ at $L/t = 1000$; `eigsh`
is itself floor-limited there ($\varepsilon\kappa \approx 10^{-4}$), so the test bounds it by
$\varepsilon\kappa$. Chunked: $V^\top MV - I$ = $6\cdot10^{-15}$, no mode returned twice (fp32
basis: $2.6\cdot10^{-9}$). Today a stalled exit returns λ below the true value (956.6276 vs
956.6574 at $L/t = 1000$): `_explicit` recomputes $r$, $t$ but not λ.

PCG tolerance (32², $L/t = 1000$, tol $10^{-6}$, before the explicit regime): $10^{-10}$,
$10^{-8}$ and $10^{-7}$ give identical polish ratios at 196 / 142 / 118 PCG its, $10^{-6}$
needs one more step. At tol $10^{-10}$ (32², $L/t = 100$) PCG $10^{-8}$ stalls; $10^{-10}$ and
$10^{-12}$ converge. Hence $\min(10^{-7}, 0.1\,\mathrm{tol})$.

### Storage (device bytes, `amg` defaults, fp64)

Per component (`nbytes` of distinct leaves; level matrices without the operator):

| case | DOF | L0 Schwarz | `schwarz.nbytes` of the L0 patches | $T_0$ B/DOF | coarse levels per shell fiber / hex-only node | total B/DOF |
|---|---|---|---|---|---|---|
| flat $64^2$ | 25k | 10.22 MB | 10.22 MB (+16 B) | 700 | 4.69 kB | 1997 |
| flat $128^2$ | 100k | 40.90 MB | equal | 571 | 5.51 kB | 1910 |
| flat $256^2$ | 396k | 163.6 MB | equal | 529 | 5.80 kB | 1934 |
| stack $128^2\times3$ | 200k | 157.3 MB | equal | 540 | 5.51 kB | 1797 |
| stack $64^2\times10$ | 139k | 290.4 MB | equal | 549 | 4.69 kB | 2801 |
| T 3/3 layers $48^2$ | 45k | 43.0 MB | equal | 614 | 3.97 kB | 1988 |
| block $32^3$ + skin | 111k | 4.6 MB | equal | 563 | 1.70 kB (hex) | 1169 |
| block $24^3$ + skin | 49k | 2.3 MB | equal | 618 | 1.76 kB (hex) | 1225 |

- Level 0's Schwarz equals `schwarz.nbytes` of the level-0 patch sizes, which the mesh alone
  gives (`_fibers` and `_columns` read only the `ssh8` connectivity).
- $T_0$ (rectangular block-ELL $P_0$ and $P_0^\top$): 501–700 B per fine DOF (518 / 526 / 501
  at $16^2$ / $24^2$ / $32^2$, 700 at $64^2$, 571 at $128^2$); not monotone in size.
- The coarse levels scale with fiber groups, not DOFs (the 10-layer stack equals the flat plate
  of the same in-plane size): 4.0–5.8 kB per shell fiber, growing with the level count by
  shrinking increments (+0.82, then +0.29 kB per 4× DOFs), and about 1.7 kB per hex-only node.
- The rest (free masks, Chebyshev scalars) is about 8 B per DOF.

### Step, polish, build and count

- **LOBPCG step** (`memory_analysis`, $m = 15$, extra over the Jacobi step, in blocks $8nm$):
  9-block 27.4 / 16.4 / 12.5 (flat $64^2$ / $128^2$ / $256^2$), 12.4 (stack ×3), 11.4 (stack
  ×10), 21.6 (T 3/3), 18.2 (block $32^3$); low-memory 40.3 / 20.4 / 17.5, 16.4, 15.8, 24.9,
  22.1. The $n$-independent part is about $5\cdot10^5$ DOF-equivalents per column (the coarse
  levels' chunk-bounded Schwarz gathers and unpacked inverses), so the fit uses the cases from
  100k DOF up. Degree 3: within ±1.1 blocks of degree 2.
- **Polish** (`memory_analysis` of `_pcg(ap_operator, V-cycle, b)` at width $m$, review round
  1, blocks $8nm$; step figures include the 6-block state):

  | case | $m$ | PCG temps | 9-block step | polish, state kept | polish, state freed |
  |---|---|---|---|---|---|
  | flat $128^2$ (100k) | 25 | 30.3 | 31.0 | 38.3 | 32.3 |
  | flat $128^2$ | 15 | 34.7 | 34.0 | 42.7 | 36.7 |
  | flat $256^2$ (396k) | 25 | 25.5 | 27.4 | 33.5 | 27.5 |
  | hex20 $20^3$ (AMG) | 25 | 21.8 | 24.3 | 29.8 | 23.8 |
  | hex8 $40^3$ (AMG) | 25 | 23.2 | 26.0 | 31.2 | 25.2 |

  With the state freed the polish is at most the 9-block step plus about 2.7 blocks (flat
  $128^2$, $m = 15$) and 0.2–0.5 blocks above `step_low`: the model takes the maximum.
- **Host setup peak** (fresh subprocess, above the built operator, CPU: includes the device
  storage): flat 0.46 / 1.33 / 2.61 / 5.14 GB at 100k / 396k / 791k / 1.58M DOF (4.6 / 3.4 /
  3.3 / 3.3 kB per DOF), stack ×3 $128^2$ 3.8 kB, stack ×10 $64^2$ 4.8 kB, T 3/3 $48^2$ 6.3 kB
  (45k DOF, the fixed part dominates), block $32^3$ / $48^3$ 3.7 / 3.9 kB. Beyond the stored
  hierarchy: 1.3–2.7 kB per DOF on shells (2.7 at 100k), 2.5–2.75 kB on the hex-dominant
  blocks.
- **Count:** not measured yet (the PR 14 recurrence with the V-cycle; the same
  `memory_analysis` method as `AMG_COUNT`). The V-cycle's chunk-bounded part (about $5\cdot10^5$
  DOF-equivalents per column, 7× the hex family's) needs an $n$-independent term.

## Design (18b)

### Dispatch (`eigen/api.py`)

- `resolve_precond`: `"auto"` → `"gmg"` on voxel meshes; on unstructured meshes `"amg"` unless
  the mesh has `ssh8` with `hex20`/`hex27` (`"jacobi"` there, as now). `precond="amg"` on such a
  mesh raises `amg()`'s "solid-shell AMG supports ssh8 and hex8 blocks" before any build.
- `no_shell` leaves `resolve_precond` and `amg_memory`; `pmg` keeps its refusal (p-MG has no
  shell support).
- The `make` / `lobpcg_precond` closures, `_precond_sigma`, band-mode aggregate reuse and the
  slices need no change: `amg` takes `nonsingular`, `return_aggregates` and `aggregates` on
  shells already (18a).

### Explicit-product regime (`eigen/lobpcg.py`)

- `EXPLICIT = 5`: in `_iterate`, once `it - it_best >= EXPLICIT`, every later 9-block step of
  the chunk runs with `recompute=True`. Until the regime starts nothing changes but the
  re-deflation below, so single-chunk solves without $R$ that never get stuck for 5
  iterations are bitwise unchanged (pinned). The low-memory step is explicit already (it does
  not leak: measured above).
- **Re-deflation:** the recompute branch (`lax.cond`) deflates $X$ and $P$ against $R$ and
  the locked basis (`_deflate`) before their explicit products: every `RECOMPUTE` iterations,
  and every iteration in the regime. Cost per recompute: $R$'s rank-$k$ update and, with a
  basis, one M apply on each of $X$, $P$ plus the projection. Solves with neither $R$ nor a
  basis are unaffected; the others change at rounding level from iteration 20 on.
- `rec["explicit"]`: the iteration at which the regime started (None if never), in
  `info["chunks"]`.

### Polish: inverse iteration on a stagnated chunk (`eigen/lobpcg.py`)

- `lobpcg(..., polish=None, polish_steps=5)`, also `_solve`. `polish`: a host callable
  `X -> Y` returning an approximation of $A_P(\sigma)^{-1} M_m X$ (masked), $A_P$ SPD.
- In `_iterate`, on a `STALL` exit with some wanted $r_i > t_i$ and `polish` given, before the
  failed-exit guard check: up to `polish_steps` steps of
  1. $Y$ = `polish(X)` on the whole block ($m$ columns); only $X$ is kept across the call (the
     state's other buffers are released first: `X = state[0]; del state`), $X$ is released
     once $Y$ exists;
  2. `_begin(Y, ..., max_refills=0)`: deflation against $R$ and the basis, SVQB,
     Rayleigh–Ritz, P reset. A needed refill or a non-finite $Y$ ends the polish: the chunk
     keeps the last state (the pre-polish one if the first step fails), `stagnated`, no raise;
  3. `_explicit` residuals and thresholds;
  4. the cluster rule (`_extent`) on the new Ritz values, gated by `near` as in the loop;
  5. accept (the chunk converges) when every $r_i \le t_i$ up to $j$.

  The polish also ends when the worst $r_i/t_i$ is not below 0.9× its best over 2 steps. No
  LOBPCG step follows a polish. If it does not accept, the failed-exit path runs as today
  (guard check with `failed=True`, explicit residuals).
- **Flags:** a polished chunk that accepts is `converged` True. `stagnated` keeps its meaning:
  True for a `STALL` exit (polished or not) and for a floor acceptance with some $r_i > $ tol.
  So a polished accepted chunk is `converged` and `stagnated`.
- **λ on a non-converged exit:** `_explicit` on the failed path also returns the Rayleigh
  quotients $x_i^\top K x_i / x_i^\top M x_i$ of the explicit products, which replace λ (a
  variational upper bound). Converged exits keep the Rayleigh–Ritz λ (unchanged).
- `info`: per chunk `polishes` (steps taken) and `polish_iterations` (the inner solver's
  iterations, max over columns, summed over steps); `info["polishes"]` the total.
- `solve_modes` passes `polish` for `precond="amg"` (shell and hex meshes), on every LOBPCG
  run except `_precond_sigma`'s 1-mode estimate: `X -> pcg(ap_operator(op, free, sigma), m M X,
  P, ptol, maxit)[0]` with LOBPCG's own `P` and σ (no second hierarchy). Band mode and the
  chunked driver get it too, so every chunk can polish. Not offered: Jacobi (its PCG needs
  thousands of iterations), GMG (no stagnating case measured; logged), `pmg` (unmeasured;
  logged).
- `solve_modes(..., polish_options=None)`: `steps` (default 5, 0 disables), `tol` (default
  $\min(10^{-7}, 0.1\,\mathrm{tol})$), `maxit` (default 1000). Validated like the other option
  dicts (unknown keys, `steps` a non-negative int, `tol` > 0, `maxit` ≥ 1 int); passed with a
  non-AMG `precond` it raises ("polish_options needs precond='amg'").
- **Memory:** with the state freed the polish peak is $X$, $b = M_m X$, PCG's blocks and the
  V-cycle temporaries at width $m$. `amg_memory` returns a `polish` term (blocks over the
  operator workspace, fitted from `memory_analysis` of `_pcg` at $m$ = 8 / 15 / 25 per family:
  shell, hex8, p); `lobpcg_memory`'s step becomes `max(step, pc.get("polish", 0))`, in both
  step modes.
- **Cost:** a polish step costs one PCG solve at width $m$ (21–221 V-cycles measured, 1 step
  in every case). Worst case `steps` × `maxit` = 5000 V-cycles, bounded by the progress stop;
  the `STALL` = 50 iterations before it stay (an earlier trigger is logged, not built).

### `amg_memory` shell family (`memory.py`)

Meshes with `ssh8` blocks (and `hex8`), at the defaults' structure:

- **storage** $= 8n + \texttt{schwarz.nbytes}(\text{level-0 patch sizes}) + \alpha_T n +
  \alpha_s n_s + \alpha_h n_h + 8\min(n_\text{coarse}, \max(\texttt{max\_coarse}, 1000))^2$,
  with $n$ the DOFs, $n_s$ the shell fiber groups, $n_h$ the nodes in no `ssh8` element, and the
  coarse term as the hex family's. The level-0 sizes come from a mesh scan (`_shell_scan(mesh)`,
  cached like `_amg_scan`): `_fibers(E, n_nodes)` and `_columns(E, group, n_nodes)` called on
  the mesh's `ssh8` connectivity (`amg` calls the same functions on the operator's tables; a
  test pins the scan's sorted patch sizes to the built hierarchy's). Constants (measured
  maximum × margin): $\alpha_T$ = 720 B (700 measured), $\alpha_s$ = 6.5 kB (5.80 kB at the
  deepest hierarchy; increments shrink, +0.29 kB per 4× last), $\alpha_h$ = 2.0 kB (1.76 kB);
  fitted per leaf group, not on totals (single-layer plates have $n_s = n/6$, so a total fit is
  collinear).
- **build** $=$ **host_build** $= \beta + \text{storage} + \gamma_s n_\text{shell} +
  \gamma_h n_\text{hex}$ (the PR 13 convention: the process peak above the operator; on a GPU
  the stored part is device memory, so this is conservative there; deferred GPU check):
  β = 50 MB (the 45k T case's fixed part), $\gamma_s$ = 2.8 kB (2.7 kB at 100k), $\gamma_h$ =
  2.9 kB per DOF (2.75 kB), with $n_\text{shell}$ / $n_\text{hex}$ the DOFs of nodes in / not in
  `ssh8` elements. The 10M extrapolation is checked against the trend from 100k to 1.58M (flat:
  falling per DOF, 4.6 → 3.3 kB).
- **step**, **step_low** $= 8n(a m + c) + 8mK$, fitted at $m$ = 8 / 15 / 25 on the flat, stack
  and block cases from 100k DOF (upper envelope); the model is stated valid from 25k DOF (the
  $K$ term carries the small sizes, checked at flat $64^2$). `AMG_LOOP_K`'s role for degree > 2
  re-measured (the study's degree-3 steps are within 1.1 blocks).
- **polish** as the step: $8n(a m + c) + 8mK$ from `_pcg` with the state freed.
- **count** $= 8n(a b + c) + 8bK$ from `memory_analysis` of `_recurrence` with the shell
  V-cycle, $b$ = 4–32.
- The hex-only path is unchanged (`AMG_STORAGE` etc.) except the new `polish` term; `amg_memory`
  picks the shell family when `ssh8` is present.

### Benchmark

`benchmarks/shell_solve_study.py` → `benchmarks/results/shell_solve_study.json` (per-step
progress printed; `--parts` selects sections). Estimated 35–45 min:
- accuracy / time (about 10 min): the cases above with the shipped code (AMG, Jacobi where it
  converges), the explicit-regime and polish cases at $L/t$ = 300 and 1000 ($32^2$, $128^2$),
  the PR 9 tolerance cases;
- calibration (about 25 min): storage per component, step / step_low / polish at
  $m$ = 8 / 15 / 25 (about 40 compiles up to 396k), count at $b$ = 4–32, host peaks in fresh
  subprocesses (up to the 1.58M plate, about 25 s each).

## Tests (18b)

Shell tests go to a new `tests/test_shell_solve.py`; LOBPCG unit tests to `tests/test_lobpcg.py`.

1. **Dispatch:**
   - `resolve_precond` gives `"amg"` on `ssh8` and `ssh8` + `hex8`, `"jacobi"` on `ssh8` +
     `hex20`;
   - `precond="amg"` there raises before any build (spy: `amg` and `lobpcg` not called);
   - `info["precond"]` and the "auto (resolved to amg)" label in option errors;
   - `solve_modes_memory` / `amg_memory` no longer raise on `ssh8`.
   - A wiring spy: `polish` reaches `lobpcg` with `"amg"` and is None with `"jacobi"`, `"gmg"`,
     `"pmg"`; `_precond_sigma`'s estimate gets none.
2. **Accuracy against `eigsh`** (the roadmap test), 20 modes, tol $10^{-10}$, `maxit` 3000,
   `precond="auto"`:
   - checks: `converged`, `info["precond"] == "amg"`, `check_lam` against `reference_modes`,
     and per eigenvalue group $\sin\theta$ ≤ $10\, r_M \lambda/\text{gap}$ (`max_sine`, `r_M`;
     the reference the sharper one), V M-orthonormal;
   - cases ($L/t$ each): flat SS $32^2$ (50); clamped $32^2$ (50); free $24^2$ (50, 6 rigid
     modes deflated); 3-layer stack $32^2$ (50 and 100); plies laminate $32^2$ (50); curved
     panel $32^2$ (50); the sweep-built cylinder (free); T-junction and L-fold $24^2$ (50);
     the hex8-web plate $32^2$ (50); a bottom-supported plate $32^2$ (50);
   - `info["polishes"]` pinned per case: ≥ 1 on free $24^2$ and the cylinder (measured: 1),
     0 elsewhere (the stacks converge by the regime alone);
   - on the free cases also $\|R^\top M V\| \le 10^{-10}$ (the leak regression).
3. **Explicit regime** (`test_lobpcg.py` and the shell file):
   - $32^2$, $L/t = 100$, tol $10^{-10}$: converges with `polishes` 0 and
     `chunks[0]["explicit"]` not None; with `EXPLICIT` monkeypatched to a huge value it stalls
     (the regime is what converges it);
   - bitwise unchanged: on the PR 9 dense test pencils without $R$ and the clamped hex20 box,
     λ, V and iteration counts equal those with `EXPLICIT` = ∞ (no regime) where `explicit`
     is None;
   - re-deflation: after a recompute step on a free pencil with an injected rigid component
     in $X$ and $P$ ($10^{-6}$), $\|R^\top M X\|$ and $\|R^\top M P\|$ ≤ $10^{-13}$, and the
     products equal explicit ones; with a locked basis the same against $B$;
   - the free $24^2$ leak (tol $10^{-10}$, `polish_options={"steps": 0}`): no near-zero λ and
     $\|R^\top M V\| \le 10^{-10}$ (fails on `main`);
   - `test_recompute_schedule` still holds (no stall there).
4. **Polish:**
   - $32^2$, $L/t = 1000$, tol $10^{-6}$: with `polish_options={"steps": 0}` `converged` False,
     `stagnated` True; with the default `converged` and `stagnated` True, `polishes` 1, the
     iteration count equal to the unpolished run's (the polish adds no LOBPCG iterations),
     `polish_iterations` within 50–300; residuals recomputed from the assembled K, M ≤ the
     thresholds; λ relative error against `eigsh` ≤ $\varepsilon\kappa$ with
     $\kappa = \|K\|/(\|M\|\lambda_1)$ from `info["norm_K"]`, `info["norm_M"]`;
   - free thin plate ($24^2$, $L/t = 1000$, rigid modes deflated): polished modes satisfy
     $\|R^\top M V\| \le 10^{-10}$ and the residual bound;
   - chunked ($n$ = 30 in chunks of 10, fp64 and fp32 basis) and band mode on the thin plate,
     one test: every chunk polishes (`chunks[i]["polishes"]` ≥ 1), no mode twice,
     $V^\top MV - I$ within the PR 9 bounds, λ against `eigsh` as above;
   - dense unit tests on a PR 9 pencil with `C_FLOOR` monkeypatched to force floor-dominated
     thresholds and a fake `polish`:
     - an exact inverse (dense solve) accepts in 1 step;
     - the identity polish ends by the progress rule after exactly 2 steps;
     - a contracting fake (ratio 0.95 per step) runs exactly `polish_steps`;
     - a fake returning a rank-deficient $Y$ ends the polish without a raise (`stagnated`);
     - a fake whose result has a cluster at the block boundary extends $j$ (cluster rule after
       the polish), and with `near` False it does not;
     - the failed-exit guard check runs only after a polish that did not accept;
     - λ on a non-converged exit equals the explicit Rayleigh quotients;
   - `maxit=1` (polish_options) still reaches PCG (spy on `pcg`, one call per step);
   - spies: `polish` is never called on solves that converge without it (hex20 box, voxel GMG,
     flat $32^2$ at $L/t = 50$, the σ estimate);
   - option errors (`steps` < 0, `tol` ≤ 0, non-int `maxit`, unknown keys, with
     `precond="jacobi"`); `precond="jacobi"` gets `info["polishes"] == 0`;
   - one `low_memory=True` polish solve ($32^2$, $L/t = 1000$) converges.
5. **`amg_memory` shell family:**
   - storage ≥ the owned device bytes on every case; per component: the level-0 Schwarz term
     equals the built one's bytes (flat, stack, T, web), $\alpha_T n$ ≥ the $T_0$ bytes and
     within 1.3× on flat $128^2$, $\alpha_s n_s$ ≥ the coarse levels and within 1.3× on flat
     $128^2$ and stack ×10, $\alpha_h n_h$ the same on block $32^3$;
   - an out-of-fit size above the largest calibration point (flat $362^2$, slow);
   - the scan's sorted level-0 patch sizes equal the built hierarchy's (flat, stack, T, web);
   - step and step_low ≥ the measured step at $m$ = 8 and 15 (flat $64^2$ and $128^2$, stack,
     T, block), within 1 block of the model on two tight cases (slow), and a degree-3 case;
   - polish ≥ the measured `_pcg` peak with the state freed at $m$ = 15 and 25 (flat $128^2$,
     hex20 $20^3$, hex8 box), within 1 block on flat $128^2$;
   - count ≥ the measured recurrence at $b$ = 4 and 16, within 1.3× at 16;
   - build ≥ the host peak and ≤ 1.3× it in fresh subprocesses (flat $256^2$, block $32^3$;
     `peak`, slow);
   - `_pc_builds` composition: `solve_modes_memory` on a shell mesh sums as on hex meshes;
   - the storage tests run on every platform (no darwin skip; the peak tests keep theirs).
6. **Band mode** on a shell: `band=(0, f)` on the flat $32^2$ plate matches `eigsh` (every mode
   in the band) with the aggregates reused (spy: one fresh `amg` setup with
   `return_aggregates`, then `aggregates=`).
7. **M4** (`examples/m4_shell.py` test): under `"auto"` now AMG, at the PR 9 tolerances.
   Existing tests that change: `test_m4_reference` and `test_m4_bands`
   (`tests/test_shell_ops.py`, `info["precond"] == "jacobi"` → `"amg"`; Jacobi stays tested by
   an explicit `precond="jacobi"` run), `test_auto_jacobi_mixed` and `test_auto_labels` (the
   `ssh8` refusal and labels), the `AMG` message constant (`tests/test_shell_ops.py:32`), the
   `amg_memory` refusal tests, and the M4 example's docstring.
8. **Iterations** (by counts, not wall time): flat $64^2$ at $L/t = 100$, 20 modes, tol
   $10^{-6}$: AMG converges in ≤ 30 iterations; Jacobi with `maxit` 500 does not converge.
9. **Benchmark smoke** (slow): `shell_solve_study --parts` at small sizes records counts,
   times and the calibration entries.

## Docs (18b)

- `docs/eigensolvers.md`:
  - :40 (the solid-shell preconditioner bullet: AMG by default; Jacobi on `ssh8` +
    `hex20`/`hex27`), :79 and :88 (the `"auto"` rules);
  - new subsections under Convergence: "Explicit-product regime" and "Polish" (mechanism:
    drift, then the κ-amplified high-frequency remainder; κ ∝ $(L/t)^4$; the measured tables;
    the defaults; inverse iteration accepts on residuals and does not recover a mode missing
    from the block, so the completeness checks still apply);
  - the `RECOMPUTE` step description (:361) and the exit check;
  - the Options table row (`polish_options`), the `info` keys (`polishes`,
    `polish_iterations`, `chunks[i]["explicit"]`, λ on a failed exit);
  - :404–415 and :419–430 (the solid-shell solve and Jacobi stagnation text), the tests list.
- `docs/preconditioners.md`: :55, :510, :565, :570, :615 (the `amg_memory` shell family with
  formulas, constants and measured tables; the "AMG on solid shells" `solve_modes` sentence;
  the polish term).
- `docs/open_items.md`: :22, :104, :210, :243, :245, :251 ("Thin plates and slender beams"
  resolved for solid shells; Jacobi stagnation; the laminate snippet); the 18b decisions.
- `docs/ROADMAP.md`: 18b status and a Contents bullet for the explicit regime and the polish,
  :184, :475, :494, :784 (M4 under AMG, the dependency table), Deferred GPU checks: the shell
  AMG host setup peak on a GPU against `host_build`, and the shell step / polish fits on a GPU.
- Code docstrings: `eigen/api.py:311` (`precond`, `polish_options`), `memory.py:338–345`
  (`amg_memory`), `precond/pmg.py:214–216` (the shell refusal's wording), the `lobpcg`
  docstring (`polish`, `polish_steps`, flags), `examples/m4_shell.py:2`.
- `docs/elements.md:218`; README status and the laminate snippet (the 4 mm plate can come back:
  AMG converges where Jacobi stagnated, open_items `:211`).

### Carried items (18a plan "Carried into 18b's plan")

- **Done here:** the dispatch, the σ pre-estimate check, band-mode reuse, the `amg_memory`
  shell family, the storage checks on every platform (not darwin-only), the build convention
  (host peak above the operator; the device inversion temporaries of `schwarz._invert` are a
  few `CHUNK_BYTES`, negligible against the step, so `pc_dev` = 0).
- **Changed:** the Jacobi stagnation test becomes the iteration-count test 8 (Jacobi
  unconverged at `maxit` 500).
- **Dropped:** the degree-3 curved fallback (degree 2 suffices on every measured shell; degree 3
  stays an option, tested by the step model only); logged in `docs/open_items.md`.

## Measured (prototypes, 18b planning)

Scripts in the session scratchpad: `s18b/` (`proto.py` dispatch patched, AMG vs Jacobi vs
`eigsh`; `thin2.py` / `floor.py` / `kappa.py` stagnation, floor, κ; `scilob.py`; `polish.py`;
`storage.py`, `step.py`, `peaks.py`), `s18b2/` (`harness.py` the patches: regime,
re-deflation, polish; `trig.py` the triggers, `trigpol.py` regime plus polish, `acc2.py` the
roadmap cases, `free.py` the free bodies), and the round-1 reviewers' folders
`plan18b-r1-*`.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **Eigensolver behaviour changes in 18b**, not 18d: the explicit regime and the polish. Without
  them the roadmap test fails at tol $10^{-10}$ from $L/t \approx 100$, and "auto" AMG stagnates
  on typical thin shells at the default tol. Solves that progress at least every 5 iterations
  are bitwise unchanged.
- **The rigid-mode leak fix** changes free-body and chunked solves at rounding level (the
  re-deflation on every recompute); logged with the before / after table.
- **Trigger choice:** 5 stuck iterations (not "near", which alters converging solves; not every
  iteration, +16% to 2× per iteration).
- **Polish scope:** `"amg"` only; GMG and p-MG unmeasured (logged).
- **Floor-limited accuracy:** at $L/t = 1000$ the λ error is about $5\cdot10^{-6}$ (residual
  floor $\varepsilon\kappa \approx 10^{-4}$); stated in the docs.
- **`ssh8` + `hex20`/`hex27`** stays Jacobi under `"auto"` (nonconforming meshes, 18a).
- **Memory constants** carry margins for levels beyond the measured sizes (10M extrapolation
  from 1.58M); the deferred GPU checks of 18a stay.
