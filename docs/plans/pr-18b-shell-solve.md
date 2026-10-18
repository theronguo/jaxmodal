# PR 18 plan: performance pass, part b: solid-shell AMG in `solve_modes`

Branch `pr-18b-shell-solve` (this plan), from `main` after PR 18a. Roadmap: PR 18b
("`"auto"` → AMG on `ssh8` meshes, the σ pre-estimate on shells, band-mode reuse, the
`amg_memory` shell family and its calibration"; test: "`solve_modes` with AMG on a solid-shell
mesh matches the PR 8 `eigsh` to PR 9 tolerances"; `docs/ROADMAP.md:184`: the shell estimator
"validated against measured peaks where the estimate is ≥ 2 GB"), the 18a plan's "Carried
into 18b's plan" list (`docs/plans/pr-18a-shell-amg.md:199–233`), and the "Thin plates and
slender beams" entry of `docs/open_items.md`.

## Revisions

- **Revision 1:** first draft, from the sizing study (scripts in the session scratchpad `s18b/`).
- **Revision 2 (plan review round 1):** stagnation diagnosed as drift in the implicit
  products; an explicit-product regime; a rigid-mode leak on free bodies found (a bug on
  `main`) and fixed by re-deflation in the recompute branch; the polish as a one-step
  finisher; PR 9 tolerance tests; per-component memory tests; docs lists.
- **Revision 3 (plan review round 2):**
  - **The regime switches the chunk to the low-memory step** instead of recomputing in the
    9-block step. W's K and M products drift too, and the 9-block step's `lax.cond` with a
    re-deflation added 1–3 blocks to its compiled peak (`test_step_memory` would fail). The low
    step is explicit in every product, its peak is below the 9-block one, and it does not
    leak. Free bodies now converge without a polish. The re-deflation in the recompute branch
    is dropped (a single deflation of X and P at the switch).
  - **Polish:** X and λ are kept until `_begin` succeeds (one fallback rule); `polish` returns
    `(Y, iterations)`; a one-step progress rule including the pre-polish ratio; `maxit`
    default 500; A_P built once; the memory term a margin over the step, not a fit.
  - **Failed exits:** an explicit Rayleigh–Ritz (λ ascending, each an upper bound for its
    index; r and t from it), not per-column Rayleigh quotients.
  - **Constants:** host build refitted as an upper envelope (β 150 MB, γ 1.4 / 2.5 kB);
    $\alpha_T$ = 880 B; tightness tests only where the constants are tight.
  - **Tests:** 7 accuracy cases; $\sin\theta$ bounded by the solver's and the reference's
    residuals (the "reference sharper" assertion fails after a polish); r_M sparse; a 791k
    `peak` test for the ≥ 2 GB rule; the existing tests that change, completed.
  - **Docs:** file:line list corrected and completed; carried items completed.

## Goal (18b)

- `solve_modes(precond="auto")` uses AMG on meshes whose blocks are all `ssh8`, or `ssh8` and
  `hex8`; `precond="amg"` is accepted there. `ssh8` with `hex20`/`hex27` keeps Jacobi under
  `"auto"`, and `precond="amg"` (also in `solve_modes_memory`) raises up front with `amg()`'s
  message.
- The roadmap test: on solid-shell meshes `solve_modes` with AMG matches the PR 8 reference
  (`reference_modes`, `eigsh` shift-invert) to the PR 9 tolerances at tol $10^{-10}$:
  `check_lam` (λ relative error ≤ $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$ for
  relative gaps ≥ $10^{-2}$, $10\,$tol cluster-wise otherwise) and principal angles per
  eigenvalue group.
- **Thin shells converge.** Today LOBPCG stagnates on solid shells: flat plates from about
  $L/t = 300$ at tol $10^{-6}$ and $L/t \approx 100$ at $10^{-10}$, 3-layer stacks from
  $L/t = 50$ at $10^{-10}$, free bodies with leaked rigid modes. 18b adds the low-memory
  switch and a polish stage that reach the floor.
- `amg_memory` covers `ssh8` meshes (storage, build, step, count), so `solve_modes_memory`, band
  mode, the slice and count estimators and the strategy work on shells.
- Band mode reuses the AMG aggregates on shells as on hex meshes (`return_aggregates`, already
  generic).

## Findings (sizing study and review rounds 1–2, CPU fp64, M3 Max)

### `solve_modes` with shell AMG works as is below $L/t \approx 200$

With the dispatch patched (prototype), 20 modes, tol $10^{-6}$, defaults:

| case | DOF | AMG: its / time | Jacobi: its / time | λ error vs `eigsh` |
|---|---|---|---|---|
| flat $32^2$, $L/t = 50$, SS | 6.5k | 16 / 3.3 s | 840 / 4.4 s | $2\cdot10^{-11}$ |
| free $24^2$, $L/t = 50$ | 3.8k | 17 / 3.0 s | 1307 / 9.0 s | (rigid modes deflated) |
| T-junction $24^2$, $L/t = 50$, clamped | 5.7k | 16 / 2.6 s | 998 / 5.0 s | $7\cdot10^{-12}$ |
| flat $64^2$, $L/t = 100$ | 25k | 17 / 4.7 s | 2392, not converged / 30 s | $5\cdot10^{-10}$ |
| curved $32^2$, $L/t = 50$ | 6.5k | 19 / 3.3 s | 1145 / 6.0 s | $1\cdot10^{-11}$ |
| stack $32^2\times3$, $L/t = 100$ | 13k | 19 / 3.8 s | 3000, not converged / 27 s | $4\cdot10^{-9}$ |
| web $32^2$ (`ssh8` + `hex8`), $L/t = 50$ | 7.1k | 19 / 3.7 s | 844 / 4.7 s | $6\cdot10^{-12}$ |

The σ pre-estimate (`_precond_sigma`) needs no change: 6–9 iterations with σ = 0 first on
supported shells, 34 on the free plate (σ = 0 refused, `sigma_estimate` first), and it lands
within 0.3% of the Kirchhoff $\lambda_1$ at $L/t = 1000$.

### Why LOBPCG stagnates on thin shells

$\kappa = \lambda_\max/\lambda_1$ grows like $(L/t)^4$ (thickness modes) and does not depend on
$h$: $4.8\cdot10^7$, $7.3\cdot10^8$, $3.7\cdot10^9$, $4.5\cdot10^{11}$ at $L/t$ = 100, 200,
300, 1000. LOBPCG's floor thresholds $t_i$ (`C_FLOOR` term) are $3\cdot10^{-4}$ (mode 1) to
$1.3\cdot10^{-6}$ (mode 20) at $L/t = 1000$; exact eigenvectors have residuals 0.06–0.2× $t_i$.
LOBPCG exits by `STALL` (50 iterations without progress) with worst $r_i/t_i$ up to
$8\cdot10^5$ although the Ritz values are within $3\cdot10^{-5}$.

1. **Drift in the implicit products.** The 9-block step updates $KX$, $MX$, $KP$, $MP$ and
   $KW$, $MW$ (through the orthogonalizations) by linear combinations; $KX$, $MX$, $KP$, $MP$
   are recomputed every `RECOMPUTE` = 20 iterations. At κ ~ $10^8$–$10^{11}$ the cancellation
   error swamps the residual. Worst $r_i/t_i$ on exit, `RECOMPUTE` patched (review round 1):

   | case | today (20) | every iteration (1) | low-memory step (all explicit) |
   |---|---|---|---|
   | $32^2$, $L/t = 1000$, tol $10^{-6}$ | 210 its, $7.9\cdot10^5$ | 134 its, 35 | 261 its, 58 |
   | $32^2$, $L/t = 300$, tol $10^{-6}$ | 190 its, $2.5\cdot10^4$ | 97 its, 6.4 | |
   | $32^2$, $L/t = 100$, tol $10^{-10}$ | 170 its, $1.7\cdot10^5$ | 35 its, converged | |

   Making $KW$, $MW$ explicit too converges the free bodies and $L/t = 300$ without a polish
   (review round 2). The low-memory step forms every product explicitly.
2. **Rigid-mode leak (free bodies, a bug on `main`).** The 9-block step deflates $W$ against
   $R$ once, before its orthogonalize-and-SVQB passes, and never re-deflates $X$ or $P$; with
   the drift, Rayleigh–Ritz pulls rigid components in (Ritz value about 0). Today, tol
   $10^{-10}$: free $24^2$ ($L/t = 50$) and the free cylinder return 6 near-zero "elastic"
   modes, $\|R^\top MV\|$ = 0.96 / 0.90, not converged. The low-memory step does not leak (also
   over 162 stalled iterations at $L/t = 300$: $3.6\cdot10^{-15}$).
3. **The κ-amplified remainder.** Even fully explicit, at $L/t \ge 1000$ the residual stays a
   few times above the floor: the iterates keep a high-frequency component that the λ-scaled
   2-norm residual amplifies by about κ. One block inverse-iteration step
   $Y = A_P(\sigma)^{-1} M X$ (PCG with the V-cycle), deflation, SVQB and Rayleigh–Ritz damps it
   by about $\lambda/\lambda_\max$. A LOBPCG restart after it brings the component back, and
   LOBPCG with the inner PCG as its preconditioner breaks down ($W \approx X$), so the polish
   is pure inverse iteration.

### Switch to the low-memory step, then polish (prototype `s18b2/harness.py`, `SWITCH=1`)

The 9-block step runs as today; after 5 iterations without progress the chunk deflates X and
P once and continues with the low-memory step; on a `STALL` exit with some wanted
$r_i > t_i$ one polish step (PCG tol $10^{-11}$ in this run), then explicit residuals.
20 modes, AMG:

| case | tol | today | switch at it | switch: its, worst $r/t$ | + polish |
|---|---|---|---|---|---|
| $32^2$, $L/t = 50$ | $10^{-10}$ | 24, converged | never | identical | none |
| $64^2$, $L/t = 100$ | $10^{-6}$ | 17, converged | never | identical | none |
| $32^2$, $L/t = 100$ | $10^{-10}$ | 170, stalled | 23 | 30, converged | none |
| $32^2$, $L/t = 300$ | $10^{-6}$ | 190, stalled | 21 | 111, converged | none |
| $32^2$, $L/t = 1000$ | $10^{-6}$ / $10^{-10}$ | 210, stalled | 27 | 109, stalled, 1.78 | 1 step (0.042), 224 PCG its, 9.2 s |
| $128^2$, $L/t = 1000$ | $10^{-6}$ | 130, stalled | 21 | 122, stalled, 0.92 (a column) | 1 step (0.043), 122 PCG its, 54 s |
| $32^2$, $L/t = 1000$, 30 modes, chunks of 10 | $10^{-6}$ | | 28 / 24 / 24 | stalled | 1 step per chunk (0.039, 0.038, 0.037) |
| free $24^2$, $L/t = 50$ | $10^{-10}$ | 130, leaked | yes | converged, `check_lam` passes (1.7e-10) | none |
| free cylinder | $10^{-10}$ | 170, leaked | yes | converged, `check_lam` passes (1.7e-8) | none |
| $32^2$, $L/t = 50$, Jacobi | $10^{-6}$ | 840, converged | 667 | 836, converged | (no polish) |

- Converging AMG and GMG solves never switch (review round 2: hex20 boxes clamped and free,
  chunked; voxel GMG clamped and free; also every hex AMG test in `test_amg`, `test_api`,
  `test_pmg`, `test_gmg`). Slow Jacobi solves can switch late (above: harmless, a little
  faster).
- The low step costs 1.4–1.9× per iteration, only after the switch.
- The polished λ error against `eigsh` at $L/t = 1000$ is $4.7\cdot10^{-6}$, inside
  `check_lam`'s floor $10\varepsilon\lambda_\max/\lambda_i$ (`eigsh` is itself floor-limited).
- Before the switch, the same polish needed 4–6 steps at $L/t = 1000$ (166 s at $128^2$).
- Today a stalled exit returns λ below the true value (956.6276 vs 956.6574, $L/t = 1000$):
  `_explicit` recomputes $r$, $t$ but not λ.

PCG tolerance (32², $L/t = 1000$, tol $10^{-6}$, round 1): $10^{-10}$, $10^{-8}$, $10^{-7}$ give
identical polish ratios at 196 / 142 / 118 PCG its, $10^{-6}$ needs one more step. At tol
$10^{-10}$ (32², $L/t = 100$, without the switch) PCG $10^{-8}$ stalls, $10^{-10}$ and
$10^{-12}$ converge. Hence $\min(10^{-7}, 0.1\,\mathrm{tol})$. One PCG iteration at width $m$
costs about one LOBPCG step (0.17 vs 0.19 s at $128^2$, $m = 25$).

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

- Level 0's Schwarz equals `schwarz.nbytes` of the level-0 patch sizes (up to 64 B), which the
  mesh alone gives (`_fibers` and `_columns` read only the `ssh8` connectivity).
- $T_0$ (rectangular block-ELL $P_0$ and $P_0^\top$): 501–700 B per fine DOF (518 / 526 / 501
  at $16^2$ / $24^2$ / $32^2$, 700 at $64^2$, 571 at $128^2$; not monotone: ELL padding).
- The coarse levels scale with fiber groups, not DOFs (the 10-layer stack equals the flat plate
  of the same in-plane size): 4.0–5.8 kB per shell fiber, growing with the level count by
  shrinking increments (+0.82, then +0.29 kB per 4× DOFs), and about 1.7 kB per hex-only node.
- The rest (free masks, Chebyshev scalars) is about 8 B per DOF.

### Step, polish, build and count

Blocks are $8nm$ bytes; "extra" is over the Jacobi step, "total" includes the step's state.

- **LOBPCG step extra** (`memory_analysis`, $m = 15$, no basis; with a basis the AMG steps are
  unchanged, review round 2): 9-block 27.4 / 16.4 / 12.5 (flat $64^2$ / $128^2$ / $256^2$),
  12.4 (stack ×3), 11.4 (stack ×10), 21.6 (T 3/3), 18.2 (block $32^3$); low 40.3 / 20.4 /
  17.5, 16.4, 15.8, 24.9, 22.1. The $n$-independent part is about $5\cdot10^5$ DOF-equivalents
  per column (the coarse levels' chunk-bounded Schwarz gathers and unpacked inverses, which
  also covers 18a's Schwarz apply temporaries $2\,\texttt{CHUNK\_BYTES}\,k/24$), so the fit
  uses the cases from 100k DOF up. Degree 3: within ±1.1 blocks of degree 2. The low total
  (11 + low extra) exceeds the 9-block total (16 + extra) on small meshes (flat $64^2$: 51.3 vs
  43.4), so a switching solve needs the larger.
- **Polish totals** (`_pcg(ap_operator, V-cycle, b)` with only X kept, review round 1):

  | case | $m$ | 9-block step total | polish total |
  |---|---|---|---|
  | flat $128^2$ (100k) | 25 / 15 | 31.0 / 34.0 | 32.3 / 36.7 |
  | flat $256^2$ (396k) | 25 | 27.4 | 27.5 |
  | hex20 $20^3$ (AMG) | 25 | 24.3 | 23.8 |
  | hex8 $40^3$ (AMG) | 25 | 26.0 | 25.2 |

  The polish is at most the 9-block total plus 2.7 blocks.
- **Host setup peak** (fresh subprocess, above the built operator, CPU: includes the device
  storage): flat 0.46 / 1.33 / 2.61 / 5.14 GB at 100k / 396k / 791k / 1.58M DOF, stack ×3
  $128^2$ 3.8 kB/DOF, stack ×10 $64^2$ 4.8 kB, T 3/3 $48^2$ 283 MB (45k DOF), block $32^3$ /
  $48^3$ 3.7 / 3.9 kB. Peak minus storage is not linear in $n$: 2.67 / 1.42 / 1.32 kB per DOF
  at 100k / 396k / 1.58M on flat plates (a fixed part of about 150 MB), 2.5 kB on the
  hex-dominant blocks.
- **Count:** not measured yet (the PR 14 recurrence with the V-cycle; `memory_analysis` as for
  `AMG_COUNT`). The chunk-bounded part needs an $n$-independent term.

## Design (18b)

### Dispatch (`eigen/api.py`)

- `resolve_precond`: `"auto"` → `"gmg"` on voxel meshes; on unstructured meshes `"amg"` unless
  the mesh has `ssh8` with `hex20`/`hex27` (`"jacobi"` there, as now). `precond="amg"` on such
  a mesh raises `amg()`'s "solid-shell AMG supports ssh8 and hex8 blocks" before any build (in
  `solve_modes` and `solve_modes_memory`).
- `no_shell` (`api.py:41`) is deleted: its callers were `resolve_precond` and `amg_memory`;
  `pmg` keeps its own refusal (`pmg.py:253`).
- The `make` / `lobpcg_precond` closures, `_precond_sigma`, band-mode aggregate reuse and the
  slices need no change: `amg` takes `nonsingular`, `return_aggregates` and `aggregates` on
  shells already (18a).

### Low-memory switch (`eigen/lobpcg.py`)

- `EXPLICIT = 5`. In `_iterate` (9-block mode), when `it - it_best >= EXPLICIT` for the first
  time in a chunk, `_to_low(state, ...)` (one jit) deflates X and P against $R$ and the locked
  basis (`_deflate`) and returns the low state `(X, P, actP, λ)`; the rest of the chunk runs
  `lobpcg_step_low` (the same `c`). Sticky: once drift is the limit, returning to the
  recurrences re-stalls (measured trigger form).
- `_iterate` returns the state in whichever form it ended; `_solve` uses only `state[0]` (X)
  and `state[-1]` (λ) after it, and the next chunk starts from `_begin` in the solve's mode.
- `lobpcg_step` and `STEP_FIT["9-block"]` are unchanged: a 9-block solve that never gets
  stuck for 5 iterations is bitwise unchanged.
- `rec["switched"]`: the iteration of the switch (None if never), in `info["chunks"]`.

### Polish: inverse iteration on a stagnated chunk (`eigen/lobpcg.py`)

- `lobpcg(..., polish=None, polish_steps=5)`, also `_solve`. `polish`: a host callable
  `X -> (Y, iterations)` with $Y \approx A_P(\sigma)^{-1} M_m X$ (masked), $A_P$ SPD.
- In `_iterate`, on a `STALL` exit with some wanted $r_i > t_i$ and `polish` given, before the
  failed-exit guard check: $w_0$ = worst $r_i/t_i$ up to $j$; up to `polish_steps` steps of
  1. keep X and λ only (the other state buffers are released); $Y$, its = `polish(X)` on the
     whole block ($m$ columns);
  2. `_begin(Y, ..., max_refills=0)` (deflation against $R$ and the basis, SVQB, Rayleigh–Ritz;
     with `max_refills=0` a needed refill returns a failure flag instead of raising);
  3. explicit residuals and thresholds (`_explicit`);
  4. if every wanted $r_i \le t_i$: the cluster rule (`_extent`) on the new Ritz values, then
     accept (the chunk converges) when every $r_i \le t_i$ up to $j$;
  5. stop if $w_s > 0.9\,w_{s-1}$ (no progress in this step).

  On a failure flag or a non-finite $Y$ the polish ends and the state is rebuilt from the kept X
  (`_explicit` products, zero P, `actP` False; low form `(X, 0, actP, λ)`). No LOBPCG step
  follows a polish. If it does not accept, the failed-exit path runs on the polished (or
  rebuilt) state: the guard check with `failed=True` and `near` from its explicit $r$, $t$,
  then the failed-exit Rayleigh–Ritz below (no second `_explicit`).
- A `maxit` exit is not polished (documented; `maxit` is the user's budget).
- **Flags:** a polished chunk that accepts is `converged` True. `stagnated` keeps its meaning:
  True for a `STALL` exit (polished or not) and for a floor acceptance with some $r_i >$ tol.
- `info`: per chunk `polishes` (steps) and `polish_iterations` (summed over steps);
  `info["polishes"]` the total.
- `solve_modes` passes `polish` for `precond="amg"` (shell and hex meshes) on every LOBPCG run
  except `_precond_sigma`'s 1-mode estimate. The closure builds $A_P$ = `ap_operator(op, free,
  σ)` once and uses LOBPCG's own `P` and σ: `X -> pcg(A_P, m M X, P, ptol, maxit)` returning
  $Y$ and the max iterations over columns. Band mode and the chunked driver get it too. Not
  offered: Jacobi (its PCG needs thousands of iterations), GMG and p-MG (no stagnating case
  measured; logged). `steps == 0` passes `polish=None`.
- `solve_modes(..., polish_options=None)`: `steps` (default 5), `tol` (default
  $\min(10^{-7}, 0.1\,\mathrm{tol})$), `maxit` (default 500). Validated like the other option
  dicts (unknown keys, `steps` a non-negative int, `tol` > 0, `maxit` ≥ 1 int); passed with a
  non-AMG `precond` (also "auto" resolved to jacobi/gmg) it raises ("polish_options needs
  precond='amg'", with the resolved label).
- **Cost:** a polish step is one PCG solve at width $m$ (21–224 V-cycles measured, each about
  one LOBPCG step); the worst case `steps` × `maxit` = 2500 V-cycles. The `STALL` = 50
  iterations before it stay (an earlier trigger is logged, not built).

### Failed-exit Rayleigh–Ritz (`eigen/lobpcg.py`)

- On every non-converged exit, `_explicit` is replaced by `_final_rr`: eigh of the explicit
  $X^\top K X$, $X^\top M X$, X rotated, λ the Ritz values (ascending; by Courant–Fischer each
  an upper bound of its eigenvalue), $r$ and $t$ from them. Converged exits keep today's path.
  A `fix:` commit of its own (all solvers, hex too).

### Memory (`memory.py`)

- `lobpcg_memory` (9-block mode): step = max(9-block step, low step, polish), with low step the
  `low_memory` formula (the switch) and polish = 9-block step + 3 blocks when the
  preconditioner estimate has `polish=True` (AMG). `low_memory=True`: max(low step, polish).
  For Jacobi and GMG the low total is below the 9-block total at every measured size, so their
  estimates do not change except where it is larger.

### `amg_memory` shell family (`memory.py`)

Meshes with `ssh8` blocks (and `hex8`), at the defaults' structure:

- **storage** $= \texttt{schwarz.nbytes}(\text{level-0 patch sizes}) + \alpha_T n +
  \alpha_s n_s + \alpha_h n_h + 8\min(n_\text{coarse}, \max(\texttt{max\_coarse}, 1000))^2$,
  with $n$ the DOFs, $n_s$ the shell fiber groups, $n_h$ the nodes in no `ssh8` element, and the
  coarse term as the hex family's. The level-0 sizes come from a mesh scan (`_shell_scan(mesh)`,
  cached like `_amg_scan`): `_fibers(E, n_nodes)` and `_columns(E, group, n_nodes)` on the
  mesh's `ssh8` connectivity (`amg` calls the same functions on the operator's tables).
  Constants: $\alpha_T$ = 880 B (700 measured max, non-monotone, 1.25×; includes the 8 B/DOF
  rest), $\alpha_s$ = 6.5 kB (5.80 kB at the deepest hierarchy; increments shrink),
  $\alpha_h$ = 2.0 kB (1.76 kB); fitted per leaf group, not on totals (single-layer plates have
  $n_s = n/6$, so a total fit is collinear). The calibration adds the sweep-built cylinder and
  the L-fold (non-box meshes).
- **build** $=$ **host_build** $= \beta + \text{storage} + \gamma_s n_\text{shell} +
  \gamma_h n_\text{hex}$ ($n_\text{shell}$ / $n_\text{hex}$: DOFs of nodes in / not in `ssh8`
  elements), the PR 13 convention (process peak above the operator; on a GPU the stored part is
  device memory, conservative there; `pc_dev` = 0: the device inversion temporaries are a few
  `CHUNK_BYTES`). β, γ fitted as the upper envelope (LP, as `AMG_STEP`) of peak minus measured
  storage over every case from 45k to 1.58M: about β = 150 MB, $\gamma_s$ = 1.4 kB,
  $\gamma_h$ = 2.5 kB.
- **step**, **step_low** $= 8n(a m + c) + 8mK$, fitted at $m$ = 8 / 15 / 25 on the flat, stack
  and block cases from 100k DOF (upper envelope), stated valid from 25k DOF (checked at flat
  $64^2$); `AMG_LOOP_K` re-measured for degree > 2.
- **count** $= 8n(a b + c) + 8bK$ from `_recurrence` with the shell V-cycle, $b$ = 4–32.
- `polish=True` in both families' estimates (the hex family is otherwise unchanged);
  `amg_memory` picks the shell family when `ssh8` is present.

### Benchmark

`benchmarks/shell_solve_study.py` → `benchmarks/results/shell_solve_study.json` (per-step
progress printed; `--parts` selects sections). Estimated 35–45 min:
- accuracy / time (about 10 min): the tables above with the shipped code (AMG; Jacobi where it
  converges), the switch and polish cases at $L/t$ = 300 and 1000 ($32^2$, $128^2$), the PR 9
  tolerance cases;
- calibration (about 25 min): storage per component (with the cylinder and L-fold), step /
  step_low at $m$ = 8 / 15 / 25, the polish peak at $m$ = 15 / 25 (model check), count at
  $b$ = 4–32, host peaks in fresh subprocesses up to the 1.58M plate (the ≥ 2 GB check of
  `ROADMAP.md:184`: model / measured at 791k and 1.58M recorded).

## Tests (18b)

Shell tests go to a new `tests/test_shell_solve.py`, LOBPCG unit tests to
`tests/test_lobpcg.py`. Slow marks for anything above about 5 s.

1. **Dispatch:**
   - `resolve_precond` gives `"amg"` on `ssh8` and `ssh8` + `hex8`, `"jacobi"` on `ssh8` +
     `hex20`; `precond="amg"` there raises before any build in `solve_modes` and
     `solve_modes_memory` (spy: `amg` and `lobpcg` not called);
   - `info["precond"]`, the "auto (resolved to amg)" label in option errors;
   - a wiring spy: `polish` reaches `lobpcg` with `"amg"` and is None with `"jacobi"`,
     `"gmg"`, `"pmg"` and for `_precond_sigma`'s estimate.
2. **Accuracy against `reference_modes`** (the roadmap test), 20 modes, tol $10^{-10}$,
   `maxit` 3000, `precond="auto"`, on flat SS $32^2$, free $24^2$ (rigid deflated), 3-layer
   stack $32^2$ ($L/t$ 100), curved panel $32^2$, the free cylinder, T-junction $24^2$, the
   hex8-web plate $32^2$ ($L/t$ 50 unless stated):
   - `converged`, `info["precond"] == "amg"`, `check_lam`;
   - per eigenvalue group $\sin\theta \le 10\,(r_M + r_{M,\text{ref}})\lambda/\text{gap}$
     (`max_sine`), and $r_{M,\text{ref}} \le 10^{-9}$; `r_M` and `max_sine` sparse (`splu` of
     $M_f$; $\|L^\top E\|_2 = \sqrt{\lambda_\max(E^\top M E)}$), no dense Cholesky;
   - V M-orthonormal; on the free cases $\|R^\top MV\| \le 10^{-10}$;
   - `info["polishes"]` 0 on all seven (measured with the switch; pinned from the
     implementation run), `chunks[0]["switched"]` not None on the stack and the free cases.
3. **Low-memory switch:**
   - $32^2$, $L/t = 100$, tol $10^{-10}$: converges, `polishes` 0, `switched` not None; with
     `EXPLICIT` monkeypatched to a huge value and `polish_options={"steps": 0}` it stalls;
   - the free $24^2$ leak (tol $10^{-10}$, steps 0): no λ below $10^{-3}\lambda_1$, and
     $\|R^\top MV\| \le 10^{-10}$ (fails on `main`);
   - unchanged: `test_gap_band`'s pencil and the clamped hex20 box at tol $10^{-8}$ assert
     `switched is None` (the existing tests then pin their results);
   - `_to_low` unit test: an injected rigid / basis component ($10^{-6}$) in X and P is gone
     ($\le 10^{-13}$) and the low step continues from it.
4. **Polish:**
   - $32^2$, $L/t = 1000$, tol $10^{-6}$: with steps 0 `converged` False, `stagnated` True;
     with the default `converged` and `stagnated` True, `polishes` 1, the iteration count equal
     to the unpolished run's, `polish_iterations` in 50–400; residuals recomputed from the
     assembled K, M ≤ the thresholds; `check_lam` against `eigsh`;
   - free thin plate ($24^2$, $L/t = 1000$, tol $10^{-6}$): `check_lam`,
     $\|R^\top M V\| \le 10^{-10}$;
   - chunked ($n$ = 30, chunks of 10, fp64 and fp32 basis) and band mode with `chunk=10` on the
     thin plate, one test: every chunk polishes, no mode twice, $V^\top MV - I$ within the PR 9
     bounds, `check_lam` (band measured before pinning);
   - one `low_memory=True` polish solve ($32^2$, $L/t = 1000$): converges, `polishes` 1;
   - dense unit tests on a PR 9 pencil with `C_FLOOR` monkeypatched (floor-dominated
     thresholds) and fake polishes:
     - an exact inverse (dense solve) accepts in 1 step;
     - the identity polish stops after exactly 1 step ($w_1 = w_0$);
     - a fake contracting by 0.5 per step from far above acceptance runs exactly
       `polish_steps`;
     - a rank-deficient $Y$ ends the polish without a raise, `stagnated`, λ and V equal to the
       steps-0 run;
     - a polished result with a cluster at the block boundary extends $j$;
     - the failed-exit guard check runs only after a polish that did not accept;
   - spies: `polish` never called on a converging AMG solve (flat $32^2$, $L/t = 50$) and the
     σ estimate; `pcg` called once per step with `maxit` from the options;
   - option errors (`steps` < 0, `tol` ≤ 0, non-int `maxit`, unknown keys, with
     `precond="jacobi"`, with "auto" on `ssh8` + `hex20` (label "auto (resolved to jacobi)")).
5. **Failed-exit Rayleigh–Ritz:**
   - on the stalled thin plate with steps 0: λ ascending, $\sum\lambda \ge
     \sum\lambda_\text{eigsh}(1 - 10^{-12})$ (Ky Fan; fails on `main`), residuals equal to ones
     recomputed with the returned λ;
   - `test_stagnation`'s maxit-exit residual check still holds (touched: it now checks against
     the Ritz λ).
6. **Memory:**
   - `lobpcg_memory`: step = max(9-block, low, polish) in both modes; no polish term for
     Jacobi, GMG, p-MG; one case where the low term wins (flat $64^2$, AMG) and one where the
     polish wins (flat $128^2$, $m = 15$);
   - the polish model ≥ the measured `_pcg` peak with X kept (flat $128^2$ at $m$ = 15 / 25, a
     hex20 AMG case);
   - `amg_memory` shell storage ≥ the owned device bytes on every case (flat, stack ×3 and ×10,
     T, web, block, cylinder, L-fold, a $48^2$ non-power-of-two plate); the level-0 Schwarz term
     equal to the built one's bytes within 64 B; the scan's sorted level-0 patch sizes equal
     the built hierarchy's; total within 1.3× on flat $128^2$ and the block $32^3$ (slow); an
     out-of-fit size (flat $362^2$, slow);
   - step and step_low ≥ the measured step at $m$ = 8 and 15 (flat $64^2$ and $128^2$, stack,
     T, block) and with a basis present, within 1 block of the model on two tight cases (slow),
     a degree-3 case;
   - count ≥ the measured recurrence at $b$ = 4 and 16 on flat $128^2$, within 1.3× at 16;
   - build ≥ the host peak in fresh subprocesses on flat $128^2$, T 3/3 and block $32^3$, ≤ 1.3×
     on flat $256^2$ (`peak`, slow); flat $362^2$ (791k DOF, estimate ≥ 2 GB, `peak`, slow):
     ≥ the peak, ≤ 1.3× (the `ROADMAP.md:184` rule);
   - `_pc_builds` composition: `solve_modes_memory` on a shell mesh sums as on hex meshes;
   - storage tests on every platform (no darwin skip; the peak tests keep theirs).
7. **Band mode** on a shell: `band=(0, f)` on the flat $32^2$ plate matches the reference
   (every mode in the band) with the aggregates reused (spy: one fresh `amg` setup with
   `return_aggregates`, then `aggregates=`).
8. **M4 and existing tests that change:**
   - `examples/m4_shell.py` and its test: under `"auto"` now AMG;
   - `test_m4_reference` (`tests/test_shell_ops.py:169`): AMG; the "reference sharper"
     assertion replaced as in test 2 (it fails under AMG and, with the switch, under Jacobi);
     the explicit `precond="jacobi"` run keeps Jacobi tested;
   - `test_m4_bands`: AMG; its shift-invert slices and count now use shell AMG;
   - `test_auto_jacobi_mixed`, `test_auto_labels` (under AMG `precond_options={"degree": 3}`
     and `sigma_options={"tol": 1e-3}` become valid: new unknown keys), `test_refusals` (three
     parts), `test_solve_modes_memory`, the `AMG` message constant (`test_shell_ops.py:32`), the
     `amg_memory` refusal tests, `test_stagnation` (test 5);
   - `test_step_memory`, `test_estimate_chunks` and the `STEP_FIT` pins: unchanged by design
     (the 9-block step is untouched), run as the check.
9. **Iterations** (by counts, not wall time): flat $64^2$ at $L/t = 100$, 20 modes, tol
   $10^{-6}$: AMG converges in ≤ 30 iterations; Jacobi with `maxit` 500 does not converge.
10. **Benchmark smoke** (slow): `shell_solve_study --parts` at small sizes records counts, times
    and the calibration entries.

## Docs (18b)

- `docs/eigensolvers.md`:
  - `:41` (the solid-shell bullet: Jacobi stagnation text and M4 timings, rewritten for AMG;
    `:40` names `ssh8` among the remedies), `:79` and `:88` (the `"auto"` rules; `:88` "since
    PR 18b");
  - `:313` (`lobpcg` signature: `polish`, `polish_steps`), `:315` (`info` / `chunks` keys:
    `polishes`, `polish_iterations`, `switched`; λ, r, t on a failed exit from the final
    Rayleigh–Ritz), `:316` (deflation: plus the one deflation at the switch);
  - `:361` (RECOMPUTE step 7: the switch to the low step after 5 stuck iterations);
  - new subsections under Convergence: "Low-memory switch" and "Polish" (the three causes, κ ∝
    $(L/t)^4$, the measured tables, the defaults; inverse iteration accepts on residuals and
    does not recover a mode missing from the block, so the completeness checks still apply; a
    `maxit` exit is not polished);
  - `:404–415` (Memory model: step = max(9-block, low, polish)), `:419–430` (Convergence exit
    table: the polished row, `converged` and `stagnated`);
  - the Options table row (`polish_options`), the tests list.
- `docs/preconditioners.md`: `:55`, `:474` ("still refuse AMG on `ssh8` until PR 18b"),
  `:510`, `:565`, `:570`, `:615` (the `test_refusals` sentence): the `amg_memory` shell family
  (formulas, constants, measured tables, the polish flag), the "AMG on solid shells"
  `solve_modes` sentence, the degree-3 note (`precond_options={"degree": 3}` for curved panels,
  as 18a promised, `open_items.md:251`).
- `docs/open_items.md`: `:16` (heading "(PR 11a → PR 18b)"), `:18` ("18b resolves the entry for
  solid shells": resolved), `:22`, `:104` (counting on `ssh8`: test 7's band), `:210`, `:211`
  (the 4 mm laminate), `:243`, `:245`, `:251`; the 18b decisions (below).
- `docs/ROADMAP.md`: `:18` (status line), the PR 18b section (status, a Contents bullet for the
  switch, the polish and the failed-exit Rayleigh–Ritz), `:184` (the ≥ 2 GB validation: the
  791k test and the benchmark), `:475`, `:494`, `:580` and `:590` (M4 under AMG), `:784` (the 18a
  host-peak check: CPU calibration done, the GPU part kept as one entry); Deferred GPU checks:
  the shell step / polish fits on a GPU.
- Code docstrings: `eigen/api.py:312–313` (`precond`, `polish_options`), `memory.py:338–345`
  (`amg_memory`), the `lobpcg` docstring (`polish`, `polish_steps`, flags), `examples/m4_shell.py:2`
  (and `info['precond']` at `:27` now amg).
- `docs/elements.md:218`; `README.md:5` (status) and `:82` ("`solve_modes` (Jacobi-preconditioned
  …"), the laminate snippet (the 4 mm plate can come back); optionally
  `docs/verification.md:239` (AMG).

### Carried items (18a plan `:199–233`)

- **Done here:** the dispatch, the σ pre-estimate check, band-mode reuse, the `amg_memory`
  shell family, the storage checks on every platform.
- **Changed:**
  - the storage form: $\alpha_T$ over all DOFs instead of $\alpha_P n_{c,\text{shell}}$; the
    step's level-0 chunk temporaries fitted ($8n(am+c) + 8mK$) instead of exact from
    `schwarz.nbytes` (the $K$ term covers the Schwarz apply temporaries); the hex-constant
    extrapolation decision kept for the hex family only;
  - the build convention: `pc_dev` = 0 instead of the device inversion temporaries (about
    7 × `CHUNK_BYTES`, negligible against the step), logged;
  - "hex-dominant meshes upper bound only": the block $32^3$ total gets a 1.3× check (its fit
    is tight at 1.24×);
  - the Jacobi stagnation test: Jacobi converges at $L/t = 50$ (840–1307 iterations), so the
    test moves to flat $64^2$ at $L/t = 100$ and counts iterations (test 9);
  - the Jacobi wall-time comparison (`serial`): replaced by the iteration counts of test 9
    (wall time on a loaded machine is the flaky part).
- **Dropped:** the degree-3 curved fallback as a default (degree 2 suffices on every measured
  shell; degree 3 stays an option, documented and tested by the step model).

## Measured (prototypes, 18b planning)

Scripts in the session scratchpad: `s18b/` (`proto.py`, `thin2.py`, `floor.py`, `kappa.py`,
`scilob.py`, `polish.py`, `storage.py`, `step.py`, `peaks.py`), `s18b2/` (`harness.py` the
patches: `SWITCH`, `TRIG`, `DEFL`, `POLISH`; `plate.py`, `free2.py`, `acc2.py`; logs `run3.log`,
`run4.log`), and the reviewers' folders `plan18b-r1-*`, `plan18b-r2-*`.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **Eigensolver behaviour changes in 18b**, not 18d: the low-memory switch, the polish and the
  failed-exit Rayleigh–Ritz. Without them the roadmap test fails at tol $10^{-10}$ and "auto"
  AMG stagnates on typical thin shells. Unchanged: 9-block solves that never get stuck for 5
  iterations (every measured converging AMG / GMG solve). Changed: slow Jacobi solves that get
  stuck (late switch, measured harmless), every non-converged exit (Ritz λ).
- **The rigid-mode leak on `main`** (free thin shells return rigid modes as elastic): fixed by
  the switch (the low step does not leak; one deflation at the switch). Logged with the before
  / after table; a `fix:` commit of its own.
- **Switch, not recompute:** recomputing in the 9-block step left W's products implicit and
  raised the step's compiled peak; the low step costs 1.4–1.9× per iteration after the switch.
- **Polish scope:** `"amg"` only (shell and hex); GMG and p-MG unmeasured. Pure inverse
  iteration (a LOBPCG restart and LOBPCG with an inner-PCG preconditioner rejected).
- **Flags:** a polished accepted chunk is `converged` and `stagnated`.
- **`STALL` = 50 kept**; an earlier trigger logged, not built.
- **Floor-limited accuracy:** at $L/t = 1000$ the λ error is about $5\cdot10^{-6}$ (residual
  floor $\varepsilon\kappa \approx 10^{-4}$); stated in the docs.
- **`ssh8` + `hex20`/`hex27`** stays Jacobi under `"auto"` (nonconforming meshes, 18a).
- **`no_shell` removed.**
- **Memory constants** carry margins for levels beyond the measured sizes (10M extrapolation
  from 1.58M); the count constants are measured in implementation; the deferred GPU checks of
  18a stay.
