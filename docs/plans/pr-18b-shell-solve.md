# PR 18 plan: performance pass, part b: solid-shell AMG in `solve_modes`

Branch `pr-18b-shell-solve` (this plan), from `main` after PR 18a. Roadmap: PR 18b
("`"auto"` → AMG on `ssh8` meshes, the σ pre-estimate on shells, band-mode reuse, the
`amg_memory` shell family and its calibration"; test: "`solve_modes` with AMG on a solid-shell
mesh matches the PR 8 `eigsh` to PR 9 tolerances"), the 18a plan's "Carried into 18b's plan"
list, and the "Thin plates and slender beams" entry of `docs/open_items.md`.

## Revisions

- **Revision 1:** first draft, from the sizing study below (scripts in the session scratchpad
  `s18b/`; the numbers are re-measured by the benchmark of this PR).

## Goal (18b)

- `solve_modes(precond="auto")` uses AMG on meshes whose blocks are all `ssh8`, or `ssh8` and
  `hex8`; `precond="amg"` is accepted there. `ssh8` with `hex20`/`hex27` keeps Jacobi under
  `"auto"`, and `precond="amg"` raises up front with `amg()`'s message.
- The roadmap test: on solid-shell meshes `solve_modes` with AMG matches the PR 8 `eigsh` to
  the PR 9 tolerances (tol $10^{-10}$, λ relative error ≤ $10^{-8}$, MAC ≥ $1 - 10^{-8}$).
- **Thin shells converge.** LOBPCG stagnates far above its round-off floor on solid shells from
  about $L/t = 300$ at tol $10^{-6}$ (and from $L/t \approx 100$ at $10^{-10}$), with AMG and
  with Jacobi alike (below). 18b adds a polish stage that reaches the floor.
- `amg_memory` covers `ssh8` meshes (storage, build, step, count), so `solve_modes_memory`,
  band mode, the slice and count estimators and the strategy work on shells.
- Band mode reuses the AMG aggregates on shells as on hex meshes (`return_aggregates`, already
  generic).

## Findings (sizing study, CPU fp64, M3 Max)

### `solve_modes` with shell AMG works as is below $L/t \approx 200$

With the dispatch patched (prototype), 20 modes, tol $10^{-6}$, defaults:

| case | DOF | AMG: its / time | Jacobi: its / time | λ error vs `eigsh` |
|---|---|---|---|---|
| flat $32^2$, $L/t = 50$, SS | 6.5k | 16 / 3.3 s | 840 / 4.4 s | $2\cdot10^{-11}$ |
| free $24^2$, $L/t = 50$ | 3.8k | 17 / 3.0 s | 1307 / 9.0 s | (rigid filter) |
| T-junction $24^2$, clamped | 5.7k | 16 / 2.6 s | 998 / 5.0 s | $7\cdot10^{-12}$ |
| flat $64^2$, $L/t = 100$ | 25k | 17 / 4.7 s | 2392, not converged / 30 s | $5\cdot10^{-10}$ |
| curved $32^2$, $L/t = 50$ | 6.5k | 19 / 3.3 s | 1145 / 6.0 s | $1\cdot10^{-11}$ |
| stack $32^2\times3$, $L/t = 100$ | 13k | 19 / 3.8 s | 3000, not converged / 27 s | $4\cdot10^{-9}$ |
| web $32^2$ (`ssh8` + `hex8`) | 7.1k | 19 / 3.7 s | 844 / 4.7 s | $6\cdot10^{-12}$ |

The σ pre-estimate (`_precond_sigma`) needs no change: 6–9 iterations with σ = 0 first on
supported shells, 34 on the free plate (σ = 0 refused, `sigma_estimate` first), and it lands
within 0.3% of the Kirchhoff $\lambda_1$ at $L/t = 1000$.

### LOBPCG stagnates on thin shells, with any preconditioner

Flat SS $32^2$, 20 modes, tol $10^{-6}$, AMG: converged in 16 / 19 iterations at
$L/t$ = 100 / 200, stagnated (`STALL` exit) at $L/t \ge 300$ with max wanted residuals 0.07
($L/t = 300$), $9\cdot10^{-3}$ (500), 208 (1000), although the Ritz values are within
$3\cdot10^{-5}$. Jacobi stagnates the same way ($L/t = 300$: residuals $10^{-4}$–$10^{-2}$
after 18,030 iterations). $128^2$ at $L/t = 1000$: 130 iterations, stagnated.

- **Not the floor.** $\kappa = \lambda_\max/\lambda_1$ grows like $(L/t)^4$ (thickness modes)
  and does not depend on $h$: $4.8\cdot10^7$, $7.3\cdot10^8$, $3.7\cdot10^9$,
  $4.5\cdot10^{11}$ at $L/t$ = 100, 200, 300, 1000. LOBPCG's floor thresholds $t_i$
  (`C_FLOOR` term) are $3\cdot10^{-4}$ (mode 1) to $1.3\cdot10^{-6}$ (mode 20) at
  $L/t = 1000$, and exact eigenvectors (`eigsh` shift-invert) have residuals 0.06–0.2× $t_i$.
  The stagnation levels follow about $\varepsilon\kappa^{3/2}$ instead (0.05 at 300, 70 at
  1000): the iterates carry high-frequency noise that the λ-scaled 2-norm residual amplifies
  by κ.
- **The preconditioner's high end decides it.** SciPy's `lobpcg` with an exact (LU)
  preconditioner reaches $10^{-6}$–$10^{-4}$ at $L/t = 1000$ (near the floor); with the
  V-cycle or Jacobi the noise stays.
- **Inverse iteration removes the noise.** One block step $Y = A_P(\sigma)^{-1} M X$ (PCG with
  the AMG V-cycle at $10^{-10}$–$10^{-12}$), deflation, SVQB and Rayleigh–Ritz damps the high
  end by about $\lambda/\lambda_\max$. Restarting LOBPCG after it brings the noise back (0.13
  after one step at $L/t = 1000$), and LOBPCG with the inner PCG solve as its preconditioner
  breaks down (W ≈ X: residuals grow to $10^3$), so the polish stays pure inverse iteration.

Measured polish (prototype: on a `STALL` exit, repeat inverse-iteration steps with explicit
residuals after each, accept when every wanted $r_i \le t_i$; $m = 25$, PCG maxit 2000):

| case | tol | LOBPCG | polish steps (worst $r_i/t_i$ after each) | PCG its per step | total |
|---|---|---|---|---|---|
| $32^2$, $L/t = 300$ | $10^{-6}$ | 190, stagnated | 1 (0.63) | 85 | 7.5 s |
| $128^2$, $L/t = 300$ | $10^{-6}$ | 170, stagnated | 1 (0.05) | 74 | 52 s |
| $32^2$, $L/t = 1000$ | $10^{-6}$ | 210, stagnated | 4 (11, 3.3, 1.7, 0.99) | 196 (PCG $10^{-10}$; 170 at $10^{-9}$, 247 at $10^{-12}$) | 16 s |
| $128^2$, $L/t = 1000$ | $10^{-6}$ | 130, stagnated | 6 (56, 14, 5.6, 2.6, 1.5, 0.86) | 135 | 166 s |
| $32^2$, $L/t = 50$ | $10^{-10}$ | 24, converged | none needed | – | 4.2 s |
| $32^2$, $L/t = 100$ | $10^{-10}$ | 170, stagnated | 8 (300 … 3.1, about 0.6× per step) | 35 | 10 s |
| $32^2$, $L/t = 1000$ | $10^{-10}$ | 210, stagnated | 4 (floor-accepted) | 196 | 17 s |

The slowest columns are the top wanted modes: per step their error falls by about
$(\lambda_\text{want} + \sigma)/(\lambda_{m+1} + \sigma)$. The worst ratio falls
monotonically, so a step cap with a progress stop bounds the cost. The polished λ error
against `eigsh` is $4.5\cdot10^{-6}$ at $L/t = 1000$ (floor-limited:
$\varepsilon\kappa \approx 10^{-4}$), $7\cdot10^{-10}$ at $L/t = 100$.

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
- $T_0$ (rectangular block-ELL $P_0$ and $P_0^\top$): 529–700 B per fine DOF, falling with
  size.
- The coarse levels scale with fiber groups, not DOFs (the 10-layer stack equals the flat
  plate of the same in-plane size): 4.0–5.8 kB per shell fiber, growing slowly with the level
  count, and about 1.7 kB per hex-only node (hex groups are single nodes).
- The rest (free masks, Chebyshev scalars) is about 8 B per DOF.

### Step, build and count

- **LOBPCG step** (`memory_analysis`, $m = 15$, extra over the Jacobi step, in blocks
  $8nm$): 9-block 27.4 / 16.4 / 12.5 (flat $64^2$ / $128^2$ / $256^2$), 12.4 (stack ×3),
  11.4 (stack ×10), 21.6 (T 3/3), 18.2 (block $32^3$); low-memory 40.3 / 20.4 / 17.5, 16.4,
  15.8, 24.9, 22.1. On flat plates the extra is about $11.2 + 5.3\cdot10^5/n$ blocks: an
  $n$-independent part of about $5\cdot10^5$ DOF-equivalents per column (the coarse levels'
  chunk-bounded Schwarz gathers and the unpacked inverses, about $2\,\texttt{CHUNK\_BYTES}/24$
  per column plus one chunk) on top of a per-DOF part. Degree 3: within ±1.1 blocks of
  degree 2.
- **Host setup peak** (fresh subprocess, above the built operator, CPU: includes the device
  storage): flat 0.46 / 1.33 / 2.61 / 5.14 GB at 100k / 396k / 791k / 1.58M DOF (4.6 / 3.4 /
  3.3 / 3.3 kB per DOF), stack ×3 $128^2$ 3.8 kB, stack ×10 $64^2$ 4.8 kB, T 3/3 $48^2$
  6.3 kB (45k DOF, the fixed part dominates), block $32^3$ / $48^3$ 3.7 / 3.9 kB. Beyond the
  stored hierarchy: 1.3–2.0 kB per DOF on shells, 2.5–2.75 kB on the hex-dominant blocks.
- **Count:** not measured yet (the PR 14 recurrence with the V-cycle; the same
  `memory_analysis` method as `AMG_COUNT`).

## Design (18b)

### Dispatch (`eigen/api.py`)

- `resolve_precond`: `"auto"` → `"gmg"` on voxel meshes; on unstructured meshes `"amg"` unless
  the mesh has `ssh8` with `hex20`/`hex27` (`"jacobi"` there, as now). `precond="amg"` on such
  a mesh raises `amg()`'s "solid-shell AMG supports ssh8 and hex8 blocks" up front.
- `no_shell` goes (its users: `resolve_precond`, `amg_memory`); `pmg` keeps its own refusal.
- The `make` / `lobpcg_precond` closures, `_precond_sigma`, band-mode aggregate reuse and the
  slices need no change: `amg` takes `nonsingular`, `return_aggregates` and `aggregates` on
  shells already (18a).

### Polish: inverse iteration on a stagnated chunk (`eigen/lobpcg.py`)

- `lobpcg(..., polish=None, polish_steps=20)`, also `_solve`. `polish`: a host callable
  `X -> Y` returning an approximation of $A_P(\sigma)^{-1} M_m X$ (masked), with $A_P$ SPD.
- In `_iterate`, on a `STALL` exit with some wanted $r_i > t_i$ and `polish` given: up to
  `polish_steps` steps of
  1. $Y$ = `polish(X)` on the whole block ($m$ columns),
  2. `_begin(Y, ...)`: deflation against $R$ and the basis, SVQB (with its refill rule),
     Rayleigh–Ritz, P reset;
  3. `_explicit` residuals and thresholds;
  4. the cluster rule (`_extent`, unchanged) on the new Ritz values;
  5. accept (the chunk converges, `stagnated` as for any floor acceptance) when every
     $r_i \le t_i$ up to $j$;

  stopping early when the worst $r_i/t_i$ does not fall below 0.9× its best over 2 steps. No
  LOBPCG step follows a polish (it brings the noise back).
- `info`: per chunk `polishes` (steps taken) and `polish_iterations` (the inner solver's
  iterations, max over columns, summed over steps); `info["polishes"]` the total.
- `solve_modes` passes `polish` for `"amg"` and fp64 `"gmg"` (not Jacobi: its PCG would need
  thousands of iterations; not the fp32 V-cycle: PCG at $10^{-10}$ with an fp32
  preconditioner is not reliable), on every LOBPCG run except `_precond_sigma`'s 1-mode
  estimate: `X -> pcg(ap_operator(op, free, sigma), m M X, P, tol, maxit)[0]` with LOBPCG's
  own `P` and σ (no second hierarchy). Band mode and the chunked driver get it too, so every
  chunk can polish.
- `solve_modes(..., polish_options=None)`: keys `steps` (default 20, 0 disables), `tol`
  (default $10^{-10}$), `maxit` (default 1000); listed in the Options table
 ; errors as the other option dicts.
- **Memory:** the polish runs between steps: the 9-block state (or the low state) stays, plus
  PCG's blocks ($x$, $r$, $z$, $p$, $Ap$, $b$) and the V-cycle temporaries at width $m$.
  Measured by `memory_analysis` of `_pcg` at width $m$ in implementation; `lobpcg_memory`'s
  step term becomes `max(step, polish)` with `polish = state + 6 b + pc["count"]`-style
  V-cycle term if that exceeds the step anywhere (expected not: 9 + 6 < 16 + the step's
  workspace terms).
- **Scope:** only on a stagnation exit, so solves that converge today are unchanged (the
  existing tests pin this: no test of `solve_modes` with GMG/AMG expects stagnation). Thin hex
  plates, which stagnate with Jacobi (`docs/eigensolvers.md:512`), gain it under AMG too;
  measured in the tests.

### `amg_memory` shell family (`memory.py`)

Meshes with `ssh8` blocks (and `hex8`), at the defaults' structure:

- **storage** $= 8n + \texttt{schwarz.nbytes}(\text{level-0 patch sizes}) + \alpha_T n +
  \alpha_s n_s + \alpha_h n_h + 8\min(n_\text{coarse}, \max(\texttt{max\_coarse}, 1000))^2$,
  with $n$ the DOFs, $n_s$ the shell fiber groups, $n_h$ the nodes in no `ssh8` element, and the
  coarse term as the hex family's. The level-0 sizes come from a mesh scan
  (`_shell_scan(mesh)`, cached like `_amg_scan`): `_fibers` and `_columns` on the mesh's `ssh8`
  connectivity (both get a mesh-level entry point; `amg` keeps calling them on the operator's
  DOF tables, and a test pins both to the same groups and patches). Constants from the study:
  $\alpha_T$ about 720 (the largest measured, falling with size), $\alpha_s$ about 6.5 kB
  (5.8 kB measured at the deepest hierarchy plus about one more level's 1.2×), $\alpha_h$ about
  2.4 kB (1.76 kB measured; the same level growth); fitted per leaf group, not on totals (on
  single-layer plates $n_s = n/6$, so a total fit is collinear).
- **build** $=$ **host_build** $= \beta + \text{storage} + \gamma_s n_\text{shell} +
  \gamma_h n_\text{hex}$ (the PR 13 convention: the process peak above the operator; on a GPU
  the stored part is device memory and this is conservative): $\gamma_s$ about 2.0 kB,
  $\gamma_h$ about 2.75 kB per DOF from the peaks above, with the 10M extrapolation checked
  against the trend from 100k to 1.58M (flat: falling per DOF).
- **step**, **step_low** $= 8n(a m + c) + 8mK$ per family, fitted at $m$ = 8, 15, 25 on the
  flat, stack and block cases (upper envelope); `AMG_LOOP_K`'s role for degree > 2 re-measured
  (the study's degree-3 steps are within 1.1 blocks).
- **count** $= 8n(a b + c)$ from `memory_analysis` of `_recurrence` with the shell V-cycle,
  $b$ = 4–32.
- The hex-only path is unchanged (`AMG_STORAGE` etc.); `amg_memory` picks the shell family when
  `ssh8` is present.

### Benchmark

`benchmarks/shell_solve_study.py` → `benchmarks/results/shell_solve_study.json` (≤ 1 h;
per-step progress printed):
- the accuracy/time table above with the shipped code (AMG, Jacobi where it converges),
  including the polish cases at $L/t$ = 300 and 1000 ($32^2$, $128^2$) and the PR 9 tolerance
  cases;
- the calibration of the shell family: storage per component, step and step_low at
  $m$ = 8 / 15 / 25, count at $b$ = 4–32, host peaks in fresh subprocesses (up to the 1.58M
  plate).

## Tests (18b)

1. **Dispatch** (`tests/test_shell_ops.py`): `resolve_precond` gives `"amg"` on `ssh8` and
   `ssh8` + `hex8`, `"jacobi"` on `ssh8` + `hex20`, `"amg"` raises there; `info["precond"]`,
   the "auto (resolved to amg)" label in option errors; `solve_modes_memory`/`amg_memory` no
   longer raise on `ssh8`. The old refusal tests are replaced.
2. **Accuracy against `eigsh`** (the roadmap test, `tests/test_shell_solve.py`), 20 modes,
   tol $10^{-10}$, `maxit` 3000, `precond="auto"`: λ relative error ≤ $10^{-8}$ and MAC ≥
   $1 - 10^{-8}$ against `reference_modes`, `info["precond"] == "amg"`, on: flat SS $32^2$
   ($L/t = 50$); clamped; free ($24^2$, 6 rigid modes deflated); 3-layer stack; plies laminate;
   curved panel; the sweep-built cylinder; T-junction and L-fold; the hex8-web plate; a
   bottom-supported plate. Where a case floor-accepts, its MAC/λ bounds follow the PR 9
   floor rule ($0.99 f_\lambda$) instead.
3. **Polish** (`tests/test_lobpcg.py` and the shell file):
   - $32^2$, $L/t = 1000$, tol $10^{-6}$: without polish (`polish_options={"steps": 0}`)
     `converged` is False and `stagnated` True; with the default it converges, `polishes` ≥ 1,
     every residual ≤ its threshold (explicit products recomputed in the test), λ against
     `eigsh` within the floor;
   - $L/t = 300$: one step;
   - a spy: `polish` is never called on a solve that converges without it (hex20 box, voxel
     GMG, flat $32^2$ at $L/t = 50$);
   - an identity `polish` (no improvement) stops after the progress rule, not `polish_steps`;
   - chunked ($n$ = 30 in chunks of 10) and band mode on the thin plate: the later chunks
     polish against the basis (deflation: no mode returned twice, MAC against `eigsh`);
   - a cluster at the block boundary found after a polish extends $j$ (the cluster rule runs
     on the polished Ritz values);
   - option errors (`steps` < 0, `tol` ≤ 0, non-int `maxit`, unknown keys), `precond="jacobi"`
     and the fp32 V-cycle get no polish (`info["polishes"] == 0`);
   - a thin hex20 plate under AMG that stagnates without the polish converges with it.
4. **`amg_memory` shell family** (`tests/test_shell_solve.py`):
   - storage ≥ the owned device bytes everywhere, the level-0 Schwarz term equal to the built
     one's bytes, and the $n$-scaling part within 1.3× on named tight cases (flat $128^2$, stack
     ×3, stack ×10); hex-dominant meshes (block) upper bound only;
   - an out-of-fit size above the largest calibration point (flat $362^2$, slow);
   - the mesh-level `_fibers`/`_columns` equal the operator-level ones;
   - step and step_low ≥ the measured step at $m$ = 8 and 15 (flat, stack, T, block), within
     1 block below the model on two tight cases (slow), and a degree-3 case;
   - count ≥ the measured recurrence at $b$ = 4 and 16;
   - build ≥ the host peak and ≤ 1.3× it in fresh subprocesses (flat $256^2$, block $32^3$;
     `peak`, slow);
   - `_pc_builds` composition: `solve_modes_memory` on a shell mesh sums as on hex meshes.
5. **Band mode** on a shell: `band=(0, f)` on the flat $32^2$ plate matches `eigsh` (every mode
   in the band) with the aggregates reused (spy: one fresh `amg` setup with
   `return_aggregates`, then `aggregates=`).
6. **M4** (`examples/m4_shell.py` test): under `"auto"` now AMG, at the PR 9 tolerances
   (the PR 16 test switches from Jacobi to AMG; Jacobi stays tested by an explicit
   `precond="jacobi"` run).
7. **Wall time** (`serial`): flat $64^2$ at $L/t = 100$, 20 modes, tol $10^{-6}$: AMG converges
   and is faster than Jacobi, which does not converge within its `maxit` (measured 4.7 s against
   30 s, Jacobi unconverged at 2392 iterations). Fallback if the margin is ever too small on a
   loaded machine: assert Jacobi's non-convergence only.
8. **Benchmark smoke** (slow): `shell_solve_study` at small sizes records counts, times and
   the calibration entries.

## Docs (18b)

- `docs/eigensolvers.md`: the "Solid shells" bullet (AMG by default from 18b; Jacobi on
  `ssh8` + `hex20`/`hex27`), a "Polish" subsection under Convergence (the stagnation
  mechanism, κ ∝ $(L/t)^4$, the measured table, the defaults), the Options table row, the
  `info` keys, the tests list.
- `docs/preconditioners.md`: the `amg_memory` shell family (formulas, constants, measured
  tables), the "AMG on solid shells" section's `solve_modes` sentence.
- `docs/open_items.md`: "Thin plates and slender beams" resolved for solid shells; the 18b
  decisions (polish design, `ssh8` + `hex20` under `"auto"`, constants' margins, the fp32 GMG
  exclusion).
- `docs/ROADMAP.md`: 18b status and section, `:580`/`:590` (M4 under AMG), the dependency
  table; `docs/elements.md:218`; README status and the laminate snippet (the 4 mm plate can
  come back: AMG converges where Jacobi stagnated, open_items `:211`).

## Measured (prototypes, 18b planning)

Scripts in the session scratchpad `s18b/`: `proto.py` (dispatch patched, AMG vs Jacobi vs
`eigsh`), `thin2.py` / `floor.py` / `kappa.py` (stagnation, floor, κ), `scilob.py` (SciPy
LOBPCG with LU), `polish.py` / `lobpolish.py` (the polish variants), `storage.py`, `step.py`,
`peaks.py` (the memory study).

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **The polish is new eigensolver behaviour** in 18b, not an 18d item: without it the
  roadmap test fails at tol $10^{-10}$ from $L/t \approx 100$, and "auto" AMG would stagnate on
  typical thin shells at the default tol. It changes only solves that stagnate today.
- **Polish cost:** at $L/t = 1000$ it costs up to 6 steps of 135–200 PCG iterations at width
  $m$, about as much as the LOBPCG phase; the stagnation detection itself spends `STALL` = 50
  iterations first. An earlier trigger (κ-based) is logged, not built.
- **Floor-limited accuracy:** at $L/t = 1000$ the λ error is about $5\cdot10^{-6}$ (the floor
  $\varepsilon\kappa \approx 10^{-4}$ in the residual); stated in the docs.
- **`ssh8` + `hex20`/`hex27`** stays Jacobi under `"auto"` (nonconforming meshes, 18a).
- **fp32 GMG** gets no polish.
- **Memory constants** carry margins for levels beyond the measured sizes (10M extrapolation
  from 1.58M); the deferred GPU checks of 18a stay.
