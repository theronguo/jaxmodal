# PR 18 plan: performance pass, part a: solid-shell AMG core

Branch `pr-18a-shell-amg` (this plan), from `main` after PR 17. Roadmap: PR 18 (M5), the
"Thin plates and slender beams" entry of `docs/open_items.md` (PR 11a → PR 18), and the
PR 16 rule "until PR 18, `solve_modes` on solid-shell meshes uses Jacobi".

## Revisions

- **Revisions 1–7** (review rounds 1–6): the design converged on these pieces, each backed by
  the measurements under "Why the PR 13 AMG fails" and "Measured":
  - fiber groups;
  - sign-free per-level curvature candidates;
  - $d_0 = 1$, $d_\ell = 2$;
  - the one-ring adjacency;
  - identity rows;
  - column and ring Schwarz patches;
  - block-key chunked extraction;
  - packed inverses with size classes and byte-sized balanced chunks;
  - 9-DOF hex-group patches;
  - the patch-size guard.

  Rounds 3–7 spent most of their findings on the `amg_memory` shell family and its calibration.
- **Revision 8** (round 7: three satisfied; simplicity and performance not):
  - **18a is split along PR 13's line** (13a core / 13b `solve_modes` and `amg_memory`):
    - 18a is `amg()` on solid shells, tested through PCG and V-cycles;
    - the new **18b** (`pr-18b-shell-solve`) gets `"auto"` → AMG, the σ pre-estimate on
      shells, the `amg_memory` shell family and its calibration, and the `solve_modes` tests;
    - the later parts are relettered 18c (kernels), 18d (band), 18e (CalculiX).

    The estimator constants depend on the final layout, so fitting them after 18a's code
    review avoids refits. The round 3–7 estimator findings are carried into 18b's plan (listed
    below).
  - **`ssh8` + `hex20`/`hex27` is dropped:** an `ssh8` face has 4 nodes and a hex20/hex27 face
    has 8 or 9, so the combination is never conforming. `amg()` raises on it, and `"auto"` keeps
    Jacobi there. This removes the corner-level Schwarz, the `cid` mapping, `_level`'s smoother
    argument and their tests.
  - **The PR 18 split and retags** land as the first, separate `docs:` commit of 18a.
  - **Round-7 correctness:**
    - the guard runs unconditionally after `_columns`;
    - constrained groups are removed from the graph (pyamg aggregates explicit zeros);
    - identity rows are detected by column norm (`fit_candidates` stores explicit zeros);
    - identity DOFs sit only in their own group's patch (Schwarz eigenvalue exactly 1).
  - **Round-7 performance (core part):**
    - prolongator smoothing applies $A$ to $T$;
    - coarse patches are bounded;
    - at most 4 size classes per level;
    - extraction keys are built per chunk;
    - the deferred GPU check has a pass criterion.

Line numbers in this plan refer to `main` (identical to this branch outside `docs/plans/`). The
implementation finds each entry by `grep "PR 18"` and treats the lists as checklists.

## PR 18 split

| part | branch | contents | plan |
|---|---|---|---|
| 18a | `pr-18a-shell-amg` | `amg()` on `ssh8` and `ssh8` + `hex8` meshes: fibers, curvature candidates, Schwarz smoothers, reuse; iteration counts vs $L/t$ and refinement; the PR 18 split | this file |
| 18b | `pr-18b-shell-solve` | `solve_modes` with shell AMG: `"auto"` → AMG on `ssh8` meshes, the σ pre-estimate on shells, band-mode reuse, the `amg_memory` shell family and its calibration, the `solve_modes` accuracy and Jacobi comparisons, the `test_shell_ops` refusal and label changes, the user-facing docs (README, eigensolvers, M4 example) | own plan |
| 18c | `pr-18c-kernels` | Morton reordering; fp32 for p-MG and AMG (estimator, the 1M Mac acceptance with fp32 AMG, fp32 Schwarz inverses); HLO fusion checks; the compile-time benchmark; the roofline tests; `docs/performance.md`; reuse of AMG aggregates across `_precond_sigma`'s σ rounds; the performance and estimator open items (below) | own plan |
| 18d | `pr-18d-band` | the PR 15d band follow-ups of the roadmap bullet (memory levers, $\hat\lambda_\max$, $\delta_\max \le 1$, MINRES remedies near depth 1000, halving before a capacity error, `err.result`, the hybrid junction, $f_1 > 0$ in `solve_modes_memory`, `np.memmap`); the `COUNT_STEP` refit at ≥ 1M DOF (on the post-18b `ssh8` default, AMG); the band, count and slice open items (below) | own plan |
| 18e | `pr-18e-calculix` | CalculiX comparison (user request, 2026-10-18): CalculiX `*FREQUENCY` and `solve_modes` on the same `.inp` meshes (C3D20/C3D8), about 100k–1M DOF, 20 and 200 modes; wall time, peak memory, frequency agreement, a SciPy shift-invert as second reference; ≤ 1 h per run; CalculiX in its own conda env | own plan |

**Owners of the `→ PR 18` entries in `docs/open_items.md`.** 18a rewrites each one to name its
part.
- **18b:** the thin-plate entry (:16).
  - 18a adds that the AMG core exists, and 18b resolves the entry for solid shells.
  - For hex and voxel plates it becomes a documented limitation with no owner (remedy:
    convert to `ssh8`).
  - The heading and ROADMAP :474 are updated by 18b.
- **18c** (performance and estimators):
  - `assembled._chunked` pipelining (PR 17);
  - the refill through `device_put(V[:, i:i+c].T).T`;
  - caching `layout` for the gather path (:169);
  - the verify skips before the gather (:255; the roadmap lists them with these items);
  - :171 (the gather on a skip, the same item as :255);
  - :256 (merge and count copies, the eager host residual, MX sent twice per chunk);
  - host transposes and small compiles (:257);
  - `_amg_scan` keeping the mesh alive;
  - `gmg_memory` recomputed per search step;
  - Metal `_default_budget` (:187);
  - new in 18a:
    - $P_0^\top$ applied as a scatter over $P_0$'s ELL (about 220 B per DOF);
    - 1.25× ELL width classes for $P_0$ (padding 1.45 → about 1.1);
    - a public `chunk_bytes` option and tuning the Schwarz chunk on GPU, if the GPU check needs
      them.
- **18d** (band, count, slices):
  - :141, :147, :152, :159–:161, :174, :175, :179, :181;
  - :207 (the `COUNT_STEP` refit part) and :226;
  - :186's locked-mode reuse, store sizing and `t_ref` with compile;
  - :245–:257 except :255, :256 and :257;
  - :259–:275 (span counts, shift-invert near depth 1000, σ-move retries, σ between the
    edges).
- **Deferred GPU checks** (ROADMAP list): "XLA kernel temporaries" (:187). The measured time
  crossover stays under the PR 15d deferred GPU checks.

**Generic "PR 18" mentions** elsewhere are retagged in 18a:
- 18b:
  - the `no_shell` message (`pmg.py:216`: "solid-shell AMG arrives in PR 18b");
  - `docs/preconditioners.md:53` (until PR 18b);
  - `docs/eigensolvers.md` :40, :41;
  - README :82;
  - ROADMAP :18's forward-looking "M4 / PR 18" mention.

  18b rewrites these statements. 18a only retags them.
- 18c:
  - `docs/operators.md` :204, :251 (roofline);
  - `docs/mesh.md:38`;
  - `docs/eigensolvers.md:160`;
  - `docs/conventions.md:89` (compile-time tests);
  - `docs/preconditioners.md:476` (`layout` per call);
  - `docs/boundary_conditions.md:53` (voxel-only search);
  - ROADMAP :92 (fp32 AMG), :168 (memory table), :203, :206 (compile time) and :741.
- 18d:
  - `docs/eigensolvers.md` :126, :152, :172, :178, :621, :723, :764;
  - ROADMAP :82, :95, :97, :176 (memmap).
- ROADMAP :788–:789 (dependency rows): "18 Performance" is split into the five parts, and row
  19 becomes "17, 18a–18d".
- `docs/eigensolvers.md:150` (time crossover) → the PR 15d deferred GPU checks.
- `docs/eigensolvers.md:783` names the parts that use the PR 8 `eigsh`: 18b, and 18d/18e if
  they use it.
- ROADMAP :183 names the estimator parts: 18b (the shell `amg_memory`), 18c (fp32 p-MG/AMG),
  18d (the `COUNT_STEP` refit).
- The ROADMAP PR 18 section (:625–:640) is split into 18a–18e subsections:
  - :632 and :640 go to 18a, and :640's fallback names the configuration that ships (test 6);
  - :589 (Jacobi `solve_modes` on M4) gets "(Jacobi in 16b; `"auto"`, AMG, from 18b)";
  - the `COUNT_STEP` bullet goes to 18d;
  - roofline, fp32, Morton, compile time and `docs/performance.md` go to 18c;
  - CalculiX goes to 18e (a Contents bullet and the test entry, text only).
- The deferred GPU check "PR 18: matvec ≥ 50% of roofline" is retagged 18c.

`benchmarks/strategy_study.py:227` and its recorded JSON are historical and stay.

**Dependencies:**
- 18a ← 13, 15d, 16c;
- 18b ← 18a;
- 18c ← 11, 12, 13, 18b;
- 18d ← 15d, 18b;
- 18e ← 18a–18d;
- 19 ← 18a–18d (18e only measures).

Each later part gets its own reviewed plan, `docs/plans/pr-18x-<slug>.md`. **18a's first
commit** (`docs: PR 18 split`) carries all of the above:
- the split, with per-part status lines in `docs/ROADMAP.md`;
- the owner rewrites in `docs/open_items.md`;
- the retags;
- the dependency rows.

The code commits follow it.

**Carried into 18b's plan** (round 3–7 findings on the estimator; 18b's plan starts from them):
- **storage terms:**
  - the level-0 Schwarz exactly by `schwarz.nbytes`;
  - $\alpha_P n_{c,\text{shell}}$;
  - $\alpha_\text{coarse} \cdot 3 n_\text{fib}$ (stacks scale with fibers, not $n_c$);
  - a shell-path hex constant (measured 1.19–1.24× PR 13's hex8 hierarchy at 24³–48³, growing
    about linearly in $n$; the extrapolation form is chosen from a per-level breakdown, else
    the larger of linear-in-$n$ and log);
  - the coarsest factor.
- **`build` keeps the existing convention** `build = host_build + device inversion
  temporaries` (`memory.py:135-137`, `_pc_builds`). `host_build` gets a shell-path hex
  constant, and α_shell's growth per decade is extrapolated.
- **step terms:**
  - level-0 chunk temporaries exactly from `schwarz.nbytes`;
  - coarse levels bounded by `CHUNK_BYTES`;
  - `AMG_LOOP_K` checked at degree 3, with the Schwarz gather term per column as fallback.
- **fitting:** each constant is fitted on its own measured leaf group (on single-layer plates
  $3 n_\text{fib} = n_c/2$, so a fit on totals is collinear).
- **tests:**
  - gates as `test_amg_memory_storage`: `got ≤ est` everywhere, and 1.3× only on the
    $n$-scaling part of named tight cases;
  - hex-dominant meshes upper-bound only;
  - an out-of-fit size above the largest calibration point;
  - a degree-3 step case;
  - storage checks not darwin-only;
  - a host-peak check on a hex-dominant mesh;
  - `_pc_builds` composition asserted.
- **solve_modes tests:**
  - accuracy against `eigsh` on plates, stacks, plies, the curved panel, the sweep-built
    cylinder, junctions, mixed meshes, clamped and free plates;
  - band-mode reuse with a spy;
  - the Jacobi wall-time comparison (`serial`, with a stated final fallback);
  - Jacobi stagnation at $L/t = 50$;
  - M4 under `"auto"`;
  - the `test_shell_ops` refusal and label changes.

## Goal (18a)

`amg()` works on meshes whose blocks are all `ssh8`, or `ssh8` and `hex8`, including junctions.
- On a simply supported single-layer plate $128^2$, PCG iterations grow ≤ 2× from $L/t = 10$ to
  $L/t = 1000$ (roadmap test).
- Under refinement at fixed $L/t$ they do not grow: $256^2$ ≤ 1.3× $128^2$.
- `ssh8` meshes with `hex20`/`hex27` blocks raise in `amg()`.
- Hex-only meshes run the unchanged PR 13 code.
- `solve_modes` and `amg_memory` are unchanged: they still refuse AMG on `ssh8` until 18b.

## Why the PR 13 AMG fails on solid shells

Measured in the 18a prototypes (Measured below): SS plate $n^2 \times 1$ `ssh8`, PCG to
$10^{-6}$ at σ = $\lambda_1$, Chebyshev degree 2 on every level, dense coarsest solve.

1. **Near-null space.** Rigid modes do not represent plate bending on coarse aggregates (the
   PR 13a hex20-plate entry). Three constant-curvature fields are added. They are the
   Kirchhoff displacements $u_\alpha = -Z\,\partial w/\partial X_\alpha$, in an aggregate's
   local frame $(t_1, t_2, n)$ with coordinates $(X, Y, Z)$:
   $$\kappa_{11}: w = \tfrac{X^2}{2},\ u_1 = -ZX; \qquad \kappa_{22}: w = \tfrac{Y^2}{2},\ u_2 = -ZY; \qquad
   \kappa_{12}: w = XY,\ u_1 = -ZY,\ u_2 = -ZX.$$
   With the 6 rigid modes, each aggregate has 9 coarse DOF.
2. **Smoothers.** Once $h/t \gg 1$, point, node-block and fiber-block Jacobi lose the
   membrane–bending coupling, which lives in the element. $n = 32$, $L/t$ = 10/100/300/1000:
   - fiber blocks: 10/17/51/183;
   - element patches: 11/13/16/38;
   - vertex patches: 11/12/15/24.
3. **Coarse levels.** Each coarsening widens aggregates about 3× in-plane at the same
   thickness. Block Jacobi on 9×9 aggregate blocks then fails as fiber blocks do on level 0.
   $n = 64$, $L/t = 1000$:
   - two-level: 22;
   - V-cycle with coarse block Jacobi: 79;
   - with coarse vertex patches: 24;
   - with coarse edge patches: 78.
4. **The bending candidates must be one field per level.** SA passes the coarse candidates down
   as coefficients ($T R = B$), so frames fixed on the fine level give piecewise level-1 fields.
   - Flat plate, $n = 64$, $L/t = 1000$: fine-level frames only take 80 iterations; global
     bending fields 26; candidates rebuilt per level from the next aggregates' frames 26.
   - 90° panel: global fields 96, per-level frames 35.
5. **Prolongator smoothing.** Bending is fourth order. With one Jacobi step on every level the
   counts grow with depth: 26 at $n = 64$, 40 at $n = 128$ (4 levels).
   - Two steps on the coarse levels fix it: one step on level 0 and two below give 21 at
     $n = 128$ and 18 at $n = 256$.
   - The second step widens coarse stencils, so patches and the next aggregation use the
     **one-ring aggregate adjacency**. Over the matrix pattern, vertex patches cost 852
     floats/DOF; over the adjacency, 114.
6. **Stacks need column patches.** With $k$ elements through the thickness an element patch
   covers $1/k$ of it: 3 layers, $n = 32$: 10/43/322/>2000. Column patches (an element's 4
   fibers) give 11/15/40/117. Stacks stay weaker at $L/t \ge 300$ (limitation).

## Design (18a)

### Scope of the shell path

`amg()` takes the shell path when any operator block is `ssh8`.
- With a `hex20`/`hex27` block as well, it raises `ValueError` ("solid-shell AMG supports ssh8
  and hex8 blocks").
- Otherwise the PR 13 code runs unchanged (test 13).
- `_sa_levels` and `_shell_levels` stay separate: the shell loop differs in its group graph,
  identity rows, $b = 9$ and per-level candidates.

New code:

| piece | about |
|---|---|
| `precond/schwarz.py`: `Schwarz`, `_extract`, chunked inversion, size classes, `nbytes` | 180 lines |
| `precond/shell.py`: `_fibers`, `_columns`, frames, candidates, smoothing and `_shell_levels` | 260 lines |
| `amg()` wiring, the hex20/hex27 refusal and `AMGAggregates` fields | 30 lines |
| `_check`'s refusal moved into `pmg()`, the `no_shell` retag | 5 lines |

That is about 450–500 source lines, plus about 500–600 test lines.
- `precond/shell.py` keeps the shell setup out of `amg.py`.
- `amg.py` keeps `amg()`, `AMGAggregates` and the dispatch, so the hex path's diff stays small.
- `schwarz.nbytes` is the single layout rule. 18b's `amg_memory` imports it, together with
  `shell._fibers` and `shell._columns`.

### Level structure

Every block has 24 DOF, so `amg()` already treats these meshes as "all hex8".
- The shell path calls `ap_operator` and `_hex8_galerkin` directly, not `_p_level` (whose
  `block_jacobi` would be built and dropped).
- Level 0 is the matrix-free $A_P$ with Chebyshev over the level-0 Schwarz. That is exact: an
  `ssh8` element's nodes are all corners and `_w8(8)` is the identity, so $A_b = A_P$.
- Its transfer is $P_0$, and the shell SA levels follow.
- **Nothing coarsens** ($n_c \le$ `max_coarse`, with $n_c$ the corner DOFs throughout, for
  example the 486-DOF M4 plate): one level, the dense Cholesky of $A_b = A_P$, as PR 13.

### Groups and aggregation

**Level-0 groups (corner space) are through-thickness fibers.**
- A fiber is a connected component (`scipy.sparse.csgraph.connected_components`) of the graph of
  node pairs $(a, a+4)$, $a = 0..3$, over the `ssh8` elements (bottom face 0–3, top 4–7,
  `docs/elements.md`).
- One layer gives pairs, and a stack of $k$ elements gives $(k+1)$-node fibers.
- At a T-junction or L-fold the two members' fibers merge into one larger group. That is
  legitimate: measured T 19 / L 24 against a flat 14 at $L/t$ 100, $n = 32$.
- Corner nodes in no `ssh8` element (`hex8` nodes) are singleton groups.

**Size guard.** It runs unconditionally right after `_columns`, before any coarsening decision.
- A level-0 patch with $s > 384$ (more than 128 distinct nodes in an element's fibers) raises
  `ValueError` ("ssh8 patch of N DOFs: check solid_shell's direction, or the stack is too
  deep").
- It allows 31-layer stacks and multi-layer junctions up to 6 layers per member.
  - A T of a 5-layer skin and a 5-layer web merges $6 \cdot 6 = 36$ nodes, with largest patch
    $s = 252$ (84 nodes: a web element on the interface row, or a skin element at the web's
    footprint edge; review).
  - Two 6-layer members give $s = 336$, and 7 layers give 432, which raises.
- It catches a consistently wrong axis, for example `direction=(1, 0, 0)` on a $40^2$ plate
  ($s = 4 \cdot 41 \cdot 3 = 492$), whose in-plane chains would make dense column patches.
- A wrong axis on smaller meshes is not detectable, as the docs state.

**Level $\ell \ge 1$:** the groups are the level-$\ell$ aggregates. Every aggregate keeps 9
coarse DOF, so coarse DOF $j$ belongs to group $\lfloor j/9 \rfloor$.

**Group graph:**
- **Level 0:** $G^\top S G$, with $S$ the `Hex8Pattern` structure (its `indptr`, `indices`;
  ones) and $G$ the node → group indicator, built over **free groups only** (a group with a
  free DOF), diagonal removed.
  - pyamg's `standard_aggregation` reads only `indptr`/`indices` and aggregates explicit zeros,
    so fully constrained groups are left out of the graph, not zeroed.
  - The graph is mask-dependent but value-independent (σ, materials). `free` is checked on
    reuse, so reuse is exactly a fresh setup.
- **Level $\ell \ge 1$:** the one-ring adjacency $\mathrm{Agg}_{\ell-1}^\top S_{\ell-1}\,
  \mathrm{Agg}_{\ell-1}$, diagonal removed, with $S_{\ell-1}$ the previous group graph and
  $\mathrm{Agg}_{\ell-1}$ its group → aggregate indicator. That is a one-ring, not a two-ring.

**Aggregation:** `standard_aggregation` on the group graph.
- Free groups left unaggregated (no neighbours) become singleton aggregates (separate parts,
  coarse leftovers).
- Fully constrained groups stay unaggregated with empty tentative rows, as PR 13.
- Fine-node membership of a level-$(\ell+1)$ aggregate is composed through the group →
  aggregate maps.

### Candidates

On level $\ell$, after aggregation:
- **$B^{(\ell)}_\text{fine}$**, $(n_c, 9)$:
  - columns 0–5 are the masked rigid modes at the corner points (`_nullspace`);
  - columns 6–8 are the curvature fields of each corner node's level-$(\ell+1)$ aggregate, in
    its frame;
  - constrained rows are zeroed.
- **Sign-free frame:**
  - Per node, $T_a = \sum v v^\top$ over the thickness edges $v$ (top minus bottom node, 4 per
    element) of the `ssh8` elements containing it.
  - Per aggregate, $n$ is the dominant eigenvector of $\sum_{a} T_a$ (batched `eigh`). The span
    of the three fields is invariant under $n \to -n$: $t_1 \to -t_1$, $\kappa_{11}$ and
    $\kappa_{22}$ change sign, $\kappa_{12}$ does not.
  - $t_1 = n \times e/\|n \times e\|$, with $e$ the coordinate axis least aligned with $n$;
    $t_2 = n \times t_1$.
  - $(X, Y, Z)$ are the projections of $x - \bar x$, with $\bar x$ the aggregate's corner-node
    mean.
  - The fields are set on every node of the aggregate, hex nodes of mixed aggregates included.
  - Aggregates with $\sum T_a = 0$ (no `ssh8` node: hex groups) get zero curvature columns.
  - Vectorized: `np.add.at`, batched `eigh`, no per-aggregate loop.
- **Projection:** $B_\ell = (T_0 \cdots T_{\ell-1})^\top B^{(\ell)}_\text{fine}$.
  - $T$ has orthonormal or zero columns, so this is the least-squares coefficient.
  - It is exact for the rigid columns. It is exact for the curvature columns on flat plates and
    complete-fiber stacks (equal mean $Z$ per aggregate; test 2).
  - Level 0 uses $B^{(0)}_\text{fine}$.
- **Fit:** `fit_candidates` → $T_\ell$, with all 9 columns per aggregate kept.
  - Its tolerance is relative per column, so it is unit-independent.
  - It writes dependent columns as exact zeros, stored explicitly in the BSR.
  - $R_\ell$ is neither stored nor used as the next $B$.

### Prolongator and coarse matrices

- **Smoothing:** $P_\ell = (I - \tfrac{\omega}{\rho} D^{-1} A)^{d_\ell}\, T_\ell$, with
  $\omega = 4/3$, $D$ the diagonal, $d_0 = 1$ and $d_\ell = 2$ for $\ell \ge 1$.
  - It is the shell path's own few lines: $P \leftarrow P - \tfrac{\omega}{\rho} D^{-1}(A P)$,
    applied $d_\ell$ times. It never forms $D^{-1}A$ or $S^2$.
  - ρ is pyamg's `approximate_spectral_radius` under the local seed. Its argument is a
    `LinearOperator` $x \mapsto D^{-1}Ax$ if pyamg accepts one (checked in implementation),
    otherwise a scaled copy freed before the smoothing.
  - `_shell_levels` returns ρ per level for test 5. It is not stored on `GMG` or
    `AMGAggregates`.
- **Galerkin:** $A_{\ell+1} = P^\top A P$, symmetrized.
- **Identity rows:**
  - The zero columns of $T_\ell$ (aggregates with fewer than 9 independent candidates: hex
    groups, or constrained) give zero rows and columns of $A_{\ell+1}$. They are detected by
    column norm $= 0$, as `amg.py:78`. That is exact, since `fit_candidates` writes columns
    that are exactly 0 or of unit norm. The flags are stored with the tentatives.
  - Their diagonal is set to the mean diagonal of $A_{\ell+1}$'s other rows. The DOFs are
    decoupled, $A_{\ell+1}$ stays SPD, and coarse sizes stay multiples of 9.
  - They are inert in the V-cycle:
    - $P_\ell$'s column is 0 there, so the restricted residual is 0;
    - $A_{\ell+1}$ and the Schwarz are decoupled there;
    - the next $P$ row is 0, since their candidate rows are zero.
  - Identity DOFs sit only in their own group's patch (below), so their Schwarz eigenvalue is
    exactly 1. They neither raise the Lanczos $\lambda_\max$ estimate nor fall below the
    Chebyshev interval while $1.1\lambda_\max/30 < 1$.
  - This replaces PR 13's zero-column drop on the shell path.
- **Stop rules:** `STALL` on the aggregate count ($9\,n_\text{agg} > 0.9\,n_\ell$), `CAP` on the
  coarsest size, as PR 13. Both count identity DOFs, so on hex-heavy meshes the coarsest dense
  factor can be up to 1.5× PR 13's for the same `max_coarse`. That is harmless and documented.

### Smoothers: additive Schwarz under Chebyshev

On every SA level, Chebyshev (`degree`, `ratio` as today) runs over
$$D^{-1} r = \sum_i R_i^\top A_i^{-1} R_i\, r, \qquad A_i = R_i A R_i^\top,$$
with patches $i$ that cover every DOF of the level. Each $A_i$ is a principal submatrix of an
SPD matrix, so $D^{-1}$ is SPD and the V-cycle stays symmetric.
- **Level 0 patches:**
  - per `ssh8` element, the DOFs of its 4 fibers, deduplicated within the patch;
  - patches deduplicated across the elements of one column, keyed by the sorted set of
    distinct group ids (at junctions an element's 4 fibers can fall into fewer groups);
  - one layer gives the element's 24 DOFs, and $k$ layers give $12(k+1)$;
  - per `hex8` corner node, a node patch (3 DOFs).
- **Level $\ell \ge 1$ patches:**
  - per shell group (one with an `ssh8` node, $\sum T_a \ne 0$): its 9 DOFs plus the
    non-identity DOFs of those one-ring neighbour groups that are shell groups too;
  - per hex group, its own 9 DOFs (block Jacobi, still stronger than PR 13's scalar Jacobi).
  - So identity DOFs lie only in their own group's patch.
  - Why: a one-ring in hex regions has about 22 neighbours ($s \approx 210$–243, about 3.2 kB
    per DOF on level 1, review), against 7.7 on plates, and skin aggregates would pull in 9–17
    hex neighbours. Shell-only rings keep every shell patch at plate size ($s \le 81$ on a
    single layer), and hex neighbours are covered by their own patches. Test 9 checks the
    counts.
  - **Bound:** a coarse patch with $s > \sqrt{\texttt{chunk\_bytes}/8}$ (2048 at the default)
    raises `ValueError`. Ring patches have no fiber guard and can grow at hubs (radially meshed
    disk centres, fans).
- **Stacks:** a column patch has $s = 12(k+1)$, so level 0 costs about $24(k+1)$ packed floats
  per DOF: 96 at $k = 3$, 264 at $k = 10$ (about 2.1 kB per DOF). The docs state the formula.
- **Extraction by block keys, in chunks:** one `_extract(A_bsr, patches, chunk)` for both level
  kinds.
  - A patch is a list of block indices: nodes in $A_b$ as BSR(3) on level 0 (built on the
    `Hex8Pattern`'s `indptr`/`indices`, `pmg.py:210`), groups in $A_\ell$ as BSR(9) below.
  - Coarse BSR matrices get `sort_indices()`; the `Hex8Pattern` is already sorted.
  - For each chunk, the keys of the chunk's patch rows are built as int64
    `indptr[row] + searchsorted` within the row. No whole-matrix key array is built (it would
    be about 72 B per DOF).
  - Hits are clipped and checked for equality, then `data` is read and expanded to DOFs by
    `b * blk + arange(b)`.
  - Padded block slots skip the lookup, so a pad never aliases a real pair.
  - Missing pairs are 0, and explicit zero blocks extract as zeros.
  - There is no CSR copy and no in-place compaction of $A_b$ on the shell path.
  - Extraction runs inside the same fixed-chunk host loop as the inversion, so its transient is
    bounded and independent of $n$.
- **Inversion:** per chunk, `_inverse_blocks(B, idx < n)` on the device: Cholesky, then the
  inverse. Padded slots get the identity, and the mask zeroes their cross terms.
  - The padded rows and columns of the inverse are then zeroed.
  - The result is packed and written into a preallocated per-class buffer by a donated update.
  - There is one jitted inversion per (level, class) shape, with the buffer offset as a traced
    `dynamic_update_slice` argument, so balanced chunks never recompile.
  - A non-finite inverse raises `ValueError`, as `_level`.
- **Storage:** the upper triangle, packed (`matfree._packed_index`), $s(s+1)/2$ floats per
  patch: 300 per single-layer element.
- **Size classes:**
  - Sizes are sorted descending. A class opens at the largest remaining size and takes smaller
    sizes while its padded bytes stay ≤ 1.1× its unpadded bytes.
  - At most 4 classes per level: above that, the adjacent pair with the least added padding is
    merged. Each class costs one compile and one scan per apply.
  - Each class is padded to its maximum. Padded slots carry index $n$ (gather `mode="fill"`
    with 0, scatter `mode="drop"`) and zero inverse rows, so no mask is stored.
  - Examples: the plate's level 0 is one class ($s = 24$); its level 1 (27/45/63/81, 95% at
    81) is one class; a hex-dominant level 0 (24 and 3) is two classes.
- **Chunks** are sized by bytes and balanced:
  - The target is $c = \max(1, \lfloor \texttt{chunk\_bytes}/(8 s \max(s, 24)) \rfloor)$. It
    bounds both the unpacked chunk ($c s^2$) and the gathers per column ($c s$).
    - `chunk_bytes` defaults to `matfree.CHUNK_BYTES` (32 MiB).
    - $c \ge 28$ on level 0 under the guard, and $c \ge 1$ on coarse levels under their bound.
  - Then $n_\text{chunks} = \lceil n_p/c \rceil$ and chunk = $\lceil n_p / n_\text{chunks}
    \rceil \le c$. That leaves fewer than $n_\text{chunks}$ pad slots per class (review:
    unbalanced chunks inflated small classes 5×).
  - Level 0 at $s = 24$ gets about 7280 patches per step (the hex8 Ke chunk is 4096).
  - `chunk_bytes` is an argument of the internal Schwarz build only (test 3 calls it with small
    values). `amg`'s signature is unchanged.
- **One layout rule:** `schwarz.nbytes(sizes, dtype, chunk_bytes)` covers the classes, the
  balanced-chunk padding and the packed `inv` plus `idx`. It also returns each level's maximum
  chunk·$s^2$ and chunk·$s$. The `Schwarz` build uses it for its layout, and 18b's
  `amg_memory` will use it too.

### `Schwarz` (device)

A registered pytree with, per class, `idx (n_p, s) int32` and `inv (n_p, s(s+1)/2)`.
- The apply on $(n, k)$ blocks is a `lax.scan` over the chunks.
- Each step does the gather, unpacks by `_packed_index`, runs an `einsum` and scatter-adds into
  the scan carry (`.at[idx].add(..., mode="drop")`).
- There are no stacked outputs, which would hold $n_p s k \approx 4 n k$ floats on level 0.

Temporaries, both independent of $n$:
- the gathers and the `einsum` output, each chunk·$s$·$k$ floats, at most `chunk_bytes`·$k$/24
  bytes;
- one unpacked chunk of chunk·$s^2$ floats, at most `chunk_bytes`.

### Cost (review device measurement, CPU fp64, $m = 15$, rev. 2 hierarchy and layout)

At $n = 256$ (396k DOF), $t_A$ = 59.6 ms (stored-Ke $A_P$), per fine DOF:

| component | storage | apply ($t_A$) |
|---|---|---|
| level-0 Schwarz ($s = 24$, packed, with `idx`) | 463 B | 0.45 |
| $P_0$ / $P_0^\top$ ($b = 3$, ELL padding 1.45) | 306 / 223 B | 0.42 / 0.25 |
| level-1 Schwarz | 517 B | 0.33 |
| $A_1$ ($b = 9$) | 148 B | 0.10 |
| $P_1$ + $P_1^\top$ ($b = 9$) | 183 B | 0.13 |
| level 2 and below | 136 B | small |
| **total** | **about 2.0 kB** (n = 64: 2.08, n = 128: 2.20) | V-cycle **7.5** (n = 128: 8.8; n = 64: 14.2) |

- **Layout:** the rev. 2 layout had int64 `idx`, a stored mask and unbalanced chunks. The
  current layout (int32 `idx`, no mask, balanced chunks) gives level 0 about 411 B and a total
  of about 1.9 kB per DOF at $n = 256$.
- **Against the operator:** the stored-Ke operator is 0.47 kB per DOF, so the preconditioner is
  about 4×. That is about 1.9–2 GB at 1M DOF and 19–20 GB at 10M (fp64).
- **Counts:** device PCG counts reproduce the host prototype, 24 / 21–22 / 18–19 at $n$ = 64 /
  128 / 256.
- **Packed against full inverses:**
  - On CPU, the unpack costs about 11% of the patch apply.
  - On a bandwidth-bound GPU the packed apply moves about 2.5× the bytes of full inverses.
  - "Packed vs full inverses (fp32 full = fp64 packed bytes)" joins the deferred GPU check and
    18c's fp32 Schwarz item.
- **Host setup** without the per-patch loop: about 4 s at 396k DOF.
- **Deferred GPU check** (ROADMAP list): the Schwarz scan at 10M.
  - The case: a single-layer plate has about 1.67M level-0 patches (229 steps per apply) and
    about 192k level-1 patches ($s = 81$, $c = 639$, about 300 steps), plus about 40 below.
    That is about 570 steps per apply and 2300 per V-cycle at degree 2.
  - Pass criterion: the scan-step overhead (steps × per-step latency) stays below 25% of the
    V-cycle time. It is recorded against `chunk_bytes` so that 18c's public option is decided
    by a measurement.
  - The same check records packed vs full inverses.
- **Benchmark:** `benchmarks/shell_amg_study.py` → `benchmarks/results/shell_amg_study.json`
  (≤ 1 h). It records:
  - the counts of tests 6, 7, 9–11;
  - storage per level and per component;
  - V-cycle time / $t_A$ and host setup time;
  - on the hex-dominant mesh, V-cycle time / $t_A$ and storage next to the PR 13 hex8 build of
    the same block without the skin.

### API

- **`amg(op, free, sigma, points, ...)`:** signature unchanged.
  - `return_aggregates`/`aggregates=` work on the shell path. `AMGAggregates` keeps the
    `Hex8Pattern`, mask and fingerprint, plus the fibers and, per level, the group → aggregate
    map, the tentative $T_\ell$ and its zero-column flags.
  - Reuse skips fibers, aggregation, frames and fits. It rebuilds the patch sets (a few
    vectorized lines from the groups) and recomputes $A_b$, the smoothed $P$, the Galerkin
    products, the patch values and inverses, and the export.
  - Moved points keep the old candidates, as PR 13.
- **Refusals:**
  - `_check` loses its `ssh8` refusal;
  - `pmg()` raises inline ("pmg does not support ssh8 blocks; use amg");
  - `amg()` raises on `ssh8` with `hex20`/`hex27`;
  - `no_shell` stays for `resolve_precond` and `amg_memory` (both moved to 18b), with its
    message retagged "PR 18b".

## Tests (18a)

`tests/test_shell_amg.py` unless noted. All checks go through `amg()`, PCG (`linalg.pcg`) and
the V-cycle; `solve_modes` checks are 18b's.

**Mesh helpers:**
- `plate(n, Lt, curved=False, layers=1, bcs="ss")`: `box_mesh((n, n, layers), spacing=(1/n,
  1/n, t/layers))` + `solid_shell(direction=(0, 0, 1))`, with $L = 1$ the in-plane side and
  $t = L/(L/t)$. `curved` maps it onto a 90° panel of radius 1, keeping $L = 1$ as the straight
  side (the arc is 1.57).
- `junction(kind, n, Lt, ks=1, kw=1)`: an all-`ssh8` skin with a web (T, or L at an edge), with
  `ks`/`kw` layers and per-element `direction`, in sorted selected order.
  - `Mesh` rejects orphan nodes and there is no merge helper, so the pieces are `box_mesh`
    pieces glued by `np.unique` on the key `np.round(pts / (1e-6 * t))`. `helpers.mixed_mesh`'s
    `rint(2*pts)` key fails on the width-$t$ intervals.
  - The skin's in-plane `coords` have $k_w$ intervals of width $t/k_w$ at the web, so the web's
    fiber nodes coincide with skin nodes. With one interval the interior fiber nodes of a
    multi-layer web would hang.
- `web(n, Lt)`: a $8^2$ or $32^2$ `ssh8` plate with a `hex8` web 3 element rows deep (so pure
  hex aggregates exist), glued the same way.
- **Hex-dominant block:** one `box_mesh(coords=...)` with `cz = r_[linspace(0, 1, n + 1), 1 + t]`
  ($n^3$ `hex8` elements plus the skin, $L/t = 100$), then `solid_shell(elements=top layer,
  direction=(0, 0, 1))`. No gluing.

**Conventions:**
- **σ:**
  - SS flat plates use the Kirchhoff $\lambda_1 = D\pi^4(2/L^2)^2/(\rho t)$, with
    $D = Et^3/(12(1-\nu^2))$. That is within 6% of $\lambda_1$ at $L/t = 10$ and 0.2% at
    ≥ 300, and the counts equal those at `eigsh` σ (review).
  - The clamped plate uses the clamped coefficient 35.99 instead of 19.74.
  - Curved panels, junctions, webs, the bottom-supported plate and the hex-dominant block use
    an `eigsh` $\lambda_1$ (σ = 0, tol 1e-3) of the same mesh. For $128^2$ panels it uses the
    $64^2$ version of the same geometry, which is cheap: counts are insensitive to σ within a
    few %, and `eigsh` on $128^2$ costs about 26 s and 2.6 GB. The flat σ would be 10–190×
    below the panel's $\lambda_1$.
- **Counts:** `pcg` to $10^{-6}$ on a fixed-seed 4-column block, max over columns (spread ≤ 3).
  Each gated count is ≤ 1.5× the implementation's measured device count, recorded in the docs
  and the benchmark JSON.
- **Hierarchy:** count tests build with `max_coarse=100` unless stated, and assert
  `len(P.levels) ≥ 2`.

1. **Fibers and the guard:**
   - 1 layer gives pairs;
   - 3 layers give the `box_mesh` columns;
   - `ssh8` + `hex8`: hex-only nodes are singletons;
   - T-junction: the joint fiber merges, with no error, and the level-0 patches have no
     repeated DOF;
   - the 3-layer stack has $n_\text{el}/3$ level-0 shell patches, each of size
     $48 = 12(k+1)$ (deduplicated across a column);
   - `direction=(1, 0, 0)` on a $40^2$ plate ($s = 492$) raises the size-guard `ValueError`;
   - guard boundary: a $2^2$ plate with 31 layers builds ($s = 384$), and 32 layers raise
     (the guard runs although nothing coarsens there);
   - the 5/5-layer T-junction builds. Its largest patch equals 3 × the distinct nodes of each
     element's groups, maximized over elements, from an independent NumPy union-find (252 by
     the review's count);
   - a mesh with an `ssh8` and a `hex20` block raises the mixed-type `ValueError`.
2. **Candidates:**
   - **Flat plate:** $\|(I - T_0 T_0^\top) B^{(1)}_\text{fine}\|$ ≤ 1e-10 relative (rigid and
     curvature columns). Per aggregate, the level-0 span equals the global
     $[\text{rigid}, \kappa]$ span (principal angles ≤ 1e-10). Same on the 3-layer stack.
   - **Level 2:** on a $16^2$ plate with `max_coarse` 20 (levels about 1734 / 324 / 36 / 9):
     - assert `len(P.levels) ≥ 3` (smoothing levels, so $A_2$ exceeds `max_coarse` and $T_2$
       exists);
     - $\|(I - T_0 T_1 T_1^\top T_0^\top) B^{(2)}_\text{fine}\|$ ≤ 1e-10 relative.
   - **Sweep-built 270° cylinder** (`solid_shell(direction=None)`, orientations flip; 96 of
     192 elements point inward, review):
     - the vectorized normals equal a per-aggregate NumPy `eigh` of $\sum v v^\top$ up to sign,
       to 1e-12;
     - no NaN;
     - $|n \cdot \hat r(\bar x)| \ge \cos$ of the aggregate's full angular span (both lie inside
       it, with different weights);
     - the same cylinder built with consistent orientations gives the same curvature span
       (principal angles ≤ 1e-12).
3. **Schwarz apply** equals a dense NumPy $\sum_i R_i^\top A_i^{-1} R_i r$ to 1e-12:
   - on level 0 of the `hex8`-web mesh, the stack, the T-junction, and a coarse level;
   - symmetric to 1e-12;
   - balanced multi-chunk (small `chunk_bytes`) against single chunk to 1e-14 relative (the
     scatter order differs);
   - a padded class against an unpadded one to 1e-14 relative;
   - padded slots inert (exactly 0 contribution);
   - the class cap: a level with more than 4 natural classes is built with 4;
   - a coarse patch above the bound raises (small `chunk_bytes`).
4. **Extraction:** the reference is `A[s][:, s]` of `assemble(mesh, mats)`, i.e. $K + \sigma M$
   with the constrained diagonal $s$, as `amg.py:150`.
   - It is independent of the `Hex8Pattern` source, so it also checks $A_b = A_P$ on `ssh8`.
   - It equals the extracted patches to 1e-12 on every class, on a clamped plate (zero blocks)
     and the T-junction.
   - Level $\ell \ge 1$ against SciPy's $A_\ell$ to 1e-14.
5. **Setup against references**, on the $8^2$ `hex8`-web mesh (levels [486+, 81+, 9+]; identity
   rows present: assert their count > 0) with `max_coarse` 20:
   - $P_\ell$ equals a dense NumPy $(I - \tfrac{\omega}{\rho} D^{-1} A)^{d_\ell} T_\ell$, with ρ
     from `_shell_levels`, to 1e-12; ρ lies within [0.9, 1.1]× the dense $\rho(D^{-1}A)$;
   - $A_{\ell+1}$ equals $P^\top A P$ with the identity rows to 1e-12;
   - the device V-cycle equals a NumPy V-cycle (SciPy levels, dense Schwarz, NumPy Chebyshev
     with the build's λ_max, dense Cholesky) to 1e-10. `cheb` and `vcycle` move from
     `tests/test_amg.py:195-235` to `tests/helpers.py`; only a dense-Schwarz helper is new;
   - the V-cycle is symmetric to 1e-12, positive on the free DOFs, exactly 0 on constrained
     ones;
   - on every level of this mesh and of a clamped $8^2$ plate, the dense
     $\lambda_\max(D^{-1}A_\ell)$ ≤ the Chebyshev upper bound (1.1 × the build's λ_max), and
     identity DOFs have Schwarz eigenvalue exactly 1.
6. **$L/t$ growth (roadmap test, slow):** flat SS plate $128^2$ (99.8k DOF), default
   `max_coarse`.
   - count($L/t$ 1000) ≤ 2 × count($L/t$ 10), for the configuration that ships (prototype
     12 / 21, 1.75×); each count is also gated.
   - Curved panel $128^2$: both counts gated, and the ratio is reported (prototype 11 / 29 at
     `eigsh` σ).
   - If the flat ratio exceeds 2 in implementation, the fallbacks are tried in order: `degree`
     3, then the gate on $256^2$. The docs record the shipped one.
   - The margin is thin (device 21–22 against about 12), so both device counts are measured
     before the gates are fixed, and the ratio and the seed are recorded.
7. **Refinement:**
   - `slow` (about 20 s): flat $L/t = 100$, $64^2$ against $128^2$ count ≤ 1.3× (counts flat at
     $h/t \le 1.6$).
   - Slow (in conftest `LONG`, not `peak`, about 3 min and 4 GB): flat $L/t = 1000$, the
     $256^2$ count ≤ 1.3× the $128^2$ count (prototype 18 vs 21).
8. **Shell cases**, counts gated at $L/t = 50$ ($32^2$ unless stated):
   - flat plate;
   - one element with 3 plies (0/90/0, Srinivas–Rao-type ply table);
   - curved panel;
   - the sweep-built 270° cylinder ($24 \times 8$);
   - T-junction and L-fold ($24^2$), and the 3/3-layer T;
   - the flat plate at $L/t = 300$.

   On the flat plate, Jacobi's PCG count is reported next to AMG's.
9. **Mixed meshes:**
   - the $32^2$ `hex8`-web mesh;
   - the hex-dominant $16^3$ block with an `ssh8` skin (about 15k DOF; hex groups get their own
     9-DOF patches).

   Counts are gated. On the hex-dominant block, the PR 13 hex8 count of the same block without
   the skin is reported.
10. **Constraints and parts:**
    - clamped-edge plate (full fibers constrained) and a bottom-face-supported plate (partly
      constrained fibers): V-cycle SPD on free DOFs, exactly 0 on constrained ones, counts
      gated (prototype clamped $64^2$: 11 / 24);
    - two separate supported plates of very different sizes ($4^2$ and $32^2$), with a
      `max_coarse` below about 130 so that the small plate becomes one isolated group while the
      large one keeps coarsening (the value is stated in the test): a singleton aggregate
      appears in the stored group → aggregate map, and the count is gated. Equal plates reach
      isolation together, so CAP or STALL stops first (review);
    - a free $24^2$ plate at σ = $\lambda_7$ from `eigsh` (above the 6 rigid modes): V-cycle
      SPD, count gated.
11. **Stacks:** the 3-layer $32^2$ at $L/t$ 10 / 100, counts gated (prototype 11 / 15); 300 /
    1000 reported (40 / 117).
12. **Reuse:**
    - `aggregates=` with new materials, $s_K$ or σ equals a fresh setup (V-cycle to 1e-12);
    - a spy confirms fibers, aggregation and fits are skipped, and the patch sets are rebuilt;
    - a connectivity or mask change raises;
    - two fresh builds are equal exactly.
13. **Hex path unchanged:**
    - `test_amg.py` passes, unmodified except for importing `cheb`/`vcycle` from helpers;
    - a spy (monkeypatch `_shell_levels` and `Schwarz` to raise) runs a hex20 and a hex8 `amg`
      build and their PCG;
    - `test_shell_ops.py`'s `amg` refusal block becomes the mixed-type refusal on an `ssh8` +
      `hex20` mesh, its `pmg` block checks the new message, and the `solve_modes`/`amg_memory`
      refusals stay (18b).

**Runtime:**
- Slow: tests 6 (including about 10 s of $64^2$ `eigsh` for the panel σ, about 2–3 min) and 7
  (20 s and about 3 min), about 6 min together.
- The rest are fast (≤ 30 s each, plates ≤ $64^2$, plus the $16^3$ block).
- No `serial` or `peak` tests: no wall-time or memory-peak assertions in 18a.

## Docs (18a)

All statements that 18a makes false are updated. 18b updates the `solve_modes`-facing ones.
- **`docs/preconditioners.md`:**
  - a new section "AMG on solid shells":
    - the causes and the measured tables;
    - fibers, junctions and the size guard;
    - sign-free frames and the curvature candidates (LaTeX as above), with their per-level
      projection;
    - $d_\ell$ and the one-ring adjacency;
    - identity rows (inert, own-patch only);
    - column, ring and hex-group patches, with their bounds;
    - block-key chunked extraction, packed inverses, size classes and balanced chunks;
    - the stack storage formula $24(k+1)$;
    - the storage and cost table and the setup time;
    - the scope: `ssh8` and `hex8` only, `solve_modes` from 18b.

    Its source line is `benchmarks/shell_amg_study.py` → `benchmarks/results/shell_amg_study.json`,
    as :168/:301/:423.
  - the module table (`precond/schwarz.py`, `precond/shell.py`) and the Tests section
    (`tests/test_shell_amg.py`);
  - the AMG hierarchy list (:369–:372: a solid-shell entry);
  - `AMGAggregates` (:416: shell fields, no drop);
  - pmg's flat-element note (:343);
  - GMG's thin-plate remedy (:208);
  - `_inverse_blocks` (:54: "and PR 18a's Schwarz patches");
  - the AMG flat-element limitation, which points to the new section.
- **`docs/mesh.md`:** :162–:166 (`solid_shell`'s `direction`):
  - a wrong thickness axis is not checked by `solid_shell`;
  - AMG raises on level-0 patches of more than 384 DOFs and otherwise converges worse;
  - T-junctions and L-folds are fine, and multi-layer junctions up to 6 layers per member.
- **`docs/ROADMAP.md`:**
  - the status paragraph (:18): the merged-PR history stays, and the 18a status is appended;
  - the package layout (:226: `schwarz`, `shell`);
  - the Preconditioners key decision (:121–:123: the shell variant);
  - the split, CalculiX, dependency and deferred GPU entries (above);
  - new deferred GPU checks: the Schwarz scan and packed vs full inverses (above), the shell AMG
    at 10M (about 19–20 GB fp64), the host setup peak;
  - "XLA kernel temporaries".
- **`docs/open_items.md`:**
  - every `→ PR 18` rewritten to its part (above);
  - :104;
  - every decision under Risks logged with its measurement;
  - "solid-shell AMG on `ssh8` + `hex20`/`hex27`: not supported (nonconforming)", with no
    owner.
- **Docstrings and comments:**
  - `amg` (`amg.py:146`, docstring :151–:155, including the near-null-space sentence);
  - the module docstring (`amg.py:1-3`);
  - `AMGAggregates` (`amg.py:131–133`: fibers, group → aggregate maps, zero-column flags);
  - `_check` (`pmg.py:219`, refusal :227–:228) and `no_shell` (:214–:216).
- All retags listed under "PR 18 split".

## Measured (prototypes, 18a planning)

Host SciPy/pyamg, fp64, scripts in
`/private/tmp/claude-501/-Users-tairoon-projects-jax-explicit/0381ac12-7b3d-4224-ac3d-6fd4d9e44c2b/scratchpad/pr18/`.
That is the session scratchpad, which is not kept; the numbers are recorded here.
- `proto_cheb.py`, `proto_cheb2.py`: the pyamg hierarchy with smoother variants;
- `proto_impl.py`: revision 1;
- `proto_rev.py`: revision 2 and later.

SS plate, one layer unless stated, PCG to $10^{-6}$ at σ = $\lambda_1$ (`eigsh`), random
right-hand side.

| configuration | n | MC | $L/t$ = 10 / 100 / 300 / 1000 |
|---|---|---|---|
| fiber-block Jacobi, coarse 9×9 block | 32 | 300 | 10 / 17 / 51 / 183 |
| element patches, coarse 9×9 block | 32 | 300 | 11 / 13 / 16 / 38 |
| vertex patches, coarse 9×9 block | 32 | 300 | 11 / 12 / 15 / 24 |
| element patches, coarse 9×9 block | 64 | 300 | 12 / 14 / 35 / 79 |
| element patches, two-level | 64 | 300 | – / – / 15 / 22 |
| element patches, coarse vertex (matrix graph), global bending | 64 | 300 | 12 / – / 18 / 24 |
| element patches, coarse edge patches | 64 | 300 | 12 / – / 33 / 78 |
| one step, fine-level frames only | 64 | 300 | 12 / – / 41 / 80 |
| one step, global bending; per-level frames | 64 | 300 | – / – / – / 26; – / – / – / 26 |
| one step, per-level frames | 128 | 300 | 12 / 13 / – / 40 |
| curved: global bending / per-level frames, one step | 64 | 300 | 13 / – / – / 96; 10 / – / – / 35 |
| rev. 1 (two steps everywhere, one-ring) | 64 / 128 / 256 | 300 | 26; 11 / 20; 18 at 1000 |
| **rev. 2+**, flat | 64 | 1000 | 12 / 14 / – / 25 |
| **rev. 2+**, flat | 128 | 1000 | 12 / – / – / 21 |
| **rev. 2+**, flat | 256 | 1000 | – / – / – / 18 |
| **rev. 2+**, curved | 64 / 128 | 1000 | 10 / 15 / – / 32; 11 / – / – / 29 |
| **rev. 2+**, clamped | 64 | 1000 | 11 / – / – / 24 |
| **rev. 2+**, 3 layers (column patches) | 32 | 1000 | 11 / 15 / 40 / 117 |
| rev. 1, 3 layers, element patches (review) | 32 | 300 | 10 / 43 / 322 / > 2000 |
| rev. 2+, T-junction / L-fold, patches deduplicated (review) | 32 | 300 | – / 19 / – / 81; – / 24 / – / 79 (flat plate: 14 / 39) |

**Storage (revision 2+):** 48–50 + 55–66 packed floats per DOF of patch inverses (level 0 +
coarse) at $n$ = 64–256. The coarse matrices sum to 0.35 of the fine nonzeros. The device
totals are in the Cost table.

**Hex regions** on the shell path (review, pure `hex8` cubes with the shell-path rules):
925 / 976 / 1026 B per corner DOF at 24³ / 36³ / 48³, against PR 13's 777 / 807 / 826
(1.19–1.24×).

**Flatness, not $L/t$:** at fixed $L/t = 1000$ the counts fall with refinement: 25 / 21 / 18 at
$n$ = 64 / 128 / 256, $h/t$ = 15.6 / 7.8 / 3.9. They follow $h/t$, so the roadmap ratio
depends on the mesh: 2.1× at $n = 64$, 1.75× at $n = 128$. The right-hand-side seed changes
counts by ±1 (review).

## Risks and decisions without user input (all logged in `docs/open_items.md`)

- **The split (rev. 8):** 18a is the core and 18b is `solve_modes` and `amg_memory`, along
  PR 13's line. Until 18b merges, `solve_modes` on `ssh8` stays Jacobi.
- **`ssh8` + `hex20`/`hex27`:** not supported, since the meshes are nonconforming. `amg()`
  raises.
- **Memory:** about 1.9–2.0 kB per DOF of preconditioner, 4× the stored-Ke operator: about
  2 GB at 1M and 20 GB at 10M in fp64.
  - Accepted, since Jacobi does not converge at $L/t \ge 50$ in practical counts (it stagnates
    at $32^2$).
  - 18c's fp32 halves the patches.
  - Logged for 18c, not built: $P_0 = S\,T$ applied matrix-free, $P_0^\top$ as a scatter over
    $P_0$'s ELL (about 220 B per DOF), and 1.25× ELL width classes (about 100 B per DOF).
- **Stack storage** grows linearly with the layers: about $24(k+1)$ packed floats per DOF on
  level 0 (about 2.1 kB per DOF at 10 layers). Documented; the patch guard allows it.
- **Hex regions** of mixed meshes get 9-DOF block patches on coarse levels, not one-rings (which
  would cost about 3.2 kB per DOF there), and shell one-rings skip hex neighbours. Hex regions
  still cost about 1.19–1.24× PR 13's hex8 hierarchy (9 coarse DOF per aggregate, BSR(9)).
  Test 9 reports the PR 13 hex8 count of the same block.
- **GPU scan steps:** the Schwarz apply is a scan over chunks, and its per-step overhead is a
  deferred GPU check with a pass criterion. The lever is the internal `chunk_bytes`, a public
  option in 18c if the check needs it.
- **Element flatness:** counts grow with $h/t$ (18 / 21 / 25 at 3.9 / 7.8 / 15.6, $L/t$ 1000).
  The roadmap gate is taken on $128^2$, and the docs state this.
- **Curved panel:** ratio 2.6× against the roadmap's 2×, which the flat plate meets. It is
  reported, and its counts are gated. The documented fallback is
  `precond_options={"degree": 3}`, measured in implementation.
- **Stacks and junctions:** stacks at $L/t \ge 300$ (40 / 117) and junctions at
  $L/t = 1000$ (about 80 against 39) stay above flat single layers.
  - That is a documented limitation.
  - Laminates inside one element (plies) are the single-layer case.
  - Logged, not built: fine vertex patches over columns, a line smoother through stacks,
    fold-aware aggregation.
- **Vertex patches on level 0** (5× the storage for 1.6× fewer iterations) are not used.
- **Folds:** the sign-free normal of an aggregate across a fold is an average. Rigid modes stay
  exact; only the curvature columns degrade (measured above).
- **Guards:**
  - a level-0 patch with more than 384 DOFs raises, and a coarse patch above
    $\sqrt{\texttt{chunk\_bytes}/8}$ raises;
  - a consistently wrong axis on smaller meshes is not detected.
- **Identity DOFs** are kept out of neighbours' ring patches, so their Schwarz eigenvalue is 1.
- **σ-round reuse** is deferred to 18c.
