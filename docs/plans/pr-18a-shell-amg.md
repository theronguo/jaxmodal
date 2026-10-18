# PR 18 plan: performance pass, part a: solid-shell AMG

Branch `pr-18a-shell-amg` (this plan), from `main` after PR 17. Roadmap: PR 18 (M5), the
"Thin plates and slender beams" entry of `docs/open_items.md` (PR 11a → PR 18), and the
PR 16 rule "until PR 18, `solve_modes` on solid-shell meshes uses Jacobi".

## Revisions

- **Revision 1:** draft.
- **Revision 2** (round 1, all five not satisfied; three asked for design changes):
  - **Level 0:** column patches (the union of an element's 4 fibers) instead of element
    patches. Multi-layer stacks need them; on one layer they are identical.
  - **Normals:** sign-free (dominant eigenvector of $\sum v v^\top$). `solid_shell`'s
    sweep flips element orientation on curved shells.
  - **Prolongator smoothing:** one step on level 0, two below. Same counts, 49% fewer $P_0$
    nonzeros and 63% fewer level-1 nonzeros.
  - **Coarse levels:** dropped tentative columns become decoupled identity rows, so coarse
    levels keep 9×9 blocks. Export with $b = 3$ ($P_0$) and $b = 9$ (coarser).
  - **Patch inverses:** stored packed (upper triangle) and inverted in fixed chunks.
  - **hex20/hex27 + `ssh8`:** the Schwarz goes on the existing assembled corner level (exact).
    The p-level is unchanged, so the approximate interface patches are gone.
  - **Patch extraction:** one vectorized mechanism (`searchsorted` keys), run before any
    in-place compaction of $A_b$.
  - **Aggregation:** isolated groups with free DOFs become singleton aggregates. A fiber
    check raises on wrong thickness axes.
  - **Reuse between σ rounds:** `solve_modes`' σ pre-estimate reuses the aggregates
    (exact).
  - **Storage accounting:** now with transfers, coarse matrices and padding. The V-cycle cost
    and the setup time are measured and documented.
  - **Tests:** hierarchies with ≥ 2 levels asserted; accuracy at moderate $L/t$ with the
    floor asserted; SciPy/pyamg/NumPy references for $P$, the Galerkin products and the
    V-cycle; clamped and bottom-supported plates; stacks; a sweep-built cylinder. The stored
    hex reference is replaced by a spy. Peak/serial marks, sizes and memory are stated.
  - **Docs:** the full list of statements that become false. The roadmap split, the CalculiX
    bullet, dependency rows and owners of every `→ PR 18` open item are recorded.

## PR 18 split

PR 18 is too large for one branch. It splits into four parts, in this order:

| part | branch | contents | plan |
|---|---|---|---|
| 18a | `pr-18a-shell-amg` | AMG on the assembled $A_P$ of solid-shell meshes, iteration counts vs $L/t$; `"auto"` → AMG on `ssh8` meshes | this file |
| 18b | `pr-18b-kernels` | Morton reordering; fp32 for p-MG and AMG (estimator; the 1M Mac acceptance with fp32 AMG; fp32 Schwarz inverses); HLO fusion checks; the compile-time benchmark; the roofline tests; `docs/performance.md`; the performance `→ PR 18` open items | own plan |
| 18c | `pr-18c-band` | the PR 15d band follow-ups (roadmap bullet), the measured time crossover (GPU, if any), the `COUNT_STEP` refit at ≥ 1M DOF (with AMG on `ssh8` since 18a), and the band/count/slice `→ PR 18` open items | own plan |
| 18d | `pr-18d-calculix` | CalculiX comparison (user request, 2026-10-18): CalculiX `*FREQUENCY` against `solve_modes` on the same `.inp` meshes, about 100k–1M DOF, 20 and 200 modes; wall time, peak memory, frequency agreement; ≤ 1 h per run | own plan |

**Ownership of the `→ PR 18` entries in `docs/open_items.md`.** Every entry not owned by 18a
goes to 18b (performance) or 18c (band, count, slices):
- **18b:** `assembled._chunked` pipelining (PR 17); the refill through
  `device_put(V[:, i:i+c].T).T`; caching `layout` for the gather path; host transposes and small
  compiles (code review round 10).
- **18c:** everything else:
  - σ-move retries;
  - span counts at small capacities;
  - σ between the counting edges;
  - verification slices;
  - per-step residuals;
  - merge and count copies;
  - the verify skips before the gather;
  - reuse of locked modes after a capacity stop;
  - store sizing;
  - `t_ref` with compile.

**Order and dependencies:**
- 18a ← 13, 16c;
- 18b ← 11, 12, 13, 18a (fp32 covers the shell AMG);
- 18c ← 15d, 18a (the `COUNT_STEP` refit runs on the post-18a `ssh8` default);
- 18d ← 18a–18c (it measures the final state).

Each later part gets its own reviewed plan, `docs/plans/pr-18x-<slug>.md`, before it starts.
18a records the split in `docs/ROADMAP.md`:
- the PR 18 section gets the four parts and per-part status lines, plus a CalculiX bullet in
  Contents and a CalculiX test;
- the dependency table row "18" becomes rows 18a–18d (19 depends on all of them);
- the deferred GPU check "PR 18: matvec ≥ 50% of roofline" becomes 18b.

## Goal (18a)

`amg()` and `solve_modes(..., precond="amg")` work on meshes with `ssh8` blocks: all `ssh8`,
`ssh8` + `hex8`, and `ssh8` + `hex20`/`hex27`.
- On a simply supported single-layer plate $128^2$, PCG iterations grow ≤ 2× from $L/t = 10$ to
  $L/t = 1000$ (roadmap test).
- Under refinement at fixed $L/t$ the counts do not grow: $256^2$ ≤ 1.3× $128^2$.
- `"auto"` selects AMG on solid-shell meshes.
- `amg_memory` covers solid-shell meshes.
- Hex-only meshes take the unchanged PR 13 path.

## Why the PR 13 AMG fails on solid shells

Measured in the 18a prototypes (Measured below): SS plate $n^2 \times 1$ `ssh8`, PCG to
$10^{-6}$ at σ = $\lambda_1$, Chebyshev degree 2 on every level, dense coarsest solve.

1. **Near-null space.** Rigid modes alone do not represent plate bending on coarse aggregates
   (the PR 13a hex20-plate entry). Three constant-curvature fields are added. They are the
   Kirchhoff displacements $u_\alpha = -Z\,\partial w/\partial X_\alpha$, written in an
   aggregate's local frame $(t_1, t_2, n)$ with local coordinates $(X, Y, Z)$:
   $$\kappa_{11}: w = \tfrac{X^2}{2},\ u_1 = -ZX; \qquad \kappa_{22}: w = \tfrac{Y^2}{2},\ u_2 = -ZY; \qquad
   \kappa_{12}: w = XY,\ u_1 = -ZY,\ u_2 = -ZX.$$
   With the 6 rigid modes each aggregate gets 9 coarse DOF.
2. **Smoothers.** Point and node-block Jacobi lose the thickness coupling. The 6×6 block of a
   through-thickness fiber is not enough once the element is flat ($h/t \gg 1$), because the
   membrane–bending coupling lives in the element. At $n = 32$ and $L/t$ = 10/100/300/1000:
   - fiber-block Jacobi: 10/17/51/183 iterations;
   - element patches: 11/13/16/38;
   - vertex patches: 11/12/15/24.
3. **Coarse levels.** Each coarsening widens aggregates about 3× in-plane at the same thickness,
   so the coarse levels are flatter still. Block Jacobi on 9×9 aggregate blocks fails there as
   fiber blocks fail on level 0. At $n = 64$ and $L/t = 1000$:
   - a two-level cycle: 22 iterations;
   - the V-cycle with coarse 9×9 block Jacobi: 79;
   - with coarse vertex patches: 24;
   - with coarse edge patches: 78.
4. **The bending candidates must be one field per level.** SA passes the coarse candidates down
   as coefficients ($T R = B$).
   - Fine-level frames only (one frame per level-1 aggregate) give piecewise level-1
     candidates, and the next level fits them wrongly: 80 iterations on the flat $n = 64$,
     $L/t = 1000$ plate.
   - Global bending fields give 26.
   - Candidates rebuilt per level from the next aggregates' frames (below) also give 26.
   - On a 90° cylindrical panel, global fields give 96 and per-level frames 35.
5. **Prolongator smoothing.** Bending is fourth order. With one Jacobi step on every level, the
   counts grow with depth (flat, $L/t = 1000$: 26 at $n = 64$, 40 at $n = 128$ with 4
   levels); two steps on every level give 20.
   - Only the coarse levels need the second step. One step on level 0 and two below gives
     21 at $n = 128$ and 19 at $n = 256$ (5 levels), with half the $P_0$ nonzeros and a third
     of the level-1 nonzeros (review measurement).
   - Two steps widen the coarse stencils. Patches and the next aggregation therefore use the
     **one-ring aggregate adjacency** (aggregates adjacent when their members are adjacent one
     level up), not the coarse matrix pattern. Over the matrix pattern, vertex patches cost 852
     floats per DOF; over the adjacency, 114.
6. **Stacks need column patches.** With $k$ elements through the thickness, an element patch
   covers $1/k$ of it (3 layers, $n = 32$: 10/43/322/>2000). A patch per element column (the
   DOFs of an element's 4 fibers) gives 11/15/40/117. Stacks stay weaker than single layers at
   $L/t \ge 300$ (limitation below).

## Design (18a)

### Scope of the shell path

`amg()` takes the shell path when any operator block is `ssh8`. Otherwise the PR 13 code runs
unchanged: no code on the hex path changes behaviour, test 13.

The shell path reuses:
- `_p_level` (`ap_operator`, `_hex8_galerkin`, the `Hex8Pattern`). An `ssh8` element's 8 nodes
  are all corners and `_w8(8)` is the identity. So on `ssh8` and `ssh8` + `hex8` meshes the
  corner-space matrix $A_b$ is the assembled $A_P$ itself.
- `_level`, which gains an optional smoother argument: `_level(A, b, degree, ratio, D=None)`.
- `ELLTransfer`, `block_ell`, `_cholesky`, the `GMG` V-cycle, `STALL`, `CAP`, and the local
  seed.

New code:
- `precond/schwarz.py`: `Schwarz`, the patch extraction and the chunked inversion, about 90
  lines.
- `_shell_levels` in `precond/amg.py`: the setup loop, about 120 lines.
- The `amg()` wiring and `AMGAggregates` fields, about 30 lines.
- `amg_memory`'s shell family, about 25 lines.
- `resolve_precond` and the `pmg` check, about 10 lines.

That is about 280 source lines and 500 test lines. The fit-candidates + `STALL` step is shared
between `_sa_levels` and `_shell_levels` as a helper only if that leaves the hex results
bit-identical (same operations); otherwise each keeps its own.

### Level structure

- **`ssh8` and `ssh8` + `hex8` meshes** (every block 24 DOF; `amg()` already treats them as
  "all hex8"):
  - level 0 is the matrix-free $A_P$ with Chebyshev over the level-0 Schwarz (exact, since
    $A_b = A_P$);
  - its transfer is $P_0$;
  - below come the shell SA levels.
- **Meshes with `hex20`/`hex27` blocks:**
  - the p-level is PR 12/13's, unchanged (Chebyshev over node-block Jacobi, `NodalTransfer`);
  - the corner level is the assembled $A_b$ as `block_ell(A_b, 3)`, with Chebyshev over the
    level-0 Schwarz (exact: $A_b$ is that level's matrix);
  - the shell SA levels follow.
  - On `ssh8` nodes the p-level smoother is a weak extra smoother ahead of an exact-patch level
    (test 9 measures it).

### Groups and aggregation

A **group** is a set of DOFs that is aggregated together.
- **Level 0 (corner space): through-thickness fibers.**
  - A fiber is a connected component (`scipy.sparse.csgraph.connected_components`) of the graph
    whose edges are the node pairs $(a, a+4)$, $a = 0..3$, of every `ssh8` element. Bottom face
    nodes are 0–3 and top 4–7, `docs/elements.md`.
  - One layer gives node pairs; a stack of $k$ elements gives $(k+1)$-node fibers.
  - Corner nodes in no `ssh8` element are singleton groups, so a pure-hex region aggregates
    nodes as PR 13 does.
  - **Fiber check:** if two nodes of one element face (bottom or top) lie in the same fiber,
    the thickness axes are inconsistent (for example a wrong per-element `direction`), and
    `amg` raises `ValueError`.
- **Level $\ell \ge 1$:** the groups are the level-$\ell$ aggregates. Every aggregate keeps 9
  coarse DOF (below), so coarse DOF $j$ belongs to group $\lfloor j/9 \rfloor$.

**Group graph:**
- On level 0: $G^\top |A_b| G$, with $G$ the DOF → group indicator. Diagonal removed, pattern
  only (θ = 0, as PR 13).
- On level $\ell \ge 1$: the one-ring adjacency $\mathrm{Agg}_{\ell-1}^\top (S_{\ell-1} + I)\,
  \mathrm{Agg}_{\ell-1}$. Here $S_{\ell-1}$ is the previous level's group graph and
  $\mathrm{Agg}_{\ell-1}$ its group → aggregate indicator.

**Aggregation:** `standard_aggregation` on the group graph. A DOF's aggregate is its group's.
- A group without neighbours is left unaggregated by pyamg. If it holds a free DOF it becomes
  a singleton aggregate; this keeps separate parts and coarse leftovers on every level.
- Only fully constrained groups stay unaggregated, with empty tentative rows as in PR 13.
- Fine-node membership of a level-$(\ell+1)$ aggregate is composed through the per-level
  group → aggregate maps.

### Candidates

On level $\ell$, after aggregation:
- **$B^{(\ell)}_\text{fine}$** is $(3 n_c, 9)$. Columns 0–5 are the masked rigid modes at the
  corner points (`_nullspace`). Columns 6–8 are the three curvature fields of each corner
  node's level-$(\ell+1)$ aggregate, in that aggregate's frame. Constrained rows are zeroed.
- **Frame (sign-free):**
  - Per node, $T_a = \sum v v^\top$ over the thickness edges $v$ (top minus bottom node,
    4 per element) of the `ssh8` elements that contain it.
  - Per aggregate, $n$ is the dominant eigenvector of $\sum_{a \in \text{agg}} T_a$. The sign
    of $n$ does not matter, since the span of the curvature fields is invariant under $n \to -n$.
  - $t_1 = n \times e / \|n \times e\|$, with $e$ the coordinate axis least aligned with $n$;
    $t_2 = n \times t_1$.
  - $(X, Y, Z)$ are the projections of $x - \bar x$ onto $(t_1, t_2, n)$, with $\bar x$ the
    mean of the aggregate's corner nodes.
  - All of this is computed vectorized: batched `eigh`, `np.add.at`, no per-aggregate loop.
  - The curvature fields are set on every node of the aggregate, hex-region nodes of mixed
    aggregates included.
  - An aggregate whose $\sum T_a$ is zero (no `ssh8` node) gets zero curvature columns.
- **Projection:** $B_\ell = (T_0 T_1 \cdots T_{\ell-1})^\top B^{(\ell)}_\text{fine}$. The
  tentative $T$ have orthonormal (or zero) columns, so this is the least-squares coefficient of
  the fine field in the level's basis. It is exact for the rigid columns, and for the curvature
  columns on flat plates (test 2). Level 0 uses $B^{(0)}_\text{fine}$.
- Then `fit_candidates` → $(T_\ell, R_\ell)$, with all 9 columns per aggregate kept (no drop).
  $R_\ell$ is stored for reuse but not used as the next level's $B$.

### Prolongator and coarse matrices

- $P_\ell = (I - \tfrac{\omega}{\rho} D^{-1} A)^{d_\ell}\, T_\ell$, from
  `jacobi_prolongation_smoother(A, T, None, R, 4/3, degree=d_ℓ, weighting="diagonal")`, with
  $d_0 = 1$ and $d_\ell = 2$ for $\ell \ge 1$. Scalar diagonal $D$, $C$ = `None` (used only when
  filtering).
- $A_{\ell+1} = P^\top A P$, symmetrized.
- A zero column of $T$ (aggregates with fewer than 9 independent candidates: no `ssh8` node,
  or constrained) gives a zero row and column of $A_{\ell+1}$. Its diagonal is set to the mean
  diagonal of the level. Those DOFs are then decoupled, $A_{\ell+1}$ stays SPD, and the coarse
  size stays a multiple of 9.
- This replaces PR 13's zero-column drop on the shell path. Their candidate rows are zero, so
  they never carry near-null energy.
- **Stop rule:** `STALL` on the aggregate count ($9\,n_\text{agg} > 0.9\,n_\ell$), and `CAP` on
  the coarsest size, as PR 13.

### Smoothers: additive Schwarz under Chebyshev

On every SA level, and on the corner level of meshes with hex20/hex27 blocks, Chebyshev
(`degree`, `ratio` as today) runs over an additive Schwarz preconditioner
$$D^{-1} r = \sum_i R_i^\top A_i^{-1} R_i\, r, \qquad A_i = R_i A R_i^\top,$$
whose patches $i$ cover every DOF of the level. Each $A_i$ is a principal submatrix of an SPD
matrix, so the sum is SPD.
- **Level 0 (corner space):**
  - one patch per element column: the DOFs of the 4 fibers of an `ssh8` element. Elements in
    the same column give the same patch, so the patches are deduplicated. One layer gives
    exactly the element's 24 DOFs; $k$ layers give $12(k+1)$.
  - Each corner node in no `ssh8` element gets one node patch (its 3 DOFs, the PR 13 block).
- **Level $\ell \ge 1$:** one patch per group: its 9 DOFs plus those of its one-ring neighbour
  groups (the adjacency above).
- **Extraction** is the same for all levels. Per size class, the $s^2$ (row, col) keys of each
  patch are looked up by `searchsorted` in the level's sorted CSR keys, `_pattern`'s idiom,
  vectorized with no per-patch Python. Missing entries are 0.
  - Level 0 is extracted from $A_b$ **before** any in-place compaction (`eliminate_zeros`
    shifts BSR data). The shell path works on a CSR copy, so $A_b$'s data is never moved under
    the extraction.
- **Inversion:** in fixed chunks of patches (host loop, device `_inverse_blocks` per chunk:
  Cholesky, then the inverse). The device transient stays bounded and independent of $n$. A
  non-finite inverse raises `ValueError`, as `_level`.
- **Storage:** the upper triangle, packed (`matfree._packed_index`, as the stored $K_e$):
  $s(s+1)/2$ floats per patch. Single-layer level 0: 300 floats per element.
- **Size classes:** patch sizes sorted ascending, greedily grouped into classes spanning at
  most a factor 1.25 and padded to the class maximum. Padded slots carry index $n$, dropped on
  scatter, and an identity diagonal with a zero mask. On the structured plate, level 0 is one
  class ($s = 24$) and the coarse levels have about 3 (36/54/81).

### `Schwarz` (device)

A registered pytree with, per class, `idx (n_p, s) int32`, `inv (n_p, s(s+1)/2)` and the mask.
Apply on $(n, k)$ blocks: a `lax.scan` over fixed-size chunks of patches, chunk ≈ $2^{16}/s$
patches. Per chunk it does:
- gather $r[\text{idx}]$;
- unpack the inverse by `_packed_index`;
- `einsum`;
- scatter-add into the output.

The temporaries are bounded by chunk·$s$·$k$ plus one unpacked chunk, an $n$-independent term in
`amg_memory`'s $K$.

### Cost estimates (prototype numbers; replaced by measurements in the docs)

Per fine DOF on the flat single-layer plate (fp64):
- level-0 patches 48 floats (384 B) and coarse patches about 55 floats (440 B), both packed;
- coarse matrices: the prototype's level nonzeros sum to about 0.35 of the fine assembled
  nonzeros, about 140 B;
- transfers $P$ and $P^\top$ (b = 3/9 ELL): about 550 B;
- with ELL padding, about **1.6–2 kB per DOF**: about 1.8 GB at 1M DOF, against 0.45 kB/DOF
  for the stored-Ke operator.

V-cycle cost, from review measurements at $n = 256$ with $m = 15$, in units of $t_A$ (one
stored-Ke $A_P$ apply):
- level-0 Schwarz 0.56;
- coarse $s = 81$ patches about 0.5;
- with $b = 3$/$9$ blocks and one level-0 smoothing step, the whole cycle about 10 $t_A$ (hex20
  AMG: 5.4).

Host setup at 396k DOF, from the review measurement with two steps everywhere: about 15 s and
1.4 GB beyond $A_b$, about 3.6 kB per corner DOF. The design here is cheaper.

Implementation measures and documents:
- the setup time;
- the per-component storage;
- the V-cycle cost in $t_A$;
- `solve_modes` wall time against Jacobi (test 8).

### API and defaults

- **`amg(op, free, sigma, points, ...)`:** signature unchanged; the shell path is internal.
  - `return_aggregates`/`aggregates=` work on it. `AMGAggregates` gains, per level, the
    tentative $(T_\ell, R_\ell)$ (as PR 13), the group arrays and the patch index arrays and
    classes. The `Hex8Pattern`, mask and fingerprint are kept as today.
  - Reuse skips fibers, aggregation, the frames and fits, and the patch index sets. It
    recomputes $A_b$, the smoothed prolongators, the Galerkin products, the patch values and
    inverses, and the export.
  - Moved points keep the old candidates, as PR 13.
- **`_precond_sigma`:** reuses the first build's aggregates for the later σ rounds, for every
  AMG mesh. This is exact: reuse equals a fresh setup for a new σ (the PR 13b reuse test, σ = 0
  aggregates at σ > 0). It saves up to 4 topology passes per `solve_modes` call.
- **`resolve_precond`:** `"auto"` → `"amg"` on every unstructured mesh, `ssh8` included. The
  refusal of `precond="amg"` on `ssh8` is removed; `"jacobi"` stays selectable.
- **The σ pre-estimate on shells:** `solve_modes` runs the multigrid-refined `_precond_sigma`
  (it skipped it under Jacobi). σ = 0 builds need `nonsingular` as today; free shells start
  from `sigma_estimate`'s bound. Test 8 checks σ ∈ [λ₁, 2λ₁] on a plate with levels.
- **`pmg`** keeps refusing `ssh8` blocks. `_check` gets a `shells` flag (False for pmg, True
  for amg). The message becomes "pmg does not support ssh8 blocks; use amg". `no_shell` stays
  in `pmg.py` for pmg only, and its imports in `memory.py` and `eigen/api.py` go.
- **`examples/m4_shell.py`:** `"auto"` (AMG), iterations printed, plus a Jacobi comparison
  line.

### `amg_memory` on solid-shell meshes

A new family `"shell"` (any `ssh8` block). `_amg_scan` counts the level-0 patches and their
sizes. The terms:
- **storage**:
  - the p-level term, for meshes with hex20/hex27 only, as today;
  - $120\,n_\text{nodes}$ for the mask and the p-level's node-block Jacobi, only where it
    exists today (no double count: on `ssh8`/`ssh8` + `hex8` meshes the level-0 smoother is
    the Schwarz, not `block_jacobi`);
  - the level-0 patches, exactly: packed bytes per class from the scan;
  - `AMG_STORAGE["shell"]` $\cdot\,n_c$ for the coarse patches, matrices and transfers;
  - the coarsest factor term.
- **build / host_build:** $\beta + \alpha_\text{ssh8} n_c$, with β shared and `AMG_BUILD`'s
  dict gaining the key `"ssh8"`. The device transient of the chunked inversion is a fixed term
  added to `build`.
- **step / step_low / count:** `AMG_STEP[("shell", ...)]` and an `AMG_COUNT` entry for the
  shell family, fitted; `AMG_LOOP_K` refitted if the Schwarz loop body changes it.

The constants are fitted as the PR 13b ones (upper envelope +2–4%). Calibration meshes:
- SS plates $64^2$, $128^2$, $256^2$ and a ≥ 1M-DOF plate ($416^2$, about 1.04M DOF; host setup
  about 40 s) at $L/t$ 100;
- a 3-layer stack $64^2 \times 3$;
- a curved panel $128^2$;
- an `ssh8` + `hex20` mesh.

The ≥ 1M point is part of the fit, so the 10M figure is a short extrapolation. The fit range and
the 10M figure go into `docs/open_items.md`.

## Tests (18a)

`tests/test_shell_amg.py` unless noted. Meshes come from a helper `plate(n, Lt, curved=False,
layers=1, bcs="ss")`: `box_mesh((n, n, layers), spacing=(1/n, 1/n, t/layers))` +
`solid_shell(direction=(0, 0, 1))`, with $L = 1$ the in-plane side and $t = L/(L/t)$.
- **Curved:** the box is mapped onto a 90° cylindrical panel of radius 1; $L$ stays the
  straight side 1, and the arc is 1.57.
- **σ** for count tests: the Kirchhoff $\lambda_1$ of the flat SS plate, $D \pi^4 (2/L^2)^2 /
  (\rho t)$ with $D = E t^3 / (12(1-\nu^2))$. Any shift of the right order serves the count
  tests; the same σ is used for the curved panel. No `eigsh` at 100k+ DOF.
- **Counts:** `pcg` to $10^{-6}$ on a fixed-seed random block of 4 columns, max over columns,
  `max_coarse` stated per test. Each gated count is ≤ 1.5× the implementation's measured
  device count, recorded in the docs (the `test_h_independence` pattern).

1. **Fibers:**
   - 1 layer gives node pairs; 3 layers give 4-node fibers equal to the `box_mesh` columns;
   - `ssh8` + `hex8`: the hex-only nodes are singletons;
   - a wrong per-element `direction` (an in-plane thickness axis) raises the fiber-check
     `ValueError`.
2. **Candidates:**
   - **Flat plate:** $\|(I - T_0 T_0^\top) B^{(1)}_\text{fine}\|$ ≤ 1e-10 relative, for the
     rigid and curvature columns; this is the exactness claim. Per aggregate, the level-0
     local-frame span equals the global $[\text{rigid}, \kappa]$ span (principal angles ≤ 1e-10).
   - **Sweep-built 270° cylinder** (`solid_shell(direction=None)`, whose element orientations
     flip): every aggregate normal is within 1e-8 of the radial direction at its centroid
     (sign-free), with no NaN. The curvature fields satisfy the Kirchhoff formulas in the frame
     on 3 aggregates.
3. **Schwarz apply** equals a dense NumPy $\sum_i R_i^\top A_i^{-1} R_i r$ to 1e-12:
   - on level 0 of an `ssh8` + `hex8` mesh, a 3-layer stack and a coarse level;
   - symmetric to 1e-12;
   - chunked against single-chunk to 1e-14;
   - a padded class against an unpadded one to 1e-14;
   - masked padded slots give exactly 0.
4. **Extraction** equals SciPy `A[s][:, s]` to 1e-14 on every class, on a clamped plate, where
   zero blocks occur.
5. **Setup against references** (small plate $8^2 \times 1$, `max_coarse` 20, and an `ssh8` +
   `hex20` mesh):
   - $P_\ell$ equals pyamg `jacobi_prolongation_smoother(degree=d_ℓ, ω=4/3)` on the same
     $T_\ell$ to 1e-12;
   - $A_{\ell+1} = P^\top A P$ plus the identity rows to 1e-12;
   - the device V-cycle equals a NumPy V-cycle (SciPy levels, dense Schwarz, NumPy Chebyshev,
     dense Cholesky) to 1e-10;
   - the V-cycle is symmetric to 1e-12, positive on the free DOFs, and exactly 0 on
     constrained ones.
   - Shell versions of `test_setup_scipy`, `test_pyamg_prolongator` and `test_vcycle_dense`.
6. **$L/t$ growth (roadmap test, slow).** Flat SS plate $128^2$ (99.8k DOF), default
   `max_coarse`:
   - count($L/t$ 1000) ≤ 2 × count($L/t$ 10); prototype (rev. 2) 12 / 21;
   - the curved panel $128^2$ is report-only for the ratio (prototype 11 → 29); both its counts
     are gated as above;
   - if the device flat ratio exceeds 2, the documented fallback is taken (degree 3, or the
     $256^2$ plate) and stated in the docs.
7. **Refinement (slow, `peak`, in conftest `LONG`).** Flat $L/t = 1000$: the $256^2$ count
   (396k DOF) ≤ 1.3× the $128^2$ count (prototype 18 vs 21). The $64^2$ count (25–27, $h/t$
   15.6) is reported.
   - Expected about 3 min and about 4 GB (no `eigsh`, analytic σ); `peak` keeps it out of the
     8-wide stage.
8. **`solve_modes` with AMG against the PR 8 `eigsh`** to the PR 9 tolerances. Each case has an
   explicit mesh, $L/t$, `tol` and `max_coarse`, and asserts `len(P.levels) ≥ 2` and
   `info["precond"] == "amg"` under `"auto"`:
   - flat plate $32^2$, $L/t = 50$, `max_coarse` 100;
   - 3-layer stack $24^2 \times 3$, $L/t = 20$;
   - one element with 3 plies (0/90/0 laminate, Srinivas–Rao-type ply table) $32^2$,
     $L/t = 50$;
   - curved panel $32^2$, $L/t = 50$;
   - a band solve on the flat plate (exercises the λ_c rebuild with aggregates);
   - one $L/t = 300$ case for robustness.
   
   For each case the PR 9 floor $\max(10^{-8}, 10\,\varepsilon\,\lambda_\max/\lambda)$ is
   asserted ≤ 1e-6 (as `test_m4_reference`'s comment), except the $L/t = 300$ case, which
   records its floor. `tol` is 1e-8 (1e-6 at $L/t = 300$).
   - The pre-estimate σ is in [λ₁, 2λ₁] on the flat plate.
   - The flat plate's `solve_modes` iterations and wall time with AMG against Jacobi are
     recorded in the docs. The gate is AMG wall time ≤ Jacobi's (`serial`).
   - The M4 plate (486 DOF, below `max_coarse`: a direct solve) keeps `test_m4_reference`,
     switched to `"auto"` → `"amg"`. The PR 16 band tests on M4 pass `precond="jacobi"`.
9. **Mixed meshes:** a $32^2$ `ssh8` plate with a `hex8` stiffener, and with a `hex20`
   stiffener (p-level path) and a `hex27` one:
   - `solve_modes` with AMG against `eigsh` to the PR 9 tolerances;
   - PCG counts gated ≤ 1.5× measured;
   - Jacobi's count reported.
10. **Constraints:**
    - a clamped-edge plate (whole fibers fully constrained) and a plate supported on the bottom
      face only (partly constrained fibers): V-cycle SPD on the free DOFs, exactly 0 on
      constrained ones, counts gated (prototype clamped $64^2$: 11 / 24 at $L/t$ 10 / 1000);
    - a mesh of two separate plates, each supported: isolated groups become singleton
      aggregates and the count is gated.
11. **Stacks:** the 3-layer plate $32^2$ at $L/t$ 10 / 100, counts gated (prototype 11 / 15).
    $L/t$ 300 / 1000 is reported (40 / 117; limitation).
12. **Reuse:**
    - `aggregates=` with new materials, $s_K$ or σ equals a fresh setup (V-cycle to 1e-12);
    - moved points (2%) stay within 1.2× a fresh setup's count;
    - a spy confirms fibers, aggregation and fits are skipped;
    - a connectivity or mask change raises;
    - setup determinism: two fresh builds equal exactly;
    - `_precond_sigma` with reuse gives the same σ as without (1e-12), on a hex20 box and the
      flat plate.
13. **Hex path unchanged:** `test_amg.py` passes unmodified. A spy (monkeypatch
    `_shell_levels` and `Schwarz` to raise) runs a hex20 and a hex8 `amg` build and their PCG.
14. **`amg_memory`, shell family:**
    - storage ≥ owned bytes and ≤ 1.3× on the calibration plates;
    - build ≥ the host setup peak and ≤ 1.3× (`peak`, subprocess per mesh as PR 13b);
    - the LOBPCG step and count with AMG ≤ `lobpcg_memory`'s (`memory_analysis`);
    - `solve_modes_memory(..., "auto")` on `ssh8` reports AMG;
    - **the ≥ 1M plate** (slow, `peak`, `LONG`): storage and build each within [1, 1.3]× of the
      owned bytes and the host setup peak measured in a subprocess (no `solve_modes`). The plan
      expects about 5 GB host and 2 GB device, about 1–2 min.
15. **Refusals updated** (`test_shell_ops.py`):
    - `test_refusals`: the `amg`, `amg_memory` and `solve_modes(precond="amg")` blocks are
      deleted; `pmg` raises the new message; otf on `ssh8` still raises;
    - the `AMG` constant goes;
    - `test_auto_labels` is rewritten: under `"auto"` (now AMG) `degree` and `sigma_options`
      are valid, so the test asserts an unknown option is labelled "auto (resolved to amg)";
    - `test_auto_jacobi_mixed` becomes `test_auto_amg_mixed`.

Runtime: tests 6, 7 and 14d are slow (about 3–8 min together, 7 and 14d in the `peak` stage);
the rest are fast (≤ 30 s each, plates ≤ $48^2$). New wall-time assertions (test 8's) are
`serial`.

## Docs (18a)

Statements that become false, all updated:
- **`docs/preconditioners.md`:**
  - a new section "AMG on solid shells" covering:
    - the causes and measured tables;
    - fibers and the fiber check;
    - sign-free frames and the curvature candidates (LaTeX as above), and their per-level
      projection;
    - $d_\ell$;
    - the one-ring adjacency;
    - identity rows;
    - column and vertex patches, packed chunked inverses, size classes;
    - the level structure for mixed meshes;
    - storage, V-cycle cost, setup time, `amg_memory`'s shell family;
  - the Jacobi section's "Solid shells" bullet (:53);
  - pmg's flat-element note (:343);
  - GMG's thin-plate remedy (:208);
  - `amg_memory`'s "for a hex8/hex20/hex27 mesh" and calibration range (:470, :476);
  - the AMG flat-element limitation, which points to the new section.
- **`docs/eigensolvers.md`:**
  - the slender-models and "Solid shells" bullets (:40, :41);
  - the `precond` table row (:79);
  - the `max_coarse` row (:88);
  - the σ pre-estimate section (shells now use it).
- **`README.md`:** the status line (:5), the solid-shell line (:82), the laminate snippet
  (its Jacobi-motivated thickness and `maxit`).
- **`docs/elements.md`:** the PR 16 Jacobi note (:218).
- **`docs/ROADMAP.md`:**
  - the status paragraph (:18);
  - :474, :493, :579;
  - the PR 18 split, the CalculiX bullet and test, the dependency rows and the deferred GPU
    entries (above);
  - new deferred GPU checks: the shell AMG at 10M (storage about 18 GB fp64 by `amg_memory`,
    the Schwarz gather/scatter speed, the host setup peak).
- **`docs/open_items.md`:**
  - the thin-plate entry (:18, :22): resolved for solid shells, kept for hex plates;
  - :104 (no thin-plate counting test);
  - the Messages entry (:210) and the README entry (:211);
  - every decision under Risks below is logged with its measurement.
- **Docstrings:** `solve_modes` (`eigen/api.py:311`), `amg` (`amg.py:150`), `amg_memory`
  (`memory.py:338-345`), `no_shell` (`pmg.py:214`), `examples/m4_shell.py:2`.

## Measured (prototypes, 18a planning)

Host SciPy/pyamg, fp64, in
`/private/tmp/claude-501/-Users-tairoon-projects-jax-explicit/0381ac12-7b3d-4224-ac3d-6fd4d9e44c2b/scratchpad/pr18/`
(session scratchpad, not kept; the numbers are recorded here):
- `proto_cheb.py`, `proto_cheb2.py`: the pyamg hierarchy with smoother variants;
- `proto_impl.py`: revision 1's scheme;
- `proto_rev.py`: this revision's scheme.

SS plate, one `ssh8` layer, PCG to $10^{-6}$ at σ = $\lambda_1$ (`eigsh`), random right-hand
side. `max_coarse` (MC) is given per row.

| configuration | n | MC | $L/t$ = 10 / 100 / 300 / 1000 |
|---|---|---|---|
| fiber-block Jacobi, coarse 9×9 block | 32 | 300 | 10 / 17 / 51 / 183 |
| element patches, coarse 9×9 block | 32 | 300 | 11 / 13 / 16 / 38 |
| vertex patches, coarse 9×9 block | 32 | 300 | 11 / 12 / 15 / 24 |
| element patches, coarse 9×9 block | 64 | 300 | 12 / 14 / 35 / 79 |
| element patches, two-level | 64 | 300 | – / – / 15 / 22 |
| element patches, coarse vertex (matrix graph), pyamg hierarchy, global bending | 64 | 300 | 12 / – / 18 / 24 |
| element patches, coarse edge patches | 64 | 300 | 12 / – / 33 / 78 |
| rev. 1, one step, fine-level frames only | 64 | 300 | 12 / – / 41 / 80 |
| rev. 1, one step, global bending | 64 | 300 | – / – / – / 26 |
| rev. 1, one step, per-level frames | 64 | 300 | 12 / – / – / 26 |
| rev. 1, one step, per-level frames | 128 | 300 | 12 / 13 / – / 40 |
| rev. 1, two steps, matrix-graph patches | 128 | 300 | 11 / – / – / 24 |
| rev. 1 (two steps, one-ring adjacency) | 64 / 128 / 256 | 300 | 11 / – / – / 26; 11 / – / – / 20; – / – / – / 18 |
| rev. 1, curved, global bending | 64 | 300 | 13 / – / – / 96 |
| rev. 1, curved, per-level frames, one step | 64 | 300 | 10 / – / – / 35 |
| rev. 1, curved | 64 / 128 | 300 | – / – / – / 32; 10 / – / – / 29 |
| **rev. 2**, flat | 64 | 1000 | 12 / 14 / – / 25 |
| **rev. 2**, curved | 64 | 1000 | 10 / 15 / – / 32 |
| **rev. 2**, clamped | 64 | 1000 | 11 / – / – / 24 |
| **rev. 2**, flat | 128 | 1000 | 12 / – / – / 21 |
| **rev. 2**, curved | 128 | 1000 | 11 / – / – / 29 |
| **rev. 2**, flat | 256 | 1000 | – / – / – / 18 |
| **rev. 2**, 3 layers (column patches) | 32 | 1000 | 11 / 15 / 40 / 117 |
| rev. 1, 3 layers, element patches (review) | 32 | 300 | 10 / 43 / 322 / > 2000 |
| rev. 1 with one level-0 step (review) | 64 / 128 / 256 | 1000 | 27; 12 / 21; 19 at $L/t$ 1000 |

Revision 2 stores 48–50 + 55–66 packed floats per DOF of patch inverses (level 0 + coarse) at
$n$ = 64–256. Its level-1 matrix has 338k nonzeros at $n = 64$ (revision 1: 876k), and the
coarse matrices sum to 0.35 of the fine nonzeros. At $n = 256$: 4 levels, $P$ nonzeros
9.8M / 3.2M / 0.8M, host setup 19 s in the prototype, which still has a per-patch Python
loop.

At fixed $L/t = 1000$ the counts fall with refinement (26 / 20 / 18 at $n$ = 64 / 128 / 256,
$h/t$ = 15.6 / 7.8 / 3.9). They follow the element flatness $h/t$, not $L/t$ itself, so the
roadmap ratio depends on the mesh: 2.4× at $n = 64$, 1.8× at $n = 128$. The seed (0–4) changes
the counts by ±1 (review).

## Risks and decisions without user input (all logged in `docs/open_items.md`)

- **Memory:** about 1.6–2 kB per DOF of preconditioner (patches, coarse matrices, transfers),
  4× the stored-Ke operator and about 1.8 GB at 1M DOF in fp64. Accepted: Jacobi does not
  converge at $L/t \ge 300$ in practical counts. 18b's fp32 halves the patches. If memory binds
  at 10M, apply $P_0 = S\,T$ matrix-free (one $A$ apply per transfer, only $T$ stored); that is
  logged, not built.
- **Element flatness:** counts grow with $h/t$ (18 / 20 / 26 at $h/t$ = 3.9 / 7.8 / 15.6,
  $L/t = 1000$). The docs state it, and the roadmap gate is taken on $128^2$.
- **The curved-panel ratio** (2.9×) is above the roadmap's 2×, which the flat plate meets. Per
  "or a fallback is documented", it is reported and its counts are gated. The documented
  fallback is `precond_options={"degree": 3}`, measured in implementation.
- **Stacks** at $L/t \ge 300$ (3 layers: 40 / 117) stay well above single layers. That is a
  documented limitation. Laminates inside one element (plies) are the single-layer case.
  Further remedies are logged, not built: fine vertex patches over columns, or a line smoother
  through the stack.
- **Vertex patches on level 0** (5× the storage for 1.6× fewer iterations) are not used.
- **Folds and junctions:** the sign-free normal is still an average across a fold. Rigid modes
  stay exact; only the curvature columns degrade there. No fold-aware strength in 18a.
- **hex20/hex27 + `ssh8`:** the p-level keeps node-block Jacobi on `ssh8` nodes, followed by the
  exact corner-level Schwarz; test 9 measures it.
- **`amg_memory`'s shell family** is calibrated on plates, a stack, a panel and one mixed mesh,
  up to about 1M DOF.
