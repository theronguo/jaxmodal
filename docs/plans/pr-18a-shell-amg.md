# PR 18 plan: performance pass, part a: solid-shell AMG

Branch `pr-18a-shell-amg` (this plan), from `main` after PR 17. Roadmap: PR 18 (M5), the
"Thin plates and slender beams" entry of `docs/open_items.md` (PR 11a → PR 18), and the
PR 16 rule "until PR 18, `solve_modes` on solid-shell meshes uses Jacobi".

## Revisions

- **Revision 1:** draft.
- **Revision 2** (round 1, all five not satisfied, three asked for design changes):
  - column patches;
  - sign-free frames;
  - one level-0 prolongator step;
  - identity rows and blocked coarse levels;
  - packed, chunked inverses;
  - the corner-level Schwarz for hex20 meshes;
  - full test and docs lists;
  - the PR 18 split.
- **Revision 3** (round 2, all five not satisfied, one design change):
  - **Fibers:** the face check is dropped, because it rejects valid `ssh8` T-junctions and
    L-folds and misses a consistently wrong axis. It is replaced by a fiber-size guard, and
    level-0 patch DOFs are deduplicated (merged junction fibers repeated DOFs).
  - **σ-round aggregate reuse** in `_precond_sigma` moves to 18b: it is a hex-path change with
    test fallout, and it saves about one topology pass.
  - **Patch extraction:** by node-pair or group-pair block keys, inside the chunked loop, with
    no CSR copy. The level-0 group graph comes from the `Hex8Pattern` structure, so it does not
    depend on σ.
  - **Smaller changes:**
    - identity rows detected structurally (zero $T$ columns);
    - balanced chunks;
    - no padding mask;
    - a donated per-class inverse buffer;
    - a constant step term for the unpacked chunk;
    - `pmg`'s refusal inline, `no_shell` deleted.
  - **Numbers:** revision-2 numbers throughout, and the device V-cycle measured by review.
  - **Tests:**
    - the Jacobi wall-time comparison on a case where Jacobi converges;
    - cylinder normals against a NumPy reference with an angular bound;
    - `max_coarse` 100 and $\ge 2$ levels asserted in tests 8–11;
    - an independent NumPy $P$;
    - identity rows asserted on a mixed mesh;
    - an `eigsh` σ for the curved panel;
    - free shells;
    - T-junction and L-fold cases;
    - the sweep-built cylinder solved.
  - **Docs:** the remaining stale statements; every `→ PR 18` entry gets an owner, rewritten in
    `docs/open_items.md` by 18a.
- **Revision 4** (round 3: four satisfied, performance not, three major findings):
  - **Hex groups on coarse levels:** an aggregate with no `ssh8` node gets a patch of its own
    9 DOFs only. One-ring patches in hex regions reach $s \approx 243$, about 3.2 kB per DOF on
    level 1.
  - **Chunks by bytes:** $c = \max(8, \texttt{CHUNK\_BYTES}/(8 s^2))$, with
    `matfree.CHUNK_BYTES`, then balanced. The option is `precond_options={"chunk_bytes": ...}`.
    The step constant is then `CHUNK_BYTES` and needs no coarse $s_\max$.
  - **`amg_memory`:**
    - coarse storage scales with the fibers ($3 n_\text{fib}$), not $n_c$, so stacks are not
      overestimated;
    - hex corners use the PR 13 constant;
    - `host_build` takes the largest α of the types present;
    - device `build` is the inversion chunk's temporaries;
    - `AMG_LOOP_K` stays shared (the hex value is kept).
  - **One extraction function** for both level kinds; the Schwarz byte layout lives in one
    `schwarz.nbytes`, used by the build and `amg_memory`.
  - **Smaller fixes:**
    - fully constrained groups are removed from the level-0 graph;
    - the one-ring is $\mathrm{Agg}^\top S\,\mathrm{Agg}$;
    - pad keys never alias real ones;
    - the case with no SA levels is stated;
    - the prolongator smoothing is the shell path's own few lines, so ρ is recorded;
    - $R_\ell$ is not stored;
    - patch sets are rebuilt on reuse.
  - **Tests:**
    - test 4's reference is `assemble`;
    - hex27 is covered;
    - a ρ check;
    - Jacobi's `maxit` 2000;
    - the band-reuse spy;
    - the cylinder bound uses the full span;
    - junction meshes are glued;
    - darwin skips on peak tests;
    - `test_auto_labels`;
    - a hex-dominant mesh with an `ssh8` skin.
  - **Docs:** the full retag list, the corrected line numbers, `docs/mesh.md`'s `direction`
    note, and the stack storage formula.

## PR 18 split

| part | branch | contents | plan |
|---|---|---|---|
| 18a | `pr-18a-shell-amg` | AMG on the assembled $A_P$ of solid-shell meshes, iteration counts vs $L/t$; `"auto"` → AMG on `ssh8` meshes | this file |
| 18b | `pr-18b-kernels` | Morton reordering; fp32 for p-MG and AMG (estimator, the 1M Mac acceptance with fp32 AMG, fp32 Schwarz inverses); HLO fusion checks; the compile-time benchmark; the roofline tests; `docs/performance.md`; reuse of AMG aggregates across `_precond_sigma`'s σ rounds; the performance and estimator open items (below) | own plan |
| 18c | `pr-18c-band` | the PR 15d band follow-ups of the roadmap bullet (memory levers, $\hat\lambda_\max$, $\delta_\max \le 1$, MINRES remedies near depth 1000, halving before a capacity error, `err.result`, the hybrid junction, $f_1 > 0$ in `solve_modes_memory`, `np.memmap`); the `COUNT_STEP` refit at ≥ 1M DOF (on the post-18a `ssh8` default, AMG); the band, count and slice open items (below) | own plan |
| 18d | `pr-18d-calculix` | CalculiX comparison (user request, 2026-10-18): CalculiX `*FREQUENCY` and `solve_modes` on the same `.inp` meshes (C3D20/C3D8), about 100k–1M DOF, 20 and 200 modes; wall time, peak memory, frequency agreement, a SciPy shift-invert as second reference; ≤ 1 h per run; CalculiX in its own conda env | own plan |

**Owners of the `→ PR 18` entries in `docs/open_items.md`** (18a rewrites each one to name its
part):
- **18a:** the thin-plate entry (:16).
  - Resolved for solid shells.
  - For hex and voxel plates it becomes a documented limitation with no owner (remedy:
    convert to `ssh8`); the heading and ROADMAP :474 are updated.
- **18b** (performance and estimators):
  - `assembled._chunked` pipelining (PR 17);
  - the refill through `device_put(V[:, i:i+c].T).T`;
  - caching `layout` for the gather path (:169);
  - the verify skips before the gather (:255; the roadmap lists them with the 18b items);
  - host transposes and small compiles (:257);
  - `_amg_scan` keeping the mesh alive;
  - `gmg_memory` recomputed per search step;
  - Metal `_default_budget` (:187);
  - new in 18a: $P_0^\top$ applied as a scatter over $P_0$'s ELL (about 220 B per DOF);
    1.25× ELL width classes for $P_0$ (padding 1.45 → about 1.1); tuning the Schwarz chunk on
    GPU.
- **18c** (band, count, slices):
  - :141, :147, :152, :159–:161, :171, :174, :175, :179, :181;
  - :207 (the `COUNT_STEP` refit part) and :226;
  - :186's locked-mode reuse, store sizing and `t_ref` with compile;
  - :245–:257 except :255 and :257;
  - :259–:275 (span counts, shift-invert near depth 1000, σ-move retries, σ between the
    edges).
- **Deferred GPU checks** (ROADMAP list): "XLA kernel temporaries" (:187). The measured time
  crossover stays under the PR 15d deferred GPU checks.

Generic "PR 18" mentions elsewhere are retagged in 18a:
- 18b:
  - `docs/operators.md:204`, `:251` (roofline);
  - `docs/mesh.md:38`;
  - `docs/eigensolvers.md:160`;
  - `docs/conventions.md:89` (compile-time tests);
  - `docs/preconditioners.md:476` (`layout` per call);
  - ROADMAP :168 (memory table), :203, :206 (compile time) and :741;
- 18c:
  - `docs/eigensolvers.md` :126, :152, :172, :178, :621, :723, :764;
  - ROADMAP :176 (memmap);
- `docs/eigensolvers.md:150` (time crossover) → the PR 15d deferred GPU checks;
- `docs/eigensolvers.md:783` and ROADMAP :183 name the parts (18a–18d).

`benchmarks/strategy_study.py:227` and its recorded JSON are historical and stay.

**Dependencies:**
- 18a ← 13, 15d, 16c;
- 18b ← 11, 12, 13, 18a;
- 18c ← 15d, 18a;
- 18d ← 18a–18c;
- 19 ← 18a–18c (18d only measures).

Each later part gets its own reviewed plan, `docs/plans/pr-18x-<slug>.md`. 18a records in
`docs/ROADMAP.md`:
- the split, with per-part status lines;
- a CalculiX bullet in Contents and the 18d test entry (text only, no code in 18a);
- the dependency rows;
- the deferred GPU check "PR 18: matvec ≥ 50% of roofline" retagged 18b.

## Goal (18a)

`amg()` and `solve_modes(..., precond="amg")` work on meshes with `ssh8` blocks: all `ssh8`
(including junctions), `ssh8` + `hex8`, and `ssh8` + `hex20`/`hex27`.
- On a simply supported single-layer plate $128^2$, PCG iterations grow ≤ 2× from $L/t = 10$ to
  $L/t = 1000$ (roadmap test).
- Under refinement at fixed $L/t$ they do not grow: $256^2$ ≤ 1.3× $128^2$.
- `"auto"` selects AMG on solid-shell meshes.
- `amg_memory` covers them.
- Hex-only meshes run the unchanged PR 13 code. `solve_modes`' σ pre-estimate is unchanged
  too: it now also runs on shells, as on every AMG mesh.

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

`amg()` takes the shell path when any operator block is `ssh8`; otherwise the PR 13 code runs
unchanged (test 13). `_sa_levels` and `_shell_levels` stay separate: the shell loop differs in
its group graph, identity rows, $b = 9$ and per-level candidates, so a shared helper would be a
few lines.

New code:

| piece | about |
|---|---|
| `precond/schwarz.py`: `Schwarz`, `_extract`, chunked inversion, size classes, `nbytes` | 100 lines |
| `_fibers`, `_columns` and `_shell_levels` in `precond/amg.py` (setup loop, candidates) | 130 lines |
| `amg()` wiring and `AMGAggregates` fields | 30 lines |
| `amg_memory`'s shell family (calling `_fibers`, `_columns` and `schwarz.nbytes`, no second copy) | 25 lines |
| `resolve_precond`, the `pmg` refusal | 10 lines |

That is about 300–380 source lines, plus about 700–800 test lines.

### Level structure

- **`ssh8` and `ssh8` + `hex8` meshes** (every block 24 DOF; `amg()` already treats them as
  "all hex8"):
  - The shell path calls `ap_operator` and `_hex8_galerkin` directly, not `_p_level` (whose
    `block_jacobi` would be built and dropped).
  - Level 0 is the matrix-free $A_P$ with Chebyshev over the level-0 Schwarz. That is exact:
    an `ssh8` element's nodes are all corners and `_w8(8)` is the identity, so
    $A_b = A_P$.
  - Its transfer is $P_0$; the shell SA levels follow.
- **Meshes with `hex20`/`hex27` blocks:**
  - `_p_level` as today: the p-level with Chebyshev over node-block Jacobi, and the
    `NodalTransfer`.
  - The corner level is `block_ell(A_b, 3)` with Chebyshev over the level-0 Schwarz (exact on
    $A_b$), through `_level(A, b, degree, ratio, D=None)`, which gains the optional smoother.
  - The shell SA levels follow.
  - On `ssh8` nodes the p-level smoother is a weak extra smoother ahead of an exact-patch level
    (test 9).
- **Nothing coarsens** ($3 n_c \le$ `max_coarse`, for example the 486-DOF M4 plate):
  - `ssh8`-only meshes have one level, the dense Cholesky of $A_b = A_P$;
  - hex20/hex27 meshes have the p-level plus the Cholesky of $A_b$, with no corner-level
    Schwarz, as PR 13.

### Groups and aggregation

**Level-0 groups (corner space) are through-thickness fibers.**
- A fiber is a connected component (`scipy.sparse.csgraph.connected_components`) of the graph of
  node pairs $(a, a+4)$, $a = 0..3$, over the `ssh8` elements (bottom face 0–3, top 4–7,
  `docs/elements.md`).
- One layer gives pairs, a stack of $k$ elements gives $(k+1)$-node fibers, and at a
  T-junction or L-fold the two members' fibers merge into one larger group. That is
  legitimate: measured T 19 / L 24 against a flat 14 at $L/t$ 100, $n = 32$.
- Corner nodes in no `ssh8` element are singleton groups.
- **Size guard:** a fiber of more than 64 nodes raises `ValueError` ("ssh8 thickness axes chain
  N nodes; check solid_shell's direction"). A consistently wrong axis, for example
  `direction=(1, 0, 0)` on a plate, makes in-plane chains whose column patches would be dense
  in $n^2$. A wrong axis on fewer nodes than that is not detectable, as the docs state.

**Level $\ell \ge 1$:** the groups are the level-$\ell$ aggregates. Every aggregate keeps 9
coarse DOF, so coarse DOF $j$ belongs to group $\lfloor j/9 \rfloor$.

**Group graph:**
- Level 0: $G^\top S G$, with $S$ the `Hex8Pattern` structure (its `indptr`, `indices`; ones)
  and $G$ the node → group indicator, diagonal removed. The rows and columns of fully
  constrained groups are zeroed, as their off-diagonal values are in the masked $A$ of PR 13
  and the prototype. The graph is mask-dependent but value-independent (σ, materials), and
  `free` is checked on reuse, so reuse is exactly a fresh setup.
- Level $\ell \ge 1$: the one-ring adjacency $\mathrm{Agg}_{\ell-1}^\top S_{\ell-1}\,
  \mathrm{Agg}_{\ell-1}$, diagonal removed, with $S_{\ell-1}$ the previous group graph and
  $\mathrm{Agg}_{\ell-1}$ its group → aggregate indicator. That is a one-ring, not a two-ring.

**Aggregation:** `standard_aggregation` on the group graph.
- Groups left unaggregated (no neighbours) that hold a free DOF become singleton aggregates
  (separate parts, coarse leftovers).
- Fully constrained groups stay unaggregated with empty tentative rows, as PR 13.
- Fine-node membership of a level-$(\ell+1)$ aggregate is composed through the group →
  aggregate maps.

### Candidates

On level $\ell$, after aggregation:
- **$B^{(\ell)}_\text{fine}$**, $(3 n_c, 9)$:
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
  - Aggregates with $\sum T_a = 0$ (no `ssh8` node) get zero curvature columns.
  - Vectorized: `np.add.at`, batched `eigh`, no per-aggregate loop.
- **Projection:** $B_\ell = (T_0 \cdots T_{\ell-1})^\top B^{(\ell)}_\text{fine}$. $T$ has
  orthonormal or zero columns, so this is the least-squares coefficient. It is exact for the
  rigid columns, and for the curvature columns on flat plates and complete-fiber stacks
  (test 2). Level 0 uses $B^{(0)}_\text{fine}$.
- Then `fit_candidates` → $T_\ell$, with all 9 columns per aggregate kept. Its tolerance is
  relative per column, so it is unit-independent, and it zeroes dependent columns exactly
  (review check). $R_\ell$ is neither stored nor used as the next $B$.

### Prolongator and coarse matrices

- $P_\ell = (I - \tfrac{\omega}{\rho} D^{-1} A)^{d_\ell}\, T_\ell$, with $\omega = 4/3$, $D$ the
  diagonal, $\rho$ = pyamg's `approximate_spectral_radius` of $D^{-1}A$ under the local seed
  (the same stream as `jacobi_prolongation_smoother`), $d_0 = 1$ and $d_\ell = 2$ for
  $\ell \ge 1$. It is the shell path's own few lines (pyamg does not return ρ), and ρ is kept on
  the build for test 5.
- $A_{\ell+1} = P^\top A P$, symmetrized.
- **Identity rows:**
  - The zero columns of $T_\ell$ (aggregates with fewer than 9 independent candidates: no
    `ssh8` node, or constrained), detected structurally and stored with the tentatives, give
    zero rows and columns of $A_{\ell+1}$.
  - Their diagonal is set to the mean diagonal of $A_{\ell+1}$'s other rows. The DOFs are
    decoupled, $A_{\ell+1}$ stays SPD, and coarse sizes stay multiples of 9.
  - Their candidate rows are zero, so their $P$ rows on the next level stay zero.
  - A decoupled DOF's eigenvalue under the Schwarz is its patch-overlap count (≤ 9 on the
    plate), inside the Chebyshev interval (review).
  - This replaces PR 13's zero-column drop on the shell path.
- **Stop rules:** `STALL` on the aggregate count ($9\,n_\text{agg} > 0.9\,n_\ell$), `CAP` on the
  coarsest size, as PR 13.

### Smoothers: additive Schwarz under Chebyshev

On every SA level and on the corner level, Chebyshev (`degree`, `ratio` as today) runs over
$$D^{-1} r = \sum_i R_i^\top A_i^{-1} R_i\, r, \qquad A_i = R_i A R_i^\top,$$
with patches $i$ that cover every DOF of the level. Each $A_i$ is a principal submatrix of an
SPD matrix, so $D^{-1}$ is SPD and the V-cycle stays symmetric.
- **Level 0 patches:**
  - per `ssh8` element, the DOFs of its 4 fibers, deduplicated within the patch, and patches
    deduplicated across elements of one column;
  - one layer gives the element's 24 DOFs; $k$ layers give $12(k+1)$;
  - per corner node in no `ssh8` element, a node patch (3 DOFs).
- **Level $\ell \ge 1$ patches:**
  - per group with an `ssh8` node ($\sum T_a \ne 0$), its 9 DOFs plus its one-ring neighbour
    groups';
  - per group without one, its own 9 DOFs (block Jacobi, still stronger than PR 13's scalar
    Jacobi). A one-ring in hex regions has about 22 neighbours ($s \approx 210$–243, about
    3.2 kB per DOF on level 1, review), against 7.7 on plates.
- **Stacks:** a column patch has $s = 12(k+1)$, so level 0 costs about $24(k+1)$ packed floats
  per DOF: 96 at $k = 3$, 264 at $k = 10$ (about 2.1 kB per DOF). `amg_memory` counts it
  exactly, and the docs state the formula.
- **Extraction by block keys, in chunks:** one `_extract(A_bsr, patches, chunk)` for both level
  kinds.
  - A patch is a list of block indices: nodes in $A_b$ as BSR(3) (which is built on the
    `Hex8Pattern`'s `indptr`/`indices`, `pmg.py:207`) on level 0, groups in $A_\ell$ as BSR(9)
    below.
  - Keys are `row * nb + col` from the BSR's `indptr`/`indices`. `searchsorted` hits are
    clipped and checked for equality, then `data` is read and expanded to DOFs by
    `b * blk + arange(b)`.
  - Padded block slots skip the lookup, so a pad key never aliases a real pair.
  - Missing pairs are 0 and explicit zero blocks extract as zeros.
  - There is no CSR copy and no in-place compaction of $A_b$ on the shell path.
  - Extraction runs inside the same fixed-chunk host loop as the inversion, so its transient is
    bounded and independent of $n$ (review: one-shot scalar keys cost about 3.6 kB per DOF).
- **Inversion:** per chunk, `_inverse_blocks(B, idx < n)` on the device (Cholesky, then the
  inverse; padded slots get the identity and their cross terms are zeroed by the mask). The
  padded rows and columns of the inverse are then zeroed, and the
  result is packed and written into a preallocated per-class buffer by a donated update. A
  non-finite inverse raises `ValueError`, as `_level`.
- **Storage:** the upper triangle, packed (`matfree._packed_index`), $s(s+1)/2$ floats per
  patch: 300 per single-layer element.
- **Size classes:** sizes sorted ascending, greedily grouped into classes spanning ≤ 1.25×,
  padded to the class maximum. Padded slots carry index $n$ (gather `mode="fill"` with 0,
  scatter `mode="drop"`) and zero inverse rows, so no mask is stored.
  - The plate's level 0 is one class ($s = 24$).
  - Its level 1 has 27/45/63/81, with 95% of the patches at 81.
- **Chunks** are sized by bytes and balanced:
  - the target is $c = \max(8, \lfloor \texttt{chunk\_bytes}/(8 s^2) \rfloor)$, with
    `chunk_bytes` defaulting to `matfree.CHUNK_BYTES` (32 MiB, the operator's Ke chunk rule);
  - then $n_\text{chunks} = \lceil n_p/c \rceil$ and chunk = $\lceil n_p / n_\text{chunks}
    \rceil$, rounded up to 8;
  - padding stays below 8 patches per chunk (review: unbalanced chunks inflated small classes
    5×);
  - level 0 at $s = 24$ gets about 7280 patches per step, like the hex8 Ke chunk.
  - `chunk_bytes` is an `amg` option, reachable through `precond_options`. The scan-step overhead
    on GPU (about 840 steps per V-cycle at 10M DOF under the old count rule) joins the deferred
    GPU check on Schwarz speed, with tuning in 18b.
- **One layout rule:** `schwarz.nbytes(sizes, dtype, chunk_bytes)` (classes, balanced-chunk
  padding, packed `inv` plus `idx`) is used by the `Schwarz` build for its layout and by
  `amg_memory`.

### `Schwarz` (device)

A registered pytree with, per class, `idx (n_p, s) int32` and `inv (n_p, s(s+1)/2)`. The apply
on $(n, k)$ blocks is a `lax.scan` over the chunks. Each step does the gather, unpacks by
`_packed_index`, runs an `einsum` and scatter-adds.

Temporaries: chunk·$s$·$k$ (gathers), plus one unpacked chunk of chunk·$s^2$ floats, at most
`chunk_bytes`. The unpacked chunk does not depend on $k$ or $s$, so it is a constant term in
`amg_memory` (below).

### Cost (review device measurement, CPU fp64, $m = 15$, rev. 2 hierarchy)

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

- The stored-Ke operator is 0.47 kB per DOF, so the preconditioner is about 4×.
- At 1M DOF that is about 2 GB; at 10M about 20 GB (fp64).
- Device PCG counts reproduce the host prototype: 24 / 21–22 / 18–19 at $n$ = 64 / 128 / 256.
- Packed against full inverses: the unpack costs about 11% of the patch apply.
- Host setup without the per-patch loop: about 4 s at 396k DOF.
- The implementation records its own numbers through `benchmarks/shell_amg_study.py` →
  `benchmarks/results/shell_amg_study.json` (counts, storage, V-cycle cost, setup time, the
  Jacobi comparison; ≤ 1 h).

### API and defaults

- **`amg(op, free, sigma, points, ...)`:** signature unchanged.
  - `return_aggregates`/`aggregates=` work on the shell path. `AMGAggregates` keeps the
    `Hex8Pattern`, mask and fingerprint, plus the fibers and, per level, the group → aggregate
    map, the tentative $T_\ell$ and its zero-column flags.
  - Reuse skips fibers, aggregation, frames and fits. It rebuilds the patch sets (a few
    vectorized lines from the groups) and recomputes $A_b$, the smoothed $P$, the Galerkin
    products, the patch values and inverses, and the export.
  - Moved points keep the old candidates, as PR 13.
  - `solve_modes`' band mode already rebuilds at λ_c with aggregates (`api.py:403-407`); that
    path now covers shells.
- **`resolve_precond`:** `"auto"` → `"amg"` on all unstructured meshes; the refusal on `ssh8`
  is removed; `"jacobi"` stays selectable.
- **σ pre-estimate:** shells now run `_precond_sigma` like every AMG mesh (Jacobi skipped
  it). σ = 0 needs `nonsingular` as today, and free shells start from `sigma_estimate`'s bound.
  Tests 8 and 10 check σ ∈ [λ₁, 2λ₁].
- **`pmg`:** `_check` loses its `ssh8` refusal, `pmg()` raises inline ("pmg does not support
  ssh8 blocks; use amg"), and `no_shell` and its imports (`memory.py`, `eigen/api.py`) are
  deleted.
- **`examples/m4_shell.py`:** `"auto"` (AMG) with iterations printed, plus a Jacobi comparison
  line.

### `amg_memory` on solid-shell meshes

A family `"shell"` (any `ssh8` block). `_amg_scan` calls `_fibers`, `_columns` and
`schwarz.nbytes`, and labels the corner nodes as shell (in an `ssh8` element) or hex.
- **storage:**
  - the p-level term as today (hex20/hex27 meshes only: node-block inverses, its mask and the
    transfer);
  - the free mask, $24\,n_\text{nodes}$ bytes;
  - the level-0 Schwarz exactly, by `schwarz.nbytes`;
  - $\alpha_{P}\,n_{c,\text{shell}}$ ($P_0$, $P_0^\top$; they scale with the corner DOFs);
  - $\alpha_\text{coarse} \cdot 3\,n_\text{fib}$ for levels 1 and below (patches, matrices,
    transfers), with $n_\text{fib}$ the fibers of shell nodes. Below $P_0$ everything scales with
    the in-plane nodes (stack against single layer: $A_1$ 6.2 against 13.4 nonzeros per fine
    DOF, review), so a per-$n_c$ constant would overestimate stacks about 1.4×;
  - `AMG_STORAGE["hex8"]` $\cdot\,n_{c,\text{hex}}$ for hex corners (their coarse patches are 9×9
    blocks);
  - the coarsest factor.
- **host_build:** $\beta + \max(\alpha_\text{shell}, \alpha_t \text{ of the other types present})
  \cdot n_c$. `AMG_BUILD`'s dict gains `"shell"`, β is shared, and the hex constants stay a floor
  on hex20-dominated meshes.
- **build** (device): the inversion chunk's temporaries, about $4 \cdot$ `chunk_bytes` (input,
  factor, inverse, packed copy). The class buffers are already in storage. `build` and
  `host_build` are no longer the same value.
- **step / step_low / count:** `AMG_STEP[("shell", ...)]` and an `AMG_COUNT` entry for shells,
  fitted, plus a constant `chunk_bytes` (the unpacked chunk) independent of $m$. `AMG_LOOP_K`
  stays shared with its hex value (the hex estimates are unchanged), and the shell fit checks
  that it bounds shells at degree 3.

Constants are fitted as in PR 13b (upper envelope +2–4%). Calibration meshes:
- SS plates $64^2$, $128^2$, $256^2$, $416^2$ (1.04M DOF) at $L/t$ 100;
- the 3-layer stack $64^2$ and a 6-layer stack $32^2$ ($\alpha_\text{coarse}$ is fitted on
  plates and stacks);
- the curved panel $128^2$;
- an `ssh8` + `hex20` mesh;
- a hex-dominant mesh: a $24^3$ `hex8` block with an `ssh8` skin;
- a T-junction.

The fit range goes into `docs/open_items.md`.

## Tests (18a)

`tests/test_shell_amg.py` unless noted. Mesh helpers:
- `plate(n, Lt, curved=False, layers=1, bcs="ss")`: `box_mesh((n, n, layers), spacing=(1/n,
  1/n, t/layers))` + `solid_shell(direction=(0, 0, 1))`, with $L = 1$ the in-plane side and
  $t = L/(L/t)$. `curved` maps it onto a 90° panel of radius 1, keeping $L = 1$ as the straight
  side (the arc is 1.57).
- `junction(kind, n, Lt)`: an all-`ssh8` skin with a web (T, or L at an edge), per-element
  `direction`.
- **Gluing:** `Mesh` rejects orphan nodes and there is no merge helper, so the junctions and the
  `hex8`/`hex20` webs are `box_mesh` pieces glued by `np.unique` on rounded coordinates, as
  `helpers.mixed_mesh` does.
  - The skin's in-plane `coords` have one interval of width $t$ at the web, so the web's fiber
    nodes coincide with skin nodes.
  - `direction` is given per element, in sorted selected order.
  - The `hex20` web interface is nonconforming (hanging midside nodes), like `mixed_mesh`. That
    is acceptable algebraically, and test 9 states it.

**Conventions:**
- **σ:**
  - flat plates use the Kirchhoff $\lambda_1 = D\pi^4(2/L^2)^2/(\rho t)$, with
    $D = Et^3/(12(1-\nu^2))$ (within 6% of $\lambda_1$ at $L/t = 10$, 0.2% at ≥ 300; counts
    identical to `eigsh` σ, review);
  - curved panels and junctions use an `eigsh` $\lambda_1$ (σ = 0, tol 1e-3) on the $64^2$
    version of the same geometry, which is cheap (counts are insensitive to σ within a few %;
    `eigsh` on $128^2$ costs about 26 s and 2.6 GB). The flat σ would be 10–190× below the
    panel's $\lambda_1$.
- **Counts:** `pcg` to $10^{-6}$ on a fixed-seed 4-column block, max over columns (spread ≤ 3).
  Each gated count is ≤ 1.5× the implementation's measured device count, recorded in the docs.
- **Hierarchy:** tests 8–11 build with `max_coarse=100` (passed to `solve_modes` through
  `precond_options`), and assert `len(P.levels) ≥ 2` on an `amg` build with the same options.

1. **Fibers:**
   - 1 layer gives pairs;
   - 3 layers give the `box_mesh` columns;
   - `ssh8` + `hex8`: hex-only nodes are singletons;
   - T-junction: the joint fiber merges, with no error, and the level-0 patches have no
     repeated DOF;
   - `direction=(1, 0, 0)` on a $70^2$ plate (71-node chains) raises the size-guard
     `ValueError`.
2. **Candidates:**
   - **Flat plate:** $\|(I - T_0 T_0^\top) B^{(1)}_\text{fine}\|$ ≤ 1e-10 relative (rigid and
     curvature columns). Per aggregate, the level-0 span equals the global
     $[\text{rigid}, \kappa]$ span (principal angles ≤ 1e-10). Same on the 3-layer stack.
   - **Sweep-built 270° cylinder** (`solid_shell(direction=None)`, orientations flip):
     - the vectorized normals equal a per-aggregate NumPy `eigh` of $\sum v v^\top$ up to sign,
       to 1e-12;
     - no NaN;
     - $|n \cdot \hat r(\bar x)| \ge \cos$ of the aggregate's full angular span (both lie inside
       it, with different weights);
     - the same cylinder built with consistent orientations gives the same curvature span
       (principal angles ≤ 1e-12).
3. **Schwarz apply** equals a dense NumPy $\sum_i R_i^\top A_i^{-1} R_i r$ to 1e-12:
   - on level 0 of an `ssh8` + `hex8` mesh, the stack, the T-junction, and a coarse level;
   - symmetric to 1e-12;
   - balanced multi-chunk against single chunk to 1e-14;
   - a padded class against an unpadded one to 1e-14;
   - padded slots inert (exactly 0 contribution).
4. **Extraction:** `A[s][:, s]` of `assemble(mesh, mats)` ($K - \sigma M$, masked; independent of
   the `Hex8Pattern` source, so it also checks $A_b = A_P$ on `ssh8`) equals the extracted
   patches to 1e-12 on every class, on a clamped plate (zero blocks) and the T-junction. Level
   $\ell \ge 1$ against SciPy's $A_\ell$ to 1e-14.
5. **Setup against references,** on an $8^2$ plate with a `hex8` web (levels [486+, 81+, 9+];
   identity rows present: assert their count > 0) and `ssh8` + `hex20` and `ssh8` + `hex27`
   meshes (parametrized), `max_coarse` 20:
   - $P_\ell$ equals a dense NumPy $(I - \tfrac{\omega}{\rho} D^{-1} A)^{d_\ell} T_\ell$, with ρ
     read from the build, to 1e-12; ρ lies within [0.9, 1.1]× the dense $\rho(D^{-1}A)$;
   - $A_{\ell+1}$ equals $P^\top A P$ with the identity rows to 1e-12;
   - the device V-cycle equals a NumPy V-cycle (SciPy levels, dense Schwarz, NumPy Chebyshev
     with the build's λ_max, dense Cholesky) to 1e-10. `cheb` and `vcycle` move from
     `tests/test_amg.py:195-235` to `tests/helpers.py`; only a dense-Schwarz helper is new;
   - symmetric to 1e-12, positive on the free DOFs, exactly 0 on constrained ones.
6. **$L/t$ growth (roadmap test, slow):** flat SS plate $128^2$ (99.8k DOF), default
   `max_coarse`.
   - count($L/t$ 1000) ≤ 2 × count($L/t$ 10), for the configuration that ships (prototype
     12 / 21, 1.75×); each count is also gated.
   - Curved panel $128^2$: both counts gated; the ratio is reported (prototype 11 / 29 at
     `eigsh` σ).
   - If the flat ratio exceeds 2 in implementation, the fallbacks are tried in order
     (`degree` 3, then the gate on $256^2$), and the docs record the shipped one.
7. **Refinement:**
   - `slow` (about 20 s): flat $L/t = 100$, $64^2$ against $128^2$ count ≤ 1.3× (counts flat at
     $h/t \le 1.6$).
   - Slow (in conftest `LONG`, not `peak`, about 3 min and 4 GB): flat $L/t = 1000$, the
     $256^2$ count ≤ 1.3× the $128^2$ count (prototype 18 vs 21).
8. **`solve_modes` with AMG against the PR 8 `eigsh`** to the PR 9 tolerances, `tol` 1e-8,
   `info["precond"] == "amg"` under `"auto"`:
   - flat plate $32^2$, $L/t = 50$;
   - 3-layer stack $24^2 \times 3$, $L/t = 20$;
   - one element with 3 plies (0/90/0, Srinivas–Rao-type ply table) $32^2$, $L/t = 50$;
   - curved panel $32^2$, $L/t = 50$;
   - the sweep-built 270° cylinder ($24 \times 8$, $L/t = 50$);
   - T-junction and L-fold ($24^2$, $L/t = 50$);
   - a band solve on the flat plate (the λ_c rebuild with aggregates; a spy on `_fibers` and
     the aggregation asserts they run once);
   - the flat plate at $L/t = 300$ with `tol` 1e-6: its floor (8.2e-6 for $\lambda_1$) is
     asserted < 1e-3 and recorded.
   
   Every other case asserts the PR 9 floor $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$
   ≤ 1e-6 (measured 1e-8). σ ∈ [λ₁, 2λ₁] on the flat plate.
   - **Jacobi comparison (`serial`):** on $32^2$, $L/t = 30$ (Jacobi converges: about 564
     iterations), both runs converge, and AMG's warm second-call wall time ≤ Jacobi's.
   - On $32^2$, $L/t = 50$, Jacobi with `maxit` 2000 stagnates (`converged=False` and
     iterations < 2000; measured stop at 1054) while AMG converges: asserted.
   - The M4 plate (486 DOF: a direct solve below `max_coarse`) keeps `test_m4_reference`,
     switched to `"auto"` → `"amg"`. The PR 16 band tests on M4 pass `precond="jacobi"`.
9. **Mixed meshes:** a $32^2$ `ssh8` plate with a `hex8` web and with a `hex20` web (p-level
   path, nonconforming interface), and the hex-dominant $24^3$ `hex8` block with an `ssh8` skin
   (hex groups get their own 9-DOF patches):
   - `solve_modes` against `eigsh` to the PR 9 tolerances;
   - PCG counts gated;
   - Jacobi's count reported.
10. **Constraints and free shells:**
    - clamped-edge plate (full fibers constrained) and a bottom-face-supported plate (partly
      constrained fibers): V-cycle SPD on free DOFs, exactly 0 on constrained ones, counts
      gated (prototype clamped $64^2$: 11 / 24);
    - two separate supported plates: singleton aggregates appear, the count is gated;
    - a free $24^2$ plate: `solve_modes` against `eigsh` with `n_rigid` = 6, σ from
      `sigma_estimate`'s start in [λ₁, 2λ₁].
11. **Stacks:** the 3-layer $32^2$ at $L/t$ 10 / 100, counts gated (prototype 11 / 15); 300 /
    1000 reported (40 / 117).
12. **Reuse:**
    - `aggregates=` with new materials, $s_K$ or σ equals a fresh setup (V-cycle to 1e-12);
    - a spy confirms fibers, aggregation, fits and patch sets are skipped;
    - a connectivity or mask change raises;
    - two fresh builds are equal exactly.
13. **Hex path unchanged:** `test_amg.py` passes unmodified. A spy (monkeypatch
    `_shell_levels` and `Schwarz` to raise) runs a hex20 and a hex8 `amg` build and their PCG.
14. **`amg_memory`, shell family:**
    - `test_shell_amg_storage`: the level-0 term equals the built `Schwarz`'s leaf bytes
      exactly; storage ≥ owned bytes and ≤ 1.3×, on the $64^2$ plate, the 3- and 6-layer
      stacks, the `ssh8` + `hex20` mesh and the hex-dominant mesh (fast);
    - `test_shell_amg_build_peak` (`peak`, `skipif` not darwin like `test_estimator_large`):
      host_build ≥ the subprocess host setup peak and ≤ 1.3×, on $128^2$;
    - `test_shell_amg_step`: `memory_analysis` of the LOBPCG step and the count recurrence with
      AMG ≤ the estimates, at $m$ = 1 and 15;
    - `solve_modes_memory(..., "auto")` on `ssh8` reports AMG;
    - `test_shell_amg_memory_1m` (slow, `peak`, `LONG`, darwin only): on the $416^2$ plate, storage and
      host_build each within [1, 1.3]× of the owned bytes and the subprocess host peak, no
      `solve_modes` (expected about 2 GB device, 4–6 GB host, 1–2 min).
15. **Refusals and labels** (`test_shell_ops.py`):
    - `test_refusals`: the amg, amg_memory and `solve_modes(precond="amg")` blocks are
      deleted, `pmg` raises the new message, otf on `ssh8` still raises;
    - the `AMG` constant goes;
    - `test_auto_labels`: the `degree` cases (:250–:253, now a valid AMG option) become an
      unknown key, labelled "auto (resolved to amg)"; the `sigma_options` case (:254–:255, now
      valid under AMG) moves to `precond="jacobi"`, which still raises with label "jacobi";
    - `test_auto_jacobi_mixed` becomes `test_auto_amg_mixed`.

**Runtime:** tests 6 (including about 10 s of $64^2$ `eigsh` for the panel σ), 7 and
`test_shell_amg_memory_1m` are slow (about 6–10 min together; the 1M one in the `peak`
stage). The rest are fast (≤ 30 s each, plates ≤ $70^2$).
The Jacobi comparison is `serial`. `test_mac_acceptance`'s timing margin is checked to hold
with the 1M test in the same peak stage.

## Docs (18a)

All statements that become false are updated:
- **`docs/preconditioners.md`:**
  - a new section "AMG on solid shells": the causes and measured tables; fibers, junctions and
    the size guard; sign-free frames and the curvature candidates (LaTeX as above) with their
    per-level projection; $d_\ell$; the one-ring adjacency; identity rows; column and vertex
    patches with block-key chunked extraction, packed inverses, size classes and balanced
    chunks; the level structure for mixed meshes; the storage and cost table; setup time;
    `amg_memory`'s shell family;
  - the module table (`precond/schwarz.py`) and the Tests section (`tests/test_shell_amg.py`);
  - the AMG hierarchy list (:369–:372: a solid-shell entry);
  - `AMGAggregates` (:416: shell fields, no drop) and its use by `solve_modes`' band mode
    (:419);
  - the Jacobi "Solid shells" bullet (:53);
  - pmg's flat-element note (:343);
  - the stack storage formula $24(k+1)$ floats per DOF and the hex-group patch rule;
  - `chunk_bytes`;
  - GMG's thin-plate remedy (:208);
  - `amg_memory` (:471: the hex-only wording, "build and host_build the same value"; :476:
    the calibration range);
  - the AMG flat-element limitation, which points to the new section.
- **`docs/eigensolvers.md`:**
  - :40, :41 (slender models, "Solid shells");
  - the `precond` row (:79);
  - `max_coarse` (:88);
  - the σ pre-estimate section (:438; the text at :459: shells now use it);
  - :160 retag.
- **`README.md`:** the status line (:5), the solid-shell line (:82), the laminate snippet, and
  :227 (`amg` also covers solid shells).
- **`docs/elements.md`:** :218.
- **`docs/mesh.md`:** :162–:166 (`solid_shell`'s `direction`): a wrong thickness axis is not
  checked by `solid_shell`; AMG raises on fibers of more than 64 nodes and otherwise converges
  worse; T-junctions and L-folds are fine.
- **`docs/ROADMAP.md`:**
  - the status paragraph (:18);
  - :474, :493, :579;
  - the package layout (:227: `schwarz`);
  - the Preconditioners key decision (:121–:123: the shell variant);
  - the split, CalculiX, dependencies and deferred GPU entries (above);
  - new deferred GPU checks: the shell AMG at 10M (about 20 GB fp64 by `amg_memory`, the
    Schwarz gather/scatter speed, the host setup peak);
  - "XLA kernel temporaries".
- **`docs/open_items.md`:**
  - the thin-plate entry and :104;
  - the Messages entry (:210: the Jacobi-default explanation no longer applies) and the README
    entry (:211);
  - every `→ PR 18` rewritten to its part (above);
  - every decision under Risks logged with its measurement.
- **Docstrings and comments:**
  - `solve_modes` (`eigen/api.py:311`);
  - `amg` (`amg.py:146`, docstring :151–:153) and the module docstring (`amg.py:1-3`);
  - `_check` (`pmg.py:219`, refusal :227–:228);
  - `amg_memory` (`memory.py:339-345`, and the `host_build=hb` comment at :361);
  - `examples/m4_shell.py:2`.
- Retags: `docs/operators.md:204` and `docs/mesh.md:38` → 18b.

## Measured (prototypes, 18a planning)

Host SciPy/pyamg, fp64, scripts in
`/private/tmp/claude-501/-Users-tairoon-projects-jax-explicit/0381ac12-7b3d-4224-ac3d-6fd4d9e44c2b/scratchpad/pr18/`
(session scratchpad, not kept; the numbers are recorded here):
- `proto_cheb.py`, `proto_cheb2.py`: the pyamg hierarchy with smoother variants;
- `proto_impl.py`: revision 1;
- `proto_rev.py`: revision 2 and 3.

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
| one step, global bending / per-level frames | 64 | 300 | – / – / – / 26 / 26 |
| one step, per-level frames | 128 | 300 | 12 / 13 / – / 40 |
| curved: global bending / per-level frames, one step | 64 | 300 | 13 / – / – / 96; 10 / – / – / 35 |
| rev. 1 (two steps everywhere, one-ring) | 64 / 128 / 256 | 300 | 26; 11 / 20; 18 at 1000 |
| **rev. 2/3**, flat | 64 | 1000 | 12 / 14 / – / 25 |
| **rev. 2/3**, flat | 128 | 1000 | 12 / – / – / 21 |
| **rev. 2/3**, flat | 256 | 1000 | – / – / – / 18 |
| **rev. 2/3**, curved | 64 / 128 | 1000 | 10 / 15 / – / 32; 11 / – / – / 29 |
| **rev. 2/3**, clamped | 64 | 1000 | 11 / – / – / 24 |
| **rev. 2/3**, 3 layers (column patches) | 32 | 1000 | 11 / 15 / 40 / 117 |
| rev. 1, 3 layers, element patches (review) | 32 | 300 | 10 / 43 / 322 / > 2000 |
| rev. 2/3, T-junction / L-fold, patches deduplicated (review) | 32 | 300 | – / 19 / – / 81; – / 24 / – / 79 (flat plate: 14 / 39) |

**Revision 2/3 storage:** 48–50 + 55–66 packed floats per DOF of patch inverses (level 0 +
coarse) at $n$ = 64–256, and the coarse matrices sum to 0.35 of the fine nonzeros. The device
totals are in the Cost table.

**Flatness, not $L/t$:** at fixed $L/t = 1000$ the counts fall with refinement (25 / 21 / 18 at
$n$ = 64 / 128 / 256, $h/t$ = 15.6 / 7.8 / 3.9). They follow $h/t$, so the roadmap ratio
depends on the mesh: 2.1× at $n = 64$, 1.75× at $n = 128$. The right-hand-side seed changes
counts by ±1 (review).

## Risks and decisions without user input (all logged in `docs/open_items.md`)

- **Memory:** about 2.0 kB per DOF of preconditioner, 4× the stored-Ke operator: 2 GB at 1M
  and 20 GB at 10M in fp64. Accepted, since Jacobi does not converge at $L/t \ge 50$ in
  practical counts (it stagnates at $32^2$). 18b's fp32 halves the patches. Logged for 18b, not
  built: $P_0 = S\,T$ applied matrix-free, $P_0^\top$ as a scatter over $P_0$'s ELL (about
  220 B per DOF), and 1.25× ELL width classes (about 100 B per DOF).
- **Stack storage** grows linearly with the layers: about $24(k+1)$ packed floats per DOF on
  level 0 (about 2.1 kB per DOF at 10 layers). Documented; the 64-node guard allows it.
- **Hex regions** of mixed meshes get 9-DOF block patches on coarse levels, not one-rings
  (one-rings would cost about 3.2 kB per DOF there).
- **GPU scan steps:** the Schwarz apply is a scan over chunks; its per-step overhead is a
  deferred GPU check, and `chunk_bytes` is the lever.
- **Element flatness:** counts grow with $h/t$ (18 / 21 / 25 at 3.9 / 7.8 / 15.6, $L/t$ 1000).
  The roadmap gate is taken on $128^2$, and the docs state this.
- **Curved panel:** ratio 2.6× against the roadmap's 2×, which the flat plate meets. It is
  reported, and its counts are gated. The documented fallback is
  `precond_options={"degree": 3}`, measured in implementation.
- **Stacks** at $L/t \ge 300$ (40 / 117) and **junctions** at $L/t = 1000$ (about 80 against
  39) stay above flat single layers. That is a documented limitation. Laminates inside one
  element (plies) are the single-layer case. Logged, not built: fine vertex patches over
  columns, a line smoother through stacks, fold-aware aggregation.
- **Vertex patches on level 0** (5× the storage for 1.6× fewer iterations) are not used.
- **Folds:** the sign-free normal of an aggregate across a fold is an average. Rigid modes stay
  exact; only curvature columns degrade (measured above).
- **Fiber guard:** a fiber of more than 64 nodes raises. A consistently wrong axis on smaller
  meshes is not detected.
- **hex20/hex27 + `ssh8`:** node-block Jacobi on the p-level ahead of the exact corner-level
  Schwarz, measured in test 9.
- **σ-round reuse** is deferred to 18b.
- **`amg_memory`'s shell family** is calibrated on plates up to 1.04M DOF, a stack, a panel, a
  mixed mesh and a junction.
