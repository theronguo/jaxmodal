# PR 18 plan: performance pass, part a: solid-shell AMG

Branch `pr-18a-shell-amg` (this plan), from `main` after PR 17. Roadmap: PR 18 (M5), the
"Thin plates and slender beams" entry of `docs/open_items.md` (PR 11a → PR 18), and the
PR 16 rule "until PR 18, `solve_modes` on solid-shell meshes uses Jacobi".

## Revisions

- **Revision 1:** draft.

## PR 18 split

PR 18 is too large for one branch. It splits into four parts, in this order:

| part | branch | contents (roadmap PR 18 bullets) | plan |
|---|---|---|---|
| 18a | `pr-18a-shell-amg` | AMG on the assembled $A_P$ of solid-shell meshes, iteration counts vs $L/t$; `solve_modes`' default on `ssh8` meshes becomes AMG | this file |
| 18b | `pr-18b-kernels` | Morton reordering; the fp32 option for p-MG and AMG (and its estimator, the 1M Mac acceptance with fp32 AMG); HLO fusion checks; the compile-time benchmark; the roofline tests; `docs/performance.md` | own plan |
| 18c | `pr-18c-band` | the PR 15d band follow-ups (memory levers, $\hat\lambda_\max$, $\delta_\max \le 1$ on slices from 0, MINRES remedies near depth 1000, halving before a capacity error, `err.result`, the hybrid junction, $f_1 > 0$ in `solve_modes_memory`, `np.memmap`, the refill transposes, caching `layout`, the verify skips); the `COUNT_STEP` refit at ≥ 1M DOF | own plan |
| 18d | `pr-18d-calculix` | the CalculiX comparison (user request, 2026-10-18): CalculiX `*FREQUENCY` against `solve_modes` on the same `.inp` meshes | own plan |

18a goes first: 18b's fp32 option and estimator work, and 18d's shell cases, build on it.
18c is independent of 18a/18b. 18d measures the final state, so it goes last. Each later
part gets its own reviewed plan, `docs/plans/pr-18x-<slug>.md`, before it starts. The roadmap
PR 18 section gets the split and per-part status lines in 18a.

## Goal (18a)

`amg()` and `solve_modes(..., precond="amg")` work on meshes with `ssh8` blocks (all `ssh8`,
`ssh8` + `hex8`, and `ssh8` + `hex20`/`hex27`). PCG iterations on a simply supported plate
grow ≤ 2× from $L/t = 10$ to $L/t = 1000$ (roadmap test). The setup stays independent of the
mesh size: same counts at 4× the elements. `"auto"` on solid-shell meshes selects AMG.
`amg_memory` covers solid-shell meshes. Hex-only meshes take the unchanged PR 13 path.

## Why the PR 13 AMG fails on solid shells

Three causes, each measured in the 18a prototypes. Setup: SS plate $n^2 \times 1$ `ssh8`, PCG to
$10^{-6}$ on $A_P = K + \lambda_1 M$, Chebyshev degree 2 on every level, dense coarsest
solve. The scripts are listed under Measured.

1. **Near-null space.** Rigid modes alone cannot represent plate bending on coarse aggregates
   (the PR 13a hex20-plate entry). Three constant-curvature fields are added, in the
   aggregate's local frame $(t_1, t_2, n)$ with local coordinates $(X, Y, Z)$:
   $$\kappa_{11}: w = X^2/2,\ u_1 = -ZX; \qquad \kappa_{22}: w = Y^2/2,\ u_2 = -ZY; \qquad
   \kappa_{12}: w = XY,\ u_1 = -ZY,\ u_2 = -ZX.$$
   These are the Kirchhoff displacements of constant curvature. With the 6 rigid modes, each
   aggregate gets 9 coarse DOF.
2. **Smoothers.** Point and node-block Jacobi lose the thickness coupling. The pair block (the
   6×6 block of the two nodes of a through-thickness fiber) is not enough when the element is
   flat ($h/t \gg 1$). The membrane–bending coupling lives in the element. At $n = 32$:
   pair-block Jacobi 10/17/51/183 iterations at $L/t$ = 10/100/300/1000; element-patch Schwarz
   11/13/16/38; vertex-patch Schwarz 11/12/15/24.
3. **Coarse levels.** Each coarsening makes the aggregates about 3× wider in-plane at the same
   thickness, so coarse levels are even "flatter" than the fine level. Block Jacobi on the
   9×9 aggregate blocks then fails the same way pair blocks fail on the fine level. At
   $n = 64$, $L/t = 1000$, element patches on the fine level:
   - a two-level cycle (exact solve on level 1) takes 22 iterations;
   - the full V-cycle with 9×9 block Jacobi below takes 79;
   - with vertex patches on the coarse levels it takes 24 (edge patches: 78).

Two more findings fix details of the setup:

4. **Bending candidates must be one function per level.** SA passes the coarse candidates
   down as coefficients ($T R = B$). Per-aggregate local frames on the fine level only (one
   frame per level-1 aggregate) give level-1 candidates that are piecewise, not one global
   field, and the next level fits them wrongly. Flat plate $n = 64$, $L/t = 1000$:
   - fine-level frames only: 80 iterations;
   - global bending fields: 26;
   - the scheme below (candidates rebuilt per level from the next aggregates' frames): 26.
   
   On a curved panel (90° cylindrical, $R = 1$) global fields give 96 and the per-level scheme
   35.
5. **Prolongator smoothing.** Bending is a fourth-order problem. One Jacobi step of
   prolongator smoothing leaves the coarse basis too stiff, and counts grow with depth (flat,
   $L/t = 1000$: 26 at $n = 64$, 40 at $n = 128$ with 4 levels). Two steps fix it (24 at
   $n = 128$). They widen the coarse stencils, though: 2.7× the level-1 nonzeros, and vertex
   patches over the coarse matrix graph cost 852 floats per DOF. Patches and the next
   aggregation therefore use the **one-ring aggregate adjacency** (aggregates adjacent when
   their members are adjacent on the level above). With it the patches cost 114 floats per
   DOF: flat 11/20, curved 10/29 at $L/t$ = 10/1000, $n = 128$.

## Design (18a)

### Scope of the shell path

`amg()` takes the shell path when any operator block is `ssh8`. Otherwise it runs the PR 13 code
unchanged (bit-identical results, test 12). The shell path reuses:
- `_p_level`'s `Hex8Pattern` and `_hex8_galerkin` (an `ssh8` element's 8 nodes are all
  corners, and `_w8(8)` is the identity, so on `ssh8` + `hex8` meshes $A_b$ is the assembled
  $A_P$ itself);
- `ELLTransfer`, `_level`'s block-ELL export, `_cholesky`, and the `GMG` V-cycle;
- the zero-column drop, `STALL`, `CAP` and the local seed of `_sa_levels`.

New: `precond/schwarz.py` (the device smoother, about 60 lines) and `_shell_levels` in
`precond/amg.py` (the setup loop, about 90 lines).

### Groups and aggregation

A **group** is a set of DOFs aggregated together.
- **Level 0 (corner space):** groups are through-thickness **fibers**. They are the connected
  components of the graph whose edges are the node pairs $(a, a+4)$, $a = 0..3$, of every
  `ssh8` element (bottom face nodes 0–3, top 4–7, `mesh/shell.py`'s permuted order). One layer
  gives node pairs; a stack of $k$ elements gives $k+1$-node fibers. Nodes in no `ssh8`
  element are singleton groups, so a pure-hex region aggregates nodes as PR 13 does.
- **Level $\ell \ge 1$:** groups are the level-$\ell$ aggregates. Coarse DOF $j$ belongs to
  group $\lfloor j_\text{orig}/c \rfloor$ with $c$ the candidate count, via the kept column
  indices of $T$.

The group graph:
- on level 0: $G^\top |A_z| G$ ($G$ the DOF → group indicator), diagonal removed, pattern only
  (θ = 0 as in PR 13);
- on level $\ell \ge 1$: the one-ring adjacency $\mathrm{Agg}^\top (C + I)\,\mathrm{Agg}$ of
  the level above, not the coarse matrix pattern (finding 5).

`standard_aggregation` runs on the group graph. A DOF's aggregate is its group's aggregate.
Unaggregated groups (isolated: fully constrained fibers) keep empty tentative rows, as in PR 13.

### Candidates

On level $\ell$, after aggregation:
- $B^{(\ell)}_\text{fine}$ is $(3 n_c, 9)$: columns 0–5 are the masked rigid modes at the corner
  points (global, `_nullspace`). Columns 6–8 are the three curvature fields of each fine node's
  level-$(\ell+1)$ aggregate, in that aggregate's frame.
- **Frame:** the normal $n$ is the normalized sum of the member nodes' thickness vectors (per
  node: the sum over its `ssh8` elements of top-minus-bottom edge vectors). $t_1 = n \times e$,
  with $e$ the coordinate axis least aligned with $n$, normalized; $t_2 = n \times t_1$.
  $(X, Y, Z)$ are the projections of $x - \bar x_\text{agg}$ onto $(t_1, t_2, n)$.
- **Aggregates without any `ssh8` node:** curvature columns are zero there, so their tentative
  columns drop out (rigid-only aggregates, as in PR 13). The rows of constrained DOFs are zeroed
  (masked).
- **On level $\ell$:** $B_\ell = (T_0 T_1 \cdots T_{\ell-1})^\top B^{(\ell)}_\text{fine}$. The
  tentative $T$ have orthonormal columns, so this is the least-squares coefficient of the fine
  field in the level's basis. It is exact for the rigid columns, and for the curvature columns
  on flat plates. Level 0 uses $B^{(0)}_\text{fine}$ directly.
- Then `fit_candidates` → $(T_\ell, R_\ell)$, the zero-column drop, `STALL`.
- $R_\ell$ is not used as the next level's $B$. The next level recomputes $B_{\ell+1}$ from its
  own aggregates' frames.

Cost: one fine-length field per level, pushed through $\ell$ sparse products with 9 columns.
That is negligible next to the Galerkin products.

### Prolongator

$P_\ell = (I - \tfrac{\omega}{\rho} D^{-1} A_z)^2\, T_\ell$: `jacobi_prolongation_smoother`
with `degree=2`, $\omega = 4/3$, scalar diagonal $D$ on every level (as the prototype; the PR 13
first-level 3×3 block weighting is measured as an alternative during implementation and taken
only if it lowers counts). Galerkin $P^\top A_z P$, symmetrized, as now.

### Smoothers: additive Schwarz under Chebyshev

On every level the Chebyshev smoother (`degree`, `ratio` as today) runs over an additive Schwarz
preconditioner
$$D^{-1} r = \sum_i R_i^\top A_i^{-1} R_i\, r,\qquad A_i = R_i A R_i^\top,$$
with patches $i$ that cover every DOF of the level. The inverse is SPD whenever the patches cover
all DOFs and every $A_i$ is SPD.
- **Level 0 (corner space):** one patch per `ssh8` element (its 24 DOFs). A corner node in no
  `ssh8` element gets one node patch (its 3 DOFs, the PR 13 block). $A_i$ is read from $A_b$ at
  the element's `Hex8Pattern.pos` (the 64 node-block positions per element, already built). No
  per-patch SciPy indexing is needed.
- **Level $\ell \ge 1$:** one patch per group, made of its DOFs and those of its one-ring
  neighbour groups. $A_i$ is extracted per patch from the level's CSR matrix on the host.
  Levels below the fine one have $\le 1/6$ of the DOFs: about 18k patches at 1M DOF, a few
  seconds.
- **Fine level of `ssh8` + `hex8` meshes:** the matrix-free $A_P$ (as all-hex8 meshes today)
  with the level-0 Schwarz. Here $A_b = A_P$, so the patches are exact.
- **Fine level of `ssh8` + `hex20`/`hex27` meshes (p-level):** Chebyshev over a Schwarz on all
  nodes:
  - the `ssh8` element patches taken from $A_b$;
  - node-block Jacobi (`block_jacobi`) on every node in no `ssh8` element.
  
  $A_b$ differs from the fine $A_P$ only on node pairs coupled through a hex20/hex27 element
  (the interface). The patch is then an SPD approximate inverse, and Chebyshev's
  $\lambda_\max$ estimate absorbs the difference. Test 9 checks the counts.
- **Inverses:** batched Cholesky on the device, the generalization of `_inverse_blocks`.
  Constrained rows of $A_P$ are $s\,e_i$ and stay SPD. Padded slots get the identity and a zero
  mask. A non-finite inverse raises `ValueError`, as `_level`.

### `Schwarz` (device)

A registered pytree: per size class, `idx (n_p, s) int32` (pad index = $n$, dropped by
`mode="drop"`), `inv (n_p, s, s)` and the mask. Apply on $(n, k)$ blocks:
- gather $r[\text{idx}]$, then `einsum` with `inv`, then scatter-add;
- a `lax.scan` over chunks of patches bounds the temporaries to
  $\text{chunk} \cdot s \cdot k$ per class.

**Size classes:** patch sizes rounded up to a multiple of 9 (level ≥ 1) or exact (level 0: 24 or
3). This keeps the padding of irregular aggregate valences small. One gather per class.

**Fine-level storage:** $8 \cdot 24^2$ B per `ssh8` element (4.6 kB; the stored condensed $K_e$
and $m_e$ are 2.7 kB). About 95 floats per DOF on flat single-layer plates. Measured on the
prototypes, the coarse levels add about 125 floats per DOF and the coarse matrices of
operator complexity 1.9.

### API and defaults

- `amg(op, free, sigma, points, ...)`: signature unchanged; the shell path is internal.
  `return_aggregates`/`aggregates=` work on it: `AMGAggregates` gains the per-level groups,
  aggregates, kept columns and patch index sets. Reuse skips fibers, aggregation, the candidate
  fit and patch sets. It recomputes everything value-dependent; moved points keep the old
  candidates, as PR 13.
- `resolve_precond`: `"auto"` → `"amg"` on unstructured meshes with `ssh8` blocks. The refusal of
  `precond="amg"` on them is removed. `"jacobi"` stays selectable.
- `amg_memory`: no refusal on `ssh8` meshes (below).
- `pmg` keeps refusing `ssh8` blocks. The message becomes "pmg does not support ssh8 blocks; use
  amg". `no_shell` moves to `pmg.py`'s only remaining use.
- `examples/m4_shell.py`: `precond` "auto" (now AMG), iterations printed. The Jacobi run moves
  to a comparison line.
- PR 16 tests that assert `info["precond"] == "jacobi"` under `"auto"` pass `precond="jacobi"`
  explicitly (test_shell_ops band tests, `test_auto_jacobi_mixed` → `test_auto_amg_mixed`).

### `amg_memory` on solid-shell meshes

New family `"shell"` (any `ssh8` block):
- **storage** = the p-level term (meshes with hex20/hex27: as today) plus $8 \cdot 24^2$ per
  `ssh8` element (exact) plus $8 \cdot 3^2$ per non-shell corner node, plus
  `AMG_STORAGE["shell"]` $\cdot n_c$ (coarse matrices, coarse Schwarz, transfers) plus the
  coarsest factor term;
- **build / host_build:** $\beta + \alpha_\text{shell} n_c$;
- **step / step_low / count:** `AMG_STEP[("shell", ...)]`, `AMG_COUNT` reused unless the fit
  says otherwise.

All new constants are fitted as the PR 13b ones (upper envelope +2–4%). Calibration meshes:
- SS plates $64^2$–$256^2 \times 1$ `ssh8` at $L/t$ 100;
- a 3-layer laminate $64^2 \times 3$;
- a curved panel $128^2$;
- an `ssh8` + hex20 mesh.

Those are about 25k–400k DOF. The fit range goes into `docs/open_items.md`, and the 1M-DOF
point is a slow test (test 11).

## Tests (18a)

`tests/test_shell_amg.py`, unless noted. Meshes are built by a helper `plate(n, Lt,
curved=False, layers=1)`. It makes `box_mesh` + `solid_shell(direction=(0, 0, 1))`, and the
curved variant maps $x$ onto a 90° cylinder of radius 1.

1. **Fibers:** a 1-layer plate gives the node pairs. A 3-layer stack gives 4-node fibers equal
   to the `box_mesh` columns. In an `ssh8` + `hex8` mesh the hex-only nodes are singletons.
2. **Candidates:**
   - on a flat plate, the level-0 local-frame candidates span, per aggregate, the same space as
     global $[\text{rigid}, \kappa]$ (principal angles ≤ 1e-10);
   - on every level, $T_\ell R_\ell = B_\ell$ to 1e-12 (the fit identity);
   - on the curved panel, each aggregate's curvature fields are the Kirchhoff fields in its
     frame (formula check on 3 aggregates).
3. **Schwarz apply** equals a dense NumPy $\sum_i R_i^\top A_i^{-1} R_i r$:
   - on an `ssh8` + `hex8` mesh (level 0) and on a coarse level, to 1e-12;
   - symmetric ($x^\top D^{-1} y = y^\top D^{-1} x$ to 1e-12);
   - chunked equals unchunked (exact);
   - padded slots are inert (a padded class against an unpadded one, exact).
4. **Level-0 patches from `pos`** equal SciPy `A[s][:, s]` to 1e-14.
5. **V-cycle** symmetric to 1e-12 and positive on random blocks, on `ssh8`, `ssh8` + `hex8` and
   `ssh8` + `hex20` meshes.
6. **$L/t$ growth (roadmap test):**
   - PCG iterations to $10^{-6}$ at σ = $\lambda_1$ (from `eigsh`), on the SS plate $128^2
     \times 1$: count at $L/t = 1000$ ≤ 2× the count at $L/t = 10$ (prototype 11 → 20);
   - the curved panel $128^2$ is report-only for the ratio (prototype 10 → 29). Its $L/t = 1000$
     count is gated at ≤ 1.5× the implementation's measured value, recorded in the docs.
   - Slow.
7. **Mesh independence:** flat $L/t = 1000$, the $256^2$ count ≤ 1.3× the $128^2$ count
   (prototype 18 vs 20; slow, about 400k DOF). The $h/t$ dependence (the $64^2$ count, 26) is
   reported in the docs.
8. **`solve_modes` with AMG** matches the PR 8 `eigsh` to the PR 9 tolerances:
   - the M4 example plate (roadmap test);
   - a 3-layer laminate (Srinivas–Rao-type ply table);
   - the curved panel;
   - a band solve on the M4 plate.
   
   `info["precond"] == "amg"` under `"auto"`. Iterations ≤ 1/5 of Jacobi's on the M4 plate
   (measured ratio in the docs).
9. **Mixed meshes:** a plate of `ssh8` with a `hex8` stiffener, and with a `hex20` stiffener
   (p-level path), with `solve_modes` AMG against `eigsh` to the PR 9 tolerances. PCG count
   below Jacobi's.
10. **Reuse:** `aggregates=` on a shell mesh with new materials, $s_K$, σ equals a fresh setup
    (V-cycle to 1e-12). A connectivity or mask change raises.
11. **`amg_memory`** (shell family):
    - storage ≥ owned bytes and within 1.3× on the calibration plates;
    - build ≥ the host peak and ≤ 1.3× (`peak` marker);
    - the LOBPCG step with AMG ≤ `lobpcg_memory`'s step (`memory_analysis`);
    - a ≥ 1M-DOF `ssh8` plate: estimate ≥ the measured peak (slow, `peak`).
12. **Hex meshes unchanged:** the existing `test_amg.py` passes unmodified. One added test pins a
    hex20 V-cycle against a stored reference output to 1e-14, so the shell branch never touches
    the hex path.
13. **Refusals updated** (`test_shell_ops.py`): `amg`, `amg_memory`, `solve_modes(precond="amg")`
    accept `ssh8`; `pmg` raises the new message; otf on `ssh8` still raises.

## Docs (18a)

- `docs/preconditioners.md`: an "AMG on solid shells" section covering:
  - the three causes, with the measured tables;
  - fibers, local-frame curvature candidates (LaTeX as above) and their per-level projection;
  - degree-2 prolongator smoothing;
  - the one-ring adjacency;
  - Schwarz patches, storage, `amg_memory`'s shell family.
  
  The flat-element limitation in the AMG section points to it.
- `docs/open_items.md`: the thin-plate entry is resolved for solid shells (hex plates keep the
  limitation; solid shells are the remedy). New decisions:
  - the approximate patches at hex20/`ssh8` interfaces;
  - aggregates across folds or T-junctions get a mean normal (rigid modes stay exact; no
    fold-aware strength yet);
  - the shell family's calibration range.
- `docs/eigensolvers.md` and `README.md`: `"auto"` → AMG on solid-shell meshes.
- `docs/elements.md`: the PR 16 Jacobi note is updated.
- `docs/ROADMAP.md`: the PR 18 split and status; deferred GPU checks for the shell AMG at 10M if
  any.

## Measured (prototypes, 18a planning)

Scripts in the session scratchpad (`pr18/`), host SciPy/pyamg, fp64. SS plate (all four edges
`SIMPLY_SUPPORTED`-style), one `ssh8` layer, PCG to $10^{-6}$ at σ = $\lambda_1$ from a random
right-hand side.
- `proto_cheb.py` / `proto_cheb2.py`: pyamg SA hierarchy (pair BSR 6, global bending), smoother
  variants.
- `proto_impl.py`: the scheme of this plan.

| configuration | n | $L/t$ = 10 / 100 / 300 / 1000 |
|---|---|---|
| pair-block Jacobi, coarse 9×9 block | 32 | 10 / 17 / 51 / 183 |
| element patches, coarse 9×9 block | 32 | 11 / 13 / 16 / 38 |
| vertex patches, coarse 9×9 block | 32 | 11 / 12 / 15 / 24 |
| element patches, coarse 9×9 block | 64 | 12 / 14 / 35 / 79 |
| element patches, two-level | 64 | – / – / 15 / 22 |
| element patches, coarse vertex patches (pyamg hierarchy) | 64 | 12 / – / 18 / 24 |
| this plan with one prolongator step, fine-level frames only | 64 | 12 / – / 41 / 80 |
| this plan with one step, per-level frames | 64 | 12 / – / – / 26 |
| this plan with one step, per-level frames | 128 | 12 / 13 / – / 40 |
| **this plan** (two steps, one-ring adjacency) | 64 | 11 / – / – / 26 |
| **this plan** | 128 | 11 / – / – / 20 |
| **this plan** | 256 | – / – / – / 18 |
| curved panel, one step, global bending | 64 | 13 / – / – / 96 |
| curved panel, one step, per-level frames | 64 | 10 / – / – / 35 |
| curved panel, this plan | 64 | – / – / – / 32 |
| **curved panel, this plan** | 128 | 10 / – / – / 29 |

At fixed $L/t = 1000$ the counts fall with refinement (26 / 20 / 18 at $n$ = 64 / 128 / 256,
$h/t$ = 15.6 / 7.8 / 3.9): they follow the element flatness $h/t$, not $L/t$ itself. The
roadmap ratio is therefore a statement about the mesh used: at $n = 64$ the flat ratio is 2.4×,
at $n = 128$ 1.8×. The $n = 256$ run (396k DOF, 5 levels) took 170 s in the host prototype.

Floats per DOF of the Schwarz inverses (fine / coarse levels): pair 6; element 93–95 /
vertex 454–470 on the fine level; coarse vertex patches over the matrix graph 117 (one step) and
852 (two steps), over the one-ring adjacency 114 + 12.

## Risks and decisions without user input

- **Memory:** the shell AMG stores about 220 floats per DOF of Schwarz inverses plus coarse
  matrices of operator complexity 1.9, against 6 for Jacobi. That is 1.8 GB at 1M DOF in fp64.
  It is accepted: Jacobi does not converge at $L/t \ge 300$ within practical counts. 18b's fp32
  option halves the inverses.
- **Vertex patches on the fine level** (454 floats/DOF, counts 24 vs 38 at $n = 32$) are not
  used: 5× the storage for 1.6× fewer iterations. Element patches plus coarse vertex patches
  give the target.
- **Element flatness:** counts grow with $h/t$ (prototype: 18 / 20 / 26 at $h/t$ = 3.9 / 7.8 /
  15.6, $L/t = 1000$). The docs state this, and the roadmap gate is taken on $128^2$
  ($h/t \le 7.8$).
- **The curved-panel ratio** (2.9× at the prototype) is above the roadmap's 2×, which the flat
  plate meets. Per the roadmap's "or a fallback is documented", the curved ratio is reported
  and gated on its absolute count. The documented fallback is a higher Chebyshev degree
  (`precond_options={"degree": 3}`, measured during implementation).
- **Folds and junctions:** a mean normal per aggregate is wrong across a fold. Rigid modes stay
  exact, and only the curvature columns degrade there. No fold-aware strength in 18a; logged.
- **hex20/hex27 + `ssh8` interfaces:** approximate fine patches (above). Measured in test 9.
