# PR 12 plan: p-multigrid for unstructured meshes

PR 12 (M3) depends on PRs 5, 9a and 10. It adds a two-level p-multigrid preconditioner for hex20/hex27 meshes:
- the fine level is the matrix-free $A_P$ with a symmetric Chebyshev smoother;
- the coarse level is the hex8 Galerkin product $P^\top A_P P$ on the corner nodes, solved exactly by SciPy `splu` through `pure_callback`.

The `splu` coarse solve is for development and tests only (roadmap): PR 13 replaces it with AMG on the same hex8 Galerkin matrix, which this PR also returns as a SciPy CSR. `solve_modes` dispatch, the σ pre-estimate for unstructured meshes and the memory estimator are PR 13.

## Measured facts (prototype `scratchpad/p12/`; jax 0.11.2, fp64, M3 Max CPU)

Setup for every case unless noted:
- steel ($E = 210\cdot10^9$, $\nu = 0.3$, $\rho = 7850$), unit box, `xmin` clamped;
- σ = $\lambda_1$ from `eigsh`;
- PCG on $A_P(K + \sigma M)$ to $10^{-8}$ with 4 random right-hand sides;
- Chebyshev degree 2, ratio 30, over node-block Jacobi;
- the coarse hex8 Galerkin matrix factored by `splu`.

| case | DOF | PCG iterations (degree 2 / 4) |
|---|---|---|
| hex20 boxes $3^3$ / $6^3$ / $12^3$ / $24^3$ | 624 – 182k | 14 / 14 / 14 / 14 (9 / 9 / 9 / 9) |
| hex27 boxes $3^3$ / $6^3$ / $12^3$ | 1k – 47k | 13 / 13 / 13 (8 / 8 / 8) |
| hex20 beam $20\times2\times2$ ($L/h = 10$) | 1.9k | 16 (10) |
| hex20 free $6^3$ (σ = first elastic λ) | 3.7k | 14 (9) |
| hex20 curved $6^3$ (`tests/helpers.distort`, curved) | 3.7k | 14 (9) |
| hex20 cube $8^3$, Chebyshev ratio 10 / 20 / 30 / 60 | 8.0k | 14 / 13 / 14 / 18 |
| hex20 single-layer plate $16^2\times1$, element aspect ratio $a$ = 1 / 2 / 4 / 10 | 5.9k | 15 / 21 / 63 / 699 |
| hex20 box $8^3$ with $h_z = h/a$, $a$ = 1 / 2 / 4 / 10 | 8.0k | 14 / 17 / 28 / 76 |
| hex27 single-layer plate $32^2\times1$, $a = 10$ | 38k | 1000 (degree 4: 691) |

**Galerkin exactness.** The element-wise device product (Design) matches the SciPy triple product $P^\top a_p(K + \sigma M, free)\,P$ to $\le 4\cdot10^{-16}$ relative. Checked on hex20 and hex27, stored and otf, consistent and lumped mass, with two materials, random `s_K`/`s_M`, nodal masses on half the nodes and a mixed clamped/sliding mask.

**LOBPCG** (10 modes, tol $10^{-6}$, σ = $\lambda_1$; times include compilation):

| mesh | Jacobi | p-MG degree 2 | p-MG degree 3 |
|---|---|---|---|
| hex20 $12^3$ (25k DOF) | 431 iterations, 6.2 s | 15, 1.2 s | 14, 1.4 s |
| hex20 beam $20\times2\times2$ | 308, 1.6 s | 14, 0.3 s | 10, 0.3 s |
| hex27 $16^3$ (108k DOF) | 508, 21.7 s | 15, 3.6 s | 14, 3.9 s |

The eigenvalues agree with the Jacobi run to $\le 3\cdot10^{-11}$.

**Cost per application.** Chebyshev degree $d$ costs $d - 1$ $A_P$ applies. A V-cycle therefore costs $2d$ $A_P$ applies plus one coarse solve. Degrees 2, 3 and 4 then give about 70, 70 and 81 $A_P$ applies per PCG solve on the hex20 boxes. The default is degree 2: it uses the fewest applies of the three, ties degree 3, and needs the least smoother workspace.

**`splu` cost** (coarse hex8 matrix of an $n^3$ box, default COLAMD ordering; `MMD_AT_PLUS_A` with symmetric mode is no better):

| $n$ | coarse DOF | factor time | $L + U$ storage | process peak |
|---|---|---|---|---|
| 16 | 15k | 1.0 s | 0.2 GB | 0.8 GB |
| 24 | 47k | 14 s | 1.5 GB | 3.9 GB |
| 32 | 108k | 140 s | 6.2 GB | 18 GB |

This scaling is why the coarse solve is test-only: the tests stay at or below the 47k-DOF coarse level of the hex20 $24^3$ box.

## Design

### Module and entry

New module `precond/pmg.py`, exported from `jaxmodal.precond`:

```python
P = pmg(op, free, sigma, *, degree=2, ratio=30.0)
Ac, T = hex8_galerkin(op, free, sigma)  # PR 13's input: SciPy CSR P^T A_P P and the transfer
```

- **`op`** must be an `Unstructured` operator (PR 5). Anything else raises `TypeError`; the message points voxel stencils to `gmg`.
  - All blocks may be hex8. Then $P = I$, and the Galerkin matrix is the assembled hex8 $A_P$, which is what PR 13 feeds to AMG for pure hex8 meshes.
  - `check_x64` runs first.
- **`sigma`** must be concrete and > 0.
  - σ < 0 or NaN raises `ValueError` through `_check_sigma`.
  - σ == 0 raises `SingularShiftError` ("pmg needs sigma > 0 ..."). With rigid modes left by the BCs, $A_P$ at σ = 0 is exactly singular. Unlike `gmg`, `pmg` has no mesh to run `rigid_modes` on, so it cannot prove nonsingularity.
  - Reusing `SingularShiftError` lets PR 13's dispatch take the same fallback as `_gmg_sigma`.
  - Decision: require σ > 0 instead of passing the mesh. The σ pre-estimate always supplies a positive σ, and σ = 0 has no use here.
- **Return value.** `pmg` returns a pytree `PMG` whose `__call__(r)` applies one V-cycle to an `(n_dof, k)` block, so it fits every `precond` slot (`pcg`, `minres`, `lobpcg`).
  - Its leaves are the fine $A_P$, the Chebyshev smoother, the transfer tables and the mask.
  - The coarse factor is a static field (below).

### Transfer (host build, device apply)

The coarse nodes are every node that is a corner (local index 0–7) of some element. They are numbered in increasing fine id, which keeps the RCM locality of the fine numbering.

Each fine node gets its parent corners and weights from the first element containing it:
$$
w_{ac} = N^{(8)}_c(\xi_a),
$$
where $N^{(8)}$ are the trilinear shape functions and $\xi_a$ is the type's reference node (`REFERENCE_NODES[:k]`, VTK order).
- A corner node gets itself with weight 1.
- A hex20 or hex27 edge node gets its 2 edge corners with ½ each.
- A hex27 face node gets its 4 face corners with ¼ each.
- The hex27 centre node gets all 8 corners with ⅛ each.

The table is `(n_nodes, w)` int32 parents and fp64 weights, with $w$ the largest parent count present (1 if all hex8, 2 with hex20, 8 with hex27). Shorter rows are padded with weight 0.

**Properties of $P$:**
- Exact for trilinear fields on straight-sided elements whose midside nodes sit at the trilinear image of their reference position (e.g. `box_mesh`, graded `coords`).
- On curved elements, $P$ still defines a nested coarse space: trilinear functions of the reference coordinates on the fine geometry. So $P^\top A_P P$ is a true Galerkin coarse operator.
- On conforming meshes every element containing a node gives it the same parents. This is documented as an assumption and not checked; nonconforming test meshes keep their midside nodes inside one block.

**Device apply.**
- Prolongation: $x_f = \sum_j w_{:,j}\,x_c[\text{parent}_{:,j}]$ per component, a gather and a weighted sum over $(n_\text{nodes}, w, 3, k)$.
- Restriction: its exact transpose via `jax.linear_transpose`.

### Galerkin hex8 matrix

The fine operator is never assembled. For element $e$ with $k$ nodes:
- $P_\text{loc}$ is the type's $k\times8$ weight table;
- $m_e$ is its free flags;
- $A_e = s_{K,e} K_e + \sigma \rho_{\phi(e)} s_{M,e} M_e$, with $M_e$ consistent or lumped as the operator's `mass`.

The coarse cell matrix is
$$
C_e = (P_\text{loc}\otimes I_3)^\top\big(\mathrm{diag}(m_e) A_e \mathrm{diag}(m_e) + D_e\big)(P_\text{loc}\otimes I_3),
\quad
P^\top A_P P = \sum_e Z_e^\top C_e Z_e .
$$
- $Z_e$ maps the element's 8 corners to the coarse numbering.
- $D_e$ holds $d = \sigma\,n_\text{nodal}\,m + s(1 - m)$ (nodal masses and the constrained diagonal of $A_P$, $s$ = `ap_operator(op, free, sigma).s`) on the DOFs of the nodes $e$ owns. Each node is owned by the first element containing it in block order, so its $d$ enters once. This is the same owner trick as GMG (PR 11a).

**Implementation:**
- A new `element_matrices(op, b, d)` in `operators/matfree.py`, next to the kernels that know the storage modes, returns $s_K K_e$ and $\rho\, s_M M_e$ as $(c, k, k)$ arrays for one chunk:
  - stored mode unpacks `Ke`;
  - otf mode forms $B^\top C B$ from `geo`;
  - $M_e$ comes from `wdet` and $N$ (consistent), or from `lumped_nodal` on the diagonal (lumped).
- `pmg.py` runs one jitted `lax.map` per block over the operator's own padded chunks. Padded elements have $s_K = s_M = 0$ and own nothing, so they contribute zero.
  - It contracts the $(c, n, 3, n, 3)$ view with $P_\text{loc}$, never forming $(P_\text{loc}\otimes I_3)$.
  - It symmetrizes each $C_e$ exactly ($\tfrac12(C_e + C_e^\top)$).
  - It returns the $(n_\text{pad}, 24, 24)$ cells.
- The host drops the pads and builds a SciPy `coo_array` over the coarse DOFs $3c + i$ with int32 indices (`check_int32`), then converts it to CSR, which sums the duplicates.
- Cells take about 4.6 KB per element (0.4 GB at 1M hex20 DOF); the COO triples are of the same order. PR 13's production assembler, chunked into a precomputed CSR pattern, replaces this host step at scale.

### Fine smoother and V-cycle

- $A$ = `ap_operator(op, free, sigma)`; $D$ = `block_jacobi(op, free, sigma)`; $S$ = `chebyshev(A, D, lmax_estimate(A, D), degree, ratio)` (PR 10). $S$ is symmetric.
- **V-cycle**, with input and output masking as in GMG:
$$
b \leftarrow m\odot b,\quad x = S b,\quad x \mathrel{+}= P\,A_c^{-1}P^\top(b - A x),\quad x \mathrel{+}= S(b - A x),\quad x \leftarrow m\odot x.
$$
- It is symmetric (same $S$ before and after, exact coarse solve).
- It is positive definite when $\|I - SA\|_A < 1$, which Chebyshev on $[\hat\lambda/\text{ratio}, \hat\lambda]$ with $\hat\lambda \ge \lambda_\max(D^{-1}A)$ gives.
- **Coarse solve.**
  - `scipy.sparse.linalg.splu(Ac.tocsc())` on the host.
  - Applied inside the jitted cycle by `jax.pure_callback(solve, ShapeDtypeStruct((n_c, k), float64), rc, vmap_method="sequential")`.
  - The factor is a static field of a small pytree `HostLU`, hashed by identity. This is the one closure exception that principle 3 allows.
  - Consequence: every new `pmg` build is a new static value, so jitted callers (the `pcg` and `lobpcg_step` traces) recompile once per build. Documented; PR 13's device AMG has only array leaves and does not recompile.

### Limitations (documented, `docs/open_items.md`)

- **Flat elements.** The hex8 coarse space locks in bending on elements with aspect ratio $a \gg 1$, so the coarse correction misses the smooth bending error and the local smoother cannot recover it:
  - single-layer plates: 15 / 21 / 63 / 699 iterations for $a$ = 1 / 2 / 4 / 10;
  - a box with $h_z = h/a$: 14 / 17 / 28 / 76;
  - slenderness alone with $a = 1$ is fine (beam $20\times2\times2$: 16).

  Revisit in PR 13 (AMG directly on the high-order level, or a near-null space that keeps bending) and in M4 (solid-shell elements).
- **`splu` scaling** (table above). Test-only, replaced in PR 13.

## Files

| file | change |
|---|---|
| `src/jaxmodal/precond/pmg.py` | new: transfer table, `hex8_galerkin`, `HostLU`, `PMG`, `pmg` |
| `src/jaxmodal/operators/matfree.py` | `element_matrices(op, b, d)` |
| `src/jaxmodal/precond/__init__.py` | export `pmg`, `PMG`, `hex8_galerkin` |
| `tests/test_pmg.py` | new |
| `benchmarks/pmg_study.py`, `benchmarks/results/pmg_study.json` | new: the iteration, aspect and LOBPCG tables above, plus the build time at hex20 $24^3$, re-measured with the shipped code |
| `docs/preconditioners.md`, `README.md`, `docs/ROADMAP.md`, `docs/open_items.md` | p-MG section, status, flat-element item |

## Tests (`tests/test_pmg.py`)

Common setup:
- Iteration tests use steel, x64 and PCG to $10^{-8}$ with 4 right-hand sides (`tests/helpers.rhs`).
- σ = the lowest nonzero eigenvalue from `eigsh` (a fixed `v0`, as `tests/helpers.box` does).
- Exactness tests use the PR 10 unit materials (`helpers.MATERIALS`, two phases and orientations) with σ = 0.37.
- The reference $P$ is built independently in the test: a Python loop over elements evaluates `np.prod(1 + ξ_a ξ_c)/8` at `REFERENCE_NODES`, then densified.

1. **Transfers** (roadmap).
   - On graded hex20 and hex27 boxes (`box_mesh(..., coords=...)`, uneven spacings), prolongation of the coarse values of a random trilinear field $f = a + bx + cy + dz + exy + fyz + gzx + hxyz$ reproduces $f$ at every fine node to $10^{-12}$.
   - Restriction equals the densified $P^\top$ to $10^{-14}$; corner rows are identity rows.
   - On `helpers.mixed_mesh` the hex8 nodes have $w = 1$ identity rows.
2. **Galerkin vs SciPy** (roadmap). `hex8_galerkin` equals $P_\text{ref}^\top\, a_p(K + \sigma M, free, s)\, P_\text{ref}$ from `assemble` to $10^{-12}$ relative and is exactly symmetric. Cases:
   - hex20 and hex27 boxes with a mixed mask (`xmin: xyz`, `ymin: y`), random `s_K`/`s_M` and nodal masses on half the nodes;
   - stored and otf; consistent and lumped;
   - `helpers.mixed_mesh` (hex8 + hex20) with a random mask;
   - a block chunk of 3 (several chunks and padded elements).

   All-hex8 boxes give the assembled hex8 $a_p$ itself (identity $P$).
3. **Dense V-cycle reference.** On a hex20 $3\times2\times2$ box (clamped and sliding):
   - densify $A_0$ (SciPy $a_p$), $P_\text{ref}$, the built smoother $S$ (on the identity) and $A_c^{-1}$;
   - compose the cycle above in NumPy with masking;
   - `pmg(...)(U)` equals it to $10^{-10}$ on random $U$.
4. **Symmetry and definiteness.**
   - $u^\top V v = v^\top V u$ to $10^{-12}$ relative on unmasked random vectors, and $u^\top V u > 0$ on masked ones (clamped and free hex20 boxes).
   - Constrained outputs are exactly 0.
5. **h-independence** (roadmap). Iterations grow ≤ 20% from the first to the last size:
   - hex20 boxes $3^3$, $6^3$, $12^3$, $24^3$ ($24^3$ `slow`);
   - hex27 $3^3$, $6^3$, $12^3$.
6. **Robustness bounds.** Each ≤ 30 iterations (measured 14–16, 2× margin):
   - beam $20\times2\times2$;
   - a free $6^3$ box at σ > 0;
   - a curved distorted $6^3$ box;
   - a box with nodal masses on half the nodes;
   - a hex20 $4^3$ box with `s_K` scaled by $10^{-3}$ on half the elements.
7. **LOBPCG.** `lobpcg(op, free, 6, tol=1e-8, precond=pmg(...))` on a clamped hex20 $6\times4\times3$ box matches `reference_modes` with the 9a check helper, in fewer iterations than with Jacobi. Also on hex27.
8. **Errors.**
   - A `Stencil` raises `TypeError` with a message naming `gmg`.
   - σ < 0 and NaN raise `ValueError` ("sigma must be >= 0").
   - σ = 0 raises `SingularShiftError`.
   - x64 off raises.
9. **Jit and callback.** The V-cycle runs inside `pcg` and `lobpcg` (tests 5 and 7). A second application with the same `PMG` reuses the compiled `pcg` (cache size unchanged). Recompiling on a new build is documented behaviour, not tested.

The iteration bounds were measured with the prototype. The implementation re-measures them with the shipped code and keeps a margin of at least 1.5×.

## Docs

- **`docs/preconditioners.md`**, a p-MG section covering:
  - the transfer table and weights, with the reference-coordinate formula and what it means on curved elements;
  - the Galerkin cell formula (above) with the owner trick;
  - `hex8_galerkin` as PR 13's input;
  - the smoother and the V-cycle with masking;
  - the host `splu` coarse solve: `pure_callback`, test-only, recompiles on a new build, the scaling table;
  - σ > 0 and `SingularShiftError`;
  - the measured iteration tables;
  - the flat-element limitation;
  - the tests list.
- **`docs/operators.md`**: `element_matrices`.
- **`README.md`**: p-MG in the status and feature list, with a short `lobpcg(..., precond=pmg(op, free, sigma))` snippet on a hex20 box (run by `test_readme`).
- **`docs/ROADMAP.md`**: the PR 12 status line and the flat-element limitation next to PR 13.
- **`docs/open_items.md`**: flat elements (open; PR 13 / M4); the σ > 0 requirement (decision taken without user input).

## Not covered

- PR 13 covers:
  - the AMG coarse solve and the production hex8 assembler;
  - the `solve_modes` dispatch and σ pre-estimate for unstructured meshes;
  - the p-MG memory estimator;
  - the 1M-DOF hex20 Mac acceptance.
- Not planned:
  - an fp32 p-MG cycle (no roadmap item);
  - hex20 → hex8 transfers for mixed meshes with hanging nodes;
  - a check that meshes are conforming.
