# PR 12 plan: p-multigrid for unstructured meshes

PR 12 (M3) depends on PRs 5, 9a and 10. It adds a two-level p-multigrid preconditioner for hex20/hex27 meshes:
- the fine level is the matrix-free $A_P$ with a symmetric Chebyshev smoother;
- the coarse level is the hex8 Galerkin product $P^\top A_P P$ on the corner nodes, solved exactly by SciPy `splu` through `pure_callback`.

The `splu` coarse solve is for development and tests only (roadmap): PR 13 replaces it with AMG on the same hex8 Galerkin matrix, which this PR also returns as a SciPy CSR. `solve_modes` dispatch, the σ pre-estimate for unstructured meshes and the memory estimator are PR 13.

## Measured facts (prototype `scratchpad/p12/`; jax 0.11.2, fp64, M3 Max CPU)

Setup for every case unless noted:
- steel ($E = 210\cdot10^9$, $\nu = 0.3$, $\rho = 7850$), unit box, `xmin` clamped;
- σ = $\lambda_1$ of the $6^3$ mesh of the same element type, from `eigsh`, fixed over each h-sequence (one continuous problem; per-mesh $\lambda_1$ gives the same counts at $3^3$–$12^3$, and `eigsh` is out of reach at $24^3$);
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
| hex27 single-layer plate $32^2\times1$, $a = 10$ | 38k | > 1000, not converged (degree 4: 691) |
| hex20 $4^3$, `s_K` = $10^{-3}$ on a seeded random half / checkerboard; $10^{-6}$ random half | 1.2k | 17 / 19 / 18 |
| hex20 $6^3$, nodal masses on half the nodes at 0.1 / 1 / 10 element masses | 3.7k | 14 / 14 / 14 |

**Galerkin exactness.** The element-wise device product (Design) matches the SciPy triple product $P^\top a_p(K + \sigma M, free)\,P$ to $\le 4\cdot10^{-16}$ relative. Checked on hex20 and hex27, stored and otf, consistent and lumped mass, with two materials, random `s_K`/`s_M`, nodal masses on half the nodes and a mixed clamped/sliding mask.

**LOBPCG** (10 modes, tol $10^{-6}$, σ = $\lambda_1$; times include compilation):

| mesh | Jacobi | p-MG degree 2 | p-MG degree 3 |
|---|---|---|---|
| hex20 $12^3$ (25k DOF) | 431 iterations, 6.2 s | 15, 1.2 s | 14, 1.4 s |
| hex20 beam $20\times2\times2$ | 308, 1.6 s | 14, 0.3 s | 10, 0.3 s |
| hex27 $16^3$ (108k DOF) | 508, 21.7 s | 15, 3.6 s | 14, 3.9 s |

The eigenvalues agree with the Jacobi run to $\le 3\cdot10^{-11}$.

**Cost per application.** Chebyshev degree $d$ costs $d - 1$ $A_P$ applies. A V-cycle therefore costs $2d$ $A_P$ applies plus one coarse solve, and a PCG iteration one more apply. Degrees 2 / 3 / 4 need 14 / 11 (10 on some boxes) / 9 iterations, i.e. 70 / 77 (70) / 81 $A_P$ applies per PCG solve on the hex20 boxes. Measured at hex20 $24^3$ ($m = 25$): $t_A$ = 57 ms, the V-cycle without its coarse solve 4.0 $t_A$; the test-only `splu` solve adds about 14 $t_A$, the `pure_callback` itself nothing measurable. With a production coarse solve of about 1 $t_A$ (PR 13) degrees 2 / 3 / 4 cost about 84 / 80 / 90 $t_A$ per solve, a tie. The default is degree 2: tied on cost, one LOBPCG iteration more than degree 3, and the least smoother workspace.

**`splu` cost** (coarse hex8 matrix of an $n^3$ box, default COLAMD ordering):

| $n$ | coarse DOF | factor time | $L + U$ storage | process peak |
|---|---|---|---|---|
| 16 | 15k | 1.0 s | 0.2 GB | 0.8 GB |
| 24 | 47k | 14 s | 1.5 GB | 3.9 GB |
| 32 | 108k | 140 s | 6.2 GB | 18 GB |

This scaling is why the coarse solve is test-only: the tests stay at or below the 47k-DOF coarse level of the hex20 $24^3$ box. Geometric nested dissection (`eigen/reference._order` on the corner coordinates, `permc_spec="NATURAL"`, symmetric mode) is 4–8× faster (3.5 s, 0.69 GB at $n = 24$; 17.8 s, 2.3 GB at $n = 32$), but `Unstructured` carries no coordinates. Decision: keep COLAMD rather than add a `points` argument to a test-only path; PR 13's AMG replaces it.

## Design

### Module and entry

New module `precond/pmg.py`, exported from `jaxmodal.precond`:

```python
P = pmg(op, free, sigma, *, degree=2, ratio=30.0)  # a GMG: one V-cycle per call
```

`_hex8_galerkin(A)` (the `APOperator`; review round 1 of the code) returns the SciPy CSR $P^\top A_P P$ and the transfer. It stays private; PR 13 (same package) consumes it and decides its public form.

- **`op`** must be an `Unstructured` operator (PR 5). Anything else raises `TypeError`; the message points voxel stencils to `gmg`.
  - All blocks may be hex8. Then $P = I$, and the Galerkin matrix is the assembled hex8 $A_P$, which is what PR 13 feeds to AMG for pure hex8 meshes.
  - `check_x64` runs first.
- **`sigma`** must be concrete and > 0.
  - σ < 0 or NaN raises `ValueError` through `_check_sigma`.
  - σ == 0 raises `SingularShiftError` ("pmg needs sigma > 0 ..."). With rigid modes left by the BCs, $A_P$ at σ = 0 is exactly singular. Unlike `gmg`, `pmg` has no mesh to run `rigid_modes` on, so it cannot prove nonsingularity.
  - Reusing `SingularShiftError` lets PR 13's dispatch take the same fallback as `_gmg_sigma`.
  - Decision (taken without user input, logged in `docs/open_items.md`): require σ > 0 in PR 12. This has a cost: a σ pre-estimate started at σ = 0 needs fewer rounds and iterations on clamped bodies (cube: 1 round / 4 iterations against 2 / 10; plate $a = 4$: 1 / 20 against 3 / 60). PR 13 owns the pre-estimate and can add the hook: accept σ = 0 when `solve_modes` has found no rigid modes and every `s_K > 0`.
- **Return value.** `pmg` returns the existing `GMG` pytree (PR 11a), so p-MG and GMG share one V-cycle (`_vcycle`, `_vcycle_apply`) instead of two copies:
  - `levels = ((A, S),)`, `transfers = (T,)`, `coarse = HostLU(...)`, `free`, `defl = None`, `dtype = "float64"`.
  - `__call__(r)` applies one V-cycle to an `(n_dof, k)` block, so it fits every `precond` slot (`pcg`, `minres`, `lobpcg`).

### Shared cycle (refactor of `precond/gmg.py`)

Two small generalizations, no behaviour change for `gmg`:
- **Coarse slot.** `GMG.L` becomes `GMG.coarse`, any pytree callable on an $(n_c, k)$ block. `gmg` stores `Cholesky(L)`, a pytree whose `__call__` keeps today's fp64 cast and `cho_solve`; `pmg` stores `HostLU`. `_vcycle` calls `P.coarse(b)`.
- **Restriction.** `_restrict(T, r)` becomes a method `T.restrict(r)`. The voxel `Transfer` keeps `jax.linear_transpose` of its prolongation; the p-MG transfer gathers from a child table (below).

The GMG docstring says it is the shared multilevel cycle (`pmg` returns one with a p-level and a host coarse solve); `docs/preconditioners.md` names the slot and renames `L` to `coarse` (a `Cholesky`) in the GMG fp32 and tests text. `tests/test_gmg.py` switches `P.L` → `P.coarse.L` (lines 115, 425, 597) and `G._restrict(T, r)` → `T.restrict(r)` (line 142).

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

**Device apply** (a pytree `NodalTransfer`), both gathers unrolled over the table columns so no $(n, w, 3, k)$ array is formed:
- Prolongation: $x_f = \sum_{j<w} w_{:,j}\,x_c[\text{parent}_{:,j}]$ per component.
- Restriction: $r_c = \sum_{j<w'} w'_{:,j}\,r_f[\text{child}_{:,j}]$ from a child table built on the host by inverting the parent table: $(n_c, w')$ with $w'$ = 7 for hex20 and 27 for hex27 conforming boxes (the largest count present), padded with weight 0.

Measured temporaries at $m = 25$, in fine blocks: the einsum prolongation 3.0 (hex20 1M) / 9.0 (hex27 $16^3$), the unrolled one 0; `linear_transpose` restriction 2.0 / 8.0, the child gather 0. In the full V-cycle this saves 1.0 block at hex20 1M and 4.2 at hex27 $16^3$ (15.2 → 11.0). Restriction is then the transpose by test (test 1), not by construction.

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
- A new `element_matrices(op, block, chunk)` in `operators/matfree.py`, next to the kernels that know the storage modes, returns $s_K K_e$ and $\rho\, s_M M_e$ as $(c, k, k)$ arrays for one chunk:
  - stored mode unpacks `Ke`;
  - otf mode forms $B^\top C B$ from `geo`;
  - $M_e$ comes from `wdet` and $N$ (consistent), or from `lumped_nodal` on the diagonal (lumped).
  - Padded elements return zero matrices ($s_K = s_M = 0$); tested directly.
- `pmg.py` runs one jitted `lax.map` per block over the operator's own padded chunks. Padded elements own nothing, so they contribute zero.
  - It contracts the $(c, n, 3, n, 3)$ view with $P_\text{loc}$, never forming $(P_\text{loc}\otimes I_3)$.
  - It returns the $(n_\text{pad}, 24, 24)$ cells.
- The host drops the pads and builds a SciPy `coo_array` over the coarse DOFs $3c + i$ with int32 indices (`check_int32`), converts it to CSR (summing duplicates), then symmetrizes once: `Ac = (Ac + Ac.T) / 2`, as `assemble` does. Per-cell symmetrization is not enough: CSR duplicate summation adds the (i,j) and (j,i) terms in different orders and leaves about $10^{-17}$ relative asymmetry.
- Measured at hex20 $44^3$ (1.08M DOF, 85k elements): the device cells take 0.26 s stored / 1.7 s otf with 0.4 GB output (4.6 KB per element), COO → CSR 0.7 s for 21M nnz (0.25 GB), host RSS +1.5 GB. Not a bottleneck for PR 13.
- The matrix is affine in σ: $G_K + \sigma G_M$ with the masked diagonal folded in. PR 13's pre-estimate rounds may build both once and recombine.

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
  - Applied inside the jitted cycle by `jax.pure_callback(solve, ShapeDtypeStruct((n_c, k), float64), rc, vmap_method="sequential")`; the callback converts explicitly, `lu.solve(np.asarray(rc, np.float64))`, and the result is declared as `ShapeDtypeStruct(rc.shape, rc.dtype)` (always fp64: `_v` runs p-MG in fp64).
  - The factor is a static field of a small pytree `HostLU`, hashed by identity. This is the one closure exception that principle 3 allows.
  - Consequences, documented next to each other in `docs/preconditioners.md`:
    - every new `pmg` build is a new static value, so jitted callers (`pcg`, `lobpcg_step`) recompile once per build;
    - the jit cache holds the treedef, hence the factor (1.5 GB at hex20 $24^3$), and the compiled executables, until `jax.clear_caches()`; `del` and `gc.collect()` do not free it (measured 3 of 3 factors alive).
  - `tests/test_pmg.py` gets a function-scoped autouse fixture that yields, then calls `jax.clear_caches()`; `benchmarks/pmg_study.py` clears between cases. PR 13's device AMG has only array leaves and has neither issue.

### Limitations (documented, `docs/open_items.md`)

- **Curved elements.** Rigid rotations are only approximately in the coarse space (trilinear in reference coordinates is not linear in $x$). Harmless for the exact coarse solve here; relevant for PR 13's AMG near-null space.
- **Mixed meshes.** With any hex27 element the table width is $w = 8$ for every node, so hex8 and hex20 nodes pay 8 gathers. Accepted; a per-type split is not worth it before a mesh needs it.
- **Flat elements** (merged into the existing "Thin plates" open item). The hex8 coarse space locks in bending on elements with aspect ratio $a \gg 1$, so the coarse correction misses the smooth bending error and the local smoother cannot recover it:
  - single-layer plates: 15 / 21 / 63 / 699 iterations for $a$ = 1 / 2 / 4 / 10;
  - a box with $h_z = h/a$: 14 / 17 / 28 / 76;
  - slenderness alone with $a = 1$ is fine (beam $20\times2\times2$: 16).

  Revisit in PR 13 (AMG directly on the high-order level, or a near-null space that keeps bending) and in M4 (solid-shell elements).
- **`splu` scaling** (table above). Test-only, replaced in PR 13.

## Files

| file | change |
|---|---|
| `src/jaxmodal/precond/pmg.py` | new: parent/child tables, `NodalTransfer`, `_hex8_galerkin`, `HostLU`, `pmg` |
| `src/jaxmodal/precond/gmg.py` | `GMG.coarse` slot with `Cholesky`; `restrict` method on `Transfer` |
| `tests/test_gmg.py` | `P.L` → `P.coarse.L`, `_restrict` → `T.restrict` |
| `src/jaxmodal/operators/matfree.py` | `element_matrices(op, block, chunk)` |
| `src/jaxmodal/precond/__init__.py` | export `pmg` |
| `tests/test_pmg.py` | new |
| `benchmarks/pmg_study.py`, `benchmarks/results/pmg_study.json` | new: the iteration, aspect and LOBPCG tables above, plus the build time at hex20 $24^3$, the V-cycle cost in $t_A$, and `_hex8_galerkin` alone (no `splu`) at hex20 $44^3$, re-measured with the shipped code |
| `docs/preconditioners.md`, `docs/operators.md`, `README.md`, `docs/ROADMAP.md`, `docs/open_items.md` | p-MG section, shared cycle, `element_matrices`, status, open items |

## Tests (`tests/test_pmg.py`)

Common setup:
- Iteration tests use steel, x64 and PCG to $10^{-8}$ with 4 right-hand sides (`tests/helpers.rhs`).
- σ = the lowest nonzero eigenvalue of the $6^3$ box of the same type from `eigsh` (a fixed `v0`, as `tests/helpers.box` does), computed once per module and reused across sizes; the beam, free and curved cases use their own.
- Exactness tests use the PR 10 unit materials (`helpers.MATERIALS`, two phases and orientations) with σ = 0.37.
- The reference $P$ is built independently in the test: a Python loop over elements evaluates `np.prod(1 + ξ_a ξ_c)/8` at `REFERENCE_NODES`, then densified. Hand-written rows pin the formula itself: a hex20 edge node has two ½ entries, a hex27 face node four ¼, the hex27 centre eight ⅛. The trilinear reproduction (test 1) is the independent oracle.
- The reference $A_c^{-1}$ is from $P_\text{ref}^\top a_p P_\text{ref}$ (SciPy `assemble`), never from `_hex8_galerkin`.
- A function-scoped autouse fixture yields, then calls `jax.clear_caches()` (HostLU retention).

1. **Transfers** (roadmap).
   - On graded hex20 and hex27 boxes (`box_mesh(..., coords=...)`, uneven spacings), prolongation of the coarse values of a random trilinear field $f = a + bx + cy + dz + exy + fyz + gzx + hxyz$ reproduces $f$ at every fine node to $10^{-12}$.
   - Restriction equals the densified $P^\top$ to $10^{-14}$; corner rows are identity rows.
   - On `helpers.mixed_mesh` the hex8 nodes have $w = 1$ identity rows, and restriction equals the densified $P^\top$ (hex8/hex20 interface).
2. **Galerkin vs SciPy** (roadmap). `_hex8_galerkin` equals $P_\text{ref}^\top\, a_p(K + \sigma M, free, s)\, P_\text{ref}$ from `assemble` to $10^{-12}$ relative and is exactly symmetric (`(Ac != Ac.T).nnz == 0`). Cases:
   - hex20 and hex27 boxes with a mixed mask (`xmin: xyz`, `ymin: y`), random `s_K`/`s_M` and nodal masses on half the nodes;
   - stored and otf; consistent and lumped;
   - `helpers.mixed_mesh` (hex8 + hex20) with a random mask;
   - a $5\times4\times3$ box with block chunk 7 (60 elements: several chunks and 3 padded elements; the test asserts the block's `n_pad > n_el`).
   - `element_matrices` on a padded chunk returns zeros for the pads.

   All-hex8 boxes give the assembled hex8 $a_p$ itself (identity $P$).
3. **Dense V-cycle reference.** On a hex20 $3\times2\times2$ box (clamped and sliding):
   - densify $A_0$ (SciPy $a_p$), $P_\text{ref}$, a smoother $S$ built in the test as `chebyshev(ap_operator(...), block_jacobi(...), lmax_estimate(...), 3, 20.0)` (seeded), and $A_c^{-1}$;
   - compose the cycle above in NumPy with masking;
   - `pmg(..., degree=3, ratio=20.0)(U)` equals it to $10^{-10}$ on random $U$, which also checks `degree` and `ratio` forwarding.
4. **Symmetry and definiteness.**
   - $u^\top V v = v^\top V u$ to $10^{-12}$ relative on unmasked random vectors (clamped and free hex20 boxes).
   - The free block of the dense $V$ from test 3 is symmetric and its smallest eigenvalue is > 0.
   - Constrained outputs are exactly 0.
   - A pure hex8 box ($P = I$, exact coarse solve): PCG converges in ≤ 2 iterations.
   - `type(pmg(...)) is GMG` (shared cycle).
5. **h-independence** (roadmap). Iterations grow ≤ 20% from the first to the last size:
   - hex20 boxes $3^3$, $6^3$, $12^3$, $24^3$ ($24^3$ `slow`);
   - hex27 $3^3$, $6^3$, $12^3$.
6. **Robustness bounds.** Each ≤ 30 iterations (measured 14–19, ≥ 1.6× margin):
   - beam $20\times2\times2$;
   - a free $6^3$ box at σ > 0;
   - a curved distorted $6^3$ box;
   - a hex20 $6^3$ box with nodal masses of one element mass on a seeded half of the nodes (own λ₁);
   - a hex20 $4^3$ box with `s_K` = $10^{-3}$ on a checkerboard of elements (the worst measured pattern; own λ₁).
7. **LOBPCG.** `lobpcg(op, free, 6, tol=1e-8, precond=pmg(...))` on a clamped hex20 $6\times4\times3$ box matches `reference_modes` (eigenvalues to $10^{-8}$, masked modes, as `test_gmg.test_lobpcg`) in ≤ 40 iterations (measured about 15). Also on hex27.
8. **Errors.**
   - A `Stencil` raises `TypeError` with a message naming `gmg`.
   - σ < 0 and NaN raise `ValueError` ("sigma must be >= 0").
   - After deleting the `pmg` result and its outputs, `jax.clear_caches()` and `gc.collect()`, a weak reference to a used `HostLU` is dead (documents jax 0.11's retention; drop if jax changes it).
   - σ = 0 raises `SingularShiftError`.
   - x64 off raises.
9. **Jit and callback.** The V-cycle runs inside `pcg` and `lobpcg` (tests 5 and 7). A second application with the same `pmg` result reuses the compiled `pcg` (cache size unchanged). Recompiling on a new build is documented behaviour, not tested.

The iteration bounds were measured with the prototype. The implementation re-measures them with the shipped code and keeps a margin of at least 1.5×.

## Docs

- **`docs/preconditioners.md`**, a p-MG section covering:
  - the transfer table and weights, with the reference-coordinate formula and what it means on curved elements;
  - the Galerkin cell formula (above) with the owner trick;
  - `_hex8_galerkin` as PR 13's input;
  - the smoother and the V-cycle with masking;
  - the shared cycle: `GMG.coarse` takes any pytree callable (`Cholesky`, `HostLU`), transfers provide `restrict`;
  - the host `splu` coarse solve: `pure_callback`, test-only, recompiles on a new build, held by the jit cache until `jax.clear_caches()`, the scaling table and the ND alternative;
  - σ > 0 and `SingularShiftError`, with the measured cost of refusing σ = 0;
  - the measured iteration tables;
  - the flat-element limitation;
  - the tests list.
- **`docs/operators.md`**: `element_matrices`.
- **`README.md`**: p-MG in the status and feature list, with a short `lobpcg(..., precond=pmg(op, free, sigma))` snippet on a hex20 box (run by `test_readme`).
- **`docs/ROADMAP.md`**: the PR 12 status line and the flat-element limitation next to PR 13.
- **`docs/open_items.md`**:
  - flat elements, merged into the existing "Thin plates" item with the p-MG aspect tables (open; PR 13 / M4);
  - the σ > 0 requirement (decision taken without user input), its measured cost and the PR 13 hook (accept σ = 0 when no rigid modes and all `s_K > 0`);
  - curved-element rigid rotations only approximately in the coarse space (PR 13 near-null space);
  - COLAMD kept over geometric ND for the test-only `splu` (decision).

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
