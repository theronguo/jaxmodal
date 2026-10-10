# PR 3 plan: element registry and hex elements

Branch `pr-03-elements`, from `main` after PR 2. Scope: the roadmap's PR 3 (Contents, Docs, Tests). Element kernels follow Principle 5: pure `jax.numpy` functions of $(X_e, C_e, \rho_e)$. Review rounds 1–3 are incorporated (round 3: 5/5 satisfied).

## Measured facts (prototype and reviewer probes, jax 0.11.2, fp64, M3 Max CPU)

Shape functions built from the inverse Vandermonde matrix of a monomial space (below):

- Kronecker property at the nodes is exact (0.0); partition of unity at the quadrature points is within 2.2e-16. The Vandermonde condition numbers are 1 (hex8), 10.5 (hex20) and 33.6 (hex27). The construction matches the hand-typed serendipity/Lagrange formulas to 2.2e-16.
- **Ke zero modes:** on 200 random elements per type (aspect ratio 2; perturbations 20% of the short edge for hex8, 10% for hex20/hex27, all nodes; isotropic $\nu = 0.49$ and a rotated orthotropic $C$), the six smallest $|\lambda|/\lambda_\max$ are ≤ 2.4e-16 and the 7th is ≥ 5.3e-4 (hex8), 1.5e-4 (hex20) and 1.1e-4 (hex27): ≥ 6 and ≥ 2 orders of margin on the 1e-10 and 1e-6 gates. A transposed $J$ breaks the zero modes (6th → 5e-5–3e-4); a tensor-shear B (factor ½) or a B row order that mismatches $C$ does **not** (rigid modes have zero strain), hence test 5's patch-energy check.
- **Symmetry:** Ke from `einsum` is symmetric only to 5e-16 relative; $\tfrac12(K + K^\top)$ is exactly symmetric and fuses at no measurable cost.
- **jit vs eager Ke:** identical (0.0). Central FD of $u^\top K_e(X)u$ w.r.t. $X$ (best of four steps): 1.0e-10 (hex8), 3.0e-10 (hex20), 5.5e-10 (hex27) relative.
- **3×3 inverse/determinant:** a closed-form adjugate is 18× faster than `jnp.linalg.inv` + `det` on 1M batched matrices (0.0078 s vs 0.145 s; linalg is an unfused LAPACK/getrf custom call), agrees to 1.5e-15 (to 2e-14 vs `np.linalg.inv` for well-conditioned $J$; the gap grows with cond $J$). `jax.grad` of $\det J$ is correct at $\det J \in \{10^{-3}, 10^{-8}, 0\}$.
- **Ke cost** (single 4-operand `einsum`, 4096 elements per batch): hex20 17 µs and 241 KB of temporaries per element (Ke itself 28.8 KB); hex27 24.6 µs and 323 KB (Ke 52.5 KB). Temporaries are 6–8× the output; a two-step $CB$-first version is slower and larger. About 75 GFLOP/s, so 10M stored hex20 Ke take ≈ 3 min on the Mac.
- **Forms:** building $\nabla u = U^\top G$ instead of $B$ (8192 hex20): forward 5.9 vs 11.4 µs and 28.6 vs 158 KB temporaries per element; `grad` w.r.t. $X$ 9.7 vs 16.2 µs and 50 vs 166 KB. Agreement 2.6e-16.
- **Lumping, undistorted box of volume $V$:** hex8 row-sum $V/8$; hex27 row-sum corners $V/216$, edges $4V/216$, faces $16V/216$, centre $64V/216$; hex20 row-sum corners $-27V/216 = -V/8$, midsides $36V/216 = V/6$ (why hex20 uses HRZ); hex20 HRZ corners $7V/248$, midsides $16V/248$.
- **hex27 row-sum positivity** (aspect 2, corners perturbed 10% of the short edge, midside/face/centre nodes at their straight-edged positions then offset by $\epsilon$ of the short edge, 2000 trials): $\epsilon = 0.05$ never negative (min $m/m_\max$ = 5.4e-3); $\epsilon = 0.10$: 2 negative; $\epsilon = 0.15$: 173.
- **Counterexamples** (unit cube, reference-scaled $\det J$):
  - hex27 with midside node 8 (edge 0–1) moved $0.45$ of the edge toward corner 0: corner-0 row-sum mass $-7.0\times10^{-4}$ (of 1), while $\min_q \det J = 0.043 > 0$.
  - hex20 with midside node 8 moved to $x = 0.95$: $\min_q \det J = -0.012$, all 8 trilinear corner Jacobians positive (PR 1's `corner_jacobians` passes).
  - hex8 with corner 6 moved to $(-0.2, -0.2, -0.2)$: 4 of 8 Gauss $\det J$ negative.
- **`check_elements` cost** (NumPy, chunk 65536, per 1M hex20): `np.einsum` for $J$ takes 10.5 s with the default `optimize=False` and 0.24–0.39 s with `optimize=True`; `np.linalg.det` 4.9 s vs 0.19 s for the explicit cofactor formula (agreement 5e-16); the point gather 0.46 s; the row-sum masses 0.03 s. So ≈ 1.1 s per 1M (≈ 11 s at 10M) with `optimize=True` and the explicit determinant. Going through $G$ and `inv` instead took 6.0 s and 0.85–1.15 GB per chunk.

## Files

```
src/jaxmodal/elements/__init__.py   re-exports the public names below
src/jaxmodal/elements/registry.py   gauss, Element, ELEMENTS (hex8, hex20, hex27), private _det3; NumPy only
src/jaxmodal/elements/kernels.py    jacobian, gradients, bmatrix, stiffness, mass, lumped_mass, the two forms; pure jnp
src/jaxmodal/elements/check.py      check_elements (host NumPy)
tests/helpers.py                    tensor() moved here from tests/test_materials.py (shared)
tests/test_elements.py
docs/elements.md
```

## Conventions

- Reference cube $[-1, 1]^3$, VTK node order (`docs/mesh.md`).
- **Element DOF order** is node-major: DOF $3a + i$ is component $i$ of node $a$ (PR 4's global numbering $3n + i$ matches). Added as one line to `docs/conventions.md`.
- Voigt order $[11, 22, 33, 23, 13, 12]$ with engineering shear strains (`VOIGT` from `materials/rotation.py`).
- **Element type** is passed as its name (`"hex8"`, ...), a hashable static value matching the `mesh.blocks` keys. It stays necessary when PR 16 adds solid-shell with hex8 topology. Element tables (quadrature, tabulated $N$, $\partial N/\partial\xi$; ≤ 2187 floats) are NumPy constants of the static type, embedded under `jit`. This is the stated reading of Principle 3's "nothing is closed over": per-element arrays are always arguments, per-type tables are part of the static type. Recorded in the ROADMAP status line.
- **dtype:** tables are cast to `jnp.result_type(X, float)`; the rest follows JAX promotion. fp32 `X`, `C`, `rho` give fp32 results ("fp32 inputs stay fp32"). PR 2 materials are always default-float, so fp32 Ke needs `C.astype(jnp.float32)`; fp32 `X` with a default-float `C` gives fp64 outputs whose geometry is only fp32-accurate.
- **Batching:** every kernel takes leading batch dimensions `...` on `X` (`(..., n, 3)`) and broadcasts `C` (`(6, 6)` or `(..., 6, 6)`) and `rho` (`()` or `(...)`). Chunking over elements is the caller's (PR 5); kernels have no loops.

## `registry.py`

```python
gauss(order) -> (points (order**3, 3), weights (order**3,))   # tensor Gauss–Legendre, xi fastest

@dataclass(frozen=True, eq=False)
class Element:
    name: str
    nodes: np.ndarray      # (n, 3) reference coordinates, VTK order
    exponents: np.ndarray  # (n, 3) monomial space
    coef: np.ndarray       # (n, n) inverse Vandermonde: N(xi) = monomials(xi) @ coef
    order: int             # Gauss points per direction
    lumping: str           # "rowsum" or "hrz"
    faces: tuple           # 6 tuples of node indices, see below
    points: np.ndarray     # (nq, 3) quadrature points
    weights: np.ndarray    # (nq,)
    N: np.ndarray          # (nq, n) shape functions at the points
    dN: np.ndarray         # (nq, n, 3) reference derivatives at the points
    def shape(self, xi) -> (..., n)        # NumPy, any points
    def grad(self, xi) -> (..., n, 3)

ELEMENTS = {"hex8": ..., "hex20": ..., "hex27": ...}

_det3(J) -> det (...)   # private; explicit 3x3 determinant; indexing and arithmetic only, so it works on NumPy and jax arrays
```

- **Shape functions** come from a monomial space $\{\xi^p\eta^q\zeta^r : (p, q, r) \in E\}$ and the node coordinates: with $V_{bk} = m_k(\xi_b)$, $N(\xi) = m(\xi)\,V^{-1}$, so $N_a(\xi_b) = \delta_{ab}$ by construction. Derivatives differentiate the monomials ($p\,\xi^{\max(p-1, 0)}$, no $0 \cdot \infty$ at $\xi = 0$).
  - hex8: $E = \{0, 1\}^3$ (trilinear), 2³ Gauss, row-sum lumping.
  - hex20: $E = \{e \in \{0, 1, 2\}^3 : \text{at most one } 2\}$ (20 monomials, the serendipity space: complete quadratic plus $\xi^2\eta, \ldots$ (6), $\xi\eta\zeta$, $\xi^2\eta\zeta, \xi\eta^2\zeta, \xi\eta\zeta^2$), 3³ Gauss, HRZ lumping.
  - hex27: $E = \{0, 1, 2\}^3$ (triquadratic), 3³ Gauss, row-sum lumping.
  - Node coordinates are generated from the 8 corners: edge nodes are the midpoints of the `docs/mesh.md` edge list, face nodes the face centres in the order $-\xi, +\xi, -\eta, +\eta, -\zeta, +\zeta$, then the centre.
  - One generic construction instead of three hand-typed formula sets; the tests check it against the hand-typed formulas with hand-typed VTK coordinates.
- **Faces** (Principle 4) in the order $-\xi, +\xi, -\eta, +\eta, -\zeta, +\zeta$. Each lists the 4 corners counter-clockwise seen from outside (outward normal by the right-hand rule): $(0, 4, 7, 3)$, $(1, 2, 6, 5)$, $(0, 1, 5, 4)$, $(3, 7, 6, 2)$, $(0, 3, 2, 1)$, $(4, 5, 6, 7)$; then (hex20, hex27) the 4 edge nodes, edge $k$ between corners $k$ and $k+1$ (looked up from the edge list); then (hex27) the face centre $20 + f$. Nothing in PR 3 uses faces; PR 7 (face adjacency) and later surface work do.
- Everything here is NumPy, computed at import (no jax at import, so the x64/cache import test is unaffected).

## `kernels.py` (pure `jax.numpy`)

```python
jacobian(name, X) -> (invJ (..., nq, 3, 3), wdet (..., nq))
gradients(name, invJ) -> G (..., nq, n, 3)
bmatrix(G) -> B (..., nq, 6, 3n)
stiffness(name, X, C) -> Ke (..., 3n, 3n)
mass(name, X, rho) -> Me (..., 3n, 3n)
lumped_mass(name, X, rho) -> diagonal of the lumped Me (..., 3n)
stiffness_form(name, X, C, u, w) -> u^T Ke w (...)      # u, w: (..., 3n)
mass_form(name, X, rho, u, w) -> u^T Me w (...)
```

With $J_{ij} = \partial x_i/\partial\xi_j = \sum_a X_{ai}\,\partial N_a/\partial\xi_j$ at each quadrature point $q$:

- **`jacobian`:** $\det J$ = `_det3(J)` $= a\cdot(b \times c)$ and $J^{-1} = \mathrm{cof}(J)^\top/\det J$ in closed form (rows $a, b, c$ of $J$; the columns of $J^{-1}$ are $(b \times c, c \times a, a \times b)/\det J$), and $\mathrm{wdet}_q = w_q\det J_q$. $\det J$ and $\mathrm{cof}(J)$ are polynomial in $J$; $J^{-1}$ (and $G$, $K_e$ and their gradients) is finite only for $\det J \ne 0$, which `check_elements` guarantees ($\det J > 0$). No sign check here (traceable). $(J^{-1}, \mathrm{wdet})$ is the Jacobian-level data (10 floats per point) that PR 5's on-the-fly mode stores ("invJ·detJ per quadrature point"); it rebuilds $G$ with `gradients`.
- **`gradients`:** $G_{qa} = J_q^{-\top}\nabla_\xi N_a(\xi_q)$, i.e. `einsum` of the tabulated `dN` with `invJ`.
- **`bmatrix`:** $\varepsilon_v = \sum_{i,j} S_{vij}\,\partial u_i/\partial x_j$ with the constant selector $S_{vij} = 1$ if $\{i, j\}$ is the Voigt pair $v$ (both orderings), else 0, so shear rows give $\gamma_{ij} = u_{i,j} + u_{j,i}$. $B = \texttt{einsum}(S, G)$ reshaped with node axis before component axis (DOF $3a + i$).
- **`stiffness`:** $K_e = \sum_q \mathrm{wdet}_q\,B_q^\top C B_q$ (one 4-operand `einsum`), then $\tfrac12(K_e + K_e^\top)$, exactly symmetric (PR 5 stores one triangle; SciPy references stay symmetric). Temporaries are 6–8× the output (measured facts); `docs/elements.md` states this for PR 5's chunking and estimator.
- **`mass`:** $M_e = \rho\,m \otimes I_3$ with the scalar $m_{ab} = \sum_q \mathrm{wdet}_q N_{qa}N_{qb}$, symmetrized. `docs/elements.md` notes that stored or on-the-fly M (PR 5) should keep the scalar $n \times n$ $m$ (9× smaller); no separate function until PR 5 needs it.
- **`lumped_mass`:** nodal masses $\bar m_a$, repeated 3× in DOF order.
  - Row-sum (hex8, hex27): $\bar m_a = \rho\sum_q \mathrm{wdet}_q N_{qa}$ ($= \sum_b \rho\,m_{ab}$ by partition of unity).
  - HRZ (hex20): $\bar m_a = \rho V\,m_{aa}/\sum_b m_{bb}$ with $V = \sum_q \mathrm{wdet}_q$.
  - Both give $\sum_a \bar m_a = \rho V$.
- **Forms** (the Principle 5 sensitivity kernels), without forming $K_e$, $M_e$ or $B$: with $U$ = `u` reshaped to $(n, 3)$, $H_q = U^\top G_q$ ($\nabla u$), $\varepsilon_q = \texttt{einsum}(S, H_q)$:
  - `stiffness_form` $= \sum_q \mathrm{wdet}_q\,\varepsilon_q(u)^\top C\,\varepsilon_q(w)$.
  - `mass_form` $= \rho\sum_q \mathrm{wdet}_q\,(N_q U)\cdot(N_q W)$.
  - Energies are $\tfrac12$ form$(u, u)$. The lumped form is $\sum \bar m\,u\,w$ (not a function).

## `check.py` (host NumPy)

```python
check_elements(mesh, mass="consistent", chunk=65536)    # raises ValueError
```

- Per block, in chunks of `chunk` elements (points gathered per chunk, never a full `(n_el, n, 3)` array), it computes in NumPy fp64 (independent of the x64 flag) only $J = $ `np.einsum(..., optimize=True)` (the default `optimize=False` is 30× slower), $\det J$ = `_det3(J)` (shared with `jacobian`; `np.linalg.det` is 25× slower), and for `mass="lumped"` with row-sum types, $\bar m = (w \det J)\,N$. It raises naming the global element index (`mesh.offsets[t]` + chunk start + row) and the type for the first element with
  - $\det J_q \le 0$ at any quadrature point (the Jacobian check), or
  - with `mass="lumped"`: a row-sum lumped nodal mass $\le 0$ ($\rho = 1$; the sign does not depend on $\rho > 0$). HRZ masses are positive whenever $\det J > 0$, so they need no extra check.
- `mass` other than `"consistent"`/`"lumped"` raises.
- Eager NumPy, not jit: no device round trip, no recompile for the last chunk; ≈ 1.1 s per 1M hex20 (≈ 11 s at 10M), per-chunk temporaries ≈ 42 MB (X) and 127 MB (J) at hex27.
- Only quadrature points are checked (the roadmap's "Jacobian check at quadrature points"); corner orientation is PR 1's `corner_jacobians`. Voxel meshes are not special-cased.
- **Callers:** the roadmap's "Lumping raises an error when any lumped nodal mass is ≤ 0" is implemented here, not in the traceable `lumped_mass` (traced values can't raise). PR 4's assembly calls `check_elements(mesh, mass)` before assembling, and PR 9's `solve_modes` calls it next to `check_materials` before tracing. The ROADMAP PR 3 test line, PR 4 Contents and PR 9 Contents are updated to say so.

## Tests (`tests/test_elements.py`, all fp64)

References are independent of the implementation: hand-typed VTK reference coordinates for all 27 nodes (not `Element.nodes`), hand-typed shape functions (hex8 trilinear; hex20 serendipity corner $(1+\xi\xi_a)(1+\eta\eta_a)(1+\zeta\zeta_a)(\xi\xi_a+\eta\eta_a+\zeta\zeta_a-2)/8$ and midside $(1-\xi^2)(1+\eta\eta_a)(1+\zeta\zeta_a)/4$ forms; hex27 products of 1D Lagrange $\xi(\xi-1)/2$, $1-\xi^2$, $\xi(\xi+1)/2$), derivatives by complex step (exact for polynomials), `np.polynomial.legendre.leggauss`, `np.linalg.inv`/`det`, explicit Python loops for $J$, $B$ and the quadrature sums, and the 4th-order tensor $C_{ijkl}$ (`tests/helpers.py`).

**Random distorted elements:** fixed `np.random.default_rng(<const>)` per test. Edge lengths drawn in $[1, 2]$ (aspect ≤ 2); all nodes (midside included) perturbed uniformly by ≤ 20% (hex8) or ≤ 10% (hex20/hex27) of the shortest edge; kept only if $\det J > 0$ at the nodes and at the quadrature points (the test asserts ≥ 90% kept; measured 98–100%). Materials: isotropic $\nu = 0.49$ and a rotated orthotropic $C$ (`rotate(orthotropic(...), from_euler(...))`). Tolerances are Principle 1's $\|a - b\|_\infty \le \mathrm{tol}\,\|b\|_\infty$.

1. **Quadrature:** `gauss(n)` integrates $\xi^p\eta^q\zeta^r$ exactly (1e-14) for $p, q, r \le 2n - 1$, $n = 2, 3$; weights sum to 8.
2. **Shape functions vs hand-typed formulas** at 50 random points per type: $N$ and $\partial N/\partial\xi$ to 1e-14. `Element.nodes` equals the hand-typed coordinates. Tabulated `N`/`dN` equal `shape`/`grad` at `points`.
3. **Partition of unity and completeness** at random points: $\sum_a N_a = 1$, $\sum_a \nabla N_a = 0$; linear completeness $\sum_a \xi_a N_a = \xi$, $\sum_a \xi_a \otimes \nabla N_a = I$; quadratic completeness ($\xi^2$, $\xi\eta$) for hex20/hex27. All to 1e-14.
4. **NumPy reference:** `jacobian`, `gradients`, `stiffness`, `mass` and `lumped_mass` on 5 distorted elements per type match the loop reference to 1e-12. The closed-form `jacobian` also matches `np.linalg.inv` (1e-12) and `np.linalg.det` (1e-14) on well-conditioned random $J = I + 0.3\,\mathcal N(0, 1)$ (asserting cond $J < 10$; Principle 1 norms over the whole batch, not per matrix), and `check.py`'s $\det J$ equals `wdet / w` from `jacobian` on the distorted elements to 1e-14 (no drift between the host and traced paths).
5. **Patch energy (independent of Voigt):** for an affine $u = Hx$ with random $H$ on distorted elements (rotated orthotropic $C$), $u^\top K_e u = V\,\varepsilon_{ij}C_{ijkl}\varepsilon_{kl}$ with $\varepsilon = \mathrm{sym}(H)$ and $V = \sum_q \mathrm{wdet}_q$, to 1e-12, every type. Catches a tensor-shear factor, a row-order mismatch with $C$, a wrong selector and a transposed $J$.
6. **Zero modes:** on 20 distorted elements per type and per material, the six smallest $|\lambda|$ of $K_e$ are < 1e-10·$\lambda_\max$ and the 7th > 1e-6·$\lambda_\max$ (`np.linalg.eigvalsh`). Also $\|K_e R\|_\infty \le 10^{-12}\|K_e\|_\infty\|R\|_\infty$ for the 6 analytic rigid modes.
7. **Lumping:**
   - Closed-form values on an undistorted box (measured facts): hex8, hex27 row-sum and hex20 HRZ per node to 1e-14; the hex20 row-sum corner value $-V/8$ (computed in-test as $\rho\sum_q\mathrm{wdet}_qN_{qa}$, documenting why HRZ).
   - Total mass per direction: $\sum$ `lumped_mass`$[i::3]$ and $t_i^\top M_e t_i$ ($t_i$ the unit translation in direction $i$) equal $\rho V$ to 1e-12, every type.
   - Positivity on mildly distorted elements (bounds stated in `docs/elements.md`), 50 elements each: hex8 at the full 20% perturbation (always positive, $N_a \ge 0$); hex20 HRZ at 10% (always positive, $m_{aa} > 0$); hex27 row-sum with corners perturbed ≤ 10% and midside/face/centre nodes offset ≤ 5% of the shortest edge from their straight-edged positions. All lumped masses > 0.
   - `check_elements(mesh, mass="lumped")` raises on the one-element hex27 counterexample; with `mass="consistent"` it passes.
8. **Gradients:** `jax.grad` of $u^\top K_e(X)u$ and of $u^\top M_e(X)u$ w.r.t. $X$ match central FD, $\|g_\text{ad} - g_\text{fd}\|_\infty \le 10^{-6}\|g_\text{fd}\|_\infty$, step $h\max(|x|, 1)$ with one $h$ for the whole gradient, best of $h \in \{10^{-4}, 10^{-5}, 10^{-6}\}$; every type, rotated orthotropic $C$. Also `jax.grad` w.r.t. the Euler angles through `rotate` (one type).
9. **Jacobian check:**
   - Passes on `voxel_mesh((3, 2, 2), ...)` and on the volume fixtures `gmsh_pair.msh` (hex20) and `gmsh_messy.msh`.
   - Raises on: a mirrored element ($X \to X\,\mathrm{diag}(1, 1, -1)$, connectivity unchanged, $\det J < 0$ everywhere; every type); the hex8 corner-6 push-through; and the hex20 quadrature-only failure (asserting `corner_jacobians` is positive there, so this is beyond PR 1's check).
   - Global index: a two-block (hex8 + hex20) mesh with the bad element at local index ≥ 1 in the hex20 block, called with `chunk=2` so it falls in a later chunk; the message contains the exact global index (offset + local) and `"hex20"`.
   - Raises for an unknown `mass`.
10. **Forms:** `stiffness_form`/`mass_form` equal $u^\top K_e w$ / $u^\top M_e w$ to 1e-12 on random $u, w$, batched.
11. **Batching, jit, dtype:** a `(2, 3, n, 3)` batch with per-element `C` and `rho` matches the per-element loop to 1e-14; jit matches eager to 1e-14; $K_e$ and $M_e$ are exactly symmetric (`==`); fp32 inputs give fp32 outputs.
12. **Faces:** every face node lies on its face plane; consecutive corners (cyclically) form an edge of the `docs/mesh.md` edge list; the corner cycle's normal is outward; edge node $k$ is the midpoint of corners $k$, $k+1$; the hex27 face centre is node $20 + f$; hex8 faces are the first four entries of the hex20 faces, and hex20 faces the first eight of hex27; node counts 4/8/9.

## Docs

- `docs/elements.md`: reference cube and node order (pointer to `docs/mesh.md`), monomial spaces and $N = mV^{-1}$, quadrature, $J$, the adjugate, $G$, $B$ with the selector, $K_e$, $M_e = \rho\,m \otimes I_3$ (store the scalar $m$), the forms, lumping (row-sum, HRZ, the closed-form values, the hex27 positivity bound and counterexample), DOF order, dtype and batching, `check_elements` and its callers, faces with the corner tables, Ke temporaries (6–8× output) and the measured zero-mode gaps.
- `docs/conventions.md`: element DOF order $3a + i$.
- `docs/mesh.md`: the corner-Jacobian section points to `check_elements` for curved-element validity.
- README: status line and a short element snippet (executed by `tests/test_readme.py`).
- `docs/ROADMAP.md`:
  - PR 3 test line: the lumped-mass raise is `check_elements(mass="lumped")`, a host check.
  - PR 4 Contents: assembly calls `check_elements(mesh, mass)`.
  - PR 9 Contents: `solve_modes` calls `check_materials` and `check_elements` before tracing.
  - PR 5 Contents: "precomputed invJ and w·detJ per quadrature point" (10 floats, from `jacobian`).
  - Status line: "PR 3 merged: element type = static name, per-type tables embedded as constants; element DOF order $3a + i$; `check_elements` is the host Jacobian/lumping check (PR 4, PR 9 call it); `jacobian` gives PR 5's invJ and w·detJ."

## Not covered

- Reduced integration (never); wedges/tets (later, one registry entry each).
- Solid-shell (PR 16): ANS/EAS with condensation and through-ply quadrature are not $B^\top C B$ with a standard $B$, so its registry entry will carry its own kernel callables (or `stiffness` dispatches on the name). PR 16 also adds `lumping=None` with its raise in `lumped_mass` (trace time, the name is static) and in `check_elements`, and their tests. Stress recovery (PR 17) adds a per-type recovery callable. Neither field is added now.
- Element-wise $M_e$ Cholesky for probes (PR 14, lives in `elements/`).
- The per-element scalings $s_K$, $s_M$ (PR 5; $K_e$ is linear in $C$, $M_e$ in $\rho$, so the caller multiplies).
- Assembly and the DOF map (PR 4).
