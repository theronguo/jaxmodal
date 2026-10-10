# PR 3 plan: element registry and hex elements

Branch `pr-03-elements`, from `main` after PR 2. Scope: the roadmap's PR 3 (Contents, Docs, Tests). Element kernels follow Principle 5: pure `jax.numpy` functions of $(X_e, C_e, \rho_e)$.

## Measured facts (prototype, jax 0.11.2, fp64, M3 Max CPU)

Shape functions built from the inverse Vandermonde matrix of a monomial space (below):

- Kronecker property at the nodes is exact (0.0); partition of unity at the quadrature points is within 2.2e-16. The Vandermonde condition numbers are 1 (hex8), 10.5 (hex20) and 33.6 (hex27).
- Ke zero modes: on 200 random elements per type (aspect ratio 2; perturbations 20% of the short edge for hex8, 10% for hex20/hex27, all nodes; isotropic $\nu = 0.49$ and a rotated orthotropic $C$), the six smallest $|\lambda|/\lambda_\max$ are ≤ 2.4e-16. The 7th is ≥ 5.3e-4 (hex8), 1.5e-4 (hex20) and 1.1e-4 (hex27). So the roadmap gates (1e-10 and 1e-6) have ≥ 6 and ≥ 2 orders of margin.
- Ke from `einsum` is symmetric to 5e-16 relative, not exactly, so `stiffness` and `mass` symmetrize (below).
- jit vs eager Ke: identical (0.0). Central FD of $u^\top K_e(X)u$ w.r.t. $X$ (best of four steps): 1.0e-10 (hex8), 3.0e-10 (hex20), 5.5e-10 (hex27) relative.
- Row-sum lumping, undistorted unit cube: hex8 all $V/8$; hex27 min $V/216$ (corners), all positive; hex20 corners $-V/8$ (this is why hex20 uses HRZ).
- hex27 row-sum on random elements (aspect 2, corners perturbed 10% of the short edge, straight-edged midside/face/centre nodes then offset by $\epsilon$ of the short edge, 2000 trials): $\epsilon = 0.05$ never negative (min $m/m_\max$ = 5.4e-3); $\epsilon = 0.10$: 2 negative; $\epsilon = 0.15$: 173.
- Deterministic counterexample: the unit-cube hex27 with midside node 8 (edge 0–1) moved $0.45$ of the edge toward corner 0 has corner-0 row-sum mass $-7.0\times10^{-4}$ (of total 1) while $\min \det J = 0.043 > 0$ at the quadrature points.
- Throughput (jitted, batched): 10k hex20 Ke in 0.17 s, 10k hex27 Ke in 0.24 s.

## Files

```
src/jaxmodal/elements/__init__.py   re-exports the public names below
src/jaxmodal/elements/registry.py   gauss, Element, ELEMENTS (hex8, hex20, hex27)
src/jaxmodal/elements/kernels.py    gradients, bmatrix, stiffness, mass, lumped_mass, the two forms, check_elements
tests/test_elements.py
docs/elements.md
```

## Conventions

- Reference cube $[-1, 1]^3$, VTK node order (`docs/mesh.md`, `docs/conventions.md`).
- **Element DOF order** is node-major: DOF $3a + i$ is component $i$ of node $a$. PR 4's global numbering $3n + i$ matches.
- Voigt order $[11, 22, 33, 23, 13, 12]$ with engineering shear strains (`VOIGT` from `materials/rotation.py`).
- **Element type** is passed as its name (`"hex8"`, `"hex20"`, `"hex27"`), a hashable static value. Element tables (nodes, quadrature, tabulated $N$, $\partial N/\partial\xi$) are small NumPy constants of the static type, embedded as constants under `jit`; this is not data closed over in the sense of Principle 3 (per-element arrays are always arguments).
- **dtype:** the tables are cast to `jnp.result_type(X, float)`; the rest follows JAX promotion. fp32 `X`, `C`, `rho` give fp32 results (the conventions' "fp32 inputs stay fp32"); the PR 2 materials are always default-float, so fp32 Ke needs an explicit `C.astype(jnp.float32)`.
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
```

- **Shape functions** come from a monomial space $\{\xi^p\eta^q\zeta^r : (p, q, r) \in E\}$ and the node coordinates: with $V_{bk} = m_k(\xi_b)$, $N(\xi) = m(\xi)\,V^{-1}$, so $N_a(\xi_b) = \delta_{ab}$ by construction. Derivatives differentiate the monomials ($p\,\xi^{\max(p-1, 0)}$, no $0 \cdot \infty$ at $\xi = 0$).
  - hex8: $E = \{0, 1\}^3$ (trilinear), 2³ Gauss, row-sum lumping.
  - hex20: $E = \{e \in \{0, 1, 2\}^3 : \text{at most one } 2\}$ (20 monomials, the serendipity space: complete quadratic plus $\xi^2\eta, \ldots, \xi\eta\zeta, \xi^2\eta\zeta, \xi\eta^2\zeta, \xi\eta\zeta^2$), 3³ Gauss, HRZ lumping.
  - hex27: $E = \{0, 1, 2\}^3$ (triquadratic), 3³ Gauss, row-sum lumping.
  - Node coordinates are generated from the 8 corners: edge nodes are the midpoints of the `docs/mesh.md` edge list, face nodes the face centres in the order $-\xi, +\xi, -\eta, +\eta, -\zeta, +\zeta$, then the centre.
  - This is one generic construction instead of three hand-typed formula sets; the tests check it against the hand-typed formulas.
- **Faces** (Principle 4: reference faces are part of the registry entry) in the order $-\xi, +\xi, -\eta, +\eta, -\zeta, +\zeta$. Each lists the 4 corners counter-clockwise seen from outside (outward normal by the right-hand rule), then (hex20, hex27) the 4 edge nodes, edge $k$ between corners $k$ and $k+1$, then (hex27) the face centre. Corner tables are typed; edge and face nodes are looked up from the edge list. Nothing in PR 3 uses faces; PR 7 (face adjacency) and later surface work do.
- Everything here is NumPy, computed at import (no jax at import, so the x64/cache import test is unaffected).

## `kernels.py`

```python
gradients(name, X) -> (G (..., nq, n, 3), wdet (..., nq))
bmatrix(G) -> B (..., nq, 6, 3n)
stiffness(name, X, C) -> Ke (..., 3n, 3n)
mass(name, X, rho) -> Me (..., 3n, 3n)
lumped_mass(name, X, rho) -> diagonal of the lumped Me (..., 3n)
stiffness_form(name, X, C, u, w) -> u^T Ke w (...)      # u, w: (..., 3n)
mass_form(name, X, rho, u, w) -> u^T Me w (...)
check_elements(mesh, mass="consistent", chunk=65536)    # host, raises ValueError
```

With $J_{ij} = \partial x_i/\partial\xi_j = \sum_a X_{ai}\,\partial N_a/\partial\xi_j$ at each quadrature point $q$:

- **`gradients`:** $G_{qa} = J_q^{-\top}\nabla_\xi N_a(\xi_q)$ (physical gradients) and $\mathrm{wdet}_q = w_q\det J_q$ (`jnp.linalg.inv`, `jnp.linalg.det`; no sign check, kernels are traceable). These are PR 5's on-the-fly quadrature data.
- **`bmatrix`:** $\varepsilon_v = \sum_{i,j} S_{vij}\,\partial u_i/\partial x_j$ with the constant selector $S_{vij} = 1$ if $\{i, j\}$ is the Voigt pair $v$ (both orderings), else 0, so shear rows give $\gamma_{ij} = u_{i,j} + u_{j,i}$. $B = \texttt{einsum}(S, G)$ reshaped to node-major columns.
- **`stiffness`:** $K_e = \sum_q \mathrm{wdet}_q\,B_q^\top C B_q$, then $\tfrac12(K_e + K_e^\top)$, so $K_e$ is exactly symmetric (PR 5 stores one triangle; SciPy references stay symmetric).
- **`mass`:** $M_e = \rho\,m \otimes I_3$ with the scalar $m_{ab} = \sum_q \mathrm{wdet}_q N_{qa}N_{qb}$, symmetrized like $K_e$.
- **`lumped_mass`:** nodal masses $\bar m_a$, repeated 3× in DOF order.
  - Row-sum (hex8, hex27): $\bar m_a = \sum_b \rho\,m_{ab} = \rho\sum_q \mathrm{wdet}_q N_{qa}$.
  - HRZ (hex20): $\bar m_a = \rho V\,m_{aa}/\sum_b m_{bb}$ with $V = \sum_q \mathrm{wdet}_q$.
  - Both give $\sum_a \bar m_a = \rho V$ (row-sum by partition of unity).
- **Forms:** `stiffness_form` $= \sum_q \mathrm{wdet}_q\,(B_q u)^\top C\,(B_q w)$ and `mass_form` $= \rho\sum_q \mathrm{wdet}_q\,(N_q U)\cdot(N_q W)$ with $U$ = `u` reshaped to $(n, 3)$, without forming $K_e$, $M_e$. Energies are $\tfrac12$ form$(u, u)$. The lumped form is $\sum \bar m\,u\,w$ (not a function).
- **`check_elements(mesh, mass)`:** host check, once per mesh (PR 9 calls it before tracing). Per block, in chunks of `chunk` elements (bounded temporaries; points gathered per chunk, never a full `(n_el, n, 3)` array), it evaluates the jnp kernels eagerly and raises `ValueError` naming the global element index (`mesh.offsets`) and type for the first element with
  - $\det J_q \le 0$ at any quadrature point ("Jacobian check"), or
  - with `mass="lumped"`, a lumped nodal mass $\le 0$ (computed with $\rho = 1$; the sign does not depend on $\rho > 0$).
  - `mass` other than `"consistent"`/`"lumped"` raises. The check runs in the default float (fp64 under x64).
  - Validity at nodes is not checked (only quadrature points); PR 1's `corner_jacobians` covers corner orientation.

## Tests (`tests/test_elements.py`, all fp64)

References are independent of the implementation: hand-typed shape functions (hex8 trilinear; hex20 serendipity corner $(1+\xi\xi_a)(1+\eta\eta_a)(1+\zeta\zeta_a)(\xi\xi_a+\eta\eta_a+\zeta\zeta_a-2)/8$ and midside $(1-\xi^2)(1+\eta\eta_a)(1+\zeta\zeta_a)/4$ forms; hex27 products of 1D Lagrange $\xi(\xi-1)/2$, $1-\xi^2$, $\xi(\xi+1)/2$), derivatives by complex step (exact for polynomials), `np.polynomial.legendre.leggauss`, and explicit Python loops for $J$, $B$ and the quadrature sums.

Elements are built as random distorted elements: edge lengths drawn in $[1, 2]$ (aspect ≤ 2), all nodes (midside included) perturbed uniformly by ≤ 20% (hex8) or ≤ 10% (hex20/hex27) of the shortest edge, kept only if $\det J > 0$ at the nodes and quadrature points (the test asserts at least 90% are kept). Materials: isotropic $\nu = 0.49$ and a rotated orthotropic $C$ (`rotate(orthotropic(...), from_euler(...))`).

1. **Quadrature:** `gauss(n)` integrates $\xi^p\eta^q\zeta^r$ exactly (1e-14) for $p, q, r \le 2n - 1$, $n = 2, 3$; weights sum to 8.
2. **Shape functions vs hand-typed formulas** at 50 random points per type: $N$ and $\partial N/\partial\xi$ to 1e-14. Kronecker property at the nodes to 1e-14; tabulated `N`/`dN` equal `shape`/`grad` at `points`.
3. **Partition of unity and completeness** at random points: $\sum_a N_a = 1$, $\sum_a \nabla N_a = 0$; linear completeness $\sum_a \xi_a N_a = \xi$, $\sum_a \xi_a \otimes \nabla N_a = I$; quadratic completeness ($\sum_a \xi_a^2 N_a = \xi^2$, and $\xi\eta$) for hex20/hex27. All to 1e-14.
4. **NumPy reference:** `stiffness`, `mass`, `lumped_mass` and `gradients` on 5 distorted elements per type match the loop reference to 1e-12 (‖·‖∞ relative).
5. **Zero modes:** on 20 distorted elements per type and per material, the six smallest $|\lambda|$ of $K_e$ are < 1e-10·$\lambda_\max$ and the 7th > 1e-6·$\lambda_\max$ (`np.linalg.eigvalsh`). Also $\|K_e R\|_\infty \le 10^{-12}\|K_e\|_\infty\|R\|_\infty$ for the 6 analytic rigid modes (translations, rotations about the centroid).
6. **Lumping:**
   - Total mass per direction: $\sum$ `lumped_mass`$[i::3]$ and $t_i^\top M_e t_i$ ($t_i$ the unit translation in direction $i$) equal $\rho V$ with $V = \sum_q \mathrm{wdet}_q$, to 1e-12, for every type.
   - Positivity on mildly distorted elements (bounds stated in `docs/elements.md`): hex8 at the full 20% perturbation (always positive: $N_a \ge 0$ inside the element); hex20 HRZ at 10% (always positive: $m_{aa} > 0$); hex27 row-sum with corners perturbed 10% and midside/face/centre nodes offset ≤ 5% from their straight-edged positions. 50 elements each, all lumped masses > 0.
   - `check_elements(mesh, mass="lumped")` raises on a one-element hex27 mesh with the counterexample above (and passes with `mass="consistent"`); an undistorted hex20 row-sum would be negative, documented, not a code path.
7. **Gradients:** `jax.grad` of $u^\top K_e(X)u$ and of $u^\top M_e(X)u$ w.r.t. $X$ matches central FD (step $h\max(|x|, 1)$, best of $h \in \{10^{-4}, 10^{-5}, 10^{-6}\}$) to < 1e-6 relative, for each type, with the rotated orthotropic $C$. Also `jax.grad` w.r.t. the Euler angles through `rotate` (one type).
8. **Jacobian check:** `check_elements` passes on `voxel_mesh` and on the Gmsh fixtures; raises on an inverted element (an element with two opposite faces swapped, $\det J < 0$ everywhere) and on a hex8 with one corner pushed through the opposite face, naming the global element index in a two-block (hex8 + hex20) mesh; raises for an unknown `mass`.
9. **Forms:** `stiffness_form`/`mass_form` equal $u^\top K_e w$ / $u^\top M_e w$ to 1e-12 on random $u, w$, batched.
10. **Batching, jit, dtype:** a `(2, 3, n, 3)` batch with per-element `C` equals the per-element loop exactly up to 1e-14; jit equals eager to 1e-14; $K_e$ and $M_e$ are exactly symmetric (`==`); fp32 inputs give fp32 outputs.
11. **Faces:** every face node lies on its face plane; the corner cycle's normal is outward; edge node $k$ is the midpoint of corners $k$, $k+1$ (reference coordinates); the hex27 face centre is node $20 + f$; each face's node count is 4/8/9.

## Docs

- `docs/elements.md`: reference cube and node order (pointer to `docs/mesh.md`), the monomial spaces and $N = mV^{-1}$, quadrature, $J$, $G$, $B$ with the selector, $K_e$, $M_e$, lumping (row-sum, HRZ, the undistorted hex20 row-sum corner value, the measured hex27 positivity bound and the counterexample), DOF order, dtype and batching rules, `check_elements`, faces, and the measured zero-mode gaps.
- README: status line and a short element snippet (executed by `tests/test_readme.py`).
- `docs/ROADMAP.md` status line: "PR 3 merged: ...".

## Not covered

- Reduced integration (never), wedges/tets (later, one registry entry each), solid-shell (PR 16), stress recovery (PR 17's registry function), the per-element scalings $s_K$, $s_M$ (PR 5).
- Assembly and the DOF map (PR 4).
