# Elements

`jaxmodal.elements` holds the element registry (hex8, hex20, hex27), batched element kernels and a host validity check.

```python
from jaxmodal.elements import ELEMENTS, check_elements, lumped_mass, mass, stiffness

check_elements(mesh, mass="consistent")  # host, once per mesh
X = mesh.points[mesh.blocks["hex20"]]  # (n_el, 20, 3); chunk this for large meshes
Ke = stiffness("hex20", X, C)  # (n_el, 60, 60)
Me = mass("hex20", X, rho)  # (n_el, 60, 60)
```

- **Registry** (`registry.py`): NumPy only, built at import. It is built from the reference node tables in `jaxmodal.mesh.model` (`REFERENCE_NODES`, `EDGES`, `FACE_CORNERS`), so `jaxmodal.mesh` stays free of JAX.
- **Kernels** (`kernels.py`): pure `jax.numpy` functions of $(X_e, C_e, \rho_e)$. `jit`, `vmap` and `grad` work through them (Principle 5).
- **Check** (`check.py`): host NumPy. It raises on invalid elements.

## Conventions

- **Reference cube** $[-1, 1]^3$ with VTK node order: corners, edge midpoints, face centres, centre (`docs/mesh.md`).
- **Element DOF order** is node-major: DOF $3a + i$ is component $i$ of node $a$ (`docs/conventions.md`).
- **Voigt order** is $[11, 22, 33, 23, 13, 12]$ with engineering shear strains.
- **Element type** is passed as its name, e.g. `stiffness("hex20", X, C)`.
  - The name is a hashable static value that matches the `mesh.blocks` keys.
  - Per-type tables (quadrature, tabulated $N$, $\partial N/\partial\xi$; at most 2187 floats) are NumPy constants of that static type, embedded under `jit`.
  - Per-element arrays are always arguments.
- **Batching:**
  - `X` is `(..., n, 3)` with any leading dimensions.
  - `C` broadcasts as `(6, 6)` or `(..., 6, 6)`.
  - `rho` broadcasts as `()` or `(...)`.
  - The kernels have no loops; chunking over elements is the caller's job (PR 5).
- **dtype:**
  - Tables are cast to `jnp.result_type(X, float)`; everything else follows JAX promotion.
  - fp32 `X`, `C` and `rho` give fp32 results.
  - PR 2 materials are always default-float, so fp32 $K_e$ needs `C.astype(jnp.float32)`.
  - fp32 `X` with a default-float `C` gives fp64 output whose geometry is only fp32-accurate.

## Shape functions

Each type has a monomial space $\{\xi^p\eta^q\zeta^r : (p, q, r) \in E\}$. With the node coordinates $\xi_b$ and the Vandermonde matrix $V_{bk} = m_k(\xi_b)$,

$$N(\xi) = m(\xi)\,V^{-1}, \qquad N_a(\xi_b) = \delta_{ab}.$$

| Type | $E$ | Space | Gauss | Lumping |
|---|---|---|---|---|
| hex8 | $\{0, 1\}^3$ | trilinear | $2^3$ | row-sum |
| hex20 | $\{e \in \{0,1,2\}^3 : \text{at most one } 2\}$ | serendipity | $3^3$ | HRZ |
| hex27 | $\{0, 1, 2\}^3$ | triquadratic | $3^3$ | row-sum |

- The hex20 space is the complete quadratic plus $\xi^2\eta, \xi^2\zeta, \eta^2\xi, \eta^2\zeta, \zeta^2\xi, \zeta^2\eta$, $\xi\eta\zeta$ and $\xi^2\eta\zeta, \xi\eta^2\zeta, \xi\eta\zeta^2$: 20 monomials.
- The condition numbers of $V$ are 1, 10.5 and 33.6. The construction equals the textbook formulas to $2\times10^{-16}$, for example the serendipity corner $\tfrac18(1+\xi\xi_a)(1+\eta\eta_a)(1+\zeta\zeta_a)(\xi\xi_a+\eta\eta_a+\zeta\zeta_a-2)$.
- `Element.shape(xi)` and `Element.grad(xi)` evaluate $N$ and $\partial N/\partial\xi$ at any points (NumPy).
- `Element.N` `(nq, n)` and `Element.dN` `(nq, n, 3)` hold them tabulated at the quadrature points.
- **Quadrature:** `gauss(order)` is the tensor Gauss–Legendre rule, $\xi$ fastest. It integrates $\xi^p\eta^q\zeta^r$ exactly for $p, q, r \le 2\,\text{order} - 1$. Integration is always full (roadmap); there is no reduced integration.

## Kernels

```python
jacobian(name, X) -> (invJ (..., nq, 3, 3), wdet (..., nq))
gradients(name, invJ) -> G (..., nq, n, 3)
bmatrix(G) -> B (..., nq, 6, 3n)
stiffness(name, X, C) -> Ke (..., 3n, 3n)
mass(name, X, rho) -> Me (..., 3n, 3n)
lumped_mass(name, X, rho) -> diagonal of the lumped Me (..., 3n)
kernels.lumped_nodal(name, wdet) -> lumped nodal masses per unit density (..., n)  # PR 5 reuses it
stiffness_form(name, X, C, u, w) -> u^T Ke w (...)   # u, w: (..., 3n)
mass_form(name, X, rho, u, w) -> u^T Me w (...)
```

At each quadrature point $q$ with weight $w_q$:

$$J_{ij} = \frac{\partial x_i}{\partial \xi_j} = \sum_a X_{ai}\frac{\partial N_a}{\partial \xi_j}, \qquad G_{a} = J^{-\top}\nabla_\xi N_a, \qquad \mathrm{wdet}_q = w_q \det J_q.$$

- **Jacobian inverse:** with rows $a, b, c$ of $J$, $\det J = a\cdot(b\times c)$ and $J^{-1} = [\,b\times c,\ c\times a,\ a\times b\,]/\det J$ (columns).
  - This closed form is 18× faster than `jnp.linalg.inv` + `det` on CPU.
  - $\det J$ and the cofactors are polynomial in $J$. $J^{-1}$, $G$, $K_e$ and their gradients are finite only for $\det J \ne 0$, which `check_elements` guarantees.
  - The kernels do no sign check, since traced values can't raise.
  - $(J^{-1}, \mathrm{wdet})$, 10 floats per point, is what PR 5's on-the-fly mode stores. It rebuilds $G$ with `gradients`.
- **Strain:** $\varepsilon_v = \sum_{ij} S_{vij}\,\partial u_i/\partial x_j$. The constant selector $S_{vij}$ is 1 when $\{i, j\}$ is Voigt pair $v$ (both orderings), so shear rows give $\gamma_{ij} = u_{i,j} + u_{j,i}$. `bmatrix` builds $B$ from $S$ and $G$, with node-major columns.
- **Stiffness:** $K_e = \sum_q \mathrm{wdet}_q\,B_q^\top C B_q$ (one `einsum`), then $\tfrac12(K_e + K_e^\top)$.
  - $K_e$ is exactly symmetric; the `einsum` alone is symmetric only to $5\times10^{-16}$.
  - XLA temporaries are 6–8× the output: 30 KB per hex8 element, 241 KB per hex20 element and 323 KB per hex27 element (exactly linear in the chunk; PR 5's estimator uses these). A chunk of 65536 hex20 needs about 16 GB, so PR 5's stored-Ke chunking and estimator must count this.
  - Throughput on the M3 Max CPU is 17 µs per hex20 element (≈ 75 GFLOP/s).
- **Mass:** $M_e = \rho\, m \otimes I_3$, with scalar $m_{ab} = \sum_q \mathrm{wdet}_q N_{qa} N_{qb}$, symmetrized. PR 5's matrix-free M stores no $m$ at all: it applies $\rho\, s_M \sum_q \mathrm{wdet}_q N_{qa} \sum_b N_{qb} u_{b}$ from the stored $\mathrm{wdet}$ (or the otf geometry) and $N$ on the fly.
- **Bilinear forms:** these are the sensitivity kernels. They build neither $K_e$, $M_e$ nor $B$.
  - With $U$ = `u` reshaped to $(n, 3)$, $H_q = U^\top G_q$ is $\nabla u$ and $\varepsilon_q = S : H_q$.
  - $\texttt{stiffness\_form} = \sum_q \mathrm{wdet}_q\,\varepsilon_q(u)^\top C\,\varepsilon_q(w)$.
  - $\texttt{mass\_form} = \rho\sum_q \mathrm{wdet}_q\,(N_q U)\cdot(N_q W)$.
  - Compared with going through $B$, they use 5.5× fewer temporaries and run 2× faster.
  - Energies are $\tfrac12\,\text{form}(u, u)$. The lumped form is $\sum \bar m\, u\, w$.
- **Scalings:** $K_e$ is linear in $C$ and $M_e$ in $\rho$, so the per-element scalings $s_K$, $s_M$ (PR 5) are applied by the caller.

### Zero-energy modes

$K_e$ has exactly six zero eigenvalues, the rigid-body modes. On random elements with aspect ratio ≤ 2, node perturbations ≤ 20% (hex8) or ≤ 10% (hex20/hex27) of the shortest edge, isotropic $\nu = 0.49$ and a rotated orthotropic $C$, measured over 200 elements per type:

- the six smallest eigenvalues satisfy $|\lambda|/\lambda_\max \le 2.4\times10^{-16}$;
- the 7th is $\ge 5.3\times10^{-4}$ (hex8), $1.5\times10^{-4}$ (hex20) and $1.1\times10^{-4}$ (hex27).

The 7th eigenvalue depends on element shape: it is small for quadratic elements even undistorted.

## Lumped mass

`lumped_mass` returns the nodal masses $\bar m_a$, each repeated three times in DOF order. Both rules give $\sum_a \bar m_a = \rho V$, with $V = \sum_q \mathrm{wdet}_q$.

- **Row-sum** (hex8, hex27):

  $$\bar m_a = \rho\sum_b m_{ab} = \rho\sum_q \mathrm{wdet}_q N_{qa}.$$

- **HRZ** (hex20):

  $$\bar m_a = \rho V\, \frac{m_{aa}}{\sum_b m_{bb}},$$

  positive whenever $\det J > 0$.

Values on an undistorted box of volume $V$:

| Rule | Corner | Edge | Face | Centre |
|---|---|---|---|---|
| hex8 row-sum | $V/8$ | | | |
| hex27 row-sum | $V/216$ | $4V/216$ | $16V/216$ | $64V/216$ |
| hex20 HRZ | $7V/248$ | $16V/248$ | | |
| hex20 row-sum (not used) | $-V/8$ | $V/6$ | | |

The negative hex20 row-sum corners are why hex20 uses HRZ.

**Positivity bounds:**

- **hex8 row-sum:** always positive, because $N_a \ge 0$ inside the element.
- **hex20 HRZ:** always positive when $\det J > 0$.
- **hex27 row-sum:** positive on mildly distorted elements.
  - Measured mesh: corners perturbed ≤ 10% of the shortest edge, other nodes offset ≤ 5% of the shortest edge from their straight-edged positions.
  - At 5% offset, none of 2000 random elements went negative. At 10% offset, 2 did; at 15%, 173 did.
  - Counterexample: on the unit cube, sliding midside node 8 by 45% of the edge toward corner 0 gives corner-0 mass $-7.0\times10^{-4}$ (of 1), while $\min_q\det J = 0.043 > 0$. `check_elements(mass="lumped")` rejects it.

## Element check

```python
check_elements(mesh, mass="consistent", chunk=65536)
```

`check_elements` raises `ValueError`, naming the global element index and the type, for the first element with:

- $\det J \le 0$ at any quadrature point, or
- with `mass="lumped"`, a row-sum lumped nodal mass $\le 0$, computed with $\rho = 1$ (the sign does not depend on $\rho > 0$). HRZ needs no extra check.

How it runs:

- It works block by block, in chunks of `chunk` elements, so temporaries stay bounded: about 42 MB for X and 127 MB for J at hex27.
- It computes in NumPy fp64, independent of the x64 flag: only $J$ (`np.einsum(..., optimize=True)`), $\det J$ (the closed form shared with `jacobian`) and, for `mass="lumped"`, the row-sum masses.
- Cost on the M3 Max is about 0.6 s per 1M hex20 elements and 0.8 s per 1M hex27.
- Corner orientation of the corner hex is covered by PR 1's `corner_jacobians`. A curved hex20 can pass that check and still fail here; for example, a midside node pulled next to a corner gives $\det J < 0$ at a quadrature point.

**Callers:**

- PR 4's assembly calls `check_elements(mesh, mass)`.
- `unstructured` (and so `solve_modes`) calls it next to `check_materials`, before any build or trace.
- The kernels themselves never raise.

## Faces

`Element.faces` holds 6 tuples, in the order $-\xi, +\xi, -\eta, +\eta, -\zeta, +\zeta$. Each tuple lists:

1. The 4 corners, counter-clockwise seen from outside (outward normal by the right-hand rule):

   | Face | $-\xi$ | $+\xi$ | $-\eta$ | $+\eta$ | $-\zeta$ | $+\zeta$ |
   |---|---|---|---|---|---|---|
   | Corners | 0, 4, 7, 3 | 1, 2, 6, 5 | 0, 1, 5, 4 | 3, 7, 6, 2 | 0, 3, 2, 1 | 4, 5, 6, 7 |

2. For hex20 and hex27, the 4 edge nodes, with edge $k$ between corners $k$ and $k+1$.
3. For hex27, the face centre $20 + f$.

The hex8 faces are prefixes of the hex20 faces, which are prefixes of the hex27 faces. PR 7 (face adjacency) and later surface work use them.

## Not covered

- Wedges and tets: later, one registry entry each.
- Solid-shell (PR 16):
  - Its ANS/EAS formulation with condensation and through-ply quadrature does not fit $B^\top C B$, so its entry will carry its own kernels.
  - It adds `lumping=None`, which makes `lumped_mass` and `check_elements(mass="lumped")` raise.
- Stress recovery: PR 17 adds a per-type recovery function.
- The element-wise $M_e$ Cholesky for probes: PR 14.
