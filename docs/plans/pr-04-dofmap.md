# PR 4 plan: DOF map, assembled reference operator, masking

Branch `pr-04-dofmap`, from `main` after PR 3. Scope: the roadmap's PR 4 (Contents, Docs, Tests), plus a structured hex8/hex20/hex27 box mesh generator, which the patch tests need and PRs 5, 7 and 8 reuse (`voxel_mesh` is hex8 only and has no node sets).

## Measured facts (prototype, jax 0.11.2, fp64, M3 Max CPU)

- **Patch test** through SciPy assembly and lifting (3×3×3 elements, all boundary nodes prescribed with $u = Ax + b$, interior corner nodes perturbed by ≤ 15% of the edge, rotated orthotropic $C$, `spsolve` on the free block): the max stress error over all quadrature points, relative to $\|\sigma\|_\infty$, is
  - straight-edged (midside, face and centre nodes on the trilinear map of the corners): 2.6e-15 (hex8), 2.9e-14 (hex20), 4.1e-14 (hex27);
  - curved (non-corner interior nodes additionally offset by ≤ 5% of the edge): 3.6e-14 (hex20), 8.3e-15 (hex27).
  - So the roadmap's 1e-10 gate has ≥ 3 orders of margin.
- **Assembly** (jitted PR 3 kernels on all elements at once, COO → CSR, isotropic $C$): 30³ hex8 (89k DOF, 6.8M nonzeros) 0.67 s; 15³ hex20 (47k DOF, 7.4M nonzeros) 0.54 s; 12³ hex27 (47k DOF, 8.2M nonzeros) 0.52 s.
- **Symmetry:** the summed CSR matrix is symmetric only to 1.2e-16 relative (summation order of duplicates), although each $K_e$ is exactly symmetric; `assemble` therefore returns $\tfrac12(A + A^\top)$.

## Files

```
src/jaxmodal/mesh/box.py                 box_mesh (structured hex8/hex20/hex27 with face node sets)
src/jaxmodal/mesh/model.py               + REFERENCE_NODES, EDGES (moved from elements/registry.py)
src/jaxmodal/operators/__init__.py       re-exports the public names below
src/jaxmodal/operators/dofmap.py         check_int32, element_dofs
src/jaxmodal/operators/assembled.py      assemble (SciPy CSR K, M; the oracle)
src/jaxmodal/operators/constraints.py    dof_mask, masked, a_p
tests/test_box.py
tests/test_operators.py
docs/operators.md
```

## Conventions

- **Global DOF numbering:** DOF $3n + i$ is component $i$ of node $n$; `n_dof = 3 * len(mesh.points)`. Element DOFs follow PR 3's node-major order $3a + i$, so the element DOF array is $3\,\text{conn}[:, a] + i$.
- **Host vs device:** everything here is host NumPy/SciPy. PR 4 produces the int32 arrays that later device code takes (`element_dofs`) and the assembled reference matrices that later PRs compare against.
- **Masks** are bool arrays of length `n_dof`, `True` = free (as in the roadmap's $m \odot \cdot$).

## `mesh/box.py`

```python
box_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), type="hex8") -> Mesh
```

- A structured block of `nx × ny × nz` elements of type `hex8`, `hex20` or `hex27`. Shared nodes, VTK order (connectivity from `ELEMENTS[type].nodes`, so it is one generic code path).
- Nodes are the points of a half-spacing grid $(2n_x+1)\times(2n_y+1)\times(2n_z+1)$ used by at least one element (hex8: even indices; hex20: at most one odd index; hex27: all), in grid order (x fastest). Elements in grid order (x fastest), like `voxel_mesh`.
- Node sets `"xmin"`, `"xmax"`, `"ymin"`, `"ymax"`, `"zmin"`, `"zmax"` (the boundary faces); no element sets; labels 0; no `voxel` metadata. `shape`/`spacing`/`origin` are validated by PR 1's `check_grid`; an unknown `type` raises.
- Lives in `jaxmodal.mesh`, which stays free of jax (`docs/mesh.md`). The VTK reference node coordinates (corners, edge midpoints, face centres, centre: `REFERENCE_NODES`, (27, 3)) and the edge list `EDGES` move from `elements/registry.py` to `mesh/model.py`, which owns node ordering; the registry imports them from there (mesh → elements, never the reverse). Connectivity is `grid_index(2·cell + REFERENCE_NODES[:n] + 1)`, one code path for all types.

## `operators/dofmap.py`

```python
check_int32(**sizes)        # raises ValueError naming every size >= 2**31
element_dofs(mesh) -> dict  # type -> (n_el_type, 3 n) int32 global DOFs, VTK node order
```

- `check_int32` is the Principle 6 assert: called with named sizes (`check_int32(n_dof=..., hex20_dofs=n_el * 60)`), it raises `ValueError` listing the names and values that are ≥ $2^{31}$. It takes plain ints, so the test can pass fake sizes just above $2^{31}$ without allocating.
- `element_dofs` asserts `n_dof` and, per block, `n_el_type × 3n` (the flattened size of the element DOF array that PR 5 gathers and scatters) with `check_int32`, then casts to int32. This is the int64 → int32 cast the roadmap assigns to PR 4.

## `operators/assembled.py` (the oracle)

```python
assemble(mesh, materials, orientations=None, mass="consistent", nodal_mass=None, chunk=4096)
    -> (K, M)   # scipy.sparse.csr_array, (n_dof, n_dof), float64
```

- Host checks first: `phases(mesh)`, `check_materials(pairs, materials, orientations)`, `check_elements(mesh, mass)` (the roadmap's PR 3/PR 4 contract), and `nodal_mass` (below).
- Per block, in chunks of `chunk` elements (the last chunk padded by repeating its first element and sliced off, so a block compiles once per chunk shape): `X = points[conn]`, the phase table `C, rho = phase_table(pairs, materials, orientations)` gathered per element (`C[index]`, `rho[index]`), and the jitted PR 3 kernels `stiffness`, `mass` or `lumped_mass`.
- COO triplets (rows/cols from `element_dofs`, as int64 for SciPy) → `csr_array` (duplicates summed) → $\tfrac12(A + A^\top)$ (exactly symmetric).
- `mass="lumped"` gives a diagonal M from the PR 3 lumped diagonals (row-sum or HRZ per type).
- **Nodal (point) masses:** `nodal_mass` is `None` or a finite, non-negative `(n_nodes,)` array; mass $m_n$ is added to the three diagonal entries of node $n$ (isotropic point mass), for consistent and lumped M alike. Wrong shape, non-finite or negative values raise `ValueError`.
- It is the reference for PRs 5–8 and 12 at reference sizes (≤ ~10⁵ DOF), not a production path. Per-element scalings $s_K$, $s_M$ are added by PR 5, whose tests need them.

## `operators/constraints.py`

```python
dof_mask(mesh, bcs) -> np.ndarray (n_dof,) bool, True = free
masked(A, mask) -> D A D                         # SciPy, D = diag(mask)
a_p(A, mask) -> D A D + s (I - D)                # s = mean of diag(A) over free DOFs
```

- **BCs** (roadmap: zero displacement per component in global axes on node sets): `bcs` maps a node-set name to its constrained components, a non-empty string of distinct letters from `"xyz"`, e.g. `{"xmin": "xyz", "ymin": "y"}`. Sets may overlap (the union is constrained). An unknown set name or invalid components raise `ValueError`; `bcs = {}` means free-free (all `True`).
- **`masked`** is the eigen-operator form $m \odot A(m \odot v)$ as a matrix: constrained rows and columns are exactly zero, the free block equals $A_{ff}$.
- **`a_p`** is the roadmap's $A_P = mAm + s(I - \mathrm{diag}\,m)$ with $s$ the mean of $\mathrm{diag}(A)$ over free DOFs: SPD whenever $A_{ff}$ is (the constrained block is $sI$), unit-independent. Used by every preconditioner and linear solve from PR 10 on; PR 4 provides the assembled version as their reference. A mask with no free DOF raises.

## Tests

Fixed seeds per test. Tolerances are Principle 1's $\|a - b\|_\infty \le \mathrm{tol}\,\|b\|_\infty$. Materials: isotropic and a rotated orthotropic $C$.

`tests/test_box.py`:

1. Node and element counts per type ($(n_x+1)(n_y+1)(n_z+1)$ corners; hex20 adds edge nodes only; hex27 has $(2n_x+1)(2n_y+1)(2n_z+1)$); `check_elements(mesh, mass="lumped")` passes; every element's nodes equal `origin + spacing * (cell + (ELEMENTS[t].nodes + 1) / 2)` (affine reference map); the hex8 box equals `voxel_mesh` on points and connectivity; face sets hold exactly the nodes on each boundary plane; no orphan nodes (the `Mesh` constructor checks); unknown type and bad grids raise.

`tests/test_operators.py`:

2. **DOF map:** `element_dofs` values equal $3\,\text{conn}[:, a] + i$ in node-major order, dtype int32, one array per block. `check_int32(n_dof=2**31 - 1)` passes; `check_int32(n_dof=2**31)` and `check_int32(n_dof=5, hex20_dofs=2**31)` raise, naming the offending size.
3. **Assembly vs a dense loop reference:** on a small two-block mesh (hex8 + hex20 sharing no nodes, or two box meshes merged), with two materials and an orientation label, $K$ and $M$ (consistent and lumped) equal a dense NumPy assembly (`np.add.at` per element, per-element $C$ from `phase_table` and `index`) to 1e-14; exact symmetry (`(K != K.T).nnz == 0`); padding: a block size that is not a multiple of `chunk` (e.g. `chunk=7`) gives the same matrices.
4. **Patch test** (distorted multi-element, lifting inside the test), per type, rotated orthotropic $C$, on a 3×3×3 `box_mesh` with interior corner nodes perturbed by ≤ 15% of the edge:
   - straight-edged distortion (non-corner nodes on the trilinear map of the element corners): constant stress at every quadrature point to 1e-10 relative to $\|\sigma\|_\infty$;
   - curved hex20/hex27: non-corner interior nodes additionally offset by ≤ 5% of the edge (stated in `docs/operators.md`); constant stress at every quadrature point to 1e-10.
   - Lifting: all boundary-face nodes prescribed $u = Ax + b$ (random $A$, $b$); interior DOFs from `spsolve` of $K_{ff}u_f = -K_{fc}u_c$; stresses from PR 3's `bmatrix` and the element displacements.
5. **Clamped model:** with `bcs = {"xmin": "xyz"}`, `masked(K, m)` and `masked(M, m)` have exactly zero constrained rows and columns, and their free blocks equal SciPy's reduced `K[free][:, free]` and `M[free][:, free]` exactly (so no spurious eigenvalues enter through the mask).
6. **$A_P$:** for $A = K + \sigma M$ ($\sigma$ = 0.1·mean diag ratio) on the clamped model, $A_P$ is SPD (dense `np.linalg.cholesky` succeeds and `eigvalsh` min > 0), $\mathrm{diag}(A_P) > 0$ (Jacobi divides by no zero), constrained diagonal equals $s$ = mean of $\mathrm{diag}(A)$ over free DOFs; scaling $A$ by $10^6$ scales $A_P$ by $10^6$ (unit independence). A mask with no free DOF raises.
7. **Nodal masses:** $M(\text{nodal}) - M(\text{none})$ equals $\mathrm{diag}(\text{repeat}(m_n, 3))$ exactly (`==`), consistent and lumped. Negative, non-finite or wrongly shaped `nodal_mass` raises.
8. **Rigid translation mass:** $t_x^\top M t_x = \rho V + \sum m_n$ (and $y$, $z$) for consistent and lumped M on a distorted box with nodal masses, $V = \sum_e\sum_q \mathrm{wdet}_q$, to 1e-12.
9. **BC masks:** components per set, overlapping sets, `{}` all free; unknown set, empty or invalid components (`"xw"`, `"xx"`) raise.
10. **Host checks are called:** `assemble(..., mass="lumped")` on the PR 3 hex27 counterexample raises the lumped-mass error; a missing material label raises the `check_materials` error.

## Docs

- `docs/operators.md` (new): global DOF numbering, `element_dofs` and the int32 asserts, `assemble` (phases, chunking, symmetrization, nodal masses), BCs and masks, `masked`, $A_P$ with the formula and why (SPD, unit-independent), the patch-test bounds, measured assembly cost, and the oracle's intended size.
- `docs/mesh.md`: `box_mesh`.
- README: status line and a short snippet (box mesh → `assemble` → mask → `a_p`), executed by `tests/test_readme.py`.
- `docs/ROADMAP.md` status line: "PR 4 merged: DOF $3n + i$; BCs are `{node_set: "xyz"}`; masks are bool, True = free; `assemble` returns symmetrized SciPy CSR (oracle); `box_mesh` (hex8/hex20/hex27 with face sets) for tests; PR 5 adds $s_K$, $s_M$ to `assemble`."

## Not covered

- Device operators (PR 5, PR 6), rigid modes and deflation (PR 7), the eigen-solves (PR 8).
- $s_K$, $s_M$ in the oracle (PR 5).
- Skew or non-zero Dirichlet conditions (never; roadmap).
