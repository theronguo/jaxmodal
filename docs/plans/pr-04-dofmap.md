# PR 4 plan: DOF map, assembled reference operator, masking

Branch `pr-04-dofmap`, from `main` after PR 3. (Code review superseded a few details below: `face_sets(d, idx, top)` works per axis, the default `chunk` is 1024, and the measured memory figures are in `docs/operators.md`.) Scope: the roadmap's PR 4 (Contents, Docs, Tests), plus a structured hex8/hex20/hex27 box mesh generator, which the patch tests need and PRs 5, 7 and 8 reuse (`voxel_mesh` is hex8 only and has no node sets).

## Measured facts (prototype, jax 0.11.2, fp64, M3 Max CPU)

- **Patch test** through SciPy assembly and lifting (3×3×3 elements, all boundary nodes prescribed with $u = Ax + b$, interior corner nodes perturbed per component by ≤ 15% of the edge, rotated orthotropic $C$, `spsolve` on the free block): the max stress error over all quadrature points, relative to $\|\sigma\|_\infty$, is
  - straight-edged (midside, face and centre nodes on the trilinear map of the corners): 2.6e-15 (hex8), 2.9e-14 (hex20), 4.1e-14 (hex27);
  - curved (non-corner interior nodes additionally offset by ≤ 5% of the edge): 3.6e-14 (hex20), 8.3e-15 (hex27).
  - So the roadmap's 1e-10 gate has ≥ 3 orders of margin.
- **Assembly** (jitted PR 3 kernels on all elements at once, COO → CSR, isotropic $C$): 30³ hex8 (89k DOF, 6.8M nonzeros) 0.67 s; 15³ hex20 (47k DOF, 7.4M nonzeros) 0.54 s; 12³ hex27 (47k DOF, 8.2M nonzeros) 0.52 s.
- **Symmetry:** the summed CSR matrix is symmetric only to 1.2e-16 relative (summation order of duplicates), although each $K_e$ is exactly symmetric; `assemble` therefore returns $\tfrac12(A + A^\top)$.

## Files

```
src/jaxmodal/mesh/box.py                 box_mesh (structured hex8/hex20/hex27 with face node sets)
src/jaxmodal/mesh/model.py               + CORNERS, EDGES, FACE_CORNERS, REFERENCE_NODES (moved from
                                         elements/registry.py), face_sets (shared by box and voxel)
src/jaxmodal/mesh/voxel.py               voxel_mesh gains the same face node sets
src/jaxmodal/elements/registry.py        imports the tables; nodes = REFERENCE_NODES[:n]
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
- **Host vs device:** `element_dofs`, the masks and $A_P$ are host NumPy/SciPy. `assemble` evaluates the jitted PR 3 kernels per chunk and assembles on the host; it requires x64 (`jax.config.jax_enable_x64`) and raises `ValueError` otherwise, so the oracle is never silently fp32. PR 4 produces the int32 element DOF arrays and the assembled reference matrices that later PRs compare against.
- **int32 casts:** `check_int32` is the one overflow assert. PR 4 applies it to `element_dofs`; every later PR that sends a host index array to the device (PR 5: the element → phase `index`; PR 6: the voxel maps) passes its sizes through `check_int32` and casts there. `docs/conventions.md` and `docs/materials.md` are updated to say so.
- **BC format:** `bcs = {node_set: "xyz"}` everywhere; `dof_mask` is the single place it is parsed, and PR 9's `solve_modes(mesh, materials, bcs, ...)` takes the same dict.
- **Nodal masses** everywhere (PRs 5, 6, 9, 11, 14, 21) are the kwarg `nodal_mass`, a `(n_nodes,)` array, isotropic: repeated ×3 onto the node's DOFs.
- **Masks** are bool arrays of length `n_dof`, `True` = free (as in the roadmap's $m \odot \cdot$).

## `mesh/box.py`

```python
box_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), type="hex8") -> Mesh
```

- A structured block of `nx × ny × nz` elements of type `hex8`, `hex20` or `hex27`. Shared nodes, VTK order.
- Nodes are the points of a half-spacing grid $(2n_x+1)\times(2n_y+1)\times(2n_z+1)$ used by at least one element (hex8: even indices; hex20: at most one odd index; hex27: all), in grid order (x fastest). Elements in grid order (x fastest), like `voxel_mesh`.
- Node sets `"xmin"`, `"xmax"`, `"ymin"`, `"ymax"`, `"zmin"`, `"zmax"` (the boundary faces, from the integer grid index, so exact) via `face_sets(ijk, top)` in `mesh/model.py`; no element sets; labels 0; no `voxel` metadata. `shape`/`spacing`/`origin` are validated by PR 1's `check_grid`; an unknown `type` raises.
- Lives in `jaxmodal.mesh`, which stays free of jax (`docs/mesh.md`); importing `jaxmodal.elements` would pull in jax through `kernels`. So `CORNERS`, `EDGES`, `FACE_CORNERS` and the new `REFERENCE_NODES` ((27, 3): corners, edge midpoints, face centres, centre, VTK order) move from `elements/registry.py` to `mesh/model.py`, which already owns node ordering. The registry imports them (elements → mesh, never the reverse), keeps `FACE_CORNERS` for `faces`, and takes `nodes = REFERENCE_NODES[:n]`. Connectivity is `grid_index(2·cell + REFERENCE_NODES[:n] + 1)`, one code path for all types.
- `voxel_mesh` gets the same six face sets (grid-plane nodes that survive the occupancy mask) through `face_sets`, so clamped voxel models (PRs 6, 11, 14, 23) need no hand-built node sets.
- Uniform spacing only. PR 8's graded NAFEMS meshes extend `spacing` to per-axis node-coordinate arrays (remapping points afterwards would move midside nodes off the edge midpoints).

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
# imports the PR 3 kernels as a module (`kernels.mass`), since the `mass=` argument shadows the name
    -> (K, M)   # scipy.sparse.csr_array, (n_dof, n_dof), float64
```

- Host checks first: x64 on, `phases(mesh)`, `check_materials(pairs, materials, orientations)`, `check_elements(mesh, mass)` (the roadmap's PR 3/PR 4 contract), and `nodal_mass` (below).
- Per block, in chunks of `c = min(chunk, n_el_type)` elements (the last chunk padded by repeating its first element and sliced off), one compile per (type, c); small blocks compile at their own size (≈ 0.1 s) instead of padding to 4096; `chunk` bounds the $K_e$ temporaries, not total memory (the COO triplets of the whole mesh dominate above ~10⁵ DOF): `X = points[conn]`, the phase table `C, rho = phase_table(pairs, materials, orientations)` gathered per element with this block's slice of `index` (`index[offsets[t] + s : ...]`), and the jitted PR 3 kernels `stiffness`, `mass` or `lumped_mass`.
- Rows/cols are built once as int32 from `element_dofs` (safe after `check_int32(n_dof)`; SciPy keeps int32 CSR indices, 12 B per nonzero instead of 16). Then K and M are built **one at a time**: values → COO → `csr_array` (duplicates summed) → $\tfrac12(A + A^\top)$ (exactly symmetric) → free temporaries. Measured on a hex8 60³ prototype: peak 6.1 GB instead of 10.4 GB with int64 indices and both matrices alive (≈ 50 B per element-matrix entry), same wall time (4.6 s).
- `mass="lumped"` gives a diagonal M from the PR 3 lumped diagonals (row-sum or HRZ per type).
- **Nodal (point) masses:** `nodal_mass` is `None` or a finite, non-negative `(n_nodes,)` array; after symmetrization, `M + diags(repeat(m, 3))` adds $m_n$ to the three diagonal entries of node $n$ (isotropic point mass), for consistent and lumped M alike. Adding last makes `M(nodal) == M(none) + diags(...)` exact. Wrong shape, non-finite or negative values raise `ValueError`.
- It is the reference for PRs 5–8 and 12 at reference sizes (≤ ~10⁵ DOF), not a production path. The limit comes from PR 8's `splu`, not assembly: on hex20 20³ (107k DOF), `splu(K − σM)` took 241 s with a 16.5 GB peak, while assembly at 10⁵ DOF needs ≈ 1.5 GB (≈ 9/15/13 KB per DOF for hex8/hex20/hex27). `assemble` is the oracle only; PR 13's production hex8 $A_P$ for AMG needs its own chunked assembler into a precomputed CSR pattern (≈ 1 GB per matrix per 1M hex8 DOF; `assemble` would need ≈ 90 GB at 10M DOF). The roadmap's PR 13 entry gets this sentence. Per-element scalings $s_K$, $s_M$ are added by PR 5, whose tests need them.

## `operators/constraints.py`

```python
dof_mask(mesh, bcs) -> np.ndarray (n_dof,) bool, True = free
masked(A, mask) -> D A D                         # SciPy, D = diag(mask)
a_p(A, mask, s=None) -> D A D + s (I - D)        # default s = mean of diag(A) over free DOFs
```

- **BCs** (roadmap: zero displacement per component in global axes on node sets): `bcs` maps a node-set name to its constrained components, a non-empty string of distinct letters from `"xyz"`, e.g. `{"xmin": "xyz", "ymin": "y"}`. Sets may overlap (the union is constrained). An unknown set name or invalid components raise `ValueError`; `bcs = {}` means free-free (all `True`).
- **`masked`** is the eigen-operator form $m \odot A(m \odot v)$ as a matrix: constrained rows and columns are exactly zero, the free block equals $A_{ff}$.
- **`a_p`** is the roadmap's $A_P = mAm + s(I - \mathrm{diag}\,m)$ with $s$ the mean of $\mathrm{diag}(A)$ over free DOFs: SPD whenever $A_{ff}$ is (the constrained block is $sI$), unit-independent. Used by every preconditioner and linear solve from PR 10 on; PR 4 provides the assembled version as their reference. A mask with no free DOF raises, and so does a default $s \le 0$ (mean diagonal of an indefinite $A$), so `a_p` never silently returns a non-SPD $A_P$. The default $s$ is for SPD $A = K + \sigma M$; for indefinite shift-invert $K - \sigma M$ (PRs 10, 15) the roadmap (Key design decisions, interior bands / inner MINRES) fixes $s = \overline{\mathrm{diag} K} + \sigma\,\overline{\mathrm{diag} M}$ over free DOFs, which the caller passes as `s` (documented in `docs/operators.md`).

## Tests

Fixed seeds per test. Tolerances are Principle 1's $\|a - b\|_\infty \le \mathrm{tol}\,\|b\|_\infty$. Materials: isotropic and a rotated orthotropic $C$.

`tests/test_box.py`:

1. Grid order of nodes (x fastest) and elements (cell index $i + n_x(j + n_y k)$); node and element counts per type ($(n_x+1)(n_y+1)(n_z+1)$ corners; hex20 adds edge nodes only; hex27 has $(2n_x+1)(2n_y+1)(2n_z+1)$); `check_elements(mesh, mass="lumped")` passes; every element's nodes equal `origin + spacing * (cell + (ELEMENTS[t].nodes + 1) / 2)` (affine reference map); the hex8 box equals `voxel_mesh` on points, connectivity and node sets; face sets hold exactly the nodes whose coordinate equals the boundary plane (computed from `points` in the test, with dyadic spacing and origin so the comparison is exact), for box meshes and a voxel mesh whose mask removes some boundary-plane cells; no orphan nodes (the `Mesh` constructor checks); unknown type and bad grids raise; `import jaxmodal.mesh` in a subprocess leaves `"jax"` out of `sys.modules`.

`tests/test_operators.py`:

2. **DOF map:** `element_dofs` values equal $3\,\text{conn}[:, a] + i$ in node-major order, dtype int32, one array per block. `check_int32(n_dof=2**31 - 1)` passes; `check_int32(n_dof=2**31)`, `check_int32(n_dof=2**31 + 1)` and `check_int32(n_dof=5, hex20_dofs=2**31, hex8_dofs=2**32)` raise, naming every offending size and not the passing one. A monkeypatched `check_int32` records that `element_dofs` calls it with `n_dof` and `<type>_dofs = n_el_type · 3n` for every block.
3. **Assembly vs a dense loop reference:** a two-block mesh built in the test from a 2×2×2 hex20 `box_mesh` and a 2×2×2 hex8 `box_mesh` glued at a shared face (points merged by grid coordinate, so the blocks share nodes), all nodes perturbed per component by ≤ 5% of the edge (`check_elements` asserted first, so a failure means an assembly bug, not a bad seed); the hex20/hex8 interface is deliberately nonconforming and only exercises DOFs shared across blocks; labels mixed within each block in a non-periodic pattern: two materials with different $\rho$ (one anisotropic) and two orientations (identity and a non-trivial rotation). The reference is a dense Python loop with explicit per-element DOF lists `[3n + i for n in conn for i in range(3)]` and per-element $C$ = `rotate(material, orientation)` (not `phases`, `phase_table` or `element_dofs`), PR 3 kernels per element, `+=` into a dense matrix. $K$ and $M$ (consistent and lumped) agree to 1e-14 for `chunk=7` (several chunks and a padded last one) and `chunk=4096` (> n_el); exact symmetry (`(K != K.T).nnz == 0`); K and M have int32 indices.
4. **Patch test** (distorted multi-element, lifting inside the test), per type, rotated orthotropic $C$, on a 3×3×3 `box_mesh` with interior corner nodes perturbed per component by ≤ 15% of the edge:
   - straight-edged distortion (non-corner nodes on the trilinear map of the element corners): constant stress at every quadrature point to 1e-10 relative to $\|\sigma\|_\infty$;
   - curved hex20/hex27: non-corner interior nodes additionally offset by ≤ 5% of the edge (stated in `docs/operators.md`); constant stress at every quadrature point to 1e-10.
   - Lifting: all boundary-face nodes prescribed $u = Ax + b$ (random $A$, $b$); interior DOFs from `spsolve` of $K_{ff}u_f = -K_{fc}u_c$; stresses from PR 3's `bmatrix` and the element displacements.
5. **Clamped model:** with `bcs = {"xmin": "xyz"}` (expected constrained DOFs computed from `points[:, 0] == 0`), `masked(K, m)` and `masked(M, m)` have exactly zero constrained rows and columns, and their free blocks equal SciPy's reduced `K[free][:, free]` and `M[free][:, free]` exactly (so no spurious eigenvalues enter through the mask).
6. **$A_P$:** for $A = K + \sigma M$ ($\sigma$ = 0.1·mean diag ratio) on the clamped model, $A_P$ is SPD (dense `np.linalg.cholesky` succeeds and `eigvalsh` min > 0), $\mathrm{diag}(A_P) > 0$ (Jacobi divides by no zero), constrained diagonal equals $s$ = mean of $\mathrm{diag}(A)$ over the coordinate-based free set; scaling $A$ by $2^{20}$ scales $A_P$ by exactly $2^{20}$ (unit independence); an all-True mask gives $A$ exactly; an explicit `s` sets the constrained diagonal to `s`. A mask with no free DOF, or a default $s \le 0$, raises.
7. **Nodal masses:** with random $m_n \ge 0$, `M(nodal) == M(none) + diags(repeat(m, 3))` exactly (no nonzero differences), consistent and lumped. Negative, non-finite or wrongly shaped `nodal_mass` raises.
8. **Rigid translation mass:** per type, on a box with interior nodes perturbed (straight and, for hex20/hex27, curved; boundary nodes fixed so $V = L_xL_yL_z$ analytically) and nodal masses, $t_x^\top M t_x = \rho L_xL_yL_z + \sum m_n$ (and $y$, $z$) for consistent and lumped M to 1e-12, and $t_x^\top M t_y = 0$.
9. **BC masks:** expected masks from node coordinates; components per set, overlapping sets, `{}` all free; unknown set, empty or invalid components (`"xw"`, `"xx"`) raise.
10. **Host checks are called:** `assemble(..., mass="lumped")` on the PR 3 hex27 counterexample raises the lumped-mass error; a missing material label raises the `check_materials` error. With x64 disabled (`jax.config.update` inside the test, restored in `finally`), `assemble` raises.

## Docs

- `docs/operators.md` (new): global DOF numbering, `element_dofs` and the int32 asserts, `assemble` (phases, chunking, symmetrization, nodal masses), BCs and masks, `masked`, $A_P$ with the formula and why (SPD, unit-independent), the patch-test bounds, measured assembly cost and peak memory (≈ 50 B per element-matrix entry), and the oracle's size limit (set by `splu`) with the PR 13 note.
- `docs/mesh.md`: `box_mesh`; voxel face sets (the Mask bullet's "there are no sets" goes; the Boundary sets example becomes `node_sets={**mesh.node_sets, "clamp": clamp}`; on a masked voxel mesh `xmin`…`zmax` are the bounding grid planes, not the boundary of the occupied region, and may be nearly or fully empty); "Node ordering" names `CORNERS`, `EDGES`, `FACE_CORNERS`, `REFERENCE_NODES` in `mesh/model.py`; the intro's int32 sentence points to `element_dofs`.
- `docs/elements.md`: the Registry bullet says the reference node tables come from `jaxmodal.mesh.model`.
- `docs/conventions.md` (Indices) and `docs/materials.md` (`phases`): the int32 rule above, pointing to `check_int32` in `docs/operators.md`.
- README: status line and a short snippet (box mesh → `assemble` → mask → `a_p`), executed by `tests/test_readme.py`.
- `docs/ROADMAP.md`: status line "PR 4 merged: DOF $3n + i$; BCs are `{node_set: "xyz"}` (parsed only by `dof_mask`; PR 9 takes the same dict); masks are bool, True = free; `nodal_mass` is `(n_nodes,)`, isotropic (×3 per DOF); `assemble` returns symmetrized SciPy CSR (oracle, x64 only); `box_mesh` (hex8/hex20/hex27) and `voxel_mesh` have face sets `xmin`…`zmax`; later PRs cast their own device index arrays through `check_int32`." PR 13's entry gains the sentence that `assemble` is the oracle only (production hex8 $A_P$ needs its own chunked assembler); PR 5 Contents gains "add $s_K$, $s_M$ to PR 4's `assemble`"; the package layout lists the box generator.

## Not covered

- Device operators (PR 5, PR 6), rigid modes and deflation (PR 7), the eigen-solves (PR 8).
- $s_K$, $s_M$ in the oracle (PR 5).
- Graded box meshes (PR 8).
- Skew or non-zero Dirichlet conditions (never; roadmap).
