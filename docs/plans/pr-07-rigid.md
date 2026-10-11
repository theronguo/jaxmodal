# PR 7 plan: rigid body modes and deflation

Branch `pr-07-rigid`, from `main` after PR 6. Scope: the roadmap's PR 7 (Contents, Docs, Tests) and the "Rigid body modes" key design decision:

- rigid modes per connected part (face adjacency), with rotations about each part's centroid; edge- or vertex-only contacts (hinges) raise;
- automatic remaining modes under Dirichlet sets, $R\,\mathrm{null}((I - m)R)$;
- M-orthonormalization;
- the projector $P = I - R R^\top M$.

With this PR the solver interface of Principle 2 is complete for both discretizations: the operators' five methods plus `n_dof` (PRs 5, 6), and the rigid modes, which depend only on the mesh and the mask, so one function serves `unstructured` and `stencil`.

## Measured facts (prototype, scratchpad `p7/`; jax 0.11.2, fp64, M3 Max CPU)

- **Exactness:** for all cases below, $\|K_m R\|_\infty / (\|K_m\|_\infty \|R\|_\infty) \le 10^{-16}$ and $\|R^\top M_m R - I\|_\max \le 2\cdot10^{-15}$, with $K_m = DKD$, $M_m = DMD$ assembled by PR 4 and $R$ M-orthonormalized. The number of modes equals the dense count of eigenvalues of $K_{ff}$ below $10^{-10}\lambda_\max$; the smallest nonzero eigenvalue is $\ge 2.8\cdot10^{-5}\lambda_\max$ in every case, so the count is unambiguous.

  | mesh | BCs | modes |
  |---|---|---|
  | hex20 $3\times2\times2$ box, spacing (1, 0.7, 0.5) | free / `xmin: x` / + `ymin: y` / + `zmin: z` / `xmin: xyz` | 6 / 3 / 1 / 0 / 0 |
  | PR 5's mixed hex8 + hex20 mesh (perturbed nodes, 2 materials, rotated orthotropic), consistent and lumped M | free | 6 |
  | two disjoint voxel blocks | free / one block clamped | 12 / 6 |
  | hex8 box | one node pinned `xyz` / one edge clamped `xyz` / one node pinned `x` | 3 / 1 / 5 |
  | the hex20 box scaled by $10^6$ | `xmin: x` | 3 (unit-independent) |

- **Hinges:** two voxels sharing only an edge or only a vertex give two face-connected parts that share nodes, which the node-label check below detects; two voxels sharing a face give one part.
- **Cost of the part search** (NumPy face sort, below), full voxel boxes: 0.66 s and ≈ 0.41 kB of transient host memory per element at 1M elements, 2.4 s at 3.4M (linear). The full `rigid_modes` with a mask on $60^3$ (680k DOF) takes 0.6 s. A voxel-only alternative (`scipy.ndimage.label` on the cell mask, 6-connectivity for parts and 26-connectivity for the hinge check) takes 0.04 s and 44 B per element; it is not used (one code path; see Not covered).

## Files

```
src/jaxmodal/mesh/parts.py             parts(mesh): face-connected parts, hinge check
src/jaxmodal/mesh/__init__.py          + parts
src/jaxmodal/operators/constraints.py  + rigid_modes, m_orthonormalize, project
src/jaxmodal/operators/__init__.py     + rigid_modes, m_orthonormalize, project
tests/test_rigid.py
docs/boundary_conditions.md (new), docs/operators.md, docs/mesh.md, docs/ROADMAP.md, README.md
```

`constraints.py` is where the roadmap's package layout puts "masking, A_P, rigid modes, deflation". `parts` is host topology like `rcm`, so it lives in `mesh/` (PR 17 and PR 23 can reuse it).

## API

```python
from jaxmodal.mesh import parts
from jaxmodal.operators import dof_mask, m_orthonormalize, project, rigid_modes

n_parts, part = parts(mesh)                 # int, (n_nodes,) int64 part per node
free = dof_mask(mesh, bcs)                  # PR 4
R = rigid_modes(mesh, free)                 # (n_dof, k) float64 NumPy, constrained rows exactly 0
R = jnp.asarray(R)
R = m_orthonormalize(R, free[:, None] * op.apply_M(R))   # masked M: R^T M R = I
Y = project(R, X, MX)                       # X - R (R^T MX), MX passed in (Principle 6)
near_null = free[:, None] * rigid_modes(mesh)            # PR 13's AMG near-null-space: all 6 per part
```

### `parts(mesh) -> (n_parts, part)`

- Face keys: for every element and each of its 6 faces, the 4 corner node ids (`ELEMENTS[t].faces[f][:4]`, the corners come first for every type), cast to int32 before the gather (`check_int32` on `n_nodes`) and sorted. Faces of hex8, hex20 and hex27 elements match on their corners, so mixed meshes connect across blocks.
- Elements sharing a face: `np.lexsort` the $(6 n_\text{el}, 4)$ keys; equal neighbouring rows give an element pair. `scipy.sparse.csgraph.connected_components` on these pairs labels the elements, in order of first appearance (deterministic).
- **Hinge check:** scatter the element labels to their nodes (`part[conn] = label[e]`), then check that every element's nodes all carry its label. A node shared by two face-connected parts fails this, so the check catches edge, vertex and any other non-face contacts, including nonconforming ones (a node shared without a shared face). It raises `ValueError("mesh parts touch only along edges or at vertices (hinge): ...")` with the number of offending elements. Rigid modes of a hinged part are not $6$ per part (a hinge adds mechanisms), so the roadmap rejects such meshes.
- Mesh has no orphan nodes (`Mesh` checks this), so every node gets a part.

### `rigid_modes(mesh, free=None) -> (n_dof, k)` float64 NumPy

- Per part $p$ with nodes $x_n$ and centroid $c_p$ (the mean of its node coordinates): three translations $e_i$ and three rotations $u_n = e_i \times (x_n - c_p)$, DOF $3n + i$ (PR 4). Columns $6p, \ldots, 6p + 5$ are $T_x, T_y, T_z, R_x, R_y, R_z$ of part $p$, each scaled to unit 2-norm. Rotating about the centroid makes the columns well conditioned regardless of where the origin is (an offset origin, as in PR 6's F mesh, does not matter).
- `free=None`: all $6 n_\text{parts}$ columns (the AMG near-null-space before masking).
- `free` given (bool `(n_dof,)` from `dof_mask`, checked like PR 4's `_check_mask`): with $A = R[\lnot m]$ (the constrained rows, unit-normalized columns), the remaining modes are $R N$ with $N$ an orthonormal basis of $\mathrm{null}(A)$:
  - if $A$ has more rows than columns, replace it by the $R$ factor of its QR (same singular values, $6P \times 6P$), so the SVD never sees an $n_c \times n_c$ matrix;
  - SVD; rank $r$ = number of $\sigma_i > 10^{-8}\sigma_\max$ (unit-independent because the columns are unit-normalized; `RANK_TOL = 1e-8`); $N$ = the last $6P - r$ right singular vectors; no constrained rows (or $\sigma_\max = 0$) gives $N = I$;
  - the result is multiplied by `free` so constrained rows are exactly 0 (they are $\le 10^{-8}$ relative before); it is then in the range of the masked eigen-operator, with $\|K_m R\|$ at round-off.
  - The columns are not M-orthonormal (the mask and M are not known here); `m_orthonormalize` does that with whichever operator the caller uses.
- Size: dense $(n_\text{dof}, 6 n_\text{parts})$ fp64 on the host (4.8 GB at 100M DOF free-free). At 100M the device copy belongs in PR 9's fp32 deflation basis, which the roadmap's budget already counts ("≲ 8 modes in total at 100M, including any rigid modes R").

### `m_orthonormalize(V, MV) -> V'` (traceable, jnp)

$G = \tfrac12(V^\top MV + MV^\top V)$, $G = LL^\top$ (Cholesky), $V' = V L^{-\top}$ (`solve_triangular`). Then $V'^\top M V' = I$ to round-off. $k = 0$ returns $(n_\text{dof}, 0)$. `MV` is the caller's M product, so the same function works with `assemble` (`Mm @ R`), `unstructured` and `stencil`; with a mask the caller passes the masked product `free[:, None] * op.apply_M(R)` (R is already masked). It raises nothing; a non-SPD $G$ gives NaN, which the tests exclude by construction (R has full column rank: $N$ is orthonormal and $R$'s columns are independent within each part and disjoint across parts).

### `project(R, X, MX) -> X - R (R^\top MX)` (traceable, jnp)

- With $R^\top M R = I$: $R^\top M (P X) = 0$ and $P^2 = P$; $P R = 0$. `MX` is passed in, not recomputed (Principle 6; LOBPCG has it).
- fp64 and unchunked: $R$ has $6 n_\text{parts}$ columns. The chunked, optionally fp32 projector over the preallocated locked basis is PR 9's; there $R$ simply occupies the first columns of that basis, so this function is the reference it is tested against.
- $k = 0$ returns $X$.

## Tests (`tests/test_rigid.py`)

References: PR 4's `assemble`, `dof_mask` and `masked`, dense `numpy.linalg.eigh`. $\|\cdot\|$ is the $\infty$-norm (Principle 1).

1. **Null space, free:** hex8 box with an offset origin, curved hex27 box (PR 5's mesh B), the mixed hex8 + hex20 mesh, PR 1's voxelized sphere; consistent and lumped M, with nodal masses (zeros included). $k = 6$; $\|K R\| \le 10^{-12}\|K\|\|R\|$; after `m_orthonormalize` with `Mm @ R`, $\|R^\top M R - I\| \le 10^{-12}$; $\mathrm{span}(R)$ equals the dense null space of $K$ (eigenvalues below $10^{-10}\lambda_\max$): largest principal angle $\sin\theta \le 10^{-10}$.
2. **Under BCs:** the roadmap's free block: `xmin: x` leaves 3, adding `ymin: y` leaves 1 ($T_z$: the mode is $\pm$ the unit z translation after M-normalization), adding `zmin: z` leaves 0. Also one node pinned `xyz` (3), one edge clamped `xyz` (1, a rotation about that edge: $u = 0$ on the edge), one node pinned `x` (5). For each: the count equals the dense count of near-zero eigenvalues of $K_{ff}$; $\|K_m R\| \le 10^{-12}\|K_m\|\|R\|$; constrained rows exactly 0; $\|R^\top M_m R - I\| \le 10^{-12}$; span equals the dense null space of $K_{ff}$ (padded by zeros on constrained DOFs).
3. **Parts:** two disjoint voxel blocks give 12 modes ($n_\text{parts} = 2$, each node labelled with its block); clamping one block (`xmin: xyz`) leaves 6, supported on the other block only. Two voxels joined only along an edge, and only at a vertex, raise the hinge error (from `parts` and `rigid_modes`); two voxels sharing a face give one part. The mixed mesh is one part (faces match across hex8/hex20 blocks). Labels are deterministic: `parts` on an `rcm`-renumbered mesh gives the same partition (mapped through the permutation), and `rigid_modes` the same span.
4. **Unit independence:** the hex8 box scaled by $10^{6}$ and $10^{-6}$ gives the same counts as unscaled for every BC case in test 2.
5. **Projector:** random $X$ $(n_\text{dof}, 5)$ on the masked mixed mesh with BCs leaving 3 modes and on the free sphere: $\|R^\top M_m\, \mathrm{project}(R, X, M_m X)\| \le 10^{-12}\|X\|$ (the roadmap's "projected random vectors are M-orthogonal to R"), $P^2 X = P X$ and $P R = 0$ to $10^{-12}$; `k = 0` returns $X$.
6. **Device operators:** on the free voxel sphere, `m_orthonormalize` with `stencil(...).apply_M` and with `unstructured(...).apply_M` equals the SciPy result to $10^{-12}$; `project` jitted with $R$ as an argument does not recompile for new values of $R$ (same shape).
7. **Inputs:** `rigid_modes` rejects a non-bool or wrong-length `free` (`ValueError`); `free` with every DOF constrained gives $(n_\text{dof}, 0)$ and `m_orthonormalize` of it $(n_\text{dof}, 0)$; `free=None` gives $6 n_\text{parts}$ unit-norm columns; `check_int32` is called with `n_nodes`.

## Docs

- **`docs/boundary_conditions.md` (new, the roadmap's doc for PR 7):** Dirichlet BCs, masks and $A_P$ move here from `docs/operators.md` (which keeps a one-line pointer), plus: parts and the hinge rule; rigid modes (formula, centroid, column order, unit normalization); the remaining modes under BCs ($R\,\mathrm{null}((I-m)R)$, QR + SVD, `RANK_TOL`, masking); M-orthonormalization; the projector and its properties; the AMG near-null-space $m \odot R$ vs the reduced deflation basis (which can be empty for clamped models); host cost of `parts`; validation.
- `docs/mesh.md`: `parts`.
- `docs/operators.md`: line 3 and the pointer to `boundary_conditions.md`; the interface statement (five methods plus `n_dof`, plus `rigid_modes(mesh, free)`).
- `docs/ROADMAP.md`: status sentence for PR 7; Principle 2 notes that the rigid modes come from one mesh-level function for both discretizations.
- `README.md`: status, and a short rigid-modes snippet after the operator snippets (run by `test_readme`).

## Not covered

- The chunked/fp32 deflation projector and inserting $R$ into the locked basis (PR 9).
- The voxel-only part search (`ndimage.label`, 10× faster and 9× less host memory, measured above): the generic search costs ≈ 0.4 kB per element (≈ 14 GB transient host memory for a 100M-DOF voxel model, 33M elements); revisit in PR 18 if host memory matters there.
- Skew constraints (the roadmap has none).
