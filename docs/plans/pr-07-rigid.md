# PR 7 plan: rigid body modes and deflation

Branch `pr-07-rigid`, from `main` after PR 6. Scope: the roadmap's PR 7 (Contents, Docs, Tests) and the "Rigid body modes" key design decision:

- rigid modes per connected part (face adjacency), with rotations about each part's centroid; edge- or vertex-only contacts (hinges) raise;
- automatic remaining modes under Dirichlet sets, $R\,\mathrm{null}((I - m)R)$;
- M-orthonormalization;
- the projector $P = I - R R^\top M$.

With this PR the solver interface of Principle 2 is complete for both discretizations: the operators' five methods plus `n_dof` (PRs 5, 6), and the rigid modes, which depend only on the mesh and the mask, so one function serves `unstructured` and `stencil`.

## Measured facts (prototype, scratchpad `p7/`, reviewer probes `r7-*/`; jax 0.11.2, fp64, M3 Max CPU)

- **Exactness:** for all cases below, $\|K_m R\|_\infty / (\|K_m\|_\infty \|R\|_\infty) \le 10^{-16}$ and $\|R^\top M_m R - I\|_\max \le 2\cdot10^{-15}$, with $K_m = DKD$, $M_m = DMD$ assembled by PR 4 and $R$ M-orthonormalized. The number of modes equals the dense count of eigenvalues of $K_{ff}$ below $10^{-10}\lambda_\max$; the smallest nonzero eigenvalue is $\ge 2.8\cdot10^{-5}\lambda_\max$ in every case, so the count is unambiguous.

  | mesh | BCs | modes |
  |---|---|---|
  | hex20 $3\times2\times2$ box, spacing (1, 0.7, 0.5) | free / `xmin: x` / + `ymin: y` / + `zmin: z` / `xmin: xyz` | 6 / 3 / 1 / 0 / 0 |
  | PR 5's mixed hex8 + hex20 mesh (perturbed nodes, 2 materials, rotated orthotropic), consistent and lumped M | free | 6 |
  | two disjoint voxel blocks | free / one block clamped | 12 / 6 |
  | hex8 box | one node pinned `xyz` / one edge clamped `xyz` / one node pinned `x` | 3 / 1 / 5 |
  | the hex20 box scaled by $10^6$ | `xmin: x` | 3 (unit-independent) |

- **Rank gap:** on these cases the singular values of the constrained rows $A$ (below) that count as zero are $\le 3.4\cdot10^{-17}\sigma_\max$, the others $\ge 0.30\,\sigma_\max$. Any rank tolerance in about $[10^{-16}, 0.1]$ gives the same counts here; the roadmap fixes $10^{-8}$.
- **Nearly degenerate supports** (two nodes pinned `xyz` on a line, a third pinned in z at distance $\varepsilon$ from it): $\sigma_\min/\sigma_\max \approx 1.2\varepsilon$. For $\varepsilon \gtrsim 10^{-8}$ the mode about the line is kept as constrained, although $K_{ff}$ is numerically singular for $\varepsilon \lesssim 10^{-6}$ (the solver then sees a ≈ 0 eigenvalue); for $\varepsilon < 10^{-8}$ it is deflated, with $\|K_m R\| \approx \varepsilon$ relative (masking removes entries of size $\varepsilon$). Pathological either way; documented, and the tests stay away from this window.
- **Orthonormalization:** one Cholesky pass leaves $\|V'^\top M V' - I\| \approx \varepsilon\,\kappa(G)$. For a slender part oblique to the axes, two rotation columns are nearly parallel and $\kappa(G) \sim (L/t)^2$: a hex8 beam $(L, 1, 1)$ rotated 45° about z gives $1.0\cdot10^{-14}$ / $1.7\cdot10^{-11}$ / $3.5\cdot10^{-10}$ for $L/t$ = 10 / 100 / 300 with one pass, and $2\cdot10^{-15}$ / $1.1\cdot10^{-14}$ / $4.7\cdot10^{-14}$ with two passes (CholQR2; $5.4\cdot10^{-14}$ at $L/t = 1000$).
- **Centroid:** rotating about the origin instead of the centroid spans the same space, but the columns get ill-conditioned with a large origin offset (after one Cholesky pass: $4\cdot10^{-15}$ at offset 0, $2\cdot10^{-8}$ at $10^3$ for a unit-size box). At an offset of $10^5$ SciPy's assembled $K$ itself loses digits (Jacobians $\sim \varepsilon\,\cdot$ offset/h) and its null space is off by $3\cdot10^{-12}$, so tests use offsets of about $10^3$.
- **Hinges:** two voxels sharing only an edge or only a vertex give two face-connected parts that share nodes, which the node-label check below detects; two voxels sharing a face give one part. Face contacts across hex8/hex20/hex27 (pairs, a chain, after `rcm` and after a random shuffle) give one part. A coarse element glued to $2\times2$ fine hex8 on a face (nonconforming) is also rejected; this is conservative (no mechanism) and the message names it.
- **Cost of the part search** (lexsort design below), full voxel boxes: 0.65 s and ≈ 0.41 kB of transient host memory per element at 1M elements, 2.3 s at 3.4M (linear); the peak comes from the face gather (128 B/el), `np.sort` and `lexsort` copies and csgraph's conversion. Two int64 keys and int32 orders are 30% faster but not leaner (507 B/el). At the Mac acceptance size (2M voxel DOF, 0.7M elements) this is ≈ 0.28 GB. A voxel-only alternative (`scipy.ndimage.label` on the cell mask) takes 0.04 s and 44 B per element (16× faster, 9× leaner); not used (one code path; see Not covered). `rigid_modes` without the part search at 10.3M DOF (`xmin: x`): 0.43 s, ≈ 0.25 GB.
- **`project`** fuses completely: 0 MB of temporaries besides the output (2M DOF, k = 6, m = 16).

## Files

```
src/jaxmodal/mesh/parts.py             parts(mesh): face-connected parts, hinge check (NumPy/SciPy only)
src/jaxmodal/mesh/__init__.py          + parts
src/jaxmodal/operators/constraints.py  + rigid_modes, m_orthonormalize, project
src/jaxmodal/operators/__init__.py     + rigid_modes, m_orthonormalize, project
tests/test_rigid.py
docs/boundary_conditions.md (new), docs/operators.md, docs/mesh.md, docs/ROADMAP.md, README.md
```

`constraints.py` is where the roadmap's package layout puts "masking, A_P, rigid modes, deflation". `parts` is host topology like `rcm`, so it lives in `mesh/` (PR 17 and PR 23 can reuse it). `jaxmodal.mesh` must not load JAX (`docs/mesh.md`, tested), so `parts` imports only `mesh.model`, NumPy and SciPy: no `elements`, no `check_int32`.

## API

```python
from jaxmodal.mesh import parts
from jaxmodal.operators import dof_mask, m_orthonormalize, project, rigid_modes

part = parts(mesh)                          # (n_nodes,) int64 part label per node, 0..n_parts-1
free = dof_mask(mesh, bcs)                  # PR 4
R0 = rigid_modes(mesh)                      # (n_dof, 6 n_parts): all modes (AMG near-null-space: free[:, None] * R0)
R = rigid_modes(mesh, free)                 # (n_dof, k) float64 NumPy, constrained rows exactly 0
R = jnp.asarray(R)
R = m_orthonormalize(R, op.apply_M(R))      # R^T M R = I (R is masked, so M's output mask is not needed)
Y = project(R, X, MX)                       # X - R (R^T MX), MX passed in (Principle 6)
```

Calling `rigid_modes` twice repeats the part search (seconds and GBs at 10M+ elements); a `part=` argument to reuse the labels is left to PR 13, the first caller that needs both.

### `parts(mesh) -> part`

- Face keys: for every block and each of the 6 faces, the 4 corner node ids `conn[:, FACE_CORNERS]` (`mesh.model.FACE_CORNERS`, shared by hex8, hex20 and hex27 because the corners come first in every type), sorted per face; int64 on the host. Faces of different types match on their corners, so mixed meshes connect across blocks.
- Elements with a repeated corner node (collapsed elements; `check_elements` does not necessarily catch these, e.g. a wedge-collapsed hex8 has $\det J > 0$ at its Gauss points) raise: a face collapsed to an edge would otherwise match another collapsed face and hide an edge contact.
- Elements sharing a face: `np.lexsort` the $(6 n_\text{el}, 4)$ keys; equal neighbouring rows give an element pair. `scipy.sparse.csgraph.connected_components` on these pairs labels the elements, in order of first appearance (deterministic).
- **Contact check:** scatter the element labels to their nodes (`part[conn] = label[e]`), then check that every element's nodes all carry its label. A node shared by two face-connected parts fails this, so the check catches edge and vertex contacts (hinges) and nonconforming contacts (nodes shared without a whole shared face). It raises `ValueError("mesh parts touch without sharing a whole face (edge, vertex or nonconforming contact): ...")` with the number of offending elements. A hinged part has mechanisms beyond the 6 rigid modes, so the roadmap rejects such meshes; nonconforming contacts are rejected conservatively.
- `Mesh` has no orphan nodes, so every node gets a part.

### `rigid_modes(mesh, free=None, max_parts=MAX_PARTS) -> (n_dof, k)` float64 NumPy

- `n_parts > max_parts` (default `MAX_PARTS = 10`) raises `ValueError` naming `n_parts` and `max_parts` and saying to remove floating islands (a legitimate multi-body model raises `max_parts`): $R$ is dense, $8\,n_\text{dof}\cdot 6 n_\text{parts}$ bytes (96 GB for 1000 islands at 2M DOF, as thresholded voxel masks can produce).
- Per part $p$ with nodes $x_n$ and centroid $c_p$ (the mean of its node coordinates, one `np.bincount` pass): three translations $e_i$ and three rotations $u_n = e_i \times (x_n - c_p)$, DOF $3n + i$ (PR 4). Columns $6p, \ldots, 6p + 5$ are $T_x, T_y, T_z, R_x, R_y, R_z$ of part $p$, filled by indexing (no per-part loop), and each full column is scaled to unit 2-norm. Rotating about the centroid keeps the columns well conditioned wherever the origin is. The node centroid is not the mass centroid; the span is the same and `m_orthonormalize` fixes the basis.
- `free=None`: all $6 n_\text{parts}$ columns (the AMG near-null-space before masking).
- `free` given (bool `(n_dof,)` from `dof_mask`, checked like PR 4's `_check_mask`): per part $p$ with constrained rows, $A_p$ = the constrained rows of the part's 6 unit-normalized columns (zero columns allowed, e.g. a rotation about a pinned node). The remaining modes of the part are $R_p N_p$, $N_p$ an orthonormal basis of $\mathrm{null}(A_p)$:
  - if $A_p$ has more than 6 rows, replace it by the $R$ factor of its QR (same singular values, $6 \times 6$);
  - `np.linalg.svd(A_p)` (`full_matrices=True`, so with fewer than 6 rows the rows of $V^\top$ beyond `len(s)` are included); rank $r$ = number of $\sigma_i > 10^{-8}\sigma_\max$ (`RANK_TOL = 1e-8`, unit-independent because the columns are unit-normalized); $N_p$ = the last $6 - r$ rows of $V^\top$, transposed; a part without constrained rows keeps $N_p = I_6$;
  - per part, because $A$ is block-diagonal by part: the decision for one part does not depend on another part's constraints (a global $\sigma_\max$ would couple them), and the cost is $O(36\,n_c)$;
  - the result is masked in place (`RN *= free[:, None]`, the full $R$ freed first) so constrained rows are exactly 0 (they are ≈ 1e-16 relative before, for exact null directions); it is then in the range of the masked eigen-operator.
  - The columns are not M-orthonormal (the mask and M are not known here); `m_orthonormalize` does that with whichever operator the caller uses.
- **Assumption:** every element has $s_K > 0$ (PRs 5 and 6 allow $s_K = 0$). Void elements can split a part or leave nodes without stiffness, so $\mathrm{null}(K_m)$ is then larger than $R$; `rigid_modes` does not see $s_K$, so designs use $s_\min > 0$ or remove void elements from the mesh (documented).
- Size: dense $(n_\text{dof}, k)$ fp64 on the host (4.8 GB at 100M DOF free-free). At 100M the device copy belongs in PR 9's fp32 deflation basis, which the roadmap's budget already counts ("≲ 8 modes in total at 100M, including any rigid modes R").

### `m_orthonormalize(V, MV) -> V'` (traceable, jnp)

CholQR2, carrying the M product along so M is applied only once (by the caller):

```
repeat 2 times:
    G = sym(V^T MV);  L = cholesky(G);  T = solve_triangular(L, I_k)^T   # T = L^{-T}
    V, MV = V @ T, MV @ T
```

Then $V'^\top M V' = I$ to round-off as long as the first Cholesky succeeds (CholQR2 needs $\kappa(V) \lesssim \varepsilon^{-1/2}$; measured: $4.7\cdot10^{-14}$ at $\kappa(G) = 4.5\cdot10^4$, where one pass gives $3.5\cdot10^{-10}$). `V @ T` avoids the transposed $(n_\text{dof}, k)$ copy of `solve_triangular(L, V.T).T`. $k = 0$ returns $(n_\text{dof}, 0)$. `MV` is the caller's M product, so the same function works with `assemble` (`Mm @ R`), `unstructured` and `stencil`; with a mask, $R$ is already masked, so $R^\top (m \odot MR) = R^\top MR$. $V$ must have full M-rank (true for $R$: $N_p$ is orthonormal, each part's 6 columns are independent and parts are disjoint); otherwise the result is NaN (docstring). PR 9's SVQB, not this function, handles rank-deficient blocks.

**Device memory at 100M:** device $R$, $MR$ and the result are $3 \cdot 8\, n_\text{dof} k$ (14.4 GB for $k = 6$) plus the operator workspace at $m = k$. This setup (device $R$, $MR$, orthonormalize, cast to fp32 into the basis) runs before PR 9 allocates the solver blocks, as does the host part search (≈ 0.28 GB at 2M voxel DOF); documented in `docs/boundary_conditions.md` for PR 9's setup order.

### `project(R, X, MX) -> X - R (R^\top MX)` (traceable, jnp)

- With $R^\top M R = I$: $R^\top M (P X) = 0$ and $P^2 = P$; $P R = 0$. `MX` is passed in, not recomputed (Principle 6; LOBPCG has it). With masked $R$ and $X$, the unmasked `apply_M(X)` gives the same result.
- fp64 and unchunked: $R$ has at most $6\,\text{MAX\_PARTS}$ columns. The chunked, optionally fp32 projector over the preallocated locked basis is PR 9's; there $R$ occupies the first columns of that basis, so this function is the reference it is tested against.
- $k = 0$ returns $X$.

## Tests (`tests/test_rigid.py`)

References: PR 4's `assemble`, `dof_mask` and `masked`, dense `numpy.linalg.eigh`. $\|\cdot\|$ is the $\infty$-norm (Principle 1).

1. **Null space, free:** hex8 box with origin $(10^3, -2\cdot10^3, 3\cdot10^3)$ on a unit-size box (an offset origin, small enough that SciPy's $K$ keeps its null space to $\le 10^{-13}$); curved hex27 box (PR 5's mesh B); the mixed hex8 + hex20 mesh; PR 1's voxelized sphere; a hex8 beam $300\times1\times1$ rotated 45° about z (needs CholQR2). Consistent and lumped M, with nodal masses (zeros included). $k = 6$; $\|K R\| \le 10^{-12}\|K\|\|R\|$; after `m_orthonormalize` with `Mm @ R`, $\|R^\top M R - I\| \le 10^{-12}$; $\mathrm{span}(R)$ equals the dense null space of $K$ (eigenvalues below $10^{-10}\lambda_\max$; beam excluded, too large for dense `eigh` at no gain): largest principal angle $\sin\theta \le 10^{-10}$ (≈ $\varepsilon\lambda_\max/\text{gap} \approx 4\cdot10^{-12}$).
2. **Under BCs:** the roadmap's free block: `xmin: x` leaves 3, adding `ymin: y` leaves 1 ($T_z$: the mode is $\pm$ the unit z translation after M-normalization), adding `zmin: z` leaves 0. Also one node pinned `xyz` (3), one edge clamped `xyz` (1, a rotation about that edge: $u = 0$ on the edge), one node pinned `x` (5). For each: the count equals the dense count of near-zero eigenvalues of $K_{ff}$; the singular values of each $A_p$ are either $\le 10^{-14}\sigma_\max$ or $\ge 10^{-3}\sigma_\max$ (the gap that makes `RANK_TOL` safe); $\|K_m R\| \le 10^{-12}\|K_m\|\|R\|$; constrained rows exactly 0; $\|R^\top M_m R - I\| \le 10^{-12}$; span equals the dense null space of $K_{ff}$ (padded by zeros on constrained DOFs).
3. **Parts:** two disjoint voxel blocks give 12 modes (two parts, each node labelled with its block); clamping one block (`xmin: xyz`) leaves 6, supported on the other block only; clamping a single node of the second block in addition gives the per-part result (3 there), unchanged by the first block's clamp. Two voxels joined only along an edge, and only at a vertex (a node shared without a shared face), raise the contact error from `parts` and `rigid_modes`; two voxels sharing a face give one part; a collapsed element raises. The mixed mesh is one part (faces match across hex8/hex20 blocks). Labels are deterministic: `parts` on an `rcm`-renumbered mesh and on a randomly renumbered mesh (`renumber` with random permutations) gives the same partition (mapped through the permutation), and `rigid_modes` the same span. `MAX_PARTS + 1` disjoint voxels raise with the default and pass with `max_parts=MAX_PARTS + 1`.
4. **Unit independence:** the hex8 box scaled by $10^{6}$ and $10^{-6}$ gives the same counts as unscaled for every BC case in test 2.
5. **Projector:** random $X$ $(n_\text{dof}, 5)$ on the masked mixed mesh with BCs leaving 3 modes and on the free sphere: $\|R^\top M_m\, \mathrm{project}(R, X, M_m X)\| \le 10^{-12}\|X\|$ (the roadmap's "projected random vectors are M-orthogonal to R"), $P^2 X = P X$ and $P R = 0$ to $10^{-12}$; `k = 0` returns $X$, also under `jit`.
6. **Device operators:** on the free voxel sphere, `m_orthonormalize` with `stencil(...).apply_M` and with `unstructured(...).apply_M` equals the SciPy result to $10^{-12}$.
7. **Inputs:** `rigid_modes` rejects a non-bool or wrong-length `free` (`ValueError`); `free` with every DOF constrained gives $(n_\text{dof}, 0)$ and `m_orthonormalize` of it $(n_\text{dof}, 0)$; `free=None` gives $6 n_\text{parts}$ unit-norm columns, and per part the translation columns are orthogonal to the rotation columns, $\|R_0[:, T]^\top R_0[:, R]\| \le 10^{-14}$ (exact because $\sum_n (x_n - c_p) = 0$; this catches rotating about the origin instead of the centroid, which CholQR2 would otherwise hide, on the offset box of test 1); `max_parts` below the part count raises, above it passes; importing `jaxmodal.mesh` still does not load jax (the existing test, now covering `parts`).

Equivalent mutants (expected to survive): the rotation sign (same span); `RANK_TOL` anywhere in about $[10^{-16}, 0.1]$ (the measured gap).

## Docs

- **`docs/boundary_conditions.md` (new, the roadmap's doc for PR 7):** Dirichlet BCs, masks and $A_P$ move here from `docs/operators.md` (with their Validation bullet), plus: parts, the face-contact rule and its message; rigid modes (formula, node centroid vs mass centroid, column order, unit normalization), `MAX_PARTS` and the dense size; the remaining modes under BCs ($R\,\mathrm{null}((I-m)R)$ per part, QR + SVD, `RANK_TOL`, masking) and the nearly degenerate support window; the $s_K > 0$ assumption; M-orthonormalization (CholQR2 and why); the projector and its properties; the AMG near-null-space $m \odot R_0$ vs the reduced deflation basis (which can be empty for clamped models); host cost of `parts`; the setup order and device peak for PR 9; validation.
- Pointers to the moved sections: `docs/operators.md` (one line, and its line 3), `README.md` ("See `docs/operators.md` for … BCs, $A_P$" → `docs/boundary_conditions.md`), and the module docstrings of `operators/constraints.py` and `operators/__init__.py`.
- `docs/mesh.md`: `parts`.
- `docs/operators.md`: the interface statement (five methods plus `n_dof`, plus `rigid_modes(mesh, free)`).
- `docs/ROADMAP.md`: status sentence for PR 7; Principle 2 notes that the rigid modes come from one mesh-level function for both discretizations; the key decision "Rigid body modes" says the SVD runs per part ($(I-m)R$ is block-diagonal by part) and mentions `max_parts` and the $s_K > 0$ assumption.
- `README.md`: status, and a short rigid-modes snippet after the operator snippets (run by `test_readme`).

## Not covered

- The chunked/fp32 deflation projector and inserting $R$ into the locked basis (PR 9).
- A `part=` argument to reuse the part labels across `rigid_modes` calls (PR 13).
- Meshes with many parts (floating islands): remove the islands first, or raise `max_parts` when R's dense size is affordable.
- The voxel-only part search (`ndimage.label`, measured above): the generic search costs ≈ 0.4 kB per element (≈ 13.5 GB transient host memory for a 100M-DOF voxel model with 33M elements, on top of its 2.1 GB connectivity and 0.8 GB points); revisit in PR 18 if host memory matters there.
- Skew constraints (the roadmap has none).
