# Boundary conditions, rigid modes and deflation

`jaxmodal.operators` (module `operators/constraints.py`) holds the Dirichlet masks, the masked eigen-operator and $A_P$ (SciPy, PR 4), and the rigid body modes, their M-orthonormalization and the deflation projector (PR 7). `jaxmodal.mesh.parts` finds the face-connected parts.

```python
import jax.numpy as jnp
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import a_p, assemble, dof_mask, m_orthonormalize, masked, project
from jaxmodal.operators import rigid_modes, unstructured

mesh = box_mesh((4, 2, 2), spacing=(0.25, 0.25, 0.25), type="hex20")
K, M = assemble(mesh, materials)  # materials: label -> Material (docs/materials.md)
free = dof_mask(mesh, {"xmin": "x"})  # (n_dof,) bool, True = free
Km, Mm = masked(K, free), masked(M, free)  # eigen-operator form, constrained rows/cols zero
AP = a_p(K + sigma * M, free)  # SPD reference for preconditioners and linear solves

R0 = rigid_modes(mesh)  # (n_dof, 6 n_parts); AMG near-null-space: free[:, None] * R0
R = jnp.asarray(rigid_modes(mesh, free))  # remaining modes: here T_y, T_z, R_x (k = 3)
op = unstructured(mesh, materials)
R = m_orthonormalize(R, op.apply_M(R))  # R^T M R = I; consumes its first argument
Y = project(R, X, MX)  # X - R (R^T MX), with MX = M X
```

## Dirichlet conditions and masks

- **BCs** are zero displacement per component in global axes on node sets: `bcs = {node_set: "xyz"}`, e.g. `{"xmin": "xyz", "ymin": "y"}`. The value is a non-empty string of distinct letters from `"xyz"`. Sets may overlap (the union is constrained); `{}` is free-free. An unknown set name or invalid components raise `ValueError`. `dof_mask` is the only parser of this dict; PR 9's `solve_modes(mesh, materials, bcs, ...)` takes the same format.
- **Mask** $m$: `(n_dof,)` bool, `True` = free. `masked` and `a_p` raise `ValueError` unless it is a bool `np.ndarray` of length `n_dof` (a 0/1 int mask would silently index and complement wrongly).
- **`masked(A, m)`** $= DAD$ with $D = \mathrm{diag}(m)$: the eigen-operator $v \mapsto m \odot A(m \odot v)$ as a matrix. Constrained rows and columns are exactly zero and the free block is exactly $A_{ff}$, so no spurious eigenvalues enter through the mask.

## $A_P$ (`a_p`)

Every preconditioner and linear solve (PR 10 on) uses
$$A_P = mAm + s\,(I - \mathrm{diag}\,m),$$
which puts the scalar $s$ on the constrained diagonal:

- **Default** $s$ = mean of $\mathrm{diag}(A)$ over free DOFs, for SPD $A = K + \sigma M$. Then $A_P$ is SPD exactly when $A_{ff}$ is (the constrained block is $sI$), its diagonal is positive (Jacobi divides by no zero), and it scales with $A$ (independent of the unit system). A default $s \le 0$ raises, so `a_p` never silently returns a non-SPD $A_P$.
- **Indefinite shift-invert** $K - \sigma M$ (PRs 10, 15): the roadmap (Key design decisions, interior bands) fixes $s = \overline{\mathrm{diag}\,K} + \sigma\,\overline{\mathrm{diag}\,M}$ over free DOFs, which the caller passes as `a_p(K - sigma * M, m, s=...)`.
- A mask with no free DOF raises.

## Parts (`parts`)

`parts(mesh)` returns an `(n_nodes,)` int64 part label per node, $0, \ldots, n_\text{parts} - 1$. Two elements are connected when they share a whole face: the 4 corner node ids of each face (`mesh.model.FACE_CORNERS`, shared by hex8, hex20 and hex27, so mixed meshes connect across blocks), sorted per face, are `np.lexsort`ed, equal neighbouring keys give an element pair, and `scipy.sparse.csgraph.connected_components` labels the elements. NumPy and SciPy only, so `jaxmodal.mesh` stays free of JAX.

- **Contact rule.** The element labels are scattered to their nodes and every element's nodes must carry its label. A node shared by two face-connected parts fails this, so edge and vertex contacts (hinges, which add mechanisms beyond the 6 rigid modes per part) and nonconforming contacts (nodes shared without a whole shared face; rejected conservatively) raise `ValueError("mesh parts touch without sharing a whole face ...")`. Parts that touch without sharing nodes are separate parts.
- Elements with a repeated corner node (collapsed elements) raise: a face collapsed to an edge would otherwise match another one and hide an edge contact. `check_elements` does not necessarily catch these (a wedge-collapsed hex8 has $\det J > 0$ at its Gauss points).
- **Cost:** ≈ 0.65 s and ≈ 0.4 kB of transient host memory per 1M elements (linear; ≈ 0.28 GB at 2M voxel DOF, ≈ 13.5 GB for a 100M-DOF voxel model). It runs before the solver allocates its device arrays. A voxel-only search (`scipy.ndimage.label` on the cell mask) would be 16× faster and 9× leaner; it is left to PR 18.

## Rigid body modes (`rigid_modes`)

`rigid_modes(mesh, free=None, max_parts=MAX_PARTS)` returns a dense `(n_dof, k)` float64 NumPy array with unit-2-norm columns.

- **Free** (`free=None`): per part $p$ with nodes $x_n$ and node centroid $c_p = \frac{1}{|p|}\sum_{n \in p} x_n$, the translations $e_i$ and rotations $u_n = e_i \times (x_n - c_p)$, DOF $3n + i$. Columns $6p, \ldots, 6p + 5$ are $T_x, T_y, T_z, R_x, R_y, R_z$ of part $p$, so $k = 6 n_\text{parts}$. Rotating about the centroid keeps the columns well conditioned wherever the origin is (translations and rotations are orthogonal, since $\sum_n (x_n - c_p) = 0$). The node centroid is not the mass centroid; the span is the same, and `m_orthonormalize` fixes the basis.
- **Under Dirichlet sets** (`free` from `dof_mask`): the remaining modes are $R\,\mathrm{null}((I - m)R)$. $(I - m)R$ is block-diagonal by part, so the null space is computed per part: with $A_p$ the constrained rows of the part's 6 unit-normalized columns (zero columns allowed, e.g. a rotation about a pinned node), $A_p$ is reduced to the $6 \times 6$ $R$ factor of its QR when it has more rows, then $\sigma_i \le 10^{-8}\sigma_\max$ (`RANK_TOL`, unit-independent because the columns are unit-normalized) counts as zero and $N_p$ is the matching right singular vectors (`full_matrices=True`). A per-part $\sigma_\max$ keeps one part's constraints from moving another part's threshold. The result $R\,\mathrm{blkdiag}(N_p)$ is multiplied by the mask, so constrained rows are exactly 0 (they are ≈ 1e-16 relative before for exact null directions), and lies in the range of the masked eigen-operator.
- **Nearly degenerate supports** (e.g. a third support at distance $\varepsilon$ from the line through two pinned nodes): $\sigma_\min/\sigma_\max \approx 1.2\,\varepsilon$. For $\varepsilon \gtrsim 10^{-8}$ the mode counts as constrained although $K_{ff}$ is numerically singular for $\varepsilon \lesssim 10^{-6}$ (the solver then sees an eigenvalue ≈ 0); below, it is deflated with $\|K_m R\| \approx \varepsilon$ relative. Avoid such supports.
- **Assumption:** every element has $s_K > 0$. Void elements ($s_K = 0$, allowed by PRs 5 and 6) can split a part or leave nodes without stiffness, so $\mathrm{null}(K_m)$ would be larger than $R$; `rigid_modes` does not see $s_K$, so use $s_\min > 0$ or remove void elements from the mesh.
- **Size:** $8\,n_\text{dof} \cdot 6 n_\text{parts}$ bytes on the host (4.8 GB at 100M DOF for one free part). More than `max_parts` parts (default `MAX_PARTS = 10`) raise, naming `n_parts`: thresholded voxel masks can leave floating islands (1000 islands at 2M DOF would be 96 GB); remove them, or raise `max_parts` for a legitimate multi-body model.
- **AMG near-null-space** (PR 13): $m \odot R_0$ with $R_0$ = `rigid_modes(mesh)`, all 6 modes per part with constrained rows zeroed. It is not the reduced deflation basis, which can be empty for clamped models.

## M-orthonormalization (`m_orthonormalize`)

`m_orthonormalize(V, MV)` takes the caller's M product, so it works with `assemble`, `unstructured` and `stencil`. With a mask, $R$ is already masked, so $R^\top (m \odot MR) = R^\top MR$ and the unmasked `apply_M(R)` suffices. It runs Cholesky QR twice (CholQR2), carrying $MV$ along so M is applied only once:
$$T(V, MV) = L^{-\top},\quad LL^\top = \tfrac12(V^\top MV + MV^\top V);\qquad V_1 = V\,T(V, MV),\ MV_1 = MV\,T(V, MV);\qquad V' = V_1\,T(V_1, MV_1).$$

- **Why two passes:** one pass leaves $\|V'^\top MV' - I\| \approx \varepsilon\,\kappa(G)$. For a slender part oblique to the axes two rotation columns are nearly parallel and $\kappa(G) \sim (L/t)^2$: a hex8 beam $300\times1\times1$ rotated 45° gives $3.5\cdot10^{-10}$ with one pass and $4.7\cdot10^{-14}$ with two.
- $V$ must have full M-rank (true for `rigid_modes`); otherwise the result is NaN. Rank-deficient blocks are PR 9's SVQB.
- **Memory:** jitted with $V$ donated (consumed; $V$ and $MV$ must be distinct arrays; $k = 0$ returns before the jitted call). The peak is $V$, $MV$ and the pass-1 temporary, the result aliasing $V$: $3 \cdot 8\,n_\text{dof} k$ bytes (14.4 GB at 100M DOF with $k = 6$, 4 blocks without donation), plus the operator workspace during `apply_M`. This setup (device $R$, $MR$, orthonormalize, cast into the deflation basis) runs before PR 9 allocates the solver blocks.

## Deflation projector (`project`)

$$P X = X - R\,(R^\top M X),\qquad R^\top M R = I,$$
so $R^\top M (PX) = 0$, $P^2 = P$ and $PR = 0$. `project(R, X, MX)` takes $MX$ (Principle 6: LOBPCG has it, so it is not recomputed); with masked $R$ and $X$ the unmasked $MX$ gives the same result. It is fp64 and unchunked ($R$ has at most $6 n_\text{parts}$ columns) and fuses to no temporaries besides the output. PR 9's chunked, optionally fp32 projector over the locked basis holds $R$ in its first columns; this function is its reference. $k = 0$ returns $X$.

## Validation

`tests/test_operators.py` (PR 4):

- **Masks and $A_P$:** expected constrained DOFs from node coordinates; exact free blocks; $A_P$ SPD (Cholesky), constrained diagonal $= s$, exact scaling by $2^{20}$, all-free mask gives $A$.

`tests/test_rigid.py` (PR 7), against PR 4's `assemble`/`masked` and dense `eigh`; norms are $\infty$-norms:

- **Free:** a unit-size hex8 box at origin $(10^3, -2\cdot10^3, 3\cdot10^3)$, a curved hex27 box, the mixed hex8 + hex20 mesh, PR 1's voxelized sphere and a hex8 beam $300\times1\times1$ rotated 45°; consistent and lumped M with nodal masses: $k = 6$, $\|KR\| \le 10^{-12}\|K\|\|R\|$, $\|R^\top MR - I\| \le 10^{-12}$ after `m_orthonormalize`, and (except the beam) the span equals the dense null space of $K$ (eigenvalues below $10^{-10}\lambda_\max$) to $\sin\theta \le 10^{-10}$.
- **Under BCs** (hex8 box): `xmin: x` → 3, + `ymin: y` → 1 ($T_z$), + `zmin: z` → 0; one node pinned `xyz` → 3, one edge clamped → 1, one node pinned `x` → 5, `xmin: xyz` → 0. Each count equals the dense count for $K_{ff}$; the singular values of $A$ are $\le 10^{-14}\sigma_\max$ or $\ge 10^{-3}\sigma_\max$ (the gap `RANK_TOL` relies on; measured $3.4\cdot10^{-17}$ vs $\ge 0.30$); $\|K_mR\|$, $\|R^\top M_mR - I\|$ and the span as above; constrained rows exactly 0; the counts are unchanged with the box scaled by $10^{\pm 6}$.
- **Parts:** two disjoint voxel blocks → 12 modes; one clamped → 6 on the other block only; a pinned node on the second block gives 3 there, with or without the first block's clamp; edge and vertex contacts raise; a face contact and the mixed mesh are one part; a collapsed element raises; labels and spans follow `rcm` and a random `renumber`; `max_parts` raises and passes.
- **Projector:** random blocks on the masked mixed mesh (3 modes) and the free sphere: $\|R^\top M_m PX\| \le 10^{-12}\|X\|$, $P^2X = PX$, $PR = 0$; $k = 0$ returns $X$.
- **Device operators:** `m_orthonormalize` with `stencil` and `unstructured` `apply_M` equals the SciPy result to $10^{-12}$; the compiled CholQR2 aliases its output to the donated $V$ with $\le 1.05$ blocks of temporaries and no donation warning.
- **Inputs:** bad masks raise; an all-constrained mask gives $(n_\text{dof}, 0)$; free columns have unit norm and translations are orthogonal to rotations ($\le 10^{-10}$; rotating about the origin would give ≈ 0.9).
