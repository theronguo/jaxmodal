# Operators

`jaxmodal.operators` holds the global DOF map, the int32 overflow assert, the assembled SciPy reference operators (the oracle) and Dirichlet masking. The matrix-free device operators (PR 5, PR 6) will live here too.

```python
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import a_p, assemble, dof_mask, element_dofs, masked

mesh = box_mesh((4, 2, 2), spacing=(0.25, 0.25, 0.25), type="hex20")
K, M = assemble(mesh, materials)  # scipy.sparse.csr_array, float64, int32 indices
free = dof_mask(mesh, {"xmin": "xyz"})  # (n_dof,) bool, True = free
Km, Mm = masked(K, free), masked(M, free)  # eigen-operator form, constrained rows/cols zero
AP = a_p(K + sigma * M, free)  # SPD reference for preconditioners and linear solves
```

## DOF numbering

- **Global DOF** $3n + i$ is component $i$ of node $n$; `n_dof = 3 * len(mesh.points)`.
- **Element DOFs** follow PR 3's node-major order $3a + i$, so `element_dofs(mesh)[t]` is $3\,\mathrm{conn}[:, a] + i$, an `(n_el_type, 3n)` int32 array per block.
- **int32:** `check_int32(**sizes)` is the one overflow assert (Principle 6). It takes named plain ints and raises `ValueError` naming every size $\ge 2^{31}$. `element_dofs` checks `n_dof` and, per block, `<type>_dofs` $= n_\text{el} \cdot 3n$ (the flattened size PR 5 gathers and scatters), then casts. Every later PR that sends a host index array to the device (PR 5: the element → phase index; PR 6: the voxel maps) passes its sizes through `check_int32` and casts there.

## Assembled reference (`assemble`)

```python
K, M = assemble(mesh, materials, orientations=None, mass="consistent", nodal_mass=None, chunk=4096)
```

1. **Host checks:** x64 must be on (`jaxmodal.config.enable_x64()`), else `ValueError`, so the oracle is never silently fp32. Then `phases(mesh)`, `check_materials` and `check_elements(mesh, mass)`, and the `nodal_mass` checks below.
2. **Element matrices:** per block, in chunks of $c = \min(\text{chunk}, n_\text{el})$ elements, the jitted PR 3 kernels `stiffness` and `mass` (or `lumped_mass`) with the rotated per-phase $C$ and $\rho$ from `phase_table`. The last chunk is padded with its first element and sliced off, so a block compiles once per (type, $c$). `chunk` bounds the $K_e$ temporaries, not total memory.
3. **Global matrices:** COO triplets with int32 rows/cols from `element_dofs` → `csr_array` (duplicates summed) → $\tfrac12(A + A^\top)$. Each $K_e$ is exactly symmetric, but the summed CSR is symmetric only to ~1e-16 (summation order of duplicates), so the result is symmetrized and then satisfies `(K != K.T).nnz == 0`. K is built before M, so only one matrix's triplets are alive at a time.
4. **Lumped mass** (`mass="lumped"`): a diagonal M from the PR 3 lumped diagonals (row-sum for hex8/hex27, HRZ for hex20).
5. **Nodal (point) masses:** `nodal_mass` is `None` or a finite, non-negative `(n_nodes,)` array, isotropic: after symmetrization,
   $$M \leftarrow M + \mathrm{diag}(\mathrm{repeat}(m, 3)),$$
   for consistent and lumped M alike. Every later PR (5, 6, 9, 11, 14, 21) uses the same kwarg and shape.

**Size.** `assemble` is the oracle for PRs 5–8 and 12 at reference sizes, up to about $10^5$ DOF. That limit comes from PR 8's sparse factorization, not from assembly: on hex20 $20^3$ (107k DOF), `splu(K - σM)` took 241 s with a 16.5 GB peak. Assembly peaks at about 50 B per element-matrix entry (about 9, 15 and 13 KB per DOF for hex8, hex20 and hex27), so about 1.5 GB at $10^5$ DOF. `assemble` is the oracle only: PR 13's production hex8 $A_P$ for AMG needs its own chunked assembler into a precomputed CSR pattern (about 1 GB per matrix per 1M hex8 DOF; `assemble` would need about 90 GB at 10M DOF).

**Cost** (M3 Max, fp64, isotropic $C$, warm):

| mesh | DOF | nonzeros of K | K and M | peak RSS |
|---|---|---|---|---|
| hex8 $30^3$ | 89k | 6.8M | 0.55 s | 1.3 GB |
| hex20 $15^3$ | 47k | 7.4M | 0.47 s | 1.9 GB |
| hex27 $12^3$ | 47k | 8.2M | 0.44 s | 1.7 GB |

## Dirichlet conditions and masks

- **BCs** are zero displacement per component in global axes on node sets: `bcs = {node_set: "xyz"}`, e.g. `{"xmin": "xyz", "ymin": "y"}`. The value is a non-empty string of distinct letters from `"xyz"`. Sets may overlap (the union is constrained); `{}` is free-free. An unknown set name or invalid components raise `ValueError`. `dof_mask` is the only parser of this dict; PR 9's `solve_modes(mesh, materials, bcs, ...)` takes the same format.
- **Mask** $m$: `(n_dof,)` bool, `True` = free.
- **`masked(A, m)`** $= DAD$ with $D = \mathrm{diag}(m)$: the eigen-operator $v \mapsto m \odot A(m \odot v)$ as a matrix. Constrained rows and columns are exactly zero and the free block is exactly $A_{ff}$, so no spurious eigenvalues enter through the mask.

## $A_P$ (`a_p`)

Every preconditioner and linear solve (PR 10 on) uses
$$A_P = mAm + s\,(I - \mathrm{diag}\,m),$$
which puts the scalar $s$ on the constrained diagonal:

- **Default** $s$ = mean of $\mathrm{diag}(A)$ over free DOFs, for SPD $A = K + \sigma M$. Then $A_P$ is SPD exactly when $A_{ff}$ is (the constrained block is $sI$), its diagonal is positive (Jacobi divides by no zero), and it scales with $A$ (independent of the unit system). A default $s \le 0$ raises, so `a_p` never silently returns a non-SPD $A_P$.
- **Indefinite shift-invert** $K - \sigma M$ (PRs 10, 15): the roadmap (Key design decisions, interior bands) fixes $s = \overline{\mathrm{diag}\,K} + \sigma\,\overline{\mathrm{diag}\,M}$ over free DOFs, which the caller passes as `a_p(K - sigma * M, m, s=...)`.
- A mask with no free DOF raises.

## Validation

`tests/test_operators.py`:

- **Assembly** equals a dense Python loop (explicit DOF lists, per-element `rotate(material, orientation)`, no `phases`/`element_dofs`) to 1e-14 on a distorted two-block mesh: hex20 and hex8 glued at a shared face, with two materials (one orthotropic, different $\rho$) and two orientations mixed in each block, for `chunk=7` and `chunk=4096`, consistent and lumped.
- **Patch test:** on a $3^3$ box per type, rotated orthotropic $C$, interior corner nodes moved per component by $\le 15\%$ of the edge, all boundary nodes prescribed $u = Ax + b$, the interior solved with `spsolve`. The stress is constant at every quadrature point to 1e-10 relative to $\|\sigma\|_\infty$; measured 1.2e-15 (hex8), 1.0e-14 (hex20), 1.3e-14 (hex27) for straight edges (non-corner nodes on the trilinear map of the corners), and 6.4e-15 (hex20), 1.6e-14 (hex27) for curved elements (non-corner interior nodes additionally offset by $\le 5\%$ per component).
- **Rigid translation mass:** $t_x^\top M t_x = \rho V + \sum m_n$ with the analytic volume, consistent and lumped, straight and curved; $t_x^\top M t_y = 0$.
- **Masks and $A_P$:** expected constrained DOFs from node coordinates; exact free blocks; $A_P$ SPD (Cholesky), constrained diagonal $= s$, exact scaling by $2^{20}$, all-free mask gives $A$.
- **Nodal masses** add exactly; `element_dofs` calls `check_int32` with every block's size.
