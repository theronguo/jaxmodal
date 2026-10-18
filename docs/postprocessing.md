# Postprocessing

`jaxmodal.post` (PR 17) works on the outputs of `solve_modes` and `reference_modes`: mode
normalization, the M-weighted MAC, participation factors and effective modal mass, clusters and a
frequency table, strain and stress recovery, and VTU export. Everything here is single-host NumPy
around jitted kernels; outputs are NumPy float64. Every function except `influence`, `clusters`,
`frequency_table`, `format_table`, `von_mises` and `write_modes` needs x64
(`jaxmodal.config.enable_x64()`).

## Conventions

- **Modes** `V` are host or device arrays `(n_dof, n)` in the full DOF space, node-major
  ($3a + i$), with constrained rows 0, as `solve_modes`, `reference_modes` and `info["rigid"]`
  return them. A 1-D array is one column; n = 0 columns are allowed everywhere.
- **Mass** `M` is an operator with `apply_M` (`unstructured`, `stencil`) or a SciPy/NumPy matrix
  (`assemble`'s M). Build the operator with the same `mass`, `nodal_mass` and `orientations` as
  the solve.
- **Batched products.** Every product with `M` runs on column batches of width
  $w = \min(\text{batch}, n)$ (`batch=64`, an argument of `normalize`, `mac` and `participation`).
  Only the last batch is zero-padded, so there is one compile per width. Each batch is cast to
  fp64, and the next batch's `apply_M` is dispatched before the host GEMM of the current one.

## Normalization

`normalize(V, M=None, kind="mass", batch=64)` returns a scaled copy:

- `kind="mass"` (needs `M`): $\phi_i \leftarrow \phi_i/\sqrt{\phi_i^\top M\phi_i}$. Columns are
  not re-orthogonalized.
- `kind="max"`: the largest nodal displacement norm $\max_a\|u_a\|_2$ is 1.
- **Sign** (both kinds): among the DOFs with $|v_j| \ge (1 - 10^{-6})\max_j|v_j|$, the one with
  the lowest index is made positive. That is deterministic under mirror ties. Ties from iterative
  solves can fall outside the $10^{-6}$ band (vector accuracy is about tol/gap), and inside a
  degenerate cluster the sign and the basis are arbitrary anyway.
- A zero (or non-positive M-norm) column raises, as does `kind="mass"` without `M`.

`solve_modes` and `reference_modes` already return M-orthonormal modes, so large models skip
`normalize("mass")`: the copy doubles host `V`.

## MAC

$$\mathrm{MAC}_{ij} = \frac{(a_i^\top M b_j)^2}{(a_i^\top M a_i)(b_j^\top M b_j)}$$

`mac(A, B, M=None, batch=64)` returns `(nA, nB)`; `M=None` is the unweighted MAC. It costs one
batched pass over `B` and one over `A` (a single pass when `B is A`). The test oracle
`tests/exact.mac` stays independent.

## Influence vectors, participation and effective mass

`influence(mesh, free, origin=None)` returns the 6 rigid-body fields `r (n_dof, 6)`:
translations $e_x, e_y, e_z$ and rotations $e_k \times (x - x_0)$, with the sign convention of
`rigid_modes`. Rows of constrained DOFs are 0. `free` is the `dof_mask` (`None` = no
constraints); it is required, because a forgotten mask on a supported body silently breaks the
sum rule. The default origin $x_0$ is the unweighted node centroid, as in `rigid_modes`. It is
neither the centre of mass nor a support point, and the rotational effective masses depend on
it.

`participation(V, M, r, batch=64)` returns `gamma (n, 6)`, `m_eff (n, 6)` and `total (6,)`:

$$b = V^\top(Mr), \quad \mu_i = \phi_i^\top M\phi_i, \quad \Gamma_{ik} = \frac{b_{ik}}{\mu_i},
\quad m^\text{eff}_{ik} = \frac{b_{ik}^2}{\mu_i}, \quad \text{total}_k = r_k^\top M r_k.$$

The modes need no normalization. `total` does not depend on `V`.

- **Masked $r$.** Modes and $r$ are both zero on constrained DOFs, so $\phi^\top Mr$ and
  $r^\top Mr$ involve only the free block $M_{ff}$. The total of a supported body is
  $r_f^\top M_{ff}r_f$, not the body mass. This is not the base-excitation load
  $-(M_{ff}r_f + M_{fc}r_c)\ddot u_g$ for consistent mass; with lumped mass the two agree.
- **Sum rule.** Over all free modes, rigid included, $\sum_i m^\text{eff}_{ik} =
  \text{total}_k$ for M-orthogonal modes (the modes span the free space).
- **Free bodies.** On a free body every elastic mode has $m^\text{eff} = 0$ in all 6
  directions, since each influence field is a rigid mode and elastic modes are M-orthogonal to
  them; the 6 rigid modes carry the totals. Effective masses are informative on supported (or
  partly supported) bodies.

## Clusters and the frequency table

`clusters(lam, rtol=1e-4, atol=0.0)` splits the ascending `lam` into index arrays. Neighbours
join when

$$\lambda_{i+1} - \lambda_i \le \text{rtol}\,|\lambda_{i+1}| + \text{atol}.$$

- Rigid modes enter as exact zeros: `solve_modes` returns no $\lambda$ for them, and the recipe
  passes zeros. Dense oracles set them to 0 or pass `atol`.
- The default rtol $10^{-4}$ is `solve_modes`' own cluster rule ($100\,\text{tol}$ at the default
  tol $10^{-6}$). Below about $100\,\text{tol}$ the per-mode vectors mix ($\sin\theta \approx
  \text{tol}/\text{gap}$, Davis–Kahan), so a split there would not be meaningful. Merging
  distinct modes only coarsens a row; it never makes it wrong.
- The solver merges at $\lambda_i - \lambda_{i-1} < 100\,t_{i-1}|\lambda_{i-1}|$, with
  $t_j \ge$ tol its per-mode acceptance threshold (or $100\,r_\text{acc}$ in shift-invert
  slices). `cluster_rtol(info)` $= 100\max(10^{-6}, \max_j t_j, r_\text{acc})$ reproduces that
  from a converged `solve_modes` info, and the recipe passes `max(1e-4, cluster_rtol(info))`.
  The looser $|\lambda_{i+1}|$ and $\le$ never split a solver cluster.
- **Basis invariance.** Per cluster $C$, $\sum_{i\in C} m^\text{eff}_{ik} = r_k^\top MP_Cr_k$,
  with $P_C$ the M-orthogonal projector onto the cluster's span, so cluster sums do not depend on
  the basis inside a degenerate cluster. Per-mode values do.

`frequency_table(lam, m_eff, total, rtol=1e-4, atol=0.0)` returns a dict:

| key | content |
|---|---|
| `clusters` | index arrays |
| `lam` | cluster mean $\lambda$ |
| `f` | $\sqrt{\max(\lambda, 0)}/2\pi$ (Hz for SI inputs; tiny negative rigid values give 0) |
| `m_eff` | summed effective mass `(n_clusters, 6)` |
| `fraction` | `m_eff / total`, 0 where the total is 0 |
| `cumulative` | cumulative sum of `fraction` |
| `total` | the totals, echoed |

After an `n_modes` solve the last cluster may be incomplete, and after a band solve from
$f_1 > 0$ the first may be too, so their sums are not basis invariant. Band solves return whole
clusters at the top edge. `format_table(t)` gives a fixed-width text table: modes (1-based), f,
the 6 fractions and the 3 translational cumulative fractions.

## Strain and stress recovery

`RECOVER[t](X, table_e, u)` (`jaxmodal.elements`, see `docs/elements.md`) returns the
quadrature points $NX$, the weights $w\det J$, and the global Voigt strain and stress (engineering
shear) at the quadrature points of one element: `hex_recover` ($\varepsilon = Bu$,
$\sigma = C\varepsilon$, with $C$ rotated to the element's orientation) for hex8/hex20/hex27,
`shell_recover` (ANS plus condensed EAS strains, per ply) for `ssh8`. For every type
$u^\top K_e u = \sum_q w_q\,\varepsilon_q\cdot\sigma_q$.

- `recover(mesh, materials, u, orientations=None, chunk=1024)` takes one field `u (n_dof,)` and
  returns per block `{"points" (n_e, nq, 3), "weights" (n_e, nq), "eps", "sigma" (n_e, nq, 6)}`.
  `ssh8` padded plies have weight 0. Use it for quadrature-point detail on a few modes and for
  per-ply values on laminates.
- `element_stress(mesh, materials, U, orientations=None, chunk=1024, batch=64)` returns the
  volume averages $\bar\sigma_e = \sum_q w_q\sigma_q/\sum_q w_q$ and $\bar\varepsilon_e$,
  `(n_el, k, 6)` in element order (1-D `U`: `(n_el, 6)`). On laminates the average mixes the
  plies.
- `von_mises(sigma)`: $\sqrt{\tfrac12[(\sigma_{11}-\sigma_{22})^2 + (\sigma_{22}-\sigma_{33})^2
  + (\sigma_{33}-\sigma_{11})^2] + 3(\sigma_{23}^2 + \sigma_{13}^2 + \sigma_{12}^2)}$ in Voigt
  order. The von Mises stress of the element average is not the average of point values.

Both run the material, laminate and element checks (`check_elements(mesh, "consistent")`), as
`assemble` does. A mass-normalized mode gives stresses per unit modal amplitude; stresses do not
depend on the mass model. Mode shapes have no physical amplitude, so only ratios (stress
distribution, hot spots) are meaningful without a forced or base-excitation response.

## Export

`write_modes(path, mesh, V, cell_data=None)` writes a binary zlib `.vtu` (other extensions
raise):

- point data `mode_000`, `mode_001`, ... $= V_{:,i}$ as `(n_nodes, 3)`, zero-padded to
  $\max(3, \text{digits}(n - 1))$ so names sort in index order;
- `cell_data` `{name: (n_el,) or (n_el, c)}` in element order, numeric (`int`, `uint`, `float`),
  split per block; names are `str` without `<>&"'` and must not be `material` or `orientation`,
  which are written as in `mesh.write`. meshio cannot read back 3-D cell arrays, so they raise.

Data is written unchanged. VTK reads a 6-component array as a symmetric tensor in the order
XX, YY, ZZ, XY, YZ, XZ with tensor shear, so write `sigma[..., [0, 1, 2, 5, 3, 4]]`, and for
strains halve the shear terms first (as the recipe does). `ssh8` blocks are written as
`hexahedron` and read back as hex8; consecutive hexahedron blocks may merge on read (see
`docs/mesh.md`). No frequencies are stored in the VTU (meshio's VTU writer has no field data);
write `format_table` next to it. XDMF/HDF5 and parallel output are PR 19.

## Recipe

After `solve_modes`, with `mesh`, `materials` and `bcs` as in the solve (here 12 modes at a tight
tolerance, so that the clusters are sharp). The operator is the one `solve_modes` builds
(`info["operator"]`), with the same `mass`, `nodal_mass` and `orientations`:

<!-- recipe -->
```python
import numpy as np

from jaxmodal import post
from jaxmodal.eigen import solve_modes
from jaxmodal.operators import dof_mask, stencil, unstructured

lam, V, info = solve_modes(mesh, materials, bcs, 12, tol=1e-10)
op = (stencil if info["operator"] == "stencil" else unstructured)(mesh, materials)
r = post.influence(mesh, dof_mask(mesh, bcs))
_, m_rigid, total = post.participation(info["rigid"], op, r)  # rigid modes: lam = 0
_, m_el, _ = post.participation(V, op, r)
lam_all, m_eff = np.r_[np.zeros(info["n_rigid"]), lam], np.vstack([m_rigid, m_el])
table = post.frequency_table(lam_all, m_eff, total, rtol=max(1e-4, post.cluster_rtol(info)))
with open("modes.txt", "w") as f:
    f.write(post.format_table(table) + "\n")

eps, sigma = post.element_stress(mesh, materials, V)  # (n_el, 12, 6); V[:, sel] for large models
vtk, half = [0, 1, 2, 5, 3, 4], np.array([1, 1, 1, 0.5, 0.5, 0.5])  # VTK tensor order and shear
cells = {f"vm_{i:03d}": post.von_mises(sigma[:, i]) for i in range(V.shape[1])}
cells |= {"sigma_000": sigma[:, 0, vtk], "eps_000": (eps[:, 0] * half)[:, vtk]}
post.write_modes("modes.vtu", mesh, V, cells)
```

Stacking the small `(n, 6)` outputs avoids `np.hstack([info["rigid"], V])`, which would copy V.
`cluster_rtol` is valid for converged solves only.

## Cost and memory

| function | `apply_M` columns | host | device |
|---|---|---|---|
| `normalize(V, M)` | n | copy of V | $\approx (2 + \text{ws})\,w\,n_\text{dof}\cdot 8$ B |
| `mac(A, B, M)` | $n_A + n_B$ ($n_A$ when `B is A`) | $n_A n_B$ | same |
| `participation(V, M, r)` | $n + 6$ | $O(6\,n_\text{dof})$ | same |
| `element_stress` | none | $96\,n_\text{el}\,k$ B | chunk × kb element values |

- $w = \min(\text{batch}, n)$, ws the operator's own workspace per column. Host memory is the
  inputs plus $O(w\,n_\text{dof})$ per batch.
- The host GEMM of `_gram` ($A^\top MB$) is $2 n_A w\,n_\text{dof}$ flops per batch.
- `element_stress` runs `ceil(k / batch)` × (blocks × chunks) kernel calls; geometry and the
  `ssh8` EAS factorization are computed once per chunk and column batch. `recover` is one pass
  per field.
- VTU size: about $24\,n_\text{nodes}$ B per mode before zlib, and zlib gains little on fp64
  fields. meshio builds each array in memory, so export `V[:, sel]` for large models.
- Compiles: one `apply_M` per batch width, one recovery kernel per (type, chunk, kb) and per ply
  count for `ssh8`; repeat calls compile nothing.

Measured once (M-series laptop CPU, 64 GB, hex20 $43^3$ box, 1.0M DOF, 79.5k elements, 256
random columns): `_gram` $V^\top MV$ 1.6 s (0.41 s per 64-column batch, nearly all `apply_M`;
the host GEMM is 0.06 s), `participation` 1.7 s, `element_stress` of 64 columns 2.5 s (32 ms
per 1024-element chunk; `ssh8` one ply: 16 ms). Peak RSS 6.8 GB, of which V is 1.9 GB and the
operator and mesh 1.7 GB.
