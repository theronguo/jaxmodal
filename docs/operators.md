# Operators

`jaxmodal.operators` holds the global DOF map, the int32 overflow assert, the assembled SciPy reference operators (the oracle), the matrix-free device operators (`unstructured`, PR 5; the voxel stencil `stencil`, PR 6), and Dirichlet masking, rigid body modes and deflation (PR 7, [`boundary_conditions.md`](boundary_conditions.md)).

```python
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import assemble

mesh = box_mesh((4, 2, 2), spacing=(0.25, 0.25, 0.25), type="hex20")
# materials: label -> Material, as in docs/materials.md
K, M = assemble(mesh, materials)  # scipy.sparse.csr_array, float64, int32 indices
```

## DOF numbering

- **Global DOF** $3n + i$ is component $i$ of node $n$; `n_dof = 3 * len(mesh.points)`.
- **Element DOFs** follow PR 3's node-major order $3a + i$, so `element_dofs(mesh)[t]` is $3\,\mathrm{conn}[:, a] + i$, an `(n_el_type, 3n)` int32 array per block.
- **int32:** `check_int32(**sizes)` is the one overflow assert (Principle 6). It takes named plain ints and raises `ValueError` naming every size $\ge 2^{31}$. `element_dofs` checks `n_dof` and, per block, `<type>_dofs` $= n_\text{el} \cdot 3n$, then casts. Every later PR that sends a host index array to the device passes its sizes through `check_int32` and casts there: `unstructured` checks `n_dof`, `<type>_dofs` $= n_\text{pad} \cdot 3n$ (padded), `phase` (the phase count) and, on the gather path, `inv` (the inverse-map size) and `window` ($\sum_b W_b k_b + 1$); `stencil` checks `n_dof`, the `node` and `cell` grid sizes and `phase`.

## Assembled reference (`assemble`)

```python
K, M = assemble(
    mesh,
    materials,
    orientations=None,
    mass="consistent",
    nodal_mass=None,
    chunk=1024,
    s_K=None,
    s_M=None,
)
```

1. **Host checks:** `chunk >= 1`; x64 must be on (`jaxmodal.config.enable_x64()`), else `ValueError`, so the oracle is never silently fp32. Then `phases(mesh)`, `check_materials` and `check_elements(mesh, mass)`, the `nodal_mass` checks below, and `s_K`, `s_M`: `None` or finite, non-negative `(n_el,)` arrays in global element order (zeros allowed).
2. **Element matrices:** per block, in chunks of $c = \min(\text{chunk}, n_\text{el})$ elements, the jitted PR 3 kernels `stiffness` and `mass` (or `lumped_mass`) with the rotated per-phase $C$ and $\rho$ from `phase_table`. The last chunk is padded with its first element and sliced off, so a block compiles once per (type, $c$). Each element's $K_e$ is multiplied by $s_{K,e}$ and its consistent $M_e$ (or lumped diagonal) by $s_{M,e}$ before summation; `None` is bit-identical to ones. `chunk` bounds the $K_e$ temporaries and the XLA workspace, not total memory; 1024 is as fast as 4096 and peaks about 0.5 GB lower for hex20/hex27.
3. **Global matrices:** COO triplets with int32 rows/cols from `element_dofs` → `csr_array` (duplicates summed) → $\tfrac12(A + A^\top)$. Each $K_e$ is exactly symmetric, but the summed CSR is symmetric only to ~1e-16 (summation order of duplicates), so the result is symmetrized and then satisfies `(K != K.T).nnz == 0`. K is built before M, so only one matrix's triplets are alive at a time.
4. **Lumped mass** (`mass="lumped"`): a diagonal M from the PR 3 lumped diagonals (row-sum for hex8/hex27, HRZ for hex20), scaled by $s_M$.
5. **Nodal (point) masses:** `nodal_mass` is `None` or a finite, non-negative `(n_nodes,)` array, isotropic: after symmetrization,
   $$M \leftarrow M + \mathrm{diag}(\mathrm{repeat}(m, 3)),$$
   for consistent and lumped M alike, never scaled by $s_M$. Every later PR (5, 6, 9, 11, 14, 21) uses the same kwarg and shape.

**Size.** `assemble` is the oracle for PRs 5–8 and 12 at reference sizes, up to about $10^5$ DOF. That limit comes from PR 8's sparse factorization, not from assembly: on hex20 $20^3$ (107k DOF), `splu(K - σM)` took 241 s with a 16.5 GB peak. Assembly is far cheaper: whole-process peak RSS is 2.5 GB for hex20 $20^3$ (107k DOF), 2.6 GB for hex27 $16^3$ (108k DOF) and 2.5 GB for hex8 $40^3$ (207k DOF), i.e. about 24 KB per DOF for hex20/hex27 and 12 KB per DOF for hex8. The NumPy part is 44–66 B per element-matrix entry (triplets, CSR and symmetrization); the rest is the JAX baseline and the per-chunk XLA workspace. `assemble` is the oracle only: PR 13's production hex8 $A_P$ for AMG needs its own chunked assembler into a precomputed CSR pattern (about 1 GB per matrix per 1M hex8 DOF; `assemble` would need about 120 GB at 10M DOF).

**Cost** (M3 Max, fp64, isotropic $C$, warm):

| mesh | DOF | nonzeros of K | K and M (warm) | whole-process peak RSS |
|---|---|---|---|---|
| hex8 $30^3$ | 89k | 6.8M | 0.59 s | 1.3 GB |
| hex20 $15^3$ | 47k | 7.4M | 0.50 s | 1.4 GB |
| hex27 $12^3$ | 47k | 8.2M | 0.46 s | 1.5 GB |
| hex20 $20^3$ | 107k | 17.3M | 1.16 s | 2.5 GB |

## Dirichlet conditions, $A_P$ and rigid modes

BCs, masks, `masked`, `a_p`, rigid body modes and the deflation projector are described in [`boundary_conditions.md`](boundary_conditions.md).

## Matrix-free operator (`unstructured`)

```python
from jaxmodal.mesh import rcm
from jaxmodal.operators import unstructured

mesh, p, q = rcm(mesh)  # optional (docs/mesh.md)
op = unstructured(
    mesh,
    materials,
    orientations=None,
    mode="stored",  # or "otf"
    mass="consistent",  # or "lumped"
    nodal_mass=None,
    path=None,  # "scatter", "gather"; None = DEFAULT_PATH[backend]
    chunk=None,
    range_nodes=None,
)
op = op.scaled(s_K=s_K, s_M=s_M)  # (n_el,) per-element scalings, default ones
y = op.apply_K(v)  # v: (n_dof, m) -> (n_dof, m), jitted
y = op.apply_M(v)
dK, dM = op.diag_K(), op.diag_M()  # (n_dof,)
```

- **Interface (Principle 2).** `apply_K`, `apply_M`, `diag_K`, `diag_M` and `scaled` are methods; the voxel operator (`stencil(...) -> Stencil`, estimator `stencil_memory`) has the same five, so PR 7 and PR 9 take either by duck typing. The solver interface is these five methods plus the `n_dof` property (both operators also have `n_el`), plus the rigid modes, which depend only on the mesh and the mask, so one function `rigid_modes(mesh, free)` serves both operators (PR 7, [`boundary_conditions.md`](boundary_conditions.md)). Each method calls a module-level `jax.jit` function of `(op, v)`: the operator is a jit argument and nothing is closed over, so the methods also work on a traced operator inside a solver loop.
- **Inputs.** Host checks as in `assemble` (x64 on, `phases`, `check_materials`, `check_elements(mesh, mass)`, `nodal_mass`), `chunk >= 1`, `range_nodes >= 1`. `v` must be 2-D with `n_dof` rows (a trace-time check); `m = 1` is `(n_dof, 1)`. No masking: callers form $m \odot K(m \odot v)$ (PR 7, PR 9).
- **`scaled(s_K=None, s_M=None)`**: `None` keeps the current value; otherwise the shape must be `(n_el,)`. Only shapes are checked, so it is traceable (use it under `jit` and `grad`). Values should be finite and $\ge 0$; $s = 0$ voids an element.
- **Stored mode bakes $C$ into $K_e$:** changing materials means building a new operator. In otf mode, `dataclasses.replace(op, C=...)` changes the stiffness without recompiling. A stored-mode operator with `C` set raises (it would be ignored).
- **After `rcm`** (or any `renumber`), user arrays follow the new numbering: `s_K[q]`, `s_M[q]`, `nodal_mass[p]`, and $v$ maps as `v.reshape(-1, 3, m)[p]` (new DOF $3n' + i$ is old DOF $3p[n'] + i$).

### Pytree (Principle 3)

`Unstructured` and `Block` are frozen dataclasses registered with `jax.tree_util.register_dataclass`. A leaf the mode or path does not use is `None`.

| leaf | shape | when |
|---|---|---|
| `blocks[b].dofs` | `(n_pad, k)` int32 | always |
| `blocks[b].phase` | `(n_pad,)` int32 | always |
| `blocks[b].Ke` | `(n_pad, k(k+1)/2)` packed upper triangle | stored |
| `blocks[b].wdet` | `(n_pad, nq)` $w_q \det J_q$ | stored |
| `blocks[b].geo` | `(n_pad, nq, 10)` $J_q^{-1}$ (9) and $w_q \det J_q$ | otf |
| `blocks[b].starts` | `(n_ranges,)` int32 window starts | gather |
| `C` | `(P, 6, 6)` rotated stiffness per phase | otf |
| `rho` | `(P,)` | always |
| `s_K`, `s_M` | `(n_el,)` global element order | always |
| `nodal` | `(n_dof,)` = `repeat(nodal_mass, 3)` | always |
| `inv` | `(n_ranges, 3R, valence)` int32 | gather |

Static: per block `type`, `offset`, `n_el`, `chunk`, window width `W`; global `n_nodes`, `mode`, `mass`, `path`, `R`, `n_ranges`. Replacing `s_K`, `s_M`, `rho`, `nodal` (and `C` in otf mode) never recompiles (tested). Blocks are padded to `n_pad`, a multiple of their chunk, by repeating their first element; padded elements get $s = 0$ inside every apply, so they contribute nothing.

**Build on the device.** Per block, one module-level `jax.jit` (static type and mode) runs `lax.map` over chunks of the padded connectivity; the stacked output is the padded leaf itself, with no slice and no host copy (a host fill or a sliced output doubles the build peak). The build blocks until done, so its temporaries are freed before the next block or the first apply. Stored $K_e$ comes from PR 3's `kernels.stiffness`, the same kernel as the oracle; otf is the independent check of the stiffness.

### Element kernels (per chunk of $c$ elements)

$u_e = v[\text{dofs}_e]$ is $(c, k, m)$ with $k = 3n$.

- **Stiffness, stored:** $y_e = s_{K,e} K_e u_e$, with $K_e$ unpacked from the upper triangle by one index gather per chunk.
- **Stiffness, otf:** with $G_q = J_q^{-\top}\nabla_\xi N$, $H_q = U_e^\top G_q$ (the displacement gradient), $\varepsilon_q = S : H_q$, $\sigma_q = C_e \varepsilon_q$:
  $$y_{e,ai} = s_{K,e} \sum_q w_q \det J_q \sum_{v,j} \sigma_{q,v} S_{vij} G_{q,aj},$$
  i.e. $\sum_q w_q\det J_q B_q^\top \sigma_q$ without forming $K_e$ or $B$.
- **Consistent mass** (from $w\det J$, the scalar $n\times n$ form of `docs/elements.md`): $y_{e,ai} = \rho_e s_{M,e} \sum_q w_q\det J_q N_{qa} \sum_b N_{qb} u_{e,bi}$.
- **Lumped mass:** $y = (d + \text{nodal}) \odot v$ with $d$ assembled per call from the element diagonals (it depends on $s_M$, $\rho$): row-sum (hex8, hex27) $d_a = \rho_e s_{M,e}\sum_q w_q\det J_q N_{qa}$; HRZ (hex20) $d_a = \rho_e s_{M,e} V_e \frac{\sum_q w_q\det J_q N_{qa}^2}{\sum_b\sum_q w_q\det J_q N_{qb}^2}$, $V_e = \sum_q w_q \det J_q$ (PR 3's `lumped_nodal`).
- **Diagonals:** `diag_K` assembles $s_{K,e}\,\mathrm{diag}\,K_e$ (stored: the packed diagonal; otf: $\sum_q w_q\det J_q\, b_{qai}^\top C_e b_{qai}$, $b_{qai}$ the $B$ column of DOF $(a, i)$). `diag_M` assembles $\rho_e s_{M,e}\sum_q w_q\det J_q N_{qa}^2$ (×3) or the lumped $d$, plus `nodal`. Diagonals and $d$ go through the operator's own path (kernels returning $(c, k, 1)$).

### Assembly paths

- **Scatter:** per block, `lax.scan` over chunks with the $(n_\text{dof}, m)$ output as the carry; `out.at[dofs].add(y)`. `v` is not donated (callers reuse it).
- **Gather:**
  - Nodes are split into $n_\text{ranges} = \lceil n_\text{nodes}/R\rceil$ ranges of exactly $R \le n_\text{nodes}$ nodes starting at $r_0 = \min(rR, n_\text{nodes} - R)$: the last range overlaps its predecessor instead of being padded, its overlap is recomputed and overwritten (equal up to rounding; the later range wins). `range_nodes` larger than `n_nodes` is clamped.
  - **Windows** do not depend on element order: per block and range, `start` and `end` are the min and max element index over elements with a node in $[r_0, r_0 + R)$ (`np.minimum.at`/`np.maximum.at`); $W_b$ is the widest interval rounded up to the chunk; starts are clipped into $[0, n_\text{pad} - W_b]$, and a range a block does not touch gets `start = 0`. A bad element order only widens $W_b$.
  - **Node-local numbering:** `inv[r, j]` lists, padded with a sentinel to the max node valence, the window-local positions $(\text{block offset} + (e - \text{start}) k + 3a + i)$ of all element contributions to range-local DOF $j$, so its entries are bounded by the window size $\sum_b W_b k_b$, not by $n_\text{el} k$.
  - `lax.scan` over ranges with the output as the carry: per block an inner scan over the window's chunks slices each chunk straight from the full leaves (`dynamic_slice` at `start + i c`; slicing the window first would copy its $K_e$), the window results are concatenated with one zero row (the sentinel), and `Y[inv[r]].sum(1)` is written with `dynamic_update_slice` at row $3r_0$. Dynamic indices are int32; the host asserts guarantee no slice clamps.
  - **$R$** (`range_nodes=None`): the candidates $\{2^j < n_\text{nodes} : 14 \le j \le 20\} \cup \{n_\text{nodes}\}$ are scored by their windows; the largest one whose per-column gather workspace (window buffers + gathered rows) is at most two vectors, $8(\sum_b W_b k_b + 3R\cdot\text{valence}) \le 2\cdot 8 n_\text{dof}$, is taken, else the smallest. Tiny meshes get one range. The rule is re-tuned in the deferred GPU check.
  - **Costs:** every range runs each block's full window, even blocks it does not touch. The inverse map is padded to the max valence (about 1.7× the mean on boxes), so one high-valence node inflates the map and the gathered rows.
- All layouts (padding, windows, inverse map, resolved path, chunk and $R$) come from one host function, `layout(mesh, path, chunk, range_nodes)`, shared by `unstructured` and the estimator.

### Defaults and the CPU benchmark

- `chunk=None`: per block, the largest power of two with $c\,k^2 \cdot 8 \le 32$ MB of unpacked $K_e$ (hex8 4096, hex20 1024, hex27 512), capped at the block size rounded up to a power of two. An explicit `chunk` is capped at the block size.
- `path=None`: `DEFAULT_PATH = {"cpu": "scatter", "gpu": "scatter"}`. The GPU entry is provisional until the deferred GPU benchmark (`docs/ROADMAP.md`, Deferred GPU checks).

**CPU benchmark** (`benchmarks/scatter_gather.py`, results in `benchmarks/results/scatter_gather.json`; M3 Max, fp64, jax 0.11.2, ≈ 1M DOF boxes, isotropic $C$; apply_K time, min of 5, each configuration in a fresh subprocess; default chunk and $R$):

| mesh | mode | order | scatter m = 1 | gather m = 1 | scatter m = 16 | gather m = 16 | $R$ |
|---|---|---|---|---|---|---|---|
| hex8 $69^3$ | stored | grid | 0.042 s | 0.050 s | 0.126 s | 0.196 s | 32768 |
| hex8 $69^3$ | stored | RCM | 0.042 s | 0.066 s | 0.128 s | 0.244 s | 32768 |
| hex8 $69^3$ | otf | grid | 0.095 s | 0.114 s | 0.450 s | 0.555 s | 32768 |
| hex8 $69^3$ | otf | RCM | 0.094 s | 0.150 s | 0.453 s | 0.723 s | 32768 |
| hex20 $44^3$ | stored | grid | 0.062 s | 0.083 s | 0.122 s | 0.192 s | 32768 |
| hex20 $44^3$ | stored | RCM | 0.062 s | 0.113 s | 0.127 s | 0.253 s | 32768 |
| hex20 $44^3$ | otf | grid | 0.167 s | 0.217 s | 0.501 s | 0.688 s | 32768 |
| hex20 $44^3$ | otf | RCM | 0.182 s | 0.333 s | 0.504 s | 0.948 s | 32768 |
| hex27 $35^3$ | stored | grid | 0.060 s | 0.086 s | 0.102 s | 0.175 s | 32768 |
| hex27 $35^3$ | stored | RCM | 0.058 s | 0.112 s | 0.108 s | 0.228 s | 32768 |
| hex27 $35^3$ | otf | grid | 0.122 s | 0.162 s | 0.259 s | 0.427 s | 32768 |
| hex27 $35^3$ | otf | RCM | 0.121 s | 0.245 s | 0.290 s | 0.590 s | 32768 |

- Scatter wins all 24 default comparisons (both orders, both modes, $m = 1, 16$), so `DEFAULT_PATH["cpu"] = "scatter"`. The apply is kernel-bound ($K_e$ unpack and contraction, 11–16% of the 185 GB/s triad in Ke traffic); the scatter itself is a small part of it, and gather adds window redundancy. RCM order widens the gather windows on boxes (grid order is narrower), so gather is slower after RCM there; scatter does not care.
- otf is 2–4× slower than stored.
- The chunk rule is within 10% of the best of $c/4$, $c$, $4c$ for every type and $m$ (`chunk_rule_ok` in the JSON; a fast test checks it and the default path). $R/2$ and $2R$ are recorded for information.
- Compile time of apply_K is 0.03–0.10 s.

### Memory estimator

```python
from jaxmodal.memory import unstructured_memory

est = unstructured_memory(
    mesh, m, mode="stored", mass="consistent", path=None, chunk=None, range_nodes=None
)
# {storage, vectors, workspace, build, overhead, total} in bytes
```

Call it on the mesh that will be used (after `rcm`, if any), with the same keyword arguments as `unstructured`.

- `storage`: every leaf, from `layout` (padded counts, $W_b$, valence).
- `vectors`: $3 \cdot 8\,n_\text{dof} m$: $v$ and two outputs (XLA frees an apply's output asynchronously, so it can overlap the next apply's).
- `workspace`: the apply temporaries. Scatter, stored: $8c(2km + k^2)$ ($u_e$, $y_e$, unpacked $K_e$). Otf: a fitted per-element envelope of apply_K, $c(a_t + b_t(m-1))$ (XLA's fusion of the strain and stress fields varies with $m$), or `diag_K`'s $B$ matrices, $c\,t_t^\text{stored}$, whichever is larger. Gather: $8\sum_b W_b k_b m$ plus the larger of the gathered rows $8\cdot 3R\cdot\text{valence}\cdot m$ (XLA does not fuse the gather with the sum) and the chunk temporaries (they are never live together).
- `build`: the padded connectivity and phase on the host, the phase index (and the int64 inverse map) plus one chunk of the build kernel's XLA temporaries, $c\,t_t$, with $t$ per element exactly linear in $c$:

  | | hex8 | hex20 | hex27 |
  |---|---|---|---|
  | stored build $t$ | 29.95 KB | 241.06 KB | 322.70 KB |
  | otf build $t$ | 1.22 KB | 4.10 KB | 4.10 KB |
  | otf apply_K $a_t$, $b_t$ | 6.1, 1.28 KB | 41.0, 3.17 KB | 54.6, 2.55 KB |

- `overhead`: 10 MB, the only constant fitted to measured peaks ($t$, $a$, $b$ come from `memory_analysis`; measured on hex20 $30^3$ and hex8 $60^3$; the formulas match `memory_analysis` without factors).
- `total = overhead + storage + max(build, vectors + workspace)`.
- Not counted: `layout`'s host transient (≈ 4× the int64 inverse map while it is built, e.g. 0.55 GB at hex20 $55^3$ gather). It is freed before any device leaf exists, so it does not set the peak of `unstructured`, but an estimator that composes this one with live device memory should add it.
- The estimate costs as much as `layout` (0.3 s at 1M DOF gather).
- **Composition:** later estimators (PR 9/11/13/14/15) add PR 5's `storage + workspace` and count `overhead` once; they count their own vectors.

## Voxel stencil (`stencil`)

```python
from jaxmodal.operators import reference_element, stencil

op = stencil(mesh, materials, orientations=None, mass="consistent", nodal_mass=None, slab=None)
op = op.scaled(s_K=s_K, s_M=s_M)  # (n_el,), default ones
y = op.apply_K(v)  # (n_dof, m) -> (n_dof, m); also apply_M, diag_K, diag_M
Ke, me = reference_element(op)  # (P, 24, 24) per phase; (8, 8) scalar mass per unit density
```

For `voxel_mesh` meshes (`mesh.voxel` set; it also works after `renumber`/`rcm`). Same interface and conventions as `unstructured` (DOF $3n + i$ in mesh node order, 2-D `v`, traceable `scaled`, nodal masses, consistent or lumped mass; on a box element row-sum and HRZ coincide at $\rho V/8$ per node), so results compare directly with `unstructured` and `assemble` on the same mesh.

- **Restriction:** one reference hex8 $K_e$ per phase (a (material, orientation) pair); elements differ only by phase, $s_K$ and $s_M$. The points must be the grid points. A distorted voxel mesh, hex20/hex27 grids or many phases (cost ∝ P: one GEMM per phase; e.g. one orientation per element) belong to `unstructured`.
- **Geometry check** (NumPy, before building): every used grid node's mesh point equals `origin + spacing * (i, j, k)` to $10^{-12}\max_d(|o_d| + h_d n_d)$, and every active cell's connectivity equals the `node_map` of its grid corners in VTK order (exact). Otherwise `ValueError("mesh does not match its voxel grid: ...")`. With `Mesh`'s bijection checks this makes the grid description equivalent to the mesh. A mesh without `voxel` raises too.
- **Inputs.** Host checks: x64 on, `phases`, `check_materials`, `nodal_mass`, `mass` ∈ {consistent, lumped}, `slab >= 1` (capped at the layer count), and the geometry check below; no `check_elements` (the reference cell is a box). `v` as in `unstructured`. `stencil_memory` also raises on a non-voxel mesh.
- **Leaves:** `node` $(L+1, B+1, A+1)$ and `cell` $(L, B, A)$ int32 id grids; `phase` $(n_\text{el})$ int32; `C` $(P, 6, 6)$, `rho` $(P)$; `spacing` $(3)$; `s_K`, `s_M` $(n_\text{el})$; `nodal` $(n_\text{dof})$. Static: `slab`, `axes`, `mass`, `n_nodes`, `n_el`. Changing `C`, `rho`, `spacing` (the operator then acts as the uniformly rescaled grid), $s$ or `nodal` never rebuilds or recompiles, and `jax.grad` with respect to all of them works.
- **Grid description** (for PR 11's coarse operators and PR 23's sensitivities): `node`/`cell` are the grid's node and cell ids transposed by `axes` (grid axes (x, y, z) in stored order, scan axis first) and padded along the scan axis from the layer count (`grid_layout(mesh.voxel.shape)[1]`) to $L$ (a multiple of `slab`); dropped nodes, inactive cells and padding hold the out-of-range sentinels `n_nodes` and `n_el`. The sentinels are out of range on purpose: `.at[].get(mode="fill", fill_value=0)` reads them as 0 and `.at[].add(mode="drop")` discards writes to them. `reference_element(op)` gives $K_e^p$ = `kernels.stiffness` and $m_e = \sum_q w_q\det J_q N_q N_q^\top$ on the reference cell $X = \text{spacing}\odot(\text{CORNERS}+1)/2$; the applies call it each time (a few µs).
- **Scan axis and slab:** slabs run along the longest grid axis (ties: the last), so a layer is the smallest cross-section (a flat $1000\times1000\times33$ plate gets 33k-cell layers instead of 1M-cell z-layers). `slab` (layers per scan step) defaults to $\max(1, \lfloor 2^{16}/A_\text{layer}\rfloor)$, about 64k cells, capped at the layer count; PR 5's `chunk` counts elements instead. A slab is at least one layer, so the workspace of a very large cross-section is what the estimator reports.

### Slab loop

One `lax.scan` over slabs carries the $(n_\text{nodes}, 3, m)$ output (initialized with `nodal * v` for `apply_M`: adding it afterwards costs a vector of temporaries). Per slab:

1. `nd` = the slab's node ids (slab + 1 layers: a one-layer halo shared with the next slab), `el` = its cell ids (`dynamic_slice`, int32 starts);
2. $u$ = `v.at[nd].get(mode="fill", fill_value=0)`, moved to component-major; the 8 corner slices (shifted by the VTK $\{0,1\}$ offsets permuted by `axes`) stacked into $U$ $(24, N)$, $N$ = cells·m;
3. **apply_K:** $Y = \sum_p w_p \odot (K_e^p U)$ with $w_p = s_K[\text{el}]\,[\text{phase}[\text{el}] = p]$, one GEMM per phase accumulated by a `lax.scan` over phases (one accumulator, so the workspace does not grow with P; P = 1 is a single GEMM); **apply_M (consistent):** $Y = \rho_{p(e)} s_{M,e}\,(m_e \otimes I_3)\,U$, the same $24\times24$ GEMM; **diagonals and lumped mass:** per-corner values ($\sum_p w_p\,\mathrm{diag} K_e^p$; $\rho s_M (m_e)_{aa}$ or the row sums $V/8$) through the same loop with $m = 1$; lumped `apply_M` $= (d + \text{nodal}) \odot v$, recomputed per call;
4. 8 shifted adds of $Y$ into a slab buffer, then `out.at[nd].add(..., mode="drop")`.

Every per-cell gather passes `fill_value=0`, so inactive and padded cells contribute exactly 0 (JAX's default fill for floats is NaN). No full-grid vector exists: the output is the only $n_\text{dof}$-sized buffer, and one code path serves full and masked grids.

**Why this layout** (prototype, M3 Max, $87^3$ full box, 2.04M DOF, m = 1 / 16): a node-major batched einsum takes 0.020 / 0.36–0.53 s; the corner-DOF-major GEMM 0.018 / 0.16–0.20 s but with $v$ and the output padded by a zero row, 558–1274 MB of temporaries at m = 16; with fill/drop instead of padding, 0.016 / 0.15–0.18 s and 98–192 MB. A dense grid-vector variant without gathers (full box only) was slower (0.024 / 0.31 s). At m = 16 the gather costs ≈ 9 ms of 157 ms; the corner stack, GEMM (≈ 280 GFLOP/s) and shifted adds take about a third each.

**Slab benchmark** (`benchmarks/stencil_slab.py`, results in `benchmarks/results/stencil_slab.json`; $87^3$, P = 1, apply_K, min of 5, fresh subprocesses; the harness unmaps freed large blocks, so these times include first-touch page faults on the output and temporaries, ≈ 3 / 23 ms; in a plain process the rule takes 0.0153 / 0.160 s):

| slab (cells) | 2 (15k) | 8 (61k, rule) | 32 (242k) |
|---|---|---|---|
| m = 1 | 0.0197 s | 0.0184 s | 0.0185 s |
| m = 16 | 0.196 s | 0.183 s | 0.231 s |

The rule is within 10% of the best for both $m$ (`slab_rule_ok`, checked by a fast test). Per DOF this is ≈ 4.5× faster than PR 5's stored-Ke `unstructured` at m = 1 and ≈ 1.4× at m = 16.

**Cost per apply** (m = 1, P = 1, for PR 18's roofline): per cell, $2 \cdot 24^2 = 1152$ flops; data moved per slab: the int32 `node`/`cell` slices (≈ 8 B per cell), the gathered $u$ and the scattered output (each ≈ 24 B per node read and written), $s$ (8 B per cell); the rest stays in slab-sized buffers.

### Memory estimator (`stencil_memory`)

```python
from jaxmodal.memory import stencil_memory

est = stencil_memory(mesh, m, mass="consistent", slab=None)  # same keys as unstructured_memory
```

- `storage`: the leaves, exact (`node`, `cell`, `phase` int32; `nodal` fp64 per DOF; `s_K`, `s_M`).
- `vectors`: $3 \cdot 8\,n_\text{dof} m$, as `unstructured_memory`.
- `workspace`: $8c\,(24 k_P m + 15)$ per slab of $c$ cells, $k_P = 2$ ($U$, $Y$) for one phase and 3 (plus the phase accumulator) otherwise; within 7% of `memory_analysis` for $m \in \{1, 4, 16, 24\}$, P ∈ {1, 2} (consistent `apply_M` equals the one-phase `apply_K`; diagonals and lumped `apply_M` are smaller).
- `build`: the int64 grids on the host; `overhead`: 10 MB; `total = overhead + storage + max(build, vectors + workspace)`; composition as for `unstructured_memory`.
- 2.04M DOF box, m = 24: storage 35 MB, workspace 0.57 GB; with six `(n_dof, 24)` arrays the estimate is 4.14 GB against a measured peak of 3.87 GB.
- 100M voxel DOF (a ≈ $322^3$ cube, one 104k-cell layer per slab): operator (grids and `phase`) ≈ 0.4 GB; $s_K$, $s_M$, `nodal` ≈ 1.3 GB; workspace ≈ 0.3 GB (P = 1) or 0.5 GB (P ≥ 2) at m = 8.

## Validation

`tests/test_operators.py`:

- **Assembly** equals a dense Python loop (explicit DOF lists, per-element `rotate(material, orientation)`, no `phases`/`element_dofs`) to 1e-14 on a distorted two-block mesh: hex20 and hex8 glued at a shared face, with two materials (one orthotropic, different $\rho$) and two orientations mixed in each block, for `chunk=7` (several chunks, padded last one) and `chunk=4096` (one chunk), consistent and lumped.
- **Patch test:** on a $3^3$ box per type, rotated orthotropic $C$, interior corner nodes moved per component by $\le 15\%$ of the edge, all boundary nodes prescribed $u = Ax + b$, the interior solved with `spsolve`. The stress is constant at every quadrature point to 1e-10 relative to $\|\sigma\|_\infty$; measured 1.2e-15 (hex8), 1.0e-14 (hex20), 1.3e-14 (hex27) for straight edges (non-corner nodes on the trilinear map of the corners), and 6.4e-15 (hex20), 1.6e-14 (hex27) for curved elements (non-corner interior nodes additionally offset per component by $\le 5\%$ of the edge).
- **Rigid translation mass:** $t_x^\top M t_x = \rho V + \sum m_n$ with the analytic volume, consistent and lumped, straight and curved; $t_x^\top M t_y = 0$.
- **Nodal masses** add exactly; `element_dofs` calls `check_int32` with every block's size.
- **Scalings** (`tests/test_matfree.py`): $K(s_K)$, $M(s_M)$ equal the dense loop with per-element scaling to 1e-14 (consistent and lumped, one $s = 0$); `None` is bit-identical to ones; nodal masses are unscaled; bad shapes and values raise.
- **Matrix-free** (`tests/test_matfree.py`), on four meshes: A, the mixed hex8 + hex20 mesh (two materials, two orientations), RCM-ordered; B, a curved hex27 $3^3$ box, RCM-ordered; C, mesh A shuffled without RCM (elements not sorted by min node); D, an asymmetric masked voxel mesh with mixed labels; E (gather only), an RCM-ordered hex20 $12 \times 2 \times 2$ box with `chunk=4`, `range_nodes=203`: narrow windows and two ranges with a large overlap, so the last range's window extension is visible. Random $s_K, s_M \in [0.1, 2]$ with one $s = 0$, nodal masses with zeros, `chunk=5` (pads every block) and `range_nodes=23` (several ranges, overlapping last one):
  - `apply_K`, `apply_M` ($m = 1, 5$), `diag_K`, `diag_M` equal the scaled `assemble` to 1e-12 for every mode, path and mass (C, E: gather); measured errors are about 1e-15. One case has one chunk and one range. The lumped HRZ formula is shared with the oracle (`kernels.lumped_nodal`), so it is validated by PR 3's element tests, not here.
  - **Permuted:** the RCM-ordered operator with `s[q]`, `nodal_mass[p]` and $v$ permuted by $p$ reproduces the original mesh's SciPy products.
  - **No recompile:** an executable compiled once gives the new SciPy products after replacing `s_K`, `s_M`, `rho`, `nodal` (and `C` in otf mode), consistent and lumped.
  - **Gradient:** `jax.grad` of $w^\top K(s_K) v$ equals the per-element $w_e^\top K_e v_e$ to 1e-12.
  - **Gather layout:** the inverse map, decoded independently, files every (element, local DOF) pair under the right DOF once per covering range (twice in the overlap), references no padded element, and keeps windows inside the padded block; the $R$ rule matches a brute-force recomputation on a host-only hex20 $44^3$ layout.
  - **int32** checks and casts; input errors; `None` leaves per mode and path.
  - **Estimator storage** equals the operator's leaf bytes exactly (except the tiny `rho`, `C`) for every mode and path.
  - **Slow:** compile time of apply_K + apply_M at 1M hex20 elements ≤ 1.5× that at 10k (fresh subprocesses, no cache, both modes and paths); the estimator total within 20% of the measured peak (build + apply_K + apply_M) at hex20 $55^3$ (2.08M DOF, estimate ≥ 2 GB) for both paths and $m = 1, 16$, and apply_K alone within 20% of workspace + output; `memory_analysis` temporaries within 10% of the workspace and build formulas.
- **Voxel stencil** (`tests/test_stencil.py`), all meshes with anisotropic spacing: F, a full $5\times4\times3$ box with an offset origin (scan axis x); S, PR 1's voxelized sphere on a $10\times12\times11$ grid (scan axis y); T, a masked $4\times5\times6$ grid (scan axis z) whose mask has a dropped node in a halo layer, an inactive cell next to an active one across a slab boundary and a corner-only contact, in three label variants (two materials, one orthotropic material in two orientations, both mixed: P = 4); R, a T variant after `rcm`. Random $s_K$, $s_M$ with one zero, nodal masses with zeros, slab ∈ {1, 4 (padded layers on F and T), None}:
  - `apply_K`, `apply_M` (m = 1, 5), `diag_K`, `diag_M` equal the scaled `assemble` to 1e-12 for both masses and every slab, and PR 5's `unstructured` to 1e-12; measured errors are about 1e-16.
  - **Gradients** (T, P = 4): with respect to $s_K$, $s_M$ (both masses), $\rho$ and $C$ equal central differences along a random direction to 1e-8, and the $s_K$ gradient equals the per-element $w_e^\top K_e v_e$ to 1e-12; with respect to `spacing` equal central differences of `assemble` on the rescaled mesh to 1e-7.
  - **No recompile:** one executable gives the new SciPy products after replacing $s_K$, $s_M$, `C`, `rho`, `spacing`, `nodal`, consistent and lumped.
  - **Geometry check:** a point moved by 1e-6, a cell with cyclically rotated corners (same nodes, positive $\det J$, wrong VTK order) and a non-voxel mesh raise.
  - `reference_element` equals PR 3's `stiffness` and `mass` on a mesh cell for every phase; `axes` and the layer count per mesh; int32 checks and casts; input errors; estimator storage equals the leaf bytes exactly.
  - **Slow:** `memory_analysis` temporaries of `apply_K`/`apply_M` within 10% of `workspace` at $87^3$ (m = 1, 4, 16, 24; P = 1, 2); the Mac acceptance (estimate within 20% of the measured peak of build + six `(n_dof, 24)` random arrays + `apply_K` + `apply_M` at 2.04M DOF, estimate ≥ 2 GB); the slab benchmark smoke run.
