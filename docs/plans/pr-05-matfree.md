# PR 5 plan: matrix-free unstructured operator

Branch `pr-05-matfree`, from `main` after PR 4. Scope: the roadmap's PR 5 (Contents, Docs, Tests): matrix-free `apply_K`, `apply_M`, `diag_K`, `diag_M` on hex8/hex20/hex27 meshes, in stored-Ke and on-the-fly modes, with scatter and gather assembly; per-element scalings $s_K$, $s_M$ (also added to PR 4's `assemble`); nodal masses; RCM reordering; a memory estimator; the CPU scatter-vs-gather benchmark.

## Measured facts (prototype, jax 0.11.2, fp64, M3 Max CPU, `box_mesh`, isotropic $C$, warm)

- **Stored Ke, scatter** (`lax.scan` over element chunks, `out.at[dofs].add`), relative error vs the full-Ke reference 0:
  - hex20 $30^3$ (349k DOF): m = 1 0.026 s (chunk 64), m = 16 0.069 s.
  - hex8 $60^3$ (681k DOF): m = 1 0.044 s (chunk 256).
  - hex27 $24^3$ (353k DOF): m = 1 0.028 s.
  - About 30 GB/s of Ke traffic: the apply is bandwidth-bound by Ke.
- **Packed symmetric Ke** (upper triangle, $k(k+1)/2$ floats; unpacked per chunk by an index gather) is as fast as full Ke (hex20 m = 1 0.029 s vs 0.026 s; m = 16 0.065 s vs 0.069 s; hex27 0.023 s vs 0.028 s) and halves the storage: the roadmap's "Ke ≈ 1.2 GB (symmetric)" at 1M hex20 DOF.
- **On the fly** (invJ and w·detJ per quadrature point, strain → stress → $B^\top\sigma$): 3–5× slower than stored (hex20 m = 1 0.124 s, m = 16 0.274 s; hex8 0.189 s; hex27 0.074 s), error ≤ 1e-15 relative.
- **Gather** (node ranges of $R$ nodes on an RCM-ordered mesh, elements sorted by min node, a padded element window per range, inverse map summed by a gather), relative difference to scatter 3.5e-16:
  - hex20 $30^3$, $R = 16384$ (8 ranges, window 6416 elements, redundancy 1.9): m = 1 0.048 s vs scatter 0.026 s; m = 16 0.062 s vs 0.043 s.
  - hex20 $30^3$, $R = 65536$ (2 ranges, redundancy 1.15): m = 16 0.042 s vs scatter 0.067 s.
  - hex8 $60^3$, $R = 32768$ (redundancy 1.35): m = 1 0.044 s vs 0.052 s; m = 16 0.088 s vs 0.145 s.
  - Max DOF valence 8 (box meshes).
  - So neither path dominates on CPU; the benchmark below picks the Mac default.

## Files

```
src/jaxmodal/mesh/reorder.py            rcm (node RCM + elements sorted by min node)
src/jaxmodal/mesh/__init__.py           + rcm
src/jaxmodal/operators/matfree.py       Unstructured (pytree), unstructured, scaled, apply_K, apply_M, diag_K, diag_M
src/jaxmodal/operators/assembled.py     + s_K, s_M
src/jaxmodal/operators/__init__.py      + the names above
src/jaxmodal/memory.py                  unstructured_memory (estimator)
benchmarks/scatter_gather.py            CPU scatter vs gather benchmark -> benchmarks/results/scatter_gather.json
tests/test_reorder.py
tests/test_matfree.py
docs/operators.md                       + matrix-free sections
docs/mesh.md                            + RCM
```

## API

```python
from jaxmodal.mesh import rcm
from jaxmodal.operators import unstructured, scaled, apply_K, apply_M, diag_K, diag_M

mesh, p, q = rcm(mesh)                        # node permutation p, element permutation q (new -> old)
op = unstructured(mesh, materials, orientations=None, mode="stored", mass="consistent",
                  nodal_mass=None, path=None, chunk=None, range_nodes=None)
op = scaled(op, s_K=s_K, s_M=s_M)             # (n_el,) per-element scalings, default ones
y = apply_K(op, v)                            # v: (n_dof, m) -> (n_dof, m); jitted
y = apply_M(op, v); dK = diag_K(op); dM = diag_M(op)   # diagonals (n_dof,)
```

- `mode`: `"stored"` (packed symmetric $K_e$ per element) or `"otf"` (on the fly).
- `mass`: `"consistent"` or `"lumped"` (row-sum hex8/hex27, HRZ hex20, as PR 3).
- `path`: `"scatter"`, `"gather"` or `None` = `DEFAULT_PATH[jax.default_backend()]` (CPU from the benchmark; GPU `"scatter"` until the deferred GPU check).
- `chunk` (elements per scan step) and `range_nodes` (nodes per gather range): `None` = per-type defaults from the benchmark.
- Host checks first, as `assemble`: `phases`, `check_materials`, `check_elements(mesh, mass)`, `nodal_mass` (finite, ≥ 0, `(n_nodes,)`).
- No masking here: callers form $m \odot K(m \odot v)$ (PR 7/PR 9). Operators apply natively to blocks `(n_dof, m)`; `m = 1` is `(n_dof, 1)`.

## The operator pytree (Principle 3)

`Unstructured` is a frozen dataclass registered with `jax.tree_util.register_dataclass`. Leaves (all `jit` arguments, nothing closed over):

- per block (a tuple of `Block`s, one per mesh block): `dofs` (n_pad, k) int32; `phase` (n_pad,) int32; stored mode: `Ke` (n_pad, k(k+1)/2) packed upper triangle and `wdet` (n_pad, nq); otf mode: `geo` (n_pad, nq, 10) = invJ (9) and w·detJ (1) per quadrature point (PR 3's `jacobian`);
- `C` (P, 6, 6) and `rho` (P,) from `phase_table` (otf stiffness and every mass use them; stored $K_e$ bakes $C$ in);
- `s_K`, `s_M` (n_el,) in global element order, default ones; `nodal` (n_dof,) = `repeat(nodal_mass, 3)` (zeros if `None`);
- gather path: per block `starts` (n_ranges,) int32 window starts; `inv` (n_ranges, 3R, valence) int32.

Static metadata: block type, element offset, `n_el` per block, chunk, `n_dof`, `mass`, `mode`, `path`, $R$, window width $W$ per block. Replacing `s_K`, `s_M`, `C`, `rho` or `nodal` therefore never recompiles (tested).

- Each block is padded to a multiple of its chunk (gather: also $W$ a multiple of the chunk) by repeating its first element; padded elements get $s = 0$ inside the apply (`jnp.pad` of the block's slice of `s_K`/`s_M`), so they contribute nothing.
- int32: `check_int32` on `n_dof`, every `n_pad · k`, the phase index, the inverse map size and the window index range `Σ_b W_b k_b + 1` before casting (the phase index cast is the one `docs/materials.md` assigns to PR 5).

## Element kernels (per chunk, block-native)

$u_e = v[\text{dofs}_e]$ is $(c, k, m)$, $k = 3n$.

- **Stiffness, stored:** $y_e = s_{K,e}\,K_e u_e$ with $K_e$ unpacked from the upper triangle by one index gather per chunk.
- **Stiffness, otf:** with $G_q = J_q^{-\top}\nabla_\xi N$ from invJ, $\varepsilon_q = B_q u_e$, $\sigma_q = C_e \varepsilon_q$, $y_e = s_{K,e}\sum_q w_q \det J_q\, B_q^\top \sigma_q$ (the same $B$ selector as PR 3), never forming $K_e$.
- **Consistent mass** (both modes, from w·detJ): $y_{e,ai} = \rho_e s_{M,e} \sum_q w_q\det J_q\, N_{qa} \sum_b N_{qb} u_{e,bi}$.
- **Lumped mass:** a diagonal vector $d$ assembled from the element lumped diagonals (row-sum $\rho s \sum_q w\det J\, N_{qa}$; HRZ $d_a = \sum_q w \det J N_{qa}^2$ scaled to the element mass), then $y = d \odot v$; recomputed per call (it depends on $s_M$).
- **Diagonals:** `diag_K` assembles $s_{K,e}\,\mathrm{diag}(K_e)$ (stored: the packed diagonal; otf: $\sum_q w\det J\, b_{qai}^\top C_e b_{qai}$ with $b_{qai}$ the $B$ column of DOF $(a, i)$); `diag_M` assembles the consistent diagonal $\rho s_M \sum_q w\det J N_{qa}^2$ (×3) or the lumped $d$. Both add `nodal` for M.
- Nodal masses: `apply_M` adds `nodal[:, None] * v`.

## Assembly paths

Both take a per-block element kernel `f(block data slice, u_e) -> y_e` and return the global $(n_\text{dof}, m)$ result.

- **Scatter:** per block, `lax.scan` over chunks with the output carried (donated); `out.at[dofs].add(y)` (`segment_sum` semantics).
- **Gather** (roadmap: on an RCM-ordered mesh, chunked by node range):
  - Nodes are split into ranges of $R$ nodes ($n_\text{ranges} = \lceil n_\text{nodes}/R\rceil$, DOFs padded to $3R\,n_\text{ranges}$).
  - Per block, elements are assumed sorted by min node (as `rcm` leaves them); range $r$'s window is the contiguous element interval from the first element with max node ≥ $r_0$ to the last with min node < $r_1$, padded to a fixed width $W_b$ (the max over ranges, rounded up to the chunk) and clipped into the block.
  - **Node-local numbering:** the inverse map is indexed by range-local DOF ($0 \le j < 3R$) and lists, padded to the max valence with a sentinel, the flat window-local positions $(\text{block offset} + e_\text{local} k + a)$ of every element contribution to that DOF. So its entries are bounded by the window size, not by $n_\text{el}k$.
  - `lax.scan` over ranges: per block, the window's element contributions $Y_b$ $(W_b k_b, m)$ are computed by an inner `scan` over chunks of the window (bounding the $K_e$ unpack); the windows are concatenated with one zero row (the sentinel) and the range result is `Y[inv_r].sum(1)`, stacked as the scan output.
  - Correctness does not depend on the element order (a bad order only widens $W$); the estimator reports the window cost.
- Layouts (padding, windows, inverse maps) are built on the host in NumPy by one function, shared with the estimator.

## RCM (`mesh/reorder.py`)

```python
mesh2, p, q = rcm(mesh)   # p: (n_nodes,) new -> old; q: (n_el,) new -> old global elements
```

- Node graph $A = E^\top E$ from the sparse element–node incidence $E$ (all blocks), `scipy.sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)`.
- Then elements within each block are stably sorted by their min (new) node index; blocks keep `TYPES` order.
- The new mesh follows `docs/mesh.md`'s renumbering rules: points, connectivity, node sets, element sets, labels, voxel maps (so a voxel mesh stays valid).

## `assemble` gains $s_K$, $s_M$

`assemble(..., s_K=None, s_M=None)`: finite, non-negative `(n_el,)` arrays in global element order; element $e$'s $K_e$ (resp. consistent $M_e$ or lumped diagonal) is multiplied by $s_{K,e}$ (resp. $s_{M,e}$) before summation. Nodal masses are not scaled. `None` means ones (bit-identical to PR 4).

## Memory estimator (`memory.py`)

```python
unstructured_memory(mesh, m, mode="stored", mass="consistent", path=None, chunk=None, range_nodes=None)
    -> dict(storage=..., vectors=..., workspace=..., total=...)   # bytes
```

- `storage`: every operator leaf (sizes from the shared host layout: padded counts, $W_b$, valence).
- `vectors`: $v$ and the output, $2 \cdot 8\,n_\text{dof} m$.
- `workspace`: the apply temporaries: scatter per chunk $u_e$, unpacked $K_e$, $y_e$ (× a factor for XLA double buffering, measured); gather the window buffers $\sum_b W_b k_b m$, the gathered range rows $3R \cdot \text{valence} \cdot m$, the per-chunk temporaries; plus the build transient (one chunk of full $K_e$ before packing) and a fixed runtime overhead (measured once, stated).

## Benchmark (`benchmarks/scatter_gather.py`)

hex8/hex20/hex27 box meshes at ≈ 1M DOF, RCM-ordered, stored and otf, m ∈ {1, 16}, several chunks and ranges; writes `benchmarks/results/scatter_gather.json` (with `versions()`); picks `DEFAULT_PATH["cpu"]` and the per-type default `chunk`/`range_nodes` (recorded in `docs/operators.md`). The GPU run is a deferred check.

## Tests

Tolerances per Principle 1 (‖a − b‖_∞ ≤ tol ‖b‖_∞ over the whole result block). Fixed seeds.

`tests/test_reorder.py`:

1. **RCM:** on a randomly node- and element-shuffled mixed hex8/hex20 mesh and a masked voxel mesh: `p`, `q` are permutations; points, connectivity, node sets, element sets, labels and voxel maps follow the renumbering rules (checked against the shuffle inverse); elements are sorted by min node within each block; the bandwidth of $A$ after RCM is ≤ that of the shuffled mesh and equal to SciPy's RCM on the same graph; `check_elements` and the `Mesh` constructor pass.

`tests/test_matfree.py` (meshes: the PR 4 mixed hex8 + hex20 glued mesh, distorted, two materials and two orientations mixed; and a distorted hex27 box; both RCM-reordered; random $s_K, s_M \in [0.1, 2]$, random nodal masses with zeros):

2. **assemble with scalings:** $K(s_K)$, $M(s_M)$ (consistent and lumped) equal the dense loop reference with per-element scaling to 1e-14; `None` is bit-identical to ones; nodal masses unscaled; wrong shape, negative or non-finite scalings raise.
3. **apply vs SciPy:** for mode ∈ {stored, otf} × path ∈ {scatter, gather} × mass ∈ {consistent, lumped} × m ∈ {1, 5}, with small `chunk` (several chunks, a padded last one) and small `range_nodes` (several ranges): `apply_K(op, v)` and `apply_M(op, v)` equal `K @ v` and `M @ v` from the scaled PR 4 `assemble` on the same (reordered) mesh to 1e-12; and on the original mesh, the reordered result equals the original SciPy product permuted by $p$ (the roadmap's "permuted").
4. **Diagonals:** `diag_K`, `diag_M` equal `K.diagonal()`, `M.diagonal()` to 1e-12 for every mode, path and mass.
5. **No recompile:** `jax.jit(apply_K).lower(op, v).compile()` called with an operator whose `s_K`, `C`, `nodal` changed gives the new SciPy product (same compiled executable).
6. **int32:** a monkeypatched `check_int32` records `n_dof`, every `<type>_dofs`, `phase`, and (gather) `inv` and `window` sizes; the leaves are int32.
7. **Inputs:** unknown `mode`/`mass`/`path`, bad `nodal_mass`, wrong-shape `scaled` arrays raise; `check_elements`/`check_materials` errors propagate; a `v` of wrong leading size raises.
8. **Gather layout:** every element contribution appears exactly once in the inverse maps (count of non-sentinel entries = $\sum_b n_{\text{el},b} k_b$, each (element, local DOF) once); windows lie inside their block.
9. **Compile time** (slow): with the chunk fixed and the persistent cache disabled, `jax.jit(apply_K).lower(...)` on `ShapeDtypeStruct` inputs for hex20 at 1M elements compiles in ≤ 1.5× the time at 10k elements (scatter and gather; the structures come from a small operator with scaled shapes and static sizes).
10. **Estimator** (slow, `benchmarks/harness.measure` in a subprocess): stored Ke, ≈ 2M hex20 DOF ($55^3$), m = 1 and m = 16, with the default path and with the gather path: build plus one `apply_K` and `apply_M` has a measured peak within 20% of `unstructured_memory(...)["total"]` (estimate ≥ 2 GB).

## Docs

- `docs/operators.md`: the matrix-free operator: API, pytree leaves and static fields, the kernel formulas (stored, otf, consistent/lumped mass, diagonals), scatter and gather (windows, node-local numbering, padding), the int32 checks, the benchmark table and the chosen defaults, the estimator and its validation, cost per apply.
- `docs/mesh.md`: `rcm`.
- README: status, and a snippet (box → `rcm` → `unstructured` → `apply_K`), executed by `tests/test_readme.py`.
- `docs/ROADMAP.md`: status line; the deferred GPU scatter-vs-gather check stays in "Deferred GPU checks".

## Not covered

- Masking, rigid modes (PR 7); voxel stencil (PR 6); solid-shell (PR 17); Morton ordering (PR 18).
- fp32 operators (never: iteration vectors are fp64).
