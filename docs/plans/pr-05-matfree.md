# PR 5 plan: matrix-free unstructured operator

Branch `pr-05-matfree`, from `main` after PR 4. Scope: the roadmap's PR 5 (Contents, Docs, Tests): matrix-free `apply_K`, `apply_M`, `diag_K`, `diag_M` on hex8/hex20/hex27 meshes, in stored-Ke and on-the-fly modes, with scatter and gather assembly; per-element scalings $s_K$, $s_M$ (also added to PR 4's `assemble`); nodal masses; RCM reordering; a memory estimator; the CPU scatter-vs-gather benchmark.

## Measured facts (prototype, jax 0.11.2, fp64, M3 Max CPU, `box_mesh`, isotropic $C$, warm)

- **Packed symmetric Ke** (upper triangle, $k(k+1)/2$ floats; unpacked per chunk by an index gather) is as fast as full Ke and halves the storage: 1.25 GB at 1M hex20 DOF, 12.2 GB at 10M (the roadmap's budget).
- **Scatter vs gather** (stored packed Ke, RCM order; gather with the best $R$ and an inner chunk scan; relative difference 3.5e-16):

  | mesh | chunk | scatter m = 1 | gather m = 1 | scatter m = 16 | gather m = 16 |
  |---|---|---|---|---|---|
  | hex20 $30^3$ (349k DOF) | 1024 | 0.020 s | 0.024 s | 0.036 s | 0.053 s |
  | hex20 $44^3$ (1.08M) | 1024 | 0.063 s | 0.078 s | 0.112 s | 0.170 s |
  | hex20 $55^3$ (2.08M) | 1024 | 0.119 s | 0.171 s | 0.215 s | 0.364 s |
  | hex8 $69^3$ (1.03M) | 4096 | 0.045 s | 0.055 s | 0.116 s | 0.186 s |
  | hex27 $35^3$ (1.07M) | 512 | 0.059 s | 0.078 s | 0.096 s | 0.148 s |

  - The apply is kernel-bound (Ke unpack + contraction), not bandwidth-bound: at 1M hex20 the scatter itself is 0.004 s of 0.063 s at m = 1 and 0.038 s of 0.112 s at m = 16; Ke traffic is 11–16% of the 185 GB/s triad. Gather can remove at most the scatter part but adds 1.26–1.7× window redundancy, so scatter wins on CPU. An earlier "gather wins at m = 16" came from a chunk-64 scatter baseline.
  - So `DEFAULT_PATH["cpu"]` is expected to be `"scatter"` (the benchmark confirms); gather stays minimal and correct for the deferred GPU benchmark.
- **Chunk:** chunk 64 is 1.5–1.9× slower than the best; the best is about 32 MB of unpacked Ke per chunk (hex8 4096, hex20 1024, hex27 256–1024 equivalent).
- **On the fly** (invJ and w·detJ per quadrature point, strain → stress → $B^\top\sigma$): 3–5× slower than stored, error ≤ 1e-15 relative; `geo` is 2160 B per hex20 element vs 14640 B packed Ke.
- **Ke build** (2M hex20 DOF, packed Ke 2.46 GB, peak of build + one apply): host NumPy fill then `jnp.asarray` 4.94 GB; device `lax.map` with a trailing slice 5.68 GB; device `lax.map` whose stacked output is exactly the padded leaf 2.79 GB.
- **Compile time** does not depend on $n_\text{el}$ (scatter 28 ms at 10k and 1M elements; gather 35 vs 38 ms); single samples jitter by ±10 ms.
- **Gather layout at scale:** with a fixed $R = 2^{17}$, redundancy is 1.26 at 1M DOF and 1.86 at 10M (the RCM front grows), so $R$ is chosen per mesh. The padded inverse map wastes about 1.7× (mean 4.8 contributions per DOF, valence 8 on boxes); one high-valence node inflates the map and the rows buffer.
- **RCM vs natural order** (node-graph bandwidth, hex8 $20^3$ box): natural grid order 463, RCM 1261, randomly shuffled 9235. RCM repairs arbitrary orderings (e.g. Gmsh files); it does not beat the grid order of `box_mesh`/`voxel_mesh`. Host cost at 10M hex20 DOF: $E^\top E$ has 196M nonzeros (2.35 GB), 3 s.

## Files

```
src/jaxmodal/mesh/reorder.py            renumber(mesh, p, q), rcm(mesh)
src/jaxmodal/mesh/__init__.py           + renumber, rcm
src/jaxmodal/operators/matfree.py       Block, Unstructured (pytree with methods), unstructured, layout
src/jaxmodal/operators/assembled.py     + s_K, s_M
src/jaxmodal/operators/__init__.py      + Unstructured, unstructured
src/jaxmodal/memory.py                  unstructured_memory (estimator)
benchmarks/scatter_gather.py            CPU scatter vs gather benchmark -> benchmarks/results/scatter_gather.json
tests/helpers.py                        + mixed_mesh, distort, dense_reference (moved from test_operators.py; dense_reference gains s_K, s_M)
tests/test_reorder.py
tests/test_matfree.py
docs/operators.md, docs/mesh.md, docs/elements.md, docs/conventions.md, docs/ROADMAP.md, README.md
```

## API

```python
from jaxmodal.mesh import rcm
from jaxmodal.operators import unstructured

mesh, p, q = rcm(mesh)  # optional; node permutation p, element permutation q (new -> old)
op = unstructured(
    mesh,
    materials,
    orientations=None,
    mode="stored",
    mass="consistent",
    nodal_mass=None,
    path=None,
    chunk=None,
    range_nodes=None,
)
op = op.scaled(s_K=s_K, s_M=s_M)  # (n_el,) per-element scalings, default ones
y = op.apply_K(v)  # v: (n_dof, m) -> (n_dof, m); jitted
y = op.apply_M(v)
dK = op.diag_K()  # (n_dof,)
dM = op.diag_M()
```

- **Interface (Principle 2).** `apply_K`, `apply_M`, `diag_K`, `diag_M` and `scaled` are methods of the operator. PR 6's voxel operator (`stencil(...) -> Stencil`, estimator `stencil_memory`) provides the same five methods, and PR 7/PR 9 call only these, so solvers take either operator by duck typing. Each method calls a module-level `jax.jit` function of `(op, v)`, so the operator is a jit argument and nothing is closed over.
- `mode`: `"stored"` (packed symmetric $K_e$ per element) or `"otf"` (on the fly).
- `mass`: `"consistent"` or `"lumped"` (row-sum hex8/hex27, HRZ hex20, as PR 3).
- `path`: `"scatter"`, `"gather"` or `None` = `DEFAULT_PATH[jax.default_backend()]` (CPU from the benchmark; GPU `"scatter"` until the deferred GPU check).
- `chunk` (elements per scan step): `None` = the largest power of two with $c\,k^2 \cdot 8 \le 32$ MB of unpacked $K_e$ (hex8 4096, hex20 1024, hex27 512), capped at the block size rounded up to a power of two; the benchmark confirms the rule. `range_nodes` (nodes per gather range): `None` = chosen per mesh by `layout` (below). `layout(mesh, path, chunk, range_nodes)` (below) resolves path, chunk and $R$ and returns them with the host arrays; `unstructured` and `unstructured_memory` both call only `layout`, and their keyword arguments match.
- Host checks first, as `assemble`: `phases`, `check_materials`, `check_elements(mesh, mass)`, `nodal_mass` (finite, ≥ 0, `(n_nodes,)`), `chunk >= 1`, `range_nodes >= 1`.
- `v` must be 2-D `(n_dof, m)` (`m = 1` is `(n_dof, 1)`); a 1-D or wrong-length `v` raises (shapes are static, so this check is trace-time).
- `op.scaled(s_K=None, s_M=None)`: `None` keeps the current value; otherwise the array must have shape `(n_el,)` (checked; traceable, so no value checks and it can be used under `jit`/`grad`). Values are expected finite and ≥ 0; $s = 0$ is allowed (void elements).
- **Stored mode bakes $C$ into $K_e$.** Changing materials in stored mode means building a new operator; in otf mode, `dataclasses.replace(op, C=...)` changes the stiffness without recompiling. A stored-mode op with `C` set raises at trace time (it would be silently ignored).
- **After `rcm`** (or any `renumber`), user arrays follow the new numbering: `s_K[q]`, `s_M[q]`, `nodal_mass[p]`, and a vector $v$ maps as `v.reshape(-1, 3, m)[p]` (new DOF $3n' + i$ is old DOF $3p[n'] + i$).
- No masking here: callers form $m \odot K(m \odot v)$ (PR 7/PR 9).

## The operator pytree (Principle 3)

`Unstructured` is a frozen dataclass registered with `jax.tree_util.register_dataclass`; `Block` likewise. Leaves (all `jit` arguments, nothing closed over); a leaf the mode or path does not use is `None` (an empty subtree), so no unused array is stored or can be changed with no effect:

- per block (`blocks`: a tuple of `Block`s, one per mesh block, in `TYPES` order): `dofs` (n_pad, k) int32; `phase` (n_pad,) int32; stored mode: `Ke` (n_pad, k(k+1)/2) packed upper triangle and `wdet` (n_pad, nq); otf mode: `geo` (n_pad, nq, 10) = invJ (9) and w·detJ (1) per quadrature point (PR 3's `jacobian`); gather path: `starts` (n_ranges,) int32 window starts and `inv` (n_ranges, 3R, valence) int32;
- `C` (P, 6, 6): otf only (`None` in stored mode); `rho` (P,) from `phase_table` (every mass uses it);
- `s_K`, `s_M` (n_el,) in global element order, default ones; `nodal` (n_dof,) = `repeat(nodal_mass, 3)` (zeros if `None`).

Static metadata: per block type, element offset, `n_el`, chunk, window width $W_b$; global `n_dof`, `mass`, `mode`, `path`, $R$, `n_ranges`. Replacing `s_K`, `s_M`, `rho`, `nodal` (and `C` in otf mode) therefore never recompiles (tested).

- Each block is padded to `n_pad` elements, a multiple of its chunk and $\ge W_b$, by repeating its first element; padded elements get $s = 0$ inside the apply (`jnp.pad` of the block's slice of `s_K`/`s_M`), so they contribute nothing to applies, diagonals and lumped $d$.
- **Build on the device.** Each block's packed `Ke`, `wdet` and `geo` are built by one jitted `lax.map` over chunks of exactly the block's chunk on the host-padded connectivity, so the stacked output is the padded leaf itself: no slice, no host round-trip (a host fill or a sliced output doubles the peak; measured above). The build peak is the leaves plus one chunk of PR 3's kernel temporaries.
- int32: `check_int32` on `n_dof`, every `<type>_dofs` $= n_\text{pad} k$, the phase index, and on the gather path the inverse map size and the window index range $\sum_b W_b k_b + 1$, before casting (the phase index cast is the one `docs/materials.md` assigns to PR 5).

## Element kernels (per chunk, block-native)

$u_e = v[\text{dofs}_e]$ is $(c, k, m)$, $k = 3n$. Stored $K_e$ comes from PR 3's `kernels.stiffness` (the same kernel as the oracle; PR 3 validated it, and PR 5 tests packing, scaling and assembly); otf is the independent check of the stiffness.

- **Stiffness, stored:** $y_e = s_{K,e}\,K_e u_e$ with $K_e$ unpacked from the upper triangle by one index gather per chunk.
- **Stiffness, otf:** with $G_q = J_q^{-\top}\nabla_\xi N$ from invJ, $\varepsilon_q = B_q u_e$, $\sigma_q = C_e \varepsilon_q$, $y_e = s_{K,e}\sum_q w_q \det J_q\, B_q^\top \sigma_q$ (the same $B$ selector as PR 3), never forming $K_e$.
- **Consistent mass** (both modes, from w·detJ, the scalar $n \times n$ form of `docs/elements.md`): $y_{e,ai} = \rho_e s_{M,e} \sum_q w_q\det J_q\, N_{qa} \sum_b N_{qb} u_{e,bi}$.
- **Lumped mass:** a diagonal vector $d$ assembled from the element lumped diagonals, then $y = d \odot v$; recomputed per call (it depends on $s_M$ and $\rho$):
  - row-sum (hex8/hex27): $d_a = \rho_e s_{M,e} \sum_q w_q\det J_q\, N_{qa}$;
  - HRZ (hex20): $d_a = \rho_e s_{M,e} V_e\, \dfrac{\sum_q w_q\det J_q N_{qa}^2}{\sum_b \sum_q w_q\det J_q N_{qb}^2}$ with $V_e = \sum_q w_q \det J_q$.
- **Diagonals:** `diag_K` assembles $s_{K,e}\,\mathrm{diag}(K_e)$ (stored: the packed diagonal; otf: $\sum_q w\det J\, b_{qai}^\top C_e b_{qai}$ with $b_{qai}$ the $B$ column of DOF $(a, i)$); `diag_M` assembles the consistent diagonal $\rho s_M \sum_q w\det J N_{qa}^2$ (×3) or the lumped $d$. Both add `nodal` for M.
- Diagonals and lumped $d$ use the operator's own assembly path: their kernels return $(c, k, 1)$ and go through the same scatter or gather as the applies (one code path, and the gather layout is exercised by them too).
- Nodal masses: `apply_M` adds `nodal[:, None] * v`.

## Assembly paths

Both take a per-block element kernel `f(block data slice, u_e) -> y_e` and return the global $(n_\text{dof}, m)$ result.

- **Scatter:** per block, `lax.scan` over chunks with the output as the carry (updated in place by XLA); `out.at[dofs].add(y)`. `v` is not donated (callers reuse it).
- **Gather** (roadmap: chunked by node range):
  - Nodes are split into $n_\text{ranges} = \lceil n_\text{nodes}/R\rceil$ ranges of exactly $R$ nodes ($R \le n_\text{nodes}$) starting at $r_0 = \min(rR,\ n_\text{nodes} - R)$, so the last range overlaps its predecessor instead of being padded ($R = \min(\text{range_nodes}, n_\text{nodes})$; $R$ need not divide $n_\text{nodes}$; the overlap is recomputed and overwritten with the same values up to rounding; the later range wins).
  - **$R$ per mesh** (`range_nodes=None`): `layout` scores the candidates $\{2^j < n_\text{nodes} : 14 \le j \le 20\} \cup \{n_\text{nodes}\}$ (scoring computes only $W_b$ and the valence; the inverse map is built only for the chosen $R$) and takes the largest whose per-column gather workspace (window buffers + gathered rows) is at most two vectors, $2 \cdot 8\,n_\text{dof}$ bytes. A per-type constant does not scale (the RCM front grows with the mesh). If none meets the cap, the smallest candidate is used (the estimator reports the cost); tiny meshes thus get one range. The CPU benchmark times $R$ and its neighbours for information only; the rule is re-tuned in the deferred GPU check.
  - **Windows, independent of element order.** Per block, `start[r] = min e` and `end[r] = max e` over elements with a node in $[r_0, r_0 + R)$ (`np.minimum.at` / `np.maximum.at` over the (element, local node) pairs with the node's range; the overlapping last range is handled the same way with its own mask). $W_b = \max_r(\text{end} - \text{start} + 1)$ rounded up to the block's chunk; a range no element of the block touches gets `start = 0` (its window is never referenced). Starts are clipped into $[0, n_{\text{pad},b} - W_b]$ and the host asserts `start + W_b <= n_pad_b`. So every element touching range $r$ is in its window for any element order; a bad order only widens $W_b$ (the estimator reports it).
  - **Node-local numbering:** the inverse map is indexed by range-local DOF ($0 \le j < 3R$) and lists, padded to the max valence with a sentinel, the flat window-local positions $(\text{block offset} + e_\text{local} k + 3a + i)$ of every element contribution to that DOF. So its entries are bounded by the window size, not by $n_\text{el}k$.
  - `lax.scan` over ranges with the $(n_\text{dof}, m)$ output as the carry: per block, the window's element contributions $Y_b$ $(W_b k_b, m)$ are computed by an inner `scan` over chunks that slices each chunk straight from the full block leaves (`dynamic_slice(Ke, start + i c, c)`; slicing the window first copies its $K_e$, measured 69 → 655 MB of temporaries); the windows are concatenated with one zero row (the sentinel, fused by XLA) and the range result `Y[inv_r].sum(1)` $(3R, m)$ is written with `dynamic_update_slice` at row $3r_0$. No padded stacked output and no slice copy.
- Layouts (padding, windows, inverse maps) are built on the host in NumPy by one function `layout(mesh, path, chunk, range_nodes)`, shared with the estimator.
- Dynamic start indices (`dynamic_slice` at `start + i c`, `dynamic_update_slice` at $3 r_0$) are int32: the scans run over `jnp.arange(n, dtype=jnp.int32)` (under x64 a plain counter is int64), and the host asserts guarantee no slice clamps.
- Cost note for docs: with several blocks, every range runs each block's full window $W_b$, even blocks that do not touch it (`start = 0`); acceptable for the minimal GPU-candidate path.

## Reordering (`mesh/reorder.py`)

```python
mesh2 = renumber(mesh, p, q)  # p: (n_nodes,) new -> old nodes; q: (n_el,) new -> old global elements
mesh2, p, q = rcm(mesh)
```

- `renumber` applies `docs/mesh.md`'s renumbering rules: points, connectivity, node sets, element sets, labels, voxel maps (so a voxel mesh stays valid). `q` must keep each block's elements within the block (blocks keep `TYPES` order); it raises otherwise. PR 18's Morton ordering reuses it.
- `rcm`: node graph $A = E^\top E$ from the sparse element–node incidence $E$ (all blocks), `scipy.sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)` (handles disconnected meshes); then elements within each block stably sorted by their min (new) node index; `renumber`.
- Docs: RCM is for arbitrarily ordered meshes (Gmsh/Abaqus); `box_mesh` and `voxel_mesh` grid order is already better (see the measured facts). Neither the operator nor the gather path requires it.

## `assemble` gains $s_K$, $s_M$

`assemble(..., s_K=None, s_M=None)`: finite, non-negative `(n_el,)` arrays in global element order (zeros allowed); element $e$'s $K_e$ (resp. consistent $M_e$ or lumped diagonal) is multiplied by $s_{K,e}$ (resp. $s_{M,e}$) before summation. Nodal masses are not scaled. `None` means ones (bit-identical to PR 4).

## Memory estimator (`memory.py`)

```python
unstructured_memory(mesh, m, mode="stored", mass="consistent", path=None, chunk=None, range_nodes=None)
    -> dict(storage=..., vectors=..., workspace=..., build=..., overhead=..., total=...)   # bytes
```

Called on the mesh that will be used (after `rcm`, if any), with the same keyword arguments as `unstructured`.

- `storage`: every operator leaf (sizes from the shared host layout: padded counts, $W_b$, valence).
- `vectors`: $v$ and the output, $2 \cdot 8\,n_\text{dof} m$.
- `workspace`: the apply temporaries. Scatter: per chunk $u_e$, unpacked $K_e$, $y_e$ (otf: about $3 c\, n_q \cdot 9\, m$ floats). Gather: the window buffers $\sum_b W_b k_b m$, the gathered range rows $3R \cdot \text{valence} \cdot m$ (XLA does not fuse the gather with the sum), and the per-chunk temporaries.
- `build`: the build transient: the host layout (int64 dofs, windows and inverse map before the int32 cast) and one chunk of PR 3's kernel temporaries (the device build writes the leaves directly).
- `overhead`: the fixed JAX runtime overhead.
- `total = overhead + storage + max(build, vectors + workspace)`.
- Composition for later PRs (PR 9/11/13/14/15): add PR 5's `storage + workspace` to their own estimates (they count the vectors themselves) and count `overhead` once.
- Calibration: `overhead` and the per-chunk temporary factor (XLA temporaries per unpacked $K_e$ byte in the scatter workspace) are measured once on hex20 $30^3$ and hex8 $60^3$ (stated in `docs/operators.md`), then validated at $55^3$ (test 11).

## Benchmark (`benchmarks/scatter_gather.py`)

hex8/hex20/hex27 box meshes at ≈ 1M DOF, in natural grid order and after `rcm`, stored and otf, m ∈ {1, 16}, chunks around the 32 MB rule (×¼ … ×4), and the per-mesh $R$ plus its neighbours. Each configuration runs through `harness.measure` (fresh subprocess) and records the apply time (min of repeats), `peak_bytes`, and the compile time of `jax.jit(apply_K).lower(...).compile()`. Writes `benchmarks/results/scatter_gather.json` (committed, with `versions()` and a pointer to `benchmarks/results/system.json` for STREAM); `main(sizes=...)` takes a size override for the smoke test. It picks `DEFAULT_PATH["cpu"]` and checks the chunk rule (within 10% of the best measured time; recorded in the JSON and in `docs/operators.md`). The GPU run is a deferred check.

## Tests

Tolerances per Principle 1 ($\|a - b\|_\infty \le \text{tol}\,\|b\|_\infty$ over the whole result block). Fixed seeds.

`tests/test_reorder.py`:

1. **renumber and RCM** on (a) a randomly node- and element-shuffled mixed hex8/hex20 mesh with node sets and element sets spanning both blocks (including an empty set) and non-constant labels, and (b) a masked voxel mesh with two disconnected islands:
   - `p`, `q` are permutations and `q` maps each block's range onto itself;
   - connectivity, labels and voxel maps follow the renumbering rules;
   - geometric check independent of the bookkeeping: per element set the multiset of element centroids, and per node set the multiset of coordinates, are unchanged;
   - elements sorted by min node within each block;
   - the node permutation equals SciPy's RCM on the same graph (catches applying $p^{-1}$);
   - `check_elements` and the `Mesh` constructor pass; `renumber` with a block-crossing `q` raises.

`tests/test_matfree.py`. Meshes:
- **A:** the PR 4 mixed hex8 + hex20 glued mesh, distorted, two materials and two orientations mixed, RCM-reordered;
- **B:** a distorted hex27 $3^3$ box, RCM-reordered;
- **C:** mesh A without `rcm`, after a random node and element shuffle (asserts that the max node per element is not monotone in any block, so the window rule is exercised);
- **D:** a masked voxel mesh with an asymmetric mask (irregular, RCM-reordered).

Random $s_K, s_M \in [0.1, 2]$ with one $s = 0$, random nodal masses with zeros. Sizes: `chunk=5` (pads both 8-element blocks and the 27-element block) and `range_nodes=23` (several ranges; $R \nmid n_\text{nodes}$, so the last range overlaps), plus one case with `chunk=None` and `range_nodes >= n_nodes` (one chunk, one range). Each op is built once per (mesh, mode, path, mass) in a module-scoped cache; m and the methods loop inside. Shared helpers (`mixed_mesh`, `distort`, `dense_reference` with $s_K$, $s_M$) move to `tests/helpers.py`.

2. **assemble with scalings:** $K(s_K)$, $M(s_M)$ (consistent and lumped) equal `dense_reference` with per-element scaling to 1e-14; `None` is bit-identical to ones; nodal masses unscaled; wrong shape, negative or non-finite scalings raise.
3. **apply vs SciPy:** for mode ∈ {stored, otf} × path ∈ {scatter, gather} × mass ∈ {consistent, lumped} × m ∈ {1, 5} on meshes A, B, D (and C for the gather path): `apply_K(v)`, `apply_M(v)` equal `K @ v`, `M @ v` from the scaled PR 4 `assemble` on the same mesh to 1e-12. **Permuted:** on mesh A, with `s_K[q]`, `s_M[q]`, `nodal_mass[p]` and $v$ permuted by $p$, the reordered result permuted back equals the SciPy product on the original mesh.
4. **Diagonals:** `diag_K`, `diag_M` equal `K.diagonal()`, `M.diagonal()` to 1e-12 for every mode, path and mass, on the same meshes, sizes, scalings and nodal masses.
5. **No recompile:** for `f` in apply_K, apply_M, `exe = jax.jit(f).lower(op, v).compile()`, then `exe` is called with a modified op (same treedef and static fields) and matches the new SciPy product: apply_K with new `s_K` (both modes) and new `C` (otf only; stored mode has no `C` leaf); apply_M with new `s_M`, `rho`, `nodal`, for consistent and lumped (catches a cached lumped $d$).
6. **int32:** a monkeypatched `check_int32` records `n_dof`, every `<type>_dofs`, `phase`, and (gather) the `inv` and `window` sizes; index leaves are int32.
7. **Inputs:** unknown `mode`/`mass`/`path`, `chunk=0`, `range_nodes=0`, bad `nodal_mass`, wrong-shape `scaled` arrays raise; `check_elements`/`check_materials` errors propagate; a 1-D `v` and a `v` of wrong leading size raise; unused leaves are `None` per mode and path; a stored-mode op with `C` replaced raises; `range_nodes > n_nodes` is clamped.
8. **Gather layout** (meshes A–D): the inverse map is rebuilt independently from all (element, local DOF) pairs: each non-sentinel entry decodes to (block, element, $a$, $i$) with `dofs[e, 3a+i] == 3 r_0 + j`, every pair appears exactly once, no padded element is referenced, windows lie inside the padded block, and some range has an empty window for some block (asserted to occur on mesh A); with `range_nodes=None` the chosen $R$ follows exactly this rule (largest candidate within the cap, else the smallest).
9. **Gradient:** `jax.grad` of $w^\top K(s_K)\, v$ with respect to $s_K$ equals the per-element $w_e^\top K_e v_e$ from SciPy element matrices (stored and otf, scatter and gather), to 1e-12.
10. **Compile time** (slow): in a fresh subprocess per size with the persistent cache disabled, min of 5 `lower().compile()` calls, m = 1, scatter and gather, stored and otf: time the compile of `apply_K` and `apply_M` together on `ShapeDtypeStruct`s for a real hex20 $22^3$ op (10,648 elements) and for 1M elements, the latter built by `jax.tree.map` to `ShapeDtypeStruct`s with `n_pad` scaled and `dataclasses.replace` on the static `n_el`, `n_ranges` ($\lceil 4 n_\text{el}/R \rceil$), keeping $W_b$ and the valence. Ratio ≤ 1.5.
11. **Estimator** (slow, darwin only since `peak_bytes` is None elsewhere): stored Ke, hex20 $55^3$ (2.08M DOF), scatter and gather explicitly, m ∈ {1, 16}, via `harness.measure`. **setup:** imports, `box_mesh`, materials, and compile-only warm-ups (`.lower().compile()` of the Ke build kernel and of `apply_K`/`apply_M` on `ShapeDtypeStruct`s from `layout`). **code:** `op = unstructured(...)`, `v = jnp.ones((n_dof, m))`, `op.apply_K(v).block_until_ready()`, `op.apply_M(v).block_until_ready()`. Pass: $|\text{est} - \text{peak}| \le 0.2\,\text{peak}$ with `est = total`, and `est >= 2e9`. Because $K_e$ is 85–95% of the total, a second measurement at m = 16 builds the op in **setup** and measures only `apply_K` + `apply_M`, against `workspace + 8 n_dof m` (the output) within 20% (measured: scatter 0.31 GB, gather 1.36 GB). And `jax.jit(apply_K).lower(op, v).compile().memory_analysis().temp_size_in_bytes` is within 10% of `workspace` for both paths (catches hidden window or padding copies).
12. **Benchmark defaults:** a fast test loads `benchmarks/results/scatter_gather.json` and asserts that `DEFAULT_PATH["cpu"]` equals the recorded choice and that the recorded chunk rule check passed; a tiny-size smoke run of `benchmarks/scatter_gather.main` (no file written).

## Docs

- `docs/operators.md`:
  - line 3: the matrix-free operators live here (no longer "will");
  - DOF numbering: the flattened size checked is $n_\text{pad} k$; the gather sizes join the `check_int32` list;
  - `assemble`: signature gains `s_K=None, s_M=None`; steps 2 and 4 scale per element before summation; Validation gains the scaled-assembly test;
  - new sections: the matrix-free operator (API and method interface, pytree leaves and static fields, stored-mode rebuild rule, user arrays after renumbering), the kernel formulas, scatter and gather (windows, overlapping last range, per-mesh $R$, node-local numbering, padding, carried output), the device build, the benchmark table and the chosen defaults, the estimator (keys, composition rule, calibration) and its validation, cost per apply; the inverse-map padding note (one high-valence node inflates the map and the rows buffer).
- `docs/mesh.md`: `renumber`, `rcm`, when RCM helps, and its host cost at 10M DOF.
- `docs/elements.md` line 83: the mass apply uses `wdet` and $N$ on the fly (no stored $n \times n$ $m$).
- `docs/conventions.md` line 20: PR 5's gather `inv`/`starts` join the `check_int32` users.
- `docs/ROADMAP.md`: status line; the "Scatter vs gather" key decision ("XLA:CPU scatter is effectively serial") is replaced by the measured CPU outcome (kernel-bound; scatter wins) and the chosen default (pointer to `docs/operators.md`); the PR 18 roofline note "(serial XLA:CPU scatter)" is corrected to the measurement; the key-decision sub-bullet "The gather runs on an RCM-ordered mesh" is reworded (the gather works for any order; RCM or grid order keeps windows narrow); the memory-budget row "Gather path" gets the window cost; PR 6 is noted to provide the same method interface (`stencil`, `Stencil`, `stencil_memory`); the deferred GPU scatter-vs-gather check stays.
- README: status, and a snippet (box → `unstructured` → `op.apply_K`), executed by `tests/test_readme.py`.

## Not covered

- Masking, rigid modes (PR 7); voxel stencil (PR 6); solid-shell (PR 16); Morton ordering (PR 18).
- fp32 operators (never: iteration vectors are fp64).
