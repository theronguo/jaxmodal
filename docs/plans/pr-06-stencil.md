# PR 6 plan: voxel stencil operator

Branch `pr-06-stencil`, from `main` after PR 5. Scope: the roadmap's PR 6 (Contents, Docs, Tests): a matrix-free operator for `voxel_mesh` meshes that applies one reference hex8 $K_e$ (and $M_e$) per phase over slabs of the grid, with PR 5's interface (`apply_K`, `apply_M`, `diag_K`, `diag_M`, `scaled`, `n_dof`, `n_el`), per-element $s_K$, $s_M$, nodal masses, occupancy masks, and an estimator `stencil_memory`.

## Measured facts (prototype, scratchpad `p6/proto2.py`, `r6-perf/`; jax 0.11.2, fp64, M3 Max CPU, warm, min of 5; full $87^3$ voxel box, 2.04M DOF; re-measured on a quiet machine in plan review)

- **Correctness:** the slab stencil equals the scaled `assemble` product and PR 5's `unstructured` to 1e-15 relative on full and masked grids with anisotropic spacing, after `rcm`, with padded slabs; `jax.grad` with respect to $s$ and $v$ through the fill/drop gathers to 2e-15.
- **Layout matters at m > 1.** Per slab: gather the slab's nodes, 8 corner slices, contract, 8 shifted adds, scatter back.

  | variant | m = 1 | m = 16 | temp at m = 16 |
  |---|---|---|---|
  | node-major `einsum("ab,zyxbm->zyxam")` (batched 24×24 @ 24×m) | 0.020 s | 0.36–0.53 s | 584 MB |
  | corner-DOF-major: one GEMM $(24, 24) @ (24, N_\text{slab} m)$, $v$ and output padded with a zero row | 0.018 s | 0.16–0.20 s | 558–1274 MB |
  | the same GEMM with `.at[].get(mode="fill")` / `.at[].add(mode="drop")` instead of the padding (**chosen**) | 0.016 s | 0.15–0.18 s | 98–192 MB |
  | full box only: grid vector + `dynamic_slice`, no gather | 0.024 s | 0.31 s | 590 MB |
  | other tries: fused corner sum; component/m-major layout; elementwise, no GEMM | ≥ 0.018 s | ≥ 0.18 s | — |

  The padded copies of $v$ and of the output cost two full vectors of temporaries; fill/drop removes them. The gather through the grid map costs ≈ 9 ms of 157 ms (m = 16); the rest is split between the corner stack (51 ms), the GEMM (44 ms, ≈ 280 GFLOP/s), the shifted adds (53 ms) and the scatter (15 ms). One code path serves full and masked grids, with no full-grid vector anywhere (the output is the only $n_\text{dof}$-sized buffer).
- **Slab size** (cells per slab, fill variant):

  | cells/slab | 7.6k | 15k | 30k | 61k | 121k |
  |---|---|---|---|---|---|
  | m = 1 | 0.0208 s | 0.0164 s | 0.0161 s | 0.0143 s | 0.0140 s |
  | m = 16 | 0.180 s | 0.167 s | 0.157 s | 0.162 s | 0.176 s |
  | temp m = 16 | 52 MB | 98 MB | 192 MB | 378 MB | 751 MB |

  Rule: $2^{16}$ cells per slab (61k here): within 3% of the best for both $m$.
- **Phases:** one GEMM per phase. Summing the phase products in a Python loop keeps every product alive (+93 MB per phase at m = 16: 187, 280, 466, 838 MB for P = 1, 2, 4, 8); a `lax.scan` over phases with one accumulator keeps the workspace at 280 MB for every P ≥ 2, at 15–30% more time (0.157, 0.267, 0.416, 0.712 s). A stacked $(24P, 24)$ GEMM is slower (0.46–1.39 s). Time is ∝ P either way.
- **Workspace** is exactly affine in cells·m: slope 48 floats per cell per column ($U$ and $Y$) for one phase, 72 with the phase accumulator; the intercept is the per-slab gathers.
- vs PR 5 (hex8 $69^3$, 1M DOF: 0.042 / 0.126 s at m = 1 / 16): per DOF the stencil is ≈ 5× faster at m = 1 and ≈ 1.6× at m = 16.

## Files

```
src/jaxmodal/operators/common.py       register, check_x64, check_nodal_mass, check_v, scaled (shared by PR 5 and PR 6)
src/jaxmodal/operators/voxel.py        Stencil (pytree with methods), stencil, reference_element, slab loop (module named like matfree.py, so `operators.stencil` stays the function)
src/jaxmodal/operators/matfree.py      uses common.py (no behaviour change)
src/jaxmodal/operators/assembled.py    uses check_nodal_mass
src/jaxmodal/operators/__init__.py     + Stencil, stencil, reference_element
src/jaxmodal/memory.py                 + stencil_memory
benchmarks/stencil_slab.py             slab-rule benchmark -> benchmarks/results/stencil_slab.json
tests/test_stencil.py
docs/operators.md, docs/mesh.md, docs/ROADMAP.md, README.md
```

## Shared helpers (`operators/common.py`)

Plain functions, now needed by both operators (CLAUDE.md: abstract once needed twice):

- `register(cls, meta)`: PR 5's `_register` (dataclass pytree with static `meta` fields).
- `check_x64(name)`: raises unless `jax_enable_x64` (iteration vectors are fp64).
- `check_nodal_mass(nodal_mass, n_nodes) -> (n_nodes,) float array`: finite, ≥ 0, shape; `None` → zeros. Used by `assemble`, `unstructured`, `stencil`.
- `check_v(v, n_dof)`: 2-D with `n_dof` rows (trace-time).
- `scaled(op, s_K, s_M)`: PR 5's `Unstructured.scaled` body (shape check against `op.n_el`, `dataclasses.replace`). Each class keeps a one-line `scaled` method.

PR 5's tests must pass unchanged after the move.

## API

```python
from jaxmodal.operators import reference_element, stencil

op = stencil(mesh, materials, orientations=None, mass="consistent", nodal_mass=None, slab=None)
op = op.scaled(s_K=s_K, s_M=s_M)  # (n_el,), default ones
y = op.apply_K(v)  # (n_dof, m) -> (n_dof, m)
y = op.apply_M(v)
dK, dM = op.diag_K(), op.diag_M()
Ke, me = reference_element(op)  # (P, 24, 24) per phase, (8, 8) scalar mass per unit density
```

- Same interface and conventions as PR 5 (`docs/operators.md`): methods call module-level jitted functions of `(op, v)`; `v` is 2-D `(n_dof, m)`; `scaled` checks shapes only and is traceable; `s = 0` allowed; properties `n_dof` ($3 n_\text{nodes}$) and `n_el`; DOF $3n + i$ in mesh node order, so results compare directly with `unstructured` and `assemble` on the same mesh. The duck-typed solver interface (PR 7/9) is the five methods plus `n_dof`, stated in `docs/operators.md`.
- `mesh` must be a voxel mesh (`mesh.voxel` set; `Mesh` guarantees one hex8 block). Works after `renumber`/`rcm` (the voxel maps follow the renumbering), although grid order is already the natural one.
- `mass`: `"consistent"` or `"lumped"` (row-sum; on a box element row-sum and HRZ coincide: $\rho V/8$ per node).
- `slab`: cell layers along the scan axis per scan step; `None` = $\max(1, \lfloor 2^{16}/A\rfloor)$ with $A$ the cells per layer, capped at the layer count. `slab >= 1` checked. (PR 5's `chunk` counts elements; here slab·A ≈ 64k cells.)
- **Scan axis:** the longest grid axis (ties: the last, z), so a layer is the smallest cross-section. A flat $1000 \times 1000 \times 33$ plate scans along x with 33k-cell layers instead of 1M-cell z-layers (≈ 3 GB of workspace at m = 8). The grids are transposed on the host and the corner shifts permuted with them; $K_e$ keeps VTK order (DOF components are not permuted). The slab cannot go below one layer; the estimator reports the workspace of very large cross-sections.
- Host checks: `check_x64`, `phases`, `check_materials`, `check_nodal_mass`, the voxel geometry check (below). No `check_elements` call: voxel cells are boxes with positive spacing (`check_grid`), and the geometry check confirms the mesh is that grid.
- **Restriction** (roadmap): one reference $K_e$ per phase; elements differ only by phase and $s_K$, $s_M$. Points must be the grid points (a distorted voxel mesh raises; use `unstructured`). Cost is ∝ P (one GEMM per phase): with many phases (e.g. one orientation per element) use `unstructured`.

## Geometry check (`docs/mesh.md`: "PR 6 checks that they agree with the geometry")

In NumPy, before building, with $g$ the grid node index and $c$ the grid cell index:

- every used grid node: `points[node_map[g]]` equals `origin + spacing * (i, j, k)` to $10^{-12}\max_d(|o_d| + h_d n_d)$ (absolute);
- every active cell: `conn[element_map[c]]` equals `node_map[g(c) + corner offsets]` in VTK order (exact), with the integer offsets $(\text{CORNERS} + 1)/2 \in \{0, 1\}^3$ from `model.CORNERS`.

Either failure raises `ValueError("mesh does not match its voxel grid: ...")`. `Mesh` already validates the maps as bijections, so these two checks make the grid description equivalent to the mesh.

## The operator pytree

Frozen dataclass `Stencil`, registered with `common.register`. Leaves:

- `node` `(L + 1, B + 1, A' + 1)` int32 for the transposed grid ($L$ padded layers along the scan axis): mesh node of each grid node; dropped nodes and padding layers hold `n_nodes` (out of range);
- `cell` `(L, B, A')` int32: mesh element of each grid cell; inactive cells and padding hold `n_el`;
- `phase` `(n_el,)` int32; `C` `(P, 6, 6)`, `rho` `(P,)` from `phase_table`;
- `spacing` `(3,)` float64 (geometry is a jit argument, Principle 3);
- `s_K`, `s_M` `(n_el,)` default ones; `nodal` `(n_dof,)` (as PR 5).

Static: `slab`, `axes` (the permutation from grid axes (x, y, z) to stored order, scan axis first), `layers` (the unpadded layer count), `mass`, `n_nodes`, `n_el`. $L$ is the smallest multiple of `slab` ≥ `layers`; padded layers are inactive. The 8 corner shifts are derived at trace time from the one $\{0,1\}$ table $(\text{CORNERS} + 1)/2$ permuted by `axes`; the phase count is `rho.shape[0]` (no duplicate static). Sentinels are `n_nodes`/`n_el`, not 0, because fill and drop act only on out-of-range ids (a padding id of 0 would be harmless only because $s = 0$ there).

- **`reference_element(op)`** (module-level, traceable): $K_e^p$ = `kernels.stiffness("hex8", X, C)` and $m_e = \sum_q w_q\det J_q N_q N_q^\top$ from `jacobian` on the reference cell $X = \text{spacing} \odot (\text{CORNERS} + 1)/2$ (VTK order, the same $\{0,1\}$ table as the shifts and the geometry check). The kernels call it on each apply (a few µs), so `C`, `rho` and `spacing` are ordinary leaves: changing them never rebuilds or recompiles, and gradients with respect to them exist. PR 11 (Galerkin coarse operators) and PR 23 (voxel sensitivities) build on `reference_element` and the `node`/`cell` grids, documented in `docs/operators.md`.
- int32: `check_int32(n_dof=..., node=node.size, cell=cell.size, phase=P)` before casting.

## Slab loop

One function `_slabs(op, kern, m)` (the counterpart of PR 5's `_assemble`):

```
out = zeros((n_nodes, 3, m))
scan over j in arange(L // slab, int32), carry out:
    nd = dynamic_slice(node, j*slab, slab+1)                 # node ids, one-layer halo
    el = dynamic_slice(cell, j*slab, slab)                   # element ids
    Ye = kern(nd, el)                                        # (8, 3, slab, B, A', m), corner-DOF-major
    y  = sum_a shift_a(Ye[a])                                # 8 shifted adds into (3, slab+1, B+1, A'+1, m)
    out = out.at[nd].add(moveaxis(y, 0, -2), mode="drop")    # dropped nodes / padding fall out
return out.reshape(n_dof, m)
```

The halo layer is shared by consecutive slabs; the scatter-add sums both contributions. Every gather passes `fill_value` explicitly (JAX's default fill is NaN for floats, which would leak through $0 \cdot \text{NaN}$): $u$ and the per-element $s$ with 0, `phase` with 0 (used only in the masks $[\text{phase} = p]$, multiplied by $s = 0$ on inactive cells). So inactive and padded cells contribute exactly 0.

Kernels ($U$ is $(24, N)$, $N = \text{slab}\cdot B A' m$, corner-DOF-major):

- $u$: `v.reshape(n_nodes, 3, m).at[nd].get(mode="fill", fill_value=0)`, moved to $(3, \ldots, m)$; the 8 shifted corner slices stacked into $U$.
- **apply_K:** $Y = \sum_p w_p \odot (K_e^p U)$, $w_p = s_K[\text{el}]\,[\text{phase}[\text{el}] = p]$, accumulated by a `lax.scan` over phases (one accumulator; with P = 1 a single GEMM, no scan).
- **apply_M, consistent:** $Y = \rho_{p(e)} s_M\, (m_e \otimes I_3) U$, computed as $m_e$ (8×8) @ $U$ reshaped $(8, 3N)$; plus `nodal * v`.
- **diag_M / lumped:** the nodal vector $d = \sum_e \rho_e s_{M,e} d^\text{ref}$ through the same loop with $m = 1$ (lumped: $d^\text{ref}_a = \sum_b (m_e)_{ab} = V/8$; consistent `diag_M`: $(m_e)_{aa}$), plus `nodal`; lumped `apply_M` = $(d + \text{nodal}) \odot v$, recomputed per call.
- **diag_K:** $\sum_p w_p \operatorname{diag}(K_e^p)$ per corner DOF through the loop with $m = 1$.

All loops have static trip counts (conventions; Principle 6), so the operator is differentiable: `jax.grad` with respect to `s_K`, `s_M`, `C`, `rho`, `spacing` flows through the fill/drop gathers (the VJP of a fill gather is a dropping scatter and vice versa; verified in the prototype).

## Memory estimator (`memory.py`)

```python
stencil_memory(mesh, m, mass="consistent", slab=None)
    -> dict(storage=..., vectors=..., workspace=..., build=..., overhead=..., total=...)
```

Same keys and composition rule as `unstructured_memory`:

- `storage`: the leaves (`node`, `cell`, `phase` int32; `s_K`, `s_M`; `nodal` fp64 per DOF); exact (tested against the leaf bytes minus `C`, `rho`, `spacing`).
- `vectors`: $3 \cdot 8\,n_\text{dof} m$, as PR 5.
- `workspace`: per slab with $c$ = slab·A cells, $8c\,(24 k_P m + g)$ with $k_P = 2$ for one phase and 3 for P ≥ 2 ($U$, $Y$, the phase accumulator) and $g$ the per-slab gathers; derived, then checked against `memory_analysis` within 10% for $m \in \{1, 4, 16, 24\}$ and P ∈ {1, 2} (test 9). apply_M is below apply_K.
- `build`: the host grid maps (int64 before the cast).
- `overhead`: PR 5's 10 MB.
- `total = overhead + storage + max(build, vectors + workspace)`.

At 100M voxel DOF (a ≈ $322^3$ cube): storage ≈ 1.7 GB (index grids and `phase` 0.4 GB, $s_K$ + $s_M$ 0.53 GB, `nodal` 0.8 GB), inside the roadmap's 3–4 GB allowance for operator, diagonals, scalings and nodal masses; workspace (one 104k-cell layer per slab) ≈ 0.3 GB (P = 1) or 0.5 GB (P ≥ 2) at m = 8.

## Benchmark (`benchmarks/stencil_slab.py`)

Full $87^3$ box, P = 1, $m \in \{1, 16\}$, slab ∈ {rule/4, rule, 4·rule}, each through `harness.measure` (fresh subprocess, min of 5): writes `benchmarks/results/stencil_slab.json` (with `versions()`, the system pointer, and `slab_rule_ok`: the rule within 10% of the best for each $m$). `main(sizes=...)` takes a size override for the smoke test. The table goes into `docs/operators.md`.

## Tests (`tests/test_stencil.py`)

Tolerances per Principle 1. Every mesh has anisotropic spacing (axis-order and diagonal-layout mutants survive on cubes). Meshes:

- **F:** full voxel box $5 \times 4 \times 3$, spacing $(0.5, 1.0, 0.75)$, origin $(1, -2, 0.5)$ (scan axis x);
- **S:** the PR 1 voxelized sphere (the `test_voxel_sphere` mask formula and spacing $(0.5, 0.4, 0.3)$) on a $10 \times 12 \times 11$ grid (scan axis y, the only y-scan mesh);
- **T:** a $4 \times 5 \times 6$ grid (scan axis z), spacing $(0.5, 1.0, 0.75)$, with a seeded mask; the test asserts that the mask has a dropped grid node in a halo layer for slab 1 and slab 4, an inactive cell next to an active one across a slab boundary, and a corner-only contact. Three variants, labels set with `dataclasses.replace`: two materials (one orthotropic, different $\rho$); one material in two orientations (PR 4's rotation); materials and orientations mixed independently (P = 4);
- **R:** mesh T after `rcm` (node and element order no longer grid order).

Random $s_K, s_M \in [0.1, 2]$ with one zero, nodal masses with zeros. Slab ∈ {1, 4 (∤ 6 on T, ∤ 5 on F: padding), None (one slab)} on every mesh in tests 1 and 2.

1. **apply vs SciPy and PR 5:** `apply_K`, `apply_M` (consistent, lumped), $m \in \{1, 5\}$, equal the scaled PR 4 `assemble` product to 1e-12 and PR 5's `unstructured` on the same mesh to 1e-12, on F, S, T (all three variants), R.
2. **Diagonals:** `diag_K`, `diag_M` (consistent, lumped) equal the SciPy diagonals to 1e-12, same meshes and slabs.
3. **Gradients:** on T, `jax.grad` of $w^\top K v$ with respect to $s_K$, and of $w^\top M v$ with respect to $s_M$ (consistent and lumped), $\rho$ and one entry of $C$, equal central finite differences along one random direction (step 1e-3; linear in $s$, $\rho$, $C$, so to 1e-8); the gradient of $w^\top K v$ and $w^\top M v$ with respect to `spacing` equals central differences (step 1e-5) of the SciPy `assemble` products on the mesh with rescaled points, to 1e-7 (the probe gives 5e-10); the $s_K$ gradient also equals the per-element $w_e^\top K_e v_e$ from SciPy element matrices to 1e-12.
4. **No recompile:** one compiled executable gives the new SciPy products after replacing `s_K`, `s_M`, `C`, `rho`, `spacing`, `nodal`, for consistent and lumped mass (catches a cached lumped $d$); for `spacing` the reference is `assemble` on the mesh with correspondingly rescaled points (the operator then acts as the uniformly rescaled grid; `mesh` itself is not updated).
5. **Geometry check:** `pytest.raises(ValueError, match="voxel grid")` for one point moved by 1e-6; one cell's corners cyclically rotated as `[1, 2, 3, 0, 5, 6, 7, 4]` (same node set, positive det J, wrong VTK order); a non-voxel mesh (`box_mesh`) raises; a renumbered mesh passes (R).
6. **Inputs:** unknown `mass`, `slab=0`, bad `nodal_mass`, x64 off, wrong-shape `scaled` arrays, 1-D or wrong-length `v` raise; `slab` above the layer count is capped.
7. **int32:** a monkeypatched `check_int32` records `n_dof`, `node`, `cell`, `phase`; index leaves are int32.
8. **reference_element and axes:** `reference_element(op)` equals `kernels.stiffness` and the scalar part of `kernels.mass` on one cell of F (both phases on T); `axes` puts x first on F, y on S and z on T, and `layers` equals the grid's extent along it.
9. **Estimator:** `storage` equals the leaf bytes minus `C`, `rho`, `spacing` exactly (F, S, T); (slow) `memory_analysis().temp_size_in_bytes` of `apply_K` and `apply_M` within 10% of `workspace` at $87^3$ for $m \in \{1, 4, 16, 24\}$, P ∈ {1, 2}.
10. **Mac acceptance** (slow, darwin): full $87^3$ voxel box (2.04M DOF), via `harness.measure`: **setup** imports and a warm-up run, freed; **code** builds `op`, allocates six `(n_dof, 24)` fp64 arrays filled with `jax.random.normal` (≈ 2.35 GB; zero pages would not be resident, `docs/conventions.md`) and $v$ `(n_dof, 24)`, runs `apply_K` and `apply_M`, blocks. Estimate: `total` plus the six arrays (≈ 4.1 GB: arrays 2.35, vectors 1.18, workspace 0.56 GB); within 20% of the measured peak, and ≥ 2e9.
11. **Benchmark:** a fast test asserts the recorded `slab_rule_ok` in `benchmarks/results/stencil_slab.json`; a slow smoke run of `main(sizes=...)` on one tiny configuration (no file written).
12. **PR 5 refactor:** PR 5's tests pass unchanged after the move to `common.py`.

Known equivalent mutants: padding layers holding node 0 instead of `n_nodes`, and the scatter with `mode="clip"` instead of `"drop"` (both only add zeros, since $s = 0$ on those cells); shift-permutation mutants on z-scan meshes (identity permutation) and the inverse instead of the forward permutation when it is a swap (its own inverse).

## Docs

- `docs/operators.md`: line 3 (the voxel stencil is here, PR 6); a voxel stencil section: API, restriction (cost ∝ P), geometry check, leaves (including the `node`/`cell` grids with their sentinels and `reference_element`, for PR 11 and PR 23), scan axis and slab rule (one-layer floor), the slab loop and kernels, why gather/GEMM/scatter (no full-grid vectors, one path for full and masked grids, fill/drop saves two vectors of temporaries), the benchmark table, the estimator with 2M and 100M figures, validation; cost per apply at m = 1 (bytes moved: index grids, gathered $v$, scatter; flops: $2 \cdot 24^2$ per cell per phase), as PR 18's roofline starting point; the duck-typed interface is the five methods plus `n_dof`; the grid description for PR 11/PR 23 (`node`/`cell` transposed by `axes`, padded to $L$ layers with sentinels `n_nodes`/`n_el`, `layers`, `reference_element`).
- `docs/mesh.md`: the voxel paragraph's "PR 6 checks that they agree with the geometry" becomes a description of the check (pointer to `docs/operators.md`).
- `docs/ROADMAP.md`: PR 6 status line; Principle 6's "(and voxel z-slabs)" becomes "(and voxel slabs along the longest grid axis)"; the PR 6 Contents line ("K matvec over z-slabs: … 8 shifted corner slices → einsum with the reference Ke → shifted adds into a donated output") reworded to slabs along the longest grid axis and the measured gather/GEMM/scatter design with a pointer to `docs/operators.md`; the memory-budget "Operator" cells confirmed for the operator alone (index grids and `phase`: < 0.1 GB at 2M, ≈ 0.4 GB at 100M); $s_K$, $s_M$ and `nodal` (≈ 1.3 GB at 100M) stay under the 3–4 GB allowance for operator, mask, diagonals, scalings and nodal masses.
- README: status and a short `stencil` snippet (executed by `tests/test_readme.py`).

## Not covered

- Rigid body modes for voxel meshes (PR 7); multigrid (PR 11); the voxel sensitivity pass (PR 23; `jax.grad` through the operator works already; PR 21's element-kernel pass can use `reference_element` too).
- Distorted voxel meshes and hex20/hex27 grids (use `unstructured`); sub-layer chunking of very large cross-sections.

## Implementation notes (code review)

- The static `layers` field was dropped (nothing read it); the layer count is `grid_layout(mesh.voxel.shape)[1]`.
- `apply_M` and `diag_M` seed the scan carry with the nodal term, and the consistent mass uses the 24×24 `kron(me, I3)` GEMM: both keep `apply_M`'s workspace at or below `apply_K`'s.
- `benchmarks/harness.measure_json` replaces the per-benchmark `run()` boilerplate.
