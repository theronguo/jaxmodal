# PR 6 plan: voxel stencil operator

Branch `pr-06-stencil`, from `main` after PR 5. Scope: the roadmap's PR 6 (Contents, Docs, Tests): a matrix-free operator for `voxel_mesh` meshes that applies one reference hex8 $K_e$ (and $M_e$) per phase over z-slabs of the grid, with PR 5's interface (`apply_K`, `apply_M`, `diag_K`, `diag_M`, `scaled`), per-element $s_K$, $s_M$, nodal masses, occupancy masks, and an estimator `stencil_memory`.

## Measured facts (prototype, scratchpad `p6/proto2.py`; jax 0.11.2, fp64, M3 Max CPU, warm, min of 5; full $87^3$ voxel box, 2.04M DOF)

- **Correctness:** the slab stencil equals PR 5's `unstructured(...).scaled(s_K=...).apply_K` to 1e-15 relative on full and 30%-masked grids, anisotropic spacing.
- **Layout matters at m > 1.** Per slab: gather the slab's nodes, 8 corner slices, contract, 8 shifted adds, scatter back.

  | variant | m = 1 | m = 4 | m = 16 | temp at m = 16 |
  |---|---|---|---|---|
  | node-major `einsum("ab,zyxbm->zyxam")` (batched 24×24 @ 24×m) | 0.020 s | 0.080 s | 0.40 s | 584 MB |
  | corner-DOF-major: one GEMM $(24, 24) @ (24, N_\text{slab} m)$ | 0.018 s | 0.052 s | 0.18 s | 584 MB |
  | GEMM + `get(mode="fill")` / `add(mode="drop")` instead of padding $v$ and the output with a zero row | 0.019 s | — | 0.17 s | 98 MB |
  | dense box only: grid vector + `dynamic_slice` (no gather) | 0.016 s | 0.069 s | 0.38 s | 590 MB |

  The padded copies of $v$ and of the output cost two full vectors of temporaries; `mode="fill"`/`"drop"` removes them. Gathering through the grid map costs nothing measurable against slicing a grid vector, so one code path serves full and masked grids (no full-grid vectors at all).
- **Slab size** (cells per slab $= \text{slab}\cdot n_x n_y$, fill variant):

  | cells/slab | 7.6k | 30k | 121k | 484k |
  |---|---|---|---|---|
  | m = 1 | 0.022 s | 0.017 s | 0.014 s | 0.016 s |
  | m = 16 | 0.19 s | 0.15 s | 0.18 s | 0.24 s |
  | temp m = 16 | 52 MB | 192 MB | 751 MB | 2987 MB |

  Rule: `slab = max(1, 2**15 // (nx * ny))` cells layers (≈ 32k cells per slab), within 20% of the best for both $m$ (timings were taken with other jobs on the machine; the implementation re-measures, below).
- vs PR 5 (hex8 $69^3$, 1M DOF: 0.042 / 0.126 s at m = 1 / 16): per DOF the stencil is ≈ 5× faster at m = 1 and ≈ 1.6× at m = 16; its storage is index grids and per-element scalars, < 0.1 GB at 2M DOF.

## Files

```
src/jaxmodal/operators/stencil.py      Stencil (pytree with methods), stencil, grid maps, slab loop
src/jaxmodal/operators/__init__.py     + Stencil, stencil
src/jaxmodal/memory.py                 + stencil_memory
tests/test_stencil.py
docs/operators.md, docs/mesh.md, docs/ROADMAP.md, README.md
```

## API

```python
from jaxmodal.operators import stencil

op = stencil(mesh, materials, orientations=None, mass="consistent", nodal_mass=None, slab=None)
op = op.scaled(s_K=s_K, s_M=s_M)  # (n_el,), default ones
y = op.apply_K(v)  # (n_dof, m) -> (n_dof, m)
y = op.apply_M(v)
dK, dM = op.diag_K(), op.diag_M()
```

- Same interface and conventions as PR 5 (`docs/operators.md`): methods call module-level jitted functions of `(op, v)`; `v` is 2-D `(n_dof, m)`; `scaled` checks shapes only and is traceable; `s = 0` allowed; DOF $3n + i$ in mesh node order, so the results are directly comparable with `unstructured` and `assemble` on the same mesh.
- `mesh` must be a voxel mesh (`mesh.voxel` set, one hex8 block). Works after `renumber`/`rcm` (the voxel maps follow the renumbering), although grid order is already the natural one.
- `mass`: `"consistent"` or `"lumped"` (row-sum; on a box element row-sum and HRZ coincide: $\rho V/8$ per node).
- `slab`: z cell layers per scan step; `None` = $\max(1, \lfloor 2^{15}/(n_x n_y)\rfloor)$, capped at $n_z$. `slab >= 1` checked.
- Host checks: `phases`, `check_materials`, `nodal_mass` (finite, ≥ 0, `(n_nodes,)`), the voxel geometry check (below). No `check_elements` call: voxel cells are boxes with positive spacing (`check_grid`), and the geometry check confirms the mesh is that grid.
- **Restriction** (roadmap): one reference $K_e$ per phase; elements differ only by phase and $s_K$, $s_M$. Points must be the grid points (a distorted voxel mesh raises; use `unstructured`).

## Geometry check (`docs/mesh.md`: "PR 6 checks that they agree with the geometry")

In NumPy, before building, with $g$ the grid node index and $c$ the grid cell index:

- every used grid node: `points[node_map[g]]` equals `origin + spacing * (i, j, k)` to $10^{-12}\max(|\text{origin}| + \text{spacing}\cdot\text{shape})$ (absolute);
- every active cell: `conn[element_map[c]]` equals `node_map[g(c) + corner offsets]` in VTK order (exact).

Either failure raises `ValueError("mesh does not match its voxel grid ...")`. `Mesh` already validates the maps as bijections, so these two checks make the grid description equivalent to the mesh.

## The operator pytree

Frozen dataclass `Stencil`, registered like `Unstructured`. Leaves:

- `node` `(n_z^\text{pad} + 1, n_y + 1, n_x + 1)` int32: mesh node of each grid node; dropped nodes and the padding layers hold `n_nodes` (out of range);
- `cell` `(n_z^\text{pad}, n_y, n_x)` int32: mesh element of each grid cell; inactive cells and padding hold `n_el`;
- `phase` `(n_el,)` int32; `C` `(P, 6, 6)` and `rho` `(P,)` from `phase_table`;
- `s_K`, `s_M` `(n_el,)` default ones; `nodal` `(n_dof,)`.

Static: `spacing` (tuple), `slab`, `mass`, `n_nodes`, `n_el` ($n_z^\text{pad}$ is the smallest multiple of `slab` ≥ $n_z$; padded layers are inactive).

- $K_e^p$ (P, 24, 24) is computed inside each `apply_K`/`diag_K` from `C` by PR 3's `kernels.stiffness` on the reference element $X = \text{spacing}\odot\text{corners}$ (a few µs), so `C` and `rho` are ordinary leaves in every case: changing materials never rebuilds or recompiles, and gradients with respect to `C` exist (unlike PR 5's stored mode). $m_e$ (8×8 scalar mass per unit density) likewise from `jacobian` and $N$.
- int32: `check_int32(n_dof=..., node=node.size, cell=cell.size, phase=P)` before casting.

## Slab loop

One function `_slabs(op, kern, m)` (the counterpart of PR 5's `_assemble`):

```
out = zeros((n_nodes, 3, m))
scan over j in arange(n_slabs, int32), carry out:
    nd = dynamic_slice(node, j*slab, slab+1)              # (slab+1, ny+1, nx+1) node ids, one-layer halo
    el = dynamic_slice(cell, j*slab, slab)                # (slab, ny, nx) element ids
    Ye = kern(nd, el)                                     # (8, 3, slab, ny, nx, m), corner-DOF-major
    y  = sum_a  shift_a(Ye[a])                            # 8 shifted adds into (3, slab+1, ny+1, nx+1, m)
    out = out.at[nd].add(moveaxis(y, 0, 3), mode="drop")  # dropped nodes / padding fall out
return out.reshape(n_dof, m)
```

The halo layer is shared by consecutive slabs; the scatter-add sums both contributions. Per-element values are gathered per slab with `.at[el].get(mode="fill", fill_value=0)`, so inactive and padded cells get $s = 0$ and contribute nothing.

Kernels (corner-DOF-major: $U$ is $(8, 3, N)$ with $N = \text{slab}\,n_y n_x m$):

- $u$: `v.reshape(n_nodes, 3, m).at[nd].get(mode="fill", fill_value=0)`, moved to $(3, \ldots, m)$; the 8 corner slices stacked: $U$ (24, N).
- **apply_K:** $Y = \sum_p w_p \odot (K_e^p\,U)$ with $w_p = s_K[\text{el}]\,[\text{phase}[\text{el}] = p]$; one GEMM per phase.
- **apply_M, consistent:** $Y = \rho_{p(e)} s_M\, (m_e \otimes I_3) U$, computed as $m_e$ (8×8) @ $U$ reshaped (8, 3N).
- **apply_M, lumped / diag_M:** the nodal diagonal $d = \sum_e \rho_e s_{M,e}\, d^\text{ref}$ through the same loop with $m = 1$ ($d^\text{ref}_a = \sum_b (m_e)_{ab}$, all equal to $V/8$; consistent `diag_M` uses $(m_e)_{aa}$); then `apply_M` = $(d + \text{nodal}) \odot v$, and consistent `apply_M` adds `nodal * v`.
- **diag_K:** $\sum_p w_p \operatorname{diag}(K_e^p)$ per corner DOF through the loop with $m = 1$.

All loops have static trip counts (differentiable, Principle 7). `jax.grad` with respect to `s_K`, `s_M`, `C`, `rho` flows through the gathers.

## Memory estimator (`memory.py`)

```python
stencil_memory(mesh, m, mass="consistent", slab=None)
    -> dict(storage=..., vectors=..., workspace=..., build=..., overhead=..., total=...)
```

Same keys and composition rule as `unstructured_memory`:

- `storage`: the leaves (`node`, `cell` int32 grids, `phase`, `s_K`, `s_M`, `nodal`); exact (tested against the leaf bytes).
- `vectors`: $3 \cdot 8\,n_\text{dof} m$, as PR 5.
- `workspace`: per slab, $u$ (gathered and moved), $U$, $Y$, $y$: an explicit formula in cells per slab and $m$, derived from the kernel and checked against `memory_analysis` within 10% (no fitted factor; if XLA's fusion makes it non-linear in $m$, a per-$m$ envelope as PR 5's otf).
- `build`: the host grid maps (int64 before the cast).
- `overhead`: PR 5's 10 MB.
- `total = overhead + storage + max(build, vectors + workspace)`.

## Tests (`tests/test_stencil.py`)

Tolerances per Principle 1. Meshes:

- **F:** full voxel box $5 \times 4 \times 3$, spacing $(0.5, 1.0, 0.75)$, origin $(1, -2, 0.5)$;
- **S:** the PR 1 voxelized sphere (the `test_voxel_sphere` mask) at a small size;
- **T:** a masked $6 \times 5 \times 4$ grid with two phases: two materials (one orthotropic, different $\rho$) in a first variant, one material in two orientations (PR 4's rotation) in a second, labels set with `dataclasses.replace`;
- **R:** mesh T after `rcm` (node and element order no longer grid order).

Random $s_K, s_M \in [0.1, 2]$ with one zero, nodal masses with zeros, `slab` ∈ {1, 2 (∤ n_z on F: padding), None (one slab)}.

1. **apply vs SciPy and PR 5:** `apply_K`, `apply_M` (consistent, lumped), $m \in \{1, 5\}$, equal the scaled PR 4 `assemble` product to 1e-12, and PR 5's `unstructured` on the same mesh to 1e-12, on F, S, T (both variants), R.
2. **Diagonals:** `diag_K`, `diag_M` (consistent, lumped) equal the SciPy diagonals to 1e-12, same meshes.
3. **Gradients:** `jax.grad` of $w^\top K(s_K) v$ with respect to $s_K$ equals central finite differences (step 1e-3, linear in $s$, so to 1e-8) and the per-element $w_e^\top K_e v_e$ from SciPy element matrices to 1e-12; likewise $s_M$ with $M$ (consistent and lumped).
4. **No recompile:** one compiled executable gives the new SciPy products after replacing `s_K`, `s_M`, `C`, `rho`, `nodal`.
5. **Geometry check:** a mesh with one point moved by 1e-6, and one with two connectivity entries of an element swapped, raise; a non-voxel mesh (`box_mesh`) raises; a renumbered mesh passes (R).
6. **Inputs:** unknown `mass`, `slab=0`, bad `nodal_mass`, wrong-shape `scaled` arrays, 1-D or wrong-length `v` raise; `slab > n_z` is capped.
7. **int32:** a monkeypatched `check_int32` records `n_dof`, `node`, `cell`, `phase`; index leaves are int32.
8. **Estimator storage** equals the leaf bytes minus `C`, `rho` exactly (F, S, T).
9. **Workspace** (slow): `memory_analysis().temp_size_in_bytes` of `apply_K` and `apply_M` within 10% of `workspace` at $87^3$ for $m \in \{1, 4, 16, 24\}$.
10. **Mac acceptance** (slow, darwin): full $87^3$ voxel box (2.04M DOF), via `harness.measure`: **setup** imports and a warm-up run, freed; **code** builds `op`, allocates six preallocated `(n_dof, 24)` fp64 arrays (≈ 2.35 GB) and $v$ `(n_dof, 24)`, runs `apply_K` and `apply_M`. Estimate: `total` plus the six arrays; within 20% of the measured peak, and ≥ 2e9.
11. **Slab rule** (slow): at $87^3$, the default slab's `apply_K` time is within 20% of the best of slab/4, slab, 4·slab for $m \in \{1, 16\}$ (fresh subprocesses, min of 5).

## Docs

- `docs/operators.md`: a voxel stencil section: API, restriction, geometry check, leaves, the slab loop and kernels, the measured layout facts (corner-major GEMM, fill/drop), slab rule, estimator, validation, cost vs PR 5.
- `docs/mesh.md`: the voxel paragraph's "PR 6 checks that they agree with the geometry" becomes the description of the check (pointer to operators.md).
- `docs/ROADMAP.md`: status line for PR 6; the 2M voxel memory row "Operator < 0.1 GB" confirmed with the measured value.
- README: status and a short `stencil` snippet (executed by `tests/test_readme.py`).

## Not covered

- Rigid body modes for voxel meshes (PR 7); multigrid (PR 11); the voxel sensitivity pass (PR 21; `jax.grad` through the operator works already).
- Distorted voxel meshes and hex20/hex27 grids (use `unstructured`).
