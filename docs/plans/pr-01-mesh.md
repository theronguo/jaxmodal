# PR 1 plan: mesh data model and parser

Branch `pr-01-mesh`, from `main` after PR 0. The scope is the roadmap's PR 1 (Contents, Docs, Tests). It is host-side NumPy only: no JAX and no element kernels. PR 3 adds those.

## Measured facts (meshio 5.3.5, this Mac)

- **Node ordering:** meshio's internal hex orderings are VTK's.
  - The Gmsh reader and writer permute `hexahedron20`/`hexahedron27` to and from Gmsh order (`meshio/gmsh/common.py`).
  - The Abaqus reader maps `C3D8*` → `hexahedron` and `C3D20*` → `hexahedron20` without permuting. The Abaqus C3D20 order is the same as VTK's.
  - meshio has no Abaqus hex27 (`C3D27` is commented out).
- **What survives a meshio write → read:**
  - **Abaqus:** `point_sets` and `cell_sets` survive (`*NSET`/`*ELSET`); `cell_data` is lost.
  - **VTU:** `cell_data` survives. meshio converts sets to point/cell data with an info message.
  - **Gmsh 2.2:** `gmsh:physical` and `gmsh:geometrical` cell data survive.
  - **Gmsh 4.1:** the meshio writer drops `gmsh:physical`.
- **Abaqus parts:** meshio's Abaqus reader does not understand `*Part`/`*Instance`/`*Assembly` (roadmap: "flat" `.inp` only).

## Files

```
src/jaxmodal/mesh/__init__.py   re-exports Mesh, Voxel, read, write, voxel_mesh, corner_jacobians
src/jaxmodal/mesh/model.py      Mesh, Voxel, corner_jacobians
src/jaxmodal/mesh/ordering.py   VTK reference coordinates for hex8/hex20/hex27; per-format permutations
src/jaxmodal/mesh/io.py         read, write
src/jaxmodal/mesh/voxel.py      voxel_mesh
tests/data/                     hand-written one-element files: hex20/hex27 in Gmsh 2.2 and 4.1 ASCII, hex20 in Abaqus, hex20/hex27 in VTU
tests/test_mesh.py
docs/mesh.md
```

## Data model (`model.py`)

```python
@dataclass(frozen=True)
class Voxel:
    shape: tuple[int, int, int]      # elements per axis (nx, ny, nz)
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    node_map: np.ndarray             # (nx+1)(ny+1)(nz+1) grid node -> mesh node, -1 if dropped
    element_map: np.ndarray          # nx*ny*nz grid cell -> mesh element, -1 if inactive

@dataclass(frozen=True)
class Mesh:
    points: np.ndarray                       # (n_nodes, 3) float64
    blocks: dict[str, np.ndarray]            # element type -> (n_el, n_nodes_per_el) int64, VTK order
    node_sets: dict[str, np.ndarray]         # name -> sorted unique node indices
    element_sets: dict[str, dict[str, np.ndarray]]  # name -> {element type -> sorted indices into that block}
    material: dict[str, np.ndarray]          # element type -> (n_el,) int64 material ID, default 0
    orientation: dict[str, np.ndarray]       # element type -> (n_el,) int64 orientation ID, default 0
    voxel: Voxel | None = None
```

- **Element types:** `"hex8"`, `"hex20"` and `"hex27"`. PR 16 adds its solid-shell type as another key. Per-block dicts let each block keep its own node count.
- **Index dtype:** indices are int64 on the host. The int32 cast and the overflow asserts belong to PR 4 (Principle 6).
- **Validation:** `__post_init__` checks shapes and dtypes, that every index is in range, that `material`/`orientation` have the same keys and lengths as `blocks`, and that there are no orphan nodes (every node is referenced by some element). It raises `ValueError` on any violation.
- **Ordering:** the element order is the block order, then the row order inside each block.
- **`corner_jacobians(mesh) -> dict[str, (n_el, 8) array]`:** the trilinear (hex8) Jacobian determinant at the 8 corners, computed from the first 8 nodes. Those are the corners in VTK order for every hex type. At corner $a$, with edge vectors to its three neighbours along $\xi, \eta, \zeta$:

  $$\det J_a = \det\left[\tfrac12(x_{a+\xi} - x_a)\;\tfrac12(x_{a+\eta} - x_a)\;\tfrac12(x_{a+\zeta} - x_a)\right]$$

  with the sign of each difference following the reference axis direction. `read` and `voxel_mesh` don't call it; users and tests do.

## Orderings (`ordering.py`)

- `REF = {"hex8": ..., "hex20": ..., "hex27": ...}`: VTK reference coordinates in $[-1, 1]^3$, following the VTK documentation (`vtkHexahedron`, `vtkQuadraticHexahedron`, `vtkTriQuadraticHexahedron`).
- Per-format permutations are needed only where meshio doesn't apply one: none for the supported formats (Gmsh is handled by meshio, and Abaqus C3D20 and VTU are already in VTK order). The module records this. A tested hand-built file per format and type guards it.
- The ordering tests (below) check meshio's conversions against `REF`. If a conversion is ever wrong, the fix is a permutation here, applied in `read`/`write`.

## IO (`io.py`)

- **`read(path) -> Mesh`**, through `meshio.read`, with the format from the extension (`.inp`, `.msh`, `.vtu`).
  - **Abaqus:**
    - The file is scanned first; any `*Part`, `*Instance` or `*Assembly` keyword raises `NotImplementedError("flat .inp only")`.
    - `*NSET`/`*ELSET` become `node_sets`/`element_sets`.
    - Material and orientation IDs come from element sets named `material_<k>`/`orientation_<k>`, the convention that `write` uses. Elements in no such set get 0.
  - **VTU:**
    - Material and orientation IDs come from the `material`/`orientation` cell data.
    - Sets come from 0/1 `set:<name>` point data and cell data.
  - **Gmsh:**
    - The material ID is `gmsh:physical`.
    - Orientation IDs default to 0.
    - There are no sets.
  - **Cell types:**
    - meshio's `hexahedron`/`hexahedron20`/`hexahedron27` map to `hex8`/`hex20`/`hex27`.
    - Lower-dimensional cells (`vertex`, `line*`, `triangle*`, `quad*`) are dropped. Gmsh files carry them for boundaries.
    - Any other 3D cell type raises `ValueError`.
    - meshio blocks of the same type are concatenated, with set indices remapped.
  - **Orphan nodes** left by the dropped cells are removed and indices renumbered. Node sets are restricted to the surviving nodes.
- **`write(path, mesh)`:** the inverse.
  - **Abaqus:** writes sets plus the `material_<k>`/`orientation_<k>` element sets. An Abaqus hex27 raises.
  - **VTU:** writes cell data plus the `set:` arrays.
  - **Gmsh:** written as `gmsh22` ASCII, with the material as `gmsh:physical` and geometrical = physical. A mesh with sets or nonzero orientations raises `ValueError`, because Gmsh can't hold them.
- **Voxel metadata:** `Mesh.voxel` is not written. Voxel meshes are regenerated, not read.

## Voxel generator (`voxel.py`)

```python
def voxel_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), mask=None) -> Mesh
```

- **Grid indexing:** cell $(i, j, k)$ has flat index $i + n_x(j + n_y k)$, so $x$ varies fastest. Grid node $(i, j, k)$ has flat index $i + (n_x+1)(j + (n_y+1)k)$.
- **`mask`:** a boolean `(nx, ny, nz)` array, defaulting to all active. Inactive cells are dropped. Nodes not used by an active cell are dropped too. The remaining nodes and elements keep grid order.
- **Output:**
  - one `hex8` block in VTK corner order;
  - material and orientation IDs of 0;
  - no sets;
  - `voxel = Voxel(shape, spacing, origin, node_map, element_map)`.
- **Vectorized:** the generator is vectorized NumPy with no Python loop over cells, so 2M-DOF meshes are built in seconds.

## Tests (`tests/test_mesh.py`)

1. **Round trips.**
   - For each format and each supported type, a two-element mesh goes through `write` → `read` → `write` → `read`. The supported types are hex8/hex20/hex27, except Abaqus hex27.
   - The mesh has two materials and, where the format allows, node and element sets and orientations.
   - After the round trip, points, blocks, IDs and the supported sets are equal. Points are compared to 1e-12 relative for the ASCII formats.
   - A mixed hex8 + hex20 mesh round-trips through VTU.
2. **Orderings against hand-built references.**
   - Each file in `tests/data/` holds one element. Its node coordinates are written by hand from that format's documentation (Gmsh node-ordering docs, the Abaqus C3D20 figure, VTK class docs), as the scaled and shifted reference coordinates of a known affine map $x = A\xi + b$ with $\det A > 0$.
   - After `read`, `points[conn]` equals $A\,\mathrm{REF} + b$ in VTK order to 1e-12.
   - This checks meshio's permutations, our type mapping and `REF` together. A deliberate swap of two midside nodes in a copy of the file fails the test.
3. **Corner Jacobians.**
   - For the affine elements of test 2 and for the voxel meshes, `corner_jacobians` equals $\det A / 8$ at all corners (Jacobian of $x(\xi)$ with $\xi \in [-1,1]^3$).
   - Mirroring one element (swapping its top and bottom faces) gives negative values.
   - A randomly perturbed hex8 is compared against a central finite-difference Jacobian of the trilinear map at the corners.
4. **Voxelized sphere.**
   - The grid is `shape = (20, 22, 24)` with spacing `(0.5, 0.4, 0.3)`. The mask is cell centres inside a sphere at the box centre.
   - The element count equals `mask.sum()`.
   - The node count equals a NumPy reference: grid nodes adjacent to any active cell, computed by OR-ing the 8 shifted copies of the padded mask.
   - The volume, $\sum_e$ (mean corner $\det J$) $\cdot 8$ (exact for boxes), equals `mask.sum() * prod(spacing)` to 1e-12.
   - No orphan nodes: every node index appears in the connectivity.
   - `node_map`/`element_map` are consistent with `points` and the grid formula.
5. **Validation.** Each of these raises:
   - an out-of-range index;
   - an orphan node;
   - mismatched material lengths;
   - an Abaqus file with `*Part`;
   - a tetra block;
   - writing Gmsh with sets.

## Docs

- **`docs/mesh.md`:**
  - the `Mesh` fields;
  - VTK node-ordering tables for hex8/20/27, with reference coordinates;
  - what each format reads and writes, including the `material_<k>` convention, Gmsh's limits and the flat-`.inp` restriction;
  - voxel grid indexing and the maps;
  - the corner Jacobian formula.
- **`README.md`:** a usage snippet (`jaxmodal.mesh.read`, `voxel_mesh`).
- **`docs/ROADMAP.md`:** no content change expected. A status note is added after the merge.

## Acceptance

- `scripts/check.sh` and `uv run --locked pytest -m "not gpu"` pass.
- Code review rounds come back clean.
- Docs and memory are updated.
- Merge `--no-ff`.
