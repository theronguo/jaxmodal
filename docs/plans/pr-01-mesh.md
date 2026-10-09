# PR 1 plan: mesh data model and parser

Branch `pr-01-mesh`, from `main` after PR 0. Scope: the roadmap's PR 1 (Contents, Docs, Tests). It is host-side NumPy only, with no JAX and no element kernels (PR 3). Review round 1 is incorporated.

## Measured facts (meshio 5.3.5; ground truth from `vtk` and `gmsh` 4.15.2 Python modules, run outside the project)

- **Orderings:**
  - meshio uses VTK hex orderings internally.
  - Its Gmsh reader and writer permute `hexahedron20`/`hexahedron27` correctly in both directions, for 2.2 and 4.1, ASCII and binary. This was checked against real Gmsh meshes and by re-opening meshio-written files in Gmsh.
  - Abaqus C3D20 order equals VTK hex20 order.
- **`meshio.read` calls `sys.exit(1)` on a read error**, even when a format is given. The format modules (`meshio.vtu.read`, `meshio.gmsh.read`) raise normally.
- **meshio's Abaqus reader is unusable:**
  - it puts `*ELEMENT, ELSET=` sets on the wrong block;
  - a repeated `*ELSET` name replaces the earlier set instead of appending;
  - a second `*NODE` section raises `KeyError`;
  - lowercase types exit the process;
  - unknown types (`CPS8`, `M3D9`, `C3D27`) raise;
  - set references and `*NSET, ELSET=` come back as nested or empty sets.
- **meshio's Abaqus writer is unusable:** it writes `TYPE=C3D8RH`/`C3D20RH`, and it puts all 21 C3D20 entries on one line, where Abaqus allows at most 16 entries per data line.
- **Gmsh:**
  - **2.2:** keeps per-element `gmsh:physical`, and `$PhysicalNames` arrive as `field_data` (name → (tag, dim)).
  - **4.1 written by meshio:** drops `gmsh:physical` and can't be opened by Gmsh.
  - **4.1 read with partially tagged entities:** meshio raises `ValueError` (incompatible `gmsh:physical` blocks).
  - **Physical tag 0** means "no group" to Gmsh, which drops such elements when it re-saves a file.
  - **Extra int cell data** passed to the gmsh writer is written as `np.int64(0)` text, which the file can't parse.
- **VTU:**
  - `cell_data` round-trips.
  - Bool arrays are rejected.
  - Given `point_sets`/`cell_sets`, meshio mutates the caller's mesh and writes −1/index arrays.
- **Cell classification:** meshio cell names vary (`VTK_LAGRANGE_HEXAHEDRON`, `polygon`), but `CellBlock.dim` is correct for all of them.

## Files

```
src/jaxmodal/mesh/__init__.py   re-exports Mesh, Voxel, read, write, voxel_mesh, corner_jacobians
src/jaxmodal/mesh/model.py      Mesh, Voxel, corner_jacobians
src/jaxmodal/mesh/io.py         read, write (Gmsh/VTU via meshio format modules; own flat-.inp reader and writer)
src/jaxmodal/mesh/voxel.py      voxel_mesh
tests/data/                     hand-written files (see Tests)
tests/test_mesh.py
docs/mesh.md
```

There is no separate ordering module (Principle 4: per-format maps live in `mesh/io`). The supported formats need no permutation beyond meshio's Gmsh one. `io.py` records this in one comment, and a permutation would be added there if one is ever needed.

## Data model (`model.py`)

```python
TYPES = ("hex8", "hex20", "hex27")   # canonical block order; PR 16 appends its type

@dataclass(frozen=True)
class Voxel:
    shape: tuple[int, int, int]          # cells per axis (nx, ny, nz)
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    node_map: np.ndarray                 # (nx+1)(ny+1)(nz+1) grid node -> mesh node, -1 if dropped
    element_map: np.ndarray              # nx*ny*nz grid cell -> mesh element, -1 if inactive

@dataclass(frozen=True)
class Mesh:
    points: np.ndarray                   # (n_nodes, 3) float64
    blocks: dict[str, np.ndarray]        # type -> (n_el_type, n_per_el) int64, VTK node order, keys in TYPES order
    node_sets: dict[str, np.ndarray]     # name -> sorted unique node indices
    element_sets: dict[str, np.ndarray]  # name -> sorted unique global element indices
    material: np.ndarray                 # (n_el,) int64 material label
    orientation: np.ndarray              # (n_el,) int64 orientation label
    voxel: Voxel | None = None
```

- **Global element index:** blocks come in `TYPES` order, and rows in order within each block. A block's global range comes from the cumulative block sizes.
- **Material and orientation IDs** are labels, not table indices. PR 2/6 map them to materials, and they may be non-contiguous (Gmsh physical tags).
- **Index dtype:** int64 on the host. `docs/conventions.md` gets a line on this: the int32 cast and the overflow asserts belong to PR 4, before anything reaches JAX.
- **`__post_init__` validation**, vectorized and O(n), raises `ValueError` on any of these:
  - `points` not `(n, 3)` float64;
  - a block key not in `TYPES`, or keys out of `TYPES` order;
  - a node count that doesn't match the type;
  - non-integer connectivity;
  - an index out of `[0, n_nodes)`;
  - an orphan node (`seen[conn] = True; seen.all()`);
  - set indices that are not sorted, unique and in range;
  - `material`/`orientation` lengths ≠ `n_el`;
  - for `voxel`:
    - a single `hex8` block;
    - map lengths matching `shape`;
    - each map's non-negative entries equal to `arange(n_nodes)` and `arange(n_el)`, in grid order.
- **Voxel maps are the source of truth.** Any later node or element permutation (PR 5 RCM, PR 18 reordering) must update them. The docs show the derived views:
  - mesh → grid: `np.flatnonzero(node_map >= 0)`, valid while grid order is kept;
  - the occupancy mask: `(element_map >= 0).reshape(nz, ny, nx)`.
- **Solid-shell (PR 16)** stores the thickness direction by permuting each element's connectivity, so no mesh field is needed. Reading Abaqus `SC8R` is added to `io.py` by PR 16.
- **`corner_jacobians(mesh) -> dict[str, (n_el_type, 8) array]`:** the trilinear Jacobian determinant at each corner, from the first 8 nodes (the corners, in VTK order for every hex type).
  - At corner $a$, column $d$ of $J_a$ is $\tfrac12(x_{a_d^+} - x_{a_d^-})$, where $(a_d^-, a_d^+)$ is the pair of corners along reference axis $d$ that contains $a$, ordered by $\xi_d$. The pairs are:
    - $\xi$: (0,1), (3,2), (4,5), (7,6);
    - $\eta$: (0,3), (1,2), (4,7), (5,6);
    - $\zeta$: (0,4), (1,5), (2,6), (3,7).
  - For $x = A\xi + b$ on $[-1,1]^3$, $\det J_a = \det A$.
  - It is computed corner by corner as $d_0 \cdot (d_1 \times d_2)$, which keeps O(n) temporaries. A batched `(n, 8, 3, 3)` `det` would need about 60 GB at 33M elements.

## IO (`io.py`)

`read(path) -> Mesh` and `write(path, mesh)` dispatch on the extension: `.inp`, `.msh`, `.vtu`.

### Abaqus (own reader and writer, flat `.inp` only)

- **Reader**, about 60 lines:
  - **Comments and keywords:** `**` lines are comments. Keywords are matched case-insensitively on the first comma-separated token.
  - **`*NODE`:** any number of sections.
  - **`*ELEMENT, TYPE=`:**
    - `C3D8*` → `hex8` and `C3D20*` → `hex20`. Suffixes such as R/H/I are ignored, because the project uses full integration. The docs say so.
    - Any other `C3D*` type raises.
    - Non-`C3D` types (S4, CPS8, M3D9, B31, …) are skipped.
    - Data lines can continue onto the next line; an element is complete once it has its node count.
    - An `ELSET=` on the `*ELEMENT` line adds the section's elements to that set.
  - **`*NSET`/`*ELSET`:**
    - integer lists, or `GENERATE` (start, stop, step);
    - a repeated name appends to the existing set;
    - labels of skipped elements are dropped from element sets.
  - **These raise `ValueError`:**
    - `*PART`, `*INSTANCE`, `*ASSEMBLY` and `*INCLUDE`. Abaqus/CAE exports always contain these, so they are rejected; the docs say so.
    - A set referenced by name inside another set.
    - `*NSET, ELSET=`.
    - A node label used before it is defined.
  - **Other keywords** are ignored, including `*SOLID SECTION` and `*ORIENTATION`. The docs state that real section assignments are not read.
  - **Labels:** node and element labels may be non-contiguous and in any order. They are mapped to indices in file order.
  - **Materials and orientations** come from element sets named `material_<k>`/`orientation_<k>`, matched case-insensitively with integer $k$. Elements in no such set get label 0, and an element in two such sets raises. These sets are removed from `element_sets`.
- **Writer**, about 20 lines:
  - `*NODE`;
  - one `*ELEMENT, TYPE=C3D8` or `TYPE=C3D20` section per block, with global 1-based labels and at most 16 entries per line;
  - `*NSET`/`*ELSET`, each written once with global labels (8 per line);
  - `material_<k>`/`orientation_<k>` sets for every label present.
  - hex27 raises.

### Gmsh (`meshio.gmsh.read`, `meshio.gmsh.write` as `gmsh22` ASCII)

- **Read:**
  - **Node sets from lower-dimensional physical groups:** for every physical tag on cells with `dim < 3`, the union of their nodes becomes a node set. It is named from `field_data` (`$PhysicalNames`), falling back to `physical_<tag>`. This is how Gmsh models mark clamped faces.
  - **Materials:** the material label of each 3D element is its `gmsh:physical` tag, or 0 without physical groups. Orientations are 0. 3D physical groups don't become element sets; materials cover them.
  - **meshio's partial-physical `ValueError`** is re-raised with a message that points at `Mesh.SaveAll`.
- **Write:**
  - only `gmsh:physical = material` and `gmsh:geometrical = material` are passed;
  - node sets, element sets or nonzero orientations raise, because Gmsh can't hold them without boundary cells;
  - the docs say material 0 is physical tag 0, which Gmsh treats as "no group".

### VTU (`meshio.vtu.read`, `meshio.vtu.write`)

- **Read:** material and orientation labels come from the `material`/`orientation` cell data, defaulting to 0. VTU carries no sets.
- **Write:** writes `material`/`orientation` cell data only; sets are not written, as the docs say. Only our own arrays are passed, never `point_sets`/`cell_sets`.

### Shared read path (Gmsh, VTU)

- **Cell classification** by `CellBlock.dim`:
  - meshio `hexahedron`/`hexahedron20`/`hexahedron27` → `hex8`/`hex20`/`hex27`;
  - `dim < 3` cells are dropped after node-set extraction;
  - any other `dim == 3` type raises `ValueError`.
- **Same-type blocks** are concatenated, with materials carried along. Blocks are ordered by `TYPES`.
- **Orphan removal:** nodes used only by dropped cells are removed and the rest renumbered in their original order. Node sets are restricted to surviving nodes, and members that are dropped are reported with a warning. Abaqus nodes that no kept element uses are removed the same way.
- **Voxel metadata** is not written. Voxel meshes are regenerated, not read.

## Voxel generator (`voxel.py`)

```python
def voxel_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), mask=None) -> Mesh
```

- **Indexing:**
  - Cell $(i, j, k)$ has flat index $i + n_x(j + n_y k)$, so $x$ varies fastest. Grid node $(i, j, k)$ has flat index $i + (n_x+1)(j + (n_y+1)k)$.
  - `mask[i, j, k]` is the cell's occupancy, a bool array of shape `shape`. Internally the code works on `mask.transpose(2, 1, 0)`, a C-order `(nz, ny, nx)` array, so flat order equals `mask.ravel(order="F")`.
- **Output:**
  - Active cells become `hex8` elements in grid order, with VTK corner order.
  - Nodes used by an active cell are kept, in grid order.
  - Material and orientation labels are 0; voxel phases are set with `dataclasses.replace(mesh, material=...)`.
  - There are no sets.
  - `voxel = Voxel(shape, spacing, origin, node_map, element_map)`.
- **Errors:** spacing ≤ 0, a mask of the wrong shape or dtype, or an all-false mask raise `ValueError`.
- **Implementation:** vectorized NumPy. Measured in a prototype: a 2.1M-DOF grid in 0.03 s, and a 99M-DOF full grid in 1.6 s with 9.5 GB peak RSS (int64 connectivity 2.1 GB).

## Tests (`tests/test_mesh.py`)

The VTK reference coordinates `REF[type]` live in the test file only. They are the test oracle; PR 3's registry owns the library's reference coordinates.

1. **`REF` checked from topology alone.**
   - hex8 rows are the $\pm1$ vertices with the bottom face counter-clockwise seen from $+z$, then the top face.
   - hex20 rows 8–19 are the midpoints of the edges (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7).
   - hex27 rows 20–25 are the means of the corners of the faces $-x, +x, -y, +y, -z, +z$, and row 26 is the origin. The first 20 rows equal hex20.
2. **Read orderings against hand-written files.**
   - `tests/data/` holds one element per format and type:
     - Gmsh 2.2 and 4.1 ASCII for hex8, hex20 and hex27, the 4.1 files with `$Entities`, physical tags and non-contiguous node tags;
     - Abaqus `C3D8`, `C3D8R` and `C3D20`;
     - VTU for hex8, hex20 and hex27.
   - Node coordinates are written in each format's documented order, as $x = A\xi + b$ with a fixed $A$ ($\det A > 0$) and $b$.
   - After `read`, `points[conn]` equals $A\,\mathrm{REF} + b$ to 1e-12. The Gmsh 4.1 material equals its physical tag.
   - Parametrized mutants (two midside nodes swapped, two face centres swapped, two corners swapped) are applied to copies of the files, and each makes the check fail.
3. **Write orderings.**
   - The affine one-element mesh is written to `.msh` and `.inp`.
   - The raw `$Elements`/`*ELEMENT` lines are parsed in the test, and the coordinates of each listed node equal the format's documented order (the hand-written files of test 2).
4. **Round trips.**
   - A mesh is written and read back, then compared with the in-memory original: points to 1e-12, with random non-integer coordinates; blocks, including `list(blocks)`; labels; and the supported sets.
   - The meshes have shared nodes, non-identity numbering, two materials and multi-digit labels (`material_12`).
   - The format cases are:
     - Abaqus: hex8 + hex20 mixed, with node and element sets and one material set spanning both blocks;
     - Gmsh: hex8 + hex20 + hex27 mixed, materials only;
     - VTU: hex8 + hex20 + hex27 mixed, with materials and orientations.
   - The Abaqus output has `TYPE=C3D8`/`TYPE=C3D20` and no data line with more than 16 entries.
5. **Messy input, one case per format, checked geometrically.**
   - **Gmsh and VTU input** (built with `meshio.Mesh` and written by meshio):
     - blocks in the order `quad, hexahedron, quad, hexahedron, vertex`;
     - two hex blocks of different sizes and materials;
     - nodes used only by quads, sitting in the middle of the point array;
     - Gmsh only: a physical surface named "clamp".
   - **Abaqus input:** a hand-written file with:
     - two `*NODE` sections;
     - unsorted, non-contiguous labels;
     - a lowercase `c3d8r` section;
     - an `S4` section;
     - `C3D20` and `C3D20R` sections;
     - `*ELEMENT, ELSET=`;
     - a repeated `*ELSET` name;
     - `GENERATE`;
     - continuation lines.
   - **References are coordinates, not indices:**
     - each element's `points[conn]`;
     - each element's material, looked up by centroid;
     - node sets compared as coordinate sets (the "clamp" set equals the quad nodes);
     - element sets compared by centroid;
     - no orphans.
6. **Corner Jacobians.**
   - On the affine elements of test 2, the value is $\det A$ at all 8 corners. On voxel meshes it is $\prod(\text{spacing})/8$.
   - A randomly rotated and perturbed hex8 matches the analytic derivative of the trilinear shape functions at the corners.
   - The same perturbed corners in a hex20 and a hex27 (midside nodes moved arbitrarily) give the same values, which shows only corners are used.
   - Mirroring a hex8 and a hex20 (swapping top and bottom faces) gives negative values.
   - Every `tests/data` file has positive values after `read`.
7. **Voxel generator.**
   - **Sphere** of `shape = (20, 22, 24)`, spacing `(0.5, 0.4, 0.3)`, centred:
     - the element count equals `mask.sum()`;
     - the node count equals a brute-force Python set of the corner grid indices of the active cells;
     - $\sum_e 8\cdot\overline{\det J_e}$ equals `mask.sum() * prod(spacing)` to 1e-12;
     - there are no orphans.
   - **Asymmetric mask:** a seeded random mask on `shape = (3, 4, 5)` with a nonzero origin, including cells that touch only at an edge or a corner. A triple Python loop over $(i, j, k)$ is the reference:
     - each active cell's element `element_map[g]` has `points[conn]` equal to origin + spacing·($(i,j,k)$ + VTK corner offsets);
     - `element_map >= 0` equals `mask.ravel(order="F")`;
     - the non-negative map entries are `arange` (grid order);
     - `node_map` is −1 exactly at unused grid nodes.
8. **Validation.** Each of these raises:
   - Mesh construction:
     - an index of −1 or `n_nodes`;
     - float connectivity;
     - `points` of shape `(n, 2)`;
     - an unknown block key;
     - blocks out of `TYPES` order;
     - an orphan node;
     - unsorted or duplicate set indices;
     - `material` or `orientation` length mismatches;
     - inconsistent voxel maps.
   - Reading Abaqus files with:
     - `*Part`;
     - `*INCLUDE`;
     - a set-name reference;
     - `*NSET, ELSET=`;
     - `C3D27`;
     - an element in two material sets.
   - Reading a VTU with a tetra block.
   - Writing:
     - Gmsh with sets or orientations;
     - Abaqus hex27.
   - Generating voxels with bad spacing, mask shape or dtype, or an all-false mask.
   - Also tested: a `.msh` without physical groups reads with material 0.
9. **Size (slow).** `voxel_mesh` of a full 88³ grid (about 2.1M DOFs) builds and validates in under 5 s.

## Docs

- **`docs/mesh.md`:**
  - **Model:** the `Mesh` fields, the global element index and the canonical type order.
  - **Ordering:** VTK node-ordering tables for hex8/20/27 with reference coordinates, and the corner-pair Jacobian formula.
  - **Formats:** what each format reads and writes:
    - Abaqus: flat `.inp` only (CAE exports rejected), `material_<k>` sets, sections not read, element suffixes ignored;
    - Gmsh: physical groups → node sets and materials, physical tag 0, write as 2.2;
    - VTU: no sets.
  - **Voxels:** grid indexing in LaTeX, the mask layout, the maps and their invariant, and setting phases.
- **`docs/conventions.md`:** add the host int64 exception to the Indices line.
- **`README.md`:** a usage snippet (`jaxmodal.mesh.read`, `voxel_mesh`).
- **`docs/ROADMAP.md`:** the Dependencies note "a small custom reader is added only if needed" becomes "PR 1 adds a flat-`.inp` reader and writer (meshio's are unusable; see `docs/mesh.md`)". A status note is added after the merge.

## Acceptance

- `scripts/check.sh` and `uv run --locked pytest -m "not gpu"` pass.
- Code review rounds come back clean.
- Docs and memory are updated.
- Merge with `--no-ff`.
