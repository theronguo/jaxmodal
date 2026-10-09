# PR 1 plan: mesh data model and parser

Branch `pr-01-mesh`, from `main` after PR 0. Scope: the roadmap's PR 1 (Contents, Docs, Tests). It is host-side NumPy only, with no JAX and no element kernels (PR 3). Review rounds 1–5 are incorporated.

## Measured facts (meshio 5.3.5; ground truth from `vtk` and `gmsh` 4.15.2 Python modules, run outside the project)

- **Orderings:**
  - meshio uses VTK hex orderings internally.
  - Its Gmsh 4.1 reader permutes `hexahedron20`/`hexahedron27` correctly (checked against real Gmsh meshes).
  - Abaqus C3D20 order equals VTK hex20 order.
- **`meshio.read` calls `sys.exit(1)` on a read error**, even when a format is given. The format modules (`meshio.vtu.read`, `meshio.gmsh.read`) raise normally.
- **meshio's Abaqus reader is unusable:** `*ELEMENT, ELSET=` lands on the wrong block, repeated sets replace instead of append, a second `*NODE` raises, lowercase types exit, unknown types raise.
- **Gmsh:**
  - **2.2 cannot be validated:** an entity in several groups is written once per group; a volume outside every group is silently left out (no `$Entities` to detect it); with SaveAll every tag is 0. Hence only 4.1 is supported.
  - **meshio's `field_data`** is keyed by name, so a name used in two dimensions overwrites the other.
  - **4.1 read:** `gmsh:physical` keeps only the first tag of each entity, and meshio builds `cell_sets` only for named groups (plus a `gmsh:bounding_entities` key). So an unnamed second group is lost entirely. `gmsh:geometrical` (the entity tag) is present on every block, and `$Entities` lists every entity's physical tags, including volumes that Gmsh left out of the file.
  - **4.1 with partially tagged entities (SaveAll):** meshio raises `ValueError` (incompatible `gmsh:physical` blocks).
  - **Header:** `$MeshFormat` line 2 is `4.1 0 8` (ASCII) or `4.1 1 8` (binary); Gmsh 4.0 writes `4 0 8`, which meshio misreads.
  - **Partitioned 4.1** (`$PartitionedEntities`) with groups makes meshio raise a bare `KeyError`.
  - **4.1 without `$Entities`** exists (meshio writes it when there are no bounding entities); `gmsh:geometrical` is then 0.
- **VTU:**
  - `cell_data` round-trips.
  - Bool arrays are rejected.
  - Given `point_sets`/`cell_sets`, meshio mutates the caller's mesh and writes −1/index arrays.
  - Points round-trip bit-exactly (ASCII and binary).
- **Cell classification:** meshio cell names vary (`VTK_LAGRANGE_HEXAHEDRON`, `polygon`), but `CellBlock.dim` is correct for all of them.

## Files

```
src/jaxmodal/mesh/__init__.py   re-exports TYPES, Mesh, Voxel, read, write, voxel_mesh, corner_jacobians
src/jaxmodal/mesh/model.py      Mesh, Voxel, corner_jacobians
src/jaxmodal/mesh/io.py         read (Gmsh/VTU via meshio format modules; own flat-.inp reader), write (VTU)
src/jaxmodal/mesh/voxel.py      voxel_mesh
tests/data/                     hand-written files (see Tests)
tests/test_mesh.py
docs/mesh.md
```

There is no ordering module or map: the supported formats need no permutation beyond meshio's Gmsh one (Abaqus C3D20 = VTK). A permutation would go into `io.py` if a later format ever needs one.

## Data model (`model.py`)

```python
TYPES = ("hex8", "hex20", "hex27")   # canonical block order; PR 16 appends its type

@dataclass(frozen=True, eq=False)
class Voxel:
    shape: tuple[int, int, int]          # cells per axis (nx, ny, nz)
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    node_map: np.ndarray                 # (nx+1)(ny+1)(nz+1) grid node -> mesh node, -1 if dropped
    element_map: np.ndarray              # nx*ny*nz grid cell -> mesh element, -1 if inactive

@dataclass(frozen=True, eq=False)
class Mesh:
    points: np.ndarray                   # (n_nodes, 3) float64
    blocks: dict[str, np.ndarray]        # type -> (n_el_type, n_per_el) int64, VTK node order, keys in TYPES order
    node_sets: dict[str, np.ndarray]     # name -> sorted unique node indices, int64
    element_sets: dict[str, np.ndarray]  # name -> sorted unique global element indices, int64
    material: np.ndarray                 # (n_el,) int64 label
    orientation: np.ndarray              # (n_el,) int64 label
    voxel: Voxel | None = None

    offsets -> dict[str, int]            # property: start of each block in the global element index; n_el property
```

- **Global element index:**
  - Blocks come in `TYPES` order, and rows in order within each block.
  - `offsets` (present types only) and `n_el` give each block's slice. The readers use them now, and PRs 4, 5, 16, 17, 21 and 23 will too.
- **Material and orientation IDs** are labels (any int64, including negative), not table indices, and may be non-contiguous (e.g. Gmsh physical tags). PR 2/6 map them to materials.
- **dtypes:** strict, deliberately (no silent copies at 100M scale): the arrays must be `np.ndarray` (host-side only) with exactly these dtypes (float64 points; int64 connectivity, sets, labels and voxel maps), or construction raises. Readers and `voxel_mesh` cast before constructing; label casts first check that values are integral.
- **Index rule:** `docs/conventions.md` and ROADMAP Principle 7 both get the same wording: index arrays passed to JAX are int32; host-side NumPy topology is int64 until PR 4's overflow asserts and cast.
- **`__post_init__` validation**, vectorized and O(n), raises `ValueError` on:
  - wrong types, shapes or dtypes (sets and maps are 1-D; set keys `str`; `voxel` a `Voxel`);
  - non-finite points;
  - no elements, or a block with zero rows;
  - a block key not in `TYPES`, or keys out of order;
  - the wrong node count for a type;
  - an index outside `[0, n_nodes)`;
  - an orphan node (`seen[conn] = True; seen.all()`);
  - set indices that aren't sorted, unique and in range (empty sets are allowed);
  - label lengths ≠ `n_el`;
  - for `voxel`:
    - anything other than a single `hex8` block;
    - `shape` not 3 positive `numbers.Integral` (not `bool`); `spacing`/`origin` not 3 finite `numbers.Real`; spacing ≤ 0 (`voxel_mesh` checks `shape` entries are non-bool `numbers.Integral` > 0 before normalizing to Python `int`/`float` tuples);
    - map lengths that don't match `shape`;
    - maps that aren't bijections onto the mesh: entries in `[-1, n)`, the non-negative count equal to `n`, and `bincount(...).max() == 1`.
- **Grid order is a property of `voxel_mesh` output only**, not a `Mesh` invariant. This keeps PR 5 RCM and PR 18 reordering legal. The checklist (in `docs/mesh.md`):
  - node permutation $p$ (new → old): `points[p]`, connectivity remapped through $p^{-1}$, `node_sets` remapped and re-sorted, `node_map` remapped;
  - element permutation $q$, within a block (the global index is fixed by `TYPES` order): block rows, `material`, `orientation`, `element_sets` (re-sorted) and `element_map`.
- **Inverse maps** work under any permutation:
  - mesh → grid nodes: `inv = np.empty(n, np.int64); inv[m[m >= 0]] = np.flatnonzero(m >= 0)`, and the same for elements (PR 23 scatters densities this way);
  - mask: `(element_map >= 0).reshape(shape, order="F")`.
- **What `Mesh` does not check:** whether the voxel maps agree with `points` and the connectivity. PR 6 checks that before dispatching to the stencil, as the first consumer that relies on regular geometry.
- **Degenerate elements** (repeated nodes) are not rejected by `Mesh`; `corner_jacobians` shows 0 at the collapsed corners.
- **Immutability:** fields, including the dicts, are treated as immutable; change them with `dataclasses.replace`, which re-validates (documented; arrays are not made read-only, since that would flip the caller's flag).
- **Orphan rule:** a mesh derived from another, e.g. PR 12's hex8 corner mesh from a hex20 mesh, must compact its nodes. This is by design and noted in the docs.
- **Voxel BCs:** voxel meshes have no sets. A clamp set is added with `replace(mesh, node_sets={"clamp": ...})`, as shown in the docs.
- **Solid-shell (PR 16)** stores the thickness direction by rotating each element's connectivity (a proper rotation, so Jacobian signs are kept), so no mesh field is needed. Reading Abaqus `SC8R` is added to `io.py` by PR 16.
- **`corner_jacobians(mesh) -> dict[str, (n_el_type, 8) array]`:** the trilinear Jacobian determinant at each corner, from the first 8 nodes (the corners, in VTK order for every hex type).
  - At corner $a$, column $d$ of $J_a$ is $\tfrac12(x_{a_d^+} - x_{a_d^-})$, where $(a_d^-, a_d^+)$ is the pair of corners along reference axis $d$ that contains $a$, ordered by $\xi_d$. The pairs are:
    - $\xi$: (0,1), (3,2), (4,5), (7,6);
    - $\eta$: (0,3), (1,2), (4,7), (5,6);
    - $\zeta$: (0,4), (1,5), (2,6), (3,7).
  - For $x = A\xi + b$ on $[-1,1]^3$, $\det J_a = \det A$.
  - It is computed corner by corner as $d_0 \cdot (d_1 \times d_2)$, which keeps O(n) temporaries.
  - For hex20/hex27 it is the Jacobian of the corner hex: a sign and orientation check only. Curved-element validity at quadrature points is PR 3's check.

## IO (`io.py`)

- `read(path) -> Mesh` dispatches on the extension (`.inp`, `.msh` = Gmsh 4.1 ASCII, `.vtu`); `write(path, mesh)` accepts `.vtu` only. Other extensions raise `ValueError`.
- Gmsh and Abaqus are read-only: no later PR writes them (PR 17 exports XDMF), and the writers would bring their own edge cases (tag 0, set-name collisions, line limits).
- Every reader raises `ValueError` when no 3D element is kept, and drops sets that end up empty, with a warning.

### Abaqus (own reader, flat `.inp` only)

About 135 lines. It reads the whole file, then resolves labels.

- **Lines:**
  - `**` lines are comments, including between continuation lines.
  - Keywords and parameters are case-insensitive, with whitespace around `=` and quotes around names stripped. A missing `TYPE=` raises `ValueError`.
  - Empty tokens from trailing commas are dropped.
- **`*NODE`:** any number of sections; an optional `NSET=` adds the section's nodes to that set; every node has a label and exactly 3 coordinates; it may continue over lines but must end at a line end.
- **`*ELEMENT, TYPE=`:**
  - the upper-cased value matching `re.fullmatch(r"C3D(8|20)[A-Z]*")` → `hex8`/`hex20`. Suffixes are ignored, since the project uses full integration; the docs say so.
  - Any other `C3D*` raises.
  - Other types (S4, CPS8, T3D3, DC3D8, SC8R, …) are skipped: their data lines are ignored, with one warning per type.
  - Kept elements may span continuation lines; an element is complete at its node count and must end at a line end. More entries than that, or an element cut off by the next keyword or end of file, raise (checked per section: the entry count, then the row ends against the line ends).
  - `ELSET=` adds the kept elements to that set; on a skipped section it is ignored (no set, no warning).
- **`*NSET` / `*ELSET`:**
  - integer lists, or `GENERATE` (start, stop, optional step defaulting to 1); any non-integer token raises (this is how set-name references are rejected);
  - names are case-insensitive: lookup uses the upper-case name, and the output dict key is the first spelling;
  - a repeated name appends; members are `np.unique`d.
- **Label resolution** after reading:
  - node labels may be non-contiguous and in any order, and map to indices in file order;
  - an element label maps to (type, row within that type in file order), and its global index is `offsets[type] + row`; element sets, materials and orientations are built from this index after the blocks are concatenated;
  - an undefined or duplicate node label raises (connectivity or node sets), and so does a duplicate element label;
  - element-set labels that aren't kept elements are dropped (they can't be told apart from skipped elements), with one warning and a count.
- **These keywords raise `ValueError`** (one tuple):
  - `*PART`, `*INSTANCE`, `*ASSEMBLY` and `*INCLUDE` (Abaqus/CAE exports are rejected; the docs say so);
  - geometry-creating keywords: `*NGEN`, `*NFILL`, `*NCOPY`, `*NMAP`, `*ELGEN`, `*ELCOPY` and `*SYSTEM`.
  - Also raising: `*NSET, ELSET=`.
- **Other keywords are ignored, data lines included.** This covers `*MATERIAL`, `*ELASTIC`, `*SOLID SECTION` and `*ORIENTATION`; the docs state that real section assignments are not read.
- **Materials and orientations:**
  - They come from sets matching `(material|orientation)_(\d+)` case-insensitively, with label `int(k)` (so disjoint `material_007` and `material_7` merge into label 7; an element in both still raises). Labels ≥ $2^{31}$ raise. Label 0 is the default; `material_0` merges with it (documented).
  - Elements in none get 0, and an element in two raises.
  - These sets are removed from `element_sets`. Names like `material_abc` stay ordinary sets.

### Gmsh (`meshio.gmsh.read`, 4.1 ASCII only)

- **Version:** the `$MeshFormat` header is read first (binary mode); anything but `4.1` ASCII raises "save as Gmsh 4.1 ASCII (Mesh.MshFileVersion = 4.1, Mesh.Binary = 0)". 2.2 cannot be validated (see Measured facts).
- **Own header parse** (about 20 lines, stops at `$Nodes`):
  - `$PhysicalNames` → `{(dim, tag): name}` (`split(maxsplit=2)`, quotes stripped; names may contain spaces);
  - `$PartitionedEntities` raises "save unpartitioned";
  - `$Entities` → `{(dim, entity): [physical tags]}`; a file without `$Entities` has no memberships.
- **Group membership** of a cell: `phys.get((dim, gmsh:geometrical), [])`. meshio's `gmsh:physical`, `field_data` and `cell_sets` are not used.
- **Names:** from `$PhysicalNames`, falling back to `physical_<dim>_<tag>`. A duplicate name among the `dim < 3` groups (which become node sets) raises, including a collision with a fallback name.
- **Node sets:** each group with `dim < 3` becomes a node set, the union of its cells' nodes. 3D groups don't become element sets.
- **Materials:** a 3D element's material label is its physical tag, or 0 when no 3D entity has one. Orientations are 0.
- **These raise `ValueError`** with the message "put every volume in exactly one physical group and don't use Mesh.SaveAll" (our checks run before `meshio.gmsh.read`):
  - a 3D entity with more than one physical tag;
  - a 3D entity without a physical tag while others have one (Gmsh left its elements out);
  - meshio's "Incompatible cell data" `ValueError` (SaveAll with partial tags).
  - The shared "no 3D element" error adds "(are your volumes in a physical group?)" for Gmsh.

### VTU (`meshio.vtu.read`, `meshio.vtu.write`)

- **Read:** labels come from the `material`/`orientation` cell data (integral values below $2^{31}$ in magnitude required, so float 3.0 reads as 3), defaulting to 0. VTU carries no sets.
- **Write:** writes only our `material`/`orientation` arrays (int64). Sets are not written; the docs say so.

### Shared read path (Gmsh, VTU; orphan removal also Abaqus)

- **Cell classification** by `CellBlock.dim`:
  - meshio `hexahedron`/`hexahedron20`/`hexahedron27` → `hex8`/`hex20`/`hex27`;
  - `dim < 3` cells are dropped after node-set extraction;
  - any other `dim == 3` type raises.
- **Same-type blocks** are concatenated in file order, then blocks are put in `TYPES` order; labels follow their rows.
- **Orphan removal:** nodes not used by kept cells are removed, and the rest keep their original order. Node sets are restricted to surviving nodes, with a warning when members are dropped.
- **dtypes:** points are cast to float64 and indices to int64.
- **Voxel metadata** is not written. Voxel meshes are regenerated, not read.

## Voxel generator (`voxel.py`)

```python
def voxel_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), mask=None) -> Mesh
```

- **Coordinates:** $x_{ijk} = o + h \odot (i, j, k)$, computed from the integer indices.
- **Indexing:** cell $(i, j, k)$ has flat index $i + n_x(j + n_y k)$, so $x$ varies fastest. Grid node $(i, j, k)$ has flat index $i + (n_x+1)(j + (n_y+1)k)$.
- **Mask:**
  - `mask[i, j, k]` is a bool array of shape `shape`; `None` means a full grid.
  - The code works on `mask.transpose(2, 1, 0)` (C-order `(nz, ny, nx)`), so flat order equals `mask.ravel(order="F")`.
- **Output:**
  - Active cells become `hex8` elements in grid order, with VTK corner offsets $[0, 1, 1+s_y, s_y, s_z, s_z+1, s_z+1+s_y, s_z+s_y]$, where $s_y = n_x+1$ and $s_z = (n_x+1)(n_y+1)$.
  - Used nodes are kept in grid order.
  - Labels are 0, and there are no sets.
- **Errors:** these raise `ValueError`:
  - `shape` entries that aren't positive integers;
  - `spacing`/`origin` that aren't length 3;
  - spacing ≤ 0;
  - a wrong mask shape or dtype;
  - an all-false mask.
- **Mask** must be `dtype == bool` (after `np.asarray`); 0/1 int masks raise.
- **Performance** (measured on the implementation): 88³ (2.1M DOFs) builds in 0.04 s, `corner_jacobians` 0.09 s. 320³ (99M DOFs): build 1.8 s, `corner_jacobians` 3.8 s (chunked rows), peak 7.9 GB.

## Tests (`tests/test_mesh.py`)

The VTK reference coordinates `REF[type]` are a literal table in the test file; it is the test oracle (PR 3's registry owns the library's copy). Each raise test mutates a base object that constructs successfully in the same test, and uses `match=`. Warnings are recorded with `warnings.catch_warnings(record=True)` + `simplefilter("always")` and compared as exact lists. Gmsh ordering fixtures are hand-written (every element entity listed in `$Entities`). All other Gmsh fixtures (messy, `.inp`/`.msh` pair, raise cases) are generated once with `gmsh` 4.15.2 by a committed script (`tests/data/gen_gmsh.py`, not run by the tests; `Mesh.SecondOrderIncomplete=1` set before meshing).

1. **`REF` checked against the topology rules.**
   - hex8: node 0 at $(-1,-1,-1)$, node 1 at $+\xi$, the bottom face counter-clockwise seen from $+z$, node $4+i$ = node $i + 2e_\zeta$.
   - hex20: rows 8–19 are the edge midpoints (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7).
   - hex27: rows 20–25 are the face means for $-x, +x, -y, +y, -z, +z$, row 26 is the origin, and the first 20 rows equal hex20.
2. **Read orderings against hand-written files.**
   - The test writes one element per format and type into `tmp_path` (text templates in the test file, so mutants are a one-line change):
     - Gmsh 4.1 ASCII for hex8, hex20 and hex27, with `$Entities`, physical tags and non-contiguous node tags;
     - Abaqus `C3D8`, `C3D8R` and `C3D20`;
     - VTU for hex8, hex20 and hex27.
   - The node coordinates are $A\xi + b$, with $\xi$ taken from each format's documented order: tables transcribed into the test file from the Gmsh, Abaqus and VTK docs, with citations, never generated by meshio.
   - After `read`, `points[conn]` equals $A\,\mathrm{REF} + b$ to 1e-12. The Gmsh material equals the physical tag.
   - One mutant per format (two midside nodes swapped in a copy of the hex20 file) makes the check fail.
   - A Gmsh-exported `.inp` (hex20, `Mesh.SaveGroupsOfNodes=1`, with CPS8/T3D3 sections) and the `.msh` of the same mesh are committed. Reading both gives equal `points[conn]` and an equal named lower-dimensional group (`clamp`) as coordinate sets. The `.inp` set keys and warnings (skipped types, dropped labels, dropped empty element sets) are asserted exactly, as measured from the generated file.
3. **VTU round trip.**
   - The written file's cell-data keys are exactly `material`, `orientation`, int64.
   - hex8 + hex20 + hex27 with shared nodes, non-identity numbering, random non-integer coordinates, and multi-digit material and orientation labels. Written and read back: points `np.array_equal` (exact, measured), `list(blocks)` and the arrays, and labels exactly.
   - A mesh with sets reads back with `node_sets == {}` and `element_sets == {}`.
   - A float32-points VTU written by meshio reads as float64.
4. **Messy input, checked geometrically.**
   - **Gmsh 4.1** (generated by `gen_gmsh.py`): two fragmented boxes, hex8;
     - named volume (3, 7) and unnamed volume (3, 8): materials 7 and 8;
     - named surface "clamp" (2, 3), and unnamed (2, 4) covering the clamp face plus another face: both node sets complete;
     - a named line group "edge" (1, 3), the same tag as clamp in another dimension: its own node set; `node_sets` keys asserted exactly;
     - a surface group whose nodes are all on hexes, so no warnings; orphan handling is covered by VTU.
   - **VTU** (written by meshio): blocks in the order `quad, hexahedron20, quad, hexahedron, vertex` (hex20 first, distinct per-element materials, checked by centroid after reordering); two hex blocks of different sizes; stray quads whose nodes sit in the middle of the point array; points equal the used input points in order.
   - **Abaqus**, one hand-written file containing:
     - two `*NODE` sections, one with `NSET=`;
     - unsorted, non-contiguous labels;
     - lowercase and mixed-case keywords, whitespace around `=`, a quoted set name;
     - `**` comments, including one between continuation lines;
     - interleaved sections `C3D20`, `c3d8r`, `C3D20R`;
     - an `S4` section with `ELSET=` (ignored), whose labels also appear in a user `*ELSET` (dropped labels, warned with a count), and whose nodes are partly in one `NSET` (restricted, warned) and wholly in another (dropped as empty, warned); a second `S4` section (still one warning for the type);
     - a `DC3D8` section (skipped, not read as hex8);
     - `*MATERIAL`, `*ELASTIC`, `*SOLID SECTION` and `*ORIENTATION` with data lines between element sections;
     - a repeated `*ELSET` and `*NSET`, first spelled `Fix` then `FIX` (key `Fix`), sharing a member;
     - `GENERATE` on both set types, one with step 2 and one without a step;
     - trailing commas;
     - `MATERIAL_3` in mixed case, `material_5` given via `*ELEMENT, ELSET=`, disjoint `material_7`/`material_007`, and an `orientation_2` set;
     - the user `*ELSET` holding S4 labels also holds a kept element;
     - an element in no material set;
     - a set named `material_abc`.
   - **References are coordinates, not indices:**
     - `points` equals the input points of used nodes, in input order;
     - each element's `points[conn]`;
     - materials and orientations looked up by centroid;
     - node sets compared as coordinate sets, and `node_sets` keys exactly;
     - element sets compared by centroid, and `element_sets` keys exactly (`{}` for Gmsh);
     - the exact warning list: skipped `S4`, skipped `DC3D8`, dropped element-set labels, restricted node set, dropped empty node set.
5. **Corner Jacobians.**
   - The affine elements of test 2 give $\det A$ at all corners, and voxels give $\prod(\text{spacing})/8$.
   - `det A` and $\prod h/8$ to rtol 1e-12.
   - A seeded rotation of the reference cube with node perturbations 0.1–0.2 (asserting `np.ptp(ref) > 1e-2`) matches, per corner, the independent reference $\det\big(\sum_i x_i \nabla N_i(\hat\xi_a)^\top\big)$ with $N_i = \tfrac18\prod_d (1 + \hat\xi_{i,d}\xi_d)$ from `REF`.
   - The affine files use $\det A > 0$ (rotation times positive diagonal plus a small shear).
   - A hex20 with perturbed corners and displaced midside nodes equals `corner_jacobians` of its `conn[:, :8]` hex8 and the oracle.
   - A mirrored hex8 gives negative values.
   - Every `tests/data` file is positive after `read`.
6. **Voxel generator.**
   - **Sphere:** `shape = (20, 22, 24)`, spacing `(0.5, 0.4, 0.3)`, mask = cell centres within $0.45\min(\text{extent})$ of the grid centre, in physical space:
     - the element count equals `mask.sum()`;
     - the node count equals a brute-force Python set of corner grid indices;
     - $\sum_e 8\cdot\overline{\det J_e}$ equals `mask.sum() * prod(spacing)` (rtol 1e-12).
   - **Asymmetric mask:** a seeded random mask on `(3, 4, 5)` with spacing `(0.5, 0.7, 1.3)` and origin `(1.0, -2.0, 0.5)`, then set explicitly: a corner-only contact, an edge-only contact, and an inactive cell whose 8 nodes are all used. A triple Python loop is the reference:
     - `points[conn]` per active cell (atol 1e-12);
     - `element_map >= 0` equals `mask.ravel(order="F")`;
     - the non-negative entries of both maps are `arange` in grid order;
     - `node_map` is −1 exactly at unused grid nodes;
     - `voxel.shape`, `spacing` and `origin` equal the inputs;
     - the inverse maps and the mask view from the docs are right;
     - `mask=None` gives a full grid.
   - **Permutation:** a random node permutation and a random hex8 row permutation, applied by the documented checklist (labels, an added element set, both maps), construct (grid order is not an invariant); `points[node_map[node_map >= 0]]` still equals the grid coordinates, and the inverse element map still gives each element's grid cell.
7. **Validation.** Each raises, one parametrized case each:
   - `Mesh` construction:
     - dtypes/types: int points, a `jax.Array`, int32 connectivity, float labels;
     - points of shape `(n, 2)`; NaN in points; a hex8 block with 7 columns; `blocks={}`; a zero-row block;
     - an unknown key, or keys out of order; a non-str set key; a 2-D set; `voxel` not a `Voxel`;
     - an index of −1 or `n_nodes`; an orphan node;
     - unsorted, duplicate or out-of-range indices, for node sets (`n_nodes`) and element sets (`n_el`);
     - a `material` or `orientation` length mismatch;
     - voxel: a hex20 block, two blocks, a wrong map length, a float map, spacing 0, spacing NaN, origin inf, shape of length 2, shape `(True,)*3`; and on both maps, each bijection condition alone (an entry of `n`, an entry of −2, a missing entry, a duplicate plus a missing entry).
     - Positive cases: an empty node set constructs; negative labels construct; a hex8 + hex27 mesh has `offsets == {"hex8": 0, "hex27": n8}` and the right `n_el`.
   - Reading Abaqus files with each of:
     - each keyword of the reject tuple, and `*NSET, ELSET=`;
     - a set-name reference;
     - `C3D27`; a missing `TYPE=`;
     - an undefined node label in connectivity, or in a node set;
     - a duplicate node label; a duplicate element label across types (C3D8 and C3D20);
     - an element in `material_1` and `material_2`; in `material_7` and `MATERIAL_007`; in two orientation sets; `material_99999999999`;
     - a node line with 2 or 4 coordinates;
     - an element line with too many entries; an element cut off by the next keyword; one cut off by end of file;
     - only skipped element types.
   - Reading Gmsh files with:
     - a volume in two groups; an untagged volume next to a tagged one; SaveAll with partial tags;
     - version 2.2, version `4 0 8`, binary 4.1; partitioned; surfaces only; a line and a surface both named `clamp`; a surface explicitly named `physical_2_4` next to unnamed (2, 4).
   - Reading a VTU with a tetra block, only quads, or non-integral float labels; integral float labels read as int64.
   - Truncated `.msh` (cut inside `$Elements`) and `.vtu` files raise an `Exception`, not `SystemExit`; an unknown extension raises `ValueError`; writing `.inp` raises.
   - `voxel_mesh` with bad shape (0, 2.5, `(True,)*3`; `np.int64` and int spacing accepted, normalized to `int`/`float`), spacing, origin, mask shape, an int or float mask, or an all-false mask.
   - Also tested: a 4.1 file without physical groups, and one without `$Entities` (meshio-written with `gmsh:physical = 5`), read with material 0.
8. **Size.** A full 88³ voxel grid (about 2.1M DOFs) builds, validates and runs `corner_jacobians` in under 2 s, timing only the two calls (validation runs inside `voxel_mesh`; measured 0.13 s; catches a Python loop over elements). Not marked slow.

## Docs

- **`docs/mesh.md`:**
  - **Model:** the fields, the global index and `offsets`, the labels, and the permutation checklist. Also the inverse maps, the orphan rule, and how to add sets to voxel meshes.
  - **Ordering:** VTK ordering tables with reference coordinates, and the corner-pair Jacobian formula (a sign check for hex20/hex27).
  - **Formats:** what each format reads; VTU is the only written format:
    - Abaqus: flat `.inp` only (CAE rejected), the `material_<k>`/`orientation_<k>` sets, sections not read, suffixes ignored, skipped types;
    - Gmsh: 4.1 ASCII only (and why: 2.2 silently drops untagged volumes), physical groups by `(dim, tag)`, exactly one physical group per volume, no SaveAll;
    - VTU: `material`/`orientation` cell data, no sets.
  - **Voxels:** grid indexing in LaTeX, and the mask layout.
- **`docs/conventions.md`** and **ROADMAP Principle 7:** the same index wording (JAX int32; host NumPy int64 until PR 4).
- **`README.md`:** a usage snippet (`jaxmodal.mesh.read`, `voxel_mesh`), and the Status line updated.
- **`docs/ROADMAP.md`:**
  - PR 1 Contents: "The `Mesh` carries grid size and spacing" → "grid shape, spacing, origin and grid↔mesh node/element maps"; PR 6: the node map comes from PR 1 and PR 6 checks it against the geometry.
  - PR 1 Contents: "meshio readers for Gmsh and VTU, plus an own flat-`.inp` reader; VTU writer", "node orderings: VTK internally; only meshio's Gmsh permutation is needed (Abaqus C3D20 = VTK)", and "VTU read → write → read round trip".
  - Principle 4: replace "per-format node-ordering maps live in `mesh/io`, and the registry refers to them" with "any per-format permutation lives in `mesh/io`; none is needed today (meshio permutes Gmsh)". Package layout: "io (incl. node-ordering maps)" → "io".
  - Dependencies note: "meshio's `.inp` reader is unusable (see `docs/plans/pr-01-mesh.md`); PR 1 adds a flat-`.inp` reader. Gmsh input must be 4.1 ASCII."
  - Principle 7 bullet "Index arrays are always int32" and the conventions "Indices" bullet are replaced by the index wording above; conventions node-ordering line: add "(only meshio's Gmsh permutation is needed today)".
  - A status note is added after the merge.

## Acceptance

- `scripts/check.sh` and `uv run --locked pytest -m "not gpu"` pass.
- Code review rounds come back clean.
- Docs and memory are updated.
- Merge with `--no-ff`.
