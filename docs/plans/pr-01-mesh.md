# PR 1 plan: mesh data model and parser

Branch `pr-01-mesh`, from `main` after PR 0. Scope: the roadmap's PR 1 (Contents, Docs, Tests). It is host-side NumPy only, with no JAX and no element kernels (PR 3). Review rounds 1–3 are incorporated.

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
- **Gmsh:**
  - **2.2:** keeps per-element `gmsh:physical`, and `$PhysicalNames` arrive as `field_data` (name → (tag, dim)).
  - **2.2, entity in several groups:** each element is written once per group, each row carrying one tag.
  - **4.1 read:** `gmsh:physical` keeps only the first tag of each entity, and meshio builds `cell_sets` only for named groups (plus a `gmsh:bounding_entities` key). So an unnamed second group is lost entirely. `gmsh:geometrical` (the entity tag) is present on every block, and `$Entities` lists every entity's physical tags, including volumes that Gmsh left out of the file.
  - **4.1 with partially tagged entities (SaveAll):** meshio raises `ValueError` (incompatible `gmsh:physical` blocks).
  - **2.2 with SaveAll:** every physical tag is 0, and `field_data` still lists named groups. With only unnamed groups this case cannot be detected.
- **VTU:**
  - `cell_data` round-trips.
  - Bool arrays are rejected.
  - Given `point_sets`/`cell_sets`, meshio mutates the caller's mesh and writes −1/index arrays.
  - Points round-trip bit-exactly (ASCII and binary).
- **Cell classification:** meshio cell names vary (`VTK_LAGRANGE_HEXAHEDRON`, `polygon`), but `CellBlock.dim` is correct for all of them.

## Files

```
src/jaxmodal/mesh/__init__.py   re-exports Mesh, Voxel, read, write, voxel_mesh, corner_jacobians
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
- **Material and orientation IDs** are labels, not table indices, and may be non-contiguous (e.g. Gmsh physical tags). PR 2/6 map them to materials.
- **dtypes:** the arrays must be `np.ndarray` (host-side only) with exactly these dtypes (float64 points; int64 connectivity, sets, labels and voxel maps), or construction raises. Readers and `voxel_mesh` cast before constructing; label casts first check that values are integral.
- **Index rule:** `docs/conventions.md` and ROADMAP Principle 7 both get the same wording: index arrays passed to JAX are int32; host-side NumPy topology is int64 until PR 4's overflow asserts and cast.
- **`__post_init__` validation**, vectorized and O(n), raises `ValueError` on:
  - wrong types, shapes or dtypes (sets and maps are 1-D);
  - no elements, or a block with zero rows;
  - a block key not in `TYPES`, or keys out of order;
  - the wrong node count for a type;
  - an index outside `[0, n_nodes)`;
  - an orphan node (`seen[conn] = True; seen.all()`);
  - set indices that aren't sorted, unique and in range (empty sets are allowed);
  - label lengths ≠ `n_el`;
  - for `voxel`:
    - anything other than a single `hex8` block;
    - `shape` not 3 positive ints; `spacing`/`origin` not 3 finite floats; spacing ≤ 0;
    - map lengths that don't match `shape`;
    - maps that aren't bijections onto the mesh: entries in `[-1, n)`, the non-negative count equal to `n`, and `bincount(...).max() == 1`.
- **Grid order is a property of `voxel_mesh` output only**, not a `Mesh` invariant. This keeps PR 5 RCM and PR 18 reordering legal. A node or element permutation must update `blocks`, the sets (re-sorted), the labels and both voxel maps; `docs/mesh.md` lists these. Element permutations stay within a block, since the global index is fixed by `TYPES` order.
- **Inverse maps** work under any permutation:
  - mesh → grid nodes: `inv = np.empty(n, np.int64); inv[m[m >= 0]] = np.flatnonzero(m >= 0)`, and the same for elements (PR 23 scatters densities this way);
  - mask: `(element_map >= 0).reshape(shape, order="F")`.
- **What `Mesh` does not check:** whether the voxel maps agree with `points` and the connectivity. PR 6 checks that before dispatching to the stencil, as the first consumer that relies on regular geometry.
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

- `read(path) -> Mesh` dispatches on the extension (`.inp`, `.msh`, `.vtu`); `write(path, mesh)` accepts `.vtu` only. Other extensions raise `ValueError`.
- Gmsh and Abaqus are read-only: no later PR writes them (PR 17 exports XDMF), and the writers would bring their own edge cases (tag 0, set-name collisions, line limits).
- Every reader raises `ValueError` when no 3D element is kept, and drops sets that end up empty, with a warning.

### Abaqus (own reader, flat `.inp` only)

About 80 lines. It reads the whole file, then resolves labels.

- **Lines:**
  - `**` lines are comments, including between continuation lines.
  - Keywords and parameters are case-insensitive, with whitespace around `=` and quotes around names stripped. A missing `TYPE=` raises `ValueError`.
  - Empty tokens from trailing commas are dropped.
- **`*NODE`:** any number of sections; an optional `NSET=` adds the section's nodes to that set; every data line has a label and exactly 3 coordinates.
- **`*ELEMENT, TYPE=`:**
  - `re.fullmatch(r"C3D(8|20)[A-Z]*")` → `hex8`/`hex20`. Suffixes are ignored, since the project uses full integration; the docs say so.
  - Any other `C3D*` raises.
  - Other types (S4, CPS8, T3D3, DC3D8, SC8R, …) are skipped: their data lines are ignored, with one warning per type.
  - Kept elements may span continuation lines; an element is complete at its node count. More entries than that, or an element cut off by the next keyword or end of file, raise.
  - `ELSET=` adds the kept elements to that set.
- **`*NSET` / `*ELSET`:**
  - integer lists, or `GENERATE` (start, stop, optional step defaulting to 1);
  - names are case-insensitive, keyed by their upper case, and keep the first spelling;
  - a repeated name appends; members are `np.unique`d.
- **Label resolution** after reading:
  - node and element labels may be non-contiguous and in any order, and are mapped to indices in file order;
  - an undefined or duplicate node label raises (connectivity or node sets), and so does a duplicate element label;
  - element-set labels that aren't kept elements are dropped (they can't be told apart from skipped elements), with one warning and a count.
- **These keywords raise `ValueError`** (one tuple):
  - `*PART`, `*INSTANCE`, `*ASSEMBLY` and `*INCLUDE` (Abaqus/CAE exports are rejected; the docs say so);
  - geometry-creating keywords: `*NGEN`, `*NFILL`, `*NCOPY`, `*NMAP`, `*ELGEN`, `*ELCOPY` and `*SYSTEM`.
  - Also raising: a set name used inside another set, and `*NSET, ELSET=`.
- **Other keywords are ignored, data lines included.** This covers `*MATERIAL`, `*ELASTIC`, `*SOLID SECTION` and `*ORIENTATION`; the docs state that real section assignments are not read.
- **Materials and orientations:**
  - They come from sets named `material_<k>`/`orientation_<k>`, matched case-insensitively with $k$ a non-negative integer.
  - Elements in none get 0, and an element in two raises.
  - These sets are removed from `element_sets`. Names like `material_abc` stay ordinary sets.

### Gmsh (`meshio.gmsh.read`, ASCII 2.2 and 4.1)

- **Group membership** per cell, as a list of `(dim, tag)` with tag 0 ignored:
  - **2.2:** from `gmsh:physical` (one tag per row; multi-group entities are duplicated rows).
  - **4.1:** from our own parse of `$Entities` (about 12 lines): `(dim, entity) → [physical tags]`, looked up through `gmsh:geometrical`. Binary 4.1 raises "save as ASCII".
- **Names:** from the inverted `field_data` `{(dim, tag): name}`, falling back to `physical_<dim>_<tag>`. A duplicate resulting name raises.
- **Node sets:** each group with `dim < 3` becomes a node set, the union of its cells' nodes. 3D groups don't become element sets.
- **Materials:** a 3D element's material label is its physical tag, or 0 when the file has no physical groups. Orientations are 0.
- **These raise `ValueError`** with the message "put every volume in exactly one physical group and don't use Mesh.SaveAll":
  - duplicate 3D connectivity rows (2.2, `np.unique(np.sort(conn, 1), axis=0)`);
  - a 3D entity with more than one physical tag (4.1);
  - a 3D entity without a physical tag while others have one (4.1; Gmsh left its elements out);
  - `field_data` listing groups while every physical tag is 0 (2.2 with SaveAll);
  - meshio's "Incompatible cell data" `ValueError` (4.1 with SaveAll).
- **Docs:** name your physical groups and never use SaveAll; 2.2 + SaveAll + only unnamed groups cannot be detected and reads as material 0 with no sets.

### VTU (`meshio.vtu.read`, `meshio.vtu.write`)

- **Read:** labels come from the `material`/`orientation` cell data (integral values required), defaulting to 0. VTU carries no sets.
- **Write:** writes only our `material`/`orientation` arrays (int64). Sets are not written; the docs say so.

### Shared read path (Gmsh, VTU; orphan removal also Abaqus)

- **Cell classification** by `CellBlock.dim`:
  - meshio `hexahedron`/`hexahedron20`/`hexahedron27` → `hex8`/`hex20`/`hex27`;
  - `dim < 3` cells are dropped after node-set extraction;
  - any other `dim == 3` type raises.
- **Same-type blocks** are concatenated in file order, with labels carried along.
- **Orphan removal:** nodes not used by kept cells are removed, and the rest keep their original order. Node sets are restricted to surviving nodes, with a warning when members are dropped.
- **dtypes:** points are cast to float64 and indices to int64.
- **Voxel metadata** is not written. Voxel meshes are regenerated, not read.

## Voxel generator (`voxel.py`)

```python
def voxel_mesh(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), mask=None) -> Mesh
```

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
- **Performance** (measured in a prototype): 88³ (2.1M DOFs) builds in 0.03 s, validates in 0.01 s, `corner_jacobians` 0.3 s. 320³ (99M DOFs): build about 2 s, `corner_jacobians` about 10 s, peak about 13–14 GB.

## Tests (`tests/test_mesh.py`)

The VTK reference coordinates `REF[type]` are a literal table in the test file; it is the test oracle (PR 3's registry owns the library's copy). Each raise test mutates a base object that constructs successfully in the same test, and uses `match=`. Each warning is checked with `pytest.warns`.

1. **`REF` checked against the topology rules.**
   - hex8: node 0 at $(-1,-1,-1)$, node 1 at $+\xi$, the bottom face counter-clockwise seen from $+z$, then the top face.
   - hex20: rows 8–19 are the edge midpoints (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7).
   - hex27: rows 20–25 are the face means for $-x, +x, -y, +y, -z, +z$, row 26 is the origin, and the first 20 rows equal hex20.
2. **Read orderings against hand-written files.**
   - `tests/data/` holds one element per format and type:
     - Gmsh 2.2 and 4.1 ASCII for hex8, hex20 and hex27, the 4.1 files with `$Entities`, physical tags and non-contiguous node tags;
     - Abaqus `C3D8`, `C3D8R` and `C3D20`;
     - VTU for hex8, hex20 and hex27.
   - The node coordinates are $A\xi + b$, with $\xi$ taken from each format's documented order: tables transcribed into the test file from the Gmsh, Abaqus and VTK docs, with citations, never generated by meshio.
   - After `read`, `points[conn]` equals $A\,\mathrm{REF} + b$ to 1e-12. The Gmsh material equals the physical tag.
   - One mutant per format (two midside nodes swapped in a copy of the hex20 file) makes the check fail.
   - A Gmsh-exported `.inp` (hex20, `Mesh.SaveGroupsOfNodes=1`, with CPS8/T3D3 sections) and the `.msh` of the same mesh are committed. Reading both gives equal `points[conn]` and an equal named lower-dimensional group (`clamp`) as coordinate sets. The extra `.inp` set keys and the dropped-element-set warning are asserted exactly.
3. **VTU round trip.**
   - hex8 + hex20 + hex27 with shared nodes, non-identity numbering, random non-integer coordinates, and multi-digit material and orientation labels. Written and read back: points `np.array_equal` (exact, measured), `list(blocks)` and the arrays, and labels exactly.
   - A mesh with sets reads back with empty sets.
   - A float32-points VTU written by meshio reads as float64.
4. **Messy input, checked geometrically.**
   - **Gmsh 2.2** (written by meshio, `fmt_version="2.2"`):
     - blocks in the order `quad, hexahedron, quad, hexahedron, vertex`;
     - two hex blocks of different sizes and materials;
     - "clamp" quads lying on hex faces, with tags (2, 7) and a `$PhysicalName`;
     - unnamed quads with tag (2, 8), named `physical_2_8`;
     - stray quads whose nodes sit in the middle of the point array and aren't used by any hex, tagged (2, 9): a set losing members and a set left empty (dropped);
     - one line cell with tag (1, 7), the same number as clamp in another dimension;
     - the vertex block with physical tag 0.
   - **Gmsh 4.1:** a hand-written file with a named physical surface in a second, unnamed group too, and a physical volume. Node sets of both groups are complete.
   - **VTU:** the same blocks as the Gmsh 2.2 case, with no `material` data, so labels read as 0.
   - **Abaqus**, one hand-written file containing:
     - two `*NODE` sections, one with `NSET=`;
     - unsorted, non-contiguous labels;
     - lowercase and mixed-case keywords, whitespace around `=`, a quoted set name;
     - `**` comments, including one between continuation lines;
     - a `C3D20` section before a `c3d8r` section, and `C3D20R`;
     - an `S4` section with `ELSET=`, whose labels also appear in a user `*ELSET`, and whose nodes are partly in an `NSET` (orphan removal drops them);
     - a `DC3D8` section (skipped, not read as hex8);
     - `*MATERIAL`, `*ELASTIC`, `*SOLID SECTION` and `*ORIENTATION` with data lines between element sections;
     - a repeated `*ELSET` and `*NSET`, each repeat in a different case;
     - `GENERATE` on both set types, one with step 2 and one without a step;
     - trailing commas;
     - `MATERIAL_3` in mixed case, `material_5` given via `*ELEMENT, ELSET=`, and an `orientation_2` set;
     - an element in no material set;
     - a set named `material_abc`.
   - **References are coordinates, not indices:**
     - `points` equals the input points of used nodes, in input order;
     - each element's `points[conn]`;
     - materials and orientations looked up by centroid;
     - node sets compared as coordinate sets, and `node_sets` keys exactly;
     - element sets compared by centroid, and `element_sets` keys exactly (`{}` for Gmsh);
     - every warning (skipped types, dropped labels, restricted and dropped sets).
5. **Corner Jacobians.**
   - The affine elements of test 2 give $\det A$ at all corners, and voxels give $\prod(\text{spacing})/8$.
   - A rotated and perturbed hex8 matches the analytic trilinear derivatives.
   - A mirrored hex8 gives negative values.
   - Every `tests/data` file is positive after `read`.
6. **Voxel generator.**
   - **Sphere:** `shape = (20, 22, 24)`, spacing `(0.5, 0.4, 0.3)`, mask = cell centres within $0.45\min(\text{extent})$ of the grid centre, in physical space:
     - the element count equals `mask.sum()`;
     - the node count equals a brute-force Python set of corner grid indices;
     - $\sum_e 8\cdot\overline{\det J_e}$ equals `mask.sum() * prod(spacing)` to 1e-12.
   - **Asymmetric mask:** a seeded random mask on `(3, 4, 5)` with a nonzero origin, including cells that touch only at an edge or a corner. A triple Python loop is the reference:
     - `points[conn]` per active cell;
     - `element_map >= 0` equals `mask.ravel(order="F")`;
     - the non-negative entries of both maps are `arange` in grid order;
     - `node_map` is −1 exactly at unused grid nodes;
     - `voxel.shape`, `spacing` and `origin` equal the inputs;
     - the inverse maps and the mask view from the docs are right;
     - `mask=None` gives a full grid.
   - **Permutation:** a random node permutation applied by the documented checklist constructs (grid order is not an invariant), and `points[node_map[node_map >= 0]]` still equals the grid coordinates of those nodes.
7. **Validation.** Each of these raises, one parametrized case each:
   - `Mesh` construction:
     - int points, a `jax.Array`, or points of shape `(n, 2)`;
     - float or int32 connectivity; int32 set indices; float labels;
     - a hex8 block with 7 columns; `blocks={}`; a zero-row block;
     - an unknown key, or keys out of order;
     - an index of −1 or `n_nodes`;
     - an orphan node;
     - unsorted, duplicate or out-of-range set indices;
     - a `material` or `orientation` length mismatch;
     - a voxel mesh with a hex20 block, two blocks, a wrong map length, a float map, bad `shape`/`spacing`; and each bijection condition alone (an entry of `n`, a missing entry, a duplicate plus a missing entry).
     - A two-block mesh's `offsets` and `n_el` are asserted.
   - Reading Abaqus files with each of:
     - each keyword in the reject tuple (parametrized over the tuple);
     - a set-name reference; `*NSET, ELSET=`;
     - `C3D27`; a missing `TYPE=`;
     - an undefined node label in connectivity, or in a node set;
     - a duplicate node or element label;
     - an element in two material sets, or in two orientation sets;
     - a node line with 2 coordinates;
     - an element line with too many entries; an element cut off by the next keyword;
     - only skipped element types (no 3D elements).
   - Reading Gmsh files with:
     - duplicate 3D rows (a volume in two groups, 2.2);
     - a volume in two unnamed groups (4.1);
     - an untagged volume next to a tagged one (4.1);
     - SaveAll 2.2 (all tags 0, groups listed); SaveAll 4.1;
     - binary 4.1; surfaces only; two groups with the same name in different dimensions.
   - Reading a VTU with a tetra block, or with non-integral float labels.
   - Truncated `.msh` and `.vtu` files raise an `Exception`, not `SystemExit`; an unknown extension raises `ValueError`; writing `.inp` raises.
   - `voxel_mesh` with bad shape (0, 2.5; `np.int64` accepted), spacing, origin, mask shape or dtype, or an all-false mask.
   - Also tested: a `.msh` without physical groups reads with material 0.
8. **Size.** A full 88³ voxel grid (about 2.1M DOFs) builds, validates and runs `corner_jacobians` in under 2 s (measured 0.3 s; catches a Python loop over elements). Not marked slow.

## Docs

- **`docs/mesh.md`:**
  - **Model:** the fields, the global index and `offsets`, the labels, and the permutation checklist. Also the inverse maps, the orphan rule, and how to add sets to voxel meshes.
  - **Ordering:** VTK ordering tables with reference coordinates, and the corner-pair Jacobian formula (a sign check for hex20/hex27).
  - **Formats:** what each format reads; VTU is the only written format:
    - Abaqus: flat `.inp` only (CAE rejected), the `material_<k>`/`orientation_<k>` sets, sections not read, suffixes ignored, skipped types;
    - Gmsh: ASCII only, physical groups by `(dim, tag)`, name your groups, exactly one physical group per volume, no SaveAll (and the undetectable 2.2 case);
    - VTU: `material`/`orientation` cell data, no sets.
  - **Voxels:** grid indexing in LaTeX, and the mask layout.
- **`docs/conventions.md`** and **ROADMAP Principle 7:** the same index wording (JAX int32; host NumPy int64 until PR 4).
- **`README.md`:** a usage snippet (`jaxmodal.mesh.read`, `voxel_mesh`), and the Status line updated.
- **`docs/ROADMAP.md`:**
  - PR 1 Contents: "meshio readers for Gmsh and VTU, plus an own flat-`.inp` reader; VTU writer", "node orderings: VTK internally; only meshio's Gmsh permutation is needed (Abaqus C3D20 = VTK)", and "VTU read → write → read round trip".
  - Principle 4 and the package layout line: replace "per-format node-ordering maps live in `mesh/io`, and the registry refers to them" with "any per-format permutation lives in `mesh/io`; none is needed today (meshio permutes Gmsh)".
  - The Dependencies note on the custom reader is updated to match.
  - A status note is added after the merge.

## Acceptance

- `scripts/check.sh` and `uv run --locked pytest -m "not gpu"` pass.
- Code review rounds come back clean.
- Docs and memory are updated.
- Merge with `--no-ff`.
