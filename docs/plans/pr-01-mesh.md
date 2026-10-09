# PR 1 plan: mesh data model and parser

Branch `pr-01-mesh`, from `main` after PR 0. Scope: the roadmap's PR 1 (Contents, Docs, Tests). It is host-side NumPy only, with no JAX and no element kernels (PR 3). Review rounds 1–2 are incorporated.

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
  - `offsets` and `n_el` give each block's slice. The writers and readers use them now, and PRs 4, 5, 16, 17, 21 and 23 will too.
- **Material and orientation IDs** are labels, not table indices, and may be non-contiguous (e.g. Gmsh physical tags). PR 2/6 map them to materials.
- **dtypes:** the fields must have exactly these dtypes (float64 points; int64 connectivity, sets and labels), or construction raises. Readers and `voxel_mesh` cast before constructing.
- **Index rule:** `docs/conventions.md` and ROADMAP Principle 7 both get the same wording: index arrays passed to JAX are int32; host-side NumPy topology is int64 until PR 4's overflow asserts and cast.
- **`__post_init__` validation**, vectorized and O(n), raises `ValueError` on:
  - wrong shapes or dtypes;
  - a block key not in `TYPES`, or keys out of order;
  - the wrong node count for a type;
  - an index outside `[0, n_nodes)`;
  - an orphan node (`seen[conn] = True; seen.all()`);
  - set indices that aren't sorted, unique and in range;
  - label lengths ≠ `n_el`;
  - for `voxel`:
    - anything other than a single `hex8` block;
    - map lengths that don't match `shape`;
    - maps that aren't bijections onto the mesh: entries in `[-1, n)`, the non-negative count equal to `n`, and `bincount(...).max() == 1`.
- **Grid order is a property of `voxel_mesh` output only**, not a `Mesh` invariant. This keeps PR 5 RCM and PR 18 reordering legal. A node or element permutation must update `blocks`, the sets (re-sorted), the labels and both voxel maps; `docs/mesh.md` lists these.
- **Inverse maps** work under any permutation:
  - mesh → grid nodes: `inv = np.empty(n, np.int64); inv[m[m >= 0]] = np.flatnonzero(m >= 0)`, and the same for elements (PR 23 scatters densities this way);
  - mask: `(element_map >= 0).reshape(shape, order="F")`.
- **What `Mesh` does not check:** whether the voxel maps agree with `points` and the connectivity. PR 6 checks that before dispatching to the stencil, as the first consumer that relies on regular geometry.
- **Orphan rule:** a mesh derived from another, e.g. PR 12's hex8 corner mesh from a hex20 mesh, must compact its nodes. This is by design and noted in the docs.
- **Voxel BCs:** voxel meshes have no sets. A clamp set is added with `replace(mesh, node_sets={"clamp": ...})`, as shown in the docs.
- **Solid-shell (PR 16)** stores the thickness direction by permuting each element's connectivity, so no mesh field is needed. Reading Abaqus `SC8R` is added to `io.py` by PR 16.
- **PR 3's registry:** its per-format ordering entry is the identity for all supported formats, since meshio handles Gmsh. `io.py` records this in one comment.
- **`corner_jacobians(mesh) -> dict[str, (n_el_type, 8) array]`:** the trilinear Jacobian determinant at each corner, from the first 8 nodes (the corners, in VTK order for every hex type).
  - At corner $a$, column $d$ of $J_a$ is $\tfrac12(x_{a_d^+} - x_{a_d^-})$, where $(a_d^-, a_d^+)$ is the pair of corners along reference axis $d$ that contains $a$, ordered by $\xi_d$. The pairs are:
    - $\xi$: (0,1), (3,2), (4,5), (7,6);
    - $\eta$: (0,3), (1,2), (4,7), (5,6);
    - $\zeta$: (0,4), (1,5), (2,6), (3,7).
  - For $x = A\xi + b$ on $[-1,1]^3$, $\det J_a = \det A$.
  - It is computed corner by corner as $d_0 \cdot (d_1 \times d_2)$, which keeps O(n) temporaries.

## IO (`io.py`)

`read(path) -> Mesh` and `write(path, mesh)` dispatch on the extension: `.inp`, `.msh`, `.vtu`.

### Abaqus (own reader and writer, flat `.inp` only)

- **Reader**, about 80 lines. It reads the whole file, then resolves labels.
  - **Lines:**
    - `**` lines are comments, including between continuation lines.
    - Keywords and parameters are case-insensitive, with whitespace around `=` and quotes around names stripped.
    - Empty tokens from trailing commas are dropped.
  - **`*NODE`:**
    - any number of sections;
    - an optional `NSET=` adds the section's nodes to that set;
    - every data line has a label and exactly 3 coordinates;
    - `SYSTEM=` other than `R` raises.
  - **`*ELEMENT, TYPE=`:**
    - `C3D8*` → `hex8` and `C3D20*` → `hex20`. Suffixes are ignored, since the project uses full integration; the docs say so.
    - Any other `C3D*` raises.
    - Other types (S4, CPS8, T3D3, DC3D8, SC8R, …) are skipped: their data lines are ignored, with one warning per type and its count.
    - Kept elements may span continuation lines; an element is complete at its node count, and more entries than that raise.
    - `ELSET=` adds the kept elements to that set.
  - **`*NSET` / `*ELSET`:**
    - integer lists, or `GENERATE` (start, stop, step);
    - names are case-insensitive, keyed by their upper case, and keep the first spelling;
    - a repeated name appends.
  - **Label resolution** after reading:
    - node and element labels may be non-contiguous and in any order, and are mapped to indices in file order;
    - an undefined or duplicate node label raises (connectivity or node sets), and so does a duplicate element label;
    - element-set labels that aren't kept elements are dropped (they can't be told apart from skipped elements), with one warning and a count;
    - element sets left empty are dropped.
  - **These keywords raise `ValueError`:**
    - `*PART`, `*INSTANCE`, `*ASSEMBLY` and `*INCLUDE` (Abaqus/CAE exports are rejected; the docs say so);
    - geometry-creating keywords: `*NGEN`, `*NFILL`, `*NCOPY`, `*NMAP`, `*ELGEN`, `*ELCOPY` and `*SYSTEM`;
    - a set name used inside another set;
    - `*NSET, ELSET=`.
  - **Other keywords are ignored, data lines included.** This covers `*MATERIAL`, `*ELASTIC`, `*SOLID SECTION` and `*ORIENTATION`; the docs state that real section assignments are not read.
  - **Materials and orientations:**
    - They come from sets named `material_<k>`/`orientation_<k>`, matched case-insensitively with $k$ a non-negative integer.
    - Elements in none get 0, and an element in two raises.
    - These sets are removed from `element_sets`. Names like `material_abc` stay ordinary sets.
- **Writer**, about 25 lines:
  - **Nodes:** `*NODE` with coordinates in `%.17g`.
  - **Elements:** one `*ELEMENT, TYPE=C3D8` or `TYPE=C3D20` section per block, with global 1-based labels. A C3D20 element is written as 16 entries plus a trailing comma, then the remaining 5 (as Gmsh writes them).
  - **Sets:**
    - `*NSET`/`*ELSET`, each written once, 8 labels per line;
    - `material_<k>`/`orientation_<k>` sets for every label present.
  - **Errors:** hex27, negative labels, and set names that aren't valid Abaqus names (letters, digits, `_`, starting with a letter) all raise.

### Gmsh (`meshio.gmsh.read`; `meshio.gmsh.write(..., fmt_version="2.2", binary=False)`)

- **Read:**
  - **Physical groups:** groups are keyed by `(dim, tag)`, since tags are unique only within a dimension, and named from the inverted `field_data`, falling back to `physical_<dim>_<tag>`. Tag 0 means "no group" and is ignored.
  - **Node sets:** each group with `dim < 3` becomes a node set, the union of its cells' nodes. The membership comes from `gmsh:physical` together with meshio's named `cell_sets` (4.1 writes an entity once with only its first tag; `cell_sets` hold the full membership).
  - **Materials:** a 3D element's material label is its physical tag, cast to int64, or 0 without physical groups. Orientations are 0. 3D groups don't become element sets.
  - **These raise `ValueError`** with the message "put every volume in exactly one physical group and don't use Mesh.SaveAll":
    - duplicate 3D connectivity rows (2.2 writes an element once per group);
    - a 3D cell in more than one named 3D `cell_set` (4.1);
    - `field_data` listing groups while every physical tag is 0 (2.2 with SaveAll);
    - meshio's "Incompatible cell data" `ValueError` (4.1 with SaveAll);
    - a file with no 3D elements (only surfaces were in groups).
  - **Docs:** untagged volumes are silently omitted by Gmsh unless SaveAll is set; the docs say so.
- **Write:**
  - only `gmsh:physical = material` and `gmsh:geometrical = material` are passed;
  - node sets, element sets or nonzero orientations each raise;
  - the docs say material 0 becomes physical tag 0, which Gmsh treats as "no group".

### VTU (`meshio.vtu.read`, `meshio.vtu.write`)

- **Read:** labels come from the `material`/`orientation` cell data, defaulting to 0. VTU carries no sets.
- **Write:** writes only our `material`/`orientation` arrays (int64). Sets are not written; the docs say so.

### Shared read path (Gmsh, VTU)

- **Cell classification** by `CellBlock.dim`:
  - meshio `hexahedron`/`hexahedron20`/`hexahedron27` → `hex8`/`hex20`/`hex27`;
  - `dim < 3` cells are dropped after node-set extraction;
  - any other `dim == 3` type raises.
- **Same-type blocks** are concatenated in file order, with labels carried along.
- **Orphan removal:** nodes not used by kept cells are removed, and the rest keep their original order. Node sets are restricted to surviving nodes, with a warning when members are dropped. The Abaqus reader applies the same orphan removal.
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
- **Performance** (measured in a prototype): 2.1M DOFs in 0.03 s; 99M DOFs in 1.5 s, 13.8 GB peak RSS including `corner_jacobians`.

## Tests (`tests/test_mesh.py`)

The VTK reference coordinates `REF[type]` live in the test file only. They are the test oracle; PR 3's registry owns the library's reference coordinates. Each raise test mutates a base object that constructs successfully in the same test, and uses `match=`.

1. **`REF` from topology alone.**
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
   - Parametrized mutants (swapped midside nodes, face centres and corners) are applied to copies of the files, and each makes the check fail.
   - A Gmsh-exported `.inp` (hex20, with CPS8/T3D3 sections and NSET/ELSET) and the `.msh` of the same mesh are committed. Reading both gives equal `points[conn]` and equal node sets.
3. **Round trips.**
   - A mesh is written and read back, then compared with the in-memory original: points to 1e-12, with random non-integer coordinates; `list(blocks)` and the arrays; labels; and the set keys and values exactly.
   - Each mesh has shared nodes, non-identity numbering and multi-digit labels. The format cases are:
     - Abaqus: hex8 + hex20, with node and element sets of more than 8 members, and material and orientation sets spanning both blocks;
     - Gmsh: hex8 + hex20 + hex27, materials only;
     - VTU: hex8 + hex20 + hex27 with materials and orientations, plus a float32-points VTU written by meshio.
   - The Abaqus output has `TYPE=C3D8`/`TYPE=C3D20` and no data line with more than 16 entries.
   - A VTU written from a mesh with sets reads back with empty sets.
4. **Messy input, checked geometrically.**
   - **Gmsh 2.2** (written by meshio, `fmt_version="2.2"`):
     - blocks in the order `quad, hexahedron, quad, hexahedron, vertex`;
     - two hex blocks of different sizes and materials;
     - "clamp" quads lying on hex faces, with tags (2, 7) and a `$PhysicalName`;
     - unnamed quads with tag (2, 8), named `physical_2_8`;
     - stray quads whose nodes sit in the middle of the point array and aren't used by any hex, tagged (2, 9) so a node set loses members;
     - one line cell with tag (1, 7), the same number as clamp in another dimension.
   - **Gmsh 4.1:** a hand-written file with a named physical surface, a physical volume and one surface in two groups.
   - **VTU:** the same blocks as the Gmsh 2.2 case, with no `material` data, so labels read as 0.
   - **Abaqus**, one hand-written file containing:
     - two `*NODE` sections, one with `NSET=`;
     - unsorted, non-contiguous labels;
     - lowercase and mixed-case keywords;
     - `**` comments, including one between continuation lines;
     - a `C3D20` section before a `c3d8r` section;
     - an `S4` section with `ELSET=`, whose labels also appear in a user `*ELSET`;
     - `C3D20R`;
     - `*MATERIAL`, `*ELASTIC`, `*SOLID SECTION` and `*ORIENTATION` with data lines between element sections;
     - a repeated `*ELSET` and a repeated `*NSET`;
     - `GENERATE` on both set types;
     - trailing commas;
     - `MATERIAL_3` in mixed case, plus `material_5` given via `*ELEMENT, ELSET=`;
     - an element in no material set;
     - a set named `material_abc`.
   - **References are coordinates, not indices:**
     - `points` equals the input points of used nodes, in input order;
     - each element's `points[conn]`;
     - materials looked up by centroid;
     - node sets compared as coordinate sets (clamp = the surviving clamp-quad nodes; the stray set restricted, with `pytest.warns`);
     - element sets compared by centroid (S4 labels dropped with a warning);
     - `element_sets` keys exactly;
     - skipped-type warnings.
5. **Corner Jacobians.**
   - The affine elements of test 2 give $\det A$ at all corners, and voxels give $\prod(\text{spacing})/8$.
   - A rotated and perturbed hex8 matches the analytic trilinear derivatives.
   - The same perturbed corners in a hex20 and a hex27 (with midside nodes moved) give the same values.
   - Mirroring a hex8 and a hex20 gives negative values.
   - Every `tests/data` file is positive after `read`.
6. **Voxel generator.**
   - **Sphere** of `shape = (20, 22, 24)`, spacing `(0.5, 0.4, 0.3)`, centred:
     - the element count equals `mask.sum()`;
     - the node count equals a brute-force Python set of corner grid indices;
     - $\sum_e 8\cdot\overline{\det J_e}$ equals `mask.sum() * prod(spacing)` to 1e-12.
   - **Asymmetric mask:** a seeded random mask on `(3, 4, 5)` with a nonzero origin, including cells that touch only at an edge or a corner. A triple Python loop is the reference:
     - `points[conn]` per active cell;
     - `element_map >= 0` equals `mask.ravel(order="F")`;
     - the non-negative map entries are `arange` in grid order;
     - `node_map` is −1 exactly at unused grid nodes;
     - `voxel.shape`, `spacing` and `origin` equal the inputs;
     - the inverse maps and the mask view from the docs are right;
     - `mask=None` gives a full grid.
   - **Permutation:** a node-permuted voxel mesh with all fields updated constructs. With stale, non-bijective maps it raises.
7. **Validation.** Each of these raises, one parametrized case each:
   - `Mesh` construction:
     - int points, or points of shape `(n, 2)`;
     - float or int32 connectivity;
     - a hex8 block with 7 columns;
     - an unknown key, or keys out of order;
     - an index of −1 or `n_nodes`;
     - an orphan node;
     - unsorted, duplicate or out-of-range set indices;
     - a `material` or `orientation` length mismatch;
     - a voxel mesh with a hex20 block, two blocks, a wrong map length, or a non-bijective map.
   - Reading Abaqus files with each of:
     - `*Part`, `*Instance`, `*Assembly` or `*Include`;
     - `*NGEN`, `*ELGEN` or `*SYSTEM`;
     - `*NODE, SYSTEM=C`;
     - a set-name reference;
     - `*NSET, ELSET=`;
     - `C3D27`;
     - an undefined node label in connectivity, or in a node set;
     - a duplicate node or element label;
     - an element in two material sets, or in two orientation sets;
     - a node line with 2 coordinates;
     - an element line with too many entries.
   - Reading Gmsh files with:
     - duplicate 3D rows (a volume in two groups, 2.2);
     - a 3D cell in two named groups (4.1);
     - SaveAll 2.2 (all tags 0, groups listed);
     - SaveAll 4.1;
     - surfaces only.
   - Reading a VTU with a tetra block.
   - Writing:
     - Gmsh with node sets, with element sets, or with orientations;
     - Abaqus with hex27, a negative label, or a set name containing a space.
   - `voxel_mesh` with bad shape, spacing, origin, mask shape or dtype, or an all-false mask.
   - Also tested: a `.msh` without physical groups reads with material 0.
8. **Size (slow).** A full 88³ voxel grid (about 2.1M DOFs) builds and validates in under 5 s.

## Docs

- **`docs/mesh.md`:**
  - **Model:** the fields, the global index and `offsets`, the labels, and the permutation checklist. Also the inverse maps, the orphan rule, and how to add sets to voxel meshes.
  - **Ordering:** VTK ordering tables with reference coordinates, and the corner-pair Jacobian formula.
  - **Formats:** what each format reads and writes:
    - Abaqus: flat `.inp` only (CAE rejected), the `material_<k>` sets, sections not read, suffixes ignored, skipped types;
    - Gmsh: physical groups by `(dim, tag)`, exactly one physical group per volume, no SaveAll, write as 2.2 with tag 0 caveat;
    - VTU: no sets.
  - **Voxels:** grid indexing in LaTeX, and the mask layout.
- **`docs/conventions.md`** and **ROADMAP Principle 7:** the same index wording (JAX int32; host NumPy int64 until PR 4).
- **`README.md`:** a usage snippet (`jaxmodal.mesh.read`, `voxel_mesh`).
- **`docs/ROADMAP.md`:**
  - PR 1 Contents: "meshio readers for Gmsh and VTU, plus an own flat-`.inp` reader and writer", and "node orderings: VTK internally; only meshio's Gmsh permutation is needed (Abaqus C3D20 = VTK)".
  - The Dependencies note on the custom reader is updated to match.
  - A status note is added after the merge.

## Acceptance

- `scripts/check.sh` and `uv run --locked pytest -m "not gpu"` pass.
- Code review rounds come back clean.
- Docs and memory are updated.
- Merge with `--no-ff`.
