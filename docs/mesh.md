# Mesh

`jaxmodal.mesh` holds the host-side mesh: a NumPy data model, file readers, a VTU writer, voxel and box generators, corner Jacobians, renumbering (`renumber`, `rcm`), the solid-shell conversion (`solid_shell`) and face-connected parts (`parts`, `docs/boundary_conditions.md`). There is no JAX here (a test imports it in a fresh interpreter and checks that `jax` is not loaded); `element_dofs` casts topology to int32 for the device (`docs/operators.md`).

```python
from jaxmodal.mesh import read, write, voxel_mesh, corner_jacobians

mesh = read("part.inp")  # .inp (flat Abaqus), .msh (Gmsh 4.1 ASCII) or .vtu
mesh.blocks["hex20"]  # (n_el_type, 20) int64 connectivity, VTK node order
write("part.vtu", mesh)  # VTU is the only written format
```

## Data model

```python
@dataclass(frozen=True, eq=False)
class Mesh:
    points: np.ndarray  # (n_nodes, 3) float64
    blocks: dict  # type -> (n_el_type, n_per_el) int64, keys in TYPES order
    node_sets: dict  # name -> sorted unique node indices, int64
    element_sets: dict  # name -> sorted unique global element indices, int64
    material: np.ndarray  # (n_el,) int64 label
    orientation: np.ndarray  # (n_el,) int64 label
    voxel: Voxel | None = None
```

- **Types:** `TYPES = ("hex8", "hex20", "hex27", "ssh8")`. Blocks appear in this order. `ssh8` is the solid shell: hex8 topology with reference $\zeta$ as the thickness ([Solid shells](#solid-shells-solid_shell)).
- **Global element index:** blocks in `TYPES` order, rows in order within a block. `mesh.offsets[t]` is the first global index of block `t` (present types only), and `mesh.n_el` the total.
- **Labels:** `material` and `orientation` are labels (any int64, possibly negative or non-contiguous, e.g. Gmsh physical tags), not table indices. `jaxmodal.materials.phases` and `phase_table` map them to materials and orientations (`docs/materials.md`). Label 0 is the default.
- **dtypes are strict:** the arrays must be `np.ndarray` with exactly these dtypes, or construction raises `ValueError`. Nothing is cast silently, so there are no hidden copies at 100M-node scale.
- **Validation** (in `__post_init__`, O(n)): shapes and dtypes; finite points; at least one element and no empty block; node counts per type; node indices in range; no orphan nodes; sets sorted, unique and in range (empty sets are allowed); label lengths; the voxel fields below.
- **Not checked:** degenerate elements (repeated nodes) construct; `corner_jacobians` shows 0 at the collapsed corners.
- **Immutability:** treat every field, including the dicts, as immutable. Change a mesh with `dataclasses.replace`, which validates again.
- **Orphan rule:** every node belongs to an element. A mesh derived from another (e.g. PR 12's hex8 corner mesh of a hex20 mesh) must compact its nodes.

### Renumbering

Grid order is not an invariant, so node and element permutations (PR 5 RCM, PR 18 Morton) are legal if they update every dependent field:

- **Node permutation** $p$ (new → old): `points[p]`; connectivity through $p^{-1}$; `node_sets` mapped through $p^{-1}$ and re-sorted; non-negative `voxel.node_map` entries mapped through $p^{-1}$.
- **Element permutation** $q$ (new → old) within one block (the global index is fixed by `TYPES` order): block rows, `material` and `orientation` indexed by $q$; `element_sets` mapped through $q^{-1}$ and re-sorted; non-negative `voxel.element_map` entries mapped through $q^{-1}$.

`renumber(mesh, p, q)` applies both rules (`p`: `(n_nodes,)`, `q`: `(n_el,)` global, both new → old) and returns a new, validated `Mesh`; it raises if `q` moves an element out of its block. `rcm(mesh)` returns `(mesh, p, q)`:

- Node order: reverse Cuthill–McKee (`scipy.sparse.csgraph.reverse_cuthill_mckee`, `symmetric_mode=True`, which handles disconnected meshes) of the node graph $A = E^\top E$, with $E$ the sparse element–node incidence of all blocks.
- Element order: within each block, stably sorted by the element's min (new) node index.
- **When it helps:** for arbitrarily ordered meshes (Gmsh, Abaqus). The grid order of `box_mesh` and `voxel_mesh` is already narrower: node-graph bandwidth on a hex8 $20^3$ box is 463 in grid order, 1261 after RCM and 9235 randomly shuffled. Neither the matrix-free operator nor its gather path requires RCM (`docs/operators.md`). Do not renumber voxel meshes for `gmg`: it needs the `voxel_mesh` node order and raises otherwise (`docs/preconditioners.md`).
- **Cost:** at 10M hex20 DOF, $A$ has 196M nonzeros (≈ 1.6 GB on the host: int32 values and column indices) and RCM takes about 3 s.
- User arrays follow the new numbering: per-element arrays as `a[q]`, per-node arrays as `a[p]`.

## Node ordering

All types use VTK ordering on the reference cube $[-1, 1]^3$:

| nodes | position |
|---|---|
| 0–3 | bottom face $\zeta = -1$: $(-1,-1)$, $(1,-1)$, $(1,1)$, $(-1,1)$ in $(\xi, \eta)$, counter-clockwise seen from $+z$ |
| 4–7 | top face: node $4+i$ = node $i$ + $2e_\zeta$ |
| 8–19 (hex20, hex27) | edge midpoints (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7) |
| 20–25 (hex27) | face centres $-\xi$, $+\xi$, $-\eta$, $+\eta$, $-\zeta$, $+\zeta$ |
| 26 (hex27) | centre |

`mesh/model.py` holds these tables as `CORNERS` (8, 3), `EDGES` (the 12 corner pairs above), `FACE_CORNERS` (corners of each face, counter-clockwise seen from outside) and `REFERENCE_NODES` (27, 3); `jaxmodal.elements` builds its registry from them, and `box_mesh` builds its connectivity from them.

Abaqus C3D20 order equals VTK hex20. meshio permutes Gmsh hex20/hex27 to VTK. No other permutation is needed; one would go into `mesh/io.py` if a later format needs it.

## Corner Jacobians

`corner_jacobians(mesh)` returns, per type, an `(n_el_type, 8)` array of $\det J$ at the corners of the trilinear map through the first 8 nodes. At corner $a$, column $d$ of $J_a$ is

$$J_a e_d = \tfrac12\left(x_{a_d^+} - x_{a_d^-}\right),$$

where $(a_d^-, a_d^+)$ is the pair of corners along reference axis $d$ that contains $a$:

- $\xi$: (0,1), (3,2), (4,5), (7,6);
- $\eta$: (0,3), (1,2), (4,7), (5,6);
- $\zeta$: (0,4), (1,5), (2,6), (3,7).

This is the exact trilinear derivative at a corner. For an affine element $x = A\xi + b$, $\det J_a = \det A$; for a voxel with spacing $h$, $\det J_a = h_x h_y h_z / 8$. For hex20/hex27 it is only a sign and orientation check on the corner hex; curved-element validity at quadrature points is `check_elements` (`docs/elements.md`).

## Formats

`read(path)` dispatches on the extension. Every reader raises `ValueError` on 3D cell types other than hex8/hex20/hex27 (tet, wedge, ...) and when no 3D element is kept, removes orphan nodes (keeping the order of the rest), restricts node sets to surviving nodes (warning) and drops sets that end up empty (warning). Blocks are put in `TYPES` order; labels follow their rows.

### Abaqus `.inp` (flat files only)

meshio's `.inp` reader is unusable (wrong `ELSET=` blocks, repeated sets replace instead of append, crashes on a second `*NODE` and on lowercase or unknown types), so jaxmodal has its own reader.

- **Supported:** `*NODE` (any number, optional `NSET=`), `*ELEMENT, TYPE=` (optional `ELSET=`, continuation lines), `*NSET`/`*ELSET` (integer lists or `GENERATE`; repeated names append), `**` comments, case-insensitive keywords and set names (the first spelling becomes the key), non-contiguous labels in any order. Keyword lines cannot continue onto the next line.
- **Types:** `C3D8*` → `hex8` and `C3D20*` → `hex20`. Suffixes (R, H, ...) are ignored because jaxmodal uses full integration. Other `C3D*` types raise. Non-solid types (S4, CPS8, T3D3, DC3D8, ...) are skipped with one warning per type; `ELSET=` on a skipped section is ignored, and element-set labels of skipped elements are dropped with a warning.
- **Materials and orientations** come from element sets named `material_<k>` / `orientation_<k>` (any case; $k$ a non-negative integer below $2^{31}$, so `material_007` is label 7). Elements in none get 0 (so `material_0` is the default label); an element in two sets of a kind raises. These sets are removed from `element_sets`.
- **Not read:** `*MATERIAL`, `*SOLID SECTION`, `*ORIENTATION` and other keywords are skipped with their data lines. Real section assignments are not used.
- **Rejected:** Abaqus/CAE structure (`*PART`, `*INSTANCE`, `*ASSEMBLY`, `*INCLUDE`), generated geometry (`*NGEN`, `*NFILL`, `*NCOPY`, `*NMAP`, `*ELGEN`, `*ELCOPY`, `*SYSTEM`), set names inside sets, `*NSET, ELSET=`, `SYSTEM=`/`INPUT=` on `*NODE`/`*ELEMENT`, `*ELEMENT` without `TYPE=`, `*NSET`/`*ELSET` without a name, labels of $2^{31}$ or more in magnitude, duplicate node or element labels, undefined node labels (in elements or node sets), malformed data lines. Element-set labels that name no solid element (skipped or undefined) are dropped with a warning.
- **Parsing:** each section's data lines are parsed in bulk, so an error names the section, not the line. A node or element may continue over lines but must end at a line end; all three node coordinates are required, and empty fields (`1,,2`) are not supported. Empty `*NODE`/`*ELEMENT` sections are ignored; an empty set is dropped with the usual warning.

### Gmsh `.msh` (4.1 ASCII only)

Save with `Mesh.MshFileVersion = 4.1` and `Mesh.Binary = 0`. Other versions raise: in 2.2 a volume outside every physical group is silently left out of the file, and nothing in the file shows it.

- **Physical groups** are keyed by `(dim, tag)`, since tags are unique only within a dimension. jaxmodal parses `$PhysicalNames` and `$Entities` itself, because meshio keeps only the first group of each entity and loses unnamed groups.
- **Node sets:** every group of dimension < 3 becomes a node set named after the group, or `physical_<dim>_<tag>` if unnamed. Duplicate names raise.
- **Materials:** a volume element's material label is its physical tag (0 if the file has no physical groups). Orientations are 0, and there are no element sets.
- **Rules:** put every volume in exactly one physical group, name your groups, and don't use `Mesh.SaveAll`. A volume in two groups, an untagged volume next to tagged ones, SaveAll with partial tags, binary or partitioned files all raise.

### VTU `.vtu`

- **Read:** `material` / `orientation` cell data (integer, or float with integral values, below $2^{31}$ in magnitude) become the labels, defaulting to 0. VTU carries no sets.
- **Write:** points, blocks and the two label arrays (int64). Sets and voxel metadata are not written. `ssh8` is written as a VTK hexahedron, so it reads back as hex8; on an all-`ssh8` mesh `solid_shell(mesh, direction=...)` restores it up to `order`, while on a mixed mesh the former `ssh8` elements must be selected again (e.g. by label). The VTU type field is deferred.

## Voxel meshes

```python
mesh = voxel_mesh(shape=(nx, ny, nz), spacing=(hx, hy, hz), origin=(ox, oy, oz), mask=None)
```

- **Grid:** cell $(i, j, k)$ has flat index $i + n_x(j + n_y k)$ ($x$ fastest). Grid node $(i, j, k)$ has flat index $i + (n_x+1)\bigl(j + (n_y+1)k\bigr)$ and position $x_{ijk} = o + h \odot (i, j, k)$.
- **Mask:** `mask[i, j, k]` is a bool array of shape `shape` selecting active cells; `None` means the full grid. Active cells become hex8 elements in grid order, and used nodes are kept in grid order. Labels are 0 and there are no element sets.
- **`mesh.voxel`** holds `shape`, `spacing`, `origin`, `node_map` (grid node → mesh node, −1 if dropped) and `element_map` (grid cell → mesh element, −1 if inactive). `Mesh` validates that a voxel mesh has exactly one hex8 block and that both maps are bijections onto the mesh; `stencil` checks that they agree with the geometry (points on the grid, connectivity equal to the grid corners in VTK order; `docs/operators.md`).
- **Inverse maps** (valid after renumbering):

  ```python
  inv = np.empty(n, np.int64)  # n = number of mesh nodes or elements
  inv[m[m >= 0]] = np.flatnonzero(m >= 0)  # mesh -> grid, for m = node_map or element_map
  mask = (mesh.voxel.element_map >= 0).reshape(shape, order="F")
  ```

- **Face sets:** node sets `"xmin"`, `"xmax"`, `"ymin"`, `"ymax"`, `"zmin"`, `"zmax"` hold the used nodes on the six bounding grid planes (from the integer grid index, so exact), e.g. `bcs = {"xmin": "xyz"}` clamps $x = o_x$. On a masked mesh these are the bounding planes of the grid, not the boundary of the occupied region: a set may be nearly or fully empty (e.g. a voxelized sphere), and then a BC on it constrains little or nothing.
- **Other boundary sets:** add them with `replace`, keeping the face sets:

  ```python
  clamp = np.flatnonzero(np.isclose(mesh.points[:, 0], origin[0]))
  mesh = dataclasses.replace(mesh, node_sets={**mesh.node_sets, "clamp": clamp})
  ```

- **Cost** (M-series laptop): $88^3$ cells (2.1M DOFs) build in 0.04 s and `corner_jacobians` takes 0.09 s. $320^3$ (99M DOFs) builds in 2.0 s and `corner_jacobians` takes 4.0 s (rows in chunks of $2^{20}$); peak memory is 8.5 GB. Coordinates and face sets are computed one axis at a time to keep temporaries bounded.

## Box meshes

```python
mesh = box_mesh(shape=(nx, ny, nz), spacing=(hx, hy, hz), origin=(ox, oy, oz), type="hex20")
mesh = box_mesh(shape, type="hex27", coords=(cx, cy, cz))  # graded
```

- A structured block of $n_x \times n_y \times n_z$ elements of type `hex8`, `hex20` or `hex27`, with shared nodes in VTK order. It is the test mesh for PRs 4, 5, 7 and 8 (`voxel_mesh` is hex8 only).
- **Nodes** are the used points of the half-spacing grid $(2n_x+1)\times(2n_y+1)\times(2n_z+1)$: hex8 uses even indices, hex20 indices with at most one odd component, hex27 all. Position $o + h \odot (i, j, k)/2$; nodes and elements in grid order ($x$ fastest), like `voxel_mesh`. Connectivity is the grid index of $2\,\text{cell} + \texttt{REFERENCE\_NODES}[:n] + 1$, one code path for all types.
- **Sets:** the six face node sets as for voxel meshes; no element sets; labels 0; no `voxel` metadata. The hex8 box equals the unmasked `voxel_mesh` on points, connectivity and node sets.
- **Validation:** `shape`, `spacing` and `origin` as for `voxel_mesh` (`check_grid`); a `type` other than hex8, hex20, hex27 raises (`ssh8` too: convert with `solid_shell`).
- **Graded meshes:** `coords=(cx, cy, cz)` gives the element boundaries per axis (lengths $n_d + 1$, finite, strictly increasing, else `ValueError`) and overrides `spacing` and `origin`. A node at half-grid index $i$ on axis $d$ sits at `np.interp(i / 2, arange(n_d + 1), coords[d])`: corner nodes on the boundaries, midside and centre nodes at element midpoints. Each element is thus an affine axis-aligned box, so geometry stays exact and refined meshes stay nested when their boundaries are. Remapping points instead would move midside nodes off the edge midpoints. The NAFEMS runs use cosine grading $x_i = \tfrac L2\big(1 - \cos(\pi i/n)\big)$ ([verification](verification.md)).
- **Cost** (M3 Max): hex8 $100^3$ (3.1M DOF) 0.44 s and 0.8 GB, hex20 $60^3$ (2.7M DOF) 0.22 s, hex27 $40^3$ (1.6M DOF) 0.11 s. It sorts all (element, node) grid indices (`np.unique`), so it is O(N log N) and about 6× slower than `voxel_mesh` for hex8; fine for test meshes, while a used-node bitmask as in `voxel_mesh` would be the route above ~10M DOF.

## Solid shells (`solid_shell`)

```python
mesh, order = solid_shell(mesh, elements=None, direction=None)
```

Converts hex8 elements to solid shells (`ssh8`, `docs/elements.md`) by permuting their connectivity so that reference $\zeta$ is the thickness.

- **Selection:** an element set name, an array of global element indices (duplicates and order ignored), or all hex8 when `None`. Non-hex8 elements in the selection, an empty selection and a mesh without hex8 elements raise `ValueError`.
- **Output:** the selected elements are appended to the `ssh8` block (created if absent), in sorted selection order; an emptied hex8 block is dropped. `material`, `orientation` and `element_sets` follow; node sets and points are unchanged; `voxel` becomes None. `order` is new → old over all elements, like `renumber`'s `q` but moving elements across blocks (map per-element arrays as `a[order]`). A second call converts another region, e.g. with another `direction`.
- **Permutations:** fixed proper rotations of the reference cube take the chosen axis to $\zeta$; the 180° turn about $\xi$ flips $\zeta$ (and $\eta$). New node $i$ is the old node at $Q^\top$`CORNERS[i]`, so $\det J$ keeps its sign.
- **`direction` given:** a (3,) vector, or `(n_sel, 3)` vectors in sorted selection order (e.g. radial directions on a cylinder); with an index array, it must then be sorted and unique, else `ValueError`. The thickness axis is the reference axis whose mean edge vector (over its 4 parallel edges) has the largest $|\cos|$ with it, and $\zeta$ points along $+$direction. If the two largest $|\cos|$ differ by $< 0.1$, `ValueError` names the element.
- **`direction=None` (sweep):** a chain runs from an element through the opposite faces of a reference axis, continuing in each face neighbour along that neighbour's axis through the shared face, and stopping at unselected elements and boundary faces; a closed ring counts its length once. Each chain is walked once ($O(n_\text{el})$) and its element count given to every member. The thickness axis is the unique smallest of an element's 3 counts; a tie raises `ValueError("ambiguous thickness direction: pass direction=")` (e.g. the $3^3$ cube, a $6\times1\times1$ strip, $8\times8\times12$). The count is topological: an $8\times10\times12$ box picks $x$ whatever the element shapes.
- **Sweep sign:** $\zeta$ along the positive global axis most aligned with the element's mean thickness edge, by a tolerant `argmax` (the first axis with $|e_i| \ge \max_j|e_j| - 10^{-8}\|e\|$), so round-off cannot split a flat plate (e.g. normal $(0,1,-1)/\sqrt2$). It is consistent on flat plates and may flip on curved shells.
- **When the sign matters:** under a $\zeta$ flip the plies are reversed, the angles negated and each ply turned 180° about an in-plane material axis. Pass `direction` for off-axis plies (even $[\pm45]_s$), non-orthotropic plies or unsymmetric stacking; this is documented, not enforced.
- **Not checked:** edge lengths (a thick element converts if its topology says so). Hand-built `ssh8` blocks are the user's responsibility; `check_elements` checks $\det J$ through the thickness.
