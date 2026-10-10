# Mesh

`jaxmodal.mesh` holds the host-side mesh: a NumPy data model, file readers, a VTU writer, a voxel generator and corner Jacobians. There is no JAX here; PR 4 casts topology to int32 for the device.

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

- **Types:** `TYPES = ("hex8", "hex20", "hex27")`. Blocks appear in this order.
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

## Node ordering

All types use VTK ordering on the reference cube $[-1, 1]^3$:

| nodes | position |
|---|---|
| 0–3 | bottom face $\zeta = -1$: $(-1,-1)$, $(1,-1)$, $(1,1)$, $(-1,1)$ in $(\xi, \eta)$, counter-clockwise seen from $+z$ |
| 4–7 | top face: node $4+i$ = node $i$ + $2e_\zeta$ |
| 8–19 (hex20, hex27) | edge midpoints (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7) |
| 20–25 (hex27) | face centres $-\xi$, $+\xi$, $-\eta$, $+\eta$, $-\zeta$, $+\zeta$ |
| 26 (hex27) | centre |

Abaqus C3D20 order equals VTK hex20. meshio permutes Gmsh hex20/hex27 to VTK. No other permutation is needed; one would go into `mesh/io.py` if a later format needs it.

## Corner Jacobians

`corner_jacobians(mesh)` returns, per type, an `(n_el_type, 8)` array of $\det J$ at the corners of the trilinear map through the first 8 nodes. At corner $a$, column $d$ of $J_a$ is

$$J_a e_d = \tfrac12\left(x_{a_d^+} - x_{a_d^-}\right),$$

where $(a_d^-, a_d^+)$ is the pair of corners along reference axis $d$ that contains $a$:

- $\xi$: (0,1), (3,2), (4,5), (7,6);
- $\eta$: (0,3), (1,2), (4,7), (5,6);
- $\zeta$: (0,4), (1,5), (2,6), (3,7).

This is the exact trilinear derivative at a corner. For an affine element $x = A\xi + b$, $\det J_a = \det A$; for a voxel with spacing $h$, $\det J_a = h_x h_y h_z / 8$. For hex20/hex27 it is only a sign and orientation check on the corner hex; curved-element validity at quadrature points is PR 3's check.

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
- **Write:** points, blocks and the two label arrays (int64). Sets and voxel metadata are not written.

## Voxel meshes

```python
mesh = voxel_mesh(shape=(nx, ny, nz), spacing=(hx, hy, hz), origin=(ox, oy, oz), mask=None)
```

- **Grid:** cell $(i, j, k)$ has flat index $i + n_x(j + n_y k)$ ($x$ fastest). Grid node $(i, j, k)$ has flat index $i + (n_x+1)\bigl(j + (n_y+1)k\bigr)$ and position $x_{ijk} = o + h \odot (i, j, k)$.
- **Mask:** `mask[i, j, k]` is a bool array of shape `shape` selecting active cells; `None` means the full grid. Active cells become hex8 elements in grid order, and used nodes are kept in grid order. Labels are 0 and there are no sets.
- **`mesh.voxel`** holds `shape`, `spacing`, `origin`, `node_map` (grid node → mesh node, −1 if dropped) and `element_map` (grid cell → mesh element, −1 if inactive). `Mesh` validates that a voxel mesh has exactly one hex8 block and that both maps are bijections onto the mesh; PR 6 checks that they agree with the geometry.
- **Inverse maps** (valid after renumbering):

  ```python
  inv = np.empty(n, np.int64)  # n = number of mesh nodes or elements
  inv[m[m >= 0]] = np.flatnonzero(m >= 0)  # mesh -> grid, for m = node_map or element_map
  mask = (mesh.voxel.element_map >= 0).reshape(shape, order="F")
  ```

- **Boundary sets:** add them with `replace`, e.g. a clamp at $x = o_x$:

  ```python
  clamp = np.flatnonzero(np.isclose(mesh.points[:, 0], origin[0]))
  mesh = dataclasses.replace(mesh, node_sets={"clamp": clamp})
  ```

- **Cost** (M-series laptop): $88^3$ cells (2.1M DOFs) build in 0.04 s and `corner_jacobians` takes 0.09 s. $320^3$ (99M DOFs) builds in 1.8 s and `corner_jacobians` takes 3.8 s (rows in chunks of $2^{20}$); peak memory is 7.9 GB.
