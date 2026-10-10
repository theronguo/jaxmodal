# PR 2 plan: materials

Branch `pr-02-materials`, from `main` after PR 1. Scope: the roadmap's PR 2 (Contents, Docs, Tests), plus the mapping from PR 1's `material`/`orientation` labels to a per-phase property table, which `docs/mesh.md` promises to PR 2/6. Review round 1 is incorporated.

## Measured facts (prototype, jax 0.11.2, fp64)

- Orthotropic $C = S^{-1}$ via `jnp.linalg.inv`: $\|C^{-1} - S\|_\infty / \|S\|_\infty$ is 5e-18 (carbon/epoxy, $\kappa(C) = 42$) and 4e-17 (a stiff orthotropic set, $\kappa = 5$).
- Bond rotation $M C M^\top$ vs. `einsum` rotation of the 4th-order tensor: 3e-17 and 1.4e-16 relative.
- Isotropic $C$ under a random rotation: 1.5e-15 relative.

So the 1e-14 tolerances below hold with about one order of margin (reviewers measured up to 1.5e-15).
- Central finite differences of $\sum W \odot MCM^\top$ (carbon/epoxy in GPa, best of three steps): 1.5e-10 relative over the gradient for the constants, 1.3e-9 for the Euler angles.
- `rotate` under `jit` differs from eager by 1.5e-18 relative (not bitwise).

## Files

```
src/jaxmodal/materials/__init__.py   re-exports the public names below
src/jaxmodal/materials/elastic.py    constructors, Material, check_stiffness, phases, phase_table, check_materials
src/jaxmodal/materials/rotation.py   rotations, Bond matrices, rotate
tests/test_materials.py
docs/materials.md
```

## Conventions

- Voigt order $[11, 22, 33, 23, 13, 12]$, engineering shear strains (`docs/conventions.md`). $\sigma = C\varepsilon$.
- **Rotation $R$** (3×3, proper orthogonal): its columns are the local material axes $e_1', e_2', e_3'$ in global coordinates, so $x_\text{global} = R\,x_\text{local}$ and $\sigma_\text{global} = R\,\sigma_\text{local}R^\top$.
- Units are the caller's (consistent SI recommended). Nothing is non-dimensionalized.
- Constructors, rotations and `phase_table` are pure `jax.numpy` functions: traceable, differentiable, `jit`/`vmap`-able. They cast inputs with `jnp.asarray(x, float)` so integer or Python-scalar inputs give floating arrays. Validation that needs concrete values lives in the host functions `check_stiffness`, `check_materials` and `phases`.

## `elastic.py`

```python
class Material(NamedTuple):   # a pytree
    C: jax.Array    # (6, 6) local stiffness
    rho: jax.Array  # () density

isotropic(E, nu) -> C
transversely_isotropic(E1, E2, nu12, nu23, G12) -> C   # symmetry axis = local 1
orthotropic(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23) -> C
anisotropic(C) -> C   # (6, 6), returned as a float array
check_stiffness(C)    # host: raises ValueError unless symmetric and PD
```

- **isotropic:** Lamé form, $\lambda = E\nu/((1+\nu)(1-2\nu))$, $\mu = E/(2(1+\nu))$.
- **orthotropic:** builds the compliance $S$ and returns `jnp.linalg.inv(S)`. $\nu_{ij} = -\varepsilon_j/\varepsilon_i$ under uniaxial $\sigma_i$, with $\nu_{ij}/E_i = \nu_{ji}/E_j$; inputs are the three $\nu_{ij}$ with $i < j$. $S_{11} = 1/E_1$, $S_{12} = -\nu_{12}/E_1$, $S_{13} = -\nu_{13}/E_1$, $S_{23} = -\nu_{23}/E_2$, $S_{44} = 1/G_{23}$, $S_{55} = 1/G_{13}$, $S_{66} = 1/G_{12}$.
- **transversely_isotropic:** `orthotropic(E1, E2, E2, nu12, nu12, nu23, G12, G12, E2 / (2 (1 + nu23)))`: isotropic in the local 2–3 plane (fiber along local 1). Other symmetry axes come from an orientation.
- **anisotropic:** a (6, 6) input is returned as a float array, otherwise unchanged (asymmetry is a `check_stiffness` error, not silently fixed).
- **check_stiffness** (NumPy, concrete input): shape (6, 6), finite, $\max|C - C^\top| \le 10^{-12}\max|C|$, and $\lambda_\min(C) > 10^{-12}\lambda_\max(C)$ (roadmap, `np.linalg.eigvalsh`). Constructors don't call it (they must stay traceable). Rotated $C$ is not re-checked: congruence with the checked $R$ preserves PD.
- **Density** is the `rho` field of `Material` (checked $> 0$ and finite by `check_materials`). It is the physical density of the material; PR 23's design densities enter through PR 5/6's $s_K$, $s_M$, not through `rho`.

## `rotation.py`

```python
from_euler(angles) -> R          # (..., 3) Bunge z-x-z, radians
from_vectors(a, b) -> R          # (..., 3), (..., 3)
bond_stress(R) -> M              # (..., 6, 6): sigma_g = M sigma_l (Voigt)
bond_strain(R) -> N              # (..., 6, 6): eps_g = N eps_l (engineering shear); N = M^{-T}
rotate(C, R) -> M C M^T          # local -> global stiffness, batched over leading axes
```

- **from_euler:** $R = R_z(\varphi_1)\,R_x(\Phi)\,R_z(\varphi_2)$ (intrinsic z–x′–z″, Bunge). Columns are the local axes.
- **from_euler** equals SciPy's `Rotation.from_euler("ZXZ", angles).as_matrix()` (intrinsic). Note: with columns as local axes, $R$ is the transpose of the texture-analysis matrix $g$ (rows = crystal axes); the docs say so.
- **from_vectors** (Abaqus `*ORIENTATION, SYSTEM=RECTANGULAR` convention): $e_1' = a/|a|$, $e_3' = (a \times b)/|a \times b|$, $e_2' = e_3' \times e_1'$. $b$ is any vector in the local 1–2 plane, not parallel to $a$. Parallel or zero vectors give NaN, and gradients are undefined there (traceable code cannot raise); `check_materials` checks every $R$ for finiteness, $\max|R^\top R - I| \le 10^{-12}$ and $\det R > 0$.
- **Per-element orientations** need no extra API: `from_euler`, `from_vectors` and `rotate` are batched, so an (n_el, 3, 3) $R$ field rotates an (n_el, 6, 6) $C$ field directly. Only the label route below is per phase.
- **Bond matrices:** for the Voigt pair $(i,j)$ (row) and $(k,l)$ (column), $M_{(ij),(kl)} = R_{ik}R_{jl}$ if $k = l$, else $R_{ik}R_{jl} + R_{il}R_{jk}$. With $D = \mathrm{diag}(1,1,1,2,2,2)$ (engineering shear), $N = D M D^{-1}$; this equals $M^{-\top}$ because $R$ is orthogonal (tested, not assumed). $M$ is built by indexing $R$ with a static (6, 2) Voigt index table, so both are vectorized over leading axes.
- **rotate:** $C_g = M C_l M^\top$. Under congruence, PD is preserved and symmetry exact up to rounding.

## Phases (in `elastic.py`)

Topology (host, fixed) and material values (traced) are separate, so the table can be built inside `jit`/`grad` (Principle 3; PR 21 differentiates it).

```python
phases(mesh) -> (index, pairs)          # host NumPy
phase_table(pairs, materials, orientations=None) -> (C, rho)   # pure jnp
check_materials(pairs, materials, orientations=None)           # host
```

- **phases:** a phase is a distinct (material label, orientation label) pair present in the mesh. `pairs` is a tuple of $P$ `(int, int)` tuples (hashable, so it can be a `static_argnames` argument of a jitted function; no closure needed), in order of first appearance in the global element order; `index` is (n_el,) int64, element → phase (host index arrays stay int64 until PR 4, `docs/conventions.md`). Built with `np.unique(..., axis=0, return_index=True, return_inverse=True)` and a reorder by first index.
- **phase_table:** `materials` is a dict label → `Material`, `orientations` a dict label → $R$ (3×3). Orientation label 0 is the identity unless given; `orientations=None` means all identity. Material label 0 has no default: voxel meshes (all label 0) need `materials = {0: ...}`. Returns `C` (P, 6, 6) = `rotate(stack of C_l, stack of R)` and `rho` (P,); each `Material` field and $R$ goes through `jnp.asarray(·, float)` first (no weak-type retraces from Python-float `rho`). Pure `jnp.stack` over the pairs (a static Python loop over P, which is small), so `jax.grad` flows from `C`, `rho` back to the constants and angles; memory is $36P$ floats. A label missing from a dict fails as a plain dict lookup (`KeyError`, raised while tracing); no extra error path.
- **check_materials:** host validation of the same inputs: missing labels (`ValueError` naming the label), `check_stiffness` on each used local $C$, $\rho > 0$ and finite, each used $R$ a rotation. PR 9's `solve_modes` calls it once before tracing, so `solve_modes` takes concrete materials (solvers are never differentiated; PR 21 differentiates `phase_table` and the element kernels directly). Extra dict entries are allowed.
- PR 5 gathers `C[index]` per chunk. PR 6 builds one reference Ke per phase; the roadmap's PR 6 restriction is reworded accordingly (see Docs).

## Tests (`tests/test_materials.py`)

Tolerances follow Principle 1 (relative $\|\cdot\|_\infty$ over entries).

1. **Bond = einsum:** for 5 random rotations and a random SPD anisotropic $C$, `rotate(C, R)` equals the `einsum("ia,jb,kc,ld,abcd->ijkl")` rotation of the 4th-order tensor (built from Voigt by a test-local helper with the Voigt↔tensor index maps written out) to 1e-14. Also `bond_strain(R) == inv(bond_stress(R)).T` to 1e-14; `bond_strain(R) @ voigt(eps)` equals the engineering-Voigt vector of $R\varepsilon R^\top$ for a random symmetric $\varepsilon$ (1e-14); and the work identity $\sigma_g\cdot\varepsilon_g = \sigma_l\cdot\varepsilon_l$ to 1e-14.
2. **Invariance:** isotropic $C$ unchanged by 5 random rotations (1e-14); transversely isotropic $C$ unchanged by rotations about local axis 1 (random angles), and changed by a 90° rotation about axis 3 (a mutant check with carbon/epoxy constants: relative change > 0.1).
3. **PD:** `check_stiffness` passes on all constructors with realistic constants (steel, carbon/epoxy, a cubic crystal via `anisotropic`) and on their rotations; it raises on: $\nu = 0.5$ isotropic (non-finite), $\nu = 0.5 - 10^{-14}$ (finite, fails the $\lambda_\min$ check), $\nu_{12}$ beyond the orthotropic bound (indefinite), an asymmetric $C$, a NaN entry, a (6, 5) shape, and $C = Q\,\mathrm{diag}(t, 0.3, 0.5, 0.7, 0.9, 1)\,Q^\top \cdot 200\text{e}9$ (random orthogonal $Q$, symmetrized) with $t = 10^{-13}$ (raises) and $t = 10^{-11}$ (passes), over 20 random $Q$.
4. **Compliance:** `inv(orthotropic(...))` equals the hand-built $S$ (entries typed out in the test from $E_i$, $\nu_{ij}$, $G_{ij}$) to 1e-14, for two constant sets with $E_1 \ne E_2 \ne E_3$, $\nu_{12} \ne \nu_{13} \ne \nu_{23}$ and distinct $G_{ij}$ (so swapped $\nu_{ij}/\nu_{ji}$ or $G_{13}/G_{23}$ change $S$); the reciprocal relation $S_{21} = -\nu_{21}/E_2$ with $\nu_{21} = \nu_{12}E_2/E_1$ is checked explicitly. Isotropic equals `orthotropic` with equal constants and $G = E/(2(1+\nu))$; transversely isotropic equals its orthotropic expansion.
5. **Orientations:**
   - `from_euler` equals SciPy `Rotation.from_euler("ZXZ", angles).as_matrix()` for 5 random triples (1e-14); known cases typed by hand: zero angles → $I$; $(\pi/2, 0, 0)$ → $e_1' = e_y$; $(\pi/2, \pi/2, 0)$ → columns $(e_y, e_z, e_x)$.
   - `from_vectors(a, b)` equals `from_euler(angles)` for $a = R e_1$ scaled by 3 and $b = 2Re_1 + 5Re_2$ (any in-plane $b$), for 5 random angle triples (1e-14).
   - Batched shapes: (n, 3) inputs give (n, 3, 3), equal to a loop; `rotate` on (n, 6, 6) and (n, 3, 3) equals a loop.
6. **Phases:**
   - A mesh with `material = [5, -1, 5, 0]`, `orientation = [0, 2, 2, 0]` gives `pairs = [[5, 0], [-1, 2], [5, 2], [0, 0]]` and `index = [0, 1, 2, 3]`; with `[5, 5, -1, 5]`, `[0, 0, 2, 0]` it gives 2 phases and `index = [0, 0, 1, 0]`. `C[index[e]]` equals `rotate(materials[m_e].C, R_{o_e})` per element.
   - `check_materials` raises on: missing material label (message names it), missing orientation label ≠ 0, non-PD material, `rho = 0`, `rho = nan`, a reflection $R$ ($\det = -1$), NaN $R$ (from parallel vectors). `phase_table` raises `KeyError` for a missing label.
   - Real meshes: `tests/data/gmsh_messy.msh` (material labels 7 and 8) and an inline flat `.inp` with `material_<k>` and `orientation_<k>` sets (written in the test) give the expected pairs.
   - `jax.jit(phase_table, static_argnames="pairs")` equals eager to 1e-15, and changing an angle does not retrace (jit cache size stays 1); a `Material` passes through `jax.jit` as a pytree; `jax.vmap(isotropic)` over (n,) constants gives (n, 6, 6) equal to a loop.
7. **Gradients:** central finite differences (step $h\max(|x|, 1)$ per parameter, $h \in \{10^{-4}, 10^{-5}, 10^{-6}\}$, best taken) of $f = \sum W \odot C_g$ (fixed random $W$), error $\|g_\text{fd} - g\|_\infty / \|g\|_\infty$ over the whole gradient vector, with respect to:
   - $(E, \nu)$ of `isotropic` and all 9 constants of `orthotropic` (scaled to $O(1)$ by using GPa-free units in the test);
   - the three Euler angles and the 6 vector components of `from_vectors`;
   - through `phase_table`: $f(C) + \sum \rho$ w.r.t. one Euler angle, one orthotropic constant and one density used to build the dicts.
   Error ≤ 1e-6. `jax.jit` of each constructor and of `rotate` equals eager to 1e-15.

## Docs

- `docs/materials.md`: conventions ($R$ columns, and that this is $g^\top$ in texture notation; Voigt; engineering strain), each constructor's formula (Lamé, $S$ entries, transversely isotropic expansion), the orthotropic stability bounds (PD of $C$ ⇔ $E_i, G_{ij} > 0$, $|\nu_{ij}| < \sqrt{E_i/E_j}$ and $1 - \nu_{12}\nu_{21} - \nu_{23}\nu_{32} - \nu_{13}\nu_{31} - 2\nu_{21}\nu_{32}\nu_{13} > 0$, with $\nu_{ji} = \nu_{ij}E_j/E_i$ derived), Bond matrices $M$ and $N = DMD^{-1}$, $C_g = MCM^\top$, Euler and vector orientations, per-element $R$ fields, phases (label 0 rules, density vs. $s_K$/$s_M$), and an example from mesh labels to `(index, C, rho)`.
- `README.md`: status line and a short materials snippet (run in a test, like PR 1's).
- `docs/ROADMAP.md`, on the branch before the merge:
  - PR 2 Contents: "Orientations: local material axes per orientation label, from Euler angles or two direction vectors; per-element $R$ fields through the batched functions" and "Phases: mesh labels → (index, pairs) on the host; a traceable per-phase table of rotated C and ρ".
  - PR 6: "Restriction: one reference Ke per phase, a (material, orientation) pair", and its test "two materials or orientations in one phase raise" becomes "a two-phase voxel mesh (two materials, or one material in two orientations) matches PR 5".
  - PR 9: `solve_modes(mesh, materials, bcs, n_modes, orientations=None, ...)`.
  - Status note "PR 2 merged: ..." written just before the merge.
- `docs/mesh.md`: the "PR 2/6 map them" sentence points to `phases`/`phase_table`.

## Out of scope

- A mesh-level API for per-element orientation fields: the batched functions already rotate per element; wiring such a field into the operators comes with the first example that needs it (PR 16/23).
- Reading Abaqus `*MATERIAL`/`*ORIENTATION` cards (PR 1 reads only labels).
- Damping, temperature dependence, nonlinearity.

## Checklist before merge

- `pytest -m "not gpu"` passes; ruff clean; `scripts/check.sh`.
- Code review rounds (3–5 reviewers) until no substantive findings.
- Docs and memory updated.
