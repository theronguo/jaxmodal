# PR 2 plan: materials

Branch `pr-02-materials`, from `main` after PR 1. Scope: the roadmap's PR 2 (Contents, Docs, Tests), plus the mapping from PR 1's `material`/`orientation` labels to per-element properties, which `docs/mesh.md` promises to PR 2/6.

## Measured facts (prototype, jax 0.11.2, fp64)

- Orthotropic $C = S^{-1}$ via `jnp.linalg.inv`: $\|C^{-1} - S\|_\infty / \|S\|_\infty$ is 5e-18 (carbon/epoxy, $\kappa(C) = 42$) and 4e-17 (a stiff orthotropic set, $\kappa = 5$).
- Bond rotation $M C M^\top$ vs. `einsum` rotation of the 4th-order tensor: 3e-17 and 1.4e-16 relative.
- Isotropic $C$ under a random rotation: 1.5e-15 relative.

So the 1e-14 tolerances below hold with two orders of margin.

## Files

```
src/jaxmodal/materials/__init__.py   re-exports the public names below
src/jaxmodal/materials/elastic.py    constructors, check_stiffness, Material
src/jaxmodal/materials/rotation.py   rotations, Bond matrices, rotate
src/jaxmodal/materials/assign.py     assign (mesh labels -> phase table)
tests/test_materials.py
docs/materials.md
```

## Conventions

- Voigt order $[11, 22, 33, 23, 13, 12]$, engineering shear strains (`docs/conventions.md`). $\sigma = C\varepsilon$.
- **Rotation $R$** (3×3, proper orthogonal): its columns are the local material axes $e_1', e_2', e_3'$ in global coordinates, so $x_\text{global} = R\,x_\text{local}$ and $\sigma_\text{global} = R\,\sigma_\text{local}R^\top$.
- Units are the caller's (consistent SI recommended). Nothing is non-dimensionalized.
- Everything in `elastic.py` and `rotation.py` is a pure `jax.numpy` function: traceable, differentiable, `jit`/`vmap`-able. Validation that needs concrete values lives in `check_stiffness` and `assign` (host side).

## `elastic.py`

```python
class Material(NamedTuple):   # a pytree
    C: jax.Array    # (6, 6) local stiffness
    rho: jax.Array  # () density

isotropic(E, nu) -> C
transversely_isotropic(E1, E2, nu12, nu23, G12) -> C   # symmetry axis = local 1
orthotropic(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23) -> C
anisotropic(C) -> C   # (6, 6) or 21 upper-triangle entries (row-major) -> symmetric (6, 6)
check_stiffness(C)    # host: raises ValueError unless symmetric and PD
```

- **isotropic:** Lamé form, $\lambda = E\nu/((1+\nu)(1-2\nu))$, $\mu = E/(2(1+\nu))$.
- **orthotropic:** builds the compliance $S$ and returns `jnp.linalg.inv(S)`. $\nu_{ij} = -\varepsilon_j/\varepsilon_i$ under uniaxial $\sigma_i$, with $\nu_{ij}/E_i = \nu_{ji}/E_j$; inputs are the three $\nu_{ij}$ with $i < j$. $S_{11} = 1/E_1$, $S_{12} = -\nu_{12}/E_1$, $S_{13} = -\nu_{13}/E_1$, $S_{23} = -\nu_{23}/E_2$, $S_{44} = 1/G_{23}$, $S_{55} = 1/G_{13}$, $S_{66} = 1/G_{12}$.
- **transversely_isotropic:** `orthotropic(E1, E2, E2, nu12, nu12, nu23, G12, G12, E2 / (2 (1 + nu23)))`: isotropic in the local 2–3 plane (fiber along local 1). Other symmetry axes come from an orientation.
- **anisotropic:** a (6, 6) input is returned unchanged (asymmetry is a `check_stiffness` error, not silently fixed); the 21-entry form fills both triangles. Keeping it a function gives all four constructors the same calling pattern for PR 21–23.
- **check_stiffness** (NumPy, concrete input): shape (6, 6), finite, symmetric to $10^{-12}\|C\|_\infty$ (absolute in entries), and $\lambda_\min(C) > 10^{-12}\lambda_\max(C)$ (roadmap). Constructors don't call it (they must stay traceable); `assign` calls it on every material, after rotation too.
- **Density** is the `rho` field of `Material`; `assign` checks $\rho > 0$ and finite.

## `rotation.py`

```python
from_euler(angles) -> R          # (..., 3) Bunge z-x-z, radians
from_vectors(a, b) -> R          # (..., 3), (..., 3)
bond_stress(R) -> M              # (..., 6, 6): sigma_g = M sigma_l (Voigt)
bond_strain(R) -> N              # (..., 6, 6): eps_g = N eps_l (engineering shear); N = M^{-T}
rotate(C, R) -> M C M^T          # local -> global stiffness, batched over leading axes
```

- **from_euler:** $R = R_z(\varphi_1)\,R_x(\Phi)\,R_z(\varphi_2)$ (intrinsic z–x′–z″, Bunge). Columns are the local axes.
- **from_vectors** (Abaqus `*ORIENTATION, SYSTEM=RECTANGULAR` convention): $e_1' = a/|a|$, $e_3' = (a \times b)/|a \times b|$, $e_2' = e_3' \times e_1'$. $b$ is any vector in the local 1–2 plane, not parallel to $a$. Parallel or zero vectors give NaN (traceable code cannot raise); `assign` checks orientations for finiteness and $|\det R - 1|$, $\|R^\top R - I\|$ to 1e-12.
- **Bond matrices:** for the Voigt pair $(i,j)$ (row) and $(k,l)$ (column), $M_{(ij),(kl)} = R_{ik}R_{jl}$ if $k = l$, else $R_{ik}R_{jl} + R_{il}R_{jk}$. With $D = \mathrm{diag}(1,1,1,2,2,2)$ (engineering shear), $N = D M D^{-1}$; this equals $M^{-\top}$ because $R$ is orthogonal (tested, not assumed). $M$ is built by indexing $R$ with a static (6, 2) Voigt index table, so both are vectorized over leading axes.
- **rotate:** $C_g = M C_l M^\top$. Under congruence, PD is preserved and symmetry exact up to rounding.

## `assign.py`

```python
class Phases(NamedTuple):
    index: np.ndarray   # (n_el,) int32: element -> phase
    C: jax.Array        # (P, 6, 6) global (rotated) stiffness per phase
    rho: jax.Array      # (P,)

assign(mesh, materials, orientations=None) -> Phases
```

- `materials`: dict label → `Material`; `orientations`: dict label → $R$ (3×3); label 0 defaults to the identity when absent (and `orientations=None` means all identity).
- A **phase** is a distinct (material label, orientation label) pair present in the mesh, numbered in order of first appearance in the global element order (`np.unique(..., return_index=True)` and a sort by first index). `index` is host NumPy int32 (PR 4 owns the int32 asserts; $P$ and $n_\text{el}$ are far below $2^{31}$ here, asserted).
- `C` and `rho` are built with `jnp.stack` from the dict values, so `jax.grad` of a function of `Phases.C` flows back to the constants and angles a caller used to build the dicts. The rotation is applied per phase, not per element: memory is $36P$ floats, not $36 n_\text{el}$.
- Raises `ValueError` for: a label present in the mesh but missing from the dict (names the label), a non-`Material` value, `check_stiffness` failure (local and rotated), non-positive or non-finite `rho`, a non-rotation $R$. Extra dict entries are allowed (unused).
- `assign` runs outside `jit` (it reads concrete labels); its output is the `jit` argument (Principle 3). PR 5 gathers `C[index]` per chunk; PR 6's one-phase-per-label restriction is checked there.

## Tests (`tests/test_materials.py`)

Tolerances follow Principle 1 (relative $\|\cdot\|_\infty$ over entries).

1. **Bond = einsum:** for 5 random rotations and a random SPD anisotropic $C$, `rotate(C, R)` equals the `einsum("ia,jb,kc,ld,abcd->ijkl")` rotation of the 4th-order tensor (built from Voigt by a test-local helper with the Voigt↔tensor index maps written out) to 1e-14. Also `bond_strain(R) == inv(bond_stress(R)).T` to 1e-14, and the work identity $\sigma_g\cdot\varepsilon_g = \sigma_l\cdot\varepsilon_l$ for random $\varepsilon$.
2. **Invariance:** isotropic $C$ unchanged by 5 random rotations (1e-14); transversely isotropic $C$ unchanged by rotations about local axis 1 (random angles), and changed by a 90° rotation about axis 3 (a mutant check: relative change > 0.1).
3. **PD:** `check_stiffness` passes on all constructors with realistic constants (steel, carbon/epoxy, a cubic crystal via `anisotropic`) and on their rotations; it raises on: $\nu = 0.5$ isotropic (singular), $\nu_{12}$ beyond the orthotropic bound (indefinite), an asymmetric $C$, a NaN entry, a (6, 5) shape, and a $C$ with $\lambda_\min = 10^{-13}\lambda_\max$ (just below the threshold; $10^{-11}$ passes).
4. **Compliance:** `inv(orthotropic(...))` equals the hand-built $S$ (entries typed out in the test from $E_i$, $\nu_{ij}$, $G_{ij}$) to 1e-14, for two constant sets; the reciprocal relation $S_{21} = -\nu_{21}/E_2$ with $\nu_{21} = \nu_{12}E_2/E_1$ is checked explicitly. Isotropic equals `orthotropic` with equal constants and $G = E/(2(1+\nu))$; transversely isotropic equals its orthotropic expansion.
5. **Orientations:**
   - `from_euler` gives proper orthogonal $R$ (1e-14); known cases: zero angles → $I$; $(\pi/2, 0, 0)$ → $e_1' = e_y$.
   - `from_vectors(a, b)` equals `from_euler(angles)` for $a = R e_1$ scaled by 3 and $b = 2Re_1 + 5Re_2$ (any in-plane $b$), for 5 random angle triples (1e-14).
   - Batched shapes: (n, 3) inputs give (n, 3, 3), equal to a loop.
6. **Assign:**
   - A mesh with labels `material = [5, -1, 5, 0]`, `orientation = [0, 2, 2, 0]` gives 3 phases in first-appearance order with the expected `index`, and `C[index[e]]` equals `rotate(materials[m_e].C, R_{o_e})` per element.
   - Raises: missing material label (message names it), missing orientation label ≠ 0, non-PD material, `rho = 0`, `rho = nan`, a reflection $R$ ($\det = -1$), NaN $R$ (from parallel vectors).
   - Works on a PR 1 voxel mesh and on a read mesh (`tests/data/gmsh_pair.msh`, material labels 7 and 8).
7. **Gradients:** central finite differences (step sweep $h \in \{10^{-4}, 10^{-5}, 10^{-6}\}$ relative, best taken) of $f = \sum W \odot C_g$ (fixed random $W$) with respect to:
   - $(E, \nu)$ of `isotropic` and all 9 constants of `orthotropic` (scaled to $O(1)$ by using GPa-free units in the test);
   - the three Euler angles and the 6 vector components of `from_vectors`;
   - through `assign`: $f(\text{Phases.C})$ w.r.t. one Euler angle used to build an orientation.
   Relative error ≤ 1e-6. `jax.jit` of each constructor and of `rotate` gives identical results to eager.

## Docs

- `docs/materials.md`: conventions ($R$ columns, Voigt, engineering strain), each constructor's formula (Lamé, $S$ entries, transversely isotropic expansion), the orthotropic stability bounds (PD of $C$ ⇔ $E_i, G_{ij} > 0$, $|\nu_{ij}| < \sqrt{E_i/E_j}$ and $1 - \nu_{12}\nu_{21} - \nu_{23}\nu_{32} - \nu_{13}\nu_{31} - 2\nu_{21}\nu_{32}\nu_{13} > 0$), Bond matrices $M$ and $N$ as tables, $C_g = MCM^\top$, Euler and vector orientations, `assign` and phases, and an example from mesh labels to `Phases`.
- `README.md`: status line and a short materials snippet (run in a test, like PR 1's).
- `docs/ROADMAP.md`: PR 2 status note after the merge; PR 2 Contents gain "`assign`: mesh labels → per-phase rotated C and ρ".
- `docs/mesh.md`: the "PR 2/6 map them" sentence points to `assign`.

## Out of scope

- Per-element orientation fields (e.g. fiber directions varying per element): `assign` takes one $R$ per orientation label. A per-element $R$ array can be added when an example needs it (PR 16/23), without changing `rotate`.
- Reading Abaqus `*MATERIAL`/`*ORIENTATION` cards (PR 1 reads only labels).
- Damping, temperature dependence, nonlinearity.

## Checklist before merge

- `pytest -m "not gpu"` passes; ruff clean; `scripts/check.sh`.
- Code review rounds (3–5 reviewers) until no substantive findings.
- Docs and memory updated.
