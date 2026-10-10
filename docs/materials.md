# Materials

`jaxmodal.materials` builds linear elastic stiffness matrices, rotates them into material axes and maps a mesh's material and orientation labels to a per-phase property table.

```python
from jaxmodal.materials import Material, from_euler, orthotropic, phase_table, phases

C = orthotropic(140e9, 10e9, 8e9, 0.3, 0.25, 0.45, 5e9, 4e9, 3.4e9)
index, pairs = phases(mesh)  # host
C_p, rho_p = phase_table(pairs, {7: Material(C, 1600.0)}, {2: from_euler((0.5, 0.2, 0.0))})
```

- **Traceable:** constructors, rotations and `phase_table` are pure `jax.numpy` functions: `jit`, `vmap` and `grad` work through them. They cast inputs with `jnp.asarray(x, float)`, so material data is always in the default float (fp64 under x64, also for fp32 or integer inputs). This is an exception to "fp32 inputs stay fp32" in `docs/conventions.md`.
- **Host checks:** `check_stiffness`, `check_materials` and `phases` need concrete values and run in NumPy.
- **Units** are the caller's; use a consistent set (e.g. SI).

## Conventions

- **Voigt order** $[11, 22, 33, 23, 13, 12]$ with engineering shear strains $\gamma_{ij} = 2\varepsilon_{ij}$; $\sigma = C\varepsilon$ (`docs/conventions.md`).
- **Rotation $R$:** a proper orthogonal 3×3 matrix whose columns are the local material axes $e_1', e_2', e_3'$ in global coordinates:

  $$x_\text{global} = R\,x_\text{local}, \qquad \sigma_\text{global} = R\,\sigma_\text{local}R^\top.$$

  In texture-analysis notation (rows = crystal axes) this $R$ is $g^\top$.

## Constructors

All take scalar constants and return a (6, 6) local stiffness $C$; use `jax.vmap` for batches of constants.

- **`isotropic(E, nu)`:** $C = \lambda\,\mathbf{1}\mathbf{1}^\top + \mu\,\mathrm{diag}(2, 2, 2, 1, 1, 1)$, with $\mathbf{1} = (1, 1, 1, 0, 0, 0)^\top$ and

  $$\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}, \qquad \mu = \frac{E}{2(1+\nu)}.$$

- **`orthotropic(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)`:** $C = S^{-1}$ with the compliance

  $$S = \begin{pmatrix}
  1/E_1 & -\nu_{12}/E_1 & -\nu_{13}/E_1 & & & \\
  -\nu_{12}/E_1 & 1/E_2 & -\nu_{23}/E_2 & & & \\
  -\nu_{13}/E_1 & -\nu_{23}/E_2 & 1/E_3 & & & \\
  & & & 1/G_{23} & & \\
  & & & & 1/G_{13} & \\
  & & & & & 1/G_{12}
  \end{pmatrix}.$$

  $\nu_{ij} = -\varepsilon_j/\varepsilon_i$ under uniaxial stress $\sigma_i$, and $\nu_{ij}/E_i = \nu_{ji}/E_j$; the inputs are the three $\nu_{ij}$ with $i < j$ (the others follow as $\nu_{ji} = \nu_{ij}E_j/E_i$).
- **`transversely_isotropic(E1, E2, nu12, nu23, G12)`:** isotropic in the local 2–3 plane, symmetry axis (fiber) along local 1:

  $$C = \texttt{orthotropic}\Bigl(E_1, E_2, E_2, \nu_{12}, \nu_{12}, \nu_{23}, G_{12}, G_{12}, \tfrac{E_2}{2(1+\nu_{23})}\Bigr).$$

  Other symmetry axes come from an orientation.
- **`anisotropic(C)`:** a full (6, 6) $C$, returned as a float array. An asymmetric input is not fixed silently; `check_stiffness` rejects it.

### Validity

`check_stiffness(C)` raises `ValueError` unless $C$ is a finite (6, 6) array with

$$\max|C - C^\top| \le 10^{-12}\max|C| \quad\text{and}\quad \lambda_\min(C) > 10^{-12}\,\lambda_\max(C).$$

Constructors don't call it (traced values can't raise). For orthotropic constants, $C$ is positive definite if and only if $E_i, G_{ij} > 0$, $|\nu_{ij}| < \sqrt{E_i/E_j}$ and

$$1 - \nu_{12}\nu_{21} - \nu_{23}\nu_{32} - \nu_{13}\nu_{31} - 2\nu_{21}\nu_{32}\nu_{13} > 0.$$

Isotropic $\nu = 1/2$ gives non-finite $C$; $\nu$ within about $10^{-12}$ of $1/2$ (e.g. $1/2 - 10^{-14}$) fails the eigenvalue ratio.

## Rotations

```python
from_euler(angles)  # (..., 3) -> (..., 3, 3)
from_vectors(a, b)  # (..., 3), (..., 3) -> (..., 3, 3)
bond_stress(R)  # (..., 6, 6)
bond_strain(R)  # (..., 6, 6)
rotate(C, R)  # (..., 6, 6): local -> global
```

All are batched over leading axes, so a per-element field of $R$ (shape `(n_el, 3, 3)`) rotates a per-element $C$ directly.

- **Euler angles** (Bunge, intrinsic z–x′–z″, radians): $R = R_z(\varphi_1)\,R_x(\Phi)\,R_z(\varphi_2)$. This equals SciPy's `Rotation.from_euler("ZXZ", angles).as_matrix()`.
- **Two vectors** (Abaqus `*ORIENTATION, SYSTEM=RECTANGULAR`): local axis 1 along $a$, $b$ any vector in the local 1–2 plane:

  $$e_1' = \frac{a}{|a|}, \qquad e_3' = \frac{a \times b}{|a \times b|}, \qquad e_2' = e_3' \times e_1'.$$

  Parallel vectors give NaN columns 2–3 (a zero $a$: all columns) and undefined gradients; `check_materials` rejects them.
- **Bond matrices.** For Voigt rows $(ij)$ and columns $(kl)$,

  $$M_{(ij),(kl)} = \begin{cases} R_{ik}R_{jl} & k = l \\ R_{ik}R_{jl} + R_{il}R_{jk} & k \ne l \end{cases}, \qquad \sigma_g = M\sigma_l,$$

  and with $D = \mathrm{diag}(1, 1, 1, 2, 2, 2)$,

  $$N = D M D^{-1} = M^{-\top}, \qquad \varepsilon_g = N\varepsilon_l.$$

- **Stiffness rotation:** $C_g = M C_l M^\top$. It is a congruence, so symmetry and positive definiteness are preserved.

## Phases

A mesh carries integer `material` and `orientation` labels per element (`docs/mesh.md`). A **phase** is a distinct (material label, orientation label) pair present in the mesh. Topology (host, fixed) and values (traced) are separate, so the table can be built inside `jit` and `grad`:

```python
index, pairs = phases(mesh)  # host NumPy
check_materials(pairs, materials, orientations)  # host, once
C, rho = phase_table(pairs, materials, orientations)  # pure jax.numpy
```

- **`phases(mesh)`:** `pairs` is a tuple of $P$ `(material, orientation)` tuples in order of first appearance in the global element order; it is hashable, so it can be a static argument (`jax.jit(f, static_argnames="pairs")`). `index` is the `(n_el,)` int64 element → phase map (host indices stay int64; PR 5 passes its size through `check_int32` and casts it to int32 for the device, `docs/operators.md`).
- **`phase_table(pairs, materials, orientations=None)`:** `materials` maps label → `Material(C, rho)`, `orientations` maps label → $R$. It returns the global stiffness `C` (P, 6, 6), each phase's $C$ rotated by its $R$, and the density `rho` (P,). Memory is $37P$ floats; PR 5 gathers `C[index]` per element chunk and PR 6 builds one reference Ke per phase. A label missing from a dict is a plain `KeyError`.
- **Label 0:** orientation label 0 means the identity unless `orientations` has an entry for it; any other orientation label must be given (`orientations=None` means `{}`). Material label 0 has no default: a voxel mesh (all labels 0) needs `materials = {0: ...}`.
- **`check_materials(pairs, materials, orientations=None)`:** raises `ValueError` for a label used in the mesh but missing from a dict, a `C` that fails `check_stiffness`, a density that is not a finite positive scalar, or an $R$ that is not a finite rotation ($\max|R^\top R - I| \le 10^{-12}$, so fp32-accurate rotations are rejected; $\det R > 0$). Extra dict entries are allowed. PR 9's `solve_modes` calls it before tracing, so it takes concrete materials.
- **Density** is the physical density $\rho$ of a material. Design densities in topology optimization (PR 23) enter through the operators' per-element scaling $s_K$, $s_M$, not through `rho`.

## Not covered

- Composing rotations, e.g. ply angles in a shell frame (PR 16): $R = R_\text{shell}R_\text{ply}$, a plain matrix product.
- Abaqus `*MATERIAL`/`*ORIENTATION` cards are not read; only the labels are (`docs/mesh.md`).
- Damping, temperature dependence, nonlinearity.
