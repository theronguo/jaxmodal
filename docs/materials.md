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
- **`check_materials(pairs, materials, orientations=None)`:** raises `ValueError` for a label used in the mesh but missing from a dict, a `C` that fails `check_stiffness`, a density that is not a finite positive scalar, or an $R$ that is not a finite rotation ($\max|R^\top R - I| \le 10^{-12}$, so fp32-accurate rotations are rejected; $\det R > 0$). Extra dict entries are allowed. `unstructured` and `stencil` (and so `solve_modes`) call it before tracing, so it takes concrete materials.
- **Density** is the physical density $\rho$ of a material. Design densities in topology optimization (PR 23) enter through the operators' per-element scaling $s_K$, $s_M$, not through `rho`.

## Laminates (solid shells)

```python
from jaxmodal.materials import Laminate, Material, Ply, ply_table

carbon = Material(orthotropic(...), 1600.0)
lam = Laminate((Ply(carbon, 0.0, 0.25), Ply(carbon, np.pi / 2, 0.5), Ply(carbon, 0.0, 0.25)))
materials = {0: steel, 1: lam}  # label -> Material | Laminate
plies, R_o = ply_table(pairs, materials, orientations)  # pure jax.numpy
```

- **`Ply(material, angle, fraction)`:** a `Material`, the angle in radians about the shell normal from the shell frame's axis 1, and the thickness fraction. **`Laminate(plies)`:** a tuple of `Ply`, bottom to top along $+\zeta$. Both are NamedTuples (pytrees).
- **Where:** a `Laminate` only on `ssh8` elements. `assemble` and `unstructured` raise "material $m$ is a Laminate on $t$ elements; laminates need ssh8" for a hex block, and `stencil` likewise (voxel meshes are hex8). A `Material` on `ssh8` is one ply (angle 0, fraction 1).
- **Ply mapping:** ply $k$ occupies $\zeta \in [-1 + 2\sum_{j<k}f_j,\ -1 + 2\sum_{j\le k}f_j]$. A trilinear map is linear in $\zeta$ along each fibre, so reference fractions equal physical thickness fractions on any element.
- **`check_materials`** checks each ply as a `Material` (errors `material m ply k: ...`), a finite angle, and fractions $> 0$ summing to 1 within $10^{-12}$ ("material $m$: ply fractions must be > 0 and sum to 1").
- **`ply_table(pairs, materials, orientations=None) -> (Ply(Material(C, rho), angle, fraction), R_o)`:** unrotated `C (P, L, 6, 6)`, `rho`, `angle`, `fraction` `(P, L)` and `R_o (P, 3, 3)`, with $L$ the largest ply count; shorter stacks are padded with copies of ply 0 at fraction 0 (weight 0). The shell kernels take one element's slice `(jax.tree.map(lambda a: a[p], plies), R_o[p])` (`docs/elements.md`). `phase_table` gives a `Laminate` phase ply 0's rotated $C$ and $\rho$ as a placeholder, read by no `ssh8` code.

**Shell frame** (per element, at the centre, Abaqus-like):

- $t_3 = g_1\times g_2/\|g_1\times g_2\|$ at $\xi = 0$ (the element normal along $+\zeta$).
- $t_1$ = axis 1 of $R_o$ (the orientation label's rotation, identity by default) projected onto the shell plane and normalized; $t_2 = t_3\times t_1$.
- **Fallback:** if that projection has norm $< 0.1$, $t_2$ = axis 2 of $R_o$ projected and normalized, and $t_1 = t_2\times t_3$. The normalization is NaN-free (double `where`); the frame is not differentiable at the switch.
- $R_\text{shell} = [t_1\ t_2\ t_3]$, and ply $k$ uses $C_k' = \mathrm{rotate}(C_k, R_\text{shell}R_z(\theta_k))$. When $R_o$'s axis 3 is $+t_3$, $R_\text{shell} = R_o$, so a one-ply `Material` acts as on hex8.
- **Stacking sign:** a $\zeta$ flip turns the frame by $\mathrm{diag}(1,-1,-1)$ (by $\mathrm{diag}(-1,1,-1)$ in the fallback), reverses the plies and negates the angles. See `solid_shell`'s `direction` (`docs/mesh.md`).
- **hex8 vs `ssh8`:** a hex8 element uses $R_o$ itself; an `ssh8` element projects it onto its shell plane. On a tilted plate, hex8 and `ssh8` parts with one orientation label are therefore oriented differently unless $R_o$'s axis 3 is the normal.

## Not covered

- Composing general rotations: the shell frame composes $R_\text{shell}R_z(\theta_k)$ (above); other compositions are a plain matrix product passed as an orientation.
- Abaqus `*MATERIAL`/`*ORIENTATION` cards are not read; only the labels are (`docs/mesh.md`).
- Damping, temperature dependence, nonlinearity.
