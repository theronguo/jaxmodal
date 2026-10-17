# PR 16 plan: solid-shell element (KGW 1999)

Branch `pr-16-solid-shell`, stacked on `pr-15d-strategy` until 15d merges, then rebased onto
`main` (`git rebase --onto main pr-15d-strategy pr-16-solid-shell`). Roadmap: PR 16 (M4), the
"Shells" and "Mass" key design decisions, PR 8b's `SIMPLY_SUPPORTED`, and the dependency row
16 ← 3, 5, 8, 9 (solve_modes since 15d).

Revision 1 (draft, not yet reviewed).

## Split

PR 16 is three PRs, each merged on its own, in this order:

- **16a, the element:** the `ssh8` type, the thickness direction, the KGW kernels (ANS, EAS,
  condensation), laminates, consistent mass, the stress recovery function, the assembled
  reference (`assemble`), and the element-level tests: patch test, zero modes, MacNeal–Harder,
  `jax.grad`.
- **16b, the operator and API:** stored condensed Ke in `unstructured` (mixed with hex blocks),
  `otf` and `mass="lumped"` refusals, the estimator, probe sampling, `solve_modes` on
  solid-shell meshes (Jacobi), the readers, and the M4 example.
- **16c, the accuracy suite:** Srinivas–Rao, the clamped cylinder, no locking, the skew plate,
  their hex27 references and `docs/verification.md`.

Why: 16a is a self-contained element with its own oracles (SciPy on `assemble`), 16b changes the
operator and the API, and 16c is long-running reference work that needs both. Each part gets
its own plan revision on its own branch (`pr-16a-element`, `pr-16b-operator`, `pr-16c-accuracy`);
this document is the shared design, reviewed once.

## Data model

- **Element type `ssh8`:** a new mesh block type with hex8 topology (8 nodes, VTK order), added
  to `TYPES = ("hex8", "hex20", "hex27", "ssh8")` and `NODES`. The block's connectivity is
  oriented so the reference $\zeta$ axis (corners 0–3 → 4–7) is the thickness direction.
- **Making a block solid-shell:** `solid_shell(mesh, elements=None, direction=None)`
  (`mesh/shell.py`, host NumPy) returns a new `Mesh` with the chosen hex8 elements (an
  element set name, an index array, or all hex8 when None) moved into an `ssh8` block, their
  connectivity permuted so $\zeta$ is the thickness, and `material`, `orientation` and element
  sets carried along (element indices renumbered, block order kept). It is the only way to get
  `ssh8` besides the `.inp` reader (SC8R/SC8 in 16b).
- **Thickness direction:**
  - `direction` given: a (3,) vector or an (n_el_sel, 3) array. Per element, the reference axis
    whose mean edge vector (the 4 parallel edges' mean) has the largest $|\cos|$ with the
    direction is the thickness; if the two largest $|\cos|$ differ by less than 0.1 the element
    is ambiguous and `ValueError` names it.
  - `direction=None` (topological sweep): per element and reference axis, the column length is
    the number of elements reached by stepping through opposite faces of that axis inside the
    selection (faces matched by their sorted corner sets; the axis carried across each shared
    face). The thickness axis is the one with the shortest column; it must be unique per
    element and every element of a column must agree. A tie (e.g. an $n^3$ cube block) or a
    disagreement raises `ValueError` ("ambiguous thickness direction: pass direction="). Test:
    a stacked plate ($8\times6\times2$, any connectivity rotation) is detected; the cube block
    $3^3$ raises.
  - The orientation keeps $\det J > 0$ (an even permutation, or a reflection fixed by swapping
    the $\zeta$ faces' corner order).
- **Laminates:** `Laminate(plies, n_gauss=2)` (`materials/`), plies a tuple of
  `(material_label, angle, fraction)`: the ply's material, its fibre angle in degrees about the
  shell normal from the local $t_1$ axis, and its thickness fraction (summing to 1, bottom
  ($\zeta = -1$) to top). A solid-shell element's material label may name a `Material` (one ply,
  angle 0, fraction 1) or a `Laminate` in a new `laminates` dict argument (`laminates={label:
  Laminate}`) passed beside `materials` to `assemble`, `unstructured`, `solve_modes` and the
  estimators; a label in both raises. hex blocks with a laminate label raise.
- **Ply rotation in the shell frame:** at each quadrature point, $t_3 = g_1 \times g_2 /
  \|g_1 \times g_2\|$ (the covariant in-plane bases from the trilinear map), $t_1 = g_1$
  projected off $t_3$ and normalized, $t_2 = t_3 \times t_1$; $R = [t_1\ t_2\ t_3]\,R_z(\theta)\,
  R_o$, with $R_o$ the element's orientation label rotation (default identity), composed as in
  `docs/materials.md` ($R = R_\text{shell}R_\text{ply}$). A flat plate in the $xy$ plane meshed
  along $x$ has $R_\text{shell} = I$, so its plies are exactly PR 8b's `plate_modes` plies.
- **Phase table:** `phase_table` stays per (material, orientation); a laminate's plies resolve to
  their materials' $C$ and $\rho$ (unrotated), stacked (n_plies, 6, 6) per laminate label, padded
  to the block's largest ply count with zero-fraction plies. The rotation is applied in the
  kernel (it depends on the point). Everything stays traceable for PR 21.

## The element (16a)

Notation: $X(\xi)$ the trilinear map, $g_i = \partial X/\partial \xi_i$, $J = [g_1\ g_2\ g_3]^\top$,
$J_0$ at the centre, displacements $u$ trilinear. Strains in covariant components
$E_{ij} = \tfrac12(g_i \cdot u_{,j} + g_j \cdot u_{,i})$ (linear), mapped to Cartesian Voigt
$\varepsilon = T(\xi)\,E$ with $T$ the contravariant transformation from $J^{-1}$.

- **ANS, transverse shear** (Bathe–Dvorkin): $E_{13}$ from the points $(0, \mp1, \zeta)$,
  $E_{13}^{ANS} = \tfrac12(1 - \eta)E_{13}(0, -1, \zeta) + \tfrac12(1 + \eta)E_{13}(0, 1, \zeta)$,
  and $E_{23}$ from $(\mp1, 0, \zeta)$ likewise in $\xi$; evaluated at each quadrature point's
  $\zeta$.
- **ANS, thickness strain** (Betsch–Stein): $E_{33}$ from the 4 corner lines $(\pm1, \pm1,
  \zeta)$, bilinear in $(\xi, \eta)$.
- **EAS** with $J_0$, 5 parameters: enhanced covariant strains $\tilde E = M(\xi)\alpha$ with
  $\tilde E_{11} = \xi\alpha_1$, $\tilde E_{22} = \eta\alpha_2$, $\tilde E_{12} = \xi\alpha_3 +
  \eta\alpha_4$ (Simo–Rifai membrane), $\tilde E_{33} = \zeta\alpha_5$ (thickness, linear in
  $\zeta$); Cartesian $\tilde\varepsilon = (\det J_0/\det J)\,T_0\,\tilde E$. $\int_\square M\,d\xi
  = 0$ holds exactly, so the patch test passes.
- **Condensation:** $K_e = K_{uu} - K_{u\alpha}K_{\alpha\alpha}^{-1}K_{\alpha u}$ (5×5 solve per
  element, Cholesky), symmetrized as `kernels._sym`. The recovered parameters are $\alpha =
  -K_{\alpha\alpha}^{-1}K_{\alpha u}u_e$.
- **Quadrature:** 2×2 in-plane Gauss times `n_gauss` Gauss points per ply in $\zeta$ (each ply
  mapped to its $\zeta$ sub-interval), stiffness and mass alike. A one-ply element uses 2×2×2.
- **Mass:** consistent, from the trilinear interpolation, per-ply density, same quadrature (exact
  for the trilinear products per ply). `lumping=None`: `lumped_mass` and `check_elements(...,
  mass="lumped")` raise "solid-shell elements are consistent-mass only".
- **Kernels** (`elements/shell.py`, pure jax.numpy, batched): `shell_stiffness(X, C, frac,
  angle, R_o)`, `shell_mass(X, rho, frac)`, `shell_recover(X, C, frac, angle, R_o, u)` (the
  Cartesian strains and stresses at every quadrature point from the ANS strains plus the EAS
  strains with the recovered $\alpha$), and the bilinear forms `shell_stiffness_form` /
  `shell_mass_form` for PR 21. The registry entry `ELEMENTS["ssh8"]` carries the hex8 nodes,
  shape functions and faces (topology code keeps working), `lumping=None` and a `kernels`
  field naming these functions; hex entries get their existing kernels in the same field, and
  `assemble` / `unstructured` dispatch through it.
- **Checks** (`check_elements`): $\det J > 0$ at the corners and quadrature points, the
  $\zeta$ axis the shortest of the three mean edge lengths (else "connectivity not oriented:
  use solid_shell"), and laminate fractions positive summing to 1 to $10^{-12}$.

### 16a tests

1. **Kernel against an independent NumPy implementation** of the same formulas (loops per
   point, explicit $T$, ANS interpolation, EAS, dense condensation) on random distorted
   elements and random anisotropic $C$, laminates of 1–3 plies: $K_e$ and $M_e$ to $10^{-12}$
   relative.
2. **Zero modes:** exactly 6 eigenvalues of $K_e$ below $10^{-10}\lambda_\max$ and the 7th above
   $10^{-8}\lambda_\max$, on an element with $h/t \le 2$ (regular and distorted).
3. **Patch test with lifting** (the flat MacNeal–Harder patch, 5 elements, plus a lifted copy
   with the interior nodes moved out of plane by 0.1 t): under a linear displacement field,
   the assembled $K u$ vanishes at the free nodes to $10^{-10}$, and `shell_recover` gives the
   constant $C\varepsilon$ at every point to $10^{-10}$ relative to $\|\sigma\|$.
4. **MacNeal–Harder** (1985 reference values; straight cantilever, regular and parallelogram
   meshes, 6×1 elements, one element through the thickness): normalized tip deflection ≥ 0.95
   regular for extension, in-plane shear, out-of-plane shear; ≥ 0.9 parallelogram for
   extension and out-of-plane shear; parallelogram in-plane shear, twist (both meshes) and the
   trapezoidal mesh reported only. Static solves with `spsolve` on `assemble`.
5. **`assemble` on mixed blocks:** a hex8 + ssh8 mesh assembles; $K$, $M$ symmetric PSD with
   the six rigid modes in the null space of $K$; $M$'s total equals $\rho V$.
6. **`jax.grad`:** $\partial(u^\top K_e v)/\partial X$ and $\partial/\partial C$ match central
   finite differences to $10^{-6}$ relative.
7. **`solid_shell` and the thickness sweep:** the stacked plate and the cube cases above;
   `direction` with a tilted plate; the oriented connectivity has $\det J > 0$; element sets
   and labels follow the elements.
8. **Refusals:** `mass="lumped"` on an `ssh8` mesh, a laminate on a hex block, fractions not
   summing to 1, an `ssh8` block with $\zeta$ not the thickness.

## The operator and API (16b)

- **Stored condensed Ke:** `unstructured(..., mode="stored")` builds `ssh8` blocks with the
  shell kernel (packed upper triangle of the 24×24 condensed $K_e$, the same leaf as hex8, so
  `apply_K`, `diag_K`, `block_diag_K` and `scaled` are unchanged); `apply_M` uses a stored
  packed $M_e$ for `ssh8` (the per-ply density makes the hex `N`-based mass apply wrong), and
  `diag_M` its diagonal. `mode="otf"` on a mesh with an `ssh8` block raises (`ValueError`,
  tested).
- **Estimator:** `unstructured_memory` adds the `ssh8` stored leaves (Ke and Me packed, 2×300
  floats) and a build term `BUILD_BYTES["stored"]["ssh8"]` measured by `memory_analysis` like
  the hex ones. Test (slow): the stored condensed-Ke matvec on an `ssh8` mesh sized so the
  estimate is ≥ 2 GB, estimate within 20% of the measured peak (`harness.measure`).
- **Probe sampling:** `operators/sampling.py`'s exact element factor uses the shell mass's
  points and weights (per-ply density) for `ssh8`; the count is exact as for hex.
- **`solve_modes`:** a mesh with an `ssh8` block gets `precond="auto"` → `"jacobi"` (until PR
  18); `"amg"`, `"gmg"` raise naming PR 18. `band` and `verify` work (Jacobi). `solve_modes_memory`
  follows. Rigid modes, masking and BCs are unchanged (hex8 topology).
- **Readers:** `.inp` element types `SC8R`/`SC8` → `ssh8` (connectivity then oriented by
  `solid_shell(direction=None)` on that block); VTU writes `ssh8` as VTK hexahedra with a cell
  field `jaxmodal_type` so `read` restores the type.
- **M4 example** (`examples/m4_shell.py`): a simply supported square plate, 8×8×1 `ssh8`, Jacobi
  `solve_modes`, 6 modes against `reference_modes` (PR 8 eigsh) to the PR 9 tolerances; also the
  test.

### 16b tests

1. Stored condensed-Ke `apply_K`, `apply_M`, `diag_K`, `diag_M` against `assemble`'s K and M to
   $10^{-12}$ relative, on an `ssh8` mesh and a hex8 + `ssh8` mesh, $m = 1$ and $m > 1$, random
   `s_K`/`s_M`; scatter and gather paths.
2. Jacobi `solve_modes` on the M4 example against `reference_modes` (PR 9 tolerances).
3. `otf`, `mass="lumped"`, `precond="amg"`/`"gmg"` refusals.
4. The ≥ 2 GB estimator test (slow).
5. The probe factors: $B_eB_e^\top = M_e$ to $10^{-12}$ for `ssh8` laminates.
6. Readers: an SC8R `.inp` round trip; a VTU round trip keeps `ssh8`.

## The accuracy suite (16c)

All discretization-error tests use PR 8's `reference_modes` (eigsh shift-invert, within its
size limit) on `assemble`; Jacobi LOBPCG is impractical at $\kappa \approx 10^{12}$.

- **Srinivas–Rao** (0.1% target against 3D elasticity, `plate_modes`): the PR 8b cross-ply and
  single-ply plates at L/t = 10 and 100, with $k$ stacked `ssh8` elements per ply ($k$ found and
  stated in the docs, the smallest meeting 0.1% on the first 6 bending modes); one layered
  element per laminate reported only.
- **Clamped cylinder** (R/t and L/R stated, e.g. R/t = 50, L/R = 2): against a hex27 reference
  whose error, estimated from ≥ 3 nested levels (Richardson, estimate only), is ≤ 0.02%; the
  `ssh8` mesh meets 0.1% on the first 6 modes. Levels and estimate recorded in
  `docs/verification.md`.
- **No locking:** the simply supported square plate, first 6 bending modes by z-reflection
  parity, the same in-plane mesh and layer count at L/t = 10 and 1000; error against a converged
  `ssh8` run at the same L/t kept ≫ 1e-4; the error at 1000 is ≤ 1.5× the error at 10.
- **Skew plate** (parallelograms only, skew angle and density stated, clamped, L/t stated):
  within 1% of a hex27 reference with estimated error ≤ 0.2% (≥ 3 nested levels).
- **Benchmarks:** the hex27 reference levels run in `benchmarks/shell_references.py` (≤ 1 h,
  progress per level, results in `benchmarks/results/shell_references.json`); the tests read
  the recorded references and recompute only the `ssh8` solves.

## Docs

- `docs/elements.md`: the `ssh8` entry, the formulation (ANS, EAS, condensation, quadrature,
  frames), laminates, the recovery function, refusals.
- `docs/mesh.md`: `ssh8`, `solid_shell`, the thickness rules.
- `docs/materials.md`: `Laminate`, the shell frame and ply angles.
- `docs/operators.md`: condensed stored Ke and Me, the estimator term.
- `docs/eigensolvers.md`: Jacobi on solid-shell meshes until PR 18.
- `docs/verification.md`: reference levels, elements per ply, the 16c results.
- README, ROADMAP status, `docs/open_items.md` (decisions below).

## Decisions without user input (to log)

- The three-way split.
- The type name `ssh8` and the `solid_shell` helper instead of a per-block flag.
- The thickness rules (mean-edge cosine with a 0.1 margin; shortest column, unique and
  consistent).
- `Laminate` in a separate `laminates` dict; angles in degrees from $t_1$; the shell frame from
  $g_1$ projected.
- Stored $M_e$ for `ssh8` (per-ply density) instead of the hex `N`-based mass apply.
- Jacobi only on `ssh8` meshes until PR 18 (AMG and GMG raise).
- `n_gauss = 2` per ply by default.

## Open questions for the reviewers

- Is the EAS transform $(\det J_0/\det J)T_0$ applied to covariant enhanced strains the KGW
  form (versus Andelfinger–Ramm's)? Does it keep the patch test on the lifted patch?
- Does the ANS thickness strain need the corner values at the quadrature point's $\zeta$ or at
  $\zeta = 0$ for a linear element?
- Are the 16c gates reachable with a reasonable $k$ per ply and within the eigsh size limit?
