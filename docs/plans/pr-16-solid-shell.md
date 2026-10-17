# PR 16 plan: solid-shell element (KGW 1999)

Branch `pr-16-solid-shell` (this plan), from `main` after PR 15d. Roadmap: PR 16 (M4), the
"Shells" and "Mass" key design decisions, PR 8b's `SIMPLY_SUPPORTED`, and the dependency row
16 ← 3, 5, 8, 9.

- Revision 1: draft.
- Revision 2 (round 1, 5 reviewers, all asked for design changes): laminates go into
  `materials`; the ply frame comes from the orientation label projected onto the shell, not
  from the node order; the stored mass is the 8×8 scalar $m_e$; the plies are integrated with
  a `scan` (one build constant); fixed node permutations with a defined $\zeta$ sign; no
  edge-length orientation check; dispatch by name, not a registry field; a patch test that the
  element can pass (consistent tractions; ANS limits stated); Srinivas–Rao per half-wave
  sub-plate; the cylinder on a 1/8 symmetry model with observed-order Richardson; a concrete
  locking floor; Jacobi tests at $L/t \le 20$; SC8R keeps the file's stacking; VTU type field
  and the bilinear forms deferred; explicit `precond="amg"` allowed; use sites enumerated.

## Split and plans

PR 16 is three PRs, each from `main`, merged in order: `pr-16a-element`, `pr-16b-operator`,
`pr-16c-accuracy`.

- **16a, the element:** `ssh8`, `solid_shell`, `Laminate`, the kernels, `assemble`, recovery,
  element-level tests.
- **16b, the operator and API:** `unstructured`, the estimator, sampling, `solve_modes`, AMG,
  the reader, the M4 example.
- **16c, the accuracy suite:** Srinivas–Rao, cylinder, no locking, skew plate.

This document is the plan for all three. 16c's meshes and gates depend on a sizing study (its
first step); its results go into an addendum to this plan, reviewed before the 16c tests are
written. If a roadmap gate is out of reach after the study, stop and ask the user.

## Data model

- **Element type `ssh8`:** hex8 topology (8 nodes, VTK order), added to `TYPES` (last) and
  `NODES`. The reference $\zeta$ axis (corners 0–3 → 4–7) is the thickness direction, and
  $+\zeta$ orders the plies (bottom to top). `box_mesh(type="ssh8")` raises (use `solid_shell`).
- **`solid_shell(mesh, elements=None, direction=None) -> (mesh, order)`** (`mesh/shell.py`,
  host NumPy): moves the chosen hex8 elements (an element set name, an index array, or all
  hex8 when None) into an `ssh8` block. It permutes their connectivity so $\zeta$ is the
  thickness, and carries `material`, `orientation` and element sets along. `order` is the
  element permutation (new element $i$ is old element `order[i]`), for user per-element arrays
  (`s_K`, `s_M`). `voxel` becomes None (`_check_voxel` needs a single hex8 block). Besides the
  `.inp` reader, it is the only producer of `ssh8`. A hand-built `ssh8` block is the user's
  responsibility (documented).
- **Node permutations:** three fixed proper rotations of the hex8 corners, one per reference
  axis, take that axis to $\zeta$. The 180° rotation about $\xi$ flips the sign of $\zeta$ (and
  $\eta$). All of them are proper, so $\det J$ keeps its sign. The faces are never swapped alone.
- **Thickness axis:**
  - `direction` given: a (3,) vector, or an (n_sel, 3) array in sorted selected-element order.
    Per element, the reference axis whose mean edge vector (mean of its 4 parallel edges) has
    the largest $|\cos|$ with the direction is the thickness. $\zeta$ points along $+$direction.
    If the two largest $|\cos|$ differ by less than 0.1, the element is ambiguous and
    `ValueError` names it.
  - `direction=None` (sweep): per element and reference axis, the column length is the number of
    selected elements reached by stepping through that axis's opposite faces. It uses face
    matching by sorted corner keys, as `parts`, carrying the axis across each shared face. The
    thickness axis is the unique shortest column. A tie raises
    `ValueError("ambiguous thickness direction: pass direction=")`. Examples: the $3^3$ cube
    block raises, the MacNeal–Harder strip $6\times1\times1$ raises, and an $8\times8\times12$
    stack is wrong. All three need `direction=`. The sign of $\zeta$ is propagated across
    in-plane face neighbours, so it is consistent on each connected region. The seed is the
    lowest-index element, with $\zeta$ along the global axis it is most aligned with, positive.
    For unsymmetric laminates, pass `direction` to fix the ply order.
- **Laminates in `materials`:** `materials={label: Material | Laminate}`.
  `Laminate(plies)` is a NamedTuple with plies a tuple of `(Material, angle, fraction)`: the
  ply material, the fibre angle in degrees about the shell normal from the local 1 axis, and
  the thickness fraction, bottom to top. It is a pytree, so it stays traceable for PR 21.
  - `check_materials` (host, skipped under tracing like the other checks) checks: fractions
    $> 0$ summing to 1 to $10^{-12}$, angles finite.
  - A `Laminate` label on a hex block raises in `assemble` and `unstructured`. `stencil`
    rejects `Laminate`.
  - A `Material` on `ssh8` is one ply at angle 0, fraction 1.
  - 2 Gauss points per ply in $\zeta$, fixed (no option).
- **Shell frame, per element at the centre:**
  - $t_3 = g_1 \times g_2 / \|g_1 \times g_2\|$ at $\xi = 0$ (the code's $J_{ij} = \partial
    x_i/\partial\xi_j$, so $g_j$ are columns).
  - $t_1$ is axis 1 of the orientation label rotation $R_o$ (identity by default) projected
    onto the plane $\perp t_3$ and normalized. If that projection is $< 0.1$ (axis 1 within 6°
    of the normal), axis 2 of $R_o$ is projected instead. This is the Abaqus convention.
  - $t_2 = t_3 \times t_1$, and $R_\text{shell} = [t_1\ t_2\ t_3]$.
  - Ply $k$ uses $C_k' = \text{rotate}(C_k, R_\text{shell}R_z(\theta_k))$ (`materials.rotate`).
  - The frame does not depend on the node order. When $R_o$'s axis 3 is $t_3$, it equals
    $R_o$, so a one-ply `Material` behaves as on hex8 (tested).
  - It is computed in the kernel from $X$ (traceable for shape optimization).
- **`ply_table(pairs, materials, orientations)`** (pure jnp, beside `phase_table`):
  - Returns `C (P, L, 6, 6)` unrotated, `rho (P, L)`, `frac (P, L)`, `angle (P, L)` and
    `R_o (P, 3, 3)` over all phases, with $L$ the largest ply count.
  - Padding copies ply 0 with fraction 0, so it stays finite under `grad`.
  - It is built only when an `ssh8` block exists, and indexed by the hex table's phase index.
  - `phase_table` is unchanged. A `Laminate` phase's `rho` entry there is 1, so the mass scale
    `rho[phase]·s_M` reduces to $s_M$ for `ssh8`.
- **Use sites** (each `ssh8` branch is stated, so no hex table is applied to a laminate by
  name):
  - `kernels.stiffness`, `mass`, `lumped_nodal`, `stiffness_form` and `mass_form` dispatch to
    the shell kernels for `ssh8`. `lumped_*` and the forms raise; the forms are deferred to
    PR 21.
  - `check_elements` checks $\det J > 0$ at the corners and the hex 2×2×2 points (unchanged).
    `mass="lumped"` raises explicitly.
  - `matfree`: `_build`, `_wdet`, `_m_scale`, `_m_apply`, `_m_diag`, `element_matrices` and
    `_block_data`.
  - `sampling.element_factor` and `xi_shapes`.
  - `memory`: `per_el`, `_chunk_temps`, `BUILD_BYTES` and `OTF_BYTES`, plus `_amg_scan`'s
    family test.
  - `eigen/api.py`: `levers` gets no otf hint on `ssh8`. `resolve_precond` maps auto to Jacobi.
  - `mesh/io.py`: the `.inp` regex, `write`, and `MESHIO`.
  - The exports in the three `__init__.py` files.
  - The registry entry is the hex8 entry with `lumping=None` (`lumping: str | None`), with no
    callables (`registry.py` stays NumPy-only). The recovery function is in
    `elements/__init__.py` as `RECOVER = {"ssh8": shell_recover}`; PR 17 adds hex entries.

## The element (16a)

Notation:
- $X(\xi)$ trilinear, $g_j = \partial X/\partial\xi_j$, $J = [g_1\ g_2\ g_3]$ (columns), $J_0$
  at the centre.
- Covariant strains $E_{ij} = \tfrac12(g_i\cdot u_{,j} + g_j\cdot u_{,i})$ in `VOIGT` order with
  engineering shear $(E_{11}, E_{22}, E_{33}, 2E_{23}, 2E_{13}, 2E_{12})$.
- Cartesian strains $\varepsilon = T(J^{-1})E$, where $T$ is `bond_strain`'s matrix with $R$
  replaced by $J^{-1}$.

Formulation:
- **ANS, transverse shear** (Bathe–Dvorkin): $E_{13}^{ANS} = \tfrac12(1-\eta)E_{13}(0,-1,\zeta) +
  \tfrac12(1+\eta)E_{13}(0,1,\zeta)$, and $E_{23}$ likewise from $(\mp1, 0, \zeta)$ in $\xi$.
  Both are linear in $\zeta$, evaluated at the point's $\zeta$ (exact).
- **ANS, thickness strain** (Betsch–Stein): $E_{33}$ bilinear in $(\xi,\eta)$ from the 4 corner
  lines. For trilinear $X$, $u$ it does not depend on $\zeta$, so it is evaluated at $\zeta = 0$
  once.
- **EAS**, 5 parameters, covariant: $\tilde E_{11} = \xi\alpha_1$, $\tilde E_{22} = \eta\alpha_2$,
  $2\tilde E_{12} = \xi\alpha_3 + \eta\alpha_4$, $\tilde E_{33} = \zeta\alpha_5$. Then
  $\tilde\varepsilon = (\det J_0/\det J)\,T(J_0^{-1})\,M\alpha$. $\int M\,d\xi = 0$, so the EAS
  part keeps the patch test on any geometry.
- **Condensation:** $K_e = K_{uu} - K_{u\alpha}K_{\alpha\alpha}^{-1}K_{\alpha u}$ (5×5
  Cholesky), symmetrized as `_sym`. Recovery: $\alpha = -K_{\alpha\alpha}^{-1}K_{\alpha u}u_e$.
- **Quadrature:**
  - 2×2 in-plane Gauss times 2 Gauss points per ply.
  - The plies are accumulated with `lax.scan`: $K_{uu}$, $K_{u\alpha}$, $K_{\alpha\alpha}$ and
    $m_e$ are summed per ply. The build temporaries are then those of one ply (8 points),
    independent of the ply count.
  - A one-ply element is 2×2×2. It is exact for prismatic elements and, like hex8, approximate
    when $\det J$ varies.
- **Mass:** consistent, $M_e = m_e \otimes I_3$, where $m_e = \sum_k \rho_k\int_{\text{ply }k}
  N^\top N \det J$ (8×8). Consistent only: lumping raises "solid-shell elements are
  consistent-mass only".
- **Locking:**
  - Covered: transverse shear (ANS), curvature thickness (corner ANS), Poisson thickness
    ($\alpha_5$), in-plane shear and parasitic in-plane bending ($\alpha_{1..4}$).
  - Not covered: membrane locking of curved elements. KGW has no membrane ANS; this is stated
    in `docs/elements.md` and the cylinder mesh accounts for it.
  - Trapezoidal in-plane bending is not covered (MacNeal–Harder, reported only).
- **Kernels** (`elements/shell.py`, pure jnp, batched):
  - `shell_stiffness(X, C, frac, angle, R_o) -> Ke (24, 24)`.
  - `shell_mass(X, rho, frac) -> m_e (8, 8)`.
  - `shell_recover(X, C, frac, angle, R_o, u) -> (eps, sigma)`: the Cartesian strains and
    stresses at every quadrature point, with points and weights. These come from the ANS
    strains plus the EAS strains with the recovered $\alpha$.
  - PR 17 owns the postprocessor stress-recovery test (the roadmap's "whichever merges last").

### 16a tests

1. **Independent NumPy kernel** (point loops, explicit $T$, ANS interpolation, EAS, dense
   condensation; `tests/shell_ref.py`): random distorted elements, random anisotropic $C$,
   laminates of 1–3 plies. $K_e$ and $m_e$ agree to $10^{-12}\max|K_e|$.
2. **Independent references for the laminate path,** all to $10^{-12}$:
   - (a) $K_e$ is invariant under the 4 in-plane connectivity rotations and the $\zeta$ flip
     (with the ply order reversed).
   - (b) A rigidly rotated element with $R_o$ rotated alike gives $QK_eQ^\top$.
   - (c) A one-ply laminate at angle $\theta$ equals the element with
     `rotate(C, R_z(θ))`.
   - (d) A laminate of identical plies equals the one-ply element.
   - (e) A flat element with $R_o$'s axis 3 normal: the same $C$ through `phase_table` on the
     shell path equals the laminate path.
3. **Mass identities** to $10^{-14}$:
   - The one-ply $m_e$, and the $m_e$ of a laminate with equal densities, equal the hex8
     consistent mass's scalar block.
   - The per-ply $m_e$ equals the sum of hex8-type integrals over the ply sub-intervals.
   - $\sum m_e = \rho V$.
4. **Zero modes:** exactly 6 eigenvalues of $K_e$ below $10^{-10}\lambda_\max$ and the 7th above
   $10^{-4}\lambda_\max$, at $h/t = 1, 2, 100$, regular and distorted (prototype: 7th
   $\ge 7\cdot10^{-3}$ at $h/t = 2$).
5. **Patch tests:**
   - Oracle: under a linear field $u = Ax$ with constant $\sigma_0$, the free-node forces must
     equal the consistent face tractions. For constant $\sigma_0$ these are exactly hex8's
     $Ku$ on the same mesh. Error relative to $\max|f|$, $\le 10^{-10}$, without a rigid offset
     added to $u$.
   - Gated cases:
     - (i) A parallelogram patch, all 6 linear states.
     - (ii) The flat distorted MacNeal–Harder patch (5 elements), states with
       $\sigma_{13} = \sigma_{23} = 0$ (in the shell frame; for anisotropic $C$ the strain is
       solved for).
     - (iii) The same patch rigidly tilted into a general plane, membrane states.
     - (iv) A lifted patch with both surfaces lifted equally (interior nodes out of plane by
       $0.1t$), membrane states, for `shell_recover`: constant $\sigma_0$ at every point to
       $10^{-10}\|\sigma_0\|$.
   - Reported only: the warped patch's residuals. Bathe–Dvorkin ANS is not variationally
     consistent on non-parallelogram or warped elements; the prototype shows shear states at
     0.04–0.6 there. This is inherent to the element.
   - Logged reading: the roadmap's "flat patch test with lifting" is (iii) plus (iv).
6. **MacNeal–Harder** (1985):
   - Setup: beam $6\times0.2\times0.1$, $E = 10^7$, $\nu = 0.3$, $6\times1\times1$ elements
     (`direction=` thickness), root face $u = 0$. A unit tip load is distributed by the
     consistent face traction.
   - References: extension $3.0\cdot10^{-5}$, in-plane shear 0.1081, out-of-plane 0.4321,
     twist 0.03208.
   - Gates (normalized tip deflection): regular mesh $\ge 0.95$ for extension, in-plane and
     out-of-plane (prototype 0.988, 0.972, 0.979); parallelogram mesh $\ge 0.9$ for extension
     and out-of-plane (0.994, 0.976).
   - Reported only: parallelogram in-plane (0.62), trapezoidal, and twist.
   - Solved by `spsolve` on `assemble`.
7. **`assemble` on hex8 + `ssh8`:** symmetric PSD, 6 rigid modes in $\mathrm{null}(K)$
   ($\|KR\| \le 10^{-12}\|K\|\|R\|$), total mass $\rho V$. A one-ply `ssh8` block with
   $R_o$ normal matches the hex8 mass exactly.
8. **`jax.grad`:** $\partial(u^\top K_e v)/\partial X$, $\partial/\partial C$,
   $\partial/\partial\theta$, $\partial/\partial\text{frac}$, $\partial(u^\top m_e v)/\partial X$
   and $\partial/\partial\rho$ against central differences to $10^{-6}$ relative. No NaN with
   padded plies.
9. **`solid_shell`:**
   - Thickness detection:
     - The stacked plate $8\times6\times2$ under each of the 24 connectivity rotations gives
       the same thickness axis and a consistent $\zeta$ sign.
     - The cube $3^3$ and the strip $6\times1\times1$ raise.
     - A tilted plate with `direction` works.
   - Output checks: $\det J > 0$; element sets and labels follow `order`; `voxel` is None.
10. **Refusals:**
    - `mass="lumped"`.
    - A `Laminate` on a hex block.
    - Fractions not summing to 1.
    - `box_mesh(type="ssh8")`.
    - The bilinear forms.

## The operator and API (16b)

- **Stored leaves:** an `ssh8` block in `unstructured(..., mode="stored")` stores the packed
  condensed $K_e$ (300 floats, the hex8 leaf) and the packed $m_e$ with $\rho$ folded in
  (36 floats, new leaf `me`, no `wdet`).
  - `apply_K`, `diag_K`, `block_diag_K` and `scaled` are unchanged.
  - `apply_M` is $(c,8,8)\,@\,(c,8,3m)$ times $s_M$, and `diag_M` the diagonal of $m_e$ times
    3 DOFs.
  - `element_matrices` returns $K_e$ and $m_e \otimes I_3$.
  - Padded elements have $m_e = 0$.
  - `mode="otf"` with an `ssh8` block raises in `unstructured` and in `unstructured_memory`.
- **Estimator:** `per_el = 300 + 36` for `ssh8`, the `apply_M` temporaries for $m_e$, and one
  `BUILD_BYTES["stored"]["ssh8"]` from `memory_analysis`. It is checked at 1 and 3 plies
  (within 10%, so the scan holds).
  - Slow test: a $780\times780\times1$ single-layer plate with a 3-ply laminate block ($\approx$
    600k elements, estimate $\ge 2$ GB). The estimate is within 20% of the `harness.measure`
    peak of the build + `apply_K` + `apply_M`.
- **Probe sampling:** the exact factor for `ssh8` is $\mathrm{chol}(m_e)$ (8×8), with
  `xi_shapes` width 8. Padded elements are masked.
- **`solve_modes`:**
  - `precond="auto"` → `"jacobi"` on a mesh with an `ssh8` block (until PR 18).
  - Explicit `"amg"` is allowed: the Hex8Pattern Galerkin of the stored $K_e$ is exact, and
    `_amg_scan` treats `ssh8` as hex8.
  - `"gmg"` already raises (non-voxel).
  - `band`, `verify` and `solve_modes_memory` work unchanged.
  - Jacobi on `ssh8` is test-scale only. Measured with a hex20 proxy, the iterations grow
    $\propto (L/t)^2$ (8×8×1 plate: 211 at $L/t = 10$, 755 at 20, 11718 at 100), and stagnation
    appears from $L/t \approx 50$. `docs/eigensolvers.md` says so, with
    $\lambda_\max/\lambda_1 \approx c\,(L/t)^4$.
- **Reader:** `.inp` `SC8R`/`SC8` → `ssh8` with the file's node order (Abaqus: nodes 1–4
  bottom, 5–8 top, which is VTK $\zeta$), checked by $\det J > 0$ only. The `C3D(8|20)` regex
  is extended.
- **Writer:** VTU writes `ssh8` as hexahedra. `read` returns hex8 and `solid_shell` restores
  the type (documented). The type field is deferred (open_items).
- **M4 example** (`examples/m4_shell.py`): a simply supported square steel plate,
  $8\times8\times1$ `ssh8`, $L/t = 20$, 6 modes, Jacobi `solve_modes` with `maxit=2000`,
  against `reference_modes` to the PR 9 tolerances. The SS dict is inline, and a test asserts
  it equals `tests/exact.py`'s `SIMPLY_SUPPORTED`.

### 16b tests

1. `apply_K`, `apply_M`, `diag_K`, `diag_M` and `element_matrices` against `assemble` to
   $10^{-12}$ relative: `ssh8` and hex8 + `ssh8`, laminates, $m = 1$ and $m > 1$, random
   `s_K`/`s_M`, scatter and gather.
2. The M4 example (Jacobi) against `reference_modes`. `band=(0, f)` on it, `verify=True`, and
   `solve_modes_memory` smoke. Explicit `precond="amg"` gives the same modes. On a mixed
   hex8 + `ssh8` mesh, `auto` resolves to Jacobi.
3. Refusals: `otf` (builder and estimator), `"gmg"`; `levers` has no otf hint on `ssh8`.
4. The $\ge 2$ GB estimator test (slow), and the build constant at 1 vs 3 plies.
5. Probe factor $FF^\top = m_e$ to $10^{-12}$ (laminates, padded); a masked $SS^\top$ on a
   hex8 + `ssh8` mesh equals the SciPy M.
6. An SC8R `.inp` with one element at $h/t \approx 1$ reads as `ssh8` with the file's
   thickness axis; VTU write and read gives hex8, and `solid_shell` round-trips it.

## The accuracy suite (16c)

All four use PR 8's `reference_modes` on `assemble` (eigsh; Jacobi LOBPCG is impractical at
these $\kappa$). The relative error is in $f$. The gates are not upper-bound checks (ANS/EAS is
not a conforming Ritz method).

- **Sizing study first** (`benchmarks/shell_sizing.py`, ≤ 1 h, progress per level). It fixes
  $k$ per ply, the in-plane densities, the cylinder and skew levels, and $n_z$. The results go
  to the 16c addendum.
- **Srinivas–Rao** (0.1%, `plate_modes`; PR 8b's `PLATES`, `ORDER` and `plate_fe` with the type
  `ssh8` via `solid_shell`):
  - Cases: the cross-ply and single-ply plates at $L/t = 10$ and 100, $k$ stacked elements per
    ply.
  - Each of the lowest 10 full-plate modes, identified by its $(m, n, \text{family})$ label, is
    gated as the matching mode of the half-wave sub-plate $L/m \times L/n$ ($L$ for $n = 0$).
    The sub-plate uses `SIMPLY_SUPPORTED`, and its mode is the full-plate mode restricted to the
    sub-plate, with the same `plate_modes` value. This avoids the $h^2$ in-plane cost of high
    $(m, n)$ on the full plate (prototype: modes 5–6 need $\approx 150$ elements per side).
  - Expected $k = 4$–8 at $L/t = 10$ and 1–3 at 100, with in-plane 24–48 per sub-plate
    ($\le 62$k DOF).
  - Reported only: the full plate at a stated mesh, and one layered element per laminate.
- **Clamped cylinder:**
  - Geometry: $R/t = 50$, $L/R = 2$, isotropic, $u = 0$ on both end faces.
  - Model: the 1/8 symmetry model (quarter circumference, half length), with $u_y = 0$ on
    $y = 0$, $u_x = 0$ on $x = 0$ and $u_z = 0$ at mid-length via `dof_mask`. Two classes,
    (S,S) and (S,A), each give one member of the lowest doublets.
  - Mesh: `box_mesh` mapped to $(\theta, z, r)$, with hex27 midside nodes on the exact
    surfaces.
  - Reference: hex27 levels $24\times12$, $48\times24$, $96\times48$ at $n_z = 2$ ($\le 280$k DOF,
    $\approx 2$ min at the finest in the pilot), plus an $n_z = 4$ check at the middle level.
    Per mode, the observed order $p$ comes from the 3 levels; convergence must be monotone with
    $1 \le p \le 5$, else a level is added. The reference error
    $|\lambda_\text{fine} - \lambda_\text{extrap}|$ must be $\le 0.02\%$ in $f$; $p$ and the
    estimate are recorded.
  - Gate: the `ssh8` mesh (sized by the study, accounting for membrane locking) meets 0.1% on
    the lowest 6 modes per class, the cut at a relative gap $\ge 10^{-2}$ (asserted).
  - Docs note that eigsh reaches about 280k DOF on thin shells.
- **No locking:**
  - Setup: the SS square isotropic plate, $6\times6\times1$ in-plane at $L/t = 10$ and 1000.
    The first 6 bending modes are selected by z-reflection parity.
  - Error: the max over them of $|f/f_\text{conv} - 1|$. $f_\text{conv}$ is a $48\times48\times1$
    `ssh8` run at the same $L/t$ (Richardson with its observed order, estimated error
    $\le 1/10$ of the tested error).
  - Floor: the eigsh floor $f_\lambda = 10\varepsilon\lambda_\max/\lambda$, with $\lambda_\max$
    from `eigsh(which="LA")`, is $\approx 4.5\cdot10^{-3}$ for mode 1 at $L/t = 1000$. Each
    tested error must be $\ge 10 f_\lambda$ (asserted).
  - Gate: the error at 1000 is $\le 1.5\times$ the error at 10.
  - Reported only: the hex8 baseline on the same mesh (prototype: $\lambda_1$ 2800× the
    Kirchhoff value).
- **Skew plate:**
  - Setup: parallelogram elements only, skew 30°, clamped, $L/t = 20$, the lowest 6 modes, cut
    at a gap $\ge 10^{-2}$.
  - Gate: within 1% of a hex27 reference with estimated error $\le 0.2\%$. The reference uses
    4 nested levels ($\le 150$k DOF) and the observed order.
- **Benchmarks:** `benchmarks/shell_references.py` (≤ 1 h, per-level progress) writes
  `benchmarks/results/shell_references.json`. The tests read it and recompute only the `ssh8`
  solves. One slow test recomputes the coarsest hex27 level against the JSON.
- **Slow marks:** every 16c test, the $\ge 2$ GB estimator, and the MacNeal–Harder
  parallelogram/trapezoid reports if over 10 s.

## Docs (each with its part)

- **16a:**
  - `docs/elements.md`: `ssh8`, the formulation, quadrature, frames, locking coverage, the
    patch-test limits, recovery and refusals.
  - `docs/mesh.md`: `ssh8`, `solid_shell`, the permutations, the thickness rules and the
    hand-built block caveat.
  - `docs/materials.md`: `Laminate`, the shell frame, ply angles and the stacking sign.
- **16b:** `docs/operators.md` (the `ssh8` leaves, estimator), `docs/eigensolvers.md` and
  `docs/preconditioners.md` (Jacobi auto, AMG allowed, the Jacobi scaling), the README
  example, and `docs/mesh.md` (SC8R, VTU).
- **16c:** `docs/verification.md` (the study, references, $p$, $k$, results).
- **Each part:** ROADMAP status and `docs/open_items.md`.

## Decisions without user input (logged in open_items)

- The three-way split, one plan with a 16c addendum after the sizing study.
- Type `ssh8` with a `solid_shell` producer.
  - Thickness rules: cosine margin 0.1, unique shortest column, $\zeta$ sign propagation and
    seed.
  - Proper-rotation permutations; no edge-length check.
  - `box_mesh(type="ssh8")` refused.
- `Laminate` inside `materials`, plies holding `Material`s, 2 points per ply fixed.
- The ply frame from $R_o$ axis 1 projected (Abaqus), at the element centre; angles in degrees.
- $m_e$ stored (36 floats), $\rho$ folded in, phase `rho` = 1 for laminates.
- Dispatch by type name; `RECOVER` in `elements`; the bilinear forms deferred to PR 21.
- Jacobi auto on `ssh8` until PR 18; explicit AMG allowed.
- Patch-test reading (tilted and lifted-equal patches gated, warped reported) and the ANS
  limit.
- Srinivas–Rao per half-wave sub-plate; the cylinder on the 1/8 model; observed-order
  Richardson.
- The no-locking mesh and floor; the M4 example at $L/t = 20$.
- SC8R keeps the file's stacking; the VTU type field deferred.
