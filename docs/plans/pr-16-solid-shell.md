# PR 16 plan: solid-shell element (KGW 1999)

Branch `pr-16-solid-shell` (this plan), from `main` after PR 15d. Roadmap: PR 16 (M4), the
"Shells" and "Mass" key design decisions, PR 8b's `SIMPLY_SUPPORTED`, and the dependency row
16 ← 3, 5, 8, 9.

## Revisions

- **Revision 1:** draft.
- **Revision 2** (round 1, all asked for design changes):
  - laminates in `materials`;
  - the ply frame from the label;
  - a stored $m_e$;
  - a ply `scan`;
  - proper permutations;
  - a passable patch test;
  - per-sub-plate Srinivas–Rao;
  - a symmetry-model cylinder;
  - a locking floor;
  - Jacobi tests at $L/t \le 20$.
- **Revision 3** (round 2: 2 design, 3 minor):
  - the transform `bond_strain`$(J^{-\top})$;
  - the $s_M$-only mass scale;
  - laminate branches;
  - direct kernel calls;
  - the per-element $\zeta$ sign;
  - AMG refused;
  - measured sizes.
- **Revision 4** (round 3: 2 design, 3 minor):
  - user decisions (the Dirichlet-lift patch test; the cylinder gate kept at 0.1%);
  - quarter sub-plates;
  - `amg()`/`pmg()` refusal;
  - radians;
  - the ply container and $\zeta$ mapping.
- **Revision 5** (round 4: 2 design, 3 minor):
  - **Patch lift:** loaded by the consistent surface traction ($K_\text{hex8}Ax$), with the
    lateral boundary prescribed.
  - **Estimator:** a constant build envelope (it never sees `materials`).
  - **16a staging:** includes the otf refusal.
  - **Quarter sub-plates:** handle $m = 0$ or $n = 0$; Srinivas–Rao $k$ sized on the thickest
    sub-plate ($k = 12$).
  - **Cylinder reference:** graded toward the clamped ends; measured errors corrected.
  - **One-layer floors:** stated for cylinder and skew.
  - **Use sites:** tagged by part; mass branches confined to the 4 mass sites; `"pmg"` is not a
    `solve_modes` option.
  - **M4:** at the PR 9 tolerances.
  - **`solid_shell`:** sweep details; a second call appends.
- **Revision 6** (round 5: 1 design, 4 minor):
  - **Srinivas–Rao sizing:** by total elements through the thickness ($n_z = 36$: `pag`
    $k = 12$ per ply, `ortho` 36), measured on (2,3) and (3,1), not (2,2).
  - **Zero-index quarters:** width, class labels and sub-plates stated.
  - **M4 comparison** in `tests/test_api.py`; measured iterations at $10^{-10}$.
  - **Cylinder $n_z$ term** at the middle level, scaled with $p = 2$; the `ssh8` mesh uniform.
  - **Patch frames:** hex8 load built with $R_\text{shell}$; tolerances normed.
  - **Fallback flip** $\mathrm{diag}(-1,1,-1)$; visited mark per (element, axis); prismatic
    defined.
  - **Staging:** one up-front `ssh8` check in `unstructured`; one "auto resolved" label in
    the three option errors; the box type tuple; empty and all-hex8 selections.
  - **Tests added:** padded plies, per-element `direction`, the cosine-margin raise, the band
    run assertions; the mass scan split from the stiffness scan.

## Split and plans

PR 16 is three PRs, each from `main`, merged in order: `pr-16a-element`, `pr-16b-operator`,
`pr-16c-accuracy`.

- **16a, the element:** `ssh8`, `solid_shell`, `Laminate`, the kernels, `assemble`, recovery,
  the VTU writer mapping, element-level tests. Staging, so nothing reaches a hex code path:
  - `unstructured` (stored) raises `ValueError("ssh8 operators arrive in PR 16b")` up front on
    any `ssh8` block; `kernels.stiffness`/`mass` also raise for `ssh8` (direct calls);
  - `unstructured(..., mode="otf")` on an `ssh8` mesh raises `ValueError` before `layout` (the
    roadmap's otf refusal, kept in 16b);
  - `unstructured_memory` and `amg_memory` raise `ValueError("ssh8 operators arrive in PR 16b")`.
  - `stencil` cannot see `ssh8` (`voxel` is None).
- **16b, the operator and API:** the stored operator, the estimator, sampling, `solve_modes`, the
  `.inp` reader, the M4 example. In 16b, `amg_memory`'s raise names PR 18.
- **16c, the accuracy suite:** a sizing study, then its own short plan (meshes, levels,
  benchmark layout; the gates are the roadmap's, narrowed only as logged), reviewed as usual. If
  a gate or a reference criterion fails after the study, stop and ask the user.

## Data model

- **Element type `ssh8`:** hex8 topology (8 nodes, VTK order), added to `TYPES` (last) and
  `NODES`. Reference $\zeta$ (corners 0–3 → 4–7) is the thickness; $+\zeta$ orders the plies
  bottom to top.
  - `box_mesh` checks `type` against `("hex8", "hex20", "hex27")` (not `NODES`), so
    `type="ssh8"` raises and no error text lists `ssh8`.
  - The registry entry is hex8's (nodes, faces, 2×2×2 rule), with `lumping=None`
    (`lumping: str | None`). It is commented "quadrature for `check_elements`, `jacobian` and
    the sampling width only".
- **`solid_shell(mesh, elements=None, direction=None) -> (mesh, order)`** (`mesh/shell.py`, host
  NumPy):
  - Selection: an element set name, an index array, or all hex8 when None. Non-hex8 elements
    in the selection raise; an empty selection raises `ValueError`.
  - The selected elements are appended to the `ssh8` block (created if absent), with their
    connectivity permuted so $\zeta$ is the thickness. A second call can therefore convert
    another region with another `direction`. An emptied hex8 block is dropped (`Mesh` rejects
    empty blocks; blocks stay in `TYPES` order).
  - `material`, `orientation` and element sets follow.
  - `order` is new → old over the whole mesh, like `renumber`'s `q` (but not a valid `q`: it
    moves elements across blocks); `voxel` becomes None.
  - Besides the `.inp` reader it is the only producer. A hand-built block is the user's
    responsibility.
- **Node permutations:** fixed proper rotations take a reference axis to $\zeta$, and the 180°
  rotation about $\xi$ flips $\zeta$ (and $\eta$). $\det J$ keeps its sign.
- **Thickness axis:**
  - `direction` given: a (3,) vector, or (n_sel, 3) in sorted selected-element order. The axis
    is the reference axis whose mean edge vector (over its 4 parallel edges) has the largest
    $|\cos|$, with $\zeta$ along $+$direction. If the two largest differ by $< 0.1$,
    `ValueError` names the element.
  - `direction=None` (sweep), with face keys as in `parts`:
    - A chain runs from an element through the opposite faces of a reference axis. At each
      shared face it continues in the neighbour along that neighbour's reference axis whose
      face pair contains the face (orientations differ between elements). It stops at
      unselected elements and boundary faces.
    - Each chain is walked once, with a visited mark per (element, axis) (a closed ring's count
      is its length), and
      its element count is given to all members ($O(n_\text{el})$).
    - The thickness axis is the unique smallest of an element's 3 counts; a tie raises
      `ValueError("ambiguous thickness direction: pass direction=")`.
    - Raising: the $3^3$ cube, the strip $6\times1\times1$, and $8\times8\times12$. Picked
      wrongly: $8\times10\times12$ (picks $x$).
    - Sign: $\zeta$ along the positive global axis most aligned with the mean thickness edge
      (`argmax`, first on ties). That is consistent on flat plates, and may flip on curved
      shells.
  - **When the sign matters:** under a $\zeta$ flip the plies are reversed, the angles negated,
    and each ply turned 180° about its own material axis 1. So `direction` is required for
    off-axis plies (even $[\pm45]_s$), non-orthotropic plies, or unsymmetric stacking
    (documented in `docs/mesh.md`, not enforced).
- **Laminates in `materials`:** `materials={label: Material | Laminate}`.
  - `Laminate(plies)` holds a tuple of `Ply(material, angle, fraction)` NamedTuples:
    - `material`: a `Material`;
    - `angle`: radians about the normal from local axis 1;
    - `fraction`: bottom to top.

    Both are pytrees.
  - **Ply mapping:** ply $k$ occupies $\zeta \in [-1 + 2\sum_{j<k}f_j,\ -1 + 2\sum_{j\le k}f_j]$.
    Its 2 Gauss points are mapped linearly, with weights scaled by $f_k$, so a fraction-0 ply
    has weight 0.
    - A trilinear map is linear in $\zeta$ along each fibre, so reference fractions equal the
      physical thickness fractions on any element.
    - Ply volumes follow $\det J$.
    - Two points per ply are exact in $\zeta$ only on prismatic elements (top face = bottom
      face + a constant vector $d$, so $J$ does not depend on $\zeta$).
  - `check_materials` (host, before tracing) branches on `Laminate`: `check_stiffness` and $\rho$
    per ply, fractions $> 0$ summing to 1 to $10^{-12}$, angles finite.
  - `phase_table` gives a `Laminate` phase a finite placeholder (ply 0's rotated $C$ and $\rho$),
    read by no `ssh8` code.
  - `assemble` and `unstructured` raise after `phases` when a hex block uses a `Laminate`
    phase; `stencil` raises on any `Laminate` before `phase_table`.
  - A `Material` on `ssh8` is one ply (angle 0, fraction 1).
- **Shell frame (per element, at the centre; Abaqus-like):**
  - $t_3 = g_1\times g_2/\|g_1\times g_2\|$ at $\xi = 0$, with $J_{ij} = \partial
    x_i/\partial\xi_j$.
  - $t_1$ = axis 1 of $R_o$ (identity by default), projected onto $\perp t_3$ and normalized;
    $t_2 = t_3\times t_1$.
  - Fallback: if that projection has norm $< 0.1$, $t_2$ = axis 2 projected and normalized,
    and $t_1 = t_2\times t_3$. NaN-safe normalization (double `where`); not differentiable at
    the switch.
  - $R_\text{shell} = [t_1\ t_2\ t_3]$; ply $k$ uses $C_k' = \text{rotate}(C_k,
    R_\text{shell}R_z(\theta_k))$.
  - The frame does not depend on the 4 in-plane connectivity rotations. The $\zeta$ flip turns
    it by $\mathrm{diag}(1,-1,-1)$ in the primary branch and $\mathrm{diag}(-1,1,-1)$ in the
    fallback (there $\theta \to \pi - \theta$, equivalent to $-\theta$ for orthotropic plies).
  - When $R_o$'s axis 3 is $+t_3$, it equals $R_o$, so a one-ply `Material` acts as on hex8.
  - hex8 and `ssh8` parts with one label on a tilted plate are oriented differently
    (documented).
- **`ply_table(pairs, materials, orientations) -> (Ply(Material(C, rho), angle, fraction), R_o)`**
  (pure jnp, beside `phase_table`):
  - Shapes: `C (P, L, 6, 6)` unrotated; `rho`, `angle`, `fraction` `(P, L)`; `R_o (P, 3, 3)`.
  - Padding copies ply 0 at fraction 0. Built only with an `ssh8` block.
  - The kernels take one element's slice, `vmap`ped. `_build("ssh8", ...)` takes the tuple
    `(Ply, R_o)` as its table argument.

### Use sites (part tagged)

- **`elements` (16a):**
  - `kernels.stiffness`, `mass`, `lumped_nodal`, `stiffness_form`, `mass_form` raise for
    `ssh8`; the forms are deferred to PR 21.
  - `RECOVER = {"ssh8": shell_recover}` lives in `elements/__init__.py`, the registry PR 17
    extends. `registry.py` stays NumPy-only (logged deviation).
  - `check.py`: `mass="lumped"` raises.
- **`operators/assembled.py` (16a):** `_KERNELS`, `_chunked` (`jax.tree.map` over the ply
  pytree), `values()`. The `ssh8` branch calls the shell kernels on `ply_table` slices, and
  $m_e$ is expanded to $m_e\otimes I_3$ before the ravel.
- **`operators/matfree.py`:**
  - 16a: the otf raise.
  - 16b: `Block` gets a leaf `me` (packed 36, $\rho$ included); the `Block(...)` construction
    branches (`wdet` stays None for `ssh8`, `me` set).
  - 16b: `_build` returns `(Ke, me)` for `ssh8`; `_block_data`'s leaf list gains `me`.
  - 16b mass: `_m_apply`, `_m_diag`, `element_matrices` and `sampling.element_factor` each get
    an early `if b.type == "ssh8"` path using `d["s_M"]` and `unpack(d["me"])`. `_m_scale` and
    `_wdet` are unchanged.
  - Padding repeats element 0 (`layout`), so the padded `me` is element 0's and is zeroed by
    $s_M = 0$.
  - The module docstring is updated.
- **`operators/sampling.py` (16b):**
  - For `ssh8`, $F = \sqrt{s_M}\,\mathrm{chol}(m_e)$, shape `(c, 8, 8)`. The draw width 8
    equals `xi_shapes`'s `len(ELEMENTS["ssh8"].weights)` because the entry copies hex8's
    2×2×2 rule (commented).
  - The docstrings are fixed.
- **`precond/pmg.py` `_check` (16b):** shared by `amg` (amg.py:166) and `pmg` (pmg.py:246).
  An `ssh8` block raises `ValueError` naming PR 18; `amg.py:188` would otherwise treat it as
  hex8.
- **`memory.py`:**
  - 16a: the two raises.
  - 16b: `per_el = 300 + 36`, `_chunk_temps` for `me`, and `BUILD_BYTES["stored"]["ssh8"]`.
  - 16b: the `amg_memory` raise names PR 18 and stays ahead of `_amg_scan`.
  - otf raises in both parts.
- **`eigen/api.py` (16b):**
  - `resolve_precond`: auto → `"jacobi"` on `ssh8`; `"amg"` raises naming PR 18. `"gmg"`
    already raises; `"pmg"` is not an option.
  - `levers`: no otf hint.
  - When the caller asked for `"auto"`, the `sigma_options` and `precond_options` errors say
    "auto resolved to jacobi on ssh8". One helper builds that label and is used at the three
    error sites (api.py `precond_options` and `sigma_options`, and memory.py `_models`).
  - The `solve_modes` docstring is updated.
- **`mesh/io.py`:**
  - 16a: `write` maps `"ssh8"` to `"hexahedron"`; `MESHIO` is unchanged.
  - 16b: the `.inp` regex adds `SC8R|SC8`, and other `SC*` raise.
- **Exports (16a):** `mesh`: `solid_shell`; `materials`: `Laminate`, `Ply` (`ply_table` is
  internal, beside `phase_table`); `elements`: `shell_stiffness`, `shell_mass`, `shell_recover`,
  `RECOVER`.

## The element (16a)

Notation:
- $g_j = \partial X/\partial\xi_j$, $J = [g_1\ g_2\ g_3]$, $J_0$ at the centre.
- Covariant strains $E_{ij} = \tfrac12(g_i\cdot u_{,j} + g_j\cdot u_{,i})$, in `VOIGT` order with
  engineering shear.
- $\varepsilon = TE$ with $T = $ `bond_strain`$(J^{-\top})$.

Formulation:
- **ANS, transverse shear** (Bathe–Dvorkin):
  $E_{13}^{ANS} = \tfrac12(1-\eta)E_{13}(0,-1,\zeta) + \tfrac12(1+\eta)E_{13}(0,1,\zeta)$, and
  $E_{23}$ likewise from $(\mp1, 0, \zeta)$. Both are linear in $\zeta$, so the tying points
  need no separate $\zeta$ sampling and are evaluated at each point's $\zeta$.
- **ANS, thickness** (Betsch–Stein): $E_{33}$ bilinear from the 4 corner lines, evaluated once
  at $\zeta = 0$ (it does not depend on $\zeta$).
- **EAS**, 5 parameters, covariant: $\tilde E_{11} = \xi\alpha_1$, $\tilde E_{22} = \eta\alpha_2$,
  $2\tilde E_{12} = \xi\alpha_3 + \eta\alpha_4$, $\tilde E_{33} = \zeta\alpha_5$. Then
  $\tilde\varepsilon = (\det J_0/\det J)\,T_0M\alpha$ with $T_0 = $ `bond_strain`$(J_0^{-\top})$,
  and $\int M\,d\xi = 0$.
- **Condensation:** $K_e = K_{uu} - K_{u\alpha}K_{\alpha\alpha}^{-1}K_{\alpha u}$ (5×5
  Cholesky), symmetrized as `_sym`; $\alpha = -K_{\alpha\alpha}^{-1}K_{\alpha u}u_e$.
- **Quadrature:** 2×2 in-plane times 2 per ply. In `shell_stiffness`, `lax.scan` over plies
  accumulates $K_{uu}$, $K_{u\alpha}$, $K_{\alpha\alpha}$; `shell_mass` has its own ply loop.
  `assemble` and `_build` call the two kernels separately.
- **Mass:** $M_e = m_e\otimes I_3$, $m_e = \sum_k\rho_k\int_{\text{ply }k}N^\top N\det J$.
  Lumping raises "solid-shell elements are consistent-mass only".
- **Locking:**
  - Covered: transverse shear, curvature thickness, Poisson thickness, in-plane shear,
    parasitic bending.
  - Not covered: membrane locking of curved elements, and trapezoidal in-plane bending
    (`docs/elements.md`).
- **Kernels** (`elements/shell.py`):
  - `shell_stiffness(X, plies, R_o) -> Ke (24, 24)`.
  - `shell_mass(X, plies) -> m_e (8, 8)`, from `rho` and `fraction` only.
  - `shell_recover(X, plies, R_o, u) -> (points, weights, eps, sigma)`: physical point
    coordinates, $w\det J$, and global Cartesian Voigt strain and stress from the ANS strains
    plus the EAS strains with the recovered $\alpha$.
  - PR 17 owns the postprocessor stress test, and its plan must include `ssh8` (logged).

### 16a tests

1. **Independent NumPy kernel** (`tests/shell_ref.py`: point loops, explicit $T$, ANS, EAS,
   dense condensation, the plan's frame and fallback). Random distorted elements (corners
   $\le 0.2h$, by `helpers.distort`), anisotropic $C$, 1–3 plies. $K_e$ and $m_e$ to
   $10^{-12}\max|K_e|$.
2. **Laminate path,** to $10^{-12}$:
   - (a) Invariant under the 4 in-plane connectivity rotations; under the $\zeta$ flip, plies
     reversed and angles negated (orthotropic plies), in the primary branch and in the
     fallback ($R_o$'s axis 1 near the normal).
   - (b) A rigid rotation $Q$ of the element and $R_o$ gives $QK_eQ^\top$.
   - (c) One ply at $\theta$ equals one ply with `rotate(C, R_z(θ))` at 0.
   - (d) On a prismatic element, identical plies equal one ply.
   - (e) `Laminate((Ply(m, 0, 1),))` equals `Material m`. A flat element in the $xy$ plane
     with $\zeta$ along $+z$ and $R_o = R_z(\varphi)$ equals `shell_stiffness` with
     `rotate(C, R_o)` and $R_o = I$.
3. **Mass,** to $10^{-14}$:
   - the one-ply $m_e$ equals hex8's scalar block on any element;
   - on a prismatic element, an equal-density laminate equals it;
   - per ply, the sum of hex8-type integrals over the sub-intervals;
   - $\sum m_e = \sum_k\rho_kV_k$.
4. **Zero modes:** gated at $h/t = 1, 2$, regular and distorted ($\le 0.2h$): 6 eigenvalues
   $\le 10^{-10}\lambda_\max$ and the 7th $\ge 10^{-3}\lambda_\max$ (prototype
   $\ge 1.4\cdot10^{-2}$). Reported only: $h/t = 100$.
5. **Patch tests** (user decision, 2026-10-17), one layer, $t = 0.01$, anisotropic $C$.
   Linear $u = Ax$ with constant $\sigma_0$, the strain solved for each target stress state.
   Frames: on (i), (ii) and (iv) $\zeta$ is along $+z$ with $R_o = I$, so $R_\text{shell} = I$;
   on (iii) $R_o = Q$ (the tilt), so $R_\text{shell} = Q$. $\sigma_0$ and $\varepsilon$ are
   given in that frame, and the hex8 mesh uses `Material(rotate(C, R_shell), rho)` with
   $R_o = I$, so both meshes carry the same $C'$.
   - **Lift (gated):** $Ax$ is prescribed on the lateral-boundary nodes. The interior DOFs
     (on the top and bottom faces) are loaded by the consistent traction of $\sigma_0$,
     $f_I = (K_\text{hex8}Ax)_I$ from `assemble` on the same mesh as hex8 (exact, since hex8
     passes the patch test), and $K_{II}u_I = f_I - K_{IB}(Ax)_B$ is solved by `spsolve`.
     $u_I = (Ax)_I$ and `shell_recover` gives $\sigma_0$, both to $10^{-10}$ relative to
     $\max|Ax|$ and $\max|\sigma_0|$ (prototype $\le 2.3\cdot10^{-13}$), on:
     - (i) a flat $3\times3$ parallelogram patch: all 6 states;
     - (ii) the flat distorted MacNeal–Harder patch (5 elements): the 4 states with
       $\sigma_{13} = \sigma_{23} = 0$ in the shell frame;
     - (iii) patch (ii) rigidly tilted into a general plane: the same 4.
   - **Recovery of the imposed field (gated):** all 6 states on (i)–(iii) and on (iv), the
     patch (ii) with both surfaces lifted equally by $0.1t$ (isotropic $C$), to $10^{-10}$
     (prototype $\le 2.3\cdot10^{-14}$).
   - **Reported only:**
     - the lift's transverse-shear states on (ii)–(iii), errors 3–7.5;
     - the lift on (iv): only $\sigma_{33}$ passes; the membrane states give $10^{-2}$–1.5
       (curved geometry);
     - the warped patch, i.e. (ii) with one interior node of each surface moved out of plane
       by $0.1t$.

     Bathe–Dvorkin ANS is not variationally consistent on non-parallelogram elements.
   - Also checked: $\sum$ `weights` equals the element volume, and `sigma` = $C'$`eps` per point.
6. **MacNeal–Harder** (1985):
   - Setup: beam $6\times0.2\times0.1$, $E = 10^7$, $\nu = 0.3$, $6\times1\times1$,
     `direction=(0,0,1)` (thickness along $z$). The 45° parallelogram and trapezoid
     distortions lie in the shell ($xy$) plane. Root $u = 0$; unit tip loads by consistent
     traction.
   - References: $3.0\cdot10^{-5}$, 0.1081, 0.4321, 0.03208.
   - Gates: regular $\ge 0.95$ for extension, in-plane and out-of-plane (0.988, 0.972, 0.979);
     parallelogram $\ge 0.9$ for extension and out-of-plane (0.994, 0.976).
   - Reported only: parallelogram in-plane, trapezoid, twist.
7. **`assemble` on hex8 + `ssh8`:**
   - symmetric PSD, $\|KR\| \le 10^{-12}\|K\|\|R\|$, total mass $\sum\rho_kV_k$;
   - two laminates with different ply counts (padding): $K$ and $M$ equal the per-element
     unpadded `shell_stiffness`/`shell_mass` assembled by SciPy, to $10^{-12}$;
   - a one-ply `ssh8` block's $M$ equals hex8's;
   - `parts`, `rigid_modes` and `dof_mask` on it.
8. **`jax.grad`** of $u^\top K_e v$ with respect to $X$, $C$, angle and fraction, and of
   $u^\top m_e v$ with respect to $X$ and $\rho$, against central differences to $10^{-6}$. No NaN
   with padded plies or with $R_o$'s axis 1 or 2 exactly normal.
9. **`solid_shell`:**
   - The $8\times6\times2$ plate under the 24 connectivity rotations gives the same axis and
     $\zeta$ along $+z$; the raising cases raise; $8\times10\times12$ picks $x$; a tilted plate
     with `direction` works.
   - A full hex8 cylinder ring (closed circumferential chain) is detected.
   - A second call appends; selecting every hex8 element drops the hex8 block; an empty
     selection raises.
   - A per-element `direction` array: a laminated hex8 ring with radial directions gives
     $\zeta$ along each radius.
   - Cosine margin: an element whose two best axes differ by $< 0.1$ raises naming it.
   - Output: $\det J > 0$; sets and labels follow `order`; `voxel` is None; non-hex8
     selections raise. On an all-`ssh8` mesh, VTU write, read, `solid_shell(direction=)`
     round-trips up to `order` (`write` drops sets and reads back hexahedra as hex8).
10. **Refusals:**
    - lumped mass;
    - a `Laminate` on a hex block, and in `stencil`;
    - bad fractions or ply $C$;
    - `box_mesh(type="ssh8")`;
    - the forms;
    - `unstructured` on `ssh8` in stored and otf modes;
    - `unstructured_memory` and `amg_memory`.

## The operator and API (16b)

- **Stored leaves:** the packed condensed $K_e$ (300) and `me` (36), 336 floats per element.
  - `apply_K`, `diag_K`, `block_diag_K` and `scaled` are unchanged.
  - `apply_M` is $s_M\cdot(c,8,8)\,@\,(c,8,3m)$; `diag_M` is $s_M\,\mathrm{diag}(m_e)$ per
    DOF; `element_matrices` gives $K_e$ and $m_e\otimes I_3$.
- **Estimator:** the estimators take no `materials`, so `BUILD_BYTES["stored"]["ssh8"]` is a
  constant upper envelope for $L \le 16$ plies: $60.2\cdot10^3$ bytes per element times $c$.
  - It follows the measured $55.4 + 0.30L$ kB (prototype: 55.69, 57.52 and 58.73 kB at 2, 8 and
    12 plies; 47.9 at 1, where XLA inlines the one-trip scan). Past 16 plies it underestimates
    the build by about 0.3 kB per ply (documented).
  - Test: at 2, 8 and 12 plies the constant is $\ge$ measured and $\le 1.10\times$ measured;
    1 ply is below.
  - Slow test: $820\times820\times1$ with a 3-ply block (672k elements, 4.04M DOF), both paths,
    $m = 1$ and 16. Estimates 2.2–5.7 GB, each asserted $\ge 2$ GB; scatter $m = 1$ is
    build-dominated. Each is within 20% of the `harness.measure` peak of build + `apply_K` +
    `apply_M`. The build is only 13–18% of the total, so the envelope test, not this one,
    checks the build term. About 3–4 min.
- **Probe sampling:** the exact factor, so band counts are exact.
- **`solve_modes`:**
  - auto → Jacobi; `"amg"` refused until PR 18.
  - `band`, `verify`, `strategy` and `solve_modes_memory` are unchanged.
  - Jacobi on `ssh8` is test-scale only (prototype 8×8×1 steel, tol 1e-6): 57 iterations at
    $L/t = 10$, 233 at 20, 1640 at 50, stagnation at 100.
  - `docs/eigensolvers.md` gives these numbers, $\lambda_\max/\lambda_1 \approx c(L/t)^4$, and
    the M4 times below.
- **Reader:** `.inp` `SC8R`/`SC8` → `ssh8` in the file's node order (1–4 bottom, 5–8 top = VTK
  $\zeta$); $\det J > 0$ is checked by `check_elements` at build, as for hex.
- **M4 example** (`examples/m4_shell.py`):
  - Setup: an SS square steel plate, $8\times8\times1$ `ssh8`, $L/t = 20$, 6 modes, Jacobi,
    `maxit=2000`.
  - Iterations at $10^{-10}$: measured 367 (1.6× the 233 at $10^{-6}$); `maxit` is 3000
    (margin 8×).
  - The SS dict is inline; a test asserts it equals `SIMPLY_SUPPORTED`. `test_examples` runs
    the example (it asserts "check passed").
  - The roadmap's comparison is in `tests/test_api.py`, on the example's setup: the PR 9
    tolerances (tol $10^{-10}$; $\lambda$ error $\le \max(10^{-8},
    10\varepsilon\hat\lambda_\max/\lambda)$ by `check_lam`; principal angles by `max_sine`
    and `r_M`, refactored to take `(Kf, Mf)`, with the degenerate pairs (1,2)/(2,1) and
    (1,3)/(3,1) compared as subspaces). The principal-angle rule needs the reference residual
    $\le 0.1 r_M$; the measured margin is recorded. If it does not hold, the test uses the
    11b/13b form ($\lambda$ to $10^{-8}$, MAC $\ge 1 - 10^{-8}$), logged.

### 16b tests

1. `apply_K`, `apply_M`, `diag_K`, `diag_M` and `element_matrices` against `assemble` to
   $10^{-12}$:
   - meshes: `ssh8`, and hex8 + `ssh8` sharing label 0;
   - laminates;
   - $m = 1$ and $>1$;
   - random `s_K`/`s_M`;
   - scatter and gather.
2. `jax.grad` through the stored build: `matfree._build("ssh8", "stored", points, conn, phase,
   (plies, R_o))`, then the packed-$K_e$ apply. $u\cdot Kv$ with respect to `points` and a ply
   angle, against central differences to $10^{-6}$.
3. M4 runs (prototype times):
   - `band=(0, f)`: 154 iterations, 2.5 s; with `verify=True`, 3.8 s;
   - `band=(f_1, f_2)` with `strategy="shift_invert"` (Jacobi MINRES): 2.8 s; with
     `verify=True`, 2.7 s;
   - both band runs assert $\lambda$ against `reference_modes` to the PR 15 tolerances, the
     count and multiplicities, and `verify` true;
   - `solve_modes_memory` smoke;
   - auto → Jacobi on hex8 + `ssh8`.
4. Refusals:
   - otf;
   - `"amg"` through `solve_modes` and `solve_modes_memory`;
   - `amg()`/`pmg()` directly;
   - `amg_memory` naming PR 18;
   - `"gmg"`;
   - no otf lever;
   - the "auto resolved" messages, through `solve_modes` and `solve_modes_memory`.
5. The build envelope; the $\ge 2$ GB estimator test (slow).
6. $FF^\top = m_e$ to $10^{-12}$ (laminates); padded elements give $F = 0$; a masked $SS^\top$ on
   hex8 + `ssh8` equals the SciPy $M$.
7. An SC8R `.inp` (one element, $h/t\approx1$) reads as `ssh8` with the file's thickness axis;
   `SC6R` raises.

## The accuracy suite (16c: design frame)

The 16c plan fixes meshes, levels and the benchmark layout after the sizing study.

- **Common rules:**
  - `reference_modes` on `assemble`; relative error in $f$; no upper-bound checks.
  - Gates compare against the finest reference level. The $n_z$ term is the change from
    $n_z = 2$ to 3 at the middle reference level, scaled by $1/(1 - (2/3)^p)$ with $p = 2$
    (conservative). The observed-order Richardson estimate
    (recomputed by the tests from the recorded levels, $p \in [1, 5]$, monotone, else fail)
    plus the $n_z$ term bounds the reference error, which must meet the roadmap's criterion
    ($\le 0.02\%$ for the cylinder, $\le 0.2\%$ for the skew plate). If it does not, stop and
    ask.
  - Every gated run asserts $2f_\lambda \le 0.1\times$ its gate, with
    $f_\lambda = 10\varepsilon\hat\lambda_\max/\lambda$ computed in the test.
  - Equivalence tests compare to $\max(10^{-10}, 2f_\lambda)$.
- **Benchmarks:**
  - Each run is $\le 1$ h with per-level progress; results are recorded with their commit.
  - The slow tests recompute the smallest-margin gated case per test plus the coarsest
    reference level, against the record to $\max(10^{-9}, 2f_\lambda)$. Prototype costs:
    cylinder about 35 s, Srinivas–Rao 60–80 s ($\le 9$ GB; `ortho` (3,1) or `pag` (2,3)), skew
    $< 10$ s, no-locking $< 2$ s.
  - `docs/eigensolvers.md` records the eigsh times and memory on these thin meshes (up to
    about 280k DOF, 14 GB).
- **Srinivas–Rao** (0.1%):
  - Plates: `ortho` and `pag`, at $L/t = 10$ and 100. `plate_fe` is generalized to `H`, `a`,
    `b`: hex8, then `solid_shell(direction=(0,0,1))`. The `pag` 90° ply is given by a ply
    angle.
  - Each of the lowest 10 modes is gated on its half-wave sub-plate $L/m\times L/n$, solved as
    a quarter:
    - SS on xmin/ymin.
    - On each cut face: the mirror (xmax `"x"`, ymax `"y"`) for a nonzero index. For a zero
      index the field does not vary in that direction, so that face's SS condition goes on the
      cut (`"yz"` on xmax for $m = 0$, `"xz"` on ymax for $n = 0$).
    - Zero-index width: one in-plane element of size $h$ (the other direction's element size)
      across. With SS on both faces, the class then holds only $m' = 0$ (resp. $n' = 0$); its
      labels are those with that index 0 and the other index odd. The test asserts that no
      excluded label ($m' \ge 2$) lies below the target.
    - Matching: by the sorted index of the mapped label among the `plate_modes` labels of the
      same symmetry class on the quarter. The rule states which class each label falls in
      (including (1,1,1) on the S/S quarter), and the sizing study confirms every one of the
      lowest 10 is matched.
  - Equivalence tests:
    - the full plate at $2N\times N$ gives mode (2,1) equal to the $N\times N$ sub-plate's;
    - the quarter equals the sub-plate for (1,1), (1,0) and (0,1); the (1,0) and (0,1)
      sub-plates are $L/m\times L$ and $L\times L/n$ (full extent in the zero-index direction).
  - Sizing by total elements through the thickness $n_z$ (per laminate: $k = n_z/L_\text{plies}$),
    on the thickest sub-plates among the lowest 10: (2,3) (shortest side $1/3$, wavenumber
    $\sqrt{13}$), (1,3) and (3,1).
    - Prototype, quarters at 32×32 in-plane (64 per sub-plate):
      - `pag` (3 plies): (2,3) at $k = 8$ gives $8.3\cdot10^{-4}$, at $k = 12$ ($n_z = 36$)
        $4.7\cdot10^{-4}$ (margin 2.1; 114k DOF, 60 s, 7.9 GB).
      - `ortho` (1 ply) at $n_z = 12$ fails on (2,2), (1,3), (2,3) and (3,1)
        ($1.0$–$1.37\cdot10^{-3}$) and is thickness-dominated; (3,1) at $n_z = 24$ gives
        $5.0\cdot10^{-4}$ (77k DOF, 34 s).
    - So $n_z = 36$ for both at $L/t = 10$ (`pag` $k = 12$, `ortho` $k = 36$; about 120k DOF,
      about 60 s per run): about 19 sub-plates, about 20–25 min. The sizing study measures
      the margin on (2,3), (1,3) and (3,1) for both laminates.
    - If a laminate needs more, $n_z$ is raised for that laminate (logged).
    - $L/t = 100$: $n = 48$, $k = 1$, 2 s.
  - `docs/verification.md` states $n_z$ per laminate and $L/t$.
  - Reported only: one layered element per laminate (roadmap), and the full plate at a stated
    mesh.
- **Clamped cylinder:**
  - Geometry: $R/t = 50$, $L/R = 2$, isotropic, $u = 0$ on both end faces.
  - Model: 1/8 symmetry, 4 classes on one mesh by `dof_mask`:
    - circumferential S about $x = 0$ ($u_x = 0$) or A ($u_y = u_z = 0$);
    - $y = 0$ symmetric ($u_y = 0$);
    - axial S at mid-length ($u_z = 0$) or A ($u_x = u_y = 0$).

    Excluded: the $y = 0$-antisymmetric doublet partners and the $n = 0$ torsion modes (logged).
  - Matching: sorted per class, the lowest 6.
  - Reference:
    - hex27 levels at circumferential × axial × $n_z$ of $24\times12\times2$, $48\times24\times2$
      and $96\times48\times2$ (276k DOF, 112 s, 14 GB at the finest; about 9 min for 4
      classes).
    - With uniform axial spacing, the clamped-end layer slows $p$ to 1.8–3.5 and the reference
      error exceeds 0.02%.
    - So the axial `coords` are cosine-graded toward the clamped end ($y_i = 1 -
      \cos(\pi i/2n)$, as FV52) at the same sizes. Prototype on the worst class (S/A):
      monotone, $p = 3.2$–3.6, Richardson error $\le 0.0053\%$; the $n_z$ term at
      $48\times24$ (about 37 s per class) is $5.6\cdot10^{-5}$, scaled $\le 0.01\%$; total
      $\le 0.015\%$.
    - The sizing study confirms $\le 0.02\%$ on all 4 classes, or stops and asks.
  - `ssh8`: $384\times96\times1$ (220k DOF, 32 s per class).
    - Prototype worst errors per class: 0.051, 0.080, 0.064 and 0.063% (S/S, S/A, A/S, A/A),
      the worst S/A mode 4, so the margin is 1.25 (0.080% against the graded reference; 0.085%,
      margin 1.18, against its Richardson value).
    - One layer converges about 0.06% above 3D, so refinement cannot raise the margin.
    - The `ssh8` mesh is uniform (grading it gave 0.081%).
  - Gate: 0.1% (user, 2026-10-17). If it fails, stop and ask with the numbers.
- **No locking:**
  - Setup: the SS square isotropic plate, $6\times6\times1$, at $L/t = 10$ and 1000. 16 modes
    are computed; assert $\ge 6$ bending modes by z-parity and keep the first 6.
  - Error, in $\lambda$: max over the 6 of $|\lambda/\lambda_{48} - 1|$, with $\lambda_{48}$ a
    $48\times48\times1$ run. Its Richardson estimate (12/24/48, observed $p \approx 2.06$)
    plus the floor must be $\le 0.1\times$ the error at both $L/t$.
  - Floor: $10\varepsilon\lambda_\max/\lambda$, computed in the test (measured $9.2\cdot10^{-4}$).
  - Gate: error at 1000 $\le 1.5\times$ error at 10 (measured fractions 0.838/0.64 = 1.31).
  - Reported only: the hex8 baseline.
- **Skew plate:**
  - Setup: parallelograms only, 30°, clamped, $L/t = 20$, lowest 6 sorted.
  - Reference: hex27 levels 6/12/24/48 at $n_z = 2$ (135k DOF; measured reference error about
    0.03%), the $n_z$ term at level 24 by the common rule.
  - Gate: 1% (roadmap).
  - `ssh8` at $128\times128\times1$: about 0.71%. One layer converges about 0.61% above 3D
    (modes 5–6), so the margin is about 1.4 and refinement cannot raise it (floor, as for the
    cylinder). If it fails, stop and ask.

## Docs (each with its part)

- **16a:**
  - `docs/elements.md`: `ssh8`, formulation, quadrature, frames, locking, patch-test limits,
    recovery, refusals.
  - `docs/mesh.md`: `solid_shell`, permutations, the sweep, sign rules, caveats, VTU.
  - `docs/materials.md`: `Laminate`/`Ply`, radians, the ply mapping, the frame and fallback,
    the stacking sign, hex vs `ssh8` orientation.
- **16b:**
  - `docs/operators.md`: leaves, estimator envelope.
  - `docs/eigensolvers.md`, `docs/preconditioners.md`: Jacobi auto, AMG until PR 18, numbers.
  - README example.
  - `docs/mesh.md`: SC8R.
- **16c:** `docs/verification.md`, amending "Richardson extrapolation is not used" for PR
  16's reference estimates.
- **Each part:** ROADMAP status, `docs/open_items.md`.

## Decisions (logged in open_items)

User decisions (2026-10-17):

- The patch test is a Dirichlet lift (loaded by the consistent surface traction). The
  transverse-shear states are gated on parallelogram patches only; on distorted, tilted,
  lifted and warped patches they are reported. Recovery of the imposed field is gated for all
  6 states.
- The cylinder gate stays 0.1%; if it fails, stop and ask.

Without user input:

- **Process:** the three-way split; the 16a staging (kernel raises, the otf raise, the two
  estimator raises); 16c's own plan after the sizing study.
- **Type and producer:**
  - type `ssh8`, last in `TYPES`;
  - the registry entry with `lumping=None`;
  - `solid_shell` returning `order` and appending on a second call;
  - `box_mesh(type="ssh8")` refused;
  - hand-built blocks are the user's responsibility.
- **Thickness rules:** the per-element `direction` array; cosine margin 0.1; unique smallest
  chain count (with ring and stop rules); the per-element sign rule; proper permutations; no
  edge-length check; `direction` required for off-axis, non-orthotropic or unsymmetric
  laminates.
- **Laminates and frame:**
  - `Laminate`/`Ply` in `materials`, radians;
  - 2 points per ply, fixed (narrows "n per ply");
  - the reference-$\zeta$ ply mapping;
  - the placeholder phase entry;
  - the frame from $R_o$ axis 1, with the axis-2 fallback at 0.1;
  - hex8 and `ssh8` parts with one label on a tilted plate oriented differently (documented).
- **Mass:** $m_e$ stored (36 floats, $\rho$ included); the $s_M$-only scale in the 4 mass sites;
  the padded `me` as element 0's.
- **Code structure:** dispatch by name with `kernels.*` raising; `RECOVER` in
  `elements/__init__.py`; the forms deferred to PR 21.
- **Preconditioners:** Jacobi auto on `ssh8`; AMG and direct `pmg()` refused until PR 18.
- **Estimator:** a constant build envelope for $\le 16$ plies.
- **Element tests:**
  - zero modes gated at $h/t \le 2$ with the 7th $\ge 10^{-3}$ (intentionally stricter than
    the roadmap's $10^{-8}$), $h/t = 100$ reported;
  - MacNeal–Harder loading, plane and `direction`.
- **16b scope:**
  - Jacobi tests at $L/t \le 20$;
  - the M4 example at $L/t = 20$ with the PR 9 tolerances;
  - SC8R stacking kept, other `SC*` refused;
  - the VTU type field deferred.
- **16c:**
  - Srinivas–Rao per sub-plate as quarters (zero-index cuts SS, one element across), sized by
    $n_z = 36$ on (2,3), (1,3) and (3,1);
  - the cylinder classes and exclusions, with axial grading;
  - observed-order Richardson with the $n_z$ term;
  - the one-layer floors (cylinder, skew);
  - the no-locking reference and floor;
  - recorded benchmark results with recomputation of the smallest-margin case.
- **PR 17:** must cover `ssh8` recovery.
