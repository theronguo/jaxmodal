# PR 16 plan: solid-shell element (KGW 1999)

Branch `pr-16-solid-shell` (this plan), from `main` after PR 15d. Roadmap: PR 16 (M4), the
"Shells" and "Mass" key design decisions, PR 8b's `SIMPLY_SUPPORTED`, and the dependency row
16 ← 3, 5, 8, 9.

## Revisions

- **Revision 1:** draft.
- **Revision 2** (round 1, 5 reviewers, all asked for design changes):
  - laminates in `materials`;
  - the ply frame from the orientation label;
  - a stored 8×8 $m_e$;
  - a ply `scan`;
  - proper permutations;
  - a passable patch test;
  - Srinivas–Rao per sub-plate;
  - the cylinder on a symmetry model;
  - a concrete locking floor;
  - Jacobi tests at $L/t \le 20$.
- **Revision 3** (round 2: 2 design, 3 minor):
  - the transform `bond_strain`$(J^{-\top})$;
  - an $s_M$-only mass scale on `ssh8`;
  - laminate branches in `phase_table`/`check_materials`;
  - direct shell-kernel calls;
  - a per-element $\zeta$ sign;
  - AMG refused;
  - measured sizes;
  - fixed test gates.
- **Revision 4** (round 3: 2 design, 3 minor; the user's decisions of 2026-10-17):
  - **User decisions:**
    - The patch test is a Dirichlet lift (the roadmap's meaning of "lifting", as in PR 3), with
      the transverse-shear states gated only on parallelogram patches.
    - The cylinder gate stays 0.1%; if it fails, stop and ask.
  - **16c sizing:** Srinivas–Rao sub-plates solved as quarters; cylinder margin measured on
    all 4 classes; the build estimate linear in the ply count.
  - **Refusals and staging:** `amg()`/`pmg()` refuse `ssh8`; the 16a staging is reduced to
    what can be reached.
  - **Spec:**
    - angles in radians;
    - the ply container and the $\zeta$ mapping specified;
    - wording fixes in frame, flip and sweep.

## Split and plans

PR 16 is three PRs, each from `main`, merged in order: `pr-16a-element`, `pr-16b-operator`,
`pr-16c-accuracy`.

- **16a, the element:** `ssh8`, `solid_shell`, `Laminate`, the kernels, `assemble`, recovery,
  the VTU writer mapping, element-level tests.
  - Operator paths are reached only through `kernels.*`, which raise for `ssh8`, so
    `unstructured` and `solve_modes` refuse an `ssh8` mesh (one test).
  - `unstructured_memory` and `amg_memory` raise `ValueError("ssh8 operators arrive in PR
    16b")` (no `KeyError`); 16b replaces the first raise.
  - `stencil` cannot see `ssh8` (`voxel` is None).
- **16b, the operator and API:** `unstructured`, the estimator, sampling, `solve_modes`, the
  `.inp` reader, the M4 example.
- **16c, the accuracy suite:** a sizing study, then its own short plan (meshes, levels,
  benchmark layout; the gates are the roadmap's, narrowed only as logged), reviewed as usual. If
  a gate fails after the study, stop and ask the user.

## Data model

- **Element type `ssh8`:** hex8 topology (8 nodes, VTK order), added to `TYPES` (last) and
  `NODES`. The reference $\zeta$ axis (corners 0–3 → 4–7) is the thickness, and $+\zeta$ orders
  the plies bottom to top.
  - `box_mesh(type="ssh8")` raises.
  - The registry entry is hex8's (nodes, faces, 2×2×2 `points`/`weights`/`N`) with
    `lumping=None` (`lumping: str | None`), commented "quadrature for `check_elements`/`jacobian`
    only".
- **`solid_shell(mesh, elements=None, direction=None) -> (mesh, order)`** (`mesh/shell.py`, host
  NumPy):
  - Selection: an element set name, an index array, or all hex8 when None. A selection with
    non-hex8 elements raises.
  - The selected elements move into an `ssh8` block with their connectivity permuted, so
    $\zeta$ is the thickness. `material`, `orientation` and element sets follow.
  - `order` is new → old (`renumber`'s `q` convention); `voxel` becomes None.
  - Besides the `.inp` reader, it is the only producer of `ssh8`. A hand-built block is the
    user's responsibility.
- **Node permutations:** fixed proper rotations of the hex8 corners take a reference axis to
  $\zeta$. The 180° rotation about $\xi$ flips $\zeta$ (and $\eta$). Being proper rotations,
  they keep $\det J$ positive.
- **Thickness axis:**
  - `direction` given: a (3,) vector, or an (n_sel, 3) array in sorted selected-element order.
    The thickness axis is the reference axis whose mean edge vector (4 parallel edges) has the
    largest $|\cos|$ with it, and $\zeta$ points along $+$direction. If the two largest
    $|\cos|$ differ by $< 0.1$, `ValueError` names the element.
  - `direction=None` (sweep), with face keys as in `parts`:
    - Each chain of elements through one reference axis's opposite faces is walked once, and
      its element count is given to all members ($O(n_\text{el})$).
    - The thickness axis is the unique smallest of an element's three counts; a tie raises
      `ValueError("ambiguous thickness direction: pass direction=")`.
    - Raising: the $3^3$ cube, the strip $6\times1\times1$, and $8\times8\times12$.
    - Picked wrongly: $8\times10\times12$ (it picks $x$). Fewer in-plane than thickness
      elements needs `direction=`, and 16c always passes it.
    - Sign: $\zeta$ along the positive global axis most aligned with the element's mean
      thickness edge (`argmax`, first on ties). That is consistent on flat plates and may flip
      between elements of a curved shell.
  - **When the sign matters:** under a $\zeta$ flip the plies are reversed, the angles negated,
    and each ply turned 180° about its own material axis 1. So `direction` is required for
    laminates with off-axis plies (even $[\pm45]_s$), non-orthotropic plies, or unsymmetric
    stacking. `docs/mesh.md` states this.
- **Laminates in `materials`:** `materials={label: Material | Laminate}`.
  - `Laminate(plies)` holds a tuple of `Ply(material, angle, fraction)` NamedTuples:
    - `material`: a `Material`;
    - `angle`: radians about the shell normal from local axis 1, as `from_euler`;
    - `fraction`: thickness fraction, bottom to top.

    Both are pytrees, for PR 21.
  - Ply $k$ occupies $\zeta \in [-1 + 2\sum_{j<k}f_j,\ -1 + 2\sum_{j\le k}f_j]$ of the reference
    thickness. 2 Gauss points are mapped linearly into it, with weights scaled by $f_k$, so a
    fraction-0 ply has weight 0. Reference fractions equal physical ones when $\det J$ is
    constant in $\zeta$ (prismatic elements).
  - `check_materials` (host, before tracing as today) branches on `Laminate`: `check_stiffness`
    and $\rho$ on every ply, fractions $> 0$ summing to 1 to $10^{-12}$, angles finite.
  - `phase_table` gives a `Laminate` phase a finite placeholder (ply 0's rotated $C$ and $\rho$),
    read by no `ssh8` code.
  - `assemble` and `unstructured` raise after `phases` when a hex block uses a `Laminate` phase.
    `stencil` raises on any `Laminate` before `phase_table`.
  - A `Material` on `ssh8` is one ply (angle 0, fraction 1).
- **Shell frame (per element, at the centre; Abaqus-like):**
  - $t_3 = g_1\times g_2/\|g_1\times g_2\|$ at $\xi = 0$, with $J_{ij} = \partial
    x_i/\partial\xi_j$, so $g_j$ are columns.
  - $t_1$ = axis 1 of $R_o$ (the label's rotation, identity by default) projected onto $\perp t_3$
    and normalized, then $t_2 = t_3\times t_1$.
  - Fallback: if that projection has norm $< 0.1$ (within 5.7° of the normal), $t_2$ = axis 2
    projected and normalized, and $t_1 = t_2\times t_3$. It uses NaN-safe normalization (double
    `where`), and is not differentiable at the switch (documented).
  - $R_\text{shell} = [t_1\ t_2\ t_3]$, and ply $k$ uses $C_k' = \text{rotate}(C_k,
    R_\text{shell}R_z(\theta_k))$.
  - The frame does not depend on the node order. When $R_o$'s axis 3 is $+t_3$, it equals
    $R_o$; when $-t_3$, it is $R_o\,\mathrm{diag}(1,-1,-1)$, the same for orthotropic $C$ only.
    hex8 and `ssh8` parts with one label on a tilted plate are oriented differently (documented).
- **`ply_table(pairs, materials, orientations) -> (Ply(Material(C, rho), angle, fraction), R_o)`**
  (pure jnp, beside `phase_table`):
  - Shapes: `C (P, L, 6, 6)` unrotated, `rho`, `angle`, `fraction` `(P, L)`, `R_o (P, 3, 3)`,
    over all phases.
  - Padding copies ply 0 at fraction 0.
  - Built only when an `ssh8` block exists; indexed by the phase index.
  - The kernels take one element's slice (fields `(L, ...)`), batched by `vmap`.

### Use sites

Each gets an explicit `ssh8` branch.

- **`elements`:**
  - `kernels.stiffness`, `mass`, `lumped_nodal`, `stiffness_form`, `mass_form` raise for
    `ssh8`; the forms are deferred to PR 21.
  - `RECOVER = {"ssh8": shell_recover}` lives in `elements/__init__.py`, which is the registry
    PR 17 extends. `registry.py` stays NumPy-only (logged deviation from "in the element
    registry").
  - `check.py`: `mass="lumped"` raises; $\det J$ is unchanged.
- **`operators/assembled.py`:** `_KERNELS`, `_chunked` (`jax.tree.map` over the ply pytree) and
  `values()`. The `ssh8` branch calls `shell_stiffness` and `shell_mass` on `ply_table` slices;
  $m_e$ is expanded to $m_e\otimes I_3$ (24×24) before the ravel.
- **`operators/matfree.py`:**
  - `Block` gets a leaf `me` (packed 36, $\rho$ included; no `wdet` for `ssh8`), and the
    `Block(...)` construction and `unstructured`'s table calls follow.
  - `_build` takes the ply pytree for `ssh8` and returns `(Ke, me)`; `_block_data`'s leaf list
    gains `me`.
  - `_m_scale(op, b, d)` is `d["s_M"]` for `ssh8`. Its callers pass `b`: `_m_apply`, `_m_diag`,
    `element_matrices` and `sampling.element_factor` (`sampling.py:24`). `_wdet`'s callers
    branch.
  - Padding repeats element 0 (`layout`), so the padded `me` is element 0's (positive definite)
    and is zeroed by $s_M = 0$, like `Ke`.
- **`operators/sampling.py`:** for `ssh8` the factor is $F = \sqrt{s_M}\,\mathrm{chol}(m_e)$;
  the docstrings are fixed (`sampling.py:6`).
- **`precond/pmg.py` (`_check`, shared by `amg` and `pmg`):** any `ssh8` block raises
  `ValueError` naming PR 18 (`amg.py:188` would otherwise treat it as hex8).
- **`memory.py`:**
  - `per_el = 300 + 36` and `_chunk_temps` for `me`;
  - `BUILD_BYTES["stored"]["ssh8"]` (16b, below);
  - otf, `amg_memory` and `pmg` refuse.
- **`eigen/api.py`:**
  - `resolve_precond`: auto → `"jacobi"` on `ssh8`; `"amg"`, `"pmg"` raise naming PR 18;
    `"gmg"` already raises.
  - `levers`: no otf hint.
  - The `sigma_options` error says "auto resolved to jacobi on ssh8".
- **`mesh/io.py`:**
  - `write`'s names map `"ssh8"` to `"hexahedron"` (16a); `MESHIO` is unchanged.
  - The `.inp` regex adds `SC8R|SC8`, and other `SC*` raise (16b).
- **Exports:** the `__init__.py` of `mesh`, `materials`, `elements`.

## The element (16a)

Notation:
- $X(\xi)$ trilinear, $g_j = \partial X/\partial\xi_j$, $J = [g_1\ g_2\ g_3]$, $J_0$ at the
  centre.
- Covariant strains $E_{ij} = \tfrac12(g_i\cdot u_{,j} + g_j\cdot u_{,i})$, in `VOIGT` order with
  engineering shear.
- $E = J^\top\varepsilon J$, so $\varepsilon = TE$ with $T = $ `bond_strain`$(J^{-\top})$.

Formulation:
- **ANS, transverse shear** (Bathe–Dvorkin):
  $E_{13}^{ANS} = \tfrac12(1-\eta)E_{13}(0,-1,\zeta) + \tfrac12(1+\eta)E_{13}(0,1,\zeta)$, and
  $E_{23}$ likewise from $(\mp1, 0, \zeta)$. Both are linear in $\zeta$, so the tying points
  need no separate $\zeta$ sampling; they are evaluated at each point's $\zeta$.
- **ANS, thickness** (Betsch–Stein): $E_{33}$ bilinear from the 4 corner lines. It does not
  depend on $\zeta$ and is evaluated once at $\zeta = 0$.
- **EAS**, 5 parameters, covariant: $\tilde E_{11} = \xi\alpha_1$, $\tilde E_{22} = \eta\alpha_2$,
  $2\tilde E_{12} = \xi\alpha_3 + \eta\alpha_4$, $\tilde E_{33} = \zeta\alpha_5$. Then
  $\tilde\varepsilon = (\det J_0/\det J)\,T_0 M\alpha$ with $T_0 = $ `bond_strain`$(J_0^{-\top})$.
  $\int M\,d\xi = 0$.
- **Condensation:** $K_e = K_{uu} - K_{u\alpha}K_{\alpha\alpha}^{-1}K_{\alpha u}$ (5×5
  Cholesky), symmetrized as `_sym`; $\alpha = -K_{\alpha\alpha}^{-1}K_{\alpha u}u_e$.
- **Quadrature:**
  - 2×2 in-plane times 2 per ply.
  - `lax.scan` over plies accumulates $K_{uu}$, $K_{u\alpha}$, $K_{\alpha\alpha}$ and $m_e$.
  - Exact in $\zeta$ for prismatic elements.
- **Mass:** $M_e = m_e\otimes I_3$, $m_e = \sum_k\rho_k\int_{\text{ply }k}N^\top N\det J$
  (8×8). Lumping raises "solid-shell elements are consistent-mass only".
- **Locking:**
  - Covered: transverse shear, curvature thickness, Poisson thickness, in-plane shear and
    parasitic bending.
  - Not covered: membrane locking of curved elements, and trapezoidal in-plane bending.
  - `docs/elements.md` says so.
- **Kernels** (`elements/shell.py`, pure jnp):
  - `shell_stiffness(X, plies, R_o) -> Ke (24, 24)`.
  - `shell_mass(X, plies) -> m_e (8, 8)`; it reads `rho` and `fraction` only.
  - `shell_recover(X, plies, R_o, u) -> (points, weights, eps, sigma)`:
    - `points`: physical coordinates of the quadrature points;
    - `weights`: $w\det J$;
    - `eps`, `sigma`: global Cartesian Voigt, from the ANS strains plus the EAS strains with
      the recovered $\alpha$.

    PR 17 owns the postprocessor stress test, and its plan must include `ssh8` (logged).

### 16a tests

1. **Independent NumPy kernel** (`tests/shell_ref.py`: point loops, explicit $T$, ANS, EAS,
   dense condensation, the same frame fallback). Random distorted elements (corners perturbed
   $\le 0.2h$), random anisotropic $C$, 1–3 plies. $K_e$ and $m_e$ to $10^{-12}\max|K_e|$.
2. **Laminate path,** to $10^{-12}$:
   - (a) Invariant under the 4 in-plane connectivity rotations; under the $\zeta$ flip, plies
     reversed and angles negated, with orthotropic plies.
   - (b) A rigid rotation $Q$ of the element and $R_o$ gives $QK_eQ^\top$.
   - (c) One ply at $\theta$ equals one ply with `rotate(C, R_z(θ))` at 0.
   - (d) On a prismatic element, identical plies equal one ply.
   - (e) `Laminate((Ply(m, 0, 1),))` equals `Material m`. A flat element in the $xy$ plane
     with $\zeta$ along $+z$ and $R_o = R_z(\varphi)$ equals `shell_stiffness` with
     `rotate(C, R_o)` and $R_o = I$.
3. **Mass,** to $10^{-14}$:
   - The one-ply $m_e$ equals hex8's scalar block on any element.
   - On a prismatic element, an equal-density laminate equals it.
   - Per ply, $m_e$ is the sum of hex8-type integrals over the sub-intervals.
   - $\sum m_e = \sum_k\rho_kV_k$.
4. **Zero modes:**
   - Gated: at $h/t = 1, 2$, regular and distorted ($\le 0.2h$), 6 eigenvalues
     $\le 10^{-10}\lambda_\max$ and the 7th $\ge 10^{-3}\lambda_\max$ (prototype $\ge 1.4\cdot
     10^{-2}$).
   - Reported only: $h/t = 100$.
5. **Patch tests** (user decision, 2026-10-17). Patches have $t = 0.01$ and anisotropic $C$.
   Under a linear field $u = Ax$ with constant $\sigma_0$ (the strain solved for the target
   stress state):
   - **Lift (gated):** boundary nodes get $Ax$, the interior is solved by `spsolve` on
     `assemble`. The solution equals $Ax$, and `shell_recover` gives $\sigma_0$ to
     $10^{-10}\|\sigma_0\|$, on these patches and states:
     - (i) a flat parallelogram patch: all 6 states;
     - (ii) the flat distorted MacNeal–Harder patch (5 elements): the 4 states with
       $\sigma_{13} = \sigma_{23} = 0$ in the shell frame;
     - (iii) patch (ii) rigidly tilted into a general plane: the same 4 states.
   - **Recovery of the imposed field (gated):** for all 6 states on (i)–(iii) and on (iv) a
     patch with both surfaces lifted equally by $0.1t$ (isotropic $C$), to $10^{-10}$
     (prototype $\le 2\cdot10^{-14}$).
   - **Reported only:** the lift test's transverse-shear states on (ii)–(iv) and the warped
     patch. Bathe–Dvorkin ANS is not variationally consistent on non-parallelogram elements
     (nodal-force errors 0.2–0.86).
6. **MacNeal–Harder** (1985):
   - Setup: beam $6\times0.2\times0.1$, $E = 10^7$, $\nu = 0.3$, $6\times1\times1$ with
     `direction=`, root $u = 0$, unit tip loads by consistent traction.
   - References: $3.0\cdot10^{-5}$, 0.1081, 0.4321, 0.03208.
   - Gates: regular $\ge 0.95$ for extension, in-plane and out-of-plane (0.988, 0.972, 0.979);
     parallelogram $\ge 0.9$ for extension and out-of-plane (0.994, 0.976).
   - Reported only: parallelogram in-plane, trapezoid, twist.
7. **`assemble` on hex8 + `ssh8`:**
   - symmetric PSD, $\|KR\| \le 10^{-12}\|K\|\|R\|$, total mass $\sum\rho_kV_k$;
   - a one-ply `ssh8` block's $M$ equals hex8's;
   - `parts`, `rigid_modes` and `dof_mask` on it.
8. **`jax.grad`** of $u^\top K_e v$ with respect to $X$, $C$, angle and fraction, and of
   $u^\top m_e v$ with respect to $X$ and $\rho$, against central differences to $10^{-6}$. No NaN
   with padded plies or with $R_o$'s axis 1 or 2 exactly normal.
9. **`solid_shell`:**
   - The $8\times6\times2$ plate under the 24 connectivity rotations gives the same axis and
     $\zeta$ along $+z$; the raising cases raise; $8\times10\times12$ picks $x$ (documented
     limit); a tilted plate with `direction` works.
   - Output: $\det J > 0$; sets and labels follow `order`; `voxel` is None; non-hex8
     selections raise; VTU write then read, then `solid_shell(direction=)`, round-trips up to
     `order`.
10. **Refusals:**
    - lumped mass;
    - a `Laminate` on a hex block, and in `stencil`;
    - bad fractions or ply $C$;
    - `box_mesh(type="ssh8")`;
    - the forms;
    - `unstructured` on `ssh8`;
    - `unstructured_memory`, `amg_memory`;
    - `amg()`/`pmg()` on an `ssh8` operator (in 16b, when one can be built).

## The operator and API (16b)

- **Stored leaves:** the packed condensed $K_e$ (300) and `me` (36), 336 floats per element vs
  hex8's 308.
  - `apply_K`, `diag_K`, `block_diag_K` and `scaled` are unchanged.
  - `apply_M` is $s_M\cdot(c,8,8)\,@\,(c,8,3m)$, and `diag_M` is $s_M\,\mathrm{diag}(m_e)$ per DOF.
  - `element_matrices` gives $K_e$ and $m_e\otimes I_3$.
  - otf raises in the builder and the estimator.
- **Estimator:** `BUILD_BYTES["stored"]["ssh8"]` is $c\,(55.4\cdot10^3 + 0.30\cdot10^3 L)$ bytes
  for $L \ge 2$. It is linear in the ply count: the swapped per-ply $C$ input.
  - Measured prototype: 55.69, 57.52 and 58.73 kB per element at 2, 8 and 12 plies, and 47.9 at
    1 ply (XLA inlines a one-trip scan); hex8 is 29.95.
  - Test: 2, 8 and 12 plies within 5%, and 1 ply below.
  - Slow test: $820\times820\times1$ with a 3-ply block (672k elements, 4.04M DOF), both paths,
    $m = 1$ and 16. Estimates 2.23–5.79 GB, asserted $\ge 2$ GB, within 20% of the
    `harness.measure` peak of build + `apply_K` + `apply_M`. About 3–4 min.
- **Probe sampling:** the exact factor $\sqrt{s_M}\,\mathrm{chol}(m_e)$, so band counts are
  exact.
- **`solve_modes`:**
  - auto → Jacobi; AMG and p-MG refused until PR 18.
  - `band`, `verify`, `strategy` and `solve_modes_memory` are unchanged.
  - Jacobi on `ssh8` is test-scale only (prototype 8×8×1 steel, tol 1e-6): 57 iterations at
    $L/t = 10$, 233 at 20, 1640 at 50, stagnation at 100.
  - `docs/eigensolvers.md` gives these numbers, $\lambda_\max/\lambda_1 \approx c(L/t)^4$, and
    the M4 slicing and count times below.
- **Reader:** `.inp` `SC8R`/`SC8` → `ssh8` in the file's node order (nodes 1–4 bottom, 5–8 top,
  which is VTK $\zeta$). $\det J > 0$ is checked by `check_elements` when the operator is built,
  as for hex.
- **M4 example** (`examples/m4_shell.py`): an SS square steel plate, $8\times8\times1$ `ssh8`,
  $L/t = 20$, 6 modes, Jacobi with `maxit=2000` (8.6× margin), against `reference_modes` to the
  PR 9 tolerances. The SS dict is inline; a test asserts it equals `SIMPLY_SUPPORTED`.

### 16b tests

1. `apply_K`, `apply_M`, `diag_K`, `diag_M` and `element_matrices` against `assemble` to
   $10^{-12}$:
   - meshes: `ssh8`, and hex8 + `ssh8` sharing label 0;
   - laminates;
   - $m = 1$ and $>1$;
   - random `s_K`/`s_M`;
   - scatter and gather.
2. `jax.grad` of $u\cdot$`apply_K`$(v)$ through the stored build, with respect to `points` and
   a ply angle, against central differences to $10^{-6}$.
3. M4 against `reference_modes`:
   - `band=(0, f)`, and `band=(f_1, f_2)` with `strategy="shift_invert"` (Jacobi MINRES; time
     recorded);
   - `verify=True` (time recorded);
   - `solve_modes_memory` smoke;
   - auto → Jacobi on hex8 + `ssh8`.
4. Refusals:
   - otf;
   - `"amg"`/`"pmg"`, through `solve_modes`, `solve_modes_memory`, and `amg()`/`pmg()`
     directly;
   - `"gmg"`;
   - no otf lever.
5. The build envelope; the $\ge 2$ GB estimator test (slow).
6. $FF^\top = m_e$ to $10^{-12}$ (laminates, padded); a masked $SS^\top$ on hex8 + `ssh8` equals
   the SciPy $M$.
7. An SC8R `.inp` (one element, $h/t\approx1$) reads as `ssh8` with the file's thickness axis;
   `SC6R` raises.

## The accuracy suite (16c: design frame)

The 16c plan fixes meshes, levels and the benchmark layout after the sizing study.

- **Common rules:**
  - `reference_modes` on `assemble`; the error is relative in $f$; no upper-bound checks.
  - Gates compare against the finest reference level. The observed-order Richardson estimate
    (recomputed by the tests from the recorded levels) bounds the reference error, including
    the $n_z$ check. The test fails if $p \notin [1, 5]$ or the levels are not monotone.
  - Every gated run asserts $2f_\lambda \le 0.1\times$ its gate, with
    $f_\lambda = 10\varepsilon\hat\lambda_\max/\lambda$ computed in the test.
  - Equivalence tests compare to $\max(10^{-10}, 2f_\lambda)$.
- **Benchmarks:**
  - Each run is $\le 1$ h with per-level progress; the results are recorded with their commit.
  - The slow tests recompute the smallest-margin gated case per test, plus the coarsest
    reference level, against the record to $\max(10^{-9}, 2f_\lambda)$. Budget: a few minutes.
  - `docs/eigensolvers.md` records the eigsh times and memory on these thin meshes (up to about
    280k DOF).
- **Srinivas–Rao** (0.1%):
  - Plates: `ortho` and `pag`, at $L/t = 10$ and 100. `plate_fe` is generalized to `H`, `a`,
    `b`: hex8, then `solid_shell(direction=(0,0,1))`. The `pag` 90° ply is set by a ply angle,
    not a pre-rotated $C$.
  - Each of the lowest 10 modes is gated on its half-wave sub-plate $L/m\times L/n$ ($L$ for
    $m = 0$ or $n = 0$), solved as a quarter:
    - SS on xmin/ymin and mirror BCs on the cut faces (xmax `"x"`, ymax `"y"`);
    - every mapped label is even about both sub-plate midplanes;
    - matching is by the sorted index of the mapped label among the doubly-even labels of
      `plate_modes` on the quarter.
  - Two equivalence tests:
    - the full plate at $2N\times N$ gives mode (2,1) equal to the $N\times N$ sub-plate's;
    - the quarter equals the sub-plate.
  - Measured prototype (`pag`, quarter): $n = 64$ full-plate-equivalent and $k = 8$–12 per ply
    give errors $4\cdot10^{-4}$–$3\cdot10^{-4}$ (margin 2.5–3.4), $\le 115$k DOF, $\le 80$ s,
    9.3 GB. About 19 sub-plates take about 9–25 min. $L/t = 100$: $n = 48$, $k = 1$, 2 s.
  - `docs/verification.md` states $k$ per $L/t$.
  - Reported only: one layered element per laminate (roadmap), and the full plate at a stated
    mesh.
- **Clamped cylinder:**
  - Geometry: $R/t = 50$, $L/R = 2$, isotropic, $u = 0$ on both end faces.
  - Model: 1/8 symmetry, with 4 classes on one mesh by `dof_mask`:
    - circumferential S about $x = 0$ ($u_x = 0$) or A ($u_y = u_z = 0$);
    - $y = 0$ symmetric ($u_y = 0$);
    - axial S at mid-length ($u_z = 0$) or A ($u_x = u_y = 0$).

    Excluded: the $y = 0$-antisymmetric doublet partners and the $n = 0$ torsion modes (logged).
  - Matching: sorted per class, the lowest 6.
  - Measured prototype:
    - hex27 levels 24/48/96 at $n_z = 2$: reference error $\le 0.008\%$, $n_z$ term
      $\le 0.006\%$; about 9 min for 4 classes.
    - `ssh8` $384\times96\times1$ (220k DOF, 32 s per class): worst about 0.075–0.09% over all
      4 classes. Margin 1.1–1.25, against a one-layer floor of about 0.06%.
  - Gate: 0.1% (user, 2026-10-17). If it fails, stop and ask with the numbers.
- **No locking:**
  - Setup: the SS square isotropic plate, $6\times6\times1$, at $L/t = 10$ and 1000. 16 modes
    are computed; assert $\ge 6$ bending modes by z-parity, and keep the first 6.
  - Error, in $\lambda$: max over the 6 of $|\lambda/\lambda_{48} - 1|$, with $\lambda_{48}$ a
    $48\times48\times1$ run.
  - Its Richardson estimate (levels 12/24/48, $p$ observed, measured 2.0) plus the floor must
    be $\le 0.1\times$ the tested error at both $L/t$.
  - Floor: $10\varepsilon\lambda_\max/\lambda$, computed in the test (measured $9.2\cdot10^{-4}$
    against errors 0.066–0.85).
  - Gate: error at 1000 $\le 1.5\times$ error at 10 (measured 1.31 in $\lambda$).
  - Reported only: the hex8 baseline.
- **Skew plate:**
  - Setup: parallelograms only, 30°, clamped, $L/t = 20$, lowest 6 sorted.
  - hex27 levels 6/12/24/48 at $n_z = 2$ (135k DOF, estimate about 0.05%).
  - `ssh8` about $128\times128\times1$ (projected 0.7% against the 1% gate).

## Docs (each with its part)

- **16a:**
  - `docs/elements.md`: `ssh8`, formulation, quadrature, frames, locking, patch-test limits,
    recovery, refusals.
  - `docs/mesh.md`: `solid_shell`, permutations, thickness and sign rules, caveats, VTU.
  - `docs/materials.md`: `Laminate`/`Ply`, radians, the $\zeta$ mapping, the shell frame and
    fallback, the stacking sign, hex vs `ssh8` orientation.
- **16b:**
  - `docs/operators.md`: leaves, estimator.
  - `docs/eigensolvers.md` and `docs/preconditioners.md`: Jacobi auto, AMG/p-MG until PR 18,
    the Jacobi numbers.
  - README example.
  - `docs/mesh.md`: SC8R.
- **16c:** `docs/verification.md`.
- **Each part:** ROADMAP status, `docs/open_items.md`.

## Decisions (logged in open_items)

User decisions (2026-10-17):

- The patch test is a Dirichlet lift. Transverse-shear states are gated on parallelogram
  patches only; on distorted, tilted and lifted patches they are reported. Recovery of the
  imposed field is gated for all 6 states.
- The cylinder gate stays 0.1%; if it fails, stop and ask.

Without user input:

- **Process:** the three-way split; 16a staging through the kernel raises plus the two
  estimator raises; 16c's own plan after the sizing study.
- **Type and producer:** type `ssh8` (last in `TYPES`); `solid_shell` returning `order`;
  `box_mesh(type="ssh8")` refused; hand-built blocks are the user's responsibility.
- **Thickness rules:** the per-element `direction` array; cosine margin 0.1; unique smallest
  chain count; the per-element sign rule; proper permutations; no edge-length check;
  `direction` required for off-axis, non-orthotropic or unsymmetric laminates.
- **Laminates and frame:**
  - `Laminate`/`Ply` in `materials`, angles in radians;
  - 2 points per ply, fixed (narrows the roadmap's "n per ply");
  - the reference-$\zeta$ ply mapping;
  - the placeholder phase entry;
  - the frame from $R_o$ axis 1, with the axis-2 fallback at 0.1.
- **Mass:** $m_e$ stored (36 floats, $\rho$ included), $s_M$-only mass scale on `ssh8`, padded
  `me` as element 0's.
- **Code structure:** dispatch by name with `kernels.*` raising; `RECOVER` in
  `elements/__init__.py`; the forms deferred to PR 21.
- **Preconditioners:** Jacobi auto on `ssh8`; AMG and p-MG refused until PR 18.
- **Element tests:**
  - zero modes gated at $h/t \le 2$ with the 7th $\ge 10^{-3}$ (stricter than the roadmap's
    $10^{-8}$, intentionally), $h/t = 100$ reported;
  - MacNeal–Harder loading and `direction`.
- **16b scope:**
  - the build estimate linear in $L$;
  - Jacobi tests at $L/t \le 20$;
  - the M4 example at $L/t = 20$;
  - SC8R stacking kept, other `SC*` refused;
  - the VTU type field deferred.
- **16c:**
  - Srinivas–Rao per sub-plate, solved as quarters;
  - the cylinder classes and exclusions;
  - observed-order Richardson with the $n_z$ term;
  - the no-locking reference and floor;
  - recorded benchmark results with recomputation of the smallest-margin case.
- **PR 17:** must cover `ssh8` recovery.
