# PR 16 plan: solid-shell element (KGW 1999)

Branch `pr-16-solid-shell` (this plan), from `main` after PR 15d. Roadmap: PR 16 (M4), the
"Shells" and "Mass" key design decisions, PR 8b's `SIMPLY_SUPPORTED`, and the dependency row
16 ← 3, 5, 8, 9.

## Revisions

- **Revision 1:** draft.
- **Revision 2** (round 1, 5 reviewers, all asked for design changes):
  - laminates in `materials`;
  - the ply frame from the orientation label, not the node order;
  - a stored 8×8 scalar $m_e$;
  - a ply `scan`;
  - fixed proper permutations;
  - no edge-length check;
  - a passable patch test;
  - Srinivas–Rao per half-wave sub-plate;
  - the cylinder on a symmetry model;
  - a concrete locking floor;
  - Jacobi tests at $L/t \le 20$;
  - SC8R keeps its stacking.
- **Revision 3** (round 2: 2 asked for design changes, 3 for minor fixes). Fixes and changes:
  - **Transform:** the strain transform is `bond_strain`$(J^{-\top})$.
  - **Mass:** on `ssh8` the mass scale is $s_M$ alone (by block type, no phase-$\rho$ trick).
  - **Laminate branches:** `phase_table` and `check_materials` get them.
  - **Kernel calls:** `assemble`/`_build` call the shell kernels directly.
  - **ζ sign:** a per-element sign rule, no propagation; the docs cover off-axis plies.
  - **Explicit AMG:** refused until PR 18.
  - **Test fixes:**
    - zero-mode, ζ-flip, prismatic identities;
    - patch recovery for all 6 states;
    - the frame-switch NaN;
    - the floor in consistent units;
    - cylinder classes per BC;
    - the build envelope;
    - the 2 GB size.
  - **Staging:** 16a refuses `ssh8` in the operator paths until 16b.
  - **16c:** gets its own short plan after the sizing study, with measured sizes recorded here.

## Split and plans

PR 16 is three PRs, each from `main`, merged in order: `pr-16a-element`, `pr-16b-operator`,
`pr-16c-accuracy`.

- **16a, the element:** `ssh8`, `solid_shell`, `Laminate`, the kernels, `assemble`, recovery,
  element-level tests. `unstructured`, `unstructured_memory`, `stencil` and `solve_modes`
  raise `NotImplementedError("ssh8 operators arrive in PR 16b")` on a mesh with an `ssh8` block
  (tested); 16b removes the raise.
- **16b, the operator and API:** `unstructured`, the estimator, sampling, `solve_modes`, the
  reader, the M4 example.
- **16c, the accuracy suite:** a sizing study, then its own short plan (meshes and levels; the
  gates are the roadmap's), reviewed as usual. If a gate is out of reach after the study, stop
  and ask the user.

This document is the plan for 16a and 16b and the design frame for 16c.

## Data model

- **Element type `ssh8`:** hex8 topology (8 nodes, VTK order), added to `TYPES` (last, so its
  block follows hex blocks) and `NODES`. The reference $\zeta$ axis (corners 0–3 → 4–7) is the
  thickness, and $+\zeta$ orders the plies bottom to top.
  - `box_mesh(type="ssh8")` raises (use `solid_shell`).
  - The registry entry is hex8's (nodes, faces, 2×2×2 `points`/`weights`/`N`) with
    `lumping=None` (`lumping: str | None`). A comment says its quadrature is for
    `check_elements`/`jacobian` only; stiffness and mass never use it.
- **`solid_shell(mesh, elements=None, direction=None) -> (mesh, order)`** (`mesh/shell.py`, host
  NumPy):
  - Selection: an element set name, an index array, or all hex8 when None. A selection with
    non-hex8 elements raises.
  - Moves the selected elements into an `ssh8` block and permutes their connectivity so
    $\zeta$ is the thickness. `material`, `orientation` and element sets follow.
  - `order` is new → old (as `renumber`'s `q`), for the user's per-element arrays.
  - `voxel` becomes None.
  - Besides the `.inp` reader, it is the only producer of `ssh8`. A hand-built block is the
    user's responsibility (documented).
- **Node permutations:** fixed proper rotations of the hex8 corners take a reference axis to
  $\zeta$ (three tables). The 180° rotation about $\xi$ flips the sign of $\zeta$ (and $\eta$).
  All are proper, so $\det J$ keeps its sign; faces are never swapped alone.
- **Thickness axis:**
  - `direction` given: a (3,) vector, or an (n_sel, 3) array in sorted selected-element order.
    The thickness axis is the reference axis whose mean edge vector (over its 4 parallel edges)
    has the largest $|\cos|$ with the direction, and $\zeta$ points along $+$direction. If the
    two largest $|\cos|$ differ by $< 0.1$, `ValueError` names the element.
  - `direction=None` (sweep): face keys as in `parts`. Each chain of elements through one
    reference axis's opposite faces is walked once and its length given to all members
    ($O(n_\text{el})$). The thickness axis is the unique shortest of an element's three chain
    lengths; a tie raises `ValueError("ambiguous thickness direction: pass direction=")`.
    - Raising: the $3^3$ cube and the MacNeal–Harder strip $6\times1\times1$.
    - Picked wrongly: an $8\times8\times12$ stack.
    - All of these need `direction=`, and 16c always passes it.
    - Sign: per element, $\zeta$ points along the positive global axis most aligned with its
      mean thickness edge. That is consistent on flat plates, but it may flip between elements
      of a curved shell.
  - **When the sign matters:** a $\zeta$ flip maps ply order to reversed and angles to negated,
    and rotates each ply by 180° about $t_1$. So `direction` is required for laminates with:
    - off-axis plies (even a symmetric $[\pm45]_s$);
    - non-orthotropic plies;
    - unsymmetric stacking.

    `docs/mesh.md` says so.
- **Laminates in `materials`:** `materials={label: Material | Laminate}`.
  - `Laminate(plies)` holds a tuple of `Ply(material, angle, fraction)` NamedTuples:
    - `material`: a `Material`;
    - `angle`: degrees about the shell normal from local axis 1;
    - `fraction`: thickness fraction, bottom to top.

    Both are pytrees, so they stay traceable for PR 21.
  - `check_materials` (host, called before tracing like today) branches on `Laminate`. It runs
    `check_stiffness` and the $\rho$ check on every ply, and checks fractions $> 0$ summing to 1
    to $10^{-12}$ and finite angles.
  - `phase_table` gives a `Laminate` phase a finite placeholder (ply 0's rotated $C$ and $\rho$);
    only hex blocks read it.
  - `assemble` and `unstructured` raise after `phases` when a hex block uses a `Laminate`
    phase. `stencil` raises on any `Laminate` before `phase_table`.
  - A `Material` on `ssh8` is one ply: angle 0, fraction 1.
  - Quadrature through the thickness: 2 Gauss points per ply (fixed).
- **Shell frame (per element, at the centre; Abaqus-like):**
  - $t_3 = g_1\times g_2/\|g_1\times g_2\|$ at $\xi = 0$. The code's $J_{ij} = \partial
    x_i/\partial\xi_j$, so $g_j$ are columns.
  - $t_1$: axis 1 of the label's rotation $R_o$ (identity by default), projected onto
    $\perp t_3$ and normalized. If that projection is $< 0.1$ (within 5.7° of the normal), axis
    2 is used instead. The switch uses a NaN-safe normalize (double `where`); it is not
    differentiable at the switch, as documented.
  - $t_2 = t_3\times t_1$, $R_\text{shell} = [t_1\ t_2\ t_3]$.
  - Ply $k$: $C_k' = \text{rotate}(C_k, R_\text{shell}R_z(\theta_k))$.
  - The frame does not depend on the node order. When $R_o$'s axis 3 is $+t_3$, it equals
    $R_o$, so a one-ply `Material` acts as on hex8. When it is $-t_3$, the two agree for
    orthotropic $C$ only.
  - On a tilted plate, hex8 and `ssh8` parts with the same label are oriented differently;
    `docs/materials.md` says so.
- **`ply_table(pairs, materials, orientations)`** (pure jnp, beside `phase_table`):
  - Returns `Ply` arrays: `C (P, L, 6, 6)` unrotated, `rho (P, L)`, `frac (P, L)`,
    `angle (P, L)`, and `R_o (P, 3, 3)`, over all phases.
  - Padding copies ply 0 at fraction 0, so it stays finite under `grad`.
  - It is built only when an `ssh8` block exists and indexed by the same phase index.

### Use sites

Each needs an explicit `ssh8` branch; nothing applies a hex table by name.

- **`elements/kernels.py`:**
  - `stiffness`, `mass`, `lumped_nodal`, `stiffness_form` and `mass_form` raise for `ssh8`.
    The shell kernels take ply arrays, and the forms are deferred to PR 21.
  - `RECOVER = {"ssh8": shell_recover}` lives in `elements/__init__.py`; PR 17 adds the hex
    entries.
- **`elements/check.py`:** $\det J > 0$ at the corners and the 2×2×2 points is unchanged;
  `mass="lumped"` raises explicitly.
- **`operators/assembled.py`:** `_KERNELS`; `_chunked` (`jax.tree.map` over the ply pytree for
  padding); `values()` (an `ssh8` branch calls `shell_stiffness`/`shell_mass` on `ply_table`
  slices; the mass uses $m_e$, not `rho`).
- **`operators/matfree.py`:**
  - `Block` gets a leaf `me` (packed 36, $\rho$ included, no `wdet` for `ssh8`); the `Block(...)`
    construction and the `phase_table`/`ply_table` calls in `unstructured` follow.
  - `_build(t, ...)` takes the ply pytree for `ssh8` and returns `(Ke, me)`; `_block_data`'s
    leaf list gains `me`.
  - `_m_scale(op, b, d)` returns `d["s_M"]` for `ssh8`. It is passed `b`, as are its callers
    `_m_apply`, `_m_diag` and `element_matrices`.
  - `_wdet`'s callers branch for `ssh8`.
- **`operators/sampling.py`:**
  - `element_factor` for `ssh8` is `chol(where(pad, I, m_e))` masked to 0 on padded elements,
    and the module docstring is updated.
  - `xi_shapes` is unchanged (width 8 already).
- **`memory.py`:**
  - `per_el = 300 + 36` for `ssh8`, and `_chunk_temps` for `me`.
  - `BUILD_BYTES["stored"]["ssh8"]` (below); `OTF_BYTES` has no `ssh8`, and otf raises.
  - `amg_memory` refuses `ssh8`, like the builder.
- **`eigen/api.py`:**
  - `resolve_precond`: auto → `"jacobi"` on `ssh8`; `"amg"` raises `ValueError` naming PR 18;
    `"gmg"` already raises (non-voxel).
  - `levers`: no otf hint on `ssh8`.
- **`mesh/io.py`:**
  - The `.inp` regex adds `SC8R|SC8`; other `SC*` raise like unknown `C3D*`.
  - `write`'s names map `"ssh8"` to `"hexahedron"`; `MESHIO` is unchanged.
- **Exports:** the `__init__.py` of `mesh`, `materials` and `elements`.

## The element (16a)

Notation:
- $X(\xi)$ trilinear, $g_j = \partial X/\partial\xi_j$, $J = [g_1\ g_2\ g_3]$, $J_0$ at the
  centre.
- Covariant strains $E_{ij} = \tfrac12(g_i\cdot u_{,j} + g_j\cdot u_{,i})$, in `VOIGT` order with
  engineering shear $(E_{11}, E_{22}, E_{33}, 2E_{23}, 2E_{13}, 2E_{12})$.
- $E = J^\top\varepsilon J$, so $\varepsilon = J^{-\top}EJ^{-1}$, i.e. $\varepsilon = T E$ with
  $T = $ `bond_strain`$(J^{-\top})$ (`bond_strain(R)` maps $\varepsilon \mapsto R\varepsilon R^\top$).

Formulation:
- **ANS, transverse shear** (Bathe–Dvorkin):
  $E_{13}^{ANS} = \tfrac12(1-\eta)E_{13}(0,-1,\zeta) + \tfrac12(1+\eta)E_{13}(0,1,\zeta)$, and
  $E_{23}$ likewise from $(\mp1, 0, \zeta)$ in $\xi$. Both are linear in $\zeta$, so evaluating
  at the point's $\zeta$ is exact.
- **ANS, thickness** (Betsch–Stein): $E_{33}$ bilinear from the 4 corner lines. For trilinear
  $X$ and $u$ it does not depend on $\zeta$, so it is evaluated once at $\zeta = 0$.
- **EAS**, 5 parameters, covariant:
  - $\tilde E_{11} = \xi\alpha_1$, $\tilde E_{22} = \eta\alpha_2$,
    $2\tilde E_{12} = \xi\alpha_3 + \eta\alpha_4$, $\tilde E_{33} = \zeta\alpha_5$.
  - $\tilde\varepsilon = (\det J_0/\det J)\,T_0 M\alpha$ with $T_0 = $
    `bond_strain`$(J_0^{-\top})$.
  - $\int M\,d\xi = 0$, so the EAS part keeps the patch test on any geometry.
- **Condensation:** $K_e = K_{uu} - K_{u\alpha}K_{\alpha\alpha}^{-1}K_{\alpha u}$ (5×5
  Cholesky), symmetrized as `_sym`. Recovery: $\alpha = -K_{\alpha\alpha}^{-1}K_{\alpha u}u_e$.
- **Quadrature:**
  - 2×2 in-plane Gauss times 2 per ply.
  - `lax.scan` over plies accumulates $K_{uu}$, $K_{u\alpha}$, $K_{\alpha\alpha}$ and $m_e$, so
    the build temporaries are those of one ply.
  - Exact for prismatic elements (top face = bottom face translated); approximate, like hex8,
    when $\det J$ varies in $\zeta$.
- **Mass:** consistent, $M_e = m_e\otimes I_3$ with
  $m_e = \sum_k\rho_k\int_{\text{ply }k}N^\top N\det J$ (8×8). Lumping raises "solid-shell
  elements are consistent-mass only".
- **Locking:**
  - Covered: transverse shear (ANS), curvature thickness (corner ANS), Poisson thickness
    ($\alpha_5$), in-plane shear and parasitic bending ($\alpha_{1..4}$).
  - Not covered:
    - membrane locking of curved elements (no membrane ANS in KGW);
    - trapezoidal in-plane bending.

    `docs/elements.md` says so.
- **Kernels** (`elements/shell.py`, pure jnp, batched):
  - `shell_stiffness(X, plies, R_o) -> Ke (24, 24)`.
  - `shell_mass(X, plies) -> m_e (8, 8)`.
  - `shell_recover(X, plies, R_o, u) -> (points, weights, eps, sigma)`: Cartesian strain and
    stress at every quadrature point, from the ANS strains plus the EAS strains with the
    recovered $\alpha$.
  - PR 17 owns the postprocessor stress test (the roadmap's "whichever merges last"); its plan
    must include `ssh8` (logged).

### 16a tests

1. **Independent NumPy kernel** (`tests/shell_ref.py`: point loops, explicit $T$, ANS
   interpolation, EAS, dense condensation). Random distorted elements, random anisotropic $C$,
   1–3 plies. $K_e$ and $m_e$ agree to $10^{-12}\max|K_e|$ (norm-relative).
2. **Laminate path,** to $10^{-12}$:
   - (a) $K_e$ is invariant under the 4 in-plane connectivity rotations. Under the $\zeta$
     flip, the ply order is reversed and the angles negated, with orthotropic plies.
   - (b) A rigid rotation $Q$ of the element and $R_o$ gives $QK_eQ^\top$.
   - (c) One ply at $\theta$ equals one ply with `rotate(C, R_z(θ))` at 0.
   - (d) On a prismatic element, identical plies equal one ply.
   - (e) `Laminate((Ply(m, 0, 1),))` equals `Material m`. A flat element with $R_o$'s axis 3
     along $t_3$ equals `shell_stiffness` with `rotate(C, R_o)` and $R_o = I$.
3. **Mass,** to $10^{-14}$:
   - The one-ply $m_e$ equals hex8's consistent scalar block on any element.
   - On a prismatic element, an equal-density laminate equals it too.
   - The per-ply $m_e$ equals the sum of hex8-type integrals over the ply sub-intervals.
   - $\sum m_e = \rho V$.
4. **Zero modes** ($h$ in-plane size, $t$ thickness):
   - Gated: at $h/t = 1$ and 2, regular and distorted, 6 eigenvalues $\le 10^{-10}\lambda_\max$
     and the 7th $\ge 10^{-3}\lambda_\max$ (prototype: $5\cdot10^{-2}$ and $1.1\cdot10^{-2}$).
   - Reported only: $h/t = 100$, where the ratio is $\approx 10^{-9}$, scaling as $(t/h)^4$.
5. **Patch tests:**
   - Recovery (all cases): under a linear $u = Ax$ with constant $\sigma_0$, `shell_recover`
     gives $\sigma_0$ at every point to $10^{-9}\|\sigma_0\|$, for all 6 states. Exact on
     prismatic patches: ANS reproduces the linear covariant strains and $\alpha = 0$.
   - Forces: the free-node forces equal the consistent face tractions, i.e. hex8's $Ku$ on the
     same mesh (exact for trilinear geometry), to $10^{-10}\max|f|$.
   - Cases:
     - (i) A parallelogram patch: recovery and forces, all 6 states.
     - (ii) The flat distorted MacNeal–Harder patch (5 elements): recovery for all 6 states;
       forces for the states with $\sigma_{13} = \sigma_{23} = 0$ in the shell frame (for
       anisotropic $C$ the strain is solved for).
     - (iii) The same patch rigidly tilted into a general plane: recovery for all 6, forces for
       the membrane states.
     - (iv) Lifted, with both surfaces lifted equally by $0.1t$, isotropic $C$: recovery for
       all 6.
   - Reported only: the forces of (iv) and of the warped patch. Bathe–Dvorkin is not
     variationally consistent there (prototype: 3e-2 to 6e-1).
   - Logged reading: the roadmap's "flat patch test with lifting" is (iii) plus (iv).
6. **MacNeal–Harder** (1985):
   - Setup: beam $6\times0.2\times0.1$, $E = 10^7$, $\nu = 0.3$, $6\times1\times1$ with
     `direction=`, root face $u = 0$, unit tip loads by consistent traction.
   - References: extension $3.0\cdot10^{-5}$, in-plane 0.1081, out-of-plane 0.4321, twist
     0.03208.
   - Gates: regular mesh $\ge 0.95$ for extension, in-plane and out-of-plane (prototype 0.988,
     0.972, 0.979); parallelogram mesh $\ge 0.9$ for extension and out-of-plane (0.994, 0.976).
   - Reported only: parallelogram in-plane (0.62), trapezoid (in-plane 0.047), twist.
   - Solved with `spsolve` on `assemble`.
7. **`assemble` on hex8 + `ssh8`:** symmetric PSD, $\|KR\| \le 10^{-12}\|K\|\|R\|$ for the 6
   rigid modes, total mass $\rho V$. A one-ply `ssh8` block's $M$ equals the hex8 $M$ on the
   same elements.
8. **`jax.grad`** of $u^\top K_e v$ with respect to $X$, $C$, $\theta$ and frac, and of
   $u^\top m_e v$ with respect to $X$ and $\rho$, against central differences to $10^{-6}$.
   No NaN with padded plies, nor when $R_o$'s axis 1 or axis 2 is exactly the normal.
9. **`solid_shell`:**
   - The stacked plate $8\times6\times2$ under each of the 24 connectivity rotations gives the
     same thickness axis, and $\zeta$ along $+z$.
   - The cube and the strip raise; a tilted plate with `direction` works.
   - $\det J > 0$; sets and labels follow `order`; `voxel` is None; non-hex8 selections raise.
10. **Refusals:**
    - lumped mass;
    - a `Laminate` on a hex block, and in `stencil`;
    - fractions not summing to 1, and a bad ply $C$;
    - `box_mesh(type="ssh8")`;
    - the forms;
    - the 16a `NotImplementedError` in `unstructured`, `unstructured_memory`, `stencil` and
      `solve_modes`.

## The operator and API (16b)

- **Stored leaves:**
  - An `ssh8` block in `unstructured(..., mode="stored")` stores the packed condensed $K_e$
    (300 floats) and `me` (36 floats; 336 per element vs hex8's 308).
  - `apply_K`, `diag_K`, `block_diag_K` and `scaled` are unchanged. `apply_M` is
    $s_M\cdot(c,8,8)\,@\,(c,8,3m)$; `diag_M` is $s_M\cdot\mathrm{diag}(m_e)$ per DOF;
    `element_matrices` gives $K_e$ and $m_e\otimes I_3$.
  - Padded elements have $m_e = 0$.
  - `mode="otf"` raises, in the builder and the estimator alike.
- **Estimator:**
  - `BUILD_BYTES["stored"]["ssh8"]` is the envelope of the `memory_analysis` build temporaries
    at 3–8 plies. Measured prototype: 47.9 kB per element at 1 ply (XLA inlines a one-trip
    scan), 56.0 at 3, 57.5 at 8; hex8 is 29.95.
  - Test: 2–8 plies within 5% of the constant, and 1 ply $\le$ it.
  - Slow test: $820\times820\times1$ with a 3-ply block (672k elements; scatter $m = 1$
    estimate $\approx 2.24$ GB), both paths, $m = 1$ and 16. The estimate is within 20% of the
    `harness.measure` peak of build + `apply_K` + `apply_M`. Build time is about 13 s.
- **Probe sampling:** the exact factor is the masked $\mathrm{chol}(m_e)$, so band counts are
  exact as for hex.
- **`solve_modes`:**
  - auto → Jacobi on `ssh8`; `"amg"` raises until PR 18 (logged).
  - `band`, `verify`, `strategy` and `solve_modes_memory` are unchanged.
  - Jacobi on `ssh8` is test-scale only. Measured prototype (8×8×1 steel plate, tol 1e-6):
    57 iterations at $L/t = 10$, 233 at 20, 1640 at 50, stagnation at 100 (32×32×1 at 20: 276).
  - `docs/eigensolvers.md` gives these numbers, $\lambda_\max/\lambda_1 \approx c(L/t)^4$, and
    that shift-invert slicing with Jacobi MINRES is test-scale too.
- **Reader:** `.inp` `SC8R`/`SC8` → `ssh8` in the file's node order (nodes 1–4 bottom, 5–8 top,
  which is VTK $\zeta$), checked by $\det J > 0$.
- **Writer:** VTU writes `ssh8` as hexahedra. `read` gives hex8; `solid_shell(direction=...)`
  restores the type up to `order` (documented). The type field is deferred (open_items).
- **M4 example** (`examples/m4_shell.py`): an SS square steel plate, $8\times8\times1$ `ssh8`,
  $L/t = 20$, 6 modes, Jacobi with `maxit=2000` (8.6× margin), against `reference_modes` to the
  PR 9 tolerances. The SS dict is inline; a test asserts it equals `SIMPLY_SUPPORTED`.

### 16b tests

1. `apply_K`, `apply_M`, `diag_K`, `diag_M` and `element_matrices` against `assemble` to
   $10^{-12}$:
   - meshes: `ssh8`, and hex8 + `ssh8` sharing material label 0;
   - laminates;
   - $m = 1$ and $>1$;
   - random `s_K`/`s_M`;
   - scatter and gather.
2. The M4 example against `reference_modes`:
   - also `band=(0, f)` (154 iterations measured) and `verify=True`;
   - `solve_modes_memory` smoke;
   - auto resolves to Jacobi on hex8 + `ssh8`.
3. Refusals:
   - otf, in the builder and the estimator;
   - `"amg"`, through `solve_modes` and `solve_modes_memory`;
   - `"gmg"`;
   - `levers` gives no otf hint on `ssh8`.
4. The build envelope; the $\ge 2$ GB estimator test (slow).
5. $FF^\top = m_e$ to $10^{-12}$ (laminates, padded). A masked $SS^\top$ on hex8 + `ssh8` equals
   the SciPy $M$.
6. Readers:
   - An SC8R `.inp`, one element at $h/t\approx1$, reads as `ssh8` with the file's thickness
     axis; an `SC6R` raises.
   - VTU write, then read, then `solid_shell(direction=)` gives the mesh back up to `order`.

## The accuracy suite (16c: design frame)

The 16c plan fixes meshes and levels after the sizing study; the gates are the roadmap's.

- **Common rules:**
  - Everything uses PR 8's `reference_modes` on `assemble`.
  - The error is relative in $f$.
  - No upper-bound checks (ANS/EAS is not conforming).
  - Gates compare against the finest reference level. The observed-order Richardson estimate
    (recomputed by the tests from the recorded levels, $1 \le p \le 5$, monotone) only bounds
    the reference error; the $n_z$ check enters that bound.
- **Benchmarks:**
  - `benchmarks/shell_references.py` (hex27 levels) and `benchmarks/shell_runs.py` (the
    expensive `ssh8` solves) each run in $\le 1$ h with per-level progress and write
    `benchmarks/results/*.json`.
  - The slow tests read them and recompute one representative case per test plus the coarsest
    reference level. JSON rather than test constants because of the number of levels (logged).
- **Srinivas–Rao** (0.1%):
  - Mesh: PR 8b's `plate_fe`, generalized to `H`, `a`, `b`, with hex8 → `solid_shell(direction=(0,0,1))`.
  - Modes: the lowest 10 at $L/t = 10$ and 100, gated per half-wave sub-plate $L/m\times L/n$
    ($L$ for $m = 0$ or $n = 0$).
  - Matching: the sub-plate mode is identified by the sorted index of its mapped label in
    `plate_modes` on the sub-plate, with a gap or MAC check.
  - Equivalence test: the full plate at $2N\times N$ gives mode (2,1) equal to the
    $N\times N$ sub-plate's mode to $10^{-10}$. The sub-plate mode, odd-reflected, is a
    full-plate FE eigenpair.
  - Measured prototype sizes:
    - $L/t = 10$ needs $k \approx 16$ per ply and $n \approx 32$–48, up to about 200k DOF and
      about 3 min per solve, so these runs go to `shell_runs.py`;
    - $L/t = 100$: $n = 48$, $k = 1$, 27k DOF, 2 s.
  - An `ORDER` at $L/t = 100$ is asserted from `plate_modes`.
- **Clamped cylinder:**
  - Geometry: $R/t = 50$, $L/R = 2$, isotropic, $u = 0$ on both end faces.
  - Model: 1/8 symmetry (quarter circumference, half length). The same mesh has 4 classes by
    `dof_mask`:
    - circumferential S about $x = 0$: $u_x = 0$; A: $u_y = u_z = 0$;
    - $y = 0$ symmetric: $u_y = 0$;
    - axial S at mid-length: $u_z = 0$; A: $u_x = u_y = 0$.

    Excluded: the $n = 0$ torsion modes and the classes antisymmetric about $y = 0$ (the
    doublet partners). Logged.
  - Matching: sorted per class, the lowest 6.
  - Measured prototype:
    - hex27 levels 24/48/96 at $n_z = 2$: $p = 2.0$–3.4; reference error $\le 0.008\%$; $n_z$
      4 vs 2: $\le 6\cdot10^{-5}$; 275k DOF and 114 s at the finest.
    - `ssh8` one layer: worst 0.080% at $384\times96\times1$ (220k DOF, 33 s) and 0.072% at
      $512\times96\times1$. One layer converges about 0.06% above 3D, so the margin is at most
      about 1.4 (recorded).
  - Two layers are stiffer and gain nothing.
- **No locking:**
  - Setup: the SS square isotropic plate on $6\times6\times1$, at $L/t = 10$ and 1000. 12 modes
    are computed, and the first 6 bending modes are kept by z-reflection parity.
  - Error, in $\lambda$: the max over those modes of $|\lambda/\lambda_\text{conv} - 1|$.
    $\lambda_\text{conv}$ comes from the levels 12/24/48, extrapolated with $p$ from $L/t = 10$.
  - Floor: $10\varepsilon\lambda_\max/\lambda$ in $\lambda$, with $\lambda_\max$ from
    `eigsh(which="LA")`, computed in the test (measured $9.2\cdot10^{-4}$ for mode 1 at 1000).
    The max error must be $\ge 10\times$ the floor.
  - Gate: error at 1000 $\le 1.5\times$ error at 10 (measured 1.27 in $f$).
  - Reported only: the hex8 baseline.
- **Skew plate:**
  - Setup: parallelograms only, 30° skew, clamped, $L/t = 20$, the lowest 6 modes sorted.
  - Reference: hex27 levels 6/12/24/48 at $n_z = 2$ (135k DOF; $p$ 2.5 → 1.9; estimate
    $\approx 0.05\%$ against the 0.2% gate).
  - `ssh8`: about $128\times128\times1$ for 1% (one layer converges about 0.6% above 3D).

## Docs (each with its part)

- **16a:**
  - `docs/elements.md`: `ssh8`, formulation, quadrature, frames, locking, patch-test limits,
    recovery, refusals.
  - `docs/mesh.md`: `solid_shell`, permutations, thickness and sign rules, the caveats.
  - `docs/materials.md`: `Laminate`/`Ply`, the shell frame, ply angles, the stacking sign,
    hex vs `ssh8` orientation.
- **16b:**
  - `docs/operators.md`: leaves, estimator.
  - `docs/eigensolvers.md` and `docs/preconditioners.md`: Jacobi auto, AMG until PR 18, the
    Jacobi numbers.
  - README example.
  - `docs/mesh.md`: SC8R, VTU.
- **16c:** `docs/verification.md`.
- **Each part:** ROADMAP status, `docs/open_items.md`.

## Decisions without user input (logged in open_items)

- **Process:** the three-way split; 16a refusing the operator paths; 16c's own plan after the
  sizing study.
- **Type and producer:**
  - type `ssh8`, placed last in `TYPES`;
  - `solid_shell` returning `order`;
  - `box_mesh(type="ssh8")` refused;
  - hand-built blocks are the user's responsibility.
- **Thickness rules:** cosine margin 0.1; unique shortest chain; the per-element sign rule;
  proper-rotation permutations; no edge-length check; `direction` required for off-axis,
  non-orthotropic or unsymmetric laminates.
- **Laminates and frame:**
  - `Laminate`/`Ply` inside `materials`, 2 points per ply;
  - the placeholder phase entry;
  - the frame from $R_o$ axis 1 projected at the centre, fallback to axis 2 at 0.1, angles in
    degrees.
- **Mass:** $m_e$ stored (36 floats, $\rho$ included); the mass scale on `ssh8` is $s_M$ alone.
- **Code structure:** dispatch by type name, with `kernels.*` raising for `ssh8`; `RECOVER`;
  the forms deferred to PR 21.
- **Preconditioners:** Jacobi auto on `ssh8`; AMG refused until PR 18.
- **Element tests:**
  - zero-mode thresholds ($h/t \le 2$ gated, 100 reported);
  - the patch-test oracle and reading (ANS limits);
  - MacNeal–Harder loading and `direction`.
- **16c:**
  - Srinivas–Rao per sub-plate;
  - the cylinder classes and exclusions;
  - observed-order Richardson with the $n_z$ term;
  - the no-locking levels and floor;
  - benchmark JSON references;
  - expensive `ssh8` runs in benchmarks.
- **16b scope:**
  - Jacobi tests at $L/t \le 20$;
  - the M4 example at $L/t = 20$;
  - SC8R stacking kept, other `SC*` refused;
  - the VTU type field deferred.
- **PR 17:** must cover `ssh8` recovery.
