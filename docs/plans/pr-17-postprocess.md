# PR 17: Postprocessor (plan)

Roadmap: M5, PR 17. New package `src/jaxmodal/post/` (Package layout: "normalization, MAC,
effective mass, stresses, export"). Depends on PR 4 (assembled oracle), PR 9 (`solve_modes`
outputs), PR 16a (`RECOVER["ssh8"]`).

Revisions:
- rev 1: first plan.
- rev 2 (review round 1, all 5 not satisfied):
  - VTU only: meshio 5.3.5 imports h5py for XDMF even with XML data, and its reader fails on
    mixed hex20/hex27 grids. XML XDMF is also 13× the size of zlib VTU (347 vs 27 MB measured).
    XDMF/HDF5 is deferred to PR 19 (parallel heavy data). The VTU frequency field data is
    dropped, because meshio's VTU writer never emits `<FieldData>`.
  - Every M product is column-batched (`_gram`). `participation` uses $M r$ (6 columns), and
    `frequency_table` takes `m_eff`.
  - `influence(mesh, free, origin)` takes `free` as a required argument.
  - Recovery kernels are jitted once. `element_stress` reduces inside the kernel and takes
    `U (n_dof, k)`.
  - `clusters`: `rtol=1e-4` (matching `solve_modes`), `atol=0`, rigid modes as exact zeros.
  - Sign rule with ties.
  - Tests:
    - The free-body m_eff checks were vacuous (elastic modes are M-orthogonal to every rigid
      field); the bodies are now partly supported.
    - The `ssh8` gate is single-material against an analytic $\sigma$; laminates are checked
      against `shell_ref`.
    - Analytic totals, independent oracles, chunk padding, a random-$u$ energy check.

## Scope

1. Mode normalization.
2. M-weighted MAC.
3. Participation factors and effective modal mass (6 directions), with rigid modes in the mode
   set.
4. Strain and stress recovery per mode through `RECOVER` (hex entries added here); element
   averages.
5. VTU export and a frequency table with effective masses summed per cluster.

Out of scope, logged in `docs/open_items.md`:

- XDMF/HDF5 (PR 19).
- Nodal stress smoothing.
- Time-series and animation files.
- `jax.grad` through the postprocessor (PR 21 differentiates eigenvalues; the `RECOVER`
  kernels stay pure JAX, so a later stress constraint can reuse them).
- `s_K` scaling of recovered stresses (SIMP stress, PR 21/23).
- Multi-host/sharded `V`: PR 17 is single-host. In PR 19 the batched reductions become
  `psum`s and the chunk loop runs per shard, with no API change.

## Conventions shared by all functions

- **Modes:** `V` are host or device arrays `(n_dof, n)` in the full DOF space, node-major
  (`3a + i`), with constrained rows 0. This is how `reference_modes` and `solve_modes` return
  them; `info["rigid"]` has the same layout. A 1-D array is one column. Outputs are NumPy
  float64.
- **Mass:** `M` is anything with `apply_M` (`unstructured`, `stencil`) or a SciPy/NumPy matrix
  (`assemble`'s M).
- **Batched products:** a private `_gram(A, M, B, batch=64)` returns the host
  $A^\top M B$:
  - over fixed-width column batches of `B`: `jax.device_put` of the batch, the last batch
    zero-padded so `apply_M` compiles once, $MB_j$ back to the host, and $A^\top(MB_j)$
    accumulated there;
  - device memory: batch × n_dof plus the operator, for any n; host memory: the inputs plus
    O(batch × n_dof);
  - a matrix `M` uses `M @ B` per batch;
  - the diagonal variant `_mdiag(B, M)` returns $\mathrm{diag}(B^\top MB)$ through the same
    loop;
  - `_gram` is also the helper for PR 21's cluster Gram and PR 23's void-mass check.
- **Influence vectors:** `r` are masked like the modes, zero on constrained DOFs. With modes and
  `r` both zero there, $\phi^\top Mr$ and $r^\top Mr$ involve only the free block $M_{ff}$
  (the sum rule "over the free DOFs").
  - This is not the base-excitation load $-(M_{ff}r_f + M_{fc}r_c)\ddot u_g$ for consistent
    mass. With lumped mass the two agree.
  - The total of a supported body is $r_f^\top M_{ff}r_f$, not the body mass.
  - The docs state both points.

## API (`jaxmodal.post`)

```python
normalize(V, M=None, kind="mass")                    # -> V scaled per column (a copy)
mac(A, B, M=None)                                    # -> (nA, nB)
influence(mesh, free, origin=None)                   # -> r (n_dof, 6)
participation(V, M, r)                               # -> gamma (n, 6), m_eff (n, 6), total (6,)
clusters(lam, rtol=1e-4, atol=0.0)                   # -> list of index arrays
frequency_table(lam, m_eff, total, rtol=1e-4, atol=0.0)  # -> dict; format_table(t) -> str
recover(mesh, materials, u, orientations=None, chunk=1024)        # -> {type: dict}
element_stress(mesh, materials, U, orientations=None, chunk=1024) # -> eps, sigma (n_el, k, 6)
von_mises(sigma)                                     # -> (...,)
write_modes(path, mesh, V, cell_data=None)           # .vtu
```

Layout:

- `post/modal.py`: normalize, mac, influence, participation, clusters, frequency_table,
  format_table, `_gram`.
- `post/stress.py`: recover, element_stress, von_mises.
- `post/export.py`: write_modes.
- `hex_recover` goes in `elements/kernels.py`, next to `_strain` and `gradients`.
- The meshio conversion of blocks and cell data is one helper in `mesh/io.py`, shared by
  `write` and `write_modes`.

### Normalization

- **`kind="mass"`** (needs `M`): $\phi_i \leftarrow \phi_i/\sqrt{\phi_i^\top M\phi_i}$ (one
  `_mdiag`). Columns are not re-orthogonalized, since solver modes are M-orthogonal to solver
  accuracy.
- **`kind="max"`:** $\max_a\|u_a\|_2 = 1$.
- **Sign** (both kinds): among the DOFs with $|v_j| \ge (1 - 10^{-6})\max|v|$, the lowest index
  is made positive. This is deterministic under mirror ties, where argmax picks by rounding.
  Inside a degenerate cluster the sign (and the basis) is meaningless (docs).
- **Errors:** a zero column, or `"mass"` without `M`, raises `ValueError`. Returns a copy.

### MAC

$$\mathrm{MAC}_{ij} = \frac{(a_i^\top M b_j)^2}{(a_i^\top M a_i)(b_j^\top M b_j)}$$

- `M=None` gives the unweighted MAC.
- Cost: `_gram(A, M, B)` plus the two diagonals (one batched pass when `B is A`).
- A zero column raises.
- `tests/exact.mac` stays as the independent NumPy oracle, and the PR 8 pairing tests keep it
  (they are oracles; `post.mac` is checked against it).

### Influence vectors, participation and effective mass

- **`influence(mesh, free, origin=None)`:** the 6 rigid-body fields about `origin`, as columns
  of `r` `(n_dof, 6)`: translations $e_x, e_y, e_z$ and rotations $e_k \times (x - x_0)$, with
  the same sign convention as `rigid_modes`.
  - The default origin is the unweighted node centroid, as in `rigid_modes`. It is neither the
    centre of mass nor a support point, and rotational $m^\text{eff}$ depend on it (docs).
  - `free` is required (`None` = no constraints). A forgotten mask on a supported body would
    silently break the sum rule.
  - Constrained rows are 0.
- **`participation(V, M, r)`:**
  - $b = V^\top(Mr)$: one 6-column `apply_M`, then a host GEMM.
  - $\mu = \mathrm{diag}(V^\top MV)$ (`_mdiag`).
  - $\Gamma_{ik} = b_{ik}/\mu_i$, $m^\text{eff}_{ik} = b_{ik}^2/\mu_i$, total$_k = r_k^\top Mr_k$.
  - The modes need not be normalized; the sum rule needs them M-orthogonal.
- **Sum rule:** over all free modes (rigid included), $\sum_i m^\text{eff}_{ik} = r_k^\top
  Mr_k$.
  - The documented recipe calls `participation(info["rigid"], op, r)` and
    `participation(V, op, r)` and stacks the small `(n, 6)` outputs, rather than
    `np.hstack([rigid, V])`, which copies V.
  - `lam` is prepended with `np.zeros(info["n_rigid"])`.
  - `op` is rebuilt with the same `mass`, `nodal_mass`, `orientations` as `solve_modes`, and
    `free = dof_mask(mesh, bcs)`. The recipe shows every line, and test 2 runs it verbatim.
- On a free body every elastic mode has $m^\text{eff} = 0$ in all 6 directions, since each
  influence field is rigid (docs).

### Clusters and frequency table

- **`clusters(lam, rtol=1e-4, atol=0.0)`:**
  - `lam` must be ascending (raises otherwise).
  - Consecutive eigenvalues join a cluster when $|\lambda_{i+1} - \lambda_i| \le
    \text{rtol}\cdot|\lambda_{i+1}| + \text{atol}$.
  - Rigid modes enter as exact zeros. `solve_modes`' rigid modes have no $\lambda$, and the
    recipe passes zeros. Dense oracles set them to 0 or pass `atol`.
  - Default `rtol=1e-4` is `solve_modes`' own cluster rule (relative gap $< 100\,\text{tol}$ at
    tol $10^{-6}$). Below about $100\,\text{tol}$ the per-mode vectors mix ($\sin\theta \approx
    \text{tol}/\text{gap}$, Davis–Kahan). Merging distinct modes only coarsens a row, never
    makes it wrong.
  - Documented: rtol $\ge 100\times$ the solve tolerance.
- **Basis invariance:** per cluster $C$, $\sum_{i\in C}m^\text{eff}_{ik} = r_k^\top MP_Cr_k$
  ($P_C$ the M-orthogonal projector onto the span) is invariant under any M-orthogonal change
  of basis within $C$.
- **`frequency_table(lam, m_eff, total, rtol, atol)`:** a host function. It returns:
  - `"clusters"`: index arrays;
  - `"lam"`, the cluster mean;
  - `"f"`: Hz, $\sqrt{\max(\lambda, 0)}/2\pi$, so tiny negative rigid values give 0;
  - `"m_eff"`: (n_clusters, 6);
  - `"fraction"`: m_eff / total, with a fraction of 0 where the total is 0;
  - `"cumulative"`;
  - `"total"`;
  - `"last_complete"`: False flags that the last cluster may be cut by `n_modes`. `band` solves
    return whole clusters; the docs explain.

  `format_table(t)` is a fixed-width text table: modes, f, the 6 fractions, and the 3
  translational cumulatives.
- PR 21's "cluster detection" reuses `post.clusters` (one rule).

### Strain and stress recovery

- **Hex entries:**
  - `kernels.hex_recover(name, X, C, u)`, batched over `...` like the other kernels, returns the
    `shell_recover` 4-tuple: quadrature points $NX$ `(nq, 3)`, weights $w\det J$ `(nq,)`, the
    global Voigt strain $\varepsilon = Bu$ (`_strain`, engineering shear), and
    $\sigma = C\varepsilon$ `(nq, 6)`.
  - `RECOVER` gains `hex8`/`hex20`/`hex27` = `partial(hex_recover, t)`.
  - The `RECOVER` comment and `docs/elements.md` say `table_e` is the type's kernel input:
    rotated $C$ (6, 6) of the element's phase (`phase_table`) for hex, and `(plies, R_o)`
    (`ply_table`) for `ssh8`.
- **Kernels and chunk loop:**
  - Jitted kernels are built once at module level, like `assembled._KERNELS`/`_SHELL`:
    `jax.jit(jax.vmap(...))` per type, a full variant and a reduced variant (weighted mean over
    q inside the kernel).
  - Compiles once per (type, chunk shape). A test checks the compile count on a repeat call,
    with PR 9's `jax.monitoring` pattern.
  - `assembled._chunked` is generalized to pytree outputs (`jax.tree.map` over the kernel
    result; the per-element $u$ goes in its `p` pytree) and shared by `assemble`, `recover` and
    `element_stress`.
- **`recover(mesh, materials, u, orientations=None, chunk=1024)`:**
  - Takes one field `u (n_dof,)`.
  - Runs `check_materials`, `check_laminates`, and `check_elements(mesh, "consistent")` (as
    `assemble` does).
  - Returns, per block, `{"points" (n_e, nq, 3), "weights" (n_e, nq), "eps" (n_e, nq, 6),
    "sigma" (n_e, nq, 6)}`; `ssh8` padded plies have weight 0.
  - Use it for quadrature-point detail on a few modes.
- **`element_stress(mesh, materials, U, ...)`:**
  - Returns, for `U (n_dof, k)` (or 1-D), the volume averages $\bar\sigma_e = \sum_q
    w_q\sigma_q/\sum_q w_q$ and $\bar\varepsilon_e$, `(n_el, k, 6)` in element order
    (`mesh.offsets`).
  - Reduced inside the kernel, so only `(c, 6)` per chunk leaves the device. It loops over the
    k columns with one compile.
  - On laminates the average mixes plies; per-ply values come from `recover`.
  - A mass-normalized mode gives stresses per unit modal amplitude. Stress does not depend on
    the mass model.
- **`von_mises(sigma)`:** Voigt order (11, 22, 33, 23, 13, 12) as in `docs/conventions.md`:
  $\sqrt{\tfrac12[(\sigma_{11}-\sigma_{22})^2 + (\sigma_{22}-\sigma_{33})^2 +
  (\sigma_{33}-\sigma_{11})^2] + 3(\sigma_{23}^2 + \sigma_{13}^2 + \sigma_{12}^2)}$. Exported von
  Mises is of the element-average stress, not the average of point values (docs).

### Export

- **`write_modes(path, mesh, V, cell_data=None)`** writes `.vtu` only (other extensions raise,
  in the style of `mesh.io.write`).
  - Point data: `mode_{i:0{w}d}` = `V[:, i].reshape(-1, 3)`, with $w = \max(3,
    \mathrm{len}(\mathrm{str}(n-1)))$ so names sort past 999.
  - `cell_data`: optional `{name: (n_el, ...)}` in element order, split per block. `material`
    and `orientation` are written as in `mesh.io.write`.
  - Names colliding with those or with `mode_*` raise.
  - Binary with zlib, through the shared helper and `meshio.vtu.write`. `ssh8` is written as
    `hexahedron` (it reads back as hex8). No node permutation: VTK order is the internal order,
    and the reviewer checked it in ParaView 6.0 for hex8/20/27 and mixed meshes.
  - Rigid columns are just columns. For large models the caller passes `V[:, sel]` (meshio
    builds each array in memory; about 24 B × n_nodes per mode before zlib), as the docs state.
- Frequencies are not stored in the VTU. The docs show writing `format_table` next to it.

## Docs

- **New `docs/postprocessing.md`:**
  - the conventions (masked $r$ vs base excitation, the totals, the origin);
  - each function with its math;
  - the sum rule, the free-body zero, and cluster invariance with the rtol rule;
  - the recovery contract per type;
  - export;
  - the full recipe after `solve_modes`;
  - a **Cost and memory** section: `apply_M` calls per function, device and host memory versus
    batch and chunk, recovery dispatches per mode, VTU size per mode.
- **`docs/elements.md`:** `RECOVER` hex entries; the "PR 17 adds" sentences updated.
- **`docs/verification.md`:** the MAC bullet (oracle stays `tests/exact.mac`, checked against
  `post.mac`); the PR 17 checks.
- **`README.md`:** status, package list, recipe snippet.
- **`docs/ROADMAP.md`:**
  - the PR 17 status entry;
  - PR 17 contents/tests "VTU/XDMF" → VTU, with XDMF/HDF5 → PR 19, also in the test line
    "XDMF written with XML data, so no h5py";
  - the M4 line "owned by whichever merges last" marked done by PR 17.
- **`docs/open_items.md`:** the decisions above (VTU only, no field data, rtol/atol defaults,
  masked r, single host, `s_K`, sign rule).

## Tests (`tests/test_post.py`)

Fast unless marked. The dense oracles use `assemble` + `scipy.linalg.eigh` on $K_{ff}, M_{ff}$
with all modes, on 100–300 DOF.

1. **Sum rule and formulas** (hex8 $3\times2\times2$, orthotropic rotated material, two phases):
   - (a) Clamped `xmin`, and a sliding base (`{"zmin": "z"}`) for partial constraints. Over all
     free modes, $\sum_i m^\text{eff}_{ik}$ = dense NumPy $r_f^\top M_{ff}r_f$ to $10^{-10}$,
     all 6 directions. This runs for consistent, lumped and `nodal_mass`, through the matrix
     and through `unstructured`.
   - (b) Modes are randomly scaled first. $\Gamma$ and $m^\text{eff}$ per mode are checked
     against a dense test-written formula to $10^{-12}$.
   - (c) `influence` columns are checked against an independent `np.cross`, and are zero on
     constrained rows.
   - (d) Free box, consistent mass: the totals equal the analytic $\rho V$ and the box inertias
     about the centroid, e.g. $\rho V(b^2+c^2)/12$, to $10^{-12}$ (2×2×2 Gauss is exact). The
     6 rigid modes give all 6 totals, and every elastic $m^\text{eff}$ is $\le 10^{-20}$ of the
     total.
   - (e) A `stencil` operator on a voxel box gives the same as its assembled M.
2. **Recipe after `solve_modes`:**
   - Setup: hex20 box with the sliding base (`info["rigid"]` = 3 modes Tx, Ty, Rz); 12 modes,
     `tol=1e-10`; the docs recipe verbatim.
   - Per-cluster $m^\text{eff}$ (clusters wholly inside the computed set) match dense `eigh` to
     $10^{-8}$. Elastic modes carry z, Rx, Ry, so this is not 0 = 0.
   - The rigid part gives the x, y, Rz totals.
   - Slow only if > 10 s.
3. **Normalization and MAC:**
   - Dense `eigh` modes, randomly scaled with random signs → `normalize("mass")`:
     $\|V^\top MV - I\|_\max \le 10^{-12}$. The off-diagonal part comes from `eigh`; a
     non-orthogonal input keeps its off-diagonal shape (checked).
   - `kind="max"`: max nodal norm 1.
   - Sign rule on a mirror-symmetric torsion mode: deterministic under a node relabelling that
     keeps the lowest index.
   - `mac(V, V, M)` = I to $10^{-12}$; it equals `tests/exact.mac` to $10^{-13}$ on random A, B.
   - Unweighted MAC against NumPy. Operator `M` = matrix `M`. JAX device input.
   - `_gram` with batch 64: `apply_M` compiles once for n = 3 and n = 130 (`jax.monitoring`),
     and the results equal the unbatched product.
4. **Cluster invariance:**
   - Setup: square prism $3\times3\times4$ hex8, isotropic, clamped at `zmin` (C4-symmetric,
     centred). Its x/y bending pairs are degenerate to round-off and carry $m^\text{eff}$ in x,
     y, Rx, Ry.
   - Rotating each cluster's M-orthonormal basis by a random orthogonal Q leaves the
     per-cluster $m^\text{eff}$ unchanged to $10^{-12}$ of the total. Per-mode $m^\text{eff}$
     changes by $\ge 10^{-3}$ of the total (not vacuous).
   - `clusters` on synthetic `lam`: a rigid cluster of zeros; a pair within rtol and a pair just
     outside; non-ascending input raises.
5. **Stress recovery = Cε** (imposed linear $u = Ax + b$, random $A$):
   - (a) hex8, hex20, hex27 meshes with distorted interiors (curved hex27 edges), 2 materials ×
     2 orientations interleaved (the `mixed_mesh` pattern). At every quadrature point:
     - $\varepsilon$ = Voigt(sym $A$) to $10^{-12}$;
     - $\sigma$ = `rotate(C[material[e]], R[orientation[e]])` (independent of `phase_table`)
       times $\varepsilon$, to $10^{-10}$ relative;
     - the points equal the trilinear map of the corners at the Gauss points on straight-edged
       meshes;
     - $\sum_e\sum_q w$ = box volume ($10^{-12}$); per hex8 element it equals a test-written
       $3\times3\times3$ Gauss sum of the trilinear map's $\det J$ ($10^{-12}$; exact, since
       $\det J$ is quadratic in each reference coordinate);
     - $\sum\rho w$ = $\mathbf 1_x^\top M\mathbf 1_x$ from `assemble`.
   - (b) **`ssh8`, the PR 16 roadmap test, owned here:** `test_shell.py::test_patch_recovery`'s
     patches i–iv and all 6 states, one material (one ply), through `post.recover(mesh,
     {0: mat}, u, orientations={0: Q})`. $\sigma$ equals the analytic
     `rotate(C, R_shell R_z(θ))`·Voigt(sym $A$) (`linear_field`'s σ) to $10^{-10}$ relative to
     $\|\sigma\|$. The weights per element equal `shell_ref`'s Gauss det J ($10^{-12}$).
   - (c) **Laminates** (2- and 3-ply, `test_shell.mixed()` mesh with hex8 blocks): `recover`
     equals `tests/shell_ref.recover` per element (α included) to $10^{-12}$. This checks
     ply_table slicing, orientations, point order and block offsets; under a linear u a
     laminate's EAS α ≠ 0, so the imposed ε is not the gate. Padded plies have w = 0 and finite
     values.
   - (d) Chunk 5 (not dividing n_e) equals chunk 1024. A repeat call compiles nothing new.
6. **Energy:** a random $u$ and a mode on distorted hex8, hex20, hex27, and a mixed hex8 + `ssh8`
   (laminate) mesh with 2 phases. $\sum_e\sum_q w_q\,\varepsilon_q\cdot\sigma_q = u^\top Ku$ with
   the assembled K, to $10^{-10}$ relative (EAS included for `ssh8`).
7. **Element stress and von Mises:**
   - Under 5(a)'s field, `element_stress` equals the constant $C_e\varepsilon$ per element.
     On a random u it equals the weighted mean of `recover`. `k` columns equal k single calls.
   - Laminate padding is ignored.
   - von Mises on random symmetric σ equals the principal-stress formula from `eigvalsh` of the
     3×3 tensor. Pure shear in each slot 4–6 gives $\sqrt3|\tau|$; uniaxial gives |σ|;
     hydrostatic gives 0.
8. **Export round trip:**
   - Mesh: hex8, hex20, hex27 and `ssh8` blocks (mixed). `write_modes` → `meshio.vtu.read`.
   - Exact equality of the points, the cells (identity order, concatenated in element order,
     since consecutive hexahedron blocks may merge) and the cell data (concatenated).
   - `point_data["mode_000"][:, i] == V[i::3, 0]` exactly, for every mode.
   - Name width at n = 1000.
9. **Frequency table** (the prism of test 4):
   - clusters, f, fractions, and the cumulative fractions reaching 1 within $10^{-10}$ on all
     modes;
   - negative tiny rigid λ gives f = 0 (free prism smoke);
   - `last_complete` with a cut cluster;
   - `format_table` has a header plus one line per cluster.
10. **Errors:**
    - wrong row counts of `V`, `r`, `free`;
    - MAC row mismatch;
    - zero columns;
    - unknown `kind`;
    - `"mass"` without M;
    - unknown extension;
    - name collisions;
    - laminate on hex (via `check_laminates`);
    - chunk < 1.

## Benchmarks

None. After the batching and in-kernel reduction, memory is bounded by `batch` and `chunk`, and
cost follows `apply_M` and `assemble`-like element loops already measured. The docs' "Cost and
memory" section states the formulas.

## Risks

None beyond the logged scope limits.
