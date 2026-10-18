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

- rev 3 (review round 2: 4 satisfied, scale 2 design points):
  - Column width `min(batch, n)`; `batch` is a public argument.
  - `element_stress` batches columns inside the kernel; geometry is computed once per chunk.
  - Batches are cast to fp64.
  - Outputs are preallocated.
  - Empty mode sets are allowed.
  - `last_complete` is dropped.
  - Block-table helper is shared with `assemble`.
  - `cell_data` contract and VTK tensor order.
  - The recipe rtol comes from `info["thresholds"]`.
  - Tests pinned: setups, directions, sign-rule cases, compile counts in a subprocess, the
    docs recipe executed.

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
- **Batched products:** a private loop `_mbatches(M, B, batch)` yields `(columns, MB_j)`, and
  `_gram(A, M, B, batch=64)` (host $A^\top MB$) and the diagonal $\mathrm{diag}(B^\top MB)$
  are one-line reductions over it.
  - Column width is $w = \min(\text{batch}, n)$. Only the last batch is zero-padded when
    $n > \text{batch}$, so a 6-column $Mr$ costs 6 columns, not 64. That is one compile per
    width.
  - Each batch is cast to fp64 before `jax.device_put` (Gram matrices are always fp64; the
    stencil `apply_M` follows the input dtype).
  - The next batch's `apply_M` is dispatched before the host GEMM of the current one.
  - A device-array `A` is pulled to the host once, before the loop.
  - With n = 0 the loop makes no `apply_M` call and returns empty arrays.
  - `batch` is an argument of `normalize`, `mac` and `participation` (defaults-and-options
    rule).
  - Device memory is about $(2 + \text{workspace}(w))\,w\,n_\text{dof}\cdot 8$ B plus the
    operator, for any n. Host memory is the inputs plus $O(w\,n_\text{dof})$.
  - A matrix `M` uses `M @ B` per batch.
  - `_gram` also serves PR 23's void-mass check and host-side diagnostics. It is host NumPy,
    not traced, so it is not PR 21's differentiable cluster Gram. PR 19 swaps the host
    accumulation for `psum`.
- **Influence vectors:** `r` are masked like the modes, zero on constrained DOFs. With modes and
  `r` both zero there, $\phi^\top Mr$ and $r^\top Mr$ involve only the free block $M_{ff}$
  (the sum rule "over the free DOFs").
  - This is not the base-excitation load $-(M_{ff}r_f + M_{fc}r_c)\ddot u_g$ for consistent
    mass. With lumped mass the two agree.
  - The total of a supported body is $r_f^\top M_{ff}r_f$, not the body mass.
  - The docs state both points.

## API (`jaxmodal.post`)

```python
normalize(V, M=None, kind="mass", batch=64)          # -> V scaled per column (a copy)
mac(A, B, M=None, batch=64)                          # -> (nA, nB)
influence(mesh, free, origin=None)                   # -> r (n_dof, 6)
participation(V, M, r, batch=64)                     # -> gamma (n, 6), m_eff (n, 6), total (6,)
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
  Ties from iterative solves can fall outside the $10^{-6}$ band, because vector accuracy is
  about tol/gap. Inside a degenerate cluster the sign and the basis are meaningless (docs).
- **Errors:** a zero-norm column, or `"mass"` without `M`, raises `ValueError` ("pass M or
  kind='max'"). Zero columns (n = 0) return an empty array. Returns a copy.

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
  - n = 0 (a clamped body's `info["rigid"]`) returns (0, 6), (0, 6) and `total`.
  - `total` does not depend on `V`, so either recipe call's value is passed on.
- **Sum rule:** over all free modes (rigid included), $\sum_i m^\text{eff}_{ik} = r_k^\top
  Mr_k$.
  - The documented recipe calls `participation(info["rigid"], op, r)` and
    `participation(V, op, r)` and stacks the small `(n, 6)` outputs, rather than
    `np.hstack([rigid, V])`, which copies V.
  - `lam` is prepended with `np.zeros(info["n_rigid"])`.
  - `op` is rebuilt with the same `mass`, `nodal_mass`, `orientations` as `solve_modes`:
    `stencil` for voxel meshes and `unstructured` otherwise, matching `info["operator"]`.
    `free = dof_mask(mesh, bcs)`. The recipe shows every line, and test 2 executes the
    docs' code block (as `tests/test_readme.py` does).
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
  - The solver merges at $\lambda_i - \lambda_{i-1} < 100\,t_{i-1}|\lambda_{i-1}|$, with
    $t_j \ge$ tol the per-mode acceptance threshold. The recipe therefore passes
    `rtol=max(1e-4, 100 * info["thresholds"].max())`.
  - The rule uses $|\lambda_{i+1}|$ and $\le$ on purpose: it is slightly looser, so it never
    splits a solver cluster, and it handles exact rigid zeros.
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
  - `"total"`.

  After an `n_modes` solve the last cluster may be cut, and after a band solve with $f_1 > 0$
  the first may be too, so their sums are not basis-invariant. The docs and the docstring say
  so; `band` solves return whole clusters at the top edge.

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
    result; the per-element $u$ goes in its `p` pytree). It preallocates the outputs and writes
    each chunk into them, with no list plus `np.concatenate` (that doubled the peak). It is
    shared by `assemble`, `recover` and `element_stress`.
  - The per-block material-table loop (`index[e]`, `C[index[e]]` for hex, the sliced
    `ply_table` pytree for `ssh8`, plus the checks) becomes one private helper
    `_block_tables(mesh, materials, orientations)` in `operators/assembled.py`. It is used by
    `assemble` and `post/stress.py`, so the `ssh8` branch lives in one place.
- **`recover(mesh, materials, u, orientations=None, chunk=1024)`:**
  - Takes one field `u (n_dof,)`.
  - Runs `check_materials`, `check_laminates`, and `check_elements(mesh, "consistent")` (as
    `assemble` does).
  - Returns, per block, `{"points" (n_e, nq, 3), "weights" (n_e, nq), "eps" (n_e, nq, 6),
    "sigma" (n_e, nq, 6)}`; `ssh8` padded plies have weight 0. Per-block dicts are used because
    nq differs per type; `element_stress` returns element-order arrays (one docstring line
    each).
  - Use it for quadrature-point detail on a few modes.
- **`element_stress(mesh, materials, U, ...)`:**
  - Returns, for `U (n_dof, k)` (or 1-D), the volume averages $\bar\sigma_e = \sum_q
    w_q\sigma_q/\sum_q w_q$ and $\bar\varepsilon_e$, `(n_el, k, 6)` in element order
    (`mesh.offsets`).
  - Reduced inside the kernel, so only `(c, kb, 6)` per chunk leaves the device. Columns are
    batched inside the kernel: `jax.jit(jax.vmap(jax.vmap(rec, in_axes=(None, None, 1),
    out_axes=1)))` with `kb = min(64, k)`, the last column batch zero-padded.
  - The geometry, B, det J and the `ssh8` Kaa factorization do not depend on u, so they are
    computed once per chunk (the reviewer measured 6× over a per-column loop). Implementation
    confirms the `ssh8` case with one timing.
  - Compiles once per (type, chunk, kb), and per ply count for `ssh8`.
  - Host output is $96\,n_\text{el}\,k$ B for eps and sigma; the docs say to pass `V[:, sel]`.
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
  - `cell_data`: optional `{name: (n_el,) or (n_el, c)}` in element order, split per block.
    meshio writes a 3-D array that then fails to read back, so `ndim > 2`, wrong row counts and
    bool dtype raise. The recipe exports per mode, e.g. `{"vm_003": von_mises(sig[:, 3])}`.
    `material` and `orientation` are written as in `mesh.io.write`.
  - Names colliding with `material`/`orientation` raise. A cell and a point array with the
    same name are legal in VTU, so `mode_*` is not reserved.
  - VTK reads 6-component arrays as symmetric tensors (XX, YY, ZZ, XY, YZ, XZ) with tensor
    shear. The docs recipe therefore writes `sigma[..., [0, 1, 2, 5, 3, 4]]` and eps with
    halved shear terms. `write_modes` itself passes data through unchanged.
  - Binary with `compression="zlib"` passed explicitly, through the shared helper and
    `meshio.vtu.write`. zlib gains little on fp64 fields (40³ hex20, 10 modes: 87 MB), and the
    docs say so. `ssh8` is written as
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
- **`docs/mesh.md`:** the VTU "Write" bullet. `post.write_modes` shares the helper, and hex8 and
  `ssh8` blocks can merge into one hexahedron block on read (which `read` handles).
- **`docs/verification.md`:** the MAC bullet (oracle stays `tests/exact.mac`, checked against
  `post.mac`); the PR 17 checks.
- **`README.md`:** status, package list, recipe snippet.
- **`docs/ROADMAP.md`:**
  - the PR 17 status entry;
  - PR 17 contents/tests "VTU/XDMF" → VTU, with XDMF/HDF5 → PR 19, also in the test line
    "XDMF written with XML data, so no h5py";
  - both "owned by whichever merges last" lines (PR 16 and PR 17 Tests) marked done by PR 17;
  - a PR 19 Contents bullet: parallel output (XDMF/HDF5 or PVTU pieces, the h5py extra
    decision);
  - PR 21's "Cluster detection" reuses `post.clusters`.
- **`docs/open_items.md`:** the decisions above:
  - VTU only; XDMF/HDF5 or PVTU and h5py in PR 19;
  - frequencies in a sidecar `format_table` file;
  - the rtol/atol defaults;
  - masked r;
  - single host;
  - `s_K`;
  - the sign rule;
  - the VTK tensor order.
- **Cost and memory** (in `postprocessing.md`): the formulas, plus one-off measured lines from
  the reviewer's probe:
  - `element_stress` about 23 ms per 1024-element hex20 chunk × 64 columns;
  - `_gram` host GEMM about 0.6 s per 64-column batch at 1M DOF × 2000 modes, remeasured in
    implementation.

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
   - (c) `influence` columns are checked against an independent `np.cross`, for the default and
     a non-default origin, and are zero on constrained rows. The default equals `rigid_modes`'
     node centroid (on a graded mesh it differs from the centre of mass).
   - (d) Free regular `box_mesh`, one density, consistent mass: the totals equal the analytic $\rho V$ and the box inertias
     about the centroid, e.g. $\rho V(b^2+c^2)/12$, to $10^{-12}$ (2×2×2 Gauss is exact). The
     6 rigid modes give all 6 totals, and every elastic $m^\text{eff}$ is $\le 10^{-20}$ of the
     total.
   - (e) A `stencil` operator on a voxel box gives the same totals and m_eff as its assembled
     M, to $10^{-12}$ relative. `normalize("mass")` through an operator equals the matrix
     result.
   - (f) n = 0 columns: `participation` returns (0, 6) and the totals; `normalize` returns
     empty.
2. **Recipe after `solve_modes`** (executes the docs' code block):
   - Setup: hex20 $2\times2\times2$ on (1, 0.83, 0.67), sliding base; 12 modes, `tol=1e-10`.
     `info["rigid"]` holds 3 modes (Tx, Ty, Rz).
   - Per-cluster $m^\text{eff}$ match dense `eigh` to $10^{-8}$ of `total_k`, on clusters
     wholly inside the computed set. Measured: $\le 3\cdot10^{-13}$.
   - The solver-λ cluster partition equals the dense-λ partition. Modes 6–7 are $2.1\cdot10^{-4}$
     apart.
   - Not vacuous: complete elastic clusters carry $\ge 0.5$ of the total in z, Rx, Ry
     (measured 0.81, 0.90, 0.87).
   - The rigid part gives the x, y, Rz totals.
   - A clamped variant (`xmin`, `n_rigid = 0`) runs the same recipe.
   - About 3 s, fast.
3. **Normalization and MAC:**
   - Dense `eigh` modes, randomly scaled with random signs → `normalize("mass")`:
     $\|V^\top MV - I\|_\max \le 10^{-12}$. The off-diagonal part comes from `eigh`; a
     non-orthogonal input keeps its off-diagonal shape (checked).
   - `kind="max"`: max nodal norm 1.
   - Sign rule (synthetic): two near-max entries of opposite sign, $|v_a| = 1$ and
     $|v_b| = 1 - 10^{-12}$ with $a < b$, and the reverse. For v and −v in all four cases,
     $v_a > 0$ after normalizing, and `normalize(v) == normalize(-v)` exactly.
   - `mac(V, V, M)` = I to $10^{-12}$; it equals `tests/exact.mac` to $10^{-13}$ on random A, B.
   - Unweighted MAC against NumPy. Operator `M` = matrix `M`. JAX device input.
   - `_gram` with batch 64: n = 130 compiles exactly one (n_dof, 64) `apply_M`, a repeat call
     compiles nothing, and the results equal the unbatched product. Compile events are all
     `backend_compile` events, counted in a subprocess with the compilation cache off, as in
     test_lobpcg. An fp32 input gives an fp64 Gram.
4. **Cluster invariance:**
   - Setup: square prism $3\times3\times4$ hex8, isotropic, clamped at `zmin` (C4-symmetric,
     centred). Its x/y bending pairs are degenerate to round-off and carry $m^\text{eff}$ in x,
     y, Rx, Ry.
   - Rotating each cluster's M-orthonormal basis by a random orthogonal Q leaves the
     per-cluster $m^\text{eff}$ unchanged to $10^{-12}$ of the total. Per-mode $m^\text{eff}$
     changes by $\ge 10^{-3}$ of the total, max over x, y, Rx, Ry (measured 0.64), so the test is
     not vacuous.
   - `clusters` on synthetic `lam`: a rigid cluster of zeros; a pair within rtol and a pair just
     outside; non-ascending input raises.
5. **Stress recovery = Cε** (imposed linear $u = Ax + b$, random $A$):
   - (a) hex8, hex20, hex27 meshes (`helpers.distort`, straight and curved), 2 materials ×
     2 orientations interleaved (the `mixed_mesh` pattern). At every quadrature point:
     - $\varepsilon$ = Voigt(sym $A$), engineering shear $A_{ij} + A_{ji}$, to $10^{-12}$
       relative to $\|A\|$;
     - $\sigma$ = `rotate(C[material[e]], R[orientation[e]])` (independent of `phase_table`)
       times $\varepsilon$, to $10^{-10}$ relative;
     - the points equal the trilinear map of the corners at the Gauss points on straight
       meshes (`distort(curved=False)` places every node on that map), where the per-element
       weights also equal the trilinear $w\det J$;
     - $\sum_e\sum_q w$ = box volume ($10^{-12}$); per hex8 element it equals a test-written
       $3\times3\times3$ Gauss sum of the trilinear map's $\det J$ ($10^{-12}$; exact, since
       $\det J$ is quadratic in each reference coordinate);
     - $\sum\rho w$ = $\mathbf 1_x^\top M\mathbf 1_x$ from `assemble`.
   - (b) **`ssh8`, the PR 16 roadmap test, owned here:** `test_shell.py::test_patch_recovery`'s
     patches i–iv and all 6 states, one material (one ply), through `post.recover(mesh,
     {0: mat}, u, orientations={0: Q})`. $\sigma$ equals the analytic `rotate(C, Q)`·Voigt(sym
     $A$) (`linear_field`'s σ, in the test's input frame) to $10^{-10}$ relative to
     $\|\sigma\|$. The weights per element equal `shell_ref`'s Gauss det J ($10^{-12}$).
   - (c) **Laminates** (2- and 3-ply, `test_shell.mixed()` mesh with hex8 blocks): `recover`
     equals `tests/shell_ref.recover` per element (α included) to $10^{-12}$. This checks
     ply_table slicing, orientations, point order and block offsets; under a linear u a
     laminate's EAS α ≠ 0, so the imposed ε is not the gate; asserted as
     $\|\alpha_\text{ref}\| \ge 10^{-4}\|\varepsilon\|$ per laminate element. Padded plies
     have w = 0 and finite values.
   - (d) Chunk 5 (not dividing n_e) equals chunk 1024. A repeat call compiles nothing new (all
     `backend_compile` events, in a subprocess as in test 3).
6. **Energy:** a random $u$ and a mode on distorted hex8, hex20, hex27, and a mixed hex8 + `ssh8`
   (laminate) mesh with 2 phases. $\sum_e\sum_q w_q\,\varepsilon_q\cdot\sigma_q = u^\top Ku$ with
   the assembled K, to $10^{-10}$ relative (EAS included for `ssh8`).
7. **Element stress and von Mises:**
   - Under 5(a)'s field, `element_stress` equals the constant $C_e\varepsilon$ per element.
     On a random u it equals the weighted mean of `recover`, also on the mixed hex8 + `ssh8`
     laminate mesh (padding ignored at block level). k = 70 columns (not a multiple of kb = 64)
     equal 70 single calls.
   - von Mises on random symmetric σ equals the principal-stress formula from `eigvalsh` of the
     3×3 tensor. Pure shear in each slot 4–6 gives $\sqrt3|\tau|$; uniaxial gives |σ|;
     hydrostatic gives 0.
8. **Export round trip:**
   - Mesh: hex8, hex20, hex27 and `ssh8` blocks (mixed). `write_modes` → `meshio.vtu.read`.
   - Exact equality of the points, the cells (identity order, concatenated in element order,
     since consecutive hexahedron blocks may merge) and the cell data (concatenated).
   - `point_data[f"mode_{j:03d}"][:, i] == V[i::3, j]` exactly, for all i, j.
   - Name width: n = 1000 gives `mode_999`; n = 1001 gives `mode_0000` … `mode_1000`, and the
     sorted names follow index order.
   - A second round trip on `test_shell.mixed()`, where hex8 + `ssh8` merge into one
     hexahedron block. `mesh.io.read` of that file gives the mesh with `ssh8` as hex8 and the
     labels intact.
   - Integration: `cell_data` = `von_mises(element_stress(...))` per mode.
9. **Frequency table** (the prism of test 4):
   - clusters (expected partition from an independent loop in the test), f, fractions, and the
     cumulative fractions reaching 1 within $10^{-10}$ on all modes;
   - free-prism smoke: the rigid λ are set to exact zeros as in the recipe, plus a synthetic
     $-10^{-12}$ entry, which gives f = 0;
   - `format_table` has a header plus one line per cluster.
10. **Errors:**
    - wrong row counts of `V`, `r`, `free`;
    - MAC row mismatch;
    - zero columns;
    - unknown `kind`;
    - `"mass"` without M;
    - unknown extension;
    - name collisions;
    - `cell_data` with ndim > 2, wrong rows or bool dtype;
    - laminate on hex (via `check_laminates`);
    - chunk < 1.

## Benchmarks

None. After the batching and in-kernel reduction, memory is bounded by `batch` and `chunk`, and
cost follows `apply_M` and `assemble`-like element loops already measured. The docs' "Cost and
memory" section states the formulas.

## Risks

None beyond the logged scope limits.
