# PR 17: Postprocessor (plan)

Roadmap: M5, PR 17. New package `src/jaxmodal/post/` (Package layout: "normalization, MAC,
effective mass, stresses, export"). Depends on PR 4 (assembled oracle), PR 9 (`solve_modes`
outputs), PR 16a (`RECOVER["ssh8"]`).

## Scope

1. Mode normalization.
2. M-weighted MAC.
3. Participation factors and effective modal mass, with rigid modes in the mode set.
4. Strain and stress recovery per mode through `RECOVER` (hex entries added here).
5. VTU/XDMF export and a frequency table with effective masses summed per cluster.

Out of scope: nodal stress smoothing (superconvergent patch recovery), animation/time series
files, HDF5 XDMF (no h5py), stresses from lumped mass models (stress does not depend on the
mass), and `jax.grad` through the postprocessor (PR 21 differentiates eigenvalues, not stresses).

## Conventions shared by all functions

- Modes `V` are host or device arrays `(n_dof, n)` in the full DOF space with constrained rows 0
  (as `reference_modes` and `solve_modes` return them; `solve_modes`'s `info["rigid"]` has the
  same layout). Outputs are NumPy float64.
- `M` is anything with `apply_M` (the matrix-free operators: `unstructured`, `stencil`) or a
  SciPy/NumPy matrix (`assemble`'s M). One private helper `_mass(M, X)` returns `M X` as NumPy:
  `np.asarray(M.apply_M(jnp.asarray(X)))` or `M @ X`. Products are formed once per call
  (`MV = M V`), so every function costs one `apply_M` with `n` columns.
- The influence vectors `r` are masked like the modes: zero on constrained DOFs. With modes and
  `r` both zero there, $\phi^\top M r$ and $r^\top M r$ involve only the free block
  $M_{ff}$, which is the sum rule's left side ("over the free DOFs").

## API (`jaxmodal.post`)

```python
normalize(V, M, kind="mass")                 # -> V scaled per column
mac(A, B, M=None)                            # -> (nA, nB)
influence(points, free=None, origin=None)    # -> r (n_dof, 6)
participation(V, M, r)                       # -> gamma (n, 6), m_eff (n, 6), total (6,)
clusters(lam, rtol=1e-6, atol=None)          # -> list of index arrays
frequency_table(lam, V, M, r, rtol=1e-6)     # -> dict of arrays; format_table(t) -> str
recover(mesh, materials, u, orientations=None, chunk=1024)   # -> {type: dict} per block
element_stress(mesh, materials, u, orientations=None, chunk=1024)   # -> eps, sigma (n_el, 6)
von_mises(sigma)                             # -> (...,)
write_modes(path, mesh, V, lam=None, cell_data=None)          # .vtu or .xdmf
```

### Normalization

- `kind="mass"`: $\phi_i \leftarrow \phi_i/\sqrt{\phi_i^\top M\phi_i}$, so that
  $\mathrm{diag}(V^\top MV) = 1$. Columns are not re-orthogonalized: modes from `eigh`,
  `reference_modes` or LOBPCG are M-orthogonal to solver accuracy already (the test measures
  $\|V^\top MV - I\|_\max \le 10^{-12}$ on dense `eigh` modes after a random column scaling).
- `kind="max"`: the largest nodal displacement magnitude $\max_a\|u_a\|_2$ is 1, with the sign
  fixed so that this node's largest component is positive (a deterministic sign for plots and
  for comparing runs). `"mass"` also fixes the sign the same way.
- A zero column raises (`ValueError`).

### MAC

$$\mathrm{MAC}_{ij} = \frac{(a_i^\top M b_j)^2}{(a_i^\top M a_i)(b_j^\top M b_j)}$$

`M=None` gives the unweighted MAC. Cost: `M A`, `M B` (one `apply_M` each; one when `B is A`).
The test suite's `tests/exact.mac` stays as the independent NumPy oracle.

### Influence vectors, participation and effective mass

- `influence(points, free, origin)`: the 6 rigid-body fields about `origin` (default the
  centroid of the nodes, unweighted): translations $e_x, e_y, e_z$ and rotations
  $e_k \times (x - x_0)$, as columns of `r` `(n_dof, 6)`; rows of constrained DOFs set to 0.
  Columns 4–6 make the rotational effective masses (moments of inertia about $x_0$) available,
  as the usual six-direction modal tables do.
- `participation(V, M, r)`: with $\mu_i = \phi_i^\top M\phi_i$,
  $\Gamma_{ik} = \phi_i^\top M r_k/\mu_i$, $m^\text{eff}_{ik} = (\phi_i^\top M r_k)^2/\mu_i$, and
  $\text{total}_k = r_k^\top M r_k$ (one extra `apply_M` with 6 columns). The modes need not be
  normalized; they must be M-orthogonal for the sum rule.
- **Sum rule:** with all free modes (rigid included), $\sum_i m^\text{eff}_{ik} = r_k^\top M r_k$.
  Rigid modes from `solve_modes` (`info["rigid"]`) are concatenated by the caller:
  `participation(np.hstack([info["rigid"], V]), op, r)`; the docs show this.
- Lumped mass and `nodal_mass` enter only through `M` (nothing to special-case).

### Clusters and frequency table

- `clusters(lam, rtol=1e-6, atol=None)`: sorted `lam`; consecutive eigenvalues join a cluster
  when $|\lambda_{i+1} - \lambda_i| \le \text{rtol}\cdot|\lambda_{i+1}| + \text{atol}$, with
  `atol` default $10^{-9}\max|\lambda|$ so that rigid modes (exact 0 for `info["rigid"]`, about
  $10^{-12}\lambda_\max(K)$ from `eigh`) form one cluster; `lam` must be ascending (raises
  otherwise). Default `rtol=1e-6`:
  degenerate pairs from symmetric meshes are equal to solver accuracy ($\le 10^{-8}$ relative
  at the default tolerances), and physically distinct modes are much further apart; the
  tolerance is an argument.
- Per cluster $C$, $\sum_{i\in C} m^\text{eff}_{ik}$ is invariant under any M-orthogonal change
  of basis within $C$ (it is $r_k^\top M P_C r_k$ with $P_C$ the M-orthogonal projector onto the
  cluster's span), so the table sums per cluster.
- `frequency_table(lam, V, M, r, rtol)` returns `{"modes": [index arrays], "lam", "f"
  (Hz, $\sqrt{\max(\lambda,0)}/2\pi$, cluster mean), "m_eff" (n_clusters, 6), "fraction"
  (m_eff / total), "cumulative" (cumulative fraction), "total" (6,)}`. `format_table(t)` gives a
  fixed-width text table (mode range, f, the 6 fractions, the 3 translational cumulative
  fractions).

### Strain and stress recovery

- `RECOVER` gains `hex8`, `hex20`, `hex27`: `hex_recover(name, X, C, u)` returns the same 4-tuple
  as `shell_recover`: quadrature points $N X$ `(nq, 3)`, weights $w\det J$ `(nq,)`, global Voigt
  strain $\varepsilon = Bu$ (engineering shear, from `kernels.gradients`/`_strain`) and
  $\sigma = C\varepsilon$ `(nq, 6)`. `table_e` is each type's kernel input: the rotated
  $C$ `(6, 6)` of the element's phase for hex (`phase_table`), `(plies, R_o)` for `ssh8`
  (`ply_table`), as `assemble` already selects them. The registry stays `{type: fn(X, table_e,
  u)}` (hex entries are `functools.partial(hex_recover, t)`).
- `recover(mesh, materials, u, orientations=None, chunk=1024)`: one displacement field
  `u (n_dof,)`, per block `{"points" (n_e, nq, 3), "weights" (n_e, nq), "eps" (n_e, nq, 6),
  "sigma" (n_e, nq, 6)}`, computed by `jax.jit(jax.vmap(RECOVER[t]))` in chunks like
  `assembled._chunked` (device memory bounded by the chunk). Materials are checked like
  `assemble` (`check_materials`, `check_laminates`). Stress is linear in `u`, so modes are
  recovered one at a time (`for i in range(n)`); a mass-normalized mode gives stresses per unit
  modal amplitude.
- `element_stress`: the weighted mean over each element's quadrature points,
  $\bar\sigma_e = \sum_q w_q\sigma_q/\sum_q w_q$ (the volume average; padded `ssh8` plies have
  weight 0), and the same for $\varepsilon$, in element order (`mesh.offsets`). This is what the
  export writes as cell data. On laminates the average mixes plies; the per-ply values are in
  `recover`.
- `von_mises(sigma)`: $\sqrt{\tfrac12[(\sigma_{11}-\sigma_{22})^2 + (\sigma_{22}-\sigma_{33})^2 +
  (\sigma_{33}-\sigma_{11})^2] + 3(\sigma_{23}^2 + \sigma_{13}^2 + \sigma_{12}^2)}$ (Voigt
  order 11, 22, 33, 23, 13, 12 as in `docs/conventions.md`; checked against the order there in
  implementation).

### Export

- `write_modes(path, mesh, V, lam=None, cell_data=None)`: point data `mode_{i:03d}` `(n_nodes, 3)`
  per column of `V`; `cell_data` an optional `{name: (n_el, k) or (n_el,)}` (e.g. element stress
  or von Mises per mode), split per block; the mesh's `material` and `orientation` cell data as in
  `mesh.io.write`. `lam` adds field data `frequencies` (Hz) where the format keeps it (VTU); the
  frequency table is the authoritative listing.
- Formats by extension: `.vtu` via `meshio.vtu.write` (binary, as `mesh.io.write`), `.xdmf` via
  `meshio.xdmf.write(..., data_format="XML")` (no h5py). `ssh8` blocks are written as
  `hexahedron` (as `mesh.io.write`; they read back as hex8). `mesh.io.write` and `write_modes`
  share the block-to-meshio conversion (one helper in `mesh/io.py`, now used twice).
- Node order: VTU uses the VTK order, which is the internal order. XDMF `Hexahedron_20` and
  `Hexahedron_27` orderings are checked against the XDMF specification in implementation; if
  they differ from VTK, `write_modes` permutes (and the round-trip test reads back through the
  inverse); if XDMF has no well-defined 27-node order, hex27 raises for `.xdmf` (logged).

## Docs

- New `docs/postprocessing.md`: the conventions, each function with its math, the sum rule and
  cluster invariance, recovery contract per type, export formats and node orders, an example
  (`solve_modes` → `participation` with rigid modes → `frequency_table` → `write_modes`).
- `docs/elements.md`: `RECOVER` now has the hex entries (the "PR 17 adds" sentences updated).
- `docs/verification.md`: the MAC bullet names `jaxmodal.post.mac`; the PR 17 checks.
- `README.md`: status, the package list, a short postprocessing snippet.
- `docs/ROADMAP.md`: PR 17 status entry; the M4 test line "owned by whichever merges last" marked
  done by PR 17 (PR 16 merged first).

## Tests (`tests/test_post.py`)

Fast unless marked; dense oracles on small meshes (`assemble` + `scipy.linalg.eigh` on $M_{ff}$,
$K_{ff}$, all modes).

1. **Sum rule:** hex8 $3\times2\times2$ cantilever (clamped `xmin`) and the same box free (6
   rigid modes from `eigh`), orthotropic rotated material, all free modes: $\sum_i
   m^\text{eff}_{ik} = r_k^\top Mr_k$ to $10^{-10}$ relative for all 6 directions; also with
   `nodal_mass` and lumped mass; and through the matrix-free operator (`unstructured`) as `M`.
   On the free box, the rigid modes alone already give the translational total (rest 0 to
   $10^{-10}$ of the total).
2. **Rigid modes from `solve_modes`:** a free hex20 box, `solve_modes` 12 modes; `participation`
   of `[info["rigid"], V]` matches the dense `eigh` per-cluster effective masses of the same
   modes to $10^{-8}$ (solver tolerance) and the translational sums to the total within
   $10^{-8}$.
3. **Normalization and MAC:** dense `eigh` modes scaled by random factors and signs →
   `normalize(kind="mass")`: $\|V^\top MV - I\|_\max \le 10^{-12}$; `mac(V, V, M)` is the identity to
   $10^{-12}$ and equals `tests/exact.mac` to $10^{-13}$ on random A, B; unweighted MAC against
   NumPy; operator `M` equals matrix `M`; `kind="max"` gives max nodal norm 1 and the sign rule;
   zero column raises.
4. **Cluster invariance:** a cube with cubic symmetry (isotropic, free or symmetrically
   clamped) has exact double/triple modes; rotate each cluster's basis by a random orthogonal
   matrix: per-cluster m_eff unchanged to $10^{-12}$ (relative to the total), per-mode m_eff
   changes (the test is not vacuous). `clusters` on synthetic `lam` (rigid cluster, a pair within
   rtol, a pair just outside).
5. **Stress recovery = Cε:** imposed linear displacement $u = Ax + b$ (random $A$) on distorted
   hex8, hex20, hex27 meshes (random interior node perturbation, curved hex27 edges), anisotropic
   rotated material with 2 phases: at every quadrature point $\varepsilon$ = Voigt of
   $\mathrm{sym}A$ and $\sigma = C_e\varepsilon$ to $10^{-10}$ relative; points equal the mapped
   Gauss points; $\sum_q w_q$ = element volume (against the hex8 analytic volume of a
   parallelepiped). **Solid shell (the PR 16 test, owned here):** a flat distorted `ssh8` patch
   with a 3-ply laminate under the PR 16a imposed fields: through `recover`, $\sigma = C\varepsilon$
   to $10^{-10}$ relative to $\|\sigma\|$ for the states PR 16a shows exact (membrane and
   $\sigma_{33}$ on distorted flat patches; all 6 on parallelograms), against
   `tests/shell_ref.recover`.
6. **Energy check:** for a mode, $\tfrac12\sum_e\sum_q w_q\,\varepsilon_q\cdot\sigma_q =
   \tfrac12\phi^\top K\phi = \lambda/2$ (mass-normalized) to $10^{-10}$ on hex8/hex27 and on a
   mixed hex8 + `ssh8` mesh (ties `recover` to `assemble`'s K, incl. EAS).
7. **Element stress and von Mises:** `element_stress` equals the weighted mean of `recover`;
   von Mises of uniaxial σ is |σ|, of pure shear $\sqrt3|\tau|$, hydrostatic 0.
8. **Export round trip:** `write_modes` to `.vtu` and `.xdmf` (hex8, hex20, hex27, `ssh8`
   blocks, mixed), read back with meshio: points, cells (after the documented permutation, if
   any), every `mode_*` point array and cell array equal to $10^{-15}$ (VTU binary) / to the XML
   ASCII precision stated in implementation (XDMF); field data frequencies on VTU.
9. **Frequency table:** on the cube of test 4: clusters, f, fractions summing to 1 with the
   rigid modes (free cube, all modes), `format_table` has one line per cluster and the header.
10. **Errors:** wrong `V`/`r` row count, `kind` unknown, unknown extension, laminate on hex
    (via `check_laminates`).

## Benchmarks

None: every function is one `apply_M` with $n$ or 6 columns or a chunked element loop like
`assemble`'s, so cost and memory follow existing measured numbers (memory feedback: no
benchmark that informs nothing). One timing line in the docs from a test-scale run is enough.

## Risks

- XDMF high-order node order (above).
- `apply_M` on device with $n$ columns: for very many modes the device holds `(n_dof, n)` twice;
  `participation` and `mac` take a `batch` argument only if a test shows the need (not added up
  front).
