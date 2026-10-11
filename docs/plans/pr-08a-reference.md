# PR 8a plan: reference eigensolver, dense inertia, sliding-block oracle

Branch `pr-08a-reference`, from `main` after PR 7.

## Split of PR 8

The roadmap's PR 8 is a solver plus three groups of oracles. It is split in dependency order (CLAUDE.md: `pr-NNa-<slug>`, ...):

| Part | Contents | Roadmap items |
|---|---|---|
| **8a** (this plan) | `reference_modes` (SciPy `eigsh` shift-invert at a negative shift, with its largest supported size), the dense inertia count, the sliding-block oracle, `docs/eigensolvers.md`, `docs/verification.md` (started) | Contents 1 and 2, the first in-test oracle; for the sliding block: the 0.1% target, the convergence rates and the upper bound |
| **8b** | Visscher Ritz (free anisotropic parallelepiped), Srinivas–Rao plates, simply supported BCs on solids, M-weighted MAC pairing | the other two in-test oracles, mode matching, and the remaining 0.1%, rate and upper-bound tests |
| **8c** | NAFEMS FV52 and FV12 (fine hex27 references), the voxel Lamb sphere | the published benchmarks and their gates |

PR 9 needs only 8a (`reference_modes` and the inertia count). 8b and 8c follow 8a directly, before PR 9, as the critical path in the roadmap has them.

## Measured facts (prototype, scratchpad `p8/`; jax 0.11.2, SciPy, fp64, M3 Max CPU)

- **Silent ARPACK failure.** `eigsh(K, k, M, sigma=σ)` with $\sigma < 0$ and $|\sigma|$ well above the wanted eigenvalues returns *wrong* eigenpairs without raising: higher modes, or rigid modes missing. Examples:
  - a free hex20 box, $\sigma = -\bar s$ with $\bar s = \overline{\mathrm{diag} K} / \overline{\mathrm{diag} M}$ (about 160 $\lambda_7$): relative errors of 1.3, and rigid modes missing (default `ncv`);
  - a free hex20 plate with $L/t = 100$, $\sigma = -10^{-2}\bar s$: errors of 2.1, after 21 s.

  The convergence ratio of shift-invert is about $(\lambda_{k+1} - \lambda_k)/|\sigma|$ when $|\sigma| \gg \lambda_k$, so $|\sigma|$ must not exceed $\lambda_k$ by much. On the free thin plate, $\lambda_7 = 3.8\cdot10^{-8}\,\bar s$, so no fixed fraction of $\bar s$ works.
- **σ too close to 0 on a free body.** $K - \sigma M$ is nearly singular along the rigid modes. The elastic eigenvalues lose about $\varepsilon\,\lambda_7/|\sigma|$ times a modest factor: on a free hex20 box ($\lambda_7/\bar s = 6\cdot10^{-3}$) the relative error is $5\cdot10^{-7}$ at $\sigma = -10^{-10}\bar s$ and $1.6\cdot10^{-12}$ at $-10^{-6}\bar s$. For $K$ SPD (clamped, sliding) the first pass at $\sigma_0 = -\sqrt\varepsilon\,\bar s$ is as accurate as a pass at $-0.01\lambda_k$: error over floor 0.40 vs 0.40 (clamped hex27) and 0.99 vs 0.99 (sliding hex20).
- **The algorithm below** matches dense `scipy.linalg.eigh` of $(K_{ff}, M_{ff})$ to $\le 0.99$ of the floor $10\,\varepsilon\,\lambda_\max/\lambda$ for modes up to 20. Each run is two floor-limited solves.
  - The cases were free hex20 boxes (consistent and lumped), a clamped hex27 box, a free hex20 plate with $L/t = 100$ ($\kappa = 10^9$) and a free hex27 plate with $L/t = 200$ ($\kappa = 7.8\cdot10^9$).
  - The 6 rigid eigenvalues come out $\le 4\cdot10^{-9}\lambda_7$ in absolute value.
  - $\|V^\top M V - I\| \le 5\cdot10^{-15}$.
- **Sparse inertia for free.** SuperLU with `permc_spec="MMD_AT_PLUS_A"`, `diag_pivot_thresh=0` and `SymmetricMode=True` keeps `perm_r == perm_c` in every case tried, so $P A P^\top = L U$ with $U = D L^\top$. By Sylvester's law, the number of negative entries of $\mathrm{diag}\,U$ is the number of eigenvalues below $\mu$ for $A = K_{ff} - \mu M_{ff}$.
  - It matched the dense `eigh` count at mid-gap $\mu$ after modes 11, 26 and 61, in 12 of 12 cases: free box, clamped hex27 and plate, consistent and lumped, with both orderings.
  - Inside the rigid cluster the count is undefined, as expected.
- **Dense inertia.** LAPACK `dsytrf` (Bunch–Kaufman, the routine `scipy.linalg.ldl` calls) with the queried `lwork` takes 0.14 s at $n = 3000$ and 7.95 s at 15k. Without `lwork` it falls back to the unblocked code and takes 123 s.
  - `scipy.linalg.ldl` returns dense `lu` and `d`: 3 $n^2$ arrays, 5.4 GB at 15k. Calling `dsytrf` directly with `overwrite_a` needs one, 1.8 GB.
  - Counts matched dense `eigh` for $\sigma$ in gaps with relative width $\ge 10^{-3}$ on the free plates, the boxes and the clamped hex27.
  - A $\sigma$ placed inside a near-degenerate pair (relative gap $10^{-8}$) or inside the rigid cluster gives an ambiguous count; the tests use well-separated gaps.
- **Cost of the routine, measured in a fresh subprocess (harness):** pass 1, pass 2 and the check, always 3 factorizations in the prototype. With the second pass only for near-zero modes, SPD problems need 2 (to be re-measured in the implementation).

  | mesh | free DOF | time | peak |
  |---|---|---|---|
  | hex20 box $10^3$ | 12.8k | 6.0 s | 0.51 GB |
  | hex20 box $14^3$ | 34.6k | 47 s | 2.4 GB |
  | hex20 box $16^3$ | 51.4k | 116 s | 4.8 GB |
  | hex8 box $30^3$ | 83.6k | 380 s | 8.8 GB |
  | hex27 plate $40\times40\times1$ | 59.0k | 5.6 s | 0.9 GB |

  SuperLU is single-threaded and dominates the time (factorization 32 s of 35 s at 51k hex20). `MMD_AT_PLUS_A` and `COLAMD` cost the same. SciPy has no faster sparse direct solver, and CHOLMOD and pypardiso are excluded by the roadmap. Thin, plate-like meshes have small separators and go much further.
- **Sliding block** (exact solution below; $L = (1, 0.83, 0.67)$, the PR 4 orthotropic material with aligned axes and $\rho = 0.7$, and the isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$):

  | element, mass | levels $n$ ($n^3$ elements) | finest error in $\lambda$ (6 modes) | min. fitted rate (6 modes) | time |
  |---|---|---|---|---|
  | hex8 consistent | 6, 8, 12, 16 | 3.2e-3 … 7.1e-3 | 2.01 | 4.4 s |
  | hex8 row-sum | 6, 8, 12, 16 | −1.5e-2 … −3.2e-3 | 1.95 | 4.3 s |
  | hex20 HRZ | 4, 6, 8, 12 | −2.1e-2 … −7.9e-3 | 1.90 | 15–17 s |
  | hex20 consistent | 3, 4, 6, 8 | 3.3e-5 … 7.3e-5 | 3.95 | 1.8 s |
  | hex27 consistent | 3, 4, 6, 8 | 3.3e-5 … 7.2e-5 | 3.92 | 2.9 s |
  | hex27 row-sum | 3, 4, 6, 8 | −1.0e-4 … −1.7e-5 | 3.94 | 2.7 s |

  - Both materials give the same rates. HRZ on the coarser levels 2, 4, 6, 8 gives only 1.46–1.93 (pre-asymptotic), and 1.95–1.97 on 6, 8, 12, 16 (35 s at $n = 16$).
  - Consistent mass converges from above and is monotone in $n$; lumped mass converges from below.
  - The orthotropic exact spectrum has relative gaps $\ge 4.6\%$ among its lowest 8 modes. The isotropic one has exact degeneracies; sorted comparison is still valid, since equal exact values pair either way.
  - At hex20 $n = 2 \ldots 8$ (2 elements is coarse) the lowest 10 modes' sorted rates scatter (3.68–5.38) where a mode crosses another; the lowest 6 at $n \ge 3$ do not.

## Files

```
src/jaxmodal/eigen/__init__.py       reference_modes
src/jaxmodal/eigen/reference.py      reference_modes: eigsh shift-invert, σ rule, inertia check
src/jaxmodal/linalg/__init__.py      count_below
src/jaxmodal/linalg/inertia.py       count_below: dense Bunch-Kaufman inertia
tests/exact.py                       sliding_block: exact sliding-block spectrum (in-test oracle)
tests/test_reference.py              reference_modes, count_below
tests/test_verification.py           sliding block: oracle self-check, 0.1% target, rates, upper bound
docs/eigensolvers.md (new), docs/verification.md (new), docs/ROADMAP.md, README.md
```

The roadmap's package layout puts the SciPy reference in `eigen/` ("reference (scipy)") and the dense inertia in `linalg/`. Neither module imports JAX: both work on assembled SciPy matrices, which need x64 to build (`assemble` checks it).

## API

```python
from jaxmodal.eigen import reference_modes
from jaxmodal.linalg import count_below
from jaxmodal.operators import assemble, dof_mask

K, M = assemble(mesh, materials)
free = dof_mask(mesh, bcs)
lam, V = reference_modes(K, M, 20, free)  # lowest 20: lam (20,) ascending, V (n_dof, 20)
n = count_below(K, M, sigma, free)  # number of eigenvalues of (K_ff, M_ff) below sigma
```

### `reference_modes(K, M, n_modes, free=None, sigma=None) -> (lam, V)`

The lowest `n_modes` eigenpairs of $K_{ff} v = \lambda M_{ff} v$, where $f$ are the free DOFs. $V$ has constrained rows exactly 0 and $V^\top M V = I$ (ARPACK's M-orthonormal vectors). Steps:

1. **Inputs.** `K` and `M` are SciPy sparse, square and of equal shape. `free` is checked like PR 4's mask (`constraints._check_mask`); `None` means all free. `1 <= n_modes` and `n_modes + 1 < n_free` (ARPACK needs $k < n$; one extra mode for the check). `sigma` is `None` or $< 0$. Anything else raises `ValueError`.
2. **Restriction.** `idx = flatnonzero(free)`, $K_f$ = `K[idx][:, idx]` in CSC form, likewise $M_f$. This restricts rather than masks: the masked operators add spurious zero eigenvalues on the constrained DOFs, which the reference must not have. PR 9 compares its masked LOBPCG against these modes padded with zeros.
3. **One shift-invert solve** at $\sigma$ (`_solve`):
   - SuperLU of $A = K_f - \sigma M_f$ with `permc_spec="MMD_AT_PLUS_A"`, `diag_pivot_thresh=0`, `SymmetricMode=True`. $A$ is SPD for $\sigma < 0$, so no pivoting is needed. These are the same options as the check below.
   - `eigsh(K_f, k, M=M_f, sigma=σ, OPinv=LinearOperator(lu.solve), tol=0, v0=…)` with $k$ = `n_modes + 1` and a seeded `v0` (standard normal, `default_rng(0)`), so runs are reproducible. Results are sorted ascending.
4. **The σ rule** when `sigma=None`:
   - **Pass 1** at $\sigma_0 = -\sqrt{\varepsilon}\,\bar s$, with $\bar s = \mathrm{mean}\,\mathrm{diag} K_f / \mathrm{mean}\,\mathrm{diag} M_f$. This is unit-independent, and small enough that $|\sigma_0| \lesssim \lambda_7$ on the free $L/t = 200$ plate ($\lambda_7/\bar s = 3.8\cdot10^{-8}$; $\sqrt\varepsilon = 1.5\cdot10^{-8}$).
   - **Pass 2**, only if pass 1 finds near-zero eigenvalues ($\tilde\lambda_1 < 100\,|\sigma_0|$, i.e. rigid modes or a nearly singular $K_f$), at $\sigma = -0.01\,\tilde\lambda_k$ from pass 1. Then $|\sigma| \le \lambda_k$, which gives good convergence, and $\lambda_7/|\sigma| \le 100\,\lambda_7/\lambda_k$, so the rigid-mode amplification costs at most about 2 digits beyond the floor (measured: none visible).
   - If $\tilde\lambda_k \le 100|\sigma_0|$ too (every wanted mode near zero), pass 1 stands.
   - An explicit `sigma` skips the rule and gives a single pass.
5. **Completeness check** (always). With the computed $\lambda_1 \le \ldots \le \lambda_{n+1}$ ($n$ = `n_modes`):
   - Let $j$ be the largest index $\le n$ with $\lambda_{j+1} - \lambda_j > 10^{-6}\,|\lambda_{n+1}|$ (the last gap at or below the requested count). Rigid-mode noise and exact FE degeneracies, about $10^{-12}$ relative, count as clusters; real gaps are far wider.
   - Factor $K_f - \mu M_f$ at $\mu = (\lambda_j + \lambda_{j+1})/2$ with the options above. If `perm_r != perm_c`, raise `RuntimeError`: the count is not available. Otherwise $c$ = the number of negative entries of $\mathrm{diag}\,U$.
   - $c \ne j$ raises `RuntimeError("eigsh missed modes: c eigenvalues below μ, j found")`. This catches the silent failure above.
   - If no gap exists ($j$ undefined, e.g. `n_modes=3` on a free body, inside the rigid cluster), raise `ValueError` asking for more modes.
   - When `n_modes` ends inside a cluster, only the modes up to the last gap below it are certified; this is documented.
6. **Output.** `lam[:n_modes]`, and `V` with `V[idx] = v[:, :n_modes]`.

Cost: 2 factorizations for SPD $K_f$, 3 with rigid modes (pass 1, pass 2, check). The factorization dominates; fill grows like $n^{4/3}$ for bulky 3D meshes. The largest supported size is about 35k free DOF for bulky hex20 meshes (≈ 47 s, 2.4 GB, three factorizations) and about 50k at ≈ 2 min and 4.8 GB; plate-like meshes reach 59k in 6 s. These are stated in `docs/eigensolvers.md` after re-measuring with the final code. Above this size, PR 9 tests use reference sizes, and 8c's fine hex27 plate references are thin meshes.

### `count_below(K, M, sigma, free=None) -> int`

The exact number of eigenvalues of $(K_{ff}, M_{ff})$ below $\sigma$ (any sign), by Sylvester's law of inertia on $A = K_{ff} - \sigma M_{ff}$ (both symmetric, $M_{ff}$ positive semidefinite):

- **Dense size limit.** `n_free > MAX_DENSE = 20_000` raises `ValueError`: 3.2 GB, about 20 s; the roadmap says ≤ ~15k.
- **Factorization.** $A$ = `(K_f - sigma * M_f).toarray()`. `scipy.linalg.lapack.dsytrf(A, lower=1, lwork=<queried>, overwrite_a=1)`, so one dense copy.
- **Counting.** `ipiv < 0` marks the rows of $2\times2$ pivot blocks. A Bunch–Kaufman $2\times2$ pivot is chosen only when $|a_{kk}|\,\sigma_r < \alpha\lambda^2$ and $|a_{rr}| < \alpha\sigma_r$, so $|a_{kk}a_{rr}| < \alpha^2\lambda^2 < \lambda^2$ and $\det < 0$: one negative eigenvalue per block. The count is therefore the number of negative $1\times1$ pivots plus `(ipiv < 0).sum() // 2` (comment in the code; tested against `eigh`).
- **Errors.** `info > 0` (an exactly zero pivot: $\sigma$ is an eigenvalue to working precision) raises `ValueError`.
- **Inputs.** As for `reference_modes`: `free` is checked; `sigma` is a finite float.
- **Choice of routine.** The roadmap's key decision names `scipy.linalg.ldl`; it is the same LAPACK routine, so the roadmap is updated to say `sytrf` and why (one $n^2$ copy instead of three).

## In-test oracle: the sliding block (`tests/exact.py`)

A box $[0, L_x]\times[0, L_y]\times[0, L_z]$ with sliding (roller) BCs on all six faces: $u_x = 0$ on $x = 0, L_x$ (node sets `xmin`, `xmax` with `"x"`), likewise for y and z. The tangential tractions vanish there. For an orthotropic material with axes aligned to the box (Voigt $C_{44}$: 23, $C_{55}$: 13, $C_{66}$: 12), the ansatz with $k = \pi(p/L_x, q/L_y, r/L_z)$,

$$u_x = A \sin k_x x \cos k_y y \cos k_z z,\quad u_y = B \cos k_x x \sin k_y y \cos k_z z,\quad u_z = C \cos k_x x \cos k_y y \sin k_z z,$$

satisfies the BCs. It reduces the Navier equations to $\Gamma(k)\,(A, B, C)^\top = \rho\lambda\,(A, B, C)^\top$ with

$$\Gamma = \begin{pmatrix} C_{11}k_x^2 + C_{66}k_y^2 + C_{55}k_z^2 & (C_{12} + C_{66})k_xk_y & (C_{13} + C_{55})k_xk_z \\ \cdot & C_{66}k_x^2 + C_{22}k_y^2 + C_{44}k_z^2 & (C_{23} + C_{44})k_yk_z \\ \cdot & \cdot & C_{55}k_x^2 + C_{44}k_y^2 + C_{33}k_z^2 \end{pmatrix}.$$

- Component $i$ exists only when its index ($p$, $q$ or $r$) is $\ge 1$. Each $(p, q, r) \in \mathbb{N}^3 \setminus \{0\}$ contributes the eigenvalues of $\Gamma$ restricted to its existing components. There are no rigid modes, since all normal displacements are zero.
- These families are complete: they are the eigenfunctions of the separable problem, a product of cosine and sine bases on each axis, so no other modes exist.
- `sliding_block(C, rho, L, n, kmax=8)` returns the lowest `n` eigenvalues, sorted. `kmax` is checked to be large enough: the $n$-th value is below the smallest eigenvalue with an index of `kmax + 1`.
- For a sign-flipped ansatz the off-diagonal signs change, but $\Gamma$ stays similar under $\mathrm{diag}(\pm1)$, so the eigenvalues do not change.
- **Self-check (isotropic).** For $p, q, r \ge 1$ the eigenvalues are $(\lambda_L + 2\mu)|k|^2/\rho$ and $\mu|k|^2/\rho$ (twice). With one index 0, only $\mu|k|^2/\rho$ and $(\lambda_L + 2\mu)|k|^2/\rho$ from the two remaining components. Tested against the closed form.

## Tests

References: dense `scipy.linalg.eigh` of $(K_{ff}, M_{ff})$ (the restricted matrices), the exact sliding-block spectrum, and the PR 4 helpers. The floor is $f_\lambda = 10\,\varepsilon\,\lambda_\max/\lambda$, with $\lambda_\max$ from dense `eigh` at test sizes or `eigsh(..., which="LA")` for the convergence meshes.

### `tests/test_reference.py`

1. **Against dense `eigh`**, `n_modes = 20`:
   - **Cases:**
     - a free hex20 box with the orthotropic material, consistent and lumped, with nodal masses including zeros;
     - a clamped curved hex27 (PR 5's mesh B, `xmin: xyz`);
     - the mixed hex8 + hex20 mesh with `xmin: x` (3 rigid modes left);
     - a free hex27 plate $16\times16\times1$ with $L/t = 200$ ($\kappa \approx 8\cdot10^9$, where a fixed $\sigma$ fails).
   - **Criteria:**
     - elastic modes: $|\lambda - \lambda_\text{dense}| \le 2 f_\lambda\,\lambda$ (two floor-limited solves);
     - rigid modes: $|\lambda| \le 10\,\varepsilon\,\lambda_\max$;
     - $\|V^\top M V - I\| \le 10^{-12}$;
     - constrained rows of $V$ exactly 0;
     - residual $\|K v - \lambda M v\| \le 10\,\varepsilon\,(\|K\| + |\lambda|\|M\|)\|v\|$ per mode ($\infty$-norms);
     - an explicit `sigma = -0.01 * lam_dense[n]` gives the same $\lambda$ to $2 f_\lambda$.
2. **Silent-failure guard.** On the free $L/t = 100$ plate, `sigma = -1e-2 * s̄` (the measured silent failure) raises the `RuntimeError` from the check. The test asserts that ARPACK's output is wrong there, i.e. that the check is what catches it.
3. **Clusters.** On an isotropic hex20 cube, ask for an `n_modes` that ends inside an exact FE multiplicity (from dense `eigh`): no error, and the modes up to the last gap match dense `eigh`. On a free box, `n_modes = 3` (inside the rigid cluster) raises `ValueError`.
4. **Inputs.** `n_modes = 0`, `n_modes + 1 >= n_free`, a bad mask, `sigma = 0` or positive, and mismatched shapes raise `ValueError`.
5. **`count_below`** (the roadmap's test). On a clamped model (hex27, `xmin: xyz`) and a free model (hex20 box), consistent and lumped: for $\sigma$ mid-gap between at least 4 consecutive pairs with relative gap $\ge 10^{-3}$ (including the first gap above the rigid cluster for the free model), the count equals the dense `eigh` count below $\sigma$. `free` restricts correctly: a masked model's count equals the count on its restricted dense matrices. `n_free > MAX_DENSE` raises; this is checked by lowering the module constant with `monkeypatch`, not by building a large model.

### `tests/test_verification.py` (sliding block)

- **Mesh and BCs.** `box_mesh((n, n, n), spacing=L/n, type=t)` with the six sliding sets via `dof_mask`. `reference_modes(K, M, 6, free)` for rates; 10 modes for the 0.1% target.
- **Comparison.** Modes are compared sorted against the exact sorted spectrum: the orthotropic gaps are $\ge 4.6\%$, and isotropic degeneracies pair either way.

6. **Oracle self-check** (isotropic closed form, above) and the `kmax` guard.
7. **0.1% target** (roadmap): consistent hex20 and hex27 at $n = 4$ (and at the smallest $n$ that meets the target, stated in `docs/verification.md`), both materials, lowest 10 modes: $|f/f_\text{exact} - 1| \le 10^{-3}$ (measured about $4\cdot10^{-4}$, to be confirmed).
8. **Convergence rates** (`slow`). Least-squares slope of $\log|\lambda^h/\lambda - 1|$ against $\log h$ over the levels of the table, per mode, for the lowest 6 modes, both materials:
   - $\ge 1.8$ for hex8 consistent and row-sum, hex20 HRZ (levels 4, 6, 8, 12) and hex27 row-sum;
   - $\ge 3.6$ for hex20 and hex27 consistent.
   - At the finest level, the error is $\ge 100 f_\lambda$ (roadmap: finest error ≥ 100× the floor).
   - Richardson extrapolation is not used.
9. **Upper bound** (consistent mass only; roadmap). At every level, $\lambda^h_k \ge \lambda_k^\text{exact} - 10\,\varepsilon\,\hat\lambda_\max$. On nested meshes (hex8 $6 \to 12$, $8 \to 16$; hex20 and hex27 $3 \to 6$, $4 \to 8$), $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\,\varepsilon\,\hat\lambda_\max$. Lumped mass converges from below; this is documented, not tested (no theorem).

Expected runtime: test_reference ≈ 15 s, the 0.1% test ≈ 2 s, the slow rate tests ≈ 60 s.

Mutants expected to be killed:
- **σ rule:** pass 2 skipped, a σ multiplier of 1 instead of 0.01;
- **check:** removed, $\mu$ off by one gap, the gap threshold $10^{-6} \to 0$;
- **restriction and output:** masking instead of restriction (spurious zeros), a wrong column order;
- **inertia:** $2\times2$ blocks counted as 0 or 2;
- **oracle:** a wrong $\Gamma$ entry ($C_{55} \leftrightarrow C_{66}$), the component rule removed.

Equivalent mutants: `COLAMD` instead of `MMD_AT_PLUS_A`; a different `v0` seed.

## Docs

- **`docs/eigensolvers.md`** (new). It describes the reference solver:
  - restriction vs masking;
  - shift-invert at a negative shift: why negative, the measured silent failure, and the σ rule with pass 2;
  - the completeness check: Sylvester's law, symmetric SuperLU, the last-gap rule and what it certifies;
  - the accuracy floor $\varepsilon\kappa$ with the measured errors;
  - cost, and the largest supported size (re-measured table);
  - the dense inertia count: Sylvester, Bunch–Kaufman, the indefinite $2\times2$ blocks, memory, `MAX_DENSE`.

  PR 9 adds LOBPCG to the same file.
- **`docs/verification.md`** (new). It covers:
  - the error measures (relative in $f$ for targets, in $\lambda$ for rates) and how modes are matched;
  - the sliding block: setup, the ansatz, $\Gamma$, completeness, meshes, measured errors and rates, the upper bound, and lumped mass from below;
  - sections for 8b and 8c named but empty ("PR 8b", "PR 8c").
- **`docs/ROADMAP.md`:**
  - the status sentence for 8a;
  - the split of PR 8 into 8a/8b/8c under PR 8, with no change to the PR 8 tests;
  - the "Exact count" key decision: `sytrf`, which `ldl` calls, with one copy;
  - the stated largest supported size of the reference solver.
- **`README.md`:** the status, and a short reference-solver snippet after the rigid-modes snippet (run by `test_readme`).

## Not covered

- The other oracles and benchmarks: 8b (Visscher, Srinivas–Rao, MAC pairing) and 8c (NAFEMS, Lamb sphere).
- A faster sparse direct solver: none in SciPy, and the roadmap excludes CHOLMOD and pypardiso. The stated size limit stands. PR 9 compares at reference sizes.
- Interior shift-invert slices with $\sigma > 0$ (PR 15 has its own Lanczos).
