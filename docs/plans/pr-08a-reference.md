# PR 8a plan: reference eigensolver, dense inertia, sliding-block oracle

Branch `pr-08a-reference`, from `main` after PR 7.

## Split of PR 8

The roadmap's PR 8 is a solver plus three groups of oracles. It is split in dependency order (CLAUDE.md: `pr-NNa-<slug>`, ...):

| Part | Contents | Roadmap items |
|---|---|---|
| **8a** (this plan) | `reference_modes` (SciPy `eigsh` shift-invert at a negative shift, with its largest supported size), the dense inertia count, the sliding-block oracle, `docs/eigensolvers.md`, `docs/verification.md` (started) | Contents 1 and 2, the first in-test oracle; for the sliding block: the 0.1% target, the convergence rates and the upper bound |
| **8b** | Visscher Ritz (free anisotropic parallelepiped), Srinivas–Rao plates, simply supported BCs on solids (PR 16 reuses them), M-weighted MAC pairing | the other two in-test oracles, mode matching, and the remaining 0.1%, rate and upper-bound tests |
| **8c** | NAFEMS FV52 and FV12 (fine hex27 references), the voxel Lamb sphere | the published benchmarks and their gates |

The PR 8 acceptance criteria do not change. PR 9 strictly needs only 8a (`reference_modes` and the inertia count). 8b and 8c still come before PR 9. The dependency table row ("8 Reference eigensolver | 4, 7") and the critical path stay as they are, with "PR 8" meaning 8a–8c.

## Measured facts (prototype, scratchpad `p8/`; reviewer probes `r8-*/`; jax 0.11.2, SciPy 1.18.1 with Accelerate, fp64, M3 Max CPU)

$\bar s = \mathrm{mean}\,\mathrm{diag} K_f / \mathrm{mean}\,\mathrm{diag} M_f$; the floor is $f_\lambda = 10\,\varepsilon\,\lambda_\max/\lambda$.

- **Silent ARPACK failure.** `eigsh(K, k, M, sigma=σ)` with $\sigma < 0$ and $|\sigma|$ well above the wanted eigenvalues can return *wrong* eigenpairs without raising: higher modes in their place, or missing rigid modes. Measured cases:
  - a free consistent hex20 box at $\sigma = -\bar s$ (about $160\,\lambda_7$): relative errors of 1.3, rigid modes missing;
  - a free hex20 plate with $L/t = 100$ at $\sigma = -10^{-2}\bar s$: errors of 2.1;
  - the free hex27 plate $8\times8\times1$ with $L/t = 200$ at $\sigma = -10^{-2}\bar s$: one mode missed.

  Whether it happens depends on the case. The lumped box at $-\bar s$ and a $10\times10\times1$ plate at $-10^{-2}\bar s$ come out right.

  Shift-invert converges at a ratio of about $(\lambda_{k+1} - \lambda_k)/|\sigma|$ when $|\sigma| \gg \lambda_k$, so $|\sigma|$ must not be far above $\lambda_k$, the highest *requested* mode.
- **$\lambda_7/\bar s$ is tiny on thin plates:**

  | plate | $\lambda_7/\bar s$ (consistent / lumped) |
  |---|---|
  | free hex20, $L/t = 100$ | $1.95\cdot10^{-8}$ / $2.10\cdot10^{-8}$ |
  | free hex27 $16\times16\times1$, $L/t = 200$ | $1.16\cdot10^{-9}$ / $2.27\cdot10^{-9}$ |
  | free hex27 $8\times8\times1$, $L/t = 200$ | $1.2\cdot10^{-9}$ |

  For comparison, a free hex20 box has $\lambda_7/\bar s = 6.2\cdot10^{-3}$. No fixed fraction of $\bar s$ works for every case.
- **σ too close to 0 on a free body.** $K - \sigma M$ is nearly singular along the rigid modes, and the elastic eigenvalues lose about $\varepsilon\,\lambda_7/|\sigma|$ times a modest factor. On the free hex20 box the error is $5\cdot10^{-7}$ at $\sigma = -10^{-10}\bar s$ and $1.6\cdot10^{-12}$ at $-10^{-6}\bar s$. With pass 2 skipped (below), err/floor is 8.1 (consistent) and 18.5 (lumped) on the box.

  For SPD $K_f$ (clamped, sliding), pass 1 at $\sigma_0 = -\sqrt\varepsilon\,\bar s$ is as accurate as a pass at $-0.01\lambda_k$: err/floor is 0.40 vs 0.40 (clamped hex27) and 0.99 vs 0.99 (sliding hex20). Pass 2 at $\sigma = -\lambda_k$, $-0.1\lambda_k$ and $-0.01\lambda_k$ gives the same accuracy (0.34, 0.16, 0.10 on the box and plate); 0.01 is a safety margin.
- **Pass 1 outside its nominal range still works**, because what matters is $|\sigma_0|$ against $\lambda_k$, not against $\lambda_7$:
  - the $L/t = 200$ plates have $|\sigma_0| \approx 13\,\lambda_7$ and still converge, with $k = 21$;
  - a free hex27 plate with $L/t = 1000$ ($|\sigma_0| \approx 10\,\lambda_7$) gives residual $5.5\varepsilon$.

  Any failure ends in the completeness check below.
- **Accuracy of the algorithm below**, against dense `scipy.linalg.eigh` of $(K_f, M_f)$, up to 20 modes. Each comparison is between two floor-limited solves.
  - **Cases:** free hex20 boxes (consistent and lumped), a clamped hex27 box, a clamped lumped hex20, free hex20 and hex27 plates with $L/t = 100, 200$ ($\kappa$ up to $7.8\cdot10^9$).
  - **Elastic modes:** err/floor $\le 0.99$ (the worst case is sliding hex20; $\le 0.64$ elsewhere).
  - **Rigid modes:** $|\lambda| \le 0.04 \cdot 10\varepsilon\lambda_\max$; dense `eigh` gives up to $0.22 \cdot 10\varepsilon\lambda_\max$.
  - **Orthonormality:** $\|V^\top M V - I\| \le 8\cdot10^{-15}$.
  - **Residuals** on the free rows, $\max_i \|(Kv_i - \lambda_i M v_i)_f\|_\infty / ((\|K_f\|_\infty + |\lambda_i|\|M_f\|_\infty)\|v_i\|_\infty)$: $1.9$–$4.9\,\varepsilon$ on boxes, $4.3$–$10.65\,\varepsilon$ on plates. The full $K$ instead includes reaction forces on constrained rows: $10^{12}$–$10^{14}\varepsilon$.
- **Sparse inertia from the solver's factorization.** SuperLU with `diag_pivot_thresh=0` and `SymmetricMode=True` kept `perm_r == perm_c` in every case tried, so $P A P^\top = L U$ with $U = D L^\top$. By Sylvester's law, the number of negative entries of $\mathrm{diag}\,U$ is then the number of eigenvalues below $\mu$, for $A = K_f - \mu M_f$.
  - It matched the dense `eigh` count at mid-gap $\mu$ after modes 11, 26 and 61, in 12 of 12 cases (free box, clamped hex27 and plate; consistent and lumped; both orderings), and with the ND ordering below (747 negative pivots, the same as MMD).
  - Inside the rigid cluster the count is undefined.
  - This is an unpivoted $LDL^\top$ of an indefinite matrix, so element growth is not controlled. A wrong count is possible in principle. It would almost always appear as a loud `RuntimeError`.
- **Geometric nested dissection** (reviewer prototype, `r8-perf/nd.py`).
  - **Method:** recursive bisection of the free DOFs by rank on the widest coordinate axis (a stable `argsort`, halves of equal size, so ties at the median cannot stall the recursion on graded meshes; the reviewer prototype used `c <= median`). The separator is the left DOFs that are coupled to the right side, read from the CSR rows. Children are ordered first, then the separator, down to leaves of 64 DOFs. Then `splu(A[p][:, p], permc_spec="NATURAL", ...)`.
  - **Overhead:** the ordering takes 0.04–0.15 s. The solve residual is the same (about $5\cdot10^{-13}$) and `perm_r == perm_c` holds.
  - SuperLU's `relax`/`panel_size` options and `COLAMD` gain nothing over MMD; a coordinate-free BFS ordering is worse than MMD.
  - **One factorization:**

  | mesh | free DOF | `MMD_AT_PLUS_A` | geometric ND |
  |---|---|---|---|
  | hex20 $12^3$ | 22k | 4.7 s, 50M fill | 1.4 s, 28M |
  | hex20 $14^3$ | 34.6k | 14.9 s, 106M | 4.7 s, 55M |
  | hex8 $24^3$ | 43k | 15.8 s, 117M | 3.4 s, 54M |
  | hex20 $16^3$ (harness) | 51k | 34.5 s, 2.12 GB | 7.2 s, 1.15 GB |
  | hex8 $30^3$ (harness) | 84k | 118 s, 5.53 GB | 11.6 s, 1.57 GB |
  | hex20 $20^3$ (harness) | ≈ 100k | — | 26.2 s, 2.89 GB |
  | hex27 plate $40\times40\times1$ | 59k | 1.5 s, 34M | 1.7 s, 46M |

- **The full prototype routine** (MMD, always 3 factorizations, harness): hex20 $10^3$ took 6.0 s / 0.51 GB; $14^3$ 47 s / 2.4 GB; $16^3$ 116 s / 4.8 GB. A hex8 $30^3$ box took 380 s / 8.8 GB, and the hex27 plate $40\times40\times1$ 5.6 s / 0.9 GB. The factorization dominates; eigsh adds about 2 s at 35k. With ND, and with 2 factorizations for SPD problems, these drop several-fold; the implementation re-measures them.
- **Dense inertia.** LAPACK `dsytrf` (Bunch–Kaufman, the routine `scipy.linalg.ldl` calls) with the queried `lwork`:
  - Time: 1.9 / 7.4–7.95 / 15.9 s at $n$ = 8k / 15k / 20k. Without `lwork`, the unblocked code takes 123 s at 15k.
  - Memory: f2py overwrites in place only for Fortran-ordered input. `toarray()` peaks at 2.01 $n^2$ blocks; `toarray(order="F")` at 1.00, with the result sharing memory with the input. `scipy.linalg.ldl` returns dense `lu` and `d`, three $n^2$ arrays.
  - Accuracy: counts matched dense `eigh` for $\sigma$ in gaps of relative width $\ge 10^{-3}$. A $\sigma$ inside a near-degenerate pair (relative gap $10^{-8}$) or inside the rigid cluster gives an ambiguous count.
- **Sliding block** (exact solution below). $L = (1, 0.83, 0.67)$, with the PR 4 orthotropic material (axes aligned, $\rho = 0.7$) and the isotropic one ($E = 3$, $\nu = 0.3$, $\rho = 2$). The lowest 6 modes:

  | element, mass | levels $n$ ($n^3$ elements) | finest error in $\lambda$ | min. fitted rate | time |
  |---|---|---|---|---|
  | hex8 consistent | 6, 8, 12, 16 | 3.2e-3 … 7.1e-3 | 2.01 | 4.4 s |
  | hex8 row-sum | 6, 8, 12, 16 | −1.5e-2 … −3.2e-3 | 1.95 | 4.3 s |
  | hex20 HRZ | 4, 6, 8, 12 | −2.1e-2 … −7.9e-3 | 1.90 | 15–17 s |
  | hex20 consistent | 3, 4, 6, 8 | 3.3e-5 … 7.3e-5 | 3.95 | 1.8 s |
  | hex27 consistent | 3, 4, 6, 8 | 3.3e-5 … 7.2e-5 | 3.92 | 2.9 s |
  | hex27 row-sum | 3, 4, 6, 8 | −1.0e-4 … −1.7e-5 | 3.94 | 2.7 s |

  - Both materials give the same rates. HRZ on levels 2, 4, 6, 8 gives only 1.46–1.93 (pre-asymptotic). Consistent mass converges from above and is monotone in $n$; lumped mass converges from below. The finest errors are $\ge 1.2\cdot10^7 f_\lambda$.
  - The orthotropic exact spectrum has relative gaps $\ge 4.6\%$ among the lowest 8 modes. The isotropic one has exact degeneracies, and equal exact values pair either way when sorted.
  - At hex20 $n = 2 \ldots 8$ the lowest 10 modes' sorted rates scatter (3.68–5.38) where modes cross; the lowest 6 at $n \ge 3$ do not.
  - **0.1% target, lowest 10 modes,** maximum error in $f$ (modes 7–10 are the worst):

  | element, material | $n = 4$ | 5 | 6 | 8 |
  |---|---|---|---|---|
  | hex20 orthotropic | 3.3e-3 | 1.3e-3 | 6.4e-4 | 2.0e-4 |
  | hex20 isotropic | 5.2e-3 | 2.1e-3 | 1.03e-3 | 3.3e-4 |
  | hex27 orthotropic | 3.0e-3 | 1.3e-3 | 6.2e-4 | 2.0e-4 |
  | hex27 isotropic | 4.8e-3 | 2.0e-3 | 1.00e-3 | 3.3e-4 |

## Files

```
src/jaxmodal/eigen/__init__.py         reference_modes
src/jaxmodal/eigen/reference.py        reference_modes: ordering, eigsh shift-invert, σ rule, certification
src/jaxmodal/linalg/__init__.py        count_below
src/jaxmodal/linalg/inertia.py         count_below: dense Bunch-Kaufman inertia
src/jaxmodal/operators/constraints.py  + restrict (mask check + restriction, shared by both)
src/jaxmodal/operators/__init__.py     + restrict
tests/exact.py                         sliding_block: exact sliding-block spectrum (in-test oracle)
tests/test_reference.py                reference_modes, count_below, restrict
tests/test_verification.py             sliding block: oracle self-check, 0.1% target, rates, upper bound
docs/eigensolvers.md (new), docs/verification.md (new), docs/boundary_conditions.md, docs/ROADMAP.md, README.md
```

The roadmap's package layout puts the SciPy reference in `eigen/` ("reference (scipy)") and the dense inertia in `linalg/`. Neither imports JAX: both work on assembled SciPy matrices, which need x64 to build (`assemble` checks it). `restrict` sits next to `masked` and `a_p`, so the mask check stays private to `constraints.py`.

## API

```python
from jaxmodal.eigen import reference_modes
from jaxmodal.linalg import count_below
from jaxmodal.operators import assemble, dof_mask

K, M = assemble(mesh, materials)
free = dof_mask(mesh, bcs)
lam, V = reference_modes(K, M, 20, free, points=mesh.points)  # lam (20,) ascending, V (n_dof, 20)
n = count_below(K, M, sigma, free)  # number of eigenvalues of (K_ff, M_ff) below sigma
```

### `restrict(K, M, free=None) -> (K_f, M_f, idx)`

`idx = flatnonzero(free)` (all DOFs for `None`). The mask is checked like PR 4's masks. `K` and `M` must be square and of equal shape. Returns $K_f$ = `K[idx][:, idx]` and $M_f$ in CSC form. Restriction, not masking: the masked operators add spurious zero eigenvalues on the constrained DOFs, which a reference must not have, and spurious pivots in an inertia count (the roadmap's reason).

### `reference_modes(K, M, n_modes, free=None, sigma=None, points=None) -> (lam, V)`

The lowest `n_modes` eigenpairs of $K_f v = \lambda M_f v$. $V$ is `(n_dof, n_modes)` with constrained rows exactly 0 and $V^\top M V = I$. PR 9 compares its masked LOBPCG modes against these.

1. **Inputs.** `restrict` checks `K`, `M` and `free`. Also `1 <= n_modes` and `n_modes + 1 < n_free` (ARPACK needs $k < n$; one extra mode for the check); `sigma` is `None` or a finite value $< 0$; `points` is `None` or `(n_nodes, 3)` with `3 * n_nodes == n_dof`: the `mesh.points` that `K`, `M` were assembled from (DOF $3n + i$). The shape check cannot catch points of another mesh of the same size; wrong points only slow the ordering, never change the result (docstring). Anything else raises `ValueError`.
2. **Ordering** (`_order(A, x)`), computed once per call from the pattern of $K_f$ (which contains that of $M_f$ for consistent mass; for lumped mass it is the union anyway).
   - With `points`: geometric nested dissection on `x = points[idx // 3]` as above (rank split on the widest axis, separator from CSR rows, leaves of `ND_LEAF = 64` DOFs, a module constant), about 15 lines, with factorizations using `permc_spec="NATURAL"`.
   - Without `points`: `p = arange` and `permc_spec="MMD_AT_PLUS_A"`, slower for bulky meshes (table).
   - Both keep `perm_r == perm_c`.
3. **Factorization** (`_lu(A)`, used by the solves and the check): `splu(A[p][:, p], permc_spec=…, diag_pivot_thresh=0, options=dict(SymmetricMode=True))`. The `OPinv` solve is `y[p] = lu.solve(b[p])`. For $\sigma < 0$, $A = K_f - \sigma M_f$ is SPD, so no pivoting is needed.
4. **One shift-invert solve** (`_solve`): `eigsh(K_f, k, M=M_f, sigma=σ, OPinv=…, tol=0, v0=…)` with $k$ = `n_modes + 1` and a seeded `v0` (standard normal, `default_rng(0)`), so runs are reproducible. Results are sorted ascending.
5. **The σ rule** when `sigma=None`:
   - **Pass 1** at $\sigma_0 = -\sqrt{\varepsilon}\,\bar s$. This is unit-independent. It only needs a rough $\tilde\lambda_k$ and works while $|\sigma_0|$ is not far above $\lambda_k$; measured on plates down to $\lambda_7/\bar s = 10^{-9}$ ($L/t = 1000$).
   - **Pass 2**, only if pass 1 finds near-zero eigenvalues ($\tilde\lambda_1 < 100\,|\sigma_0|$: rigid modes or a nearly singular $K_f$), at $\sigma = -0.01\,\tilde\lambda_k$. Then $|\sigma| \le \lambda_k$, and the rigid-mode amplification $\lambda_7/|\sigma| \le 100\,\lambda_7/\lambda_k$ is harmless (measured: none visible).
   - If $\tilde\lambda_k \le 100|\sigma_0|$ (every wanted mode near zero), pass 1 stands.
   - An explicit `sigma` skips the rule and gives a single pass. Use it beyond the validated range: $\kappa \gtrsim 10^{11}$ (PR 16's solid shells at $\kappa \approx 10^{12}$), with $\sigma$ from a coarser run. The certification below still guards the result.
6. **Certification** (always; `RuntimeError` on failure). With the computed $\lambda_1 \le \ldots \le \lambda_{n+1}$ ($n$ = `n_modes`):
   - **Pairs:** $\|V^\top M_f V - I\|_\max \le 10^{-10}$. Every per-mode residual satisfies $\|(K_f v - \lambda M_f v)\|_\infty \le 100\,\varepsilon\,(\|K_f\|_\infty + |\lambda|\,\|M_f\|_\infty)\|v\|_\infty$; the measured maximum is $10.65\varepsilon$. This costs $k$ matvecs and rules out a ghost copy standing in for a missing mode: with accurate, M-orthonormal eigenvectors, a count equal to $j$ proves that the first $j$ vectors span the lowest invariant subspace.
   - **Count:** let $j$ be the largest index $\le n$ with $\lambda_{j+1} - \lambda_j > 10^{-6}\,|\lambda_{n+1}|$, the last gap at or below the requested count. Rigid-mode noise and exact FE degeneracies, about $10^{-12}$ relative, count as clusters; real gaps are far wider. Factor $K_f - \mu M_f$ at $\mu = (\lambda_j + \lambda_{j+1})/2$. If `perm_r != perm_c`, raise: the count is unavailable. Otherwise $c$ = the number of negative entries of $\mathrm{diag}\,U$, and $c \ne j$ raises `RuntimeError("eigsh missed modes: c eigenvalues below μ, j found")`.
   - If no gap exists ($j$ undefined, e.g. `n_modes=3` on a free body), raise `ValueError` asking for more modes.
   - When `n_modes` ends inside a cluster, only modes $1..j$ are certified by the count (documented). The tail cluster's members are still certified as eigenpairs by the residuals, and a missed member is caught if $\lambda_{n+1}$ lies past the gap.
7. **Output:** `lam[:n_modes]`, and `V` with `V[idx] = v[:, :n_modes]`.

**Cost and size.** 2 factorizations for SPD $K_f$, 3 with rigid modes (pass 1, pass 2, the count). Pass 1, pass 2 and the check use different shifts, so nothing can be shared. With `points`, the largest supported size for bulky meshes is about 100k free DOF: ≈ 26 s and 2.9 GB per factorization (ND table). The routine peaks higher, because the count's `lu.U` builds a full CSC copy of $U$ next to the factor (SciPy has no pivot accessor): measured end to end at hex20 $20^3$, 62 s and 5.9 GB for SPD $K_f$ (99.5k DOF, 2 factorizations), 124 s and 7.1 GB with rigid modes (107k DOF, 3); without `points` (MMD) the bulky limit stays about 35k. Plate-like meshes go further. The implementation re-measures the whole routine at the sizes of the table (harness) and states the routine peak (not only the per-factorization peak) and time, including the check's share, in `docs/eigensolvers.md` and the roadmap.

### `count_below(K, M, sigma, free=None) -> int`

The exact number of eigenvalues of $(K_f, M_f)$ below $\sigma$ (any finite float), by Sylvester's law of inertia on $A = K_f - \sigma M_f$ ($K_f$, $M_f$ symmetric; $M_f$ positive semidefinite):

1. **Size guard.** `restrict`, then `n_free > MAX_DENSE = 20_000` raises `ValueError`: 3.2 GB, 16 s measured. The roadmap's "≤ ~15k" becomes "≤ 20k".
2. **Factorization.** `A = (K_f - sigma * M_f).toarray(order="F")`, then `dsytrf(A, lower=1, lwork=<queried via dsytrf_lwork>, overwrite_a=1)`: in place, one dense copy. `lower=1` reads one triangle, so round-off asymmetry does not matter.
3. **Counting.** `ipiv < 0` marks the rows of $2\times2$ pivot blocks. A Bunch–Kaufman $2\times2$ pivot is chosen only when $|a_{kk}|\,\sigma_r < \alpha\lambda^2$ and $|a_{rr}| < \alpha\sigma_r$ ($\alpha = (1 + \sqrt{17})/8$), on the updated entries that form the stored block. So $|a_{kk}a_{rr}| < \alpha^2\lambda^2 < \lambda^2$ and $\det < 0$: exactly one negative eigenvalue per block. The count is the number of negative $1\times1$ pivots plus `(ipiv < 0).sum() // 2`, excluding the diagonal entries of $2\times2$ blocks. This is commented in the code and tested against `eigh`.
4. **Errors.** `info > 0` (an exactly zero pivot: $\sigma$ is an eigenvalue to working precision) raises `ValueError`.
5. **Choice of routine.** The roadmap names `scipy.linalg.ldl`, which calls the same LAPACK routine but returns three $n^2$ arrays. The roadmap is updated to say `sytrf`.

**Two inertia implementations, on purpose.** `count_below` is dense and pivoted (Bunch–Kaufman is backward stable), so it is exact for well-separated $\sigma$: the roadmap's exact count at reference sizes, used by PR 9 and PR 14. The check inside `reference_modes` is sparse and unpivoted; it reuses the solver's factorization options and runs up to about 100k DOF. A one-line comment in each points to `docs/eigensolvers.md`.

**Sizes.** "Reference size" (roadmap: PR 9's dense $r_M$ and inertia comparisons, PR 14's exact count) means $\le$ `MAX_DENSE` = 20k free DOF, where dense `eigh`, $M^{-1}$ and `count_below` are affordable. `reference_modes` alone, "the PR 8 `eigsh`", reaches about 100k bulky with `points` (about 35k without) (used by PRs 11, 13, 16, 18). The roadmap and `docs/eigensolvers.md` define both terms in one place.

## In-test oracle: the sliding block (`tests/exact.py`)

A box $[0, L_x]\times[0, L_y]\times[0, L_z]$ with sliding (roller) BCs on all six faces: $u_x = 0$ on $x = 0, L_x$ (node sets `xmin` and `xmax` with `"x"`), likewise for y and z. The tangential tractions there vanish. For an orthotropic material with axes aligned to the box (Voigt $C_{44}$: 23, $C_{55}$: 13, $C_{66}$: 12) and $k = \pi(p/L_x, q/L_y, r/L_z)$, the ansatz

$$u_x = A \sin k_x x \cos k_y y \cos k_z z,\quad u_y = B \cos k_x x \sin k_y y \cos k_z z,\quad u_z = C \cos k_x x \cos k_y y \sin k_z z$$

satisfies the BCs ($\sin(p\pi) = 0$; each term of the tangential tractions carries the face's sine factor). It reduces the Navier equations to $\Gamma(k)\,(A, B, C)^\top = \rho\lambda\,(A, B, C)^\top$, with

$$\Gamma = \begin{pmatrix} C_{11}k_x^2 + C_{66}k_y^2 + C_{55}k_z^2 & (C_{12} + C_{66})k_xk_y & (C_{13} + C_{55})k_xk_z \\ \cdot & C_{66}k_x^2 + C_{22}k_y^2 + C_{44}k_z^2 & (C_{23} + C_{44})k_yk_z \\ \cdot & \cdot & C_{55}k_x^2 + C_{44}k_y^2 + C_{33}k_z^2 \end{pmatrix}.$$

- **Component rule.** Component $i$ exists only when its index ($p$, $q$ or $r$) is $\ge 1$. Each $(p, q, r) \in \mathbb{N}^3 \setminus \{0\}$ spans an invariant block and contributes the eigenvalues of $\Gamma$ restricted to its existing components.
- **Completeness.** The cosine and sine families are complete on each axis, so the union of the block spectra is the whole spectrum. There are no rigid modes. Flipping ansatz signs is a $\mathrm{diag}(\pm1)$ similarity, so the eigenvalues do not change.
- **`kmax` guard.** `sliding_block(C, rho, L, n, kmax=8)` returns the lowest `n` eigenvalues, sorted, with a provable check on `kmax`:
  - With $\varepsilon = \mathrm{sym}(v \otimes \hat k)$, $\|\varepsilon\|^2 = \tfrac12(|v|^2 + (v\cdot\hat k)^2) \ge \tfrac12|v|^2$, and the engineering Voigt vector $e$ of $\varepsilon$ has $|e|^2 \ge \|\varepsilon\|^2$. So $v^\top\Gamma(\hat k)v = e^\top C e \ge \lambda_\min(C)\,|v|^2/2$.
  - By interlacing, every block eigenvalue is $\ge \lambda_\min(C)|k|^2/(2\rho)$. Every mode with an index above `kmax` therefore has $\lambda \ge \lambda_\min(C)\,\pi^2(k_\max + 1)^2 / (2\rho L_\max^2)$.
  - The `n`-th returned value must lie below this bound, else `ValueError`.

  (A monotonicity argument would be wrong: going from index 0 to 1 adds a component and can lower a block minimum.)
- **Self-check (isotropic):**
  - for $p, q, r \ge 1$: $(\lambda_L + 2\mu)|k|^2/\rho$ and $\mu|k|^2/\rho$ (twice);
  - with exactly one index 0: $\mu|k|^2/\rho$ and $(\lambda_L + 2\mu)|k|^2/\rho$;
  - with two indices 0: only $(\lambda_L + 2\mu)|k|^2/\rho$.

  These are tested against the closed form.

## Tests

References: dense `scipy.linalg.eigh` of $(K_f, M_f)$, the exact sliding-block spectrum, and the PR 4 helpers. $\lambda_\max$ comes from dense `eigh` at test sizes, or from `eigsh(..., which="LA")` for the convergence meshes.

### `tests/test_reference.py`

1. **Against dense `eigh`**, `n_modes = 20`, all with `points` (plus one case without `points`, which must match the ND run to $2 f_\lambda$; `_order` returns a permutation, `np.array_equal(np.sort(p), np.arange(n))`):
   - **Cases:**
     - a free hex20 box with the orthotropic material, consistent and lumped, with nodal masses including zeros; this is the case that kills "pass 2 skipped";
     - a clamped curved hex27 (PR 5's mesh B, `xmin: xyz`);
     - the mixed hex8 + hex20 mesh with `xmin: x`;
     - a free hex27 plate $8\times8\times1$ with $L/t = 200$ ($\kappa = 7.8\cdot10^9$, $\lambda_7/\bar s = 1.2\cdot10^{-9}$; dense `eigh` 1.7 s).
   - **Criteria:**
     - rigid modes: a fixed count (6, or 3 for `xmin: x`), $|\lambda| \le 10\,\varepsilon\,\lambda_\max$;
     - elastic modes: $|\lambda - \lambda_\text{dense}| \le 2 f_\lambda\,\lambda$;
     - $\|V^\top M V - I\| \le 10^{-12}$;
     - constrained rows exactly 0;
     - residuals on the free rows with $K_f$, $M_f$ at $100\varepsilon$, as in step 6;
     - `sigma = -0.01 * lam_dense[n]` gives the same $\lambda$ to $2 f_\lambda$.
2. **Certification.** The count part is tested two ways.
   - **Deterministic:** `monkeypatch` `reference._solve` to drop one eigenpair below the gap (return modes $1..j-1, j+1..n+2$). The `RuntimeError` names the counts.
   - **Real ARPACK:** on the $8\times8\times1$ plate at `sigma = -1e-2 * s̄`, assert that ARPACK's raw output is wrong (computed with `_solve` directly), then that `reference_modes` raises. `pytest.skip` if a future ARPACK gets it right; this test records the behaviour, it does not depend on it.

   The pair part, each check pinned by its message: `_solve` patched to replace mode $j$ by a copy of mode $j-1$ (a ghost: count and residuals pass, measured 4.1ε) raises the orthonormality error; `_solve` patched to perturb one eigenvector by $10^{-6}$ times another (orthonormality $10^{-12}$ and count pass) raises the residual error. `splu` patched to return `perm_r != perm_c` raises.
3. **Clusters.** On a free isotropic hex20 $4^3$ cube, take the first 4 `n_modes` values (from dense `eigh` at test time; currently 9, 10, 12, 13) that end inside an exact FE multiplicity, asserting that the multiplicity is real (relative gap $< 10^{-10}$). Each gives no error, and the modes up to the last gap match dense `eigh`. Under the mutant "gap threshold → 0", at least one of the four miscounts. `n_modes = 3` on the free box (inside the rigid cluster) raises `ValueError`.
4. **Inputs.** `n_modes = 0`, `n_modes + 1 >= n_free`, a bad mask, `sigma` of 0 or positive, mismatched shapes, and a wrong `points` shape raise `ValueError`. `restrict` returns `idx` and the restricted matrices (checked against dense slicing).
5. **`count_below`** (the roadmap's test). On a clamped model (hex27, `xmin: xyz`) and a free model (hex20 box), consistent and lumped: for $\sigma$ mid-gap between at least 4 consecutive pairs with relative gap $\ge 10^{-3}$, including the first gap above the rigid cluster of the free model, the count equals the dense `eigh` count below $\sigma$. Every one of these cases has $2\times2$ pivots (4–16 rows), so the block-counting mutants die. `n_free > MAX_DENSE` raises; this is checked by lowering the module constant with `monkeypatch`.

### `tests/test_verification.py` (sliding block)

- **Mesh and BCs.** `box_mesh((n, n, n), spacing=L/n, type=t)`, with the six sliding sets via `dof_mask`, and `points=mesh.points`. 6 modes for rates; 10 for the 0.1% target.
- **Comparison.** Modes are compared sorted against the exact sorted spectrum (justified above).

6. **Oracle self-check:** the isotropic closed form, and the `kmax` guard raising when `kmax` is too small for `n`.
7. **0.1% target** (roadmap): consistent hex20 and hex27 at $n = 8$ (6.7k / 13.0k free DOF for hex20 / hex27, about 2 s each), both materials, lowest 10 modes: $|f/f_\text{exact} - 1| \le 10^{-3}$ (measured $\le 3.3\cdot10^{-4}$: a 3× margin; $n = 6$ fails for the isotropic material). The mesh is stated in `docs/verification.md`.
8. **Convergence rates** (`slow`). Least-squares slope of $\log|\lambda^h/\lambda - 1|$ against $\log h$, per mode, for the lowest 6 modes, over the levels of the table, both materials:
   - $\ge 1.8$ for hex8 consistent and row-sum, hex20 HRZ and hex27 row-sum;
   - $\ge 3.6$ for hex20 and hex27 consistent.
   - The finest error must be $\ge 100 f_\lambda$.
   - Richardson extrapolation is not used.
9. **Upper bound** (consistent mass; roadmap). At every level, $\lambda^h_k \ge \lambda_k^\text{exact} - 10\,\varepsilon\,\hat\lambda_\max$. On nested meshes (hex8 $6 \to 12$ and $8 \to 16$; hex20 and hex27 $3 \to 6$ and $4 \to 8$), $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\,\varepsilon\,\hat\lambda_\max$. Lumped mass converges from below; this is documented, not tested (no theorem).

Expected runtime: test_reference ≈ 10 s, the 0.1% test ≈ 8 s, the slow rate tests ≈ 40 s with ND.

**Mutants expected to be killed:**
- **σ rule:** pass 2 skipped (the box);
- **certification:** check removed, $\mu$ off by one gap, gap threshold $\to 0$ (test 3's loop), residual check removed (test 2's perturbed vector), orthonormality check removed (test 2's ghost);
- **restriction:** masking instead of restriction (spurious zeros);
- **output:** a wrong column order (residuals);
- **inertia:** $2\times2$ blocks counted as 0 or 2, raw diagonal signs;
- **ND ordering:** a separator left out of the permutation (`p` not a permutation, which the solve residual catches);
- **oracle:** a wrong $\Gamma$ entry ($C_{55} \leftrightarrow C_{66}$, killed by the orthotropic material), the component rule removed.

**Equivalent mutants:**
- the σ multiplier 0.01 → 0.1 or 1 (measured: same accuracy);
- `COLAMD` instead of `MMD_AT_PLUS_A`;
- a different `v0` seed;
- leaf size 64 → 32, ND replaced by `arange` or MMD, an empty separator (cost only; `p` stays a permutation);
- $C_{55} \leftrightarrow C_{66}$ on the isotropic material.

## Docs

- **`docs/eigensolvers.md`** (new). PR 9 adds LOBPCG to it. For now it covers:
  - the reference solver: restriction vs masking, and that PR 9 compares against these modes padded with zeros;
  - shift-invert at a negative shift: why negative, the measured silent failure, the σ rule with pass 2, the explicit `sigma` beyond $\kappa \approx 10^{11}$ (taken near $\lambda_k$, e.g. from a coarser run: an explicit $\sigma$ far below $\lambda_k$ raises the residual to 45–90ε on the plates at $-10^{-2}\bar s$ and can trip the 100ε check, loudly);
  - certification: residuals, orthonormality, Sylvester's law, symmetric SuperLU, the last-gap rule and what it certifies, and the caveat that the sparse count is unpivoted (growth uncontrolled; `perm_r == perm_c` guards the structure, not the growth; failures are loud);
  - geometric nested dissection and the MMD fallback;
  - the accuracy floor $\varepsilon\kappa$ with the measured errors;
  - cost, and the largest supported size (re-measured table, with the check's share);
  - the definitions of "reference size" and "the PR 8 `eigsh`";
  - the dense inertia count: Sylvester, Bunch–Kaufman, why the $2\times2$ blocks are indefinite, the Fortran-order memory, `MAX_DENSE`;
  - why there are two inertia implementations.
- **`docs/verification.md`** (new):
  - error measures (relative in $f$ for targets, in $\lambda$ for rates) and how modes are matched;
  - the sliding block: setup, ansatz, $\Gamma$, completeness, the `kmax` bound, the stated meshes ($n = 8$ for the 0.1% target), measured errors and rates, the upper bound, lumped from below;
  - sections for 8b and 8c named but empty.
- **`docs/boundary_conditions.md`:** `restrict`, next to `masked` and `a_p`.
- **`docs/ROADMAP.md`:**
  - the status sentence for 8a;
  - the split note under PR 8 (8a/8b/8c; acceptance criteria unchanged; PR 9 strictly needs only 8a; the dependency row and critical path unchanged);
  - "Dense inertia via `scipy.linalg.ldl` (≤ ~15k DOF)" in PR 8 Contents and its test, and the "Exact count" key decision: both become `sytrf` (which `ldl` calls; one dense copy) and ≤ 20k;
  - the definition of reference size vs the PR 8 `eigsh` limit, in the PR 8 contents;
  - PR 16's "simply supported BCs as defined in PR 8" becomes "PR 8b".
- **`README.md`:** the status, and a short reference-solver snippet after the rigid-modes snippet (run by `test_readme`), ending with a pointer to `docs/eigensolvers.md` and `docs/verification.md`.

## Not covered

- The other oracles and benchmarks: 8b (Visscher, Srinivas–Rao, MAC pairing) and 8c (NAFEMS, Lamb sphere).
- A sparse direct solver beyond SuperLU: SciPy has none, and the roadmap excludes CHOLMOD and pypardiso. Geometric ND is the cheap win.
- Validation at $\kappa \gtrsim 10^{11}$: PR 16 validates `reference_modes` at its own $\kappa$, with an explicit `sigma`.
- Interior shift-invert slices with $\sigma > 0$ (PR 15 has its own Lanczos).
