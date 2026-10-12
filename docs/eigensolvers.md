# Eigensolvers

`jaxmodal.eigen.reference_modes` (module `eigen/reference.py`) is the SciPy reference eigensolver on assembled matrices; `jaxmodal.linalg.count_below` (module `linalg/inertia.py`) is the exact dense inertia count (PR 8a). They are oracles for the matrix-free LOBPCG and for later PRs; both run on the host on the SciPy matrices from `assemble`. `jaxmodal.eigen.lobpcg` (module `eigen/lobpcg.py`, PR 9a) is the matrix-free solver, with `jacobi` and `sigma_estimate`; see [LOBPCG](#lobpcg).

```python
from jaxmodal.eigen import reference_modes
from jaxmodal.linalg import count_below
from jaxmodal.operators import assemble, dof_mask

K, M = assemble(mesh, materials)
free = dof_mask(mesh, bcs)
lam, V = reference_modes(K, M, 20, free, points=mesh.points)  # lam (20,) ascending, V (n_dof, 20)
n = count_below(K, M, sigma, free)  # number of eigenvalues of (K_ff, M_ff) below sigma
```

## Reference solver (`reference_modes`)

`reference_modes(K, M, n_modes, free=None, sigma=None, points=None)` returns the lowest `n_modes` eigenpairs of $K_f v = \lambda M_f v$, where $K_f$, $M_f$ are the free blocks: $\lambda$ ascending, and $V$ of shape `(n_dof, n_modes)` with constrained rows exactly 0 and $V^\top M V = I$. PR 9 compares its masked LOBPCG modes against these padded modes.

- **Restriction, not masking** (`restrict`, [`boundary_conditions.md`](boundary_conditions.md)): the masked operators keep the constrained DOFs as zero rows, which would add spurious zero eigenvalues and spurious zero pivots.
- **Inputs.** `1 <= n_modes` and `n_modes + 1 < n_free` (ARPACK needs $k < n$, and one extra mode is computed for the check); `sigma` is `None` or a finite value $< 0$; `points` is `None` or the `(n_nodes, 3)` `mesh.points` that `K`, `M` were assembled from (DOF $3n + i$). Anything else raises `ValueError`. Points of another mesh of the same size cannot be detected; they only slow the factorization and never change the result.

### Shift-invert at a negative shift

`eigsh(K_f, n_modes + 1, M=M_f, sigma=σ, OPinv=…, tol=0, v0=…)` with $\sigma < 0$, so $K_f - \sigma M_f$ is SPD and its factorization needs no pivoting. The start vector is seeded (`default_rng(0)`), so runs are reproducible.

**ARPACK fails silently when $|\sigma|$ is far above the wanted eigenvalues.** Shift-invert converges at a ratio of about $(\lambda_{k+1} - \lambda_k)/|\sigma|$ when $|\sigma| \gg \lambda_k$. In that regime `eigsh` can return wrong eigenpairs without raising: higher modes in their place, or missing rigid modes. Measured, with $\bar s = \mathrm{mean}\,\mathrm{diag} K_f / \mathrm{mean}\,\mathrm{diag} M_f$:

- a free consistent hex20 box at $\sigma = -\bar s$ (about $160\lambda_7$): relative errors of 1.3, rigid modes missing;
- a free hex20 plate with $L/t = 100$ at $\sigma = -10^{-2}\bar s$: errors of 2.1;
- the free hex27 plate $8\times8\times1$ with $L/t = 200$ at $\sigma = -10^{-2}\bar s$ ($8\cdot10^6\lambda_7$): wrong pairs after 40 s (`test_arpack_failure` records it). At $-10^{-3}\bar s$ and $-10^{-4}\bar s$ the same plate comes out right.

On thin plates $\lambda_7/\bar s$ is tiny ($2\cdot10^{-8}$ at $L/t = 100$, $1.2\cdot10^{-9}$ at $L/t = 200$; a free hex20 box has $6\cdot10^{-3}$), so no fixed fraction of $\bar s$ works. Too close to 0 fails too on a free body: $K_f - \sigma M_f$ is nearly singular along the rigid modes, and the elastic eigenvalues lose about $\varepsilon\lambda_7/|\sigma|$.

**σ rule** (`sigma=None`):

1. **Pass 1** at $\sigma_0 = -\sqrt\varepsilon\,\bar s$ (unit-independent). It needs only rough $\tilde\lambda$, and works while $|\sigma_0|$ is not far above $\lambda_k$ ($k$ = `n_modes + 1`); measured on plates down to $\lambda_7/\bar s = 10^{-9}$ ($L/t = 1000$).
2. **Pass 2**, if pass 1 finds near-zero eigenvalues ($\tilde\lambda_1 < 100|\sigma_0|$: rigid modes, a nearly singular $K_f$, or a thin structure whose low modes lie below $100|\sigma_0|$), at $\sigma = -0.3\,\tilde\lambda_k$. Then $|\sigma| < \lambda_k$, and the rigid-mode amplification is at most $\lambda_k/|\sigma| \approx 3$ (see below). Only if every wanted mode is rigid-mode noise ($\tilde\lambda_k \le 100\,\varepsilon\,\bar s$, which then raises `ValueError` "no gap") does pass 1 stand; on very thin structures that request can instead fail loudly (`ArpackNoConvergence`, or the residual check).

For SPD $K_f$ (clamped, sliding), pass 1 is as accurate as a pass near $-\lambda_k$.

**Why $-0.3\tilde\lambda_k$ (PR 8c).** In shift-invert the shifted rigid modes dominate the operator ($1/|\sigma|$ against $1/(\lambda - \sigma)$), and ARPACK's accuracy is relative to the largest Ritz value, so the elastic residuals grow by about $\lambda/|\sigma|$. PR 8a used $-0.01\tilde\lambda_k$ (a factor of about 100), which passed its small tests but failed on the free voxel sphere with 32 cells per diameter (59.4k DOF, `n_modes = 18`): residual 172ε against the 100ε bound, so `reference_modes` raised. The failure depends on size and $k$ (spheres with 12, 16, 20, 32 cells: 24, 14–19, 49, 172ε at 18 modes; 50ε at 32 cells with 22 modes). Pass-2 factors 0.01, 0.1, 0.3 and 1 give 172, 8.0, 3.6 and 3.7ε on that sphere (one pass-2 solve takes 12.0–13.1 s, about 4% more at 0.3 than at 0.01; the whole call with certification about 34 s), and 5.5, 1.5, 1.1, 1.5ε on a free hex20 $12^3$ cube. 0.3 keeps $|\sigma| < \lambda_k$, away from the opposite failure below (σ far below $\lambda_k$).

The rule is validated against dense `eigh` up to $\kappa = \lambda_\max/\lambda_7 \approx 8\cdot10^9$, and certifies on the free hex27 plate $8\times8\times1$ with spacing $(1, 1, 0.005)$ ($L/t = 1600$, $\kappa = 3\cdot10^{13}$) for `n_modes` 7, 10, 12, 17, 21 and 25; there the floor $f_\lambda = 10\,\varepsilon\,\kappa \approx 0.07$ for the lowest elastic mode, and the measured error against dense `eigh` is $0.09 f_\lambda$ (lowest 12; the same under PR 8a's $-0.01$ and the current $-0.3$). (Without the thin-structure case of pass 2, pass 1 at $|\sigma_0| \approx 10^5\lambda_k$ fails the residual check there, at 700–7800ε.)

**Explicit `sigma`** skips the rule (one pass), e.g. $-0.3\lambda_k$ from a coarser run, for PR 16's solid shells ($\kappa \approx 10^{12}$) if the rule ever fails there. A $\sigma$ far below $\lambda_k$ raises the residuals (45–90ε on the $L/t = 200$ plates at $-10^{-2}\bar s$) and can trip the residual check, loudly.

### Certification

Every result is checked; a failure raises `RuntimeError` instead of returning wrong modes. With the computed $\lambda_1 \le \ldots \le \lambda_{n+1}$ ($n$ = `n_modes`):

1. **Orthonormality:** $\|V^\top M_f V - I\|_\max \le 10^{-10}$.
2. **Residuals:** every mode has $\|K_f v - \lambda M_f v\|_\infty \le 100\,\varepsilon\,(\|K_f\|_\infty + |\lambda|\,\|M_f\|_\infty)\,\|v\|_\infty$ (measured maximum $10.65\varepsilon$). With accurate, M-orthonormal eigenvectors, a count equal to $j$ proves that the first $j$ vectors span the lowest invariant subspace; together, 1 and 2 rule out a ghost copy standing in for a missing mode.
3. **Count.** Let $j$ be the largest index $\le n$ with $\lambda_{j+1} - \lambda_j > \tau = \max(10^{-6}|\lambda_{n+1}|,\ 100\,\varepsilon\,\bar s)$: the last gap at or below the requested count. Rigid-mode noise ($\lesssim 4\varepsilon\bar s$), exact FE degeneracies and the splitting of degenerate pairs by `eigsh` (up to $2.6\cdot10^{-5}$ relative, $\le 1.7\varepsilon\bar s$ absolute, on very thin square plates) count as clusters; real gaps are far wider. $K_f - \mu M_f$ at $\mu = (\lambda_j + \lambda_{j+1})/2$ is factored, and by Sylvester's law of inertia the number $c$ of negative pivots is the number of eigenvalues below $\mu$; $c \ne j$ raises `eigsh missed modes: c eigenvalues below μ, j found`. No gap at all (e.g. `n_modes=3` on a free body, inside the rigid cluster) raises `ValueError` asking for more modes.

A request that ends inside a cluster of near-singular soft modes (soft springs, near-mechanisms, $\lambda \lesssim 10^{-9}\bar s$) can fail the residual check, loudly; ask for modes past the cluster.

When `n_modes` ends inside a cluster, the count certifies modes $1..j$ only; the tail cluster's members are still certified as eigenpairs by the residuals, and a missed member is caught if $\lambda_{n+1}$ lies past the gap.

**Sparse inertia from SuperLU.** The count reuses the solver's factorization: `splu(A[p][:, p], permc_spec=…, diag_pivot_thresh=0, options=dict(SymmetricMode=True))` keeps `perm_r == perm_c`, so $PAP^\top = LU$ with $U = DL^\top$, and $c$ = the number of negative entries of $\mathrm{diag}\,U$. If SuperLU ever pivots off the diagonal (`perm_r != perm_c`), the count is unavailable and the routine raises. This is an unpivoted $LDL^\top$ of an indefinite matrix: element growth is not controlled, and `perm_r == perm_c` guards the structure, not the growth. A wrong count is possible in principle and would almost always show up as a loud mismatch. It matched dense `eigh` in every case tried (free box, clamped hex27, plates; consistent and lumped; both orderings; up to 747 negative pivots).

### Ordering

- **With `points`: geometric nested dissection.** The free DOFs are bisected recursively at the median of the widest coordinate axis (`c <= median`, which keeps whole node planes on one side so the separator is one clean layer; if all DOFs tie at or below the median, `c < median`; if that is empty too, the subset becomes a leaf). The separator is the left DOFs coupled to the right side, read from the CSR rows of $|K_f|$. Children are ordered first, then the separator, down to leaves of `ND_LEAF = 64` DOFs; SuperLU then factors with `permc_spec="NATURAL"`. The ordering takes 0.04–0.15 s. A rank split (equal halves) cuts through node planes and costs 30–65% more factorization time.
- **Without `points`:** SuperLU's `MMD_AT_PLUS_A`, fine for plates, slower and denser for bulky meshes.

One factorization of the prototype's boxes (harness, SuperLU fill or peak; the DOF counts differ slightly from the cost table below, which uses other BCs):

| mesh | DOF (prototype) | `MMD_AT_PLUS_A` | geometric ND |
|---|---|---|---|
| hex20 $12^3$ | 22k | 4.7 s, 50M fill | 1.4 s, 28M |
| hex20 $14^3$ | 34.6k | 14.9 s, 106M | 4.7 s, 55M |
| hex8 $24^3$ | 43k | 15.8 s, 117M | 3.4 s, 54M |
| hex20 $16^3$ | 51k | 34.5 s, 2.12 GB | 7.2 s, 1.15 GB |
| hex8 $30^3$ | 84k | 118 s, 5.53 GB | 11.6 s, 1.57 GB |
| hex20 $20^3$ | ≈ 100k | — | 26.2 s, 2.89 GB |
| hex27 plate $40\times40\times1$ | 59k | 1.5 s, 34M | 1.7 s, 46M |

### Accuracy

The attainable accuracy is the floor $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$ ($\hat\lambda_\max$: the largest eigenvalue, from dense `eigh` or `eigsh(which="LA")`; the verification tests use `tol=1e-3`, enough for an order of magnitude) (relative), i.e. about $\varepsilon\kappa$ for the lowest elastic mode. Against dense `scipy.linalg.eigh` of $(K_f, M_f)$, up to 20 modes on free hex20 boxes (consistent and lumped), clamped hex27, sliding hex20, and free hex20/hex27 plates with $L/t = 100, 200$ ($\kappa \le 7.8\cdot10^9$):

- elastic modes: $|\lambda - \lambda_\text{dense}| \le 0.99 f_\lambda\lambda$ (each side of the comparison is floor-limited);
- rigid modes: $|\lambda| \le 0.04\cdot10\varepsilon\lambda_\max$ (dense `eigh`: up to $0.22$);
- $\|V^\top M V - I\| \le 8\cdot10^{-15}$;
- residuals $1.9$–$4.9\varepsilon$ on boxes, $4.3$–$10.65\varepsilon$ on plates (relative as in the check; on the full $K$ the constrained rows carry reaction forces).

The tests (`tests/test_reference.py`) assert $2 f_\lambda$, $10^{-12}$ and $100\varepsilon$.

### Cost and size

2 factorizations for SPD $K_f$ (one solve and the count), 3 with rigid modes (pass 1, pass 2, the count); the shifts differ, so nothing is shared. The routine peaks above one factorization, because the count's `lu.U` builds a full CSC copy of $U$ next to the factor (SciPy has no pivot accessor). The whole routine, `n_modes = 20`, orthotropic material, measured with the benchmark harness (peak over the setup):

| mesh, BCs | ordering | free DOF | time | of which the check | peak |
|---|---|---|---|---|---|
| hex20 $10^3$, free | ND | 14.9k | 3.3 s | 0.7 s | 0.42 GB |
| hex20 $16^3$, clamped | ND | 53.9k | 19.7 s | 8.4 s | 2.57 GB |
| hex20 $16^3$, free | ND | 56.4k | 36.3 s | 9.8 s | 2.73 GB |
| hex8 $30^3$, clamped | ND | 86.5k | 28.2 s | 12.2 s | 3.39 GB |
| hex8 $30^3$, free | ND | 89.4k | 49.9 s | 13.6 s | 3.56 GB |
| hex20 $20^3$, clamped | ND | 103k | 67.7 s | 30.4 s | 6.48 GB |
| hex20 $20^3$, free | ND | 107k | 120 s | 34.5 s | 6.98 GB |
| hex20 $14^3$, clamped | MMD | 36.5k | 29.6 s | 13.5 s | 2.49 GB |
| hex20 $14^3$, free | MMD | 38.5k | 53.6 s | 15.8 s | 2.78 GB |
| hex27 plate $40\times40\times1$, $L/t = 200$, free | ND | 59.0k | 7.7 s | 1.9 s | 1.26 GB |
| the same plate | MMD | 59.0k | 5.7 s | 1.4 s | 0.90 GB |

The factorizations dominate; the check is about 20–45% of the time. The largest supported size is about 100k free DOF on bulky meshes with `points` (2 minutes, 7 GB), and about 35k without. Plate-like meshes factor much faster and go further; on the plate MMD is as good as ND.

## LOBPCG

```python
from jaxmodal.eigen import jacobi, lobpcg, sigma_estimate

lam, V, info = lobpcg(op, free, 20, tol=1e-8, R=R)  # R: M-orthonormal rigid modes or None
lam, V, info = lobpcg(op, free, 20, precond=jacobi(op, free, sigma))  # any pytree callable
sigma, its = sigma_estimate(op, free, R)  # lam_1 upper bound, for preconditioners of K + sigma M
```

`lobpcg(op, free, n_modes, tol=1e-6, R=None, precond=None, X0=None, maxit=500, seed=0)` computes the lowest `n_modes` eigenpairs of the masked pencil $K_m v = m \odot K(m \odot v)$, $M_m$ likewise, for any operator with `apply_K`, `apply_M`, `diag_K`, `diag_M` and `n_dof` (PR 5 or PR 6).

- **Returns:** `lam` `(n_modes,)` ascending (NumPy); `V` `(n_dof, n_modes)` on the device, M-orthonormal with constrained rows exactly 0; and `info` with `iterations`, `converged`, `stagnated`, `residuals` and `thresholds` (from explicitly recomputed products, on every exit), `p_drops`, `refills`, `norm_K`, `norm_M`, `history` (max wanted residual per iteration) and `time` (`setup`, `iterations`, in seconds).
- **Deflation basis `R`:** `(n_dof, k)`, M-orthonormal (rigid modes after `m_orthonormalize` with the same operator). $MR = M_m R$ is computed once. The start block and every new direction are projected, $W \leftarrow W - R\,(MR^\top W)$, so no extra M apply is needed per iteration. Without `R`, a free body returns its rigid modes as λ ≈ 0; their λ-scaled residual is meaningless, so the solve ends at the no-progress exit (`converged` False, `stagnated` True).
- **Preconditioner `precond`:** a pytree callable `(n, j) -> (n, j)` (e.g. `jax.tree_util.Partial`), fixed, linear and SPD on the free DOFs, with masked output. It is a step argument, so new preconditioner data does not recompile. The default is `jacobi(op, free)`.
- **Block size** $m = n_\mathrm{modes} + \max(5, \lfloor 0.2\,n_\mathrm{modes}\rfloor)$.
- **Errors:**
  - `ValueError` if `n_modes` or `maxit` is below 1, if $m$ exceeds the free DOFs minus the rank of $R$, if $\|R^\top M_mR - I\|_\max > 10^{-8}$ (e.g. R orthonormalized with another mass), or for a bad mask or `X0`;
  - `RuntimeError` if the start block stays rank-deficient after 3 refills;
  - `FloatingPointError` for a non-finite step.

  Non-convergence never raises: check `info`.
- **`jacobi(op, free, sigma=0.0)`:** Jacobi on $A_P = K + \sigma M$, i.e. $v \mapsto m \odot v / (\mathrm{diag}K + \sigma\,\mathrm{diag}M)$, with output 0 on constrained DOFs. The constrained diagonal $s$ of $A_P$ ([`boundary_conditions.md`](boundary_conditions.md)) never reaches the output.

### Algorithm

Hetmaniuk–Lehoucq / Duersch et al. (2018), with fixed shapes: every block is $(n, m)$, and inactive columns are exact zeros tracked by boolean masks.

- **One compiled step.** `lobpcg_step(state, op, mask, precond, R, MR, consts, recompute)` is a module-level `jax.jit`.
  - The state $(X, KX, MX, P, KP, MP, \mathrm{act}_P, \lambda)$ is donated.
  - `consts` $= (\|K\|, \|M\|, \mathrm{tol}, c)$ and `recompute` are traced.
  - Nothing is closed over: closing over the operator would embed it in the program (6.4 MB of HLO for a diagonal operator at 100k DOF, against 0.05 MB with arguments; 2.2 GB more compile memory with a real operator at 182k) and recompile on every solve.
  - $P$ starts as a masked zero block, so there is one compiled variant per shape, and a second solve with new data (`op.scaled`, a new preconditioner, a new tol) compiles nothing.
- **Setup.**
  - $\|K_m\|_2$ and $\|M_m\|_2$ come from 30 power iterations each.
  - The start block (random, or `X0` padded with random columns) is masked, deflated and SVQB-orthonormalized. Columns SVQB drops (a rank-deficient `X0`) are refilled with random vectors, up to 3 times.
  - Rayleigh–Ritz on $X$ completes the setup.
- **SVQB.** Inactive rows and columns of $G = V^\top MV$ are zeroed, so they form the exact $\theta = 0$ eigenspace and kept directions carry nothing from them. The rest is scaled to unit diagonal, and directions with $\theta \le 10^{-14}\theta_\max$ are dropped (exact zero columns).
- **Each iteration:**
  1. Residuals give the active columns $a = r > t$. This is soft locking: converged columns get no new direction but stay in $X$.
  2. $W = T(R_X) \odot a$ is deflated, then $KW$ and $MW$ are applied. These are the only two width-$m$ applies per iteration.
  3. $W$ is M-orthogonalized against $[X, P]$, then SVQB, then orthogonalized again, then SVQB.
  4. Rayleigh–Ritz on $[X, W, P]$, with the Grams built pairwise from $m\times m$ products.
     - Masked rows and columns are zeroed off the diagonal, with $G_M = 1$ and $G_K = \beta = 10\max_i|\lambda_i| + \|K\|/\|M\|$ on it.
     - $X$ is fully active and M-orthonormal, so by Courant–Fischer the lowest $m$ active Ritz values are $\le \max_i\lambda_i < \beta$. Masked slots never enter, and no $\hat\lambda_\max$ estimate is needed.
     - **Pivot test:** if the Cholesky of $G_M$ is not finite or $\min\mathrm{diag}(L)^2 \le 10^{-10}$, the small problem is redone with the P block masked: a `jnp.where` between two small solutions. This bounds $\lambda_\min(G_M)$ only from above, not the condition number; in practice the pivots are about 1.
  5. The new P coefficients are built as in Duersch:
     - start from the Ritz coefficients with the $X$ rows zeroed;
     - $G_M$-orthogonalize them against $C_X$ twice, in the small space;
     - mask columns whose $G_M$-norm fell to $\le 10^{-8}$ of their norm before (the **noise rule**; without it a noise column would carry about $10^{-4}$ of non-orthogonality into $P$);
     - SVQB in coefficient space.

     `p_drops` counts the iterations in which the pivot fallback, the noise rule or this SVQB lost a P column. It can also fire under heavy soft locking, when the active W and P columns span fewer than $m$ directions; it is a diagnostic, not a failure.
     
     $[X', P']$ is then M-orthonormal without any n-sized orthogonalization of $P$ (measured: $\max|G_M - I| \le 7\cdot10^{-8}$).
  6. **Update.** $X' = XC_x + WC_w + PC_p$, and $P'$ likewise, with the same GEMMs for the K and M products. It is sequenced with `optimization_barrier`: XLA does not fuse a sum of GEMMs, and this saves 2 blocks.
  7. Every `RECOMPUTE` = 20 iterations, a traced `lax.cond` replaces the implicitly updated products by explicit applies.
- **Exit check.** When the wanted pairs meet their thresholds, $KX$ and $MX$ are recomputed explicitly and written back into the state. The solve exits only if the check still passes.

### Convergence

- **Residual:** the λ-scaled $r_i = \|K_mx_i - \lambda_iM_mx_i\| / (|\lambda_i|\,\|M_mx_i\|)$.
- **Threshold:** $t_i = \max(\mathrm{tol},\ c\,\varepsilon\,(\|K\| + |\lambda_i|\,\|M\|)\,\|x_i\| / (|\lambda_i|\,\|M_mx_i\|))$ with $c = 10$. It stays bounded as M becomes singular.
- **Progress** is a new minimum of the max wanted residual by a factor 0.9, or a decrease of the sum of the wanted Ritz values by more than a relative $10^{-10}$. Both clauses are needed: while λ̃ falls, the λ-scaled residual rises. On a 1-mode hex20 $6^3$ run, the residual alone had no new minimum for 85 iterations.

| exit | `converged` | `stagnated` |
|---|---|---|
| all wanted $r_i \le$ tol (explicitly confirmed) | True | False |
| all wanted $r_i \le t_i$, some above tol: accepted at the floor | True | True |
| no progress for 50 iterations | False | True |
| `maxit` reached | False | False |

**Theory.** For M-normalized $x$ and the M⁻¹-norm residual $r_M = \|Kx - \lambda Mx\|_{M^{-1}}/|\lambda|$, with the absolute gap to the rest of the spectrum:

$$\sin\theta \le \frac{r_M\,\lambda}{\mathrm{gap}}, \qquad \frac{|\delta\lambda|}{\lambda} \le \frac{r_M^2\,\lambda}{\mathrm{gap}}$$

These are Davis–Kahan and Kato–Temple in the M-inner product. The 2-norm $r$ is the cheap proxy; it equals $r_M$ up to the conditioning of M.

### σ pre-estimate

`sigma_estimate(op, free, R=None, maxit=300, seed=0)` returns `(sigma, iterations)`.
- It runs the same loop with 1 mode ($m = 6$) and Jacobi at σ = 0, and returns the Rayleigh quotient λ̃₁ ≥ λ₁.
- It stops when λ̃₁ has decreased by less than 5% over the last 10 iterations.
- The λ-scaled residual is no stop criterion here: it is small while λ̃ is huge (a stop at 0.5 gave 432× λ₁). A fixed count is no good either: 20 iterations give 4–98× λ₁.
- A free body needs `R`; without it σ is a rigid mode.

| mesh | DOF | iterations | λ̃₁/λ₁ |
|---|---|---|---|
| hex20 $8^3$ clamped | 8019 | 89 | 1.29 |
| hex20 $8^3$ free | 8019 | 102 | 1.04 |
| hex8 $16^3$ clamped | 14739 | 59 | 1.27 |
| hex27 $6^3$ clamped | 6591 | 76 | 1.01 |
| hex20 $12^3$ clamped | 24843 | 106 | 1.32 |
| hex8 $24^3$ free | 46875 | 82 | 1.02 |

The ratio grows slowly with size. That costs 60–110 iterations at $m = 6$, about 10–15% of a solve.

**The default Jacobi uses σ = 0.** For Jacobi, σ·diag M/diag K ≤ 4·10⁻⁴, and the iterations to $10^{-8}$ barely change: 330 / 338 / 334 for σ = 0 / λ₁ / λ₁₀ on hex20 $8^3$, and 231 / 231 / 229 on hex8 $16^3$. `lobpcg` therefore never runs the pre-estimate. It is meant for the preconditioners of K + σM in PRs 11–13, whose `solve_modes` dispatch runs it.

### Measured behaviour

All runs: isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$ on the box $(1, 0.83, 0.67)$, `xmin` clamped unless free, 12 modes, Jacobi, M3 Max CPU, warm.

| mesh | DOF | tol $10^{-6}$ | tol $10^{-10}$ | max rel. λ error |
|---|---|---|---|---|
| hex20 $6^3$ clamped | 3675 | 197 its | 320 its (1.5 s) | $1\cdot10^{-13}$ |
| hex20 $6^3$ free (rigid deflation) | 3675 | 209 | 346 (1.5 s) | $4\cdot10^{-14}$ |
| hex27 $4^3$ clamped | 2187 | 109 | 169 (0.5 s) | $2\cdot10^{-13}$ |
| hex8 $10^3$ clamped | 3993 | 103 | 159 (0.7 s) | $1\cdot10^{-12}$ |
| hex20 $8^3$ clamped | 8019 | 281 | 439 (2.1 s) | $1\cdot10^{-12}$ |

- **Iterations:** these grow roughly like $h^{-1}$, as expected for Jacobi ($\sqrt\kappa$). No P drops occurred in these runs.
- **Floor:** at tol $10^{-16}$ on hex20 $4^3$ the solve accepts at the floor threshold (`converged` and `stagnated`). Over seeds 0–3, the lowest max wanted residual is $1.2$–$3.4\cdot10^{-12}$ clamped and $3.4$–$4.4\cdot10^{-13}$ free, that is $0.5$–$1.9\times\varepsilon\lambda_\max/\lambda_1$.
- **Near-singular M:** hex20 $6^3$ with 30% void elements ($E\cdot10^{-9}$, $\rho\cdot10^{-12}$) converges to $10^{-8}$ in 337–374 iterations (3 seeds), with λ errors of $10^{-12}$.
- **Thin plates are out of reach for Jacobi.**
  - A clamped hex20 plate $10\times10\times1$ with $L/t = 100$ has $\lambda_\max/\lambda_1 = 7\cdot10^{13}$. After 6000 iterations λ̃₁ is still $8\cdot10^4$ times too high.
  - λ̃₁ decreases monotonically. The λ-scaled residual grows (3 → 70), because λ̃ shrinks faster than the absolute residual.
  
  Plates need the preconditioners of PRs 10–13.
- **Cost per iteration** (hex20 $24^3$, 182k DOF):
  - 72 ms at $m = 15$ (K and M applies about 40 ms) and 216 ms at $m = 48$ (about 90 ms). The rest is GEMMs at about 450 GFLOP/s.
  - Recompute iterations cost 150 / 390 ms.
  - The setup (power iterations, start block, R check) takes about 0.5 s warm (≈ 7 iterations) and 1.1 s with the first-call compiles.
- **Step memory:**
  - With a diagonal operator, so that no operator workspace counts, the compiled step holds temp + donated state = 16.0 blocks of $(n, m)$ fp64. This is pinned by a test; without the barriers it is 18.
  - The roadmap's 9 + 2 blocks were not reached in any measured variant. A donated buffer takes its output only after every read of it, so the new $X$ and $P$ coexist with the old ones and $W$.
  - With a real operator the peak is the larger of that and the apply phase plus the operator workspace: 52–56 blocks at hex20 $8^3$ (8k DOF), 16.05 at 182k and 16.00 at 421k, so the step reaches its 16 blocks by about 180k DOF.
  - PR 9b's estimator models this as max(16 blocks, apply phase + operator workspace), together with the memory model.

## Sizes

- **Reference size** means $\le$ `MAX_DENSE` = 20k free DOF: dense `eigh`, $M^{-1}$ and `count_below` are affordable (PR 9's dense $r_M$ and inertia comparisons, PR 14's exact count).
- **The PR 8 `eigsh`** means `reference_modes` alone: about 100k free DOF on bulky meshes with `points`, about 35k without, further on plates (used by PRs 11, 13, 16, 18).

## Dense inertia count (`count_below`)

`count_below(K, M, sigma, free=None)` returns the exact number of eigenvalues of $(K_f, M_f)$ below $\sigma$ (any finite float; $M_f$ positive semidefinite), by Sylvester's law of inertia on $A = K_f - \sigma M_f$:

- **Factorization.** `A = (K_f - sigma * M_f).toarray(order="F")`, then LAPACK `dsytrf` (Bunch–Kaufman $A = LDL^\top$, the routine `scipy.linalg.ldl` calls) with the queried `lwork` and `overwrite_a`: in place on one dense copy. f2py overwrites only Fortran-ordered input; C order would double the memory, and `scipy.linalg.ldl` returns three $n^2$ arrays. Without the queried `lwork` the unblocked code takes 123 s at 15k instead of 7.5 s. `lower=1` reads one triangle, so round-off asymmetry does not matter.
- **Counting.** $D$ has $1\times1$ and $2\times2$ blocks (`ipiv < 0` marks the rows of $2\times2$ blocks). Bunch–Kaufman chooses a $2\times2$ pivot only when $|a_{kk}|\,\sigma_r < \alpha\lambda^2$ and $|a_{rr}| < \alpha\sigma_r$, $\alpha = (1 + \sqrt{17})/8$, on the entries that form the block. So $|a_{kk}a_{rr}| < \alpha^2\lambda^2 < \lambda^2$ and $\det < 0$: each $2\times2$ block has exactly one negative eigenvalue. The count is the number of negative $1\times1$ pivots plus half the number of $2\times2$ rows.
- **Errors.** An exactly zero pivot ($\sigma$ is an eigenvalue to working precision) raises `ValueError`, as does $n_\text{free} >$ `MAX_DENSE` = 20 000 (3.2 GB, 16 s; 1.9 s at 8k). Counts are exact for $\sigma$ in gaps of relative width $\ge 10^{-3}$; a $\sigma$ inside a near-degenerate pair or the rigid cluster gives an ambiguous count.

**Why two inertia counts.** `count_below` is dense and pivoted (Bunch–Kaufman is backward stable), so it is exact for well-separated $\sigma$: the roadmap's exact count at reference sizes (PRs 9, 14). The check inside `reference_modes` is sparse and unpivoted; it reuses the solver's factorization and runs to about 100k DOF, and a failure is loud.

## Validation

`tests/test_reference.py`, against dense `scipy.linalg.eigh` of $(K_f, M_f)$:

- **Accuracy**, `n_modes = 20`: a free orthotropic hex20 box (consistent and lumped, with nodal masses including zeros: the case that needs pass 2), a clamped curved hex27, the mixed hex8 + hex20 mesh with `xmin: x` (3 rigid modes), and a free hex27 plate $8\times8\times1$ with $L/t = 200$. Rigid modes $|\lambda| \le 10\varepsilon\lambda_\max$, elastic $\le 2 f_\lambda$, $\|V^\top MV - I\| \le 10^{-12}$, constrained rows 0, residuals $\le 100\varepsilon$; an explicit $\sigma = -0.3\lambda_{n+1}$ and the run without `points` give the same $\lambda$. `_order` returns a permutation, also in the tie fallback.
- **Certification**, each check pinned by its message: a dropped mode (count), the real ARPACK failure above (`slow`), a ghost copy (orthonormality), a vector perturbed by $10^{-6}$ M-orthogonally to the others (residuals), a patched SuperLU with `perm_r != perm_c`.
- **Clusters:** the first four `n_modes` ending inside an exact multiplicity of a free isotropic hex20 $4^3$ cube certify and match up to the last gap; `n_modes = 3` on a free box raises; on the $L/t = 1600$ plate, `n_modes` 7, 10, 12, 17, 21, 25 certify under the automatic σ rule (a purely relative gap threshold falsely raises at 10, 12, 17).
- **Inputs** raise `ValueError`; `restrict` matches dense slicing.
- **`count_below`** matches the dense `eigh` count at mid-gap $\sigma$ (relative gap $\ge 10^{-3}$, including the first gap above the rigid cluster) on a clamped hex27 and a free hex20 box, consistent and lumped; these cases have $2\times2$ pivots.

`tests/test_lobpcg.py`, against `reference_modes` and dense linear algebra:

- **Accuracy** at tol $10^{-10}$, 12 modes, on clamped hex20 $6^3$, free hex20 $6^3$ (rigid deflation, $\|R^\top MV\| \le 10^{-10}$), clamped hex27 $4^3$ and clamped hex8 $10^3$.
  - Modes with relative gap $\ge 10^{-2}$: λ error $\le \max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$, with $\lambda_\max$ from `eigsh(which="LA")`. Gaps in $[10^{-4}, 10^{-2})$: $\le 10\,$tol, cluster-wise.
  - Principal sines $\le 10\,r_M\lambda/\mathrm{gap}$, with the dense $r_M$. The reference's $r_M$ is $\ge 10\times$ smaller. Sines are computed stably, as $\sigma_\max(L^\top(Q - Q_rQ_r^\top MQ))$; $\sqrt{1-\cos^2}$ bottoms out near $10^{-8}$. Clusters below a relative gap of $10^{-4}$ are compared by subspace.
  - $\|V^\top MV - I\| \le 10^{-10}$, constrained rows 0, and `info["residuals"]` equal to residuals recomputed with the operator.
  - No P drops, and `norm_K`, `norm_M` within $[0.9, 1]\times$ the exact 2-norms (`eigsh`).
- **Exact multiplicities** on a free square-section beam and a free cube. A beam of section $1\times1.002$ gives bending pairs in the $[10^{-4}, 10^{-2})$ band.
- **Stagnation:**
  - tol $10^{-16}$ gives floor acceptance;
  - with the floor term patched to 0, the no-progress exit;
  - a 1-mode run is not flagged.
- **Robustness:**
  - a start block with duplicated columns (refill);
  - a forced $P \approx XC$ at noise $10^{-12}$ (the Cholesky fails) and $10^{-6}$ (finite, pivot below `PIVOT`): P is dropped, the new $X$ stays M-orthonormal;
  - the recompute schedule (iterations 20, 40) and explicit products after a recompute step;
  - the P invariant after 30 steps ($\le 10^{-6}$);
  - `_p_coeffs` masks exactly a noise column;
  - near-singular M within 500 iterations;
  - a NaN preconditioner raises `FloatingPointError`.
- **Compiles:** one step compile per solve, and 0 for a second solve with a new operator, preconditioner and tol (subprocess, cache off, `jax.monitoring`).
- **Memory:** step memory $\le 16.5$ blocks.
- **Floor** (slow): within $[0.1, 10]\times\varepsilon\lambda_\max/\lambda_1$.
- **Jacobi, σ and errors:** Jacobi against the assembled diagonals; σ within $[\lambda_1, 2\lambda_1]$; `ValueError` for a too-large block, a bad `X0`, `n_modes` or `maxit` below 1, and a non-M-orthonormal R; `RuntimeError` after 3 refills; the `maxit` and no-progress exits run the explicit confirmation (counted).

The discretization itself is verified in [`verification.md`](verification.md).
