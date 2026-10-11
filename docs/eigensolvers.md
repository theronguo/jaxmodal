# Eigensolvers

`jaxmodal.eigen.reference_modes` (module `eigen/reference.py`) is the SciPy reference eigensolver on assembled matrices; `jaxmodal.linalg.count_below` (module `linalg/inertia.py`) is the exact dense inertia count (PR 8a). They are oracles for the matrix-free LOBPCG (PR 9 adds it here) and for later PRs. Both run on the host on the SciPy matrices from `assemble`.

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
2. **Pass 2**, if pass 1 finds near-zero eigenvalues ($\tilde\lambda_1 < 100|\sigma_0|$: rigid modes, a nearly singular $K_f$, or a thin structure whose low modes lie below $100|\sigma_0|$), at $\sigma = -0.01\,\tilde\lambda_k$. Then $|\sigma| \le \lambda_k$, and the rigid-mode amplification $\lambda_7/|\sigma|$ is harmless. Only if every wanted mode is rigid-mode noise ($\tilde\lambda_k \le 100\,\varepsilon\,\bar s$, which then raises \"no gap\") does pass 1 stand.

For SPD $K_f$ (clamped, sliding), pass 1 is as accurate as a pass at $-0.01\lambda_k$. Pass 2 at $-\lambda_k$, $-0.1\lambda_k$ and $-0.01\lambda_k$ gives the same accuracy; 0.01 is a margin.

The rule is validated against dense `eigh` up to $\kappa = \lambda_\max/\lambda_7 \approx 8\cdot10^9$, and certifies on the free hex27 plate $8\times8\times1$ with spacing $(1, 1, 0.005)$ ($L/t = 1600$, $\kappa = 3\cdot10^{13}$) for `n_modes` 7–25; there the floor $f_\lambda = 10\,\varepsilon\,\kappa \approx 0.07$ for the lowest elastic mode, and the measured error against dense `eigh` is $0.09 f_\lambda$ (lowest 12). (Without the thin-structure case of pass 2, pass 1 at $|\sigma_0| \approx 10^5\lambda_k$ fails the residual check there, at 700–7800ε.)

**Explicit `sigma`** skips the rule (one pass), e.g. $-0.01\lambda_k$ from a coarser run, for PR 16's solid shells ($\kappa \approx 10^{12}$) if the rule ever fails there. A $\sigma$ far below $\lambda_k$ raises the residuals (45–90ε on the $L/t = 200$ plates at $-10^{-2}\bar s$) and can trip the residual check, loudly.

### Certification

Every result is checked; a failure raises `RuntimeError` instead of returning wrong modes. With the computed $\lambda_1 \le \ldots \le \lambda_{n+1}$ ($n$ = `n_modes`):

1. **Orthonormality:** $\|V^\top M_f V - I\|_\max \le 10^{-10}$.
2. **Residuals:** every mode has $\|K_f v - \lambda M_f v\|_\infty \le 100\,\varepsilon\,(\|K_f\|_\infty + |\lambda|\,\|M_f\|_\infty)\,\|v\|_\infty$ (measured maximum $10.65\varepsilon$). With accurate, M-orthonormal eigenvectors, a count equal to $j$ proves that the first $j$ vectors span the lowest invariant subspace; together, 1 and 2 rule out a ghost copy standing in for a missing mode.
3. **Count.** Let $j$ be the largest index $\le n$ with $\lambda_{j+1} - \lambda_j > \tau = \max(10^{-6}|\lambda_{n+1}|,\ 100\,\varepsilon\,\bar s)$: the last gap at or below the requested count. Rigid-mode noise ($\lesssim 4\varepsilon\bar s$), exact FE degeneracies and the splitting of degenerate pairs by `eigsh` (up to $2.6\cdot10^{-5}$ relative, $\le 1.7\varepsilon\bar s$ absolute, on very thin square plates) count as clusters; real gaps are far wider. $K_f - \mu M_f$ at $\mu = (\lambda_j + \lambda_{j+1})/2$ is factored, and by Sylvester's law of inertia the number $c$ of negative pivots is the number of eigenvalues below $\mu$; $c \ne j$ raises `eigsh missed modes: c eigenvalues below μ, j found`. No gap at all (e.g. `n_modes=3` on a free body, inside the rigid cluster) raises `ValueError` asking for more modes.

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

The attainable accuracy is the floor $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$ ($\hat\lambda_\max$: the largest eigenvalue, from dense `eigh` or `eigsh(which="LA")`) (relative), i.e. about $\varepsilon\kappa$ for the lowest elastic mode. Against dense `scipy.linalg.eigh` of $(K_f, M_f)$, up to 20 modes on free hex20 boxes (consistent and lumped), clamped hex27, sliding hex20, and free hex20/hex27 plates with $L/t = 100, 200$ ($\kappa \le 7.8\cdot10^9$):

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

- **Accuracy**, `n_modes = 20`: a free orthotropic hex20 box (consistent and lumped, with nodal masses including zeros: the case that needs pass 2), a clamped curved hex27, the mixed hex8 + hex20 mesh with `xmin: x` (3 rigid modes), and a free hex27 plate $8\times8\times1$ with $L/t = 200$. Rigid modes $|\lambda| \le 10\varepsilon\lambda_\max$, elastic $\le 2 f_\lambda$, $\|V^\top MV - I\| \le 10^{-12}$, constrained rows 0, residuals $\le 100\varepsilon$; an explicit $\sigma = -0.01\lambda_{n+1}$ and the run without `points` give the same $\lambda$. `_order` returns a permutation, also in the tie fallback.
- **Certification**, each check pinned by its message: a dropped mode (count), the real ARPACK failure above (`slow`), a ghost copy (orthonormality), a vector perturbed by $10^{-6}$ M-orthogonally to the others (residuals), a patched SuperLU with `perm_r != perm_c`.
- **Clusters:** the first four `n_modes` ending inside an exact multiplicity of a free isotropic hex20 $4^3$ cube certify and match up to the last gap; `n_modes = 3` on a free box raises; on the $L/t = 1600$ plate, `n_modes` 7, 10, 12, 17, 21, 25 certify under the automatic σ rule (a purely relative gap threshold falsely raises at 10, 12, 17).
- **Inputs** raise `ValueError`; `restrict` matches dense slicing.
- **`count_below`** matches the dense `eigh` count at mid-gap $\sigma$ (relative gap $\ge 10^{-3}$, including the first gap above the rigid cluster) on a clamped hex27 and a free hex20 box, consistent and lumped; these cases have $2\times2$ pivots.

The discretization itself is verified in [`verification.md`](verification.md).
