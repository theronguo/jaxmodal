# PR 9a plan: generalized LOBPCG core, Jacobi, σ pre-estimate

Branch `pr-09a-lobpcg`, from `main` after PR 8c.

## Split of PR 9

PR 9 is large: a solver core, a memory-managed outer loop (chunks, locking, fp32 basis, 6-block mode, estimator) and the public API. It is split in dependency order like PR 8:

| Part | Contents | Roadmap items (Contents / Tests) |
|---|---|---|
| **9a** (this plan) | `lobpcg` (9-block, fixed shapes, masks, SVQB, P drop, implicit updates with periodic explicit recompute, deflation every iteration, floor-aware acceptance, stagnation flag), `jacobi` on $A_P$, `sigma_estimate`, `docs/eigensolvers.md` (algorithm, convergence theory, measured floor) | Contents 1 (without the 6-block mode) and 2. Tests: λ error, principal angles, multiplicities, free-free, gap $[10^{-4}, 10^{-2})$, stagnation, rank-deficient start, forced P drop, near-singular M, floor measured, compiles ≤ 2, σ pre-estimate within 2× and the hint skip |
| **9b** | chunked computation with a preallocated locked basis (optionally fp32) and in-place lock insertion, the guard-vector error, the 6-block mode, the workspace estimator | Contents 1 (6-block mode), 3, 4. Tests: guard error, both-mode λ/angle runs and the 6-block match, estimator at 1M DOF, projector memory, chunked fp32 run, in-place insertion, nearly-full assert, step temp independent of capacity and within the estimator, 6-block cost |
| **9c** | `solve_modes(mesh, materials, bcs, n_modes, orientations=None, ...)`, `examples/`, Mac acceptance budgets | Contents 5–7, Milestone |

The PR 9 acceptance criteria do not change. The dependency table row ("9 LOBPCG, public API | 5, 6, 7, 8") and the critical path stay; PRs 10–13 strictly need only 9a (the solver they precondition), 14 needs 9b (chunks), and the M3 milestone needs 9c. The roadmap gets a split note like PR 8's.

## Measured facts (prototype, scratchpad `p9/`; jax 0.11.2, fp64, M3 Max CPU)

Unless stated, isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$ on the box $L = (1, 0.83, 0.67)$, `xmin` clamped, 10 wanted modes, block $m = 15$, Jacobi with $\sigma = 0$, explicit recompute every 20 iterations; errors against `reference_modes`.

- **Convergence (9-block, Jacobi):**

  | mesh | DOF | tol $10^{-6}$ | tol $10^{-10}$ | max rel. λ error | P drops |
  |---|---|---|---|---|---|
  | hex20 $4^3$ clamped | 1275 | 125 its | 211 its | $8\cdot10^{-12}$ | 0 |
  | hex20 $4^3$ free (rigid deflation) | 1275 | 149 | 236 | $1\cdot10^{-12}$ | 0 |
  | hex20 $8^3$ clamped | 8019 | 254 (3.8 s) | 420 (3.4 s) | | 0 |
  | hex20 $8^3$ free | 8019 | 296 | 466 | | 0 |
  | hex20 $8^3$ clamped, 20 modes ($m = 25$) | 8019 | 261 | 464 | | 0 |
  | hex8 $16^3$ clamped | 14739 | 185 | 281 | | 0 |
  | hex27 $6^3$ clamped | 6591 | 159 | 251 | | 0 |

  Times include compilation (≈ 1 s). Iterations grow roughly like $h^{-1}$, as expected for Jacobi ($\sqrt\kappa$).
- **P drop rule.** A relative-θ test (drop P when SVQB of P has $\theta_\min < 10^{-6}\theta_\max$) dropped P on most iterations and the free case stopped converging. The rule below (a P column is *noisy* when its M-norm after orthogonalization against $[X, W]$ is $\le 10^{-8}$ of its M-norm before) never fires on the cases above (0 drops) and fires on a forced $P = XC + 10^{-12}\,\xi$.
- **λ̂_max.** The power iteration on $\mathrm{diag}(M)^{-1}K$ gave $\hat\lambda_\max/\lambda_\max = 0.094$: too low for the Rayleigh–Ritz mask value and the test tolerances. A block-1, unpreconditioned LOBPCG that *maximizes* the Rayleigh quotient (30 iterations) gives 0.93. $\|K\|_2$ and $\|M\|_2$ (for the floor) use 30 plain power iterations on the masked operators.
- **Rank-deficient start** (columns 5–9 duplicating 0–4): without a refill the initial Rayleigh quotients of the zeroed columns are 0 and the residual is NaN. With SVQB dropping the dependent directions and the dropped columns refilled with fresh random vectors (at most 3 rounds), it converges in 171 iterations ($4^3$, tol $10^{-8}$).
- **Floor.** hex20 $4^3$ clamped at tol $10^{-16}$: the floor term of the acceptance threshold is $7.6\cdot10^{-12}$ and the solve accepts there after 274 iterations; the lowest residual reached is $3.3\cdot10^{-12}$, against $\varepsilon\hat\lambda_\max/\lambda_1 = 2.1\cdot10^{-12}$. So the floor-aware threshold is reached, and a tol below it ends in floor acceptance, not in the iteration cap.
- **Thin plates are out of reach for Jacobi.** A clamped hex20 plate $10\times10\times1$ with $L/t = 100$ has $\lambda_\max/\lambda_1 = 7\cdot10^{13}$. After 6000 iterations λ̃₁ is still $8\cdot10^4$ times too high. It decreases monotonically (largest relative rise $< 10^{-12}$, i.e. no instability), while the λ-scaled residual *grows* (3 → 70) because λ̃ shrinks faster than the absolute residual. A hex27 $8\times8\times1$ plate ($L/t = 200$) behaves the same. PR 9 tests therefore use bulky meshes; plates wait for PRs 10–13.
- **Near-singular M (SIMP-like).** hex20 $6^3$, 30% of the elements (random) with $E\cdot10^{-9}$ and $\rho\cdot10^{-12}$: tol $10^{-8}$ in 361 iterations, max relative λ error $8\cdot10^{-12}$ against `reference_modes` on the same materials.
- **σ pre-estimate.** Jacobi-LOBPCG for 1 mode ($m = 6$) from a random start. The λ-scaled residual is a bad early stop: it is small while λ̃ is huge (tol 0.5 stopped after 1 iteration at 432× λ₁). Stopping when λ̃₁ has decreased by less than 5% over the last 10 iterations gives:

  | mesh | DOF | stop at | λ̃₁/λ₁ | full 1-mode solve ($10^{-8}$) |
  |---|---|---|---|---|
  | hex20 $8^3$ clamped | 8019 | 89 | 1.29 | 520 |
  | hex20 $8^3$ free | 8019 | 102 | 1.04 | 430 |
  | hex8 $16^3$ clamped | 14739 | 59 | 1.27 | 353 |
  | hex27 $6^3$ clamped | 6591 | 76 | 1.01 | 296 |
  | hex20 $12^3$ clamped | 24843 | 106 | 1.32 | 744 |
  | hex8 $24^3$ free | 46875 | 82 | 1.02 | 397 |

  A fixed count does not work (20 iterations: 4–98×). Change $< 1\%$ per single iteration stops too early on the free box (51 its, 2.12×).
- **Stagnation.** "No new minimum of the max wanted residual (by a factor 0.9) for 50 iterations" alone fires falsely in the first iterations, where the λ-scaled residual rises while λ̃ falls (1-mode run stopped at 51 iterations). Counting a decrease of the wanted Ritz-value sum by more than a relative $10^{-10}$ as progress too removes this: the 1-mode runs then converge (620 / 520 iterations) without a flag.
- **Compiles.** With `has_P` static and the recompute flag traced, the step compiles exactly twice per solve (counted with `jax.monitoring` `backend_compile_duration` events filtered by `fun_name`, cache disabled).

## Files

| File | Change |
|---|---|
| `src/jaxmodal/eigen/lobpcg.py` | new: `lobpcg`, `jacobi`, `sigma_estimate`, the step and its helpers |
| `src/jaxmodal/eigen/__init__.py` | export `lobpcg`, `jacobi`, `sigma_estimate` |
| `tests/test_lobpcg.py` | new: the tests below |
| `tests/helpers.py` | `r_M` (dense M⁻¹-norm residual) and `principal_sines` if shared; otherwise in the test file |
| `docs/eigensolvers.md` | LOBPCG section |
| `docs/ROADMAP.md`, `README.md` | split note, status, API |

## API

```python
lam, V, info = lobpcg(op, free, n_modes, tol=1e-6, R=None, precond=None, sigma=None,
                      X0=None, maxit=500, recompute_every=20, seed=0)
```

- `op`: any operator with `apply_K`, `apply_M`, `diag_K`, `diag_M`, `n_dof` (PR 5/6). `free`: `(n_dof,)` bool from `dof_mask`. The solver uses the masked operators $K_m v = m \odot K(m \odot v)$, $M_m$ likewise.
- `R`: `(n_dof, k)` M-orthonormal deflation basis (rigid modes after `m_orthonormalize`) or `None`. Every new direction W is projected, $W \leftarrow W - R(R^\top M W)$, and so is the start block.
- `precond`: a function `(n, j) -> (n, j)`, fixed, linear, SPD on the free DOFs, masked output. Default `jacobi(op, free, sigma)`.
- `sigma`: the σ of $A = K + \sigma M$. `None` runs `sigma_estimate` when the preconditioner needs it (the default Jacobi does). A user value skips it.
- `X0`: `(n_dof, j)` start block, $j \le m$; filled with random columns up to $m$.
- Returns `lam` `(n_modes,)` ascending, `V` `(n_dof, n_modes)` M-orthonormal with constrained rows exactly 0, and `info` (a dict of host values): `iterations`, `converged` (every wanted pair met its threshold, confirmed by an explicit recompute), `stagnated` (see Convergence), `residuals` and `thresholds` `(n_modes,)`, `p_drops`, `refills`, `sigma`, `sigma_iterations`, `lmax`, `norm_K`, `norm_M`, `history` (max wanted residual per iteration).
- Block size $m = n_\mathrm{modes} + \max(5, \lfloor 0.2\,n_\mathrm{modes} \rfloor)$; $m \le$ the number of free DOFs minus $k$ is checked, else `ValueError`.
- `jacobi(op, free, sigma) -> callable`: $v \mapsto m \odot v / (\mathrm{diag}K + \sigma\,\mathrm{diag}M)$. That is Jacobi on $A_P$ with masked output (the constrained diagonal $s$ of $A_P$ never reaches the output).
- `sigma_estimate(op, free, R=None, maxit=300) -> (sigma, iterations)`: the Rayleigh quotient λ̃₁ ≥ λ₁ from 1-mode Jacobi(σ = 0)-LOBPCG, stopped when λ̃₁ decreased by less than 5% over the last 10 iterations.

## Algorithm (`lobpcg.py`)

Hetmaniuk–Lehoucq / Duersch et al. 2018, with fixed shapes: every block is $(n, m)$, inactive columns are exact zeros with a boolean mask.

1. **Setup (host loop, eager JAX).** $\|K_m\|_2$, $\|M_m\|_2$ by 30 power iterations; $\hat\lambda_\max$ by the 30-step Rayleigh-quotient-maximizing block-1 LOBPCG (Measured facts). σ (if needed), the preconditioner. Start block $X$: `X0` and/or random, masked, deflated, then SVQB; columns that SVQB drops are refilled with fresh random vectors, up to 3 rounds (`info["refills"]`). Then Rayleigh–Ritz on $X$.
2. **SVQB** `svqb(V, KV, MV, act, tau=1e-14)`: zero the inactive rows/columns of $G = V^\top M V$ (symmetrized), scale by $D = \mathrm{diag}(G)^{-1/2}$ (1 on inactive or zero columns), `eigh`, keep $\theta > \tau\,\theta_\max$, $T = D Q\,\mathrm{diag}(\mathrm{keep}/\sqrt\theta)$. Inactive columns are the exact $\theta = 0$ eigenspace, so kept directions carry nothing from them. Returns the transformed $V, KV, MV$ and the new active mask; dropped columns are exact zeros.
3. **Step** (jitted, `static_argnames="has_P"`, state `(X, KX, MX, P, KP, MP, lam)` donated, traced bool `recompute`):
   1. Residuals $R = KX - MX\Lambda$, $r_i$ and thresholds $t_i$ (Convergence). Active columns $a = r > t$ (soft locking: converged columns get no new direction but stay in X).
   2. $W = T(R) \odot a$, masked by the preconditioner; deflate by $R$; $KW$, $MW$.
   3. M-orthogonalize $W$ against $X$ twice, SVQB, once more against $X$, SVQB again.
   4. If `has_P`: M-orthogonalize $P$ against $X$ and $W$ twice; columns with M-norm after $\le 10^{-8}\times$ before are noisy; SVQB of the clean live columns. If any column was noisy or SVQB dropped one, the whole P block is dropped for this iteration (counted in `p_drops`).
   5. Rayleigh–Ritz on $S = [X, W, P]$ with the active mask: $G_K = S^\top K S$, $G_M = S^\top M S$ (symmetrized), masked rows and columns zeroed off the diagonal with $G_M = 1$ and $G_K = 10\,\hat\lambda_\max$ on their diagonal; Cholesky of $G_M$, `eigh`, the lowest $m$.
   6. $X' = SC$, $P' = SC_P$ with the X-rows of $C$ zeroed, and their K and M products by the same GEMMs. `lax.cond(recompute, ...)` replaces $KX', MX', KP', MP'$ by explicit applies.
   7. Return the new state, the residuals and thresholds of $X'$, and the drop flag.

   `has_P = False` is used only in the first iteration, so at most 2 compiled variants. The step's jit `fun_name` is `lobpcg_step` (for the compile-count test).
4. **Outer loop (Python).** One step per iteration with `recompute = (it + 1) % recompute_every == 0`; residuals come back to the host ($m$ floats). When the lowest $n_\mathrm{modes}$ meet their thresholds, recompute $KX$, $MX$ explicitly and check again; exit only if that passes.

## Convergence

- Residual: the λ-scaled $r_i = \|Kx_i - \lambda_i Mx_i\| / (|\lambda_i|\,\|Mx_i\|)$ on the masked operators.
- Threshold: $t_i = \max(\mathrm{tol},\ c\,\varepsilon\,(\|K\| + |\lambda_i|\,\|M\|)\,\|x_i\| / (|\lambda_i|\,\|Mx_i\|))$, $c = 10$.
- Exit: all wanted pairs meet $t_i$ (explicitly confirmed) → `converged = True`; `stagnated = True` as well if any wanted $r_i > \mathrm{tol}$ (accepted at the floor). No progress for 50 iterations → `stagnated = True`, `converged = False`; progress is a new minimum of the max wanted residual by a factor 0.9 *or* a decrease of the sum of the wanted Ritz values by more than a relative $10^{-10}$ (Measured facts). `maxit` reached → both False. The solver never raises on non-convergence; the caller checks `info`.
- Theory for the docs: with $x$ M-normalized and the M⁻¹-norm residual $r_M = \|Kx - \lambda Mx\|_{M^{-1}}/|\lambda|$, $\sin\theta \le r_M\lambda/\mathrm{gap}$ and $|\delta\lambda|/\lambda \le r_M^2\lambda/\mathrm{gap}$ (Davis–Kahan / Kato–Temple in the M-inner product); the 2-norm $r$ above is the cheap proxy, equal up to the conditioning of M.

## Tests (`tests/test_lobpcg.py`)

All against `reference_modes` (requesting enough modes past the last wanted one to finish clusters), tol $= 10^{-10}$, at sizes of 1–8k DOF. $\hat\lambda_\max$ is `info["lmax"]`; $f_\lambda = \max(10^{-8}, 10\,\varepsilon\,\hat\lambda_\max/\lambda)$.

1. **λ error and principal angles** (parametrized: clamped hex20 $6^3$, free hex20 $6^3$ with rigid deflation, clamped hex27 $4^3$, clamped hex8 $10^3$; 12 modes):
   - modes with relative gap $\ge 10^{-2}$: $|\lambda - \lambda_\mathrm{ref}|/\lambda_\mathrm{ref} \le f_\lambda$;
   - principal angles in the M-inner product, $\sin\theta \le 10\,r_M\lambda/\mathrm{gap}$ (absolute gap, dense $r_M$ via a Cholesky of $M_f$); the reference's $r_M$ is asserted $\ge 10\times$ smaller (the reference reaches about $10^{-14}$, LOBPCG stops near $10^{-10}$); clusters with relative gap $< 10^{-4}$ by the largest principal angle between the cluster subspaces, gap to the nearest outside eigenvalue;
   - $\|V^\top MV - I\|_\max \le 10^{-10}$, constrained rows exactly 0, and $\lambda$ ascending.
2. **Multiplicities**: free square-section beam ($2\times1\times1$, hex20 $6\times3\times3$) and a free cube (hex20 $4^3$): every reference cluster (relative gap $< 10^{-8}$) gets exactly as many LOBPCG eigenvalues within $f_\lambda$ of it.
3. **Free-free**: $\|R^\top M V\|_\max \le 10^{-10}$ (in test 1's free case).
4. **Gap $[10^{-4}, 10^{-2})$**: clamped beam with section $1\times1.004$ (bending pairs at relative gap ≈ $4\cdot10^{-3}$, measured in the test): λ error $\le 10\,\mathrm{tol}$ cluster-wise, sorted.
5. **Stagnation**: tol $= 10^{-16}$ on hex20 $4^3$ → `converged` and `stagnated` both True, `iterations < maxit`, and every wanted residual ≤ its threshold. With a monkeypatched stall (`maxit` large, tol $10^{-16}$, $c = 0$ via a module constant) the no-progress exit fires before `maxit` with `converged = False`.
6. **Rank-deficient start**: `X0` with duplicated columns → `refills ≥ 1`, converges, λ match as in test 1.
7. **Forced P drop**: call the step with $P = XC + 10^{-12}\xi$ (random $C$, $\xi$) → drop flag True; with a random $P$ → False.
8. **Near-singular M**: hex20 $6^3$ with 30% void elements ($E\cdot10^{-9}$, $\rho\cdot10^{-12}$) at tol $10^{-8}$: converged within 500 iterations, λ error $\le f_\lambda$ against `reference_modes` on the same materials.
9. **Floor** (slow): the measured floor at reference size (box and free box): the lowest attained max residual at tol $10^{-16}$ is within $[0.1, 10]\times\varepsilon\hat\lambda_\max/\lambda_1$; the values go into `docs/eigensolvers.md`.
10. **Compiles**: in a fresh subprocess with the cache disabled, `jax.monitoring` backend-compile events with `fun_name` containing `lobpcg_step` count ≤ 2 for one solve (σ given, so the pre-estimate is not counted).
11. **σ pre-estimate**: on clamped hex20 $8^3$ and free hex20 $6^3$: $\lambda_1 \le \sigma \le 2\lambda_1$; with `sigma=` given, `info["sigma_iterations"] == 0` and $\sigma$ is the given value.
12. **Jacobi**: `jacobi(op, free, s)(v)` equals $m\odot v/(\mathrm{diag}K + s\,\mathrm{diag}M)$ against the assembled diagonals; constrained rows 0.
13. **Errors**: block size larger than the free DOFs minus the deflation rank raises `ValueError`.

### Mutants (each must fail a test)

| Mutant | Caught by |
|---|---|
| no deflation of W | 3 |
| Rayleigh–Ritz mask diagonal $G_K = 0$ instead of $10\hat\lambda_\max$ | 1, 6 (spurious zero Ritz values) |
| no refill of dropped start columns | 6 |
| P never dropped | 7 |
| floor term removed ($c = 0$) | 5 |
| explicit confirmation at exit removed | 1 (independent residual check in the test) or survivor, then noted |
| Jacobi ignores σ | 12 |
| σ stop window 1 iteration | 11 |
| no masking of K (unmasked apply) | 1 (constrained rows) |
| soft-locking mask removed ($a$ = all) | survivor expected (only cost); noted |

## Docs

- `docs/eigensolvers.md`: a LOBPCG section: algorithm (steps above, why SVQB with zeroed inactive columns, the mask values, P drop rule, implicit updates), convergence (residual, threshold, theory, stagnation), λ̂_max estimator, σ pre-estimate, Jacobi, the measured floor and iteration tables, the thin-plate limit of Jacobi.
- `docs/ROADMAP.md`: split note for PR 9 (like line 367) and the 9a status line; the "power iteration" for λ̂_max becomes the max-Rayleigh-quotient LOBPCG (measured 0.093 vs 0.93).
- `README.md`: `lobpcg` in the API overview.

## Not covered

- Locking, chunks, fp32 basis, 6-block mode, estimator, memory tests: 9b.
- `solve_modes`, examples, budgets: 9c.
- Thin plates: need the PR 10–13 preconditioners.
