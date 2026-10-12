# PR 9a plan: generalized LOBPCG core, Jacobi, σ pre-estimate

Branch `pr-09a-lobpcg`, from `main` after PR 8c.

## Split of PR 9

PR 9 is large: a solver core, a memory-managed outer loop (chunks, locking, fp32 basis, 6-block mode, estimator) and the public API. It is split in dependency order like PR 8:

| Part | Contents | Roadmap items (Contents / Tests) |
|---|---|---|
| **9a** (this plan) | `lobpcg` (9-block, fixed shapes, masks, SVQB, P drop, implicit updates with periodic explicit recompute, deflation every iteration, floor-aware acceptance, stagnation flag), `jacobi` on $A_P$, `sigma_estimate`, `docs/eigensolvers.md` (algorithm, convergence theory, measured floor) | Contents 1 (without the 6-block mode) and 2. Tests: λ error, principal angles, multiplicities, free-free, gap $[10^{-4}, 10^{-2})$, stagnation, rank-deficient start, forced P drop, near-singular M, floor measured, compile count, σ pre-estimate within 2× |
| **9b** | chunked computation with a preallocated locked basis (optionally fp32) and in-place lock insertion, the guard-vector error, the 6-block mode, the workspace estimator and the memory model docs | Contents 1 (6-block mode), 3, 4. Tests: guard error, both-mode λ/angle runs and the 6-block match, estimator at 1M DOF, projector memory, chunked fp32 run, in-place insertion, nearly-full assert, step temp independent of capacity and within the estimator, 6-block cost, compile count on a chunked solve |
| **9c** | `solve_modes(mesh, materials, bcs, n_modes, orientations=None, ...)`, `examples/`, Mac acceptance budgets | Contents 5–7, Milestone (M2) |

The PR 9 acceptance criteria do not change, except for the two deviations listed below. The dependency table row stays "9 LOBPCG, API | 5, 7, 8", and so does the critical path. Which part each later PR needs:
- PR 10 needs no part of PR 9.
- PR 12 strictly needs 9a.
- PRs 11 and 13 need 9c, for the `solve_modes` dispatch, and 9b's estimator for their Mac acceptance.
- PR 14 needs 9b and 9c.
- The M2 milestone needs 9c.

The roadmap gets a split note like PR 8's.

**Deviations from the roadmap** (measured below; recorded in the roadmap note and `docs/eigensolvers.md`):
1. **The default Jacobi uses σ = 0, and `lobpcg` never runs the σ pre-estimate.**
   - For Jacobi, σ·diag M/diag K ≤ 4·10⁻⁴, and iterations change by < 3% between σ = 0, λ₁ and λ₁₀.
   - The pre-estimate needs 60–110 iterations (not "a few"), about 10–15% of a solve.
   - `sigma_estimate` is still implemented and tested here (within 2× of λ₁) for the preconditioners that need σ.
   - The roadmap test "a user hint skips it" moves to PR 11, whose `solve_modes` dispatch is the first to run the pre-estimate (roadmap PR 11: "including the σ pre-estimate").
2. **λ̂_max is not needed by the solver.**
   - The Rayleigh–Ritz mask value is $10\max_i|\lambda_i| + \|K\|/\|M\|$, built from the current Ritz values (Algorithm).
   - The tests take λ_max from SciPy (`eigsh(which="LA")` on the free blocks) for their tolerances, in place of "λ̂_max from the power iteration".
   - The power iteration on diag(M)⁻¹K would have been unusable anyway: it measured 0.094 λ_max.

## Measured facts (prototype `scratchpad/p9/lob2.py`; jax 0.11.2, fp64, M3 Max CPU)

Unless stated otherwise:
- isotropic material, $E = 3$, $\nu = 0.3$, $\rho = 2$;
- the box $L = (1, 0.83, 0.67)$ with `xmin` clamped;
- 12 wanted modes, block $m = 17$;
- Jacobi with σ = 0, explicit recompute every 20 iterations;
- errors against `reference_modes`.

The prototype is the design below: one module-level jitted step with every array passed as an argument, P kept M-orthogonal to X in coefficient space, and pairwise Rayleigh–Ritz.

- **Convergence:**

  | mesh | DOF | tol $10^{-6}$ | tol $10^{-10}$ | max rel. λ error | P drops |
  |---|---|---|---|---|---|
  | hex20 $6^3$ clamped | 3675 | 182 its | 310 its (1.2 s) | $3\cdot10^{-13}$ | 0 |
  | hex20 $6^3$ free (rigid deflation) | 3675 | 209 | 343 (1.3 s) | $3\cdot10^{-14}$ | 0 |
  | hex27 $4^3$ clamped | 2187 | 106 | 163 (0.4 s) | $7\cdot10^{-13}$ | 0 |
  | hex8 $10^3$ clamped | 3993 | 109 | 161 (0.6 s) | $3\cdot10^{-13}$ | 0 |
  | hex20 $8^3$ clamped | 8019 | 277 | 444 (1.9 s) | $1\cdot10^{-12}$ | 0 |

  Times are warm, with compilation and setup excluded. Iterations grow roughly like $h^{-1}$, as expected for Jacobi ($\sqrt\kappa$).

  The first prototype (`lob.py`) differed in three ways: it closed over the arrays, re-orthogonalized P in n-space, and used two step variants. It needed similar iteration counts.
- **Closures vs arguments** (performance review, hex20 $24^3$, 182k DOF).
  - A step that closes over the operator, mask, R and preconditioner embeds them as HLO constants:
    - diagonal operator at 100k DOF: 6.4 MB of HLO, against 0.05 MB with arguments;
    - real operator: 1.86 s compile and 3.0 GB peak RSS, against 0.30 s and 0.82 GB.
  
    A factory that builds a new jit per solve also recompiles every solve.
  - The design below compiles one step per shape. In the prototype, a second solve with the same shapes added 0 compiles, and a second tol reused the executable.
- **Step memory** (performance review, `lob2`, diagonal operator, unit = one $n\times m$ fp64 block):
  - The step measures temp + aliased state = 12 + 6 blocks at $m = 15$ and $48$, $n = 10^5$ and $10^6$.
  - The temp is the same with R of 6 or 60 columns, so the temp does not grow with the basis.
  - XLA:CPU does not fuse a sum of GEMMs: `X C_x + W C_w + P C_p` holds all three products at once. Sequencing the update with `optimization_barrier` (below) gives 10 + 6 = 16 blocks at the same speed (73 / 215 ms against 72 / 216 ms).
  - The roadmap's "9 blocks + ≈ 2 Rayleigh–Ritz temporaries ≈ 11 blocks" was not reached in any measured variant. A donated buffer takes its output only after every read of it, so the 6 new X/P blocks coexist with the 6 old ones and W (3). One $(n, 3m)$ array or a stacked $(3, n, m)$ layout measured worse (21 + 9).

  9a records the measured temp + alias of the final step in `docs/eigensolvers.md`, with the diagonal operator so that the operator workspace is excluded, and pins it (test 14). With a real operator, the peak is the larger of that and the apply phase plus the operator workspace, measured as (temp + alias) blocks:

  | mesh | DOF | m | blocks |
  |---|---|---|---|
  | hex20 $8^3$ | 8k | 15 | 52–56 |
  | hex20 $24^3$ | 182k | 15 | 18 |
  | hex20 $32^3$ | 421k | 15 | 16.9 |

  The barrier saving shows only from about 400k DOF. 9b's estimator models the step as max(16 blocks, apply phase + the operator estimator's workspace) and updates the Memory budget rows, for example 1M at $m = 25$: 3.2 GB instead of 1.8 + 0.4 GB. That is noted there as a deviation. Both figures go into `docs/eigensolvers.md`.
- **Per-iteration cost** (182k DOF, first prototype):

  | m | step | K + M applies | dense part |
  |---|---|---|---|
  | 15 | 77 ms | 40 ms | 48% |
  | 48 | 230 ms | 90 ms | 61% |

  - The dense part is GEMMs at about 450 GFLOP/s.
  - Deflating W through `M(W)` cost a full extra M apply, +26% per iteration. With $M R$ precomputed it costs nothing extra.
  - Pairwise Grams (no concatenation of $[X, W, P]$) save 9% at $m = 48$.
  - `lob2` measures 72 ms at $m = 15$ and 216 ms at $m = 48$. Recompute iterations cost 150 / 390 ms. HLO is 77 KB at $n = 10^6$.
- **P drop rule.**
  - A relative-θ test (drop when SVQB of P has $\theta_\min < 10^{-6}\theta_\max$) dropped P on most iterations, and the free case stopped converging.
  - The noise test used instead: a P column is dropped when its M-norm after orthogonalization against X is $\le 10^{-8}$ of its norm before. It gives 0 drops in all runs above.
  - With P orthogonal in coefficient space, a forced $P = XC + 10^{-12}\xi$ (which breaks the invariant) makes the Gram matrix ill-conditioned; the Cholesky check catches it and P is dropped. A random P is not dropped.
- **Rank-deficient start** (columns 5–9 duplicating 0–4, hex20 $4^3$).
  - Without a refill, the zeroed columns get Rayleigh quotient 0 and the residual is NaN.
  - With the dropped columns refilled once with random vectors, the solve converges in 150 iterations.
- **Floor.** hex20 $4^3$ clamped at tol $10^{-16}$:
  - the solve accepts at the floor threshold after 269 iterations;
  - the threshold is $7.6\cdot10^{-12}$, and the lowest residual reached is $1.9\cdot10^{-12}$;
  - for comparison, $\varepsilon\lambda_\max/\lambda_1 \approx 2.1\cdot10^{-12}$.
- **Thin plates are out of reach for Jacobi.**
  - A clamped hex20 plate $10\times10\times1$ with $L/t = 100$ has $\lambda_\max/\lambda_1 = 7\cdot10^{13}$. After 6000 iterations λ̃₁ is still $8\cdot10^4$ times too high.
  - λ̃₁ decreases monotonically (largest relative rise $< 10^{-12}$, so there is no instability). The λ-scaled residual *grows* (3 → 70) because λ̃ shrinks faster than the absolute residual.
  - A hex27 $8\times8\times1$ plate ($L/t = 200$) behaves the same.
  
  PR 9 tests therefore use bulky meshes; plates wait for PRs 10–13.
- **Near-singular M (SIMP-like).** hex20 $6^3$, with 30% of the elements (random, seed 0) at $E\cdot10^{-9}$ and $\rho\cdot10^{-12}$: tol $10^{-8}$ in 358 / 374 / 337 iterations for start seeds 0 / 1 / 2, 0 P drops, max relative λ error $1\cdot10^{-12}$.
- **σ pre-estimate.** Jacobi-LOBPCG for 1 mode ($m = 6$) from a random start.
  - The λ-scaled residual is a bad early stop: it is small while λ̃ is huge (tol 0.5 stopped after 1 iteration at 432× λ₁). A fixed count fails too (20 iterations: 4–98×).
  - Stopping when λ̃₁ decreased by less than 5% over the last 10 iterations gives:

  | mesh | DOF | stop at | λ̃₁/λ₁ |
  |---|---|---|---|
  | hex20 $8^3$ clamped | 8019 | 89 | 1.29 |
  | hex20 $8^3$ free | 8019 | 102 | 1.04 |
  | hex8 $16^3$ clamped | 14739 | 59 | 1.27 |
  | hex27 $6^3$ clamped | 6591 | 76 | 1.01 |
  | hex20 $12^3$ clamped | 24843 | 106 | 1.32 |
  | hex8 $24^3$ free | 46875 | 82 | 1.02 |

  - The ratio grows slowly with size.
  - The tests review checked other stop windows over 3 seeds:
    - a 1-iteration window gives 3–12×;
    - a 5-iteration window gives up to 2.10×.
  - For Jacobi itself, σ does not matter: iterations to $10^{-8}$ are 330 / 338 / 334 for σ = 0 / λ₁ / λ₁₀ on hex20 $8^3$, and 231 / 231 / 229 on hex8 $16^3$.
- **Stagnation.** "No new minimum of the max wanted residual (by a factor 0.9) for 50 iterations" alone fires falsely on ordinary runs, because the λ-scaled residual rises while λ̃ falls.

  | run | longest gap without a new residual minimum |
  |---|---|
  | hex20 $6^3$, 1 mode, tol $10^{-10}$ | 85 iterations |
  | hex20 $12^3$ clamped | 140 iterations |
  
  Also counting a decrease of the wanted Ritz-value sum by more than a relative $10^{-10}$ as progress fixes this: the longest gap becomes ≤ 12, and no run is flagged.
- **Compiles.** One step variant (P starts as a masked zero block), so 1 compile per shape, counted with `jax.monitoring` `backend_compile_duration` events filtered by `fun_name`, cache disabled.

## Files

| File | Change |
|---|---|
| `src/jaxmodal/eigen/lobpcg.py` | new: `lobpcg`, `jacobi`, `sigma_estimate`, `lobpcg_step` and helpers |
| `src/jaxmodal/eigen/__init__.py` | export `lobpcg`, `jacobi`, `sigma_estimate` |
| `tests/test_lobpcg.py` | new: the tests below, with the `r_M`, principal-sine and `lmax` helpers (moved to `helpers.py` when 9b reuses them) |
| `docs/eigensolvers.md` | LOBPCG section |
| `docs/ROADMAP.md`, `README.md` | split note, deviations, status, API |

## API

```python
lam, V, info = lobpcg(op, free, n_modes, tol=1e-6, R=None, precond=None, X0=None,
                      maxit=500, seed=0)
```

- `op`: any pytree operator with `apply_K`, `apply_M`, `diag_K`, `diag_M`, `n_dof` (PR 5 or 6). `free`: `(n_dof,)` bool from `dof_mask`. The solver uses the masked operators $K_m v = m \odot K(m \odot v)$ and $M_m$ likewise.
- `R`: `(n_dof, k)` M-orthonormal deflation basis (rigid modes after `m_orthonormalize`), or `None` (stored as a zero-width array). $MR = M_m R$ is computed once. Every new direction and the start block are projected: $W \leftarrow W - R\,(MR^\top W)$.
- `precond`: a **pytree callable** `(n, j) -> (n, j)` (e.g. `jax.tree_util.Partial`), fixed, linear and SPD on the free DOFs, with masked output. It is passed to the jitted step as an argument, so new preconditioner data never recompiles. The default is `jacobi(op, free)`.
- `X0`: `(n_dof, j)` start block, $j \le m$, filled with random columns up to $m$.
- Returns:
  - `lam`: `(n_modes,)` NumPy array, ascending;
  - `V`: `(n_dof, n_modes)` device array, M-orthonormal, constrained rows exactly 0;
  - `info`: a dict of host values (below).
- `info` fields:

  | field | meaning |
  |---|---|
  | `iterations` | step calls |
  | `converged`, `stagnated` | Convergence |
  | `residuals`, `thresholds` | `(n_modes,)`, from explicitly recomputed products |
  | `p_drops` | iterations in which any P column was dropped (noise rule or pivot-test fallback) |
  | `refills` | start-block refill rounds |
  | `norm_K`, `norm_M` | power-iteration estimates |
  | `history` | max wanted residual per iteration |
  | `time` | dict: `setup`, `iterations` (s) |

- Block size $m = n_\mathrm{modes} + \max(5, \lfloor 0.2\,n_\mathrm{modes} \rfloor)$. `ValueError` if $m >$ free DOFs − $k$.
- `jacobi(op, free, sigma=0.0)` returns `Partial(_jacobi, inv)`, i.e. $v \mapsto \mathrm{inv} \odot v$ with $\mathrm{inv} = m / (\mathrm{diag}K + \sigma\,\mathrm{diag}M)$ (0 on constrained DOFs). This is Jacobi on $A_P$ with masked output: the constrained diagonal $s$ of $A_P$ never reaches the output.
- `sigma_estimate(op, free, R=None, maxit=300, seed=0) -> (sigma, iterations)`:
  - the Rayleigh quotient λ̃₁ ≥ λ₁ from the same solver loop with `n_modes=1` and Jacobi (σ = 0);
  - it stops when λ̃₁ has decreased by less than 5% over the last 10 iterations;
  - `R` must be passed for free bodies, or σ ≈ 0 (a rigid mode).

## Algorithm (`lobpcg.py`)

Hetmaniuk–Lehoucq / Duersch et al. 2018, with fixed shapes: every block is $(n, m)$; inactive columns are exact zeros, tracked by a boolean mask.

1. **Setup (host).**
   - $\|K_m\|_2$ and $\|M_m\|_2$ by 30 power iterations each.
   - $MR$ and the default preconditioner.
   - Start block: `X0` and/or random columns, masked, deflated, then SVQB. Columns that SVQB drops are refilled with fresh random vectors and the round repeats, up to 3 refills (`info["refills"]`). If the block is still rank-deficient after that, raise `RuntimeError`.
   - Rayleigh–Ritz on $X$. $P = 0$ with an all-False P mask.
   - The setup's small eager ops (SVQB, Rayleigh–Ritz) are jitted, to avoid about 160 tiny compiles.
2. **SVQB** `svqb(V, KV, MV, act)`, $\tau = 10^{-14}$:
   - zero the inactive rows and columns of $G = V^\top M V$ (symmetrized);
   - scale by $D = \mathrm{diag}(G)^{-1/2}$, using 1 on inactive or zero columns;
   - `eigh`, keep $\theta > \tau\,\theta_\max$, and form $T = D Q\,\mathrm{diag}(\mathrm{keep}/\sqrt\theta)$.
   
   Inactive columns are the exact $\theta = 0$ eigenspace, so kept directions carry nothing from them. It returns the transformed $V, KV, MV$ and the new mask; dropped columns are exact zeros.
3. **Step** `lobpcg_step(state, op, mask, precond, R, MR, consts, recompute)`.
   - A module-level `jax.jit`. The state `(X, KX, MX, P, KP, MP, actP, lam)` is donated. `consts = (‖K‖, ‖M‖, tol, c)` are traced scalars, with $c$ read from `C_FLOOR` on the host at each call, so a monkeypatch takes effect without a recompile and `recompute` is a traced bool. Everything is an argument, nothing is closed over, so there is one compiled variant per shape.
   1. **Residuals.** $R = KX - MX\Lambda$, residuals $r_i$ and thresholds $t_i$ (Convergence). Active columns are $a = r > t$ (soft locking: converged columns get no new direction but stay in $X$).
   2. **New directions.** $W = T(R) \odot a$, deflated with $MR$; then $KW$ and $MW$, the only two applies of width $m$ per iteration.
   3. **Orthogonalize W.** M-orthogonalize against $[X, P]$ ($C = [MX, MP]^\top W$), SVQB, orthogonalize again, SVQB again.
   4. **Rayleigh–Ritz** on $[X, W, P]$.
      - Build $G_K$ and $G_M$ pairwise from $m\times m$ products, without concatenating the blocks.
      - Zero the masked rows and columns off the diagonal, with $G_M = 1$ and $G_K = \beta = 10\max_i|\lambda_i| + \|K\|/\|M\|$ on their diagonal. X is fully active and M-orthonormal, so by Courant–Fischer the lowest $m$ active Ritz values are $\le \max_i\lambda_i < \beta$. Masked slots never enter, and no λ̂_max is needed.
      - Cholesky of $G_M$, then `eigh`; take the lowest $m$.
      - **Pivot test:** if the Cholesky is not finite or $\min\mathrm{diag}(L)^2 \le 10^{-10}$, redo the small problem with the P block masked. That is a `jnp.where` between the two small solutions; the n-sized work is unchanged.
        - This bounds $\lambda_\min(G_M)$ only from above, not the condition number. In practice $G_M \approx I$: the measured minimum pivot is 1.00 and $\max|G_M - I| \le 7\cdot10^{-8}$.
        - The whole P block is masked, and this counts as a drop.
   5. **New P coefficients** (Duersch).
      - Take $C_P$ = the Ritz coefficients with the X rows zeroed.
      - M-orthogonalize $C_P$ against $C_X$ in the $G_M$ inner product, twice, entirely in the small space.
      - A column whose $G_M$-norm falls to $\le 10^{-8}$ of its norm before is masked. This is the column-level noise rule; it counts as a drop.
      - The noise rule is kept. Without it, a pure-noise column ($n_1/n_0 \approx 10^{-12}$) is rescaled to unit norm, and $P'$ picks up about $10^{-4}$ of non-orthogonality to $X'$. The pivot test does not see that ($G_M \approx I$).
      - Step 5's small-matrix work is the helper `_p_coeffs(C_X, C_P, G, act)`, so that test 7c can pin it.
      - SVQB of the rest in coefficient space.
      - The new $[X', P']$ is then M-orthonormal without any n-space orthogonalization of P.
   6. **Update.** $X' = X C_{X,x} + W C_{X,w} + P C_{X,p}$, and $P'$ likewise from $C_P$; $KX'$, $MX'$, $KP'$, $MP'$ by the same GEMMs.
      - For each of $V$, $KV$, $MV$ the update is sequenced: the X products for $X'$ and $P'$, then `optimization_barrier`, then the added W products, another barrier, then the P products. This saves 2 blocks (Measured facts).
      - `lax.cond(recompute, ...)` replaces the products by explicit applies.
   7. **Return.** The new state, and one packed array (residuals, thresholds and λ of $X'$, the drop flag), giving a single device-to-host transfer per iteration.
4. **Outer loop (Python).**
   - One step per iteration, with `recompute = (it + 1) % RECOMPUTE == 0`.
   - A non-finite packed output raises `FloatingPointError`. That is a free check on an array already fetched, and it covers a failed fallback.
   - When the lowest $n_\mathrm{modes}$ meet their thresholds, recompute $KX$ and $MX$ explicitly, write them back into the state, and check again. Exit only if that check passes.
   - An internal `stop(lam_history)` predicate serves `sigma_estimate`.
   - Module constants: `RECOMPUTE = 20`, `C_FLOOR = 10`, `TAU = 1e-14`, `TAU_P = 1e-8`, `STALL = 50`.

9b keeps `lobpcg_step` and changes only around it:
- the deflation $W \leftarrow W - R(MR^\top W)$ becomes the chunked projector over $[R, \text{locked}]$, already a step argument;
- the 6-block mode is a second step variant (the roadmap's "at most two compiled step variants" becomes 9-block and 6-block);
- the outer loop gains chunks and locking.

## Convergence

- **Residual:** the λ-scaled $r_i = \|Kx_i - \lambda_i Mx_i\| / (|\lambda_i|\,\|Mx_i\|)$, on the masked operators.
- **Threshold:** $t_i = \max(\mathrm{tol},\ c\,\varepsilon\,(\|K\| + |\lambda_i|\,\|M\|)\,\|x_i\| / (|\lambda_i|\,\|Mx_i\|))$, with $c = 10$.
- **Outcomes** (the solver never raises on non-convergence; the caller checks `info`):

  | exit | `converged` | `stagnated` |
  |---|---|---|
  | all wanted $r_i \le$ tol (explicitly confirmed) | True | False |
  | all wanted $r_i \le t_i$, some $> $ tol (accepted at the floor) | True | True |
  | no progress for `STALL` iterations | False | True |
  | `maxit` reached | False | False |

  The roadmap's "tol below the floor sets the stagnation flag and exits before the iteration cap" is the second row. The pair of flags tells the cases apart.
- **Progress** is either of two signals:
  - a new minimum of the max wanted residual, by a factor 0.9;
  - a decrease of the sum of the wanted Ritz values by more than a relative $10^{-10}$.
  
  Both are required: the residual clause alone misfires (Measured facts).
- **Theory** (for the docs). Take $x$ M-normalized and the M⁻¹-norm residual $r_M = \|Kx - \lambda Mx\|_{M^{-1}}/|\lambda|$. Then, with the absolute gap:
  - $\sin\theta \le r_M\lambda/\mathrm{gap}$ (Davis–Kahan in the M-inner product);
  - $|\delta\lambda|/\lambda \le r_M^2\lambda/\mathrm{gap}$ (Kato–Temple).
  
  The 2-norm $r$ above is the cheap proxy; it equals $r_M$ up to the conditioning of M.

## Tests (`tests/test_lobpcg.py`)

All tests compare against `reference_modes`, requesting at least 6 modes past the last wanted one so that clusters are complete.
- tol $= 10^{-10}$ unless stated, on 1–8k DOF; fixed seeds.
- $\lambda_\max$ = `eigsh(Kf, 1, Mf, which="LA", tol=1e-3)`.
- $f_\lambda = \max(10^{-8}, 10\,\varepsilon\,\lambda_\max/\lambda)$.
- Principal sines are always computed stably, never as $\sqrt{1 - \cos^2}$ or via `arccos`; that form bottoms out near $10^{-8}$, the size of the expected sines:
  - single mode: $\|x - x_\mathrm{ref}(x_\mathrm{ref}^\top M x)\|_M$, both vectors M-normalized;
  - cluster: the largest singular value of $L^\top(Q - Q_\mathrm{ref}Q_\mathrm{ref}^\top M Q)$, with $M_f = LL^\top$ and $Q$, $Q_\mathrm{ref}$ M-orthonormal.

1. **λ error and principal angles.** Parametrized over clamped hex20 $6^3$, free hex20 $6^3$ (rigid deflation), clamped hex27 $4^3$ and clamped hex8 $10^3$; 12 modes.
   - Modes with relative gap $\ge 10^{-2}$: $|\lambda - \lambda_\mathrm{ref}|/\lambda_\mathrm{ref} \le f_\lambda$.
   - Principal angles: $\sin\theta \le 10\,r_M\lambda/\mathrm{gap}$, with the absolute gap and a dense $r_M$ through the Cholesky of $M_f$. The reference's $r_M$ is asserted $\ge 10\times$ smaller; measured margin 20–25×.
   - Clusters with relative gap $< 10^{-4}$ are compared by subspace, with the gap to the nearest outside eigenvalue.
   - $\|V^\top MV - I\|_\max \le 10^{-10}$, constrained rows exactly 0, $\lambda$ ascending.
   - `info["residuals"]` equals the residuals recomputed in the test from the returned $V$, $\lambda$ with the solver's own operator, $\|K_m v - \lambda M_m v\|/(|\lambda|\,\|M_m v\|)$ through `op.apply_K`/`op.apply_M` and the mask (relative $10^{-12}$).
     - Assembled SciPy matrices differ from it by about $2\cdot10^{-4}$ even for a correct solver, because of round-off at $\varepsilon\lambda_\max/\lambda$.
     - The confirmation-removed mutant differs by $10^{-3}$.
   - `converged` is True and `stagnated` is False.
2. **Multiplicities.** A free square-section beam ($2\times1\times1$, hex20 $6\times3\times3$) and a free cube (hex20 $4^3$).
   - `n_modes` is chosen so that the reference has a relative gap $> 10^{-6}$ after the last wanted mode (asserted).
   - Every reference cluster (relative gap $< 10^{-8}$) gets exactly as many LOBPCG eigenvalues within $f_\lambda$ of it.
3. **Free-free.** In test 1's free case, $\|R^\top M V\|_\max \le 10^{-10}$.
4. **Gap $[10^{-4}, 10^{-2})$.** A clamped beam with section $1\times1.002$; its bending pairs sit at a λ relative gap of about $4\cdot10^{-3}$, which the test asserts lies in the band. The test checks λ error $\le \max(10\,\mathrm{tol}, f_\lambda)$, cluster-wise and sorted.
5. **Stagnation.**
   - (a) tol $= 10^{-16}$ on hex20 $4^3$: `converged` and `stagnated` are both True, `iterations < maxit`, and every wanted residual is ≤ its threshold.
   - (b) With `C_FLOOR` monkeypatched to 0, so the floor is unreachable (tol $10^{-16}$, same mesh; $c$ is a traced argument, so no stale executable): the no-progress exit fires before `maxit`, with `converged = False` and `stagnated = True`.
   - (c) A 1-mode solve on clamped hex20 $6^3$ at tol $10^{-10}$ converges without a flag. This run has an 85-iteration gap without a new residual minimum, so it kills the residual-only rule.
6. **Rank-deficient start.** `X0` with duplicated columns: `refills ≥ 1`, converges, λ match as in test 1.
7. **Forced P drop and the P invariant.**
   - (a) Call `lobpcg_step` on a state whose $P = XC + 10^{-12}\xi$ (random $C$, $\xi$, P mask all True): the drop flag is set and the output is finite. With a random $P$ (M-orthonormalized against $X$), the flag is not set.
   - (b) After 30 steps from a real start on hex20 $4^3$, explicit M applies give $\|P'^\top M X'\|_\max \le 10^{-6}$ and $\|P'^\top M P' - I\|_\max \le 10^{-6}$ on the live columns. Correct code measures $\le 7\cdot10^{-8}$; without the coefficient-space orthogonalization it is about 0.6.
   - (c) `_p_coeffs` returns the coefficients and both masks: the noise-rule live mask, over the original columns, and the SVQB keep mask.
     - Input: $C_P = C_X B + 10^{-12}\xi$ in one column, random elsewhere, with $G = I + 0.1\,\mathrm{sym}(\text{random})$.
     - The live mask is all True except that column, and $k - 1$ columns are kept.
     - The kept columns are $G$-orthonormal and $G$-orthogonal to $C_X$ to $10^{-12}$.
8. **Near-singular M.** hex20 $6^3$ with 30% void elements ($E\cdot10^{-9}$, $\rho\cdot10^{-12}$, seed 0) at tol $10^{-8}$, seed 0: converged within 500 iterations (measured 358), λ error $\le f_\lambda$ against `reference_modes` on the same materials.
9. **Floor** (slow). At tol $10^{-16}$ on the clamped and the free hex20 $4^3$ (clamped measured: ratio 0.9), the lowest attained max residual (`min(history)`) is within $[0.1, 10]\times\varepsilon\lambda_\max/\lambda_1$. The values are recorded in `docs/eigensolvers.md`.
10. **Compiles.** In a fresh subprocess with the cache disabled, count `jax.monitoring` backend-compile events whose `fun_name` contains `lobpcg_step`.
    - One solve: exactly 1. The roadmap's ≤ 2 leaves room for 9b's 6-block variant, and exactly 1 also kills a static `recompute`.
    - A second solve with the same shapes but new data (`op.scaled(...)` with a new `s_K`, a new `jacobi` σ, a new tol) adds 0.
    - `sigma_estimate` runs separately, at $m = 6$, and is counted separately: ≤ 1.
11. **σ pre-estimate.** On clamped hex20 $8^3$ and free hex20 $6^3$ (with `R`): $\lambda_1 \le \sigma \le 2\lambda_1$.
12. **Jacobi.** `jacobi(op, free, s)(v)` with $s = 0.7\,\bar s \ne 0$ equals $m\odot v/(\mathrm{diag}K + s\,\mathrm{diag}M)$ against the assembled diagonals; constrained rows are 0.
13. **Errors.** A block size larger than the free DOFs minus the deflation rank raises `ValueError`.
14. **Step memory.** The step is compiled with a diagonal pytree test operator (`apply_K = d_K ⊙ v`, `apply_M = d_M ⊙ v`, the five-method interface) at $n = 10^5$ and $m = 17$: the same configuration as the measurement, with no operator workspace. Only lowering and compilation run.
    - `memory_analysis()` must give (temp + alias)$/(n m \cdot 8) \le 16.5$, which pins the sequenced update.
    - With a real operator the temp also holds the K/M apply workspace (PR 5: per element chunk), which does not scale with $nm$.
15. **Non-finite step.** A preconditioner returning NaN (`Partial(lambda r: r * jnp.nan)`) makes `lobpcg` raise `FloatingPointError` on the first iteration.

### Mutants (each must fail a test)

| Mutant | Caught by |
|---|---|
| no deflation of W | 3 |
| start block not deflated | 3 |
| mask diagonal $G_K = 0$ instead of β | 6 (spurious zero Ritz values) |
| no refill of dropped start columns | 6 |
| pivot-test fallback removed | 7a (non-finite output; the flag is still set, since it includes the failed pivot test) |
| coefficient-space orthogonalization of $C_P$ removed | 7b (measured 0.6 vs $\le 7\cdot10^{-8}$; tests 1 and 7a do not see it) |
| noise rule removed | 7c |
| floor term removed ($c = 0$) | 5a |
| no-progress exit removed | 5b (runs to `maxit`) |
| λ-sum progress clause removed | 5c |
| explicit confirmation at exit removed | 1 (residuals vs recomputed with `op`) |
| `recompute` made static | 10 (2 compiles instead of 1) |
| operator closed over instead of passed | 10 (second solve recompiles) |
| Jacobi ignores σ | 12 |
| σ stop window of 1 iteration | 11 |
| unmasked K apply | 1 (constrained rows) |
| soft-locking mask removed ($a$ = all) | survivor expected (only cost); noted |
| `optimization_barrier`s removed | 14 (18 blocks) |
| host non-finite check removed | 15 |

## Docs

- **`docs/eigensolvers.md`:** module list in the intro, plus a LOBPCG section with:
  - the `lobpcg` / `jacobi` / `sigma_estimate` signatures, the `info` fields, the block-size rule, the errors (`ValueError` for the block size, `RuntimeError` for a start block that stays rank-deficient, `FloatingPointError` for a non-finite step; non-convergence never raises), and the `R` contract;
  - the algorithm: the steps above, why SVQB zeroes inactive columns, the mask value and the Courant–Fischer argument, the two P levels (the column noise rule and the whole-block pivot-test fallback), implicit updates, the sequenced update;
  - convergence: residual, threshold, the outcome table, both progress clauses, the theory;
  - the σ pre-estimate and its cost, and why the default Jacobi uses σ = 0;
  - the measured floor, the iteration tables, the SIMP case, the per-iteration cost and step memory in blocks, and the thin-plate limit of Jacobi;
  - LOBPCG entries in "Validation";
  - a note that the memory model comes in 9b.
- **`docs/ROADMAP.md`:**
  - the PR 9 split note, with the dependency text above;
  - the 9a status line;
  - the two deviations:
    - **σ = 0 default:** add only the exception "PR 9's default Jacobi uses σ = 0 (iterations change < 3%)" to the general σ > 0 preconditioning rule (line 65). Add "the σ pre-estimate is within 2× of λ₁ and a user hint skips it" to PR 11's Tests, and point PR 9's test clause (line 395) to it.
    - **No λ̂_max in the solver:** the Rayleigh–Ritz decision "A = 10·λ̂_max" becomes "A = 10·max|λ| + ‖K‖/‖M‖". λ̂_max is defined once (at the PR 9 λ-error test) as "λ_max from SciPy `eigsh(which="LA")` on the free blocks (tests only)", so the uses at lines 390, 411, 414, 418, 526 and 536 inherit it;
  - line 74: "at most two compiled step variants (with and without P)" becomes "(9-block and 6-block)";
  - a note that the step's measured memory (about 16 blocks in the 9-block mode) exceeds the budget's 9 + 2. 9b updates the Memory budget rows with its estimator.
- **`README.md`:** the status line and `lobpcg` in the API overview.

## Not covered

- Locking, chunks, the fp32 basis, the 6-block mode, the estimator, memory tests and the memory model: 9b.
- `solve_modes`, examples and budgets: 9c. The σ hint test: PR 11.
- Thin plates: they need the PR 10–13 preconditioners.
