# PR 14a plan: the stochastic eigenvalue count

Status: draft (plan review round 1 pending).

PR 14 (ROADMAP: "Eigenvalue counting and [0, f_max]") is split:
- **14a (this plan):** the completeness count as a building block. It covers the probes (element-wise $M_e$ Cholesky factors), the Chebyshev–Jackson step polynomial and its degree rule, the deflated estimator with preconditioned inner PCG solves, the solve tolerance $\tau_c$ and the unverifiable flag, and the probe-block memory model. Also the counting benchmark: degree, time and error propagation against the mode index.
- **14b (next):** `solve_modes(..., band=(0, f_max), verify=None)`. This covers the [0, f_max] LOBPCG mode (chunks until λ > λ_c, the band-edge cluster rule), counting-edge placement, `verify` with the predicted counting cost, the cost function with the LOBPCG-from-0 selection, and the end-to-end tests against the dense inertia.

14a's own roadmap tests:
- the 97-of-100-seeds detection;
- V complete gives an estimate < 0.5;
- the κ ≥ 10⁷ case;
- the τ_c floor flag and the recorded error propagation;
- the $L L^\top = M_e$ test and the probe covariance test;
- the probe-block memory model test;
- counting cost against mode index.

14b keeps the `verify` tests and the band tests: no miss against the dense inertia, f_max below the first frequency, λ_c inside a double eigenvalue.

## Facts

- The counting operator is $T = (K + \sigma M)^{-1} M$ on the free DOFs. A mode with eigenvalue λ maps to $\mu = 1/(\lambda + \sigma)$, i.e. to $x = 2\sigma\mu - 1 = (\sigma - \lambda)/(\sigma + \lambda) \in (-1, 1]$ for every $\lambda \ge 0$.
  - No spectral bound is needed: λ = 0 (rigid) maps to 1 and λ → ∞ to −1.
  - With σ = λ_c the step sits at x = 0, where the Chebyshev resolution is best.
  - σ = λ_c also maximizes $|dx/d\ln\lambda|$ at λ_c (= 1/2): a relative gap g becomes the interval $[-g/4, g/4]$ in x.
- Estimator, with $\Pi = I - W W^\top M$ (W the deflation basis: computed modes and R, M-orthonormal), $y_j = M^{-1} g_j$ and $g_j \sim N(0, M)$ on the free DOFs:
  - $s_j = \|\Pi q(T) \Pi y_j\|_M^2$ and $s = \frac{1}{N}\sum_j s_j$.
  - In the M-orthonormal eigenbasis Φ (with $M^{-1} = \Phi\Phi^\top$), $\mathrm{Cov}(y) = M^{-1}$ gives $E\,s = \sum_{i \notin W} q(\lambda_i)^2$.
  - One missing in-band mode ($q^2 \ge 0.95$) makes $s \gtrsim 0.95\,\chi^2_N/N$; $P(\chi^2_{50}/50 < 0.5/0.95) \approx 0.3\%$.
- Prototype (scratchpad `p14/poly.py`): Jackson-damped Chebyshev expansion of the indicator of $[x_c, 1]$, $x_c = 0$.
  - Coefficients: $c_0 = \theta_c/\pi$ and $c_k = 2\sin(k\theta_c)/(k\pi)$ with $\theta_c = \arccos x_c$, times the Jackson factors $g_k$.
  - Smallest d with $q^2 \ge 0.95$ on $[0, \lambda_c(1 - g/2)]$ and Weyl tail bias ≤ 0.05:

    | g | 0.5 | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 |
    |---|---|---|---|---|---|---|
    | d | 39 | 108 | 222 | 451 | 1136 | 2279 |

    This is $d \approx 23/g$ (ROADMAP: about 25/g).
  - The in-band condition sets d; the tail bias is ≤ 0.013 at that d.
  - The bias assumes $a = 3/2$, $N_\text{top}$ = 50 computed modes, $n = 10^6$, and W holding only the first mode beyond the cut.
- Dense prototype (`p14/dense.py`, hex20 $6\times5\times4$, clamped, exact solves): n = 1962 free DOFs, cut between modes 10 and 11 (g = 0.0046, a near-double pair, so d = 4989), W = the first 14 modes: W complete gives $s \approx 3\cdot10^{-16}$ (3 seeds; the tail beyond mode 14 lies far outside the transition); mode 3 left out gives $s$ = 0.999, 0.842, 0.999. White-noise solve errors of relative size τ per apply give $|s(\tau) - s(0)|$ = 1.3e-7 / 1.3e-11 / 1.6e-15 at τ = 1e-4 / 1e-6 / 1e-8: they scale as τ², far below the $(\tau d)^2 n$ bound (490 at τ = 1e-4), since q damps most of the spectrum. Real PCG errors are structured (concentrated in low modes), so Design 4 measures them.
- Code to reuse:
  - `lobpcg._project` / `_deflate`: chunked projector over a preallocated basis, upcast per chunk.
  - `krylov.pcg`: column-wise, explicit-residual exit, `converged`/`stagnated` per column.
  - `ap_operator(op, free, σ)`.
  - The preconditioners `gmg` / `amg` / `jacobi` at σ > 0.
  - `matfree.element_matrices` (the per-element $M_e$ formula) and `voxel.reference_element` (the reference $m_e$).

## Design

### 1. Probes: element-wise Cholesky factors (`operators/sampling.py`)

`mass_sample(op, xi)` is linear in the standard-normal input `xi` and returns $g = \sum_e P_e \sqrt{w_e}(L_e \otimes I_3)\xi_e + \sqrt{m_\text{nodal}} \odot \xi_n$ with shape (n_dof, b). Here $L_e L_e^\top = m_e$ is the element's nodal mass matrix ($k/3 \times k/3$; $M_e = w_e\, m_e \otimes I_3$). The Cholesky factor is taken of the nodal block, not the 3× larger $M_e$.

- **Unstructured:** per block, the nodal $m_e$ comes from `element_matrices`' formula:
  - consistent: $\sum_q w\det J\, N_a N_b$;
  - lumped: diag(`lumped_nodal`).

  $w_e = \rho\,s_M$. Padded elements get $m_e = I$ and $w_e = 0$, so the Cholesky factor stays finite. The result is scatter-added into `dofs`.
- **Voxel:** one reference $m_e$ (8×8) and one Cholesky factor. $w_e = \rho\,s_M$ per cell. Corner node ids come from the `node` grid in `reference_element`'s corner order.
- **Nodal masses:** $\sqrt{\texttt{nodal}}\,\xi_n$ (`op.nodal` is per DOF).
- `xi` is a dict of arrays (per block (n_pad, k/3, 3, b), or voxel (n_el, 8, 3, b), plus nodal (n_dof, b)).
  - `mass_sample_keys(op, keys)` draws them per probe (key j → column j), so probes do not depend on the block size.
  - `xi` is about 5 vectors per probe on hex20 and 8 on voxels. It lives only during sampling, before the recurrence's peak, and the memory model counts both phases.
- `mass_factors(op)` (per block, or the voxel reference): $(L_e, w_e)$ for the tests.
- Masking: $g \leftarrow m \odot g$. Then $y = M_P^{-1} g$ by Jacobi PCG on the masked $M_P = m M m + (1 - m)$ to $\tau_c$ (one solve per probe; exact in one step for lumped M).

### 2. Step polynomial and degree (`eigen/counting.py`)

- `jackson_step(d, x_c)` returns the $d + 1$ damped coefficients γ.
- `count_degree(gap, lam, n_free, n_basis, *, q_min=0.95, bias=0.05)` returns `(d, bias_estimate, a)`.
  - It finds the smallest d (doubling, then bisection) with $\min q^2 \ge q_\text{min}$ on $x \in [x(\lambda_c(1 - g/2)), 1]$ and tail bias ≤ `bias`.
  - q is evaluated through $\cos(k\theta)$ on a θ grid with at least $8d$ points (a DCT-II, `scipy.fft`), so the cost is $O(d \log d)$ per trial.
  - Tail: $N(\lambda) = N_\text{top}(\lambda/\lambda_\text{top})^a$ above the largest basis eigenvalue, up to $N = n_\text{free}$. The bias is $\int q^2\,dN$ on a log grid.
    - a is the least-squares slope of $\ln i$ against $\ln\lambda_i$ over the upper half of the computed elastic modes, clipped to [1/4, 3/2] (beams 1/4, plates 1/2, solids 3/2).
    - With fewer than 6 modes, a = 3/2 (the conservative 3D value).
- Counting edge: `edge(lam_last, lam_next)` returns $(\lambda_c, g) = ((\lambda_l + \lambda_n)/2,\ (\lambda_n - \lambda_l)/\lambda_c)$. 14b places the edge; 14a takes `lam_last` and `lam_next`.

### 3. The count (`eigen/counting.py`)

```
count(op, free, W, lam, lam_last, lam_next, *, R=None, precond=None, n_probes=50, seed=0,
      tau=None, block=None, memory=None, basis_dtype="float64", maxit=1000, q_min=0.95, bias=0.05)
  -> (estimate, info)
```

Inputs:
- `W` (n, k): the computed modes, M-orthonormal; NumPy or device array.
- `lam` (k,): their eigenvalues.
- `R`: the rigid modes, as in `lobpcg`.
- `precond`: an SPD preconditioner for $K + \lambda_c M$ (default `jacobi(op, free, λ_c)`); 14b builds the fp64 GMG/AMG at λ_c.

Steps:
1. λ_c, g; d from `count_degree`; $\tau_c$ = `tau` or the default rule (Design 4).
2. W goes into a preallocated basis B, `basis_dtype`, capacity rounded up to the projector chunk c (`lobpcg._capacity`'s rule). Π is `lobpcg._deflate` (W chunks, then R).
3. Probe blocks of size b (Design 5); per block:
   - sample g;
   - $y = M_P^{-1} g$ (Jacobi PCG);
   - $w_0 = \Pi y$, $w_1 = X w_0$, $u = \gamma_0 w_0 + \gamma_1 w_1$;
   - for k = 2..d, $w_{k} = 2 X w_{k-1} - w_{k-2}$ and $u \mathrel{+}= \gamma_k w_k$;
   - $u \leftarrow \Pi u$, $s_j = u_j^\top M u_j$.

   Here $X w = 2\lambda_c\,T w - w$ and $T w$ = PCG on `ap_operator(op, free, λ_c)` with right-hand side $m \odot M(m \odot w)$, to $\tau_c$.
4. One jitted step `_cheb_step(A, P, op, m, w_prev, w, u, gamma_k, tau, maxit)` carries $(w_{k-1}, w_k, u)$ and returns the PCG info. A Python loop over k calls it, one compiled variant per block width.

`info` contents:
- `probes` ($s_j$), `estimate` (= s), `std`;
- `degree`, `gap`, `lam_c`, `tau`, `bias` (estimated), `weyl_a`, `block`;
- `pcg_iterations` (mean and max per solve), `verifiable`, `time` (sampling, recurrence);
- `passed` (= verifiable and s < 0.5).

### 4. Solve tolerance and the unverifiable flag

- **Default rule:** $\tau_c = 0.01/(d\sqrt{n_\text{free}})$ (ROADMAP). An error of relative size $\tau d$ in $q(T)y$ adds about $(\tau d)^2 n$ to $s_j$, since $\|y\|_M^2 \approx n$.
- **Measurement:** the counting benchmark measures the actual error, $|s(\tau) - s(10^{-13})|$ against d at several τ. If the measurement shows growth faster than $O(d)$ (ROADMAP: worst case $O(d^2)$), the rule changes to the measured law before merging, documented.
- **Flag:** any PCG column that stagnates (krylov's `stagnated`) or hits `maxit` without converging sets `verifiable = False`. The estimate is still returned but `passed` is False. This is measured, not predicted: τ_c below the attainable PCG accuracy is detected by the stagnation exit.

### 5. Probe-block memory model (`memory.py`)

`count_memory(op_memory, n, n_basis, b, precond_memory=None, basis_dtype="float64", sample_vectors=...)` returns the lobpcg_memory-style dict: storage (basis $n \cdot \text{cap} \cdot$ bytes, R, MR, the operator and preconditioner storage), the per-phase workspaces, and total.

- **Sampling phase:** `xi` and the scattered contributions, plus g and y.
- **Recurrence phase:** $(w_{k-1}, w_k, u)$, the PCG vectors and the operator and preconditioner workspace at width b, with constants fitted by `memory_analysis` of `_cheb_step` like `GMG_STEP`/`AMG_STEP`.
- **Projector:** the (c + b) columns.
- `solve_modes_memory`-style composition: `count_memory_for(mesh, bcs, n_basis, b, precond=..., ...)`.
- `probe_block(...)` picks the largest $b \le n_\text{probes}$ whose total fits `memory`.
  - `memory=None` means 0.8 × the device limit: `memory_stats()["bytes_limit"]` on GPU, physical RAM on CPU.
  - `block` overrides.

### 6. Counting benchmark (`benchmarks/counting_study.py` → `results/counting_study.json`)

- **Degree and wall time against mode index:** hex20 and voxel boxes, cut between modes k and k + 1 for k up to about 200, W from `lobpcg`. Recorded: d, the PCG iterations per solve, the time per probe block, and the predicted total for 50 probes.
- **Error propagation:** s at τ ∈ {1e-4, 1e-6, 1e-8, 1e-10} against $s(10^{-13})$ for d from 20 to about 2000.
- **Memory:** the memory model against the measured process peak (Design 5).

## Tests (`tests/test_counting.py`)

1. **Element factors:** $L_e L_e^\top = m_e$ to $10^{-12}$ for hex8, hex20 and hex27 (consistent and lumped) and the voxel reference.
2. **Probe covariance:** $S S^\top$ masked equals the masked SciPy M to $10^{-12}$, with S = `mass_sample` applied to unit `xi` columns. Cases:
   - hex20 $2^3$ with nodal masses, two phases and $s_M$ scaling;
   - a mixed hex8/hex20 mesh;
   - lumped;
   - voxel with nodal masses.
3. **Polynomial:**
   - `jackson_step` equals a direct cosine-sum evaluation;
   - $0 \le q \le 1$ (Jackson positivity) on a grid;
   - `count_degree` meets both criteria at d, and d − 1 fails one;
   - $d \cdot g \in [18, 27]$ for g from 0.5 to 0.01.
4. **Estimator against dense:** on a reference-size mesh with exact dense T and the same probes, `count` equals the dense NumPy estimator to $10^{-8}$ relative (τ = 1e-12).
5. **Detection (ROADMAP):** one in-band mode (not next to the cut) left out of W: `passed` is False for ≥ 97 of a fixed list of 100 seeds (deterministic); report the count. W complete: estimate < 0.5 for the same seeds' default.
6. **κ ≥ 10⁷ (ROADMAP):** thin hex20 plate (κ = λ_max/λ_1 ≥ 10⁷, asserted with `eigsh(which="LA")`), sized so `verifiable` is True (asserted), W complete gives estimate < 0.5, with the AMG preconditioner at λ_c.
7. **Floor (ROADMAP):** τ forced to $10^{-17}$ gives `verifiable` False and `passed` False.
8. **Block independence:** b = 7 and b = 50 give the same $s_j$ to $10^{-10}$ (keys per probe).
9. **Memory (ROADMAP, slow):** at a size where `count_memory` ≥ 2 GB, the measured process peak is within 20%.
10. **Benchmark smoke** (slow): `counting_study.run(small=True)`.
11. **Errors:** `lam_next <= lam_last`, `lam_last < 0`, W wrong shape, non-M-orthonormal W (as in `lobpcg`'s R check), n_probes < 1.

## Docs

- `docs/eigensolvers.md`: new section "Completeness count". It covers T and the x map, the estimator and its expectation, the Jackson polynomial and the degree rule (table), the Weyl tail, τ_c and the measured error propagation, the flag, the probe-block memory model, the counting cost against mode index, and the validation list.
- `docs/operators.md`: `mass_sample` and the element factors.
- ROADMAP: the PR 14 split note and the status line.
- `open_items.md`: decisions made without user input.
- README: status line.

## Decisions without user input (to log)

- The 14a/14b split.
- σ = λ_c.
- The Weyl exponent fit and its clipping.
- The default memory budget (0.8 × device limit).
- Detecting unverifiable counts by PCG stagnation.
