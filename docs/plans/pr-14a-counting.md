# PR 14a plan: the stochastic eigenvalue count

Status: draft, revised after plan review rounds 1–3 (log at the end); round 4 pending.

PR 14 (ROADMAP: "Eigenvalue counting and [0, f_max]") is split:
- **14a (this plan):** the completeness count as a building block: probes from element-wise factors of $M_e$, the Chebyshev–Jackson step polynomial and its degree rule, the deflated estimator with preconditioned inner PCG solves, the solve tolerance $\tau_c$ and the unverifiable flag, the probe-block memory model, and the counting benchmark (degree, time and error propagation against mode index).
- **14b (next):** `solve_modes(..., band=(0, f_max), verify=None, count_options=None)`:
  - the [0, f_max] LOBPCG mode (chunks until λ > λ_c, the band-edge cluster rule) and counting-edge placement;
  - `verify` with the predicted counting cost, and the cost function with the LOBPCG-from-0 selection;
  - the probe block from the solve's memory estimate;
  - the count's preconditioner, the fp64 multigrid at λ_c;
  - `count_options` (the `count` keywords), checked by `_check_options`, pinned by PR 13c's signature test, and with `RESERVED` for W, lam, R and precond;
  - the Options table rows for `band`, `verify` and the count defaults;
  - the end-to-end tests: `verify`, no miss against the dense inertia, f_max below the first frequency, λ_c inside a double eigenvalue.

14a carries the roadmap tests on detection, a complete W, κ ≥ 10⁷, the floor flag with the recorded error propagation, $L L^\top = M_e$, probe covariance, the probe-block memory model, and counting cost against mode index.

## Facts

- **The map.** The counting operator is $T = (K + \sigma M)^{-1} M$ on the free DOFs. A mode λ maps to $\mu = 1/(\lambda + \sigma)$, and $X = 2\sigma T - I$ has eigenvalue $x = (\sigma - \lambda)/(\sigma + \lambda) \in (-1, 1]$ for every $\lambda \ge 0$.
  - No spectral bound is needed: λ = 0 maps to 1, λ → ∞ to −1.
  - With σ = λ_c the step sits at x = 0, and $|dx/d\ln\lambda| = 1/2$ there (the maximum over σ), so a relative gap g becomes $[-g/4, g/4]$.
- **The estimator.** Let $\Pi = I - W W^\top M$, with W the computed modes and R (M-orthonormal), and $y_j = M_{ff}^{-1} g_j$ with $g_j \sim N(0, M)$ masked, so $\mathrm{Cov}(y) = M_{ff}^{-1}$.
  - $s_j = \|\Pi q(T)\Pi y_j\|_M^2$, and $E\,s_j = \sum_{i \notin W} q(\lambda_i)^2$.
  - One missing in-band mode ($q^2 \ge 0.95$) gives $s \gtrsim 0.95\,\chi^2_{50}/50$, and $P(\chi^2_{50}/50 < 0.5/0.95) = 0.23\%$ (ROADMAP: 0.2%).
- **Degree.** Prototype `p14/poly.py`; the reviewers' probes re-checked it.
  - Coefficients: $c_0 = 1/2$ and $c_k = 2\sin(k\pi/2)/(k\pi)$ (the indicator of $[0, 1]$ in x), times the Jackson factors $g_k$ (Weiße et al. 2006).
  - The smallest d with $q^2 \ge 0.95$ at $\lambda_c(1 - g/2)$ is $d \approx 23/g$:

    | g | 0.5 | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 |
    |---|---|---|---|---|---|---|
    | d | 39 | 108 | 222 | 451 | 1136 | 2279 |

    The ROADMAP says about 25/g.
  - The in-band minimum of $q^2$ sits exactly at the in-band edge.
  - The Weyl tail (a = 3/2) binds only at large gaps with ≥ 1000 modes below the cut (g = 0.5, N = 1000: 47 against 39), which is not a realistic case. Elsewhere the bias at the in-band d is ≤ 0.013.
- **Dense prototype** (`p14/dense.py`, hex20 $6\times5\times4$ clamped, n = 1962, exact solves):
  - W complete: $s \approx 3\cdot10^{-16}$.
  - Mode 3 left out: s = 0.999, 0.842, 0.999.
  - The Weyl bias estimate against the exact $\sum q^2$ (correctness probe): $4.35\cdot10^{-5}$ against $4.26\cdot10^{-5}$ at cut 7, and $1.19\cdot10^{-6}$ against $1.18\cdot10^{-6}$ at cut 19.
- **Error propagation** (correctness probe, residual-structured errors $A^{-1}r$ with $\|r\|_2 = \tau\|b\|_2$):
  - $|\Delta s|$ is linear in τ: 2.7e-9 / 2.7e-11 / 2.7e-13 at τ = 1e-6 / 1e-8 / 1e-10 (W complete, d = 62), and 7.3e-8 / 7.35e-10 / 7.35e-12 with a mode left out (s ≈ 1.1, d = 129).
  - It enters through the cross term $2\langle u_j, \delta u_j\rangle_M$.
  - Errors along deflated low modes are harmless ($\sim 10^{-18}$).
  - The model: $|\Delta s_j| \lesssim 2\sqrt{s_j}\,\varepsilon + \varepsilon^2$, with ε the accumulated M-norm error, $\varepsilon \lesssim \tau d\sqrt{n}$ in the worst case.
  - `krylov.pcg` stops on the 2-norm relative residual, so the M-norm bound carries a $\sqrt{\kappa(M)}$ factor: $\kappa(D^{-1/2} M D^{-1/2})$ = 144 on uniform hex20, more on graded or contrast meshes.
- **Cost** (performance probe, hex20 $12^3$, n = 24,843):
  - PCG iterations per solve at τ = 1e-6 / 1e-8 / 1e-10 / 1e-12: AMG at λ_c 11 / 14 / 17 / 20, Jacobi 215 / 283 / 351 / 408.
  - One AMG solve at τ = 1e-10 takes 0.44 s at width 8 and 1.22 s at width 32.
  - A count costs about $\lceil N/b\rceil \cdot d \cdot (\text{it}_\text{PCG}(t_A + t_P) + t_M)$: $10^2$–$10^3$ times the LOBPCG solve time at moderate g. So 14b's `verify=None` default rarely turns it on beyond the lowest modes.
- **Memory** (performance probe, `memory_analysis` of one step: M apply, PCG, update), temporaries $8n(a\,b + c)$:
  - Jacobi: a ≈ 10 vectors per probe, c ≈ 150.
  - AMG: a ≈ 21, c ≈ 200.

  The preconditioner's share is large, so the model separates it (Design 5).
- **Code to reuse:**
  - the body of `lobpcg._project` (chunk projector, upcast);
  - `krylov._pcg` (jitted; tol and maxit traced; `converged`/`stagnated` per column);
  - `ap_operator(op, free, σ)`;
  - the preconditioners;
  - `element_matrices`' $m_e$ formula and `ELEMENTS[t].N`;
  - `is_count`.

## Design

### 1. Probes (`operators/sampling.py`)

`mass_sample(op, xi)` is linear in a dict of standard-normal arrays `xi` and returns g (n_dof, b) with $\mathrm{Cov}(g) = M$.

- **Consistent mass, quadrature factor (no Cholesky).** The nodal element mass is $m_e = \sum_q (w\det J)_q N_q N_q^\top = B_e B_e^\top$ with $B_e = N^\top \mathrm{diag}(\sqrt{(w\det J)_q})$, of size $(k/3) \times n_q$. So $M_e = w_e\,m_e \otimes I_3$ has the exact factor $\sqrt{w_e}\,B_e \otimes I_3$ (with $w_e = \rho\,s_M$).
  - Element contribution: $u_e = \sqrt{w_e}\,B_e\,\xi_e$ with $\xi_e$ of shape $(n_q, 3, b)$, scatter-added into `dofs`.
  - Padded elements have $w\det J = 0$, so $B_e = 0$. No identity padding, no breakdown at $s_M = 0$.
  - $B_e$ is recomputed per block from the stored `wdet`, or from `geo[..., 9]` on the fly, and never kept.
  - Voxel: one reference $B$ (hex8 $2^3$ Gauss points, $w\det J = h_x h_y h_z/8 \cdot w_q$), corner node ids from the `node` grid in `reference_element`'s corner order, $w_e$ per cell.
  - Decision (logged): the ROADMAP says "Cholesky factors of $M_e$". The quadrature factor is an exact factor $L L^\top = M_e$, cheaper, and robust; test 1 checks $L L^\top = M_e$.
- **Lumped mass** (any operator; diagonal, nodal masses included): $g = \sqrt{\mathrm{diag}\,M} \odot \xi$ and $y = g/\mathrm{diag}\,M$ exactly (no PCG).
- **Nodal masses** (consistent): $+\sqrt{\texttt{op.nodal}} \odot \xi_n$.
- **Keys.** `mass_sample_keys(op, keys)` draws `xi` per probe (key j gives column j), so probes do not depend on the block size.
  - `xi` holds about 7 vectors per probe on hex20 and 8 on voxels. It exists only in the sampling phase, below the recurrence's 10–21 per probe.
  - Masking: $g \leftarrow m \odot g$.
  - Then $y = M_P^{-1} g$ by Jacobi PCG on $M_P = mMm + (1 - m)$ to $\tau_c$. The constrained diagonal is 1 rather than the ROADMAP's scaled diagonal; g vanishes there, so the result is identical (logged).

### 2. Step polynomial and degree (`eigen/counting.py`)

- `jackson_step(d)` returns the $d + 1$ damped coefficients γ of the indicator of $x \ge 0$.
- **Module constants:** `Q_MIN = 0.95`, `BIAS = 0.05`, `WEYL_A = 1.5`, `MAX_DEGREE = 100_000`. They are acceptance criteria, not options.
- `count_degree(gap, lam_top, n_top, n_free)` returns `(d, bias)`: the smallest passing d found by doubling then bisection. Measured, $q(x_l)$ is monotone in d.
  - **In-band:** $q(x_l)^2 \ge 0.95$ at $x_l = x(\lambda_c(1 - g/2))$, evaluated exactly by an $O(d)$ cosine sum.
  - Also the minimum over a DCT-II θ grid of $8d$ points on $[x_l, 1]$, which guards against ripples.
  - **Tail bias:** $\int q^2\,dN$ with $N(\lambda) = n_\text{top}(\lambda/\lambda_\text{top})^{3/2}$ from $\lambda_\text{top}$ up to $N = n_\text{free}$. It is evaluated on the same uniform θ grid (spacing $\pi/(8d)$, which resolves the transition), with $dN/d\theta$, and $\theta_\text{max} = \theta(\lambda_\text{max})$ is a grid endpoint, where $dN/d\theta$ peaks.
  - **Weyl exponent.** a = 3/2 is fixed: the 3D exponent, an upper bound on fits to the computed modes (bulky 1.1, plate 0.4–0.5, beam 0.3–1.3, correctness probe). This is stricter than the ROADMAP's "a fitted to computed modes". The ROADMAP wording is updated and the change logged.
  - **Degree cap.** If $23/g >$ `MAX_DEGREE`, the search is skipped and `count` returns without running: `estimate` NaN, `verifiable` and `passed` False, `degree` = $\lceil 23/g\rceil$, `steps` 0 (the ROADMAP's cost-model cap; 14b's cost model may lower it, and it is a candidate `count_options` key).
- **Inputs.** `count` computes $\lambda_c = (\lambda_l + \lambda_n)/2$ and $g = (\lambda_n - \lambda_l)/\lambda_c$ from `lam_last` and `lam_next`.
  - `lam_last` may be the band edge $(2\pi f)^2$ when no mode is returned.
  - `lam_top` = max(lam); `n_top` = len(lam) + R columns.

### 3. The count (`eigen/counting.py`, exported from `jaxmodal.eigen`)

```
count(op, free, W, lam, lam_last, lam_next, *, R=None, precond=None, n_probes=50, seed=0,
      tau=None, degree=None, block=None, maxit=1000) -> (estimate, info)
```

**Inputs:**
- `W` (n, k): the computed modes, M-orthonormal, NumPy or device, fp64 or fp32 (the host copy `lobpcg` returns); `lam` (k,) their eigenvalues.
- `R`: the rigid modes.
- `precond`: an SPD preconditioner for $K + \lambda_c M$ (default `jacobi(op, free, λ_c)`). The ROADMAP's fp64 V-cycle is 14b's choice, and the docstring says so.
- `degree`: overrides `count_degree` (tests, benchmark).
- `block`: probes per block (default `n_probes`). A ragged last block is filled with extra probes (keys j ≥ N) whose $s_j$ are discarded, so there is one compiled width.

**Checks:**
- $0 \le$ `lam_last` < `lam_next` ≤ max(lam), since the tail model needs every mode up to `lam_next` in W;
- W shape, lam length, and M-orthonormality by a streamed sketch (two passes over W, not the $O((k/c)^2)$ Gram): $v = Wz$ with $z \sim N(0, I_k)$ of 3 columns from a key derived from `seed`, $e = W^\top M v - z$, $\|e\| \le \text{tol}\,\|z\|$ with tol $10^{-6}$ for fp64 W and $10^{-4}$ for fp32 W;
- `n_probes`, `block`, `maxit` pass `is_count`;
- τ > 0.

**Π: streamed, no basis copy.** `_project_chunks(op, m, W, Y, c=64)` applies Π to Y (n, b):
- MY once;
- then W in column chunks of width c (host-to-device per chunk, upcast) through one jitted kernel `Y -= Wc (Wcᵀ MY)` (the `_project` body);
- then R.

It is applied at the start and end of each block. Only one chunk of W is on the device at a time.

**Per block:**
1. Sample g; y by Jacobi PCG (Design 1). A y column that stagnates or does not converge sets `bad` too.
2. $w_0 = \Pi y$.
3. `_recurrence(A, P, op, m, w0, gamma, tau, maxit)` (jitted, one compile per width) runs $w_1 = X w_0$, $w_k = 2Xw_{k-1} - w_{k-2}$, $u = \sum_k \gamma_k w_k$ in a `lax.while_loop`.
   - $Xw = 2\lambda_c\,T w - w$, with $Tw$ = `_pcg(ap_operator(op, free, λ_c), P, m ⊙ M(m ⊙ w), tau, maxit)`.
   - The carry holds the PCG iteration sum and max and a `bad` flag (any column stagnated or not converged).
   - The loop stops early when `bad` is set: an unverifiable block need not run d stagnating solves. The ROADMAP's 10M-near-index-1000 case expects that. The remaining blocks are skipped, and `estimate` is NaN (a truncated sum means nothing).
   - γ is zero-padded to the next power of two and d is a traced loop bound, so one compile serves every degree up to that size.
4. $u \leftarrow \Pi u$, $s_j = u_j^\top M u_j$.

Returns `estimate` = mean $s_j$ and `info`:
- `probes` ($s_j$; NaN for columns of a stopped or skipped block), `std`;
- `degree`, `gap`, `lam_c`, `tau`, `bias`, `block`;
- `pcg_iterations` (mean per solve, max), `steps` (recurrence steps run);
- `verifiable`, `passed` (= verifiable and estimate < 0.5);
- `time` (sampling, recurrence, per step).

### 4. Solve tolerance and the unverifiable flag

- **Default** $\tau_c = 0.01/(d\sqrt{n_\text{free}})$ (ROADMAP). With the linear model of Facts, $|\Delta s| \lesssim 0.02\sqrt{\kappa(M)\,s}$ in the worst case (about $0.24\sqrt{s}$ on uniform hex20). The measured error is far smaller (Facts), and the benchmark's contrast case checks it.
- **The benchmark measures** $|s(\tau) - s(10^{-13})|$ against d and τ. Cases:
  - W complete;
  - a mode left out ($s = O(1)$);
  - a graded mesh with density contrast ($\kappa(M)$ larger).

  Also recorded: PCG iterations against τ.
- **If the measured law differs** from the rule by more than 10× in either direction, the rule changes to the measured law with a 10× safety factor before merging, documented. Loosening is allowed: a needlessly strict τ_c hits the fp64 floor and gives a spurious "unverifiable".
- **Flag** (measured, not predicted): any PCG column that stagnates or hits `maxit` without converging sets `verifiable = False`, and the estimate is still returned. Above `MAX_DEGREE` the count is also unverifiable. τ_c below the attainable accuracy shows up as the stagnation exit; the docs record the floor measured by the benchmark on the test mesh.

### 5. Probe-block memory model (`memory.py`)

`count_memory(op_memory, n, b, precond_memory=None, n_rigid=0, chunk=64, w_dtype="float64")` returns a lobpcg_memory-style dict.

- **storage:** the operator and preconditioner storage, R and MR.
- **sampling:** about 7–8 vectors per probe, below the recurrence. Stated, and checked by test 9b.
- **step:** $8n(a_c b + c_c)$ (`COUNT_STEP`, fitted by `memory_analysis` of `_recurrence` with Jacobi), plus the operator workspace and the preconditioner's step term at width b (`gmg_memory`'s workspace, `amg_memory`'s step), as in `lobpcg_memory`.
- **projector:** a chunk of W, $n \cdot c \cdot$ bytes, plus its fp64 upcast $8nc$ when `w_dtype` is fp32, plus (c + b) temporaries during Π.
- **total:** storage + max(phases).

`probe_block(budget, ...)` gives b in closed form. The total is affine in b, so $b = \min(n_\text{probes}, \lfloor(\text{budget} - \text{fixed})/\text{per probe}\rfloor)$. If b < 1 it raises. 14b passes the budget (the solve's own estimate, so counting never raises the peak). 14a's `count` takes an explicit `block`.

### 6. Counting benchmark (`benchmarks/counting_study.py` → `results/counting_study.json`)

- **Degree and cost against mode index:** hex20 $16^3$ and voxel $32^3$ boxes, W from `lobpcg` (≈ 200 modes).
  - For each cut k: g, d (cheap).
  - At 6 cuts (k = 5, 10, 25, 50, 100, 200): S = 30 recurrence steps timed (`degree=S`, AMG/GMG at λ_c, b = 10) and PCG iterations per solve. The other cuts use the nearest timed $t_\text{step}$. The predicted total $\lceil 50/b\rceil \cdot d \cdot t_\text{step}$ is reported against the LOBPCG solve time. `run(small=True)` times 2 cuts.
- **Error propagation:** full counts at small n (hex20 $6\times5\times4$, and a graded box with a density-contrast phase) for d from 20 to about 1000. W complete and a mode left out, τ ∈ {1e-4, …, 1e-12} against τ = 1e-13.
- **Floor:** the smallest τ at which PCG converges on the test mesh, for the docs (test 8 measures its own).
- **Memory:** `count_memory` against the measured process peak at the test-9 size.

## Tests (`tests/test_counting.py`)

Fast tests use the reference mesh hex20 $6\times5\times4$ (n = 1962). The cut is after mode 6 (g = 0.41, d ≈ 56), and the precond is an exact dense Cholesky pytree (PCG in 1–2 iterations), unless the test is about the preconditioner.

1. **Element factors (ROADMAP):** $(\sqrt{w_e}B_e \otimes I_3)(\cdot)^\top = M_e$ from `element_matrices` to $10^{-12}$ (hex8, hex20, hex27; padded elements give 0). Voxel: the same against `reference_element`'s $m_e$ times $w_e$.
2. **Probe covariance (ROADMAP):** $S S^\top$ masked equals the masked SciPy M to $10^{-12}$, with S = `mass_sample` on unit `xi` columns. Cases:
   - hex20 $2^3$ with nodal masses, two phases and $s_M$ scaling;
   - mixed hex8/hex20;
   - lumped unstructured;
   - voxel with nodal masses;
   - voxel lumped.
3. **Polynomial:**
   - `jackson_step` against a direct cosine sum;
   - $q \ge -10^{-15}$ and $q \le 1 + 10^{-15}$ on a grid;
   - `count_degree` meets $q(x_l)^2 \ge 0.95$ (direct sum) at d and fails at d − 1;
   - $d \cdot g \in [19, 24]$ at the poly.py inputs ($n_\text{free} = 10^6$, $n_\text{top} = 50$);
   - a case where the bias sets d (g = 0.5, $n_\text{top} = 5000$): d > the in-band d, and the bias ≤ 0.05;
   - $23/g >$ `MAX_DEGREE`: estimate NaN, `verifiable` and `passed` False, `degree` = $\lceil 23/g\rceil$, `steps` 0.
4. **Expected value against the exact eigen-sum:** 2000 probes (one `count` call, `n_probes = 2000`, `block = 500`), mean $s_j$ against $\sum_{i \notin W} q(\lambda_i)^2$ from `scipy.linalg.eigh(K, M)` and a direct cosine-sum q, within 5 standard errors. Cases:
   - W complete;
   - mode 3 left out (sum ≈ 1);
   - a free body with R (k = 6), plus R omitted from the call (sum ≈ 6);
   - lumped mass, and consistent mass with nodal masses, mode 3 left out (the diagonal y path and the nodal term through `count`).
5. **Estimator against dense, same probes:** the dense NumPy estimator with exact T. τ = 1e-12: mode left out, relative $10^{-8}$; W complete, absolute $10^{-10}$. Default τ: within the linear model's bound. Cases: hex20, lumped hex20, and a voxel box ($4^3$).
6. **Detection (ROADMAP):** mode 3 (not next to the cut) left out of W. One `count` with `n_probes = 5000`, `block = 500`, `info["probes"]` grouped into 100 consecutive sets of 50: ≥ 97 sets have mean ≥ 0.5. Deterministic; the count is reported. W complete (slow): all 100 sets < 0.5 (Jackson bias included).
7. **κ ≥ 10⁷ (ROADMAP):** slender hex20 cantilever ($64 \times 1 \times 1$, element side 1/64, xmin clamped, n = 2304) with κ = λ_max/λ_1 ≥ 10⁷ asserted (`eigsh(which="LA")`; measured 2.2e9). W from `scipy.linalg.eigh(K, M)` on the free DOFs, cut in the gap after mode 4 (g ≈ 1.55, d ≈ 15, τ_c ≈ 1.4e-5). AMG at λ_c (measured 10–11 PCG iterations per solve), default τ; `verifiable` True asserted; W complete gives estimate < 0.5.
   - A thin plate is not used: AMG on its flat elements (aspect ≈ 8) needs 400–480 iterations per solve and stagnates at τ ≤ 1e-10 (reviewer probes; the known flat-element open item).
8. **Floor and flags (ROADMAP):**
   - The floor measured inside the test on the count's own right-hand sides: `count(..., degree=1)` (Jacobi at λ_c, same seed) at τ = 1e-10, 1e-11, … downward; the floor is the smallest τ with `verifiable` True (converged, not stagnated). At τ = floor/10 the first recurrence step sees the same w₀ and fails deterministically. That case, and τ = 1e-17: `verifiable` False, `passed` False, estimate NaN, `steps` < d (early stop).
   - `maxit=2` at the default τ with Jacobi: `verifiable` False.
   - The y solve forced to fail (monkeypatched to return non-converged, the recurrence uses the real PCG): `verifiable` False, `steps` 0.
   - A normal run: `verifiable` True.
9. **Memory:**
   - (a) `memory_analysis` of `_recurrence` at width b is ≤ `count_memory`'s step on hex20 $8^3$ and voxel $16^3$ with Jacobi, GMG and AMG, and the fit is tight (≥ 0.6×) at one size.
   - (b) The sampling phase ≤ the step phase.
   - (c) `probe_block` gives the largest b with total(b) ≤ budget < total(b + 1), capped at n_probes, and raises below b = 1.
   - (d) ROADMAP, slow: at a size where `count_memory` ≥ 2 GB (`degree` = 3 steps), the measured process peak is within 20%, and the peak holds no n × k copy of W.
10. **Block independence and compiles:** b = 7 and b = 50 give the same $s_j$ to $10^{-10}$ at τ = 1e-12. `degree=100` then `degree=120` (one size class) compile `_recurrence` once (jit cache size).
11. **fp32 W:** the estimate is within $10^{-4}$ absolute of the fp64-W estimate, and `passed` agrees.
12. **Benchmark smoke (slow):** `counting_study.run(small=True)`.
13. **Errors:**
    - lam_next ≤ lam_last, lam_last < 0, lam_next > max(lam);
    - W shape, lam length, non-M-orthonormal W (one column scaled by 1.01);
    - n_probes = 0, True or 2.5;
    - τ ≤ 0.

## Docs

- `docs/eigensolvers.md`, new section "Completeness count":
  - T and the x map, the estimator and its expectation;
  - the Jackson polynomial and degree rule (table, $d \approx 23/g$), the Weyl tail with a = 3/2, the degree cap;
  - τ_c and the measured error propagation (linear model) and floor, the flag;
  - streamed Π, the probe-block memory model, the counting cost against mode index;
  - each `count` keyword's default and why it is robust (50 probes: P(miss) = 0.23%; τ rule; block; maxit), for 14b's Options table;
  - the "kept internal" sentence gains `Q_MIN` and `BIAS` (acceptance criteria), `WEYL_A` (the 3D bound), and `MAX_DEGREE` (cost cap; a candidate 14b `count_options` key);
  - the validation list.
- `docs/operators.md`: `mass_sample` and the quadrature factor.
- ROADMAP:
  - "Cholesky factors of $M_e$" → "quadrature factors of $M_e$" (the test, the sampling text and the validation list);
  - the PR 14 split note, with each PR 14 test tagged 14a / 14b;
  - dependency rows "14a Counting | 9, 10, 11 (13 optional)" and "14b [0, f_max], verify | 14a";
  - the d ≈ 25/g note (measured 23/g), the a = 3/2 wording, the status line.
- `open_items.md`: decisions without user input.
- README: status line and a short `count` example (exported building block).

## Decisions without user input (to log)

- The 14a/14b split.
- σ = λ_c.
- The quadrature factor instead of Cholesky, and the lumped diagonal shortcut.
- Constrained diagonal 1 in $M_P$.
- a = 3/2 fixed (stricter than a fit).
- The degree search: smallest passing d, in-band at the exact edge.
- `MAX_DEGREE` = 1e5.
- Default precond Jacobi at λ_c in 14a.
- The τ_c rule may change by measurement in either direction.
- The memory budget moves to 14b.
- Early stop of unverifiable blocks; NaN estimate after an early stop.
- Detection probes grouped as 100 sets of 50 from one 5000-probe call: probes are i.i.d. per key, so this equals 100 independent 50-probe counts.
- The M-orthonormality check by a random sketch, not the full Gram.

## Review log

- **Round 1** (all 5 NOT SATISFIED). Changes:
  - Π streamed over chunks of the caller's W, with no device basis copy (perf B1, simplicity S3).
  - Fixed a = 3/2 (simplicity S1).
  - Exact in-band evaluation at the edge, and the tail on a $\pi/(8d)$ θ grid (correctness S1).
  - Linear error model and measurement at $s = O(1)$ and on a contrast mesh; τ rule in either direction (correctness S2/S3, perf S5).
  - Quadrature factor, lumped shortcut, factors not stored (correctness S4/S5).
  - `MAX_DEGREE` and the `degree` override (perf S1/S3).
  - Memory model split into the recurrence fit plus preconditioner terms; closed-form `probe_block`; budget moved to 14b (perf S4, simplicity S2).
  - One jitted while loop with early stop (perf N1, correctness N3).
  - Tests: expected value against the eigen-sum, rigid modes, the maxit flag, non-slow memory tests, sizes and fast preconditioners, grouped detection probes (tests S1–S5, perf S2, correctness S6).
  - `lam_next ≤ max(lam)` (correctness S7).
  - 14b scope gains `count_options` and Options rows (docs S1).
  - Constants instead of options for q_min/bias (simplicity N4); no duplicate estimate (N3); `jackson_step(d)` (correctness N6); and the remaining NITs.
- **Round 2** (tests, simplicity, correctness, performance SATISFIED; docs NOT SATISFIED). Changes:
  - The τ_c bound carries $\sqrt{\kappa(M)}$ (docs S1).
  - Streamed sketch for the W check with a dtype-dependent tolerance (perf S1, correctness N3).
  - `block = 500` in tests 4 and 6 (perf S2, simplicity N).
  - Benchmark times 6 cuts (perf S3).
  - γ padded to a power of two, one compile per size class (perf N1); fp32 upcast in the projector term (perf N2); test 7 mesh and gap (perf N3).
  - y solve in `bad`; NaN and skipped blocks after an early stop; θ_max on the grid (correctness N1/N2).
  - `MAX_DEGREE` exit values (simplicity N).
  - Lumped and nodal paths through `count` in test 4, voxel in test 5, test 8 measures its own floor (tests N).
  - ROADMAP Cholesky wording, internal constants listed, the probe grouping logged, cost units (docs N).
- **Round 3** (simplicity, tests, performance SATISFIED; docs NOT SATISFIED; correctness NOT SATISFIED). Changes:
  - Test 7: slender hex20 cantilever 64×1×1 (κ 2.2e9, AMG 10–11 iterations), W from dense `eigh` (docs S1, correctness S1: AMG struggles on the aspect-8 plate).
  - Test 8's floor measured by `count(degree=1)` on the same probes, so floor/10 fails deterministically (correctness N1).
  - Fast-test cut after mode 6 (d ≈ 56), test 6's W-complete half slow (perf N1).
  - `probes` NaN for stopped blocks (simplicity N2).
  - Seeded sketch, 1.01 perturbation, all `MAX_DEGREE` exit values, a y-solve-only failure test, the one-compile check (tests N1–N4).
  - Not taken: a single in-band criterion (simplicity N1, optional); dropping τ = 1e-17 in test 8 (simplicity N3, optional; both cases kept).
