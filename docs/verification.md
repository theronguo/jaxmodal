# Verification

Exact solutions and published benchmarks that check the discretization, with their setups, reference values and measured errors. The eigenvalues come from the certified SciPy reference solver ([`eigensolvers.md`](eigensolvers.md)). The in-test oracles live in `tests/exact.py`; the tests are in `tests/test_verification.py`.

## Error measures and mode matching

- **Targets** are relative errors in frequency, $|f^h/f - 1|$ with $f = \sqrt\lambda/2\pi$; the roadmap's 0.1% in $f$ is about 0.2% in $\lambda$.
- **Rates** are fitted to relative errors in $\lambda$, $|\lambda^h/\lambda - 1|$: the least-squares slope of $\log$ error against $\log h$, per mode, over the stated levels. Richardson extrapolation is not used. The finest error must be $\ge 100 f_\lambda$, with the solver floor $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$, so that round-off cannot bend the fit.
- **Matching.** For the sliding block, the computed modes are compared sorted against the exact sorted spectrum. This is valid because each list is the lowest part of a spectrum: the $k$-th smallest values correspond, whatever the mode shapes. Exactly degenerate exact values pair either way. Benchmarks with labelled modes (8b, 8c) pair by MAC.

## Sliding block

A box $[0, L_x]\times[0, L_y]\times[0, L_z]$ with sliding (roller) BCs on all six faces: $u_x = 0$ on $x = 0, L_x$, likewise for y and z (`SLIDING` in `tests/exact.py`: `{"xmin": "x", "xmax": "x", ...}` for `dof_mask`). The tangential tractions vanish there.

**Exact solution** (`sliding_block(C, rho, L, n, kmax=8)`). For an orthotropic material with axes aligned to the box and $k = \pi(p/L_x, q/L_y, r/L_z)$, the ansatz

$$u_x = A \sin k_x x \cos k_y y \cos k_z z,\quad u_y = B \cos k_x x \sin k_y y \cos k_z z,\quad u_z = C \cos k_x x \cos k_y y \sin k_z z$$

satisfies the BCs ($\sin(p\pi) = 0$; each term of a tangential traction carries the face's sine factor) and reduces the Navier equations to $\Gamma(k)\,(A, B, C)^\top = \rho\lambda\,(A, B, C)^\top$, with the Christoffel matrix $\Gamma_{ik} = C_{ijkl}k_jk_l$:

$$\Gamma = \begin{pmatrix} C_{11}k_x^2 + C_{66}k_y^2 + C_{55}k_z^2 & (C_{12} + C_{66})k_xk_y & (C_{13} + C_{55})k_xk_z \\ \cdot & C_{66}k_x^2 + C_{22}k_y^2 + C_{44}k_z^2 & (C_{23} + C_{44})k_yk_z \\ \cdot & \cdot & C_{55}k_x^2 + C_{44}k_y^2 + C_{33}k_z^2 \end{pmatrix}$$

(Voigt $C_{44}$: 23, $C_{55}$: 13, $C_{66}$: 12).

- **Component rule.** Component $i$ exists only when its index ($p$, $q$ or $r$) is $\ge 1$. Each $(p, q, r) \in \mathbb{N}^3 \setminus \{0\}$ spans an invariant block and contributes the eigenvalues of $\Gamma$ restricted to its existing components.
- **Completeness.** The cosine and sine families are complete on each axis, so the union of the block spectra is the whole spectrum. There are no rigid modes.
- **`kmax` bound.** With $\varepsilon = \mathrm{sym}(v \otimes \hat k)$, $\|\varepsilon\|^2 = \tfrac12(|v|^2 + (v\cdot\hat k)^2) \ge \tfrac12|v|^2$, and the engineering Voigt vector $e$ of $\varepsilon$ has $|e|^2 \ge \|\varepsilon\|^2$. So $v^\top\Gamma(\hat k)v = e^\top C e \ge \lambda_\min(C)|v|^2/2$, and by interlacing every block eigenvalue is $\ge \lambda_\min(C)|k|^2/(2\rho)$. Every mode with an index above `kmax` thus has $\lambda \ge \lambda_\min(C)\,\pi^2(k_\max + 1)^2/(2\rho L_\max^2)$. The $n$-th returned value must lie below this bound, else `ValueError`.
- **Self-check** (isotropic, Lamé $\lambda_L$, $\mu$): for $p, q, r \ge 1$ the block gives $(\lambda_L + 2\mu)|k|^2/\rho$ and $\mu|k|^2/\rho$ (twice); with one index 0, $\mu|k|^2/\rho$ and $(\lambda_L + 2\mu)|k|^2/\rho$; with two indices 0, only $(\lambda_L + 2\mu)|k|^2/\rho$. The test compares the lowest 30 against this closed form.

**Setup.** $L = (1, 0.83, 0.67)$; `box_mesh((n, n, n), spacing=L / n, type=t)`. Two materials: the orthotropic one of the PR 4 tests (`orthotropic(10, 4, 2, 0.3, 0.2, 0.25, 1.5, 1.2, 0.8)`: $E_{1,2,3}$, $\nu_{12,13,23}$, $G_{12,13,23}$; $\rho = 0.7$; relative gaps $\ge 4.6\%$ among the lowest 8 modes) and an isotropic one ($E = 3$, $\nu = 0.3$, $\rho = 2$; exact degeneracies).

**0.1% target** (lowest 10 modes, consistent mass, $n = 8$: 6.7k free DOF for hex20, 13.0k for hex27). Maximum error in $f$:

| element, material | $n = 4$ | 5 | 6 | 8 |
|---|---|---|---|---|
| hex20 orthotropic | 3.3e-3 | 1.3e-3 | 6.4e-4 | 2.0e-4 |
| hex20 isotropic | 5.2e-3 | 2.1e-3 | 1.03e-3 | 3.3e-4 |
| hex27 orthotropic | 3.0e-3 | 1.3e-3 | 6.2e-4 | 2.0e-4 |
| hex27 isotropic | 4.8e-3 | 2.0e-3 | 1.00e-3 | 3.3e-4 |

Modes 7–10 are the worst. $n = 8$ meets the target with a 3× margin; $n = 6$ fails for the isotropic material. The test asserts $\le 10^{-3}$ at $n = 8$.

**Convergence rates** (lowest 6 modes, both materials give the same rates; the test asserts $\ge 1.8$ for the quadratic and $\ge 3.6$ for the quartic cases):

| element, mass | levels $n$ | finest error in $\lambda$ | min. fitted rate |
|---|---|---|---|
| hex8 consistent | 6, 8, 12, 16 | 3.2e-3 … 7.1e-3 | 2.01 |
| hex8 row-sum | 6, 8, 12, 16 | −1.5e-2 … −3.2e-3 | 1.95 |
| hex20 HRZ | 4, 6, 8, 12 | −2.1e-2 … −7.9e-3 | 1.90 |
| hex20 consistent | 3, 4, 6, 8 | 3.3e-5 … 7.3e-5 | 3.95 |
| hex27 consistent | 3, 4, 6, 8 | 3.3e-5 … 7.2e-5 | 3.92 |
| hex27 row-sum | 3, 4, 6, 8 | −1.0e-4 … −1.7e-5 | 3.94 |

- HRZ on levels 2, 4, 6, 8 gives only 1.46–1.93 (pre-asymptotic), hence the finer levels. The finest errors are $\ge 1.2\cdot10^7 f_\lambda$.
- At hex20 $n = 2 \ldots 8$ the sorted rates of the lowest 10 modes scatter (3.68–5.38) where modes cross; the lowest 6 at $n \ge 3$ do not.

**Bounds.** Consistent mass gives upper bounds (Rayleigh–Ritz with a conforming space): at every level $\lambda^h_k \ge \lambda_k - 10\,\varepsilon\,\hat\lambda_\max$, and on nested meshes (hex8 $6 \to 12$, $8 \to 16$; hex20 and hex27 $3 \to 6$, $4 \to 8$) $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\,\varepsilon\,\hat\lambda_\max$; both are tested. Lumped mass converges from below here; there is no theorem, so this is documented, not tested.

## Rayleigh–Ritz parallelepiped, Srinivas–Rao plates (PR 8b)

Not yet implemented.

## NAFEMS FV52 and FV12, Lamb sphere (PR 8c)

Not yet implemented.
