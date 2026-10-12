# Verification

Exact solutions and published benchmarks that check the discretization, with their setups, reference values and measured errors. The eigenvalues come from the certified SciPy reference solver ([`eigensolvers.md`](eigensolvers.md)). The in-test oracles live in `tests/exact.py`; the tests are in `tests/test_verification.py`.

## Error measures and mode matching

- **Targets** are relative errors in frequency, $|f^h/f - 1|$ with $f = \sqrt\lambda/2\pi$; the roadmap's 0.1% in $f$ is about 0.2% in $\lambda$.
- **Rates** are fitted to relative errors in $\lambda$, $|\lambda^h/\lambda - 1|$: the least-squares slope of $\log$ error against $\log h$, per mode, over the stated levels. Richardson extrapolation is not used. The finest error must be $\ge 100 f_\lambda$, with the solver floor $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$ ($\hat\lambda_\max$ from `eigsh(which="LA", tol=1e-3)`: an order-of-magnitude estimate suffices, and the default tolerance costs about 20 s on a 30k-DOF mesh), so that round-off cannot bend the fit.
- **Matching.** For the sliding block, the computed modes are compared sorted against the exact sorted spectrum. This is valid because each list is the lowest part of a spectrum: the $k$-th smallest values correspond, whatever the mode shapes. Exactly degenerate exact values pair either way. The Visscher box is compared the same way from mode 7 (generic anisotropy, relative gaps $\ge 10^{-2}$). Srinivas–Rao (8b) and FV52 (8c) pair by M-weighted MAC with labels, degenerate sets by summed MAC (see "Mode matching" below).

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

**Convergence rates** (lowest 6 modes, both materials give the same rates; the test asserts $\ge 3.6$ for hex20 and hex27 consistent and $\ge 1.8$ otherwise, the roadmap's bounds; hex27 row-sum measures 3.94):

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

## Simply supported plates (Srinivas–Rao)

**Simply supported BCs on solids** (`SIMPLY_SUPPORTED` in `tests/exact.py`, reused by PR 16). For a plate $[0, a]\times[0, b]\times[0, h]$, face-wise over each whole edge face, through the thickness: $u_y = u_z = 0$ on $x = 0, a$ and $u_x = u_z = 0$ on $y = 0, b$; top and bottom are free. In `dof_mask` form: `{"xmin": "yz", "xmax": "yz", "ymin": "xz", "ymax": "xz"}`. The normal stress on the edge faces is free. This is exactly the boundary condition of the exact solution, and no rigid modes remain.

**Exact solution** (`plate_modes(plies, a, b, k, points=None, P=12, mmax=8)`; Srinivas et al. 1970 (Srinivas, Joga Rao & Rao), J. Sound Vib. 12, 187–199). `plies` are `(C, rho, t)` stacked from $z = 0$, each $C$ orthotropic in the plate axes (a 90° ply is `rotate(C, Rz90)`). With $p = m\pi/a$, $q = n\pi/b$, $m, n \ge 0$ not both 0, the ansatz

$$u = U(z)\cos px \sin qy,\quad v = V(z)\sin px\cos qy,\quad w = W(z)\sin px \sin qy$$

satisfies the BCs (the normal stresses $\sigma_{xx}$, $\sigma_{yy}$ carry $\sin px\sin qy$ and vanish on the edge faces where needed), and the energy separates per $(m, n)$ for plies aligned with the plate.

- **Component rule** (as for the sliding block): $U$ exists iff $n \ge 1$, $V$ iff $m \ge 1$, $W$ iff $m, n \ge 1$. The cosine/sine families are complete on each in-plane axis, so the union over $(m, n)$ is the whole spectrum.
- **1D Ritz in z.** Strains in Voigt order $e = (-pU, -qV, W', V' + qW, U' + pW, qU + pV)$; $\int e^\top C e\,dz = \lambda\int\rho(U^2 + V^2 + W^2)\,dz$ (the in-plane factors cancel). GLL nodal basis of degree $P$ per ply, $C^0$ across ply interfaces, $P + 3$ Gauss points, dense `eigh`. The family index counts the modes within one $(m, n)$; labels are $(m, n, \text{family})$. With `points`, `Phi` holds the shapes in DOF order $3n + i$.
- **Validation.** Degree 12 and 24 agree to $1.1\cdot10^{-10}$ (tested to $10^{-9}$); the $(0, 1, 0)$ mode equals $C_{66}\pi^2/(\rho b^2)$ ($U$ constant through the thickness, tested). A prototype transfer-matrix determinant (state $(U, V, W, \tau_{xz}, \tau_{yz}, \sigma_z)$, free top and bottom) changes sign within $\lambda(1 \pm 10^{-9})$ at all 20 Ritz eigenvalues tried; it is not in the tests.
- **Truncation is a check, not a bound.** Bending modes scale like $h^2|k|^4$ and the Rayleigh quotient has mixed-sign $p$, $q$ cross terms, so the sliding block's Christoffel bound does not carry over. Measured on $m, n \le 12$ (both plates and a plate with $h = 0.01$): $\lambda_\min(m, n)$ is monotone in $m$ and $n$ for $m, n \ge 1$, and the $m = 0$, $n = 0$ rows lie 1.9–790× above the $m = 1$, $n = 1$ rows. `plate_modes` raises unless the smallest eigenvalue on the boundary rows $\max(m, n) = m_\max$ exceeds $2\lambda_k$ (measured 6.4–9.7× $\lambda_{10}$ at $m_\max = 8$), and the test checks that $m_\max = 10$ returns the same lowest 10.

**Setup.** $a = b = 1$, $h = 0.1$ ($L/t = 10$). ORTHO: the sliding block's orthotropic material as one ply, $\rho = 0.7$. PAG $[0/90/0]$: `orthotropic(25, 1, 1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.2)` ($E_1/E_2 = 25$), $\rho = 1$, three equal plies, the middle one rotated by 90° about $z$. Meshes `box_mesh((n, n, n_z), spacing=(a/n, b/n, h/n_z))` with ply faces on element faces; the element material is the ply containing the element centroid.

**Reference values** (lowest 10, `plate_modes`, $P = 12$):

| mode | ORTHO label | $\lambda$ | PAG label | $\lambda$ |
|---|---|---|---|---|
| 1 | (1,1,0) | 2.367845 | (1,1,0) | 1.312709 |
| 2 | (1,2,0) | 10.053452 | (1,2,0) | 3.316784 |
| 3 | (2,1,0) | 17.035882 | (0,1,0) | 4.934802 |
| 4 | (0,1,0) | 21.149152 | (1,0,0) | 4.934802 |
| 5 | (1,0,0) | 21.149152 | (2,1,0) | 7.942125 |
| 6 | (2,2,0) | 28.579488 | (1,3,0) | 9.341781 |
| 7 | (1,3,0) | 31.749378 | (2,2,0) | 10.170902 |
| 8 | (2,3,0) | 54.593566 | (2,3,0) | 16.432516 |
| 9 | (3,1,0) | 58.557304 | (2,0,0) | 19.739209 |
| 10 | (1,1,1) | 65.009811 | (0,2,0) | 19.739209 |

Exactly degenerate (equal $C_{66}$ and $\rho$ in every ply, square plate): ORTHO modes {4, 5}; PAG {3, 4} and {9, 10} ($4C_{66}\pi^2/\rho$). The FE pairs are degenerate to $10^{-12}$.

**0.1% target** (lowest 10, consistent mass; maximum error in $f$, all errors positive):

| plate | mesh $(n, n, n_z)$ | hex20 | hex27 |
|---|---|---|---|
| ORTHO | 12, 12, 4 | 1.00e-3 | 9.9e-4 |
| ORTHO | 16, 16, 4 (tested) | 4.2e-4 (14.1k free DOF) | 4.2e-4 (27.1k) |
| PAG | 12, 12, 3 per ply | 9.4e-4 | 9.3e-4 |
| PAG | 16, 16, 3 per ply (tested) | 3.9e-4 (29.0k) | 3.9e-4 (57.1k) |

One element through the thickness stalls at 3.4% (ORTHO, $n = 12$): the thickness direction dominates until about 4 elements per ply.

**Convergence rates** (lowest 6; rate levels $n\times n\times(n/4 \text{ per ply})$, so every direction is refined; the test asserts $\ge 3.6$ for hex20/hex27 and $\ge 1.8$ for hex8):

| plate, element | levels $n$ | finest error in $\lambda$ | min. fitted rate |
|---|---|---|---|
| ORTHO hex8 | 8, 12, 16, 24 | 1.4e-3 … 3.6e-2 | 1.88 |
| ORTHO hex20 | 4, 8, 12, 16 | 2.1e-6 … 3.5e-4 | 3.92 |
| ORTHO hex27 | 4, 8, 12, 16 | 2.1e-6 … 3.5e-4 | 3.88 |
| PAG hex20 | 4, 8, 12 | 6.5e-6 … 1.9e-3 | 3.85 |
| PAG hex27 | 4, 8, 12 | 6.5e-6 … 1.9e-3 | 3.77 |

- The ORTHO hex8 margin is small (1.88 against 1.8; mode 3).
- hex8 on PAG is not tested: on levels 8, 12, 16, 24 modes 1–5 give $\ge 1.97$ but mode 6 only 1.64 (4.7% error at 32k DOF, pre-asymptotic).
- Refining only in-plane at fixed $n_z$ gives 2.3–2.7 for hex20 (the thickness error dominates).
- The finest errors are $\ge 2.7\cdot10^4 f_\lambda$.

**Bounds** (tested): at every level $\lambda^h_k \ge \lambda_k - 10\,\varepsilon\,\hat\lambda_\max$; on nested levels (ORTHO hex20/hex27 $4 \to 8$, $8 \to 16$; ORTHO hex8 $8 \to 16$, $12 \to 24$; PAG hex20/hex27 $4 \to 8$) $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\,\varepsilon\,\hat\lambda_\max$.

## Anisotropic free box (Visscher Ritz)

**Reference** (`visscher(C, rho, L, N, k=16)`; Visscher et al. 1991, J. Acoust. Soc. Am. 90, 2154–2162). Rayleigh–Ritz on the free box $[0, L_x]\times[0, L_y]\times[0, L_z]$ for any $C$, with the basis $P_a(\xi)P_b(\eta)P_c(\zeta)$, $a + b + c \le N$, for each displacement component ($P_a$ Legendre on the box mapped to $[-1, 1]^3$). $K$ and $M$ come from separable 1D Gauss–Legendre integrals ($N + 2$ points, exact).

- **Parity split.** Point inversion about the centre, $u(x) \mapsto -u(-x)$, maps the strain at $x$ to the strain at $-x$, so the problem is invariant for any $C$. Each basis function maps to $-(-1)^{a+b+c}$ times itself, so the basis splits by the parity of $a + b + c$ into two uncoupled blocks. Legendre orthogonality makes $M$ diagonal, $M = \operatorname{diag}(m)$, so each block is the standard problem $m^{-1/2}Km^{-1/2}$ (`eigh` with `subset_by_index`). The result equals the full generalized problem to $6\cdot10^{-12}$.
- **Convergence** ($L = (1, 0.83, 0.67)$, `rotate(ORTHO, R)` with the PR 4 rotation, fully anisotropic, $\rho = 0.7$):

  | $N$ | basis | time | max change in modes 7–16 |
  |---|---|---|---|
  | 16 | 2907 | 0.35 s | 6.3e-6 (from 14) |
  | 18 | 3990 | 0.8 s | 2.2e-6 (from 16) |
  | 20 | 5313 | 1.7 s | 8.4e-7 (from 18) |

  Convergence is algebraic (about 2.6× per 2 degrees), so $N = 20$ is the first degree meeting the roadmap's rule (change $< 10^{-6}$ for modes 7–16, tested); the remaining error is about $5\cdot10^{-7}$. The six rigid eigenvalues are $\le 10^{-11}\lambda_7$ (tested $\le 10^{-8}$).
- **Reference values**, modes 7–16 at $N = 20$ (uncertainty $\approx 5\cdot10^{-7}$ relative): 13.809954, 25.099289, 27.976500, 29.207402, 30.521183, 36.202710, 37.862134, 39.583096, 40.305630, 41.812027. The smallest relative gap is $1.8\cdot10^{-2}$.

**FE comparison** (same box, `box_mesh((n, n, n))` with orientation $R$; modes 7–16 sorted; maximum error in $f$, all positive): hex20 $n = 4, 6, 8$: 3.3e-3, 7.4e-4, 2.5e-4; hex27: 3.1e-3, 7.1e-4, 2.4e-4. The test asserts $\le 10^{-3}$ at $n = 8$, $\lambda^h \ge \lambda_\text{Ritz}(1 - 10^{-6})$ (the Ritz value is an upper bound within its error), and monotonicity for hex20 $4 \to 8$.

## Mode matching

- **MAC** (`mac(A, B, M)` in `tests/exact.py`, M-weighted, FE consistent $M$): $\text{MAC}_{ij} = (a_i^\top M b_j)^2/((a_i^\top M a_i)(b_j^\top M b_j))$. PR 17's library version is checked against it.
- **Sets.** The analytic modes are split where the relative gap exceeds $10^{-2}$; an isolated mode is a set of size 1. The gap after the last compared mode must be $\ge 10^{-2}$, so the cutoff does not split a set. The sets and their labels are asserted.
- **Rule.** Each FE degenerate pair is first rotated by 45°: any basis of the eigenspace is valid, and this forces mixing whatever eigsh returns. Then for every set $S$, $\sum_{j\in S}\text{MAC}_{ij} \ge 0.9999$ for $i \in S$, so FE mode $i$ carries the label of its set.
- **Measured** (hex20 target meshes): isolated modes 0.999998–1.000000; a pair as returned by eigsh, e.g. 0.9997/0.0003, sums to 1.000000. Flipping the sign of the analytic $U$ gives 0.991–0.994 on PAG.
- **Checks on the analytic shapes:** they vanish on the constrained DOFs ($4\cdot10^{-16}$ of the maximum), and their self-MAC with the FE $M$ is the identity to $10^{-31}$ (unweighted: $3.5\cdot10^{-6}$ off-diagonal), which catches an unweighted MAC. This M-orthogonality is a property of these meshes and layups: sampled sines and cosines are discretely orthogonal across $(m, n)$ on a uniform in-plane grid, and within one $(m, n)$ the two lowest families differ in $z$-parity on a mid-plane-symmetric layup. It does not hold for sampled shapes in general.

**Published tables** of Srinivas et al. (1970) and Visscher et al. (1991) are not reproduced. The oracles are validated by their own convergence ($P$, $N$), the closed form and the prototype transfer matrix, and the FE converges to both at the expected rates.

## NAFEMS FV52 and FV12, Lamb sphere (PR 8c)

Not yet implemented.
