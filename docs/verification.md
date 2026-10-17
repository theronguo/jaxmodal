# Verification

Exact solutions and published benchmarks that check the discretization, with their setups, reference values and measured errors. The eigenvalues come from the certified SciPy reference solver ([`eigensolvers.md`](eigensolvers.md)). The in-test oracles live in `tests/exact.py`; the tests are in `tests/test_verification.py`.

## Error measures and mode matching

- **Targets** are relative errors in frequency, $|f^h/f - 1|$ with $f = \sqrt\lambda/2\pi$; the roadmap's 0.1% in $f$ is about 0.2% in $\lambda$.
- **Rates** are fitted to relative errors in $\lambda$, $|\lambda^h/\lambda - 1|$: the least-squares slope of $\log$ error against $\log h$, per mode, over the stated levels. Richardson extrapolation is not used. The finest error must be $\ge 100 f_\lambda$, with the solver floor $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$ ($\hat\lambda_\max$ from `eigsh(which="LA", tol=1e-3)`: an order-of-magnitude estimate suffices, and the default tolerance costs about 20 s on a 30k-DOF mesh), so that round-off cannot bend the fit.
- **Matching.** For the sliding block, the computed modes are compared sorted against the exact sorted spectrum. This is valid because each list is the lowest part of a spectrum: the $k$-th smallest values correspond, whatever the mode shapes. Exactly degenerate exact values pair either way. The Visscher box is compared the same way from mode 7 (generic anisotropy, relative gaps $\ge 10^{-2}$). Srinivas–Rao (8b) pairs by M-weighted MAC with labels, degenerate sets by summed MAC (see "Mode matching" below). FV52 (from mode 4) and FV12 (from mode 7) are compared sorted; the Lamb sphere by cluster means (from mode 7).

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

## NAFEMS FV52 and FV12

**Sources.** NAFEMS R0015, Abbassian, Dawswell & Knowles, *Selected Benchmarks for Natural Frequency Analysis* (1987), and NAFEMS TNSB Rev. 3, *The Standard NAFEMS Benchmarks* (1990); the targets as listed by Altair OptiStruct verification problems OS-V 0455 (FV52) and OS-V 0420 (FV12). Abaqus Benchmarks 4.4.10 gives the same FV52 boundary conditions, but its "NAFEMS" reference row (44.092, 106.66, …) equals its own C3D8I results, not the published targets. Steel: $E = 200$ GPa, $\nu = 0.3$, $\rho = 8000$ kg/m³. Meshes are `box_mesh` with `coords`, cosine-graded in-plane for the fine runs, $x_i = \tfrac L2(1 - \cos(\pi i/n))$.

**Gating (roadmap).** A mode is gated against its target (1%) only when the refinement-converged fine hex27 run is within 0.5% of it; otherwise against the fine run (1%), with the gap to the target recorded here.

### FV52: simply supported thick plate

- **Setup:** $10\times10\times1$ m; NAFEMS puts $u_z = 0$ along the four edges of the bottom face. Modes 1–3 are in-plane rigid modes (partial rigid modes of the solver). Targets, modes 4–10: 45.897, 109.44, 109.44, 167.89, 193.59, 206.19, 206.19 Hz.
- **Placement of $u_z = 0$.** No placement reproduces mode 4 ($f/f_\text{target}$, modes 4–10):

  | $u_z = 0$ on | mesh | ratios | converged? |
  |---|---|---|---|
  | bottom edges (NAFEMS) | hex27 8×8×2 → 20×20×3 graded; hex20 16×16×{2, 4, 8} | mode 4: 0.951 → 0.930; 0.936, 0.931, 0.926 | no, still falling |
  | mid-plane edges | hex27 32×32×4 graded; hex20 16×16×{2, 4, 8} | 0.954, 0.967, 0.967, 0.950, 1.000, 0.966, 0.968; mode 4: 0.958, 0.956, 0.955 | no, falls with $n_z$ |
  | whole edge faces | hex27 32×32×4 graded; hex20 16×16×{2, 4} | 0.963, 0.981, 0.981, 0.972, 1.000, 0.986, 0.988; mode 4: 0.9630, 0.9626 | yes |

  A $u_z = 0$ line on a solid (bottom or mid-plane edges) is a singular line support: its frequencies keep falling under refinement. Only the face support is regular. The targets presumably come from a thick-plate model rather than 3D elasticity. **Used here:** $u_z = 0$ on the whole edge faces (node sets `xmin`, `xmax`, `ymin`, `ymax`, component `z`), which keeps the 3 in-plane rigid modes.
- **Fine run:** hex27 32×32×4, cosine-graded in-plane (114k DOF, 35 s, 4.1 GB). Modes 4–10: 44.17932, 107.32841 (×2), 163.12638, 193.64577, 203.32401, 203.67752 Hz; modes 11/12: 206.758 (×2). In-plane refinement 24 → 32 changes $f$ by $\le 2\cdot10^{-5}$, $n_z$ 4 → 8 by $\le 6\cdot10^{-5}$ (downward), and hex20 at the same mesh agrees to $8\cdot10^{-6}$: the values are accurate to about $10^{-4}$, above the converged ones.
- **Pairing of modes 9/10.** The fine run has a split pair at 203.32/203.68 (FE 9/10) and a degenerate pair at 206.76 (FE 11/12, 0.28% above the 206.19 target). On the square, a pair $(m, n)/(n, m)$ is exactly degenerate when $m + n$ is odd (the two shapes have different parities in $x$ and $y$: the 2D irrep E); for $m + n$ even, the symmetric and antisymmetric combinations are 1D irreps, degenerate only by accident in a separable plate model and split in 3D. The target's degenerate pair is most likely (1,3)/(3,1), so sorted pairing (target 9/10 with FE 9/10) is used; it is also the conservative choice (larger recorded gaps).
- **Gating and gaps.** Only mode 8 (193.646 Hz, +0.03%) is within 0.5% of its target; it is gated against the target. Modes 4–7, 9, 10 are gated against the fine run; their gaps to the targets are −3.7%, −1.9%, −1.9%, −2.8%, −1.4%, −1.2%.
- **Test mesh** 12×12×2 uniform (hex20 0.5 s, hex27 1.1 s): $f/f_\text{fine} - 1$ = +0.15, +0.17, +0.17, +0.24, +0.02, +0.26, +0.26% (hex20; hex27 within 0.01 points). A conforming consistent-mass mesh lies above the exact value, so the test also asserts $f \ge f_\text{fine}(1 - 10^{-4})$; this pins the placement, since the mid-plane-edge support gives $-0.03$ to $-0.43\%$ on the same mesh, inside the two-sided 1%. The rigid eigenvalues are $\le 10\,\varepsilon\,\hat\lambda_\max$ (tested; about $10^{-12}\lambda_4$). The slow test reruns the fine mesh and reproduces the recorded values to $10^{-6}$.
- **Upper bound:** hex20 6×6×1 → 12×12×2 (nested, uniform), modes 4–10 decrease within $10\,\varepsilon\,\hat\lambda_\max$.

### FV12: free thin plate

- **Setup:** $10\times10\times0.05$ m ($L/t = 200$), free; hex20/hex27 only (hex8 locks). The NAFEMS shell model fixes $x$, $y$ and $\theta_z$ at all nodes, removing the in-plane rigid modes, so its modes 4–10 are the solid's modes 7–13. Targets: 1.622, 2.360, 2.922, 4.233, 4.233, 7.416, 7.416 Hz. Altair also lists "7.392 (HOE)" beside 7.416; the R0015 value 7.416 is used for both modes of the pair (degenerate by symmetry). Modes 12/13 are gated against the fine run, so this affects only the recorded gap.
- **Fine runs** (hex27, one element through the thickness, cosine-graded in-plane; the edge element is $1.5\cdot10^{-2}$ m at $n = 80$, below $t$):

  | mesh | DOF | time | modes 7–13 (Hz) |
  |---|---|---|---|
  | 40×40 | 59k | 8 s | 1.6188, 2.3601, 2.9236, 4.1843 (×2), 7.3688 (×2) |
  | 60×60 | 132k | 23 s, 3.5 GB | 1.618722, 2.359530, 2.922390, 4.183217 (×2), 7.358644 (×2) |
  | 80×80 | 233k | 51 s, 9.5 GB | 1.618696, 2.359358, 2.922055, 4.182914 (×2), 7.355642 (×2) |

  60 → 80 changes $f$ by $\le 0.041\%$ (modes 12/13), and $n_z$ 1 → 2 (at 40×40) by $\le 0.017\%$, so the 80×80 values are converged to $< 0.1\%$. Uniform 40×40 and 60×60 meshes differ from the graded ones by up to 0.15% (modes 10/11): the free-edge layer needs the grading. The 80×80 run is too large for the suite; the slow test reruns 60×60 and checks $|f_{60}/f_{80} - 1| \le 0.1\%$ (two-sided: graded meshes of different $n$ are not nested).
- **Gating and gaps.** Modes 7–9 are within 0.5% of their targets (−0.20%, −0.03%, 0.00%) and are gated against them; modes 10–13 sit 1.2% (10/11) and 0.8% (12/13) below the targets and are gated against the fine run.
- **Test mesh** 20×20×1 uniform, hex20 and hex27 (0.8–1.6 s): $f/f_\text{fine} - 1$ = 0.18, 0.11, 0.17, 0.24 (×2), 0.50% (×2). Six rigid eigenvalues $\le 10\,\varepsilon\,\hat\lambda_\max$.
- **Upper bound:** hex20 10×10×1 → 20×20×1 (nested, uniform), from mode 7.

## Lamb sphere

**Exact solution** (free isotropic sphere of radius $a$; Lamb, *Proc. London Math. Soc.* s1-13, 189–212, 1882). With $\eta = \omega a/c_s$, $\xi = \omega a/c_p$ and the spherical Bessel functions $j_n$, the modes of order $n \ge 1$ have multiplicity $2n + 1$ and satisfy

- torsional: $(n - 1)\,j_n(\eta) - \eta\, j_{n+1}(\eta) = 0$ ($n = 1$: $j_2(\eta) = 0$; its root $\eta = 0$ is the rigid rotation);
- spheroidal: the standard form (Saviot, Champagnon, Duval, Kudriavtsev & Ekimov, *J. Non-Cryst. Solids* 197, 238–246, 1996; Eringen & Şuhubi, *Elastodynamics* II, 1975) uses $F(x) = x j_{n+1}(x)/j_n(x)$, which has poles. Multiplied by $j_n(\eta)j_n(\xi)$ it becomes entire:
  $$2\big[\eta^2 j_n(\eta) + (n-1)(n+2)\big(\eta j_{n+1}(\eta) - (n+1)j_n(\eta)\big)\big]\,\xi j_{n+1}(\xi) + \Big(-\tfrac{\eta^4}{2} + (n-1)(2n+1)\eta^2\Big)j_n(\eta)j_n(\xi) + \big[\eta^2 - 2n(n-1)(n+2)\big]\,\eta j_{n+1}(\eta)\,j_n(\xi) = 0.$$
  At a zero of $j_n$ the old form has a simple pole, so the product is generically nonzero there: no new roots.

`lamb_sphere(nu, k)` scans both functions for $n = 1 \ldots 8$ on $(0.3, 12]$, refines each sign change with `brentq`, and returns the lowest $k$ roots as `(eta, kind, n)`. It raises unless the $k$-th root lies below the lowest root of order 8 (lowest roots rise with $n$). The breathing mode ($n = 0$) is not included; the form above gives a spurious root at 2.459 for $n = 0$, while the true breathing mode at $\nu = 0.3$ is at 4.9955.

**Values at $\nu = 0.3$:** T2 2.501133 (×5), S2 2.646217 (×5), S1 3.529987 (×3), T3 3.864700 (×7), S3 3.937401 (×7); T1 = 5.763459 is the first zero of $j_2$ (Abramowitz & Stegun, table 10.6), and T2 does not depend on $\nu$. Checks: a monomial Rayleigh–Ritz on the unit ball ($x^ay^bz^c$, total degree $\le 10$, closed-form ball integrals) reproduces modes 7–30 to $10^{-7}$; an independent derivation of the spheroidal $2\times2$ traction determinant ($\sigma_{rr} = \sigma_{r\theta} = 0$ for $u = \nabla\varphi + \nabla\times\nabla\times(r\psi)$) gives the same roots for $n = 1 \ldots 5$ to 7 digits.

**Voxel FE.** hex8 `voxel_mesh` of the cells whose centre lies in the unit ball, $n$ cells per diameter; $E = 1$, $\nu = 0.3$, $\rho = 1$, consistent mass. The voxelized volume differs from the ball's, so the exact value is taken for the sphere of equal volume, $a = (3V/4\pi)^{1/3}$. The cubic grid splits each 5-fold cluster into 3 + 2; torsional modes have a radial kinetic-energy fraction $\le 0.001$, spheroidal ones 0.55–0.60, so the fraction $< 0.1$ classifies them (every level gives exactly 5 + 5 among the lowest 10 elastic modes). Each cluster is compared by its mean $\eta$:

| $n$ | DOF | $a$ | time | T2 mean | S2 mean | max |
|---|---|---|---|---|---|---|
| 8 | 1.4k | 1.0146 | 0.3 s | −3.34% | +1.49% | 3.34% |
| 12 | 3.9k | 1.0027 | 0.5 s | −2.57% | +0.59% | 2.57% |
| 16 | 8.5k | 1.0048 | 1.4 s | −1.82% | +0.27% | 1.82% |
| 20 | 15.7k | 1.0028 | 3 s | −1.26% | +0.16% | 1.26% |
| 24 | 25.9k | 0.9986 | 8 s | −1.02% | +0.09% | 1.02% |
| 32 | 59.4k | 1.0019 | 34 s | −0.74% | +0.02% | 0.74% |
| 40 | 112k | 1.0004 | 107 s | −0.60% | −0.01% | 0.60% |

The test (`slow`) runs levels 8, 12, 16, 24, 32 with `n_modes = 18` and asserts the 5 + 5 classification, the recorded means to 0.02 percentage points (this pins the equal-volume radius: the nominal $a = 1$ moves them by 1.4 points at level 8 and still drops 5×), and the roadmap's criterion: the max error drops $\ge 2\times$ (measured 4.5×). The least-squares rate of the max is about 1.1 (reported, not tested): the staircase boundary limits convergence. No upper bound is checked, since the voxel geometry is not nested. With `n_modes = 18`, level 32 also guards the σ rule: under PR 8a's $-0.01\tilde\lambda_k$ it raises (172ε; [eigensolvers](eigensolvers.md)).

## Solid-shell element (`ssh8`, PR 16a)

Element-level checks of the solid shell ([elements](elements.md#solid-shell-ssh8)); the tests are in `tests/test_shell.py`, against the independent NumPy element in `tests/shell_ref.py` (point loops, explicit tensor transforms, dense condensation; $K_e$, $m_e$ and recovery agree to $10^{-12}$ on random distorted elements with 1–3 anisotropic plies at $h/t = 1, 2, 20$).

**MacNeal–Harder beam** (MacNeal & Harder, *Finite Elem. Anal. Des.* 1, 3–20, 1985). Beam $6\times0.2\times0.1$, $E = 10^7$, $\nu = 0.3$, $6\times1\times1$ elements converted with `direction=(0, 0, 1)` (thickness along $z$, shell plane $xy$). The 45° parallelogram and the trapezoid distortions skew the interior element boundaries by $\pm0.1$ at $y = 0$ and $0.2$, in the shell plane. Root nodes fixed; unit tip loads split equally over the 4 tip nodes (the consistent traction on the rectangular end face). Result: mean tip displacement in the load direction over the reference.

| load | reference | regular | parallelogram | trapezoid |
|---|---|---|---|---|
| extension | $3.0\cdot10^{-5}$ | 0.988 | 0.994 | 0.994 |
| in-plane shear | 0.1081 | 0.972 | 0.620 | 0.047 |
| out-of-plane shear | 0.4321 | 0.979 | 0.976 | 0.960 |
| twist (unit couple, reported) | 0.03208 | 0.903 | 0.826 | 0.855 |

Gated: regular $\ge 0.95$ for extension, in-plane and out-of-plane; parallelogram $\ge 0.9$ for extension and out-of-plane. The in-plane results on distorted elements show the known trapezoidal in-plane bending locking of the formulation (no in-plane incompatible modes beyond the 4 EAS membrane terms). The twist couple is applied as nodal forces $f = e_x\times r/\sum|r|^2$ on the tip nodes about the end-face centre; the rotation is the mean of $(r\times u)_x/|r|^2$.

**Patch tests** (one layer, $t = 0.01$). Patches: (i) a flat $3\times3$ parallelogram grid of $0.08$ elements sheared by $0.3y$ (the size of patch (ii); at element size $1/3$, $h/t = 33$, $\mathrm{cond}(K_{II}) = 2\cdot10^6$ amplifies the round-off residual to $10^{-10}$); (ii) the MacNeal–Harder patch $0.24\times0.12$ with interior nodes $(0.04, 0.02)$, $(0.18, 0.03)$, $(0.16, 0.08)$, $(0.08, 0.08)$ (5 elements); (iii) patch (ii) rigidly tilted (ZXZ Euler angles 0.4, 0.9, −0.3) with $R_o$ the tilt, so $R_\text{shell}$ is the tilt; (iv) patch (ii) with the 4 interior nodes of both surfaces lifted by $0.1t$. Linear fields $u = Ax$ give the 6 unit stress states in the shell frame (on (iv) global, isotropic $C$), with $\varepsilon = C'^{-1}\sigma_0$ and anisotropic $C$ on (i)–(iii).

- **Lift:** $Ax$ prescribed on the lateral-boundary nodes; the interior surface DOFs are loaded by the consistent traction $f_I = (K_\text{hex8}Ax)_I$ (`assemble` on the same mesh as hex8 with $C' = $ `rotate`$(C, R_\text{shell})$, exact since hex8 passes the patch test) and solved by `spsolve`. Gated at $10^{-10}$ (measured $\le 2\cdot10^{-13}$) for $u_I$ and the recovered $\sigma$: all 6 states on (i), the 4 states with $\sigma_{13} = \sigma_{23} = 0$ on (ii) and (iii).
- **Recovery of the imposed field:** `shell_recover` returns $\sigma_0$ at every point for all 6 states on (i)–(iv), to $10^{-10}$; $\sigma = C'\varepsilon$ per point and $\sum$ weights = element volume to $10^{-12}$.
- **Reported only:** the lift's transverse-shear states on (ii)–(iii) (stress errors 3–8); the lift on (iv), where only $\sigma_{33}$ passes and the membrane states give $3\cdot10^{-2}$–$0.34$ (shear 3–8); the warped patch, (ii) with interior node 4 of both surfaces moved out of plane by $0.1t$ (membrane and $\sigma_{33}$ $2\cdot10^{-2}$–$0.11$, shear 3–11). Bathe–Dvorkin ANS is not variationally consistent on non-parallelogram elements.
