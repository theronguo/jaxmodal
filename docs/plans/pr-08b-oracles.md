# PR 8b: Visscher Ritz, Srinivas–Rao plates, simply supported BCs, MAC pairing

Second part of PR 8 (ROADMAP, "PR 8": 8a merged; 8c is NAFEMS and the Lamb sphere). Acceptance criteria are the roadmap's PR 8 items for these two oracles: the 0.1% target, convergence rates, mode matching (MAC and labels, degenerate sets by subspace) and the nested upper bound. No library code changes: both oracles are in-test (`tests/exact.py`), like the sliding block. The FE eigenvalues come from PR 8a's `reference_modes`.

## Measured facts (prototype, scratchpad `p8b/`; jax 0.11.2, SciPy 1.18.1 with Accelerate, fp64, M3 Max CPU)

Materials: `ORTHO` = `orthotropic(10, 4, 2, 0.3, 0.2, 0.25, 1.5, 1.2, 0.8)`, $\rho = 0.7$ (PR 4's); `PAG` = `orthotropic(25, 1, 1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.2)`, $\rho = 1$ (the classic high-contrast cross-ply material, $E_1/E_2 = 25$). $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$ as in 8a.

- **Srinivas–Rao, two independent computations agree.** Plate $a = b = 1$, $h = 0.1$ ($L/t = 10$), simply supported (below).
  - *1D spectral Ritz in z* per $(m, n)$: GLL nodal basis of degree $P$ per ply, $C^0$ across plies. Degree 12 vs 24 changes the lowest 5 eigenvalues per $(m, n)$ by $\le 1.1\cdot10^{-10}$ relative.
  - *Transfer matrix*: the $6\times6$ first-order system for $y = (U, V, W, \tau_{xz}, \tau_{yz}, \sigma_z)$ per ply, $T = \prod_k \exp(A_k h_k)$; eigenvalues are the zeros of $\det T_{[\tau], [u]}(\lambda)$ (free faces). At every Ritz eigenvalue tried (lowest 5 of (1,1), (1,2), (0,1), (2,0)), the determinant changes sign between $\lambda(1 \pm 10^{-9})$.
  - So the Ritz eigenvalues are roots of the exact characteristic equation, and the Ritz count rules out missed roots (which a sign scan of the determinant could miss).
  - Checks: the (0,1) family starts at $C_{66}q^2/\rho$ (the pure $u_x$ shear mode, $U$ constant), as it must.
- **Mode order** (ORTHO plate): (1,1,0), (1,2,0), (2,1,0), (1,0,0), (0,1,0), (2,2,0), (1,3,0), (2,3,0), (3,1,0), (1,1,1), labels $(m, n, \text{family})$. (1,0,0) and (0,1,0) are exactly degenerate ($C_{66}\pi^2/\rho a^2$ on the square plate).
- **0.1% target, lowest 10 modes,** max error in $f$ (all positive):

  | plate | mesh $(n, n, n_z)$ | hex20 | hex27 |
  |---|---|---|---|
  | ORTHO | 12, 12, 4 | 1.00e-3 (0.6 s) | 9.9e-4 |
  | ORTHO | 16, 16, 3 | 7.9e-4 | 7.8e-4 |
  | ORTHO | **16, 16, 4** | **4.2e-4** (1.2 s, 14.1k DOF) | **4.2e-4** (2.5 s, 27.1k) |
  | ORTHO | 20, 20, 4 | 2.8e-4 | 2.8e-4 |
  | PAG [0/90/0] | 12, 12, 3/ply | 9.4e-4 | 9.3e-4 |
  | PAG [0/90/0] | **16, 16, 3/ply** | **3.9e-4** (7.2 s, 29.0k) | **3.9e-4** (17 s, 57.1k) |
  | PAG [0/90/0] | 16, 16, 4/ply | 3.0e-4 (14 s) | 3.0e-4 (34 s) |

  One element through the thickness plateaus at 3.4% (ORTHO, $n = 12$); the thickness direction dominates until about 4 elements per ply.
- **Rates** (lowest 6 modes, least squares over the levels, $n_z = n/4$ per ply; ply faces conform):

  | plate, element | levels $n$ | min rate | finest error in $\lambda$ |
  |---|---|---|---|
  | ORTHO hex8 | 8, 12, 16, 24 | 1.88 | 1.4e-3 … 3.6e-2 |
  | ORTHO hex20 | 4, 8, 12, 16 | 3.92 | 2e-6 … 3.5e-4 |
  | ORTHO hex27 | 4, 8, 12, 16 | 3.88 | 2e-6 … 3.5e-4 |
  | PAG [0/90/0] hex20 | 4, 8, 12 | 3.85 | 7e-6 … 1.9e-3 |
  | PAG [0/90/0] hex27 | 4, 8, 12 | 3.77 | 7e-6 … 1.9e-3 |
  | PAG [0/90/0] hex8 | 8, 12, 16, 24 | **1.64** (mode 6; others ≥ 1.97) | 1.4e-3 … 4.7e-2 |

  The finest errors are $\ge 277 \cdot 100 f_\lambda$ ($\hat\lambda_\max$ by `eigsh(which="LA")`). hex8 on the cross-ply is pre-asymptotic for mode 6 (4.7% error at 32k DOF); the roadmap's hex8 rate is taken on the ORTHO plate. A level set refining only in-plane (fixed $n_z$) gives rates 2.3–2.7 for hex20 (thickness error dominates), so the levels refine all directions.
- **MAC** (M-weighted, FE `M`; analytic shapes evaluated at the FE nodes; ORTHO hex20 16×16×4): every isolated mode pairs with MAC $\ge 0.9999$ with the analytic mode of the same sorted index; the degenerate (1,0,0)/(0,1,0) pair is mixed arbitrarily by eigsh (MAC 0.9997/0.0003), and the summed MAC over the set is 1.0000. Analytic shapes vanish on the constrained DOFs to $2\cdot10^{-15}$.
- **Visscher Ritz** (free box $L = (1, 0.83, 0.67)$, `rotate(ORTHO, R)` with PR 4's rotation $R$: fully anisotropic, no symmetry; Legendre total degree $\le N$ per component, separable 1D Gauss integrals, dense `eigh`):

  | $N$ | basis | time | max change, modes 7–16 |
  |---|---|---|---|
  | 14 | 2040 | 0.7 s | 1.8e-4 (vs 12, modes 7–26) |
  | 16 | 2907 | 1.8 s | 6.3e-6 |
  | 18 | 3990 | 4.6 s | 2.2e-6 |
  | 20 | 5313 | 12.3 s (8.1 s with `subset_by_index`) | 8.4e-7 |

  Convergence is algebraic (about 2.6× per 2 degrees, presumably edge/corner singularities), so $N = 20$ is the first degree meeting the roadmap's $< 10^{-6}$ for modes 7–16; the remaining error is about $5\cdot10^{-7}$. Modes 7–26 would need $N \approx 24$ (not affordable in a test).
- **FE vs Visscher** (modes 7–16, max error in $f$, all positive): hex20 $n = 4, 6, 8$: 3.3e-3, 7.4e-4, **2.5e-4** (1.2 s); hex27: 3.1e-3, 7.1e-4, **2.4e-4** (2.8 s).

## Files

```
tests/exact.py              + SIMPLY_SUPPORTED, plate_modes (Srinivas–Rao), visscher, mac
tests/test_verification.py  + Srinivas–Rao and Visscher tests
docs/verification.md        8b sections: simply supported BCs, Srinivas–Rao, Visscher, MAC pairing
docs/ROADMAP.md             status sentence for 8b
README.md                   status
```

## Oracles (`tests/exact.py`)

### Simply supported BCs on solids

For a plate in the $xy$ plane, applied face-wise over each whole edge face, through the thickness: $u_y = u_z = 0$ on $x = 0, a$ and $u_x = u_z = 0$ on $y = 0, b$; top and bottom free. In `dof_mask` form: `SIMPLY_SUPPORTED = {"xmin": "yz", "xmax": "yz", "ymin": "xz", "ymax": "xz"}`. The normal stress $\sigma_{xx}$ ($\sigma_{yy}$) on those faces is free. This is exactly the Srinivas–Rao boundary condition, so the FE model and the exact solution solve the same problem. No rigid modes remain. PR 16 uses the same definition.

### `plate_modes(plies, a, b, k, P=12, mmax=8)` (Srinivas–Rao)

`plies` = list of `(C, rho, z0, z1)` with `C` orthotropic in the plate axes (a cross-ply 90° layer is `rotate(C, Rz90)`, still axis-aligned). Returns the lowest `k` modes as `(lam, (m, n, family), shape)`, sorted by `lam`, where `shape(points)` gives the `(n_points, 3)` displacement.

- **Ansatz.** $u = U(z)\cos px \sin qy$, $v = V(z)\sin px\cos qy$, $w = W(z)\sin px \sin qy$, $p = m\pi/a$, $q = n\pi/b$, $m, n \ge 0$ not both 0. Each term of the BCs carries $\sin(m\pi) = 0$ or $\sin(n\pi) = 0$; the energy separates per $(m, n)$ for orthotropic plies aligned with the plate. **Component rule** as for the sliding block: $U$ exists iff $n \ge 1$, $V$ iff $m \ge 1$, $W$ iff $m, n \ge 1$. Completeness: cosine/sine families on each in-plane axis, all $z$ functions.
- **1D Ritz in z** per $(m, n)$: strains $(-pU, -qV, W', V' + qW, U' + pW, qU + pV)$, energy $\int e^\top C e\,dz$, kinetic $\int\rho(U^2 + V^2 + W^2)dz$ (the common in-plane factors cancel, also for $m = 0$ or $n = 0$); GLL nodal basis of degree $P$ per ply, $C^0$ across ply interfaces, Gauss rule of $P + 3$ points; dense `eigh`. Family = index within $(m, n)$.
- **Self-checks (in the test):** degree 12 vs 24 agree to $10^{-9}$; at each returned eigenvalue of the first few $(m, n)$, the transfer-matrix determinant changes sign within $\lambda(1 \pm 10^{-8})$ (the transfer matrix is written out in the test, about 25 lines, with the ODE system in `docs/verification.md`); the (0,1,0) mode equals $C_{66}\pi^2/(\rho b^2)$ for a single ply.
- **Truncation guard.** No clean provable bound exists here (bending modes scale like $h^2|k|^4$, so the Christoffel bound of the sliding block does not apply). The guard is a consistency check: the lowest eigenvalue over the boundary rows $m = m_\max$ or $n = m_\max$ must exceed $2\lambda_k$, else `ValueError`. Documented as a check, not a proof.

### `visscher(C, rho, L, N)`

Eigenvalues of the free box $[0, L_x]\times[0, L_y]\times[0, L_z]$, any (anisotropic) `C`: basis $P_a(\xi)P_b(\eta)P_c(\zeta)$, $a + b + c \le N$, for each component; $K$ and $M$ from separable 1D Gauss–Legendre integrals ($N + 2$ points, exact); `eigh(K, M, subset_by_index=[0, k-1])`. Returns the lowest `k` (default 16).

### `mac(A, B, M)`

M-weighted MAC matrix $(a_i^\top M b_j)^2 / ((a_i^\top M a_i)(b_j^\top M b_j))$. PR 17 adds the library version (`post/`), which will be checked against this one.

## Tests (`tests/test_verification.py`)

Plates: $a = b = 1$, $h = 0.1$; ORTHO single ply ($\rho = 0.7$) and PAG $[0/90/0]$ (equal plies, $\rho = 1$); `box_mesh((n, n, nz), spacing=(a/n, b/n, h/nz), type=t)`, element material label by ply (from the element centroid $z$), BCs `SIMPLY_SUPPORTED`.

1. **Oracle self-checks** (fast): the plate Ritz at degree 12 vs 24 and the transfer-matrix sign changes on ORTHO and PAG (first 5 eigenvalues of (1,1), (1,2), (0,1), (2,0)); (0,1,0) closed form; the truncation guard raises for `mmax=1`. Visscher: 6 rigid eigenvalues $|\lambda| \le 10^{-8}\lambda_7$; $N = 18 \to 20$ changes modes 7–16 by $< 10^{-6}$ (the roadmap's convergence rule, asserted).
2. **0.1% target** (fast except hex27 cross-ply): lowest 10 modes, $|f/f_\text{exact} - 1| \le 10^{-3}$ and $\lambda^h \ge \lambda - 10\varepsilon\hat\lambda_\max$:
   - ORTHO, hex20 and hex27, $16\times16\times4$;
   - PAG $[0/90/0]$, hex20 (fast) and hex27 (`slow`, 17 s), $16\times16\times(3\text{ per ply})$;
   - Visscher modes 7–16, hex20 and hex27, $n = 8$, `rotate(ORTHO, R)` via orientations.
3. **Mode matching** (fast; ORTHO and PAG hex20 target meshes): analytic shapes at the FE nodes (exact zeros on constrained DOFs, $\le 10^{-12}$ relative), MAC with the FE modes. Degenerate sets = analytic relative gap $< 10^{-2}$. Isolated mode $i$: MAC with analytic mode $i$ $\ge 0.99$ and the largest in its row (the $(m, n, \text{family})$ label of mode $i$ is the analytic one); each degenerate set: summed MAC over the set $\ge 0.99$ for every FE mode in it. The FE frequencies are compared through this pairing (identical to sorted order here). Visscher: sorted from mode 7 (generic anisotropy, no degeneracy: relative gaps measured $\ge$ 1e-2 asserted in the test).
4. **Rates** (`slow`): levels as in the table; $\ge 1.8$ for ORTHO hex8, $\ge 3.6$ for ORTHO and PAG hex20/hex27; finest error $\ge 100 f_\lambda$.
5. **Upper bound** (inside the rate tests, consistent mass): at every level $\lambda^h_k \ge \lambda_k - 10\varepsilon\hat\lambda_\max$; nested ORTHO levels $4 \to 8$, $8 \to 16$ (hex20, hex27) and $8 \to 16$, $12 \to 24$ (hex8), and PAG $4 \to 8$: $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\varepsilon\hat\lambda_\max$. Visscher (fast): hex20 $4 \to 8$ nested, from mode 7, and $\lambda^h \ge \lambda_\text{Ritz}(1 - 10^{-6})$ (the Ritz value is an upper bound within $\approx 5\cdot10^{-7}$ of the exact one).

Expected runtime: fast additions ≈ 30 s (Visscher 2 × 8 s cached once, ORTHO targets 4 s, PAG hex20 7 s, MAC 2 s); slow ≈ 60 s.

**Mutants expected to be killed:** plate ansatz sign or a strain entry wrong ($V' + qW \to V' - qW$; target and transfer check), component rule removed (spurious modes), missing $m = 0$ / $n = 0$ families (missed (1,0,0)/(0,1,0) modes), ply order reversed in the cross-ply (target), Visscher: a wrong derivative scaling $2/L_d$, $C_{ijkl}$ index transposed in the coupling (target), MAC without M (measured in the implementation; listed as equivalent if it survives), degenerate-set handling removed (the test also pairs a 45°-rotated basis of the FE (1,0)/(0,1) pair, so an isolated-mode rule fails there whatever basis eigsh returns).

## Docs

- **`docs/verification.md`:** simply supported BCs (definition above); Srinivas–Rao (setup, ansatz, component rule, the 1D Ritz and the transfer-matrix check with the ODE system, materials, meshes, measured errors and rates, the hex8 cross-ply note); Visscher (basis, convergence table and rule, measured FE errors); mode matching (MAC definition, degenerate sets, labels).
- **`docs/ROADMAP.md`:** 8b status sentence.
- **`README.md`:** status.

## Not covered

- Published tables of Srinivas & Rao (1970) and Visscher et al. (1991) are not reproduced: the exact solution is validated by two independent computations (Ritz in $z$ and the transfer-matrix determinant), and the Ritz solution by its convergence; the FE converges to both at the expected rates.
- hex8 rates on the cross-ply (pre-asymptotic at affordable sizes).
- Thin plates ($L/t \ge 100$): PR 16 (solid shells).
- 8c: NAFEMS FV52/FV12, Lamb sphere.
