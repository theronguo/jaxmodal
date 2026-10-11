# PR 8b: Visscher Ritz, Srinivas–Rao plates, simply supported BCs, MAC pairing

Second part of PR 8 (ROADMAP, "PR 8": 8a merged; 8c is NAFEMS and the Lamb sphere). Acceptance criteria are the roadmap's PR 8 items for these two oracles: the 0.1% target, convergence rates, mode matching (MAC and labels, degenerate sets by subspace) and the nested upper bound. No library code changes: both oracles are in-test (`tests/exact.py`), like the sliding block. The FE eigenvalues come from PR 8a's `reference_modes`.

## Measured facts (prototype, scratchpad `p8b/`, reviewer probes `r8b-*/`; jax 0.11.2, SciPy 1.18.1 with Accelerate, fp64, M3 Max CPU)

Materials: `ORTHO` = `orthotropic(10, 4, 2, 0.3, 0.2, 0.25, 1.5, 1.2, 0.8)`, $\rho = 0.7$ (PR 4's); `PAG` = `orthotropic(25, 1, 1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.2)`, $\rho = 1$ (the classic high-contrast cross-ply material, $E_1/E_2 = 25$). Plates: $a = b = 1$, $h = 0.1$ ($L/t = 10$), simply supported (below); ORTHO as one ply, PAG as $[0/90/0]$ with equal plies, the 90° ply `rotate(PAG, Rz90)`. $f_\lambda = 10\,\varepsilon\,\hat\lambda_\max/\lambda$ as in 8a.

- **Srinivas–Rao, two independent computations agree.**
  - *1D spectral Ritz in z* per $(m, n)$: GLL nodal basis of degree $P$ per ply, $C^0$ across plies. Degree 12 vs 24 changes the lowest 5 eigenvalues per $(m, n)$ by $\le 1.1\cdot10^{-10}$ relative.
  - *Transfer matrix* (prototype only): the $6\times6$ first-order system for $y = (U, V, W, \tau_{xz}, \tau_{yz}, \sigma_z)$ per ply, $T = \prod_k \exp(A_k h_k)$; eigenvalues are the zeros of $\det T_{[\tau], [u]}(\lambda)$ (free faces). At every Ritz eigenvalue tried (lowest 5 of (1,1), (1,2), (0,1), (2,0); 20 of 20, re-run by a reviewer), the determinant changes sign between $\lambda(1 \pm 10^{-9})$. So the Ritz eigenvalues are roots of the exact characteristic equation, and the Ritz count rules out missed roots.
  - The (0,1,0) mode is $C_{66}q^2/\rho$ ($U$ constant, $\tau_{xz} = 0$), for any stack with equal $C_{66}$, $\rho$.
- **Truncation.** On a grid $m, n \le 12$ (ORTHO, PAG, and ORTHO with $h = 0.01$), $\lambda_\min(m, n)$ is monotone in $m$ and in $n$ for $m, n \ge 1$ (0 violations), and the $m = 0$ / $n = 0$ rows lie 1.9–790× above the $m = 1$ / $n = 1$ rows. At $m_\max = 8$ the smallest eigenvalue on the boundary rows is 6.4×, 6.5×, 9.7× $\lambda_{10}$.
- **Mode order** (labels $(m, n, \text{family})$): ORTHO (1,1,0), (1,2,0), (2,1,0), (1,0,0), (0,1,0), (2,2,0), (1,3,0), (2,3,0), (3,1,0), (1,1,1). Exactly degenerate pairs (any stack with equal $C_{66}$, $\rho$ on the square plate): ORTHO modes {4, 5} = (1,0,0)/(0,1,0); PAG modes {3, 4} = (0,1,0)/(1,0,0) and {9, 10} = (0,2,0)/(2,0,0) at $4C_{66}\pi^2/\rho$. The FE pairs are degenerate to $\le 10^{-12}$ relative.
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

  Times are `reference_modes` only; each test adds about 1 s (exact spectrum 0.5 s, mesh and assembly 0.45 s). One element through the thickness plateaus at 3.4% (ORTHO, $n = 12$); the thickness direction dominates until about 4 elements per ply.
- **Rates** (lowest 6 modes, least squares over the levels; rate levels use $n_z = n/4$ per ply; ply faces conform):

  | plate, element | levels $n$ | min rate | finest error in $\lambda$ |
  |---|---|---|---|
  | ORTHO hex8 | 8, 12, 16, 24 | 1.88 | 1.4e-3 … 3.6e-2 |
  | ORTHO hex20 | 4, 8, 12, 16 | 3.92 | 2e-6 … 3.5e-4 |
  | ORTHO hex27 | 4, 8, 12, 16 | 3.88 | 2e-6 … 3.5e-4 |
  | PAG [0/90/0] hex20 | 4, 8, 12 | 3.85 | 7e-6 … 1.9e-3 |
  | PAG [0/90/0] hex27 | 4, 8, 12 | 3.77 | 7e-6 … 1.9e-3 |
  | PAG [0/90/0] hex8 | 8, 12, 16, 24 | 1.97 for modes 1–5; **1.64** for mode 6 | 1.4e-3 … 4.7e-2 |

  The finest errors are $\ge 2.7\cdot10^4 f_\lambda$ ($\hat\lambda_\max$ by `eigsh(which="LA")`). hex8 on the cross-ply is pre-asymptotic for mode 6 (4.7% error at 32k DOF). Levels refining only in-plane (fixed $n_z$) give rates 2.3–2.7 for hex20 (thickness error dominates), so the levels refine all directions.
- **MAC** (M-weighted, FE `M`; analytic shapes evaluated at the FE nodes; hex20 target meshes): isolated modes $\ge 0.999998$ with the analytic mode of the same sorted index; degenerate pairs are mixed arbitrarily by eigsh (e.g. 0.9997/0.0003), after a 45° rotation each entry is about 0.5, and the summed MAC over each set is 1.000000. The analytic shapes vanish on the constrained DOFs to $2\cdot10^{-15}$, and their M-weighted self-MAC is diagonal to $6\cdot10^{-31}$ (unweighted: $3.5\cdot10^{-6}$ off-diagonal). Mutants: the sign of the analytic $U$ flipped gives isolated MACs 0.9909–0.9940 on PAG (0.56–0.98 on ORTHO).
- **Visscher Ritz** (free box $L = (1, 0.83, 0.67)$, `rotate(ORTHO, R)` with PR 4's rotation $R$: fully anisotropic; Legendre total degree $\le N$ per component, separable 1D Gauss integrals). Point inversion $u(x) \to -u(-x)$ leaves the problem invariant for any $C$, so the basis splits exactly by the parity of $a + b + c$ into two uncoupled blocks; Legendre orthogonality makes $M$ diagonal, so each block is a standard symmetric problem. This gives the same eigenvalues as the full generalized problem to $6\cdot10^{-12}$:

  | $N$ | basis | time (parity split) | max change, modes 7–16 |
  |---|---|---|---|
  | 14 | 2040 | 0.15 s | 1.8e-4 (vs 12, modes 7–26) |
  | 16 | 2907 | 0.35 s | 6.3e-6 |
  | 18 | 3990 | 0.8 s | 2.2e-6 |
  | 20 | 5313 | 1.7 s (0.21 GB) | 8.4e-7 |

  Convergence is algebraic (about 2.6× per 2 degrees, presumably edge/corner singularities), so $N = 20$ is the first degree meeting the roadmap's $< 10^{-6}$ for modes 7–16; the remaining error is about $5\cdot10^{-7}$. The margin to $10^{-6}$ is deterministic (fixed basis and quadrature). Rigid eigenvalues $\le 1.3\cdot10^{-11}\lambda_7$; smallest relative gap among modes 7–16: $1.8\cdot10^{-2}$.
- **FE vs Visscher** (modes 7–16, max error in $f$, all positive): hex20 $n = 4, 6, 8$: 3.3e-3, 7.4e-4, **2.5e-4** (1.2 s); hex27: 3.1e-3, 7.1e-4, **2.4e-4** (2.8 s).
- **Mutants measured on the prototypes** (all killed by the 0.1% target): plate strain sign $V' + qW \to V' - qW$ (errors 0.35, 0.07), component rule removed (2.0), $m = 0$ / $n = 0$ families missing (0.28, 0.30), the 90° ply on top instead of in the middle (0.20); Visscher with $T_{ijkl}$ index-transposed (0.46), derivative scaling $1/L$ instead of $2/L$ (0.75). Reversing the ply order ($z \to h - z$) is a mirror image with the same spectrum: equivalent.

## Files

```
tests/exact.py              + SIMPLY_SUPPORTED, plate_modes (Srinivas–Rao), visscher, mac
tests/test_verification.py  + Srinivas–Rao and Visscher tests
docs/verification.md        8b sections: simply supported BCs, Srinivas–Rao, Visscher, mode matching
docs/ROADMAP.md             8b status sentence; hex8 Srinivas–Rao rate clause
README.md                   status
```

## Oracles (`tests/exact.py`)

### Simply supported BCs on solids

For a plate in the $xy$ plane, applied face-wise over each whole edge face, through the thickness: $u_y = u_z = 0$ on $x = 0, a$ and $u_x = u_z = 0$ on $y = 0, b$; top and bottom free. In `dof_mask` form: `SIMPLY_SUPPORTED = {"xmin": "yz", "xmax": "yz", "ymin": "xz", "ymax": "xz"}`. The normal stress $\sigma_{xx}$ ($\sigma_{yy}$) on those faces is free. This is exactly the Srinivas–Rao boundary condition, so the FE model and the exact solution solve the same problem. No rigid modes remain. PR 16 uses the same definition.

### `plate_modes(plies, a, b, k, points=None, P=12, mmax=8)` (Srinivas–Rao)

`plies` = list of `(C, rho, t)` stacked from $z = 0$, with `C` orthotropic in the plate axes (a 90° layer is `rotate(C, Rz90)`, still axis-aligned). Returns `lam` (k,) ascending, `labels` (k, 3) int $(m, n, \text{family})$, and `Phi` (3 n_points, k) in DOF order $3n + i$ at `points` (None without `points`).

- **Ansatz.** $u = U(z)\cos px \sin qy$, $v = V(z)\sin px\cos qy$, $w = W(z)\sin px \sin qy$, $p = m\pi/a$, $q = n\pi/b$, $m, n \ge 0$ not both 0. The BCs hold through the sine factors and $\sigma_{xx} \propto \sin px\sin qy$ vanishes on the $x$ faces (likewise $\sigma_{yy}$); the energy separates per $(m, n)$ for orthotropic plies aligned with the plate (only the normal strains couple, and they share $\sin px \sin qy$). **Component rule** as for the sliding block: $U$ exists iff $n \ge 1$, $V$ iff $m \ge 1$, $W$ iff $m, n \ge 1$. Completeness: cosine/sine families on each in-plane axis, all $z$ functions.
- **1D Ritz in z** per $(m, n)$: strains $(-pU, -qV, W', V' + qW, U' + pW, qU + pV)$ in Voigt order, energy $\int e^\top C e\,dz$, kinetic $\int\rho(U^2 + V^2 + W^2)dz$ (the in-plane factors $ab/4$, or $ab/2$ for $m = 0$ or $n = 0$, are common to all terms and cancel); GLL nodal basis of degree $P$ per ply, $C^0$ across ply interfaces, Gauss rule of $P + 3$ points (basis tables computed once per $P$); dense `eigh`. Family = index within $(m, n)$. `Phi` evaluates the z profiles only at the distinct node $z$ values and indexes back.
- **Truncation guard.** No clean provable bound exists (bending modes scale like $h^2|k|^4$, and the Rayleigh quotient has mixed-sign $p$, $q$ cross terms, so neither the Christoffel bound of the sliding block nor monotonicity follows). Two checks, documented as checks, not proof: the smallest eigenvalue on the boundary rows $m = m_\max$ or $n = m_\max$ must exceed $2\lambda_k$ (measured 6.4–9.7×), else `ValueError`; and the self-check test asserts that `mmax + 2` returns the same lowest $k$.

### `visscher(C, rho, L, N, k=16)`

Lowest `k` eigenvalues of the free box $[0, L_x]\times[0, L_y]\times[0, L_z]$, any `C`: basis $P_a(\xi)P_b(\eta)P_c(\zeta)$, $a + b + c \le N$, for each component; $K$ and $M$ from separable 1D Gauss–Legendre integrals ($N + 2$ points, exact). Per parity class of $a + b + c$: $M$ is diagonal $m$, so `eigh(m^{-1/2} K m^{-1/2}, subset_by_index=[0, k-1])`; the two classes are merged and sorted.

### `mac(A, B, M)`

M-weighted MAC matrix $(a_i^\top M b_j)^2 / ((a_i^\top M a_i)(b_j^\top M b_j))$. PR 17 adds the library version (`post/`), checked against this one.

## Tests (`tests/test_verification.py`)

Plates as above; `box_mesh((n, n, nz), spacing=(a/n, b/n, h/nz), type=t)`, element material label = ply index from the element centroid $z$ (`np.cumsum(t)`), BCs `SIMPLY_SUPPORTED`.

1. **Oracle self-checks** (fast): plate Ritz degree 12 vs 24 agree to $10^{-9}$ on ORTHO and PAG ((1,1), (1,2), (0,1), (2,0)); (0,1,0) closed form; the guard raises for `mmax=1`, and `mmax=10` returns the same lowest 10 as `mmax=8`. Visscher: 6 rigid eigenvalues $|\lambda| \le 10^{-8}\lambda_7$; $N = 18 \to 20$ changes modes 7–16 by $< 10^{-6}$ (the roadmap's convergence rule).
2. **0.1% target** (fast except hex27 cross-ply): lowest 10 modes, $|f/f_\text{exact} - 1| \le 10^{-3}$ and $\lambda^h \ge \lambda - 10\varepsilon\hat\lambda_\max$:
   - ORTHO, hex20 and hex27, $16\times16\times4$;
   - PAG $[0/90/0]$, hex20 (fast) and hex27 (`slow`, 18 s), $16\times16\times(3\text{ per ply})$;
   - Visscher modes 7–16, hex20 and hex27, $n = 8$, `rotate(ORTHO, R)` via orientations.
3. **Mode matching** (fast; ORTHO and PAG hex20 target meshes):
   - `mac(Phi, Phi, M)` off-diagonal $\le 10^{-12}$ (the analytic shapes are M-orthogonal on the FE `M`; kills an unweighted MAC), and `Phi` on constrained DOFs $\le 10^{-12}\max|\text{Phi}|$.
   - Sets: the analytic modes split where the relative gap exceeds $10^{-2}$ (`np.split` at the cuts); an isolated mode is a set of size 1. The cutoff must not split a set (gap between modes 10 and 11 $\ge 10^{-2}$, asserted). The non-trivial sets are asserted equal to the expected ones: ORTHO [[3, 4]]; PAG [[2, 3], [8, 9]] (0-based; modes {4, 5} and {3, 4}, {9, 10}).
   - Labels: `labels[:10]` equals the documented mode order (ORTHO and PAG lists written out in the test), compared as sets inside each degenerate set.
   - Each FE degenerate pair is rotated by 45° before the MAC (any basis of the eigenspace is valid; this forces mixing whatever eigsh returns).
   - One rule for every set $S$: `mac(V, Phi, M)[S][:, S].sum(1) >= 0.9999`. FE mode $i$ thus carries the $(m, n, \text{family})$ label of its set; the FE frequencies are compared through this pairing.
   - Visscher: sorted from mode 7 (generic anisotropy; relative gaps $\ge 10^{-2}$ asserted).
4. **Rates** (`slow`): levels as in the table; $\ge 1.8$ for ORTHO hex8, $\ge 3.6$ for ORTHO and PAG hex20/hex27; finest error $\ge 100 f_\lambda$. PAG hex8 is not tested (mode 6 pre-asymptotic; its measured rates are documented). $\hat\lambda_\max$ by `eigsh(Kf, 1, Mf, which="LA", tol=1e-3)`: the default tolerance costs about 20 s per 30k-DOF mesh (measured), and these checks only need the order of magnitude (the change is $\le 1.1\cdot10^{-4}$ relative). The sliding-block helper `modes()` switches to the same call.
5. **Upper bound** (inside the rate tests, consistent mass): at every level $\lambda^h_k \ge \lambda_k - 10\varepsilon\hat\lambda_\max$; nested ORTHO levels $4 \to 8$, $8 \to 16$ (hex20, hex27) and $8 \to 16$, $12 \to 24$ (hex8), PAG $4 \to 8$ (hex20, hex27): $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\varepsilon\hat\lambda_\max$. Visscher (fast): hex20 $4 \to 8$ nested, from mode 7, and $\lambda^h \ge \lambda_\text{Ritz}(1 - 10^{-6})$ (the Ritz value is an upper bound within $\approx 5\cdot10^{-7}$ of the exact one).

Expected runtime including setup: fast additions ≈ 25 s (Visscher 2.5 s cached, ORTHO targets 6 s, PAG hex20 11 s, Visscher targets 6 s, MAC reuses the target runs); slow ≈ 50 s (PAG hex27 target 21 s, rates about 30 s; measured by the performance reviewer with `tol=1e-3`).

**Mutants expected to be killed:** plate strain sign, component rule removed, missing $m = 0$ / $n = 0$ families, the 90° rotation on the wrong ply or a wrong ply index from the centroid $z$ (target); Visscher derivative scaling and index-transposed $T$ (target); a wrong parity split ($a + b$, $a + b + c + i$, $a$ alone, one class dropped, per-class subset too small: 0.07–0.56, target); sign of an analytic shape component flipped (MAC $\ge 0.9999$); MAC without M (self-MAC check); degenerate-set handling removed (45° rotation); labels permuted or $m$/$n$ swapped (label assert); wrong gap cut (set assert). **Equivalent:** reversed ply order (mirror image, same spectrum); no parity split (the full problem).

## Docs

- **`docs/verification.md`:**
  - simply supported BCs (definition above, PR 16 reuses it);
  - Srinivas–Rao: source (Srinivas & Rao 1970, J. Sound Vib.), exact setup (geometry, materials, layup and ply labels), ansatz and component rule, the 1D Ritz, one sentence on the prototype transfer-matrix agreement (state $(U, V, W, \tau_{xz}, \tau_{yz}, \sigma_z)$, free faces, sign changes within $\lambda(1 \pm 10^{-9})$ at all 20 values tried), the truncation checks (not a proof, with the reason and the measured margins and monotonicity), reference values (lowest 10 $\lambda$ with labels for ORTHO and PAG), target meshes and errors, rates with the hex8 margin (1.88 vs 1.8) and the measured cross-ply hex8 rates (1.97 for modes 1–5, 1.64 for mode 6; not tested);
  - Visscher: source (Visscher et al. 1991, J. Acoust. Soc. Am.), basis, parity split and diagonal $M$, convergence table and rule, reference values (modes 7–16 at $N = 20$, uncertainty $\approx 5\cdot10^{-7}$), FE errors;
  - mode matching: MAC definition, sets by relative gap, the 45° rotation, labels; the self-MAC $\le 10^{-12}$ holds for these meshes and mid-plane-symmetric layups (discrete sine orthogonality across $(m, n)$, z-parity within one $(m, n)$), not for sampled shapes in general;
  - update the "Matching" bullet of the error-measures section: Srinivas–Rao (8b) and FV52 (8c) pair by M-weighted MAC with labels, degenerate sets by summed MAC; Visscher is compared sorted from mode 7;
  - why published tables are not reproduced (the two independent computations and FE convergence validate the oracles).
- **`docs/ROADMAP.md`:** the 8b status sentence, and in it: "the hex8 Srinivas–Rao rate is taken on the single-ply orthotropic plate (the cross-ply is pre-asymptotic for hex8 at affordable sizes)".
- **`README.md`:** status.

## Not covered

- Published tables of Srinivas & Rao (1970) and Visscher et al. (1991) are not reproduced: the Srinivas–Rao Ritz is validated by its $P$-convergence, the (0,1,0) closed form and the prototype's transfer-matrix agreement; the Visscher Ritz by its $N$-convergence; the FE converges to both at the expected rates.
- The transfer-matrix check is not in the tests (prototype fact; it would share the ansatz and conventions with the Ritz).
- hex8 rates on the cross-ply (mode 6 pre-asymptotic at affordable sizes; documented only).
- Thin plates ($L/t \ge 100$): PR 16 (solid shells).
- 8c: NAFEMS FV52/FV12, Lamb sphere.

## Implementation notes

- The Lagrange basis is the Legendre Vandermonde times its inverse at the GLL nodes (vectorized; degree ≤ 24 is well conditioned). `np.einsum(..., optimize=True)` for the ply stiffness: without it, `plate_modes` at $P = 24$ took 3.2 s instead of 0.24 s.
- Labels and sets are one assert per plate (simplicity NIT): the expected order is a list of label sets, compared with `[set(map(tuple, labels[S])) for S in sets]`.
- `plate_fe` always computes 10 modes, so the rate levels reuse the target runs (ORTHO hex20/hex27 $n = 16$).
- Measured: fast additions 30 s, slow additions 48 s (PAG hex27 target 27 s, rates 22 s), on a loaded machine.
