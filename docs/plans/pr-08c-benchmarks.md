# PR 8c: NAFEMS FV52 and FV12, Lamb sphere, graded box meshes, σ-rule fix

Third part of PR 8 (8a, 8b merged). Acceptance criteria are the roadmap's PR 8 items for the published benchmarks: NAFEMS FV52 and FV12 gated as the roadmap specifies, the voxel Lamb sphere rate criterion, mode matching from mode 4 (FV52) / 7 (FV12, Lamb), and the nested upper bound for FV52/FV12. Two library changes come with it: graded `box_mesh` (per-axis node coordinates, promised in `docs/mesh.md`), and a fix of PR 8a's σ rule that the voxel sphere exposed.

## Measured facts (prototype, scratchpad `p8c/`; jax 0.11.2, SciPy 1.18.1, fp64, M3 Max CPU)

### σ-rule bug in `reference_modes` (PR 8a)

- Pass 2 (free bodies and other near-zero modes) runs shift-invert at $\sigma = -0.01\tilde\lambda_k$. The shifted rigid modes then dominate the operator ($1/|\sigma|$ against $1/(\lambda - \sigma)$), and ARPACK's accuracy relative to the largest Ritz value costs the elastic residuals a factor of about $\lambda/|\sigma| \approx 100$. 8a's docs said "0.01 is a margin"; the margin is in the wrong direction.
- Voxel sphere ($n = 32$ cells per diameter, 59.4k DOF, 18 modes): residual **172ε** against the 100ε bound, `reference_modes` raises. Residuals at pass-2 factors 0.01, 0.1, 0.3, 1: 172, 8.0, 3.6, 3.7 ε; time 12.0, 12.4, 12.2, 13.1 s. hex20 $12^3$ cube: 5.5, 1.5, 1.1, 1.5 ε; 3.3–3.7 s. The residual at 0.01 grows with size (spheres 12, 16, 20, 32: 24, 14–19, 49, 172 ε).
- Fix: $\sigma = -0.3\tilde\lambda_k$. The rigid amplification is then $\le \lambda_k/|\sigma| = 3.3$, and $|\sigma| < \lambda_k$ keeps clear of the other failure ("σ far below $\lambda_k$ raises the residuals", 8a). All 25 tests of `tests/test_reference.py` (slow included) pass; spheres 32 and 40 certify.

### Lamb sphere

- **Exact** (free isotropic sphere, radius $a$, $\eta = \omega a/c_s$, $\xi = \omega a/c_p$, $j_n$ spherical Bessel):
  - torsional, order $n \ge 1$: $(n - 1)j_n(\eta) - \eta j_{n+1}(\eta) = 0$; multiplicity $2n + 1$; $n = 1$ is rotation;
  - spheroidal, $n \ge 1$ (Lamb 1882; the standard form, e.g. Saviot et al.): with $F(x) = x j_{n+1}(x)/j_n(x)$,
    $$2\big[\eta^2 + (n-1)(n+2)(F(\eta) - (n+1))\big]F(\xi) - \tfrac{\eta^4}{2} + (n-1)(2n+1)\eta^2 + \big[\eta^2 - 2n(n-1)(n+2)\big]F(\eta) = 0.$$
    (For $n = 0$ this form gives a spurious root at 2.459; the breathing mode is not needed.)
  - $\nu = 0.3$: the lowest elastic modes are T2 $\eta = 2.501133$ (×5), S2 $2.646217$ (×5), S1 3.529987 (×3), T3 3.864700 (×7), S3 3.937401 (×7).
- **Independent check:** monomial Ritz on the unit ball (closed-form ball integrals, $x^ay^bz^c$, total degree $\le N$) gives modes 7–30 equal to the roots: degree 8 to $2\cdot10^{-7}$, degree 10 to $1\cdot10^{-7}$ (printed precision). Prototype only.
- **Voxel FE** (hex8, `voxel_mesh` with cells whose centre lies in the unit ball, $E = 1$, $\nu = 0.3$, $\rho = 1$, consistent mass; exact solution for the sphere of equal voxelized volume, $a = (3V/4\pi)^{1/3}$): the cubic grid splits each 5-fold cluster into 3 + 2. Torsional modes have radial kinetic-energy fraction $\le 0.001$, spheroidal 0.55–0.60, so the fraction $< 0.1$ separates them (all levels). Errors of the cluster means:

  | $n$ | DOF | time | T2 mean | S2 mean | max |
  |---|---|---|---|---|---|
  | 8 | 1.4k | 0.1 s | −3.34% | +1.49% | 3.34% |
  | 12 | 3.9k | 0.3 s | −2.57% | +0.59% | 2.57% |
  | 16 | 8.5k | 1.3 s | −1.82% | +0.27% | 1.82% |
  | 20 | 15.7k | 3.2 s | −1.26% | +0.16% | 1.26% |
  | 24 | 25.9k | 7.3 s | −1.02% | +0.09% | 1.02% |
  | 32 | 59.4k | 33 s (fixed rule) | −0.74% | +0.02% | 0.74% |
  | 40 | 112k | 107 s | −0.60% | −0.01% | 0.60% |

  Levels 8, 12, 16, 24, 32: max error drops 4.5× (≥ 2× required); the least-squares rate of the max is about 1.1 (reported only). Level 32 is the case that fails under the old σ rule, so this test also guards the fix.

### NAFEMS FV52 (simply supported thick plate)

- Setup (NAFEMS; e.g. Abaqus Benchmarks FV52, Altair OS-V 0455): $10\times10\times1$ m, $E = 200$ GPa, $\nu = 0.3$, $\rho = 8000$ kg/m³; $u_z = 0$ along the four edges of the bottom face; modes 1–3 are in-plane rigid. Targets, modes 4–10: 45.897, 109.44, 109.44, 167.89, 193.59, 206.19, 206.19 Hz.
- **No placement of $u_z = 0$ reproduces mode 4** (hex27, $f/f_\text{target}$, modes 4–10):

  | $u_z = 0$ on | mesh | ratios | converged? |
  |---|---|---|---|
  | bottom edges (NAFEMS) | 8×8×2 → 20×20×3 graded | mode 4: 0.951 → 0.930, still falling | no: a line support on a solid is singular |
  | mid-plane edges | 32×32×4 graded | 0.954, 0.967, 0.967, 0.950, 1.000, 0.966, 0.968 | yes (24 → 32: < 0.01%) |
  | whole edge faces | 32×32×4 graded | 0.963, 0.981, 0.981, 0.972, 1.000, 0.986, 0.988 | yes (24 → 32: < 0.01%) |

  The targets are presumably from thick-plate theory, not 3D elasticity. **Decision:** $u_z = 0$ on the whole edge faces (face-wise, like the 8b simply supported BCs but $z$ only, so the 3 in-plane rigid modes remain). It converges fast and is closest to the targets. The NAFEMS bottom-edge line support is documented with its non-convergence.
- Fine hex27 (edge faces, 32×32×4, cosine-graded in-plane, 114k DOF, 33 s): modes 4–12: 44.179, 107.328, 107.328, 163.126, 193.646, 203.324, 203.678, 206.758, 206.758 Hz. The last refinement (24 → 32) changes $f$ by $\le 2\cdot10^{-5}$ relative. Modes 9/10 are a split pair (203.3/203.7), and 11/12 (206.76, 0.3% above the 206.19 target) a degenerate one; sorted by index, target modes 9/10 compare with FE 9/10.
- Gating (roadmap rule): only mode 8 (193.646, +0.03%) is within 0.5% of its target and is gated against it (1%); modes 4–7, 9, 10 are gated against the fine run (1%); the gaps (−3.7%, −1.9%, −1.9%, −2.8%, −1.4%, −1.2%) are recorded.
- Test mesh 12×12×2 uniform, edge faces: against the fine run, max $|f/f_\text{fine} - 1|$ = 0.26% (hex20) and 0.26% (hex27), modes 9/10; 0.35 s (hex20), 0.8 s (hex27). (hex27 8×8×2: 0.84%, too close to 1%.) Nested coarse level 6×6×1: up to 2.5%. Rigid modes: $f le 8cdot10^{-7} f_4$.

### NAFEMS FV12 (free thin plate)

- Setup: $10\times10\times0.05$ m ($L/t = 200$), steel as above, free. Targets (plate model, modes 4–10 there = solid modes 7–13): 1.622, 2.360, 2.922, 4.233, 4.233, 7.416, 7.416 Hz.
- One element through the thickness (hex20/hex27 bending is exact enough at nz = 1 here); hex27, modes 7–13 at the finest graded meshes:

  | mesh | DOF | time | modes 7–13 (Hz) |
  |---|---|---|---|
  | 40×40 graded | 59k | 7.8 s | 1.6188, 2.3601, 2.9236, 4.1843 (×2), 7.3688 (×2) |
  | 60×60 graded | 132k | 22 s (4.4 GB) | 1.61872, 2.35953, 2.92239, 4.18322 (×2), 7.35864 (×2) |
  | 80×80 graded | 233k | 51 s (9.5 GB) | 1.61870, 2.35936, 2.92205, 4.18291 (×2), 7.35564 (×2) |

  60 → 80 changes $f$ by ≤ 0.04% (modes 12/13), so 80×80 graded is the refinement-converged fine run (< 0.1%). Mode 7 is 0.2% below its target and gated against it; modes 8, 9 (−0.03%, 0.00%) too; modes 10–13 sit 1.2% and 0.8% below the targets (the roadmap's "≈ 1%") and are gated against the fine run.
- Test mesh hex20 (and hex27) 20×20×1 uniform: against the fine run 0.18%, 0.11%, 0.17%, 0.24%, 0.50% (modes 12/13); 0.6–1.3 s. (Roadmap: "20×20 hex20 is ≈ 0.6% above converged for modes 12–13".)
- Cosine grading $x_i = \tfrac L2(1 - \cos(\pi i/n))$; the edge element is $1.5\cdot10^{-2}$ m at $n = 80$… $3\cdot10^{-2}$ at 40, below $t$. Uniform 40 and 60 differ from graded by up to 0.15% (modes 10/11), so the free-edge layer needs the grading.

## Files

```
src/jaxmodal/mesh/box.py        box_mesh(..., coords=None): per-axis node coordinates (graded)
src/jaxmodal/eigen/reference.py pass-2 shift -0.3 lam_k
tests/test_mesh.py              (or the box_mesh tests' file) coords tests
tests/test_reference.py         the σ-rule docstring/mutant note only
tests/exact.py                  + lamb_sphere
tests/test_verification.py      + FV52, FV12, Lamb tests; FV reference values as constants
docs/mesh.md                    graded box meshes
docs/eigensolvers.md            σ rule pass 2 at -0.3 lam_k, with the measured failure
docs/verification.md            FV52, FV12, Lamb sections
docs/ROADMAP.md, README.md      status; FV52 placement note
```

## Library changes

### Graded `box_mesh`

`box_mesh(shape, spacing=(1, 1, 1), origin=(0, 0, 0), type="hex8", coords=None)`. `coords`: None or a tuple of three 1D arrays of length $n_d + 1$, strictly increasing and finite, the element boundaries per axis; it replaces `spacing` and `origin`. A node at half-grid index $i$ along axis $d$ gets `np.interp(i / 2, arange(n_d + 1), coords[d])`: corner nodes on the boundaries, midside and centre nodes at element midpoints (straight edges, affine per axis, so the elements stay parallelepipeds and the upper-bound theory holds). Uniform `coords` give the same points as `spacing`/`origin` (to rounding). Invalid `coords` raise `ValueError`.

### σ rule

Pass 2 at $-0.3\tilde\lambda_k$ instead of $-0.01\tilde\lambda_k$; `docs/eigensolvers.md` gets the measured failure and the amplification argument. PR 8a's "σ multiplier 0.01 → 0.1 or 1 (same accuracy)" equivalent-mutant note is replaced: 0.01 is now killed by the Lamb level 32 (slow).

## Oracle (`tests/exact.py`)

`lamb_sphere(nu, k)`: the lowest torsional and spheroidal roots $\eta$ for orders $n = 1 \ldots$, returned as `(eta, kind, n)` sorted, the first `k`; brackets from a sign scan of each frequency function on $(0.3, 8]$, roots by `brentq`, poles of $F$ excluded by requiring $|f(\text{root})|$ small. Only T and S with $n \ge 1$ (no breathing mode; $n = 1$ torsional is the rigid rotation and is skipped). The tests use T2 and S2.

## Tests (`tests/test_verification.py`)

1. **`test_box_coords`** (in the mesh tests): uniform `coords` equal `spacing`/`origin` points (allclose 1e-15); graded hex20/hex27 midside and centre nodes are element midpoints; each element's corners are an axis-aligned box; `check_elements` passes; invalid coords (wrong length, non-increasing, non-finite, wrong count) raise.
2. **`test_lamb_oracle`** (fast): T2 2.501133 and S2 2.646217 at $\nu = 0.3$ to $10^{-6}$ (prototype values, matched by the independent ball Ritz), torsional roots satisfy $(n-1)j_n = \eta j_{n+1}$ to 1e-12 (self-consistency), and ordering for $\nu = 0.3$: T2, S2, S1, T3, S3.
3. **`test_lamb_rates`** (`slow`): levels 8, 12, 16, 24, 32; modes 7–22 computed (16 elastic, so both 5-fold clusters are inside with margin); torsional/spheroidal split by radial kinetic-energy fraction $< 0.1$; take the 5 lowest torsional and 5 lowest spheroidal; error of each cluster mean against the equal-volume exact value; assert the max over both drops ≥ 2× from level 8 to 32, and each level classifies exactly 5 + 5 among the lowest 10. Upper bound not checked (geometry not nested).
4. **`test_fv52`** (fast): hex20 and hex27 12×12×2 uniform, $u_z = 0$ on the four edge faces (node sets `xmin`, `xmax`, `ymin`, `ymax`, `"z"`), modes 4–10: mode 8 within 1% of its target; all modes within 1% of the fine values (constants, from the fine run); 3 rigid modes $|\lambda| \le 10^{-8}\lambda_4$.
5. **`test_fv12`** (fast): hex20 and hex27 20×20×1 uniform, free, modes 7–13: modes 7–9 within 1% of the targets, all within 1% of the fine values; 6 rigid modes.
6. **Upper bound** (fast, hex20): FV52 $6\times6\times1 \to 12\times12\times2$ from mode 4, FV12 $10\times10\times1 \to 20\times20\times1$ from mode 7 (nested, uniform, consistent mass): $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\varepsilon\hat\lambda_\max$; and the test mesh above the fine values ($f \ge f_\text{fine}(1 - 10^{-4})$, the fine run being converged to $2\cdot10^{-5}$ / $4\cdot10^{-4}$).
7. **Fine runs** (`slow`): FV52 hex27 32×32×4 graded (33 s) reproduces the recorded fine values to $10^{-6}$; FV12's 80×80 run (51 s, 9.5 GB) is too large for the suite: the slow test reruns 60×60 graded (22 s, 4.4 GB) and asserts it lies within 0.05% above the recorded 80×80 values (convergence evidence), the recorded values coming from the prototype run (documented).

Mode matching (roadmap): FV52 from mode 4, FV12 and Lamb from mode 7, sorted (FV) or by cluster (Lamb); no labelled modes, so no MAC.

Runtime: fast additions ≈ 8 s (FV52 2 × 0.9 s, FV12 2 × 1.3 s, bounds ≈ 1 s, coords < 1 s); slow ≈ 100 s (Lamb 42 s, FV52 fine 33 s, FV12 60×60 22 s).

**Mutants expected to be killed:** σ factor back to 0.01 (Lamb level 32); `coords` midside at the wrong place, e.g. `interp(i)` instead of `interp(i / 2)` (coords test); FV52 constraint only on bottom edges (fine values, 1%: 5–7% off) or in all directions (target and fine values); FV12 with a hidden constraint (rigid count); Lamb classification threshold inverted, equal-volume radius replaced by the nominal one (level 8: radius 1.0146 vs 1, so a 1.4% shift; the 2× drop then fails at level 8 or is caught by the mean check), torsional equation with $n$ instead of $n - 1$ (oracle values).

## Docs

- `docs/mesh.md`: graded `coords` (replacing the "will take per-axis arrays" sentence).
- `docs/eigensolvers.md`: the σ rule's pass 2 at $-0.3\tilde\lambda_k$, the amplification $\lambda/|\sigma|$ and the measured failure; correct the "0.01 is a margin" sentence.
- `docs/verification.md`: FV52 (setup, sources, the placement study and decision, fine values, gating and recorded gaps), FV12 (setup, grading, fine values and convergence, gating, gaps), Lamb sphere (equations, source, ball-Ritz check, voxelization, equal-volume radius, cluster selection by radial fraction, error table and the 2× criterion).
- `docs/ROADMAP.md`: 8c status; the FV52 sentence "the docs fix the $u_z = 0$ placement that the fine hex27 run shows reproduces mode 4" becomes "… no placement reproduces mode 4 (3D elasticity is 3.7% below the plate-theory target); edge faces are used, gated against the fine run". `README.md`: status.

## Not covered

- FV52's bottom-edge line support (singular, does not converge; documented).
- Breathing and higher Lamb modes; Lamb with curved (non-voxel) meshes.
- The FV12 80×80 fine run in the suite (9.5 GB); its 60×60 check is.
