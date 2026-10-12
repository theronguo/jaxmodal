# PR 8c: NAFEMS FV52 and FV12, Lamb sphere, graded box meshes, σ-rule fix

PR 8 is split into three parts; 8a and 8b are merged, and this is the third. Its acceptance criteria are the roadmap's PR 8 items for the published benchmarks:

- NAFEMS FV52 and FV12, gated as the roadmap specifies;
- the rate criterion for the voxel Lamb sphere;
- mode matching from mode 4 (FV52) and from mode 7 (FV12 and Lamb);
- the nested upper bound for FV52 and FV12.

Two library changes come with it:

- graded `box_mesh`, with per-axis node coordinates (promised in `docs/mesh.md`);
- a fix to PR 8a's σ rule, which the voxel sphere exposed.

## Measured facts (prototype, scratchpad `p8c/`; jax 0.11.2, SciPy 1.18.1, fp64, M3 Max CPU)

### σ-rule bug in `reference_modes` (PR 8a)

- **The bug.** Pass 2 (free bodies and other near-zero modes) runs shift-invert at $\sigma = -0.01\tilde\lambda_k$.
  - The shifted rigid modes then dominate the operator: $1/|\sigma|$ against $1/(\lambda - \sigma)$.
  - ARPACK's accuracy is relative to the largest Ritz value, so the elastic residuals grow by a factor of about $\lambda/|\sigma| \approx 100$.
  - 8a's docs called 0.01 "a margin"; the margin points the wrong way.
- **Voxel sphere** ($n = 32$ cells per diameter, 59.4k DOF), full `reference_modes`:
  - With 18 modes the residual is **172ε** against the 100ε bound, so `reference_modes` raises.
  - With 22 modes it passes at 50ε (performance reviewer). **The failure depends on $k$.**
  - Residuals at pass-2 factors 0.01, 0.1, 0.3 and 1 (18 modes): 172, 8.0, 3.6 and 3.7 ε, in 12.0, 12.4, 12.2 and 13.1 s.
  - One pass-2 solve with $k = 23$: 50, 4.8 and 3.2 ε at factors 0.01, 0.3 and 1; 0.3 costs +4% time.
- **hex20 $12^3$ cube:** 5.5, 1.5, 1.1 and 1.5 ε, in 3.3–3.7 s.
- At 0.01 the residual grows with size: spheres 12, 16, 20 and 32 give 24, 14–19, 49 and 172 ε.
- **Fix:** $\sigma = -0.3\tilde\lambda_k$.
  - The rigid amplification is then at most $\lambda_k/|\sigma| = 3.3$.
  - Keeping $|\sigma| < \lambda_k$ avoids the other failure from 8a: σ far below $\lambda_k$ raises the residuals.
  - With the fix, all of `test_reference.py` and `test_verification.py` pass, slow tests included: 59 tests in 169 s (correctness reviewer). Spheres 32 and 40 certify.

### Lamb sphere

**Exact solution.** A free isotropic sphere of radius $a$, with $\eta = \omega a/c_s$, $\xi = \omega a/c_p$ and $j_n$ the spherical Bessel functions (Lamb 1882).

- **Torsional, order $n \ge 1$:**
  $$(n - 1)j_n(\eta) - \eta j_{n+1}(\eta) = 0,$$
  with multiplicity $2n + 1$.
  - $n = 1$ reduces to $j_2(\eta) = 0$. Its root $\eta = 0$ is the rigid rotation. Its elastic roots are 5.7635, 9.095, … (zeros of $j_2$, Abramowitz & Stegun table 10.6), and they are kept.
- **Spheroidal, $n \ge 1$:** the standard form (Saviot et al. 1996; Eringen & Şuhubi 1975) uses $F(x) = x j_{n+1}(x)/j_n(x)$, which has poles. Multiplying by $j_n(\eta)\,j_n(\xi)$ gives the pole-free form used here:
  $$2\big[\eta^2 j_n(\eta) + (n-1)(n+2)\big(\eta j_{n+1}(\eta) - (n+1)j_n(\eta)\big)\big]\,\xi j_{n+1}(\xi) + \Big(-\tfrac{\eta^4}{2} + (n-1)(2n+1)\eta^2\Big)j_n(\eta)j_n(\xi) + \big[\eta^2 - 2n(n-1)(n+2)\big]\,\eta j_{n+1}(\eta)\,j_n(\xi) = 0.$$
  - The function is entire, so every sign change is a root.
  - At a zero of $j_n$ the old form had a simple pole, so the product is finite and generically nonzero there: no new roots appear.
  - For $n = 0$ the form gives a spurious root at 2.459. The true breathing mode is at 4.9955 (correctness reviewer) and is not needed.
- **Values at $\nu = 0.3$,** the lowest elastic modes:

  | mode | $\eta$ | multiplicity |
  |---|---|---|
  | T2 | 2.501133 | 5 |
  | S2 | 2.646217 | 5 |
  | S1 | 3.529987 | 3 |
  | T3 | 3.864700 | 7 |
  | S3 | 3.937401 | 7 |

- **Independent checks.** These values were reproduced three times:
  - monomial Ritz on the unit ball (prototype, $x^ay^bz^c$ with total degree $\le N$, closed-form ball integrals): modes 7–30 equal the roots, to $2\cdot10^{-7}$ at degree 8 and $1\cdot10^{-7}$ at degree 10 (printed precision);
  - the docs reviewer, solving the plan's equations independently;
  - the correctness reviewer, who derived the spheroidal 2×2 traction determinant from $\nabla\varphi + \nabla\times\nabla\times(r\psi)$ and matched every digit.
- **Voxel FE setup.** hex8 elements; `voxel_mesh` with the cells whose centre lies in the unit ball; $E = 1$, $\nu = 0.3$, $\rho = 1$; consistent mass. The exact value is taken for the sphere of equal voxelized volume, $a = (3V/4\pi)^{1/3}$.
- **Cluster splitting and classification.** The cubic grid splits each 5-fold cluster into 3 + 2.
  - Torsional modes have a radial kinetic-energy fraction of at most 0.001; spheroidal modes have 0.55–0.60.
  - A threshold of $< 0.1$ therefore separates them at every level.
  - With `n_modes = 18` (12 elastic modes) every level classifies 5 + 5 among the lowest 10.
- **Errors of the cluster means:**

  | $n$ | DOF | $a_\text{eq}$ | time | T2 mean | S2 mean | max |
  |---|---|---|---|---|---|---|
  | 8 | 1.4k | 1.0146 | 0.1 s | −3.34% | +1.49% | 3.34% |
  | 12 | 3.9k | 1.0027 | 0.3 s | −2.57% | +0.59% | 2.57% |
  | 16 | 8.5k | 1.0048 | 1.3 s | −1.82% | +0.27% | 1.82% |
  | 20 | 15.7k | | 3.2 s | −1.26% | +0.16% | 1.26% |
  | 24 | 25.9k | 0.9986 | 7.3 s | −1.02% | +0.09% | 1.02% |
  | 32 | 59.4k | 1.0019 | 26 s (18 modes) | −0.74% | +0.02% | 0.74% |
  | 40 | 112k | | 107 s | −0.60% | −0.01% | 0.60% |

  - Over levels 8, 12, 16, 24 and 32, the max error drops 4.5× (at least 2× is required).
  - The least-squares rate of the max is about 1.1. It goes in the docs only.
  - The nominal radius ($a = 1$) changes the level-8 means by 1.4 points (T) and 1.5 points (S), and level 32 by 0.19 points. The max still drops 5.1×, so only checking the recorded means catches it.

### NAFEMS FV52 (simply supported thick plate)

- **Sources.**
  - Primary: NAFEMS R0015, Abbassian, Dawswell & Knowles, *Selected Benchmarks for Natural Frequency Analysis* (1987); and NAFEMS TNSB Rev. 3, *The Standard NAFEMS Benchmarks* (1990).
  - Secondary listing of the targets: Altair OS-V 0455.
  - Abaqus Benchmarks 4.4.10 is used only for the BC wording. Its "NAFEMS" reference row (44.092, 106.66, …) equals its own C3D8I results, not the published targets.
- **Setup.** A $10\times10\times1$ m plate with $E = 200$ GPa, $\nu = 0.3$ and $\rho = 8000$ kg/m³. $u_z = 0$ along the four edges of the bottom face. Modes 1–3 are in-plane rigid modes.
- **Targets,** modes 4–10: 45.897, 109.44, 109.44, 167.89, 193.59, 206.19, 206.19 Hz.
- **No placement of $u_z = 0$ reproduces mode 4.** Ratios $f/f_\text{target}$ for modes 4–10 (hex27, except where noted; through-thickness data from the correctness reviewer, hex20):

  | $u_z = 0$ on | mesh | ratios | converged? |
  |---|---|---|---|
  | bottom edges (NAFEMS) | 8×8×2 → 20×20×3 graded; hex20 16×16×{2, 4, 8} | mode 4: 0.951 → 0.930; 0.936, 0.931, 0.926 | no, still falling |
  | mid-plane edges | 32×32×4 graded; hex20 16×16×{2, 4, 8} | 0.954, 0.967, 0.967, 0.950, 1.000, 0.966, 0.968; mode 4: 0.958, 0.956, 0.955 | no, falls with $n_z$ |
  | whole edge faces | 32×32×4 graded; hex20 16×16×{2, 4} | 0.963, 0.981, 0.981, 0.972, 1.000, 0.986, 0.988; mode 4: 0.9630, 0.9626 | yes |

  - A $u_z = 0$ line on a solid, whether at the bottom or mid-plane edges, is a singular line support, and its frequencies fall without limit as the mesh is refined. Only the face support is regular.
  - The targets presumably come from thick-plate theory, not 3D elasticity.
- **Decision:** $u_z = 0$ on the whole edge faces.
  - It is applied face-wise, like the 8b simply supported BCs but in $z$ only, so the 3 in-plane rigid modes remain.
  - It converges and is closest to the targets.
  - The NAFEMS bottom-edge line support is documented, together with its non-convergence.
- **Fine run:** hex27, edge faces, 32×32×4, cosine-graded in-plane, 114k DOF, 35 s, 4.1 GB.
  - Modes 4–12: 44.179, 107.328, 107.328, 163.126, 193.646, 203.324, 203.678, 206.758 and 206.758 Hz.
  - hex20 at 32×32×4 reproduces these to $8\cdot10^{-6}$.
  - In-plane refinement 24 → 32 changes $f$ by at most $2\cdot10^{-5}$. $n_z$ 4 → 8 lowers $f$ by at most $6\cdot10^{-5}$.
  - So the fine values are accurate to about $10^{-4}$, and sit above the truth.
- **Pairing modes 9/10.**
  - The fine run has a split pair at FE 9/10 (203.32/203.68) and a degenerate pair at FE 11/12 (206.76, +0.28% from the 206.19 target).
  - The square's symmetry group makes a pair exactly degenerate (E) when $m + n$ is odd, as for FE 11/12. Pairs like (1,3)/(3,1) form symmetric and antisymmetric combinations, which are degenerate only accidentally in separable plate theory, and 3D splits them (FE 9/10).
  - Which FE pair the theory's degenerate 206.19 pair corresponds to is therefore not decidable from the targets alone.
  - **Decision:** pair by index (target 9/10 with FE 9/10). This is the conservative choice: the gaps −1.4% and −1.2% are recorded, both modes are gated against the fine run, and the docs state that FE 11/12 lie 0.28% from the target.
- **Gating (roadmap rule).**
  - Only mode 8 (193.646, +0.03%) is within 0.5% of its target, so it is gated against the target (1%).
  - Modes 4–7, 9 and 10 are gated against the fine run (1%).
  - The gaps are recorded: −3.7%, −1.9%, −1.9%, −2.8%, −1.4% and −1.2%.
- **Test mesh:** 12×12×2 uniform, edge faces.
  - Max $|f/f_\text{fine} - 1|$ against the fine run is 0.26% for both hex20 and hex27, at modes 9/10.
  - Runtimes: 0.35 s (hex20), 0.8 s (hex27).
  - hex27 8×8×2 gives 0.84%, too close to 1%.
  - The nested coarse level 6×6×1 is off by up to 2.5%.
  - Rigid modes: $f \le 8\cdot10^{-7} f_4$.

### NAFEMS FV12 (free thin plate)

- **Sources:** as for FV52 (NAFEMS R0015 and TNSB; Altair OS-V 0420 as the secondary listing).
- **Setup:** a $10\times10\times0.05$ m plate ($L/t = 200$), steel as above, free.
- **Shell constraint, and the mode shift.** NAFEMS constrains $x$, $y$ and $\theta_z$ at all nodes of the shell model, which removes the in-plane rigid modes. Its modes 4–10 are therefore solid modes 7–13.
- **Targets:** 1.622, 2.360, 2.922, 4.233, 4.233, 7.416 and 7.416 Hz.
  - Altair also lists "7.392 (HOE)" beside 7.416. This plan uses 7.416 ×2, the R0015 value; the pair is degenerate by symmetry, so it shares one target.
  - Modes 12/13 are gated against the fine run, so the choice affects only the recorded gap.
- **Fine meshes:** one element through the thickness, hex27, cosine-graded in-plane. Modes 7–13:

  | mesh | DOF | time | modes 7–13 (Hz) |
  |---|---|---|---|
  | 40×40 graded | 59k | 7.8 s | 1.6188, 2.3601, 2.9236, 4.1843 (×2), 7.3688 (×2) |
  | 60×60 graded | 132k | 23 s (3.5 GB) | 1.61872, 2.35953, 2.92239, 4.18322 (×2), 7.35864 (×2) |
  | 80×80 graded | 233k | 51 s (9.5 GB) | 1.61870, 2.35936, 2.92205, 4.18291 (×2), 7.35564 (×2) |

  - Going from 60 to 80 changes $f$ by at most 0.04% (modes 12/13).
  - Going from $n_z = 1$ to 2 at 40×40 changes $f$ by at most 0.017% (correctness reviewer).
  - So the 80×80 graded run is converged to < 0.1%.
- **Gating against the targets:**
  - Mode 7 is 0.2% below its target; modes 8 and 9 are −0.03% and 0.00%. All three are gated against their targets.
  - Modes 10–13 sit 1.2% and 0.8% below the targets (the roadmap's "≈ 1%") and are gated against the fine run.
- **Test mesh:** hex20 (and hex27) 20×20×1 uniform. Against the fine run: 0.18%, 0.11%, 0.17%, 0.24% and 0.50% (modes 12/13); 0.6–1.3 s.
- **Grading:** cosine, $x_i = \tfrac L2(1 - \cos(\pi i/n))$.
  - The edge element is $1.5\cdot10^{-2}$ m at $n = 80$ and $3\cdot10^{-2}$ m at $n = 40$, below $t$.
  - Uniform 40 and 60 meshes differ from the graded ones by up to 0.15% (modes 10/11), so the free-edge layer needs the grading.

## Files

```
src/jaxmodal/mesh/box.py        box_mesh(..., coords=None): per-axis node coordinates (graded)
src/jaxmodal/eigen/reference.py pass-2 shift -0.3 lam_k
tests/test_box.py               coords tests
tests/test_reference.py         explicit-sigma test at -0.3 lam_{n+1}; mutant note
tests/exact.py                  + lamb_sphere
tests/test_verification.py      + FV52, FV12, Lamb tests; FV fine values, Lamb errors as constants
docs/mesh.md                    graded box meshes
docs/eigensolvers.md            σ rule pass 2 at -0.3 lam_k, with the measured failure
docs/verification.md            FV52, FV12, Lamb sections
docs/ROADMAP.md, README.md      status; FV52 and FV12 wording
```

## Library changes

### Graded `box_mesh`

`box_mesh(shape, spacing=(1, 1, 1), origin=(0, 0, 0), type="hex8", coords=None)`.

- **`coords`** is None or a tuple of three 1D arrays: the element boundaries per axis.
  - Array $d$ must have length $n_d + 1$ and be strictly increasing and finite; otherwise `ValueError`.
  - It overrides `spacing` and `origin`, which are then ignored (documented).
- **Node placement.** A node at half-grid index $i$ along axis $d$ gets `np.interp(i / 2, arange(n_d + 1), coords[d])`.
  - Corner nodes land on the boundaries; midside and centre nodes land at element midpoints.
  - Each element is therefore an affine, axis-aligned box. That keeps the geometry exact and makes the coarse and fine meshes of the nested test nested.
- Uniform `coords` give the same points as `spacing` and `origin`, up to rounding.

### σ rule

- Pass 2 runs at $-0.3\tilde\lambda_k$ instead of $-0.01\tilde\lambda_k$.
- `docs/eigensolvers.md` gets the measured failure and the amplification argument (see Docs).
- PR 8a's equivalent-mutant note ("σ multiplier 0.01 → 0.1 or 1, same accuracy") is replaced: the factor 0.01 is now killed by Lamb level 32 with `n_modes = 18` (slow). That kill depends on $k$: with 22 modes the residual is 50ε and the mutant survives.

## Oracle (`tests/exact.py`)

`lamb_sphere(nu, k)` returns the lowest `k` distinct roots as a sorted list of `(eta, kind, n)`.

- It covers torsional and spheroidal orders $n = 1 \ldots 8$ (no breathing mode) and uses the two entire frequency functions above.
- **Root finding.** A sign scan of each function on a fine grid over $(0.3, 12]$ gives the brackets, and `brentq` refines them. There is no pole filter.
  - The scan starts at 0.3, which excludes the torsional $n = 1$ rotation at $\eta = 0$.
  - Torsional $n = 1$ is kept (T1 = 5.7635).
- It raises `ValueError` if it finds fewer than `k` roots.

## Tests

1. **`test_box_coords`** (fast, `tests/test_box.py`):
   - uniform `coords` equal the points from `spacing`/`origin` (allclose 1e-15);
   - in graded hex20 and hex27 meshes, the midside and centre nodes are element midpoints;
   - invalid `coords` raise: wrong length for the shape, non-increasing, non-finite.
2. **`test_lamb_oracle`** (fast), against published or independent values:
   - T2 = 2.501133, which is independent of $\nu$ (Lamb; matches the ball Ritz), to $10^{-6}$;
   - T1 = 5.763459 (the first zero of $j_2$, A&S 10.6), to $10^{-6}$;
   - S2 = 2.646217 at $\nu = 0.3$ (ball Ritz and the reviewers' independent derivations), to $10^{-6}$;
   - at $\nu = 0.3$ the ordering is T2, S2, S1, T3, S3;
   - `k` beyond the scan range raises.
3. **`test_lamb_rates`** (`slow`):
   - levels 8, 12, 16, 24 and 32, with `n_modes = 18` (12 elastic modes);
   - torsional and spheroidal modes are split by radial kinetic-energy fraction $< 0.1$;
   - at each level, exactly 5 + 5 of the lowest 10 elastic modes are classified, and the cluster-mean errors (T2 and S2, against the exact value for the equal-volume radius) match recorded constants within 0.02 percentage points;
   - the max error over both clusters drops by at least 2× from level 8 to level 32.

   No rate is computed in the test, and the upper bound is not checked (the voxel geometry is not nested).
4. **`test_fv52`** (fast): hex20 and hex27 12×12×2 uniform, with $u_z = 0$ on the four edge faces (node sets `xmin`, `xmax`, `ymin`, `ymax`, component `"z"`). For modes 4–10:
   - mode 8 is within 1% of its target;
   - all modes are within 1% of the recorded fine values;
   - the 3 rigid modes satisfy $|\lambda| \le 10^{-8}\lambda_4$.
5. **`test_fv12`** (fast): hex20 and hex27 20×20×1 uniform, free. For modes 7–13:
   - modes 7–9 are within 1% of their targets;
   - all modes are within 1% of the recorded fine values;
   - there are 6 rigid modes.
6. **Upper bound** (fast, hex20; nested, uniform meshes, consistent mass):
   - FV52 from mode 4, coarse $6\times6\times1$ to fine $12\times12\times2$;
   - FV12 from mode 7, coarse $10\times10\times1$ to fine $20\times20\times1$;
   - assert $\lambda_k(\text{fine}) \le \lambda_k(\text{coarse}) + 10\varepsilon\hat\lambda_\max$.
7. **Fine runs** (`slow`):
   - FV52: hex27 32×32×4 graded (35 s) reproduces the recorded fine values to $10^{-6}$.
   - FV12: the 80×80 run (51 s, 9.5 GB) is too large for the suite. The slow test reruns 60×60 graded (23 s, 3.5 GB) and asserts it lies within 0.1% above the recorded 80×80 values (the roadmap's "last refinement < 0.1%"; measured 0.041%). The 80×80 values come from the prototype run, and the docs say so.

**Mode matching (roadmap):** FV52 is compared from mode 4, FV12 and Lamb from mode 7. FV modes are taken in sorted order and Lamb modes by cluster. No modes are labelled, so no MAC is needed.

**Runtime:**
- Fast additions: about 8 s (FV52 2 × 0.9 s, FV12 2 × 1.3 s, bounds about 1 s, coords < 1 s).
- Slow additions: about 95 s (Lamb about 35 s, FV52 fine 35 s, FV12 60×60 23 s).

**Mutants expected to be killed:**
- σ factor back to 0.01: Lamb level 32 at `n_modes = 18`.
- `coords` midside nodes misplaced, e.g. `interp(i)` instead of `interp(i / 2)`: coords test.
- FV52 constraint only on the bottom or mid-plane edges (1–7% off), or in all directions: fine values at 1%.
- FV12 with a hidden constraint: rigid count.
- Lamb classification threshold inverted: the 5 + 5 count and the recorded means.
- Equal-volume radius replaced by the nominal one: the recorded means (1.4 points off at level 8, 0.19 points at level 32, against a 0.02-point tolerance).
- Torsional equation with $n$ instead of $n - 1$: T2 and T1.
- A wrong spheroidal coefficient: S2.

## Docs

- **`docs/mesh.md`:** graded `coords`, replacing the "will take per-axis arrays" sentence.
- **`docs/eigensolvers.md`:** every −0.01 mention.
  - Pass 2 (line 38): $-0.3\tilde\lambda_k$, with the amplification $\lambda/|\sigma|$ and the measured failure (sphere 32: 172ε at 18 modes; depends on $k$).
  - Line 40: replace "same accuracy … 0.01 is a margin" with the measured residuals at 0.01, 0.1, 0.3 and 1.
  - Line 42: re-measure the $L/t = 1600$ paragraph ($0.09 f_\lambda$) under the new rule.
  - Line 44: the explicit-`sigma` advice for PR 16 becomes $-0.3\lambda_k$.
  - Line 127: the explicit $\sigma$ in the Validation bullet becomes $-0.3\lambda_{n+1}$ (the test changes with it).
- **`docs/verification.md`:**
  - **FV52:** sources (primary and secondary, and the Abaqus row), setup, the placement study (including the through-thickness data) and the decision, the fine values and their accuracy (about $10^{-4}$, including $n_z$), the 9/10 pairing, gating, and the recorded gaps.
  - **FV12:** sources, the shell constraint as the reason for the mode shift, the 7.416/7.392 note, grading, fine values and convergence (in-plane and $n_z$), gating, and gaps.
  - **Lamb sphere:**
    - the equations, in pole-free form;
    - sources: Lamb, *Proc. London Math. Soc.* s1-13, 189–212 (1882); Saviot, Champagnon, Duval, Kudriavtsev & Ekimov, *J. Non-Cryst. Solids* 197, 238–246 (1996); Eringen & Şuhubi, *Elastodynamics* II (1975);
    - the numerical checks;
    - voxelization and the equal-volume radius;
    - cluster selection by radial fraction;
    - the error table, the 2× criterion, and the rate (reported only).
- **`docs/ROADMAP.md`:**
  - 8c status.
  - The FV52 sentence becomes: "no $u_z = 0$ placement reproduces mode 4 (the converged 3D value with $u_z = 0$ on the edge faces is 3.7% below the target; the NAFEMS bottom-edge line support does not converge); edge faces are used, other modes gated as the rule says".
  - FV12: "≈ 0.6%" becomes "≈ 0.5%".
- **`README.md`:** status.

## Not covered

- FV52's bottom-edge line support (singular, does not converge; documented).
- Breathing and higher Lamb modes; Lamb on curved (non-voxel) meshes.
- The FV12 80×80 fine run in the suite (9.5 GB); its 60×60 check is in the suite.
