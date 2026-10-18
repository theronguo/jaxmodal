# PR 16c plan: solid-shell accuracy suite

Branch `pr-16c-accuracy`. The design frame is `docs/plans/pr-16-solid-shell.md` ("The accuracy
suite"); this plan fixes meshes, levels and the layout from the sizing study below, and lists
where it departs from the frame. Roadmap: PR 16 tests "0.1% target", "No locking", "Distorted
meshes", and the `docs/verification.md` entries ("reference levels, elements per ply").

## Revisions

- rev 1: first draft after the sizing study.
- rev 2 (round 1, all 5 not satisfied): zero-index filter component fixed and tested on a wide
  strip; Richardson rule capped at $p \le 4$ with a 1.25 factor, no round-off branch, and a
  one-level-back check on a fourth level; skew reference graded toward the clamped edges; finer
  `ssh8` cylinder and skew meshes; `ortho` $L/t = 10$ at $n_z = 24$; oracle tolerance per $L/t$;
  estimator unit tests; the stacking ($n_z = 1$ vs 2) explanation; the hash replaced by builder
  digests plus per-path recomputes; no-locking as a direct test with a bounded ratio; layout
  (one analysis function, constants in `tests/helpers.py`); `COUNT_STEP` re-deferred; corrected
  times and peaks.
- rev 3 (round 2: math and tests not satisfied, simplicity, docs and perf satisfied): the $n_z$
  term's $p_z$ is fixed at the corner-singularity rate 1.42 (observed $p_z$ only checked against
  it) and extrapolated in $h$ from two in-plane levels; the cylinder reference moves to $n_z = 3$
  (the $n_z = 2$ reference had margin 1.07 under that rule); the half-cylinder antisymmetric check
  excludes the torsion mode; a mode-crossing guard and the back-check counts; the filter
  classifies by a ratio and asserts the kept count; a strip recompute key, a multi-chunk hex27
  recompute, equivalence tests through the builders, a synthetic `analyze` test, a fuller digest;
  the benchmark split into parts `srao`, `cyl`, `skew`; the docs minors.

## Sizing study (shipped `ssh8`, 2026-10-18)

Shipped code only (`solid_shell`, `assemble`, `reference_modes`), on the Mac (64 GB). Errors
are relative in $f$. Peaks are the child's maximum RSS. Times include the
$\hat\lambda_\max$ `eigsh(which="LA", tol=1e-3)` (30–40% of a run) unless stated. Scripts and
logs: the session scratchpad `s16c/` (`srao.py`, `cases.py`, `analyze3.py`); the round 1 perf
reviewer reproduced the smallest-margin cases bit for bit.

### Srinivas–Rao

Each label on its half-wave sub-plate quarter ($q$ in-plane elements per quarter side, $k$ stacked
`ssh8` elements per ply).

| laminate | $L/t$ | $q$, $k$ ($n_z$) | worst (label, error) | margin | solve time per run, peak | part (solve only) |
|---|---|---|---|---|---|---|
| `pag` | 10 | 32, 12 (36) | (1,3) $5.77\cdot10^{-4}$ | 1.73 | 58–95 s, 9.65 GB | 11.5 min |
| `ortho` | 10 | 32, 24 (24) | (3,1) $4.99\cdot10^{-4}$ | 2.00 | 42–58 s, 5.3 GB | 6.5 min |
| `pag` | 100 | 48, 3 (9) | (1,5) $4.19\cdot10^{-4}$ | 2.39 | 15–31 s, 4.3 GB | 4.2 min |
| `ortho` | 100 | 48, 4 (4) | (2,4) $3.37\cdot10^{-4}$ | 2.97 | 3–4 s | 0.5 min |

- Times are solve only (srao.py had no $\hat\lambda_\max$; it adds about 39 s per 113k-DOF run,
  18 s per 77k, 5 s per 69k): the part is about 23 min solve plus about 7 min $\hat\lambda_\max$.
- Lowest 10 at $L/t = 10$: the frame's lists; at $L/t = 100$: `pag` (1,1), (1,2), (1,3), (2,1),
  (2,2), (1,4), (2,3), (2,4), (1,5), (2,5); `ortho` (1,1), (1,2), (2,1), (2,2), (1,3), (2,3), (3,1),
  (1,4), (3,2), (2,4) (no zero-index or family-1 label).
- Every target is rank 0 of its quarter class except `ortho` (1,1,1), rank 3 on the (1,1) quarter;
  the nearest same-class neighbour is $\ge 5.1\%$ away in $f$ for all 40 targets (round 1 math).
  The rank needs at least 12 sub-plate `plate_modes` at $L/t = 10$ (round 2 perf), so the
  benchmark asks for 12. The zero-index sub-plates ((1,0), (0,1), (2,0), (0,2)) give
  $1.0\cdot10^{-4}$ (in-plane only).
- `ortho` $L/t = 10$ at $n_z = 36$ (the frame) gives $3.36\cdot10^{-4}$ (2.98) at 95 s, 9.6 GB per
  run; $n_z = 24$ keeps margin 2.0 and saves 5 min. `pag` at $q = 24$ (margin 1.34) is not adopted.
- $L/t = 100$ is thickness-dominated: `pag` (2,5) is $9.90\cdot10^{-4}$ at $k = 1$ ($q = 48$),
  $9.35$/$8.95\cdot10^{-4}$ at $q = 64$/96, but $6.19\cdot10^{-4}$ at $k = 2$, $3.68\cdot10^{-4}$ at
  $k = 3$ and $2.67\cdot10^{-4}$ at $k = 4$. The frame's $k = 1$ leaves margin 1.01 (`pag`) and 1.17
  (`ortho`).
- **Stacking, $n_z = 1$ vs 2.** On a bending mode of a symmetric plate $u_x, u_y$ are odd in $z$, so
  with two layers the mid node lies on the neutral plane ($u_x = u_y = 0$) and the in-plane
  displacement space through the thickness is the one-layer linear field; the extra freedom is in
  the even $u_z$ only. The transverse shear stays constant through the thickness (the stiffening
  that layers remove), so $n_z = 2$ gains nothing: SS `ortho` $L/t = 20$, $q = 32$, (1,1) gives
  $2.347$, $2.341$, $1.313$, $0.877$, $0.542$, $0.419\cdot10^{-3}$ at $k = 1, 2, 3, 4, 6, 8$, and
  (2,3) $1.050$, $1.052$, $0.545$, $0.330$, $0.164$, $0.103\cdot10^{-2}$. At clamped edges with
  $\nu \ne 0$ one layer is in addition unusually soft (round 2 math): the EAS thickness parameter
  ($\zeta\alpha_5$) is unrestrained there, and its softening cancels part of the stiffening. Clamped
  square, $L/t = 20$, (1,1): $\nu = 0$ gives $4.31$ vs $4.17\cdot10^{-3}$ ($n_z = 1$ vs 2), $\nu =
  0.3$ gives $-0.82$ vs $4.14\cdot10^{-3}$; SS shows no such difference. Hence on the skew plate
  ($128\times128$) $n_z = 2$ is stiffer than 1 ($1.05$ vs $0.79\cdot10^{-2}$, then
  $0.60\cdot10^{-2}$ at 3), and the cylinder S/A gives $9.4$ vs $8.0\cdot10^{-4}$. Not a stacking
  defect: sweep, $+z$ and $-z$ directions agree to $10^{-13}$ at $n_z = 1$ and 2. Every chosen $k$ is
  $\ge 3$ or $n_z \ge 4$ (decreasing regime); the cylinder and skew gate $n_z = 1$ directly.

### Clamped cylinder

$R/t = 50$, $L/R = 2$, 1/8 symmetry, 4 classes; hex27 at $n_z = 3$, axially cosine-graded toward
the clamped end ($y_i = 1 - \cos(\pi i/2n)$, nested); levels $12\times6\times3$ (back-check only),
$24\times12\times3$, $48\times24\times3$, $96\times48\times3$; $n_z$ runs $24\times12\times\{3,4,5\}$
(coarse) and $48\times24\times\{3,4\}$ (mid); `ssh8` $512\times96\times1$ uniform.

| class | $p$ (24/48/96) | $p$ (12/24/48) | back-check (modes) | Richardson | $n_z$ term | reference error | `ssh8` worst | margin (+ ref) |
|---|---|---|---|---|---|---|---|---|
| S/S | 3.38–3.72 | 3.10–3.52 | $\ge 1.42$ (6/6) | $\le 4.9\cdot10^{-5}$ | $\le 4.8\cdot10^{-5}$ | $8.6\cdot10^{-5}$ | $6.00\cdot10^{-4}$ | 1.67 (1.46) |
| S/A | 3.19–3.60 | 2.25–3.36, 8.21 | $\ge 1.47$ (5/6) | $\le 6.5\cdot10^{-5}$ | $\le 5.1\cdot10^{-5}$ | $1.17\cdot10^{-4}$ | $7.64\cdot10^{-4}$ | 1.31 (1.14) |
| A/S | 3.32–3.71 | 3.00–3.50 | $\ge 1.41$ (6/6) | $\le 6.2\cdot10^{-5}$ | $\le 4.8\cdot10^{-5}$ | $1.06\cdot10^{-4}$ | $5.69\cdot10^{-4}$ | 1.76 (1.48) |
| A/A | 3.15–3.60 | 2.76–3.33 | $\ge 1.48$ (6/6) | $\le 5.0\cdot10^{-5}$ | $\le 5.5\cdot10^{-5}$ | $1.06\cdot10^{-4}$ | $6.32\cdot10^{-4}$ | 1.58 (1.36) |

- Reference criterion $\le 2\cdot10^{-4}$ holds on all 4 (smallest margin 1.71, S/A). Observed
  $p_z$ at the coarse level 2.44–3.79 ($\ge 1.42$ on every mode).
- At $n_z = 2$ the same rule gives S/A $1.88\cdot10^{-4}$ (margin 1.07), hence $n_z = 3$.
- $p$ rises from the coarser to the finer triplet (approach from below), except S/A mode 5, whose
  coarse triplet ($p = 8.21$, a near-crossing at $12\times6$: $|\Delta|$ equals the gap there) is
  skipped by the back-check.
- `ssh8` $384\times96\times1$ (the frame) gives S/A $7.96\cdot10^{-4}$; in-plane refinement
  approaches a one-layer floor of about $6.2\cdot10^{-4}$ (round 1 perf: $7.80$ at $384\times128$,
  $7.18$ at $512\times96$ against the $n_z = 2$ reference). The margins with the reference
  (1.14–1.48) are reported, not gated.
- Per class: hex27 1, 6, 43, 361–381 s (385k DOF, 25.7 GB at the finest), $n_z$ runs 6, 11, 16,
  43, 80 s, `ssh8` 66 s (8.9 GB): about 10 min, 40 min for 4.

### Skew plate

30°, clamped, $L/t = 20$; hex27 at $n_z = 3$, both in-plane directions cosine-graded toward the
clamped edges ($x_i = (1 - \cos(\pi i/n))/2$, nested); levels 6 (back-check), 12, 24, 48; $n_z$ runs
$12\times\{3,4,5\}$ (coarse) and $24\times\{3,4\}$ (mid); `ssh8` $192\times192\times1$ uniform.

- $p$ = 3.48–3.89 (12/24/48), 3.06–3.41 (6/12/24); back-check $\ge 1.39$ (6/6); Richardson
  $\le 2.0\cdot10^{-4}$, $n_z$ term $\le 9.3\cdot10^{-4}$ (observed $p_z$ 2.55–2.91); reference
  error $1.13\cdot10^{-3}$ (criterion $2\cdot10^{-3}$, margin 1.77).
- **Uniform levels are pre-asymptotic** (round 1 math): $p$ falls from 2.47–3.40 to 1.87–2.71 and
  the same rule underestimates the middle-level error (back-check 0.67–1.0), so the frame's uniform
  reference is replaced.
- `ssh8` $192\times192\times1$: worst $7.42\cdot10^{-3}$ (modes 5–6), margin 1.35 (1.18 with the
  reference); $128\times128$ gives 1.26.
- Modes 5 and 6 are $0.22\%$ apart in $f$ (hex27 48) and $0.23\%$ (`ssh8`), so sorted matching pairs
  them consistently; a swap would give $5.1$ and $9.7\cdot10^{-3}$, still within the gate.
- Times: hex27 0.6, 2, 17, 147 s (48: 190k DOF, 16.6 GB), $n_z$ runs 3, 5, 33 s; `ssh8` 28 s. About
  4 min.

### No locking

$6\times6\times1$, SS square, 16 modes, first 6 bending by z-parity; reference $48\times48\times1$.

- Error in $\lambda$, max over the 6: 0.640 at $L/t = 10$, 0.838 at 1000; ratio 1.31.
- Observed $p$ (12/24/48) 2.01–2.13. Every one of the 16 modes is cleanly even or odd (residuals
  $\le 10^{-10}$ at 1000, $\le 10^{-13}$ at 10); $\ge 8$ bending among 16 at every size.
- Floors $f_\lambda$: $3.2\cdot10^{-10}$ at 10, $1.0\cdot10^{-3}$ at 1000. Seconds in total.

## Rules

All differences are relative, in $f$ ($\Delta/f$ of the finer run), except no-locking (in
$\lambda$). $f_\lambda = 10\varepsilon\hat\lambda_\max/\lambda$ per mode is a $\lambda$ floor; using
it on $f$ errors is conservative by 2×.

- **Richardson (in-plane),** per mode, from three nested levels (ratio 2), $\Delta_1 = f_1 - f_2$,
  $\Delta_2 = f_2 - f_3$: require $\Delta_1\Delta_2 > 0$ and $p = \log_2(\Delta_1/\Delta_2) \ge 1$
  (else fail); estimate $1.25\,|\Delta_2|/(2^{\min(p, 4)} - 1)$ (4 is hex27's rate; 1.25 the GCI
  safety factor for three-level studies). No round-off branch (no gated mode is near round-off;
  the frame's branch misfired on the cylinder).
- **One-level-back check:** the same rule on the coarser triplet (levels 0, 1, 2) must bound the
  middle-level error $|f_2/f_3 - 1| + \text{est}_3$ for every mode whose coarser $p \in [1, 4]$
  (measured $\ge 1.39$); `analyze` returns the number of checked modes, and the tests pin it
  (S/A 5 of 6, all others 6 of 6).
- **Mode-crossing guard:** per mode at the finest pair, $|\Delta_2|/f_3$ is below the relative gap
  to its nearest neighbour at the finest level (measured $\le 0.10$ on the cylinder, 0.72 on the
  skew modes 5–6), so sorted matching cannot have swapped modes between the last two levels; the
  coarser pairs are covered by the $p$ checks (a crossing gives an erratic $p$, as S/A mode 5).
  The record holds no vectors, so no MAC.
- **$n_z$ term:** the thickness error converges at $p_z = 2\lambda_s = 1.42$, the rate of the
  clamped-free corner singularity in plane strain ($\nu = 0.3$, $\lambda_s \approx 0.711$); smooth
  modes converge faster, so the fixed rate is conservative. At an in-plane level with runs at
  $n_z$, $n_z + 1$: term $|\Delta_1|/(1 - (n_z/(n_z+1))^{1.42})$. It is evaluated at the coarse
  ($t_c$) and mid ($t_m$) levels and extrapolated in $h$ at the same rate:
  $t = 1.25\,(t_m + |t_m - t_c|/(2^{1.42} - 1))$. Check: the observed $p_z$ at the coarse level
  (runs $n_z$, $n_z + 1$, $n_z + 2$; `brentq` on
  $\frac{n_z^{-p} - (n_z+1)^{-p}}{(n_z+1)^{-p} - (n_z+2)^{-p}} = \Delta_1/\Delta_2$ over
  $[0.05, 20]$; no sign change in the bracket or non-monotone data fails explicitly) is
  $\ge 1.42$ on every mode.
- **Reference error** per mode: Richardson + $n_z$ term + $f_\lambda$ of the finest level; the max
  over modes meets the roadmap criterion ($\le 0.02\%$ cylinder, $\le 0.2\%$ skew), else stop and
  ask. Gates compare against the finest level.
- Every gated `ssh8` run asserts $2f_\lambda \le 0.1\times$ its gate.
- **No locking:** $e = \max_6|\lambda/\lambda_{48} - 1|$ per $L/t$; $\delta$ = the Richardson rule in
  $\lambda$ on 12/24/48 (max over the 6) + the larger floor of the $6\times6$ and $48\times48$ runs;
  gate $\frac{e_{1000} + (1 + e_{1000})\delta_{1000}}{e_{10} - (1 + e_{10})\delta_{10}} \le 1.5$
  (measured 1.37) and $\delta \le 0.1\,e$ at both (measured $e/\delta$ = 75 and 78; with the
  $10^{-3}$ floor inside $\delta$ this is the roadmap's "≫ 1e-4").
- **Srinivas–Rao:** per label, error $\le 10^{-3}$ against `plate_modes` on the full plate; the
  target is the class rank from `plate_modes` (12 modes) on the sub-plate, computed live.

## Layout

- `tests/helpers.py` gains `PAG`, `RZ`, `H`, `PLATES` (from `test_verification.py`, which imports
  them); `tests/exact.py` imports `from tests.helpers import tensor` (pytest has
  `pythonpath = ["."]`, and `tests/exact.py` has no other test-local import), so the benchmark can
  import `tests.exact`.
- **`benchmarks/shell_accuracy.py`** (`python -m benchmarks.shell_accuracy --parts srao cyl skew`,
  merged into `benchmarks/results/shell_accuracy.json`, per-run progress lines; runs sequential in
  one process, the peak is the largest single run):
  - Module constants: the label lists per (laminate, $L/t$), $q$, $k$, levels and $n_z$ sets.
  - Builders `srao_case(name, Lt, label, q, k)`, `cylinder(typ, cls, nt, na, nz)`,
    `skew(typ, n, nz)`, each returning (mesh, materials, bcs).
  - `run(builder, args, lmax=True)` solves one case: a fixed mode count (6, or 6 + rank + 1 on a
    zero-index strip, filtered), $\hat\lambda_\max$ unless `lmax=False` (the recompute tests use
    the recorded one). The record is a list of `{"builder", "args", "lam", "flam", "dof", "time",
    "digest", "removed"}`; each part stores its commit.
  - `analyze(rec) -> dict`: numbers only, per case: errors, Richardson and $n_z$ terms, reference
    errors, $p$, observed $p_z$, back-check ratios and counts, crossing ratios (Srinivas–Rao
    targets via `tests.exact.plate_modes`). The benchmark prints it; the tests hold the gate
    constants and assert on it.
  - **Zero-index filter** (`filter_strip(mesh, label, V)`): per mode, the free in-plane component
    ($u_y$ on an $(m, 0)$ strip, $u_x$ on $(0, n)$) on the two cut faces gives $s = \|u_{lo} -
    u_{hi}\|/\|u_{lo} + u_{hi}\|$; keep $s \le 10^{-6}$, drop $s \ge 10^6$, fail in between; assert
    the kept count $>$ the target rank and the target's component $\ge 0.1\|v\|$. The frame's
    assertion "no label $m' \ge 1$ below the target" is replaced by this classification (logged).
    The record keeps the filtered $\lambda$ and the count removed.
  - Reported only (printed, in `docs/verification.md`): one layered element per laminate (each
    label's quarter at the same $q$, one `ssh8` layer with the whole `Laminate`; about 1 min).
  - Parts: `srao` about 30 min, `cyl` about 40 min, `skew` about 4 min (sizing times plus
    $\hat\lambda_\max$); each under the 1 h limit.
- **Builder digest:** a SHA-256 over the mesh points, connectivity, block type, free mask, material
  ids, and per material $C$, $\rho$, ply angles and fractions of each run. The frame's source hash
  is replaced (logged): the per-path recomputes catch numeric changes in any module (`assemble`,
  kernels, `reference_modes` included), and the digests catch builder drift for every recorded run.

## Tests (`tests/test_shell_accuracy.py`)

- **Record test** (fast, about 2 s to rebuild every key without assembly): load the record;
  rebuild every key with the builders and assert the digest; assert the recorded key set equals
  the module constants and the label constants equal the live `plate_modes` lowest 10; `analyze`
  and assert every gate and reference criterion, the matched labels, the back-check counts, the
  crossing guard, the zero-index filter counts, and $2f_\lambda \le 0.1\times$ gate.
- **Slow recompute** (one per code path, `run(..., lmax=False)` against the record to
  $\max(10^{-9}, 2f_\lambda)$): `pag` $L/t = 100$ (1,5) (laminate, 90° ply, 31 s); `ortho` $L/t =
  100$ (2,4) (4 s); `pag` $L/t = 10$ (2,0) (zero-index strip and filter, a few seconds); cylinder S/A
  `ssh8` (66 s) and S/A $24\times12\times3$ hex27 (6 s); skew `ssh8` (28 s) and hex27 24 graded
  (1728 elements, several chunks, 17 s). About 2.5 min. The selection (one key per builder path,
  laminate, strip, element type, multi-chunk) is logged.
- **Fast tests** (small meshes, to $\max(10^{-10}, 2f_\lambda)$):
  - Estimators on synthetic data: $f_h = f^* + Ch^p$ ($p$ = 2, 3.5) recovers the exact error
    (times 1.25); non-monotone data and $p < 1$ raise; $p > 4$ uses 4. The observed $p_z$ recovers
    $p$ = 1.5, 3 from $f(n_z) = f^* + Cn_z^{-p}$ (closed form); a ratio outside the bracket raises.
    The $n_z$ term on $f = f^* + Ah^4 + Bn_z^{-1.42}(1 + h)$ bounds the exact thickness error.
  - `analyze` on a synthetic record with known answers (one case per family, values from the
    formulas above), so its wiring (level order, coarse vs mid, the floor) is tested without runs.
  - Srinivas–Rao, through `srao_case` and `filter_strip`: the full $2N\times N$ plate ($N = 4$,
    `pag`, $n_z = 3$) has an eigenvalue among its lowest 6 equal to the $N\times N$ sub-plate's
    (1,1); the quarter equals the sub-plate for (1,1).
  - Zero-index strip on a wide strip ($h = 2$, `pag`, $q = 4$), one test: spurious modes fall below
    the target and are removed; the kept modes equal the full sub-plate's $(m', 0)$ modes, the
    (1,0) one the full plate's. The wide strip is the only case with spurious modes below the
    target, so it is the filter's one check (the record holds no vectors).
  - Cylinder classes, through `cylinder`: a half model ($\theta \in [0, \pi]$, full length,
    $u_y = 0$ on both $\theta$ faces, both ends clamped, mirrored grading), `ssh8` $8\times4\times1$
    and hex27 $8\times4\times2$: its lowest 16 equal the sorted union of the 4 class spectra on the
    1/8 model; on the $y$-antisymmetric half model every mode below $0.9\lambda_T$ lies in that
    spectrum, where $\lambda_T = (\pi/L)^2 G/\rho \approx 0.949$ is the $n = 0$ torsion mode the 1/8
    model excludes (above every gated mode, $\le 0.560$), and one mode within 1% of $\lambda_T$ is
    the torsion mode. About 3 s.
  - Skew builder sanity: the corner angle is 30° and every boundary face is `"xyz"`.
  - **No locking** runs directly (6 s): the rules above, every mode classified even or odd,
    $\ge 6$ bending; the hex8 baseline printed.
  - `plate_modes` oracle at every (laminate, sub-plate) used, slow and deduplicated: $P = 24$
    equals $P = 12$ to $10^{-9}$ at $L/t = 10$ and $10^{-5}$ at 100 (round-off in the z-Ritz:
    measured $\le 3.6\cdot10^{-6}$, no trend in $P$), and `mmax = 10` gives the same target.
  - A smoke test: `run` on one tiny key per builder (no `analyze`; tiny levels break the rules).

## Docs

- `docs/verification.md`: "Solid-shell accuracy (PR 16c)": setups, levels, $k$ per ply and $n_z$
  per laminate and $L/t$, the tables from the record (time columns labelled), the rules (amending
  "Richardson extrapolation is not used" for PR 16's reference estimates), the stacking note, the
  one-layer floors, the reported one-layer laminate results; only 0° and 90° plies here (the
  ply-angle sign is covered by 16a). The frame's `plate_fe` generalization is superseded by
  `srao_case`.
- `docs/eigensolvers.md` ("The largest supported size" paragraph and the "PR 8 `eigsh`" bullet),
  all times including $\hat\lambda_\max$: thin hex27 meshes with `points` reach 385k free DOF at
  $n_z = 3$ (25.7 GB, 6 min); 190k at $n_z = 3$ on the skew plate (16.6 GB, 2.5 min); 276k at
  $n_z = 2$ (14.2–15.4 GB, 2.5 min). Memory grows with the DOF and $n_z$.
- `docs/elements.md` "Locking" points to the new section.
- ROADMAP status for PR 16c and the M4 line; a PR 18 Contents bullet for the `COUNT_STEP` refit at
  $\ge 1$M DOF; README status; `docs/open_items.md` (below, and the 16b Reader entry rewritten as
  "re-deferred to PR 18"). No deferred GPU checks.

## Changes from the frame and decisions (logged in `docs/open_items.md`)

1. Srinivas–Rao $L/t = 100$: $k = 3$ (`pag`), $k = 4$ (`ortho`); `ortho` $L/t = 10$ at $n_z = 24$.
2. The Richardson rule above (cap 4, factor 1.25, no round-off branch, back-check level, crossing
   guard).
3. Skew reference graded, levels 6 (back-check)/12/24/48; cylinder reference at $n_z = 3$ with
   $12\times6\times3$ added.
4. The $n_z$ term at the fixed corner rate 1.42, extrapolated in $h$; observed $p_z$ only checked.
5. `ssh8` meshes: cylinder $512\times96\times1$, skew $192\times192\times1$.
6. No-locking bounded-ratio gate; run directly, not from the record.
7. The source hash replaced by builder digests and per-path recomputes; the recompute selection.
8. The full-plate report dropped (the fast equivalence test proves sub-plate = full plate).
9. The strip assertion "no label $m' \ge 1$ below the target" replaced by the filter's
   classification and kept-count assertion.
10. The half-cylinder antisymmetric check limited below the torsion mode.
11. Oracle tolerance $10^{-5}$ at $L/t = 100$; `mmax = 10` kept.
12. The `eigsh` limit stated per $n_z$ (the frame's "280k / 14 GB" does not hold at $n_z = 3$).
13. The stacking note ($n_z = 2$ no better than 1 on bending; one layer soft at clamped edges).
14. `COUNT_STEP` refit (open_items, 16b): re-deferred to PR 18 (performance), which fits the
    estimators at $\ge 1$M DOF; 16c is accuracy only.
