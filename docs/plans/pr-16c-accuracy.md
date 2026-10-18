# PR 16c plan: solid-shell accuracy suite

Branch `pr-16c-accuracy`. The design frame is `docs/plans/pr-16-solid-shell.md` ("The accuracy
suite"); this plan fixes meshes, levels and the benchmark layout from the sizing study below,
and lists where it departs from the frame. Roadmap: PR 16 tests "0.1% target", "No locking",
"Distorted meshes", and the `docs/verification.md` entries.

## Revisions

- rev 1: first draft after the sizing study.

## Sizing study (shipped `ssh8`, 2026-10-18)

All runs use the shipped code (`solid_shell`, `assemble`, `reference_modes`), on the Mac (64 GB).
Errors are relative in $f$. Peaks are the process maximum RSS, including the $\hat\lambda_\max$
`eigsh(which="LA", tol=1e-3)`.

**Srinivas–Rao** (each label on its sub-plate quarter, $q$ in-plane elements per quarter side,
$k$ stacked `ssh8` elements per ply; the frame's quarter rules, class ranks from `plate_modes`
on the sub-plate):

| laminate | $L/t$ | $q$, $k$ ($n_z$) | worst (label, error) | margin | per run | total |
|---|---|---|---|---|---|---|
| `pag` | 10 | 32, 12 (36) | (1,3) $5.77\cdot10^{-4}$ | 1.73 | 58–95 s, 10.2 GB | 7.5 min |
| `ortho` | 10 | 32, 36 (36) | (3,1) $3.36\cdot10^{-4}$ | 2.98 | 60–95 s, 10.2 GB | 10.5 min |
| `pag` | 100 | 48, 1 (3) | (2,5) $9.90\cdot10^{-4}$ | 1.01 | 2 s | |
| `pag` | 100 | 48, 3 (9) | (1,5) $4.19\cdot10^{-4}$ | 2.39 | 15–31 s | 3.2 min |
| `ortho` | 100 | 48, 1 (1) | (2,4) $8.53\cdot10^{-4}$ | 1.17 | 0.5 s | |
| `ortho` | 100 | 48, 4 (4) | (2,4) $3.37\cdot10^{-4}$ | 2.97 | 3–4 s | 0.5 min |

- $L/t = 10$ reproduces the frame's prototype to three digits (`pag` (1,3) $5.77\cdot10^{-4}$,
  `ortho` (3,1) $3.36\cdot10^{-4}$); every one of the lowest 10 is matched (rank 0 on its
  quarter class, (1,1,1) rank 3 on the (1,1) quarter). The zero-index sub-plates ((1,0), (0,1),
  (2,0), (0,2)) give $1.0\cdot10^{-4}$ (one strip, in-plane error only), 2368 DOF, 0.1 s.
- $L/t = 100$ is thickness-dominated, not in-plane: `pag` (2,5) is $9.35\cdot10^{-4}$ at
  $q = 64$ and $8.95\cdot10^{-4}$ at $q = 96$ ($k = 1$), but $6.19\cdot10^{-4}$ at $k = 2$ and
  $2.67\cdot10^{-4}$ at $k = 4$ ($q = 48$). The frame's $k = 1$ leaves margin 1.01, so $k$ is
  raised per laminate (the frame allows it, logged): `pag` $k = 3$ ($n_z = 9$), `ortho` $k = 4$.
- Lowest 10 at $L/t = 100$: `pag` (1,1), (1,2), (1,3), (2,1), (2,2), (1,4), (2,3), (2,4), (1,5),
  (2,5); `ortho` (1,1), (1,2), (2,1), (2,2), (1,3), (2,3), (3,1), (1,4), (3,2), (2,4). No
  zero-index or family-1 label, so every sub-plate is a nonzero-index quarter.

**Clamped cylinder** ($R/t = 50$, $L/R = 2$; hex27 levels $24\times12\times2$, $48\times24\times2$,
$96\times48\times2$, axially cosine-graded; $n_z$ runs $48\times24\times\{2,3,4\}$; `ssh8`
$384\times96\times1$ uniform):

| class | observed $p$ | $p_z$ | reference error (max over 6) | `ssh8` worst | + reference |
|---|---|---|---|---|---|
| S/S | 3.37–3.72 | 1.6–2 | $1.16\cdot10^{-4}$ | $5.95\cdot10^{-4}$ (mode 6) | $7.11\cdot10^{-4}$ |
| S/A | 3.18–3.59 | 1.83–2 | $1.49\cdot10^{-4}$ | $7.96\cdot10^{-4}$ (mode 4) | $9.45\cdot10^{-4}$ |
| A/S | 3.31–3.72 | 1.6–2 | $1.24\cdot10^{-4}$ | $6.06\cdot10^{-4}$ (mode 4) | $7.31\cdot10^{-4}$ |
| A/A | 3.15–3.59 | 1.85–2 | $1.28\cdot10^{-4}$ | $6.48\cdot10^{-4}$ (mode 4) | $7.71\cdot10^{-4}$ |

- All 24 modes are monotone with $p \in [3.15, 3.72]$; the reference criterion ($\le 0.02\%$)
  holds on all 4 classes (margin 1.34), dominated by the $n_z$ term ($\le 9.5\cdot10^{-5}$).
- `ssh8` margin 1.26 against the finest level (the gate), 1.06 with the reference error added.
- Per class: hex27 levels 3, 19, 146 s (finest 14.2–15.4 GB), $n_z$ runs 43 and 80 s, `ssh8`
  38 s (6 GB): about 5.5 min per class, 22 min for 4.
- **The frame's round-off rule fails here:** the finest hex27 level has $f_\lambda = 3.3\cdot10^{-6}$
  (S/S, A/S), so "$|\Delta| \le 100 f_\lambda$" catches real differences of $4\cdot10^{-5}$ and
  replaces their $|\Delta|/(2^p - 1)$ by $|\Delta|$, giving $3.7\cdot10^{-4}$ (fail) although
  every mode has a clean $p$. Revised rule below.

**Skew plate** (30°, clamped, $L/t = 20$; hex27 at $n_z = 3$):

- Levels 12/24/48: $p = 1.87$–$2.71$, Richardson $\le 5.6\cdot10^{-4}$, $n_z$ term (24 at 3, 4, 5)
  $\le 3.0\cdot10^{-4}$; reference error $7.1\cdot10^{-4}$ (criterion $2\cdot10^{-3}$, margin 2.8).
- Levels 6/12/24 fail the criterion ($2.1\cdot10^{-3}$), so the finest level is 48.
- `ssh8` $128\times128\times1$: worst $7.53\cdot10^{-3}$ (modes 5–6) against the 48 level, margin
  1.33 (1.22 with the reference error).
- Times: hex27 48×48×3 154 s, 16.6 GB (190k DOF); the rest 0.5–58 s; `ssh8` 9 s. About 4.5 min.

**No locking** ($6\times6\times1$, SS square, 16 modes, first 6 bending by z-parity):

- Error in $\lambda$ against $48\times48\times1$, max over the 6: 0.640 at $L/t = 10$, 0.838 at
  1000; ratio 1.31 (gate 1.5). Observed $p$ (12/24/48) 2.01–2.13; Richardson + floor
  $6.8\cdot10^{-3}$ and $8.8\cdot10^{-3}$, $\le 0.1\times$ the max error ($6.4$, $8.4\cdot10^{-2}$).
  Floors: $3.2\cdot10^{-10}$ at 10, $1.0\cdot10^{-3}$ at 1000. At least 8 bending modes among 16
  at every size. Seconds.

## Changes from the design frame (logged)

1. **Srinivas–Rao $L/t = 100$:** $k = 3$ per ply for `pag` ($n_z = 9$), $k = 4$ for `ortho`
   ($n_z = 4$), $q = 48$; the frame's $k = 1$ has margin 1.01 (`pag`) and 1.17 (`ortho`).
2. **Richardson rule order:** per mode, the observed-$p$ estimate $|\Delta_2|/(2^p - 1)$ is used
   when $\Delta_1\Delta_2 > 0$, $|\Delta_2| < |\Delta_1|$ and $p = \log_2(\Delta_1/\Delta_2) \in
   [1, 5]$. Only when that check fails is the round-off fallback $|\Delta_2| + f_\lambda$ used, and
   then only if $|\Delta_2| \le 100 f_\lambda$; otherwise the test fails. (The frame applied the
   round-off branch first.)
3. **Skew levels:** 12/24/48 (the frame listed 6/12/24/48; 6 is not used).
4. **No-locking estimate:** compared with $0.1\times$ the gated error (max over the 6), at each $L/t$.
5. **`eigsh` limit:** `docs/eigensolvers.md` lines 300 and 783 say "further on plates"; they
   gain the measured thin-mesh sizes: hex27 cylinder 276k DOF in 146 s, 15.4 GB, hex27 skew 190k
   DOF ($n_z = 3$) in 154 s, 16.6 GB, with the $\hat\lambda_\max$ solve.

## Common rules (from the frame, unchanged unless listed above)

- `reference_modes` on `assemble`; relative error in $f$; no upper-bound checks.
- Gates compare against the finest reference level. Its per-mode error bound is the in-plane
  Richardson estimate (rule 2) plus the $n_z$ term plus the finest level's $f_\lambda$; the max over
  modes must meet the roadmap criterion ($\le 0.02\%$ cylinder, $\le 0.2\%$ skew), else stop and
  ask.
- $n_z$ term: runs at $n_z$, $n_z + 1$, $n_z + 2$ at the middle level; per mode observed $p_z$
  (monotone, $\ge 1$, else fail), clamped to $\le 2$; term $|\Delta_1|/(1 - (n_z/(n_z+1))^{p_z})$.
- $f_\lambda = 10\varepsilon\hat\lambda_\max/\lambda$ per run; every gated run asserts
  $2f_\lambda \le 0.1\times$ its gate.
- `plate_modes` oracle checks at each sub-plate size and class used: $P = 24$ equals $P = 12$ to
  $10^{-9}$, and `mmax` + 2 gives the same target.

## Layout

**Benchmark** `benchmarks/shell_accuracy.py` (`--parts srao cases`, merged into
`benchmarks/results/shell_accuracy.json` like `shift_invert_study.py`; per-run progress lines):

- `srao`: the 2 laminates × 2 $L/t$ above, each label's quarter run: about 22 min.
- `cases`: cylinder (4 classes), skew, no-locking: about 27 min.
- Each run records its raw $\lambda$ (all computed modes kept), $\hat\lambda_\max$, DOF, time,
  and the mesh parameters; the record holds a SHA-256 of `src/jaxmodal/elements/shell.py`,
  `src/jaxmodal/materials/elastic.py` (`ply_table`) and `src/jaxmodal/mesh/shell.py`, and the commit.
- One module-level function per case builds the mesh, BCs and materials; tests import them
  (one source for both).

**Tests** `tests/test_shell_accuracy.py`:

- Slow, from the record (no solves except the recompute): assert the hash matches; recompute every
  error, every reference criterion and every gate from the raw values with the rules above
  (Srinivas–Rao per label ≤ $10^{-3}$, cylinder ≤ $10^{-3}$, skew ≤ $10^{-2}$, no-locking ratio
  ≤ 1.5 and its estimate rule; $2f_\lambda \le 0.1\times$ gate). Assert the matched labels and the
  zero-index filter counts.
- Slow recompute: the smallest-margin gated case per test (`pag` (1,3) at $L/t = 10$ (99 s,
  10 GB), cylinder S/A `ssh8` (38 s), skew `ssh8` (9 s), no-locking (seconds)) plus the coarsest
  reference level of each (cylinder S/A $24\times12\times2$, skew 12), against the record to
  $\max(10^{-9}, 2f_\lambda)$ relative.
- Fast equivalence tests (small meshes, $N \le 8$, $n_z \le 4$, to $\max(10^{-10}, 2f_\lambda)$):
  - Srinivas–Rao full plate $2N\times N$ (2,1) equals the $N\times N$ sub-plate (1,1);
  - the quarter equals the full sub-plate for (1,1), (1,0) and (0,1) (zero-index sub-plates at full
    extent in that direction), and the quarter filter keeps exactly the $m'$-class modes;
  - the cylinder classes: the union of the 4 class spectra on a coarse 1/8 model equals the
    $y$-symmetric part of a coarse 1/4 model's spectrum (excluded families listed);
  - the oracle checks above.
- Reported only (printed by the benchmark, in `docs/verification.md`): one layered element per
  laminate (Srinivas–Rao), the full plate at a stated mesh, the hex8 no-locking baseline.

## Docs

- `docs/verification.md`: a "Solid-shell accuracy (PR 16c)" section: setups, levels, $n_z$ per
  laminate and $L/t$, the tables above from the record, the reference-error rules (amending
  "Richardson extrapolation is not used" for PR 16's reference estimates), the one-layer floors.
- `docs/eigensolvers.md` lines 300 and 783 (change 5).
- ROADMAP status for PR 16c and the M4 line; README status; `docs/open_items.md` (changes 1–5).

## Decisions (without user input, logged)

- Changes 1–5 above.
- The record hash covers the three element/laminate/mesh-conversion files (not the solver).
- The recompute uses `pag` (1,3) at $L/t = 10$ as the Srinivas–Rao smallest-margin case (1.73),
  and the S/A cylinder (1.26); margins come from the record.
