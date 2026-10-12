# PR 9b plan: chunked LOBPCG, locked basis, low-memory mode, workspace estimator

Branch `pr-09b-chunks`, from `main` after PR 9a (split and dependencies: [`pr-09a-lobpcg.md`](pr-09a-lobpcg.md)).

## Scope

The PR 9 roadmap items of the 9b row: Contents 1 (the 6-block mode), 3 (chunked computation, optionally fp32 deflation basis, fp64 returned modes) and 4 (workspace estimator); Docs: the memory model. Tests:
- the guard-vector error;
- the λ/angle runs in both modes and the 6-block match;
- the estimator at 1M DOF;
- projector memory;
- the chunked fp32 run;
- in-place insertion and the nearly-full assert;
- step temp independent of capacity and within the estimator;
- the 6-block cost;
- the compile count on a chunked solve.

**Deviations from the roadmap** (measured below; recorded in the roadmap note and `docs/eigensolvers.md`):
1. **The low-memory mode holds 11 blocks with a real operator, not 6.**
   - Its state is $(X, P)$; every K and M product is recomputed and used at once, so it needs 6 applies per iteration instead of 2.
   - With a diagonal operator, so without operator workspace, the compiled step holds 7 blocks. With the hex20 operator it holds 11.0 at 182k and 1.08M DOF, for $m = 15$ and $48$. An apply inside the step costs about 4 blocks of its own (input copy, accumulator, output, chunk workspace).
   - That is still 31% below the 9-block step (16 blocks).
   - The name "6-block mode" stays, for the roadmap's terminology. The Memory budget rows are updated: "LOBPCG, 6 blocks" becomes 11 blocks, "9 blocks" becomes 16.
2. **The rigid modes stay a separate fp64 block** with $MR$ precomputed, as in 9a. They do not fill the first basis columns, as the PR 7 note says.
   - The locked basis $V$ holds only locked modes, in fp64 or fp32.
   - Projecting against $V$ needs $MW$ of the undeflated $W$, which costs one extra M apply per iteration (storing $MV$ would double the basis).
   - Keeping $R$ apart means single-chunk free-body solves pay no extra apply (measured +15% time with $R$ in the basis), and $R$ stays fp64 under an fp32 basis.
3. **A chunk that does not converge ends the solve.**
   - The modes locked so far plus that chunk's wanted columns are returned, so fewer than `n_modes` can come back, with `info["converged"]` False.
   - Locking unconverged vectors would corrupt the deflation of every later chunk.
4. **`V` is returned as a NumPy fp64 array** (host), assembled from the fp64 host copies made at each lock. The roadmap's "returned locked modes come from an fp64 host copy" applies to every solve; 9a returned a device array.

## Measured facts (prototype `scratchpad/p9b/proto.py`; jax 0.11.2, fp64, M3 Max CPU)

Material, box and BCs as in 9a (isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$, box $(1, 0.83, 0.67)$, `xmin` clamped unless free).

- **Chunked solves** (hex20 $6^3$, 30 modes in 3 chunks of 10, block $m = 15$, tol $10^{-10}$, Jacobi):

  | case | basis | chunk iterations | max rel. λ error | $\max\lvert V^\top MV - I\rvert$ |
  |---|---|---|---|---|
  | clamped | fp64 | 313 / 276 / 260 | $7\cdot10^{-13}$ | $9\cdot10^{-15}$ |
  | free | fp64 | 349 / 340 / 293 | $4\cdot10^{-14}$ | $5\cdot10^{-15}$ |
  | clamped, tol $10^{-6}$ | fp32 | 197 / 175 / 158, all converged | $1\cdot10^{-12}$ | $9\cdot10^{-9}$ |
  | free, tol $10^{-6}$ | fp32 | 218 / 216 / 181, all converged | $8\cdot10^{-13}$ | $4\cdot10^{-9}$ |
  | clamped, tol $10^{-10}$ | fp32 | chunks 2–3 stagnate at the floor | $7\cdot10^{-13}$ | $1\cdot10^{-8}$ |

  A single chunk of 30 modes ($m = 36$) needs 266 iterations: 2.4 s, against 4.9 s for the 3 chunks. Chunking trades time for memory.
- **Clusters at a chunk boundary** (free cube $4^3$, rigid deflation; elastic eigenvalues with exact multiplicities 2, 3, 3, 2, 1, 3, 3, …):
  - chunk 4 with the default 5 guards: chunks lock 5, 5 and 4 modes (each extended to the end of its triple), λ error $10^{-14}$;
  - chunk 4 or 2 with 1 guard: the cluster reaches the block end, and the guard error is raised;
  - chunk 5 with 1 guard (no cluster split): runs, but every chunk hits `maxit` (the one guard gives a gap ratio of 0.97).
- **Step memory**, temp + alias in blocks of $(n, m)$ fp64:

  | operator | n | m | 9-block | low-memory |
  |---|---|---|---|---|
  | diagonal | 100k | 17 | 16.00 | 7.00 |
  | hex20 $24^3$ | 182k | 15 / 48 | 16.00 / 16.00 | 11.71 / 11.01 |
  | hex20 $44^3$ | 1.08M | 15 / 48 | 16.00 / 16.00 | 11.01 / 11.00 |
  | hex20 $14^3$ | 38k | 15 / 48 | 22.66 / 18.93 | 19.60 / 15.22 |

  - Both are unchanged between basis capacities $2m$ and $32m$, fp64 or fp32.
  - At small n the operator's chunk workspace dominates (9-block: 55.6 blocks at 8k, $m = 15$, where the PR 5 workspace is 23 blocks).
  - A first low-memory variant that kept KX and MX in the state measured 13.0 blocks at the same time per iteration.
- **Projector** $Y \leftarrow Y - V_c(V_c^\top MY)$ over column chunks of width $c$, basis stored as $(n, \text{capacity})$:
  - temp is exactly $(c + m)\,n \cdot 8$ B for fp64 and fp32 bases, independent of capacity;
  - at 182k DOF, 200 locked columns, $m = 15$: 36 ms with $c = 16$ and 25 ms with $c = 64$, against 24 ms unchunked fp64.
  - The transposed layout $(\text{capacity}, n)$ was 2–3× slower (60 / 72 / 26 ms).
- **Time per iteration** (hex20 $24^3$, 182k DOF, warm):

  | m | locked | 9-block | low-memory |
  |---|---|---|---|
  | 15 | 0 | 74 ms | 138 ms |
  | 15 | 200 | 135 ms | 199 ms |
  | 48 | 0 | 219 ms | 348 ms |
  | 48 | 200 | 315 ms | 442 ms |

  With 200 locked modes, the extra M apply and the projector add 61 ms at $m = 15$ and 96 ms at $m = 48$.
- **Accuracy of the low-memory mode** (12 modes, tol $10^{-10}$): same iterations ±2 as the 9-block mode, λ matching it to $\le 3\cdot10^{-13}$ (hex20 clamped and free $6^3$, hex27 $4^3$, hex8 $10^3$).
- **Process peak at 1.08M DOF** (hex20 $44^3$, 40 modes, $m = 48$, capacity 96, 3 iterations; harness `peak_bytes`):

  | mode | measured | storage + basis + max(build, blocks·n·m·8) |
  |---|---|---|
  | 9-block | 9.48 GB | 8.75 GB (ratio 0.92) |
  | low-memory | 7.70 GB | 6.69 GB (ratio 0.87) |

  The difference (0.7–1.0 GB) is about the host copy of the locked modes (0.34 GB) plus one block (0.41 GB). The estimator adds both.

## Files

| File | Change |
|---|---|
| `src/jaxmodal/eigen/lobpcg.py` | locked basis and projector, chunked driver, low-memory step, `block_size` / `capacity` helpers |
| `src/jaxmodal/memory.py` | `lobpcg_memory` |
| `tests/test_lobpcg.py` | the tests below |
| `docs/eigensolvers.md`, `docs/ROADMAP.md`, `README.md` | see Docs |

## API

```python
lam, V, info = lobpcg(op, free, n_modes, tol=1e-6, R=None, precond=None, X0=None, maxit=500,
                      seed=0, chunk=None, guards=None, basis_dtype="float64", low_memory=False)
```

- **`chunk`:** wanted modes per chunk, default `n_modes` (one chunk). `guards` defaults to $\max(5, \lfloor \text{chunk}/5\rfloor)$, and the block size is $m = \text{chunk} + \text{guards}$, one shape for every chunk (the last chunk wants fewer).
- **`maxit`** is per chunk. `X0` seeds the first chunk only.
- **`basis_dtype`:** `"float64"` or `"float32"`, the device dtype of the locked basis.
- **`low_memory`:** the 6-block mode (a static step variant).
- **Returns:**
  - `lam` (NumPy, ascending) and `V` (NumPy fp64, $(n_\mathrm{dof}, \cdot)$), `n_modes` columns unless a chunk failed (deviation 3);
  - `info`:
    - `iterations` (total), `converged` (every chunk converged), `stagnated` (any chunk), `residuals` and `thresholds` (per returned mode, from the explicit confirmation at lock time), `p_drops`, `refills`, `norm_K`, `norm_M`, `history` (concatenated), `time`;
    - `chunks`: a list of dicts `want`, `locked`, `iterations`, `converged`, `stagnated`.
- **Errors:**
  - `ValueError` for `chunk` < 1 or > `n_modes`, `guards` < 1, a bad `basis_dtype`, or a block larger than the free DOFs minus the rank of R;
  - the guard error below.

`jaxmodal.memory.lobpcg_memory(op_memory, n_dof, n_modes, chunk=None, guards=None, low_memory=False, basis_dtype="float64", n_rigid=0)`:
- `op_memory` is a callable $m \mapsto$ the operator estimate, e.g. `lambda m: unstructured_memory(mesh, m)`.
- Returns `step`, `basis`, `rigid`, `host` and `total`, in bytes (Memory model).

## Algorithm

**Basis.**
- **Layout:** a device array `B` of shape $(n, \text{capacity})$ in `basis_dtype`, zero-initialized, plus a host count.
- **Capacity:** $\lceil (n_\mathrm{modes} + m) / c \rceil\,c$ with the projector chunk width $c = m$ (static). The last lock starts at a count $\le n_\mathrm{modes} - 1$ and writes $m$ columns, so writes never clamp, and unused columns stay zero, so reads never clamp onto earlier columns.
- **Insertion:** `_insert(B, X, count)` is a jitted `dynamic_update_slice` along axis 1 with `B` donated and an `int32` start. It always writes $m$ columns: X with the unlocked columns zeroed.
- **Overflow check:** the host checks count + m ≤ capacity before every insert and raises `RuntimeError("locked basis full ...")`; the solver's capacity rule never triggers it.
- **Host copy:** each lock also keeps an fp64 host copy of its columns (`np.asarray`), and the returned `V` is assembled from these.

**Projector.**
- `_project(B, count, Y, MY)` runs a `fori_loop` over $\lceil \text{count}/c \rceil$ column chunks, with traced bound and `int32` starts.
- Each chunk is `dynamic_slice`d, upcast to fp64, and applied fused: $Y \leftarrow Y - V_c(V_c^\top MY)$, with MY passed in.
- One classical pass suffices because the coefficients come from the original $MY$ and $V$ is M-orthonormal.
- **Deflation** of a block (the start block, and $W$ every iteration): when count > 0, a `lax.cond` applies M to the undeflated block and projects it. Then the 9a deflation against $R$ runs, $W \leftarrow W - R(MR^\top W)$, and the result is masked.
- The count is traced, so locking never recompiles.

**Chunked driver.** `_solve` loops over chunks; the 9a iteration loop runs inside it, unchanged except for the cluster rule.
1. `want` = min(chunk, modes still missing). The start block is the previous chunk's unlocked columns (seed), padded with random columns, then deflated against $R$ and $V$ and SVQB-orthonormalized (9a refill rule).
2. **Iterate as in 9a** on the first $j$ columns, with $j$ = `want` at the start.
3. **Cluster rule.** Once the first `want` columns pass their thresholds, $j$ extends while
   $$\lambda_{j+1} - \lambda_j < 100\,t_j\,|\lambda_j|,$$
   i.e. the roadmap's "relative gap < 100·tol", with the acceptance threshold $t_j \ge$ tol so that floor-accepted pairs are covered. The extended columns must then converge too.
   - The rule is evaluated only after the first `want` columns have converged: early Ritz values are far from converged and could form spurious clusters.
4. **Guard error.** If $j$ reaches $m$ (the cluster's end is not visible inside the block), raise `ValueError`: "a cluster at the chunk boundary (from mode i) fills the block: it needs more than g guard vectors; pass guards >= g + 1 or a chunk size that ends in a gap".
5. **On convergence:** lock columns $1..j$ (insert, host copy, residuals and thresholds from the explicit confirmation), and carry $X_{:, j+1..m}$ as the next seed.
6. **On a no-progress or `maxit` exit:** stop (deviation 3) and return the locked modes plus this chunk's first `want` columns.

The returned set is the first `n_modes` locked modes. A cluster that straddles `n_modes` is computed whole, but the return is cut at `n_modes`.

**Low-memory step** `lobpcg_step_low(state, op, mask, precond, R, MR, B, count, consts)`, with state $(X, P, \mathrm{act}_P, \lambda)$ donated.
1. $KX$, then $MX$; residuals and thresholds of the input $X$; $W = T(KX - MX\lambda) \odot a$; the Gram blocks $X^\top KX$ and $P^\top KX$; KX dies.
2. Deflate $W$.
3. Compute $MP$. M-orthogonalize $W$ against $X$ (with $MX$) and $P$ (with $MP$), twice. Form $X^\top MX$, $P^\top MX$, $P^\top MP$; MX and MP die.
4. $MW$ and the SVQB of $W$ (drop rule as 9a); the Gram blocks with $MW$; MW dies.
5. $KW$ and its Gram blocks; then $KP$ and $P^\top KP$.
6. Rayleigh–Ritz, the pivot test and `_p_coeffs` as in 9a; sequenced update of $X$ and $P$ only.

- `optimization_barrier`s between the phases keep each product's lifetime to its phase.
- The step reports the residuals of the input $X$. When the wanted columns pass, the host confirms the updated $X$ explicitly (`_confirm_low`: $KX$, $MX$ and residuals) before accepting, as in 9a's exit check.
- There are no implicit products, so there is no recompute.

**Compile count.** One step variant per solve, 9-block or low-memory, compiled once. The count, the number of wanted columns and the new data are all traced or host-side, so a chunked solve, and a second solve with new data, compile no further steps.

## Memory model

$b = 8nm$ is one block, with $n = n_\mathrm{dof}$.

- **`step`:** $\max(S\,b,\ A\,b + w)$.
  - $S = 16$ for the 9-block step, $S = 7$ for the low-memory step: the diagonal-operator figures, which are pinned.
  - $A$ is the blocks alive during an apply plus the apply's own vectors.
  - $w$ is the operator's `workspace` at width $m$.
  - $A$ and a factor on $w$ are fitted on the measured step temp at 3 sizes per mode; the fit is an upper envelope, and the test below checks it.
- **`basis`:** $n \cdot \text{capacity} \cdot (8 \text{ or } 4)$ B.
- **`rigid`:** $2 \cdot 8\,n\,n_\mathrm{rigid}$ ($R$ and $MR$).
- **`host`:** $8\,n\,n_\mathrm{modes}$ (the fp64 copies of the locked modes) plus one block (the asynchronously freed output of the last step).
- **`total`:** overhead + operator storage + basis + rigid + host + max(operator build, step).

The 9-block and low-memory Memory budget rows and the "Max locked modes" row are recomputed with this model in `docs/ROADMAP.md`.

## Tests (`tests/test_lobpcg.py`)

1. **Chunked accuracy.** Clamped and free hex20 $6^3$, 30 modes in 3 chunks of 10, tol $10^{-10}$, fp64 basis, in both modes. 9a's `check_modes` applies: λ error, principal angles, M-orthonormality over all modes, constrained rows 0, residuals. Also `info["chunks"]`: 3 chunks, all converged.
2. **Both modes on the 9a cases.** `test_modes` is parametrized over `low_memory`. The low-memory λ match the 9-block λ to $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$.
3. **Cluster at a chunk boundary.** Free cube $4^3$, 12 modes, chunk 4, tol $10^{-10}$:
   - the locked counts are 5, 5, … (each chunk extended over its triple);
   - multiplicities equal the reference's.
4. **Guard error.** The same cube with chunk 4 and `guards=1` raises `ValueError` matching "guards >= 2".
5. **fp32 basis.** Clamped hex20 $6^3$, 30 modes, chunk 10 (3 chunks), tol $10^{-6}$:
   - λ match the fp64-basis run to $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$ for relative gaps $\ge 10^{-2}$, $10\,$tol otherwise;
   - $\|V^\top MV - I\| \le 10^{-5}$;
   - `count_below` at the midpoint of $\lambda_{30}$ and $\lambda_{31}$ (the reference's) equals 30, which rules out duplicates and misses.
6. **Projector.**
   - With an fp32 basis at $n = 10^5$, $m = 17$: `memory_analysis().temp_size_in_bytes` $\le 1.01(c + m)\,n \cdot 8$, equal for capacities $2m$ and $8m$.
   - A count that is not a multiple of $c$ gives the unchunked fp64 projector's result: to $10^{-14}$ relative with an fp64 basis, and exactly against the upcast basis for fp32.
7. **In-place insertion.** After `_insert`, the donated input `is_deleted()`, and `unsafe_buffer_pointer()` of the result equals the input's.
8. **Nearly full.** The host lock with count + m > capacity raises `RuntimeError`, and the basis is bit-identical afterwards.
9. **Step temp vs capacity.** For both modes, with the diagonal operator: temp + alias equal for capacities $2m$ and $8m$. The 9-block step stays $\le 16.5$ blocks (9a test) and the low-memory step $\le 7.5$.
10. **Step temp within the estimator.** For both modes with the hex20 operator at $24^3$ (182k DOF) and $14^3$ (38k), $m = 15$ and 48: temp + alias ≤ `lobpcg_memory(...)["step"]`. `memory_analysis` already takes the maximum over the `lax.cond` branches.
11. **Estimator at 1M DOF** (slow, macOS). Hex20 $44^3$, 40 modes, $m = 48$, 3 iterations, both modes: estimate $\ge 2$ GB and within 20% of the measured `peak_bytes` (harness, as PR 5's estimator test).
12. **Compile count.** In the `COMPILES` subprocess, a chunked solve (3 chunks) compiles the step once, and a second chunked solve with new data compiles 0. Also once for `low_memory=True`, counted by its own `fun_name`.
13. **Failed chunk.** `maxit=5` with 3 chunks:
    - the solve stops after chunk 1 with `converged` False;
    - `len(lam)` = chunk;
    - `info["chunks"]` has one entry.
14. **Errors.** `chunk=0`, `chunk > n_modes`, `guards=0` and `basis_dtype="float16"` raise `ValueError`.
15. **Low-memory cost** (not asserted): the time per iteration of both modes at 182k DOF goes into `docs/eigensolvers.md` (Measured facts table).

**Mutants**, each to be killed by the test given:

| mutant | killed by |
|---|---|
| projector skipped (W not deflated against V) | 1, 5 |
| projector uses the deflated $MY$ again (two-step) or a wrong chunk start | 6 |
| capacity without the $+m$ | 8, 1 |
| cluster rule off | 3 |
| cluster rule evaluated before convergence (spurious extension) | 3 (counts), 1 |
| guard check off (silent lock inside a cluster) | 4 |
| seed not carried (random restart per chunk) | none: performance only, expected survivor |
| low-memory: residuals of the updated X without the explicit confirmation | 2 |
| insert not donated | 7 |
| failed chunk locked and continued | 13 |
| basis cast to fp32 for the fp64 option | 1 (orthonormality) |

## Docs

- **`docs/eigensolvers.md`:**
  - chunked solve, locking, the cluster rule and guard error;
  - basis layout and projector, the fp32 basis and its attainable residual;
  - the low-memory mode (phases, cost);
  - memory model and estimator, with the measured tables;
  - deviations;
  - Validation entries.
- **`docs/ROADMAP.md`:**
  - the 9b status line;
  - the four deviations in the PR 9 note;
  - the Memory budget rows (16 and 11 blocks, Max locked modes) recomputed with the estimator.
- **`README.md`:** status line and a chunked-solve snippet, e.g. `lobpcg(op, free, 10, chunk=4, R=R)`.

## Not covered

- `solve_modes`, examples, Mac budgets (9c).
- Automatic choice of the chunk size, low-memory mode or fp32 basis from the estimator (9c's `solve_modes`).
- Reallocating a full basis.
- An `np.memmap` for the host copies (roadmap: only if the locked modes exceed RAM).
