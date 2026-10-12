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

**Deviations from the roadmap** (measured below; recorded in the roadmap note and `docs/eigensolvers.md`; 5 and 6 come from plan review):
1. **The low-memory mode holds 11 blocks with a real operator, not 6.**
   - Its state is $(X, P)$; every K and M product is recomputed and used at once, so it needs 6 applies per iteration instead of 2.
   - With a diagonal operator, so without operator workspace, the compiled step holds 7 blocks. With the hex20 operator it holds 11.0 at 182k and 1.08M DOF, for $m = 15$ and $48$. An apply inside the step costs about 4 blocks of its own (input copy, accumulator, output, chunk workspace).
   - That is still 31% below the 9-block step (16 blocks). Variants measured by review (no donation, fewer or no barriers, fused masks) all hold 11–12 blocks: XLA copies the donated X and P early so their buffers can take the new X and P, and JAX offers no control over that copy.
   - The name "6-block mode" stays, for the roadmap's terminology. The Memory budget rows are updated: "LOBPCG, 6 blocks" becomes 11 blocks, "9 blocks" becomes 16. Both counts already include the Rayleigh–Ritz update temporaries and the projector temporaries, so the separate "Rayleigh–Ritz update temporaries" row is folded into them (no double counting).
   - **Consequences for the roadmap budgets:**
     - 10M hex20 on 80 GB (column "10M", $m = 30$, block 2.4 GB): the low-memory step goes from 14 + 5 GB (6 blocks plus the RR row) to 26.4 GB, which takes about 7 GB from the fp32 basis (40 MB per mode). The basis limit drops from ≈ 900–1200 to ≈ 700–1000 modes; the exact figures come from the estimator and replace the Max-locked-modes row and the line in the Shift-invert section ("basis limit ≈ 900–1200 modes").
     - 100M voxel on one 80 GB GPU ($m = 8$, block 6.4 GB): 11 blocks are 70.4 GB before the preconditioner (12.8 + 5.2 GB), so it does not fit. With the diagonal-operator figure (7 blocks, 44.8 GB) plus the stencil's own apply workspace it might. Decision: the 100M paragraph says that a single-GPU 100M solve fits only if the voxel stencil's in-step cost, measured on the GPU, keeps the low-memory step at ≤ 7.5 blocks at $m = 8$; otherwise 100M needs multi-GPU (PR 19). The Deferred GPU check of PR 11 ("a 100M matvec plus the LOBPCG and preconditioner workspace fits") is reworded to that measurement.
     - 100M free bodies: $R$ and $MR$ in fp64 take $2 \cdot 8 \cdot 10^8 \cdot 6 = 9.6$ GB (deviation 2), so they need multi-GPU. Appending $R$ to an fp32 basis instead would halve that and cost one M apply per iteration; it is left to PR 19, where the 100M budget is decided.
2. **The rigid modes stay a separate fp64 block** with $MR$ precomputed, as in 9a. They do not fill the first basis columns, as the PR 7 note says.
   - The locked basis $V$ holds only locked modes, in fp64 or fp32.
   - Projecting against $V$ needs $MW$ of the undeflated $W$, which costs one extra M apply per iteration (storing $MV$ would double the basis).
   - Keeping $R$ apart means single-chunk free-body solves pay no extra apply (measured +15% time with $R$ in the basis), and $R$ stays fp64 under an fp32 basis.
3. **A chunk that does not converge ends the solve.**
   - Only a no-progress or `maxit` exit stops. A floor-accepted chunk (`converged` and `stagnated` both True, 9a's outcome table) counts as converged and is locked.
   - The modes locked so far plus that chunk's wanted columns are returned, so fewer than `n_modes` can come back, with `info["converged"]` False. The failed chunk's columns are Ritz vectors (upper bounds, M-orthonormal); their residuals and thresholds come from 9a's `_confirm`, which runs on every non-converged exit.
   - Locking unconverged vectors would corrupt the deflation of every later chunk.
4. **`V` is returned as a NumPy fp64 array** (host). The roadmap's "returned locked modes come from an fp64 host copy" applies to every solve; 9a returned a device array.
   - fp32 basis: each lock keeps an fp64 host copy of its columns, and `V` is assembled from them.
   - fp64 basis: no host copies; `V = np.asarray(B[:, :n_modes])` at the end.
5. **Cluster rule threshold** (interpretation of the roadmap's "relative gap < 100·tol"): the relative gap is $(\lambda_{j+1} - \lambda_j)/|\lambda_j|$, and it is compared with $100\,t_j$, where $t_j \ge$ tol is column $j$'s acceptance threshold. Floor-accepted pairs and the fp32 basis term below are then covered.
6. **fp32 basis threshold term.** The locked vectors rounded to fp32 are M-orthonormal only to about $u_{32} = 2^{-24} \approx 6\cdot10^{-8}$, so every chunk after the first has a residual floor near $10^{-8}$ (review probe: chunks 2–3 stuck at $1.0\cdot10^{-8}$ and $6.2\cdot10^{-9}$ at tol $10^{-10}$). With an fp32 basis and count > 0, the acceptance threshold becomes $t_i = \max(\text{tol}, \text{floor}_i, 2u_{32})$, about $1.2\cdot10^{-7}$; such chunks floor-accept instead of stopping the solve (deviation 3). Without this term, a tol below about $10^{-8}$ would silently return only the first chunk's modes.

## Measured facts (prototype `scratchpad/p9b/proto.py`; jax 0.11.2, fp64, M3 Max CPU)

Material, box and BCs as in 9a (isotropic $E = 3$, $\nu = 0.3$, $\rho = 2$, box $(1, 0.83, 0.67)$, `xmin` clamped unless free).

- **Chunked solves** (hex20 $6^3$, 30 modes in 3 chunks of 10, block $m = 15$, tol $10^{-10}$, Jacobi):

  | case | basis | chunk iterations | max rel. λ error | $\max\lvert V^\top MV - I\rvert$ |
  |---|---|---|---|---|
  | clamped | fp64 | 313 / 276 / 260 | $7\cdot10^{-13}$ | $9\cdot10^{-15}$ |
  | free | fp64 | 349 / 340 / 293 | $4\cdot10^{-14}$ | $5\cdot10^{-15}$ |
  | clamped, tol $10^{-6}$ | fp32 | 197 / 175 / 158, all converged | $1\cdot10^{-12}$ | $9\cdot10^{-9}$ |
  | free, tol $10^{-6}$ | fp32 | 218 / 216 / 181, all converged | $8\cdot10^{-13}$ | $4\cdot10^{-9}$ |
  | clamped, tol $10^{-10}$ | fp32 | chunks 2–3 stagnate at the floor (prototype locks after the stall; the planned driver floor-accepts by deviation 6) | $7\cdot10^{-13}$ | $1\cdot10^{-8}$ |

  - The fp32 basis reaches residuals of about $10^{-8}$ (one classical projector pass leaves about $u_{32}$ of locked content in W); use tol $\ge 10^{-6}$ with it. With an fp64 basis, one pass is exact to round-off.
  - Review probe (correctness): X does not drift toward the locked modes during a chunk; $\max|V^\top M X|$ stays at $10^{-8}$ (fp32) and $10^{-15}$ (fp64) over 800 iterations, since Rayleigh–Ritz gains locked content only to second order. Deflating W alone suffices.

  A single chunk of 30 modes ($m = 36$) needs 266 iterations: 2.4 s, against 4.9 s for the 3 chunks. Chunking trades time for memory.
- **Clusters at a chunk boundary** (free cube $4^3$, rigid deflation; elastic eigenvalues with exact multiplicities 2, 3, 3, 2, 1, 3, 3, …):
  - chunk 4 with the default 5 guards: chunks lock 5, 5 and 4 modes (each extended to the end of its triple), λ error $10^{-14}$;
  - chunk 4 or 2 with 1 guard: the cluster reaches the block end, and the guard error is raised;
  - chunk 5 with 1 guard (no cluster split): runs, but every chunk hits `maxit` (the one guard gives a gap ratio of 0.97).
- **Step memory**, temp + alias in blocks of $(n, m)$ fp64:

  | operator | n | m | 9-block | low-memory |
  |---|---|---|---|---|
  | diagonal | 100k | 17 | 16.00 | 7.00 |
  | hex20 $24^3$ | 182k | 15 / 48 | 16.00 / 16.00 | 11.71 / 11.01 (quoted as "11.0–11.7 blocks") |
  | hex20 $44^3$ | 1.08M | 15 / 48 | 16.00 / 16.00 | 11.01 / 11.00 |
  | hex20 $14^3$ | 38k | 15 / 48 | 22.66 / 18.93 | 19.60 / 15.22 |

  - Both are unchanged between basis capacities $2m$ and $32m$, fp64 or fp32.
  - At small n the operator's chunk workspace dominates (9-block: 55.6 blocks at 8k, $m = 15$, where the PR 5 workspace is 23 blocks).
  - A first low-memory variant that kept KX and MX in the state measured 13.0 blocks at the same time per iteration.
- **Projector** $Y \leftarrow Y - V_c(V_c^\top MY)$ over column chunks of width $c$, basis stored as $(n, \text{capacity})$:
  - temp is exactly $(c + m)\,n \cdot 8$ B for fp64 and fp32 bases, independent of capacity;
  - at 182k DOF, 200 locked columns, $m = 15$: 36 ms with $c = 16$ and 25 ms with $c = 64$, against 24 ms unchunked fp64.
  - The transposed layout $(\text{capacity}, n)$ was 2–3× slower: 60 / 72 / 26 ms at $c = 16 / 64 / 256$, against the column layout's 36 / 25 / 11 ms.
  - Chunk width (performance review, projector alone): at small $m$ it is bound by re-reading Y and MY per chunk.

    | m | locked | $c = m$ | $2m$ | $4m$ |
    |---|---|---|---|---|
    | 15 | 200 | 40.6 ms | 32.7 | 25.5 |
    | 15 | 1000 | 221 ms | 167 | 105 |
    | 48 | 200 | 43 ms | 41 | 47 |

    The 9-block step's temp stays 16.00 blocks for $c \le 4m$ (19.01 at $8m$): deflation sits below the update-phase peak. The low-memory step pays +0.3 blocks at $2m$ and +2.3 at $4m$.
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
| `src/jaxmodal/eigen/lobpcg.py` | locked basis and projector, chunked driver, low-memory step, private `_block_size` / `_capacity` helpers |
| `src/jaxmodal/memory.py` | `lobpcg_memory` |
| `tests/test_lobpcg.py` | the tests below |
| `docs/eigensolvers.md`, `docs/ROADMAP.md`, `README.md` | see Docs |

## API

```python
lam, V, info = lobpcg(op, free, n_modes, tol=1e-6, R=None, precond=None, X0=None, maxit=500,
                      seed=0, *, chunk=None, guards=None, basis_dtype="float64", low_memory=False)
```

The new parameters are keyword-only.

- **`chunk`:** wanted modes per chunk, default `n_modes` (one chunk). `guards` defaults to $\max(5, \lfloor \text{chunk}/5\rfloor)$, and the block size is $m = \text{chunk} + \text{guards}$, one shape for every chunk (the last chunk wants fewer).
- **`maxit`** is per chunk. `X0` seeds the first chunk only.
- **`basis_dtype`:** `"float64"` or `"float32"`, the device dtype of the locked basis.
- **`low_memory`:** the 6-block mode (a static step variant).
- **Returns:**
  - `lam` (NumPy, ascending) and `V` (NumPy fp64, $(n_\mathrm{dof}, \cdot)$), `n_modes` columns unless a chunk failed (deviation 3);
  - `info`:
    - `iterations` (total), `converged` (every chunk converged), `stagnated` (any chunk), `residuals` and `thresholds` (per returned mode, from the explicit confirmation at lock time, or from `_confirm` for a failed chunk's columns), `p_drops`, `refills`, `norm_K`, `norm_M`, `history` (concatenated), `time`;
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
- **Capacity:** $\lceil (n_\mathrm{modes} + m) / c \rceil\,c$ with the static projector chunk width $c = \max(m, \min(4m, 64))$ for the 9-block step and $c = m$ for the low-memory step (performance table above). The last lock starts at a count $\le n_\mathrm{modes} - 1$ and writes $m$ columns, so writes never clamp, and unused columns stay zero, so reads never clamp onto earlier columns.
- **Insertion:** `_insert(B, X, count)` is a jitted `dynamic_update_slice` along axis 1 with `B` donated and an `int32` start. It always writes $m$ columns: X with the unlocked columns zeroed.
- **Lock helper:** `_lock(B, count, X, j)` checks count + m ≤ capacity on the host *before* inserting and raises `RuntimeError("locked basis full ...")`, leaving B intact; otherwise it zeroes columns $j..m$ of X, calls `_insert` and returns the new B. The solver's capacity rule never triggers the error.
- **Host copy** (fp32 basis only): each lock keeps an fp64 host copy of its columns (`np.asarray`), and the returned `V` is assembled from these (deviation 4).

**Projector.**
- `_project(B, count, Y, MY)` runs a `fori_loop` over $\lceil \text{count}/c \rceil$ column chunks, with traced bound and `int32` starts.
- Each chunk is `dynamic_slice`d, upcast to fp64, and applied fused: $Y \leftarrow Y - V_c(V_c^\top MY)$, with MY passed in.
- One classical pass suffices because the coefficients come from the original $MY$ and $V$ is M-orthonormal: exactly (to round-off) for an fp64 basis; for fp32 it leaves about $u_{32}$ of locked content, the floor of deviation 6.
- An extra M apply per deflation is cheaper than storing $MV$ (which doubles the basis); deflating after the block's K and M products would instead need both products again.
- **Deflation** of a block (the start block, and $W$ every iteration): when count > 0, a `lax.cond` applies M to the undeflated block and projects it. Then the 9a deflation against $R$ runs, $W \leftarrow W - R(MR^\top W)$, and the result is masked.
- The count is traced, so locking never recompiles.

**Chunked driver.** `_solve` loops over chunks; the 9a iteration loop runs inside it, unchanged except for the cluster rule.
1. `want` = min(chunk, modes still missing). The start block is the previous chunk's unlocked columns (seed), padded with random columns, then deflated against $R$ and $V$ and SVQB-orthonormalized (9a refill rule).
2. **Iterate as in 9a** on the first $j$ columns, with $j$ = `want` at the start.
3. **Cluster rule.** Once the first `want` columns pass their thresholds, $j$ extends while
   $$\lambda_{j+1} - \lambda_j < 100\,t_j\,|\lambda_j|,$$
   i.e. the roadmap's "relative gap < 100·tol" (deviation 5), with $t_j$ column $j$'s acceptance threshold. The extended columns must then converge too.
   - The rule is evaluated only after the first `want` columns have converged: early Ritz values are far from converged and could form spurious clusters.
   - From then on it is re-evaluated at every iteration, not once: Ritz values decrease monotonically, so $j$ only grows, and the guard error can fire on a later iteration. Guard Ritz values are reliable here because their error is about $r^2\lambda/\text{gap}$ (quadratic).
4. **Guard error.** If $j$ reaches $m$ (the cluster's end is not visible inside the block), raise `ValueError`: "a cluster at the chunk boundary (from mode i) fills the block: it needs more than g guard vectors; pass guards >= g + 1 or a chunk size that ends in a gap".
5. **On convergence** (including floor acceptance): lock columns $1..j$ with `_lock` (residuals and thresholds from the explicit confirmation), and carry $X_{:, j+1..m}$ as the next seed.
6. **On a no-progress or `maxit` exit:** stop (deviation 3) and return the locked modes plus this chunk's first `want` columns.

`maxit` counts per chunk. The fp32 threshold term (deviation 6) is passed through the 9a constants when the basis is fp32 and count > 0.

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

**Compile count.** One step variant per solve, 9-block (`lobpcg_step`) or low-memory (`lobpcg_step_low`), compiled once; compile counts filter by exact `fun_name`, since one name is a prefix of the other. The count, the number of wanted columns and the new data are all traced or host-side, so a chunked solve, and a second solve with new data, compile no further steps.

## Memory model

$b = 8nm$ is one block, with $n = n_\mathrm{dof}$.

- **`step`:** $\max(S\,b,\ A\,b + w)$.
  - $S = 16$ for the 9-block step, $S = 7$ for the low-memory step: the diagonal-operator figures, which are pinned.
  - $A$ is the blocks alive during an apply plus the apply's own vectors.
  - $w$ is the operator's `workspace` at width $m$.
  - $A$ and a factor on $w$ are fitted on the measured step temp (at the chosen $c$) at hex20 $24^3$, $44^3$ (from the prototype) and $14^3$, $m = 15$ and 48; the fit is an upper envelope. The fitted constants and their margins are stated in `docs/eigensolvers.md`. Test 10 checks a fit size and one held-out size.
- **`basis`:** $n \cdot \text{capacity} \cdot (8 \text{ or } 4)$ B.
- **`rigid`:** $2 \cdot 8\,n\,n_\mathrm{rigid}$ ($R$ and $MR$).
- **`host`:** one block (the asynchronously freed output of the last step), plus $8\,n\,n_\mathrm{modes}$ (the fp64 host copies, fp32 basis; for fp64 the same amount is the final `np.asarray` of the basis).
- **`total`:** overhead + operator storage + basis + rigid + host + max(operator build, step).

The 9-block and low-memory Memory budget rows and the "Max locked modes" row are recomputed with this model in `docs/ROADMAP.md`, and the "Rayleigh–Ritz update temporaries" row is folded into them (deviation 1).

## Tests (`tests/test_lobpcg.py`)

1. **Chunked accuracy.** Clamped and free hex20 $6^3$, 30 modes in 3 chunks of 10, tol $10^{-10}$, fp64 basis, default `maxit` (per chunk; the 3 chunks need about 900 iterations in total), in both modes. 9a's `check_modes` applies: λ error, principal angles, M-orthonormality over all modes, constrained rows 0, residuals. Also `info["chunks"]`: 3 chunks, all converged.
2. **Both modes on the 9a cases.** `test_modes` is parametrized over `low_memory`. The low-memory λ match the 9-block λ to $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$. 9a's P-invariant test (7b) also runs on the low-memory step.
3. **Cluster at a chunk boundary.** Free cube $4^3$, 12 modes, chunk 4, tol $10^{-10}$:
   - `[c["locked"] for c in info["chunks"]] == [5, 5, 4]` (each chunk extended over its triple; the last triple, modes 12–14, straddles `n_modes`);
   - `len(lam) == 12` (cut at `n_modes`);
   - multiplicities equal the reference's.
4. **Guard error.** The same cube with chunk 4 and `guards=1` raises `ValueError` matching "guards >= 2".
5. **fp32 basis.** Clamped hex20 $6^3$, 30 modes, chunk 10 (3 chunks), tol $10^{-6}$:
   - λ match the fp64-basis run to $\max(10^{-8}, 10\varepsilon\lambda_\max/\lambda)$ for relative gaps $\ge 10^{-2}$, $10\,$tol otherwise;
   - $\|V^\top MV - I\| \le 10^{-5}$;
   - with $\sigma = (\lambda^\mathrm{ref}_{30} + \lambda^\mathrm{ref}_{31})/2$, after asserting the reference relative gap there is $\ge 10^{-3}$: `np.sum(lam < σ) == count_below(K, M, σ, free) == 30`. With the orthonormality check this rules out misses and duplicates.
5b. **fp32 basis at a tight tol.** The same case at tol $10^{-10}$: 3 chunks, `converged` and `stagnated` both True, 30 modes, λ within $\max(10^{-8}, f_\lambda)$ of the reference (deviation 6).
6. **Projector.**
   - With an fp32 basis at $n = 10^5$, $m = 17$: `memory_analysis().temp_size_in_bytes` $\le 1.01(c + m)\,n \cdot 8$, equal for capacities $2m$ and $8m$.
   - A count that is not a multiple of $c$ (e.g. $3c + 5$, basis zero past count) gives the unchunked fp64 projector's result to $10^{-14}$ relative, for an fp64 basis and for an fp32 basis (upcast in the reference). The chunked loop rounds in a different order, so exact equality is not expected.
7. **In-place insertion.** After `_insert`, the donated input `is_deleted()`, and `unsafe_buffer_pointer()` of the result equals the input's.
8. **Nearly full.** `_lock(B, count, X, j)` with count + m > capacity raises `RuntimeError`; afterwards `not B.is_deleted()` and `np.asarray(B)` equals a pre-copy bit for bit.
9. **Step temp vs capacity.** For both modes, with the diagonal operator, at the chosen $c$: temp + alias equal for capacities $2m$ and $8m$. The 9-block step stays $\le 16.5$ blocks (9a test) and the low-memory step $\le 7.5$.
10. **Step temp within the estimator.** For both modes with the hex20 operator at $24^3$ (182k DOF, a fit size), $m = 15$ and 48, and at the held-out $8^3$ (where the PR 5 workspace dominates), $m = 15$; compile only: temp + alias ≤ `lobpcg_memory(...)["step"]`. `memory_analysis` already takes the maximum over the `lax.cond` branches.
11. **Estimator at 1M DOF** (slow, macOS). Hex20 $44^3$, 40 modes, $m = 48$, 3 iterations, both modes: estimate $\ge 2$ GB and within 20% of the measured `peak_bytes` (harness, as PR 5's estimator test).
12. **Compile count.** In the `COMPILES` subprocess, a chunked solve (3 chunks) compiles `lobpcg_step` once, and a second chunked solve with new data compiles 0. Also once for `low_memory=True`, counted by the exact `fun_name` `lobpcg_step_low`.
13. **Failed chunk.** Clamped hex20 $6^3$, 3 chunks of 10, `X0` = the first 15 reference modes (chunk 1 converges in about 1 iteration), `maxit=5` (chunk 2 needs about 276):
    - `len(lam) == 20`, `converged` False;
    - `[c["converged"] for c in info["chunks"]] == [True, False]`;
    - the first chunk's λ match the reference to $f_\lambda$.
14. **Errors.** `chunk=0`, `chunk > n_modes`, `guards=0`, `basis_dtype="float16"`, and chunk + guards larger than the free DOFs minus the rank of R raise `ValueError`.
15. **Low-memory cost** (not asserted): the time per iteration of both modes at 182k DOF goes into `docs/eigensolvers.md` (Measured facts table).

**Mutants**, each to be killed by the test given:

| mutant | killed by |
|---|---|
| projector skipped (W not deflated against V) | 1, 5 |
| projector uses the deflated $MY$ again (two-step) or a wrong chunk start | 6 |
| projector loop bound `count // c` instead of the ceiling | 6 |
| fp32 basis used without the upcast | 6 |
| unlocked columns not zeroed at insert | 1, 3 |
| capacity without the $+m$ | 8, 1 |
| insert before the capacity check | 8 (`is_deleted`) |
| cluster rule off | 3 |
| cluster rule evaluated once, $j$ frozen at the first converged iteration | 3, else recorded as an expected survivor |
| straddling cluster not cut at `n_modes` | 3 |
| guard check off (silent lock inside a cluster) | 4 |
| fp32 threshold term off | 5b |
| seed not carried (random restart per chunk) | none: performance only, expected survivor |
| low-memory: residuals of the updated X without the explicit confirmation | 2 |
| low-memory barriers removed | 9 (if the 7.5 bound catches it; measured otherwise and recorded) |
| insert not donated | 7 |
| failed chunk locked and continued | 13 |
| failed chunk drops its want columns | 13 |
| `maxit` counted over the whole solve instead of per chunk | 1 |
| basis cast to fp32 for the fp64 option | 1 (orthonormality) |

## Docs

- **`docs/eigensolvers.md`:**
  - chunked solve, locking, the cluster rule (deviation 5) and guard error;
  - basis layout and projector, the chunk width rule, the fp32 basis, its attainable residual (≈ $10^{-8}$, use tol $\ge 10^{-6}$) and threshold term;
  - the low-memory mode (phases, cost, why 11 blocks);
  - memory model and estimator, the fitted constants and margins, the measured tables;
  - deviations;
  - the "Returns" bullet: `V` is a NumPy fp64 array on the host;
  - Validation entries.
- **`docs/ROADMAP.md`:**
  - the 9b status line (line 16), including that PR 7's note "R fills the first basis columns" is superseded (deviation 2);
  - line 74: the two step variants are the 9-block and the low-memory step (11 blocks);
  - the six deviations in the PR 9 note;
  - the Memory budget rows (16 and 11 blocks, Max locked modes) recomputed with the estimator, the Rayleigh–Ritz row folded in, and the note under the table updated;
  - the 10M basis limit in the Shift-invert section (line 90);
  - the 100M paragraph (line 175) and PR 11's Deferred GPU check (line 463), as decided in deviation 1.
- **`README.md`:** status line and a chunked-solve snippet, e.g. `lobpcg(op, free, 10, chunk=4, R=R)`.

## Not covered

- `solve_modes`, examples, Mac budgets (9c).
- Automatic choice of the chunk size, low-memory mode or fp32 basis from the estimator (9c's `solve_modes`).
- Reallocating a full basis.
- An `np.memmap` for the host copies (roadmap: only if the locked modes exceed RAM).
- Dropping the low-memory step's KP and MP applies via the coefficient-space invariant (about 138 → 100 ms per iteration at $m = 15$, with a recompute schedule); a lever for 9c's budgets if the 1.9× cost matters.
- Appending $R$ to an fp32 basis for memory-bound free-body runs (PR 19).
