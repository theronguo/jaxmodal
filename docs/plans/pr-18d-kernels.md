# PR 18 plan: performance pass, part d: kernels

Branch `pr-18d-kernels` from `main` (1c6b651, PR 18c merged). Roadmap: "PR 18d: Kernels".

## Revisions

- Rev 1: first draft.
- Rev 2 (plan review round 1: correctness, tests, docs not satisfied; simplicity, performance satisfied):
  - Morton reference test on the quantized keys, plus `check_renumbered`; `bits` dropped.
  - `lru_cache` callers (`test_amg_scan_cached`, `strategy_study.py:254`) listed; lifetime test by `weakref.ref`.
  - fp64 polish: the API effects (lever, band `c_max`) and the docstrings listed; "no change to solver results" narrowed.
  - Voxel `parts`: `(nz, ny, nx)` reshape, no dead guard, a non-cubic test case; host peak measured (93 against 352 B per element).
  - The roofline test is a fast formula check; an implementation-bytes column; the GPU peak defined as the measured fp64 matmul.
  - Smaller benchmarks; structure counts next to compile times; bimodal probes and the roadmap rewording; `_default_budget` catches a raising `memory_stats()`; `_chunked` order spelled out; docs map.
- Rev 3 (round 2: tests satisfied; correctness, simplicity, docs, performance not satisfied):
  - fp64 hex polish = `POLISH64` (1.25) × the fp32 polish term, checked on the 11 recorded `polish64` rows; no study rerun (simplicity). `test_polish_lever_hex` listed.
  - `cost_analysis` counts a scan body once: the flop check runs on one-trip meshes (asserted), with both bounds; never used in the benchmark.
  - Fusion report walks every computation reachable from ENTRY (loop bodies), not ENTRY alone.
  - Stored-Ke implementation bytes $2400 + 2 \cdot 4608$ B per hex8 element (the unpack is materialized: 52% of the triad at m = 1); full-Ke storage recorded as a GPU decision.
  - Byte counts corrected (stencil `phase`, no double `node`; per-node vector bytes $48m$); the roofline timing excludes first-touch page faults.
  - Bimodal: a setting is adopted only if it also holds for production runs; a parent-side footprint timeline and interleaved baselines; the harness described as it is.
  - Voxel `parts` labels ordered by lowest element index: exactly the generic path's labels.
  - Stencil equivalence after `morton` (exact on a masked $5 \times 4 \times 3$ probe); Morton element order in the reference test; non-cubic spacing.
  - Cache tests by membership and `len(_CACHE[mesh])`.
  - Docs: the missed locations (preconditioners 595/597/107, eigensolvers 417/983, open_items 38/42/266/283, ROADMAP 806, operators 44/247/253, postprocessing 212, mesh 38/48).
- Rev 4 (round 3: simplicity, docs satisfied; correctness, tests, performance not satisfied):
  - `POLISH64` = 1.05, not 1.25: today's max(9-block, low) already bounds 10 of the 11 rows; $\max(\text{low}, 1.05 \cdot \text{polish})$ bounds all 11 (lowest 1.046) and raises the step by at most 4.5% (m = 120 only); TIGHT fp64 ≈ 1.26 (1.25 gave 1.50). The loop term is not scaled. `test_lobpcg_memory_max` listed.
  - Bimodal: a live-bytes curve (`jax.live_arrays()` at LOBPCG's host chunk boundaries, probe-only) next to the footprint timeline; wall times recorded; setting caps.
  - Roofline: decimal MB (46.0 MB, 16.5 flop/B); the unpack temporary as formula (18.9 MB) and measured (20.8 MB); "through the memory hierarchy" (the chunk fits the 48 MB SLC) with a chunk-512 probe; a bare 24×24 GEMM ceiling.
  - `cost_analysis` check: stencil on a cubic $12^3$ (small non-cubic grids reach 2.4×), bounds per kernel (stencil 1.5, unstructured 1.25), one trip asserted by "no `while` in the compiled HLO".
  - Docs: ROADMAP :794 / :800 / :801 / :663–666, preconditioners :107 for $P_0^\top$, operators :244, open_items 183 and 284 outcomes, boundary_conditions :53 replaced.
- Rev 5 (round 4: simplicity satisfied; one major each from correctness, tests, docs, performance):
  - `test_hex_polish_below_step` listed (rewritten as a max(low, polish) bound); the whole fp32 term (with $K$) scaled by 1.05; numbers corrected (max(9-block, low) bounds all 11 rows, lowest max(low, polish) 1.042, TIGHT 1.27–1.28, the `low_memory` step up to +6.2%).
  - One-trip check on the `while(` opcode with a multi-trip control; explicit `chunk=n_el`.
  - `test_amg_scan_cached` on a fresh mesh; lifetime without a global count.
  - Bimodal: per-iteration live bytes (the solve is one chunk), the `sys.modules` patch, the temporaries reading, unprobed baselines.
  - `conn` bytes, hex20 implementation bytes, small doc citations.
- Rev 6 (round 5: simplicity, tests, docs, performance satisfied; correctness one major):
  - The step increases are stated on the recorded rows (m ≤ 120), plus the 200-mode figures (k = 240, box hex20 $36^3$: step +8.2%, total +7.3%, `low_memory` step +10%) and hex8 $32^3$ m = 60 (+5.3%); `low_memory` percentages corrected.
  - `test_amg_step` keeps polish-free step assertions and adds a separate fp64 polish bound; TIGHT with both bounds; `test_lobpcg_memory_max`'s phrases named.
  - Bimodal: settings without the wrapper, a wrapper-only group, fp32 AMG; the polling-thread run dropped.
  - Small fixes: "Ke only" intensity, the chunk-512 reading, hex20 by hand, `test_default_budget` extended, docs (preconditioners :611, verification :303, eigensolvers :979, ROADMAP :798, open_items 38/42).
- Rev 7 (round 6: simplicity, tests, docs, performance satisfied; correctness two majors):
  - `test_ladder_real` and `test_polish_lever_hex`'s budget listed (a reviewer ran the suite with the term); the whole suite runs with the term before review.
  - `low_memory` no longer lowers the fp64 total at 200 modes on hex20 $36^3$ (stated, open_items 284).
  - Headline roofline percentage on lower-bound bytes; a conditional allocator probe; small wording (chunk key, `_morton_keys`, the `ndimage` import, ROADMAP :659, eigensolvers :414).

## Goal (18d)

Measure the kernels against the machine (roofline, fusion, compile time), add Morton ordering as a helper, and close the non-band follow-ups that PRs 15d–18c deferred here: per-mesh caches, a pipelined chunk loop, the Metal budget fallback, the voxel part finder, an fp64 hex polish term, the bimodal peak investigation. Eigenpairs do not change; the fp64 hex AMG memory estimate grows by the polish term, which can change a band strategy or chunk chosen from it (below).

## Findings (prototypes, CPU fp64, M3 Max, `benchmarks/results/system.json`: triad 185 GB/s, fp64 matmul 490 GFLOP/s)

### Ordering (apply_K, scatter path, m = 1 and 16, and the AMG V-cycle)

- **Morton vs RCM vs grid:** on box hex20 $44^3$ (812k DOF) Morton is not faster than RCM or grid order on the scatter path (within ±3%) and makes the gather path about 11× slower (Morton's jumps between octants widen the gather windows). The median element node-index span: grid 8146, RCM 12399, Morton 326 (local in space, not in the index range a window covers).
- **Shuffled meshes** (random node and per-block element permutations; the worst case for a Gmsh or Abaqus file) cost +30–100% on apply_K and +50% on the AMG V-cycle. RCM recovers both (0.66 s at 358k nodes; about 3 s at 10M DOF, `docs/mesh.md:48`).
- **Grid order on a plate** (`ssh8` $256^2$, 396k DOF): RCM leaves apply_K (23.7 / 23.1 ms at m = 16), the AMG build (5.48 / 5.69 s) and the V-cycle (382 / 381 ms) unchanged, although the median node span is 16771 in grid order against 367 after RCM.
- The voxel `stencil` on a Morton-renumbered masked voxel mesh equals the permuted original exactly (it works on the grid through the renumbered voxel maps).

### Roofline (first numbers from recorded results and review probes)

Lower-bound bytes count each stored or streamed array once and the vectors once per apply (read $u$, write the output): per node $4 + 48m$ B (`node` id, vectors), plus per cell or element the stored data.

- **Voxel stencil**, $87^3$, m = 1: 1152 flops per cell, 0.759 GFLOP per apply; per cell 16 B (`cell` 4, `phase` 4, $s$ 8): 46.0 MB (decimal, as everywhere in `performance.md`), intensity 16.5 flop/B, above $490/185 = 2.65$, so the roofline is the fp64 peak. Its GEMM is $24 \times 24$ by $(24, N)$, which reaches about 280 GFLOP/s (`docs/operators.md:242`): the `roofline` part times that bare GEMM as an attainable ceiling next to the peak. In a plain process an apply takes 15.3 ms (`docs/operators.md:244`; the 18.4 ms of `stencil_slab.json` includes about 3 ms of first-touch page faults): 50 GFLOP/s, about 10%. The implementation's traffic is the corner stack and shifted adds through U and Y (about $4 \cdot 24 \cdot 8\,m$ B per cell, `docs/operators.md`, "Why this layout").
- **Stored-Ke apply** (hex8, packed upper triangle, 300 doubles): 1152 flops per element and m, 2400 B of Ke (+ 32 B `conn`) per element: 0.48 flop/B at m = 1 on Ke bytes only (0.46 with `conn` and the per-node vectors, the lower bound `kernels.json` records), a Ke-bytes cap of about 89 GFLOP/s (hex20 0.49 flop/B, + 80 B `conn`); the intensity grows with m (ridge near m = 6 for hex8). The unpack `Ke[:, _packed_index(k)]` is materialized ($576 \cdot 4096 \cdot 8$ B = 18.9 MB per 4096-element chunk; 20.8 MB of measured temporaries at m = 1), so the implementation moves $2400 + 2 \cdot 4608 = 11616$ B per element through the memory hierarchy plus the gathered $u$ and the scattered output ($2 \cdot 192m$ B each; hex20 $14640 + 2 \cdot 28800$ B, its 1024-element chunk unpacks 29.5 MB, also within the SLC): at $48^3$, m = 1, 13.2 ms, 9.7 GFLOP/s, 20 GB/s of Ke traffic but about 97 GB/s (52% of the triad) through the hierarchy. The chunk fits the 48 MB SLC, so DRAM traffic may be lower: the `roofline` part also times chunk 512 (2.4 MB, L2-sized); faster at 512 means the 4096 chunk is limited by bandwidth beyond L2 (SLC to DRAM), which the GPU decision notes; not faster is inconclusive (8× the loop trips); the trip count is recorded beside the result. Storing the full Ke (4608 B, 1.9× the storage) or a packed symmetric matvec would roughly halve the m = 1 traffic: recorded as a GPU-side decision (deferred check), not changed on CPU (storage is the 10M reach's limit).
- `cost_analysis()["flops"]` counts a `lax.scan` / while body once, not per trip (a 64-step scan of 24×24 matvecs reports 1178 flops against 73728): usable only on one-trip programs.

### Follow-ups

- **`unstructured_memory` (gather):** `layout` costs 0.22 s at 812k DOF (half windows, half inverse map) and 3.2 s at 10.2M per call, about 35 s per band call there. The estimator reads only scalars (per block `type`, `n_pad`, `chunk`, `k`, `W`; `path`, `n_nodes`, `R`, `n_ranges`, `valence`, the inverse-map size).
- **`gmg_memory`:** 0.11 s per call at 10.3M DOF (voxel $150^3$), with or without `s_K`. Closed with the number.
- **`_amg_scan` / `_shell_scan`:** `lru_cache(maxsize=1)` keeps the last mesh and its scan alive after the caller drops it.
- **Voxel part finder:** `ndimage.label` on the cell mask (face connectivity), mapped through `element_map`, plus the existing node-label contact check: 0.025 s and 93 B per element of transient host memory, against 0.72 s and 352 B for the generic search, on $100^3$ with a gap layer (2 parts, the same partition): 29× faster, 3.8× leaner (`docs/boundary_conditions.md:53` estimated 20× and 8×).
- **$P_0^\top$ as a scatter over $P_0$'s ELL:** 3–9× slower than the stored transpose on CPU (18c probe). Closed, not adopted.
- **1.25× ELL width classes:** the current classes pad only 6–10% of the ELL bytes. Closed, not adopted.
- **Size-dependent `STORAGE32`:** the per-family ratio (hex8 0.636, p 0.548, shell 0.574) falls with size (0.44 at hex20 $44^3$, 0.59–0.61 hex8, 0.50 flat $362^2$ / block $64^3$): at 1M DOF the fp32 storage is 58–86 MB high, about 7–12 basis columns, under 2% of the reach, and conservative. Closed, not refitted.
- **fp64 hex polish:** exceeds the fp64 low model by up to 0.33 blocks (hex20 $52^3$, m = 120; `amg_fp32_study.json` `polish64_slack`). On the 11 recorded `polish64` rows the fp32 polish model (`AMG_STEP32[(fam, "polish")]` with its $f_l\,ws$) over the measured fp64 polish is 0.836–1.054 (hex8 $80^3$ m = 10 lowest, hex27 $40^3$ m = 10 highest); the factor on the term that bounds each row is 0.95–1.20, above 1.05 at hex8 $80^3$, m = 10 and 25 (1.20, 1.052), both bounded by the fp64 low model (1.046, 1.049). The fp64 low model bounds 10 of the 11 rows (all but hex20 $52^3$, m = 120: 0.985, where the 9-block model gives 1.002).
- **fp64 hex step at m = 120:** 3.2% above the model's extra at hex8 $40^3$ (about 1% of the step; `docs/open_items.md:283`). Kept and logged (within the ±20% band; no fit data beyond the 18c grid).
- **`assembled._chunked`:** the loop gathers, dispatches and copies one chunk at a time. Pipelined, 8% in `element_stress` on CPU (PR 17); a reviewer probe (host gather + jitted kernel + copy, 400k elements in 50k chunks) gave 17%.

## Design (18d)

### `morton(mesh)` (`mesh/reorder.py`)

```python
def morton(mesh):
    """Morton (Z-order) node order, then elements by the Morton key of their centroid within
    each block. Returns (new mesh, p, q) like rcm."""
```

- Keys: per axis $q = \lfloor (x - x_\min) / \max(x_\max - x_\min, 10^{-300}) \cdot (2^{21} - 1) \rfloor$ on the nodes' bounding box, bits interleaved (x lowest) into a uint64 (63 bits); `np.argsort(kind="stable")`. Elements: per block, `s + argsort(keys(centroids))` on the same box.
- Exported from `jaxmodal.mesh` next to `rcm`. Not a default anywhere (Findings).
- **No automatic reordering and no warning** in `solve_modes` (decision, logged): reordering would permute the mode vectors back and forth, and a span threshold would need calibration across element types. `docs/mesh.md` gets the measured cost of a shuffled mesh and the advice to call `rcm` on meshes read from files.

### Per-mesh caches (`memory.py`)

```python
_CACHE = weakref.WeakKeyDictionary()  # mesh -> {key: value}; entries die with the mesh

def _per_mesh(f):
    @functools.wraps(f)
    def g(mesh, *args):
        d = _CACHE.setdefault(mesh, {})
        key = (f.__name__, *args)
        if key not in d:
            d[key] = f(mesh, *args)
        return d[key]
    return g
```

- Replaces `lru_cache(maxsize=1)` on `_shell_scan` and `_amg_scan` (`Mesh` is `frozen=True, eq=False` without slots: identity hash, weak references work).
- New `_layout_sizes(mesh, path, chunk, range_nodes)` under `_per_mesh`: calls `layout` once and keeps the scalars `unstructured_memory` reads (Findings). `path` is resolved before the call (`DEFAULT_PATH` by backend); `chunk` is not normalized (an explicit default chunk makes a second, equal entry: scalars only). `unstructured_memory` reads it instead of `layout`; its values are unchanged.
- Cached values are NumPy arrays and ints; none references the mesh.
- `lru_cache` callers updated: `tests/test_strategy.py::test_amg_scan_cached` (`cache_clear`, `cache_info`); `benchmarks/strategy_study.py:254`'s `mem._amg_scan.cache_clear()` is removed (a fresh mesh starts a fresh entry).

### `_default_budget` without `bytes_limit` (`memory.py`)

`memory_stats()` can raise (some plugins), return `None`, or lack `bytes_limit` (Metal). In those cases the budget is the CPU rule (`ram, None`: a unified pool). Unchanged where the key exists.

### `assembled._chunked` pipelined (`operators/assembled.py`)

Per chunk: build chunk $i+1$'s host inputs (`X[conn[e]]`, the parameter slices), dispatch its kernel, then copy chunk $i$'s result into the host arrays (asynchronous dispatch runs kernel $i+1$ during the gather and the copy; the copy blocks only on kernel $i$). Two chunks (inputs and results) are alive at once; `docs/postprocessing.md:212` ("+ one chunk") and `assemble`'s memory text (`docs/operators.md:36–44`) say two. The output arrays and the padding rule are unchanged.

### Voxel `parts` (`mesh/parts.py`)

If `mesh.voxel is not None` (voxel meshes are hex8 only, `Mesh._check_voxel`; `parts.py` imports `from scipy import ndimage`, which the test spy patches): `lab, n = ndimage.label(element_map.reshape(nz, ny, nx) >= 0)` (the default cross structure is face connectivity); element labels `lab - 1` scattered through `element_map` (label 0, inactive cells, dropped), then renumbered by each component's lowest element index (`np.minimum.at` and a rank over $n$), which is `connected_components`' order: the labels equal the generic path's, so `rigid_modes`' column order (`src/jaxmodal/operators/constraints.py:127–139`) does not depend on the path, also after `rcm` / `renumber`. The node scatter and the contact check (edge and vertex contacts raise) are shared with the generic path. Valid after `renumber` (the voxel maps are renumbered with the mesh). No collapsed-element check (voxel cells cannot collapse).

### fp64 hex polish term (`memory.py`, `eigen/api.py`)

- `POLISH64 = 1.05`: for fp64 hex AMG (`hex8` and `p`), `amg_memory` returns `polish` $= 1.05\,(8n(a\,m + c) + 8m\,K)$ from the fp32 polish constants (`AMG_STEP32[(fam, "polish")]`; the whole fp32 term scaled) plus the unscaled loop term ($8m \cdot$ `AMG_LOOP_K` at degree > 2, the fp64 bound already), and `lobpcg_memory`'s step is $\max(\text{9-block}, \text{low}, f_l\,ws + \text{polish})$ as for the other families; `polish_options={"on": False}` drops it. Today's $\max(\text{9-block}, \text{low})$ already bounds all 11 recorded rows (the 9-block model 1.002 at hex20 $52^3$, m = 120); only the low model alone misses there (0.985), so the term matters for `low_memory=True`. The polish term alone does not bound the fp64 polish on hex8 at small m (hex8 $80^3$ m = 10: 0.877; hex8 $40^3$ m = 8: 0.959), where the low model does. On the 11 rows $\max(\text{low}, \text{polish})$ is 1.042–1.146 of the measured fp64 polish; on those rows (m ≤ 120) it raises the default step over today's by at most 4.5% (hex20 $52^3$, m = 120; 1.4% at m = 40; none at m ≤ 25) and the `low_memory` step by up to 6.3% (hex20 $52^3$, m = 120; 4.3% at m = 40; 2.5–2.7% at m = 25). Production blocks are wider (200 modes: k = 240, an extrapolation of the term as for fp32): box hex20 $36^3$, 200 modes, `solve_modes_memory`'s step goes from 24.96 to 27.01 GB (+8.2%) and the total from 28.20 to 30.25 GB (+7.3%); the `low_memory` step from 24.56 to 27.01 GB (+10%), so there `low_memory=True` no longer lowers the total (30.25 GB both ways; before 27.79 against 28.20), and fp64 now models above fp32 (28.80 GB); box hex8 $32^3$, m = 60: +5.3% (polish 1.600 GB over low 1.519 GB). The fp32 study is not rerun; if `test_amg_step` or the TIGHT point fails, the fallback is the full `polish64` grid refit (logged).
- API effects: the lever list (`eigen/api.py:499–503`) offers `polish_options={"on": False}` for fp64 hex AMG too; a band `c_max` from `solve_modes_memory` (`api.py:523`) can fall for fp64 hex AMG, so a band strategy or chunk can change. Docstrings and comments that say "shell AMG, fp32 hex AMG" (`memory.py:154`, `:173`, `:432`, `api.py:499`) are updated; recorded strategy, `c_max` and lever expectations in `tests/test_strategy.py` and the band tests are re-checked.

### Benchmarks

- **`benchmarks/kernels.py` → `benchmarks/results/kernels.json`**, parts `roofline`, `fusion`, `orders`, each in fresh subprocesses (`harness.measure`):
  - `roofline` (about 5 min): the triad and fp64 peak from `harness.system_info` (rerun in the same invocation); the voxel stencil ($87^3$) and stored-Ke `unstructured` (hex8 $48^3$, hex20 $32^3$) apply_K at m = 1, 4, 16, min of 5 after two warm calls (first touch excluded). Per kernel: analytic flops, lower-bound bytes and intensity, implementation bytes and intensity (formulas above), the roofline $\min(B \cdot I, F)$ on lower-bound bytes, achieved GFLOP/s and its percentage of that roofline (the headline, and the basis of the GPU ≥ 50% check); the implementation-bytes percentage is a secondary column labelled as such. A null chunk-512 result is recorded as "not determined" and drives nothing. No `cost_analysis` (loops).
  - `fusion` (about 2 min): for the stencil apply_K, the stored-Ke apply_K (scatter) and the hex20 AMG V-cycle: the compiled HLO text, walked over every computation reachable from ENTRY (fusions, while / scan / conditional bodies): per computation the fusion count, the non-fused ops by opcode and the largest materialized buffers; `memory_analysis` temp bytes against the vector bytes. Report only; more targets only if one of these shows large unfused ops.
  - `orders` (about 15 min): apply_K (scatter) at m = 1, 16 and the AMG V-cycle at m = 16 on box hex20 $44^3$ and the `ssh8` $256^2$ plate in grid, RCM, Morton and shuffled order; gather on the box at m = 1 only (the 11×), after `unstructured_memory` confirms it fits.
- **`benchmarks/compile_study.py` → `benchmarks/results/compile_study.json`** (about 20 min): full `solve_modes` (20 modes) with GMG (voxel $20^3$, $40^3$, $80^3$) and AMG (box hex20 $8^3$, $24^3$; the `ssh8` plate $64^2$, $256^2$), each in a fresh subprocess with the persistent cache disabled. A `solve_modes` call compiles many jits, so the compile time is the sum of the `/jax/core/compile/backend_compile_duration` events (`jax.monitoring.register_event_duration_secs_listener`), with the summed `jaxpr_to_mlir_module_duration` (lowering) and `jaxpr_trace_duration`, the number of compiles and the wall time; the run asserts that compile events arrived (no silent cache hit). Next to each, the structure counts that drive it (GMG / AMG levels, ELL classes, slab count).

### Bimodal LOBPCG peak (time-boxed, about 45 min of runs, each invocation ≤ 1 h)

The 791k plate cold `solve_modes` (`docs/open_items.md:293`), fresh subprocesses. The peak as today: the child resets the footprint interval after setup and reads the interval maximum and `ru_maxrss` at the end (`harness.py:41–97`). New, in the probe script only: the parent samples the child's `phys_footprint` every 100 ms (`proc_pid_rusage(child_pid)`, no thread in the child), and the child logs the summed `jax.live_arrays()` bytes with start and end timestamps of every LOBPCG iteration (the 20-mode solve is one chunk, so chunk boundaries give two points): a probe-only wrapper rebinds `lobpcg_step` and `lobpcg_step_low` in `sys.modules["jaxmodal.eigen.lobpcg"]` (`jaxmodal.eigen.lobpcg` is the function: `eigen/__init__` shadows the submodule); the loop already syncs each step (`np.asarray(out)`), so the wrapper adds no sync. Reading: extra bytes in the live bytes mean held buffers; a footprint high between steps without them means allocator retention; spikes only inside steps mean executable temporaries. Wall time is recorded per run. The runs use fp32 AMG (the dtype of `test_shell_791k_peak32`, whose gate covers both regimes); fp64 only if a setting is adopted. Each setting gets 5 runs without the wrapper, interleaved with 5 unmodified baseline runs in the same session (so a one-regime result is not session drift); a separate group of 3 wrapper-only runs gives the live-bytes curve, each classified by its own peak (if the wrapper alone always lands in the low regime, its curve is read as low-regime only):
1. XLA CPU intra-op threads (`--xla_cpu_multi_thread_eigen=false`; both flags are accepted by jaxlib 0.11.2); 3 runs if it is more than 2× slower; no wall-time change means the flag had no effect, not that threading is excluded.
2. `--xla_cpu_enable_concurrency_optimized_scheduler`.
3. Only if the curve reads as allocator retention: `MallocNanoZone=0` and `MallocLargeCache` unset (the harness sets it to 0), 3 runs each, inside the box. Otherwise the allocator hypothesis is diagnosed by the curve, not probed.

The probe script lives in `benchmarks/bimodal_probe.py` and is run, not tested.

18c's in-process polling thread (low regime every time) is not rerun: it perturbs the process and is never adopted.

Outcome: a setting is adopted only if it puts every run in the estimators' regime and applies to production runs (set by `solve_modes` or documented as the user default), then in the peak tests too. Otherwise the findings (timeline, which settings move the regime) are logged, 18c's gates stay, and the roadmap's 18d bullet is reworded (cause not found; deferred to the GPU check, where the allocator differs). The estimators do not change.

## Tests (18d)

- `tests/test_reorder.py`:
  - `morton` on a $4 \times 3 \times 2$ hex8 box with spacing (1, 2, 0.5) (the key helper `reorder._morton_keys(x, lo, hi)` also on coplanar points, for the zero-extent guard): the node order and each block's element order equal an independent reference that quantizes with the formula above and interleaves the integers with a per-bit Python loop; `check_renumbered(old, new, p, q)` on `morton`'s output for `with_sets(mixed_mesh)` and a voxel mesh (sets, material, orientation, voxel maps), as for `rcm`.
  - Equivalence (roadmap): after `morton`, apply_K and apply_M of `unstructured` (box hex8, hex20, the mixed hex8 + hex20 mesh, an `ssh8` + hex8 plate) and of `stencil` (a masked non-cubic voxel mesh) equal the permuted original to $10^{-12}$ relative, m = 3.
- `tests/test_rigid.py` (parts): the voxel path's labels equal the generic path's exactly on the existing cases and a non-cubic $6 \times 3 \times 2$ mask with a gap and an L-shaped part, also after `rcm` and a random `renumber`; edge and vertex contacts raise on the voxel path; a spy on `parts.ndimage.label` confirms the voxel cases take the path and the non-voxel ones do not.
- `tests/test_operators.py` / `test_post.py`: `_chunked` for $n$ divisible by the chunk, not divisible (padded last chunk), $n$ < chunk, and a pytree output (`element_stress`), against one chunk with the existing tolerances (different chunk shapes compile different kernels); existing cases checked first and only missing ones added.
- Memory (`tests/test_strategy.py`):
  - `test_amg_scan_cached` rewritten on a fresh, uncached mesh (`prob("short")` is shared and other tests add `_layout_sizes` keys): the same object on a second call and one `_amg_scan` key in `mem._CACHE[mesh]`.
  - Lifetime: a fresh mesh (no cached helper); `_amg_scan`, `_shell_scan` (shell mesh) and `_layout_sizes` (scatter and gather) called; `mesh in _CACHE`; `r = weakref.ref(mesh)`; `del mesh; gc.collect()`: `r() is None`, and a second mesh created in the test is still a member (no global count: `gc.collect()` may free other tests' meshes).
  - `_layout_sizes` against a fresh `layout` on gather and scatter.
  - The existing `test_default_budget` (`test_strategy.py:1731`, beside `fake_gpu`) extended with `memory_stats` raising, returning `None`, a dict without `bytes_limit` (all `(ram, None)`), and a full dict.
- `tests/test_amg.py`:
  - `test_amg_step` keeps its step assertions polish-free (`step_model(..., polish=False)`, so the max does not hide the 9-block and low terms) and adds, for fp64, `SSS.polish(op, free, P, k)` (the measured fp64 polish `memory_analysis`) ≤ $f_l\,ws$ + `amg_memory(...)["polish"]` on its existing parametrisation (1.50–2.14× in review: loose at these sizes, so TIGHT is the tight check); docstring updated.
  - The TIGHT point (hex20 $24^3$, m = 25, `slow`): $1 \le (f_l\,ws + \text{polish}) / \text{measured} \le 1.3$ for fp64 (1.280 in review, a thin margin; the recorded value goes in the docstring), as `test_amg_step_tight32` does for fp32. With `test_amg_step` this is the independent validation.
  - A regression pin: $\max(\text{low}, \text{polish})$ ≥ the 11 recorded `polish64` rows of `amg_fp32_study.json` (the data `POLISH64` was chosen from).
  - `polish_options={"on": False}` drops the fp64 term.
  - Docstrings say which tests are the independent checks (`test_amg_step`, TIGHT) and which is the data pin.
- `tests/test_shell_solve.py::test_polish_lever_hex` (`:583–595`): its budget `0.8 * solve_modes_memory(...)["total"]` (fp64) grows with the term, so the hex20 $6^3$ band solve no longer raises (both dtypes fail at `:590`); the budget is re-derived from each dtype's own total with a smaller factor that raises, the lever is asserted for both dtypes, the docstring ("the fp64 hex polish stays below the step") rewritten.
- `tests/test_strategy.py::test_ladder_real` (`:424–432`): its budget from `chunk_need` grows with the term, so `_ladder` picks chunk 20 (C_max 2048 against 1980 at chunk 10) and `< 20` fails; the budget is re-picked so chunk 20 does not fit, and the docstring's recorded C_max (960 / 1320 / 1360) re-measured.
- Before review the whole suite (`scripts/test.sh`) runs with the term; any further broken expectation is fixed in the same commit and listed in `docs/open_items.md` (a review-round simulation found exactly the four listed here plus `test_amg_scan_cached`).
- `tests/test_shell_solve.py::test_hex_polish_below_step` (`:536–548`, slow): it asserts `"polish" not in pc` for fp64 hex; rewritten (and renamed `test_hex_polish_bounded`) to $\max(\text{low}, f_l\,ws + \text{polish})$ ≥ the measured fp64 polish on its cases (hex20 $14^3$, hex8 $24^3$, $40^3$), docstring updated.
- `tests/test_shell_solve.py::test_lobpcg_memory_max` (`:607–634`): it asserts `"polish" not in hp` and `hstep == hterms["low"]` for fp64 hex8 AMG; rewritten to the max of the three terms (on box $32^3$, m = 60, the polish term wins: 1.600 against 1.519 GB low) and the `low_memory` step equal to max(low, polish) for fp64 hex; the `"polish" not in gmg_memory(...)` half kept; the docstring phrases "no polish term with ... fp64 hex AMG" and "on hex8 AMG at 100k and m = 60 the low term wins" rewritten.
- `tests/test_performance.py` (new, fast, no wall-time assertion):
  - `test_roofline_model`: the flop and byte formulas by hand ($87^3$: 0.759 GFLOP, 46.0 MB; hex8: 1152 flops and 2400 B per element; hex20: 7200 flops and 14640 B of packed Ke + 80 B `conn`); and analytic ≤ `cost_analysis()["flops"]` ≤ bound × analytic on one-trip programs: the stencil on a cubic $12^3$ voxel box with `slab` = the full length and one material (bound 1.5; probes 1.03–1.29 at m = 1, the measured value goes in the docstring; small non-cubic grids reach 2.4× from the padded accumulator), `unstructured` hex8 and hex20 with an explicit `chunk=n_el` (not the default, whose padding gives 1.244 on hex8; bound 1.25; probe 1.03–1.17). One trip is asserted by no `while(` opcode in the compiled HLO (`re.search(r" while\(", text)`; the substring "while" stays in op metadata), with a multi-trip control (stencil `slab=4`, `chunk=32`) where the opcode is present.
  - `test_fusion_report` (imports the walker from `benchmarks/kernels.py`): on a tiny two-chunk stored-Ke apply the walker finds the loop body and its fusions (not only ENTRY's).
  - `test_compile_listener`: a subprocess with `jax.config.update("jax_enable_compilation_cache", False)` (the pattern of `tests/test_krylov.py:228`) and the listener style of `tests/test_config.py:55`; a tiny jit yields ≥ 1 backend-compile event and > 0 s.
  - `test_results_pinned`: `kernels.json` and `compile_study.json` have the keys and cases the docs tables read, and the recorded roofline and percentages recompute from their recorded fields.
- If the bimodal outcome adopts a setting: `test_harness.py` checks the peak tests' subprocess environment carries it.

## Docs (18d)

- `docs/performance.md` (new): the roofline (definition, the byte counts, the implementation bytes, the table from `kernels.json`, the full-Ke GPU decision, the `cost_analysis` loop caveat), the fusion report, the compile times with their structure counts, the ordering costs, the bimodal outcome, the deferred GPU checks (pointing to `docs/ROADMAP.md`).
- `docs/conventions.md:89` and `docs/ROADMAP.md:207`: compile timing of one jit by `jit(f).lower(...).compile()`, and of a multi-jit call by the summed backend-compile events; both in a fresh subprocess with the cache disabled.
- `docs/mesh.md`: `morton` in the helper list (`:3`) and a section (`:38` names "PR 18d Morton"); the ordering advice and the shuffled-mesh costs next to the RCM cost (`:48`).
- `docs/operators.md`: `:206`, the layout cache replaces "caching `layout` is PR 18d" (0.22 s at 812k, 3.2 s at 10.2M, once per mesh); `:36–44`, `assemble`'s two chunks; `:244` and `:253`, the cost per apply points to `docs/performance.md` (lower bound vs implementation bytes); Morton in the equivalence test list (`:287ff`).
- `docs/boundary_conditions.md`: `:53`, the closing sentence ("≈ 20× faster and 8× leaner; it is left to PR 18d") replaced by the voxel path and its measured cost (0.025 s and 93 B per element at $100^3$: 29× faster, 3.8× leaner); `:90`, the parts tests.
- `docs/eigensolvers.md`: `:167`, a device without `bytes_limit` uses the CPU rule; `:154`, the polish lever for every AMG family; `:414–417`, the polish term exists for every AMG family and `POLISH64`, and the winner sentence and the `low_memory` remark rewritten for fp64 hex (the polish beats low at wider m, and beats the 9-block term at k = 240 on hex20 $36^3$); `:487`, the bimodal outcome; `:901–908` and `:983`, the test summaries (polish, cache and lifetime tests).
- `docs/preconditioners.md`: a new `polish` bullet after `:595` (fp64 hex, `POLISH64`); `:597`, the weak per-mesh cache (replaces "`lru_cache(maxsize=1)`" and "the gather path still runs `layout` per call"); `:611`, the "exceeds the fp64 low model by up to 0.33 blocks" sentence pointed to the new bullet; `:612`, the test list; `:627`, the bimodal outcome; `:672`, the test summary; `:107` (the width-bound paragraph, which discusses the $P_0^\top$ apply), the 1.25× width classes and the $P_0^\top$ scatter closed.
- `docs/postprocessing.md:212`: two chunks (inputs and results).
- `docs/verification.md:303`: the new `_chunked` cases in the `test_post.py` summary; `docs/eigensolvers.md:979` ("Fit test and ladder"): the extended `test_default_budget`.
- `docs/open_items.md`, line to outcome:

  | Line | Item | Outcome |
  |---|---|---|
  | 38, 42 | GMG intermittent peak (11b), 300k pencil peak | a pointer to the 18d bimodal section either way (covered if the cause is shared) |
  | 165, 182 | layout cache, scan cache, `gmg_memory` | per-mesh weak cache; `gmg_memory` closed (0.11 s at 10M) |
  | 183 | Metal / budget; the operator build's XLA kernel temporaries | CPU rule without `bytes_limit`; the temporaries stay a deferred GPU check (ROADMAP `:801`) |
  | 234 | `_chunked` | pipelined |
  | 241 | $P_0^\top$ scatter, 1.25× classes | closed with numbers (3–9× slower; 6–10% padding); $P_0 = S\,T$ matrix-free unchanged |
  | 245 | Schwarz `chunk_bytes` "public option in 18d if needed" | deferred to the GPU check (PR 18a's) |
  | 266 | "polish term for shell and fp32 hex" | every AMG family (`POLISH64`) |
  | 268, 293 | 791k solve peak, bimodal | the investigation's outcome |
  | 280 | size-dependent `STORAGE32` | closed with numbers |
  | 283 | fp64 hex step at m = 120 (3.2%) | kept, logged |
  | 283, 284 | fp64 hex polish; fp32 above fp64 on hex | the term added; 284 replaced by the recomputed `solve_modes_memory` pair for box hex20 $36^3$, 200 modes (fp32 28.8 GB against the new fp64 value), its levers sentence updated (`low_memory` no longer lowers the fp64 total there) |

  plus a PR 18d section with the decisions below.
- `docs/ROADMAP.md`: the 18d status; `:794` (PR 18a's Schwarz scan check) reworded: the remedy is deferred with the check (not needed on CPU); `:800` ("PR 18d: matvec ≥ 50% of roofline at m = 1") reworded to both kernels against the measured fp64 matmul peak; `:801` (XLA kernel temporaries) kept; PR 18b's 791k async-peak entry (about `:798`) extended with the bimodal reading if no cause is found; added: the full-Ke / packed matvec decision for stored Ke, Morton against RCM on the GPU scatter path, compile times on GPU; the 18d entry (`:659` "Morton reordering and reordering tuning" → Morton ordering as a helper, measured against RCM and grid order, no default change; `:663` bimodal bullet, "make the peak tests deterministic", and the Tests bullets near `:666`) reworded to what was done.
- `README.md`: the status line (PR 18d); a link to `docs/performance.md` next to the operators / eigensolvers links; `morton` next to the `docs/mesh.md` link (`:37`).

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **18d stays one PR:** the items are small and independent.
- **No automatic reordering and no warning;** `morton` is a helper only.
- **Closed without code:** `gmg_memory` caching, the $P_0^\top$ scatter, the 1.25× classes, the size-dependent `STORAGE32`, the fp64 hex step at m = 120 (numbers above).
- **Stored Ke stays packed on CPU;** full-Ke storage or a packed matvec is a GPU decision (the unpack doubles the m = 1 traffic; full Ke costs 1.9× the storage).
- **Deferred to GPU:** the Schwarz scan remedy, the XLA temporaries on GPU, the roofline ≥ 50%.
- **The roofline test checks the model, not the speed:** the timed percentages are a benchmark (a timed bound at 8–16% could never fail).
- **fp64 hex polish as a factor on the fp32 term** (1.05, from 11 recorded rows) instead of a refit; the refit is the fallback. It raises the default fp64 hex step by at most 4.5% and the `low_memory` step by up to 6.3% on the recorded rows (m ≤ 120), and by 8.2% / 10% (total +7.3%) at 200 modes (k = 240) on box hex20 $36^3$, so a band chunk chosen from the estimate can shrink by up to about that much; eigenpairs are unchanged.
- **The bimodal investigation is time-boxed;** a setting is adopted only if it holds for production runs; otherwise 18c's gates stay.
- **Cache lifetime:** results live as long as the mesh (scan arrays of a few bytes per DOF; the layout entry is scalars).
