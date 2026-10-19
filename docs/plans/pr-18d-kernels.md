# PR 18 plan: performance pass, part d: kernels

Branch `pr-18d-kernels` from `main` (1c6b651, PR 18c merged). Roadmap: "PR 18d: Kernels".

## Revisions

- Rev 1: first draft.
- Rev 2 (plan review round 1: correctness, tests, docs not satisfied; simplicity, performance satisfied):
  - Morton reference test on the quantized keys, plus `check_renumbered`; `bits` dropped.
  - `lru_cache` callers (`test_amg_scan_cached`, `strategy_study.py:254`) listed; lifetime test by `weakref.ref`.
  - fp64 polish: the API effects (lever, band `c_max`), the docstrings, the `polish64` plumbing in the fit and the existing tests that change are listed; "no change to solver results" narrowed.
  - Voxel `parts`: `(nz, ny, nx)` reshape, `label - 1`, no dead guard, a non-cubic test case; host peak measured (93 against 352 B per element).
  - The roofline test is now a fast formula check against `cost_analysis` (the timed test could not fail); an implementation-bytes column; the GPU peak defined as the measured fp64 matmul.
  - Smaller benchmarks (fusion on 3 targets, compile study without hex20 $16^3$, gather at one point); structure counts next to compile times.
  - Bimodal probes: parent-side sampling, the polling thread as the first candidate, time estimate; the roadmap rewording if no cause is found.
  - `_default_budget` also catches a raising `memory_stats()`.
  - `_chunked`: the order spelled out (inputs, dispatch, copy); two chunk results alive.
  - Docs: the compile-time rule in `conventions.md` and the roadmap, every affected location named, an open-items outcome table.

## Goal (18d)

Measure the kernels against the machine (roofline, fusion, compile time), add Morton ordering as a helper, and close the non-band follow-ups that PRs 15d–18c deferred here: per-mesh caches, a pipelined chunk loop, the Metal budget fallback, the voxel part finder, an fp64 hex polish term, the bimodal peak investigation. Eigenpairs do not change; the fp64 hex AMG memory estimate grows by the polish term, which can change a band strategy or chunk chosen from it (below).

## Findings (prototypes, CPU fp64, M3 Max, `benchmarks/results/system.json`: triad 185 GB/s, fp64 matmul 490 GFLOP/s)

### Ordering (apply_K, scatter path, m = 1 and 16, and the AMG V-cycle)

- **Morton vs RCM vs grid:** on box hex20 $44^3$ (812k DOF) Morton is not faster than RCM or grid order on the scatter path (within ±3%) and makes the gather path about 11× slower (Morton's jumps between octants widen the gather windows). The median element node-index span: grid 8146, RCM 12399, Morton 326 (local in space, not in the index range a window covers).
- **Shuffled meshes** (random node and per-block element permutations; the worst case for a Gmsh or Abaqus file) cost +30–100% on apply_K and +50% on the AMG V-cycle. RCM recovers both (0.66 s at 358k nodes; about 3 s at 10M DOF, `docs/mesh.md`).
- **Grid order on a plate** (`ssh8` $256^2$, 396k DOF): RCM leaves apply_K (23.7 / 23.1 ms at m = 16), the AMG build (5.48 / 5.69 s) and the V-cycle (382 / 381 ms) unchanged, although the median node span is 16771 in grid order against 367 after RCM.

### Roofline (first numbers from recorded results)

- Voxel stencil, $87^3$, m = 1 (`stencil_slab.json`): 18.4 ms per apply_K; 1152 flops per cell, 0.759 GFLOP, so 41 GFLOP/s. Lower-bound bytes ≈ 16 B per cell (`node`/`cell` ids, $s$) + 52 B per node ($u$ read, output written, `node` id) ≈ 46 MB: intensity ≈ 17 flop/B, above 490/185, so the roofline is the fp64 peak: about 8%. The implementation's traffic is the corner stack and shifted adds through U and Y (about $4 \cdot 24 \cdot 8\,m$ B per cell, `docs/operators.md`, "Why this layout").
- Stored-Ke apply (hex8, packed upper triangle, 300 doubles per element): 1152 flops over 2400 B, 0.48 flop/B, so bandwidth-capped at about 89 GFLOP/s (hex20 0.49 flop/B). It runs at 11–16% of the triad in Ke traffic (PR 5); the unpack `Ke[:, _packed_index(k)]` writes and reads another 4608 B per hex8 element.

### Follow-ups

- **`unstructured_memory` (gather):** `layout` costs 0.22 s at 812k DOF (half windows, half inverse map) and 3.2 s at 10.2M per call, about 35 s per band call there. The estimator reads only scalars (per block `type`, `n_pad`, `chunk`, `k`, `W`; `path`, `n_nodes`, `R`, `n_ranges`, `valence`, the inverse-map size).
- **`gmg_memory`:** 0.11 s per call at 10.3M DOF (voxel $150^3$), with or without `s_K`. Closed with the number.
- **`_amg_scan` / `_shell_scan`:** `lru_cache(maxsize=1)` keeps the last mesh and its scan alive after the caller drops it.
- **Voxel part finder:** `ndimage.label` on the cell mask (face connectivity), mapped through `element_map`, plus the existing node-label contact check: 0.025 s and 93 B per element of transient host memory, against 0.72 s and 352 B for the generic search, on $100^3$ with a gap layer (2 parts, the same partition): 29× faster, 3.8× leaner (`docs/boundary_conditions.md:53`'s estimate said 20× and 8×).
- **$P_0^\top$ as a scatter over $P_0$'s ELL:** 3–9× slower than the stored transpose on CPU (18c probe). Closed, not adopted.
- **1.25× ELL width classes:** the current classes pad only 6–10% of the ELL bytes. Closed, not adopted.
- **Size-dependent `STORAGE32`:** the per-family ratio (hex8 0.636, p 0.548, shell 0.574) falls with size (0.44 at hex20 $44^3$, 0.59–0.61 hex8, 0.50 flat $362^2$ / block $64^3$): at 1M DOF the fp32 storage is 58–86 MB high, about 7–12 basis columns, under 2% of the reach, and conservative. Closed, not refitted.
- **fp64 hex polish:** exceeds the fp64 low model by up to 0.33 blocks (hex20 $52^3$, m = 120; `amg_fp32_study.json` `polish64_slack`). The model must bound it: add the term.
- **`assembled._chunked`:** the loop gathers, dispatches and copies one chunk at a time. Pipelined, 8% in `element_stress` on CPU (PR 17); a reviewer probe (host gather + jitted kernel + copy, 400k elements in 50k chunks) gave 17%.

## Design (18d)

### `morton(mesh)` (`mesh/reorder.py`)

```python
def morton(mesh):
    """Morton (Z-order) node order, then elements by the Morton key of their centroid within
    each block. Returns (new mesh, p, q) like rcm."""
```

- Keys: per axis $q = \lfloor (x - x_\min) / \max(x_\max - x_\min, 10^{-300}) \cdot (2^{21} - 1) \rfloor$ on the mesh's bounding box, bits interleaved (x lowest) into a uint64 (63 bits); `np.argsort(kind="stable")`. Elements: per block, `s + argsort(keys(centroids))` (the same box as the nodes).
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
- New `_layout_sizes(mesh, path, chunk, range_nodes)` under `_per_mesh`: calls `layout` once and keeps the scalars `unstructured_memory` reads (Findings). `path` is resolved before the call (`DEFAULT_PATH` by backend), so the key is stable. `unstructured_memory` reads it instead of `layout`; its values are unchanged.
- Cached values are NumPy arrays and ints; none references the mesh.
- `lru_cache` callers updated: `tests/test_strategy.py::test_amg_scan_cached` (`cache_clear`, `cache_info`) becomes an identity and call-count check; `benchmarks/strategy_study.py:254`'s `mem._amg_scan.cache_clear()` is removed (a fresh mesh starts a fresh entry).

### `_default_budget` without `bytes_limit` (`memory.py`)

`memory_stats()` can raise (some plugins), return `None`, or lack `bytes_limit` (Metal). In those cases the budget is the CPU rule (`ram, None`: a unified pool). Unchanged where the key exists.

### `assembled._chunked` pipelined (`operators/assembled.py`)

Per chunk: build chunk $i+1$'s host inputs (`X[conn[e]]`, the parameter slices), dispatch its kernel, then copy chunk $i$'s result into the host arrays (asynchronous dispatch runs kernel $i+1$ during the gather and the copy). Two chunk results are alive at once; the chunk defaults (`element_stress`, `assemble`) are a few tens of MB per chunk, and `docs/postprocessing.md`'s memory line is checked and updated. The output arrays and the padding rule are unchanged.

### Voxel `parts` (`mesh/parts.py`)

If `mesh.voxel is not None` (voxel meshes are hex8 only, `Mesh._check_voxel`): `lab, _ = ndimage.label(element_map.reshape(nz, ny, nx) >= 0)` (the default cross structure is face connectivity), element labels `lab - 1` scattered through `element_map` (label 0, inactive cells, dropped). The node scatter and the contact check (edge and vertex contacts raise) are shared with the generic path. Labels are `0..n-1` (the only consumer, `rigid_modes`, uses `max()` and `bincount`); their order can differ from the generic path's, so tests compare partitions. Valid after `renumber` (the voxel maps are renumbered with the mesh). No collapsed-element check (voxel cells cannot collapse).

### fp64 hex polish term (`memory.py`, `eigen/api.py`, `benchmarks/amg_fp32_study.py`)

- `steps_hex` records `polish64` on the whole `HEX` grid (one fp64 AMG build per case next to the fp32 one, the polish compiled at each width). `targets`, `asymptotes` and the TIGHT gate take a key for the polish series, so `fit` fits `AMG_STEP[(fam, "polish")]` for `hex8` and `p` from `polish64` with the same LP (upper envelope, rounded up, gate 1.3); the fp32 constants are refitted from the same data and asserted unchanged. Before the full run, two cases are timed; if the projection exceeds 1 h the run is split by family (one invocation each).
- `amg_memory` (fp64, hex) returns `polish` like the fp32 and shell families; `lobpcg_memory`'s step is $\max(\text{9-block}, \text{low}, f_l\,ws + \text{polish})$ as for them. `polish_options={"on": False}` drops it.
- API effects: the lever list (`eigen/api.py:499–503`) now also offers `polish_options={"on": False}` for fp64 hex AMG; a band `c_max` from `solve_modes_memory` (`api.py:523`) can fall for fp64 hex AMG, so a band strategy or chunk can change. Docstrings and comments that say "shell AMG, fp32 hex AMG" (`memory.py:154`, `:173`, `:432`, `api.py:499`) are updated; recorded strategy, `c_max` and lever expectations in `tests/test_strategy.py` and the band tests are re-checked.

### Benchmarks

- **`benchmarks/kernels.py` → `benchmarks/results/kernels.json`**, parts `roofline`, `fusion`, `orders`, each in fresh subprocesses (`harness.measure`):
  - `roofline` (about 5 min): the triad and fp64 peak from `harness.system_info` (rerun in the same invocation); the voxel stencil ($87^3$) and stored-Ke `unstructured` (hex8 $48^3$, hex20 $32^3$) apply_K at m = 1, 4, 16, min of 5 after a warm call. Per kernel: flops, lower-bound bytes and intensity, implementation bytes and intensity (formula), the roofline $\min(B \cdot I, F)$, achieved GFLOP/s and its percentage.
  - `fusion` (about 2 min): for the stencil apply_K, the stored-Ke apply_K (scatter) and the hex20 AMG V-cycle: the compiled HLO's fusion count, the non-fused ops of the entry computation by opcode, and `memory_analysis` temp bytes against the vector bytes. Report only; more targets only if one of these shows unfused large ops.
  - `orders` (about 15 min): apply_K (scatter) at m = 1, 16 and the AMG V-cycle at m = 16 on box hex20 $44^3$ and the `ssh8` $256^2$ plate in grid, RCM, Morton and shuffled order; gather on the box at m = 1 only (the 11×), after `unstructured_memory` confirms it fits.
- **`benchmarks/compile_study.py` → `benchmarks/results/compile_study.json`** (about 20 min): full `solve_modes` (20 modes) with GMG (voxel $20^3$, $40^3$, $80^3$) and AMG (box hex20 $8^3$, $24^3$; the `ssh8` plate $64^2$, $256^2$), each in a fresh subprocess with the persistent cache disabled. A `solve_modes` call compiles many jits, so the compile time is the sum of `/jax/core/compile/backend_compile_duration` events (`jax.monitoring.register_event_duration_secs_listener`), with the summed `jaxpr_trace_duration` and the wall time; next to each, the structure counts that drive it (GMG / AMG levels, ELL classes, slab count, number of compiles).

### Bimodal LOBPCG peak (time-boxed, about 45 min of runs, each invocation ≤ 1 h)

The 791k plate cold `solve_modes` (5 fresh runs per setting, `docs/open_items.md`), the peak read by the parent (`proc_pid_rusage(child_pid)` interval footprint and `ru_maxrss`, `harness.py`), so no sampler runs in the measured process:
1. An in-process polling thread (`proc_pid_rusage` every 5 ms; 18c found it gives the low regime every time): the cheapest deterministic setting if it holds over 5 runs on fp64 and fp32.
2. XLA CPU intra-op thread count (`--xla_cpu_multi_thread_eigen=false`; both flags are accepted by jaxlib 0.11.2).
3. `--xla_cpu_enable_concurrency_optimized_scheduler`.

Outcome: either a setting that puts every run in one regime (and the estimators' regime), adopted in the peak tests' subprocess environment, or the findings logged and 18c's gates kept, with the roadmap's 18d bullet reworded (cause not found; deferred to the GPU check, where the allocator differs). The estimators do not change.

## Tests (18d)

- `tests/test_reorder.py`:
  - `morton` on a $4^3$ hex8 box: the node order equals an independent reference that quantizes with the formula above and interleaves the integers with a per-bit Python loop; `check_renumbered(old, new, p, q)` on `morton`'s output for `with_sets(mixed_mesh)` and a voxel mesh (sets, material, orientation, voxel maps), as for `rcm`.
  - Equivalence (roadmap): after `morton`, apply_K and apply_M of `unstructured` (box hex8, hex20, the mixed hex8 + hex20 mesh, an `ssh8` + hex8 plate) equal the permuted original to $10^{-12}$ relative, m = 3; the same for `rcm` on the `ssh8` + hex8 plate.
- `tests/test_rigid.py` (parts): the voxel path equals the generic path as partitions on the existing cases and a non-cubic $6 \times 3 \times 2$ mask with a gap and an L-shaped part; edge and vertex contacts raise on the voxel path; after `rcm` and a random `renumber`; a spy on `parts.ndimage.label` confirms the voxel cases take the path and the non-voxel ones do not.
- `tests/test_operators.py` / `test_post.py`: `_chunked` against one chunk with exact equality for $n$ divisible by the chunk, not divisible (padded last chunk), $n$ < chunk, and a pytree output (`element_stress`); existing cases checked first and only missing ones added.
- Memory (`tests/test_strategy.py`):
  - `test_amg_scan_cached` rewritten: the same object on a second call, and a spy on the scan's inner work counts one call.
  - Lifetime: a fresh mesh (no cached helper), `_amg_scan`, `_shell_scan` (shell mesh) and `_layout_sizes` (scatter and gather) called, `r = weakref.ref(mesh)`, `del mesh; gc.collect()`: `r() is None` and its `_CACHE` entry is gone; two meshes keep two entries.
  - `_layout_sizes` against a fresh `layout` on gather and scatter.
  - `_default_budget` under `fake_gpu` with `memory_stats` raising, `None`, `{}`, `{"bytes_in_use": x}` (all `(ram, None)`) and a full dict.
- `tests/test_amg.py`:
  - `test_amg_step` includes the polish for fp64 (its docstring updated); the fp64 term bounds $f_l\,ws + \text{polish}$ against the measured `memory_analysis` at hex20 $12^3$, hex27 $12^3$ and hex8 $24^3$, m = 25.
  - The TIGHT point (hex20 $24^3$, m = 25, `slow`) within the 1.3 gate for fp64 as `test_amg_step_tight32` does for fp32.
  - `test_fp32_constants_pinned` pins the new `AMG_STEP[(fam, "polish")]` against the recorded fit (its constant count updated).
  - `polish_options={"on": False}` drops the fp64 term.
- `tests/test_performance.py` (new, fast, no wall-time assertion):
  - `test_roofline_model`: the flop and byte formulas on $87^3$ (0.759 GFLOP) and hex8 (1152 flops, 2400 B per element) by hand, and the analytic flops of the stencil and stored-Ke apply ≤ 1.01× `jit(...).lower(x).compile().cost_analysis()["flops"]` on small meshes (the analytic count is the minimum).
  - `test_fusion_report`: the HLO parser on a tiny stencil apply returns ≥ 1 fusion and counts every entry op.
  - `test_compile_listener`: the compile-time collector returns > 0 s for a tiny jitted function in a subprocess with the cache disabled.
  - `test_results_pinned`: `kernels.json` and `compile_study.json` have the keys and cases the docs tables read.
- If the bimodal outcome adopts a setting: the peak tests' subprocess environment carries it, and `test_harness.py` checks it is set.

## Docs (18d)

- `docs/performance.md` (new): the roofline (definition, the table from `kernels.json`, the byte counts, the implementation bytes), the fusion report, the compile times with their structure counts, the ordering costs, the bimodal outcome, the deferred GPU checks (pointing to `docs/ROADMAP.md`).
- `docs/conventions.md:89` and `docs/ROADMAP.md:207`: compile timing of one jit by `jit(f).lower(...).compile()`, and of a multi-jit call by the summed backend-compile events; both in a fresh subprocess with the cache disabled.
- `docs/mesh.md`: `morton` in the helper list (line 3) and a section; the ordering advice with the measured costs.
- `docs/operators.md:206`: the layout cache replaces "caching `layout` is PR 18d" (0.22 s at 812k, 3.2 s at 10.2M once per mesh); `_chunked` pipelining in the assembled section; Morton in the equivalence test list (`:287ff`).
- `docs/boundary_conditions.md:53`: the voxel path and its measured cost (29×, 93 B per element); `:90`, the parts tests.
- `docs/eigensolvers.md`: `:167`, a device without `bytes_limit` uses the CPU rule; `:414–417`, the polish term exists for every AMG family; `:487`, the bimodal outcome; `:901–908`, the test summary.
- `docs/preconditioners.md`: `:597–611`, the fp64 hex polish constants; `:627`, the bimodal outcome; `:672`, the test summary; the $P_0^\top$ and 1.25× closures next to the shell storage text.
- `docs/postprocessing.md`: the `_chunked` memory line (two chunk results).
- `docs/open_items.md`, line to outcome:

  | Line | Item | Outcome |
  |---|---|---|
  | 165, 182 | layout cache, scan cache, `gmg_memory` | per-mesh weak cache; `gmg_memory` closed (0.11 s at 10M) |
  | 183 | Metal / budget | CPU rule without `bytes_limit` |
  | 234 | `_chunked` | pipelined |
  | 241 | $P_0^\top$ scatter, 1.25× classes | closed with numbers (3–9× slower; 6–10% padding) |
  | 245 | Schwarz `chunk_bytes` "public option in 18d if needed" | deferred to the GPU check (PR 18a's) |
  | 268, 293 | 791k solve peak, bimodal | the investigation's outcome |
  | 280 | size-dependent `STORAGE32` | closed with numbers |
  | 283, 284 | fp64 hex polish; fp32 above fp64 on hex | the term added; 284's ranking statement re-measured or removed |

  plus a PR 18d section with the decisions below.
- `docs/ROADMAP.md`: the 18d status; deferred GPU checks: the roofline ≥ 50% at m = 1 for both kernels (against the measured fp64 matmul peak, as on the Mac), Morton against RCM on the GPU scatter path, compile times on GPU, the Schwarz scan steps (PR 18a, unchanged); the bimodal bullet reworded if no cause is found.
- `README.md`: the status line (PR 18d); a link to `docs/performance.md` next to the operators / eigensolvers links; `morton` where `rcm` is mentioned.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **18d stays one PR:** the items are small and independent.
- **No automatic reordering and no warning;** `morton` is a helper only.
- **Closed without code:** `gmg_memory` caching, the $P_0^\top$ scatter, the 1.25× classes, the size-dependent `STORAGE32` (numbers above).
- **Deferred to GPU:** the Schwarz scan remedy, the XLA temporaries on GPU, the roofline ≥ 50%.
- **The roofline test checks the model, not the speed:** the timed percentages are a benchmark (a timed bound at 8–16% could never fail).
- **The bimodal investigation is time-boxed;** without a cause, 18c's gates stay.
- **The fp64 hex polish term can shrink a band chunk** chosen from the estimate; eigenpairs are unchanged.
- **Cache lifetime:** results live as long as the mesh (scan arrays of a few bytes per DOF; the layout entry is scalars).
