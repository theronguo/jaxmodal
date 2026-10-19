# PR 18 plan: performance pass, part d: kernels

Branch `pr-18d-kernels` from `main` (1c6b651, PR 18c merged). Roadmap: "PR 18d: Kernels".

## Revisions

- Rev 1: first draft.

## Goal (18d)

Measure the kernels against the machine (roofline, fusion, compile time), add Morton ordering as a helper, and close the non-band follow-ups that PRs 15d–18c deferred here: cheap per-mesh caches, a pipelined chunk loop, the Metal budget fallback, the voxel part finder, an fp64 hex polish term, the bimodal peak investigation. No change to solver results.

## Findings (prototypes, CPU fp64, M3 Max, `benchmarks/results/system.json`: triad 185 GB/s, fp64 matmul 490 GFLOP/s)

### Ordering (apply_K, scatter path, m = 1 and 16, and the AMG V-cycle)

- **Morton vs RCM vs grid:** on box hex20 $44^3$ (811k DOF) Morton is not faster than RCM or grid order on the scatter path (within ±3%) and makes the gather path about 11× slower (Morton's jumps between octants widen the gather windows to most of the mesh). The median element node-index span: grid 8146, RCM 12399, Morton 326 (Morton is local in space but not in the index range a window must cover).
- **Shuffled meshes** (random node and per-block element permutations; the worst case for a Gmsh or Abaqus file) cost +30–100% on apply_K and +50% on the AMG V-cycle. RCM recovers both (0.66 s at 358k nodes; about 3 s and 1.6 GB of host memory at 10M DOF, extrapolated).
- **Grid order on a plate** (`ssh8` $256^2$, 396k DOF): RCM leaves apply_K (23.7 / 23.1 ms at m = 16), the AMG build (5.48 / 5.69 s) and the V-cycle (382 / 381 ms) unchanged, although the median node span is 16771 in grid order against 367 after RCM. The span metric does not predict the cost here, so a span-based "reorder if badly ordered" rule would misfire.

### Roofline (first numbers from recorded results)

- Voxel stencil, $87^3$, m = 1 (`stencil_slab.json`): 18.4 ms per apply_K; 1152 flops per cell, so 41 GFLOP/s. Data moved ≈ 16 B per cell + 48 B per node ≈ 42 MB, so the arithmetic intensity is ≈ 18 flop/B and the roofline is the fp64 peak (490 GFLOP/s): about 8%.
- Stored-Ke apply (hex8, packed upper triangle, 300 doubles per element): about 0.48 flop/B, so it is bandwidth-capped at about 89 GFLOP/s; it runs at 11–16% of the triad in Ke traffic (PR 5, kernel-bound: the unpack and contraction).

### Follow-ups

- **`unstructured_memory` (gather):** `layout` costs 0.22 s at 812k DOF (half the windows, half the inverse map) and 3.2 s at 10.2M per call, about 35 s per band call there. The estimator needs only scalars from it (per block `n_pad`, `chunk`, `k`, `W`; `R`, `n_ranges`, `valence`, the inverse-map size).
- **`gmg_memory`:** 0.11 s per call at 10.3M DOF (voxel $150^3$), with or without `s_K`. Not worth a cache: closed with the number.
- **`_amg_scan` / `_shell_scan`:** `lru_cache(maxsize=1)` keeps the last mesh (and its scan) alive after the caller drops it.
- **Voxel part finder:** `ndimage.label` on the cell mask (6-connectivity), mapped through `element_map`, plus the existing node-label contact check: 0.025 s against 0.72 s for the generic search on $100^3$ with a gap layer (2 parts, same partition); 29×.
- **$P_0^\top$ as a scatter over $P_0$'s ELL:** 3–9× slower than the stored transpose on CPU (18c probe). Closed, not adopted.
- **1.25× ELL width classes:** the padding of the current classes is only 6–10% of the ELL bytes. Closed, not adopted.
- **Size-dependent `STORAGE32`:** the fp32 storage overestimate at 1M DOF is 58–86 MB, about 7–12 basis columns, under 2% of the reach, and it is conservative. Closed, not refitted.
- **fp64 hex polish:** the fp64 hex polish exceeds the fp64 low model by up to 0.33 blocks (hex20 $52^3$, m = 120; `amg_fp32_study.json` `polish64_slack`). The model must bound it: add the term.
- **`assembled._chunked`:** a dispatch-then-pull loop serializes the device and the host copy; dispatching chunk $i+1$ before copying chunk $i$ gave 8% in `element_stress` on CPU (PR 17).

## Design (18d)

### `morton(mesh)` (`mesh/reorder.py`)

```python
def morton(mesh, bits=21):
    """Morton (Z-order) node order, then elements by the Morton key of their centroid within
    each block. Returns (new mesh, p, q) like rcm."""
```

- Keys: coordinates scaled to $[0, 2^{21} - 1]$ per axis on the mesh's bounding box, bits interleaved (x lowest) in a uint64; `np.argsort(kind="stable")`. Elements: per block, `s + argsort(keys(centroids))`.
- Exported from `jaxmodal.mesh` next to `rcm`. Not a default anywhere: on CPU it does not help (Findings); its value on GPU is a deferred check (below).
- **No automatic reordering** in `solve_modes` (decision, logged): the mode vectors would have to be permuted back, and no cheap metric predicts the gain (the plate). `docs/mesh.md` gets the measured costs of a shuffled mesh and the advice to call `rcm` on meshes from files.

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

- Replaces `lru_cache(maxsize=1)` on `_shell_scan` and `_amg_scan` (Mesh is `eq=False`, so it hashes by identity and takes weak references).
- New `_layout_sizes(mesh, path, chunk, range_nodes)` under `_per_mesh`: calls `layout` once and keeps only the scalars `unstructured_memory` reads (above). `path` is resolved before the call (`DEFAULT_PATH` by backend), so the key is stable. `unstructured_memory` reads it instead of `layout`; its values are unchanged.
- Cached values are NumPy arrays and ints; none hold the mesh.

### `_default_budget` on devices without `bytes_limit` (`memory.py`)

`memory_stats()` can be `None` or lack `bytes_limit` (Metal, some plugins). Then the budget is the host RAM rule as on CPU (`ram, None`: a unified pool). The rule is unchanged where the key exists.

### `assembled._chunked` pipelined (`operators/assembled.py`)

Dispatch chunk $i+1$, then copy chunk $i$ to the host (JAX dispatch is asynchronous, so the kernel runs during the copy). One pending result is held; the output arrays and the padding rule are unchanged.

### Voxel `parts` (`mesh/parts.py`)

If `mesh.voxel is not None` and the mesh is hex8 only: element labels from `ndimage.label(element_map >= 0)` (the default cross structure is face connectivity), relabelled to `0..n-1` in first-element order as the generic path does (via `connected_components`' order: both are made canonical by `np.unique(label, return_index)` order). The node scatter and the contact check (edge and vertex contacts raise) stay as they are, shared by both paths. Valid after `renumber` (the voxel maps are renumbered with the mesh). The collapsed-element check is skipped (voxel cells cannot collapse).

### fp64 hex polish term (`memory.py`, `benchmarks/amg_fp32_study.py`)

- `steps_hex` records the fp64 polish (`polish64`) on the whole `HEX` grid, not only the held-out points; `fit` fits `AMG_STEP[(fam, "polish")]` for `hex8` and `p` with the same LP as the fp32 terms (upper envelope, rounded up), gated at 1.3 on the tight cases.
- `amg_memory` (fp64, hex) then returns `polish` like the fp32 and shell families, and `lobpcg_memory`'s step takes $\max(\text{9-block}, \text{low}, \text{polish})$ as for them. `polish_options={"on": False}` drops it as before.
- The fp32 constants and the shell constants are unchanged. The re-run of `steps_hex` (with `fit`) is the only rerun of 18c's study, ≤ 1 h.

### Benchmarks

- **`benchmarks/kernels.py` → `benchmarks/results/kernels.json`**, parts `roofline`, `fusion`, `orders`, each in fresh subprocesses (`harness.measure`):
  - `roofline`: the jitted triad (`harness.system_info`'s, 24 B per element) and the fp64 peak from `system.json`; the voxel stencil ($87^3$) and stored-Ke `unstructured` (hex8 $48^3$, hex20 $32^3$) apply_K at m = 1, 4, 16, min of 5. Per kernel: flops per apply, bytes per apply (the lower bound: each stored or streamed array once, vectors read and written once), intensity, roofline $\min(B \cdot I, F)$, achieved GFLOP/s and its percentage.
  - `fusion`: for the stencil apply_K, the stored-Ke apply_K (scatter and gather), the AMG V-cycle (hex8, hex20, shell) and one LOBPCG step: the compiled HLO's fusion count, the non-fused ops of the entry computation by opcode, and `memory_analysis` temp bytes against the vector bytes. Report only.
  - `orders`: apply_K (scatter, gather) at m = 1, 16 and the AMG V-cycle at m = 16 on box hex20 $44^3$ and the `ssh8` $256^2$ plate in grid, RCM, Morton and shuffled order.
- **`benchmarks/compile_study.py` → `benchmarks/results/compile_study.json`:** full `solve_modes` (20 modes) compile time with GMG (voxel $20^3$, $40^3$, $80^3$) and AMG (box hex20 $8^3$, $16^3$, $24^3$; the `ssh8` plate $64^2$, $256^2$), each in a fresh subprocess with the persistent cache disabled (ROADMAP: compile tests run that way): the summed `/jax/core/compile/backend_compile_duration` events (`jax.monitoring.register_event_duration_secs_listener`), the jaxpr trace time and the total wall time.
- Budget: each part ≤ 20 min, each script ≤ 1 h.

### Bimodal LOBPCG peak (time-boxed, ≤ 3 h of runs)

Probe the 791k plate cold `solve_modes` (5 fresh runs per setting, as in `docs/open_items.md`):
1. The peak counter: `ri_phys_footprint` interval max against `ru_maxrss`, and the footprint sampled every 50 ms, to tell a real allocation from deferred reclaim.
2. The allocator: `MallocNanoZone=0`, and the XLA CPU allocator flags.
3. XLA CPU concurrency: intra-op thread count and `--xla_cpu_enable_concurrency_optimized_scheduler`.

Outcome: either a setting that puts every run in one regime, adopted in the peak-test subprocess environment and documented in `docs/performance.md`, or the findings logged and the gates kept as they are (18c's user decision). The estimators are not changed by this item.

## Tests (18d)

- `tests/test_reorder.py`:
  - `morton` returns permutations (p, q block-preserving); on a $4^3$ grid the node order equals the bit-interleaved order computed independently (a loop over grid indices).
  - Equivalence (roadmap): after `morton`, apply_K and apply_M of `unstructured` (box hex8, hex20, a mixed hex8 + hex20 mesh, and an `ssh8` + hex8 plate) equal the permuted original to $10^{-12}$ relative, m = 3; also `rcm` on the same cases (already tested for hex8; added for `ssh8`).
- `tests/test_rigid.py` (parts): the voxel path equals the generic path (as partitions) on the existing part cases (two blocks, a gap layer, a mask with an L-shaped part); edge and vertex contacts raise on the voxel path; after `rcm` and a random `renumber`; a voxel mesh without hex8-only blocks never takes the path.
- `tests/test_operators.py` / `test_post.py`: `_chunked` results unchanged (the existing assembled and `element_stress` tests), plus a chunked case with 3 chunks against one chunk.
- `tests/test_strategy.py` or `test_api.py` (memory):
  - the per-mesh cache: after `del mesh; gc.collect()` the entry is gone (`len(_CACHE)` back to the start); two meshes keep two entries.
  - `_layout_sizes` against a fresh `layout` on gather and scatter, and `unstructured_memory` unchanged against recorded values on two meshes.
  - `_default_budget` with a monkeypatched device whose `memory_stats()` is `None`, `{}`, and a full dict.
- `tests/test_amg.py`: the fp64 hex polish term ≥ the measured fp64 polish `memory_analysis` (at hex20 $12^3$ and hex8 $24^3$, m = 25) and within the 1.3 gate on one tight point; `polish_options={"on": False}` drops it.
- `tests/test_performance.py` (new):
  - `test_roofline` (`serial`): stencil $48^3$ and stored-Ke hex8 $32^3$ at m = 1 and 16; the achieved rate is ≤ 1.1× the roofline (the bound holds; the intensity model is a lower bound on bytes) and > 0; the percentages are printed. ≥ 50% at m = 1 is a deferred GPU check.
  - `test_fusion_report`: the HLO parser on a tiny stencil apply returns ≥ 1 fusion and counts every entry op.
  - `test_compile_listener`: the compile-time collector returns > 0 s for a tiny jitted function in a subprocess with the cache disabled.

## Docs (18d)

- `docs/performance.md` (new): roofline (definition, the table from `kernels.json`, how the bytes are counted), fusion report, compile times, ordering costs, the bimodal peak outcome, and the deferred GPU checks pointing to `docs/ROADMAP.md`.
- `docs/mesh.md`: `morton`; ordering advice with the measured costs.
- `docs/operators.md:206`: the layout cache; `_chunked` pipelining in the assembled section.
- `docs/boundary_conditions.md:53`: the voxel part finder and its cost.
- `docs/preconditioners.md` / `docs/eigensolvers.md`: the fp64 hex polish term; $P_0^\top$ and the 1.25× classes closed with numbers.
- `docs/open_items.md`: each follow-up resolved or closed; the decisions below.
- `docs/ROADMAP.md`: 18d status; deferred GPU checks: the roofline ≥ 50% at m = 1 for both kernels, Morton against RCM on the GPU scatter and gather paths, the compile times on GPU, the Schwarz scan steps (unchanged, PR 18a).
- `README.md`: the docs list (`performance.md`), `morton` in the mesh helpers.

## Risks and decisions without user input (logged in `docs/open_items.md`)

- **18d stays one PR:** the items are small and independent; splitting adds merges without a review gain.
- **No automatic reordering;** `morton` is a helper only (Findings: no CPU gain, the span metric misfires on plates).
- **Closed without code:** `gmg_memory` caching (0.11 s per call at 10M), $P_0^\top$ scatter (3–9× slower), 1.25× classes (6–10% padding), size-dependent `STORAGE32` (< 2% of the reach, conservative).
- **Deferred to GPU:** the Schwarz scan remedy (PR 18a's check decides), the XLA temporaries on GPU, the roofline ≥ 50%.
- **The bimodal investigation is time-boxed;** without a cause, 18c's gates stay.
- **Cache lifetime:** the per-mesh cache holds results as long as the mesh lives (scan arrays of a few bytes per DOF; the layout entry is scalars only).
