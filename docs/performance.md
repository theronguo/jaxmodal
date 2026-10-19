# Performance

Measured kernel efficiency, compile times and the cost of node/element order (PR 18d). These numbers are for CPU fp64 on an M3 Max: $B$ = 185.9 GB/s JAX triad (C triad 189.6 GB/s), $F$ = 475 GFLOP/s fp64 matmul, both rerun by `benchmarks/kernels.py` (`benchmarks/results/kernels.json`, part `roofline`, field `system`). Byte counts are decimal. Timed runs are the min of 5 after two warm calls, each case in a fresh subprocess with macOS's large-block cache on (`harness.measure(..., large_cache=True)`, [conventions](conventions.md#benchmark-harness)), so freed blocks are reused and first touch is excluded. Peaks keep `MallocLargeCache=0`.

The GPU acceptance criteria (roofline ≥ 50%, full-Ke storage, Morton on the GPU scatter path, compile times) are deferred checks ([`ROADMAP.md`](ROADMAP.md), "Deferred GPU checks").

## Roofline

The attainable rate is $\min(B\,I, F)$ with intensity $I$ = flops / bytes. The **lower-bound bytes** count each stored or streamed array once and the vectors once per apply (read $u$, write the output). The **implementation bytes** add the traffic the current code moves through the memory hierarchy. The headline is the achieved rate over the lower-bound roofline; the implementation column is secondary.

- **Voxel stencil** ($87^3$ full box, one material): $2 \cdot 24^2 = 1152$ flops per active cell and column. Lower bound: 12 B per active cell (`cell` id 4, $s$ 8; one material, so `phase` is not read) and $48m + 4$ B per active node ($u$, $y$ and the `node` id), counted on the active grid, not the padded `grid_layout` arrays. At $m$ = 1: $12 \cdot 87^3 + 52 \cdot 88^3$ = 43.34 MB for 0.759 GFLOP, so $I$ = 17.5 flop/B, above $F/B$ = 2.6: the roofline is the fp64 peak. Implementation: plus the corner stack and shifted adds through U and Y, about $4 \cdot 24 \cdot 8\,m$ B per cell ([operators](operators.md#voxel-stencil-stencil), "Why this layout"). Its GEMM is $24 \times 24$ by $(24, N)$; the bare GEMM's rate (measured beside each row) is a second, attainable ceiling.
- **Stored-Ke `unstructured`** (scatter path; hex8 packed upper triangle, 300 doubles): $2k^2$ flops per element and column ($k$ = 24: 1152; hex20 $k$ = 60: 7200). Lower bound: the packed $K_e$ ($8 \cdot k(k+1)/2$ B), the int32 `dofs` ($4k$) and `s_K` (8) per element, 2504 B for hex8 and 14888 B for hex20, plus $48m$ B per node. At $m$ = 1 this is 0.45–0.48 flop/B, a cap of about 84–89 GFLOP/s; the intensity grows with $m$. Implementation: the unpack `Ke[:, _packed_index(k)]` is materialized ($8k^2$ B written and read per element; 18.9 MB per 4096-element hex8 chunk, within the 48 MB SLC) and the gathered $u$ and scattered output ($8km$ B, each written and read).
- `cost_analysis()["flops"]` counts a `lax.scan` or while body once, not per trip, so it is only usable on one-trip programs. `tests/test_performance.py::test_roofline_model` checks analytic ≤ `cost_analysis` ≤ 1.5× (stencil, 1.057 measured) or 1.25× (stored Ke, 1.008–1.021) on programs with no `while` opcode, and that the multi-trip controls have one.

| kernel | size | $m$ | time | $I$ (flop/B) | roofline (GFLOP/s) | achieved (GFLOP/s) | % roofline | % impl. roofline | % bare GEMM |
|---|---|---|---|---|---|---|---|---|---|
| stencil | $87^3$ (2.04M DOF) | 1 | 15.3 ms | 17.5 | 475 | 49.6 | 10.4 | 19.3 | 16 (307 GFLOP/s) |
| | | 4 | 44.1 ms | 21.4 | 475 | 68.7 | 14.5 | 26.4 | 20 (338) |
| | | 16 | 167 ms | 22.7 | 475 | 72.5 | 15.3 | 27.7 | 22 (334) |
| hex8 | $48^3$ (353k DOF) | 1 | 13.4 ms | 0.45 | 83.8 | 9.5 | 11.3 | 55.7 | |
| | | 4 | 18.3 ms | 1.70 | 316 | 27.8 | 8.8 | 48.7 | |
| | | 16 | 41.8 ms | 5.55 | 475 | 48.8 | 10.3 | 35.3 | |
| hex20 | $32^3$ (421k DOF) | 1 | 23.3 ms | 0.48 | 88.7 | 10.1 | 11.4 | 56.5 | |
| | | 4 | 25.1 ms | 1.83 | 341 | 37.6 | 11.0 | 56.9 | |
| | | 16 | 43.6 ms | 6.34 | 475 | 86.6 | 18.2 | 43.1 | |

Reading:

- Both kernels reach 9–18% of the lower-bound roofline. The stored-Ke apply at $m \le 4$ moves about half of the implementation roofline's bytes per second through the hierarchy (49–57%), consistent with the unpack (the largest $m$-independent traffic) being the limit; the stored bytes alone move at about 21 GB/s at $m$ = 1. Thread use is the other candidate, not determined: a later CPU-time over wall-time probe on a loaded machine saw about 2.3 busy cores for the stored-Ke apply at $m$ = 1 (5.3 at $m$ = 16) and 4.2–4.9 for the stencil, against the 12 that B and F assume. The stencil reaches 16–22% of its own bare GEMM.
- **Chunk-512 probe** (hex8 $48^3$, $m$ = 1, 2.4 MB of unpacked $K_e$ per chunk, L2-sized, 216 loop trips): 15.9 ms against 13.4 ms at the default 4096 (27 trips). Not faster, so it is **not determined** whether the 4096 chunk is limited by bandwidth beyond L2: the 8× trip count biases it toward not faster, and with 16 MB of L2 per P-cluster (the default chunk's unpack is split across both) the probe cannot separate L2 from the SLC.
- **Full-Ke storage on GPU:** storing the full $K_e$ (4608 B per hex8 element, 1.9× the storage) or a packed symmetric matvec would roughly halve the $m$ = 1 traffic. On CPU the packed storage stays, since storage limits the 10M reach. On GPU this is a decision (deferred check).

## Fusion

`kernels.py`, part `fusion`, walks the compiled HLO of each program over every computation reachable from ENTRY (while, conditional and call bodies; fused computations count as fusions). It records per computation the fusion count, the non-fused ops by opcode, the largest materialized outputs and `memory_analysis` temp bytes. Sizes: voxel $24^3$, box hex8 $24^3$ (13824 elements, 4 chunks), hex20 $12^3$ AMG, all at $m$ = 4.

- **Stencil:** one computation, 17 fusions, no unfused op; temporaries 21.2 MB against 1.5 MB of vectors (the two 10.6 MB slab buffers).
- **Stored Ke:** ENTRY holds the chunk `while`, 2 fusions and one copy; the loop body has 7 fusions and one copy, the largest the 18.9 MB unpack; temporaries 25.7 MB.
- **AMG V-cycle (hex20):** ENTRY holds 16 while loops (the block-ELL scans and p-level chunk loops), 11 unfused `dot`s (small projections), 2 custom-calls (the dense coarse solve) and 27 copies; the loop bodies add 4 `dot`s and 16 copies, each body 4–8 fusions. Temporaries 51.4 MB against 0.79 MB of vectors (the p-level unpacks, 29.5 MB each).

No large unfused elementwise op appears, so no further target was added.

## Compile times

`benchmarks/compile_study.py` (`benchmarks/results/compile_study.json`): one cold `solve_modes` (20 modes, default options) per case in a fresh subprocess with the persistent compilation cache disabled. A `solve_modes` call compiles many jits, so the compile time is the sum of the `/jax/core/compile/backend_compile_duration` events (`jax.monitoring.register_event_duration_secs_listener`); beside it the summed lowering (`jaxpr_to_mlir_module`) and trace durations, the number of compiles and the wall time. A run with no compile event or a cache hit raises. These runs keep `MallocLargeCache=0` (the harness default), so the wall times include first-touch faults. The structure counts are read from a preconditioner built after the solve: levels (including the coarsest), block-ELL buckets per AMG level matrix, and for GMG the fine stencil's slab and slab count.

| case | DOF | wall | backend compile | lowering | trace | compiles | structure |
|---|---|---|---|---|---|---|---|
| voxel $20^3$ (GMG) | 27.8k | 6.6 s | 4.79 s | 0.42 s | 0.38 s | 254 | 5 levels, 1 slab |
| voxel $40^3$ | 207k | 13.5 s | 5.42 s | 0.51 s | 0.41 s | 281 | 6 levels, 1 slab |
| voxel $80^3$ | 1.59M | 72.3 s | 6.41 s | 0.60 s | 0.47 s | 308 | 7 levels, 8 slabs of 10 |
| box hex20 $8^3$ (AMG) | 8.0k | 3.4 s | 2.57 s | 0.29 s | 0.18 s | 169 | 3 levels, 1 ELL bucket |
| box hex20 $24^3$ | 182k | 13.8 s | 3.15 s | 0.32 s | 0.23 s | 204 | 4 levels, 1 / 1 buckets |
| plate `ssh8` $64^2$ (shell AMG) | 25.4k | 5.1 s | 2.54 s | 0.36 s | 0.31 s | 144 | 3 levels, 1 bucket |
| plate `ssh8` $256^2$ | 396k | 30.8 s | 3.16 s | 0.37 s | 0.37 s | 159 | 4 levels, 1 / 1 buckets |

- Compile time tracks the number of levels (each level compiles its own smoother and transfer applies) more than the size: 4.8 → 6.4 s for GMG from 28k to 1.6M DOF, 2.5–3.2 s for AMG. It is 9–10% of the wall time at 0.4–1.6M DOF, 23–40% at 182–207k and 50–75% below 30k.
- The AMG level matrices on these boxes and plates are plain ELL (one bucket: padding ≤ 1.5), so width classes add no compiles here.
- Lowering and tracing together are under 1.1 s in every case.

## Node and element order

`kernels.py`, part `orders`: apply_K (scatter path, $m$ = 1 and 16), the AMG V-cycle ($m$ = 16), and on the box the gather path at $m$ = 1 (`unstructured_memory` confirms it fits). Box hex20 $44^3$ (1.08M DOF) and the `ssh8` $256^2$ plate (396k DOF) in grid (`box_mesh` order), RCM, Morton and shuffled order. Shuffled means random node and per-block element permutations, the worst case for a Gmsh or Abaqus file. The span is the median over elements of max − min node index.

| mesh | order | span | apply $m$ = 1 | apply $m$ = 16 | V-cycle $m$ = 16 | gather $m$ = 1 |
|---|---|---|---|---|---|---|
| box hex20 $44^3$ | grid | 8146 | 61.5 ms | 117 ms | 1.14 s | 84 ms |
| | RCM | 14879 | 61.0 ms | 119 ms | 1.16 s | 116 ms |
| | Morton | 326 | 62.2 ms | 123 ms | 1.22 s | 1.09 s |
| | shuffled | 328865 | 80.3 ms | 235 ms | 2.19 s | 1.46 s |
| plate `ssh8` $256^2$ | grid | 66307 | 9.1 ms | 23.9 ms | 386 ms | |
| | RCM | 729 | 9.0 ms | 22.8 ms | 386 ms | |
| | Morton | 25 | 9.0 ms | 23.8 ms | 397 ms | |
| | shuffled | 105630 | 10.8 ms | 53.2 ms | 728 ms | |

- A shuffled mesh costs +19–123% on apply_K and +89–92% on the V-cycle; RCM recovers both (`rcm` itself 0.66 s at 358k nodes in the PR 18d plan's prototype, about 3 s at 10M DOF; [mesh](mesh.md#renumbering)); on the gather path it is 38% slower than grid order (116 against 84 ms). Call `rcm` on meshes read from files.
- Morton is not faster than grid or RCM order on the scatter path (within 7%; on the plate its span is the smallest, 25, with apply_K equal and the V-cycle 3% above grid order; that row was rerun alone after code review round 1 fixed the key scaling) and makes the gather path 13× slower than grid order (9.4× RCM): its jumps between octants widen the gather windows, although the span is small: `layout(mesh, "gather")` on the box gives (R, ranges, W) = (32768, 11, 10240) grid, (32768, 11, 14336) RCM and (16384, 22, 68608) Morton, a window work ranges × W of 1.40× and 13.4× grid order against the measured 1.38× and 12.9×.
- `morton` stays a helper; `solve_modes` does not reorder (it would permute the mode vectors back and forth) and does not warn.

## Bimodal LOBPCG peak

`benchmarks/bimodal_probe.py` (`benchmarks/results/bimodal_probe.json`), the cold fp32 `solve_modes` on the flat $362^2$ plate (791k DOF, 20 modes, σ given, synchronous CPU dispatch; `shell_solve_study.solve_peak`'s child) in fresh processes on an otherwise idle machine (load average 3–8, from the runs themselves), each classified by its interval peak above 5.8 GB (midway between 4.75 and 6.9–7.1 GB; [open items](open_items.md#bimodal-lobpcg-peak-on-the-791k-plate-pr-18c--pr-18d-user-decision-2026-10-19)). About 30 minutes of recorded runs, inside the time box.

| runs | peak | regime |
|---|---|---|
| baseline, 5 interleaved | 7.07–7.15 GB | high, 5 of 5 |
| `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, 5 | 6.79–7.23 GB | high, 5 of 5 (solve time unchanged: the flag has no visible effect) |
| `XLA_FLAGS=--xla_cpu_enable_concurrency_optimized_scheduler`, 5 | 7.04–7.14 GB | high, 5 of 5 (solve time unchanged) |
| step wrapper (live bytes + own `proc_pid_rusage` + full-output sync) with a parent sampler, 3 | 4.73–4.74 GB | low, 3 of 3 |
| the wrapper's full-output sync alone, 3 | 6.99–7.12 GB | high, 3 of 3 |
| the wrapper's instruments alone (no sync), 3 | 4.73–4.74 GB | low, 3 of 3 |
| `jax.live_arrays()` alone at each step's start and end, 2 | 7.10–7.13 GB | high, 2 of 2 |
| the process's own `proc_pid_rusage` alone at each step's start and end, 2 | 4.74 GB | low, 2 of 2 |
| a parent sampling the child's footprint every 100 ms, no wrapper, 2 | 5.52–5.56 GB | between |

- The extra is resident memory, not footprint accounting: in two more baseline runs (part `checks`) `ru_maxrss` reads 7.01–7.09 GB, the footprint interval maximum 7.03–7.10 GB.
- The switch is the process's own `proc_pid_rusage` call between LOBPCG steps (8 of 8 runs low; 22 of the 24 runs without it high in this session, the 2 parent-sampler runs between, against a mix in PR 18c's sessions). PR 18c's polling thread made the same call every 5 ms. A query from another process moves it only partly. The solve time does not change (58.6–59.0 s in every row; process wall time 61.4–61.8 s).
- In the low regime the live `jax.Array` bytes stay at 2.5 GB over the 25 steps; the interval-maximum footprint (operator included) reaches 5.3 GB in step 9 (3.85 GB before) and no later step exceeds it, while the current footprint stays at 2.8–2.9 GB.
- **Cause not found** within the time box. The mechanism by which a footprint query keeps 2.3 GB from becoming resident (allocator retention or reuse timing inside XLA:CPU's runtime) is open; no XLA flag tried changes it, and a syscall per step in `solve_modes` would be a macOS-specific workaround for an unexplained effect, so nothing is adopted. The estimators stay the `memory_analysis` envelope, PR 18c's gates stay (`test_shell_791k_peak32`: $0.75\,\text{peak} \le$ estimate $\le 1.2\,\text{peak}$), and the question moves to the GPU check, where the allocator differs.
