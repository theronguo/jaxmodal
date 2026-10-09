# Conventions

Fixed in PR 0 (roadmap Principle 7). Later PRs follow these without restating them.

## Data and numerics

- **Node ordering:** VTK internally. Other formats are converted at import (`mesh/io`).
- **Voigt order:** $[11, 22, 33, 23, 13, 12]$ with engineering shear strains, $\gamma_{ij} = 2\varepsilon_{ij}$.
- **Precision:** fp64 by default, but only through an explicit call:

  ```python
  import jaxmodal.config

  jaxmodal.config.enable_x64()
  ```

  Importing `jaxmodal` or any submodule changes neither the x64 nor the compilation-cache setting (tested). fp32 inputs stay fp32 under x64.
- **Mixed precision:** iteration vectors, returned modes, Gram matrices and Rayleigh–Ritz are always fp64. fp32 is allowed only inside preconditioners and for the device-side deflation basis. Returned locked modes come from an fp64 host copy.
- **Indices:** index arrays are int32. Under x64, `argsort`, `nonzero` and `arange` return int64, so they go through `config.index` (pass `jnp.nonzero(a)[0]`, not the tuple). `config.index` is an unchecked cast: values $\ge 2^{31}$ wrap. The overflow asserts come in PR 4 and run in NumPy before casting.
- **Loops:** differentiated loops (operator chunk loops) use static trip counts (`scan`, or `fori_loop` with Python-int bounds). Traced bounds appear only in solvers, which are never differentiated.
- **Block size:** $m$ always denotes the LOBPCG block size (nev plus guard vectors).

## Tolerances (Principle 1)

"$a$ equals $b$ to tol" means

$$\|a - b\|_\infty \le \mathrm{tol}\,\|b\|_\infty,$$

with the max over all points for pointwise fields such as stress. Absolute tolerances are used only for dimensionless quantities, e.g. $V^\top M V - I$.

## Memory (Principle 6)

- All `dynamic_slice`/`dynamic_update_slice` start indices are `jnp.int32`, also under x64.
- Capacities are padded so neither ever clamps. On the read side, capacity is a multiple of the chunk size, and unused columns are zero. On the write side, capacity $\ge$ max count + insert width, and the host asserts this before every insert.

## Measured facts (jax 0.11.2, macOS arm64)

- **Persistent compilation cache:** works on CPU. The tests enable it at `.jax_cache/`, which grows without bound; delete it after jax upgrades. If a later jax stops storing entries, the cache test fails, and the remedy is removing `enable_cache` from `tests/conftest.py`. Nothing depends on the cache for correctness.
- **Buffer donation:** works on CPU. A donated input is deleted, and a donated `dynamic_update_slice` keeps its buffer pointer.
- **`jax_num_cpu_devices`:** available.
- A jax bump re-runs these tests.

## Benchmark harness

`benchmarks/harness.py`:

- `measure(code, setup)` runs `setup`, then `code`, in a fresh subprocess and returns `peak_bytes`, `rss_bytes` and `time_s` for `code`.
- `python benchmarks/harness.py` writes the machine peaks to `benchmarks/results/system.json`. These are fp64 matmul GFLOP/s, jitted and pthread C STREAM triad GB/s, and versions. The C triad runs at $P$ and $P+E$ threads (P/E cores; `os.cpu_count()` when unknown) and records the best run with its thread count. Later results reference this file instead of re-measuring.
- `write_result(name, data, out_dir)` writes `<out_dir>/<name>.json` (default `benchmarks/results/`) with `data` plus `versions()`.

### Peak definition

On macOS, `peak_bytes` is

$$\Delta = \max\left(F_{\max} - F_0,\ R_{\max} - R_0\right).$$

- $F$ is `phys_footprint` from `proc_pid_rusage`. $F_{\max}$ is the interval maximum after `proc_reset_footprint_interval` (a private libSystem symbol), and $F_0$ is the footprint right after the reset.
- $R$ is `ru_maxrss`, read at the same baseline point and after `code`. This $R$ delta is also returned as `rss_bytes`.
- Both terms are lower bounds:
  - footprint does not recharge freed malloc blocks of 2 MB or less when they are reused;
  - `ru_maxrss` misses peaks below an earlier high-water mark.
- Elsewhere `peak_bytes` is `None`. The Linux and GPU paths come in PR 24.

### Mechanisms

Each mechanism exists because of a measured failure, and each has a test that fails without it.

| Mechanism | Failure without it (measured) |
|---|---|
| `MallocLargeCache=0` in the child env (undocumented) | a 1.6 GB block freed in `setup` stays resident and is reused uncharged; a 0.8 GB allocation in `code` reads 0 |
| `gc.collect()` after `setup` | cyclic garbage left by `setup` is freed during `code`; the allocation reads 0 |
| footprint poll after `setup` (10 ms steps until it changes by < 1 MiB; at least 50 ms, at most 2 s) | XLA:CPU frees executable temporaries on another thread after `block_until_ready`; 3.2e9 reads ≈ 2.1–2.4e9 |
| interval reset before the baseline | a 2 GB transient in `setup` inflates the peak to 2.0e9 |
| interval maximum, not the current footprint | 0.8 GB allocated and freed in `code` after a higher `setup` mark reads 0 (`ru_maxrss` reads 0 too) |
| max with the `ru_maxrss` delta | 800 × 1 MB allocated, freed, then 1600 × 1 MB allocated, all in `code`: footprint reads 0.82e9 instead of 1.6e9 |

### Rules

- `setup` and `code` block on their results (`.block_until_ready()`). Otherwise async work spills into `code`, and `time_s` measures only dispatch.
- Warm up in `setup` with compiles only (`f.lower(...).compile()`) plus backend start-up. This gives the same peak as a full warm-up run.
  - Run the full function in `setup` only when its buffers are larger than 2 MB. Blocks of 2 MB or less freed in `setup` are reused by `code` uncharged by both footprint and `ru_maxrss`. Measured: 0.8 GB of 1 MB blocks freed in `setup` hide an identical allocation in `code` completely.
  - After a compile-only warm-up, `time_s` includes about 2 ms of first-run overhead.
- `ru_maxrss` also counts library code pages first touched in `code`, so warm up the backend in `setup`.
- Footprint counts compressed pages at their compressed size, so estimator checks use non-constant data.
- On macOS `time_s` includes the page faults of fresh mmaps, since freed large blocks are unmapped.
- The STREAM triad counts 24 B per element ($a = b + s c$; write-allocate traffic is not counted).
- `measure` records no compile time. Compile-time tests (PRs 5 and 18) time `jit(f).lower(...).compile()` in a fresh subprocess with the cache disabled.
