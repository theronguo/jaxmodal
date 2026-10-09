# PR 0 plan: project scaffold

Branch `pr-00-scaffold`. The scope is exactly the roadmap's PR 0 (Contents, Docs, Tests) plus Principle 7 (conventions). No FEM code. Review rounds 1–4 are incorporated.

## Measured facts (round-1 reviewers, this Mac)

- uv 0.12.23 resolves jax/jaxlib 0.11.2, numpy 2.5.3, scipy 1.18.1, meshio 5.3.5 and pyamg 5.3.0, all with cp312 macOS arm64 wheels (or pure Python). Homebrew CPython 3.12.10 is used.
- On jax 0.11.2 CPU:
  - donation works: no warning, `is_deleted()` is true, and a donated `dynamic_update_slice` keeps the buffer pointer;
  - the persistent cache stores entries;
  - `jax_num_cpu_devices` exists.
- The `pymetis` extra is deferred to PR 19, its only user. The oldest cp312 arm64 wheel is 2025.1.1, checked with `--only-binary :all:`.

## Files

```
pyproject.toml           hatchling, src layout, ruff + pytest config
uv.lock
.python-version          3.12
.gitignore               .venv, __pycache__, .pytest_cache, .ruff_cache, .jax_cache/, dist/, *.egg-info/, .DS_Store, .claude/worktrees/, .claude/settings.local.json
LICENSE                  MIT, "Copyright (c) 2026 jaxmodal authors"
README.md
scripts/check.sh         executable (git mode 100755)
src/jaxmodal/__init__.py __version__ = "0.1.0" (single version source); no other code
src/jaxmodal/config.py
benchmarks/harness.py
benchmarks/triad.c
benchmarks/results/system.json   written by `python benchmarks/harness.py`, committed
docs/conventions.md
tests/conftest.py
tests/test_config.py
tests/test_harness.py
```

## pyproject.toml

- `[project]`:
  - `requires-python = ">=3.12"`, with `.python-version` pinning 3.12;
  - `license = "MIT"`;
  - `dynamic = ["version"]`, with `[tool.hatch.version] path = "src/jaxmodal/__init__.py"`;
  - `dependencies = ["jax>=0.11.2", "numpy", "scipy", "meshio", "pyamg"]`.
- `[dependency-groups] dev = ["pytest", "ruff"]`.
- `[tool.uv] required-environments = ["sys_platform == 'darwin' and platform_machine == 'arm64'"]`, so a re-lock can't silently fall back to sdists on the Mac.
- `[tool.ruff]` sets `line-length = 100`; `[tool.ruff.lint]` sets `select = ["E", "F", "I", "UP", "B"]`.
- `[tool.pytest.ini_options]`:
  - `testpaths = ["tests"]`;
  - `pythonpath = ["."]`, so tests can import `benchmarks.harness`;
  - `markers = ["slow: long-running", "gpu: needs a CUDA device"]`;
  - `addopts = "--strict-markers"`.

## config.py

```python
def enable_x64(): jax.config.update("jax_enable_x64", True)
def enable_cache(path):  # jax_compilation_cache_dir=str(path), min_compile_time_secs=0, min_entry_size_bytes=-1
def index(x): return jnp.asarray(x, dtype=jnp.int32)
```

- Nothing runs at import.
- `index` is the single int32 cast point for index arrays. It is an unchecked cast: values ≥ 2³¹ wrap silently. The overflow asserts come in PR 4, in NumPy, before casting.
- `index` takes an array, so `nonzero` results are passed as `jnp.nonzero(a)[0]`.

## tests/conftest.py

- At session start, calls `config.enable_x64()` and `config.enable_cache(<repo>/.jax_cache)`. The cache works on the pinned jax, so this is unconditional.
- If a future jax stops storing entries, test 6 fails. The fix then is to drop the `enable_cache` call; nothing depends on the cache for correctness.
- Sets `collect_ignore = ["multidevice"]`.

## Benchmark harness (`benchmarks/harness.py`)

### `measure(code, setup="") -> {"peak_bytes", "rss_bytes", "time_s"}`

1. Runs `sys.executable -c <child>` with `capture_output=True`. On a nonzero exit it raises `RuntimeError(proc.stderr)`. The child's env drops `JAX_ENABLE_X64`. It also sets `MallocLargeCache=0` (macOS; undocumented, not in `man malloc`), so large freed blocks are unmapped instead of being kept resident in libmalloc's large cache and reused uncharged. Measured: with the cache on, a 1.6 GB block freed in `setup` keeps the footprint at 1.62 GB and an 0.8 GB `code` allocation reads 0. `malloc_zone_pressure_relief` releases 0 bytes, so it is not used. With the cache off the reading is 0.800. Fresh mmaps make repeated large allocations slower (20 × 800 MB alloc/free: 0.66 s vs 0.04 s); a cache-on timing mode is added only when a wall-time budget needs it (PR 9 or later).
2. The child returns its results as JSON written to a temp file whose path is passed in argv, so prints in `code` can't corrupt the result.
3. The child runs a fixed preamble:
   - `import numpy as np, jax, jax.numpy as jnp`;
   - x64 is off unless `setup` enables it.
4. `setup` and `code` run with `exec` in their own dict, seeded with `np`, `jax`, `jnp`, so user names can't shadow harness locals. The child then runs `setup`, takes the baseline, runs and times `code`, and reads the peak.
   - **Settling after `setup`:** call `gc.collect()` (jax leaves cyclic garbage that is otherwise freed during `code`), then poll `ri_phys_footprint` every 10 ms until two consecutive reads differ by < 1 MB, for at most 2 s (XLA:CPU frees executable temporaries on another thread ~10 ms after `block_until_ready`). Measured without settling: a setup that runs `jax.random.uniform` (3.2 GB of temporaries) once makes the same `code` read 32 KB instead of 3.2e9.
   - **macOS peak:**
     - after settling, call `proc_reset_footprint_interval(getpid())`, a private libSystem symbol, asserting that it exists and returns 0 (no silent fallback);
     - record `ri_phys_footprint` as the baseline;
     - after `code`, `peak_bytes = ri_interval_max_phys_footprint − baseline`.
     - Fields are read from `proc_pid_rusage(pid, RUSAGE_INFO_V4)` as a 16-byte uuid followed by `c_uint64 * 35`, with indices taken from the SDK header `sys/resource.h`: `ri_phys_footprint` = 7, `ri_interval_max_phys_footprint` = 33.
     - `MallocLargeCache=0` and settling stop freed `setup` memory from hiding the delta; the interval reset stops `setup` transients from inflating it. Test 4 covers each.
     - It counts compressed pages, as the roadmap's 10M runs need.
   - **Other platforms:** `peak_bytes` is `None` in PR 0. The Linux (`malloc_trim`, `clear_refs`, VmHWM − VmRSS) and GPU (`memory_stats()`) peaks are written and tested in PR 24, listed under Deferred GPU checks.
   - **`rss_bytes`:**
     - the plain `ru_maxrss` delta, recorded as the roadmap asks;
     - converted to bytes by `maxrss_bytes(value, platform)` (bytes on darwin, KiB elsewhere).
5. Documented rules:
   - `code` must block on its results (`.block_until_ready()`), or `time_s` measures only dispatch.
   - Warm up in `setup` (backend start-up, compiles, even a full run of the measured function) whatever should not count toward `peak_bytes` or `time_s`; settling makes this safe.
   - Small blocks still held by malloc's tiny/small zones can be reused by `code` uncharged; this is negligible against the GB-scale peaks the estimators check.
   - Footprint counts compressed pages at their compressed size, so estimator checks use non-constant data.
   - macOS `time_s` includes page-fault costs of fresh mmaps (`MallocLargeCache=0`).
   - `rss_bytes` is the roadmap's plain RSS delta and inherits `ru_maxrss`'s high-water caveat; `peak_bytes` is the primary number.
   - Compile time is not recorded by `measure`. Compile-time tests (PRs 5, 18) time `jit(f).lower(...).compile()` in a fresh subprocess with the cache disabled, as the roadmap says.

### `versions() -> dict` (cheap)

- jax/jaxlib/numpy versions, `platform.platform()` and `os.cpu_count()`;
- P/E core counts from `sysctl hw.perflevel0/1.physicalcpu`, or `None` on a missing binary or nonzero exit (Linux); `c_triad` then uses `os.cpu_count()` threads;
- `XLA_FLAGS` verbatim (or `None`) next to `os.cpu_count()`, XLA:CPU's default intra-op pool size;
- whether `jax_num_cpu_devices` exists.

### `system_info() -> dict`

- Calls `config.enable_x64()` and asserts that its arrays are float64.
- fp64 peak: jitted 4096² matmul, arrays passed as arguments, warm-up first, `block_until_ready` in each timed run, best of 5, 2n³ flops.
- Jitted STREAM triad: `a = b + s·c` on 2·10⁷ fp64 elements, same timing rules, best of 10, 24 B per element (STREAM convention; write-allocate traffic is not counted).
- C triad, `c_triad() -> {"gbps", "threads"} | str`:
  - `benchmarks/triad.c`, built without a shell as `[os.environ.get("CC", "cc"), "-O3", "-pthread", ...]` into a temp dir; the compiler is looked up with `shutil.which`;
  - pthreads with a mutex-and-condition-variable barrier (macOS has no `pthread_barrier_t`), first-touch initialisation inside the threads, best of 20, and the result is used so the loop can't be optimised away;
  - run at the thread counts from `triad_threads(versions())`; the best value and its thread count are reported;
  - a missing compiler gives `"skipped: no C compiler"`, a failed build `"skipped: compile failed"`;
  - the ratio to the jitted triad is recorded, not asserted.
- Includes `versions()`.

### Results

- `write_result(name, data, out_dir="benchmarks/results")` writes `<out_dir>/<name>.json` with `data` plus `versions()`.
- The peaks live only in `system.json`. Later results reference it instead of re-measuring.
- `python benchmarks/harness.py [--out DIR]` writes `DIR/system.json`; `DIR` defaults to `benchmarks/results`.

## Tests

Tests are fast unless marked `slow`. Every test that uses jit builds a fresh `jax.jit` object inside the test.

1. **fp32 stays fp32 under x64.**
   - With x64 enabled, jitted `2.0*x + jnp.sum(x)`, `x @ x.T` and `jnp.linalg.norm(x)` on fp32 inputs return fp32.
   - Control: `jnp.ones(3).dtype == float64`, which proves x64 is on.
2. **No import side effects.**
   - A subprocess imports `jaxmodal` and every submodule (`pkgutil.walk_packages(jaxmodal.__path__, "jaxmodal.")` plus `importlib.import_module`), then asserts `jax.config.jax_enable_x64` is unchanged.
   - It runs once with `JAX_ENABLE_X64=0` and once with `=1`, which catches a module that turns x64 on or off. Later modules are covered automatically.
3. **int32 index casts.**
   - Under x64, the raw `jnp.argsort`, `jnp.nonzero(a)[0]` and `jnp.arange` results are int64. This checks the premise.
   - `config.index` of each is int32 with equal values.
4. **Harness peak and RSS.** Marked `slow`.
   - `setup="rng = np.random.default_rng(0)"`, `code="a = rng.random(100_000_000)"` (800 MB, incompressible): `peak_bytes` = 0.8e9 ± 5%.
   - Over 3 runs, max/min − 1 < 5% for both `peak_bytes` and (darwin) `rss_bytes`.
   - Control: `code="pass"` gives `peak_bytes` < 20 MB.
   - Unmapped transient: `setup` touches 2 GB in an mmap and closes it, then the same `code` gives 0.8e9 ± 5%. Without the interval reset this reads ≈ 2.0e9; with `ru_maxrss` it reads ≈ 0.
   - Malloc-freed transient: `setup` also runs `t = rng.random(200_000_000); del t` (1.6 GB); the same `code` gives 0.8e9 ± 5%. Without `MallocLargeCache=0` it reads ≈ 0 (measured), so this case guards the mechanism.
   - JAX garbage: `setup` builds `x0 = jnp.asarray(rng.random(200_000_000, np.float32))` (blocked) and `f = jax.jit(lambda x: x * 2 + 1)`, runs `f(x0)` once and deletes the result; `code` is `y = f(x0).block_until_ready()`: 0.8e9 ± 5%. Without `gc.collect()` this reads ≈ 0–0.4e9 (measured).
   - XLA async temporaries: `setup` runs `g = jax.jit(lambda k: jax.random.uniform(k, (200_000_000,)))` once and deletes the result; `code` runs `g` again. `peak_bytes` is within 5% of the same `code` measured with only `g.lower(k).compile()` in `setup` (≈ 3.2e9). Without the settling poll this reads ≈ 0.
   - Namespace isolation: `measure("time = None", setup="json = os = sys = None")` returns all three numeric fields.
   - On darwin, `rss_bytes` of the plain case is 0.8e9 ± 5%.
   - `maxrss_bytes(1024, "linux") == 1024²` and `maxrss_bytes(1024, "darwin") == 1024`.
5. **Wall time and results.**
   - `measure("import time; time.sleep(0.2)")` gives 0.2 ≤ `time_s` < 0.5.
   - `measure("print('noise')")` still returns all three fields.
   - `measure("raise ValueError('boom')")` raises `RuntimeError` whose message contains `boom`.
   - `measure("assert not jax.config.jax_enable_x64")` passes even with `JAX_ENABLE_X64=1` set in the parent (via monkeypatch).
   - `write_result("t", {"x": 1}, out_dir=tmp_path)` writes valid JSON with `x` and every `versions()` key.
   - `versions()` values: on darwin, P and E counts are positive ints with P + E = `os.cpu_count()`. `XLA_FLAGS` is recorded verbatim under monkeypatch. `jax_num_cpu_devices` is `True` on 0.11.2.
   - `versions()` with `sysctl` failing (monkeypatched `subprocess.run` raising, and returning nonzero) gives P/E = `None`, and `triad_threads(versions)` then returns `[os.cpu_count()]` (otherwise `[P, P + E]`).
   - `importlib.metadata.version("jaxmodal") == jaxmodal.__version__`.
   - `c_triad()` with `CC=/nonexistent` gives `"skipped: no C compiler"`, and with `CC=false` gives `"skipped: compile failed"`.
6. **Persistent cache.**
   - Each subprocess gets an env without `JAX_COMPILATION_CACHE_DIR`, calls `config.enable_cache(tmp)`, counts `/jax/compilation_cache/cache_hits` events with `jax.monitoring.register_event_listener`, then jits and runs a small function.
   - Run 1: 0 hits, ≥ 1 `cache_misses` event and at least one file written. Run 2 (same dir and function): ≥ 1 hit.
   - The roadmap allows skipping cache-dependent checks if the cache doesn't work. This plan deliberately fails instead, because the cache works on the pinned jax and a regression should be loud.
7. **Donation on CPU.**
   - Inside `warnings.catch_warnings(record=True)` with `warnings.simplefilter("always")`, a fresh `jax.jit(lambda x: x.at[0].set(1.0), donate_argnums=0)` on a 1e6 fp64 array:
     - emits no warning containing "donated buffers were not usable";
     - leaves `x.is_deleted()` true;
     - gives `y[0] == 1`.
   - A donated `lax.dynamic_update_slice` into a (1e5, 16) fp64 buffer with `jnp.int32` start indices: the output's `unsafe_buffer_pointer()` equals the input's pointer read before the call.
   - Fallback per the roadmap: if no jax version donates on CPU, the test becomes `xfail` with a reason, and `docs/conventions.md` records that the estimators budget without donation. This is not needed on 0.11.2.
8. **System info.** Marked `slow`.
   - Runs `python benchmarks/harness.py` with `--out` set to a temp dir.
   - All keys are present.
   - fp64 peak is in [10, 1e5] GFLOP/s and the jitted triad in [10, 2000] GB/s. This catches unit or flop-count errors of 10³.
   - The C triad is either a skip string or in [10, 2000] GB/s.

## Docs

- **`docs/conventions.md`:**
  - Principle 7 in full:
    - VTK node ordering, converted at import;
    - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear;
    - explicit x64 and no import side effects;
    - int32 indices through `config.index` (an unchecked cast);
    - static trip counts in differentiated loops;
    - iteration vectors, returned modes, Gram matrices and Rayleigh–Ritz in fp64, with fp32 only inside preconditioners and the device deflation basis;
    - returned locked modes from an fp64 host copy;
    - m as the LOBPCG block size.
  - Principle 1: relative tolerances $\|a-b\|_\infty \le \mathrm{tol}\,\|b\|_\infty$.
  - Principle 6:
    - `jnp.int32` start indices for `dynamic_slice`/`dynamic_update_slice`;
    - capacity padding so neither ever clamps.
  - Measured facts: jax 0.11.2, cache works (delete `.jax_cache/` after jax upgrades; if a later jax stops storing entries, test 6 fails and the remedy is removing `enable_cache` from conftest), donation works, `jax_num_cpu_devices` available (a jax bump re-checks this test too).
  - Harness rules:
    - peak definition per platform;
    - `code` must block;
    - `MallocLargeCache=0`, settling (`gc.collect()` plus footprint poll) and the interval reset, with the measured failure each prevents; the small-block and compression caveats;
    - warm up in `setup`;
    - 24 B per triad element.
- **`README.md`:**
  - goal;
  - status;
  - install (`uv sync`);
  - usage note (`import jaxmodal; jaxmodal.config.enable_x64()`, since import has no side effects);
  - `scripts/check.sh` and the merge gate `uv run pytest -m "not gpu"`;
  - license;
  - a link to `docs/ROADMAP.md`.
- **`docs/ROADMAP.md`:** the Development environment line on `ri_lifetime_max_phys_footprint` becomes the interval peak after `proc_reset_footprint_interval` (a private libSystem symbol) with `MallocLargeCache=0` (undocumented), recorded for every macOS measurement as the primary number, the RSS delta being secondary. The PR 0 Tests line and the Compilation cache bullet say the cache works on jax 0.11.2 and test 6 fails rather than skips if a later jax stops storing entries; the remedy is removing the `enable_cache` call in conftest. The Dependencies bullet on pymetis says it is added in PR 19, minimum 2025.1.1. The Deferred GPU checks list gains "PR 24: harness Linux and GPU peak paths, checked with the test 4 cases". The status notes that PR 0 is merged, with the pinned jax 0.11.2; that the cache and donation work; and that the pymetis extra is deferred to PR 19 (minimum 2025.1.1).

## scripts/check.sh

```sh
#!/usr/bin/env sh
set -e
cd "$(dirname "$0")/.."
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -m "not slow and not gpu" -q
```

## Acceptance

- `scripts/check.sh` passes in under 5 minutes.
- `uv run pytest -m "not gpu"` passes.
- The code review rounds per CLAUDE.md come back clean.
- Merge with `git merge --no-ff`.
