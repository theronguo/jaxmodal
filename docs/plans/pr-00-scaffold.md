# PR 0 plan: project scaffold

Branch `pr-00-scaffold`. The scope is exactly the roadmap's PR 0 (Contents, Docs, Tests) plus Principle 7 (conventions). No FEM code. Review round 1 is incorporated.

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
.gitignore               .venv, __pycache__, .pytest_cache, .ruff_cache, .jax_cache/, dist/, *.egg-info/, .DS_Store
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
def enable_cache(path):  # jax_compilation_cache_dir=path, min_compile_time_secs=0, min_entry_size_bytes=-1
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

1. Runs `sys.executable -c <child>` with `check=True`. The child's stderr goes into the raised error.
2. The child returns its results as JSON written to a temp file whose path is passed in argv, so prints in `code` can't corrupt the result.
3. The child runs a fixed preamble:
   - `import numpy as np, jax, jax.numpy as jnp`;
   - x64 is off unless `setup` enables it.
4. The child then runs `setup`, takes the baseline, runs and times `code`, and reads the peak.
   - **macOS peak:**
     - after `setup`, call `proc_reset_footprint_interval(getpid())` from libSystem and record `ri_phys_footprint` as the baseline;
     - after `code`, `peak_bytes = ri_interval_max_phys_footprint − baseline`.
     - Fields are read from `proc_pid_rusage(pid, RUSAGE_INFO_V4)` as a 16-byte uuid followed by `c_uint64 * 35`. The field indices are named constants checked by test 4.
     - Because the interval is reset, a large transient in `setup` neither hides nor inflates the delta.
     - It counts compressed pages, as the roadmap's 10M runs need.
   - **Linux peak:**
     - write `5` to `/proc/self/clear_refs` after `setup`;
     - `peak_bytes = VmHWM − VmRSS(after setup)`.
     - This path is not exercised on the Mac. PR 24 runs it.
   - **`rss_bytes`:**
     - the plain `ru_maxrss` delta, recorded as the roadmap asks;
     - converted to bytes by `maxrss_bytes(value, platform)` (bytes on darwin, KiB elsewhere).
5. Two documented rules:
   - `code` must block on its results (`.block_until_ready()`), or `time_s` measures only dispatch.
   - Memory freed in `setup` can be reused by `code` without RSS growing, so setups should avoid large transients they don't need.

### `versions() -> dict` (cheap)

- jax/jaxlib/numpy versions, `platform.platform()` and `os.cpu_count()`;
- P/E core counts from `sysctl hw.perflevel0/1.physicalcpu`, or `None` when unavailable (Linux);
- the XLA:CPU intra-op pool size: the `XLA_FLAGS` value if pinned, otherwise `os.cpu_count()` (XLA's default);
- whether `jax_num_cpu_devices` exists.

### `system_info() -> dict`

- Calls `config.enable_x64()` and asserts that its arrays are float64.
- fp64 peak: jitted 4096² matmul, arrays passed as arguments, warm-up first, `block_until_ready` in each timed run, best of 5, 2n³ flops.
- Jitted STREAM triad: `a = b + s·c` on 2·10⁷ fp64 elements, same timing rules, best of 10, 24 B per element (STREAM convention; write-allocate traffic is not counted).
- C triad:
  - `benchmarks/triad.c`, built with `${CC:-cc} -O3 -pthread` into a temp dir;
  - pthreads with first-touch initialisation inside the threads, best of 20, and the result is used so the loop can't be optimised away;
  - run with P-core and P+E-core thread counts, both recorded, best reported;
  - a missing compiler gives `"skipped: no C compiler"`, a failed build `"skipped: compile failed"`.
- Includes `versions()`.

### Results

- `write_result(name, data, out_dir="benchmarks/results")` writes `<out_dir>/<name>.json` with `data` plus `versions()`.
- The peaks live only in `system.json`. Later results reference it instead of re-measuring.
- `python benchmarks/harness.py` writes `system.json`.

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
   - Over 3 runs, max/min − 1 < 5%.
   - Control: `code="pass"` gives `peak_bytes` < 20 MB.
   - Transient case: `setup` touches 2 GB in an mmap and closes it, then the same `code` gives 0.8e9 ± 5%. Without the interval reset this reads ≈ 0.
   - On darwin, `rss_bytes` of the plain case is 0.8e9 ± 5%.
   - `maxrss_bytes(1024, "linux") == 1024²` and `maxrss_bytes(1024, "darwin") == 1024`.
5. **Wall time and results.**
   - `measure("import time; time.sleep(0.2)")` gives `time_s` = 0.2 ± 0.05.
   - `write_result("t", {"x": 1}, out_dir=tmp_path)` writes valid JSON with `x` and every `versions()` key.
6. **Persistent cache.**
   - A subprocess calls `config.enable_cache(tmp)`, then jits and runs a small function: at least one entry is written.
   - A second subprocess with the same dir and function adds no new entries, so the cache is reused.
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
   - The C triad is either the skip string or within 3× of the jitted triad.

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
  - Measured facts: jax 0.11.2, cache works (delete `.jax_cache/` after jax upgrades), donation works, `jax_num_cpu_devices` available.
  - Harness rules:
    - peak definition per platform;
    - `code` must block;
    - allocator-reuse caveat;
    - 24 B per triad element.
- **`README.md`:**
  - goal;
  - status;
  - install (`uv sync`);
  - usage note (`import jaxmodal; jaxmodal.config.enable_x64()`, since import has no side effects);
  - `scripts/check.sh` and the merge gate `uv run pytest -m "not gpu"`;
  - license;
  - a link to `docs/ROADMAP.md`.
- **`docs/ROADMAP.md`:** status notes that PR 0 is merged, with the pinned jax 0.11.2; that the cache and donation work; and that the pymetis extra is deferred to PR 19 (minimum 2025.1.1).

## scripts/check.sh

```sh
#!/usr/bin/env sh
set -e
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not slow and not gpu" -q
```

## Acceptance

- `scripts/check.sh` passes in under 5 minutes.
- `uv run pytest -m "not gpu"` passes.
- The code review rounds per CLAUDE.md come back clean.
- Merge with `git merge --no-ff`.
