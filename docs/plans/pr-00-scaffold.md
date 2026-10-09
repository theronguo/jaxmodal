# PR 0 plan: project scaffold

Branch `pr-00-scaffold`. Scope is exactly the roadmap's PR 0 (Contents, Docs, Tests) plus Principle 7 (conventions). No FEM code.

## Files

```
pyproject.toml          hatchling build, src layout, ruff + pytest config
uv.lock                 from `uv lock`
.python-version         3.12
.gitignore              .venv, __pycache__, .pytest_cache, .ruff_cache, benchmarks/meshes/, .jax_cache/
LICENSE                 MIT, "Copyright (c) 2026 jaxmodal authors"
README.md
scripts/check.sh
src/jaxmodal/__init__.py    __version__ = "0.1.0"; imports nothing from jax.config
src/jaxmodal/config.py
benchmarks/harness.py
benchmarks/triad.c
benchmarks/results/.gitkeep
docs/conventions.md
tests/conftest.py
tests/test_config.py
tests/test_harness.py
```

## pyproject.toml

- `requires-python = "==3.12.*"`. uv installs 3.12 (the system Python is 3.11).
- Dependencies: `jax>=J`, `numpy`, `scipy`, `meshio`, `pyamg`. J is the jax version that `uv lock` resolves today, provided the donation test passes on it. Otherwise J is raised (Tests, item 5).
- Optional extra `partition = ["pymetis>=P"]`, where P is the oldest version with macOS arm64 wheels, checked with `uv pip install --dry-run`. Nothing imports it yet.
- Dev dependency group: `pytest`, `ruff`.
- `[tool.pytest.ini_options]`:
  - `testpaths = ["tests"]`;
  - `markers = ["slow: long-running", "gpu: needs a CUDA device"]`;
  - `addopts = "--strict-markers"`.
- `[tool.ruff]`: line length 100, rules `E, F, I, UP, B`, target py312.

## config.py

```python
def enable_x64(on=True): jax.config.update("jax_enable_x64", on)
def enable_cache(path): sets jax_compilation_cache_dir = path,
    jax_persistent_cache_min_compile_time_secs = 0,
    jax_persistent_cache_min_entry_size_bytes = -1
def index(x): return jnp.asarray(x, dtype=jnp.int32)   # cast for argsort/nonzero/arange results
```

- Nothing runs at import.
- `index` is the single cast point that later PRs use for every index array (Principle 6).
- The int32 overflow asserts belong to PR 4, where the sizes first appear.

## tests/conftest.py

- Calls `config.enable_x64()` once at session start.
- If the PR 0 cache probe (below) found that the cache stores entries, it also calls `config.enable_cache(<repo>/.jax_cache)`. The probe result is a module constant `CACHE_WORKS` in `conftest.py`, set from the measured outcome when this PR lands and recorded in `docs/conventions.md`.
- Adds `collect_ignore = ["multidevice"]` now, so later PRs only add files.

## Benchmark harness (`benchmarks/harness.py`)

- `measure(code: str, setup: str = "") -> dict`:
  - runs `python -c` in a fresh subprocess;
  - inside the child: execute `setup`, read `ru_maxrss` as the baseline, execute `code`, read `ru_maxrss` again;
  - returns peak minus baseline in bytes, plus the wall time of `code`.
  - `ru_maxrss` is in bytes on macOS and KiB on Linux, selected with `sys.platform`.
  - With `footprint=True` on macOS, it also reads `ri_lifetime_max_phys_footprint` through `ctypes` `proc_pid_rusage(RUSAGE_INFO_V4)`.
- `system_info() -> dict` collects:
  - jax and jaxlib versions;
  - the P/E core counts (`sysctl hw.perflevel0.physicalcpu`, `hw.perflevel1.physicalcpu`) and `os.cpu_count()`;
  - the XLA:CPU thread setting (`XLA_FLAGS` as given, or "default");
  - the measured fp64 peak, from a jitted 4096² fp64 matmul (best of 5, 2n³ flops);
  - the jitted STREAM triad, `a = b + s*c` on 2·10⁷ fp64 elements (best of 10, 24 bytes per element);
  - the C triad: `benchmarks/triad.c`, pthreads with one thread per core, compiled with `cc -O3` into a temp directory. If no `cc` is found it is recorded as `"skipped: no C compiler"`.
- `write_result(name, data)` writes `benchmarks/results/<name>.json` with `system_info()` merged in.
- `python benchmarks/harness.py` writes `system.json`. That is the PR 0 benchmark artifact, and it is committed.

## Tests

All tests are fast (not `slow`) unless marked otherwise.

1. **fp32 stays fp32 under x64** (`test_config.py`): with x64 enabled, a jitted function `f(x) = 2.0*x + jnp.sum(x)` on an fp32 array returns fp32. The same holds for `x @ x.T` and `jnp.linalg.norm`. If any of these upcasts, the test fails.
2. **No import side effects:** a subprocess runs `import jax; v0 = jax.config.jax_enable_x64; import jaxmodal; assert jax.config.jax_enable_x64 == v0`, once with `JAX_ENABLE_X64=0` and once with `JAX_ENABLE_X64=1`.
3. **int32 index casts:** with x64 on, `argsort`, `nonzero` and `arange` return int64. `config.index` of each returns int32 with the same values. The test also checks that the raw results really are int64 under x64, so the test would notice if JAX changed this.
4. **Harness RSS:** `measure("a = np.ones(100_000_000)")` (800 MB) reports 0.8e9 ± 5%. Over 3 runs, max/min − 1 < 5%. A 0-byte control reports < 20 MB. Marked `slow` because it takes 3 subprocesses with a jax import.
5. **Donation on CPU:**
   - `jax.jit(lambda x: x.at[0].set(1.0), donate_argnums=0)` on a 1e6 fp64 array, run under `warnings.catch_warnings(record=True)`;
   - asserts that no warning contains "donated buffers were not usable", and that `x.is_deleted()` is true;
   - also asserts that a donated `lax.dynamic_update_slice` into a (1e5, 16) fp64 buffer keeps `unsafe_buffer_pointer()` (the roadmap's in-place lock-insert pattern).
   - If this fails on J, J is raised until it passes.
6. **Persistent cache probe:**
   - a subprocess enables the cache into a temp dir, then jits and runs a small function;
   - the test asserts the outcome matches `CACHE_WORKS`: entries exist if `CACHE_WORKS` is true, and none if it is false;
   - a JAX upgrade that changes cache behaviour then fails loudly instead of silently changing test timing.
7. **System info:** `system_info()` has all keys. fp64 peak and triad bandwidth are > 0. The C triad is either a positive number or the exact skip string. Marked `slow`.

## Docs

- **`docs/conventions.md`:**
  - **Principle 7:**
    - VTK node ordering;
    - Voigt order `[11, 22, 33, 23, 13, 12]` with engineering shear;
    - explicit x64;
    - int32 indices through `config.index`;
    - static trip counts in differentiated loops;
    - fp64 iteration vectors, with fp32 only inside preconditioners and the deflation basis;
    - m as the LOBPCG block size.
  - **Principle 1:** relative tolerances ‖a − b‖_∞ ≤ tol·‖b‖_∞, written in LaTeX.
  - **Principle 6:** the capacity and clamping rules, as a pointer for later PRs.
  - The measured cache outcome and the jax version J.
- **`README.md`:**
  - one paragraph on the goal;
  - status ("PR 0: scaffold");
  - install (`uv sync`), checks (`scripts/check.sh`), the merge gate (`uv run pytest -m "not gpu"`);
  - license;
  - a link to `docs/ROADMAP.md`.
- **`docs/ROADMAP.md`:** status line notes PR 0 merged.

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
- Review rounds per CLAUDE.md.
- Merge with `git merge --no-ff`.
