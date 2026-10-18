import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import jaxmodal
from benchmarks import harness

pytestmark = pytest.mark.serial  # measures process peaks and wall time
darwin = pytest.mark.skipif(sys.platform != "darwin", reason="peak_bytes is macOS-only in PR 0")
GB = 1e9
RNG = "rng = np.random.default_rng(0)"
PLAIN = "a = rng.random(100_000_000)"  # 0.8 GB, incompressible
MMAP = (  # touch and unmap a 2 GB transient
    "import mmap; n = 488_281 * 4096; m = mmap.mmap(-1, n); m[::4096] = b'x' * (n // 4096); "
    "m.close()"
)


def close(x, ref, tol=0.05):
    return abs(x - ref) <= tol * ref


@pytest.mark.slow
@darwin
def test_plain_peak_is_stable():
    runs = [harness.measure(PLAIN, RNG) for _ in range(3)]
    for key in ("peak_bytes", "rss_bytes"):
        vals = [r[key] for r in runs]
        assert all(close(v, 0.8 * GB) for v in vals), vals
        assert max(vals) / min(vals) - 1 < 0.05


@pytest.mark.slow
@darwin
@pytest.mark.parametrize(
    "setup",
    [
        # unmapped transient: without the interval reset this reads ~2.0e9
        MMAP,
        # malloc-freed transient: without MallocLargeCache=0 this reads ~0
        "t = rng.random(200_000_000); del t",
    ],
    ids=["mmap", "malloc"],
)
def test_setup_transients_dont_leak(setup):
    assert close(harness.measure(PLAIN, f"{RNG}; {setup}")["peak_bytes"], 0.8 * GB)


@pytest.mark.slow
@darwin
def test_peak_freed_below_high_water():
    # ru_maxrss reads 0 (setup's mmap set a higher mark); the current footprint reads 0 too.
    r = harness.measure(f"{PLAIN}; del a", f"{RNG}; {MMAP}")
    assert r["rss_bytes"] < 20e6
    assert close(r["peak_bytes"], 0.8 * GB)


def test_measure_json():
    r = harness.measure_json('open("{out}", "w").write(json.dumps(dict(a=1)))', "import json")
    assert r["a"] == 1
    assert isinstance(r["peak_bytes"], int) if sys.platform == "darwin" else r["peak_bytes"] is None


@pytest.mark.slow
@darwin
def test_peak_controls():
    assert harness.measure("pass")["peak_bytes"] < 20e6
    # namespace isolation: user names can't shadow harness locals
    r = harness.measure("time = None", setup="json = os = sys = None")
    assert all(isinstance(r[k], int | float) for k in ("peak_bytes", "rss_bytes", "time_s"))


@pytest.mark.slow
@darwin
def test_jax_cyclic_garbage():
    # Without the harness's gc.collect() the cycle is freed by the gc.collect() in `code`: ~0.
    setup = (
        f"{RNG}; x0 = jnp.asarray(rng.random(200_000_000, np.float32)).block_until_ready(); "
        "f = jax.jit(lambda x: x * 2 + 1); c = [f(x0).block_until_ready()]; c.append(c); del c"
    )
    code = "import gc; gc.collect(); y = f(x0).block_until_ready()"
    assert close(harness.measure(code, setup)["peak_bytes"], 0.8 * GB)


@pytest.mark.slow
@darwin
def test_small_block_reuse():
    # Footprint alone reads ~0.82e9 here; the max with ru_maxrss recovers the true peak.
    code = (
        "L = [rng.random(125_000) for _ in range(800)]; del L; "
        "L = [rng.random(125_000) for _ in range(1600)]"
    )
    assert close(harness.measure(code, RNG)["peak_bytes"], 1.6 * GB)


@pytest.mark.slow
@darwin
def test_xla_async_temporaries():
    # Without the settling poll the full warm-up reads ~2.1-2.4e9.
    g = "k = jax.random.key(0); g = jax.jit(lambda k: jax.random.uniform(k, (200_000_000,)))"
    code = "g(k).block_until_ready()"
    full = harness.measure(code, f"{g}; g(k).block_until_ready()")["peak_bytes"]
    ref = harness.measure(code, f"{g}; g.lower(k).compile()")["peak_bytes"]
    assert ref > 3 * GB
    assert close(full, ref)


def test_maxrss_bytes():
    assert harness.maxrss_bytes(1024, "linux") == 1024**2
    assert harness.maxrss_bytes(1024, "darwin") == 1024


def test_measure_time_and_errors(monkeypatch):
    assert 0.2 <= harness.measure("import time; time.sleep(0.2)")["time_s"] < 0.5
    assert set(harness.measure("print('noise')")) == {"peak_bytes", "rss_bytes", "time_s"}
    with pytest.raises(RuntimeError, match="boom"):
        harness.measure("raise ValueError('boom')")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/tmp/should-not-be-used")
    harness.measure(
        "assert not (jax.config.jax_enable_x64 or jax.config.jax_compilation_cache_dir)"
    )


def test_write_result(tmp_path):
    data = json.loads(harness.write_result("t", {"x": 1}, out_dir=tmp_path).read_text())
    assert data["x"] == 1 and set(data["versions"]) == set(harness.versions())


def test_versions(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_foo=1")
    v = harness.versions()
    assert v["xla_flags"] == "--xla_foo=1"
    assert v["jax_num_cpu_devices"] is True
    if sys.platform == "darwin":
        assert all(isinstance(v[k], int) and v[k] > 0 for k in ("p_cores", "e_cores"))
        assert v["p_cores"] + v["e_cores"] == os.cpu_count()
        assert harness.triad_threads(v) == [v["p_cores"], os.cpu_count()]


@pytest.mark.parametrize("mode", ["raise", "nonzero"])
def test_versions_without_sysctl(monkeypatch, mode):
    def fake_run(*args, **kwargs):
        if mode == "raise":
            raise FileNotFoundError
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    v = harness.versions()
    assert v["p_cores"] is None and v["e_cores"] is None
    assert harness.triad_threads(v) == [os.cpu_count()]


def test_version_metadata():
    assert importlib.metadata.version("jaxmodal") == jaxmodal.__version__


def test_c_triad_skips(monkeypatch):
    monkeypatch.setenv("CC", "/nonexistent")
    assert harness.c_triad() == "skipped: no C compiler"
    monkeypatch.setenv("CC", "false")
    assert harness.c_triad() == "skipped: compile failed"


@pytest.mark.slow
def test_system_info(tmp_path):
    script = Path(harness.__file__)
    subprocess.run([sys.executable, str(script), "--out", str(tmp_path)], check=True)
    data = json.loads((tmp_path / "system.json").read_text())
    assert set(data) == {"fp64_matmul_gflops", "triad_gbps", "c_triad", "c_triad_ratio", "versions"}
    assert 10 <= data["fp64_matmul_gflops"] <= 1e5
    assert 10 <= data["triad_gbps"] <= 2000
    ct = data["c_triad"]
    if isinstance(ct, str):
        assert ct.startswith("skipped:")
    else:
        assert 10 <= ct["gbps"] <= 2000
        assert data["c_triad_ratio"] == pytest.approx(ct["gbps"] / data["triad_gbps"])
