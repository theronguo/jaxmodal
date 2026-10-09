import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jaxmodal import config


def test_fp32_stays_fp32_under_x64():
    assert jnp.ones(3).dtype == jnp.float64  # x64 is on
    x = jnp.asarray(np.random.default_rng(0).random((4, 4)), jnp.float32)
    for f in (lambda x: 2.0 * x + jnp.sum(x), lambda x: x @ x.T, jnp.linalg.norm):
        assert jax.jit(f)(x).dtype == jnp.float32


IMPORT_ALL = """
import importlib, pkgutil, jax, jaxmodal
before = (jax.config.jax_enable_x64, jax.config.jax_compilation_cache_dir)
for m in pkgutil.walk_packages(jaxmodal.__path__, "jaxmodal."):
    importlib.import_module(m.name)
assert (jax.config.jax_enable_x64, jax.config.jax_compilation_cache_dir) == before
"""


def test_import_has_no_side_effects():
    for value in ("0", "1"):
        env = {**os.environ, "JAX_ENABLE_X64": value}
        env.pop("JAX_COMPILATION_CACHE_DIR", None)
        p = subprocess.run([sys.executable, "-c", IMPORT_ALL], env=env, capture_output=True)
        assert p.returncode == 0, p.stderr.decode()


def test_index_casts_to_int32():
    a = jnp.asarray([3.0, 0.0, 1.0, 0.0, 2.0])
    for raw in (jnp.argsort(a), jnp.nonzero(a)[0], jnp.arange(5)):
        assert raw.dtype == jnp.int64
        idx = config.index(raw)
        assert idx.dtype == jnp.int32
        np.testing.assert_array_equal(idx, raw)


CACHE_RUN = """
import sys, jax, jax.numpy as jnp
from jaxmodal import config
events = []
jax.monitoring.register_event_listener(lambda name, **kw: events.append(name))
config.enable_cache(sys.argv[1])
jax.jit(lambda x: jnp.sin(x) * 2)(jnp.ones(8)).block_until_ready()
print(events.count("/jax/compilation_cache/cache_hits"),
      events.count("/jax/compilation_cache/cache_misses"))
"""


def test_persistent_cache(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "JAX_COMPILATION_CACHE_DIR"}

    def run():
        p = subprocess.run(
            [sys.executable, "-c", CACHE_RUN, str(tmp_path)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
        return [int(n) for n in p.stdout.split()]

    hits, misses = run()
    assert hits == 0 and misses >= 1 and any(tmp_path.iterdir())
    hits, _ = run()
    assert hits >= 1


def test_donation_on_cpu():
    x = jnp.zeros(1_000_000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = jax.jit(lambda x: x.at[0].set(1.0), donate_argnums=0)(x)
        y.block_until_ready()
    assert not any("donated buffers were not usable" in str(m.message) for m in w)
    assert x.is_deleted()
    assert y[0] == 1

    buf = jnp.zeros((100_000, 16))
    ptr = buf.unsafe_buffer_pointer()
    upd = jax.jit(
        lambda b, u, i: lax.dynamic_update_slice(b, u, (i, jnp.int32(0))), donate_argnums=0
    )
    out = upd(buf, jnp.ones((10, 16)), jnp.int32(5))
    assert out.unsafe_buffer_pointer() == ptr
