"""Explicit global JAX settings. Nothing here runs at import."""

import jax
import jax.numpy as jnp


def enable_x64():
    jax.config.update("jax_enable_x64", True)


def enable_cache(path):
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def index(x):
    """Cast an index array to int32. Unchecked: array values >= 2**31 wrap."""
    return jnp.asarray(x, dtype=jnp.int32)
