"""Helpers shared by the matrix-free operators and the assembled oracle. See docs/operators.md."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def register(cls, meta):
    """Register a frozen dataclass as a pytree; `meta` fields are static, the rest are leaves."""
    data = [f.name for f in dataclasses.fields(cls) if f.name not in meta]
    jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)


def check_x64(name):
    if not jax.config.jax_enable_x64:
        raise ValueError(f"{name} needs float64: call jaxmodal.config.enable_x64() first")


def check_nodal_mass(nodal_mass, n_nodes):
    """(n_nodes,) finite, non-negative point masses as a float array; None gives zeros."""
    if nodal_mass is None:
        return np.zeros(n_nodes)
    nodal_mass = np.asarray(nodal_mass, float)
    if nodal_mass.shape != (n_nodes,) or not (np.isfinite(nodal_mass) & (nodal_mass >= 0)).all():
        raise ValueError(f"nodal_mass must be a finite, non-negative ({n_nodes},) array")
    return nodal_mass


def check_v(v, n_dof):
    if v.ndim != 2 or v.shape[0] != n_dof:
        raise ValueError(f"v must have shape ({n_dof}, m), got {v.shape}")


def scaled(op, s_K=None, s_M=None):
    """Copy of op with new per-element scalings (n_el,); None keeps the current one. Traceable."""
    new = {}
    for name, s in (("s_K", s_K), ("s_M", s_M)):
        if s is not None:
            if jnp.shape(s) != (op.n_el,):
                raise ValueError(f"{name} must have shape ({op.n_el},), got {jnp.shape(s)}")
            new[name] = jnp.asarray(s, jnp.float64)
    return dataclasses.replace(op, **new)
