"""Global DOF numbering and the int32 overflow assert (host NumPy). See docs/operators.md."""

import numpy as np


def check_int32(**sizes):
    """Raise ValueError naming every size >= 2**31 (sizes are plain ints, nothing allocated)."""
    bad = [f"{k} = {v}" for k, v in sizes.items() if v >= 2**31]
    if bad:
        raise ValueError(f"int32 index overflow (>= 2**31): {', '.join(bad)}")


def element_dofs(mesh):
    """type -> (n_el_type, 3n) int32 global DOFs 3 conn[:, a] + i (node-major, as the kernels)."""
    sizes = {f"{t}_dofs": 3 * c.size for t, c in mesh.blocks.items()}
    check_int32(n_dof=3 * len(mesh.points), **sizes)
    return {t: conn_dofs(c) for t, c in mesh.blocks.items()}


def conn_dofs(conn):
    """(n_el, n) connectivity -> (n_el, 3n) int32 DOFs 3 conn[:, a] + i."""
    return (3 * conn[:, :, None] + np.arange(3)).reshape(len(conn), -1).astype(np.int32)
