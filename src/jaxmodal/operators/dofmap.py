"""Global DOF numbering and the int32 overflow assert (host NumPy). See docs/operators.md."""

import numpy as np


def check_int32(**sizes):
    """Raise ValueError naming every size >= 2**31 (sizes are plain ints, nothing allocated)."""
    bad = [f"{k} = {v}" for k, v in sizes.items() if v >= 2**31]
    if bad:
        raise ValueError(f"int32 index overflow (>= 2**31): {', '.join(bad)}")


def element_dofs(mesh):
    """type -> (n_el_type, 3n) int32 global DOFs 3 conn[:, a] + i, node-major as in PR 3."""
    sizes = {f"{t}_dofs": 3 * c.size for t, c in mesh.blocks.items()}
    check_int32(n_dof=3 * len(mesh.points), **sizes)
    return {
        t: (3 * c[:, :, None] + np.arange(3)).reshape(len(c), -1).astype(np.int32)
        for t, c in mesh.blocks.items()
    }
