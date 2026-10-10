"""Dirichlet BC masks, the masked eigen-operator and A_P (SciPy). See docs/operators.md."""

import numpy as np
import scipy.sparse as sp


def dof_mask(mesh, bcs):
    """(n_dof,) bool, True = free. bcs maps a node-set name to constrained components, e.g.
    {"xmin": "xyz", "ymin": "y"}: zero displacement in global axes; {} is free-free."""
    free = np.ones((len(mesh.points), 3), bool)
    for name, comps in bcs.items():
        if name not in mesh.node_sets:
            raise ValueError(f"unknown node set {name!r}")
        ok = isinstance(comps, str) and comps and set(comps) <= set("xyz")
        if not ok or len(set(comps)) != len(comps):
            raise ValueError(f"bcs[{name!r}] must be distinct letters from 'xyz', got {comps!r}")
        free[np.ix_(mesh.node_sets[name], ["xyz".index(c) for c in comps])] = False
    return free.ravel()


def _check_mask(A, mask):
    if not (isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (A.shape[0],)):
        raise ValueError(f"mask must be a bool ({A.shape[0]},) np.ndarray")


def masked(A, mask):
    """D A D with D = diag(mask): constrained rows and columns zero, free block A_ff."""
    _check_mask(A, mask)
    D = sp.diags_array(mask.astype(float))
    return (D @ A @ D).tocsr()


def a_p(A, mask, s=None):
    """A_P = D A D + s (I - D); default s = mean of diag(A) over free DOFs (for SPD A)."""
    _check_mask(A, mask)
    if not mask.any():
        raise ValueError("mask has no free DOF")
    if s is None:
        s = A.diagonal()[mask].mean()
        if s <= 0:
            raise ValueError(f"mean free diagonal s = {s:.3g} <= 0; pass s for indefinite A")
    return (masked(A, mask) + sp.diags_array(s * ~mask)).tocsr()
