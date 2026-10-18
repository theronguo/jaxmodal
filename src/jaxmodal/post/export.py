"""VTU export of mode shapes and element data. See docs/postprocessing.md."""

import numpy as np

from ..mesh.io import write_vtu
from .modal import _cols


def write_modes(path, mesh, V, cell_data=None):
    """Write a .vtu with point data mode_000, mode_001, ... = V[:, i] as (n_nodes, 3) and
    cell_data {name: (n_el,) or (n_el, c), element order}, plus material/orientation. Data is
    written unchanged; VTK reads 6 components as tensors XX, YY, ZZ, XY, YZ, XZ with tensor
    shear, so pass Voigt (11, 22, 33, 23, 13, 12) arrays as a[..., [0, 1, 2, 5, 3, 4]], strains
    with the shear halved first (docs/postprocessing.md)."""
    n_nodes = len(mesh.points)
    V = _cols(V, 3 * n_nodes)
    n = V.shape[1]
    w = max(3, len(str(n - 1)))
    point_data = {f"mode_{i:0{w}d}": V[:, i].reshape(n_nodes, 3) for i in range(n)}
    cell_data = dict(cell_data or {})
    for name, a in cell_data.items():
        a = np.asarray(a)
        if not isinstance(name, str) or set(name) & set("<>&\"'"):
            raise ValueError(f"cell data name must be a str without <>&\"', got {name!r}")
        if name in ("material", "orientation"):
            raise ValueError(f"cell data name {name!r} is reserved")
        if a.ndim not in (1, 2) or len(a) != mesh.n_el or a.dtype.kind not in "iuf":
            raise ValueError(
                f"cell data {name!r} must be a numeric ({mesh.n_el},) or ({mesh.n_el}, c) array, "
                f"got {a.dtype} {a.shape}"
            )
        cell_data[name] = a
    write_vtu(path, mesh, point_data, cell_data)
