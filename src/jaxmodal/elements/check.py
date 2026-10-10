"""Host check of element validity: det J at quadrature points and lumped-mass signs (NumPy)."""

import numpy as np

from .registry import ELEMENTS, _det3


def check_elements(mesh, mass="consistent", chunk=65536):
    """Raise ValueError for the first element with det J <= 0 at a quadrature point, or (with
    mass="lumped") a row-sum lumped nodal mass <= 0. HRZ masses are positive when det J > 0."""
    if mass not in ("consistent", "lumped"):
        raise ValueError(f"mass must be 'consistent' or 'lumped', got {mass!r}")
    for t, cells in mesh.blocks.items():
        el = ELEMENTS[t]
        for s in range(0, len(cells), chunk):
            X = mesh.points[cells[s : s + chunk]]
            det = _det3(np.einsum("eai,qaj->eqij", X, el.dN, optimize=True))
            bad = (det <= 0).any(1)
            if bad.any():
                e = int(np.argmax(bad))
                raise ValueError(
                    f"element {mesh.offsets[t] + s + e} ({t}): det J = {det[e].min():.3g} <= 0 "
                    "at a quadrature point"
                )
            if mass == "lumped" and el.lumping == "rowsum":
                m = (det * el.weights) @ el.N
                bad = (m <= 0).any(1)
                if bad.any():
                    e = int(np.argmax(bad))
                    raise ValueError(
                        f"element {mesh.offsets[t] + s + e} ({t}): row-sum lumped nodal mass "
                        f"{m[e].min():.3g} <= 0; use mass='consistent' or a less distorted mesh"
                    )
