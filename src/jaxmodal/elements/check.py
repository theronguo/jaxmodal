"""Host check of element validity: det J at quadrature points and lumped-mass signs (NumPy)."""

import numpy as np

from .registry import ELEMENTS, _det3

# ssh8: det J is quadratic in zeta at fixed (xi, eta); sampled at zeta = -1, 0, 1 on the 2x2 rule.
_G = 1 / np.sqrt(3)
_SHELL = np.array([[x, y, z] for z in (-1, 0, 1) for y in (-_G, _G) for x in (-_G, _G)])


def _thickness_min(det):
    """Minimum over zeta in [-1, 1] of the quadratic through det (e, 3, 4) at zeta = -1, 0, 1."""
    dm, d0, dp = det[:, 0], det[:, 1], det[:, 2]
    a, b = (dp + dm) / 2 - d0, (dp - dm) / 2
    inside = (a > 0) & (np.abs(b) < 2 * a)  # vertex -b / 2a inside (-1, 1)
    vertex = np.where(inside, d0 - b**2 / (4 * np.where(inside, a, 1)), np.inf)
    return np.minimum(np.minimum(dm, dp), vertex)


def check_elements(mesh, mass="consistent", chunk=65536):
    """Raise ValueError for the first element with det J <= 0 at a quadrature point (for ssh8:
    anywhere on the in-plane points' thickness lines, where ply points lie), or (with
    mass="lumped") a row-sum lumped nodal mass <= 0. HRZ masses are positive when det J > 0."""
    if mass not in ("consistent", "lumped"):
        raise ValueError(f"mass must be 'consistent' or 'lumped', got {mass!r}")
    for t, cells in mesh.blocks.items():
        el = ELEMENTS[t]
        if mass == "lumped" and el.lumping is None:
            raise ValueError("solid-shell elements are consistent-mass only")
        dN = el.grad(_SHELL) if t == "ssh8" else el.dN
        for s in range(0, len(cells), chunk):
            X = mesh.points[cells[s : s + chunk]]
            det = _det3(np.einsum("eai,qaj->eqij", X, dN, optimize=True))
            if t == "ssh8":
                det = _thickness_min(det.reshape(-1, 3, 4))
            tests = [(det, "det J = {:.3g} <= 0 at a quadrature point")]
            if mass == "lumped" and el.lumping == "rowsum":
                msg = "row-sum lumped nodal mass {:.3g} <= 0; use mass='consistent'"
                tests.append(((det * el.weights) @ el.N, msg))
            for values, msg in tests:  # det J first: masses of inverted elements mean nothing
                bad = (values <= 0).any(1)
                if bad.any():
                    e = int(np.argmax(bad))
                    where = f"element {mesh.offsets[t] + s + e} ({t}): "
                    raise ValueError(where + msg.format(values[e].min()))
