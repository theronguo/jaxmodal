"""Mesh data model (host-side NumPy) and corner Jacobians. See docs/mesh.md."""

import numbers
from dataclasses import dataclass

import numpy as np

TYPES = ("hex8", "hex20", "hex27")  # canonical block order
NODES = {"hex8": 8, "hex20": 20, "hex27": 27}

# Reference hex [-1, 1]^3 in VTK order: corners, edge midpoints, face centres, centre.
CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    float,
)  # fmt: skip
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))  # fmt: skip
# Corners of the faces -xi, +xi, -eta, +eta, -zeta, +zeta, counter-clockwise seen from outside.
FACE_CORNERS = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))
REFERENCE_NODES = np.vstack(
    [CORNERS]
    + [[(CORNERS[a] + CORNERS[b]) / 2 for a, b in EDGES]]
    + [[CORNERS[list(f)].mean(0) for f in FACE_CORNERS]]
    + [[[0.0, 0.0, 0.0]]]
)  # (27, 3)

# Corner pairs (minus, plus) along each reference axis, VTK hex order.
PAIRS = (
    ((0, 1), (3, 2), (4, 5), (7, 6)),
    ((0, 3), (1, 2), (4, 7), (5, 6)),
    ((0, 4), (1, 5), (2, 6), (3, 7)),
)


def _check_array(name, a, dtype, ndim):
    if not isinstance(a, np.ndarray) or a.dtype != dtype or a.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D {np.dtype(dtype)} np.ndarray")


def check_grid(shape, spacing, origin):
    """shape: 3 positive ints; spacing, origin: 3 finite reals, spacing > 0 (bools rejected)."""

    def ok(x, kind):
        return np.shape(x) == (3,) and all(
            isinstance(a, kind) and not isinstance(a, bool) and np.isfinite(a) for a in x
        )

    if not ok(shape, numbers.Integral) or min(shape) <= 0:
        raise ValueError(f"shape must be 3 positive ints, got {shape!r}")
    if not ok(spacing, numbers.Real) or not ok(origin, numbers.Real) or min(spacing) <= 0:
        raise ValueError("spacing/origin must be 3 finite reals, spacing > 0")


def face_sets(d, idx, top):
    """Node sets "<axis>min", "<axis>max" of axis d: nodes whose integer grid index is 0 or top."""
    a = "xyz"[d]
    return {f"{a}min": np.flatnonzero(idx == 0), f"{a}max": np.flatnonzero(idx == top)}


def _check_bijection(name, m, size, n):
    _check_array(name, m, np.int64, 1)
    if len(m) != size:
        raise ValueError(f"{name} has length {len(m)}, expected {size}")
    used = m[m >= 0]
    bad = (m < -1).any() or (m >= n).any() or len(used) != n
    if bad or np.bincount(used, minlength=n).max() > 1:
        raise ValueError(f"{name} is not a bijection onto the mesh")


@dataclass(frozen=True, eq=False)
class Voxel:
    shape: tuple  # cells per axis (nx, ny, nz)
    spacing: tuple
    origin: tuple
    node_map: np.ndarray  # grid node -> mesh node, -1 if dropped
    element_map: np.ndarray  # grid cell -> mesh element, -1 if inactive


@dataclass(frozen=True, eq=False)
class Mesh:
    points: np.ndarray  # (n_nodes, 3) float64
    blocks: dict  # type -> (n_el_type, n_per_el) int64, VTK order, keys in TYPES order
    node_sets: dict  # name -> sorted unique node indices
    element_sets: dict  # name -> sorted unique global element indices
    material: np.ndarray  # (n_el,) int64 label
    orientation: np.ndarray  # (n_el,) int64 label
    voxel: Voxel | None = None

    @property
    def offsets(self):
        """Start of each block in the global element index."""
        starts = np.cumsum([0] + [len(c) for c in self.blocks.values()])
        return {t: int(s) for t, s in zip(self.blocks, starts[:-1], strict=True)}

    @property
    def n_el(self):
        return sum(len(c) for c in self.blocks.values())

    def __post_init__(self):
        _check_array("points", self.points, np.float64, 2)
        if self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError("points must be (n_nodes, 3) and finite")
        n = len(self.points)
        if not isinstance(self.blocks, dict) or not self.blocks:
            raise ValueError("mesh has no elements")
        keys = list(self.blocks)
        if not set(keys) <= set(TYPES) or keys != sorted(keys, key=TYPES.index):
            raise ValueError(f"block keys {keys} must be in TYPES order {TYPES}")
        seen = np.zeros(n, bool)
        for t, c in self.blocks.items():
            _check_array(f"blocks[{t!r}]", c, np.int64, 2)
            if c.shape[1] != NODES[t] or len(c) == 0:
                raise ValueError(f"blocks[{t!r}] must have shape (n > 0, {NODES[t]})")
            if c.min() < 0 or c.max() >= n:
                raise ValueError(f"blocks[{t!r}] has node indices outside [0, {n})")
            seen[c] = True
        if not seen.all():
            raise ValueError(f"{(~seen).sum()} orphan nodes (unused by any element)")
        all_sets = (("node", self.node_sets, n), ("element", self.element_sets, self.n_el))
        for kind, sets, size in all_sets:
            if not isinstance(sets, dict):
                raise ValueError(f"{kind}_sets must be a dict")
            for k, s in sets.items():
                if not isinstance(k, str):
                    raise ValueError(f"{kind} set key {k!r} must be a str")
                _check_array(f"{kind} set {k!r}", s, np.int64, 1)
                if len(s) and (s[0] < 0 or s[-1] >= size or (np.diff(s) <= 0).any()):
                    raise ValueError(f"{kind} set {k!r} must be sorted, unique, in [0, {size})")
        for name in ("material", "orientation"):
            a = getattr(self, name)
            _check_array(name, a, np.int64, 1)
            if len(a) != self.n_el:
                raise ValueError(f"{name} has length {len(a)}, expected n_el = {self.n_el}")
        if self.voxel is not None:
            self._check_voxel()

    def _check_voxel(self):
        v = self.voxel
        if not isinstance(v, Voxel):
            raise ValueError("voxel must be a Voxel")
        if list(self.blocks) != ["hex8"]:
            raise ValueError("a voxel mesh has exactly one hex8 block")
        check_grid(v.shape, v.spacing, v.origin)
        nx, ny, nz = v.shape
        _check_bijection("node_map", v.node_map, (nx + 1) * (ny + 1) * (nz + 1), len(self.points))
        _check_bijection("element_map", v.element_map, nx * ny * nz, self.n_el)


def corner_jacobians(mesh, chunk=1 << 20):
    """Trilinear Jacobian determinant at the 8 corners of every element: type -> (n_el_type, 8).

    Column d of J at corner a is (x[a_d+] - x[a_d-]) / 2. For hex20/hex27 this uses the corner
    hex only (a sign/orientation check). Rows go in chunks to bound temporaries.
    """
    pairs = [[next(p for p in axis if a in p) for axis in PAIRS] for a in range(8)]
    out = {}
    for t, c in mesh.blocks.items():
        det = np.empty((len(c), 8))
        for s in range(0, len(c), chunk):
            x = [mesh.points[c[s : s + chunk, :8], i] for i in range(3)]  # 3 x (rows, 8)
            for a, axes in enumerate(pairs):
                (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = (
                    [(xi[:, hi] - xi[:, lo]) / 2 for xi in x] for lo, hi in axes
                )
                det[s : s + chunk, a] = (
                    a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)
                )
        out[t] = det
    return out
