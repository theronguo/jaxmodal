"""Device memory estimators (bytes). See docs/operators.md."""

from .elements import ELEMENTS
from .operators.matfree import layout

# XLA temporaries of the build kernel per element (memory_analysis, exactly linear in chunk).
BUILD_BYTES = {
    "stored": {"hex8": 29.95e3, "hex20": 241.06e3, "hex27": 322.70e3},
    "otf": {"hex8": 1.22e3, "hex20": 4.10e3, "hex27": 4.10e3},
}
OVERHEAD = 10e6  # fixed JAX runtime overhead beyond the formulas, measured (hex20 30^3, hex8 60^3)


# otf apply_K temporaries per element: a + b (m - 1) bytes, an upper envelope of memory_analysis
# at m = 1, 4, 16 (XLA's fusion of the strain/stress fields varies with m).
OTF_BYTES = {"hex8": (6.1e3, 1.28e3), "hex20": (41.0e3, 3.17e3), "hex27": (54.6e3, 2.55e3)}


def _chunk_temps(mode, t, c, m):
    """XLA temporaries of one chunk: apply_K (stored: u_e, y_e, unpacked K_e; otf: fitted) and,
    in otf mode, diag_K, which forms B like the stored build."""
    k = 3 * ELEMENTS[t].N.shape[1]
    if mode == "stored":
        return 8 * c * (2 * k * m + k * k)
    a, b = OTF_BYTES[t]
    return c * max(a + b * (m - 1), BUILD_BYTES["stored"][t])


def unstructured_memory(
    mesh, m, mode="stored", mass="consistent", path=None, chunk=None, range_nodes=None
):
    """Estimated peak bytes of unstructured(...) plus apply_K and apply_M on (n_dof, m) blocks.

    Returns storage, vectors, workspace, build, overhead and
    total = overhead + storage + max(build, vectors + workspace). Later estimators add
    storage + workspace and count overhead once.
    """
    lay = layout(mesh, path, chunk, range_nodes)
    n_dof, n_el = 3 * lay["n_nodes"], mesh.n_el
    gather = lay["path"] == "gather"
    storage = 8 * (n_dof + 2 * n_el)  # nodal, s_K, s_M
    host = 8 * n_el  # phase index
    kernel, temps, window = 0, 0, 0
    for b in lay["blocks"]:
        n, k, c, n_pad = b["k"] // 3, b["k"], b["chunk"], b["n_pad"]
        nq = len(ELEMENTS[b["type"]].weights)
        per_el = k * (k + 1) // 2 + nq if mode == "stored" else 10 * nq
        storage += n_pad * (4 * k + 4 + 8 * per_el)  # dofs, phase, Ke + wdet or geo
        host += 8 * n_pad * (n + 1)  # padded connectivity and phase, alive during the build
        kernel = max(kernel, c * BUILD_BYTES[mode][b["type"]])
        temps = max(temps, _chunk_temps(mode, b["type"], c, m))
        if gather:
            storage += 4 * lay["n_ranges"]
            window += 8 * b["W"] * k * m
    if gather:
        inv = lay["inv"].size
        storage += 4 * inv
        host += 8 * inv  # int64 inverse map, alive until the int32 copy
        workspace = window + max(8 * 3 * lay["R"] * lay["valence"] * m, temps)
    else:
        workspace = temps
    out = dict(
        storage=storage,
        vectors=3 * 8 * n_dof * m,  # v and two outputs (XLA frees the last one asynchronously)
        workspace=workspace,
        build=host + kernel,
        overhead=OVERHEAD,
    )
    out["total"] = out["overhead"] + storage + max(out["build"], out["vectors"] + workspace)
    return {k: int(v) for k, v in out.items()}
