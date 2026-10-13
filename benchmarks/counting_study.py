"""Counting study (PR 14a): degree and cost of the stochastic count against mode index (hex20
16^3 with AMG, voxel 32^3 with GMG, 200 modes from solve_modes, steel, xmin clamped), the
propagation of inner-solve errors through the recurrence (reference box and a graded box with a
100x density contrast; W complete and one mode left out; AMG at lam_c against exact dense
solves), and the Jacobi PCG floor on the reference box. Writes results/counting_study.json.
"""

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from jax.tree_util import Partial

from benchmarks import harness
from jaxmodal import config
from jaxmodal.eigen import solve_modes
from jaxmodal.eigen.counting import count, count_degree
from jaxmodal.materials import Material, isotropic
from jaxmodal.mesh import box_mesh, voxel_mesh
from jaxmodal.operators import a_p, ap_operator, assemble, dof_mask, stencil, unstructured
from jaxmodal.precond import amg, gmg

STEEL = {0: Material(isotropic(210e9, 0.3), 7850.0)}
MAT = {0: Material(isotropic(3.0, 0.3), 2.0), 1: Material(isotropic(3.0, 0.3), 200.0)}
BCS = {"xmin": "xyz"}
S = 30  # recurrence steps timed per cut


def _dense_apply(Ainv, r):
    return Ainv @ r


def cuts(mesh, n_modes, timed, precond):
    """g and d at every cut k (between modes k and k + 1) of an n_modes solve, the count step
    timed at the cuts in `timed` (S steps, b = 10, tau from the rule at that cut's d), and the
    predicted count time ceil(50 / 10) d t_step against the solve time."""
    t = time.perf_counter()
    lam, V, info = solve_modes(mesh, STEEL, BCS, n_modes)
    t_solve = time.perf_counter() - t
    op = (stencil if mesh.voxel is not None else unstructured)(mesh, STEEL)
    free = dof_mask(mesh, BCS)
    n_free = int(free.sum())
    rows = []
    for k in range(1, n_modes):
        lam_c = (lam[k - 1] + lam[k]) / 2
        gap = (lam[k] - lam[k - 1]) / lam_c
        d, bias = count_degree(gap, lam[-1] / lam_c, n_modes, n_free) if gap > 23e-5 else (None, 0)
        rows.append(dict(k=k, gap=gap, degree=d, bias=bias))
    timing = []
    for k in timed:
        r = rows[k - 1]
        if r["degree"] is None:
            continue
        lam_c = (lam[k - 1] + lam[k]) / 2
        tau = 0.01 / (r["degree"] * math.sqrt(n_free))
        P = precond(mesh, op, free, lam_c)
        count(op, free, V, lam, lam[k - 1], lam[k], precond=P, degree=2, n_probes=10, tau=tau)
        _, ci = count(op, free, V, lam, lam[k - 1], lam[k], precond=P, degree=S, n_probes=10,
                      tau=tau)  # fmt: skip
        t_step = ci["time"]["step"]
        its = ci["pcg_iterations"]
        timing.append(dict(k=k, tau=tau, t_step=t_step, pcg=its[0], pcg_max=its[1],
                           verifiable=ci["verifiable"]))  # fmt: skip
    steps = np.array([x["k"] for x in timing])
    for r in rows:
        if r["degree"] is not None and len(steps):
            near = timing[int(np.argmin(np.abs(steps - r["k"])))]
            r["predicted_s"] = 5 * r["degree"] * near["t_step"]
            r["ratio"] = r["predicted_s"] / t_solve
    return dict(n_dof=op.n_dof, solve_s=t_solve, iterations=info["iterations"], cuts=rows,
                timed=timing)  # fmt: skip


def _graded():
    coords = [np.cumsum(np.r_[0, 1.3 ** np.arange(6)]) / 12, np.linspace(0, 0.2, 6),
              np.linspace(0, 0.25, 5)]  # fmt: skip
    mesh = box_mesh((6, 5, 4), type="hex20", coords=coords)
    c = mesh.points[mesh.blocks["hex20"][:, :8]].mean(1)
    mesh.material[:] = (c[:, 2] > 0.125).astype(int)  # the upper half 100x denser
    return mesh


def errors(small=False):
    """|s(tau) - s(exact)| (max over 10 probes) against d and tau, AMG at lam_c."""
    meshes = {"reference": box_mesh((6, 5, 4), (1 / 6, 1 / 5, 1 / 4), type="hex20"),
              "graded": _graded()}  # fmt: skip
    degrees = (20, 100) if small else (20, 100, 400)
    taus = (1e-4, 1e-8) if small else (1e-4, 1e-6, 1e-8, 1e-10)
    out = []
    for name, mesh in meshes.items():
        free = dof_mask(mesh, BCS)
        K, M = assemble(mesh, MAT)
        Kf, Mf = K.toarray()[np.ix_(free, free)], M.toarray()[np.ix_(free, free)]
        lam, V = sla.eigh(Kf, Mf)
        W = np.zeros((len(free), 14))
        W[free] = V[:, :14]
        dm = np.sqrt(np.diag(Mf))
        ev = np.linalg.eigvalsh(Mf / np.outer(dm, dm))
        op = unstructured(mesh, MAT)
        j = int(np.argmax((lam[3:12] - lam[2:11]) / lam[3:12])) + 2  # cut after mode j + 1
        lam_c = (lam[j] + lam[j + 1]) / 2
        s = float(ap_operator(op, free, lam_c).s)
        Ainv = np.linalg.inv(a_p(K + lam_c * M, free, s).toarray())
        exact = Partial(_dense_apply, jnp.asarray(Ainv))
        P = amg(op, free, lam_c, mesh.points)
        for miss in ((), (1,)):
            kept = np.setdiff1d(np.arange(14), miss)
            args = (op, free, W[:, kept], lam[kept], lam[j], lam[j + 1])
            for d in degrees:
                s_ref = count(*args, precond=exact, degree=d, n_probes=10, tau=1e-13)[1]["probes"]
                for tau in taus:
                    _, ci = count(*args, precond=P, degree=d, n_probes=10, tau=tau)
                    err = float(np.abs(ci["probes"] - s_ref).max())
                    out.append(dict(mesh=name, n_free=int(free.sum()),
                                    kappa_M=float(ev[-1] / ev[0]), missing=bool(miss), degree=d,
                                    tau=tau, s=float(s_ref.max()),
                                    error=err, pcg=ci["pcg_iterations"][0],
                                    verifiable=ci["verifiable"]))  # fmt: skip
    return out


def floor():
    """The smallest tau (decades) at which the reference box's Jacobi count (degree 1) is
    verifiable, and which exit set it."""
    mesh = box_mesh((6, 5, 4), (1 / 6, 1 / 5, 1 / 4), type="hex20")
    free = dof_mask(mesh, BCS)
    K, M = assemble(mesh, MAT)
    lam, V = sla.eigh(K.toarray()[np.ix_(free, free)], M.toarray()[np.ix_(free, free)])
    W = np.zeros((len(free), 14))
    W[free] = V[:, :14]
    op = unstructured(mesh, MAT)
    last = None
    for tau in 10.0 ** -np.arange(10, 18):
        _, ci = count(op, free, W, lam[:14], lam[5], lam[6], degree=1, tau=tau)
        if not ci["verifiable"]:
            return dict(floor=last, failed_at=tau, pcg_max=ci["pcg_iterations"][1])
        last = tau
    return dict(floor=last, failed_at=None)


def run(small=False):
    config.enable_x64()
    if small:
        h, v, n_modes, timed = 6, 8, 12, (5, 10)
    else:
        h, v, n_modes, timed = 16, 32, 200, (5, 10, 25, 50, 100, 199)
    r = dict(
        hex20=cuts(
            box_mesh((h,) * 3, (1 / h,) * 3, type="hex20"),
            n_modes,
            timed,
            lambda mesh, op, free, s: amg(op, free, s, mesh.points),
        ),  # fmt: skip
    )
    jax.clear_caches()
    r["voxel"] = cuts(voxel_mesh((v,) * 3, (1 / v,) * 3), n_modes, timed,
                      lambda mesh, op, free, s: gmg(op, free, s))  # fmt: skip
    r["errors"] = errors(small)
    r["floor"] = floor()
    r["cuts"] = r["hex20"]["cuts"]
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=harness.RESULTS)
    a = p.parse_args()
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "cuts"}, indent=1, default=str))
    harness.write_result("counting_study", r, a.out)
