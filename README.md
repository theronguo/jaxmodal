# jaxmodal

Large-scale 3D linear modal analysis, $K v = \lambda M v$, in JAX. It uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. The targets are the lowest modes and modes in frequency bands, with FFD shape optimization and topology optimization later.

**Status:** early development. PR 0 (scaffold, benchmark harness, conventions), PR 1 (mesh data model, readers, voxel generator), PR 2 (anisotropic materials, rotations, phases), PR 3 (hex8/hex20/hex27 element kernels), PR 4 (DOF map, assembled SciPy reference operators, Dirichlet masking, box meshes), PR 5 (matrix-free unstructured operator, RCM reordering, memory estimator), PR 6 (voxel stencil operator), PR 7 (rigid body modes, M-orthonormalization, deflation projector), PR 8a (certified SciPy reference eigensolver, dense inertia count, sliding-block verification), PR 8b (Srinivas–Rao plates, anisotropic Visscher box, MAC mode matching), PR 8c (NAFEMS FV52/FV12, Lamb sphere, graded box meshes), PR 9a (matrix-free LOBPCG with Jacobi), PR 9b (chunked LOBPCG with locking, fp32 deflation basis, low-memory mode, LOBPCG memory estimator), PR 9c (the public `solve_modes` API, milestone examples, Mac acceptance budgets), PR 10 (PCG, MINRES, block-ELL, node-block Jacobi, Chebyshev, the matrix-free $A_P$), PR 11a (voxel geometric multigrid), PR 11b (GMG by default in `solve_modes` with a GMG-refined σ pre-estimate, fp32 V-cycle, GMG memory estimator, `solve_modes_memory`; a 2M-DOF voxel solve in about 1 min on a laptop), PR 12 (p-multigrid for hex20/hex27 meshes, with a test-scale coarse solve), PR 13a (smoothed-aggregation AMG below the hex8 level, `amg`), PR 13b (AMG by default in `solve_modes` on unstructured meshes, production hex8 assembler, aggregate reuse, AMG memory estimator; a 1M-DOF hex20 solve of 20 modes in about 72 s on a laptop), PR 13c (every solver option reachable from `solve_modes`, `X0` warm starts, the Chebyshev smoother as a loop), PR 14a (the stochastic completeness count `count`, exact element-wise mass sampling, its probe-block memory model) and PR 14b (the $[0, f_{max}]$ band mode of `solve_modes` with `verify`) are done: milestones M2 (first modes end to end) and M3 (multigrid preconditioners, voxel and unstructured). PR 15 (interior bands: shift-invert slices and the band strategy choice) comes next. See [`docs/ROADMAP.md`](docs/ROADMAP.md); decisions taken without user input and deferred weaknesses are in [`docs/open_items.md`](docs/open_items.md).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Development happens on macOS arm64.

```sh
uv sync --locked
```

## Usage

Importing `jaxmodal` changes neither the x64 nor the compilation-cache setting. Enable fp64 explicitly:

```python
import jaxmodal.config

jaxmodal.config.enable_x64()
```

Read a mesh (flat Abaqus `.inp`, Gmsh 4.1 ASCII `.msh`, or `.vtu`), or build a voxel mesh:

```python
import numpy as np
from jaxmodal.mesh import read, voxel_mesh, write

mesh = read("tests/data/gmsh_pair.msh")  # any .inp, .msh or .vtu
x, y, z = np.meshgrid(*[np.arange(40) - 19.5] * 3, indexing="ij")
sphere = voxel_mesh((40, 40, 40), spacing=(0.5, 0.5, 0.5), mask=x**2 + y**2 + z**2 < 400)
write("sphere.vtu", sphere)
```

See [`docs/mesh.md`](docs/mesh.md) for the data model and format rules.

Define materials (traceable, differentiable) and map the mesh's material and orientation labels to a per-phase table of rotated stiffness and density:

```python
from jaxmodal.materials import Material, check_materials, from_euler, isotropic, phase_table, phases
from jaxmodal.materials import transversely_isotropic

materials = {0: Material(isotropic(210e9, 0.3), 7850.0)}  # label -> Material(C, rho)
index, pairs = phases(sphere)  # host: element -> phase, distinct (material, orientation) labels
check_materials(pairs, materials)
C, rho = phase_table(pairs, materials)  # (P, 6, 6) global stiffness, (P,) density

fiber = Material(transversely_isotropic(140e9, 10e9, 0.3, 0.45, 5e9), 1600.0)
C, rho = phase_table(((7, 1),), {7: fiber}, {1: from_euler((0.5, 0.0, 0.0))})
```

See [`docs/materials.md`](docs/materials.md) for the constructors, rotation conventions and phases.

Check the elements once on the host, then compute batched element matrices (pure `jax.numpy`, differentiable with respect to coordinates and material):

```python
from jaxmodal.elements import check_elements, lumped_mass, mass, stiffness

check_elements(mesh, mass="lumped")  # host: det J > 0 at quadrature points, lumped masses > 0
X = mesh.points[mesh.blocks["hex20"]]  # (n_el, 20, 3)
Ke = stiffness("hex20", X, isotropic(210e9, 0.3))  # (n_el, 60, 60), DOF 3a + i
Me = mass("hex20", X, 7850.0)  # consistent; lumped_mass gives the HRZ diagonal
```

See [`docs/elements.md`](docs/elements.md) for shape functions, quadrature, lumping and the element check.

Assemble the SciPy reference matrices (the test oracle for the matrix-free operators) on a structured box mesh, clamp a face and build the preconditioning operator $A_P$:

```python
from jaxmodal.mesh import box_mesh
from jaxmodal.operators import a_p, assemble, dof_mask, masked

box = box_mesh((8, 2, 2), spacing=(0.25, 0.25, 0.25), type="hex20")  # face sets xmin ... zmax
K, M = assemble(box, materials)  # scipy.sparse.csr_array (n_dof, n_dof), DOF 3n + i
free = dof_mask(box, {"xmin": "xyz"})  # bool, True = free
Km, Mm = masked(K, free), masked(M, free)  # constrained rows and columns zero
AP = a_p(K + 1e3 * M, free)  # SPD: constrained diagonal = mean free diagonal
```

Apply K and M matrix-free on the device, on blocks of vectors, with per-element scalings for density design:

```python
import jax.numpy as jnp
import numpy as np
from jaxmodal.memory import unstructured_memory
from jaxmodal.operators import unstructured

op = unstructured(box, materials, mass="lumped")  # stored packed Ke; scatter on CPU
op = op.scaled(s_K=np.full(box.n_el, 0.5))  # per-element stiffness scaling, no recompile
v = jnp.ones((op.n_dof, 4))  # (n_dof, m) block
y = op.apply_K(v)  # == 0.5 * K @ v
d = op.diag_M()  # lumped mass diagonal
unstructured_memory(box, m=4)["total"]  # bytes
```

On voxel meshes, the stencil operator applies one reference element per phase over slabs of the grid (same methods; masked grids included):

```python
from jaxmodal.memory import stencil_memory
from jaxmodal.operators import stencil

vop = stencil(sphere, materials)
y = vop.apply_K(jnp.ones((vop.n_dof, 4)))
stencil_memory(sphere, m=4)["total"]  # bytes
```

Rigid body modes per face-connected part, the ones left under Dirichlet sets, M-orthonormalized for deflation:

```python
from jaxmodal.operators import m_orthonormalize, project, rigid_modes

free = dof_mask(box, {"xmin": "x"})  # sliding on xmin: T_y, T_z and R_x remain
R = jnp.asarray(rigid_modes(box, free))  # (n_dof, 3), constrained rows zero
R = m_orthonormalize(R, op.apply_M(R))  # R^T M R = I (consumes its first argument)
X = jnp.ones((op.n_dof, 4))
Y = project(R, X, op.apply_M(X))  # X - R (R^T M X): M-orthogonal to the rigid modes
```

See [`docs/operators.md`](docs/operators.md) for the DOF map, assembly and the matrix-free and voxel stencil operators, and [`docs/boundary_conditions.md`](docs/boundary_conditions.md) for BCs, $A_P$, rigid modes and deflation.

The certified SciPy reference eigensolver and the exact inertia count, on the assembled matrices (the oracle for the matrix-free solver):

```python
from jaxmodal.eigen import reference_modes
from jaxmodal.linalg import count_below

lam, V = reference_modes(K, M, 10, free, points=box.points)  # lowest 10, V (n_dof, 10), V^T M V = I
n = count_below(K, M, 0.5 * (lam[2] + lam[3]), free)  # 3 below sigma: the rigid modes (dense)
```

The matrix-free LOBPCG eigensolver (Jacobi-preconditioned, with rigid-mode deflation):

```python
from jaxmodal.eigen import lobpcg

op = unstructured(box, materials)  # consistent mass, as K and M above
R = m_orthonormalize(R, op.apply_M(R))  # again: R above was M-orthonormal for the lumped mass
mu, V, info = lobpcg(op, free, 5, tol=1e-10, R=R)  # the lowest 5 elastic modes
assert info["converged"] and np.allclose(mu, lam[3:8], rtol=1e-8)
mu, V, info = lobpcg(op, free, 6, tol=1e-10, R=R, chunk=3)  # 2 chunks, locked; V on the host
assert info["converged"] and np.allclose(mu, lam[3:9], rtol=1e-8)
```

The public entry point builds the operator (voxel stencil for voxel meshes, matrix-free otherwise), deflates the rigid modes the BCs leave and runs LOBPCG:

```python
from jaxmodal.eigen import solve_modes

mu, V, info = solve_modes(box, materials, {"xmin": "x"}, 5, tol=1e-10)  # the lowest 5 elastic modes
assert info["n_rigid"] == 3 and np.allclose(mu, lam[3:8], rtol=1e-8)
f = info["frequencies"]  # sqrt(lam) / (2 pi), Hz for SI inputs; info["rigid"]: the rigid modes
```

Linear solves on $A_P$ (PR 10): PCG and MINRES on blocks of right-hand sides, Jacobi, node-block Jacobi and Chebyshev preconditioners, and a block-ELL SpMV for assembled matrices:

```python
from jaxmodal.linalg import pcg
from jaxmodal.operators import ap_operator
from jaxmodal.precond import block_jacobi, chebyshev, lmax_estimate

A = ap_operator(op, free, lam[3])  # matrix-free A_P of K + sigma M
D = block_jacobi(op, free, lam[3])  # inverted 3x3 node blocks of K + sigma M
S = chebyshev(A, D, lmax_estimate(A, D))  # degree-2 Chebyshev over block Jacobi
b = jnp.asarray(np.random.default_rng(0).standard_normal((op.n_dof, 2)) * free[:, None])
x, info = pcg(A, b, S, tol=1e-10)  # info: iterations, residuals, converged, stagnated, ...
assert info["converged"].all() and np.allclose(a_p(K + lam[3] * M, free) @ np.asarray(x), b)
```

Geometric multigrid for voxel meshes (PR 11a): a V-cycle with Galerkin coarse levels, h-independent PCG iterations, and a preconditioner for `pcg`, `minres` or `lobpcg`:

```python
from jaxmodal.precond import gmg

cube = voxel_mesh((16, 16, 16), spacing=(1 / 16,) * 3)
vop, vfree = stencil(cube, materials), dof_mask(cube, {"xmin": "xyz"})
sigma = 1e7  # any sigma in [0, 3 lam_1] works; sigma = 0 is refused unless provably safe
P = gmg(vop, vfree, sigma)  # pytree V-cycle on (n_dof, k) blocks, masked output
b = jnp.asarray(np.random.default_rng(0).standard_normal((vop.n_dof, 2)) * vfree[:, None])
x, info = pcg(ap_operator(vop, vfree, sigma), b, P, tol=1e-8)
assert info["converged"].all() and info["iterations"].max() <= 20  # 13
mu, V, info = lobpcg(vop, vfree, 4, tol=1e-8, precond=P)
assert info["converged"]
```

`solve_modes` uses this GMG automatically on voxel meshes (PR 11b): it pre-estimates σ with GMG itself (or takes a `sigma` hint), and `precond_dtype="float32"` runs the V-cycle in fp32 (same iterations, a 39% faster V-cycle, half its memory). `solve_modes_memory` estimates the peak of the same call beforehand:

```python
from jaxmodal.memory import solve_modes_memory

mu, V, info = solve_modes(cube, materials, {"xmin": "xyz"}, 6, precond_dtype="float32")
assert info["precond"] == "gmg" and info["converged"]  # info["sigma"]: the shift GMG used
solve_modes_memory(cube, {"xmin": "xyz"}, 6, precond_dtype="float32")["total"]  # bytes
```

p-multigrid for unstructured hex20/hex27 meshes (PR 12): Chebyshev smoothing on the matrix-free operator and the hex8 Galerkin matrix on the corner nodes as the coarse level, about 14 PCG iterations at every mesh size. `pmg` solves the coarse level by SciPy `splu` on the host (test scale, about 50k coarse DOF); `amg` (PR 13a) is the production variant: smoothed-aggregation AMG on the hex8 level, exported to the device as one flat V-cycle with the p-level on top (all-hex8 meshes skip the p-level). It needs the mesh coordinates in the operator's node order for its near-null space (the rigid modes):

```python
from jaxmodal.precond import amg, pmg

hbox = box_mesh((12, 3, 3), spacing=(1 / 12,) * 3, type="hex20")
hop, hfree = unstructured(hbox, materials), dof_mask(hbox, {"xmin": "xyz"})
P = pmg(hop, hfree, 1e5)  # a GMG (the shared V-cycle) with a p-level; sigma > 0 is required
mu, V, info = lobpcg(hop, hfree, 4, tol=1e-8, precond=P)
assert info["converged"] and info["iterations"] <= 40
P = amg(hop, hfree, 1e5, hbox.points, max_coarse=100)  # p-level, AMG levels, dense Cholesky
mu, V, info = lobpcg(hop, hfree, 4, tol=1e-8, precond=P)
assert info["converged"] and info["iterations"] <= 40
```

`solve_modes` uses this AMG automatically on unstructured meshes (PR 13b), with the same σ pre-estimate as GMG (started at σ = 0 when the BCs leave no rigid modes); `solve_modes_memory` models it too. `amg(..., return_aggregates=True)` and `amg(..., aggregates=...)` reuse the setup's aggregates for new materials or densities on the same mesh:

```python
mu, V, info = solve_modes(hbox, materials, {"xmin": "xyz"}, 4)
assert info["precond"] == "amg" and info["converged"]
solve_modes_memory(hbox, {"xmin": "xyz"}, 4)["total"]  # bytes
P, agg = amg(hop, hfree, 1e5, hbox.points, max_coarse=100, return_aggregates=True)
P2 = amg(hop.scaled(s_K=0.5 * hop.s_K), hfree, 1e5, hbox.points, aggregates=agg)
```

The defaults are the robust choice; every option of the operator, the preconditioner and the σ pre-estimate is reachable from `solve_modes` through dicts (PR 13c), and `X0` warm-starts from earlier modes:

```python
mu2, V2, info = solve_modes(
    hbox,
    materials,
    {"xmin": "xyz"},
    4,
    X0=V,
    operator_options={"mode": "otf"},
    precond_options={"degree": 3, "max_coarse": 100},
)
assert info["converged"]
solve_modes_memory(hbox, {"xmin": "xyz"}, 4, operator_options={"mode": "otf"})["total"]
```

`count` checks that no mode is missing below a cut between two computed eigenvalues: a stochastic estimate of the number of eigenvalues below the cut that `V` does not contain (a Jackson–Chebyshev step of $(K + \lambda_c M)^{-1} M$, PR 14a). It passes below 0.5 and flags itself unverifiable when the inner solves cannot reach their tolerance:

```python
from jaxmodal.eigen import count

est, cinfo = count(hop, hfree, V, mu, mu[1], mu[2], n_probes=10)  # cut between modes 2 and 3
assert cinfo["verifiable"] and cinfo["passed"] and est < 0.5
```

`band=(0, f_max)` returns every elastic mode up to $f_{max}$ (PR 14b) without knowing their number: chunked LOBPCG from 0 stops at the first chunk with a converged Ritz value beyond the edge, then `count` checks the cut between the last mode inside and the first beyond. `verify=True` always counts (unless the solve failed or the count cannot fit its memory; then it warns), `verify=None` (the default) only when its predicted cost is below the solve, `verify=False` never:

```python
f = info["frequencies"]
mub, Vb, binfo = solve_modes(hbox, materials, {"xmin": "xyz"}, band=(0, (f[1] + f[2]) / 2), verify=True)
assert len(mub) == 2 and binfo["count"]["passed"]  # the two bending modes, none missing
```

See [`docs/preconditioners.md`](docs/preconditioners.md) for the solvers' stopping rules, the preconditioners, block-ELL, the measured h-study, the voxel GMG (fp32 V-cycle, memory estimator), p-MG and AMG.

The scripts in [`examples/`](examples) walk through each milestone (`m0_foundations.py`: meshes, VTU, materials; `m1_operators.py`: element matrices and operators against the assembled matrices; `m2_modes.py`: a cantilever against Euler–Bernoulli and a free voxel cube); each ends with a check and runs as a smoke test.

See [`docs/eigensolvers.md`](docs/eigensolvers.md) for `solve_modes`, its options and their defaults, its Mac acceptance budgets, the frequency band mode and `verify`, the completeness count, LOBPCG (algorithm, chunks and locking, low-memory mode, memory model, convergence, measured floor and cost), the reference solver, its certification and size limits, and [`docs/verification.md`](docs/verification.md) for the exact solutions and measured convergence.

## Development

```sh
scripts/check.sh                        # lint + fast tests (< 5 min)
uv run --locked pytest -m "not gpu"     # merge gate, slow tests included
uv run --locked python benchmarks/harness.py   # machine peaks -> benchmarks/results/system.json
```

Conventions (node ordering, Voigt order, precision, indices, tolerances, benchmark harness rules) are in [`docs/conventions.md`](docs/conventions.md).

## License

MIT, see [`LICENSE`](LICENSE).
