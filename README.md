# jaxmodal

Large-scale 3D linear modal analysis, $K v = \lambda M v$, in JAX. It uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. The targets are the lowest modes and modes in frequency bands, with FFD shape optimization and topology optimization later.

**Status:** early development. PR 0 (scaffold, benchmark harness, conventions), PR 1 (mesh data model, readers, voxel generator), PR 2 (anisotropic materials, rotations, phases), PR 3 (hex8/hex20/hex27 element kernels), PR 4 (DOF map, assembled SciPy reference operators, Dirichlet masking, box meshes), PR 5 (matrix-free unstructured operator, RCM reordering, memory estimator), PR 6 (voxel stencil operator), PR 7 (rigid body modes, M-orthonormalization, deflation projector), PR 8a (certified SciPy reference eigensolver, dense inertia count, sliding-block verification), PR 8b (Srinivas–Rao plates, anisotropic Visscher box, MAC mode matching), PR 8c (NAFEMS FV52/FV12, Lamb sphere, graded box meshes), PR 9a (matrix-free LOBPCG with Jacobi), PR 9b (chunked LOBPCG with locking, fp32 deflation basis, low-memory mode, LOBPCG memory estimator), PR 9c (the public `solve_modes` API, milestone examples, Mac acceptance budgets), PR 10 (PCG, MINRES, block-ELL, node-block Jacobi, Chebyshev, the matrix-free $A_P$), PR 11a (voxel geometric multigrid) and PR 11b (GMG by default in `solve_modes` with a GMG-refined σ pre-estimate, fp32 V-cycle, GMG memory estimator, `solve_modes_memory`; a 2M-DOF voxel solve in about 1 min on a laptop) are done: milestone M2, first modes end to end. PRs 12–13 (multigrid for unstructured meshes) come next. See [`docs/ROADMAP.md`](docs/ROADMAP.md); decisions taken without user input and deferred weaknesses are in [`docs/open_items.md`](docs/open_items.md).

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

See [`docs/preconditioners.md`](docs/preconditioners.md) for the solvers' stopping rules, the preconditioners, block-ELL, the measured h-study and the voxel GMG (fp32 V-cycle, memory estimator).

The scripts in [`examples/`](examples) walk through each milestone (`m0_foundations.py`: meshes, VTU, materials; `m1_operators.py`: element matrices and operators against the assembled matrices; `m2_modes.py`: a cantilever against Euler–Bernoulli and a free voxel cube); each ends with a check and runs as a smoke test.

See [`docs/eigensolvers.md`](docs/eigensolvers.md) for `solve_modes` and its Mac acceptance budgets, LOBPCG (algorithm, chunks and locking, low-memory mode, memory model, convergence, measured floor and cost), the reference solver, its certification and size limits, and [`docs/verification.md`](docs/verification.md) for the exact solutions and measured convergence.

## Development

```sh
scripts/check.sh                        # lint + fast tests (< 5 min)
uv run --locked pytest -m "not gpu"     # merge gate, slow tests included
uv run --locked python benchmarks/harness.py   # machine peaks -> benchmarks/results/system.json
```

Conventions (node ordering, Voigt order, precision, indices, tolerances, benchmark harness rules) are in [`docs/conventions.md`](docs/conventions.md).

## License

MIT, see [`LICENSE`](LICENSE).
