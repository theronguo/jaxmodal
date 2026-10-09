# jaxmodal

Large-scale 3D linear modal analysis, $K v = \lambda M v$, in JAX. It uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. The targets are the lowest modes and modes in frequency bands, with FFD shape optimization and topology optimization later.

**Status:** early development. PR 0 (scaffold, benchmark harness, conventions) and PR 1 (mesh data model, readers, voxel generator) are done; there is no solver yet. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

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

## Development

```sh
scripts/check.sh                        # lint + fast tests (< 5 min)
uv run --locked pytest -m "not gpu"     # merge gate, slow tests included
uv run --locked python benchmarks/harness.py   # machine peaks -> benchmarks/results/system.json
```

Conventions (node ordering, Voigt order, precision, indices, tolerances, benchmark harness rules) are in [`docs/conventions.md`](docs/conventions.md).

## License

MIT, see [`LICENSE`](LICENSE).
