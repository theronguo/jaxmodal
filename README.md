# jaxmodal

Large-scale 3D linear modal analysis, $K v = \lambda M v$, in JAX. It uses matrix-free operators, preconditioned LOBPCG, hex8/hex20/hex27 and solid-shell elements, and linear anisotropic elasticity. The targets are the lowest modes and modes in frequency bands, with FFD shape optimization and topology optimization later.

**Status:** early development. PR 0 (scaffold, benchmark harness, conventions) is done; there is no solver yet. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Development happens on macOS arm64.

```sh
uv sync --locked
```

## Usage

Importing `jaxmodal` changes no JAX setting. Enable fp64 explicitly:

```python
import jaxmodal.config

jaxmodal.config.enable_x64()
```

## Development

```sh
scripts/check.sh                        # lint + fast tests (< 5 min)
uv run --locked pytest -m "not gpu"     # merge gate, slow tests included
uv run --locked python benchmarks/harness.py   # machine peaks -> benchmarks/results/system.json
```

Conventions (node ordering, Voigt order, precision, indices, tolerances, benchmark harness rules) are in [`docs/conventions.md`](docs/conventions.md).

## License

MIT, see [`LICENSE`](LICENSE).
