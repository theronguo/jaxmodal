#!/usr/bin/env sh
set -e
cd "$(dirname "$0")/.."
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -m "not slow and not gpu" -q
