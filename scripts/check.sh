#!/usr/bin/env sh
set -e
cd "$(dirname "$0")/.."
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -m "not slow and not gpu and not serial and not peak" -n "${JOBS:-8}" -q
uv run --locked pytest -m "not slow and not gpu and (serial or peak)" -q
