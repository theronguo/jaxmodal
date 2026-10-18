#!/usr/bin/env sh
# The full suite in three stages: JOBS parallel workers (default 8), then the process-peak tests
# 3 at a time, then the wall-time tests alone. Extra arguments go to every pytest run.
set -e
cd "$(dirname "$0")/.."
uv run --locked pytest -m "not gpu and not serial and not peak" -n "${JOBS:-8}" "$@"
uv run --locked pytest -m "not gpu and peak" -n 3 "$@"
uv run --locked pytest -m "not gpu and serial" "$@"
