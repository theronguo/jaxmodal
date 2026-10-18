#!/usr/bin/env sh
# The full suite in three stages: JOBS parallel workers (default 8), then the process-peak tests
# 3 at a time, then the wall-time tests alone. Extra arguments go to every pytest run; every
# stage runs (a stage with no selected tests counts as passed) and the exit status is 1 if any
# stage failed.
cd "$(dirname "$0")/.."
status=0
run() { uv run --locked pytest "$@" || [ $? -eq 5 ] || status=1; }
run -m "not gpu and not serial and not peak" -n "${JOBS:-8}" "$@"
run -m "not gpu and peak" -n 3 "$@"
run -m "not gpu and serial" "$@"
exit $status
