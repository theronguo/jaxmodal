"""The examples run end to end and pass their own checks."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("m*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example(path, tmp_path):
    r = subprocess.run(
        [sys.executable, str(path), str(tmp_path)], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stderr
    assert "check passed" in r.stdout
