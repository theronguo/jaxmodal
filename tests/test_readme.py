"""The README's Python snippets run, in order, in a scratch directory."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_readme_snippets(tmp_path, monkeypatch):
    code = "\n".join(re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S))
    monkeypatch.chdir(tmp_path)
    exec(code.replace('"tests/', f'"{ROOT}/tests/'), {})
