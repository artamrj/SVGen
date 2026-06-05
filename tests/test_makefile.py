from __future__ import annotations

from pathlib import Path


def test_makefile_svg_target_uses_python_generate() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "svg:" in makefile
    assert "$(PYTHON) -m training.generate" in makefile
    assert "--prompt \"$(PROMPT)\"" in makefile
