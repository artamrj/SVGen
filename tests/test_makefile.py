from __future__ import annotations

from pathlib import Path


def test_makefile_svg_target_uses_mlx_generator() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "svg:" in makefile
    assert '$(PYTHON) -m svgen.generate "$(PROMPT)"' in makefile
    assert "mlx_lm lora -c configs/lora_mono.yaml" in makefile


def test_makefile_keeps_legacy_python_generate() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "legacy-svg:" in makefile
    assert "$(PYTHON) -m training.generate" in makefile
    assert "--prompt \"$(PROMPT)\"" in makefile
