from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.export import export_artifacts


def test_export_fails_without_checkpoint_by_default(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        export_artifacts(tmp_path / "missing-checkpoint", tmp_path / "export")


def test_export_allows_explicit_placeholder(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"vocab": {"<PAD>": 0}}), encoding="utf-8")

    out_dir = export_artifacts(
        tmp_path / "missing-checkpoint",
        tmp_path / "export",
        allow_placeholder=True,
        tokenizer=tokenizer,
    )

    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"] == "SVGen 0.2.5 Tiny Generator"
    assert config["status"] == "placeholder"
    assert config["runtime_inference"] == "unavailable"
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "generation_config.json").exists()
