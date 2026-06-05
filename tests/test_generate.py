from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from training.generate import GenerationError, extract_svg, generate_svg, require_export_artifacts, write_svg
from training.tokenizer import SvgTokenizer


def test_require_export_artifacts_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match="Run make train first"):
        require_export_artifacts(tmp_path)


def test_extract_svg_accepts_complete_safe_svg() -> None:
    svg = extract_svg('noise <svg viewBox="0 0 24 24"><path d="M0 0"/></svg> trailing')
    assert svg == '<svg viewBox="0 0 24 24"><path d="M0 0"/></svg>'


def test_extract_svg_rejects_incomplete_output() -> None:
    with pytest.raises(GenerationError, match="complete <svg>"):
        extract_svg('<svg viewBox="0 0 24 24"><path d="M0 0"/>')


def test_extract_svg_skips_invalid_outer_candidate() -> None:
    svg = extract_svg(
        '<svg viewBox="0 0 24 24"> junk '
        '<svg viewBox="0 0 24 24"><path d="M0 0"/></svg>'
    )
    assert svg == '<svg viewBox="0 0 24 24"><path d="M0 0"/></svg>'


def test_write_svg_validates_before_writing(tmp_path: Path) -> None:
    out = tmp_path / "dog.svg"
    write_svg('<svg viewBox="0 0 24 24"><path d="M0 0"/></svg>', out)
    assert out.read_text(encoding="utf-8").startswith("<svg")


def test_write_svg_refuses_unsafe_svg(tmp_path: Path) -> None:
    out = tmp_path / "bad.svg"
    with pytest.raises(GenerationError, match="refusing to write unsafe SVG"):
        write_svg('<svg viewBox="0 0 24 24"><script>alert(1)</script></svg>', out)
    assert not out.exists()


def test_generate_svg_with_mocked_model_writes_safe_svg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    tokenizer = SvgTokenizer(
        vocab={
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<PROMPT>": 4,
            "running": 5,
            "dog": 6,
            "icon": 7,
            "<SVG>": 8,
            '<svg viewBox="0 0 24 24">': 9,
            '<path d="M0 0"/>': 10,
            "</svg>": 11,
        }
    )
    tokenizer.save(export_dir / "tokenizer.json")
    (export_dir / "model.safetensors").write_bytes(b"mock")
    (export_dir / "config.json").write_text(
        json.dumps(
            {
                "status": "trained",
                "model_config": {"context": 16, "hidden": 8, "layers": 1, "heads": 1},
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "generation_config.json").write_text(json.dumps({"max_new_tokens": 8}), encoding="utf-8")

    class FakeTorch:
        long = object()

        def manual_seed(self, _seed: int) -> None:
            return None

        def inference_mode(self):
            return nullcontext()

        def tensor(self, value, *, dtype=None, device=None):
            return value

    class FakeModel:
        def __call__(self, _input_ids):
            return self

        def __getitem__(self, _key):
            return object()

    next_ids = iter([9, 10, 11, 3])

    monkeypatch.setattr("training.generate.load_model", lambda *_args, **_kwargs: (FakeTorch(), FakeModel(), 16))
    monkeypatch.setattr("training.generate.sample_next_token", lambda *_args, **_kwargs: next(next_ids))

    svg = generate_svg(model_dir=export_dir, prompt="running dog icon", max_new_tokens=8)
    out = write_svg(svg, tmp_path / "dog.svg")
    assert out.read_text(encoding="utf-8").strip() == '<svg viewBox="0 0 24 24"><path d="M0 0"/></svg>'
