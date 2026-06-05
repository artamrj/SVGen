from __future__ import annotations

from pathlib import Path

from training.tokenizer import SvgTokenizer


def test_tokenizer_round_trip_contains_svg_shape() -> None:
    svg = '<svg viewBox="0 0 24 24"><path d="M3 12 L12 3 Z"/></svg>'
    tokenizer = SvgTokenizer.train_from_texts([svg])
    decoded = tokenizer.decode(tokenizer.encode(svg))
    assert "<svg" in decoded
    assert "path" in decoded
    assert "M3" in decoded or "M 3" in decoded


def test_tokenizer_save_load_round_trip(tmp_path: Path) -> None:
    svg = '<svg viewBox="0 0 24 24"><path d="M3 12 L12 3 Z"/></svg>'
    tokenizer = SvgTokenizer.train_from_texts([svg])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = SvgTokenizer.load(path)
    assert loaded.decode(loaded.encode(svg))
