from __future__ import annotations

from pathlib import Path

import pytest

from training.dataset import load_jsonl


def test_load_jsonl_accepts_valid_records() -> None:
    records = load_jsonl(Path("data/sample/dataset.jsonl"))
    assert len(records) == 3
    assert records[0].text == "minimal black home icon"


def test_load_jsonl_rejects_malformed_records() -> None:
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        load_jsonl(Path("data/sample/invalid_dataset.jsonl"))
