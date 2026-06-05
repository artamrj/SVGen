from __future__ import annotations

from training.validate_dataset import build_report


def test_dataset_validator_accepts_sample_data() -> None:
    report = build_report("data/sample/dataset.jsonl")
    assert report.total == 3
    assert report.valid == 3
    assert report.rejected == 0
    assert report.tag_usage["svg"] == 3


def test_dataset_validator_reports_rejections() -> None:
    report = build_report("data/sample/invalid_dataset.jsonl")
    assert report.rejected == 2
