from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetRecord:
    text: str
    svg: str


def parse_record(raw: object, line_number: int) -> DatasetRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"Line {line_number}: record must be a JSON object")

    text = raw.get("text")
    svg = raw.get("svg")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Line {line_number}: text must be a non-empty string")
    if not isinstance(svg, str) or not svg.strip():
        raise ValueError(f"Line {line_number}: svg must be a non-empty string")

    return DatasetRecord(text=text, svg=svg)


def load_jsonl(path: str | Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON") from exc
            records.append(parse_record(raw, line_number))
    return records


def iter_training_sequences(records: Iterable[DatasetRecord]) -> Iterable[str]:
    for record in records:
        yield f"<PROMPT> {record.text} <SVG> {record.svg}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an SVGen JSONL dataset.")
    parser.add_argument("path", help="Path to a JSONL dataset")
    args = parser.parse_args()
    records = load_jsonl(args.path)
    print(f"Loaded {len(records)} records from {args.path}")


if __name__ == "__main__":
    main()
