from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from training.dataset import DatasetRecord, parse_record

ALLOWED_TAGS = {"svg", "path", "circle", "rect", "line", "polyline", "polygon", "g"}
ALLOWED_ATTRS = {
    "viewBox",
    "d",
    "cx",
    "cy",
    "r",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "points",
    "fill",
    "stroke",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "transform",
    "opacity",
}
UNSAFE_VALUE_PATTERN = re.compile(r"(javascript:|data:|https?://)", re.IGNORECASE)


@dataclass
class DatasetReport:
    total: int
    valid: int
    rejected: int
    tag_usage: dict[str, int]
    attribute_usage: dict[str, int]
    avg_prompt_length: float
    avg_svg_length: float
    rejection_reasons: dict[str, int]


def validate_svg(svg: str) -> tuple[Counter[str], Counter[str]]:
    if len(svg.encode("utf-8")) > 16 * 1024:
        raise ValueError("svg exceeds 16 KB")
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        raise ValueError("invalid XML") from exc
    if strip_namespace(root.tag) != "svg":
        raise ValueError("root element must be svg")
    if root.attrib.get("viewBox") != "0 0 24 24":
        raise ValueError("viewBox must be 0 0 24 24")

    tag_usage: Counter[str] = Counter()
    attr_usage: Counter[str] = Counter()
    for element in root.iter():
        tag = strip_namespace(element.tag)
        if ":" in element.tag or tag not in ALLOWED_TAGS:
            raise ValueError(f"unsupported tag: {element.tag}")
        tag_usage[tag] += 1
        for attr, value in element.attrib.items():
            attr_name = strip_namespace(attr)
            if ":" in attr or attr_name not in ALLOWED_ATTRS:
                raise ValueError(f"unsupported attribute: {attr}")
            if UNSAFE_VALUE_PATTERN.search(value):
                raise ValueError(f"unsafe attribute value: {attr}")
            attr_usage[attr_name] += 1
    return tag_usage, attr_usage


def strip_namespace(name: str) -> str:
    if name.startswith("{") and "}" in name:
        return name.rsplit("}", 1)[1]
    return name


def build_report(path: str | Path) -> DatasetReport:
    total = 0
    valid = 0
    prompt_words = 0
    svg_chars = 0
    tag_usage: Counter[str] = Counter()
    attr_usage: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                raw = json.loads(line)
                record = parse_record(raw, line_number)
                record_tags, record_attrs = validate_record(record)
            except Exception as exc:
                rejection_reasons[str(exc)] += 1
                continue
            valid += 1
            prompt_words += len(record.text.split())
            svg_chars += len(record.svg)
            tag_usage.update(record_tags)
            attr_usage.update(record_attrs)

    return DatasetReport(
        total=total,
        valid=valid,
        rejected=total - valid,
        tag_usage=dict(sorted(tag_usage.items())),
        attribute_usage=dict(sorted(attr_usage.items())),
        avg_prompt_length=round(prompt_words / valid, 2) if valid else 0.0,
        avg_svg_length=round(svg_chars / valid, 2) if valid else 0.0,
        rejection_reasons=dict(sorted(rejection_reasons.items())),
    )


def validate_record(record: DatasetRecord) -> tuple[Counter[str], Counter[str]]:
    return validate_svg(record.svg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an SVGen JSONL dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = build_report(args.input)
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"valid={report.valid} rejected={report.rejected} total={report.total}")


if __name__ == "__main__":
    main()
