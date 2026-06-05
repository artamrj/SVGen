from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

from training.validate_dataset import validate_svg


def prompt_from_filename(name: str) -> str:
    stem = Path(name).stem
    words = stem.replace("-", " ")
    return f"minimal outline {words} icon"


def normalize_svg(svg: str) -> str:
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.DOTALL)
    svg = re.sub(r"\s+", " ", svg.strip())
    svg = svg.replace(' xmlns="http://www.w3.org/2000/svg"', "")
    svg = svg.replace(' width="24"', "").replace(' height="24"', "")
    svg = re.sub(r'\sclass="[^"]*"', "", svg)
    return svg


def build_records(tarball: Path, limit: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with tarfile.open(tarball, "r:gz") as archive:
        members = sorted(
            (member for member in archive.getmembers() if member.name.startswith("package/icons/") and member.name.endswith(".svg")),
            key=lambda member: member.name,
        )
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            svg = normalize_svg(extracted.read().decode("utf-8"))
            try:
                validate_svg(svg)
            except ValueError:
                continue
            records.append({"text": prompt_from_filename(member.name), "svg": svg})
            if len(records) >= limit:
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SVGen JSONL data from a lucide-static npm tarball.")
    parser.add_argument("--tarball", required=True)
    parser.add_argument("--out", default="data/train_1000.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    records = build_records(Path(args.tarball), args.limit)
    if len(records) < args.limit:
        raise RuntimeError(f"Only found {len(records)} SVG records, expected {args.limit}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
