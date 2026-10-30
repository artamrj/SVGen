"""Build the monochrome line-icon dataset from open-source icon packs.

Sources (all 24x24 line icons, permissive licenses):
  lucide-static (ISC), @tabler/icons outline (MIT), iconoir regular (MIT),
  heroicons 24/outline (MIT).

Download the packs with `make data-download`, then run
`python -m svgen.data --raw data/raw/extracted --out data/processed`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from svgen.prompting import chat_record
from svgen.svg import SvgError, extract_body


@dataclass
class Icon:
    source: str
    name: str
    body: str
    tags: list[str] = field(default_factory=list)
    category: str = ""

    @property
    def group(self) -> str:
        """Base concept name, used so train and test never share a concept."""
        words = [w for w in self.name.split("-") if not w.isdigit()]
        return "-".join(words) or self.name


def _find_pack(raw: Path, prefix: str) -> Path | None:
    matches = sorted(raw.glob(f"{prefix}*/package"))
    return matches[-1] if matches else None


def _read_icons(directory: Path, source: str, meta: dict[str, dict]) -> Iterator[Icon | str]:
    for path in sorted(directory.glob("*.svg")):
        name = path.stem
        try:
            body = extract_body(path.read_text(encoding="utf-8"))
        except SvgError as exc:
            yield f"{source}: {exc}"
            continue
        info = meta.get(name, {})
        yield Icon(source, name, body, list(info.get("tags", [])), str(info.get("category", "")))


def load_icons(raw: Path) -> tuple[list[Icon], Counter[str]]:
    icons: list[Icon] = []
    rejected: Counter[str] = Counter()

    def collect(items: Iterator[Icon | str]) -> None:
        for item in items:
            if isinstance(item, Icon):
                icons.append(item)
            else:
                rejected[re.sub(r"[-\d.]+$", "", item)] += 1

    if pack := _find_pack(raw, "lucide-static"):
        tags = json.loads((pack / "tags.json").read_text(encoding="utf-8"))
        collect(_read_icons(pack / "icons", "lucide", {k: {"tags": v} for k, v in tags.items()}))
    if pack := _find_pack(raw, "tabler-icons"):
        meta = json.loads((pack / "icons.json").read_text(encoding="utf-8"))
        collect(_read_icons(pack / "icons" / "outline", "tabler", meta))
    if pack := _find_pack(raw, "iconoir"):
        collect(_read_icons(pack / "icons" / "regular", "iconoir", {}))
    if pack := _find_pack(raw, "heroicons"):
        collect(_read_icons(pack / "24" / "outline", "heroicons", {}))
    return icons, rejected


# --- prompts -------------------------------------------------------------------

_WORD_FIXES = {"ai": "AI", "tv": "TV", "usb": "USB", "pdf": "PDF", "wifi": "wi-fi", "sd": "SD", "qr": "QR"}


def describe_name(name: str) -> str:
    """'brand-github' -> 'github logo', 'bell-off' -> 'bell crossed out'."""
    words = [w for w in name.split("-") if w and not w.isdigit()]
    suffix = ""
    if words and words[0] == "brand":
        words, suffix = words[1:], " logo"
    if words and words[-1] == "off" and len(words) > 1:
        words, suffix = words[:-1], suffix + " crossed out"
    text = " ".join(_WORD_FIXES.get(w, w) for w in words) + suffix
    return text.strip() or name


_NAME_TEMPLATES = [
    "{d}",
    "{d} icon",
    "a {d} icon",
    "line icon of a {d}",
    "simple outline {d}",
    "minimal {d} symbol",
    "black and white {d} icon",
    "{d} pictogram, thin lines",
]
_TAG_TEMPLATES = [
    "{d}, {t}",
    "icon for {t}: {d}",
    "{d} ({t})",
]


def make_prompts(icon: Icon, rng: random.Random, count: int) -> list[str]:
    desc = describe_name(icon.name)
    tags = [str(t) for t in icon.tags if str(t).strip() and str(t).lower() not in desc.lower()]
    prompts: list[str] = []
    pool = list(_NAME_TEMPLATES)
    rng.shuffle(pool)
    for k in range(count):
        if tags and k % 2 == 1:
            chosen = ", ".join(rng.sample(tags, min(len(tags), rng.randint(1, 3))))
            prompts.append(rng.choice(_TAG_TEMPLATES).format(d=desc, t=chosen))
        else:
            prompts.append(pool[k % len(pool)].format(d=desc))
    return list(dict.fromkeys(prompts))


# --- build ---------------------------------------------------------------------


def split_groups(icons: list[Icon], rng: random.Random, n_valid: int, n_test: int) -> dict[str, list[Icon]]:
    groups = sorted({icon.group for icon in icons})
    rng.shuffle(groups)
    test = set(groups[:n_test])
    valid = set(groups[n_test : n_test + n_valid])
    split: dict[str, list[Icon]] = {"train": [], "valid": [], "test": []}
    for icon in icons:
        key = "test" if icon.group in test else "valid" if icon.group in valid else "train"
        split[key].append(icon)
    return split


def variant_names(icons: list[Icon]) -> set[tuple[str, str]]:
    """Icons that extend another icon of the same pack, e.g. tabler 'heart-bolt' extends 'heart'.

    Tabler alone has dozens of variants per concept. Without rebalancing, the
    plain prompt "heart" mostly teaches the model to draw some heart variant.
    """
    names = {(icon.source, icon.name) for icon in icons}
    variants: set[tuple[str, str]] = set()
    for icon in icons:
        parts = icon.name.split("-")
        if any((icon.source, "-".join(parts[:k])) in names for k in range(1, len(parts))):
            variants.add((icon.source, icon.name))
    return variants


def build(raw: Path, out: Path, seed: int, prompts_per_icon: int, max_body_chars: int, n_valid: int, n_test: int,
          root_bonus: int = 3) -> dict:
    rng = random.Random(seed)
    icons, rejected = load_icons(raw)
    seen: set[str] = set()
    unique: list[Icon] = []
    too_long = 0
    for icon in icons:
        digest = hashlib.sha1(icon.body.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if len(icon.body) > max_body_chars:
            too_long += 1
            continue
        unique.append(icon)

    split = split_groups(unique, rng, n_valid, n_test)
    variants = variant_names(unique)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, items in split.items():
        rng.shuffle(items)
        rows = []
        for icon in items:
            n = 1
            if name == "train":
                is_variant = (icon.source, icon.name) in variants
                n = 1 if is_variant and root_bonus else prompts_per_icon + (0 if is_variant else root_bonus)
            for prompt in make_prompts(icon, rng, n):
                rows.append(chat_record(prompt, icon.body))
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[name] = len(rows)
        if name == "test":
            meta = [{"source": i.source, "name": i.name, "prompt": describe_name(i.name), "body": i.body} for i in items]
            (out / "test_icons.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    report = {
        "icons_loaded": len(icons),
        "icons_unique": len(unique),
        "dropped_duplicates": len(icons) - len(unique) - too_long,
        "dropped_too_long": too_long,
        "rejected": dict(rejected),
        "by_source": dict(Counter(icon.source for icon in unique)),
        "variant_icons": len(variants),
        "rows": counts,
        "avg_body_chars": round(sum(len(i.body) for i in unique) / max(1, len(unique)), 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw/extracted", type=Path)
    parser.add_argument("--out", default="data/processed", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompts-per-icon", type=int, default=2)
    parser.add_argument("--max-body-chars", type=int, default=1400)
    parser.add_argument("--root-bonus", type=int, default=3,
                        help="extra prompts for base icons; variants get 1. 0 = no rebalancing (stages 1-2)")
    parser.add_argument("--valid", type=int, default=120)
    parser.add_argument("--test", type=int, default=150)
    args = parser.parse_args()
    report = build(args.raw, args.out, args.seed, args.prompts_per_icon, args.max_body_chars, args.valid, args.test, args.root_bonus)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
