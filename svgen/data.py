"""Build the monochrome line-icon dataset from open-source icon packs.

Sources (all line icons, permissive licenses):
  lucide-static (ISC), @tabler/icons outline (MIT), iconoir regular (MIT),
  heroicons 24/outline (MIT), plus the Iconify sets in `svgen.download.ICONIFY_SETS`
  (MIT / Apache-2.0), converted to the 24x24 canonical format by `svgen.svg.import_icon`.

The held-out concepts are frozen in data/heldout/, so evaluation scores stay
comparable when sources are added. Icons of new sources that belong to a held-out
concept are left out of every split.

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
from svgen.download import ICONIFY_SETS
from svgen.svg import SvgError, extract_body, import_icon


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


# Original packs. Held-out concepts were drawn from these, and only these fill valid/test.
CLASSIC_SOURCES = ("lucide", "tabler", "iconoir", "heroicons")
HELDOUT = Path(__file__).resolve().parent.parent / "data" / "heldout"

# Style suffixes that are not part of the concept name, per Iconify set.
_ICONIFY_SUFFIXES = {"mingcute": ("-line",), "majesticons": ("-line",)}
# IconPark numbers its variants with words ("bank-card-one"); digits are already dropped from prompts.
_NUMBER_WORDS = {w: str(k) for k, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def _read_iconify(package: Path, prefix: str) -> Iterator[Icon | str]:
    data = json.loads((package / "icons.json").read_text(encoding="utf-8"))
    source = prefix.removesuffix("-outline")
    tags: dict[str, list[str]] = {}
    for category, names in (data.get("categories") or {}).items():
        for name in names:
            tags.setdefault(name, []).append(category.lower())
    width, height = data.get("width", 16), data.get("height", 16)
    for name, icon in sorted(data["icons"].items()):
        if icon.get("hidden"):
            continue
        box = f'{icon.get("left", 0)} {icon.get("top", 0)} {icon.get("width", width)} {icon.get("height", height)}'
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}">{icon["body"]}</svg>'
        try:
            body = import_icon(svg)
        except SvgError as exc:
            yield f"{source}: {exc}"
            continue
        concept = name
        for suffix in _ICONIFY_SUFFIXES.get(prefix, ()):
            concept = concept.removesuffix(suffix)
        if prefix == "icon-park-outline":
            head, _, last = concept.rpartition("-")
            if head and last in _NUMBER_WORDS:
                concept = f"{head}-{_NUMBER_WORDS[last]}"
        yield Icon(source, concept, body, tags.get(name, []))


def load_icons(raw: Path, sources: set[str] | None = None) -> tuple[list[Icon], Counter[str]]:
    """Load every downloaded pack. `sources` limits which ones (None = all)."""
    icons: list[Icon] = []
    rejected: Counter[str] = Counter()

    def collect(items: Iterator[Icon | str]) -> None:
        for item in items:
            if isinstance(item, Icon):
                icons.append(item)
            else:
                rejected[re.sub(r"[-\d.]+$", "", item)] += 1

    def wanted(source: str) -> bool:
        return sources is None or source in sources

    if wanted("lucide") and (pack := _find_pack(raw, "lucide-static")):
        tags = json.loads((pack / "tags.json").read_text(encoding="utf-8"))
        collect(_read_icons(pack / "icons", "lucide", {k: {"tags": v} for k, v in tags.items()}))
    if wanted("tabler") and (pack := _find_pack(raw, "tabler-icons")):
        meta = json.loads((pack / "icons.json").read_text(encoding="utf-8"))
        collect(_read_icons(pack / "icons" / "outline", "tabler", meta))
    if wanted("iconoir") and (pack := _find_pack(raw, "iconoir")):
        collect(_read_icons(pack / "icons" / "regular", "iconoir", {}))
    if wanted("heroicons") and (pack := _find_pack(raw, "heroicons")):
        collect(_read_icons(pack / "24" / "outline", "heroicons", {}))
    for prefix in ICONIFY_SETS:
        if wanted(prefix.removesuffix("-outline")) and (pack := _find_pack(raw, f"iconify-json-{prefix}-")):
            collect(_read_iconify(pack, prefix))
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


def load_heldout(directory: Path = HELDOUT) -> tuple[set[str], set[str], list[dict]] | None:
    """Frozen (valid groups, test groups, test icons), or None to draw a fresh split."""
    if not (directory / "split.json").exists():
        return None
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    test_icons = json.loads((directory / "test_icons.json").read_text(encoding="utf-8"))
    return set(split["valid_groups"]), set(split["test_groups"]), test_icons


def frozen_split(icons: list[Icon], valid_groups: set[str], test_groups: set[str]) -> tuple[dict[str, list[Icon]], int]:
    """Split by the frozen held-out concepts. Only the original packs fill valid and test,
    so their contents never change; new-source icons of held-out concepts are dropped."""
    split: dict[str, list[Icon]] = {"train": [], "valid": [], "test": []}
    dropped = 0
    for icon in icons:
        key = "test" if icon.group in test_groups else "valid" if icon.group in valid_groups else "train"
        if key != "train" and icon.source not in CLASSIC_SOURCES:
            dropped += 1
            continue
        split[key].append(icon)
    return split, dropped


def build(raw: Path, out: Path, seed: int, prompts_per_icon: int, max_body_chars: int, n_valid: int, n_test: int,
          root_bonus: int = 3, sources: set[str] | None = None, heldout: Path | None = HELDOUT) -> dict:
    rng = random.Random(seed)
    icons, rejected = load_icons(raw, sources)
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

    frozen = load_heldout(heldout) if heldout else None
    dropped_heldout = 0
    if frozen:
        valid_groups, test_groups, test_icons = frozen
        split, dropped_heldout = frozen_split(unique, valid_groups, test_groups)
        # Keep the frozen order: `svgen.evaluate` samples from it with a fixed seed.
        by_key = {(icon.source, icon.name): icon for icon in split["test"]}
        missing = [t["name"] for t in test_icons if (t["source"], t["name"]) not in by_key]
        if missing or len(by_key) != len(test_icons):
            raise SystemExit(f"held-out test icons do not match the downloaded packs (missing: {missing[:5]})")
        split["test"] = [by_key[(t["source"], t["name"])] for t in test_icons]
    else:
        split = split_groups(unique, rng, n_valid, n_test)
    variants = variant_names(unique)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, items in split.items():
        if not (frozen and name == "test"):
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
        "dropped_heldout_concepts": dropped_heldout,
        "frozen_split": bool(frozen),
        "by_source": dict(Counter(icon.source for icon in unique)),
        "concepts": len({icon.group for icon in unique}),
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
    parser.add_argument("--sources", default="all",
                        help="comma-separated sources to load, 'classic' for the original 4 packs (v0.3/v0.4 data), or 'all'")
    parser.add_argument("--fresh-split", action="store_true", help="draw a new random split instead of data/heldout/")
    args = parser.parse_args()
    sources = None if args.sources == "all" else set(CLASSIC_SOURCES) if args.sources == "classic" else set(args.sources.split(","))
    report = build(args.raw, args.out, args.seed, args.prompts_per_icon, args.max_body_chars, args.valid, args.test,
                   args.root_bonus, sources, heldout=None if args.fresh_split else HELDOUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
