"""Download the icon packs used for training straight from the npm registry.

    python -m svgen.download --out data/raw/extracted

No Node.js needed. Versions are pinned so every machine builds the same dataset.
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

PACKS = {
    "lucide-static": "1.65.0",
    "@tabler/icons": "3.50.0",
    "iconoir": "7.12.1",
    "heroicons": "2.2.0",
}


def tarball_url(name: str, version: str) -> str:
    base = name.split("/")[-1]
    return f"https://registry.npmjs.org/{name}/-/{base}-{version}.tgz"


def folder_name(name: str, version: str) -> str:
    # Matches `npm pack` file names: @tabler/icons -> tabler-icons-3.50.0
    return f"{name.lstrip('@').replace('/', '-')}-{version}"


def download(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, version in PACKS.items():
        target = out / folder_name(name, version)
        if (target / "package").exists():
            print(f"have {target.name}")
            written.append(target)
            continue
        with urllib.request.urlopen(tarball_url(name, version), timeout=60) as response:
            payload = response.read()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            archive.extractall(target, filter="data")  # refuses paths that escape target
        print(f"got  {target.name} ({len(payload) / 1e6:.1f} MB)")
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw/extracted"))
    download(parser.parse_args().out)


if __name__ == "__main__":
    main()
