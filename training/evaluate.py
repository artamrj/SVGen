from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SVGen MVP evaluation scaffold.")
    parser.add_argument("--config", default="configs/generation_default.json")
    parser.add_argument("--out", default="reports/eval_svgen_0.1.json")
    args = parser.parse_args()

    config = load_config(args.config)
    report = {
        "model": config.get("model", "SVGen 0.2"),
        "valid_svg_rate": None,
        "render_success_rate": None,
        "status": "placeholder",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
