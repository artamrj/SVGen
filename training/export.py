from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MODEL_NAME = "SVGen 0.2.6 Easy Python Inference"


def export_artifacts(
    checkpoint: str | Path = "checkpoints/svgen-0.2.5",
    out: str | Path = "exports/svgen-0.2.5",
    *,
    allow_placeholder: bool = False,
    tokenizer: str | Path = "tokenizer/tokenizer.json",
) -> Path:
    checkpoint_path = Path(checkpoint) / "checkpoint.pt"
    if not checkpoint_path.exists():
        if not allow_placeholder:
            raise FileNotFoundError(
                f"missing checkpoint: {checkpoint_path}. "
                "Train first or pass --allow-placeholder for scaffold/debug exports."
            )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        try:
            from safetensors.torch import save_file
            import torch
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to export safetensors artifacts") from exc
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        save_file(checkpoint["model_state_dict"], out_dir / "model.safetensors")
        config = {
            "model": MODEL_NAME,
            "status": "trained",
            "checkpoint": str(checkpoint_path),
            "model_config": checkpoint.get("config"),
            "runtime_inference": "unavailable",
        }
    else:
        (out_dir / "model.safetensors").write_bytes(b"")
        config = {
            "model": MODEL_NAME,
            "status": "placeholder",
            "missing_checkpoint": str(checkpoint_path),
            "runtime_inference": "unavailable",
        }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tokenizer_src = Path(tokenizer)
    if tokenizer_src.exists():
        shutil.copyfile(tokenizer_src, out_dir / "tokenizer.json")
    else:
        (out_dir / "tokenizer.json").write_text(json.dumps({"status": "placeholder"}, indent=2) + "\n", encoding="utf-8")
    (out_dir / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.9}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SVGen 0.2.6 artifacts.")
    parser.add_argument("--checkpoint", default="checkpoints/svgen-0.2.5")
    parser.add_argument("--out", default="exports/svgen-0.2.5")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--allow-placeholder", action="store_true", help="Write placeholder artifacts when checkpoint.pt is missing.")
    args = parser.parse_args()

    try:
        out_dir = export_artifacts(args.checkpoint, args.out, allow_placeholder=args.allow_placeholder, tokenizer=args.tokenizer)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    print(f"Wrote export artifacts to {out_dir}")


if __name__ == "__main__":
    main()
