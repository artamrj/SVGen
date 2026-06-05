from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SVGen 0.2 artifacts.")
    parser.add_argument("--checkpoint", default="checkpoints/mvp")
    parser.add_argument("--out", default="exports/svgen-0.2")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) / "checkpoint.pt"
    if checkpoint_path.exists():
        try:
            from safetensors.torch import save_file
            import torch
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to export safetensors artifacts") from exc
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        save_file(checkpoint["model_state_dict"], out_dir / "model.safetensors")
        config = {"model": "SVGen 0.2", "status": "trained", "checkpoint": str(checkpoint_path), "model_config": checkpoint.get("config")}
    else:
        (out_dir / "model.safetensors").write_bytes(b"")
        config = {"model": "SVGen 0.2", "status": "placeholder", "missing_checkpoint": str(checkpoint_path)}

    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tokenizer_src = Path("tokenizer/tokenizer.json")
    if tokenizer_src.exists():
        shutil.copyfile(tokenizer_src, out_dir / "tokenizer.json")
    else:
        (out_dir / "tokenizer.json").write_text(json.dumps({"status": "placeholder"}, indent=2) + "\n", encoding="utf-8")
    (out_dir / "generation_config.json").write_text(json.dumps({"max_new_tokens": 384, "temperature": 0.7}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote export artifacts to {out_dir}")


if __name__ == "__main__":
    main()
