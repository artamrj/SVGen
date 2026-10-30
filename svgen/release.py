"""Package a trained LoRA checkpoint as a small, committed model folder.

    python -m svgen.release --checkpoint checkpoints/svgen-0.3-mono-stage3 --out models/svgen-0.3-mono

The adapter weights are stored in float16 (half the size, no visible quality
loss for LoRA deltas). Pass --step 1500 to release an intermediate checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def release(checkpoint: Path, out: Path, step: int | None = None) -> Path:
    import mlx.core as mx

    name = f"{step:07d}_adapters.safetensors" if step else "adapters.safetensors"
    weights = mx.load(str(checkpoint / name))
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), {k: v.astype(mx.float16) for k, v in weights.items()})
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    # Keep only what is needed to load the adapter; drop local training paths.
    for key in ("data", "adapter_path", "resume_adapter_file", "wandb", "report_to", "project_name"):
        config.pop(key, None)
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/svgen-0.3-mono-stage3"))
    parser.add_argument("--out", type=Path, default=Path("models/svgen-0.3-mono"))
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    out = release(args.checkpoint, args.out, args.step)
    size = (out / "adapters.safetensors").stat().st_size / 1e6
    print(f"released {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
