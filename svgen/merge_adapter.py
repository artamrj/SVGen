"""Merge a PEFT LoRA adapter into its base model and save a full bf16 model.

    python -m svgen.merge_adapter --adapter models/svgen-0.4-lora --out runs/svgen-1.5b/merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RELEASE_URL = "https://github.com/artamrj/SVGen/releases/download/v0.4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, default=Path("models/svgen-0.4-lora"))
    parser.add_argument("--base", help="defaults to base_model_name_or_path from adapter_config.json")
    parser.add_argument("--out", type=Path, default=Path("runs/svgen-1.5b/merged"))
    args = parser.parse_args()

    weights = args.adapter / "adapter_model.safetensors"
    if not weights.exists():
        # GitHub limits files to 100 MB, so the 141 MB adapter lives on the v0.4 release.
        import urllib.request

        url = f"{RELEASE_URL}/adapter_model.safetensors"
        print(f"downloading adapter weights from {url}", flush=True)
        urllib.request.urlretrieve(url, weights)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base or json.loads((args.adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(args.out)
    print(f"merged {base} + {args.adapter} -> {args.out}")


if __name__ == "__main__":
    main()
