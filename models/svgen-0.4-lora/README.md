# svgen-0.4-lora

PEFT LoRA adapter for `Qwen/Qwen2.5-Coder-1.5B-Instruct`, trained on Google Colab. The 141 MB weights file is attached to the [v0.4 release](https://github.com/artamrj/SVGen/releases/tag/v0.4) (too big for git) and is downloaded on first use. See the [model card](../../docs/model_card_svgen_0.4.md).

Build the 8-bit MLX model for a Mac:

```bash
make build-1.5b          # merges this adapter into the base model, then converts
make svg PROMPT="star" BASE=models/svgen-1.5b ADAPTER=none OUT=star.svg
```
