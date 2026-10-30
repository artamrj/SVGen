# svgen-0.3-mono

LoRA adapter (fp16, 35 MB) for [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), trained with Apple MLX to draw black-and-white 24×24 line icons as SVG.

```bash
python -m svgen.generate "camera" --adapter models/svgen-0.3-mono --out camera.svg
```

* Base model: Qwen2.5-Coder-0.5B-Instruct (Apache-2.0)
* Training data: Lucide (ISC), Tabler Icons, Iconoir, Heroicons (MIT)
* Details: [model card](../../docs/model_card_svgen_0.3.md)
