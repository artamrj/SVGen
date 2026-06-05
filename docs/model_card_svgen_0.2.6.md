# Model Card: SVGen 0.2.6 Easy Python Inference

SVGen 0.2.6 Easy Python Inference adds the first real prompt-to-SVG generation entrypoint for the project. It uses the exported tiny PyTorch model, tokenizer, and generation config to produce SVG text, then validates the SVG before writing it.

## Intended Use

This version is intended for local experimentation with the full text-to-SVG pipeline:

```bash
make setup
make train
make svg PROMPT="running dog icon" OUT=reports/dog.svg
```

Output quality may be rough because the model is intentionally small and trained on a compact icon dataset.

## Runtime Behavior

Python generation runs through:

```bash
python -m training.generate --model exports/svgen-0.2.5 --prompt "running dog icon" --out reports/dog.svg
```

The command loads exported safetensors weights, runs autoregressive token generation, extracts the first complete SVG document, validates it against the safe SVG subset, and writes the file only if validation passes.

The Rust runtime remains fallback-only in this release.

## Dataset and Artifacts

The default dataset is `data/train_1000.jsonl`. It is expected to validate at 1000 total records with at least 950 valid records.

Default checkpoint path: `checkpoints/svgen-0.2.5/`

Default export path: `exports/svgen-0.2.5/`
