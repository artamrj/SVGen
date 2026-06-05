# SVGen

SVGen is a text-to-SVG model project. The MVP scaffold provides the project layout, dataset format, tokenizer wrapper, training/evaluation/export entrypoints, Rust runtime skeleton, SVG validation, and tests for the first development milestone.

Display name: `SVGen 0.1`

Artifact path: `exports/svgen-0.1/`

## Architecture

```text
User prompt
   -> Rust CLI / API
   -> Tokenizer
   -> Exported model inference
   -> Generated SVG tokens
   -> Rust SVG validator / postprocess
   -> Final .svg output
```

## Layout

```text
svgen/
  training/      Python/PyTorch training scaffold
  runtime/       Rust CLI/API/runtime scaffold
  configs/       Training, model, and generation configs
  schemas/       JSON Schemas for public records and API payloads
  data/          Sample and future training data
  tokenizer/     Tokenizer artifacts
  checkpoints/   Training checkpoints
  exports/       Versioned exported model artifacts
  docs/          Dataset, coding standards, and model card
  reports/       Evaluation reports
  tests/         Python tests
```

## Quickstart

Run Python checks:

```bash
cd svgen
python -m pytest
```

Load the MVP dataset:

```bash
python -m training.dataset data/sample/dataset.jsonl
```

Run placeholder training/evaluation/export:

```bash
python -m training.train --config configs/train_mvp.yaml
python -m training.evaluate --config configs/generation_default.json
python -m training.export --out exports/svgen-0.1
```

Run Rust tests and the CLI:

```bash
cd runtime
cargo test
cargo run -- generate --model ../exports/svgen-0.1 --prompt "minimal rocket icon" --out ../reports/rocket.svg
```

## MVP Scope

This repository is a scaffold. It does not include trained production weights. The Rust runtime emits deterministic placeholder SVGs until real `safetensors` model loading and generation are implemented.

## Dataset Format

Each line is a JSON object:

```json
{"text":"minimal black home icon","svg":"<svg viewBox=\"0 0 24 24\"><path d=\"M3 12 L12 3 L21 12 V21 H3 Z\"/></svg>"}
```

See `docs/dataset_format.md` for details.
