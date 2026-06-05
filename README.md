# SVGen

SVGen is a text-to-SVG model project. SVGen 0.2 moves the scaffold into a real training MVP: validated Lucide-derived data, tokenizer artifacts, a tiny PyTorch training path, export artifacts, and a Rust-safe output path.

Display name: `SVGen 0.2`

Artifact path: `exports/svgen-0.2/`

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
training/      Python/PyTorch training, validation, tokenizer, export tools
runtime/       Rust CLI/API/runtime scaffold
configs/       Training, model, and generation configs
schemas/       JSON Schemas for public records and API payloads
data/          Sample data and train_1000.jsonl
tokenizer/     Tokenizer artifacts
checkpoints/   Training checkpoints
exports/       Versioned exported model artifacts
docs/          Dataset, coding standards, and model cards
reports/       Evaluation and dataset reports
tests/         Python tests
```

## Quickstart

Install Python dev dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

Run checks:

```bash
make check
```

Generate the Lucide MVP dataset:

```bash
npm pack lucide-static
python3 -m training.build_lucide_dataset --tarball lucide-static-1.17.0.tgz --out data/train_1000.jsonl --limit 1000
python3 -m training.validate_dataset --input data/train_1000.jsonl --report reports/dataset_report.json
```

Train tokenizer, train tiny model, and export:

```bash
python3 -m training.train_tokenizer --input data/train_1000.jsonl --out tokenizer/tokenizer.json --vocab-size 4096
python3 -m training.train --config configs/train_mvp.yaml --data data/train_1000.jsonl --tokenizer tokenizer/tokenizer.json --out checkpoints/mvp
python3 -m training.export --checkpoint checkpoints/mvp --out exports/svgen-0.2
```

Run Rust CLI:

```bash
cd runtime
cargo run -- generate --model ../exports/svgen-0.2 --prompt "minimal rocket icon" --out ../reports/rocket.svg
```

## Safety Rules

The Rust validator rejects unsafe tags, unsafe attributes, namespace-prefixed XML tricks, external references, oversized SVGs, excessive depth, excessive child counts, long path data, and very large numeric values.

## MVP Scope

SVGen 0.2 proves the pipeline, not production quality. The Rust runtime checks real artifacts and validates output, but generation is still deterministic placeholder output until Candle inference lands in SVGen 0.3.

## Dataset Format

Each line is a JSON object:

```json
{"text":"minimal black home icon","svg":"<svg viewBox=\"0 0 24 24\"><path d=\"M3 12 L12 3 L21 12 V21 H3 Z\"/></svg>"}
```

See `docs/dataset_format.md` for details.
