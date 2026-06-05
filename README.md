# SVGen

SVGen is a text-to-SVG model project. SVGen 0.2.6 Easy Python Inference adds the first user-friendly generation path on top of validated Lucide-derived data, tokenizer artifacts, tiny PyTorch training, export artifacts, and the Rust-safe fallback runtime.

Display name: `SVGen 0.2.6 Easy Python Inference`

Artifact path: `exports/svgen-0.2.5/`

## Architecture

```text
User prompt
   -> Python generation CLI
   -> Tokenizer
   -> Exported PyTorch model
   -> Generated SVG tokens
   -> Python SVG validator
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

The easiest workflow is:

```bash
make setup
make train
make svg PROMPT="running dog icon" OUT=reports/dog.svg
```

Open `reports/dog.svg` in a browser or image viewer.

Install Python dev dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
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
make train
```

Generate with Python inference:

```bash
.venv/bin/python -m training.generate \
  --model exports/svgen-0.2.5 \
  --prompt "running dog icon" \
  --out reports/dog.svg
```

Or use the Make shortcut:

```bash
make svg PROMPT="running dog icon" OUT=reports/dog.svg
```

The Rust CLI remains fallback-only until Rust inference lands:

```bash
cd runtime
cargo run -- generate --model ../exports/svgen-0.2.5 --prompt "minimal rocket icon" --out ../reports/rocket.svg
```

Until Rust inference is implemented, the CLI writes a valid fallback SVG and prints:

```text
Using fallback placeholder because no trained model is loaded.
```

## Safety Rules

The Rust validator rejects unsafe tags, unsafe attributes, namespace-prefixed XML tricks, external references, oversized SVGs, excessive depth, excessive child counts, long path data, and very large numeric values.

## MVP Scope

SVGen 0.2.6 proves a real Python inference workflow, not production-quality icon generation. Output quality may be rough because the model is tiny, but generated SVG is validated before writing. The Rust runtime still emits deterministic fallback output until Rust inference lands in a later release.

## Dataset Format

Each line is a JSON object:

```json
{"text":"minimal black home icon","svg":"<svg viewBox=\"0 0 24 24\"><path d=\"M3 12 L12 3 L21 12 V21 H3 Z\"/></svg>"}
```

See `docs/dataset_format.md` for details.
