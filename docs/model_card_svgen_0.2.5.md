# Model Card: SVGen 0.2.5 Tiny Generator

SVGen 0.2.5 Tiny Generator is a truthful milestone for the SVGen text-to-SVG pipeline. It includes a validated 1000-record Lucide-derived dataset, tokenizer training, a tiny PyTorch model training path, stricter export behavior, and a Rust-safe fallback output path.

## Intended Use

This version is intended for local pipeline validation and runtime safety testing. It is not a production text-to-SVG model.

## Runtime Behavior

The Rust runtime does not perform neural inference in 0.2.5. `svgen generate` inspects model artifacts, writes a valid fallback SVG, and prints:

```text
Using fallback placeholder because no trained model is loaded.
```

This warning is intentional. It prevents fallback output from being mistaken for prompt-conditioned generation.

## Dataset

The default dataset is `data/train_1000.jsonl`. It is generated from Lucide SVGs under the ISC license and is expected to validate at 1000 total records with at least 950 valid records.

## Artifacts

Default checkpoint path: `checkpoints/svgen-0.2.5/`

Default export path: `exports/svgen-0.2.5/`

Export fails by default when `checkpoint.pt` is missing. Use `--allow-placeholder` only for scaffold or debug exports.
