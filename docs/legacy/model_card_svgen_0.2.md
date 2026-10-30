# Model Card: SVGen 0.2

## Overview

SVGen 0.2 is a training MVP for text-to-SVG generation. It includes a validated 1000-record Lucide-derived dataset, tokenizer training, a tiny PyTorch model path, export artifacts, and a Rust-safe output path.

## Intended Use

- Prove the text-to-SVG training and export pipeline.
- Generate placeholder SVGs through the Rust CLI while enforcing strict SVG validation.
- Provide a foundation for real Rust inference in SVGen 0.3.

## Current Limitations

- The tiny model is for pipeline testing, not icon quality.
- Rust generation is still deterministic placeholder output.
- Dataset captions are filename-derived and intentionally simple.

## Dataset

The default 0.2 dataset is generated from Lucide SVGs under the ISC license. Source metadata attributes such as `class` are stripped because the runtime validator blocks them.

## Safety Rules

The runtime allows only a small SVG subset and rejects scripts, external references, embedded images, style/class/id attributes, namespace-prefixed tricks, large numeric values, excessive depth, excessive child counts, and oversized path data.
