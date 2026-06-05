# Model Card: SVGen 0.1

## Overview

SVGen 0.1 is the initial scaffold milestone for a text-to-SVG generation project. It defines the expected training, export, validation, and runtime interfaces.

## Intended Use

- Generate small icon-like SVGs from text prompts.
- Support developer workflows through a CLI and API.
- Provide a foundation for future trained checkpoints.

## Current Limitations

- No production model weights are included.
- Rust generation is a deterministic placeholder.
- The sample dataset is only for tests and smoke checks.

## Dataset Limitations

The MVP sample data is intentionally tiny. Future datasets should be sourced from permissively licensed icon sets, normalized, captioned, validated, and split into train/validation/test sets.

## Safety Rules

Generated SVG must be validated before use. The runtime blocks scripts, external links, embedded images, styles, filters, masks, and unsupported tags.

## SVG Restrictions

Allowed tags for the MVP runtime are `svg`, `path`, `circle`, `rect`, `line`, `polyline`, `polygon`, and `g`.

## Superseded By

SVGen 0.2 adds validated Lucide-derived training data, tokenizer artifacts, a tiny trainable PyTorch model path, export artifacts, and stricter Rust validation.
