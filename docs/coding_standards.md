# Coding Standards

## Python

- Keep training entrypoints importable and runnable with `python -m`.
- Prefer typed dataclasses for small records and configs.
- Keep dataset validation strict at the boundary.
- Do not hardcode paths that belong in `configs/`.

## Rust

- Keep CLI/API payloads explicit with `serde` types.
- Validate generated SVG before writing or returning it.
- Treat postprocessing as constrained normalization, not silent repair.
- Keep runtime modules small: generation, tokenization, validation, postprocess, and API.

## Artifacts

- Use lowercase paths: `exports/svgen-0.2/`.
- Use display names in user-facing responses: `SVGen 0.1`.
- For new user-facing SVGen 0.2 responses, use `SVGen 0.2`.
