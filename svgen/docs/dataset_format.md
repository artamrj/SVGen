# Dataset Format

SVGen uses JSON Lines. Each line contains one text-to-SVG pair.

```json
{"text":"minimal black home icon","svg":"<svg viewBox=\"0 0 24 24\"><path d=\"M3 12 L12 3 L21 12 V21 H3 Z\"/></svg>"}
```

## Required Fields

- `text`: non-empty natural-language prompt or caption.
- `svg`: non-empty SVG string.

## MVP Normalization Rules

- Prefer `viewBox="0 0 24 24"`.
- Keep SVG small and icon-like.
- Remove scripts, external references, filters, masks, styles, embedded images, and unsupported tags.
- Use simple geometry where possible: `path`, `circle`, `rect`, `line`, `polyline`, `polygon`, and `g`.

## Splits

Future datasets should be split into train, validation, and test JSONL files. The MVP scaffold uses `data/sample/dataset.jsonl` for all commands.
