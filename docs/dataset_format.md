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

## SVGen 0.2 Data

`data/train_1000.jsonl` is generated from Lucide SVGs. Prompts are generated from filenames:

```text
shopping-cart.svg -> minimal outline shopping cart icon
```

Generate and validate it with:

```bash
npm pack lucide-static
python3 -m training.build_lucide_dataset --tarball lucide-static-1.17.0.tgz --out data/train_1000.jsonl --limit 1000
python3 -m training.validate_dataset --input data/train_1000.jsonl --report reports/dataset_report.json
```

Future datasets should be split into train, validation, and test JSONL files.
