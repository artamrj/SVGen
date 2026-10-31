# Dataset Format

`make data` writes chat-format JSON Lines to `data/processed/`, the format `mlx_lm lora` expects:

```json
{"messages": [
  {"role": "system", "content": "Draw a 24x24 black-and-white line icon as SVG shapes."},
  {"role": "user", "content": "line icon of a lock"},
  {"role": "assistant", "content": "<rect x=\"5\" y=\"11\" width=\"14\" height=\"10\" rx=\"2\"/><path d=\"M8 11V7A4 4 0 0 1 16 7V11\"/>"}
]}
```

The assistant answer is the **canonical body** (see `svgen/svg.py`):

- only `path`, `circle`, `rect`, `line` and `ellipse`; polylines and polygons become paths
- path data in absolute commands (`M L H V C S Q T A Z`), numbers rounded to 0.1
- fixed attribute order; the only styling allowed is `fill="currentColor"`
- coordinates within ±48 of the 24×24 canvas; no transforms, groups, or external references

## Sources

| Source | License | Grid | Icons kept |
|---|---|---|---|
| Lucide, Tabler (outline), Iconoir (regular), Heroicons (24/outline) | ISC / MIT | 24 | 8,873 (the v0.3/v0.4 data) |
| IconPark outline (ByteDance) | Apache-2.0 | 48 | 2,436 |
| Hugeicons (free) | MIT | 24 | 5,843 |
| MingCute | Apache-2.0 | 24 | 1,294 |
| TDesign (Tencent) | MIT | 24 | 1,253 |
| ProIcons, Majesticons, Meteor, Akar, Humbleicons | MIT | 24 | 1,774 |

21,473 unique icons, ~12.5k concepts, ~71k training rows. The extra sets come from Iconify's npm packages (`@iconify-json/*`, pinned in `svgen/download.py`). `svgen.svg.import_icon` scales them to 24×24, flattens `<g>` groups, and keeps only line art. It rejects filled paths, transparency, dashes, transforms, animation, and stroke widths outside 1–2.5. About 4,000 icons are rejected this way; `report.json` lists the reasons.

Datasets such as SVG-1M, SVGX and MMSVG are licensed for non-commercial use only (CC BY-NC), so they are not used.

`make data SOURCES=classic` rebuilds the exact v0.3/v0.4 data.

## Held-out split

Validation and test concepts are frozen in `data/heldout/`: `split.json` lists them and `test_icons.json` holds the test icons in the order that `svgen.evaluate` samples from. Only the four original packs fill valid and test, so eval scores stay comparable with v0.3 and v0.4. Icons from other sources that belong to a held-out concept are left out entirely, so nothing leaks into training. `--fresh-split` draws a new random split instead.

## Files

Files: `train.jsonl`, `valid.jsonl`, `test.jsonl`, `test_icons.json` (held-out icons with their real SVG, used by `svgen.evaluate`) and `report.json` (counts and rejections).

The 0.2 format (`{"text": ..., "svg": ...}`, used by `data/train_1000.jsonl` and the legacy `training/` package) is described in `docs/legacy/`.
