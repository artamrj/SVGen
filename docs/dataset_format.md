# Dataset Format (0.3)

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

Files: `train.jsonl`, `valid.jsonl`, `test.jsonl`, `test_icons.json` (held-out icons with their real SVG, used by `svgen.evaluate`) and `report.json` (counts and rejections).

The 0.2 format (`{"text": ..., "svg": ...}`, used by `data/train_1000.jsonl` and the legacy `training/` package) is described in `docs/legacy/`.
