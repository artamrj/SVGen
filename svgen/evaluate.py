"""Evaluate SVGen on held-out icon concepts and free-form prompts.

    python -m svgen.evaluate --limit 64

Writes reports/eval.json and reports/eval_sheet.png (generated icons) and
reports/eval_reference.png (the real held-out icons, for comparison).

Metrics:
  valid_rate    share of raw samples that pass the strict SVG parser
  usable_rate   share of prompts that end with at least one drawable icon
  clip_best     mean CLIP similarity of the chosen icon to its prompt
  clip_ref      the same score for the real held-out icon (an upper reference)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from svgen.generate import DEFAULT_ADAPTER, DEFAULT_BASE, rank, sample
from svgen.render import contact_sheet, render_png
from svgen.rerank import clip_available, clip_scores
from svgen.svg import wrap_body

FREE_PROMPTS = [
    "rocket launching", "coffee cup with steam", "cat face", "house with a chimney", "shopping cart",
    "lightning bolt", "umbrella", "music note", "camera", "paper plane", "magnifying glass", "battery half full",
    "cloud with rain", "lock", "trash can", "bicycle", "envelope", "lightbulb", "star", "tree",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, default=Path("data/processed/test_icons.json"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("-n", "--candidates", type=int, default=4)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = json.loads(args.test.read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(items)
    items = items[: args.limit]
    prompts = [item["prompt"] for item in items] + FREE_PROMPTS
    use_clip = clip_available()

    start = time.time()
    raw_valid = raw_total = usable = 0
    chosen_scores: list[float] = []
    images, labels, rows = [], [], []
    for k, prompt in enumerate(prompts):
        candidates = sample(prompt, n=args.candidates, base=args.base, adapter=args.adapter, seed=args.seed + k)
        raw_total += len(candidates)
        raw_valid += sum(c.valid for c in candidates)
        ranked = rank(prompt, candidates, use_clip=use_clip)
        if ranked:
            usable += 1
            best = ranked[0]
            svg = wrap_body(best.body)
            images.append(render_png(svg, 128))
            if use_clip:
                chosen_scores.append(best.score)
            rows.append({"prompt": prompt, "svg": svg, "valid": best.valid, "score": round(best.score, 4)})
        else:
            from PIL import Image

            images.append(Image.new("RGB", (128, 128), "#fdd"))
            rows.append({"prompt": prompt, "svg": None})
        labels.append(prompt)
        print(f"[{k + 1}/{len(prompts)}] {prompt}: {'ok' if ranked else 'FAILED'}", flush=True)

    reference_score = None
    if use_clip:
        ref_images = [render_png(wrap_body(item["body"])) for item in items]
        ref_scores = clip_scores_per_prompt([item["prompt"] for item in items], ref_images)
        reference_score = sum(ref_scores) / len(ref_scores)
        sheet = contact_sheet([render_png(wrap_body(item["body"]), 128) for item in items], [item["prompt"] for item in items])
        sheet.save(args.out / "eval_reference.png")

    held_out = len(items)
    summary = {
        "base": args.base,
        "adapter": args.adapter,
        "prompts": len(prompts),
        "candidates_per_prompt": args.candidates,
        "valid_rate": round(raw_valid / max(1, raw_total), 4),
        "usable_rate": round(usable / len(prompts), 4),
        "clip_best_heldout": round(sum(chosen_scores[:held_out]) / max(1, len(chosen_scores[:held_out])), 4) if use_clip else None,
        "clip_best_free": round(sum(chosen_scores[held_out:]) / max(1, len(chosen_scores[held_out:])), 4) if use_clip else None,
        "clip_ref_heldout": round(reference_score, 4) if reference_score is not None else None,
        "seconds": round(time.time() - start, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps({"summary": summary, "samples": rows}, indent=1) + "\n", encoding="utf-8")
    contact_sheet(images, labels).save(args.out / "eval_sheet.png")
    print(json.dumps(summary, indent=2))


def clip_scores_per_prompt(prompts: list[str], images) -> list[float]:
    return [clip_scores(prompt, [image])[0] for prompt, image in zip(prompts, images)]


if __name__ == "__main__":
    main()
