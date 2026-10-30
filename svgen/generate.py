"""Text to black-and-white SVG icon.

    python -m svgen.generate "rocket launching" --out rocket.svg

Pipeline: prompt -> fine-tuned LLM samples N candidates in one batch -> each is
parsed by the strict SVG validator -> valid ones are rendered and ranked with
CLIP (if available) -> the best one is wrapped in a safe <svg> and written.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from svgen.prompting import messages
from svgen.svg import SvgError, parse_body, salvage_body, wrap_body

DEFAULT_BASE = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16"
DEFAULT_ADAPTER = "models/svgen-0.3-mono"


@dataclass
class Candidate:
    raw: str
    body: str
    valid: bool
    score: float = 0.0


@lru_cache(maxsize=2)
def load_model(base: str = DEFAULT_BASE, adapter: str | None = DEFAULT_ADAPTER):
    from mlx_lm import load

    adapter_path = adapter if adapter and Path(adapter).exists() else None
    return load(base, adapter_path=adapter_path)


def sample(prompt: str, n: int = 4, temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 700,
           base: str = DEFAULT_BASE, adapter: str | None = DEFAULT_ADAPTER, seed: int | None = None) -> list[Candidate]:
    import mlx.core as mx
    from mlx_lm import batch_generate
    from mlx_lm.sample_utils import make_sampler

    if seed is not None:
        mx.random.seed(seed)
    model, tokenizer = load_model(base, adapter)
    ids = tokenizer.apply_chat_template(messages(prompt), add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(ids, add_special_tokens=False)
    sampler = make_sampler(temp=temperature, top_p=top_p) if temperature > 0 else None
    response = batch_generate(model, tokenizer, [ids] * n, max_tokens=max_tokens, sampler=sampler)
    candidates = []
    for text in response.texts:
        try:
            body = "".join(shape.to_xml() for shape in parse_body(text.strip()))
            candidates.append(Candidate(text, body, True))
        except SvgError:
            # Output cut off or a broken last shape: keep the complete shapes before it.
            body = salvage_body(text)
            candidates.append(Candidate(text, body, False))
    return candidates


def _heuristic(body: str) -> float:
    """Prefer drawings with a sensible amount of detail when CLIP is unavailable."""
    return min(len(body), 600) / 600


def rank(prompt: str, candidates: list[Candidate], use_clip: bool = True) -> list[Candidate]:
    usable = [c for c in candidates if c.body]
    if not usable:
        return []
    if use_clip:
        from svgen.rerank import clip_available, clip_scores
        from svgen.render import render_png

        if clip_available():
            images = [render_png(wrap_body(c.body)) for c in usable]
            for candidate, score in zip(usable, clip_scores(prompt, images)):
                candidate.score = score + (0.02 if candidate.valid else 0.0)
            return sorted(usable, key=lambda c: c.score, reverse=True)
    for candidate in usable:
        candidate.score = _heuristic(candidate.body) + (1.0 if candidate.valid else 0.0)
    return sorted(usable, key=lambda c: c.score, reverse=True)


def text_to_svg(prompt: str, n: int = 4, use_clip: bool = True, stroke_width: float = 2.0, **kwargs) -> str:
    ranked = rank(prompt, sample(prompt, n=n, **kwargs), use_clip=use_clip)
    if not ranked:
        raise RuntimeError("The model produced no usable shapes. Try again or rephrase the prompt.")
    return wrap_body(ranked[0].body, stroke_width=stroke_width)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a black-and-white SVG icon from text.")
    parser.add_argument("prompt")
    parser.add_argument("--out", type=Path, help="write the SVG here (default: print it)")
    parser.add_argument("-n", "--candidates", type=int, default=4, help="samples to draw before picking the best")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--stroke-width", type=float, default=2.0)
    parser.add_argument("--no-clip", action="store_true", help="skip CLIP reranking")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    args = parser.parse_args(argv)

    svg = text_to_svg(
        args.prompt, n=args.candidates, use_clip=not args.no_clip, stroke_width=args.stroke_width,
        temperature=args.temperature, base=args.base, adapter=args.adapter, seed=args.seed,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(svg + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
