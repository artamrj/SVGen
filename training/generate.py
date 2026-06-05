from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.model import build_tiny_language_model, require_torch
from training.tokenizer import SvgTokenizer
from training.validate_dataset import validate_svg

DEFAULT_MODEL_DIR = Path("exports/svgen-0.2.5")
REQUIRED_ARTIFACTS = ("model.safetensors", "config.json", "tokenizer.json", "generation_config.json")
SVG_START_PATTERN = re.compile(r"<svg\b", re.IGNORECASE)
SVG_END_PATTERN = re.compile(r"</svg>", re.IGNORECASE)


class GenerationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GenerationError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GenerationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise GenerationError(f"JSON artifact must be an object: {path}")
    return payload


def require_export_artifacts(model_dir: str | Path) -> Path:
    export_dir = Path(model_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (export_dir / name).is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise GenerationError(f"missing export artifacts in {export_dir}: {missing_list}. Run make train first.")

    config = load_json(export_dir / "config.json")
    if config.get("status") != "trained":
        raise GenerationError(f"export is not trained: {export_dir}. Run make train first.")
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise GenerationError(f"export config is missing model_config: {export_dir / 'config.json'}")
    return export_dir


def extract_svg(text: str) -> str:
    starts = [match.start() for match in SVG_START_PATTERN.finditer(text)]
    ends = [match.end() for match in SVG_END_PATTERN.finditer(text)]
    last_error: Exception | None = None
    for start in starts:
        for end in ends:
            if end <= start:
                continue
            svg = text[start:end].strip()
            try:
                validate_svg(svg)
            except Exception as exc:
                last_error = exc
                continue
            return svg
    if last_error is not None:
        raise GenerationError(f"model output was not a valid safe SVG: {last_error}") from last_error
    raise GenerationError("model output did not contain a complete <svg>...</svg> document")


def encode_generation_prefix(tokenizer: SvgTokenizer, prompt: str) -> list[int]:
    unk = tokenizer.vocab["<UNK>"]
    tokens = tokenizer.tokenize(f"<PROMPT> {prompt} <SVG>")
    return [tokenizer.vocab["<BOS>"], *[tokenizer.vocab.get(token, unk) for token in tokens]]


def sample_next_token(logits: Any, *, temperature: float, top_p: float | None, torch: Any) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        filtered_probs = sorted_probs[keep]
        filtered_indices = sorted_indices[keep]
        filtered_probs = filtered_probs / filtered_probs.sum()
        sampled = torch.multinomial(filtered_probs, num_samples=1)
        return int(filtered_indices[sampled].item())
    return int(torch.multinomial(probs, num_samples=1).item())


def load_model(export_dir: Path, tokenizer: SvgTokenizer, device: str):
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to generate SVGs with exported safetensors artifacts") from exc

    torch, _nn = require_torch()
    config = load_json(export_dir / "config.json")
    model_config = config["model_config"]
    context = int(model_config["context"])
    model = build_tiny_language_model(
        vocab_size=len(tokenizer.vocab),
        context=context,
        hidden=int(model_config["hidden"]),
        layers=int(model_config["layers"]),
        heads=int(model_config["heads"]),
    )
    weights = load_file(export_dir / "model.safetensors", device=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return torch, model, context


def generate_svg(
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    prompt: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> str:
    export_dir = require_export_artifacts(model_dir)
    generation_config = load_json(export_dir / "generation_config.json")
    tokenizer = SvgTokenizer.load(export_dir / "tokenizer.json")
    torch, model, context = load_model(export_dir, tokenizer, device)

    torch.manual_seed(seed)
    max_tokens = int(max_new_tokens if max_new_tokens is not None else generation_config.get("max_new_tokens", 384))
    temp = float(temperature if temperature is not None else generation_config.get("temperature", 0.7))
    nucleus = float(top_p if top_p is not None else generation_config.get("top_p", 0.9))

    ids = encode_generation_prefix(tokenizer, prompt)
    prefix_len = len(ids)
    eos_id = tokenizer.vocab["<EOS>"]
    with torch.inference_mode():
        for _ in range(max_tokens):
            window = ids[-context:]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1]
            next_id = sample_next_token(logits, temperature=temp, top_p=nucleus, torch=torch)
            ids.append(next_id)
            if next_id == eos_id:
                break

    generated_text = tokenizer.decode(ids[prefix_len:])
    return extract_svg(generated_text)


def write_svg(svg: str, out: str | Path) -> Path:
    try:
        validate_svg(svg)
    except Exception as exc:
        raise GenerationError(f"refusing to write unsafe SVG: {exc}") from exc
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SVG with the SVGen Python inference path.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="Exported model directory")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--out", required=True, help="Output SVG path")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        svg = generate_svg(
            model_dir=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            device=args.device,
        )
        out_path = write_svg(svg, args.out)
    except GenerationError as exc:
        parser.error(str(exc))
    print(f"Wrote SVG to {out_path}")


if __name__ == "__main__":
    main()
