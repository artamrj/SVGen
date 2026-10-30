"""Pick the candidate whose rendering best matches the prompt, using a CLIP-style model.

Uses SigLIP2 ViT-B/16 through open_clip on CPU. On held-out line icons it matches
image to text 88% of the time (MobileCLIP2 and CLIP ViT-B/32 did worse). It is optional: if torch/open_clip
or rsvg-convert are missing, generation falls back to a simple shape heuristic.
"""

from __future__ import annotations

from functools import lru_cache

CLIP_MODEL = "hf-hub:timm/ViT-B-16-SigLIP2-256"


@lru_cache(maxsize=1)
def _load():
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    torch.set_num_threads(2)  # stay light on a busy laptop
    return model, preprocess, tokenizer


def clip_available() -> bool:
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    from svgen.render import renderer_available

    return renderer_available()


def clip_scores(prompt: str, images) -> list[float]:
    import torch

    model, preprocess, tokenizer = _load()
    text = tokenizer([f"a black line icon of {prompt}"])
    with torch.no_grad():
        image_features = model.encode_image(torch.stack([preprocess(image) for image in images]))
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).squeeze(-1).tolist()
