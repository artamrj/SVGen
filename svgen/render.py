"""Render SVG to pixels with rsvg-convert (brew install librsvg)."""

from __future__ import annotations

import io
import shutil
import subprocess

from PIL import Image


def renderer_available() -> bool:
    return shutil.which("rsvg-convert") is not None


def render_png(svg: str, size: int = 224) -> Image.Image:
    """Render black ink on white. The SVG uses currentColor, so pin it to black."""
    svg = svg.replace("currentColor", "#000")
    result = subprocess.run(
        ["rsvg-convert", "-w", str(size), "-h", str(size), "-b", "white"],
        input=svg.encode("utf-8"),
        capture_output=True,
        check=True,
        timeout=20,
    )
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def contact_sheet(images: list[Image.Image], labels: list[str] | None = None, cols: int = 8, cell: int = 128) -> Image.Image:
    from PIL import ImageDraw

    rows = (len(images) + cols - 1) // cols
    label_h = 16 if labels else 0
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, image in enumerate(images):
        x, y = (k % cols) * cell, (k // cols) * (cell + label_h)
        sheet.paste(image.resize((cell - 8, cell - 8)), (x + 4, y + 4))
        if labels:
            draw.text((x + 4, y + cell), labels[k][:20], fill="black")
    return sheet
