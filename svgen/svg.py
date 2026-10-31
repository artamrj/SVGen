"""Canonical monochrome SVG format used by SVGen.

The model never sees the outer <svg> element. It only writes the shapes inside a
fixed 24x24 line-icon canvas, in one canonical form:

* only path, circle, rect, line and ellipse elements
* path data in absolute commands, numbers rounded to 0.1
* attributes in a fixed order, no styling except an optional fill="currentColor"

`wrap_body` turns those shapes back into a complete, safe SVG document.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from xml.etree import ElementTree

VIEWBOX = "0 0 24 24"
SVG_NS = "http://www.w3.org/2000/svg"

GEOMETRY_ATTRS: dict[str, tuple[str, ...]] = {
    "path": ("d",),
    "circle": ("cx", "cy", "r"),
    "ellipse": ("cx", "cy", "rx", "ry"),
    "rect": ("x", "y", "width", "height", "rx", "ry"),
    "line": ("x1", "y1", "x2", "y2"),
}
# Coordinates outside this range are almost always junk in a 24x24 icon.
COORD_LIMIT = 48.0
MAX_BODY_CHARS = 6000

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARGS_PER_COMMAND = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


class SvgError(ValueError):
    """Raised when an SVG cannot be represented in the canonical format."""


def fmt(value: float, precision: int = 1) -> str:
    """Format a number compactly: 3.0 -> '3', 0.5 -> '.5', -0.5 -> '-.5'."""
    rounded = round(value, precision)
    if rounded == 0:
        return "0"
    text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _check_coord(value: float) -> float:
    if not math.isfinite(value) or abs(value) > COORD_LIMIT:
        raise SvgError(f"coordinate out of range: {value}")
    return value


def _tokenize_path(d: str) -> list[str]:
    """Split path data into commands and numbers, keeping packed arc flags apart."""
    tokens: list[str] = []
    pos = 0
    command = ""
    arg_index = 0
    while pos < len(d):
        char = d[pos]
        if char in " ,\t\r\n":
            pos += 1
            continue
        if char.isalpha():
            if char.upper() not in _ARGS_PER_COMMAND:
                raise SvgError(f"unknown path command: {char}")
            tokens.append(char)
            command = char.upper()
            arg_index = 0
            pos += 1
            continue
        # The 4th and 5th arc arguments are single-digit flags that may be packed ("011").
        if command == "A" and arg_index % 7 in (3, 4):
            if char not in "01":
                raise SvgError("invalid arc flag")
            tokens.append(char)
            arg_index += 1
            pos += 1
            continue
        match = _NUMBER.match(d, pos)
        if not match:
            raise SvgError(f"bad path data near: {d[pos:pos + 12]!r}")
        tokens.append(match.group())
        arg_index += 1
        pos = match.end()
    return tokens


def path_to_absolute(d: str) -> list[tuple[str, list[float]]]:
    """Parse path data and return absolute segments (M L H V C S Q T A Z)."""
    tokens = _tokenize_path(d)
    segments: list[tuple[str, list[float]]] = []
    cx = cy = 0.0  # current point
    sx = sy = 0.0  # subpath start
    i = 0
    command = ""
    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
            if command in "Zz":
                segments.append(("Z", []))
                cx, cy = sx, sy
                continue
        elif not command:
            raise SvgError("path data must start with a command")
        upper = command.upper()
        count = _ARGS_PER_COMMAND[upper]
        if upper == "Z":
            raise SvgError("numbers after Z")
        args = tokens[i : i + count]
        if len(args) != count or any(token.isalpha() for token in args):
            raise SvgError(f"wrong argument count for {command}")
        values = [float(token) for token in args]
        i += count
        relative = command.islower()

        if upper == "M":
            x, y = values
            if relative:
                x, y = x + cx, y + cy
            segments.append(("M", [x, y]))
            cx, cy = sx, sy = x, y
            # Extra coordinate pairs after M are implicit line-tos.
            command = "l" if relative else "L"
        elif upper == "L" or upper == "T":
            x, y = values
            if relative:
                x, y = x + cx, y + cy
            segments.append((upper, [x, y]))
            cx, cy = x, y
        elif upper == "H":
            x = values[0] + (cx if relative else 0)
            segments.append(("H", [x]))
            cx = x
        elif upper == "V":
            y = values[0] + (cy if relative else 0)
            segments.append(("V", [y]))
            cy = y
        elif upper in "CSQ":
            points = values[:]
            if relative:
                points = [v + (cx if k % 2 == 0 else cy) for k, v in enumerate(points)]
            segments.append((upper, points))
            cx, cy = points[-2], points[-1]
        elif upper == "A":
            rx, ry, rotation, large, sweep, x, y = values
            if relative:
                x, y = x + cx, y + cy
            segments.append(("A", [abs(rx), abs(ry), rotation, large, sweep, x, y]))
            cx, cy = x, y
    if not segments or segments[0][0] != "M":
        raise SvgError("path must start with M")
    return segments


def format_path(segments: list[tuple[str, list[float]]]) -> str:
    parts: list[str] = []
    for command, values in segments:
        if command == "Z":
            parts.append("Z")
            continue
        if command == "A":
            rx, ry, rotation, large, sweep, x, y = values
            numbers = [fmt(rx), fmt(ry), fmt(rotation, 0), str(int(large)), str(int(sweep)), fmt(_check_coord(x)), fmt(_check_coord(y))]
        else:
            numbers = [fmt(_check_coord(v)) for v in values]
        parts.append(command + " ".join(numbers))
    return "".join(parts)


def normalize_path(d: str) -> str:
    return format_path(path_to_absolute(d))


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_float(value: str, name: str) -> float:
    value = value.strip().removesuffix("px")
    if not _NUMBER.fullmatch(value):
        raise SvgError(f"bad number for {name}: {value!r}")
    return float(value)


@dataclass(frozen=True)
class Shape:
    tag: str
    attrs: tuple[tuple[str, str], ...]

    def to_xml(self) -> str:
        inner = " ".join(f'{key}="{value}"' for key, value in self.attrs)
        return f"<{self.tag} {inner}/>"


def _points_to_path(points: str, close: bool) -> str:
    numbers = [float(n) for n in _NUMBER.findall(points)]
    if len(numbers) < 4 or len(numbers) % 2:
        raise SvgError("bad points list")
    pairs = [f"{fmt(numbers[k])} {fmt(numbers[k + 1])}" for k in range(0, len(numbers), 2)]
    return "M" + "L".join(pairs) + ("Z" if close else "")


def _is_invisible(element: ElementTree.Element) -> bool:
    return element.attrib.get("stroke") == "none" and element.attrib.get("fill", "none") == "none"


def normalize_element(element: ElementTree.Element) -> Shape | None:
    tag = _strip_ns(element.tag)
    attrib = {_strip_ns(key): value for key, value in element.attrib.items()}
    if _is_invisible(element):
        return None  # e.g. Tabler's bounding-box helper path
    if "transform" in attrib:
        raise SvgError("transforms are not supported")
    if tag in ("polyline", "polygon"):
        attrib = {"d": _points_to_path(attrib.get("points", ""), close=tag == "polygon"), "fill": attrib.get("fill", "")}
        tag = "path"
    if tag not in GEOMETRY_ATTRS:
        raise SvgError(f"unsupported element: {tag}")

    out: list[tuple[str, str]] = []
    for name in GEOMETRY_ATTRS[tag]:
        if name not in attrib:
            if name in ("rx", "ry") and tag == "rect":
                continue
            raise SvgError(f"{tag} missing {name}")
        if name == "d":
            out.append(("d", normalize_path(attrib["d"])))
        else:
            out.append((name, fmt(_check_coord(_parse_float(attrib[name], name)))))
    fill = attrib.get("fill", "").strip().lower()
    if fill and fill not in ("none", "currentcolor", "transparent"):
        # Any solid fill becomes the foreground color: we only draw in one ink.
        fill = "currentcolor"
    if fill == "currentcolor":
        out.append(("fill", "currentColor"))
    return Shape(tag, tuple(out))


def extract_body(svg_text: str) -> str:
    """Convert a full monochrome 24x24 icon SVG into the canonical body string."""
    try:
        root = ElementTree.fromstring(svg_text)
    except ElementTree.ParseError as exc:
        raise SvgError("invalid XML") from exc
    if _strip_ns(root.tag) != "svg":
        raise SvgError("root must be svg")
    view_box = " ".join(root.attrib.get("viewBox", VIEWBOX).replace(",", " ").split())
    if view_box != VIEWBOX:
        raise SvgError(f"unsupported viewBox: {view_box}")
    shapes: list[str] = []
    for element in root:
        if _strip_ns(element.tag) in ("title", "desc"):
            continue
        if len(element):
            raise SvgError("nested elements are not supported")
        shape = normalize_element(element)
        if shape is not None:
            shapes.append(shape.to_xml())
    if not shapes:
        raise SvgError("icon has no visible shapes")
    body = "".join(shapes)
    if len(body) > MAX_BODY_CHARS:
        raise SvgError("icon is too large")
    return body


# --- importing icons from other sets ---------------------------------------------
#
# Other icon sets use other grids (48x48, 21x21), nest shapes in <g> groups and
# mix line and filled styles. `import_icon` maps a line icon onto the canonical
# 24x24 body and rejects anything that is not line art. `extract_body` above
# stays strict, so the original training packs build exactly as before.

_INHERITED = ("fill", "stroke", "stroke-width")
# Styling that the canonical format cannot express. Icons that use it are rejected.
_UNSUPPORTED_STYLE = ("opacity", "fill-opacity", "stroke-opacity", "stroke-dasharray", "mask", "clip-path", "filter", "style")
STROKE_WIDTH_RANGE = (1.0, 2.5)  # after scaling to 24x24
_MAX_FILLED_SIZE = 4.0  # filled circles/rects up to this size are dots, not filled-style art


def _square_viewbox(root: ElementTree.Element) -> tuple[float, float, float]:
    """Return (min_x, min_y, scale to 24) for a square viewBox."""
    parts = root.attrib.get("viewBox", VIEWBOX).replace(",", " ").split()
    if len(parts) != 4:
        raise SvgError("bad viewBox")
    min_x, min_y, width, height = (_parse_float(p, "viewBox") for p in parts)
    if width <= 0 or abs(width - height) > 1e-6:
        raise SvgError("viewBox is not square")
    return min_x, min_y, 24.0 / width


def _scale_segments(segments: list[tuple[str, list[float]]], dx: float, dy: float, s: float) -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []
    for command, values in segments:
        if command == "H":
            values = [(values[0] - dx) * s]
        elif command == "V":
            values = [(values[0] - dy) * s]
        elif command == "A":
            rx, ry, rotation, large, sweep, x, y = values
            values = [rx * s, ry * s, rotation, large, sweep, (x - dx) * s, (y - dy) * s]
        else:
            values = [(v - (dx if k % 2 == 0 else dy)) * s for k, v in enumerate(values)]
        out.append((command, values))
    return out


def _scaled_geometry(tag: str, attrib: dict[str, str], dx: float, dy: float, s: float) -> dict[str, str]:
    """Rewrite the geometry attributes of one shape into 24x24 canvas units."""
    if tag in ("polyline", "polygon"):
        attrib = {"d": _points_to_path(attrib.get("points", ""), close=tag == "polygon")}
        tag = "path"
    if tag == "path":
        return {"d": format_path(_scale_segments(path_to_absolute(attrib.get("d", "")), dx, dy, s))}
    out: dict[str, str] = {}
    for name in GEOMETRY_ATTRS.get(tag, ()):
        if name not in attrib:
            continue  # normalize_element reports missing attributes
        value = _parse_float(attrib[name], name)
        if name in ("cx", "x", "x1", "x2"):
            value -= dx
        elif name in ("cy", "y", "y1", "y2"):
            value -= dy
        out[name] = repr(value * s)
    return out


def _is_none(value: str) -> bool:
    return value.strip().lower() in ("none", "transparent")


def import_icon(svg_text: str) -> str:
    """Convert a line icon from any square grid into the canonical 24x24 body.

    Groups without transforms are flattened (fill and stroke are inherited), the
    grid is scaled to 24x24, and invisible helper shapes are dropped. Filled
    paths, transparency, dashes and stroke widths far from the house style are
    rejected, because the canonical format draws every shape with one stroke.
    """
    try:
        root = ElementTree.fromstring(svg_text)
    except ElementTree.ParseError as exc:
        raise SvgError("invalid XML") from exc
    if _strip_ns(root.tag) != "svg":
        raise SvgError("root must be svg")
    dx, dy, s = _square_viewbox(root)
    # SVG defaults: shapes are filled black and not stroked.
    defaults = {"fill": "currentColor", "stroke": "none", "stroke-width": "1"}

    def inherit(element: ElementTree.Element, parent: dict[str, str]) -> dict[str, str]:
        attrib = {_strip_ns(k): v for k, v in element.attrib.items()}
        if "transform" in attrib:
            raise SvgError("transforms are not supported")
        for name in _UNSUPPORTED_STYLE:
            if name in attrib and not (name.endswith("opacity") and attrib[name].strip() in ("1", "1.0")):
                raise SvgError(f"unsupported style: {name}")
        return {**parent, **{k: attrib[k] for k in _INHERITED if k in attrib}}

    shapes: list[str] = []

    def walk(element: ElementTree.Element, style: dict[str, str]) -> None:
        for child in element:
            tag = _strip_ns(child.tag)
            if tag in ("title", "desc"):
                continue
            child_style = inherit(child, style)
            if tag == "g":
                walk(child, child_style)
                continue
            if len(child):
                raise SvgError("nested elements are not supported")  # e.g. <animate> inside a path
            if tag not in GEOMETRY_ATTRS and tag not in ("polyline", "polygon"):
                raise SvgError(f"unsupported element: {tag}")
            filled = not _is_none(child_style["fill"])
            stroked = not _is_none(child_style["stroke"])
            if not filled and not stroked:
                continue  # invisible helper shape
            if stroked:
                width = _parse_float(child_style["stroke-width"], "stroke-width") * s
                if not STROKE_WIDTH_RANGE[0] <= width <= STROKE_WIDTH_RANGE[1]:
                    raise SvgError("stroke width outside the line-icon range")
            attrib = {_strip_ns(k): v for k, v in child.attrib.items()}
            geometry = _scaled_geometry(tag, attrib, dx, dy, s)
            if filled:
                if "d" in geometry:
                    raise SvgError("filled paths are not line art")
                size = max(_parse_float(geometry.get(k, "0"), k) for k in ("r", "rx", "ry", "width", "height"))
                if tag in ("circle", "ellipse"):
                    size *= 2
                if size > _MAX_FILLED_SIZE:
                    raise SvgError("large filled shape is not line art")
            canonical_tag = "path" if "d" in geometry else tag
            element = ElementTree.Element(canonical_tag, {**geometry, "fill": "currentColor" if filled else "none"})
            shape = normalize_element(element)
            if shape is not None:
                shapes.append(shape.to_xml())

    walk(root, inherit(root, defaults))
    if not shapes:
        raise SvgError("icon has no visible shapes")
    body = "".join(shapes)
    if len(body) > MAX_BODY_CHARS:
        raise SvgError("icon is too large")
    return body


def parse_body(body: str) -> list[Shape]:
    """Strictly parse model output. Raises SvgError if it is not canonical-safe."""
    try:
        root = ElementTree.fromstring(f"<g>{body}</g>")
    except ElementTree.ParseError as exc:
        raise SvgError("invalid XML") from exc
    shapes: list[Shape] = []
    for element in root:
        if len(element) or (element.text or "").strip():
            raise SvgError("shapes must be empty elements")
        tag = _strip_ns(element.tag)
        if tag not in GEOMETRY_ATTRS:
            raise SvgError(f"unsupported element: {tag}")
        allowed = set(GEOMETRY_ATTRS[tag]) | {"fill"}
        for key, value in element.attrib.items():
            if key not in allowed:
                raise SvgError(f"unsupported attribute: {key}")
            if key == "fill" and value not in ("currentColor", "none"):
                raise SvgError("fill must be currentColor")
        shape = normalize_element(element)
        if shape is not None:
            shapes.append(shape)
    if not shapes:
        raise SvgError("no shapes")
    return shapes


def wrap_body(body: str, stroke_width: float = 2.0, size: int | None = None) -> str:
    """Validate a canonical body and wrap it in a complete black-and-white SVG."""
    shapes = parse_body(body)
    size_attrs = f' width="{size}" height="{size}"' if size else ""
    return (
        f'<svg xmlns="{SVG_NS}"{size_attrs} viewBox="{VIEWBOX}" fill="none" stroke="currentColor" '
        f'stroke-width="{fmt(stroke_width, 2)}" stroke-linecap="round" stroke-linejoin="round">'
        + "".join(shape.to_xml() for shape in shapes)
        + "</svg>"
    )


def salvage_body(text: str) -> str:
    """Keep the longest prefix of complete, valid shapes from raw model output."""
    elements = re.findall(r"<(?:path|circle|ellipse|rect|line)\b[^<>]*/>", text)
    kept: list[str] = []
    for element in elements:
        try:
            parse_body(element)
        except SvgError:
            break
        kept.append(element)
    return "".join(kept)
