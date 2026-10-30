import pytest

from svgen.data import Icon, describe_name, make_prompts
from svgen.svg import SvgError, extract_body, fmt, normalize_path, parse_body, salvage_body, wrap_body

import random


def test_fmt_is_compact():
    assert fmt(3.0) == "3"
    assert fmt(0.54) == ".5"
    assert fmt(-0.46) == "-.5"
    assert fmt(12.04) == "12"
    assert fmt(-0.01) == "0"


def test_relative_path_becomes_absolute():
    assert normalize_path("M2 2l3 4h1v-2z") == "M2 2L5 6H6V4Z"


def test_implicit_lineto_after_moveto():
    assert normalize_path("m1 1 2 2 3 3") == "M1 1L3 3L6 6"


def test_packed_arc_flags_are_parsed():
    # "a2 2 0 01.586 1" means flags 0,1 then x=.586
    assert normalize_path("M10 10a2 2 0 01.586 1") == "M10 10A2 2 0 0 1 10.6 11"


def test_relative_rounding_does_not_drift():
    d = "M0 0" + "l.04 .04" * 50
    assert normalize_path(d).endswith("L2 2")


def test_extract_body_drops_tabler_helper_and_styling():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor">'
        '<path stroke="none" d="M0 0h24v24H0z" fill="none"/>'
        '<circle cx="12" cy="12" r="9" stroke-linecap="round"/>'
        '<polyline points="8 12 12 16 16 12"/></svg>'
    )
    assert extract_body(svg) == '<circle cx="12" cy="12" r="9"/><path d="M8 12L12 16L16 12"/>'


def test_extract_body_rejects_other_canvas_and_transforms():
    with pytest.raises(SvgError):
        extract_body('<svg viewBox="0 0 16 16"><path d="M1 1L2 2"/></svg>')
    with pytest.raises(SvgError):
        extract_body('<svg viewBox="0 0 24 24"><rect x="1" y="1" width="2" height="2" transform="rotate(4)"/></svg>')


@pytest.mark.parametrize(
    "body",
    [
        '<script>alert(1)</script>',
        '<path d="M1 1" onload="x()"/>',
        '<image href="https://example.com/a.png"/>',
        '<path d="M1 1L999 2"/>',
        '<path d="M1 1" fill="red"/>',
        "",
    ],
)
def test_parse_body_rejects_unsafe_or_bad_output(body):
    with pytest.raises(SvgError):
        parse_body(body)


def test_wrap_body_builds_complete_svg():
    svg = wrap_body('<path d="M4 12H20"/>')
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"')
    assert 'stroke="currentColor"' in svg and svg.endswith("</svg>")


def test_salvage_keeps_complete_shapes_from_cut_output():
    raw = '<circle cx="12" cy="12" r="3"/><path d="M4 4L8 8"/><path d="M1 1L'
    assert salvage_body(raw) == '<circle cx="12" cy="12" r="3"/><path d="M4 4L8 8"/>'


def test_describe_name_and_prompts():
    assert describe_name("brand-github") == "github logo"
    assert describe_name("bell-off") == "bell crossed out"
    assert describe_name("arrow-big-down-2") == "arrow big down"
    icon = Icon("tabler", "heart", "<path/>", tags=["love", "like"])
    prompts = make_prompts(icon, random.Random(0), 2)
    assert len(prompts) == 2 and all("heart" in p for p in prompts)
