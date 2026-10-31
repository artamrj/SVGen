"""Importing line icons from other sets (Iconify JSON) into the canonical format."""

import json

import pytest

from svgen.data import CLASSIC_SOURCES, Icon, _read_iconify, build, frozen_split, load_heldout
from svgen.svg import SvgError, import_icon, parse_body


def svg(body: str, box: str = "0 0 24 24") -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}">{body}</svg>'


def test_import_scales_48_grid_and_flattens_groups():
    # IconPark style: 48x48 grid, stroke width 4, styling on a <g>.
    body = import_icon(svg(
        '<g fill="none" stroke="currentColor" stroke-width="4">'
        '<path d="M8 8h32v32"/><circle cx="24" cy="24" r="10"/></g>', "0 0 48 48"))
    assert body == '<path d="M4 4H20V20"/><circle cx="12" cy="12" r="5"/>'


def test_import_handles_arcs_and_viewbox_offset():
    body = import_icon(svg('<path fill="none" stroke="currentColor" stroke-width="4" d="M14 14a4 4 0 0 1 8 0"/>', "2 2 48 48"))
    assert body == '<path d="M6 6A2 2 0 0 1 10 6"/>'


def test_import_drops_invisible_shapes_and_keeps_small_dots():
    body = import_icon(svg(
        '<g fill="none"><path d="M0 0h24v24H0z"/>'
        '<path stroke="currentColor" stroke-width="1.5" d="M2 2L20 20"/>'
        '<circle cx="12" cy="6" r="1" fill="currentColor"/></g>'))
    assert body == '<path d="M2 2L20 20"/><circle cx="12" cy="6" r="1" fill="currentColor"/>'
    parse_body(body)


@pytest.mark.parametrize("body, reason", [
    ('<path fill="currentColor" d="M2 2h20v20H2z"/>', "filled paths"),  # filled-style icon
    ('<rect x="2" y="2" width="20" height="20" fill="currentColor"/>', "large filled"),
    ('<path stroke="currentColor" stroke-width="2" opacity=".4" d="M2 2L4 4"/>', "opacity"),
    ('<path stroke="currentColor" stroke-width="2" stroke-dasharray="2" d="M2 2L4 4"/>', "dasharray"),
    ('<path stroke="currentColor" stroke-width="5" d="M2 2L4 4"/>', "stroke width"),
    ('<g transform="rotate(4)"><path stroke="currentColor" d="M2 2L4 4"/></g>', "transforms"),
    ('<path stroke="currentColor" stroke-width="2" d="M2 2L4 4"><animate/></path>', "nested"),
])
def test_import_rejects_what_is_not_line_art(body, reason):
    with pytest.raises(SvgError, match=reason):
        import_icon(svg(body))


def test_import_rejects_non_square_grid():
    with pytest.raises(SvgError, match="square"):
        import_icon(svg('<path stroke="currentColor" d="M1 1L2 2"/>', "0 0 24 25"))


def test_read_iconify_names_tags_and_suffixes(tmp_path):
    line = '<path fill="none" stroke="currentColor" stroke-width="4" d="M8 8L40 40"/>'
    (tmp_path / "icons.json").write_text(json.dumps({
        "prefix": "icon-park-outline", "width": 48, "height": 48,
        "icons": {"cat": {"body": line}, "bank-card-one": {"body": line}, "blob": {"body": '<path d="M0 0h48v48z"/>'},
                  "old": {"body": line, "hidden": True}},
        "categories": {"Animals": ["cat"]},
    }))
    items = list(_read_iconify(tmp_path, "icon-park-outline"))
    icons = [i for i in items if isinstance(i, Icon)]
    assert [(i.source, i.name, i.tags) for i in icons] == [("icon-park", "bank-card-1", []), ("icon-park", "cat", ["animals"])]
    assert icons[1].body == '<path d="M4 4L20 20"/>'
    assert [i for i in items if isinstance(i, str)] == ["icon-park: filled paths are not line art"]


def test_frozen_split_keeps_new_sources_out_of_heldout_concepts():
    icons = [Icon("tabler", "cat", "a"), Icon("hugeicons", "cat-2", "b"), Icon("hugeicons", "dog", "c"),
             Icon("lucide", "lock", "d"), Icon("icon-park", "lock", "e")]
    split, dropped = frozen_split(icons, valid_groups={"lock"}, test_groups={"cat"})
    assert [i.body for i in split["test"]] == ["a"]
    assert [i.body for i in split["valid"]] == ["d"]
    assert [i.body for i in split["train"]] == ["c"]
    assert dropped == 2


def test_committed_heldout_split_is_consistent():
    valid_groups, test_groups, test_icons = load_heldout()
    assert len(valid_groups) == 120 and len(test_groups) == 150 and not valid_groups & test_groups
    assert {Icon(t["source"], t["name"], t["body"]).group for t in test_icons} == test_groups
    assert {t["source"] for t in test_icons} <= set(CLASSIC_SOURCES)
    for t in test_icons:
        parse_body(t["body"])


def test_build_fails_loudly_when_heldout_icons_are_missing(tmp_path):
    with pytest.raises(SystemExit, match="held-out"):
        build(tmp_path / "empty", tmp_path / "out", 7, 2, 1400, 120, 150)
