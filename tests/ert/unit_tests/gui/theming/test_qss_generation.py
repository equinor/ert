from __future__ import annotations

import json
import re

import pytest
from PyQt6.QtGui import QColor

from ert.gui.theming import ColorScheme
from ert.gui.theming import theme as theme_module
from ert.gui.theming.eds import data as eds_data
from ert.gui.theming.eds import scale, semantic
from ert.gui.theming.theme import load_qss, resolve_color

_TOKEN = re.compile(r"@[a-z0-9-]+")
_DARK_OVERRIDES = theme_module._ERT_SEMANTIC_OVERRIDES[ColorScheme.DARK]


def _relative_luminance(color: QColor) -> float:
    red, green, blue = (
        component / 12.92
        if component <= 0.03928
        else ((component + 0.055) / 1.055) ** 2.4
        for component in (color.redF(), color.greenF(), color.blueF())
    )
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground: QColor, background: QColor) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _first_rule_body(qss: str, selector: str) -> str:
    pattern = rf"^{re.escape(selector)}\s*\{{(.*?)\}}"
    match = re.search(pattern, qss, re.MULTILINE | re.DOTALL)
    assert match is not None, f"no '{selector}' rule in the generated QSS"
    return match.group(1)


@pytest.mark.parametrize("color_scheme", list(ColorScheme))
def test_that_loaded_qss_contains_no_unresolved_tokens(
    color_scheme: ColorScheme,
) -> None:
    assert not _TOKEN.search(load_qss(color_scheme))


def test_that_selected_nav_button_uses_success_subtle_fill_in_light() -> None:
    active_fill = semantic(ColorScheme.LIGHT, "text-success-subtle-on-emphasis")
    assert active_fill in load_qss(ColorScheme.LIGHT)


def test_that_selected_nav_button_is_filled_near_black_in_dark() -> None:
    active_fill = resolve_color(ColorScheme.DARK, "text-success-subtle-on-emphasis")
    assert active_fill == scale(ColorScheme.DARK, "neutral-2")
    assert QColor(active_fill).lightness() < 45
    assert active_fill in load_qss(ColorScheme.DARK)


def test_that_selected_nav_button_is_darker_than_the_sidebar_it_sits_on() -> None:
    active_fill = QColor(scale(ColorScheme.DARK, "neutral-2")).lightness()
    sidebar = QColor(scale(ColorScheme.DARK, "neutral-14")).lightness()
    assert active_fill < sidebar


def test_that_selected_nav_label_stays_legible_on_the_active_fill() -> None:
    fill = QColor(scale(ColorScheme.DARK, "neutral-2"))
    label = QColor(semantic(ColorScheme.DARK, "bg-accent-fill-emphasis-active"))
    assert _contrast_ratio(label, fill) >= 4.5


def test_that_resolve_color_returns_the_eds_value_for_an_unoverridden_token() -> None:
    assert resolve_color(ColorScheme.DARK, "text-accent-strong") == semantic(
        ColorScheme.DARK, "text-accent-strong"
    )


def test_that_resolve_color_applies_the_ert_override_for_an_overridden_token() -> None:
    assert resolve_color(ColorScheme.DARK, "bg-neutral-canvas") == scale(
        ColorScheme.DARK, "neutral-3"
    )


def test_that_resolve_color_leaves_light_scheme_tokens_unchanged() -> None:
    for token in _DARK_OVERRIDES:
        assert resolve_color(ColorScheme.LIGHT, token) == semantic(
            ColorScheme.LIGHT, token
        )


def test_that_resolve_color_rejects_a_token_absent_from_the_eds_set() -> None:
    with pytest.raises(KeyError, match="not an EDS semantic token"):
        resolve_color(ColorScheme.DARK, "bg-not-a-real-token")


def test_that_light_scheme_resolves_background_tokens_straight_from_eds() -> None:
    qss = load_qss(ColorScheme.LIGHT)
    for token in ("bg-neutral-canvas", "bg-neutral-surface", "bg-accent-surface"):
        assert semantic(ColorScheme.LIGHT, token) in qss


@pytest.mark.parametrize(
    ("token", "scale_step"),
    [
        ("bg-neutral-canvas", "neutral-3"),
        ("bg-neutral-surface", "neutral-14"),
        ("bg-accent-surface", "accent-14"),
    ],
)
def test_that_dark_background_tokens_resolve_to_the_ert_override_scale_step(
    token: str, scale_step: str
) -> None:
    resolved = resolve_color(ColorScheme.DARK, token)
    assert resolved == scale(ColorScheme.DARK, scale_step)
    assert resolved != semantic(ColorScheme.DARK, token)
    assert resolved in load_qss(ColorScheme.DARK)


def test_that_dark_canvas_rule_paints_the_overridden_background() -> None:
    canvas_rule = _first_rule_body(load_qss(ColorScheme.DARK), "QWidget")
    assert f"background-color: {scale(ColorScheme.DARK, 'neutral-3')};" in canvas_rule


def test_that_dark_canvas_is_lighter_than_the_sidebar_surface() -> None:
    canvas = QColor(scale(ColorScheme.DARK, "neutral-3")).lightness()
    sidebar = QColor(scale(ColorScheme.DARK, "neutral-14")).lightness()
    nav_item = QColor(scale(ColorScheme.DARK, "accent-14")).lightness()
    assert canvas > sidebar > nav_item


def test_that_dark_canvas_keeps_body_text_above_the_wcag_aa_contrast_ratio() -> None:
    canvas = QColor(scale(ColorScheme.DARK, "neutral-3"))
    text = QColor(semantic(ColorScheme.DARK, "text-neutral-strong"))
    assert _contrast_ratio(text, canvas) >= 4.5


def test_that_an_override_of_an_unknown_semantic_token_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        theme_module._ERT_SEMANTIC_OVERRIDES,
        ColorScheme.DARK,
        {"bg-not-a-real-token": "neutral-3"},
    )
    monkeypatch.setattr(
        theme_module,
        "_read_template",
        lambda: "QWidget { background-color: @bg-not-a-real-token; }",
    )
    with pytest.raises(KeyError, match="not an EDS semantic token"):
        load_qss(ColorScheme.DARK)


def test_that_load_qss_raises_for_a_token_absent_from_the_eds_set(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        theme_module,
        "_read_template",
        lambda: "QWidget { color: @bg-not-a-real-token; }",
    )
    with pytest.raises(KeyError, match="not an EDS semantic token"):
        load_qss(ColorScheme.LIGHT)


def test_that_semantic_lookup_rejects_a_token_absent_from_the_eds_set() -> None:
    with pytest.raises(KeyError, match="not an EDS semantic token"):
        semantic(ColorScheme.LIGHT, "bg-not-a-real-token")


@pytest.mark.parametrize("color_scheme", list(ColorScheme))
def test_that_load_qss_resolves_a_scale_step_reference_to_its_hex_value(
    monkeypatch, color_scheme: ColorScheme
) -> None:
    monkeypatch.setattr(
        theme_module,
        "_read_template",
        lambda: "QWidget { background-color: @success-6; }",
    )
    assert scale(color_scheme, "success-6") in load_qss(color_scheme)


def test_that_load_qss_raises_for_a_scale_step_absent_from_the_eds_set(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        theme_module,
        "_read_template",
        lambda: "QWidget { color: @success-999; }",
    )
    with pytest.raises(KeyError, match="not an EDS scale step"):
        load_qss(ColorScheme.LIGHT)


def test_that_bundled_light_and_dark_tokens_define_the_same_token_names() -> None:
    light = json.loads(eds_data.bundled_path(ColorScheme.LIGHT).read_text())
    dark = json.loads(eds_data.bundled_path(ColorScheme.DARK).read_text())
    assert light["semantic"].keys() == dark["semantic"].keys()
    assert light["scale"].keys() == dark["scale"].keys()
