from __future__ import annotations

import json

import pytest

from ert.gui.theming import ColorScheme
from ert.gui.theming.eds import data as eds_data
from ert.gui.theming.eds import semantic
from ert.gui.theming.generate_qss import _substitute, build_qss
from ert.gui.theming.theme import load_qss
from ert.gui.theming.tokens import _SEMANTIC_MAP, palette


@pytest.mark.parametrize("color_scheme", list(ColorScheme))
def test_that_committed_qss_matches_freshly_compiled_template(
    color_scheme: ColorScheme,
) -> None:
    assert load_qss(color_scheme) == build_qss(color_scheme), (
        f"The committed {color_scheme.value}.qss is out of sync with the template. "
        "Run: uv run python -m ert.gui.theming.generate_qss"
    )


@pytest.mark.parametrize("color_scheme", list(ColorScheme))
def test_that_compiled_qss_contains_no_unresolved_variables(
    color_scheme: ColorScheme,
) -> None:
    assert "@" not in build_qss(color_scheme)


def test_that_substitute_raises_for_a_variable_missing_from_the_palette() -> None:
    with pytest.raises(KeyError, match="undefined variable '@does-not-exist'"):
        _substitute("QWidget { color: @does-not-exist; }", {"primary": "#007079"})


@pytest.mark.parametrize("color_scheme", list(ColorScheme))
def test_that_every_colour_variable_resolves_to_an_eds_semantic_token(
    color_scheme: ColorScheme,
) -> None:
    resolved = palette(color_scheme)
    for variable, token in _SEMANTIC_MAP.items():
        assert resolved[variable] == semantic(color_scheme, token)


def test_that_semantic_lookup_rejects_a_token_absent_from_the_eds_set() -> None:
    with pytest.raises(KeyError, match="not an EDS semantic token"):
        semantic(ColorScheme.LIGHT, "bg-not-a-real-token")


@pytest.mark.parametrize("color_scheme", list(ColorScheme))
def test_that_the_selected_navigation_button_uses_the_eds_accent_colour(
    color_scheme: ColorScheme,
) -> None:
    accent_fill = semantic(color_scheme, "bg-accent-fill-muted-default")
    assert palette(color_scheme)["nav-active-bg"] == accent_fill


def test_that_bundled_light_and_dark_tokens_define_the_same_token_names() -> None:
    light = json.loads(eds_data.bundled_path(ColorScheme.LIGHT).read_text())
    dark = json.loads(eds_data.bundled_path(ColorScheme.DARK).read_text())
    assert light["semantic"].keys() == dark["semantic"].keys()
    assert light["scale"].keys() == dark["scale"].keys()
