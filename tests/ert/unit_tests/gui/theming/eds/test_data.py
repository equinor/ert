import pytest

from ert.gui.theming import eds
from ert.gui.theming.eds import data
from ert.gui.theming.theme import ColorScheme


def test_that_semantic_returns_the_bundled_hex_for_a_known_token():
    value = eds.semantic(ColorScheme.LIGHT, "bg-accent-fill-muted-default")

    assert value.startswith("#")


def test_that_scale_returns_the_bundled_hex_for_a_known_step():
    value = eds.scale(ColorScheme.LIGHT, "accent-11")

    assert value.startswith("#")


def test_that_semantic_raises_key_error_for_an_unknown_token():
    with pytest.raises(KeyError, match="not an EDS semantic token"):
        eds.semantic(ColorScheme.LIGHT, "not-a-real-token")


def test_that_scale_raises_key_error_for_an_unknown_step():
    with pytest.raises(KeyError, match="not an EDS scale step"):
        eds.scale(ColorScheme.LIGHT, "not-a-real-step")


def test_that_bundle_raises_file_not_found_when_the_token_file_is_missing(
    tmp_path, monkeypatch
):
    data._bundle.cache_clear()
    missing = tmp_path / "light.json"
    monkeypatch.setattr(data, "bundled_path", lambda _scheme: missing)

    with pytest.raises(FileNotFoundError, match="Run: uv run python -m"):
        data._bundle(ColorScheme.LIGHT)

    data._bundle.cache_clear()
