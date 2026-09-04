from __future__ import annotations

import pytest

from ert.gui.theming import ColorScheme
from ert.gui.theming import theme as theme_module
from ert.gui.theming.theme import load_qss


class _MissingResource:
    def is_file(self) -> bool:
        return False

    def __str__(self) -> str:
        return "<missing>"


class _MissingPackage:
    def joinpath(self, _name: str) -> _MissingResource:
        return _MissingResource()


@pytest.mark.parametrize("color_scheme", list(ColorScheme))
def test_that_qss_file_is_loaded_for_each_color_scheme(
    color_scheme: ColorScheme,
) -> None:
    content = load_qss(color_scheme)
    assert content.strip(), (
        f"QSS for {color_scheme.value} colour scheme must not be empty"
    )
    assert "QWidget" in content, (
        f"QSS for {color_scheme.value} colour scheme should style QWidget as a baseline"
    )


def test_that_dark_and_light_qss_differ() -> None:
    assert load_qss(ColorScheme.DARK) != load_qss(ColorScheme.LIGHT)


def test_that_load_qss_raises_file_not_found_when_theme_file_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(theme_module, "files", lambda _pkg: _MissingPackage())

    with pytest.raises(FileNotFoundError, match="dark"):
        load_qss(ColorScheme.DARK)
