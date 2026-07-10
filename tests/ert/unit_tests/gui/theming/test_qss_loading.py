from __future__ import annotations

import pytest

from ert.gui.theming import Theme
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


@pytest.mark.parametrize("theme", list(Theme))
def test_that_qss_file_is_loaded_for_each_theme(theme: Theme) -> None:
    content = load_qss(theme)
    assert content.strip(), f"QSS for {theme.value} theme must not be empty"
    assert "QWidget" in content, (
        f"QSS for {theme.value} theme should style QWidget as a baseline"
    )


def test_that_dark_and_light_qss_differ() -> None:
    assert load_qss(Theme.DARK) != load_qss(Theme.LIGHT)


def test_that_load_qss_raises_file_not_found_when_theme_file_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(theme_module, "files", lambda _pkg: _MissingPackage())

    with pytest.raises(FileNotFoundError, match="dark"):
        load_qss(Theme.DARK)
