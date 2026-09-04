from __future__ import annotations

import pytest

from ert.gui.theming import Theme
from ert.gui.theming.theme import load_qss


@pytest.mark.parametrize("theme", list(Theme))
def test_that_qss_file_is_loaded_for_each_theme(theme: Theme) -> None:
    content = load_qss(theme)
    assert content.strip(), f"QSS for {theme.value} theme must not be empty"
    assert "QWidget" in content, (
        f"QSS for {theme.value} theme should style QWidget as a baseline"
    )


def test_that_dark_and_light_qss_differ() -> None:
    assert load_qss(Theme.DARK) != load_qss(Theme.LIGHT)
