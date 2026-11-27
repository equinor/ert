import pytest
from PyQt6.QtGui import QIcon

from ert.gui import main_window
from ert.gui.main_window import ErtMainWindow

DISABLED_DARK = "#738696"


def _opaque_colors(icon: QIcon, mode: QIcon.Mode) -> set[str]:
    image = icon.pixmap(32, 32, mode).toImage()
    return {
        image.pixelColor(x, y).name()
        for x in range(image.width())
        for y in range(image.height())
        if image.pixelColor(x, y).alpha() == 255
    }


@pytest.mark.usefixtures("qtbot")
def test_that_disabled_sidebar_icon_in_dark_mode_uses_the_eds_disabled_color(
    monkeypatch,
):
    monkeypatch.setattr(main_window, "is_dark_mode", lambda: True)
    icon = ErtMainWindow._sidebar_icon("in_progress.svg")

    assert _opaque_colors(icon, QIcon.Mode.Disabled) == {DISABLED_DARK}
    assert DISABLED_DARK not in _opaque_colors(icon, QIcon.Mode.Normal)
