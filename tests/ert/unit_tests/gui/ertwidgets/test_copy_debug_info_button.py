from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from ert.gui.simulation.run_dialog import CopyDebugInfoButton


def test_copy_debug_info_button_alterates_text_when_pressed(qtbot: QtBot):
    button_clicked = False

    def on_click():
        nonlocal button_clicked
        button_clicked = True

    button = CopyDebugInfoButton(on_click=on_click)
    qtbot.addWidget(button)

    assert button.text() == CopyDebugInfoButton._initial_text
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert button.text() == CopyDebugInfoButton._clicked_text
    qtbot.wait_until(
        lambda: button.text() == CopyDebugInfoButton._initial_text, timeout=2000
    )
    assert button_clicked
