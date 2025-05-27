import tempfile

from PyQt6 import QtCore
from pytestqt.qtbot import QtBot

from ert.gui.tools.file.file_dialog import FileDialog


def test_file_dialog_default_keyclicks(qtbot: QtBot):
    FileDialog._init_thread = lambda self: None
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as tmp_file:
        dialog = FileDialog(tmp_file.name, "the_step", 0, 0, 0)
        assert dialog.windowTitle() == "the_step # 0 Realization: 0 Iteration: 0"
        qtbot.keyClick(dialog, QtCore.Qt.Key.Key_Return)
        assert dialog.isVisible()
        qtbot.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
        assert not dialog.isVisible()
