from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton
from pytestqt.qtbot import QtBot

from ert.gui.tools.plot.plot_window import PlotWindow, create_error_dialog
from ert.services import ErtServer


def test_pressing_copy_button_in_error_dialog(qtbot: QtBot):
    qd = create_error_dialog("hello", "world")
    qtbot.addWidget(qd)

    qtbot.mouseClick(
        qd.findChild(QPushButton, name="copy_button"), Qt.MouseButton.LeftButton
    )
    assert QApplication.clipboard().text() == "world"


@pytest.mark.integration_test
def test_warning_is_visible_on_incompatible_plot_api_version(
    qtbot: QtBot, tmp_path, monkeypatch, use_tmpdir
):
    mock_get_data = MagicMock()
    mock_get_data.return_value = "0.2"
    monkeypatch.setattr(
        "ert.gui.tools.plot.plot_api.PlotApi.api_version", mock_get_data
    )

    with ErtServer.init_service(project=tmp_path):
        pw = PlotWindow("", tmp_path, None)
        qtbot.addWidget(pw)
        pw.show()

        label = pw.findChild(QLabel, name="plot_api_warning_label")
        assert label
        assert label.isVisible()
        assert label.text().startswith("<b>Plot API version mismatch detected")
