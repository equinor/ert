import pytest
from PyQt5.QtWidgets import QAction, QPushButton, QWidget
from qtpy.QtCore import Qt
from seba_sqlite.snapshot import SebaSnapshot
from tests.everest.dialogs_mocker import mock_dialogs_all

from everest.config import EverestConfig
from everest.detached import (
    ServerStatus,
    context_stop_and_wait,
    everserver_status,
    wait_for_context,
)
from ieverest import IEverest

CONFIG_FILE_MINIMAL = "config_minimal.yml"


@pytest.mark.flaky(reruns=5)
@pytest.mark.ui_test
@pytest.mark.xdist_group(name="starts_everest")
def test_ui_optimization(qapp, qtbot, mocker, copy_math_func_test_data_to_tmp):
    """Load a configuration and run it from the UI"""

    wait_for_context()

    ieverest = IEverest()

    # Load the configuration
    mock_dialogs_all(mocker, open_file_name=CONFIG_FILE_MINIMAL)

    # check that about dialog can be opened
    about_action = ieverest._gui.findChild(QAction, "about_action")
    about_action.trigger()

    qtbot.waitUntil(lambda: ieverest._gui.about_widget is not None)
    msgbox = ieverest._gui.findChild(QWidget, "about_widget")
    assert msgbox.windowTitle() == "About Everest"

    close_button = msgbox.findChild(QPushButton, "button_close_about")
    qtbot.mouseClick(close_button, Qt.LeftButton)

    qtbot.mouseClick(ieverest._gui._startup_gui.open_btn, Qt.LeftButton)
    # Start the mocked optimization
    qtbot.mouseClick(ieverest._gui.monitor_gui.start_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: ieverest.server_monitor is not None, timeout=10 * 1e6)
    qtbot.waitUntil(lambda: ieverest.server_monitor is None, timeout=10 * 1e6)

    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    status = everserver_status(config)

    assert status["status"] == ServerStatus.completed
    assert status["message"] == "Maximum number of batches reached."

    snapshot = SebaSnapshot(config.optimization_output_dir).get_snapshot()

    best_settings = snapshot.optimization_data[-1]
    assert abs(best_settings.controls["point_x"] - 0.5) <= 0.05
    assert abs(best_settings.controls["point_y"] - 0.5) <= 0.05
    assert abs(best_settings.controls["point_z"] - 0.5) <= 0.05

    assert abs(best_settings.objective_value) <= 0.0005

    context_stop_and_wait()
