import pytest
from qtpy.QtCore import Qt

from ieverest import IEverest
from tests.everest.dialogs_mocker import mock_dialogs_all
from tests.everest.utils import relpath

CASE_DIR = relpath("test_data", "mocked_test_case")
CONFIG_FILE = "mocked_test_case.yml"


@pytest.mark.ui_test
def test_load_run(qapp, qtbot, mocker, copy_mocked_test_data_to_tmp):
    """Load a configuration and run it from the UI"""

    qapp.setAttribute(Qt.AA_X11InitThreads)
    qapp.setOrganizationName("Equinor/TNO")
    qapp.setApplicationName("IEverest")
    ieverest = IEverest()

    # Load the configuration
    mock_dialogs_all(mocker, open_file_name=CONFIG_FILE)

    start_server_mock = mocker.patch("ieverest.ieverest.start_server")
    wait_for_server_mock = mocker.patch("ieverest.ieverest.wait_for_server")
    start_monitor_mock = mocker.patch("ieverest.ieverest.start_monitor")

    qtbot.mouseClick(ieverest._gui._startup_gui.open_btn, Qt.LeftButton)
    # Start the mocked optimization
    qtbot.mouseClick(ieverest._gui.monitor_gui.start_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: ieverest.server_monitor is not None, timeout=10 * 1e3)
    qtbot.waitUntil(lambda: ieverest.server_monitor is None, timeout=10 * 1e3)

    start_server_mock.assert_called_once()
    wait_for_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
