from unittest.mock import Mock, patch

import pytest
from qtpy.QtCore import QThread
from qtpy.QtGui import QIcon

from ert.analysis import ErtAnalysisError, ESUpdate
from ert.enkf_main import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.statusdialog import StatusDialog
from ert.gui.tools.run_analysis import Analyse, RunAnalysisPanel, RunAnalysisTool


# Mocks are all instances and can never fool type checking, like in QAction's
# initializer. Thus we need to subclass and pass a QIcon to it---without
# instantiating a qt gui application.
# TODO: remove as part of https://github.com/equinor/ert/issues/422
class MockedQIcon(QIcon):
    pass


pytestmark = pytest.mark.requires_window_manager


@pytest.fixture
def mock_tool(mock_storage):
    with patch("ert.gui.tools.run_analysis.run_analysis_tool.resourceIcon") as rs:
        rs.return_value = MockedQIcon()
        (target, source) = mock_storage

        run_widget = Mock(spec_set=RunAnalysisPanel)
        run_widget.source_case.return_value = source
        run_widget.target_case.return_value = target.name
        notifier = Mock(spec_set=ErtNotifier)
        notifier.storage.to_accessor.return_value = notifier.storage
        notifier.storage.create_ensemble.return_value = target

        tool = RunAnalysisTool(Mock(spec_set=EnKFMain), notifier)
        tool._run_widget = run_widget
        tool._dialog = Mock(spec_set=StatusDialog)

        return tool


@pytest.fixture
def mock_storage(storage):
    experiment = storage.create_experiment()
    source = experiment.create_ensemble(ensemble_size=1, name="source")
    target = experiment.create_ensemble(ensemble_size=1, name="target")
    return target, source


@pytest.mark.requires_window_manager
def test_analyse_success(mock_storage, qtbot):
    (target, source) = mock_storage

    analyse = Analyse(Mock(spec_set=EnKFMain), target, source)
    thread = QThread()
    with qtbot.waitSignals(
        [analyse.finished, thread.finished], timeout=2000, raising=True
    ):
        analyse.moveToThread(thread)
        thread.started.connect(analyse.run)
        analyse.finished.connect(thread.quit)
        thread.start()


@pytest.mark.requires_window_manager
@patch("ert.gui.tools.run_analysis.run_analysis_tool.QMessageBox")
@patch("ert.gui.tools.run_analysis.run_analysis_tool.ESUpdate")
@patch.object(RunAnalysisTool, "_enable_dialog", new=lambda self, enable: None)
def test_success(
    mock_esupdate,
    mock_msgbox,
    mock_tool,
    qtbot,
):
    mock_tool.run()

    qtbot.waitUntil(lambda: len(mock_msgbox.information.mock_calls) > 0, timeout=2000)
    mock_msgbox.critical.assert_not_called()
    mock_msgbox.information.assert_called_once_with(
        None, "Analysis finished", "Successfully ran analysis for case 'source'."
    )
    mock_esupdate.assert_called_once()

    mock_tool._dialog.accept.assert_called_once()


@pytest.mark.requires_window_manager
@patch("ert.gui.tools.run_analysis.run_analysis_tool.QMessageBox")
def test_failure(
    mock_msgbox,
    mock_tool,
    qtbot,
    monkeypatch,
):
    monkeypatch.setattr(
        ESUpdate,
        "smootherUpdate",
        Mock(spec_set=ESUpdate, side_effect=ErtAnalysisError("some error")),
    )

    mock_tool.run()

    qtbot.waitUntil(lambda: len(mock_msgbox.warning.mock_calls) > 0, timeout=2000)
    mock_msgbox.critical.assert_not_called()
    mock_msgbox.warning.assert_called_once_with(
        None,
        "Failed",
        "Unable to run analysis for case 'source'.\n"
        "The following error occurred: some error",
    )

    mock_tool._dialog.accept.assert_not_called()
