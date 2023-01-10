from unittest.mock import Mock, patch

import pytest
from qtpy.QtGui import QIcon

from ert.analysis import ErtAnalysisError
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import run_analysis


# Mocks are all instances and can never fool type checking, like in QAction's
# initializer. Thus we need to subclass and pass a QIcon to it---without
# instantiating a qt gui application.
# TODO: remove as part of https://github.com/equinor/ert/issues/422
class MockedQIcon(QIcon):
    pass


pytestmark = pytest.mark.requires_window_manager


@pytest.fixture
def mock_tool():
    with patch("ert.gui.tools.run_analysis.run_analysis_tool.resourceIcon") as rs:
        rs.return_value = MockedQIcon()
        tool = run_analysis.RunAnalysisTool(Mock(), Mock())
        tool._run_widget = Mock(spec=run_analysis.RunAnalysisPanel)
        tool._dialog = Mock(spec=ClosableDialog)
        yield tool


@pytest.mark.requires_window_manager
@patch("ert.gui.tools.run_analysis.run_analysis_tool.analyse", return_value=None)
@patch("ert.gui.tools.run_analysis.run_analysis_tool.QMessageBox")
def test_show_dialogue_at_success(mock_messagebox, mock_analyse, mock_tool, storage):
    experiment = storage.create_experiment()
    source = experiment.create_ensemble(ensemble_size=1, name="source")
    target = experiment.create_ensemble(ensemble_size=1, name="target")

    mock_tool._run_widget.source_case.return_value = source
    mock_tool._run_widget.target_case.return_value = target.name
    mock_tool.notifier.storage.create_ensemble.return_value = target

    ert_mock = Mock()
    mock_tool.ert = ert_mock
    mock_tool.run()

    mock_analyse.assert_called_once_with(ert_mock, target, source)
    mock_messagebox.return_value.setText.assert_called_once_with(
        "Successfully ran analysis for case 'source'."
    )
    mock_tool._dialog.accept.assert_called_once_with()


@pytest.mark.requires_window_manager
@patch(
    "ert.gui.tools.run_analysis.run_analysis_tool.analyse",
    side_effect=ErtAnalysisError("some error"),
)
@patch("ert.gui.tools.run_analysis.run_analysis_tool.QMessageBox")
def test_show_dialogue_at_failure(mock_messagebox, mock_analyse, mock_tool, storage):
    experiment = storage.create_experiment()
    source = experiment.create_ensemble(ensemble_size=1, name="source")
    target = experiment.create_ensemble(ensemble_size=1, name="target")

    mock_tool._run_widget.source_case.return_value = source
    mock_tool._run_widget.target_case.return_value = target.name
    mock_tool.notifier.storage.create_ensemble.return_value = target

    ert_mock = Mock()
    mock_tool.ert = ert_mock
    mock_tool.run()

    mock_analyse.assert_called_once_with(ert_mock, target, source)
    mock_messagebox.return_value.setText.assert_called_once_with(
        "Unable to run analysis for case 'source'.\n"
        "The following error occured: some error"
    )
    mock_tool._dialog.accept.assert_not_called()
