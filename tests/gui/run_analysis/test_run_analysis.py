import sys

from qtpy.QtGui import QIcon

from ert_gui.tools import run_analysis
from ert_gui.ertwidgets.closabledialog import ClosableDialog
from tests import ErtTest

from unittest.mock import Mock, patch

# Mocks are all instances and can never fool type checking, like in QAction's
# initializer. Thus we need to subclass and pass a QIcon to it---without
# instantiating a qt gui application.
# TODO: remove as part of https://github.com/equinor/ert/issues/422
class MockedQIcon(QIcon):
    pass


class RunAnalysisTests(ErtTest):
    def __init__(self, *args, **kwargs):
        super(ErtTest, self).__init__(*args, **kwargs)

        with patch("ert_gui.tools.run_analysis.run_analysis_tool.resourceIcon") as rs:
            rs.return_value = MockedQIcon()
            self.tool = run_analysis.RunAnalysisTool()
        self.tool._run_widget = Mock(spec=run_analysis.RunAnalysisPanel)
        self.tool._dialog = Mock(spec=ClosableDialog)

    def tearDown(self):
        ErtTest.tearDown(self)

        self.tool._run_widget.reset_mock()
        self.tool._dialog.reset_mock()

    @patch("ert_gui.tools.run_analysis.run_analysis_tool.analyse", return_value=True)
    @patch("ert_gui.tools.run_analysis.run_analysis_tool.QMessageBox")
    def test_show_dialogue_at_success(self, mock_messagebox, mock_analyse):
        self.tool._run_widget.source_case.return_value = "source"
        self.tool._run_widget.target_case.return_value = "target"

        with patch("ert_gui.tools.run_analysis.run_analysis_tool.ert_shared.ERT"):
            self.tool.run()

        mock_analyse.assert_called_once_with("target", "source")
        mock_messagebox.return_value.setText.assert_called_once_with(
            "Successfully ran analysis for case 'source'."
        )
        self.tool._dialog.accept.assert_called_once_with()

    @patch("ert_gui.tools.run_analysis.run_analysis_tool.analyse", return_value=False)
    @patch("ert_gui.tools.run_analysis.run_analysis_tool.QMessageBox")
    def test_show_dialogue_at_failure(self, mock_messagebox, mock_analyse):
        self.tool._run_widget.source_case.return_value = "source"
        self.tool._run_widget.target_case.return_value = "target"

        self.tool.run()

        mock_analyse.assert_called_once_with("target", "source")
        mock_messagebox.return_value.setText.assert_called_once_with(
            "Unable to run analysis for case 'source'."
        )
        self.tool._dialog.accept.assert_not_called()
