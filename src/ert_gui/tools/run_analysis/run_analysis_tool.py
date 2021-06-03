#  Copyright (C) 2016  Equinor ASA, Norway.
#
#  The file 'run_analysis_tool.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.

from qtpy.QtWidgets import QMessageBox

from res.enkf import ErtRunContext
from res.enkf import ESUpdate

from ert_gui.ertwidgets import resourceIcon
from ert_gui.ertwidgets.closabledialog import ClosableDialog
from ert_gui.tools import Tool
from ert_gui.tools.run_analysis import RunAnalysisPanel
import ert_shared


def analyse(target, source):
    """Runs analysis using target and source cases. Returns whether or not
    the analysis was successful."""
    ert = ert_shared.ERT.ert
    fs_manager = ert.getEnkfFsManager()
    es_update = ESUpdate(ert)

    target_fs = fs_manager.getFileSystem(target)
    source_fs = fs_manager.getFileSystem(source)
    run_context = ErtRunContext.ensemble_smoother_update(
        source_fs,
        target_fs,
    )
    return es_update.smootherUpdate(run_context)


class RunAnalysisTool(Tool):
    def __init__(self):
        super(RunAnalysisTool, self).__init__(
            "Run Analysis", "tools/run_analysis", resourceIcon("ide/table_import")
        )
        self._run_widget = None
        self._dialog = None
        self._selected_case_name = None

    def trigger(self):
        if self._run_widget is None:
            self._run_widget = RunAnalysisPanel()
        self._dialog = ClosableDialog("Run Analysis", self._run_widget, self.parent())
        self._dialog.addButton("Run", self.run)
        self._dialog.exec_()

    def run(self):
        target = self._run_widget.target_case()
        source = self._run_widget.source_case()

        if len(target) == 0:
            self._report_empty_target()
            return

        success = analyse(target, source)

        msg = QMessageBox()
        msg.setWindowTitle("Run Analysis")
        msg.setStandardButtons(QMessageBox.Ok)

        if success:
            msg.setIcon(QMessageBox.Information)
            msg.setText("Successfully ran analysis for case '{}'.".format(source))
            msg.exec_()
        else:
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Unable to run analysis for case '{}'.".format(source))
            msg.exec_()
            return

        ert_shared.ERT.ertChanged.emit()
        self._dialog.accept()

    def _report_empty_target(self):
        msg = QMessageBox()
        msg.setWindowTitle("Invalid Target")
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Target case can not be empty")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
