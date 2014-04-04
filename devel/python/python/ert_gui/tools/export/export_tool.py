#  Copyright (C) 2011  Statoil ASA, Norway.
#
#  The file '__init__.py' is part of ERT - Ensemble based Reservoir Tool.
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

from ert_gui.tools import Tool
from ert_gui.tools.export import ExportWindow, ExportPanel
from ert_gui.widgets import util
from ert_gui.widgets.closable_dialog import ClosableDialog


class ExportTool(Tool):
    def __init__(self):
        super(ExportTool, self).__init__("Export Data", "tools/export", util.resourceIcon("ide/table_export"))

    def trigger(self):
        run_workflow_widget = ExportPanel()
        dialog = ClosableDialog("Export", run_workflow_widget, self.parent())
        dialog.exec_()
        #export_window = ExportWindow(self.parent())
        #export_window.show()

