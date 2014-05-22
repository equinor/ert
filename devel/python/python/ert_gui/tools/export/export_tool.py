#  Copyright (C) 2014  Statoil ASA, Norway.
#
#  The file 'export_tool.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ert_gui.models.connectors.init import CaseSelectorModel

from ert_gui.tools import Tool
from ert_gui.tools.export import ExportPanel
from ert_gui.widgets import util
from ert_gui.widgets.closable_dialog import ClosableDialog


class ExportTool(Tool):
    def __init__(self):
        super(ExportTool, self).__init__("Export Data", "tools/export", util.resourceIcon("ide/table_export"))
        self.__export_widget = None
        self.__dialog = None

    def trigger(self):
        if self.__export_widget is None:
            self.__export_widget = ExportPanel()

        self.__export_widget.setSelectedCase(CaseSelectorModel().getCurrentChoice())
        self.__dialog = ClosableDialog("Export", self.__export_widget, self.parent())
        self.__export_widget.updateExportButton.connect(self.__dialog.toggelButton)
        self.__dialog.addButton("Export", self.export)
        self.__dialog.exec_()

    def export(self):
        self.__export_widget.export()
        self.__dialog.accept()
