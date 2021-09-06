#  Copyright (C) 2014  Equinor ASA, Norway.
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
from weakref import ref

from ert_gui.ertwidgets import resourceIcon
from ert_gui.ertwidgets.closabledialog import ClosableDialog
from ert_gui.tools import Tool
from ert_gui.tools.export import ExportPanel
from ert_shared.exporter import Exporter
from qtpy.QtWidgets import QMessageBox
import logging


class ExportTool(Tool):
    def __init__(self):
        super(ExportTool, self).__init__(
            "Export Data", "tools/export", resourceIcon("ide/table_export")
        )
        self.__export_widget = None
        self.__dialog = None
        self.__exporter = Exporter()
        self.setEnabled(self.__exporter.is_valid())

    def trigger(self):
        if self.__export_widget is None:
            self.__export_widget = ref(ExportPanel(self.parent()))
            self.__export_widget().runExport.connect(self._run_export)

        self.__dialog = ref(
            ClosableDialog("Export", self.__export_widget(), self.parent())
        )
        self.__export_widget().updateExportButton.connect(self.__dialog().toggleButton)
        self.__dialog().addButton("Export", self.export)
        self.__dialog().show()

    def _run_export(self, params):
        try:
            self.__exporter.run_export(params)
            QMessageBox.information(
                None, "Success", """Export completed!""", QMessageBox.Ok
            )
        except UserWarning as usrwarning:
            logging.error(str(usrwarning))
            QMessageBox.warning(
                None,
                "Failure",
                """Export failed with the following message:\n{}""".format(
                    str(usrwarning)
                ),
                QMessageBox.Ok,
            )

    def export(self):
        self.__export_widget().export()
        self.__dialog().accept()
