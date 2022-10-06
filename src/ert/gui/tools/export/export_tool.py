import logging
from weakref import ref

from qtpy.QtWidgets import QMessageBox

from ert.gui.ertwidgets import resourceIcon
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.export import ExportPanel
from ert.shared.exporter import Exporter


class ExportTool(Tool):
    def __init__(self, ert):
        super().__init__("Export data", "tools/export", resourceIcon("share.svg"))
        self.__export_widget = None
        self.__dialog = None
        self.__exporter = Exporter(ert)
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
                f"Export failed with the following message:\n{usrwarning}",
                QMessageBox.Ok,
            )

    def export(self):
        self.__export_widget().export()
        self.__dialog().accept()
