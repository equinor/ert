from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, no_type_check
from weakref import ref

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from ert.config import ErtConfig

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.export import ExportPanel
from ert.shared.exporter import Exporter


class ExportTool(Tool):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier):
        super().__init__("Export data", QIcon("img:share.svg"))
        self.__export_widget = None
        self.__dialog = None
        self.__exporter = Exporter(
            config.workflow_jobs.get("CSV_EXPORT2"),
            config.workflow_jobs.get("EXPORT_RUNPATH"),
            notifier,
            config.runpath_file,
        )
        self.setEnabled(self.__exporter.is_valid())

    @no_type_check
    def trigger(self) -> None:
        if self.__export_widget is None:
            self.__export_widget = ref(ExportPanel(self.parent()))
            self.__export_widget().runExport.connect(self._run_export)

        self.__dialog = ref(
            ClosableDialog("Export", self.__export_widget(), self.parent())
        )
        self.__export_widget().updateExportButton.connect(self.__dialog().toggleButton)
        self.__dialog().addButton("Export", self.export)
        self.__dialog().show()

    def _run_export(self, params: dict[str, Any]) -> None:
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

    @no_type_check
    def export(self) -> None:
        self.__export_widget().export()
        self.__dialog().accept()
