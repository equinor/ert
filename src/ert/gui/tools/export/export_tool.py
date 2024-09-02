from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast, no_type_check

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QMessageBox, QWidget

if TYPE_CHECKING:
    from ert.config import ErtConfig

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.tools import Tool
from ert.gui.tools.export.export_panel import ExportDialog

from .exporter import Exporter

logger = logging.getLogger(__name__)


class ExportTool(Tool):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier):
        super().__init__("Export data", QIcon("img:share.svg"))
        export_job = config.workflow_jobs.get("CSV_EXPORT")
        self.setEnabled(export_job is not None)
        self.__exporter = Exporter(
            export_job,
            notifier,
            config,
        )
        self.config = config
        self.notifier = notifier

    @no_type_check
    def trigger(self) -> None:
        dialog = ExportDialog(self.config, self.notifier.storage, self.parent())
        success = dialog.showAndTell()

        if success:
            self._run_export(
                [
                    dialog.output_path,
                    dialog.ensemble_data_as_json,
                    dialog.design_matrix_path,
                    True,
                    dialog.drop_const_columns,
                ]
            )

    def _run_export(self, params: list[Any]) -> None:
        try:
            self.__exporter.run_export(params)
            QMessageBox.information(
                cast(QWidget, self.parent()),
                "Success",
                """Export completed!""",
                QMessageBox.Ok,
            )
        except UserWarning as usrwarning:
            logger.error(str(usrwarning))
            QMessageBox.warning(
                cast(QWidget, self.parent()),
                "Failure",
                f"Export failed with the following message:\n{usrwarning}",
                QMessageBox.Ok,
            )
