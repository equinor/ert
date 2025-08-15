import logging
from typing import Any

from PyQt6.QtGui import QIcon

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.load_results import LoadResultsPanel

logger = logging.getLogger(__name__)


class LoadResultsTool(Tool):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier) -> None:
        super().__init__(
            "Load results manually",
            QIcon("img:upload.svg"),
        )
        self._import_widget: LoadResultsPanel | None = None
        self._dialog: ClosableDialog | None = None
        self._notifier = notifier
        self._config = config

    def trigger(self) -> None:
        if self._import_widget is None:
            self._import_widget = LoadResultsPanel(self._config, self._notifier)
            self._import_widget.panelConfigurationChanged.connect(
                self.validationStatusChanged
            )
            self._dialog = ClosableDialog(
                "Load results manually",
                self._import_widget,
                self.parent(),  # type: ignore
            )
            self._loadButton = self._dialog.addButton("Load", self.load)
            self._dialog.setObjectName("load_results_manually_tool")

        else:
            self._import_widget.refresh()

        if not self._import_widget._ensemble_selector.isEnabled():
            self._loadButton.setEnabled(False)
            self._loadButton.setToolTip("Must load into a ensemble")
        assert self._dialog is not None
        self._dialog.exec()

    def load(self, _: Any) -> None:
        logger.info("Gui utility: LoadResults tool was used")
        assert self._dialog is not None
        assert self._import_widget is not None
        self._dialog.disableCloseButton()
        self._dialog.toggleButton(caption="Load", enabled=False)
        self._import_widget.load()
        self._dialog.accept()

    def validationStatusChanged(self) -> None:
        assert self._dialog is not None
        assert self._import_widget is not None
        self._dialog.toggleButton(
            caption="Load", enabled=self._import_widget.isConfigurationValid()
        )
