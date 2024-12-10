from typing import Any

from qtpy.QtGui import QIcon

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.load_results import LoadResultsPanel
from ert.libres_facade import LibresFacade


class LoadResultsTool(Tool):
    def __init__(self, facade: LibresFacade, notifier: ErtNotifier) -> None:
        self.facade = facade
        super().__init__(
            "Load results manually",
            QIcon("img:upload.svg"),
        )
        self._import_widget: LoadResultsPanel | None = None
        self._dialog: ClosableDialog | None = None
        self._notifier = notifier

    def trigger(self) -> None:
        if self._import_widget is None:
            self._import_widget = LoadResultsPanel(self.facade, self._notifier)
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
        self._dialog.exec_()

    def load(self, _: Any) -> None:
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
