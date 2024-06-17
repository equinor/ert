from typing import Any, Optional

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QPushButton

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import showWaitCursorWhileWaiting
from ert.gui.ertwidgets.closabledialog import ClosableDialog
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
        self._import_widget: Optional[LoadResultsPanel] = None
        self._dialog: Optional[ClosableDialog] = None
        self._notifier = notifier

    def trigger(self) -> None:
        if self._import_widget is None:
            self._import_widget = LoadResultsPanel(self.facade, self._notifier)
        self._dialog = ClosableDialog(
            "Load results manually",
            self._import_widget,
            self.parent(),  # type: ignore
        )
        self._dialog.setObjectName("load_results_manually_tool")
        self._dialog.addButton("Load", self.load)
        if not self._import_widget._ensemble_selector.isEnabled():
            button = self._dialog.findChild(QPushButton, "Load")
            button.setEnabled(False)
            button.setToolTip("Must load into a ensemble")
        self._dialog.exec_()

    @showWaitCursorWhileWaiting
    def load(self, _: Any) -> None:
        assert self._dialog is not None
        assert self._import_widget is not None
        self._dialog.disableCloseButton()
        self._dialog.toggleButton(caption="Load", enabled=False)
        self._import_widget.load()
        self._dialog.accept()
