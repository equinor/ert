from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import resourceIcon, showWaitCursorWhileWaiting
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.load_results import LoadResultsPanel
from ert.libres_facade import LibresFacade


class LoadResultsTool(Tool):
    def __init__(self, facade: LibresFacade, notifier: ErtNotifier):
        self.facade = facade
        super().__init__(
            "Load results manually",
            "tools/load_manually",
            resourceIcon("upload.svg"),
        )
        self.__import_widget = None
        self.__dialog = None
        self._notifier = notifier

    def trigger(self):
        if self.__import_widget is None:
            self.__import_widget = LoadResultsPanel(self.facade, self._notifier)
        self.__dialog = ClosableDialog(
            "Load results manually", self.__import_widget, self.parent()
        )
        self.__dialog.setObjectName("load_results_manually_tool")
        self.__import_widget.setCurrentCase()
        self.__dialog.addButton("Load", self.load)
        self.__dialog.exec_()

    @showWaitCursorWhileWaiting
    def load(self, _):
        self.__dialog.disableCloseButton()
        self.__dialog.toggleButton(caption="Load", enabled=False)
        self.__import_widget.load()
        self.__dialog.accept()
