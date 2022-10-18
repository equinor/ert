from ert.gui.ertwidgets import resourceIcon, showWaitCursorWhileWaiting
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.load_results import LoadResultsPanel


class LoadResultsTool(Tool):
    def __init__(self, facade):
        self.facade = facade
        super().__init__(
            "Load results manually",
            "tools/load_manually",
            resourceIcon("upload.svg"),
        )
        self.__import_widget = None
        self.__dialog = None
        self.setEnabled(self.is_valid_run_path())

    def trigger(self):
        if self.__import_widget is None:
            self.__import_widget = LoadResultsPanel(self.facade)
        self.__dialog = ClosableDialog(
            "Load results manually", self.__import_widget, self.parent()
        )
        self.__import_widget.setCurrectCase()
        self.__dialog.addButton("Load", self.load)
        self.__dialog.exec_()

    @showWaitCursorWhileWaiting
    def load(self, _):
        self.__dialog.disableCloseButton()
        self.__dialog.toggleButton(caption="Load", enabled=False)
        self.__import_widget.load()
        self.__dialog.accept()

    def is_valid_run_path(self) -> bool:
        """A run path is considered valid if we can
        insert realisation and iteration numbers"""
        try:
            # pylint: disable=pointless-statement
            self.facade.run_path % (0, 0)
            return True
        except TypeError:
            try:
                # pylint: disable=pointless-statement
                self.facade.run_path % 0
                return True
            except TypeError:
                return False
