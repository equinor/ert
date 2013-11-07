from weakref import ref
from ert_gui.tools import Tool
from ert_gui.tools.ide import IdeWindow
from ert_gui.widgets import util


class IdeTool(Tool):
    def __init__(self, path):
        super(IdeTool, self).__init__("Configure", util.resourceIcon("ide/widgets"))

        self.ide_window = None
        self.path = path

    def trigger(self):
        if self.ide_window is None:
            self.ide_window = ref(IdeWindow(self.path, self.parent()))

        self.ide_window().show()
        self.ide_window().activateWindow()