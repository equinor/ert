from ert_gui.tools import Tool
from ert_gui.tools.ide import IdeWindow
from ert_gui.widgets import util


class IdeTool(Tool):
    def __init__(self, path):
        super(IdeTool, self).__init__("Configure", util.resourceIcon("ide/cog_edit"))

        self.ide_window = None
        self.path = path

    def trigger(self):
        if self.ide_window is None:
            self.ide_window = IdeWindow(self.path, self.parent())

        self.ide_window.show()