from os.path import basename

from weakref import ref

from ert_shared import ERT
from ert_gui.ertwidgets import resourceIcon
from ert_gui.tools import Tool
from ert_gui.tools.ide import IdeWindow


class IdeTool(Tool):
    def __init__(self, config_file, help_tool):
        super(IdeTool, self).__init__(
            "Configure", "tools/ide", resourceIcon("ide/widgets")
        )

        self.ide_window = None
        self.config_file = config_file
        self.path = basename(config_file)
        self.help_tool = help_tool

    def trigger(self):
        if self.ide_window is None:
            self.ide_window = ref(
                IdeWindow(self.config_file, self.parent(), self.help_tool)
            )
            self.ide_window().reloadTriggered.connect(ERT.reloadERT)

        self.ide_window().show()
        self.ide_window().raise_()
        self.ide_window().activateWindow()
