from qtpy.QtCore import QObject, Slot
from qtpy.QtGui import QIcon

from ert.gui.tools import Tool
from ert.gui.tools.event_viewer import EventViewerPanel


class EventViewerTool(Tool, QObject):
    def __init__(self, gui_handler):
        super().__init__(
            "Event viewer",
            QIcon("img:notifications.svg"),
        )
        self.log_handler = gui_handler
        self.logging_window = EventViewerPanel(self.log_handler)
        self.setEnabled(True)

    def trigger(self):
        self.logging_window.show()

    @Slot()
    def close_wnd(self):
        self.logging_window.close()
