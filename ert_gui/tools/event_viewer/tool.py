from ert_gui.ertwidgets import resourceIcon
from ert_gui.tools import Tool
from ert_gui.tools.event_viewer import EventViewerPanel


class EventViewerTool(Tool):
    def __init__(self, gui_handler):
        super().__init__(
            "Event viewer",
            "tools/event_viewer",
            resourceIcon("notifications.svg"),
        )
        self.log_handler = gui_handler
        self.logging_window = EventViewerPanel(self.log_handler)
        self.setEnabled(True)

    def trigger(self):
        self.logging_window.show()
