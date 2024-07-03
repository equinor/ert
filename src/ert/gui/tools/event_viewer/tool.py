from typing import Optional

from qtpy.QtCore import QObject, Slot
from qtpy.QtGui import QIcon

from ert.gui.tools import Tool
from ert.gui.tools.event_viewer import EventViewerPanel, GUILogHandler


class EventViewerTool(Tool, QObject):
    def __init__(
        self, gui_handler: GUILogHandler, config_filename: Optional[str] = None
    ):
        super().__init__(
            "Event viewer",
            QIcon("img:notifications.svg"),
        )
        self.log_handler = gui_handler
        self.logging_window = EventViewerPanel(self.log_handler)
        if config_filename:
            self.logging_window.setWindowTitle(f"Event viewer: {config_filename}")
        self.setEnabled(True)

    def trigger(self) -> None:
        self.logging_window.show()

    @Slot()
    def close_wnd(self) -> None:
        self.logging_window.close()
