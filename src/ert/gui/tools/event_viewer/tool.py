from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QIcon

from ert.gui.tools import Tool

from .panel import EventViewerPanel, GUILogHandler


class EventViewerTool(Tool, QObject):
    def __init__(
        self, gui_handler: GUILogHandler, config_filename: str | None = None
    ) -> None:
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
