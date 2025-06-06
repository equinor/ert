import logging

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QIcon

from ert.gui.tools import Tool

from .panel import EventViewerPanel, GUILogHandler

logger = logging.getLogger(__name__)


class EventViewerTool(Tool, QObject):
    def __init__(
        self,
        gui_handler: GUILogHandler,
        config_filename: str | None = None,
        trigger_source: str = "",
    ) -> None:
        super().__init__(
            "Event viewer",
            QIcon("img:notifications.svg"),
        )
        self.log_handler = gui_handler
        self.logging_window = EventViewerPanel(self.log_handler)
        if config_filename:
            self.logging_window.setWindowTitle(f"Event viewer: {config_filename}")
        self.trigger_source = trigger_source
        self.setEnabled(True)

    def trigger(self) -> None:
        logger.info(
            f"WorkflowsTool triggered from {self.trigger_source}"
            if self.trigger_source
            else ""
        )
        self.logging_window.show()

    @Slot()
    def close_wnd(self) -> None:
        self.logging_window.close()
