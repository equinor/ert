import logging
from collections.abc import Iterator
from contextlib import contextmanager

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QDialog, QPlainTextEdit, QVBoxLayout

from ert.gui.tools.search_bar import SearchBar

# Need to separate GUILogHandler into _Signaler & _GUILogHandler
# to avoid a object lifetime issue where logging keeps around a reference
# to the handler until application exit

logger = logging.getLogger(__name__)


class _Signaler(QObject):
    append_log_statement = Signal(str)


class _GUILogHandler(logging.Handler):
    def __init__(self, signaler: _Signaler) -> None:
        super().__init__()
        self.signaler = signaler

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.signaler.append_log_statement.emit(msg)


class GUILogHandler(_Signaler):
    """
    Log handler which will emit a qt signal every time a
    log is emitted
    """

    def __init__(self) -> None:
        super().__init__()

        self.handler = _GUILogHandler(self)
        self.handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        self.handler.setLevel(logging.INFO)

    @property
    def level(self) -> int:
        return self.handler.level

    def handle(self, record: logging.LogRecord) -> bool:
        return self.handler.handle(record)


class EventViewerPanel(QDialog):
    def __init__(self, log_handler: GUILogHandler) -> None:
        super().__init__()
        self.log_handler = log_handler

        self.setMinimumWidth(500)
        self.setMinimumHeight(800)
        self._dynamic = False

        self.setWindowTitle("Event viewer")
        self.activateWindow()

        layout = QVBoxLayout()
        self.text_box = QPlainTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setMaximumBlockCount(1000)
        self.search_bar = SearchBar(self.text_box)
        layout.addLayout(self.search_bar.get_layout())
        layout.addWidget(self.text_box)
        self.setLayout(layout)
        log_handler.append_log_statement.connect(self.val_changed)

    @Slot(str)
    def val_changed(self, value: str) -> None:
        self.text_box.appendPlainText(value)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        logger.info("Gui utility: EventViewer tool was used")
        super().closeEvent(event)


@contextmanager
def add_gui_log_handler() -> Iterator[GUILogHandler]:
    """
    Context manager for the GUILogHandler singleton. Will make sure that the
    handler is removed prior to program exit.
    """
    logger = logging.getLogger()

    gui_handler = GUILogHandler()
    logger.addHandler(gui_handler.handler)

    yield gui_handler

    logger.removeHandler(gui_handler.handler)
