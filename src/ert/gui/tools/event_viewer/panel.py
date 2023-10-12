import logging
from contextlib import contextmanager
from typing import Iterator

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QPlainTextEdit, QVBoxLayout


class _Signal(QObject):
    append_log_statement = Signal(str)

    def __init__(self):
        super().__init__()


class GUILogHandler(logging.Handler, QObject):
    """
    Log handler which will emit a qt signal every time a
    log is emitted
    """

    def __init__(self):
        super().__init__()
        self._signal = _Signal()
        self.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        self.setLevel(logging.INFO)

    def emit(self, record):
        msg = self.format(record)
        self.append_log_statement.emit(msg)

    @property
    def append_log_statement(self):
        return self._signal.append_log_statement


class EventViewerPanel(QPlainTextEdit):
    def __init__(self, log_handler: GUILogHandler):
        self.log_handler = log_handler
        QPlainTextEdit.__init__(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self._dynamic = False

        self.setWindowTitle("Event viewer")
        self.activateWindow()

        layout = QVBoxLayout()
        self.text_box = QPlainTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setMaximumBlockCount(1000)
        layout.addWidget(self.text_box)

        self.setLayout(layout)
        log_handler.append_log_statement.connect(self.val_changed)

    def val_changed(self, value):
        self.text_box.appendPlainText(value)


@contextmanager
def add_gui_log_handler() -> Iterator[GUILogHandler]:
    """
    Context manager for the GUILogHandler class. Will make sure that the handler
    is removed prior to program exit.
    """
    logger = logging.getLogger()

    handler = GUILogHandler()
    logger.addHandler(handler)

    yield handler

    logger.removeHandler(handler)
