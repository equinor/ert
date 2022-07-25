from qtpy.QtWidgets import QVBoxLayout, QPlainTextEdit

from qtpy import QtCore
import logging


class GUILogHandler(logging.Handler, QtCore.QObject):
    """
    Log handler which will emit a qt signal every time a
    log is emitted
    """

    append_log_statement = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        self.setLevel(logging.INFO)

        QtCore.QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.append_log_statement.emit(msg)


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

    @QtCore.Slot(str)
    def val_changed(self, value):
        self.text_box.appendPlainText(value)
