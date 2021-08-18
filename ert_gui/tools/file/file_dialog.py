from qtpy.QtCore import QThread, Slot, Qt
from qtpy.QtWidgets import (
    QDialog,
    QMessageBox,
    QDialogButtonBox,
    QVBoxLayout,
    QPlainTextEdit,
)
from qtpy.QtGui import QTextOption, QTextCursor, QFont, QFontDatabase

from .file_update_worker import FileUpdateWorker


class FileDialog(QDialog):
    def __init__(
        self, file_name, job_name, job_number, realization, iteration, parent=None
    ):
        super(FileDialog, self).__init__(parent)

        self.setWindowTitle(
            "{} # {} Realization: {} Iteration: {}".format(
                job_name, job_number, realization, iteration
            )
        )

        try:
            self._file = open(file_name, "r")
        except OSError as error:
            self._mb = QMessageBox(
                QMessageBox.Critical,
                "Error opening file",
                error.strerror,
                QMessageBox.Ok,
                self,
            )
            self._mb.finished.connect(self.accept)
            self._mb.show()
            return

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setWordWrapMode(QTextOption.NoWrap)

        # There isn't a standard way of getting the system default monospace
        # font in Qt4 (it was introduced in Qt5.2). If QFontDatabase.FixedFont
        # exists, then we can assume that this functionality exists and ask for
        # the correct font directly. Otherwise we ask for a font that doesn't
        # exist and specify our requirements. Qt then finds an existing font
        # that best matches our parameters.
        if hasattr(QFontDatabase, "systemFont") and hasattr(QFontDatabase, "FixedFont"):
            font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        else:
            font = QFont("")
            font.setFixedPitch(True)
            font.setStyleHint(QFont.Monospace)
        self._view.setFont(font)

        self._init_layout()
        self._init_thread()

        self.show()

    @Slot()
    def _stop_thread(self):
        self._thread.quit()
        self._thread.wait()

    def _init_layout(self):
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        dialog_buttons.accepted.connect(self.accept)

        self._follow = dialog_buttons.addButton("Follow", QDialogButtonBox.ActionRole)
        self._follow.setCheckable(True)
        self._follow.toggled.connect(self._enable_follow_mode)
        self._enable_follow_mode(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self._view)
        layout.addWidget(dialog_buttons)

    def _init_thread(self):
        self._thread = QThread()

        self._worker = FileUpdateWorker(self._file)
        self._worker.moveToThread(self._thread)
        self._worker.read.connect(self._append_text)

        self._thread.started.connect(self._worker.setup)
        self._thread.finished.connect(self._worker.stop)
        self._thread.finished.connect(self._worker.deleteLater)
        self.finished.connect(self._stop_thread)

        self._thread.start()

    def _enable_follow_mode(self, b: bool) -> None:
        if b:
            self._view.moveCursor(QTextCursor.End)
            self._view.setCenterOnScroll(False)
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        else:
            self._view.setCenterOnScroll(True)
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def _append_text(self, s: str) -> None:
        # Save current selection before inserting text at the end
        cursor = self._view.textCursor()
        self._view.moveCursor(QTextCursor.End)
        self._view.insertPlainText(s)
        self._view.setTextCursor(cursor)
