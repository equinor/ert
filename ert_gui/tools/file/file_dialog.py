from qtpy.QtCore import QThread, Slot, Qt, QEvent
from qtpy.QtWidgets import (
    QDialog,
    QMessageBox,
    QDialogButtonBox,
    QVBoxLayout,
    QPlainTextEdit,
    QApplication,
)
from qtpy.QtGui import QTextOption, QTextCursor, QClipboard, QFontDatabase

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
        # for moving the actual slider
        self._view.verticalScrollBar().sliderMoved.connect(self._update_cursor)
        # for mouse wheel and keyboard arrows
        self._view.verticalScrollBar().valueChanged.connect(self._update_cursor)

        self._view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        self._follow_mode = False

        self._init_layout()
        self._init_thread()

        self.show()

    @Slot()
    def _stop_thread(self):
        self._thread.quit()
        self._thread.wait()

    def _init_layout(self):
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        dialog_buttons.accepted.connect(self.accept)

        self._copy_all_button = dialog_buttons.addButton(
            "Copy All", QDialogButtonBox.ActionRole
        )
        self._copy_all_button.clicked.connect(self._copy_all)

        self._follow_button = dialog_buttons.addButton(
            "Follow", QDialogButtonBox.ActionRole
        )
        self._follow_button.setCheckable(True)
        self._follow_button.toggled.connect(self._enable_follow_mode)
        self._enable_follow_mode(self._follow_mode)

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

    def _copy_all(self) -> None:
        text = self._view.toPlainText()
        QApplication.clipboard().setText(text, QClipboard.Clipboard)
        pass

    def _update_cursor(self, value: int) -> None:
        if not self._view.textCursor().hasSelection():
            block = self._view.document().findBlockByLineNumber(value)
            cursor = QTextCursor(block)
            self._view.setTextCursor(cursor)

    def _enable_follow_mode(self, enable: bool) -> None:
        if enable:
            self._view.moveCursor(QTextCursor.End)
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._view.verticalScrollBar().setDisabled(True)
            self._view.setTextInteractionFlags(Qt.NoTextInteraction)
            self._follow_mode = True
        else:
            self._view.verticalScrollBar().setDisabled(False)
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._view.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            self._follow_mode = False

    def _append_text(self, text: str) -> None:
        # Remove trailing newline as appendPlainText adds this
        if text[-1:] == "\n":
            text = text[:-1]
        if self._follow_mode:
            self._view.moveCursor(QTextCursor.End)
        self._view.appendPlainText(text)
