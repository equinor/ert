from math import floor

from qtpy.QtCore import QSize, Qt, QThread, Slot
from qtpy.QtGui import QClipboard, QFontDatabase, QTextCursor, QTextOption
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

from .file_update_worker import FileUpdateWorker


class FileDialog(QDialog):
    def __init__(
        self, file_name, job_name, job_number, realization, iteration, parent=None
    ):
        super().__init__(parent)

        self.setWindowTitle(
            f"{job_name} # {job_number} "
            f"Realization: {realization} Iteration: {iteration}"
        )

        try:
            # pylint: disable=consider-using-with
            # We take care to close this file in _quit_thread()
            self._file = open(file_name, "r", encoding="utf-8")
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
    def _quit_thread(self):
        self._file.close()
        self._thread.quit()
        self._thread.wait()

    def _calculate_font_based_width(self):
        font_metrics = self._view.fontMetrics()
        desired_width_in_characters = 120
        extra_bit_of_margin_space = 2
        extra_space_for_vertical_scroll_bar = 5
        return (
            desired_width_in_characters
            + extra_bit_of_margin_space
            + extra_space_for_vertical_scroll_bar
        ) * font_metrics.averageCharWidth()

    def _init_layout(self):
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok)  # type: ignore
        dialog_buttons.accepted.connect(self.accept)

        self._copy_all_button = dialog_buttons.addButton(
            "Copy all", QDialogButtonBox.ActionRole  # type: ignore
        )
        self._copy_all_button.clicked.connect(self._copy_all)

        self._follow_button = dialog_buttons.addButton(
            "Follow", QDialogButtonBox.ActionRole  # type: ignore
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
        self.finished.connect(self._quit_thread)

        self._thread.start()

    def _copy_all(self) -> None:
        text = self._view.toPlainText()
        QApplication.clipboard().setText(text, QClipboard.Clipboard)  # type: ignore

    def _update_cursor(self, value: int) -> None:
        if not self._view.textCursor().hasSelection():
            block = self._view.document().findBlockByLineNumber(value)
            cursor = QTextCursor(block)
            self._view.setTextCursor(cursor)

    def _enable_follow_mode(self, enable: bool) -> None:
        if enable:
            self._view.moveCursor(QTextCursor.End)  # type: ignore
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
            self._view.verticalScrollBar().setDisabled(True)
            self._view.setTextInteractionFlags(Qt.NoTextInteraction)  # type: ignore
            self._follow_mode = True
        else:
            self._view.verticalScrollBar().setDisabled(False)
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore
            self._view.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard  # type: ignore
            )
            self._follow_mode = False

    def _append_text(self, text: str) -> None:
        # Remove trailing newline as appendPlainText adds this
        if text[-1:] == "\n":
            text = text[:-1]
        if self._follow_mode:
            self._view.moveCursor(QTextCursor.End)  # type: ignore
        self._view.appendPlainText(text)
        self.adjustSize()

    def sizeHint(self) -> QSize:
        return QSize(
            self._calculate_font_based_width(),
            _calculate_screen_size_based_height(),
        )


def _calculate_screen_size_based_height():
    screen_height = QApplication.primaryScreen().geometry().height()
    max_ratio_of_screen = 1.0 / 3.0
    return floor(screen_height * max_ratio_of_screen)
