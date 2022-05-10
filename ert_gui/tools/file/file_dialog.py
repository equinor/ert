from qtpy.QtCore import QThread, Slot, Qt, QSize
from math import ceil, floor
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
        super().__init__(parent)

        self.setWindowTitle(
            f"{job_name} # {job_number} "
            f"Realization: {realization} Iteration: {iteration}"
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

    def _get_file_line_count(self):
        return len(self._view.toPlainText().split("\n"))

    def _get_length_of_files_longest_line(self):
        return max(map(len, self._view.toPlainText().split("\n")))

    def _calculate_desired_width(self):
        font_metrics = self._view.fontMetrics()
        chars_in_longest_line = self._get_length_of_files_longest_line()
        extra_bit_of_margin_space = 2
        extra_space_for_vertical_scroll_bar = 5
        font_based_width = (
            chars_in_longest_line
            + extra_bit_of_margin_space
            + extra_space_for_vertical_scroll_bar
        ) * font_metrics.averageCharWidth()
        screen_width = QApplication.primaryScreen().geometry().width()
        max_ratio_of_screen = 1.0 / 3.0
        max_width = floor(screen_width * max_ratio_of_screen)
        return min(font_based_width, max_width)

    def _calculate_desired_height(self):
        font_metrics = self._view.fontMetrics()

        line_height = ceil(font_metrics.lineSpacing())
        extra_lines_to_make_it_actually_fit = 5
        extra_lines_for_horizontal_scroll_bar = 3
        file_line_count = self._get_file_line_count()
        screen_height = QApplication.primaryScreen().geometry().height()
        max_ratio_of_screen = 1.0 / 3.0
        show_max_lines = floor(screen_height * max_ratio_of_screen / line_height)
        min_lines = min(
            file_line_count
            + extra_lines_to_make_it_actually_fit
            + extra_lines_for_horizontal_scroll_bar,
            show_max_lines,
        )
        font_based_height = min_lines * line_height
        return font_based_height

    def _init_layout(self):
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        dialog_buttons.accepted.connect(self.accept)

        self._copy_all_button = dialog_buttons.addButton(
            "Copy all", QDialogButtonBox.ActionRole
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
        self.adjustSize()

    def sizeHint(self) -> QSize:
        width = self._calculate_desired_width()
        height = self._calculate_desired_height()
        return QSize(width, height)
