from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import (
    QClipboard,
    QFontDatabase,
    QTextCursor,
    QTextOption,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.gui.tools.search_bar import SearchBar

from .file_update_worker import FileUpdateWorker


class FileDialog(QDialog):
    def __init__(
        self,
        file_name: str,
        step_name: str,
        step_number: int,
        realization: int,
        iteration: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(
            f"{step_name} # {step_number} "
            f"Realization: {realization} Iteration: {iteration}"
        )
        self.setObjectName(file_name)

        try:
            # We take care to close this file in _quit_thread()
            self._file = open(file_name, encoding="utf-8")  # noqa: SIM115
        except OSError as error:
            self._mb = QMessageBox(
                QMessageBox.Icon.Critical,
                "Error opening file",
                error.strerror or "",
                QMessageBox.StandardButton.Ok,
                self,
            )
            self._mb.finished.connect(self.accept)
            self._mb.show()
            return

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        # for moving the actual slider
        scroll_bar = self._view.verticalScrollBar()
        assert scroll_bar is not None
        scroll_bar.sliderMoved.connect(self._update_cursor)
        # for mouse wheel and keyboard arrows
        scroll_bar.valueChanged.connect(self._update_cursor)

        self._view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._search_bar = SearchBar(self._view)
        self._follow_mode = False

        self._init_layout()
        self._init_thread()

        self.show()

    def _quit_thread(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self._file.close()

    def _init_layout(self) -> None:
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        dialog_buttons.rejected.connect(self.reject)

        self._copy_all_button = dialog_buttons.addButton(
            "Copy all",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        assert self._copy_all_button is not None
        self._copy_all_button.clicked.connect(self._copy_all)

        self._follow_button = dialog_buttons.addButton(
            "Follow",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        assert self._follow_button is not None
        self._follow_button.setCheckable(True)
        self._follow_button.toggled.connect(self._enable_follow_mode)
        self._enable_follow_mode(self._follow_mode)
        layout = QVBoxLayout(self)
        layout.addLayout(self._search_bar.get_layout())
        layout.addWidget(self._view)
        layout.addWidget(dialog_buttons)

    def _init_thread(self) -> None:
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
        clipboard = QApplication.clipboard()
        assert clipboard is not None
        clipboard.setText(text, QClipboard.Mode.Clipboard)

    def _update_cursor(self, value: int) -> None:
        if not self._view.textCursor().hasSelection():
            document = self._view.document()
            assert document is not None
            block = document.findBlockByLineNumber(value)
            cursor = QTextCursor(block)
            self._view.setTextCursor(cursor)

    def _enable_follow_mode(self, enable: bool) -> None:
        vertical_scroll_bar = self._view.verticalScrollBar()
        assert vertical_scroll_bar is not None
        vertical_scroll_bar.setDisabled(enable)
        self._follow_mode = enable
        if enable:
            self._view.moveCursor(QTextCursor.MoveOperation.End)
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._view.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        else:
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._view.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )

    def _append_text(self, text: str) -> None:
        # Remove trailing newline as appendPlainText adds this
        if text[-1:] == "\n":
            text = text[:-1]
        if self._follow_mode:
            self._view.moveCursor(QTextCursor.MoveOperation.End)
        self._view.appendPlainText(text)
