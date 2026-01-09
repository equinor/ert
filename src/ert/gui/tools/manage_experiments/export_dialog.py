import contextlib
import logging
from pathlib import Path
from typing import cast

import polars as pl
from PyQt6.QtCore import (
    Qt,
)
from PyQt6.QtCore import (
    pyqtSlot as Slot,
)
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ExportDialog(QDialog):
    """Base dialog for exporting ensemble-related data to files."""

    def __init__(
        self,
        export_data: pl.DataFrame,
        window_title: str = "Export data",
        parent: QWidget | None = None,
    ) -> None:
        QDialog.__init__(self, parent)
        self._export_data = export_data
        self.setWindowTitle(window_title)

        self.setWindowFlags(Qt.WindowType.Dialog)
        self.setModal(True)
        self.setMinimumWidth(450)
        self.setMinimumHeight(200)

        main_layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self._file_path_edit = QLineEdit(self)
        self._file_path_edit.setPlaceholderText("Select output file...")
        self._file_path_edit.textChanged.connect(self.validate_file)
        browse_button = QPushButton("Browse...", self)
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self._file_path_edit)
        file_layout.addWidget(browse_button)
        main_layout.addLayout(file_layout)

        self._export_text_area = QTextEdit(self)
        self._export_text_area.setReadOnly(True)
        self._export_text_area.setFixedHeight(100)
        main_layout.addWidget(self._export_text_area)

        button_box = QDialogButtonBox(self)
        button_box.setStandardButtons(QDialogButtonBox.StandardButton.Cancel)
        button_box.rejected.connect(self.cancel)

        self._export_button = cast(
            QPushButton,
            button_box.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole),
        )
        self._export_button.clicked.connect(self.export)
        self._export_button.setEnabled(False)  # Initially disabled
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    @Slot()
    def export(self) -> None:
        self._export_text_area.insertPlainText("Exporting...\n")
        try:
            output_file: str = self._file_path_edit.text().strip()
            self._export_button.setEnabled(False)
            self._export_data.write_csv(output_file, float_precision=6)
            self._export_text_area.insertPlainText(f"Data exported to: {output_file}\n")
        except Exception as e:
            self._export_text_area.insertHtml(
                f"<span style='color: red;'>Could not export data: {e!s}</span><br>"
            )
        finally:
            logger.info(f"Export dialog used: '{self.windowTitle()}'")
            self._export_button.setEnabled(True)

    @Slot()
    def cancel(self) -> None:
        self.reject()

    @Slot()
    def validate_file(self) -> None:
        """Validation to check if the file path is not empty or invalid."""

        def _set_invalid(tooltip_text: str = "Invalid file path") -> None:
            palette = self._file_path_edit.palette()
            palette.setColor(QPalette.ColorRole.Text, QColor("red"))
            self._file_path_edit.setPalette(palette)
            self._file_path_edit.setToolTip(tooltip_text)
            self._export_button.setToolTip(tooltip_text)
            self._export_button.setEnabled(False)

        def _set_valid() -> None:
            palette = self._file_path_edit.palette()
            palette.setColor(QPalette.ColorRole.Text, QColor("black"))
            self._file_path_edit.setPalette(palette)
            self._file_path_edit.setToolTip("")
            self._export_button.setToolTip("")
            self._export_button.setEnabled(True)

        path = Path(self._file_path_edit.text().strip())
        if str(path) in {"", "."}:
            _set_invalid(tooltip_text="No filename provided")
            return

        if path.is_dir():
            _set_invalid(tooltip_text=f"'{path!s}' is an existing directory.")
            return

        with contextlib.suppress(Exception):
            if path.parent.is_dir():
                _set_valid()
                return

        _set_invalid()

    @Slot()
    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "", "All Files (*)"
        )
        if file_path:
            self._file_path_edit.setText(file_path)
