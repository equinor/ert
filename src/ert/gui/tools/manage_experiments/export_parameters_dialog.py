from pathlib import Path
from typing import cast

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

from ert.storage import Ensemble


class ExportParametersDialog(QDialog):
    """Dialog for exporting ensemble parameters to a CSV file."""

    def __init__(self, ensemble: Ensemble, parent: QWidget | None = None) -> None:
        QDialog.__init__(self, parent)
        self._ensemble = ensemble

        self.setWindowFlags(Qt.WindowType.Dialog)
        self.setModal(True)
        self.setWindowTitle("Export ensemble")
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
        """Export the ensemble parameters to the specified file."""
        self._export_text_area.insertPlainText("Exporting...\n")
        output_file: str = self._file_path_edit.text().strip()
        try:
            self._export_button.setEnabled(False)
            parameters_df = self._ensemble.load_all_scalar_keys(transformed=True)
            parameters_df.write_csv(output_file, float_precision=6)
            self._export_text_area.insertPlainText(
                f"Ensemble parameters exported to: {output_file}\n"
            )
        except Exception as e:
            self._export_text_area.insertHtml(
                "<span style='color: red;'>"
                f"Error exporting ensemble parameters: {e!s}"
                "</span><br>"
            )
            self._export_text_area.insertPlainText(
                f"Error exporting ensemble parameters: {e!s}\n"
            )
        finally:
            self._export_button.setEnabled(True)

    @Slot()
    def cancel(self) -> None:
        self.reject()

    @Slot()
    def validate_file(self) -> None:
        """Validation to check if the file path is not empty or invalid."""

        def _set_invalid() -> None:
            palette = self._file_path_edit.palette()
            palette.setColor(QPalette.ColorRole.Text, QColor("red"))
            self._file_path_edit.setPalette(palette)
            self._file_path_edit.setToolTip("Invalid file path")
            self._export_button.setEnabled(False)

        def _set_valid() -> None:
            palette = self._file_path_edit.palette()
            palette.setColor(QPalette.ColorRole.Text, QColor("black"))
            self._file_path_edit.setPalette(palette)
            self._file_path_edit.setToolTip("")
            self._export_button.setEnabled(True)

        file_path = self._file_path_edit.text().strip()
        if not file_path or file_path.endswith(("/", "\\")):
            _set_invalid()
            return

        try:
            path = Path(file_path)
            if path.parent.exists() and path.parent.is_dir():
                _set_valid()
                return
            _set_invalid()
        except (ValueError, OSError):
            _set_invalid()

    @Slot()
    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "", "All Files (*)"
        )
        if file_path:
            self._file_path_edit.setText(file_path)
