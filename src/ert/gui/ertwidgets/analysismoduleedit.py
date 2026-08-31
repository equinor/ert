from __future__ import annotations

from PyQt6.QtCore import QMargins, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ert.config import AnalysisConfig
from ert.gui.icon_utils import load_icon

from .analysismodulevariablespanel import AnalysisModuleVariablesPanel


class AnalysisModuleEdit(QWidget):
    on_dialog_closed = Signal(dict)

    def __init__(
        self,
        analysis_config: AnalysisConfig,
        ensemble_size: int,
    ) -> None:
        QWidget.__init__(self)

        self.analysis_config = analysis_config
        self.ensemble_size = ensemble_size

        layout = QHBoxLayout()

        variables_popup_button = QPushButton("Edit")
        variables_popup_button.setObjectName("analysis_variables_popup_button")
        variables_popup_button.setIcon(load_icon("edit.svg"))
        variables_popup_button.clicked.connect(self._show_update_settings_dialog)

        layout.addWidget(variables_popup_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addStretch()

        self.setLayout(layout)

    def _show_update_settings_dialog(self) -> None:
        dialog = QDialog(self.parent())  # type: ignore
        dialog.setWindowTitle("Update settings")
        dialog.setModal(True)
        dialog.setWindowFlag(Qt.WindowType.CustomizeWindowHint, True)
        dialog.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        dialog.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout()
        update_settings_dialog = AnalysisModuleVariablesPanel(
            self.analysis_config,
            self.ensemble_size,
        )

        layout.addWidget(update_settings_dialog, stretch=1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = button_box.button(QDialogButtonBox.StandardButton.Save)
        assert save_button is not None
        save_button.setAutoDefault(False)
        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        assert cancel_button is not None
        cancel_button.setAutoDefault(False)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.setFixedSize(450, 300)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.on_dialog_closed.emit(
                update_settings_dialog.changed_updated_parameter_strategies
            )
