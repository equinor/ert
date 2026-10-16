from __future__ import annotations

from collections.abc import Callable, Mapping

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

from ert.config import ESSettings, LocalizationType
from ert.gui.icon_utils import load_icon

from .analysismodulevariablespanel import AnalysisModuleVariablesPanel


class AnalysisModuleEdit(QWidget):
    update_strategies_changed = Signal(dict)

    def __init__(
        self,
        es_settings: ESSettings,
        get_update_strategies: Callable[[], Mapping[str, LocalizationType]],
        ensemble_size: int,
    ) -> None:
        QWidget.__init__(self)

        self._es_settings: ESSettings = es_settings
        self._get_update_strategies = get_update_strategies
        self._ensemble_size: int = ensemble_size

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

        correlation_threshold = 1.0
        if self._ensemble_size != 0:
            correlation_threshold = self._es_settings.correlation_threshold(
                self._ensemble_size
            )

        update_settings_dialog = AnalysisModuleVariablesPanel(
            update_strategies=dict(self._get_update_strategies()),
            correlation_threshold=correlation_threshold,
            enkf_truncation=self._es_settings.enkf_truncation,
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
            self._es_settings.localization_correlation_threshold = (
                update_settings_dialog.correlation_threshold
            )
            self._es_settings.enkf_truncation = update_settings_dialog.enkf_truncation
            self.update_strategies_changed.emit(
                dict(update_settings_dialog.update_strategies)
            )
