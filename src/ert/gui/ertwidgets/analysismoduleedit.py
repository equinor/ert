from __future__ import annotations

from collections import defaultdict

from PyQt6.QtCore import QMargins, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ert.config import ESSettings, LocalizationType, ParameterConfig
from ert.gui.icon_utils import load_icon

from .analysismodulevariablespanel import AnalysisModuleVariablesPanel


class AnalysisModuleEdit(QWidget):
    def __init__(
        self,
        es_settings: ESSettings,
        parameter_config: list[ParameterConfig],
        ensemble_size: int,
    ) -> None:
        QWidget.__init__(self)

        self._es_settings: ESSettings = es_settings
        self._parameter_config: list[ParameterConfig] = parameter_config
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

        update_strategies: dict[str, LocalizationType] = defaultdict(
            lambda: LocalizationType.GLOBAL
        )
        for parameter_config in self._parameter_config:
            if parameter_config.update_strategy:
                update_strategies[parameter_config.type.upper()] = (
                    parameter_config.update_strategy
                )

        correlation_threshold = 1.0
        if self._ensemble_size != 0:
            correlation_threshold = self._es_settings.correlation_threshold(
                self._ensemble_size
            )

        update_settings_dialog = AnalysisModuleVariablesPanel(
            update_strategies=update_strategies,
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
            for name, strategy in update_settings_dialog.update_strategies.items():
                for parameter_config in self._parameter_config:
                    if parameter_config.type.upper() == name:
                        parameter_config.update_strategy = strategy
