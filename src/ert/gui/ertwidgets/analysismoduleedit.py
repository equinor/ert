from __future__ import annotations

from PyQt6.QtCore import QMargins, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ert.config import ESSettings, LocalizationType, ParameterConfig
from ert.gui.icon_utils import load_icon

from .analysismodulevariablespanel import AnalysisModuleVariablesPanel
from .models.parameter_update_draft_model import ParameterUpdateDraftModel


class AnalysisModuleEdit(QWidget):
    settings_changed = Signal()

    def __init__(
        self,
        es_settings: ESSettings,
        parameter_config: list[ParameterConfig] | ParameterUpdateDraftModel,
        ensemble_size: int,
    ) -> None:
        QWidget.__init__(self)

        self._es_settings: ESSettings = es_settings
        self.parameter_state = (
            parameter_config
            if isinstance(parameter_config, ParameterUpdateDraftModel)
            else ParameterUpdateDraftModel(parameter_config)
        )
        self._ensemble_size: int = ensemble_size

        layout = QHBoxLayout()

        variables_popup_button = QPushButton("Edit")
        variables_popup_button.setObjectName("analysis_variables_popup_button")
        variables_popup_button.setIcon(load_icon("edit.svg"))
        variables_popup_button.clicked.connect(self._show_update_settings_dialog)

        layout.addWidget(variables_popup_button, 0, Qt.AlignmentFlag.AlignLeft)
        reset_button = QPushButton("Reset parameter changes")
        reset_button.setObjectName("reset_parameter_changes")
        reset_button.setToolTip(
            "Restore configured and selected-prior parameter localizations. "
            "General settings are unchanged."
        )
        reset_button.clicked.connect(self.parameter_state.reset_all_overrides)
        layout.addWidget(reset_button)

        def refresh_buttons() -> None:
            variables_popup_button.setEnabled(self.parameter_state.has_parameter_source)
            reset_button.setEnabled(self.parameter_state.has_changes)

        self.parameter_state.changed.connect(refresh_buttons)
        refresh_buttons()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addStretch()

        self.setLayout(layout)

    @property
    def parameter_config(self) -> list[ParameterConfig]:
        return self.parameter_state.parameters

    def _show_update_settings_dialog(self) -> None:
        dialog = QDialog(self.parent())  # type: ignore
        dialog.setWindowTitle("Update settings")
        dialog.setModal(True)
        dialog.setWindowFlag(Qt.WindowType.CustomizeWindowHint, True)
        dialog.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        dialog.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout()

        revision = self.parameter_state.revision
        strategies_by_type: dict[str, set[LocalizationType]] = {}
        for parameter in self.parameter_config:
            if parameter.update_strategy is not None:
                strategies_by_type.setdefault(parameter.type.upper(), set()).add(
                    parameter.update_strategy
                )
        update_strategies: dict[str, LocalizationType | None] = {
            name: next(iter(strategies)) if len(strategies) == 1 else None
            for name, strategies in strategies_by_type.items()
        }
        original_strategies = dict(update_strategies)

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
            if revision != self.parameter_state.revision:
                QMessageBox.warning(
                    self,
                    "Parameter configuration changed",
                    "The parameter source changed while editing. "
                    "Reopen Update settings to edit the current parameters.",
                )
                return
            self._es_settings.localization_correlation_threshold = (
                update_settings_dialog.correlation_threshold
            )
            self._es_settings.enkf_truncation = update_settings_dialog.enkf_truncation
            selected_strategies = update_settings_dialog.update_strategies
            self.parameter_state.apply_strategies_by_type(
                {
                    name.lower(): strategy
                    for name, strategy in selected_strategies.items()
                    if strategy is not None
                    and strategy != original_strategies.get(name)
                }
            )
            self.settings_changed.emit()
