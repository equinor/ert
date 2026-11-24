from __future__ import annotations

from typing import override

from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ert.config import ESSettings

from .analysismodulevariablespanel import AnalysisModuleVariablesPanel


class AnalysisModuleEdit(QWidget):
    """Update settings (truncation and correlation threshold), edited inline.

    Changes to the settings are written to ``es_settings`` straight away.
    """

    def __init__(self, es_settings: ESSettings, ensemble_size: int) -> None:
        QWidget.__init__(self)

        self._es_settings: ESSettings = es_settings
        self._ensemble_size: int = ensemble_size

        self._variables_panel = AnalysisModuleVariablesPanel(
            correlation_threshold=self._correlation_threshold(),
            enkf_truncation=self._es_settings.enkf_truncation,
        )
        self._variables_panel.correlation_threshold_changed.connect(
            lambda value: setattr(
                self._es_settings, "localization_correlation_threshold", value
            )
        )
        self._variables_panel.enkf_truncation_changed.connect(
            lambda value: setattr(self._es_settings, "enkf_truncation", value)
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._variables_panel)

    @property
    def variables_panel(self) -> AnalysisModuleVariablesPanel:
        return self._variables_panel

    def set_ensemble_size(self, ensemble_size: int) -> None:
        self._ensemble_size = ensemble_size
        self.refresh()

    def _correlation_threshold(self) -> float:
        if self._ensemble_size == 0:
            return 1.0
        return self._es_settings.correlation_threshold(self._ensemble_size)

    def refresh(self) -> None:
        self._variables_panel.set_values(
            self._correlation_threshold(), self._es_settings.enkf_truncation
        )

    @override
    def showEvent(self, event: QShowEvent | None) -> None:
        # The settings are shared between experiment types, so pick up
        # changes made in another one.
        self.refresh()
        super().showEvent(event)
