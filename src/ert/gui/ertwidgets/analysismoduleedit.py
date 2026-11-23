from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import override

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ert.config import ESSettings, LocalizationType, ParameterType

from .analysismodulevariablespanel import AnalysisModuleVariablesPanel


class AnalysisModuleEdit(QWidget):
    """Update settings (localization per parameter type, truncation and
    correlation threshold), edited inline.

    Changes to the settings are written to ``es_settings`` straight away and
    edited update strategies are emitted through ``update_strategies_changed``.
    Call :meth:`refresh` when the update strategies or whether they can be
    edited might have changed.
    """

    update_strategies_changed = Signal(dict)

    def __init__(
        self,
        es_settings: ESSettings,
        get_update_strategies: Callable[[], Mapping[ParameterType, LocalizationType]],
        ensemble_size: int,
        can_edit_update_strategies: Callable[[], bool] = lambda: True,
    ) -> None:
        QWidget.__init__(self)

        self._es_settings: ESSettings = es_settings
        self._get_update_strategies = get_update_strategies
        self._can_edit_update_strategies = can_edit_update_strategies
        self._ensemble_size: int = ensemble_size
        self._variables_panel: AnalysisModuleVariablesPanel | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.refresh()

    @property
    def variables_panel(self) -> AnalysisModuleVariablesPanel:
        assert self._variables_panel is not None
        return self._variables_panel

    def set_ensemble_size(self, ensemble_size: int) -> None:
        self._ensemble_size = ensemble_size
        self.refresh()

    def _correlation_threshold(self) -> float:
        if self._ensemble_size == 0:
            return 1.0
        return self._es_settings.correlation_threshold(self._ensemble_size)

    def refresh(self) -> None:
        update_strategies = dict(self._get_update_strategies())
        editable = self._can_edit_update_strategies()
        panel = self._variables_panel
        if panel is not None and set(panel.update_strategies) == set(update_strategies):
            panel.set_values(
                update_strategies,
                self._correlation_threshold(),
                self._es_settings.enkf_truncation,
                strategies_editable=editable,
            )
            return

        if panel is not None:
            self._layout.removeWidget(panel)
            panel.deleteLater()
        panel = AnalysisModuleVariablesPanel(
            update_strategies=update_strategies,
            correlation_threshold=self._correlation_threshold(),
            enkf_truncation=self._es_settings.enkf_truncation,
            strategies_editable=editable,
        )
        panel.update_strategies_changed.connect(self.update_strategies_changed)
        panel.correlation_threshold_changed.connect(
            lambda value: setattr(
                self._es_settings, "localization_correlation_threshold", value
            )
        )
        panel.enkf_truncation_changed.connect(
            lambda value: setattr(self._es_settings, "enkf_truncation", value)
        )
        self._layout.addWidget(panel)
        self._variables_panel = panel

    @override
    def showEvent(self, event: QShowEvent | None) -> None:
        # The settings are shared between experiment types, so pick up
        # changes made in another one.
        self.refresh()
        super().showEvent(event)
