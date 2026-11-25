from __future__ import annotations

from typing import cast, override

from annotated_types import Ge, Gt, Le
from PyQt6.QtCore import QLocale, QSignalBlocker
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QWidget,
)

from ert.config import AnalysisModule


class _CompactDoubleSpinBox(QDoubleSpinBox):
    @override
    def textFromValue(self, v: float) -> str:
        return self.locale().toString(
            v, "f", QLocale.FloatingPointPrecisionOption.FloatingPointShortest
        )


class AnalysisModuleVariablesPanel(QWidget):
    correlation_threshold_changed = Signal(float)
    enkf_truncation_changed = Signal(float)

    def __init__(
        self,
        correlation_threshold: float,
        enkf_truncation: float,
    ) -> None:
        QWidget.__init__(self)

        self._correlation_threshold = correlation_threshold
        self._enkf_truncation = enkf_truncation

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.blockSignals(True)

        var_name = "enkf_truncation"
        metadata = AnalysisModule.model_fields[var_name]
        self.truncation_spinner = self._create_double_spinbox(
            var_name,
            self._enkf_truncation,
            cast(float, next(v for v in metadata.metadata if isinstance(v, Gt)).gt)
            + 0.001,
            cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
            0.01,
        )
        self.truncation_spinner.valueChanged.connect(self._on_truncation_changed)

        layout.addRow("Singular value truncation", self.truncation_spinner)

        var_name = "localization_correlation_threshold"
        metadata = AnalysisModule.model_fields[var_name]
        self.threshold_spinner = self._create_double_spinbox(
            var_name,
            self._correlation_threshold,
            cast(float, next(v for v in metadata.metadata if isinstance(v, Ge)).ge),
            cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
            0.1,
        )
        self.threshold_spinner.setObjectName("localization_correlation_threshold")
        self.threshold_spinner.valueChanged.connect(self._on_threshold_changed)

        layout.addRow(
            "Adaptive localization correlation threshold", self.threshold_spinner
        )

        self.setLayout(layout)
        self.blockSignals(False)

    def _on_truncation_changed(self, value: float) -> None:
        self._enkf_truncation = value
        self.enkf_truncation_changed.emit(value)

    def _on_threshold_changed(self, value: float) -> None:
        self._correlation_threshold = value
        self.correlation_threshold_changed.emit(value)

    def set_values(
        self,
        correlation_threshold: float,
        enkf_truncation: float,
    ) -> None:
        """Show new values without emitting change signals."""
        self._correlation_threshold = correlation_threshold
        with QSignalBlocker(self.threshold_spinner):
            self.threshold_spinner.setValue(correlation_threshold)
        self._enkf_truncation = enkf_truncation
        with QSignalBlocker(self.truncation_spinner):
            self.truncation_spinner.setValue(enkf_truncation)

    @property
    def correlation_threshold(self) -> float:
        return self._correlation_threshold

    @property
    def enkf_truncation(self) -> float:
        return self._enkf_truncation

    def _create_double_spinbox(
        self,
        variable_name: str,
        variable_value: float,
        min_value: float,
        max_value: float,
        step_length: float,
    ) -> QDoubleSpinBox:
        spinner = _CompactDoubleSpinBox()
        spinner.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinner.setDecimals(6)
        spinner.setFixedWidth(180)
        spinner.setObjectName(variable_name)
        spinner.setRange(
            min_value,
            max_value,
        )
        spinner.setSingleStep(step_length)
        spinner.setValue(variable_value)
        return spinner
