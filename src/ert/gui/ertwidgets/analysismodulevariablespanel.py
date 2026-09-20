from __future__ import annotations

from functools import partial
from typing import cast

from annotated_types import Ge, Gt, Le
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QWidget,
)

from ert.config import AnalysisModule


class AnalysisModuleVariablesPanel(QWidget):
    def __init__(self, analysis_module: AnalysisModule, ensemble_size: int) -> None:
        QWidget.__init__(self)
        self.analysis_module = analysis_module

        layout = QFormLayout()

        self.blockSignals(True)

        var_name = "enkf_truncation"
        metadata = AnalysisModule.model_fields[var_name]
        self.truncation_spinner = self.createDoubleSpinBox(
            var_name,
            analysis_module.enkf_truncation,
            cast(float, next(v for v in metadata.metadata if isinstance(v, Gt)).gt)
            + 0.001,
            cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
            0.01,
        )
        layout.addRow("Singular value truncation", self.truncation_spinner)

        var_name = "localization_correlation_threshold"
        metadata = AnalysisModule.model_fields[var_name]
        self.local_spinner = self.createDoubleSpinBox(
            var_name,
            analysis_module.correlation_threshold(ensemble_size),
            cast(float, next(v for v in metadata.metadata if isinstance(v, Ge)).ge),
            cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
            0.1,
        )
        self.local_spinner.setObjectName("localization_correlation_threshold")
        layout.addRow("Adaptive localization correlation threshold", self.local_spinner)

        self.setLayout(layout)
        self.blockSignals(False)

    def createDoubleSpinBox(
        self,
        variable_name: str,
        variable_value: float,
        min_value: float,
        max_value: float,
        step_length: float,
    ) -> QDoubleSpinBox:
        spinner = QDoubleSpinBox()
        spinner.setDecimals(6)
        spinner.setFixedWidth(180)
        spinner.setObjectName(variable_name)

        spinner.setRange(
            min_value,
            max_value,
        )

        spinner.setSingleStep(step_length)
        spinner.setValue(variable_value)
        spinner.valueChanged.connect(partial(self.valueChangedSpinner, variable_name))
        return spinner

    def valueChangedSpinner(self, name: str, value: float) -> None:
        setattr(self.analysis_module, name, value)
