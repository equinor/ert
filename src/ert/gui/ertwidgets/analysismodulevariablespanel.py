from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from annotated_types import Ge, Gt, Le
from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QWidget,
)

from ert.config import AnalysisModule, LocalizationType, ParameterType


class _LocalizationTypeModel(QStandardItemModel):
    def __init__(self, localization_types: Iterable[LocalizationType]) -> None:
        super().__init__()

        for localization_type in sorted(localization_types, key=lambda lt: lt.name):
            item = QStandardItem(localization_type.name)
            item.setData(localization_type, Qt.ItemDataRole.UserRole)
            self.appendRow(item)


class AnalysisModuleVariablesPanel(QWidget):
    update_strategies_changed = Signal(dict)
    correlation_threshold_changed = Signal(float)
    enkf_truncation_changed = Signal(float)

    def __init__(
        self,
        update_strategies: dict[ParameterType, LocalizationType],
        correlation_threshold: float,
        enkf_truncation: float,
        *,
        strategies_editable: bool = True,
    ) -> None:
        QWidget.__init__(self)

        self._update_strategies = update_strategies
        self._correlation_threshold = correlation_threshold
        self._enkf_truncation = enkf_truncation
        self._strategy_comboboxes: dict[ParameterType, QComboBox] = {}

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.blockSignals(True)

        layout.addRow(
            QLabel("<b>Select the localization method for each parameter type</b>")
        )

        for parameter_type in ParameterType:
            if parameter_type not in update_strategies:
                continue
            layout.addRow(
                parameter_type.name,
                self._create_strategy_combobox(parameter_type, strategies_editable),
            )

        layout.addRow(QLabel("<b>General settings</b>"))

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

    def _on_strategy_changed(self, parameter_type: ParameterType, index: int) -> None:
        combobox = self._strategy_comboboxes[parameter_type]
        self._update_strategies[parameter_type] = combobox.itemData(
            index, Qt.ItemDataRole.UserRole
        )
        self.update_strategies_changed.emit(dict(self._update_strategies))

    def set_values(
        self,
        update_strategies: dict[ParameterType, LocalizationType],
        correlation_threshold: float,
        enkf_truncation: float,
        *,
        strategies_editable: bool,
    ) -> None:
        """Show new values without emitting change signals.

        The parameter types must be the same as the panel was created with.
        """
        self._update_strategies = update_strategies
        for parameter_type, combobox in self._strategy_comboboxes.items():
            with QSignalBlocker(combobox):
                combobox.setCurrentIndex(
                    self._find_correct_index(combobox, parameter_type)
                )
            self._set_strategy_editable(combobox, strategies_editable)
        self._correlation_threshold = correlation_threshold
        with QSignalBlocker(self.threshold_spinner):
            self.threshold_spinner.setValue(correlation_threshold)
        self._enkf_truncation = enkf_truncation
        with QSignalBlocker(self.truncation_spinner):
            self.truncation_spinner.setValue(enkf_truncation)

    @staticmethod
    def _set_strategy_editable(combobox: QComboBox, editable: bool) -> None:
        combobox.setEnabled(editable)
        combobox.setToolTip(
            "" if editable else "Update strategy is defined by the selected ensemble"
        )

    @property
    def update_strategies(self) -> dict[ParameterType, LocalizationType]:
        return self._update_strategies

    @property
    def correlation_threshold(self) -> float:
        return self._correlation_threshold

    @property
    def enkf_truncation(self) -> float:
        return self._enkf_truncation

    def _create_strategy_combobox(
        self, parameter_type: ParameterType, editable: bool
    ) -> QComboBox:
        combobox = QComboBox(self)
        combobox.setObjectName(parameter_type)
        combobox.setModel(
            _LocalizationTypeModel(parameter_type.supported_localization_types)
        )
        combobox.setCurrentIndex(self._find_correct_index(combobox, parameter_type))
        combobox.currentIndexChanged.connect(
            lambda index: self._on_strategy_changed(parameter_type, index)
        )
        self._set_strategy_editable(combobox, editable)
        self._strategy_comboboxes[parameter_type] = combobox
        return combobox

    def _find_correct_index(
        self, combobox: QComboBox, parameter_type: ParameterType
    ) -> int:
        if parameter_type in self._update_strategies:
            localization_type = self._update_strategies[parameter_type]
            if (
                index := combobox.findData(localization_type, Qt.ItemDataRole.UserRole)
            ) != -1:
                return index
        return combobox.findData(LocalizationType.GLOBAL, Qt.ItemDataRole.UserRole)

    def _create_double_spinbox(
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
        return spinner
