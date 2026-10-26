from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from annotated_types import Ge, Gt, Le
from PyQt6.QtCore import Qt
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

        layout = QFormLayout()
        self.blockSignals(True)

        layout.addRow(
            QLabel("<b>Select the localization method for each parameter type</b>")
        )

        gen_kw_combobox = QComboBox(self)
        gen_kw_combobox.setModel(
            _LocalizationTypeModel(ParameterType.GEN_KW.supported_localization_types)
        )
        gen_kw_combobox.setCurrentIndex(
            self._find_correct_index(gen_kw_combobox, ParameterType.GEN_KW)
        )
        gen_kw_combobox.currentIndexChanged.connect(
            lambda index: self._update_strategies.__setitem__(
                ParameterType.GEN_KW,
                gen_kw_combobox.itemData(index, Qt.ItemDataRole.UserRole),
            )
        )
        layout.addRow(ParameterType.GEN_KW.name, gen_kw_combobox)

        field_combobox = QComboBox(self)
        field_combobox.setModel(
            _LocalizationTypeModel(ParameterType.FIELD.supported_localization_types)
        )
        field_combobox.setCurrentIndex(
            self._find_correct_index(field_combobox, ParameterType.FIELD)
        )
        field_combobox.currentIndexChanged.connect(
            lambda index: self._update_strategies.__setitem__(
                ParameterType.FIELD,
                field_combobox.itemData(index, Qt.ItemDataRole.UserRole),
            )
        )
        layout.addRow(ParameterType.FIELD.name, field_combobox)

        surface_combobox = QComboBox(self)
        surface_combobox.setModel(
            _LocalizationTypeModel(ParameterType.SURFACE.supported_localization_types)
        )
        surface_combobox.setCurrentIndex(
            self._find_correct_index(surface_combobox, ParameterType.SURFACE)
        )
        surface_combobox.currentIndexChanged.connect(
            lambda index: self._update_strategies.__setitem__(
                ParameterType.SURFACE,
                surface_combobox.itemData(index, Qt.ItemDataRole.UserRole),
            )
        )
        layout.addRow(ParameterType.SURFACE.name, surface_combobox)

        for combobox in (gen_kw_combobox, field_combobox, surface_combobox):
            combobox.setEnabled(strategies_editable)
            if not strategies_editable:
                combobox.setToolTip(
                    "Update strategy is defined by the selected ensemble"
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
        self.truncation_spinner.valueChanged.connect(
            lambda value: setattr(self, "_enkf_truncation", value)
        )

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
        self.threshold_spinner.valueChanged.connect(
            lambda value: setattr(self, "_correlation_threshold", value)
        )

        layout.addRow(
            "Adaptive localization correlation threshold", self.threshold_spinner
        )

        self.setLayout(layout)
        self.blockSignals(False)

    @property
    def update_strategies(self) -> dict[ParameterType, LocalizationType]:
        return self._update_strategies

    @property
    def correlation_threshold(self) -> float:
        return self._correlation_threshold

    @property
    def enkf_truncation(self) -> float:
        return self._enkf_truncation

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
