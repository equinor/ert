from __future__ import annotations

from collections import defaultdict
from functools import partial
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

from ert.config import AnalysisConfig, AnalysisModule, LocalizationType


class _LocalizationTypeModel(QStandardItemModel):
    def __init__(self, exclude: set[LocalizationType] | None = None) -> None:
        super().__init__()

        type_set = set(LocalizationType)
        if exclude is not None:
            type_set -= exclude

        for localization_type in type_set:
            item = QStandardItem(localization_type.name)
            item.setData(localization_type, Qt.ItemDataRole.UserRole)
            self.appendRow(item)


class AnalysisModuleVariablesPanel(QWidget):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        ensemble_size: int,
    ) -> None:
        QWidget.__init__(self)

        self.analysis_config = analysis_config
        self._changed_updated_parameter_strategies: dict[str, LocalizationType] = (
            defaultdict(lambda: LocalizationType.GLOBAL)
        )

        layout = QFormLayout()

        self.blockSignals(True)

        layout.addRow(QLabel("<b>Parameter type update strategies</b>"))
        layout.addRow(
            QLabel(
                "The following strategies determine how the localization is applied to each parameter for each parameter type."
            )
        )

        gen_kw_combobox = QComboBox(self)
        gen_kw_combobox.setModel(
            _LocalizationTypeModel(exclude={LocalizationType.DISTANCE})
        )
        gen_kw_combobox.setCurrentIndex(
            self._find_correct_index(gen_kw_combobox, "GEN_KW")
        )
        gen_kw_combobox.currentIndexChanged.connect(
            lambda index: self._changed_updated_parameter_strategies.__setitem__(
                "GEN_KW",
                gen_kw_combobox.itemData(index, Qt.ItemDataRole.UserRole),
            )
        )
        layout.addRow("GEN_KW", gen_kw_combobox)

        field_combobox = QComboBox(self)
        field_combobox.setModel(_LocalizationTypeModel())
        field_combobox.setCurrentIndex(
            self._find_correct_index(field_combobox, "FIELD")
        )
        field_combobox.currentIndexChanged.connect(
            lambda index: self._changed_updated_parameter_strategies.__setitem__(
                "FIELD",
                field_combobox.itemData(index, Qt.ItemDataRole.UserRole),
            )
        )
        layout.addRow("FIELD", field_combobox)

        surface_combobox = QComboBox(self)
        surface_combobox.setModel(_LocalizationTypeModel())
        surface_combobox.setCurrentIndex(
            self._find_correct_index(surface_combobox, "SURFACE")
        )
        surface_combobox.currentIndexChanged.connect(
            lambda index: self._changed_updated_parameter_strategies.__setitem__(
                "SURFACE",
                surface_combobox.itemData(index, Qt.ItemDataRole.UserRole),
            )
        )
        layout.addRow("SURFACE", surface_combobox)

        layout.addRow(QLabel("<b>General update settings</b>"))
        layout.addRow(QLabel("Maybe something here"))

        var_name = "enkf_truncation"
        metadata = AnalysisModule.model_fields[var_name]
        self.truncation_spinner = self._create_double_spinbox(
            var_name,
            analysis_config.es_settings.enkf_truncation,
            cast(float, next(v for v in metadata.metadata if isinstance(v, Gt)).gt)
            + 0.001,
            cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
            0.01,
        )
        layout.addRow("Singular value truncation", self.truncation_spinner)

        var_name = "localization_correlation_threshold"
        metadata = AnalysisModule.model_fields[var_name]
        self.local_spinner = self._create_double_spinbox(
            var_name,
            analysis_config.es_settings.correlation_threshold(ensemble_size),
            cast(float, next(v for v in metadata.metadata if isinstance(v, Ge)).ge),
            cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
            0.1,
        )
        self.local_spinner.setObjectName("localization_correlation_threshold")
        layout.addRow("Adaptive localization correlation threshold", self.local_spinner)

        self.setLayout(layout)
        self.blockSignals(False)

    @property
    def changed_updated_parameter_strategies(self) -> dict[str, LocalizationType]:
        return self._changed_updated_parameter_strategies

    def _find_correct_index(self, combobox: QComboBox, type_name: str) -> int:
        if type_name in self.analysis_config.parameter_type_update_strategies:
            localization_type = self.analysis_config.parameter_type_update_strategies[
                type_name
            ]
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
        spinner.valueChanged.connect(
            partial(self._value_changed_spinner, variable_name)
        )
        return spinner

    def _value_changed_spinner(self, name: str, value: float) -> None:
        setattr(self, name, value)
