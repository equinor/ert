from __future__ import annotations

from functools import partial
from typing import cast, get_args

from annotated_types import Ge, Gt, Le
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QVBoxLayout,
    QWidget,
)

from ert.config import (
    AnalysisModule,
    InversionTypeES,
)


class AnalysisModuleVariablesPanel(QWidget):
    def __init__(self, analysis_module: AnalysisModule, ensemble_size: int) -> None:
        QWidget.__init__(self)
        self.analysis_module = analysis_module

        layout = QFormLayout()
        layout.setVerticalSpacing(5)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setHorizontalSpacing(150)

        self.blockSignals(True)

        layout.addRow(QLabel("Inversion Algorithm"))
        dropdown = QComboBox(self)
        options = AnalysisModule.model_fields["inversion"]
        layout.addRow(QLabel(options.description))
        default_index = 0
        for i, option in enumerate(get_args(options.annotation)):
            dropdown.addItem(option.upper())
            if analysis_module.inversion == option:
                default_index = i
        dropdown.setCurrentIndex(default_index)
        dropdown.currentTextChanged.connect(self.update_inversion_algorithm)
        layout.addRow(dropdown)
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
        self.truncation_spinner.setEnabled(False)
        layout.addRow("Singular value truncation", self.truncation_spinner)

        layout.addRow(self.create_horizontal_line())
        layout.addRow(QLabel("Localization"))

        loc_options = AnalysisModule.model_fields["localization"]
        layout.addRow(QLabel(loc_options.description))

        localization_frame = QFrame()
        localization_frame.setLayout(QVBoxLayout())
        main_loc_layout = cast(QLayout, localization_frame.layout())
        main_loc_layout.setContentsMargins(0, 0, 0, 0)

        row1_frame = QFrame()
        row1_layout = QHBoxLayout()
        row1_frame.setLayout(row1_layout)
        row1_layout.setContentsMargins(0, 0, 0, 0)

        localization_metadata = AnalysisModule.model_fields["localization"]
        localization_checkbox = QCheckBox(localization_metadata.title)
        localization_checkbox.setObjectName("localization")
        localization_checkbox.clicked.connect(
            partial(
                self.valueChangedCheckBox,
                "localization",
                localization_checkbox,
            )
        )
        localization_checkbox.setChecked(analysis_module.localization)

        row1_layout.addWidget(localization_checkbox)
        row1_layout.addStretch(1)
        main_loc_layout.addWidget(row1_frame)

        row2_frame = QFrame()
        row2_layout = QHBoxLayout()
        row2_frame.setLayout(row2_layout)
        row2_layout.setContentsMargins(0, 0, 0, 0)

        var_name = "localization_correlation_threshold"
        metadata = AnalysisModule.model_fields[var_name]
        custom_local_threshold_checkbox = QCheckBox(metadata.title)
        custom_local_threshold_checkbox.setObjectName("custom_localization_threshold")
        custom_local_threshold_checkbox.clicked.connect(
            partial(
                self.valueChangedCheckBox,
                "custom_localization_threshold",
                custom_local_threshold_checkbox,
            )
        )

        self.local_spinner = self.createDoubleSpinBox(
            var_name,
            analysis_module.correlation_threshold(ensemble_size),
            cast(float, next(v for v in metadata.metadata if isinstance(v, Ge)).ge),
            cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
            0.1,
        )
        self.local_spinner.setObjectName("localization_threshold")
        self.local_spinner.setEnabled(custom_local_threshold_checkbox.isChecked())

        row2_layout.addWidget(custom_local_threshold_checkbox)
        row2_layout.addWidget(self.local_spinner)
        row2_layout.addStretch(1)
        main_loc_layout.addWidget(row2_frame)

        localization_checkbox.stateChanged.connect(
            custom_local_threshold_checkbox.setEnabled
        )
        custom_local_threshold_checkbox.stateChanged.connect(
            self.local_spinner.setEnabled
        )
        custom_local_threshold_checkbox.setChecked(
            analysis_module.custom_localization_threshold
        )
        custom_local_threshold_checkbox.setEnabled(analysis_module.localization)

        layout.addRow(localization_frame)

        self.setLayout(layout)
        self.blockSignals(False)

    def update_inversion_algorithm(self, text: InversionTypeES) -> None:
        self.truncation_spinner.setEnabled(
            not any(val in text.upper() for val in ["DIRECT", "EXACT"])
        )
        self.analysis_module.inversion = text

    @staticmethod
    def create_horizontal_line() -> QFrame:
        hline = QFrame()
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        hline.setFixedHeight(20)
        return hline

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

    def valueChangedCheckBox(self, name: str, control: QCheckBox) -> None:
        setattr(self.analysis_module, name, control.isChecked())
