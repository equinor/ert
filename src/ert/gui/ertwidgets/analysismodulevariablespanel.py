from __future__ import annotations

from functools import partial
from typing import Type, cast

from annotated_types import Ge, Gt, Le
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QWidget,
)
from typing_extensions import get_args

from ert.config.analysis_module import (
    AnalysisModule,
    IESSettings,
    InversionTypeES,
    InversionTypeIES,
)


class AnalysisModuleVariablesPanel(QWidget):
    def __init__(self, analysis_module: AnalysisModule, ensemble_size: int):
        QWidget.__init__(self)
        self.analysis_module = analysis_module

        layout = QFormLayout()
        layout.setVerticalSpacing(5)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setHorizontalSpacing(150)

        self.blockSignals(True)

        layout.addRow(
            QLabel(
                "AnalysisModule: STD_ENKF"
                if type(analysis_module) != IESSettings
                else "AnalysisModule: IES_ENKF"
            )
        )
        layout.addRow(self.create_horizontal_line())

        if isinstance(analysis_module, IESSettings):
            for variable_name in (
                name for name in analysis_module.model_fields if "steplength" in name
            ):
                metadata = analysis_module.model_fields[variable_name]
                layout.addRow(
                    metadata.title,
                    self.createDoubleSpinBox(
                        variable_name,
                        analysis_module.__getattribute__(variable_name),
                        cast(
                            float,
                            next(v for v in metadata.metadata if isinstance(v, Ge)).ge,
                        ),
                        cast(
                            float,
                            next(v for v in metadata.metadata if isinstance(v, Le)).le,
                        ),
                        0.1,
                    ),
                )

            lab = QLabel(analysis_module.__doc__)
            lab.setStyleSheet("font-style: italic; font-size: 10pt; font-weight: 300")
            layout.addRow(lab)

            layout.addRow(self.create_horizontal_line())

        layout.addRow(QLabel("Inversion Algorithm"))
        dropdown = QComboBox(self)
        options = analysis_module.model_fields["inversion"]
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
        metadata = analysis_module.model_fields[var_name]
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

        if not isinstance(analysis_module, IESSettings):
            layout.addRow(self.create_horizontal_line())
            layout.addRow(QLabel("[EXPERIMENTAL]"))

            localization_frame = QFrame()
            localization_frame.setLayout(QHBoxLayout())
            lf_layout = cast(QLayout, localization_frame.layout())
            lf_layout.setContentsMargins(0, 0, 0, 0)

            metadata = analysis_module.model_fields[
                "localization_correlation_threshold"
            ]
            local_checkbox = QCheckBox(metadata.title)
            local_checkbox.setObjectName("localization")
            local_checkbox.clicked.connect(
                partial(
                    self.valueChanged,
                    "localization",
                    bool,
                    local_checkbox,
                )
            )
            var_name = "localization_correlation_threshold"
            metadata = analysis_module.model_fields[var_name]
            self.local_spinner = self.createDoubleSpinBox(
                var_name,
                analysis_module.correlation_threshold(ensemble_size),
                cast(float, next(v for v in metadata.metadata if isinstance(v, Ge)).ge),
                cast(float, next(v for v in metadata.metadata if isinstance(v, Le)).le),
                0.1,
            )
            self.local_spinner.setObjectName("localization_threshold")
            self.local_spinner.setEnabled(local_checkbox.isChecked())

            lf_layout.addWidget(local_checkbox)
            lf_layout.addWidget(self.local_spinner)
            layout.addRow(localization_frame)

            local_checkbox.stateChanged.connect(self.local_spinner.setEnabled)
            local_checkbox.setChecked(analysis_module.localization)

        self.setLayout(layout)
        self.blockSignals(False)

    def update_inversion_algorithm(
        self, text: InversionTypeES | InversionTypeIES
    ) -> None:
        self.truncation_spinner.setEnabled(
            not any(val in text.lower() for val in ["direct", "exact"])
        )
        self.analysis_module.inversion = text

    @staticmethod
    def create_horizontal_line() -> QFrame:
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
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
        spinner.valueChanged.connect(
            partial(self.valueChanged, variable_name, float, spinner)
        )
        return spinner

    def valueChanged(
        self,
        variable_name: str,
        variable_type: Type[bool] | Type[float],
        variable_control: QWidget,
    ) -> None:
        value: bool | float | None = None
        if variable_type == bool:
            assert isinstance(variable_control, QCheckBox)
            value = variable_control.isChecked()
        elif variable_type == float:
            assert isinstance(variable_control, QDoubleSpinBox)
            value = variable_control.value()

        if value is not None:
            self.analysis_module.__setattr__(variable_name, value)  # noqa: PLC2801
