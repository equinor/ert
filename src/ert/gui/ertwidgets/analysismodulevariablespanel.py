from functools import partial

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QWidget,
)

from ert.config import AnalysisMode
from ert.config.analysis_module import AnalysisModule
from ert.gui.ertwidgets.models.analysismodulevariablesmodel import (
    AnalysisModuleVariablesModel,
)


class AnalysisModuleVariablesPanel(QWidget):
    def __init__(self, analysis_module: AnalysisModule, ensemble_size: int):
        QWidget.__init__(self)
        self.analysis_module = analysis_module

        layout = QFormLayout()
        layout.setVerticalSpacing(5)
        layout.setLabelAlignment(Qt.AlignLeft)
        layout.setHorizontalSpacing(150)
        variable_names = analysis_module.get_variable_names()

        analysis_module_variables_model = AnalysisModuleVariablesModel
        self.blockSignals(True)

        layout.addRow(QLabel(str(analysis_module)))
        layout.addRow(self.create_horizontal_line())

        if analysis_module.mode == AnalysisMode.ITERATED_ENSEMBLE_SMOOTHER:
            for variable_name in (
                name for name in variable_names if "STEPLENGTH" in name
            ):
                layout.addRow(
                    analysis_module_variables_model.getVariableLabelName(variable_name),
                    self.createDoubleSpinBox(
                        variable_name,
                        analysis_module.get_variable_value(variable_name),
                        analysis_module_variables_model,
                    ),
                )

            for s in [
                "A good start is max steplength of 0.6, min steplength of 0.3, and decline of 2.5",
                "A steplength of 1.0 and one iteration results in ES update",
            ]:
                lab = QLabel(s)
                lab.setStyleSheet(
                    "font-style: italic; font-size: 10pt; font-weight: 300"
                )
                layout.addRow(lab)

            layout.addRow(self.create_horizontal_line())

        layout.addRow(QLabel("Inversion Algorithm"))
        bg = QButtonGroup(self)
        for button_id, s in enumerate(
            [
                "Exact inversion with diagonal R=I",
                "Subspace inversion with exact R",
                "Subspace inversion using R=EE'",
                "Subspace inversion using E",
            ],
            start=0,
        ):
            b = QRadioButton(s, self)
            b.setObjectName("IES_INVERSION_" + str(button_id))
            bg.addButton(b, button_id)
            layout.addRow(b)

        self.truncation_spinner = self.createDoubleSpinBox(
            "ENKF_TRUNCATION",
            analysis_module.get_truncation(),
            analysis_module_variables_model,
        )
        self.truncation_spinner.setEnabled(False)
        layout.addRow("Singular value truncation", self.truncation_spinner)

        bg.idClicked.connect(self.update_inversion_algorithm)
        bg.buttons()[analysis_module.inversion].click()  # update the current value

        layout.addRow(self.create_horizontal_line())
        layout.addRow(QLabel("[EXPERIMENTAL]"))

        localization_frame = QFrame()
        localization_frame.setLayout(QHBoxLayout())
        localization_frame.layout().setContentsMargins(0, 0, 0, 0)

        local_checkbox = QCheckBox("Adaptive localization correlation threshold")
        local_checkbox.clicked.connect(
            partial(
                self.valueChanged,
                "LOCALIZATION",
                bool,
                local_checkbox,
            )
        )

        self.local_spinner = self.createDoubleSpinBox(
            "LOCALIZATION_CORRELATION_THRESHOLD",
            analysis_module.localization_correlation_threshold(ensemble_size),
            analysis_module_variables_model,
        )
        self.local_spinner.setEnabled(local_checkbox.isChecked())

        localization_frame.layout().addWidget(local_checkbox)
        localization_frame.layout().addWidget(self.local_spinner)
        layout.addRow(localization_frame)

        local_checkbox.stateChanged.connect(
            lambda localization_is_on: self.local_spinner.setEnabled(localization_is_on)
        )
        local_checkbox.setChecked(analysis_module.localization())

        self.setLayout(layout)
        self.blockSignals(False)

    def update_inversion_algorithm(self, button_id):
        self.truncation_spinner.setEnabled(button_id != 0)  # not for exact inversion
        self.analysis_module.inversion = button_id

    def create_horizontal_line(self) -> QFrame:
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        hline.setFixedHeight(20)
        return hline

    def createDoubleSpinBox(
        self,
        variable_name,
        variable_value,
        analysis_module_variables_model,
    ):
        spinner = QDoubleSpinBox()
        spinner.setDecimals(6)
        spinner.setFixedWidth(100)

        spinner.setRange(
            analysis_module_variables_model.getVariableMinimumValue(variable_name),
            analysis_module_variables_model.getVariableMaximumValue(variable_name),
        )

        spinner.setSingleStep(
            analysis_module_variables_model.getVariableStepValue(variable_name)
        )
        spinner.setValue(variable_value)
        spinner.valueChanged.connect(
            partial(self.valueChanged, variable_name, float, spinner)
        )
        return spinner

    def valueChanged(self, variable_name, variable_type, variable_control):
        value = None
        if variable_type == bool:
            assert isinstance(variable_control, QCheckBox)
            value = variable_control.isChecked()
        elif variable_type == float:
            assert isinstance(variable_control, QDoubleSpinBox)
            value = variable_control.value()

        if value is not None:
            self.analysis_module.set_var(variable_name, value)
