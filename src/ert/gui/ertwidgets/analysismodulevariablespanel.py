from functools import partial

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QWidget,
)

from ert.config.analysis_module import AnalysisModule, correlation_threshold
from ert.gui.ertwidgets.models.analysismodulevariablesmodel import (
    AnalysisModuleVariablesModel,
)


class AnalysisModuleVariablesPanel(QWidget):
    def __init__(self, analysis_module: AnalysisModule, ensemble_size: int):
        QWidget.__init__(self)
        self.analysis_module = analysis_module

        layout = QFormLayout()
        variable_names = analysis_module.get_variable_names()

        if len(variable_names) == 0:
            label = QLabel("No variables found to edit")
            boxlayout = QHBoxLayout()
            layout.addRow(label, boxlayout)

        else:
            analysis_module_variables_model = AnalysisModuleVariablesModel
            self.blockSignals(True)

            for variable_name in variable_names:
                variable_type = analysis_module_variables_model.getVariableType(
                    variable_name
                )
                variable_value = analysis_module.get_variable_value(variable_name)

                if variable_name == "LOCALIZATION_CORRELATION_THRESHOLD":
                    variable_value = correlation_threshold(
                        ensemble_size, variable_value
                    )

                label_name = analysis_module_variables_model.getVariableLabelName(
                    variable_name
                )
                if variable_type == bool:
                    layout.addRow(
                        label_name,
                        self.createCheckBox(
                            variable_name, variable_value, variable_type
                        ),
                    )

                elif variable_type == float:
                    layout.addRow(
                        label_name,
                        self.createDoubleSpinBox(
                            variable_name,
                            variable_value,
                            variable_type,
                            analysis_module_variables_model,
                        ),
                    )

                if variable_name == "IES_INVERSION":
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
                        bg.addButton(b, button_id)
                        layout.addRow(b)

                    bg.buttons()[0].setChecked(True)  # check the first option
                    bg.idClicked.connect(self.update_inversion_algorithm)

                if variable_name == "IES_DEC_STEPLENGTH":
                    for s in [
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">'
                        "A good start is max steplength of 0.6, min steplength of 0.3, and decline of 2.5</span>",
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">'
                        "A steplength of 1.0 and one iteration results in ES update</span>",
                    ]:
                        layout.addRow(QLabel(s))

        self.truncation_spinner = self.widget_from_layout(layout, "ENKF_TRUNCATION")
        self.truncation_spinner.setEnabled(False)

        localization_checkbox = self.widget_from_layout(layout, "LOCALIZATION")
        localization_correlation_spinner = self.widget_from_layout(
            layout, "LOCALIZATION_CORRELATION_THRESHOLD"
        )
        localization_correlation_spinner.setEnabled(localization_checkbox.isChecked())
        localization_checkbox.stateChanged.connect(
            lambda localization_is_on: localization_correlation_spinner.setEnabled(True)
            if localization_is_on
            else localization_correlation_spinner.setEnabled(False)
        )

        self.setLayout(layout)
        self.blockSignals(False)

    def update_inversion_algorithm(self, button_id):
        self.truncation_spinner.setEnabled(button_id != 0)  # not for exact inversion
        self.analysis_module.inversion = button_id

    def widget_from_layout(self, layout: QFormLayout, widget_name: str) -> QWidget:
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget.objectName() == widget_name:
                return widget

        return None

    def createCheckBox(self, variable_name, variable_value, variable_type):
        spinner = QCheckBox()
        spinner.setChecked(variable_value)
        spinner.setObjectName(variable_name)
        spinner.clicked.connect(
            partial(self.valueChanged, variable_name, variable_type, spinner)
        )
        return spinner

    def createDoubleSpinBox(
        self,
        variable_name,
        variable_value,
        variable_type,
        analysis_module_variables_model,
    ):
        spinner = QDoubleSpinBox()
        spinner.setDecimals(6)
        spinner.setMinimumWidth(75)
        spinner.setMaximum(
            analysis_module_variables_model.getVariableMaximumValue(variable_name)
        )
        spinner.setMinimum(
            analysis_module_variables_model.getVariableMinimumValue(variable_name)
        )
        spinner.setSingleStep(
            analysis_module_variables_model.getVariableStepValue(variable_name)
        )
        spinner.setValue(variable_value)
        spinner.setObjectName(variable_name)
        spinner.valueChanged.connect(
            partial(self.valueChanged, variable_name, variable_type, spinner)
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
        elif variable_type == str:
            assert isinstance(variable_control, QLineEdit)
            value = variable_control.text()
            value = str(value).strip()
            if len(value) == 0:
                value = None

        if value is not None:
            self.analysis_module.set_var(variable_name, value)
