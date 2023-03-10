from functools import partial

from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from ert.config.analysis_module import correlation_threshold
from ert.gui.ertwidgets.models.analysismodulevariablesmodel import (
    AnalysisModuleVariablesModel,
)
from ert.libres_facade import LibresFacade


class AnalysisModuleVariablesPanel(QWidget):
    def __init__(self, analysis_module_name: str, facade: LibresFacade):
        QWidget.__init__(self)
        self.facade = facade
        self._analysis_module_name = analysis_module_name

        layout = QFormLayout()
        variable_names = AnalysisModuleVariablesModel.getVariableNames(
            facade=self.facade, analysis_module_name=self._analysis_module_name
        )

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

                variable_value = analysis_module_variables_model.getVariableValue(
                    self.facade, self._analysis_module_name, variable_name
                )

                if variable_name == "LOCALIZATION_CORRELATION_THRESHOLD":
                    variable_value = correlation_threshold(
                        self.facade.get_ensemble_size(), variable_value
                    )

                label_name = analysis_module_variables_model.getVariableLabelName(
                    variable_name
                )
                if variable_type == bool:
                    spinner = self.createCheckBox(
                        variable_name, variable_value, variable_type
                    )

                elif variable_type == float:
                    spinner = self.createDoubleSpinBox(
                        variable_name,
                        variable_value,
                        variable_type,
                        analysis_module_variables_model,
                    )

                elif variable_type == int:
                    spinner = self.createSpinBox(
                        variable_name,
                        variable_value,
                        variable_type,
                        analysis_module_variables_model,
                    )

                layout.addRow(label_name, spinner)

                if variable_name == "IES_INVERSION":
                    label = QLabel(
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">   '
                        "0: Exact inversion with diagonal R=I</span>"
                    )
                    layout.addRow(label, None)
                    label = QLabel(
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">   '
                        "1: Subspace inversion with exact R  </span>"
                    )
                    layout.addRow(label, None)
                    label = QLabel(
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">   '
                        "2: Subspace inversion using R=EE'   </span>"
                    )
                    layout.addRow(label, None)
                    label = QLabel(
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">   '
                        "3: Subspace inversion using E       </span>"
                    )
                    layout.addRow(label, None)

                if variable_name == "IES_DEC_STEPLENGTH":
                    label = QLabel(
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">   '
                        "A good start is max steplength of 0.6, min steplength of 0.3, "
                        "and decline of 2.5</span>"
                    )
                    layout.addRow(label, None)
                    label = QLabel(
                        "<span style="
                        '"font-size:10pt; font-weight:300;font-style:italic;">   '
                        "A steplength of 1.0 and one iteration "
                        "results in ES update</span>"
                    )
                    layout.addRow(label, None)

        # Truncation of the eigenvalue spectrum is not possible when using exact
        # inversion, hence the spinners for setting the amount of truncation are
        # deactivated for exact inversion.
        inversion_spinner = self.widget_from_layout(layout, "IES_INVERSION")
        truncation_spinner = self.widget_from_layout(layout, "ENKF_TRUNCATION")
        self.update_truncation_spinners(inversion_spinner.value(), truncation_spinner)
        inversion_spinner.valueChanged.connect(
            lambda value: self.update_truncation_spinners(value, truncation_spinner)
        )

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

    def update_truncation_spinners(
        self, value: int, truncation_spinner: QDoubleSpinBox
    ) -> None:
        if value == 0:  # Exact inversion
            truncation_spinner.setEnabled(False)
        else:
            truncation_spinner.setEnabled(True)

    def widget_from_layout(self, layout: QFormLayout, widget_name: str) -> QWidget:
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget.objectName() == widget_name:
                return widget

        return None

    def createSpinBox(
        self,
        variable_name,
        variable_value,
        variable_type,
        analysis_module_variables_model,
    ):
        spinner = QSpinBox()
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
        spinner.setObjectName(variable_name)
        if variable_value is not None:
            spinner.setValue(variable_value)
            spinner.valueChanged.connect(
                partial(self.valueChanged, variable_name, variable_type, spinner)
            )

        return spinner

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
        elif variable_type == int:
            assert isinstance(variable_control, QSpinBox)
            value = variable_control.value()

        if value is not None:
            AnalysisModuleVariablesModel.setVariableValue(
                self.facade, self._analysis_module_name, variable_name, value
            )
