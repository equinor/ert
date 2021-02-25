#  Copyright (C) 2016 Equinor ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from qtpy.QtWidgets import QFormLayout, QLabel

from ert_gui.ertwidgets import (
    addHelpToWidget,
    CaseSelector,
    ActiveLabel,
    AnalysisModuleSelector,
)
from ert_gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount, getRunPath
from ert_gui.ertwidgets.models.init_iter_value import IterValueModel
from ert_gui.ertwidgets.models.targetcasemodel import TargetCaseModel
from ert_gui.ertwidgets.models.valuemodel import ValueModel
from ert_gui.ertwidgets.stringbox import StringBox
from ert_shared.ide.keywords.definitions import (
    NumberListStringArgument,
    RangeStringArgument,
    ProperNameFormatArgument,
    IntegerArgument,
)
from ert_gui.simulation import SimulationConfigPanel
from ert_shared.models import MultipleDataAssimilation


class MultipleDataAssimilationPanel(SimulationConfigPanel):
    def __init__(self):
        SimulationConfigPanel.__init__(self, MultipleDataAssimilation)

        layout = QFormLayout()

        case_selector = CaseSelector()
        layout.addRow("Current case:", case_selector)

        run_path_label = QLabel("<b>%s</b>" % getRunPath())
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        number_of_realizations_label = QLabel("<b>%d</b>" % getRealizationCount())
        addHelpToWidget(
            number_of_realizations_label, "config/ensemble/num_realizations"
        )
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._target_case_format_model = TargetCaseModel(format_mode=True)
        self._target_case_format_field = StringBox(
            self._target_case_format_model, "config/simulation/target_case_format"
        )
        self._target_case_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Target case format:", self._target_case_format_field)

        self.weights = MultipleDataAssimilation.default_weights
        self._createInputForWeights(layout)

        self._iter_field = StringBox(
            IterValueModel(),
            "config/simulation/iter_num",
        )
        self._iter_field.setValidator(
            IntegerArgument(from_value=0),
        )
        layout.addRow("Start iteration:", self._iter_field)

        self._analysis_module_selector = AnalysisModuleSelector(
            iterable=False, help_link="config/analysis/analysis_module"
        )
        layout.addRow("Analysis Module:", self._analysis_module_selector)

        self._active_realizations_model = ActiveRealizationsModel()
        self._active_realizations_field = StringBox(
            self._active_realizations_model, "config/simulation/active_realizations"
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(getRealizationCount())
        )
        layout.addRow("Active realizations:", self._active_realizations_field)

        self._target_case_format_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._relative_iteration_weights_box.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )

        self.setLayout(layout)

    def _createInputForWeights(self, layout):
        relative_iteration_weights_model = ValueModel(self.weights)
        self._relative_iteration_weights_box = StringBox(
            relative_iteration_weights_model,
            help_link="config/simulation/iteration_weights",
            continuous_update=True,
        )
        self._relative_iteration_weights_box.setValidator(NumberListStringArgument())
        layout.addRow("Relative Weights:", self._relative_iteration_weights_box)

        relative_iteration_weights_model.valueChanged.connect(self.setWeights)

        normalized_weights_model = ValueModel()
        normalized_weights_widget = ActiveLabel(
            normalized_weights_model, help_link="config/simulation/iteration_weights"
        )
        layout.addRow("Normalized weights:", normalized_weights_widget)

        def updateVisualizationOfNormalizedWeights():
            if self._relative_iteration_weights_box.isValid():
                weights = MultipleDataAssimilation.parseWeights(
                    relative_iteration_weights_model.getValue()
                )
                normalized_weights = MultipleDataAssimilation.normalizeWeights(weights)
                normalized_weights_model.setValue(
                    ", ".join("%.2f" % x for x in normalized_weights)
                )
            else:
                normalized_weights_model.setValue("The weights are invalid!")

        self._relative_iteration_weights_box.getValidationSupport().validationChanged.connect(
            updateVisualizationOfNormalizedWeights
        )

        updateVisualizationOfNormalizedWeights()  # To normalize the default weights

    def isConfigurationValid(self):
        return (
            self._target_case_format_field.isValid()
            and self._active_realizations_field.isValid()
            and self._relative_iteration_weights_box.isValid()
        )

    def getSimulationArguments(self):
        arguments = {
            "active_realizations": self._active_realizations_model.getActiveRealizationsMask(),
            "target_case": self._target_case_format_model.getValue(),
            "analysis_module": self._analysis_module_selector.getSelectedAnalysisModuleName(),
            "weights": self.weights,
            "start_iteration": int(self._iter_field.model.getValue()),
        }
        return arguments

    def setWeights(self, weights):
        str_weights = str(weights)
        print("Weights changed: %s" % str_weights)
        self.weights = str_weights
