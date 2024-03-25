from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QFormLayout, QLabel, QLineEdit, QSpinBox

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import AnalysisModuleEdit, EnsembleSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.targetensemblemodel import TargetEnsembleModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.run_models import IteratedEnsembleSmoother
from ert.validation import ProperNameFormatArgument, RangeStringArgument

from .simulation_config_panel import SimulationConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig


@dataclass
class Arguments:
    mode: str
    current_ensemble: str
    target_ensemble: str
    realizations: str
    num_iterations: int
    experiment_name: str


class IteratedEnsembleSmootherPanel(SimulationConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        run_path: str,
        notifier: ErtNotifier,
        ensemble_size: int,
    ):
        self.notifier = notifier
        SimulationConfigPanel.__init__(self, IteratedEnsembleSmoother)
        self.analysis_config = analysis_config
        layout = QFormLayout()

        self._name_field = QLineEdit()
        self._name_field.setPlaceholderText("iterated_ensemble_smoother")
        self._name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._name_field)

        ensemble_selector = EnsembleSelector(notifier)
        layout.addRow("Current ensemble:", ensemble_selector)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        # The num_iterations_spinner does not track any external changes (will
        # that ever happen?)
        self._num_iterations_spinner = QSpinBox()
        self._num_iterations_spinner.setMinimum(1)
        self._num_iterations_spinner.setMaximum(100)
        self._num_iterations_spinner.setValue(analysis_config.num_iterations)
        self._num_iterations_spinner.valueChanged[int].connect(self.setNumberIterations)

        layout.addRow("Number of iterations:", self._num_iterations_spinner)

        self._iterated_target_ensemble_format_model = TargetEnsembleModel(
            analysis_config, notifier
        )
        self._iterated_target_ensemble_format_field = StringBox(
            self._iterated_target_ensemble_format_model,
            "config/simulation/iterated_target_ensemble_format",
        )
        self._iterated_target_ensemble_format_field.setValidator(
            ProperNameFormatArgument()
        )
        layout.addRow(
            "Target ensemble format:", self._iterated_target_ensemble_format_field
        )

        self._analysis_module_edit = AnalysisModuleEdit(
            analysis_config.ies_module, ensemble_size
        )
        layout.addRow("Analysis module:", self._analysis_module_edit)

        self._active_realizations_model = ActiveRealizationsModel(ensemble_size)
        self._active_realizations_field = StringBox(
            self._active_realizations_model, "config/simulation/active_realizations"
        )
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        layout.addRow("Active realizations", self._active_realizations_field)

        self._iterated_target_ensemble_format_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )

        self.setLayout(layout)

    def setNumberIterations(self, iteration_count):
        if iteration_count != self.analysis_config.num_iterations:
            self.analysis_config.set_num_iterations(iteration_count)
            self.notifier.emitErtChange()

    def isConfigurationValid(self):
        return (
            self._iterated_target_ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
        )

    def getSimulationArguments(self):
        return Arguments(
            mode="iterative_ensemble_smoother",
            current_ensemble=self.notifier.current_ensemble_name,
            target_ensemble=self._iterated_target_ensemble_format_model.getValue(),
            realizations=self._active_realizations_field.text(),
            num_iterations=self._num_iterations_spinner.value(),
            experiment_name=(
                self._name_field.text()
                if self._name_field.text()
                else self._name_field.placeholderText()
            ),
        )
