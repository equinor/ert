from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import Slot
from qtpy.QtWidgets import QFormLayout, QLabel, QSpinBox

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import AnalysisModuleEdit, StringBox, TextModel
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.targetensemblemodel import TargetEnsembleModel
from ert.mode_definitions import ITERATIVE_ENSEMBLE_SMOOTHER_MODE
from ert.run_models import IteratedEnsembleSmoother
from ert.validation import ProperNameFormatArgument, RangeStringArgument
from ert.validation.range_string_argument import NotInStorage

from .experiment_config_panel import ExperimentConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig


@dataclass
class Arguments:
    mode: str
    target_ensemble: str
    realizations: str
    num_iterations: int
    experiment_name: str


class IteratedEnsembleSmootherPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        run_path: str,
        notifier: ErtNotifier,
        ensemble_size: int,
    ):
        self.notifier = notifier
        ExperimentConfigPanel.__init__(self, IteratedEnsembleSmoother)
        self.analysis_config = analysis_config
        layout = QFormLayout()

        self._experiment_name_field = StringBox(
            TextModel(""),
            placeholder_text=self.notifier.storage.get_unique_experiment_name(
                ITERATIVE_ENSEMBLE_SMOOTHER_MODE
            ),
        )
        self._experiment_name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._experiment_name_field)
        self._experiment_name_field.setValidator(
            NotInStorage(self.notifier.storage, "experiments")
        )

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
            self._iterated_target_ensemble_format_model,  # type: ignore
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
            self._active_realizations_model,  # type: ignore
            "config/simulation/active_realizations",
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

        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @Slot(ExperimentConfigPanel)
    def experimentTypeChanged(self, w: ExperimentConfigPanel) -> None:
        if isinstance(w, IteratedEnsembleSmootherPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(
                ITERATIVE_ENSEMBLE_SMOOTHER_MODE
            )
        )

    def setNumberIterations(self, iteration_count: int) -> None:
        if iteration_count != self.analysis_config.num_iterations:
            self.analysis_config.set_num_iterations(iteration_count)
            self.notifier.emitErtChange()

    def isConfigurationValid(self) -> bool:
        return (
            self._iterated_target_ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
        )

    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
            target_ensemble=self._iterated_target_ensemble_format_model.getValue(),  # type: ignore
            realizations=self._active_realizations_field.text(),
            num_iterations=self._num_iterations_spinner.value(),
            experiment_name=self._experiment_name_field.get_text,
        )
