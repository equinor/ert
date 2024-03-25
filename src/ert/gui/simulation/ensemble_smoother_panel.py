from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QFormLayout, QLabel, QLineEdit

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import AnalysisModuleEdit
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.targetensemblemodel import TargetEnsembleModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.run_models import EnsembleSmoother
from ert.validation import ProperNameFormatArgument, RangeStringArgument

from .simulation_config_panel import SimulationConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig


@dataclass
class Arguments:
    mode: str
    target_ensemble: str
    realizations: str
    current_ensemble: str
    experiment_name: str


class EnsembleSmootherPanel(SimulationConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        run_path: str,
        notifier: ErtNotifier,
        ensemble_size: int,
    ) -> None:
        super().__init__(EnsembleSmoother)
        self.notifier = notifier
        layout = QFormLayout()

        self.setObjectName("ensemble_smoother_panel")

        self._name_field = QLineEdit()
        self._name_field.setPlaceholderText("ensemble_smoother")
        self._name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._name_field)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)
        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._ensemble_format_model = TargetEnsembleModel(analysis_config, notifier)
        self._ensemble_format_field = StringBox(
            self._ensemble_format_model,
            self._ensemble_format_model.getDefaultValue(),
            True,
        )
        self._ensemble_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Ensemble format:", self._ensemble_format_field)

        self._analysis_module_edit = AnalysisModuleEdit(
            analysis_config.es_module, ensemble_size
        )
        self._analysis_module_edit.setObjectName("ensemble_smoother_edit")
        layout.addRow("Analysis module:", self._analysis_module_edit)

        active_realizations_model = ActiveRealizationsModel(ensemble_size)
        self._active_realizations_field = StringBox(
            active_realizations_model, "config/simulation/active_realizations"
        )
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        layout.addRow("Active realizations", self._active_realizations_field)

        self.setLayout(layout)

        self._ensemble_format_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )

    def isConfigurationValid(self) -> bool:
        return (
            self._ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
        )

    def getSimulationArguments(self) -> Arguments:
        arguments = Arguments(
            mode="ensemble_smoother",
            current_ensemble=self._ensemble_format_model.getValue() % 0,
            target_ensemble=self._ensemble_format_model.getValue() % 1,
            realizations=self._active_realizations_field.text(),
            experiment_name=(
                self._name_field.text()
                if self._name_field.text()
                else self._name_field.placeholderText()
            ),
        )
        return arguments
