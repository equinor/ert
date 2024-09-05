from dataclasses import dataclass

import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import QFormLayout, QLabel

from ert.config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    AnalysisModuleEdit,
    CopyableLabel,
    EnsembleSelector,
    StringBox,
    TargetEnsembleModel,
)
from ert.gui.simulation.experiment_config_panel import ExperimentConfigPanel
from ert.mode_definitions import MANUAL_UPDATE_MODE
from ert.run_models.manual_update import ManualUpdate
from ert.validation import ProperNameFormatArgument, RangeStringArgument


@dataclass
class Arguments:
    mode: str
    realizations: str
    ensemble_id: str
    target_ensemble: str


class ManualUpdatePanel(ExperimentConfigPanel):
    def __init__(
        self,
        ensemble_size: int,
        run_path: str,
        notifier: ErtNotifier,
        analysis_config: AnalysisConfig,
    ):
        self.notifier = notifier
        super().__init__(ManualUpdate)
        self.setObjectName("Manual_update_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(ManualUpdate.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addRow(lab)
        self._ensemble_selector = EnsembleSelector(notifier)
        layout.addRow("Ensemble:", self._ensemble_selector)
        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._ensemble_format_model = TargetEnsembleModel(analysis_config, notifier)
        self._ensemble_format_field = StringBox(
            self._ensemble_format_model,  # type: ignore
            self._ensemble_format_model.getDefaultValue(),  # type: ignore
            continuous_update=True,
        )
        self._ensemble_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Ensemble format:", self._ensemble_format_field)

        self._analysis_module_edit = AnalysisModuleEdit(
            analysis_config.es_module, ensemble_size
        )
        self._analysis_module_edit.setObjectName("ensemble_smoother_edit")
        layout.addRow("Analysis module:", self._analysis_module_edit)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(ensemble_size, show_default=False),  # type: ignore
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(ensemble_size),
        )
        self._realizations_from_fs()
        layout.addRow("Active realizations", self._active_realizations_field)

        self.setLayout(layout)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._ensemble_selector.ensemble_populated.connect(self._realizations_from_fs)
        self._ensemble_selector.ensemble_populated.connect(
            self.simulationConfigurationChanged
        )
        self._ensemble_selector.currentIndexChanged.connect(self._realizations_from_fs)

    def isConfigurationValid(self) -> bool:
        return (
            self._active_realizations_field.isValid()
            and self._ensemble_selector.currentIndex() != -1
            and bool(self._active_realizations_field.text())
        )

    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=MANUAL_UPDATE_MODE,
            ensemble_id=str(self._ensemble_selector.selected_ensemble.id),
            realizations=self._active_realizations_field.text(),
            target_ensemble=self._ensemble_format_model.getValue(),  # type: ignore
        )

    def _realizations_from_fs(self) -> None:
        ensemble = self._ensemble_selector.selected_ensemble
        if ensemble:
            parameters = ensemble.get_realization_mask_with_parameters()
            responses = ensemble.get_realization_mask_with_responses()
            mask = np.logical_and(parameters, responses)
            self._active_realizations_field.model.setValueFromMask(mask)  # type: ignore
