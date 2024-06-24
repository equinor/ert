from dataclasses import dataclass

import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.gui.simulation.experiment_config_panel import ExperimentConfigPanel
from ert.mode_definitions import EVALUATE_ENSEMBLE_MODE
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.validation import RangeStringArgument


@dataclass
class Arguments:
    mode: str
    realizations: str
    ensemble_id: str


class EvaluateEnsemblePanel(ExperimentConfigPanel):
    def __init__(self, ensemble_size: int, run_path: str, notifier: ErtNotifier):
        self.notifier = notifier
        super().__init__(EvaluateEnsemble)
        self.setObjectName("Evaluate_parameters_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(EvaluateEnsemble.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addRow(lab)
        self._ensemble_selector = EnsembleSelector(notifier, show_only_no_children=True)
        layout.addRow("Ensemble:", self._ensemble_selector)
        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

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
            mode=EVALUATE_ENSEMBLE_MODE,
            ensemble_id=str(self._ensemble_selector.selected_ensemble.id),
            realizations=self._active_realizations_field.text(),
        )

    def _realizations_from_fs(self) -> None:
        ensemble = self._ensemble_selector.selected_ensemble
        if ensemble:
            parameters = ensemble.get_realization_mask_with_parameters()
            missing_responses = ~ensemble.get_realization_mask_with_responses()
            failures = ~ensemble.get_realization_mask_without_failure()
            mask = np.logical_and(
                parameters, np.logical_or(missing_responses, failures)
            )
            self._active_realizations_field.model.setValueFromMask(mask)  # type: ignore
