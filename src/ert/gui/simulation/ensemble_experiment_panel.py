from dataclasses import dataclass

from qtpy import QtCore
from qtpy.QtCore import Slot
from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import StringBox, TextModel
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.run_models import EnsembleExperiment
from ert.validation import RangeStringArgument
from ert.validation.range_string_argument import NotInStorage

from .experiment_config_panel import ExperimentConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    current_ensemble: str
    experiment_name: str


class EnsembleExperimentPanel(ExperimentConfigPanel):
    def __init__(self, ensemble_size: int, run_path: str, notifier: ErtNotifier):
        self.notifier = notifier
        super().__init__(EnsembleExperiment)
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(EnsembleExperiment.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addRow(lab)

        self._experiment_name_field = StringBox(
            TextModel(""),
            placeholder_text=self.notifier.storage.get_unique_experiment_name(
                ENSEMBLE_EXPERIMENT_MODE
            ),
        )
        self._experiment_name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._experiment_name_field)
        self._experiment_name_field.setValidator(
            NotInStorage(self.notifier.storage, "experiments")
        )

        self._ensemble_name_field = StringBox(
            TextModel(""), placeholder_text="ensemble"
        )
        self._ensemble_name_field.setMinimumWidth(250)

        layout.addRow("Ensemble name:", self._ensemble_name_field)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(ensemble_size),  # type: ignore
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(ensemble_size),
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        self.setLayout(layout)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._experiment_name_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._ensemble_name_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )

        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @Slot(ExperimentConfigPanel)
    def experimentTypeChanged(self, w: ExperimentConfigPanel) -> None:
        if isinstance(w, EnsembleExperimentPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ENSEMBLE_EXPERIMENT_MODE)
        )

    def isConfigurationValid(self) -> bool:
        self.blockSignals(True)
        self._experiment_name_field.validateString()
        self._ensemble_name_field.validateString()
        self.blockSignals(False)
        return (
            self._active_realizations_field.isValid()
            and self._experiment_name_field.isValid()
            and self._ensemble_name_field.isValid()
        )

    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ENSEMBLE_EXPERIMENT_MODE,
            current_ensemble=self._ensemble_name_field.get_text,
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
