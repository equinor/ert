from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout, QLabel, QLineEdit

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.ertmodel import get_runnable_realizations_mask
from ert.gui.ertwidgets.models.init_iter_value import IterValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.run_models import EnsembleExperiment
from ert.validation import IntegerArgument, RangeStringArgument

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    iter_num: int
    current_case: str
    experiment_name: str


class EnsembleExperimentPanel(SimulationConfigPanel):
    def __init__(self, ensemble_size: int, run_path: str, notifier: ErtNotifier):
        self.notifier = notifier
        super().__init__(EnsembleExperiment)
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()

        self._name_field = QLineEdit()
        self._name_field.setPlaceholderText("ensemble_experiment")
        self._name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._name_field)

        self._case_selector = CaseSelector(notifier)
        layout.addRow("Current case:", self._case_selector)
        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(ensemble_size),
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(ensemble_size),
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        self._iter_field = StringBox(
            IterValueModel(notifier),
            "config/simulation/iter_num",
        )
        self._iter_field.setValidator(
            IntegerArgument(from_value=0),
        )
        layout.addRow("Iteration", self._iter_field)

        self.setLayout(layout)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._case_selector.currentIndexChanged.connect(self._realizations_from_fs)

    def isConfigurationValid(self):
        return self._active_realizations_field.isValid() and self._iter_field.isValid()

    def getSimulationArguments(self):
        return Arguments(
            mode="ensemble_experiment",
            current_case=self.notifier.current_case_name,
            iter_num=int(self._iter_field.text()),
            realizations=self._active_realizations_field.text(),
            experiment_name=(
                self._name_field.text()
                if self._name_field.text() != ""
                else self._name_field.placeholderText()
            ),
        )

    def _realizations_from_fs(self):
        case = str(self._case_selector.currentText())
        mask = get_runnable_realizations_mask(self.notifier.storage, case)
        self._active_realizations_field.model.setValueFromMask(mask)
