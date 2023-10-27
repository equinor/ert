from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.ertmodel import get_runnable_realizations_mask
from ert.gui.ertwidgets.models.init_iter_value import IterValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.run_models import EnsembleExperiment
from ert.validation import IntegerArgument, RangeStringArgument

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    iter_num: int
    current_case: str


class EnsembleExperimentPanel(SimulationConfigPanel):
    def __init__(self, facade: LibresFacade, notifier: ErtNotifier):
        self.notifier = notifier
        self.facade = facade
        super().__init__(EnsembleExperiment)
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()

        self._case_selector = CaseSelector(notifier)
        layout.addRow("Current case:", self._case_selector)
        runpath_label = CopyableLabel(text=self.facade.run_path_stripped)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(
            f"<b>{self.facade.get_ensemble_size()}</b>"
        )
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(self.facade.get_ensemble_size()),
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(self.facade.get_ensemble_size()),
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
        )

    def _realizations_from_fs(self):
        case = str(self._case_selector.currentText())
        mask = get_runnable_realizations_mask(self.notifier.storage, case)
        self._active_realizations_field.model.setValueFromMask(mask)
