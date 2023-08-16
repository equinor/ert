from dataclasses import dataclass
from typing import Optional

from qtpy.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QWidget

from ert.enkf_main import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.ertmodel import get_runnable_realizations_mask
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.run_models import EnsembleExperiment
from ert.storage import EnsembleReader
from ert.validation import RangeStringArgument

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    prior_ensemble: Optional[EnsembleReader]
    current_case: str


class EnsembleExperimentPanel(SimulationConfigPanel):
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier):
        self.ert = ert
        self.notifier = notifier
        self.facade = LibresFacade(ert)
        super().__init__(EnsembleExperiment)
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()

        self._case_selector = CaseSelector(notifier, placeholder="Empty case")
        layout.addRow("Current case:", self._case_selector)
        runpath_label = CopyableLabel(text=self.facade.run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(
            f"<b>{self.facade.get_ensemble_size()}</b>"
        )
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(self.facade),
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(self.facade.get_ensemble_size()),
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        self._prior_case = self.setup_prior_case(layout)

        self.setLayout(layout)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._case_selector.currentIndexChanged.connect(self._realizations_from_fs)

    def setup_prior_case(self, parent_layout: QFormLayout) -> CaseSelector:
        prior = CaseSelector(self.notifier, update_ert=False)
        prior.currentIndexChanged.connect(self.simulationConfigurationChanged)
        parent_layout.addRow("Prior case", prior)

        return prior

    def get_prior_case(self) -> Optional[EnsembleReader]:
        if self._prior_case.isEnabled():
            return self._prior_case.currentData()
        return None

    def isConfigurationValid(self):
        return self._active_realizations_field.isValid() and (
            (self.notifier.current_case is None and self.get_prior_case() is None)
            or (self.get_prior_case() != self.notifier.current_case)
        )

    def getSimulationArguments(self):
        return Arguments(
            mode="ensemble_experiment",
            current_case=self.notifier.current_case_name,
            prior_ensemble=self.get_prior_case(),
            realizations=self._active_realizations_field.text(),
        )

    def _realizations_from_fs(self):
        case = str(self._case_selector.currentText())
        mask = get_runnable_realizations_mask(self.notifier.storage, case)
        self._active_realizations_field.model.setValueFromMask(mask)
