from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout, QLabel

from ert._c_wrappers.enkf import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import addHelpToWidget
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.ertmodel import get_runnable_realizations_mask
from ert.gui.ertwidgets.models.init_iter_value import IterValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.shared.ide.keywords.definitions import IntegerArgument, RangeStringArgument
from ert.shared.models import EnsembleExperiment

from .simulation_config_panel import SimulationConfigPanel, escape_string


@dataclass
class Arguments:
    mode: str
    realizations: str
    iter_num: int


class EnsembleExperimentPanel(SimulationConfigPanel):
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier):
        self.ert = ert
        self.facade = LibresFacade(ert)
        super().__init__(EnsembleExperiment)
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()

        self._case_selector = CaseSelector(self.facade, notifier)
        layout.addRow("Current case:", self._case_selector)
        run_path_label = QLabel(f"<b>{escape_string(self.facade.run_path)}</b>")
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        number_of_realizations_label = QLabel(
            f"<b>{self.facade.get_ensemble_size()}</b>"
        )
        addHelpToWidget(
            number_of_realizations_label, "config/ensemble/num_realizations"
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

        self._realizations_from_fs()  # update with the current case

    def isConfigurationValid(self):
        return self._active_realizations_field.isValid() and self._iter_field.isValid()

    def getSimulationArguments(self):
        return Arguments(
            mode="ensemble_experiment",
            iter_num=int(self._iter_field.text()),
            realizations=self._active_realizations_field.text(),
        )

    def _realizations_from_fs(self):
        case = str(self._case_selector.currentText())
        mask = get_runnable_realizations_mask(self.ert, case)
        self._active_realizations_field.model.setValueFromMask(mask)
