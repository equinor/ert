from dataclasses import dataclass

from qtpy import QtCore
from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import StringBox, TextModel
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.init_iter_value import IterValueModel
from ert.run_models import EnsembleExperiment
from ert.validation import IntegerArgument, RangeStringArgument
from ert.validation.range_string_argument import NotInStorage

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    iter_num: int
    current_ensemble: str
    experiment_name: str


class EnsembleExperimentPanel(SimulationConfigPanel):
    def __init__(self, ensemble_size: int, run_path: str, notifier: ErtNotifier):
        self.notifier = notifier
        super().__init__(EnsembleExperiment)
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(EnsembleExperiment.__doc__.split()))
        lab.setWordWrap(True)
        lab.setAlignment(QtCore.Qt.AlignLeft)
        layout.addRow(lab)

        self._name_field = StringBox(
            TextModel(""), placeholder_text="ensemble-experiment"
        )
        self._name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._name_field)
        self._name_field.setValidator(
            NotInStorage(self.notifier.storage, "experiments")
        )
        self._ensemble_name_field = StringBox(
            TextModel(""), placeholder_text="ensemble"
        )
        self._ensemble_name_field.setMinimumWidth(250)
        self._ensemble_name_field.setValidator(
            NotInStorage(self.notifier.storage, "ensembles")
        )

        layout.addRow("Ensemble name:", self._ensemble_name_field)

        self._ensemble_selector = EnsembleSelector(notifier)
        layout.addRow("Current ensemble:", self._ensemble_selector)
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
        self._name_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._ensemble_name_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )

    def isConfigurationValid(self):
        self.blockSignals(True)
        self._name_field.validateString()
        self._ensemble_name_field.validateString()
        self.blockSignals(False)
        return (
            self._active_realizations_field.isValid()
            and self._iter_field.isValid()
            and self._name_field.isValid()
            and self._ensemble_name_field.isValid()
        )

    def getSimulationArguments(self):
        return Arguments(
            mode="ensemble_experiment",
            current_ensemble=self._ensemble_name_field.get_text,
            iter_num=int(self._iter_field.text()),
            realizations=self._active_realizations_field.text(),
            experiment_name=self._name_field.get_text,
        )
