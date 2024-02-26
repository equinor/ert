from dataclasses import dataclass

from qtpy import QtCore
from qtpy.QtWidgets import QFormLayout, QLabel

from ert.cli import EVALUATE_ENSEMBLE_MODE
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.gui.simulation.simulation_config_panel import SimulationConfigPanel
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.validation import RangeStringArgument


@dataclass
class Arguments:
    mode: str
    realizations: str
    ensemble_name: str


class EvaluateEnsemblePanel(SimulationConfigPanel):
    def __init__(self, ensemble_size: int, run_path: str, notifier: ErtNotifier):
        self.notifier = notifier
        super().__init__(EvaluateEnsemble)
        self.setObjectName("Evaluate_parameters_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(EvaluateEnsemble.__doc__.split()))
        lab.setWordWrap(True)
        lab.setAlignment(QtCore.Qt.AlignLeft)
        layout.addRow(lab)
        self._case_selector = CaseSelector(notifier, show_only_initialized=True)
        layout.addRow("Ensemble:", self._case_selector)
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

        self.setLayout(layout)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._case_selector.case_populated.connect(self._realizations_from_fs)

    def isConfigurationValid(self):
        return self._active_realizations_field.isValid()

    def getSimulationArguments(self):
        return Arguments(
            mode=EVALUATE_ENSEMBLE_MODE,
            ensemble_name=self._case_selector.currentText(),
            realizations=self._active_realizations_field.text(),
        )

    def _realizations_from_fs(self):
        case = str(self._case_selector.currentText())
        if case:
            mask = self.notifier.storage.get_ensemble_by_name(
                case
            ).get_realization_mask_with_parameters()
            self._active_realizations_field.model.setValueFromMask(mask)
