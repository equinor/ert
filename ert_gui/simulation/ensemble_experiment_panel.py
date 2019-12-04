from qtpy.QtWidgets import QFormLayout, QLabel

from ert_gui.ertwidgets import addHelpToWidget
from ert_gui.ertwidgets.caseselector import CaseSelector
from ert_gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount, getRunPath, get_runnable_realizations_mask
from ert_gui.ertwidgets.stringbox import StringBox
from ert_gui.ide.keywords.definitions import RangeStringArgument
from ert_shared.models import EnsembleExperiment
from ert_gui.simulation.simulation_config_panel import SimulationConfigPanel


class EnsembleExperimentPanel(SimulationConfigPanel):

    def __init__(self):
        SimulationConfigPanel.__init__(self, EnsembleExperiment)

        layout = QFormLayout()

        self._case_selector = CaseSelector()
        layout.addRow("Current case:", self._case_selector)

        run_path_label = QLabel("<b>%s</b>" % getRunPath())
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        number_of_realizations_label = QLabel("<b>%d</b>" % getRealizationCount())
        addHelpToWidget(number_of_realizations_label, "config/ensemble/num_realizations")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(),
            "config/simulation/active_realizations",
            )
        self._active_realizations_field.setValidator(
            RangeStringArgument(getRealizationCount()),
            )
        layout.addRow("Active realizations", self._active_realizations_field)

        self.setLayout(layout)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(self.simulationConfigurationChanged)
        self._case_selector.currentIndexChanged.connect(self._realizations_from_fs)

        self._realizations_from_fs()  # update with the current case


    def isConfigurationValid(self):
        return self._active_realizations_field.isValid()


    def getSimulationArguments(self):
        active_realizations_mask = \
            self._active_realizations_field.model.getActiveRealizationsMask()
        return {"active_realizations": active_realizations_mask}


    def _realizations_from_fs(self):
        case = str(self._case_selector.currentText())
        mask = get_runnable_realizations_mask(case)
        self._active_realizations_field.model.setValueFromMask(mask)
