from qtpy.QtWidgets import QFormLayout, QLabel

from ert_gui.ertwidgets import addHelpToWidget
from ert_gui.ertwidgets.caseselector import CaseSelector
from ecl.util.util import BoolVector
from ert_gui.ertwidgets.models.ertmodel import (
    getRunPath,
)
from ert_gui.simulation.simulation_config_panel import SimulationConfigPanel
from ert_shared.models import EnsemblePrefectExperiment
import yaml


class PrefectEnsembleExperimentPanel(SimulationConfigPanel):
    def __init__(self, config_file):
        SimulationConfigPanel.__init__(self, EnsemblePrefectExperiment)
        self.setObjectName("Prefect_Ensemble_experiment_panel")
        self.config_file = config_file

        layout = QFormLayout()

        self._case_selector = CaseSelector()
        layout.addRow("Current case:", self._case_selector)

        run_path_label = QLabel("<b>%s</b>" % getRunPath())
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        self.number_of_realizations = self.get_realizations(config_file)
        number_of_realizations_label = QLabel("<b>%d</b>" % self.number_of_realizations)
        addHelpToWidget(
            number_of_realizations_label, "config/ensemble/num_realizations"
        )
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self.setLayout(layout)

    def isConfigurationValid(self):
        return True

    def getSimulationArguments(self):
        return {
            "active_realizations": BoolVector(
                default_value=True, initial_size=self.number_of_realizations
            ),
            "config_file": self.config_file,
        }

    @staticmethod
    def get_realizations(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            return config["realizations"]
