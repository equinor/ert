from PyQt4.QtCore import Qt, QSize
from PyQt4.QtGui import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget, QFormLayout, QFrame, QPushButton, QToolButton, QSpacerItem, QLayout, QListView
from ert_gui.ide.keywords.definitions import ProperNameArgument
from ert_gui.ide.keywords.definitions.range_string_argument import RangeStringArgument
from ert_gui.models.connectors import RunPathModel, EnsembleSizeModel, RerunPathModel
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.models.connectors.run import SimulationModeModel, EnsembleSmoother, TargetCaseModel, ActiveRealizationsModel, SimulationRunner
from ert_gui.models.connectors.run.analysis_module_model import AnalysisModuleModel
from ert_gui.models.connectors.run.ensemble_experiment import EnsembleExperiment
from ert_gui.pages.run_dialog import RunDialog
from ert_gui.simulation.ensemble_experiment_panel import EnsembleExperimentPanel
from ert_gui.simulation.ensemble_smoother_panel import EnsembleSmootherPanel
from ert_gui.simulation.simulation_config_panel import SimulationConfigPanel
from ert_gui.widgets import util
from ert_gui.widgets.active_label import ActiveLabel

from ert_gui.widgets.combo_choice import ComboChoice
from ert_gui.widgets.string_box import StringBox
from ert_gui.widgets.warning_panel import WarningPanel


class SimulationPanel(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        layout = QVBoxLayout()

        simulation_mode_layout = QHBoxLayout()
        simulation_mode_layout.addSpacing(10)
        simulation_mode_model = SimulationModeModel()
        simulation_mode_model.observable().attach(SimulationModeModel.CURRENT_CHOICE_CHANGED_EVENT, self.toggleSimulationMode)
        simulation_mode_combo = ComboChoice(simulation_mode_model, "Simulation mode", "run/simulation_mode")
        simulation_mode_layout.addWidget(QLabel(simulation_mode_combo.getLabel()), 0, Qt.AlignVCenter)
        simulation_mode_layout.addWidget(simulation_mode_combo, 0, Qt.AlignVCenter)

        # simulation_mode_layout.addStretch()
        simulation_mode_layout.addSpacing(20)

        self.run_button = QToolButton()
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.setText("Start Simulation")
        self.run_button.setIcon(util.resourceIcon("ide/gear_in_play"))
        self.run_button.clicked.connect(self.runSimulation)
        self.run_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)


        simulation_mode_layout.addWidget(self.run_button)
        simulation_mode_layout.addStretch(1)

        layout.addSpacing(5)
        layout.addLayout(simulation_mode_layout)
        layout.addSpacing(10)

        self.simulation_stack = QStackedWidget()
        self.simulation_stack.setLineWidth(1)
        self.simulation_stack.setFrameStyle(QFrame.StyledPanel)


        layout.addWidget(self.simulation_stack)

        self.simulation_widgets = {}

        self.addSimulationConfigPanel(EnsembleExperimentPanel())
        self.addSimulationConfigPanel(EnsembleSmootherPanel())

        # case_model = CaseSelectorModel()
        # case_selector = ComboChoice(case_model, "Current case", "init/current_case_selection")
        # self.addRow(case_selector, CaseInitializationConfigurationPanel(), configuration_title="Manage cases")
        #
        # # Give a warning if the case is not initialized!
        # IsCaseInitializedModel().observable().attach(IsCaseInitializedModel.TEXT_VALUE_CHANGED_EVENT, self.updateSimulationStatus)
        #
        # runpath_model = RunPathModel()
        # self.addRow(PathFormatChooser(runpath_model, "Runpath", "config/simulation/runpath"))
        #
        # ensemble_size_model = EnsembleSizeModel()
        # self.addRow(IntegerSpinner(ensemble_size_model, "Number of realizations", "config/ensemble/num_realizations"))
        #
        #
        #
        # self.addSpace(10)
        # self.run = FunctionButtonModel("Run", self.runSimulation)
        # self.run.setEnabled(SimulationModeModel().getCurrentChoice().buttonIsEnabled())
        #
        # self.run_button = Button(self.run, label="Start simulation", help_link="run/run")
        # self.run_button.addStretch()
        #
        # self.config_and_run = FunctionButtonModel("Configure and Run", self.configureAndRunSimulation)
        # self.config_and_run.setEnabled(SimulationModeModel().getCurrentChoice().buttonIsEnabled())
        #
        # self.run_button.addOption(self.config_and_run)
        # self.run_button.addOption(OneMoreIteration())
        # self.addRow(self.run_button)
        # self.addSpace(10)
        #
        # self.warning_panel = WarningPanel()
        # layout.addWidget(self.warning_panel)
        #
        # simulation_mode_model.observable().attach(SimulationModeModel.CURRENT_CHOICE_CHANGED_EVENT, self.toggleSimulationMode)
        #
        # self.updateSimulationStatus()

        self.setLayout(layout)

    def addSimulationConfigPanel(self, panel):
        assert isinstance(panel, SimulationConfigPanel)

        self.simulation_stack.addWidget(panel)
        self.simulation_widgets[panel.getSimulationModel()] = panel

        panel.simulationConfigurationChanged.connect(self.validationStatusChanged)


    def getCurrentSimulationMode(self):
        return SimulationModeModel().getCurrentChoice()

    def runSimulation(self):
        simulation_runner = SimulationRunner(self.getCurrentSimulationMode())

        dialog = RunDialog(simulation_runner)
        simulation_runner.start()
        dialog.exec_()


    def toggleSimulationMode(self):
        widget = self.simulation_widgets[self.getCurrentSimulationMode()]
        self.simulation_stack.setCurrentWidget(widget)
        self.validationStatusChanged()

    def validationStatusChanged(self):
        widget = self.simulation_widgets[self.getCurrentSimulationMode()]
        self.run_button.setEnabled(widget.isConfigurationValid())




