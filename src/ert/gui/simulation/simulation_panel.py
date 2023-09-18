from collections import OrderedDict
from typing import Any, Dict

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.cli.model_factory import create_model
from ert.enkf_main import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import resourceIcon
from ert.libres_facade import LibresFacade

from .ensemble_experiment_panel import EnsembleExperimentPanel
from .ensemble_smoother_panel import EnsembleSmootherPanel
from .iterated_ensemble_smoother_panel import IteratedEnsembleSmootherPanel
from .multiple_data_assimilation_panel import MultipleDataAssimilationPanel
from .run_dialog import RunDialog
from .simulation_config_panel import SimulationConfigPanel
from .single_test_run_panel import SingleTestRunPanel


class SimulationPanel(QWidget):
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier, config_file: str):
        QWidget.__init__(self)
        self._notifier = notifier
        self.ert = ert
        self.facade = LibresFacade(ert)
        self._config_file = config_file

        self.setObjectName("Simulation_panel")
        layout = QVBoxLayout()

        self._simulation_mode_combo = QComboBox()
        self._simulation_mode_combo.setObjectName("Simulation_mode")

        self._simulation_mode_combo.currentIndexChanged.connect(
            self.toggleSimulationMode
        )

        simulation_mode_layout = QHBoxLayout()
        simulation_mode_layout.addSpacing(10)
        simulation_mode_layout.addWidget(QLabel("Simulation mode:"), 0, Qt.AlignVCenter)
        simulation_mode_layout.addWidget(
            self._simulation_mode_combo, 0, Qt.AlignVCenter
        )

        simulation_mode_layout.addSpacing(20)

        self.run_button = QToolButton()
        self.run_button.setObjectName("start_simulation")
        self.run_button.setText("Run Experiment")
        self.run_button.setIcon(resourceIcon("play_circle.svg"))
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.clicked.connect(self.runSimulation)
        self.run_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        simulation_mode_layout.addWidget(self.run_button)
        simulation_mode_layout.addStretch(1)

        layout.addSpacing(5)
        layout.addLayout(simulation_mode_layout)
        layout.addSpacing(10)

        self._simulation_stack = QStackedWidget()
        self._simulation_stack.setLineWidth(1)
        self._simulation_stack.setFrameStyle(QFrame.StyledPanel)

        layout.addWidget(self._simulation_stack)

        self._simulation_widgets = OrderedDict()
        """ :type: OrderedDict[BaseRunModel,SimulationConfigPanel]"""
        self.addSimulationConfigPanel(SingleTestRunPanel(ert, notifier), True)
        self.addSimulationConfigPanel(EnsembleExperimentPanel(ert, notifier), True)

        simulation_mode_valid = (
            self.facade.have_smoother_parameters and self.facade.have_observations
        )

        self.addSimulationConfigPanel(
            EnsembleSmootherPanel(ert, notifier), simulation_mode_valid
        )
        self.addSimulationConfigPanel(
            MultipleDataAssimilationPanel(self.facade, notifier), simulation_mode_valid
        )
        self.addSimulationConfigPanel(
            IteratedEnsembleSmootherPanel(self.facade, notifier),
            simulation_mode_valid,
        )

        self.setLayout(layout)

    def addSimulationConfigPanel(self, panel, mode_enabled: bool):
        assert isinstance(panel, SimulationConfigPanel)
        self._simulation_stack.addWidget(panel)
        simulation_model = panel.getSimulationModel()
        self._simulation_widgets[simulation_model] = panel
        self._simulation_mode_combo.addItem(simulation_model.name(), simulation_model)

        if not mode_enabled:
            item_count = self._simulation_mode_combo.count() - 1
            sim_item = self._simulation_mode_combo.model().item(item_count)
            sim_item.setEnabled(False)
            sim_item.setToolTip("Both observations and parameters must be defined")
            sim_item.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxWarning))

        panel.simulationConfigurationChanged.connect(self.validationStatusChanged)

    def getActions(self):
        return []

    def getCurrentSimulationModel(self):
        return self._simulation_mode_combo.itemData(
            self._simulation_mode_combo.currentIndex(), Qt.UserRole
        )

    def getSimulationArguments(self) -> Dict[str, Any]:
        simulation_widget = self._simulation_widgets[self.getCurrentSimulationModel()]
        args = simulation_widget.getSimulationArguments()
        return args

    def runSimulation(self):
        message = (
            f"Are you sure you want to use case '{self._notifier.current_case_name}' "
            "for initialization of the initial ensemble when running the experiments?"
        )
        if (
            QMessageBox.question(
                self, "Run experiments?", message, QMessageBox.Yes | QMessageBox.No
            )
            == QMessageBox.Yes
        ):
            abort = False
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                args = self.getSimulationArguments()
                experiment = self._notifier.storage.create_experiment(
                    parameters=self.ert.ensembleConfig().parameter_configuration,
                    responses=self.ert.ensembleConfig().response_configuration,
                )
                model = create_model(
                    self.ert,
                    self._notifier.storage,
                    args,
                    experiment.id,
                )
                experiment.write_simulation_arguments(model.simulation_arguments)

            except ValueError as e:
                QMessageBox.warning(
                    self, "ERROR: Failed to create experiment", (str(e)), QMessageBox.Ok
                )
                abort = True

            QApplication.restoreOverrideCursor()

            if (
                not abort
                and model.check_if_runpath_exists()
                and (
                    QMessageBox.warning(
                        self,
                        "Run experiments?",
                        (
                            "ERT is running in an existing runpath.\n\n"
                            "Please be aware of the following:\n"
                            "- Previously generated results "
                            "might be overwritten.\n"
                            "- Previously generated files might "
                            "be used if not configured correctly.\n"
                            "Are you sure you want to continue?"
                        ),
                        QMessageBox.Yes | QMessageBox.No,
                    )
                    == QMessageBox.No
                )
            ):
                abort = True

            if not abort:
                dialog = RunDialog(
                    self._config_file, model, self._notifier, self.parent()
                )
                self.run_button.setDisabled(True)
                self.run_button.setText("Experiment running...")
                dialog.startSimulation()
                dialog.show()

                def exit_handler():
                    self.run_button.setText("Run Experiment")
                    self.run_button.setDisabled(False)
                    self._notifier.emitErtChange()

                dialog.finished.connect(exit_handler)

                self._notifier.emitErtChange()  # experiments may have added new cases

    def toggleSimulationMode(self):
        current_model = self.getCurrentSimulationModel()
        if current_model is not None:
            widget = self._simulation_widgets[self.getCurrentSimulationModel()]
            self._simulation_stack.setCurrentWidget(widget)
            self.validationStatusChanged()

    def validationStatusChanged(self):
        widget = self._simulation_widgets[self.getCurrentSimulationModel()]
        self.run_button.setEnabled(widget.isConfigurationValid())
