from collections import OrderedDict
from typing import Any, Dict

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
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

from ert.cli import ENSEMBLE_SMOOTHER_MODE, ITERATIVE_ENSEMBLE_SMOOTHER_MODE
from ert.cli.model_factory import create_model
from ert.enkf_main import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.libres_facade import LibresFacade

from .ensemble_experiment_panel import EnsembleExperimentPanel
from .ensemble_smoother_panel import EnsembleSmootherPanel
from .evaluate_ensemble_panel import EvaluateEnsemblePanel
from .iterated_ensemble_smoother_panel import IteratedEnsembleSmootherPanel
from .multiple_data_assimilation_panel import MultipleDataAssimilationPanel
from .run_dialog import RunDialog
from .simulation_config_panel import SimulationConfigPanel
from .single_test_run_panel import SingleTestRunPanel

EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE = "Run Experiment"
EXPERIMENT_IS_RUNNING_BUTTON_MESSAGE = "Experiment running..."


class SimulationPanel(QWidget):
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier, config_file: str):
        QWidget.__init__(self)
        self._notifier = notifier
        self.ert = ert
        self.facade = LibresFacade(ert)
        ensemble_size = self.facade.get_ensemble_size()
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
        self.run_button.setText(EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE)
        self.run_button.setIcon(QIcon("img:play_circle.svg"))
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
        self.addSimulationConfigPanel(
            SingleTestRunPanel(self.facade.run_path, notifier),
            True,
        )
        self.addSimulationConfigPanel(
            EnsembleExperimentPanel(ensemble_size, self.facade.run_path, notifier),
            True,
        )
        self.addSimulationConfigPanel(
            EvaluateEnsemblePanel(ensemble_size, self.facade.run_path, notifier),
            True,
        )

        config = self.facade.config
        simulation_mode_valid = (
            config.ensemble_config.parameter_configs and config.observations
        )
        analysis_config = self.facade.config.analysis_config
        self.addSimulationConfigPanel(
            EnsembleSmootherPanel(
                analysis_config, self.facade.run_path, notifier, ensemble_size
            ),
            simulation_mode_valid,
        )
        self.addSimulationConfigPanel(
            MultipleDataAssimilationPanel(
                analysis_config, self.facade.run_path, notifier, ensemble_size
            ),
            simulation_mode_valid,
        )
        self.addSimulationConfigPanel(
            IteratedEnsembleSmootherPanel(
                analysis_config, self.facade.run_path, notifier, ensemble_size
            ),
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

    def getExperimentName(self) -> str:
        """Get the experiment name as provided by the user. Defaults to run mode if not set."""
        return self.getSimulationArguments().experiment_name

    def runSimulation(self):
        args = self.getSimulationArguments()
        if args.mode == "es_mda":
            if args.restart_run:
                message = (
                    "Are you sure you want to restart from ensemble"
                    f" '{self._notifier.current_ensemble_name}'?"
                )
            else:
                message = (
                    "Are you sure you want to use "
                    f"target ensemble format '{args.target_ensemble}'?"
                )
        elif args.mode in [ENSEMBLE_SMOOTHER_MODE, ITERATIVE_ENSEMBLE_SMOOTHER_MODE]:
            message = (
                "Are you sure you want to use ensemble "
                f"'{self._notifier.current_ensemble_name}' for initialization"
                " of the initial ensemble when running the experiment?"
            )
        else:
            message = ""
        if (
            not message
            or message
            and (
                QMessageBox.question(
                    self, "Run experiments?", message, QMessageBox.Yes | QMessageBox.No
                )
                == QMessageBox.Yes
            )
        ):
            abort = False
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            config = self.facade.config
            try:
                model = create_model(
                    config,
                    self._notifier.storage,
                    args,
                )

            except ValueError as e:
                QMessageBox.warning(
                    self, "ERROR: Failed to create experiment", (str(e)), QMessageBox.Ok
                )
                abort = True

            QApplication.restoreOverrideCursor()

            delete_runpath_checkbox = None

            if not abort and model.check_if_runpath_exists():
                msg_box = QMessageBox(self)
                msg_box.setObjectName("RUN_PATH_WARNING_BOX")

                msg_box.setIcon(QMessageBox.Warning)

                msg_box.setText("Run experiments")
                msg_box.setInformativeText(
                    (
                        "ERT is running in an existing runpath.\n\n"
                        "Please be aware of the following:\n"
                        "- Previously generated results "
                        "might be overwritten.\n"
                        "- Previously generated files might "
                        "be used if not configured correctly.\n"
                        "Are you sure you want to continue?"
                    )
                )

                delete_runpath_checkbox = QCheckBox()
                delete_runpath_checkbox.setText("Delete run_path")
                msg_box.setCheckBox(delete_runpath_checkbox)

                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.No)

                msg_box.setWindowModality(Qt.ApplicationModal)

                msg_box_res = msg_box.exec()

            if (
                not abort
                and model.check_if_runpath_exists()
                and msg_box_res == QMessageBox.No
            ):
                abort = True

            if not abort:
                if (
                    delete_runpath_checkbox is not None
                    and delete_runpath_checkbox.checkState() == Qt.Checked
                ):
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                    model.rm_run_path()
                    QApplication.restoreOverrideCursor()

                dialog = RunDialog(
                    self._config_file, model, self._notifier, self.parent()
                )
                self.run_button.setEnabled(False)
                self.run_button.setText(EXPERIMENT_IS_RUNNING_BUTTON_MESSAGE)
                dialog.startSimulation()
                dialog.show()

                def exit_handler():
                    self.run_button.setText(EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE)
                    self.run_button.setEnabled(True)
                    self.toggleSimulationMode()
                    self._notifier.emitErtChange()

                dialog.finished.connect(exit_handler)

    def toggleSimulationMode(self):
        current_model = self.getCurrentSimulationModel()
        if current_model is not None:
            widget = self._simulation_widgets[self.getCurrentSimulationModel()]
            self._simulation_stack.setCurrentWidget(widget)
            self.validationStatusChanged()

    def validationStatusChanged(self):
        widget = self._simulation_widgets[self.getCurrentSimulationModel()]
        self.run_button.setEnabled(
            self.run_button.text() == EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE
            and widget.isConfigurationValid()
        )
