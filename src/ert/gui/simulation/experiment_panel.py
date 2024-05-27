from collections import OrderedDict
from queue import SimpleQueue
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

from ert.cli.model_factory import create_model
from ert.enkf_main import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.libres_facade import LibresFacade
from ert.mode_definitions import (
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
)

from .ensemble_experiment_panel import EnsembleExperimentPanel
from .ensemble_smoother_panel import EnsembleSmootherPanel
from .evaluate_ensemble_panel import EvaluateEnsemblePanel
from .experiment_config_panel import ExperimentConfigPanel
from .iterated_ensemble_smoother_panel import IteratedEnsembleSmootherPanel
from .multiple_data_assimilation_panel import MultipleDataAssimilationPanel
from .run_dialog import RunDialog
from .single_test_run_panel import SingleTestRunPanel

EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE = "Run Experiment"
EXPERIMENT_IS_RUNNING_BUTTON_MESSAGE = "Experiment running..."


class ExperimentPanel(QWidget):
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier, config_file: str):
        QWidget.__init__(self)
        self._notifier = notifier
        self.ert = ert
        self.facade = LibresFacade(ert)
        ensemble_size = self.facade.get_ensemble_size()
        self._config_file = config_file

        self.setObjectName("experiment_panel")
        layout = QVBoxLayout()

        self._experiment_type_combo = QComboBox()
        self._experiment_type_combo.setObjectName("experiment_type")

        self._experiment_type_combo.currentIndexChanged.connect(
            self.toggleExperimentType
        )

        experiment_type_layout = QHBoxLayout()
        experiment_type_layout.addSpacing(10)
        experiment_type_layout.addWidget(QLabel("Experiment type:"), 0, Qt.AlignVCenter)
        experiment_type_layout.addWidget(
            self._experiment_type_combo, 0, Qt.AlignVCenter
        )

        experiment_type_layout.addSpacing(20)

        self.run_button = QToolButton()
        self.run_button.setObjectName("run_experiment")
        self.run_button.setText(EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE)
        self.run_button.setIcon(QIcon("img:play_circle.svg"))
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.clicked.connect(self.run_experiment)
        self.run_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        experiment_type_layout.addWidget(self.run_button)
        experiment_type_layout.addStretch(1)

        layout.addSpacing(5)
        layout.addLayout(experiment_type_layout)
        layout.addSpacing(10)

        self._experiment_stack = QStackedWidget()
        self._experiment_stack.setLineWidth(1)
        self._experiment_stack.setFrameStyle(QFrame.StyledPanel)

        layout.addWidget(self._experiment_stack)

        self._experiment_widgets = OrderedDict()
        self.addExperimentConfigPanel(
            SingleTestRunPanel(self.facade.run_path, notifier),
            True,
        )
        self.addExperimentConfigPanel(
            EnsembleExperimentPanel(ensemble_size, self.facade.run_path, notifier),
            True,
        )
        self.addExperimentConfigPanel(
            EvaluateEnsemblePanel(ensemble_size, self.facade.run_path, notifier),
            True,
        )

        config = self.facade.config
        experiment_type_valid = (
            config.ensemble_config.parameter_configs and config.observations
        )
        analysis_config = self.facade.config.analysis_config
        self.addExperimentConfigPanel(
            EnsembleSmootherPanel(
                analysis_config, self.facade.run_path, notifier, ensemble_size
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            MultipleDataAssimilationPanel(
                analysis_config, self.facade.run_path, notifier, ensemble_size
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            IteratedEnsembleSmootherPanel(
                analysis_config, self.facade.run_path, notifier, ensemble_size
            ),
            experiment_type_valid,
        )

        self.setLayout(layout)

    def addExperimentConfigPanel(self, panel, mode_enabled: bool):
        assert isinstance(panel, ExperimentConfigPanel)
        self._experiment_stack.addWidget(panel)
        experiment_type = panel.get_experiment_type()
        self._experiment_widgets[experiment_type] = panel
        self._experiment_type_combo.addItem(experiment_type.name(), experiment_type)

        if not mode_enabled:
            item_count = self._experiment_type_combo.count() - 1
            sim_item = self._experiment_type_combo.model().item(item_count)
            sim_item.setEnabled(False)
            sim_item.setToolTip("Both observations and parameters must be defined")
            sim_item.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxWarning))

        panel.simulationConfigurationChanged.connect(self.validationStatusChanged)

    @staticmethod
    def getActions():
        return []

    def get_current_experiment_type(self):
        return self._experiment_type_combo.itemData(
            self._experiment_type_combo.currentIndex(), Qt.UserRole
        )

    def get_experiment_arguments(self) -> Dict[str, Any]:
        simulation_widget = self._experiment_widgets[self.get_current_experiment_type()]
        args = simulation_widget.get_experiment_arguments()
        return args

    def getExperimentName(self) -> str:
        """Get the experiment name as provided by the user. Defaults to run mode if not set."""
        return self.get_experiment_arguments().experiment_name

    def run_experiment(self):
        args = self.get_experiment_arguments()
        if args.mode == ES_MDA_MODE:
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
            event_queue = SimpleQueue()
            try:
                model = create_model(
                    config,
                    self._notifier.storage,
                    args,
                    event_queue,
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

            delete_runpath = (
                delete_runpath_checkbox is not None
                and delete_runpath_checkbox.checkState() == Qt.Checked
            )
            if not abort and delete_runpath:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    model.rm_run_path()
                except OSError as e:
                    QApplication.restoreOverrideCursor()
                    msg_box = QMessageBox(self)
                    msg_box.setObjectName("RUN_PATH_ERROR_BOX")
                    msg_box.setIcon(QMessageBox.Warning)
                    msg_box.setText("ERT could not delete the existing runpath")
                    msg_box.setInformativeText(
                        (f"{e}\n\n" "Continue without deleting the runpath?")
                    )
                    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg_box.setDefaultButton(QMessageBox.No)
                    msg_box.setWindowModality(Qt.ApplicationModal)
                    msg_box_res = msg_box.exec()
                    abort = msg_box_res == QMessageBox.No
                QApplication.restoreOverrideCursor()

            if not abort:
                dialog = RunDialog(
                    self._config_file, model, event_queue, self._notifier, self.parent()
                )
                self.run_button.setEnabled(False)
                self.run_button.setText(EXPERIMENT_IS_RUNNING_BUTTON_MESSAGE)
                dialog.run_experiment()
                dialog.show()

                def exit_handler():
                    self.run_button.setText(EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE)
                    self.run_button.setEnabled(True)
                    self.toggleExperimentType()
                    self._notifier.emitErtChange()

                dialog.finished.connect(exit_handler)

    def toggleExperimentType(self):
        current_model = self.get_current_experiment_type()
        if current_model is not None:
            widget = self._experiment_widgets[self.get_current_experiment_type()]
            self._experiment_stack.setCurrentWidget(widget)
            self.validationStatusChanged()

    def validationStatusChanged(self):
        widget = self._experiment_widgets[self.get_current_experiment_type()]
        self.run_button.setEnabled(
            self.run_button.text() == EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE
            and widget.isConfigurationValid()
        )
