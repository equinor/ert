from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction, QIcon, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from _ert.threading import ErtThread
from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.run_models import RunModel, StatusEvents, create_model

from ..find_ert_info import find_ert_info
from ..summarypanel import SummaryPanel
from .combobox_with_description import QComboBoxWithDescription
from .ensemble_experiment_panel import EnsembleExperimentPanel
from .ensemble_information_filter_panel import EnsembleInformationFilterPanel
from .ensemble_smoother_panel import EnsembleSmootherPanel
from .evaluate_ensemble_panel import EvaluateEnsemblePanel
from .experiment_config_panel import ExperimentConfigPanel
from .manual_update_panel import ManualUpdatePanel
from .multiple_data_assimilation_panel import MultipleDataAssimilationPanel
from .run_dialog import RunDialog
from .single_test_run_panel import SingleTestRunPanel

if TYPE_CHECKING:
    from ert.config import ErtConfig

EXPERIMENT_IS_MANUAL_UPDATE_MESSAGE = "Execute Selected"


def create_md_table(kv: dict[str, str], output: str) -> str:
    for k, v in kv.items():
        v = v.replace("_", r"\_")
        output += f"| {k} | {v} |\n"
    output += "\n"
    return output


def get_simulation_thread(
    model: Any, rerun_failed_realizations: bool = False, use_ipc_protocol: bool = False
) -> ErtThread:
    evaluator_server_config = EvaluatorServerConfig(use_ipc_protocol=use_ipc_protocol)

    def run() -> None:
        model.api.start_simulations_thread(
            evaluator_server_config=evaluator_server_config,
            rerun_failed_realizations=rerun_failed_realizations,
        )

    return ErtThread(name="ert_gui_simulation_thread", target=run, daemon=True)


class ExperimentPanel(QWidget):
    experiment_type_changed = Signal(ExperimentConfigPanel)
    experiment_started = Signal(RunDialog)

    def __init__(
        self,
        config: ErtConfig,
        notifier: ErtNotifier,
        config_file: str,
    ) -> None:
        QWidget.__init__(self)
        self._notifier = notifier
        self.config = config
        run_path = config.runpath_config.runpath_format_string
        self._config_file = config_file

        self.setObjectName("experiment_panel")
        layout = QVBoxLayout()

        self._experiment_type_combo = QComboBoxWithDescription()
        self._experiment_type_combo.setObjectName("experiment_type")

        self._experiment_type_combo.currentIndexChanged.connect(
            self.toggleExperimentType
        )

        experiment_type_layout = QHBoxLayout()
        experiment_type_layout.setContentsMargins(0, 0, 0, 0)
        experiment_type_layout.addWidget(
            self._experiment_type_combo, 0, Qt.AlignmentFlag.AlignVCenter
        )

        self._simulation_done: bool = True
        self.run_button = QToolButton()
        self.run_button.setObjectName("run_experiment")
        self.run_button.setIcon(QIcon("img:play_circle.svg"))
        self.run_button.setToolTip(EXPERIMENT_IS_MANUAL_UPDATE_MESSAGE)
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.clicked.connect(self.run_experiment)
        self.run_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.run_button.setMinimumWidth(60)
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet(
            """
            QToolButton {
            border-radius: 10px;
            background-color: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #f0f0f0,
                stop:1 #d9d9d9
            );
            border: 1px solid #bfbfbf;
            padding: 5px;
            }
            QToolButton:hover {
                background-color: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #d8d8d8,
                stop:1 #c3c3c3
            );
            }
        """
        )

        experiment_type_layout.addWidget(self.run_button)
        experiment_type_layout.addStretch(1)

        layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(experiment_type_layout)

        self._experiment_stack = QStackedWidget()
        self._experiment_stack.setLineWidth(1)
        self._experiment_stack.setFrameStyle(QFrame.Shape.StyledPanel)

        layout.addWidget(self._experiment_stack)

        self._experiment_widgets: dict[type[RunModel], ExperimentConfigPanel] = (
            OrderedDict()
        )
        self.addExperimentConfigPanel(
            SingleTestRunPanel(run_path, notifier),
            True,
        )

        ensemble_size = config.ensemble_size
        active_realizations = config.active_realizations
        analysis_config = config.analysis_config
        config_num_realization = config.runpath_config.num_realizations
        self.addExperimentConfigPanel(
            EnsembleExperimentPanel(
                analysis_config,
                active_realizations,
                config_num_realization,
                run_path,
                notifier,
            ),
            True,
        )
        self.addExperimentConfigPanel(
            EvaluateEnsemblePanel(ensemble_size, run_path, notifier),
            True,
        )

        experiment_type_valid = any(
            p.update for p in config.ensemble_config.parameter_configs.values()
        ) and bool(config.observations)

        self.addExperimentConfigPanel(
            MultipleDataAssimilationPanel(
                analysis_config,
                run_path,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            EnsembleSmootherPanel(
                analysis_config,
                run_path,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            EnsembleInformationFilterPanel(
                analysis_config,
                run_path,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            ManualUpdatePanel(ensemble_size, run_path, notifier, analysis_config),
            experiment_type_valid,
        )

        self.configuration_summary = SummaryPanel(config)
        layout.addWidget(self.configuration_summary)

        self.setLayout(layout)

    def addExperimentConfigPanel(
        self, panel: ExperimentConfigPanel, mode_enabled: bool
    ) -> None:
        assert isinstance(panel, ExperimentConfigPanel)
        self._experiment_stack.addWidget(panel)
        experiment_type = panel.get_experiment_type()
        self._experiment_widgets[experiment_type] = panel
        self._experiment_type_combo.addDescriptionItem(
            experiment_type.display_name(),
            experiment_type.description(),
            experiment_type.group(),
        )

        if not mode_enabled:
            item_count = self._experiment_type_combo.count() - 1
            model = self._experiment_type_combo.model()
            assert isinstance(model, QStandardItemModel)
            sim_item = model.item(item_count)
            assert sim_item is not None
            sim_item.setEnabled(False)
            sim_item.setToolTip(
                "Both observations and parameters must be defined.\n"
                "There must be parameters to update."
            )
            style = self.style()
            assert style is not None
            sim_item.setIcon(
                style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            )

        panel.simulationConfigurationChanged.connect(self.validationStatusChanged)
        self.experiment_type_changed.connect(panel.experimentTypeChanged)

    @staticmethod
    def getActions() -> list[QAction]:
        return []

    def get_current_experiment_type(self) -> Any:
        experiment_type_display_name = self._experiment_type_combo.currentText()
        return next(
            w
            for w in self._experiment_widgets
            if w.display_name() == experiment_type_display_name
        )

    def get_experiment_arguments(self) -> Any:
        simulation_widget = self._experiment_widgets[self.get_current_experiment_type()]
        return simulation_widget.get_experiment_arguments()

    def run_experiment(self) -> None:
        args = self.get_experiment_arguments()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        event_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        try:
            model = create_model(
                self.config,
                args,
                event_queue,
            )

        except ValueError as e:
            QMessageBox.warning(
                self,
                "ERROR: Failed to create experiment",
                str(e),
                QMessageBox.StandardButton.Ok,
            )
            return

        self._model = model

        QApplication.restoreOverrideCursor()
        if model.check_if_runpath_exists():
            msg_box = QMessageBox(self)
            msg_box.setObjectName("RUN_PATH_WARNING_BOX")

            msg_box.setIcon(QMessageBox.Icon.Warning)

            msg_box.setText("Run experiments")
            msg_box.setInformativeText(
                "ERT is running in an existing runpath.\n\n"
                "Please be aware of the following:\n"
                "- Previously generated results "
                "might be overwritten.\n"
                "- Previously generated files might "
                "be used if not configured correctly.\n"
                f"- {model.get_number_of_existing_runpaths()} out "
                f"of {model.get_number_of_active_realizations()} realizations "
                "are running in existing runpaths.\n"
                "Are you sure you want to continue?"
            )

            delete_runpath_checkbox = QCheckBox()
            delete_runpath_checkbox.setText("Delete run_path")
            msg_box.setCheckBox(delete_runpath_checkbox)

            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)

            msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)

            msg_box_res = msg_box.exec()
            if msg_box_res == QMessageBox.StandardButton.No:
                self._model._storage.close()
                return

            if delete_runpath_checkbox.checkState() == Qt.CheckState.Checked:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    model.rm_run_path()
                except OSError as e:
                    QApplication.restoreOverrideCursor()
                    msg_box = QMessageBox(self)
                    msg_box.setObjectName("RUN_PATH_ERROR_BOX")
                    msg_box.setIcon(QMessageBox.Icon.Warning)
                    msg_box.setText("ERT could not delete the existing runpath")
                    msg_box.setInformativeText(
                        f"{e}\n\nContinue without deleting the runpath?"
                    )
                    msg_box.setStandardButtons(
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    msg_box.setDefaultButton(QMessageBox.StandardButton.No)
                    msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)
                    msg_box_res = msg_box.exec()
                    if msg_box_res == QMessageBox.StandardButton.No:
                        return
                QApplication.restoreOverrideCursor()

        self._dialog = RunDialog(
            f"Experiment - {self._config_file} {find_ert_info()}",
            model.api,
            event_queue,
            self._notifier,
            self.parent(),  # type: ignore
            output_path=self.config.analysis_config.log_path,
            run_path=Path(self.config.runpath_config.runpath_format_string),
            storage_path=self._notifier.storage.path,
        )
        self._dialog.set_queue_system_name(model.queue_system)
        self.experiment_started.emit(self._dialog)
        self._simulation_done = False
        self.run_button.setEnabled(self._simulation_done)

        def start_simulation_thread(rerun_failed_realizations: bool = False) -> None:
            simulation_thread = get_simulation_thread(
                self._model,
                rerun_failed_realizations,
                use_ipc_protocol=self.config.queue_config.queue_system
                == QueueSystem.LOCAL,
            )
            self._dialog.setup_event_monitoring(rerun_failed_realizations)
            simulation_thread.start()
            self._notifier.set_is_simulation_running(True)

        def rerun_failed_realizations() -> None:
            start_simulation_thread(rerun_failed_realizations=True)

        self._dialog.rerun_failed_realizations_experiment.connect(
            rerun_failed_realizations
        )
        start_simulation_thread(rerun_failed_realizations=False)

        def simulation_done_handler() -> None:
            self._simulation_done = True
            self.run_button.setEnabled(self._simulation_done)
            self._notifier.emitErtChange()
            self.toggleExperimentType()

        self._dialog.simulation_done.connect(simulation_done_handler)

    def toggleExperimentType(self) -> None:
        current_model = self.get_current_experiment_type()
        if current_model is not None:
            widget = self._experiment_widgets[self.get_current_experiment_type()]
            self._experiment_stack.setCurrentWidget(widget)
            self.validationStatusChanged()
            self.experiment_type_changed.emit(widget)

    def validationStatusChanged(self) -> None:
        widget = self._experiment_widgets[self.get_current_experiment_type()]
        self.run_button.setEnabled(
            self._simulation_done and widget.isConfigurationValid()
        )
