from __future__ import annotations

import os
import platform
from collections import OrderedDict
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any, Dict, List, Type

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon, QStandardItemModel
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
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

import ert.shared
from ert.gui.ertnotifier import ErtNotifier
from ert.run_models import BaseRunModel, SingleTestRun, StatusEvents, create_model
from ert.shared.status.utils import (
    byte_with_unit,
    format_running_time,
    get_ert_memory_usage,
)

from .combobox_with_description import QComboBoxWithDescription
from .ensemble_experiment_panel import EnsembleExperimentPanel
from .ensemble_smoother_panel import EnsembleSmootherPanel
from .evaluate_ensemble_panel import EvaluateEnsemblePanel
from .experiment_config_panel import ExperimentConfigPanel
from .iterated_ensemble_smoother_panel import IteratedEnsembleSmootherPanel
from .manual_update_panel import ManualUpdatePanel
from .multiple_data_assimilation_panel import MultipleDataAssimilationPanel
from .run_dialog import RunDialog
from .single_test_run_panel import SingleTestRunPanel

if TYPE_CHECKING:
    from ert.config import ErtConfig

EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE = "Run Experiment"
EXPERIMENT_IS_RUNNING_BUTTON_MESSAGE = "Experiment running..."


def create_md_table(kv: Dict[str, str], output: str) -> str:
    for k, v in kv.items():
        v = v.replace("_", r"\_")
        output += f"| {k} | {v} |\n"
    output += "\n"
    return output


class ExperimentPanel(QWidget):
    experiment_type_changed = Signal(ExperimentConfigPanel)

    def __init__(
        self,
        config: ErtConfig,
        notifier: ErtNotifier,
        config_file: str,
        ensemble_size: int,
    ):
        QWidget.__init__(self)
        self._notifier = notifier
        self.config = config
        run_path = config.model_config.runpath_format_string
        self._config_file = config_file

        self.setObjectName("experiment_panel")
        layout = QVBoxLayout()

        self._experiment_type_combo = QComboBoxWithDescription()
        self._experiment_type_combo.setObjectName("experiment_type")

        self._experiment_type_combo.currentIndexChanged.connect(
            self.toggleExperimentType
        )

        experiment_type_layout = QHBoxLayout()
        experiment_type_layout.addSpacing(10)
        experiment_type_layout.addWidget(
            QLabel("Experiment type:"), 0, Qt.AlignmentFlag.AlignVCenter
        )
        experiment_type_layout.addWidget(
            self._experiment_type_combo, 0, Qt.AlignmentFlag.AlignVCenter
        )

        experiment_type_layout.addSpacing(20)

        self.run_button = QToolButton()
        self.run_button.setObjectName("run_experiment")
        self.run_button.setText(EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE)
        self.run_button.setIcon(QIcon("img:play_circle.svg"))
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.clicked.connect(self.run_experiment)
        self.run_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        experiment_type_layout.addWidget(self.run_button)
        experiment_type_layout.addStretch(1)

        layout.addSpacing(5)
        layout.addLayout(experiment_type_layout)
        layout.addSpacing(10)

        self._experiment_stack = QStackedWidget()
        self._experiment_stack.setLineWidth(1)
        self._experiment_stack.setFrameStyle(QFrame.StyledPanel)

        layout.addWidget(self._experiment_stack)

        self._experiment_widgets: dict[Type[BaseRunModel], QWidget] = OrderedDict()
        self.addExperimentConfigPanel(
            SingleTestRunPanel(run_path, notifier),
            True,
        )
        self.addExperimentConfigPanel(
            EnsembleExperimentPanel(ensemble_size, run_path, notifier),
            True,
        )
        self.addExperimentConfigPanel(
            EvaluateEnsemblePanel(ensemble_size, run_path, notifier),
            True,
        )

        experiment_type_valid = bool(
            config.ensemble_config.parameter_configs and config.observations
        )
        analysis_config = config.analysis_config
        self.addExperimentConfigPanel(
            EnsembleSmootherPanel(analysis_config, run_path, notifier, ensemble_size),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            ManualUpdatePanel(ensemble_size, run_path, notifier, analysis_config),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            MultipleDataAssimilationPanel(
                analysis_config, run_path, notifier, ensemble_size
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            IteratedEnsembleSmootherPanel(
                analysis_config, run_path, notifier, ensemble_size
            ),
            experiment_type_valid,
        )

        self.setLayout(layout)

    def addExperimentConfigPanel(
        self, panel: ExperimentConfigPanel, mode_enabled: bool
    ) -> None:
        assert isinstance(panel, ExperimentConfigPanel)
        self._experiment_stack.addWidget(panel)
        experiment_type = panel.get_experiment_type()
        self._experiment_widgets[experiment_type] = panel
        self._experiment_type_combo.addDescriptionItem(
            experiment_type.name(), experiment_type.description()
        )

        if not mode_enabled:
            item_count = self._experiment_type_combo.count() - 1
            model = self._experiment_type_combo.model()
            assert isinstance(model, QStandardItemModel)
            sim_item = model.item(item_count)
            assert sim_item is not None
            sim_item.setEnabled(False)
            sim_item.setToolTip("Both observations and parameters must be defined")
            style = self.style()
            assert style is not None
            sim_item.setIcon(
                style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            )

        panel.simulationConfigurationChanged.connect(self.validationStatusChanged)
        self.experiment_type_changed.connect(panel.experimentTypeChanged)

    @staticmethod
    def getActions() -> List[QAction]:
        return []

    def get_current_experiment_type(self) -> Any:
        experiment_type_name = self._experiment_type_combo.currentText()
        return next(
            w for w in self._experiment_widgets if w.name() == experiment_type_name
        )

    def get_experiment_arguments(self) -> Any:
        simulation_widget = self._experiment_widgets[self.get_current_experiment_type()]
        args = simulation_widget.get_experiment_arguments()
        return args

    def getExperimentName(self) -> str:
        """Get the experiment name as provided by the user. Defaults to run mode if not set."""
        return self.get_experiment_arguments().experiment_name

    def run_experiment(self) -> None:
        args = self.get_experiment_arguments()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        event_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        try:
            model = create_model(
                self.config,
                self._notifier.storage,
                args,
                event_queue,
            )

        except ValueError as e:
            QMessageBox.warning(
                self, "ERROR: Failed to create experiment", (str(e)), QMessageBox.Ok
            )
            return

        self._model = model

        QApplication.restoreOverrideCursor()
        if model.check_if_runpath_exists():
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
                    f"- {model.get_number_of_existing_runpaths()} out of {model.get_number_of_active_realizations()} realizations "
                    "are running in existing runpaths.\n"
                    "Are you sure you want to continue?"
                )
            )

            delete_runpath_checkbox = QCheckBox()
            delete_runpath_checkbox.setText("Delete run_path")
            msg_box.setCheckBox(delete_runpath_checkbox)

            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)

            msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)

            msg_box_res = msg_box.exec()
            if msg_box_res == QMessageBox.No:
                return

            if delete_runpath_checkbox.checkState() == Qt.CheckState.Checked:
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
                    msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)
                    msg_box_res = msg_box.exec()
                    if msg_box_res == QMessageBox.No:
                        return
                QApplication.restoreOverrideCursor()

        dialog = RunDialog(
            self._config_file,
            model,
            event_queue,
            self._notifier,
            self.parent(),  # type: ignore
            output_path=self.config.analysis_config.log_path,
        )
        dialog.produce_clipboard_debug_info.connect(self.populate_clipboard_debug_info)

        self.run_button.setEnabled(False)
        self.run_button.setText(EXPERIMENT_IS_RUNNING_BUTTON_MESSAGE)
        dialog.run_experiment()
        dialog.show()

        def exit_handler() -> None:
            self.run_button.setText(EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE)
            self.run_button.setEnabled(True)
            self.toggleExperimentType()
            self._notifier.emitErtChange()

        dialog.finished.connect(exit_handler)

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
            self.run_button.text() == EXPERIMENT_READY_TO_RUN_BUTTON_MESSAGE
            and widget.isConfigurationValid()
        )

    def populate_clipboard_debug_info(self) -> None:
        kv = {"**Platform**": "", ":-----": ":-----"}
        kv["Date"] = datetime.now().isoformat(" ", "seconds")
        kv["OS"] = (
            platform.system() + " " + platform.release() + " " + platform.machine()
        )
        kv["Hostname"] = ert.shared.get_machine_name()
        kv["Komodo release"] = os.environ.get("KOMODO_RELEASE", "")
        kv["Python version"] = platform.python_version()

        kv["**Ensemble**"] = ""
        queue_system = self.config.queue_config.queue_system
        kv["Queue"] = queue_system.name.capitalize()
        kv["Simulation mode"] = self.get_current_experiment_type().name()
        kv["Config file"] = str(Path(self._config_file).absolute())
        kv["Storage path"] = self.config.ens_path
        kv["Run path"] = str(self.config.model_config.runpath_format_string)
        kv["Ensemble size"] = str(self.config.model_config.num_realizations)

        if self.config.queue_config.realization_memory > 0:
            kv["Realization memory"] = byte_with_unit(
                self.config.queue_config.realization_memory
            )

        if self.config.queue_config.max_submit > 1:
            kv["Max submit"] = str(self.config.queue_config.max_submit)
        if self.config.queue_config.stop_long_running:
            kv["Stop long running"] = str(self.config.queue_config.stop_long_running)

        queue_opts = self.config.queue_config.queue_options

        if isinstance(self.get_current_experiment_type(), SingleTestRun):
            queue_opts = self.config.queue_config.queue_options_test_run

        for field in fields(queue_opts):
            field_value = getattr(queue_opts, field.name)
            if field_value is not None:
                kv[field.name.replace("_", " ").capitalize()] = str(field_value)

        kv["**Status**"] = ""
        kv["Running time"] = (
            format_running_time(self._model.get_runtime()).split(":")[1].strip()
        )
        kv["Ert max memory"] = byte_with_unit(get_ert_memory_usage())
        kv["Forward model max memory"] = byte_with_unit(
            self._model.get_memory_consumption()
        )

        for status, count in self._model.get_current_status().items():
            kv[status] = str(count)

        output = create_md_table(kv, "")
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(output)
