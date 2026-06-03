from __future__ import annotations

import json
import ssl
from collections import OrderedDict
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any

import requests
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction, QStandardItemModel
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

from ert.gui.detect_mode import is_dark_mode
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.gui.icon_utils import load_icon
from ert.gui.summarypanel import SummaryPanel
from ert.run_models import RunModel, RunModelAPI, StatusEvents
from ert.run_models.model_factory import build_run_model_config

from .combobox_with_description import QComboBoxWithDescription
from .ensemble_experiment_panel import EnsembleExperimentPanel
from .ensemble_information_filter_panel import EnsembleInformationFilterPanel
from .ensemble_smoother_panel import EnsembleSmootherPanel
from .evaluate_ensemble_panel import EvaluateEnsemblePanel
from .experiment_client import ExperimentClient
from .experiment_config_panel import ExperimentConfigPanel
from .manual_update_panel import ManualUpdatePanel
from .multiple_data_assimilation_panel import MultipleDataAssimilationPanel
from .run_dialog import RunDialog
from .single_test_run_panel import SingleTestRunPanel

if TYPE_CHECKING:
    from ert.config import ErtConfig

EXPERIMENT_IS_MANUAL_UPDATE_MESSAGE = "Execute selected"


def create_md_table(kv: dict[str, str], output: str) -> str:
    for key, unescaped_value in kv.items():
        value = unescaped_value.replace("_", r"\_")
        output += f"| {key} | {value} |\n"
    output += "\n"
    return output


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

        self._experiment_done: bool = True
        self._client: ExperimentClient | None = None
        self.run_button = QToolButton()
        self.run_button.setObjectName("run_experiment")
        self.run_button.setIcon(load_icon("play_circle.svg"))
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
                stop:0 #484848,
                stop:1 #323232
            );
            border: 1px solid #1e1e1e;
            padding: 5px;
            }
            QToolButton:hover {
                background-color: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #575757,
                stop:1 #424242
            );
            }
        """
            if is_dark_mode()
            else """
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
        analysis_config = config.analysis_config
        self.addExperimentConfigPanel(
            SingleTestRunPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                run_path,
                notifier,
            ),
            True,
        )

        active_realizations = config.active_realizations
        config_num_realization = config.runpath_config.num_realizations
        self.addExperimentConfigPanel(
            EnsembleExperimentPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                active_realizations,
                config_num_realization,
                run_path,
                notifier,
            ),
            True,
        )
        self.addExperimentConfigPanel(
            EvaluateEnsemblePanel(run_path, notifier),
            True,
        )

        experiment_type_valid = any(
            p.update_strategy is not None
            for p in config.ensemble_config.parameter_configs.values()
        ) and bool(config.observation_declarations)

        self.addExperimentConfigPanel(
            MultipleDataAssimilationPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
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
                config.ensemble_config.parameter_configuration,
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
                config.ensemble_config.parameter_configuration,
                run_path,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            ManualUpdatePanel(run_path, notifier, analysis_config),
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

        panel.experiment_configuration_changed.connect(self.validationStatusChanged)
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

    def _create_experiment_client(self) -> ExperimentClient:
        conn_info_path = Path(self.config.ens_path) / "storage_server.json"
        conn_info: dict[str, Any] = json.loads(
            conn_info_path.read_text(encoding="utf-8")
        )
        cert_file = conn_info["cert"]
        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations(cafile=cert_file)

        errors: list[str] = []
        for url in conn_info["urls"]:
            try:
                requests.get(
                    f"{url}/healthcheck",
                    auth=("__token__", conn_info["authtoken"]),
                    verify=cert_file,
                    timeout=5,
                ).raise_for_status()
                return ExperimentClient(
                    url=f"{url}/experiment_server",
                    cert_file=cert_file,
                    username="__token__",
                    password=conn_info["authtoken"],
                    ssl_context=ssl_context,
                )
            except Exception as e:
                errors.append(f"{url}: {e}")

        raise RuntimeError(
            f"Cannot connect to storage server at {self.config.ens_path}: "
            + "; ".join(errors)
        )

    def run_experiment(self) -> None:
        args = self.get_experiment_arguments()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            run_model_config = build_run_model_config(self.config, args)
            config_json = run_model_config.model_dump(mode="json")
            client = self._create_experiment_client()
            self._client = client
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self,
                "ERROR: Failed to prepare experiment",
                str(e),
                QMessageBox.StandardButton.Ok,
            )
            return

        QApplication.restoreOverrideCursor()

        try:
            runpath_check = client.check_runpath(config_json)
        except Exception as e:
            QMessageBox.warning(
                self,
                "ERROR: run_path check failed",
                str(e),
                QMessageBox.StandardButton.Ok,
            )
            return

        if runpath_check.get("runpath_exists"):
            num_existing = runpath_check.get("num_existing", 0)
            num_active = runpath_check.get("num_active", 0)
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
                f"- {num_existing} out of {num_active} realizations "
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

            if msg_box.exec() == QMessageBox.StandardButton.No:
                return

            if delete_runpath_checkbox.checkState() == Qt.CheckState.Checked:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    client.delete_runpaths(config_json)
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
                    if msg_box.exec() == QMessageBox.StandardButton.No:
                        return
                else:
                    QApplication.restoreOverrideCursor()

        self.configuration_summary.log_summary(
            args.mode, runpath_check.get("num_active", 0)
        )

        try:
            _run_id, supports_rerunning = client.start_experiment(config_json)
        except Exception as e:
            QMessageBox.warning(
                self,
                "ERROR: Failed to start experiment",
                str(e),
                QMessageBox.StandardButton.Ok,
            )
            return

        event_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        _, monitor_thread = client.setup_event_queue_from_ws_endpoint(
            event_queue=event_queue
        )

        experiment_name = self.get_current_experiment_type().display_name()

        def start_fn(
            evaluator_server_config: Any = None,
            rerun_failed_realizations: bool = False,
        ) -> None:
            pass

        run_model_api = RunModelAPI(
            experiment_name=experiment_name,
            supports_rerunning_failed_realizations=supports_rerunning,
            start_simulations_thread=start_fn,
            cancel=client.stop,
            has_failed_realizations=client.has_failed_realizations,
        )

        self._dialog = RunDialog(
            f"Experiment - {self._config_file} {find_ert_info()}",
            run_model_api,
            event_queue,
            self._notifier,
            self.parent(),  # type: ignore
            output_path=self.config.analysis_config.log_path,
            run_path=Path(self.config.runpath_config.runpath_format_string),
            storage_path=self._notifier.storage.path,
        )
        self._dialog.queue_system.setText(
            f"Queue system:\n{self.config.queue_config.queue_system.formatted_name}"
        )
        self.experiment_started.emit(self._dialog)
        self._experiment_done = False
        self.run_button.setEnabled(self._experiment_done)

        def do_rerun_failed() -> None:
            try:
                _new_run_id, _new_supports_rerunning = client.rerun_failed()
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "ERROR: Failed to rerun",
                    str(e),
                    QMessageBox.StandardButton.Ok,
                )
                return
            _, rerun_thread = client.setup_event_queue_from_ws_endpoint(
                event_queue=event_queue
            )
            self._dialog.setup_event_monitoring(rerun_failed_realizations=True)
            rerun_thread.start()
            self._notifier.set_is_experiment_running(True)

        self._dialog.rerun_failed_realizations_experiment.connect(do_rerun_failed)

        self._dialog.setup_event_monitoring(rerun_failed_realizations=False)
        monitor_thread.start()
        self._notifier.set_is_experiment_running(True)

        def simulation_done_handler() -> None:
            self._experiment_done = True
            self.run_button.setEnabled(self._experiment_done)
            self._notifier.emitErtChange()
            self.toggleExperimentType()

        self._dialog.experiment_done.connect(simulation_done_handler)

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
            self._experiment_done and widget.isConfigurationValid()
        )
