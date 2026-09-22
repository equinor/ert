from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
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
from ert.config import parameter_config
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.gui.detect_mode import is_dark_mode
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.gui.icon_utils import load_icon
from ert.gui.summarypanel import SummaryPanel
from ert.run_models import RunModel
from ert.run_models.run_model import RunModelAPI
from ert.services.ert_client import ErtClient

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

EXPERIMENT_IS_MANUAL_UPDATE_MESSAGE = "Execute selected"

logger = logging.getLogger(__name__)


def create_md_table(kv: dict[str, str], output: str) -> str:
    for key, unescaped_value in kv.items():
        value = unescaped_value.replace("_", r"\_")
        output += f"| {key} | {value} |\n"
    output += "\n"
    return output


def get_simulation_thread(
    model: Any,
    *,
    rerun_failed_realizations: bool = False,
    use_ipc_protocol: bool = False,
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
        runpath = config.runpath_config.runpath_format_string
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
                runpath,
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
                runpath,
                notifier,
            ),
            True,
        )
        self.addExperimentConfigPanel(
            EvaluateEnsemblePanel(runpath, notifier),
            True,
        )

        has_observation_declarations = bool(config.observation_declarations)
        has_updatable_parameters = parameter_config.has_updatable_parameters(
            config.ensemble_config.parameter_configs.values()
        )

        self.addExperimentConfigPanel(
            MultipleDataAssimilationPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                runpath,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            has_observation_declarations,
        )
        self.addExperimentConfigPanel(
            EnsembleSmootherPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                runpath,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            has_observation_declarations and has_updatable_parameters,
        )
        self.addExperimentConfigPanel(
            EnsembleInformationFilterPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                runpath,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            has_observation_declarations and has_updatable_parameters,
        )
        self.addExperimentConfigPanel(
            ManualUpdatePanel(runpath, notifier, analysis_config),
            has_observation_declarations and has_updatable_parameters,
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
        item_index = self._experiment_type_combo.addDescriptionItem(
            experiment_type.display_name(),
            experiment_type.description(),
            experiment_type.group(),
        )

        if not mode_enabled:
            model = self._experiment_type_combo.model()
            assert isinstance(model, QStandardItemModel)
            sim_item = model.item(item_index)
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

    @property
    def _ert_client(self) -> ErtClient:
        return ErtClient.get_client(Path(self.config.ens_path))

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
        delete = False
        if self._ert_client.runpath_exists(self.config, args):
            model_data = self._ert_client.get_runmodel_data(self.config, args)
            msg_box = QMessageBox(self)
            msg_box.setObjectName("RUNPATH_WARNING_BOX")

            msg_box.setIcon(QMessageBox.Icon.Warning)

            msg_box.setText("Run experiments")
            msg_box.setInformativeText(
                "ERT is running in an existing runpath.\n\n"
                "Please be aware of the following:\n"
                "- Previously generated results "
                "might be overwritten.\n"
                "- Previously generated files might "
                "be used if not configured correctly.\n"
                f"- {model_data['number_of_existing_runpaths']} out "
                f"of {model_data['number_of_active_realizations']} realizations "
                "are running in existing runpaths.\n"
                "Are you sure you want to continue?"
            )

            delete_runpath_checkbox = QCheckBox()
            delete_runpath_checkbox.setText("Delete runpath")
            msg_box.setCheckBox(delete_runpath_checkbox)

            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)

            msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)

            msg_box_res = msg_box.exec()
            if msg_box_res == QMessageBox.StandardButton.No:
                return

            if delete_runpath_checkbox.checkState() == Qt.CheckState.Checked:
                delete = True

        if delete:
            progress_dialog = QDialog(self)
            progress_dialog.setObjectName("RUNPATH_PROGRESS_DIALOG")
            progress_dialog.setWindowTitle("Deleting runpaths")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_layout = QVBoxLayout(progress_dialog)
            progress_layout.setContentsMargins(0, 0, 0, 0)

            progress_dialog.resize(420, 120)
            progress_dialog.show()
            QApplication.processEvents()

            try:
                successfully_removed = self._ert_client.runpath_delete(
                    self.config, args
                )
            except Exception as e:
                logger.error("Failed to delete runpath: %s", e)
                successfully_removed = False
            if not successfully_removed:
                progress_dialog.close()
                progress_dialog.deleteLater()
                msg_box = QMessageBox(self)
                msg_box.setObjectName("RUNPATH_ERROR_BOX")
                msg_box.setIcon(QMessageBox.Icon.Warning)
                msg_box.setText("ERT could not delete the existing runpath")
                msg_box.setInformativeText(
                    "Failed to delete the runpath.\n"
                    "Continue without deleting the runpath?"
                )
                msg_box.setStandardButtons(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                msg_box.setDefaultButton(QMessageBox.StandardButton.No)
                msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)
                msg_box_res = msg_box.exec()
                if msg_box_res == QMessageBox.StandardButton.No:
                    return
            else:
                progress_dialog.close()
                progress_dialog.deleteLater()

        # Ready to start experiment:
        experiment_id = self._ert_client.start_experiment_ert(self.config, args)

        # Setup websocket for update
        event_queue, thread = self._ert_client.setup_event_queue_from_ws_endpoint(
            experiment_id
        )

        # Dummy func
        def dummy(
            evaluator_server_config: EvaluatorServerConfig,
            *,
            rerun_failed_realizations: bool = False,
        ) -> None: ...

        # Dummy RunModelAPI for the RunDialog similar to how everset gui does it
        dummy_run_model_api = RunModelAPI(
            experiment_name=self._config_file,
            supports_rerunning_failed_realizations=False,
            start_simulations_thread=dummy,
            cancel=self._ert_client.stop_server,  # type: ignore
            has_failed_realizations=lambda: False,
        )
        self._dialog = RunDialog(
            f"Experiment - {self._config_file} {find_ert_info()}",
            dummy_run_model_api,
            event_queue,
            self._notifier,
            self.parent(),  # type: ignore
            output_path=self.config.analysis_config.log_path,
            runpath=Path(self.config.runpath_config.runpath_format_string),
            storage_path=self._notifier.storage.path,
        )
        self._dialog.queue_system.setText(
            f"Queue system:\n{self.config.queue_config.queue_system.formatted_name}"
        )
        self.experiment_started.emit(self._dialog)
        self._experiment_done = False
        self.run_button.setEnabled(self._experiment_done)
        self._dialog.setup_event_monitoring()
        thread.start()
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
