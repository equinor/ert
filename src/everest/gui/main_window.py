from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
)

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.experiments import RunDialog
from ert.plugins import ErtPluginManager
from ert.run_models.run_model import RunModelAPI
from ert.services.ert_client import ErtClient
from everest.config import ServerConfig
from everest.detached import wait_for_server


class EverestMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(
        self,
        output_dir: str,
    ) -> None:
        QMainWindow.__init__(self)
        self.output_dir = output_dir

        self.plugin_manager = ErtPluginManager()
        self.central_widget = QFrame(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.central_widget.setLayout(self.central_layout)

        self._run_dialog: RunDialog | None = None

        self.central_widget.setMinimumWidth(1500)
        self.central_widget.setMinimumHeight(800)
        self.setCentralWidget(self.central_widget)

    def run(self) -> None:
        client = ErtClient.get_client(
            Path(ServerConfig.get_session_dir(self.output_dir))
        )
        wait_for_server(client, 60)

        experiment_id = client.experiment_ids()[-1]
        config = client.experiment_config(experiment_id)

        config_filename = Path(config["config_path"]).name
        self.setWindowTitle(f"EVEREST - {config_filename}")

        def start_fn(
            evaluator_server_config: EvaluatorServerConfig,
            *,
            rerun_failed_realizations: bool = False,
        ) -> None:
            pass

        run_model_api = RunModelAPI(
            experiment_name=config_filename,
            supports_rerunning_failed_realizations=False,
            start_simulations_thread=start_fn,
            cancel=client.stop_experiment_server,  # type: ignore
            has_failed_realizations=lambda: False,
        )
        event_queue, event_monitor_thread = client.setup_event_queue_from_ws_endpoint(
            experiment_id=experiment_id,
            refresh_interval=0.02,
            open_timeout=40,
            websocket_recv_timeout=1.0,
        )

        run_dialog = RunDialog(
            title=config_filename,
            run_model_api=run_model_api,
            event_queue=event_queue,
            notifier=ErtNotifier(),
            run_path=Path(config["run_path"]),
            storage_path=Path(config["storage_path"]),
        )

        self.central_layout.addWidget(run_dialog)
        self._run_dialog = run_dialog
        event_monitor_thread.start()
        self._run_dialog.setup_event_monitoring()
