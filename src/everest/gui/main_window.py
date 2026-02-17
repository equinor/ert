from __future__ import annotations

import ssl
from pathlib import Path

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
)

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.experiments import RunDialog
from ert.plugins import ErtPluginManager
from ert.services import create_ertserver_client
from everest.config import ServerConfig
from everest.detached import wait_for_server
from everest.gui.everest_client import EverestClient


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
        storage_client = create_ertserver_client(
            Path(ServerConfig.get_session_dir(self.output_dir))
        )
        wait_for_server(storage_client, 60)

        server_context = ServerConfig.get_server_context_from_conn_info(
            storage_client.conn_info
        )
        url, cert, auth = server_context

        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations(cafile=cert)
        username, password = auth

        client = EverestClient(
            url=url,
            cert_file=cert,
            username=username,
            password=password,
            ssl_context=ssl_context,
        )

        config = client.config
        title = Path(config["config_path"]).name
        self.setWindowTitle(f"Everest - {title}")

        run_model_api = client.create_run_model_api()
        event_queue, event_monitor_thread = client.setup_event_queue_from_ws_endpoint(
            refresh_interval=0.02, open_timeout=40, websocket_recv_timeout=1.0
        )

        run_dialog = RunDialog(
            title=title,
            run_model_api=run_model_api,
            event_queue=event_queue,
            is_everest=True,
            notifier=ErtNotifier(),
            run_path=Path(config["run_path"]),
            storage_path=Path(config["storage_path"]),
        )

        self.central_layout.addWidget(run_dialog)
        self._run_dialog = run_dialog
        event_monitor_thread.start()
        self._run_dialog.setup_event_monitoring()
