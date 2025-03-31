from __future__ import annotations

from queue import SimpleQueue

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
)

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.simulation.run_dialog import RunDialog
from ert.plugins import ErtPluginManager
from ert.run_models.base_run_model import BaseRunModelAPI
from ert.run_models.event import StatusEvents


class EverestMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(
        self,
        run_model_api: BaseRunModelAPI,
        event_queue: SimpleQueue[StatusEvents],
    ):
        QMainWindow.__init__(self)
        self.run_model_api = run_model_api
        self.event_queue = event_queue

        self.setWindowTitle(f"Everest - {run_model_api.config_file}")
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
        run_dialog = RunDialog(
            config_file=self.run_model_api.config_file,
            run_model_api=self.run_model_api,
            event_queue=self.event_queue,
            is_everest=True,
            notifier=ErtNotifier(),
        )

        self.central_layout.addWidget(run_dialog)
        self._run_dialog = run_dialog
        self._run_dialog.setup_event_monitoring()
