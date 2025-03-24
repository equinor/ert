from __future__ import annotations

from importlib.resources import files
from signal import SIG_DFL, SIGINT, signal

from PyQt6.QtCore import QDir
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.simulation.run_dialog import RunDialog
from everest.gui.everest_client import EverestClient
from everest.gui.main_window import EverestMainWindow


def run_gui(client: EverestClient, config_file: str) -> None:
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath(
        "img", str(files("ert.gui").joinpath("../../ert/gui/resources/gui/img"))
    )

    app = QApplication(
        ["everest"]
    )  # Early so that QT is initialized before other imports
    app.setWindowIcon(QIcon("img:ert_icon.svg"))

    # Add arg parser if we are to pass more opts

    main_window = EverestMainWindow(window_title=f"Everest - {config_file}")
    main_window.adjustSize()
    main_window.show()
    main_window.activateWindow()
    main_window.raise_()

    run_model_api = client.create_run_model_api()
    event_queue, event_monitor_thread = client.setup_event_queue_from_ws_endpoint(
        refresh_interval=0.02, open_timeout=40, websocket_recv_timeout=1.0
    )

    run_dialog = RunDialog(
        config_file=config_file,
        run_model_api=run_model_api,
        event_queue=event_queue,
        is_everest=True,
        notifier=ErtNotifier(),
    )

    main_window.central_layout.addWidget(run_dialog)

    event_monitor_thread.start()
    run_dialog.setup_event_monitoring()

    app.exec()
