from __future__ import annotations

import ssl
from importlib.resources import files
from signal import SIG_DFL, SIGINT, signal

from PyQt6.QtCore import QDir
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from everest.config.everest_config import EverestConfig
from everest.config.server_config import ServerConfig
from everest.gui.everest_client import EverestClient
from everest.gui.main_window import EverestMainWindow


def run_gui(config: EverestConfig) -> None:
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

    server_context = ServerConfig.get_server_context(config.output_dir)
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

    assert config.simulator is not None
    assert config.simulator.queue_system is not None
    sim_dir = config.simulation_dir
    assert sim_dir is not None

    run_model_api = client.create_run_model_api(
        config.simulator.queue_system.name,
        runpath_format_string=sim_dir,
        config_file=config.config_file,
    )
    event_queue, event_monitor_thread = client.setup_event_queue_from_ws_endpoint(
        refresh_interval=0.02, open_timeout=40, websocket_recv_timeout=1.0
    )

    window = EverestMainWindow(run_model_api, event_queue)
    event_monitor_thread.start()
    window.run()
    window.adjustSize()
    window.show()
    window.activateWindow()
    window.raise_()
    app.exec()
