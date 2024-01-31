import logging
import socket
import subprocess
import sys
import time
from qtpy.QtCore import QUrl
from qtpy.QtWebEngineWidgets import QWebEngineView
from qtpy.QtWidgets import QMainWindow

from ert.shared.port_handler import find_available_port

logger = logging.getLogger(__name__)
import os


class WebPlotWindow(QMainWindow):
    def __init__(self, parent, ens_path):
        QMainWindow.__init__(self, parent)

        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)

        # Need to:
        hostname, port, sock = find_available_port()

        print(f"Found hostname: {hostname}:{port}")

        static_html_path = os.path.join(
            os.path.dirname(__file__),
            "web_plot_html_assets",
        )

        # Infer root location for file server

        python_fileserver_executable_path = os.path.join(
            os.path.dirname(__file__),
            "web_plot_server.py",
        )

        try:
            sock.close()
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        url_for_browser = QUrl(f"http://{hostname}:{port}/index.html?serverURL=.")
        print(f"Serving static files to browser @ {url_for_browser}")
        self.server_process = subprocess.Popen(
            [
                sys.executable,
                python_fileserver_executable_path,
                ens_path,
                static_html_path,
                hostname,
                str(port),
                "auto",
            ],
        )

        time.sleep(1)

        self.browser.setUrl(url_for_browser)
        self.showMaximized()

    def closeEvent(self, a0: any) -> None:
        self.server_process.terminate()
