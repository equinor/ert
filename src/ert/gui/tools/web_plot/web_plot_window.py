import logging
import subprocess
import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from qtpy.QtWidgets import QMainWindow
from textual._win_sleep import sleep

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

        static_html_path = os.path.join(
            os.path.dirname(__file__),
            "web_plot_html_assets",
        )

        # Infer root location for file server

        python_fileserver_executable_path = os.path.join(
            os.path.dirname(__file__),
            "web_plot_server.py",
        )

        sock.close()

        url_for_browser = QUrl(f"http://{hostname}:{port}")
        print(f"Serving static files to browser @ {url_for_browser}")
        subprocess.Popen(
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

        sleep(1500)

        self.browser.setUrl(url_for_browser)
        self.showMaximized()
