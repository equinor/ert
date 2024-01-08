import json
import logging
import subprocess
import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from qtpy.QtWidgets import (
    QMainWindow,
)

from ert.shared.port_handler import find_available_port

CROSS_CASE_STATISTICS = "Cross case statistics"
DISTRIBUTION = "Distribution"
GAUSSIAN_KDE = "Gaussian KDE"
ENSEMBLE = "Ensemble"
HISTOGRAM = "Histogram"
STATISTICS = "Statistics"

logger = logging.getLogger(__name__)
import os


class WebPlotWindow(QMainWindow):
    def __init__(self, parent, ens_path):
        QMainWindow.__init__(self, parent)

        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)

        # Need to:
        hostname, port, sock = find_available_port()

        # Infer root location for file server
        static_fileserver_root = os.path.join(ens_path, "ensembles")

        # Create the overview index.json
        ensembles = [
            x
            for x in os.listdir(static_fileserver_root)
            if os.path.isdir(os.path.join(static_fileserver_root, x))
        ]

        # 1. run static file server in storage, to serve all the netcdf files
        # 2. On this static file server, also serve index.html and all that
        ensembles_list = []
        for ens_key in ensembles:
            with open(
                f"{static_fileserver_root}/{ens_key}/index.json", encoding="utf-8"
            ) as f:
                loaded = json.load(f)
                ens_index = loaded["iteration"]
                ensembles_list.append((ens_key, ens_index))

        ensembles_list.sort(key=lambda x: x[1])
        sorted_ens_key_list = [x[0] for x in ensembles_list]

        ensembles_data = {"ensembles": sorted_ens_key_list}

        with open(f"{static_fileserver_root}/index.json", "w+", encoding="utf-8") as f:
            json.dump(ensembles_data, f)

        python_fileserver_executable_path = os.path.join(
            os.path.dirname(__file__),
            "web_plot_static_fileserver.py",
        )

        sock.close()
        # Now set up the server w/ cors
        subprocess.Popen(
            [
                sys.executable,
                python_fileserver_executable_path,
                static_fileserver_root,
                hostname,
                str(port),
            ]
        )

        # Now place the index.html file here so we can access it from that same url
        # and pass the root as a http param rather than having it hardcoded

        self.browser.setUrl(QUrl(f"{hostname}:{port}"))
        self.showMaximized()
