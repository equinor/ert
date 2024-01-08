import json
import logging
import subprocess
import sys
import tempfile

from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from qtpy.QtWidgets import QMainWindow

from ert.shared.port_handler import find_available_port

logger = logging.getLogger(__name__)
import os


def symlink(target, link_name, overwrite=False):
    """
    (https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python/55742015#55742015)
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.
    """

    if not overwrite:
        os.symlink(target, link_name)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary filename
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Pre-empt os.replace on a directory with a nicer message
        if not os.path.islink(link_name) and os.path.isdir(link_name):
            raise IsADirectoryError(
                f"Cannot symlink over existing directory: '{link_name}'"
            )
        os.replace(temp_link_name, link_name)
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise


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
        ensembles_folder = os.path.join(ens_path, "ensembles")

        ensemble_keys = [
            x
            for x in os.listdir(ensembles_folder)
            if os.path.isdir(os.path.join(ensembles_folder, x))
        ]

        ensembles_list = []
        for ens_key in ensemble_keys:
            with open(
                f"{ensembles_folder}/{ens_key}/index.json", encoding="utf-8"
            ) as f:
                loaded = json.load(f)
                ens_index = loaded["iteration"]
                ensembles_list.append((ens_key, ens_index))

        ensembles_list.sort(key=lambda x: x[1])
        sorted_ens_key_list = [x[0] for x in ensembles_list]

        ensembles_index_data = {"ensembles": sorted_ens_key_list}

        with open(
            os.path.join(static_html_path, "index.json"),
            "w+",
            encoding="utf-8",
        ) as f:
            json.dump(ensembles_index_data, f)

        symlink(
            ensembles_folder,
            os.path.join(static_html_path, "ensembles"),
            overwrite=True,
        )

        python_fileserver_executable_path = os.path.join(
            os.path.dirname(__file__),
            "web_plot_static_fileserver.py",
        )

        sock.close()

        subprocess.Popen(
            [
                sys.executable,
                python_fileserver_executable_path,
                static_html_path,
                hostname,
                str(port),
            ]
        )

        url_for_browser = QUrl(f"http://{hostname}:{port}?serverURL=.")
        print(f"Serving static files to browser @ {url_for_browser}")
        self.browser.setUrl(url_for_browser)
        self.showMaximized()
