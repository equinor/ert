from __future__ import annotations

import functools
import webbrowser

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
)

from ert.gui.about_dialog import AboutDialog
from ert.gui.simulation.run_dialog import RunDialog
from ert.plugins import ErtPluginManager
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


class EverestMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(
        self,
        config_file: str,
    ):
        QMainWindow.__init__(self)
        self.config_file = config_file

        self.setWindowTitle(f"Everest - {config_file}")
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

        self.__add_help_menu()

    def post_init(self) -> None:
        ever_config = EverestConfig.load_file(self.config_file)
        run_model = EverestRunModel.create(ever_config)
        run_dialog = RunDialog(self.config_file, run_model_api=run_model.api)
        self.central_layout.addWidget(run_dialog)
        self._run_dialog = run_dialog

    def __add_help_menu(self) -> None:
        menuBar = self.menuBar()
        assert menuBar is not None
        help_menu = menuBar.addMenu("&Help")
        assert help_menu is not None

        help_links = self.plugin_manager.get_help_links() if self.plugin_manager else {}

        for menu_label, link in help_links.items():
            help_link_item = help_menu.addAction(menu_label)
            assert help_link_item is not None
            help_link_item.setMenuRole(QAction.MenuRole.ApplicationSpecificRole)
            help_link_item.triggered.connect(functools.partial(webbrowser.open, link))

        show_about = help_menu.addAction("About")
        assert show_about is not None
        show_about.setMenuRole(QAction.MenuRole.ApplicationSpecificRole)
        show_about.setObjectName("about_action")
        show_about.triggered.connect(self.__showAboutMessage)

        self.help_menu = help_menu

    def __showAboutMessage(self) -> None:
        diag = AboutDialog(self)
        diag.show()
