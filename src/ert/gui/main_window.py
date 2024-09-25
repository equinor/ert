from __future__ import annotations

import datetime
import functools
import webbrowser
from typing import Dict, Optional

from qtpy.QtCore import QSize, Qt, Signal, Slot
from qtpy.QtGui import QCloseEvent, QCursor, QIcon
from qtpy.QtWidgets import (
    QAction,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ert import LibresFacade
from ert.config import ErtConfig
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.gui.simulation import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.event_viewer import EventViewerTool, GUILogHandler
from ert.gui.tools.export import ExportTool
from ert.gui.tools.load_results import LoadResultsTool
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.plot.plot_window import PlotWindow
from ert.gui.tools.plugins import PluginHandler, PluginsTool
from ert.gui.tools.workflows import WorkflowsTool
from ert.plugins import ErtPluginManager

BUTTON_STYLE_SHEET: str = """
    QPushButton {
    border: 2px solid darkgrey;
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0);
    padding-bottom: 10px;
    }
    QPushButton:hover {
       background-color: rgba(255, 255, 255, 200);
    }
"""
MENU_ITEM_STYLE_SHEET: str = """
    QMenu:item {
        border: 2px solid darkgrey;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0);
        padding-top:5px;
        padding-left: 5px;
        background: rgba(0,0,0,0);
        font-weight: bold;
        font-size: 13px;
    }
    QMenu:item:selected {
        background-color: rgba(255, 255, 255, 200);
    }
"""


class ErtMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(
        self,
        config_file: str,
        ert_config: ErtConfig,
        plugin_manager: Optional[ErtPluginManager] = None,
        log_handler: Optional[GUILogHandler] = None,
    ):
        QMainWindow.__init__(self)
        self.notifier = ErtNotifier(config_file)
        self.plugins_tool: Optional[PluginsTool] = None
        self.ert_config = ert_config
        self.config_file = config_file
        self.log_handler = log_handler

        self.setWindowTitle(f"ERT - {config_file} - {find_ert_info()}")
        self.plugin_manager = plugin_manager
        self.central_widget = QFrame(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.central_widget.setLayout(self.central_layout)
        self.facade = LibresFacade(self.ert_config)
        self.side_frame = QFrame(self)
        self.vbox_layout = QVBoxLayout(self.side_frame)
        self.side_frame.setLayout(self.vbox_layout)

        self.central_panels_map: Dict[str, QWidget] = {}

        button = self._add_sidebar_button(
            "Start Simulation", QIcon("img:play_circle_outlined.svg")
        )
        menu = QMenu()
        menu.setStyleSheet(MENU_ITEM_STYLE_SHEET)

        for sim_mode in [
            "Single test run",
            "Ensemble Experiment",
            "Manual Update",
            "ES MDA",
            "Ensemble Smoother",
        ]:
            act = menu.addAction(sim_mode)
            act.triggered.connect(self.select_central_widget)
            # todo use sim modes to select correct panels directly
            act.setProperty("index", "Start Simulation")
        button.setMenu(menu)

        self._plot_window: Optional[PlotWindow] = None
        self._add_sidebar_button("Create plot", QIcon("img:timeline.svg"))

        self._manage_experiments_panel: Optional[ManageExperimentsPanel] = None
        self._add_sidebar_button("Manage experiments", QIcon("img:build_wrench.svg"))
        self.results_button = self._add_sidebar_button(
            "Show Results", QIcon("img:in_progress.svg")
        )
        self.results_button.setEnabled(False)
        self.results_button.setMenu(QMenu())

        self.vbox_layout.addStretch()
        self.central_layout.addWidget(self.side_frame)

        self.central_widget.setMaximumWidth(2400)
        self.central_widget.setMaximumHeight(1200)
        self.setCentralWidget(self.central_widget)

        self.__add_tools_menu()
        self.__add_help_menu()

    def select_central_widget(self):
        actor = self.sender()
        index_name = actor.property("index")

        if index_name == "Create plot" and not self._plot_window:
            self._plot_window = PlotWindow(self.config_file, self)
            self.central_layout.addWidget(self._plot_window)
            self.central_panels_map["Create plot"] = self._plot_window

        for i, widget in self.central_panels_map.items():
            widget.setVisible(i == index_name)

    @Slot(object)
    def slot_add_widget(self, run_dialog: RunDialog) -> None:
        for widget in self.central_panels_map.values():
            widget.setVisible(False)

        run_dialog.setParent(self)
        self.central_layout.addWidget(run_dialog)
        self.results_button.setEnabled(True)
        date_time = datetime.datetime.utcnow().strftime("%Y-%d-%m %H:%M:%S")
        self.central_panels_map[date_time] = run_dialog
        act = self.results_button.menu()

        if act:
            act.addAction(date_time)
            act.setProperty("index", date_time)
            act.triggered.connect(self.select_central_widget)

    def post_init(self) -> None:
        experiment_panel = ExperimentPanel(
            self.ert_config,
            self.notifier,
            self.config_file,
            self.facade.get_ensemble_size(),
        )
        self.central_layout.addWidget(experiment_panel)
        self.central_panels_map["Start Simulation"] = experiment_panel

        experiment_panel.experiment_started.connect(self.slot_add_widget)

        plugin_handler = PluginHandler(
            self.notifier,
            [wfj for wfj in self.ert_config.workflow_jobs.values() if wfj.is_plugin()],
            self,
        )
        self.plugins_tool = PluginsTool(plugin_handler, self.notifier, self.ert_config)
        if self.plugins_tool:
            self.plugins_tool.setParent(self)
            self.menuBar().addMenu(self.plugins_tool.get_menu())

        self._manage_experiments_panel = ManageExperimentsPanel(
            self.ert_config,
            self.notifier,
            self.ert_config.model_config.num_realizations,
        )

        self.central_panels_map["Manage experiments"] = self._manage_experiments_panel
        self._manage_experiments_panel.hide()
        self.central_layout.addWidget(self._manage_experiments_panel)

    def _add_sidebar_button(self, name: str, icon: QIcon) -> QPushButton:
        button = QPushButton(self.side_frame)
        button.setFixedSize(80, 80)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.setStyleSheet(BUTTON_STYLE_SHEET)
        pad = 30
        icon_size = QSize(button.size().width() - pad, button.size().height() - pad)
        button.setIconSize(icon_size)
        button.setIcon(icon)
        button.setToolTip(name)
        self.vbox_layout.addWidget(button)

        button.clicked.connect(self.select_central_widget)
        button.setProperty("index", name)
        return button

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

    def __add_tools_menu(self) -> None:
        menu_bar = self.menuBar()
        assert menu_bar is not None
        tools_menu = menu_bar.addMenu("&Tools")
        assert tools_menu is not None

        if self.log_handler:
            self._event_viewer_tool = EventViewerTool(
                self.log_handler, self.config_file
            )
            self._event_viewer_tool.setParent(self)
            tools_menu.addAction(self._event_viewer_tool.getAction())
            self.close_signal.connect(self._event_viewer_tool.close_wnd)

        self._export_tool = ExportTool(self.ert_config, self.notifier)
        self._export_tool.setParent(self)
        tools_menu.addAction(self._export_tool.getAction())

        self._workflows_tool = WorkflowsTool(self.ert_config, self.notifier)
        self._workflows_tool.setParent(self)
        tools_menu.addAction(self._workflows_tool.getAction())

        self._load_results_tool = LoadResultsTool(self.facade, self.notifier)
        self._load_results_tool.setParent(self)
        tools_menu.addAction(self._load_results_tool.getAction())

    def closeEvent(self, closeEvent: Optional[QCloseEvent]) -> None:
        if closeEvent is not None:
            if self.notifier.is_simulation_running:
                closeEvent.ignore()
            else:
                self.close_signal.emit()
                QMainWindow.closeEvent(self, closeEvent)

    def __showAboutMessage(self) -> None:
        diag = AboutDialog(self)
        diag.show()
