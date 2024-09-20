from __future__ import annotations

import datetime
import functools
import webbrowser
from typing import TYPE_CHECKING, Dict, Optional

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
)

from ert import LibresFacade
from ert.config import ert_config
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.gui.simulation import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.event_viewer import EventViewerTool, GUILogHandler
from ert.gui.tools.export import ExportTool
from ert.gui.tools.load_results import LoadResultsTool
from ert.gui.tools.manage_experiments import ManageExperimentsTool
from ert.gui.tools.plot import PlotTool
from ert.gui.tools.plugins import PluginHandler, PluginsTool
from ert.gui.tools.workflows import WorkflowsTool
from ert.plugins import ErtPluginManager

if TYPE_CHECKING:
    from ert.gui.tools import Tool

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
        ert_config: ert_config,
        plugin_manager: Optional[ErtPluginManager] = None,
        log_handler: Optional[GUILogHandler] = None,
    ):
        QMainWindow.__init__(self)
        self.notifier = ErtNotifier(config_file)
        self.tools: Dict[str, Tool] = {}
        self.ert_config = ert_config
        self.config_file = config_file
        self.log_handler = log_handler

        self.setWindowTitle(f"ERT - {config_file} - {find_ert_info()}")
        self.dialog_panels = []
        self.central_panels = []

        self.plugin_manager = plugin_manager
        self.central_widget = QFrame(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_widget.setLayout(self.central_layout)

        self.facade = LibresFacade(self.ert_config)

        self.side_frame = QFrame(self)
        self.side_frame.setFrameShape(QFrame.Box)
        self.vbox_layout = QVBoxLayout(self.side_frame)
        self.side_frame.setLayout(self.vbox_layout)

        self.add_experiment_button()
        self._plot_tool = PlotTool(self.config_file, self)
        self._create_sidebar_button(self._plot_tool)

        self._manage_experiments_tool = ManageExperimentsTool(
            self.ert_config,
            self.notifier,
            self.ert_config.model_config.num_realizations,
        )

        self._create_sidebar_button(self._manage_experiments_tool)

        self.results_button = self._create_sidebar_button()
        self.results_button.setIcon(QIcon("img:in_progress.svg"))
        self.results_button.setToolTip("Show Results")
        self.results_button.setEnabled(False)
        menu = QMenu()
        self.results_button.setMenu(menu)

        self.vbox_layout.addStretch()
        self.central_layout.addWidget(self.side_frame)

        self.central_widget.setMaximumWidth(2400)
        self.central_widget.setMaximumHeight(1200)
        self.setCentralWidget(self.central_widget)

        self.__add_tools_menu()
        self.__add_help_menu()

    @Slot(object)
    def slot_add_widget(self, run_dialog: RunDialog):
        print("hello!")
        for widget in self.central_panels:
            widget.setVisible(False)

        self.dialog_panels.append(run_dialog)

        run_dialog.setParent(self)
        self.central_layout.addWidget(run_dialog)
        self.results_button.setEnabled(True)
        date_time = datetime.datetime.utcnow().strftime("%Y-%d-%m %H:%M:%S")
        act = self.results_button.menu().addAction(date_time)
        act.setProperty("index", len(self.dialog_panels) - 1)
        act.triggered.connect(self.select_dialog_panel)

    def select_dialog_panel(self):
        actor = self.sender()
        index = int(actor.property("index"))

        for w in self.central_panels:
            w.hide()

        for i in range(len(self.dialog_panels)):
            should_be_visible = i == index
            self.dialog_panels[i].setVisible(should_be_visible)

    def select_widget(self) -> None:
        index = 0

        for w in self.dialog_panels:
            w.hide()

        for i in range(len(self.central_panels)):
            should_be_visible = i == index
            self.central_panels[i].setVisible(should_be_visible)

    def post_init(self):
        experiment_panel = ExperimentPanel(
            self.ert_config,
            self.notifier,
            self.config_file,
            self.facade.get_ensemble_size(),
        )
        self.central_layout.addWidget(experiment_panel)
        self.central_panels.append(experiment_panel)

        experiment_panel.experiment_started.connect(self.slot_add_widget)

        plugin_handler = PluginHandler(
            self.notifier,
            [wfj for wfj in self.ert_config.workflow_jobs.values() if wfj.is_plugin()],
            self,
        )
        plugins_tool = PluginsTool(plugin_handler, self.notifier, self.ert_config)
        plugins_tool.setParent(self)
        self.menuBar().addMenu(plugins_tool.get_menu())

    def _create_sidebar_button(self, tool: Optional[Tool] = None) -> QPushButton:
        button = QPushButton(self.side_frame)
        button.setFixedSize(80, 80)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.setStyleSheet(BUTTON_STYLE_SHEET)
        padding = 30
        button.setIconSize(
            QSize(button.size().width() - padding, button.size().height() - padding)
        )
        if tool:
            button.setIcon(QIcon(tool.getIcon()))
            button.setToolTip(tool.getName())
            button.clicked.connect(tool.trigger)
        self.vbox_layout.addWidget(button)
        button.setProperty("index", len(self.central_panels) - 1)
        return button

    def add_experiment_button(self) -> None:
        button = self._create_sidebar_button()
        button.setIcon(QIcon("img:play_circle_outlined.svg"))
        button.setToolTip("Start Simulation")
        button.clicked.connect(self.select_widget)

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
            act.triggered.connect(self.select_widget)
        button.setMenu(menu)

    def toggle_visibility(self) -> None:
        for panel in self.central_panels:
            panel.show()

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
            help_link_item.triggered.connect(functools.partial(webbrowser.open, link))  # type: ignore

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
        if closeEvent is not None and self.notifier.is_simulation_running:
            closeEvent.ignore()
        else:
            self.close_signal.emit()
            QMainWindow.closeEvent(self, closeEvent)

    def __showAboutMessage(self) -> None:
        diag = AboutDialog(self)
        diag.show()
