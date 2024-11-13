from __future__ import annotations

import datetime
import functools
import webbrowser
from typing import Dict, List, Optional

from qtpy.QtCore import QSize, Qt, Signal, Slot
from qtpy.QtGui import QCloseEvent, QCursor, QIcon, QMouseEvent
from qtpy.QtWidgets import (
    QAction,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QToolButton,
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
from ert.trace import get_trace_id

BUTTON_STYLE_SHEET: str = """
    QToolButton {
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0);
        padding-top: 5px;
        padding-bottom: 10px;
    }
    QToolButton::menu-indicator {
        right: 10px; bottom: 5px;
    }
"""

BUTTON_STYLE_SHEET_LIGHT: str = (
    BUTTON_STYLE_SHEET
    + """
    QToolButton:hover {background-color: rgba(50, 50, 50, 90);}
    """
)

BUTTON_STYLE_SHEET_DARK: str = (
    BUTTON_STYLE_SHEET
    + """
    QToolButton:hover {background-color: rgba(30, 30, 30, 150);}
    """
)


class SidebarToolButton(QToolButton):
    right_clicked = Signal()

    def mousePressEvent(self, event: Optional[QMouseEvent]) -> None:
        if event:
            if event.button() == Qt.MouseButton.RightButton:
                self.right_clicked.emit()
            else:
                super().mousePressEvent(event)


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

        self.setWindowTitle(
            f"ERT - {config_file} - {find_ert_info()} - {get_trace_id()[:8]}"
        )
        self.plugin_manager = plugin_manager
        self.central_widget = QFrame(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.central_widget.setLayout(self.central_layout)
        self.facade = LibresFacade(self.ert_config)
        self.side_frame = QFrame(self)
        self._external_plot_windows: List[PlotWindow] = []

        if self.is_dark_mode():
            self.side_frame.setStyleSheet("background-color: rgb(64, 64, 64);")
        else:
            self.side_frame.setStyleSheet("background-color: lightgray;")

        self.vbox_layout = QVBoxLayout(self.side_frame)
        self.side_frame.setLayout(self.vbox_layout)

        self.central_panels_map: Dict[str, QWidget] = {}
        self._experiment_panel: Optional[ExperimentPanel] = None
        self._plot_window: Optional[PlotWindow] = None
        self._manage_experiments_panel: Optional[ManageExperimentsPanel] = None
        self._add_sidebar_button("Start simulation", QIcon("img:library_add.svg"))
        plot_button = self._add_sidebar_button("Create plot", QIcon("img:timeline.svg"))
        plot_button.setToolTip("Right click to open external window")
        self._add_sidebar_button("Manage experiments", QIcon("img:build_wrench.svg"))
        self.results_button = self._add_sidebar_button(
            "Simulation status", QIcon("img:in_progress.svg")
        )
        self.results_button.setEnabled(False)
        self.run_dialog_counter = 0

        self.vbox_layout.addStretch()
        self.central_layout.addWidget(self.side_frame)

        self.central_widget.setMinimumWidth(1500)
        self.central_widget.setMinimumHeight(800)
        self.setCentralWidget(self.central_widget)

        self.__add_tools_menu()
        self.__add_help_menu()

    def is_dark_mode(self) -> bool:
        return self.palette().base().color().value() < 70

    def right_clicked(self) -> None:
        actor = self.sender()
        if actor and actor.property("index") == "Create plot":
            pw = PlotWindow(self.config_file, None)
            pw.show()
            self._external_plot_windows.append(pw)

    def select_central_widget(self) -> None:
        actor = self.sender()
        if actor:
            index_name = actor.property("index")

            for widget in self.central_panels_map.values():
                widget.setVisible(False)

            if (
                index_name == "Manage experiments"
                and not self._manage_experiments_panel
            ):
                self._manage_experiments_panel = ManageExperimentsPanel(
                    self.ert_config,
                    self.notifier,
                    self.ert_config.model_config.num_realizations,
                )

                self.central_panels_map["Manage experiments"] = (
                    self._manage_experiments_panel
                )
                self.central_layout.addWidget(self._manage_experiments_panel)

            if index_name == "Create plot":
                if self._plot_window:
                    self._plot_window.close()
                self._plot_window = PlotWindow(self.config_file, self)
                self.central_layout.addWidget(self._plot_window)
                self.central_panels_map["Create plot"] = self._plot_window

            if index_name == "Simulation status":
                # select the only available simulation
                for k, v in self.central_panels_map.items():
                    if isinstance(v, RunDialog):
                        index_name = k
                        break

            for i, widget in self.central_panels_map.items():
                widget.setVisible(i == index_name)

    @Slot(object)
    def slot_add_widget(self, run_dialog: RunDialog) -> None:
        for widget in self.central_panels_map.values():
            widget.setVisible(False)

        run_dialog.setParent(self)
        date_time = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%d-%m %H:%M:%S"
        )
        experiment_type = run_dialog._run_model.name()
        simulation_id = experiment_type + " : " + date_time
        self.central_panels_map[simulation_id] = run_dialog
        self.run_dialog_counter += 1
        self.central_layout.addWidget(run_dialog)

        def add_sim_run_option(simulation_id: str) -> None:
            menu = self.results_button.menu()
            if menu:
                action_list = menu.actions()
                act = QAction(text=simulation_id, parent=menu)
                act.setProperty("index", simulation_id)
                act.triggered.connect(self.select_central_widget)

                if action_list:
                    menu.insertAction(action_list[0], act)
                else:
                    menu.addAction(act)

        if self.run_dialog_counter == 2:
            # swap from button to menu selection
            self.results_button.clicked.disconnect(self.select_central_widget)
            self.results_button.setMenu(QMenu())
            self.results_button.setPopupMode(QToolButton.InstantPopup)

            for prev_date_time, widget in self.central_panels_map.items():
                if isinstance(widget, RunDialog):
                    add_sim_run_option(prev_date_time)
        elif self.run_dialog_counter > 2:
            add_sim_run_option(simulation_id)

        self.results_button.setEnabled(True)

    def post_init(self) -> None:
        experiment_panel = ExperimentPanel(
            self.ert_config,
            self.notifier,
            self.config_file,
            self.facade.get_ensemble_size(),
        )
        self.central_layout.addWidget(experiment_panel)
        self._experiment_panel = experiment_panel
        self.central_panels_map["Start simulation"] = self._experiment_panel

        experiment_panel.experiment_started.connect(self.slot_add_widget)

        plugin_handler = PluginHandler(
            self.notifier,
            [wfj for wfj in self.ert_config.workflow_jobs.values() if wfj.is_plugin()],
            self,
        )
        self.plugins_tool = PluginsTool(plugin_handler, self.notifier, self.ert_config)
        if self.plugins_tool:
            self.plugins_tool.setParent(self)
            menubar = self.menuBar()
            if menubar:
                menubar.insertMenu(
                    self.help_menu.menuAction(), self.plugins_tool.get_menu()
                )

    def _add_sidebar_button(self, name: str, icon: QIcon) -> SidebarToolButton:
        button = SidebarToolButton(self.side_frame)
        button.setFixedSize(85, 95)
        button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        button.setStyleSheet(
            BUTTON_STYLE_SHEET_DARK
        ) if self.is_dark_mode() else button.setStyleSheet(BUTTON_STYLE_SHEET_LIGHT)

        pad = 45
        icon_size = QSize(button.size().width() - pad, button.size().height() - pad)
        button.setIconSize(icon_size)
        button.setIcon(icon)
        button.setToolTip(name)
        objname = name.replace(" ", "_")
        button_text = name.replace(" ", "\n")
        button.setObjectName(f"button_{objname}")
        button.setText(button_text)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.vbox_layout.addWidget(button)

        button.clicked.connect(self.select_central_widget)
        button.right_clicked.connect(self.right_clicked)
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

        self.export_tool = ExportTool(self.ert_config, self.notifier)
        self.export_tool.setParent(self)
        tools_menu.addAction(self.export_tool.getAction())

        self.workflows_tool = WorkflowsTool(self.ert_config, self.notifier)
        self.workflows_tool.setParent(self)
        tools_menu.addAction(self.workflows_tool.getAction())

        self.load_results_tool = LoadResultsTool(self.facade, self.notifier)
        self.load_results_tool.setParent(self)
        tools_menu.addAction(self.load_results_tool.getAction())

    def closeEvent(self, closeEvent: Optional[QCloseEvent]) -> None:
        for plot_window in self._external_plot_windows:
            if plot_window:
                plot_window.close()

        if closeEvent is not None:
            if self.notifier.is_simulation_running:
                closeEvent.ignore()
            else:
                self.close_signal.emit()
                QMainWindow.closeEvent(self, closeEvent)

    def __showAboutMessage(self) -> None:
        diag = AboutDialog(self)
        diag.show()
