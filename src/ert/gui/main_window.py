from __future__ import annotations

import functools
import logging
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QWidget,
)

from ert.config import ErtConfig, ErtScriptWorkflow
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.gui.plotting.plot_window import PlotWindow
from ert.gui.tools.event_viewer import EventViewerTool, GUILogHandler
from ert.gui.tools.load_results import LoadResultsTool
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.plugins import PluginHandler, PluginsTool
from ert.gui.tools.workflows import WorkflowsTool
from ert.plugins import ErtRuntimePlugins
from ert.trace import get_trace_id

from .experiments import ExperimentPanel, RunDialog
from .sidebar import (
    CREATE_PLOT,
    EXPERIMENT_STATUS,
    MANAGE_EXPERIMENTS,
    START_EXPERIMENT,
    Sidebar,
)

logger = logging.getLogger(__name__)

NAVIGATION_PAGE_NAMES: frozenset[str] = frozenset(
    {START_EXPERIMENT, CREATE_PLOT, MANAGE_EXPERIMENTS, EXPERIMENT_STATUS}
)


def _clicked_help_link(menu_label: str, link: str) -> None:
    logger.info(f"Gui utility: {menu_label} help link was used from main window")
    webbrowser.open(link)


class ErtMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(
        self,
        config_file: str,
        ert_config: ErtConfig,
        runtime_plugins: ErtRuntimePlugins | None = None,
        log_handler: GUILogHandler | None = None,
    ) -> None:
        QMainWindow.__init__(self)
        self._init_state(config_file, ert_config, runtime_plugins, log_handler)
        self._init_central_widget()
        self._init_sidebar()
        self._init_panels()
        self.__add_tools_menu()
        self.__add_help_menu()

    def _init_state(
        self,
        config_file: str,
        ert_config: ErtConfig,
        runtime_plugins: ErtRuntimePlugins | None,
        log_handler: GUILogHandler | None,
    ) -> None:
        self.notifier = ErtNotifier()
        self.plugins_tool: PluginsTool | None = None
        self.ert_config = ert_config
        self.config_file = config_file
        self.log_handler = log_handler
        self.runtime_plugins = runtime_plugins

        self.setWindowTitle(
            f"ERT - {config_file} - {find_ert_info()} - {get_trace_id()[:8]}"
        )

    def _init_central_widget(self) -> None:
        self.central_widget = QFrame(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.central_widget.setLayout(self.central_layout)
        self._external_plot_windows: list[PlotWindow] = []

        self.central_widget.setMinimumWidth(1500)
        self.central_widget.setMinimumHeight(800)
        self.setCentralWidget(self.central_widget)

    def _init_sidebar(self) -> None:
        self.sidebar = Sidebar(self)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.sidebar)
        self.sidebar.page_requested.connect(self.select_central_widget)
        self.sidebar.external_plot_requested.connect(self.right_clicked)
        self.sidebar.status_entry_selected.connect(self.select_central_widget)
        self.sidebar.set_status_enabled(False)
        self.sidebar.set_current(START_EXPERIMENT)

    def _init_panels(self) -> None:
        self.central_panels_map: dict[str, QWidget] = {}
        self._experiment_panel: ExperimentPanel | None = None
        self._plot_window: PlotWindow | None = None
        self._panel_factories: dict[str, Callable[[], QWidget]] = {
            MANAGE_EXPERIMENTS: self._build_manage_experiments_panel,
            CREATE_PLOT: self._build_plot_window,
        }
        # Panels that must be rebuilt on every selection so they pick up newly
        # created ensembles/experiments instead of showing stale data.
        self._rebuild_on_select: frozenset[str] = frozenset({CREATE_PLOT})

    def right_clicked(self) -> None:
        pw = PlotWindow(
            self.config_file, Path(self.ert_config.ens_path).absolute(), None
        )
        pw.show()
        self._external_plot_windows.append(pw)

    def get_external_plot_windows(self) -> list[PlotWindow]:
        return self._external_plot_windows

    def _build_manage_experiments_panel(self) -> QWidget:
        return ManageExperimentsPanel(
            self.ert_config,
            self.notifier,
            self.ert_config.ensemble_size,
        )

    def _build_plot_window(self) -> QWidget:
        if self._plot_window:
            self._plot_window.close()
        self._plot_window = PlotWindow(
            self.config_file, Path(self.ert_config.ens_path).absolute(), self
        )
        return self._plot_window

    def _ensure_panel(self, name: str) -> None:
        factory = self._panel_factories.get(name)
        if factory is None:
            return
        if name in self.central_panels_map and name not in self._rebuild_on_select:
            return
        panel = factory()
        self.central_panels_map[name] = panel
        self.central_layout.addWidget(panel)

    def _first_run_dialog_name(self) -> str | None:
        for name, widget in self.central_panels_map.items():
            if isinstance(widget, RunDialog):
                return name
        return None

    def _show_only(self, name: str) -> None:
        for panel_name, widget in self.central_panels_map.items():
            widget.setVisible(panel_name == name)

    def select_central_widget(self, index_name: str) -> None:
        self._ensure_panel(index_name)

        if index_name == EXPERIMENT_STATUS:
            index_name = self._first_run_dialog_name() or index_name

        self._show_only(index_name)

        if index_name not in NAVIGATION_PAGE_NAMES:
            self.sidebar.set_current(EXPERIMENT_STATUS)

    @Slot(RunDialog)
    def slot_add_widget(self, run_dialog: RunDialog) -> None:
        for widget in self.central_panels_map.values():
            widget.setVisible(False)

        run_dialog.setParent(self)
        experiment_name = run_dialog.property("experiment_name")
        self.central_panels_map[experiment_name] = run_dialog
        self.central_layout.addWidget(run_dialog)

        self.sidebar.add_status_entry(experiment_name)
        self.sidebar.set_status_enabled(True)

    def post_init(self) -> None:
        experiment_panel = ExperimentPanel(
            self.ert_config,
            self.notifier,
            self.config_file,
        )
        experiment_panel.experiment_started.connect(
            lambda _: self.sidebar.set_current(EXPERIMENT_STATUS)
        )
        self.central_layout.addWidget(experiment_panel)
        self._experiment_panel = experiment_panel
        self.central_panels_map[START_EXPERIMENT] = self._experiment_panel

        experiment_panel.experiment_started.connect(self.slot_add_widget)

        plugin_handler = PluginHandler(
            self.notifier,
            [
                wfj
                for wfj in self.ert_config.workflow_jobs.values()
                if isinstance(wfj, ErtScriptWorkflow) and wfj.is_plugin()
            ],
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

    def __add_help_menu(self) -> None:
        menuBar = self.menuBar()
        assert menuBar is not None
        help_menu = menuBar.addMenu("&Help")
        assert help_menu is not None

        help_links = self.runtime_plugins.help_links if self.runtime_plugins else {}

        for menu_label, link in help_links.items():
            help_link_item = help_menu.addAction(menu_label)
            assert help_link_item is not None
            help_link_item.setMenuRole(QAction.MenuRole.ApplicationSpecificRole)
            help_link_item.triggered.connect(
                functools.partial(_clicked_help_link, menu_label, link)
            )
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

        self.workflows_tool = WorkflowsTool(self.ert_config, self.notifier)
        self.workflows_tool.setParent(self)
        tools_menu.addAction(self.workflows_tool.getAction())

        self.load_results_tool = LoadResultsTool(self.ert_config, self.notifier)
        self.load_results_tool.setParent(self)
        tools_menu.addAction(self.load_results_tool.getAction())

    @override
    def closeEvent(self, closeEvent: QCloseEvent | None) -> None:
        for plot_window in self._external_plot_windows:
            if plot_window:
                plot_window.close()

        if closeEvent is not None:
            if self.notifier.is_experiment_running:
                closeEvent.ignore()
            else:
                self.close_signal.emit()
                QMainWindow.closeEvent(self, closeEvent)

    def __showAboutMessage(self) -> None:
        diag = AboutDialog(self)
        diag.show()
