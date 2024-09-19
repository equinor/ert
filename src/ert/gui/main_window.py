from __future__ import annotations

import functools
import webbrowser
from typing import TYPE_CHECKING, Dict, Optional

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QCloseEvent, QCursor, QIcon
from qtpy.QtWidgets import (
    QAction,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert import LibresFacade
from ert.config import ErtConfig
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.gui.tools.export import ExportTool
from ert.gui.tools.load_results import LoadResultsTool
from ert.gui.tools.manage_experiments import ManageExperimentsTool
from ert.gui.tools.plot import PlotTool
from ert.gui.tools.workflows import WorkflowsTool
from ert.plugins import ErtPluginManager

if TYPE_CHECKING:
    from ert.gui.tools import Tool


class ErtMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(
        self,
        config_file: str,
        ertconfig: ErtConfig,
        plugin_manager: Optional[ErtPluginManager] = None,
    ):
        QMainWindow.__init__(self)
        self.notifier = ErtNotifier(config_file)
        self.tools: Dict[str, Tool] = {}
        self.ertconfig = ertconfig
        self.config_file = config_file

        self.setWindowTitle(f"ERT - {config_file} - {find_ert_info()}")

        self.plugin_manager = plugin_manager
        self.central_widget = QFrame(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_widget.setLayout(self.central_layout)

        self.facade = LibresFacade(self.ertconfig)

        self.side_frame = QFrame(self)
        self.side_frame.setFrameShape(QFrame.Box)
        self.vbox_layout = QVBoxLayout(self.side_frame)
        self.side_frame.setLayout(self.vbox_layout)

        self.add_experiment_button()
        self._plot_tool = PlotTool(self.config_file, self)
        self.add_sidebar_button(self._plot_tool)

        self._manage_experiments_tool = ManageExperimentsTool(
            self.ertconfig, self.notifier, self.ertconfig.model_config.num_realizations
        )
        self.add_sidebar_button(self._manage_experiments_tool)

        self.vbox_layout.addStretch()
        self.central_layout.addWidget(self.side_frame)

        self.central_widget.setMaximumWidth(2400)
        self.central_widget.setMaximumHeight(1200)

        self.setCentralWidget(self.central_widget)

        toolbar = self.addToolBar("Tools")
        assert toolbar is not None
        self.toolbar = toolbar
        self.toolbar.setObjectName("Toolbar")
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        self.setCorner(Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(
            Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self.setCorner(Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea
        )

        menuBar = self.menuBar()
        assert menuBar is not None
        view_menu = menuBar.addMenu("&View")
        assert view_menu is not None
        self.__view_menu = view_menu
        self.__add_tools_menu()
        self.__add_help_menu()

    def add_experiment_button(self) -> None:
        button = QPushButton(self.side_frame)
        button.setFixedSize(80, 80)
        button.setIcon(QIcon("img:play_circle_outlined.svg"))
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.setStyleSheet(
            """
            QPushButton {
            border: 2px solid darkgrey;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0);
            padding-bottom: 10px;
            }
            """
        )
        padding = 30
        button.setIconSize(
            QSize(button.size().width() - padding, button.size().height() - padding)
        )
        button.setToolTip("Start Simulation")
        # button.clicked.connect(self.experiment_panel.show)
        self.vbox_layout.addWidget(button)

    def add_sidebar_button(self, tool: Tool) -> None:
        button = QPushButton(self.side_frame)
        button.setFixedSize(80, 80)
        button.setIcon(tool.getIcon())
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.setStyleSheet(
            """
            QPushButton {
            border: 2px solid darkgrey;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0);
            padding-bottom: 10px;
            }
            """
        )
        padding = 30
        button.setIconSize(
            QSize(button.size().width() - padding, button.size().height() - padding)
        )
        button.setToolTip(tool.getName())
        button.clicked.connect(tool.trigger)
        self.vbox_layout.addWidget(button)

    def addDock(
        self,
        name: str,
        widget: Optional[QWidget],
        area: Qt.DockWidgetArea = Qt.DockWidgetArea.RightDockWidgetArea,
        allowed_areas: Qt.DockWidgetArea = Qt.DockWidgetArea.AllDockWidgetAreas,
    ) -> QDockWidget:
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName(f"{name}Dock")
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)

        self.addDockWidget(area, dock_widget)

        self.__view_menu.addAction(dock_widget.toggleViewAction())
        return dock_widget

    def addTool(self, tool: Tool) -> None:
        tool.setParent(self)
        self.tools[tool.getName()] = tool
        self.toolbar.addAction(tool.getAction())

        if tool.isPopupMenu():
            tool_button = self.toolbar.widgetForAction(tool.getAction())
            assert tool_button is not None
            tool_button.setPopupMode(QToolButton.InstantPopup)

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

    def __add_tools_menu(self) -> None:
        menu_bar = self.menuBar()
        assert menu_bar is not None
        tools_menu = menu_bar.addMenu("&Tools")
        assert tools_menu is not None

        self._export_tool = ExportTool(self.ertconfig, self.notifier)
        self._export_tool.setParent(self)
        tools_menu.addAction(self._export_tool.getAction())

        self._workflows_tool = WorkflowsTool(self.ertconfig, self.notifier)
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

    def setWidget(self, widget: QWidget) -> None:
        actions = widget.getActions()
        for action in actions:
            self.__view_menu.addAction(action)

        self.central_layout.addWidget(widget)

    def __showAboutMessage(self) -> None:
        diag = AboutDialog(self)
        diag.show()
