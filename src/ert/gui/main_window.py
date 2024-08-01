from __future__ import annotations

import functools
import webbrowser
from typing import TYPE_CHECKING, Dict, Optional

from qtpy.QtCore import QSettings, Qt, Signal
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QAction,
    QDockWidget,
    QMainWindow,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.about_dialog import AboutDialog
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.plugins import ErtPluginManager

if TYPE_CHECKING:
    from ert.gui.tools import Tool


class ErtMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(
        self, config_file: str, plugin_manager: Optional[ErtPluginManager] = None
    ):
        QMainWindow.__init__(self)
        self.notifier = ErtNotifier(config_file)
        self.tools: Dict[str, Tool] = {}

        self.setWindowTitle(f"ERT - {config_file} - {find_ert_info()}")

        self.plugin_manager = plugin_manager
        self.__main_widget: Optional[QWidget] = None

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout()
        self.central_widget.setLayout(self.central_layout)

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
        self.__add_help_menu()
        self.__fetchSettings()

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

    def __saveSettings(self) -> None:
        settings = QSettings("Equinor", "Ert-Gui")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def closeEvent(self, closeEvent: Optional[QCloseEvent]) -> None:
        # Use QT settings saving mechanism
        # settings stored in ~/.config/Equinor/ErtGui.conf

        if closeEvent is not None and self.notifier.is_simulation_running:
            closeEvent.ignore()
        else:
            self.__saveSettings()
            self.close_signal.emit()
            QMainWindow.closeEvent(self, closeEvent)

    def __fetchSettings(self) -> None:
        settings = QSettings("Equinor", "Ert-Gui")
        geo = settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        wnd = settings.value("windowState")
        if wnd:
            self.restoreState(wnd)

    def setWidget(self, widget: QWidget) -> None:
        self.__main_widget = widget
        actions = widget.getActions()
        for action in actions:
            self.__view_menu.addAction(action)

        self.central_layout.addWidget(widget)

    def __showAboutMessage(self) -> None:
        diag = AboutDialog(self)
        diag.show()
