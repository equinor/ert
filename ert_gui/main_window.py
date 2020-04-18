import functools
import os
import sys
import webbrowser
from subprocess import Popen

import pkg_resources
import yaml
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import (
    QAction,
    QDockWidget,
    QMainWindow,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert_gui.about_dialog import AboutDialog
from ert_shared.plugins import ErtPluginManager


class GertMainWindow(QMainWindow):
    def __init__(self, args, storage_client):
        QMainWindow.__init__(self)
        self._storage_client = storage_client
        self._service_discovery = (
            args.storage_api_url is None and storage_client is not None
        )

        if self._service_discovery:
            # TODO: add restarting of this server?
            Popen([sys.argv[0], "api", "--port", "0", "--project", args.config])

        self.tools = {}

        self.resize(300, 700)
        self.setWindowTitle("ERT - {}".format(args.config))

        self.__main_widget = None

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout()
        self.central_widget.setLayout(self.central_layout)

        self.setCentralWidget(self.central_widget)

        self.toolbar = self.addToolBar("Tools")
        self.toolbar.setObjectName("Toolbar")
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)

        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)
        self.__view_menu = None
        self.__help_menu = None

        self.__createMenu()
        self.__fetchSettings()

    def addDock(
        self,
        name,
        widget,
        area=Qt.RightDockWidgetArea,
        allowed_areas=Qt.AllDockWidgetAreas,
    ):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName("%sDock" % name)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)

        self.addDockWidget(area, dock_widget)

        self.__view_menu.addAction(dock_widget.toggleViewAction())
        return dock_widget

    def addTool(self, tool):
        tool.setParent(self)
        self.tools[tool.getName()] = tool
        self.toolbar.addAction(tool.getAction())

        if tool.isPopupMenu():
            tool_button = self.toolbar.widgetForAction(tool.getAction())
            tool_button.setPopupMode(QToolButton.InstantPopup)

    def __createMenu(self):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Close", self.close)
        self.__view_menu = self.menuBar().addMenu("&View")
        self.__help_menu = self.menuBar().addMenu("&Help")
        """:type: QMenu"""

        """ @rtype: list of QAction """
        show_about = self.__help_menu.addAction("About")
        show_about.setMenuRole(QAction.ApplicationSpecificRole)
        show_about.triggered.connect(self.__showAboutMessage)

        if sys.version_info.major >= 3:
            pm = ErtPluginManager()
            help_links = pm.get_help_links()
        else:
            with pkg_resources.resource_stream(
                "ert_gui", os.path.join("resources", "gui", "help", "help_links.yml")
            ) as stream:
                help_links = yaml.safe_load(stream)

        for menu_label, link in help_links.items():
            help_link_item = self.__help_menu.addAction(menu_label)
            help_link_item.setMenuRole(QAction.ApplicationSpecificRole)
            help_link_item.triggered.connect(functools.partial(webbrowser.open, link))

    def __saveSettings(self):
        settings = QSettings("Equinor", "Ert-Gui")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        # Use QT settings saving mechanism
        # settings stored in ~/.config/Equinor/ErtGui.conf
        self.__saveSettings()
        QMainWindow.closeEvent(self, event)

        if self._service_discovery:
            self._storage_client.shutdown_server()

    def __fetchSettings(self):
        settings = QSettings("Equinor", "Ert-Gui")
        geo = settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        wnd = settings.value("windowState")
        if wnd:
            self.restoreState(wnd)

    def setWidget(self, widget):
        self.__main_widget = widget
        actions = widget.getActions()
        for action in actions:
            self.__view_menu.addAction(action)

        self.central_layout.addWidget(widget)

    def __showAboutMessage(self):
        diag = AboutDialog(self)
        diag.show()
        pass
