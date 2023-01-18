import functools
import webbrowser

from qtpy.QtCore import QSettings, Qt, Signal
from qtpy.QtWidgets import (
    QAction,
    QDockWidget,
    QMainWindow,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.about_dialog import AboutDialog
from ert.shared.plugins import ErtPluginManager


class ErtMainWindow(QMainWindow):
    close_signal = Signal()

    def __init__(self, config_file):
        QMainWindow.__init__(self)
        self.tools = {}

        self.resize(300, 700)
        self.setWindowTitle(f"ERT - {config_file}")

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
        dock_widget.setObjectName(f"{name}Dock")
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
        self.__view_menu = self.menuBar().addMenu("&View")
        self.__help_menu = self.menuBar().addMenu("&Help")
        show_about = self.__help_menu.addAction("About")
        show_about.setMenuRole(QAction.ApplicationSpecificRole)
        show_about.triggered.connect(self.__showAboutMessage)

        pm = ErtPluginManager()
        help_links = pm.get_help_links()

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
        self.close_signal.emit()
        QMainWindow.closeEvent(self, event)

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
