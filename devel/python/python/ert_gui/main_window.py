from PyQt4.QtCore import QSettings, Qt
from PyQt4.QtGui import QMainWindow, qApp, QWidget, QVBoxLayout, QDockWidget, QAction
from ert_gui.widgets.help_dock import HelpDock
from ert_gui.widgets.helped_widget import HelpedWidget


class GertMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.tools = {}

        self.resize(300, 700)
        self.setWindowTitle('gERT')

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout()
        self.central_widget.setLayout(self.central_layout)

        self.setCentralWidget(self.central_widget)

        self.toolbar = self.addToolBar("Tools")
        self.toolbar.setObjectName("Toolbar")
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        self.__createMenu()
        self.__help_dock = HelpDock()
        self.addDock("&Help", self.__help_dock, Qt.BottomDockWidgetArea, Qt.AllDockWidgetAreas)


        self.__fetchSettings()

    def addDock(self, name, widget, area=Qt.RightDockWidgetArea, allowed_areas=Qt.AllDockWidgetAreas):
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
        action = self.toolbar.addAction(tool.getIcon(), tool.getName())
        action.setIconText(tool.getName())
        action.setEnabled(tool.isEnabled())
        action.triggered.connect(tool.trigger)

        HelpedWidget.addHelpToAction(action, tool.getHelpLink())


    def __createMenu(self):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Close", self.__quit)
        self.__view_menu = self.menuBar().addMenu("&View")


    def __quit(self):
        self.__saveSettings()
        qApp.quit()


    def __saveSettings(self):
        settings = QSettings("Statoil", "ErtGui")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())


    def closeEvent(self, event):
        #Use QT settings saving mechanism
        #settings stored in ~/.config/Statoil/ErtGui.conf
        self.__saveSettings()
        QMainWindow.closeEvent(self, event)


    def __fetchSettings(self):
        settings = QSettings("Statoil", "ErtGui")
        self.restoreGeometry(settings.value("geometry").toByteArray())
        self.restoreState(settings.value("windowState").toByteArray())


    def setWidget(self, widget):
        actions = widget.getActions()
        for action in actions:
            self.__view_menu.addAction(action)

        self.central_layout.addWidget(widget)




