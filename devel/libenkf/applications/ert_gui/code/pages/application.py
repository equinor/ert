from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QDockWidget
from PyQt4.QtCore import Qt, QSettings

class Application(QtGui.QMainWindow):
    """An application (window widget) with a list of "tasks" on the left side and a panel on the right side"""


    def __init__(self):
        """Constructor"""
        QtGui.QMainWindow.__init__(self)

        self.resize(900, 700)
        self.setWindowTitle('ERT GUI')

        centralWidget = QtGui.QWidget()
        widgetLayout = QtGui.QVBoxLayout()

        self.contentsWidget = QtGui.QListWidget()
        self.contentsWidget.setViewMode(QtGui.QListView.IconMode)
        self.contentsWidget.setIconSize(QtCore.QSize(96, 96))
        self.contentsWidget.setMovement(QtGui.QListView.Static)
        self.contentsWidget.setMaximumWidth(128)
        self.contentsWidget.setMinimumWidth(128)
        self.contentsWidget.setSpacing(12)

        dock = self.createDock()

        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.pagesWidget = QtGui.QStackedWidget()


        horizontalLayout = QtGui.QHBoxLayout()
        horizontalLayout.addWidget(self.pagesWidget, 1)
        widgetLayout.addLayout(horizontalLayout)

        self.createMenu(dock)

        centralWidget.setLayout(widgetLayout)
        self.setCentralWidget(centralWidget)

        self.save_function = None

        settings = QSettings("Statoil", "ErtGui")
        self.restoreGeometry(settings.value("geometry").toByteArray())
        self.restoreState(settings.value("windowState").toByteArray())

    def setSaveFunction(self, save_function):
        self.save_function = save_function

    def save(self):
        if not self.save_function is None:
            self.save_function()

    def createDock(self):
        dock = QDockWidget("Workflow")
        dock.setObjectName("ERTGUI Workflow")
        dock.setWidget(self.contentsWidget)
        dock.setFeatures(QDockWidget.DockWidgetClosable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        return dock

    def createMenu(self, dock):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Save configuration", self.save)
        file_menu.addAction("Close", QtGui.qApp.quit)
        
        self.view_menu = self.menuBar().addMenu("&View")
        self.view_menu.addAction(dock.toggleViewAction())
        self.view_menu.addSeparator()

    def addPage(self, name, icon, page):
        """Add another page to the appliation"""
        button = QtGui.QListWidgetItem(self.contentsWidget)
        button.setIcon(icon)
        button.setText(name)
        button.setTextAlignment(QtCore.Qt.AlignHCenter)
        button.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

        def switchPage():
            self.contentsWidget.setCurrentRow(self.contentsWidget.row(button))

        self.view_menu.addAction(name, switchPage)

        self.pagesWidget.addWidget(page)
        self.connect(self.contentsWidget, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), self.changePage)

        self.contentsWidget.setCurrentRow(0)
        

    def changePage(self, current, previous):
        """Switch page. Connected to the: currentItemChanged() signal of the list widget on the left side"""
        if current is None:
            current = previous

        self.pagesWidget.setCurrentIndex(self.contentsWidget.row(current))

    def closeEvent(self, event):
        settings = QSettings("Statoil", "ErtGui")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        QtGui.QMainWindow.closeEvent(self, event)
