from PyQt4 import QtGui, QtCore


class Application(QtGui.QMainWindow):
    def __init__(self):
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

        self.pagesWidget = QtGui.QStackedWidget()


        horizontalLayout = QtGui.QHBoxLayout()
        horizontalLayout.addWidget(self.contentsWidget)
        horizontalLayout.addWidget(self.pagesWidget, 1)
        widgetLayout.addLayout(horizontalLayout)


        quitButton = QtGui.QPushButton("Close", self)
        self.connect(quitButton, QtCore.SIGNAL('clicked()'), QtGui.qApp, QtCore.SLOT('quit()'))

        buttonWidget = QtGui.QWidget(self)
        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(quitButton)

        buttonWidget.setLayout(buttonLayout)
        widgetLayout.addWidget(buttonWidget)

        centralWidget.setLayout(widgetLayout)
        self.setCentralWidget(centralWidget)


    def addPage(self, name, icon, page):
        button = QtGui.QListWidgetItem(self.contentsWidget)
        button.setIcon(icon)
        button.setText(name)
        button.setTextAlignment(QtCore.Qt.AlignHCenter)
        button.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        
        self.pagesWidget.addWidget(page)
        self.connect(self.contentsWidget, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), self.changePage)

        self.contentsWidget.setCurrentRow(0)
        

    def changePage(self, current, previous):
        if current is None:
            current = previous

        self.pagesWidget.setCurrentIndex(self.contentsWidget.row(current))
