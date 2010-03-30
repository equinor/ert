import os
from PyQt4 import QtGui, QtCore
from helpedwidget import *

class PathChooser(HelpedWidget):
    """PathChooser shows, enables choosing of and validates paths."""

    invalidColor = QtGui.QColor(255, 235, 235)
    #invalidColor = QtGui.QColor(255, 204, 153)

    def __init__(self, parent=None, pathLabel="Path", help="", files=False):
        """Construct a PathChooser widget"""
        HelpedWidget.__init__(self, parent, pathLabel, help)

        self.selectFiles = files

        self.pathLine = QtGui.QLineEdit()
        #self.pathLine.setMinimumWidth(250)
        
        self.connect(self.pathLine, QtCore.SIGNAL('editingFinished()'), self.validatePath)
        self.connect(self.pathLine, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)
        self.connect(self.pathLine, QtCore.SIGNAL('textChanged(QString)'), self.validatePath)
        self.addWidget(self.pathLine)

        dialogButton = QtGui.QToolButton(self)
        dialogButton.setIcon(QtGui.QIcon.fromTheme("folder"))
        dialogButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(dialogButton, QtCore.SIGNAL('clicked()'), self.selectDirectory)
        self.addWidget(dialogButton)

        #self.addStretch()
        self.addHelpButton()

        self.validColor = self.pathLine.palette().color(self.pathLine.backgroundRole())

        #self.pathLine.setText(os.path.expanduser("~"))
        self.pathLine.setText(os.getcwd())

    #palette = self.palette()
    #palette.setColor(self.backgroundRole(), QtCore.Qt.red)
    #self.setPalette(palette)
    #self.setAutoFillBackground(True)


    def validatePath(self):
        palette = self.pathLine.palette()

        if os.path.exists(self.pathLine.text()):
            palette.setColor(self.pathLine.backgroundRole(), self.validColor)
            self.pathLine.setToolTip("")
        else:
            palette.setColor(self.pathLine.backgroundRole(), self.invalidColor)
            self.pathLine.setToolTip("The specififed path does not exist.")

        self.pathLine.setPalette(palette)


    def selectDirectory(self):
        currentDirectory = self.pathLine.text()

        #if not os.path.exists(currentDirectory):
        #    currentDirectory = "~"

        if self.selectFiles:
            currentDirectory = QtGui.QFileDialog.getOpenFileName(self, "Select a path", currentDirectory)
        else:
            currentDirectory = QtGui.QFileDialog.getExistingDirectory(self, "Select a directory", currentDirectory)

        if not currentDirectory == "":
            self.pathLine.setText(currentDirectory)


    def contentsChanged(self):
        self.updateContent(self.pathLine.text())

    def fetchContent(self):
        self.pathLine.setText(self.getFromModel())

