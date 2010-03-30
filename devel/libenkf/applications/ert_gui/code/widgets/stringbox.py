from PyQt4 import QtGui, QtCore
from helpedwidget import *

class StringBox(HelpedWidget):
    """PathChooser shows, enables choosing of and validates paths."""

    def __init__(self, parent=None, pathLabel="String", help="", defaultString=""):
        """Construct a PathChooser widget"""
        HelpedWidget.__init__(self, parent, pathLabel, help)


        self.pathLine = QtGui.QLineEdit()
        self.connect(self.pathLine, QtCore.SIGNAL('editingFinished()'), self.validateString)
        self.connect(self.pathLine, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)
        self.connect(self.pathLine, QtCore.SIGNAL('textChanged(QString)'), self.validateString)
        self.addWidget(self.pathLine)


        self.addHelpButton()

        self.validColor = self.pathLine.palette().color(self.pathLine.backgroundRole())

        self.pathLine.setText(defaultString)


    def validateString(self):
        stringToValidate = self.pathLine.text()
        #todo implement validation possibility


    def contentsChanged(self):
        self.updateContent(self.pathLine.text())

    def fetchContent(self):
        self.pathLine.setText(self.getFromModel())
