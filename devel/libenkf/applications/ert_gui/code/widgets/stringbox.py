from PyQt4 import QtGui, QtCore
from helpedwidget import *

class StringBox(HelpedWidget):
    """StringBox shows a string. The data structure expected and sent to the getter and setter is a string."""

    def __init__(self, parent=None, pathLabel="String", help="", defaultString=""):
        """Construct a StringBox widget"""
        HelpedWidget.__init__(self, parent, pathLabel, help)

        self.boxString = QtGui.QLineEdit()
        self.connect(self.boxString, QtCore.SIGNAL('editingFinished()'), self.validateString)
        self.connect(self.boxString, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)
        self.connect(self.boxString, QtCore.SIGNAL('textChanged(QString)'), self.validateString)
        self.addWidget(self.boxString)

        self.addHelpButton()

        self.boxString.setText(defaultString)


    def validateString(self):
        """Override this to provide validation of the contained string. NOT SUPPORTED YET!"""
        stringToValidate = self.boxString.text()
        #todo implement validation possibility


    def contentsChanged(self):
        """Called whenever the contents of the editline changes."""
        self.updateContent(self.boxString.text())

    def fetchContent(self):
        """Retrieves data from the model and inserts it into the edit line"""
        self.boxString.setText(self.getFromModel())
