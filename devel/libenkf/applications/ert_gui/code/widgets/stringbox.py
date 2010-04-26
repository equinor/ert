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
        box_string_text = str(self.boxString.text())
        if box_string_text == "":
            box_string_text = None

        self.updateContent(box_string_text)

    def fetchContent(self):
        """Retrieves data from the model and inserts it into the edit line"""
        self_get_from_model = self.getFromModel()
        if self_get_from_model is None:
            self_get_from_model = ""

        self.boxString.setText(self_get_from_model)



class DoubleBox(HelpedWidget):
    """DoubleBox shows a double value. The data structure expected and sent to the getter and setter is a double."""

    def __init__(self, parent=None, pathLabel="Double", help=""):
        """Construct a DoubleBox widget"""
        HelpedWidget.__init__(self, parent, pathLabel, help)

        self.doubleBox = QtGui.QLineEdit()
        self.doubleBox.setValidator(QtGui.QDoubleValidator(self))
        self.doubleBox.setMaximumWidth(75)

        self.connect(self.doubleBox, QtCore.SIGNAL('editingFinished()'), self.validateString)
        self.connect(self.doubleBox, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)
        self.connect(self.doubleBox, QtCore.SIGNAL('textChanged(QString)'), self.validateString)
        self.addWidget(self.doubleBox)

        self.addStretch()
        self.addHelpButton()


    def validateString(self):
        """Override this to provide validation of the contained string. NOT SUPPORTED YET!"""
        stringToValidate = self.doubleBox.text()
        #todo implement validation possibility


    def contentsChanged(self):
        """Called whenever the contents of the editline changes."""
        self.updateContent(self.doubleBox.text().toDouble()[0])

    def fetchContent(self):
        """Retrieves data from the model and inserts it into the edit line"""
        self.doubleBox.setText(str(self.getFromModel()))
