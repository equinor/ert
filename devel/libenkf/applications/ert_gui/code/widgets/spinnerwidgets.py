from PyQt4 import QtGui, QtCore
from helpedwidget import *

class IntegerSpinner(HelpedWidget):
    """A spinner widget for integers. The data structure expected and sent to the getter and setter is an integer."""
    def __init__(self, parent=None, spinnerLabel="Number", help="", min=0, max=10):
        """Construct a spinner widget for integers"""
        HelpedWidget.__init__(self, parent, spinnerLabel, help)

        self.spinner = QtGui.QSpinBox(self)
        self.spinner.setMinimum(min)
        self.spinner.setMaximum(max)
        #self.connect(self.pathLine, QtCore.SIGNAL('textChanged(QString)'), self.validatePath)
        self.addWidget(self.spinner)

        self.addStretch()
        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        self.connect(self.spinner, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)

    def contentsChanged(self):
        """Called whenever the contents of the spinner changes."""
        self.updateContent(self.spinner.value())

    def fetchContent(self):
        """Retrieves data from the model and inserts it into the spinner"""
        self.spinner.setValue(self.getFromModel())


class DoubleSpinner(HelpedWidget):
    """A spinner widget for doubles. The data structure expected and sent to the getter and setter is a double."""
    def __init__(self, parent=None, spinnerLabel="Double Number", help="", min=0.0, max=1.0, decimals=2):
        """Construct a spinner widget for doubles"""
        HelpedWidget.__init__(self, parent, spinnerLabel, help)

        self.spinner = QtGui.QDoubleSpinBox(self)
        self.spinner.setMinimum(min)
        self.spinner.setMaximum(max)
        self.spinner.setDecimals(decimals)
        self.spinner.setSingleStep(0.01)

        self.addWidget(self.spinner)

        self.addStretch()
        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        self.connect(self.spinner, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)

    def contentsChanged(self):
        """Called whenever the contents of the spinner changes."""
        self.updateContent(self.spinner.value())


    def fetchContent(self):
        """Retrieves data from the model and inserts it into the spinner"""
        self.spinner.setValue(self.getFromModel())