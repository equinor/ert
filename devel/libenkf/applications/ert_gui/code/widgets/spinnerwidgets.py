from PyQt4 import QtGui, QtCore
from helpedwidget import *

class IntegerSpinner(HelpedWidget):
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
        self.updateContent(self.spinner.value())

    def fetchContent(self):
        self.spinner.setValue(self.getFromModel())


class DoubleSpinner(HelpedWidget):
    def __init__(self, parent=None, spinnerLabel="Double Number", help="", min=0.0, max=1.0, decimals=2):
        """Construct a spinner widget for integers"""
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
        self.updateContent(self.spinner.value())

    def fetchContent(self):
        self.spinner.setValue(self.getFromModel())