from PyQt4 import QtGui, QtCore
from helpedwidget import *

class CheckBox(HelpedWidget):
    def __init__(self, parent=None, checkLabel="Do this", help="", altLabel="", defaultCheckState=True):
        """Construct a spinner widget for integers"""
        HelpedWidget.__init__(self, parent, checkLabel, help)

        if altLabel:
            self.check = QtGui.QCheckBox(altLabel, self)
        else:
            self.check = QtGui.QCheckBox(checkLabel, self)

        self.check.setTristate(False)

        if defaultCheckState:
            self.check.setCheckState(QtCore.Qt.Checked)
        else:
            self.check.setCheckState(QtCore.Qt.Unchecked)


        self.addWidget(self.check)

        self.addStretch()
        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        self.connect(self.check, QtCore.SIGNAL('stateChanged(int)'), self.contentsChanged)


    def contentsChanged(self):
        if self.check.checkState() == QtCore.Qt.Checked:
            self.updateContent(True)
        else:
            self.updateContent(False)


    def fetchContent(self):
        if self.getFromModel():
            self.check.setCheckState(QtCore.Qt.Checked)
        else:
            self.check.setCheckState(QtCore.Qt.Unchecked)

