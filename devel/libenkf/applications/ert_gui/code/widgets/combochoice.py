from PyQt4 import QtGui, QtCore
from helpedwidget import *

class ComboChoice(HelpedWidget):
    def __init__(self, parent=None, choiceList=["No choices"], comboLabel="Choice", help=""):
        """Construct a PathChooser widget"""
        HelpedWidget.__init__(self, parent, comboLabel, help)

        self.combo = QtGui.QComboBox(self)

        for choice in choiceList:
            self.combo.addItem(choice)

        self.addWidget(self.combo)
        self.addStretch()
        self.addHelpButton()

        self.connect(self.combo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.updateContent)


    def fetchContent(self):
        newValue = self.getFromModel()

        indexSet = False
        for i in range(self.combo.count()):
            if self.combo.itemText(i) == newValue:
                self.combo.setCurrentIndex(i)
                indexSet = True
                break

        if not indexSet:
            raise AssertionError("ComboBox can not be set to: " + newValue)