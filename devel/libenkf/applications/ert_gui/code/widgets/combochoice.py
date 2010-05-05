from PyQt4 import QtGui, QtCore
from helpedwidget import *

class ComboChoice(HelpedWidget):
    """
    A combo box widget for choices. The data structure expected and sent to the getter and setter is a string
    that is equal to one of the available ones.
    """
    def __init__(self, parent=None, choiceList=None, comboLabel="Choice", help=""):
        """Construct a ComboChoice widget"""
        HelpedWidget.__init__(self, parent, comboLabel, help)

        self.combo = QtGui.QComboBox(self)

        if choiceList is None:
            choiceList = ["No choices"]

        for choice in choiceList:
            self.combo.addItem(choice)

        self.addWidget(self.combo)
        self.addStretch()
        self.addHelpButton()

        self.connect(self.combo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.updateContent)


    def fetchContent(self):
        """Retrieves data from the model and updates the combo box."""
        newValue = self.getFromModel()

        indexSet = False
        for i in range(self.combo.count()):
            if str(self.combo.itemText(i)).lower() == str(newValue).lower():
                self.combo.setCurrentIndex(i)
                indexSet = True
                break

        if not indexSet:
            raise AssertionError("ComboBox can not be set to: " + str(newValue))

    def updateList(self, choiceList):
        """Replace the list of choices with the specified items"""
        self.disconnect(self.combo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.updateContent)

        self.combo.clear()
        for choice in choiceList:
            self.combo.addItem(choice)

        self.connect(self.combo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.updateContent)