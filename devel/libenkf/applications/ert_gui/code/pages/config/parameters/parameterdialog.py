from PyQt4 import QtGui, QtCore
from widgets.validateddialog import ValidatedDialog

class ParameterDialog(ValidatedDialog):
    """A dialog for creating parameters based on type and name. Performs validation of name."""

    def __init__(self, parent, types, uniqueNames):
        """Creates a new dialog that validates uniqueness against the provided list"""
        ValidatedDialog.__init__(self, parent, 'Create new parameter', "Select type and enter name of parameter:", uniqueNames)

        self.paramCombo = QtGui.QComboBox(self)

        keys = types.keys()
        keys.sort()
        for key in keys:
            self.paramCombo.addItem(types[key], key)

        self.layout.insertRow(2, "Type:", self.paramCombo)


    def getTypeName(self):
        """Return the type selected by the user"""
        return str(self.paramCombo.currentText().strip())

