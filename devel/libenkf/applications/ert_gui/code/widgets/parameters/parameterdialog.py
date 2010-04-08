from PyQt4 import QtGui, QtCore

class ParameterDialog(QtGui.QDialog):
    """A dialog for creating parameters based on type and name. Performs validation of name."""

    invalidColor = QtGui.QColor(255, 235, 235)

    def __init__(self, parent, types, uniqueNames):
        """Creates a new dialog that validates uniqueness against the provided list"""
        QtGui.QDialog.__init__(self, parent)
        self.setModal(True)
        self.setWindowTitle('Create new parameter')

        self.uniqueNames = uniqueNames

        layout = QtGui.QFormLayout()

        layout.addRow(QtGui.QLabel("Select type and enter name of parameter:"))

        layout.addRow(self.createSpace())

        self.paramCombo = QtGui.QComboBox(self)

        keys = types.keys()
        keys.sort()
        for key in keys:
            self.paramCombo.addItem(types[key], key)

        layout.addRow("Type:", self.paramCombo)

        self.paramName = QtGui.QLineEdit(self)
        self.connect(self.paramName, QtCore.SIGNAL('textChanged(QString)'), self.validateName)
        self.validColor = self.paramName.palette().color(self.paramName.backgroundRole())

        layout.addRow("Name:", self.paramName)

        layout.addRow(self.createSpace())

        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        self.okbutton = buttons.button(QtGui.QDialogButtonBox.Ok)
        self.okbutton.setEnabled(False)

        layout.addRow(buttons)


        self.connect(buttons, QtCore.SIGNAL('accepted()'), self.accept)
        self.connect(buttons, QtCore.SIGNAL('rejected()'), self.reject)

        self.setLayout(layout)

    def notValid(self, msg):
        """Called when the parameter name is not valid."""
        self.okbutton.setEnabled(False)
        palette = self.paramName.palette()
        palette.setColor(self.paramName.backgroundRole(), self.invalidColor)
        self.paramName.setToolTip(msg)
        self.paramName.setPalette(palette)

    def valid(self):
        """Called when the parameter name is valid."""
        self.okbutton.setEnabled(True)
        palette = self.paramName.palette()
        palette.setColor(self.paramName.backgroundRole(), self.validColor)
        self.paramName.setToolTip("")
        self.paramName.setPalette(palette)


    def validateName(self, value):
        """Called to perform validation of a parameter name"""
        value = str(value)

        if value == "":
            self.notValid("Can not be empty!")
        elif not value.find(" ") == -1:
            self.notValid("No spaces allowed!")
        elif value in self.uniqueNames:
            self.notValid("Name must be unique!")
        else:
            self.valid()


    def createSpace(self):
        """Create some space in the layout"""
        space = QtGui.QFrame()
        space.setMinimumSize(QtCore.QSize(10, 10))
        return space

    def getType(self):
        """Return the type selected by the user"""
        return str(self.paramCombo.currentText())

    def getName(self):
        """Return the parameter name chosen by the user"""
        return str(self.paramName.text())