from PyQt4 import QtGui, QtCore
import util

class ValidatedDialog(QtGui.QDialog):
    """A dialog for creating a validated. Performs validation of name."""

    invalidColor = QtGui.QColor(255, 235, 235)

    def __init__(self, parent, title = "Title", description = "Description", uniqueNames = None):
        """Creates a new dialog that validates uniqueness against the provided list"""
        QtGui.QDialog.__init__(self, parent)
        self.setModal(True)
        self.setWindowTitle(title)
        self.setMinimumWidth(250)
        self.setMinimumHeight(150)

        if uniqueNames is None:
            uniqueNames = []
        self.uniqueNames = uniqueNames

        self.layout = QtGui.QFormLayout()

        self.layout.addRow(QtGui.QLabel(description))
        self.layout.addRow(util.createSpace(10))


        self.paramName = QtGui.QLineEdit(self)
        self.connect(self.paramName, QtCore.SIGNAL('textChanged(QString)'), self.validateName)
        self.validColor = self.paramName.palette().color(self.paramName.backgroundRole())

        self.layout.addRow("Name:", self.paramName)
        self.layout.addRow(util.createSpace(10))

        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        self.okbutton = buttons.button(QtGui.QDialogButtonBox.Ok)
        self.okbutton.setEnabled(False)

        self.layout.addRow(buttons)

        self.connect(buttons, QtCore.SIGNAL('accepted()'), self.accept)
        self.connect(buttons, QtCore.SIGNAL('rejected()'), self.reject)

        self.setLayout(self.layout)

    def notValid(self, msg):
        """Called when the name is not valid."""
        self.okbutton.setEnabled(False)
        palette = self.paramName.palette()
        palette.setColor(self.paramName.backgroundRole(), self.invalidColor)
        self.paramName.setToolTip(msg)
        self.paramName.setPalette(palette)

    def valid(self):
        """Called when the name is valid."""
        self.okbutton.setEnabled(True)
        palette = self.paramName.palette()
        palette.setColor(self.paramName.backgroundRole(), self.validColor)
        self.paramName.setToolTip("")
        self.paramName.setPalette(palette)

    def validateName(self, value):
        """Called to perform validation of a name. For specific needs override this function and call valid() and notValid(msg)."""
        value = str(value)

        if value == "":
            self.notValid("Can not be empty!")
        elif not value.find(" ") == -1:
            self.notValid("No spaces allowed!")
        elif value in self.uniqueNames:
            self.notValid("Name must be unique!")
        else:
            self.valid()

    def getName(self):
        """Return the new name chosen by the user"""
        return str(self.paramName.text())

    def showAndTell(self):
        """Shows the dialog and returns the result"""
        if self.exec_():
            return str(self.getName()).strip()

        return ""