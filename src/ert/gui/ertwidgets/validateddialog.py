from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QWidget,
)


class ValidatedDialog(QDialog):
    """A dialog for creating a validated new value that is not already on
    a deny-list."""

    INVALID_COLOR = QColor(255, 235, 235)

    def __init__(
        self,
        title="Title",
        description="Description",
        unique_names=None,
    ):
        QDialog.__init__(self)
        self.setModal(True)
        self.setWindowTitle(title)

        if unique_names is None:
            unique_names = []

        self.unique_names = unique_names

        self.layout = QFormLayout()
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        label = QLabel(description)
        label.setAlignment(Qt.AlignHCenter)

        self.layout.addRow(self.createSpace(5))
        self.layout.addRow(label)
        self.layout.addRow(self.createSpace(10))

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)

        self.param_name = QLineEdit(self)
        self.param_name.setFocus()
        self.param_name.textChanged.connect(self.validateName)
        self.validColor = self.param_name.palette().color(
            self.param_name.backgroundRole()
        )

        self.layout.addRow("Name:", self.param_name)

        self.layout.addRow(self.createSpace(10))

        self.layout.addRow(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.setLayout(self.layout)

    def notValid(self, msg):
        """Called when the name is not valid."""
        self.ok_button.setEnabled(False)
        palette = self.param_name.palette()
        palette.setColor(self.param_name.backgroundRole(), self.INVALID_COLOR)
        self.param_name.setToolTip(msg)
        self.param_name.setPalette(palette)

    def valid(self):
        """Called when the name is valid."""
        self.ok_button.setEnabled(True)
        palette = self.param_name.palette()
        palette.setColor(self.param_name.backgroundRole(), self.validColor)
        self.param_name.setToolTip("")
        self.param_name.setPalette(palette)

    def validateName(self, value):
        """Called to perform validation of a name. For specific needs override
        this function and call valid() and notValid(msg)."""
        value = str(value)

        if value == "":
            self.notValid("Can not be empty!")
        elif value.find(" ") != -1:
            self.notValid("No spaces allowed!")
        elif value in self.unique_names:
            self.notValid("Name must be unique!")
        else:
            self.valid()

    def validateChoice(self, choice):
        """Only called when using selection mode."""
        self.ok_button.setEnabled(not choice == "")

    def getName(self):
        """Return the new name chosen by the user"""
        return str(self.param_name.text())

    def showAndTell(self):
        """Shows the dialog and returns the result"""
        if self.exec_():
            return str(self.getName()).strip()

        return ""

    def createSpace(self, size=5):
        """Creates a widget that can be used as spacing on  a panel."""
        qw = QWidget()
        qw.setMinimumSize(QSize(size, size))

        return qw
