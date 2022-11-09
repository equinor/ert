from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLayout,
    QWidget,
)


class CustomDialog(QDialog):
    INVALID_COLOR = QColor(255, 235, 235)

    def __init__(self, title="Title", description="Description", parent=None):
        QDialog.__init__(self, parent)

        self._option_list = []
        """ :type: list of QWidget """

        self.setModal(True)
        self.setWindowTitle(title)

        self.layout = QFormLayout()
        self.layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        label = QLabel(description)
        label.setAlignment(Qt.AlignHCenter)

        self.layout.addRow(self.createSpace(5))
        self.layout.addRow(label)
        self.layout.addRow(self.createSpace(10))

        self.ok_button = None

        self.setLayout(self.layout)

    def notValid(self, msg):
        """Called when the name is not valid."""
        self.ok_button.setEnabled(False)

    def valid(self):
        """Called when the name is valid."""
        self.ok_button.setEnabled(True)

    def optionValidationChanged(self):
        valid = True
        for option in self._option_list:
            if hasattr(option, "isValid") and not option.isValid():
                valid = False
                self.notValid("One or more options are incorrectly set!")

        if valid:
            self.valid()

    def showAndTell(self) -> bool:
        """
        Shows the dialog modally and returns the true or false (accept/reject)
        """
        self.optionValidationChanged()
        return self.exec_()

    def createSpace(self, size=5):
        """Creates a widget that can be used as spacing on  a panel."""
        qw = QWidget()
        qw.setMinimumSize(QSize(size, size))

        return qw

    def addSpace(self, size=10):
        """Add some vertical spacing"""
        space_widget = self.createSpace(size)
        self.layout.addRow("", space_widget)

    def addLabeledOption(self, label, option_widget):
        """
        @type option_widget: QWidget
        """
        self._option_list.append(option_widget)

        if hasattr(option_widget, "validationChanged"):
            option_widget.validationChanged.connect(self.optionValidationChanged)

        if hasattr(option_widget, "getValidationSupport"):
            validation_support = option_widget.getValidationSupport()
            validation_support.validationChanged.connect(self.optionValidationChanged)

        self.layout.addRow(f"{label}:", option_widget)

    def addWidget(self, widget, label=""):
        if not label.endswith(":"):
            label = f"{label}:"
        self.layout.addRow(label, widget)

    def addButtons(self):
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)

        self.layout.addRow(self.createSpace(10))
        self.layout.addRow(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
