from typing import Any, List, Optional, Union

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLayout,
    QPushButton,
    QWidget,
)


class CustomDialog(QDialog):
    INVALID_COLOR = QColor(255, 235, 235)

    def __init__(
        self,
        title: str = "Title",
        description: str = "Description",
        parent: Optional[QWidget] = None,
    ) -> None:
        QDialog.__init__(self, parent)

        self._option_list: List[QWidget] = []

        self.setModal(True)
        self.setWindowTitle(title)

        self._layout = QFormLayout()
        self._layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self._layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        label = QLabel(description)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._layout.addRow(self.createSpace(5))
        self._layout.addRow(label)
        self._layout.addRow(self.createSpace(10))

        self.ok_button: Optional[QPushButton] = None

        self.setLayout(self._layout)

    def notValid(self, msg: str) -> None:
        """Called when the name is not valid."""
        if self.ok_button:
            self.ok_button.setEnabled(False)

    def valid(self) -> None:
        """Called when the name is valid."""
        if self.ok_button:
            self.ok_button.setEnabled(True)

    def optionValidationChanged(self) -> None:
        valid = True
        for option in self._option_list:
            if hasattr(option, "isValid") and not option.isValid():
                valid = False
                self.notValid("One or more options are incorrectly set!")

        if valid:
            self.valid()

    def showAndTell(self) -> int:
        """
        Shows the dialog modally and returns the true or false (accept/reject)
        """
        self.optionValidationChanged()
        return self.exec_()

    @staticmethod
    def createSpace(size: int = 5) -> QWidget:
        """Creates a widget that can be used as spacing on  a panel."""
        qw = QWidget()
        qw.setMinimumSize(QSize(size, size))

        return qw

    def addLabeledOption(self, label: Any, option_widget: QWidget) -> None:
        self._option_list.append(option_widget)

        if hasattr(option_widget, "validationChanged"):
            option_widget.validationChanged.connect(self.optionValidationChanged)

        if hasattr(option_widget, "getValidationSupport"):
            validation_support = option_widget.getValidationSupport()
            validation_support.validationChanged.connect(self.optionValidationChanged)

        self._layout.addRow(f"{label}:", option_widget)

    def addWidget(self, widget: Union[QWidget, QLayout, None], label: str = "") -> None:
        if not label.endswith(":"):
            label = f"{label}:"
        self._layout.addRow(label, widget)

    def addButtons(self) -> None:
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        if self.ok_button:
            self.ok_button.setEnabled(False)

        self._layout.addRow(self.createSpace(10))
        self._layout.addRow(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
