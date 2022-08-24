from qtpy.QtCore import QSize
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import addHelpToWidget, resourceIcon
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.libres_facade import LibresFacade


class AddRemoveWidget(QWidget):
    """
    A simple class that provides to vertically positioned buttons for adding and
    removing something.  The addFunction and removeFunction functions must be
    provided.
    """

    def __init__(self, addFunction=None, removeFunction=None, horizontal=False):
        QWidget.__init__(self)

        self.addButton = QToolButton(self)
        self.addButton.setIcon(resourceIcon("add_circle_outlined.svg"))
        self.addButton.setIconSize(QSize(16, 16))
        self.addButton.clicked.connect(addFunction)

        self.removeButton = QToolButton(self)
        self.removeButton.setIcon(resourceIcon("remove_outlined.svg"))
        self.removeButton.setIconSize(QSize(16, 16))
        self.removeButton.clicked.connect(removeFunction)

        if horizontal:
            self.buttonLayout = QHBoxLayout()
        else:
            self.buttonLayout = QVBoxLayout()

        self.buttonLayout.setContentsMargins(0, 0, 0, 0)

        if horizontal:
            self.buttonLayout.addStretch(1)

        self.buttonLayout.addWidget(self.addButton)
        self.buttonLayout.addWidget(self.removeButton)

        if not horizontal:
            self.buttonLayout.addStretch(1)
        else:
            self.buttonLayout.addSpacing(2)

        self.setLayout(self.buttonLayout)

    def enableAddButton(self, state):
        """Enable or disable the add button"""
        self.addButton.setEnabled(state)

    def enableRemoveButton(self, state):
        """Enable or disable the remove button"""
        self.removeButton.setEnabled(state)


class CaseList(QWidget):
    def __init__(self, facade: LibresFacade, notifier: ErtNotifier):
        self.facade = facade
        self.notifier = notifier
        QWidget.__init__(self)

        addHelpToWidget(self, "init/case_list")

        layout = QVBoxLayout()

        self._list = QListWidget(self)
        self._list.setMinimumHeight(100)
        self._list.setMaximumHeight(250)
        self._default_selection_mode = self._list.selectionMode()
        self.setSelectable(False)

        layout.addWidget(QLabel("Available cases:"))
        layout.addWidget(self._list)

        self._addRemoveWidget = AddRemoveWidget(
            self.addItem, self.removeItem, horizontal=True
        )
        self._addRemoveWidget.enableRemoveButton(False)
        layout.addWidget(self._addRemoveWidget)

        self._title = "New keyword"
        self._description = "Enter name of keyword:"

        self.setLayout(layout)

        notifier.ertChanged.connect(self.updateList)
        self.updateList()

    def setSelectable(self, selectable):
        if selectable:
            self._list.setSelectionMode(self._default_selection_mode)
        else:
            self._list.setSelectionMode(QAbstractItemView.NoSelection)

    def addItem(self):
        dialog = ValidatedDialog(
            "New case", "Enter name of new case:", self.facade.cases()
        )
        new_case_name = dialog.showAndTell()
        if not new_case_name == "":
            self.facade.select_or_create_new_case(new_case_name)
            self.notifier.ertChanged.emit()

    def removeItem(self):
        message = "Support for removal of items has not been implemented!"
        QMessageBox.information(self, "Not implemented!", message)

    def updateList(self):
        """Retrieves data from the model and inserts it into the list"""
        case_list = self.facade.cases()

        self._list.clear()

        for case in case_list:
            self._list.addItem(case)
