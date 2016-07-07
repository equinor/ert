from PyQt4.QtGui import QListWidget, QMessageBox, QAbstractItemView, QWidget, QVBoxLayout, QLabel

from ert_gui import ERT
from ert_gui.ertwidgets import addHelpToWidget
from ert_gui.ertwidgets.models.ertmodel import getAllCases, selectOrCreateNewCase
from ert_gui.ertwidgets.validateddialog import ValidatedDialog
from ert_gui.widgets.add_remove_widget import AddRemoveWidget


class CaseList(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        addHelpToWidget(self, "init/case_list")

        layout = QVBoxLayout()

        self._list = QListWidget(self)
        self._list.setMinimumHeight(100)
        self._list.setMaximumHeight(250)
        self._default_selection_mode = self._list.selectionMode()
        self.setSelectable(False)

        layout.addWidget(QLabel("Available Cases:"))
        layout.addWidget(self._list)

        self._addRemoveWidget = AddRemoveWidget(self.addItem, self.removeItem, horizontal=True)
        self._addRemoveWidget.enableRemoveButton(False)
        layout.addWidget(self._addRemoveWidget)

        self._title = "New keyword"
        self._description = "Enter name of keyword:"

        self.setLayout(layout)

        ERT.ertChanged.connect(self.updateList)
        self.updateList()

    def setSelectable(self, selectable):
        if selectable:
            self._list.setSelectionMode(self._default_selection_mode)
        else:
            self._list.setSelectionMode(QAbstractItemView.NoSelection)


    def addItem(self):
        dialog = ValidatedDialog("New case", "Enter name of new case:", getAllCases())
        new_case_name = dialog.showAndTell()
        if not new_case_name == "":
            selectOrCreateNewCase(new_case_name)

    def removeItem(self):
        message = "Support for removal of items has not been implemented!"
        QMessageBox.information(self, "Not implemented!", message)


    def updateList(self):
        """Retrieves data from the model and inserts it into the list"""
        case_list = getAllCases()

        self._list.clear()

        for case in case_list:
            self._list.addItem(case)
