from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.storage import StorageAccessor


class AddRemoveWidget(QWidget):
    """
    A simple class that provides to vertically positioned buttons for adding and
    removing something.  The addFunction and removeFunction functions must be
    provided.
    """

    def __init__(self, addFunction=None, removeFunction=None):
        QWidget.__init__(self)

        self.addButton = QToolButton(self)
        self.addButton.setIcon(QIcon("img:add_circle_outlined.svg"))
        self.addButton.setIconSize(QSize(16, 16))
        self.addButton.clicked.connect(addFunction)

        self.removeButton = QToolButton(self)
        self.removeButton.setIcon(QIcon("img:remove_outlined.svg"))
        self.removeButton.setIconSize(QSize(16, 16))
        self.removeButton.clicked.connect(removeFunction)

        self.buttonLayout = QHBoxLayout()

        self.buttonLayout.setContentsMargins(0, 0, 0, 0)

        self.buttonLayout.addStretch(1)

        self.buttonLayout.addWidget(self.addButton)
        self.buttonLayout.addWidget(self.removeButton)

        self.buttonLayout.addSpacing(2)

        self.setLayout(self.buttonLayout)

    def enableRemoveButton(self, state):
        """Enable or disable the remove button"""
        self.removeButton.setEnabled(state)


class CaseList(QWidget):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int):
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self.notifier = notifier
        QWidget.__init__(self)

        layout = QVBoxLayout()

        self._list = QListWidget(self)
        self._default_selection_mode = self._list.selectionMode()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)

        layout.addWidget(QLabel("Available cases:"))
        layout.addWidget(self._list, stretch=1)

        self._addRemoveWidget = AddRemoveWidget(self.addItem, self.removeItem)
        self._addRemoveWidget.enableRemoveButton(False)
        layout.addWidget(self._addRemoveWidget)

        self._title = "New keyword"
        self._description = "Enter name of keyword:"

        self.setLayout(layout)

        notifier.ertChanged.connect(self.updateList)
        self.updateList()

    @property
    def storage(self) -> StorageAccessor:
        return self.notifier.storage

    def addItem(self):
        dialog = ValidatedDialog(
            "New case",
            "Enter name of new case:",
            [x.name for x in self.storage.ensembles],
            parent=self,
        )
        new_case_name = dialog.showAndTell()
        if new_case_name != "":
            ensemble = self.storage.create_experiment(
                parameters=self.ert_config.ensemble_config.parameter_configuration,
                responses=self.ert_config.ensemble_config.response_configuration,
                observations=self.ert_config.observations,
            ).create_ensemble(
                name=new_case_name,
                ensemble_size=self.ensemble_size,
            )
            self.notifier.set_current_case(ensemble)
            self.notifier.ertChanged.emit()

    def removeItem(self):
        message = "Support for removal of items has not been implemented!"
        QMessageBox.information(self, "Not implemented!", message)

    def updateList(self):
        """Retrieves data from the model and inserts it into the list"""
        case_list = sorted(
            self.storage.ensembles, key=lambda x: x.started_at, reverse=True
        )

        self._list.clear()

        for case in case_list:
            item = QListWidgetItem(f"{case.name} - {case.started_at} ({case.id})")
            item.setData(Qt.UserRole, case)
            self._list.addItem(item)
