from typing import Callable

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.storage import Storage


class AddWidget(QWidget):
    """
    A widget with an add button.

    Parameters
    ----------
    addFunction: Callable to be connected to the add button.
    """

    def __init__(self, addFunction: Callable[[], None]) -> None:
        super().__init__()

        self.addButton = QToolButton(self)
        self.addButton.setIcon(QIcon("img:add_circle_outlined.svg"))
        self.addButton.setIconSize(QSize(16, 16))
        self.addButton.clicked.connect(addFunction)

        self.removeButton = None

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.addStretch(1)
        self.buttonLayout.addWidget(self.addButton)
        self.buttonLayout.addSpacing(2)

        self.setLayout(self.buttonLayout)


class EnsembleList(QWidget):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int):
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self.notifier = notifier
        QWidget.__init__(self)

        layout = QVBoxLayout()

        self._list = QListWidget(self)
        self._default_selection_mode = self._list.selectionMode()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)

        layout.addWidget(QLabel("Available ensembles:"))
        layout.addWidget(self._list, stretch=1)

        self._addWidget = AddWidget(self.addItem)
        layout.addWidget(self._addWidget)

        self._title = "New keyword"
        self._description = "Enter name of keyword:"

        self.setLayout(layout)

        notifier.ertChanged.connect(self.updateList)
        self.updateList()

    @property
    def storage(self) -> Storage:
        return self.notifier.storage

    def addItem(self) -> None:
        dialog = ValidatedDialog(
            "New ensemble",
            "Enter name of new ensemble:",
            [x.name for x in self.storage.ensembles],
            parent=self,
        )
        new_ensemble_name = dialog.showAndTell()
        if new_ensemble_name:
            ensemble = self.storage.create_experiment(
                parameters=self.ert_config.ensemble_config.parameter_configuration,
                responses=self.ert_config.ensemble_config.response_configuration,
                observations=self.ert_config.observations,
            ).create_ensemble(
                name=new_ensemble_name,
                ensemble_size=self.ensemble_size,
            )
            self.notifier.set_current_ensemble(ensemble)
            self.notifier.ertChanged.emit()

    def updateList(self) -> None:
        """Retrieves data from the model and inserts it into the list"""
        ensemble_list = sorted(
            self.storage.ensembles, key=lambda x: x.started_at, reverse=True
        )

        self._list.clear()

        for ensemble in ensemble_list:
            item = QListWidgetItem(
                f"{ensemble.name} - {ensemble.started_at} ({ensemble.id})"
            )
            item.setData(Qt.ItemDataRole.UserRole, ensemble)
            self._list.addItem(item)
