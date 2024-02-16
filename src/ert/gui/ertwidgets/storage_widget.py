from qtpy.QtCore import QSortFilterProxyModel, Qt
from qtpy.QtWidgets import (
    QHeaderView,
    QLineEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.caselist import AddWidget
from ert.gui.ertwidgets.create_experiment_dialog import CreateExperimentDialog
from ert.gui.ertwidgets.models.storage_model import StorageModel


class StorageWidget(QWidget):
    def __init__(
        self, notifier: ErtNotifier, ert_config: ErtConfig, ensemble_size: int
    ):
        QWidget.__init__(self)

        self._notifier = notifier
        self._ert_config = ert_config
        self._ensemble_size = ensemble_size

        tree_view = QTreeView(self)
        storage_model = StorageModel(self._notifier.storage)
        notifier.storage_changed.connect(storage_model.reloadStorage)
        notifier.ertChanged.connect(
            lambda: storage_model.reloadStorage(self._notifier.storage)
        )

        if isinstance(tree_view.header(), QHeaderView):
            tree_view.header().hide()

        search_bar = QLineEdit(self)
        search_bar.setPlaceholderText("Filter")
        proxy_model = QSortFilterProxyModel()
        proxy_model.setFilterKeyColumn(-1)  # Search all columns.
        proxy_model.setSourceModel(storage_model)
        proxy_model.sort(0, Qt.SortOrder.AscendingOrder)

        tree_view.setModel(proxy_model)
        search_bar.textChanged.connect(proxy_model.setFilterFixedString)

        self._create_experiment_button = AddWidget(self.add_item)

        layout = QVBoxLayout()
        layout.addWidget(search_bar)
        layout.addWidget(tree_view)
        layout.addWidget(self._create_experiment_button)

        self.setLayout(layout)

    def add_item(self) -> None:
        create_experiment_dialog = CreateExperimentDialog(parent=self)
        create_experiment_dialog.show()
        if create_experiment_dialog.exec_():
            ensemble = self._notifier.storage.create_experiment(
                parameters=self._ert_config.ensemble_config.parameter_configuration,
                responses=self._ert_config.ensemble_config.response_configuration,
                observations=self._ert_config.observations,
                name=create_experiment_dialog.experiment_name,
            ).create_ensemble(
                name=create_experiment_dialog.ensemble_name,
                ensemble_size=self._ensemble_size,
            )
            self._notifier.set_current_case(ensemble)
            self._notifier.ertChanged.emit()
