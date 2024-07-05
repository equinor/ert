from qtpy.QtCore import (
    QAbstractItemModel,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
    Slot,
)
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.create_experiment_dialog import CreateExperimentDialog
from ert.gui.ertwidgets.models.storage_model import (
    EnsembleModel,
    ExperimentModel,
    RealizationModel,
    StorageModel,
)
from ert.storage import Ensemble, Experiment


class _SortingProxyModel(QSortFilterProxyModel):
    def __init__(self, model: QAbstractItemModel):
        super().__init__()
        self.setSourceModel(model)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        left_data = left.data()
        right_data = right.data()

        if (
            isinstance(left_data, str)
            and "Realization" in left_data
            and isinstance(right_data, str)
            and "Realization" in right_data
        ):
            left_realization_number = int(left_data.split(" ")[1])
            right_realization_number = int(right_data.split(" ")[1])

            return left_realization_number < right_realization_number

        return super().lessThan(left, right)


class StorageWidget(QWidget):
    onSelectEnsemble = Signal(Ensemble)
    onSelectExperiment = Signal(Experiment)
    onSelectRealization = Signal(Ensemble, int)

    def __init__(
        self, notifier: ErtNotifier, ert_config: ErtConfig, ensemble_size: int
    ):
        QWidget.__init__(self)

        self._notifier = notifier
        self._ert_config = ert_config
        self._ensemble_size = ensemble_size
        self.setMinimumWidth(500)

        self._tree_view = QTreeView(self)
        storage_model = StorageModel(self._notifier.storage)
        notifier.storage_changed.connect(storage_model.reloadStorage)
        notifier.ertChanged.connect(
            lambda: storage_model.reloadStorage(self._notifier.storage)
        )

        search_bar = QLineEdit(self)
        search_bar.setPlaceholderText("Filter")
        proxy_model = _SortingProxyModel(storage_model)
        proxy_model.setFilterKeyColumn(-1)  # Search all columns.
        proxy_model.setSourceModel(storage_model)
        proxy_model.sort(0, Qt.SortOrder.AscendingOrder)

        self._tree_view.setModel(proxy_model)
        search_bar.textChanged.connect(proxy_model.setFilterFixedString)

        selection_model = QItemSelectionModel(proxy_model)
        selection_model.currentChanged.connect(self._current_changed)
        self._tree_view.setSelectionModel(selection_model)
        self._tree_view.setColumnWidth(0, 225)
        self._tree_view.setColumnWidth(1, 125)
        self._tree_view.setColumnWidth(2, 100)

        actions_layout = QHBoxLayout()

        add_button = QToolButton()
        add_button.setIcon(QIcon("img:add_circle_outlined.svg"))
        add_button.setText("New")
        add_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        add_button.clicked.connect(self._add_experiment)
        # for tests
        add_button.setObjectName("add_experiment_tool_button")

        delete_button = QToolButton()
        delete_button.setIcon(QIcon("img:delete_to_trash.svg"))
        delete_button.setText("Delete")
        delete_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        delete_button.clicked.connect(self._delete_experiment)
        # for tests
        delete_button.setObjectName("delete_experiment_tool_button")

        # Only able delete experiments
        self.onSelectExperiment.connect(lambda: delete_button.setEnabled(True))
        self.onSelectEnsemble.connect(lambda: delete_button.setEnabled(False))
        self.onSelectRealization.connect(lambda: delete_button.setEnabled(False))
        delete_button.setEnabled(False)

        actions_layout.addWidget(add_button)
        actions_layout.addWidget(delete_button)
        actions_layout.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(search_bar)
        layout.addWidget(self._tree_view)
        layout.addLayout(actions_layout)

        self.setLayout(layout)

    @Slot(QModelIndex, QModelIndex)
    def _current_changed(self, selected: QModelIndex, previous: QModelIndex) -> None:
        if not previous.isValid() or not selected.isValid():
            return

        idx = self._tree_view.model().mapToSource(selected)  # type: ignore
        cls = idx.internalPointer()

        if isinstance(cls, EnsembleModel):
            ensemble = self._notifier.storage.get_ensemble(cls._id)
            self.onSelectEnsemble.emit(ensemble)
        elif isinstance(cls, ExperimentModel):
            experiment = self._notifier.storage.get_experiment(cls._id)
            self.onSelectExperiment.emit(experiment)
        elif isinstance(cls, RealizationModel):
            ensemble = self._notifier.storage.get_ensemble(cls.ensemble_id)
            self.onSelectRealization.emit(ensemble, cls.realization)

    @Slot()
    def _delete_experiment(self) -> None:
        assert self._tree_view.selectedIndexes()
        model = (
            self._tree_view.model()
            .mapToSource(self._tree_view.selectedIndexes()[0])
            .internalPointer()
        )  # type: ignore
        assert isinstance(model, ExperimentModel)
        print(model._id)

        # self._notifier.storage.delete_experiment()

    @Slot()
    def _add_experiment(self) -> None:
        create_experiment_dialog = CreateExperimentDialog(self._notifier, parent=self)
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
            self._notifier.set_current_ensemble(ensemble)
            self._notifier.ertChanged.emit()
