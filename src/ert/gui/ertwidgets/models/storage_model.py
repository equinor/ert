from typing import Any, List

from qtpy.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
    Slot,
)

from ert.storage import EnsembleReader, ExperimentReader, StorageReader


class Ensemble:
    def __init__(self, ensemble: EnsembleReader, parent: Any):
        self._parent = parent
        self._name = ensemble.name
        self._id = ensemble.id
        self._start_time = ensemble.started_at

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return f"{self._name} - {self._start_time} ({self._id})"

        return None


class Experiment:
    def __init__(self, experiment: ExperimentReader, parent: Any):
        self._parent = parent
        self._id = experiment.id
        self._name = experiment.name
        self._experiment_type = (
            experiment.simulation_arguments["analysis_module"]
            if "analysis_module" in experiment.simulation_arguments
            else ""
        )
        self._children: List[Ensemble] = []

    def add_ensemble(self, ensemble: Ensemble) -> None:
        self._children.append(ensemble)

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return f"{self._name} - {self._experiment_type} ({self._id})"

        return None


class StorageModel(QAbstractItemModel):
    def __init__(self, storage: StorageReader):
        super().__init__(None)
        self._children: List[Experiment] = []
        self._load_storage(storage)

    @Slot(StorageReader)
    def reloadStorage(self, storage: StorageReader) -> None:
        self.beginResetModel()
        self._load_storage(storage)
        self.endResetModel()

    @Slot()
    def add_experiment(self, experiment: Experiment) -> None:
        idx = QModelIndex()
        self.beginInsertRows(idx, 0, 0)
        self._children.append(experiment)
        self.endInsertRows()

    def _load_storage(self, storage: StorageReader) -> None:
        self._children = []
        for experiment in storage.experiments:
            ex = Experiment(experiment, self)
            for ensemble in experiment.ensembles:
                ens = Ensemble(ensemble, ex)
                ex.add_ensemble(ens)
            self._children.append(ex)

    def columnCount(self, parent: QModelIndex) -> int:
        return 1

    def rowCount(self, parent: QModelIndex) -> int:
        if parent.isValid():
            if isinstance(parent.internalPointer(), Ensemble):
                return 0
            return len(parent.internalPointer()._children)
        else:
            return len(self._children)

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parentItem = child_item._parent

        if parentItem == self:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        return index.internalPointer().data(index, role)

    def index(self, row: int, column: int, parent: QModelIndex) -> QModelIndex:
        parentItem = parent.internalPointer() if parent.isValid() else self
        try:
            childItem = parentItem._children[row]
        except KeyError:
            childItem = None
        if childItem:
            return self.createIndex(row, column, childItem)
        return QModelIndex()
