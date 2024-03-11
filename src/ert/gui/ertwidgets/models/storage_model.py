from enum import IntEnum
from typing import Any, List

import humanize
from qtpy.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
    Slot,
)
from qtpy.QtWidgets import QApplication

from ert.storage import Ensemble, Experiment, Storage


class _Column(IntEnum):
    NAME = 0
    TIME = 1
    TYPE = 2
    UUID = 3


_NUM_COLUMNS = max(_Column).value + 1
_COLUMN_TEXT = {
    0: "Name",
    1: "Created at",
    2: "Type",
    3: "ID",
}


class EnsembleModel:
    def __init__(self, ensemble: Ensemble, parent: Any):
        self._parent = parent
        self._name = ensemble.name
        self._id = ensemble.id
        self._start_time = ensemble.started_at

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == _Column.NAME:
                return self._name
            if col == _Column.TIME:
                return humanize.naturaltime(self._start_time)
            if col == _Column.UUID:
                return str(self._id)
        elif role == Qt.ItemDataRole.ToolTipRole:
            if col == _Column.TIME:
                return str(self._start_time)

        return None


class ExperimentModel:
    def __init__(self, experiment: Experiment, parent: Any):
        self._parent = parent
        self._id = experiment.id
        self._name = experiment.name
        self._experiment_type = experiment.metadata.get("ensemble_type")
        self._children: List[EnsembleModel] = []

    def add_ensemble(self, ensemble: EnsembleModel) -> None:
        self._children.append(ensemble)

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == _Column.NAME:
                return self._name
            if col == _Column.TYPE:
                return self._experiment_type or "None"
            if col == _Column.UUID:
                return str(self._id)
        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == _Column.TYPE and not self._experiment_type:
                qapp = QApplication.instance()
                assert isinstance(qapp, QApplication)
                return qapp.palette().mid()

        return None


class StorageModel(QAbstractItemModel):
    def __init__(self, storage: Storage):
        super().__init__(None)
        self._children: List[ExperimentModel] = []
        self._load_storage(storage)

    @Slot(Storage)
    def reloadStorage(self, storage: Storage) -> None:
        self.beginResetModel()
        self._load_storage(storage)
        self.endResetModel()

    @Slot()
    def add_experiment(self, experiment: ExperimentModel) -> None:
        idx = QModelIndex()
        self.beginInsertRows(idx, 0, 0)
        self._children.append(experiment)
        self.endInsertRows()

    def _load_storage(self, storage: Storage) -> None:
        self._children = []
        for experiment in storage.experiments:
            ex = ExperimentModel(experiment, self)
            for ensemble in experiment.ensembles:
                ens = EnsembleModel(ensemble, ex)
                ex.add_ensemble(ens)
            self._children.append(ex)

    def columnCount(self, parent: QModelIndex) -> int:
        return _NUM_COLUMNS

    def rowCount(self, parent: QModelIndex) -> int:
        if parent.isValid():
            if isinstance(parent.internalPointer(), EnsembleModel):
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

    def headerData(self, section: int, orientation: int, role: int) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        return _COLUMN_TEXT[_Column(section)]

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
