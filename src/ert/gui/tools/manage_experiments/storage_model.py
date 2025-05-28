from enum import IntEnum
from typing import Any, Self, cast, overload
from uuid import UUID

import humanize
from PyQt6.QtCore import QAbstractItemModel, QModelIndex, QObject, Qt
from PyQt6.QtCore import pyqtSlot as Slot
from typing_extensions import override

from ert.storage import Ensemble, Experiment, Storage


class _Column(IntEnum):
    NAME = 0
    TIME = 1


_NUM_COLUMNS = max(_Column).value + 1
_COLUMN_TEXT = {
    0: "Name",
    1: "Created at",
}


class RealizationModel:
    def __init__(self, realization: int, parent: Any) -> None:
        self._parent = parent
        self._name = f"Realization {realization}"
        self._ensemble_id = parent._id
        self._realization = realization
        self._id = f"{parent._id}_{realization}"

    @property
    def ensemble_id(self) -> UUID:
        return self._ensemble_id

    @property
    def realization(self) -> int:
        return self._realization

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole and col == _Column.NAME:
            return self._name

        return None


class EnsembleModel:
    def __init__(self, ensemble: Ensemble, parent: Any) -> None:
        self._parent = parent
        self._name = ensemble.name
        self._id = ensemble.id
        self._start_time = ensemble.started_at
        self._children: list[RealizationModel] = []

    def add_realization(self, realization: RealizationModel) -> None:
        self._children.append(realization)

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == _Column.NAME:
                return self._name
            if col == _Column.TIME:
                return humanize.naturaltime(self._start_time)
        elif role == Qt.ItemDataRole.ToolTipRole:
            if col == _Column.TIME:
                return str(self._start_time)

        return None


class ExperimentModel:
    def __init__(self, experiment: Experiment, parent: Any) -> None:
        self._parent = parent
        self._id = experiment.id
        self._name = experiment.name
        self._children: list[EnsembleModel] = []

    def add_ensemble(self, ensemble: EnsembleModel) -> None:
        self._children.append(ensemble)

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(
        self, index: QModelIndex, role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == _Column.NAME:
                return self._name
            if col == _Column.TIME:
                return (
                    humanize.naturaltime(self._children[0]._start_time)
                    if self._children
                    else "None"
                )

        return None


ChildModel = ExperimentModel | EnsembleModel | RealizationModel


class StorageModel(QAbstractItemModel):
    def __init__(self, storage: Storage) -> None:
        super().__init__(None)
        self._children: list[ExperimentModel] = []
        self._load_storage(storage)

    @Slot(Storage)
    def reloadStorage(self, storage: Storage) -> None:
        self.beginResetModel()
        self._load_storage(storage)
        self.endResetModel()

    @Slot(ExperimentModel)
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
                for realization in range(ensemble.ensemble_size):
                    ens.add_realization(RealizationModel(realization, ens))

            self._children.append(ex)

    @override
    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return _NUM_COLUMNS

    @override
    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            data = cast(ChildModel | Self, parent.internalPointer())
            return 0 if isinstance(data, RealizationModel) else len(data._children)
        else:
            return len(self._children)

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...
    @overload
    def parent(self) -> QObject: ...
    @override
    def parent(self, child: QModelIndex | None = None) -> QObject | QModelIndex:
        if child is None or not child.isValid():
            return QModelIndex()

        child_item = cast(ChildModel, child.internalPointer())
        parentItem = child_item._parent

        if parentItem == self:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    @override
    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        return _COLUMN_TEXT[_Column(section)]

    @override
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        return cast(ChildModel | Self, index.internalPointer()).data(
            index, Qt.ItemDataRole(role)
        )

    @override
    def index(
        self, row: int, column: int, parent: QModelIndex | None = None
    ) -> QModelIndex:
        if parent is None:
            parent = QModelIndex()

        model = (
            cast(ChildModel | Self, parent.internalPointer())
            if parent.isValid()
            else self
        )
        if type(model) is not RealizationModel:
            model = cast(Self | EnsembleModel | ExperimentModel, model)
            try:
                childItem = model._children[row]
                return self.createIndex(row, column, childItem)
            except KeyError:
                pass
        return QModelIndex()
