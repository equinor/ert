from typing import Any, overload

from PyQt6.QtCore import (
    QAbstractItemModel,
    QAbstractProxyModel,
    QModelIndex,
    QObject,
    Qt,
)
from PyQt6.QtCore import pyqtSlot as Slot
from typing_extensions import override

from ert.ensemble_evaluator import identifiers as ids
from ert.gui.model.snapshot import (
    FM_STEP_COLUMN_SIZE,
    FM_STEP_COLUMNS,
    IsEnsembleRole,
    IsFMStepRole,
    IsRealizationRole,
    NodeRole,
)


class FMStepListProxyModel(QAbstractProxyModel):
    """This proxy model presents two-dimensional views (row-column) of
    forward model data for a specific realization in a specific iteration."""

    def __init__(self, parent: QObject | None, iter_: int, real_: int) -> None:
        super().__init__(parent=parent)
        self._iter = iter_
        self._real = real_

    @Slot(int, int)
    def set_real(self, iter_: int, real: int) -> None:
        """Called when the user clicks a specific realization in
        the run_dialog window."""
        self._disconnect()
        self.modelAboutToBeReset.emit()
        self._iter = iter_
        self._real = real
        self.modelReset.emit()
        self._connect()

    def _disconnect(self) -> None:
        source_model = self.sourceModel()
        if source_model is None:
            return
        source_model.dataChanged.disconnect(self._source_data_changed)
        source_model.modelAboutToBeReset.disconnect(self.modelAboutToBeReset)
        source_model.modelReset.disconnect(self.modelReset)

    def _connect(self) -> None:
        source_model = self.sourceModel()
        if source_model is None:
            return
        source_model.dataChanged.connect(self._source_data_changed)
        source_model.modelAboutToBeReset.connect(self.modelAboutToBeReset)
        source_model.modelReset.connect(self.modelReset)

    def _get_source_parent_index(self) -> QModelIndex:
        start = self.index(0, 0, QModelIndex())
        if not start.isValid():
            return QModelIndex()
        if start.internalPointer() is None:
            return QModelIndex()
        source_parent = self.mapToSource(start).parent()
        return source_parent

    @override
    def setSourceModel(self, sourceModel: QAbstractItemModel | None) -> None:
        if not sourceModel:
            raise ValueError("need source model")
        self.beginResetModel()
        self._disconnect()
        super().setSourceModel(sourceModel)
        self._connect()
        self.endResetModel()

    @override
    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                header = FM_STEP_COLUMNS[section]
                if header in {ids.STDOUT, ids.STDERR}:
                    return header.upper()
                elif header == ids.MAX_MEMORY_USAGE:
                    header = header.replace("_", " ")
                return header.capitalize()
            if orientation == Qt.Orientation.Vertical:
                return section
        return None

    @override
    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return FM_STEP_COLUMN_SIZE

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        parent = parent or QModelIndex()
        if not parent.isValid():
            source_model = self.sourceModel()
            assert source_model is not None
            source_index = self._get_source_parent_index()
            if source_index.isValid():
                return source_model.rowCount(source_index)
        return 0

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...
    @overload
    def parent(self) -> QObject: ...
    @override
    def parent(self, child: QModelIndex | None = None) -> QObject | QModelIndex:
        return QModelIndex()

    @override
    def index(
        self, row: int, column: int, parent: QModelIndex | None = None
    ) -> QModelIndex:
        parent = parent or QModelIndex()
        if not parent.isValid():
            job_index = self.mapToSource(self.createIndex(row, column, parent))
            return self.createIndex(row, column, job_index.data(NodeRole))
        return QModelIndex()

    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if proxyIndex.isValid():
            sm = self.sourceModel()
            assert sm is not None
            iter_index = sm.index(self._iter, 0, QModelIndex())
            if iter_index.isValid() and sm.hasChildren(iter_index):
                real_index = sm.index(self._real, 0, iter_index)
                if real_index.isValid() and sm.hasChildren(real_index):
                    return sm.index(proxyIndex.row(), proxyIndex.column(), real_index)
        return QModelIndex()

    def mapFromSource(self, sourceIndex: QModelIndex) -> QModelIndex:
        return (
            self.index(sourceIndex.row(), sourceIndex.column(), QModelIndex())
            if sourceIndex.isValid() and self._accept_index(sourceIndex)
            else QModelIndex()
        )

    def _source_data_changed(
        self, top_left: QModelIndex, bottom_right: QModelIndex, roles: list[int]
    ) -> None:
        if self._accept_index(top_left):
            proxy_top_left = self.mapFromSource(top_left)
            proxy_bottom_right = self.mapFromSource(bottom_right)
            if all([proxy_top_left.isValid(), proxy_bottom_right.isValid()]):
                self.dataChanged.emit(proxy_top_left, proxy_bottom_right, roles)

    def _accept_index(self, index: QModelIndex) -> bool:
        if not index.internalPointer() or not index.data(IsFMStepRole):
            return False

        # traverse upwards and check real and iter against parents of this index
        while index.isValid() and index.internalPointer():
            if (index.data(IsRealizationRole) and (index.row() != self._real)) or (
                index.data(IsEnsembleRole) and (index.row() != self._iter)
            ):
                return False
            index = index.parent()
        return True

    def get_iter(self) -> int:
        return self._iter

    def get_real(self) -> int:
        return self._real
