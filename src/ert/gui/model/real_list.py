from typing import List, Optional, overload

from qtpy.QtCore import (
    QAbstractItemModel,
    QAbstractProxyModel,
    QModelIndex,
    QObject,
    Slot,
)
from typing_extensions import override

from ert.gui.model.snapshot import IsEnsembleRole, IsRealizationRole, NodeRole


class RealListModel(QAbstractProxyModel):
    def __init__(self, parent: Optional[QObject], iter_: int) -> None:
        super().__init__(parent=parent)
        self._iter: int = iter_

    def get_iter(self) -> int:
        return self._iter

    @Slot(int)
    def setIter(self, iter_: int) -> None:
        self._disconnect()
        self.modelAboutToBeReset.emit()
        self._iter = iter_
        self.modelReset.emit()
        self._connect()

    def _disconnect(self) -> None:
        source_model = self.sourceModel()
        if source_model:
            source_model.dataChanged.disconnect(self._source_data_changed)
            source_model.rowsAboutToBeInserted.disconnect(
                self._source_rows_about_to_be_inserted
            )
            source_model.rowsInserted.disconnect(self._source_rows_inserted)
            source_model.modelAboutToBeReset.disconnect(self.modelAboutToBeReset)
            source_model.modelReset.disconnect(self.modelReset)

    def _connect(self) -> None:
        source_model = self.sourceModel()
        if source_model:
            source_model.dataChanged.connect(self._source_data_changed)
            source_model.rowsAboutToBeInserted.connect(
                self._source_rows_about_to_be_inserted
            )
            source_model.rowsInserted.connect(self._source_rows_inserted)
            source_model.modelAboutToBeReset.connect(self.modelAboutToBeReset)
            source_model.modelReset.connect(self.modelReset)

    @override
    def setSourceModel(self, sourceModel: Optional[QAbstractItemModel]) -> None:
        if not sourceModel:
            raise ValueError("need source model")
        self.beginResetModel()
        self._disconnect()
        super().setSourceModel(sourceModel)
        self._connect()
        self.endResetModel()

    @override
    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 1

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        parent = parent if parent else QModelIndex()
        if not parent.isValid():
            source_model = self.sourceModel()
            assert source_model is not None
            iter_index = source_model.index(self._iter, 0, QModelIndex())
            if iter_index.isValid():
                return source_model.rowCount(iter_index)
        return 0

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...
    @overload
    def parent(self) -> Optional[QObject]: ...
    @override
    def parent(self, child: Optional[QModelIndex] = None) -> Optional[QObject]:
        return QModelIndex()

    @override
    def index(
        self, row: int, column: int, parent: Optional[QModelIndex] = None
    ) -> QModelIndex:
        parent = parent if parent else QModelIndex()
        if not parent.isValid():
            real_index = self.mapToSource(self.createIndex(row, 0, parent))
            return self.createIndex(row, column, real_index.data(NodeRole))
        return QModelIndex()

    @override
    def hasChildren(self, parent: Optional[QModelIndex] = None) -> bool:
        # Reimplemented, since in the source model, the realizations have
        # children (i.e. valid indices.). Realizations do not have children in
        # this model.
        parent = parent if parent else QModelIndex()
        if not parent.isValid():
            source_model = self.sourceModel()
            assert source_model is not None
            return source_model.hasChildren(self.mapToSource(parent))
        return False

    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if proxyIndex.isValid():
            sm = self.sourceModel()
            assert sm is not None
            iter_index = sm.index(self._iter, 0, QModelIndex())
            if iter_index.isValid() and sm.hasChildren(iter_index):
                return sm.index(proxyIndex.row(), proxyIndex.column(), iter_index)
        return QModelIndex()

    def mapFromSource(self, src_index: QModelIndex) -> QModelIndex:
        return (
            self.index(src_index.row(), src_index.column(), QModelIndex())
            if src_index.isValid() and self._accept_index(src_index)
            else QModelIndex()
        )

    def _source_data_changed(
        self, top_left: QModelIndex, bottom_right: QModelIndex, roles: List[int]
    ) -> None:
        if top_left.internalPointer() and top_left.data(IsRealizationRole):
            proxy_top_left = self.mapFromSource(top_left)
            proxy_bottom_right = self.mapFromSource(bottom_right)

            if all([proxy_top_left.isValid(), proxy_bottom_right.isValid()]):
                self.dataChanged.emit(proxy_top_left, proxy_bottom_right, roles)

    def _source_rows_about_to_be_inserted(
        self, parent: QModelIndex, start: int, end: int
    ) -> None:
        if parent.isValid() and self._accept_index(parent):
            self.beginInsertRows(self.mapFromSource(parent), start, end)

    def _source_rows_inserted(
        self, parent: QModelIndex, _start: int, _end: int
    ) -> None:
        if parent.isValid() and self._accept_index(parent):
            self.endInsertRows()

    def _accept_index(self, index: QModelIndex) -> bool:
        # If the index under test isn't a realization, it is of no interest as
        # this model should only consist of realization indices.
        if not index.internalPointer() or not index.data(IsRealizationRole):
            return False

        # traverse upwards the tree, checking whether this index is accepted or not
        while index.isValid() and index.internalPointer():
            if index.data(IsEnsembleRole) and index.row() != self._iter:
                return False
            index = index.parent()
        return True
