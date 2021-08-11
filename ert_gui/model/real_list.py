from ert_gui.model.snapshot import NodeRole
from ert_gui.model.node import NodeType
import typing
from qtpy.QtCore import (
    QObject,
    Qt,
    Signal,
    Slot,
    QAbstractItemModel,
    QAbstractProxyModel,
    QModelIndex,
    QVariant,
)


class RealListModel(QAbstractProxyModel):
    def __init__(
        self,
        parent: typing.Optional[QObject],
        iter_: int,
    ) -> None:
        super().__init__(parent=parent)
        self._iter = iter_

    def get_iter(self):
        return self._iter

    iter_changed = Signal(int)

    @Slot(int)
    def setIter(self, iter_):
        self._disconnect()
        self.modelAboutToBeReset.emit()
        self._iter = iter_
        self.modelReset.emit()
        self._connect()
        self.iter_changed.emit(iter_)

    def _disconnect(self):
        sm = self.sourceModel()
        if sm is None:
            return
        sm.dataChanged.disconnect(self._source_data_changed)
        sm.rowsAboutToBeInserted.disconnect(self._source_rows_about_to_be_inserted)
        sm.rowsInserted.disconnect(self._source_rows_inserted)
        sm.modelAboutToBeReset.disconnect(self.modelAboutToBeReset)
        sm.modelReset.disconnect(self.modelReset)

    def _connect(self):
        sm = self.sourceModel()
        if sm is None:
            return
        sm.dataChanged.connect(self._source_data_changed)
        sm.rowsAboutToBeInserted.connect(self._source_rows_about_to_be_inserted)
        sm.rowsInserted.connect(self._source_rows_inserted)
        sm.modelAboutToBeReset.connect(self.modelAboutToBeReset)
        sm.modelReset.connect(self.modelReset)

    def setSourceModel(self, sourceModel: QAbstractItemModel) -> None:
        if not sourceModel:
            raise ValueError("need source model")
        self.beginResetModel()
        self._disconnect()
        super().setSourceModel(sourceModel)
        self._connect()
        self.endResetModel()

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        iter_index = self.sourceModel().index(self._iter, 0, QModelIndex())
        if not iter_index.isValid():
            return 0
        return self.sourceModel().columnCount(iter_index)

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        iter_index = self.sourceModel().index(self._iter, 0, QModelIndex())
        if not iter_index.isValid():
            return 0
        return self.sourceModel().rowCount(iter_index)

    def parent(self, index: QModelIndex):
        return QModelIndex()

    def index(self, row: int, column: int, parent=QModelIndex()) -> QModelIndex:
        if parent.isValid():
            return QModelIndex()
        real_index = self.mapToSource(self.createIndex(row, 0, parent))
        ret_index = self.createIndex(row, column, real_index.data(NodeRole))
        return ret_index

    def hasChildren(self, parent: QModelIndex) -> bool:
        # Reimplemented, since in the source model, the realizations have
        # children (i.e. valid indices.). Realizations do not have children in
        # this model.
        if parent.isValid():
            return False
        return self.sourceModel().hasChildren(self.mapToSource(parent))

    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if not proxyIndex.isValid():
            return QModelIndex()
        sm = self.sourceModel()
        iter_index = sm.index(self._iter, 0, QModelIndex())
        if not iter_index.isValid() or not sm.hasChildren(iter_index):
            return QModelIndex()
        real_index = sm.index(proxyIndex.row(), proxyIndex.column(), iter_index)
        return real_index

    def mapFromSource(self, sourceIndex: QModelIndex) -> QModelIndex:
        if not sourceIndex.isValid():
            return QModelIndex()
        source_node = sourceIndex.internalPointer()
        if source_node is None:
            return QModelIndex()
        if not self._index_is_on_our_branch(sourceIndex):
            return QModelIndex()
        return self.index(sourceIndex.row(), sourceIndex.column(), QModelIndex())

    def _source_data_changed(
        self, top_left: QModelIndex, bottom_right: QModelIndex, roles: typing.List[int]
    ):
        if top_left.internalPointer() is None:
            return
        if top_left.internalPointer().type != NodeType.REAL:
            return
        proxy_top_left = self.mapFromSource(top_left)
        proxy_bottom_right = self.mapFromSource(bottom_right)
        if not proxy_top_left.isValid() or not proxy_bottom_right.isValid():
            return
        self.dataChanged.emit(proxy_top_left, proxy_bottom_right, roles)

    def _source_rows_about_to_be_inserted(
        self, parent: QModelIndex, start: int, end: int
    ):
        if not parent.isValid():
            return
        if not self._index_is_on_our_branch(parent):
            return
        self.beginInsertRows(self.mapFromSource(parent), start, end)

    def _source_rows_inserted(self, parent: QModelIndex, start: int, end: int):
        if not parent.isValid():
            return
        if not self._index_is_on_our_branch(parent):
            return
        self.endInsertRows()

    def _index_is_on_our_branch(self, index: QModelIndex) -> bool:
        if index.internalPointer() is None:
            return False
        # the tree is only traversed towards the root
        if index.internalPointer().type != NodeType.REAL:
            return False
        while index.isValid() and index.internalPointer() is not None:
            node = index.internalPointer()
            if node.type == NodeType.ITER and node.row() != self._iter:
                return False
            index = index.parent()
        return True
