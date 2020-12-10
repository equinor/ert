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
    QPersistentModelIndex,
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
        sm.modelAboutToBeReset.disconnect(self._source_model_about_to_be_reset)
        sm.modelReset.disconnect(self._source_model_reset)

    def _connect(self):
        sm = self.sourceModel()
        if sm is None:
            return
        sm.dataChanged.connect(self._source_data_changed)
        sm.rowsAboutToBeInserted.connect(self._source_rows_about_to_be_inserted)
        sm.rowsInserted.connect(self._source_rows_inserted)
        sm.modelAboutToBeReset.connect(self._source_model_about_to_be_reset)
        sm.modelReset.connect(self._source_model_reset)

    def setSourceModel(self, sourceModel: QAbstractItemModel) -> None:
        if not sourceModel:
            raise Exception("need source model")
        self.beginResetModel()
        self._disconnect()
        super().setSourceModel(sourceModel)
        self._connect()
        self.endResetModel()

    def columnCount(self, parent: QModelIndex) -> int:
        return 1

    def rowCount(self, parent: QModelIndex) -> int:
        iter_index = self.sourceModel().index(self._iter, 0, QModelIndex())
        if iter_index.isValid():
            return len(iter_index.internalPointer().children)
        return 0

    def parent(self, index):
        return QModelIndex()

    def index(self, row: int, column: int, parent: QModelIndex) -> QModelIndex:
        if parent.isValid():
            return QModelIndex()
        real_index = self.mapToSource(self.createIndex(row, column, parent))
        ret_index = self.createIndex(row, column, real_index.data(NodeRole))
        return ret_index

    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if not proxyIndex.isValid():
            return QModelIndex()
        sm = self.sourceModel()
        iter_index = sm.index(self._iter, proxyIndex.column(), QModelIndex())
        if not iter_index.isValid() or not sm.hasChildren(iter_index):
            return QModelIndex()
        real_index = sm.index(proxyIndex.row(), proxyIndex.column(), iter_index)
        if not real_index.isValid():
            return QModelIndex()
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
        # TODO: map before emit
        self.dataChanged.emit(
            self.mapFromSource(top_left), self.mapFromSource(bottom_right), roles
        )

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

    def _source_model_about_to_be_reset(self):
        self.modelAboutToBeReset.emit()

    def _source_model_reset(self):
        self.modelReset.emit()

    def _index_is_on_our_branch(self, index: QModelIndex) -> bool:
        # # the tree is only traversed towards the root
        if index.internalPointer().type not in (NodeType.ITER, NodeType.REAL):
            return False
        while index.isValid() and index.internalPointer() is not None:
            node = index.internalPointer()
            if node.type == NodeType.ITER and node.row() != self._iter:
                return False
            index = index.parent()
        return True
