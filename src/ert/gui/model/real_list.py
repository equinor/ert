import typing

from qtpy.QtCore import (
    QAbstractItemModel,
    QAbstractProxyModel,
    QModelIndex,
    QObject,
    Signal,
    Slot,
)

from ert.gui.model.snapshot import IsEnsembleRole, IsRealizationRole, NodeRole


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
    # pylint: disable=invalid-name
    def setIter(self, iter_: int):
        self._disconnect()
        self.modelAboutToBeReset.emit()
        self._iter: int = iter_
        self.modelReset.emit()
        self._connect()
        self.iter_changed.emit(iter_)

    def _disconnect(self):
        source_model = self.sourceModel()
        if source_model is None:
            return
        source_model.dataChanged.disconnect(self._source_data_changed)
        source_model.rowsAboutToBeInserted.disconnect(
            self._source_rows_about_to_be_inserted
        )
        source_model.rowsInserted.disconnect(self._source_rows_inserted)
        source_model.modelAboutToBeReset.disconnect(self.modelAboutToBeReset)
        source_model.modelReset.disconnect(self.modelReset)

    def _connect(self):
        source_model = self.sourceModel()
        if source_model is None:
            return
        source_model.dataChanged.connect(self._source_data_changed)
        source_model.rowsAboutToBeInserted.connect(
            self._source_rows_about_to_be_inserted
        )
        source_model.rowsInserted.connect(self._source_rows_inserted)
        source_model.modelAboutToBeReset.connect(self.modelAboutToBeReset)
        source_model.modelReset.connect(self.modelReset)

    # pylint: disable=invalid-name
    def setSourceModel(self, sourceModel: QAbstractItemModel) -> None:
        if not sourceModel:
            raise ValueError("need source model")
        self.beginResetModel()
        self._disconnect()
        super().setSourceModel(sourceModel)
        self._connect()
        self.endResetModel()

    # pylint: disable=invalid-name
    def columnCount(self, parent: QModelIndex = None) -> int:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return 0
        iter_index = self.sourceModel().index(self._iter, 0, QModelIndex())
        if not iter_index.isValid():
            return 0
        return self.sourceModel().columnCount(iter_index)

    # pylint: disable=invalid-name
    def rowCount(self, parent: QModelIndex = None) -> int:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return 0
        iter_index = self.sourceModel().index(self._iter, 0, QModelIndex())
        if not iter_index.isValid():
            return 0
        return self.sourceModel().rowCount(iter_index)

    # pylint: disable=no-self-use
    def parent(self, _index: QModelIndex):
        return QModelIndex()

    def index(self, row: int, column: int, parent: QModelIndex = None) -> QModelIndex:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return QModelIndex()
        real_index = self.mapToSource(self.createIndex(row, 0, parent))
        ret_index = self.createIndex(row, column, real_index.data(NodeRole))
        return ret_index

    # pylint: disable=invalid-name
    def hasChildren(self, parent: QModelIndex) -> bool:
        # Reimplemented, since in the source model, the realizations have
        # children (i.e. valid indices.). Realizations do not have children in
        # this model.
        if parent.isValid():
            return False
        return self.sourceModel().hasChildren(self.mapToSource(parent))

    # pylint: disable=invalid-name
    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if not proxyIndex.isValid():
            return QModelIndex()
        sm = self.sourceModel()
        iter_index = sm.index(self._iter, 0, QModelIndex())
        if not iter_index.isValid() or not sm.hasChildren(iter_index):
            return QModelIndex()
        real_index = sm.index(proxyIndex.row(), proxyIndex.column(), iter_index)
        return real_index

    # pylint: disable=invalid-name
    def mapFromSource(self, sourceIndex: QModelIndex) -> QModelIndex:
        if not sourceIndex.isValid():
            return QModelIndex()
        source_node = sourceIndex.internalPointer()
        if source_node is None:
            return QModelIndex()
        if not self._accept_index(sourceIndex):
            return QModelIndex()
        return self.index(sourceIndex.row(), sourceIndex.column(), QModelIndex())

    def _source_data_changed(
        self, top_left: QModelIndex, bottom_right: QModelIndex, roles: typing.List[int]
    ):
        if top_left.internalPointer() is None:
            return
        if not top_left.data(IsRealizationRole):
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
        if not self._accept_index(parent):
            return
        self.beginInsertRows(self.mapFromSource(parent), start, end)

    def _source_rows_inserted(self, parent: QModelIndex, _start: int, _end: int):
        if not parent.isValid():
            return
        if not self._accept_index(parent):
            return
        self.endInsertRows()

    def _accept_index(self, index: QModelIndex) -> bool:
        if index.internalPointer() is None:
            return False
        # If the index under test isn't a realization, it is of no interest as
        # this model should only consist of realization indices.
        if not index.data(IsRealizationRole):
            return False

        # traverse upwards the tree, checking whether this index is accepted or
        # not.
        while index.isValid() and index.internalPointer() is not None:
            if index.data(IsEnsembleRole) and index.row() != self._iter:
                return False
            index = index.parent()
        return True
