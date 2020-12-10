from ert_gui.model.snapshot import NodeRole
from ert_gui.model.node import NodeType
import typing
from qtpy.QtCore import (
    QObject,
    Qt,
    QAbstractItemModel,
    QAbstractProxyModel,
    QModelIndex,
    QPersistentModelIndex,
    QVariant,
)


class JobListProxyModel(QAbstractProxyModel):
    def __init__(
        self,
        parent: typing.Optional[QObject],
        iter_: int,
        real: int,
        stage: int,
        step: int,
    ) -> None:
        super().__init__(parent=parent)
        self._iter = iter_
        self._real = real
        self._stage = stage
        self._step = step

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

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int
    ) -> typing.Any:
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            # corresponds to the structure in _job_data in main model
            if section == 0:
                return "Name"
            elif section == 1:
                return "Status"
            elif section == 2:
                return "Start time"
            elif section == 3:
                return "End time"
            elif section == 4:
                return "Current Memory Usage"
            elif section == 5:
                return "Max Memory Usage"
        else:
            return QVariant()

    def columnCount(self, parent: QModelIndex) -> int:
        return 6

    def rowCount(self, parent: QModelIndex) -> int:
        start = self.index(0, 0, QModelIndex())
        if not start.isValid():
            return 0
        if start.internalPointer() is None:
            return 0
        return len(start.internalPointer().parent.children)

    def parent(self, index):
        return QModelIndex()

    def index(self, row: int, column: int, parent: QModelIndex) -> QModelIndex:
        if parent.isValid():
            return QModelIndex()
        job_index = self.mapToSource(self.createIndex(row, column, parent))
        ret_index = self.createIndex(row, column, job_index.data(NodeRole))
        return ret_index

    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if not proxyIndex.isValid():
            return QModelIndex()
        sm = self.sourceModel()
        iter_index = sm.index(self._iter, proxyIndex.column(), QModelIndex())
        if not iter_index.isValid() or not sm.hasChildren(iter_index):
            return QModelIndex()
        real_index = sm.index(self._real, proxyIndex.column(), iter_index)
        if not real_index.isValid() or not sm.hasChildren(real_index):
            return QModelIndex()
        stage_index = sm.index(self._stage, proxyIndex.column(), real_index)
        if not stage_index.isValid() or not sm.hasChildren(stage_index):
            return QModelIndex()
        step_index = sm.index(self._step, proxyIndex.column(), stage_index)
        if not step_index.isValid() or not sm.hasChildren(step_index):
            return QModelIndex()
        job_index = sm.index(proxyIndex.row(), proxyIndex.column(), step_index)
        return job_index

    def mapFromSource(self, sourceIndex: QModelIndex) -> QModelIndex:
        if not sourceIndex.isValid():
            return QModelIndex()
        source_node = sourceIndex.internalPointer()
        if source_node is None or source_node.type != NodeType.JOB:
            return QModelIndex()

        if not self._index_is_on_our_branch(sourceIndex.parent()):
            return QModelIndex()
        return self.index(source_node.row(), sourceIndex.column(), QModelIndex())

    def _source_data_changed(
        self, top_left: QModelIndex, bottom_right: QModelIndex, roles: typing.List[int]
    ):
        proxy_top_left = self.mapFromSource(top_left)
        proxy_bottom_right = self.mapFromSource(bottom_right)
        if not proxy_top_left.isValid() or not proxy_bottom_right.isValid():
            return
        self.dataChanged.emit(
            proxy_top_left, proxy_bottom_right, roles
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
        if index.internalPointer().type not in (NodeType.STEP, NodeType.JOB):
            return False
        while index.isValid() and index.internalPointer() is not None:
            node = index.internalPointer()
            if node.type == NodeType.STEP and node.row() != self._step:
                return False
            elif node.type == NodeType.STAGE and node.row() != self._stage:
                return False
            elif node.type == NodeType.REAL and node.row() != self._real:
                return False
            elif node.type == NodeType.ITER and node.row() != self._iter:
                return False
            index = index.parent()
        return True
