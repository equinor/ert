from ert_gui.model.snapshot import COLUMNS, NodeRole
from ert_gui.model.node import Node, NodeType
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

    step_changed = Signal(int, int, int, int)

    @Slot(int, int, int, int)
    def set_step(self, iter_, real, stage, step):
        self._disconnect()
        self.modelAboutToBeReset.emit()
        self._iter = iter_
        self._real = real
        self._stage = stage
        self._step = step
        self.modelReset.emit()
        self._connect()
        self.step_changed.emit(iter_, real, stage, step)

    def _disconnect(self):
        sm = self.sourceModel()
        if sm is None:
            return
        sm.dataChanged.disconnect(self._source_data_changed)
        sm.modelAboutToBeReset.disconnect(self.modelAboutToBeReset)
        sm.modelReset.disconnect(self.modelReset)

    def _connect(self):
        sm = self.sourceModel()
        if sm is None:
            return
        sm.dataChanged.connect(self._source_data_changed)
        sm.modelAboutToBeReset.connect(self.modelAboutToBeReset)
        sm.modelReset.connect(self.modelReset)

    def _get_source_parent_index(self) -> QModelIndex:
        start = self.index(0, 0, QModelIndex())
        if not start.isValid():
            return QModelIndex()
        if start.internalPointer() is None:
            return QModelIndex()
        source_parent = self.mapToSource(start).parent()
        return source_parent

    def setSourceModel(self, sourceModel: QAbstractItemModel) -> None:
        if not sourceModel:
            raise ValueError("need source model")
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
            return COLUMNS[NodeType.STEP][section][0]
        if orientation == Qt.Vertical:
            return section
        return QVariant()

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        source_index = self._get_source_parent_index()
        if not source_index.isValid():
            return 0
        return self.sourceModel().columnCount(source_index)

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        source_index = self._get_source_parent_index()
        if not source_index.isValid():
            return 0
        return self.sourceModel().rowCount(source_index)

    def parent(self, index: QModelIndex):
        return QModelIndex()

    def index(self, row: int, column: int, parent=QModelIndex()) -> QModelIndex:
        if parent.isValid():
            return QModelIndex()
        job_index = self.mapToSource(self.createIndex(row, column, parent))
        ret_index = self.createIndex(row, column, job_index.data(NodeRole))
        return ret_index

    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if not proxyIndex.isValid():
            return QModelIndex()
        sm = self.sourceModel()
        iter_index = sm.index(self._iter, 0, QModelIndex())
        if not iter_index.isValid() or not sm.hasChildren(iter_index):
            return QModelIndex()
        real_index = sm.index(self._real, 0, iter_index)
        if not real_index.isValid() or not sm.hasChildren(real_index):
            return QModelIndex()
        step_index = sm.index(self._step, 0, real_index)
        if not step_index.isValid() or not sm.hasChildren(step_index):
            return QModelIndex()
        job_index = sm.index(proxyIndex.row(), proxyIndex.column(), step_index)
        return job_index

    def mapFromSource(self, sourceIndex: QModelIndex) -> QModelIndex:
        if not sourceIndex.isValid():
            return QModelIndex()
        if not self._index_is_on_our_branch(sourceIndex):
            return QModelIndex()
        source_node = sourceIndex.internalPointer()
        return self.index(source_node.row(), sourceIndex.column(), QModelIndex())

    def _source_data_changed(
        self, top_left: QModelIndex, bottom_right: QModelIndex, roles: typing.List[int]
    ):
        if not self._index_is_on_our_branch(top_left):
            return
        proxy_top_left = self.mapFromSource(top_left)
        proxy_bottom_right = self.mapFromSource(bottom_right)
        if not proxy_top_left.isValid() or not proxy_bottom_right.isValid():
            return
        self.dataChanged.emit(proxy_top_left, proxy_bottom_right, roles)

    def _index_is_on_our_branch(self, index: QModelIndex) -> bool:
        if index.internalPointer() is None:
            return False
        # the tree is only traversed towards the root
        if index.internalPointer().type != NodeType.JOB:
            return False
        while index.isValid() and index.internalPointer() is not None:
            node = index.internalPointer()
            if node.type == NodeType.STEP and node.row() != self._step:
                return False
            elif node.type == NodeType.REAL and node.row() != self._real:
                return False
            elif node.type == NodeType.ITER and node.row() != self._iter:
                return False
            index = index.parent()
        return True

    def get_iter(self):
        return self._iter

    def get_real(self):
        return self._real
