from typing import Any, List, Optional

from qtpy.QtCore import (
    QAbstractItemModel,
    QAbstractProxyModel,
    QModelIndex,
    QObject,
    Qt,
    QVariant,
    Signal,
    Slot,
)

from ert.gui.model.node import NodeType
from ert.gui.model.snapshot import (
    COLUMNS,
    IsEnsembleRole,
    IsJobRole,
    IsRealizationRole,
    IsStepRole,
    NodeRole,
)


class JobListProxyModel(QAbstractProxyModel):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        parent: Optional[QObject],
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
    def set_step(self, iter_: int, real: int, stage: int, step: int):
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
        source_model = self.sourceModel()
        if source_model is None:
            return
        source_model.dataChanged.disconnect(self._source_data_changed)
        source_model.modelAboutToBeReset.disconnect(self.modelAboutToBeReset)
        source_model.modelReset.disconnect(self.modelReset)

    def _connect(self):
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

    # pylint: disable=invalid-name
    def setSourceModel(self, sourceModel: QAbstractItemModel) -> None:
        if not sourceModel:
            raise ValueError("need source model")
        self.beginResetModel()
        self._disconnect()
        super().setSourceModel(sourceModel)
        self._connect()
        self.endResetModel()

    # pylint: disable=invalid-name, no-self-use
    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.UserRole
    ) -> Any:
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return COLUMNS[NodeType.STEP][section][0]
        if orientation == Qt.Vertical:
            return section
        return QVariant()

    # pylint: disable=invalid-name
    def columnCount(self, parent=None) -> int:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return 0
        source_index = self._get_source_parent_index()
        if not source_index.isValid():
            return 0
        return self.sourceModel().columnCount(source_index)

    def rowCount(self, parent=None) -> int:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return 0
        source_index = self._get_source_parent_index()
        if not source_index.isValid():
            return 0
        return self.sourceModel().rowCount(source_index)

    # pylint: disable=no-self-use
    def parent(self, _index: QModelIndex):
        return QModelIndex()

    def index(self, row: int, column: int, parent=None) -> QModelIndex:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return QModelIndex()
        job_index = self.mapToSource(self.createIndex(row, column, parent))
        ret_index = self.createIndex(row, column, job_index.data(NodeRole))
        return ret_index

    # pylint: disable=invalid-name
    def mapToSource(self, proxyIndex: QModelIndex) -> QModelIndex:
        if not proxyIndex.isValid():
            return QModelIndex()
        source_model = self.sourceModel()
        iter_index = source_model.index(self._iter, 0, QModelIndex())
        if not iter_index.isValid() or not source_model.hasChildren(iter_index):
            return QModelIndex()
        real_index = source_model.index(self._real, 0, iter_index)
        if not real_index.isValid() or not source_model.hasChildren(real_index):
            return QModelIndex()
        step_index = source_model.index(self._step, 0, real_index)
        if not step_index.isValid() or not source_model.hasChildren(step_index):
            return QModelIndex()
        job_index = source_model.index(
            proxyIndex.row(), proxyIndex.column(), step_index
        )
        return job_index

    # pylint: disable=invalid-name
    def mapFromSource(self, sourceIndex: QModelIndex) -> QModelIndex:
        if not sourceIndex.isValid():
            return QModelIndex()
        if not self._accept_index(sourceIndex):
            return QModelIndex()
        return self.index(sourceIndex.row(), sourceIndex.column(), QModelIndex())

    def _source_data_changed(
        self, top_left: QModelIndex, bottom_right: QModelIndex, roles: List[int]
    ):
        if not self._accept_index(top_left):
            return
        proxy_top_left = self.mapFromSource(top_left)
        proxy_bottom_right = self.mapFromSource(bottom_right)
        if not proxy_top_left.isValid() or not proxy_bottom_right.isValid():
            return
        self.dataChanged.emit(proxy_top_left, proxy_bottom_right, roles)

    # pylint: disable=too-many-boolean-expressions
    def _accept_index(self, index: QModelIndex) -> bool:
        if index.internalPointer() is None:
            return False
        # This model should only consist of job indices, so anything else mean
        # the index is not on "our branch" of the state graph.
        if not index.data(IsJobRole):
            return False

        # traverse upwards and check step, real and iter against parents of
        # this index.
        while index.isValid() and index.internalPointer() is not None:
            if (
                (index.data(IsStepRole) and (index.row() != self._step))
                or (index.data(IsRealizationRole) and (index.row() != self._real))
                or (index.data(IsEnsembleRole) and (index.row() != self._iter))
            ):
                return False
            index = index.parent()
        return True

    def get_iter(self):
        return self._iter

    def get_real(self):
        return self._real
