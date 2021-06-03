import logging

import ert_shared.status.entity.state as state
from ert_gui.model.node import Node, NodeType, snapshot_to_tree
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    Snapshot,
    SnapshotDict,
)
from ert_shared.status.utils import byte_with_unit
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt, QVariant
from qtpy.QtGui import QColor

logger = logging.getLogger(__name__)


NodeRole = Qt.UserRole + 1
RealJobColorHint = Qt.UserRole + 2
RealStatusColorHint = Qt.UserRole + 3
RealLabelHint = Qt.UserRole + 4
ProgressRole = Qt.UserRole + 5
FileRole = Qt.UserRole + 6
RealIens = Qt.UserRole + 7

STEP_COLUMN_NAME = "Name"
STEP_COLUMN_ERROR = "Error"
STEP_COLUMN_STATUS = "Status"
STEP_COLUMN_START_TIME = "Start time"
STEP_COLUMN_END_TIME = "End time"
STEP_COLUMN_STDOUT = "STDOUT"
STEP_COLUMN_STDERR = "STDERR"
STEP_COLUMN_CURRENT_MEMORY_USAGE = "Current Memory Usage"
STEP_COLUMN_MAX_MEMORY_USAGE = "Max Memory Usage"


COLUMNS = {
    NodeType.ROOT: ["Name", "Status"],
    NodeType.ITER: ["Name", "Status", "Active"],
    NodeType.REAL: ["Name", "Status"],
    NodeType.STEP: [
        (STEP_COLUMN_NAME, ids.NAME),
        (STEP_COLUMN_ERROR, ids.ERROR),
        (STEP_COLUMN_STATUS, ids.STATUS),
        (STEP_COLUMN_START_TIME, ids.START_TIME),
        (STEP_COLUMN_END_TIME, ids.END_TIME),
        (STEP_COLUMN_STDOUT, ids.STDOUT),
        (STEP_COLUMN_STDERR, ids.STDERR),
        (STEP_COLUMN_CURRENT_MEMORY_USAGE, ids.CURRENT_MEMORY_USAGE),
        (STEP_COLUMN_MAX_MEMORY_USAGE, ids.MAX_MEMORY_USAGE),
    ],
    NodeType.JOB: [],
}


class SnapshotModel(QAbstractItemModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root = Node(None, {}, NodeType.ROOT)

    def _add_partial_snapshot(self, partial: PartialSnapshot, iter_: int):
        partial_dict = partial.to_dict()
        partial_s = SnapshotDict(**partial_dict)
        if iter_ not in self.root.children:
            logger.debug("no full snapshot yet, bailing")
            return
        iter_index = self.index(iter_, 0, QModelIndex())
        iter_node = self.root.children[iter_]
        if not partial_s.reals:
            logger.debug(f"no realizations in partial for iter {iter_}")
            return
        for real_id in sorted(partial_s.reals, key=int):
            real = partial_s.reals[real_id]
            real_node = iter_node.children[real_id]
            if real.status:
                real_node.data[ids.STATUS] = real.status

            real_index = self.index(real_node.row(), 0, iter_index)
            real_index_bottom_right = self.index(
                real_node.row(), self.columnCount(iter_index) - 1, iter_index
            )

            if not real.steps:
                continue

            for step_id, step in real.steps.items():
                step_node = real_node.children[step_id]
                if step.status:
                    step_node.data[ids.STATUS] = step.status

                step_index = self.index(step_node.row(), 0, real_index)
                step_index_bottom_right = self.index(
                    step_node.row(), self.columnCount(real_index) - 1, real_index
                )

                if not step.jobs:
                    continue

                for job_id in sorted(step.jobs, key=int):
                    job = step.jobs[job_id]
                    job_node = step_node.children[job_id]

                    if job.status:
                        job_node.data[ids.STATUS] = job.status
                    if job.start_time:
                        job_node.data[ids.START_TIME] = job.start_time
                    if job.end_time:
                        job_node.data[ids.END_TIME] = job.end_time
                    if job.stdout:
                        job_node.data[ids.STDOUT] = job.stdout
                    if job.stderr:
                        job_node.data[ids.STDERR] = job.stderr

                    # Errors may be unset as the queue restarts the job
                    job_node.data[ids.ERROR] = job.error if job.error else ""

                    for attr in (ids.CURRENT_MEMORY_USAGE, ids.MAX_MEMORY_USAGE):
                        if job.data and attr in job.data:
                            job_node.data[ids.DATA][attr] = job.data.get(attr)

                    job_index = self.index(job_node.row(), 0, step_index)
                    job_index_bottom_right = self.index(
                        job_node.row(), self.columnCount() - 1, step_index
                    )
                    self.dataChanged.emit(job_index, job_index_bottom_right)
                self.dataChanged.emit(step_index, step_index_bottom_right)
            self.dataChanged.emit(real_index, real_index_bottom_right)
            # TODO: there is no check that any of the data *actually* changed
            # https://github.com/equinor/ert/issues/1374

        top_left = self.index(0, 0, iter_index)
        bottom_right = self.index(0, 1, iter_index)
        self.dataChanged.emit(top_left, bottom_right)

    def _add_snapshot(self, snapshot: Snapshot, iter_: int):
        snapshot_tree = snapshot_to_tree(snapshot, iter_)
        if iter_ in self.root.children:
            self.modelAboutToBeReset.emit()
            self.root.children[iter_] = snapshot_tree
            snapshot_tree.parent = self.root
            self.modelReset.emit()
            return

        parent = QModelIndex()
        next_iter = len(self.root.children)
        self.beginInsertRows(parent, next_iter, next_iter)
        self.root.add_child(snapshot_tree)
        self.root.children[iter_] = snapshot_tree
        self.rowsInserted.emit(parent, snapshot_tree.row(), snapshot_tree.row())

    def columnCount(self, parent=QModelIndex()):
        parent_node = parent.internalPointer()
        if parent_node is None:
            return len(COLUMNS[NodeType.ROOT])
        return len(COLUMNS[parent_node.type])

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()

        if parent.column() > 0:
            return 0

        return len(parentItem.children)

    def parent(self, index: QModelIndex):
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        if not hasattr(child_item, "parent"):
            raise ValueError(
                f"index r{index.row()}/c{index.column()} pointed to parent-less item {child_item}"
            )
        parentItem = child_item.parent

        if parentItem == self.root:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        node = index.internalPointer()

        if role == NodeRole:
            return node

        if node.type == NodeType.JOB:
            return self._job_data(index, node, role)
        elif node.type == NodeType.REAL:
            return self._real_data(index, node, role)

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return f"{node.type}:{node.id}"
            if index.column() == 1:
                return f"{node.data['status']}"

        return QVariant()

    def _real_data(self, index: QModelIndex, node: Node, role: int):
        if role == RealJobColorHint:
            colors = []
            assert node.type == NodeType.REAL
            for step in node.children.values():
                for job_id in sorted(step.children.keys(), key=int):
                    status = step.children[job_id].data[ids.STATUS]
                    color = state.JOB_STATE_TO_COLOR[status]
                    colors.append(QColor(*color))
            return colors
        elif role == RealLabelHint:
            return str(node.id)
        elif role == RealIens:
            return int(node.id)
        elif role == RealStatusColorHint:
            return QColor(*state.REAL_STATE_TO_COLOR[node.data[ids.STATUS]])
        else:
            return QVariant()

    def _job_data(self, index: QModelIndex, node: Node, role: int):
        if role == Qt.BackgroundRole:
            return QColor(*state.REAL_STATE_TO_COLOR[node.data.get(ids.STATUS)])
        if role == Qt.DisplayRole:
            _, data_name = COLUMNS[NodeType.STEP][index.column()]
            if data_name in [ids.CURRENT_MEMORY_USAGE, ids.MAX_MEMORY_USAGE]:
                data = node.data.get(ids.DATA)
                bytes = data.get(data_name) if data else None
                if bytes:
                    return byte_with_unit(bytes)
            if data_name in [ids.STDOUT, ids.STDERR]:
                return "OPEN" if node.data.get(data_name) else QVariant()
            if data_name in [ids.START_TIME, ids.END_TIME]:
                _time = node.data.get(data_name)
                if _time is not None:
                    return str(_time)
                return QVariant()
            return node.data.get(data_name)
        if role == FileRole:
            _, data_name = COLUMNS[NodeType.STEP][index.column()]
            if data_name in [ids.STDOUT, ids.STDERR]:
                return (
                    node.data.get(data_name) if node.data.get(data_name) else QVariant()
                )
        if role == Qt.ToolTipRole:
            _, data_name = COLUMNS[NodeType.STEP][index.column()]
            if data_name in [ids.ERROR, ids.START_TIME, ids.END_TIME]:
                data = node.data.get(data_name)
                if data is not None:
                    return str(data)

        return QVariant()

    def index(self, row: int, column: int, parent=QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()

        childItem = None
        try:
            childItem = list(parentItem.children.values())[row]
        except KeyError:
            return QModelIndex()
        else:
            return self.createIndex(row, column, childItem)

    def reset(self):
        self.modelAboutToBeReset.emit()
        self.root = Node(None, {}, NodeType.ROOT)
        self.modelReset.emit()
