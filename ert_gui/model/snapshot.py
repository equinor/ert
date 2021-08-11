import logging
from contextlib import ExitStack
from typing import List, Union

import ert_shared.status.entity.state as state
import pyrsistent
from ert_gui.model.node import Node, NodeType
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, Snapshot
from ert_shared.status.utils import byte_with_unit
from qtpy.QtCore import QAbstractItemModel, QModelIndex, QSize, Qt, QVariant
from qtpy.QtGui import QColor, QFont

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

SORTED_REALIZATION_IDS = "_sorted_real_ids"
SORTED_JOB_IDS = "_sorted_job_ids"
REAL_JOB_STATUS_AGGREGATED = "_aggr_job_status_colors"
REAL_STATUS_COLOR = "_real_status_colors"

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

_QCOLORS = {
    state.COLOR_WAITING: QColor(*state.COLOR_WAITING),
    state.COLOR_PENDING: QColor(*state.COLOR_PENDING),
    state.COLOR_RUNNING: QColor(*state.COLOR_RUNNING),
    state.COLOR_FAILED: QColor(*state.COLOR_FAILED),
    state.COLOR_UNKNOWN: QColor(*state.COLOR_UNKNOWN),
    state.COLOR_FINISHED: QColor(*state.COLOR_FINISHED),
    state.COLOR_NOT_ACTIVE: QColor(*state.COLOR_NOT_ACTIVE),
}


class SnapshotModel(QAbstractItemModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root = Node(None, {}, NodeType.ROOT)

    @staticmethod
    def prerender(
        snapshot: Union[Snapshot, PartialSnapshot]
    ) -> Union[Snapshot, PartialSnapshot]:
        """Pre-render some data that is required by this model. Ideally, this
        is called outside the GUI thread. This is a requirement of the model,
        so it has to be called."""

        # If there are no realizations, there's nothing to prerender.
        if not snapshot.data().get(ids.REALS):
            return

        metadata = {
            # A mapping from real to job to that job's QColor status representation
            REAL_JOB_STATUS_AGGREGATED: {},
            # A mapping from real to that real's QColor status representation
            REAL_STATUS_COLOR: {},
        }
        if isinstance(snapshot, Snapshot):
            metadata[SORTED_REALIZATION_IDS] = sorted(
                snapshot.data()[ids.REALS].keys(), key=int
            )
            for real_id, real in snapshot.data()[ids.REALS].items():
                for step in real[ids.STEPS].values():
                    metadata[SORTED_JOB_IDS] = sorted(step[ids.JOBS].keys(), key=int)
                    break
                break

        for real_id, real in snapshot.data()[ids.REALS].items():
            if real.get(ids.STATUS):
                metadata[REAL_STATUS_COLOR][real_id] = _QCOLORS[
                    state.REAL_STATE_TO_COLOR[real[ids.STATUS]]
                ]
            metadata[REAL_JOB_STATUS_AGGREGATED][real_id] = {}
            if real.get(ids.STEPS):
                for step in real[ids.STEPS].values():
                    if not ids.JOBS in step:
                        continue
                    for job_id in sorted(step[ids.JOBS].keys(), key=int):
                        status = step[ids.JOBS][job_id][ids.STATUS]
                        color = _QCOLORS[state.JOB_STATE_TO_COLOR[status]]
                        metadata[REAL_JOB_STATUS_AGGREGATED][real_id][job_id] = color

        if isinstance(snapshot, Snapshot):
            snapshot.merge_metadata(metadata)
        elif isinstance(snapshot, PartialSnapshot):
            snapshot.update_metadata(metadata)
        return snapshot

    def _add_partial_snapshot(self, partial: PartialSnapshot, iter_: int):
        metadata = partial.data().get(ids.METADATA)
        if not metadata:
            logger.debug("no metadata in partial, ignoring partial")
            return

        if iter_ not in self.root.children:
            logger.debug("no full snapshot yet, ignoring partial")
            return

        if not partial.data().get(ids.REALS):
            logger.debug(f"no realizations in partial for iter {iter_}")
            return

        # Stack onto which we push change events for entities, since we branch
        # the code based on what is in the partial. This way we're guaranteed
        # that the change events will be emitted when the stack is unwound.
        with ExitStack() as stack:
            iter_node = self.root.children[iter_]
            iter_index = self.index(iter_node.row(), 0, QModelIndex())
            iter_index_bottom_right = self.index(
                iter_node.row(), iter_index.column(), QModelIndex()
            )
            stack.callback(self.dataChanged.emit, iter_index, iter_index_bottom_right)

            for real_id in iter_node.data[SORTED_REALIZATION_IDS]:
                real = partial.data()[ids.REALS].get(real_id)
                if not real:
                    continue
                real_node = iter_node.children[real_id]
                if real.get(ids.STATUS):
                    real_node.data[ids.STATUS] = real[ids.STATUS]

                real_index = self.index(real_node.row(), 0, iter_index)
                real_index_bottom_right = self.index(
                    real_node.row(), self.columnCount(iter_index) - 1, iter_index
                )
                stack.callback(
                    self.dataChanged.emit, real_index, real_index_bottom_right
                )

                for job_id, color in (
                    metadata[REAL_JOB_STATUS_AGGREGATED].get(real_id, {}).items()
                ):
                    real_node.data[REAL_JOB_STATUS_AGGREGATED][job_id] = color
                if real_id in metadata[REAL_STATUS_COLOR]:
                    real_node.data[REAL_STATUS_COLOR] = metadata[REAL_STATUS_COLOR][
                        real_id
                    ]

                if not real.get(ids.STEPS):
                    continue

                for step_id, step in real[ids.STEPS].items():
                    step_node = real_node.children[step_id]
                    if step.get(ids.STATUS):
                        step_node.data[ids.STATUS] = step[ids.STATUS]

                    step_index = self.index(step_node.row(), 0, real_index)

                    if not step.get(ids.JOBS):
                        continue

                    for job_id, job in step[ids.JOBS].items():
                        job_node = step_node.children[job_id]

                        job_index = self.index(job_node.row(), 0, step_index)
                        job_index_bottom_right = self.index(
                            job_node.row(), self.columnCount() - 1, step_index
                        )
                        stack.callback(
                            self.dataChanged.emit, job_index, job_index_bottom_right
                        )

                        if job.get(ids.STATUS):
                            job_node.data[ids.STATUS] = job[ids.STATUS]
                        if job.get(ids.START_TIME):
                            job_node.data[ids.START_TIME] = job[ids.START_TIME]
                        if job.get(ids.END_TIME):
                            job_node.data[ids.END_TIME] = job[ids.END_TIME]
                        if job.get(ids.STDOUT):
                            job_node.data[ids.STDOUT] = job[ids.STDOUT]
                        if job.get(ids.STDERR):
                            job_node.data[ids.STDERR] = job[ids.STDERR]

                        # Errors may be unset as the queue restarts the job
                        job_node.data[ids.ERROR] = (
                            job[ids.ERROR] if job.get(ids.ERROR) else ""
                        )

                        for attr in (ids.CURRENT_MEMORY_USAGE, ids.MAX_MEMORY_USAGE):
                            if job.get(ids.DATA) and attr in job.get(ids.DATA):
                                job_node.data[ids.DATA] = job_node.data[ids.DATA].set(
                                    attr, job.get(ids.DATA).get(attr)
                                )

    def _add_snapshot(self, snapshot: Snapshot, iter_: int):
        # Parts of the metadata will be used in the underlying data model,
        # which is be mutable, hence we thaw it here—once.
        metadata = pyrsistent.thaw(snapshot.data()[ids.METADATA])
        snapshot_tree = Node(
            iter_,
            {
                ids.STATUS: snapshot.data()[ids.STATUS],
                SORTED_REALIZATION_IDS: metadata[SORTED_REALIZATION_IDS],
                SORTED_JOB_IDS: metadata[SORTED_JOB_IDS],
            },
            NodeType.ITER,
        )
        for real_id in snapshot_tree.data[SORTED_REALIZATION_IDS]:
            real = snapshot.data()[ids.REALS][real_id]
            real_node = Node(
                real_id,
                {
                    ids.STATUS: real[ids.STATUS],
                    ids.ACTIVE: real[ids.ACTIVE],
                    REAL_JOB_STATUS_AGGREGATED: metadata[REAL_JOB_STATUS_AGGREGATED][
                        real_id
                    ],
                    REAL_STATUS_COLOR: metadata[REAL_STATUS_COLOR][real_id],
                },
                NodeType.REAL,
            )
            snapshot_tree.add_child(real_node)
            for step_id, step in real[ids.STEPS].items():
                step_node = Node(step_id, {ids.STATUS: step[ids.STATUS]}, NodeType.STEP)
                real_node.add_child(step_node)
                for job_id in metadata[SORTED_JOB_IDS]:
                    job = step[ids.JOBS][job_id]
                    job_dict = dict(job)
                    job_dict[ids.DATA] = job.data
                    job_node = Node(job_id, job_dict, NodeType.JOB)
                    step_node.add_child(job_node)

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

        if role in (Qt.StatusTipRole, Qt.WhatsThisRole, Qt.ToolTipRole):
            return ""

        if role == Qt.SizeHintRole:
            return QSize()

        if role == Qt.FontRole:
            return QFont()

        if role in (Qt.BackgroundRole, Qt.ForegroundRole, Qt.DecorationRole):
            return QColor()

        return QVariant()

    def _real_data(self, index: QModelIndex, node: Node, role: int):
        if role == RealJobColorHint:
            colors: List[QColor] = []
            for job_id in node.parent.data[SORTED_JOB_IDS]:
                colors.append(node.data[REAL_JOB_STATUS_AGGREGATED][job_id])
            return colors
        elif role == RealLabelHint:
            return node.id
        elif role == RealIens:
            return node.id
        elif role == RealStatusColorHint:
            return node.data[REAL_STATUS_COLOR]
        else:
            return QVariant()

    def _job_data(self, index: QModelIndex, node: Node, role: int):
        if role == Qt.BackgroundRole:
            real = node.parent.parent
            return real.data[REAL_JOB_STATUS_AGGREGATED][node.id]
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
