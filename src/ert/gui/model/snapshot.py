import datetime
import logging
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyrsistent
from dateutil import tz
from qtpy.QtCore import QAbstractItemModel, QModelIndex, QSize, Qt, QVariant
from qtpy.QtGui import QColor, QFont

from ert.ensemble_evaluator import PartialSnapshot, Snapshot
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state
from ert.gui.model.node import Node, NodeType
from ert.shared.status.utils import byte_with_unit

logger = logging.getLogger(__name__)


NodeRole = Qt.UserRole + 1
RealJobColorHint = Qt.UserRole + 2
RealStatusColorHint = Qt.UserRole + 3
RealLabelHint = Qt.UserRole + 4
ProgressRole = Qt.UserRole + 5
FileRole = Qt.UserRole + 6
RealIens = Qt.UserRole + 7

# Indicates what type the underlying data is
IsEnsembleRole = Qt.UserRole + 8
IsRealizationRole = Qt.UserRole + 9
IsStepRole = Qt.UserRole + 10
IsJobRole = Qt.UserRole + 11
StatusRole = Qt.UserRole + 12

STEP_COLUMN_NAME = "Name"
STEP_COLUMN_ERROR = "Error"
STEP_COLUMN_STATUS = "Status"
STEP_COLUMN_DURATION = "Duration"
STEP_COLUMN_STDOUT = "STDOUT"
STEP_COLUMN_STDERR = "STDERR"
STEP_COLUMN_CURRENT_MEMORY_USAGE = "Current memory usage"
STEP_COLUMN_MAX_MEMORY_USAGE = "Max memory usage"

SORTED_REALIZATION_IDS = "_sorted_real_ids"
SORTED_JOB_IDS = "_sorted_job_ids"
REAL_JOB_STATUS_AGGREGATED = "_aggr_job_status_colors"
REAL_STATUS_COLOR = "_real_status_colors"

DURATION = "Duration"

COLUMNS: Dict[NodeType, Sequence[Union[str, Tuple[str, str]]]] = {
    NodeType.ROOT: ["Name", "Status"],
    NodeType.ITER: ["Name", "Status", "Active"],
    NodeType.REAL: ["Name", "Status"],
    NodeType.STEP: [
        (STEP_COLUMN_NAME, ids.NAME),
        (STEP_COLUMN_ERROR, ids.ERROR),
        (STEP_COLUMN_STATUS, ids.STATUS),
        (
            STEP_COLUMN_DURATION,
            DURATION,
        ),  # Duration is based on two data fields, not coming directly from ert
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


def _estimate_duration(start_time, end_time=None):
    timezone = None
    if start_time.tzname() is not None:
        timezone = tz.gettz(start_time.tzname())
    end_time = end_time or datetime.datetime.now(timezone)
    return end_time - start_time


class SnapshotModel(QAbstractItemModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root = Node(0, {}, NodeType.ROOT)

    @staticmethod
    def prerender(
        snapshot: Union[Snapshot, PartialSnapshot]
    ) -> Optional[Union[Snapshot, PartialSnapshot]]:
        """Pre-render some data that is required by this model. Ideally, this
        is called outside the GUI thread. This is a requirement of the model,
        so it has to be called."""

        # If there are no realizations, there's nothing to prerender.
        if not snapshot.data().get(ids.REALS):
            return None

        metadata: Dict[str, Any] = {
            # A mapping from real to job to that job's QColor status representation
            REAL_JOB_STATUS_AGGREGATED: {},
            # A mapping from real to that real's QColor status representation
            REAL_STATUS_COLOR: {},
        }
        if isinstance(snapshot, Snapshot):
            metadata[SORTED_REALIZATION_IDS] = sorted(
                snapshot.data()[ids.REALS].keys(), key=int
            )
            metadata[SORTED_JOB_IDS] = {}
            for real_id, real in snapshot.data()[ids.REALS].items():
                metadata[SORTED_JOB_IDS][real_id] = {}
                for step_id, step in real[ids.STEPS].items():
                    indices = [
                        (job.index, job_id) for job_id, job in step[ids.JOBS].items()
                    ]
                    metadata[SORTED_JOB_IDS][real_id][step_id] = [
                        index[1]
                        for index in sorted(indices, key=lambda indx: int(indx[0]))
                    ]

        for real_id, real in snapshot.data()[ids.REALS].items():
            if real.get(ids.STATUS):
                metadata[REAL_STATUS_COLOR][real_id] = _QCOLORS[
                    state.REAL_STATE_TO_COLOR[real[ids.STATUS]]
                ]
            metadata[REAL_JOB_STATUS_AGGREGATED][real_id] = {}
            if real.get(ids.STEPS):
                for step in real[ids.STEPS].values():
                    if ids.JOBS not in step:
                        continue
                    for job_id in step[ids.JOBS].keys():
                        status = step[ids.JOBS][job_id][ids.STATUS]
                        color = _QCOLORS[state.JOB_STATE_TO_COLOR[status]]
                        metadata[REAL_JOB_STATUS_AGGREGATED][real_id][job_id] = color

        if isinstance(snapshot, Snapshot):
            snapshot.merge_metadata(metadata)
        elif isinstance(snapshot, PartialSnapshot):
            snapshot.update_metadata(metadata)
        return snapshot

    # pylint: disable=too-many-branches
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

            # A list of changed realizations, so that we can send _one_
            # dataChange signal for all realizations at the end
            reals_changed: List[int] = []

            for real_id in iter_node.data[SORTED_REALIZATION_IDS]:
                real = partial.data()[ids.REALS].get(real_id)
                if not real:
                    continue
                real_node = iter_node.children[real_id]
                if real.get(ids.STATUS):
                    real_node.data[ids.STATUS] = real[ids.STATUS]

                real_index = self.index(real_node.row(), 0, iter_index)

                for job_id, color in (
                    metadata[REAL_JOB_STATUS_AGGREGATED].get(real_id, {}).items()
                ):
                    real_node.data[REAL_JOB_STATUS_AGGREGATED][job_id] = color
                if real_id in metadata[REAL_STATUS_COLOR]:
                    real_node.data[REAL_STATUS_COLOR] = metadata[REAL_STATUS_COLOR][
                        real_id
                    ]

                # If the realization is included in the delta, assume it was
                # changed
                reals_changed.append(real_node.row())

                if not real.get(ids.STEPS):
                    continue

                for step_id, step in real[ids.STEPS].items():
                    step_node = real_node.children[step_id]
                    if step.get(ids.STATUS):
                        step_node.data[ids.STATUS] = step[ids.STATUS]

                    step_index = self.index(step_node.row(), 0, real_index)

                    if not step.get(ids.JOBS):
                        continue

                    # a list of jobs that were changed for this step. This list
                    # is used to send _one_ dataChanged signal for all changed
                    # jobs. Jobs changed are always consecutive and it makes a
                    # lot of sense to merge changes.
                    jobs_changed: List[int] = []

                    for job_id, job in step[ids.JOBS].items():
                        job_node = step_node.children[job_id]

                        # if a job is included in this delta, assume it has been changed
                        jobs_changed.append(job_node.row())

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
                        if job.get(ids.INDEX):
                            job_node.data[ids.INDEX] = job[ids.INDEX]

                        # Errors may be unset as the queue restarts the job
                        job_node.data[ids.ERROR] = (
                            job[ids.ERROR] if job.get(ids.ERROR) else ""
                        )

                        for attr in (ids.CURRENT_MEMORY_USAGE, ids.MAX_MEMORY_USAGE):
                            if job.get(ids.DATA) and attr in job.get(ids.DATA):
                                job_node.data[ids.DATA] = job_node.data[ids.DATA].set(
                                    attr, job.get(ids.DATA).get(attr)
                                )

                    if jobs_changed:
                        job_top_left = self.index(min(jobs_changed), 0, step_index)
                        job_bottom_right = self.index(
                            max(jobs_changed),
                            self.columnCount(step_index) - 1,
                            step_index,
                        )
                        stack.callback(
                            self.dataChanged.emit, job_top_left, job_bottom_right
                        )

            if reals_changed:
                real_top_left = self.index(min(reals_changed), 0, iter_index)
                real_bottom_right = self.index(
                    max(reals_changed), self.columnCount(iter_index) - 1, iter_index
                )
                stack.callback(self.dataChanged.emit, real_top_left, real_bottom_right)

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
                for job_id in metadata[SORTED_JOB_IDS][real_id][step_id]:
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
        self.root.add_child(snapshot_tree, node_id=iter_)
        self.rowsInserted.emit(parent, snapshot_tree.row(), snapshot_tree.row())

    # pylint: disable=invalid-name, no-self-use
    def columnCount(self, parent: QModelIndex = None):
        if parent is None:
            parent = QModelIndex()
        parent_node = parent.internalPointer()
        if parent_node is None:
            return len(COLUMNS[NodeType.ROOT])
        return len(COLUMNS[parent_node.type])

    # pylint: disable=invalid-name
    def rowCount(self, parent: QModelIndex = None):
        if parent is None:
            parent = QModelIndex()
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
                f"index r{index.row()}/c{index.column()} pointed to parent-less item "
                + f"{child_item}"
            )
        parentItem = child_item.parent

        if parentItem == self.root:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    # pylint: disable=too-many-return-statements
    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        node = index.internalPointer()

        if role == NodeRole:
            return node

        if role == IsEnsembleRole:
            return node.type == NodeType.ITER
        if role == IsRealizationRole:
            return node.type == NodeType.REAL
        if role == IsStepRole:
            return node.type == NodeType.STEP
        if role == IsJobRole:
            return node.type == NodeType.JOB

        if node.type == NodeType.JOB:
            return self._job_data(index, node, role)
        if node.type == NodeType.REAL:
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

    def _real_data(self, _index: QModelIndex, node: Node, role: int):
        if role == RealJobColorHint:
            colors: List[QColor] = []
            assert node.parent  # mypy
            for step_id in node.parent.data[SORTED_JOB_IDS][node.id]:
                for job_id in node.parent.data[SORTED_JOB_IDS][node.id][step_id]:
                    colors.append(node.data[REAL_JOB_STATUS_AGGREGATED][job_id])
            return colors
        if role == RealLabelHint:
            return node.id
        if role == RealIens:
            return node.id
        if role == RealStatusColorHint:
            return node.data[REAL_STATUS_COLOR]
        if role == StatusRole:
            return node.data[ids.STATUS]
        return QVariant()

    # pylint: disable=too-many-return-statements, no-self-use
    def _job_data(self, index: QModelIndex, node: Node, role: int):
        if role == Qt.BackgroundRole:
            assert node.parent  # mypy
            assert node.parent.parent  # mypy
            real = node.parent.parent
            return real.data[REAL_JOB_STATUS_AGGREGATED][node.id]
        if role == Qt.DisplayRole:
            _, data_name = COLUMNS[NodeType.STEP][index.column()]
            if data_name in [ids.CURRENT_MEMORY_USAGE, ids.MAX_MEMORY_USAGE]:
                data = node.data.get(ids.DATA)
                _bytes = data.get(data_name) if data else None
                if _bytes:
                    return byte_with_unit(_bytes)
            if data_name in [ids.STDOUT, ids.STDERR]:
                return "OPEN" if node.data.get(data_name) else QVariant()
            if data_name in [DURATION]:
                start_time = node.data.get(ids.START_TIME)
                if start_time is None:
                    return QVariant()
                delta = _estimate_duration(
                    start_time, end_time=node.data.get(ids.END_TIME)
                )
                # There is no method for truncating microseconds, so we remove them
                delta -= datetime.timedelta(microseconds=delta.microseconds)
                return str(delta)
            return node.data.get(data_name)
        if role == FileRole:
            _, data_name = COLUMNS[NodeType.STEP][index.column()]
            if data_name in [ids.STDOUT, ids.STDERR]:
                return (
                    node.data.get(data_name) if node.data.get(data_name) else QVariant()
                )
        if role == Qt.ToolTipRole:
            _, data_name = COLUMNS[NodeType.STEP][index.column()]
            data = None
            if data_name == ids.ERROR:
                data = node.data.get(data_name)
            elif data_name == DURATION:
                start_time = node.data.get(ids.START_TIME)
                if start_time is not None:
                    delta = _estimate_duration(
                        start_time, end_time=node.data.get(ids.END_TIME)
                    )
                    data = f"Start time: {str(start_time)}\nDuration: {str(delta)}"
            if data is not None:
                return str(data)

        return QVariant()

    def index(self, row: int, column: int, parent: QModelIndex = None) -> QModelIndex:
        if parent is None:
            parent = QModelIndex()
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root
        else:
            parent_item = parent.internalPointer()

        child_item = None
        try:
            child_item = list(parent_item.children.values())[row]
        except KeyError:
            return QModelIndex()
        else:
            return self.createIndex(row, column, child_item)

    def reset(self):
        self.modelAboutToBeReset.emit()
        self.root = Node(0, {}, NodeType.ROOT)
        self.modelReset.emit()
