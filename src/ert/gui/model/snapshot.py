import datetime
import logging
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple, Union

from dateutil import tz
from qtpy.QtCore import QAbstractItemModel, QModelIndex, QSize, Qt, QVariant
from qtpy.QtGui import QColor, QFont

from ert.ensemble_evaluator import PartialSnapshot, Snapshot, state
from ert.ensemble_evaluator import identifiers as ids
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
IterNum = Qt.UserRole + 12

# Indicates what type the underlying data is
IsEnsembleRole = Qt.UserRole + 8
IsRealizationRole = Qt.UserRole + 9
IsJobRole = Qt.UserRole + 10
StatusRole = Qt.UserRole + 11

JOB_COLUMN_NAME = "Name"
JOB_COLUMN_ERROR = "Error"
JOB_COLUMN_STATUS = "Status"
JOB_COLUMN_DURATION = "Duration"
JOB_COLUMN_STDOUT = "STDOUT"
JOB_COLUMN_STDERR = "STDERR"
JOB_COLUMN_CURRENT_MEMORY_USAGE = "Current memory usage"
JOB_COLUMN_MAX_MEMORY_USAGE = "Max memory usage"

SORTED_REALIZATION_IDS = "_sorted_real_ids"
SORTED_JOB_IDS = "_sorted_job_ids"
REAL_JOB_STATUS_AGGREGATED = "_aggr_job_status_colors"
REAL_STATUS_COLOR = "_real_status_colors"

DURATION = "Duration"

COLUMNS: Dict[NodeType, Sequence[Union[str, Tuple[str, str]]]] = {
    NodeType.ROOT: ["Name", "Status"],
    NodeType.ITER: ["Name", "Status", "Active"],
    NodeType.REAL: [
        (JOB_COLUMN_NAME, ids.NAME),
        (JOB_COLUMN_ERROR, ids.ERROR),
        (JOB_COLUMN_STATUS, ids.STATUS),
        (
            JOB_COLUMN_DURATION,
            DURATION,
        ),  # Duration is based on two data fields, not coming directly from ert
        (JOB_COLUMN_STDOUT, ids.STDOUT),
        (JOB_COLUMN_STDERR, ids.STDERR),
        (JOB_COLUMN_CURRENT_MEMORY_USAGE, ids.CURRENT_MEMORY_USAGE),
        (JOB_COLUMN_MAX_MEMORY_USAGE, ids.MAX_MEMORY_USAGE),
    ],
    NodeType.JOB: [],
}

COLOR_WAITING: Final[QColor] = QColor(*state.COLOR_WAITING)
COLOR_PENDING: Final[QColor] = QColor(*state.COLOR_PENDING)
COLOR_RUNNING: Final[QColor] = QColor(*state.COLOR_RUNNING)

_QCOLORS = {
    state.COLOR_WAITING: QColor(*state.COLOR_WAITING),
    state.COLOR_PENDING: QColor(*state.COLOR_PENDING),
    state.COLOR_RUNNING: QColor(*state.COLOR_RUNNING),
    state.COLOR_FAILED: QColor(*state.COLOR_FAILED),
    state.COLOR_UNKNOWN: QColor(*state.COLOR_UNKNOWN),
    state.COLOR_FINISHED: QColor(*state.COLOR_FINISHED),
    state.COLOR_NOT_ACTIVE: QColor(*state.COLOR_NOT_ACTIVE),
}


def _estimate_duration(
    start_time: datetime.datetime, end_time: datetime.datetime = None
):
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

        reals = snapshot.reals
        job_states = snapshot.get_job_status_for_all_reals()
        if not reals and not job_states:
            return None

        metadata: Dict[str, Any] = {
            # A mapping from real to job to that job's QColor status representation
            REAL_JOB_STATUS_AGGREGATED: defaultdict(dict),
            # A mapping from real to that real's QColor status representation
            REAL_STATUS_COLOR: defaultdict(dict),
        }

        for real_id, real in reals.items():
            if real.status:
                metadata[REAL_STATUS_COLOR][real_id] = _QCOLORS[
                    state.REAL_STATE_TO_COLOR[real.status]
                ]

        isSnapshot = False
        if isinstance(snapshot, Snapshot):
            isSnapshot = True
            metadata[SORTED_REALIZATION_IDS] = sorted(snapshot.reals.keys(), key=int)
            metadata[SORTED_JOB_IDS] = defaultdict(list)
        for (real_id, job_id), job_status in job_states.items():
            if isSnapshot:
                metadata[SORTED_JOB_IDS][real_id].append(job_id)
            color = _QCOLORS[state.JOB_STATE_TO_COLOR[job_status]]
            metadata[REAL_JOB_STATUS_AGGREGATED][real_id][job_id] = color

        if isSnapshot:
            snapshot.merge_metadata(metadata)
        elif isinstance(snapshot, PartialSnapshot):
            snapshot.update_metadata(metadata)
        return snapshot

    def _add_partial_snapshot(self, partial: PartialSnapshot, iter_: int):
        metadata = partial.metadata
        if not metadata:
            logger.debug("no metadata in partial, ignoring partial")
            return

        if iter_ not in self.root.children:
            logger.debug("no full snapshot yet, ignoring partial")
            return

        job_infos = partial.get_all_jobs()
        if not partial.reals and not job_infos:
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

            reals_changed: List[int] = []

            for real_id in partial.get_real_ids():
                real_node = iter_node.children[real_id]
                real = partial.get_real(real_id)
                if real and real.status:
                    real_node.data[ids.STATUS] = real.status
                for real_job_id, color in (
                    metadata[REAL_JOB_STATUS_AGGREGATED].get(real_id, {}).items()
                ):
                    real_node.data[REAL_JOB_STATUS_AGGREGATED][real_job_id] = color
                if real_id in metadata[REAL_STATUS_COLOR]:
                    real_node.data[REAL_STATUS_COLOR] = metadata[REAL_STATUS_COLOR][
                        real_id
                    ]
                reals_changed.append(real_node.row())

            jobs_changed_by_real: Mapping[str, Sequence[int]] = defaultdict(list)

            for (real_id, job_id), job in partial.get_all_jobs().items():
                real_node = iter_node.children[real_id]
                job_node = real_node.children[job_id]

                jobs_changed_by_real[real_id].append(job_node.row())

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
                if job.index:
                    job_node.data[ids.INDEX] = job.index
                if job.current_memory_usage:
                    job_node.data[ids.CURRENT_MEMORY_USAGE] = job.current_memory_usage
                if job.max_memory_usage:
                    job_node.data[ids.MAX_MEMORY_USAGE] = job.max_memory_usage

                # Errors may be unset as the queue restarts the job
                job_node.data[ids.ERROR] = job.error if job.error else ""

            for real_idx, changed_jobs in jobs_changed_by_real.items():
                real_node = iter_node.children[real_idx]
                real_index = self.index(real_node.row(), 0, iter_index)

                job_top_left = self.index(min(changed_jobs), 0, real_index)
                job_bottom_right = self.index(
                    max(changed_jobs),
                    self.columnCount(real_index) - 1,
                    real_index,
                )
                stack.callback(self.dataChanged.emit, job_top_left, job_bottom_right)

            if reals_changed:
                real_top_left = self.index(min(reals_changed), 0, iter_index)
                real_bottom_right = self.index(
                    max(reals_changed), self.columnCount(iter_index) - 1, iter_index
                )
                stack.callback(self.dataChanged.emit, real_top_left, real_bottom_right)

            return

    def _add_snapshot(self, snapshot: Snapshot, iter_: int):
        metadata = snapshot.metadata
        snapshot_tree = Node(
            iter_,
            {
                ids.STATUS: snapshot.status,
                SORTED_REALIZATION_IDS: metadata[SORTED_REALIZATION_IDS],
                SORTED_JOB_IDS: metadata[SORTED_JOB_IDS],
            },
            NodeType.ITER,
        )
        for real_id in snapshot_tree.data[SORTED_REALIZATION_IDS]:
            real = snapshot.get_real(real_id)
            real_node = Node(
                real_id,
                {
                    ids.STATUS: real.status,
                    ids.ACTIVE: real.active,
                    REAL_JOB_STATUS_AGGREGATED: metadata[REAL_JOB_STATUS_AGGREGATED][
                        real_id
                    ],
                    REAL_STATUS_COLOR: metadata[REAL_STATUS_COLOR][real_id],
                },
                NodeType.REAL,
            )
            snapshot_tree.add_child(real_node)

            for job_id in metadata[SORTED_JOB_IDS][real_id]:
                job = snapshot.get_job(real_id, job_id)
                job_node = Node(job_id, dict(job), NodeType.JOB)
                real_node.add_child(job_node)

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

    def columnCount(self, parent: QModelIndex = None):
        if parent is None:
            parent = QModelIndex()
        parent_node = parent.internalPointer()
        if parent_node is None:
            count = len(COLUMNS[NodeType.ROOT])
        else:
            count = len(COLUMNS[parent_node.type])
        return count

    def rowCount(self, parent: QModelIndex = None):
        if parent is None:
            parent = QModelIndex()
        parentItem = self.root if not parent.isValid() else parent.internalPointer()

        if parent.column() > 0:
            return 0

        return len(parentItem.children)

    def parent(self, index: QModelIndex):
        if not index.isValid():
            return QModelIndex()

        parent_item = index.internalPointer().parent
        if parent_item == self.root:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

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

            is_running = False

            if COLOR_RUNNING in node.data[REAL_JOB_STATUS_AGGREGATED].values():
                is_running = True

            for job_id in node.parent.data[SORTED_JOB_IDS][node.id]:
                # if queue system status is WAIT, jobs should indicate WAIT
                if (
                    node.data[REAL_JOB_STATUS_AGGREGATED][job_id] == COLOR_PENDING
                    and node.data[REAL_STATUS_COLOR] == COLOR_WAITING
                    and not is_running
                ):
                    colors.append(COLOR_WAITING)
                else:
                    colors.append(node.data[REAL_JOB_STATUS_AGGREGATED][job_id])

            return colors
        if role == RealLabelHint:
            return node.id
        if role == RealIens:
            return node.id
        if role == IterNum:
            return node.parent.id
        if role == RealStatusColorHint:
            return node.data[REAL_STATUS_COLOR]
        if role == StatusRole:
            return node.data[ids.STATUS]
        return QVariant()

    def _job_data(self, index: QModelIndex, node: Node, role: int):
        if role == Qt.BackgroundRole:
            real = node.parent
            if COLOR_RUNNING in real.data[REAL_JOB_STATUS_AGGREGATED].values():
                return real.data[REAL_JOB_STATUS_AGGREGATED][node.id]

            # if queue system status is WAIT, jobs should indicate WAIT
            if (
                real.data[REAL_JOB_STATUS_AGGREGATED][node.id] == COLOR_PENDING
                and real.data[REAL_STATUS_COLOR] == COLOR_WAITING
            ):
                return COLOR_WAITING
            return real.data[REAL_JOB_STATUS_AGGREGATED][node.id]

        if role == Qt.DisplayRole:
            _, data_name = COLUMNS[NodeType.REAL][index.column()]
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

            real = node.parent
            if COLOR_RUNNING in real.data[REAL_JOB_STATUS_AGGREGATED].values():
                return node.data.get(data_name)

            # if queue system status is WAIT, jobs should indicate WAIT
            if (
                data_name in [ids.STATUS]
                and real.data[REAL_STATUS_COLOR] == COLOR_WAITING
                and real.data[REAL_JOB_STATUS_AGGREGATED][node.id] == COLOR_PENDING
            ):
                return str("Waiting")
            return node.data.get(data_name)

        if role == FileRole:
            _, data_name = COLUMNS[NodeType.REAL][index.column()]
            if data_name in [ids.STDOUT, ids.STDERR]:
                return (
                    node.data.get(data_name) if node.data.get(data_name) else QVariant()
                )

        if role == RealIens:
            return node.parent.id

        if role == IterNum:
            return node.parent.parent.id

        if role == Qt.ToolTipRole:
            _, data_name = COLUMNS[NodeType.REAL][index.column()]
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

        parent_item = self.root if not parent.isValid() else parent.internalPointer()
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
