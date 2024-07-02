import datetime
import logging
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Dict, Final, List, Optional, Sequence, Union, overload

from dateutil import tz
from qtpy.QtCore import QAbstractItemModel, QModelIndex, QObject, QSize, Qt, QVariant
from qtpy.QtGui import QColor, QFont
from typing_extensions import override

from ert.ensemble_evaluator import PartialSnapshot, Snapshot, state
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator.snapshot import SnapshotMetadata
from ert.gui.model.node import (
    ForwardModelStepNode,
    IterNode,
    IterNodeData,
    RealNode,
    RealNodeData,
    RootNode,
)
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
MemoryUsageRole = Qt.UserRole + 13

# Indicates what type the underlying data is
IsEnsembleRole = Qt.UserRole + 8
IsRealizationRole = Qt.UserRole + 9
IsJobRole = Qt.UserRole + 10
StatusRole = Qt.UserRole + 11

DURATION = "Duration"

JOB_COLUMNS: Sequence[str] = [
    ids.NAME,
    ids.ERROR,
    ids.STATUS,
    DURATION,  # Duration is based on two data fields, not coming directly from ert
    ids.STDOUT,
    ids.STDERR,
    ids.CURRENT_MEMORY_USAGE,
    ids.MAX_MEMORY_USAGE,
]
JOB_COLUMN_SIZE: Final[int] = len(JOB_COLUMNS)

COLOR_FINISHED: Final[QColor] = QColor(*state.COLOR_FINISHED)

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
    start_time: datetime.datetime, end_time: Optional[datetime.datetime] = None
) -> datetime.timedelta:
    timezone = None
    if start_time.tzname() is not None:
        timezone = tz.gettz(start_time.tzname())
    if not end_time or end_time < start_time:
        end_time = datetime.datetime.now(timezone)
    return end_time - start_time


class SnapshotModel(QAbstractItemModel):
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.root: RootNode = RootNode(0)

    @staticmethod
    def prerender(
        snapshot: Union[Snapshot, PartialSnapshot],
    ) -> Optional[Union[Snapshot, PartialSnapshot]]:
        """Pre-render some data that is required by this model. Ideally, this
        is called outside the GUI thread. This is a requirement of the model,
        so it has to be called."""

        reals = snapshot.reals
        forward_model_states = snapshot.get_forward_model_status_for_all_reals()

        if not reals and not forward_model_states:
            return None

        metadata = SnapshotMetadata(
            # A mapping from real to job to that job's QColor status representation
            aggr_job_status_colors=defaultdict(dict),
            # A mapping from real to that real's QColor status representation
            real_status_colors=defaultdict(dict),
        )

        for real_id, real in reals.items():
            if real.status:
                metadata["real_status_colors"][real_id] = _QCOLORS[
                    state.REAL_STATE_TO_COLOR[real.status]
                ]

        isSnapshot = False
        if isinstance(snapshot, Snapshot):
            isSnapshot = True
            metadata["sorted_real_ids"] = sorted(snapshot.reals.keys(), key=int)
            metadata["sorted_forward_model_ids"] = defaultdict(list)

        running_forward_model_id: Dict[str, int] = {}
        for (
            real_id,
            forward_model_id,
        ), forward_model_status in forward_model_states.items():
            if forward_model_status == state.FORWARD_MODEL_STATE_RUNNING:
                running_forward_model_id[real_id] = int(forward_model_id)

        for (
            real_id,
            forward_model_id,
        ), forward_model_status in forward_model_states.items():
            if isSnapshot:
                metadata["sorted_forward_model_ids"][real_id].append(forward_model_id)
            if (
                real_id in running_forward_model_id
                and int(forward_model_id) > running_forward_model_id[real_id]
            ):
                # Triggered on resubmitted realizations
                color = _QCOLORS[
                    state.FORWARD_MODEL_STATE_TO_COLOR[state.FORWARD_MODEL_STATE_START]
                ]
            else:
                color = _QCOLORS[
                    state.FORWARD_MODEL_STATE_TO_COLOR[forward_model_status]
                ]
            metadata["aggr_job_status_colors"][real_id][forward_model_id] = color

        if isSnapshot:
            snapshot.merge_metadata(metadata)
        elif isinstance(snapshot, PartialSnapshot):
            snapshot.update_metadata(metadata)
        return snapshot

    def _add_partial_snapshot(self, partial: PartialSnapshot, iter_: int) -> None:
        metadata = partial.metadata
        if not metadata:
            logger.debug("no metadata in partial, ignoring partial")
            return

        if iter_ not in self.root.children:
            logger.debug("no full snapshot yet, ignoring partial")
            return

        job_infos = partial.get_all_forward_models()
        reals = partial.reals
        if not reals and not job_infos:
            logger.debug(f"no realizations in partial for iter {iter_}")
            return

        # Stack onto which we push change events for entities, since we branch
        # the code based on what is in the partial. This way we're guaranteed
        # that the change events will be emitted when the stack is unwound.
        with ExitStack() as stack:
            iter_node = self.root.children[iter_]
            iter_index = self.index(iter_node.row(), 0, QModelIndex())
            reals_changed: List[int] = []

            for real_id, real in reals.items():
                real_node = iter_node.children[real_id]
                if real and real.status:
                    real_node.data.status = real.status
                for real_forward_model_id, color in (
                    metadata["aggr_job_status_colors"].get(real_id, {}).items()
                ):
                    real_node.data.forward_model_step_status_color_by_id[
                        real_forward_model_id
                    ] = color
                if real_id in metadata["real_status_colors"]:
                    real_node.data.real_status_color = metadata["real_status_colors"][
                        real_id
                    ]
                reals_changed.append(real_node.row())

            jobs_changed_by_real: Dict[str, List[int]] = defaultdict(list)

            for (
                real_id,
                forward_model_id,
            ), job in job_infos.items():
                real_node = iter_node.children[real_id]
                job_node = real_node.children[forward_model_id]

                jobs_changed_by_real[real_id].append(job_node.row())

                job_node.data.update(job)
                if (
                    "current_memory_usage" in job
                    and job["current_memory_usage"] is not None
                ):
                    cur_mem_usage = int(float(job["current_memory_usage"]))
                    real_node.data.current_memory_usage = cur_mem_usage
                if "max_memory_usage" in job and job["max_memory_usage"] is not None:
                    max_mem_usage = int(float(job["max_memory_usage"]))

                    real_node.data.max_memory_usage = max(
                        real_node.data.max_memory_usage or 0, max_mem_usage
                    )
                    self.root.max_memory_usage = max(
                        self.root.max_memory_usage or 0, max_mem_usage
                    )

                # Errors may be unset as the queue restarts the job
                job_node.data[ids.ERROR] = job.get(ids.ERROR, "")

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

    def _add_snapshot(self, snapshot: Snapshot, iter_: int) -> None:
        metadata = snapshot.metadata
        snapshot_tree = IterNode(
            id_=iter_,
            data=IterNodeData(
                status=snapshot.status,
            ),
        )
        for real_id in metadata["sorted_real_ids"]:
            real = snapshot.get_real(real_id)
            real_node = RealNode(
                real_id,
                data=RealNodeData(
                    status=real.status,
                    active=real.active,
                    forward_model_step_status_color_by_id=metadata[
                        "aggr_job_status_colors"
                    ][real_id],
                    real_status_color=metadata["real_status_colors"][real_id],
                ),
            )
            snapshot_tree.add_child(real_node)

            for forward_model_id in metadata["sorted_forward_model_ids"][real_id]:
                job = snapshot.get_job(real_id, forward_model_id)
                job_node = ForwardModelStepNode(
                    id_=forward_model_id, data=job, parent=real_node
                )
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
        self.root.add_child(snapshot_tree)
        self.rowsInserted.emit(parent, snapshot_tree.row(), snapshot_tree.row())

    @override
    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        if parent and isinstance(parent.internalPointer(), RealNode):
            return JOB_COLUMN_SIZE
        return 1

    def rowCount(self, parent: Optional[QModelIndex] = None):
        if parent is None:
            parent = QModelIndex()
        parent_item = self.root if not parent.isValid() else parent.internalPointer()

        if parent.column() > 0:
            return 0

        return len(parent_item.children)

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...
    @overload
    def parent(self) -> Optional[QObject]: ...
    @override
    def parent(self, child: Optional[QModelIndex] = None) -> Optional[QObject]:
        if child is None or not child.isValid():
            return QModelIndex()

        parent_item = child.internalPointer().parent
        if parent_item == self.root:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    @override
    def data(
        self, index: QModelIndex, role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        node: Union[RootNode, IterNode, RealNode, ForwardModelStepNode] = (
            index.internalPointer()
        )
        if role == NodeRole:
            return node

        if role == IsEnsembleRole:
            return isinstance(node, IterNode)
        if role == IsRealizationRole:
            return isinstance(node, RealNode)
        if role == IsJobRole:
            return isinstance(node, ForwardModelStepNode)

        if isinstance(node, ForwardModelStepNode):
            return self._job_data(index, node, role)
        if isinstance(node, RealNode):
            return self._real_data(index, node, role)

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return f"{type(node).__name__}:{node.id_}"
            if index.column() == 1:
                return f"{node.data.status}"

        if role in (
            Qt.ItemDataRole.StatusTipRole,
            Qt.ItemDataRole.WhatsThisRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            return ""

        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize()

        if role == Qt.ItemDataRole.FontRole:
            return QFont()

        if role in (
            Qt.ItemDataRole.BackgroundRole,
            Qt.ItemDataRole.ForegroundRole,
            Qt.ItemDataRole.DecorationRole,
        ):
            return QColor()

        return QVariant()

    @staticmethod
    def _real_data(_index: QModelIndex, node: RealNode, role: int):
        if role == RealJobColorHint:
            total_count = len(node.data.forward_model_step_status_color_by_id)
            finished_count = sum(
                1
                for color in node.data.forward_model_step_status_color_by_id.values()
                if color == COLOR_FINISHED
            )

            queue_color = node.data.real_status_color
            return queue_color, finished_count, total_count
        if role == RealLabelHint:
            return node.id_
        if role == RealIens:
            return node.id_
        if role == IterNum:
            return node.parent.id_
        if role == RealStatusColorHint:
            return node.data.real_status_color
        if role == StatusRole:
            return node.data.status
        if role == MemoryUsageRole:
            return (
                node.data.current_memory_usage,
                node.data.max_memory_usage,
            )
        return QVariant()

    @staticmethod
    def _job_data(
        index: QModelIndex, node: ForwardModelStepNode, role: Qt.ItemDataRole
    ):
        node_id = str(node.id_)

        if role == Qt.ItemDataRole.FontRole:
            data_name = JOB_COLUMNS[index.column()]
            if data_name in [ids.STDOUT, ids.STDERR]:
                font = QFont()
                font.setUnderline(True)
                return font

        if role == Qt.ItemDataRole.ForegroundRole:
            data_name = JOB_COLUMNS[index.column()]
            if data_name in [ids.STDOUT, ids.STDERR]:
                return QColor(Qt.GlobalColor.blue)

        if role == Qt.ItemDataRole.BackgroundRole:
            return node.parent.data.forward_model_step_status_color_by_id[node_id]

        if role == Qt.ItemDataRole.DisplayRole:
            data_name = JOB_COLUMNS[index.column()]
            if data_name in [ids.CURRENT_MEMORY_USAGE, ids.MAX_MEMORY_USAGE]:
                data = node.data
                _bytes: Optional[str] = data.get(data_name)
                if _bytes:
                    return byte_with_unit(float(_bytes))

            if data_name in [ids.STDOUT, ids.STDERR]:
                return "OPEN" if data_name in node.data else QVariant()

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
            data_name = JOB_COLUMNS[index.column()]
            if data_name in [ids.STDOUT, ids.STDERR]:
                return node.data.get(data_name, QVariant())

        if role == RealIens:
            return node.parent.id_

        if role == IterNum:
            return node.parent.parent.id_

        if role == Qt.ItemDataRole.ToolTipRole:
            data_name = JOB_COLUMNS[index.column()]
            data = None
            if data_name == ids.ERROR:
                data = node.data.get(ids.ERROR)
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

    @override
    def index(
        self, row: int, column: int, parent: Optional[QModelIndex] = None
    ) -> QModelIndex:
        if parent is None:
            parent = QModelIndex()
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        parent_item = self.root if not parent.isValid() else parent.internalPointer()
        try:
            child_item = list(parent_item.children.values())[row]
        except KeyError:
            return QModelIndex()
        else:
            return self.createIndex(row, column, child_item)

    def reset(self) -> None:
        self.modelAboutToBeReset.emit()
        self.root = RootNode(0)
        self.modelReset.emit()
