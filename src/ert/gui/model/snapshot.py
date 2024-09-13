import logging
import sys
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timedelta
from typing import Any, Dict, Final, List, Optional, Sequence, Union, overload

from qtpy.QtCore import QAbstractItemModel, QModelIndex, QObject, QSize, Qt, QVariant
from qtpy.QtGui import QColor, QFont
from typing_extensions import override

from ert.ensemble_evaluator import EnsembleSnapshot, state
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshotMetadata,
    RealId,
    convert_iso8601_to_datetime,
)
from ert.gui.model.node import (
    ForwardModelStepNode,
    IterNode,
    IterNodeData,
    RealNode,
    RealNodeData,
    RootNode,
)
from ert.shared.status.utils import byte_with_unit, file_has_content

if sys.version_info < (3, 11):
    from backports.datetime_fromisoformat import MonkeyPatch  # type: ignore

    MonkeyPatch.patch_fromisoformat()

logger = logging.getLogger(__name__)

UserRole = Qt.ItemDataRole.UserRole
NodeRole = UserRole + 1
FMStepColorHint = UserRole + 2
RealLabelHint = UserRole + 4
ProgressRole = UserRole + 5
FileRole = UserRole + 6
RealIens = UserRole + 7
IterNum = UserRole + 12
MemoryUsageRole = UserRole + 13
CallbackStatusMessageRole = UserRole + 14

# Indicates what type the underlying data is
IsEnsembleRole = UserRole + 8
IsRealizationRole = UserRole + 9
IsFMStepRole = UserRole + 10
StatusRole = UserRole + 11

DURATION = "Duration"

FM_STEP_COLUMNS: Sequence[str] = [
    ids.NAME,
    ids.ERROR,
    ids.STATUS,
    DURATION,  # Duration is based on two data fields, not coming directly from ert
    ids.STDOUT,
    ids.STDERR,
    ids.MAX_MEMORY_USAGE,
]
FM_STEP_COLUMN_SIZE: Final[int] = len(FM_STEP_COLUMNS)

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
    start_time: datetime, end_time: Optional[datetime] = None
) -> timedelta:
    if not end_time or end_time < start_time:
        end_time = datetime.now(start_time.tzinfo)
    return end_time - start_time


class SnapshotModel(QAbstractItemModel):
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.root: RootNode = RootNode("0")

    @staticmethod
    def prerender(ensemble: EnsembleSnapshot) -> Optional[EnsembleSnapshot]:
        """Pre-render some data that is required by this model. Ideally, this
        is called outside the GUI thread. This is a requirement of the model,
        so it has to be called."""

        reals = ensemble.reals
        fm_step_snapshots = ensemble.get_fm_steps_for_all_reals()

        if not reals and not fm_step_snapshots:
            return None

        metadata = EnsembleSnapshotMetadata(
            aggr_fm_step_status_colors=defaultdict(dict),
            real_status_colors={},
            sorted_real_ids=[],
            sorted_fm_step_ids=defaultdict(),
        )

        for real_id, real in reals.items():
            if status := real.get("status"):
                metadata["real_status_colors"][real_id] = _QCOLORS[
                    state.REAL_STATE_TO_COLOR[status]
                ]

        metadata["sorted_real_ids"] = sorted(ensemble.reals.keys(), key=int)
        metadata["sorted_fm_step_ids"] = defaultdict(list)

        running_fm_step_id: Dict[RealId, int] = {}
        for (real_id, fm_step_id), fm_step_snapshot in fm_step_snapshots.items():
            if fm_step_snapshot == state.FORWARD_MODEL_STATE_RUNNING:
                running_fm_step_id[real_id] = int(fm_step_id)

        for (real_id, fm_step_id), fm_step_snapshot in fm_step_snapshots.items():
            metadata["sorted_fm_step_ids"][real_id].append(fm_step_id)
            if (
                real_id in running_fm_step_id
                and int(fm_step_id) > running_fm_step_id[real_id]
            ):
                # Triggered on resubmitted realizations
                color = _QCOLORS[
                    state.FORWARD_MODEL_STATE_TO_COLOR[state.FORWARD_MODEL_STATE_START]
                ]
            else:
                color = _QCOLORS[state.FORWARD_MODEL_STATE_TO_COLOR[fm_step_snapshot]]
            metadata["aggr_fm_step_status_colors"][real_id][fm_step_id] = color

        ensemble.merge_metadata(metadata)
        return ensemble

    def _update_snapshot(self, snapshot: EnsembleSnapshot, iter_: str) -> None:
        metadata = snapshot.metadata
        if not metadata:
            logger.debug("no metadata in update snapshot, ignoring snapshot")
            return

        if iter_ not in self.root.children:
            logger.debug("no full snapshot to update yet, ignoring snapshot")
            return

        fm_steps = snapshot.get_all_fm_steps()
        reals = snapshot.reals
        if not reals and not fm_steps:
            logger.debug(f"no realizations in snapshot for iter {iter_}")
            return

        # Stack onto which we push change events for entities, since we branch
        # the code based on what is in the snapshot. This way we're guaranteed
        # that the change events will be emitted when the stack is unwound.
        with ExitStack() as stack:
            iter_node = self.root.children[iter_]
            iter_index = self.index(iter_node.row(), 0, QModelIndex())
            reals_changed: List[int] = []

            for real_id, real in reals.items():
                real_node = iter_node.children[real_id]
                data = real_node.data
                if real_status := real.get("status"):
                    data.status = real_status
                for real_fm_step_id, color in (
                    metadata["aggr_fm_step_status_colors"].get(real_id, {}).items()
                ):
                    data.fm_step_status_color_by_id[real_fm_step_id] = color
                if real_id in metadata["real_status_colors"]:
                    data.real_status_color = metadata["real_status_colors"][real_id]
                reals_changed.append(real_node.row())
                if real.get("callback_status_message"):
                    data.callback_status_message = real["callback_status_message"]

            fm_steps_changed_by_real: Dict[str, List[int]] = defaultdict(list)
            for (real_id, fm_step_id), fm_step in fm_steps.items():
                real_node = iter_node.children[real_id]
                fm_step_node = real_node.children[fm_step_id]

                fm_steps_changed_by_real[real_id].append(fm_step_node.row())
                if start_time := fm_step.get("start_time", None):
                    fm_step["start_time"] = convert_iso8601_to_datetime(start_time)
                if end_time := fm_step.get("end_time", None):
                    fm_step["end_time"] = convert_iso8601_to_datetime(end_time)
                # Errors may be unset as the queue restarts the job
                fm_step[ids.ERROR] = fm_step.get(ids.ERROR, "")
                fm_step_node.data.update(fm_step)
                if cur_mem_usage := fm_step.get("current_memory_usage", None):
                    real_node.data.current_memory_usage = int(float(cur_mem_usage))
                if maximum_mem_usage := fm_step.get("max_memory_usage", None):
                    max_mem_usage = int(float(maximum_mem_usage))
                    real_node.data.max_memory_usage = max(
                        real_node.data.max_memory_usage or 0, max_mem_usage
                    )
                    self.root.max_memory_usage = max(
                        self.root.max_memory_usage or 0, max_mem_usage
                    )

            for real_idx, changed_fm_steps in fm_steps_changed_by_real.items():
                real_node = iter_node.children[real_idx]
                real_index = self.index(real_node.row(), 0, iter_index)

                fm_step_top_left = self.index(min(changed_fm_steps), 0, real_index)
                fm_step_bottom_right = self.index(
                    max(changed_fm_steps),
                    self.columnCount(real_index) - 1,
                    real_index,
                )
                stack.callback(
                    self.dataChanged.emit, fm_step_top_left, fm_step_bottom_right
                )

            if reals_changed:
                real_top_left = self.index(min(reals_changed), 0, iter_index)
                real_bottom_right = self.index(
                    max(reals_changed), self.columnCount(iter_index) - 1, iter_index
                )
                stack.callback(self.dataChanged.emit, real_top_left, real_bottom_right)

            return

    def _add_snapshot(self, snapshot: EnsembleSnapshot, iter_: str) -> None:
        metadata = snapshot.metadata
        snapshot_tree = IterNode(
            id_=iter_,
            data=IterNodeData(
                status=snapshot.status,
            ),
        )
        for real_id in metadata.get("sorted_real_ids", {}):
            real = snapshot.get_real(real_id)
            real_node = RealNode(
                id_=real_id,
                data=RealNodeData(
                    status=real.get("status"),
                    active=real.get("active"),
                    fm_step_status_color_by_id=metadata.get(
                        "aggr_fm_step_status_colors", defaultdict(None)
                    )[real_id],
                    real_status_color=metadata.get(
                        "real_status_colors", defaultdict(None)
                    )[real_id],
                    callback_status_message=real.get("callback_status_message"),
                ),
            )
            snapshot_tree.add_child(real_node)

            for fm_step_id in metadata.get("sorted_fm_step_ids", defaultdict(None))[
                real_id
            ]:
                fm_step = snapshot.get_fm_step(real_id, fm_step_id)
                if start_time := fm_step.get("start_time", None):
                    fm_step["start_time"] = convert_iso8601_to_datetime(start_time)
                if end_time := fm_step.get("end_time", None):
                    fm_step["end_time"] = convert_iso8601_to_datetime(end_time)
                fm_step_node = ForwardModelStepNode(
                    id_=fm_step_id, data=fm_step, parent=real_node
                )
                real_node.add_child(fm_step_node)

        if iter_ in self.root.children:
            self.beginResetModel()
            snapshot_tree.parent = self.root
            self.root.children[iter_] = snapshot_tree
            self.endResetModel()
            return

        parent = QModelIndex()
        next_iter = len(self.root.children)
        self.beginInsertRows(parent, next_iter, next_iter)
        self.root.add_child(snapshot_tree)
        self.rowsInserted.emit(parent, snapshot_tree.row(), snapshot_tree.row())

    @override
    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        if parent and isinstance(parent.internalPointer(), RealNode):
            return FM_STEP_COLUMN_SIZE
        return 1

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
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
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        node: Union[IterNode, RealNode, ForwardModelStepNode] = index.internalPointer()
        if role == NodeRole:
            return node

        if role == IsEnsembleRole:
            return isinstance(node, IterNode)
        if role == IsRealizationRole:
            return isinstance(node, RealNode)
        if role == IsFMStepRole:
            return isinstance(node, ForwardModelStepNode)

        if isinstance(node, ForwardModelStepNode):
            return self._fm_step_data(index, node, role)
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
    def _real_data(_: QModelIndex, node: RealNode, role: int) -> Any:
        if role == FMStepColorHint:
            total_count = len(node.data.fm_step_status_color_by_id)
            finished_count = sum(
                1
                for color in node.data.fm_step_status_color_by_id.values()
                if color == COLOR_FINISHED
            )

            queue_color = node.data.real_status_color
            return queue_color, finished_count, total_count
        if role == RealLabelHint:
            return node.id_
        if role == RealIens:
            return node.id_
        if role == IterNum:
            return node.parent.id_ if node.parent else None
        if role == StatusRole:
            return node.data.status
        if role == MemoryUsageRole:
            return node.data.max_memory_usage
        if role == CallbackStatusMessageRole:
            return node.data.callback_status_message

        return QVariant()

    @staticmethod
    def _fm_step_data(
        index: QModelIndex,
        node: ForwardModelStepNode,
        role: int,  # Qt.ItemDataRole
    ) -> Any:
        node_id = str(node.id_)

        if role == Qt.ItemDataRole.FontRole:
            data_name = FM_STEP_COLUMNS[index.column()]
            if data_name in [ids.STDOUT, ids.STDERR] and file_has_content(
                index.data(FileRole)
            ):
                font = QFont()
                font.setUnderline(True)
                return font

        if role == Qt.ItemDataRole.ForegroundRole:
            data_name = FM_STEP_COLUMNS[index.column()]
            if data_name in [ids.STDOUT, ids.STDERR] and file_has_content(
                index.data(FileRole)
            ):
                return QColor(Qt.GlobalColor.blue)

        if role == Qt.ItemDataRole.BackgroundRole:
            return (
                node.parent.data.fm_step_status_color_by_id[node_id]
                if node.parent
                else None
            )

        if role == Qt.ItemDataRole.DisplayRole:
            data_name = FM_STEP_COLUMNS[index.column()]
            if data_name in [ids.MAX_MEMORY_USAGE]:
                data = node.data
                _bytes: Optional[str] = data.get(data_name)  # type: ignore
                if _bytes:
                    return byte_with_unit(float(_bytes))

            if data_name in [ids.STDOUT, ids.STDERR]:
                if not file_has_content(index.data(FileRole)):
                    return "-"
                return "View" if data_name in node.data else QVariant()

            if data_name in [DURATION]:
                start_time = node.data.get(ids.START_TIME)
                if start_time is None:
                    return QVariant()
                delta = _estimate_duration(
                    start_time, end_time=node.data.get(ids.END_TIME)
                )
                # There is no method for truncating microseconds, so we remove them
                delta -= timedelta(microseconds=delta.microseconds)
                return str(delta)

            return node.data.get(data_name)

        if role == FileRole:
            data_name = FM_STEP_COLUMNS[index.column()]
            if data_name in [ids.STDOUT, ids.STDERR]:
                return node.data.get(data_name, QVariant())

        if role == RealIens:
            return node.parent.id_ if node.parent else None

        if role == IterNum:
            return (
                node.parent.parent.id_ if node.parent and node.parent.parent else None
            )

        if role == Qt.ItemDataRole.ToolTipRole:
            data_name = FM_STEP_COLUMNS[index.column()]
            tt_text = None
            if data_name == ids.ERROR:
                tt_text = node.data.get(ids.ERROR)
            elif data_name == DURATION:
                start_time = node.data.get(ids.START_TIME)
                if start_time is not None:
                    delta = _estimate_duration(
                        start_time, end_time=node.data.get(ids.END_TIME)
                    )
                    tt_text = f"Start time: {start_time!s}\nDuration: {delta!s}"
            if tt_text is not None:
                return str(tt_text)

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
        self.root = RootNode("0")
        self.modelReset.emit()
