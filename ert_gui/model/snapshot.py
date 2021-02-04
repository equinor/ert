import logging

from ert_gui.model.node import Node, NodeType, snapshot_to_tree
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    Snapshot,
)
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt, QVariant, Slot
from qtpy.QtGui import QColor

from ert_shared.status.entity.state import JOB_STATE_FINISHED, REAL_STATE_TO_COLOR


logger = logging.getLogger(__name__)


NodeRole = Qt.UserRole + 1
RealJobColorHint = Qt.UserRole + 2
RealStatusColorHint = Qt.UserRole + 3
RealLabelHint = Qt.UserRole + 4
ProgressRole = Qt.UserRole + 5


COLUMNS = {
    NodeType.ROOT: ["Name", "Status"],
    NodeType.ITER: ["Name", "Status", "Active"],
    NodeType.REAL: ["Name", "Status"],
    NodeType.STAGE: ["Name", "Status"],
    NodeType.STEP: [
        "Name",
        "Status",
        "Start time",
        "End time",
        "Current Memory Usage",
        "Max Memory Usage",
    ],
    NodeType.JOB: [],
}


class SnapshotModel(QAbstractItemModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root = Node(None, {"status": "n/a"}, NodeType.ROOT)

    def _add_partial_snapshot(self, partial: PartialSnapshot, iter_: int):
        partial_d = partial.to_dict()
        if iter_ not in self.root.children:
            logger.debug("no full snapshot yet, bailing")
            return
        iter_index = self.index(iter_, 0, QModelIndex())
        iter_node = self.root.children[iter_]
        if "reals" not in partial_d:
            logger.debug(f"no realizations in partial for iter {iter_}")
            return
        for real_id in sorted(partial_d["reals"], key=int):
            real = partial_d["reals"][real_id]
            real_node = iter_node.children[real_id]
            if real.get("status"):
                real_node.data["status"] = real.get("status")

            real_index = self.index(real_node.row(), 0, iter_index)
            real_index_bottom_right = self.index(
                real_node.row(), self.columnCount(iter_index) - 1, iter_index
            )

            if "stages" not in real:
                continue

            # TODO: sort stages, but wait till after https://github.com/equinor/ert/issues/1220 ?
            for stage_id, stage in real["stages"].items():
                stage_node = real_node.children[stage_id]

                if stage.get("status"):
                    stage_node.data["status"] = stage.get("status")

                stage_index = self.index(stage_node.row(), 0, real_index)
                stage_index_bottom_right = self.index(
                    stage_node.row(), self.columnCount(real_index) - 1, real_index
                )

                if "steps" not in stage:
                    continue

                # TODO: sort steps, but wait till after https://github.com/equinor/ert/issues/1220 ?
                for step_id, step in stage["steps"].items():
                    step_node = stage_node.children[step_id]
                    if step.get("status"):
                        step_node.data["status"] = step.get("status")

                    step_index = self.index(step_node.row(), 0, stage_index)
                    step_index_bottom_right = self.index(
                        step_node.row(), self.columnCount(stage_index) - 1, stage_index
                    )

                    if "jobs" not in step:
                        continue

                    for job_id in sorted(step["jobs"], key=int):
                        job = step["jobs"][job_id]
                        job_node = step_node.children[job_id]
                        if job.get("status"):
                            job_node.data["status"] = job["status"]
                        if job.get("start_time"):
                            job_node.data["start_time"] = job["start_time"]
                        if job.get("end_time"):
                            job_node.data["end_time"] = job["end_time"]
                        if job.get("data", {}).get("current_memory_usage"):
                            job_node.data["data"]["current_memory_usage"] = job["data"][
                                "current_memory_usage"
                            ]
                        if job.get("data", {}).get("max_memory_usage"):
                            job_node.data["data"]["max_memory_usage"] = job["data"][
                                "max_memory_usage"
                            ]

                        job_index = self.index(job_node.row(), 0, step_index)
                        job_index_bottom_right = self.index(
                            job_node.row(), self.columnCount() - 1, step_index
                        )
                        self.dataChanged.emit(job_index, job_index_bottom_right)
                    self.dataChanged.emit(step_index, step_index_bottom_right)
                self.dataChanged.emit(stage_index, stage_index_bottom_right)
            self.dataChanged.emit(real_index, real_index_bottom_right)

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
        # if child_item is root? might be misbehaving proxy model...
        if not hasattr(child_item, "parent"):
            print(
                "index pointed to parent-less item",
                child_item,
                index.row(),
                index.column(),
                index.parent(),
                index.parent().isValid(),
            )
            return QModelIndex()
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

        if node.type == NodeType.JOB:
            return self._job_data(index, node, role)

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return f"{node.type}:{node.id}"
            if index.column() == 1:
                return f"{node.data['status']}"

        if role == NodeRole:
            return node

        if node.type == NodeType.REAL and role == RealJobColorHint:
            # TODO: make nicer + make sure it works with thare more than 1 job
            for _, stage in node.children.items():
                # print("1: ", stage)
                for _, step in stage.children.items():
                    # print("node2: ", step)
                    for _, job in step.children.items():
                        # print("node3: ", job.data)
                        status = job.data["status"]
                        # print("job ", s)
                        # FIXME: Success shouldn't really exist
                        if status == "Success":
                            status = JOB_STATE_FINISHED
                        return QColor(*REAL_STATE_TO_COLOR[status])

        if node.type == NodeType.REAL and role == RealLabelHint:
            return f"{node.id}"

        if node.type == NodeType.REAL and role == RealStatusColorHint:
            return QColor(*REAL_STATE_TO_COLOR[node.data["status"]])

        else:
            return QVariant()

    def _job_data(self, index: QModelIndex, node: Node, role: int):
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return node.data.get("name")
            elif index.column() == 1:
                return node.data.get("status")
            elif index.column() == 2:
                return node.data.get("start_time")
            elif index.column() == 3:
                return node.data.get("end_time")
            elif index.column() == 4:
                data = node.data.get("data")
                return data.get("current_memory_usage") if data else QVariant()
            elif index.column() == 5:
                data = node.data.get("data")
                return data.get("max_memory_usage") if data else QVariant()
        if role == NodeRole:
            return node
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
