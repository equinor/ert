import logging

from ert_gui.model.dev_helpers import create_partial_snapshot, create_snapshot
from ert_gui.model.node import Node, NodeType, snapshot_to_tree
from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, _SnapshotDict
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt, QVariant, Slot
from qtpy.QtGui import QColor

from ert_shared.status.entity.state import JOB_STATE_FINISHED, REAL_STATE_TO_COLOR


logger = logging.getLogger(__name__)


NodeRole = Qt.UserRole + 1
RealJobColorHint = Qt.UserRole + 2
RealStatusColorHint = Qt.UserRole + 3
RealLabelHint = Qt.UserRole + 4
ProgressRole = Qt.UserRole + 5
SimpleProgressRole = Qt.UserRole + 6


class SnapshotModel(QAbstractItemModel):
    @Slot()
    def add_job(self, real=0):
        iter_0_idx = self.index(0, 0, QModelIndex())
        real_0_idx = self.index(real, 0, iter_0_idx)
        stage_0_idx = self.index(0, 0, real_0_idx)
        step_0_idx = self.index(0, 0, stage_0_idx)
        step_0 = step_0_idx.internalPointer()
        next_job_id = len(step_0.children)
        self.beginInsertRows(step_0_idx, next_job_id, next_job_id)
        step_0.add_child(
            Node(
                str(next_job_id),
                {
                    "start_time": str(123),
                    "end_time": str(123),
                    "name": f"poly_job_{next_job_id}",
                    "status": "Unknown",
                    "error": "error",
                    "stdout": "std_out_file",
                    "stderr": "std_err_file",
                    "data": {
                        "current_memory_usage": "123",
                        "max_memory_usage": "312",
                    },
                },
                NodeType.JOB,
            )
        )
        self.endInsertRows()

    @Slot()
    def add_job1(self):
        self.add_job(real=1)

    @Slot()
    def add_snapshot(self):
        next_iter = 0
        snapshot = snapshot_to_tree(create_snapshot(), next_iter)
        self._add_snapshot(snapshot, next_iter)

    @Slot()
    def mutate_snapshot(self):
        iter_0_idx = self.index(0, 0, QModelIndex())
        iter_0 = iter_0_idx.internalPointer()
        next_real_id = len(iter_0.children)
        next_real = Node(str(next_real_id), {"status": "Unknown"}, NodeType.REAL)
        self.beginInsertRows(iter_0_idx, next_real_id, next_real_id)
        iter_0.add_child(next_real)
        self.endInsertRows()

    @Slot()
    def mutate_snapshot_with_partial(self):
        iter_0_idx = self.index(0, 0, QModelIndex())
        iter_0 = iter_0_idx.internalPointer()
        children = list(iter_0.children.values())
        next_real_id = children[len(children) - 1].id

        partial = create_partial_snapshot(next_real_id, "Running")
        self._add_partial_snapshot(partial, 0)

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
        for real_id, real in partial_d["reals"].items():
            real_node = iter_node.children[real_id]
            if real.get("status"):
                real_node.data["status"] = real.get("status")

            real_index = self.index(real_node.row(), 0, iter_index)
            real_index_bottom_right = self.index(
                real_node.row(), self.columnCount(iter_index) - 1, iter_index
            )

            if "stages" not in real:
                continue

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

                    for job_id, job in step["jobs"].items():
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

    def _add_snapshot(self, snapshot: _SnapshotDict, iter_: int):
        snapshot_tree = snapshot_to_tree(snapshot, iter_)
        if iter_ in self.root.children:
            self.modelAboutToBeReset.emit()
            self.root.children[iter_] = snapshot_tree
            snapshot_tree.parent = self.root
            self.modelReset.emit()
            return

        # this should be moved to after beginInsertRows
        self.root.add_child(snapshot_tree)

        parent = QModelIndex()
        self.beginInsertRows(parent, snapshot_tree.row(), snapshot_tree.row())
        self.root.children[iter_] = snapshot_tree
        self.rowsInserted.emit(parent, snapshot_tree.row(), snapshot_tree.row())

    def columnCount(self, parent=QModelIndex()):
        return 6

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()
        return len(parentItem.children)

    def parent(self, index):
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

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

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

    def _job_data(self, index, node, role):
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

    def index(self, row: int, column: int, parent: QModelIndex) -> QModelIndex:
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
