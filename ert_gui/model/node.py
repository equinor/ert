from enum import Enum, auto
from ert_shared.ensemble_evaluator.entity.snapshot import _SnapshotDict


class NodeType(Enum):
    ROOT = auto()
    ITER = auto()
    REAL = auto()
    STAGE = auto()
    STEP = auto()
    JOB = auto()


class Node:
    def __init__(self, id_, data, type_) -> None:
        self.parent = None
        self.data = data
        self.children = {}
        self.id = id_
        self.type = type_

    def __repr__(self) -> str:
        parent = "no " if self.parent is None else ""
        children = "no " if len(self.children) == 0 else f"{len(self.children)} "
        return f"Node<{self.type}>@{self.id} with {parent}parent and {children}children"

    def add_child(self, node) -> None:
        node.parent = self
        self.children[node.id] = node

    def row(self):
        if self.parent:
            return list(self.parent.children.keys()).index(self.id)
        raise ValueError("row on root node?")


def snapshot_to_tree(snapshot: _SnapshotDict, iter_: int) -> Node:
    iter_node = Node(iter_, {"status": snapshot.status}, NodeType.ITER)
    for real_id, real in snapshot.reals.items():
        real_node = Node(
            real_id, {"status": real.status, "active": real.active}, NodeType.REAL
        )
        iter_node.add_child(real_node)
        for stage_id, stage in real.stages.items():
            stage_node = Node(stage_id, {"status": stage.status}, NodeType.STAGE)
            real_node.add_child(stage_node)
            for step_id, step in stage.steps.items():
                step_node = Node(step_id, {"status": step.status}, NodeType.STEP)
                stage_node.add_child(step_node)
                for job_id, job in step.jobs.items():
                    job_node = Node(job_id, job.dict(), NodeType.JOB)
                    step_node.add_child(job_node)
    return iter_node
