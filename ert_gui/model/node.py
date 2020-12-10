from enum import Enum, auto
from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot
from ert_shared.ensemble_evaluator.entity import identifiers as ids


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
        raise ValueError(f"{self} had no parent")


def snapshot_to_tree(snapshot: Snapshot, iter_: int) -> Node:
    iter_node = Node(iter_, {ids.STATUS: snapshot.get_status()}, NodeType.ITER)
    for real_id in sorted(snapshot.get_reals(), key=int):
        real = snapshot.get_reals()[real_id]
        real_node = Node(
            real_id,
            {ids.STATUS: real.status, ids.ACTIVE: real.active},
            NodeType.REAL,
        )
        iter_node.add_child(real_node)
        # TODO: sort stages, but wait till after https://github.com/equinor/ert/issues/1220 ?
        for stage_id, stage in real.stages.items():
            stage_node = Node(stage_id, {ids.STATUS: stage.status}, NodeType.STAGE)
            real_node.add_child(stage_node)
            # TODO: sort steps, but wait till after https://github.com/equinor/ert/issues/1220 ?
            for step_id, step in stage.steps.items():
                step_node = Node(step_id, {ids.STATUS: step.status}, NodeType.STEP)
                stage_node.add_child(step_node)
                for job_id in sorted(step.jobs, key=int):
                    job = step.jobs[job_id]
                    job_dict = dict(job)
                    job_dict[ids.DATA] = dict(job[ids.DATA])
                    job_node = Node(job_id, job_dict, NodeType.JOB)
                    step_node.add_child(job_node)
    return iter_node
