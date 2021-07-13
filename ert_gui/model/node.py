from enum import Enum, auto
from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot, SnapshotDict
from ert_shared.ensemble_evaluator.entity import identifiers as ids


class NodeType(Enum):
    ROOT = auto()
    ITER = auto()
    REAL = auto()
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
