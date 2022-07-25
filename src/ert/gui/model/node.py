from enum import Enum, auto
from typing import Dict, Optional


class NodeType(Enum):
    ROOT = auto()
    ITER = auto()
    REAL = auto()
    STEP = auto()
    JOB = auto()


class Node:
    def __init__(self, id_: int, data: Dict, type_: NodeType) -> None:
        self.parent: Optional[Node] = None
        self.data: Dict = data
        self.children: Dict[int, Node] = {}
        self.id = id_
        self.type = type_

    def __repr__(self) -> str:
        parent = "no " if self.parent is None else ""
        children = "no " if len(self.children) == 0 else f"{len(self.children)} "
        return f"Node<{self.type}>@{self.id} with {parent}parent and {children}children"

    def add_child(self, node: "Node", node_id: Optional[int] = None) -> None:
        node.parent = self
        if node_id is None:
            node_id = node.id
        self.children[node_id] = node

    def row(self) -> int:
        if "index" in self.data:
            return int(self.data["index"])
        if self.parent:
            return list(self.parent.children.keys()).index(self.id)
        raise ValueError(f"{self} had no parent")
