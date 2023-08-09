from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class NodeType(Enum):
    ROOT = auto()
    ITER = auto()
    REALIZATION = auto()
    STEP = auto()
    JOB = auto()


@dataclass
class Node:
    id: int
    data: Dict[Any, Any]
    type: NodeType
    children: Dict[int, Node] = field(default_factory=dict)
    parent: Optional[Node] = None

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
