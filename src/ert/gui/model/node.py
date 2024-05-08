from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from qtpy.QtGui import QColor

from ert.ensemble_evaluator.snapshot import ForwardModel


class NodeType(Enum):
    ROOT = auto()
    ITER = auto()
    REAL = auto()
    JOB = auto()


@dataclass
class _Node(ABC):
    id_: int
    data: dict[Any, Any] = field(default_factory=dict)
    parent: Optional[RootNode | IterNode | RealNode] = None
    children: dict[int, IterNode | RealNode | ForwardModelStepNode] = field(
        default_factory=dict
    )

    def __repr__(self) -> str:
        parent = "no " if self.parent is None else ""
        children = "no " if len(self.children) == 0 else f"{len(self.children)} "
        return f"Node<{type(self).__name__}>@{self.id_} with {parent}parent and {children}children"

    @abstractmethod
    def add_child(
        self,
        node: IterNode | RealNode | ForwardModelStepNode,
        node_id: Optional[int] = None,
    ) -> None:
        pass

    def row(self) -> int:
        if "index" in self.data:
            return int(self.data["index"])
        if self.parent:
            return list(self.parent.children.keys()).index(self.id_)
        raise ValueError(f"{self} had no parent")


@dataclass
class RootNodeData:
    current_memory_usage: Optional[int] = None
    max_memory_usage: Optional[int] = None


@dataclass
class RootNode(_Node):
    parent: None = field(default=None, init=False)
    children: dict[int, IterNode] = field(default_factory=dict)
    data: RootNodeData = field(default_factory=RootNodeData)

    def add_child(self, node: IterNode, node_id: Optional[int] = None) -> None:
        node.parent = self
        if node_id is None:
            node_id = node.id_
        self.children[node_id] = node


@dataclass
class IterNodeData:
    index: Optional[str] = None
    status: Optional[str] = None
    sorted_realization_ids: list[str] = field(default_factory=list)
    sorted_forward_model_step_ids_by_realization_id: dict[str, list[str]] = field(
        default_factory=dict
    )
    current_memory_usage: Optional[int] = None
    max_memory_usage: Optional[int] = None


@dataclass
class IterNode(_Node):
    parent: RootNode
    data: IterNodeData
    children: dict[int, RealNode] = field(default_factory=dict)

    def add_child(self, node: RealNode, node_id: Optional[int] = None) -> None:
        node.parent = self
        if node_id is None:
            node_id = node.id_
        self.children[node_id] = node

    def row(self) -> int:
        if self.data.index is not None:
            return int(self.data.index)
        if self.parent:
            return list(self.parent.children.keys()).index(self.id_)
        raise ValueError(f"{self} had no parent")


@dataclass
class RealNodeData:
    index: Optional[str] = None
    status: Optional[str] = None
    active: Optional[bool] = False
    forward_model_step_status_color_by_id: dict[str, QColor] = field(
        default_factory=dict
    )
    real_status_color: Optional[str] = None
    current_memory_usage: Optional[int] = None
    max_memory_usage: Optional[int] = None


@dataclass
class RealNode(_Node):
    parent: IterNode
    data: RealNodeData
    children: dict[int, ForwardModelStepNode] = field(default_factory=dict)

    def add_child(
        self, node: ForwardModelStepNode, node_id: Optional[int] = None
    ) -> None:
        node.parent = self
        if node_id is None:
            node_id = node.id_
        self.children[node_id] = node

    def row(self) -> int:
        if self.data.index is not None:
            return int(self.data.index)
        if self.parent:
            return list(self.parent.children.keys()).index(self.id_)
        raise ValueError(f"{self} had no parent")


@dataclass
class ForwardModelStepNode(_Node):
    parent: RealNode
    data: ForwardModel

    def add_child(self, *args, **kwargs):
        pass
