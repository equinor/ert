from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from qtpy.QtGui import QColor

from ert.ensemble_evaluator.snapshot import ForwardModel


@dataclass
class _Node(ABC):
    id_: int
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
    ) -> None:
        pass

    def row(self) -> int:
        if self.parent:
            return list(self.parent.children.keys()).index(self.id_)
        raise ValueError(f"{self} had no parent")


@dataclass
class RootNode(_Node):
    parent: None = field(default=None, init=False)
    children: dict[int, IterNode] = field(default_factory=dict)
    max_memory_usage: Optional[int] = None

    def add_child(self, node: IterNode) -> None:
        node.parent = self
        self.children[node.id_] = node


@dataclass
class IterNodeData:
    index: Optional[str] = None
    status: Optional[str] = None


@dataclass
class IterNode(_Node):
    parent: RootNode
    data: IterNodeData = field(default_factory=IterNodeData)
    children: dict[str, RealNode] = field(default_factory=dict)

    def add_child(self, node: RealNode) -> None:
        node.parent = self
        self.children[str(node.id_)] = node


@dataclass
class RealNodeData:
    status: Optional[str] = None
    active: Optional[bool] = False
    forward_model_step_status_color_by_id: dict[str, QColor] = field(
        default_factory=dict
    )
    real_status_color: Optional[QColor] = None
    current_memory_usage: Optional[int] = None
    max_memory_usage: Optional[int] = None


@dataclass
class RealNode(_Node):
    parent: IterNode
    data: RealNodeData = field(default_factory=RealNodeData)
    children: dict[str, ForwardModelStepNode] = field(default_factory=dict)

    def add_child(self, node: ForwardModelStepNode) -> None:
        node.parent = self
        self.children[str(node.id_)] = node


@dataclass
class ForwardModelStepNode(_Node):
    parent: RealNode
    data: ForwardModel = field(default_factory=ForwardModel)

    def add_child(self, _):
        pass
