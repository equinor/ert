from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, cast

from qtpy.QtGui import QColor

from ert.ensemble_evaluator.snapshot import FMStepSnapshot


@dataclass
class _Node(ABC):
    id_: str
    parent: Optional[RootNode | IterNode | RealNode] = None
    children: (
        dict[str, IterNode] | dict[str, RealNode] | dict[str, ForwardModelStepNode]
    ) = field(default_factory=dict)
    _index: Optional[int] = None

    def __repr__(self) -> str:
        parent = "no " if self.parent is None else ""
        children = "no " if len(self.children) == 0 else f"{len(self.children)} "
        return f"Node<{type(self).__name__}>@{self.id_} with {parent}parent and {children}children"

    @abstractmethod
    def add_child(self, node: _Node) -> None:
        pass

    def row(self) -> int:
        if not self._index:
            if self.parent:
                self._index = list(self.parent.children.keys()).index(self.id_)
            else:
                raise ValueError(f"{self} had no parent")
        return self._index


@dataclass
class RootNode(_Node):
    parent: None = field(default=None, init=False)
    children: dict[str, IterNode] = field(default_factory=dict)
    max_memory_usage: Optional[int] = None

    def add_child(self, node: _Node) -> None:
        node = cast(IterNode, node)
        node.parent = self
        self.children[node.id_] = node


@dataclass
class IterNodeData:
    index: Optional[str] = None
    status: Optional[str] = None


@dataclass
class IterNode(_Node):
    parent: Optional[RootNode] = None
    data: IterNodeData = field(default_factory=IterNodeData)
    children: dict[str, RealNode] = field(default_factory=dict)

    def add_child(self, node: _Node) -> None:
        node = cast(RealNode, node)
        node.parent = self
        self.children[node.id_] = node


@dataclass
class RealNodeData:
    status: Optional[str] = None
    active: Optional[bool] = False
    fm_step_status_color_by_id: dict[str, QColor] = field(default_factory=dict)
    real_status_color: Optional[QColor] = None
    current_memory_usage: Optional[int] = None
    max_memory_usage: Optional[int] = None
    stderr: Optional[str] = None
    callback_status_message: Optional[str] = None


@dataclass
class RealNode(_Node):
    parent: Optional[IterNode] = None
    data: RealNodeData = field(default_factory=RealNodeData)
    children: dict[str, ForwardModelStepNode] = field(default_factory=dict)

    def add_child(self, node: _Node) -> None:
        node = cast(ForwardModelStepNode, node)
        node.parent = self
        self.children[node.id_] = node


@dataclass
class ForwardModelStepNode(_Node):
    parent: Optional[RealNode]
    data: FMStepSnapshot = field(default_factory=lambda: FMStepSnapshot())  # noqa: PLW0108

    def add_child(self, node: _Node) -> None:
        pass
