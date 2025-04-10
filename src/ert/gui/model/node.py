from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from ert.ensemble_evaluator.snapshot import FMStepSnapshot


@dataclass
class _NodeBase(ABC):
    id_: str
    _index: int | None = None


def _repr(node: RootNode | IterNode | RealNode | ForwardModelStepNode) -> str:
    parent = "no " if node.parent is None else ""
    children = "no " if not node.children else f"{len(node.children)} "
    return (
        f"Node<{type(node).__name__}>@{node.id_} "
        f"with {parent}parent and {children}children"
    )


def _row(node: RootNode | IterNode | RealNode | ForwardModelStepNode) -> int:
    if not node._index:
        if node.parent:
            node._index = list(node.parent.children.keys()).index(node.id_)
        else:
            raise ValueError(f"{node} had no parent")
    return node._index


@dataclass
class RootNode(_NodeBase):
    parent: None = field(default=None, init=False)
    children: dict[str, IterNode] = field(default_factory=dict)
    max_memory_usage: int | None = None

    def add_child(self, node: IterNode) -> None:
        node.parent = self
        self.children[node.id_] = node

    def row(self) -> int:
        return _row(self)

    def __repr__(self) -> str:
        return _repr(self)


@dataclass
class IterNodeData:
    index: str | None = None
    status: str | None = None


@dataclass
class IterNode(_NodeBase):
    parent: RootNode | None = None
    data: IterNodeData = field(default_factory=IterNodeData)
    children: dict[str, RealNode] = field(default_factory=dict)

    def add_child(self, node: RealNode) -> None:
        node.parent = self
        self.children[node.id_] = node

    def row(self) -> int:
        return _row(self)

    def __repr__(self) -> str:
        return _repr(self)


@dataclass
class RealNodeData:
    status: str | None = None
    active: bool | None = False
    fm_step_status_by_id: dict[str, str] = field(default_factory=dict)
    real_status: str | None = None
    current_memory_usage: int | None = None
    max_memory_usage: int | None = None
    exec_hosts: str | None = None
    stderr: str | None = None
    message: str | None = None


@dataclass
class RealNode(_NodeBase):
    parent: IterNode | None = None
    data: RealNodeData = field(default_factory=RealNodeData)
    children: dict[str, ForwardModelStepNode] = field(default_factory=dict)

    def add_child(self, node: ForwardModelStepNode) -> None:
        node.parent = self
        self.children[node.id_] = node

    def row(self) -> int:
        return _row(self)

    def __repr__(self) -> str:
        return _repr(self)


@dataclass
class ForwardModelStepNode(_NodeBase):
    parent: RealNode | None = None
    data: FMStepSnapshot = field(default_factory=lambda: FMStepSnapshot())  # noqa: PLW0108
    children: dict[str, None] = field(default_factory=dict)

    def row(self) -> int:
        return _row(self)

    def __repr__(self) -> str:
        return _repr(self)
