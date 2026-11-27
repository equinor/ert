from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ert.config import Field, GenKwConfig, ParameterConfig, SurfaceConfig


class _LazyItem(QTreeWidgetItem):
    """Tree item whose children are only created on first expansion."""

    def __init__(
        self,
        parent: QTreeWidget | QTreeWidgetItem,
        label: str,
        populate: Callable[[QTreeWidgetItem], None] | None = None,
    ) -> None:
        super().__init__(parent, [label])
        self._populate = populate
        if populate is not None:
            self.setChildIndicatorPolicy(
                QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
            )

    def ensure_populated(self) -> None:
        if self._populate is not None:
            populate, self._populate = self._populate, None
            populate(self)
            self.setChildIndicatorPolicy(
                QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicatorWhenChildless
            )


def _add_details(item: QTreeWidgetItem, parameter: ParameterConfig) -> None:
    _LazyItem(item, f"Update: {parameter.update_strategy}")
    _LazyItem(item, f"Forward Init: {parameter.forward_init}")
    if parameter.group_name:
        _LazyItem(item, f"Group: {parameter.group_name.upper()}")
    if hasattr(parameter, "input_source"):
        _LazyItem(item, f"Source: {parameter.input_source}")


def _add_parameters(item: QTreeWidgetItem, parameters: list[ParameterConfig]) -> None:
    for parameter in sorted(parameters, key=lambda p: p.name.lower()):
        _LazyItem(item, parameter.name, partial(_add_details, parameter=parameter))


def _top_level_nodes(
    parameters: Sequence[ParameterConfig],
) -> list[tuple[str, Callable[[QTreeWidgetItem], None]]]:
    """Group parameters the same way as the summary has always listed them:
    GEN_KW parameters per group with a count, and fields and surfaces with
    their dimensions.
    """
    nodes: list[tuple[str, Callable[[QTreeWidgetItem], None]]] = []
    gen_kw_groups: dict[str, list[ParameterConfig]] = defaultdict(list)
    for parameter in parameters:
        match parameter:
            case GenKwConfig():
                gen_kw_groups[parameter.group_name or "gen_kw"].append(parameter)
            case Field(name=name, ertbox_params=box):
                nodes.append(
                    (
                        f"{name} ({box.nx}, {box.ny}, {box.nz})",
                        partial(_add_details, parameter=parameter),
                    )
                )
            case SurfaceConfig(name=name, ncol=ncol, nrow=nrow):
                nodes.append(
                    (
                        f"{name} ({ncol}, {nrow})",
                        partial(_add_details, parameter=parameter),
                    )
                )
            case _:
                nodes.append(
                    (parameter.name, partial(_add_details, parameter=parameter))
                )
    nodes.extend(
        (
            f"{group} ({len(group_parameters)})",
            partial(_add_parameters, parameters=group_parameters),
        )
        for group, group_parameters in gen_kw_groups.items()
    )
    return sorted(nodes, key=lambda node: node[0].lower())


def parameter_count(parameters: Sequence[ParameterConfig]) -> int:
    return sum(
        len(parameter) if isinstance(parameter, Field | SurfaceConfig) else 1
        for parameter in parameters
    )


class ParameterTree(QTreeWidget):
    """Parameters grouped like the summary, collapsed by default.

    Children are created on first expansion so that large parameter sets
    do not slow down building the tree.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Take the space that is available rather than requiring a minimum
        # height; the tree scrolls when its content does not fit.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.itemExpanded.connect(self._on_item_expanded)

    def set_parameters(self, parameters: Sequence[ParameterConfig]) -> None:
        self.clear()
        for label, populate in _top_level_nodes(parameters):
            _LazyItem(self, label, populate)

    @staticmethod
    def _on_item_expanded(item: QTreeWidgetItem) -> None:
        if isinstance(item, _LazyItem):
            item.ensure_populated()


class ParametersSummary(QWidget):
    """Parameter count with expandable trees of updatable and
    non-updatable parameters.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._signature: tuple[tuple[str, str, object], ...] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title)
        # 8px is the EDS small spacing.
        layout.addSpacing(8)

        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        self.updatable_tree = ParameterTree()
        self.updatable_tree.setObjectName("updatable_parameters")
        self.non_updatable_tree = ParameterTree()
        self.non_updatable_tree.setObjectName("non_updatable_parameters")
        for header, tree in (
            ("Updatable", self.updatable_tree),
            ("Non-updatable", self.non_updatable_tree),
        ):
            column = QVBoxLayout()
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            column.addWidget(label)
            column.addWidget(tree, 1)
            columns.addLayout(column)
        layout.addLayout(columns, 1)

    def set_parameters(self, parameters: Sequence[ParameterConfig]) -> None:
        signature = tuple((p.type, p.name, p.update_strategy) for p in parameters)
        # Panels report configuration changes on every edit, so only rebuild
        # (and lose the expansion state) when the parameters actually change.
        if signature == self._signature:
            return
        self._signature = signature

        self.title.setText(f"Parameters ({parameter_count(parameters):,})")
        self.updatable_tree.set_parameters(
            [p for p in parameters if p.update_strategy is not None]
        )
        self.non_updatable_tree.set_parameters(
            [p for p in parameters if p.update_strategy is None]
        )
