from __future__ import annotations

from collections.abc import Callable
from logging import Logger

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ert.gui.plotting.utils.logging_utils import log_plot_option_usage_once


def create_side_panel(title: str, widget: QWidget) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)

    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setStyleSheet("padding-bottom: 7px;")
    title_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    layout.addWidget(title_label)
    layout.addWidget(widget)
    return panel


def create_group_layout(widgets: list[QWidget] | None = None) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    for w in widgets or []:
        layout.addWidget(w)
    return layout


def create_group_box(title: str, layout: QVBoxLayout) -> QGroupBox:
    group_box = QGroupBox(title)
    group_box.setStyleSheet("QGroupBox { font-style: italic; }")
    group_box.setLayout(layout)
    return group_box


def create_checkbox_with_tooltip(
    name: str,
    tooltip: str,
    connection_point: Callable[..., object],
    *,
    initial_checked: bool = True,
    logger: Logger,
) -> QCheckBox:
    checkbox = QCheckBox(name)
    checkbox.setObjectName(f"{name.lower().replace(' ', '_')}_checkbox")
    checkbox.setToolTip(tooltip)
    checkbox.setChecked(initial_checked)
    checkbox.stateChanged.connect(connection_point)
    log_plot_option_usage_once(checkbox.clicked, logger, name)
    return checkbox


def create_spinbox_with_tooltip(
    name: str,
    tooltip: str,
    connection_point: Callable[..., object],
    *,
    minimum: int,
    maximum: int,
    initial_value: int,
    logger: Logger,
) -> QSpinBox:
    spinbox = QSpinBox()
    spinbox.setObjectName(f"{name.lower().replace(' ', '_')}_spinbox")
    spinbox.setToolTip(tooltip)
    spinbox.setRange(minimum, maximum)
    spinbox.setValue(initial_value)
    spinbox.valueChanged.connect(connection_point)
    log_plot_option_usage_once(spinbox.valueChanged, logger, name)
    return spinbox


def create_labeled_row(label: str, widget: QWidget) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(label))
    layout.addWidget(widget)
    layout.addStretch()
    return row
