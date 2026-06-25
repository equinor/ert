from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def create_side_panel(title: str, widget: QWidget) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)

    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setStyleSheet("padding-bottom: 7px;")

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
