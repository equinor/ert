from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    def __init__(
        self, title: str, content_layout: QVBoxLayout, *, expanded: bool = False
    ) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self._toggle_button = QToolButton()
        self._toggle_button.setStyleSheet(
            "QToolButton { border: none; text-align: left; padding: 4px 2px;"
            " font-style: italic; }"
            "QToolButton:hover { background-color: rgba(128, 128, 128, 40); }"
        )
        self._toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle_button.setText(title)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)

        self._content_widget = QWidget()
        self._content_widget.setLayout(content_layout)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._toggle_button)
        main_layout.addWidget(self._content_widget)

        self._toggle_button.toggled.connect(self._on_toggled)
        self._toggle_button.setChecked(expanded)
        self._on_toggled(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content_widget.setVisible(checked)

    def set_title(self, title: str) -> None:
        self._toggle_button.setText(title)
