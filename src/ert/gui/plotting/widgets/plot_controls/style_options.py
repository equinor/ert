from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.plotting.customization_dialog.style_chooser import (
    STYLESET_DEFAULT,
    STYLESET_TOGGLE,
)
from ert.gui.plotting.utils import PlotConfig, PlotStyle

from .style_edit import STYLE_NAME_WIDTH, _StyleEdit


class StyleOptions(QWidget):
    def __init__(
        self,
        connection_point: Callable[..., object],
    ) -> None:
        super().__init__()
        self._connection_point = connection_point
        self.setObjectName("style_options_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolTip("Show style options")
        self._toggle.toggled.connect(self._set_content_visible)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self._toggle)
        header_layout.addWidget(QLabel("Style options"))
        header_layout.addStretch()
        layout.addWidget(header)

        self._content = QWidget()
        self._content.setObjectName("style_options")
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        layout.addWidget(self._content)
        self._content.setVisible(False)

        default_config = PlotConfig()
        self._style_edits = {
            "default": _StyleEdit(
                connection_point=connection_point,
                initial_style=default_config.default_style(),
                title="Default style",
                object_name="default_style_edit",
                line_style_set=STYLESET_DEFAULT,
            ),
            "history": _StyleEdit(
                connection_point=connection_point,
                initial_style=default_config.history_style(),
                title="History style",
                object_name="history_style_edit",
                line_style_set=STYLESET_DEFAULT,
            ),
            "observations": _StyleEdit(
                connection_point=connection_point,
                initial_style=default_config.observations_style(),
                title="Observations style",
                object_name="observations_style_edit",
                line_style_set=STYLESET_TOGGLE,
                display_name="Obs.",
                tool_tip="Observations",
            ),
        }

        self._style_edits["history"].setVisible(False)
        self._style_edits["observations"].setVisible(False)

        self._individual_style_options = QGroupBox()
        self._individual_style_options.setObjectName("individual_style_options")
        individual_layout = QVBoxLayout(self._individual_style_options)
        individual_layout.setContentsMargins(0, 0, 0, 0)
        individual_layout.setSpacing(2)

        headers_layout = QHBoxLayout()
        headers_layout.setContentsMargins(0, 0, 0, 0)
        headers_layout.setSpacing(2)

        style_header = QWidget()
        style_header.setFixedWidth(STYLE_NAME_WIDTH)
        headers_layout.addWidget(style_header)

        for title, width in (
            ("Line", 48),
            ("Width", 55),
            ("Marker", 48),
            ("Size", 55),
        ):
            header = QLabel(title)
            header.setFixedWidth(width)
            headers_layout.addWidget(header)

        individual_layout.addLayout(headers_layout)

        for style_edit in self._style_edits.values():
            individual_layout.addWidget(style_edit)

        self._reset_button = QPushButton("Reset")
        self._reset_button.setObjectName("reset_individual_styles_button")
        self._reset_button.setToolTip("Reset default, history, and observation styles")
        self._reset_button.clicked.connect(self._reset_styles)

        reset_layout = QHBoxLayout()
        reset_layout.setContentsMargins(0, 0, 0, 0)
        reset_layout.addStretch()
        reset_layout.addWidget(self._reset_button)
        individual_layout.addLayout(reset_layout)

        content_layout.addWidget(self._individual_style_options)

    def get_widget(self) -> QWidget:
        return self

    def _set_content_visible(self, visible: bool) -> None:
        self._content.setVisible(visible)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        self._toggle.setToolTip(
            "Hide style options" if visible else "Show style options"
        )

    def set_history_available(self, available: bool) -> None:
        self._style_edits["history"].setVisible(available)

    def set_observations_available(self, available: bool) -> None:
        self._style_edits["observations"].setVisible(available)

    def get_history_style(self) -> PlotStyle:
        return self._style_edits["history"].get_style()

    def get_default_style(self) -> PlotStyle:
        return self._style_edits["default"].get_style()

    def get_observations_style(self) -> PlotStyle:
        return self._style_edits["observations"].get_style()

    def _reset_styles(self) -> None:
        for style_edit in self._style_edits.values():
            style_edit.reset_style()
        self._connection_point()
