from collections.abc import Callable

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.plotting.customization_dialog.style_chooser import (
    STYLESET_AREA,
    STYLESET_DEFAULT,
)
from ert.gui.plotting.utils import PlotConfig, PlotStyle

from .style_edit import STYLE_NAME_WIDTH, _StyleEdit

STATISTICS_STYLE_SETS = {
    "mean": STYLESET_DEFAULT,
    "p50": STYLESET_DEFAULT,
    "std": STYLESET_AREA,
    "min-max": STYLESET_AREA,
    "p10-p90": STYLESET_AREA,
    "p33-p67": STYLESET_AREA,
}

PRESET_NAMES = [
    "Statistics default",
    "Cross ensemble statistics default",
    "Overview",
    "All statistics",
]

PRESET_STYLE_UPDATES: dict[int, dict[str, tuple[str | None, str | None]]] = {
    0: {
        "mean": ("-", None),
        "p50": (None, None),
        "std": (None, None),
        "min-max": (None, None),
        "p10-p90": ("--", None),
        "p33-p67": (None, None),
    },
    1: {
        "mean": ("-", "o"),
        "p50": (None, None),
        "std": ("--", "D"),
        "min-max": (None, None),
        "p10-p90": (None, None),
        "p33-p67": (None, None),
    },
    2: {
        "mean": (None, None),
        "p50": (None, None),
        "std": (None, None),
        "min-max": ("area", None),
        "p10-p90": (None, None),
        "p33-p67": (None, None),
    },
    3: {
        "mean": ("-", None),
        "p50": ("--", "x"),
        "std": (":", None),
        "min-max": ("--", None),
        "p10-p90": ("area", None),
        "p33-p67": ("area", None),
    },
}


class StatisticsOptions(QWidget):
    def __init__(self, connection_point: Callable[..., object]) -> None:
        super().__init__()
        self._connection_point = connection_point
        self.setObjectName("statistics_options_panel")

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
        self._content.setObjectName("statistics_options")
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        layout.addWidget(self._content)
        self._content.setVisible(False)

        self._statistics_controls = QWidget()
        statistics_controls_layout = QVBoxLayout(self._statistics_controls)
        statistics_controls_layout.setContentsMargins(0, 0, 0, 0)
        statistics_controls_layout.setSpacing(4)

        presets_row = QWidget()
        presets_layout = QHBoxLayout(presets_row)
        presets_layout.setContentsMargins(0, 0, 0, 0)
        presets_layout.addWidget(QLabel("Presets"))
        self._presets = QComboBox()
        self._presets.setObjectName("statistics_presets")
        for preset in PRESET_NAMES:
            self._presets.addItem(preset)
        self._presets.currentIndexChanged.connect(self._preset_selected)
        presets_layout.addWidget(self._presets)
        statistics_controls_layout.addWidget(presets_row)

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
            header_label = QLabel(title)
            header_label.setFixedWidth(width)
            headers_layout.addWidget(header_label)
        statistics_controls_layout.addLayout(headers_layout)

        default_config = PlotConfig()
        self._style_edits = {}
        for style_name, line_style_set in STATISTICS_STYLE_SETS.items():
            initial_style = default_config.get_statistics_style(style_name)
            style_edit = _StyleEdit(
                connection_point=connection_point,
                initial_style=initial_style,
                title=f"{initial_style.name} style",
                object_name=f"statistics_{style_name.replace('-', '_')}_style_edit",
                line_style_set=line_style_set,
                display_name=initial_style.name,
            )
            self._style_edits[style_name] = style_edit
            statistics_controls_layout.addWidget(style_edit)

        self._apply_preset(0)

        std_dev_row = QWidget()
        std_dev_layout = QHBoxLayout(std_dev_row)
        std_dev_layout.setContentsMargins(0, 0, 0, 0)
        std_dev_layout.addWidget(QLabel("Std dev multiplier"))
        self._std_dev_factor = QSpinBox()
        self._std_dev_factor.setObjectName("statistics_std_dev_factor")
        self._std_dev_factor.setMinimum(1)
        self._std_dev_factor.setMaximum(3)
        self._std_dev_factor.setValue(default_config.get_standard_deviation_factor())
        self._std_dev_factor.valueChanged.connect(connection_point)
        std_dev_layout.addWidget(self._std_dev_factor)
        std_dev_layout.addStretch()
        statistics_controls_layout.addWidget(std_dev_row)
        self._std_dev_row = std_dev_row

        self._distribution_lines = QCheckBox("Connection lines")
        self._distribution_lines.setObjectName("distribution_lines_checkbox")
        self._distribution_lines.setToolTip(
            "Toggle distribution connection lines visibility."
        )
        self._distribution_lines.setChecked(
            default_config.is_distribution_line_enabled()
        )
        self._distribution_lines.stateChanged.connect(connection_point)

        self._reset_button = QPushButton("Reset")
        self._reset_button.setObjectName("reset_statistics_styles_button")
        self._reset_button.setToolTip("Reset all statistics settings")
        self._reset_button.clicked.connect(self._reset_statistics)
        reset_layout = QHBoxLayout()
        reset_layout.setContentsMargins(0, 0, 0, 0)
        reset_layout.addStretch()
        reset_layout.addWidget(self._reset_button)

        content_layout.addWidget(self._statistics_controls)
        content_layout.addWidget(self._distribution_lines)
        content_layout.addLayout(reset_layout)

        self._statistics_available = False
        self._distribution_lines_available = False
        self.set_options_available(statistics=False, distribution_lines=False)

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

    def _preset_selected(self, index: int) -> None:
        self._apply_preset(index)
        self._connection_point()

    def _apply_preset(self, index: int) -> None:
        for style_name, (line_style, marker_style) in PRESET_STYLE_UPDATES.get(
            index, {}
        ).items():
            self._update_style(style_name, line_style, marker_style)

    def _style_edit_for(self, style_name: str) -> _StyleEdit:
        return self._style_edits[style_name]

    def _update_style(
        self,
        style_name: str,
        line_style: str | None,
        marker_style: str | None,
    ) -> None:
        style_edit = self._style_edit_for(style_name)
        style = style_edit.get_style()
        style.line_style = line_style or ""
        style.marker = marker_style or ""
        style_edit.set_style(style)

    def get_statistics_style(self, style_name: str) -> PlotStyle:
        return self._style_edit_for(style_name).get_style()

    def get_standard_deviation_factor(self) -> int:
        return self._std_dev_factor.value()

    def is_distribution_line_enabled(self) -> bool:
        return self._distribution_lines.isChecked()

    def apply_to_plot_config(self, plot_config: PlotConfig) -> None:
        for style_name in self._style_edits:
            plot_config.set_statistics_style(
                style_name,
                self.get_statistics_style(style_name),
            )
        plot_config.set_standard_deviation_factor(self.get_standard_deviation_factor())
        plot_config.set_distribution_line_enabled(self.is_distribution_line_enabled())

    def set_options_available(
        self, *, statistics: bool, distribution_lines: bool
    ) -> None:
        self._statistics_available = statistics
        self._distribution_lines_available = distribution_lines
        self._statistics_controls.setVisible(statistics)
        self._distribution_lines.setVisible(distribution_lines)

    def set_statistics_available(self, available: bool) -> None:
        self.set_options_available(
            statistics=available,
            distribution_lines=self._distribution_lines_available,
        )

    def set_distribution_lines_available(self, available: bool) -> None:
        self.set_options_available(
            statistics=self._statistics_available,
            distribution_lines=available,
        )

    def _reset_statistics(self) -> None:
        default_config = PlotConfig()

        for style_name in self._style_edits:
            self._style_edit_for(style_name).set_style(
                default_config.get_statistics_style(style_name)
            )
        self._apply_preset(self._presets.currentIndex())

        with QSignalBlocker(self._std_dev_factor):
            self._std_dev_factor.setValue(
                default_config.get_standard_deviation_factor()
            )
        with QSignalBlocker(self._distribution_lines):
            self._distribution_lines.setChecked(
                default_config.is_distribution_line_enabled()
            )
        self._connection_point()
