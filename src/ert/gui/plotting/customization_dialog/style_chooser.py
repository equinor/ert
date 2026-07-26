from collections.abc import Iterator
from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QWidget,
)

from ert.gui.plotting.utils import PlotStyle

STYLE_OFF = ("Off", None)
STYLE_AREA = ("Area", "#")
STYLE_SOLID = ("Solid", "-")
STYLE_DASHED = ("Dashed", "--")
STYLE_DOTTED = ("Dotted", ":")
STYLE_DASH_DOTTED = ("Dash dotted", "-.")

STYLESET_DEFAULT = "default"
STYLESET_AREA = "area"
STYLESET_TOGGLE = "toggle_only"

COMPACT_FONT_SIZE_INCREASE = 2.0

STYLES = {
    STYLESET_DEFAULT: [
        STYLE_OFF,
        STYLE_SOLID,
        STYLE_DASHED,
        STYLE_DOTTED,
        STYLE_DASH_DOTTED,
    ],
    STYLESET_AREA: [
        STYLE_OFF,
        STYLE_AREA,
        STYLE_SOLID,
        STYLE_DASHED,
        STYLE_DOTTED,
        STYLE_DASH_DOTTED,
    ],
    STYLESET_TOGGLE: [STYLE_OFF, STYLE_SOLID],
}

MARKER_OFF = ("Off", None)
MARKER_X = ("X", "x")
MARKER_CIRCLE = ("Circle", "o")
MARKER_POINT = ("Point", ".")
MARKER_PIXEL = ("Pixel", ",")
MARKER_PLUS = ("Plus", "+")
MARKER_STAR = ("Star", "*")
MARKER_DIAMOND = ("Diamond", "D")
MARKER_PENTAGON = ("Pentagon", "p")
MARKER_SQUARE = ("Square", "s")
MARKER_HLINE = ("H Line", "_")
MARKER_VLINE = ("V Line", "|")
MARKER_HEXAGON1 = ("Hexagon 1", "h")
MARKER_HEXAGON2 = ("Hexagon 2", "H")

MARKERS: list[tuple[str, str | None]] = [
    MARKER_OFF,
    MARKER_X,
    MARKER_CIRCLE,
    MARKER_POINT,
    MARKER_STAR,
    MARKER_DIAMOND,
    MARKER_PLUS,
    MARKER_PENTAGON,
    MARKER_SQUARE,
    MARKER_HEXAGON1,
    MARKER_HEXAGON2,
]

COMPACT_MARKER_SYMBOLS: dict[str | None, str] = {
    None: "Off",
    "x": "x",
    "o": "○",
    ".": "•",
    "*": "★",
    "D": "◇",
    "+": "+",
    "p": "⬠",
    "s": "□",
    "h": "⬡",
    "H": "⬢",
}


class StyleChooser(QWidget):
    styleChanged = Signal()

    def __init__(
        self, line_style_set: str = STYLESET_DEFAULT, *, compact: bool = False
    ) -> None:
        QWidget.__init__(self)
        self._style = PlotStyle("StyleChooser internal style")

        self._styles: list[tuple[str, str | None]] = (
            STYLES["default"]
            if line_style_set not in STYLES
            else STYLES[line_style_set]
        )

        layout = QHBoxLayout()

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.line_chooser = QComboBox()
        self.line_chooser.setToolTip("Select line style.")
        compact_font = QFont(self.line_chooser.font())
        compact_font.setBold(True)
        compact_font.setPointSizeF(
            compact_font.pointSizeF() + COMPACT_FONT_SIZE_INCREASE
        )
        for full_name, line_style in self._styles:
            display_name = (
                (".." if line_style == ":" else line_style or "Off")
                if compact
                else full_name
            )
            self.line_chooser.addItem(display_name, line_style)

            if compact:
                line_index = self.line_chooser.count() - 1
                self.line_chooser.setItemData(
                    line_index, full_name, Qt.ItemDataRole.ToolTipRole
                )
                self.line_chooser.setItemData(
                    line_index, full_name, Qt.ItemDataRole.AccessibleTextRole
                )
                if line_style is not None:
                    self.line_chooser.setItemData(
                        line_index,
                        compact_font,
                        Qt.ItemDataRole.FontRole,
                    )

        self.marker_chooser = QComboBox()
        self.marker_chooser.setToolTip("Select marker style.")
        for full_name, marker in MARKERS:
            display_name = COMPACT_MARKER_SYMBOLS[marker] if compact else full_name
            self.marker_chooser.addItem(display_name, marker)
            if compact:
                marker_index = self.marker_chooser.count() - 1
                self.marker_chooser.setItemData(
                    marker_index, full_name, Qt.ItemDataRole.ToolTipRole
                )
                self.marker_chooser.setItemData(
                    marker_index, full_name, Qt.ItemDataRole.AccessibleTextRole
                )

        self.thickness_spinner = QDoubleSpinBox()
        self.thickness_spinner.setToolTip("Line thickness")
        self.thickness_spinner.setMinimum(0.1)
        self.thickness_spinner.setDecimals(1)
        self.thickness_spinner.setSingleStep(0.1)

        self.size_spinner = QDoubleSpinBox()
        self.size_spinner.setToolTip("Marker size")
        self.size_spinner.setMinimum(0.1)
        self.size_spinner.setDecimals(1)
        self.size_spinner.setSingleStep(0.1)
        if compact:
            self.line_chooser.setFixedWidth(48)
            self.marker_chooser.setFixedWidth(48)
            self.thickness_spinner.setFixedWidth(55)
            self.size_spinner.setFixedWidth(55)
        else:
            # The text content of the spinner varies, but should not push the
            # full-size chooser out of its dialog boundaries.
            self.line_chooser.setMinimumWidth(110)
            self.setMinimumWidth(140)
            self.setMaximumHeight(25)

        for control in (
            self.line_chooser,
            self.thickness_spinner,
            self.marker_chooser,
            self.size_spinner,
        ):
            layout.addWidget(control)

        self.setLayout(layout)

        self.line_chooser.currentIndexChanged.connect(self._update_style)
        self.marker_chooser.currentIndexChanged.connect(self._update_style)
        self.thickness_spinner.valueChanged.connect(self._update_style)
        self.size_spinner.valueChanged.connect(self._update_style)

        self._update_line_style_and_marker(
            self._style.line_style,
            self._style.marker,
            self._style.width,
            self._style.size,
        )
        self._layout = layout

    def get_item_sizes(self) -> tuple[int, ...]:
        def _iter() -> Iterator[int]:
            for i in range(4):
                item = self._layout.itemAt(i)
                assert item is not None
                yield item.sizeHint().width()

        return tuple(_iter())

    def _find_line_style_index(self, line_style: str) -> int:
        for index, style in enumerate(self._styles):
            if (style[1] == line_style) or (style[1] is None and not line_style):
                return index
        return -1

    @staticmethod
    def _find_marker_style_index(marker: str) -> int:
        for index, style in enumerate(MARKERS):
            if (style[1] == marker) or (style[1] is None and not marker):
                return index
        return -1

    def _update_line_style_and_marker(
        self, line_style: str, marker: str, thickness: float, size: float
    ) -> None:
        self.line_chooser.setCurrentIndex(self._find_line_style_index(line_style))
        self.marker_chooser.setCurrentIndex(self._find_marker_style_index(marker))
        self.thickness_spinner.setValue(thickness)
        self.size_spinner.setValue(size)

    def _update_style(self) -> None:
        line_style = self.line_chooser.currentData()
        marker_style = self.marker_chooser.currentData()

        self.marker_chooser.setEnabled(line_style != STYLE_AREA[1])
        thickness = float(self.thickness_spinner.value())
        size = float(self.size_spinner.value())

        self._style.line_style = line_style
        self._style.marker = marker_style
        self._style.width = thickness
        self._style.size = size
        self.styleChanged.emit()

    @override
    def setStyle(self, style: PlotStyle) -> None:  # type: ignore
        self._style.copy_style_from(style)
        self._update_line_style_and_marker(
            style.line_style, style.marker, style.width, style.size
        )

    def get_style(self) -> PlotStyle:
        style = PlotStyle("Generated style from StyleChooser")
        style.copy_style_from(self._style)
        return style

    def create_label_layout(self, layout: QLayout | None = None) -> QLayout:
        if layout is None:
            layout = QHBoxLayout()

        titles = ["Line style", "Width", "Marker style", "Size"]
        sizes = self.get_item_sizes()
        for title, size in zip(titles, sizes, strict=False):
            label = QLabel(title)
            label.setFixedWidth(size)
            layout.addWidget(label)

        return layout
