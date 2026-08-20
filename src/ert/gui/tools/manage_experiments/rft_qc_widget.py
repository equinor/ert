from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.backend_bases import MouseEvent, PickEvent
from matplotlib.backends.backend_qt5agg import FigureCanvas  # type: ignore
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PyQt6.QtCore import (
    QModelIndex,
    QRect,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
)
from PyQt6.QtGui import QBrush, QColor, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.config.rft_config import RFTConfig
from ert.gui.detect_mode import is_dark_mode
from ert.runpaths import Runpaths
from ert.storage import Ensemble
from ert.trace import trace, tracer

if TYPE_CHECKING:
    from ert.config import ErtConfig

_FILTER_WIDTH = 200
_DETAILS_WIDTH = 310


def divider_color() -> str:
    DIVIDER_COLOR_DARK = "#2d2d2d"
    DIVIDER_COLOR_LIGHT = "#b0b0b0"
    return DIVIDER_COLOR_DARK if is_dark_mode() else DIVIDER_COLOR_LIGHT


class _PointStatus(StrEnum):
    MATCHED = "Observation with response"
    INVALID_ZONE = "Observation in unexpected zone"
    NOT_IN_GRID = "Observation not in grid"
    NO_RESPONSE = "Observation without response"
    RESPONSE = "Response known by ERT"
    APPROXIMATED = "Response approximated by ERT"
    FILE_RFT = "RFT from file"


_OBSERVATION_STATUSES = frozenset(
    {
        _PointStatus.MATCHED,
        _PointStatus.INVALID_ZONE,
        _PointStatus.NOT_IN_GRID,
        _PointStatus.NO_RESPONSE,
    }
)

_RESPONSE_STATUSES = frozenset(
    {
        _PointStatus.RESPONSE,
        _PointStatus.APPROXIMATED,
    }
)


TRANSPARENT = "None"

GREY = "#999999"

# Okabe-Ito Palette
OKABE_ORANGE = "#E69F00"
OKABE_SKY_BLUE = "#56B4E9"
OKABE_GREEN = "#009E73"
OKABE_YELLOW = "#F0E442"
OKABE_BLUE = "#0072B2"
OKABE_VERMILLION = "#D55E00"
OKABE_PURPLE = "#CC79A7"
OKABE_BLACK = "#000000"


SELECTION_RING_COLOR = OKABE_BLUE
HOVER_RING_COLOR = OKABE_BLACK

_DEFAULT_STYLE: dict[str, str | float] = {
    "facecolors": "gray",
    "edgecolors": "gray",
    "linewidths": 0.5,
    "s": 20.25,
}

_OVERLAY_STYLE: dict[str, str | float] = {
    "facecolors": TRANSPARENT,
    "edgecolors": "gray",
    "linewidths": 1.5,
    "s": 72.25,
}


_POINT_STYLE: dict[str, dict[str, str | float]] = {
    _PointStatus.MATCHED: {
        **_DEFAULT_STYLE,
        "facecolors": OKABE_GREEN,
        "edgecolors": OKABE_GREEN,
        "s": 56.25,
        "linewidths": 2.5,
    },
    _PointStatus.INVALID_ZONE: {
        **_DEFAULT_STYLE,
        "facecolors": OKABE_VERMILLION,
        "edgecolors": OKABE_VERMILLION,
        "s": 56.25,
        "linewidths": 2.5,
    },
    _PointStatus.NOT_IN_GRID: {
        **_DEFAULT_STYLE,
        "facecolors": OKABE_VERMILLION,
        "edgecolors": OKABE_VERMILLION,
        "s": 56.25,
        "linewidths": 2.5,
    },
    _PointStatus.NO_RESPONSE: {
        **_DEFAULT_STYLE,
        "facecolors": OKABE_VERMILLION,
        "edgecolors": OKABE_VERMILLION,
        "s": 56.25,
        "linewidths": 2.5,
    },
    _PointStatus.RESPONSE: {
        **_DEFAULT_STYLE,
        "facecolors": OKABE_BLACK,
        "edgecolors": OKABE_BLACK,
        "linewidths": 1.0,
        "s": 16,
    },
    _PointStatus.APPROXIMATED: {
        **_DEFAULT_STYLE,
        "facecolors": OKABE_PURPLE,
        "edgecolors": OKABE_PURPLE,
        "linewidths": 1.0,
        "s": 16,
    },
    _PointStatus.FILE_RFT: {
        **_DEFAULT_STYLE,
        "facecolors": OKABE_YELLOW,
        "edgecolors": OKABE_BLACK,
        "linewidths": 0.5,
        "s": 9.0,
    },
}


def _transform_point_style_to_overlay_style(
    point_style: dict[str, str | float],
) -> dict[str, str | float]:
    # Use facecolor as overlay edgecolor, fallback to inherit edgecolor if facecolor is
    # transparent
    edgecolor = point_style.get("facecolors", "gray")
    if edgecolor == TRANSPARENT:
        edgecolor = point_style.get("edgecolors", "gray")
    return {
        **point_style,
        **_OVERLAY_STYLE,
        "edgecolors": edgecolor,
    }


_CELL_CENTER_OVERLAY_STYLE: dict[str, dict[str, str | float]] = {
    _PointStatus.MATCHED: _transform_point_style_to_overlay_style(
        _POINT_STYLE[_PointStatus.MATCHED]
    ),
    _PointStatus.INVALID_ZONE: _transform_point_style_to_overlay_style(
        _POINT_STYLE[_PointStatus.INVALID_ZONE]
    ),
    _PointStatus.NO_RESPONSE: _transform_point_style_to_overlay_style(
        _POINT_STYLE[_PointStatus.NO_RESPONSE]
    ),
}


def _point_style(
    statuses: Sequence[str],
    style_dict: Mapping[str, Mapping[str, str | float]] = _POINT_STYLE,
) -> dict[str, list[str | float]]:
    return {
        prop: [style_dict.get(_PointStatus(s), {}).get(prop, default) for s in statuses]
        for prop, default in _DEFAULT_STYLE.items()
    }


def _concat_point_styles(
    *styles: dict[str, list[str | float]],
) -> dict[str, list[str | float]]:
    return {
        prop: [value for style in styles for value in style.get(prop, [])]
        for prop in _DEFAULT_STYLE
    }


def _apply_overlap_overrides(
    style: dict[str, list[str | float]],
    statuses: Sequence[str],
    coords: Sequence[tuple[Any, ...]],
) -> dict[str, list[str | float]]:
    """Restyle points that overlap a lower layer so the layer beneath shows through.

    Observations coinciding with a response or file response are hollowed out, and
    responses coinciding with a file response get a file-colored border and smaller
    size, giving a stacked-on-top impression without relying on 3D z-order.
    """
    response_coords = {
        c for c, s in zip(coords, statuses, strict=True) if s in _RESPONSE_STATUSES
    }
    file_coords = {
        c for c, s in zip(coords, statuses, strict=True) if s == _PointStatus.FILE_RFT
    }
    for i, (c, s) in enumerate(zip(coords, statuses, strict=True)):
        if s in _OBSERVATION_STATUSES:
            if c in response_coords:
                style["facecolors"][i] = TRANSPARENT
            elif c in file_coords:
                style["facecolors"][i] = TRANSPARENT
                style["s"][i] = 45.5625
                style["linewidths"][i] = 3.25
        if s in _RESPONSE_STATUSES and c in file_coords:
            style["facecolors"][i] = TRANSPARENT
    return style


def _add_status_col_to_df(df: pl.DataFrame, status: str) -> pl.DataFrame:
    return df.with_columns(pl.lit(status).alias("status"))


def _ensure_well_connection_cell_center(df: pl.DataFrame) -> pl.DataFrame:
    if "well_connection_cell_center" in df.columns:
        return df
    if "cell_center" in df.columns:
        return df.rename({"cell_center": "well_connection_cell_center"})
    return df.with_columns(
        pl.lit(None).cast(pl.Array(pl.Float32, 3)).alias("well_connection_cell_center")
    )


def _deduplicate_points_per_coordinate_and_status(
    df: pl.DataFrame,
    coordinate_columns: Sequence[str] = ("east", "north", "tvd"),
) -> pl.DataFrame:
    """
    For each status, returns one point per coordinate from the given DataFrame

    Used for visualization, where multiple points may share the same coordinates.
    """

    STATUS_PRIORITY: dict[str, int] = {
        _PointStatus.MATCHED: 0,
        _PointStatus.INVALID_ZONE: 1,
        _PointStatus.NOT_IN_GRID: 2,
        _PointStatus.NO_RESPONSE: 3,
        _PointStatus.RESPONSE: 4,
        _PointStatus.APPROXIMATED: 5,
        _PointStatus.FILE_RFT: 6,
    }

    subset = set(coordinate_columns) | {"status"}
    return (
        df.with_columns(
            pl.col("status").replace_strict(STATUS_PRIORITY).alias("_priority")
        )
        .sort("_priority")
        .unique(subset=subset, keep="first", maintain_order=True)
        .drop("_priority")
    )


class _ToggleSwitch(QAbstractButton):
    """A checkable on/off switch with a sliding knob, exposing the standard
    ``toggled(bool)`` signal so it can stand in for a checkbox.
    """

    _TRACK_WIDTH = 36
    _TRACK_HEIGHT = 18
    _MARGIN = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._knob_position = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(120)
        self._animation.valueChanged.connect(self._on_animation_value_changed)
        self.toggled.connect(self._animate_knob)

    @override
    def sizeHint(self) -> QSize:
        return QSize(self._TRACK_WIDTH, self._TRACK_HEIGHT)

    def _animate_knob(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._knob_position)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def _on_animation_value_changed(self, value: Any) -> None:
        self._knob_position = float(value)
        self.update()

    @override
    def paintEvent(self, e: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        palette = self.palette()
        on_color = palette.color(QPalette.ColorRole.Highlight)
        off_color = palette.color(QPalette.ColorRole.Mid)
        track_color = on_color if self.isChecked() else off_color

        outline = palette.color(QPalette.ColorRole.WindowText)
        outline.setAlpha(120)

        radius = self._TRACK_HEIGHT / 2
        track_rect = QRectF(0, 0, self._TRACK_WIDTH, self._TRACK_HEIGHT).adjusted(
            0.5, 0.5, -0.5, -0.5
        )
        painter.setPen(QPen(outline, 1))
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_rect, radius, radius)

        knob_diameter = self._TRACK_HEIGHT - 2 * self._MARGIN
        travel = self._TRACK_WIDTH - knob_diameter - 2 * self._MARGIN
        knob_x = self._MARGIN + self._knob_position * travel
        painter.setPen(QPen(outline, 1))
        painter.setBrush(palette.color(QPalette.ColorRole.BrightText))
        painter.drawEllipse(QRectF(knob_x, self._MARGIN, knob_diameter, knob_diameter))


class _SelectionIndicatorDelegate(QStyledItemDelegate):
    """Paints a checkbox (multi-select) or radio button (single/extended) glyph next
    to each item, reflecting its selection state so the two filter kinds look distinct.
    """

    _INDICATOR_MARGIN = 4

    def __init__(self, *, exclusive: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._exclusive = exclusive

    def set_exclusive(self, exclusive: bool) -> None:
        self._exclusive = exclusive

    def _indicator_size(
        self, style: QStyle, option: QStyleOptionViewItem
    ) -> tuple[QStyle.PrimitiveElement, int, int]:
        if self._exclusive:
            return (
                QStyle.PrimitiveElement.PE_IndicatorRadioButton,
                style.pixelMetric(
                    QStyle.PixelMetric.PM_ExclusiveIndicatorWidth, option, option.widget
                ),
                style.pixelMetric(
                    QStyle.PixelMetric.PM_ExclusiveIndicatorHeight,
                    option,
                    option.widget,
                ),
            )
        return (
            QStyle.PrimitiveElement.PE_IndicatorCheckBox,
            style.pixelMetric(
                QStyle.PixelMetric.PM_IndicatorWidth, option, option.widget
            ),
            style.pixelMetric(
                QStyle.PixelMetric.PM_IndicatorHeight, option, option.widget
            ),
        )

    @override
    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        assert painter is not None
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        assert style is not None

        indicator, ind_w, ind_h = self._indicator_size(style, opt)

        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        text = opt.text
        opt.text = ""
        # Selection is conveyed by the indicator glyph, so suppress the
        # highlight fill to avoid redundant visual noise.
        opt.state &= ~QStyle.StateFlag.State_Selected
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget
        )

        rect = opt.rect
        ind_rect = QRect(
            rect.left() + self._INDICATOR_MARGIN,
            rect.top() + (rect.height() - ind_h) // 2,
            ind_w,
            ind_h,
        )
        button = QStyleOptionButton()
        button.rect = ind_rect
        button.state = QStyle.StateFlag.State_Enabled | (
            QStyle.StateFlag.State_On if selected else QStyle.StateFlag.State_Off
        )
        style.drawPrimitive(indicator, button, painter, opt.widget)

        text_left = ind_rect.right() + self._INDICATOR_MARGIN
        text_rect = QRect(
            text_left, rect.top(), rect.right() - text_left, rect.height()
        )
        foreground = index.data(Qt.ItemDataRole.ForegroundRole)
        if isinstance(foreground, QBrush):
            color = foreground.color()
        elif isinstance(foreground, QColor):
            color = foreground
        else:
            color = opt.palette.color(QPalette.ColorRole.Text)
        painter.save()
        painter.setPen(color)
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            text,
        )
        painter.restore()

    @override
    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        size = super().sizeHint(option, index)
        style = (
            option.widget.style() if option.widget is not None else QApplication.style()
        )
        assert style is not None
        _, ind_w, _ = self._indicator_size(style, option)
        size.setWidth(size.width() + ind_w + 2 * self._INDICATOR_MARGIN)
        return size


class PlotControlPanel(QWidget):
    def __init__(
        self,
        on_item_selection_change: Callable[..., None],
        on_reset_view_button_clicked: Callable[[], None],
        on_fit_to_selection_button_clicked: Callable[[], None],
        on_center_on_selected_button_clicked: Callable[[], None],
        on_toggle_utm_coords_clicked: Callable[[bool], None],
        use_utm: bool,
    ) -> None:
        QWidget.__init__(self)

        self.setMinimumWidth(_FILTER_WIDTH)
        filter_layout = QVBoxLayout()
        filter_layout.setContentsMargins(4, 4, 4, 4)
        filter_layout.setSpacing(2)

        self.dfs_to_filter: list[pl.DataFrame] = []

        self._well_list = self._make_filter_list(on_item_selection_change)
        self._date_list = self._make_filter_list(
            on_item_selection_change,
            selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
        )
        self._property_list = self._make_filter_list(
            on_item_selection_change,
            selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
        )
        self._status_list = self._make_filter_list(on_item_selection_change)

        self._filter_specs = {
            "well": self._well_list,
            "date": self._date_list,
            "property": self._property_list,
            "status": self._status_list,
        }

        for title, widget, multiselect_toggle in [
            ("Well", self._well_list, False),
            ("Date", self._date_list, True),
            ("Property", self._property_list, True),
            ("Status", self._status_list, False),
        ]:
            self._add_filter_section(filter_layout, title, widget, multiselect_toggle)

        filter_layout.addStretch()

        self._reset_view_button = QPushButton("Reset view")
        self._reset_view_button.clicked.connect(on_reset_view_button_clicked)
        filter_layout.addWidget(self._reset_view_button)

        self._fit_button = QPushButton("Fit to visible points")
        self._fit_button.clicked.connect(on_fit_to_selection_button_clicked)
        filter_layout.addWidget(self._fit_button)

        self._center_button = QPushButton("Center on selected")
        self._center_button.clicked.connect(on_center_on_selected_button_clicked)
        filter_layout.addWidget(self._center_button)

        self._toggle_utm_coords = QCheckBox("Show UTM coordinates")
        self._toggle_utm_coords.setChecked(use_utm)
        self._toggle_utm_coords.toggled.connect(on_toggle_utm_coords_clicked)
        filter_layout.addWidget(self._toggle_utm_coords)

        self.setLayout(filter_layout)

    def update_utm_available(self, available: bool) -> None:
        self._toggle_utm_coords.setEnabled(available)
        if not available and self._toggle_utm_coords.isChecked():
            self._toggle_utm_coords.blockSignals(True)
            self._toggle_utm_coords.setChecked(False)
            self._toggle_utm_coords.blockSignals(False)
        self._toggle_utm_coords.setToolTip(
            ""
            if available
            else "Some points are missing east/north/tvd coordinates; "
            "UTM view is unavailable"
        )

    def populate_filters(self, dataframes: list[pl.DataFrame]) -> None:
        self.dfs_to_filter = dataframes
        for col, list_widget in self._filter_specs.items():
            list_widget.blockSignals(True)
            list_widget.clear()
            values: set[str] = set()
            for df in self.dfs_to_filter:
                if col in df.columns:
                    values.update(
                        str(v) for v in df[col].drop_nulls().unique().to_list()
                    )
            for val in sorted(values, key=str):
                item = QListWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, val)
                list_widget.addItem(item)
            if (
                list_widget.selectionMode()
                == QAbstractItemView.SelectionMode.SingleSelection
                and list_widget.count() > 0
            ):
                list_widget.setCurrentRow(0)
            else:
                list_widget.clearSelection()
            list_widget.blockSignals(False)
        self.refresh_facet_decorations()

    def apply_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return self._apply_filters_except(df, except_col=None)

    def refresh_facet_decorations(self) -> None:
        for col, list_widget in self._filter_specs.items():
            counts = self._facet_counts(col, self.dfs_to_filter)
            for item in [list_widget.item(i) for i in range(list_widget.count())]:
                if item is None:
                    continue
                val = item.data(Qt.ItemDataRole.UserRole)
                n = counts.get(str(val), 0)
                item.setText(f"{val}  ({n})")
                if n == 0:
                    item.setForeground(QColor(GREY))
                else:
                    item.setData(Qt.ItemDataRole.ForegroundRole, None)

    def _make_filter_list(
        self,
        on_item_selection_change: Callable[..., None],
        selection_mode: QAbstractItemView.SelectionMode = (
            QAbstractItemView.SelectionMode.MultiSelection
        ),
    ) -> QListWidget:
        lw = QListWidget()
        lw.setSelectionMode(selection_mode)
        exclusive = selection_mode == QAbstractItemView.SelectionMode.SingleSelection
        lw.setItemDelegate(_SelectionIndicatorDelegate(exclusive=exclusive, parent=lw))
        lw.itemSelectionChanged.connect(on_item_selection_change)
        lw.itemSelectionChanged.connect(self.refresh_facet_decorations)
        return lw

    def _add_filter_section(
        self,
        layout: QVBoxLayout,
        title: str,
        widget: QListWidget,
        multiselect_toggle: bool,
    ) -> None:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"<b>{title}</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header.addWidget(lbl)
        header.addStretch()

        buttons = self._make_select_buttons(widget)
        if multiselect_toggle:
            toggle_label = QLabel("Enable multiselect")
            toggle_label.setStyleSheet("font-size: small;")
            toggle = _ToggleSwitch()
            toggle.setToolTip("Enable multiselect")
            toggle.toggled.connect(
                lambda enabled, w=widget: self._on_multiselect_toggled(w, enabled)
            )
            toggle.toggled.connect(buttons.setVisible)
            header.addWidget(toggle_label)
            header.addWidget(toggle)
            # Reserve the buttons' space while hidden so toggling doesn't shift layout.
            size_policy = buttons.sizePolicy()
            size_policy.setRetainSizeWhenHidden(True)
            buttons.setSizePolicy(size_policy)
            buttons.setVisible(False)

        layout.addLayout(header)
        layout.addWidget(widget)
        layout.addWidget(buttons)

    def _on_multiselect_toggled(self, list_widget: QListWidget, enabled: bool) -> None:
        delegate = list_widget.itemDelegate()
        assert isinstance(delegate, _SelectionIndicatorDelegate)
        delegate.set_exclusive(not enabled)
        if enabled:
            list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        else:
            selected = list_widget.selectedItems()
            list_widget.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection
            )
            # Collapse any multi-selection down to a single item.
            keep = selected[0] if selected else list_widget.item(0)
            list_widget.clearSelection()
            if keep is not None:
                keep.setSelected(True)
        viewport = list_widget.viewport()
        if viewport is not None:
            viewport.update()

    def _make_select_buttons(self, list_widget: QListWidget) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 2)
        row.setSpacing(2)
        all_btn = QPushButton("Select All")
        none_btn = QPushButton("Clear")
        for btn in (all_btn, none_btn):
            btn.setFlat(True)
            btn.setStyleSheet("padding: 1px 4px; font-size: small;")
        all_btn.clicked.connect(list_widget.selectAll)
        none_btn.clicked.connect(list_widget.clearSelection)
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        return container

    def _apply_filters_except(
        self, df: pl.DataFrame, except_col: str | None
    ) -> pl.DataFrame:
        for col, list_widget in self._filter_specs.items():
            if col == except_col or col not in df.columns:
                continue
            selected = {
                item.data(Qt.ItemDataRole.UserRole)
                for item in list_widget.selectedItems()
            }
            if selected:
                df = df.filter(pl.col(col).cast(pl.String).is_in(selected))
        return df

    def _facet_counts(self, col: str, dataframes: list[pl.DataFrame]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for df in dataframes:
            if df.is_empty() or col not in df.columns:
                continue
            filtered = self._apply_filters_except(df, except_col=col)
            for v, n in filtered.group_by(col).len().iter_rows():
                if v is None:
                    continue
                counts[str(v)] = counts.get(str(v), 0) + int(n)
        return counts


class DisplayPoints:
    def __init__(self) -> None:
        self._use_utm: bool = False
        self._df: pl.DataFrame = pl.DataFrame(schema=self.schema())

    @staticmethod
    def schema() -> dict[str, Any]:
        return {
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "status": pl.String,
        }

    def update_display_points(
        self, display_points: pl.DataFrame, use_utm: bool
    ) -> None:
        self._use_utm = use_utm
        self._df = display_points.select(self.schema().keys())

    def get_point(self, index: int | None) -> dict[str, Any] | None:
        if index is not None and 0 <= index < len(self._df):
            return self._df.row(index, named=True)
        return None

    def get_point_coordinates(
        self, index: int | None
    ) -> tuple[float, float, float] | None:
        point = self.get_point(index)
        if point is None:
            return None
        if self._use_utm:
            return (
                point["east"],
                point["north"],
                point["tvd"],
            )
        return tuple(point["well_connection_cell"])

    def get_points_to_plot(self) -> tuple[list[float], list[float], list[float]]:
        if self._use_utm:
            return (
                self._df["east"].to_list(),
                self._df["north"].to_list(),
                self._df["tvd"].to_list(),
            )
        return (
            self._df["well_connection_cell"].arr.get(0).to_list(),
            self._df["well_connection_cell"].arr.get(1).to_list(),
            self._df["well_connection_cell"].arr.get(2).to_list(),
        )


class RftPlot:
    def __init__(
        self,
        show_details: Callable[[dict[str, Any]], None],
        clear_details: Callable[[], None],
    ) -> None:
        figure = Figure()
        self._canvas: FigureCanvas = FigureCanvas(figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._ax: Axes3D = figure.add_subplot(projection="3d")
        self._ax.mouse_init(rotate_btn=1, pan_btn=3, zoom_btn=2)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._autoscaled_limits: tuple[Any, Any, Any] | None = None

        self._point_artist: PathCollection | None = None
        self._display_points: DisplayPoints = DisplayPoints()
        self._selection_artist: Any = None
        self._hover_artist: Any = None
        self._selected_index: int | None = None
        self._hover_index: int | None = None

        self._show_details = show_details
        self._clear_details = clear_details

    @property
    def canvas(self) -> FigureCanvas:
        return self._canvas

    @tracer.start_as_current_span(f"{__name__}.RftPlot.redraw")
    def redraw(
        self,
        obs_df: pl.DataFrame,
        response_df: pl.DataFrame,
        file_rft_df: pl.DataFrame,
        *,
        use_utm: bool = False,
        preserve_view: bool = True,
    ) -> None:
        current_span = trace.get_current_span()
        current_span.set_attribute("use_utm", use_utm)
        prior_limits: tuple[Any, Any, Any] | None = None
        if preserve_view and self._autoscaled_limits is not None:
            # Read the current view limits before clearing the axes, so we can restore
            # them after redrawing.
            prior_limits = (
                self._ax.get_xlim(),
                self._ax.get_ylim(),
                self._ax.get_zlim(),
            )
        previous_selected_point = self._display_points.get_point(self._selected_index)
        self._ax.cla()
        self._ax.set_zinverted(True)
        self._point_artist = None
        self._selection_artist = None
        self._hover_artist = None
        self._selected_index = None
        self._clear_details()
        self._hover_index = None

        if use_utm:
            self._ax.set_xlabel("east", labelpad=6)
            self._ax.set_ylabel("north", labelpad=6)
            self._ax.set_zlabel("tvd", labelpad=6)
        else:
            self._ax.set_xlabel("i", labelpad=6)
            self._ax.set_ylabel("j", labelpad=6)
            self._ax.set_zlabel("k", labelpad=6)

        if obs_df.is_empty() and response_df.is_empty() and file_rft_df.is_empty():
            self._draw_canvas(prior_limits)
            return

        point_columns = self._display_points.schema().keys()

        points = pl.concat(
            [
                obs_df.select(point_columns),
                response_df.select(point_columns),
                file_rft_df.select(point_columns),
            ]
        )

        def _get_observation_cell_center_overlay(
            obs_df: pl.DataFrame,
        ) -> tuple[
            pl.DataFrame,
            list[tuple[tuple[float, float, float], tuple[float, float, float]]],
            list[str],
        ]:
            """"""
            # Get observations where the location does not match the cell center.
            observations_not_at_cell_centers = obs_df.with_columns(
                pl.concat_arr("east", "north", "tvd")
                .cast(pl.Array(pl.Float32, 3))
                .alias("obs_location")
            ).filter(pl.col("well_connection_cell_center") != pl.col("obs_location"))

            # Prepare segments for showing relation between the observations and its
            # cell center (responses are always at cell centers)
            segments = [
                (tuple(p1), tuple(p2))
                for p1, p2 in zip(
                    observations_not_at_cell_centers[
                        "well_connection_cell_center"
                    ].to_list(),
                    observations_not_at_cell_centers["obs_location"].to_list(),
                    strict=True,
                )
            ]

            segment_colors = [
                str(_POINT_STYLE.get(_PointStatus(s), {}).get("edgecolors", "gray"))
                for s in observations_not_at_cell_centers["status"].to_list()
            ]

            observations_not_at_cell_centers = (
                observations_not_at_cell_centers.with_columns(
                    pl.col("well_connection_cell_center").arr.get(0).alias("east"),
                    pl.col("well_connection_cell_center").arr.get(1).alias("north"),
                    pl.col("well_connection_cell_center").arr.get(2).alias("tvd"),
                )
            )

            # Keep only one observation cell center overlay point per coordinate for
            # visualization
            observations_not_at_cell_centers = (
                _deduplicate_points_per_coordinate_and_status(
                    observations_not_at_cell_centers.select(point_columns)
                )
            )

            return observations_not_at_cell_centers, segments, segment_colors

        points_not_in_grid = False
        has_cell_center_overlays = False
        if use_utm:
            points = _deduplicate_points_per_coordinate_and_status(points)

            cell_center_overlays, segments, segment_colors = (
                _get_observation_cell_center_overlay(obs_df)
            )
            has_cell_center_overlays = not cell_center_overlays.is_empty()

            main_statuses = points["status"].to_list()
            main_coords = list(
                zip(
                    points["east"].to_list(),
                    points["north"].to_list(),
                    points["tvd"].to_list(),
                    strict=True,
                )
            )
            point_style = _concat_point_styles(
                _apply_overlap_overrides(
                    _point_style(main_statuses), main_statuses, main_coords
                ),
                _point_style(
                    cell_center_overlays["status"].to_list(),
                    _CELL_CENTER_OVERLAY_STYLE,
                ),
            )
            points = pl.concat(
                [
                    points,
                    cell_center_overlays,
                ]
            )

            if segments:
                lc = Line3DCollection(
                    segments,
                    colors=segment_colors,
                    linewidths=2,
                    linestyles="dashed",
                )
                self._ax.add_collection3d(lc)

        else:
            points_not_in_grid = points["well_connection_cell"].is_null().any()
            points = points.filter(pl.col("well_connection_cell").is_not_null())
            points = _deduplicate_points_per_coordinate_and_status(
                points, coordinate_columns=["well_connection_cell"]
            )

            statuses = points["status"].to_list()
            coords = [tuple(cell) for cell in points["well_connection_cell"].to_list()]
            point_style = _apply_overlap_overrides(
                _point_style(statuses), statuses, coords
            )

        self._display_points.update_display_points(points, use_utm)
        xs, ys, zs = self._display_points.get_points_to_plot()
        self._point_artist = self._ax.scatter(
            xs,
            ys,
            zs,
            **point_style,
            picker=5,
            depthshade=False,
        )

        displayed_statuses = points["status"].unique().to_list()
        for status, style in _POINT_STYLE.items():
            if status in displayed_statuses:
                self._ax.scatter(
                    [],
                    [],
                    [],
                    **style,
                    label=status,
                )

        if has_cell_center_overlays:
            self._ax.scatter(
                [],
                [],
                [],
                **_OVERLAY_STYLE,
                label="Observation's grid-cell center",
            )

        if points_not_in_grid:
            self._ax.scatter(
                [],
                [],
                [],
                **_DEFAULT_STYLE,
                label="Activate utm coordinates to see points not in the grid",
            )

        self._ax.legend(loc="upper left", fontsize="x-small")
        self._create_overlay_artists()

        if previous_selected_point is not None:
            # Try to restore the previously selected point

            fallback_index = None
            fallback_point = None
            p2 = previous_selected_point
            p2_utm = np.array([p2["east"], p2["north"], p2["tvd"]])
            for i, p1 in enumerate(points.rows(named=True)):
                if p1["well_connection_cell"] == p2["well_connection_cell"]:
                    if fallback_point is None:
                        fallback_index = i
                        fallback_point = p1
                    p1_utm = np.array([p1["east"], p1["north"], p1["tvd"]])
                    if (
                        None not in p1_utm
                        and None not in p2_utm
                        and np.linalg.norm(p1_utm - p2_utm) < 1e-4
                    ):
                        self._update_selected_point(i, p1)
                        break
            # If we didn't find an exact match, but did find a point with the same cell
            # center, use that as a fallback.
            if fallback_index is not None and fallback_point is not None:
                self._update_selected_point(fallback_index, fallback_point)

        self._draw_canvas(prior_limits)

    def _draw_canvas(self, prior_limits: tuple[Any, Any, Any] | None = None) -> None:
        self._canvas.draw()

        # Store the autoscale limits after the redraw, so we can restore them later if
        # requested by the user
        self._autoscaled_limits = (
            self._ax.get_xlim(),
            self._ax.get_ylim(),
            self._ax.get_zlim(),
        )
        # Apply the prior limits to not disturb the user's current zoom level.
        if prior_limits is not None:
            xlim, ylim, zlim = prior_limits
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.set_zlim(*zlim)
            self._canvas.draw_idle()

    def _reset_view(self) -> None:
        self._ax.view_init()
        self._fit_view_to_displayed_points()

    def _fit_view_to_displayed_points(self) -> None:
        if self._autoscaled_limits is None:
            return
        xlim, ylim, zlim = self._autoscaled_limits
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.set_zlim(*zlim)
        self._canvas.draw_idle()

    def _center_on_selected(self) -> None:
        if self._selected_index is None:
            return
        coordinates = self._display_points.get_point_coordinates(self._selected_index)
        if coordinates is None:
            return
        cx, cy, cz = coordinates
        for getter, setter, center in (
            (self._ax.get_xlim, self._ax.set_xlim, cx),
            (self._ax.get_ylim, self._ax.set_ylim, cy),
            (self._ax.get_zlim, self._ax.set_zlim, cz),
        ):
            lo, hi = getter()
            offset = 0.5 * (hi - lo)
            setter(center - offset, center + offset)

        self._canvas.draw_idle()

    def _create_overlay_artists(self) -> None:
        self._selection_artist = self._ax.scatter(
            [],
            [],
            [],
            s=350,
            facecolors=TRANSPARENT,
            edgecolors=SELECTION_RING_COLOR,
            linewidths=2.5,
            depthshade=False,
            zorder=20,
        )
        self._hover_artist = self._ax.scatter(
            [],
            [],
            [],
            s=300,
            facecolors=TRANSPARENT,
            edgecolors=HOVER_RING_COLOR,
            linewidths=1.2,
            alpha=0.6,
            depthshade=False,
            zorder=19,
        )

    def _refresh_overlays(self) -> None:
        if self._selection_artist is None or self._hover_artist is None:
            return

        def _coords_for(
            idx: int | None,
        ) -> tuple[list[float], list[float], list[float]]:
            coordinates = self._display_points.get_point_coordinates(idx)
            if coordinates is None:
                return ([], [], [])
            x, y, z = coordinates
            return ([x], [y], [z])

        hover_idx = (
            self._hover_index if self._hover_index != self._selected_index else None
        )

        # Matplotlib currently has no public setter for 3D scatter offsets,
        # so using the private attribute directly as a workaround.
        # See https://github.com/matplotlib/matplotlib/issues/784
        self._selection_artist._offsets3d = _coords_for(self._selected_index)
        self._hover_artist._offsets3d = _coords_for(hover_idx)
        self._canvas.draw_idle()

    def _on_pick(self, event: PickEvent) -> None:
        if (
            self._point_artist is None
            or event.artist is not self._point_artist
            or event.mouseevent.name != "button_press_event"
        ):
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return
        idx = int(event.ind[0])
        selected_point = self._display_points.get_point(idx)
        if selected_point is not None:
            self._update_selected_point(idx, selected_point)

    def _update_selected_point(
        self, index: int, selected_point: dict[str, Any]
    ) -> None:
        self._selected_index = index
        self._show_details(selected_point)
        self._refresh_overlays()

    def _on_hover(self, event: MouseEvent) -> None:
        if (
            self._point_artist is None
            or event.inaxes is not self._ax
            or event.x is None
            or event.y is None
        ):
            new_index: int | None = None
        else:
            new_index = self._closest_point_within(
                self._point_artist, event, radius_px=5
            )
        if new_index == self._hover_index:
            return
        self._hover_index = new_index
        self._refresh_overlays()

    @staticmethod
    def _closest_point_within(
        point_artist: PathCollection, event: MouseEvent, radius_px: float
    ) -> int | None:
        # Mirrors the picker tolerance used by _on_pick so hover and pick detection
        # agree
        offsets = cast(npt.NDArray[np.float64], point_artist.get_offsets())
        if len(offsets) == 0:
            return None
        display_xy = point_artist.get_offset_transform().transform(offsets)
        dx = display_xy[:, 0] - event.x
        dy = display_xy[:, 1] - event.y
        d2 = dx * dx + dy * dy
        idx = int(d2.argmin())
        return idx if d2[idx] <= radius_px * radius_px else None

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes is not self._ax:
            return
        scale = 0.8 if event.button == "up" else 1.25
        coordinates = self._display_points.get_point_coordinates(self._selected_index)
        (cx, cy, cz) = (None, None, None) if coordinates is None else coordinates

        # Assert to reassure mypy that self._ax is indeed an Axes3D instance:
        assert isinstance(self._ax, Axes3D)
        for getter, setter, center in (
            (self._ax.get_xlim, self._ax.set_xlim, cx),
            (self._ax.get_ylim, self._ax.set_ylim, cy),
            (self._ax.get_zlim, self._ax.set_zlim, cz),
        ):
            lo, hi = getter()
            c = 0.5 * (lo + hi) if center is None else center
            setter(c + (lo - c) * scale, c + (hi - c) * scale)
        self._canvas.draw_idle()


@dataclass
class CurrentRealization:
    ensemble: Ensemble
    number: int
    runpath: str | None
    rft_config: RFTConfig | None
    loaded: bool
    rft_file_path: Path | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.runpath is not None and self.rft_config is not None:
            self.rft_file_path = Path(
                self.rft_config._rft_filepath(
                    self.rft_config.expected_input_files[0],
                    self.runpath,
                    self.number,
                    self.ensemble.iteration,
                )
            )
        else:
            self.rft_file_path = None


class RftQcWidget(QWidget):
    def __init__(self, ert_config: ErtConfig | None = None) -> None:
        QWidget.__init__(self)
        self._runpaths: Runpaths | None = (
            Runpaths.from_config(ert_config) if ert_config else None
        )

        self._current_realization: CurrentRealization | None = None

        self._observations: pl.DataFrame = pl.DataFrame(
            schema=self._required_obs_subschema()
        )
        self._responses: pl.DataFrame = pl.DataFrame(
            schema=self._required_response_subschema()
        )
        self._file_responses: pl.DataFrame = pl.DataFrame(
            schema=self._required_file_response_subschema()
        )

        self._load_rft_file: bool = False

        self._use_utm = False
        self._plot = RftPlot(self._show_details, self._clear_details)
        self._filter_panel = PlotControlPanel(
            self._apply_filter_and_redraw,
            self._plot._reset_view,
            self._plot._fit_view_to_displayed_points,
            self._plot._center_on_selected,
            self._on_coord_toggle,
            self._use_utm,
        )

        # ── Top row: load status ─────────────────────────────────────
        self._load_status_label = QLabel("")
        self._load_status_label.setObjectName("RftLoadStatusLabel")
        self._load_status_label.setTextFormat(Qt.TextFormat.RichText)
        self._load_status_label.setWordWrap(True)
        self._load_status_label.setStyleSheet("color: #a33; font-size: small;")
        self._load_status_label.hide()

        # ── Bottom row: File RFT load controls ────────────────────────
        file_rft_panel = QWidget()
        file_rft_layout = QHBoxLayout()
        file_rft_layout.setContentsMargins(4, 4, 4, 4)
        self._load_rft_file_toggle = QCheckBox("Load RFT file content into plot")
        self._load_rft_file_toggle.setChecked(self._load_rft_file)
        self._load_rft_file_toggle.setEnabled(False)
        self._load_rft_file_toggle.toggled.connect(self._on_toggle_file_rft)
        file_rft_layout.addWidget(self._load_rft_file_toggle)
        self._rft_file_label = QLabel("<i>No RFT file found</i>")
        self._rft_file_label.setTextFormat(Qt.TextFormat.RichText)
        self._rft_file_label.setWordWrap(True)
        self._rft_file_label.setStyleSheet("color: #555; font-size: small;")
        file_rft_layout.addWidget(self._rft_file_label, stretch=1)
        file_rft_panel.setLayout(file_rft_layout)

        # ── Right: point details panel ───────────────────────────────
        details_panel = QWidget()
        details_panel.setMinimumWidth(_DETAILS_WIDTH)
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(4, 4, 4, 4)
        details_layout.addWidget(QLabel("<b>Point details</b>"))
        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setPlaceholderText("Click a point to inspect.")
        details_layout.addWidget(self._details)
        details_panel.setLayout(details_layout)

        # ── Main layout ──────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._filter_panel)
        splitter.addWidget(self._plot.canvas)
        splitter.addWidget(details_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([_FILTER_WIDTH, 900, _DETAILS_WIDTH])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setHandleWidth(7)
        splitter.setStyleSheet(
            f"""
            QSplitter::handle:horizontal {{
                background-color: {divider_color()};
                margin: 0 2px;
            }}
            """
        )

        columns_layout = QHBoxLayout()
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.addWidget(splitter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._load_status_label)
        layout.addLayout(columns_layout, stretch=1)
        layout.addWidget(file_rft_panel)
        self.setLayout(layout)

    def _update_load_rft_file_toggle_enabled_state(self) -> None:
        self._load_rft_file_toggle.setEnabled(
            self._load_rft_file
            or (
                self._current_realization is not None
                and self._current_realization.rft_file_path is not None
                and self._current_realization.rft_file_path.exists()
            )
        )

    def _set_current_realization(self, ensemble: Ensemble, realization: int) -> None:
        self._current_realization = CurrentRealization(
            ensemble=ensemble,
            number=realization,
            runpath=self._get_runpath(ensemble, realization),
            rft_config=self._create_rft_config(ensemble),
            loaded=False,
        )

        self._update_load_rft_file_toggle_enabled_state()
        rft_file_path = self._current_realization.rft_file_path
        self._rft_file_label.setText(
            str(rft_file_path)
            if rft_file_path is not None and rft_file_path.exists()
            else "<i>No RFT file found</i>"
        )

    @tracer.start_as_current_span(f"{__name__}.load_current_realization")
    def load_current_realization(self) -> None:
        if (
            self._current_realization is not None
            and not self._current_realization.loaded
        ):
            self._load_realization(
                self._current_realization.ensemble,
                self._current_realization.number,
                self._current_realization.rft_config,
            )
            self._current_realization.loaded = True

    @tracer.start_as_current_span(f"{__name__}.update_realization")
    def update_realization(
        self, ensemble: Ensemble, realization: int, *, load: bool = True
    ) -> None:
        self._set_current_realization(ensemble, realization)
        if load:
            self.load_current_realization()

    def _get_runpath(self, ensemble: Ensemble, realization: int) -> str | None:
        if self._runpaths:
            return self._runpaths.get_paths([realization], ensemble.iteration)[0]
        return None

    def _create_rft_config(self, ensemble: Ensemble) -> RFTConfig | None:
        rft_cfg = ensemble.experiment.response_configuration.get("rft")
        if isinstance(rft_cfg, RFTConfig):
            return RFTConfig(
                input_files=rft_cfg.input_files,
                data_to_read={"*": {"*": ["*"]}},  # Read all available RFT data
                zonemap=rft_cfg.zonemap,
                approximate_missing_values=rft_cfg.approximate_missing_values,
            )
        return None

    @staticmethod
    def _validate_required_columns(
        obs_df: pl.DataFrame,
        required_subschema: dict[str, Any],
        context: str = "Observations",
    ) -> None:
        missing_cols = required_subschema.keys() - set(obs_df.columns)
        if missing_cols:
            raise AssertionError(
                f"{context} DataFrame is missing expected columns: {missing_cols}"
            )
        required_df = obs_df.select(required_subschema.keys())
        if required_df.schema != required_subschema:
            msg = (
                f"Expected schema {required_subschema} for {context}, "
                f"got {required_df.schema}."
            )
            raise AssertionError(msg)

    @staticmethod
    def _required_obs_subschema() -> dict[str, Any]:
        return {
            "response_key": pl.String,
            "well": pl.String,
            "date": pl.String,
            "property": pl.String,
            "zone": pl.String,
            "observations": pl.Float32,
            "std": pl.Float32,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "well_connection_cell_center": pl.Array(pl.Float32, 3),
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "actual_zones": pl.List(pl.String),
            "qc_error": pl.String,
            "status": pl.String,
            "values": pl.Float32,
        }

    @staticmethod
    def _required_response_subschema() -> dict[str, Any]:
        return {
            "response_key": pl.String,
            **RftQcWidget._required_file_response_subschema(),
        }

    @staticmethod
    def _required_file_response_subschema() -> dict[str, Any]:
        return {
            "well": pl.String,
            "date": pl.String,
            "property": pl.String,
            "values": pl.Float32,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "well_connection_cell_center": pl.Array(pl.Float32, 3),
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "depth": pl.Float32,
            "cell_zones": pl.List(pl.String),
            "status": pl.String,
        }

    @staticmethod
    def _get_responses(
        ensemble: Ensemble,
        realization: int,
        observations: pl.DataFrame,
        approximate_missing_values: bool,
    ) -> pl.DataFrame:
        responses = ensemble.load_responses("rft", (realization,))
        if approximate_missing_values:
            responses_including_approximations = (
                RFTConfig.approximate_missing_rft_responses(
                    responses.lazy(), observations
                ).collect()
            )
            approximated = responses_including_approximations.join(
                responses, on=responses.columns, how="anti"
            )
            responses = pl.concat(
                [
                    _add_status_col_to_df(responses, _PointStatus.RESPONSE),
                    _add_status_col_to_df(approximated, _PointStatus.APPROXIMATED),
                ]
            )
        else:
            responses = _add_status_col_to_df(responses, _PointStatus.RESPONSE)
        responses = _ensure_well_connection_cell_center(responses).with_columns(
            pl.col("well_connection_cell_center").arr.get(0).alias("east"),
            pl.col("well_connection_cell_center").arr.get(1).alias("north"),
            pl.col("well_connection_cell_center").arr.get(2).alias("tvd"),
        )
        RftQcWidget._validate_required_columns(
            responses,
            RftQcWidget._required_response_subschema(),
            context="Responses",
        )
        return responses

    @staticmethod
    def _get_observations(ensemble: Ensemble, realization: int) -> pl.DataFrame:
        observations = ensemble.experiment.observations.get("rft")
        if observations is None or observations.is_empty():
            # No observation file in this experiment.
            return pl.DataFrame(schema=RftQcWidget._required_obs_subschema())
        observations = ensemble.add_rft_metadata_and_qc(observations, realization)
        observations = observations.with_columns(
            pl.col("response_key").str.split(":").list.last().alias("property")
        )
        return _ensure_well_connection_cell_center(observations)

    def _load_realization(
        self, ensemble: Ensemble, realization: int, rft_config: RFTConfig | None
    ) -> None:
        self._clear_load_status()
        errors: list[str] = []
        try:
            self._observations = self._get_observations(ensemble, realization)
        except Exception as err:
            self._observations = pl.DataFrame(schema=self._required_obs_subschema())
            errors.append(f"Could not load observations: {err}")
        try:
            approximate_missing_values = (
                rft_config.approximate_missing_values
                if rft_config is not None
                else False
            )
            self._responses = self._get_responses(
                ensemble,
                realization,
                observations=self._observations,
                approximate_missing_values=approximate_missing_values,
            )
        except Exception as err:
            self._responses = pl.DataFrame(schema=self._required_response_subschema())
            errors.append(f"Could not load responses: {err}")
        try:
            self._observations = RftQcWidget._attach_status(
                self._observations, self._responses
            )
            RftQcWidget._validate_required_columns(
                self._observations,
                RftQcWidget._required_obs_subschema(),
                context="Observations",
            )
        except Exception as err:
            self._observations = pl.DataFrame(schema=self._required_obs_subschema())
            errors.append(f"Could not load observations: {err}")
        if errors:
            self._show_load_status("<br>".join(errors))
        if self._load_rft_file:
            self._load_file_rft(self._current_realization)
        self._filter_panel.populate_filters(self._dfs_for_filters())
        self._refresh_utm_availability()
        self._apply_filter_and_redraw()

    def _show_load_status(self, message: str) -> None:
        self._load_status_label.setText(f"{message}")
        self._load_status_label.show()

    def _clear_load_status(self) -> None:
        self._load_status_label.clear()
        self._load_status_label.hide()

    def _apply_filter_and_redraw(self, *, preserve_view: bool = True) -> None:
        obs_df = self._filter_panel.apply_filter(self._observations)
        response_df = self._filter_panel.apply_filter(self._responses)
        file_rft_df = self._filter_panel.apply_filter(self._file_responses)
        self._plot.redraw(
            obs_df,
            response_df,
            file_rft_df,
            use_utm=self._use_utm,
            preserve_view=preserve_view,
        )

    @staticmethod
    def _attach_status(
        observations: pl.DataFrame, responses: pl.DataFrame
    ) -> pl.DataFrame:
        if "status" in observations.columns:
            # The status column is already present, so no need to recompute it.
            return observations
        joined = observations.join(
            responses.select("response_key", "well_connection_cell", "values"),
            on=["response_key", "well_connection_cell"],
            how="left",
        )
        return joined.with_columns(
            pl.when(pl.col("well_connection_cell").is_null())
            .then(pl.lit(_PointStatus.NOT_IN_GRID.value))
            .when(~RFTConfig.is_zone_valid())
            .then(pl.lit(_PointStatus.INVALID_ZONE.value))
            .when(pl.col("values").is_not_null())
            .then(pl.lit(_PointStatus.MATCHED.value))
            .otherwise(pl.lit(_PointStatus.NO_RESPONSE.value))
            .alias("status")
        )

    def _dfs_for_filters(self) -> list[pl.DataFrame]:
        return [self._observations, self._responses, self._file_responses]

    def _utm_coords_available(self) -> bool:
        coord_columns = ("east", "north", "tvd")
        for df in self._dfs_for_filters():
            if df.is_empty():
                continue
            if not all(c in df.columns for c in coord_columns):
                return False
            if any(df.select(coord_columns).null_count().row(0)):
                return False
        return True

    def _refresh_utm_availability(self) -> None:
        utm_available = self._utm_coords_available()
        if not utm_available:
            self._use_utm = False
        self._filter_panel.update_utm_available(utm_available)

    def _on_coord_toggle(self, checked: bool) -> None:
        self._use_utm = checked
        self._apply_filter_and_redraw(preserve_view=False)

    def _on_toggle_file_rft(self, checked: bool) -> None:
        self._load_rft_file = checked
        if self._load_rft_file:
            self._load_file_rft(self._current_realization)
        else:
            self._file_responses = pl.DataFrame(
                schema=self._required_file_response_subschema()
            )
        self._update_load_rft_file_toggle_enabled_state()
        self._filter_panel.populate_filters(self._dfs_for_filters())
        self._refresh_utm_availability()
        self._apply_filter_and_redraw()

    def _load_file_rft(self, current_realization: CurrentRealization | None) -> None:
        if (
            current_realization is not None
            and current_realization.rft_config is not None
            and current_realization.runpath is not None
        ):
            try:
                rft_file_df = _ensure_well_connection_cell_center(
                    current_realization.rft_config.read_from_file(
                        current_realization.runpath,
                        current_realization.number,
                        current_realization.ensemble.iteration,
                    )
                ).with_columns(
                    pl.col("well_connection_cell_center").arr.get(0).alias("east"),
                    pl.col("well_connection_cell_center").arr.get(1).alias("north"),
                    pl.col("well_connection_cell_center").arr.get(2).alias("tvd"),
                )
                self._file_responses = _add_status_col_to_df(
                    rft_file_df, _PointStatus.FILE_RFT
                )
                self._validate_required_columns(
                    self._file_responses,
                    self._required_file_response_subschema(),
                    context="File Responses",
                )
            except Exception as err:
                self._file_responses = pl.DataFrame(
                    schema=self._required_file_response_subschema()
                )
                self._rft_file_label.setText(str(err))

    def _show_details(self, point: dict[str, Any]) -> None:
        def _fmt(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.2f}"
            if isinstance(v, list):
                return ", ".join(str(x) for x in v)
            return html.escape(str(v))

        def _entry_header(row: dict[str, Any]) -> str:
            header = f"{row.get('well')} - {row.get('date')} - {row.get('property')}"
            return f"<tr><td colspan=2><b>{_fmt(header)}</b></td></tr>"

        def _labeled_item(val: Any, label: str) -> str:
            val = _fmt(val)
            if val:
                return (
                    f'<tr><td style="white-space:nowrap;">{label}:</td>'
                    f'<td style="white-space:nowrap;">{val}</td></tr>'
                )
            return ""

        def _status_note(val: Any) -> str:
            return (
                "<tr><td colspan=2><i style='color:gray;font-size:small;'>"
                + str(val)
                + "</i></td></tr>"
                if val
                else ""
            )

        well_connection_cell = point.get("well_connection_cell")
        well_connection_cell_equals = pl.col("well_connection_cell").eq_missing(
            well_connection_cell
        )

        point_observations = self._observations.filter(
            well_connection_cell_equals
            | (
                pl.col("east").eq_missing(point.get("east"))
                & pl.col("north").eq_missing(point.get("north"))
                & pl.col("tvd").eq_missing(point.get("tvd"))
            )
        )

        point_responses = self._responses.filter(well_connection_cell_equals)
        file_responses = self._file_responses.filter(well_connection_cell_equals)

        obs_zones = (
            point_observations["actual_zones"].explode(empty_as_null=False).to_list()
        )
        response_zones = (
            point_responses["cell_zones"].explode(empty_as_null=False).to_list()
        )
        zones: set[str] = set(obs_zones) | set(response_zones)

        def _utm_coords(row: dict[str, Any]) -> str | None:
            obs_location = (
                row.get("east"),
                row.get("north"),
                row.get("tvd"),
            )
            if None in obs_location:
                return None
            return ",".join([_fmt(x) for x in obs_location])

        ijk_coordinates = (
            (
                f"i={well_connection_cell[0]}, "
                f"j={well_connection_cell[1]}, "
                f"k={well_connection_cell[2]}"
            )
            if well_connection_cell is not None
            else "Not in grid"
        )

        utm_coordinates = ",".join(
            [_fmt(x) for x in (point.get("east"), point.get("north"), point.get("tvd"))]
        )

        observation_details_html = ""
        for row in point_observations.iter_rows(named=True):
            observation_details_html += f"""
                {_entry_header(row)}
                {_labeled_item(row.get("observations"), "Observation")}
                {_labeled_item(row.get("values"), "Response")}
                {_labeled_item(_utm_coords(row), "Utm Coordinates")}
                {_labeled_item(row.get("std"), "Error")}
                {_labeled_item(row.get("zone"), "Expected Zone")}
                {_labeled_item(row.get("status"), "Status")}
                {_status_note(row.get("qc_error"))}
            """

        response_details_html = _labeled_item(point_responses["depth"].first(), "Depth")
        for row in point_responses.iter_rows(named=True):
            response_details_html += f"""
                {_entry_header(row)}
                {_labeled_item(row.get("values"), "Response")}
                {_labeled_item(row.get("status"), "Status")}
            """

        file_point_details_html = _labeled_item(
            file_responses["depth"].first(), "Depth"
        )
        for row in file_responses.iter_rows(named=True):
            file_point_details_html += f"""
            {_entry_header(row)}
            {_labeled_item(row.get("values"), "File Response")}
            """

        zones_label = "Zones" if len(zones) > 1 else "Zone"
        utm_row = (
            _labeled_item(utm_coordinates, "Utm Coordinates") if self._use_utm else ""
        )

        self._details.setHtml(f"""
                <table style="border-spacing:2px 4px;">
                    {_labeled_item(ijk_coordinates, "Grid cell")}
                    {utm_row}
                    {_labeled_item(sorted(zones), zones_label)}
                </table>
                {"<h3>Observations:</h3>" if observation_details_html else ""}
                <table style="border-spacing:2px 4px;">
                    {observation_details_html}
                </table>
                {"<h3>Responses:</h3>" if response_details_html else ""}
                <table style="border-spacing:2px 4px;">
                    {response_details_html}
                </table>
                {"<h3>RFT in file:</h3>" if file_point_details_html else ""}
                <table style="border-spacing:2px 4px;">
                    {file_point_details_html}
                </table>
        """)

    def _clear_details(self) -> None:
        self._details.clear()
