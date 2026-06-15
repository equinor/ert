from __future__ import annotations

import itertools
from copy import copy
from typing import Any

from .plot_limits import PlotLimits
from .plot_style import PlotStyle


class PlotConfig:
    def __init__(
        self,
        plot_settings: dict[str, Any] | None = None,
        title: str | None = "Unnamed",
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> None:
        self._title = title
        self._plot_settings = plot_settings
        if self._plot_settings is None:
            self._plot_settings = {"SHOW_HISTORY": True}

        okabe_ito_hex = [
            "#E69F00",  # Orange
            "#56B4E9",  # Sky Blue
            "#009E73",  # Bluish Green
            "#F0E442",  # Yellow
            "#0072B2",  # Blue
            "#D55E00",  # Vermillion
            "#CC79A7",  # Reddish Purple
            "#000000",  # Black
        ]

        alpha_value = 1.0
        self.set_line_color_cycle([(colour, alpha_value) for colour in okabe_ito_hex])

        self._legend_items: list[Any] = []
        self._legend_labels: list[str] = []

        self._x_label = x_label
        self._y_label = y_label

        self._limits = PlotLimits()

        self._default_style = PlotStyle(
            name="Default", color=None, marker="", alpha=0.8
        )

        self._history_style = PlotStyle(
            name="History",
            alpha=0.8,
            marker=".",
            width=2.0,
            enabled=self._plot_settings["SHOW_HISTORY"],
        )

        self._observations_style = PlotStyle(
            name="Observations",
            line_style="-",
            alpha=0.8,
            marker=".",
            width=1.0,
            color="#000000",
        )

        self._histogram_style = PlotStyle(name="Histogram", width=2.0)
        self._distribution_style = PlotStyle(
            name="Distribution", line_style="", marker="o", alpha=0.5, size=10.0
        )
        self._distribution_line_style = PlotStyle(
            name="Distribution lines", line_style="-", alpha=0.25, width=1.0
        )
        self._distribution_line_style.set_enabled(False)
        self._current_color: tuple[str, float] | None = None

        self._legend_enabled = True
        self._grid_enabled = True

        self._statistics_style = {
            "mean": PlotStyle("Mean", line_style=""),
            "p50": PlotStyle("P50", line_style=""),
            "min-max": PlotStyle("Min/Max", line_style=""),
            "p10-p90": PlotStyle("P10-P90", line_style=""),
            "p33-p67": PlotStyle("P33-P67", line_style=""),
            "std": PlotStyle("Std dev", line_style=""),
        }

        self._std_dev_factor = 1  # sigma 1 is default std dev

        self.flip_response_axis = False
        self.flip_observation_axis = False

    def get_number_of_colors(self) -> int:
        return len(self._line_color_cycle_colors)

    def set_current_color(self, index: int) -> None:
        self._current_color = self._line_color_cycle_colors[
            index % len(self._line_color_cycle_colors)
        ]

    def current_color(self) -> tuple[str, float]:
        if self._current_color is None:
            return self.next_color()

        return self._current_color

    def next_color(self) -> tuple[str, float]:
        color = next(self._line_color_cycle)
        self._current_color = color
        return self._current_color

    def set_line_color_cycle(self, color_list: list[tuple[str, float]]) -> None:
        self._line_color_cycle_colors = color_list
        self._line_color_cycle = itertools.cycle(color_list)

    def line_color_cycle(self) -> list[tuple[str, float]]:
        return self._line_color_cycle_colors

    def add_legend_item(self, label: str, item: Any) -> None:
        self._legend_items.append(item)
        self._legend_labels.append(label)

    def title(self) -> str:
        return self._title if self._title is not None else "Unnamed"

    def set_title(self, title: str | None) -> None:
        self._title = title

    def is_unnamed(self) -> bool:
        return self._title is None

    def default_style(self) -> PlotStyle:
        style = PlotStyle("Default style")
        style.copy_style_from(self._default_style)
        style.color, style.alpha = self.current_color()
        return style

    def observations_color(self) -> tuple[str, float]:
        assert self._observations_style.color
        return (self._observations_style.color, self._observations_style.alpha)

    def observations_style(self) -> PlotStyle:
        style = PlotStyle("Observations style")
        style.copy_style_from(self._observations_style)
        return style

    def history_style(self) -> PlotStyle:
        style = PlotStyle("History style")
        style.copy_style_from(self._history_style)
        return style

    def histogram_style(self) -> PlotStyle:
        style = PlotStyle("Histogram style")
        style.copy_style_from(self._histogram_style)
        style.color, style.alpha = self.current_color()
        return style

    def distribution_style(self) -> PlotStyle:
        style = PlotStyle("Distribution style")
        style.copy_style_from(self._distribution_style)
        style.color, style.alpha = self.current_color()
        return style

    def distributionLineStyle(self) -> PlotStyle:
        style = PlotStyle("Distribution line style")
        style.copy_style_from(self._distribution_line_style)
        return style

    def x_label(self) -> str | None:
        return self._x_label

    def y_label(self) -> str | None:
        return self._y_label

    def legend_items(self) -> list[Any]:
        return self._legend_items

    def legend_labels(self) -> list[str]:
        return self._legend_labels

    def set_x_label(self, label: str) -> None:
        self._x_label = label

    def set_y_label(self, label: str) -> None:
        self._y_label = label

    def set_observations_enabled(self, enabled: bool) -> None:
        self._observations_style.set_enabled(enabled)

    def is_observations_enabled(self) -> bool:
        return self._observations_style.is_enabled()

    def set_history_enabled(self, enabled: bool) -> None:
        self._history_style.set_enabled(enabled)

    def is_history_enabled(self) -> bool:
        return self._history_style.is_enabled()

    def is_legend_enabled(self) -> bool:
        return self._legend_enabled

    def is_distribution_line_enabled(self) -> bool:
        return self._distribution_line_style.is_enabled()

    def set_distribution_line_enabled(self, enabled: bool) -> None:
        self._distribution_line_style.set_enabled(enabled)

    def set_standard_deviation_factor(self, value: int) -> None:
        self._std_dev_factor = value

    def get_standard_deviation_factor(self) -> int:
        return self._std_dev_factor

    def set_legend_enabled(self, enabled: bool) -> None:
        self._legend_enabled = enabled

    def is_grid_enabled(self) -> bool:
        return self._grid_enabled

    def set_grid_enabled(self, enabled: bool) -> None:
        self._grid_enabled = enabled

    def set_statistics_style(self, statistic: str, style: PlotStyle) -> None:
        statistics_style = self._statistics_style[statistic]
        statistics_style.line_style = style.line_style
        statistics_style.marker = style.marker
        statistics_style.width = style.width
        statistics_style.size = style.size

    def get_statistics_style(self, statistic: str) -> PlotStyle:
        style = self._statistics_style[statistic]
        copy_style = PlotStyle(style.name)
        copy_style.copy_style_from(style)
        copy_style.color, copy_style.alpha = self.current_color()
        return copy_style

    def set_history_style(self, style: PlotStyle) -> None:
        self._history_style.line_style = style.line_style
        self._history_style.marker = style.marker
        self._history_style.width = style.width
        self._history_style.size = style.size

    def set_observations_color(self, color_tuple: tuple[str, float]) -> None:
        self._observations_style.color, self._observations_style.alpha = color_tuple

    def set_observations_style(self, style: PlotStyle) -> None:
        self._observations_style.line_style = style.line_style
        self._observations_style.marker = style.marker
        self._observations_style.width = style.width
        self._observations_style.size = style.size

    def set_default_style(self, style: PlotStyle) -> None:
        self._default_style.line_style = style.line_style
        self._default_style.marker = style.marker
        self._default_style.width = style.width
        self._default_style.size = style.size

    @property
    def limits(self) -> PlotLimits:
        return copy(self._limits)

    @limits.setter
    def limits(self, value: PlotLimits) -> None:
        self._limits = copy(value)

    def copy_config_from(self, other: PlotConfig) -> None:
        self._default_style.copy_style_from(
            other._default_style, copy_enabled_state=True
        )
        self._history_style.copy_style_from(
            other._history_style, copy_enabled_state=True
        )
        self._histogram_style.copy_style_from(
            other._histogram_style, copy_enabled_state=True
        )
        self._observations_style.copy_style_from(
            other._observations_style, copy_enabled_state=True
        )
        self._distribution_style.copy_style_from(
            other._distribution_style, copy_enabled_state=True
        )
        self._distribution_line_style.copy_style_from(
            other._distribution_line_style, copy_enabled_state=True
        )

        self._statistics_style["mean"].copy_style_from(
            other._statistics_style["mean"], copy_enabled_state=True
        )
        self._statistics_style["p50"].copy_style_from(
            other._statistics_style["p50"], copy_enabled_state=True
        )
        self._statistics_style["min-max"].copy_style_from(
            other._statistics_style["min-max"], copy_enabled_state=True
        )
        self._statistics_style["p10-p90"].copy_style_from(
            other._statistics_style["p10-p90"], copy_enabled_state=True
        )
        self._statistics_style["p33-p67"].copy_style_from(
            other._statistics_style["p33-p67"], copy_enabled_state=True
        )
        self._statistics_style["std"].copy_style_from(
            other._statistics_style["std"], copy_enabled_state=True
        )

        self._std_dev_factor = other._std_dev_factor
        self._legend_enabled = other._legend_enabled
        self._grid_enabled = other._grid_enabled

        self.set_line_color_cycle(other._line_color_cycle_colors)

        self._legend_items = other._legend_items[:]
        self._legend_labels = other._legend_labels[:]

        self._x_label = other._x_label
        self._y_label = other._y_label

        self._limits = copy(other._limits)

        if other._title is not None:
            self._title = other._title

    @classmethod
    def create_copy(cls, other: PlotConfig) -> PlotConfig:
        copy = PlotConfig(other._plot_settings)
        copy.copy_config_from(other)
        return copy
