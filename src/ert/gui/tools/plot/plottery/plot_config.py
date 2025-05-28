from __future__ import annotations

import itertools
from copy import copy
from typing import Any

from .plot_limits import PlotLimits
from .plot_style import PlotStyle


class PlotConfig:
    # The plot_settings input argument is an internalisation of the (quite few) plot
    # policy settings which can be set in the configuration file.

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

        # Blueish, Greenlike, Beigeoid, Pinkness, Orangy-Brown with alpha=1.0
        self.setLineColorCycle(
            [
                ("#386CB0", 1.0),
                ("#7FC97F", 1.0),
                ("#FDC086", 1.0),
                ("#F0027F", 1.0),
                ("#BF5B17", 1.0),
            ]
        )

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

        self._observs_style = PlotStyle(
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
        self._distribution_line_style.setEnabled(False)
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

    def currentColor(self) -> tuple[str, float]:
        if self._current_color is None:
            return self.nextColor()

        return self._current_color

    def nextColor(self) -> tuple[str, float]:
        color = next(self._line_color_cycle)
        self._current_color = color
        return self._current_color

    def setLineColorCycle(self, color_list: list[tuple[str, float]]) -> None:
        self._line_color_cycle_colors = color_list
        self._line_color_cycle = itertools.cycle(color_list)

    def lineColorCycle(self) -> list[tuple[str, float]]:
        return self._line_color_cycle_colors

    def addLegendItem(self, label: str, item: Any) -> None:
        self._legend_items.append(item)
        self._legend_labels.append(label)

    def title(self) -> str:
        return self._title if self._title is not None else "Unnamed"

    def setTitle(self, title: str | None) -> None:
        self._title = title

    def isUnnamed(self) -> bool:
        return self._title is None

    def defaultStyle(self) -> PlotStyle:
        style = PlotStyle("Default style")
        style.copyStyleFrom(self._default_style)
        style.color, style.alpha = self.currentColor()
        return style

    def observationsColor(self) -> tuple[str, float]:
        assert self._observs_style.color
        return (self._observs_style.color, self._observs_style.alpha)

    def observationsStyle(self) -> PlotStyle:
        style = PlotStyle("Observations style")
        style.copyStyleFrom(self._observs_style)
        return style

    def historyStyle(self) -> PlotStyle:
        style = PlotStyle("History style")
        style.copyStyleFrom(self._history_style)
        return style

    def histogramStyle(self) -> PlotStyle:
        style = PlotStyle("Histogram style")
        style.copyStyleFrom(self._histogram_style)
        style.color, style.alpha = self.currentColor()
        return style

    def distributionStyle(self) -> PlotStyle:
        style = PlotStyle("Distribution style")
        style.copyStyleFrom(self._distribution_style)
        style.color, style.alpha = self.currentColor()
        return style

    def distributionLineStyle(self) -> PlotStyle:
        style = PlotStyle("Distribution line style")
        style.copyStyleFrom(self._distribution_line_style)
        return style

    def xLabel(self) -> str | None:
        return self._x_label

    def yLabel(self) -> str | None:
        return self._y_label

    def legendItems(self) -> list[Any]:
        return self._legend_items

    def legendLabels(self) -> list[str]:
        return self._legend_labels

    def setXLabel(self, label: str) -> None:
        self._x_label = label

    def setYLabel(self, label: str) -> None:
        self._y_label = label

    def setObservationsEnabled(self, enabled: bool) -> None:
        self._observs_style.setEnabled(enabled)

    def isObservationsEnabled(self) -> bool:
        return self._observs_style.isEnabled()

    def setHistoryEnabled(self, enabled: bool) -> None:
        self._history_style.setEnabled(enabled)

    def isHistoryEnabled(self) -> bool:
        return self._history_style.isEnabled()

    def isLegendEnabled(self) -> bool:
        return self._legend_enabled

    def isDistributionLineEnabled(self) -> bool:
        return self._distribution_line_style.isEnabled()

    def setDistributionLineEnabled(self, enabled: bool) -> None:
        self._distribution_line_style.setEnabled(enabled)

    def setStandardDeviationFactor(self, value: int) -> None:
        self._std_dev_factor = value

    def getStandardDeviationFactor(self) -> int:
        return self._std_dev_factor

    def setLegendEnabled(self, enabled: bool) -> None:
        self._legend_enabled = enabled

    def isGridEnabled(self) -> bool:
        return self._grid_enabled

    def setGridEnabled(self, enabled: bool) -> None:
        self._grid_enabled = enabled

    def setStatisticsStyle(self, statistic: str, style: PlotStyle) -> None:
        statistics_style = self._statistics_style[statistic]
        statistics_style.line_style = style.line_style
        statistics_style.marker = style.marker
        statistics_style.width = style.width
        statistics_style.size = style.size

    def getStatisticsStyle(self, statistic: str) -> PlotStyle:
        style = self._statistics_style[statistic]
        copy_style = PlotStyle(style.name)
        copy_style.copyStyleFrom(style)
        copy_style.color, copy_style.alpha = self.currentColor()
        return copy_style

    def setHistoryStyle(self, style: PlotStyle) -> None:
        self._history_style.line_style = style.line_style
        self._history_style.marker = style.marker
        self._history_style.width = style.width
        self._history_style.size = style.size

    def setObservationsColor(self, color_tuple: tuple[str, float]) -> None:
        self._observs_style.color, self._observs_style.alpha = color_tuple

    def setObservationsStyle(self, style: PlotStyle) -> None:
        self._observs_style.line_style = style.line_style
        self._observs_style.marker = style.marker
        self._observs_style.width = style.width
        self._observs_style.size = style.size

    def setDefaultStyle(self, style: PlotStyle) -> None:
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

    def copyConfigFrom(self, other: PlotConfig) -> None:
        self._default_style.copyStyleFrom(other._default_style, copy_enabled_state=True)
        self._history_style.copyStyleFrom(other._history_style, copy_enabled_state=True)
        self._histogram_style.copyStyleFrom(
            other._histogram_style, copy_enabled_state=True
        )
        self._observs_style.copyStyleFrom(other._observs_style, copy_enabled_state=True)
        self._distribution_style.copyStyleFrom(
            other._distribution_style, copy_enabled_state=True
        )
        self._distribution_line_style.copyStyleFrom(
            other._distribution_line_style, copy_enabled_state=True
        )

        self._statistics_style["mean"].copyStyleFrom(
            other._statistics_style["mean"], copy_enabled_state=True
        )
        self._statistics_style["p50"].copyStyleFrom(
            other._statistics_style["p50"], copy_enabled_state=True
        )
        self._statistics_style["min-max"].copyStyleFrom(
            other._statistics_style["min-max"], copy_enabled_state=True
        )
        self._statistics_style["p10-p90"].copyStyleFrom(
            other._statistics_style["p10-p90"], copy_enabled_state=True
        )
        self._statistics_style["p33-p67"].copyStyleFrom(
            other._statistics_style["p33-p67"], copy_enabled_state=True
        )
        self._statistics_style["std"].copyStyleFrom(
            other._statistics_style["std"], copy_enabled_state=True
        )

        self._std_dev_factor = other._std_dev_factor
        self._legend_enabled = other._legend_enabled
        self._grid_enabled = other._grid_enabled

        self.setLineColorCycle(other._line_color_cycle_colors)

        self._legend_items = other._legend_items[:]
        self._legend_labels = other._legend_labels[:]

        self._x_label = other._x_label
        self._y_label = other._y_label

        self._limits = copy(other._limits)

        if other._title is not None:
            self._title = other._title

    @classmethod
    def createCopy(cls, other: PlotConfig) -> PlotConfig:
        copy = PlotConfig(other._plot_settings)
        copy.copyConfigFrom(other)
        return copy
