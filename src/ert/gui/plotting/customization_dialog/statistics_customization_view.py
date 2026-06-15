from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PyQt6.QtWidgets import QComboBox, QHBoxLayout
from typing_extensions import override

from .customization_view import CustomizationView, WidgetProperty
from .style_chooser import STYLESET_AREA, StyleChooser

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotConfig


class StatisticsCustomizationView(CustomizationView):
    mean_style = WidgetProperty()
    p50_style = WidgetProperty()
    std_style = WidgetProperty()
    min_max_style = WidgetProperty()
    p10_p90_style = WidgetProperty()
    p33_p67_style = WidgetProperty()
    std_dev_factor = WidgetProperty()
    distribution_lines = WidgetProperty()

    def __init__(self) -> None:
        CustomizationView.__init__(self)

        self._presets = [
            "Statistics default",
            "Cross ensemble statistics default",
            "Overview",
            "All statistics",
        ]

        self.add_row("Presets", self.create_presets())
        self.add_spacing(10)
        layout = QHBoxLayout()
        self.add_row("", layout)  # type: ignore
        self.add_style_chooser(
            "mean_style", "Mean", "Line and marker style for the mean line."
        )
        self.add_style_chooser(
            "p50_style", "P50", "Line and marker style for the P50 line."
        )
        self.add_style_chooser(
            "std_style",
            "Std dev",
            "Line and marker style for the unbiased standard deviation lines.",
            line_style_set=STYLESET_AREA,
        )
        self.add_style_chooser(
            "min_max_style",
            "Min/max",
            "Line and marker style for the min/max lines.",
            line_style_set=STYLESET_AREA,
        )
        self.add_style_chooser(
            "p10_p90_style",
            "P10-P90",
            "Line and marker style for the P10-P90 lines.",
            line_style_set=STYLESET_AREA,
        )
        self.add_style_chooser(
            "p33_p67_style",
            "P33-P67",
            "Line and marker style for the P33-P67 lines.",
            line_style_set=STYLESET_AREA,
        )
        self.add_spacing()

        self.add_integer_selection_box(
            "std_dev_factor",
            "Std dev multiplier",
            "Choose which standard deviation to plot",
            max_value=3,
        )

        self.add_check_box(
            "distribution_lines",
            "Connection lines",
            "Toggle distribution connection lines visibility.",
        )

        style = cast(StyleChooser, self["mean_style"])
        style.create_label_layout(layout)

    def create_presets(self) -> QComboBox:
        preset_combo = QComboBox()
        for preset in self._presets:
            preset_combo.addItem(preset)

        preset_combo.currentIndexChanged.connect(self.preset_selected)
        return preset_combo

    def preset_selected(self, index: int) -> None:
        if index == 0:  # Default
            self.update_style("mean_style", "-", None)
            self.update_style("p50_style", None, None)
            self.update_style("std_style", None, None)
            self.update_style("min_max_style", None, None)
            self.update_style("p10_p90_style", "--", None)
            self.update_style("p33_p67_style", None, None)
        elif index == 1:  # CCS Default
            self.update_style("mean_style", "-", "o")
            self.update_style("p50_style", None, None)
            self.update_style("std_style", "--", "D")
            self.update_style("min_max_style", None, None)
            self.update_style("p10_p90_style", None, None)
            self.update_style("p33_p67_style", None, None)
        elif index == 2:  # Overview
            self.update_style("mean_style", None, None)
            self.update_style("p50_style", None, None)
            self.update_style("std_style", None, None)
            self.update_style("min_max_style", "#", None)
            self.update_style("p10_p90_style", None, None)
            self.update_style("p33_p67_style", None, None)
        elif index == 3:  # All statistics
            self.update_style("mean_style", "-", None)
            self.update_style("p50_style", "--", "x")
            self.update_style("std_style", ":", None)
            self.update_style("min_max_style", "--", None)
            self.update_style("p10_p90_style", "#", None)
            self.update_style("p33_p67_style", "#", None)

    def update_style(
        self,
        attribute_name: str,
        line_style: str | None,
        marker_style: str | None,
    ) -> None:
        style = getattr(self, attribute_name)
        style.line_style = line_style
        style.marker = marker_style
        setattr(self, attribute_name, style)

    @override
    def apply_customization(self, plot_config: PlotConfig) -> None:
        plot_config.set_statistics_style("mean", self.mean_style)
        plot_config.set_statistics_style("p50", self.p50_style)
        plot_config.set_statistics_style("std", self.std_style)
        plot_config.set_statistics_style("min-max", self.min_max_style)
        plot_config.set_statistics_style("p10-p90", self.p10_p90_style)
        plot_config.set_statistics_style("p33-p67", self.p33_p67_style)

        plot_config.set_standard_deviation_factor(self.std_dev_factor)
        plot_config.set_distribution_line_enabled(self.distribution_lines)

    @override
    def revert_customization(self, plot_config: PlotConfig) -> None:
        self.mean_style = plot_config.get_statistics_style("mean")
        self.p50_style = plot_config.get_statistics_style("p50")
        self.std_style = plot_config.get_statistics_style("std")
        self.min_max_style = plot_config.get_statistics_style("min-max")
        self.p10_p90_style = plot_config.get_statistics_style("p10-p90")
        self.p33_p67_style = plot_config.get_statistics_style("p33-p67")

        self.std_dev_factor = plot_config.get_standard_deviation_factor()
        self.distribution_lines = plot_config.is_distribution_line_enabled()
