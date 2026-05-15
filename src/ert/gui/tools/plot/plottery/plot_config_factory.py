from __future__ import annotations

from typing import TYPE_CHECKING

from .plot_config import PlotConfig

if TYPE_CHECKING:
    from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition


class PlotConfigFactory:
    @classmethod
    def create_plot_config_for_key(cls, key_def: PlotApiKeyDefinition) -> PlotConfig:
        plot_config = PlotConfig(plot_settings=None, title=key_def.key)

        # The styling of statistics changes based on the nature of the data
        if key_def.dimensionality == 2:
            mean_style = plot_config.get_statistics_style("mean")
            mean_style.line_style = "-"
            plot_config.set_statistics_style("mean", mean_style)

            p10p90_style = plot_config.get_statistics_style("p10-p90")
            p10p90_style.line_style = "--"
            plot_config.set_statistics_style("p10-p90", p10p90_style)
        elif key_def.dimensionality == 1:
            mean_style = plot_config.get_statistics_style("mean")
            mean_style.line_style = "-"
            mean_style.marker = "o"
            plot_config.set_statistics_style("mean", mean_style)

            std_style = plot_config.get_statistics_style("std")
            std_style.line_style = "--"
            std_style.marker = "D"
            plot_config.set_statistics_style("std", std_style)

        return plot_config
