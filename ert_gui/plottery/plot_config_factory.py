from ert_gui.plottery import PlotConfig


class PlotConfigFactory(object):
    @classmethod
    def createPlotConfigForKey(cls, key_def):
        """
        @param key_def: dict with definition of a key
        @return: PlotConfig
        """
        plot_config = PlotConfig(plot_settings=None, title=key_def["key"])

        # The styling of statistics changes based on the nature of the data
        if key_def["dimensionality"] == 2:
            mean_style = plot_config.getStatisticsStyle("mean")
            mean_style.line_style = "-"
            plot_config.setStatisticsStyle("mean", mean_style)

            p10p90_style = plot_config.getStatisticsStyle("p10-p90")
            p10p90_style.line_style = "--"
            plot_config.setStatisticsStyle("p10-p90", p10p90_style)
        elif key_def["dimensionality"] == 1:
            mean_style = plot_config.getStatisticsStyle("mean")
            mean_style.line_style = "-"
            mean_style.marker = "o"
            plot_config.setStatisticsStyle("mean", mean_style)

            std_style = plot_config.getStatisticsStyle("std")
            std_style.line_style = "--"
            std_style.marker = "D"
            plot_config.setStatisticsStyle("std", std_style)

        return plot_config
