from ert_gui.plottery import PlotConfig
from ert_shared import ERT


class PlotConfigFactory(object):

    @classmethod
    def createPlotConfigForKey(cls, key):
        """
        @type ert: res.enkf.enkf_main.EnKFMain
        @param key: str
        @return: PlotConfig
        """
        plot_config = PlotConfig(plot_settings=None , title = key)
        return PlotConfigFactory.updatePlotConfigForKey(key, plot_config)


    @classmethod
    def updatePlotConfigForKey(cls, key, plot_config):
        """
        @type ert: res.enkf.enkf_main.EnKFMain
        @param key: str
        @return: PlotConfig
        """
        # The styling of statistics changes based on the nature of the data
        if ERT.enkf_facade.is_summary_key(key) or ERT.enkf_facade.is_gen_data_key(key):
            mean_style = plot_config.getStatisticsStyle("mean")
            mean_style.line_style = "-"
            plot_config.setStatisticsStyle("mean", mean_style)

            p10p90_style = plot_config.getStatisticsStyle("p10-p90")
            p10p90_style.line_style = "--"
            plot_config.setStatisticsStyle("p10-p90", p10p90_style)
        else:
            mean_style = plot_config.getStatisticsStyle("mean")
            mean_style.line_style = "-"
            mean_style.marker = "o"
            plot_config.setStatisticsStyle("mean", mean_style)

            std_style = plot_config.getStatisticsStyle("std")
            std_style.line_style = "--"
            std_style.marker = "D"
            plot_config.setStatisticsStyle("std", std_style)

        return plot_config
