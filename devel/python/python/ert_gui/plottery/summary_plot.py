from ert.enkf.export import SummaryCollector
from ert_gui.plottery import PlotConfig, PlotContext, EnsemblePlot

class SummaryPlot(object):

    @staticmethod
    def summaryEnsemblePlot(plot_context):
        """
        @type plot_context: PlotContext
        """
        ert = plot_context.ert()
        key = plot_context.key()
        axes = plot_context.figure().add_subplot(111)
        config = plot_context.plotConfig()

        if config.xLabel() is None:
            config.setXLabel("Date")

        if config.yLabel() is None:
            config.setYLabel("Value")

            if ert.eclConfig().hasRefcase():
                unit = ert.eclConfig().getRefcase().unit(key)
                if unit != "":
                    config.setYLabel(unit)

        axes.set_title(config.title())

        for case in plot_context.cases():
            data = SummaryCollector.loadAllSummaryData(ert, case, [key])
            if not data.empty:
                data = data.reset_index()
                data = data.pivot(index="Date", columns="Realization", values=key)

                EnsemblePlot.plot(axes, config, data, case)
                config.nextColor()

        axes.legend(config.legendItems(), config.legendLabels())
        axes.grid()